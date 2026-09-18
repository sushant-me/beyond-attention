"""The task used to compare the two architectures.

**Multi-query associative recall (MQAR).** A sequence of key/value pairs is
followed by a separator and one or more query keys; the model must emit the
value that was paired with each query key. Every query is answered from context,
so the model has to store associations as it reads and retrieve them on demand.

Why this task, and not language modelling:

* It is **exact**. Accuracy is a number out of 1, not a loss to be interpreted,
  and a model either retrieves the value or it does not.
* It is **position-invariant**. The pairing is what matters, not where the pair
  sat, so a Transformer with no positional encoding is a fair baseline rather
  than a crippled one. Giving one architecture positional information the other
  does not have would decide the comparison before it started.
* It is the **known hard case**. A linear recurrent model compresses the past
  into a fixed-size state, so recalling one of many associations requires
  superposing them; attention can address any position directly. The published
  result is that state-space models degrade as the number of pairs grows. That
  is a real, falsifiable prediction, and this harness is built to test it rather
  than to produce a desired answer.

Keys and values are drawn from disjoint halves of the vocabulary so that a model
cannot score by predicting a likely token fill; it has to have stored the pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SEPARATOR = 0  # reserved token between the pairs and the queries


@dataclass(frozen=True)
class Batch:
    tokens: Tensor  # (B, L) int64
    targets: Tensor  # (B, Q) int64 — the value each query key is paired with
    query_positions: Tensor  # (B, Q) int64 — where each value must be emitted


def vocabulary_size(n_keys: int) -> int:
    """Keys, values, and one separator. Keys and values never overlap."""
    return 1 + 2 * n_keys


def mqar_batch(
    batch_size: int,
    n_pairs: int,
    n_queries: int = 1,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    n_keys: int | None = None,
) -> Batch:
    """Sample one batch of MQAR sequences.

    Layout: ``k1 v1 k2 v2 ... kN vN SEP q1 q2 ... qQ``, with the queries drawn
    (with replacement) from the keys that appeared. Length is ``2N + 1 + Q``.

    ``n_keys`` fixes the size of the key space, and therefore the vocabulary,
    independently of how many pairs a given sequence uses. It defaults to
    ``max(n_pairs, 8)``, which is right when you only ever sample at one size.
    It matters when you do not: a model trained at N pairs has an embedding
    sized for N pairs' vocabulary, so evaluating it at 4N pairs through the
    default would generate token ids its embedding cannot index -- a crash
    rather than a measurement. Passing a fixed ``n_keys`` holds the token space
    still and lets the *length* be the only thing that changes, which is the
    whole point of a length-extrapolation test.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    if n_queries < 1:
        raise ValueError("n_queries must be >= 1")
    if n_queries > n_pairs:
        # Queries are drawn from the keys present, so asking for more distinct
        # lookups than there are pairs is a malformed request, not a hard one.
        raise ValueError("n_queries cannot exceed n_pairs")

    if n_keys is None:
        n_keys = max(n_pairs, 8)
    elif n_keys < n_pairs:
        # Keys are drawn without replacement, so a sequence cannot have more
        # distinct keys than the space contains.
        raise ValueError("n_keys must be >= n_pairs")
    vocab = vocabulary_size(n_keys)

    # Distinct keys per sequence, so "the pair" is unambiguous.
    keys = torch.argsort(
        torch.rand(batch_size, n_keys, generator=generator, device=device), dim=1
    )[:, :n_pairs] + 1  # keys occupy [1, n_keys]
    values = (
        torch.randint(
            0, n_keys, (batch_size, n_pairs), generator=generator, device=device
        )
        + 1
        + n_keys
    )  # values occupy [1 + n_keys, 2 * n_keys]

    # Which pairs get asked about. Without replacement, so a batch is not scored
    # twice on one lookup.
    query_slots = torch.argsort(
        torch.rand(batch_size, n_pairs, generator=generator, device=device), dim=1
    )[:, :n_queries]

    pairs = torch.stack([keys, values], dim=2).reshape(batch_size, 2 * n_pairs)
    separator = torch.full(
        (batch_size, 1), SEPARATOR, dtype=torch.long, device=device
    )
    query_keys = torch.gather(keys, 1, query_slots)
    tokens = torch.cat([pairs, separator, query_keys], dim=1)

    targets = torch.gather(values, 1, query_slots)
    query_positions = (
        torch.arange(n_queries, device=device).unsqueeze(0).repeat(batch_size, 1)
        + 2 * n_pairs
        + 1
    )
    return Batch(tokens=tokens, targets=targets, query_positions=query_positions)


def vocabulary_for(n_pairs: int, n_queries: int = 1) -> int:
    """Vocabulary a model needs to handle this task configuration."""
    del n_queries
    return vocabulary_size(max(n_pairs, 8))


def accuracy(logits: Tensor, batch: Batch) -> float:
    """Fraction of queries whose argmax is the paired value."""
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    picked = logits[rows, batch.query_positions]  # (B, Q, vocab)
    predicted = picked.argmax(-1)
    return (predicted == batch.targets).float().mean().item()


def loss_and_accuracy(model, batch: Batch) -> tuple[Tensor, float]:
    """Cross-entropy over the query positions only, plus exact-match accuracy."""
    logits = model(batch.tokens)
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    picked = logits[rows, batch.query_positions]  # (B, Q, vocab)
    loss = torch.nn.functional.cross_entropy(
        picked.reshape(-1, picked.shape[-1]), batch.targets.reshape(-1)
    )
    return loss, accuracy(logits, batch)
