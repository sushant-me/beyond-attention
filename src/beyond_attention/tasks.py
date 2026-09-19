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


# ------------------------------------------------------- state tracking ------
#
# MQAR and the register task are deliberately opposite tests of the same thing.
# MQAR asks the model to hold *every* pair it has seen, so the memory it needs
# grows with the sequence: a fixed-size recurrent state is a handicap there, and
# that handicap is the finding the main sweep reports. This task asks the model
# to hold **one bit**, no matter how long the sequence is. Nothing about the
# answer depends on the history except a single parity, so a fixed state is
# sufficient by construction -- and an attention model still has to pool over
# the whole prefix to compute it.
#
# Running the two together is the point: one shows where constant state costs
# you, the other shows where it does not.

REG_NOOP = 0      # leaves the register alone
REG_TOGGLE = 1    # flips the register
REG_QUERY = 2     # the model must emit the register's current value here
REG_ANS_ZERO = 3  # emitted at a query when the register is 0
REG_ANS_ONE = 4   # emitted at a query when the register is 1
REGISTER_VOCAB = 5


def register_batch(
    batch_size: int,
    length: int,
    n_queries: int = 1,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Batch:
    """Sample one batch of register-tracking sequences.

    Each position is a no-op or a toggle with equal probability, except for
    ``n_queries`` positions per row which are replaced by a query. At a query the
    model must emit the parity of the toggles seen up to and including that
    position -- equivalently, the value of a single bit that every toggle flips
    and no no-op changes.

    The register starts at 0, and queries are placed at distinct positions drawn
    uniformly from 1..length-1, so a query always has at least one preceding
    token and there is no positional shortcut to the answer.
    """
    if length < 2:
        raise ValueError("length must be >= 2")
    if n_queries < 1:
        raise ValueError("n_queries must be >= 1")
    if n_queries > length - 1:
        # Positions are distinct and drawn from 1..length-1.
        raise ValueError("n_queries cannot exceed length - 1")

    toggles = torch.rand(batch_size, length, generator=generator, device=device)
    tokens = torch.where(
        toggles < 0.5,
        torch.full((batch_size, length), REG_TOGGLE, dtype=torch.long,
                   device=device),
        torch.full((batch_size, length), REG_NOOP, dtype=torch.long,
                   device=device),
    )

    # Distinct query positions per row, in 1..length-1, then sorted so the
    # targets read left to right.
    order = torch.argsort(
        torch.rand(batch_size, length - 1, generator=generator, device=device),
        dim=1,
    )[:, :n_queries] + 1
    positions, _ = order.sort(dim=1)

    rows = torch.arange(batch_size, device=device).unsqueeze(1)
    tokens[rows, positions] = REG_QUERY

    # Parity of the toggles in tokens[0..t]. A query is not a toggle, so reading
    # this at a query position gives the value the model must emit.
    parity = (tokens == REG_TOGGLE).cumsum(dim=1).remainder(2)
    targets = torch.where(
        parity[rows, positions] == 1,
        torch.full((batch_size, n_queries), REG_ANS_ONE, dtype=torch.long,
                   device=device),
        torch.full((batch_size, n_queries), REG_ANS_ZERO, dtype=torch.long,
                   device=device),
    )
    return Batch(tokens=tokens, targets=targets, query_positions=positions)


def register_chance() -> float:
    """The bar to beat, *conditional on the model emitting an answer token*.

    Only two tokens are ever correct at a query position, so once a model has
    learned to answer at all it is choosing between two, and 1/2 is what it has
    to beat. Reporting 1/vocab would understate that.

    It is not a floor, and accuracy can legitimately fall below it: scoring is an
    argmax over the whole vocabulary, so a model that has not yet learned to emit
    an answer token at a query position scores near zero rather than near a half.
    A below-0.5 result means "has not learned the response format yet", not
    "worse than guessing", and the two are worth keeping apart.
    """
    return 0.5
