"""MQAR batch generation, and the fixed key space that length extrapolation needs.

The interesting claim here is not that a batch is shaped correctly. It is that
the key space can be **held still while the sequence grows**, which is the only
way to ask a model something at a length it never trained at. By default the key
space is a function of the sequence length, so a model trained at N pairs has an
embedding that cannot index a batch drawn at 8N -- that is a crash, not a
measurement, and these tests pin the fix in place.
"""

from __future__ import annotations

import pytest
import torch

from beyond_attention.model import LanguageModel
from beyond_attention.tasks import (
    REG_ANS_ONE,
    REG_ANS_ZERO,
    REG_QUERY,
    REG_TOGGLE,
    REGISTER_VOCAB,
    SEPARATOR,
    mqar_batch,
    register_batch,
    register_chance,
    vocabulary_size,
)
from beyond_attention.train import evaluate, evaluate_mqar, train

N_KEYS = 32
VOCAB = vocabulary_size(N_KEYS)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_default_key_space_follows_length() -> None:
    """Unchanged behaviour: n_keys defaults to max(n_pairs, 8)."""
    for n_pairs in (2, 8, 16):
        batch = mqar_batch(2, n_pairs, 1, _gen())
        ceiling = vocabulary_size(max(n_pairs, 8))
        assert batch.tokens.max().item() < ceiling


def test_length_is_2n_plus_separator_plus_queries() -> None:
    for n_pairs, n_queries in ((1, 1), (8, 1), (16, 4), (32, 8)):
        batch = mqar_batch(3, n_pairs, n_queries, _gen(), n_keys=N_KEYS)
        assert batch.tokens.shape == (3, 2 * n_pairs + 1 + n_queries)


def test_fixed_key_space_holds_while_length_grows() -> None:
    """The property the extrapolation experiment depends on."""
    for n_pairs in (8, 16, 32):
        batch = mqar_batch(4, n_pairs, 1, _gen(), n_keys=32)
        assert batch.tokens.max().item() < VOCAB
        assert batch.targets.max().item() < VOCAB
        assert batch.tokens.min().item() >= 0


def test_n_keys_below_n_pairs_is_rejected() -> None:
    """Keys are drawn without replacement, so this cannot be satisfied."""
    with pytest.raises(ValueError, match="n_keys must be >= n_pairs"):
        mqar_batch(1, 32, 1, _gen(), n_keys=16)


def test_separator_sits_between_pairs_and_queries() -> None:
    batch = mqar_batch(4, 8, 3, _gen(), n_keys=N_KEYS)
    assert (batch.tokens[:, 2 * 8] == SEPARATOR).all()


def test_query_positions_land_on_the_query_keys() -> None:
    """The scored positions must be the query slots, not the pairs."""
    batch = mqar_batch(4, 8, 2, _gen(), n_keys=N_KEYS)
    rows = torch.arange(4).unsqueeze(1)
    at_query = batch.tokens[rows, batch.query_positions]  # (B, Q)
    assert (at_query == batch.targets).sum().item() == 0  # keys != values
    # Every query key appeared among the pairs, so it is a real key token.
    assert at_query.min().item() >= 1
    assert at_query.max().item() <= N_KEYS


def test_keys_within_a_sequence_are_distinct() -> None:
    """Otherwise "the pair for this key" would be ambiguous."""
    batch = mqar_batch(8, 16, 1, _gen(), n_keys=N_KEYS)
    keys = batch.tokens[:, 0 : 2 * 16 : 2]  # k1, k2, ... k16
    for row in keys:
        assert len(set(row.tolist())) == 16


def test_values_never_overlap_keys() -> None:
    """A model could otherwise pass by echoing the key."""
    batch = mqar_batch(8, 16, 1, _gen(), n_keys=N_KEYS)
    keys = set(batch.tokens[:, 0 : 2 * 16 : 2].flatten().tolist())
    values = set(batch.tokens[:, 1 : 2 * 16 : 2].flatten().tolist())
    assert not (keys & values)


def test_a_model_trained_at_one_length_can_be_evaluated_beyond_it() -> None:
    """End to end: the whole point of the parameter.

    Without a fixed key space this raises an index error inside the embedding
    rather than returning a number, which is why it is worth a test.
    """
    model = LanguageModel(VOCAB, 32, 2, "ssm", d_state=8, expand=2, conv_kernel=4)
    result = train(
        model, "ssm", n_pairs=8, steps=4, batch_size=4, seed=0, n_keys=N_KEYS
    )
    assert result.parameters > 0
    # 8 -> 32 pairs is 18 -> 66 tokens: well past what it trained on.
    score = evaluate_mqar(model, 32, batch_size=32, n_keys=N_KEYS)
    assert 0.0 <= score <= 1.0


def test_attention_model_can_also_be_evaluated_beyond_its_training_length() -> None:
    model = LanguageModel(VOCAB, 32, 2, "attention", n_heads=4, mlp_ratio=2.0)
    train(model, "attention", n_pairs=8, steps=4, batch_size=4, seed=0,
          n_keys=N_KEYS)
    assert 0.0 <= evaluate_mqar(model, 32, batch_size=32, n_keys=N_KEYS) <= 1.0


# ------------------------------------------------------- state tracking ------

def test_register_shapes_and_query_slots() -> None:
    batch = register_batch(4, 32, n_queries=3, generator=_gen())
    assert batch.tokens.shape == (4, 32)
    assert batch.targets.shape == (4, 3)
    assert batch.query_positions.shape == (4, 3)
    rows = torch.arange(4).unsqueeze(1)
    assert (batch.tokens[rows, batch.query_positions] == REG_QUERY).all()


def test_register_targets_encode_the_parity() -> None:
    """Re-derive the answer from the token stream, independently of the task."""
    batch = register_batch(6, 40, n_queries=4, generator=_gen(3))
    for row in range(6):
        parity = 0
        for t in range(40):
            token = int(batch.tokens[row, t])
            if token == REG_TOGGLE:
                parity ^= 1
            elif token == REG_QUERY:
                for q in range(4):
                    if int(batch.query_positions[row, q]) == t:
                        expected = REG_ANS_ONE if parity else REG_ANS_ZERO
                        assert int(batch.targets[row, q]) == expected


def test_register_query_positions_are_distinct_and_not_first() -> None:
    batch = register_batch(8, 24, n_queries=5, generator=_gen(1))
    for row in batch.query_positions:
        positions = row.tolist()
        assert len(set(positions)) == 5
        # Position 0 would have no preceding context.
        assert min(positions) >= 1
        assert max(positions) <= 23


def test_register_only_emits_answer_tokens_as_targets() -> None:
    batch = register_batch(4, 32, n_queries=2, generator=_gen(2))
    assert set(batch.targets.flatten().tolist()) <= {REG_ANS_ZERO, REG_ANS_ONE}


def test_register_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError, match="length must be >= 2"):
        register_batch(1, 1, 1, _gen())
    with pytest.raises(ValueError, match="n_queries cannot exceed length - 1"):
        register_batch(1, 8, 8, _gen())


def test_register_chance_is_one_half_not_one_over_vocab() -> None:
    """Only two tokens are ever correct, and the vocabulary is larger."""
    assert register_chance() == 0.5
    assert 1.0 / REGISTER_VOCAB < register_chance()


def test_a_second_task_trains_through_the_same_loop() -> None:
    """The sampler parameter is what keeps one training path, not two."""
    model = LanguageModel(REGISTER_VOCAB, 32, 2, "ssm",
                          d_state=8, expand=2, conv_kernel=4)
    sample = lambda bs, gen, dev: register_batch(bs, 16, 2, gen, dev)
    result = train(model, "ssm", n_pairs=1, steps=4, batch_size=4, seed=0,
                   sample_batch=sample)
    assert result.parameters > 0
    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 <= evaluate(model, sample, batch_size=32, seed=0) <= 1.0
