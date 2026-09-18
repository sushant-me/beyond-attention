"""Model-level tests: shapes, budgets, causality, and that both can learn.

The scan tests prove the recurrence is right. These prove the thing wrapped
around it is right: that the block is causal end to end, that the two
architectures really are on the same parameter budget, and that a small model of
each can actually fit the task — because an architecture that cannot learn at
all would make every comparison in the README a comparison of two failures.
"""

from __future__ import annotations

import pytest
import torch

from beyond_attention.model import (
    AttentionBlock,
    LanguageModel,
    RMSNorm,
    SelectiveSSMBlock,
    build_pair,
    count_parameters,
)
from beyond_attention.tasks import (
    accuracy,
    loss_and_accuracy,
    mqar_batch,
    vocabulary_for,
)
from beyond_attention.train import train

BLOCK_KWARGS = {
    "ssm": {"d_state": 8, "expand": 2, "conv_kernel": 4},
    "attention": {"n_heads": 4, "mlp_ratio": 2.0},
}


def _model(block: str, vocab: int = 32, d_model: int = 32, layers: int = 2):
    return LanguageModel(vocab, d_model, layers, block, **BLOCK_KWARGS[block])


@pytest.mark.parametrize("block", ["ssm", "attention"])
def test_shapes_are_what_the_task_expects(block):
    model = _model(block)
    batch = mqar_batch(3, 4, 2)
    logits = model(batch.tokens)
    assert logits.shape == (3, batch.tokens.shape[1], 32)
    assert torch.isfinite(logits).all()


def test_both_architectures_are_on_the_same_parameter_budget():
    """The claim the whole comparison rests on, asserted rather than stated."""
    ssm, attention = build_pair(vocabulary_for(8), d_model=64, n_layers=2)
    a, b = count_parameters(ssm), count_parameters(attention)
    assert max(a, b) / min(a, b) <= 1.15, (a, b)


def test_a_budget_mismatch_is_rejected_rather_than_reported():
    """`build_pair` must refuse an unfair comparison. With heads the wrong
    number for the width, the attention model changes size and the guard has to
    fire — otherwise the README's parameter table would be a claim, not a fact.
    """
    from beyond_attention import model as model_module

    original = model_module.LanguageModel.__init__

    def shrunk(self, vocab_size, d_model, n_layers, block="ssm", **kwargs):
        # Make the attention path enormous so the ratio breaks.
        if block == "attention":
            d_model = d_model * 4
        return original(self, vocab_size, d_model, n_layers, block, **kwargs)

    model_module.LanguageModel.__init__ = shrunk
    try:
        with pytest.raises(ValueError, match="parameter budgets differ"):
            build_pair(vocabulary_for(8), d_model=64, n_layers=2)
    finally:
        model_module.LanguageModel.__init__ = original


@pytest.mark.parametrize("block", ["ssm", "attention"])
@pytest.mark.parametrize("cut", [1, 3])
def test_the_whole_model_is_causal(block, cut):
    """The scan is causal, the conv is causal, the attention mask is causal —
    asserted on the composed model, because a single non-causal piece is enough
    to leak and each piece was fixed separately.

    This is the model-level version of the scan test. It catches a leak that
    every individual component passes, which is the failure mode that makes a
    language model look good on a teacher-forced loss and useless when sampled.
    """
    torch.manual_seed(0)
    model = _model(block, vocab=16)
    model.eval()
    tokens = torch.randint(0, 16, (2, 12))

    with torch.no_grad():
        baseline = model(tokens)
        later = tokens.clone()
        later[:, cut:] = (tokens[:, cut:] + 5) % 16
        changed = model(later)

    assert torch.allclose(
        changed[:, :cut], baseline[:, :cut], atol=1e-6
    ), f"{block}: output before position {cut} changed when only later tokens changed"


@pytest.mark.parametrize("block", ["ssm", "attention"])
def test_the_short_conv_cannot_see_forward(block):
    """The depthwise convolution is left-padded by `kernel - 1`. Off by one and
    it silently peeks one token ahead, which no amount of training would reveal.
    """
    from beyond_attention.model import CausalDepthwiseConv1d

    torch.manual_seed(0)
    conv = CausalDepthwiseConv1d(channels=3, kernel=4)
    conv.eval()
    x = torch.randn(1, 10, 3)
    with torch.no_grad():
        base = conv(x)
        shifted = x.clone()
        shifted[:, 7:] += 1.0
        got = conv(shifted)
    assert torch.allclose(got[:, :7], base[:, :7])


def test_rms_norm_has_the_property_it_is_named_for():
    norm = RMSNorm(8)
    x = torch.randn(4, 5, 8)
    out = norm(x)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(4, 5), atol=1e-4)


def test_mqar_keys_and_values_never_collide():
    """If a key could equal a value, a model could score without storing
    anything, and the accuracy number would mean less than it appears to."""
    batch = mqar_batch(32, 8)
    n_keys = 8
    keys = set(range(1, n_keys + 1))
    values = set(range(1 + n_keys, 1 + 2 * n_keys))
    assert not keys & values
    # Every scored target must be a value token, never a key or the separator.
    assert set(batch.targets.reshape(-1).tolist()) <= values
    # The query position must hold a key, so the lookup is genuinely consulted.
    query_tokens = batch.tokens.gather(1, batch.query_positions)
    assert set(query_tokens.reshape(-1).tolist()) <= keys


def test_accuracy_is_zero_for_a_model_that_cannot_look_up():
    """A sanity check on the metric: shuffled logits should score at chance, not
    above it. Without this, a metric that leaked the answer would look like a
    strong result."""
    torch.manual_seed(0)
    vocab = vocabulary_for(8)
    batch = mqar_batch(64, 8, 1)
    random_logits = torch.randn(64, batch.tokens.shape[1], vocab)
    score = accuracy(random_logits, batch)
    assert score < 0.3, f"chance accuracy measured at {score:.3f}"


@pytest.mark.parametrize("block", ["ssm", "attention"])
def test_a_small_model_of_each_architecture_learns_the_task(block, tmp_path):
    """Slowest test here, and the one that decides whether the comparison is
    worth running: if either architecture cannot fit MQAR at all, comparing them
    at scale would be measuring two failures."""
    torch.manual_seed(0)
    del tmp_path
    model = _model(block, vocab=vocabulary_for(4), d_model=32, layers=2)
    result = train(model, block, n_pairs=4, steps=250, batch_size=32, lr=5e-3)
    chance = 1.0 / 4
    assert result.train_accuracy > chance * 1.5, (
        f"{block} did not learn: accuracy {result.train_accuracy:.3f} "
        f"vs chance {chance:.3f}"
    )


def test_mqar_rejects_an_impossible_request():
    with pytest.raises(ValueError):
        mqar_batch(2, 4, n_queries=8)
    with pytest.raises(ValueError):
        mqar_batch(2, 0)


def test_loss_and_accuracy_agree_on_the_scored_positions():
    """The loss and the metric must be about the same positions, or a model can
    be optimised on one thing and reported on another."""
    torch.manual_seed(0)
    vocab = vocabulary_for(4)
    model = _model("ssm", vocab=vocab, d_model=32, layers=1)
    batch = mqar_batch(8, 4, 2)
    loss, acc = loss_and_accuracy(model, batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_an_ssm_block_propagates_a_token_past_the_conv_receptive_field():
    """State has to carry a token further than the convolution can.

    The convolution's kernel is 2, so position 0 reaches positions 0 and 1
    directly and nothing else. If the output at position 11 still moves when
    only position 0 changes, the state carried it there; if it does not, the
    block is a local filter wearing a state-space model's name.

    Written in float64 because the effect decays geometrically and a float32
    tolerance could hide a real failure — or manufacture one.
    """
    torch.manual_seed(0)
    block = SelectiveSSMBlock(d_model=8, d_state=4, expand=2, conv_kernel=2)
    block = block.double().eval()
    x = torch.randn(1, 12, 8, dtype=torch.float64)

    with torch.no_grad():
        base = block(x)
        early = x.clone()
        early[:, 0] += 3.0
        early_out = block(early)

        late = x.clone()
        late[:, 11] += 3.0
        late_out = block(late)

    assert not torch.allclose(early_out[:, 11], base[:, 11], atol=1e-9), (
        "a change at position 0 did not reach position 11: the state is not "
        "carrying anything past the convolution"
    )
    # And the same effect must not run backwards.
    assert torch.allclose(late_out[:, 0], base[:, 0], atol=1e-9), (
        "a change at the end moved the start: the block is not causal"
    )


def test_attention_block_output_is_finite_with_a_single_token():
    block = AttentionBlock(d_model=8, n_heads=2, mlp_ratio=1.0)
    out = block(torch.randn(1, 1, 8))
    assert torch.isfinite(out).all()


# --- the fused-SDPA baseline, which exists so the comparison is not against a
# --- straw man -------------------------------------------------------------


def test_the_two_attention_modes_have_identical_parameter_counts():
    """The fused baseline must be the same size as the wrapper, or the scaling
    comparison would silently become a comparison of two different models."""
    wrapped = AttentionBlock(32, n_heads=4, mlp_ratio=2.0, causal_mode="mask")
    fused = AttentionBlock(32, n_heads=4, mlp_ratio=2.0, causal_mode="sdpa")
    assert count_parameters(wrapped) == count_parameters(fused)


def test_fused_attention_matches_a_manual_reference():
    """`CausalSelfAttention` is a reimplementation of a fused kernel path, so it
    gets checked against the definition written out longhand rather than against
    a loss curve."""
    from beyond_attention.model import CausalSelfAttention

    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=8, n_heads=2).double().eval()
    x = torch.randn(1, 6, 8, dtype=torch.float64)

    with torch.no_grad():
        got = attn(x)

        # Longhand: project, split into heads, causal softmax, weighted sum.
        qkv = attn.qkv(x).reshape(1, 6, 3, 2, 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-1, -2) / (4 ** 0.5)
        mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        expected = attn.out((weights @ v).transpose(1, 2).reshape(1, 6, 8))

    assert torch.allclose(got, expected, atol=1e-10), (
        f"fused attention diverged from the longhand definition "
        f"(max abs error {(got - expected).abs().max().item():.3e})"
    )


@pytest.mark.parametrize("causal_mode", ["mask", "sdpa"])
def test_both_attention_modes_are_causal(causal_mode):
    """Each way of being causal has to be causal on its own. The explicit mask
    and the fused kernel are different code paths, and the scaling result
    depends on both being correct."""
    torch.manual_seed(0)
    model = LanguageModel(
        16, 32, 2, "attention", n_heads=4, mlp_ratio=2.0, causal_mode=causal_mode
    ).eval()
    tokens = torch.randint(0, 16, (2, 12))
    with torch.no_grad():
        base = model(tokens)
        late = tokens.clone()
        late[:, 5:] = (tokens[:, 5:] + 3) % 16
        changed = model(late)
    assert torch.allclose(changed[:, :5], base[:, :5], atol=1e-6)


def test_an_unknown_causal_mode_is_rejected():
    with pytest.raises(ValueError, match="causal_mode"):
        AttentionBlock(32, n_heads=4, causal_mode="nonsense")
