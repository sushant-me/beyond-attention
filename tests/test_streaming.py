"""Streaming inference: equivalence, and the one property that matters.

Two things are checked here, and they are different kinds of claim:

* **Correctness** -- stepping one token at a time reproduces the parallel
  forward pass. Without this the rest is worthless, because a streaming path
  that is fast and wrong is worse than no streaming path.
* **The property** -- the carried state does not grow with the number of tokens
  consumed. For the SSM this must hold exactly; for attention it must *fail*,
  because that contrast is the finding.
"""

from __future__ import annotations

import pytest
import torch

from beyond_attention.model import build_pair, count_parameters
from beyond_attention.streaming import (
    init_stream,
    kv_cache_bytes,
    ssm_state_bytes,
    stream_sequence,
    stream_step,
)

TORCH_SEED = 0
D_MODEL = 32
N_LAYERS = 2
VOCAB = 24


def _models():
    torch.manual_seed(TORCH_SEED)
    return build_pair(
        VOCAB,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        block_kwargs={"d_state": 8, "expand": 2, "conv_kernel": 4},
    )


def _tokens(batch: int, length: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(TORCH_SEED + 1)
    return torch.randint(0, VOCAB, (batch, length), generator=gen)


# ------------------------------------------------------------- correctness ---

def test_ssm_streaming_matches_parallel():
    """Token-by-token must equal the whole-sequence forward, to tolerance.

    A mismatch here is the convolution tail being off by one, or the recurrence
    carrying the wrong `h`, or the readout using `C` at the wrong position --
    all of which produce output that looks reasonable and is wrong.
    """
    ssm, _ = _models()
    tokens = _tokens(2, 64)
    ssm.eval()
    with torch.no_grad():
        parallel = ssm(tokens)
    streamed = stream_sequence(ssm, tokens)
    assert parallel.shape == streamed.shape
    max_diff = (parallel - streamed).abs().max().item()
    assert max_diff < 1e-4, f"streaming diverged from parallel by {max_diff}"


def test_attention_streaming_matches_parallel():
    """The same check for the KV-cache path, so the comparison is honest.

    If the attention baseline streamed incorrectly it would look artificially
    cheap to run, and the SSM's advantage would be manufactured.
    """
    _, attention = _models()
    tokens = _tokens(2, 64)
    attention.eval()
    with torch.no_grad():
        parallel = attention(tokens)
    streamed = stream_sequence(attention, tokens)
    assert parallel.shape == streamed.shape
    max_diff = (parallel - streamed).abs().max().item()
    assert max_diff < 1e-4, f"attention streaming diverged by {max_diff}"


def test_streaming_matches_after_each_step_not_just_at_the_end():
    """A divergence can cancel. Check every position, not only the last."""
    ssm, _ = _models()
    tokens = _tokens(1, 32)
    ssm.eval()
    with torch.no_grad():
        parallel = ssm(tokens)
    state = init_stream(ssm, 1)
    for t in range(tokens.shape[1]):
        logits, state = stream_step(ssm, tokens[:, t], state)
        diff = (logits - parallel[:, t]).abs().max().item()
        assert diff < 1e-4, f"position {t} diverged by {diff}"


# ------------------------------------------------------------ the property ---

def test_ssm_state_is_constant_in_length():
    """The claim: state size is fixed by the model, not by the context.

    This is the difference that matters for long context. If this ever grows,
    the module's docstring is a lie and the measurement is meaningless.
    """
    ssm, _ = _models()
    sizes = []
    for length in (8, 64, 256, 1024):
        state = init_stream(ssm, 1)
        tokens = _tokens(1, length)
        for t in range(length):
            _, state = stream_step(ssm, tokens[:, t], state)
        sizes.append(state.numel())
    assert len(set(sizes)) == 1, f"SSM state grew with length: {sizes}"


def test_attention_state_grows_linearly():
    """And the contrast: attention's cache is the thing with a length in it."""
    _, attention = _models()
    sizes = []
    for length in (8, 64, 256):
        state = init_stream(attention, 1)
        tokens = _tokens(1, length)
        for t in range(length):
            _, state = stream_step(attention, tokens[:, t], state)
        sizes.append(state.numel())
    # Linear means size/length is constant. Comparing successive *deltas* would
    # be wrong here: the lengths differ between samples (8->64->256), so the
    # deltas differ for a perfectly linear cache.
    per_token = [s / n for s, n in zip(sizes, (8, 64, 256))]
    assert len(set(per_token)) == 1, f"KV cache is not linear in length: {sizes}"
    assert sizes[-1] > sizes[0]


def test_ssm_state_has_no_length_dependence_in_bytes():
    ssm, _ = _models()
    fixed = ssm_state_bytes(ssm, batch_size=1)
    assert fixed > 0
    # the helper takes no length argument at all -- that is the point
    assert ssm_state_bytes(ssm, batch_size=1) == fixed


def test_kv_cache_bytes_scales_with_length():
    _, attention = _models()
    a = kv_cache_bytes(attention, 1024)
    b = kv_cache_bytes(attention, 2048)
    assert b == 2 * a, "KV cache should be exactly linear in length"
    ssm, _ = _models()
    assert kv_cache_bytes(ssm, 100000) == 0, "a pure SSM has no KV cache"


# --------------------------------------------------------------- guardrails --

def test_causality_is_structural():
    """Feeding a later token must not change an earlier prediction.

    `stream_step` only ever sees one position, so this cannot fail by
    construction -- which is exactly why it is worth pinning: it documents that
    causality here is not an argument about masks.
    """
    ssm, _ = _models()
    ssm.eval()
    first = _tokens(1, 16)
    second = first.clone()
    second[:, 8:] = 0  # change only the future

    state_a = init_stream(ssm, 1)
    state_b = init_stream(ssm, 1)
    for t in range(8):
        la, state_a = stream_step(ssm, first[:, t], state_a)
        lb, state_b = stream_step(ssm, second[:, t], state_b)
        assert torch.allclose(la, lb, atol=1e-6), f"future leaked into step {t}"


def test_unsupported_block_type_is_rejected_loudly():
    """A silent fallback would report a number for something never measured."""
    from beyond_attention.streaming import StreamState, stream_step

    class NotAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4, 4)
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.norm_f = torch.nn.Identity()
            self.head = torch.nn.Linear(4, 4)

    with pytest.raises(TypeError):
        stream_step(NotAModel(), torch.tensor([1]), StreamState(blocks=[None]))
