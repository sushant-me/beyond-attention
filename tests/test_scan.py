"""The scan is the whole model, so it is checked against itself three ways.

A state-space model whose scan is subtly wrong still trains and still produces a
descending loss curve, because the gradient will happily push a slightly
incorrect recurrence into a configuration that fits the training data. The
failure shows up as a model that quietly cannot remember anything past its
chunk boundary, which is exactly the property the architecture exists for.

So the assertions here are about *equality between independent implementations*
and about *causality*, never about loss.
"""

from __future__ import annotations

import pytest
import torch

from beyond_attention.ssm import (
    selective_scan_associative,
    selective_scan_chunked,
    selective_scan_reference,
)

IMPLEMENTATIONS = {
    "reference": selective_scan_reference,
    "associative": selective_scan_associative,
}

#: Every path, named, for the tests that must exercise all three. `chunked` is
#: listed explicitly because it is *not* in `IMPLEMENTATIONS` - it takes a chunk
#: size - and for a while it was therefore missing from the causality test
#: altogether, so a reversed chunk scan (a genuine future leak across chunk
#: boundaries) passed it. A parametrize list is a claim about coverage, and this
#: one was false.
ALL_PATHS = ["reference", "associative", "chunked"]


def _run(name, chunk, x, delta, A, B, C):
    if name == "reference":
        return selective_scan_reference(x, delta, A, B, C)
    if name == "associative":
        return selective_scan_associative(x, delta, A, B, C)
    return selective_scan_chunked(x, delta, A, B, C, chunk=chunk)


def _random_inputs(batch=2, length=37, dim=5, state=4, seed=0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, dim, generator=generator, dtype=dtype)
    # delta is softplus-positive in the model; keep it positive here too.
    delta = torch.rand(batch, length, dim, generator=generator, dtype=dtype) + 1e-3
    # A must be negative for the recurrence to be a decay.
    A = -torch.rand(dim, state, generator=generator, dtype=dtype) - 0.05
    B = torch.randn(batch, length, state, generator=generator, dtype=dtype)
    C = torch.randn(batch, length, state, generator=generator, dtype=dtype)
    return x, delta, A, B, C


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_implementation_agrees_with_the_definition(name):
    """Equality between independent implementations, not agreement with a loss
    curve."""
    x, delta, A, B, C = _random_inputs()
    expected = selective_scan_reference(x, delta, A, B, C)
    got = IMPLEMENTATIONS[name](x, delta, A, B, C)
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-10), (
        f"{name} diverged from the definition (max abs error "
        f"{(got - expected).abs().max().item():.3e})"
    )


@pytest.mark.parametrize("chunk", [1, 2, 5, 37, 100])
def test_chunked_matches_the_definition_for_any_chunk_size(chunk):
    """Including chunk sizes that do not divide the length, which is where a
    padding bug in the chunked path would hide."""
    x, delta, A, B, C = _random_inputs(length=37)
    expected = selective_scan_reference(x, delta, A, B, C)
    got = selective_scan_chunked(x, delta, A, B, C, chunk=chunk)
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_the_shortest_sequences_are_handled(length):
    x, delta, A, B, C = _random_inputs(length=length)
    expected = selective_scan_reference(x, delta, A, B, C)
    assert torch.allclose(
        selective_scan_chunked(x, delta, A, B, C, chunk=64), expected, atol=1e-12
    )
    assert torch.allclose(
        selective_scan_associative(x, delta, A, B, C), expected, atol=1e-12
    )


def test_a_zero_input_produces_a_zero_state_in_every_path():
    """Degenerate but legal: no input means no accumulated state, and a chunked
    scan that leaked its summary into the first chunk would fail here."""
    x, delta, A, B, C = _random_inputs()
    x = torch.zeros_like(x)
    expected = selective_scan_reference(x, delta, A, B, C)
    assert torch.count_nonzero(expected) == 0
    assert torch.count_nonzero(selective_scan_chunked(x, delta, A, B, C)) == 0
    assert torch.count_nonzero(selective_scan_associative(x, delta, A, B, C)) == 0


def test_a_tiny_delta_degenerates_to_an_unweighted_accumulator():
    """As delta -> 0, a -> 1 and the recurrence becomes a plain running sum.

    This is a limit case with a closed form, so it can be checked without
    reference to any of the three implementations.
    """
    batch, length, dim, state = 1, 9, 3, 2
    dtype = torch.float64
    x = torch.randn(batch, length, dim, dtype=dtype)
    delta = torch.full((batch, length, dim), 1e-12, dtype=dtype)
    A = -torch.ones(dim, state, dtype=dtype)
    B = torch.ones(batch, length, state, dtype=dtype)
    C = torch.ones(batch, length, state, dtype=dtype)

    got = selective_scan_chunked(x, delta, A, B, C, chunk=4)
    # y_t[d] = state * delta * sum_{s<=t} x_s[d], to first order in delta.
    running = torch.cumsum(x, dim=1)
    expected = running * state * delta
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("name", ALL_PATHS)
@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 7, 8, 64])
def test_no_implementation_can_see_the_future(name, chunk):
    """Causality, asserted the only way that catches a real leak: run the model,
    then change an input at position ``t`` and require every output before ``t``
    to be bit-for-bit what it was.

    A scan that is only tested for numerical agreement can still be a correct
    *non-causal* function if all three implementations share the same masking
    mistake - so this test is deliberately independent of the other two.

    The chunk sizes are the point. A chunked scan leaks across chunk boundaries,
    not within them, so a causality test at a chunk size that swallows the whole
    sequence cannot fail: every position is in one chunk and the carry is never
    exercised. Chunks strictly smaller than the cut are what make the assertion
    mean something.
    """
    x, delta, A, B, C = _random_inputs(length=16)

    def run(x_, delta_, B_):
        return _run(name, chunk, x_, delta_, A, B_, C)

    baseline = run(x, delta, B)

    cut = 7
    # Perturb the input at `cut` and after, and nothing before it.
    far = x.clone()
    far[:, cut:] = x[:, cut:] + 5.0
    far_out = run(far, delta, B)
    assert torch.equal(far_out[:, :cut], baseline[:, :cut]), (
        f"{name} (chunk={chunk}) output before the cut changed when only later "
        f"inputs changed"
    )

    # Perturb the decay (delta) after the cut too: `delta` feeds `a` and `b`, so
    # a leak through either is caught.
    later_delta = delta.clone()
    later_delta[:, cut:] = delta[:, cut:] * 3.0 + 0.5
    delta_out = run(x, later_delta, B)
    assert torch.equal(delta_out[:, :cut], baseline[:, :cut])

    # And through B, the state-input projection.
    later_b = B.clone()
    later_b[:, cut:] = B[:, cut:] + 2.0
    b_out = run(x, delta, later_b)
    assert torch.equal(b_out[:, :cut], baseline[:, :cut])


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_gradients_match_between_the_chunked_path_and_the_definition(chunk):
    """A scan can be correct in forward and wrong in backward, and the backward
    error is invisible until training stalls for no visible reason."""
    tensors = _random_inputs(length=20)
    names = ["x", "delta", "A", "B", "C"]

    def grads(fn, chunk_size=None):
        leaves = [t.clone().requires_grad_(True) for t in tensors]
        out = fn(*leaves) if chunk_size is None else fn(*leaves, chunk=chunk_size)
        out.sum().backward()
        return [leaf.grad for leaf in leaves]

    reference = grads(selective_scan_reference)
    chunked = grads(selective_scan_chunked, chunk_size=chunk)
    for name, expected, got in zip(names, reference, chunked):
        assert torch.allclose(got, expected, atol=1e-10, rtol=1e-8), (
            f"gradient w.r.t. {name} differs between the two paths "
            f"(max abs error {(got - expected).abs().max().item():.3e})"
        )


def test_the_state_stays_bounded_over_a_long_sequence():
    """a is in (0, 1) and b is bounded, so the state cannot blow up. If a were
    ever >= 1 the recurrence would be unstable, and this is the assertion that
    would fail."""
    x, delta, A, B, C = _random_inputs(length=4000)
    out = selective_scan_chunked(x, delta, A, B, C, chunk=64)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e4
