"""The selective scan: the linear recurrence a state-space model is built on.

Everything here implements the same equation, and implements it three different
ways:

    a_t = exp(delta_t * A)                        # (B, L, D, N), in (0, 1)
    b_t = delta_t * B_t * x_t                     # (B, L, D, N)
    h_t = a_t * h_{t-1} + b_t                     # (B, D, N),  h_{-1} = 0
    y_t = sum_n C_t[n] * h_t[d, n]                # (B, L, D)

`A` is negative and `delta` is positive, so `a` is in `(0, 1)`: the state decays
and the recurrence is stable by construction. That is also what makes the prefix
product well behaved.

Why three:

* `selective_scan_reference` is the definition, written as the loop it is. It is
  the thing the other two are checked against.
* `selective_scan_chunked` is what a real kernel does: a short loop *inside* each
  chunk, then a scan over chunk summaries. O(L) work, but few loop iterations.
* `selective_scan_associative` uses the fact that the recurrence is an
  associative monoid, and scans with log2(L) parallel steps and no sequential
  dependence at all.

A recurrence this simple is easy to get subtly wrong, and the usual way the
mistake survives is that the fast path is never compared against the slow one —
it is only ever compared against a loss curve, which will happily go down for a
model that is quietly discarding the past. So the tests assert the three are
equal to float64 tolerance on random, adversarial and degenerate inputs, and
assert that none of them can see the future.

The monoid, for reference. An element `(a, b)` stands for "multiply the incoming
state by `a`, then add `b`". Composing two of them in time order is

    (a1, b1) then (a2, b2)  =  (a2 * a1, a2 * b1 + b2)

which is associative, so a prefix scan is well defined. The identity is `(1, 0)`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def build_scan_terms(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor
) -> tuple[Tensor, Tensor]:
    """Turn inputs into the `(a, b)` pair the recurrence consumes.

    Shapes in: ``x, delta`` are ``(B, L, D)``; ``A`` is ``(D, N)``; ``B, C`` are
    ``(B, L, N)``. Shapes out: ``a, b`` are ``(B, L, D, N)``.
    """
    a = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
    b = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D, N)
    return a, b


def _readout(h: Tensor, C: Tensor) -> Tensor:
    """``y_t = sum_n C_t[n] * h_t[d, n]``; ``h`` is ``(B, L, D, N)``."""
    return (h * C.unsqueeze(2)).sum(-1)  # (B, L, D)


def selective_scan_reference(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor
) -> Tensor:
    """The definition, as an explicit loop over time. Slow, and the ground truth."""
    a, b = build_scan_terms(x, delta, A, B, C)
    BATCH, LENGTH, DIM, _ = a.shape

    h = x.new_zeros(BATCH, DIM, A.shape[-1])
    ys = []
    for t in range(LENGTH):
        h = a[:, t] * h + b[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1)  # (B, L, D)


def selective_scan_chunked(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk: int = 64
) -> Tensor:
    """Chunked scan: a short loop inside each chunk, then a scan over summaries.

    Within a chunk, run the recurrence from zero to get ``hl`` and keep the
    running prefix product ``cum``. Because the recurrence is linear,

        h_t = cum_t * h_in + hl_t

    and a chunk's summary is exactly ``(cum_last, hl_last)``. Scan those
    summaries for the incoming state of each chunk, then correct every position
    inside the chunk with one multiply and one add.
    """
    a, b = build_scan_terms(x, delta, A, B, C)
    BATCH, LENGTH, DIM, N = a.shape
    chunk = max(1, min(chunk, LENGTH))
    n_chunks = math.ceil(LENGTH / chunk)
    padded = n_chunks * chunk

    def pad(t: Tensor) -> Tensor:
        if padded == LENGTH:
            return t
        tail = t.new_zeros(BATCH, padded - LENGTH, *t.shape[2:])
        return torch.cat([t, tail], dim=1)

    ap = pad(a).view(BATCH, n_chunks, chunk, DIM, N)
    bp = pad(b).view(BATCH, n_chunks, chunk, DIM, N)

    # Local recurrence from zero, and the inclusive prefix product, per chunk.
    local = ap.new_empty(BATCH, n_chunks, chunk, DIM, N)
    prefix = ap.new_empty(BATCH, n_chunks, chunk, DIM, N)
    acc = ap.new_zeros(BATCH, n_chunks, DIM, N)
    cum = ap.new_ones(BATCH, n_chunks, DIM, N)
    for i in range(chunk):
        acc = ap[:, :, i] * acc + bp[:, :, i]
        cum = cum * ap[:, :, i]
        local[:, :, i] = acc
        prefix[:, :, i] = cum

    # Scan the per-chunk summaries for each chunk's incoming state.
    a_end = prefix[:, :, -1]  # (B, C, D, N)
    b_end = local[:, :, -1]
    h_in = a_end.new_zeros(BATCH, n_chunks, DIM, N)
    running = a_end.new_zeros(BATCH, DIM, N)
    for c in range(n_chunks):
        h_in[:, c] = running
        running = a_end[:, c] * running + b_end[:, c]

    h = prefix * h_in.unsqueeze(2) + local
    h = h.view(BATCH, padded, DIM, N)[:, :LENGTH]
    return _readout(h, C)


def selective_scan_associative(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor
) -> Tensor:
    """Hillis-Steele inclusive scan over the ``(a, b)`` monoid: log2(L) steps.

    No sequential dependence at all, which is the point of the formulation. The
    operator is ``(a1,b1) + (a2,b2) = (a2*a1, a2*b1 + b2)``, so step ``d`` reads
    position ``t-d`` and combines it in front of position ``t``.
    """
    a, b = build_scan_terms(x, delta, A, B, C)
    LENGTH = a.shape[1]

    step = 1
    while step < LENGTH:
        shifted_a = a.new_ones(a.shape)
        shifted_b = b.new_zeros(b.shape)
        shifted_a[:, step:] = a[:, :-step]
        shifted_b[:, step:] = b[:, :-step]
        # (a_{t-step}, b_{t-step}) composed before (a_t, b_t).
        b = b + a * shifted_b
        a = a * shifted_a
        step *= 2

    # After the scan, b holds h_t directly (h_{-1} = 0, so prefix b is the state).
    return _readout(b, C)


def _scan_one_chunk(
    x_c: Tensor, delta_c: Tensor, A: Tensor, B_c: Tensor, C_c: Tensor, h_in: Tensor
) -> tuple[Tensor, Tensor]:
    """Run the recurrence over one chunk, from a given incoming state.

    Only `(B, chunk, D, N)` temporaries exist at any moment. That is the point:
    the previous implementation built `a` and `b` for the *entire* sequence at
    once, which made a linear-time algorithm cost more memory than the quadratic
    one it was supposed to beat.
    """
    a = torch.exp(delta_c.unsqueeze(-1) * A)  # (B, K, D, N)
    b = delta_c.unsqueeze(-1) * B_c.unsqueeze(2) * x_c.unsqueeze(-1)
    h = h_in
    outputs = []
    for i in range(x_c.shape[1]):
        h = a[:, i] * h + b[:, i]
        outputs.append((h * C_c[:, i].unsqueeze(1)).sum(-1))
    return torch.stack(outputs, dim=1), h


def selective_scan_streaming(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk: int = 64
) -> Tensor:
    """Constant-memory scan: the peak working set is one chunk, not the sequence.

    Memory is `O(B * chunk * D * N)` regardless of length, so the state a model
    carries while reading a long context genuinely does not grow with it. Under
    `torch.no_grad()` nothing is retained and this is the whole story.

    With grad enabled this is still *correct* but not memory-efficient, because
    autograd keeps every chunk's intermediates alive for the backward pass —
    which is exactly what `selective_scan_checkpointed` exists to fix.
    """
    length = x.shape[1]
    h = x.new_zeros(x.shape[0], x.shape[2], A.shape[-1])
    pieces = []
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        y_c, h = _scan_one_chunk(
            x[:, start:end], delta[:, start:end], A, B[:, start:end],
            C[:, start:end], h,
        )
        pieces.append(y_c)
    return torch.cat(pieces, dim=1)


def selective_scan_checkpointed(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk: int = 64
) -> Tensor:
    """Training path: stream the chunks, and recompute each one in backward.

    Every chunk is wrapped in `torch.utils.checkpoint`, so the forward pass keeps
    only that chunk's inputs and its incoming state instead of every intermediate
    it produced. Peak memory falls from `O(B * L * D * N)` to
    `O(B * L * D * N / chunk)` for the saved states plus `O(B * chunk * D * N)`
    for the one chunk being recomputed.

    The gradients are identical to the un-checkpointed path. `tests/test_scan.py`
    asserts that against the reference autograd rather than assuming it, because
    a checkpointing mistake produces plausible-looking wrong gradients that
    training quietly absorbs.
    """
    from torch.utils.checkpoint import checkpoint

    length = x.shape[1]
    h = x.new_zeros(x.shape[0], x.shape[2], A.shape[-1])
    pieces = []
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        chunk_args = (
            x[:, start:end], delta[:, start:end], A, B[:, start:end],
            C[:, start:end],
        )
        if torch.is_grad_enabled() and any(t.requires_grad for t in chunk_args):
            y_c, h = checkpoint(
                _scan_one_chunk, *chunk_args, h, use_reentrant=False
            )
        else:
            y_c, h = _scan_one_chunk(*chunk_args, h)
        pieces.append(y_c)
    return torch.cat(pieces, dim=1)


#: What the model uses. The other implementations exist to check it.
def selective_scan(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk: int = 64
) -> Tensor:
    if torch.is_grad_enabled():
        return selective_scan_checkpointed(x, delta, A, B, C, chunk=chunk)
    return selective_scan_streaming(x, delta, A, B, C, chunk=chunk)
