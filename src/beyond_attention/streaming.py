"""Incremental inference with a state that does not grow with context.

This module answers a question the rest of the project does not: **what does it
cost to keep reading?** Training compares architectures on a fixed sequence. Real
use is a stream, where the model must produce token ``t`` before it has seen
token ``t+1``, and where the resource that decides what you can build is the
state carried between steps.

A state-space model carries a fixed-size state, so that cost is constant. An
attention model cannot do this without a key/value cache that grows linearly in
the number of tokens seen. Both are implemented here, and
``experiments/stream_cost.py`` measures the difference rather than asserting it.

Two properties are enforced by tests, because both are easy to get subtly wrong:

* **Equivalence.** Stepping token by token must reproduce the parallel forward
  pass to within floating-point tolerance. A streaming path that is fast and
  wrong is worthless, and an off-by-one in the convolution tail is exactly the
  kind of bug that produces plausible-looking output.
* **Constant state (for the SSM).** The number of elements in the state must not
  depend on how many tokens have been consumed. That is the whole claim.

Causality is structural here: ``stream_step`` is only ever given one position,
so there is no future to leak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import AttentionBlock, CausalSelfAttention, LanguageModel, SelectiveSSMBlock


@dataclass
class BlockState:
    """One block's carried state. Every shape is fixed by the model, not by length."""

    # SSM blocks
    h: Tensor | None = None            # (B, d_inner, d_state) recurrence state
    conv_tail: Tensor | None = None    # (B, conv_kernel - 1, d_inner) conv history
    # Attention blocks -- note the sequence dimension: this is the KV cache, and
    # it is the thing that grows.
    k_cache: Tensor | None = None      # (B, n_heads, t, head_dim)
    v_cache: Tensor | None = None      # (B, n_heads, t, head_dim)


@dataclass
class StreamState:
    """The full carried state: one BlockState per layer, plus the embedding output."""

    blocks: list[BlockState] = field(default_factory=list)
    tokens_seen: int = 0

    def numel(self) -> int:
        """Total elements carried. Constant for an SSM, linear for attention."""
        total = 0
        for b in self.blocks:
            for t in (b.h, b.conv_tail, b.k_cache, b.v_cache):
                if t is not None:
                    total += t.numel()
        return total


def init_stream(model: LanguageModel, batch_size: int, device=None) -> StreamState:
    """Allocate a zeroed state. This is the only allocation; steps reuse it."""
    states: list[BlockState] = []
    for block in model.blocks:
        if isinstance(block, SelectiveSSMBlock):
            kernel = block.conv.conv.kernel_size[0]
            states.append(
                BlockState(
                    h=torch.zeros(
                        batch_size, block.d_inner, block.d_state, device=device
                    ),
                    conv_tail=torch.zeros(
                        batch_size, kernel - 1, block.d_inner, device=device
                    ),
                )
            )
        elif isinstance(block, AttentionBlock):
            states.append(BlockState())
        else:  # pragma: no cover - defensive
            raise TypeError(f"no streaming path for {type(block).__name__}")
    return StreamState(blocks=states)


# --------------------------------------------------------------------- SSM ---

def _ssm_step(
    block: SelectiveSSMBlock, x_t: Tensor, state: BlockState
) -> tuple[Tensor, BlockState]:
    """One timestep of the S6 recurrence. ``x_t`` is ``(B, d_model)``.

    This is `SelectiveSSMBlock.forward` with the sequence dimension removed. The
    arithmetic is deliberately identical, term for term, so the equivalence test
    is a real check rather than a check of a rewritten approximation.
    """
    residual = x_t
    h_in = block.norm(x_t)

    x_branch, gate = block.in_proj(h_in).chunk(2, dim=-1)  # (B, d_inner) each

    # Causal depthwise conv: the padded parallel form sees kernel-1 zeros then
    # the inputs, so the streaming form must carry exactly those last inputs.
    window = torch.cat([state.conv_tail, x_branch.unsqueeze(1)], dim=1)  # (B, k, d_inner)
    kernel = block.conv.conv.kernel_size[0]
    # nn.Conv1d stores depthwise weight as (channels, 1, kernel); squeeze the
    # group axis and transpose so it broadcasts against (B, k, d_inner).
    weight = block.conv.conv.weight.squeeze(1).t().unsqueeze(0)  # (1, k, d_inner)
    bias = block.conv.conv.bias                                  # (d_inner,)
    conv_out = (window * weight).sum(dim=1) + bias
    x_branch = F.silu(conv_out)

    projected = block.x_proj(x_branch)
    delta, B_t, C_t = projected.split(
        [block.dt_rank, block.d_state, block.d_state], dim=-1
    )
    delta = F.softplus(block.dt_proj(delta))            # (B, d_inner)
    A = -torch.exp(block.A_log)                         # (d_inner, d_state)

    a = torch.exp(delta.unsqueeze(-1) * A)              # (B, d_inner, d_state)
    b = delta.unsqueeze(-1) * B_t.unsqueeze(1) * x_branch.unsqueeze(-1)
    h = a * state.h + b
    y = (h * C_t.unsqueeze(1)).sum(-1) + block.D * x_branch
    y = y * F.silu(gate)
    out = residual + block.out_proj(y)

    new_state = BlockState(
        h=h,
        conv_tail=window[:, -(kernel - 1):, :] if kernel > 1 else window[:, :0, :],
    )
    return out, new_state


# --------------------------------------------------------------- attention ---

def _attention_step(
    block: AttentionBlock, x_t: Tensor, state: BlockState
) -> tuple[Tensor, BlockState]:
    """One timestep of the Transformer block, appending to a growing KV cache.

    The query is the newest position and the cache holds every position up to and
    including it, so no causal mask is needed: causality is what the cache *is*.
    """
    h = block.norm1(x_t).unsqueeze(1)                   # (B, 1, d_model)

    if block.causal_mode == "sdpa":
        attn: CausalSelfAttention = block.attn
        batch, _, dim = h.shape
        qkv = attn.qkv(h).reshape(batch, 1, 3, attn.n_heads, attn.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)            # (B, n_heads, 1, head_dim)
        k_cache = k if state.k_cache is None else torch.cat([state.k_cache, k], dim=2)
        v_cache = v if state.v_cache is None else torch.cat([state.v_cache, v], dim=2)
        attended = F.scaled_dot_product_attention(q, k_cache, v_cache)
        attended = attended.transpose(1, 2).reshape(batch, 1, dim)
        attended = attn.out(attended)
    else:
        # nn.MultiheadAttention path, done longhand.
        #
        # Passing already-projected q/k/v back into `mha(...)` double-projects
        # them: MultiheadAttention applies `in_proj_weight` to whatever you give
        # it. The first version of this did exactly that and diverged from the
        # parallel path by ~9. So project once here, cache K/V, and run the
        # attention arithmetic directly -- which is also what makes it a cache
        # rather than a re-computation of the whole prefix each step.
        mha = block.attn
        q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
        heads = mha.num_heads
        head_dim = mha.embed_dim // heads

        def _split(t: Tensor) -> Tensor:            # (B, 1, d) -> (B, H, 1, hd)
            b = t.shape[0]
            return t.reshape(b, 1, heads, head_dim).transpose(1, 2)

        q = _split(F.linear(h, q_w, None))
        k = _split(F.linear(h, k_w, None))
        v = _split(F.linear(h, v_w, None))
        k_cache = k if state.k_cache is None else torch.cat([state.k_cache, k], dim=2)
        v_cache = v if state.v_cache is None else torch.cat([state.v_cache, v], dim=2)

        scores = (q @ k_cache.transpose(-2, -1)) / math.sqrt(head_dim)
        probs = torch.softmax(scores, dim=-1)
        ctx = probs @ v_cache                          # (B, H, 1, hd)
        b = ctx.shape[0]
        ctx = ctx.transpose(1, 2).reshape(b, 1, mha.embed_dim)
        attended = mha.out_proj(ctx)

    x = x_t + attended.squeeze(1)
    x = x + block.mlp(block.norm2(x))
    return x, BlockState(k_cache=k_cache, v_cache=v_cache)


# ------------------------------------------------------------------- model ---

def stream_step(
    model: LanguageModel, token: Tensor, state: StreamState
) -> tuple[Tensor, StreamState]:
    """Consume one token per batch element. ``token`` is ``(B,)`` int64.

    Returns the logits for that position and the state to pass to the next call.
    The state returned is a new object, but its *size* never depends on how many
    tokens have been consumed.
    """
    if token.dim() == 2:
        token = token[:, 0]
    x = model.embed(token)                              # (B, d_model)
    new_blocks: list[BlockState] = []
    for block, bstate in zip(model.blocks, state.blocks):
        if isinstance(block, SelectiveSSMBlock):
            x, nb = _ssm_step(block, x, bstate)
        elif isinstance(block, AttentionBlock):
            x, nb = _attention_step(block, x, bstate)
        else:
            # Refuse loudly. A silent fallback would report a number for a
            # configuration that was never actually measured.
            raise TypeError(f"no streaming path for {type(block).__name__}")
        new_blocks.append(nb)
    logits = model.head(model.norm_f(x))
    return logits, StreamState(blocks=new_blocks, tokens_seen=state.tokens_seen + 1)


@torch.no_grad()
def stream_sequence(model: LanguageModel, tokens: Tensor) -> Tensor:
    """Run a whole sequence one position at a time.

    Output shape matches ``model(tokens)``, so the two can be compared directly.
    Slow by construction -- the point is the state, not the speed.
    """
    model.eval()
    batch, length = tokens.shape
    state = init_stream(model, batch, device=tokens.device)
    outs = []
    for t in range(length):
        logits, state = stream_step(model, tokens[:, t], state)
        outs.append(logits)
    return torch.stack(outs, dim=1)


def kv_cache_bytes(model: LanguageModel, length: int, batch_size: int = 1) -> int:
    """Bytes an attention model's cache would occupy after ``length`` tokens.

    Returns 0 for a pure SSM model. This is the quantity that decides whether a
    long context fits, and it is the one that has a length in it.
    """
    per_token = 0
    for block in model.blocks:
        if isinstance(block, AttentionBlock):
            if block.causal_mode == "sdpa":
                heads, head_dim = block.attn.n_heads, block.attn.head_dim
            else:
                heads = block.attn.num_heads
                head_dim = block.attn.embed_dim // heads
            # two tensors (K and V), one element each per head per position
            per_token += 2 * heads * head_dim * 4          # float32
    return per_token * length * batch_size


def ssm_state_bytes(model: LanguageModel, batch_size: int = 1) -> int:
    """Bytes an SSM model carries, independent of how long the context is."""
    total = 0
    for block in model.blocks:
        if isinstance(block, SelectiveSSMBlock):
            total += batch_size * block.d_inner * block.d_state
            total += batch_size * (block.conv.conv.kernel_size[0] - 1) * block.d_inner
    return total * 4
