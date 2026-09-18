"""Two sequence models with the same budget, so a comparison means something.

Both are deliberately small and deliberately plain. The point is not to compete
with a production model; it is to put a selective state-space model and a
Transformer on the same parameter count, the same data, the same optimiser and
the same step budget, and report what each one can and cannot learn.

The comparison is only worth reading if the budgets really are matched, so
`count_parameters` is asserted in the tests and printed by every experiment
rather than quoted from memory.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ssm import selective_scan, selective_scan_vectorized


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


def count_parameters(module: nn.Module) -> int:
    """Trainable parameters, counted from the tensors rather than declared."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class CausalDepthwiseConv1d(nn.Module):
    """A short convolution that cannot see the future.

    Left-padded to `kernel - 1` so output position `t` depends on inputs
    `t-kernel+1 .. t`. Getting this off by one is a silent leak, which is why
    the model-level causality test exists.
    """

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(
            channels, channels, kernel, groups=channels, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:  # (B, L, C)
        y = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(y).transpose(1, 2)


class SelectiveSSMBlock(nn.Module):
    """One Mamba-style block: project, short conv, selective scan, gate.

    `delta`, `B` and `C` are produced from the input, which is what makes the
    scan *selective*: the model chooses per token how much to write into the
    state, how much to decay it, and what to read back out. A non-selective SSM
    would use constants here, and that difference is the whole reason this
    architecture can do content-based lookup at all.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | None = None,
        scan_inner: str = "loop",
        scan_chunk: int = 64,
    ) -> None:
        super().__init__()
        if scan_inner not in ("loop", "vectorized"):
            raise ValueError(
                f"scan_inner must be 'loop' or 'vectorized', got {scan_inner!r}"
            )
        # Default is the sequential loop at chunk=64, which the comparison in
        # `experiments/scan_inner.py` found to be the only configuration that is
        # best on *both* axes: 3.89 s and 451 MB, against the vectorised scan's
        # 4.31 s / 3,303 MB at the same chunk and 8.56 s / 4,898 MB at chunk=256.
        #
        # The vectorised scan is not uniformly worse - at chunk=256 it is 1.9x
        # *faster* than the loop - but it pays roughly ten times the memory to
        # get there, because the Hillis-Steele scan holds several full
        # `(B, chunk, D, N)` tensors at once. It never wins on both axes, so the
        # loop is the default. It is kept because the tradeoff should invert on
        # a GPU, where log2 depth is the whole point, and because deleting a
        # measured negative result loses the evidence.
        #
        # An earlier version of this comment said the vectorised scan was slower
        # outright. That came from timings taken while a `torch.compile` job was
        # competing for CPU; re-measured on an idle machine, the time result
        # reversed. The memory result held.
        self.scan_inner = scan_inner
        self.scan_chunk = scan_chunk
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, conv_kernel)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # A is stored in log space and negated, so every decay is in (0, 1) and
        # the recurrence is stable for any parameter value the optimiser picks.
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(self.d_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Start delta small and positive so the state integrator begins in a
        # well-conditioned regime instead of saturated at one extreme.
        with torch.no_grad():
            self.dt_proj.bias.uniform_(
                math.log(math.expm1(0.02)), math.log(math.expm1(0.1))
            )

    def forward(self, x: Tensor) -> Tensor:  # (B, L, d_model)
        residual = x
        h = self.norm(x)

        x_branch, gate = self.in_proj(h).chunk(2, dim=-1)
        x_branch = F.silu(self.conv(x_branch))

        projected = self.x_proj(x_branch)
        delta, B, C = projected.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner), positive
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative

        if self.scan_inner == "vectorized":
            y = selective_scan_vectorized(
                x_branch, delta, A, B, C, chunk=self.scan_chunk
            )
        else:
            y = selective_scan(x_branch, delta, A, B, C, chunk=self.scan_chunk)
        y = y + self.D * x_branch  # skip connection straight from the input
        y = y * F.silu(gate)
        return residual + self.out_proj(y)


class CausalSelfAttention(nn.Module):
    """Causal attention written the modern way: the fused SDPA kernel directly.

    `nn.MultiheadAttention` is a convenient wrapper, but asking it for a causal
    mask means handing it an explicit `L x L` boolean tensor, and that can take
    it off the fused `scaled_dot_product_attention` path. Comparing a
    state-space model against attention-crippled-by-its-wrapper would be a
    comparison against a straw man, so this exists to give the baseline its best
    shot.

    Parameter count is identical to the wrapper with `bias=False`:
    `qkv` is `3 * d_model^2` and `out` is `d_model^2`, the same as
    `in_proj_weight` plus `out_proj.weight`.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"{d_model} is not divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.out(attended)


class AttentionBlock(nn.Module):
    """A pre-norm Transformer block: causal attention, then a small MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        causal_mode: str = "mask",
    ) -> None:
        super().__init__()
        if causal_mode not in ("mask", "sdpa"):
            raise ValueError(
                f"causal_mode must be 'mask' or 'sdpa', got {causal_mode!r}"
            )
        self.causal_mode = causal_mode
        self.norm1 = RMSNorm(d_model)
        self.attn = (
            CausalSelfAttention(d_model, n_heads)
            if causal_mode == "sdpa"
            else nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        )
        self.norm2 = RMSNorm(d_model)
        hidden = int(mlp_ratio * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        if self.causal_mode == "sdpa":
            # The fused kernel builds the causal mask itself and never
            # materialises the L x L matrix.
            attended = self.attn(h)
        else:
            # Explicit causal mask: position t attends to positions <= t only.
            length = h.shape[1]
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=h.device),
                diagonal=1,
            )
            attended, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class LanguageModel(nn.Module):
    """Embedding, a stack of blocks, and a tied output head.

    Weight tying keeps the comparison honest: the embedding table is usually one
    of the largest tensors at this scale, and letting one model pay for it twice
    would hand the other a free parameter advantage.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        block: str = "ssm",
        max_length: int = 512,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        if block == "ssm":
            self.blocks = nn.ModuleList(
                SelectiveSSMBlock(d_model, **block_kwargs) for _ in range(n_layers)
            )
        elif block == "attention":
            self.blocks = nn.ModuleList(
                AttentionBlock(d_model, **block_kwargs) for _ in range(n_layers)
            )
        else:
            raise ValueError(f"unknown block {block!r}")
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))


def build_pair(
    vocab_size: int,
    d_model: int = 64,
    n_layers: int = 2,
    block_kwargs: dict | None = None,
) -> tuple[LanguageModel, LanguageModel]:
    """Return `(ssm, attention)` models and assert the budgets are close.

    Raises rather than warns: an unfair comparison that reports itself as fair is
    worse than no comparison, and this is the one place the claim is made.
    """
    block_kwargs = block_kwargs or {"d_state": 16, "expand": 2, "conv_kernel": 4}
    ssm = LanguageModel(vocab_size, d_model, n_layers, "ssm", **block_kwargs)
    attention = LanguageModel(
        vocab_size, d_model, n_layers, "attention", n_heads=4, mlp_ratio=2.0
    )
    a, b = count_parameters(ssm), count_parameters(attention)
    ratio = max(a, b) / min(a, b)
    if ratio > 1.15:
        raise ValueError(
            f"parameter budgets differ by more than 15%: ssm={a} attention={b}"
        )
    return ssm, attention
