"""Does the stream actually use constant memory, or does it just say so?

The streaming section elsewhere reports the carried state as `numel * 4`. That is
arithmetic on a tensor shape, and arithmetic is not evidence: it says what the
state *should* cost, not what the process does. This experiment measures the
process.

Three things get measured, and the second two exist because the first is only
meaningful next to them:

1. **The SSM streamed for real**, with resident memory sampled throughout. The
   claim is that RSS is flat over hundreds of thousands of tokens -- not flat
   after subtracting a model of what should be there, just flat.
2. **What attention must hold at the same length.** Its cache is `(B, heads, L,
   head_dim)` per layer, twice, and the streaming loop builds it one token at a
   time. Those tensors are allocated here exactly as `_attention_step` allocates
   them, because the honest comparison is against the allocation the model is
   forced to make -- not against a re-derivation of it. This is an allocation and
   not a streamed run: streaming attention to these lengths costs `O(L^2)` time,
   which is a separate and already-measured fact.
3. **The outputs, which do grow.** `stream_step` discards each step's logits, so
   the loop is bounded. `stream_sequence` appends them and returns a stacked
   `(B, L, vocab)` tensor, so that path is `O(L)`. That is the caller's choice
   rather than the model's state, but a reader who conflates the two would be
   wrong in a way that flatters the architecture, so both are measured.

On reading RSS: the CPU allocator caches freed blocks rather than returning them
to the OS, so RSS is an upper bound that plateaus instead of a precise live-byte
count. That makes it the *right* instrument for a flatness claim -- if the loop
were accumulating, no amount of allocator reuse would hide it.

Usage:
    python -u experiments/stream_memory.py --out stream-memory.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from beyond_attention.model import build_pair
from beyond_attention.streaming import (
    init_stream,
    kv_cache_bytes,
    ssm_state_bytes,
    stream_step,
)

VOCAB = 33
D_MODEL = 64
N_LAYERS = 2


def rss_bytes() -> int:
    """Current resident set size, from /proc. 0 if unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def mb(n: int) -> float:
    return n / 1024**2


def stream_and_sample(model, length: int, sample_every: int) -> dict:
    """Stream `length` tokens, discarding logits, sampling RSS as we go."""
    tokens = torch.randint(0, VOCAB, (1, length))
    state = init_stream(model, 1)
    baseline_state = state.numel()
    samples: list[tuple[int, int]] = []
    start = time.perf_counter()
    largest_state = baseline_state
    with torch.no_grad():
        for t in range(length):
            logits, state = stream_step(model, tokens[:, t], state)
            del logits  # the loop keeps nothing; see PART 3 for the other choice
            if state.numel() > largest_state:
                largest_state = state.numel()
            if t % sample_every == 0:
                samples.append((t, rss_bytes()))
    elapsed = time.perf_counter() - start
    samples.append((length - 1, rss_bytes()))

    # Flatness is judged after a warm-up: the first samples cover interpreter and
    # allocator growth, which is a start-up cost, not a per-token one.
    warm = [r for (t, r) in samples if t >= length // 8]
    return {
        "length": length,
        "seconds": round(elapsed, 2),
        "us_per_token": round(1e6 * elapsed / length, 1),
        "state_elements_start": baseline_state,
        "state_elements_max": largest_state,
        "state_constant": largest_state == baseline_state,
        "state_bytes_arithmetic": ssm_state_bytes(model, 1),
        "samples": [{"token": t, "rss_bytes": r} for t, r in samples],
        "rss_after_warmup_min": min(warm),
        "rss_after_warmup_max": max(warm),
        "rss_after_warmup_growth_bytes": max(warm) - min(warm),
        "rss_after_warmup_growth_fraction": (max(warm) - min(warm)) / min(warm),
    }


def attention_cache_rss(attention, length: int) -> dict:
    """Allocate the KV cache the attention streaming loop is forced to build.

    Mirrors `_attention_step`: `(B, n_heads, L, head_dim)` for keys and values,
    per attention layer.
    """
    caches = []
    before = rss_bytes()
    for block in attention.blocks:
        heads = getattr(block.attn, "n_heads", None) or block.attn.num_heads
        head_dim = getattr(block.attn, "head_dim", None) or (
            block.attn.embed_dim // heads
        )
        shape = (1, heads, length, head_dim)
        k = torch.zeros(shape, dtype=torch.float32)
        v = torch.zeros(shape, dtype=torch.float32)
        k[0, 0, 0, 0] = 1.0  # touch a page so the allocation is resident
        v[0, 0, 0, 0] = 1.0
        caches.extend([k, v])
    after = rss_bytes()
    live = sum(c.numel() for c in caches) * 4
    del caches
    return {
        "length": length,
        "cache_bytes_arithmetic": kv_cache_bytes(attention, length, 1),
        "candidate_bytes_allocated": live,
        "rss_delta_bytes": after - before,
    }


def cache_growth_control(attention, length: int, sample_every: int) -> dict:
    """Positive control: the same RSS sampling, over a really-growing allocation.

    PART 1 reports a flat line. A flat line is only evidence if the instrument
    can see growth at all, so this walks the identical sampling loop while the
    attention KV cache grows underneath it, touching one page per row as it
    goes. If this does not rise, PART 1 means nothing.
    """
    heads = getattr(attention.blocks[0].attn, "n_heads", None) or (
        attention.blocks[0].attn.num_heads
    )
    head_dim = getattr(attention.blocks[0].attn, "head_dim", None) or (
        attention.blocks[0].attn.embed_dim // heads
    )
    n_attn_layers = sum(
        1 for b in attention.blocks if hasattr(b.attn, "num_heads")
        or hasattr(b.attn, "n_heads")
    )
    shape = (1, heads, length, head_dim)
    # `torch.empty`, not `zeros`: zeros writes every page immediately, so the
    # whole allocation becomes resident before the first sample and the control
    # reads flat. That is exactly how the first version of this control failed.
    caches = [torch.empty(shape) for _ in range(2 * max(n_attn_layers, 1))]
    samples: list[tuple[int, int]] = []
    for t in range(0, length, sample_every):
        stop = min(t + sample_every, length)
        for c in caches:
            c[0, :, t:stop, :] = 1.0  # make the pages resident
        samples.append((t, rss_bytes()))
    samples.append((length - 1, rss_bytes()))
    live = sum(c.numel() for c in caches) * 4
    warm = [r for (t, r) in samples if t >= length // 8]
    del caches
    return {
        "length": length,
        "candidate_bytes_allocated": live,
        "samples": [{"token": t, "rss_bytes": r} for t, r in samples],
        "rss_after_warmup_min": min(warm),
        "rss_after_warmup_max": max(warm),
        "rss_after_warmup_growth_bytes": max(warm) - min(warm),
        "rss_after_warmup_growth_fraction": (max(warm) - min(warm)) / min(warm),
    }


def output_retention_rss(length: int) -> dict:
    """What `stream_sequence` returns: every step's logits, stacked."""
    before = rss_bytes()
    rows = [torch.zeros(1, VOCAB) for _ in range(min(length, 1 << 16))]
    stacked = torch.stack(rows, dim=1)
    after = rss_bytes()
    live = stacked.numel() * 4
    del rows, stacked
    return {
        "length_allocated": min(length, 1 << 16),
        "note": "capped at 65536 rows for runtime; growth is linear in rows",
        "stacked_bytes_if_full_length": length * VOCAB * 4,
        "rss_delta_bytes": after - before,
        "candidate_bytes_allocated": live,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stream-memory.json")
    ap.add_argument("--length", type=int, default=1 << 18,
                    help="tokens to stream for real (default 262,144)")
    ap.add_argument("--sample-every", type=int, default=1 << 14)
    ap.add_argument("--cache-lengths", type=int, nargs="*",
                    default=[1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20])
    ap.add_argument("--control-length", type=int, default=1 << 18,
                    help="length for the positive control; its job is to prove "
                         "the instrument sees growth, not to allocate a gigabyte")
    a = ap.parse_args()

    torch.manual_seed(0)
    ssm, attention = build_pair(VOCAB, d_model=D_MODEL, n_layers=N_LAYERS)
    ssm.eval()
    attention.eval()

    results: dict = {
        "config": {"d_model": D_MODEL, "n_layers": N_LAYERS, "vocab": VOCAB,
                   "torch": torch.__version__},
    }
    out_path = Path(a.out)

    print("=" * 78)
    print("PART 1  The SSM, streamed for real, with RSS sampled throughout")
    print("=" * 78)
    r = stream_and_sample(ssm, a.length, a.sample_every)
    results["ssm_stream"] = r
    print(f"streamed {r['length']:,} tokens in {r['seconds']}s "
          f"({r['us_per_token']} us/token)")
    print(f"{'token':>12}  {'RSS (MB)':>10}")
    for s in r["samples"]:
        print(f"{s['token']:>12,}  {mb(s['rss_bytes']):>10.1f}")
    print()
    print(f"state elements: start {r['state_elements_start']}, "
          f"max {r['state_elements_max']} -> constant: {r['state_constant']}")
    print(f"state bytes (arithmetic)      : {r['state_bytes_arithmetic']:,}")
    print(f"RSS after warm-up             : {mb(r['rss_after_warmup_min']):.1f} - "
          f"{mb(r['rss_after_warmup_max']):.1f} MB "
          f"(spread {mb(r['rss_after_warmup_growth_bytes']):.1f} MB, "
          f"{100 * r['rss_after_warmup_growth_fraction']:.2f}%)")
    print()
    print(f"  positive control -- same instrument, KV cache growing to "
          f"{a.control_length:,} tokens:")
    ctrl = cache_growth_control(attention, a.control_length, a.sample_every)
    results["cache_growth_control"] = ctrl
    print(f"    RSS after warm-up: {mb(ctrl['rss_after_warmup_min']):.1f} -> "
          f"{mb(ctrl['rss_after_warmup_max']):.1f} MB "
          f"(growth {mb(ctrl['rss_after_warmup_growth_bytes']):.1f} MB, "
          f"{100 * ctrl['rss_after_warmup_growth_fraction']:.1f}%)")
    print("    -> the instrument sees growth; the flat line above is a result")
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    print("=" * 78)
    print("PART 2  What attention must hold at the same lengths")
    print("=" * 78)
    print(f"{'length':>12} {'cache (arithmetic)':>20} {'RSS delta (measured)':>22}")
    results["attention_cache"] = []
    for n in a.cache_lengths:
        c = attention_cache_rss(attention, n)
        results["attention_cache"].append(c)
        print(f"{n:>12,} {mb(c['cache_bytes_arithmetic']):>17.1f} MB "
              f"{mb(c['rss_delta_bytes']):>19.1f} MB")
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    print("=" * 78)
    print("PART 3  The outputs, which do grow -- the caller's choice, not the state")
    print("=" * 78)
    o = output_retention_rss(a.length)
    results["output_retention"] = o
    print(f"stream_sequence returns (B, L, vocab); at this length that is "
          f"{mb(o['stacked_bytes_if_full_length']):.1f} MB")
    print(f"measured for a capped allocation: RSS delta "
          f"{mb(o['rss_delta_bytes']):.1f} MB")
    print("stream_step keeps nothing, which is what makes PART 1 flat.")
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    print("=" * 78)
    ssm_bytes = r["state_bytes_arithmetic"]
    biggest = max(results["attention_cache"], key=lambda c: c["length"])
    ratio = biggest["cache_bytes_arithmetic"] / ssm_bytes
    print(f"At {biggest['length']:,} tokens:")
    print(f"  SSM, RSS growth per token, measured : "
          f"{mb(r['rss_after_warmup_growth_bytes']):.2f} MB "
          f"({100 * r['rss_after_warmup_growth_fraction']:.2f}%) over "
          f"{r['length']:,} tokens")
    print(f"  (positive control, same instrument  : "
          f"{mb(ctrl['rss_after_warmup_growth_bytes']):.1f} MB)")
    print(f"  SSM carried state, size (arithmetic): {mb(ssm_bytes):.3f} MB")
    print(f"  attention KV cache, size (arithmetic): "
          f"{mb(biggest['cache_bytes_arithmetic']):.1f} MB")
    print(f"  size ratio                           : {ratio:,.0f}x")
    print("=" * 78)

    # The claim is flatness, so it is asserted rather than described.
    assert r["state_constant"], "carried state grew while streaming"
    assert r["rss_after_warmup_growth_fraction"] < 0.10, (
        f"RSS grew {100 * r['rss_after_warmup_growth_fraction']:.1f}% after "
        f"warm-up; the constant-memory claim does not hold"
    )
    # The instrument check is relative, not absolute: the control must show
    # clearly more growth than the stream did, or a flat line proves nothing.
    ctrl_growth = ctrl["rss_after_warmup_growth_bytes"]
    ssm_growth = r["rss_after_warmup_growth_bytes"]
    assert ctrl_growth > 2 * 1024**2 and ctrl_growth > 10 * max(ssm_growth, 1), (
        f"positive control grew {mb(ctrl_growth):.1f} MB against the stream's "
        f"{mb(ssm_growth):.1f} MB; the instrument is not demonstrably able to "
        f"see growth, so the flat result is not evidence"
    )
    print("asserted: state constant; RSS spread under 10% after warm-up; and the")
    print("          positive control registered real growth with the same "
          "instrument")

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
