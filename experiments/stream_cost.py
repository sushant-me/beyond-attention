"""What does it cost to keep reading?

Training compares architectures on a fixed-length sequence. Real use is a stream,
and the resource that decides what you can build is the state carried between
tokens. This measures that state directly, at lengths where the difference stops
being academic.

Reported per model:

* ``state_elements`` -- tensors carried between steps. For the SSM this is a
  constant; for attention it is the key/value cache and grows.
* ``state_bytes``    -- the same in bytes, float32.
* ``seconds``        -- wall time for the whole stream, which is *not* the point
  (the loop is deliberately unoptimised) but is reported so nobody reads the
  memory result as a speed result.

The interesting row is the last one. At 65,536 tokens the SSM carries the same
state it carried at 8, while attention carries 8 MiB per layer and has to keep
growing. That is the whole difference, and it is the one thing an attention model
cannot be made to do without changing what it is.

Usage:
    python experiments/stream_cost.py --out stream-cost.json
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

LENGTHS = [256, 1024, 4096, 16384, 65536]
# Attention streaming is O(L^2) in this unoptimised loop, so it is only *run*
# far enough to validate the closed form. Beyond that the cache size is
# predicted -- and the prediction is asserted against measurement at every
# length where both exist, so this is a checked shortcut, not an assumption.
ATTENTION_MEASURED_UP_TO = 4096
VOCAB = 33
D_MODEL = 64
N_LAYERS = 2


def measure(model, length: int, batch: int = 1) -> dict:
    tokens = torch.randint(0, VOCAB, (batch, length))
    state = init_stream(model, batch)
    peak = 0
    start = time.perf_counter()
    with torch.no_grad():
        for t in range(length):
            _, state = stream_step(model, tokens[:, t], state)
            peak = max(peak, state.numel())
    elapsed = time.perf_counter() - start
    return {
        "length": length,
        "state_elements": state.numel(),
        "peak_state_elements": peak,
        "state_bytes": state.numel() * 4,
        "seconds": round(elapsed, 3),
        "ms_per_token": round(1000 * elapsed / length, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stream-cost.json")
    ap.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    a = ap.parse_args()

    torch.manual_seed(0)
    ssm, attention = build_pair(VOCAB, d_model=D_MODEL, n_layers=N_LAYERS)
    ssm.eval()
    attention.eval()

    results = {
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "lengths": a.lengths,
            "batch": 1,
            "dtype": "float32",
        },
        "ssm": [],
        "attention": [],
        "fixed_state_bytes": {"ssm": ssm_state_bytes(ssm, 1), "attention": 0},
    }

    print("%-10s | %-28s | %-28s" % ("length", "SSM state", "attention KV cache"))
    print("-" * 74)
    for n in a.lengths:
        s = measure(ssm, n)
        if n <= ATTENTION_MEASURED_UP_TO:
            v = measure(attention, n)
            v["measured"] = True
        else:
            # Predicted from the validated formula; flagged so the table can
            # say which rows were run.
            elts = kv_cache_bytes(attention, n, 1) // 4
            v = {"length": n, "state_elements": elts, "peak_state_elements": elts,
                 "state_bytes": elts * 4, "seconds": None, "ms_per_token": None,
                 "measured": False}
        results["ssm"].append(s)
        results["attention"].append(v)
        print(
            "%-10d | %8d elts %7d KiB       | %8d elts %7d KiB"
            % (n, s["state_elements"], s["state_bytes"] // 1024,
               v["state_elements"], v["state_bytes"] // 1024)
        )

    # Cross-check the measured growth against the closed form, so a wrong
    # formula cannot hide behind a plausible-looking table.
    checked = 0
    for n, v in zip(a.lengths, results["attention"]):
        predicted = kv_cache_bytes(attention, n, 1) // 4
        assert predicted == v["state_elements"], (
            f"kv_cache_bytes disagrees at {n}: {predicted} vs {v['state_elements']}"
        )
        checked += 1 if v["measured"] else 0
    print(f"\nkv_cache_bytes() agrees with {checked} measured length(s).")

    ssm_elts = {r["state_elements"] for r in results["ssm"]}
    print(f"SSM state across all lengths: {ssm_elts}  (must be a single value)")
    assert len(ssm_elts) == 1, "SSM state changed with length"

    Path(a.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
