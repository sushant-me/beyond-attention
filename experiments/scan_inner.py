"""Compare the two inner scans: the sequential loop and the vectorised scan.

Written because I expected the vectorised scan to win and it did not, and a
negative result that lives only in a commit message is not evidence.

Two things this script has to get right, both learned the hard way here:

* **One configuration per process.** Peak-RSS rise is only attributable if
  nothing large has been allocated and freed before it: torch reuses freed
  memory, so a second configuration measured in the same process can report a
  rise of nearly zero. The driver spawns a fresh interpreter per row.
* **The same inputs for both paths.** The first version of this script drew
  fresh random tensors inside the measurement, so the two paths got different
  inputs and the "difference" it printed between them was meaningless — a
  comparison that could not have failed. Here the inputs come from a fixed seed
  and each process reports a checksum, which the driver compares. A faster path
  that computes something else is not a faster path.

    python experiments/scan_inner.py --out scan-inner.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

SEED = 0


def _inputs(batch, length, dim, state):
    import torch

    torch.manual_seed(SEED)
    return (
        torch.randn(batch, length, dim, requires_grad=True),
        (torch.rand(batch, length, dim) + 1e-3).requires_grad_(True),
        (-torch.rand(dim, state) - 0.05).requires_grad_(True),
        torch.randn(batch, length, state, requires_grad=True),
        torch.randn(batch, length, state, requires_grad=True),
    )


def run_one(args) -> int:
    """One configuration, one process."""
    import torch

    from beyond_attention.measurement import peak_rss_during
    from beyond_attention.ssm import selective_scan_checkpointed
    from beyond_attention.ssm import selective_scan_vectorized

    torch.set_num_threads(args.threads)
    x, delta, A, B, C = _inputs(args.batch, args.length, args.dim, args.state)
    fn = (
        selective_scan_vectorized
        if args.inner == "vectorized"
        else selective_scan_checkpointed
    )

    def call():
        out = fn(x, delta, A, B, C, chunk=args.chunk)
        out.float().pow(2).mean().backward()
        return out

    for tensor in (x, delta, A, B, C):
        tensor.grad = None
    start = time.perf_counter()
    out, rise_kb = peak_rss_during(call)
    elapsed = time.perf_counter() - start

    print(json.dumps({
        "inner": args.inner,
        "chunk": args.chunk,
        "length": args.length,
        "batch": args.batch,
        "seconds": round(elapsed, 3),
        "peak_rss_kb": rise_kb,
        # Checksums, so the driver can prove both paths computed the same thing.
        "out_sum": float(out.detach().double().sum()),
        "grad_sum": float(x.grad.double().sum()),
        "finite": bool(torch.isfinite(out).all()),
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner", choices=["loop", "vectorized"])
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="scan-inner.json")
    args = parser.parse_args()

    if args.inner and args.chunk:
        return run_one(args)

    print(f"length={args.length} batch={args.batch} d_inner={args.dim} "
          f"d_state={args.state} threads={args.threads}")
    print(f"{'inner':<12}{'chunk':>7}{'chunks':>8}{'seconds':>10}"
          f"{'peak MB':>10}{'agrees to':>16}")

    results: dict[str, dict] = {}
    for chunk in args.chunks:
        rows: dict[str, dict] = {}
        for inner in ("loop", "vectorized"):
            sub = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__)),
                 "--inner", inner, "--chunk", str(chunk),
                 "--length", str(args.length), "--batch", str(args.batch),
                 "--dim", str(args.dim), "--state", str(args.state),
                 "--threads", str(args.threads)],
                capture_output=True, text=True,
            )
            if sub.returncode != 0:
                print(f"{inner:<12}{chunk:>7} FAILED: {sub.stderr.strip()[-120:]}")
                continue
            row = json.loads(sub.stdout.strip().splitlines()[-1])
            rows[inner] = row
            results[f"{inner}@{chunk}"] = row

        loop, vec = rows.get("loop"), rows.get("vectorized")
        for inner in ("loop", "vectorized"):
            row = rows.get(inner)
            if row is None:
                continue
            agree = "—"
            if loop and vec:
                # Relative agreement, not bit equality: the two paths order the
                # same arithmetic differently, so demanding identical floats
                # would report "different" for two correct implementations.
                worst = max(
                    abs(loop[k] - vec[k]) / max(1.0, abs(loop[k]))
                    for k in ("out_sum", "grad_sum")
                )
                # These are float32 tensors, so eps is ~1.2e-07 and a sum over 8192
                # steps accumulates a few of those. A threshold tighter than
                # float32 resolution reports "different" for two correct
                # implementations - which is what the first version of this
                # check did.
                agree = (f"yes ({worst:.1e})" if worst < 1e-5
                         else f"NO ({worst:.1e})")
            print(f"{inner:<12}{chunk:>7}{args.length // chunk:>8}"
                  f"{row['seconds']:>10.2f}{row['peak_rss_kb'] / 1024:>10.1f}"
                  f"{agree:>14}")

    payload = {"config": vars(args), "results": results}
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
