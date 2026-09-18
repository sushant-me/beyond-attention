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
import math
import os
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
    """One configuration, one process.

    Under a hard address-space cap, for the same reason `long_context.py` is:
    the vectorised scan holds several full `(B, chunk, D, N)` tensors at once,
    so a large chunk at a long length can ask for more memory than the machine
    has. An uncapped attempt then becomes a system-level event rather than a
    failed measurement. With the cap the kernel refuses the allocation inside
    this child and the driver records the refusal as a row.
    """
    import os
    import resource

    import torch

    from beyond_attention.measurement import peak_rss_during
    from beyond_attention.ssm import selective_scan_checkpointed
    from beyond_attention.ssm import selective_scan_vectorized

    limit = int(os.environ.get("BA_AS_LIMIT_BYTES", 8 * 1024**3))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass

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
    parser.add_argument("--lengths", type=int, nargs="+",
                        help="sweep several lengths; the best configuration can "
                             "depend on length, which one length cannot show")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budget-gb", type=float, default=8.0,
                        help="address-space cap per child")
    parser.add_argument("--out", default="scan-inner.json")
    args = parser.parse_args()

    if args.inner and args.chunk:
        return run_one(args)

    lengths = args.lengths or [args.length]
    print(f"lengths={lengths} batch={args.batch} d_inner={args.dim} "
          f"d_state={args.state} threads={args.threads}")
    print(f"each child capped at {args.budget_gb:.1f} GiB of address space")

    env = dict(os.environ)
    env["BA_AS_LIMIT_BYTES"] = str(int(args.budget_gb * 1024**3))

    results: dict[str, dict] = {}
    for length in lengths:
        print()
        print(f"length={length}")
        print(f"{'inner':<12}{'chunk':>7}{'chunks':>8}{'seconds':>10}"
              f"{'peak MB':>10}{'agrees to':>16}")
        rows: dict[str, dict] = {}
        for chunk in args.chunks:
            for inner in ("loop", "vectorized"):
                sub = subprocess.run(
                    [sys.executable, str(pathlib.Path(__file__)),
                     "--inner", inner, "--chunk", str(chunk),
                     "--length", str(length), "--batch", str(args.batch),
                     "--dim", str(args.dim), "--state", str(args.state),
                     "--threads", str(args.threads)],
                    capture_output=True, text=True, env=env,
                )
                key = f"{inner}@{chunk}@L{length}"
                if sub.returncode != 0:
                    # A refused allocation is a result, not a crash.
                    note = (sub.stderr or sub.stdout).strip().splitlines()
                    results[key] = {
                        "inner": inner, "chunk": chunk, "length": length,
                        "failed": True,
                        "note": note[-1][:150] if note else
                                f"exit {sub.returncode}",
                    }
                    print(f"{inner:<12}{chunk:>7}{'—':>8}{'FAILED':>10}"
                          f"{'—':>10}  {results[key]['note'][:40]}")
                    continue
                row = json.loads(sub.stdout.strip().splitlines()[-1])
                results[key] = row
                rows[inner] = row

            loop, vec = rows.get("loop"), rows.get("vectorized")
            for inner in ("loop", "vectorized"):
                row = rows.get(inner)
                if row is None or row.get("length") != length:
                    continue
                agree = "—"
                if loop and vec:
                    # Relative agreement, not bit equality: the two paths order
                    # the same arithmetic differently, so demanding identical
                    # floats would report "different" for two correct
                    # implementations.
                    worst = max(
                        abs(loop[k] - vec[k]) / max(1.0, abs(loop[k]))
                        for k in ("out_sum", "grad_sum")
                    )
                    # These are float32 tensors, so eps is ~1.2e-07 and a sum
                    # accumulates a few of those. A threshold tighter than
                    # float32 resolution reports "different" for two correct
                    # implementations - which is what the first version of this
                    # check did.
                    agree = (f"yes ({worst:.1e})" if worst < 1e-5
                             else f"NO ({worst:.1e})")
                print(f"{inner:<12}{chunk:>7}{length // chunk:>8}"
                      f"{row['seconds']:>10.2f}{row['peak_rss_kb'] / 1024:>10.1f}"
                      f"{agree:>14}")

    if len(lengths) > 1:
        print()
        print("=" * 78)
        print("How each configuration scales with length")
        print("=" * 78)
        print("cost ~ length ** exponent, fitted by least squares over the "
              "lengths that succeeded")
        print(f"{'configuration':<22}{'exponent':>10}{'points':>8}"
              f"{'slowest':>10}{'peak MB':>10}")
        exponents: dict[str, float] = {}
        for chunk in args.chunks:
            for inner in ("loop", "vectorized"):
                pts = [
                    (length, results[f"{inner}@{chunk}@L{length}"]["seconds"])
                    for length in lengths
                    if not results.get(f"{inner}@{chunk}@L{length}", {})
                    .get("failed", False)
                ]
                if len(pts) < 2:
                    continue
                # Least squares on log-log. The exponent is the whole question:
                # 1.0 is linear in length, and anything above it means the cost
                # is growing faster than the sequence does.
                n = len(pts)
                lx = [math.log(p[0]) for p in pts]
                ly = [math.log(p[1]) for p in pts]
                mx, my = sum(lx) / n, sum(ly) / n
                denom = sum((v - mx) ** 2 for v in lx)
                slope = (sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
                         / denom) if denom else float("nan")
                exponents[f"{inner}@{chunk}"] = slope
                peak = max(
                    results[f"{inner}@{chunk}@L{length}"]["peak_rss_kb"]
                    for length in lengths
                    if not results.get(f"{inner}@{chunk}@L{length}", {})
                    .get("failed", False)
                )
                print(f"{inner + '@' + str(chunk):<22}{slope:>10.2f}{n:>8}"
                      f"{max(p[1] for p in pts):>10.2f}{peak / 1024:>10.1f}")
        results["_exponents"] = exponents

        print()
        print("fastest configuration at each length:")
        for length in lengths:
            cands = [
                (results[f"{inner}@{chunk}@L{length}"]["seconds"],
                 f"{inner}@{chunk}")
                for chunk in args.chunks
                for inner in ("loop", "vectorized")
                if not results.get(f"{inner}@{chunk}@L{length}", {})
                .get("failed", False)
            ]
            if cands:
                fastest = min(cands)
                print(f"  {length:>7}: {fastest[1]:<18}{fastest[0]:>9.2f}s")

    payload = {"config": vars(args), "results": results}
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
