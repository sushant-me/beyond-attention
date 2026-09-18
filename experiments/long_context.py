"""Read a million tokens in nineteen kilobytes.

The scaling section asks how fast each model runs on a sequence it can both fit
and hold. This asks a different question: **what is the longest context each one
can actually read?** For a state-space model that is bounded by patience, because
its state is fixed. For attention it is bounded by memory.

The first version of this experiment answered that question by *trying*, and a
failure mode showed up that the design had no room for. At 131,072 tokens the
attempt did not raise -- it took the whole machine with it. `systemd-oomd` saw
memory at 15.6/16.0 GiB and swap at 28.8/31.9 GiB, and killed the entire process
scope: seventeen processes, no traceback, no exit code, nothing to catch. The
measurement was not "attention needs more memory than we have". The measurement
was a dead box.

So an attempt now runs in a **child process with a hard address-space cap**
(`RLIMIT_AS`). That changes the physics of the question in the way that matters:

* The kernel refuses the allocation *inside one child*, before the machine can
  start swapping. What used to be an unkillable system event becomes an ordinary
  `RuntimeError` that this script can catch.
* If the child dies anyway, the parent survives to record *how* -- signal, exit
  code, or a reported allocator failure -- and that record is the result.
* Peak resident memory comes back from `/proc/self/status` (`VmHWM`), so the
  growth curve is **measured rather than assumed**. Whether attention's memory
  is linear or quadratic in length is an empirical question about which SDPA
  kernel the runtime picks, and it gets an empirical answer here.

The distinction the docstring used to draw -- "a `RuntimeError` at a given length
is evidence; 'it would need 4 TB' is arithmetic" -- still holds, and now the
evidence is obtainable without risking the host.

Usage:
    python -u experiments/long_context.py --out long-context.json
    python -u experiments/long_context.py --attention-once 4096   # internal
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from beyond_attention.model import build_pair
from beyond_attention.streaming import init_stream, kv_cache_bytes, stream_step

VOCAB = 33
D_MODEL = 64
N_LAYERS = 2

# Lengths to stream the SSM at. The top one is a million tokens.
SSM_LENGTHS = [1 << 14, 1 << 16, 1 << 18, 1 << 20]

# Lengths to attempt the attention parallel forward at, ascending, in an
# isolated child. Whatever the child does, the parent keeps going.
ATTENTION_LENGTHS = [1 << 12, 1 << 14, 1 << 16, 1 << 17]

# Address-space cap for one attention attempt. This is the number that keeps the
# host alive, so it is deliberately a fraction of the smallest machine this is
# likely to run on rather than a fraction of *this* machine's RAM. 4 GiB is
# enough to locate the wall without letting a child drag a 16 GiB host into swap
# the way the uncapped version did.
DEFAULT_BUDGET_BYTES = 4 * 1024**3

# Threads per child. The forward pass is not the thing being timed here; hogging
# every core would just make the box unpleasant for whoever else is using it.
CHILD_THREADS = "4"

ATTENTION_ONCE = "--attention-once"


def peak_rss_bytes() -> int:
    """This process's high-water resident set, from /proc. 0 if unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def sdpa_context_probe() -> str:
    """What can be established cheaply about the fused-attention path.

    Deliberately worded: entering `SDPBackend.FLASH_ATTENTION` as a context
    manager raising nothing does **not** establish that a flash kernel runs on
    this device. It only means the enum is recognised. The evidence for which
    path is actually taken is the measured memory curve in PART 1, which is why
    that gets reported and this does not get trusted.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            pass
        return ("SDPBackend.FLASH_ATTENTION accepted as a context manager; "
                "this does NOT establish that a flash kernel is used on CPU")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return f"flash context rejected: {type(exc).__name__}"


def attention_once(length: int) -> int:
    """Child entry point: one forward pass under a hard address-space cap.

    Prints a single JSON object describing what happened and exits 0 either way,
    because "it failed" is a result and the parent needs to read it.
    """
    limit = int(os.environ.get("BA_AS_LIMIT_BYTES", DEFAULT_BUDGET_BYTES))
    result: dict = {"length": length, "ok": False, "note": "", "peak_rss_bytes": 0}
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError) as exc:
        result["note"] = f"could not set RLIMIT_AS to {limit}: {exc}"
        print(json.dumps(result))
        return 0

    try:
        torch.manual_seed(0)
        _, attention = build_pair(VOCAB, d_model=D_MODEL, n_layers=N_LAYERS)
        attention.eval()
        tokens = torch.randint(0, VOCAB, (1, length))
        started = time.perf_counter()
        with torch.no_grad():
            attention(tokens)
        result["forward_seconds"] = round(time.perf_counter() - started, 4)
        result["ok"] = True
        result["note"] = "ok"
    except (RuntimeError, torch.OutOfMemoryError, MemoryError) as exc:
        text = str(exc).replace("\n", " ")[:150]
        result["note"] = f"{type(exc).__name__}: {text}"
    except BaseException as exc:  # noqa: BLE001 - recorded as data
        result["note"] = f"{type(exc).__name__}: {str(exc)[:150]}"

    result["peak_rss_bytes"] = peak_rss_bytes()
    print(json.dumps(result))
    return 0


def attention_forward_succeeds(length: int, budget: int) -> dict:
    """Attempt one forward pass in a capped child; never risk the parent.

    Returns a dict with `ok`, a human-readable `note`, and the child's peak RSS
    when the child got far enough to report one.
    """
    env = dict(os.environ)
    env["BA_AS_LIMIT_BYTES"] = str(budget)
    env["OMP_NUM_THREADS"] = CHILD_THREADS
    env["MKL_NUM_THREADS"] = CHILD_THREADS
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), ATTENTION_ONCE,
           str(length)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=900
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "note": "child exceeded 900s", "peak_rss_bytes": 0}

    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            return {"ok": False, "note": "child exited 0 without a JSON line",
                    "peak_rss_bytes": 0}

    if proc.returncode < 0:
        return {
            "ok": False,
            "note": (f"child killed by signal {-proc.returncode} under a "
                     f"{budget / 1024**3:.1f} GiB address-space cap"),
            "peak_rss_bytes": 0,
        }

    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {
        "ok": False,
        "note": tail[-1][:160] if tail else f"child exit {proc.returncode}",
        "peak_rss_bytes": 0,
    }


def stream_ssm(model, length: int) -> dict:
    """Stream `length` tokens, checking the state never grows."""
    tokens = torch.randint(0, VOCAB, (1, length))
    state = init_stream(model, 1)
    baseline = state.numel()
    start = time.perf_counter()
    with torch.no_grad():
        for t in range(length):
            _, state = stream_step(model, tokens[:, t], state)
    elapsed = time.perf_counter() - start
    return {
        "length": length,
        "state_elements": state.numel(),
        "state_bytes": state.numel() * 4,
        "state_constant": state.numel() == baseline,
        "seconds": round(elapsed, 3),
        "us_per_token": round(1e6 * elapsed / length, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="long-context.json")
    ap.add_argument("--ssm-lengths", type=int, nargs="*", default=SSM_LENGTHS)
    ap.add_argument("--attention-lengths", type=int, nargs="*",
                    default=ATTENTION_LENGTHS)
    ap.add_argument("--attention-budget-gb", type=float,
                    default=DEFAULT_BUDGET_BYTES / 1024**3,
                    help="address-space cap for one attention attempt")
    ap.add_argument(ATTENTION_ONCE, type=int, default=None,
                    help="internal: run one capped attempt and exit")
    a = ap.parse_args()

    if a.attention_once is not None:
        return attention_once(a.attention_once)

    budget = int(a.attention_budget_gb * 1024**3)
    out_path = Path(a.out)

    torch.manual_seed(0)
    ssm, attention = build_pair(VOCAB, d_model=D_MODEL, n_layers=N_LAYERS)
    ssm.eval()
    attention.eval()

    results: dict = {
        "config": {"d_model": D_MODEL, "n_layers": N_LAYERS, "dtype": "float32",
                   "torch": torch.__version__,
                   "attention_budget_bytes": budget},
        "attention_forward": [],
        "ssm_stream": [],
    }

    def save() -> None:
        """Write what we have. A result on disk survives a later surprise."""
        out_path.write_text(json.dumps(results, indent=2) + "\n")

    print("=" * 78)
    print("PART 1  Can the attention baseline read the input at all?")
    print("=" * 78)
    print(f"each attempt runs in a child capped at {budget / 1024**3:.1f} GiB "
          f"of address space")
    print(f"fused-attention path: {sdpa_context_probe()}")
    print()
    print("%-10s %-9s %-12s %-11s %s"
          % ("length", "result", "peak RSS", "ms/token", "note"))
    for n in a.attention_lengths:
        start = time.perf_counter()
        r = attention_forward_succeeds(n, budget)
        elapsed = time.perf_counter() - start
        r["length"] = n
        r["seconds"] = round(elapsed, 3)
        r["kv_cache_bytes"] = kv_cache_bytes(attention, n, 1)
        # Worst case if the runtime ever takes the naive path: an L x L float32
        # score matrix per head. Recorded so the arithmetic sits next to the
        # measurement rather than replacing it.
        r["one_head_score_matrix_bytes"] = 4 * n * n
        peak = r.get("peak_rss_bytes") or 0
        if peak:
            # Interpretation only. The peak is the measurement; this is the
            # comparison that gives it meaning.
            r["peak_over_one_head_matrix"] = round(peak / (4 * n * n), 3)
        results["attention_forward"].append(r)
        save()

        peak = r.get("peak_rss_bytes") or 0
        fwd = r.get("forward_seconds")
        print("%-10s %-9s %-12s %-11s %s"
              % (f"{n:,}", "ok" if r["ok"] else "FAILED",
                 f"{peak / 1024**2:,.0f} MB" if peak else "-",
                 f"{fwd / n * 1e3:.3f}" if fwd else "-",
                 r["note"][:60]))
        if not r["ok"]:
            print(f"\n  -> the parallel forward stops working at {n:,} tokens.")
            break

    print()
    print("=" * 78)
    print("PART 2  The same model family, read as a stream")
    print("=" * 78)
    print("%-12s %-14s %-12s %-12s %s"
          % ("length", "state", "constant?", "seconds", "us/token"))
    for n in a.ssm_lengths:
        r = stream_ssm(ssm, n)
        results["ssm_stream"].append(r)
        save()
        print("%-12s %-14s %-12s %-12.2f %s"
              % (f"{n:,}", f"{r['state_elements']} elts", r["state_constant"],
                 r["seconds"], r["us_per_token"]))

    longest = max(a.ssm_lengths)
    cache_at_longest = kv_cache_bytes(attention, longest, 1)
    results["contrast"] = {
        "longest_streamed": longest,
        "ssm_state_bytes": results["ssm_stream"][-1]["state_bytes"],
        "attention_kv_cache_bytes": cache_at_longest,
    }

    print()
    print("=" * 78)
    print(f"At {longest:,} tokens:")
    print(f"  SSM carried state  : {results['ssm_stream'][-1]['state_bytes']:,} bytes")
    print(f"  attention KV cache : {cache_at_longest:,} bytes")
    print("=" * 78)

    states = {r["state_elements"] for r in results["ssm_stream"]}
    assert len(states) == 1, f"state grew while streaming: {states}"
    per_token = [r["us_per_token"] for r in results["ssm_stream"]]
    spread = max(per_token) / min(per_token)
    results["per_token_spread"] = round(spread, 2)
    print(f"state across all streamed lengths: {states} (constant)")
    print(f"per-token time spread: {spread:.2f}x across a "
          f"{longest // min(a.ssm_lengths):,}x range of lengths "
          f"({'linear' if spread < 3 else 'NOT linear -- investigate'})")
    assert spread < 3, "per-token cost grew with length; that is not linear time"

    save()
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
