"""Measure one architecture at one sequence length, in a fresh process.

Peak memory is read from `ru_maxrss`, which is monotonic for the process's
lifetime — so it is only meaningful if nothing large has been allocated before
the measurement. That is why this runs one configuration per process and the
driver spawns it: measuring two lengths in one process would report the second
length's cost as the maximum of the two.
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import torch

from beyond_attention.model import LanguageModel, count_parameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=["ssm", "attention"], required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    # `ru_maxrss` is the process's peak and never falls, and importing torch
    # already costs several hundred megabytes. The attributable cost of this
    # configuration is therefore the growth over the peak reached *before any of
    # it existed* - so the baseline is taken here, before the model is built.
    # Taken after the model, the peak is usually already higher than anything
    # the forward pass adds and every configuration reports the same constant
    # (or zero), which looks like a measurement and is not one.
    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    kwargs = (
        {"d_state": 16, "expand": 2, "conv_kernel": 4}
        if args.block == "ssm"
        else {"n_heads": 4, "mlp_ratio": 2.0}
    )
    model = LanguageModel(
        args.vocab, args.d_model, args.n_layers, args.block, **kwargs
    )
    tokens = torch.randint(0, args.vocab, (args.batch, args.length))

    start = time.perf_counter()
    logits = model(tokens)
    loss = logits.float().pow(2).mean()
    loss.backward()
    elapsed = time.perf_counter() - start
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(json.dumps({
        "block": args.block,
        "length": args.length,
        "batch": args.batch,
        "parameters": count_parameters(model),
        "seconds": round(elapsed, 4),
        "baseline_rss_kb": baseline_kb,
        "peak_rss_kb": peak_kb,
        "activation_rss_kb": max(0, peak_kb - baseline_kb),
        "finite": bool(torch.isfinite(loss).item()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
