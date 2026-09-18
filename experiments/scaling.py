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

from beyond_attention.measurement import peak_rss_during
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
    parser.add_argument("--causal-mode", choices=["mask", "sdpa"],
                        default="mask")
    parser.add_argument("--scan-inner", choices=["loop", "vectorized"],
                        default="loop")
    parser.add_argument("--scan-chunk", type=int, default=64)
    parser.add_argument("--compile", action="store_true",
                        help="wrap the model in torch.compile")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    kwargs = (
        {"d_state": 16, "expand": 2, "conv_kernel": 4,
         "scan_inner": args.scan_inner, "scan_chunk": args.scan_chunk}
        if args.block == "ssm"
        else {"n_heads": 4, "mlp_ratio": 2.0,
              "causal_mode": args.causal_mode}
    )
    model = LanguageModel(
        args.vocab, args.d_model, args.n_layers, args.block, **kwargs
    )
    tokens = torch.randint(0, args.vocab, (args.batch, args.length))
    if args.compile:
        model = torch.compile(model, dynamic=False)
        # A compiled model's first call is a compile, not a measurement, so it
        # is warmed up outside the timed region.
        model(tokens).float().pow(2).mean().backward()
        model.zero_grad(set_to_none=True)

    def run():
        logits = model(tokens)
        loss = logits.float().pow(2).mean()
        loss.backward()
        return loss

    # Memory is measured by sampling *current* RSS while the call runs, not by
    # subtracting `ru_maxrss` marks. `ru_maxrss` is reported out of
    # `signal_struct`, which a forked child inherits from its parent: a child of
    # this driver starts with the driver's own peak already recorded, so
    # anything smaller is invisible and the measurement reads zero. The first
    # version of this probe did exactly that and printed the same constant for
    # every configuration, which looks like a result and is not one.
    start = time.perf_counter()
    loss, activation_kb = peak_rss_during(run)
    elapsed = time.perf_counter() - start

    print(json.dumps({
        "block": args.block,
        "causal_mode": args.causal_mode,
        "scan_inner": args.scan_inner,
        "scan_chunk": args.scan_chunk,
        "compiled": args.compile,
        "length": args.length,
        "batch": args.batch,
        "parameters": count_parameters(model),
        "seconds": round(elapsed, 4),
        "activation_rss_kb": activation_kb,
        "finite": bool(torch.isfinite(loss).item()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
