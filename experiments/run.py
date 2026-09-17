"""Run the comparison and print it as tables.

Two experiments, answering two different questions:

1. **MQAR sweep** — can the model retrieve an association it has stored? Both
   architectures are trained on the same number of pairs, with the same step
   budget, batch size, learning rate and seed, and the parameter counts are
   printed so the reader can check the budgets match. Each trained model is also
   evaluated at pair counts it never trained on.

2. **Length scaling** — what does a sequence cost? Time and peak memory for a
   forward and backward pass as length grows, one fresh process per point. This
   one needs no training and no interpretation: attention is quadratic in
   length and a state-space model is linear, and the measurement either shows
   that or it does not.

Both are printed as they complete and written to `results.json`, so the numbers
in the README can be regenerated rather than trusted.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

from beyond_attention.model import LanguageModel, build_pair, count_parameters
from beyond_attention.tasks import vocabulary_for
from beyond_attention.train import _evaluate, train

BLOCK_KWARGS = {
    "ssm": {"d_state": 16, "expand": 2, "conv_kernel": 4},
    "attention": {"n_heads": 4, "mlp_ratio": 2.0},
}


def mqar_sweep(
    pair_counts: list[int],
    steps: int,
    seeds: list[int],
    d_model: int,
    n_layers: int,
    batch_size: int,
    lr: float,
    n_queries: int = 1,
    blocks: tuple[str, ...] = ("attention", "ssm"),
) -> dict:
    vocab = max(vocabulary_for(p) for p in pair_counts)
    print(f"\n## MQAR sweep — d_model={d_model}, layers={n_layers}, "
          f"steps={steps}, batch={batch_size}, lr={lr}, seeds={seeds}, "
          f"scored queries per sequence={n_queries}")
    print(f"vocabulary={vocab}, key/value tokens disjoint, accuracy is exact match")
    print(f"{'task':<10} {'model':<12} {'params':>9} {'train acc':>10} "
          f"{'best off-size':>14} {'chance':>7}")

    results: dict[str, dict] = {}
    for n_pairs in pair_counts:
        for block in blocks:
            accs, params, off_by_size = [], 0, {}
            others = [p for p in pair_counts if p != n_pairs]
            for seed in seeds:
                model = LanguageModel(
                    vocab, d_model, n_layers, block, **BLOCK_KWARGS[block]
                )
                params = count_parameters(model)
                result = train(
                    model, block, n_pairs=n_pairs, steps=steps,
                    batch_size=batch_size, lr=lr, seed=seed,
                    n_train_queries=n_queries,
                )
                accs.append(result.train_accuracy)
                # Evaluate the trained model at pair counts it never saw. Fresh
                # batches, frozen weights: this measures the task, not the batch.
                generator = torch.Generator().manual_seed(seed + 1)
                for other in others:
                    off_by_size.setdefault(other, []).append(
                        _evaluate(model, other, 1, 256, generator, "cpu")
                    )
            mean = sum(accs) / len(accs)
            chance = 1.0 / max(n_pairs, 1)
            spread = f"{mean:.3f}" + (
                f" ±{(max(accs) - min(accs)) / 2:.3f}" if len(accs) > 1 else ""
            )
            off_mean = {p: sum(v) / len(v) for p, v in off_by_size.items()}
            if off_mean:
                best = max(off_mean, key=off_mean.get)
                off = f"{off_mean[best]:.3f} @{best}"
            else:
                off = "—"
            print(f"{n_pairs:<10} {block:<12} {params:>9,} {spread:>10} "
                  f"{off:>14} {chance:>7.3f}")
            results[f"{block}@{n_pairs}"] = {
                "block": block, "n_pairs": n_pairs, "parameters": params,
                "accuracy_mean": mean, "accuracy_runs": accs,
                "chance": chance, "off_size_accuracy": off_mean,
            }
    return results


def length_scaling(
    lengths: list[int], batch: int, d_model: int, n_layers: int
) -> dict:
    print(f"\n## Length scaling — forward+backward, batch={batch}, "
          f"d_model={d_model}, layers={n_layers}, one process per point")
    print("activation memory is peak RSS growth over the post-import baseline,")
    print("so it excludes the ~500 MB that importing torch costs by itself.")
    print(f"{'length':>7} {'model':<11} {'seconds':>9} {'activ MB':>12} "
          f"{'MB / token':>11}")
    results: dict[str, dict] = {}
    for length in lengths:
        for block in ("attention", "ssm"):
            cpu = min(4, len(os.sched_getaffinity(0)))
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).parent / "scaling.py"),
                 "--block", block, "--length", str(length), "--batch", str(batch),
                 "--d-model", str(d_model), "--n-layers", str(n_layers),
                 "--threads", str(cpu)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"{length:>7} {block:<11} FAILED: {proc.stderr.strip()[:80]}")
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            activation_mb = row["activation_rss_kb"] / 1024
            per_token = row["activation_rss_kb"] / 1024 / (length * batch)
            print(f"{length:>7} {block:<11} {row['seconds']:>9.3f} "
                  f"{activation_mb:>12.1f} {per_token:>11.4f}")
            results[f"{block}@{length}"] = row
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--queries", type=int, default=1,
                        help="scored lookups per training sequence")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024])
    parser.add_argument("--scaling-batch", type=int, default=8)
    parser.add_argument("--blocks", nargs="+", default=["attention", "ssm"],
                        choices=["attention", "ssm"])
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))
    started = time.time()
    payload: dict = {"config": vars(args)}

    # Print the matched budgets once, up front, so the claim is visible rather
    # than buried in the numbers.
    ssm, attention = build_pair(max(vocabulary_for(p) for p in args.pairs))
    payload["parameter_check_vocab"] = {
        "ssm": count_parameters(ssm),
        "attention": count_parameters(attention),
    }
    print(f"parameter check at vocab={max(vocabulary_for(p) for p in args.pairs)}: "
          f"ssm={count_parameters(ssm):,} attention={count_parameters(attention):,}")

    if not args.skip_sweep:
        payload["mqar"] = mqar_sweep(
            args.pairs, args.steps, args.seeds, args.d_model, args.n_layers,
            args.batch_size, args.lr, args.queries, tuple(args.blocks),
        )
    if not args.skip_scaling:
        payload["scaling"] = length_scaling(
            args.lengths, args.scaling_batch, args.d_model, args.n_layers
        )

    payload["wall_seconds"] = round(time.time() - started, 1)
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
