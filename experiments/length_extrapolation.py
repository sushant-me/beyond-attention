"""Train at one length, then ask at lengths it never saw.

Every other experiment here trains and evaluates at the same sequence length.
That is the easy question. The hard one, and the one that decides whether a
model can be pointed at a document longer than anything it was trained on, is
what happens past the training length.

The setup: train both architectures at **2 pairs** (5 tokens), then evaluate the
same frozen weights at 4, 8 and 16 pairs -- 10, 18 and 34 tokens, up to **5.7x**
the training length. Nothing is retrained between rows.

2 pairs is not an arbitrary choice. It is the longest training length at which
*both* architectures solve the task: attention reaches 1.000 at 2 pairs and
stalls around 0.5 at 4, so training longer would measure extrapolation for one
architecture and failure-to-fit for the other.

Two things make this a fair test rather than a demonstration:

* **The key space is fixed.** `mqar_batch` sizes its vocabulary from the pair
  count by default, so evaluating a model trained at 8 pairs on a 128-pair batch
  would index outside its embedding -- a crash, not a number. Passing an
  explicit `n_keys` holds the vocabulary still and lets length be the only
  variable. `tests/test_tasks.py` pins this.

* **A trained-at-that-length reference.** Longer MQAR sequences are also *harder*
  -- there are more associations to store and the same fixed-size model must
  hold them -- so a falling curve conflates "cannot extrapolate" with "cannot do
  the task at this size at all". The reference column trains a fresh model at
  each evaluation length, with the same budget, and reports what was achievable
  there. The gap between a model's extrapolated score and that reference is the
  part attributable to extrapolation; without it the curve is not interpretable.

What this does **not** establish: that either architecture generalises in
general. It is one synthetic recall task, two layers, `d_model = 64`, and a
single training length. A model that fails here failed *this*, at *this* size.

Usage:
    python -u experiments/length_extrapolation.py --out length-extrapolation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from beyond_attention.model import LanguageModel, count_parameters
from beyond_attention.tasks import vocabulary_size
from beyond_attention.train import evaluate_mqar, train

BLOCK_KWARGS = {
    "ssm": {"d_state": 16, "expand": 2, "conv_kernel": 4},
    "attention": {"n_heads": 4, "mlp_ratio": 2.0},
}
BLOCKS = ("attention", "ssm")

# Training short and evaluating long is what buys the extrapolation ratio. The
# key space must cover the *longest* evaluation, so the vocabulary is driven by
# the eval length, not the training length -- which is why training at 4 pairs
# and evaluating at 32 costs no more vocabulary than the reverse, and is a much
# easier task to learn in the first place.
# 2 pairs is the longest training length at which BOTH architectures solve the
# task outright (attention reaches 1.000 at 2 pairs and stalls near 0.5 at 4),
# so it is the longest length from which extrapolation can be measured for both
# rather than only for the one that learned.
TRAIN_PAIRS = 2
# 2 -> 16 pairs is 6 -> 34 tokens, a 5.7x extrapolation.
EVAL_PAIRS = (2, 4, 8, 16)
# The vocabulary is vocabulary_size(16) = 33 and it is the same for every row.
N_KEYS = 16


def sequence_length(n_pairs: int, n_queries: int = 1) -> int:
    return 2 * n_pairs + 1 + n_queries


def build(block: str, vocab: int, d_model: int, n_layers: int) -> LanguageModel:
    return LanguageModel(
        vocab, d_model, n_layers, block, **BLOCK_KWARGS[block]
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def spread(xs: list[float]) -> float:
    return (max(xs) - min(xs)) / 2 if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="length-extrapolation.json")
    ap.add_argument("--train-pairs", type=int, default=TRAIN_PAIRS)
    ap.add_argument("--eval-pairs", type=int, nargs="*", default=list(EVAL_PAIRS))
    ap.add_argument("--n-keys", type=int, default=N_KEYS)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reference-seeds", type=int, nargs="+", default=[0],
                    help="seeds for the trained-at-that-length reference")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the reference models (much faster, but then a "
                         "falling curve cannot be attributed to extrapolation)")
    ap.add_argument("--reference-pairs", type=int, nargs="*", default=None,
                    help="lengths to train reference models at; defaults to the "
                         "shortest and longest eval length (the two that bound "
                         "the interpretation)")
    args = ap.parse_args()
    if args.no_reference:
        args.reference_seeds = []
    if args.reference_pairs is None:
        args.reference_pairs = sorted({min(args.eval_pairs), max(args.eval_pairs)})

    if args.n_keys < max(args.eval_pairs):
        raise SystemExit(
            f"n_keys={args.n_keys} < max eval pairs {max(args.eval_pairs)}: a "
            f"sequence cannot hold more distinct keys than the space contains"
        )
    if args.train_pairs not in args.eval_pairs:
        raise SystemExit("--train-pairs should be one of --eval-pairs")

    vocab = vocabulary_size(args.n_keys)
    print("=" * 78)
    print("LENGTH EXTRAPOLATION  train at one length, evaluate past it")
    print("=" * 78)
    print(f"vocabulary      : {vocab} (fixed via n_keys={args.n_keys})")
    print(f"train length    : {args.train_pairs} pairs = "
          f"{sequence_length(args.train_pairs)} tokens")
    print(f"eval lengths    : "
          + ", ".join(f"{p}={sequence_length(p)}" for p in args.eval_pairs))
    print(f"steps / seeds   : {args.steps} / {args.seeds}")
    print("nothing is retrained between rows; these are frozen weights")
    print()

    results: dict = {
        "config": {
            "vocab": vocab, "n_keys": args.n_keys,
            "train_pairs": args.train_pairs,
            "train_length": sequence_length(args.train_pairs),
            "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "d_model": args.d_model, "n_layers": args.n_layers,
            "seeds": args.seeds,
        },
        "extrapolated": {},
        "reference": {},
        "parameters": {},
    }

    out_path = Path(args.out)
    long_names = {p: sequence_length(p) for p in args.eval_pairs}

    for block in BLOCKS:
        per_length: dict[int, list[float]] = {p: [] for p in args.eval_pairs}
        for seed in args.seeds:
            model = build(block, vocab, args.d_model, args.n_layers)
            results["parameters"][block] = count_parameters(model)
            trained = train(
                model, block, n_pairs=args.train_pairs, steps=args.steps,
                batch_size=args.batch_size, lr=args.lr, seed=seed,
                n_keys=args.n_keys,
            )
            for p in args.eval_pairs:
                # Same weights, different length.
                per_length[p].append(
                    evaluate_mqar(model, p, batch_size=256, seed=seed + 100,
                                  n_keys=args.n_keys)
                )
            print(f"  {block:<10} seed {seed}: trained "
                  f"acc@train={trained.train_accuracy:.3f}")
        results["extrapolated"][block] = {
            str(p): {"mean": mean(v), "spread": spread(v), "per_seed": v}
            for p, v in per_length.items()
        }

    if args.reference_seeds:
        print()
        print(f"reference: a fresh model trained at {args.reference_pairs} "
              f"(seeds {args.reference_seeds})")
        for block in BLOCKS:
            ref: dict[int, list[float]] = {p: [] for p in args.reference_pairs}
            for seed in args.reference_seeds:
                for p in args.reference_pairs:
                    model = build(block, vocab, args.d_model, args.n_layers)
                    train(
                        model, block, n_pairs=p, steps=args.steps,
                        batch_size=args.batch_size, lr=args.lr, seed=seed,
                        n_keys=args.n_keys,
                    )
                    ref[p].append(
                        evaluate_mqar(model, p, batch_size=256,
                                      seed=seed + 100, n_keys=args.n_keys)
                    )
            results["reference"][block] = {
                str(p): {"mean": mean(v), "spread": spread(v), "per_seed": v}
                for p, v in ref.items()
            }
            print(f"  {block:<10} reference done")

    print()
    print("=" * 78)
    print("exact-match accuracy, weights frozen after training at "
          f"{args.train_pairs} pairs")
    print("=" * 78)
    header = (f"{'pairs':>6} {'tokens':>7} {'x train':>8} "
              f"{'attention':>18} {'ssm':>18} {'chance':>8}")
    if args.reference_seeds:
        header += f" {'ref attn':>9} {'ref ssm':>9}"
    print(header)
    for p in args.eval_pairs:
        ratio = sequence_length(p) / sequence_length(args.train_pairs)
        row = f"{p:>6} {long_names[p]:>7} {ratio:>7.1f}x "
        for block in BLOCKS:
            e = results["extrapolated"][block][str(p)]
            cell = f"{e['mean']:.3f}"
            if e["spread"]:
                cell += f" ±{e['spread']:.3f}"
            row += f"{cell:>18} "
        row += f"{1.0 / args.n_keys:>8.4f}"
        if args.reference_seeds:
            for block in BLOCKS:
                entry = results["reference"][block].get(str(p))
                row += (f"{entry['mean']:>9.3f}" if entry else f"{'-':>9}")
        print(row)

    print()
    print(f"chance = 1/{args.n_keys} (uniform over the value tokens)")
    results["chance"] = 1.0 / args.n_keys
    results["note"] = (
        "Longer MQAR sequences are harder as well as longer, so 'ref' columns "
        "give what a model trained at that length reached; the gap between a "
        "model's extrapolated score and its reference is the part attributable "
        "to extrapolation."
    )
    print(results["note"])

    # Structural checks only. Whether either architecture holds up is the
    # measurement, so nothing here asserts which one wins.
    assert all(
        0.0 <= v["mean"] <= 1.0
        for block in results["extrapolated"].values()
        for v in block.values()
    ), "accuracy outside [0, 1]"

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
