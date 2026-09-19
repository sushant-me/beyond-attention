"""A second task, chosen to be the opposite test of the same property.

The main sweep uses MQAR, where the answer requires holding *every* association
in the sequence, so the memory needed grows with length and a fixed-size
recurrent state is a handicap. That is the honest finding it reports, and it is
also a finding about one task.

This runs the complementary task. A single bit is toggled by some tokens and
queried at random positions; the model must report its value. The state required
is **one bit, at any length** -- nothing about the answer depends on the history
beyond a parity -- so a fixed-size state is sufficient by construction, while an
attention model must still pool over the whole prefix to compute it.

Two controls are built in, and both exist because this repository has already
published one headline that a control falsified:

* **A step-budget control for the baseline.** The MQAR section reports an
  apparent architectural advantage that vanished when the Transformer was given
  20,000 steps instead of 3,000. The same trap is available here, so the
  attention model is trained at a much larger budget and reported next to the
  matched one. If the gap closes, the gap was the budget.
* **A trained-at-that-length reference.** A model evaluated beyond its training
  length may fail because it cannot generalise or because the task is hard at
  that length for this size. Training a fresh model at the longest evaluated
  length separates the two.

What this does not establish: that either architecture tracks state in general.
It is one synthetic task, two layers, `d_model = 64`.

Usage:
    python -u experiments/state_tracking.py --out state-tracking.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from beyond_attention.model import LanguageModel, count_parameters
from beyond_attention.tasks import REGISTER_VOCAB, register_batch, register_chance
from beyond_attention.train import evaluate, train

BLOCK_KWARGS = {
    "ssm": {"d_state": 16, "expand": 2, "conv_kernel": 4},
    "attention": {"n_heads": 4, "mlp_ratio": 2.0},
}
BLOCKS = ("attention", "ssm")

TRAIN_LENGTH = 32
EVAL_LENGTHS = (32, 64, 128)
QUERIES_PER_SEQUENCE = 4


def sampler(length: int):
    """A Batch sampler for the register task at one sequence length."""
    def sample(batch_size: int, generator: torch.Generator, device: str):
        return register_batch(
            batch_size, length, QUERIES_PER_SEQUENCE, generator, device
        )
    return sample


def build(block: str, d_model: int, n_layers: int) -> LanguageModel:
    return LanguageModel(
        REGISTER_VOCAB, d_model, n_layers, block, **BLOCK_KWARGS[block]
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def spread(xs: list[float]) -> float:
    return (max(xs) - min(xs)) / 2 if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="state-tracking.json")
    ap.add_argument("--train-length", type=int, default=TRAIN_LENGTH)
    ap.add_argument("--eval-lengths", type=int, nargs="*",
                    default=list(EVAL_LENGTHS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--control-steps", type=int, nargs="+", default=[8000],
                    help="extra attention budgets, to test whether a gap is "
                         "the budget rather than the architecture")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--reference-seeds", type=int, nargs="+", default=None,
                    help="seeds for the reference; defaults to --seeds")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the trained-at-the-longest-length reference")
    args = ap.parse_args()

    if args.reference_seeds is None:
        args.reference_seeds = list(args.seeds)
    if args.train_length not in args.eval_lengths:
        raise SystemExit("--train-length should be one of --eval-lengths")

    chance = register_chance()
    print("=" * 78)
    print("STATE TRACKING  one bit, queried at random positions")
    print("=" * 78)
    print(f"task            : register toggled by some tokens, read at queries")
    print(f"state required  : 1 bit, independent of length")
    print(f"train length    : {args.train_length} tokens, "
          f"{QUERIES_PER_SEQUENCE} queries per sequence")
    print(f"eval lengths    : {args.eval_lengths}")
    print(f"steps / seeds   : {args.steps} / {args.seeds}")
    print(f"chance          : {chance}, conditional on emitting an answer token")
    print(f"                  accuracy is argmax over the whole vocabulary, so a")
    print(f"                  model that has not learned the response format")
    print(f"                  scores near 0, not near {chance}. Below-chance means")
    print(f"                  'has not learned the format', not 'worse than luck'.")
    print("nothing is retrained between rows; these are frozen weights")
    print()

    results: dict = {
        "config": {
            "task": "register", "train_length": args.train_length,
            "eval_lengths": args.eval_lengths, "steps": args.steps,
            "queries_per_sequence": QUERIES_PER_SEQUENCE,
            "d_model": args.d_model, "n_layers": args.n_layers,
            "seeds": args.seeds, "chance": chance,
        },
        "matched_budget": {},
        "control": {},
        "reference": {},
        "parameters": {},
    }
    out_path = Path(args.out)

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2) + "\n")

    models: dict[str, list[LanguageModel]] = {b: [] for b in BLOCKS}
    per_length: dict[str, dict[int, list[float]]] = {
        b: {n: [] for n in args.eval_lengths} for b in BLOCKS
    }

    for block in BLOCKS:
        for seed in args.seeds:
            model = build(block, args.d_model, args.n_layers)
            results["parameters"][block] = count_parameters(model)
            trained = train(
                model, block, n_pairs=1, steps=args.steps,
                batch_size=args.batch_size, lr=args.lr, seed=seed,
                sample_batch=sampler(args.train_length),
            )
            models[block].append(model)
            for n in args.eval_lengths:
                per_length[block][n].append(
                    evaluate(model, sampler(n), batch_size=256, seed=seed + 100)
                )
            print(f"  {block:<10} seed {seed}: train-length acc "
                  f"{trained.train_accuracy:.3f}")
            save()

    results["matched_budget"] = {
        b: {str(n): {"mean": mean(v), "spread": spread(v), "per_seed": v}
            for n, v in per_length[b].items()}
        for b in BLOCKS
    }

    # The control. Both architectures, same everything, more steps.
    #
    # Controlling only the baseline would be half a control: it can show the
    # baseline was under-trained, but not that the other model was not. The
    # first version of this experiment made exactly that mistake, and the
    # one-sided control made the recurrent model look better than it was.
    for steps in args.control_steps:
        print()
        print(f"control: both architectures at {steps} steps "
              f"(matched budget was {args.steps})")
        for block in BLOCKS:
            accs: dict[int, list[float]] = {n: [] for n in args.eval_lengths}
            for seed in args.seeds:
                model = build(block, args.d_model, args.n_layers)
                train(model, block, n_pairs=1, steps=steps,
                      batch_size=args.batch_size, lr=args.lr, seed=seed,
                      sample_batch=sampler(args.train_length))
                for n in args.eval_lengths:
                    accs[n].append(
                        evaluate(model, sampler(n), batch_size=256,
                                 seed=seed + 100)
                    )
            results["control"][f"{block}@{steps}"] = {
                str(n): {"mean": mean(v), "spread": spread(v), "per_seed": v}
                for n, v in accs.items()
            }
            print(f"  {block:<10} @{steps}: "
                  f"{results['control'][f'{block}@{steps}'][str(args.train_length)]['mean']:.3f} "
                  f"at the training length")
            save()

    if not args.no_reference:
        longest = max(args.eval_lengths)
        print()
        print(f"reference: a fresh model trained at {longest} tokens")
        for block in BLOCKS:
            accs: list[float] = []
            for seed in args.reference_seeds:
                model = build(block, args.d_model, args.n_layers)
                train(model, block, n_pairs=1, steps=args.steps,
                      batch_size=args.batch_size, lr=args.lr, seed=seed,
                      sample_batch=sampler(longest))
                accs.append(
                    evaluate(model, sampler(longest), batch_size=256,
                             seed=seed + 100)
                )
            results["reference"][block] = {
                "length": longest, "mean": mean(accs),
                "spread": spread(accs), "per_seed": accs,
            }
            print(f"  {block:<10} trained at {longest}: {mean(accs):.3f}")
            save()

    print()
    print("=" * 78)
    print(f"exact-match accuracy, weights frozen after training at "
          f"{args.train_length} tokens")
    print("=" * 78)
    control_keys = [f"{b}@{s}" for s in args.control_steps for b in BLOCKS]
    w = 16
    header = f"{'length':>8} {'x train':>9}"
    for b in BLOCKS:
        header += f"{b:>{w}}"
    for k in control_keys:
        header += f"{k:>{w}}"
    header += f"{'chance*':>9}"
    if not args.no_reference:
        for b in BLOCKS:
            header += f"{'ref ' + b:>12}"
    print(header)
    for n in args.eval_lengths:
        row = f"{n:>8} {n / args.train_length:>8.1f}x"
        for b in BLOCKS:
            e = results["matched_budget"][b][str(n)]
            cell = f"{e['mean']:.3f}" + (f"±{e['spread']:.3f}"
                                         if e["spread"] else "")
            row += f"{cell:>{w}}"
        for k in control_keys:
            e = results["control"].get(k, {}).get(str(n))
            if e is None:
                row += f"{'-':>{w}}"
                continue
            cell = f"{e['mean']:.3f}" + (f"±{e['spread']:.3f}"
                                         if e["spread"] else "")
            row += f"{cell:>{w}}"
        row += f"{chance:>9.3f}"
        if not args.no_reference:
            for b in BLOCKS:
                ref = results["reference"][b]
                row += (f"{ref['mean']:>{w}.3f}" if ref["length"] == n
                        else f"{'-':>{w}}")
        print(row)

    print()
    print("chance* = 1/2 conditional on the model emitting an answer token; see")
    print("the note above for why a row can sit below it.")
    print("Reading it: if the control closes the gap, the gap was the budget.")
    print("If the reference at the longest length is also low, the number is")
    print("task difficulty at that length, not a failure to generalise.")

    assert all(
        0.0 <= v["mean"] <= 1.0
        for group in (results["matched_budget"],)
        for block in group.values()
        for v in block.values()
    ), "accuracy outside [0, 1]"

    save()
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
