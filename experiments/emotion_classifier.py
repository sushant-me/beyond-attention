"""Train an emotion classifier on synthesised prosody and measure it honestly.

The operator asked for a system that understands emotion -- crying, excited, and
similar. The previous increment produced prosodic *descriptors* and said
explicitly that no classifier had been trained. This closes that gap and reports
what it does and does not establish.

The experiment, in order:

1. **Build the conditions.** Five conditions whose parameters come from the
   published acoustic correlates of those states (crying/sad-sobbing, excited,
   angry, calm, afraid/anxious), each cited in `emotion.py`. Nothing is tuned
   until after the measurement; the correlate check below is a *test* that the
   generated signals show the profiles the citations describe.
2. **Extract features.** 26 descriptors per utterance: the 15 prosodic
   descriptors from `voice.py` plus 11 new voice-quality features (HNR,
   shimmer, tremor rate, F0 slope and final ratio, pause structure, onset
   sharpness, centroid spread).
3. **Train.** A PyTorch MLP, split by utterance (60/20/20, stratified), fixed
   seed, model selection on the validation split only. The headline number is
   **test** accuracy, read once.
4. **Control it.** Shuffled labels, trained-on-noise features, chance and
   majority baselines, the untrained nearest-centroid rule from the previous
   increment.
5. **Ablate it.** Remove the F0 level/contour family, remove everything
   pitch-derived, remove each feature, keep each feature, and keep each feature
   *family* alone -- the last of which answers "what is left when the model
   cannot see pitch at all?"
6. **Break it.** Train on four conditions and hold the fifth out entirely. The
   held-out label is not in the trained label space, so accuracy is 0 by
   construction; what is reported is where the model puts a condition it has
   never seen.

    python -u experiments/emotion_classifier.py --out emotion-classifier.json

Every number in the README's emotion block is rendered from the JSON this
writes. Nothing here downloads anything or touches the network.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from beyond_attention.emotion import (
    CONDITIONS,
    CORRELATES,
    EMOTION_FEATURES,
    F0_FAMILY,
    FEATURE_FAMILIES,
    PITCH_DERIVED_FAMILY,
    ablation,
    binary_cry_vs_excited,
    build_dataset,
    confusion_matrix,
    correlate_checks,
    greedy_forward_selection,
    grouped_means,
    labels_at,
    leave_one_condition_out,
    majority_accuracy,
    nearest_centroid_accuracy,
    per_feature_cry_vs_excited,
    random_feature_control,
    shuffle_label_control,
    stratified_split,
    train_classifier,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=int, default=40,
                        help="utterances per condition")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--shuffle-rounds", type=int, default=25)
    parser.add_argument("--random-rounds", type=int, default=5)
    parser.add_argument("--out", default="emotion-classifier.json")
    parser.add_argument("--quick", action="store_true",
                        help="tiny configuration, for a smoke test; the numbers "
                             "are not the published ones")
    args = parser.parse_args()

    if args.quick:
        args.utterances = 10
        args.seconds = 0.8
        args.steps = 300
        args.shuffle_rounds = 4
        args.random_rounds = 2

    started = time.perf_counter()
    train_kwargs = {
        "seed": args.seed, "hidden": args.hidden, "steps": args.steps,
    }

    # --- 1. data ----------------------------------------------------------
    dataset = build_dataset(
        utterances=args.utterances, seconds=args.seconds,
        sample_rate=args.sample_rate, seed=args.seed,
    )
    split = stratified_split(dataset.labels, seed=args.seed)
    means = grouped_means(dataset)
    checks = correlate_checks(means)

    # --- 2. the trained classifier ---------------------------------------
    classifier = train_classifier(dataset, split=split, **train_kwargs)
    predictions = classifier.predict(dataset.features[split.test])
    truth = labels_at(dataset.labels, split.test)
    classes = classifier.classes
    confusion = confusion_matrix(predictions, truth, classes)
    chance = 1.0 / len(classes)
    per_class_recall = {
        classes[i]: (
            confusion[i][i] / sum(confusion[i]) if sum(confusion[i]) else 0.0
        )
        for i in range(len(classes))
    }

    # --- 3. controls ------------------------------------------------------
    shuffled = shuffle_label_control(
        dataset, split, rounds=args.shuffle_rounds,
        control_seed=args.seed + 100, **train_kwargs,
    )
    random_features = random_feature_control(
        dataset, split, rounds=args.random_rounds,
        control_seed=args.seed + 200, **train_kwargs,
    )
    centroid_baseline = nearest_centroid_accuracy(dataset, split)

    # --- 4. ablations -----------------------------------------------------
    families = {
        f"only_{name}": {
            "kept": members,
        }
        for name, members in FEATURE_FAMILIES.items()
    }
    families["without_f0_family"] = {"removed": F0_FAMILY}
    families["without_pitch_derived"] = {"removed": PITCH_DERIVED_FAMILY}

    family_results: dict[str, dict[str, object]] = {}
    for label, kwargs in families.items():
        result = ablation(dataset, split, **train_kwargs, **kwargs)
        family_results[label] = {
            "n_features": result["n_features"],
            "test_accuracy": result["test_accuracy"],
            "selected_epoch": result["selected_epoch"],
            "features": result["features"],
        }

    per_feature: dict[str, dict[str, float]] = {}
    for name in EMOTION_FEATURES:
        only = ablation(dataset, split, kept=(name,), **train_kwargs)
        without = ablation(dataset, split, removed=(name,), **train_kwargs)
        per_feature[name] = {
            "only_accuracy": float(only["test_accuracy"]),
            "without_accuracy": float(without["test_accuracy"]),
            "drop": float(classifier.test_accuracy - without["test_accuracy"]),
        }

    # --- 5. crying against excited ---------------------------------------
    cry_vs_excited = binary_cry_vs_excited(
        dataset, seed=args.seed, steps=args.steps, hidden=args.hidden,
    )
    cry_vs_excited_features = per_feature_cry_vs_excited(
        dataset, seed=args.seed, steps=args.steps, hidden=args.hidden,
    )
    # The single most discriminative feature and the damage when it is removed,
    # which is the ablation the README's cry-versus-excited claim rests on.
    ranked = sorted(
        cry_vs_excited_features.items(),
        key=lambda kv: -kv[1]["single_accuracy"],
    )
    top_feature, top_stats = ranked[0]
    minimal = greedy_forward_selection(
        dataset, seed=args.seed, steps=args.steps, hidden=args.hidden,
    )

    # --- 6. cross-condition generalisation --------------------------------
    cross = leave_one_condition_out(
        dataset, seed=args.seed, steps=args.steps, hidden=args.hidden,
    )

    conditions_payload = {
        name: {
            "parameters": asdict(next(c for c in CONDITIONS if c.name == name)),
            "correlates": list(CORRELATES[name]),
            "descriptors": means[name],
        }
        for name in dataset.conditions
    }

    payload = {
        "config": {
            "sample_rate": args.sample_rate,
            "duration_s": args.seconds,
            "utterances_per_condition": args.utterances,
            "seed": args.seed,
            "quick": bool(args.quick),
            "n_conditions": len(dataset.conditions),
            "n_utterances": dataset.n_utterances,
            "n_features": dataset.n_features,
            "feature_names": list(EMOTION_FEATURES),
            "feature_families": {
                name: list(members)
                for name, members in FEATURE_FAMILIES.items()
            },
            "split": {
                "unit": "utterance",
                "fractions": [0.6, 0.2, 0.2],
                "train": int(len(split.train)),
                "val": int(len(split.val)),
                "test": int(len(split.test)),
            },
            "classifier": classifier.architecture,
            "chance": chance,
        },
        "conditions": conditions_payload,
        "correlate_checks": {
            "checks": checks,
            "all_pass": bool(all(checks.values())),
        },
        "training": {
            "selected_epoch": classifier.selected_epoch,
            "train_accuracy": classifier.train_accuracy,
            "val_accuracy": classifier.val_accuracy,
            "test_accuracy": classifier.test_accuracy,
            "chance": chance,
            "majority_baseline": majority_accuracy(dataset, split),
            "nearest_centroid_baseline": centroid_baseline["test_accuracy"],
            "n_test": int(len(split.test)),
            "z_vs_chance": (
                (classifier.test_accuracy - chance)
                / float(np.sqrt(chance * (1.0 - chance) / len(split.test)))
            ),
            "confusion": confusion,
            "classes": classes,
            "per_class_recall": per_class_recall,
            "history": classifier.history,
        },
        "controls": {
            "label_shuffle": shuffled,
            "random_features": random_features,
            "nearest_centroid_confusion": centroid_baseline["confusion"],
        },
        "ablations": family_results,
        "per_feature": per_feature,
        "crying_vs_excited": {
            **cry_vs_excited,
            "per_feature": cry_vs_excited_features,
            "ranked": [name for name, _ in ranked],
            "top_feature": top_feature,
            "top_single_accuracy": float(top_stats["single_accuracy"]),
            "top_without_accuracy": float(top_stats["without_accuracy"]),
            "top_drop": float(top_stats["drop"]),
            "greedy_forward": minimal,
        },
        "cross_condition": cross,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }

    Path(args.out).write_text(json.dumps(payload, indent=2))

    # --- human-readable summary ------------------------------------------
    failed = [name for name, ok in checks.items() if not ok]
    print(f"{dataset.n_utterances} utterances "
          f"({args.utterances} per condition, {args.seconds}s each), "
          f"{dataset.n_features} features")
    print(f"split: {len(split.train)} train / {len(split.val)} val / "
          f"{len(split.test)} test, by utterance")
    print(f"correlate checks: {sum(checks.values())}/{len(checks)} pass"
          + (f" -- FAILED {failed}" if failed else ""))
    print()
    print(f"published held-out accuracy: {classifier.test_accuracy:.3f} "
          f"(chance {chance:.3f}, majority "
          f"{majority_accuracy(dataset, split):.3f}, nearest centroid "
          f"{centroid_baseline['test_accuracy']:.3f})")
    print(f"confusion (rows true, columns predicted), classes {classes}:")
    for name, row in zip(classes, confusion):
        print(f"  {name:<9}{row}")
    print()
    print(f"controls: shuffled labels mean {shuffled['mean']:.3f} "
          f"(p95 {shuffled['p95']:.3f}, max {shuffled['max']:.3f}) over "
          f"{int(shuffled['rounds'])} rounds; "
          f"random features mean {random_features['mean']:.3f} "
          f"(max {random_features['max']:.3f})")
    print()
    print("ablations:")
    for label, result in family_results.items():
        print(f"  {label:<24} n={result['n_features']:>2} "
              f"test={result['test_accuracy']:.3f}")
    print()
    print(f"crying vs excited: {cry_vs_excited['test_accuracy']:.3f} on "
          f"{cry_vs_excited['n_test']} utterances, confusion "
          f"{cry_vs_excited['confusion']}, recall {cry_vs_excited['recall']}")
    print(f"  best single feature `{top_feature}` {top_stats['single_accuracy']:.3f}; "
          f"without it {top_stats['without_accuracy']:.3f} "
          f"(drop {top_stats['drop']:+.3f})")
    print(f"  greedy forward selection: {minimal['n_features_needed']} feature(s) "
          f"{minimal['features']}, validation "
          f"{minimal['final_val_accuracy']:.3f}, test "
          f"{minimal['final_test_accuracy']:.3f} "
          f"({minimal['stopped_because']})")
    print()
    print("cross-condition (train on four, look at the fifth):")
    for name, entry in cross["per_condition"].items():
        print(f"  hold out {name:<9} -> {entry['dominant_assignment']:<9} "
              f"{entry['dominant_fraction']:.2f} of rows, "
              f"nll {entry['assigned_class_nll']:.2f}, "
              f"distance {entry['distance_to_nearest_training_centroid']:.2f} "
              f"spreads")
    print(f"  mean dominant fraction {cross['mean_dominant_fraction']:.3f} "
          f"(one training class would be "
          f"{cross['chance_for_one_training_class']:.2f}); the MLP's dominant "
          f"class agrees with nearest-centroid on "
          f"{cross['mlp_agrees_with_nearest_centroid']:.2f} of conditions")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
