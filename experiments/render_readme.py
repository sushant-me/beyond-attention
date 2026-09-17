"""Render the results tables from `results.json` into the README.

The numbers in the README are generated from the measurement file rather than
typed in, because a hand-copied table drifts from its run and there is no way for
a reader to tell. Re-running the experiment and this script is the whole
reproduction procedure.

    python experiments/run.py --out results.json
    python experiments/render_readme.py results.json README.md
"""

from __future__ import annotations

import json
import pathlib
import sys

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def mqar_table(payload: dict) -> str:
    rows = payload.get("mqar", {})
    if not rows:
        return "_no sweep results_"
    pair_counts = sorted({v["n_pairs"] for v in rows.values()})
    lines = [
        "| pairs in context | model | parameters | accuracy (exact match) | "
        "chance | best off-size accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for n_pairs in pair_counts:
        for block in ("attention", "ssm"):
            row = rows.get(f"{block}@{n_pairs}")
            if row is None:
                continue
            runs = row["accuracy_runs"]
            spread = f"{row['accuracy_mean']:.3f}"
            if len(runs) > 1:
                spread += f" ± {(max(runs) - min(runs)) / 2:.3f}"
            off = row.get("off_size_accuracy") or {}
            if off:
                best = max(off, key=off.get)
                off_text = f"{off[best]:.3f} (at {best})"
            else:
                off_text = "—"
            lines.append(
                f"| {n_pairs} | {block} | {row['parameters']:,} | {spread} | "
                f"{row['chance']:.3f} | {off_text} |"
            )
    return "\n".join(lines)


def scaling_table(payload: dict) -> str:
    rows = payload.get("scaling", {})
    if not rows:
        return "_no scaling results_"
    lengths = sorted({v["length"] for v in rows.values()})
    lines = [
        "| sequence length | model | forward+backward (s) | "
        "activation memory (MB) | MB per token |",
        "|---|---|---|---|---|",
    ]
    for length in lengths:
        for block in ("attention", "ssm"):
            row = rows.get(f"{block}@{length}")
            if row is None:
                continue
            mb = row["activation_rss_kb"] / 1024
            per_token = row["activation_rss_kb"] / 1024 / (length * row["batch"])
            lines.append(
                f"| {length} | {block} | {row['seconds']:.3f} | {mb:.1f} | "
                f"{per_token:.4f} |"
            )
    return "\n".join(lines)


def render(payload: dict) -> str:
    config = payload.get("config", {})
    counts = payload.get("parameter_check_vocab", {})
    body = [
        mqar_table(payload),
        "",
        scaling_table(payload),
        "",
        "```",
        f"config: d_model={config.get('d_model')}, layers={config.get('n_layers')}, "
        f"steps={config.get('steps')}, seeds={config.get('seeds')}, "
        f"batch={config.get('batch_size')}, lr={config.get('lr')}",
        f"parameter check at the sweep vocabulary: ssm={counts.get('ssm'):,}, "
        f"attention={counts.get('attention'):,}",
        f"total wall time: {payload.get('wall_seconds')}s",
        "```",
    ]
    return "\n".join(body)


def main() -> int:
    results = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results.json")
    readme = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "README.md")
    payload = json.loads(results.read_text())

    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print(f"README has no {BEGIN} / {END} block; add one first", file=sys.stderr)
        return 1
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    rendered = render(payload)

    # Refuse to publish a table that lost its content, which is what a broken
    # renderer produces and what a reader cannot detect.
    if "|" not in rendered or len(rendered.splitlines()) < 5:
        print("refusing to write: the rendered block is empty or malformed",
              file=sys.stderr)
        return 1

    readme.write_text(f"{head}{BEGIN}\n{rendered}\n{END}{tail}")
    print(f"wrote {readme} from {results} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
