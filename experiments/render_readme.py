"""Render the results tables from the measurement files into the README.

The numbers in the README are generated from the JSON the experiments write
rather than typed in, because a hand-copied table drifts from its run and a
reader has no way to tell. Re-running the experiments and this script is the
whole reproduction procedure.

    python experiments/run.py --pairs 2 4 8 16 --steps 3000 --seeds 0 \
        --out results.json
    python experiments/run.py --skip-sweep --out scaling.json
    python experiments/run.py --pairs 8 16 --steps 20000 --blocks attention \
        --out control-attention.json
    python experiments/render_readme.py --mqar results.json \
        --scaling scaling.json --control control-attention.json --readme README.md

Each section is rendered from its own file, and a missing file is reported as a
missing file rather than rendered as an empty table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def _load(path: str | None, label: str) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read {label} from {path}: {exc}", file=sys.stderr)
        return None


def mqar_table(payload: dict, caption: str) -> str:
    rows = payload.get("mqar", {})
    if not rows:
        return f"_{caption}: no rows_"
    config = payload.get("config", {})
    seeds = config.get("seeds") or []
    lines = [
        f"**{caption}** — d_model={config.get('d_model')}, "
        f"layers={config.get('n_layers')}, steps={config.get('steps')}, "
        f"seeds={seeds}, batch={config.get('batch_size')}, "
        f"lr={config.get('lr')}",
        "",
        "| pairs in context | model | parameters | accuracy (exact match) | "
        "chance | best accuracy at an unseen size |",
        "|---|---|---|---|---|---|",
    ]
    for n_pairs in sorted({v["n_pairs"] for v in rows.values()}):
        for block in ("attention", "ssm"):
            row = rows.get(f"{block}@{n_pairs}")
            if row is None:
                continue
            runs = row["accuracy_runs"]
            spread = f"{row['accuracy_mean']:.3f}"
            if len(runs) > 1:
                spread += f" ± {(max(runs) - min(runs)) / 2:.3f}"
            off = row.get("off_size_accuracy") or {}
            off_text = "—"
            if off:
                best = max(off, key=off.get)
                off_text = f"{off[best]:.3f} (at {best})"
            lines.append(
                f"| {n_pairs} | {block} | {row['parameters']:,} | {spread} | "
                f"{row['chance']:.3f} | {off_text} |"
            )
    return "\n".join(lines)


def scaling_table(payloads: list[dict]) -> str:
    """One table per baseline run, so the reader sees which one is which.

    Which baseline attention is measured against decides the headline: against
    an explicit causal mask the state-space model crosses over, and against the
    fused kernel it does not. Reporting one and not the other would be choosing
    the answer, so both are rendered.
    """
    if not payloads:
        return "_scaling: missing_"

    blocks: list[str] = []
    for payload in payloads:
        rows = payload.get("scaling", {})
        if not rows:
            blocks.append("_scaling: no rows_")
            continue
        config = payload.get("config", {})
        mode = config.get("causal_mode", "unknown")
        lines = [
            f"**attention baseline: causal_mode={mode}** "
            f"(batch={config.get('scaling_batch')}, d_model={config.get('d_model')}, "
            f"layers={config.get('n_layers')})",
            "",
            "| sequence length | model | forward+backward (s) | "
            "peak activation memory (MB) | MB per token |",
            "|---|---|---|---|---|",
        ]
        for length in sorted({v["length"] for v in rows.values()}):
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
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render(mqar: dict, scaling: list[dict], control: dict | None) -> str:
    parts = [
        mqar_table(mqar, "Main sweep"),
        "",
        "The control below is the same architecture at the same size, with the "
        "step budget raised and nothing else changed.",
        "",
        mqar_table(control, "Control: attention, 20,000 steps")
        if control
        else "_control: not run_",
        "",
        "### Length scaling",
        "",
        "Activation memory is the *rise in current RSS* sampled while the "
        "forward and backward pass runs, one fresh process per point. It is not "
        "a difference of `ru_maxrss` high-water marks: that value is reported "
        "out of `signal_struct`, which a forked child inherits from its parent, "
        "so a child of a large parent starts with the parent's peak already "
        "recorded and its own allocations are invisible. The first version of "
        "this probe measured that way and printed zero, or the same constant, "
        "for every configuration.",
        "",
        scaling_table(scaling),
    ]
    counts = mqar.get("parameter_check_vocab", {})
    config = mqar.get("config", {})
    parts += [
        "",
        "```",
        f"parameter check at the sweep vocabulary: ssm={counts.get('ssm'):,} vs "
        f"attention={counts.get('attention'):,}",
        f"main sweep wall time: {mqar.get('wall_seconds')}s",
    ]
    if control:
        parts.append(f"control wall time: {control.get('wall_seconds')}s")
    parts += ["```"]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqar", required=True)
    parser.add_argument("--scaling", action="append", default=[],
                        help="repeatable: one table per baseline")
    parser.add_argument("--control")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    mqar = _load(args.mqar, "the main sweep")
    if mqar is None:
        return 1
    scaling = [p for p in (_load(path, "a scaling run") for path in args.scaling) if p]
    control = _load(args.control, "the control run")

    readme = pathlib.Path(args.readme)
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print(f"{args.readme} has no {BEGIN} / {END} block", file=sys.stderr)
        return 1
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)

    rendered = render(mqar, scaling, control)
    # Refuse to publish a table that lost its content: a broken renderer produces
    # an empty block, and a reader cannot tell an empty result from a bug.
    for marker in ("| pairs in context |", "| sequence length |"):
        if marker not in rendered:
            print(f"refusing to write: {marker!r} missing from the render",
                  file=sys.stderr)
            return 1

    readme.write_text(f"{head}{BEGIN}\n{rendered}\n{END}{tail}")
    print(f"wrote {args.readme} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
