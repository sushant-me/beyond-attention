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

`--voice` renders the voice/affect section from `voice-affect.json` into its own
`VOICE:BEGIN`/`VOICE:END` block, so it can be regenerated without touching the
results block.

    python experiments/voice_affect.py --out voice-affect.json
    python experiments/render_readme.py --voice voice-affect.json --readme README.md

`--agent` does the same for the agent loop, from `agent-loop.json` into
`AGENT:BEGIN`/`AGENT:END`.

    python experiments/agent_loop.py --out agent-loop.json
    python experiments/render_readme.py --agent agent-loop.json --readme README.md

`--memory` renders the long-context memory section from `long-memory.json` into
`MEMORY:BEGIN`/`MEMORY:END`.

    python experiments/long_memory.py --out long-memory.json
    python experiments/render_readme.py --memory long-memory.json --readme README.md

`--length-extrapolation` is part of the *results* render, not a separate block:
that section used to be hand-written inside the results region, so re-rendering
the results deleted it. It is required alongside `--mqar` for exactly that
reason — a renderer that cannot produce a section the README contains must
refuse, not quietly drop it. `tests/test_render_readme.py` re-renders every block
from its JSON and fails if any of them differs from the committed README, which
is what keeps hand-written content out of a generated region.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"
VOICE_BEGIN = "<!-- VOICE:BEGIN -->"
VOICE_END = "<!-- VOICE:END -->"
AGENT_BEGIN = "<!-- AGENT:BEGIN -->"
AGENT_END = "<!-- AGENT:END -->"
MEMORY_BEGIN = "<!-- MEMORY:BEGIN -->"
MEMORY_END = "<!-- MEMORY:END -->"


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


def scan_inner_table(payload: dict | None) -> str:
    """The loop-versus-vectorised comparison, with the agreement column.

    Both paths doing the same arithmetic to float32 resolution is part of the
    result, not a footnote: a faster path that computes something else would not
    be a faster path, and the first version of this experiment gave the two
    paths different random inputs so its difference column was meaningless.
    """
    if payload is None:
        return "_inner-scan comparison: missing_"
    rows = payload.get("results", {})
    if not rows:
        return "_inner-scan comparison: no rows_"
    config = payload.get("config", {})
    lines = [
        f"length={config.get('length')}, batch={config.get('batch')}, "
        f"d_inner={config.get('dim')}, d_state={config.get('state')}, "
        f"threads={config.get('threads')}",
        "",
        "| inner scan | chunk | chunks | seconds | peak MB | agrees with loop |",
        "|---|---|---|---|---|---|",
    ]
    for chunk in sorted({v["chunk"] for v in rows.values()}):
        for inner in ("loop", "vectorized"):
            row = rows.get(f"{inner}@{chunk}")
            if row is None:
                continue
            loop = rows.get(f"loop@{chunk}")
            vec = rows.get(f"vectorized@{chunk}")
            agree = "—"
            if loop and vec:
                worst = max(
                    abs(loop[k] - vec[k]) / max(1.0, abs(loop[k]))
                    for k in ("out_sum", "grad_sum")
                )
                agree = f"yes ({worst:.1e})"
            lines.append(
                f"| {inner} | {chunk} | {config.get('length', 0) // chunk} | "
                f"{row['seconds']:.2f} | {row['peak_rss_kb'] / 1024:.1f} | {agree} |"
            )
    return "\n".join(lines)


def extrapolation_section(payload: dict | None, mqar: dict | None = None) -> str:
    """The length-extrapolation section, rendered from `length-extrapolation.json`.

    This section used to be written by hand inside the results block, which meant
    the documented results command **deleted** it: the renderer's guard checks for
    the markers its own tables contain, and a section it cannot produce is not one
    it notices losing. It is generated now, and `tests/test_render_readme.py`
    fails if any rendered block differs from what this module produces -- which is
    the property that keeps hand-written content out of a generated region.

    Two numbers in the prose come from `tasks.mqar_batch`'s documented default key
    space (`max(n_pairs, 8)`) rather than from the JSON, because that default is
    the reason the main sweep's "unseen size" column reads the way it does. They
    are computed here from `train_pairs`, not typed.
    """
    if payload is None:
        return ("### Does either model read longer than it trained?\n\n"
                "_length extrapolation: not run_ (pass "
                "`--length-extrapolation length-extrapolation.json`; the section "
                "is rendered, and refusing is better than deleting it)_")

    config = payload.get("config", {})
    extrapolated = payload.get("extrapolated", {})
    reference = payload.get("reference", {})
    train_pairs = config.get("train_pairs")
    train_length = config.get("train_length")
    n_keys = config.get("n_keys")
    seeds = config.get("seeds") or []
    blocks = ("attention", "ssm")
    pairs_list = sorted({int(k) for model in extrapolated.values()
                         for k in model})

    def tokens(n_pairs: int) -> int:
        return 2 * n_pairs + 1 + 1

    def fmt(model: str, n_pairs: int) -> tuple[str, float | None, float | None]:
        row = extrapolated.get(model, {}).get(str(n_pairs))
        if row is None:
            return "—", None, None
        text = f"{row['mean']:.3f}"
        if len(row.get("per_seed") or []) > 1:
            text += f" ±{row['spread']:.3f}"
        return text, row["mean"], row.get("spread")

    def ref(model: str, n_pairs: int) -> str:
        row = reference.get(model, {}).get(str(n_pairs))
        return "—" if row is None else f"{row['mean']:.3f}"

    # The default key space the main sweep's "unseen size" column was built
    # with: `mqar_batch` uses `max(n_pairs, 8)` when `n_keys` is not given.
    default_keys = max(int(train_pairs), 8)
    longest = max(pairs_list)
    chance = payload.get("chance")
    chance_text = f"1/{round(1 / chance)}" if chance else "—"
    ratio = tokens(longest) / tokens(int(train_pairs)) if train_pairs else 0.0

    lines = [
        "### Does either model read longer than it trained?",
        "",
        f"The \"unseen size\" column above is not a clean measure of that, and it "
        f"took building the experiment below to see why. `mqar_batch` sizes its "
        f"key space from the pair count by default: trained at {train_pairs} "
        f"pairs a model meets keys 1–{default_keys} and values "
        f"{default_keys + 1}–{2 * default_keys}, while a {longest}-pair batch "
        f"hands it keys 1–{n_keys} and values {n_keys + 1}–{2 * n_keys} — token "
        f"ids it has never seen, in roles it has never seen them in. Every "
        f"`0.000` in that column is an evaluation at a *longer* length than "
        f"training, which is precisely the case the column was meant to measure. "
        f"Those zeros are a vocabulary mismatch, not a failure to generalise.",
        "",
        f"Holding the key space fixed so the vocabulary is identical at every "
        f"length changes the picture completely. Both models are trained at "
        f"**{train_pairs} pairs ({tokens(int(train_pairs))} tokens)** — the "
        f"longest length at which *both* solve the task outright — then evaluated "
        f"with frozen weights at the lengths below (up to "
        f"**{ratio:.1f}x**):",
        "",
        "| pairs | tokens | x train | attention | ssm | chance | a model trained "
        "at that length: attention | ssm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    means: dict[str, dict[int, float]] = {model: {} for model in blocks}
    spreads: dict[str, dict[int, float]] = {model: {} for model in blocks}
    for n_pairs in pairs_list:
        cells = []
        for model in blocks:
            text, mean, spread = fmt(model, n_pairs)
            cells.append(text)
            if mean is not None:
                means[model][n_pairs] = mean
                spreads[model][n_pairs] = spread or 0.0
        train_length_here = tokens(n_pairs) / tokens(int(train_pairs))
        lines.append(
            f"| {n_pairs} | {tokens(n_pairs)} | {train_length_here:.1f}x | "
            f"{cells[0]} | {cells[1]} | {chance_text} | "
            f"{ref('attention', n_pairs)} | {ref('ssm', n_pairs)} |"
        )
    seeds_text = {1: "One seed", 2: "Two seeds",
                  3: "Three seeds"}.get(len(seeds), f"{len(seeds)} seeds")
    lines += [
        "",
        f"{seeds_text}, spread shown, nothing retrained between rows.",
        "",
        "What this says, including the parts that are unflattering:",
        "",
    ]

    # Every bullet below is conditional on the numbers above it.
    longer = [p for p in pairs_list if p > int(train_pairs)]
    perfect = [p for p in pairs_list if means["attention"].get(p) == 1.0
               and means["ssm"].get(p) == 1.0]
    if perfect and longer and all(
            (means[m].get(p) or 0.0) < 1.0 for p in longer for m in blocks):
        where = ", ".join(str(p) for p in perfect)
        first = longer[0]
        lines.append(
            f"* **Neither architecture extrapolates here.** Both are perfect at "
            f"the length they trained on ({where} pairs) and both fall below "
            f"1.000 at every longer length — attention to "
            f"{means['attention'][first]:.3f} and the SSM to "
            f"{means['ssm'][first]:.3f} at {first} pairs, the shortest step past "
            f"training."
        )

    attention_ahead = [p for p in longer
                       if means["attention"].get(p, 0.0)
                       > means["ssm"].get(p, 0.0)]
    if attention_ahead and len(attention_ahead) == len(longer):
        detail = ", ".join(
            f"{means['attention'][p]:.3f} against {means['ssm'][p]:.3f} "
            f"at {p} pairs" for p in longer)
        lines.append(
            f"* **Attention decays more slowly than the SSM** at every step — "
            f"{detail}. On this task the state-space model is the weaker of the "
            f"two past its training length, which is the opposite of what the "
            f"architecture's reputation would predict."
        )

    if longer:
        # The reference is the column that stops the curve being read as a
        # generalisation result, so it is reported with its gap and its spread
        # rather than with an adjective. Where the gap is smaller than the
        # three-seed spread it is not a difference; where it is larger, the
        # decay is part extrapolation and part task difficulty, and no column
        # here separates the two.
        reference_notes = []
        reference_seeds = set()
        for model in blocks:
            for p in longer:
                row = reference.get(model, {}).get(str(p))
                if row is None or p not in means[model]:
                    continue
                reference_seeds.add(len(row.get("per_seed") or []))
                gap = abs(means[model][p] - row["mean"])
                reference_notes.append(
                    f"{model} {means[model][p]:.3f} against {row['mean']:.3f} "
                    f"(gap {gap:.3f}, three-seed spread ±{spreads[model][p]:.3f})"
                )
        if reference_notes:
            seeds_note = (f"{max(reference_seeds)} seed"
                          f"{'s' if max(reference_seeds) != 1 else ''}")
            lines.append(
                f"* **Part of the decay is task difficulty, and the reference "
                f"column is what says so.** A model trained from scratch at the "
                f"longest length reaches "
                + "; ".join(reference_notes)
                + f". The reference is {seeds_note}, so a gap smaller than the "
                f"spread is not a difference and a larger one is part "
                f"extrapolation and part how hard MQAR is at that length for a "
                f"two-layer, `d_model`-{config.get('d_model')} model — which no "
                f"column here separates. Without it the curve looks like a "
                f"generalisation result and is not one."
            )

    rows = (mqar or {}).get("mqar", {})
    solved_long = sorted(
        int(key.split("@")[1]) for key, value in rows.items()
        if key.startswith("ssm@") and value.get("accuracy_mean", 0) >= 0.999)
    if solved_long:
        biggest = max(solved_long)
        attention_row = rows.get(f"attention@{biggest}", {})
        attention_mean = attention_row.get("accuracy_mean")
        if attention_mean is not None and attention_mean < 0.999:
            lines.append(
                f"* The SSM does fit *longer training lengths* better than "
                f"attention: in the main sweep it reaches 1.000 at {biggest} "
                f"pairs where attention reaches {attention_mean:.3f}. So \"fits "
                f"long sequences when trained on them\" and \"generalises to "
                f"longer ones when trained short\" are separate properties, and "
                f"the two architectures sit on opposite sides of them."
            )

    lines += [
        "",
        "One training length, one task, one model size. This measures MQAR at "
        f"{train_pairs} pairs, {train_length} tokens, "
        f"d_model={config.get('d_model')}, {config.get('n_layers')} layers, on "
        f"{seeds_text.lower()}.",
    ]
    return "\n".join(lines)


def render(mqar: dict, scaling: list[dict], control: dict | None,
           scan_inner: dict | None = None,
           extrapolation: dict | None = None) -> str:
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
        extrapolation_section(extrapolation, mqar),
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
        "",
        "### Which inner scan to use",
        "",
        "The recurrence inside each chunk can be run as a sequential loop, or as "
        "a Hillis-Steele scan vectorised along the chunk axis. The loop does "
        "`O(chunk)` work and the scan does `O(chunk * log2(chunk))`, so on a CPU "
        "the scan is doing more arithmetic to remove interpreter overhead that "
        "is cheaper than the arithmetic it adds.",
        "",
        scan_inner_table(scan_inner),
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


def voice_section(payload: dict) -> str:
    """The prosody/affect section, rendered from `voice-affect.json`.

    Two things this renderer is careful about, because the section is about
    measurement discipline and would be self-refuting otherwise:

    * every row comes from the JSON, including the parameters the synthesiser
      was given, so a reader can see what was manipulated rather than take the
      condition names' word for it;
    * the controls are rendered *next to* the accuracy rather than in a
      footnote, because "0.969 against chance 0.250" and "two identical
      conditions score 0.562" are the same claim read two ways.
    """
    config = payload.get("config", {})
    conditions = payload.get("conditions", {})
    sep = payload.get("separability", {})
    controls = payload.get("controls", {})
    single = payload.get("single_descriptor", {})
    classes = sep.get("classes") or list(conditions)

    features = config.get("cluster_features") or []
    lines = [
        f"**Synthetic utterances** — {sep['n_utterances']} utterances "
        f"({config.get('utterances_per_condition')} per condition, "
        f"{config.get('duration_s')} s each) at "
        f"{config.get('sample_rate'):,} Hz, seed {config.get('seed')}. "
        f"Classifier: {config.get('classifier')}, over {len(features)} "
        f"descriptors.",
        "",
        "| condition | F0 target (Hz) | F0 mean (Hz) | F0 std (Hz) | "
        "energy std | energy std (voiced) | voiced runs/s | jitter (Hz) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in classes:
        entry = conditions[name]
        d = entry["descriptors"]
        lines.append(
            f"| {name} | {entry['parameters']['f0_base']:.0f} | "
            f"{d['f0_mean']:.1f} | {d['f0_std']:.2f} | {d['energy_std']:.4f} | "
            f"{d['energy_std_voiced']:.4f} | {d['speaking_rate']:.2f} | "
            f"{d['jitter']:.3f} |"
        )

    lines += [
        "",
        f"**Separability in descriptor space** — leave-one-out nearest centroid, "
        f"chance {sep['chance']:.3f}",
        "",
        f"Accuracy **{sep['accuracy']:.3f}** over {sep['n_utterances']} "
        f"utterances ({sep['z_vs_chance']:+.1f}σ against chance). The closest "
        f"pair of condition centroids is {sep['closest_centroids']:.2f} apart "
        f"against a mean within-condition spread of {sep['within_spread']:.2f} "
        f"— a ratio of **{sep['ratio']:.1f}x**.",
        "",
        "| true \\ predicted | " + " | ".join(classes) + " |",
        "|---|" + "---:|" * len(classes),
    ]
    for name, row in zip(classes, sep["confusion"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "Each descriptor on its own, same classifier:",
        "",
        "| descriptor alone | accuracy |",
        "|---|---:|",
    ]
    for name, row in sorted(single.items(), key=lambda kv: -kv[1]["accuracy"]):
        lines.append(f"| `{name}` | {row['accuracy']:.3f} |")

    shuffle = controls["label_shuffle"]
    twin = controls["identical_parameters"]
    noise = controls["white_noise_voicing"]
    recovery = controls["f0_recovery"]
    lines += [
        "",
        "| control | measured | what it rules out |",
        "|---|---|---|",
        f"| labels shuffled, {int(shuffle['rounds'])} rounds | mean "
        f"{shuffle['mean']:.3f}, 95th pct {shuffle['p95']:.3f}, max "
        f"{shuffle['max']:.3f} | that the accuracy is the classifier's rather "
        f"than the descriptors' (chance {sep['chance']:.3f}) |",
        f"| two conditions, identical parameters | {twin['accuracy']:.3f} "
        f"({twin['z_vs_chance']:+.1f}σ) | that anything other than the "
        f"generator's parameters separates the conditions (chance "
        f"{twin['chance']:.3f}) |",
        f"| white noise through the voiced decision | voiced ratio max "
        f"{noise['voiced_ratio_max']:.3f}, largest autocorrelation peak "
        f"{noise['max_confidence']:.3f} | a voiced/unvoiced decision that "
        f"always says yes (threshold {noise['threshold']}) |",
        f"| F0 recovery against the generator | mean "
        f"{recovery['mean_abs_error_pct']:.2f}%, worst "
        f"{recovery['worst_abs_error_pct']:.2f}% | descriptors that do not "
        f"track what the synthesiser was asked for |",
        f"| encoder separates two utterances | "
        f"{controls['encoder_separates_two_utterances']:.4f} max difference | a "
        f"degenerate all-zero projection |",
        f"| bridge output | `{tuple(config.get('bridge_output_shape') or [])}` "
        f"| a front end that never reaches the model's `(B, L, D)` layout |",
    ]
    return "\n".join(lines)


def agent_section(payload: dict) -> str:
    """The agent-loop section, rendered from `agent-loop.json`.

    Three things this renderer is careful about, because the section is about
    measurement discipline and would be self-refuting otherwise:

    * the **controls are rendered beside the solve rate**, in the same table
      shape, because "1.000 solved" and "0.000 solved once the state is wiped"
      are one claim read two ways, and a control in a footnote is a control
      nobody reads;
    * every cell comes from the JSON, including the number of loop steps each
      family needs, so a reader can see why the budget columns have the shape
      they do rather than take the family names' word for it;
    * the example trace shows **what the policy read**, not only what it called,
      because a table of actions alone is consistent with a controller that read
      its operand out of the task text.
    """
    config = payload.get("config", {})
    solve = payload.get("solve_rate", {})
    controls = payload.get("controls", {})
    families = config.get("families") or []
    budgets = config.get("budgets") or []
    step_counts = config.get("step_counts") or {}
    max_budget = max(budgets) if budgets else 0

    def rate(cells: dict, family: str, budget: int) -> str:
        cell = cells.get(f"{family}@{budget}")
        return "—" if cell is None else f"{cell['rate']:.3f}"

    def overall(cells: dict, budget: int) -> str:
        chosen = [cells[f"{f}@{budget}"] for f in families
                  if f"{f}@{budget}" in cells]
        solved = sum(c["solved"] for c in chosen)
        total = sum(c["total"] for c in chosen)
        return f"{solved / total:.3f}" if total else "—"

    lines = [
        f"**Seeded task suite** — {config.get('suite_size')} tasks "
        f"({config.get('tasks_per_family')} per family), seed "
        f"{config.get('seed')}. Each task is a plan over bounded integers plus a "
        f"bounded key/value table, and its answer is computed in Python integers "
        f"by `evaluate_plan`, so correctness is a property of the task. Tools: "
        + ", ".join(f"`{t}`" for t in config.get("tools", [])) + ".",
        "",
        f"The memory is a selective-scan state "
        f"{len(config.get('registers', []))} registers wide, one named register "
        f"per dimension: "
        + ", ".join(f"`{r}`" for r in config.get("registers", [])) + ".",
        "",
        "| task family | loop steps needed | "
        + " | ".join(f"budget {b}" for b in budgets) + " |",
        "|---|---:|" + "---:|" * len(budgets),
    ]
    for family in families:
        lines.append(
            f"| {family} | {step_counts.get(family)} | "
            + " | ".join(rate(solve, family, b) for b in budgets) + " |"
        )

    labels = {
        "no_memory": "no memory: the state is wiped before each decision",
        "scalar_carry": "scalar carry: the same controller, one Python int",
        "fixed_decay": "fixed decay: the same width, a constant gate",
        "random_action": "random action: tools and arguments uniform",
    }
    lines += [
        "",
        f"**Controls at budget {max_budget}** — per-family solve rate, "
        f"with the agent's own row for comparison. \"Overall\" averages the "
        f"{len(families)} families.",
        "",
        "| condition | " + " | ".join(families) + " | overall |",
        "|---|" + "---:|" * (len(families) + 1),
    ]
    rows = [("the agent, memory = the SSM state", solve, overall(solve, max_budget))]
    for key in ("no_memory", "scalar_carry", "fixed_decay", "random_action"):
        cells = controls.get(key, {})
        rows.append((labels[key], cells, overall(cells, max_budget)))
    for label, cells, total in rows:
        lines.append(
            f"| {label} | "
            + " | ".join(rate(cells, f, max_budget) for f in families)
            + f" | **{total}** |"
        )

    one = controls.get("one_step", {}).get("all", {})
    if one:
        lines += [
            "",
            f"The budget-1 slice of the agent is the one-step control: "
            f"**{one['rate']:.3f}** solved over all {one['total']} tasks. Only "
            f"the `literal` family, whose answer is written in the task text, is "
            f"reachable in a single decision — so the suite is not one call deep.",
        ]

    zero = controls.get("zero_carry_branch", {})
    if zero:
        lines += [
            "",
            f"**Where the first version's residual bit.** Of the "
            f"{zero['total']} `branch` tasks, {zero['count']} arrive at the "
            f"`IFPOS` decision (step {zero['step']}) carrying exactly 0 — the "
            f"value a write residual turns positive. Those are exactly the tasks "
            f"the `A = -50` register got wrong, and the last column below is the "
            f"fix.",
        ]

    exactness = controls.get("write_exactness", {})
    if exactness:
        lines += [
            "",
            "**Register write exactness** — what a register holds after being "
            "written 9 and then 0, for four choices of the decay. A residual "
            "above zero is not a rounding detail: `IFPOS` branches on the sign "
            "of this number.",
            "",
            "| A | exp(A) | residual after overwriting | sign test holds |",
            "|---:|---:|---:|---|",
        ]
        for a, row in exactness.items():
            lines.append(
                f"| {a} | {row['multiplier']:.3e} | "
                f"{row['residual_after_overwrite']:.3e} | "
                f"{'yes' if row['sign_test_holds'] else 'no'} |"
            )

    trace = payload.get("example_trace", {})
    if trace:
        lines += [
            "",
            f"**The published trace** — `{trace.get('text')}`, answer "
            f"`{trace.get('answer')}`, {len(trace.get('steps') or [])} steps, "
            f"stop reason `{trace.get('stop_reason')}`, solved "
            f"`{trace.get('solved')}`. \"Reads\" is the state the policy acted "
            f"on, before the call.",
            "",
            "| step | instruction | reads: op / arg / carry / observations | "
            "action | result |",
            "|---:|---|---|---|---:|",
        ]
        for step in trace.get("steps", []):
            reads = step["reads"]
            action = step["action"] + "(" + ", ".join(
                f"{k}={v}" for k, v in step["args"].items()
            ) + ")"
            shown = step["result"]
            if step.get("error"):
                shown = f"`{step['error']}`"
            lines.append(
                f"| {step['index']} | {step['instruction']} | "
                f"{reads['op_code']} / {reads['arg']} / {reads['carry']:g} / "
                f"{reads['observations']} | `{action}` | {shown} |"
            )
    lines += ["", selective_section(payload)]
    return "\n".join(lines)


def selective_section(payload: dict) -> str:
    """The selective-memory subsection, rendered from the same JSON.

    Everything numeric here is interpolated, including the two claims this
    subsection exists to falsify, because a false claim whose numbers are typed
    by hand is a false claim that can drift away from its own evidence.
    """
    config = payload.get("config", {})
    controls = payload.get("controls", {})
    shape = config.get("selective", {})
    widths = controls.get("state_width", {})
    distractors = controls.get("distractor_rate", {})
    correspondence = controls.get("scalar_selective", {})
    budget = shape.get("step_count", 0)
    families = config.get("families") or []
    selective_at_budget = payload.get("solve_rate", {}).get(
        f"selective@{budget}", {})

    def rate(cells: dict, *path: str) -> float:
        for key in path:
            cells = cells.get(key, {})
        return cells.get("rate", 0.0)

    agent = rate(selective_at_budget)
    scalar = rate(controls.get("scalar_carry", {}), f"selective@{budget}")
    fixed = rate(controls.get("fixed_decay", {}), f"selective@{budget}")
    no_memory = rate(controls.get("no_memory", {}), f"selective@{budget}")
    random_at = controls.get("random_action", {}).get(f"selective@{budget}", {})
    floor = random_at.get("rate", 0.0)

    lines = [
        "### Selective memory: retention that depends on the input",
        "",
        f"**What this family asks.** {shape.get('stores')} events carry a value "
        f"tagged with a key, {shape.get('distractors')} carry a value tagged "
        f"with nothing, and the events arrive in a shuffled order; then a query "
        f"asks for the value of one of the keys, drawn uniformly. The answer is "
        f"a dict lookup inside `evaluate_plan`, computed in Python integers — "
        f"the target never consults the memory, which is what makes it ground "
        f"truth rather than a restatement. What makes the family *selective* "
        f"rather than merely multi-step is that retention depends on the event: "
        f"a keyed event is written to the slot its key addresses and a "
        f"distractor is written nowhere, so which events survive, and where "
        f"they land, is a property of the input and not of the position. The "
        f"memory is {shape.get('state_width')} slots addressed by "
        f"`key % width`, appended to the same eight named registers the "
        f"arithmetic families use and run through the same recurrence, so the "
        f"state is {8 + shape.get('state_width', 0)} registers wide.",
        "",
        f"**Controls on the selective family, at budget {budget}** — the same "
        f"controller with a different memory, on the same {selective_at_budget.get('total')} "
        f"tasks. The `fixed decay` and `scalar carry` rows are what decide what "
        f"the architecture is worth.",
        "",
        "| condition | selective | what the memory is |",
        "|---|---:|---|",
        f"| the agent, memory = the gated SSM state | **{agent:.3f}** | "
        f"{shape.get('state_width')} slots, written by the key of the event |",
        f"| fixed decay: the same width, a constant gate | {fixed:.3f} | the "
        f"same slots, every value-carrying event written to every one |",
        f"| scalar carry: one Python int | {scalar:.3f} | the last keyed value "
        f"seen; distractors do not move it |",
        f"| no memory: the state is wiped before each decision | {no_memory:.3f} "
        f"| nothing survives a step |",
        f"| random action: tools and arguments uniform | {floor:.3f} | the floor |",
        "",
        f"The scalar's failure is exact rather than statistical. It holds the "
        f"last keyed value it saw, so it must be right precisely when the "
        f"queried key is the one the last `PUT` named, and wrong otherwise: "
        f"**{correspondence.get('solved')} of {correspondence.get('tasks')}** "
        f"tasks solved against "
        f"**{correspondence.get('queried_the_last_store')}** tasks whose query "
        f"named the last store, with the predicted set matching the solved set "
        f"on **{correspondence.get('exact_matches')} of "
        f"{correspondence.get('tasks')}** tasks. A rate below 1.000 on its own "
        f"would be consistent with a merely harder task; the correspondence is "
        f"what says a second register is what was missing.",
        "",
        f"**State width** — the same tasks read by a memory of "
        + ", ".join(str(row["slots"]) for row in widths.values()) +
        f" slots. The eight named registers are always present, so the state is "
        f"that much wider again; \"slots\" is the column that matters. A "
        f"one-slot state aliases every key onto slot 0, which is why its column "
        f"is the scalar's.",
        "",
        "| memory slots | total state width | agent | scalar | fixed decay | "
        "no memory |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in widths.values():
        lines.append(
            f"| {row['slots']} | {row['total_width']} | "
            f"{rate(row['agent']):.3f} | {rate(row['scalar']):.3f} | "
            f"{rate(row['fixed']):.3f} | {rate(row['none']):.3f} |"
        )

    smallest_solved = next(
        (row["slots"] for row in widths.values() if rate(row["agent"]) >= 1.0),
        None,
    )
    lines += [
        "",
        f"The family starts to solve at **{smallest_solved} memory slots** on "
        f"this key space of {shape.get('keys')} keys, and the curve between "
        f"zero and there is the honest capacity statement rather than a cliff: "
        f"a narrow state does not fail, it aliases, and it is right exactly "
        f"when no two keys in the stream collide.",
        "",
        f"**Distractor rate** — the same shape with more of the stream "
        f"discarded, every point its own task set and its own budget. At zero "
        f"distractors the fixed-decay state *is* the scalar "
        f"({rate(distractors.get('0', {}).get('fixed', {})):.3f} against "
        f"{rate(distractors.get('0', {}).get('scalar', {})):.3f}, and the "
        f"experiment asserts the per-task agreement). The distractors are what "
        f"the gate is for.",
        "",
        "| distractor share | agent | scalar | fixed decay | no memory |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in distractors.values():
        lines.append(
            f"| {row['rate']:.3f} | {rate(row['agent']):.3f} | "
            f"{rate(row['scalar']):.3f} | {rate(row['fixed']):.3f} | "
            f"{rate(row['none']):.3f} |"
        )

    trace = payload.get("selective_example_trace", {})
    if trace:
        lines += [
            "",
            f"**The published selective trace** — `{trace.get('text')}`, answer "
            f"`{trace.get('answer')}`, {len(trace.get('steps') or [])} steps, "
            f"stop reason `{trace.get('stop_reason')}`, solved "
            f"`{trace.get('solved')}`. The `slots` column is the whole memory "
            f"at that step: three keyed values held at once, and the distractor "
            f"in no slot.",
            "",
            "| step | instruction | reads: op / arg / slots | action | result |",
            "|---:|---|---|---|---:|",
        ]
        for step in trace.get("steps", []):
            reads = step["reads"]
            slots = ", ".join(f"{v:g}" for v in reads.get("slots", []))
            action = step["action"] + "(" + ", ".join(
                f"{k}={v}" for k, v in step["args"].items()
            ) + ")"
            shown = step["result"]
            if step.get("error"):
                shown = f"`{step['error']}`"
            lines.append(
                f"| {step['index']} | {step['instruction']} | "
                f"{reads['op_code']} / {reads['arg']} / [{slots}] | "
                f"`{action}` | {shown} |"
            )

    width_one = widths.get("1", {})
    widest = max(widths.values(), key=lambda row: row["slots"]) if widths else {}
    lines += [
        "",
        "Three claims this table makes easy, and that are false:",
        "",
        f"* *\"The architecture is what makes the agent work.\"* On the "
        f"{len(families) - 1} arithmetic families the scalar carry still "
        f"reproduces the agent row for row, and the recurrence earns nothing "
        f"there. On `selective` the gated state reaches **{agent:.3f}** where "
        f"the scalar reaches **{scalar:.3f}** and a fixed-decay state of the "
        f"*same width* reaches **{fixed:.3f}** — so what is doing the work is "
        f"the input-dependent gate, and the width alone is not enough. That "
        f"gate is **hand-set**, in `embed`: it is not learned, it is not "
        f"produced by a projection, and a Python dict in `evaluate_plan` "
        f"computes the same answers with no model at all. What is established "
        f"is narrower than the claim: *this* hand-designed gate, at *this* "
        f"width, on *this* family, does something a scalar and a constant gate "
        f"do not.",
        "",
        f"* *\"The state is doing something a scalar cannot.\"* True on this "
        f"family, and it rests on one design decision: the query key is drawn "
        f"uniformly from the stored keys, so one register is insufficient on "
        f"most tasks. It is not a general statement about state-space models — "
        f"a scalar with a Python dict beside it would tie the agent exactly, "
        f"and no task here requires more than the four slots the family "
        f"declares.",
        "",
        f"* *\"This generalises.\"* It does not. The family is closed and "
        f"synthetic: a fixed grammar, a shuffled event stream, a bounded key "
        f"space, and an answer computed by a dict. Nothing is trained — no "
        f"gradient anywhere in this path — the controller is hand-written "
        f"branching, and the gate is set by hand. The clearest evidence that "
        f"the *selectivity* rather than the capacity is what this family "
        f"measures is the fixed-decay row of the width table: it scores "
        f"{rate(width_one.get('fixed', {})):.3f} at "
        f"{width_one.get('slots')} slot and "
        f"{rate(widest.get('fixed', {})):.3f} at {widest.get('slots')} slots, "
        f"so widening it changes nothing, while the gated state at the same "
        f"{widest.get('slots')} slots reaches "
        f"{rate(widest.get('agent', {})):.3f}. The part of the model this "
        f"repository has never learned is the part that decides what to write.",
    ]
    return "\n".join(lines)


def memory_section(payload: dict) -> str:
    """The long-context memory section, rendered from `long-memory.json`.

    Three things this renderer is careful about, because the section is about
    measurement discipline and would be self-refuting otherwise:

    * the **controls are in the same table as the agent** and every cell is
      ``solved/total`` rather than a rate, because the long distances afford
      fewer tasks and a rate quoted over two of them would hide that;
    * the **bytes are the measured ones** -- ``MemoryStore.bytes()`` on a live
      store, ``state_bytes`` from the module -- with the analytic floor beside the
      allocated figure, since a growing array over-allocates and only one of the
      two numbers is what a process holds;
    * the **crossover is computed from the cells**, not asserted: every threshold
      in the prose is the first measured distance or fact count at which a
      condition's rate changes, so a rerun that moves it moves the sentence.
    """
    config = payload.get("config", {})
    distance = payload.get("distance", {})
    capacity = payload.get("capacity", {})
    stale = payload.get("stale", {})
    ledger = payload.get("bytes", {})
    cross = payload.get("crossover", {})
    model = payload.get("model_scale", {})
    examples = payload.get("examples", {})

    labels = {
        "store": "the store: 8 tagged slots",
        "single_slot_store": "the store: 1 slot",
        "unbounded_store": "an unbounded store",
        "working_state": "the working state alone (no store)",
        "wide_state": "a 64x wider state, same write rule",
        "retrieval_disabled": "the store, retrieval disabled",
        "random_retrieval": "random live value (floor)",
        "untagged_store": "the store, untagged",
        "stale_store": "first-write-wins store",
    }
    conditions = config.get("conditions") or []

    def cell_text(cell: dict) -> str:
        return f"{cell['solved']}/{cell['total']}"

    def carried(cell: dict) -> str:
        return f"{cell['state_bytes'] + cell['store_bytes']:,}"

    distances = sorted(int(key) for key in distance)
    capacities = sorted(int(key) for key in capacity)
    stale_distances = sorted(int(key) for key in stale)

    lines = [
        f"**What each condition carries, and what it solves** — "
        f"{sum(distance[str(d)]['tasks'] for d in distances)} recall tasks at "
        f"distances {', '.join(str(d) for d in distances)} (a plan of "
        f"`distance + 7` instructions), seed {config.get('seed')}. Every cell is "
        f"**solved/total**, not a rate: the long distances afford fewer tasks, "
        f"and that is worth seeing. \"Bytes carried\" is the working state plus "
        f"the store, measured from the live objects.",
        "",
        "| condition | bytes carried | "
        + " | ".join(f"d={d}" for d in distances) + " |",
        "|---|---:|" + "---:|" * len(distances),
    ]
    for name in conditions:
        first = distance[str(distances[0])]["conditions"][name]
        lines.append(
            f"| {labels.get(name, name)} | {carried(first)} | "
            + " | ".join(cell_text(distance[str(d)]["conditions"][name])
                         for d in distances)
            + " |"
        )

    single = distance[str(distances[0])]["conditions"].get("single_slot_store")
    single_bytes = (f"{single['state_bytes'] + single['store_bytes']:,}"
                    if single else "17")
    lines += [
        "",
        "Three rows need reading carefully. The **random live value** floor is "
        "not a floor at all in this table: with one fact written there is one "
        "live value, so drawing a live value at random *is* the answer. Its "
        "discriminating power appears only where several facts are live — the "
        "capacity table below — which is why it is reported in both places. The "
        "**unbounded store** costs less than the fixed store at every length "
        "here, because the single-fact task writes once: the unbounded "
        "alternative is cheap precisely when there is nothing to keep. And the "
        "**one-slot store** solves every distance in this table at "
        f"{single_bytes} B, from which the only honest reading is that the "
        "*width* of the store is not what carries the fact — the key is. What "
        "one slot cannot do is hold two facts, and that is the capacity table.",
    ]

    rows = ledger.get("rows") or []
    if rows:
        has_kv = any("model_kv_cache_bytes" in row for row in rows)
        header = ("| episode length | working state | store (8 slots) | "
                  "untagged store | unbounded store | its arithmetic floor | "
                  "transcript (1 int64/event) |")
        rule = "|---:|---:|---:|---:|---:|---:|---:|"
        if has_kv:
            header += " model KV cache |"
            rule += "---:|"
        lines += [
            "",
            "**Bytes carried against length** — the store's figure is measured "
            "from a live `MemoryStore`, the unbounded one after that many writes "
            "(which is why it is a staircase: a growing `numpy` array doubles), "
            "and the transcript is arithmetic — one int64 per event, the least a "
            "replay needs. The run object retains every step whether or not "
            "anyone asks it to.",
            "",
            header,
            rule,
        ]
        for row in rows:
            line = (
                f"| {row['length']:,} | {row['working_state_bytes']:,} | "
                f"{row['store_bytes']:,} | {row['untagged_store_bytes']:,} | "
                f"{row['unbounded_store_bytes']:,} | "
                f"{row['unbounded_store_arithmetic']:,} | "
                f"{row['transcript_bytes']:,} |"
            )
            if has_kv:
                line += f" {row.get('model_kv_cache_bytes', 0):,} |"
            lines.append(line)
        if model.get("ssm_state_bytes"):
            lines += [
                "",
                f"For scale: the model's own state is "
                f"**{model['ssm_state_bytes']:,} B at every length** "
                f"(`{model.get('source_ssm')}`), and its attention cache is "
                f"**{model.get('kv_cache_bytes', 0):,} B at "
                f"{model.get('kv_cache_length', 0):,} tokens** "
                f"(`{model.get('source_kv')}`) — "
                f"{model.get('kv_cache_bytes_per_token', 0):,} B a token, quoted "
                f"from the streaming measurements rather than recomputed here.",
            ]

    if cross:
        bullets = []
        working_fail = cross.get("working_state_fails_at_distance")
        wide_fail = cross.get("wide_state_fails_at_distance")
        if working_fail is not None:
            same = (wide_fail == working_fail)
            bullets.append(
                f"**The working state starts failing at distance "
                f"{working_fail}**"
                + (f" — and so does the {cross.get('wide_state_bytes', 0):,}-byte "
                   f"state at exactly the same distance, because the write rule "
                   f"puts every observation in the same register. "
                   if same else ". ")
                + f"It is {cross.get('working_state_bytes', 0):,} B and it holds "
                f"the last observation; nothing about the *width* of a state "
                f"that is not addressed by key changes that."
            )
        if single := distance[str(distances[0])]["conditions"].get(
                "single_slot_store"):
            size = single["state_bytes"] + single["store_bytes"]
            bullets.append(
                f"**A single 17-byte slot does the same job at every distance "
                f"measured**, so this trade is not about the store's size: "
                f"{size:,} B addressed by key beats "
                f"{cross.get('wide_state_bytes', 0):,} B that is not. The size "
                f"starts to matter only when more than one fact is live, which "
                f"is the capacity table below."
            )
        if cross.get("store_fails_at_distance") is None:
            bullets.append(
                f"**The store does not fail at any distance measured** (to "
                f"{cross.get('longest_distance')}) — it is "
                f"{cross.get('store_bytes', 0):,} B and constant. Its failure "
                f"mode is capacity, not distance: "
                f"{cross.get('store_exact_to_facts')} facts fit in the eight "
                f"slots and the "
                f"{cross.get('store_first_failure_at_facts')}th evicts the first. "
                f"Every extra fact costs {config.get('slot_bytes')} B, so holding "
                f"F facts exactly costs {config.get('slot_bytes')}×F bytes."
            )
        else:
            bullets.append(
                f"**The store starts failing at distance "
                f"{cross['store_fails_at_distance']}** at "
                f"{cross.get('store_bytes', 0):,} B."
            )
        if cross.get("transcript_exceeds_store_at_events"):
            kv = model.get("kv_cache_bytes_per_token")
            kv_text = (f" At the model's scale a single attention token of KV "
                       f"cache is {kv:,} B, so the agent's entire store is "
                       f"{max(1, cross.get('store_bytes', 0) // kv)} token"
                       f"{'' if max(1, cross.get('store_bytes', 0) // kv) == 1 else 's'} "
                       f"of it." if kv else "")
            bullets.append(
                f"**What the unbounded alternative costs.** The unbounded store "
                f"is {config.get('slot_bytes')} B a write and passes the fixed "
                f"store's {cross.get('store_bytes', 0):,} B at write "
                f"{cross.get('unbounded_store_exceeds_store_at_writes')}; a full "
                f"transcript at 8 B an event passes it at event "
                f"{cross.get('transcript_exceeds_store_at_events')}. Below those "
                f"lengths, keeping everything is *cheaper* than a bounded store "
                f"— and above them the bounded store is exact only up to its "
                f"capacity.{kv_text}"
            )
        lines += ["", "### The honest crossover", ""] + [
            f"* {b}" for b in bullets]

    if capacities:
        lines += [
            "",
            "### Capacity: what eight slots hold, and what a cheaper tag costs",
            "",
            f"Keys are `1..F` against `key % 8`, so the collision is **designed** "
            f"rather than drawn from a random key space: this is what a bounded "
            f"store does at a known load, not the birthday-paradox rate a wider "
            f"key space would give. The oldest fact is queried first because it "
            f"is the one a direct-mapped store evicts first.",
            "",
            "| condition | bytes carried | "
            + " | ".join(f"F={f}" for f in capacities) + " |",
            "|---|---:|" + "---:|" * len(capacities),
        ]
        capacity_conditions = config.get("capacity_conditions") or []
        for name in capacity_conditions:
            # The peak, not the first cell: an unbounded store grows with the
            # number of writes, and quoting its size at F=1 would understate it
            # by the width of the table.
            peak = max(capacity[str(f)]["conditions"][name]["state_bytes"]
                       + capacity[str(f)]["conditions"][name]["store_bytes"]
                       for f in capacities)
            lines.append(
                f"| {labels.get(name, name)} | {peak:,} | "
                + " | ".join(cell_text(capacity[str(f)]["conditions"][name])
                             for f in capacities)
                + " |"
            )

        biggest = capacities[-1]
        lines += [
            "",
            f"**Retrieval precision at F={biggest}** — what the loop did with "
            f"the answer. \"Wrong fact\" is a value stored under a *different* "
            f"key: the store handed over another fact confidently, which is worse "
            f"than a miss, and it is the failure the tag exists to prevent. "
            f"Precision is over answered runs, so refusing to answer cannot "
            f"raise it.",
            "",
            "| condition | correct | wrong fact | stale | no answer | precision |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in capacity_conditions:
            cell = capacity[str(biggest)]["conditions"][name]
            lines.append(
                f"| {labels.get(name, name)} | {cell['correct']} | "
                f"{cell['wrong_fact']} | {cell['stale']} | {cell['no_answer']} | "
                f"{cell['precision']:.3f} |"
            )

        first_fail = cross.get("store_first_failure_at_facts")
        if first_fail is not None and str(first_fail) in capacity:
            tagged = capacity[str(first_fail)]["conditions"]["store"]
            untagged = capacity[str(first_fail)]["conditions"].get("untagged_store")
            if untagged:
                lines += [
                    "",
                    f"At F={first_fail} the tagged store and the untagged one "
                    f"solve the same number "
                    f"({tagged['solved']}/{tagged['total']} against "
                    f"{untagged['solved']}/{untagged['total']}) and fail "
                    f"differently. The tagged store **misses**: its retrievals "
                    f"find the slot occupied by another key, report a miss, and "
                    f"the loop answers its default zero — "
                    f"{tagged['wrong_other']} of {tagged['total']} runs end in a "
                    f"wrong number rather than another fact's. The untagged store "
                    f"returns **another fact's value** in "
                    f"{untagged['wrong_fact']} of {untagged['total']}, which is "
                    f"the same solve rate and a worse failure — at "
                    f"{cross.get('untagged_store_bytes', 0):,} B instead of "
                    f"{cross.get('store_bytes', 0):,} B. The tag costs "
                    f"{cross.get('store_bytes', 0) - cross.get('untagged_store_bytes', 0):,} "
                    f"bytes in total, and it is the difference between a miss and "
                    f"a confident wrong answer.",
                ]

    if stale_distances:
        stale_conditions = config.get("stale_conditions") or []
        totals = {}
        for name in stale_conditions:
            agg = {"solved": 0, "total": 0, "correct": 0, "stale": 0,
                   "wrong_fact": 0, "wrong_other": 0, "no_answer": 0}
            for d in stale_distances:
                cell = stale[str(d)]["conditions"][name]
                for key in agg:
                    agg[key] += cell[key]
            totals[name] = agg
        lines += [
            "",
            "### The stale-fact control",
            "",
            f"A fact written, overwritten, and queried — "
            f"{sum(totals[name]['total'] for name in stale_conditions[:1])} tasks "
            f"over distances "
            f"{', '.join(str(d) for d in stale_distances)}. The answer is the "
            f"second value; a store that returns the first hands the loop "
            f"something that *was* true, which a caller cannot tell from "
            f"something that is.",
            "",
            "| condition | solved | returned the new value | returned the "
            "superseded value | no answer |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in stale_conditions:
            agg = totals[name]
            lines.append(
                f"| {labels.get(name, name)} | {agg['solved']}/{agg['total']} | "
                f"{agg['correct']} | {agg['stale']} | {agg['no_answer']} |"
            )

    trace = examples.get("recall")
    if trace:
        store_run = next((run for run in trace["conditions"]
                          if run["condition"] == "store"), None)
        if store_run:
            lines += [
                "",
                f"**The published trace** — `{trace['text']}`, answer "
                f"`{trace['answer']}`, table `{trace['table']}`, fact stored "
                f"under key `{trace['query_key']}`, {len(store_run['steps'])} "
                f"steps. \"Reads\" is the state the policy acted on, before the "
                f"call.",
                "",
                "| step | instruction | reads: op / arg / carry | action | result |",
                "|---:|---|---|---|---:|",
            ]
            for step in store_run["steps"]:
                reads = step["reads"]
                action = step["action"] + "(" + ", ".join(
                    f"{k}={v}" for k, v in step["args"].items()) + ")"
                shown = step["result"]
                if step.get("error"):
                    shown = f"`{step['error']}`"
                lines.append(
                    f"| {step['index']} | {step['instruction']} | "
                    f"{reads['op_code']} / {reads['arg']} / {reads['carry']:g} | "
                    f"`{action}` | {shown} |"
                )
            others = [run for run in trace["conditions"]
                      if run["condition"] != "store"]
            if others:
                summary = "; ".join(
                    f"`{run['condition']}` answers {run['answer']} "
                    f"({run['outcome']})" for run in others)
                lines += [
                    "",
                    f"The same task under the controls: {summary}. The working "
                    f"state answers with the last thing it was told — a wrong "
                    f"value, not an error — which is what makes the control a "
                    f"measurement of memory rather than of a broken loop.",
                ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqar")
    parser.add_argument("--scaling", action="append", default=[],
                        help="repeatable: one table per baseline")
    parser.add_argument("--control")
    parser.add_argument("--scan-inner")
    parser.add_argument("--voice",
                        help="voice-affect.json, which renders the voice block")
    parser.add_argument("--agent",
                        help="agent-loop.json, which renders the agent block")
    parser.add_argument("--memory",
                        help="long-memory.json, which renders the memory block")
    parser.add_argument("--length-extrapolation",
                        help="length-extrapolation.json, which renders the "
                             "length-extrapolation section of the results "
                             "block (required with --mqar)")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    if not (args.mqar or args.voice or args.agent or args.memory):
        parser.error("give --mqar (to render the results block), --voice "
                     "(to render the voice block), --agent (to render the agent "
                     "block), --memory (to render the memory block), or a "
                     "combination")

    readme = pathlib.Path(args.readme)
    text = readme.read_text()
    status = 0

    if args.mqar:
        mqar = _load(args.mqar, "the main sweep")
        if mqar is None:
            return 1
        scaling = [p for p in (_load(path, "a scaling run")
                               for path in args.scaling) if p]
        control = _load(args.control, "the control run")
        scan_inner = _load(args.scan_inner, "the inner-scan comparison")

        if not args.length_extrapolation:
            # Refuse rather than delete. The results block contains a section
            # this renderer has to be able to reproduce; rendering without it
            # would silently remove a published section, which is the failure
            # this flag exists to fix.
            print("refusing to write the results block without "
                  "--length-extrapolation: the length-extrapolation section is "
                  "part of it, and rendering without it would delete that "
                  "section rather than reproduce it", file=sys.stderr)
            return 1
        extrapolation = _load(args.length_extrapolation,
                              "the length-extrapolation run")
        if extrapolation is None:
            return 1

        if BEGIN not in text or END not in text:
            print(f"{args.readme} has no {BEGIN} / {END} block", file=sys.stderr)
            return 1
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)

        rendered = render(mqar, scaling, control, scan_inner, extrapolation)
        # Refuse to publish a table that lost its content: a broken renderer
        # produces an empty block, and a reader cannot tell an empty result from
        # a bug.
        for marker in ("| pairs in context |", "| sequence length |",
                       "### Does either model read longer than it trained?",
                       "| x train |"):
            if marker not in rendered:
                print(f"refusing to write: {marker!r} missing from the render",
                      file=sys.stderr)
                return 1

        text = f"{head}{BEGIN}\n{rendered}\n{END}{tail}"
        print(f"wrote {args.readme} results block "
              f"({len(rendered.splitlines())} lines)")

    if args.voice:
        voice = _load(args.voice, "the voice/affect run")
        if voice is None:
            return 1
        if VOICE_BEGIN not in text or VOICE_END not in text:
            print(f"{args.readme} has no {VOICE_BEGIN} / {VOICE_END} block",
                  file=sys.stderr)
            return 1
        head, rest = text.split(VOICE_BEGIN, 1)
        _, tail = rest.split(VOICE_END, 1)

        rendered = voice_section(voice)
        for marker in ("| condition |", "| control |"):
            if marker not in rendered:
                print(f"refusing to write: {marker!r} missing from the voice "
                      f"render", file=sys.stderr)
                return 1

        text = f"{head}{VOICE_BEGIN}\n{rendered}\n{VOICE_END}{tail}"
        print(f"wrote {args.readme} voice block "
              f"({len(rendered.splitlines())} lines)")

    if args.agent:
        agent = _load(args.agent, "the agent-loop run")
        if agent is None:
            return 1
        if AGENT_BEGIN not in text or AGENT_END not in text:
            print(f"{args.readme} has no {AGENT_BEGIN} / {AGENT_END} block",
                  file=sys.stderr)
            return 1
        head, rest = text.split(AGENT_BEGIN, 1)
        _, tail = rest.split(AGENT_END, 1)

        rendered = agent_section(agent)
        for marker in ("| task family |", "| condition |", "| A |",
                       "| memory slots |", "| distractor share |"):
            if marker not in rendered:
                print(f"refusing to write: {marker!r} missing from the agent "
                      f"render", file=sys.stderr)
                return 1

        text = f"{head}{AGENT_BEGIN}\n{rendered}\n{AGENT_END}{tail}"
        print(f"wrote {args.readme} agent block "
              f"({len(rendered.splitlines())} lines)")

    if args.memory:
        memory = _load(args.memory, "the long-context memory run")
        if memory is None:
            return 1
        if MEMORY_BEGIN not in text or MEMORY_END not in text:
            print(f"{args.readme} has no {MEMORY_BEGIN} / {MEMORY_END} block",
                  file=sys.stderr)
            return 1
        head, rest = text.split(MEMORY_BEGIN, 1)
        _, tail = rest.split(MEMORY_END, 1)

        rendered = memory_section(memory)
        for marker in ("| condition |", "| episode length |",
                       "### The honest crossover", "### The stale-fact control"):
            if marker not in rendered:
                print(f"refusing to write: {marker!r} missing from the memory "
                      f"render", file=sys.stderr)
                return 1

        text = f"{head}{MEMORY_BEGIN}\n{rendered}\n{MEMORY_END}{tail}"
        print(f"wrote {args.readme} memory block "
              f"({len(rendered.splitlines())} lines)")

    readme.write_text(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
