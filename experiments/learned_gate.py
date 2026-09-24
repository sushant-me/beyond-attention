"""Is the selective family's write gate learnable, or is it still hand-set?

``agent.py`` decides which memory slot a value lands in with a gate written out
by hand in ``embed``: ``one_hot(key % state_width)`` for a ``PUT`` and ``zeros``
for a ``NOISE``. The README's own strongest self-criticism is that the gate "is
not learned, it is not produced by a projection, and a Python dict in
``evaluate_plan`` computes the same answers with no model at all". This script
measures how much of that survives when the gate is a single ``nn.Linear`` over
one-hot features, trained by gradient descent on the *state*.

Every row runs the agent's own ``choose_action`` on the decoded state, so the
reader is identical everywhere and a difference between rows is a difference in
the gate. The conditions, all on the same tasks:

* **hand-set gate** -- the existing behaviour, through ``run_agent``.
* **hand-set gate through the learned path** -- ``run_learned`` with the gate
  forced to the hand-set one-hot. It must equal the row above task by task; that
  equality is what makes the learned path the same mechanism rather than a
  lookalike.
* **fixed / constant gate** -- the repository's own ``memory="fixed"`` control:
  a state of the same width whose write is input-independent.
* **learned gate, trained** -- the gate the optimiser produced, used raw.
* **learned gate, trained, sharpened** -- the same gate at a sigmoid temperature
  of 0.05, which pushes it to a hard 0/1.
* **learned gate, trained with the temperature annealed** during training.
* **learned gate, untrained** -- random init, same architecture, both
  initialisations, both temperatures. This is the control that says training is
  doing the work.
* **saturation weights** -- weights constructed so the layer's sigmoid is
  *exactly* the hand-set gate. This says the architecture can represent it; it
  says nothing about whether training finds it.

Three further measurements, because they are where the honest answer lives:

* **held-out keys** -- train on tasks whose keys come from half the vocabulary
  and evaluate on tasks whose keys come from the other half. Reported for the
  prescribed key one-hot, and for an address one-hot (``key % state_width``)
  that shares weights between keys that address the same slot, so the cause of
  whatever happens is visible rather than inferred.
* **gate sharpness** -- the mean and maximum deviation of ``w`` from 0/1, and
  what the deviation costs: under ``A_HOLD`` a hold multiplies the slot by
  ``exp(-800 * w)``, so ``w = 0.01`` already leaves 0.03% of it. A soft gate is
  not a gentle blend.
* **the search itself** -- the initialisation, the loss shape (per-event or
  final-only) and the annealing schedule each change the outcome, so every
  configuration that was run is in the payload with its own numbers.
* **the NOISE-operand feature** -- whether the gate still holds a distractor at
  zero when the distractor's value arrives along the same key dimension a
  ``PUT`` key would. The default keeps the value out of the key block; the
  ablation puts it in and reports what it costs.

    python -u experiments/learned_gate.py --out learned-gate.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from beyond_attention.agent import (
    A_HOLD,
    OP_PUT,
    SELECTIVE_DISTRACTORS,
    SELECTIVE_KEYS,
    SELECTIVE_STORES,
    Event,
    Task,
    instruction_event,
    run_agent,
    selective_suite,
)
from beyond_attention.learned_gate import (
    HOLD_INIT,
    MIDPOINT_INIT,
    HandSetGate,
    LearnedGate,
    at_temperature,
    learned_embed_matches_hand_set,
    run_learned,
    saturation_weights,
    sharpness,
    train_gate,
    value_instructions,
)

# The main family is the repository's own shape, so the tasks are the ones the
# README's selective table is about.
TRAIN_TASKS = 64
EVAL_TASKS = 50
SEEDS = (0, 1, 2, 3, 4)
TASK_SEEDS = (0, 1, 2, 3, 4)
STEPS = 2000
LR = 0.05
TEMPERATURES = (1.0, 0.5, 0.2, 0.1, 0.05)
SHARP_TEMPERATURE = 0.05

# Held-out keys need a vocabulary larger than the address space, so that "a key
# never seen in training" is a distinct thing from "a slot never addressed".
# 32 keys over 16 slots: the training pool is 0..15 (every slot addressed) and
# the held-out pool is 16..31 (every slot addressed, by keys the gate has never
# seen). Keys are drawn without replacement, so within either pool no two stored
# keys share a slot and no task is unsolvable by construction.
HELD_KEYS = 32
HELD_WIDTH = 16
HELD_TRAIN_TASKS = 256
HELD_EVAL_TASKS = 128
HELD_SEEDS = (0, 1, 2, 3, 4)


def summarise(rates: list[float], totals: list[int] | None = None) -> dict:
    """Mean, spread and the per-seed rates, with the counts kept."""
    if not rates:
        raise ValueError("no rates to summarise")
    return {
        "rates": [round(r, 6) for r in rates],
        "mean": round(statistics.fmean(rates), 6),
        "min": round(min(rates), 6),
        "max": round(max(rates), 6),
        "stdev": round(statistics.stdev(rates), 6) if len(rates) > 1 else 0.0,
        "tasks": totals[0] if totals else None,
    }


SHARPNESS_FIELDS = ("mean_deviation", "max_deviation", "min_deviation",
                    "rounded_one_hot_fraction",
                    "mean_weight_on_addressed_slots",
                    "mean_weight_on_held_slots")


def mean_fields(rows: list[dict]) -> dict:
    """Average the named fields over several gates, keeping the shape."""
    averaged = {
        field: round(statistics.fmean(row[field] for row in rows), 6)
        for field in SHARPNESS_FIELDS
    }
    averaged["events"] = rows[0]["events"]
    averaged["weights"] = rows[0]["weights"]
    averaged["written_slots"] = rows[0]["written_slots"]
    averaged["held_slots"] = rows[0]["held_slots"]
    return averaged


def solve_rate(tasks: tuple[Task, ...], gate, n_keys: int,
               state_width: int) -> tuple[float, int]:
    """Solved fraction for one gate over one task set, through ``run_learned``."""
    solved = sum(run_learned(t, gate, n_keys, state_width=state_width).solved
                 for t in tasks)
    return solved / len(tasks), len(tasks)


def agent_rate(tasks: tuple[Task, ...], **kwargs) -> tuple[float, int]:
    """Solved fraction for the agent's own loop -- the hand-set path."""
    solved = sum(run_agent(t, **kwargs).solved for t in tasks)
    return solved / len(tasks), len(tasks)


def noise_hold_fraction(gate, tasks: tuple[Task, ...], state_width: int,
                        temperature: float = 1.0) -> float:
    """Fraction of ``NOISE`` events whose rounded gate is all zeros.

    The ablation's question in one number: with the noise operand present as a
    key feature, does the gate still hold every slot when the event is a
    distractor?
    """
    hard = total = 0
    for task in tasks:
        for instr in value_instructions(task):
            if instr.op != "NOISE":
                continue
            w = np.asarray(gate.gate(instr, temperature), dtype=np.float64)
            hard += int(np.array_equal(np.round(w), np.zeros(state_width)))
            total += 1
    if not total:  # pragma: no cover - defensive
        raise ValueError("no NOISE events to measure")
    return hard / total


def hold_cost(gate, tasks: tuple[Task, ...], n_keys: int, state_width: int,
              temperature: float) -> dict:
    """What a soft gate costs the slots it is supposed to leave alone.

    For every value-carrying event and every slot that event should *not* write,
    the hold multiplier is ``exp(A_HOLD * w)``. ``w = 0`` is 1.0 (the slot is
    untouched); anything above a few thousandths is a decay the reader has to
    survive. Reported as the mean multiplier, the worst one, and the fraction of
    held slots that lose more than 1% of their contents in a single event.
    """
    multipliers: list[float] = []
    for task in tasks:
        for instr in value_instructions(task):
            addressed = (instr.args[0] % state_width) if instr.op == OP_PUT else None
            w = np.asarray(gate.gate(instr, temperature), dtype=np.float64)
            for slot in range(state_width):
                if slot == addressed:
                    continue
                with np.errstate(under="ignore"):
                    multipliers.append(float(np.exp(A_HOLD * w[slot])))
    values = np.array(multipliers, dtype=np.float64)
    return {
        "held_slots": int(values.size),
        "mean_hold_multiplier": float(values.mean()),
        "min_hold_multiplier": float(values.min()),
        "fraction_holds_destroyed": float((values < 0.99).mean()),
        "note": "exp(A_HOLD * w) for slots the event should not write",
    }


def soft_gate_cost_block(gates: dict, tasks: tuple[Task, ...], n_keys: int,
                         state_width: int) -> dict:
    return {
        name: hold_cost(gate, tasks, n_keys, state_width, temperature)
        for name, (gate, temperature) in gates.items()
    }


def trained_gate(tasks: tuple[Task, ...], n_keys: int, state_width: int, *,
                 seed: int, init: str = HOLD_INIT, dense: bool = True,
                 anneal_to: float | None = None, steps: int = STEPS,
                 key_mode: str = "key", noise_operand: bool = False):
    return train_gate(tasks, n_keys, state_width, seed=seed, steps=steps, lr=LR,
                      init=init, key_mode=key_mode, dense=dense,
                      noise_operand=noise_operand, anneal_to=anneal_to)


def differ(a, b, path: str = "") -> list[str]:
    """Paths at which two result structures disagree, as readable strings."""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"only in the fresh run: {path}/{key}")
            elif key not in b:
                out.append(f"only in the committed file: {path}/{key}")
            else:
                out += differ(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"length differs ({len(a)} vs {len(b)}): {path}")
        for index, (x, y) in enumerate(zip(a, b)):
            out += differ(x, y, f"{path}[{index}]")
    elif a != b:
        out.append(f"{path}: committed={a!r} fresh={b!r}")
    return out


def verify_against(path: Path, payload: dict) -> int:
    """Compare a fresh run against the committed results file, and do not write.

    The tests check that the committed file agrees with *itself*. That is a
    different claim from agreeing with the code, and the gap is not theoretical:
    four ``sharpness`` fields were added to the held-out block and the file was
    not regenerated, so the published numbers came from an older revision of this
    script while the tests stayed green. "The file parses and its numbers are
    internally consistent" cannot see that.

    ``wall_seconds`` is excluded because it is a property of the machine rather
    than of the result.
    """
    if not path.exists():
        print(f"{path} does not exist; nothing to verify against", file=sys.stderr)
        return 1

    def scrub(obj):
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items() if k != "wall_seconds"}
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        return obj

    committed = scrub(json.loads(path.read_text()))
    fresh = scrub(payload)

    if committed == fresh:
        print(f"{path} matches a fresh run, apart from wall_seconds")
        return 0

    print(f"{path} does NOT match a fresh run:", file=sys.stderr)
    for line in differ(committed, fresh):
        print(f"  {line}", file=sys.stderr)
    print("\nregenerate it with: python experiments/learned_gate.py",
          file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="learned-gate.json")
    parser.add_argument(
        "--verify", action="store_true",
        help="compare a fresh run against the file at --out instead of writing it")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--task-seeds", type=int, nargs="+",
                        default=list(TASK_SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--train-tasks", type=int, default=TRAIN_TASKS)
    parser.add_argument("--eval-tasks", type=int, default=EVAL_TASKS)
    parser.add_argument("--held-train-tasks", type=int,
                        default=HELD_TRAIN_TASKS)
    parser.add_argument("--held-eval-tasks", type=int, default=HELD_EVAL_TASKS)
    args = parser.parse_args()

    started = time.perf_counter()
    keys = SELECTIVE_KEYS
    width = SELECTIVE_KEYS
    family = dict(n_store=SELECTIVE_STORES, n_distractors=SELECTIVE_DISTRACTORS,
                  n_keys=keys)

    train_tasks = selective_suite(args.train_tasks, seed=0, **family)
    eval_tasks = selective_suite(args.eval_tasks, seed=1, **family)
    budget = len(eval_tasks[0].plan)

    handset = HandSetGate(keys, width)

    # --- row 1 and 2: the hand-set gate, both paths -----------------------
    hand_set_rate, hand_set_total = agent_rate(eval_tasks, budget=budget)
    learned_path_rate, learned_path_total = solve_rate(
        eval_tasks, handset.gate, keys, width)

    # The forced gate's *embedding* must equal agent.embed's, bit for bit, on
    # every event of every evaluation task. A rate can coincide; the arrays
    # cannot, unless this is the same mechanism.
    events: list[Event] = [instruction_event(i) for t in eval_tasks
                           for i in t.plan]
    embed_matches = learned_embed_matches_hand_set(events, handset.gate, keys,
                                                   width)
    if not embed_matches:
        raise AssertionError("the forced hand-set gate does not reproduce "
                             "agent.embed bit for bit")
    for task in eval_tasks:
        with_state = run_agent(task, budget=budget)
        with_gate = run_learned(task, handset.gate, keys, state_width=width)
        if (with_state.solved, with_state.answer, with_state.actions) != \
                (with_gate.solved, with_gate.answer, with_gate.actions):
            raise AssertionError(
                f"{task.task_id}: the learned path does not reproduce the "
                f"hand-set loop"
            )

    # --- row 3: the fixed / constant gate ---------------------------------
    fixed_rate, fixed_total = agent_rate(
        eval_tasks, budget=budget, memory="fixed", state_width=width)

    # --- rows 4-6: the learned gate, trained ------------------------------
    trained: dict[int, LearnedGate] = {}
    trained_reports = []
    annealed: dict[int, LearnedGate] = {}
    annealed_reports = []
    for seed in args.seeds:
        gate, report = trained_gate(train_tasks, keys, width, seed=seed,
                                    steps=args.steps)
        trained[seed] = gate
        trained_reports.append(report)
        hardened, hard_report = trained_gate(
            train_tasks, keys, width, seed=seed, steps=args.steps,
            anneal_to=SHARP_TEMPERATURE)
        annealed[seed] = hardened
        annealed_reports.append(hard_report)

    trained_raw = [solve_rate(eval_tasks, at_temperature(trained[s], 1.0),
                              keys, width)[0] for s in args.seeds]
    trained_sharp = [solve_rate(
        eval_tasks, at_temperature(trained[s], SHARP_TEMPERATURE), keys,
        width)[0] for s in args.seeds]
    annealed_sharp = [solve_rate(
        eval_tasks, at_temperature(annealed[s], SHARP_TEMPERATURE), keys,
        width)[0] for s in args.seeds]

    # --- rows 7-8: the untrained controls ---------------------------------
    untrained_midpoint_raw, untrained_midpoint_sharp = [], []
    untrained_hold_raw, untrained_hold_sharp = [], []
    untrained = {}
    for seed in args.seeds:
        for name, init, raw_list, sharp_list in (
            ("midpoint", MIDPOINT_INIT, untrained_midpoint_raw,
             untrained_midpoint_sharp),
            ("hold", HOLD_INIT, untrained_hold_raw, untrained_hold_sharp),
        ):
            gate = LearnedGate(keys, width, seed=seed, init=init)
            untrained[(name, seed)] = gate
            raw_list.append(solve_rate(eval_tasks, at_temperature(gate, 1.0),
                                       keys, width)[0])
            sharp_list.append(solve_rate(
                eval_tasks, at_temperature(gate, SHARP_TEMPERATURE), keys,
                width)[0])

    if max(untrained_hold_sharp) >= hand_set_rate:
        raise AssertionError("the untrained control matches the hand-set gate: "
                             "it is not a control")
    if min(trained_sharp) <= max(untrained_hold_sharp):
        raise AssertionError("training did not do anything the untrained gate "
                             "did not")

    # --- row 9: the architecture can represent the hand-set gate ----------
    weight, bias = saturation_weights(keys, width)
    saturated = LearnedGate(keys, width, seed=0)
    with torch.no_grad():
        saturated.weight.copy_(torch.tensor(weight))
        saturated.bias.copy_(torch.tensor(bias))
    saturation_matches = all(
        np.array_equal(saturated.gate(instr, 1.0),
                       handset.gate(instr))
        for task in eval_tasks for instr in value_instructions(task)
    )
    saturated_rate, saturated_total = solve_rate(
        eval_tasks, at_temperature(saturated, 1.0), keys, width)

    # --- the temperature sweep --------------------------------------------
    temperature_sweep = {}
    for temperature in TEMPERATURES:
        per_seed = [solve_rate(eval_tasks,
                               at_temperature(trained[s], temperature),
                               keys, width)[0] for s in args.seeds]
        untrained_per_seed = [
            solve_rate(eval_tasks, at_temperature(untrained[("hold", s)],
                                                  temperature), keys, width)[0]
            for s in args.seeds
        ]
        temperature_sweep[f"{temperature:g}"] = {
            "temperature": temperature,
            "trained": summarise(per_seed),
            "untrained_hold": summarise(untrained_per_seed),
        }

    # --- task-suite seed spread: is the result one lucky task sample? ------
    gate_seed_zero, _ = trained_gate(train_tasks, keys, width, seed=0,
                                     steps=args.steps)
    task_seed_rates = []
    for task_seed in args.task_seeds:
        other_eval = selective_suite(args.eval_tasks, seed=task_seed, **family)
        task_seed_rates.append(solve_rate(
            other_eval, at_temperature(gate_seed_zero, SHARP_TEMPERATURE), keys,
            width)[0])

    # --- the search: init, loss shape, annealing ---------------------------
    variants = []
    for name, init, dense, anneal_to in (
        ("midpoint_init_final_only", MIDPOINT_INIT, False, None),
        ("midpoint_init_dense", MIDPOINT_INIT, True, None),
        ("hold_init_final_only", HOLD_INIT, False, None),
        ("hold_init_dense", HOLD_INIT, True, None),
        ("hold_init_dense_annealed", HOLD_INIT, True, SHARP_TEMPERATURE),
    ):
        raw_rates, sharp_rates, losses = [], [], []
        for seed in args.seeds[:2]:
            gate, report = trained_gate(train_tasks, keys, width, seed=seed,
                                        init=init, dense=dense,
                                        anneal_to=anneal_to, steps=args.steps)
            raw_rates.append(solve_rate(eval_tasks,
                                        at_temperature(gate, 1.0),
                                        keys, width)[0])
            sharp_rates.append(solve_rate(
                eval_tasks, at_temperature(gate, SHARP_TEMPERATURE), keys,
                width)[0])
            losses.append(round(report.final_loss, 6))
        variants.append({
            "condition": name,
            "init": init,
            "dense_loss": dense,
            "anneal_to": anneal_to,
            "raw": summarise(raw_rates),
            "sharpened": summarise(sharp_rates),
            "final_losses": losses,
        })

    # --- held-out keys -----------------------------------------------------
    held_suite = selective_suite(4000, seed=0, n_keys=HELD_KEYS)
    stored_keys = {
        task.task_id: [i.args[0] for i in task.plan if i.op == OP_PUT]
        for task in held_suite
    }
    train_pool = [t for t in held_suite
                  if all(k < HELD_WIDTH for k in stored_keys[t.task_id])]
    held_pool = [t for t in held_suite
                 if all(k >= HELD_WIDTH for k in stored_keys[t.task_id])]
    held_train = train_pool[:args.held_train_tasks]
    held_train_eval = train_pool[args.held_train_tasks:
                                 args.held_train_tasks + args.held_eval_tasks]
    held_eval = held_pool[:args.held_eval_tasks]
    if not held_train or not held_train_eval or not held_eval:
        raise AssertionError("the held-out key pools are empty")

    held_handset = HandSetGate(HELD_KEYS, HELD_WIDTH)
    held_handset_train = solve_rate(held_train_eval, held_handset.gate,
                                    HELD_KEYS, HELD_WIDTH)
    held_handset_held = solve_rate(held_eval, held_handset.gate, HELD_KEYS,
                                   HELD_WIDTH)
    held_block: dict = {
        "keys": HELD_KEYS,
        "state_width": HELD_WIDTH,
        "training_key_range": [0, HELD_WIDTH - 1],
        "held_out_key_range": [HELD_WIDTH, HELD_KEYS - 1],
        "training_tasks": len(held_train),
        "train_key_eval_tasks": len(held_train_eval),
        "held_out_eval_tasks": len(held_eval),
        "hand_set_train_keys": summarise([held_handset_train[0]],
                                         [held_handset_train[1]]),
        "hand_set_held_out_keys": summarise([held_handset_held[0]],
                                            [held_handset_held[1]]),
        "conditions": {},
    }
    held_gates: dict = {}
    for key_mode in ("key", "address"):
        raw_train, sharp_train, raw_held, sharp_held = [], [], [], []
        train_sharpness, held_sharpness = [], []
        for seed in HELD_SEEDS:
            gate, _ = trained_gate(held_train, HELD_KEYS, HELD_WIDTH,
                                   seed=seed, key_mode=key_mode,
                                   steps=args.steps)
            held_gates[(key_mode, seed)] = gate
            raw_train.append(solve_rate(held_train_eval,
                                        at_temperature(gate, 1.0),
                                        HELD_KEYS, HELD_WIDTH)[0])
            sharp_train.append(solve_rate(
                held_train_eval, at_temperature(gate, SHARP_TEMPERATURE),
                HELD_KEYS, HELD_WIDTH)[0])
            raw_held.append(solve_rate(held_eval, at_temperature(gate, 1.0),
                                       HELD_KEYS, HELD_WIDTH)[0])
            sharp_held.append(solve_rate(
                held_eval, at_temperature(gate, SHARP_TEMPERATURE),
                HELD_KEYS, HELD_WIDTH)[0])
            train_sharpness.append(sharpness(gate, held_train_eval, HELD_KEYS,
                                             HELD_WIDTH, temperature=1.0))
            held_sharpness.append(sharpness(gate, held_eval, HELD_KEYS,
                                            HELD_WIDTH, temperature=1.0))
        held_block["conditions"][key_mode] = {
            "feature_map": ("one dimension per key"
                            if key_mode == "key"
                            else "one dimension per slot (key % state_width)"),
            "raw_train_keys": summarise(raw_train),
            "sharpened_train_keys": summarise(sharp_train),
            "raw_held_out_keys": summarise(raw_held),
            "sharpened_held_out_keys": summarise(sharp_held),
            "sharpness_train_keys": mean_fields(train_sharpness),
            "sharpness_held_out_keys": mean_fields(held_sharpness),
        }

    # --- the NOISE-operand feature ablation --------------------------------
    # The default gives NOISE no key dimension because its operand is a value.
    # Whether the optimiser can still hold NOISE at zero when the feature is
    # there is an empirical question, so it is measured rather than argued.
    feature_ablation = {}
    for flag in (False, True):
        raw_rates, sharp_rates, losses, noise_hard = [], [], [], []
        sharpness_rows = []
        for seed in args.seeds:
            gate, report = trained_gate(train_tasks, keys, width, seed=seed,
                                        steps=args.steps,
                                        noise_operand=flag)
            raw_rates.append(solve_rate(eval_tasks,
                                        at_temperature(gate, 1.0), keys,
                                        width)[0])
            sharp_rates.append(solve_rate(
                eval_tasks, at_temperature(gate, SHARP_TEMPERATURE), keys,
                width)[0])
            losses.append(round(report.final_loss, 6))
            noise_hard.append(noise_hold_fraction(gate, eval_tasks, width))
            sharpness_rows.append(sharpness(gate, eval_tasks, keys, width,
                                            temperature=1.0))
        feature_ablation["noise_operand" if flag else "noise_only_opcode"] = {
            "noise_key_feature": flag,
            "raw": summarise(raw_rates, [hand_set_total] * len(args.seeds)),
            "sharpened": summarise(sharp_rates,
                                   [hand_set_total] * len(args.seeds)),
            "final_losses": losses,
            "noise_hard_zero_fraction": round(
                statistics.fmean(noise_hard), 6),
            "sharpness": mean_fields(sharpness_rows),
        }

    # --- sharpness, and what the softness costs ----------------------------
    sharp_block = {
        "temperature_1_0": {},
        "temperature_0_05": {},
    }
    for label, temperature in (("temperature_1_0", 1.0),
                               ("temperature_0_05", SHARP_TEMPERATURE)):
        entries = {
            "trained": [sharpness(trained[s], eval_tasks, keys, width,
                                  temperature=temperature)
                        for s in args.seeds],
            "untrained_hold": [sharpness(untrained[("hold", s)], eval_tasks,
                                         keys, width, temperature=temperature)
                               for s in args.seeds],
            "untrained_midpoint":
                [sharpness(untrained[("midpoint", s)], eval_tasks, keys, width,
                           temperature=temperature) for s in args.seeds],
            "hand_set_representation":
                [sharpness(saturated, eval_tasks, keys, width,
                           temperature=temperature)],
        }
        summary = {}
        for name, rows in entries.items():
            summary[name] = mean_fields(rows)
        sharp_block[label] = summary

    cost_block = soft_gate_cost_block({
        "trained_raw": (trained[args.seeds[0]], 1.0),
        "trained_sharpened": (trained[args.seeds[0]], SHARP_TEMPERATURE),
        "untrained_hold_raw": (untrained[("hold", args.seeds[0])], 1.0),
        "hand_set": (handset, 1.0),
    }, eval_tasks, keys, width)

    # --- agreement with the published results -----------------------------
    reference_tasks = selective_suite(args.eval_tasks, seed=0, **family)
    reference_budget = len(reference_tasks[0].plan)
    agent_py = agent_rate(reference_tasks, budget=reference_budget)
    learned_py = solve_rate(reference_tasks, handset.gate, keys, width)
    published = None
    published_path = Path("agent-loop.json")
    if published_path.exists():
        published = json.loads(published_path.read_text())["solve_rate"].get(
            f"selective@{reference_budget}", {}).get("rate")
    reference = {
        "tasks": agent_py[1],
        "task_seed": 0,
        "budget": reference_budget,
        "agent_py_rate": round(agent_py[0], 6),
        "experiment_hand_set_rate": round(learned_py[0], 6),
        "agent_loop_json_rate": published,
        "agree": agent_py[0] == learned_py[0] and (
            published is None or abs(published - agent_py[0]) < 1e-9),
    }
    if not reference["agree"]:
        raise AssertionError(f"the experiment disagrees with agent.py: {reference}")

    # --- the payload -------------------------------------------------------
    conditions = {
        "hand_set_gate": {
            "source": "run_agent(task, memory='ssm')",
            "rate": round(hand_set_rate, 6), "tasks": hand_set_total,
            "spread": summarise([hand_set_rate], [hand_set_total]),
        },
        "hand_set_gate_learned_path": {
            "source": "run_learned(task, HandSetGate)",
            "rate": round(learned_path_rate, 6), "tasks": learned_path_total,
            "embedding_matches_agent_embed": bool(embed_matches),
        },
        "fixed_constant_gate": {
            "source": "run_agent(task, memory='fixed', state_width=width)",
            "rate": round(fixed_rate, 6), "tasks": fixed_total,
        },
        "learned_gate_trained_raw": {
            "source": f"trained {args.steps} steps at temperature 1.0",
            "spread": summarise(trained_raw, [hand_set_total] * len(args.seeds)),
        },
        "learned_gate_trained_sharpened": {
            "source": f"trained, evaluated at temperature "
                      f"{SHARP_TEMPERATURE:g}",
            "spread": summarise(trained_sharp,
                                [hand_set_total] * len(args.seeds)),
        },
        "learned_gate_trained_annealed": {
            "source": f"trained with the temperature annealed to "
                      f"{SHARP_TEMPERATURE:g}, evaluated there",
            "spread": summarise(annealed_sharp,
                                [hand_set_total] * len(args.seeds)),
        },
        "learned_gate_untrained_midpoint_raw": {
            "spread": summarise(untrained_midpoint_raw,
                                [hand_set_total] * len(args.seeds)),
        },
        "learned_gate_untrained_midpoint_sharpened": {
            "spread": summarise(untrained_midpoint_sharp,
                                [hand_set_total] * len(args.seeds)),
        },
        "learned_gate_untrained_hold_raw": {
            "spread": summarise(untrained_hold_raw,
                                [hand_set_total] * len(args.seeds)),
        },
        "learned_gate_untrained_hold_sharpened": {
            "spread": summarise(untrained_hold_sharp,
                                [hand_set_total] * len(args.seeds)),
        },
        "saturation_weights": {
            "source": "constructed weights whose sigmoid is exactly the "
                      "hand-set gate",
            "rate": round(saturated_rate, 6), "tasks": saturated_total,
            "every_gate_vector_equals_hand_set": bool(saturation_matches),
        },
    }
    if not saturation_matches:
        raise AssertionError("the constructed weights are not the hand-set gate")

    payload = {
        "config": {
            "family": "selective",
            "stores": SELECTIVE_STORES,
            "distractors": SELECTIVE_DISTRACTORS,
            "keys": keys,
            "state_width": width,
            "budget": budget,
            "train_tasks": args.train_tasks,
            "train_task_seed": 0,
            "eval_tasks": args.eval_tasks,
            "eval_task_seed": 1,
            "seeds": list(args.seeds),
            "task_seeds": list(args.task_seeds),
            "steps": args.steps,
            "lr": LR,
            "optimiser": "Adam (full batch)",
            "loss": "mean squared error between the slot contents and the "
                    "state a correct gate produces, over every value event "
                    "(dense) or at the end alone",
            "init": HOLD_INIT,
            "temperatures": list(TEMPERATURES),
            "sharp_temperature": SHARP_TEMPERATURE,
            "features": "one-hot opcode ++ one-hot key (NOISE gets no key "
                        "dimension unless the ablation is on)",
            "gate": "nn.Linear(features, state_width) then sigmoid, float64",
            "held_out": {
                "keys": HELD_KEYS, "state_width": HELD_WIDTH,
                "training_tasks": args.held_train_tasks,
                "eval_tasks": args.held_eval_tasks,
                "seeds": list(HELD_SEEDS),
            },
        },
        "conditions": conditions,
        "temperature_sweep": temperature_sweep,
        "task_seed_spread": {
            "eval_seeds": list(args.task_seeds),
            "sharpened_rate": summarise(task_seed_rates,
                                        [hand_set_total] * len(task_seed_rates)),
        },
        "held_out_keys": held_block,
        "feature_ablation": feature_ablation,
        "sharpness": sharp_block,
        "soft_gate_cost": cost_block,
        "training_variants": variants,
        "training_reports": [
            {"kind": "raw", **report.__dict__} for report in trained_reports
        ] + [
            {"kind": "annealed", **report.__dict__}
            for report in annealed_reports
        ],
        "representation": {
            "saturation_weights_match_hand_set": bool(saturation_matches),
            "note": "the linear layer can represent the hand-set gate exactly; "
                    "whether gradient descent finds it is a separate question",
        },
        "reference_agreement": reference,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    target = Path(args.out)
    if args.verify:
        return verify_against(target, payload)
    target.write_text(json.dumps(payload, indent=2))

    # --- human-readable summary -------------------------------------------
    print(f"selective family: {args.eval_tasks} eval tasks (seed 1), "
          f"{args.train_tasks} training tasks (seed 0), "
          f"{args.steps} steps, {len(args.seeds)} seeds")
    print(f"the hand-set gate is one_hot(key % {width}) for PUT and 0 for NOISE; "
          f"the reader is agent.choose_action in every row")
    print()
    print(f"{'condition':<52}{'tasks':>6}{'rate':>8}{'min':>8}{'max':>8}"
          f"{'stdev':>8}")
    rows = [
        ("hand-set gate (agent.py)", conditions["hand_set_gate"], True),
        ("hand-set gate through the learned path",
         conditions["hand_set_gate_learned_path"], True),
        ("fixed / constant gate (memory='fixed')",
         conditions["fixed_constant_gate"], True),
        ("learned gate, trained, raw sigmoid",
         conditions["learned_gate_trained_raw"], False),
        ("learned gate, trained, sharpened",
         conditions["learned_gate_trained_sharpened"], False),
        ("learned gate, trained, annealed",
         conditions["learned_gate_trained_annealed"], False),
        ("learned gate, untrained (midpoint init), raw",
         conditions["learned_gate_untrained_midpoint_raw"], False),
        ("learned gate, untrained (midpoint init), sharpened",
         conditions["learned_gate_untrained_midpoint_sharpened"], False),
        ("learned gate, untrained (hold init), raw",
         conditions["learned_gate_untrained_hold_raw"], False),
        ("learned gate, untrained (hold init), sharpened",
         conditions["learned_gate_untrained_hold_sharpened"], False),
        ("saturation weights (exact hand-set gate)",
         conditions["saturation_weights"], True),
    ]
    for label, row, single in rows:
        if single:
            print(f"{label:<52}{row['tasks']:>6}{row['rate']:>8.3f}"
                  f"{'-':>8}{'-':>8}{'-':>8}")
        else:
            spread = row["spread"]
            print(f"{label:<52}{spread['tasks']:>6}{spread['mean']:>8.3f}"
                  f"{spread['min']:>8.3f}{spread['max']:>8.3f}"
                  f"{spread['stdev']:>8.3f}")
    print()
    print("temperature sweep (trained / untrained-hold), mean over seeds:")
    print(f"{'temperature':>12}{'trained':>10}{'untrained':>11}")
    for row in temperature_sweep.values():
        print(f"{row['temperature']:>12g}{row['trained']['mean']:>10.3f}"
              f"{row['untrained_hold']['mean']:>11.3f}")
    print()
    print(f"task-suite seeds {list(args.task_seeds)}: sharpened rate "
          f"{payload['task_seed_spread']['sharpened_rate']['mean']:.3f} "
          f"[{payload['task_seed_spread']['sharpened_rate']['min']:.3f}, "
          f"{payload['task_seed_spread']['sharpened_rate']['max']:.3f}]")
    print()
    print(f"held-out keys: vocabulary {HELD_KEYS}, width {HELD_WIDTH}, "
          f"trained on keys 0..{HELD_WIDTH - 1}, evaluated on keys "
          f"{HELD_WIDTH}..{HELD_KEYS - 1}")
    print(f"{'feature map':<34}{'train keys':>12}{'held-out keys':>15}")
    for name, row in held_block["conditions"].items():
        print(f"{row['feature_map']:<34}"
              f"{row['sharpened_train_keys']['mean']:>12.3f}"
              f"{row['sharpened_held_out_keys']['mean']:>15.3f}")
    print(f"  (hand-set gate on the same task sets: train keys "
          f"{held_handset_train[0]:.3f}, held-out keys "
          f"{held_handset_held[0]:.3f}; raw-sigmoid held-out rates are in the "
          f"payload)")
    print()
    print("NOISE-operand feature ablation (does the gate still hold NOISE at "
          "zero when its value arrives as a key feature?):")
    print(f"{'variant':<24}{'raw':>8}{'sharpened':>11}{'noise hard 0':>14}"
          f"{'final loss':>12}")
    for name, row in feature_ablation.items():
        print(f"{name:<24}{row['raw']['mean']:>8.3f}"
              f"{row['sharpened']['mean']:>11.3f}"
              f"{row['noise_hard_zero_fraction']:>14.3f}"
              f"{row['final_losses'][0]:>12.4f}")
    print()
    print("gate sharpness (mean deviation from 0/1, and the rounded match):")
    print(f"{'gate':<34}{'dev@1.0':>9}{'dev@0.05':>10}{'rounded':>9}"
          f"{'w(addr)':>9}{'w(held)':>9}")
    for name in ("trained", "untrained_hold", "untrained_midpoint",
                 "hand_set_representation"):
        a = sharp_block["temperature_1_0"][name]
        b = sharp_block["temperature_0_05"][name]
        print(f"{name:<34}{a['mean_deviation']:>9.4f}"
              f"{b['mean_deviation']:>10.4f}"
              f"{a['rounded_one_hot_fraction']:>9.3f}"
              f"{a['mean_weight_on_addressed_slots']:>9.3f}"
              f"{a['mean_weight_on_held_slots']:>9.3f}")
    print()
    print("what softness costs (hold multiplier exp(-800 w) on slots that "
          "should not be written):")
    print(f"{'gate':<24}{'mean':>10}{'worst':>10}{'fraction<0.99':>15}")
    for name, row in cost_block.items():
        print(f"{name:<24}{row['mean_hold_multiplier']:>10.4f}"
              f"{row['min_hold_multiplier']:>10.4f}"
              f"{row['fraction_holds_destroyed']:>15.3f}")
    print()
    print("the search (2 seeds each): init / loss shape / annealing")
    print(f"{'condition':<30}{'raw':>8}{'sharpened':>11}{'final loss':>12}")
    for row in variants:
        print(f"{row['condition']:<30}{row['raw']['mean']:>8.3f}"
              f"{row['sharpened']['mean']:>11.3f}"
              f"{row['final_losses'][0]:>12.4f}")
    print()
    print(f"representation: weights exist whose sigmoid is exactly the hand-set "
          f"gate -> {saturation_matches}; they solve at "
          f"{saturated_rate:.3f}")
    print(f"agreement: agent.py {reference['agent_py_rate']:.3f}, experiment "
          f"{reference['experiment_hand_set_rate']:.3f}, agent-loop.json "
          f"{reference['agent_loop_json_rate']}")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
