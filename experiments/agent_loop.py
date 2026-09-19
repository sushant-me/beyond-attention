"""Does the loop actually solve tasks, and what makes it do so?

This is the measurement behind the README's agent block. It runs one
hand-designed controller over a seeded task suite at several step budgets and
records the solve rate, then runs the controls that decide what the number
means. The controls are the point; a solve rate on its own is a number about the
person who wrote the tasks.

The task suite is closed and synthetic. Each task is a short plan over bounded
integers -- ``ADD``/``MUL``/``SUB``, a ``KEY`` step that swaps the running value
for the key paired with it, one ``IFPOS`` branch whose *tool* is chosen from the
state, and ``RET`` -- plus a bounded key/value table. The answer is computed by
``evaluate_plan`` in Python integers, so it is analytic: correctness is a
property of the task, not of a model.

The ``selective`` family is a different question. A stream of keyed events and
distractors arrives, then a query asks for the value of one key, and which
events are retained -- and where they land -- is decided by each event's own
content. Retention there has to depend on the input, which is what the
architecture's gate exists to do, and the controls below are what decide whether
the gate or merely the width is doing the work.

Controls, each of which can fail:

* **no memory** -- the state is wiped before every decision and only the current
  instruction is re-streamed. The instruction registers refill, so the agent
  still knows which tool to reach for; only the carried result is gone. This
  must collapse on every family whose answer is an intermediate result, and must
  **not** collapse on the family whose answer is written in the task text. That
  contrast is the measurement.
* **scalar carry** -- the same controller with the values held in one Python int
  instead of the state-space state. On the five arithmetic families it matches
  the full agent row for row, which is the honest statement that the recurrence
  earns nothing there. On ``selective`` it **fails**, and the failure is exact:
  it is right precisely when the queried key is the last one stored, because one
  register cannot hold two keys.
* **fixed decay** -- a memory of the *same width* as the gated one with a
  constant, input-independent write gate: every value-carrying event is written
  into every slot. This separates "the architecture" from "a sufficiently wide
  memory". If it tied the gated state, the gating would not be what does the
  work.
* **state width** -- the same selective tasks read by a state of 0, 1, 2, 3, 4
  and 8 memory slots. A one-slot state aliases every key onto one slot and *is*
  the scalar; the curve is where the family starts to solve.
* **distractor rate** -- the same family with 0, 1, 2, 4, 6 and 8 distractors in
  the stream. A memory claim that dies at the first distractor is not a memory
  claim.
* **random action** -- tools and arguments chosen uniformly. The floor.
* **budget ceiling** -- the same suite at budgets 1..6. Each family needs a known
  number of loop steps, so this is the ceiling the budget imposes, and the
  one-step column of it is the "is this suite trivially solvable in one step"
  control.

    python -u experiments/agent_loop.py --out agent-loop.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from beyond_attention.agent import (
    AGENT_DIM,
    A_HOLD,
    DELTA_WRITE,
    FAMILIES,
    OP_PUT,
    OP_RECALL,
    RANDOM_ARG_SPAN,
    REGISTER_NAMES,
    SELECTIVE_DISTRACTORS,
    SELECTIVE_KEYS,
    SELECTIVE_STORES,
    SELECTIVE_WIDTH,
    STEP_COUNTS,
    TOOL_NAMES,
    AgentRun,
    Task,
    example_task,
    replay,
    run_agent,
    selective_example_task,
    selective_suite,
    task_suite,
)

# The two sweeps. Widths are memory slots, not the whole state: the eight named
# registers are always there, and ``total_width`` in the payload records the sum
# so the two are not confused.
WIDTHS = (0, 1, 2, 3, 4, 8)
NOISE_COUNTS = (0, 1, 2, 4, 6, 8)


def solve_rate(runs: list[tuple[Task, AgentRun]]) -> dict:
    """Solved over attempted, with both counts kept so the rate can be checked."""
    solved = sum(1 for _, run in runs if run.solved)
    total = len(runs)
    return {
        "solved": solved,
        "total": total,
        "rate": solved / total if total else 0.0,
    }


def grid(
    tasks: tuple[Task, ...],
    budgets: list[int],
    memory: str = "ssm",
    policy: str = "plan",
    seed: int = 0,
    state_width: int | None = None,
) -> dict[str, dict]:
    """Solve rate for every (family, budget) cell, keyed ``family@budget``.

    ``state_width`` is passed through to the loop; ``None`` lets the loop size
    the memory to each task's own key space, which is what the main table wants.
    The width sweep passes an explicit value instead.
    """
    cells: dict[str, dict] = {}
    for family in FAMILIES:
        family_tasks = [t for t in tasks if t.family == family]
        for budget in budgets:
            runs = [
                (task, run_agent(task, budget=budget, memory=memory,
                                 policy=policy, seed=seed + index,
                                 state_width=state_width))
                for index, task in enumerate(family_tasks)
            ]
            cells[f"{family}@{budget}"] = solve_rate(runs)
    return cells


def condition_rate(tasks: tuple[Task, ...], **kwargs) -> dict:
    """Solve rate for one condition over one family."""
    return solve_rate([(task, run_agent(task, **kwargs)) for task in tasks])


def selective_shape(tasks_per_family: int, seed: int, n_store: int,
                    n_distractors: int, n_keys: int) -> tuple[Task, ...]:
    return selective_suite(tasks_per_family, seed, n_store=n_store,
                           n_distractors=n_distractors, n_keys=n_keys)


def width_sweep(tasks_per_family: int, seed: int, n_keys: int,
                n_store: int, n_distractors: int) -> dict[str, dict]:
    """The selective family read by states of increasing memory width.

    The same fifty tasks at every width, so the curve is about the state and
    nothing else. The scalar column is constant by construction -- it replaces
    the memory, so the width cannot reach it -- and it is repeated in every row
    rather than stated once, because a constant next to a rising curve is the
    comparison the table exists to make.
    """
    budget = n_store + n_distractors + 1
    tasks = selective_shape(tasks_per_family, seed, n_store, n_distractors,
                            n_keys)
    rows: dict[str, dict] = {}
    for width in WIDTHS:
        rows[str(width)] = {
            "slots": width,
            "total_width": AGENT_DIM + width,
            "agent": condition_rate(tasks, budget=budget, state_width=width),
            "scalar": condition_rate(tasks, budget=budget, memory="scalar"),
            "fixed": condition_rate(tasks, budget=budget, memory="fixed",
                                    state_width=width),
            "none": condition_rate(tasks, budget=budget, memory="none",
                                   state_width=width),
        }
    return rows


def distractor_sweep(tasks_per_family: int, seed: int, n_keys: int,
                     n_store: int) -> dict[str, dict]:
    """The selective family with more and more of the stream discarded.

    Every point is a different task set, because the number of distractors is
    part of the shape -- so the four conditions are run on the same tasks as
    each other at each point, and the budget grows with the stream.
    """
    rows: dict[str, dict] = {}
    for n_distractors in NOISE_COUNTS:
        budget = n_store + n_distractors + 1
        tasks = selective_shape(tasks_per_family, seed, n_store, n_distractors,
                                n_keys)
        rows[str(n_distractors)] = {
            "distractors": n_distractors,
            "events": n_store + n_distractors,
            "rate": n_distractors / (n_store + n_distractors),
            "budget": budget,
            "agent": condition_rate(tasks, budget=budget, state_width=n_keys),
            "scalar": condition_rate(tasks, budget=budget, memory="scalar"),
            "fixed": condition_rate(tasks, budget=budget, memory="fixed",
                                    state_width=n_keys),
            "none": condition_rate(tasks, budget=budget, memory="none",
                                   state_width=n_keys),
        }
    return rows


def scalar_selective_correspondence(tasks: tuple[Task, ...],
                                    budget: int) -> dict:
    """Per-task: is the scalar right exactly when the query names the last store?

    A rate below 1.0 would be consistent with a merely harder task. The
    correspondence is the argument -- one register, so it is right precisely
    when the key it is asked about is the key it holds last -- and it is
    recorded per task rather than summarised so that a single mismatch is
    visible.
    """
    solved = 0
    queried_the_last_store = 0
    exact = 0
    for task in tasks:
        stored = [i.args[0] for i in task.plan if i.op == OP_PUT]
        queried = [i.args[0] for i in task.plan if i.op == OP_RECALL][0]
        holds_the_answer = queried == stored[-1]
        run = run_agent(task, budget=budget, memory="scalar")
        solved += int(run.solved)
        queried_the_last_store += int(holds_the_answer)
        exact += int(bool(run.solved) == holds_the_answer)
    return {
        "tasks": len(tasks),
        "solved": solved,
        "queried_the_last_store": queried_the_last_store,
        "exact_matches": exact,
        "correspondence_holds": exact == len(tasks),
    }


def random_grid(
    tasks: tuple[Task, ...],
    budget: int,
    rounds: int,
    seed: int,
) -> dict[str, dict]:
    """The random-action floor, over ``rounds`` independent draws per task.

    Every draw is a trial, because a random agent that happens to stumble onto
    an answer on its third attempt has not solved the task; counting per draw
    rather than per task is what keeps the floor from being inflated by
    retries.
    """
    cells: dict[str, dict] = {}
    for family in FAMILIES:
        family_tasks = [t for t in tasks if t.family == family]
        runs: list[tuple[Task, AgentRun]] = []
        for index, task in enumerate(family_tasks):
            for r in range(rounds):
                run = run_agent(task, budget=budget, policy="random",
                                seed=seed + index * 1_000 + r)
                runs.append((task, run))
        cells[f"{family}@{budget}"] = solve_rate(runs)
    return cells


def write_residual(a_hold: float, first: float = 9.0, second: float = 0.0) -> float:
    """What a register holds after being written 9 and then 0, for a given A.

    This mirrors the register arithmetic exactly -- a write multiplies the old
    contents by ``exp(delta * A)`` and adds the new value -- and it is measured
    rather than asserted because a residual above zero is a bug rather than a
    rounding detail: ``IFPOS`` branches on the sign of this number, and the
    first version of the module, at ``A = -50``, took the wrong arm on 8 of 50
    branch tasks because 1.7e-21 is greater than zero.
    """
    held = np.exp(DELTA_WRITE * a_hold) * 0.0 + first
    held = np.exp(DELTA_WRITE * a_hold) * held + second
    return float(held)


def trace_payload(run: AgentRun, task: Task) -> dict:
    """The example run, step by step, including what the state held.

    The registers are in the payload because the claim being published is about
    *where the policy read its operand from*. A trace that showed only the
    actions would be consistent with a controller that had read the operand out
    of the task text.
    """
    return {
        "task_id": task.task_id,
        "text": task.text,
        "plan": [{"op": i.op, "args": list(i.args)} for i in task.plan],
        "table": [list(pair) for pair in task.table],
        "answer": task.answer,
        "budget": run.budget,
        "stop_reason": run.stop_reason,
        "solved": run.solved,
        "steps": [
            {
                "index": step.index,
                "instruction": f"{step.instruction.op}"
                               + (f" {step.instruction.args[0]}"
                                  if step.instruction.args else ""),
                "reads": {
                    "op_code": step.registers.op_code,
                    "arg": step.registers.arg,
                    "carry": round(step.registers.carry, 6),
                    "observations": step.registers.observations,
                    "instructions": step.registers.instructions,
                    # Empty for the arithmetic families, which have no memory
                    # slots; the selective trace is read out of this field.
                    "slots": [round(v, 6) for v in step.registers.slots],
                },
                "action": step.action.tool,
                "args": dict(sorted(step.action.args.items())),
                "result": step.result.value,
                "error": step.result.error,
            }
            for step in run.steps
        ],
        "replayed": [
            {"value": r.value, "error": r.error} for r in replay(run, task.table)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--random-rounds", type=int, default=25,
                        help="random-action draws per task")
    parser.add_argument("--out", default="agent-loop.json")
    args = parser.parse_args()

    started = time.perf_counter()
    tasks = task_suite(args.tasks_per_family, args.seed)
    max_budget = max(args.budgets)

    # --- the agent --------------------------------------------------------
    solve = grid(tasks, args.budgets)

    # --- controls ---------------------------------------------------------
    no_memory = grid(tasks, args.budgets, memory="none")
    scalar_carry = grid(tasks, args.budgets, memory="scalar")
    fixed_decay = grid(tasks, args.budgets, memory="fixed")
    random_action = random_grid(tasks, max_budget, args.random_rounds, args.seed)

    # --- the selective family's own sweeps --------------------------------
    widths = width_sweep(args.tasks_per_family, args.seed, SELECTIVE_KEYS,
                         SELECTIVE_STORES, SELECTIVE_DISTRACTORS)
    distractors = distractor_sweep(args.tasks_per_family, args.seed,
                                   SELECTIVE_KEYS, SELECTIVE_STORES)

    selective_tasks = selective_suite(args.tasks_per_family, args.seed)
    selective_budget = STEP_COUNTS["selective"]
    correspondence = scalar_selective_correspondence(selective_tasks,
                                                     selective_budget)

    # The fixed-decay state is a constant gate, so with nothing to discard it
    # collapses onto the scalar exactly. Asserted here rather than described,
    # because it is the statement that separates the two controls.
    calm = selective_suite(args.tasks_per_family, args.seed,
                           n_distractors=0)
    fixed_equals_scalar = all(
        run_agent(t, budget=SELECTIVE_STORES + 1, memory="fixed",
                  state_width=SELECTIVE_KEYS).solved
        == run_agent(t, budget=SELECTIVE_STORES + 1, memory="scalar").solved
        for t in calm
    )

    # The scalar control is right exactly when the query names the last store,
    # so it cannot be right on every selective task -- and it is wrong on the
    # published example for the same reason.
    if correspondence["solved"] == correspondence["tasks"]:
        raise AssertionError("the scalar solved the selective family: the family "
                             "does not require a second register")
    if not correspondence["correspondence_holds"]:
        raise AssertionError("the scalar control is not the predicted register")

    # The one-step agent is the budget-1 column of the main grid, pulled out
    # rather than re-run so the two cannot drift apart.
    one_step = {
        family: dict(solve[f"{family}@1"]) for family in FAMILIES
    }
    one_step["all"] = solve_rate(
        [(t, run_agent(t, budget=1, seed=i))
         for i, t in enumerate(tasks)]
    )

    example = example_task()
    example_run = run_agent(example, budget=max_budget)
    # The trace must replay to the same results, or it is not a trace.
    replayed = replay(example_run, example.table)
    for step, again in zip(example_run.steps, replayed):
        assert (step.result.value, step.result.error) == (again.value, again.error), \
            "the example trace does not replay"

    write_exactness = {
        f"{a:.0f}": {
            "multiplier": float(np.exp(a)),
            "residual_after_overwrite": write_residual(a),
            "sign_test_holds": not write_residual(a) > 0,
        }
        for a in (-50.0, -400.0, -745.0, A_HOLD)
    }

    # How many branch tasks reach the branch with a carry of exactly zero. That
    # is the value a write residual turns positive: at A = -50 the register held
    # 1.7e-21 instead of 0.0, `IFPOS` read it as positive, and these are exactly
    # the tasks it got wrong. Measured rather than asserted, because "it only
    # affected an edge case" is the kind of claim that should be countable.
    branch_tasks = [t for t in tasks if t.family == "branch"]
    zero_carry_branch = {
        "count": sum(
            1 for t in branch_tasks
            if run_agent(t, budget=max_budget).steps[2].registers.carry == 0.0
        ),
        "total": len(branch_tasks),
        "step": 2,
    }

    controls = {
        "zero_carry_branch": zero_carry_branch,
        "no_memory": no_memory,
        "scalar_carry": scalar_carry,
        "fixed_decay": fixed_decay,
        "random_action": random_action,
        "one_step": one_step,
        "write_exactness": write_exactness,
        "state_width": widths,
        "distractor_rate": distractors,
        "scalar_selective": correspondence,
        "fixed_equals_scalar_at_zero_distractors": bool(fixed_equals_scalar),
    }

    selective_example = selective_example_task()
    selective_run = run_agent(selective_example, budget=len(selective_example.plan))
    selective_replayed = replay(selective_run, selective_example.table)
    for step, again in zip(selective_run.steps, selective_replayed):
        assert (step.result.value, step.result.error) == (again.value, again.error), \
            "the selective example trace does not replay"

    payload = {
        "config": {
            "tasks_per_family": args.tasks_per_family,
            "seed": args.seed,
            "budgets": args.budgets,
            "families": list(FAMILIES),
            "step_counts": dict(STEP_COUNTS),
            "tools": list(TOOL_NAMES),
            "registers": list(REGISTER_NAMES),
            "random_rounds": args.random_rounds,
            "random_arg_span": RANDOM_ARG_SPAN,
            "suite_size": len(tasks),
            "selective": {
                "stores": SELECTIVE_STORES,
                "distractors": SELECTIVE_DISTRACTORS,
                "keys": SELECTIVE_KEYS,
                "state_width": SELECTIVE_WIDTH,
                "step_count": STEP_COUNTS["selective"],
                "widths": list(WIDTHS),
                "noise_counts": list(NOISE_COUNTS),
            },
        },
        "solve_rate": solve,
        "controls": controls,
        "example_trace": trace_payload(example_run, example),
        "selective_example_trace": trace_payload(selective_run, selective_example),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    # --- human-readable summary ------------------------------------------
    families = list(FAMILIES)
    print(f"{len(tasks)} tasks, {args.tasks_per_family} per family, "
          f"seed {args.seed}")
    print()
    header = f"{'family':<10}{'steps':>6}" + "".join(
        f"{'b' + str(b):>7}" for b in args.budgets
    )
    print(header)
    for family in families:
        row = f"{family:<10}{STEP_COUNTS[family]:>6}"
        row += "".join(f"{solve[f'{family}@{b}']['rate']:>7.3f}"
                       for b in args.budgets)
        print(row)
    print()
    print(f"per-family solve rate at budget {max_budget} "
          f"(columns: " + " ".join(families) + "), and overall:")
    for name, cells in (
        ("no memory", no_memory),
        ("scalar carry", scalar_carry),
        ("fixed decay", fixed_decay),
        ("random action", random_action),
        ("agent", solve),
    ):
        at_max = [cells[f"{family}@{max_budget}"] for family in families]
        solved = sum(c["solved"] for c in at_max)
        total = sum(c["total"] for c in at_max)
        row = f"  {name:<14}{solved / total:>6.3f}   "
        row += " ".join(f"{c['rate']:.3f}" for c in at_max)
        print(row)
    print()
    print("selective: scalar carry is right exactly when the query names the "
          "last store")
    print(f"  solved {correspondence['solved']}/{correspondence['tasks']}, "
          f"queried the last store {correspondence['queried_the_last_store']}, "
          f"correspondence holds={correspondence['correspondence_holds']}")
    print(f"  fixed decay == scalar with no distractors: {fixed_equals_scalar}")
    print()
    print(f"{'slots':>6}{'total':>7}{'agent':>8}{'scalar':>8}{'fixed':>8}{'none':>8}")
    for row in widths.values():
        print(f"{row['slots']:>6}{row['total_width']:>7}"
              f"{row['agent']['rate']:>8.3f}{row['scalar']['rate']:>8.3f}"
              f"{row['fixed']['rate']:>8.3f}{row['none']['rate']:>8.3f}")
    print()
    print(f"{'noise':>6}{'rate':>7}{'agent':>8}{'scalar':>8}{'fixed':>8}{'none':>8}")
    for row in distractors.values():
        print(f"{row['distractors']:>6}{row['rate']:>7.3f}"
              f"{row['agent']['rate']:>8.3f}{row['scalar']['rate']:>8.3f}"
              f"{row['fixed']['rate']:>8.3f}{row['none']['rate']:>8.3f}")
    print()
    print(f"one-step agent (budget 1), all families: "
          f"{one_step['all']['rate']:.3f}")
    print("write exactness (residual after overwriting 9 with 0):")
    for a, row in write_exactness.items():
        print(f"  A={a:>5}  exp(A)={row['multiplier']:.3e}  "
              f"residual={row['residual_after_overwrite']:.3e}  "
              f"sign test holds={row['sign_test_holds']}")
    print()
    print(f"example: {example.text}")
    for step in example_run.steps:
        print(f"  {step.index}  carry={step.registers.carry:>6.1f} "
              f"{step.action.tool}{tuple(sorted(step.action.args.items()))} "
              f"-> {step.result.value}")
    print(f"  stop={example_run.stop_reason} solved={example_run.solved} "
          f"answer={example_run.answer}")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
