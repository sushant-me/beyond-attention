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

Controls, each of which can fail:

* **no memory** -- the state is wiped before every decision and only the current
  instruction is re-streamed. The instruction registers refill, so the agent
  still knows which tool to reach for; only the carried result is gone. This
  must collapse on every family whose answer is an intermediate result, and must
  **not** collapse on the family whose answer is written in the task text. That
  contrast is the measurement.
* **scalar carry** -- the same controller with the carried value held in one
  Python int instead of the state-space state. If this matches the full agent,
  the recurrence is not what makes the loop work, and the README says so.
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
    A_HOLD,
    DELTA_WRITE,
    FAMILIES,
    RANDOM_ARG_SPAN,
    REGISTER_NAMES,
    STEP_COUNTS,
    TOOL_NAMES,
    AgentRun,
    Task,
    example_task,
    replay,
    run_agent,
    task_suite,
)


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
) -> dict[str, dict]:
    """Solve rate for every (family, budget) cell, keyed ``family@budget``."""
    cells: dict[str, dict] = {}
    for family in FAMILIES:
        family_tasks = [t for t in tasks if t.family == family]
        for budget in budgets:
            runs = [
                (task, run_agent(task, budget=budget, memory=memory,
                                 policy=policy, seed=seed + index))
                for index, task in enumerate(family_tasks)
            ]
            cells[f"{family}@{budget}"] = solve_rate(runs)
    return cells


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
    random_action = random_grid(tasks, max_budget, args.random_rounds, args.seed)

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
        "random_action": random_action,
        "one_step": one_step,
        "write_exactness": write_exactness,
    }

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
        },
        "solve_rate": solve,
        "controls": controls,
        "example_trace": trace_payload(example_run, example),
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
