"""Long-context memory: what an explicit store buys, over what distance, at what cost.

The repository has already measured the two extremes of memory. ``stream-memory.json``
and ``long-context.json`` put an attention KV cache at **1,024 MiB after a million
tokens** next to a state-space model's **19,456-byte state at every length**, and
the agent block measures an eight-register working state that holds the last
observation and nothing further back. This experiment builds the third thing and
measures the trade rather than the extremes: an explicit, keyed store
(``src/beyond_attention/memory.py``) that a loop writes facts into and reads them
out of, at a **constant** byte cost, long after the working state has lost them.

Four questions, each with its own control:

* **Solve rate against distance.** The same task shape with a controllable number
  of distractor steps between the fact and the query, run by the store, by a
  working state with no store, by a state **64x wider** with the same write rule,
  by a store with retrieval disabled, and by a store that returns a random live
  value. The working state must collapse and the store must not; the wide state
  is what separates "more bytes" from "a key".
* **Bytes against length.** Working state, store, untagged store, an unbounded
  store, a full transcript, and -- quoted from the committed streaming results --
  the model-scale KV cache and SSM state. The crossover is reported as the
  smallest length at which each bounded thing stops working and what the unbounded
  alternative costs there.
* **Capacity against precision.** More facts than slots, querying the oldest and
  the newest write. A tagged store forgets (a miss); an untagged one, which is
  8 bytes a slot cheaper, returns **another fact's value**. Correct, wrong-fact
  and miss are counted separately, because "it remembered" is meaningless if it
  remembers the wrong thing.
* **The stale-fact control.** A fact written twice. The answer is the second
  value; a first-write-wins store returns the first, which is the failure mode
  that a recall returning no answer at all does not have.

Nothing here is trained and nothing uses the network, the filesystem or a clock.
Every number is written to ``long-memory.json`` and rendered into the README by
``experiments/render_readme.py``.

    python -u experiments/long_memory.py --out long-memory.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Callable, Sequence

from beyond_attention.agent import (
    AGENT_DIM,
    ARG_BOUND,
    Task,
    example_recall_task,
    example_stale_task,
    capacity_suite,
    recall_suite,
    replay,
    run_agent,
    stale_suite,
    state_bytes,
)
from beyond_attention.memory import (
    DEFAULT_CAPACITY,
    TAGGED_SLOT_BYTES,
    MemoryStore,
)

# The wide-state control: 64x the working state's registers, and the same write
# rule, so the only thing that changes is how many bytes are allocated.
WIDE_DIM = AGENT_DIM * 64

# Distances between the fact and the query. The step count grows with them (the
# plan is `distance + 7` instructions), so the sample size shrinks: a solve rate
# quoted as solved/total rather than as a rate is the honest way to publish that.
DISTANCES = (1, 4, 16, 64, 256, 1024)
TASKS_PER_DISTANCE = {1: 16, 4: 16, 16: 16, 64: 8, 256: 4, 1024: 2}

# Facts against eight slots. The oldest fact is the one a direct-mapped store
# evicts first; the newest is the one it cannot have lost.
# 9 is in the list on purpose: eight slots hold eight facts exactly, and the
# ninth key shares a slot with the first, so the measured cliff lands on the
# designed one instead of on the nearest sweep point.
FACTS_PER_SIZE = (1, 2, 4, 8, 9, 16, 32)
TASKS_PER_SIZE = 8
CAPACITY_DISTANCE = 4

STALE_DISTANCES = (1, 4, 16, 64)
TASKS_PER_STALE = 4

# The random-retrieval floor draws from a *different* stream for every task, so
# the floor is the floor of a fresh draw rather than of one repeated draw.
RANDOM_RETRIEVAL_SEED = 1_000

# Lengths for the byte ledger. 1,048,576 is the length the model-scale rows are
# quoted at, and it is in the ledger because that is the length where the KV
# cache reaches a gibibyte.
# 9 and 17 are in the list on purpose: they are where a doubling array's
# allocated bytes stop matching its arithmetic floor, which is the difference
# between what a process holds and what the writes add up to.
BYTE_LENGTHS = (1, 8, 9, 17, 64, 100, 256, 1024, 16384, 1048576)


def condition_factory(
    name: str,
) -> Callable[[int], tuple[MemoryStore | None, dict]]:
    """A (store, run_agent kwargs) factory for a condition, indexed by run.

    A fresh store per run, on purpose: a store shared between runs would carry
    facts from one task into the next, which is a different experiment (and would
    make the capacity sweep pass for the wrong reason).

    The run index reaches the store's seed for ``random_retrieval``. Seeding it
    once for the whole sweep looks harmless and is not: every task then draws the
    *same* slot, which in the first version made the random floor pick the newest
    fact on every query and score 0.500 at four facts. A floor that is really a
    constant is worse than no floor.
    """
    if name == "store":
        return lambda index: (MemoryStore(DEFAULT_CAPACITY), {})
    if name == "single_slot_store":
        # One slot, 17 bytes. It exists to separate the store's *size* from its
        # *addressing*: the single-fact family cannot tell eight slots from one,
        # and the capacity sweep can.
        return lambda index: (MemoryStore(1), {})
    if name == "unbounded_store":
        return lambda index: (MemoryStore(None), {})
    if name == "working_state":
        return lambda index: (None, {})
    if name == "wide_state":
        # ``state_width`` counts the *extra* registers, so the total state is 64x
        # the working one while the eight named registers are unchanged: the same
        # write rule, more bytes, and nothing addressed by a key.
        return lambda index: (None, {"state_width": WIDE_DIM - AGENT_DIM})
    if name == "retrieval_disabled":
        return lambda index: (MemoryStore(DEFAULT_CAPACITY, retrieval=False), {})
    if name == "untagged_store":
        return lambda index: (MemoryStore(DEFAULT_CAPACITY, verify_key=False), {})
    if name == "stale_store":
        return lambda index: (MemoryStore(DEFAULT_CAPACITY, overwrite=False), {})
    if name == "random_retrieval":
        return lambda index: (
            MemoryStore(DEFAULT_CAPACITY, random_retrieval=True,
                        seed=RANDOM_RETRIEVAL_SEED + index), {})
    raise ValueError(f"unknown condition {name!r}")


CONDITIONS = (
    "store",
    "single_slot_store",
    "unbounded_store",
    "working_state",
    "wide_state",
    "retrieval_disabled",
    "random_retrieval",
)

# The conditions where the untagged store is defined (a bounded store), and the
# stale store is meaningful (a task that writes the same key twice).
CAPACITY_CONDITIONS = CONDITIONS + ("untagged_store",)
STALE_CONDITIONS = CONDITIONS + ("stale_store",)


def classify(task: Task, run) -> str:
    """What a run's answer was, in the terms the controls are about.

    ``wrong_fact`` is a value stored under a *different* memory key -- the store
    handed over another fact's answer, confidently. ``stale`` is the value that
    used to be under the queried key. Both are worse than ``no_answer``, and the
    precision column is computed over answered runs so that a store cannot look
    precise by refusing to answer.
    """
    if run.solved:
        return "correct"
    if run.answer is None:
        return "no_answer"
    if run.answer in task.superseded:
        return "stale"
    others = [value for key, value in task.facts if key != task.query_key]
    if run.answer in others:
        return "wrong_fact"
    return "wrong_other"


def measure(tasks: Sequence[Task], condition: str) -> dict:
    """Run one condition over a task list and account for every outcome."""
    factory = condition_factory(condition)
    counts = {"correct": 0, "wrong_fact": 0, "stale": 0, "wrong_other": 0,
              "no_answer": 0}
    solved = 0
    store_bytes: list[int] = []
    state_bytes_seen: list[int] = []
    steps = 0
    for index, task in enumerate(tasks):
        store, kwargs = factory(index)
        run = run_agent(task, budget=len(task.plan), store=store, **kwargs)
        counts[classify(task, run)] += 1
        solved += int(run.solved)
        steps += len(run.steps)
        store_bytes.append(0 if store is None else store.bytes())
        state_bytes_seen.append(state_bytes(run.state_dim))
    total = len(tasks)
    answered = total - counts["no_answer"]
    return {
        "solved": solved,
        "total": total,
        "rate": solved / total if total else 0.0,
        **counts,
        "answered": answered,
        "precision": (counts["correct"] / answered) if answered else 0.0,
        "wrong_confident": counts["wrong_fact"] + counts["stale"]
                            + counts["wrong_other"],
        "store_bytes": max(store_bytes) if store_bytes else 0,
        "state_bytes": max(state_bytes_seen) if state_bytes_seen else 0,
        "steps": steps,
    }


def distance_sweep(tasks: Sequence[Task],
                   conditions: Sequence[str] = CONDITIONS) -> dict:
    """Solve rate for every (distance, condition) cell, keyed ``distance``."""
    cells: dict[str, dict] = {}
    for distance in sorted({task.distance for task in tasks}):
        group = [task for task in tasks if task.distance == distance]
        cells[str(distance)] = {
            "tasks": len(group),
            "plan_steps": len(group[0].plan),
            "conditions": {name: measure(group, name) for name in conditions},
        }
    return cells


def capacity_sweep(tasks: Sequence[Task],
                   conditions: Sequence[str] = CAPACITY_CONDITIONS) -> dict:
    cells: dict[str, dict] = {}
    for size in sorted({task.n_facts for task in tasks}):
        group = [task for task in tasks if task.n_facts == size]
        cells[str(size)] = {
            "tasks": len(group),
            "plan_steps": len(group[0].plan),
            "conditions": {name: measure(group, name) for name in conditions},
        }
    return cells


def stale_sweep(tasks: Sequence[Task],
                conditions: Sequence[str] = STALE_CONDITIONS) -> dict:
    cells: dict[str, dict] = {}
    for distance in sorted({task.distance for task in tasks}):
        group = [task for task in tasks if task.distance == distance]
        cells[str(distance)] = {
            "tasks": len(group),
            "plan_steps": len(group[0].plan),
            "conditions": {name: measure(group, name) for name in conditions},
        }
    return cells


def byte_ledger(lengths: Sequence[int], model_scale: dict) -> dict:
    """Bytes carried at each episode length, by substrate.

    The store's own figure is *measured* from a live ``MemoryStore``; the
    unbounded store's is measured after that many writes, which is why it is a
    staircase -- a growing ``numpy`` array doubles. The transcript is arithmetic
    (one int64 per event, the least a replay needs) and is labelled as such,
    because the run object retains every step whether or not anyone asks it to.
    """
    store = MemoryStore(DEFAULT_CAPACITY)
    untagged = MemoryStore(DEFAULT_CAPACITY, verify_key=False)
    kv_per_token = model_scale.get("kv_cache_bytes_per_token")
    rows = []
    for length in lengths:
        unbounded = MemoryStore(None)
        for index in range(length):
            unbounded.write(index, index)
        row = {
            "length": length,
            "working_state_bytes": state_bytes(AGENT_DIM),
            "wide_state_bytes": state_bytes(WIDE_DIM),
            "store_bytes": store.bytes(),
            "untagged_store_bytes": untagged.bytes(),
            "unbounded_store_bytes": unbounded.bytes(),
            "unbounded_store_arithmetic": length * TAGGED_SLOT_BYTES,
            "transcript_bytes": length * 8,
        }
        if kv_per_token is not None:
            row["model_kv_cache_bytes"] = int(kv_per_token * length)
        rows.append(row)
    return {
        "capacity": DEFAULT_CAPACITY,
        "slot_bytes": TAGGED_SLOT_BYTES,
        "store_bytes": store.bytes(),
        "rows": rows,
    }


def crossover(distance_cells: dict, capacity_cells: dict, ledger: dict,
              model_scale: dict) -> dict:
    """Every threshold in the measurement, derived from the cells above.

    Nothing here is asserted: each number is the first (or last) cell at which a
    condition's measured solve rate changes, so a rerun that changes the answer
    changes this block rather than contradicting it.
    """
    distances = sorted(int(key) for key in distance_cells)

    def first_failure(cells: dict, condition: str, scales: Sequence[int]) -> int | None:
        for scale in scales:
            cell = cells.get(str(scale))
            if cell is None:
                continue
            if cell["conditions"][condition]["rate"] < 1.0:
                return scale
        return None

    facts = sorted(int(key) for key in capacity_cells)
    exact_to = None
    first_failure_facts = None
    for size in facts:
        rate = capacity_cells[str(size)]["conditions"]["store"]["rate"]
        if rate >= 1.0:
            exact_to = size
        elif first_failure_facts is None:
            first_failure_facts = size

    store_bytes = ledger["store_bytes"]
    slot_bytes = ledger["slot_bytes"]
    return {
        "working_state_fails_at_distance": first_failure(
            distance_cells, "working_state", distances),
        "wide_state_fails_at_distance": first_failure(
            distance_cells, "wide_state", distances),
        "store_fails_at_distance": first_failure(
            distance_cells, "store", distances),
        "unbounded_fails_at_distance": first_failure(
            distance_cells, "unbounded_store", distances),
        "longest_distance": distances[-1] if distances else None,
        "store_exact_to_facts": exact_to,
        "store_first_failure_at_facts": first_failure_facts,
        "working_state_bytes": state_bytes(AGENT_DIM),
        "wide_state_bytes": state_bytes(WIDE_DIM),
        "store_bytes": store_bytes,
        "untagged_store_bytes": ledger["rows"][0]["untagged_store_bytes"]
                                if ledger["rows"] else None,
        "unbounded_store_exceeds_store_at_writes":
            store_bytes // slot_bytes + 1,
        "transcript_exceeds_store_at_events": store_bytes // 8 + 1,
        "model_kv_exceeds_store_at_tokens": (
            store_bytes // model_scale["kv_cache_bytes_per_token"] + 1
            if model_scale.get("kv_cache_bytes_per_token") else None),
        "model_ssm_state_bytes": model_scale.get("ssm_state_bytes"),
    }


def quote_model_scale(root: pathlib.Path) -> dict:
    """The model-scale byte figures, read from the committed streaming results.

    Read rather than recomputed: those numbers belong to ``streaming.py`` and its
    experiments, and re-deriving them here would be a second implementation of
    somebody else's measurement. If the files are absent the rows are omitted
    rather than guessed.
    """
    quoted: dict = {}
    stream = root / "stream-memory.json"
    long_context = root / "long-context.json"
    if stream.exists():
        payload = json.loads(stream.read_text())
        config = payload.get("ssm_stream", {})
        if "state_bytes_arithmetic" in config:
            quoted["ssm_state_bytes"] = config["state_bytes_arithmetic"]
        quoted["source_ssm"] = "stream-memory.json"
    if long_context.exists():
        payload = json.loads(long_context.read_text())
        contrast = payload.get("contrast", {})
        length = contrast.get("longest_streamed")
        cache = contrast.get("attention_kv_cache_bytes")
        if length and cache:
            quoted["kv_cache_bytes"] = cache
            quoted["kv_cache_length"] = length
            quoted["kv_cache_bytes_per_token"] = cache // length
        quoted["source_kv"] = "long-context.json"
    return quoted


def trace_payload(task: Task, conditions: Sequence[str]) -> dict:
    """The example task under each condition, step by step.

    The reads are in the payload for the same reason the agent block keeps them:
    a table of actions alone is consistent with a controller that recovered the
    fact from somewhere the claim says it cannot.
    """
    runs = []
    for index, condition in enumerate(conditions):
        store, kwargs = condition_factory(condition)(index)
        run = run_agent(task, budget=len(task.plan), store=store, **kwargs)
        replayed = replay(run, task.table)
        assert [(r.value, r.error) for r in replayed] == \
            [(s.result.value, s.result.error) for s in run.steps], \
            f"the {condition} trace does not replay"
        runs.append({
            "condition": condition,
            "solved": run.solved,
            "answer": run.answer,
            "outcome": classify(task, run),
            "store": None if store is None else store.summary(),
            "steps": [
                {
                    "index": step.index,
                    "instruction": step.instruction.op
                                   + (f" {step.instruction.args[0]}"
                                      if step.instruction.args else ""),
                    "reads": {
                        "op_code": step.registers.op_code,
                        "arg": step.registers.arg,
                        "carry": round(step.registers.carry, 6),
                        "observations": step.registers.observations,
                    },
                    "action": step.action.tool,
                    "args": dict(sorted(step.action.args.items())),
                    "result": step.result.value,
                    "error": step.result.error,
                }
                for step in run.steps
            ],
        })
    return {
        "task_id": task.task_id,
        "family": task.family,
        "text": task.text,
        "plan": [{"op": i.op, "args": list(i.args)} for i in task.plan],
        "table": [list(pair) for pair in task.table],
        "facts": [list(pair) for pair in task.facts],
        "answer": task.answer,
        "superseded": list(task.superseded),
        "query_key": task.query_key,
        "distance": task.distance,
        "conditions": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="long-memory.json")
    parser.add_argument("--root", default=".",
                        help="where the committed JSON results live")
    args = parser.parse_args()

    started = time.perf_counter()
    root = pathlib.Path(args.root)

    recall_tasks = tuple(
        task for distance in DISTANCES
        for task in recall_suite((distance,),
                                 TASKS_PER_DISTANCE[distance], args.seed)
    )
    capacity_tasks = capacity_suite(FACTS_PER_SIZE, distance=CAPACITY_DISTANCE,
                                    tasks_per_size=TASKS_PER_SIZE,
                                    seed=args.seed)
    stale_tasks = stale_suite(STALE_DISTANCES, TASKS_PER_STALE, args.seed)

    model_scale = quote_model_scale(root)
    distance_cells = distance_sweep(recall_tasks)
    capacity_cells = capacity_sweep(capacity_tasks)
    stale_cells = stale_sweep(stale_tasks)
    ledger = byte_ledger(BYTE_LENGTHS, model_scale)

    examples = {
        "recall": trace_payload(example_recall_task(),
                                ("store", "working_state")),
        "stale": trace_payload(example_stale_task(),
                               ("store", "stale_store", "working_state")),
    }

    payload = {
        "config": {
            "seed": args.seed,
            "distances": list(DISTANCES),
            "tasks_per_distance": {str(k): v
                                   for k, v in TASKS_PER_DISTANCE.items()},
            "capacity": DEFAULT_CAPACITY,
            "facts_per_size": list(FACTS_PER_SIZE),
            "tasks_per_size": TASKS_PER_SIZE,
            "capacity_distance": CAPACITY_DISTANCE,
            "stale_distances": list(STALE_DISTANCES),
            "tasks_per_stale": TASKS_PER_STALE,
            "agent_dim": AGENT_DIM,
            "wide_dim": WIDE_DIM,
            "arg_bound": ARG_BOUND,
            "slot_bytes": TAGGED_SLOT_BYTES,
            "conditions": list(CONDITIONS),
            "capacity_conditions": list(CAPACITY_CONDITIONS),
            "stale_conditions": list(STALE_CONDITIONS),
            "budget_rule": "len(plan)",
        },
        "distance": distance_cells,
        "capacity": capacity_cells,
        "stale": stale_cells,
        "bytes": ledger,
        "model_scale": model_scale,
        "crossover": crossover(distance_cells, capacity_cells, ledger,
                               model_scale),
        "examples": examples,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))

    # --- human-readable summary ------------------------------------------
    print(f"{len(recall_tasks)} recall + {len(capacity_tasks)} capacity + "
          f"{len(stale_tasks)} stale tasks, seed {args.seed}")
    print()
    header = f"{'condition':<20}" + "".join(f"{'d' + str(d):>12}" for d in DISTANCES)
    print(header)
    for condition in CONDITIONS:
        row = f"{condition:<20}"
        for distance in DISTANCES:
            cell = distance_cells[str(distance)]["conditions"][condition]
            row += f"{cell['solved']:>6}/{cell['total']:<5}"
        print(row)
    print()
    print(f"{'condition':<20}"
          + "".join(f"{'F' + str(f):>12}" for f in FACTS_PER_SIZE))
    for condition in CAPACITY_CONDITIONS:
        row = f"{condition:<20}"
        for size in FACTS_PER_SIZE:
            cell = capacity_cells[str(size)]["conditions"][condition]
            row += f"{cell['solved']:>6}/{cell['total']:<5}"
        print(row)
    print()
    print(f"{'condition':<20}"
          + "".join(f"{'d' + str(d):>12}" for d in STALE_DISTANCES))
    for condition in STALE_CONDITIONS:
        row = f"{condition:<20}"
        for distance in STALE_DISTANCES:
            cell = stale_cells[str(distance)]["conditions"][condition]
            row += f"{cell['solved']:>6}/{cell['total']:<5}"
        print(row)
    print()
    print("bytes by episode length:")
    for row in ledger["rows"]:
        print(f"  L={row['length']:>8}  working={row['working_state_bytes']:>6}"
              f"  store={row['store_bytes']:>5}"
              f"  untagged={row['untagged_store_bytes']:>4}"
              f"  unbounded={row['unbounded_store_bytes']:>8}"
              f"  transcript={row['transcript_bytes']:>9}")
    print()
    print("crossover:", json.dumps(payload["crossover"], indent=2))
    print()
    for name, trace in examples.items():
        print(f"example {name}: {trace['text']}")
        for run in trace["conditions"]:
            print(f"  {run['condition']:<16} -> {run['outcome']:<12} "
                  f"answer={run['answer']}")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
