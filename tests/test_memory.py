"""Long-context memory: the store, the recall family, and the controls.

The same two kinds of test as the rest of the repository, applied to the one
capability where an overclaim is most tempting: "it remembers" is easy to make
true and easy to make meaningless.

**Analytic ground truth.** ``3 * 4 - 5 = 7``, the table pairs 7 with key 9, and
the second write under a key replaces the first. Those answers exist before the
store does, and the loop's answers are compared with them exactly.

**Controls that can fail.** Every claim here has a condition that must break it,
and the test asserts the breakage rather than the claim alone: a working state
with no store must fail at *every* distance while the store passes; a state
**64x wider** with the same write rule must fail in exactly the same place; a
first-write-wins store must return the superseded value; an untagged store must
return **another fact's value** where the tagged one reports a miss; and a
retrieval-disabled store must pay the bytes and answer nothing. A test that only
checked "the store solves the recall task" would pass against a hard-coded
answer.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from beyond_attention.agent import (
    AGENT_DIM,
    ARG_BOUND,
    OP_ADD,
    OP_KEY,
    OP_MUL,
    OP_FETCH,
    OP_REMEMBER,
    OP_RET,
    OP_SUB,
    ToolCall,
    capacity_suite,
    example_recall_task,
    example_stale_task,
    execute,
    read_registers,
    recall_suite,
    replay,
    required_state_width,
    run_agent,
    stale_suite,
    state_bytes,
)
from beyond_attention.memory import (
    DEFAULT_CAPACITY,
    TAGGED_SLOT_BYTES,
    UNTAGGED_SLOT_BYTES,
    MemoryStore,
    store_bytes,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# The store itself
# --------------------------------------------------------------------------

def test_the_store_returns_exactly_what_was_written() -> None:
    store = MemoryStore(DEFAULT_CAPACITY)
    store.write(key=3, value=42)
    store.write(key=7, value=-5)

    assert store.retrieve(3) == 42
    assert store.retrieve(7) == -5
    # A key that was never written is a miss, not a zero: 0 is a legitimate
    # value, so a store that returned it could not be told from one that forgot.
    assert store.retrieve(4) is None
    assert (store.hits, store.misses) == (2, 1)


def test_a_tagged_store_reports_a_miss_when_a_key_is_evicted() -> None:
    """Eight slots, keys 1 and 9: the later write takes the earlier one's slot."""
    store = MemoryStore(DEFAULT_CAPACITY)
    assert store.slot(1) == store.slot(1 + DEFAULT_CAPACITY)
    store.write(1, 100)
    store.write(1 + DEFAULT_CAPACITY, 200)

    assert store.retrieve(1) is None, "the evicted key is a miss, never a guess"
    assert store.retrieve(1 + DEFAULT_CAPACITY) == 200
    assert store.evictions == 1


def test_an_untagged_store_returns_another_keys_value() -> None:
    """The control the tag exists for, asserted as the failure it is.

    Both stores solve the same number of tasks; one of them hands over a fact
    that belongs to a different key, and a caller cannot tell.
    """
    tagged = MemoryStore(DEFAULT_CAPACITY)
    untagged = MemoryStore(DEFAULT_CAPACITY, verify_key=False)
    for store in (tagged, untagged):
        store.write(1, 100)
        store.write(1 + DEFAULT_CAPACITY, 200)

    assert tagged.retrieve(1) is None
    assert untagged.retrieve(1) == 200, "the slot answers for whoever asks"
    # ...and it is cheaper, which is why the failure is worth measuring.
    assert untagged.bytes() < tagged.bytes()


def test_a_later_write_replaces_and_a_first_write_wins_store_does_not() -> None:
    replacing = MemoryStore(DEFAULT_CAPACITY)
    replacing.write(key=5, value=12)
    replacing.write(key=5, value=17)
    assert replacing.retrieve(5) == 17

    stale = MemoryStore(DEFAULT_CAPACITY, overwrite=False)
    stale.write(key=5, value=12)
    stale.write(key=5, value=17)
    assert stale.retrieve(5) == 12
    assert stale.overwritten == 1


def test_retrieval_disabled_pays_for_the_writes_and_reads_nothing() -> None:
    store = MemoryStore(DEFAULT_CAPACITY, retrieval=False)
    store.write(1, 5)
    assert store.retrieve(1) is None
    assert store.writes == 1
    assert store.bytes() == store_bytes(DEFAULT_CAPACITY)


def test_bytes_are_the_measured_allocation_and_the_arithmetic_agrees() -> None:
    """The published figure is `nbytes` on the live arrays, not a convention."""
    tagged = MemoryStore(DEFAULT_CAPACITY)
    untagged = MemoryStore(DEFAULT_CAPACITY, verify_key=False)

    assert tagged.bytes() == TAGGED_SLOT_BYTES * DEFAULT_CAPACITY
    assert tagged.bytes() == store_bytes(DEFAULT_CAPACITY)
    assert untagged.bytes() == UNTAGGED_SLOT_BYTES * DEFAULT_CAPACITY
    assert tagged.bytes() == tagged.keys.nbytes + tagged.values.nbytes \
        + tagged.present.nbytes

    # Constant in what is written: the whole claim about a bounded store.
    for key in range(1, DEFAULT_CAPACITY + 1):
        tagged.write(key, key * 1000)
    assert tagged.bytes() == store_bytes(DEFAULT_CAPACITY)

    # ...and it is not a rounding of the working state's figure.
    assert state_bytes(AGENT_DIM) == AGENT_DIM * 8 == 64
    assert tagged.bytes() > state_bytes(AGENT_DIM)


def test_an_unbounded_store_grows_and_keeps_the_last_write() -> None:
    """The baseline: no eviction, bytes that grow, and the newest value wins."""
    store = MemoryStore(None)
    empty = store.bytes()
    for key in range(1, 33):
        store.write(key, key * 10)
        store.write(key, key * 10 + 1)
    assert store.bytes() > empty
    assert store.retrieve(7) == 71          # the second write, not the first
    assert store.retrieve(99) is None
    # Allocated bytes are a staircase (a growing array doubles) and are never
    # below the arithmetic floor of one slot per entry.
    assert store.bytes() >= store.writes * TAGGED_SLOT_BYTES


def test_random_retrieval_is_seeded_and_draws_live_values() -> None:
    def draw(seed: int) -> list[int | None]:
        store = MemoryStore(DEFAULT_CAPACITY, random_retrieval=True, seed=seed)
        for key, value in ((1, 10), (2, 20), (3, 30)):
            store.write(key, value)
        return [store.retrieve(1) for _ in range(6)]

    assert draw(0) == draw(0)               # deterministic given the seed
    assert all(value in (10, 20, 30) for value in draw(0))
    # Two different seeds are two different streams; the first version of the
    # sweep seeded once and every task drew the same slot, which made the
    # "floor" a constant.
    assert draw(0) != draw(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0},
        {"capacity": -1},
        {"capacity": True},
        {"capacity": None, "verify_key": False},
        {"verify_key": False, "overwrite": False},
        {"retrieval": False, "random_retrieval": True},
    ],
)
def test_bad_store_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MemoryStore(**kwargs)


def test_the_store_module_reads_nothing_but_numpy() -> None:
    """Same check as the agent's: no clock, no filesystem, no network, no RNG
    outside the one seeded control."""
    source = ROOT / "src" / "beyond_attention" / "memory.py"
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy", "__future__"}, imported


# --------------------------------------------------------------------------
# Analytic ground truth for the recall family
# --------------------------------------------------------------------------

def test_the_recall_example_has_the_hand_computed_answer() -> None:
    """3 * 4 - 5 = 7, the table pairs 7 with key 9, and the query returns 9."""
    task = example_recall_task()

    assert [i.op for i in task.plan] == [
        OP_ADD, OP_MUL, OP_SUB, OP_KEY, OP_REMEMBER, OP_ADD, OP_SUB,
        OP_FETCH, OP_RET,
    ]
    assert dict(task.table)[9] == 7
    assert task.answer == 9
    assert dict(task.facts)[4] == 9          # the fact is the *key*, 9
    assert task.distance == 2 and task.query_key == 4


def test_the_stale_example_returns_the_new_value_and_the_old_one_is_recorded() -> (
        None
):
    """Key 4 holds 9, then 7. The answer is 7; 9 is what a stale store gives."""
    task = example_stale_task()

    assert task.answer == 7
    assert task.superseded == (9,)
    assert dict(task.facts) == {4: 7}

    fresh = run_agent(task, budget=len(task.plan), store=MemoryStore(8))
    assert fresh.solved and fresh.answer == 7

    stale = run_agent(task, budget=len(task.plan),
                      store=MemoryStore(8, overwrite=False))
    assert stale.solved is False
    assert stale.answer in task.superseded, \
        "the stale control must return the value that used to be true"


# --------------------------------------------------------------------------
# The controls, over distance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("distance", [1, 64, 1024])
def test_the_working_state_alone_fails_at_every_distance(distance: int) -> None:
    """Long and short: the state holds the last observation, and that is all.

    The failure is a *valid wrong number*, not an exception: a memory call with
    no store is a typed error, and a failed observation writes the carry to zero
    (the same rule that makes a missed lookup a miss rather than a zero value).
    A control that crashed would only show the loop was broken.
    """
    task = recall_suite((distance,), 4, seed=0)[0]
    budget = len(task.plan)

    answered = run_agent(task, budget=budget, store=None)
    assert answered.solved is False
    assert answered.stop_reason == "finish"
    assert answered.answer is not None, "a wrong answer, not an error"
    assert answered.answer != task.answer
    assert [s.result.error for s in answered.steps
            if s.action.tool in ("remember", "fetch")] == \
        ["no_store", "no_store"]

    stored = run_agent(task, budget=budget, store=MemoryStore(8))
    assert stored.solved and stored.answer == task.answer


def test_a_wider_state_with_the_same_write_rule_does_not_help() -> None:
    """64x the bytes, the same register, the same failure: the key is the point.

    The wide state is not a straw man with the store removed; it is the same
    embedding and the same policy over a state 512 registers wide. What it does
    not have is any way to *address* one register from a key.
    """
    task = recall_suite((1,), 1, seed=0)[0]
    wide_dim = AGENT_DIM * 64

    narrow = run_agent(task, budget=len(task.plan), store=None)
    wide = run_agent(task, budget=len(task.plan), store=None,
                     state_width=wide_dim - AGENT_DIM)
    assert wide.solved is False
    assert wide.actions == narrow.actions
    assert wide.state_dim == wide_dim
    assert state_bytes(wide_dim) > MemoryStore(8).bytes() > state_bytes(AGENT_DIM)

    # ...and the extra registers were never written, which is *why* it fails:
    # the same events, embedded into a 512-register state, leave at most the
    # eight named registers nonzero. (The store's own keys are not addresses in
    # this state -- that is the whole point -- so ``required_state_width`` stays
    # 0 for a recall task and the state is eight registers unless a caller
    # widens it.)
    from beyond_attention.agent import (
        embed, instruction_event, new_state, stream_step,
    )

    state = new_state(wide_dim - AGENT_DIM)
    assert state.shape == (wide_dim,)
    for instr in task.plan:
        state = stream_step(state, *embed(instruction_event(instr),
                                          state_width=wide_dim - AGENT_DIM))
    assert np.count_nonzero(state) <= AGENT_DIM
    assert required_state_width(task) == 0


@pytest.mark.parametrize("distance", [1, 16, 256])
def test_the_store_solves_distances_the_state_cannot(distance: int) -> None:
    """Both halves asserted: the store passes what the state fails."""
    tasks = recall_suite((distance,), 4, seed=3)
    assert tasks
    for task in tasks:
        budget = len(task.plan)
        assert run_agent(task, budget=budget, store=MemoryStore(8)).solved
        assert run_agent(task, budget=budget, store=None).solved is False
        assert run_agent(task, budget=budget,
                         store=MemoryStore(8, retrieval=False)).solved is False


def test_a_single_slot_does_the_same_job_as_eight_at_one_fact() -> None:
    """The store's *size* is not what carries the fact -- the key is."""
    task = recall_suite((64,), 2, seed=5)[0]
    one = run_agent(task, budget=len(task.plan), store=MemoryStore(1))
    eight = run_agent(task, budget=len(task.plan), store=MemoryStore(8))
    assert one.solved and eight.solved
    assert one.actions == eight.actions
    assert MemoryStore(1).bytes() < MemoryStore(8).bytes()


# --------------------------------------------------------------------------
# Capacity, and retrieval precision
# --------------------------------------------------------------------------

def test_the_tag_is_what_stops_a_wrong_fact_being_returned() -> None:
    """At one fact past capacity the two stores solve the same number and fail
    differently: a miss against another fact's value."""
    tasks = capacity_suite((DEFAULT_CAPACITY + 1,), distance=4,
                           tasks_per_size=4, seed=0)
    assert tasks
    tagged_wrong_fact = 0
    untagged_wrong_fact = 0
    values = {value for task in tasks for _, value in task.facts}
    for task in tasks:
        budget = len(task.plan)
        tagged = run_agent(task, budget=budget, store=MemoryStore(8))
        untagged = run_agent(task, budget=budget,
                             store=MemoryStore(8, verify_key=False))
        assert tagged.solved == untagged.solved
        other = {value for key, value in task.facts if key != task.query_key}
        if tagged.answer is not None and tagged.answer in other:
            tagged_wrong_fact += 1
        if untagged.answer is not None and untagged.answer in other:
            untagged_wrong_fact += 1
    assert tagged_wrong_fact == 0, "a tagged store never returns another fact"
    assert untagged_wrong_fact > 0, "the untagged store must show the failure"
    assert values, "the tasks must carry facts for the comparison to mean anything"


def test_eight_slots_hold_eight_facts_and_the_ninth_evicts_the_first() -> None:
    holding = capacity_suite((DEFAULT_CAPACITY,), distance=4,
                             tasks_per_size=4, seed=1)
    spilling = capacity_suite((DEFAULT_CAPACITY + 1,), distance=4,
                              tasks_per_size=4, seed=1)
    for task in holding:
        assert run_agent(task, budget=len(task.plan),
                         store=MemoryStore(8)).solved, task.task_id

    evicted = [task for task in spilling if task.query_key == 1]
    assert evicted
    for task in evicted:
        store = MemoryStore(8)
        run = run_agent(task, budget=len(task.plan), store=store)
        assert run.solved is False
        assert store.evictions == 1
        # The newest fact is still there, so the failure is the eviction rather
        # than the store being unable to answer anything.
        newest = [t for t in spilling if t.query_key == task.n_facts]
        if newest:
            assert run_agent(newest[0], budget=len(newest[0].plan),
                             store=MemoryStore(8)).solved


def test_the_unbounded_store_never_evicts_where_the_bounded_one_does() -> None:
    tasks = capacity_suite((32,), distance=4, tasks_per_size=2, seed=2)
    for task in tasks:
        unbounded = run_agent(task, budget=len(task.plan), store=MemoryStore(None))
        assert unbounded.solved, task.task_id
        bounded = run_agent(task, budget=len(task.plan), store=MemoryStore(8))
        assert bounded.solved == (task.query_key > task.n_facts - 8), task.task_id


# --------------------------------------------------------------------------
# Determinism, replay, and staying inside the schema
# --------------------------------------------------------------------------

def test_the_recall_trace_replays_and_a_seed_reproduces_it() -> None:
    task = recall_suite((16,), 2, seed=7)[0]

    def once():
        return run_agent(task, budget=len(task.plan), store=MemoryStore(8))

    first, again = once(), once()
    assert first.actions == again.actions
    assert [s.result for s in first.steps] == [s.result for s in again.steps]
    assert first.answer == again.answer == task.answer

    # The trace alone reproduces the run: `replay` rebuilds a *fresh* store from
    # the recorded spec and re-executes the recorded actions, including the
    # writes, in order.
    replayed = replay(first, task.table)
    assert replayed == tuple(step.result for step in first.steps)
    assert first.store is not None
    assert first.store.writes == 1


def test_generated_recall_plans_stay_inside_the_tool_bounds() -> None:
    """Every step of every generated plan must be executable.

    The first version of the generator walked a carry past the tools' argument
    bound, so the loop hit ``out_of_range`` in the middle of a plan and the
    measurement was of the schema rather than of memory. This is the invariant
    that catches it.
    """
    suites = {
        "recall": recall_suite((1, 8, 64), 3, seed=0),
        "capacity": capacity_suite((8, 12), distance=4, tasks_per_size=3, seed=0),
        "stale": stale_suite((1, 8), 3, seed=0),
    }
    misses = 0
    for name, tasks in suites.items():
        assert tasks, name
        for task in tasks:
            run = run_agent(task, budget=len(task.plan), store=MemoryStore(8))
            # A query for a key the store has evicted is a legitimate miss and a
            # correct failure, so it is counted rather than forbidden; every
            # other step has to succeed.
            evicted = (name == "capacity"
                       and task.n_facts > DEFAULT_CAPACITY
                       and task.query_key <= task.n_facts - DEFAULT_CAPACITY)
            if evicted:
                misses += 1
                assert run.solved is False, (name, task.task_id)
                continue
            assert run.solved, (name, task.task_id, run.answer, task.answer)
            for step in run.steps:
                assert step.result.ok, (name, task.task_id, step.index,
                                        step.action.tool, step.result.error)
                assert abs(step.result.value) <= ARG_BOUND
            assert abs(task.answer) <= ARG_BOUND, (name, task.task_id)
    assert misses, "the suite must exercise the eviction path, or it is not one"


def test_isolation_between_runs() -> None:
    """A store belongs to a run: the next task must not find the last one's facts."""
    task = recall_suite((2,), 2, seed=11)[0]
    shared = MemoryStore(8)
    first = run_agent(task, budget=len(task.plan), store=shared)
    assert first.solved
    # Re-running with the *same* store is a different experiment (the fact is
    # already there and the write is an overwrite), which is why the results
    # file builds a fresh store per run.
    shared.write(999, 12345)
    assert shared.retrieve(999) == 12345
    assert run_agent(task, budget=len(task.plan),
                     store=MemoryStore(8)).actions == first.actions


# --------------------------------------------------------------------------
# The tool registry: malformed calls, and a missing store
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "call, expected",
    [
        (ToolCall("remember", {"key": 1}), "bad_params"),
        (ToolCall("remember", {"key": 1, "value": 2, "extra": 3}), "bad_params"),
        (ToolCall("remember", {"key": 1.0, "value": 2}), "bad_type"),
        (ToolCall("remember", {"key": 1, "value": True}), "bad_type"),
        (ToolCall("remember", {"key": 1, "value": 10_001}), "out_of_range"),
        (ToolCall("fetch", {}), "bad_params"),
        (ToolCall("fetch", {"key": -10_001}), "out_of_range"),
        (ToolCall("fetch", {"key": True}), "bad_type"),
        (ToolCall("fetch", {"key": 1, "value": 2}), "bad_params"),
    ],
)
def test_malformed_memory_calls_are_rejected(call: ToolCall,
                                             expected: str) -> None:
    store = MemoryStore(8)
    assert execute(call, (), store).error == expected
    assert store.writes == 0, "a rejected call must not reach the store"
    assert store.retrieve(1) is None


def test_a_memory_call_with_no_store_is_a_typed_error_not_an_exception() -> None:
    result = execute(ToolCall("remember", {"key": 1, "value": 2}), (), None)
    assert (result.ok, result.value, result.error) == (False, None, "no_store")
    assert execute(ToolCall("fetch", {"key": 1}), (), None).error == "no_store"

    # And the loop records it and keeps going, exactly like any other bad call.
    calls = [ToolCall("remember", {"key": 1, "value": 2}),
             ToolCall("finish", {"answer": 9})]

    def policy(registers) -> ToolCall:
        del registers
        return calls.pop(0)

    run = run_agent(example_recall_task(), budget=4, store=None, policy=policy)
    assert [step.result.error for step in run.steps] == ["no_store", None]
    assert run.solved and run.answer == 9


def test_the_carry_survives_a_remember_and_moves_on_a_fetch() -> None:
    """The two instructions are read out of the register file, not the plan."""
    task = example_recall_task()
    run = run_agent(task, budget=len(task.plan), store=MemoryStore(8))
    remember = run.steps[4]
    fetched = run.steps[7]

    assert remember.action.tool == "remember"
    assert remember.action.args == {"key": 4, "value": 9}
    assert remember.result.value == 9            # the carry is unchanged
    assert fetched.action.args == {"key": 4}
    assert fetched.result.value == 9
    assert read_registers(np.zeros(AGENT_DIM)).miss is False
