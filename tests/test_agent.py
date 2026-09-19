"""The agent loop, checked against tasks whose answers are known by hand.

An agent loop is easy to make look like it works: if the tasks are written by the
same person who wrote the controller, a solve rate of 1.000 says almost nothing.
So the tests here come in the same two kinds as the rest of the repository.

**Analytic ground truth.** ``3 + 4 = 7``, ``7 * 5 = 35``, and the key paired with
35 in a table written out by hand is 7. Those answers exist before the code runs.
The same is true of the state: a register written with a value must hold *that*
value, and the tests assert equality rather than a tolerance, because the
recurrence is built so that a write is exact.

**Controls that can fail.** A test that only checks "the loop answers the
carry-over task" cannot distinguish a working memory from a hard-coded one, so
the same tests run the loop with the state wiped between decisions and require
it to *fail* the carry-over task while still passing the one-step task. A test
that only checks shapes cannot see an all-zero encoder or an inert tool, so the
tool tests feed malformed calls and the module test reads the source's imports.

Where a claim is checkable against something outside this module -- that the
memory is the repository's own recurrence -- it is checked against that thing
rather than against a restatement of it.

The selective family gets the same treatment plus one more requirement: its
controls are asserted to **fail**. The scalar-carry control solves a selective
task exactly when the queried key is the last one stored and never otherwise --
asserted task by task rather than as a rate, because a rate below 1.0 could mean
a hard task instead of a missing register. The fixed-decay control, a state of
the same width with a constant gate, is asserted to tie the scalar when nothing
is distracting and to be broken by a distractor when something is, which is what
says the *gating* rather than the width is doing the work.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from beyond_attention.agent import (
    AGENT_DIM,
    A_DIAG,
    A_HOLD,
    DELTA_WRITE,
    FAMILIES,
    OP_ADD,
    OP_CODES,
    OP_IFPOS,
    OP_KEY,
    OP_MUL,
    OP_NOISE,
    OP_PUT,
    OP_RECALL,
    OP_RET,
    OP_RETLIT,
    OP_SUB,
    R_ARG,
    R_CARRY,
    R_COUNT,
    R_INDEX,
    R_MISS,
    R_OP,
    R_SLOT_BASE,
    SELECTIVE_DISTRACTORS,
    SELECTIVE_KEYS,
    SELECTIVE_STORES,
    STEP_COUNTS,
    TOOL_NAMES,
    ToolCall,
    Registers,
    bridge_to_selective_scan,
    carried_value,
    choose_action,
    decay_vector,
    embed,
    embed_all,
    evaluate_plan,
    example_task,
    execute,
    instruction,
    new_state,
    observation_event,
    read_registers,
    replay,
    required_state_width,
    run_agent,
    scan_states,
    selective_example_task,
    selective_suite,
    slot_value,
    stream_step,
    task_suite,
    validate_call,
)


def make_step_result(value: int, ok: bool = True, error: str | None = None):
    """A ToolResult-shaped observation, without going through a tool."""
    from beyond_attention.agent import ToolResult

    return ToolResult(ok=ok, value=value, error=error)


# --------------------------------------------------------------------------
# Analytic ground truth: tasks whose answers are known by hand
# --------------------------------------------------------------------------

def test_the_example_task_has_the_hand_computed_answer() -> None:
    """3 + 4 = 7, 7 * 5 = 35, and 35 is paired with key 7 in the hand table."""
    task = example_task()

    assert [i.op for i in task.plan] == [
        OP_ADD, OP_ADD, OP_MUL, OP_KEY, OP_RET
    ]
    assert dict(task.table)[7] == 35
    assert len({value for _, value in task.table}) == len(task.table)
    assert task.answer == 7


def test_evaluate_plan_matches_arithmetic_done_in_the_test() -> None:
    """Every instruction, against the arithmetic written out longhand."""
    assert evaluate_plan((instruction(OP_ADD, 3), instruction(OP_RET)), ()) == 3
    assert evaluate_plan((instruction(OP_ADD, 3),
                          instruction(OP_ADD, 4),
                          instruction(OP_RET)), ()) == 7
    assert evaluate_plan((instruction(OP_MUL, 6),
                          instruction(OP_SUB, 5),
                          instruction(OP_RET)), ()) == -5
    # acc = 0 + 4 = 4 > 0, so IFPOS multiplies: 4 * 3 = 12
    assert evaluate_plan((instruction(OP_ADD, 4),
                          instruction(OP_IFPOS, 3),
                          instruction(OP_RET)), ()) == 12
    # acc = 0 - 4 = -4 <= 0, so IFPOS adds: -4 + 3 = -1
    assert evaluate_plan((instruction(OP_SUB, 4),
                          instruction(OP_IFPOS, 3),
                          instruction(OP_RET)), ()) == -1
    # The KEY step swaps the running value for the key paired with it.
    assert evaluate_plan((instruction(OP_ADD, 35),
                          instruction(OP_KEY),
                          instruction(OP_RET)), ((7, 35), (2, 8))) == 7
    assert evaluate_plan((instruction(OP_RETLIT, 42),), ()) == 42


def test_the_loop_reaches_the_known_answer_and_carries_the_known_values() -> None:
    """Not just the answer: the carry at each step, which is the mechanism."""
    run = run_agent(example_task(), budget=6)

    assert run.solved
    assert run.stop_reason == "finish"
    assert run.answer == 7
    assert [step.action.tool for step in run.steps] == [
        "add", "add", "mul", "lookup", "finish"
    ]
    # The carry the policy read *before* each decision: 0 until the first result
    # exists, then the previous step's value. Written out by hand.
    assert [step.registers.carry for step in run.steps] == [0.0, 3.0, 7.0, 35.0, 7.0]
    assert [step.registers.observations for step in run.steps] == [0, 1, 2, 3, 4]
    assert [step.result.value for step in run.steps] == [3, 7, 35, 7, 7]


# --------------------------------------------------------------------------
# The state is the memory, and it is exact
# --------------------------------------------------------------------------

def test_a_write_replaces_a_register_exactly() -> None:
    """Asserted at equality, not to a tolerance, because the design is exact.

    The first version of the module used ``A = -50`` and left 1.7e-21 of the
    previous value behind, which is greater than zero; the branch instruction
    read it as "positive" and took the wrong arm on 8 of 50 tasks. ``A_HOLD``
    makes the write exact, so this test has no epsilon, and that is the point.
    """
    state = np.zeros(AGENT_DIM)
    x, delta = embed(observation_event(make_step_result(9)))
    state = stream_step(state, x, delta)

    assert state[R_CARRY] == 9.0
    assert state[R_COUNT] == 1.0
    # Overwrite with zero: the readout must be exactly zero, not nearly zero.
    x, delta = embed(observation_event(make_step_result(0)))
    state = stream_step(state, x, delta)

    assert state[R_CARRY] == 0.0
    assert not state[R_CARRY] > 0
    assert np.exp(DELTA_WRITE * A_HOLD) == 0.0


def test_a_hold_leaves_every_register_bit_for_bit_unchanged() -> None:
    """An action writes (zeros, zeros); that must be a no-op on the state."""
    state = np.zeros(AGENT_DIM)
    state = stream_step(state, *embed(observation_event(make_step_result(35))))
    state = stream_step(state, *embed(instruction_event_of(OP_MUL, 5)))
    before = state.copy()

    from beyond_attention.agent import action_event

    state = stream_step(state, *embed(action_event(ToolCall("mul", {"a": 7, "b": 5}))))

    np.testing.assert_array_equal(state, before)
    # And the hold is a real hold across a long run, not a slow leak.
    for _ in range(500):
        state = stream_step(state, np.zeros(AGENT_DIM), np.zeros(AGENT_DIM))
    np.testing.assert_array_equal(state, before)


def instruction_event_of(op: str, *args: int):
    from beyond_attention.agent import instruction_event

    return instruction_event(instruction(op, *args))


def test_the_instruction_registers_and_the_counters_are_exact() -> None:
    """Every register, after a known sequence of events."""
    from beyond_attention.agent import action_event

    events = [
        instruction_event_of(OP_ADD, 3),
        action_event(ToolCall("add", {"a": 0, "b": 3})),
        observation_event(make_step_result(3)),
        instruction_event_of(OP_KEY),
        action_event(ToolCall("lookup", {"value": 3})),
        observation_event(make_step_result(7)),
    ]
    x, delta = embed_all(events)
    states = scan_states(x, delta)
    final = read_registers(states[-1])

    assert final.carry == 7.0          # the second observation's value
    assert final.observations == 2     # two observations, actions not counted
    assert final.instructions == 2     # two instructions, observations not counted
    assert final.op_code == 6          # KEY
    assert final.arg == 0
    assert final.kind == 1 and final.miss is False

    # And the same readout as each event lands, so the memory can be watched
    # changing rather than only inspected at the end.
    assert [read_registers(s).carry for s in states] == [0.0, 0.0, 3.0, 3.0, 3.0, 7.0]
    assert [read_registers(s).instructions for s in states] == [1, 1, 1, 2, 2, 2]


def test_a_missed_lookup_is_recorded_as_a_miss_not_as_a_zero() -> None:
    """0 is a legitimate value; the state has to tell "no key" from "key is 0"."""
    from beyond_attention.agent import action_event

    hit = run_agent(example_task(), budget=6).steps[3]
    assert hit.result.error is None and hit.registers.miss is False

    miss_result = execute(ToolCall("lookup", {"value": 9999}),
                          example_task().table)
    assert miss_result.error == "no_key"
    state = np.zeros(AGENT_DIM)
    for event in (action_event(ToolCall("lookup", {"value": 9999})),
                  observation_event(miss_result)):
        state = stream_step(state, *embed(event))
    registers = read_registers(state)
    assert registers.miss is True
    assert registers.observations == 1
    assert registers.carry == 0.0
    # `carried_value` still reports 0 because an observation happened; the miss
    # flag is what says the 0 is not a result.
    assert carried_value(registers) == 0


def test_the_numpy_recurrence_is_the_repositorys_selective_scan() -> None:
    """The claim "the memory is the SSM's recurrence", checked against the SSM.

    ``selective_scan`` is the repository's own implementation. The numpy state
    above is fed to it through ``bridge_to_selective_scan`` and must come back
    identical -- float64 throughout, so the comparison is exact rather than
    indicative.
    """
    torch = pytest.importorskip("torch")
    from beyond_attention.ssm import selective_scan

    events = [
        instruction_event_of(OP_ADD, 3),
        observation_event(make_step_result(3)),
        instruction_event_of(OP_MUL, 5),
        observation_event(make_step_result(15)),
        instruction_event_of(OP_IFPOS, 2),
        observation_event(make_step_result(30)),
    ]
    x, delta = embed_all(events)
    mine = scan_states(x, delta)

    arrays = bridge_to_selective_scan(x, delta)
    with torch.no_grad():
        theirs = selective_scan(
            torch.from_numpy(arrays["x"]),
            torch.from_numpy(arrays["delta"]),
            torch.from_numpy(arrays["A"]),
            torch.from_numpy(arrays["B"]),
            torch.from_numpy(arrays["C"]),
        )
    theirs = theirs.numpy()[0]

    assert theirs.shape == mine.shape
    np.testing.assert_array_equal(theirs, mine)
    # The readout is the state, so it carries the same registers.
    registers = read_registers(theirs[-1])
    assert registers.carry == 30.0
    assert registers.observations == 3
    assert registers.instructions == 3


def test_the_state_is_deterministic_and_the_wipe_changes_it() -> None:
    x, delta = embed_all([instruction_event_of(OP_ADD, 3),
                          observation_event(make_step_result(3))])

    first = scan_states(x, delta)[-1]
    again = scan_states(x, delta)[-1]
    np.testing.assert_array_equal(first, again)

    wiped = scan_states(x[:1], delta[:1])[-1]  # only the instruction, no result
    assert read_registers(wiped).carry == 0.0
    assert read_registers(wiped).observations == 0
    assert read_registers(wiped).op_code == read_registers(first).op_code


# --------------------------------------------------------------------------
# Tools: schemas, typed errors, purity
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "call, expected",
    [
        (ToolCall("power", {"a": 1, "b": 2}), "unknown_tool"),
        (ToolCall("add", {"a": 1}), "bad_params"),
        (ToolCall("add", {"a": 1, "b": 2, "c": 3}), "bad_params"),
        (ToolCall("add", {"a": 1, "c": 2}), "bad_params"),
        (ToolCall("add", {"a": 1.0, "b": 2}), "bad_type"),
        (ToolCall("add", {"a": True, "b": 2}), "bad_type"),
        (ToolCall("add", {"a": "1", "b": 2}), "bad_type"),
        (ToolCall("add", {"a": None, "b": 2}), "bad_type"),
        (ToolCall("add", {"a": 10_001, "b": 0}), "out_of_range"),
        (ToolCall("lookup", {"value": -10_001}), "out_of_range"),
        (ToolCall("remember", {"key": 1}), "bad_params"),
        (ToolCall("fetch", {"key": True}), "bad_type"),
        (ToolCall("fetch", {"key": 1, "value": 2}), "bad_params"),
        (ToolCall("finish", {}), "bad_params"),
    ],
)
def test_schema_validation_rejects_malformed_calls(call: ToolCall,
                                                   expected: str) -> None:
    assert validate_call(call) == expected
    result = execute(call, example_task().table)
    assert result.ok is False
    assert result.value is None
    assert result.error == expected


def test_well_formed_calls_are_accepted_and_pure() -> None:
    for tool in TOOL_NAMES:
        from beyond_attention.agent import TOOL_SCHEMAS

        args = {p.name: 3 for p in TOOL_SCHEMAS[tool].params}
        call = ToolCall(tool, args)
        assert validate_call(call) is None

    call = ToolCall("mul", {"a": 6, "b": 7})
    first = execute(call, ())
    assert (first.ok, first.value, first.error) == (True, 42, None)
    assert execute(call, ()) == first  # pure: same call, same result

    table = example_task().table
    assert execute(ToolCall("lookup", {"value": 35}), table).value == 7
    assert execute(ToolCall("lookup", {"value": 9_999}), table).error == "no_key"
    # A table with two keys sharing a value is a malformed task, not a miss.
    assert execute(ToolCall("lookup", {"value": 1}), ((1, 1), (2, 1))).error == \
        "ambiguous_table"


def test_a_malformed_call_cannot_end_the_loop() -> None:
    """The loop must record the error and keep going, not raise."""
    calls = [ToolCall("add", {"a": 1}),          # missing b
             ToolCall("explode", {}),            # unknown tool
             ToolCall("finish", {"answer": 7})]  # the real answer, at last

    def policy(registers: Registers) -> ToolCall:
        del registers
        return calls.pop(0)

    run = run_agent(example_task(), budget=6, policy=policy)

    assert run.solved
    assert [step.result.error for step in run.steps[:2]] == \
        ["bad_params", "unknown_tool"]
    assert [step.result.ok for step in run.steps] == [False, False, True]


def test_the_module_reads_nothing_but_numpy() -> None:
    """No clock, no filesystem, no network, no RNG in the module itself.

    Read from the source rather than asserted in prose, because "the tools are
    pure" is exactly the kind of claim that quietly stops being true.
    """
    source = pathlib.Path(__file__).resolve().parents[1] / "src" / \
        "beyond_attention" / "agent.py"
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    # ``memory`` is the one sibling import, and it is itself numpy-only --
    # ``tests/test_memory.py`` reads its imports the same way. Anything else,
    # and the store could reach a clock or a socket without this test noticing.
    assert imported <= {"numpy", "dataclasses", "typing", "__future__",
                        "memory"}, imported
    for forbidden in ("time", "random", "os", "socket", "pathlib", "json"):
        assert forbidden not in imported
    # Every tool is a pure function of its arguments: none of them is handed a
    # generator, and the only RNG in the module is the one ``run_agent`` builds
    # for the random-action control.
    for name in ("add", "mul", "sub", "lookup", "remember", "fetch", "finish"):
        assert name in source.read_text()


# --------------------------------------------------------------------------
# The loop: budget, termination, replay
# --------------------------------------------------------------------------

def test_the_budget_is_honoured_exactly() -> None:
    task = next(t for t in task_suite(3, seed=0) if t.family == "lookup")
    assert STEP_COUNTS["lookup"] == 5

    for budget in (1, 2, 3, 4):
        run = run_agent(task, budget=budget)
        assert len(run.steps) == budget
        assert run.stop_reason == "budget"
        assert run.solved is False
        assert run.answer is None

    run = run_agent(task, budget=5)
    assert len(run.steps) == 5
    assert run.stop_reason == "finish"
    assert run.solved is True
    # More budget than the task needs does not add steps: the loop stops at
    # `finish`, and the budget is a ceiling rather than a schedule.
    assert len(run_agent(task, budget=50).steps) == 5


def test_each_family_uses_exactly_its_documented_number_of_steps() -> None:
    for task in task_suite(4, seed=7):
        run = run_agent(task, budget=12)
        assert run.stop_reason == "finish", task.task_id
        assert run.solved, task.task_id
        assert len(run.steps) == STEP_COUNTS[task.family], task.task_id
        assert run.steps[-1].action.tool == "finish"


def test_finish_is_required_to_terminate() -> None:
    task = example_task()

    # A policy that never calls finish runs the budget out, and is not solved
    # even though every call it made succeeded.
    run = run_agent(task, budget=4, policy="never")
    assert run.stop_reason == "budget"
    assert run.solved is False
    assert run.answer is None
    assert len(run.steps) == 4
    assert all(step.result.ok for step in run.steps)

    # A plan that never reaches RET is exhausted rather than terminated.
    from beyond_attention.agent import Task
    from dataclasses import replace

    endless = replace(task, plan=(instruction(OP_ADD, 1),), answer=1)
    exhausted = run_agent(endless, budget=6)
    assert exhausted.stop_reason == "plan_exhausted"
    assert len(exhausted.steps) == 1


def test_the_trace_replays_and_the_run_is_deterministic() -> None:
    task = next(t for t in task_suite(10, seed=3) if t.family == "branch")

    first = run_agent(task, budget=8)
    again = run_agent(task, budget=8)

    assert first.actions == again.actions
    assert [s.result for s in first.steps] == [s.result for s in again.steps]
    assert len(first.steps) == len(again.steps)

    replayed = replay(first, task.table)
    assert replayed == tuple(step.result for step in first.steps)
    # The trace alone is enough to see what happened: every recorded action is
    # a valid call, and the answer is the last one.
    assert all(validate_call(step.action) is None for step in first.steps)
    assert first.answer == task.answer


def test_bad_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_agent(example_task(), budget=0)
    with pytest.raises(ValueError):
        run_agent(example_task(), memory="vibes")
    with pytest.raises(ValueError):
        run_agent(example_task(), policy="guess")


# --------------------------------------------------------------------------
# The selective family: retention that depends on the input
# --------------------------------------------------------------------------

def test_the_selective_example_has_the_hand_computed_answer() -> None:
    """Key 1 holds 40, key 3 holds 60, key 2 holds 90, and 70 was a distractor."""
    task = selective_example_task()

    assert [i.op for i in task.plan] == [
        OP_PUT, OP_NOISE, OP_PUT, OP_PUT, OP_RECALL
    ]
    assert task.answer == 60
    assert evaluate_plan(
        (instruction(OP_PUT, 1, 40),
         instruction(OP_NOISE, 70),
         instruction(OP_PUT, 3, 60),
         instruction(OP_PUT, 2, 90),
         instruction(OP_RECALL, 3)), ()) == 60
    # A later PUT does not overwrite an earlier key, and the distractor is
    # stored nowhere at all.
    assert evaluate_plan(
        (instruction(OP_PUT, 1, 40),
         instruction(OP_PUT, 3, 60),
         instruction(OP_NOISE, 40),
         instruction(OP_RECALL, 1)), ()) == 40
    with pytest.raises(ValueError):
        evaluate_plan((instruction(OP_PUT, 1, 40),
                       instruction(OP_RECALL, 2)), ())


def test_a_stored_value_lives_only_in_its_memory_slot() -> None:
    """The value is not also parked in a named register for the policy to read.

    If it were, the policy could answer the query without the memory, and the
    family would stop testing anything.
    """
    x, delta = embed(instruction_event_of(OP_PUT, 3, 77), state_width=4)

    assert x[R_SLOT_BASE + 3] == 77.0
    assert delta[R_SLOT_BASE + 3] == DELTA_WRITE
    assert 77.0 not in x[:R_SLOT_BASE]
    # The key *is* in an instruction register: the key is the input, not memory.
    assert x[R_ARG] == 3.0


def test_a_put_writes_its_own_slot_and_leaves_the_others_alone() -> None:
    state = new_state(4)
    state = stream_step(state, *embed(instruction_event_of(OP_PUT, 1, 40),
                                      state_width=4))
    state = stream_step(state, *embed(instruction_event_of(OP_PUT, 3, 60),
                                      state_width=4))

    assert tuple(state[R_SLOT_BASE:]) == (0.0, 40.0, 0.0, 60.0)


def test_a_distractor_writes_no_slot_and_is_nowhere_in_the_state() -> None:
    """A NOISE event is seen -- its opcode moves -- and remembered nowhere."""
    state = new_state(4)
    state = stream_step(state, *embed(instruction_event_of(OP_PUT, 2, 55),
                                      state_width=4))
    before = state.copy()

    state = stream_step(state, *embed(instruction_event_of(OP_NOISE, 99),
                                      state_width=4))

    np.testing.assert_array_equal(state[R_SLOT_BASE:], before[R_SLOT_BASE:])
    assert 99.0 not in state[R_SLOT_BASE:]
    assert read_registers(state).op_code == OP_CODES[OP_NOISE]
    # The distractor's value *is* in the instruction register, because the event
    # was read; it is in no memory slot, which is the distinction the family is
    # built on.
    assert state[R_ARG] == 99.0


def test_the_state_holds_every_keyed_value_at_once() -> None:
    """The memory at the moment of the query, read out slot by slot.

    This is the mechanism the family exists to test: 40, 60 and 90 are all in
    the state when the query for key 3 is answered, and the distractor 70 is in
    none of them.
    """
    run = run_agent(selective_example_task(), budget=6)

    assert run.solved
    assert run.answer == 60
    assert run.steps[-1].registers.slots == (0.0, 40.0, 90.0, 60.0)
    assert all(step.action.tool == "add" for step in run.steps[:-1])
    assert run.steps[-1].action == ToolCall("finish", {"answer": 60})


def test_the_selective_family_needs_its_whole_stream_before_the_question() -> None:
    task = selective_suite(1, seed=0)[0]

    assert STEP_COUNTS["selective"] == len(task.plan)
    assert len(task.plan) == SELECTIVE_STORES + SELECTIVE_DISTRACTORS + 1
    for budget in range(1, len(task.plan)):
        run = run_agent(task, budget=budget)
        assert run.stop_reason == "budget"
        assert run.solved is False and run.answer is None
    run = run_agent(task, budget=len(task.plan))
    assert run.stop_reason == "finish" and run.solved
    assert len(run_agent(task, budget=50).steps) == len(task.plan)


def test_the_selective_trace_replays_and_the_run_is_deterministic() -> None:
    task = selective_suite(10, seed=3)[4]

    first = run_agent(task, budget=8)
    again = run_agent(task, budget=8)

    assert first.actions == again.actions
    assert [s.result for s in first.steps] == [s.result for s in again.steps]
    assert replay(first, task.table) == tuple(s.result for s in first.steps)
    assert all(validate_call(step.action) is None for step in first.steps)
    assert first.answer == task.answer


def test_a_malformed_call_cannot_end_the_selective_loop() -> None:
    calls = [ToolCall("add", {"a": 1}),           # missing b
             ToolCall("explode", {}),             # unknown tool
             ToolCall("finish", {"answer": 60})]  # the real answer, at last

    def policy(registers: Registers) -> ToolCall:
        del registers
        return calls.pop(0)

    run = run_agent(selective_example_task(), budget=8, policy=policy)

    assert run.solved
    assert [step.result.error for step in run.steps[:2]] == \
        ["bad_params", "unknown_tool"]


def test_the_scalar_carry_control_fails_the_selective_family() -> None:
    """A single register cannot hold two keys, asserted task by task.

    The correspondence is exact and it is the whole argument: the charitable
    scalar answers with the last *keyed* value it saw, so it is right exactly
    when the queried key is the one the last PUT named. A rate below 1.0 alone
    would be consistent with a merely harder task, which is why the per-task
    equality is what this test asserts. That it stays below 1.0 over fifty tasks
    is then a consequence rather than the evidence.
    """
    suite = task_suite(20, seed=2)
    for task in suite:
        with_state = run_agent(task, budget=8)
        with_scalar = run_agent(task, budget=8, memory="scalar")
        if task.family == "selective":
            last_stored = [i.args[0] for i in task.plan if i.op == OP_PUT][-1]
            queried = [i.args[0] for i in task.plan if i.op == OP_RECALL][0]
            assert with_state.solved, task.task_id
            assert with_scalar.solved == (queried == last_stored), task.task_id
        else:
            assert with_state.solved, task.task_id
            assert with_scalar.solved == with_state.solved, task.task_id
            assert with_scalar.actions == with_state.actions, task.task_id

    tasks = selective_suite(50, seed=0)
    solved = sum(run_agent(t, budget=6, memory="scalar").solved for t in tasks)
    assert 0 < solved < len(tasks), solved
    assert all(run_agent(t, budget=6).solved for t in tasks)


def test_the_fixed_decay_control_ties_the_scalar_and_stores_distractors() -> None:
    """A wide state with a constant gate is not a memory, measured both ways.

    With no distractor every slot ends up holding the last value written, which
    is exactly what one scalar holds -- so the width buys nothing. With a
    distractor after the last store it holds the distractor's value, which the
    charitable scalar does not even look at.
    """
    task = selective_example_task()
    fixed = run_agent(task, budget=6, memory="fixed", state_width=4)

    assert run_agent(task, budget=6).answer == 60
    assert run_agent(task, budget=6, memory="scalar").answer == 90
    assert fixed.answer == 90
    assert fixed.solved is False
    assert fixed.steps[-1].registers.slots == (90.0, 90.0, 90.0, 90.0)

    # A trailing distractor: the scalar still holds the last store, and the
    # constant gate replaces it with the noise.
    from beyond_attention.agent import Task

    plan = (instruction(OP_PUT, 1, 11),
            instruction(OP_PUT, 3, 22),
            instruction(OP_NOISE, 77),
            instruction(OP_RECALL, 1))
    trailing = Task("trailing", "selective", plan, (), 11, "by hand")
    assert run_agent(trailing, budget=4).answer == 11
    assert run_agent(trailing, budget=4, memory="scalar").answer == 22
    assert run_agent(trailing, budget=4, memory="fixed",
                     state_width=4).answer == 77


def test_the_fixed_decay_state_is_the_scalar_when_nothing_distracts() -> None:
    """The two controls coincide exactly where the gate has nothing to do."""
    tasks = selective_suite(20, seed=7, n_store=3, n_distractors=0, n_keys=4)
    for task in tasks:
        with_fixed = run_agent(task, budget=4, memory="fixed", state_width=4)
        with_scalar = run_agent(task, budget=4, memory="scalar")
        assert with_fixed.solved == with_scalar.solved, task.task_id


def test_a_distractor_breaks_the_fixed_decay_state_and_not_the_gated_one() -> None:
    """The gate, not the width: the same state width, one constant gate."""
    tasks = selective_suite(20, seed=7, n_store=3, n_distractors=4, n_keys=4)
    gated = sum(run_agent(t, budget=8).solved for t in tasks)
    fixed = sum(run_agent(t, budget=8, memory="fixed", state_width=4).solved
                for t in tasks)

    assert gated == len(tasks)
    assert fixed < gated


def test_the_no_memory_control_collapses_the_selective_family() -> None:
    for task in selective_suite(20, seed=1):
        wiped = run_agent(task, budget=6, memory="none")
        assert wiped.solved is False, task.task_id
        assert wiped.stop_reason == "finish"
        assert wiped.answer == 0 and wiped.answer != task.answer
        assert all(step.registers.observations == 0 for step in wiped.steps)
        assert run_agent(task, budget=6).solved, task.task_id


def test_the_state_width_is_the_capacity_that_matters() -> None:
    """Where the family starts to solve: at least as many slots as keys.

    Two anchors, both analytic. A one-slot state aliases every key onto slot 0,
    so it *is* the scalar -- the same solve set task by task, not merely a
    similar rate. A state as wide as the key space aliases nothing and solves
    every task. Everything between is the experiment's curve.
    """
    tasks = selective_suite(20, seed=0)
    for task in tasks:
        width = required_state_width(task)
        assert width == 1 + max(i.args[0] for i in task.plan
                                if i.op in (OP_PUT, OP_RECALL))
        assert run_agent(task, budget=6, state_width=width).solved, task.task_id
        assert run_agent(task, budget=6, state_width=1).solved == \
            run_agent(task, budget=6, memory="scalar").solved, task.task_id

    assert run_agent(tasks[0], budget=6, state_width=0).solved is False
    narrow = sum(run_agent(t, budget=6, state_width=1).solved for t in tasks)
    wide = sum(run_agent(t, budget=6).solved for t in tasks)
    assert wide == len(tasks)
    assert narrow < wide


def test_the_recall_reads_the_slot_its_key_addresses() -> None:
    """The accessor the policy uses, directly, including the aliasing rule."""
    slots = (0.0, 11.0, 22.0, 33.0)
    registers = Registers(carry=0.0, observations=0,
                          op_code=OP_CODES[OP_RECALL], arg=2, arg2=0,
                          miss=False, kind=0, instructions=1, slots=slots)

    assert slot_value(registers, 2) == 22
    assert slot_value(registers, 3) == 33
    assert choose_action(registers) == ToolCall("finish", {"answer": 22})

    # Aliasing is the documented behaviour of a narrow state, not an error: the
    # address is the key modulo the width.
    narrow = Registers(carry=0.0, observations=0,
                       op_code=OP_CODES[OP_RECALL], arg=3, arg2=0,
                       miss=False, kind=0, instructions=1, slots=(11.0, 22.0))
    assert slot_value(narrow, 3) == 22
    # A state with no memory slots holds no keyed value, and says so with 0.
    empty = Registers(carry=0.0, observations=0,
                      op_code=OP_CODES[OP_RECALL], arg=1, arg2=0,
                      miss=False, kind=0, instructions=1)
    assert slot_value(empty, 1) == 0


def test_the_selective_solve_rate_fails_a_broken_gate(monkeypatch) -> None:
    """The check the other checks rest on: break the gate, lose the family.

    A solve rate of 1.000 is worth nothing if it survives a memory that
    addresses nothing, so two specific faults are injected into ``embed`` and
    the family is required to stop being solved. The first removes the
    addressing -- every keyed event lands in slot 0, which is exactly what the
    one-slot state does -- and the second removes retention altogether. Both are
    faults a test that only counted solves would never see.
    """
    import beyond_attention.agent as agent

    tasks = selective_suite(10, seed=0)
    assert all(run_agent(t, budget=6).solved for t in tasks)

    original = agent.embed

    def slot_zero(event, state_width=0, fixed_gate=False):
        x, delta = original(event, state_width, fixed_gate)
        if event.kind == "instruction" and event.instruction.op == OP_PUT:
            address = event.instruction.args[0] % state_width
            if address:
                x[R_SLOT_BASE] = x[R_SLOT_BASE + address]
                delta[R_SLOT_BASE] = DELTA_WRITE
                x[R_SLOT_BASE + address] = 0.0
                delta[R_SLOT_BASE + address] = 0.0
        return x, delta

    def no_retention(event, state_width=0, fixed_gate=False):
        x, delta = original(event, state_width, fixed_gate)
        if event.kind == "instruction" and event.instruction.op == OP_PUT:
            delta[R_SLOT_BASE:] = 0.0
        return x, delta

    try:
        for mutation in (slot_zero, no_retention):
            monkeypatch.setattr(agent, "embed", mutation)
            solved = sum(run_agent(t, budget=6).solved for t in tasks)
            assert solved < len(tasks), mutation.__name__
            monkeypatch.setattr(agent, "embed", original)
    finally:
        monkeypatch.setattr(agent, "embed", original)

    assert all(run_agent(t, budget=6).solved for t in tasks)


# --------------------------------------------------------------------------
# The controls, which are the point
# --------------------------------------------------------------------------

def test_the_no_memory_control_fails_carry_over_and_passes_the_literal_task() -> (
        None
):
    """Both halves are asserted, because either alone is a worthless control.

    A control that failed everything would look like evidence for memory while
    actually showing the loop is broken; a control that passed everything would
    be evidence against nothing. The contrast -- the family whose answer is
    written in the task survives, every family whose answer is an intermediate
    result does not -- is the measurement.
    """
    suite = task_suite(20, seed=1)

    factual = [t for t in suite if t.family == "literal"]
    assert factual
    for task in factual:
        assert run_agent(task, budget=6, memory="none").solved, task.task_id

    for family in FAMILIES:
        if family == "literal":
            continue
        tasks = [t for t in suite if t.family == family]
        assert tasks
        for task in tasks:
            wiped = run_agent(task, budget=6, memory="none")
            assert wiped.solved is False, task.task_id
            assert wiped.stop_reason == "finish"  # it still terminates...
            assert wiped.answer != task.answer     # ...with the wrong answer
            # ...and the only reason is the missing carry.
            assert all(step.registers.observations == 0 for step in wiped.steps)
            assert run_agent(task, budget=6, memory="ssm").solved, task.task_id


def test_the_scalar_carry_control_ties_the_arithmetic_families() -> None:
    """Where one register is enough, the recurrence earns nothing measurable.

    This is the control behind the README's third false claim, and it is scoped
    to the five arithmetic families on purpose. On the `selective` family the
    same control fails, which is the test above and the reason that family
    exists. Publishing only the half that flatters the architecture would be the
    easy version; publishing only the half that flatters the control would be
    the other one.
    """
    suite = task_suite(20, seed=2)
    arithmetic = [t for t in suite if t.family != "selective"]
    assert arithmetic
    for task in arithmetic:
        with_state = run_agent(task, budget=8)
        with_scalar = run_agent(task, budget=8, memory="scalar")
        assert with_state.solved, task.task_id
        assert with_scalar.solved == with_state.solved, task.task_id
        assert with_scalar.actions == with_state.actions, task.task_id


def test_the_one_step_agent_cannot_solve_the_suite() -> None:
    """The task-difficulty control: the suite is not one call deep."""
    suite = task_suite(20, seed=4)
    solved = [t for t in suite if run_agent(t, budget=1).solved]

    assert solved, "the budget-1 agent must solve *something*, or the control is broken"
    assert {t.family for t in solved} == {"literal"}
    assert len(solved) < len(suite)
    # Only the one family whose answer is in the task text, and only that one.
    assert len(solved) == sum(1 for t in suite if t.family == "literal")


def test_the_random_action_control_is_near_the_floor() -> None:
    """Random tool choice must not look like a solver.

    The floor is not zero -- the random agent may call ``finish`` with an answer
    that happens to be right -- so this asserts a bound rather than an exact
    zero, and the measured rate is recorded in the results file alongside it.
    """
    suite = task_suite(20, seed=5)
    trials = 0
    solved = 0
    for index, task in enumerate(suite):
        for round_index in range(10):
            run = run_agent(task, budget=8, policy="random",
                            seed=1_000 * index + round_index)
            trials += 1
            solved += int(run.solved)

    # Counted per draw rather than per task, and the suite grew when the
    # selective family was added, so the total is derived rather than typed.
    assert trials == len(suite) * 10
    assert solved / trials < 0.05, f"random action solved {solved}/{trials}"


# --------------------------------------------------------------------------
# The policy is a function of the state, and the branch reads it
# --------------------------------------------------------------------------

def test_the_policy_is_a_pure_function_of_the_state() -> None:
    state = np.zeros(AGENT_DIM)
    state = stream_step(state, *embed(instruction_event_of(OP_MUL, 5)))
    state = stream_step(state, *embed(observation_event(make_step_result(6))))

    first = choose_action(read_registers(state))
    again = choose_action(read_registers(state.copy()))
    assert first.signature() == again.signature() == ("mul", (("a", 6), ("b", 5)))


def test_the_branch_instruction_takes_its_tool_from_the_state() -> None:
    """The one instruction whose *tool*, not just its operand, is state-driven."""
    positive = Registers(carry=4.0, observations=1, op_code=7, arg=3, arg2=0,
                         miss=False, kind=1, instructions=2)
    negative = Registers(carry=-4.0, observations=1, op_code=7, arg=3, arg2=0,
                         miss=False, kind=1, instructions=2)

    assert choose_action(positive).tool == "mul"
    assert choose_action(negative).tool == "add"
    # The same instruction, the same operand, a different tool: the only
    # difference is the carried value, and it came out of the state.
    assert choose_action(positive).args["b"] == choose_action(negative).args["b"]

    # And a real task reaches both arms, or the branch would be decoration.
    tasks = [t for t in task_suite(50, seed=0) if t.family == "branch"]
    tools = {run_agent(t, budget=6).steps[2].action.tool for t in tasks}
    assert tools == {"mul", "add"}


def test_the_carry_is_zero_when_nothing_has_been_observed() -> None:
    """The default is not a guess that can be right: no answer in the suite is 0."""
    registers = read_registers(np.zeros(AGENT_DIM))
    assert carried_value(registers) == 0
    assert registers.observations == 0

    for task in task_suite(30, seed=6):
        assert task.answer != 0, task.task_id


def test_suite_tables_have_exactly_one_key_per_value() -> None:
    """The lookup step is well posed, so a miss means a miss."""
    for task in task_suite(10, seed=8):
        values = [value for _, value in task.table]
        assert len(set(values)) == len(values), task.task_id
        if task.family == "lookup":
            assert task.answer in dict(task.table), task.task_id


def test_the_state_width_and_the_decay_are_the_documented_ones() -> None:
    """The register file is exactly the eight named registers, plus any slots."""
    assert A_DIAG.shape == (AGENT_DIM,)
    assert A_DIAG[R_CARRY] == A_HOLD
    assert A_DIAG[R_COUNT] == 0.0
    assert A_DIAG[R_INDEX] == 0.0
    assert A_DIAG[R_OP] == A_HOLD
    assert A_DIAG[R_MISS] == A_HOLD

    # A memory slot holds what it is written until it is written again, so its
    # decay is the same exact-write constant the carry register uses.
    wide = decay_vector(AGENT_DIM + 3)
    assert wide.shape == (AGENT_DIM + 3,)
    np.testing.assert_array_equal(wide[:AGENT_DIM], A_DIAG)
    np.testing.assert_array_equal(wide[AGENT_DIM:], np.full(3, A_HOLD))
    with pytest.raises(ValueError):
        decay_vector(AGENT_DIM - 1)
