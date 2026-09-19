"""An agentic loop whose working memory is a selective state-space state.

The operator's aim for this project was for it to *act* -- to run a loop with
tools rather than only consume input. This module is that loop, and it is a
small one. It contains:

* a **tool registry** with strict input schemas and pure, deterministic tools:
  ``add``/``mul``/``sub`` over bounded integers, ``lookup`` over a bounded
  key/value table supplied with the task, and ``finish``, which ends the loop and
  returns the answer;
* a **loop** with an explicit step budget that observes, chooses, executes,
  appends the observation and repeats, recording every step so the run can be
  replayed;
* a **hand-designed, deterministic policy** -- not a learned one, and not an
  external model. It reads its next action out of the **selective state-space
  recurrence**: the trajectory's events are embedded into the ``(x, delta)``
  pair ``build_scan_terms`` consumes, streamed through the same recurrence the
  rest of this repository implements, and the resulting state is decoded into an
  opcode, an operand and the carried value.

The last point is where an overclaim is easiest, so it is worth being exact. The
state here is the S6 recurrence with **hand-set** parameters:

    a_t = exp(delta_t * A)          # A diagonal, negative
    h_t = a_t * h_{t-1} + delta_t * x_t

``A`` is chosen so that a *write* (``delta = 1``) replaces a register exactly --
``exp(-800)`` underflows to ``0.0`` in float64 -- and a *hold* (``delta = 0``)
leaves it bit-for-bit unchanged. That read/write gate is the mechanism the
model's learned ``delta`` provides; here it is set by hand. Nothing is trained,
and there is no gradient anywhere in this path. ``tests/test_agent.py`` checks this numpy
recurrence against the repository's own ``selective_scan`` on the same inputs, so
"this is the SSM's recurrence" is a check rather than a claim.

What it is **not**, stated here rather than left to be discovered:

* **It is not a general agent.** The task text is a closed instruction grammar
  (``ADD``/``MUL``/``SUB``/``KEY``/``IFPOS``/``RET``) delivered as structured
  events, not natural language. There is no tokenizer, no language
  understanding, and no open-ended tool use: the tool set is five functions
  fixed at import time.
* **It does not learn.** The policy is hand-written branching, the SSM
  parameters are constants, and a seed only decides which tasks get generated.
* **The SSM is not what makes it work.** ``experiments/agent_loop.py`` runs the
  same controller with the carried value held in one Python int and solves the
  same tasks, and it runs a no-memory control that collapses on every task whose
  answer is an intermediate result while still solving the ones whose answer is
  written in the task text. The recurrence carries the value; it is not the
  source of the capability, and both controls are published rather than one.
* **The tools cannot touch anything.** No network, no filesystem, no clock. A
  test parses this module's imports and asserts that ``numpy`` is the only one.

Only ``numpy`` is used, and every run is deterministic given its seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# The register file. The state is AGENT_DIM wide and every dimension has a
# documented job, because a state whose contents are not named cannot be read
# out honestly.
# --------------------------------------------------------------------------

AGENT_DIM = 8

R_CARRY = 0      # the value the last action returned -- the agent's memory
R_COUNT = 1      # how many observations have been written (accumulator)
R_OP = 2         # the current instruction's opcode, written from the task text
R_ARG = 3        # the current instruction's first operand
R_ARG2 = 4       # the current instruction's second operand
R_MISS = 5       # 1.0 if the last observation was "no such key"
R_KIND = 6       # the last observation's kind (see OBS_*)
R_INDEX = 7      # how many instructions have been streamed (accumulator)

REGISTER_NAMES = (
    "carry", "observations", "op", "arg", "arg2", "miss", "kind", "instructions",
)

# `A` is stored negated, as the rest of the repository stores it, so that
# `a_t = exp(delta_t * A)`. Two values are used:
#
#   A_HOLD: with delta=1 the previous contents are multiplied by exp(-800),
#           which underflows to exactly 0.0 in float64, so a write *replaces*
#           the register rather than nearly replacing it. With delta=0 the
#           multiplier is exp(0) = 1 and the additive term is 0, so a hold is
#           exact too. Both properties are load-bearing: `IFPOS` tests the sign
#           of the carry, and the first version of this module used exp(-50) ~
#           2e-22, which left 1.7e-21 of a previous value behind -- greater than
#           zero, so the branch took the wrong arm on 8 of 50 branch tasks.
#           Making the write exact is what removes the tolerance that would
#           otherwise have had to be threaded through every sign test.
#   A_KEEP: a multiplier of exactly 1 whatever delta is, so the additive term
#           accumulates. Used for the two counters.
A_HOLD = -800.0
A_KEEP = 0.0

A_DIAG = np.array(
    [A_HOLD, A_KEEP, A_HOLD, A_HOLD, A_HOLD, A_HOLD, A_HOLD, A_KEEP],
    dtype=np.float64,
)

DELTA_HOLD = 0.0
DELTA_WRITE = 1.0

# Observation kinds, written into R_KIND.
OBS_NONE = 0
OBS_VALUE = 1
OBS_MISS = 2
OBS_ERROR = 3

# --------------------------------------------------------------------------
# Instructions. The task text is a sequence of these; each one is streamed into
# the state before the decision it belongs to.
# --------------------------------------------------------------------------

OP_ADD = "ADD"        # acc = acc + arg
OP_MUL = "MUL"        # acc = acc * arg
OP_SUB = "SUB"        # acc = acc - arg
OP_KEY = "KEY"        # acc = the key whose value equals acc
OP_IFPOS = "IFPOS"    # acc = acc * arg if acc > 0 else acc + arg
OP_RET = "RET"        # finish with the carried value
OP_RETLIT = "RETLIT"  # finish with the literal operand (the one-step family)

OP_CODES = {
    OP_RET: 1,
    OP_RETLIT: 2,
    OP_ADD: 3,
    OP_MUL: 4,
    OP_SUB: 5,
    OP_KEY: 6,
    OP_IFPOS: 7,
}
CODE_OPS = {code: op for op, code in OP_CODES.items()}
NONE_CODE = 0  # nothing has been streamed into R_OP yet


@dataclass(frozen=True)
class Instruction:
    """One step of the task text.

    ``args`` is empty, one or two bounded integers depending on ``op``. It is a
    tuple rather than a list so that a task is hashable and cannot be mutated
    underneath a run.
    """

    op: str
    args: tuple[int, ...] = ()


def instruction(op: str, *args: int) -> Instruction:
    """Build an instruction, rejecting anything the grammar does not define."""
    if op not in OP_CODES:
        raise ValueError(f"unknown instruction {op!r}")
    arity = {OP_ADD: 1, OP_MUL: 1, OP_SUB: 1, OP_KEY: 0, OP_IFPOS: 1,
             OP_RET: 0, OP_RETLIT: 1}[op]
    if len(args) != arity:
        raise ValueError(f"{op} takes {arity} operand(s), got {len(args)}")
    return Instruction(op=op, args=tuple(int(a) for a in args))


# --------------------------------------------------------------------------
# Tools: strict schemas, pure functions, and typed errors rather than
# exceptions. Nothing here reads a clock, a file or a socket.
# --------------------------------------------------------------------------

ARG_BOUND = 10_000  # the schema's bound on every integer argument

# Where the random-action control draws its arguments from. See
# `random_action` for why this is not ARG_BOUND.
RANDOM_ARG_SPAN = 100


@dataclass(frozen=True)
class Param:
    name: str
    low: int = -ARG_BOUND
    high: int = ARG_BOUND


@dataclass(frozen=True)
class ToolSchema:
    name: str
    params: tuple[Param, ...]
    summary: str


TOOL_SCHEMAS: dict[str, ToolSchema] = {
    "add": ToolSchema("add", (Param("a"), Param("b")), "a + b"),
    "mul": ToolSchema("mul", (Param("a"), Param("b")), "a * b"),
    "sub": ToolSchema("sub", (Param("a"), Param("b")), "a - b"),
    "lookup": ToolSchema(
        "lookup", (Param("value"),),
        "the key paired with value in the task's table, or a miss",
    ),
    "finish": ToolSchema("finish", (Param("answer"),), "end the loop"),
}

TOOL_NAMES = tuple(TOOL_SCHEMAS)  # insertion order: add, mul, sub, lookup, finish

ToolTable = Sequence[tuple[int, int]]


@dataclass(frozen=True)
class ToolCall:
    """A proposed call. Nothing about it is trusted until it is validated."""

    tool: str
    args: dict[str, int]

    def signature(self) -> tuple[str, tuple[tuple[str, int], ...]]:
        """A hashable, order-stable rendering, used to compare two calls."""
        return (self.tool, tuple(sorted(self.args.items())))


@dataclass(frozen=True)
class ToolResult:
    """A validated outcome: a value, or a typed error, or a valid miss."""

    ok: bool
    value: int | None
    error: str | None

    @property
    def kind(self) -> int:
        if self.ok:
            return OBS_VALUE
        return OBS_MISS if self.error == "no_key" else OBS_ERROR


def validate_call(call: ToolCall) -> str | None:
    """Return a typed error code, or ``None`` if the call matches its schema.

    Rejection is a value, not an exception: a malformed call must not be able to
    end the loop. ``bool`` is refused explicitly because it is a subclass of
    ``int`` and ``True`` as an operand is a caller bug, not an integer.
    """
    schema = TOOL_SCHEMAS.get(call.tool)
    if schema is None:
        return "unknown_tool"
    expected = {p.name for p in schema.params}
    if set(call.args) != expected:
        return "bad_params"
    for param in schema.params:
        value = call.args[param.name]
        if isinstance(value, bool) or not isinstance(value, int):
            return "bad_type"
        if not (param.low <= value <= param.high):
            return "out_of_range"
    return None


def execute(call: ToolCall, table: ToolTable = ()) -> ToolResult:
    """Run one validated call. Pure: the same call and table give the same result."""
    error = validate_call(call)
    if error is not None:
        return ToolResult(ok=False, value=None, error=error)

    tool = call.tool
    if tool == "add":
        value = call.args["a"] + call.args["b"]
    elif tool == "mul":
        value = call.args["a"] * call.args["b"]
    elif tool == "sub":
        value = call.args["a"] - call.args["b"]
    elif tool == "lookup":
        wanted = call.args["value"]
        found = [key for key, paired in table if paired == wanted]
        if len(found) != 1:
            # Zero hits is a miss. More than one is an ambiguous table, which is
            # a malformed task rather than a failed lookup, and is reported as
            # such instead of silently taking the first.
            return ToolResult(
                ok=False, value=None,
                error="no_key" if not found else "ambiguous_table",
            )
        value = found[0]
    else:  # finish -- there is no arithmetic, the answer is what was passed
        value = call.args["answer"]

    # The bound is on the arguments, not on the result, and a runaway product
    # would silently leave float64's exact-integer range -- which would make the
    # state's readout wrong in a way no test looking at the answer would catch.
    if abs(value) > 2**52:
        return ToolResult(ok=False, value=None, error="overflow")
    return ToolResult(ok=True, value=value, error=None)


# --------------------------------------------------------------------------
# The recurrence. This is the memory.
# --------------------------------------------------------------------------

def new_state() -> np.ndarray:
    """A zeroed state. The only allocation; streaming reuses the array shape."""
    return np.zeros(AGENT_DIM, dtype=np.float64)


def stream_step(state: np.ndarray, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """One timestep: ``h = exp(delta * A) * h + delta * x``.

    ``x`` and ``delta`` are both AGENT_DIM wide, because the gate is per
    register: an observation writes the carry without disturbing the
    instruction, and an instruction writes the instruction without disturbing
    the carry. That is the selective part, and here it is hand-set rather than
    produced by a projection.
    """
    a = np.exp(delta * A_DIAG)
    return a * state + delta * x


def scan_states(x: np.ndarray, delta: np.ndarray,
                state: np.ndarray | None = None) -> np.ndarray:
    """The recurrence over a whole sequence, returning the state at each step.

    ``x`` and ``delta`` are ``(L, AGENT_DIM)``. The returned array is
    ``(L, AGENT_DIM)``: row ``t`` is the state *after* consuming token ``t``,
    which is what a decision at that point observes.
    """
    h = new_state() if state is None else state
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        h = stream_step(h, x[t], delta[t])
        out[t] = h
    return out


def bridge_to_selective_scan(
    x: np.ndarray, delta: np.ndarray
) -> dict[str, np.ndarray]:
    """The same input, in the shapes ``selective_scan`` documents.

    ``A`` is ``(D, N)`` with ``N = 1``, ``B`` and ``C`` are ``(B, L, N)`` ones,
    so the readout ``y_t = sum_n C_t[n] h_t[d, n]`` is exactly ``h_t[d, 0]``.
    ``tests/test_agent.py`` runs the repository's ``selective_scan`` on this
    dict and asserts it reproduces ``scan_states`` above -- which is what makes
    "the memory is the SSM's recurrence" a measurement.
    """
    length = x.shape[0]
    return {
        "x": x.reshape(1, length, AGENT_DIM),
        "delta": delta.reshape(1, length, AGENT_DIM),
        "A": A_DIAG.reshape(AGENT_DIM, 1),
        "B": np.ones((1, length, 1)),
        "C": np.ones((1, length, 1)),
    }


# --------------------------------------------------------------------------
# Events, and their embedding into (x, delta).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """One event of the trajectory: an instruction, an action, or an observation.

    The trajectory records all three because a replayable trace needs all three.
    Only instructions and observations write to the state; an action is recorded
    and contributes nothing, because the agent's memory is of the task and its
    results, not a log of what it did.
    """

    kind: str                 # "instruction" | "action" | "observation"
    instruction: Instruction | None = None
    call: ToolCall | None = None
    result: ToolResult | None = None


def instruction_event(instr: Instruction) -> Event:
    return Event(kind="instruction", instruction=instr)


def action_event(call: ToolCall) -> Event:
    return Event(kind="action", call=call)


def observation_event(result: ToolResult) -> Event:
    return Event(kind="observation", result=result)


def embed(event: Event) -> tuple[np.ndarray, np.ndarray]:
    """Map an event to the ``(x, delta)`` pair the recurrence consumes.

    An action embeds to ``(zeros, zeros)``: it is in the transcript and in no
    register. An instruction writes only the instruction registers, an
    observation only the memory registers.
    """
    x = np.zeros(AGENT_DIM, dtype=np.float64)
    delta = np.zeros(AGENT_DIM, dtype=np.float64)

    if event.kind == "instruction":
        assert event.instruction is not None
        instr = event.instruction
        x[R_OP] = OP_CODES[instr.op]
        delta[R_OP] = DELTA_WRITE
        # Both operand registers are written on *every* instruction, with 0 for
        # an instruction that has no operand. Writing only the operands that
        # exist would leave the previous instruction's value in place, so `arg`
        # would mean "the last operand seen" rather than "this instruction's
        # operand" -- and a later instruction with an optional operand would
        # silently read a stale one.
        x[R_ARG] = float(instr.args[0]) if instr.args else 0.0
        delta[R_ARG] = DELTA_WRITE
        x[R_ARG2] = float(instr.args[1]) if len(instr.args) > 1 else 0.0
        delta[R_ARG2] = DELTA_WRITE
        x[R_INDEX] = 1.0
        delta[R_INDEX] = DELTA_WRITE
    elif event.kind == "observation":
        assert event.result is not None
        result = event.result
        x[R_CARRY] = float(result.value) if result.value is not None else 0.0
        delta[R_CARRY] = DELTA_WRITE
        x[R_COUNT] = 1.0
        delta[R_COUNT] = DELTA_WRITE
        x[R_KIND] = float(result.kind)
        delta[R_KIND] = DELTA_WRITE
        x[R_MISS] = 1.0 if result.error == "no_key" else 0.0
        delta[R_MISS] = DELTA_WRITE
    elif event.kind == "action":
        pass  # recorded, not remembered -- see the Event docstring
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown event kind {event.kind!r}")

    return x, delta


def embed_all(events: Sequence[Event]) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return np.zeros((0, AGENT_DIM)), np.zeros((0, AGENT_DIM))
    pairs = [embed(e) for e in events]
    return (
        np.stack([p[0] for p in pairs]),
        np.stack([p[1] for p in pairs]),
    )


# --------------------------------------------------------------------------
# Reading the state, and the policy that acts on it.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Registers:
    """The state, decoded. Every field is a register the recurrence wrote."""

    carry: float
    observations: int
    op_code: int
    arg: int
    arg2: int
    miss: bool
    kind: int
    instructions: int


def read_registers(state: np.ndarray) -> Registers:
    """Decode the state.

    A write is exact (``exp(-800)`` underflows), so every integer register is
    recovered by rounding with nothing to absorb, and the carry is read back as
    the float it was written as. The tests assert the carry *equals* the
    analytic value rather than approximating it, which is why no tolerance is
    used here: a tolerance would hide exactly the drift those tests exist to
    find.
    """
    return Registers(
        carry=float(state[R_CARRY]),
        observations=int(round(state[R_COUNT])),
        op_code=int(round(state[R_OP])),
        arg=int(round(state[R_ARG])),
        arg2=int(round(state[R_ARG2])),
        miss=bool(round(state[R_MISS])),
        kind=int(round(state[R_KIND])),
        instructions=int(round(state[R_INDEX])),
    )


def carried_value(registers: Registers) -> int:
    """What the agent believes its last result was.

    With no observation written the carry is 0, which is the honest reading: a
    state that has seen no result holds no result. The generator guarantees no
    task's answer is 0, so this default cannot accidentally be right -- see
    ``task_suite``.
    """
    if registers.observations == 0:
        return 0
    return int(round(registers.carry))


def choose_action(registers: Registers) -> ToolCall:
    """The hand-designed controller. A pure function of the decoded state.

    No branching on the task, the step index, or anything else: given the state
    the action is determined. ``tests/test_agent.py`` asserts that, and asserts
    that the branch instruction really does depend on the state by flipping the
    sign of the carry and requiring the tool to change.
    """
    op = CODE_OPS.get(registers.op_code)
    if op is None:
        # No instruction has been streamed. Terminating on the carried value is
        # the only non-invented action available.
        return ToolCall("finish", {"answer": carried_value(registers)})

    carry = carried_value(registers)
    arg = registers.arg
    if op == OP_ADD:
        return ToolCall("add", {"a": carry, "b": arg})
    if op == OP_MUL:
        return ToolCall("mul", {"a": carry, "b": arg})
    if op == OP_SUB:
        return ToolCall("sub", {"a": carry, "b": arg})
    if op == OP_KEY:
        return ToolCall("lookup", {"value": carry})
    if op == OP_RET:
        return ToolCall("finish", {"answer": carry})
    if op == OP_RETLIT:
        return ToolCall("finish", {"answer": arg})
    if op == OP_IFPOS:
        # The one instruction whose *tool* is chosen from the state rather than
        # from the task text.
        if registers.carry > 0:
            return ToolCall("mul", {"a": carry, "b": arg})
        return ToolCall("add", {"a": carry, "b": arg})
    raise AssertionError(f"unhandled op {op!r}")  # pragma: no cover


def random_action(rng: np.random.Generator,
                  span: int = RANDOM_ARG_SPAN) -> ToolCall:
    """A uniformly chosen tool with uniformly chosen arguments.

    This is the floor the policy has to beat, and it is allowed to call
    ``finish`` so that it is not handicapped out of ever succeeding.

    Arguments are drawn from ``[-span, span]`` rather than from the schema's
    full ``+/-ARG_BOUND``. The schema bound exists to reject nonsense, not to
    describe where answers live: drawing a finish answer uniformly from 20,001
    values would make the measured floor a statement about the schema's width
    instead of about the task. The span used here is recorded in the results
    file, so which floor was measured is not left to the reader.
    """
    tool = TOOL_NAMES[int(rng.integers(len(TOOL_NAMES)))]
    schema = TOOL_SCHEMAS[tool]
    args = {
        param.name: int(rng.integers(-span, span + 1))
        for param in schema.params
    }
    return ToolCall(tool, args)


def never_finish_action(registers: Registers) -> ToolCall:
    """A policy that never terminates, for the "finish is required" test."""
    del registers
    return ToolCall("add", {"a": 0, "b": 1})


# --------------------------------------------------------------------------
# Tasks.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """A verifiable task: a plan, a lookup table, and an analytic answer."""

    task_id: str
    family: str
    plan: tuple[Instruction, ...]
    table: tuple[tuple[int, int], ...]
    answer: int
    text: str


def evaluate_plan(plan: Sequence[Instruction],
                  table: ToolTable) -> int:
    """The task's answer, by definition, in Python integers.

    This is the ground truth the loop is graded against, and it is deliberately
    a separate, slower piece of code from the loop: ``run_agent`` decides when to
    stop and what to call, and this decides what the answer is. They share the
    instruction *names* and nothing else.
    """
    acc = 0
    for instr in plan:
        if instr.op == OP_ADD:
            acc = acc + instr.args[0]
        elif instr.op == OP_MUL:
            acc = acc * instr.args[0]
        elif instr.op == OP_SUB:
            acc = acc - instr.args[0]
        elif instr.op == OP_IFPOS:
            acc = acc * instr.args[0] if acc > 0 else acc + instr.args[0]
        elif instr.op == OP_KEY:
            found = [key for key, paired in table if paired == acc]
            if len(found) != 1:
                raise ValueError(f"table has {len(found)} keys with value {acc}")
            acc = found[0]
        elif instr.op == OP_RETLIT:
            return instr.args[0]
        elif instr.op == OP_RET:
            return acc
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown op {instr.op!r}")
    raise ValueError("plan does not terminate: no RET instruction")


def task_text(plan: Sequence[Instruction]) -> str:
    """The plan as a sentence. This is the 'task text' the README quote uses.

    It is a rendering for humans. The loop never parses it: it is streamed as
    the structured instructions it was rendered from, which is the honest
    description of what the task input actually is.
    """
    parts: list[str] = []
    for instr in plan:
        if instr.op == OP_ADD:
            parts.append(f"add {instr.args[0]}")
        elif instr.op == OP_MUL:
            parts.append(f"multiply the result by {instr.args[0]}")
        elif instr.op == OP_SUB:
            parts.append(f"subtract {instr.args[0]}")
        elif instr.op == OP_IFPOS:
            parts.append(
                f"if the result is positive multiply it by {instr.args[0]}, "
                f"otherwise add {instr.args[0]}"
            )
        elif instr.op == OP_KEY:
            parts.append("report the key whose value equals the result")
        elif instr.op == OP_RETLIT:
            parts.append(f"report {instr.args[0]}")
        elif instr.op == OP_RET:
            parts.append("report the result")
    return ", then ".join(parts)


# The families, and how many loop steps each one needs. A task is solved within
# a budget exactly when its family's step count fits inside it, which is what
# makes the budget curve a ceiling rather than a mystery.
STEP_COUNTS = {
    "literal": 1,   # RETLIT v          -- the answer is written in the task
    "one_op": 2,    # ADD a, RET
    "two_op": 3,    # MUL a, SUB b, RET
    "branch": 4,    # ADD a, SUB c, IFPOS b, RET
    "lookup": 5,    # ADD a, MUL b, SUB c, KEY, RET
}
FAMILIES = tuple(STEP_COUNTS)
CARRY_FAMILIES = tuple(f for f in FAMILIES if STEP_COUNTS[f] > 1)


def _build_task(family: str, rng: np.random.Generator, task_id: str) -> Task:
    """One task of a family, with its table and its analytic answer.

    Operands are drawn from 2..9 and there are at most three of them, so every
    intermediate value stays far inside float64's exact-integer range -- the
    state's readout is a float, and a task whose answer needed 60 bits would be
    testing the tolerance rather than the mechanism.
    """
    operands = [int(v) for v in rng.integers(2, 10, size=4)]
    table: tuple[tuple[int, int], ...] = ()

    if family == "literal":
        plan = (instruction(OP_RETLIT, operands[0]),)
    elif family == "one_op":
        plan = (instruction(OP_ADD, operands[0]), instruction(OP_RET))
    elif family == "two_op":
        plan = (instruction(OP_MUL, operands[0]),
                instruction(OP_SUB, operands[1]), instruction(OP_RET))
    elif family == "branch":
        plan = (instruction(OP_ADD, operands[0]),
                instruction(OP_SUB, operands[1]),
                instruction(OP_IFPOS, operands[2]),
                instruction(OP_RET))
    elif family == "lookup":
        head = (instruction(OP_ADD, operands[0]),
                instruction(OP_MUL, operands[1]),
                instruction(OP_SUB, operands[2]))
        acc = evaluate_plan(head + (instruction(OP_RET),), ())
        # The key the agent must find, plus decoys that do not collide with it.
        key = int(rng.integers(1, 100))
        decoys: list[tuple[int, int]] = []
        used = {acc}
        for candidate in rng.permutation(100):
            if len(decoys) >= 3:
                break
            other_key = int(candidate) + 1
            other_value = int(rng.integers(1, 100))
            if other_key == key or other_value in used:
                continue
            used.add(other_value)
            decoys.append((other_key, other_value))
        table = tuple(sorted([(key, acc)] + decoys))
        plan = head + (instruction(OP_KEY), instruction(OP_RET))
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown family {family!r}")

    answer = evaluate_plan(plan, table)
    return Task(
        task_id=task_id,
        family=family,
        plan=plan,
        table=table,
        answer=answer,
        text=task_text(plan),
    )


def task_suite(tasks_per_family: int = 50, seed: int = 0) -> tuple[Task, ...]:
    """A seeded suite, one family after another.

    Two properties the generator enforces, both relied on by the controls:

    * **No answer is 0.** The no-memory control substitutes 0 for every carried
      value, so a task whose answer is 0 would be solved by forgetting.
    * **Every lookup hits exactly one key.** A table with two keys sharing a
      value would make the answer ambiguous rather than hard.
    """
    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be >= 1")
    tasks: list[Task] = []
    for family in FAMILIES:
        made = 0
        attempt = 0
        while made < tasks_per_family:
            # Per-task seeds, so a task does not depend on how many were drawn
            # before it in its family.
            rng = np.random.default_rng((seed * 1_000_003) + made * 97 + attempt)
            task = _build_task(family, rng, f"{family}-{made}")
            attempt += 1
            if task.answer == 0:
                continue
            tasks.append(task)
            made += 1
    return tuple(tasks)


def example_task() -> Task:
    """The task the README quotes: add 3 and 4, multiply by 5, look up the key.

    Written out by hand rather than generated, so the trace published in the
    README has a hand-checkable answer: 3 + 4 = 7, 7 * 5 = 35, and 35 is paired
    with key 7 in this table.
    """
    plan = (
        instruction(OP_ADD, 3),
        instruction(OP_ADD, 4),
        instruction(OP_MUL, 5),
        instruction(OP_KEY),
        instruction(OP_RET),
    )
    table = ((2, 11), (3, 42), (5, 99), (7, 35))
    return Task(
        task_id="example",
        family="example",
        plan=plan,
        table=table,
        answer=evaluate_plan(plan, table),
        text=task_text(plan),
    )


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """One turn: what was read, chosen, and observed, and the state behind it."""

    index: int
    instruction: Instruction
    action: ToolCall
    result: ToolResult
    registers: Registers


@dataclass(frozen=True)
class AgentRun:
    """A finished (or exhausted) run, with everything needed to replay it."""

    task_id: str
    family: str
    memory: str
    budget: int
    steps: tuple[Step, ...]
    answer: int | None
    solved: bool
    stop_reason: str  # "finish" | "budget" | "plan_exhausted"

    @property
    def actions(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return tuple(step.action.signature() for step in self.steps)


Policy = Callable[[Registers], ToolCall]


def run_agent(
    task: Task,
    budget: int = 6,
    memory: str = "ssm",
    policy: str | Policy = "plan",
    seed: int = 0,
) -> AgentRun:
    """Run the loop until ``finish`` or the budget runs out.

    ``memory`` selects what carries the last result between decisions:

    * ``"ssm"`` -- the selective-scan state implemented above, which is the
      subject of the measurement;
    * ``"scalar"`` -- one Python int, updated by the same rule. This is the
      control for "is the recurrence doing anything the carry is not";
    * ``"none"`` -- wiped before every decision, with only the current
      instruction re-streamed. This is the control for "does carrying the value
      matter at all": the instruction registers refill from the task text, so
      the agent still knows which tool to reach for, and only the carried result
      is gone.

    ``policy`` is ``"plan"``, ``"random"``, ``"never"``, or any callable taking
    the decoded state and returning a (possibly malformed) call.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if memory not in ("ssm", "scalar", "none"):
        raise ValueError(f"unknown memory {memory!r}")

    rng = np.random.default_rng(seed)
    state = new_state()
    carry: int | None = None  # used by memory="scalar" only

    def act(regs: Registers) -> ToolCall:
        if callable(policy):
            return policy(regs)
        if policy == "plan":
            return choose_action(regs)
        if policy == "random":
            return random_action(rng)
        if policy == "never":
            return never_finish_action(regs)
        raise ValueError(f"unknown policy {policy!r}")

    steps: list[Step] = []
    for index in range(budget):
        if index >= len(task.plan):
            return AgentRun(task.task_id, task.family, memory, budget, tuple(steps),
                            None, False, "plan_exhausted")

        instr = task.plan[index]

        if memory == "scalar":
            # Same controller, same instruction: only the substrate for the
            # carried value differs.
            x, delta = embed(instruction_event(instr))
            state = stream_step(state, x, delta)
            regs = read_registers(state)
            regs = Registers(
                carry=float(carry) if carry is not None else 0.0,
                observations=0 if carry is None else 1,
                op_code=regs.op_code, arg=regs.arg, arg2=regs.arg2,
                miss=regs.miss, kind=regs.kind, instructions=regs.instructions,
            )
        else:
            if memory == "none":
                state = new_state()  # forget everything, then re-read the task
            x, delta = embed(instruction_event(instr))
            state = stream_step(state, x, delta)
            regs = read_registers(state)

        call = act(regs)
        result = execute(call, task.table)
        steps.append(Step(index, instr, call, result, regs))

        if call.tool == "finish":
            solved = result.ok and result.value == task.answer
            return AgentRun(task.task_id, task.family, memory, budget,
                            tuple(steps), result.value, solved, "finish")

        if memory == "ssm":
            x, delta = embed(observation_event(result))
            state = stream_step(state, x, delta)
        elif memory == "scalar":
            carry = result.value if result.ok else None
        # memory == "none": the observation is recorded in the trace and never
        # streamed, which is exactly what "wiped between steps" means here.

    return AgentRun(task.task_id, task.family, memory, budget, tuple(steps),
                    None, False, "budget")


def replay(run: AgentRun, table: ToolTable) -> tuple[ToolResult, ...]:
    """Re-execute a run's recorded actions and return the results.

    The trace is the artifact the experiment publishes, so it has to be enough
    to reproduce the run: this takes nothing but the trace and the task's table.
    A test asserts the results match the recorded ones, and that a fresh run
    produces the identical action sequence.
    """
    return tuple(execute(step.action, table) for step in run.steps)
