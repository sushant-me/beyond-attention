"""An agentic loop whose working memory is a selective state-space state.

The operator's aim for this project was for it to *act* -- to run a loop with
tools rather than only consume input. This module is that loop, and it is a
small one. It contains:

* a **tool registry** with strict input schemas and pure, deterministic tools:
  ``add``/``mul``/``sub`` over bounded integers, ``lookup`` over a bounded
  key/value table supplied with the task, ``remember``/``fetch`` against an
  explicit :class:`~beyond_attention.memory.MemoryStore`, and ``finish``, which
  ends the loop and returns the answer;
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

The register file is not only a carry. ``selective`` is a second kind of task
family: a stream of events, some of them a value tagged with a key and some of
them distractors, followed by a query for the value of one key. A tagged event
is written to the memory slot its key addresses and a distractor is written
nowhere, so **whether an event is retained, and where it lands, is decided by
the event's own content** rather than by its position -- which is what the
model's learned gate exists to do, and here is set by hand. The memory is
``state_width`` slots appended to the same eight registers, and a slot is
addressed by ``key % state_width``, so the family has a capacity that can be
swept rather than asserted.

What it is **not**, stated here rather than left to be discovered:

* **It is not a general agent.** The task text is a closed instruction grammar
  (``ADD``/``MUL``/``SUB``/``KEY``/``IFPOS``/``RET``, plus ``PUT``/``NOISE``/
  ``RECALL`` for the selective family and ``REMEMBER``/``FETCH`` for the external
  store) delivered as structured events, not natural language. There is no
  tokenizer, no language understanding, and no open-ended tool use: the tool set
  is seven functions fixed at import time.
* **It does not learn.** The policy is hand-written branching, the SSM
  parameters are constants, and a seed only decides which tasks get generated.
* **The SSM is not what makes the arithmetic families work.**
  ``experiments/agent_loop.py`` runs the same controller with the carried value
  held in one Python int and solves the same tasks, and it runs a no-memory
  control that collapses on every task whose answer is an intermediate result
  while still solving the ones whose answer is written in the task text. The
  recurrence carries the value; it is not the source of the capability.
* **The selective family is where the state earns something measurable, and
  that is a narrow statement.** One scalar register cannot hold two keyed values
  at once, so the scalar-carry control solves the selective family only on the
  tasks whose queried key happens to be the last one stored, and nothing else. A
  *wide but non-selective* state -- constant decay, every event written to every
  slot -- fails with it. Both rows are published, because the second is what
  says the gating (rather than the width) is doing the work. Neither says the
  architecture is what makes the loop work: the gate is hand-set, the controller
  is hand-written, and the family is closed and synthetic.
* **The two memories are named apart on purpose.** ``RECALL`` belongs to the
  selective family and reads a *slot of the state*; ``FETCH`` reads the
  *external store*. Two instructions with one name and two meanings would be a
  bug waiting to happen, so the external one is ``FETCH`` and its tool is
  ``fetch``. The prose still calls the result "recall" -- that is what the task
  asks for -- but no symbol is shared.
* **The SSM is not what gives it long memory either.** The register file is
  eight wide and holds the last observation, so a fact from a thousand steps
  earlier is not in it and cannot be recovered from it.
  ``experiments/long_memory.py`` measures the distance at which that state fails
  and the explicit store does not, and it measures a state with **64x the bytes**
  failing in exactly the same place, because the width is not what is doing the
  work -- the key is. Nothing in that result is a property of the recurrence.
* **The tools cannot touch anything.** No network, no filesystem, no clock. A
  test parses this module's imports and asserts that ``numpy`` and its own
  numpy-only siblings are the only ones.

Only ``numpy`` is used, and every run is deterministic given its seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .memory import DEFAULT_CAPACITY, MemoryStore

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

# The selective family's memory starts here: ``state_width`` value slots are
# appended after the eight named registers, so a state's width is
# ``AGENT_DIM + state_width`` and ``state_width = 0`` is exactly the register
# file the arithmetic families use. A slot is addressed by
# ``key % state_width``, which is why a narrower state degrades smoothly into
# aliasing rather than failing outright.
R_SLOT_BASE = AGENT_DIM

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

# The decay vector for a state of a given total width. Every memory slot uses
# A_HOLD, so a slot is written exactly or not at all. Built rather than cached
# because these states are a few dozen floats wide and a cache is a second place
# for the layout to be wrong.
def decay_vector(width: int) -> np.ndarray:
    """The ``A`` diagonal for a state ``width`` registers wide."""
    if width < AGENT_DIM:
        raise ValueError(f"a state is at least {AGENT_DIM} registers wide, not {width}")
    if width == AGENT_DIM:
        return A_DIAG
    return np.concatenate(
        [A_DIAG, np.full(width - AGENT_DIM, A_HOLD, dtype=np.float64)]
    )

DELTA_HOLD = 0.0
DELTA_WRITE = 1.0


def state_bytes(dim: int = AGENT_DIM) -> int:
    """Bytes the register file occupies: one float64 per register.

    Constant in the number of steps, which is the whole claim about the working
    state -- and, at eight registers, 64 bytes, against a store whose slots are
    counted in ``memory.py``. Published so the two can be put side by side
    without either number being a guess.
    """
    return dim * 8

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
OP_PUT = "PUT"        # write arg2 into the memory slot addressed by arg
OP_NOISE = "NOISE"    # a distractor: carries a value and is written nowhere
OP_RECALL = "RECALL"  # finish with the value the slot addressed by arg holds
# The external store's two instructions. ``RECALL`` above is the selective
# family's read of a *state slot*; ``FETCH`` is this module's read of the
# ``MemoryStore``, and the two are deliberately different names for different
# memories rather than one name with two meanings.
OP_REMEMBER = "REMEMBER"  # store the carried value in the store under key arg
OP_FETCH = "FETCH"    # acc = the value the store holds under key arg

OP_CODES = {
    OP_RET: 1,
    OP_RETLIT: 2,
    OP_ADD: 3,
    OP_MUL: 4,
    OP_SUB: 5,
    OP_KEY: 6,
    OP_IFPOS: 7,
    OP_PUT: 8,
    OP_NOISE: 9,
    OP_RECALL: 10,
    OP_REMEMBER: 11,
    OP_FETCH: 12,
}
CODE_OPS = {code: op for op, code in OP_CODES.items()}
NONE_CODE = 0  # nothing has been streamed into R_OP yet

# Instructions that carry a value into the memory. The fixed-decay control is
# exactly the statement that this distinction is *not* available to it: see
# ``embed``.
VALUE_INSTRUCTIONS = (OP_PUT, OP_NOISE)


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
             OP_RET: 0, OP_RETLIT: 1,
             OP_PUT: 2, OP_NOISE: 1, OP_RECALL: 1,
             OP_REMEMBER: 1, OP_FETCH: 1}[op]
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
    "remember": ToolSchema(
        "remember", (Param("key"), Param("value")),
        "store value under key in the episodic store",
    ),
    "fetch": ToolSchema(
        "fetch", (Param("key"),),
        "the value the episodic store holds under key, or a miss",
    ),
    "finish": ToolSchema("finish", (Param("answer"),), "end the loop"),
}

# The keys and values the memory tools are exercised with live far below the
# schema's bound, which bounds nonsense rather than describing the task.
TOOL_NAMES = tuple(TOOL_SCHEMAS)

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


def execute(call: ToolCall, table: ToolTable = (),
            store: MemoryStore | None = None) -> ToolResult:
    """Run one validated call. Pure: the same call and table give the same result.

    ``store`` is the episodic store the ``remember`` and ``fetch`` tools act on.
    It is a parameter rather than a module global so that a run owns its memory:
    two runs cannot see each other's facts, and a control can be handed a store
    configured to fail. A memory call with no store is a typed error
    (``no_store``) rather than an exception, because the loop must record it and
    keep going -- which is exactly the working-state-only control.

    The selective family's memory is not here: it lives in the *state*, is
    written by ``embed`` and read by ``slot_value``, and no tool touches it.
    """
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
    elif tool == "remember":
        if store is None:
            return ToolResult(ok=False, value=None, error="no_store")
        value = call.args["value"]
        store.write(call.args["key"], value)
    elif tool == "fetch":
        if store is None:
            return ToolResult(ok=False, value=None, error="no_store")
        stored = store.retrieve(call.args["key"])
        if stored is None:
            return ToolResult(ok=False, value=None, error="no_key")
        value = stored
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

def new_state(state_width: int = 0) -> np.ndarray:
    """A zeroed state: ``AGENT_DIM`` registers plus ``state_width`` memory slots.

    ``state_width = 0`` is the register file the arithmetic families use, and is
    what every caller got before the selective family existed.
    """
    if state_width < 0:
        raise ValueError("state_width must be >= 0")
    return np.zeros(AGENT_DIM + state_width, dtype=np.float64)


def stream_step(state: np.ndarray, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """One timestep: ``h = exp(delta * A) * h + delta * x``.

    ``x`` and ``delta`` are as wide as ``state``, because the gate is per
    register: an observation writes the carry without disturbing the
    instruction, an instruction writes the instruction without disturbing the
    carry, and a selective event writes the one memory slot its key addresses
    without disturbing any other. That is the selective part, and here it is
    hand-set rather than produced by a projection.
    """
    a = np.exp(delta * decay_vector(state.shape[0]))
    return a * state + delta * x


def scan_states(x: np.ndarray, delta: np.ndarray,
                state: np.ndarray | None = None) -> np.ndarray:
    """The recurrence over a whole sequence, returning the state at each step.

    ``x`` and ``delta`` are ``(L, width)``. The returned array is
    ``(L, width)``: row ``t`` is the state *after* consuming token ``t``, which
    is what a decision at that point observes.
    """
    h = np.zeros(x.shape[1], dtype=np.float64) if state is None else state
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
    "the memory is the SSM's recurrence" a measurement, for a state of any
    width.
    """
    length, width = x.shape
    return {
        "x": x.reshape(1, length, width),
        "delta": delta.reshape(1, length, width),
        "A": decay_vector(width).reshape(width, 1),
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


def embed(event: Event, state_width: int = 0,
          fixed_gate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Map an event to the ``(x, delta)`` pair the recurrence consumes.

    An action embeds to ``(zeros, zeros)``: it is in the transcript and in no
    register. An instruction writes only the instruction registers, an
    observation only the memory registers.

    ``state_width`` appends that many value slots to the state, and the three
    selective instructions use them:

    * ``PUT`` writes its value into the slot its key addresses and leaves every
      other slot holding -- the gate is computed from the event's content, so
      *which* slot is written is a property of the input;
    * ``NOISE`` carries a value and writes no slot at all, so a distractor is
      discarded rather than stored;
    * ``RECALL`` writes nothing; it is a question, not an event.

    The value is deliberately *not* also written to a named register: the only
    place it exists after a ``PUT`` is the slot, which is what makes the memory
    load-bearing rather than decorative.

    ``fixed_gate`` is the control. It replaces that gate with a constant: **every
    event that carries a value writes that value into every slot**, with the same
    write, so which event arrived and which key it named cannot matter. That is
    what "a state with a constant, input-independent decay" means here, and it
    is the control that separates the architecture from a merely wide memory.
    The instruction registers are not memory -- they are the current input, and a
    constant gate there would stop the loop from knowing what it was asked to do
    -- so they are left alone.
    """
    if state_width < 0:
        raise ValueError("state_width must be >= 0")
    width = AGENT_DIM + state_width
    x = np.zeros(width, dtype=np.float64)
    delta = np.zeros(width, dtype=np.float64)

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

        if state_width > 0 and instr.op in VALUE_INSTRUCTIONS:
            value = float(instr.args[1] if instr.op == OP_PUT else instr.args[0])
            if fixed_gate:
                # Constant gate: no addressing, no retain/discard decision.
                x[R_SLOT_BASE:] = value
                delta[R_SLOT_BASE:] = DELTA_WRITE
            elif instr.op == OP_PUT:
                address = instr.args[0] % state_width
                x[R_SLOT_BASE + address] = value
                delta[R_SLOT_BASE + address] = DELTA_WRITE
            # OP_NOISE under the selective gate: no slot is written, so the
            # default delta of zero leaves every slot bit-for-bit unchanged.
            if instr.op == OP_PUT:
                # The value lives in the memory, not in a named register.
                x[R_ARG2] = 0.0
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


def embed_all(events: Sequence[Event], state_width: int = 0,
              fixed_gate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return (np.zeros((0, AGENT_DIM + state_width)),
                np.zeros((0, AGENT_DIM + state_width)))
    pairs = [embed(e, state_width, fixed_gate) for e in events]
    return (
        np.stack([p[0] for p in pairs]),
        np.stack([p[1] for p in pairs]),
    )


# --------------------------------------------------------------------------
# Reading the state, and the policy that acts on it.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Registers:
    """The state, decoded. Every field is a register the recurrence wrote.

    ``slots`` holds the selective family's memory, in slot order, and is empty
    for a state with no memory slots. It is a tuple rather than an array so that
    a decoded readout is hashable and cannot be mutated underneath a decision.
    """

    carry: float
    observations: int
    op_code: int
    arg: int
    arg2: int
    miss: bool
    kind: int
    instructions: int
    slots: tuple[float, ...] = ()


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
        slots=tuple(float(v) for v in state[R_SLOT_BASE:]),
    )


def slot_value(registers: Registers, key: int) -> int:
    """The value the memory holds under ``key``, or 0 if there is no memory.

    A state with no slots cannot hold a keyed value, and reporting 0 is the
    honest reading of that -- the same choice ``carried_value`` makes for a
    state that has seen no result. No task's answer is 0, so this default cannot
    accidentally be right.
    """
    if not registers.slots:
        return 0
    return int(round(registers.slots[key % len(registers.slots)]))


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


# The action at an event step. A ``PUT`` and a ``NOISE`` are already in the
# state by the time the policy is asked, so there is nothing to call; this is a
# well-formed, pure, harmless call rather than a new tool, which keeps the tool
# registry at the five functions the rest of the repository documents.
NOOP_CALL = ToolCall("add", {"a": 0, "b": 0})


def choose_action(registers: Registers) -> ToolCall:
    """The hand-designed controller. A pure function of the decoded state.

    No branching on the task, the step index, or anything else: given the state
    the action is determined. ``tests/test_agent.py`` asserts that, and asserts
    that the branch instruction really does depend on the state by flipping the
    sign of the carry and requiring the tool to change.

    Two instructions read the *memory* rather than the task text. ``RECALL``
    finishes with the value the slot addressed by its key holds -- the key comes
    from the instruction, the value only from the state, which is what makes the
    selective family a memory test. ``PUT`` and ``NOISE`` have nothing to do:
    the event has already been streamed into the state, so the agent
    acknowledges with an arithmetic no-op. The value is never in the task text
    read by the policy, and it is not in a named register either -- see
    ``embed``.
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
    if op == OP_RECALL:
        return ToolCall("finish", {"answer": slot_value(registers, arg)})
    if op in VALUE_INSTRUCTIONS:
        return NOOP_CALL
    if op == OP_REMEMBER:
        # The fact is whatever the last observation produced; the key is the
        # instruction. Writing a fact is therefore a decision the *plan* makes,
        # not one the policy makes -- see the module docstring.
        return ToolCall("remember", {"key": arg, "value": carry})
    if op == OP_FETCH:
        return ToolCall("fetch", {"key": arg})
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
    """A verifiable task: a plan, a lookup table, and an analytic answer.

    The last five fields describe a *recall* task -- one that uses the external
    store -- and default to 0/empty for the families that do not. They are on the
    task rather than recomputed from the plan by whoever reads it, because the
    distance between the fact and the query is the independent variable of the
    long-memory measurement, and a results file should not have to parse a plan
    to find it.
    """

    task_id: str
    family: str
    plan: tuple[Instruction, ...]
    table: tuple[tuple[int, int], ...]
    answer: int
    text: str
    distance: int = 0                 # loop steps between the fact and the query
    n_facts: int = 0                  # distinct facts written before the query
    query_key: int = 0                # the key the plan finally fetches
    superseded: tuple[int, ...] = ()  # values a later write replaced
    facts: tuple[tuple[int, int], ...] = ()  # (memory key, stored value), in order


def evaluate_plan(plan: Sequence[Instruction],
                  table: ToolTable) -> int:
    """The task's answer, by definition, in Python integers.

    This is the ground truth the loop is graded against, and it is deliberately
    a separate, slower piece of code from the loop: ``run_agent`` decides when to
    stop and what to call, and this decides what the answer is. They share the
    instruction *names* and nothing else.

    For the selective family the memory here is a Python dict, so the answer is
    what a correct reader of the event stream would report regardless of how the
    event stream is implemented -- the state-space memory is not consulted, and
    a bug in it cannot move the target.

    The external store has its own dict for the same reason: a bounded
    ``MemoryStore`` that cannot hold a fact fails a task whose answer was fixed
    before any array existed.
    """
    acc = 0
    memory: dict[int, int] = {}
    episodic: dict[int, int] = {}
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
        elif instr.op == OP_PUT:
            key, value = instr.args
            memory[key] = value
        elif instr.op == OP_NOISE:
            pass  # a distractor changes nothing, by definition
        elif instr.op == OP_RECALL:
            key = instr.args[0]
            if key not in memory:
                raise ValueError(f"nothing was stored under key {key}")
            return memory[key]
        elif instr.op == OP_REMEMBER:
            episodic[instr.args[0]] = acc
        elif instr.op == OP_FETCH:
            key = instr.args[0]
            if key not in episodic:
                raise ValueError(f"FETCH {key} before it was written")
            acc = episodic[key]
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
        elif instr.op == OP_PUT:
            parts.append(f"note that key {instr.args[0]} holds {instr.args[1]}")
        elif instr.op == OP_NOISE:
            parts.append(f"ignore {instr.args[0]}")
        elif instr.op == OP_RECALL:
            parts.append(f"report the value of key {instr.args[0]}")
        elif instr.op == OP_REMEMBER:
            parts.append(f"remember the result under key {instr.args[0]}")
        elif instr.op == OP_FETCH:
            parts.append(f"fetch the value under key {instr.args[0]}")
        elif instr.op == OP_RETLIT:
            parts.append(f"report {instr.args[0]}")
        elif instr.op == OP_RET:
            parts.append("report the result")
    return ", then ".join(parts)


# The families, and how many loop steps each one needs. A task is solved within
# a budget exactly when its family's step count fits inside it, which is what
# makes the budget curve a ceiling rather than a mystery.
#
# The selective family's shape is fixed here -- three keyed events, two
# distractors, then the query -- because the budget table needs one step count
# per family. The experiment sweeps the shape separately, through
# ``selective_suite``, and that sweep is where the distractor and state-width
# questions are answered.
SELECTIVE_STORES = 3
SELECTIVE_DISTRACTORS = 2
SELECTIVE_KEYS = 4
SELECTIVE_WIDTH = 4

STEP_COUNTS = {
    "literal": 1,   # RETLIT v          -- the answer is written in the task
    "one_op": 2,    # ADD a, RET
    "two_op": 3,    # MUL a, SUB b, RET
    "branch": 4,    # ADD a, SUB c, IFPOS b, RET
    "lookup": 5,    # ADD a, MUL b, SUB c, KEY, RET
    # PUT/NOISE events (in a shuffled order), then RECALL. Every event is a loop
    # step, so the budget has to cover the whole stream before the question.
    "selective": SELECTIVE_STORES + SELECTIVE_DISTRACTORS + 1,
}
FAMILIES = tuple(STEP_COUNTS)
CARRY_FAMILIES = tuple(f for f in FAMILIES if STEP_COUNTS[f] > 1)


def _build_selective(rng: np.random.Generator, task_id: str, n_store: int,
                     n_distractors: int, n_keys: int) -> Task:
    """One selective task: keyed events, distractors, and a query for one key.

    Four properties, each of which a control depends on:

    * **Store keys are distinct and drawn from ``0 .. n_keys - 1``**, which is
      the state's address space. A state at least ``n_keys`` slots wide has no
      aliasing; a narrower one aliases, which is what makes the width sweep a
      capacity curve rather than a cliff.
    * **Every value in the plan is distinct and nonzero** -- stored or
      distractor. A control that happens to hold *some* value therefore cannot
      score by coincidence, and no answer is 0.
    * **The queried key is one of the stored keys, drawn uniformly.** The task
      does not prefer the last store, the way a single register would.
    * **The events are shuffled.** Retention is a property of an event's content
      -- is it keyed, and which key -- rather than of its position in the
      stream, so a controller cannot do better by counting steps.
    """
    if not 1 <= n_store <= n_keys:
        raise ValueError(f"need 1 <= n_store <= n_keys, got {n_store}, {n_keys}")
    if n_distractors < 0:
        raise ValueError("n_distractors must be >= 0")

    keys = [int(k) for k in rng.permutation(n_keys)[:n_store]]
    pool = rng.permutation(np.arange(1, 2 * (n_store + n_distractors) + 1))
    values = [int(v) for v in pool[:n_store + n_distractors]]

    events = [instruction(OP_PUT, key, value)
              for key, value in zip(keys, values[:n_store])]
    events += [instruction(OP_NOISE, value) for value in values[n_store:]]
    order = [int(i) for i in rng.permutation(len(events))]
    query = keys[int(rng.integers(n_store))]

    plan = tuple(events[i] for i in order) + (instruction(OP_RECALL, query),)
    return Task(
        task_id=task_id,
        family="selective",
        plan=plan,
        table=(),
        answer=evaluate_plan(plan, ()),
        text=task_text(plan),
    )


def selective_suite(tasks_per_family: int = 50, seed: int = 0,
                    n_store: int = SELECTIVE_STORES,
                    n_distractors: int = SELECTIVE_DISTRACTORS,
                    n_keys: int = SELECTIVE_KEYS) -> tuple[Task, ...]:
    """The selective family alone, with its shape exposed for the sweeps.

    ``task_suite`` builds its ``selective`` family through this function with
    the default shape, so the main table and the sweeps cannot drift apart. The
    per-task seeds match ``task_suite``'s, so ``selective-7`` is the same task
    in both.
    """
    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be >= 1")
    return tuple(
        _build_selective(
            np.random.default_rng((seed * 1_000_003) + index * 97),
            f"selective-{index}", n_store, n_distractors, n_keys,
        )
        for index in range(tasks_per_family)
    )


def _build_task(family: str, rng: np.random.Generator, task_id: str) -> Task:
    """One task of a family, with its table and its analytic answer.

    Operands are drawn from 2..9 and there are at most three of them, so every
    intermediate value stays far inside float64's exact-integer range -- the
    state's readout is a float, and a task whose answer needed 60 bits would be
    testing the tolerance rather than the mechanism.
    """
    if family == "selective":
        return _build_selective(rng, task_id, SELECTIVE_STORES,
                                SELECTIVE_DISTRACTORS, SELECTIVE_KEYS)

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
        if family == "selective":
            # Built by the same function the sweeps use, so the main table and
            # the width/distractor sweeps are measuring the same tasks.
            tasks.extend(selective_suite(tasks_per_family, seed))
            continue
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


def selective_example_task() -> Task:
    """The selective task the README quotes, with a hand-checkable answer.

    Three keyed events and one distractor arrive, then the query asks for key 3.
    The answers are written out by hand: key 1 holds 40, key 3 holds 60, key 2
    holds 90, and 70 was a distractor. So the answer is **60**, and both controls
    would answer **90** -- the scalar because 90 was the last keyed value, and
    the fixed-decay state because 90 was the last value of any kind. Four slots
    hold all three values at once, which is the whole point of the family.
    """
    plan = (
        instruction(OP_PUT, 1, 40),
        instruction(OP_NOISE, 70),
        instruction(OP_PUT, 3, 60),
        instruction(OP_PUT, 2, 90),
        instruction(OP_RECALL, 3),
    )
    return Task(
        task_id="selective-example",
        family="selective",
        plan=plan,
        table=(),
        answer=evaluate_plan(plan, ()),
        text=task_text(plan),
    )


# --------------------------------------------------------------------------
# Recall over a long horizon: the family the store exists for.
#
# The five families above are solvable with the working state alone, because
# each one is decided from the instruction being streamed and the value the
# previous decision produced -- adjacent steps, never a distant one. These
# families are the opposite: a fact is produced and stored, a controllable
# number of distractor steps overwrite the carried value, and the query arrives
# afterwards. The answer is the *stored* fact, so the working state alone cannot
# produce it, and `distance` is the independent variable of the measurement.
#
# The distractors are load-bearing, and so is the shape of the failure. A
# working state with no store does not fail with an exception: the memory call
# comes back as a typed error (or a miss), a failed observation is an observation
# like any other, and a miss writes the carry to zero -- a documented and tested
# property of the register file, since 0 is a legitimate value and the miss flag
# is what distinguishes them. So the control terminates with a *valid wrong
# number*, which is what makes it a memory control rather than a broken loop.
# The distance is what the store is insensitive to: the fact sits `distance`
# steps back, and the working state's contents say nothing about it at any
# distance measured.
# --------------------------------------------------------------------------

RECALL_FAMILIES = ("recall", "recall_capacity", "recall_stale")

# Operands for a fact chain and for its distractors. The same range the five
# generated families use, so a fact's magnitude is comparable with theirs.
FACT_OPERAND_LOW = 2
FACT_OPERAND_HIGH = 9

# Memory keys are drawn from a small positive range so a slot index is readable
# in a trace. The capacity sweep uses consecutive integers instead, on purpose.
KEY_LOW = 1
KEY_HIGH = 64

def _running_value(plan: Sequence[Instruction], table: ToolTable = ()) -> int:
    """What the carry holds after ``plan``, read out of the ground truth itself.

    The table is needed because a plan that contains ``KEY`` cannot be evaluated
    without it: the ground truth is the only thing allowed to decide what a
    lookup returns, here as everywhere else.
    """
    return evaluate_plan(tuple(plan) + (instruction(OP_RET),), table)


def _try_fact_step(rng: np.random.Generator, prefix: Sequence[Instruction],
                   table: ToolTable, used_values: set[int],
                   used_keys: set[int],
                   ) -> tuple[tuple[Instruction, ...], int, int] | None:
    """Four instructions producing one fact: ``ADD a, MUL b, SUB c, KEY``.

    The fact is **the key the table pairs with the derived value**, not the
    derived value itself, and that choice is load-bearing twice over.

    It keeps the fact out of reach of the working state: the key exists only in
    the observations and in the table, so a state that holds the last observation
    holds nothing that yields it. And it keeps the carry small for *any* number
    of facts, because every fact ends as a key rather than as the product of the
    previous one -- so a thirty-fact task is a plan whose every intermediate
    value is inside the tools' argument bound rather than one that has quietly
    left it.

    Every intermediate value is passed as an *argument* to the next tool, so all
    of them -- not only the last -- have to sit inside the schema's argument
    bound. Checking that here is what keeps a generated plan executable.

    Returns ``None`` rather than raising when no draw fits, because its callers
    differ: the single-fact families treat that as a bug, and the stale family
    treats it as a reason to redraw the walk it landed on.
    """
    start = abs(_running_value(prefix, table))
    for _ in range(1000):
        a, b, c = (int(v) for v in rng.integers(
            FACT_OPERAND_LOW, FACT_OPERAND_HIGH + 1, size=3))
        if start + a > ARG_BOUND or (start + a) * b > ARG_BOUND \
                or (start + a) * b + c > ARG_BOUND:
            continue
        chain = (instruction(OP_ADD, a), instruction(OP_MUL, b),
                 instruction(OP_SUB, c), instruction(OP_KEY))
        derived = _running_value(tuple(prefix) + chain[:3], table)
        if derived == 0 or derived in used_values:
            continue
        key = int(rng.integers(KEY_LOW, KEY_HIGH + 1))
        if key in used_keys:
            continue
        return chain, key, derived
    return None


def _fact_step(rng: np.random.Generator, prefix: Sequence[Instruction],
               table: ToolTable, used_values: set[int],
               used_keys: set[int]) -> tuple[tuple[Instruction, ...], int, int]:
    """``_try_fact_step`` where a failure is a generator bug, not a redraw."""
    got = _try_fact_step(rng, prefix, table, used_values, used_keys)
    if got is None:  # pragma: no cover - defensive
        raise RuntimeError("no fact inside the bounds was found")
    return got


def _distractor_chain(rng: np.random.Generator,
                      prefix: Sequence[Instruction],
                      table: ToolTable,
                      length: int,
                      avoid: set[int]) -> tuple[Instruction, ...]:
    """``length`` instructions that move the carry somewhere it should not rest.

    Each is an ordinary ``ADD``/``SUB``/``MUL`` whose observation overwrites the
    carried value, and each is chosen so that every value the chain can reach
    stays inside ``ARG_BOUND`` -- the carry is passed as an *argument* to the next
    tool, so a chain that walked out of range would produce a plan the loop
    physically cannot execute, and the measurement would be of the tool schema
    rather than of memory.

    Staying inside the bound is a look-ahead, not a rejection: a step is only
    taken if what it leaves still has room for the worst case of the steps that
    follow (``9 * remaining``). The first version checked the bound and redrew the
    chain when it was exceeded, which for a thousand-step chain never terminated
    -- a multiplicative step always exceeds it eventually, so every draw failed.
    """
    ops = (OP_ADD, OP_SUB, OP_MUL)
    start = abs(_running_value(prefix, table))
    for _ in range(1000):
        chain: list[Instruction] = []
        bound = start  # an upper bound on |carry| after every step so far
        for _ in range(length):
            remaining = length - len(chain) - 1
            room = ARG_BOUND - FACT_OPERAND_HIGH * remaining
            op = ops[int(rng.integers(len(ops)))]
            arg = int(rng.integers(FACT_OPERAND_LOW, FACT_OPERAND_HIGH + 1))
            grown = bound * arg if op == OP_MUL else bound + arg
            if grown > room:
                # Fall back to the cheapest additive step. If even that does not
                # fit, the chain cannot be completed at this starting value.
                op, arg = OP_ADD, FACT_OPERAND_LOW
                grown = bound + FACT_OPERAND_LOW
                if grown > room:
                    break
            bound = grown
            chain.append(instruction(op, arg))
        if len(chain) != length:
            continue
        final = _running_value(tuple(prefix) + tuple(chain), table)
        if final not in avoid:
            return tuple(chain)
    raise RuntimeError("no distractor chain inside the bounds was found")  # pragma: no cover


def _decoys(rng: np.random.Generator, used_values: set[int],
            used_keys: set[int], count: int = 3) -> list[tuple[int, int]]:
    """Table entries that are not facts, so a lookup has something to miss.

    A table holding only the fact's own value would make ``KEY`` a formality:
    there would be no other entry to return, and a retrieval that ignored the
    table would look identical.
    """
    decoys: list[tuple[int, int]] = []
    attempts = 0
    while len(decoys) < count and attempts < 1000:
        attempts += 1
        key = int(rng.integers(KEY_LOW, KEY_HIGH + 1))
        value = int(rng.integers(1, 100))
        if key in used_keys or value in used_values:
            continue
        used_keys.add(key)
        used_values.add(value)
        decoys.append((key, value))
    return decoys


def recall_task(rng: np.random.Generator, task_id: str, distance: int,
                key: int | None = None) -> Task:
    """One fact, ``distance`` distractors, one query.

    The shortest version of the long-memory question, and the one the solve-rate
    curve is drawn against: everything but the distance is fixed.
    """
    if distance < 1:
        raise ValueError("a recall task needs at least one distractor step")
    entries: list[tuple[int, int]] = []
    chain, fact, derived = _fact_step(rng, (), entries, set(), set())
    entries.append((fact, derived))
    if key is None:
        key = int(rng.integers(KEY_LOW, KEY_HIGH + 1))
    prefix = chain + (instruction(OP_REMEMBER, key),)
    distractors = _distractor_chain(rng, prefix, entries, distance,
                                    avoid={fact})
    plan = prefix + distractors + (instruction(OP_FETCH, key),
                                   instruction(OP_RET))
    table = tuple(sorted(entries + _decoys(rng, {derived}, {fact})))
    return Task(
        task_id=task_id, family="recall", plan=plan, table=table,
        answer=evaluate_plan(plan, table), text=task_text(plan),
        distance=distance, n_facts=1, query_key=key, facts=((key, fact),),
    )


def recall_suite(distances: Sequence[int], tasks_per_distance: int = 8,
                 seed: int = 0) -> tuple[Task, ...]:
    """One task per (distance, draw). The per-task seed does not depend on order."""
    if tasks_per_distance < 1:
        raise ValueError("tasks_per_distance must be >= 1")
    tasks: list[Task] = []
    for distance in distances:
        for made in range(tasks_per_distance):
            rng = np.random.default_rng(
                (seed * 1_000_003) + distance * 9_973 + made * 97)
            tasks.append(recall_task(rng, f"recall-d{distance}-{made}",
                                     distance))
    return tuple(tasks)


def capacity_task(rng: np.random.Generator, task_id: str, n_facts: int,
                  distance: int, query_index: int) -> Task:
    """``n_facts`` facts, then a query for one of them.

    Memory keys are ``1..n_facts`` and the store is direct-mapped at
    ``key % capacity``, so with more facts than slots key ``i`` and key
    ``i + capacity`` share a slot and the later write evicts the earlier one. The
    collision is **designed** rather than drawn from a random key space: this
    measures what a bounded store does at a known load, not the birthday-paradox
    rate a wider key space would give, and the README says which of the two it is.

    ``facts`` holds the writes in order, so a reader -- and the precision
    accounting -- can tell the queried fact's value from another fact's value
    without re-deriving either.
    """
    if n_facts < 1:
        raise ValueError("a capacity task needs at least one fact")
    if distance < 1:
        raise ValueError("a capacity task needs at least one distractor step")

    prefix: tuple[Instruction, ...] = ()
    used_values: set[int] = set()
    used_keys: set[int] = set()
    facts: list[tuple[int, int]] = []
    entries: list[tuple[int, int]] = []
    for index in range(n_facts):
        chain, fact, derived = _fact_step(rng, prefix, entries, used_values,
                                          used_keys)
        entries.append((fact, derived))
        memory_key = index + 1
        prefix = prefix + chain + (instruction(OP_REMEMBER, memory_key),)
        used_values.add(derived)
        used_keys.add(fact)
        facts.append((memory_key, fact))

    query_key, answer = facts[query_index]
    stored = {value for _, value in facts}
    distractors = _distractor_chain(rng, prefix, entries, distance,
                                    avoid=stored)
    plan = prefix + distractors + (instruction(OP_FETCH, query_key),
                                   instruction(OP_RET))
    table = tuple(sorted(entries
                         + _decoys(rng, set(used_values), set(used_keys))))
    return Task(
        task_id=task_id, family="recall_capacity", plan=plan, table=table,
        answer=evaluate_plan(plan, table), text=task_text(plan),
        distance=distance, n_facts=n_facts, query_key=query_key,
        facts=tuple(facts),
    )


def capacity_suite(n_facts: Sequence[int], distance: int = 4,
                   tasks_per_size: int = 4, seed: int = 0,
                   query_indices: Sequence[int] = (0, -1)) -> tuple[Task, ...]:
    """Facts against slots, querying the oldest and the newest fact written.

    The oldest is the one a bounded store evicts first and the newest is the one
    it cannot have lost, so the two together separate "this store forgot" from
    "this store was never asked anything hard".
    """
    tasks: list[Task] = []
    for size in n_facts:
        for query_index in query_indices:
            for made in range(tasks_per_size):
                rng = np.random.default_rng(
                    (seed * 1_000_003) + size * 9_973 + (query_index + 64) * 379
                    + made * 97)
                tasks.append(capacity_task(
                    rng, f"capacity-f{size}-q{query_index}-{made}", size,
                    distance, query_index))
    return tuple(tasks)


def stale_task(rng: np.random.Generator, task_id: str, distance: int,
               key: int | None = None) -> Task:
    """A fact written, overwritten, and queried.

    The write under ``key`` happens twice with distractors in between, so the
    second value is the answer and the first is *stale*. A store that returns the
    first is not merely wrong about this task: it hands the loop a value that was
    true earlier and is not true now, which is the failure mode the control
    exists to catch.
    """
    if distance < 1:
        raise ValueError("a stale task needs at least one distractor step")
    if key is None:
        key = int(rng.integers(KEY_LOW, KEY_HIGH + 1))

    used_values: set[int] = set()
    used_keys: set[int] = set()
    entries: list[tuple[int, int]] = []
    first_chain, first, first_derived = _fact_step(rng, (), entries,
                                                   used_values, used_keys)
    entries.append((first, first_derived))
    used_values.add(first_derived)
    used_keys.add(first)
    prefix = first_chain + (instruction(OP_REMEMBER, key),)

    # The second fact's arithmetic starts from wherever the walk above ended, so
    # a walk that landed on a large value leaves no room to multiply inside the
    # tools' argument bound. That is a property of *that draw*, not of the task,
    # so the walk is redrawn rather than the bound loosened: the number of steps
    # between the two writes is the distance under test and does not change.
    for _ in range(200):
        middle = _distractor_chain(rng, prefix, entries, distance,
                                   avoid={first})
        candidate = prefix + middle
        got = _try_fact_step(rng, candidate, entries, used_values, used_keys)
        if got is None:
            continue
        second_chain, second, second_derived = got
        entries.append((second, second_derived))
        used_values.add(second_derived)
        used_keys.add(second)
        plan_prefix = candidate + second_chain + (instruction(OP_REMEMBER, key),)
        tail = _distractor_chain(rng, plan_prefix, entries, distance,
                                 avoid={first, second})
        plan = plan_prefix + tail + (instruction(OP_FETCH, key),
                                     instruction(OP_RET))
        table = tuple(sorted(entries + _decoys(rng, used_values, used_keys)))
        return Task(
            task_id=task_id, family="recall_stale", plan=plan, table=table,
            answer=evaluate_plan(plan, table), text=task_text(plan),
            distance=distance, n_facts=1, query_key=key, superseded=(first,),
            facts=((key, first), (key, second)),
        )
    raise RuntimeError("no stale task inside the bounds was found")  # pragma: no cover


def stale_suite(distances: Sequence[int], tasks_per_distance: int = 4,
                seed: int = 0) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for distance in distances:
        for made in range(tasks_per_distance):
            rng = np.random.default_rng(
                (seed * 1_000_003) + distance * 9_973 + made * 97)
            tasks.append(stale_task(rng, f"stale-d{distance}-{made}", distance))
    return tuple(tasks)


def example_recall_task() -> Task:
    """The recall trace in the README, with a hand-checkable answer.

    ``0 + 3 = 3``, ``3 * 4 = 12``, ``12 - 5 = 7``; the table pairs **7 with key
    9**, so the fact is 9, and it is remembered under memory key 4. Then
    ``9 + 5 - 2 = 12`` overwrites the carry, and the query returns 9.

    Without a store the same plan ends differently: the ``remember`` call comes
    back as a typed ``no_store`` error and the ``recall`` as a miss, and a
    failed observation writes the carry to zero. The loop therefore terminates
    with **0** -- a valid wrong answer rather than an exception, which is what
    makes the contrast a measurement of memory instead of a broken loop.
    """
    plan = (
        instruction(OP_ADD, 3),
        instruction(OP_MUL, 4),
        instruction(OP_SUB, 5),
        instruction(OP_KEY),
        instruction(OP_REMEMBER, 4),
        instruction(OP_ADD, 5),
        instruction(OP_SUB, 2),
        instruction(OP_FETCH, 4),
        instruction(OP_RET),
    )
    table = ((2, 11), (5, 99), (9, 7))
    return Task(
        task_id="example-recall", family="recall", plan=plan, table=table,
        answer=evaluate_plan(plan, table), text=task_text(plan),
        distance=2, n_facts=1, query_key=4, facts=((4, 9),),
    )


def example_stale_task() -> Task:
    """The stale-fact trace, with both values known by hand.

    Memory key 4 holds **9** after the first write. Then ``9 + 5 + 2 = 16``,
    ``16 * 3 = 48``, ``48 - 6 = 42``, and the table pairs 42 with key 7, so the
    second write puts **7** under key 4. After one more distractor the query
    returns 7 -- the value that is true now, not the 9 that was true four steps
    earlier and is the answer a first-write-wins store gives.
    """
    plan = (
        instruction(OP_ADD, 3),
        instruction(OP_MUL, 4),
        instruction(OP_SUB, 5),
        instruction(OP_KEY),
        instruction(OP_REMEMBER, 4),
        instruction(OP_ADD, 5),
        instruction(OP_ADD, 2),
        instruction(OP_MUL, 3),
        instruction(OP_SUB, 6),
        instruction(OP_KEY),
        instruction(OP_REMEMBER, 4),
        instruction(OP_SUB, 2),
        instruction(OP_FETCH, 4),
        instruction(OP_RET),
    )
    table = ((2, 11), (7, 42), (9, 7))
    return Task(
        task_id="example-stale", family="recall_stale", plan=plan, table=table,
        answer=evaluate_plan(plan, table), text=task_text(plan),
        distance=1, n_facts=1, query_key=4, superseded=(9,),
        facts=((4, 9), (4, 7)),
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
    """A finished (or exhausted) run, with everything needed to replay it.

    ``store`` is the episodic store the run wrote to, or ``None`` if it had none.
    A live reference rather than a description, because the store's byte count
    and its hit/miss accounting are part of what a run is: a results file that
    recorded only the actions could not say what the memory cost. ``state_dim``
    is the *total* width of the state the run carried, so the byte ledger can be
    read off a finished run rather than assumed.
    """

    task_id: str
    family: str
    memory: str
    budget: int
    steps: tuple[Step, ...]
    answer: int | None
    solved: bool
    stop_reason: str  # "finish" | "budget" | "plan_exhausted"
    store: MemoryStore | None = None
    state_dim: int = AGENT_DIM

    @property
    def actions(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return tuple(step.action.signature() for step in self.steps)


Policy = Callable[[Registers], ToolCall]


def required_state_width(task: Task) -> int:
    """The narrowest state that can hold this task's key space without aliasing.

    Every key a plan stores or asks for is an address, and a slot is
    ``key % state_width``; a state one slot wider than the largest key aliases
    nothing. Arithmetic plans name no keys, so this is 0 for them and the
    register file is exactly the eight named registers. It is the default width
    ``run_agent`` uses, so a selective task is never silently run with no memory
    -- the width sweep passes an explicit width to override it and measure the
    aliasing.
    """
    keys = [instr.args[0] for instr in task.plan
            if instr.op in (OP_PUT, OP_RECALL)]
    return max(keys) + 1 if keys else 0


def run_agent(
    task: Task,
    budget: int = 6,
    memory: str = "ssm",
    policy: str | Policy = "plan",
    seed: int = 0,
    state_width: int | None = None,
    store: MemoryStore | None = None,
) -> AgentRun:
    """Run the loop until ``finish`` or the budget runs out.

    ``memory`` selects what carries the values between decisions:

    * ``"ssm"`` -- the selective-scan state implemented above, which is the
      subject of the measurement;
    * ``"scalar"`` -- one Python int standing in for the whole value memory. For
      the arithmetic families it holds the last observation; for the selective
      family it holds the last keyed event's value. It is deliberately the
      *charitable* scalar: it is allowed the retain/discard decision for free --
      a distractor does not move it -- and fails only because one register
      cannot hold two keys' values at once. This is the control for "is the
      recurrence doing anything the carry is not".
    * ``"fixed"`` -- the same width of memory as ``"ssm"`` with a constant,
      input-independent write gate: every value-carrying event is written into
      every slot. This is the control for "is it the gating or the width".
    * ``"none"`` -- wiped before every decision, with only the current
      instruction re-streamed. This is the control for "does carrying the value
      matter at all": the instruction registers refill from the task text, so
      the agent still knows which tool to reach for, and only the carried result
      is gone.

    ``state_width`` appends that many memory slots to the state. ``None``, the
    default, sizes the memory to the task's own key space through
    ``required_state_width`` -- so a selective task gets a state wide enough to
    hold its keys and an arithmetic task gets the eight named registers -- and
    an explicit width is how the experiment measures aliasing.

    ``store`` is the episodic memory the ``remember``/``fetch`` tools act on.
    ``None`` is the working-state-only control: the instructions still stream and
    the policy still reaches for the right tool, and the call comes back as a
    typed ``no_store`` error. It is independent of ``state_width``: the store is
    outside the state, which is why a task that uses it runs with the same eight
    named registers the arithmetic families use.

    ``policy`` is ``"plan"``, ``"random"``, ``"never"``, or any callable taking
    the decoded state and returning a (possibly malformed) call.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if memory not in ("ssm", "scalar", "none", "fixed"):
        raise ValueError(f"unknown memory {memory!r}")
    if state_width is None:
        state_width = required_state_width(task)
    if state_width < 0:
        raise ValueError("state_width must be >= 0")

    # A scalar *is* the memory, so it has no slots. Every other memory keeps the
    # state's width, because "wiped" and "a fixed gate" are claims about the
    # state that only mean anything if the state exists.
    width = 0 if memory == "scalar" else state_width

    rng = np.random.default_rng(seed)
    state = new_state(width)
    scalar: int | None = None  # used by memory="scalar" only

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
            return AgentRun(task.task_id, task.family, memory, budget,
                            tuple(steps), None, False, "plan_exhausted", store,
                            AGENT_DIM + width)

        instr = task.plan[index]
        keyed = instr.op in VALUE_INSTRUCTIONS

        if memory == "none":
            state = new_state(width)  # forget everything, then re-read the task
        x, delta = embed(instruction_event(instr), state_width=width,
                         fixed_gate=(memory == "fixed"))
        state = stream_step(state, x, delta)

        if memory == "scalar":
            # The scalar is written exactly where the state's memory would be:
            # by a keyed event for the selective family, and by an observation
            # for the arithmetic ones. The no-op action at an event step returns
            # 0 and must not overwrite what the event just stored.
            if instr.op == OP_PUT:
                scalar = instr.args[1]
            regs = read_registers(state)
            regs = Registers(
                carry=float(scalar) if scalar is not None else 0.0,
                observations=0 if scalar is None else 1,
                op_code=regs.op_code, arg=regs.arg, arg2=regs.arg2,
                miss=regs.miss, kind=regs.kind, instructions=regs.instructions,
                slots=() if scalar is None else (float(scalar),),
            )
        else:
            regs = read_registers(state)

        call = act(regs)
        result = execute(call, task.table, store)
        steps.append(Step(index, instr, call, result, regs))

        if call.tool == "finish":
            solved = result.ok and result.value == task.answer
            return AgentRun(task.task_id, task.family, memory, budget,
                            tuple(steps), result.value, solved, "finish", store,
                            AGENT_DIM + width)

        if memory in ("ssm", "fixed"):
            x, delta = embed(observation_event(result), state_width=width,
                             fixed_gate=(memory == "fixed"))
            state = stream_step(state, x, delta)
        elif memory == "scalar" and not keyed and instr.op != OP_RECALL:
            scalar = result.value if result.ok else None
        # memory == "none": the observation is recorded in the trace and never
        # streamed, which is exactly what "wiped between steps" means here.

    return AgentRun(task.task_id, task.family, memory, budget, tuple(steps),
                    None, False, "budget", store, AGENT_DIM + width)


def replay(run: AgentRun, table: ToolTable,
           store: MemoryStore | None = None) -> tuple[ToolResult, ...]:
    """Re-execute a run's recorded actions and return the results.

    The trace is the artifact the experiment publishes, so it has to be enough
    to reproduce the run: this takes nothing but the trace and the task's table.
    A test asserts the results match the recorded ones, and that a fresh run
    produces the identical action sequence.

    A run that used a store is replayed against a **fresh** store built from that
    store's ``spec()``, not against the live one: the recorded actions include
    the writes, in order, so replaying them from empty rebuilds the same state.
    Replaying against the store the run already filled would find every fact
    already there and prove nothing.
    """
    if store is None and run.store is not None:
        store = MemoryStore.from_spec(run.store.spec())
    return tuple(execute(step.action, table, store) for step in run.steps)
