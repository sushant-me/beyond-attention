"""The selective family's write gate, produced by a learned projection.

This module closes -- or fails to close -- the objection the README raises
against the agent loop: the gate that decides which memory slot a value lands
in is **hand-set** in :func:`beyond_attention.agent.embed`. It is not learned,
it is not produced by a projection, and a Python dict in ``evaluate_plan``
computes every answer with no model at all. Here the same recurrence is driven
by a gate that is a function of a linear layer's output, trained by gradient
descent.

**The mechanism, exactly.** An instruction's features are a one-hot of its
opcode concatenated with a one-hot of its key over the key vocabulary. A single
``nn.Linear(features, state_width)`` produces ``gate_logits`` and a sigmoid maps
them into ``w`` in ``(0, 1)``. For a value-carrying instruction the slots are
written as

    x[R_SLOT_BASE:]     = w * value
    delta[R_SLOT_BASE:] = w * DELTA_WRITE

so ``w = 1`` is an exact write, ``w = 0`` an exact hold, and anything between is
a partial write *and* a partial erase, because ``a_t = exp(delta_t * A)`` with
``A = A_HOLD = -800``. The hand-set gate is the special case
``w = one_hot(key % state_width)`` for ``PUT`` and ``w = 0`` for ``NOISE``, and
:func:`learned_embed` is a second implementation of :func:`agent.embed` whose
slot half is taken from the gate and whose register half is written again from
the same rules -- the tests compare the two **bit for bit** when the gate is
forced to the hand-set one-hot, which is what makes "this is the same mechanism"
a check rather than a claim.

**The reader is deliberately unchanged.** Every measurement in
``experiments/learned_gate.py`` runs the agent's own ``choose_action`` on the
decoded state. Not one line of the controller is learned, so a difference
between two rows is a difference in the gate and nothing else.

**What is trained, and against what.** The supervision is the *state*, not the
answer string: the slot contents are compared with the contents a correct gate
produces -- each stored value at the slot its key addresses, and zero elsewhere
-- and the loss is the mean squared error, summed over the value-carrying events
of the task (``dense=True``) or taken at the end alone (``dense=False``). The
end-only objective is the literal form of "the final slot contents must hold the
queried key's value"; the dense one is the same condition applied after every
event, and the experiment measures both, because the end-only objective turns
out to stall in a way the dense one does not. Neither trains on the answer
string: the reader is never in the loop.

**Three things this module is not.** It is not a learned agent: the controller,
the task grammar and the task generator are all unchanged, and no gradient
touches anything but this one linear layer. It is not a generalisation claim:
the family is closed and synthetic, the values are small integers, and the
state's ``A`` is still hand-set -- only the write gate is learned. And it is
not evidence that the architecture is necessary: a Python dict still computes
every answer, and the scalar and fixed-gate controls of ``agent_loop.py`` still
stand as the reason the gating (rather than the width) is what does the work.

**One design detail that is a judgement, and is measured rather than argued.**
``NOISE`` carries a value, not a key, so :func:`gate_features` gives it no key
feature; only ``PUT`` and ``RECALL`` -- the two instructions whose operand
addresses the memory -- turn on a key dimension. A feature that is not a key
should not be presented to the gate as one. It is *not* a representational
necessity: the opcode weights could absorb the difference, at the cost of a
large shared offset per output row. Whether the optimiser finds that is an
empirical question, so the experiment trains the same architecture with the
feature added (``noise_operand=True``) and publishes both the accuracy and the
sharpness of the result.

**Two knobs that the experiment is explicit about.** ``init`` selects the
sigmoid's starting point: ``"midpoint"`` is ``nn.Linear``'s own uniform
initialisation, and ``"hold"`` biases the layer to a near-exact hold. Under
``A_HOLD`` a gate at the sigmoid's midpoint multiplies every slot by
``exp(-400) = 1.9e-174`` at every event -- the slot is erased -- which makes the
loss almost independent of anything that happened before the last event; the
experiment measures both, so the choice is a published result rather than a
hidden advantage. ``temperature`` divides the logits before the sigmoid. At
``1.0`` the gate is the bare sigmoid; below ``1.0`` it is sharpened toward 0/1,
and at ``0.05`` a logit of magnitude 1 becomes ``2.06e-9`` or ``1 - 2.06e-9``,
whose hold multiplier is ``1 - 1.6e-6`` and whose write is within ``1.4e-8`` of
the value -- so any logit beyond about +/-1 is a hard decision in float64.

**Dependencies.** ``numpy`` and ``torch``. ``torch`` is the repository's
optional ``train`` extra, so this module is deliberately *not* re-exported from
``beyond_attention/__init__.py``: importing the package must not require torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .agent import (
    AGENT_DIM,
    A_HOLD,
    DELTA_WRITE,
    OP_CODES,
    OP_NOISE,
    OP_PUT,
    OP_RECALL,
    R_ARG,
    R_ARG2,
    R_CARRY,
    R_COUNT,
    R_INDEX,
    R_KIND,
    R_MISS,
    R_OP,
    R_SLOT_BASE,
    VALUE_INSTRUCTIONS,
    AgentRun,
    Event,
    Instruction,
    Step,
    Task,
    choose_action,
    embed,
    execute,
    new_state,
    observation_event,
    read_registers,
    stream_step,
)

# The instructions whose first operand is a *key* -- that is, an address in the
# selective memory. ``PUT`` stores under it and ``RECALL`` reads from it;
# ``NOISE``'s first operand is a value, and the arithmetic instructions' operands
# name neither.
KEY_OPERAND_OPS = (OP_PUT, OP_RECALL)

# ``OP_CODES`` starts at 1 and reserves 0 for "no instruction has been streamed"
# (``NONE_CODE``), so feature ``j`` of the opcode block means "the opcode's code
# is ``j``" and dimension 0 is present but never set by a real instruction.
OPCODE_FEATURES = len(OP_CODES) + 1

# How an instruction's key becomes a one-hot. ``"key"`` is the prescribed map:
# one dimension per key in the vocabulary, so two keys share no weights and a
# key never seen in training has an untrained dimension. ``"address"`` is the
# experiment's control: one dimension per memory slot, indexed by
# ``key % state_width``, so an unseen key that addresses a trained slot arrives
# along a trained direction. The two coincide whenever ``n_keys == state_width``.
KeyMode = Literal["key", "address"]

# The two initialisations of the linear layer. "midpoint" is what ``nn.Linear``
# does; "hold" biases the gate to a near-exact hold, and the bias is chosen so
# that ``exp(A_HOLD * w)`` is within half a percent of 1.
MIDPOINT_INIT = "midpoint"
HOLD_INIT = "hold"
HOLD_BIAS = -12.0  # sigmoid(-12) = 6.1e-6, exp(-800 * 6.1e-6) = 0.995
INITS = (MIDPOINT_INIT, HOLD_INIT)


def feature_dim(n_keys: int, *, key_mode: KeyMode = "key",
                state_width: int | None = None) -> int:
    """Width of one instruction's feature vector: opcode block plus key block."""
    if n_keys < 1:
        raise ValueError(f"n_keys must be >= 1, got {n_keys}")
    if key_mode == "key":
        return OPCODE_FEATURES + n_keys
    if key_mode == "address":
        if state_width is None:
            raise ValueError("the address feature map needs a state_width")
        if state_width < 1:
            raise ValueError("state_width must be >= 1")
        return OPCODE_FEATURES + state_width
    raise ValueError(f"unknown key_mode {key_mode!r}")


def gate_features(instr: Instruction, n_keys: int, *, key_mode: KeyMode = "key",
                  state_width: int | None = None,
                  noise_operand: bool = False) -> np.ndarray:
    """One instruction's features: one-hot opcode ++ one-hot key.

    The key block is set for the instructions that address the memory (``PUT``,
    ``RECALL``), and for ``NOISE`` only when ``noise_operand`` is set -- the
    ablation, not the default, because a ``NOISE`` operand is a value and not a
    key. An operand outside the vocabulary sets no key dimension either way: a
    value that happens to lie inside the key range is not thereby a key.

    A key outside the vocabulary is an error rather than a wrap: silently
    reducing it modulo the vocabulary would invent an addressing rule the gate
    was supposed to learn. (The ``"address"`` map *does* reduce, because reducing
    is the addressing rule it encodes, and it only needs the key to be
    non-negative.)
    """
    features = np.zeros(feature_dim(n_keys, key_mode=key_mode,
                                    state_width=state_width), dtype=np.float64)
    features[OP_CODES[instr.op]] = 1.0
    addresses = instr.op in KEY_OPERAND_OPS or (
        instr.op == OP_NOISE and noise_operand)
    if addresses:
        key = instr.args[0]
        if key_mode == "key":
            if not 0 <= key < n_keys:
                if instr.op == OP_NOISE:
                    # A noise value outside the key vocabulary has no key
                    # dimension to set; the opcode one-hot has to carry it.
                    return features
                raise ValueError(
                    f"{instr.op} names key {key}, outside the key vocabulary "
                    f"0..{n_keys - 1}"
                )
            features[OPCODE_FEATURES + key] = 1.0
        else:
            assert state_width is not None  # checked by feature_dim
            if key < 0:
                raise ValueError(f"{instr.op} names a negative key {key}")
            features[OPCODE_FEATURES + key % state_width] = 1.0
    return features


def handset_gate_vector(instr: Instruction, n_keys: int,
                        state_width: int) -> np.ndarray:
    """The existing, hand-set gate as an explicit vector.

    ``PUT`` writes the one slot its key addresses and ``NOISE`` writes none, so
    this is ``one_hot(key % state_width)`` or ``zeros``. It is the target the
    learned gate is trying to reach, and :class:`HandSetGate` is the same thing
    behind the gate interface.
    """
    if state_width < 1:
        raise ValueError("state_width must be >= 1")
    vector = np.zeros(state_width, dtype=np.float64)
    if instr.op == OP_PUT:
        if not 0 <= instr.args[0] < n_keys:
            raise ValueError(f"PUT names key {instr.args[0]}, outside 0..{n_keys - 1}")
        vector[instr.args[0] % state_width] = 1.0
    return vector


class LearnedGate(nn.Module):
    """A single linear layer over the features, then a sigmoid.

    float64 throughout, because the whole path it replaces is float64 and a
    float32 gate would make "``w = 1`` is an exact write" a different claim:
    in float64 ``sigmoid`` is exactly ``1.0`` for ``z >= 36.74`` and exactly
    ``0.0`` for ``z <= -709.79`` (the first because ``1 + exp(-z)`` rounds to 1,
    the second because ``exp(-z)`` overflows), and both thresholds move inward
    sharply in float32.

    The weights are drawn from an **explicit** generator, so two runs at the
    same seed produce the same gate whether or not something else in the process
    has touched torch's global RNG.

    ``noise_operand`` is the feature ablation: it lets a ``NOISE`` value set a
    key dimension. It is off by default and is measured, not assumed.
    """

    def __init__(self, n_keys: int, state_width: int, *, seed: int = 0,
                 init: str = HOLD_INIT, key_mode: KeyMode = "key",
                 noise_operand: bool = False) -> None:
        super().__init__()
        if state_width < 1:
            raise ValueError(f"state_width must be >= 1, got {state_width}")
        if init not in INITS:
            raise ValueError(f"unknown init {init!r}; expected one of {INITS}")
        self.n_keys = int(n_keys)
        self.state_width = int(state_width)
        self.init = init
        self.key_mode = key_mode
        self.noise_operand = bool(noise_operand)
        features = feature_dim(n_keys, key_mode=key_mode,
                               state_width=state_width)

        bound = 1.0 / math.sqrt(features)
        generator = torch.Generator().manual_seed(int(seed))
        weight = (
            torch.rand(state_width, features, generator=generator,
                       dtype=torch.float64) * 2.0 - 1.0
        ) * bound
        if init == HOLD_INIT:
            bias = torch.full((state_width,), HOLD_BIAS, dtype=torch.float64)
        else:
            bias = (
                torch.rand(state_width, generator=generator,
                           dtype=torch.float64) * 2.0 - 1.0
            ) * bound
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def logits(self, features: Tensor) -> Tensor:
        """The pre-sigmoid activations, ``(..., feature_dim) -> (..., width)``."""
        return features @ self.weight.T + self.bias

    def forward(self, features: Tensor, temperature: float = 1.0) -> Tensor:
        """``(..., feature_dim) -> (..., state_width)``, every value in (0, 1)."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        return torch.sigmoid(self.logits(features) / temperature)

    def gate(self, instr: Instruction, temperature: float = 1.0) -> np.ndarray:
        """The gate vector for one instruction, as float64 numpy."""
        features = torch.from_numpy(gate_features(
            instr, self.n_keys, key_mode=self.key_mode,
            state_width=self.state_width, noise_operand=self.noise_operand))
        with torch.no_grad():
            return self.forward(features, temperature).numpy()


@dataclass(frozen=True)
class HandSetGate:
    """The hand-set gate behind the same interface as :class:`LearnedGate`.

    Forcing the gate to this is how the learned path is checked against the
    implementation it is replacing: with it installed, every number the learned
    path produces must equal the hand-set path's, not approximately.
    """

    n_keys: int
    state_width: int

    def gate(self, instr: Instruction, temperature: float = 1.0) -> np.ndarray:
        del temperature  # the hand-set gate is already hard
        return handset_gate_vector(instr, self.n_keys, self.state_width)

    def __call__(self, instr: Instruction) -> np.ndarray:
        return self.gate(instr)


Gate = Callable[[Instruction], np.ndarray]


def at_temperature(gate: LearnedGate, temperature: float) -> Gate:
    """A one-argument gate at a fixed sigmoid temperature, for ``run_learned``."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    def sharpened(instr: Instruction) -> np.ndarray:
        return gate.gate(instr, temperature)

    return sharpened


def saturation_weights(n_keys: int, state_width: int,
                       big: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Linear weights whose sigmoid is **exactly** the hand-set gate.

    A single layer can only represent the hand-set gate if such weights exist,
    and this constructs them: the bias holds every output at ``-big``, a ``PUT``
    key's own key-dimension row adds ``2 * big`` to the slot it addresses, and
    ``NOISE`` gets no key feature at all, so it stays at ``-big``. In float64
    ``sigmoid`` is exactly ``1.0`` for ``z >= 36.74`` and exactly ``0.0`` for
    ``z <= -709.79``, so ``big >= 710`` makes the result the one-hot rather than
    a near-one-hot.

    Returned as ``(weight, bias)`` with shapes ``(state_width, feature_dim)``
    and ``(state_width,)`` so they can be loaded into a :class:`LearnedGate`.
    """
    if state_width < 1:
        raise ValueError("state_width must be >= 1")
    if big < 710.0:
        raise ValueError("big must be at least 710 so sigmoid saturates in float64")
    weight = np.zeros((state_width, feature_dim(n_keys)), dtype=np.float64)
    bias = np.full(state_width, -big, dtype=np.float64)
    for key in range(n_keys):
        weight[key % state_width, OPCODE_FEATURES + key] = 2.0 * big
    return weight, bias


# --------------------------------------------------------------------------
# The embedding, and the loop.
# --------------------------------------------------------------------------

def learned_embed(event: Event, gate: Gate, n_keys: int,
                  state_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Map an event to ``(x, delta)``, with the slot gate taken from ``gate``.

    This is a second implementation of :func:`beyond_attention.agent.embed`
    rather than a wrapper around it, so that comparing the two is a test of the
    mechanism instead of a restatement of it. The register half writes exactly
    what ``embed`` writes; the slot half is ``w * value`` and ``w * DELTA_WRITE``
    for the value-carrying instructions and nothing at all for the rest, since a
    ``RECALL`` is a question and an observation is not a memory event.

    The gate is called **only** for ``PUT`` and ``NOISE``. That is the same
    restriction ``embed`` applies, and it is what keeps the claim narrow: the
    gate decides whether a value-carrying event is retained, not what the loop
    knows about the instruction it is executing.
    """
    if state_width < 0:
        raise ValueError("state_width must be >= 0")
    width = AGENT_DIM + state_width
    x = np.zeros(width, dtype=np.float64)
    delta = np.zeros(width, dtype=np.float64)

    if event.kind == "instruction":
        instr = event.instruction
        if instr is None:  # pragma: no cover - defensive
            raise ValueError("an instruction event carries no instruction")
        x[R_OP] = OP_CODES[instr.op]
        delta[R_OP] = DELTA_WRITE
        x[R_ARG] = float(instr.args[0]) if instr.args else 0.0
        delta[R_ARG] = DELTA_WRITE
        x[R_ARG2] = float(instr.args[1]) if len(instr.args) > 1 else 0.0
        delta[R_ARG2] = DELTA_WRITE
        x[R_INDEX] = 1.0
        delta[R_INDEX] = DELTA_WRITE

        if state_width > 0 and instr.op in VALUE_INSTRUCTIONS:
            value = float(instr.args[1] if instr.op == OP_PUT else instr.args[0])
            w = np.asarray(gate(instr), dtype=np.float64)
            if w.shape != (state_width,):
                raise ValueError(
                    f"the gate returned shape {w.shape}, expected ({state_width},)"
                )
            if not np.all(np.isfinite(w)):
                raise ValueError("the gate returned a non-finite weight")
            x[R_SLOT_BASE:] = w * value
            delta[R_SLOT_BASE:] = w * DELTA_WRITE
            if instr.op == OP_PUT:
                # The value lives in the memory, not in a named register.
                x[R_ARG2] = 0.0
    elif event.kind == "observation":
        result = event.result
        if result is None:  # pragma: no cover - defensive
            raise ValueError("an observation event carries no result")
        x[R_CARRY] = float(result.value) if result.value is not None else 0.0
        delta[R_CARRY] = DELTA_WRITE
        x[R_COUNT] = 1.0
        delta[R_COUNT] = DELTA_WRITE
        x[R_KIND] = float(result.kind)
        delta[R_KIND] = DELTA_WRITE
        x[R_MISS] = 1.0 if result.error == "no_key" else 0.0
        delta[R_MISS] = DELTA_WRITE
    elif event.kind == "action":
        pass  # recorded, not remembered
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown event kind {event.kind!r}")

    return x, delta


def run_learned(task: Task, gate: Gate, n_keys: int,
                state_width: int | None = None,
                budget: int | None = None,
                memory: str = "learned") -> AgentRun:
    """The agent loop with a learned gate, and the agent's own reader.

    Returns an :class:`~beyond_attention.agent.AgentRun`, so the trace, the
    actions and ``replay`` are the same objects the hand-set loop produces. The
    controller is literally :func:`beyond_attention.agent.choose_action` and the
    tools are literally :func:`beyond_attention.agent.execute`; only the
    embedding's slot half differs.

    ``state_width`` defaults to ``n_keys``, the width at which no two keys
    alias, so a learned gate is never measured through an addressing collision
    it did not choose. ``budget`` defaults to the whole plan.
    """
    if state_width is None:
        state_width = n_keys
    if state_width < 0:
        raise ValueError("state_width must be >= 0")
    if budget is None:
        budget = len(task.plan)
    if budget < 1:
        raise ValueError("budget must be >= 1")

    state = new_state(state_width)
    steps: list[Step] = []
    for index in range(budget):
        if index >= len(task.plan):
            return AgentRun(task.task_id, task.family, memory, budget,
                            tuple(steps), None, False, "plan_exhausted", None,
                            AGENT_DIM + state_width)

        instr = task.plan[index]
        state = stream_step(state, *learned_embed(
            instruction_event(instr), gate, n_keys, state_width))
        registers = read_registers(state)
        call = choose_action(registers)
        result = execute(call, task.table)
        steps.append(Step(index, instr, call, result, registers))

        if call.tool == "finish":
            solved = result.ok and result.value == task.answer
            return AgentRun(task.task_id, task.family, memory, budget,
                            tuple(steps), result.value, solved, "finish", None,
                            AGENT_DIM + state_width)

        state = stream_step(state, *learned_embed(
            observation_event(result), gate, n_keys, state_width))

    return AgentRun(task.task_id, task.family, memory, budget, tuple(steps),
                    None, False, "budget", None, AGENT_DIM + state_width)


def instruction_event(instr: Instruction) -> Event:
    """The instruction's event. Re-exported from ``agent`` for the caller's sake."""
    return Event(kind="instruction", instruction=instr)


# --------------------------------------------------------------------------
# Supervision, and the training step.
# --------------------------------------------------------------------------

def value_instructions(task: Task) -> tuple[Instruction, ...]:
    """The task's value-carrying events, in the order they arrive."""
    return tuple(i for i in task.plan if i.op in VALUE_INSTRUCTIONS)


@dataclass(frozen=True)
class GateBatch:
    """One task's supervision, assembled for the differentiable path.

    ``features`` is ``(events, feature_dim)`` for the value-carrying events only:
    the other instructions leave every slot held, so they do not move the state
    the loss is computed on. ``target`` is the state a correct gate produces
    *after the last value event* -- each stored value at the slot its key
    addresses, zero elsewhere -- and ``steps`` is that same target after every
    value event, which is what the dense objective compares against. ``query``
    is the key the task asks about, kept so that the target can be checked to
    hold the answer the plan declares.
    """

    task_id: str
    features: np.ndarray
    values: np.ndarray
    target: np.ndarray
    steps: np.ndarray
    query: int
    answer: int


def gate_batch(tasks: Sequence[Task], n_keys: int, state_width: int, *,
               key_mode: KeyMode = "key",
               noise_operand: bool = False) -> tuple[GateBatch, ...]:
    """Build one supervised example per task, rejecting unsolvable shapes.

    A target that does not hold the queried key's value at the queried key's
    slot means the correct gate cannot answer the task -- which happens when two
    stored keys alias onto one slot. That is a property of the width, not of the
    gate, so it is an error here rather than a loss the optimiser can only fail.
    """
    batches: list[GateBatch] = []
    for task in tasks:
        events = value_instructions(task)
        if not events:
            raise ValueError(f"{task.task_id} carries no value event")
        query = [i.args[0] for i in task.plan if i.op == OP_RECALL]
        if len(query) != 1:
            raise ValueError(f"{task.task_id} has {len(query)} RECALL steps")
        target = np.zeros(state_width, dtype=np.float64)
        steps: list[np.ndarray] = []
        for instr in events:
            if instr.op == OP_PUT:
                target = target.copy()
                target[instr.args[0] % state_width] = float(instr.args[1])
            steps.append(target.copy())
        if target[query[0] % state_width] != float(task.answer):
            raise ValueError(
                f"{task.task_id}: the correct state does not hold the answer "
                f"-- keys collide at this width"
            )
        batches.append(GateBatch(
            task_id=task.task_id,
            features=np.stack([
                gate_features(i, n_keys, key_mode=key_mode,
                              state_width=state_width,
                              noise_operand=noise_operand)
                for i in events
            ]),
            values=np.array(
                [float(i.args[1] if i.op == OP_PUT else i.args[0])
                 for i in events],
                dtype=np.float64,
            ),
            target=target,
            steps=np.stack(steps),
            query=query[0],
            answer=task.answer,
        ))
    return tuple(batches)


def stack_batch(batches: Sequence[GateBatch]
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(features, values, target, steps)`` as ``(B, E, F)``, ``(B, E)``, ``(B, W)``, ``(B, E, W)``.

    A batch is only stackable when every task has the same number of
    value-carrying events, which the selective family's fixed shape guarantees.
    Raising on a ragged batch is better than padding it with events that never
    happened.
    """
    events = {batch.features.shape[0] for batch in batches}
    if len(events) != 1:
        raise ValueError(
            f"tasks have different value-event counts {sorted(events)}; a batch "
            f"of these would need padding, which would invent events"
        )
    return (
        torch.tensor(np.stack([b.features for b in batches])),
        torch.tensor(np.stack([b.values for b in batches])),
        torch.tensor(np.stack([b.target for b in batches])),
        torch.tensor(np.stack([b.steps for b in batches])),
    )


def slot_trajectory(features: Tensor, values: Tensor,
                    gate: nn.Module, temperature: float = 1.0) -> Tensor:
    """The slot contents after every value event, from the loop's own recurrence.

    ``s_t = exp(w_t * A) * s_{t-1} + w_t * value_t`` with ``A = A_HOLD`` and
    ``delta_t = w_t``, which is :func:`beyond_attention.agent.stream_step`
    restricted to the slots: every other instruction writes ``delta = 0`` there,
    so only the value-carrying events appear. ``w`` is a per-slot vector, so a
    task's first key and second key can be written to two different slots by two
    different rows of one linear layer.

    Returns ``(B, E, W)``: row ``e`` is the state *after* event ``e``, which is
    the shape the dense supervision needs.
    """
    batch, events, _ = features.shape
    width = gate.state_width
    slots = torch.zeros(batch, width, dtype=torch.float64)
    trajectory = []
    for index in range(events):
        w = gate(features[:, index, :], temperature)
        slots = torch.exp(w * A_HOLD) * slots + w * values[:, index].unsqueeze(1)
        trajectory.append(slots)
    return torch.stack(trajectory, dim=1)


def state_loss(gate: nn.Module, features: Tensor, values: Tensor, target: Tensor,
               steps: Tensor | None = None, *, dense: bool = True,
               temperature: float = 1.0) -> Tensor:
    """Squared error between the slots and the state a correct gate produces.

    ``dense`` averages that error over every value event; ``dense=False`` keeps
    only the last one, which is the literal "the final slot contents hold the
    queried key's value" condition.
    """
    trajectory = slot_trajectory(features, values, gate, temperature)
    if dense:
        if steps is None:
            raise ValueError("the dense loss needs the per-event targets")
        return (trajectory - steps).pow(2).mean()
    return (trajectory[:, -1, :] - target).pow(2).mean()


@dataclass(frozen=True)
class TrainingReport:
    """What the optimiser did, kept next to what it achieved."""

    seed: int
    steps: int
    lr: float
    init: str
    key_mode: str
    noise_operand: bool
    dense: bool
    anneal_to: float | None
    initial_loss: float
    final_loss: float
    final_state_loss: float
    parameters: int


def train_gate(tasks: Sequence[Task], n_keys: int, state_width: int, *,
               seed: int = 0, steps: int = 2000, lr: float = 0.05,
               init: str = HOLD_INIT, key_mode: KeyMode = "key",
               noise_operand: bool = False,
               dense: bool = True, anneal_to: float | None = None
               ) -> tuple[LearnedGate, TrainingReport]:
    """Fit the gate to the tasks' correct states by gradient descent.

    Full-batch Adam over every supplied task, so a run is a deterministic
    function of ``seed``: there is no shuffling, no dropout, and no sampling
    whose order could depend on anything but the seed.

    ``anneal_to``, when set, divides the logits by a temperature that falls
    linearly from 1 to that value over training, so the gate hardens as it is
    fitted. ``None`` trains at temperature 1 and leaves the gate soft.
    ``noise_operand`` selects the feature ablation described on
    :func:`gate_features`.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if anneal_to is not None and not 0 < anneal_to <= 1:
        raise ValueError("anneal_to must be in (0, 1]")

    gate = LearnedGate(n_keys, state_width, seed=seed, init=init,
                       key_mode=key_mode, noise_operand=noise_operand)
    features, values, target, steps_target = stack_batch(
        gate_batch(tasks, n_keys, state_width, key_mode=key_mode,
                   noise_operand=noise_operand))
    optimiser = torch.optim.Adam(gate.parameters(), lr=lr)

    with torch.no_grad():
        initial = float(state_loss(gate, features, values, target, steps_target,
                                   dense=dense))
    for step in range(steps):
        temperature = 1.0
        if anneal_to is not None:
            progress = step / max(1, steps - 1)
            temperature = 1.0 + (anneal_to - 1.0) * progress
        optimiser.zero_grad(set_to_none=True)
        loss = state_loss(gate, features, values, target, steps_target,
                          dense=dense, temperature=temperature)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        final = float(state_loss(gate, features, values, target, steps_target,
                                 dense=dense, temperature=anneal_to or 1.0))
        final_state = float(state_loss(gate, features, values, target,
                                       steps_target, dense=False,
                                       temperature=anneal_to or 1.0))

    report = TrainingReport(
        seed=seed, steps=steps, lr=lr, init=init, key_mode=key_mode,
        noise_operand=bool(noise_operand), dense=dense, anneal_to=anneal_to,
        initial_loss=initial, final_loss=final, final_state_loss=final_state,
        parameters=sum(p.numel() for p in gate.parameters()),
    )
    return gate, report


# --------------------------------------------------------------------------
# Reading what the gate learned.
# --------------------------------------------------------------------------

def sharpness(gate: LearnedGate, tasks: Sequence[Task], n_keys: int,
              state_width: int, *, temperature: float = 1.0) -> dict:
    """How far the learned ``w`` sits from the only two values that work.

    A write is exact and a hold is exact; in between, ``w`` both partly writes
    and *erases* (``exp(w * A_HOLD)`` is already ``exp(-8)`` at ``w = 0.01``), so
    the honest reading of a soft gate is not "a gentle blend" but "a destroyed
    slot". Deviation is ``min(w, 1 - w)``: 0 for a hard 0 or 1, 0.5 for a gate
    stuck at the sigmoid's midpoint.

    ``rounded_one_hot_fraction`` is the fraction of value-carrying events whose
    *rounded* gate vector is exactly the hand-set one -- that is, what a 0.5
    threshold would recover. Deviation alone cannot tell a gate that is hard and
    right from one that is hard and inert (an all-zero gate holds every slot, so
    its deviation is 0 and its accuracy is 0), which is why the mean weight on
    the addressed slot and on the slots that must hold are reported next to it.
    """
    deviations: list[np.ndarray] = []
    written: list[float] = []
    held: list[float] = []
    rounded_matches = 0
    events = 0
    for task in tasks:
        for instr in value_instructions(task):
            w = np.asarray(gate.gate(instr, temperature), dtype=np.float64)
            expected = handset_gate_vector(instr, n_keys, state_width)
            deviations.append(np.minimum(w, 1.0 - w))
            addressed = (instr.args[0] % state_width) if instr.op == OP_PUT else None
            for slot in range(state_width):
                if slot == addressed:
                    written.append(float(w[slot]))
                else:
                    held.append(float(w[slot]))
            rounded_matches += int(np.array_equal(np.round(w), expected))
            events += 1
    if not events:  # pragma: no cover - defensive
        raise ValueError("no value-carrying events to measure")
    deviation = np.concatenate(deviations)
    return {
        "events": events,
        "weights": int(deviation.size),
        "mean_deviation": float(deviation.mean()),
        "max_deviation": float(deviation.max()),
        "min_deviation": float(deviation.min()),
        "rounded_one_hot_fraction": rounded_matches / events,
        "written_slots": len(written),
        "held_slots": len(held),
        "mean_weight_on_addressed_slots": float(np.mean(written)),
        "mean_weight_on_held_slots": float(np.mean(held)),
    }


def learned_embed_matches_hand_set(events: Sequence[Event], gate: Gate,
                                   n_keys: int, state_width: int) -> bool:
    """True when this gate's embedding is *bit-for-bit* ``agent.embed``'s.

    The comparison that makes the claim checkable: same ``x``, same ``delta``,
    for every event of a trajectory, with no tolerance anywhere.
    """
    for event in events:
        mine = learned_embed(event, gate, n_keys, state_width)
        theirs = embed(event, state_width=state_width)
        if not (np.array_equal(mine[0], theirs[0])
                and np.array_equal(mine[1], theirs[1])):
            return False
    return True
