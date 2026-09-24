"""The learned write gate, checked against the hand-set one it replaces.

A learned gate that "works" is not evidence of much: the family is closed, the
controller is hand-written, and a Python dict computes every answer. So the tests
here are the same two kinds the rest of the repository uses.

**Equality with the implementation it replaces.** The claim is that
``learned_gate.learned_embed`` is the *same mechanism* as ``agent.embed`` with the
slot gate swapped. That is not asserted in prose: with the gate forced to the
hand-set one-hot, the two embeddings must agree **bit for bit** on every event of
a trajectory, and the whole loop -- solved flags, answers, actions, decoded slots
-- must agree task by task. A tolerance here would hide exactly the drift the
test exists to find.

**Controls that can fail.** An untrained gate must *not* match the hand-set one,
or training is not what is doing the work. A constructed pair of weights must
reproduce the hand-set gate exactly, or the architecture cannot represent the
answer at all and every negative result would be about the representation rather
than about learning. Training twice at one seed must produce the same gate, or
"it learned" is a statement about the RNG. The softness of the raw sigmoid gate
must be visible as a destroyed hold, not merely as a lower accuracy. And the
experiment's hand-set row is required to equal what ``agent.py`` reports and what
``agent-loop.json`` published, so the measurement cannot drift from the thing it
measures.

Two facts about scale are worth stating here rather than leaving to be
discovered: the tests train on 16 tasks for 300 steps, which is enough for the
gate to be an exact one-hot but is *not* the published run (64 tasks, 2,000
steps, five seeds); and only the write gate is learned -- ``A`` is still
``A_HOLD`` and the reader is still ``agent.choose_action``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

from beyond_attention.agent import (
    A_HOLD,
    OP_CODES,
    OP_NOISE,
    OP_PUT,
    OP_RECALL,
    SELECTIVE_DISTRACTORS,
    SELECTIVE_KEYS,
    SELECTIVE_STORES,
    STEP_COUNTS,
    Task,
    action_event,
    embed,
    instruction,
    instruction_event,
    observation_event,
    replay,
    required_state_width,
    run_agent,
    selective_suite,
)
from beyond_attention.learned_gate import (
    HOLD_INIT,
    MIDPOINT_INIT,
    OPCODE_FEATURES,
    HandSetGate,
    LearnedGate,
    at_temperature,
    gate_batch,
    gate_features,
    handset_gate_vector,
    learned_embed,
    learned_embed_matches_hand_set,
    run_learned,
    saturation_weights,
    sharpness,
    stack_batch,
    state_loss,
    train_gate,
    value_instructions,
)

REPO = pathlib.Path(__file__).resolve().parent.parent

# A small, fast training configuration. The published run is larger; the tests
# only need enough for the gate to reach an exact one-hot deterministically.
TEST_TRAIN_TASKS = 16
TEST_EVAL_TASKS = 30
TEST_STEPS = 300
KEYS = SELECTIVE_KEYS
WIDTH = SELECTIVE_KEYS


def load_experiment():
    """Import `experiments/learned_gate.py` without installing it as a package."""
    path = REPO / "experiments" / "learned_gate.py"
    spec = importlib.util.spec_from_file_location("exp_learned_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp_learned_gate"] = module
    spec.loader.exec_module(module)
    return module


def small_suites() -> tuple[tuple[Task, ...], tuple[Task, ...]]:
    return (selective_suite(TEST_TRAIN_TASKS, seed=0),
            selective_suite(TEST_EVAL_TASKS, seed=1))


def trained_test_gate(seed: int = 3):
    train, _ = small_suites()
    return train_gate(train, KEYS, WIDTH, seed=seed, steps=TEST_STEPS, lr=0.05)


def trajectory_events(task: Task) -> list:
    """Every event of one hand-set run: instructions, actions and observations."""
    run = run_agent(task, budget=len(task.plan))
    events: list = []
    for step in run.steps:
        events.append(instruction_event(step.instruction))
        events.append(action_event(step.action))
        events.append(observation_event(step.result))
    return events


# --------------------------------------------------------------------------
# The mechanism: the learned path is agent.embed with the gate swapped
# --------------------------------------------------------------------------

def test_the_forced_hand_set_gate_reproduces_agent_embed_bit_for_bit() -> None:
    """No tolerance anywhere: same x, same delta, for every event.

    The gate is forced to ``one_hot(key % W)`` for ``PUT`` and ``zeros`` for
    ``NOISE``, which is what ``agent.embed`` computes by hand. If the two
    implementations disagree by one ulp this fails, which is the point: a rate
    can coincide between different mechanisms, and arrays cannot.
    """
    handset = HandSetGate(KEYS, WIDTH)
    tasks = selective_suite(5, seed=0) + selective_suite(5, seed=7)

    for task in tasks:
        for event in trajectory_events(task):
            mine = learned_embed(event, handset.gate, KEYS, WIDTH)
            theirs = embed(event, state_width=WIDTH)
            assert np.array_equal(mine[0], theirs[0]), event
            assert np.array_equal(mine[1], theirs[1]), event

    assert learned_embed_matches_hand_set(
        [event for task in tasks for event in trajectory_events(task)],
        handset.gate, KEYS, WIDTH)


def test_the_forced_gate_matches_at_zero_width_and_with_aliasing() -> None:
    """The two edge widths: no memory at all, and two keys per slot."""
    task = selective_suite(3, seed=2)[1]
    events = trajectory_events(task)

    for width in (0, 2, 7):
        handset = HandSetGate(KEYS, width) if width else HandSetGate(KEYS, 1)
        for event in events:
            mine = learned_embed(event, handset.gate, KEYS, width)
            theirs = embed(event, state_width=width)
            assert np.array_equal(mine[0], theirs[0]), (event, width)
            assert np.array_equal(mine[1], theirs[1]), (event, width)


def test_the_learned_path_reproduces_the_hand_set_loop_task_by_task() -> None:
    """Same controller, same tools, same answers -- the reader is not learned.

    At the width ``run_agent`` itself picks, even the decoded slots line up; at a
    wider state the slot tuple is longer (the extra slots are simply zero) and
    the answers and actions still agree.
    """
    for task in selective_suite(30, seed=1):
        budget = len(task.plan)
        theirs = run_agent(task, budget=budget)
        native = required_state_width(task)

        mine = run_learned(task, HandSetGate(KEYS, native).gate, KEYS,
                           state_width=native, budget=budget)
        assert mine.solved == theirs.solved == True  # noqa: E712 - explicit
        assert mine.answer == theirs.answer
        assert mine.actions == theirs.actions
        assert mine.stop_reason == theirs.stop_reason == "finish"
        assert [s.registers.slots for s in mine.steps] == \
            [s.registers.slots for s in theirs.steps]
        assert replay(mine, task.table) == tuple(s.result for s in mine.steps)

        wider = run_learned(task, HandSetGate(KEYS, WIDTH).gate, KEYS,
                           state_width=WIDTH, budget=budget)
        assert (wider.solved, wider.answer, wider.actions) == \
            (theirs.solved, theirs.answer, theirs.actions)
        assert all(len(s.registers.slots) == WIDTH for s in wider.steps)


def test_the_budget_and_the_stop_reasons_are_the_agents_own() -> None:
    task = selective_suite(1, seed=0)[0]
    handset = HandSetGate(KEYS, WIDTH)
    full = len(task.plan)

    cut = run_learned(task, handset.gate, KEYS, state_width=WIDTH, budget=3)
    assert cut.stop_reason == "budget" and cut.answer is None and not cut.solved

    finished = run_learned(task, handset.gate, KEYS, state_width=WIDTH,
                           budget=full)
    assert finished.stop_reason == "finish" and finished.solved

    # `finish` ends the loop, so a larger budget changes nothing on a selective
    # task. `plan_exhausted` needs a plan whose last event is not a question.
    over = run_learned(task, handset.gate, KEYS, state_width=WIDTH,
                       budget=full + 5)
    assert over.stop_reason == "finish" and over.solved

    endless = Task("endless", "selective",
                   (instruction(OP_PUT, 1, 5), instruction(OP_NOISE, 3)),
                   (), 0, "by hand")
    exhausted = run_learned(endless, handset.gate, KEYS, state_width=WIDTH,
                            budget=5)
    assert exhausted.stop_reason == "plan_exhausted"
    assert exhausted.answer is None and not exhausted.solved


def test_a_gate_of_the_wrong_width_is_rejected() -> None:
    task = selective_suite(1, seed=0)[0]
    with pytest.raises(ValueError, match="gate returned shape"):
        run_learned(task, lambda instr: np.zeros(WIDTH + 1), KEYS,
                    state_width=WIDTH)
    with pytest.raises(ValueError, match="state_width"):
        run_learned(task, HandSetGate(KEYS, WIDTH).gate, KEYS, state_width=-1)
    with pytest.raises(ValueError, match="budget"):
        run_learned(task, HandSetGate(KEYS, WIDTH).gate, KEYS, budget=0)


# --------------------------------------------------------------------------
# The controls: training is the thing that works, and the layer can represent it
# --------------------------------------------------------------------------

def test_the_untrained_gate_does_not_match_the_hand_set_gate() -> None:
    """Random init, same architecture, both initialisations, both temperatures.

    A control that happened to match would make every learned row meaningless,
    so this asserts the failure rather than the absence of an exception.
    """
    handset = HandSetGate(KEYS, WIDTH)
    tasks = selective_suite(TEST_EVAL_TASKS, seed=0)
    hand_set_rate = np.mean([run_learned(t, handset.gate, KEYS, state_width=WIDTH)
                             .solved for t in tasks])
    assert hand_set_rate == 1.0

    for init in (MIDPOINT_INIT, HOLD_INIT):
        gate = LearnedGate(KEYS, WIDTH, seed=0, init=init)
        for temperature in (1.0, 0.05):
            rate = np.mean([
                run_learned(t, at_temperature(gate, temperature), KEYS,
                            state_width=WIDTH).solved
                for t in tasks
            ])
            assert rate < hand_set_rate, (init, temperature, rate)


def test_the_untrained_hold_gate_is_inert_and_not_merely_soft() -> None:
    """Deviation from 0/1 cannot tell "hard and right" from "hard and inert".

    The hold-biased initialisation is an *all-hold* gate: its deviation is 0
    because every weight is 0, and it solves nothing. The metric that sees the
    difference is the rounded match, which is the ``NOISE`` fraction and nothing
    more.
    """
    _, tasks = small_suites()
    gate = LearnedGate(KEYS, WIDTH, seed=0, init=HOLD_INIT)
    measured = sharpness(gate, tasks, KEYS, WIDTH, temperature=1.0)

    assert measured["mean_weight_on_addressed_slots"] < 1e-4
    assert measured["mean_weight_on_held_slots"] < 1e-4
    assert measured["mean_deviation"] < 1e-4
    # Two of the five value events are NOISE, and an all-zero gate is right
    # about exactly those.
    assert measured["rounded_one_hot_fraction"] == pytest.approx(2 / 5)


def test_the_architecture_can_represent_the_hand_set_gate_exactly() -> None:
    """Constructed weights whose sigmoid *is* the one-hot, aliasing included.

    Without this, a learned gate that failed would leave it open whether the
    failure is the optimiser's or the representation's.
    """
    weight, bias = saturation_weights(8, 4)
    gate = LearnedGate(8, 4, seed=0)
    with torch.no_grad():
        gate.weight.copy_(torch.tensor(weight))
        gate.bias.copy_(torch.tensor(bias))

    for key in range(8):
        put = instruction(OP_PUT, key, 5)
        assert np.array_equal(gate.gate(put, 1.0),
                              handset_gate_vector(put, 8, 4)), key
    for value in range(1, 6):
        noise = instruction(OP_NOISE, value)
        assert np.array_equal(gate.gate(noise, 1.0), np.zeros(4))

    tasks = selective_suite(20, seed=0)
    rate = np.mean([run_learned(t, gate.gate, KEYS, state_width=4).solved
                    for t in tasks])
    assert rate == 1.0


def test_training_does_the_work_an_untrained_gate_does_not() -> None:
    """The learned row against the control, on the same tasks."""
    _, tasks = small_suites()
    gate, report = trained_test_gate()
    untrained = LearnedGate(KEYS, WIDTH, seed=3, init=HOLD_INIT)

    trained_rate = np.mean([
        run_learned(t, at_temperature(gate, 0.05), KEYS, state_width=WIDTH).solved
        for t in tasks
    ])
    untrained_rate = np.mean([
        run_learned(t, at_temperature(untrained, 0.05), KEYS, state_width=WIDTH)
        .solved for t in tasks
    ])

    assert report.final_loss < report.initial_loss
    assert trained_rate == 1.0
    assert untrained_rate == 0.0
    measured = sharpness(gate, tasks, KEYS, WIDTH, temperature=1.0)
    assert measured["rounded_one_hot_fraction"] == 1.0
    assert measured["mean_weight_on_addressed_slots"] > 0.9
    assert sharpness(untrained, tasks, KEYS, WIDTH,
                     temperature=1.0)["rounded_one_hot_fraction"] < 0.5


def test_training_is_deterministic_given_a_fixed_seed() -> None:
    """Two runs at one seed are the same gate; two seeds are not."""
    first, first_report = trained_test_gate(seed=3)
    second, second_report = trained_test_gate(seed=3)
    other, _ = trained_test_gate(seed=4)

    assert first_report == second_report
    assert np.array_equal(first.weight.detach().numpy(),
                          second.weight.detach().numpy())
    assert np.array_equal(first.bias.detach().numpy(),
                          second.bias.detach().numpy())
    for key in range(KEYS):
        put = instruction(OP_PUT, key, 3)
        assert np.array_equal(first.gate(put), second.gate(put))
    assert not np.array_equal(first.weight.detach().numpy(),
                              other.weight.detach().numpy())


def test_the_gate_does_not_depend_on_the_global_torch_rng() -> None:
    """The initialisation is drawn from an explicit generator.

    A gate whose weights moved because something else in the process consumed
    random numbers would make every "trained at seed 3" claim unreproducible.
    """
    torch.manual_seed(11)
    first = LearnedGate(KEYS, WIDTH, seed=5)
    torch.manual_seed(999)
    torch.rand(17)
    second = LearnedGate(KEYS, WIDTH, seed=5)

    assert np.array_equal(first.weight.detach().numpy(),
                          second.weight.detach().numpy())
    assert np.array_equal(first.bias.detach().numpy(),
                          second.bias.detach().numpy())
    assert not np.array_equal(
        first.weight.detach().numpy(),
        LearnedGate(KEYS, WIDTH, seed=6).weight.detach().numpy())


def test_the_state_loss_is_zero_for_the_hand_set_gate_and_not_for_a_soft_one() -> None:
    """Analytic, not fitted: the exact gate reaches the target exactly."""
    tasks = selective_suite(8, seed=0)
    batches = gate_batch(tasks, KEYS, WIDTH)
    features, values, target, steps = stack_batch(batches)

    weight, bias = saturation_weights(KEYS, WIDTH)
    exact = LearnedGate(KEYS, WIDTH, seed=0)
    with torch.no_grad():
        exact.weight.copy_(torch.tensor(weight))
        exact.bias.copy_(torch.tensor(bias))
    soft = LearnedGate(KEYS, WIDTH, seed=0, init=MIDPOINT_INIT)

    with torch.no_grad():
        assert float(state_loss(exact, features, values, target, steps,
                                dense=True)) == 0.0
        assert float(state_loss(exact, features, values, target, steps,
                                dense=False)) == 0.0
        assert float(state_loss(soft, features, values, target, steps,
                                dense=True)) > 0.0


def test_a_soft_gate_is_not_a_gentle_blend_it_is_an_erase() -> None:
    """The measured reason the raw sigmoid gate is useless at this A.

    ``exp(A_HOLD * w)`` is the multiplier a hold applies to a slot. At ``w = 0``
    it is exactly 1; at ``w = 0.1`` it is below 1e-30. A gate that is soft is
    therefore not blending -- it is overwriting at a scale.
    """
    assert np.exp(A_HOLD * 0.0) == 1.0
    assert np.exp(A_HOLD * 0.1) < 1e-30
    assert np.exp(A_HOLD * 0.01) < 1e-3

    gate, _ = trained_test_gate()
    _, tasks = small_suites()
    measured = sharpness(gate, tasks, KEYS, WIDTH, temperature=1.0)

    assert measured["mean_weight_on_held_slots"] > 0.05
    assert measured["mean_deviation"] > 0.01
    # ...and yet every rounded gate is the right one, which is what says the
    # learning got the addressing right and left the hardness to the sigmoid.
    assert measured["rounded_one_hot_fraction"] == 1.0


# --------------------------------------------------------------------------
# Features, supervision shapes, and the module's own imports
# --------------------------------------------------------------------------

def test_the_features_are_one_hot_and_noise_carries_no_key() -> None:
    """``NOISE``'s operand is a value, so it gets no key dimension.

    A single linear layer sums feature contributions, so a shared key dimension
    would make "this key matters for PUT and not for NOISE" unrepresentable.
    """
    put = instruction(OP_PUT, 2, 7)
    noise = instruction(OP_NOISE, 7)
    recall = instruction(OP_RECALL, 2)

    put_features = gate_features(put, 4)
    assert put_features.sum() == 2.0
    assert put_features[OP_CODES[OP_PUT]] == 1.0
    assert put_features[OPCODE_FEATURES + 2] == 1.0

    noise_features = gate_features(noise, 4)
    assert noise_features.sum() == 1.0
    assert noise_features[OP_CODES[OP_NOISE]] == 1.0
    assert noise_features[OPCODE_FEATURES:].sum() == 0.0

    assert gate_features(recall, 4)[OPCODE_FEATURES + 2] == 1.0

    with pytest.raises(ValueError, match="outside the key vocabulary"):
        gate_features(instruction(OP_PUT, 4, 1), 4)
    with pytest.raises(ValueError, match="n_keys"):
        gate_features(put, 0)


def test_the_noise_operand_feature_is_off_by_default_and_on_in_the_ablation() -> None:
    """A ``NOISE`` value is not a key unless the ablation asks for it.

    The default keeps the value out of the key block; with ``noise_operand`` set
    the same value arrives along the key's dimension, and a value outside the
    vocabulary still sets nothing either way.
    """
    noise = instruction(OP_NOISE, 3)
    outside = instruction(OP_NOISE, 9)

    assert gate_features(noise, 4)[OPCODE_FEATURES:].sum() == 0.0
    assert gate_features(noise, 4, noise_operand=True)[OPCODE_FEATURES + 3] == 1.0
    assert gate_features(outside, 4,
                         noise_operand=True)[OPCODE_FEATURES:].sum() == 0.0
    # PUT still requires its key to be in the vocabulary either way.
    with pytest.raises(ValueError, match="outside the key vocabulary"):
        gate_features(instruction(OP_PUT, 9, 1), 4, noise_operand=True)

    _, tasks = small_suites()
    train, _ = small_suites()
    gate, _ = train_gate(train, KEYS, WIDTH, seed=1, steps=TEST_STEPS, lr=0.05,
                         noise_operand=True)
    hard = noise_hold_fraction(gate, tasks, WIDTH)
    assert 0.0 < hard <= 1.0


def noise_hold_fraction(gate, tasks, state_width: int) -> float:
    """Fraction of NOISE events whose rounded gate is all zeros."""
    hard = total = 0
    for task in tasks:
        for instr in value_instructions(task):
            if instr.op != OP_NOISE:
                continue
            w = np.asarray(gate.gate(instr), dtype=np.float64)
            hard += int(np.array_equal(np.round(w), np.zeros(state_width)))
            total += 1
    return hard / total


def test_the_address_feature_map_shares_weights_between_keys_of_one_slot() -> None:
    """``key % state_width`` is what makes an unseen key address a seen slot."""
    near = gate_features(instruction(OP_PUT, 13, 5), 32, key_mode="address",
                         state_width=16)
    far = gate_features(instruction(OP_PUT, 29, 5), 32, key_mode="address",
                        state_width=16)

    assert near[OPCODE_FEATURES + 13] == 1.0
    assert np.array_equal(near, far)
    assert gate_features(instruction(OP_PUT, 13, 5), 32).shape == \
        (OPCODE_FEATURES + 32,)
    with pytest.raises(ValueError, match="state_width"):
        gate_features(instruction(OP_PUT, 13, 5), 32, key_mode="address")


def test_a_batch_whose_keys_collide_is_rejected_rather_than_optimised() -> None:
    """Two stored keys in one slot is unsolvable, not hard."""
    plan = (instruction(OP_PUT, 1, 11),
            instruction(OP_PUT, 5, 22),
            instruction(OP_RECALL, 1))
    task = Task("collide", "selective", plan, (), 11, "by hand")

    with pytest.raises(ValueError, match="collide"):
        gate_batch((task,), n_keys=8, state_width=4)
    # At a width where they do not collide, the same task is accepted.
    batches = gate_batch((task,), n_keys=8, state_width=8)
    assert batches[0].answer == 11


def test_the_supervision_targets_hold_the_queried_answer() -> None:
    tasks = selective_suite(10, seed=4)
    for batch in gate_batch(tasks, KEYS, WIDTH):
        assert batch.target[batch.query % WIDTH] == float(batch.answer)
        assert batch.steps[-1].tolist() == batch.target.tolist()
        assert batch.steps.shape == (len(value_instructions(
            next(t for t in tasks if t.task_id == batch.task_id))), WIDTH)


def test_sigmoid_saturation_is_where_the_docstrings_say_it_is() -> None:
    """The two thresholds the representability claim rests on, in float64.

    ``1.0`` arrives early because ``1 + exp(-z)`` rounds to 1; ``0.0`` needs the
    exponential to overflow, which is 700 logits further out. ``saturation_weights``
    needs both, so its guard is set by the second.
    """
    with np.errstate(over="ignore"):
        assert 1.0 / (1.0 + np.exp(-36.74)) == 1.0
        assert 1.0 / (1.0 + np.exp(709.79)) == 0.0
        assert 1.0 / (1.0 + np.exp(-36.7)) < 1.0
        # One hundredth short of the threshold: a subnormal, but not zero.
        assert 1.0 / (1.0 + np.exp(709.78)) > 0.0
    with pytest.raises(ValueError, match="big"):
        saturation_weights(KEYS, WIDTH, big=709.0)


def test_bad_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="init"):
        LearnedGate(KEYS, WIDTH, init="nonsense")
    with pytest.raises(ValueError, match="state_width"):
        LearnedGate(KEYS, 0)
    with pytest.raises(ValueError, match="temperature"):
        at_temperature(LearnedGate(KEYS, WIDTH), 0.0)
    with pytest.raises(ValueError, match="big"):
        saturation_weights(KEYS, WIDTH, big=10.0)
    with pytest.raises(ValueError, match="anneal_to"):
        train_gate(selective_suite(2, seed=0), KEYS, WIDTH, anneal_to=0.0)
    with pytest.raises(ValueError, match="steps"):
        train_gate(selective_suite(2, seed=0), KEYS, WIDTH, steps=0)


def test_the_module_reads_nothing_but_numpy_torch_and_the_agent() -> None:
    """No clock, no filesystem, no network, no global RNG in the module.

    ``torch`` is the repository's optional extra, which is also why this module
    is not re-exported from ``beyond_attention/__init__.py``: importing the
    package must not require torch. That decision is asserted, not assumed.
    """
    source = REPO / "src" / "beyond_attention" / "learned_gate.py"
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"numpy", "torch", "dataclasses", "math", "typing",
                        "__future__", "agent"}, imported
    for forbidden in ("time", "random", "os", "socket", "pathlib", "json"):
        assert forbidden not in imported

    init = (REPO / "src" / "beyond_attention" / "__init__.py").read_text()
    assert "learned_gate" not in init


# --------------------------------------------------------------------------
# The experiment cannot drift from the implementation it measures
# --------------------------------------------------------------------------

def test_the_experiments_hand_set_row_is_what_agent_py_reports() -> None:
    """The published number, the module's number, and the learned path's number.

    ``experiments/learned_gate.py`` reports its hand-set row by running
    ``agent.run_agent``; its learned-path row runs the same tasks through
    ``run_learned`` with the gate forced. Both must equal each other and the
    figure the committed ``agent-loop.json`` already published for the same
    suite, or the experiment is measuring something other than the agent.
    """
    module = load_experiment()
    tasks = selective_suite(50, seed=0, n_store=SELECTIVE_STORES,
                            n_distractors=SELECTIVE_DISTRACTORS,
                            n_keys=SELECTIVE_KEYS)
    budget = STEP_COUNTS["selective"]

    agent_rate, total = module.agent_rate(tasks, budget=budget)
    learned_rate, learned_total = module.solve_rate(
        tasks, HandSetGate(KEYS, WIDTH).gate, KEYS, WIDTH)

    published = json.loads((REPO / "agent-loop.json").read_text())
    published_rate = published["solve_rate"][f"selective@{budget}"]["rate"]

    assert agent_rate == learned_rate == published_rate == 1.0
    assert total == learned_total == 50
    assert published["config"]["selective"]["stores"] == SELECTIVE_STORES
    assert published["config"]["selective"]["keys"] == SELECTIVE_KEYS


def test_the_committed_results_file_is_the_runs_shape() -> None:
    """`learned-gate.json` is a result, so its own claims must be checkable.

    The file is regenerated by the experiment rather than edited, so this checks
    the shape and the internal agreement of the published numbers: the hand-set
    row equals the published agent rate, the two paths of the hand-set gate
    agree, the untrained controls do not match, and the saturation weights do.
    """
    path = REPO / "learned-gate.json"
    payload = json.loads(path.read_text())
    conditions = payload["conditions"]

    assert conditions["hand_set_gate"]["rate"] == 1.0
    assert conditions["hand_set_gate_learned_path"]["rate"] == 1.0
    assert conditions["hand_set_gate_learned_path"][
        "embedding_matches_agent_embed"] is True
    assert conditions["fixed_constant_gate"]["rate"] < 1.0
    assert conditions["learned_gate_untrained_hold_sharpened"]["spread"][
        "max"] < conditions["hand_set_gate"]["rate"]
    assert conditions["learned_gate_trained_sharpened"]["spread"]["min"] > \
        conditions["learned_gate_untrained_hold_sharpened"]["spread"]["max"]
    assert conditions["saturation_weights"][
        "every_gate_vector_equals_hand_set"] is True
    assert payload["representation"]["saturation_weights_match_hand_set"] is True
    assert payload["reference_agreement"]["agree"] is True
    assert payload["reference_agreement"]["experiment_hand_set_rate"] == \
        payload["reference_agreement"]["agent_py_rate"]

    # The two measured results the README's objection turns on: the raw sigmoid
    # gate does not reach the hand-set rate, and the prescribed key feature map
    # does not generalise to held-out keys while the address map does.
    raw = conditions["learned_gate_trained_raw"]["spread"]["max"]
    assert raw < conditions["hand_set_gate"]["rate"]
    held = payload["held_out_keys"]["conditions"]
    assert held["key"]["sharpened_held_out_keys"]["max"] == 0.0
    assert held["key"]["sharpened_train_keys"]["min"] == 1.0
    assert held["address"]["sharpened_held_out_keys"]["min"] == 1.0
    # The cause is in the gate, not in the reader: on a held-out key the feature
    # row is at its initialisation, so the gate writes essentially nothing.
    assert held["key"]["sharpness_held_out_keys"][
        "mean_weight_on_addressed_slots"] < 0.2
    assert held["key"]["sharpness_held_out_keys"][
        "rounded_one_hot_fraction"] < 0.5
    assert held["address"]["sharpness_held_out_keys"][
        "rounded_one_hot_fraction"] == 1.0
    assert held["address"]["sharpness_held_out_keys"][
        "mean_weight_on_addressed_slots"] > 0.9

    # The ablation: with the noise operand present as a key feature the gate
    # *almost* always still holds NOISE at zero, and pays a little accuracy for
    # it. Both numbers are published, and this asserts the direction rather
    # than pretending the cost is not there.
    ablation = payload["feature_ablation"]
    assert ablation["noise_only_opcode"]["noise_hard_zero_fraction"] == 1.0
    assert ablation["noise_operand"]["noise_hard_zero_fraction"] < 1.0
    assert ablation["noise_operand"]["sharpened"]["mean"] <=         ablation["noise_only_opcode"]["sharpened"]["mean"]
    # Training did not leave the raw gate hard: the hold weights are soft.
    assert payload["sharpness"]["temperature_1_0"]["trained"][
        "mean_weight_on_held_slots"] > 0.05
