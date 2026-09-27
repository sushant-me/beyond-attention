"""Every experiment that trains a model seeds the RNG before building it.

Both sweeps here shared one construction bug: the model was built *before*
`train()` was called, and `train()` is what seeds the global RNG. So the first
model -- or in `length_extrapolation.py`, every model -- drew its initial weights
from process entropy, and `config["seeds"]` recorded a seed that did not describe
the run.

It was caught in `run.py` by re-running the documented command. Every one of the
eight in-distribution accuracies came back bit-identical, the parameter counts
were identical, and one off-size cell moved: `attention@2` evaluated at 4 pairs
returned 0.776 (1,590 of 2,048) against the published 0.769 (1,574 of 2,048). The
inference was that the weights were identical and the arithmetic was not, and the
measurement that refuted it was a fingerprint of the *first* model's parameters
after replicating `main()`'s preamble: 30.85, then -22.88, then 28.55 on three
successive launches, against 28.80 three times with a seed set first.

`experiments/state_tracking.py` was the first place this was fixed and has its
own module, `tests/test_state_tracking.py`, with the same two checks. This one
covers the other two. It is deliberately a construction test rather than a
training test: the fault shows up in two parameter tensors in milliseconds, where
reproducing it end to end costs seventeen minutes for one sweep.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_EVAL_VOCAB = 33  # vocabulary_size(16), what both sweeps train with


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_for_tests", ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_run = _load("run")
_length_extrapolation = _load("length_extrapolation")

# module, and the one call in it that constructs a model to be trained
BUILDERS = {
    "run.py": (_run, lambda m, block, seed: m.build_model(
        m.vocabulary_for(16), 64, 2, block, seed)),
    "length_extrapolation.py": (_length_extrapolation, lambda m, block, seed: m.build(
        block, _EVAL_VOCAB, 64, 2, seed)),
}


def _parameters(model) -> dict:
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


@pytest.mark.parametrize("experiment", sorted(BUILDERS))
@pytest.mark.parametrize("block", ["attention", "ssm"])
def test_building_the_same_model_twice_gives_the_same_weights(
        experiment: str, block: str) -> None:
    """The reproducibility of every published row rests on this and nothing else.

    Without it each row is a single draw from an unknown distribution, the
    recorded seeds describe only part of the run, and the ``±`` spread the
    README prints understates the real variation by however much the
    initialisation contributes -- which is an independent source of the same
    size, not a smaller one.
    """
    module, build = BUILDERS[experiment]
    first = _parameters(build(module, block, 0))
    second = _parameters(build(module, block, 0))
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].equal(second[name]), (
            f"{experiment} {block}: {name} differs between two builds at the "
            f"same seed, so construction is not seeded and no run is "
            f"reproducible")


@pytest.mark.parametrize("experiment", sorted(BUILDERS))
def test_a_different_seed_still_gives_a_different_model(experiment: str) -> None:
    """Seeding must not have collapsed the seeds, or `--seeds` is decoration."""
    module, build = BUILDERS[experiment]
    zero = _parameters(build(module, "attention", 0))
    one = _parameters(build(module, "attention", 1))
    assert any(not zero[name].equal(one[name]) for name in zero), (
        f"{experiment}: seeds 0 and 1 build identical models, so the seeds "
        f"never reach the initialisation")


@pytest.mark.parametrize("experiment", sorted(BUILDERS))
def test_a_build_does_not_inherit_whatever_the_process_left_behind(
        experiment: str) -> None:
    """The defect was specifically that a build had no seed of its own.

    A build that only appeared deterministic because a previous call had seeded
    the generator would pass the two tests above and still leave the first model
    of a sweep at the mercy of process entropy. So consume the generator the way
    a training run does, then build and compare against the seeded weights.
    """
    module, build = BUILDERS[experiment]
    expected = _parameters(build(module, "attention", 0))
    module.LanguageModel(
        _EVAL_VOCAB, 64, 2, "attention", **module.BLOCK_KWARGS["attention"])
    rebuilt = _parameters(build(module, "attention", 0))
    for name in expected:
        assert expected[name].equal(rebuilt[name]), (
            f"{experiment}: {name} depends on what ran before the build, so the "
            f"initialisation is inherited from the process rather than set from "
            f"the seed")
