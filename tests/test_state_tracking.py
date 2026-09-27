"""The second-task experiment is reproducible, which it was not.

`experiments/state_tracking.py` runs a register/parity task with a matched-budget
comparison, a step-budget control and a trained-at-that-length reference. Its
committed results file could not be reproduced by any configuration, and the
cause was not a different setting: `train()` seeds the global RNG when it starts,
but the model is *built* before that call, so every run drew different initial
weights from an unseeded process. Two runs at identical flags disagreed on
accuracy while both reported the same `seeds`.

`build()` seeds now, and this pins that. It is deliberately a construction test
rather than a training test: the failure it guards against is a model
initialised from an unseeded RNG, which is visible in two parameter tensors and
costs milliseconds, where reproducing it end to end costs twenty minutes.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _state_tracking():
    spec = importlib.util.spec_from_file_location(
        "state_tracking_for_tests", ROOT / "experiments" / "state_tracking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state_tracking = _state_tracking()


def _parameters(model) -> dict:
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


@pytest.mark.parametrize("block", ["attention", "ssm"])
def test_building_the_same_model_twice_gives_the_same_weights(block: str) -> None:
    """The experiment's reproducibility rests on this and nothing else.

    Without it the reference, the matched-budget arm and the control arm are all
    single samples from an unknown distribution, and `--verify` cannot exist.
    """
    first = _parameters(state_tracking.build(block, 64, 2, seed=0))
    second = _parameters(state_tracking.build(block, 64, 2, seed=0))
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].equal(second[name]), (
            f"{block}: {name} differs between two builds at the same seed, so "
            f"model construction is not seeded and no run is reproducible")


def test_a_different_seed_still_gives_a_different_model() -> None:
    """Seeding must not have flattened the seeds into one model.

    A `build()` that ignored its argument and used a constant would pass the test
    above while making the two-seed spreads in the file meaningless.
    """
    zero = _parameters(state_tracking.build("ssm", 64, 2, seed=0))
    one = _parameters(state_tracking.build("ssm", 64, 2, seed=1))
    differing = [name for name in zero if not zero[name].equal(one[name])]
    assert differing, "seed 0 and seed 1 produced identical weights"


def test_the_config_payload_records_what_a_run_needs_to_be_reproduced() -> None:
    """The code side of the same claim, checked without running the experiment.

    Asserting only that the committed file contains the fields would pass after
    somebody removed them from the script, because the file would still have
    them. This calls the function that builds the payload instead, so a field
    dropped from the code fails here in milliseconds rather than being discovered
    by whoever next regenerates the file two hours later.
    """
    import argparse

    def args(**overrides):
        base = dict(train_length=32, eval_lengths=[32, 64, 128], steps=1500,
                    d_model=64, n_layers=2, seeds=[0, 1], batch_size=32, lr=5e-3,
                    control_steps=[3000], reference_seeds=[0, 1],
                    no_reference=False, threads=4)
        base.update(overrides)
        return argparse.Namespace(**base)

    config = state_tracking.config_payload(args(), chance=0.5)
    for field in ("batch_size", "lr", "control_steps", "reference_seeds",
                  "reference_run", "threads"):
        assert field in config, field
    assert config["reference_run"] is True
    assert config["threads"] == 4

    # A run that skipped the reference has to say so, or a file with an empty
    # reference is indistinguishable from one whose reference was never asked for.
    skipped = state_tracking.config_payload(args(no_reference=True), chance=0.5)
    assert skipped["reference_run"] is False


def test_the_payload_records_the_configuration_needed_to_reproduce_it() -> None:
    """The artifact side: the committed file carries that configuration.

    Four fields were missing from it, and the ambiguity was real -- a fresh run
    at the script's defaults did not reproduce the committed numbers and there
    was no way to tell a stale file from a differently-configured one.
    """
    committed = json.loads((ROOT / "state-tracking.json").read_text())
    config = committed["config"]
    for field in ("batch_size", "lr", "control_steps", "reference_seeds",
                  "reference_run", "threads"):
        assert field in config, (
            f"{field} is not recorded in state-tracking.json, so that run's "
            f"configuration cannot be recovered from it")

    # And the second of the two controls the experiment documents has to be in
    # the file rather than an empty dict: it was `{}` until this was fixed, and
    # the docstring promises it.
    assert committed["reference"], (
        "the reference control is empty in the committed file, so the "
        "trained-at-the-longest-length comparison the docstring describes "
        "never ran")
    longest = max(config["eval_lengths"])
    for block in ("attention", "ssm"):
        entry = committed["reference"][block]
        assert entry["length"] == longest, (block, entry)
        assert len(entry["per_seed"]) == len(config["reference_seeds"]), (block, entry)
