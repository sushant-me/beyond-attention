"""The one-call API, checked against signals and tasks whose answer is known.

The API is thin -- it frames, measures, phrases and runs -- and thin code is easy
to test in a way that proves nothing: a shape check on a dict passes whether or
not the numbers in it are right, and a "the reading is a non-empty string" check
passes for a reading that says the opposite of the measurement. So the tests here
come in the same two kinds as `tests/test_voice.py` and `tests/test_agent.py`.

**Analytic ground truth.** A 220 Hz tone has one right answer for `f0_mean` and
it is known before the code runs. The example task's answer is 7 by hand, and its
trace is checked step by step against the published sequence -- including the
property the whole agent block rests on, that each decision's carried value is
the previous step's observation.

**Controls that can fail.** The reading must change when the pitch changes, because
a constant string would pass any check that only looks for a number; it must say
there is *no* pitch measurement when nothing was voiced, rather than reporting
0.0 Hz as a pitch; and every reading must carry the correlates framing, because
the failure mode of a plain-language summary is that it reads like a diagnosis.
The `memory="none"` control is exercised here too: an API boundary that cannot
reach the controls cannot reproduce the measurements the README publishes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import beyond_attention
from beyond_attention import (
    AFFECT_FRAMING,
    AffectSummary,
    affect_descriptors,
    affect_summary,
    frame_features,
    prosody_reading,
    run_task,
    trace_lines,
)

SAMPLE_RATE = 16_000
F0_TOLERANCE = 0.02  # the tolerance test_voice.py asserts, for the same reason


def tone(freq_hz: float, seconds: float = 1.0,
         amplitude: float = 0.5) -> np.ndarray:
    """A pure sine: the signal whose F0 is known before the code runs."""
    t = np.arange(int(round(seconds * SAMPLE_RATE))) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# --------------------------------------------------------------------------
# affect_summary: analytic ground truth
# --------------------------------------------------------------------------

@pytest.mark.parametrize("freq_hz", [150.0, 220.0, 300.0])
def test_a_known_tone_is_summarised_with_the_known_pitch(freq_hz: float) -> None:
    summary = affect_summary(tone(freq_hz), SAMPLE_RATE)

    assert isinstance(summary, AffectSummary)
    assert isinstance(summary.descriptors, dict)
    assert all(isinstance(value, float) for value in summary.descriptors.values())
    assert summary.descriptors["f0_mean"] == pytest.approx(
        freq_hz, rel=F0_TOLERANCE
    )
    # The reading is phrased from the same dict, so the number the tone was
    # generated at must appear in it: a summary that measured correctly and then
    # said something else would pass every check above.
    assert f"{summary.descriptors['f0_mean']:.0f} Hz" in summary.reading


def test_the_summary_is_the_measurement_and_nothing_else() -> None:
    """The API adds no number: the dict is exactly what voice.py computes."""
    signal = tone(180.0)
    expected = affect_descriptors(frame_features(signal, SAMPLE_RATE))

    assert affect_summary(signal, SAMPLE_RATE).descriptors == expected


def test_the_reading_follows_the_pitch_rather_than_a_template() -> None:
    """Two signals an octave and a half apart must not read the same."""
    low = affect_summary(tone(90.0), SAMPLE_RATE)
    high = affect_summary(tone(300.0), SAMPLE_RATE)

    assert low.reading != high.reading
    assert "low" in low.reading and "very high" in high.reading
    assert low.descriptors["f0_mean"] < high.descriptors["f0_mean"]


def test_white_noise_is_summarised_without_a_pitch_claim() -> None:
    """The control that can fail: noise must not be described as having a pitch."""
    signal = 0.3 * np.random.default_rng(0).standard_normal(SAMPLE_RATE)
    summary = affect_summary(signal, SAMPLE_RATE)

    assert summary.descriptors["voiced_ratio"] == 0.0
    assert summary.descriptors["f0_mean"] == 0.0
    assert "no pitch measurement" in summary.reading
    assert "centred on 0 Hz" not in summary.reading
    assert AFFECT_FRAMING in summary.reading


def test_every_reading_is_framed_as_correlates_not_an_emotion_judgement() -> None:
    for signal in (tone(200.0), 0.3 * np.random.default_rng(1).standard_normal(
            SAMPLE_RATE)):
        reading = affect_summary(signal, SAMPLE_RATE).reading
        assert AFFECT_FRAMING in reading
        assert "not a judgement about how anyone felt" in reading
        assert "emotion label" in reading


def test_the_reading_states_the_numbers_in_the_dict_it_is_given() -> None:
    """``prosody_reading`` is a pure function of the dict, and this pins it.

    The dict here is written out by hand rather than measured, so the expected
    sentence is a statement about the phrasing and not about the estimator.
    """
    descriptors = {
        "f0_mean": 240.0, "f0_std": 50.0, "jitter": 3.5,
        "energy_mean_voiced": 0.25, "energy_std_voiced": 0.01,
        "voiced_ratio": 0.75, "speaking_rate": 4.0,
    }
    reading = prosody_reading(descriptors)

    assert "Pitch is high" in reading
    assert "centred on 240 Hz" in reading
    assert "widely moving" in reading
    assert "spread 50.0 Hz" in reading
    assert "across 75% of frames" in reading
    assert "voiced runs per second" in reading
    assert "3.50 Hz" in reading
    assert reading == prosody_reading(dict(descriptors))


def test_a_reading_of_a_mostly_unvoiced_signal_says_the_pitch_is_provisional() -> None:
    descriptors = {
        "f0_mean": 120.0, "f0_std": 1.0, "jitter": 0.0,
        "energy_mean_voiced": 0.2, "energy_std_voiced": 0.01,
        "voiced_ratio": 0.05, "speaking_rate": 0.5,
    }
    reading = prosody_reading(descriptors)

    assert "provisional" in reading
    assert "5% of frames were voiced" in reading


def test_a_non_waveform_input_is_rejected_rather_than_measured() -> None:
    with pytest.raises((ValueError, TypeError)):
        affect_summary("not a waveform", SAMPLE_RATE)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# run_task: the published trace, at the API boundary
# --------------------------------------------------------------------------

def test_run_task_returns_the_published_trace_step_by_step() -> None:
    run = run_task()

    assert run.task_id == "example"
    assert run.family == "example"  # hand-written, so it is its own family
    assert run.budget == 6
    assert run.stop_reason == "finish"
    assert run.solved is True
    assert run.answer == 7
    assert [(tool, args) for tool, args in run.actions] == [
        ("add", (("a", 0), ("b", 3))),
        ("add", (("a", 3), ("b", 4))),
        ("mul", (("a", 7), ("b", 5))),
        ("lookup", (("value", 35),)),
        ("finish", (("answer", 7),)),
    ]


def test_each_decision_reads_the_previous_decision_s_observation() -> None:
    """The property the agent block rests on, checked at the API boundary.

    Step 0 has no observation behind it; every later step's carried value is the
    value the previous step's tool returned, read out of the state rather than
    out of the task text.
    """
    run = run_task()

    for previous, following in zip(run.steps, run.steps[1:]):
        assert following.registers.carry == previous.result.value
        assert following.registers.observations == previous.registers.observations + 1
    assert run.steps[0].registers.carry == 0.0
    assert run.steps[0].registers.observations == 0


def test_run_task_accepts_a_family_name() -> None:
    run = run_task("lookup")

    assert run.family == "lookup"
    assert run.solved is True
    assert len(run.steps) == beyond_attention.STEP_COUNTS["lookup"]


def test_an_unknown_family_is_rejected_with_the_families_that_exist() -> None:
    with pytest.raises(ValueError) as info:
        run_task("no_such_family")

    assert "no_such_family" in str(info.value)
    assert "literal" in str(info.value)


def test_the_no_memory_control_is_reachable_from_the_api() -> None:
    """A trace that cannot be compared against a control is half a measurement."""
    assert run_task("two_op").solved is True
    assert run_task("two_op", memory="none").solved is False
    # The same control keeps the family whose answer is in the task text, which
    # is what makes it a control rather than a broken loop.
    assert run_task("literal", memory="none").solved is True


# --------------------------------------------------------------------------
# trace_lines
# --------------------------------------------------------------------------

def test_trace_lines_reads_its_values_out_of_the_run() -> None:
    run = run_task()
    lines = trace_lines(run)

    assert len(lines) == len(run.steps) + 2  # header, one per step, answer
    for step, line in zip(run.steps, lines[1:-1]):
        assert f"step {step.index}:" in line
        assert step.action.tool in line
        assert str(step.result.value) in line
    assert lines[-1] == f"answer {run.answer}"


def test_trace_lines_follows_the_run_rather_than_recomputing_it() -> None:
    """Alter a step's result and the rendered line has to change with it."""
    run = run_task()
    step = run.steps[1]
    altered_step = dataclasses.replace(
        step, result=dataclasses.replace(step.result, value=99)
    )
    altered = dataclasses.replace(
        run, steps=(run.steps[0], altered_step) + run.steps[2:]
    )

    assert "99" in trace_lines(altered)[2]
    assert "99" not in trace_lines(run)[2]


# --------------------------------------------------------------------------
# Export surface
# --------------------------------------------------------------------------

def test_the_api_is_exported_from_the_package() -> None:
    for name in ("affect_summary", "prosody_reading", "AffectSummary",
                 "AFFECT_FRAMING", "run_task", "trace_lines"):
        assert name in beyond_attention.__all__, name
        assert hasattr(beyond_attention, name), name
