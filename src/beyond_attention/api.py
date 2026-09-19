"""Two calls that make this repository's measurements usable from code.

Everything else here is a module you have to know your way around: ``voice.py``
exposes per-frame features and a descriptor dict, ``agent.py`` exposes a loop and
a trace, and both expect the caller to assemble them. This module is the small
importable surface on top:

    >>> import numpy as np
    >>> from beyond_attention import affect_summary, run_task
    >>> t = np.arange(16_000) / 16_000
    >>> summary = affect_summary(0.5 * np.sin(2 * np.pi * 220 * t), 16_000)
    >>> round(summary.descriptors["f0_mean"], 1)
    221.6
    >>> summary.reading.split(".")[0]
    'Pitch is high, centred on 222 Hz and almost level (spread 1.2 Hz) across 100% of frames'
    >>> run = run_task()                       # the example task, start to finish
    >>> [(step.action.tool, step.result.value) for step in run.steps]
    [('add', 3), ('add', 7), ('mul', 35), ('lookup', 7), ('finish', 7)]

``affect_summary`` returns the descriptor dict **and** a sentence about it, and
the sentence is generated *from the dict* rather than from the waveform or from a
template with the numbers pasted in, so the words and the numbers cannot drift
apart. ``run_task`` returns the whole ``AgentRun`` -- every step, with the
registers the policy read, the call it made and the observation it got -- because
the trace is the artifact; ``trace_lines`` renders it as text when that is what
is wanted.

What this module is **not**, stated here rather than left to be discovered:

* **``affect_summary`` is not an emotion judgement.** It measures the classic
  acoustic correlates of prosody -- pitch level and spread, loudness and its
  movement, voiced ratio, jitter, a speaking-rate proxy -- and describes them.
  Nothing in this path was trained on labelled affect data, so a summary is a
  measurement of the signal, not a reading of how anyone felt. The reading says
  so in its own last sentence.
* **``run_task`` is not a general agent.** It runs the repository's one
  hand-written controller over its closed task family and closed tool set, with
  no gradient anywhere, and the task it is given has to come from that family.
* **Neither function adds a number.** Every figure returned is computed by
  ``voice.py`` or ``agent.py``; this module only assembles them, and the reading
  is a phrasing of values that are also in the dict it is returned with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .agent import AgentRun, Task, example_task, run_agent, task_suite
from .voice import (
    HOP_MS,
    WINDOW_MS,
    affect_descriptors,
    frame_features,
)

# The sentence every reading ends on. It is a module constant rather than a line
# inside the generator so that a caller -- and the tests -- can check that the
# framing is present without duplicating the wording.
AFFECT_FRAMING = (
    "These are acoustic correlates -- measurements of how the signal moves, not "
    "a judgement about how anyone felt: nothing in this path was trained on "
    "labelled affect data, and no emotion label is being read off."
)

# Bands used to put the numbers into words. They are round numbers on purpose:
# the point is to describe which part of the range a measurement sits in, not to
# draw a boundary that means something. The reading always prints the value as
# well, so a reader who disagrees with a band still has the number.
_PITCH_LEVELS: tuple[tuple[float, str], ...] = (
    (140.0, "low"),
    (200.0, "mid-range"),
    (250.0, "high"),
)
_PITCH_MOVEMENT: tuple[tuple[float, str], ...] = (
    (0.03, "almost level"),
    (0.08, "gently moving"),
    (0.15, "moving"),
)
_LOUDNESS_MOVEMENT: tuple[tuple[float, str], ...] = (
    (0.10, "steady"),
    (0.30, "modulated"),
)


@dataclass(frozen=True)
class AffectSummary:
    """A prosody measurement and one generated reading of it.

    ``descriptors`` is exactly the dict ``voice.affect_descriptors`` returns --
    the measured numbers, with ``voiced_ratio == 0.0`` as the flag that the pitch
    entries are 0.0 placeholders rather than a pitch. ``reading`` is a sentence
    built from those same numbers, phrased as correlates and ending with
    ``AFFECT_FRAMING``.
    """

    descriptors: dict[str, float]
    reading: str


def _band(value: float, stops: tuple[tuple[float, str], ...], top: str) -> str:
    """The first label whose upper stop the value is below, else the last one."""
    for stop, label in stops:
        if value < stop:
            return label
    return top


def prosody_reading(descriptors: Mapping[str, float]) -> str:
    """A plain-language reading of a descriptor dict.

    Generated from the dict, not from the waveform, so the numbers in the
    sentence are the numbers in the dict by construction and a test can require
    that the formatted value of ``f0_mean`` appears in the reading. It describes
    pitch level and movement, loudness and its movement, and the timing proxies,
    then says what the numbers are not -- see ``AFFECT_FRAMING``.
    """
    voiced_ratio = float(descriptors.get("voiced_ratio", 0.0))
    if voiced_ratio <= 0.0:
        return (
            "No frame in this signal was judged voiced, so there is no pitch "
            "measurement here: f0_mean, f0_std, f0_range and jitter are 0.0 "
            "placeholders rather than a pitch, and only the energy and "
            "zero-crossing numbers describe the signal. " + AFFECT_FRAMING
        )

    f0_mean = float(descriptors.get("f0_mean", 0.0))
    f0_std = float(descriptors.get("f0_std", 0.0))
    energy_mean = float(descriptors.get("energy_mean_voiced", 0.0))
    energy_std = float(descriptors.get("energy_std_voiced", 0.0))
    jitter = float(descriptors.get("jitter", 0.0))
    speaking_rate = float(descriptors.get("speaking_rate", 0.0))

    level = _band(f0_mean, _PITCH_LEVELS, "very high")
    movement = _band(f0_std / f0_mean if f0_mean > 0.0 else 0.0,
                     _PITCH_MOVEMENT, "widely moving")
    loudness = _band(energy_std / energy_mean if energy_mean > 0.0 else 0.0,
                     _LOUDNESS_MOVEMENT, "strongly modulated")

    sentences = [
        f"Pitch is {level}, centred on {f0_mean:.0f} Hz and {movement} "
        f"(spread {f0_std:.1f} Hz) across {voiced_ratio:.0%} of frames.",
        f"Loudness while voiced is {loudness}: RMS spread {energy_std:.3f} "
        f"around {energy_mean:.3f}.",
        f"Timing reads {speaking_rate:.1f} voiced runs per second (a "
        f"syllable-rate proxy) with successive-frame pitch steps averaging "
        f"{jitter:.2f} Hz.",
    ]
    if voiced_ratio < 0.1:
        sentences.append(
            f"Only {voiced_ratio:.0%} of frames were voiced, so the pitch numbers "
            "rest on very few frames and should be read as provisional."
        )
    sentences.append(AFFECT_FRAMING)
    return " ".join(sentences)


def affect_summary(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = WINDOW_MS,
    hop_ms: float = HOP_MS,
) -> AffectSummary:
    """Waveform in, descriptors plus a reading of them out. One call.

    The waveform is framed and measured by ``voice.frame_features`` and
    summarised by ``voice.affect_descriptors``; this function does no arithmetic
    of its own beyond passing the frame parameters through. A one-dimensional
    signal is the expected input, and a longer ``window_ms`` is what low-pitched
    audio needs -- the module's documented search range is wider than the range
    the estimator works over.

    See ``AffectSummary`` for what the two returned pieces are, and the module
    docstring for what this is not.
    """
    signal = np.asarray(waveform, dtype=np.float64)
    features = frame_features(signal, sample_rate, window_ms=window_ms,
                              hop_ms=hop_ms)
    descriptors = affect_descriptors(features)
    return AffectSummary(descriptors=descriptors,
                         reading=prosody_reading(descriptors))


def run_task(
    task: Task | str | None = None,
    *,
    budget: int = 6,
    memory: str = "ssm",
    seed: int = 0,
) -> AgentRun:
    """Run one agent task and return the full trace. One call.

    ``task`` is a ``Task``, the name of a family (``"lookup"``, ``"selective"``,
    the first task of that family from the seeded suite), or ``None`` for
    ``agent.example_task()`` -- the task whose trace the README publishes.

    Everything the loop did is in the returned ``AgentRun``: each ``Step`` holds
    the decoded registers the policy read *before* the call, the ``ToolCall`` it
    chose, and the ``ToolResult`` the tool returned, and ``AgentRun.actions`` is
    the call sequence on its own. ``memory`` and ``budget`` are the same knobs
    ``agent.run_agent`` takes -- ``memory="none"`` is the control that wipes the
    state before every decision, and it is one call away because a trace that
    cannot be compared against a control is half a measurement.
    """
    if task is None:
        chosen = example_task()
    elif isinstance(task, str):
        suite = task_suite()
        match = next((candidate for candidate in suite
                      if candidate.family == task), None)
        if match is None:
            families = sorted({candidate.family for candidate in suite})
            raise ValueError(f"unknown task family {task!r}; the suite has "
                             f"{', '.join(families)}")
        chosen = match
    else:
        chosen = task
    return run_agent(chosen, budget=budget, memory=memory, seed=seed)


def trace_lines(run: AgentRun) -> tuple[str, ...]:
    """The trace as one line per decision, for printing or asserting on.

    Every field is read out of the run: the registers the policy acted on, the
    call it made, the observation it got, and the carried value the *next*
    decision read. Recomputing any of them here would be a second place for the
    trace and its rendering to disagree, so nothing is recomputed.
    """
    lines = [
        f"task {run.task_id} ({run.family}), memory {run.memory}, "
        f"budget {run.budget}, stop {run.stop_reason}, solved {run.solved}"
    ]
    for position, step in enumerate(run.steps):
        arguments = ", ".join(f"{key}={value}"
                              for key, value in sorted(step.action.args.items()))
        observation = (step.result.error if step.result.error is not None
                       else step.result.value)
        if position + 1 < len(run.steps):
            carried = f"next read carry {run.steps[position + 1].registers.carry:g}"
        else:
            carried = f"loop stopped ({run.stop_reason})"
        lines.append(
            f"step {step.index}: {step.instruction.op} "
            f"{' '.join(str(arg) for arg in step.instruction.args)} | read carry "
            f"{step.registers.carry:g} obs {step.registers.observations} | "
            f"{step.action.tool}({arguments}) -> {observation} | {carried}"
        )
    lines.append(f"answer {run.answer}")
    return tuple(lines)
