"""A *trained* classifier over synthesised prosody, and the limits of what it shows.

The previous increment (`voice.py`) produced interpretable prosodic descriptors
and said, in the module and in the README, that **no classifier had been
trained**. This module closes that specific gap. It does not close the larger
one, and the distinction is the whole point of the file.

What is here
------------

* An **acoustically-grounded condition set** -- five conditions (crying /
  sad-sobbing, excited, angry, calm, afraid / anxious) whose parameters are not
  invented but taken from the published correlates of those states, cited below
  and next to every parameter.
* A **synthesiser** that can vary exactly the axes those correlates name: F0
  level and contour, F0 modulation (tremor), jitter, shimmer, breathiness
  (harmonic-to-noise ratio), pause structure, rate, energy and onset sharpness.
* **New voice-quality features** the front end did not have: harmonics-to-noise
  ratio, shimmer, tremor rate, F0 slope and final-to-initial ratio, pause
  structure, and onset sharpness. They are extracted here rather than assumed
  to exist in `voice.py`.
* A **trained classifier** -- a small PyTorch MLP with an actual optimisation
  loop, a train/validation/test split **by utterance**, a fixed seed, and model
  selection on the validation set only. The headline number is held-out test
  accuracy.
* Controls that are able to fail: shuffled labels, chance and majority
  baselines, a trained-on-random-features control, per-feature and F0-family
  ablations, a cry-versus-excited confusion analysis, and a
  leave-one-condition-out cross-condition generalisation check.

What it is not -- the important part
------------------------------------

**The labels are the synthesiser's.** Every "crying" utterance is a signal this
module generated from a parameter tuple that this module chose, and its label is
that tuple's name. No human listener ever heard it, no annotator ever labelled
it, and no real recording is involved anywhere. A classifier that scores highly
here has learned **our acoustic model of these emotions**, which is a statement
about five clusters in a 26-dimensional space we placed there by hand. It is
**not** evidence that it recognises emotion in a voice, and it cannot be: the
target is our own generator.

**It is not validated on speech.** There is no affect corpus in this repository
and none is downloaded here. All five conditions are synthetic, so nothing below
transfers to real speech without a measurement that has not been made.

**Five conditions is our partition, not the literature's.** Published affect
work reports continuous dimensions (arousal, valence) and heavily overlapping
acoustic profiles, not five boxes. The five names are a convenience for a
controlled experiment, and the confusion matrix is a statement about five
clusters, not about five emotions.

**A high held-out accuracy is the expected result, not a finding.** The
conditions were built to be separable along the axes the features measure, so
the interesting numbers are the controls: what happens when the labels are
shuffled, when the features are random, when F0 is removed, and when a whole
condition is held out.

Citations for the condition parameters
--------------------------------------

The acoustic profiles are taken from:

* Banse, R., & Scherer, K. R. (1996). Acoustic profiles in vocal emotion
  expression. *Journal of Personality and Social Psychology*, 70(3), 614-636.
  The reference study for F0 level/spread, energy, rate and spectral balance
  across anger, fear, sadness, happiness and neutrality.
* Juslin, P. N., & Laukka, P. (2003). Communication of emotions in vocal
  expression and music performance: Different channels, same code?
  *Psychological Bulletin*, 129(5), 770-814. Meta-analysis; the source of the
  directional expectations used below (arousal raises energy and rate, etc.).
* Scherer, K. R. (1986). Vocal affect expression: A review and a model for
  future research. *Psychological Bulletin*, 99(2), 143-165.
* Eyben, F., Scherer, K. R., Schuller, B. W., et al. (2016). The Geneva
  Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective
  computing. *IEEE Transactions on Affective Computing*, 7(2), 190-202. The
  reason jitter, shimmer and HNR are in the feature set at all: they are the
  standard voice-quality parameters for this task.
* Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency
  and the harmonics-to-noise ratio of a spoken utterance. *IFA Proceedings* 17,
  97-110. The HNR estimator used in `frame_hnr_db`.
* LaGasse, L. L., Neal, A. R., & Lester, B. M. (2005). Assessment of infant
  cry: Acoustic cry analysis and parental perception. *Mental Retardation and
  Developmental Disabilities Research Reviews*, 11(1), 83-93. The cry profile:
  high F0, a falling contour, harsh/breathy phonation, irregular rhythm.
* Patel, S., Scherer, K. R., Bjorkner, E., & Sundberg, J. (2011). Mapping
  emotions into acoustic space: The role of voice production. *Biological
  Psychology*, 87(1), 93-98. Why arousal and valence are treated as separate
  axes rather than one emotion label.

Every parameter below carries the correlate it encodes. Where a correlate is
directional ("higher", "faster") and the published literature does not give a
number, the number here is **this module's choice**, and it is not a measurement
of anything until the experiment measures it.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from .voice import (
    VoiceFeatures,
    affect_descriptors,
    frame_features,
    frame_signal,
)

# --------------------------------------------------------------------------
# Constants. Each is a choice; each is named and used where it is justified.
# --------------------------------------------------------------------------

# An unvoiced stretch counts as a *pause* rather than an unvoiced consonant
# only if it lasts at least this long. At the default 10 ms hop a stop closure
# is a few frames, so 80 ms separates "a gap inside a word" from "a pause".
PAUSE_MIN_MS = 80.0

# The band the tremor estimator searches. Below 0.5 Hz is a trend, not a
# tremor; above ~12 Hz is outside the expressive-vibrato and pathological-tremor
# range this feature is a proxy for.
TREMOR_BAND_HZ = (0.5, 12.0)

# HNR is reported in dB with the correlation clipped just below 1, which caps
# the value at 60 dB. A perfectly periodic frame is not a 200 dB frame; the cap
# is stated here rather than left as an inf in the output.
HNR_MAX_DB = 60.0
_HNR_R_CLIP = 1.0 - 1e-6

# The breath-noise component is low-passed at this corner frequency rather than
# left white to Nyquist. Aspiration noise in a real voice is not flat to
# Nyquist, and unfiltered white noise contributes so much energy around 4 kHz
# that it dominates the spectral centroid of every breathy condition -- measured
# at 3.1 kHz for the breathiest condition against 1.8 kHz for the brightest,
# which turns a voice-quality knob into a spectral-centroid knob.
BREATH_NOISE_CUTOFF_HZ = 2500.0

_EPS = 1e-12

# --------------------------------------------------------------------------
# The condition set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmotionCondition:
    """One synthetic affective condition, and the published correlate per field.

    The field names are the generator's knobs. The comment on each field in
    ``CONDITIONS`` below is the correlate that fixed its value for that
    condition; the class-level meaning of each knob is:

    ``f0_base``
        Pitch level in Hz. Rises with arousal and with distress.
    ``f0_range_st``
        Peak-to-peak excursion of the slow pitch contour, in semitones, driven
        as a sinusoid at ``tremor_hz``. This is F0 *spread*, not F0 *jitter*:
        the movement is slow enough to be a deliberate contour rather than
        phonatory instability.
    ``f0_slope_st``
        Linear drift across the utterance, in semitones. Positive is rising.
    ``f0_final_drop_st``
        Extra fall applied over the last 28% of the utterance, in semitones.
        The falling terminal contour of a cry or a sob.
    ``tremor_hz`` / ``tremor_depth_st``
        Rate and depth of the fast pitch modulation. A fast, deep modulation
        raises the frame-level successive-F0 difference (the `jitter`
        descriptor), which is how vocal tremor presents to this front end.
    ``jitter_st``
        Standard deviation of a per-syllable random pitch offset, in semitones:
        phonatory instability between syllables.
    ``shimmer``
        Standard deviation of a per-frame amplitude perturbation, relative to
        the local envelope. Cycle-to-cycle amplitude perturbation; at this
        front end's 10 ms hop it is a frame-level shimmer proxy, not the
        cycle-to-cycle measure a Praat-style analysis reports.
    ``breathiness``
        Amplitude of low-passed noise added to the unit-RMS harmonic source, as
        a ratio (see ``BREATH_NOISE_CUTOFF_HZ`` for why it is filtered).
        Turbulence noise in the glottal source is what lowers HNR, so this is
        the breathiness knob. 0.0 is a clean voice.
    ``syllables_per_sec``
        Voiced-segment rate: the ground truth for the speaking-rate proxy.
    ``duty``
        Fraction of each syllable slot that is voiced; the rest is a pause.
    ``slot_jitter``
        Relative standard deviation of syllable slot lengths: irregular rhythm.
    ``amplitude``
        RMS scale of the voiced source.
    ``amplitude_var``
        Relative standard deviation of per-syllable loudness.
    ``attack_s``
        Fade-in time of each syllable, in seconds. Small is a sharp onset.
    ``harmonic_alpha``
        Exponent of the glottal harmonic rolloff: harmonic ``k`` has amplitude
        ``1 / k**alpha``. Smaller is a brighter source and a higher spectral
        centroid, which is how tension presents spectrally.
    """

    name: str
    f0_base: float
    f0_range_st: float
    f0_slope_st: float
    f0_final_drop_st: float
    tremor_hz: float
    tremor_depth_st: float
    jitter_st: float
    shimmer: float
    breathiness: float
    syllables_per_sec: float
    duty: float
    slot_jitter: float
    amplitude: float
    amplitude_var: float
    attack_s: float
    harmonic_alpha: float


CONDITIONS: tuple[EmotionCondition, ...] = (
    # CRYING / SAD-SOBBING.
    # LaGasse et al. (2005) and Scherer (1986): a cry is high in F0 with a
    # *falling* terminal contour, harsh/breathy phonation, irregular rhythm
    # broken by pauses, and -- relative to anger or excitement -- reduced
    # energy. Banse & Scherer (1996) place sadness at low F0 and low energy, so
    # this condition deliberately mixes the cry's high F0 with the reduced
    # energy and broken delivery of distress: "crying" is not the same state as
    # "sad", and conflating them is the usual error.
    EmotionCondition(
        "crying",
        f0_base=320.0,          # high F0: cry correlate
        f0_range_st=2.0,        # a cry holds its pitch rather than ranging widely
        f0_slope_st=-1.0,       # falling overall
        f0_final_drop_st=-6.0,  # strong terminal fall: the sob
        tremor_hz=6.5,          # fast, unstable phonation
        tremor_depth_st=2.5,    # deep enough to be heard as roughness
        jitter_st=0.06,         # high jitter: irregular phonation
        shimmer=0.35,           # high shimmer: amplitude instability
        breathiness=0.55,       # breathy / noisy: low HNR
        syllables_per_sec=3.0,
        duty=0.45,              # irregular rhythm with pauses
        slot_jitter=0.55,       # the pauses are not metronomic
        amplitude=0.22,         # reduced energy
        amplitude_var=0.40,     # sobbing swells
        attack_s=0.060,         # cries swell rather than punch
        harmonic_alpha=1.6,     # energy concentrated low despite the noise
    ),
    # EXCITED.
    # Banse & Scherer (1996) and Juslin & Laukka (2003): happiness/excitement
    # raises F0 level and F0 *range*, raises energy and its variability, and
    # speeds speech up, while phonation stays well controlled (low jitter).
    # The large pitch excursion here is deliberately *slow* (1.2 Hz): it is a
    # contour, not tremor, which is what keeps the frame-level jitter low while
    # the F0 range stays wide.
    EmotionCondition(
        "excited",
        f0_base=280.0,          # high F0
        f0_range_st=3.5,        # wide F0 range
        f0_slope_st=0.5,
        f0_final_drop_st=0.0,
        tremor_hz=1.2,          # slow excursion, not tremor
        tremor_depth_st=3.0,
        jitter_st=0.02,         # low jitter relative to level
        shimmer=0.04,
        breathiness=0.10,
        syllables_per_sec=5.5,  # fast rate
        duty=0.75,              # few pauses
        slot_jitter=0.15,
        amplitude=0.50,         # high energy
        amplitude_var=0.45,     # high energy variance
        attack_s=0.010,
        harmonic_alpha=1.0,
    ),
    # ANGRY.
    # Banse & Scherer (1996): anger raises energy and rate, raises the spectral
    # centroid relative to calm/neutral (a tenser, brighter source), holds F0 at
    # a moderate-to-high level, and produces sharp onsets. Jitter is moderate,
    # not extreme. The centroid correlates are *within*-study comparisons; the
    # measured table below shows that fear's breathy high-frequency voice
    # measures brighter still, which is reported rather than hidden.
    EmotionCondition(
        "angry",
        f0_base=190.0,          # moderate-high F0
        f0_range_st=1.5,
        f0_slope_st=0.0,
        f0_final_drop_st=-1.0,
        tremor_hz=3.0,          # moderate instability, not tremor
        tremor_depth_st=1.5,
        jitter_st=0.03,         # moderate jitter
        shimmer=0.06,
        breathiness=0.15,
        syllables_per_sec=5.0,  # fast rate
        duty=0.70,
        slot_jitter=0.20,
        amplitude=0.72,         # high energy: the loudest condition
        amplitude_var=0.28,
        attack_s=0.005,         # sharp onsets
        harmonic_alpha=0.6,     # high spectral centroid
    ),
    # CALM / NEUTRAL.
    # The low-arousal corner of Banse & Scherer (1996) and Juslin & Laukka
    # (2003): low F0 range, steady energy, a moderate rate, low jitter and
    # shimmer, and a clean (high-HNR) voice.
    EmotionCondition(
        "calm",
        f0_base=135.0,
        f0_range_st=0.5,        # low F0 range
        f0_slope_st=0.0,
        f0_final_drop_st=-0.3,
        tremor_hz=1.0,
        tremor_depth_st=0.4,    # almost no modulation
        jitter_st=0.02,         # low jitter
        shimmer=0.02,           # low shimmer
        breathiness=0.08,       # clean voice, high HNR
        syllables_per_sec=2.2,  # moderate rate
        duty=0.62,
        slot_jitter=0.10,       # steady energy and rhythm
        amplitude=0.30,
        amplitude_var=0.03,     # steady energy
        attack_s=0.040,
        harmonic_alpha=1.5,
    ),
    # AFRAID / ANXIOUS.
    # Juslin & Laukka (2003) and Banse & Scherer (1996): fear raises F0 and
    # jitter, raises rate, and is produced with a weak, poorly supported voice
    # -- hence low energy and a soft onset rather than the hard attack of
    # anger. The fast, deep pitch modulation is the tremulous voice. The rising
    # drift is this module's choice, not a cited correlate, and is flagged as
    # such in `CORRELATES`.
    EmotionCondition(
        "afraid",
        f0_base=300.0,          # high F0
        f0_range_st=2.5,
        f0_slope_st=0.6,        # a rising, questioning contour (our choice)
        f0_final_drop_st=0.0,
        tremor_hz=6.0,          # tremulous
        tremor_depth_st=2.0,
        jitter_st=0.05,         # high jitter
        shimmer=0.16,
        breathiness=0.30,       # breathy, weakly phonated
        syllables_per_sec=5.0,  # fast rate
        duty=0.65,
        slot_jitter=0.35,
        amplitude=0.26,         # low energy
        amplitude_var=0.22,
        attack_s=0.050,         # low energy at onsets: a soft attack
        harmonic_alpha=1.2,
    ),
)

CONDITION_NAMES: tuple[str, ...] = tuple(c.name for c in CONDITIONS)

# The published correlate behind each condition, one string per correlate so
# the README can render them next to the parameters the generator was given.
# A string marked "(our choice)" is a parameter this module picked for
# plausibility, not a correlate taken from a source.
CORRELATES: dict[str, tuple[str, ...]] = {
    "crying": (
        "high F0 with a falling terminal contour (LaGasse et al. 2005)",
        "harsh/breathy phonation, low HNR (LaGasse et al. 2005)",
        "high jitter and shimmer, irregular phonation (LaGasse et al. 2005)",
        "irregular rhythm broken by pauses (Scherer 1986)",
        "reduced energy relative to anger or excitement (Banse & Scherer 1996)",
    ),
    "excited": (
        "high F0 level and wide F0 range (Banse & Scherer 1996)",
        "fast speaking rate (Juslin & Laukka 2003)",
        "high energy and high energy variance (Juslin & Laukka 2003)",
        "low jitter relative to level: controlled phonation (Banse & Scherer 1996)",
        "the wide excursion is a slow contour, not tremor (our choice)",
    ),
    "angry": (
        "high energy (Banse & Scherer 1996)",
        "higher spectral centroid than calm/neutral (Banse & Scherer 1996)",
        "fast speaking rate (Juslin & Laukka 2003)",
        "moderate-high F0 (Banse & Scherer 1996)",
        "sharp onsets (Scherer 1986)",
    ),
    "calm": (
        "low F0 range (Banse & Scherer 1996)",
        "steady energy (Juslin & Laukka 2003)",
        "moderate speaking rate (Juslin & Laukka 2003)",
        "low jitter: stable phonation (Eyben et al. 2016)",
    ),
    "afraid": (
        "high F0 (Juslin & Laukka 2003)",
        "high jitter: tremulous, unstable phonation (Banse & Scherer 1996)",
        "fast speaking rate (Juslin & Laukka 2003)",
        "low energy, weak phonation at onsets (Scherer 1986)",
        "rising contour (our choice)",
    ),
}


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------


def _syllable_edges(
    n_samples: int,
    sample_rate: int,
    condition: EmotionCondition,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample indices at which syllable slots start and end.

    Slot lengths are drawn with relative standard deviation ``slot_jitter`` and
    normalised to fill the utterance exactly, so the rate the caller asked for
    is the rate that comes out and only the *regularity* is randomised.
    """
    seconds = n_samples / float(sample_rate)
    n_syllables = max(1, int(round(condition.syllables_per_sec * seconds)))
    weights = np.maximum(
        0.15, 1.0 + condition.slot_jitter * rng.standard_normal(n_syllables)
    )
    weights /= weights.sum()
    edges = np.round(
        np.concatenate([[0.0], np.cumsum(weights)]) * n_samples
    ).astype(np.int64)
    edges[0] = 0
    edges[-1] = n_samples
    return edges


def _lowpass_noise(
    rng: np.random.Generator, n_samples: int, sample_rate: int
) -> np.ndarray:
    """One-pole low-passed unit-variance noise, for the breathiness component.

    ``y[n] = a y[n-1] + (1-a) x[n]`` with ``a = exp(-2 pi fc / fs)``. The
    impulse response decays as ``a**k``, so at the default corner (2.5 kHz at
    16 kHz, ``a = 0.375``) twenty taps hold all but 1e-8 of the energy and the
    filter is a short convolution rather than a Python loop. The result is
    renormalised to unit variance so ``breathiness`` stays the noise-to-tone
    ratio and only the noise's *shape* changes.
    """
    noise = rng.standard_normal(n_samples)
    a = math.exp(-2.0 * math.pi * BREATH_NOISE_CUTOFF_HZ / sample_rate)
    taps = min(n_samples, 20)
    kernel = (1.0 - a) * a ** np.arange(taps)
    filtered = np.convolve(noise, kernel)[:n_samples]
    return filtered / max(float(filtered.std()), _EPS)


def _interior_voiced(voiced: np.ndarray) -> np.ndarray:
    """Voiced frames that are neither the first nor the last of their run.

    The first and last frame of a voiced segment sit on the onset and offset
    ramps, where the envelope is changing by construction. A perturbation
    measured across them is measuring the attack, not the phonation.
    """
    interior = np.asarray(voiced, dtype=bool).copy()
    starts = np.flatnonzero(
        voiced & ~np.concatenate(([False], voiced[:-1]))
    )
    ends = np.flatnonzero(voiced & ~np.concatenate((voiced[1:], [False])))
    interior[starts] = False
    interior[ends] = False
    return interior


def synthesise(
    condition: EmotionCondition,
    seconds: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one utterance for ``condition``. Deterministic given ``rng``.

    The signal is a harmonic source (12 harmonics, amplitudes ``1/k**alpha``,
    seeded per-harmonic phases) whose instantaneous frequency follows a base
    pitch plus a slow contour, a terminal fall, a fast tremor and a per-syllable
    offset; a low-passed noise component at the breathiness ratio is added to
    the source; and the result is gated by a per-syllable envelope with a
    configurable attack, per-syllable loudness variation and per-frame shimmer.

    Every knob is one of the correlates named on `EmotionCondition` and nothing
    else is added, so the claim "the conditions differ along exactly the axes the
    features measure" is checkable by reading this function rather than by
    trusting the condition names.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")
    n_samples = int(round(seconds * sample_rate))
    if n_samples < 1:
        return np.zeros(0, dtype=np.float64)

    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    duration = n_samples / sample_rate
    hop = max(1, int(round(0.010 * sample_rate)))  # the front end's 10 ms hop

    # --- syllable layout --------------------------------------------------
    edges = _syllable_edges(n_samples, sample_rate, condition, rng)
    n_syllables = edges.size - 1
    syllable_of_sample = np.clip(
        np.searchsorted(edges, np.arange(n_samples), side="right") - 1,
        0,
        n_syllables - 1,
    )
    syllable_offsets = condition.jitter_st * rng.standard_normal(n_syllables)
    syllable_loudness = np.maximum(
        0.05, 1.0 + condition.amplitude_var * rng.standard_normal(n_syllables)
    )

    # --- pitch contour, in semitones relative to f0_base -------------------
    phase_offset = rng.uniform(0.0, 2.0 * np.pi)
    contour_st = (
        condition.f0_range_st / 2.0
        * np.sin(2.0 * np.pi * condition.tremor_hz * t + phase_offset)
        + condition.f0_slope_st * (t / duration - 0.5)
        + condition.f0_final_drop_st
        * np.clip((t - 0.72 * duration) / (0.28 * duration), 0.0, 1.0)
        + syllable_offsets[syllable_of_sample]
    )
    instantaneous_hz = condition.f0_base * np.exp2(contour_st / 12.0)
    phase = 2.0 * np.pi * np.cumsum(instantaneous_hz) / sample_rate

    # --- harmonic source, normalised to unit RMS --------------------------
    n_harmonics = 12
    harmonics = np.arange(1, n_harmonics + 1, dtype=np.float64)
    amplitudes = 1.0 / harmonics**condition.harmonic_alpha
    # Random per-harmonic phase: a zero-phase harmonic stack is a pulse train
    # with a high crest factor, which is not a voice.
    phases = rng.uniform(0.0, 2.0 * np.pi, n_harmonics)
    source = np.zeros(n_samples, dtype=np.float64)
    for k in range(n_harmonics):
        source += amplitudes[k] * np.sin(harmonics[k] * phase + phases[k])
    source /= max(float(np.sqrt(np.mean(source * source))), _EPS)

    # --- breathiness: turbulence noise in the source ----------------------
    if condition.breathiness > 0.0:
        source = source + condition.breathiness * _lowpass_noise(
            rng, n_samples, sample_rate
        )

    # --- envelope: syllables, attack, loudness, shimmer --------------------
    envelope = np.zeros(n_samples, dtype=np.float64)
    for j in range(n_syllables):
        start, stop = int(edges[j]), int(edges[j + 1])
        voiced = int(round((stop - start) * condition.duty))
        if voiced < 1:
            continue
        local = np.arange(voiced, dtype=np.float64) / sample_rate
        attack = np.clip(local / condition.attack_s, 0.0, 1.0)
        release = np.clip(
            (voiced / sample_rate - local) / condition.attack_s, 0.0, 1.0
        )
        envelope[start:start + voiced] = (
            condition.amplitude * syllable_loudness[j] * attack * release
        )

    n_frames = int(math.ceil(n_samples / hop))
    shimmer = np.maximum(
        0.0, 1.0 + condition.shimmer * rng.standard_normal(n_frames)
    )
    envelope *= np.repeat(shimmer, hop)[:n_samples]

    return source * envelope


# --------------------------------------------------------------------------
# New voice-quality features
#
# `voice.py` measures F0, energy, zero-crossing rate and the spectral shape.
# The published correlates of the five conditions need more than that, so these
# eleven features are extracted here. Every one is a *frame-level proxy* and
# none is the cycle-level measurement a phonetics tool reports:
#
# * ``hnr_db`` / ``hnr_db_std`` -- breathiness. Mean and spread of the Boersma
#   (1993) harmonics-to-noise ratio over voiced frames. Low is breathy or harsh.
#   ``hnr_db`` is 0.0 when nothing is voiced, so gate on ``voiced_ratio``.
# * ``shimmer`` -- steady-state amplitude perturbation: the standard deviation
#   of the RMS envelope with a 5-frame moving average removed, over the interior
#   frames of voiced runs, divided by the mean voiced RMS. Both exclusions were
#   forced by measurement (see `_quality_from_parts`). It separates a jagged
#   loudness contour from a smooth one carrying the same total variance, which
#   ``energy_std_voiced`` cannot do.
# * ``tremor_rate_hz`` -- dominant F0 modulation rate in 0.5-12 Hz, from the
#   largest spectral peak of the linearly detrended, gap-interpolated F0
#   contour. It recovers the generator's rate on excited, angry, afraid and
#   calm, and it **fails on crying**, whose breathy pitch track is noisy enough
#   that low-frequency estimation error outweighs the tremor (measured 0.56 Hz
#   against a 6.5 Hz generator setting). It is reported because the failure is
#   part of the result; it is not a feature to trust without checking the
#   pitch track's own confidence.
# * ``f0_slope_st_per_s`` -- linear pitch drift in semitones per second,
#   relative to the utterance's own mean F0. Semitones rather than Hz so the
#   same contour shape reads the same at 135 Hz and at 320 Hz.
# * ``f0_final_ratio`` -- mean F0 of the last third of voiced frames over the
#   first third, so a falling terminal contour reads below 1. It is 1.0 with
#   fewer than six voiced frames, which is a placeholder rather than a
#   measurement.
# * ``pause_rate`` / ``pause_fraction`` / ``pause_mean_s`` -- pause structure:
#   internal unvoiced runs of at least ``PAUSE_MIN_MS`` between two voiced
#   runs, per second, as a fraction of frames, and their mean duration. Leading
#   and trailing silence is not a pause.
# * ``onset_sharpness`` -- the largest single-hop rise of the 2-frame smoothed
#   RMS, as a fraction of the peak RMS. Near 0.47 is a gated onset, near 0.21 a
#   50 ms swell; the window/hop overlap caps the value well below 1. See
#   `_quality_from_parts`.
# * ``centroid_std`` -- spread of the spectral centroid over voiced frames: a
#   steady voice holds its spectral balance, a transient-heavy one moves it.
# --------------------------------------------------------------------------


def frame_hnr_db(frame: np.ndarray, sample_rate: int, f0_hz: float) -> float:
    """Harmonics-to-noise ratio of one frame, in dB, from its pitch period.

    Boersma (1993): with ``r`` the normalised autocorrelation at the pitch
    period, ``HNR = 10 log10(r / (1 - r))``. ``r`` is the fraction of the
    frame's energy that is periodic, so a clean voice is high and a breathy one
    is low.

    Two corrections are needed against `voice.autocorrelation_f0`'s biased
    correlation, and both are measured rather than assumed:

    * the biased estimator is ``(1 - tau/n) * true(tau)``, which at a 320 Hz
      pitch and a 25 ms window is a 20% attenuation. It is undone here
      (``* n / (n - tau)``); unlike in the peak *search*, where the taper is
      doing useful work suppressing long-lag noise peaks, this is a magnitude
      being reported.
    * the lag is fractional, so ``r`` is linearly interpolated between the two
      neighbouring integer lags.

    Returns 0.0 when there is no pitch to measure. That value is a placeholder,
    not a measurement: gate on the caller's ``voiced`` mask.
    """
    signal = np.asarray(frame, dtype=np.float64).reshape(-1)
    n = signal.size
    if n < 4 or not np.isfinite(f0_hz) or f0_hz <= 0.0:
        return 0.0
    centered = signal - signal.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 0.0:
        return 0.0
    lag = sample_rate / f0_hz
    if not 1.0 < lag < n - 1:
        return 0.0

    corr = np.correlate(centered, centered, mode="full")[n - 1 :] / energy
    r = float(np.interp(lag, np.arange(n), corr))
    r *= n / (n - lag)  # undo the biased estimator's linear taper
    r = float(np.clip(r, 0.0, _HNR_R_CLIP))
    return float(np.clip(10.0 * np.log10(r / (1.0 - r)), -HNR_MAX_DB, HNR_MAX_DB))


def dominant_modulation_hz(
    contour: np.ndarray,
    frame_rate: float,
    band: tuple[float, float] = TREMOR_BAND_HZ,
) -> float:
    """Dominant modulation frequency of a pitch contour, in Hz.

    The contour is linearly detrended, Hann-windowed and transformed; the
    largest magnitude in ``band`` is parabolically interpolated to a frequency.
    Counting zero crossings would be simpler and would be wrong here: the
    contour carries per-syllable offsets and frame noise, and every one of those
    is a crossing.

    Returns 0.0 when the contour is too short, is a pure trend (a linear ramp
    detrends to numerical dust, and calling that dust a 0.5 Hz modulation would
    be an artefact), or carries no energy in the band.
    """
    values = np.asarray(contour, dtype=np.float64).reshape(-1)
    if values.size < 8 or frame_rate <= 0.0:
        return 0.0
    values = values[np.isfinite(values)]
    if values.size < 8:
        return 0.0
    centred = values - values.mean()
    index = np.arange(centred.size, dtype=np.float64)
    slope = float(np.polyfit(index, centred, 1)[0])
    residual = centred - slope * (index - index.mean())
    # A trend with no modulation detrends to floating-point dust. The threshold
    # is relative to the contour's own spread so it is scale-free.
    if float(residual.std()) <= 1e-6 * max(float(values.std()), _EPS):
        return 0.0
    centred = residual
    windowed = centred * np.hanning(centred.size)
    magnitude = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(centred.size, d=1.0 / frame_rate)
    low, high = band
    inside = np.flatnonzero((freqs >= low) & (freqs <= high))
    if inside.size == 0 or float(magnitude[inside].max()) <= 0.0:
        return 0.0
    peak = int(inside[int(np.argmax(magnitude[inside]))])
    if 0 < peak < magnitude.size - 1:
        left, middle, right = (
            float(magnitude[peak - 1]),
            float(magnitude[peak]),
            float(magnitude[peak + 1]),
        )
        denominator = left - 2.0 * middle + right
        shift = 0.5 * (left - right) / denominator if denominator != 0.0 else 0.0
        return float(freqs[peak] + float(np.clip(shift, -0.5, 0.5))
                     * float(freqs[1] - freqs[0]))
    return float(freqs[peak])


def unvoiced_runs(voiced: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` frame ranges of contiguous unvoiced frames.

    Written with True sentinels on both ends so a run that touches either
    boundary is still reported as a run; the caller decides whether a leading
    or trailing silence is a pause (``_quality_from_parts`` says it is not).
    """
    voiced = np.asarray(voiced, dtype=bool)
    if voiced.size == 0:
        return []
    padded = np.concatenate(([True], ~voiced, [True]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    edges = np.concatenate(([0], transitions + 1, [padded.size]))
    runs: list[tuple[int, int]] = []
    # padded[0] is a True sentinel, so the runs alternate starting with
    # "unvoiced": the even-indexed runs are the ones wanted.
    for i in range(0, edges.size - 1, 2):
        start = max(int(edges[i]) - 1, 0)
        stop = min(int(edges[i + 1]) - 1, voiced.size)
        if stop > start:
            runs.append((start, stop))
    return runs


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Boxcar smoothing with edge replication, so the length is preserved."""
    if width <= 1 or x.size == 0:
        return x
    width = min(width, x.size)
    half = width // 2
    padded = np.concatenate((np.full(half, x[0]), x, np.full(half, x[-1])))
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")[: x.size]


# The feature set the classifier is trained on: every scalar descriptor
# `voice.affect_descriptors` returns except the two design constants
# (`n_frames` and `duration_s`, which are identical for every utterance here
# and would add columns that cannot carry information), plus every
# voice-quality feature below. Derived from `affect_descriptors` rather than
# typed out, so the two cannot drift apart.
_VOICE_DESCRIPTOR_NAMES: tuple[str, ...] = tuple(
    name
    for name in affect_descriptors(frame_features(np.zeros(1600), 16_000))
    if name not in ("n_frames", "duration_s")
)

EMOTION_FEATURES: tuple[str, ...] = _VOICE_DESCRIPTOR_NAMES + (
    "hnr_db",
    "hnr_db_std",
    "shimmer",
    "tremor_rate_hz",
    "f0_slope_st_per_s",
    "f0_final_ratio",
    "pause_rate",
    "pause_fraction",
    "pause_mean_s",
    "onset_sharpness",
    "centroid_std",
)

# The F0 *level and contour* family, for the ablation that asks whether the
# model is doing anything other than reading pitch height. `jitter`,
# `jitter_relative` and `tremor_rate_hz` are deliberately NOT members: they are
# pitch *dynamics*, and the wider ablation below is reported separately so the
# narrower claim is not overstated in either direction.
F0_FAMILY: tuple[str, ...] = (
    "f0_mean",
    "f0_std",
    "f0_range",
    "f0_slope_st_per_s",
    "f0_final_ratio",
)

# Everything derived from the pitch track at all.
PITCH_DERIVED_FAMILY: tuple[str, ...] = F0_FAMILY + (
    "jitter",
    "jitter_relative",
    "tremor_rate_hz",
)

# The 26 features partitioned by what they measure, so the ablation table can
# ask "is this family sufficient on its own?" as well as "is it necessary?".
# `tests/test_emotion.py` asserts this is a partition of `EMOTION_FEATURES`, so
# a feature added to the set without a family fails the suite rather than being
# silently un-ablatable.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "f0_level_and_contour": F0_FAMILY,
    "pitch_dynamics": ("jitter", "jitter_relative", "tremor_rate_hz"),
    "voice_quality": ("hnr_db", "hnr_db_std", "shimmer", "flatness_mean"),
    "energy_dynamics": (
        "energy_mean",
        "energy_std",
        "energy_mean_voiced",
        "energy_std_voiced",
        "energy_flux_mean",
        "onset_sharpness",
    ),
    "rhythm_and_pauses": (
        "speaking_rate",
        "voiced_ratio",
        "pause_rate",
        "pause_fraction",
        "pause_mean_s",
    ),
    "spectral_shape": ("zcr_mean", "centroid_mean", "centroid_std"),
}


def emotion_descriptors(
    waveform: np.ndarray, sample_rate: int
) -> dict[str, float]:
    """Every descriptor the classifier consumes, for one waveform.

    The single entry point that needs the waveform as well as the frame
    features, because HNR is recomputed per voiced frame from the raw samples.
    The returned dict has exactly ``EMOTION_FEATURES`` as its keys, in order.
    """
    signal = np.asarray(waveform, dtype=np.float64)
    features = frame_features(signal, sample_rate)
    summary = dict(affect_descriptors(features))
    summary.pop("n_frames", None)
    summary.pop("duration_s", None)

    voiced = features.voiced
    if np.any(voiced):
        frames = frame_signal(signal, features.frame_length, features.hop_length)
        hnr = np.array([
            frame_hnr_db(np.asarray(frames[i], dtype=np.float64), sample_rate,
                         features.f0_hz[i])
            for i in np.flatnonzero(voiced)
        ])
    else:
        hnr = np.zeros(0, dtype=np.float64)

    summary.update(_quality_from_parts(features, hnr))
    return {name: float(summary[name]) for name in EMOTION_FEATURES}


def _quality_from_parts(
    features: VoiceFeatures, hnr: np.ndarray
) -> dict[str, float]:
    """The voice-quality features, given the per-voiced-frame HNR already computed.

    Every feature here is a frame-level proxy and is named as one in the module
    docstring. Each is finite for every input, including silence: the
    placeholders (``hnr_db`` 0.0, ``f0_final_ratio`` 1.0) are flagged by
    ``voiced_ratio == 0.0`` rather than by a nan that would propagate into the
    classifier's arithmetic.
    """
    rms = features.rms
    voiced = features.voiced
    f0 = features.f0_hz
    hop_s = features.hop_length / float(features.sample_rate)
    duration = features.duration_s
    n_voiced = int(np.count_nonzero(voiced))

    hnr_db = float(hnr.mean()) if hnr.size else 0.0
    hnr_db_std = float(hnr.std()) if hnr.size else 0.0

    # --- shimmer -----------------------------------------------------------
    # Steady-state amplitude perturbation: the envelope with a 5-frame moving
    # average removed, so a slow swell is not counted as shimmer, measured over
    # the interior frames of voiced runs only, so the onset and offset ramps are
    # not counted either. Both exclusions were forced by measurement --
    # without them this read 0.215 for `afraid` against 0.220 for `crying`,
    # whose generator shimmer ratios are 0.16 and 0.35.
    shimmer = 0.0
    interior = _interior_voiced(voiced)
    if n_voiced >= 2:
        residual = rms - _moving_average(rms, 5)
        if np.any(interior):
            shimmer = float(
                residual[interior].std() / max(float(rms[voiced].mean()), _EPS)
            )

    # --- pitch contour -----------------------------------------------------
    f0_slope_st_per_s = 0.0
    f0_final_ratio = 1.0
    tremor_rate_hz = 0.0
    if n_voiced >= 6:
        voiced_f0 = f0[voiced]
        voiced_times = features.times_s[voiced]
        semitones = 12.0 * np.log2(voiced_f0 / max(float(voiced_f0.mean()), _EPS))
        f0_slope_st_per_s = float(np.polyfit(voiced_times, semitones, 1)[0])

        span = float(voiced_times[-1] - voiced_times[0])
        if span > 0.0:
            third = span / 3.0
            first = semitones[voiced_times <= voiced_times[0] + third]
            last = semitones[voiced_times >= voiced_times[-1] - third]
            if first.size >= 2 and last.size >= 2:
                f0_final_ratio = float(
                    np.exp2((last.mean() - first.mean()) / 12.0)
                )

        # Interpolate the contour onto the frame grid across the unvoiced gaps
        # so the modulation estimator sees one evenly sampled sequence.
        voiced_index = np.flatnonzero(voiced)
        grid = np.arange(voiced_index[0], voiced_index[-1] + 1)
        contour = np.interp(grid, voiced_index, semitones)
        tremor_rate_hz = dominant_modulation_hz(contour, 1.0 / hop_s)

    # --- pause structure ---------------------------------------------------
    pause_rate = 0.0
    pause_fraction = 0.0
    pause_mean_s = 0.0
    min_frames = max(1, int(round((PAUSE_MIN_MS / 1000.0) / hop_s)))
    voiced_index = np.flatnonzero(voiced)
    if voiced_index.size and features.n_frames:
        first_voiced = int(voiced_index[0])
        last_voiced = int(voiced_index[-1])
        pauses = [
            (start, stop)
            for start, stop in unvoiced_runs(voiced)
            if stop - start >= min_frames
            and start > first_voiced
            and stop <= last_voiced
        ]
        pause_fraction = (
            sum(stop - start for start, stop in pauses) / features.n_frames
        )
        if duration > 0.0:
            pause_rate = len(pauses) / duration
        if pauses:
            lengths = np.array(
                [stop - start for start, stop in pauses], dtype=np.float64
            )
            pause_mean_s = float(lengths.mean()) * hop_s

    # --- onset sharpness ---------------------------------------------------
    # The largest single-hop rise of the lightly smoothed RMS, as a fraction of
    # the peak RMS. A 25 ms analysis window against a 10 ms hop overlaps by
    # 15 ms, so a perfectly gated onset still rises over about three frames and
    # the achievable value is bounded well below 1 (measured: 0.47 for a 5 ms
    # attack, 0.21 for a 50 ms one). The width-2 smoothing is there so
    # per-frame shimmer noise does not read as an onset; width 3 was measured
    # and it compressed the range without changing the ordering.
    onset_sharpness = 0.0
    if rms.size >= 2:
        smoothed = _moving_average(rms, 2)
        peak = float(smoothed.max())
        if peak > _EPS:
            onset_sharpness = float(
                np.maximum(0.0, np.diff(smoothed)).max() / peak
            )

    # --- spectral movement -------------------------------------------------
    centroid_std = float(features.centroid_hz[voiced].std()) if n_voiced else 0.0

    return {
        "hnr_db": hnr_db,
        "hnr_db_std": hnr_db_std,
        "shimmer": shimmer,
        "tremor_rate_hz": tremor_rate_hz,
        "f0_slope_st_per_s": f0_slope_st_per_s,
        "f0_final_ratio": f0_final_ratio,
        "pause_rate": pause_rate,
        "pause_fraction": pause_fraction,
        "pause_mean_s": pause_mean_s,
        "onset_sharpness": onset_sharpness,
        "centroid_std": centroid_std,
    }


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Utterance-level feature matrix and labels.

    ``features`` is ``(n_utterances, len(EMOTION_FEATURES))`` and ``labels`` is
    the condition name per row. The unit of the split is the row, i.e. the
    utterance: features are reduced to one vector per utterance *before* the
    split, so no frame of a test utterance is ever in training.
    """

    features: np.ndarray
    labels: np.ndarray
    conditions: tuple[str, ...]

    @property
    def n_utterances(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


def build_dataset(
    *,
    conditions: tuple[EmotionCondition, ...] = CONDITIONS,
    utterances: int = 40,
    seconds: float = 2.0,
    sample_rate: int = 16_000,
    seed: int = 0,
) -> Dataset:
    """Synthesise ``utterances`` per condition and reduce each to one vector.

    One RNG stream, drawn in condition order, so the dataset is a deterministic
    function of ``seed`` and the argument list.
    """
    if utterances < 1:
        raise ValueError("need at least one utterance per condition")
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[str] = []
    for condition in conditions:
        for _ in range(utterances):
            waveform = synthesise(condition, seconds, sample_rate, rng)
            summary = emotion_descriptors(waveform, sample_rate)
            rows.append([summary[name] for name in EMOTION_FEATURES])
            labels.append(condition.name)
    return Dataset(
        features=np.array(rows, dtype=np.float64),
        labels=np.array(labels, dtype=object),
        conditions=tuple(c.name for c in conditions),
    )


def grouped_means(dataset: Dataset) -> dict[str, dict[str, float]]:
    """Mean descriptor vector per condition, straight from the dataset rows.

    Used for the README table, so the numbers rendered next to the generator
    parameters are the same rows the classifier was trained and tested on
    rather than a second independent draw.
    """
    return {
        label: {
            name: float(dataset.features[dataset.labels == label, i].mean())
            for i, name in enumerate(EMOTION_FEATURES)
        }
        for label in dataset.conditions
    }


def correlate_checks(means: dict[str, dict[str, float]]) -> dict[str, bool]:
    """The documented correlate of each condition, checked against its measurement.

    These are assertions about the generator, not about emotion: each says the
    signal this module produced shows the acoustic profile its citation names.
    A failure here means the condition does not implement the correlate it
    claims, which would make every accuracy below a statement about a condition
    that is not the one advertised.
    """
    names = list(means)
    return {
        "crying: highest jitter":
            means["crying"]["jitter"] == max(means[n]["jitter"] for n in names),
        "crying: highest shimmer":
            means["crying"]["shimmer"] == max(means[n]["shimmer"] for n in names),
        "crying: lowest HNR":
            means["crying"]["hnr_db"] == min(means[n]["hnr_db"] for n in names),
        "crying: lowest final/initial F0 ratio":
            means["crying"]["f0_final_ratio"]
            == min(means[n]["f0_final_ratio"] for n in names),
        "crying: lower energy than angry":
            means["crying"]["energy_mean"] < means["angry"]["energy_mean"],
        "crying: more pauses than excited":
            means["crying"]["pause_rate"] > means["excited"]["pause_rate"],
        "excited: widest F0 range of the low-jitter conditions":
            means["excited"]["f0_range"] > 5.0 * means["calm"]["f0_range"],
        "excited: highest voiced-frame energy spread":
            means["excited"]["energy_std_voiced"]
            == max(means[n]["energy_std_voiced"] for n in names),
        "excited: lower jitter than crying":
            means["excited"]["jitter"] < means["crying"]["jitter"],
        "angry: highest energy":
            means["angry"]["energy_mean"] == max(means[n]["energy_mean"] for n in names),
        "angry: sharpest onsets":
            means["angry"]["onset_sharpness"]
            == max(means[n]["onset_sharpness"] for n in names),
        "angry: higher centroid than calm":
            means["angry"]["centroid_mean"] > means["calm"]["centroid_mean"],
        "afraid: higher F0 than calm":
            means["afraid"]["f0_mean"] > 2.0 * means["calm"]["f0_mean"],
        "afraid: more jitter than excited":
            means["afraid"]["jitter"] > 2.0 * means["excited"]["jitter"],
        "afraid: softer onsets than angry":
            means["afraid"]["onset_sharpness"] < means["angry"]["onset_sharpness"],
        "calm: lowest F0 spread":
            means["calm"]["f0_std"] == min(means[n]["f0_std"] for n in names),
        "calm: lowest voiced-frame energy spread":
            means["calm"]["energy_std_voiced"]
            == min(means[n]["energy_std_voiced"] for n in names),
    }


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """Row indices for a stratified, utterance-level split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split(
    labels: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Split:
    """Split row indices within each condition, so every class is in every split.

    Rounding is done per class with an explicit rule, so the split sizes are
    exact and the test accuracy is over a number the caller can count. At least
    one row of every class reaches validation and test: otherwise a class is
    unmeasurable and the confusion matrix has a silently empty row.
    """
    train_fraction, val_fraction, _ = fractions
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("split fractions must be a probability triple")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train + validation must leave a test set")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        if members.size < 3:
            raise ValueError(f"class {label!r} has too few rows to split")
        shuffled = members[rng.permutation(members.size)]
        n_train = int(np.floor(train_fraction * members.size))
        n_val = int(np.floor(val_fraction * members.size))
        n_train = min(n_train, members.size - 2)
        n_val = max(1, min(n_val, members.size - n_train - 1))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
    return Split(
        train=np.array(sorted(train), dtype=np.int64),
        val=np.array(sorted(val), dtype=np.int64),
        test=np.array(sorted(test), dtype=np.int64),
    )


@contextmanager
def _one_thread():
    """Pin torch to one thread for the duration of a training run.

    Full-batch reductions on a small MLP were measured to be reproducible on
    this machine with the default thread count, but "reproducible on the machine
    it was written on" is not the claim the tests make. One thread removes the
    reduction-order question rather than relying on it.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _standardise(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Z-score using the *training* rows only, then apply to the others.

    Refitting the scaling on held-out rows would leak their mean and spread into
    the model; it is the cheapest way to make a test number look better than it
    is.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return ((train - mean) / std,) + tuple((other - mean) / std for other in others)


@dataclass
class TrainedClassifier:
    """A trained MLP plus the standardisation it was trained under.

    ``selected_epoch`` is the validation-selected snapshot; ``test_accuracy`` is
    measured once, from that snapshot, and never used to choose anything.
    """

    classes: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    columns: np.ndarray
    selected_epoch: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    history: list[dict[str, float]]
    architecture: str

    def logits(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)[:, self.columns]
        x = (matrix - self.mean) / self.std
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ weight + bias, 0.0)
        return x @ self.weights[-1] + self.biases[-1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.logits(features), axis=1)
        return np.array([self.classes[i] for i in indices], dtype=object)

    def nll(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Mean cross-entropy of ``labels`` under the model, in nats."""
        logits = self.logits(features)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )
        index = np.array([self.classes.index(label) for label in labels])
        return float(-log_probs[np.arange(len(labels)), index].mean())

    def as_dict(self) -> dict:
        return {
            "classes": list(self.classes),
            "architecture": self.architecture,
            "selected_epoch": self.selected_epoch,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
        }


def train_classifier(
    dataset: Dataset,
    *,
    split: Split,
    columns: np.ndarray | None = None,
    seed: int = 0,
    hidden: int = 32,
    steps: int = 1500,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    evaluate_every: int = 10,
) -> TrainedClassifier:
    """Train an MLP on utterance feature vectors; select on validation; report test.

    The loop is full-batch Adam with a fixed seed: deterministic, and small
    enough here that minibatching would add a sampler whose RNG is one more
    thing to get wrong. Model selection keeps the parameters with the best
    **validation** accuracy, and **test** accuracy is measured from that
    snapshot, so the test set is read exactly once and is never used to choose
    an epoch.

    ``columns`` selects a feature subset for the ablations; the full feature
    order is ``EMOTION_FEATURES``.
    """
    features = dataset.features if columns is None else dataset.features[:, columns]
    selected_columns = (
        np.arange(dataset.features.shape[1], dtype=np.int64)
        if columns is None
        else np.asarray(columns, dtype=np.int64)
    )
    labels = list(dataset.labels)
    classes = sorted(set(labels))
    train_rows, val_rows, test_rows = split.train, split.val, split.test

    x_train, x_val, x_test = _standardise(
        features[train_rows], features[val_rows], features[test_rows]
    )
    y_train = np.array([classes.index(labels[i]) for i in train_rows])
    y_val = np.array([classes.index(labels[i]) for i in val_rows])
    y_test = np.array([classes.index(labels[i]) for i in test_rows])

    with _one_thread():
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(classes)),
        )
        # Double precision: the MLP is tiny, so there is no cost, and it removes
        # one more place a float32 reduction order can differ between runs.
        model = model.double()
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        x_val_t = torch.from_numpy(x_val)
        x_test_t = torch.from_numpy(x_test)

        best_val = -1.0
        best_epoch = -1
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        history: list[dict[str, float]] = []
        for epoch in range(steps):
            model.train()
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_train_t), y_train_t)
            loss.backward()
            optimiser.step()
            if epoch % evaluate_every == 0 or epoch == steps - 1:
                with torch.no_grad():
                    model.eval()
                    val_logits = model(x_val_t).numpy()
                val_accuracy = float((val_logits.argmax(axis=1) == y_val).mean())
                history.append({
                    "epoch": float(epoch),
                    "loss": float(loss.item()),
                    "val_accuracy": val_accuracy,
                })
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }

        model.load_state_dict(best_state)
        with torch.no_grad():
            model.eval()
            train_logits = model(x_train_t).numpy()
            test_logits = model(x_test_t).numpy()
        train_accuracy = float((train_logits.argmax(axis=1) == y_train).mean())
        test_accuracy = float((test_logits.argmax(axis=1) == y_test).mean())
        weights = [
            model[0].weight.detach().numpy().T.copy(),
            model[2].weight.detach().numpy().T.copy(),
        ]
        biases = [
            model[0].bias.detach().numpy().copy(),
            model[2].bias.detach().numpy().copy(),
        ]

    return TrainedClassifier(
        classes=classes,
        mean=features[train_rows].mean(axis=0),
        std=np.where(
            features[train_rows].std(axis=0) > 0.0,
            features[train_rows].std(axis=0),
            1.0,
        ),
        weights=weights,
        biases=biases,
        columns=selected_columns,
        selected_epoch=best_epoch,
        train_accuracy=train_accuracy,
        val_accuracy=best_val,
        test_accuracy=test_accuracy,
        history=history,
        architecture=(
            f"MLP {x_train.shape[1]}->{hidden}->{len(classes)}, ReLU, "
            f"full-batch Adam lr={lr}, weight_decay={weight_decay}, "
            f"{steps} steps, snapshot chosen on validation accuracy"
        ),
    )


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, classes: list[str]
) -> list[list[int]]:
    """Rows are true classes, columns predicted, in ``classes`` order."""
    index = {label: i for i, label in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for truth, predicted in zip(labels, predictions):
        matrix[index[truth]][index[predicted]] += 1
    return matrix


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.mean([p == t for p, t in zip(predictions, labels)]))


def z_above_chance(score: float, n: int, chance: float) -> float:
    """Normal-approximation z of an accuracy against chance.

    Reported next to every accuracy for the same reason `voice_affect.py`
    reports it: at n=40 an accuracy of 0.55 against chance 0.50 is 0.6 sigma,
    which is noise, and a bare percentage hides that.
    """
    if n <= 0 or not 0.0 < chance < 1.0:
        return 0.0
    return (score - chance) / float(np.sqrt(chance * (1.0 - chance) / n))


def labels_at(labels: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.array([labels[i] for i in rows], dtype=object)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def shuffle_label_control(
    dataset: Dataset,
    split: Split,
    *,
    rounds: int = 20,
    control_seed: int = 0,
    **train_kwargs,
) -> dict[str, float]:
    """Train on permuted labels, evaluate on the real test rows.

    The permutation is applied once to the whole label vector, so the training
    and validation label sets are independently shuffled and the model cannot
    recover the truth from either. If this control does not fall to chance the
    pipeline is leaking labels, not learning features.

    ``control_seed`` seeds the permutations; the model seed for round ``i`` is
    the caller's ``train_kwargs['seed']`` plus ``i``, so the control uses the
    same recipe as the real run rather than an independent one.
    """
    rng = np.random.default_rng(control_seed)
    scores: list[float] = []
    for round_index in range(rounds):
        shuffled = Dataset(
            features=dataset.features,
            labels=np.array(rng.permutation(dataset.labels), dtype=object),
            conditions=dataset.conditions,
        )
        classifier = train_classifier(
            shuffled, split=split,
            seed=int(train_kwargs.get("seed", 0)) + round_index,
            **{k: v for k, v in train_kwargs.items() if k != "seed"},
        )
        predictions = classifier.predict(dataset.features[split.test])
        scores.append(
            accuracy(predictions, labels_at(dataset.labels, split.test))
        )
    values = np.array(scores)
    return {
        "rounds": float(rounds),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def random_feature_control(
    dataset: Dataset,
    split: Split,
    *,
    rounds: int = 5,
    control_seed: int = 0,
    **train_kwargs,
) -> dict[str, float]:
    """Train and test on Gaussian noise of the same shape as the real matrix.

    The positive control for "the learning is real": identical architecture,
    identical split, identical number of columns, so any accuracy above chance
    would have to come from the labels being learnable from those columns.
    Noise columns have none of the structure, so this pins the floor.
    """
    rng = np.random.default_rng(control_seed + 991)
    scores: list[float] = []
    for round_index in range(rounds):
        noise = rng.standard_normal(dataset.features.shape)
        noisy = Dataset(
            features=noise, labels=dataset.labels, conditions=dataset.conditions
        )
        classifier = train_classifier(
            noisy, split=split,
            seed=int(train_kwargs.get("seed", 0)) + round_index,
            **{k: v for k, v in train_kwargs.items() if k != "seed"},
        )
        predictions = classifier.predict(noise[split.test])
        scores.append(
            accuracy(predictions, labels_at(dataset.labels, split.test))
        )
    values = np.array(scores)
    return {
        "rounds": float(rounds),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------


def _columns_for(
    removed: tuple[str, ...] = (), kept: tuple[str, ...] | None = None
) -> np.ndarray:
    """Column indices for a feature subset, by name.

    An unknown name is an error rather than a silently ignored no-op: an
    ablation that removed nothing would be published as evidence that nothing
    was needed.
    """
    names = list(EMOTION_FEATURES)
    for name in tuple(removed) + tuple(kept or ()):
        if name not in names:
            raise ValueError(f"unknown feature {name!r}")
    if kept is not None:
        selected = [names.index(name) for name in kept]
    else:
        selected = [i for i, name in enumerate(names) if name not in removed]
    if not selected:
        raise ValueError("ablation would leave no features")
    return np.array(selected, dtype=np.int64)


def ablation(
    dataset: Dataset,
    split: Split,
    *,
    removed: tuple[str, ...] = (),
    kept: tuple[str, ...] | None = None,
    seed: int = 0,
    **train_kwargs,
) -> dict[str, object]:
    """Retrain with a feature family removed (or only one family kept)."""
    columns = _columns_for(removed, kept)
    names = [EMOTION_FEATURES[i] for i in columns]
    classifier = train_classifier(
        dataset, split=split, columns=columns, seed=seed, **train_kwargs
    )
    return {
        "features": names,
        "n_features": len(names),
        "test_accuracy": classifier.test_accuracy,
        "val_accuracy": classifier.val_accuracy,
        "train_accuracy": classifier.train_accuracy,
        "selected_epoch": classifier.selected_epoch,
    }


def binary_cry_vs_excited(
    dataset: Dataset,
    *,
    seed: int = 0,
    columns: np.ndarray | None = None,
    steps: int = 1200,
    **train_kwargs,
) -> dict[str, object]:
    """Crying against excited alone, on a stratified utterance split.

    The task the README's false claim about telling crying from excitement is
    about. Returns the test confusion, the per-class recall, and the accuracy
    reached when the feature set is restricted to ``columns``.
    """
    keep = np.array(
        [i for i, label in enumerate(dataset.labels)
         if label in ("crying", "excited")],
        dtype=np.int64,
    )
    subset = Dataset(
        features=dataset.features[keep],
        labels=np.array([dataset.labels[i] for i in keep], dtype=object),
        conditions=("crying", "excited"),
    )
    split = stratified_split(subset.labels, seed=seed)
    classifier = train_classifier(
        subset, split=split, seed=seed, columns=columns, steps=steps,
        **train_kwargs,
    )
    predictions = classifier.predict(subset.features[split.test])
    truth = labels_at(subset.labels, split.test)
    classes = ["crying", "excited"]
    matrix = confusion_matrix(predictions, truth, classes)
    recall = {
        classes[i]: (
            matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0.0
        )
        for i in range(len(classes))
    }
    return {
        "test_accuracy": classifier.test_accuracy,
        "val_accuracy": classifier.val_accuracy,
        "chance": 0.5,
        "n_test": int(sum(sum(row) for row in matrix)),
        "confusion": matrix,
        "classes": classes,
        "recall": recall,
        "selected_epoch": classifier.selected_epoch,
        "n_features": int(
            subset.features.shape[1] if columns is None else len(columns)
        ),
    }


def per_feature_cry_vs_excited(
    dataset: Dataset, *, seed: int = 0, **train_kwargs
) -> dict[str, dict[str, float]]:
    """Each feature alone, and each feature removed, on crying versus excited.

    The first number answers "which single measurement separates them"; the
    second answers "how much does the model lose without it", which is the
    ablation the README requires next to the confusion matrix instead of an
    assertion about which feature is doing the work.
    """
    names = list(EMOTION_FEATURES)
    full = binary_cry_vs_excited(dataset, seed=seed, **train_kwargs)
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        one = binary_cry_vs_excited(
            dataset, seed=seed, columns=np.array([index], dtype=np.int64),
            **train_kwargs,
        )
        without = binary_cry_vs_excited(
            dataset, seed=seed,
            columns=np.array(
                [i for i in range(len(names)) if i != index], dtype=np.int64
            ),
            **train_kwargs,
        )
        result[name] = {
            "single_accuracy": float(one["test_accuracy"]),
            "without_accuracy": float(without["test_accuracy"]),
            "drop": float(full["test_accuracy"] - without["test_accuracy"]),
        }
    return result


def greedy_forward_selection(
    dataset: Dataset,
    *,
    target: tuple[str, str] = ("crying", "excited"),
    feature_names: tuple[str, ...] | None = None,
    seed: int = 0,
    max_features: int = 6,
    steps: int = 1200,
    **train_kwargs,
) -> dict[str, object]:
    """How few features does the two-condition task need? Selected on validation.

    Written because the leave-one-feature-out ablation came back with a drop of
    exactly 0.000 for all 26 features: removing the best single feature changed
    nothing, because twenty other features each separate the pair on their own.
    "Which feature is doing it" therefore has no single answer, and the honest
    way to report that is to find the smallest set that reaches the accuracy
    instead of naming one feature and implying it is load-bearing.

    Selection is on validation accuracy only; each round records the held-out
    test accuracy of the growing set, and the loop stops as soon as validation
    is perfect, the set stops improving, or ``max_features`` is reached. Ties
    are broken by ``EMOTION_FEATURES`` order so the result is deterministic.
    """
    if len(target) != 2:
        raise ValueError("greedy selection is written for one condition pair")
    keep = np.array(
        [i for i, label in enumerate(dataset.labels) if label in target],
        dtype=np.int64,
    )
    subset = Dataset(
        features=dataset.features[keep],
        labels=np.array([dataset.labels[i] for i in keep], dtype=object),
        conditions=target,
    )
    split = stratified_split(subset.labels, seed=seed)
    if feature_names is not None:
        names = list(feature_names)
    elif dataset.n_features == len(EMOTION_FEATURES):
        names = list(EMOTION_FEATURES)
    else:
        names = [f"feature_{i}" for i in range(dataset.n_features)]
    if len(names) != dataset.n_features:
        raise ValueError(
            f"{len(names)} feature names for {dataset.n_features} columns"
        )
    chosen: list[int] = []
    remaining = list(range(len(names)))
    history: list[dict[str, object]] = []
    stopped = "max_features"
    while remaining and len(chosen) < max_features:
        best_index = None
        best_val = -1.0
        best_test = 0.0
        for index in remaining:
            classifier = train_classifier(
                subset, split=split, seed=seed,
                columns=np.array(chosen + [index], dtype=np.int64),
                steps=steps, **train_kwargs,
            )
            if classifier.val_accuracy > best_val:
                best_val = classifier.val_accuracy
                best_test = classifier.test_accuracy
                best_index = index
        if best_index is None:
            stopped = "no_candidate"
            break
        chosen.append(best_index)
        remaining.remove(best_index)
        history.append({
            "round": len(chosen),
            "added": names[best_index],
            "val_accuracy": float(best_val),
            "test_accuracy": float(best_test),
            "features": [names[i] for i in chosen],
        })
        if best_val >= 1.0:
            stopped = "validation_perfect"
            break
        if len(history) >= 2 and best_val <= history[-2]["val_accuracy"]:
            stopped = "no_improvement"
            break
    if stopped == "max_features" and not remaining:
        stopped = "exhausted"
    return {
        "target": list(target),
        "chance": 0.5,
        "history": history,
        "n_features_needed": len(chosen),
        "features": [names[i] for i in chosen],
        "final_val_accuracy": (
            float(history[-1]["val_accuracy"]) if history else 0.0
        ),
        "final_test_accuracy": (
            float(history[-1]["test_accuracy"]) if history else 0.0
        ),
        "stopped_because": stopped,
    }


# --------------------------------------------------------------------------
# Cross-condition generalisation
# --------------------------------------------------------------------------


def leave_one_condition_out(
    dataset: Dataset,
    *,
    seed: int = 0,
    hidden: int = 32,
    steps: int = 1500,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    evaluate_every: int = 10,
) -> dict[str, object]:
    """Train on four conditions, then look at the fifth.

    Accuracy is *undefined* here and saying so is the result: the held-out
    condition is not in the label space the model was trained on, so every
    prediction is wrong by construction and an "accuracy" of 0.000 would be a
    restatement of that fact rather than a measurement.

    What is measured instead is what the model *does* with a condition it has
    never seen:

    * ``assignment`` -- over the four training classes, the fraction of
      held-out utterances each one claims. One class claiming nearly all of
      them is collapse, and it is the honest picture of a model with no
      representation for the unseen condition.
    * ``assigned_class_nll`` -- the mean cross-entropy, in nats, of the class
      the model did predict. Low means it is confidently extrapolating a
      boundary it never saw evidence for, which is worse than being surprised.
    * ``distance_to_nearest_training_centroid`` -- in units of the mean
      within-training-class spread. Above 1 means the held-out condition sits
      farther from everything the model knows than its own classes sit from
      each other.
    * ``nearest_centroid_assignment`` -- the same assignment computed by the
      untrained z-scored nearest-centroid rule from `voice_affect.py`. It is
      here so the collapse can be attributed: if the MLP and the untrained
      nearest-centroid rule put the unseen condition in the same place, the
      collapse is a property of the feature space, not of the trained model.
    """
    names = list(dataset.conditions)
    per_condition: dict[str, dict[str, object]] = {}
    dominant_fractions: list[float] = []
    training_nlls: list[float] = []
    for held_out in names:
        keep = np.array(
            [i for i, label in enumerate(dataset.labels) if label != held_out],
            dtype=np.int64,
        )
        train_dataset = Dataset(
            features=dataset.features[keep],
            labels=np.array([dataset.labels[i] for i in keep], dtype=object),
            conditions=tuple(n for n in names if n != held_out),
        )
        split = stratified_split(train_dataset.labels, seed=seed)
        classifier = train_classifier(
            train_dataset, split=split, seed=seed, hidden=hidden, steps=steps,
            lr=lr, weight_decay=weight_decay, evaluate_every=evaluate_every,
        )
        rows = np.flatnonzero(dataset.labels == held_out)
        predictions = classifier.predict(dataset.features[rows])
        counts = {name: 0 for name in train_dataset.conditions}
        for predicted in predictions:
            counts[str(predicted)] += 1
        assignment = {name: counts[name] / len(predictions) for name in counts}
        dominant = max(assignment, key=assignment.get)
        dominant_fractions.append(assignment[dominant])

        # The held-out label is not in the label space, so the surprise is
        # measured as the cross-entropy of the class the model *did* assign:
        # how confident is it in an answer it has no basis for?
        assigned_nll = classifier.nll(dataset.features[rows], predictions)
        train_rows = split.test
        training_nlls.append(
            classifier.nll(
                train_dataset.features[train_rows],
                labels_at(train_dataset.labels, train_rows),
            )
        )

        train_features = train_dataset.features
        mean = train_features.mean(axis=0)
        std = np.where(
            train_features.std(axis=0) > 0.0, train_features.std(axis=0), 1.0
        )
        z_train = (train_features - mean) / std
        z_held = (dataset.features[rows] - mean) / std
        centroids = np.stack([
            z_train[train_dataset.labels == name].mean(axis=0)
            for name in train_dataset.conditions
        ])
        within = float(np.mean([
            np.linalg.norm(
                z_train[train_dataset.labels == name] - centroids[i], axis=1
            ).mean()
            for i, name in enumerate(train_dataset.conditions)
        ]))
        held_distances = np.linalg.norm(
            z_held[:, None, :] - centroids[None, :, :], axis=2
        )
        nearest = float(held_distances.min(axis=1).mean() / max(within, _EPS))

        # The untrained reference: which training centroid each held-out
        # utterance is nearest to.
        nearest_index = held_distances.argmin(axis=1)
        centroid_counts = {name: 0 for name in train_dataset.conditions}
        for index in nearest_index:
            centroid_counts[train_dataset.conditions[int(index)]] += 1
        nearest_assignment = {
            name: centroid_counts[name] / len(rows)
            for name in centroid_counts
        }
        nearest_dominant = max(nearest_assignment, key=nearest_assignment.get)

        per_condition[held_out] = {
            "n_held_out": int(len(rows)),
            "assignment": assignment,
            "dominant_assignment": dominant,
            "dominant_fraction": float(assignment[dominant]),
            "nearest_centroid_assignment": nearest_assignment,
            "nearest_centroid_dominant": nearest_dominant,
            "assigned_class_nll": float(assigned_nll),
            "distance_to_nearest_training_centroid": nearest,
            "within_class_spread": within,
            "training_test_accuracy": classifier.test_accuracy,
        }
    agreements = [
        per_condition[name]["dominant_assignment"]
        == per_condition[name]["nearest_centroid_dominant"]
        for name in names
    ]
    return {
        "per_condition": per_condition,
        "mean_dominant_fraction": float(np.mean(dominant_fractions)),
        "mean_training_test_nll": float(np.mean(training_nlls)),
        "chance_for_one_training_class": 1.0 / len(names),
        "mlp_agrees_with_nearest_centroid": float(np.mean(agreements)),
        "note": (
            "the held-out label is not in the trained label space, so accuracy "
            "is 0 by construction; the assignment distribution, the "
            "cross-entropy of the assigned class and the distance to the "
            "nearest training centroid are what carry the result"
        ),
    }


# --------------------------------------------------------------------------
# Baselines that need no training
# --------------------------------------------------------------------------


def nearest_centroid_accuracy(
    dataset: Dataset, split: Split, columns: np.ndarray | None = None
) -> dict[str, object]:
    """The previous increment's classifier: z-scored nearest centroid.

    Included so the trained model can be compared against the untrained one on
    the same split, rather than the README implying that training is what made
    the difference without measuring it.
    """
    features = dataset.features if columns is None else dataset.features[:, columns]
    classes = sorted(set(dataset.labels.tolist()))
    x_train, x_test = _standardise(features[split.train], features[split.test])
    y_train = labels_at(dataset.labels, split.train)
    y_test = labels_at(dataset.labels, split.test)
    centroids = np.stack([
        x_train[y_train == label].mean(axis=0) for label in classes
    ])
    distances = np.linalg.norm(x_test[:, None, :] - centroids[None, :, :], axis=2)
    predictions = np.array(
        [classes[i] for i in distances.argmin(axis=1)], dtype=object
    )
    return {
        "test_accuracy": accuracy(predictions, y_test),
        "confusion": confusion_matrix(predictions, y_test, classes),
    }


def majority_accuracy(dataset: Dataset, split: Split) -> float:
    """Predict the most common training class everywhere."""
    y_train = labels_at(dataset.labels, split.train)
    y_test = labels_at(dataset.labels, split.test)
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[int(np.argmax(counts))]
    return accuracy(np.array([majority] * len(y_test), dtype=object), y_test)
