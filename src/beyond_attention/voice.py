"""A prosody front-end: waveform in, affect-bearing frame features out.

The operator asked the model to understand *how* something was said, not what
words were said. This module is the front end for that: it turns a raw waveform
into per-frame prosodic features and a per-frame embedding sequence the
selective scan can consume.

What it does, precisely:

* frames the waveform (Hann window, 25 ms / 10 ms by default),
* estimates **F0** per frame by autocorrelation over a bounded lag range, with a
  voiced/unvoiced decision,
* measures **energy**, **zero-crossing rate** and the **spectral centroid /
  rolloff / flatness** of each frame,
* reduces all of that to a small dict of the *classic affect correlates*
  (F0 mean and spread, energy mean and spread, voiced ratio, jitter, a
  speaking-rate proxy), and
* projects the per-frame features into a ``(frames, d_model)`` sequence through
  a **fixed random projection**.

What it is **not**, stated here rather than left to be discovered:

* **It is not a text-to-speech system and it does not generate audio.** The
  direction is audio in, features out.
* **It is not an emotion classifier.** ``affect_descriptors`` exposes the
  correlates that a valence/arousal head could regress from -- pitch level and
  spread, energy dynamics, tempo, jitter. It has not been trained on, or
  validated against, any labelled emotion data, and a regression fitted to it
  would be a hypothesis, not a result.
* **The embedding is a projection, not a learned model.** ``VoiceEncoder`` is a
  seeded linear map with no bias and no nonlinearity, so it cannot represent
  anything the feature vector does not already contain. It exists to put the
  features in the shape the SSM expects, and it is deliberately incapable of
  adding information.
* **``energy_mean`` and ``energy_std`` are absolute amplitude.** They are not
  loudness-normalised, so they are not comparable across recordings at different
  gains without a normalisation step this module does not perform.

Only ``numpy`` is used, and everything is deterministic given the input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Documented constants. Every one of these is a choice, so each is named and
# explained rather than inlined.
# --------------------------------------------------------------------------

# The F0 search range. 60 Hz is below most adult male speaking F0 (~85-180 Hz)
# and 400 Hz above most adult female speaking F0 (~165-255 Hz), so both the
# speaking range and the expressive excursions outside it are covered. Clamping
# matters: an unclamped autocorrelation search will happily report a lag of 3
# samples as a 5 kHz "pitch" for a noisy frame. A consequence worth stating is
# that a true F0 below 60 Hz is reported as an octave above it (~120 Hz), which
# is the standard failure mode of a bounded search.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0

# Voiced/unvoiced decision, in two parts:
#   * the normalised autocorrelation peak must clear this, and
#   * the frame RMS must clear RMS_FLOOR.
# The peak threshold is what separates a periodic frame from a noise frame: a
# pure tone lands near 0.9-1.0, and the largest normalised peak a white-noise
# frame produces over a realistic lag range is far below this (the experiment
# measures it and prints it, rather than asserting it here).
VOICED_PEAK_THRESHOLD = 0.45
RMS_FLOOR = 1e-6

# Autocorrelation peaks at multiples of the true period. The one at the
# fundamental is the largest for a clean tone, but the *biased* estimator
# tapers with lag, so for a low-pitched signal a harmonic can win the global
# max and report an octave error. Taking the first local peak that reaches this
# fraction of the global maximum picks the fundamental instead. See
# `autocorrelation_f0`.
F0_FIRST_PEAK_FRACTION = 0.85

# Framing default: 25 ms window / 10 ms hop is the standard speech-analysis
# setting -- long enough to hold a couple of periods at 100 Hz, short enough
# that a 10 ms hop tracks pitch movement rather than smearing it.
WINDOW_MS = 25.0
HOP_MS = 10.0

# Spectral rolloff: the frequency below which this fraction of the frame's
# power lies.
ROLLOFF_FRACTION = 0.85

_EPS = 1e-12


def hann_window(n: int) -> np.ndarray:
    """A symmetric Hann window of length ``n`` (``numpy.hanning``).

    Symmetric rather than periodic: this is used for analysis of a single
    frame, not for overlap-add resynthesis, so the symmetric form is the one
    whose endpoints are zero.
    """
    if n < 1:
        raise ValueError("window length must be >= 1")
    return np.hanning(n)


def frame_count(n_samples: int, frame_length: int, hop_length: int) -> int:
    """How many frames ``frame_signal`` will produce. No padding.

    ``1 + (n_samples - frame_length) // hop_length`` for a signal at least one
    frame long, and 0 otherwise. Written out because the frame count is the
    thing a test can pin exactly, and an off-by-one here silently shifts every
    feature sequence relative to the audio.
    """
    if frame_length < 1 or hop_length < 1:
        raise ValueError("frame_length and hop_length must be >= 1")
    if n_samples < frame_length:
        return 0
    return 1 + (n_samples - frame_length) // hop_length


def frame_signal(
    x: np.ndarray, frame_length: int, hop_length: int
) -> np.ndarray:
    """Split a 1-D signal into overlapping frames, shape ``(n_frames, frame_length)``.

    No padding: only whole frames are returned, so the frame count is exactly
    ``frame_count``. The result is a read-only view onto ``x`` (via
    ``sliding_window_view``), which is why it is not modified in place anywhere
    in this module.
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {signal.shape}")
    n_frames = frame_count(signal.size, frame_length, hop_length)
    if n_frames == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(signal, frame_length)
    return windows[::hop_length]


def autocorrelation_f0(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = F0_MIN_HZ,
    f0_max: float = F0_MAX_HZ,
) -> tuple[float, bool, float]:
    """Pitch of one frame by autocorrelation. Returns ``(f0_hz, voiced, confidence)``.

    The estimator is the standard one: remove the frame mean, autocorrelate,
    and look for the lag whose normalised correlation is a peak inside the lag
    range implied by ``f0_min``/``f0_max``.

    Three choices in here are deliberate and each fixes a specific failure:

    * **The biased normalisation** ``r(tau) / r(0)`` is used, not an
      overlap-corrected one. ``r(0)`` is the frame energy, so the estimate
      tapers as the lag approaches the frame length. That taper is what keeps
      the *long-lag* peaks of a noise frame (where few samples overlap and an
      unbiased estimator has its largest variance) from being read as pitch.
    * **The first strong peak, not the global maximum.** Autocorrelation peaks
      at every multiple of the period, so a global-max search can report a
      harmonic as the pitch. The first local maximum reaching
      ``F0_FIRST_PEAK_FRACTION`` of the global maximum is taken instead.
    * **A parabolic refinement** of the peak lag, because the lag grid is
      quantised to samples: at 16 kHz a 150 Hz tone has a period of 106.7
      samples, and reporting 107 is a 0.3% (5 cent) error before refinement.

    Measured accuracy at the default 25 ms window and 16 kHz, on steady tones:
    0.0-1.5% error from 100 Hz to 380 Hz (0.63% at 150 Hz, 0.72% at 220 Hz),
    degrading to 2.5% at 80 Hz and 3.2% at 70 Hz, where the frame holds fewer
    than two periods and the peak the search is looking for stops being a peak.
    The tests assert a 2% tolerance at frequencies where the estimator is
    actually reliable rather than a tolerance that only holds in the middle of
    the range.

    Unvoiced frames return ``(nan, False, confidence)``: the pitch of an
    aperiodic frame does not exist, and returning 0.0 would quietly enter any
    downstream mean as a real value.
    """
    signal = np.asarray(frame, dtype=np.float64).reshape(-1)
    n = signal.size
    if n < 4:
        return float("nan"), False, 0.0
    centered = signal - signal.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 0.0:
        return float("nan"), False, 0.0

    # Biased autocorrelation: lag 0 .. n-1, normalised by lag-0 energy.
    corr = np.correlate(centered, centered, mode="full")[n - 1 :] / energy

    lag_min = max(1, int(np.floor(sample_rate / f0_max)))
    lag_max = min(n - 1, int(np.ceil(sample_rate / f0_min)))
    if lag_max <= lag_min + 1:
        return float("nan"), False, 0.0

    search = corr[lag_min : lag_max + 1]
    global_max = float(search.max())
    if global_max <= 0.0:
        return float("nan"), False, 0.0

    # Local maxima strictly inside the search window.
    interior = slice(1, search.size - 1)
    is_peak = (search[interior] > search[:-2]) & (search[interior] >= search[2:])
    peak_offsets = np.flatnonzero(is_peak)
    if peak_offsets.size:
        strong = peak_offsets[
            search[interior][peak_offsets] >= F0_FIRST_PEAK_FRACTION * global_max
        ]
        # The first strong peak is the fundamental; the global max is the
        # fallback when no interior peak clears the bar (e.g. the peak sits on
        # the boundary of the range).
        offset = int(strong[0]) if strong.size else int(np.argmax(search))
    else:
        offset = int(np.argmax(search))
    lag = lag_min + offset
    confidence = float(search[offset])
    if confidence < VOICED_PEAK_THRESHOLD:
        return float("nan"), False, confidence

    # Parabolic interpolation of the peak location, then clamp to the search
    # range so a boundary peak cannot be refined outside it.
    #
    # The interpolation is done on the *taper-corrected* correlation, not on the
    # biased one used for the search. The biased estimate is roughly
    # `(1 - tau/n) * true(tau)`, and that linear taper pulls the parabola's
    # vertex toward short lags: interpolating the biased values put a 150 Hz
    # tone at 151.5 Hz (+1.0%), while correcting the three points first puts it
    # at +0.6%. Correcting the whole array instead would amplify exactly the
    # long-lag noise peaks the biased form exists to suppress, which is why only
    # these three points are corrected.
    if 0 < lag < n - 1:
        def tapered(value: float, at: int) -> float:
            return value * n / (n - at)

        left = tapered(float(corr[lag - 1]), lag - 1)
        middle = tapered(float(corr[lag]), lag)
        right = tapered(float(corr[lag + 1]), lag + 1)
        denom = left - 2.0 * middle + right
        if denom != 0.0:
            shift = 0.5 * (left - right) / denom
            refined = lag + float(np.clip(shift, -1.0, 1.0))
        else:
            refined = float(lag)
    else:
        refined = float(lag)
    refined = float(np.clip(refined, lag_min, lag_max))
    if refined <= 0.0:
        return float("nan"), False, confidence
    return float(sample_rate / refined), True, confidence


@dataclass(frozen=True, eq=False)
class VoiceFeatures:
    """Per-frame features for one waveform.

    Every array is length ``n_frames`` and aligned frame-by-frame. ``f0_hz``
    carries ``nan`` exactly where ``voiced`` is false: an unvoiced frame has no
    pitch, and every consumer is expected to mask on ``voiced`` rather than
    average ``f0_hz`` blindly. ``voiced_confidence`` is the normalised
    autocorrelation peak the decision was made on, kept so a caller can apply
    its own threshold instead of trusting this one.
    """

    sample_rate: int
    frame_length: int
    hop_length: int
    times_s: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    voiced_confidence: np.ndarray
    rms: np.ndarray
    energy_flux: np.ndarray
    zcr: np.ndarray
    centroid_hz: np.ndarray
    rolloff_hz: np.ndarray
    flatness: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.rms.size)

    @property
    def duration_s(self) -> float:
        """Wall-clock span covered by the frames (not the raw signal length)."""
        if self.n_frames == 0:
            return 0.0
        return float(self.n_frames * self.hop_length / self.sample_rate)


def _spectral_frame(
    frame: np.ndarray, window: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum and frequencies of one Hann-windowed frame."""
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
    return spectrum, freqs


def frame_features(
    x: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = WINDOW_MS,
    hop_ms: float = HOP_MS,
    f0_min: float = F0_MIN_HZ,
    f0_max: float = F0_MAX_HZ,
) -> VoiceFeatures:
    """Frame a waveform and measure every per-frame feature.

    ``window_ms`` and ``hop_ms`` are converted to whole samples by rounding, so
    at 16 kHz the defaults are 400 and 160 samples exactly.

    This is a plain Python loop over frames: correct and fast enough for
    seconds of audio, and not a vectorised STFT. The loop is not hidden because
    a caller feeding minutes of audio should know where the time goes.

    ``f0_hz`` is ``nan`` on unvoiced frames. Every other returned array is
    finite, including for silence and for an empty input.
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {signal.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 < f0_min < f0_max:
        raise ValueError("need 0 < f0_min < f0_max")

    frame_length = int(round(sample_rate * window_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_ms / 1000.0))
    if frame_length < 2:
        raise ValueError("window is shorter than two samples; raise window_ms")
    if hop_length < 1:
        raise ValueError("hop is shorter than one sample; raise hop_ms")

    frames = frame_signal(signal, frame_length, hop_length)
    n_frames = frames.shape[0]
    window = hann_window(frame_length)

    times = np.empty(n_frames, dtype=np.float64)
    f0 = np.full(n_frames, np.nan, dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=bool)
    confidence = np.zeros(n_frames, dtype=np.float64)
    rms = np.zeros(n_frames, dtype=np.float64)
    flux = np.zeros(n_frames, dtype=np.float64)
    zcr = np.zeros(n_frames, dtype=np.float64)
    centroid = np.zeros(n_frames, dtype=np.float64)
    rolloff = np.zeros(n_frames, dtype=np.float64)
    flatness = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        frame = np.asarray(frames[i], dtype=np.float64)
        times[i] = (i * hop_length + frame_length / 2.0) / sample_rate
        level = float(np.sqrt(np.mean(frame * frame)))
        rms[i] = level

        # Zero-crossing rate over the raw frame: a cheap, gain-invariant
        # correlate of spectral balance (high for fricatives and noise, low for
        # voiced vowels).
        if frame.size > 1:
            zcr[i] = float(np.count_nonzero(np.diff(np.signbit(frame)))
                           / (frame.size - 1))

        spectrum, freqs = _spectral_frame(frame, window, sample_rate)
        power = spectrum * spectrum
        total = float(power.sum())
        if total > 0.0:
            centroid[i] = float((freqs * spectrum).sum() / spectrum.sum())
            cumulative = np.cumsum(power)
            idx = int(np.searchsorted(cumulative, ROLLOFF_FRACTION * total))
            rolloff[i] = float(freqs[min(idx, freqs.size - 1)])
            # Spectral flatness: geometric mean over arithmetic mean of the
            # power spectrum. Near 0 for a tone, near 1 for white noise, so it
            # is the frame-level "is this periodic" companion to the
            # autocorrelation decision.
            floor = _EPS * float(power.max())
            positive = power + floor
            flatness[i] = float(
                np.exp(np.mean(np.log(positive))) / max(positive.mean(), floor)
            )

        # The energy gate is separate from the autocorrelation test on purpose:
        # a frame of digital silence (or of dither) has no pitch regardless of
        # what a normalised correlation of rounding noise happens to say.
        if level <= RMS_FLOOR:
            continue
        pitch, is_voiced, peak = autocorrelation_f0(
            frame, sample_rate, f0_min, f0_max
        )
        confidence[i] = peak
        voiced[i] = is_voiced
        f0[i] = pitch

    # Energy contour dynamics: relative frame-to-frame change, in [0, 1), so it
    # is amplitude-scale free. Zero at the first frame (nothing precedes it).
    for i in range(1, n_frames):
        denom = rms[i] + rms[i - 1] + _EPS
        flux[i] = abs(rms[i] - rms[i - 1]) / denom

    return VoiceFeatures(
        sample_rate=int(sample_rate),
        frame_length=frame_length,
        hop_length=hop_length,
        times_s=times,
        f0_hz=f0,
        voiced=voiced,
        voiced_confidence=confidence,
        rms=rms,
        energy_flux=flux,
        zcr=zcr,
        centroid_hz=centroid,
        rolloff_hz=rolloff,
        flatness=flatness,
    )


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

def _voiced_runs(voiced: np.ndarray) -> int:
    """Number of contiguous voiced segments in the frame sequence."""
    if voiced.size == 0:
        return 0
    starts = np.count_nonzero(voiced & ~np.concatenate(([False], voiced[:-1])))
    return int(starts)


def affect_descriptors(features: VoiceFeatures) -> dict[str, float]:
    """The interpretable prosody summary a valence/arousal head could regress.

    These are the correlates affect literature uses, exposed as numbers:

    * ``f0_mean`` / ``f0_std`` / ``f0_range`` -- pitch level and *spread*. Spread
      is the one that matters: monotone speech is not expressive speech, and two
      utterances with the same mean F0 can differ entirely in how much the pitch
      moves.
    * ``energy_mean`` / ``energy_std`` / ``energy_flux_mean`` -- loudness level
      and how much it moves, over *all* frames.
    * ``energy_mean_voiced`` / ``energy_std_voiced`` -- the same two statistics
      restricted to voiced frames. This distinction was added after the
      experiment measured it: over all frames, the RMS variance is dominated by
      where the pauses are, and it did not track the amplitude spread the
      synthesis varied (0.135, 0.157, 0.137, 0.131 across conditions whose
      amplitude spread was 0.03, 0.45, 0.05, 0.15). Restricted to voiced
      frames, the same four conditions read 0.014, 0.115, 0.009, 0.047 -- it is
      loudness spread *of the speech* rather than of the silence, and it costs
      nothing to carry both.
    * ``voiced_ratio`` -- fraction of frames with periodic structure.
    * ``jitter`` -- mean absolute successive-F0 difference over consecutive
      voiced frame *pairs*, in Hz. Only pairs of adjacent voiced frames count; a
      jump across an unvoiced gap is a voicing boundary, not jitter, and
      including it would make any signal with pauses look maximally jittery.
      ``jitter_relative`` is the same quantity divided by ``f0_mean``, which is
      the dimensionless ("local jitter %") form used in the phonetics
      literature.
    * ``speaking_rate`` -- voiced *runs* per second, a syllable-rate proxy. It is
      a proxy and not a measurement of speech rate: it counts acoustic voiced
      segments, which in real speech approximate syllables only when the
      segmentation works, and it cannot see unvoiced consonants at all.

    **It does not predict emotion.** Nothing here has been fitted to labelled
    affect data, and a score computed from these numbers would be a hypothesis
    about the correlates rather than a measurement of feeling.

    With no voiced frames (silence, or pure noise) every pitch statistic is
    ``0.0`` rather than ``nan``, so the dict stays finite for downstream
    arithmetic; ``voiced_ratio == 0.0`` is the flag that says the pitch numbers
    are placeholders. Consumers must gate on it.
    """
    f0 = features.f0_hz
    voiced = features.voiced
    rms = features.rms
    n_voiced = int(np.count_nonzero(voiced))
    voiced_rms = rms[voiced] if n_voiced else rms[:0]

    if n_voiced:
        voiced_f0 = f0[voiced]
        f0_mean = float(voiced_f0.mean())
        f0_std = float(voiced_f0.std())
        f0_range = float(voiced_f0.max() - voiced_f0.min())
    else:
        f0_mean = f0_std = f0_range = 0.0

    # Successive differences, restricted to pairs of *adjacent* voiced frames.
    jitter = 0.0
    if n_voiced >= 2:
        adjacent = voiced[1:] & voiced[:-1]
        if np.any(adjacent):
            deltas = np.abs(np.diff(f0)[adjacent])
            jitter = float(deltas.mean())

    duration = features.duration_s
    runs = _voiced_runs(voiced)
    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "jitter": jitter,
        "jitter_relative": jitter / f0_mean if n_voiced and f0_mean > 0 else 0.0,
        "energy_mean": float(rms.mean()) if rms.size else 0.0,
        "energy_std": float(rms.std()) if rms.size else 0.0,
        "energy_mean_voiced": float(voiced_rms.mean()) if n_voiced else 0.0,
        "energy_std_voiced": float(voiced_rms.std()) if n_voiced else 0.0,
        "energy_flux_mean": float(features.energy_flux.mean())
        if features.n_frames
        else 0.0,
        "voiced_ratio": n_voiced / features.n_frames if features.n_frames else 0.0,
        "speaking_rate": runs / duration if duration > 0 else 0.0,
        "zcr_mean": float(features.zcr.mean()) if features.n_frames else 0.0,
        "centroid_mean": float(features.centroid_hz.mean())
        if features.n_frames
        else 0.0,
        "flatness_mean": float(features.flatness.mean())
        if features.n_frames
        else 0.0,
        "n_frames": float(features.n_frames),
        "duration_s": duration,
    }


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

# Column order of the per-frame vector the encoder projects. Fixed and named so
# the projection can be reasoned about, and so a test can index it.
FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "energy_flux",
    "zcr",
    "centroid_norm",
    "rolloff_norm",
    "flatness",
    "f0_norm",
    "voiced",
)


def feature_matrix(features: VoiceFeatures) -> np.ndarray:
    """Stack the frame features into ``(n_frames, len(FEATURE_NAMES))``.

    Columns are scaled into roughly ``[0, 1]`` so that one random projection
    does not weight a kilohertz-valued column (the centroid) a thousand times
    more than a unitless one (flatness):

    * ``rms`` is left in amplitude units and is the one column that is not
      bounded, because normalising it would discard the loudness information
      the encoder is meant to carry.
    * ``centroid_norm`` and ``rolloff_norm`` are divided by Nyquist.
    * ``f0_norm`` is ``(log f0 - log 60) / (log 400 - log 60)`` on voiced
      frames and 0.0 elsewhere; pitch is perceptually logarithmic, so the log
      spacing is the honest one. The ``voiced`` column lets the projection
      distinguish "unvoiced" from "a very low pitch".
    """
    f0 = features.f0_hz
    voiced = features.voiced
    nyquist = features.sample_rate / 2.0
    f0_norm = np.zeros(features.n_frames, dtype=np.float64)
    if np.any(voiced):
        f0_norm[voiced] = (
            np.log(f0[voiced]) - np.log(F0_MIN_HZ)
        ) / (np.log(F0_MAX_HZ) - np.log(F0_MIN_HZ))
    columns = (
        features.rms,
        features.energy_flux,
        features.zcr,
        features.centroid_hz / nyquist,
        features.rolloff_hz / nyquist,
        features.flatness,
        f0_norm,
        voiced.astype(np.float64),
    )
    if features.n_frames == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.stack(columns, axis=1)


class VoiceEncoder:
    """Project per-frame features to ``(n_frames, d_model)``. Seeded, linear.

    **This is a projection, not a trained model.** ``encode`` is exactly
    ``feature_matrix(features) @ self.projection``: no bias, no nonlinearity,
    no learned parameters of any kind. It is a fixed random map from the
    interpretable features into the width the state-space model expects, and it
    is deliberately incapable of representing anything the features do not
    already contain -- a test asserts that linearity, because a projection that
    quietly added a nonlinearity would be making a claim this module does not
    support.

    Determinism is the point of the seed: the same seed gives the same map on
    every machine and every run, so a downstream measurement is reproducible.
    """

    def __init__(self, d_model: int = 64, seed: int = 0) -> None:
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        self.d_model = int(d_model)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # Scaled by 1/sqrt(n_features) so the projected embedding has unit
        # variance for unit-variance inputs, which keeps the magnitudes
        # independent of how many features the front end happens to expose.
        scale = 1.0 / np.sqrt(len(FEATURE_NAMES))
        self.projection = rng.normal(
            0.0, scale, size=(len(FEATURE_NAMES), self.d_model)
        )

    def encode(self, features: VoiceFeatures) -> np.ndarray:
        """``(n_frames, d_model)`` embeddings. Linear in the feature matrix."""
        return feature_matrix(features) @ self.projection

    def encode_waveform(
        self,
        x: np.ndarray,
        sample_rate: int,
        *,
        window_ms: float = WINDOW_MS,
        hop_ms: float = HOP_MS,
    ) -> np.ndarray:
        """Frames ``x`` and encodes it in one call. Returns ``(n_frames, d_model)``."""
        return self.encode(
            frame_features(
                x, sample_rate, window_ms=window_ms, hop_ms=hop_ms
            )
        )


# --------------------------------------------------------------------------
# Bridge into the model
# --------------------------------------------------------------------------

def waveform_to_model_input(
    waveform: np.ndarray,
    sample_rate: int,
    d_model: int,
    *,
    encoder: VoiceEncoder | None = None,
    window_ms: float = WINDOW_MS,
    hop_ms: float = HOP_MS,
) -> np.ndarray:
    """Waveform to the ``(B, L, D)`` layout the SSM consumes.

    ``B`` is 1 for a single 1-D waveform, or the number of rows of a 2-D
    ``(B, n_samples)`` array; ``L`` is the frame count the window/hop arithmetic
    predicts; ``D`` is ``d_model``, the width ``SelectiveSSMBlock`` expects.

    This is a thin adapter on purpose: it frames, encodes, and returns the
    array. It does **not** build ``delta``, ``B`` or ``C``, because those are
    produced *inside* the SSM block from its input -- this front end replaces
    the token embedding for a waveform, and nothing else. The conversion to
    ``selective_scan`` is two lines and is not hidden behind a torch import, so
    this module stays numpy-only:

        >>> import numpy as np, torch
        >>> from beyond_attention.ssm import selective_scan
        >>> x = waveform_to_model_input(wave, 16_000, d_model=64)   # (1, L, 64)
        >>> x = torch.from_numpy(x).float()
        >>> B, L, D = x.shape
        >>> A = -torch.ones(D, 16)
        >>> delta = torch.full_like(x, 0.01)
        >>> b = torch.randn(1, L, 16)
        >>> c = torch.randn(1, L, 16)
        >>> y = selective_scan(x, delta, A, b, c)                    # (1, L, 64)

    A 2-D input is a rectangular ``(B, n_samples)`` array, so every row has the
    same length and therefore frames to the same ``L``; ``np.stack`` is safe.
    A batch of *ragged* recordings is a caller-side problem (pad or process them
    one at a time), not one this function can solve silently.
    """
    signal = np.asarray(waveform, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError(
            f"expected a 1-D waveform or a (B, n_samples) array, got {signal.shape}"
        )
    encoder = encoder or VoiceEncoder(d_model)
    if encoder.d_model != d_model:
        raise ValueError(
            f"encoder produces d_model={encoder.d_model}, asked for {d_model}"
        )

    encoded = [
        encoder.encode_waveform(row, sample_rate, window_ms=window_ms,
                                hop_ms=hop_ms)
        for row in signal
    ]
    return np.stack(encoded, axis=0)  # (B, L, d_model)
