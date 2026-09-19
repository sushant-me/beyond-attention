"""The voice front-end, checked against signals whose right answer is known.

Prosody features are easy to compute and easy to compute *wrongly in a way that
still produces a plausible curve*: a pitch tracker that reports a harmonic as
the pitch, or that calls every frame voiced, produces numbers that look fine and
mean nothing. So the tests here come in two kinds.

**Analytic ground truth.** A pure tone has one right answer for F0 and it is
known before the code runs. A linear amplitude ramp has a known energy contour.
An octave is a factor of two. These are asserted to a stated tolerance.

**Controls that can fail.** A test that only checks "a 150 Hz tone gives
150 Hz" cannot distinguish a working voiced/unvoiced decision from one that
always answers yes, so the same test also feeds the detector white noise and
requires it to *not* be voiced. Likewise the encoder tests do not only check
shapes -- they check that it has no bias, that it is exactly the linear map it
documents, and that different pitches give different embeddings, because a
projection of all zeros would pass a shape check.

Where a tolerance is loose, the measured error is recorded next to it rather
than hidden behind the number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from beyond_attention.voice import (
    FEATURE_NAMES,
    F0_MAX_HZ,
    F0_MIN_HZ,
    VoiceEncoder,
    affect_descriptors,
    autocorrelation_f0,
    feature_matrix,
    frame_count,
    frame_features,
    frame_signal,
    waveform_to_model_input,
)

SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz

# The estimator is measured to about 0.6% error at 150 Hz and 0.7% at 220 Hz at
# this window length; 2% is a tolerance that also survives a different machine's
# floating point without being so loose it would accept an octave error.
F0_TOLERANCE = 0.02


def tone(freq_hz: float, seconds: float, sample_rate: int = SAMPLE_RATE,
         amplitude: float = 0.5) -> np.ndarray:
    """A pure sine, the signal with a known F0."""
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def bursts(freq_hz: float, seconds: float, n_bursts: int,
           duty: float = 0.5) -> np.ndarray:
    """``n_bursts`` tone segments separated by silence: a known voiced-run count."""
    n = int(round(seconds * SAMPLE_RATE))
    signal = np.zeros(n)
    span = n // n_bursts
    for i in range(n_bursts):
        start = i * span
        stop = start + int(span * duty)
        segment = np.arange(stop - start) / SAMPLE_RATE
        signal[start:stop] = 0.5 * np.sin(2 * np.pi * freq_hz * segment)
    return signal


def descriptors(x: np.ndarray) -> dict[str, float]:
    return affect_descriptors(frame_features(x, SAMPLE_RATE))


# --------------------------------------------------------------------------
# Framing: the arithmetic is exact, so it is pinned exactly
# --------------------------------------------------------------------------

def test_frame_count_is_exactly_the_window_hop_arithmetic() -> None:
    """No padding, so the count is a closed form and the test is exact."""
    cases = [
        (0, 0), (1, 0), (399, 0), (400, 1), (401, 1),
        (559, 1), (560, 2), (16_000, 98), (16_001, 98),
    ]
    for n_samples, expected in cases:
        assert frame_count(n_samples, FRAME_LENGTH, HOP_LENGTH) == expected
        frames = frame_signal(np.zeros(n_samples), FRAME_LENGTH, HOP_LENGTH)
        assert frames.shape == (expected, FRAME_LENGTH)

    # The count is not just any formula: the frames it promises must fit, and
    # one more hop must not. An off-by-one silently shifts every feature
    # sequence against the audio, which is the failure this pins down.
    n_samples = 3_000
    n_frames = frame_count(n_samples, FRAME_LENGTH, HOP_LENGTH)
    assert (n_frames - 1) * HOP_LENGTH + FRAME_LENGTH <= n_samples
    assert n_frames * HOP_LENGTH + FRAME_LENGTH > n_samples


def test_frames_are_the_requested_stretch_of_the_signal() -> None:
    """A frame index must mean a position, not just a row."""
    x = np.arange(1_000, dtype=np.float64)
    frames = frame_signal(x, FRAME_LENGTH, HOP_LENGTH)
    assert frames[0, 0] == 0.0
    assert frames[0, -1] == FRAME_LENGTH - 1
    assert frames[2, 0] == 2 * HOP_LENGTH
    assert frames[2, -1] == 2 * HOP_LENGTH + FRAME_LENGTH - 1


# --------------------------------------------------------------------------
# F0: analytic ground truth, and the control that can falsify the decision
# --------------------------------------------------------------------------

@pytest.mark.parametrize("freq_hz", [150.0, 220.0, 300.0])
def test_pure_tone_f0_is_recovered_within_two_percent(freq_hz: float) -> None:
    summary = descriptors(tone(freq_hz, 0.5))
    assert summary["voiced_ratio"] == pytest.approx(1.0)
    error = abs(summary["f0_mean"] - freq_hz) / freq_hz
    assert error < F0_TOLERANCE, (
        f"{freq_hz} Hz recovered as {summary['f0_mean']:.2f} Hz ({error:.3%})"
    )


def test_single_frame_f0_is_recovered_too() -> None:
    """One frame is the smallest input the estimator sees, and it is not special."""
    frame = tone(220.0, 0.5)[:FRAME_LENGTH]
    f0, voiced, confidence = autocorrelation_f0(frame, SAMPLE_RATE)
    assert voiced
    assert abs(f0 - 220.0) / 220.0 < F0_TOLERANCE
    assert confidence > 0.5  # a clean tone is a confident decision


def test_white_noise_is_not_reported_as_voiced() -> None:
    """The control for the voiced/unvoiced decision.

    This test fails if the decision is broken in the direction that matters:
    a detector that always answers "voiced" passes every pure-tone assertion in
    this file and is useless. The positive half of the control is inside the
    same test on purpose, so a tone and noise are always measured by the same
    code path in the same run.
    """
    rng = np.random.default_rng(0)
    noise = 0.3 * rng.standard_normal(SAMPLE_RATE)
    noisy = frame_features(noise, SAMPLE_RATE)

    assert noisy.n_frames > 90
    assert affect_descriptors(noisy)["voiced_ratio"] < 0.05
    # Measured across eight seeds the largest normalised peak on white noise was
    # 0.17, against a 0.45 threshold. The margin is the point of the assertion.
    assert noisy.voiced_confidence.max() < 0.3

    sine = affect_descriptors(frame_features(tone(150.0, 1.0), SAMPLE_RATE))
    assert sine["voiced_ratio"] > 0.95
    assert sine["f0_std"] < 1.0  # a steady tone has a steady pitch


def test_silence_yields_zero_energy_and_no_voiced_frames() -> None:
    features = frame_features(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    summary = affect_descriptors(features)

    assert features.n_frames == 98
    assert summary["energy_mean"] == 0.0
    assert summary["energy_std"] == 0.0
    assert summary["voiced_ratio"] == 0.0
    # Pitch statistics are 0.0 rather than nan so downstream arithmetic stays
    # finite; voiced_ratio is what says they are placeholders.
    assert summary["f0_mean"] == 0.0
    assert summary["f0_std"] == 0.0
    assert summary["jitter"] == 0.0
    assert all(np.isfinite(list(summary.values())))

    # An all-zero frame reaching the estimator directly must not invent a pitch.
    f0, voiced, _ = autocorrelation_f0(np.zeros(FRAME_LENGTH), SAMPLE_RATE)
    assert not voiced
    assert np.isnan(f0)


def test_the_energy_floor_is_what_it_says() -> None:
    """A frame below the floor is unvoiced even when its autocorrelation is clean.

    This is the one place the energy gate and the autocorrelation decision can
    disagree: a very quiet but perfectly periodic frame has a normalised peak
    near 0.9 and would be called voiced on that evidence alone. Measured: a
    150 Hz tone at 1e-6 amplitude has RMS 7.1e-7, under the 1e-6 floor (unvoiced
    ratio 0.000), and the same tone at 1e-5 has RMS 7.1e-6 and is voiced for
    100% of frames. Both halves are asserted because a gate that rejected
    everything would pass the first one.
    """
    quiet = descriptors(tone(150.0, 0.5, amplitude=1e-6))
    loud = descriptors(tone(150.0, 0.5, amplitude=1e-5))

    assert quiet["voiced_ratio"] == 0.0
    assert quiet["f0_mean"] == 0.0
    assert loud["voiced_ratio"] > 0.95
    assert loud["f0_mean"] == pytest.approx(150.0, rel=F0_TOLERANCE)


# --------------------------------------------------------------------------
# Energy, and the contour dynamics
# --------------------------------------------------------------------------

def test_amplitude_ramp_has_energy_variance_and_peaks_in_the_middle() -> None:
    """A ramp up then down: the envelope shape is known analytically."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    envelope = np.clip(np.minimum(t / 0.5, (1.0 - t) / 0.5), 0.0, 1.0)
    features = frame_features(envelope * np.sin(2 * np.pi * 200.0 * t), SAMPLE_RATE)
    summary = affect_descriptors(features)

    assert summary["energy_std"] > 0.0
    assert summary["energy_std"] < summary["energy_mean"]  # not a square wave
    assert summary["energy_flux_mean"] > 0.0  # the contour actually moves

    third = features.n_frames // 3
    quiet_ends = np.concatenate([features.rms[:third], features.rms[-third:]])
    assert features.rms[third:2 * third].mean() > 2.0 * quiet_ends.mean()
    # The ends of the ramp are silent and the middle is at full amplitude.
    assert features.rms.max() > 5.0 * features.rms[0]
    assert features.rms.max() > 5.0 * features.rms[-1]
    # "The loud half": the argmax is at the middle of a symmetric ramp.
    middle = features.rms.argmax() / (features.n_frames - 1)
    assert 0.35 < middle < 0.65


def test_a_flat_tone_has_no_energy_variance() -> None:
    """The complement of the ramp test: a constant envelope must read flat."""
    summary = descriptors(tone(200.0, 0.5))
    assert summary["energy_std"] < 0.01 * summary["energy_mean"] + 1e-6


def test_voiced_frame_energy_spread_tracks_loudness_not_pauses() -> None:
    """Why ``energy_std_voiced`` exists alongside ``energy_std``.

    Two signals with the same burst pattern, pitch and duty, differing only in
    whether the bursts are equally loud or alternate loud/soft. Over all frames
    the RMS spread is dominated by where the silence is and the two barely
    differ (measured 0.097 against 0.127, a factor of 1.3); restricted to voiced
    frames the amplitude spread is the dominant term (0.021 against 0.104, a
    factor of 4.9). Both halves are asserted, so the test fails if the
    distinction between the two statistics is ever collapsed.
    """
    def gated(amplitudes: list[float]) -> np.ndarray:
        signal = np.zeros(SAMPLE_RATE)
        span = SAMPLE_RATE // len(amplitudes)
        for i, amplitude in enumerate(amplitudes):
            start = i * span
            stop = start + int(span * 0.6)
            t = np.arange(stop - start) / SAMPLE_RATE
            signal[start:stop] = amplitude * np.sin(2 * np.pi * 200.0 * t)
        return signal

    even = descriptors(gated([0.3, 0.3, 0.3, 0.3]))
    varied = descriptors(gated([0.15, 0.45, 0.15, 0.45]))

    assert varied["energy_mean_voiced"] == pytest.approx(
        even["energy_mean_voiced"], rel=0.05
    )
    assert varied["energy_std_voiced"] > 3.0 * even["energy_std_voiced"]
    assert varied["energy_std"] < 2.0 * even["energy_std"]


# --------------------------------------------------------------------------
# The affect-relevant discriminators
# --------------------------------------------------------------------------

def test_two_tones_an_octave_apart_have_a_two_to_one_pitch_ratio() -> None:
    low = descriptors(tone(150.0, 0.5))
    high = descriptors(tone(300.0, 0.5))
    assert high["f0_mean"] != low["f0_mean"]
    assert high["f0_mean"] / low["f0_mean"] == pytest.approx(2.0, rel=0.05)


def test_vibrato_raises_f0_spread_above_a_steady_tone() -> None:
    """The discriminator that makes F0 spread worth carrying.

    Two utterances at the same pitch level differ in how much the pitch moves,
    and *that* is the affect-relevant axis (monotone versus expressive), not the
    mean. The test asserts the spread separates them while the means stay close,
    so a broken estimator that reported a wildly different mean for the vibrato
    would fail rather than pass by accident.
    """
    steady = descriptors(tone(200.0, 1.0))

    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    depth, rate = 0.10, 4.0
    instantaneous = 2 * np.pi * 200.0 * (
        t + depth / (2 * np.pi * rate) * np.sin(2 * np.pi * rate * t)
    )
    vibrato = descriptors(0.5 * np.sin(instantaneous))

    assert vibrato["f0_std"] > steady["f0_std"] + 5.0
    assert vibrato["f0_range"] > steady["f0_range"] + 10.0
    # Measured 13.8 Hz of spread for the vibrato against 0.0 Hz steady, with the
    # means 0.8% apart -- so this is spread, not level.
    assert abs(vibrato["f0_mean"] - steady["f0_mean"]) < 0.05 * steady["f0_mean"]
    assert vibrato["jitter"] > steady["jitter"]


def test_speaking_rate_counts_voiced_runs_per_second() -> None:
    """Runs per second, not "how much of the signal is voiced".

    The two signals compared here have almost the same voiced *fraction* -- both
    are half tone and half silence -- and differ only in how many times the tone
    starts. A descriptor that were just re-measuring the voiced fraction could
    not tell them apart; the rate proxy must.
    """
    sparse = descriptors(bursts(200.0, 1.0, n_bursts=2))
    dense = descriptors(bursts(200.0, 1.0, n_bursts=8))
    continuous = descriptors(tone(200.0, 1.0))

    assert sparse["speaking_rate"] == pytest.approx(2.0, abs=0.5)
    assert dense["speaking_rate"] == pytest.approx(8.0, abs=0.5)
    assert continuous["speaking_rate"] == pytest.approx(1.0, abs=0.5)
    assert dense["speaking_rate"] > 3.0 * sparse["speaking_rate"]
    assert abs(sparse["voiced_ratio"] - dense["voiced_ratio"]) < 0.15
    assert sparse["voiced_ratio"] < continuous["voiced_ratio"]


def test_spectral_features_order_a_tone_below_noise() -> None:
    """Spectral centroid and flatness, against a known ordering.

    A pure 200 Hz tone has all its power in one bin and a white-noise frame
    spreads its power over the band, so centroid(noise) > centroid(tone) and
    flatness(noise) > flatness(tone) must both hold. The measured values are
    200 Hz / 0.00 against 3,996 Hz / 0.56.
    """
    rng = np.random.default_rng(0)
    noisy = descriptors(0.3 * rng.standard_normal(SAMPLE_RATE))
    tonal = descriptors(tone(200.0, 1.0))

    assert noisy["centroid_mean"] > 10.0 * tonal["centroid_mean"]
    assert noisy["flatness_mean"] > 0.3
    assert tonal["flatness_mean"] < 0.01
    assert noisy["zcr_mean"] > 5.0 * tonal["zcr_mean"]


def test_centroid_weights_by_amplitude_not_by_power() -> None:
    """Two tones with known amplitudes pin down *which* weighting is used.

    A 500 Hz tone at amplitude 0.5 and a 2000 Hz tone at 0.25. The
    magnitude-weighted mean frequency is 1000 Hz and the power-weighted one is
    800 Hz, so the two implementations of "spectral centroid" differ by 20%
    here. The module documents magnitude weighting and the measured value is
    981 Hz: 1.9% from the analytic 1000 and 23% from the alternative, which is
    what makes this a test rather than an acceptance of either.
    """
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    x = 0.5 * np.sin(2 * np.pi * 500.0 * t) + 0.25 * np.sin(2 * np.pi * 2000.0 * t)
    centroid = descriptors(x)["centroid_mean"]

    assert abs(centroid - 1000.0) < 0.05 * 1000.0
    assert abs(centroid - 800.0) > 0.10 * 800.0


@pytest.mark.parametrize("freq_hz", [150.0, 200.0, 300.0, 1000.0])
def test_centroid_and_rolloff_sit_at_the_one_frequency_present(
    freq_hz: float,
) -> None:
    """Spectral features against analytic ground truth, not just an ordering.

    With a single tone in the frame the magnitude-weighted mean frequency *is*
    that frequency, so the centroid is checkable to within a couple of percent
    (measured +1.1% at 150 Hz, +0.05% at 200 Hz). The rolloff is the 85% power
    point: a Hann window puts about half the power in the peak bin and the rest
    in its two neighbours, so the cumulative power passes 85% at the peak bin or
    the one above it. Measured 160, 240, 320 and 1040 Hz for tones at 150, 200,
    300 and 1000 Hz — the nearest bin at or above the tone, plus at most one
    more. The tolerance is therefore two bins (80 Hz), which is loose enough to
    be robust and tight enough that a rolloff reported from the wrong end of the
    spectrum fails.
    """
    features = frame_features(tone(freq_hz, 0.5), SAMPLE_RATE)
    bin_hz = SAMPLE_RATE / FRAME_LENGTH

    assert features.centroid_hz.mean() == pytest.approx(freq_hz, rel=0.02)
    assert 0.0 <= features.rolloff_hz.mean() - freq_hz <= 2.0 * bin_hz + 1.0
    assert features.rolloff_hz.mean() >= features.centroid_hz.mean()


# --------------------------------------------------------------------------
# Contracts: shapes, finiteness, degenerate inputs
# --------------------------------------------------------------------------

def test_every_output_is_finite_and_f0_is_nan_exactly_where_unvoiced() -> None:
    rng = np.random.default_rng(1)
    mixed = np.concatenate([
        np.zeros(2_000),
        tone(180.0, 0.5),
        0.3 * rng.standard_normal(4_000),
        tone(300.0, 0.5),
    ])
    features = frame_features(mixed, SAMPLE_RATE)

    assert features.voiced.any() and (~features.voiced).any()
    assert np.isnan(features.f0_hz[~features.voiced]).all()
    assert np.isfinite(features.f0_hz[features.voiced]).all()
    for name in ("times_s", "voiced_confidence", "rms", "energy_flux", "zcr",
                 "centroid_hz", "rolloff_hz", "flatness"):
        assert np.isfinite(getattr(features, name)).all(), name
    assert all(np.isfinite(list(affect_descriptors(features).values())))


def test_descriptors_are_floats_with_the_documented_names() -> None:
    summary = affect_descriptors(frame_features(tone(200.0, 0.5), SAMPLE_RATE))
    required = {
        "f0_mean", "f0_std", "energy_mean", "energy_std", "voiced_ratio",
        "jitter", "speaking_rate",
    }
    assert required <= set(summary)
    assert all(isinstance(value, float) for value in summary.values())


def test_framing_parameters_are_honoured() -> None:
    """A different window and hop must move the frame count as advertised."""
    x = tone(150.0, 1.0)
    for window_ms, hop_ms in ((25.0, 10.0), (40.0, 20.0), (10.0, 5.0)):
        features = frame_features(x, SAMPLE_RATE, window_ms=window_ms,
                                  hop_ms=hop_ms)
        expected_frame = int(round(SAMPLE_RATE * window_ms / 1000.0))
        expected_hop = int(round(SAMPLE_RATE * hop_ms / 1000.0))
        assert features.frame_length == expected_frame
        assert features.hop_length == expected_hop
        assert features.n_frames == frame_count(x.size, expected_frame,
                                                expected_hop)


def test_empty_and_short_inputs_do_not_crash() -> None:
    for n_samples in (0, 1, 100, 399):
        features = frame_features(np.zeros(n_samples), SAMPLE_RATE)
        assert features.n_frames == 0
        assert features.f0_hz.size == 0
        assert features.duration_s == 0.0
        summary = affect_descriptors(features)
        assert all(np.isfinite(list(summary.values())))
        assert VoiceEncoder(8).encode(features).shape == (0, 8)
        assert waveform_to_model_input(
            np.zeros(n_samples), SAMPLE_RATE, 8).shape == (1, 0, 8)

    # Exactly one frame is a boundary case of its own: both the frame count and
    # the "nothing precedes this frame" energy-flux rule land on it.
    one = frame_features(np.zeros(FRAME_LENGTH), SAMPLE_RATE)
    assert one.n_frames == 1
    assert one.energy_flux[0] == 0.0


def test_impossible_requests_are_rejected() -> None:
    with pytest.raises(ValueError, match="1-D signal"):
        frame_features(np.zeros((2, 1_000)), SAMPLE_RATE)
    with pytest.raises(ValueError, match="sample_rate"):
        frame_features(np.zeros(1_000), 0)
    with pytest.raises(ValueError, match="f0_min"):
        frame_features(np.zeros(1_000), SAMPLE_RATE, f0_min=500.0, f0_max=100.0)
    with pytest.raises(ValueError, match="d_model"):
        VoiceEncoder(0)
    with pytest.raises(ValueError, match="asked for"):
        waveform_to_model_input(
            np.zeros(1_000), SAMPLE_RATE, 8, encoder=VoiceEncoder(16)
        )


# --------------------------------------------------------------------------
# The encoder: a projection, and provably nothing more
# --------------------------------------------------------------------------

def test_encoder_is_deterministic_under_its_seed() -> None:
    features = frame_features(tone(180.0, 0.5), SAMPLE_RATE)
    first = VoiceEncoder(16, seed=0).encode(features)
    again = VoiceEncoder(16, seed=0).encode(features)
    other = VoiceEncoder(16, seed=1).encode(features)

    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)
    assert first.shape == (features.n_frames, 16)
    assert np.isfinite(first).all()


def test_encoder_is_exactly_the_linear_map_it_documents() -> None:
    """No bias, no nonlinearity, no hidden normalisation.

    ``encode`` is defined as ``feature_matrix @ projection``; assert it to the
    last bit rather than to a tolerance, because the claim being made is that
    the embedding *cannot* contain anything the interpretable features do not.
    """
    encoder = VoiceEncoder(12, seed=3)
    features = frame_features(tone(220.0, 0.5), SAMPLE_RATE)
    matrix = feature_matrix(features)

    assert encoder.projection.shape == (len(FEATURE_NAMES), 12)
    np.testing.assert_array_equal(encoder.encode(features), matrix @ encoder.projection)

    # A bias term would make silence non-zero; there is none.
    silence = frame_features(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert np.all(encoder.encode(silence) == 0.0)
    assert np.all(feature_matrix(silence) == 0.0)


def test_encoder_distinguishes_inputs_it_is_supposed_to() -> None:
    """A control against a degenerate projection.

    An all-zero encoder, or one whose pitch column is unused, would pass every
    shape and determinism test above. Two pitches an octave apart must land in
    measurably different places, and two utterances at the *same* mean pitch
    but different spread must too.
    """
    encoder = VoiceEncoder(32, seed=0)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE

    def encode(x: np.ndarray) -> np.ndarray:
        return encoder.encode(frame_features(x, SAMPLE_RATE))

    # All one second, so every embedding has the same number of frames and the
    # comparisons below are frame-aligned.
    low = encode(tone(150.0, 1.0))
    high = encode(tone(300.0, 1.0))
    steady = encode(tone(200.0, 1.0))
    vibrato = encode(0.5 * np.sin(2 * np.pi * 200.0 * (
        t + 0.10 / (2 * np.pi * 4.0) * np.sin(2 * np.pi * 4.0 * t)
    )))

    assert low.shape == high.shape == steady.shape == vibrato.shape
    assert np.abs(low - high).max() > 0.05          # pitch level
    assert np.abs(vibrato - steady).max() > 0.05    # pitch spread at one level

    # And the projection really does use the pitch column: zeroing one row of
    # the projection changes the embedding it produces.
    features = frame_features(tone(150.0, 0.5), SAMPLE_RATE)
    matrix = feature_matrix(features)
    ablated = np.array(encoder.projection)
    ablated[FEATURE_NAMES.index("f0_norm")] = 0.0
    assert not np.allclose(matrix @ encoder.projection, matrix @ ablated)


# --------------------------------------------------------------------------
# The bridge into the model
# --------------------------------------------------------------------------

def test_bridge_returns_the_batch_length_dimension_layout() -> None:
    x = tone(220.0, 0.5)
    out = waveform_to_model_input(x, SAMPLE_RATE, d_model=24)
    assert out.shape == (1, frame_count(x.size, FRAME_LENGTH, HOP_LENGTH), 24)
    assert np.isfinite(out).all()

    batch = np.stack([tone(150.0, 0.5), tone(300.0, 0.5)])
    assert waveform_to_model_input(batch, SAMPLE_RATE, 8).shape == (2, 48, 8)


def test_bridge_output_is_consumable_by_the_selective_scan() -> None:
    """End to end into the model, which is the only claim the bridge makes.

    The front end supplies ``x`` only. ``delta``, ``B`` and ``C`` are produced
    inside the SSM block, so this test builds arbitrary ones and checks that the
    scan accepts the array and returns the expected shape.
    """
    torch = pytest.importorskip("torch")
    from beyond_attention.ssm import selective_scan

    x = waveform_to_model_input(tone(220.0, 0.5), SAMPLE_RATE, d_model=8)
    tensor = torch.from_numpy(x).float()
    batch, length, dim = tensor.shape
    assert (batch, length, dim) == (1, 48, 8)

    torch.manual_seed(0)
    delta = torch.full_like(tensor, 0.01)
    a = -torch.ones(dim, 4)
    b = torch.randn(batch, length, 4)
    c = torch.randn(batch, length, 4)
    with torch.no_grad():
        out = selective_scan(tensor, delta, a, b, c)
    assert out.shape == (batch, length, dim)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# A property the descriptors are relied on for
# --------------------------------------------------------------------------

def test_jitter_ignores_the_gap_between_voiced_runs() -> None:
    """Jitter must not count a voicing boundary as a pitch jump.

    Six tone bursts alternating between 150 and 300 Hz, separated by silence.
    Within each burst the pitch is steady, so the adjacent-frame jitter is
    essentially zero (measured 0.099 Hz); the only large pitch steps are the
    five across the pauses, and those are voicing boundaries rather than
    jitter. A naive successive difference over all voiced frames reports
    12.309 Hz here — a factor of 124 — so this test fails loudly if the
    adjacency rule is dropped, and its margin is that ratio rather than a
    tolerance.
    """
    def alternate(freqs: list[float], duty: float = 0.6) -> np.ndarray:
        signal = np.zeros(SAMPLE_RATE)
        span = SAMPLE_RATE // len(freqs)
        for i, freq in enumerate(freqs):
            start = i * span
            stop = start + int(span * duty)
            t = np.arange(stop - start) / SAMPLE_RATE
            signal[start:stop] = 0.5 * np.sin(2 * np.pi * freq * t)
        return signal

    gated = descriptors(alternate([150.0, 300.0] * 3))
    steady_bursts = descriptors(alternate([200.0] * 4))

    assert gated["jitter"] < 1.0
    assert gated["jitter_relative"] < 0.01
    # The signal really does contain 150 Hz steps, so this is not a case of
    # measuring something with nothing in it.
    assert gated["f0_range"] > 140.0
    # Same pitch in every burst: no jitter and no spread at all.
    assert steady_bursts["jitter"] == 0.0
    assert steady_bursts["f0_range"] == 0.0

    # And the descriptor is not constant-zero: a real pitch step shows up.
    half = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    stepped = descriptors(np.concatenate([
        0.5 * np.sin(2 * np.pi * 180.0 * half),
        0.5 * np.sin(2 * np.pi * 240.0 * half),
    ]))
    assert stepped["jitter"] > 1.0
    assert stepped["jitter_relative"] > gated["jitter_relative"]


def test_descriptors_do_not_require_frozen_dataclass_equality() -> None:
    """``VoiceFeatures`` holds arrays, so it compares by identity, not field-wise.

    A dataclass with ``eq=True`` (the default) raises "the truth value of an
    array with more than one element is ambiguous" the first time anyone writes
    ``assert a == b`` on two feature objects. ``eq=False`` makes that comparison
    identity, and ``dataclasses.replace`` on a frozen instance still has to
    work, which is the other half of the contract.
    """
    features = frame_features(tone(200.0, 0.5), SAMPLE_RATE)
    louder = dataclasses.replace(features, rms=features.rms * 2.0)
    assert features == features
    assert features != louder
    assert louder.rms.mean() == pytest.approx(2.0 * features.rms.mean())


def test_search_range_is_clamped_to_the_documented_band() -> None:
    """The F0 search is bounded on both sides, and the bounds are the constants.

    A 30 Hz tone is outside the band, so the estimator either declines or
    reports an octave/aliased value -- what it must never do is report 30 Hz,
    because that would mean the lag range is not actually clamped.
    """
    f0, voiced, _ = autocorrelation_f0(tone(30.0, 0.5)[:FRAME_LENGTH],
                                       SAMPLE_RATE)
    assert not voiced or f0 >= F0_MIN_HZ * 0.95
    assert not voiced or f0 <= F0_MAX_HZ * 1.05

    # Inside the band, a 380 Hz tone is accepted (the band's top is 400 Hz).
    f0_high, voiced_high, _ = autocorrelation_f0(
        tone(380.0, 0.5)[:FRAME_LENGTH], SAMPLE_RATE
    )
    assert voiced_high and abs(f0_high - 380.0) / 380.0 < F0_TOLERANCE
