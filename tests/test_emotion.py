"""The trained emotion classifier, checked against ground truth and against itself.

Two kinds of test, the same two as `tests/test_voice.py`, because this module
makes both kinds of claim.

**Analytic ground truth.** An HNR of a pure tone, the modulation rate of a
sinusoidal pitch contour, the slope of a known chirp, the rise time of a gated
onset: each has one right answer that is known before the code runs, and each is
asserted to a stated tolerance.

**Controls that can fail.** A classifier test that only asserts "held-out
accuracy is high" cannot distinguish learning from a pipeline that leaks labels,
and this repository has already published one headline that a control
falsified. So the training loop is tested on a planted cluster task (it must
learn), on shuffled labels and on Gaussian noise (it must *not* learn), and the
published numbers are re-derived from the committed JSON so that a table drifting
from its run fails the suite.

**The disclaimer is a test.** Two tests read `README.md` directly: one requires
the limitation sentences to be present, the other requires the rendered
emotion block to be byte-identical to a fresh render of the committed JSON. A
disclaimer that can be deleted silently is not a disclaimer, and a published
number that can be edited by hand is not a measurement.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

from beyond_attention.emotion import (
    CONDITIONS,
    CONDITION_NAMES,
    EMOTION_FEATURES,
    F0_FAMILY,
    FEATURE_FAMILIES,
    HNR_MAX_DB,
    PITCH_DERIVED_FAMILY,
    Dataset,
    ablation,
    build_dataset,
    correlate_checks,
    dominant_modulation_hz,
    emotion_descriptors,
    frame_hnr_db,
    greedy_forward_selection,
    grouped_means,
    random_feature_control,
    shuffle_label_control,
    stratified_split,
    synthesise,
    train_classifier,
    unvoiced_runs,
)
from beyond_attention.voice import frame_features

SAMPLE_RATE = 16_000
REPO = pathlib.Path(__file__).resolve().parent.parent
README = REPO / "README.md"
RESULTS = REPO / "emotion-classifier.json"

# The margin the published held-out accuracy must beat chance by. Stated here,
# in the test, because an unstated margin is not a claim: the README publishes
# 1.000 against 0.200, and this asserts at least 0.30 above chance so that a
# regression to a barely-better-than-guessing model fails loudly.
REQUIRED_MARGIN_OVER_CHANCE = 0.30


# --------------------------------------------------------------------------
# Analytic ground truth: HNR
# --------------------------------------------------------------------------

def tone(freq_hz: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(round(seconds * SAMPLE_RATE))) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def test_hnr_is_high_for_a_periodic_frame_and_falls_as_noise_is_added() -> None:
    """HNR against an analytic value, and monotone in the noise ratio.

    A 200 Hz tone at 16 kHz has an exactly 80-sample period, so its HNR is at
    the estimator's cap. Adding white noise of standard deviation ``sigma`` to a
    tone of amplitude ``A`` makes the normalised autocorrelation at the period
    ``(A^2/2) / (A^2/2 + sigma^2)`` -- the tone's mean power over the total --
    so the analytic HNR is ``10 log10( (A^2/2) / sigma^2 )``. The two quiet-noise
    cases are asserted against that prediction to within 1.5 dB; the loudest is
    only asserted to be lower, because near a correlation of 0.5 the dB is
    sensitive enough (about 17 dB per unit of correlation) that a single noise
    realisation moves it by more than a decibel.
    """
    amplitude = 0.5
    frame = tone(200.0, 0.5, amplitude=amplitude)[:400]
    rng = np.random.default_rng(0)
    cleaned = [
        frame + rng.standard_normal(400) * sigma for sigma in (0.05, 0.15, 0.30)
    ]
    noisy = [frame_hnr_db(row, SAMPLE_RATE, 200.0) for row in cleaned]

    clean = frame_hnr_db(frame, SAMPLE_RATE, 200.0)
    assert clean > 25.0, f"clean tone HNR {clean:.1f} dB"
    assert clean <= HNR_MAX_DB

    predicted = [
        10.0 * np.log10((amplitude**2 / 2.0) / sigma**2)
        for sigma in (0.05, 0.15)
    ]
    assert noisy[0] == pytest.approx(predicted[0], abs=1.5)
    assert noisy[1] == pytest.approx(predicted[1], abs=1.5)

    assert clean > noisy[0] > noisy[1] > noisy[2]


def test_hnr_declines_to_measure_without_a_pitch() -> None:
    """No pitch, no HNR: 0.0 is a placeholder, not a measurement."""
    frame = tone(200.0, 0.5)[:400]
    assert frame_hnr_db(frame, SAMPLE_RATE, float("nan")) == 0.0
    assert frame_hnr_db(frame, SAMPLE_RATE, 0.0) == 0.0
    assert frame_hnr_db(np.zeros(400), SAMPLE_RATE, 200.0) == 0.0
    assert frame_hnr_db(frame[:3], SAMPLE_RATE, 200.0) == 0.0


# --------------------------------------------------------------------------
# Analytic ground truth: modulation rate, slope, terminal fall
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rate_hz", [1.5, 4.0, 7.0])
def test_dominant_modulation_recovers_a_known_sinusoid(rate_hz: float) -> None:
    """A pure sinusoidal contour has a modulation rate that is known exactly."""
    frame_rate = 100.0
    t = np.arange(int(2.0 * frame_rate)) / frame_rate
    contour = 2.0 * np.sin(2 * np.pi * rate_hz * t)
    assert dominant_modulation_hz(contour, frame_rate) == pytest.approx(
        rate_hz, abs=0.15
    )


def test_dominant_modulation_is_zero_when_there_is_nothing_to_find() -> None:
    frame_rate = 100.0
    assert dominant_modulation_hz(np.full(200, 3.0), frame_rate) == 0.0
    assert dominant_modulation_hz(np.arange(5.0), frame_rate) == 0.0
    # A pure ramp has no in-band peak after detrending.
    assert dominant_modulation_hz(np.arange(200.0), frame_rate) == 0.0


@pytest.mark.parametrize("semitones_per_second", [-6.0, 3.0])
def test_f0_slope_recovers_a_known_chirp(semitones_per_second: float) -> None:
    """A chirp that is linear in log frequency has an exactly known slope.

    Built by integrating the instantaneous frequency, so the signal really does
    glide at the stated rate rather than being a frequency-modulated
    approximation of one.
    """
    seconds = 2.0
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    base = 220.0
    instantaneous = base * np.exp2(semitones_per_second * t / 12.0)
    phase = 2 * np.pi * np.cumsum(instantaneous) / SAMPLE_RATE
    summary = emotion_descriptors(0.5 * np.sin(phase), SAMPLE_RATE)

    assert summary["f0_slope_st_per_s"] == pytest.approx(
        semitones_per_second, abs=0.5
    )


def test_final_ratio_is_below_one_for_a_terminal_fall() -> None:
    """A pitch that drops in its last third must read below 1."""
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    steady = emotion_descriptors(0.5 * np.sin(2 * np.pi * 250.0 * t), SAMPLE_RATE)

    semitones = np.where(t < 1.4, 0.0, -12.0 * (t - 1.4) / 0.6)
    phase = 2 * np.pi * np.cumsum(250.0 * np.exp2(semitones / 12.0)) / SAMPLE_RATE
    falling = emotion_descriptors(0.5 * np.sin(phase), SAMPLE_RATE)

    assert steady["f0_final_ratio"] == pytest.approx(1.0, abs=0.02)
    assert falling["f0_final_ratio"] < 0.90
    assert falling["f0_final_ratio"] < steady["f0_final_ratio"]


# --------------------------------------------------------------------------
# Analytic ground truth: pause structure, onset, shimmer
# --------------------------------------------------------------------------

def test_unvoiced_runs_are_the_gaps_exactly() -> None:
    """Run detection is arithmetic on a boolean array, so it is pinned exactly."""
    cases = [
        # A signal that is entirely unvoiced has one unvoiced run covering it;
        # a signal that is entirely voiced has none.
        (np.array([False, False, False]), [(0, 3)]),
        (np.array([True, True, True]), []),
        (np.array([False, True, True, False]), [(0, 1), (3, 4)]),
        (np.array([True, False, False, True]), [(1, 3)]),
        (np.array([False, True, False, True, False]), [(0, 1), (2, 3), (4, 5)]),
        (np.array([True, False, True, False]), [(1, 2), (3, 4)]),
        (np.array([True, True]), []),
    ]
    for voiced, expected in cases:
        assert unvoiced_runs(voiced) == expected, voiced


def test_pause_structure_counts_internal_gaps_not_leading_silence() -> None:
    """Two bursts with a 300 ms gap, plus half a second of silence on each end.

    The internal gap is a pause; the leading and trailing silence is not. A
    feature that counted all unvoiced runs would report three.
    """
    signal = np.zeros(2 * SAMPLE_RATE)
    for start in (0.5, 1.2):
        span = np.arange(int(0.4 * SAMPLE_RATE)) / SAMPLE_RATE
        offset = int(start * SAMPLE_RATE)
        signal[offset:offset + span.size] = 0.5 * np.sin(2 * np.pi * 200.0 * span)
    summary = emotion_descriptors(signal, SAMPLE_RATE)

    assert summary["pause_rate"] == pytest.approx(1.0 / 2.0, abs=0.6)
    assert 0.05 < summary["pause_fraction"] < 0.20
    assert summary["pause_mean_s"] == pytest.approx(0.29, abs=0.06)


def test_a_gated_onset_is_sharper_than_a_sixty_millisecond_swell() -> None:
    """The onset measure against a known envelope, not against a condition name.

    A 25 ms window against a 10 ms hop overlaps by 15 ms, so a gated onset
    cannot reach 1.0: measured 0.39 for the gate against 0.16 for the swell, a
    factor of 2.4. The assertion is on that ratio, which is the part that does
    not depend on the window, plus an absolute bound on the swell.
    """
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * 200.0 * t)

    gate = carrier * (t >= 0.2)
    swell = carrier * np.clip((t - 0.2) / 0.06, 0.0, 1.0) * (t >= 0.2)

    gated = emotion_descriptors(gate, SAMPLE_RATE)["onset_sharpness"]
    soft = emotion_descriptors(swell, SAMPLE_RATE)["onset_sharpness"]

    assert gated > 1.8 * soft
    assert soft < 0.25
    assert 0.3 < gated < 0.6


def test_shimmer_is_zero_for_a_steady_envelope_and_positive_for_a_jagged_one() -> None:
    """The distinction ``energy_std_voiced`` cannot make.

    Both signals have the same *level*; only one has frame-to-frame amplitude
    perturbation. A feature that were just re-measuring the energy spread could
    not tell them apart, and this one must.
    """
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    base = 0.5 * np.sin(2 * np.pi * 200.0 * t)
    rng = np.random.default_rng(0)
    frames = frame_features(base, SAMPLE_RATE).n_frames
    per_frame = np.maximum(0.2, 1.0 + 0.5 * rng.standard_normal(frames))
    jagged = base * per_frame[np.minimum(np.arange(base.size) // 160, frames - 1)]

    steady_summary = emotion_descriptors(base, SAMPLE_RATE)
    jagged_summary = emotion_descriptors(jagged, SAMPLE_RATE)

    assert steady_summary["shimmer"] < 0.01
    assert jagged_summary["shimmer"] > 0.10
    # The mean level is close, so this is perturbation rather than loudness.
    assert jagged_summary["energy_mean_voiced"] == pytest.approx(
        steady_summary["energy_mean_voiced"], rel=0.25
    )


def test_every_descriptor_is_finite_including_for_silence() -> None:
    """Silence is the degenerate input the classifier must not see a nan from."""
    summary = emotion_descriptors(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert set(summary) == set(EMOTION_FEATURES)
    assert all(np.isfinite(value) for value in summary.values())
    assert summary["voiced_ratio"] == 0.0
    # The placeholders are documented, and voiced_ratio is the flag for them.
    assert summary["hnr_db"] == 0.0
    assert summary["f0_final_ratio"] == 1.0


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset() -> Dataset:
    """Ten utterances per condition, short, built once for the module.

    One second each rather than two keeps the whole file's audio work under a
    few seconds; the correlate checks are about ordering, which does not depend
    on the utterance length.
    """
    return build_dataset(utterances=10, seconds=1.0, seed=0)


def test_generator_is_deterministic_under_its_rng() -> None:
    condition = CONDITIONS[0]
    first = synthesise(condition, 1.0, SAMPLE_RATE, np.random.default_rng(0))
    again = synthesise(condition, 1.0, SAMPLE_RATE, np.random.default_rng(0))
    other = synthesise(condition, 1.0, SAMPLE_RATE, np.random.default_rng(1))

    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)
    assert first.size == SAMPLE_RATE
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) > 0.0


def test_generator_rejects_impossible_requests() -> None:
    condition = CONDITIONS[0]
    rng = np.random.default_rng(0)
    for seconds, sample_rate, message in (
        (1.0, 0, "sample_rate"),
        (0.0, SAMPLE_RATE, "seconds"),
        (-1.0, SAMPLE_RATE, "seconds"),
    ):
        with pytest.raises(ValueError, match=message):
            synthesise(condition, seconds, sample_rate, rng)


def test_every_condition_shows_the_correlates_it_advertises(
    small_dataset: Dataset,
) -> None:
    """The check that makes the condition names mean something.

    Each condition is built from published acoustic correlates; this asserts
    that the signal the generator produced actually shows the profile its
    citation names. A failure here would mean every accuracy in the README is a
    statement about a condition that is not the one advertised.
    """
    checks = correlate_checks(grouped_means(small_dataset))
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"documented correlates not measured: {failed}"
    assert len(checks) >= 17


def test_conditions_are_distinct_points_in_descriptor_space(
    small_dataset: Dataset,
) -> None:
    """No two conditions are the same signal under a different name."""
    means = grouped_means(small_dataset)
    matrix = np.array([
        [means[name][key] for key in EMOTION_FEATURES] for name in CONDITION_NAMES
    ])
    z = (matrix - matrix.mean(axis=0)) / np.where(
        matrix.std(axis=0) > 0.0, matrix.std(axis=0), 1.0
    )
    for i in range(len(CONDITION_NAMES)):
        for j in range(i + 1, len(CONDITION_NAMES)):
            distance = float(np.linalg.norm(z[i] - z[j]))
            assert distance > 0.5, (
                f"{CONDITION_NAMES[i]} and {CONDITION_NAMES[j]} are "
                f"{distance:.3f} apart in z-scored descriptor space"
            )


def test_the_feature_families_partition_the_feature_set() -> None:
    """Every feature is in exactly one family, and the families are complete.

    The ablation table's "only family X" rows are only interpretable if the
    families cover the feature set without overlap; a feature added without a
    family would be silently un-ablatable.
    """
    seen: list[str] = []
    for members in FEATURE_FAMILIES.values():
        seen.extend(members)
    assert len(seen) == len(set(seen)), "a feature is in two families"
    assert set(seen) == set(EMOTION_FEATURES)
    assert set(F0_FAMILY) <= set(PITCH_DERIVED_FAMILY)


# --------------------------------------------------------------------------
# The training loop, on a task whose answer is planted
# --------------------------------------------------------------------------

def planted_dataset(
    n_per_class: int = 20, n_features: int = 6, seed: int = 0
) -> Dataset:
    """Five Gaussian clusters with a 2.5-sigma separation: learnable, not free."""
    rng = np.random.default_rng(seed)
    names = ["a", "b", "c", "d", "e"]
    centres = rng.normal(0.0, 2.5, size=(len(names), n_features))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, name in enumerate(names):
        rows.append(centres[index] + rng.normal(0.0, 1.0, size=(n_per_class, n_features)))
        labels.extend([name] * n_per_class)
    return Dataset(
        features=np.concatenate(rows),
        labels=np.array(labels, dtype=object),
        conditions=tuple(names),
    )


def test_the_training_loop_learns_a_planted_task() -> None:
    """The positive control for every "it fell to chance" result below.

    If this test fails, the shuffle and noise controls are meaningless: they
    would be at chance because the loop cannot learn anything, not because the
    labels were destroyed.
    """
    dataset = planted_dataset()
    split = stratified_split(dataset.labels, seed=0)
    classifier = train_classifier(dataset, split=split, seed=0, steps=400)

    assert classifier.test_accuracy > 0.9
    assert classifier.train_accuracy > 0.9
    assert classifier.val_accuracy > 0.9
    assert classifier.selected_epoch >= 0


def test_training_is_deterministic_under_a_fixed_seed() -> None:
    """Two runs of the identical recipe must agree exactly, not approximately.

    Determinism is the reason the published JSON can be regenerated and
    compared byte for byte, so it is asserted on the predictions and on the
    selected epoch rather than on a loss curve.
    """
    dataset = planted_dataset()
    split = stratified_split(dataset.labels, seed=0)
    first = train_classifier(dataset, split=split, seed=0, steps=300)
    again = train_classifier(dataset, split=split, seed=0, steps=300)
    other = train_classifier(dataset, split=split, seed=1, steps=300)

    np.testing.assert_array_equal(
        first.predict(dataset.features), again.predict(dataset.features)
    )
    assert first.selected_epoch == again.selected_epoch
    assert first.test_accuracy == again.test_accuracy
    assert first.train_accuracy == again.train_accuracy
    # And a different seed is a different model, so the equality above is not
    # the equality of two constants.
    assert not np.allclose(first.weights[-1], other.weights[-1])


def test_a_single_class_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="too few rows"):
        stratified_split(np.array(["a", "a"], dtype=object), seed=0)
    for fractions in ((0.9, 0.2, 0.2), (0.0, 0.2, 0.2), (0.6, -0.1, 0.5)):
        with pytest.raises(ValueError, match="fraction|test set"):
            stratified_split(
                np.array(["a"] * 6 + ["b"] * 6, dtype=object),
                seed=0, fractions=fractions,
            )


def test_every_class_reaches_every_split() -> None:
    """The confusion matrix must not have a silently empty row."""
    dataset = planted_dataset(n_per_class=7)
    split = stratified_split(dataset.labels, seed=0)
    for rows in (split.train, split.val, split.test):
        assert set(dataset.labels[rows]) == set(dataset.conditions)


def test_shuffle_control_falls_to_chance() -> None:
    """The control the whole section rests on.

    The same architecture, split and step budget, on permuted labels. If this
    does not fall to chance the pipeline is leaking the labels into the
    features, and no accuracy in the README would mean anything.
    """
    dataset = planted_dataset()
    split = stratified_split(dataset.labels, seed=0)
    result = shuffle_label_control(
        dataset, split, rounds=3, control_seed=0, seed=0, steps=200
    )
    chance = 1.0 / len(dataset.conditions)

    assert result["mean"] < chance + 0.15, result
    # And the real task is still learned by the identical recipe, so this is not
    # a loop that has stopped working.
    assert train_classifier(
        dataset, split=split, seed=0, steps=400
    ).test_accuracy > 0.9


def test_random_feature_control_is_at_chance() -> None:
    """26 random columns must not carry a five-class task."""
    dataset = planted_dataset()
    split = stratified_split(dataset.labels, seed=0)
    result = random_feature_control(
        dataset, split, rounds=3, control_seed=0, seed=0, steps=200
    )
    chance = 1.0 / len(dataset.conditions)

    assert result["mean"] < chance + 0.15, result
    assert result["max"] < chance + 0.25, result


def test_ablation_rejects_an_unknown_feature_name() -> None:
    """An ablation that silently removes nothing is worse than one that fails."""
    dataset = planted_dataset()
    split = stratified_split(dataset.labels, seed=0)
    with pytest.raises(ValueError, match="unknown feature"):
        ablation(dataset, split, removed=("not_a_feature",))
    with pytest.raises(ValueError, match="unknown feature"):
        ablation(dataset, split, kept=("not_a_feature",))


def test_greedy_selection_stops_and_reports_what_it_needed() -> None:
    dataset = planted_dataset()
    result = greedy_forward_selection(
        dataset, target=("a", "b"), seed=0, steps=200, max_features=3
    )
    assert 1 <= result["n_features_needed"] <= 3
    assert len(result["features"]) == result["n_features_needed"]
    assert result["final_test_accuracy"] > 0.8
    assert result["stopped_because"] in {
        "validation_perfect", "no_improvement", "max_features", "exhausted"
    }


# --------------------------------------------------------------------------
# The published results, re-derived from the committed JSON
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def published() -> dict:
    if not RESULTS.exists():
        pytest.fail(
            f"{RESULTS.name} is missing; regenerate it with "
            f"`python experiments/emotion_classifier.py` before running the "
            f"suite, because the README is rendered from it"
        )
    return json.loads(RESULTS.read_text())


def test_the_published_run_is_not_a_quick_smoke_test(published: dict) -> None:
    """The README's numbers must come from the full configuration."""
    config = published["config"]
    assert not config["quick"]
    assert config["utterances_per_condition"] >= 20
    assert config["n_utterances"] == (
        config["utterances_per_condition"] * config["n_conditions"]
    )
    assert config["split"]["test"] >= 20
    assert config["n_features"] == len(EMOTION_FEATURES)


def test_published_held_out_accuracy_beats_chance_by_the_stated_margin(
    published: dict,
) -> None:
    """Held-out accuracy, over the stated margin, with the test set the size it is.

    The margin is `REQUIRED_MARGIN_OVER_CHANCE`, written at the top of this
    file. The assertion is on the *test* split, which was read once by
    `train_classifier`; the validation accuracy is asserted separately so that
    a run which overfits validation cannot pass by accident here.
    """
    training = published["training"]
    chance = training["chance"]

    assert chance == pytest.approx(1.0 / published["config"]["n_conditions"])
    assert training["test_accuracy"] >= chance + REQUIRED_MARGIN_OVER_CHANCE
    assert training["val_accuracy"] >= chance + REQUIRED_MARGIN_OVER_CHANCE
    assert training["n_test"] >= 20

    # The baselines are recorded next to it, so "better than chance" cannot be
    # confused with "better than a model that ignores its input".
    assert training["majority_baseline"] == pytest.approx(chance, abs=0.05)
    assert "nearest_centroid_baseline" in training


def test_published_confusion_matrix_is_consistent(published: dict) -> None:
    training = published["training"]
    classes = training["classes"]
    matrix = training["confusion"]

    assert len(matrix) == len(classes)
    assert all(len(row) == len(classes) for row in matrix)
    assert sum(sum(row) for row in matrix) == training["n_test"]
    for name, row in zip(classes, matrix):
        assert sum(row) == published["config"]["split"]["test"] // len(classes)
        assert training["per_class_recall"][name] == pytest.approx(
            row[classes.index(name)] / sum(row)
        )


def test_published_controls_are_at_chance(published: dict) -> None:
    """Shuffled labels and noise features must both land on the floor."""
    chance = published["training"]["chance"]
    controls = published["controls"]

    shuffle = controls["label_shuffle"]
    assert shuffle["rounds"] >= 20
    assert shuffle["mean"] < chance + 0.10, shuffle
    assert shuffle["p95"] < chance + 0.30, shuffle

    noise = controls["random_features"]
    assert noise["mean"] < chance + 0.10, noise
    assert noise["max"] < chance + 0.20, noise


def test_published_correlate_checks_all_hold(published: dict) -> None:
    checks = published["correlate_checks"]
    assert checks["all_pass"]
    assert all(checks["checks"].values())
    assert len(checks["checks"]) >= 17


def test_published_cross_condition_result_shows_no_generalisation(
    published: dict,
) -> None:
    """The honest negative, pinned so it cannot be quietly dropped.

    A held-out condition is not in the trained label space, so its accuracy is 0
    by construction. What the test asserts is the *shape* of the failure: the
    model puts most of an unseen condition onto a single training class it is
    confident about, instead of spreading its answers or abstaining.
    """
    cross = published["cross_condition"]
    per_condition = cross["per_condition"]
    assert set(per_condition) == set(published["training"]["classes"])

    assert cross["mean_dominant_fraction"] > 0.6
    for name, entry in per_condition.items():
        assert entry["dominant_fraction"] >= 0.45, name
        assert entry["assigned_class_nll"] >= 0.0
        assert entry["distance_to_nearest_training_centroid"] > 1.0, name


def test_published_f0_ablation_did_not_collapse_the_accuracy(
    published: dict,
) -> None:
    """A negative result, published as one and asserted so it stays honest.

    Removing the F0 level/contour family, and even removing everything
    pitch-derived, leaves the accuracy essentially where it was. That is what
    the run measured, and the README says so: with six acoustic axes varied at
    once, no single family is necessary. A future change that made the
    classifier depend on one family would change this number, and this test
    would fail rather than let the README's claim go stale.
    """
    ablations = published["ablations"]
    chance = published["training"]["chance"]
    for key in ("without_f0_family", "without_pitch_derived"):
        assert ablations[key]["test_accuracy"] >= chance + REQUIRED_MARGIN_OVER_CHANCE
        assert ablations[key]["n_features"] < published["config"]["n_features"]


def test_published_cry_vs_excited_ablation_is_redundant(published: dict) -> None:
    """The answer to "which feature tells crying from excitement".

    The run measured a drop of 0.000 for every one of the 26 features and 20
    features that separate the pair perfectly on their own, so the answer is
    that there is no single load-bearing feature. The greedy selection then
    reports the smallest set that works.
    """
    cry = published["crying_vs_excited"]
    assert cry["test_accuracy"] >= 0.5 + REQUIRED_MARGIN_OVER_CHANCE
    assert sum(sum(row) for row in cry["confusion"]) == cry["n_test"]

    drops = [row["drop"] for row in cry["per_feature"].values()]
    assert max(drops) <= 1e-9, "a single feature is now load-bearing"

    perfect_alone = [
        name for name, row in cry["per_feature"].items()
        if row["single_accuracy"] >= 1.0
    ]
    assert len(perfect_alone) >= 10

    greedy = cry["greedy_forward"]
    assert 1 <= greedy["n_features_needed"] <= len(EMOTION_FEATURES)
    assert greedy["final_val_accuracy"] >= 0.5 + REQUIRED_MARGIN_OVER_CHANCE


# --------------------------------------------------------------------------
# The rendered README, and the disclaimer inside it
# --------------------------------------------------------------------------

def load_render_readme():
    """Import `experiments/render_readme.py` without installing it as a package."""
    import importlib.util

    path = REPO / "experiments" / "render_readme.py"
    spec = importlib.util.spec_from_file_location("render_readme", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_readme"] = module
    spec.loader.exec_module(module)
    return module


def test_the_rendered_emotion_block_matches_a_fresh_render() -> None:
    """The published block is the renderer's output, not a hand-edited table.

    This is the guard that makes every number in the README traceable to the
    JSON: editing the block by hand, or regenerating the JSON without
    re-rendering, fails here.
    """
    module = load_render_readme()
    text = README.read_text()
    assert module.EMOTION_BEGIN in text and module.EMOTION_END in text

    block = text.split(module.EMOTION_BEGIN, 1)[1].split(module.EMOTION_END, 1)[0]
    rendered = module.emotion_section(json.loads(RESULTS.read_text()))
    assert block.strip("\n") == rendered.strip("\n")


def test_the_render_is_deterministic_back_to_back() -> None:
    module = load_render_readme()
    payload = json.loads(RESULTS.read_text())
    first = module.emotion_section(payload)
    second = module.emotion_section(json.loads(RESULTS.read_text()))
    assert first == second
    for marker in ("| condition |", "| control |", "| held out |"):
        assert marker in first


def readme_plain_text() -> str:
    """The README with markdown emphasis removed and whitespace collapsed.

    A disclaimer is prose, and requiring an exact byte sequence would make the
    test fail on a reflowed paragraph or a word moved inside `**bold**` while
    the disclaimer itself was still there. Removing the markers and normalising
    whitespace leaves the sentences the test actually cares about, still in the
    order they appear, and still impossible to delete silently.
    """
    import re

    text = README.read_text()
    text = re.sub(r"[*`#>]", "", text)
    return re.sub(r"\s+", " ", text)


def test_the_readme_states_the_synthetic_label_limitation() -> None:
    """The disclaimer is a test, not a paragraph.

    The single most important sentence in the deliverable is that the labels
    are the generator's and the classifier therefore learns our acoustic model
    of these emotions rather than a listener's. A test that requires it to be
    present is what stops it being deleted silently later.
    """
    text = readme_plain_text()
    required = [
        "The labels are the synthesiser's",
        "learns our acoustic model of these emotions, not a listener's",
        "It is not validated on speech",
        "not evidence that it recognises emotion in a voice",
        "Cross-condition generalisation",
        "does not abstain",
    ]
    missing = [phrase for phrase in required if phrase not in text]
    assert not missing, f"the README no longer states: {missing}"


def test_the_readme_answers_all_three_false_claims() -> None:
    """Each claim the operator would make, quoted, and answered with a number."""
    text = readme_plain_text()
    lowered = text.lower()
    for claim in (
        "it understands emotion",
        "it can tell crying from excitement",
        "this transfers to real speech",
    ):
        assert claim in lowered, f"the false claim {claim!r} is not answered"
    # Each answer must be attached to a measured number rather than prose alone.
    for anchor in ("held-out accuracy", "confusion", "ablation"):
        assert anchor in lowered


def test_the_readme_prose_numbers_match_the_json(published: dict) -> None:
    """The prose quotes counts too, so the counts are pinned to the JSON.

    Tables are rendered, but the bullets above the block name a few counts in
    words. A count typed into prose drifts from its run exactly like a table
    cell does, so the two counts that appear in the prose are recomputed here
    and required to appear verbatim.
    """
    text = README.read_text()

    perfect_alone = sum(
        1 for row in published["crying_vs_excited"]["per_feature"].values()
        if row["single_accuracy"] >= 1.0
    )
    assert f"{perfect_alone} of the 26 features separate the pair perfectly" in text

    perfect_five = sum(
        1 for row in published["per_feature"].values()
        if row["only_accuracy"] >= 1.0
    )
    assert f"only {perfect_five} of the 26 features reach 1.000" in text

    # The qualitative claims the prose makes about the block's own columns.
    descriptors = published["conditions"]
    assert descriptors["crying"]["descriptors"]["jitter"] == max(
        descriptors[name]["descriptors"]["jitter"] for name in descriptors
    )
    assert descriptors["crying"]["descriptors"]["hnr_db"] == min(
        descriptors[name]["descriptors"]["hnr_db"] for name in descriptors
    )
    assert (
        descriptors["afraid"]["descriptors"]["centroid_mean"]
        > descriptors["angry"]["descriptors"]["centroid_mean"]
        > descriptors["calm"]["descriptors"]["centroid_mean"]
    )
    # The tremor feature's documented failure: the estimate is far from the
    # generator's setting on crying and close to it on the clean conditions.
    tremor = {
        name: (
            descriptors[name]["descriptors"]["tremor_rate_hz"],
            descriptors[name]["parameters"]["tremor_hz"],
        )
        for name in descriptors
    }
    assert abs(tremor["crying"][0] - tremor["crying"][1]) > 4.0
    for name in ("excited", "angry", "afraid", "calm"):
        assert abs(tremor[name][0] - tremor[name][1]) < 0.5, name


def test_the_readme_reports_the_cross_condition_failure_in_the_block() -> None:
    """The failure is in the rendered block, not only in the prose."""
    module = load_render_readme()
    text = README.read_text()
    block = text.split(module.EMOTION_BEGIN, 1)[1].split(module.EMOTION_END, 1)[0]
    assert "Mean dominant fraction" in block
    assert "Cross-condition generalisation" in block
    assert "does not abstain" in block


def test_the_renderer_refuses_a_payload_with_a_missing_section() -> None:
    """The guard, exercised: a truncated payload must not render as zeros.

    An empty table reads as a measurement of zero, and a reader cannot tell it
    from a truncated file, so the renderer raises and `main` refuses to write
    rather than publishing a block of placeholder numbers.
    """
    module = load_render_readme()
    for key in ("cross_condition", "training", "crying_vs_excited"):
        payload = json.loads(RESULTS.read_text())
        payload.pop(key)
        with pytest.raises(ValueError, match=key):
            module.emotion_section(payload)

    # And a complete payload still renders, so the guard is not rejecting
    # everything.
    rendered = module.emotion_section(json.loads(RESULTS.read_text()))
    assert "| held out |" in rendered
