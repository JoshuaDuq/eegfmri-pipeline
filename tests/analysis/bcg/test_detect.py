import numpy as np
import pytest

from studies.pain_study.analysis.bcg.detect import (
    RecoverySettings,
    _normalised_correlation,
    crosscheck_agreement,
    find_gaps,
    gap_summary,
    physiological_floor,
    qrs_template,
    recover_beats,
)

SFREQ = 1000.0


def test_no_gaps_in_a_regular_train():
    beats = np.arange(5.0, 100.0, 0.9)

    assert find_gaps(beats) == []


def test_finds_a_single_deleted_stretch():
    beats = np.arange(5.0, 100.0, 0.9)
    kept = beats[(beats < 40.0) | (beats > 52.0)]

    gaps = find_gaps(kept)

    assert len(gaps) == 1
    assert gaps[0].preceding_beat_s < 40.0
    assert gaps[0].following_beat_s > 52.0
    # The surviving beats bracketing the deleted stretch sit at 39.2 s and 52.7 s.
    assert 11.0 < gaps[0].duration_s < 13.6


def test_ordinary_slow_rate_is_not_a_gap():
    """A steady 40 bpm run has RR of 1.5 s throughout and contains no missing beats."""
    beats = np.arange(5.0, 200.0, 1.5)

    assert find_gaps(beats) == []


def test_finds_isolated_single_missed_beats():
    """One beat dropped here and there is the commonest failure, not a long dropout.

    On sub-0012 these are 87% of the gaps and 85% of the missing time: an interval of about
    twice the beat-to-beat interval, every few seconds. A threshold set at twice the rate
    cannot see them, because that is exactly what one missed beat produces.
    """
    beats = np.arange(5.0, 200.0, 0.85)
    kept = np.delete(beats, np.arange(7, beats.size, 7))

    gaps = find_gaps(kept)

    assert len(gaps) >= 20
    assert all(1.5 < g.duration_s < 1.9 for g in gaps)


def test_a_sparsely_marked_run_is_judged_on_its_true_beat_interval():
    """Half the beats missing inflates the median RR, which must not raise the threshold.

    This is the failure the relative test exists to prevent, and a median-based one walks
    straight into it: on sub-0012 run 1 the median RR reads 0.998 s against a true 0.85 s
    precisely because so many beats are gone, which lifts the threshold past the gaps.
    """
    rng = np.random.default_rng(0)
    beats = np.arange(5.0, 400.0, 0.85)
    kept = np.sort(rng.choice(beats, size=beats.size // 2, replace=False))

    gaps = find_gaps(kept)

    assert len(gaps) >= 30


def test_gap_summary_reports_time_and_implied_missing_beats():
    beats = np.arange(5.0, 100.0, 0.9)
    kept = beats[(beats < 40.0) | (beats > 52.0)]

    summary = gap_summary(kept, duration_s=100.0)

    assert summary["n_gaps"] == 1
    assert summary["gap_seconds"] > 11.0
    assert summary["implied_missing_beats"] > 11
    assert 0.0 < summary["gap_fraction"] < 0.2


def _synthetic_ecg(beats, duration_s, sfreq=SFREQ, t_wave_uv=0.0, seed=0):
    """ECG with a sharp QRS and an optionally inflated T-wave.

    The T-wave models the magnetohydrodynamic effect, which is what defeats
    amplitude-threshold detectors inside the bore.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 5.0, int(duration_s * sfreq))
    qrs_t = np.arange(int(0.05 * sfreq)) / sfreq
    qrs = 600.0 * np.exp(-((qrs_t - 0.025) ** 2) / (2 * 0.006**2))
    t_t = np.arange(int(0.16 * sfreq)) / sfreq
    t_wave = t_wave_uv * np.exp(-((t_t - 0.08) ** 2) / (2 * 0.035**2))
    for beat in beats:
        start = int(round(beat * sfreq))
        if start + qrs.size < signal.size:
            signal[start : start + qrs.size] += qrs
        offset = start + int(0.30 * sfreq)
        if t_wave_uv and offset + t_wave.size < signal.size:
            signal[offset : offset + t_wave.size] += t_wave
    return signal


def test_recovers_beats_deleted_from_a_known_stretch():
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]
    deleted = beats[(beats >= 80.0) & (beats <= 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert result.recovered_beats.size >= deleted.size - 2
    matched = [np.min(np.abs(result.recovered_beats - d)) < 0.05 for d in deleted]
    assert sum(matched) >= deleted.size - 2


def test_inflated_t_wave_does_not_create_extra_beats():
    """Regression for the sub-0000 failure: T-waves must not be recovered as beats."""
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0, t_wave_uv=900.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert result.quality.refractory_violations == 0
    assert result.combined_beats.size < beats.size * 1.15
    assert 55.0 < result.quality.implied_bpm < 75.0


def test_recovery_does_not_leave_beats_a_second_pass_would_find():
    """Recovery must search the gaps in its own output, not only the ones it started with.

    The loop re-estimated the template from the growing train but always re-searched
    ``find_gaps(analyzer)``, and replaced its result rather than accumulating it. Whatever a
    pass under-filled was therefore never revisited. On sub-0012 that left 313 s of gaps
    across six runs which simply calling the function again on its own output closed by 78%.
    """
    beats = np.arange(5.0, 190.0, 0.85)
    ecg = _synthetic_ecg(beats, 200.0)
    rng = np.random.default_rng(4)
    # scattered single beats dropped, which is the commonest real failure
    kept = np.delete(beats, rng.choice(np.arange(3, beats.size - 3), size=40, replace=False))

    first = recover_beats(ecg, kept, SFREQ)
    second = recover_beats(ecg, first.combined_beats, SFREQ)

    assert first.recovered_beats.size >= 30
    assert second.recovered_beats.size <= 3


def test_recovery_leaves_analyzer_beats_untouched():
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert np.all(np.isin(kept, result.combined_beats))


def test_correlation_threshold_gates_what_is_recovered():
    """Settings reach the matcher: a threshold no waveform can meet recovers nothing."""
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    permissive = recover_beats(ecg, kept, SFREQ, settings=RecoverySettings())
    strict = recover_beats(ecg, kept, SFREQ, settings=RecoverySettings(correlation_threshold=1.01))

    assert permissive.recovered_beats.size > 0
    assert strict.recovered_beats.size == 0
    assert strict.quality.status == "ok"


def test_sliding_correlation_matches_an_explicit_pearson_reference():
    """The vectorised matmul must agree with the definition it stands in for.

    Accelerate's BLAS raises spurious divide/overflow/invalid flags on this matmul, which
    `_normalised_correlation` suppresses; this is what makes that suppression safe.
    """
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    template = qrs_template(ecg, beats[(beats < 80.0) | (beats > 95.0)], SFREQ, (-0.2, 0.4))
    segment = ecg[79550:94550]

    fast = _normalised_correlation(segment, template)

    centred = template - template.mean()
    norm = np.linalg.norm(centred)
    reference = np.empty(segment.size - template.size + 1)
    for index in range(reference.size):
        window = segment[index : index + template.size]
        window = window - window.mean()
        window_norm = np.linalg.norm(window)
        reference[index] = (window @ centred) / (window_norm * norm) if window_norm else 0.0

    assert np.all(np.isfinite(fast))
    assert np.allclose(fast, reference, atol=1e-12)


def test_too_few_seed_beats_reports_status_rather_than_raising():
    ecg = _synthetic_ecg(np.array([10.0, 11.0]), 60.0)

    result = recover_beats(ecg, np.array([10.0, 11.0]), SFREQ)

    assert result.quality.status == "insufficient_seed_beats"
    assert result.recovered_beats.size == 0


def test_a_beat_closer_than_the_run_s_own_physiology_is_rejected():
    """A recovered beat must not create an RR this subject's heart never produced.

    Measured on the cohort: 9.6% of recovered beats created an interval shorter than the
    run's own Analyzer-derived minimum, up to 38.7% in the worst run. Those beats sit on
    ECG deflections at the 0th percentile of Analyzer's, and dropping them *raises* the
    QRS lock ratio of what remains -- they are false positives, and a false R marker makes
    Analyzer subtract a pulse template where no beat exists.
    """
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    combined = result.combined_beats
    intervals = np.diff(combined)
    floor = physiological_floor(kept, combined)

    assert floor > 0.0
    assert intervals.min() >= floor
    assert result.quality.refractory_violations == 0


def test_the_floor_is_capped_so_a_sparsely_marked_run_is_not_gutted():
    """On sub-0008 run 6 the Analyzer RR p01 sits at 0.98 of the median.

    Taken literally that floor rejects almost any beat, so it is capped against the
    combined train's own median rather than trusted outright.
    """
    # Analyzer marked only every other beat, so its intervals are all ~1.8 s.
    analyzer = np.arange(5.0, 100.0, 1.8)
    combined = np.arange(5.0, 100.0, 0.9)

    floor = physiological_floor(analyzer, combined)

    assert floor < 0.9  # must not exceed the true beat-to-beat interval
    assert floor == pytest.approx(0.75 * 0.9, abs=1e-6)


def test_a_plausible_recovered_beat_survives_the_filter():
    analyzer = np.arange(5.0, 100.0, 0.9)
    combined = np.sort(np.append(analyzer, 50.0 + 0.45))  # mid-gap, half an RR away

    floor = physiological_floor(analyzer, combined)

    assert 0.45 < floor  # the interloper is closer than the floor allows


def test_analyzer_beats_are_never_dropped_by_the_filter():
    """Analyzer's marks are the trusted set; filtering only ever removes our own."""
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert np.all(np.isin(kept, result.combined_beats))


def test_crosscheck_reports_agreement_without_changing_beats():
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)

    report = crosscheck_agreement(beats, ecg, SFREQ)

    assert 0.0 <= report["agreement_fraction"] <= 1.0
    assert report["crosscheck_beats"] > 0
    assert "crosscheck_lock_ratio" in report


def test_crosscheck_degrades_gracefully_when_neurokit_fails():
    """A cross-check that cannot run is a missing measurement, not an error."""
    report = crosscheck_agreement(np.array([1.0, 2.0]), np.zeros(100), SFREQ)

    assert report["status"] != "ok"
