import numpy as np

from eeg_pipeline.preprocessing.bcg.detect import (
    RecoverySettings,
    _normalised_correlation,
    find_gaps,
    gap_summary,
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
