import numpy as np
import pytest

correct_cardiac_gaps = pytest.importorskip("studies.pain_study.scripts.correct_cardiac_gaps")


def test_benchmark_row_carries_both_arms():
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    beats = 5.0 + np.arange(120) * 0.9
    data = rng.normal(0, 10.0, (4, int(130 * sfreq)))

    rows = correct_cardiac_gaps.benchmark_arrays(
        data, beats, sfreq, methods=("aas",), n_components=(4,), n_surrogate=5
    )

    assert len(rows) == 1
    row = rows[0]
    for field in (
        "method",
        "removal_max",
        "removal_null_max",
        "removal_channels_above_null",
        "sham_alpha_retained",
        "real_alpha_retained",
    ):
        assert field in row


def test_sham_retention_is_higher_than_real_when_artifact_present():
    """The sham must remove less than the real correction, or it is not a control."""
    rng = np.random.default_rng(1)
    sfreq = 1000.0
    beats = 5.0 + np.arange(120) * 0.9
    clean = rng.normal(0, 10.0, (4, int(130 * sfreq)))
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = 60.0 * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    dirty = clean.copy()
    for beat in beats:
        start = int(round(beat * sfreq))
        dirty[:, start : start + shape.size] += shape

    rows = correct_cardiac_gaps.benchmark_arrays(
        dirty, beats, sfreq, methods=("aas",), n_components=(4,), n_surrogate=5
    )

    assert rows[0]["sham_band_retained"] > rows[0]["real_band_retained"]


def test_apply_only_changes_gap_stretches(tmp_path):
    """Everything outside a gap must survive byte-for-byte from Analyzer's output."""
    rng = np.random.default_rng(2)
    sfreq = 1000.0
    n = int(200 * sfreq)
    analyzer_corrected = rng.normal(0, 10.0, (4, n))
    uncorrected = analyzer_corrected + rng.normal(0, 1.0, (4, n))

    out = correct_cardiac_gaps.substitute_gap_stretches(
        analyzer_corrected, uncorrected, [(80.0, 95.0)], sfreq, pad_seconds=0.5
    )

    lo, hi = int(79.5 * sfreq), int(95.5 * sfreq)
    assert np.array_equal(out[:, :lo], analyzer_corrected[:, :lo])
    assert np.array_equal(out[:, hi:], analyzer_corrected[:, hi:])
    assert not np.array_equal(out[:, lo:hi], analyzer_corrected[:, lo:hi])
