import numpy as np

from eeg_pipeline.preprocessing.bcg.detect import find_gaps, gap_summary


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
