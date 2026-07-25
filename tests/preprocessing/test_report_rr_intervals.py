"""Beat detection evidence must separate variability from detector dropout."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.analyzer_qc import (  # noqa: E402
    add_rr_interval_section,
    compute_rr_intervals,
    plot_rr_intervals,
    rr_intervals_html,
)

SFREQ = 250.0
DURATION = 180.0


def _raw_with_beats(beat_onsets):
    info = mne.create_info(["C0", "C1"], SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (2, int(DURATION * SFREQ))),
        info,
        verbose="ERROR",
    )
    if len(beat_onsets):
        raw.set_annotations(
            mne.Annotations(
                onset=beat_onsets,
                duration=0.0,
                description=["Pulse Artifact/R"] * len(beat_onsets),
            )
        )
    return raw


def _regular_beats(period=0.85, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    onsets = np.cumsum(rng.normal(period, jitter, 200))
    return onsets[onsets < DURATION]


def test_the_median_rate_is_recovered() -> None:
    series = compute_rr_intervals(
        _raw_with_beats(_regular_beats(period=0.75)), recording_id="run-1"
    )

    assert series is not None
    assert series.median_bpm == pytest.approx(80.0, abs=1.0)


def test_variability_alone_is_not_counted_as_dropout() -> None:
    """A participant with a variable rate must not be reported as a failing detector."""
    series = compute_rr_intervals(
        _raw_with_beats(_regular_beats(jitter=0.08)), recording_id="run-1"
    )

    assert series.dropout_count == 0


def test_a_missed_beat_is_counted() -> None:
    beats = _regular_beats()
    # Drop every twentieth beat, which doubles the interval that spans it.
    kept = np.array([beat for index, beat in enumerate(beats) if index % 20 != 0])

    series = compute_rr_intervals(_raw_with_beats(kept), recording_id="run-1")

    assert series.dropout_count >= 8


def test_a_detector_that_fails_partway_shows_in_the_interval_series() -> None:
    beats = _regular_beats()
    survived = np.concatenate([beats[beats < 90.0], beats[beats >= 90.0][::3]])

    series = compute_rr_intervals(_raw_with_beats(survived), recording_id="run-1")

    late = series.intervals_s[series.beat_times_s >= 90.0]
    early = series.intervals_s[series.beat_times_s < 90.0]
    assert float(np.median(late)) > 2.0 * float(np.median(early))
    assert series.dropout_count > 0


def test_a_run_without_markers_yields_no_series() -> None:
    assert compute_rr_intervals(_raw_with_beats([]), recording_id="run-1") is None


def test_too_few_markers_yield_no_series() -> None:
    assert compute_rr_intervals(_raw_with_beats([1.0, 2.0]), recording_id="run-1") is None


def test_the_section_renders_and_replaces_on_rebuild() -> None:
    series = compute_rr_intervals(_raw_with_beats(_regular_beats()), recording_id="run-1")
    report = mne.Report(title="analyzer", verbose="ERROR")

    add_rr_interval_section(report=report, series=[series])
    add_rr_interval_section(report=report, series=[series])

    assert len(report._content) == 2
    assert all("rr-intervals" in element.tags for element in report._content)
    assert "run-1" in rr_intervals_html([series])
    assert plot_rr_intervals([series]).axes


def test_an_empty_section_is_an_error_rather_than_a_blank_panel() -> None:
    report = mne.Report(title="analyzer", verbose="ERROR")

    with pytest.raises(ValueError, match="at least one run"):
        add_rr_interval_section(report=report, series=[])
