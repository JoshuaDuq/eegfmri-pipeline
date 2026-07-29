from __future__ import annotations

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.pulse_artifact_qc import (
    PulseMarkerCriteria,
    measure_pulse_markers,
    pulse_marker_bound_notes,
    summarize_pulse_marker_recordings,
)


def _raw_with_pulse_markers(onsets: np.ndarray, *, duration_seconds: float = 100.0):
    sfreq = 100.0
    info = mne.create_info(["Cz", "ECG"], sfreq=sfreq, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(
        np.zeros((2, int(duration_seconds * sfreq))),
        info,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=np.zeros(len(onsets)),
            description=["Pulse Artifact/R"] * len(onsets),
        )
    )
    return raw


CRITERIA = PulseMarkerCriteria(
    minimum_bpm=45.0,
    maximum_bpm=80.0,
    minimum_marker_fraction=0.8,
    minimum_recording_coverage=0.8,
)


def test_measurement_reports_run_metrics_and_what_they_derive_from() -> None:
    onsets = np.arange(5.0, 100.0, 1.0)

    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    assert metrics.recording_id == "sub-0001_run-1"
    assert metrics.marker_count == 95
    assert metrics.median_bpm == pytest.approx(60.0)
    assert metrics.marker_fraction == pytest.approx(1.0)
    assert metrics.recording_coverage == pytest.approx(0.94)
    # Carried so a reader can recompute the derived quantities their own way.
    assert metrics.duration_seconds == pytest.approx(100.0)
    assert metrics.expected_marker_count == 95


def test_measurement_does_not_stop_at_the_first_bound_a_run_falls_outside() -> None:
    """The defect this replaced: a run outside one bound recorded nothing else.

    On this cohort that blanked every measurement for 24 of 90 runs, so the runs whose
    coverage most needed describing were the ones the table described least.
    """
    sparse = np.array([10.0, 40.0, 70.0])  # ~2 bpm, and covering 60% of the recording

    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(sparse), recording_id="sub-0001_run-1"
    )

    assert metrics.marker_count == 3
    assert np.isfinite(metrics.median_bpm)
    assert np.isfinite(metrics.marker_fraction)
    assert metrics.recording_coverage == pytest.approx(0.6)

    notes = pulse_marker_bound_notes(metrics, CRITERIA)
    # Both the rate and the coverage are reported, not just whichever came first.
    assert any("median rate" in note for note in notes)
    assert any("covers" in note for note in notes)


def test_a_run_too_sparse_to_yield_an_interval_is_still_described() -> None:
    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(np.array([10.0])), recording_id="sub-0001_run-1"
    )

    assert metrics.marker_count == 1
    assert np.isnan(metrics.median_bpm)
    assert metrics.duration_seconds == pytest.approx(100.0)
    assert pulse_marker_bound_notes(metrics, CRITERIA) == (
        "marker count 1 is too few to derive an interval",
    )


def test_notes_name_the_measured_value_and_the_bound_it_is_compared_against() -> None:
    """So a reader who disagrees with the bound still has the measurement."""
    complete = np.arange(2.0, 98.0, 0.8)
    dropped = np.delete(complete, np.arange(2, len(complete), 3))

    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(dropped), recording_id="sub-0001_run-1"
    )
    notes = pulse_marker_bound_notes(metrics, CRITERIA)

    assert any(f"{metrics.marker_fraction:.3f}" in note and "0.800" in note for note in notes)


def test_non_increasing_onsets_still_raise() -> None:
    """Not a quality question: the marker train is not a time series, so nothing is
    defined on it."""
    with pytest.raises(ValueError, match="strictly increasing"):
        measure_pulse_markers(
            _raw_with_pulse_markers(np.array([10.0, 10.0, 20.0])),
            recording_id="sub-0001_run-1",
        )


def test_every_run_keeps_every_measurement_in_the_written_table(tmp_path) -> None:
    import csv

    good = _raw_with_pulse_markers(np.arange(5.0, 100.0, 1.0))
    poor = _raw_with_pulse_markers(np.array([10.0, 40.0, 70.0]))
    out = tmp_path / "qc.tsv"

    summarize_pulse_marker_recordings(
        [("run-good", good), ("run-poor", poor)],
        CRITERIA,
        output_path=out,
        strict=False,
    )

    with out.open(encoding="utf-8") as handle:
        rows = {row["recording_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    assert set(rows) == {"run-good", "run-poor"}
    for row in rows.values():
        assert row["marker_count"] != ""
        assert row["recording_coverage"] != ""
        assert row["duration_seconds"] != ""
    assert rows["run-good"]["outside_configured_bounds"] == "no"
    assert rows["run-poor"]["outside_configured_bounds"] == "yes"
    assert rows["run-poor"]["notes"]
    # The bounds travel with the measurements so the comparison stays re-derivable.
    assert rows["run-poor"]["configured_minimum_marker_fraction"] == "0.800"


def test_strict_is_opt_in_and_reports_every_run_at_once(tmp_path) -> None:
    poor = _raw_with_pulse_markers(np.array([10.0, 40.0, 70.0]))
    out = tmp_path / "qc.tsv"

    # Default is descriptive: the table is written and nothing is raised.
    summarize_pulse_marker_recordings([("run-poor", poor)], CRITERIA, output_path=out)
    assert out.is_file()

    with pytest.raises(ValueError, match="outside the configured bounds"):
        summarize_pulse_marker_recordings(
            [("run-poor", poor)], CRITERIA, output_path=out, strict=True
        )


def test_the_table_is_written_before_strict_raises(tmp_path) -> None:
    poor = _raw_with_pulse_markers(np.array([10.0, 40.0, 70.0]))
    out = tmp_path / "qc.tsv"

    with pytest.raises(ValueError):
        summarize_pulse_marker_recordings(
            [("run-poor", poor)], CRITERIA, output_path=out, strict=True
        )

    assert out.is_file(), "the measurements must survive the raise"


def test_gap_free_coverage_separates_a_sparse_train_from_a_short_one() -> None:
    """A run can span almost all of its recording while marking a third of the beats.

    ``recording_coverage`` is (last - first) / duration, so it reads high for both. The
    pulse-correction recovery investigation reports the gap-aware quantity instead, and
    that is what distinguishes the two cases.
    """
    # Beats every second for the first 30 s, then a 40 s hole, then beats again.
    onsets = np.concatenate([np.arange(5.0, 35.0, 1.0), np.arange(75.0, 99.0, 1.0)])

    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    # Spans nearly the whole recording...
    assert metrics.recording_coverage > 0.9
    # ...but the hole is excluded from the gap-aware measure.
    assert metrics.gap_free_coverage == pytest.approx(0.53, abs=0.02)
    assert metrics.gap_count == 1


def test_a_continuous_train_scores_the_same_on_both_coverages() -> None:
    onsets = np.arange(5.0, 100.0, 1.0)

    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    assert metrics.gap_count == 0
    assert metrics.gap_free_coverage == pytest.approx(metrics.recording_coverage)


def test_a_slow_heart_is_not_mistaken_for_a_dropout() -> None:
    """The gap threshold is relative to the run's own median, so an evenly slow train
    has no gaps."""
    onsets = np.arange(5.0, 100.0, 2.0)  # 30 bpm, perfectly regular

    metrics = measure_pulse_markers(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    assert metrics.gap_count == 0
    assert metrics.gap_free_coverage == pytest.approx(metrics.recording_coverage)
