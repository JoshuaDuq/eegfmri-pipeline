from __future__ import annotations

import mne
import numpy as np
import pytest

from studies.pain_study.analysis.bcg.pulse_markers import (
    UNCORRECTED_PULSE_DESCRIPTION,
    PulseMarkerCriteria,
    measure_pulse_markers,
    pulse_marker_bound_notes,
    summarize_pulse_marker_recordings,
    uncorrected_pulse_intervals,
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

    metrics = measure_pulse_markers(_raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1")

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

    metrics = measure_pulse_markers(_raw_with_pulse_markers(sparse), recording_id="sub-0001_run-1")

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

    metrics = measure_pulse_markers(_raw_with_pulse_markers(dropped), recording_id="sub-0001_run-1")
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

    metrics = measure_pulse_markers(_raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1")

    # Spans nearly the whole recording...
    assert metrics.recording_coverage > 0.9
    # ...but the hole is excluded from the gap-aware measure.
    assert metrics.gap_free_coverage == pytest.approx(0.53, abs=0.02)
    assert metrics.gap_count == 1


def test_a_continuous_train_scores_the_same_on_both_coverages() -> None:
    onsets = np.arange(5.0, 100.0, 1.0)

    metrics = measure_pulse_markers(_raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1")

    assert metrics.gap_count == 0
    assert metrics.gap_free_coverage == pytest.approx(metrics.recording_coverage)


def test_a_slow_heart_is_not_mistaken_for_a_dropout() -> None:
    """The gap threshold is relative to the run's own median, so an evenly slow train
    has no gaps."""
    onsets = np.arange(5.0, 100.0, 2.0)  # 30 bpm, perfectly regular

    metrics = measure_pulse_markers(_raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1")

    assert metrics.gap_count == 0
    assert metrics.gap_free_coverage == pytest.approx(metrics.recording_coverage)


def test_a_gap_becomes_an_interval_covering_the_unmarked_beats() -> None:
    """Where no beat was marked, no pulse template was subtracted.

    The interval starts half a beat after the last marked beat and ends half a beat
    before the next, so it spans the region closer to the missing beats than to the
    corrected ones.
    """
    onsets = np.concatenate([np.arange(5.0, 35.0, 1.0), np.arange(75.0, 99.0, 1.0)])

    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    assert {a["description"] for a in annotations} == {UNCORRECTED_PULSE_DESCRIPTION}
    hole = [a for a in annotations if a["onset"] > 30.0 and a["duration"] > 10.0]
    assert len(hole) == 1
    assert hole[0]["onset"] == pytest.approx(34.5)
    assert hole[0]["onset"] + hole[0]["duration"] == pytest.approx(74.5)


def test_the_intervals_and_the_gap_count_come_from_the_same_rule() -> None:
    """A reader comparing the table against the annotations must not find two answers."""
    onsets = np.concatenate(
        [np.arange(5.0, 25.0, 1.0), np.arange(45.0, 60.0, 1.0), np.arange(80.0, 99.0, 1.0)]
    )
    raw = _raw_with_pulse_markers(onsets)

    metrics = measure_pulse_markers(raw, recording_id="sub-0001_run-1")
    annotations = uncorrected_pulse_intervals(raw, recording_id="sub-0001_run-1")

    # Drop the head and tail, which sit outside the marker train rather than inside it.
    interior = [a for a in annotations if a["onset"] > 0.0 and a["onset"] + a["duration"] < 100.0]
    assert len(interior) == metrics.gap_count == 2


def test_the_head_and_tail_outside_the_marker_train_are_marked() -> None:
    """The correction can only run where beats were marked, and the train rarely spans
    the whole recording."""
    onsets = np.arange(20.0, 80.0, 1.0)

    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(onsets, duration_seconds=100.0),
        recording_id="sub-0001_run-1",
    )

    spans = sorted((a["onset"], a["onset"] + a["duration"]) for a in annotations)
    assert spans[0][0] == pytest.approx(0.0)
    assert spans[0][1] == pytest.approx(19.5)
    assert spans[-1][1] == pytest.approx(100.0)


def test_a_continuous_train_spanning_the_run_leaves_nothing_marked() -> None:
    onsets = np.arange(0.5, 100.0, 1.0)

    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    assert len(annotations) == 0


def test_a_run_with_no_usable_marker_train_is_marked_end_to_end() -> None:
    """Fewer than three markers yields no interval to measure against, and the whole run
    is uncorrected rather than unmeasurable."""
    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(np.array([10.0]), duration_seconds=100.0),
        recording_id="sub-0001_run-1",
    )

    assert len(annotations) == 1
    assert annotations[0]["onset"] == pytest.approx(0.0)
    assert annotations[0]["duration"] == pytest.approx(100.0)


def test_a_sparse_train_is_not_measured_against_its_own_corruption() -> None:
    """A run missing most of its beats must not hide behind its own inflated interval.

    On this cohort sub-0008 run-4 carries 110 markers over 497 s -- a median interval of
    4.07 s where the same subject's other five runs sit at 1.0 s. Judged against its own
    median, almost nothing in it counts as a gap, so the run with the largest uncorrected
    ballistocardiogram in the cohort would be reported as nearly fully covered. Given the
    configured physiological floor, the beat period cannot be 4 s and the run is marked.
    """
    onsets = np.arange(2.0, 100.0, 4.0)  # 15 bpm: below any configured minimum
    raw = _raw_with_pulse_markers(onsets)

    against_itself = uncorrected_pulse_intervals(raw, recording_id="sub-0008_run-4")
    against_the_floor = uncorrected_pulse_intervals(
        raw,
        recording_id="sub-0008_run-4",
        maximum_plausible_interval=60.0 / CRITERIA.minimum_bpm,
    )

    assert sum(a["duration"] for a in against_itself) == 0.0, (
        "judged against its own inflated interval the run looks evenly covered — "
        "this is the blind spot the floor exists to close"
    )
    # 25 marked beats cover half a plausible period either side; the rest is uncorrected.
    assert float(sum(a["duration"] for a in against_the_floor)) == pytest.approx(64.0, abs=1.0)


def test_doubled_markers_do_not_flag_a_well_marked_run() -> None:
    """sub-0005 marks an extra beat mid-cycle often enough to halve a percentile estimate
    of the beat period, which would then report most of a well-marked run as uncorrected."""
    beats = np.arange(0.5, 100.0, 1.0)
    doubled = np.sort(np.concatenate([beats, beats[:30] + 0.52]))

    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(doubled),
        recording_id="sub-0005_run-2",
        maximum_plausible_interval=60.0 / CRITERIA.minimum_bpm,
    )

    assert len(annotations) == 0


def test_a_plausibly_slow_train_is_left_alone_by_the_same_bound() -> None:
    """The floor only catches rates the configuration calls impossible."""
    onsets = np.arange(0.6, 100.0, 1.2)  # 50 bpm, inside the configured 45-80

    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(onsets),
        recording_id="sub-0001_run-1",
        maximum_plausible_interval=60.0 / CRITERIA.minimum_bpm,
    )

    assert len(annotations) == 0


def test_a_slow_heart_produces_no_intervals() -> None:
    """The same relative rule that keeps a slow train out of gap_count keeps it out of
    the annotations."""
    onsets = np.arange(0.5, 100.0, 2.0)

    annotations = uncorrected_pulse_intervals(
        _raw_with_pulse_markers(onsets), recording_id="sub-0001_run-1"
    )

    assert len(annotations) == 0


def test_the_intervals_are_written_per_recording(tmp_path) -> None:
    good = _raw_with_pulse_markers(np.arange(0.5, 100.0, 1.0))
    holed = _raw_with_pulse_markers(
        np.concatenate([np.arange(0.5, 35.0, 1.0), np.arange(75.0, 100.0, 1.0)])
    )

    summarize_pulse_marker_recordings(
        [("run-good", good), ("run-holed", holed)],
        CRITERIA,
        output_path=tmp_path / "qc.tsv",
        annotations_dir=tmp_path / "intervals",
    )

    written = sorted(p.name for p in (tmp_path / "intervals").glob("*.tsv"))
    assert written == [
        "run-good_desc-bcguncorrected_annotations.tsv",
        "run-holed_desc-bcguncorrected_annotations.tsv",
    ], "a run with nothing to mark still gets a file, so a missing one means a missing run"

    holed_rows = tmp_path / "intervals" / "run-holed_desc-bcguncorrected_annotations.tsv"
    assert len(holed_rows.read_text().strip().splitlines()) == 2  # header + one gap
