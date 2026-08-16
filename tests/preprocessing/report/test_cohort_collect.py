"""Discovery must account for everyone it finds, including those it cannot use.

The failure this guards against is quiet: a participant whose sidecar is unreadable
vanishing from the cohort, leaving a denominator that is smaller than the study and gives
the reader no reason to doubt it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from eeg_pipeline.preprocessing.report.cohort.collect import (
    Cohort,
    collect_cohort,
    discover_reports,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    SCHEMA_VERSION,
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
    sidecar_paths,
    write_sidecar,
)


def _runs(*, in_scanner: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run": ["run-1"],
            "n_channels": [63],
            "duration_s": [720.0],
            "flagged_fraction": [0.04],
            "continuity_median_db": [0.5],
            "continuity_max_db": [6.0],
        }
    )
    if in_scanner:
        frame["n_volumes"] = [360]
        frame["repetition_time_s"] = [2.0]
        frame["volume_jitter_s"] = [0.003]
        frame["volume_locked_rms_before_uv"] = [2.1]
        frame["volume_locked_floor_before_uv"] = [0.5]
        frame["volume_locked_excess_power_before_uv2"] = [4.16]
        frame["volume_locked_resolved_before"] = [True]
        frame["volume_locked_rms_after_uv"] = [1.26]
        frame["volume_locked_floor_after_uv"] = [0.4]
        frame["volume_locked_excess_power_after_uv2"] = [1.44]
        frame["volume_locked_resolved_after"] = [True]
        frame["median_bpm"] = [62.0]
        frame["n_beats"] = [700.0]
        frame["beat_dropouts"] = [2.0]
        frame["marker_matched_fraction"] = [0.96]
        frame["marker_median_lag_s"] = [0.004]
        frame["marker_lag_iqr_s"] = [0.010]
        frame["n_markers"] = [704.0]
        frame["n_detected_beats"] = [700.0]
        frame["n_matched_beats"] = [672.0]
        frame["pulse_marker_count"] = [704.0]
        frame["beat_source"] = ["ECG"]
        frame["bcg_residual_uv"] = [0.8]
        frame["bcg_beat_train_coverage"] = [0.98]
        frame["bcg_noise_floor_uv"] = [0.5]
        frame["bcg_excess_power_uv2"] = [0.39]
        frame["bcg_resolved"] = [True]
        frame["bcg_n_beats"] = [700.0]
    return frame


def _spectra() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": ["run-1"],
            "stage": ["after"],
            "freq_hz": [10.0],
            "median_db": [-10.0],
            "max_db": [-5.0],
        }
    )


def _write(
    root,
    subject: str,
    *,
    task: str = "thermalactive",
    context: AcquisitionContext = AcquisitionContext.IN_SCANNER,
    paradigm: Paradigm = Paradigm.TASK,
):
    report = root / f"sub-{subject}" / "eeg" / f"sub-{subject}_report.h5"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.touch()
    write_sidecar(
        report,
        SubjectSidecar(
            subject=subject,
            task=task,
            context=context,
            paradigm=paradigm,
            runs=_runs(in_scanner=context is AcquisitionContext.IN_SCANNER),
            spectrum_curves=_spectra(),
        ),
    )
    return report


def _report_without_sidecar(root, subject: str):
    report = root / f"sub-{subject}" / "eeg" / f"sub-{subject}_report.h5"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.touch()
    return report


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def test_reports_are_discovered_under_the_derivatives_root(tmp_path) -> None:
    _write(tmp_path, "0014")
    _write(tmp_path, "0015")

    assert sorted(discover_reports(tmp_path)) == ["0014", "0015"]


def test_a_requested_participant_that_does_not_exist_is_an_error(tmp_path) -> None:
    """A typo must not quietly produce a smaller cohort that looks correct."""
    _write(tmp_path, "0014")

    with pytest.raises(ValueError, match="0099"):
        discover_reports(tmp_path, subjects=["0014", "0099"])


def test_a_missing_derivatives_root_is_an_error(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        discover_reports(tmp_path / "nowhere")


# --------------------------------------------------------------------------------------
# Accounting for who was left out
# --------------------------------------------------------------------------------------


def test_a_participant_without_a_sidecar_is_listed_rather_than_dropped(tmp_path) -> None:
    _write(tmp_path, "0014")
    _report_without_sidecar(tmp_path, "0015")

    cohort = collect_cohort(tmp_path)

    assert cohort.subjects == ("0014",)
    assert [entry.subject for entry in cohort.not_aggregated] == ["0015"]
    assert "sidecar" in cohort.not_aggregated[0].reason


def test_one_stale_sidecar_does_not_cost_the_reader_the_others(tmp_path) -> None:
    """Strictness belongs to the reader; continuing belongs here."""
    _write(tmp_path, "0014")
    stale = _write(tmp_path, "0015")
    paths = sidecar_paths(stale)
    document = json.loads(paths.subject_json.read_text(encoding="utf-8"))
    document["schema_version"] = SCHEMA_VERSION + 1
    paths.subject_json.write_text(json.dumps(document), encoding="utf-8")

    cohort = collect_cohort(tmp_path)

    assert cohort.subjects == ("0014",)
    assert "schema" in cohort.not_aggregated[0].reason


def test_a_cohort_with_nobody_left_explains_who_was_skipped(tmp_path) -> None:
    _report_without_sidecar(tmp_path, "0014")

    with pytest.raises(ValueError, match="0014"):
        collect_cohort(tmp_path)


def test_a_participant_recorded_under_another_task_is_listed_with_the_reason(tmp_path) -> None:
    _write(tmp_path, "0014", task="thermalactive")
    _write(tmp_path, "0015", task="rest")

    cohort = collect_cohort(tmp_path, task="thermalactive")

    assert cohort.subjects == ("0014",)
    assert "rest" in cohort.not_aggregated[0].reason


# --------------------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------------------


def test_a_single_context_cohort_is_not_mixed(tmp_path) -> None:
    _write(tmp_path, "0014")
    _write(tmp_path, "0015")

    cohort = collect_cohort(tmp_path)

    assert not cohort.is_mixed
    assert cohort.contexts == (AcquisitionContext.IN_SCANNER,)


def test_a_mixed_cohort_is_detected_and_splits_along_the_context(tmp_path) -> None:
    """The axis the report refuses to pool across."""
    _write(tmp_path, "0014", context=AcquisitionContext.IN_SCANNER)
    _write(tmp_path, "0015", context=AcquisitionContext.OUT_OF_SCANNER)

    cohort = collect_cohort(tmp_path)

    assert cohort.is_mixed
    strata = dict(cohort.stratified())
    assert strata[AcquisitionContext.IN_SCANNER].subjects == ("0014",)
    assert strata[AcquisitionContext.OUT_OF_SCANNER].subjects == ("0015",)


def test_selecting_a_paradigm_narrows_the_cohort(tmp_path) -> None:
    _write(tmp_path, "0014", paradigm=Paradigm.TASK)
    _write(tmp_path, "0015", paradigm=Paradigm.REST, task="thermalactive")

    cohort = collect_cohort(tmp_path)

    assert cohort.select(paradigm=Paradigm.TASK).subjects == ("0014",)
    assert cohort.select(paradigm=Paradigm.REST).subjects == ("0015",)


def test_a_selection_keeps_the_record_of_who_was_skipped(tmp_path) -> None:
    """A section's denominator and the document's must be traceable to one account."""
    _write(tmp_path, "0014")
    _report_without_sidecar(tmp_path, "0015")

    cohort = collect_cohort(tmp_path)
    narrowed = cohort.select(context=AcquisitionContext.IN_SCANNER)

    assert narrowed.not_aggregated == cohort.not_aggregated


def test_a_cohort_cannot_hold_a_participant_twice(tmp_path) -> None:
    participant = SubjectSidecar(
        subject="0014",
        task="thermalactive",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=Paradigm.TASK,
    )

    with pytest.raises(ValueError, match="0014"):
        Cohort(participants=(participant, participant))


def test_the_task_a_cohort_turned_out_to_cover_is_reported(tmp_path) -> None:
    """Naming the document is the caller's decision, so collection only reports."""
    _write(tmp_path, "0014", task="thermalactive")
    _write(tmp_path, "0015", task="rest")

    assert collect_cohort(tmp_path).tasks == ("rest", "thermalactive")
