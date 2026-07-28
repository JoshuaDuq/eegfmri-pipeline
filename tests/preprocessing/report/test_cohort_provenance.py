"""The document's account of itself.

A cohort figure is no fresher than the oldest sidecar behind it, and a panel with no median
is either a small cohort or a broken measurement. Both facts are unrecoverable from the
figures, so this section carries them, and these tests pin that it does.
"""

from __future__ import annotations

from eeg_pipeline.preprocessing.report.cohort.provenance import provenance_html


def _log(**overrides) -> dict:
    log = {
        "schema_version": 1,
        "sidecar_schema_version": 1,
        "written_at": "2026-07-26T12:00:00+00:00",
        "task": "thermalactive",
        "n_participants": 2,
        "participants": ["0014", "0015"],
        "n_runs": 4,
        "not_aggregated": [],
        "band_gates": {"min_subjects_for_median": 5, "min_subjects_for_outer_band": 10},
        "versions": {"mne": "1.12.1", "eeg_pipeline": "1.0.0"},
        "audit_tables": ["task-thermalactive_desc-cohortspectra_qc.tsv"],
    }
    log.update(overrides)
    return log


def test_the_contributing_participants_are_named() -> None:
    html = provenance_html(_log())

    assert "0014" in html
    assert "0015" in html


def test_a_participant_that_was_left_out_is_named_with_its_reason() -> None:
    """A reader needs to know the study was larger than the document describes."""
    html = provenance_html(
        _log(not_aggregated=[{"subject": "0016", "reason": "No QC sidecar beside the report."}])
    )

    assert "0016" in html
    assert "No QC sidecar" in html


def test_a_complete_run_carries_no_exclusion_table() -> None:
    html = provenance_html(_log())

    assert "not aggregated" not in html.lower()


def test_the_gates_a_missing_median_is_explained_by_are_recorded() -> None:
    html = provenance_html(_log())

    assert "5 participants" in html
    assert "10 participants" in html
    assert "too few contributors" in html


def test_the_versions_that_assembled_the_document_are_distinguished_from_the_measurements() -> None:
    """The aggregate is no fresher than the sidecars; the reader must not conflate them."""
    html = provenance_html(_log())

    assert "1.12.1" in html
    assert "when each participant&rsquo;s report was built" in html or (
        "when each participant's report was built" in html
    )


def test_the_audit_tables_are_named_so_a_figure_can_be_checked() -> None:
    html = provenance_html(_log())

    assert "task-thermalactive_desc-cohortspectra_qc.tsv" in html


def test_the_sidecar_layout_is_stated() -> None:
    """A document assembled under one layout must say which, so a reader can tell."""
    html = provenance_html(_log())

    assert "Sidecar layout" in html


def test_an_unnamed_task_reads_as_every_task_found() -> None:
    html = provenance_html(_log(task=None))

    assert "every task found" in html


def test_a_log_with_nothing_optional_still_renders() -> None:
    html = provenance_html({"schema_version": 1})

    assert "assembled from per-subject QC sidecars" in html
