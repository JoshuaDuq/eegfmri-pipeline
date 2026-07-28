"""A subject report is written by several stages at several times.

Each stage reopens the same file, appends its own sections and saves again, so a report
on disk can hold sections from a run this morning beside sections from a run last week
under different settings. Nothing in the document said so: sub-0015's report file carried
a timestamp of 16:19 over sections generated at 15:56, and a reader had no way to tell.

These tests pin the record that makes that visible, and the sidecar that carries it.
"""

from __future__ import annotations

import json

import mne
import pytest

from eeg_pipeline.preprocessing.report.build_record import (
    build_record_html,
    read_build_record,
    record_path_for_report,
    record_stage,
    save_subject_report,
)


def _report_path(tmp_path):
    path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="sub-0001", verbose="ERROR").save(path, overwrite=True, open_browser=False)
    return path


def test_the_record_sits_beside_the_report_it_describes(tmp_path) -> None:
    """A log kept anywhere else is separated from its report by the first file move."""
    path = record_path_for_report(tmp_path / "sub-0001_report.h5")

    assert path.parent == tmp_path
    assert path.name == "sub-0001_desc-reportbuild_log.json"


def test_a_stage_records_when_it_wrote_and_what_it_was_running(tmp_path) -> None:
    report_path = _report_path(tmp_path)

    record_stage(report_path, stage="ica-cardiac-review")

    record = read_build_record(report_path)
    entry = record["stages"][0]
    assert entry["stage"] == "ica-cardiac-review"
    assert entry["written_at"].endswith("+00:00")
    assert entry["versions"]["mne"] == mne.__version__
    assert entry["versions"]["eeg_pipeline"]


def test_rerunning_a_stage_replaces_its_entry_rather_than_appending(tmp_path) -> None:
    """The record describes the report as it stands, not every attempt that built it.

    An accumulating log would say a report contains sections from six runs of the cardiac
    stage when it contains the sections written by the last one.
    """
    report_path = _report_path(tmp_path)

    record_stage(report_path, stage="ica-cardiac-review")
    record_stage(report_path, stage="ica-ocular-review")
    record_stage(report_path, stage="ica-cardiac-review")

    stages = [entry["stage"] for entry in read_build_record(report_path)["stages"]]
    assert sorted(stages) == ["ica-cardiac-review", "ica-ocular-review"]


def test_a_stage_can_record_the_numbers_it_measured(tmp_path) -> None:
    """The measurements a stage renders as prose are the ones an exclusion decision needs
    to read without parsing HTML."""
    report_path = _report_path(tmp_path)

    record_stage(
        report_path,
        stage="ica-decomposition",
        measurements={"n_components": 22, "n_excluded": 16, "variance_removed": 0.99},
    )

    entry = read_build_record(report_path)["stages"][0]
    assert entry["measurements"]["variance_removed"] == 0.99


def test_the_record_is_valid_json_on_disk(tmp_path) -> None:
    """It exists to be read by something other than this module."""
    report_path = _report_path(tmp_path)
    record_stage(report_path, stage="epochs", measurements={"retained": 64})

    parsed = json.loads(record_path_for_report(report_path).read_text(encoding="utf-8"))

    assert parsed["schema_version"] >= 1
    assert parsed["stages"][0]["measurements"]["retained"] == 64


def test_the_panel_names_every_stage_that_contributed(tmp_path) -> None:
    report_path = _report_path(tmp_path)
    record_stage(report_path, stage="ica-cardiac-review")
    record_stage(report_path, stage="ica-ocular-review")

    document = build_record_html(read_build_record(report_path))

    assert "ica-cardiac-review" in document and "ica-ocular-review" in document


def test_the_panel_says_when_the_stages_disagree_about_the_pipeline_version(tmp_path) -> None:
    """Two versions in one document is the case the panel exists for, and it is invisible
    unless the versions are compared rather than merely listed."""
    record = {
        "schema_version": 1,
        "stages": [
            {
                "stage": "epochs",
                "written_at": "2026-07-25T15:56:00+00:00",
                "versions": {"eeg_pipeline": "1.0.0", "mne": "1.12.1"},
                "measurements": {},
            },
            {
                "stage": "ica-cardiac-review",
                "written_at": "2026-07-25T16:19:00+00:00",
                "versions": {"eeg_pipeline": "1.1.0", "mne": "1.12.1"},
                "measurements": {},
            },
        ],
    }

    document = build_record_html(record)

    assert "more than one version" in document


def test_one_consistent_version_is_not_reported_as_a_disagreement() -> None:
    record = {
        "schema_version": 1,
        "stages": [
            {
                "stage": "epochs",
                "written_at": "2026-07-25T15:56:00+00:00",
                "versions": {"eeg_pipeline": "1.0.0", "mne": "1.12.1"},
                "measurements": {},
            }
        ],
    }

    assert "more than one version" not in build_record_html(record)


def test_saving_through_the_helper_records_the_stage_and_refreshes_the_panel(tmp_path) -> None:
    """Recording at the save is what makes it unforgettable: a stage that writes sections
    and does not save has changed nothing, and one that saves has recorded itself."""
    report_path = _report_path(tmp_path)
    report = mne.open_report(report_path)

    save_subject_report(report, report_path, stage="ica-ocular-review")

    assert [entry["stage"] for entry in read_build_record(report_path)["stages"]] == [
        "ica-ocular-review"
    ]
    reopened = mne.open_report(report_path)
    panel = next(element for element in reopened._content if "report-build" in element.tags)
    assert "ica-ocular-review" in panel.html


def test_saving_writes_both_the_archive_and_the_browsable_report(tmp_path) -> None:
    """Every stage previously wrote the pair by hand, and a stage that forgot the second
    left the HTML a reader opens older than the .h5 the next stage reopens."""
    report_path = _report_path(tmp_path)

    save_subject_report(mne.open_report(report_path), report_path, stage="epochs")

    assert report_path.is_file()
    assert report_path.with_suffix(".html").is_file()


def test_a_report_saved_twice_carries_one_build_panel(tmp_path) -> None:
    """The panel is rewritten on every save, so it must replace rather than accumulate."""
    report_path = _report_path(tmp_path)
    save_subject_report(mne.open_report(report_path), report_path, stage="epochs")
    save_subject_report(mne.open_report(report_path), report_path, stage="ica-ocular-review")

    reopened = mne.open_report(report_path)
    panels = [element for element in reopened._content if "report-build" in element.tags]
    assert len(panels) == 1


def test_saving_refreshes_the_landing_panel_from_the_accumulated_record(tmp_path) -> None:
    """The panel has to be rebuilt at every save, not written once by one stage.

    It summarises measurements from several stages, and which of them have run yet
    depends on where the pipeline is. Built at the save, it always describes the document
    as it currently stands; built by any single stage, it would freeze at whatever had
    been measured by the time that stage happened to run.
    """
    report_path = _report_path(tmp_path)

    save_subject_report(
        mne.open_report(report_path),
        report_path,
        stage="band-ica-report",
        measurements={"n_components": 22, "n_excluded": 16},
    )
    save_subject_report(
        mne.open_report(report_path),
        report_path,
        stage="epoch-rejection",
        measurements={"epochs_kept": 60, "epochs_total": 66},
    )

    reopened = mne.open_report(report_path)
    panels = [element for element in reopened._content if "at-a-glance" in element.tags]
    assert len(panels) == 1
    # Both stages' numbers, though neither stage knew about the other.
    assert "22" in panels[0].html and "60" in panels[0].html
    assert "at-a-glance" in reopened._content[0].tags


def test_a_report_with_nothing_measured_yet_gets_no_landing_panel(tmp_path) -> None:
    report_path = _report_path(tmp_path)

    save_subject_report(mne.open_report(report_path), report_path, stage="epochs")

    reopened = mne.open_report(report_path)
    assert not any("at-a-glance" in element.tags for element in reopened._content)


def test_reading_a_record_that_does_not_exist_yet_is_not_an_error(tmp_path) -> None:
    """The first stage to run finds no record and must create one rather than fail."""
    record = read_build_record(tmp_path / "sub-0001_report.h5")

    assert record["stages"] == []


def test_a_corrupt_record_fails_loudly_rather_than_silently_restarting(tmp_path) -> None:
    """Silently overwriting it would erase the provenance the file exists to hold."""
    report_path = tmp_path / "sub-0001_report.h5"
    record_path_for_report(report_path).write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_build_record(report_path)
