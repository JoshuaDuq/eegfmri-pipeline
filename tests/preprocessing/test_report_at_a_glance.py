"""The landing panel: what a reviewer meets before scrolling.

A subject report runs to tens of megabytes across roughly twenty-five sections, and the
first thing it currently shows is a configuration table. Nothing in the document states
the handful of numbers that decide whether the subject is usable, and nothing puts the
one anomalous run beside the five ordinary ones at the top of the page.

The panel is assembled from the build record rather than recomputed, so it cannot
disagree with the sections it summarises: every number on it was measured by the stage
that owns that section and written down at the same moment.
"""

from __future__ import annotations

import matplotlib
import mne
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.at_a_glance import (  # noqa: E402
    AT_A_GLANCE_TAG,
    HEADLINES,
    add_at_a_glance_section,
    at_a_glance_html,
)


def _record(*stages):
    return {"schema_version": 1, "stages": list(stages)}


def _stage(name, written_at, **measurements):
    return {
        "stage": name,
        "written_at": written_at,
        "versions": {"eeg_pipeline": "1.0.0", "mne": "1.12.1"},
        "measurements": measurements,
    }


def test_headline_numbers_are_lifted_out_of_the_record() -> None:
    document = at_a_glance_html(
        _record(
            _stage("band-ica-report", "2026-07-25T23:51:09+00:00", n_components=22, n_excluded=16),
        )
    )

    assert "22" in document
    assert "16" in document


def test_only_measurements_that_were_recorded_appear() -> None:
    """An EEG-only dataset has no scanner stage, and rest has no trial retention.

    The panel must shrink rather than carry rows reading "not measured", which is how a
    reader ends up scanning a column of blanks looking for the numbers that exist.
    """
    document = at_a_glance_html(
        _record(_stage("band-ica-report", "2026-07-25T23:51:09+00:00", n_components=22))
    )

    assert "22" in document
    assert "Trials retained" not in document
    assert "not measured" not in document.lower()


def test_a_record_with_no_headline_numbers_produces_no_panel() -> None:
    """A report built from the bad-channel stage alone has nothing to summarise."""
    assert at_a_glance_html(_record(_stage("report-review", "2026-07-25T23:56:35+00:00"))) == ""
    assert at_a_glance_html(_record()) == ""


def test_the_latest_stage_to_measure_something_is_the_one_reported() -> None:
    """The record describes the report as it stands, so a rebuild supersedes its predecessor."""
    document = at_a_glance_html(
        _record(
            _stage("band-ica-report", "2026-07-25T20:00:00+00:00", n_excluded=16),
            _stage("ica-condition-tfr", "2026-07-25T23:56:03+00:00", n_excluded=9),
        )
    )

    assert "9" in document
    assert ">16<" not in document


def test_the_panel_states_measurements_without_grading_them() -> None:
    """Reference values belong beside the measurement in its own section, not here."""
    document = at_a_glance_html(
        _record(
            _stage(
                "band-ica-report",
                "2026-07-25T23:51:09+00:00",
                n_components=22,
                n_excluded=16,
                variance_removed=0.939313,
            )
        )
    )

    for verdict in ("fail", "pass", "warning", "&#9888;", "poor", "acceptable", "unusable"):
        assert verdict not in document.lower()


def test_a_measured_fraction_is_shown_as_a_percentage() -> None:
    """0.939313 is not how anyone reports a share of variance."""
    document = at_a_glance_html(
        _record(_stage("band-ica-report", "2026-07-25T23:51:09+00:00", variance_removed=0.939313))
    )

    assert "93.9%" in document
    assert "0.939313" not in document


def test_the_panel_is_placed_before_everything_else() -> None:
    report = mne.Report(title="subject", verbose="ERROR")
    report.add_html(html="<p>config</p>", title="Configuration", section="Configuration")

    add_at_a_glance_section(
        report=report,
        record=_record(_stage("band-ica-report", "2026-07-25T23:51:09+00:00", n_components=22)),
    )

    assert AT_A_GLANCE_TAG in report._content[0].tags


def test_rebuilding_replaces_the_panel_rather_than_stacking_another() -> None:
    report = mne.Report(title="subject", verbose="ERROR")
    record = _record(_stage("band-ica-report", "2026-07-25T23:51:09+00:00", n_components=22))

    add_at_a_glance_section(report=report, record=record)
    add_at_a_glance_section(report=report, record=record)

    assert sum(AT_A_GLANCE_TAG in element.tags for element in report._content) == 1


def test_nothing_is_added_when_there_is_nothing_to_summarise() -> None:
    report = mne.Report(title="subject", verbose="ERROR")

    add_at_a_glance_section(report=report, record=_record())

    assert report._content == []


def test_an_unrecognised_measurement_is_left_to_its_own_section() -> None:
    """The panel is a selected summary, not a dump of every key any stage recorded."""
    document = at_a_glance_html(
        _record(
            _stage(
                "band-ica-report",
                "2026-07-25T23:51:09+00:00",
                n_components=22,
                samples_per_squared_component=1500.14,
            )
        )
    )

    assert "1500" not in document


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_measurement_is_omitted_rather_than_printed(value: float) -> None:
    """"nan" on the landing panel reads as a measured value, and it is not one."""
    document = at_a_glance_html(
        _record(_stage("band-ica-report", "2026-07-25T23:51:09+00:00", variance_removed=value))
    )

    assert "nan" not in document.lower()
    assert "inf" not in document.lower()


def test_every_headline_points_at_a_section_the_document_has() -> None:
    """The panel's last column is a navigation aid, so a wrong name is a dead pointer.

    The two epoch rows named "Trial retention", which is the *title of a figure* inside the
    rejection section rather than a section of either document. A reader following it finds
    nothing, and the cohort panel reuses this same list to point at its own sections.
    """
    sections = {headline.section for headline in HEADLINES}

    assert "Trial retention" not in sections
    assert "Epoch rejection" in sections
