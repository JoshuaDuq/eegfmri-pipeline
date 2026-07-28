"""The landing panel: the subject report's list, each value as a distribution.

The two failures worth pinning are both about denominators. A row must count only the
participants that recorded it, because a participant contributes to some of these and not
to others. And a headline nobody recorded must produce no row at all, rather than one that
states the measurement was attempted and came out empty.
"""

from __future__ import annotations

from eeg_pipeline.preprocessing.report.at_a_glance import HEADLINES
from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates, BandRegime
from eeg_pipeline.preprocessing.report.cohort.at_a_glance import (
    glance_frame,
    glance_html,
    glance_rows,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)


def _participant(subject: str, **measurements) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=Paradigm.TASK,
        measurements=measurements,
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


def _row(rows, key):
    return next(row for row in rows if row.key == key)


def test_the_rows_follow_the_subject_panel_order() -> None:
    """A reader arriving from a subject report keeps the mapping they already have."""
    rows = glance_rows(
        _cohort(
            _participant("0014", n_runs=4, n_channels=63, variance_removed=0.86),
            _participant("0015", n_runs=3, n_channels=63, variance_removed=0.81),
        )
    )

    order = [headline.key for headline in HEADLINES]
    assert [row.key for row in rows] == [key for key in order if key in {r.key for r in rows}]


def test_each_row_counts_only_who_recorded_it() -> None:
    """One count in the heading would misstate every row but one."""
    rows = glance_rows(
        _cohort(
            _participant("0014", n_runs=4, worst_marker_agreement=0.97),
            _participant("0015", n_runs=3),
        )
    )

    assert _row(rows, "n_runs").n_subjects == 2
    assert _row(rows, "worst_marker_agreement").n_subjects == 1


def test_a_headline_nobody_recorded_produces_no_row() -> None:
    rows = glance_rows(_cohort(_participant("0014", n_runs=4)))

    assert {row.key for row in rows} == {"n_runs"}


def test_a_non_finite_measurement_is_not_a_value() -> None:
    """A measurement attempted and unresolved must not become a number in a median."""
    rows = glance_rows(
        _cohort(
            _participant("0014", variance_removed=0.86),
            _participant("0015", variance_removed=float("nan")),
        )
    )

    assert _row(rows, "variance_removed").n_subjects == 1


def test_a_boolean_is_a_state_rather_than_a_number() -> None:
    rows = glance_rows(_cohort(_participant("0014", n_runs=True)))

    assert rows == []


def test_the_median_is_withheld_below_the_gate() -> None:
    rows = glance_rows(
        _cohort(*(_participant(f"00{index}", n_runs=index) for index in range(10, 13)))
    )

    row = _row(rows, "n_runs")
    assert row.regime is BandRegime.INDIVIDUALS
    assert row.median is None
    assert row.span == (10.0, 12.0)


def test_the_median_appears_once_the_gate_is_met() -> None:
    rows = glance_rows(
        _cohort(*(_participant(f"00{index}", n_runs=index) for index in range(10, 15))),
        gates=BandGates(),
    )

    row = _row(rows, "n_runs")
    assert row.regime is BandRegime.MEDIAN_IQR
    assert row.median == 12.0


def test_the_participants_at_each_end_are_named_by_position() -> None:
    rows = glance_rows(
        _cohort(
            _participant("0014", variance_removed=0.60),
            _participant("0015", variance_removed=0.86),
            _participant("0016", variance_removed=0.72),
        )
    )

    assert _row(rows, "variance_removed").extremes == ("0014", "0015")


def test_one_participant_holds_no_two_ends() -> None:
    rows = glance_rows(_cohort(_participant("0014", variance_removed=0.60)))

    assert _row(rows, "variance_removed").extremes is None


def test_a_withheld_median_is_explained_as_a_participant_count() -> None:
    html = glance_html(_cohort(_participant("0014", n_runs=4)))

    assert "No median is shown" in html
    assert "too few for a summary" in html


def test_a_cohort_that_recorded_nothing_says_so() -> None:
    html = glance_html(_cohort(_participant("0014")))

    assert "nothing to summarise" in html


def test_the_audit_frame_carries_every_participant_value() -> None:
    frame = glance_frame(
        glance_rows(
            _cohort(
                _participant("0014", n_runs=4, variance_removed=0.86),
                _participant("0015", n_runs=3, variance_removed=0.81),
            )
        )
    )

    assert set(frame["subject"]) == {"0014", "0015"}
    assert set(frame["measurement"]) == {"n_runs", "variance_removed"}
    assert len(frame) == 4
