"""The composition section is the denominator every other panel is read against.

Its failures are all of one kind: a reader forming a correct impression of a cohort that
is not the cohort in front of them. A skipped participant that goes unmentioned, a mixed
acquisition presented as one population, or an absent median that reads as a failed
measurement rather than as a count too small to support one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort, NotAggregated
from eeg_pipeline.preprocessing.report.cohort.composition import (
    composition_html,
    participant_rows,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)


def _runs(n_runs: int = 2, *, n_channels: int = 63, duration_s: float = 1800.0):
    return pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [n_channels] * n_runs,
            "duration_s": [duration_s] * n_runs,
            "flagged_fraction": [0.01] * n_runs,
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [4.0] * n_runs,
        }
    )


def _participant(
    subject: str,
    *,
    context: AcquisitionContext = AcquisitionContext.IN_SCANNER,
    paradigm: Paradigm = Paradigm.TASK,
    n_runs: int = 2,
    version: str = "1.0.0",
    acquired: str | None = "2026-03-04",
) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=context,
        paradigm=paradigm,
        versions={"eeg_pipeline": version},
        acquisition_date=acquired,
        runs=_runs(n_runs),
    )


def _cohort(*participants, not_aggregated=()) -> Cohort:
    return Cohort(participants=tuple(participants), not_aggregated=tuple(not_aggregated))


def test_every_aggregated_participant_gets_a_row() -> None:
    rows = participant_rows(_cohort(_participant("0014"), _participant("0015")))

    assert [row[0] for row in rows] == ["0014", "0015"]


def test_a_row_states_the_context_the_paradigm_and_the_pipeline() -> None:
    rows = participant_rows(_cohort(_participant("0014", version="1.2.0")))

    assert rows[0][1] == "In scanner"
    assert rows[0][2] == "Task"
    assert rows[0][-1] == "1.2.0"


def test_recorded_time_totals_the_runs() -> None:
    rows = participant_rows(_cohort(_participant("0014", n_runs=4)))

    # Four half-hour runs.
    assert rows[0][6] == "2.0"


def test_the_participant_count_and_its_runs_are_both_reported() -> None:
    html = composition_html(_cohort(_participant("0014", n_runs=6), _participant("0015", n_runs=2)))

    assert "Participants aggregated" in html
    assert "Runs behind them" in html
    assert ">8<" in html


def test_a_small_cohort_says_why_it_draws_no_summary() -> None:
    """An absent median must not read as a measurement that failed."""
    html = composition_html(_cohort(_participant("0014"), _participant("0015")))

    assert "no cohort summary is drawn" in html
    assert "No medians, quantile bands or summary statistics are drawn" in html


def test_a_larger_cohort_states_what_it_may_draw() -> None:
    html = composition_html(_cohort(*(_participant(f"{index:04d}") for index in range(12))))

    assert "10th-to-90th" in html
    assert "no cohort summary is drawn" not in html


def test_a_larger_cohort_carries_the_note_on_reading_extremes() -> None:
    """Met up front, not for the first time halfway through a distribution."""
    html = composition_html(_cohort(*(_participant(f"{index:04d}") for index in range(12))))

    assert "On reading extremes" in html
    assert "convergence" in html
    assert "no p-values" in html or "reports no " in html


def test_a_skipped_participant_is_named_with_its_reason() -> None:
    html = composition_html(
        _cohort(
            _participant("0014"),
            not_aggregated=[NotAggregated(subject="0015", reason="No QC sidecar.")],
        )
    )

    assert "Found but not aggregated" in html
    assert "0015" in html
    assert "No QC sidecar." in html


def test_a_cohort_with_nobody_skipped_has_no_such_table() -> None:
    html = composition_html(_cohort(_participant("0014"), _participant("0015")))

    assert "Found but not aggregated" not in html


def test_a_mixed_cohort_warns_that_it_will_not_be_pooled() -> None:
    html = composition_html(
        _cohort(
            _participant("0014", context=AcquisitionContext.IN_SCANNER),
            _participant("0015", context=AcquisitionContext.OUT_OF_SCANNER),
        )
    )

    assert "spans more than one acquisition context" in html
    assert "describes neither group" in html


def test_a_single_context_cohort_carries_no_such_warning() -> None:
    html = composition_html(_cohort(_participant("0014"), _participant("0015")))

    assert "spans more than one acquisition context" not in html


def test_the_two_kinds_of_date_are_distinguished() -> None:
    """Cap ageing and pipeline change drift independently and cannot share an axis."""
    html = composition_html(_cohort(_participant("0014")))

    assert "not the processing date" in html


def test_a_participant_without_an_acquisition_date_still_gets_a_row() -> None:
    rows = participant_rows(_cohort(_participant("0014", acquired=None)))

    assert rows[0][0] == "0014"
    assert rows[0][7] is None


def test_a_participant_with_no_runs_reports_no_fabricated_totals() -> None:
    participant = SubjectSidecar(
        subject="0014",
        task="thermalactive",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=Paradigm.REST,
    )

    rows = participant_rows(_cohort(participant))

    assert rows[0][4] == 0
    assert rows[0][5] is None
    assert rows[0][6] is None


def test_the_report_never_claims_to_have_tested_anything() -> None:
    html = composition_html(_cohort(*(_participant(f"{index:04d}") for index in range(12))))

    assert "runs no statistical tests" in html
    assert "not as a confidence interval" in html


@pytest.mark.parametrize("paradigm", list(Paradigm))
def test_every_paradigm_has_a_reader_facing_label(paradigm) -> None:
    rows = participant_rows(_cohort(_participant("0014", paradigm=paradigm)))

    assert rows[0][2] in {"Task", "Rest"}
