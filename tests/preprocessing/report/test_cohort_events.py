"""What was presented, and the cell that must not be a zero.

A condition a participant never saw and a condition they saw and got none of are different
facts with different consequences, and a grid that fills missing values with zero renders
them identically. The second is a rejection problem; the first silently changes what a
cohort contrast is over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.events import (
    balance_frame,
    cohort_events,
    condition_table,
    events_audit,
    missing_note,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)


def _participant(
    subject: str,
    *,
    presented: dict[str, float] | None = None,
    paradigm: Paradigm = Paradigm.TASK,
) -> SubjectSidecar:
    conditions = (
        pd.DataFrame(
            [
                {"condition": name, "n_total": total, "n_kept": total}
                for name, total in sorted((presented or {}).items())
            ]
        )
        if presented
        else pd.DataFrame()
    )
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=paradigm,
        conditions=conditions,
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


def test_the_grid_is_participants_by_condition() -> None:
    events = cohort_events(
        _cohort(
            _participant("0014", presented={"painful": 60, "neutral": 60}),
            _participant("0015", presented={"painful": 60, "neutral": 60}),
        )
    )

    assert events.conditions == ("neutral", "painful")
    assert events.n_participants == 2
    assert events.is_complete


def test_a_condition_a_participant_never_saw_is_missing_rather_than_zero() -> None:
    events = cohort_events(
        _cohort(
            _participant("0014", presented={"painful": 60, "neutral": 60}),
            _participant("0015", presented={"painful": 60}),
        )
    )

    assert np.isnan(events.grid.loc["0015", "neutral"])
    assert events.missing_cells == (("0015", "neutral"),)
    assert not events.is_complete


def test_a_condition_presented_and_entirely_rejected_is_still_presented() -> None:
    """Zero retained is a rejection fact; the presented count is what this section reads."""
    events = cohort_events(_cohort(_participant("0014", presented={"painful": 60, "warm": 0})))

    assert events.grid.loc["0014", "warm"] == 0
    assert events.is_complete


def test_a_missing_condition_is_named_with_its_consequence() -> None:
    note = missing_note(
        cohort_events(
            _cohort(
                _participant("0014", presented={"painful": 60, "neutral": 60}),
                _participant("0015", presented={"painful": 60}),
            )
        )
    )

    assert "0015" in note
    assert "neutral" in note
    assert "rather than over the cohort" in note


def test_a_complete_design_says_so() -> None:
    note = missing_note(
        _cohort_events_of({"0014": {"painful": 60}, "0015": {"painful": 60}})
    )

    assert "saw all" in note


def _cohort_events_of(spec: dict[str, dict[str, float]]):
    return cohort_events(
        _cohort(*(_participant(name, presented=counts) for name, counts in spec.items()))
    )


def test_balance_is_scale_free() -> None:
    """Forty trials against four hundred is not comparable as a difference in counts."""
    frame = balance_frame(
        _cohort_events_of(
            {"0014": {"a": 20, "b": 40}, "0015": {"a": 200, "b": 400}}
        )
    ).set_index("subject")

    assert frame.loc["0014", "balance"] == frame.loc["0015", "balance"] == 0.5


def test_each_condition_carries_how_many_participants_saw_it() -> None:
    html = condition_table(
        _cohort_events_of({"0014": {"painful": 60, "neutral": 60}, "0015": {"painful": 60}})
    )

    assert "painful" in html
    assert "neutral" in html


def test_a_task_participant_with_no_condition_column_is_named_not_dropped() -> None:
    events = cohort_events(
        _cohort(
            _participant("0014", presented={"painful": 60}),
            _participant("0015", presented=None),
        )
    )

    assert events.without_conditions == ("0015",)
    assert list(events.grid.index) == ["0014"]


def test_a_resting_state_cohort_has_no_section() -> None:
    assert cohort_events(_cohort(_participant("0014", paradigm=Paradigm.REST))) is None


def test_the_audit_table_holds_every_cell_including_the_missing_ones() -> None:
    """A reader checking the figure needs to see which cells were absent."""
    audit = events_audit(
        _cohort_events_of({"0014": {"painful": 60, "neutral": 60}, "0015": {"painful": 60}})
    )

    assert list(audit.columns) == ["subject", "condition", "n_total"]
    assert len(audit) == 4
    assert audit["n_total"].isna().sum() == 1
