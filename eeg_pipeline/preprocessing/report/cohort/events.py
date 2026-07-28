"""What was presented, before anything decided what to keep.

The rejection section asks what survived. This one asks what was there to survive, which is
a different question with a different failure mode: a participant who saw sixty trials of
one condition and twenty of another has an unbalanced design, and no retention figure can
distinguish that from a balanced design that lost trials unevenly. The two sections read
the same table from opposite ends deliberately.

The finding this exists to surface is a condition that is missing for some participants and
present for others. That is invisible per subject -- a report cannot know what the other
participants saw -- and it silently changes what a cohort contrast is over, because a
condition present in eighteen of twenty participants gets averaged as though it were
present in twenty.

Task cohorts only, and only for participants whose events named a condition. A recording
with trials but no condition column has nothing to break down.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import Paradigm
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

EVENTS_SECTION = "Events"
EVENTS_TITLE = "What was presented"
EVENTS_TAG = "cohort-events"


@dataclass(frozen=True)
class CohortEvents:
    """Trials presented per participant and condition.

    ``grid`` is participants by conditions, holding a missing value where a participant
    never saw a condition -- which is the cell this section exists to make visible, and
    which a zero would render as "saw it, got none of it".
    """

    grid: pd.DataFrame
    #: Participants whose paradigm is task but whose events named no condition.
    without_conditions: tuple[str, ...] = ()

    @property
    def n_participants(self) -> int:
        return int(len(self.grid))

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.grid.columns)

    @property
    def missing_cells(self) -> tuple[tuple[str, str], ...]:
        """Participant-and-condition pairs where nothing was presented."""
        absent = self.grid.isna()
        return tuple(
            (str(subject), str(condition))
            for subject in absent.index
            for condition in absent.columns
            if bool(absent.loc[subject, condition])
        )

    @property
    def is_complete(self) -> bool:
        return not self.missing_cells


def cohort_events(cohort: Cohort) -> CohortEvents | None:
    """Build the presented-trials grid, or nothing when no participant recorded one."""
    task = cohort.select(paradigm=Paradigm.TASK)
    rows: dict[str, dict[str, float]] = {}
    without: list[str] = []
    for participant in task.participants:
        if not participant.has_condition_evidence:
            without.append(participant.subject)
            continue
        table = participant.conditions
        totals = pd.to_numeric(table["n_total"], errors="coerce")
        entry = {
            str(name): float(value)
            for name, value in zip(table["condition"], totals)
            if np.isfinite(value)
        }
        if entry:
            rows[participant.subject] = entry
        else:
            without.append(participant.subject)
    if not rows:
        return None
    grid = pd.DataFrame.from_dict(rows, orient="index")
    return CohortEvents(
        grid=grid[sorted(grid.columns)].sort_index(),
        without_conditions=tuple(sorted(without)),
    )


def events_audit(events: CohortEvents) -> pd.DataFrame:
    """Every cell of the grid, one row each."""
    stacked = events.grid.stack(future_stack=True).reset_index()
    stacked.columns = ["subject", "condition", "n_total"]
    return stacked


def balance_frame(events: CohortEvents) -> pd.DataFrame:
    """Per participant: trials presented, and how uneven the design was for them.

    The imbalance is reported as the ratio of the smallest condition to the largest, which
    is scale-free and reads directly as "the sparsest condition had this share of the
    richest". A difference in counts would not be comparable between a participant who saw
    forty trials and one who saw four hundred.
    """
    rows = []
    for subject, values in events.grid.iterrows():
        counts = values.to_numpy(dtype=float)
        counts = counts[np.isfinite(counts)]
        if counts.size == 0:
            continue
        rows.append(
            {
                "subject": str(subject),
                "n_conditions": int(counts.size),
                "n_trials": float(counts.sum()),
                "smallest": float(counts.min()),
                "largest": float(counts.max()),
                "balance": (
                    float(counts.min() / counts.max()) if counts.max() > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def participant_table(events: CohortEvents) -> str:
    """Every participant, least balanced design first."""
    frame = balance_frame(events).sort_values("balance", na_position="first")
    rows = [
        [
            str(row["subject"]),
            int(row["n_conditions"]),
            f"{int(row['n_trials']):,}",
            f"{int(row['smallest']):,}",
            f"{int(row['largest']):,}",
            f"{float(row['balance']):.2f}" if np.isfinite(row["balance"]) else None,
        ]
        for _, row in frame.iterrows()
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Conditions"),
        Column("Trials"),
        Column("Sparsest condition"),
        Column("Richest condition"),
        Column("Smallest ÷ largest"),
    )
    return grid_table(columns, rows)


def condition_table(events: CohortEvents) -> str:
    """Every condition, with the number of participants that saw it at all."""
    rows = []
    for name in events.conditions:
        column = events.grid[name].dropna()
        rows.append(
            [
                str(name),
                int(len(column)),
                events.n_participants,
                f"{int(column.median()):,}" if not column.empty else None,
                f"{int(column.min()):,}" if not column.empty else None,
                f"{int(column.max()):,}" if not column.empty else None,
            ]
        )
    columns = (
        Column("Condition", align=Align.TEXT, code=True),
        Column("Presented to"),
        Column("Of participants"),
        Column("Median trials"),
        Column("Fewest"),
        Column("Most"),
    )
    return grid_table(columns, rows)


def missing_note(events: CohortEvents) -> str:
    """State which participants never saw which conditions, or that all of them did."""
    if events.is_complete:
        return (
            f"<p>Every one of the {events.n_participants} participant(s) saw all "
            f"{len(events.conditions)} condition(s), so a cohort contrast over any of them "
            "is over the whole cohort.</p>"
        )
    by_condition: dict[str, list[str]] = {}
    for subject, condition in events.missing_cells:
        by_condition.setdefault(condition, []).append(subject)
    described = "; ".join(
        f"<code>{condition}</code> is absent for {', '.join(sorted(subjects))}"
        for condition, subjects in sorted(by_condition.items())
    )
    return (
        f"<p><strong>Not every participant saw every condition.</strong> {described}. A "
        "contrast involving one of those conditions is over the participants that have it "
        "rather than over the cohort, and nothing downstream will say so &mdash; the "
        "average is taken over whoever contributed a value.</p>"
    )


def add_events_section(*, report: mne.Report, cohort: Cohort) -> CohortEvents | None:
    """Add the events section, or nothing when no participant recorded conditions."""
    events = cohort_events(cohort)
    if events is None:
        return None

    parts = [
        "<p>The rejection section asks what survived; this one asks what there was to "
        "survive. A participant who saw sixty trials of one condition and twenty of "
        "another has an unbalanced design, which no retention figure can tell apart from a "
        "balanced design that lost trials unevenly.</p>",
        missing_note(events),
        "<h4>By condition</h4>",
        condition_table(events),
        "<h4>By participant</h4>",
        participant_table(events),
        "<p>Balance is the sparsest condition as a share of the richest, which is "
        "scale-free: a difference in counts is not comparable between a participant who "
        "saw forty trials and one who saw four hundred.</p>",
    ]
    if events.without_conditions:
        parts.append(
            "<p>"
            + ", ".join(events.without_conditions)
            + " ran a task but recorded no condition breakdown, so they contribute to the "
            "retention totals and not to this section. Their trials are counted; which "
            "condition each belonged to was not written down.</p>"
        )

    report.add_html(
        html="".join(parts),
        title=EVENTS_TITLE,
        section=EVENTS_SECTION,
        tags=(EVENTS_TAG,),
        replace=True,
    )
    return events


__all__ = [
    "EVENTS_SECTION",
    "EVENTS_TAG",
    "EVENTS_TITLE",
    "CohortEvents",
    "add_events_section",
    "balance_frame",
    "cohort_events",
    "events_audit",
    "missing_note",
]
