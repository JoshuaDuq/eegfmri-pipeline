"""How many trials survived, and whether they survived evenly across the design.

The retained fraction is the number every study reports, and on its own it answers the
wrong question. Losing thirty percent of trials costs power, which is a nuisance. Losing
thirty percent of one condition and five percent of another costs the contrast, which is
the experiment -- and both cases print the same total.

So the total is reported because a reader expects it, and the per-condition breakdown is
reported because it is the one that can invalidate an analysis. The within-participant
range across conditions is given its own column: it is the quantity that says "this
participant's conditions were not rejected alike", and it survives averaging across the
cohort in a way the individual per-condition rates do not.

Task cohorts only. A resting-state recording has no trials to retain, and the section is
absent rather than empty for one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    BandRegime,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import Paradigm, SubjectSidecar
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    FLAG_COLOR,
    apply_report_style,
    report_image_format,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

REJECTION_SECTION = "Epoch rejection"
REJECTION_TITLE = "What survived rejection, and whether it survived evenly"
REJECTION_TAG = "cohort-rejection"

#: Prefix the epochs stage records each drop reason's count under.
REASON_PREFIX = "epochs_dropped_"


@dataclass(frozen=True)
class CohortRejection:
    """Trial retention per participant, and the per-condition breakdown behind it."""

    #: One row per participant: totals, retained fraction, and the across-condition range.
    participants: pd.DataFrame
    #: One row per participant and condition, for the participants that recorded any.
    conditions: pd.DataFrame

    @property
    def n_participants(self) -> int:
        return int(len(self.participants))

    @property
    def has_conditions(self) -> bool:
        return not self.conditions.empty

    @property
    def condition_names(self) -> tuple[str, ...]:
        if self.conditions.empty:
            return ()
        return tuple(sorted({str(name) for name in self.conditions["condition"]}))


def drop_reasons(participant: SubjectSidecar) -> dict[str, int]:
    """Counts per drop reason, as the epochs stage recorded them.

    Read from the recorded measurements rather than recomputed, so the cohort tally and the
    subject panel are the same count. Reasons are free text from MNE and autoreject and are
    not normalised here: collapsing two spellings into one would report a tally no stage
    ever measured.
    """
    found: dict[str, int] = {}
    for key, value in dict(participant.measurements).items():
        if not str(key).startswith(REASON_PREFIX):
            continue
        reason = str(key)[len(REASON_PREFIX) :]
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if reason and count > 0:
            found[reason] = count
    return found


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _condition_rows(participant: SubjectSidecar) -> list[dict[str, Any]]:
    table = participant.conditions
    if table.empty:
        return []
    rows = []
    for _, row in table.iterrows():
        total = _finite(row.get("n_total"))
        kept = _finite(row.get("n_kept"))
        rows.append(
            {
                "subject": participant.subject,
                "condition": str(row.get("condition")),
                "n_total": total,
                "n_kept": kept,
                # Absent rather than zero where the presented count is unknown: a retained
                # fraction with no denominator is not a fraction.
                "retained": kept / total if np.isfinite(total) and total > 0 else float("nan"),
            }
        )
    return rows


def cohort_rejection(cohort: Cohort) -> CohortRejection | None:
    """Assemble retention for the task participants, or nothing when none has trials."""
    task = cohort.select(paradigm=Paradigm.TASK)
    participant_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for participant in task.participants:
        measurements: Mapping[str, Any] = participant.measurements
        total = _finite(measurements.get("epochs_total"))
        kept = _finite(measurements.get("epochs_kept"))
        rows = _condition_rows(participant)
        condition_rows.extend(rows)

        retained_by_condition = [
            row["retained"] for row in rows if np.isfinite(row["retained"])
        ]
        if not np.isfinite(total) and rows:
            # A participant whose epochs stage recorded no totals can still be placed from
            # its own conditions, which sum to the same trials.
            total = float(np.nansum([row["n_total"] for row in rows]))
            kept = float(np.nansum([row["n_kept"] for row in rows]))
        if not np.isfinite(total) or total <= 0:
            continue

        participant_rows.append(
            {
                "subject": participant.subject,
                "n_total": total,
                "n_kept": kept,
                "retained": kept / total if np.isfinite(kept) else float("nan"),
                "n_conditions": len(retained_by_condition),
                "condition_range": (
                    float(max(retained_by_condition) - min(retained_by_condition))
                    if len(retained_by_condition) > 1
                    else float("nan")
                ),
                "n_reasons": len(drop_reasons(participant)),
            }
        )

    if not participant_rows:
        return None
    return CohortRejection(
        participants=pd.DataFrame(participant_rows).sort_values("subject").reset_index(drop=True),
        conditions=(
            pd.DataFrame(condition_rows).sort_values(["subject", "condition"]).reset_index(drop=True)
            if condition_rows
            else pd.DataFrame(
                columns=["subject", "condition", "n_total", "n_kept", "retained"]
            )
        ),
    )


def reason_totals(cohort: Cohort) -> pd.DataFrame:
    """Drop reasons pooled across the cohort, each with the participants that recorded it.

    A count and a participant count, not a rate. The denominators differ per reason -- a
    reason only some participants' pipelines can produce is not a reason the others scored
    zero on -- so dividing would produce a rate over a population that varies by row.
    """
    tallies: dict[str, dict[str, Any]] = {}
    for participant in cohort.select(paradigm=Paradigm.TASK).participants:
        for reason, count in drop_reasons(participant).items():
            entry = tallies.setdefault(reason, {"reason": reason, "n_epochs": 0, "subjects": []})
            entry["n_epochs"] += count
            entry["subjects"].append(participant.subject)
    if not tallies:
        return pd.DataFrame(columns=["reason", "n_epochs", "n_participants", "participants"])
    rows = [
        {
            "reason": entry["reason"],
            "n_epochs": entry["n_epochs"],
            "n_participants": len(entry["subjects"]),
            "participants": ", ".join(sorted(entry["subjects"])),
        }
        for entry in tallies.values()
    ]
    return pd.DataFrame(rows).sort_values("n_epochs", ascending=False).reset_index(drop=True)


def rejection_audit(rejection: CohortRejection) -> pd.DataFrame:
    """Exactly the plotted values: one row per participant and condition."""
    if rejection.conditions.empty:
        return rejection.participants.copy()
    return rejection.conditions.copy()


def _has_condition_spread(rejection: CohortRejection) -> bool:
    """Whether any participant has two conditions to be spread between.

    The figure exists to show where a participant's conditions sit inside their own range.
    A design with one condition has no inside, and the panel collapses to one dot per
    participant -- the retained column of the table beside it, drawn sideways.
    """
    if rejection.conditions.empty:
        return False
    counts = rejection.conditions.dropna(subset=["retained"]).groupby("subject").size()
    return bool((counts > 1).any())


def plot_retention(rejection: CohortRejection) -> plt.Figure:
    """Retained fraction per participant, with the per-condition spread drawn on it.

    The per-condition points sit on the participant's own row rather than in a separate
    panel, because the question is not what each condition retained but how far apart a
    participant's conditions were. Two panels would put the comparison on the reader.

    Only worth drawing where there is a spread to draw; see :func:`_has_condition_spread`.
    """
    apply_report_style()
    frame = rejection.participants.sort_values("retained", na_position="first")
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(
        figsize=(5.4, max(2.4, 0.26 * len(frame) + 1.4)), layout="constrained"
    )
    axis.scatter(
        frame["retained"].to_numpy(dtype=float),
        positions,
        s=34,
        color=AFTER_COLOR,
        zorder=3,
        label="All trials",
    )

    if rejection.has_conditions:
        by_subject = {
            subject: group for subject, group in rejection.conditions.groupby("subject")
        }
        labelled = False
        for position, subject in zip(positions, frame["subject"]):
            group = by_subject.get(subject)
            if group is None:
                continue
            values = group["retained"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            axis.plot(
                [values.min(), values.max()],
                [position, position],
                color=FLAG_COLOR,
                linewidth=1.2,
                alpha=0.7,
                zorder=2,
                label=None if labelled else "Per-condition range",
            )
            axis.scatter(values, np.full(values.shape, position), s=10, color=FLAG_COLOR, zorder=2)
            labelled = True

    axis.set_yticks(positions)
    axis.set_yticklabels(list(frame["subject"]))
    axis.set_xlim(0.0, 1.02)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_xlabel("Trials retained")
    axis.grid(axis="x", alpha=0.3)
    # Only what was actually drawn. A cohort with one condition has no per-condition range,
    # and a legend entry for it invites the reader to look for a span that does not exist.
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="lower left", frameon=False, fontsize="small")
    axis.set_title(f"Trial retention · {rejection.n_participants} participant(s)")
    return figure


def _percentage(value: object) -> str | None:
    number = _finite(value)
    return f"{number:.1%}" if np.isfinite(number) else None


def _count(value: object) -> str | None:
    number = _finite(value)
    return f"{int(round(number)):,}" if np.isfinite(number) else None


def participant_table(rejection: CohortRejection) -> str:
    """Every participant, worst retention first."""
    frame = rejection.participants.sort_values("retained", na_position="first")
    rows = [
        [
            str(row["subject"]),
            _count(row["n_kept"]),
            _count(row["n_total"]),
            _percentage(row["retained"]),
            int(row["n_conditions"]),
            _percentage(row["condition_range"]),
        ]
        for _, row in frame.iterrows()
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Retained"),
        Column("Presented"),
        Column("Share"),
        Column("Conditions"),
        Column("Across-condition range"),
    )
    return grid_table(columns, rows)


def condition_table(rejection: CohortRejection, *, gates: BandGates) -> str:
    """Each condition across the cohort, with its own participant count."""
    if not rejection.has_conditions:
        return ""
    rows = []
    for name, group in rejection.conditions.groupby("condition"):
        retained = group["retained"].to_numpy(dtype=float)
        retained = retained[np.isfinite(retained)]
        if retained.size == 0:
            continue
        regime = gates.regime_for(int(retained.size))
        rows.append(
            [
                str(name),
                int(retained.size),
                (
                    f"{float(np.median(retained)):.1%}"
                    if regime is not BandRegime.INDIVIDUALS
                    else None
                ),
                f"{float(retained.min()):.1%}",
                f"{float(retained.max()):.1%}",
            ]
        )
    if not rows:
        return ""
    columns = (
        Column("Condition", align=Align.TEXT, code=True),
        Column("Participants"),
        Column("Median retained"),
        Column("Lowest"),
        Column("Highest"),
    )
    return grid_table(columns, rows)


def reason_table(cohort: Cohort) -> str:
    """Drop reasons across the cohort, each with the participants that recorded it."""
    totals = reason_totals(cohort)
    if totals.empty:
        return ""
    rows = [
        [str(row["reason"]), int(row["n_epochs"]), int(row["n_participants"])]
        for _, row in totals.iterrows()
    ]
    columns = (
        Column("Reason", align=Align.TEXT, code=True),
        Column("Epochs dropped"),
        Column("Participants affected"),
    )
    return grid_table(columns, rows) + (
        "<p>Counts, not rates. A reason only some participants' pipelines can produce is "
        "not a reason the others scored zero on, so each row has its own denominator and "
        "dividing by the cohort would compute a rate over a population that changes by "
        "row.</p>"
    )


def add_rejection_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> CohortRejection | None:
    """Add the epoch-rejection section, or nothing for a cohort with no trials."""
    rejection = cohort_rejection(cohort)
    if rejection is None:
        return None

    # Drawn only where there is more than one condition to spread across. With one, every
    # row is a single dot at the participant's retained fraction and the panel is the first
    # column of the table below, drawn horizontally. With several, the span shows where each
    # condition sits inside a participant's range, which the range column can only summarise.
    spread = _has_condition_spread(rejection)
    if spread:
        report.add_figure(
            fig=plot_retention(rejection),
            title=REJECTION_TITLE,
            section=REJECTION_SECTION,
            tags=(REJECTION_TAG,),
            image_format=report_image_format(),
            replace=True,
        )
        plt.close("all")

    parts = [
        "<p>Losing trials costs power, which is a nuisance. Losing them unevenly across "
        "conditions confounds the contrast, which is the experiment &mdash; and both cases "
        "print the same total.</p>",
        participant_table(rejection),
    ]
    if not spread:
        parts.append(
            "<p>No figure is drawn. No participant contributed more than one condition, so "
            "there is no spread across conditions to see and a chart would be the retained "
            "column above with the numbers taken off it.</p>"
        )
    conditions = condition_table(rejection, gates=gates)
    if conditions:
        parts.append("<h4>By condition</h4>" + conditions)
        parts.append(
            "<p>Each condition is scored against the participants that recorded it, not "
            "against the cohort, so a condition only some participants saw carries its own "
            "denominator.</p>"
        )
    else:
        parts.append(
            "<p>No participant recorded a per-condition breakdown, so retention is "
            "reported over all trials only. A total cannot show differential rejection, "
            "which is the failure this section exists to surface.</p>"
        )
    reasons = reason_table(cohort)
    if reasons:
        parts.append("<h4>Why trials were dropped</h4>" + reasons)

    report.add_html(
        html="".join(parts),
        title="Retention per participant and condition",
        section=REJECTION_SECTION,
        tags=(REJECTION_TAG,),
        replace=True,
    )
    return rejection


__all__ = [
    "REASON_PREFIX",
    "REJECTION_SECTION",
    "REJECTION_TAG",
    "REJECTION_TITLE",
    "CohortRejection",
    "add_rejection_section",
    "cohort_rejection",
    "drop_reasons",
    "plot_retention",
    "reason_totals",
    "rejection_audit",
]
