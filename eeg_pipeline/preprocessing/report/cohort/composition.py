"""Who the cohort report describes, and who it does not.

This is the section every other one refers back to. A cohort figure is a claim about a
population, and the claim is only readable next to the population: twenty participants
whose median variance removed is 84% means one thing when all twenty were scanned on the
same day and another when six were added a year later under a different pipeline version.

Two things therefore appear here that a summary would normally omit. Participants the run
found and could not use are listed with the reason, because a document that silently drops
them reports a smaller study than the one that was run and gives the reader no way to
notice. And the band regime is stated outright, so a reader who wonders why no median is
drawn learns that the participant count cannot support one, rather than assuming the
measurement failed.

The section also carries the note on reading extremes. With this many panels, some
participant is unusual somewhere by chance alone, and a reader who meets that fact for the
first time in the middle of the gradient section will already have drawn a conclusion from
it.
"""

from __future__ import annotations

from typing import Any, Sequence

import mne
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    BandRegime,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    Paradigm,
    SubjectSidecar,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    Metric,
    grid_table,
    metric_table,
)

COMPOSITION_SECTION = "Cohort composition"
COMPOSITION_TITLE = "Who this report describes"
COMPOSITION_TAG = "cohort-composition"


_PARADIGM_LABELS = {
    Paradigm.TASK: "Task",
    Paradigm.REST: "Rest",
}

#: What each regime permits, phrased as what the reader will and will not see.
_REGIME_LABELS = {
    BandRegime.INDIVIDUALS: "Individual participants only — no cohort summary is drawn",
    BandRegime.MEDIAN_IQR: "Median and interquartile band",
    BandRegime.MEDIAN_IQR_DECILES: "Median, interquartile and 10th-to-90th bands",
}


def _duration_hours(participant: SubjectSidecar) -> float | None:
    if "duration_s" not in participant.runs.columns or participant.runs.empty:
        return None
    total = pd.to_numeric(participant.runs["duration_s"], errors="coerce").sum()
    return None if pd.isna(total) else float(total) / 3600.0


def _pipeline_version(participant: SubjectSidecar) -> str | None:
    return participant.versions.get("eeg_pipeline")


def participant_rows(cohort: Cohort) -> list[list[Any]]:
    """One row per aggregated participant, in the order the cohort holds them."""
    rows: list[list[Any]] = []
    for participant in cohort.participants:
        hours = _duration_hours(participant)
        rows.append(
            [
                participant.subject,
                _PARADIGM_LABELS[participant.paradigm],
                participant.task,
                participant.n_runs,
                _channel_count(participant),
                None if hours is None else f"{hours:.1f}",
                participant.acquisition_date,
                _pipeline_version(participant),
            ]
        )
    return rows


def _channel_count(participant: SubjectSidecar) -> int | None:
    if "n_channels" not in participant.runs.columns or participant.runs.empty:
        return None
    counts = pd.to_numeric(participant.runs["n_channels"], errors="coerce").dropna()
    if counts.empty:
        return None
    # The maximum rather than the first: a run that lost channels describes that run, and
    # the montage the participant was recorded with is what this column is naming.
    return int(counts.max())


def _summary_rows(cohort: Cohort, *, gates: BandGates) -> list[Metric]:
    regime = gates.regime_for(cohort.n_participants)
    paradigms = ", ".join(_PARADIGM_LABELS[paradigm] for paradigm in cohort.paradigms)
    total_runs = sum(participant.n_runs for participant in cohort.participants)
    rows = [
        Metric("Participants aggregated", cohort.n_participants, emphasis=True),
        Metric("Runs behind them", total_runs),
        Metric("Tasks", ", ".join(cohort.tasks) or None),
        Metric("Paradigms", paradigms or None),
        Metric("What the participant count supports", _REGIME_LABELS[regime]),
    ]
    if cohort.not_aggregated:
        rows.append(Metric("Found but not aggregated", len(cohort.not_aggregated)))
    return rows


def _not_aggregated_table(cohort: Cohort) -> str:
    if not cohort.not_aggregated:
        return ""
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Why not", align=Align.TEXT),
    )
    rows = [[entry.subject, entry.reason] for entry in cohort.not_aggregated]
    return (
        "<h4>Found but not aggregated</h4>"
        "<p>These participants have a subject report under the derivatives root and could "
        "not be read into the cohort. They are listed rather than dropped, because a "
        "denominator that quietly excludes them describes a smaller study than the one "
        "that was run.</p>" + grid_table(columns, rows)
    )




def _reading_note(cohort: Cohort, *, gates: BandGates) -> str:
    """The standing caution about reading extremes, stated once and up front."""
    if cohort.n_participants < gates.min_subjects_for_median:
        return (
            "<p>Every panel in this report shows each participant individually and states "
            "the number behind it. No medians, quantile bands or summary statistics are "
            "drawn anywhere, because a summary over this many participants would describe "
            "the arithmetic of a handful of values rather than a cohort.</p>"
        )
    return (
        "<p>This report describes a cohort. It runs no statistical tests, reports no "
        "p-values and fits no trend lines, and nothing in it grades a participant. Spread "
        "is drawn as the range the participants actually occupy, not as a confidence "
        "interval: the question a reader is asking here is where a participant sits among "
        "the others, which the observed distribution answers directly.</p>"
        "<p><strong>On reading extremes.</strong> This document carries dozens of panels. "
        "With that many measurements, some participant is at the edge of some distribution "
        "by chance alone, and a single extreme value is therefore expected rather than "
        "informative. What warrants a closer look is convergence: the same participant "
        "sitting at the edge of several measurements that share a mechanism &mdash; "
        "channel-level, decomposition-level, scanner-level or physiological.</p>"
    )


def composition_html(cohort: Cohort, *, gates: BandGates = DEFAULT_GATES) -> str:
    """Render the composition section."""
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Context", align=Align.TEXT),
        Column("Paradigm", align=Align.TEXT),
        Column("Task", align=Align.TEXT, code=True),
        Column("Runs"),
        Column("Channels"),
        Column("Recorded (h)"),
        Column("Acquired", align=Align.TEXT),
        Column("Pipeline", align=Align.TEXT, code=True),
    )
    return (
        metric_table(_summary_rows(cohort, gates=gates))
        + grid_table(columns, participant_rows(cohort))
        + "<p>Acquisition date indexes what changes in the recording &mdash; cap ageing, "
        "electrode wear, a replaced amplifier. It is deliberately not the processing date, "
        "which indexes what changes in the pipeline; the two drift independently and a "
        "panel that mixed them could not be read.</p>"
        + _not_aggregated_table(cohort)
        + _reading_note(cohort, gates=gates)
    )


def add_composition_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> None:
    """Add the composition section to a cohort report."""
    report.add_html(
        html=composition_html(cohort, gates=gates),
        title=COMPOSITION_TITLE,
        section=COMPOSITION_SECTION,
        tags=("summary", COMPOSITION_TAG),
        replace=True,
    )


__all__: Sequence[str] = [
    "COMPOSITION_SECTION",
    "COMPOSITION_TAG",
    "COMPOSITION_TITLE",
    "add_composition_section",
    "composition_html",
    "participant_rows",
]
