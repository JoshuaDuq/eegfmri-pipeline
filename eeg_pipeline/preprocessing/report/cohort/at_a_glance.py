"""The subject report's landing panel, with each value as a distribution.

A reader who knows the subject report arrives here looking for the same short list they
read at the top of every participant's document. Giving them a different list, in a
different order, under different names, costs them the mapping they already have -- so the
rows and their order are taken from the subject panel rather than chosen again here.

What changes is the right-hand side. A single value becomes a median with a range and a
denominator, and the denominator is per row: a participant contributes to "components
excluded" and not to "beat-marker agreement", and a table that printed one count in its
header would misstate every row but one.

The median is withheld below the gate. A row that shows a range and no median is telling
the reader that five participants cannot support a summary, which is a fact about the
cohort rather than a gap in the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.at_a_glance import HEADLINES, Headline
from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    BandRegime,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

GLANCE_SECTION = "At a glance"
GLANCE_TITLE = "The cohort in one screen"
GLANCE_TAG = "cohort-glance"


@dataclass(frozen=True)
class GlanceRow:
    """One headline measurement, summarised over the participants that recorded it."""

    key: str
    label: str
    section: str
    values: dict[str, float]
    regime: BandRegime
    format: Callable[[Any], str]

    @property
    def n_subjects(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float | None:
        if self.regime is BandRegime.INDIVIDUALS or not self.values:
            return None
        return float(np.median(list(self.values.values())))

    @property
    def span(self) -> tuple[float, float] | None:
        if not self.values:
            return None
        numbers = list(self.values.values())
        return float(min(numbers)), float(max(numbers))

    @property
    def extremes(self) -> tuple[str, str] | None:
        """The participants at each end, named by position rather than flagged.

        ``None`` when a single participant holds both ends, since naming them twice would
        read as two findings.
        """
        if len(self.values) < 2:
            return None
        ordered = sorted(self.values.items(), key=lambda item: item[1])
        return ordered[0][0], ordered[-1][0]


def _numeric(value: Any) -> float:
    """Coerce a recorded measurement to a number, or to missing.

    Booleans are rejected rather than counted as one and zero: ``True`` is a state a stage
    recorded, and averaging states produces a number no measurement supports.
    """
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def glance_rows(
    cohort: Cohort,
    *,
    gates: BandGates = DEFAULT_GATES,
    headlines: Sequence[Headline] = HEADLINES,
) -> list[GlanceRow]:
    """Summarise each headline over the participants that recorded it.

    A headline no participant recorded produces no row, matching what the subject panel
    does for a stage that did not run: an EEG-only cohort has no marker agreement, and a
    row reading "—" for it would state that the measurement was attempted.
    """
    rows: list[GlanceRow] = []
    for headline in headlines:
        values = {}
        for participant in cohort.participants:
            number = _numeric(dict(participant.measurements).get(headline.key))
            if np.isfinite(number):
                values[participant.subject] = number
        if not values:
            continue
        rows.append(
            GlanceRow(
                key=headline.key,
                label=headline.label,
                section=headline.section,
                values=values,
                regime=gates.regime_for(len(values)),
                format=headline.format,
            )
        )
    return rows


def _rendered(row: GlanceRow, value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return row.format(value)
    except (TypeError, ValueError):
        return None


def glance_frame(rows: Sequence[GlanceRow]) -> pd.DataFrame:
    """The table's own values, for the audit file."""
    return pd.DataFrame(
        [
            {
                "measurement": row.key,
                "label": row.label,
                "n_subjects": row.n_subjects,
                "median": row.median,
                "lowest": None if row.span is None else row.span[0],
                "highest": None if row.span is None else row.span[1],
                "subject": subject,
                "value": value,
            }
            for row in rows
            for subject, value in sorted(row.values.items())
        ]
    )


def glance_html(cohort: Cohort, *, gates: BandGates = DEFAULT_GATES) -> str:
    """Render the landing table, or say why there is nothing on it."""
    rows = glance_rows(cohort, gates=gates)
    if not rows:
        return (
            "<p>No participant recorded any of the headline measurements, so there is "
            "nothing to summarise here. The sections below report whatever was measured.</p>"
        )

    table_rows = []
    for row in rows:
        span = row.span
        extremes = row.extremes
        table_rows.append(
            [
                row.label,
                row.n_subjects,
                _rendered(row, row.median),
                None if span is None else _rendered(row, span[0]),
                None if span is None else _rendered(row, span[1]),
                None if extremes is None else f"{extremes[0]} … {extremes[1]}",
                row.section,
            ]
        )
    columns = (
        Column("Measurement", align=Align.TEXT),
        Column("Participants"),
        Column("Median"),
        Column("Lowest"),
        Column("Highest"),
        Column("At each end", align=Align.TEXT, code=True),
        Column("Evidence in", align=Align.TEXT),
    )

    withheld = [row for row in rows if row.regime is BandRegime.INDIVIDUALS]
    document = (
        "<p>The same short list the subject report opens with, each value as a "
        "distribution. Every row carries its own participant count, because a participant "
        "contributes to some of these and not to others &mdash; one count in the heading "
        "would misstate every row but one.</p>" + grid_table(columns, table_rows)
    )
    document += (
        "<p>The two participants named at each end are the ones holding the extremes of "
        "that row. They are named by position, not flagged: with this many measurements "
        "somebody sits at an end of each one, and doing so is not a finding.</p>"
    )
    if withheld:
        document += (
            "<p>No median is shown for "
            + ", ".join(f"<em>{row.label}</em>" for row in withheld)
            + f": fewer than {gates.min_subjects_for_median} participants recorded them, "
            "which is too few for a summary to describe anything the range does not.</p>"
        )
    return document


def add_at_a_glance_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> None:
    """Add the cohort landing panel."""
    report.add_html(
        html=glance_html(cohort, gates=gates),
        title=GLANCE_TITLE,
        section=GLANCE_SECTION,
        tags=("summary", GLANCE_TAG),
        replace=True,
    )


__all__ = [
    "GLANCE_SECTION",
    "GLANCE_TAG",
    "GLANCE_TITLE",
    "GlanceRow",
    "add_at_a_glance_section",
    "glance_frame",
    "glance_html",
    "glance_rows",
]
