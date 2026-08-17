"""Flagged recording time, laid out participant by run.

The subject report already says what fraction of a run was flagged. Reading forty of those
documents one after another is the only way to notice that it is always the last run, and
nobody reads forty documents that way.

That finding is about the study rather than about any participant: a session too long, a
cap drying out, an ordering effect nobody designed. It is invisible in the per-participant
column, which averages the good runs with the bad, and it is invisible in the per-run
column, which averages across participants who ran different numbers of runs. It needs the
grid.

The grid is therefore the figure, and the two margins are drawn beside it rather than
instead of it. No trend is fitted across run index: the whole report declines to read a
line off a few dozen points, and a run axis with four positions is the last place to start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    BandRegime,
    Contribution,
    pool_runs_rate,
    pool_participants_scalar,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.style import (
    apply_report_style,
    report_image_format,
    run_entity,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

CONTINUITY_SECTION = "Data quality over time"
CONTINUITY_TITLE = "Flagged time, by participant and run"
CONTINUITY_TAG = "cohort-continuity"

#: Beyond this many participants the row labels stop being legible and the heatmap becomes
#: a texture. It is still the right figure; only the labelling gives way.
_MAX_LABELLED_ROWS = 40


#: What the grid can be built from, in the order they are preferred.
#:
#: Flagged time is the quantity the section is about, but it counts ``BAD`` annotations and
#: a pipeline that annotates nothing reports zero for every run of every participant --
#: a structurally blank heatmap that reads as a pristine cohort rather than as an
#: unused column. The worst one-second excursion is measured for every run regardless, so
#: where nothing was annotated the grid falls back to it and says so.
MEASURES: tuple[tuple[str, str, str], ...] = (
    (
        "flagged_fraction",
        "Recording time flagged",
        "the fraction of each run marked bad",
    ),
    (
        "continuity_max_db",
        "Worst 1 s excursion (dB)",
        "the largest one-second amplitude excursion in each run",
    ),
)


@dataclass(frozen=True)
class CohortContinuity:
    """Time-resolved quality as a participant-by-run grid.

    ``grid`` is indexed by participant and columned by run position, holding a missing
    value where a participant contributed no such run. Missing is not zero: a participant
    who ran three runs did not have a clean fourth one.
    """

    grid: pd.DataFrame
    #: Which column the grid holds, and how to say so on the panel.
    measure: str = MEASURES[0][0]
    label: str = MEASURES[0][1]
    description: str = MEASURES[0][2]
    #: Heading per grid column, parallel to :attr:`run_positions`. Empty falls back to the
    #: keys themselves, which is never wrong -- only plainer than ``run-2``.
    headings: tuple[str, ...] = ()

    @property
    def is_fraction(self) -> bool:
        return self.measure == "flagged_fraction"

    @property
    def position_headings(self) -> tuple[str, ...]:
        """What each column is called on the panel."""
        return self.headings or self.run_positions

    @property
    def is_single_position(self) -> bool:
        """Whether there is only one run position, so the grid has no second dimension.

        A grid is read two ways: down a column, which asks whether a run position is
        consistently worse across the cohort, and across a row, which asks whether one
        participant degraded through their session. With one column neither question
        exists, and what is left is one value per participant -- which this report renders
        as a table, because a column of cells drawn as an image is a strip plot with a
        colour scale bolted to it.

        The ordinary case for resting state and for single-run EEG-only acquisitions.
        """
        return int(self.grid.shape[1]) <= 1

    @property
    def n_participants(self) -> int:
        return int(len(self.grid))

    @property
    def n_runs(self) -> int:
        return int(self.grid.notna().to_numpy().sum())

    @property
    def run_positions(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.grid.columns)

    @property
    def is_ragged(self) -> bool:
        """Whether participants contributed different numbers of runs."""
        return bool(self.grid.isna().to_numpy().any())


class RunPosition(NamedTuple):
    """A recording's column in the grid, and the heading that column is drawn under.

    Two fields rather than one because the heading cannot be recovered from the key. A key
    of ``2`` is a run entity the dataset carries and heads its column ``run-2``; the same
    key reached as an ordinal, because BIDS omitted the entity, heads it ``recording 2`` --
    the dataset does not number those, and the panel should not invent it. Deciding between
    them from the key alone is impossible, and guessing from the string is what produced a
    tick reading ``run-sub-01_task-rest_eeg``.
    """

    #: Grid column, and the value the audit table records.
    key: str
    #: Tick label and table heading.
    heading: str


def _positions(recordings: Sequence[object]) -> list[RunPosition]:
    """Assign one participant's recordings their columns in the grid.

    A run entity *is* the position, and is used in preference to the order the runs were
    read: a participant whose first run was dropped upstream has a ``run-2`` that belongs
    under ``run-2``, and sliding it into the first column would align it against everybody
    else's first run and manufacture exactly the ordering effect this panel exists to
    detect.

    Where BIDS omitted the run entity there is no token to read, and the position is the
    recording's ordinal within this participant instead. It emphatically cannot be the
    recording id: an id carries the subject label, so ``sub-0014_task-rest_eeg`` and
    ``sub-0015_task-rest_eeg`` are different strings naming the same position in each
    participant's session. Keyed on them, every participant took a column of its own and a
    six-participant rest cohort came out as a six-by-six diagonal matrix -- one filled cell
    per row, drawn as a heatmap, which cannot be read down a column and so cannot answer
    the only question the grid exists for.

    An ordinal is headed "recording N" rather than "run-N", because BIDS omitting the
    entity means the dataset does not number these and the panel should not invent it.
    """
    ordinal = 0
    positions: list[RunPosition] = []
    for recording in recordings:
        token = run_entity(recording)
        if token is not None:
            positions.append(RunPosition(token, f"run-{token}"))
            continue
        ordinal += 1
        positions.append(RunPosition(str(ordinal), f"recording {ordinal}"))
    return positions


def _grid_for(cohort: Cohort, column: str) -> tuple[pd.DataFrame, dict[str, str]] | None:
    rows: dict[str, dict[str, float]] = {}
    headings: dict[str, str] = {}
    for participant in cohort.participants:
        table = participant.runs
        if table.empty or column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        entry: dict[str, float] = {}
        for position, value in zip(_positions(list(table["run"])), values):
            if not np.isfinite(value):
                continue
            entry[position.key] = float(value)
            # A run entity is the more specific fact, so it wins the heading where a cohort
            # mixes recordings that carry one with recordings that do not.
            if position.heading.startswith("run-") or position.key not in headings:
                headings[position.key] = position.heading
        if entry:
            rows[participant.subject] = entry
    if not rows:
        return None
    grid = pd.DataFrame.from_dict(rows, orient="index")
    # Numerically where the run label is a number, so run-10 sorts after run-9 rather than
    # between run-1 and run-2.
    ordered = sorted(grid.columns, key=_column_order)
    return grid[ordered].sort_index(), {key: headings[key] for key in ordered}


def cohort_continuity(cohort: Cohort) -> CohortContinuity | None:
    """Build the participant-by-run grid from the first measure that separates anybody.

    A measure on which every run of every participant agrees cannot be read as a grid: it
    renders as one flat colour, and the colour scale beneath it becomes a fifth decimal
    place of nothing. That is not a pristine cohort, it is a column that was never
    populated -- flagged time counts ``BAD`` annotations, so a pipeline that writes none
    reports zero everywhere. The measured excursion, which is taken for every run whatever
    was annotated, then carries the panel instead.
    """
    for column, label, description in MEASURES:
        built = _grid_for(cohort, column)
        if built is None:
            continue
        grid, headings = built
        values = grid.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.max() - finite.min()) > 0.0:
            return CohortContinuity(
                grid=grid,
                measure=column,
                label=label,
                description=description,
                headings=tuple(headings[str(name)] for name in grid.columns),
            )
    return None


def _column_order(name: object) -> tuple[int, str]:
    text = str(name)
    return (0, f"{int(text):09d}") if text.isdigit() else (1, text)


def participant_totals(cohort: Cohort) -> dict[str, float]:
    """Flagged fraction per participant, pooled over the whole session.

    A sum over a sum rather than a median of the per-run fractions, because the quantity is
    defined over the session: every recorded second counts once, so a thirty-second run
    that was entirely flagged cannot outvote a clean twelve-minute one.
    """
    totals: dict[str, float] = {}
    for participant in cohort.participants:
        table = participant.runs
        if table.empty:
            continue
        fractions = pd.to_numeric(table.get("flagged_fraction"), errors="coerce")
        durations = pd.to_numeric(table.get("duration_s"), errors="coerce")
        usable = np.isfinite(fractions) & np.isfinite(durations) & (durations > 0)
        if not usable.any():
            continue
        totals[participant.subject] = pool_runs_rate(
            (fractions[usable] * durations[usable]).tolist(),
            durations[usable].tolist(),
        )
    return totals


def flagged_audit(continuity: CohortContinuity) -> pd.DataFrame:
    """Every plotted cell, one row each, so the figure can be checked without redrawing."""
    stacked = continuity.grid.stack(future_stack=True).reset_index()
    stacked.columns = ["subject", "run", "flagged_fraction"]
    return stacked.dropna(subset=["flagged_fraction"]).reset_index(drop=True)


def plot_flagged_time(continuity: CohortContinuity) -> plt.Figure:
    """The grid itself, with a missing run drawn as absent rather than as clean."""
    apply_report_style()
    values = continuity.grid.to_numpy(dtype=float)
    height = max(2.6, 0.26 * continuity.n_participants + 1.4)
    figure, axis = plt.subplots(
        figsize=(max(3.4, 0.7 * values.shape[1] + 2.2), height), layout="constrained"
    )
    masked = np.ma.masked_invalid(values)
    colormap = plt.get_cmap("Reds").with_extremes(bad="0.92")
    # Zero anchors the scale only where zero is the floor of the quantity. For a flagged
    # fraction it is: none of the run was flagged. For an excursion in decibels it is not,
    # and anchoring there paints a cohort spanning 19 to 25 dB in a single shade at the
    # dark end, which hides exactly the run-to-run variation the grid is read for.
    finite = values[np.isfinite(values)]
    vmin = 0.0 if continuity.is_fraction else float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    image = axis.imshow(
        masked,
        aspect="auto",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_xticks(range(values.shape[1]))
    axis.set_xticklabels(list(continuity.position_headings))
    if continuity.n_participants <= _MAX_LABELLED_ROWS:
        axis.set_yticks(range(continuity.n_participants))
        axis.set_yticklabels(list(continuity.grid.index))
    else:
        axis.set_yticks([0, continuity.n_participants - 1])
        axis.set_yticklabels([continuity.grid.index[0], continuity.grid.index[-1]])
    axis.set_xlabel("Run position within the session")
    bar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.85,
        format=PercentFormatter(xmax=1.0) if continuity.is_fraction else None,
    )
    bar.set_label(continuity.label)
    axis.set_title(
        f"{continuity.label} · {continuity.n_participants} participant(s), "
        f"{continuity.n_runs} run(s)"
    )
    return figure


def run_position_table(continuity: CohortContinuity, *, gates: BandGates) -> str:
    """The run margin: what each run position looked like across the cohort.

    Each column carries its own participant count, because a fourth run exists for only
    the participants who ran one and scoring it against the whole cohort would dilute it.
    """
    render = (lambda value: f"{value:.1%}") if continuity.is_fraction else (
        lambda value: f"{value:.1f}"
    )
    rows = []
    for name, heading in zip(continuity.run_positions, continuity.position_headings):
        column = continuity.grid[name].dropna()
        if column.empty:
            continue
        regime = gates.regime_for(len(column))
        rows.append(
            [
                heading,
                int(len(column)),
                render(float(column.median()))
                if regime is not BandRegime.INDIVIDUALS
                else None,
                render(float(column.min())),
                render(float(column.max())),
            ]
        )
    columns = (
        Column("Run position", align=Align.TEXT, code=True),
        Column("Participants"),
        Column(f"Median · {continuity.label}"),
        Column("Lowest"),
        Column("Highest"),
    )
    return grid_table(columns, rows)


def participant_measure_table(continuity: CohortContinuity) -> str:
    """Each participant's value for the measure, where the grid has only one column.

    The same numbers the grid would have held, in the shape they actually have. Sorted
    worst first, which is the only emphasis: nothing here is graded.
    """
    column = continuity.grid.iloc[:, 0].dropna()
    if column.empty:
        return ""
    render = (
        (lambda value: f"{value:.1%}") if continuity.is_fraction else (lambda value: f"{value:.1f}")
    )
    rows = [
        [str(subject), render(float(value))]
        for subject, value in column.sort_values(ascending=False).items()
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column(continuity.label),
    )
    return grid_table(columns, rows)


def participant_table(cohort: Cohort) -> str:
    """The participant margin, sorted by how much of the session was flagged."""
    totals = participant_totals(cohort)
    if not totals:
        return ""
    rows = [
        [subject, f"{fraction:.1%}"]
        for subject, fraction in sorted(totals.items(), key=lambda item: -item[1])
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Session time flagged"),
    )
    return grid_table(columns, rows)


def _add_single_position(
    *,
    report: mne.Report,
    cohort: Cohort,
    continuity: CohortContinuity,
) -> CohortContinuity:
    """Report the measure as a table, for a cohort with one run position.

    No figure. The grid's argument is that reading down a column and reading across a row
    ask two questions no margin can answer, and with one column there is no down and no
    across -- what is left is one value per participant, which this report renders as a
    table everywhere else. Drawn as a one-column image it would be a strip plot carrying a
    colour scale, and a reader would go looking for the second dimension it implies.

    Said out loud rather than silently omitted, because a section that simply lost its
    figure reads as a rendering failure.
    """
    parts = [
        f"<p>Every participant contributed one recording, so there is no run axis to lay "
        f"out and {continuity.label.lower()} is reported per participant below. The grid "
        "this section usually draws exists to be read two ways &mdash; down a column, for "
        "whether a run position is consistently worse across the cohort, and across a row, "
        "for whether a participant degraded through their session. Neither question exists "
        "for a single recording, and one column drawn as a heatmap would imply a second "
        "dimension the data does not have.</p>",
        participant_measure_table(continuity),
    ]
    if not continuity.is_fraction:
        parts.append(
            "<p>Reported as the measured excursion rather than as flagged time, because no "
            "recording in this cohort carried a <code>BAD</code> annotation. That is a "
            "column nothing populated, not a cohort with nothing wrong with it. The "
            "excursion is measured for every recording whatever was annotated.</p>"
        )
    report.add_html(
        html="".join(part for part in parts if part),
        title=CONTINUITY_TITLE,
        section=CONTINUITY_SECTION,
        tags=(CONTINUITY_TAG,),
        replace=True,
    )
    return continuity


def add_continuity_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> CohortContinuity | None:
    """Add the flagged-time section, or nothing when no participant measured it."""
    continuity = cohort_continuity(cohort)
    if continuity is None:
        return None

    if continuity.is_single_position:
        return _add_single_position(report=report, cohort=cohort, continuity=continuity)

    # Titled after the measure the grid actually holds. Headed "flagged time" while showing
    # an excursion, the panel would name a quantity it does not draw -- the one thing a
    # figure caption must never do.
    report.add_figure(
        fig=plot_flagged_time(continuity),
        title=f"{continuity.label}, by participant and run",
        section=CONTINUITY_SECTION,
        tags=(CONTINUITY_TAG,),
        image_format=report_image_format(has_dense_image=True),
        replace=True,
    )
    plt.close("all")

    parts = [
        f"<p>Each cell is one run, coloured by {continuity.description}. Reading down a "
        "column asks whether a run position is consistently worse across the cohort, which "
        "is a fact about the session design rather than about any participant, and which no "
        "subject report can show. Reading across a row asks whether one participant "
        "degraded through their session.</p>",
    ]
    if not continuity.is_fraction:
        parts.append(
            "<p>The grid is drawn from the measured excursion rather than from flagged "
            "time, because no run in this cohort carried a <code>BAD</code> annotation and "
            "a flagged-time grid would therefore have been one flat colour. That is a "
            "column nothing populated, not a cohort with nothing wrong with it, and the "
            "two must not look alike. The excursion is measured for every run whatever was "
            "annotated.</p>"
        )
    parts += [
        "<p>A blank cell is a run that participant did not contribute, drawn as absent "
        "rather than as clean: an unrecorded run and a spotless one are the same colour in "
        "any scheme that fills missing values with zero.</p>",
        "<h4>By run position</h4>",
        run_position_table(continuity, gates=gates),
    ]
    if continuity.is_ragged:
        parts.append(
            "<p>Participants contributed different numbers of runs, so each run position "
            "above carries its own participant count. A late run exists only for the "
            "participants who ran one, and scoring it against the whole cohort would "
            "dilute exactly the effect the grid is being read for.</p>"
        )
    # The participant margin is flagged time over the whole session, which is only worth a
    # table when something was flagged. Where the grid fell back to the excursion it would
    # be a column of zeroes restating the note above.
    margin = participant_table(cohort) if continuity.is_fraction else ""
    if margin:
        parts.append("<h4>By participant</h4>" + margin)
    parts.append(
        "<p>No trend is fitted across run position. A slope read off a handful of run "
        "positions is not a finding, and the grid above is the evidence.</p>"
    )

    report.add_html(
        html="".join(parts),
        title="Reading the grid",
        section=CONTINUITY_SECTION,
        tags=(CONTINUITY_TAG,),
        replace=True,
    )
    return continuity


def cohort_flagged_fraction(cohort: Cohort, *, gates: BandGates = DEFAULT_GATES):
    """Flagged session time pooled across participants, with its own denominator."""
    totals = participant_totals(cohort)
    if not totals:
        return None
    runs = {
        participant.subject: participant.n_runs for participant in cohort.participants
    }
    return pool_participants_scalar(
        [
            Contribution(
                subject=subject, value=np.asarray([value]), n_runs=runs.get(subject, 0)
            )
            for subject, value in totals.items()
        ],
        gates=gates,
    )


__all__ = [
    "CONTINUITY_SECTION",
    "CONTINUITY_TAG",
    "CONTINUITY_TITLE",
    "CohortContinuity",
    "RunPosition",
    "add_continuity_section",
    "cohort_continuity",
    "cohort_flagged_fraction",
    "flagged_audit",
    "participant_measure_table",
    "participant_totals",
    "plot_flagged_time",
]
