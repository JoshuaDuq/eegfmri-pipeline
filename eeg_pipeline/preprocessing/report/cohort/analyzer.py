"""What the pulse correction had to work with, across the cohort.

Analyzer removes the cardiac artifact by averaging around a marker train it detected
itself. Everything downstream of that assumes the train was complete and in the right
places, and nothing downstream of it can tell that it was not: a correction built on a
train that lost a fifth of its beats produces a clean-looking recording with a fifth of the
artifact left in.

Two independent detectors are therefore compared where a recording carried an ECG channel
-- Analyzer's markers against R peaks detected from the signal -- and the physiology is
reported beside them. The physiology is here for one reason: an interval series that is
implausible as a heart rate is a detector failure, not an unusual participant, and it is
the only check available where there is no second detector to disagree with.

No thresholds are applied to any of it. What counts as an acceptable agreement depends on
the montage, the sequence and what the correction is for, and a cutoff invented here would
be applied to every study that ever runs this. Participants are placed in a sorted table
and named by position.

No figure either. Every quantity here is one number per participant, and a chart of one
number per participant is the table drawn with dots -- carrying less than the table, which
also holds the worst run, the dropout rate and the run count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    pool_runs_rate,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import AcquisitionContext
from eeg_pipeline.preprocessing.report.tables import MISSING, Align, Column, grid_table

ANALYZER_SECTION = "Scanner artifact correction (Analyzer)"
ANALYZER_TITLE = "What the pulse correction had to work with"
ANALYZER_TAG = "cohort-analyzer"

#: Intervals outside this range are not a heart rate. Used to count implausible intervals,
#: never to reject a participant: the count is the measurement, and what it means about a
#: recording is the reader's call.
PLAUSIBLE_BPM = (30.0, 220.0)


@dataclass(frozen=True)
class AnalyzerCohort:
    """Per-participant marker agreement and cardiac physiology.

    One row per participant, with a missing value wherever a participant's runs supported
    no such measurement. Missing rather than absent, so that every panel below counts its
    own contributors and the section cannot claim a denominator one of its panels did not
    have.
    """

    frame: pd.DataFrame

    @property
    def n_participants(self) -> int:
        return int(len(self.frame))

    def contributors(self, column: str) -> pd.DataFrame:
        if column not in self.frame.columns:
            return self.frame.iloc[0:0]
        return self.frame[np.isfinite(self.frame[column])]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """A column as floats, or an all-missing column of the right length when it is absent.

    Always a :class:`~pandas.Series`, so a caller can index it against another column
    without discovering that one of them degraded to an array. A sidecar legitimately
    lacks these columns -- they are required only of an in-scanner acquisition -- so an
    absent column is an ordinary case rather than a malformed table.
    """
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), np.nan), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _participant_row(subject: str, runs: pd.DataFrame) -> dict[str, object]:
    """Reduce one participant's runs to the row the cohort panels read.

    Sensitivity and precision are pooled from their primitive counts. They answer different
    questions and therefore have different denominators: detected ECG beats for sensitivity,
    and Analyzer markers for precision.
    """
    matched = _numeric(runs, "n_matched_beats")
    detected = _numeric(runs, "n_detected_beats")
    markers = _numeric(runs, "n_markers")
    sensitivity_runs = np.isfinite(matched) & np.isfinite(detected) & (detected > 0)
    precision_runs = np.isfinite(matched) & np.isfinite(markers) & (markers > 0)

    row: dict[str, object] = {"subject": subject, "n_runs": int(len(runs))}
    if sensitivity_runs.any():
        row["marker_agreement"] = pool_runs_rate(
            matched[sensitivity_runs].tolist(), detected[sensitivity_runs].tolist()
        )
        run_sensitivity = matched[sensitivity_runs] / detected[sensitivity_runs]
        row["worst_run_agreement"] = float(run_sensitivity.min())
        row["n_detected_beats"] = float(detected[sensitivity_runs].sum())
    else:
        row["marker_agreement"] = float("nan")
        row["worst_run_agreement"] = float("nan")
        row["n_detected_beats"] = float("nan")
    if precision_runs.any():
        row["marker_precision"] = pool_runs_rate(
            matched[precision_runs].tolist(), markers[precision_runs].tolist()
        )
        row["n_markers"] = float(markers[precision_runs].sum())
    else:
        row["marker_precision"] = float("nan")
        row["n_markers"] = float("nan")

    rate = _numeric(runs, "median_bpm")
    measured_rate = rate[np.isfinite(rate)]
    # The median across runs: each run is an independent estimate of the same resting
    # physiology, and a longer run does not estimate it more correctly.
    row["median_bpm"] = float(measured_rate.median()) if not measured_rate.empty else float("nan")
    row["implausible_rate"] = (
        bool(
            (row["median_bpm"] < PLAUSIBLE_BPM[0]) or (row["median_bpm"] > PLAUSIBLE_BPM[1])
        )
        if np.isfinite(row["median_bpm"])
        else False
    )

    beats = _numeric(runs, "n_beats")
    dropouts = _numeric(runs, "beat_dropouts")
    countable = np.isfinite(beats) & np.isfinite(dropouts) & (beats > 0)
    row["dropout_fraction"] = (
        pool_runs_rate(dropouts[countable].tolist(), beats[countable].tolist())
        if countable.any()
        else float("nan")
    )
    row["n_beats"] = float(beats[countable].sum()) if countable.any() else float("nan")
    return row


def analyzer_cohort(cohort: Cohort) -> AnalyzerCohort | None:
    """Assemble the per-participant frame, or nothing outside a scanner.

    Restricted to in-scanner participants by the recorded context rather than by whether a
    number happens to be present. A recording made outside a bore has no pulse correction
    to describe, and a row of missing values for it would put it in a denominator it does
    not belong to.
    """
    in_scanner = cohort.select(context=AcquisitionContext.IN_SCANNER)
    rows = [
        _participant_row(participant.subject, participant.runs)
        for participant in in_scanner.participants
        if not participant.runs.empty
    ]
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    if not np.isfinite(
        frame[["marker_agreement", "marker_precision", "median_bpm", "dropout_fraction"]]
    ).any().any():
        return None
    return AnalyzerCohort(frame=frame)


def analyzer_audit(analyzer: AnalyzerCohort) -> pd.DataFrame:
    """Exactly the plotted values, one row per participant."""
    return analyzer.frame.copy()


def _percentage(value: object) -> str | None:
    number = float(value) if value is not None else float("nan")
    return f"{number:.1%}" if np.isfinite(number) else None


def _decimal(value: object, *, places: int = 1) -> str | None:
    number = float(value) if value is not None else float("nan")
    return f"{number:.{places}f}" if np.isfinite(number) else None


def analyzer_table(analyzer: AnalyzerCohort) -> str:
    """Every participant, sorted by the agreement that decides what the rest means."""
    frame = analyzer.frame.sort_values(
        "marker_agreement", ascending=True, na_position="last"
    )
    rows = [
        [
            str(row["subject"]),
            int(row["n_runs"]),
            _percentage(row["marker_agreement"]),
            _percentage(row["marker_precision"]),
            _percentage(row["worst_run_agreement"]),
            _decimal(row["median_bpm"]),
            _percentage(row["dropout_fraction"]),
        ]
        for _, row in frame.iterrows()
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Runs"),
        Column("Beat sensitivity"),
        Column("Marker precision"),
        Column("Worst-run sensitivity"),
        Column("Median bpm"),
        Column("Interval dropouts"),
    )
    return grid_table(columns, rows)


def implausible_note(analyzer: AnalyzerCohort) -> str:
    """Name the participants whose interval series is not a heart rate.

    An algebraic statement rather than an empirical one, which is why this section is
    allowed to make it: a median interval outside the plausible range is not an unusual
    participant but a detector that was not tracking beats, and the correction built on it
    describes something other than the pulse.
    """
    flagged = analyzer.frame[analyzer.frame["implausible_rate"].astype(bool)]
    if flagged.empty:
        return (
            "<p>Every participant's median interval sits inside a physiologically possible "
            f"heart rate ({PLAUSIBLE_BPM[0]:.0f}&ndash;{PLAUSIBLE_BPM[1]:.0f} bpm), so no "
            "interval series can be ruled out as a detection failure on its own terms.</p>"
        )
    named = ", ".join(str(subject) for subject in flagged["subject"])
    return (
        f"<p><strong>{named}</strong> recorded a median interval outside a physiologically "
        f"possible heart rate ({PLAUSIBLE_BPM[0]:.0f}&ndash;{PLAUSIBLE_BPM[1]:.0f} bpm). "
        "That is a statement about the detector rather than about the participant: an "
        "interval series that is not a heart rate was not tracking beats, and the pulse "
        "correction built on it removed something other than the pulse.</p>"
    )


def add_analyzer_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> AnalyzerCohort | None:
    """Add the Analyzer section, or nothing when no participant was in a scanner."""
    analyzer = analyzer_cohort(cohort)
    correction = correction_html(cohort)
    if analyzer is None:
        # No agreement to measure and no heart rate, which is what a cohort whose runs
        # carry no marker train at all looks like. The worklist is then the only thing this
        # section has to say, and it is also the most important: skipping it with the
        # agreement panels made the worst case the one that rendered nothing.
        if correction:
            report.add_html(
                html=correction,
                title="Runs the pulse correction never ran on",
                section=ANALYZER_SECTION,
                tags=(ANALYZER_TAG,),
                replace=True,
            )
        return None

    agreement = analyzer.contributors("marker_agreement")
    parts = [
        "<p>Analyzer builds its pulse correction by averaging around a marker train it "
        "detected itself, and nothing downstream can tell whether that train was complete: "
        "a correction built on a train missing a fifth of its beats leaves a fifth of the "
        "artifact behind in a recording that looks clean.</p>",
        f"<p>Agreement is measured for the {len(agreement)} of {analyzer.n_participants} "
        "in-scanner participant(s) whose recording carried an ECG channel, by comparing "
        "Analyzer's markers against R peaks detected independently from that signal. Where "
        "there is no ECG channel there is one detector rather than two and nothing to "
        "reconcile, so those participants carry no agreement rather than a zero.</p>",
        analyzer_table(analyzer),
        implausible_note(analyzer),
        "<p>No threshold is applied to agreement. What counts as acceptable depends on the "
        "montage, the sequence and what the correction is for, so participants are placed "
        "in the distribution and named by position rather than graded.</p>",
        # No figure. Everything this section measures is one number per participant, and a
        # strip plot of one number per participant is the table above drawn with dots: it
        # carries less, since the table also holds the worst run, the dropout rate and the
        # run count. The table is sorted by agreement, which is the ordering the figure
        # existed to show.
        "<p>This section has no figure. Every quantity in it is a single number per "
        "participant, which the sorted table above already places them by; a chart of the "
        "same column would cost a reader attention and tell them less than the table it "
        "sits beside.</p>",
        # Placed last because it is the actionable part: everything above describes how well
        # the correction worked where it ran, and this says where it never ran at all.
        correction,
    ]
    report.add_html(
        html="".join(parts),
        title="Marker agreement and cardiac physiology",
        section=ANALYZER_SECTION,
        tags=(ANALYZER_TAG,),
        replace=True,
    )
    return analyzer


@dataclass(frozen=True)
class UncorrectedRun:
    """One run the upstream pulse correction never ran on."""

    subject: str
    run: str
    marker_count: int
    beat_source: str | None
    #: ``None`` where neither beat source resolved, so the harm could not be measured.
    residual_uv: float | None


@dataclass(frozen=True)
class ParticipantCorrection:
    """One participant's share of runs the pulse correction never ran on."""

    subject: str
    n_runs: int
    n_uncorrected: int
    #: Which runs, compactly: ``2-6`` for a consecutive block, ``2, 3, 5`` for scattered
    #: ones. The distinction is mechanical -- a lead that came off at run 2 and stayed off
    #: is a different fault from three intermittent dropouts -- and it is the one thing a
    #: sorted worklist cannot show.
    affected_runs: str
    worst_residual_uv: float | None


def _run_label(recording_id: object) -> str:
    """How a recording is named in the worklist.

    The run entity where the dataset carries one, since the participant column already
    disambiguates. Where BIDS omitted it -- the ordinary case for resting-state and baseline
    recordings, which go in the bore and fail the same way -- the label is what remains after
    the subject prefix, because a worklist has to name the file to re-export. Not prefixed
    with ``run-``: that would invent an entity the dataset does not have.
    """
    text = str(recording_id)
    _, separator, tail = text.partition("_run-")
    if separator:
        return tail.split("_")[0]
    parts = text.split("_")
    if len(parts) > 1 and parts[0].startswith("sub-"):
        return "_".join(parts[1:])
    return text


#: Consecutive runs below this many are listed rather than collapsed to a range.
#:
#: The range notation exists to signal a block -- a lead that came off and stayed off --
#: and two adjacent runs are not a block. Rendering them ``2-3`` is no shorter than
#: ``2, 3`` and spends the notation's meaning on a case that does not have it.
MINIMUM_RUN_SPAN = 3


def _compact_runs(labels: Sequence[str]) -> str:
    """Render run labels as ranges where enough of them are consecutive."""
    numeric = sorted(int(label) for label in labels if str(label).isdigit())
    if len(numeric) != len(labels):
        return ", ".join(str(label) for label in labels)
    spans: list[str] = []
    start = previous = numeric[0]

    def flush(first: int, last: int) -> None:
        if last - first + 1 >= MINIMUM_RUN_SPAN:
            spans.append(f"{first}-{last}")
        else:
            spans.extend(str(value) for value in range(first, last + 1))

    for value in numeric[1:]:
        if value == previous + 1:
            previous = value
            continue
        flush(start, previous)
        start = previous = value
    flush(start, previous)
    return ", ".join(spans)


def _residual_rows(participant) -> list[tuple[str, int, str | None, float | None]]:
    """``(run label, marker count, beat source, residual)`` for one participant.

    Returns nothing where the sidecar predates these columns, so a cohort assembled from
    older sidecars renders the rest of the section and omits this part rather than failing.
    """
    runs = participant.runs
    if runs.empty or "pulse_marker_count" not in runs.columns:
        return []
    counts = pd.to_numeric(runs["pulse_marker_count"], errors="coerce")
    residuals = (
        pd.to_numeric(runs["bcg_residual_uv"], errors="coerce")
        if "bcg_residual_uv" in runs.columns
        else pd.Series([np.nan] * len(runs))
    )
    sources = runs["beat_source"] if "beat_source" in runs.columns else pd.Series([None] * len(runs))
    rows = []
    for recording, count, residual, source in zip(runs["run"], counts, residuals, sources):
        if not np.isfinite(count):
            continue
        rows.append(
            (
                _run_label(recording),
                int(count),
                None if pd.isna(source) else str(source),
                float(residual) if np.isfinite(residual) else None,
            )
        )
    return rows


def uncorrected_runs(cohort: Cohort) -> list[UncorrectedRun]:
    """Every run with no Analyzer marker train, worst residual first.

    Marker absence is the criterion, not the residual. No markers means the upstream
    correction had no beat train to key on, so no subtraction was applied -- which is true
    whatever the residual came out at, and remains true for a run whose residual could not
    be measured at all. Those sort last but are never dropped: a run where both beat
    sources failed is the one most in need of re-exporting, and a worklist that omitted it
    for want of a number would be exactly backwards.
    """
    found: list[UncorrectedRun] = []
    for participant in cohort.participants:
        for label, count, source, residual in _residual_rows(participant):
            if count == 0:
                found.append(
                    UncorrectedRun(participant.subject, label, count, source, residual)
                )
    # Descending residual, unmeasured last, then by participant and run so the order is
    # stable across rebuilds rather than depending on how the sidecars were read.
    return sorted(
        found,
        key=lambda row: (
            row.residual_uv is None,
            -(row.residual_uv or 0.0),
            row.subject,
            row.run,
        ),
    )


def participant_correction_rows(cohort: Cohort) -> list[ParticipantCorrection]:
    """Per participant: how many runs went uncorrected, and which ones."""
    rows: list[ParticipantCorrection] = []
    for participant in cohort.participants:
        measured = _residual_rows(participant)
        if not measured:
            continue
        affected = [label for label, count, _, _ in measured if count == 0]
        residuals = [
            residual
            for label, count, _, residual in measured
            if count == 0 and residual is not None
        ]
        rows.append(
            ParticipantCorrection(
                subject=participant.subject,
                n_runs=len(measured),
                n_uncorrected=len(affected),
                affected_runs=_compact_runs(affected) if affected else "",
                worst_residual_uv=max(residuals) if residuals else None,
            )
        )
    return sorted(rows, key=lambda row: (-row.n_uncorrected, row.subject))


def uncorrected_audit(cohort: Cohort) -> pd.DataFrame:
    """The worklist, as the table a reader works through outside the document."""
    return pd.DataFrame(
        [
            {
                "subject": row.subject,
                "run": row.run,
                "pulse_marker_count": row.marker_count,
                "beat_source": row.beat_source,
                "bcg_residual_uv": row.residual_uv,
            }
            for row in uncorrected_runs(cohort)
        ]
    )


def _residual_decimal(value: float | None) -> str | None:
    """Two places, and a blank rather than a number where nothing was measured."""
    return None if value is None else f"{value:.2f}"


def correction_html(cohort: Cohort) -> str:
    """The re-export worklist and the per-participant view of it."""
    listed = uncorrected_runs(cohort)
    per_participant = participant_correction_rows(cohort)
    if not per_participant:
        return ""

    total_runs = sum(row.n_runs for row in per_participant)
    affected_participants = sum(1 for row in per_participant if row.n_uncorrected)
    if not listed:
        return (
            "<h4>Upstream pulse correction</h4>"
            f"<p>Every one of the {total_runs} run(s) carries an Analyzer R-marker train, "
            "so the upstream pulse correction had a beat train to key on throughout.</p>"
        )

    document = (
        "<h4>Runs the upstream pulse correction never ran on</h4>"
        f"<p><strong>{len(listed)} of {total_runs} run(s), across {affected_participants} "
        f"of {len(per_participant)} participant(s), carry no Analyzer R-marker "
        "train.</strong> Analyzer marks R peaks before it subtracts a pulse template, so a "
        "run with no markers had no subtraction applied and still carries its "
        "ballistocardiogram. These are the runs to re-export.</p>"
    )
    rows = [
        [row.subject, row.run, row.beat_source or MISSING, _residual_decimal(row.residual_uv)]
        for row in listed
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Run", align=Align.TEXT, code=True),
        Column("Beats measured from", align=Align.TEXT, code=True),
        Column("Residual (µV)"),
    )
    document += grid_table(columns, rows)
    document += (
        "<p>Ordered by how much beat-locked artifact each run still carries, so the worst "
        "come first. The residual is measured before the ICA exclusions, because the "
        "question is what the upstream correction left rather than what the decomposition "
        "cleaned up after it. A blank residual is a run where neither the markers nor the "
        "ECG channel yielded a beat train, so the artifact could not be measured &mdash; "
        "not a run where there is none.</p>"
        "<p>No threshold is applied. Marker absence is definitional and is stated as such; "
        "how large a residual has to be before it matters depends on what is measured "
        "downstream, which is the reader's call rather than this table's.</p>"
    )

    summary_rows = [
        [
            row.subject,
            row.n_uncorrected,
            row.n_runs,
            row.affected_runs or MISSING,
            _residual_decimal(row.worst_residual_uv),
        ]
        for row in per_participant
    ]
    summary_columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Uncorrected"),
        Column("Runs"),
        Column("Which", align=Align.TEXT, code=True),
        Column("Worst residual (µV)"),
    )
    document += "<h4>By participant</h4>" + grid_table(summary_columns, summary_rows)
    document += (
        "<p>The <em>Which</em> column separates two faults that a count cannot. A "
        "consecutive block such as <code>2-6</code> is a lead that came off partway through "
        "the session and stayed off; scattered runs such as <code>2, 3, 5</code> are "
        "intermittent contact. The first is fixed once, between sessions; the second "
        "recurs.</p>"
    )
    return document


__all__ = [
    "ANALYZER_SECTION",
    "ANALYZER_TAG",
    "ANALYZER_TITLE",
    "PLAUSIBLE_BPM",
    "AnalyzerCohort",
    "ParticipantCorrection",
    "UncorrectedRun",
    "add_analyzer_section",
    "analyzer_audit",
    "analyzer_cohort",
    "analyzer_table",
    "correction_html",
    "implausible_note",
    "participant_correction_rows",
    "uncorrected_audit",
    "uncorrected_runs",
]
