"""Residual scanner gradient across the cohort.

Three questions, and they are not the same question.

*Is the comb still there.* Gradient switching repeats once per volume, so whatever
survives upstream correction appears as a comb of narrow lines at multiples of the volume
rate. The cohort view is the excess of each line over the background beside it, before and
after the ICA exclusions, pooled participant-first.

*How much did cleaning remove.* Before and after are two measurements of one participant,
so the comparison is paired. Drawn as independent distributions it would carry the
between-participant spread, which is the larger of the two and is not what the comparison
is about.

*Was the correction driven by good timing.* Marker jitter smears the comb across
neighbouring bins, which lowers every measured excess without the residual having changed.
A participant whose volume markers are irregular therefore looks *better* on the two panels
above, so the timing panel is what stops the section from reading backwards.

The axis is chosen rather than assumed. Harmonics sit at multiples of the volume rate, so
a cohort scanned at one repetition time can be drawn against frequency, and a cohort that
mixes repetition times cannot: one participant's third harmonic and another's fourth would
land in the same bin. Where they differ, the axis becomes the harmonic index, which is the
thing the participants actually share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    BandRegime,
    CohortCurve,
    Contribution,
    align_grids,
    paired_differences,
    pool_participants_curve,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import SubjectSidecar
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    apply_report_style,
    report_image_format,
    separated_labels,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

GRADIENT_SECTION = "Residual scanner gradient"
GRADIENT_TITLE = "Gradient residual across the cohort"
GRADIENT_TAG = "cohort-gradient"

#: Repetition times within this of each other are treated as one rate, so that ordinary
#: measurement scatter in the marker train does not force a cohort onto the index axis.
REPETITION_TIME_TOLERANCE_S = 1e-3

#: Excess of a comb line over the background beside it, below which the line is
#: indistinguishable from the surrounding spectrum. Zero by construction, drawn as a guide
#: rather than as a threshold: it is where "no measurable comb" sits, not a pass mark.
NO_EXCESS_DB = 0.0

#: Participants above which individual traces stop being labelled on the figure.
#:
#: Below it the traces are the panel's content and an unlabelled one is unusable: a reader
#: who can see that one correction failed and cannot see whose has learned nothing they
#: can act on. Above it the labels collide with each other, the cohort median carries the
#: panel, and the individual traces are there to show spread rather than identity.
MAX_LABELLED_PARTICIPANTS = 8


@dataclass(frozen=True)
class CohortComb:
    """The comb either side of the exclusions, on an axis the participants share."""

    before: CohortCurve
    after: CohortCurve
    harmonic_index: np.ndarray
    #: Harmonic frequencies, when every contributing participant shares a repetition
    #: time. ``None`` when they do not, in which case the index is the only honest axis.
    harmonic_hz: np.ndarray | None
    repetition_times_s: tuple[float, ...]

    @property
    def on_frequency_axis(self) -> bool:
        return self.harmonic_hz is not None


def participant_comb(participant: SubjectSidecar) -> pd.DataFrame | None:
    """One participant's comb, pooled across its runs.

    An unweighted median across runs: each run is an independent estimate of the same
    comb, and a longer run does not measure a participant's gradient residual more
    correctly, only more precisely.
    """
    curves = participant.comb_curves
    if curves.empty:
        return None
    numeric = curves.copy()
    for column in (
        "harmonic_index",
        "harmonic_hz",
        "before_excess_db_median",
        "after_excess_db_median",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    # A harmonic inside the notch stopband measures the filter rather than the residual,
    # at something like -25 dB. Left in, it is the deepest excursion in the figure, sets
    # the whole vertical scale, and reads as the correction having worked spectacularly at
    # exactly the frequency where it was never tested. The subject panel already withholds
    # these; the cohort has to withhold the same ones.
    if "notched" in numeric.columns:
        keep = ~numeric["notched"].astype(str).str.lower().isin({"true", "1", "1.0"})
        numeric = numeric[keep]
    if numeric.empty:
        return None
    pooled = numeric.groupby("harmonic_index", as_index=False).median(numeric_only=True)
    return pooled.sort_values("harmonic_index").reset_index(drop=True)


def _repetition_times(cohort: Cohort) -> tuple[float, ...]:
    values: list[float] = []
    for participant in cohort.participants:
        if "repetition_time_s" not in participant.runs.columns:
            continue
        rates = pd.to_numeric(participant.runs["repetition_time_s"], errors="coerce").dropna()
        if not rates.empty:
            values.append(float(rates.median()))
    return tuple(values)


def cohort_comb(cohort: Cohort, *, gates: BandGates = DEFAULT_GATES) -> CohortComb | None:
    """Pool the comb across participants, choosing the axis they can share.

    Returns ``None`` when no participant resolved a comb, which is an ordinary outcome:
    the measurement declines where the frequency resolution cannot separate a line from
    its background, and that is a property of the volume rate and the run length.
    """
    combs = {
        participant.subject: pooled
        for participant in cohort.participants
        if (pooled := participant_comb(participant)) is not None
    }
    if not combs:
        return None

    alignment = align_grids(
        {subject: frame["harmonic_index"].to_numpy(dtype=float) for subject, frame in combs.items()}
    )
    runs_by_subject = {
        participant.subject: participant.n_runs for participant in cohort.participants
    }

    def _contributions(column: str) -> list[Contribution]:
        return [
            Contribution(
                subject=subject,
                value=frame.loc[alignment.masks[subject], column].to_numpy(dtype=float),
                n_runs=runs_by_subject.get(subject, 1),
            )
            for subject, frame in combs.items()
        ]

    rates = _repetition_times(cohort)
    shared_rate = bool(rates) and float(np.ptp(rates)) <= REPETITION_TIME_TOLERANCE_S
    harmonic_hz = None
    if shared_rate:
        reference = combs[sorted(combs)[0]]
        harmonic_hz = reference.loc[
            alignment.masks[sorted(combs)[0]], "harmonic_hz"
        ].to_numpy(dtype=float)

    return CohortComb(
        before=pool_participants_curve(
            _contributions("before_excess_db_median"), grid=alignment.grid, gates=gates
        ),
        after=pool_participants_curve(
            _contributions("after_excess_db_median"), grid=alignment.grid, gates=gates
        ),
        harmonic_index=alignment.grid,
        harmonic_hz=harmonic_hz,
        repetition_times_s=rates,
    )


def comb_attenuation(comb: CohortComb) -> dict[str, float]:
    """Median excess removed per participant, paired within participant."""
    before = {
        subject: float(np.median(values)) for subject, values in comb.before.per_subject.items()
    }
    after = {
        subject: float(np.median(values)) for subject, values in comb.after.per_subject.items()
    }
    # Negated: the difference is after minus before, and what a reader wants named is how
    # much was taken out.
    return {subject: -value for subject, value in paired_differences(before, after).items()}


def comb_audit(comb: CohortComb) -> pd.DataFrame:
    """The plotted values, so a figure can be checked without being re-derived."""
    rows: list[dict[str, object]] = []
    for index, harmonic in enumerate(comb.harmonic_index):
        row: dict[str, object] = {
            "harmonic_index": int(harmonic),
            "harmonic_hz": (
                None if comb.harmonic_hz is None else float(comb.harmonic_hz[index])
            ),
            "n_subjects": comb.after.denominator.n_subjects,
        }
        for stage, curve in (("before", comb.before), ("after", comb.after)):
            row[f"{stage}_median_db"] = (
                None if curve.median is None else float(curve.median[index])
            )
            for subject, values in curve.per_subject.items():
                row[f"{stage}_{subject}_db"] = float(values[index])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cohort_comb(comb: CohortComb) -> plt.Figure:
    """Draw the cohort comb, with every participant visible beneath the summary."""
    apply_report_style()
    figure, axis = plt.subplots(figsize=(9.0, 4.0), layout="constrained")
    x = comb.harmonic_hz if comb.on_frequency_axis else comb.harmonic_index
    n_participants = comb.after.denominator.n_subjects
    # Faint enough that forty traces read as a band rather than a scribble, opaque enough
    # that two read as two recordings.
    trace_alpha = float(np.clip(3.0 / max(n_participants, 1), 0.08, 0.45))
    label_traces = n_participants <= MAX_LABELLED_PARTICIPANTS

    endpoints: list[tuple[float, str]] = []

    for stage, curve, colour in (
        ("Before ICA", comb.before, BEFORE_COLOR),
        ("After ICA", comb.after, AFTER_COLOR),
    ):
        for subject, values in curve.per_subject.items():
            axis.plot(x, values, color=colour, alpha=trace_alpha, linewidth=0.8)
            if label_traces and curve is comb.after:
                # Named at the trace rather than in a legend: colour already carries the
                # stage, so a second colour scale for identity would collide with it, and
                # a participant whose correction failed is only actionable once named.
                #
                # Labelled on one stage only, and nudged apart afterwards. Labelling both
                # printed each identifier twice at the same right edge, and labelling
                # without separation stacked them wherever two participants' combs landed
                # together -- which, after a correction that worked, is everywhere.
                endpoints.append((float(values[-1]), str(subject)))
        if curve.median is not None:
            axis.plot(
                x, curve.median, color=colour, linewidth=2.0, label=f"{stage} (cohort median)"
            )
        else:
            # No summary is drawn below the gate, so the legend names the participants
            # rather than implying a cohort curve that was withheld.
            axis.plot([], [], color=colour, linewidth=2.0, label=f"{stage} (per participant)")
        if curve.quartiles is not None:
            axis.fill_between(x, curve.quartiles[0], curve.quartiles[1], color=colour, alpha=0.2)

    axis.axhline(NO_EXCESS_DB, color=GUIDE_COLOR, linewidth=1.0, linestyle="--")
    if endpoints:
        low, high = axis.get_ylim()
        for value, subject in separated_labels(endpoints, minimum_gap=(high - low) * 0.045):
            axis.annotate(
                subject,
                xy=(x[-1], value),
                xytext=(4, 0),
                textcoords="offset points",
                color=AFTER_COLOR,
                fontsize="x-small",
                va="center",
                annotation_clip=False,
            )
    axis.set_xlabel(
        "Harmonic frequency (Hz)"
        if comb.on_frequency_axis
        else "Harmonic index (multiple of each participant's own volume rate)"
    )
    axis.set_ylabel("Excess over local background (dB)")
    axis.set_title(
        f"Gradient comb, {comb.after.denominator.n_subjects} participant(s), "
        f"{comb.after.denominator.n_runs} run(s)"
    )
    axis.legend(loc="upper right", frameon=False)
    return figure


def _measured(runs: pd.DataFrame, column: str) -> pd.Series:
    """The finite values of one run column, or nothing where it is inapplicable."""
    if column not in runs.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(runs[column], errors="coerce")
    return values[np.isfinite(values)]


def _locked_amplitude(runs: pd.DataFrame, stage: str) -> tuple[str | None, float | None]:
    """Participant-median locked amplitude, preserving an unresolved estimate."""
    excess = _measured(runs, f"volume_locked_excess_power_{stage}_uv2")
    if excess.empty:
        return None, None
    power = float(excess.median())
    if power <= 0.0:
        return "unresolved", None
    amplitude = float(np.sqrt(power))
    return f"{amplitude:.2f}", amplitude


def _timing_rows(cohort: Cohort) -> list[Sequence[object]]:
    rows: list[Sequence[object]] = []
    for participant in cohort.participants:
        runs = participant.runs
        if "repetition_time_s" not in runs.columns:
            continue
        rate = _measured(runs, "repetition_time_s")
        jitter = _measured(runs, "volume_jitter_s")
        volumes = _measured(runs, "n_volumes")
        before_text, before_amplitude = _locked_amplitude(runs, "before")
        after_text, after_amplitude = _locked_amplitude(runs, "after")
        observed_after = _measured(runs, "volume_locked_rms_after_uv")
        floor_after = _measured(runs, "volume_locked_floor_after_uv")
        # Paired within the participant: the removed column is one recording measured twice,
        # not a difference between two cohort summaries. Withheld unless both sides are
        # present, because a difference against a missing side is not a difference.
        removed = (
            f"{before_amplitude - after_amplitude:.2f}"
            if before_amplitude is not None and after_amplitude is not None
            else None
        )
        rows.append(
            [
                participant.subject,
                None if rate.empty else f"{rate.median():.3f}",
                None if jitter.empty else f"{jitter.max() * 1e3:.1f}",
                None if volumes.empty else int(volumes.sum()),
                before_text,
                after_text,
                removed,
                None if observed_after.empty else f"{observed_after.median():.2f}",
                None if floor_after.empty else f"{floor_after.median():.2f}",
            ]
        )
    return rows


def timing_table(cohort: Cohort) -> str:
    """Volume timing and the floor-corrected residual, per participant."""
    rows = _timing_rows(cohort)
    if not rows:
        return ""
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("TR (s)"),
        Column("Worst jitter (ms)"),
        Column("Volumes"),
        Column("Before ICA", group="Volume-locked residual (µV)"),
        Column("After ICA", group="Volume-locked residual (µV)"),
        Column("Removed", group="Volume-locked residual (µV)"),
        Column("Observed after (µV RMS)"),
        Column("Noise floor after (µV RMS)"),
    )
    return (
        grid_table(columns, rows)
        + "<p>The residual is derived from the signed difference between observed locked "
        "power and its odd-even averaging-floor estimate. A non-positive difference is "
        "reported as <em>unresolved</em>, not zero. Observed RMS and the estimated floor "
        "remain beside it so the censored measurement is auditable. The odd-even floor "
        "uses the same number of epochs as the observed average, so their comparison "
        "does not fall merely because a participant was scanned for longer.</p>"
        "<p>Reported either side of the exclusions, and paired within the participant. The "
        "after column alone cannot separate a recording whose correction removed a locked "
        "residual from one that never had a measurable residual to remove, and those are "
        "different recordings: only the second tells a reader nothing needs looking at.</p>"
        "<p>Jitter is reported because it works against the panel above. Irregular volume "
        "markers smear the comb across neighbouring frequency bins, which lowers every "
        "measured excess without the residual itself having changed &mdash; so a "
        "participant with poor timing can appear to have the cleanest comb in the "
        "cohort.</p>"
    )


def attenuation_table(comb: CohortComb) -> str:
    """What the exclusions removed from each participant's comb, paired within it."""
    attenuation = comb_attenuation(comb)
    rows = [
        [
            subject,
            f"{float(np.median(comb.before.per_subject[subject])):.1f}",
            f"{float(np.median(comb.after.per_subject[subject])):.1f}",
            f"{removed:.1f}",
        ]
        for subject, removed in attenuation.items()
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Before ICA", group="Median comb excess (dB)"),
        Column("After ICA", group="Median comb excess (dB)"),
        Column("Removed (dB)"),
    )
    return (
        grid_table(columns, rows)
        + "<p>Each row is one participant measured twice, so the removed column is a "
        "within-participant difference rather than a difference of two cohort summaries. "
        "Reading it as the latter would carry the spread between participants, which is "
        "the larger of the two and not what this comparison is about.</p>"
    )


def add_gradient_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> CohortComb | None:
    """Add the cohort gradient section, or nothing when no participant was in a scanner."""
    comb = cohort_comb(cohort, gates=gates)
    timing = timing_table(cohort)
    if comb is None and not timing:
        return None

    parts = [
        "<p>Gradient switching repeats once per scanner volume, so whatever survives "
        "upstream correction appears as a comb of narrow lines at multiples of the volume "
        "rate. Each line is compared with the background measured beside it, within each "
        "channel, so 0 dB means the line is indistinguishable from the surrounding "
        "spectrum. Runs are pooled within a participant first, and every participant then "
        "carries equal weight.</p>"
    ]
    if comb is not None:
        if not comb.on_frequency_axis:
            parts.append(
                "<p>Participants were scanned at different repetition times, so their "
                "harmonics fall at different frequencies. The axis is therefore the "
                "harmonic index &mdash; each participant's own volume rate and its "
                "multiples &mdash; because pooling by frequency would put one "
                "participant's third harmonic in the same bin as another's fourth.</p>"
            )
        if comb.after.regime is BandRegime.INDIVIDUALS:
            parts.append(
                "<p>Each participant is drawn individually and no cohort median is shown: "
                "the participant count cannot support one.</p>"
            )
        report.add_figure(
            fig=plot_cohort_comb(comb),
            title="Gradient comb before and after ICA",
            section=GRADIENT_SECTION,
            tags=(GRADIENT_TAG,),
            image_format=report_image_format(),
            replace=True,
        )
        plt.close("all")
        parts.append(attenuation_table(comb))
    if timing:
        parts.append("<h4>Volume timing and locked residual</h4>")
        parts.append(timing)

    report.add_html(
        html="".join(parts),
        title=GRADIENT_TITLE,
        section=GRADIENT_SECTION,
        tags=(GRADIENT_TAG,),
        replace=True,
    )
    return comb


__all__ = [
    "GRADIENT_SECTION",
    "GRADIENT_TAG",
    "GRADIENT_TITLE",
    "NO_EXCESS_DB",
    "REPETITION_TIME_TOLERANCE_S",
    "CohortComb",
    "add_gradient_section",
    "attenuation_table",
    "comb_attenuation",
    "comb_audit",
    "cohort_comb",
    "participant_comb",
    "plot_cohort_comb",
    "timing_table",
]
