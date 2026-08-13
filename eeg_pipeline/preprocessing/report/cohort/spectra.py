"""Sensor spectra across the cohort, and what cleaning did to the background beneath them.

Two panels answering two different questions.

The first is the spectrum itself, pooled participant-first, before and after the ICA
exclusions. Every participant stays drawn beneath the summary, because the cohort median
is not any participant's spectrum and a recording that sits far from it is the thing a
reader is looking for.

The second is the aperiodic background. Broadband artifact raises a spectrum roughly
uniformly and flattens its slope; removing too many components takes the background down
with it. Reporting the exponent and offset either side of cleaning turns "86% of variance
removed" into two numbers that say whether what left was artifact or signal, and a cohort
whose exponent shifted systematically has lost something broadband that no other panel
here would reveal. It is drawn paired, one line per participant, because before and after
are two measurements of one recording.

Both panels use a logarithmic frequency axis, matching the subject report: on a linear
axis a 1-100 Hz range gives delta and theta a few pixels while the high-frequency tail,
which is mostly noise floor, takes most of the width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
from eeg_pipeline.preprocessing.report.cohort.record import AFTER, BEFORE
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    AcquisitionContext,
    SubjectSidecar,
)
from eeg_pipeline.preprocessing.report.filtering import (
    NOTCH_EXCLUSION_HALF_WIDTH_HZ,
    notch_windows,
)
from eeg_pipeline.preprocessing.report.spectra import POWER_UNIT_LABEL
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    MARK_COLOR,
    apply_report_style,
    report_image_format,
    separated_labels,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

SPECTRA_SECTION = "Sensor spectra"
SPECTRA_TITLE = "Cohort spectra before and after ICA"
SPECTRA_TAG = "cohort-spectra"

#: Participants above which the spectrum's traces stop being labelled. Same reasoning as
#: the gradient panel: below it a trace nobody can name is unusable, above it the labels
#: collide and the cohort median carries the figure.
MAX_LABELLED_PARTICIPANTS = 8

#: Participants above which the paired aperiodic panel stops labelling its endpoints.
#:
#: Higher than the spectrum allows, because the geometry differs: endpoints sit in one
#: column and are nudged apart when they crowd, so twenty fit where twenty overlapping
#: spectra would not.
MAX_LABELLED_PAIRED = 20

#: Percentiles bounding the power axis, so that one participant's notch floor -- tens of
#: decibels below anything else -- cannot squeeze the spectra into the top of the panel.
_AXIS_PERCENTILES = (0.5, 99.5)

#: Headroom left above and below the bounded range, in decibels.
_AXIS_MARGIN_DB = 3.0

#: Frequencies labelled on the logarithmic axis, where they fall inside the range.
#:
#: Matplotlib's default marks decades only, so a reader cannot locate a band edge without
#: counting minor ticks. These are the frequencies an EEG spectrum is actually read
#: against.
_FREQUENCY_TICKS = (1, 2, 5, 10, 20, 50, 100, 200)

#: Smallest span the paired axis is allowed to cover.
#:
#: A paired plot autoscales to the change it contains, which is what makes a real shift
#: visible -- and what makes a negligible one look like a cliff. A floor on the span keeps
#: the slope proportional to the change: below it the lines flatten, which is the honest
#: rendering of "these two measurements agree".
MINIMUM_EXPONENT_SPAN = 0.3


@dataclass(frozen=True)
class CohortSpectra:
    """The cohort spectrum either side of the exclusions, on a shared frequency grid."""

    frequencies: np.ndarray
    before: CohortCurve
    after: CohortCurve
    #: Participants whose own filter narrowed the axis, named on the panel so a reader
    #: can see the range stops where it does because of a recording, not a method.
    limiting_subjects: tuple[str, ...]


def participant_spectrum(participant: SubjectSidecar, *, stage: str) -> pd.Series | None:
    """One participant's spectrum for one stage, pooled across its runs.

    An unweighted median across runs: each run is an independent estimate of the same
    spectrum, and a longer run estimates it more precisely rather than more correctly.
    """
    curves = participant.spectrum_curves
    if curves.empty:
        return None
    selected = curves[curves["stage"].astype(str) == stage]
    if selected.empty:
        return None
    frame = selected.copy()
    frame["freq_hz"] = pd.to_numeric(frame["freq_hz"], errors="coerce")
    frame["median_db"] = pd.to_numeric(frame["median_db"], errors="coerce")
    return frame.groupby("freq_hz")["median_db"].median().sort_index()


def cohort_spectra(cohort: Cohort, *, gates: BandGates = DEFAULT_GATES) -> CohortSpectra | None:
    """Pool both stages across participants on the frequency range they all support.

    The range is the intersection, never an interpolation: resampling a spectrum onto a
    foreign grid smears exactly the narrow features -- a line-noise tooth, a gradient
    harmonic, an alpha peak -- that this figure exists to show, and does so invisibly.
    """
    per_stage = {
        stage: {
            participant.subject: series
            for participant in cohort.participants
            if (series := participant_spectrum(participant, stage=stage)) is not None
            # A curve with a gap in it cannot be pooled: the cohort median is missing
            # wherever the gap sits, and the denominator beside the figure would still
            # count the participant that made the hole. Dropped here, where the panel's
            # own denominator shrinks with it and the loss is visible, rather than pooled
            # into a curve that is quietly wrong over part of its range.
            and bool(np.all(np.isfinite(series.to_numpy(dtype=float))))
        }
        for stage in (BEFORE, AFTER)
    }
    if not per_stage[AFTER]:
        return None

    alignment = align_grids(
        {subject: series.index.to_numpy(dtype=float) for subject, series in per_stage[AFTER].items()}
    )
    runs_by_subject = {
        participant.subject: participant.n_runs for participant in cohort.participants
    }

    def _curve(stage: str) -> CohortCurve:
        contributions = [
            Contribution(
                subject=subject,
                value=series.to_numpy(dtype=float)[alignment.masks[subject]],
                n_runs=runs_by_subject.get(subject, 1),
            )
            for subject, series in per_stage[stage].items()
            if subject in alignment.masks
        ]
        return pool_participants_curve(contributions, grid=alignment.grid, gates=gates)

    return CohortSpectra(
        frequencies=alignment.grid,
        before=_curve(BEFORE),
        after=_curve(AFTER),
        limiting_subjects=alignment.limiting_subjects,
    )


def _agreed_setting(cohort: Cohort, key: str) -> float | None:
    """A setting's value when every participant recorded the same one, else nothing.

    A frequency marked on a cohort figure has to be the frequency every contributing
    recording was filtered at. Where they disagree the mark would be true of some traces
    and false of others, which is worse than no mark.
    """
    values = {
        participant.settings.get(key)
        for participant in cohort.participants
        if participant.settings.get(key) is not None
    }
    if len(values) != 1:
        return None
    return float(next(iter(values)))


def _power_limits(
    spectra: CohortSpectra,
    *,
    line_frequency: float | None,
    notch_half_width_hz: float = NOTCH_EXCLUSION_HALF_WIDTH_HZ,
) -> tuple[float, float]:
    """Bound the power axis by the spectra rather than by the notch they contain.

    A notch drives its band to the numerical floor, tens of decibels below anything else.
    Left in the limits it stretches the axis and leaves the spectra the panel exists to
    show squeezed into the top third. The trace is still drawn through the notch and
    simply leaves the axis, which reads as a filtered band rather than as missing data.
    """
    keep = np.ones_like(spectra.frequencies, dtype=bool)
    for low, high in notch_windows(
        line_frequency,
        fmax=float(spectra.frequencies[-1]),
        half_width=notch_half_width_hz,
    ):
        keep &= (spectra.frequencies < low) | (spectra.frequencies > high)
    if not keep.any():
        keep = np.ones_like(spectra.frequencies, dtype=bool)
    stacked = np.concatenate(
        [values[keep] for curve in (spectra.before, spectra.after) for values in curve.per_subject.values()]
    )
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return (-1.0, 1.0)
    low, high = np.percentile(finite, _AXIS_PERCENTILES)
    return (float(low) - _AXIS_MARGIN_DB, float(high) + _AXIS_MARGIN_DB)


def _informative_range(spectra: CohortSpectra, *, floor_db: float) -> tuple[float, float]:
    """Stop the frequency axis where every participant has left the plot.

    The anti-alias filter takes the spectrum down by forty decibels or more, and the grid
    runs on past it to the Nyquist. Drawn to the full grid, a third of the panel's width is
    rolloff: no participant's trace is on the axis there, no feature can be read from it,
    and the frequencies that matter are compressed into what is left.

    The bound follows from the vertical limits rather than from a guess at where the filter
    sits, so a dataset filtered somewhere else gets the right answer without being told.
    """
    frequencies = np.asarray(spectra.frequencies, dtype=float)
    on_axis = np.zeros(frequencies.size, dtype=bool)
    for curve in (spectra.before, spectra.after):
        for values in curve.per_subject.values():
            array = np.asarray(values, dtype=float)
            on_axis |= np.isfinite(array) & (array >= floor_db)
    if not on_axis.any():
        return float(frequencies[0]), float(frequencies[-1])
    highest = float(frequencies[np.flatnonzero(on_axis)[-1]])
    # A little headroom, so the last visible feature is not pressed against the spine.
    return float(frequencies[0]), min(float(frequencies[-1]), highest * 1.08)


def _mark_frequencies(
    axis: plt.Axes,
    *,
    line_frequency: float | None,
    marked_frequencies: Sequence[float],
) -> None:
    """Draw the few frequencies a reader needs to locate, and no more.

    Every mark is ink over the data. The line-noise fundamental is drawn because a reader
    has to know which peak is the mains; a handful of gradient harmonics are drawn so the
    comb can be placed relative to the spectrum, with the dedicated comb panel measuring
    every harmonic properly.
    """
    if line_frequency:
        axis.axvline(line_frequency, color=MARK_COLOR, linewidth=0.8, alpha=0.45)
        axis.annotate(
            f"{line_frequency:g} Hz line",
            xy=(line_frequency, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize="x-small",
            color=MARK_COLOR,
        )
    for frequency in marked_frequencies:
        axis.axvline(frequency, color=GUIDE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)


def _label_frequency_ticks(axis: plt.Axes, frequencies: np.ndarray) -> None:
    """Label the logarithmic axis at frequencies a reader recognises.

    The default marks decades, which leaves the alpha band somewhere between the first
    two labels and gives no way to read a band edge off the figure.
    """
    low, high = float(frequencies[0]), float(frequencies[-1])
    ticks = [value for value in _FREQUENCY_TICKS if low <= value <= high]
    if not ticks:
        return
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(ticker.ScalarFormatter())
    axis.xaxis.set_minor_formatter(ticker.NullFormatter())


def _widen_to_minimum_span(axis: plt.Axes, minimum_span: float) -> None:
    """Stop a paired axis from magnifying a change too small to matter."""
    low, high = axis.get_ylim()
    span = high - low
    if span >= minimum_span:
        return
    centre = 0.5 * (low + high)
    axis.set_ylim(centre - 0.5 * minimum_span, centre + 0.5 * minimum_span)


def plot_cohort_spectra(
    spectra: CohortSpectra,
    *,
    line_frequency: float | None = None,
    marked_frequencies: Sequence[float] = (),
    notch_half_width_hz: float = NOTCH_EXCLUSION_HALF_WIDTH_HZ,
) -> plt.Figure:
    """Draw the cohort spectra with every participant visible beneath the summary."""
    apply_report_style()
    figure, axis = plt.subplots(figsize=(9.0, 4.6), layout="constrained")
    n_participants = spectra.after.denominator.n_subjects
    trace_alpha = float(np.clip(3.0 / max(n_participants, 1), 0.08, 0.45))
    label_traces = n_participants <= MAX_LABELLED_PARTICIPANTS
    limits = _power_limits(
        spectra,
        line_frequency=line_frequency,
        notch_half_width_hz=notch_half_width_hz,
    )
    view = _informative_range(spectra, floor_db=limits[0])
    # The label belongs where the trace leaves the panel, not where the grid ends. Anchored
    # to the last grid point it lands beyond the right spine, out where every participant
    # has converged into the filter rolloff and their labels print on top of each other.
    anchor = int(np.searchsorted(spectra.frequencies, view[1], side="right")) - 1
    anchor = int(np.clip(anchor, 0, spectra.frequencies.size - 1))
    endpoints: list[tuple[float, str]] = []

    for stage, curve, colour in (
        ("Before ICA", spectra.before, BEFORE_COLOR),
        ("After ICA", spectra.after, AFTER_COLOR),
    ):
        for subject, values in curve.per_subject.items():
            axis.plot(spectra.frequencies, values, color=colour, alpha=trace_alpha, linewidth=0.8)
            if label_traces and stage == "After ICA":
                # Labelled on one stage only: two labels per participant at the same right
                # edge would overlap, and the cleaned trace is the one a reader tracks.
                endpoints.append((float(values[anchor]), str(subject)))

        if curve.median is not None:
            axis.plot(
                spectra.frequencies,
                curve.median,
                color=colour,
                linewidth=2.0,
                label=f"{stage} (cohort median)",
            )
        else:
            axis.plot([], [], color=colour, linewidth=2.0, label=f"{stage} (per participant)")
        if curve.quartiles is not None:
            axis.fill_between(
                spectra.frequencies,
                curve.quartiles[0],
                curve.quartiles[1],
                color=colour,
                alpha=0.18,
                linewidth=0,
            )

    if endpoints:
        span = limits[1] - limits[0]
        for value, subject in separated_labels(endpoints, minimum_gap=span * 0.045):
            axis.annotate(
                subject,
                xy=(float(spectra.frequencies[anchor]), value),
                xytext=(4, 0),
                textcoords="offset points",
                color=AFTER_COLOR,
                fontsize="x-small",
                va="center",
                annotation_clip=False,
            )

    _mark_frequencies(
        axis, line_frequency=line_frequency, marked_frequencies=marked_frequencies
    )
    axis.set_xscale("log")
    axis.set_xlim(*view)
    _label_frequency_ticks(axis, spectra.frequencies)
    axis.set_ylim(*limits)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(f"PSD ({POWER_UNIT_LABEL})")
    limiting = (
        f" · range set by {', '.join(spectra.limiting_subjects)}"
        if spectra.limiting_subjects
        else ""
    )
    axis.set_title(
        f"Across-channel median spectrum, {n_participants} participant(s), "
        f"{spectra.after.denominator.n_runs} run(s){limiting}"
    )
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    return figure


def spectra_audit(spectra: CohortSpectra) -> pd.DataFrame:
    """The plotted values, so the figure can be checked without being re-derived."""
    frame = pd.DataFrame({"freq_hz": spectra.frequencies})
    for stage, curve in ((BEFORE, spectra.before), (AFTER, spectra.after)):
        frame[f"{stage}_median_db"] = (
            np.full(spectra.frequencies.size, np.nan) if curve.median is None else curve.median
        )
        for subject, values in curve.per_subject.items():
            frame[f"{stage}_{subject}_db"] = values
    frame["n_subjects"] = spectra.after.denominator.n_subjects
    return frame


#: Bands the removal is compared between, both inside the aperiodic fit range.
#:
#: The exponent is a slope, so it can only move if removal was uneven across frequency.
#: Measuring how uneven turns "the exponent moved" into a statement about what moved it,
#: which is the difference between a correction working and signal being lost.
REMOVAL_LOW_BAND_HZ = (2.0, 6.0)
REMOVAL_HIGH_BAND_HZ = (30.0, 45.0)


def _band_removal(participant: SubjectSidecar, band: tuple[float, float]) -> float:
    """Median power the exclusions took out of one band, in decibels.

    Negative where power was removed. Read from the recorded spectra rather than from the
    fitted line, so it is a measurement of the spectrum itself and not a property of the
    model fitted to it.
    """
    curves = participant.spectrum_curves
    if curves.empty:
        return float("nan")
    frame = curves.copy()
    frame["freq_hz"] = pd.to_numeric(frame["freq_hz"], errors="coerce")
    frame["median_db"] = pd.to_numeric(frame["median_db"], errors="coerce")
    inside = frame[(frame["freq_hz"] >= band[0]) & (frame["freq_hz"] < band[1])]
    if inside.empty:
        return float("nan")
    pooled = inside.groupby(["stage", "freq_hz"])["median_db"].median().unstack("stage")
    if BEFORE not in pooled.columns or AFTER not in pooled.columns:
        return float("nan")
    difference = (pooled[AFTER] - pooled[BEFORE]).dropna()
    return float(difference.median()) if not difference.empty else float("nan")


def aperiodic_frame(cohort: Cohort) -> pd.DataFrame:
    """Exponent and offset per participant per stage, pooled across runs.

    Carries the acquisition context and the band-resolved removal alongside, because the
    exponent shift cannot be read without them: inside a scanner the pre-ICA spectrum is
    dominated at low frequency by the ballistocardiogram, so removing it must flatten the
    slope. The same shift means opposite things in and out of a bore.
    """
    rows: list[dict[str, object]] = []
    for participant in cohort.participants:
        runs = participant.runs
        row: dict[str, object] = {
            "subject": participant.subject,
            "context": participant.context,
        }
        found = False
        for stage in (BEFORE, AFTER):
            for name in ("exponent", "offset_db", "r_squared"):
                column = f"aperiodic_{name}_{stage}"
                if column not in runs.columns:
                    row[f"{name}_{stage}"] = np.nan
                    continue
                values = pd.to_numeric(runs[column], errors="coerce").dropna()
                row[f"{name}_{stage}"] = np.nan if values.empty else float(values.median())
                found = found or not values.empty
        row["removed_low_db"] = _band_removal(participant, REMOVAL_LOW_BAND_HZ)
        row["removed_high_db"] = _band_removal(participant, REMOVAL_HIGH_BAND_HZ)
        row["removal_tilt_db"] = row["removed_low_db"] - row["removed_high_db"]
        if found:
            rows.append(row)
    return pd.DataFrame(rows)


def plot_aperiodic_shift(frame: pd.DataFrame) -> plt.Figure:
    """The 1/f exponent before against after, one line per participant.

    A paired display, because before and after are two measurements of one recording.
    Drawn as two independent distributions the comparison would carry the spread between
    participants, which is the larger of the two and is not what is being asked.

    The exponent only. The offset moves down whenever variance is removed, near enough by
    construction, so a panel of it restates the variance-removed column rather than adding
    to it; the exponent is the one that says whether the background changed *shape*, which
    is what distinguishes artifact from signal having been taken out. Both numbers stay in
    the table for a reader who wants them.
    """
    apply_report_style()
    figure, axis = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    label_traces = len(frame) <= MAX_LABELLED_PAIRED

    endpoints: list[tuple[float, str]] = []
    contexts: set = set()
    for _, participant in frame.iterrows():
        before = participant.get(f"exponent_{BEFORE}")
        after = participant.get(f"exponent_{AFTER}")
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        # Dashed inside a scanner, solid outside it. A downward shift means opposite things
        # in the two, so a reader must be able to tell which lines are which without
        # cross-referencing the composition table.
        context = participant.get("context")
        contexts.add(context)
        in_scanner = context is AcquisitionContext.IN_SCANNER
        axis.plot(
            [0, 1],
            [before, after],
            color=GUIDE_COLOR,
            linewidth=1.0,
            linestyle="--" if in_scanner else "-",
            zorder=1,
        )
        axis.plot([0], [before], marker="o", markersize=4.0, color=BEFORE_COLOR, zorder=2)
        axis.plot([1], [after], marker="o", markersize=4.0, color=AFTER_COLOR, zorder=2)
        endpoints.append((float(after), str(participant["subject"])))

    axis.set_xticks([0, 1], ["Before ICA", "After ICA"])
    axis.set_xlim(-0.25, 1.4)
    axis.set_ylabel("1/f exponent")
    if len(contexts) > 1:
        axis.plot([], [], color=GUIDE_COLOR, linestyle="--", label="In scanner")
        axis.plot([], [], color=GUIDE_COLOR, linestyle="-", label="Outside scanner")
        axis.legend(frameon=False, fontsize=8, loc="lower left")
    scanner_note = (
        " \u00b7 in scanner"
        if contexts == {AcquisitionContext.IN_SCANNER}
        else (" \u00b7 outside scanner" if contexts == {AcquisitionContext.OUT_OF_SCANNER} else "")
    )
    axis.set_title(
        f"Aperiodic background \u00b7 {len(endpoints)} participant(s){scanner_note}"
    )
    _widen_to_minimum_span(axis, MINIMUM_EXPONENT_SPAN)
    if label_traces and endpoints:
        low, high = axis.get_ylim()
        for value, label in separated_labels(endpoints, minimum_gap=(high - low) * 0.05):
            axis.annotate(
                label,
                xy=(1.06, value),
                fontsize="x-small",
                color=GUIDE_COLOR,
                va="center",
                annotation_clip=False,
            )
    return figure


def aperiodic_table(frame: pd.DataFrame) -> str:
    """Per-participant exponent and offset, with the within-participant shift."""
    if frame.empty:
        return ""
    exponents = {
        str(row["subject"]): float(row[f"exponent_{BEFORE}"])
        for _, row in frame.iterrows()
        if np.isfinite(row[f"exponent_{BEFORE}"]) and np.isfinite(row[f"exponent_{AFTER}"])
    }
    after_exponents = {
        str(row["subject"]): float(row[f"exponent_{AFTER}"])
        for _, row in frame.iterrows()
        if np.isfinite(row[f"exponent_{BEFORE}"]) and np.isfinite(row[f"exponent_{AFTER}"])
    }
    shifts = paired_differences(exponents, after_exponents) if exponents else {}

    rows = []
    for _, participant in frame.iterrows():
        subject = str(participant["subject"])
        rows.append(
            [
                subject,
                _decimal(participant[f"exponent_{BEFORE}"]),
                _decimal(participant[f"exponent_{AFTER}"]),
                _decimal(shifts.get(subject)),
                _decimal(participant.get("removed_low_db"), places=1),
                _decimal(participant.get("removed_high_db"), places=1),
                _decimal(participant.get("removal_tilt_db"), places=1),
                _decimal(participant[f"r_squared_{AFTER}"]),
            ]
        )
    low = f"{REMOVAL_LOW_BAND_HZ[0]:g}–{REMOVAL_LOW_BAND_HZ[1]:g} Hz"
    high = f"{REMOVAL_HIGH_BAND_HZ[0]:g}–{REMOVAL_HIGH_BAND_HZ[1]:g} Hz"
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Before", group="1/f exponent"),
        Column("After", group="1/f exponent"),
        Column("Shift", group="1/f exponent"),
        Column(low, group="Power removed (dB)"),
        Column(high, group="Power removed (dB)"),
        Column("Difference", group="Power removed (dB)"),
        Column("Fit r²"),
    )
    return grid_table(columns, rows) + (
        "<p>The exponent is a slope, so it can only move if the exclusions took power out "
        "unevenly across frequency. The two removal columns are that unevenness measured "
        "directly, from the recorded spectra rather than from the line fitted to them, and "
        "their difference is what moved the exponent. Removal of roughly equal depth in "
        "both bands leaves the slope alone whatever it does to the offset, which is why "
        "the offset is not tabled beside them: it falls whenever variance is removed, near "
        "enough by construction, and says nothing the variance-removed column has not.</p>"
    )


def _exponent_reading(frame: pd.DataFrame) -> str:
    """State what an exponent shift means for the acquisitions that produced it.

    The same shift means opposite things in and out of a scanner, and the report already
    refuses to pool across that axis for variance removed for the same reason. Inside a
    bore the pre-ICA spectrum is dominated at low frequency by the ballistocardiogram, and
    removing it *must* take more power out of the bottom of the band than the top -- which
    flattens the slope and lowers the exponent. That is the arithmetic of a correction
    working, not evidence of signal lost, and a panel that reads it the other way sends a
    reader to look for a problem their cleaning does not have.

    Outside a scanner there is no such dominant low-frequency artifact, so the alarming
    reading is the ordinary one. Both are stated; which applies is decided by the recorded
    context, never asserted about a participant.
    """
    contexts = {row["context"] for _, row in frame.iterrows() if "context" in row}
    in_scanner = AcquisitionContext.IN_SCANNER in contexts
    out_of_scanner = AcquisitionContext.OUT_OF_SCANNER in contexts

    parts = []
    if in_scanner:
        parts.append(
            "<p><strong>For the in-scanner participants</strong>, a downward shift is what "
            "removing the ballistocardiogram looks like. That artifact dominates the "
            "spectrum at low frequency, so taking it out removes more power from the "
            "bottom of the fit range than the top and flattens the slope by arithmetic. "
            "The removal columns below are how to tell the two apart: removal concentrated "
            "at low frequency is a cardiac correction doing its job, while removal of "
            "similar depth across the whole range that still moves the exponent is the "
            "case worth looking into.</p>"
        )
    if out_of_scanner:
        parts.append(
            "<p><strong>For the participants recorded outside a scanner</strong>, there is "
            "no dominant low-frequency artifact for cleaning to remove, so an exponent that "
            "shifted in one direction across participants is harder to explain as anything "
            "but broadband signal having gone with it &mdash; and nothing else in this "
            "report would show it.</p>"
        )
    if in_scanner and out_of_scanner:
        parts.append(
            "<p>This cohort spans both, so the panel above is marked by context and the two "
            "groups are not read against each other.</p>"
        )
    return "".join(parts)


def _decimal(value: object, *, places: int = 2) -> str | None:
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else f"{number:.{places}f}"


#: Gradient harmonics drawn on the sensor spectrum.
#:
#: A handful, not the comb. The comb has tens of teeth inside the plotted range and drawing
#: all of them would bury the spectrum under vertical lines; the dedicated comb panel
#: measures every harmonic properly. These are here only so a reader can see where the comb
#: sits relative to everything else.
MARKED_GRADIENT_HARMONICS = 3

#: Band the comb is searched in when no participant recorded the setting.
DEFAULT_COMB_RANGE_HZ = (15.0, 90.0)


def gradient_marks(cohort: Cohort) -> tuple[float, ...]:
    """A few gradient harmonics from the band where the comb is actually measured.

    Derived here rather than asked of the caller, because the cohort is what knows its own
    volume rates. Withheld where participants were scanned at different repetition times: a
    mark then sits on one participant's harmonic and between another's, which is worse than
    no mark because it is read as applying to every trace on the panel.

    The harmonics chosen are the lowest few *inside the comb's own search band*, not the
    first three of the series. At a 0.9 s repetition time the series starts at 1.11 Hz, so
    the first three sit at 1, 2 and 3 Hz -- below anything the comb measures, drawn straight
    across the alpha band, and marking a gradient artifact where none of the visible comb
    is. A reader takes three dotted lines under the alpha peak to mean the gradient lives
    there. The teeth actually visible on this figure run from about 25 Hz up.
    """
    rates: set[float] = set()
    for participant in cohort.participants:
        if "repetition_time_s" not in participant.runs.columns:
            continue
        measured = pd.to_numeric(
            participant.runs["repetition_time_s"], errors="coerce"
        ).dropna()
        rates.update(round(float(value), 4) for value in measured)
    if len(rates) != 1:
        return ()
    repetition_time = next(iter(rates))
    if repetition_time <= 0.0:
        return ()

    fundamental = 1.0 / repetition_time
    band = _agreed_range(cohort, "comb_frequency_range_hz") or DEFAULT_COMB_RANGE_HZ
    lowest = max(1, int(np.ceil(band[0] / fundamental)))
    orders = range(lowest, lowest + MARKED_GRADIENT_HARMONICS)
    return tuple(
        fundamental * order for order in orders if fundamental * order <= band[1]
    )


def _agreed_range(cohort: Cohort, key: str) -> tuple[float, float] | None:
    """A two-valued setting where every participant recorded the same pair."""
    found: set[tuple[float, float]] = set()
    for participant in cohort.participants:
        value = participant.settings.get(key)
        if value is None:
            continue
        try:
            low, high = (float(value[0]), float(value[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        found.add((low, high))
    return next(iter(found)) if len(found) == 1 else None


def add_spectra_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
    marked_frequencies: Sequence[float] = (),
) -> CohortSpectra | None:
    """Add the cohort spectra section, or nothing when no participant contributed one."""
    spectra = cohort_spectra(cohort, gates=gates)
    if spectra is None:
        return None
    line_frequency = _agreed_setting(cohort, "spectra_line_frequency")
    # The half-width the participants were measured at, where they agree. Where they do
    # not, the cohort has no single answer for how wide the filtered band is, so the
    # packaged width draws the axis and the homogeneity panel reports the disagreement.
    notch_half_width_hz = _agreed_setting(cohort, "notch_exclusion_half_width_hz")
    marks = tuple(marked_frequencies) or gradient_marks(cohort)

    report.add_figure(
        fig=plot_cohort_spectra(
            spectra,
            line_frequency=line_frequency,
            marked_frequencies=marks,
            notch_half_width_hz=(
                NOTCH_EXCLUSION_HALF_WIDTH_HZ
                if notch_half_width_hz is None
                else notch_half_width_hz
            ),
        ),
        title="Cohort spectrum before and after ICA",
        section=SPECTRA_SECTION,
        tags=(SPECTRA_TAG,),
        image_format=report_image_format(),
        replace=True,
    )
    plt.close("all")

    parts = [
        "<p>Each participant's spectrum is the across-channel median, pooled across its "
        "runs, and every participant stays drawn beneath the cohort summary: the median "
        "is not any participant's spectrum, and a recording sitting far from it is what a "
        "reader is looking for.</p>"
    ]
    if spectra.after.regime is BandRegime.INDIVIDUALS:
        parts.append(
            "<p>No cohort median is drawn: the participant count cannot support one.</p>"
        )
    if line_frequency is None:
        parts.append(
            "<p>No line-noise frequency is marked, because the participants were not all "
            "recorded under one. A mark true of some traces and false of others would be "
            "worse than none.</p>"
        )

    aperiodic = aperiodic_frame(cohort)
    if not aperiodic.empty:
        report.add_figure(
            fig=plot_aperiodic_shift(aperiodic),
            title="Aperiodic background either side of the exclusions",
            section=SPECTRA_SECTION,
            tags=(SPECTRA_TAG,),
            image_format=report_image_format(),
            replace=True,
        )
        plt.close("all")
        parts.append(
            "<h4>Aperiodic background</h4>"
            "<p>Broadband artifact raises a spectrum roughly uniformly and flattens its "
            "slope; removing too many components takes the background down with it. The "
            "exponent either side of cleaning is what turns a variance-removed percentage "
            "into a number that says whether what left was artifact or signal.</p>"
            + _exponent_reading(aperiodic)
            + aperiodic_table(aperiodic)
        )

    report.add_html(
        html="".join(parts),
        title=SPECTRA_TITLE,
        section=SPECTRA_SECTION,
        tags=(SPECTRA_TAG,),
        replace=True,
    )
    return spectra


__all__ = [
    "MAX_LABELLED_PAIRED",
    "MAX_LABELLED_PARTICIPANTS",
    "SPECTRA_SECTION",
    "SPECTRA_TAG",
    "SPECTRA_TITLE",
    "CohortSpectra",
    "add_spectra_section",
    "aperiodic_frame",
    "aperiodic_table",
    "cohort_spectra",
    "participant_spectrum",
    "plot_aperiodic_shift",
    "plot_cohort_spectra",
    "spectra_audit",
]
