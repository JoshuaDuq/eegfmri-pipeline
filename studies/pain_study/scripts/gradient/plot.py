"""Figures for the residual scanner gradient.

Draws what ``studies.pain_study.analysis.gradient`` measured: the comb against its local
background, the volume-locked residual envelope, and the cohort comb. Saved as PNG rather
than rendered into a report section.

    eeg-pipeline gradient plot
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from eeg_pipeline.preprocessing.report.style import run_label, separated_labels  # noqa: E402
from studies.pain_study.analysis.gradient.cohort import CohortComb  # noqa: E402
from studies.pain_study.analysis.gradient.comb import CombResidual  # noqa: E402
from studies.pain_study.analysis.gradient.locked import VolumeLockedAverage  # noqa: E402

WORKFLOW = "gradient"
DPI = 200

# Okabe-Ito, matching the stage colours the report section used: the reader compares
# these figures against the rest of the study's before/after panels.
BEFORE_COLOR = "#E69F00"
AFTER_COLOR = "#0072B2"
GUIDE_COLOR = "0.35"

NO_EXCESS_DB = 0.0
MAX_LABELLED_PARTICIPANTS = 8

_CHARACTER_WIDTH_RATIO = 0.62
_LEGEND_HANDLE_POINTS = 34.0


def _comb_run_label(recording_id: str) -> str:
    return run_label(recording_id)


def legend_columns(figure: plt.Figure, labels: Sequence[str], *, fontsize: float) -> int:
    """Columns that keep the widest legend entry inside ``figure``.

    A fixed column count sets the legend's width from the number of entries and ignores
    the figure it has to fit in. On a single-column panel layout that ran the first and
    last of six entries off opposite edges, cut mid-word, so the key explaining which
    trace was which could not be read at all.

    Estimated from the label text rather than measured from a render, because the figure
    is built without a canvas and drawing one here to place a legend would pay a full
    render on every panel. The estimate only has to be good enough to choose between one,
    two and three columns.
    """
    if not labels:
        return 1
    widest = max(len(label) for label in labels)
    entry_points = widest * fontsize * _CHARACTER_WIDTH_RATIO + _LEGEND_HANDLE_POINTS
    figure_points = figure.get_figwidth() * 72.0
    return max(1, min(len(labels), int(figure_points // entry_points)))


def _locked_stage_summary(locked: VolumeLockedAverage) -> tuple[tuple[float, float | None], ...]:
    return (
        (
            float(locked.before_noise_floor_uv),
            locked.before_resolved_amplitude_uv,
        ),
        (
            float(locked.after_noise_floor_uv),
            locked.after_resolved_amplitude_uv,
        ),
    )


def plot_comb_residual(combs: Sequence[CombResidual]) -> plt.Figure:
    """Plot comb excess against frequency for every run, before and after ICA.

    The runs are laid out as a grid on one shared pair of axes rather than as a column
    of independently scaled panels. A session's runs differ from each other by a
    fraction of a decibel, so a stack in which every panel repeated the subject, task
    and repetition time in its title and then chose its own y limits made the runs look
    both more distinct and less comparable than they are.
    """
    if not combs:
        raise ValueError("The comb figure requires at least one measured run.")
    columns = 2 if len(combs) > 3 else 1
    rows = math.ceil(len(combs) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        # The trailing inches are the legend's, not the panels'. Six entries that no
        # longer fit on one line need two or three rows beneath the axes, and taking that
        # space out of the panels instead pushed the tick labels of vertically adjacent
        # panels into each other and into the y-axis label.
        figsize=(5.6 * columns, 2.5 * rows + 1.8),
        squeeze=False,
        sharex=True,
        # Shared limits are the point of the grid: an eye moving between panels should
        # be comparing residuals, not silently recalibrating to each panel's own scale.
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for axis, comb in zip(flat, combs, strict=False):
        for typical, worst, color, label in (
            (comb.before_typical_db, comb.before_worst_db, BEFORE_COLOR, "Before ICA"),
            (comb.after_typical_db, comb.after_worst_db, AFTER_COLOR, "After ICA"),
        ):
            axis.plot(
                comb.harmonic_frequencies_hz,
                typical,
                color=color,
                linewidth=0.9,
                label=label,
            )
            # Filled markers on the harmonics that carry a measurement, hollow on the
            # ones inside the notch stopband. Drawing them alike let the notch's -25 dB
            # trough read as the deepest correction in the figure.
            for mask, facecolor in ((comb.scored, color), (comb.notched, "white")):
                if not mask.any():
                    continue
                axis.plot(
                    comb.harmonic_frequencies_hz[mask],
                    typical[mask],
                    linestyle="none",
                    marker="o",
                    markersize=2.5,
                    markerfacecolor=facecolor,
                    markeredgecolor=color,
                    markeredgewidth=0.6,
                )
            # A residual confined to a few peripheral sensors leaves the median flat. This
            # is an envelope whose contributing channel may change between harmonics, not
            # the trace of one sensor.
            axis.plot(
                comb.harmonic_frequencies_hz,
                worst,
                color=color,
                linewidth=0.8,
                linestyle=":",
            )
        # The zero line is explained in the legend rather than by a caption pinned to
        # the line itself, which landed on top of the traces in every panel.
        axis.axhline(
            0.0,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="0 dB: peak equals background",
        )
        axis.set_title(
            f"{_comb_run_label(comb.recording_id)} · median "
            f"{comb.median_before_excess_db:.1f} → {comb.median_after_excess_db:.1f} dB · "
            f"worst {comb.worst_excess_db:.1f} dB at {comb.worst_harmonic_hz:.1f} Hz "
            f"({comb.worst_channel})",
            fontsize=8,
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    # Limits from the scored harmonics alone. A notch drives its harmonic tens of
    # decibels below background, and that trough -- the pipeline's own filter, excluded
    # from every reported statistic -- was setting the scale for the residual it is not
    # part of: on sub-0012 a -25 dB stopband compressed the +/-10 dB the comb lives in
    # into the top fifth of each panel. The trough stays drawn and now runs off the axis,
    # which is the honest picture of a value this figure does not measure.
    scored_values = np.concatenate(
        [
            trace[comb.scored]
            for comb in combs
            for trace in (
                comb.before_typical_db,
                comb.after_typical_db,
                comb.before_worst_db,
                comb.after_worst_db,
            )
        ]
    )
    finite = scored_values[np.isfinite(scored_values)]
    if finite.size:
        margin = max(0.05 * float(np.ptp(finite)), 1.0)
        flat[0].set_ylim(float(finite.min()) - margin, float(finite.max()) + margin)
    # Few enough ticks that the bottom label of one panel and the top label of the panel
    # below it cannot meet. Stacked panels share an edge, so the default density puts two
    # numbers within a few points of each other and both become hard to read.
    flat[0].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    for axis in flat[len(combs) :]:
        axis.remove()
    for row in range(rows):
        # Sized with the panel titles rather than at the default. Rotated, two lines of
        # default-size text stand almost as tall as a panel, so the label of one row
        # reached the bottom tick label of the row above it.
        flat[row * columns].set_ylabel("Excess over\nbackground (dB)", fontsize=8)
    # The bottom-most surviving panel in each column carries the frequency axis. When
    # the last row is short, ``sharex`` has already hidden the tick labels of the panel
    # above the removed slot, so they are turned back on explicitly.
    for column in range(columns):
        present = [index for index in range(len(combs)) if index % columns == column]
        if not present:
            continue
        axis = flat[present[-1]]
        axis.set_xlabel("Frequency (Hz) · one point per gradient harmonic")
        axis.tick_params(axis="x", labelbottom=True)

    # Repetition time belongs to the acquisition, not to a run, so it is stated once.
    # A session whose runs disagree about it is a finding in itself and is named as one.
    repetition_times = {round(comb.timing.repetition_time_s, 6) for comb in combs}
    if len(repetition_times) == 1:
        timing = combs[0].timing
        timing_text = (
            f"TR {timing.repetition_time_s:.4f} s ({timing.fundamental_hz:.3f} Hz fundamental)"
        )
    else:
        timing_text = "runs differ in repetition time: " + ", ".join(
            f"{_comb_run_label(comb.recording_id)} {comb.timing.repetition_time_s:.4f} s"
            for comb in combs
        )
    figure.suptitle(f"Gradient comb against its local background · {timing_text}", fontsize=10)
    handles, labels = flat[0].get_legend_handles_labels()
    # Colour carries the stage and line style carries the statistic, so the legend states
    # each of those once instead of spelling out all four combinations. As four sentences
    # the widest entry was 39 characters, which no arrangement fits across a single-column
    # panel layout: the key ran off both edges of the figure, cut mid-word.
    for linestyle, linewidth, name in (
        ("-", 0.9, "median channel"),
        (":", 0.8, "channelwise maximum envelope"),
    ):
        handles.append(Line2D([], [], color=GUIDE_COLOR, linestyle=linestyle, linewidth=linewidth))
        labels.append(name)
    # The hollow marker is a legend concept, so it is explained in the legend rather than
    # in a footnote below it. As free-floating figure text the explanation sat in
    # coordinates ``constrained_layout`` never reads, and printed through this very
    # legend; as an entry it also puts the symbol beside the sentence describing it.
    if any(comb.notched.any() for comb in combs):
        notched_count = max(int(comb.notched.sum()) for comb in combs)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=2.5,
                markerfacecolor="white",
                markeredgecolor=GUIDE_COLOR,
                markeredgewidth=0.6,
            )
        )
        labels.append(f"notch stopband: {notched_count} drawn, not scored")
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=legend_columns(figure, labels, fontsize=7),
        frameon=False,
        fontsize=7,
    )
    return figure


def plot_volume_locked_average(averages: Sequence[VolumeLockedAverage]) -> plt.Figure:
    """Plot the gradient-locked residual envelope per run, before and after ICA.

    Laid out as a grid on the same rule as :func:`plot_comb_residual`. A row of one panel
    per run put six panels across 1814 points, four times the width of the report column
    the figure sits in, so the browser scaled or clipped it and each waveform ended up a
    couple of hundred pixels wide.
    """
    if not averages:
        raise ValueError("The volume-locked figure requires at least one measured run.")
    columns = 2 if len(averages) > 3 else 1
    rows = math.ceil(len(averages) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.6 * columns, 2.5 * rows + 1.0),
        squeeze=False,
        sharex=True,
        # The runs are being compared against each other, so they cannot each rescale.
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for axis, locked in zip(flat, averages, strict=False):
        before, after = _locked_stage_summary(locked)
        for values, floor, color, label in (
            (locked.before_rms_uv, before[0], BEFORE_COLOR, "Before ICA"),
            (locked.after_rms_uv, after[0], AFTER_COLOR, "After ICA"),
        ):
            axis.plot(
                locked.times_s,
                values,
                color=color,
                linewidth=1.0,
                label=label,
            )
            # Each trace has its own floor. ICA changes the non-locked variance as well as
            # the average, so applying the after-ICA floor to the before trace can turn an
            # unresolved estimate into an apparently resolved one.
            axis.axhline(
                floor,
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.75,
                label=f"{label} noise floor",
            )
        before_summary = "unresolved" if before[1] is None else f"{before[1]:.2f} µV resolved"
        after_summary = "unresolved" if after[1] is None else f"{after[1]:.2f} µV resolved"
        title = (
            f"{_comb_run_label(locked.recording_id)} · {locked.n_volumes} volumes · "
            f"before {before_summary} → after {after_summary}"
        )
        axis.set_title(title, fontsize=8)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in flat[len(averages) :]:
        axis.remove()
    for row in range(rows):
        flat[row * columns].set_ylabel("RMS across channels of the\nvolume-locked average (µV)")
    # The bottom-most surviving panel in each column carries the time axis. When the last
    # row is short, ``sharex`` has already hidden the tick labels of the panel above the
    # removed slot, so they are turned back on explicitly.
    for column in range(columns):
        present = [index for index in range(len(averages)) if index % columns == column]
        if not present:
            continue
        axis = flat[present[-1]]
        axis.set_xlabel("Time within volume (s)")
        axis.tick_params(axis="x", labelbottom=True)
    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=7,
    )
    figure.suptitle(
        "Residual gradient envelope: each trace has its own odd–even noise floor; only "
        "positive signed excess power supports a resolved amplitude",
        fontsize=9,
    )
    return figure


def plot_cohort_comb(comb: CohortComb) -> plt.Figure:
    """Draw the cohort comb, with every participant visible beneath the summary."""
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


def save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


__all__ = [
    "DPI",
    "legend_columns",
    "plot_cohort_comb",
    "plot_comb_residual",
    "plot_volume_locked_average",
    "save",
]
