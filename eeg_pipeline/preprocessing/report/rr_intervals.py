# Beat-to-beat intervals: ECG physiology, and nothing acquisition-specific is read here.
"""Beat-to-beat interval evidence for one subject.

The tachogram is what makes a detector that worked for four minutes and then lost the
trace visible; a marker count and a median rate cannot show it. Nothing in this module
reads anything acquisition-specific, so it stays in the shared pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets
from eeg_pipeline.preprocessing.report.style import (
    FLAG_COLOR,
    GUIDE_COLOR,
    MARK_COLOR,
    PRIMARY_COLOR,
    RUN_COLORS,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    SpanningRow,
    grid_table,
)

# No default beat annotation. A label is a search instruction, not evidence a train
# exists, so a caller that names none gets no interval series rather than a search for
# somebody else's spelling. Studies set ica.cardiac_review.marker_description.
DEFAULT_BEAT_MARKER_DESCRIPTION: str | None = None

RR_SECTION = "Cardiac rhythm"
RR_TAG = "rr-intervals"

#: Beats needed before an interval series describes a rhythm rather than a few markers.
MINIMUM_BEATS = 3


#: Multiple of the run's median interval at which a single missed beat lands.
MISSED_BEAT_FACTOR = 1.5


#: Physiologically possible heart rate, in beats per minute.
#:
#: The default for ``report.thresholds.plausible_heart_rate_bpm``. 30–220 bpm covers an
#: adult at rest through an adult under load; a paediatric or developmental cohort sits
#: higher and configures it.
#:
#: Stated in bpm because that is the unit the claim is made in, and because the cohort
#: panel counts against it directly. The subject tachogram needs it as intervals and
#: derives them below rather than carrying a second number: the two constants this
#: replaced were a range in seconds here and a range in bpm in the cohort module, and
#: they disagreed at the ceiling — 200 against 220 — so a participant at 210 bpm was
#: implausible in one document and ordinary in the other.
DEFAULT_PLAUSIBLE_HEART_RATE_BPM = (30.0, 220.0)


#: Interval range a working detector stays inside, as the rate above implies.
#:
#: Derived, never written down separately, so the seconds and the bpm cannot drift apart.
#: This is the physiological statement; it is not the axis. See :data:`DRAWN_RR_RANGE_S`.
PLAUSIBLE_RR_RANGE_S = (
    60.0 / DEFAULT_PLAUSIBLE_HEART_RATE_BPM[1],
    60.0 / DEFAULT_PLAUSIBLE_HEART_RATE_BPM[0],
)


#: Interval window the tachogram panels are drawn over, in seconds, on a log axis.
#:
#: Fixed rather than taken from the data, so a run with a failed detector cannot rescale
#: the panels beside it and the same interval occupies the same height in every report.
#:
#: Wider than :data:`PLAUSIBLE_RR_RANGE_S`, and logarithmic, because clipping to the
#: plausible range censored exactly the runs the panel exists to expose. On sub-0012
#: run-1, 82 of 84 long intervals fell outside a linear 0.3–2 s window and were drawn
#: stacked on the boundary: the count reached the title, but a 2.1 s gap and a 60 s one
#: became the same mark, and the magnitude is the measurement. Logarithmic keeps the
#: ordinary rhythm legible while placing a lapse where it actually falls — the plausible
#: band still owns more than half the panel height, which is what the fixed window was
#: protecting. Samples outside even this window are drawn on the boundary and counted.
DRAWN_RR_RANGE_S = (0.3, 10.0)


@dataclass(frozen=True)
class RrIntervals:
    """Beat-to-beat intervals derived from one run's R markers."""

    recording_id: str
    #: Onset of each beat, in seconds from the run start.
    beat_times_s: np.ndarray
    #: Interval preceding each beat after the first, in seconds.
    intervals_s: np.ndarray

    @property
    def median_interval_s(self) -> float:
        return float(np.median(self.intervals_s))

    @property
    def median_bpm(self) -> float:
        return 60.0 / self.median_interval_s

    @property
    def dropout_threshold_s(self) -> float:
        """Interval above which a beat was more likely missed than merely slow."""
        return MISSED_BEAT_FACTOR * self.median_interval_s

    @property
    def dropout_count(self) -> int:
        """Intervals long enough to be a missed beat rather than a slow one.

        A detector that misses one beat produces an interval near twice the median, so a
        threshold below that catches it while leaving ordinary variability alone. This
        separates "this participant's heart rate varied" from "the detector lost beats",
        which the median rate alone cannot distinguish and which decides whether the
        pulse correction had a complete marker set to work from.
        """
        return int(np.sum(self.intervals_s > self.dropout_threshold_s))


def compute_rr_intervals(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    description: str | None = None,
) -> RrIntervals | None:
    """Extract beat-to-beat intervals from the R markers in one run.

    Bad pulse correction traces back to bad R detection, and the QC table records only a
    marker count and a median rate. Neither shows a detector that worked for four
    minutes and then lost the trace, which is what the interval series makes visible.
    """
    label = description or DEFAULT_BEAT_MARKER_DESCRIPTION
    if not label:
        return None
    onsets = annotation_onsets(raw, label)
    if onsets.size < MINIMUM_BEATS:
        return None
    intervals = np.diff(onsets)
    if not np.all(intervals > 0):
        raise ValueError(f"{recording_id}: R markers are not strictly increasing.")
    return RrIntervals(
        recording_id=recording_id,
        beat_times_s=onsets[1:],
        intervals_s=intervals,
    )


def plot_rr_intervals(
    series: Sequence[RrIntervals],
    *,
    missing: Sequence[str] = (),
    plausible_rr_range_s: tuple[float, float] = PLAUSIBLE_RR_RANGE_S,
) -> plt.Figure:
    """Plot the beat-to-beat interval series for every run.

    ``missing`` names runs whose marker train was too short to describe a rhythm. They are
    listed rather than dropped: a reader who sees runs 1, 3, 5 and 6 cannot tell whether
    runs 2 and 4 were not acquired, were not measured, or failed, and the answer decides
    whether the pulse correction had markers to work from at all.

    Every panel shares one fixed logarithmic window, :data:`DRAWN_RR_RANGE_S`, so a run
    can be read against its neighbours.

    The window is a constant rather than the range of the data. Letting the data set it
    meant one run whose detector had collapsed — seven markers across eight minutes,
    intervals of two minutes — stretched the shared axis, and the ordinary beat-to-beat
    variation of every working run was flattened into a band a few pixels tall. What the
    figure resolves does not depend on which runs happen to share it.

    It is logarithmic, and wider than :data:`PLAUSIBLE_RR_RANGE_S`, because a linear
    window clipped to the plausible range censored the runs the panel exists to expose:
    on sub-0012 run-1, 82 of 84 long intervals landed on the boundary, which reported how
    many there were and not how long any of them was. The plausible band still occupies
    more than half the height, so the rhythm stays readable.

    Intervals outside even this window are drawn as markers on the boundary they exceeded
    and counted in the panel title, so a detector that collapsed entirely still reads as
    collapsed, and no threshold decides which runs "look like a rhythm".
    """
    if not series:
        raise ValueError("The tachogram requires at least one run with R markers.")
    low, high = DRAWN_RR_RANGE_S
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(10.0, 1.9 * len(series) + 1.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, run in zip(axes[:, 0], series, strict=True):
        minutes = run.beat_times_s / 60.0
        intervals = run.intervals_s
        inside = (intervals >= low) & (intervals <= high)
        axis.plot(
            minutes,
            np.where(inside, intervals, np.nan),
            color=PRIMARY_COLOR,
            linewidth=0.7,
            marker=".",
            markersize=1.6,
        )
        # Drawn on the boundary rather than dropped, and in the annotation colour rather
        # than the series colour, so a clipped sample cannot be mistaken for a measured
        # one sitting at the edge of the range.
        for outside, edge, marker in (
            (intervals > high, high, "^"),
            (intervals < low, low, "v"),
        ):
            if outside.any():
                axis.plot(
                    minutes[outside],
                    np.full(int(outside.sum()), edge),
                    linestyle="none",
                    marker=marker,
                    markersize=4.0,
                    color=MARK_COLOR,
                    clip_on=False,
                )
        axis.axhline(run.median_interval_s, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
        # The same threshold the dropout count uses, so the figure and the table cannot
        # disagree about which intervals were counted.
        axis.axhline(run.dropout_threshold_s, color=FLAG_COLOR, linestyle=":", linewidth=1.0)
        clipped = int((~inside).sum())
        title = (
            f"{run_label(run.recording_id)} · {run.intervals_s.size + 1} beats · "
            f"median {run.median_bpm:.0f} bpm · {run.dropout_count} interval(s) "
            f"above {MISSED_BEAT_FACTOR:g}× the median"
        )
        if clipped:
            title += f" · {clipped} outside the drawn range"
        # The band a working detector stays inside, drawn so the widened axis still says
        # where "plausible" ends without clipping anything to it.
        axis.axhspan(
            *plausible_rr_range_s,
            color=GUIDE_COLOR,
            alpha=0.07,
            linewidth=0,
            zorder=0,
        )
        axis.set(title=title, ylabel="RR (s)", ylim=(low, high), yscale="log")
        # Plain seconds rather than the powers of ten a log axis labels by default: the
        # reader is comparing intervals against a heart rate, and "10^0" is not a number
        # anyone converts to bpm in their head.
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_yticks([0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time in run (min)")
    # The title names the figure. What the lines and shading mean is four sentences of
    # method, and four sentences set in 9 pt above six stacked panels is a paragraph
    # nobody reads in the one place it cannot be made larger; ``rr_interval_html`` states
    # it in the section, where it is prose.
    #
    # What stays is what belongs to *this* subject rather than to the method: the runs
    # that produced no series at all, which is a fact about the recording and is not
    # recoverable from the panels, because a run with no series has no panel.
    caption = "Beat-to-beat intervals from the R markers"
    if missing:
        caption += (
            "\nNo interval series for "
            + ", ".join(run_label(recording_id) for recording_id in missing)
            + f": fewer than {MINIMUM_BEATS} R markers were found"
        )
    figure.suptitle(caption, fontsize=9)
    plt.close(figure)
    return figure


def plot_rr_poincare(
    series: Sequence[RrIntervals],
    *,
    plausible_rr_range_s: tuple[float, float] = PLAUSIBLE_RR_RANGE_S,
) -> plt.Figure:
    """Plot each interval against the one after it, for every run.

    The time series answers "when did detection go wrong". This answers "what went
    wrong", which it cannot: a missed beat and a spuriously doubled one both appear there
    as a single point away from the median, and they call for opposite responses.

    Here they separate. Detection is self-correcting in a specific way — a missed beat
    merges two intervals into one near twice the median and the next interval is normal,
    so the pair lands on the horizontal 2× reference; a double detection splits one
    interval into two halves, landing the pair on the 0.5× reference. Genuine
    variability, which changes both intervals of a pair together, stays on the identity
    line. Reference lines are drawn for all three, and interpreting the scatter against
    them is left to the reviewer: an arrhythmia can put points off the identity line too.

    Intervals are pooled per run and coloured by run, so a detector that failed in one
    run only is visible as a cloud of one colour away from the diagonal.
    """
    if not series:
        raise ValueError("The Poincaré plot requires at least one run with R markers.")
    low, high = plausible_rr_range_s
    figure, axis = plt.subplots(figsize=(5.6, 5.4), layout="constrained")

    for index, run in enumerate(series):
        intervals = run.intervals_s
        if intervals.size < 2:
            continue
        axis.scatter(
            intervals[:-1],
            intervals[1:],
            s=6,
            alpha=0.55,
            linewidths=0.0,
            color=RUN_COLORS[index % len(RUN_COLORS)],
            label=run_label(run.recording_id),
        )

    reference = np.array([low, high])
    # The detection-failure guides carry MARK_COLOR rather than FLAG_COLOR: the scatter
    # already spends the hues on runs, and RUN_COLORS hands vermillion to the second run,
    # so a vermillion reference line and run-2's cloud were the same ink. They also take
    # different dash patterns, because two guides that differ only in slope are told
    # apart by their legend entries otherwise, and the legend is not beside the line.
    for factor, style, color, label in (
        (1.0, "--", GUIDE_COLOR, "RRₙ₊₁ = RRₙ (no change)"),
        (2.0, ":", MARK_COLOR, "2× — one beat missed"),
        (0.5, "-.", MARK_COLOR, "0.5× — one beat counted twice"),
    ):
        axis.plot(
            reference,
            np.clip(reference * factor, low, high),
            color=color,
            linestyle=style,
            linewidth=1.0,
            label=label,
        )
    axis.set(
        title="Each interval against the next",
        xlabel="RRₙ (s)",
        ylabel="RRₙ₊₁ (s)",
        xlim=(low, high),
        ylim=(low, high),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    plt.close(figure)
    return figure


def rr_intervals_html(series: Sequence[RrIntervals], *, missing: Sequence[str] = ()) -> str:
    """Render the per-run beat detection record.

    Runs named in ``missing`` get a row stating that no series could be built, so the
    table lists every run that was measured rather than only those that succeeded.
    """
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Beats"),
        Column("Median (bpm)"),
        Column("RR 5th–95th percentile (s)"),
        Column(f"Intervals above {MISSED_BEAT_FACTOR:g}× median"),
    )
    rows: list[Sequence[object] | SpanningRow] = [
        [
            run_label(run.recording_id),
            run.intervals_s.size + 1,
            f"{run.median_bpm:.0f}",
            f"{float(np.percentile(run.intervals_s, 5)):.2f}–"
            f"{float(np.percentile(run.intervals_s, 95)):.2f}",
            run.dropout_count,
        ]
        for run in series
    ]
    rows += [
        SpanningRow(
            lead=[run_label(recording_id)],
            note=f"No interval series: fewer than {MINIMUM_BEATS} R markers",
        )
        for recording_id in missing
    ]
    return (
        "<p>Pulse-artifact correction can only be as good as the R markers it was "
        "driven by, and a marker count with a median rate cannot show a detector that "
        "worked for part of a run and then lost the trace. These are the intervals "
        "themselves.</p>"
        + grid_table(columns, rows)
        + "<p>A missed beat produces an interval near twice the median, so the last "
        "column separates heart-rate variability from detection dropout. Interpreting "
        "the count is left to the reviewer: a run with genuine arrhythmia and a run "
        "with a failing detector both raise it, and only the ECG trace distinguishes "
        "them.</p>"
        # Stated here rather than above the figure. It is four sentences of method, the
        # same for every subject and every run, and set small enough to fit over six
        # stacked panels it was a paragraph in the one place on the page that cannot be
        # made bigger.
        "<p>In the panels below, the dashed line is each run's median interval and the "
        f"dotted line {MISSED_BEAT_FACTOR:g}&times; that median, above which an interval "
        "is counted as a missed beat. The axis is logarithmic and fixed across runs so "
        f"the same interval sits at the same height in every panel; shading marks the "
        f"{PLAUSIBLE_RR_RANGE_S[0]:.2f}&ndash;{PLAUSIBLE_RR_RANGE_S[1]:.2f} s a working "
        "detector stays inside. Intervals beyond the axis are drawn as triangles on the "
        "boundary they exceeded, so a lapse is visible as a lapse rather than lost off "
        "the top.</p>"
    )


def add_rr_interval_section(
    *,
    report: mne.Report,
    series: Sequence[RrIntervals],
    missing: Sequence[str] = (),
    section: str = RR_SECTION,
    plausible_rr_range_s: tuple[float, float] = PLAUSIBLE_RR_RANGE_S,
) -> None:
    """Append the beat-detection record to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not series:
        raise ValueError("The tachogram requires at least one run with R markers.")
    remove_tagged_content(report, tag=RR_TAG)
    report.add_html(
        html=rr_intervals_html(series, missing=missing),
        title="Beat detection by run",
        section=section,
        tags=("raw", RR_TAG),
        replace=True,
    )
    report.add_figure(
        fig=plot_rr_intervals(
            series, missing=missing, plausible_rr_range_s=plausible_rr_range_s
        ),
        title="Beat-to-beat intervals",
        section=section,
        tags=("raw", RR_TAG),
        image_format=report_image_format(),
        replace=True,
    )
    report.add_figure(
        fig=plot_rr_poincare(series, plausible_rr_range_s=plausible_rr_range_s),
        title="Each interval against the next",
        section=section,
        tags=("raw", RR_TAG),
        image_format=report_image_format(),
        replace=True,
    )


__all__ = [
    "DEFAULT_BEAT_MARKER_DESCRIPTION",
    "DEFAULT_PLAUSIBLE_HEART_RATE_BPM",
    "DRAWN_RR_RANGE_S",
    "MINIMUM_BEATS",
    "MISSED_BEAT_FACTOR",
    "PLAUSIBLE_RR_RANGE_S",
    "RR_SECTION",
    "RR_TAG",
    "RrIntervals",
    "add_rr_interval_section",
    "compute_rr_intervals",
    "plot_rr_intervals",
    "plot_rr_poincare",
    "rr_intervals_html",
]
