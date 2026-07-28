"""Time-resolved data quality across each continuous run.

Every other panel in the report summarises a run into one number per channel, per
component, or per epoch. None of them answer *when*. That matters most inside a scanner,
where the dominant failure is not a uniformly poor recording but a good recording with a
bad stretch in it: the participant shifts in the bore, the gradient correction template
stops matching, and from that moment the run is contaminated while its run-level
averages stay unremarkable.

The epoch-rejection panel comes closest, but it is indexed by epoch position and only
covers time that was epoched at all. Breaks, pre-task segments, and the gaps between
trials are invisible to it, and those are exactly where a scanner run goes wrong.

Amplitude is shown relative to each channel's own median over the run, so the panel
reads as change over time rather than as a map of which channels are loud. A channel
that is uniformly noisy is the bad-channel panel's business; this one is for the moment
something changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets
from eeg_pipeline.preprocessing.report.style import (
    DIVERGING_POWER_COLORMAP,
    FLAG_COLOR,
    GUIDE_COLOR,
    robust_symmetric_limit,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Window over which amplitude is pooled. Short enough to localise a movement to a few
#: seconds, long enough that the RMS of a single window is a stable estimate.
WINDOW_SECONDS = 1.0

#: Multiple of the median volume interval above which a gap in the marker train is
#: treated as an interruption rather than jitter.
VOLUME_GAP_FACTOR = 1.5

#: Annotation prefix MNE uses for spans excluded from processing.
BAD_ANNOTATION_PREFIX = "BAD"

#: High-pass time constants a run is given to settle before its amplitude is comparable.
#:
#: A high-pass filter rings at the start of a record, and that ringing is the largest
#: amplitude in the run: every run of one subject reported its worst excursion at 0.0 min
#: with the same value, which is the filter rather than the participant, and no real
#: excursion later in the run could exceed it. The settling span is therefore excluded
#: from the excursion *statistic* while still being drawn in the trace.
#:
#: Three time constants leaves under 5% of the step response, which is the usual
#: engineering convention for "settled" rather than a threshold tuned on this data. The
#: span itself comes from the recording's own high-pass, so a differently filtered dataset
#: gets a differently sized exclusion and an unfiltered one gets none.
SETTLING_TIME_CONSTANTS = 3.0


def _settling_seconds(highpass_hz: float) -> float:
    """Seconds a high-pass at ``highpass_hz`` needs to settle, or 0 when there is none."""
    if not np.isfinite(highpass_hz) or highpass_hz <= 0.0:
        return 0.0
    return float(SETTLING_TIME_CONSTANTS / (2.0 * np.pi * highpass_hz))


@dataclass(frozen=True)
class RunContinuity:
    """Windowed amplitude over one run, with the spans that were excluded from it."""

    recording_id: str
    window_seconds: float
    #: Centre time of each window, in seconds from the run start.
    times_s: np.ndarray
    channel_names: tuple[str, ...]
    #: ``(n_channels, n_windows)`` amplitude relative to each channel's run median, dB.
    relative_db: np.ndarray
    #: ``(onset, duration)`` of every BAD_* annotation.
    bad_spans: tuple[tuple[float, float], ...]
    #: ``(onset, duration)`` of every interruption in the volume-marker train.
    volume_gaps: tuple[tuple[float, float], ...]
    duration_s: float
    #: Span at the run start excluded from the excursion statistic, in seconds.
    settling_s: float = 0.0
    #: Onset of every task event, in seconds from the run start.
    #:
    #: Drawn as a rug beneath the time axis so that a bad stretch can be read against the
    #: trials it covers. "A +12 dB excursion at 5.5 min" is a fact about the recording;
    #: "it covers four trials" is the fact that decides what to do about it, and nothing
    #: else in the report puts the two on one axis.
    #:
    #: Empty for a resting-state recording, which has no events, and for a run whose
    #: annotations are all scanner and artifact marks. The rug is then not drawn at all,
    #: rather than drawn empty.
    event_onsets: tuple[float, ...] = ()
    #: Whether this run was searched for a volume-marker train at all.
    #:
    #: Distinct from ``volume_gaps`` being empty, which means the train was searched and
    #: found continuous. An EEG-only recording has no train to search, and reporting "0
    #: gaps" for it states a measurement that was never made.
    has_volume_markers: bool = False
    #: Whether the channel rows were sorted down the head rather than left in file order.
    #:
    #: Carried so the axis can say which of the two it is. At a full montage most rows have
    #: no tick label and are read by position, and that reading is wrong if the reader
    #: assumes the ordering the figure does not use.
    ordered_by_position: bool = False

    @property
    def bad_fraction(self) -> float:
        """Share of the run inside a BAD_* span."""
        if self.duration_s <= 0:
            return 0.0
        return sum(duration for _, duration in self.bad_spans) / self.duration_s

    @property
    def excursion_db(self) -> np.ndarray:
        """Across-channel median deviation per window.

        Movement moves the whole montage at once, so the across-channel median rises
        with it. A single channel misbehaving leaves this trace flat, which is the
        distinction between a bad channel and a bad moment.
        """
        return np.median(self.relative_db, axis=0)

    @property
    def settled_mask(self) -> np.ndarray:
        """Windows whose amplitude is comparable with the rest of the run.

        Excludes the high-pass settling span at the run start. Never excludes every
        window: a run shorter than its own settling time would otherwise have no
        excursion at all, and reporting the transient is better than reporting nothing.
        """
        mask = self.times_s >= self.settling_s
        return mask if mask.any() else np.ones_like(self.times_s, dtype=bool)

    @property
    def worst_window_s(self) -> float:
        settled = self.settled_mask
        excursion = np.where(settled, self.excursion_db, -np.inf)
        return float(self.times_s[int(np.argmax(excursion))])

    @property
    def worst_excursion_db(self) -> float:
        return float(np.max(self.excursion_db[self.settled_mask]))


def _bad_spans(raw: mne.io.BaseRaw) -> tuple[tuple[float, float], ...]:
    """Return every BAD_* annotation as ``(onset, duration)`` from the run start."""
    start = raw.first_time
    return tuple(
        (float(annotation["onset"] - start), float(annotation["duration"]))
        for annotation in raw.annotations
        if str(annotation["description"]).upper().startswith(BAD_ANNOTATION_PREFIX)
    )


def _volume_gaps(
    raw: mne.io.BaseRaw,
    *,
    description: str,
    factor: float = VOLUME_GAP_FACTOR,
) -> tuple[tuple[float, float], ...]:
    """Find interruptions in the volume-marker train.

    A gap means the scanner stopped, or the markers were lost. Either way the gradient
    correction on both sides of it was built from different conditions, so the boundary
    is worth seeing next to the amplitude.
    """
    onsets = annotation_onsets(raw, description)
    if onsets.size < 3:
        return ()
    intervals = np.diff(onsets)
    threshold = float(np.median(intervals)) * factor
    return tuple(
        (float(onsets[index]), float(intervals[index]))
        for index in np.flatnonzero(intervals > threshold)
    )


#: Annotation descriptions that are not task events.
#:
#: Everything the acquisition and this pipeline write for their own purposes: scanner
#: volume and R markers, BAD spans, recording-segment boundaries. Identified by exclusion
#: rather than by listing the event names, because the event names belong to whatever
#: paradigm produced the data and cannot be enumerated here.
_NON_EVENT_PREFIXES = ("BAD", "EDGE", "NEW SEGMENT", "VOLUME/", "R  ", "R/", "RESPONSE/")


def _event_onsets(raw: mne.io.BaseRaw, *, volume_description: str | None) -> tuple[float, ...]:
    """Return the onset of every task event, in seconds from the run start."""
    start = raw.first_time
    excluded = set(_NON_EVENT_PREFIXES)
    if volume_description:
        excluded.add(volume_description.upper())
    onsets = []
    for annotation in raw.annotations:
        description = str(annotation["description"]).upper()
        if description.startswith(tuple(excluded)):
            continue
        onsets.append(float(annotation["onset"] - start))
    return tuple(onsets)


def _anterior_to_posterior(raw: mne.io.BaseRaw, picks: np.ndarray) -> np.ndarray:
    """Order picks front-to-back on the scalp, or leave them alone if positions are absent.

    Acquisition order spirals around the head, so consecutive rows of the amplitude map
    are not neighbours on the scalp. Every artifact this panel is meant to catch is
    spatially smooth — a blink, a head movement, a lead loop pulling on one region — and
    in acquisition order each of them breaks into stripes with rows from the opposite side
    of the head in between. Sorting by the anterior-posterior axis makes a region moving
    together look like a region moving together.

    Only that axis. A full spatial ordering of a 2-D montage onto one axis does not exist,
    and anterior-posterior is the one that matters here: it is the axis ocular artifact,
    neck muscle, and the gradient's own topography are organised along.

    Positions are optional. A montage that was never set leaves every location at the
    origin, and sorting on that would impose an arbitrary order while looking principled,
    so the file order is kept instead.
    """
    positions = np.array([raw.info["chs"][index]["loc"][:3] for index in picks], dtype=float)
    if not np.all(np.isfinite(positions)) or np.allclose(positions, 0.0):
        return picks
    anterior = positions[:, 1]
    if np.allclose(anterior, anterior[0]):
        return picks
    # Descending: the most anterior channel is the top row, which is how a topography is
    # drawn and how a reader expects "down the head" to run.
    return picks[np.argsort(-anterior, kind="stable")]


def compute_run_continuity(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    window_seconds: float = WINDOW_SECONDS,
    volume_description: str | None = None,
) -> RunContinuity:
    """Measure windowed amplitude across one continuous run."""
    if window_seconds <= 0:
        raise ValueError("The continuity window must be positive.")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size == 0:
        raise ValueError("Time-resolved quality requires at least one good EEG channel.")
    ordered_picks = _anterior_to_posterior(raw, picks)
    ordered_by_position = not np.array_equal(ordered_picks, picks)
    picks = ordered_picks

    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(window_seconds * sfreq))
    if window_samples < 1:
        raise ValueError("The continuity window is shorter than one sample.")
    n_windows = raw.n_times // window_samples
    if n_windows < 2:
        raise ValueError(
            f"{recording_id}: the run is shorter than two {window_seconds:g} s windows."
        )

    data = raw.get_data(picks=picks)[:, : n_windows * window_samples]
    windowed = data.reshape(len(picks), n_windows, window_samples)
    rms = np.sqrt(np.mean(windowed**2, axis=2))

    # Relative to each channel's own median, so the panel shows change over time rather
    # than which channels are loud, and so one high-amplitude sensor cannot set the
    # colour scale for the whole run.
    reference = np.median(rms, axis=1, keepdims=True)
    tiny = np.finfo(float).tiny
    relative_db = 20.0 * np.log10(np.maximum(rms, tiny) / np.maximum(reference, tiny))

    centres = (np.arange(n_windows) + 0.5) * window_seconds
    return RunContinuity(
        recording_id=recording_id,
        window_seconds=float(window_seconds),
        times_s=centres,
        channel_names=tuple(raw.ch_names[index] for index in picks),
        relative_db=relative_db,
        bad_spans=_bad_spans(raw),
        volume_gaps=(
            _volume_gaps(raw, description=volume_description) if volume_description else ()
        ),
        duration_s=float(raw.n_times / sfreq),
        settling_s=_settling_seconds(float(raw.info["highpass"] or 0.0)),
        event_onsets=_event_onsets(raw, volume_description=volume_description),
        has_volume_markers=bool(volume_description),
        ordered_by_position=ordered_by_position,
    )


def continuity_html(runs: Sequence[RunContinuity]) -> str:
    """Render when each run departed from its own baseline."""
    if not runs:
        raise ValueError("The continuity summary requires at least one run.")
    # A column that can only ever read zero is not a measurement. Outside a scanner there
    # is no volume-marker train to interrupt, so the column and the paragraphs explaining
    # it are dropped rather than left to answer a question about absent equipment.
    scanner = any(run.has_volume_markers for run in runs)
    columns = [
        Column("Run", align=Align.TEXT),
        Column("Duration (min)"),
        Column("Marked bad"),
        *([Column("Volume-marker gaps")] if scanner else []),
        Column("Largest excursion", align=Align.TEXT),
    ]
    rows = [
        [
            run_label(run.recording_id),
            f"{run.duration_s / 60.0:.1f}",
            f"{run.bad_fraction:.1%}",
            *([len(run.volume_gaps)] if scanner else []),
            f"{run.worst_excursion_db:+.1f} dB at {run.worst_window_s / 60.0:.1f} min",
        ]
        for run in runs
    ]
    why_it_matters = (
        " and inside a scanner the second is the common case: once the participant "
        "shifts, the gradient template stops matching and everything after that moment "
        "is contaminated"
        if scanner
        else ", and only the second is recoverable by excluding the stretch that failed"
    )
    gap_note = (
        " A gap in the volume-marker train means the scanner stopped or markers were "
        "lost, which leaves the correction either side of it built under different "
        "conditions."
        if scanner
        else ""
    )
    return (
        "<p>Amplitude in "
        f"{runs[0].window_seconds:g} s windows, expressed relative to each channel's own "
        "median over the run. A run-level average cannot distinguish a uniformly "
        "marginal recording from a good one with a bad stretch in it"
        f"{why_it_matters}.</p>"
        + grid_table(columns, rows)
        + "<p>The excursion is the across-channel median, so it responds to the whole "
        "montage moving together rather than to one sensor misbehaving."
        f"{gap_note}</p>" + _settling_note(runs)
    )


def _settling_note(runs: Sequence[RunContinuity]) -> str:
    """State the span excluded from the excursion, or say nothing when none was."""
    spans = {round(run.settling_s, 3) for run in runs if run.settling_s > 0.0}
    if not spans:
        return ""
    span = f"{max(spans):.1f} s" if len(spans) == 1 else f"up to {max(spans):.1f} s"
    return (
        f"<p>The first {span} of each run is drawn but excluded from the largest-excursion "
        "column. A high-pass filter rings as a record starts, and that ringing is the "
        f"largest amplitude in the run — {SETTLING_TIME_CONSTANTS:g} time constants of the "
        "recording's own high-pass are allowed for it to settle, so the column reports the "
        "worst moment during the run rather than the moment the filter started.</p>"
    )


#: Headroom left above and below the settled excursion range, in decibels.
_TRACE_MARGIN_DB = 1.5

#: Most channel names labelled on the map's vertical axis.
#:
#: A 63-channel montage previously labelled every second row, which at the size those
#: labels have to be to fit is roughly 4 pt: present, and unreadable. Rows are ordered
#: down the head, so a reader locates an unlabelled channel by position between two
#: labelled ones — which only works if the labels can be read at all.
_MAX_CHANNEL_TICKS = 14


def _trace_limits(run: RunContinuity) -> tuple[float, float]:
    """Bound the excursion trace by the span the excursion column is measured over.

    The high-pass transient at the record start is the largest amplitude in the run by a
    wide margin — on a 0.1 Hz high-pass it reached +25 dB where the rest of the run spans
    ±10 — so an autoscaled axis is set by the one span the figure has already declared it
    is not reporting. The trace and its hatch still cross the top of the axis, which is
    what a clipped transient should look like; what changes is that the remaining 8
    minutes are no longer flattened into the bottom fifth of the panel.

    Zero is always included: the trace is a deviation from each channel's own median, and
    an axis that excluded its own reference would misstate the sign of everything on it.
    """
    settled = run.excursion_db[run.settled_mask]
    low = min(0.0, float(np.min(settled))) - _TRACE_MARGIN_DB
    high = max(0.0, float(np.max(settled))) + _TRACE_MARGIN_DB
    return low, high


def plot_run_continuity(run: RunContinuity) -> plt.Figure:
    """Plot the channel-by-time amplitude map with excluded spans marked."""
    figure, (map_axis, trace_axis) = plt.subplots(
        2,
        1,
        figsize=(11.0, 5.6),
        height_ratios=(3, 1),
        sharex=True,
        layout="constrained",
    )
    limit = robust_symmetric_limit(run.relative_db)
    minutes = run.times_s / 60.0
    mesh = map_axis.pcolormesh(
        minutes,
        np.arange(len(run.channel_names)),
        run.relative_db,
        cmap=DIVERGING_POWER_COLORMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="nearest",
        rasterized=True,
    )
    figure.colorbar(
        mesh,
        ax=map_axis,
        label=f"Amplitude vs channel median (dB, clipped at ±{limit:.1f})",
        shrink=0.85,
    )
    step = max(1, int(np.ceil(len(run.channel_names) / _MAX_CHANNEL_TICKS)))
    map_axis.set(
        title=(
            f"{run.recording_id} · {run.duration_s / 60.0:.1f} min · "
            f"{run.bad_fraction:.1%} marked bad"
        ),
        ylabel=(
            "EEG channel (anterior → posterior)"
            if run.ordered_by_position
            else "EEG channel (acquisition order)"
        ),
        yticks=np.arange(0, len(run.channel_names), step),
        yticklabels=run.channel_names[::step],
    )
    map_axis.tick_params(axis="y", labelsize=7)
    if run.ordered_by_position:
        # Row 0 is the most anterior channel, and pcolormesh draws row 0 at the bottom.
        # Left alone the head is upside down relative to the axis label and to every
        # topography in the report, both of which put the front at the top.
        map_axis.invert_yaxis()

    trace_axis.plot(minutes, run.excursion_db, color="0.30", linewidth=0.9)
    trace_axis.axhline(0.0, color="black", linewidth=0.8)
    # The settling span stays on the trace and is hatched instead, so the transient is
    # visibly set aside rather than quietly missing from a figure that reports a maximum.
    if run.settling_s > 0.0:
        for axis in (map_axis, trace_axis):
            axis.axvspan(
                0.0,
                run.settling_s / 60.0,
                facecolor="none",
                edgecolor=GUIDE_COLOR,
                hatch="///",
                linewidth=0.0,
                alpha=0.5,
            )
        trace_axis.annotate(
            "filter settling,\nexcluded from the maximum",
            xy=(run.settling_s / 60.0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6,
            color=GUIDE_COLOR,
        )
    trace_axis.set(
        xlabel="Time in run (min)",
        ylabel="Median across\nchannels (dB)",
        ylim=_trace_limits(run),
    )
    # Task events as a rug along the bottom of the trace, inside the axis rather than in
    # a panel of their own: the point is to read an excursion against the trials it
    # covers, and a separate strip puts a gap between the two things being compared.
    # Ticks, not full-height lines — on a 66-trial run those would be a picket fence
    # drawn over the measurement.
    #
    # Drawn after the limits are fixed, because the rug is pinned to the axis floor and
    # would otherwise sit at whatever the autoscale had chosen a moment earlier.
    if run.event_onsets:
        onsets = np.asarray(run.event_onsets, dtype=float) / 60.0
        # Positioned in a blended transform — time from the data, height from the axis —
        # so the ticks sit just inside the floor whatever the decibel limits turn out to
        # be. Pinned to the floor in data coordinates they landed under the spine and
        # were invisible.
        trace_axis.vlines(
            onsets,
            0.0,
            0.08,
            transform=trace_axis.get_xaxis_transform(),
            color=GUIDE_COLOR,
            linewidth=0.9,
            label=f"{len(run.event_onsets)} event(s)",
        )
        trace_axis.legend(frameon=False, fontsize=6, loc="upper right")
    trace_axis.grid(alpha=0.2)
    trace_axis.spines[["top", "right"]].set_visible(False)

    for axis in (map_axis, trace_axis):
        for onset, duration in run.bad_spans:
            axis.axvspan(
                onset / 60.0,
                (onset + duration) / 60.0,
                color=FLAG_COLOR,
                alpha=0.18,
                linewidth=0,
            )
        for onset, duration in run.volume_gaps:
            axis.axvspan(
                onset / 60.0,
                (onset + duration) / 60.0,
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.6,
                alpha=0.8,
            )
    labels = []
    if run.bad_spans:
        labels.append(f"{len(run.bad_spans)} BAD_* span(s), shaded")
    if run.volume_gaps:
        labels.append(f"{len(run.volume_gaps)} volume-marker gap(s), hatched")
    if labels:
        # Appended to the title rather than annotated just above the axis, which is the
        # same space the title occupies: on any run with a BAD span the two overlapped
        # and both became unreadable.
        map_axis.set_title(
            f"{map_axis.get_title()}\n{' · '.join(labels)}",
            fontsize=plt.rcParams["axes.titlesize"],
        )
    plt.close(figure)
    return figure


def add_continuity_section(
    *,
    report: mne.Report,
    runs: Sequence[RunContinuity],
    section: str = "Data quality over time",
) -> None:
    """Append the time-resolved quality panels to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        drop_replaced_raw_time_series,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not runs:
        raise ValueError("Time-resolved quality requires at least one run.")
    remove_tagged_content(report, tag="run-continuity")
    # Paired with the panels below, which answer the same question over the whole run
    # rather than over five arbitrary seconds of it. Dropped here rather than on open so
    # that a report without this section keeps MNE's panel instead of losing both.
    drop_replaced_raw_time_series(report)
    report.add_html(
        html=continuity_html(runs),
        title="When each run departed from its baseline",
        section=section,
        tags=("raw", "run-continuity"),
        replace=True,
    )
    report.add_figure(
        fig=[plot_run_continuity(run) for run in runs],
        title=f"Amplitude over time by channel — {len(runs)} figures, use the slider",
        caption=[run.recording_id for run in runs],
        section=section,
        tags=("raw", "run-continuity"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    # Measured from the raw run and independent of the ICA, so it belongs with the other
    # input-quality evidence. Anchored on the ICA review it landed between the ocular
    # review and the decomposition summary, splitting the ICA sections around it.
    move_tagged_content_before(report, tag="run-continuity", anchor=before_raw_sections)


__all__ = [
    "BAD_ANNOTATION_PREFIX",
    "VOLUME_GAP_FACTOR",
    "WINDOW_SECONDS",
    "RunContinuity",
    "add_continuity_section",
    "compute_run_continuity",
    "continuity_html",
    "plot_run_continuity",
]
