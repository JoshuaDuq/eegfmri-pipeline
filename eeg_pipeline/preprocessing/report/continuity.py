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

import html
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
    robust_symmetric_limit,
)

#: Window over which amplitude is pooled. Short enough to localise a movement to a few
#: seconds, long enough that the RMS of a single window is a stable estimate.
WINDOW_SECONDS = 1.0

#: Multiple of the median volume interval above which a gap in the marker train is
#: treated as an interruption rather than jitter.
VOLUME_GAP_FACTOR = 1.5

#: Annotation prefix MNE uses for spans excluded from processing.
BAD_ANNOTATION_PREFIX = "BAD"


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
    def worst_window_s(self) -> float:
        return float(self.times_s[int(np.argmax(self.excursion_db))])

    @property
    def worst_excursion_db(self) -> float:
        return float(np.max(self.excursion_db))


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
    )


def continuity_html(runs: Sequence[RunContinuity]) -> str:
    """Render when each run departed from its own baseline."""
    if not runs:
        raise ValueError("The continuity summary requires at least one run.")
    rows = "".join(
        f"<tr><td>{html.escape(run.recording_id)}</td>"
        f"<td>{run.duration_s / 60.0:.1f}</td>"
        f"<td>{run.bad_fraction:.1%}</td>"
        f"<td>{len(run.volume_gaps)}</td>"
        f"<td>{run.worst_excursion_db:+.1f} dB at {run.worst_window_s / 60.0:.1f} min</td></tr>"
        for run in runs
    )
    return (
        "<p>Amplitude in "
        f"{runs[0].window_seconds:g} s windows, expressed relative to each channel's own "
        "median over the run. A run-level average cannot distinguish a uniformly "
        "marginal recording from a good one with a bad stretch in it, and inside a "
        "scanner the second is the common case: once the participant shifts, the "
        "gradient template stops matching and everything after that moment is "
        "contaminated.</p>"
        "<table><thead><tr><th>Run</th><th>Duration (min)</th>"
        "<th>Marked bad</th><th>Volume-marker gaps</th>"
        "<th>Largest excursion</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p>The excursion is the across-channel median, so it responds to the whole "
        "montage moving together rather than to one sensor misbehaving. A gap in the "
        "volume-marker train means the scanner stopped or markers were lost, which "
        "leaves the correction either side of it built under different conditions.</p>"
    )


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
    step = max(1, len(run.channel_names) // 30)
    map_axis.set(
        title=(
            f"{run.recording_id} · {run.duration_s / 60.0:.1f} min · "
            f"{run.bad_fraction:.1%} marked bad"
        ),
        ylabel="EEG channel",
        yticks=np.arange(0, len(run.channel_names), step),
        yticklabels=run.channel_names[::step],
    )
    map_axis.tick_params(axis="y", labelsize=5)

    trace_axis.plot(minutes, run.excursion_db, color="0.30", linewidth=0.9)
    trace_axis.axhline(0.0, color="black", linewidth=0.8)
    trace_axis.set(
        xlabel="Time in run (min)",
        ylabel="Median across\nchannels (dB)",
    )
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
        map_axis.annotate(
            " · ".join(labels),
            xy=(0.5, 1.01),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.30",
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
        before_ica_component_review,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not runs:
        raise ValueError("Time-resolved quality requires at least one run.")
    remove_tagged_content(report, tag="run-continuity")
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
    move_tagged_content_before(
        report,
        tag="run-continuity",
        anchor=before_ica_component_review,
    )


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
