"""Report evidence for the BrainVision Analyzer scanner-artifact correction.

Gradient and pulse-artifact correction happen in Analyzer, before anything in this
pipeline runs. Its quality is therefore an input, not something the pipeline controls,
and it is the one thing a reviewer cannot infer from the MNE stages: an uncorrected
pulse artifact simply looks like unusually strong cardiac structure later on.

The measurements already exist as cohort QC tables. This module puts the rows for one
subject in front of the person reviewing that subject, because a table nobody opens is
not quality control.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION
from eeg_pipeline.preprocessing.report.annotations import annotation_onsets
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    PRIMARY_COLOR,
)

PULSE_MARKER_QC_SUFFIX = "desc-pulsemarkers_qc.tsv"
CARDIAC_ATTENUATION_QC_SUFFIX = "desc-cardiacattenuation_qc.tsv"

#: Beats needed before an interval series describes a rhythm rather than a few markers.
MINIMUM_BEATS = 3

#: Multiple of the run's median interval at which a single missed beat lands.
MISSED_BEAT_FACTOR = 1.5


@dataclass(frozen=True)
class AnalyzerCorrectionQc:
    """Per-run Analyzer correction evidence for one subject."""

    subject: str
    runs: pd.DataFrame

    @property
    def failed_runs(self) -> tuple[str, ...]:
        if "status" not in self.runs.columns:
            return ()
        failed = self.runs.loc[self.runs["status"].astype(str) == "fail", "run"]
        return tuple(str(value) for value in failed)

    @property
    def fallback_runs(self) -> tuple[str, ...]:
        if "is_fallback" not in self.runs.columns:
            return ()
        fallback = self.runs.loc[self.runs["is_fallback"].fillna(False).astype(bool), "run"]
        return tuple(str(value) for value in fallback)


def _run_label(recording_id: object) -> str:
    text = str(recording_id)
    marker = "_run-"
    if marker not in text:
        return text
    return text.split(marker)[1].split("_")[0]


def load_analyzer_qc(
    *,
    qc_dir: Path,
    task: str,
    subject: str,
) -> AnalyzerCorrectionQc | None:
    """Join the pulse-marker and cardiac-attenuation QC rows for one subject.

    Returns ``None`` when neither table exists, so a dataset corrected outside Analyzer
    simply has no such section rather than an empty one.
    """
    frames = {}
    for name, suffix in (
        ("markers", PULSE_MARKER_QC_SUFFIX),
        ("attenuation", CARDIAC_ATTENUATION_QC_SUFFIX),
    ):
        path = qc_dir / f"task-{task}_{suffix}"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, sep="\t")
        if "recording_id" not in frame.columns:
            raise ValueError(f"{path} is missing the recording_id column.")
        frame = frame[frame["recording_id"].astype(str).str.startswith(f"sub-{subject}_")]
        if frame.empty:
            continue
        frame = frame.assign(run=frame["recording_id"].map(_run_label)).drop(
            columns=["recording_id"]
        )
        frames[name] = frame
    if not frames:
        return None

    if len(frames) == 2:
        runs = frames["markers"].merge(
            frames["attenuation"],
            on="run",
            how="outer",
            suffixes=("_markers", "_attenuation"),
        )
    else:
        runs = next(iter(frames.values()))
    runs = runs.sort_values("run").reset_index(drop=True)
    if {"before_rms_uv", "after_rms_uv"}.issubset(runs.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            runs["attenuation_db"] = 20.0 * np.log10(runs["before_rms_uv"] / runs["after_rms_uv"])
    return AnalyzerCorrectionQc(subject=subject, runs=runs)


def analyzer_qc_html(qc: AnalyzerCorrectionQc) -> str:
    """Render the per-run Analyzer correction table and any failures."""
    display_columns = [
        column
        for column in (
            "run",
            "status",
            "marker_count",
            "median_bpm",
            "marker_fraction",
            "recording_coverage",
            "before_rms_uv",
            "after_rms_uv",
            "attenuation_db",
            "is_fallback",
        )
        if column in qc.runs.columns
    ]
    table = qc.runs[display_columns].to_html(
        index=False,
        na_rep="—",
        float_format=lambda value: f"{value:.2f}",
        border=0,
        classes="table table-striped table-sm",
    )
    document = (
        "<p>Gradient and pulse-artifact correction were performed in BrainVision "
        "Analyzer before this pipeline ran, so their quality is an input rather than "
        "something the MNE stages can fix. <code>before_rms_uv</code> and "
        "<code>after_rms_uv</code> are the R-locked EEG amplitude either side of the "
        "pipeline's own ICA, so a large <code>before</code> value means residual pulse "
        "artifact reached this pipeline.</p>"
        f"{table}"
    )
    if qc.failed_runs or qc.fallback_runs:
        notes = []
        if qc.failed_runs:
            notes.append(
                f"Analyzer's own pulse-marker check did not pass for run(s) "
                f"{', '.join(qc.failed_runs)}."
            )
        if qc.fallback_runs:
            notes.append(
                f"Run(s) {', '.join(qc.fallback_runs)} carried no Analyzer R markers, so "
                "the R peaks used for this measurement came from automated detection on "
                "the ECG channel. That substitution affects the measurement only, not the "
                "correction Analyzer applied."
            )
        document += "<p>" + " ".join(notes) + "</p>"
    return document


def plot_analyzer_qc(qc: AnalyzerCorrectionQc) -> plt.Figure:
    """Plot residual R-locked amplitude per run, and the attenuation achieved."""
    runs = qc.runs
    if not {"before_rms_uv", "after_rms_uv"}.issubset(runs.columns):
        raise ValueError("Analyzer QC figure requires before and after R-locked amplitudes.")
    labels = [f"run-{value}" for value in runs["run"]]
    positions = np.arange(len(runs))
    fallback = (
        runs["is_fallback"].fillna(False).astype(bool).to_numpy()
        if "is_fallback" in runs.columns
        else np.zeros(len(runs), dtype=bool)
    )

    figure, (amplitude_axis, attenuation_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        layout="constrained",
    )
    # Residual amplitude spans more than an order of magnitude between runs, so the axis
    # is logarithmic and the two states are drawn as paired markers rather than bars.
    for position, before, after in zip(
        positions,
        runs["before_rms_uv"],
        runs["after_rms_uv"],
        strict=True,
    ):
        amplitude_axis.plot([position, position], [before, after], color="0.75", linewidth=1.2)
    amplitude_axis.scatter(
        positions,
        runs["before_rms_uv"],
        color=BEFORE_COLOR,
        s=34,
        label="Reaching this pipeline",
        zorder=3,
    )
    amplitude_axis.scatter(
        positions,
        runs["after_rms_uv"],
        color=AFTER_COLOR,
        s=34,
        label="After pipeline ICA",
        zorder=3,
    )
    amplitude_axis.set(
        title="R-locked EEG amplitude per run",
        ylabel="Median RMS (µV)",
        yscale="log",
        xticks=positions,
        xticklabels=labels,
    )
    amplitude_axis.grid(axis="y", alpha=0.2)
    amplitude_axis.spines[["top", "right"]].set_visible(False)
    amplitude_axis.legend(frameon=False, fontsize=8)

    if "attenuation_db" in runs.columns:
        colors = [FLAG_COLOR if flag else "0.80" for flag in fallback]
        attenuation_axis.bar(positions, runs["attenuation_db"], color=colors)
        attenuation_axis.axhline(
            float(np.nanmedian(runs["attenuation_db"])),
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="median",
        )
        title = "Cardiac attenuation achieved by pipeline ICA"
        if fallback.any():
            title += "\nhighlighted runs used fallback R-peak detection"
        attenuation_axis.set(
            title=title,
            ylabel="Attenuation (dB)",
            xticks=positions,
            xticklabels=labels,
        )
        attenuation_axis.legend(frameon=False, fontsize=8)
        attenuation_axis.grid(axis="y", alpha=0.2)
        attenuation_axis.spines[["top", "right"]].set_visible(False)
    for axis in (amplitude_axis, attenuation_axis):
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.suptitle(
        f"sub-{html.unescape(qc.subject)} · BrainVision Analyzer correction quality",
        fontsize=10,
    )
    plt.close(figure)
    return figure


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
    onsets = annotation_onsets(raw, description or PULSE_MARKER_DESCRIPTION)
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


def plot_rr_intervals(series: Sequence[RrIntervals]) -> plt.Figure:
    """Plot the beat-to-beat interval series for every run."""
    if not series:
        raise ValueError("The tachogram requires at least one run with R markers.")
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(10.0, 1.9 * len(series) + 1.0),
        squeeze=False,
        layout="constrained",
    )
    for axis, run in zip(axes[:, 0], series, strict=True):
        axis.plot(
            run.beat_times_s / 60.0,
            run.intervals_s,
            color=PRIMARY_COLOR,
            linewidth=0.7,
            marker=".",
            markersize=1.6,
        )
        axis.axhline(run.median_interval_s, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
        # The same threshold the dropout count uses, so the figure and the table cannot
        # disagree about which intervals were counted.
        axis.axhline(run.dropout_threshold_s, color=FLAG_COLOR, linestyle=":", linewidth=1.0)
        axis.set(
            title=(
                f"{run.recording_id} · {run.intervals_s.size + 1} beats · "
                f"median {run.median_bpm:.0f} bpm · {run.dropout_count} interval(s) "
                f"above {MISSED_BEAT_FACTOR:g}× the median"
            ),
            ylabel="RR (s)",
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time in run (min)")
    figure.suptitle(
        "Beat-to-beat intervals from the R markers · dashed line is the run median, "
        f"dotted line is {MISSED_BEAT_FACTOR:g}× the median, above which an interval is "
        "counted as a missed beat",
        fontsize=9,
    )
    plt.close(figure)
    return figure


def rr_intervals_html(series: Sequence[RrIntervals]) -> str:
    """Render the per-run beat detection record."""
    rows = "".join(
        f"<tr><td>{html.escape(run.recording_id)}</td>"
        f"<td>{run.intervals_s.size + 1}</td>"
        f"<td>{run.median_bpm:.0f}</td>"
        f"<td>{float(np.percentile(run.intervals_s, 5)):.2f}&ndash;"
        f"{float(np.percentile(run.intervals_s, 95)):.2f}</td>"
        f"<td>{run.dropout_count}</td></tr>"
        for run in series
    )
    return (
        "<p>Pulse-artifact correction can only be as good as the R markers it was "
        "driven by, and a marker count with a median rate cannot show a detector that "
        "worked for part of a run and then lost the trace. These are the intervals "
        "themselves.</p>"
        "<table><thead><tr><th>Run</th><th>Beats</th><th>Median (bpm)</th>"
        "<th>RR 5th&ndash;95th percentile (s)</th>"
        f"<th>Intervals above {MISSED_BEAT_FACTOR:g}× median</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p>A missed beat produces an interval near twice the median, so the last "
        "column separates heart-rate variability from detection dropout. Interpreting "
        "the count is left to the reviewer: a run with genuine arrhythmia and a run "
        "with a failing detector both raise it, and only the ECG trace distinguishes "
        "them.</p>"
    )


def add_rr_interval_section(
    *,
    report: mne.Report,
    series: Sequence[RrIntervals],
    section: str = "Scanner artifact correction (Analyzer)",
) -> None:
    """Append the beat-detection record to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not series:
        raise ValueError("The tachogram requires at least one run with R markers.")
    remove_tagged_content(report, tag="rr-intervals")
    report.add_html(
        html=rr_intervals_html(series),
        title="Beat detection by run",
        section=section,
        tags=("raw", "rr-intervals"),
        replace=True,
    )
    report.add_figure(
        fig=plot_rr_intervals(series),
        title="Beat-to-beat intervals",
        section=section,
        tags=("raw", "rr-intervals"),
        image_format=report_image_format(),
        replace=True,
    )
    move_tagged_content_before(report, tag="rr-intervals", anchor=before_raw_sections)


def add_analyzer_correction_review(
    *,
    report: mne.Report,
    qc_dir: Path,
    task: str,
    subject: str,
    section: str = "Scanner artifact correction (Analyzer)",
) -> AnalyzerCorrectionQc | None:
    """Append Analyzer correction evidence to a subject report, if any exists."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    qc = load_analyzer_qc(qc_dir=qc_dir, task=task, subject=subject)
    if qc is None:
        return None

    remove_tagged_content(report, tag="analyzer-correction")
    report.add_html(
        html=analyzer_qc_html(qc),
        title="Analyzer correction quality by run",
        section=section,
        tags=("raw", "analyzer-correction"),
        replace=True,
    )
    if {"before_rms_uv", "after_rms_uv"}.issubset(qc.runs.columns):
        report.add_figure(
            fig=plot_analyzer_qc(qc),
            title="Residual pulse artifact and attenuation by run",
            section=section,
            tags=("raw", "analyzer-correction"),
            image_format=report_image_format(),
            replace=True,
        )
    # This describes the data entering the pipeline, so it belongs ahead of the raw
    # sections rather than after everything the pipeline then did to it.
    move_tagged_content_before(
        report,
        tag="analyzer-correction",
        anchor=before_raw_sections,
    )
    return qc


__all__ = [
    "AnalyzerCorrectionQc",
    "CARDIAC_ATTENUATION_QC_SUFFIX",
    "PULSE_MARKER_QC_SUFFIX",
    "RrIntervals",
    "add_analyzer_correction_review",
    "add_rr_interval_section",
    "analyzer_qc_html",
    "compute_rr_intervals",
    "load_analyzer_qc",
    "plot_analyzer_qc",
    "plot_rr_intervals",
    "rr_intervals_html",
]
