"""Continuous muscle-artifact screening with MNE's z-score detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.organize import remove_tagged_content
from eeg_pipeline.preprocessing.report.style import (
    FLAG_COLOR,
    GUIDE_COLOR,
    PRIMARY_COLOR,
    report_image_format,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

_MAX_DISPLAY_POINTS = 6_000


@dataclass(frozen=True)
class MuscleReview:
    """MNE muscle scores and thresholded spans for one continuous run."""

    recording_id: str
    times_s: np.ndarray
    scores: np.ndarray
    spans_s: tuple[tuple[float, float], ...]
    duration_s: float
    filter_freq_hz: tuple[float, float]
    threshold: float
    min_length_good_s: float
    excluded_bad_channels: tuple[str, ...]

    @property
    def artifact_duration_s(self) -> float:
        return float(sum(stop - start for start, stop in self.spans_s))

    @property
    def artifact_fraction(self) -> float:
        return self.artifact_duration_s / self.duration_s if self.duration_s else 0.0


def compute_muscle_review(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    filter_freq_hz: tuple[float, float],
    threshold: float,
    min_length_good_s: float,
) -> MuscleReview:
    """Screen one run without attaching the returned annotations to the recording."""
    low_hz, high_hz = map(float, filter_freq_hz)
    if not 0.0 < low_hz < high_hz:
        raise ValueError("Muscle screening band must satisfy 0 < low < high.")
    if threshold <= 0:
        raise ValueError("Muscle screening threshold must be positive.")
    if min_length_good_s < 0:
        raise ValueError("Muscle screening minimum good duration must be non-negative.")
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if not eeg_picks.size:
        raise ValueError("Muscle screening requires at least one EEG channel.")
    good_eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if not good_eeg_picks.size:
        raise ValueError("Muscle screening requires at least one non-bad EEG channel.")
    bad_eeg_channels = tuple(
        raw.ch_names[index] for index in eeg_picks if raw.ch_names[index] in raw.info["bads"]
    )
    nyquist_hz = float(raw.info["sfreq"]) / 2.0
    recorded_lowpass_hz = min(float(raw.info["lowpass"]), nyquist_hz)
    if high_hz >= recorded_lowpass_hz:
        raise ValueError(
            f"{recording_id}: muscle screening band {low_hz:g}–{high_hz:g} Hz must "
            f"end below the recorded low-pass/Nyquist limit ({recorded_lowpass_hz:g} Hz)."
        )

    annotations, scores = mne.preprocessing.annotate_muscle_zscore(
        raw.copy().pick(good_eeg_picks),
        ch_type="eeg",
        threshold=float(threshold),
        min_length_good=float(min_length_good_s),
        filter_freq=(low_hz, high_hz),
        n_jobs=1,
        verbose="ERROR",
    )
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (raw.n_times,):
        raise ValueError("MNE muscle scores must contain one value per raw sample.")
    spans = tuple(
        (float(onset), float(onset + duration))
        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        )
        if description == "BAD_muscle"
    )
    return MuscleReview(
        recording_id=str(recording_id),
        times_s=np.asarray(raw.times, dtype=float),
        scores=scores,
        spans_s=spans,
        duration_s=float(raw.n_times / raw.info["sfreq"]),
        filter_freq_hz=(low_hz, high_hz),
        threshold=float(threshold),
        min_length_good_s=float(min_length_good_s),
        excluded_bad_channels=bad_eeg_channels,
    )


def muscle_review_html(reviews: Sequence[MuscleReview]) -> str:
    """Describe the diagnostic scope and summarize every run."""
    if not reviews:
        raise ValueError("Muscle artifact review requires at least one run.")
    bands = {review.filter_freq_hz for review in reviews}
    thresholds = {review.threshold for review in reviews}
    minimum_lengths = {review.min_length_good_s for review in reviews}
    if len(bands) != 1 or len(thresholds) != 1 or len(minimum_lengths) != 1:
        raise ValueError("Muscle review runs must use one common detection method.")
    low_hz, high_hz = next(iter(bands))
    threshold = next(iter(thresholds))
    minimum_length = next(iter(minimum_lengths))
    excluded_rows = [
        f"{run_label(review.recording_id)} — {', '.join(review.excluded_bad_channels)}"
        for review in reviews
        if review.excluded_bad_channels
    ]
    excluded_html = ""
    if excluded_rows:
        excluded_html = (
            "<p>Channels already marked bad were excluded before the cross-channel "
            "muscle score was computed: " + "; ".join(excluded_rows) + ".</p>"
        )

    rows = [
        [
            run_label(review.recording_id),
            f"{review.artifact_duration_s:.1f}",
            f"{100.0 * review.artifact_fraction:.2f}",
        ]
        for review in reviews
    ]
    return (
        "<p><strong>Diagnostic only:</strong> MNE's continuous EEG muscle score was "
        f"computed in the {low_hz:g}–{high_hz:g} Hz band with a z-score threshold of "
        f"{threshold:g}. Good gaps shorter than {minimum_length:g} s were merged into "
        "adjacent candidate spans. These <code>BAD_muscle</code> annotations were not "
        "added to the data and did not reject samples or epochs; they expose periods a "
        "reviewer should compare with the raw trace and ICA components.</p>"
        + excluded_html
        + grid_table(
            (
                Column("Run", align=Align.TEXT),
                Column("Candidate time (s)"),
                Column("Recording flagged (%)"),
            ),
            rows,
        )
    )


def _display_score(review: MuscleReview) -> tuple[np.ndarray, np.ndarray]:
    """Return a maximum-preserving trace small enough for an embedded report."""
    if review.scores.size <= _MAX_DISPLAY_POINTS:
        return review.times_s, review.scores
    edges = np.linspace(0, review.scores.size, _MAX_DISPLAY_POINTS + 1, dtype=int)
    times = np.empty(_MAX_DISPLAY_POINTS, dtype=float)
    scores = np.empty(_MAX_DISPLAY_POINTS, dtype=float)
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        block = review.scores[start:stop]
        times[index] = review.times_s[(start + stop - 1) // 2]
        scores[index] = np.nanmax(block) if np.isfinite(block).any() else np.nan
    return times, scores


def plot_muscle_review(reviews: Sequence[MuscleReview]) -> plt.Figure:
    """Plot the score time course and candidate spans for each run."""
    if not reviews:
        raise ValueError("Muscle artifact review requires at least one run.")
    figure, axes = plt.subplots(
        len(reviews),
        1,
        figsize=(10.0, max(2.5, 2.1 * len(reviews))),
        squeeze=False,
        layout="constrained",
    )
    for axis, review in zip(axes[:, 0], reviews):
        times_s, scores = _display_score(review)
        axis.plot(times_s / 60.0, scores, color=PRIMARY_COLOR, linewidth=0.8)
        axis.axhline(
            review.threshold,
            color=FLAG_COLOR,
            linestyle="--",
            linewidth=0.9,
            label=f"Threshold ({review.threshold:g})",
        )
        for start_s, stop_s in review.spans_s:
            axis.axvspan(
                start_s / 60.0,
                stop_s / 60.0,
                color=FLAG_COLOR,
                alpha=0.16,
                linewidth=0,
            )
        axis.set(
            title=run_label(review.recording_id),
            ylabel="Muscle z-score",
            xlabel="Time (min)",
        )
        axis.axhline(0.0, color=GUIDE_COLOR, linewidth=0.5)
        axis.legend(frameon=False, fontsize=7, loc="upper right")
    figure.suptitle(
        "Continuous high-frequency muscle screening · per-bin maximum at display resolution"
    )
    return figure


def add_muscle_review(
    *,
    report: mne.Report,
    reviews: Sequence[MuscleReview],
    section: str = "Muscle artifact screening",
) -> None:
    """Append the muscle-artifact method, summary, and complete score trace."""
    if not reviews:
        return
    figure = plot_muscle_review(reviews)
    remove_tagged_content(report, tag="muscle-artifact-screening")
    report.add_html(
        html=muscle_review_html(reviews),
        title="Candidate muscle periods and diagnostic scope",
        section=section,
        tags=("raw", "muscle-artifact-screening"),
        replace=True,
    )
    report.add_figure(
        fig=figure,
        title="MNE muscle z-score over time",
        section=section,
        tags=("raw", "muscle-artifact-screening"),
        image_format=report_image_format(),
        replace=True,
    )
    plt.close(figure)


__all__ = [
    "MuscleReview",
    "add_muscle_review",
    "compute_muscle_review",
    "muscle_review_html",
    "plot_muscle_review",
]
