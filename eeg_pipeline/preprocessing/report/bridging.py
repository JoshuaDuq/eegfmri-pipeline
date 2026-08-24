"""Electrode-bridge diagnostic following MNE's EEG QC example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.organize import remove_tagged_content
from eeg_pipeline.preprocessing.report.style import report_image_format, run_label
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

BRIDGE_DIAGNOSTIC_DURATION_S = 180.0


@dataclass(frozen=True)
class BridgingReview:
    recording_id: str
    duration_s: float
    requested_duration_s: float
    pair_indices: tuple[tuple[int, int], ...]
    pair_names: tuple[tuple[str, str], ...]
    previously_bad_channels: tuple[str, ...]
    ed_matrix: np.ndarray
    info: mne.Info


def compute_bridging_review(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    duration_seconds: float = BRIDGE_DIAGNOSTIC_DURATION_S,
) -> BridgingReview:
    """Measure bridges on the final diagnostic segment, including marked EEG sites."""
    if duration_seconds <= 0:
        raise ValueError("Bridge diagnostic duration must be positive.")
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) < 2:
        raise ValueError("Electrode-bridge diagnostic requires at least two EEG channels.")
    diagnostic = raw.copy().pick(eeg_picks)
    previously_bad = tuple(
        channel for channel in diagnostic.ch_names if channel in raw.info.get("bads", [])
    )
    sample_count = min(
        diagnostic.n_times,
        max(1, int(round(duration_seconds * diagnostic.info["sfreq"]))),
    )
    first_sample = diagnostic.n_times - sample_count
    diagnostic.crop(
        tmin=float(diagnostic.times[first_sample]),
        tmax=float(diagnostic.times[-1]),
    ).load_data()
    # Marked sites are exactly where a bridge may be hiding. MNE's detector otherwise
    # excludes bads from its channel picks, so clear the copied metadata for this
    # diagnostic only; the source raw and all preprocessing decisions remain unchanged.
    diagnostic.info["bads"] = []
    pair_indices, ed_matrix = mne.preprocessing.compute_bridged_electrodes(diagnostic)
    pair_indices = tuple((int(left), int(right)) for left, right in pair_indices)
    return BridgingReview(
        recording_id=str(recording_id),
        duration_s=float(diagnostic.n_times / diagnostic.info["sfreq"]),
        requested_duration_s=float(duration_seconds),
        pair_indices=pair_indices,
        pair_names=tuple(
            (diagnostic.ch_names[left], diagnostic.ch_names[right]) for left, right in pair_indices
        ),
        previously_bad_channels=previously_bad,
        ed_matrix=np.asarray(ed_matrix, dtype=float),
        info=diagnostic.info.copy(),
    )


def bridging_html(reviews: Sequence[BridgingReview]) -> str:
    """Describe the scope and candidates for every screened run."""
    if not reviews:
        raise ValueError("Electrode-bridge review requires at least one run.")
    requested_durations = {review.requested_duration_s for review in reviews}
    if len(requested_durations) != 1:
        raise ValueError("Bridge review runs must use one diagnostic duration.")
    requested_duration = next(iter(requested_durations))
    previous_bads = sorted(
        {channel for review in reviews for channel in review.previously_bad_channels}
    )
    rows = [
        [
            run_label(review.recording_id),
            f"{review.duration_s:.1f}",
            len(review.pair_names),
            ", ".join(f"{left}–{right}" for left, right in review.pair_names) or "none detected",
        ]
        for review in reviews
    ]
    return (
        "<p>MNE's electrical-distance diagnostic was run on the final segment of each run "
        f"(up to {requested_duration:g} s, fixed by configuration). This screens every "
        "acquisition while keeping the calculation "
        "bounded; it is not proof that a transient bridge never occurred earlier in a "
        "run. Previously marked bad EEG sites were included"
        + (f": {', '.join(previous_bads)}. " if previous_bads else ": none. ")
        + "No channel status or "
        "interpolation decision was changed.</p>"
        + grid_table(
            (
                Column("Run", align=Align.TEXT),
                Column("Screened (s)"),
                Column("Candidate pairs"),
                Column("Pair names", align=Align.TEXT),
            ),
            rows,
        )
    )


def add_bridging_reviews(
    *,
    report: mne.Report,
    reviews: Sequence[BridgingReview],
    section: str = "Electrode bridging",
) -> None:
    """Append per-run bridge candidates and MNE diagnostic plots."""
    if not reviews:
        return
    figures = [
        mne.viz.plot_bridged_electrodes(
            review.info,
            review.pair_indices,
            review.ed_matrix,
            title=f"Electrode bridge diagnostic · {run_label(review.recording_id)}",
        )
        for review in reviews
    ]
    remove_tagged_content(report, tag="electrode-bridging")
    report.add_html(
        html=bridging_html(reviews),
        title="Bridge candidates and diagnostic scope",
        section=section,
        tags=("raw", "electrode-bridging"),
        replace=True,
    )
    report.add_figure(
        fig=figures,
        title="Electrical distance and candidate bridged electrodes",
        caption=[review.recording_id for review in reviews],
        section=section,
        tags=("raw", "electrode-bridging"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    for figure in figures:
        plt.close(figure)


__all__ = [
    "BRIDGE_DIAGNOSTIC_DURATION_S",
    "BridgingReview",
    "add_bridging_reviews",
    "bridging_html",
    "compute_bridging_review",
]
