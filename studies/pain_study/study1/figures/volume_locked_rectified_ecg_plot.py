"""Plot native-grid volume-locked rectified ECG traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)
from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
    VolumeLockedEcgParticipant,
)

STAGES = ("raw", "processed")
STAGE_TITLES = {
    "raw": "Original BrainVision · 5 kHz",
    "processed": "BrainVision corrected · 1 kHz",
}


def build_volume_locked_ecg_figure(
    participants: tuple[VolumeLockedEcgParticipant, ...],
    peaks: pd.DataFrame,
    troughs: pd.DataFrame,
    config: Any,
) -> Figure:
    """Render one participant row and independent native-scale stage columns."""
    participant_index, subjects = _index_participants(participants)
    figure_config = require_config_value(
        config,
        "study1.figures.volume_locked_rectified_ecg",
    )
    colors = figure_config["colors"]
    if not isinstance(colors, Mapping) or set(colors) != set(STAGES):
        raise ValueError("Volume-locked ECG colors must define raw and processed stages.")
    dimensions = figure_config["dimensions_mm"]
    epoch_duration_ms = float(figure_config["epoch_duration_s"]) * 1_000.0
    style = require_config_value(config, "study1.figures.validity.style")

    with publication_style(config):
        figure, axes = plt.subplots(
            len(subjects),
            len(STAGES),
            figsize=figure_size_inches(dimensions),
            sharex=True,
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.10,
            right=0.985,
            bottom=0.12,
            top=0.78,
            hspace=0.32,
            wspace=0.23,
        )
        for row, subject_id in enumerate(subjects):
            for column, stage in enumerate(STAGES):
                axis = axes[row, column]
                participant = participant_index[(subject_id, stage)]
                color = str(colors[stage])
                for run in participant.runs:
                    axis.plot(
                        run.times_ms,
                        run.mean_rectified_ecg_uv,
                        color=color,
                        alpha=0.22,
                        linewidth=float(style["participant_line_width_pt"]),
                        zorder=1,
                    )
                axis.plot(
                    participant.times_ms,
                    participant.mean_rectified_ecg_uv,
                    color=color,
                    linewidth=float(style["cohort_line_width_pt"]),
                    label="Equal-run participant mean",
                    zorder=3,
                )
                _draw_peaks(axis, peaks, participant, color)
                _draw_troughs(axis, troughs, participant, color)
                axis.axvline(0.0, color="#333333", linewidth=0.6, linestyle=":", zorder=0)
                axis.set_xlim(0.0, epoch_duration_ms)
                axis.margins(y=0.08)
                axis.set_ylabel(f"{subject_id}\nMean |ECG| (µV)")
                if row == 0:
                    axis.set_title(STAGE_TITLES[stage], fontweight="bold", pad=5.0)
                if row == len(subjects) - 1:
                    axis.set_xlabel("Time from Volume/V  1 (ms)")

        figure.text(
            0.10,
            0.965,
            "Volume-marker-locked rectified ECG",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )
        figure.text(
            0.10,
            0.925,
            "Rectification precedes volume averaging · six faint run means · "
            "bold equal-run participant mean",
            ha="left",
            va="top",
            fontsize=6.0,
        )
        figure.text(
            0.10,
            0.89,
            "Separate native sampling grids and µV scales; markers identify stable magnitude "
            "peaks and Analyzer-display troughs",
            ha="left",
            va="top",
            fontsize=6.0,
        )
        figure.legend(
            handles=(
                Line2D([], [], color="#777777", alpha=0.35, linewidth=0.7, label="Run mean"),
                Line2D([], [], color="#222222", linewidth=1.2, label="Participant mean"),
                Line2D(
                    [],
                    [],
                    color="#222222",
                    marker="o",
                    linestyle="none",
                    markersize=3.0,
                    label="Stable magnitude peak",
                ),
                Line2D(
                    [],
                    [],
                    color="#222222",
                    marker="v",
                    linestyle="none",
                    markersize=3.0,
                    label="Analyzer-display trough",
                ),
            ),
            loc="upper right",
            bbox_to_anchor=(0.985, 0.965),
            frameon=False,
            ncol=1,
            handlelength=1.5,
        )
    return figure


def _index_participants(
    participants: tuple[VolumeLockedEcgParticipant, ...],
) -> tuple[dict[tuple[str, str], VolumeLockedEcgParticipant], tuple[str, ...]]:
    if not participants:
        raise ValueError("At least one participant ECG trace is required.")
    indexed = {(item.subject_id, item.stage): item for item in participants}
    if len(indexed) != len(participants):
        raise ValueError("Duplicate participant-stage ECG trace.")
    subjects = tuple(sorted({item.subject_id for item in participants}))
    expected = {(subject, stage) for subject in subjects for stage in STAGES}
    if set(indexed) != expected:
        raise ValueError("Each participant requires raw and processed ECG traces.")
    return indexed, subjects


def _draw_peaks(
    axis,
    peaks: pd.DataFrame,
    participant: VolumeLockedEcgParticipant,
    color: str,
) -> None:
    _draw_extrema(
        axis,
        peaks,
        participant,
        color,
        identifier_column="peak_id",
        marker="o",
        draw_timing_line=False,
    )


def _draw_troughs(
    axis,
    troughs: pd.DataFrame,
    participant: VolumeLockedEcgParticipant,
    color: str,
) -> None:
    _draw_extrema(
        axis,
        troughs,
        participant,
        color,
        identifier_column="trough_id",
        marker="v",
        draw_timing_line=True,
    )


def _draw_extrema(
    axis,
    extrema: pd.DataFrame,
    participant: VolumeLockedEcgParticipant,
    color: str,
    *,
    identifier_column: str,
    marker: str,
    draw_timing_line: bool,
) -> None:
    required = {
        "subject_id",
        "stage",
        identifier_column,
        "latency_ms",
        "mean_rectified_ecg_uv",
    }
    if not required.issubset(extrema.columns):
        missing = ", ".join(sorted(required.difference(extrema.columns)))
        raise ValueError(f"Artifact-extremum table is missing columns: {missing}.")
    selected = extrema.loc[
        (extrema["subject_id"] == participant.subject_id) & (extrema["stage"] == participant.stage)
    ]
    for extremum in selected.itertuples(index=False):
        latency_ms = float(extremum.latency_ms)
        amplitude_uv = float(extremum.mean_rectified_ecg_uv)
        if draw_timing_line:
            axis.axvline(latency_ms, color=color, alpha=0.18, linewidth=0.5, zorder=0)
        axis.plot(
            latency_ms,
            amplitude_uv,
            marker=marker,
            markersize=2.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.6,
            linestyle="none",
            zorder=4,
        )


__all__ = ["build_volume_locked_ecg_figure"]
