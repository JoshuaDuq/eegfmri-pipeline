"""Publication rendering for Study 1 primary held-out prediction."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.primary_prediction import (
    PrimaryPredictionSummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

TARGET_COLOR_KEYS = {"NPS": "nps", "SIIPS1": "siips1"}


def build_primary_prediction_figure(
    summary: PrimaryPredictionSummary,
    config: Any,
) -> Figure:
    """Render paired absolute performance and incremental effects."""

    _validate_summary(summary)
    dimensions = require_config_value(
        config,
        "study1.figures.primary_prediction.dimensions_mm",
    )
    colors = require_config_value(config, "study1.figures.validity.colors")
    if not isinstance(colors, Mapping):
        raise ValueError("study1.figures.validity.colors must be a mapping.")

    absolute_limits = _absolute_limits(summary)
    delta_limits = _delta_limits(summary)
    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        outer_grid = figure.add_gridspec(
            1,
            2,
            left=0.075,
            right=0.985,
            bottom=0.21,
            top=0.76,
            wspace=0.30,
        )
        axes = []
        for target_index, target in enumerate(summary.targets):
            target_grid = outer_grid[0, target_index].subgridspec(
                1,
                2,
                width_ratios=(1.05, 0.82),
                wspace=0.45,
            )
            absolute_axis = figure.add_subplot(target_grid[0, 0])
            delta_axis = figure.add_subplot(target_grid[0, 1])
            target_color = str(colors[TARGET_COLOR_KEYS[target]])
            _draw_absolute_axis(
                absolute_axis,
                summary,
                target=target,
                target_color=target_color,
                limits=absolute_limits,
                show_y_axis=target_index == 0,
                config=config,
            )
            _draw_delta_axis(
                delta_axis,
                summary,
                target=target,
                target_color=target_color,
                limits=delta_limits,
                show_y_axis=target_index == 0,
                config=config,
            )
            axes.extend((absolute_axis, delta_axis))
            _add_target_heading(
                figure,
                absolute_axis,
                delta_axis,
                target=target,
                panel_index=target_index,
                n_subjects=_target_subject_count(summary, target),
            )
        figure.legend(
            handles=_legend_handles(config),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=3,
            frameon=False,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.25,
        )
    return figure


def _draw_absolute_axis(
    axis,
    summary: PrimaryPredictionSummary,
    *,
    target: str,
    target_color: str,
    limits: tuple[float, float],
    show_y_axis: bool,
    config: Any,
) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    participants = _target_participants(summary, target)
    participant_color = str(style["participant_color"])
    line_width = float(style["participant_line_width_pt"])
    marker_size = max(float(style["participant_marker_size_pt"]), 2.4)
    alpha = max(float(style["participant_alpha"]), 0.48)

    axis.axhline(
        0.0,
        color="#555555",
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    for participant in participants.itertuples(index=False):
        axis.plot(
            (0.0, 1.0),
            (float(participant.r2_nuisance), float(participant.r2)),
            color=participant_color,
            alpha=alpha,
            linewidth=line_width,
            zorder=1,
        )
    axis.scatter(
        np.zeros(len(participants)),
        participants["r2_nuisance"].to_numpy(dtype=float),
        s=marker_size**2,
        facecolors="white",
        edgecolors=participant_color,
        linewidths=0.55,
        alpha=0.9,
        zorder=2,
    )
    axis.scatter(
        np.ones(len(participants)),
        participants["r2"].to_numpy(dtype=float),
        s=marker_size**2,
        facecolors=target_color,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.9,
        zorder=3,
    )
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(*limits)
    axis.set_xticks((0.0, 1.0), labels=("Nuisance\nonly", "Nuisance\n+ EEG"))
    axis.set_ylabel("Held-out R²" if show_y_axis else "")
    axis.tick_params(axis="y", labelleft=show_y_axis)
    _format_axis(axis)


def _draw_delta_axis(
    axis,
    summary: PrimaryPredictionSummary,
    *,
    target: str,
    target_color: str,
    limits: tuple[float, float],
    show_y_axis: bool,
    config: Any,
) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    participants = _target_participants(summary, target)
    cohort = summary.cohort_performance.loc[summary.cohort_performance["target"].eq(target)].iloc[0]
    participant_x = -0.045 + _deterministic_jitter(len(participants), half_width=0.065)
    cohort_x = 0.28
    marker_size = max(float(style["participant_marker_size_pt"]), 2.4)

    axis.axhline(
        0.0,
        color="#555555",
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    axis.scatter(
        participant_x,
        participants["delta_r2"].to_numpy(dtype=float),
        s=marker_size**2,
        color=str(style["participant_color"]),
        alpha=max(float(style["participant_alpha"]), 0.52),
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )
    mean = float(cohort["mean_delta_r2"])
    axis.errorbar(
        cohort_x,
        mean,
        yerr=np.asarray(
            [
                [mean - float(cohort["ci_low_delta_r2"])],
                [float(cohort["ci_high_delta_r2"]) - mean],
            ]
        ),
        color=target_color,
        linestyle="none",
        elinewidth=float(style["confidence_line_width_pt"]),
        marker="D",
        markersize=float(style["cohort_marker_size_pt"]),
        markerfacecolor="white",
        markeredgecolor=target_color,
        markeredgewidth=float(style["confidence_line_width_pt"]),
        capsize=2.0,
        zorder=3,
    )
    axis.set_xlim(-0.22, 0.45)
    axis.set_ylim(*limits)
    axis.set_xticks((-0.045, cohort_x), labels=("Participants", "Mean"))
    axis.set_ylabel("Incremental prediction, ΔR²" if show_y_axis else "")
    axis.tick_params(axis="y", labelleft=show_y_axis)
    _format_axis(axis)


def _format_axis(axis) -> None:
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out")


def _absolute_limits(summary: PrimaryPredictionSummary) -> tuple[float, float]:
    values = np.concatenate(
        (
            summary.participant_performance[["r2_nuisance", "r2"]].to_numpy(dtype=float).ravel(),
            summary.cohort_performance[["ci_low_r2", "ci_high_r2"]].to_numpy(dtype=float).ravel(),
            np.asarray((0.0,)),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Primary-prediction figure received non-finite R² values.")
    low = float(values.min())
    high = float(values.max())
    padding = max(0.08 * (high - low), 0.05)
    return low - padding, high + padding


def _delta_limits(summary: PrimaryPredictionSummary) -> tuple[float, float]:
    values = np.concatenate(
        (
            summary.participant_performance["delta_r2"].to_numpy(dtype=float),
            summary.cohort_performance[["ci_low_delta_r2", "ci_high_delta_r2"]]
            .to_numpy(dtype=float)
            .ravel(),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Primary-prediction figure received non-finite ΔR² values.")
    half_range = max(1.12 * float(np.max(np.abs(values))), 0.05)
    return -half_range, half_range


def _target_participants(
    summary: PrimaryPredictionSummary,
    target: str,
):
    return summary.participant_performance.loc[
        summary.participant_performance["target"].eq(target)
    ].sort_values("subject_id", kind="stable")


def _target_subject_count(summary: PrimaryPredictionSummary, target: str) -> int:
    return len(_target_participants(summary, target))


def _deterministic_jitter(count: int, *, half_width: float) -> np.ndarray:
    if count <= 0:
        raise ValueError("Participant jitter requires at least one observation.")
    if count == 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-half_width, half_width, count)


def _validate_summary(summary: PrimaryPredictionSummary) -> None:
    if summary.targets != ("NPS", "SIIPS1"):
        raise ValueError("Primary-prediction figure requires targets ('NPS', 'SIIPS1').")
    participant_columns = {
        "target",
        "subject_id",
        "r2_nuisance",
        "r2",
        "delta_r2",
    }
    cohort_columns = {
        "target",
        "mean_delta_r2",
        "ci_low_r2",
        "ci_high_r2",
        "ci_low_delta_r2",
        "ci_high_delta_r2",
        "n_subjects",
    }
    missing_participant = sorted(
        participant_columns.difference(summary.participant_performance.columns)
    )
    missing_cohort = sorted(cohort_columns.difference(summary.cohort_performance.columns))
    if missing_participant or missing_cohort:
        raise ValueError(
            "Primary-prediction summary is missing plot columns: "
            f"participant={missing_participant}, cohort={missing_cohort}."
        )
    if summary.cohort_performance["target"].tolist() != list(summary.targets):
        raise ValueError("Primary-prediction cohort targets are missing, duplicated, or unordered.")
    for target in summary.targets:
        participants = _target_participants(summary, target)
        if participants.empty:
            raise ValueError(f"Primary-prediction target {target} has no participants.")
        if participants["subject_id"].duplicated().any():
            raise ValueError(f"Primary-prediction target {target} has duplicate participants.")
        cohort = summary.cohort_performance.loc[
            summary.cohort_performance["target"].eq(target)
        ].iloc[0]
        if len(participants) != int(cohort["n_subjects"]):
            raise ValueError(f"Primary-prediction subject count disagrees for {target}.")
        participant_values = participants[["r2_nuisance", "r2", "delta_r2"]].to_numpy(dtype=float)
        if not np.isfinite(participant_values).all():
            raise ValueError(f"Primary-prediction target {target} has non-finite participants.")


def _add_target_heading(
    figure: Figure,
    absolute_axis,
    delta_axis,
    *,
    target: str,
    panel_index: int,
    n_subjects: int,
) -> None:
    left = absolute_axis.get_position().x0
    right = delta_axis.get_position().x1
    center = (left + right) / 2.0
    figure.text(
        center,
        0.825,
        target,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=7.0,
    )
    figure.text(
        center,
        0.79,
        f"n = {n_subjects}",
        ha="center",
        va="center",
        color="#333333",
        fontsize=5.5,
    )
    figure.text(
        left - 0.035,
        0.82,
        chr(ord("a") + panel_index),
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=8.0,
    )


def _legend_handles(config: Any) -> tuple[Line2D, Line2D, Line2D]:
    style = require_config_value(config, "study1.figures.validity.style")
    marker_size = max(float(style["participant_marker_size_pt"]), 2.4)
    participant_color = str(style["participant_color"])
    return (
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=participant_color,
            markeredgewidth=0.55,
            label="Nuisance only",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=marker_size,
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="Nuisance + EEG",
        ),
        Line2D(
            [],
            [],
            color="#333333",
            linewidth=float(style["confidence_line_width_pt"]),
            marker="D",
            markersize=float(style["cohort_marker_size_pt"]),
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=float(style["confidence_line_width_pt"]),
            label="Mean ΔR² ± 95% CI",
        ),
    )


__all__ = ["build_primary_prediction_figure"]
