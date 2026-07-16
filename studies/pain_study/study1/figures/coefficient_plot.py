"""Publication rendering for Study 1 behavioral-validity coefficients."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.figure_style import outside_top_legend
from studies.pain_study.study1.figures.behavioral_validity import (
    BehavioralValiditySummary,
)
from studies.pain_study.study1.figures.validity_style import (
    configured_figure_size,
    publication_style,
)

TERM_ROWS = (
    ("within_scale_intensity", "Within-scale intensity", "within_scale_intensity_beta", 0.0),
    ("painful_report", "Painful report", "painful_report_beta", 1.0),
)


def build_behavioral_validity_figure(
    summary: BehavioralValiditySummary,
    *,
    color_config_key: str,
    config: Any,
) -> Figure:
    colors = require_config_value(config, "study1.figures.validity.colors")
    if color_config_key not in colors:
        raise ValueError(f"Unknown Study 1 validity color key: {color_config_key!r}.")
    target_color = str(colors[color_config_key])

    with publication_style(config):
        figure, axis = plt.subplots(
            figsize=configured_figure_size(config),
            layout="constrained",
        )
        _draw_zero_reference(axis)
        _draw_participants(axis, summary, config)
        _draw_cohort(axis, summary, target_color, config)
        _format_axis(axis, summary)
        outside_top_legend(figure, axis)
    return figure


def _draw_zero_reference(axis) -> None:
    axis.axvline(
        0.0,
        color="#555555",
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        zorder=0,
    )


def _draw_participants(axis, summary: BehavioralValiditySummary, config: Any) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    participants = _estimable_participants(summary)
    offsets = np.linspace(-0.10, 0.10, len(participants))
    for term_index, (_, _, coefficient_column, row) in enumerate(TERM_ROWS):
        axis.scatter(
            participants[coefficient_column].to_numpy(dtype=float),
            row + offsets,
            s=float(style["participant_marker_size_pt"]) ** 2,
            color=str(style["participant_color"]),
            alpha=float(style["participant_alpha"]),
            edgecolors="none",
            label="Participants" if term_index == 0 else None,
            zorder=1,
        )


def _draw_cohort(
    axis,
    summary: BehavioralValiditySummary,
    color: str,
    config: Any,
) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    font = require_config_value(config, "study1.figures.validity.font")
    cohort = summary.cohort_estimates.set_index("term")
    for term_index, (term, _, _, row) in enumerate(TERM_ROWS):
        estimate = cohort.loc[term]
        mean = float(estimate["mean"])
        error = np.array(
            [
                [mean - float(estimate["ci_low"])],
                [float(estimate["ci_high"]) - mean],
            ]
        )
        axis.errorbar(
            mean,
            row,
            xerr=error,
            color=color,
            linestyle="none",
            elinewidth=float(style["confidence_line_width_pt"]),
            marker="D",
            markersize=float(style["cohort_marker_size_pt"]),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=float(style["confidence_line_width_pt"]),
            capsize=2.0,
            label="Mean ± 95% CI" if term_index == 0 else None,
            zorder=3,
        )
        axis.annotate(
            f"n = {int(estimate['n_subjects'])}",
            xy=(0.99, row + 0.18),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="center",
            fontsize=float(font["annotation_pt"]),
            color="#222222",
        )


def _format_axis(axis, summary: BehavioralValiditySummary) -> None:
    axis.set_xlabel("Standardized partial coefficient (β)")
    axis.set_yticks(
        [row for _, _, _, row in TERM_ROWS],
        labels=[label for _, label, _, _ in TERM_ROWS],
    )
    axis.set_ylim(-0.35, 1.35)
    half_range = _symmetric_half_range(summary)
    axis.set_xlim(-half_range, half_range)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")


def _estimable_participants(summary: BehavioralValiditySummary):
    participants = summary.participant_models.loc[
        summary.participant_models["estimable"]
    ].sort_values("subject_id", kind="stable")
    if participants.empty:
        raise ValueError(f"{summary.target} coefficient plot requires estimable participants.")
    columns = ["painful_report_beta", "within_scale_intensity_beta"]
    values = participants[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{summary.target} coefficient plot received non-finite participants.")
    return participants


def _symmetric_half_range(summary: BehavioralValiditySummary) -> float:
    participants = _estimable_participants(summary)
    cohort = summary.cohort_estimates
    values = np.concatenate(
        (
            participants[
                ["painful_report_beta", "within_scale_intensity_beta"]
            ].to_numpy(dtype=float).ravel(),
            cohort[["ci_low", "ci_high"]].to_numpy(dtype=float).ravel(),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{summary.target} coefficient plot received non-finite estimates.")
    return max(1.08 * float(np.max(np.abs(values))), 0.10)


__all__ = ["build_behavioral_validity_figure"]
