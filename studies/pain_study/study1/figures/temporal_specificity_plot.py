"""Publication rendering for Study 1 temporal-specificity results."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.temporal_specificity import (
    TemporalSpecificitySummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

TARGET_COLOR_KEYS = {"NPS": "nps", "SIIPS1": "siips1"}


def build_temporal_specificity_figure(
    summary: TemporalSpecificitySummary,
    config: Any,
) -> Figure:
    """Render held-out participant effects and cohort bootstrap intervals."""

    _validate_summary(summary)
    dimensions = require_config_value(
        config,
        "study1.figures.temporal_specificity.dimensions_mm",
    )
    colors = require_config_value(config, "study1.figures.validity.colors")
    if not isinstance(colors, Mapping):
        raise ValueError("study1.figures.validity.colors must be a mapping.")

    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            1,
            2,
            left=0.225,
            right=0.965,
            bottom=0.225,
            top=0.79,
            wspace=0.43,
        )
        axes = (
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
        )
        half_range = _symmetric_half_range(summary)
        for panel_index, (axis, target) in enumerate(zip(axes, summary.targets, strict=True)):
            color_key = TARGET_COLOR_KEYS[target]
            if color_key not in colors:
                raise ValueError(f"Missing validity color for target {target!r}.")
            _draw_target_panel(
                axis,
                summary,
                target=target,
                target_color=str(colors[color_key]),
                half_range=half_range,
                show_window_labels=panel_index == 0,
                config=config,
            )
            _add_panel_label(axis, panel_index)
        figure.legend(
            handles=_legend_handles(config),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.55,
            columnspacing=1.5,
        )
    return figure


def _draw_target_panel(
    axis,
    summary: TemporalSpecificitySummary,
    *,
    target: str,
    target_color: str,
    half_range: float,
    show_window_labels: bool,
    config: Any,
) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    font = require_config_value(config, "study1.figures.validity.font")
    participant_color = str(style["participant_color"])
    participant_effects = summary.participant_effects.loc[
        summary.participant_effects["target"].eq(target)
    ]
    cohort_effects = summary.cohort_effects.loc[
        summary.cohort_effects["target"].eq(target)
    ].sort_values("window_order", kind="stable")

    axis.axvline(
        0.0,
        color="#555555",
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    for window_order in range(len(summary.windows)):
        participants = participant_effects.loc[
            participant_effects["window_order"].eq(window_order)
        ].sort_values("subject_id", kind="stable")
        jitter = np.linspace(-0.12, 0.12, len(participants))
        axis.scatter(
            participants["delta_r2"].to_numpy(dtype=float),
            window_order + jitter,
            s=float(style["participant_marker_size_pt"]) ** 2,
            color=participant_color,
            alpha=max(float(style["participant_alpha"]), 0.48),
            edgecolors="white",
            linewidths=0.2,
            zorder=2,
        )

    for estimate in cohort_effects.itertuples(index=False):
        mean = float(estimate.mean_delta_r2)
        error = np.asarray(
            [
                [mean - float(estimate.ci_low_delta_r2)],
                [float(estimate.ci_high_delta_r2) - mean],
            ]
        )
        axis.errorbar(
            mean,
            int(estimate.window_order),
            xerr=error,
            color=target_color,
            linestyle="none",
            elinewidth=float(style["confidence_line_width_pt"]),
            marker="D",
            markersize=float(style["cohort_marker_size_pt"]),
            markerfacecolor="white",
            markeredgecolor=target_color,
            markeredgewidth=float(style["confidence_line_width_pt"]),
            capsize=0.0,
            zorder=3,
        )
        axis.text(
            1.025,
            int(estimate.window_order),
            f"n={int(estimate.n_subjects)}",
            ha="left",
            va="center",
            transform=axis.get_yaxis_transform(),
            clip_on=False,
            color="#333333",
            fontsize=float(font["annotation_pt"]),
        )

    labels = cohort_effects["window_label"].astype(str).tolist()
    axis.set_yticks(range(len(summary.windows)), labels=labels)
    axis.set_ylim(len(summary.windows) - 0.45, -0.45)
    axis.set_xlim(-half_range, half_range)
    axis.set_title(target, pad=6.0, fontweight="bold")
    axis.set_xlabel("Incremental held-out prediction, ΔR² (LOSO)")
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="y", length=0.0, labelleft=show_window_labels)


def _symmetric_half_range(summary: TemporalSpecificitySummary) -> float:
    values = np.concatenate(
        (
            summary.participant_effects["delta_r2"].to_numpy(dtype=float),
            summary.cohort_effects[["ci_low_delta_r2", "ci_high_delta_r2"]]
            .to_numpy(dtype=float)
            .ravel(),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Temporal-specificity figure received non-finite effect estimates.")
    return max(1.08 * float(np.max(np.abs(values))), 0.05)


def _validate_summary(summary: TemporalSpecificitySummary) -> None:
    if summary.targets != ("NPS", "SIIPS1"):
        raise ValueError("Temporal-specificity figure requires targets ('NPS', 'SIIPS1').")
    if not summary.windows:
        raise ValueError("Temporal-specificity figure requires at least one window.")

    participant_columns = {
        "target",
        "window_name",
        "window_order",
        "subject_id",
        "delta_r2",
    }
    cohort_columns = {
        "target",
        "window_name",
        "window_order",
        "window_label",
        "mean_delta_r2",
        "ci_low_delta_r2",
        "ci_high_delta_r2",
        "n_subjects",
    }
    missing_participant = sorted(
        participant_columns.difference(summary.participant_effects.columns)
    )
    missing_cohort = sorted(cohort_columns.difference(summary.cohort_effects.columns))
    if missing_participant or missing_cohort:
        raise ValueError(
            "Temporal-specificity summary is missing plot columns: "
            f"participant={missing_participant}, cohort={missing_cohort}."
        )

    expected_cells = {(target, window) for target in summary.targets for window in summary.windows}
    cohort_cells = set(
        summary.cohort_effects[["target", "window_name"]].itertuples(
            index=False,
            name=None,
        )
    )
    participant_cells = set(
        summary.participant_effects[["target", "window_name"]].itertuples(
            index=False,
            name=None,
        )
    )
    if cohort_cells != expected_cells or participant_cells != expected_cells:
        raise ValueError("Temporal-specificity summary does not contain every target/window cell.")
    if len(summary.cohort_effects) != len(expected_cells):
        raise ValueError("Temporal-specificity summary contains duplicate cohort cells.")

    for target, window in expected_cells:
        participants = summary.participant_effects.loc[
            summary.participant_effects["target"].eq(target)
            & summary.participant_effects["window_name"].eq(window)
        ]
        if participants.empty:
            raise ValueError(f"Temporal-specificity cell {target}/{window} has no participants.")
        if participants["subject_id"].duplicated().any():
            raise ValueError(
                f"Temporal-specificity cell {target}/{window} contains duplicate participants."
            )


def _legend_handles(config: Any) -> tuple[Line2D, Line2D]:
    style = require_config_value(config, "study1.figures.validity.style")
    return (
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=float(style["participant_marker_size_pt"]),
            markerfacecolor=str(style["participant_color"]),
            markeredgecolor="white",
            markeredgewidth=0.2,
            alpha=max(float(style["participant_alpha"]), 0.48),
            label="Held-out participant",
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
            label="Mean ± 95% CI",
        ),
    )


def _add_panel_label(axis, panel_index: int) -> None:
    axis.text(
        -0.07,
        1.08,
        chr(ord("a") + panel_index),
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=8.0,
        ha="right",
        va="bottom",
    )


__all__ = ["build_temporal_specificity_figure"]
