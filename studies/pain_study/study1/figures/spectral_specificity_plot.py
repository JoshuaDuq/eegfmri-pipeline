"""Publication rendering for Study 1 spectral specificity."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.spectral_specificity import (
    SpectralSpecificitySummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

TARGET_COLOR_KEYS = {"NPS": "nps", "SIIPS1": "siips1"}


def build_spectral_specificity_figure(
    summary: SpectralSpecificitySummary,
    config: Any,
) -> Figure:
    """Render participant effects and cohort intervals by spectral family."""

    _validate_summary(summary)
    dimensions = require_config_value(
        config,
        "study1.figures.spectral_specificity.dimensions_mm",
    )
    colors = require_config_value(config, "study1.figures.validity.colors")
    if not isinstance(colors, Mapping):
        raise ValueError("study1.figures.validity.colors must be a mapping.")

    half_range = _symmetric_half_range(summary)
    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            1,
            2,
            left=0.255,
            right=0.965,
            bottom=0.225,
            top=0.79,
            wspace=0.44,
        )
        axes = (
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
        )
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
                show_feature_labels=panel_index == 0,
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
    summary: SpectralSpecificitySummary,
    *,
    target: str,
    target_color: str,
    half_range: float,
    show_feature_labels: bool,
    config: Any,
) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    font = require_config_value(config, "study1.figures.validity.font")
    participants = summary.participant_effects.loc[summary.participant_effects["target"].eq(target)]
    cohort = summary.cohort_effects.loc[summary.cohort_effects["target"].eq(target)].sort_values(
        "feature_order", kind="stable"
    )

    axis.axvline(
        0.0,
        color="#555555",
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    for feature_order in range(len(summary.feature_specs)):
        feature_participants = participants.loc[
            participants["feature_order"].eq(feature_order)
        ].sort_values("subject_id", kind="stable")
        jitter = _deterministic_jitter(len(feature_participants), half_width=0.12)
        axis.scatter(
            feature_participants["delta_r2"].to_numpy(dtype=float),
            feature_order + jitter,
            s=max(float(style["participant_marker_size_pt"]), 2.2) ** 2,
            color=str(style["participant_color"]),
            alpha=max(float(style["participant_alpha"]), 0.52),
            edgecolors="white",
            linewidths=0.2,
            zorder=2,
        )

    for estimate in cohort.itertuples(index=False):
        mean = float(estimate.mean_delta_r2)
        axis.errorbar(
            mean,
            int(estimate.feature_order),
            xerr=np.asarray(
                [
                    [mean - float(estimate.ci_low_delta_r2)],
                    [float(estimate.ci_high_delta_r2) - mean],
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
            capsize=0.0,
            zorder=3,
        )
        axis.text(
            1.025,
            int(estimate.feature_order),
            f"n={int(estimate.n_subjects)}",
            ha="left",
            va="center",
            transform=axis.get_yaxis_transform(),
            clip_on=False,
            color="#333333",
            fontsize=float(font["annotation_pt"]),
        )

    axis.set_yticks(
        range(len(summary.feature_specs)),
        labels=cohort["feature_label"].astype(str).tolist(),
    )
    axis.set_ylim(len(summary.feature_specs) - 0.45, -0.45)
    axis.set_xlim(-half_range, half_range)
    axis.set_title(target, pad=6.0, fontweight="bold")
    axis.set_xlabel("Incremental held-out prediction, ΔR² (LOSO)")
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="y", length=0.0, labelleft=show_feature_labels)


def _symmetric_half_range(summary: SpectralSpecificitySummary) -> float:
    values = np.concatenate(
        (
            summary.participant_effects["delta_r2"].to_numpy(dtype=float),
            summary.cohort_effects[["ci_low_delta_r2", "ci_high_delta_r2"]]
            .to_numpy(dtype=float)
            .ravel(),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Spectral-specificity figure received non-finite effects.")
    return max(1.08 * float(np.max(np.abs(values))), 0.05)


def _deterministic_jitter(count: int, *, half_width: float) -> np.ndarray:
    if count <= 0:
        raise ValueError("Participant jitter requires at least one observation.")
    if count == 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-half_width, half_width, count)


def _validate_summary(summary: SpectralSpecificitySummary) -> None:
    if summary.targets != ("NPS", "SIIPS1"):
        raise ValueError("Spectral-specificity figure requires targets ('NPS', 'SIIPS1').")
    if not summary.feature_specs:
        raise ValueError("Spectral-specificity figure requires feature specifications.")
    participant_columns = {
        "target",
        "feature_spec",
        "feature_order",
        "subject_id",
        "delta_r2",
    }
    cohort_columns = {
        "target",
        "feature_spec",
        "feature_order",
        "feature_label",
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
            "Spectral-specificity summary is missing plot columns: "
            f"participant={missing_participant}, cohort={missing_cohort}."
        )
    expected_cells = {
        (target, feature_spec)
        for target in summary.targets
        for feature_spec in summary.feature_specs
    }
    participant_cells = set(
        summary.participant_effects[["target", "feature_spec"]].itertuples(
            index=False,
            name=None,
        )
    )
    cohort_cells = set(
        summary.cohort_effects[["target", "feature_spec"]].itertuples(
            index=False,
            name=None,
        )
    )
    if participant_cells != expected_cells or cohort_cells != expected_cells:
        raise ValueError("Spectral-specificity summary lacks required target/feature cells.")
    if len(summary.cohort_effects) != len(expected_cells):
        raise ValueError("Spectral-specificity summary contains duplicate cohort cells.")
    for target, feature_spec in expected_cells:
        participants = summary.participant_effects.loc[
            summary.participant_effects["target"].eq(target)
            & summary.participant_effects["feature_spec"].eq(feature_spec)
        ]
        cohort = summary.cohort_effects.loc[
            summary.cohort_effects["target"].eq(target)
            & summary.cohort_effects["feature_spec"].eq(feature_spec)
        ].iloc[0]
        if participants["subject_id"].duplicated().any():
            raise ValueError(f"Duplicate participants in {target}/{feature_spec}.")
        if len(participants) != int(cohort["n_subjects"]):
            raise ValueError(f"Subject count disagrees for {target}/{feature_spec}.")


def _legend_handles(config: Any) -> tuple[Line2D, Line2D]:
    style = require_config_value(config, "study1.figures.validity.style")
    return (
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=max(float(style["participant_marker_size_pt"]), 2.2),
            markerfacecolor=str(style["participant_color"]),
            markeredgecolor="white",
            markeredgewidth=0.2,
            alpha=max(float(style["participant_alpha"]), 0.52),
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


__all__ = ["build_spectral_specificity_figure"]
