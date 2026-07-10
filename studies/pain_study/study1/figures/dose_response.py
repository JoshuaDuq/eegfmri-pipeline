"""Single-axis dose-response rendering for Study 1 validity figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.validity_data import DoseResponseSummary
from studies.pain_study.study1.figures.validity_style import (
    configured_figure_size,
    publication_style,
)


@dataclass(frozen=True)
class HorizontalReference:
    value: float
    label: str


@dataclass(frozen=True)
class DoseResponseSpecification:
    ylabel: str
    color_config_key: str
    y_limits: tuple[float, float] | None = None
    reference: HorizontalReference | None = None


def build_dose_response_figure(
    summary: DoseResponseSummary,
    specification: DoseResponseSpecification,
    config: Any,
) -> Figure:
    style = require_config_value(config, "study1.figures.validity.style")
    colors = require_config_value(config, "study1.figures.validity.colors")
    if specification.color_config_key not in colors:
        raise ValueError(
            "Unknown Study 1 validity color key: "
            f"{specification.color_config_key!r}."
        )
    cohort_color = str(colors[specification.color_config_key])

    with publication_style(config):
        figure, axis = plt.subplots(figsize=configured_figure_size(config), layout="constrained")
        _draw_participants(axis, summary, style)
        _draw_cohort(axis, summary, style, cohort_color)
        _format_axis(axis, summary, specification, config)
    return figure


def _draw_participants(axis, summary: DoseResponseSummary, style: dict[str, Any]) -> None:
    temperatures = np.asarray(summary.temperatures, dtype=float)
    for participant_index, (_, values) in enumerate(summary.participant_matrix.iterrows()):
        axis.plot(
            temperatures,
            values.to_numpy(dtype=float),
            color=str(style["participant_color"]),
            alpha=float(style["participant_alpha"]),
            linewidth=float(style["participant_line_width_pt"]),
            marker="o",
            markersize=float(style["participant_marker_size_pt"]),
            markeredgewidth=0.0,
            label="Participants" if participant_index == 0 else None,
            zorder=1,
        )


def _draw_cohort(
    axis,
    summary: DoseResponseSummary,
    style: dict[str, Any],
    color: str,
) -> None:
    cohort = summary.cohort_estimates
    means = cohort["mean"].to_numpy(dtype=float)
    errors = np.vstack(
        (
            means - cohort["ci_low"].to_numpy(dtype=float),
            cohort["ci_high"].to_numpy(dtype=float) - means,
        )
    )
    axis.errorbar(
        cohort["stimulus_temp"].to_numpy(dtype=float),
        means,
        yerr=errors,
        color=color,
        linewidth=float(style["cohort_line_width_pt"]),
        elinewidth=float(style["confidence_line_width_pt"]),
        marker="o",
        markersize=float(style["cohort_marker_size_pt"]),
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=float(style["confidence_line_width_pt"]),
        capsize=2.0,
        label="Mean ± 95% CI",
        zorder=3,
    )


def _format_axis(
    axis,
    summary: DoseResponseSummary,
    specification: DoseResponseSpecification,
    config: Any,
) -> None:
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel(specification.ylabel)
    axis.set_xticks(summary.temperatures)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    axis.legend(frameon=False, loc="best", handlelength=1.5)
    if specification.y_limits is None:
        axis.set_ylim(_data_limits(summary))
    else:
        axis.set_ylim(specification.y_limits)
    if specification.reference is not None:
        _draw_reference(axis, specification.reference, config)


def _data_limits(summary: DoseResponseSummary) -> tuple[float, float]:
    participant_values = summary.participant_matrix.to_numpy(dtype=float).ravel()
    cohort = summary.cohort_estimates
    values = np.concatenate(
        (
            participant_values[np.isfinite(participant_values)],
            cohort["ci_low"].to_numpy(dtype=float),
            cohort["ci_high"].to_numpy(dtype=float),
        )
    )
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    margin = 0.05 * span if span > 0.0 else max(abs(lower) * 0.05, 1.0)
    return lower - margin, upper + margin


def _draw_reference(axis, reference: HorizontalReference, config: Any) -> None:
    font = require_config_value(config, "study1.figures.validity.font")
    axis.axhline(
        reference.value,
        color="#555555",
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    axis.annotate(
        reference.label,
        xy=(1.0, reference.value),
        xycoords=("axes fraction", "data"),
        xytext=(-2.0, 2.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#222222",
        fontsize=float(font["annotation_pt"]),
    )


__all__ = [
    "DoseResponseSpecification",
    "HorizontalReference",
    "build_dose_response_figure",
]
