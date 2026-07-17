"""Publication rendering for Study 1 EEG power construct validity."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.power_construct_validity import (
    PowerConstructValiditySummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

FIGURE_CONFIG_KEY = "study1.figures.power_construct_validity"
LABEL_BACKGROUND = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.6}


def build_power_construct_validity_figure(
    summary: PowerConstructValiditySummary,
    config: Any,
) -> Figure:
    """Render temperature trajectories and adjusted rating associations."""

    figure_config = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    _validate_summary(summary, figure_config)
    dimensions = figure_config["dimensions_mm"]
    color = str(figure_config["colors"]["cohort"])
    style = require_config_value(config, "study1.figures.validity.style")
    font = require_config_value(config, "study1.figures.validity.font")

    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            5,
            2,
            left=0.105,
            right=0.965,
            bottom=0.11,
            top=0.72,
            width_ratios=(1.35, 1.0),
            hspace=0.12,
            wspace=0.34,
        )
        temperature_axes = []
        for index in range(len(summary.bands)):
            shared_axis = temperature_axes[0] if temperature_axes else None
            temperature_axes.append(
                figure.add_subplot(
                    grid[index, 0],
                    sharex=shared_axis,
                    sharey=shared_axis,
                )
            )
        y_limits = _temperature_y_limits(summary)
        for index, (axis, band, band_spec) in enumerate(
            zip(temperature_axes, summary.bands, figure_config["bands"], strict=True)
        ):
            _draw_temperature_panel(
                axis,
                summary,
                band=band,
                color=color,
                style=style,
                band_spec=band_spec,
                show_x_labels=index == len(temperature_axes) - 1,
                y_limits=y_limits,
            )

        rating_axis = figure.add_subplot(grid[:, 1], label="adjusted-intensity")
        _draw_rating_panel(
            rating_axis,
            summary,
            color=color,
            style=style,
            font=font,
            labels=[str(spec["label"]) for spec in figure_config["bands"]],
        )
        figure.legend(
            handles=_legend_handles(color, style),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86),
            ncol=2,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.5,
            columnspacing=1.4,
        )
        figure.text(
            0.105,
            0.965,
            "EEG band-power construct validity",
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
        figure.text(
            0.105,
            0.925,
            _sample_metadata(summary),
            ha="left",
            va="top",
            fontsize=6.0,
        )
        left_position = temperature_axes[0].get_position()
        right_position = rating_axis.get_position()
        figure.text(
            left_position.x0,
            0.79,
            "a  Temperature response",
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
        )
        figure.text(
            left_position.x0,
            0.76,
            "Simultaneous 95% participant-bootstrap band across 30 band-temperature cells",
            ha="left",
            va="top",
            fontsize=5.5,
            color="#444444",
        )
        figure.text(
            right_position.x0,
            0.79,
            "b  Adjusted intensity association",
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
        )
        figure.text(
            right_position.x0,
            0.76,
            "Pointwise 95% participant-bootstrap CI across participants",
            ha="left",
            va="top",
            fontsize=5.5,
            color="#444444",
        )
        figure.text(
            0.025,
            0.415,
            "Within-participant centered global power (dB)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.0,
        )
    return figure


def _draw_temperature_panel(
    axis,
    summary: PowerConstructValiditySummary,
    *,
    band: str,
    color: str,
    style: Mapping[str, object],
    band_spec: Mapping[str, object],
    show_x_labels: bool,
    y_limits: tuple[float, float],
) -> None:
    participant = summary.temperature_by_subject.loc[
        summary.temperature_by_subject["band"].eq(band)
    ]
    cohort = summary.temperature_summary.loc[
        summary.temperature_summary["band"].eq(band)
    ].sort_values("stimulus_temp")
    temperatures = np.asarray(summary.temperatures, dtype=float)
    for _subject, rows in participant.groupby("subject_id", sort=True):
        rows = rows.sort_values("stimulus_temp")
        axis.plot(
            rows["stimulus_temp"],
            rows["centered_power_db"],
            color=str(style["participant_color"]),
            alpha=max(float(style["participant_alpha"]), 0.25),
            linewidth=float(style["participant_line_width_pt"]),
            marker="o",
            markersize=1.7,
            markeredgewidth=0.0,
            zorder=1,
        )
    axis.fill_between(
        cohort["stimulus_temp"].to_numpy(dtype=float),
        cohort["ci_low"].to_numpy(dtype=float),
        cohort["ci_high"].to_numpy(dtype=float),
        color=color,
        alpha=0.15,
        linewidth=0.0,
        zorder=2,
    )
    axis.plot(
        cohort["stimulus_temp"],
        cohort["mean"],
        color=color,
        linewidth=float(style["cohort_line_width_pt"]),
        marker="o",
        markersize=float(style["cohort_marker_size_pt"]),
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=float(style["confidence_line_width_pt"]),
        zorder=3,
    )
    axis.axhline(0.0, color="#666666", linewidth=0.5, linestyle=(0, (3, 2)), zorder=0)
    axis.set_xlim(temperatures[0] - 0.25, temperatures[-1] + 0.25)
    axis.set_ylim(y_limits)
    axis.set_xticks(temperatures)
    axis.tick_params(axis="x", labelbottom=show_x_labels, labelrotation=0.0)
    axis.set_xlabel("Temperature (°C)" if show_x_labels else "")
    axis.set_label(f"temperature:{band}")
    low, high = band_spec["frequency_hz"]
    axis.text(
        0.012,
        0.89,
        str(band_spec["label"]),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color=color,
        bbox=LABEL_BACKGROUND,
    )
    axis.text(
        0.22,
        0.89,
        f"{float(low):g}–{float(high):g} Hz",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#555555",
        bbox=LABEL_BACKGROUND,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def _draw_rating_panel(
    axis,
    summary: PowerConstructValiditySummary,
    *,
    color: str,
    style: Mapping[str, object],
    font: Mapping[str, object],
    labels: list[str],
) -> None:
    participants = summary.rating_by_subject.loc[summary.rating_by_subject["estimable"]]
    cohort = summary.rating_summary.set_index("band")
    axis.axvline(0.0, color="#555555", linewidth=0.55, linestyle=(0, (3, 2)), zorder=0)
    for band_index, band in enumerate(summary.bands):
        values = participants.loc[participants["band"].eq(band)].sort_values("subject_id")
        jitter = _deterministic_jitter(len(values), half_width=0.12)
        axis.scatter(
            values["partial_r"].to_numpy(dtype=float),
            band_index + jitter,
            s=max(float(style["participant_marker_size_pt"]), 2.2) ** 2,
            color=str(style["participant_color"]),
            alpha=max(float(style["participant_alpha"]), 0.52),
            edgecolors="white",
            linewidths=0.2,
            zorder=2,
        )
        estimate = cohort.loc[band]
        mean = float(estimate["mean_partial_r"])
        axis.errorbar(
            mean,
            band_index,
            xerr=np.asarray(
                [
                    [mean - float(estimate["ci_low"])],
                    [float(estimate["ci_high"]) - mean],
                ]
            ),
            linestyle="none",
            color=color,
            elinewidth=float(style["confidence_line_width_pt"]),
            marker="D",
            markersize=float(style["cohort_marker_size_pt"]),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=float(style["confidence_line_width_pt"]),
            capsize=0.0,
            zorder=3,
        )
        axis.text(
            1.025,
            band_index,
            f"n={int(estimate['n_subjects'])}",
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=float(font["annotation_pt"]),
            color="#333333",
            clip_on=False,
        )
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(len(summary.bands) - 0.5, -0.5)
    axis.set_yticks(range(len(summary.bands)), labels=labels)
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.set_xlabel("Adjusted within-participant correlation, partial r")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.35, zorder=0)


def _temperature_y_limits(summary: PowerConstructValiditySummary) -> tuple[float, float]:
    values = np.concatenate(
        (
            summary.temperature_by_subject["centered_power_db"].to_numpy(dtype=float),
            summary.temperature_summary[["ci_low", "ci_high"]].to_numpy(dtype=float).ravel(),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Temperature association plot values must be finite.")
    half_range = max(1.08 * float(np.max(np.abs(values))), 0.1)
    return -half_range, half_range


def _deterministic_jitter(count: int, *, half_width: float) -> np.ndarray:
    if count < 1:
        raise ValueError("Rating association requires at least one estimable participant per band.")
    if count == 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-half_width, half_width, count)


def _legend_handles(color: str, style: Mapping[str, object]) -> list[object]:
    return [
        Line2D(
            [],
            [],
            color=str(style["participant_color"]),
            alpha=max(float(style["participant_alpha"]), 0.52),
            marker="o",
            linewidth=float(style["participant_line_width_pt"]),
            markersize=3.0,
            label="Participant estimate",
        ),
        Line2D(
            [],
            [],
            color=color,
            linewidth=float(style["cohort_line_width_pt"]),
            marker="D",
            markerfacecolor="white",
            markeredgecolor=color,
            markersize=3.0,
            label="Equal-weight cohort mean",
        ),
    ]


def _sample_metadata(summary: PowerConstructValiditySummary) -> str:
    required = {"subject_id", "trial_id"}
    missing = sorted(required.difference(summary.trials.columns))
    if missing:
        raise ValueError(
            f"Power construct-validity trials are missing metadata columns: {missing}."
        )
    unique_trials = summary.trials.loc[:, ["subject_id", "trial_id"]].drop_duplicates()
    counts = unique_trials.groupby("subject_id", sort=True).size().to_numpy(dtype=int)
    if len(counts) != summary.n_subjects or np.any(counts < 1):
        raise ValueError("Power construct-validity sample metadata is inconsistent.")
    readiness = (
        "Article-ready cohort" if summary.article_ready else "Preliminary descriptive analysis"
    )
    channel_text = "Fp1/Fp2 included" if summary.primary_include_fp1_fp2 else "Fp1/Fp2 excluded"
    median_trials = float(np.median(counts))
    return (
        f"{readiness} · n={summary.n_subjects} participants · retained trials/participant: "
        f"median {median_trials:g}, range {counts.min()}–{counts.max()} · {channel_text}"
    )


def _validate_summary(
    summary: PowerConstructValiditySummary,
    figure_config: Mapping[str, object],
) -> None:
    bands = tuple(str(spec["name"]) for spec in figure_config["bands"])
    if summary.bands != bands or len(summary.bands) != 5:
        raise ValueError("Power construct-validity figure requires five configured clean bands.")
    if len(summary.temperatures) != 6:
        raise ValueError("Power construct-validity figure requires six temperature levels.")
    if summary.n_subjects < 2:
        raise ValueError("Power construct-validity figure requires at least two participants.")
    if "interval_type" not in summary.temperature_summary:
        raise ValueError("Power construct-validity temperature summary requires interval metadata.")
    interval_types = set(summary.temperature_summary["interval_type"].astype(str))
    expected_interval = {"simultaneous_max_studentized_participant_bootstrap"}
    if interval_types != expected_interval:
        raise ValueError(
            "Power construct-validity temperature intervals must be simultaneous max-studentized "
            "participant-bootstrap bands."
        )
    if set(summary.rating_summary["band"].astype(str)) != set(summary.bands):
        raise ValueError("Power construct-validity rating summary is incomplete.")


__all__ = ["build_power_construct_validity_figure"]
