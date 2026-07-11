"""Publication rendering for Study 2 Haufe forward patterns."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study2.figures.style import figure_size_inches, publication_style
from studies.pain_study.study2.sensor_patterns import SensorPatternSummary

FIGURE_CONFIG_KEY = "study2.figures.haufe_forward_patterns"


def build_haufe_forward_patterns_figure(
    summary: SensorPatternSummary,
    config: Any,
) -> Figure:
    """Render NPS scalp maps and descriptive LOSO fold stability."""

    figure_config = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    _validate_summary(summary, figure_config)
    dimensions = figure_config["dimensions_mm"]
    font_family = str(figure_config["font_family"])
    information = _sensor_information(summary, str(figure_config["montage"]))
    half_range = _shared_half_range(summary)

    with publication_style(font_family):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            2,
            5,
            left=0.09,
            right=0.88,
            bottom=0.14,
            top=0.80,
            height_ratios=(1.0, 0.78),
            hspace=0.56,
            wspace=0.18,
        )
        topomap_axes = [figure.add_subplot(grid[0, index]) for index in range(5)]
        image = None
        for index, (axis, band, band_spec) in enumerate(
            zip(topomap_axes, summary.bands, figure_config["bands"], strict=True)
        ):
            image, _ = mne.viz.plot_topomap(
                _map_values(summary, band),
                information,
                axes=axis,
                show=False,
                sensors=True,
                contours=0,
                cmap="RdBu_r",
                vlim=(-half_range, half_range),
                extrapolate="local",
                border="mean",
                image_interp="cubic",
                sphere=(0.0, 0.0, 0.0, 0.095),
                res=160,
            )
            low, high = band_spec["frequency_hz"]
            axis.set_title(
                f"{band_spec['label']}\n{low:g}–{high:g} Hz",
                pad=5.0,
                fontweight="bold" if index < 2 else "normal",
            )
        if image is None:
            raise RuntimeError("No Study 2 Haufe topomap was created.")

        stability_axis = figure.add_subplot(grid[1, :])
        _draw_stability(
            stability_axis,
            summary,
            color=str(figure_config["nps_color"]),
            labels=[str(spec["label"]) for spec in figure_config["bands"]],
        )
        color_axis = figure.add_axes((0.905, 0.49, 0.012, 0.27))
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label(
            "Normalized Haufe forward-pattern loading (a.u.)",
            rotation=270,
            labelpad=7.0,
        )
        colorbar.outline.set_linewidth(0.5)

        readiness = (
            f"Article-ready cohort (n={summary.n_subjects})"
            if summary.article_ready
            else f"Preliminary cohort; descriptive patterns (n={summary.n_subjects})"
        )
        figure.text(
            0.5,
            0.965,
            f"NPS sensor-level forward patterns · {readiness}",
            ha="center",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
        figure.text(0.025, 0.87, "a", ha="left", va="top", fontsize=8.0, fontweight="bold")
        figure.text(0.025, 0.40, "b", ha="left", va="top", fontsize=8.0, fontweight="bold")
        figure.text(
            0.5,
            0.018,
            "Sensor-level multivariate forward patterns; not cortical source localization",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#444444",
        )
    return figure


def _sensor_information(summary: SensorPatternSummary, montage_name: str) -> mne.Info:
    montage = mne.channels.make_standard_montage(montage_name)
    missing = sorted(set(summary.channels).difference(montage.ch_names))
    if missing:
        raise ValueError(f"Channels are absent from montage {montage_name!r}: {missing}.")
    information = mne.create_info(list(summary.channels), sfreq=1.0, ch_types="eeg")
    information.set_montage(montage, on_missing="raise")
    return information


def _map_values(summary: SensorPatternSummary, band: str) -> np.ndarray:
    cell = summary.aggregate_patterns.loc[summary.aggregate_patterns["band"].eq(band)].set_index(
        "channel"
    )
    if set(cell.index.astype(str)) != set(summary.channels):
        raise ValueError(f"Aggregate sensor map for {band!r} has an incomplete channel set.")
    return cell.loc[list(summary.channels), "median_normalized_pattern"].to_numpy(dtype=float)


def _shared_half_range(summary: SensorPatternSummary) -> float:
    values = summary.aggregate_patterns["median_normalized_pattern"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.any(values):
        raise ValueError("Aggregate Haufe patterns must contain finite non-zero values.")
    return float(np.max(np.abs(values)))


def _draw_stability(
    axis,
    summary: SensorPatternSummary,
    *,
    color: str,
    labels: list[str],
) -> None:
    for band_index, band in enumerate(summary.bands):
        values = summary.stability.loc[
            summary.stability["band"].eq(band), "spatial_correlation"
        ].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Fold stability is missing for band {band!r}.")
        jitter = np.linspace(-0.09, 0.09, len(values))
        axis.scatter(
            values,
            band_index + jitter,
            s=8.0,
            color="#777777",
            alpha=0.35,
            linewidths=0.0,
            zorder=2,
        )
        axis.scatter(
            [float(np.median(values))],
            [band_index],
            marker="D",
            s=19.0,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
    axis.axvline(0.0, color="#555555", linewidth=0.55, linestyle=(0, (3, 2)), zorder=0)
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(len(summary.bands) - 0.5, -0.5)
    axis.set_yticks(range(len(summary.bands)), labels=labels)
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.set_xlabel("Pairwise LOSO fold-map spatial correlation, r")
    axis.set_title("Descriptive fold stability", pad=5.0, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.35, zorder=0)


def _validate_summary(summary: SensorPatternSummary, figure_config: Mapping[str, object]) -> None:
    bands = tuple(str(spec["name"]) for spec in figure_config["bands"])
    if summary.target != figure_config["target"] or summary.target != "NPS":
        raise ValueError("Study 2 Haufe figure requires the configured NPS target.")
    if summary.bands != bands or len(summary.bands) != 5:
        raise ValueError("Study 2 Haufe figure requires the five configured clean-band maps.")
    if len(summary.channels) < 4:
        raise ValueError("Study 2 topographies require at least four EEG channels.")


__all__ = ["build_haufe_forward_patterns_figure"]
