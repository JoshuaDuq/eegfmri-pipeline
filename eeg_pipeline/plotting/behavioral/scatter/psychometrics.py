from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from eeg_pipeline.plotting.config import PlotConfig, get_plot_config
from eeg_pipeline.plotting.behavioral.builders import generate_correlation_scatter
from eeg_pipeline.utils.analysis.stats import compute_kde_scale
from eeg_pipeline.utils.data import _pick_first_column
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir, _load_events_df
from eeg_pipeline.plotting.io.figures import (
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    save_fig,
)
from eeg_pipeline.infra.logging import get_subject_logger


_MIN_SAMPLES_FOR_HISTOGRAM = 3


def _compute_histogram_bins(plot_config: PlotConfig) -> int:
    """Extract histogram bin count from configuration."""
    behavioral_config = plot_config.get_behavioral_config()
    default_bins = plot_config.get_histogram_bins(plot_type="behavioral")
    return int(behavioral_config.get("histogram_bins", default_bins))


def _add_kde_overlay(
    ax: plt.Axes,
    data: pd.Series,
    bins: int,
    plot_config: PlotConfig,
) -> None:
    """Add KDE overlay to histogram if sufficient data available."""
    min_samples_for_kde = plot_config.validation.get("min_samples_for_kde", _MIN_SAMPLES_FOR_HISTOGRAM)
    if len(data) <= min_samples_for_kde:
        return

    kde = gaussian_kde(data)
    data_range = np.linspace(data.min(), data.max(), plot_config.style.kde_points)
    kde_values = kde(data_range)
    kde_scale = compute_kde_scale(
        data,
        hist_bins=bins,
        kde_points=plot_config.style.kde_points,
    )
    scaled_kde = kde_values * kde_scale

    ax.plot(
        data_range,
        scaled_kde,
        color=plot_config.style.kde_color,
        linewidth=plot_config.style.kde_linewidth,
        alpha=plot_config.style.kde_alpha,
    )


def _add_statistics_text(ax: plt.Axes, data: pd.Series, plot_config: PlotConfig) -> None:
    """Add statistics text box to the plot."""
    sample_count = len(data)
    mean_value = data.mean()
    std_value = data.std()
    stats_text = f"n = {sample_count}\nMean = {mean_value:.2f}\nSD = {std_value:.2f}"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_config.font.title,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=plot_config.style.alpha_text_box,
        ),
    )


def _plot_distribution_histogram(
    data: pd.Series,
    x_label: str,
    title: str,
    output_path: Path,
    plot_config: PlotConfig,
    config,
    logger: logging.Logger,
) -> None:
    """Plot histogram with optional KDE overlay for distribution visualization."""
    n_valid = data.notna().sum()
    if data.empty or n_valid < _MIN_SAMPLES_FOR_HISTOGRAM:
        logger.warning(f"Insufficient data for histogram: {title}")
        return

    fig_size = plot_config.get_figure_size("standard", plot_type="behavioral")
    fig, ax = plt.subplots(figsize=fig_size)

    band_color = get_band_color("alpha", config)
    bins = _compute_histogram_bins(plot_config)

    ax.hist(
        data,
        bins=bins,
        color=band_color,
        alpha=plot_config.style.scatter.alpha,
        edgecolor=plot_config.style.histogram.edgecolor,
        linewidth=plot_config.style.histogram.edgewidth,
    )

    _add_kde_overlay(ax, data, bins, plot_config)

    ax.set_xlabel(x_label, fontsize=plot_config.font.label)
    ax.set_ylabel("Frequency", fontsize=plot_config.font.label)
    ax.set_title(title, fontsize=plot_config.font.title, fontweight="bold")

    _add_statistics_text(ax, data, plot_config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()

    save_fig(
        fig,
        output_path,
        formats=plot_config.formats,
        dpi=plot_config.dpi,
        bbox_inches=plot_config.bbox_inches,
        pad_inches=plot_config.pad_inches,
        footer=_get_behavior_footer(config),
        logger=logger,
        config=config,
    )
    plt.close(fig)


def _load_and_validate_psychometric_data(
    events: pd.DataFrame,
    temperature_columns: list[str],
    rating_columns: list[str],
    logger: logging.Logger,
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Load and validate temperature and rating data from events DataFrame."""
    temperature_column = _pick_first_column(events, temperature_columns)
    if temperature_column is None:
        return None, None

    rating_column = _pick_first_column(events, rating_columns)
    temperature = pd.to_numeric(events[temperature_column], errors="coerce")

    valid_mask = temperature.notna()
    if rating_column is not None:
        rating = pd.to_numeric(events[rating_column], errors="coerce")
        valid_mask = valid_mask & rating.notna()
    else:
        rating = None

    temperature_valid = temperature[valid_mask]
    rating_valid = rating[valid_mask] if rating is not None else None

    return temperature_valid, rating_valid


def _plot_temperature_rating_correlation(
    temperature: pd.Series,
    rating: pd.Series,
    subject: str,
    output_dir: Path,
    plot_config: PlotConfig,
    config,
    logger: logging.Logger,
) -> None:
    """Generate scatter plot of temperature vs rating with correlation statistics."""
    behavioral_config = plot_config.get_behavioral_config()
    rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = np.random.default_rng(rng_seed)

    output_path = output_dir / f"psychometrics_temp_vs_rating_sub-{subject}"

    generate_correlation_scatter(
        x_data=temperature,
        y_data=rating,
        x_label="Temperature (°C)",
        y_label="Rating",
        title_prefix=f"Psychometrics — Temperature vs Rating — sub-{subject}",
        band_color=get_band_color("alpha", config),
        output_path=output_path,
        rng=rng,
        logger=logger,
        config=config,
    )


def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    """Generate psychometric plots for temperature and rating data."""
    if config is None:
        raise ValueError("config is required for psychometrics plotting")

    logger = get_subject_logger("behavior_analysis", subject)
    plot_config = get_plot_config(config)
    behavioral_config = plot_config.get_behavioral_config()

    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    events = _load_events_df(subject, task, config=config)
    if events is None or len(events) == 0:
        logger.warning(f"No events for psychometrics: sub-{subject}")
        return

    temperature_columns = config.get("event_columns.temperature", [])
    rating_columns = config.get("event_columns.rating", [])

    temperature_valid, rating_valid = _load_and_validate_psychometric_data(
        events,
        temperature_columns,
        rating_columns,
        logger,
    )

    if temperature_valid is None:
        logger.warning(
            f"Psychometrics: no temperature column found; skipping for sub-{subject}."
        )
        return

    n_valid = len(temperature_valid)
    min_samples_for_plot = plot_config.validation.get("min_samples_for_plot", 5)
    if n_valid < min_samples_for_plot:
        logger.warning(
            f"Insufficient valid data for psychometrics (n={n_valid} < {min_samples_for_plot}); "
            f"skipping for sub-{subject}"
        )
        return

    psychometrics_dir = plots_dir / "psychometrics"
    ensure_dir(psychometrics_dir)

    if rating_valid is not None:
        _plot_temperature_rating_correlation(
            temperature_valid,
            rating_valid,
            subject,
            psychometrics_dir,
            plot_config,
            config,
            logger,
        )

    _plot_distribution_histogram(
        data=temperature_valid,
        x_label="Temperature (°C)",
        title=f"Temperature Distribution — sub-{subject}",
        output_path=psychometrics_dir / f"psychometrics_temp_distribution_sub-{subject}",
        plot_config=plot_config,
        config=config,
        logger=logger,
    )

    logger.info(f"Completed psychometrics plotting for sub-{subject}")
