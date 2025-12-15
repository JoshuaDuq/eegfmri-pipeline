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
from eeg_pipeline.io.paths import deriv_plots_path, deriv_stats_path, ensure_dir, _load_events_df
from eeg_pipeline.plotting.io.figures import (
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
    save_fig,
)
from eeg_pipeline.io.logging import get_subject_logger


def _plot_distribution_histogram(
    data: pd.Series,
    x_label: str,
    title: str,
    output_path: Path,
    plot_cfg: PlotConfig,
    config,
    logger: logging.Logger,
) -> None:
    if data.empty or data.notna().sum() < 3:
        logger.warning(f"Insufficient data for histogram: {title}")
        return

    fig_size = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    fig, ax = plt.subplots(figsize=fig_size)

    band_color = get_band_color("alpha", config)
    behavioral_config = plot_cfg.get_behavioral_config()
    default_bins = behavioral_config.get("histogram_default_bins", 30)
    bins = (
        plot_cfg.style.histogram.bins
        if hasattr(plot_cfg.style.histogram, "bins")
        else default_bins
    )

    ax.hist(
        data,
        bins=bins,
        color=band_color,
        alpha=plot_cfg.style.scatter.alpha,
        edgecolor=plot_cfg.style.histogram.edgecolor,
        linewidth=plot_cfg.style.histogram.edgewidth,
    )

    if len(data) > plot_cfg.validation.get("min_samples_for_kde", 3):
        kde = gaussian_kde(data)
        data_range = np.linspace(data.min(), data.max(), plot_cfg.style.kde_points)
        kde_vals = kde(data_range)
        kde_scale = compute_kde_scale(data, hist_bins=bins, kde_points=plot_cfg.style.kde_points)
        scaled_kde = kde_vals * kde_scale
        ax.plot(
            data_range,
            scaled_kde,
            color=plot_cfg.style.kde_color,
            linewidth=plot_cfg.style.kde_linewidth,
            alpha=plot_cfg.style.kde_alpha,
        )

    ax.set_xlabel(x_label, fontsize=plot_cfg.font.label)
    ax.set_ylabel("Frequency", fontsize=plot_cfg.font.label)
    ax.set_title(title, fontsize=plot_cfg.font.title, fontweight="bold")

    stats_text = f"n = {len(data)}\nMean = {data.mean():.2f}\nSD = {data.std():.2f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_cfg.font.title,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=plot_cfg.style.alpha_text_box,
        ),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()

    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)


def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    events = _load_events_df(subject, task, config=config)
    if events is None or len(events) == 0:
        logger.warning(f"No events for psychometrics: sub-{subject}")
        return

    psych_temp_columns = config.get("event_columns.temperature", [])
    rating_columns = config.get("event_columns.rating", [])

    temp_col = _pick_first_column(events, psych_temp_columns)
    rating_col = _pick_first_column(events, rating_columns)

    if temp_col is None:
        logger.warning(
            f"Psychometrics: no temperature column found; skipping for sub-{subject}."
        )
        return

    temp = pd.to_numeric(events[temp_col], errors="coerce")

    psychometrics_dir = plots_dir / "psychometrics"
    ensure_dir(psychometrics_dir)

    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)

    valid_mask = temp.notna()
    if rating_col is not None:
        rating = pd.to_numeric(events[rating_col], errors="coerce")
        valid_mask = valid_mask & rating.notna()
    else:
        rating = None

    if not valid_mask.sum() >= min_samples_for_plot:
        logger.warning(
            f"Insufficient valid data for psychometrics (n={valid_mask.sum()} < {min_samples_for_plot}); "
            f"skipping for sub-{subject}"
        )
        return

    temp_valid = temp[valid_mask]

    if rating is not None:
        rating_valid = rating[valid_mask]

        method_code = behavioral_config.get("method_spearman", "spearman")
        default_rng_seed = behavioral_config.get("default_rng_seed", 42)
        rng = np.random.default_rng(default_rng_seed)

        x_label = "Temperature (°C)"
        y_label = "Rating"

        output_path = psychometrics_dir / f"psychometrics_temp_vs_rating_sub-{subject}"

        generate_correlation_scatter(
            x_data=temp_valid,
            y_data=rating_valid,
            x_label=x_label,
            y_label=y_label,
            title_prefix=f"Psychometrics — Temperature vs Rating — sub-{subject}",
            band_color=get_band_color("alpha", config),
            output_path=output_path,
            method_code=method_code,
            Z_covars=None,
            covar_names=None,
            bootstrap_ci=0,
            rng=rng,
            roi_channels=None,
            logger=logger,
            annotated_stats=None,
            annot_ci=None,
            config=config,
        )

    _plot_distribution_histogram(
        data=temp_valid,
        x_label="Temperature (°C)",
        title=f"Temperature Distribution — sub-{subject}",
        output_path=psychometrics_dir / f"psychometrics_temp_distribution_sub-{subject}",
        plot_cfg=plot_cfg,
        config=config,
        logger=logger,
    )

    logger.info(f"Completed psychometrics plotting for sub-{subject}")
