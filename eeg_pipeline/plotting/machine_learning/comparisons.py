from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.stats import compute_error_bars_from_ci_dicts
from eeg_pipeline.plotting.machine_learning.helpers import (
    despine,
    add_zero_reference_line,
    create_bar_plot,
)

logger = logging.getLogger(__name__)

def _calculate_y_limits_with_padding(values: list[float], padding_factor: float) -> tuple[float, float]:
    """Calculate y-axis limits with padding, ensuring zero is included if values are negative."""
    value_min = min(values)
    value_max = max(values)
    padded_min = min(0, value_min * padding_factor)
    padded_max = value_max * padding_factor
    return padded_min, padded_max


def plot_model_comparison(models_dict: dict, save_path: Path, config: Optional[Any] = None) -> None:
    """Plot comparison of performance metrics across multiple models."""
    if not models_dict:
        logger.warning("Empty models dictionary for comparison plot")
        return
    
    model_names = []
    r_values = []
    r2_values = []
    
    for name, metrics in models_dict.items():
        if metrics is not None and isinstance(metrics, dict):
            model_names.append(name)
            r_values.append(metrics.get('pearson_r', np.nan))
            r2_values.append(metrics.get('r2', np.nan))
    
    if not model_names:
        logger.warning("No valid models found for comparison plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="machine_learning")
    ml_config = plot_cfg.plot_type_configs.get("machine_learning", {})
    y_lim_padding = ml_config.get("y_lim_padding_factor", 1.1)
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    x_positions = np.arange(len(model_names))
    
    create_bar_plot(axes[0], x_positions, np.array(r_values), model_names, "Pearson's r", plot_cfg)
    r_min, r_max = _calculate_y_limits_with_padding(r_values, y_lim_padding)
    axes[0].set_ylim([r_min, r_max])
    
    create_bar_plot(axes[1], x_positions, np.array(r2_values), model_names, 'R²', plot_cfg)
    r2_min, r2_max = _calculate_y_limits_with_padding(r2_values, y_lim_padding)
    axes[1].set_ylim([r2_min, r2_max])
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved model comparison: {save_path}")


def plot_riemann_band_comparison(band_results: dict, save_path: Path, config: Optional[Any] = None) -> None:
    """Plot comparison of performance metrics across Riemann frequency bands."""
    if not band_results:
        logger.warning("Empty band results dictionary for Riemann comparison")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="machine_learning")
    
    bands = list(band_results.keys())
    r_values = [band_results[band].get('pearson_r', np.nan) for band in bands]
    r2_values = [band_results[band].get('r2', np.nan) for band in bands]
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    x_positions = np.arange(len(bands))
    
    create_bar_plot(axes[0], x_positions, np.array(r_values), bands, "Pearson's r", plot_cfg)
    create_bar_plot(axes[1], x_positions, np.array(r2_values), bands, 'R²', plot_cfg)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved Riemann band comparison: {save_path}")


def _plot_time_series_metric(
    ax, time_points: np.ndarray, values: np.ndarray, ylabel: str, plot_cfg, marker_size: float
) -> None:
    """Plot a time series metric with standard styling."""
    ax.plot(
        time_points,
        values,
        'o-',
        color=plot_cfg.style.colors.gray,
        linewidth=plot_cfg.style.line.width_thick,
        markersize=marker_size
    )
    add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel(ylabel)
    despine(ax)


def plot_riemann_sliding_window(sliding_df: pd.DataFrame, save_path: Path, config: Optional[Any] = None) -> None:
    """Plot performance metrics over time using sliding window analysis."""
    if sliding_df is None or len(sliding_df) == 0:
        logger.warning("Empty sliding window dataframe for Riemann plot")
        return
    
    required_columns = ['t_center', 'pearson_r', 'r2']
    missing_columns = [col for col in required_columns if col not in sliding_df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns for Riemann sliding window plot: {missing_columns}")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("sliding", plot_type="machine_learning")
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="machine_learning")
    
    fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    
    time_centers = sliding_df['t_center'].values
    r_values = sliding_df['pearson_r'].values
    r2_values = sliding_df['r2'].values
    
    _plot_time_series_metric(axes[0], time_centers, r_values, "Pearson's r", plot_cfg, marker_size)
    _plot_time_series_metric(axes[1], time_centers, r2_values, 'R²', plot_cfg, marker_size)
    axes[1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved Riemann sliding window: {save_path}")


def plot_incremental_validity(inc_summary: dict, save_path: Path, config: Optional[Any] = None) -> None:
    """Plot incremental validity analysis comparing models with and without temperature."""
    if not inc_summary:
        logger.warning("Empty incremental validity summary dictionary")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("tall", plot_type="machine_learning")
    
    random_forest_metrics = inc_summary.get('RandomForest', {})
    temperature_only_metrics = inc_summary.get('TemperatureOnly', {})
    rf_pearson_r = random_forest_metrics.get('pearson_r', np.nan)
    temperature_pearson_r = temperature_only_metrics.get('pearson_r', np.nan)
    
    delta_r_dict = inc_summary.get('delta_r', {})
    delta_r_estimate = delta_r_dict.get('estimate', np.nan)
    delta_r_ci = delta_r_dict.get('ci95', [np.nan, np.nan])
    
    partial_r_dict = inc_summary.get('partial_r_given_temperature', {})
    partial_r_estimate = partial_r_dict.get('estimate', np.nan)
    partial_r_ci = partial_r_dict.get('ci95', [np.nan, np.nan])
    
    metric_labels = ['RF', 'Temperature', 'Δr', 'Partial r']
    metric_values = [rf_pearson_r, temperature_pearson_r, delta_r_estimate, partial_r_estimate]
    confidence_intervals = [
        None,
        None,
        {'ci95': delta_r_ci},
        {'ci95': partial_r_ci}
    ]
    
    errors_lower, errors_upper = compute_error_bars_from_ci_dicts(metric_values, confidence_intervals)
    
    fig, ax = plt.subplots(figsize=fig_size)
    x_positions = np.arange(len(metric_labels))
    
    error_bar_config = {'linewidth': plot_cfg.style.line.width_standard}
    ax.bar(
        x_positions,
        metric_values,
        yerr=[errors_lower, errors_upper],
        color=plot_cfg.style.colors.gray,
        alpha=plot_cfg.style.bar.alpha,
        width=plot_cfg.style.bar.width,
        capsize=plot_cfg.style.errorbar_capsize_large,
        error_kw=error_bar_config
    )
    
    add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel("Pearson's r")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    despine(ax)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved incremental validity: {save_path}")

