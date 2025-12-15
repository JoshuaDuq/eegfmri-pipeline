from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.stats import compute_error_bars_from_ci_dicts

logger = logging.getLogger(__name__)


###################################################################
# Helper Functions (imported from helpers module)
###################################################################

from eeg_pipeline.plotting.decoding.helpers import (
    _despine,
    _add_zero_reference_line,
    _create_bar_plot,
)


###################################################################
# Model Comparison Plots
###################################################################

def plot_model_comparison(models_dict: dict, save_path: Path, config: Optional[Any] = None) -> None:
    """
    Plot comparison of performance metrics across multiple models.
    """
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
    
    if len(model_names) == 0:
        logger.warning("No valid models found for comparison plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="decoding")
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    x_positions = np.arange(len(model_names))
    
    _create_bar_plot(axes[0], x_positions, np.array(r_values), model_names, "Pearson's r", plot_cfg)
    y_lim_padding = plot_cfg.plot_type_configs.get("decoding", {}).get("y_lim_padding_factor", 1.1)
    r_min = min(0, min(r_values) * y_lim_padding)
    r_max = max(r_values) * y_lim_padding
    axes[0].set_ylim([r_min, r_max])
    
    _create_bar_plot(axes[1], x_positions, np.array(r2_values), model_names, 'R²', plot_cfg)
    r2_min = min(0, min(r2_values) * y_lim_padding)
    r2_max = max(r2_values) * y_lim_padding
    axes[1].set_ylim([r2_min, r2_max])
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved model comparison: {save_path}")


def plot_riemann_band_comparison(band_results: dict, save_path: Path, config: Optional[Any] = None) -> None:
    """
    Plot comparison of performance metrics across Riemann frequency bands.
    """
    if not band_results:
        logger.warning("Empty band results dictionary for Riemann comparison")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="decoding")
    
    bands = list(band_results.keys())
    r_vals = [band_results[band].get('pearson_r', np.nan) for band in bands]
    r2_vals = [band_results[band].get('r2', np.nan) for band in bands]
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    x_positions = np.arange(len(bands))
    
    _create_bar_plot(axes[0], x_positions, np.array(r_vals), bands, "Pearson's r", plot_cfg)
    _create_bar_plot(axes[1], x_positions, np.array(r2_vals), bands, 'R²', plot_cfg)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved Riemann band comparison: {save_path}")


def plot_riemann_sliding_window(sliding_df: pd.DataFrame, save_path: Path, config: Optional[Any] = None) -> None:
    """
    Plot performance metrics over time using sliding window analysis.
    """
    if sliding_df is None or len(sliding_df) == 0:
        logger.warning("Empty sliding window dataframe for Riemann plot")
        return
    
    required_columns = ['t_center', 'pearson_r', 'r2']
    if not all(col in sliding_df.columns for col in required_columns):
        logger.warning("Missing required columns for Riemann sliding window plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("sliding", plot_type="decoding")
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="decoding")
    
    fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    
    time_centers = sliding_df['t_center'].values
    r_values = sliding_df['pearson_r'].values
    r2_values = sliding_df['r2'].values
    
    axes[0].plot(time_centers, r_values, 'o-', color=plot_cfg.style.colors.gray, 
                 linewidth=plot_cfg.style.line.width_thick, markersize=marker_size)
    _add_zero_reference_line(axes[0], plot_cfg)
    axes[0].set_ylabel("Pearson's r")
    _despine(axes[0])
    
    axes[1].plot(time_centers, r2_values, 'o-', color=plot_cfg.style.colors.gray, 
                 linewidth=plot_cfg.style.line.width_thick, markersize=marker_size)
    _add_zero_reference_line(axes[1], plot_cfg)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('R²')
    _despine(axes[1])
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved Riemann sliding window: {save_path}")


def plot_incremental_validity(inc_summary: dict, save_path: Path, config: Optional[Any] = None) -> None:
    """
    Plot incremental validity analysis comparing models with and without temperature.
    """
    if not inc_summary:
        logger.warning("Empty incremental validity summary dictionary")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("tall", plot_type="decoding")
    
    rf_r = inc_summary.get('RandomForest', {}).get('pearson_r', np.nan)
    temp_r = inc_summary.get('TemperatureOnly', {}).get('pearson_r', np.nan)
    
    delta_r_dict = inc_summary.get('delta_r', {})
    delta_r = delta_r_dict.get('estimate', np.nan)
    delta_r_ci = delta_r_dict.get('ci95', [np.nan, np.nan])
    
    partial_r_dict = inc_summary.get('partial_r_given_temperature', {})
    partial_r = partial_r_dict.get('estimate', np.nan)
    partial_r_ci = partial_r_dict.get('ci95', [np.nan, np.nan])
    
    metrics = ['RF', 'Temperature', 'Δr', 'Partial r']
    values = [rf_r, temp_r, delta_r, partial_r]
    ci_dicts = [None, None, {'ci95': delta_r_ci}, {'ci95': partial_r_ci}]
    
    errors_lower, errors_upper = compute_error_bars_from_ci_dicts(values, ci_dicts)
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    x_positions = np.arange(len(metrics))
    ax.bar(x_positions, values, yerr=[errors_lower, errors_upper], 
           color=plot_cfg.style.colors.gray, alpha=plot_cfg.style.bar.alpha, 
           width=plot_cfg.style.bar.width, capsize=plot_cfg.style.errorbar_capsize_large, 
           error_kw={'linewidth': plot_cfg.style.line.width_standard})
    _add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel("Pearson's r")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    _despine(ax)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved incremental validity: {save_path}")

