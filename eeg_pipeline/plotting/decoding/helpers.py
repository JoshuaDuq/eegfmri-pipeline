"""
Decoding Plotting Helpers (Canonical)
=====================================

Shared helper functions for decoding visualization modules.
All decoding plotting modules should import these helpers from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.utils.analysis.stats import extract_finite_mask

logger = logging.getLogger(__name__)


###################################################################
# Axis/Styling Helpers
###################################################################

def despine(ax) -> None:
    """Remove top and right spines from axis."""
    sns.despine(ax=ax, trim=True)


def calculate_axis_limits(
    values: np.ndarray, 
    plot_cfg, 
    margin_factor: Optional[float] = None
) -> tuple[float, float]:
    """Calculate axis limits with margin."""
    if margin_factor is None:
        margin_factor = plot_cfg.plot_type_configs.get("decoding", {}).get("axis_margin_factor", 0.05)
    value_min = np.nanmin(values)
    value_max = np.nanmax(values)
    margin = (value_max - value_min) * margin_factor
    return value_min - margin, value_max + margin


def calculate_shared_axis_limits(
    values1: np.ndarray, 
    values2: np.ndarray, 
    plot_cfg,
    margin_factor: Optional[float] = None
) -> tuple[float, float]:
    """Calculate shared axis limits for two value arrays."""
    combined_values = np.concatenate([values1, values2])
    return calculate_axis_limits(combined_values, plot_cfg, margin_factor)


def add_zero_reference_line(
    ax, 
    plot_cfg, 
    linewidth: Optional[float] = None, 
    linestyle: str = '--', 
    alpha: Optional[float] = None
) -> None:
    """Add horizontal zero reference line to axis."""
    if linewidth is None:
        linewidth = plot_cfg.style.line.width_thin
    if alpha is None:
        alpha = plot_cfg.style.line.alpha_zero
    ax.axhline(0, color=plot_cfg.style.colors.black, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def create_bar_plot(
    ax, 
    x_positions: np.ndarray, 
    values: np.ndarray, 
    labels: list,
    ylabel: str, 
    plot_cfg, 
    color: Optional[str] = None, 
    alpha: Optional[float] = None, 
    width: Optional[float] = None, 
    add_zero_line: bool = True
) -> None:
    """Create a bar plot with standard styling."""
    if color is None:
        color = plot_cfg.style.colors.gray
    if alpha is None:
        alpha = plot_cfg.style.bar.alpha
    if width is None:
        width = plot_cfg.style.bar.width
    ax.bar(x_positions, values, color=color, alpha=alpha, width=width)
    if add_zero_line:
        add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    despine(ax)


###################################################################
# Residual Diagnostics
###################################################################

def plot_residual_diagnostics(
    pred_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
    config: Optional[Any] = None,
) -> None:
    """Plot residual diagnostics: residuals vs predicted values and Q-Q plot."""
    if pred_df is None or len(pred_df) == 0:
        logger.warning(f"Empty prediction dataframe for {model_name} residual diagnostics")
        return
    
    required_columns = ['y_true', 'y_pred']
    if not all(col in pred_df.columns for col in required_columns):
        logger.warning(f"Missing required columns for {model_name} residual diagnostics")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="decoding")
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="decoding")
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values
    y_true_finite, y_pred_finite, mask = extract_finite_mask(y_true, y_pred)
    residuals = y_true_finite - y_pred_finite
    
    axes[0].scatter(y_pred_finite, residuals, s=marker_size, alpha=plot_cfg.style.scatter.alpha, 
                    c=plot_cfg.style.colors.gray, edgecolors='none')
    add_zero_reference_line(axes[0], plot_cfg, 
                            linewidth=plot_cfg.style.line.width_standard, 
                            alpha=plot_cfg.style.line.alpha_reference)
    axes[0].set_xlabel('Predicted Rating')
    axes[0].set_ylabel('Residual')
    despine(axes[0])
    
    residuals_sorted = np.sort(residuals)
    decoding_config = plot_cfg.plot_type_configs.get("decoding", {})
    quantile_min = decoding_config.get("quantile_min", 0.01)
    quantile_max = decoding_config.get("quantile_max", 0.99)
    quantile_range = np.linspace(quantile_min, quantile_max, len(residuals_sorted))
    theoretical_quantiles = stats.norm.ppf(quantile_range)
    axes[1].scatter(theoretical_quantiles, residuals_sorted, s=marker_size, 
                    alpha=plot_cfg.style.scatter.alpha, c=plot_cfg.style.colors.gray, edgecolors='none')
    
    lim_min, lim_max = calculate_shared_axis_limits(theoretical_quantiles, residuals_sorted, plot_cfg)
    reference_limits = [lim_min, lim_max]
    axes[1].plot(reference_limits, reference_limits, color=plot_cfg.style.colors.black, linestyle='--',
                 linewidth=plot_cfg.style.line.width_standard, alpha=plot_cfg.style.line.alpha_reference)
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')
    despine(axes[1])
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} residual diagnostics: {save_path}")
