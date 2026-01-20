"""
Machine Learning Plotting Helpers (Canonical)
=====================================

Shared helper functions for machine learning visualization modules.
All machine learning plotting modules should import these helpers from here.
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


def despine(ax) -> None:
    """Remove top and right spines from axis."""
    sns.despine(ax=ax, trim=True)


def _get_machine_learning_config(plot_cfg) -> dict[str, Any]:
    """Extract machine learning plot configuration."""
    return plot_cfg.plot_type_configs.get("machine_learning", {})


def _get_axis_margin_factor(plot_cfg) -> float:
    """Get axis margin factor from configuration."""
    machine_learning_config = _get_machine_learning_config(plot_cfg)
    return machine_learning_config.get("axis_margin_factor", 0.05)


def calculate_axis_limits(
    values: np.ndarray,
    plot_cfg: Any,
    margin_factor: Optional[float] = None,
) -> tuple[float, float]:
    """Calculate axis limits with margin."""
    if margin_factor is None:
        margin_factor = _get_axis_margin_factor(plot_cfg)
    
    value_min = np.nanmin(values)
    value_max = np.nanmax(values)
    value_range = value_max - value_min
    margin = value_range * margin_factor
    
    return value_min - margin, value_max + margin


def calculate_shared_axis_limits(
    values1: np.ndarray,
    values2: np.ndarray,
    plot_cfg: Any,
    margin_factor: Optional[float] = None,
) -> tuple[float, float]:
    """Calculate shared axis limits for two value arrays."""
    combined_values = np.concatenate([values1, values2])
    return calculate_axis_limits(combined_values, plot_cfg, margin_factor)


def add_zero_reference_line(
    ax: Any,
    plot_cfg: Any,
    linewidth: Optional[float] = None,
    linestyle: str = "--",
    alpha: Optional[float] = None,
) -> None:
    """Add horizontal zero reference line to axis."""
    if linewidth is None:
        linewidth = plot_cfg.style.line.width_thin
    if alpha is None:
        alpha = plot_cfg.style.line.alpha_zero
    
    ax.axhline(
        0,
        color=plot_cfg.style.colors.black,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )


def create_bar_plot(
    ax: Any,
    x_positions: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    ylabel: str,
    plot_cfg: Any,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    width: Optional[float] = None,
) -> None:
    """Create a bar plot with standard styling and zero reference line."""
    if color is None:
        color = plot_cfg.style.colors.gray
    if alpha is None:
        alpha = plot_cfg.style.bar.alpha
    if width is None:
        width = plot_cfg.style.bar.width
    
    ax.bar(x_positions, values, color=color, alpha=alpha, width=width)
    add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    despine(ax)


def _validate_prediction_dataframe(
    pred_df: pd.DataFrame,
    model_name: str,
) -> bool:
    """Validate prediction dataframe has required columns and data."""
    if pred_df is None or len(pred_df) == 0:
        logger.warning(
            f"Empty prediction dataframe for {model_name} residual diagnostics"
        )
        return False
    
    required_columns = ["y_true", "y_pred"]
    missing_columns = [col for col in required_columns if col not in pred_df.columns]
    if missing_columns:
        logger.warning(
            f"Missing required columns {missing_columns} for {model_name} "
            "residual diagnostics"
        )
        return False
    
    return True


def _extract_residuals(pred_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite residuals from prediction dataframe."""
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    y_true_finite, y_pred_finite, _ = extract_finite_mask(y_true, y_pred)
    residuals = y_true_finite - y_pred_finite
    return residuals, y_pred_finite


def _plot_residuals_vs_predicted(
    ax: Any,
    y_pred_finite: np.ndarray,
    residuals: np.ndarray,
    plot_cfg: Any,
    marker_size: float,
) -> None:
    """Plot residuals against predicted values."""
    ax.scatter(
        y_pred_finite,
        residuals,
        s=marker_size,
        alpha=plot_cfg.style.scatter.alpha,
        c=plot_cfg.style.colors.gray,
        edgecolors="none",
    )
    add_zero_reference_line(
        ax,
        plot_cfg,
        linewidth=plot_cfg.style.line.width_standard,
        alpha=plot_cfg.style.line.alpha_reference,
    )
    ax.set_xlabel("Predicted Rating")
    ax.set_ylabel("Residual")
    despine(ax)


def _get_quantile_range(plot_cfg: Any) -> tuple[float, float]:
    """Get quantile range from configuration."""
    machine_learning_config = _get_machine_learning_config(plot_cfg)
    quantile_min = machine_learning_config.get("quantile_min", 0.01)
    quantile_max = machine_learning_config.get("quantile_max", 0.99)
    return quantile_min, quantile_max


def _calculate_theoretical_quantiles(
    residuals: np.ndarray,
    plot_cfg: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate theoretical quantiles for Q-Q plot."""
    residuals_sorted = np.sort(residuals)
    quantile_min, quantile_max = _get_quantile_range(plot_cfg)
    n_samples = len(residuals_sorted)
    quantile_range = np.linspace(quantile_min, quantile_max, n_samples)
    theoretical_quantiles = stats.norm.ppf(quantile_range)
    return theoretical_quantiles, residuals_sorted


def _plot_qq_plot(
    ax: Any,
    theoretical_quantiles: np.ndarray,
    sample_quantiles: np.ndarray,
    plot_cfg: Any,
    marker_size: float,
) -> None:
    """Plot Q-Q plot with reference line."""
    ax.scatter(
        theoretical_quantiles,
        sample_quantiles,
        s=marker_size,
        alpha=plot_cfg.style.scatter.alpha,
        c=plot_cfg.style.colors.gray,
        edgecolors="none",
    )
    
    lim_min, lim_max = calculate_shared_axis_limits(
        theoretical_quantiles, sample_quantiles, plot_cfg
    )
    reference_limits = [lim_min, lim_max]
    ax.plot(
        reference_limits,
        reference_limits,
        color=plot_cfg.style.colors.black,
        linestyle="--",
        linewidth=plot_cfg.style.line.width_standard,
        alpha=plot_cfg.style.line.alpha_reference,
    )
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    despine(ax)


def plot_residual_diagnostics(
    pred_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
    config: Optional[Any] = None,
) -> None:
    """Plot residual diagnostics: residuals vs predicted values and Q-Q plot."""
    if not _validate_prediction_dataframe(pred_df, model_name):
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="machine_learning")
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="machine_learning")
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    residuals, y_pred_finite = _extract_residuals(pred_df)
    
    _plot_residuals_vs_predicted(
        axes[0], y_pred_finite, residuals, plot_cfg, marker_size
    )
    
    theoretical_quantiles, residuals_sorted = _calculate_theoretical_quantiles(
        residuals, plot_cfg
    )
    _plot_qq_plot(axes[1], theoretical_quantiles, residuals_sorted, plot_cfg, marker_size)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info(f"Saved {model_name} residual diagnostics: {save_path}")
