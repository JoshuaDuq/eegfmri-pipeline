from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.stats import (
    extract_finite_mask,
    fit_linear_regression,
    compute_binned_statistics,
    format_p_value,
)

logger = logging.getLogger(__name__)


from eeg_pipeline.plotting.machine_learning.helpers import (
    despine,
    calculate_shared_axis_limits,
    create_bar_plot,
    _validate_prediction_dataframe,
)


def plot_prediction_scatter(
    pred_df: pd.DataFrame,
    model_name: str,
    pooled_metrics: dict,
    save_path: Path,
    config: Optional[Any] = None,
) -> None:
    """Plot predicted vs true values scatter plot with performance metrics."""
    if not _validate_prediction_dataframe(pred_df, model_name):
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("square", plot_type="machine_learning")
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="machine_learning")
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values
    
    ax.scatter(
        y_true,
        y_pred,
        s=marker_size,
        alpha=plot_cfg.style.scatter.alpha,
        c=plot_cfg.style.colors.gray,
        edgecolors='none',
    )
    
    lim_min, lim_max = calculate_shared_axis_limits(y_true, y_pred, plot_cfg)
    axis_limits = [lim_min, lim_max]
    
    ax.plot(
        axis_limits,
        axis_limits,
        color=plot_cfg.style.colors.black,
        linewidth=plot_cfg.style.line.width_standard,
        alpha=plot_cfg.style.line.alpha_diagonal,
        zorder=1,
    )
    
    min_samples = plot_cfg.validation.get("min_samples_for_fit", 2)
    y_true_finite, y_pred_finite, mask = extract_finite_mask(y_true, y_pred)
    
    if mask.sum() >= min_samples:
        machine_learning_config = plot_cfg.plot_type_configs.get("machine_learning", {})
        fit_points = machine_learning_config.get("fit_points", 100)
        x_fit = np.linspace(axis_limits[0], axis_limits[1], fit_points)
        y_fit = fit_linear_regression(
            y_true_finite, y_pred_finite, x_fit, min_samples=min_samples
        )
        
        if np.any(np.isfinite(y_fit)):
            ax.plot(
                x_fit,
                y_fit,
                '-',
                color=plot_cfg.style.colors.red,
                linewidth=plot_cfg.style.line.width_thick,
                alpha=plot_cfg.style.line.alpha_fit,
                zorder=2,
            )
    
    pearson_r = pooled_metrics.get('pearson_r', np.nan)
    r2 = pooled_metrics.get('r2', np.nan)
    p_value = pooled_metrics.get('p_value', np.nan)
    
    stats_text = f'r = {pearson_r:.3f}\nR² = {r2:.3f}\np = {p_value:.2e}'
    ax.text(
        plot_cfg.text_position.stats_x,
        plot_cfg.text_position.stats_y,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_cfg.font.small,
        verticalalignment='top',
        family='monospace',
    )
    
    ax.set_xlabel('True Rating')
    ax.set_ylabel('Predicted Rating')
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect('equal')
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info(f"Saved {model_name} prediction scatter: {save_path}")


def plot_per_subject_performance(
    per_subj_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
    config: Optional[Any] = None,
) -> None:
    """Plot performance metrics (Pearson's r and R²) per subject."""
    if per_subj_df is None or len(per_subj_df) == 0:
        logger.warning(f"Empty per-subject dataframe for {model_name}")
        return
    
    required_columns = ['group', 'pearson_r', 'r2']
    missing_columns = [
        col for col in required_columns if col not in per_subj_df.columns
    ]
    if missing_columns:
        logger.warning(
            f"Missing required columns {missing_columns} in per-subject "
            f"dataframe for {model_name}"
        )
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="machine_learning")
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    subjects = per_subj_df['group'].astype(str).values
    pearson_r = per_subj_df['pearson_r'].values
    r2 = per_subj_df['r2'].values
    
    x_positions = np.arange(len(subjects))
    subject_labels = subjects.tolist()
    
    create_bar_plot(
        axes[0], x_positions, pearson_r, subject_labels, "Pearson's r", plot_cfg
    )
    create_bar_plot(axes[1], x_positions, r2, subject_labels, 'R²', plot_cfg)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info(f"Saved {model_name} per-subject performance: {save_path}")


def plot_ml_null_hist(
    null_r: np.ndarray,
    empirical_r: float,
    save_path: Path,
    config: Optional[Any] = None,
    title: str = "LOSO null distribution",
) -> None:
    """Plot histogram of null distribution with empirical value marked."""
    if null_r is None or null_r.size == 0:
        logger.warning("Null distribution empty; skipping null histogram")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("standard", plot_type="machine_learning")
    bins = plot_cfg.get_histogram_bins(plot_type="machine_learning")
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.hist(
        null_r,
        bins=bins,
        color=plot_cfg.style.colors.gray,
        alpha=0.7,
        label="Null (shuffled)",
    )
    ax.axvline(
        empirical_r,
        color="crimson",
        linestyle="--",
        linewidth=plot_cfg.style.line.width_standard,
        label=f"Empirical r={empirical_r:.2f}",
    )
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    despine(ax)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info("Saved machine learning null histogram to %s", save_path)


def plot_calibration_curve(
    pred_df: pd.DataFrame,
    model_name: str,
    cal_metrics: dict,
    save_path: Path,
    config: Optional[Any] = None,
) -> None:
    """Plot calibration curve showing predicted vs true values in bins."""
    if not _validate_prediction_dataframe(pred_df, f"{model_name} calibration curve"):
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("square", plot_type="machine_learning")
    
    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values
    y_true_finite, y_pred_finite, _ = extract_finite_mask(y_true, y_pred)

    min_samples = plot_cfg.validation.get("min_samples_for_calibration", 10)
    if len(y_true_finite) < min_samples:
        logger.warning(
            f"Insufficient samples ({len(y_true_finite)}) for {model_name} "
            "calibration curve"
        )
        return
    
    max_bins = plot_cfg.validation.get("max_bins_for_calibration", 10)
    min_bins = plot_cfg.validation.get("min_bins_for_calibration", 3)
    samples_per_bin = plot_cfg.validation.get("samples_per_bin", 10)
    n_bins = min(max_bins, len(y_true_finite) // samples_per_bin)
    
    if n_bins < min_bins:
        logger.warning(
            f"Insufficient bins ({n_bins}) for {model_name} calibration curve"
        )
        return
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    bin_centers, bin_means, bin_stds = compute_binned_statistics(
        y_pred_finite, y_true_finite, n_bins
    )
    
    if len(bin_centers) > 0:
        ax.errorbar(
            bin_centers,
            bin_means,
            yerr=bin_stds,
            fmt='o',
            markersize=plot_cfg.style.errorbar_markersize,
            color=plot_cfg.style.colors.red,
            linewidth=plot_cfg.style.line.width_standard,
            capsize=plot_cfg.style.errorbar_capsize,
        )
    
    lim_min, lim_max = calculate_shared_axis_limits(
        y_pred_finite, y_true_finite, plot_cfg
    )
    axis_limits = [lim_min, lim_max]
    
    ax.plot(
        axis_limits,
        axis_limits,
        color=plot_cfg.style.colors.black,
        linestyle='--',
        linewidth=plot_cfg.style.line.width_standard,
        alpha=plot_cfg.style.line.alpha_diagonal,
        label='Perfect calibration',
    )
    
    slope = cal_metrics.get('slope', np.nan)
    intercept = cal_metrics.get('intercept', np.nan)
    
    if np.isfinite(slope) and np.isfinite(intercept):
        x_fit = np.array(axis_limits)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit,
            y_fit,
            '-',
            color=plot_cfg.style.colors.blue,
            linewidth=plot_cfg.style.line.width_thick,
            alpha=plot_cfg.style.line.alpha_fit,
            label='Fitted',
        )
    
    stats_text = f'slope = {slope:.3f}\nintercept = {intercept:.3f}'
    ax.text(
        plot_cfg.text_position.stats_x,
        plot_cfg.text_position.stats_y,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_cfg.font.small,
        verticalalignment='top',
        family='monospace',
    )
    
    ax.set_xlabel('Predicted Rating (binned)')
    ax.set_ylabel('True Rating (mean ± SE)')
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect('equal')
    ax.legend(fontsize=plot_cfg.font.small, loc='lower right', frameon=False)
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info(f"Saved {model_name} calibration curve: {save_path}")


def plot_permutation_null(
    null_rs: np.ndarray,
    observed_r: float,
    p_value: float,
    save_path: Path,
    config: Optional[Any] = None,
) -> None:
    """Plot permutation null distribution with observed value and p-value."""
    if null_rs is None or len(null_rs) == 0:
        logger.warning("Empty null distribution array for permutation plot")
        return
    
    if not np.isfinite(observed_r) or not np.isfinite(p_value):
        logger.warning("Invalid observed_r or p_value for permutation plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("tall", plot_type="machine_learning")
    bins = plot_cfg.get_histogram_bins(plot_type="machine_learning")
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.hist(
        null_rs,
        bins=bins,
        color=plot_cfg.style.colors.light_gray,
        alpha=plot_cfg.style.histogram.alpha,
        edgecolor='none',
        density=True,
    )
    
    ax.axvline(
        observed_r,
        color=plot_cfg.style.colors.red,
        linewidth=plot_cfg.style.line.width_bold,
        label=f'Observed r = {observed_r:.3f}',
    )
    
    p_text = format_p_value(p_value)
    text_bbox = dict(
        boxstyle='round',
        facecolor='white',
        alpha=plot_cfg.style.alpha_text_box,
        edgecolor='none',
    )
    ax.text(
        plot_cfg.text_position.p_value_x,
        plot_cfg.text_position.p_value_y,
        p_text,
        transform=ax.transAxes,
        fontsize=plot_cfg.font.medium,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=text_bbox,
    )
    
    ax.set_xlabel('Max |r| (null distribution)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=plot_cfg.font.small, loc='upper left', frameon=False)
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info("Saved permutation null distribution: %s", save_path)

