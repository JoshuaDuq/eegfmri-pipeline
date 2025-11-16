from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from eeg_pipeline.utils.io_utils import save_fig
from eeg_pipeline.utils.plotting_config import get_plot_config
from eeg_pipeline.utils.stats_utils import (
    extract_finite_mask,
    fit_linear_regression,
    compute_binned_statistics,
    compute_error_bars_from_ci_dicts,
    format_p_value,
)
from eeg_pipeline.utils.data_loading import (
    extract_channel_importance_from_coefficients,
    extract_importance_column,
)

logger = logging.getLogger(__name__)


###################################################################
# Plotting Configuration
###################################################################

def configure_plotting(config: Optional[Any] = None):
    plot_cfg = get_plot_config(config)
    plt.rcParams.update({
        'font.size': plot_cfg.font.medium,
        'axes.labelsize': plot_cfg.font.large,
        'axes.titlesize': plot_cfg.font.large,
        'xtick.labelsize': plot_cfg.font.medium,
        'ytick.labelsize': plot_cfg.font.medium,
        'legend.fontsize': plot_cfg.font.small,
        'figure.dpi': plot_cfg.dpi,
        'savefig.dpi': plot_cfg.dpi,
        'savefig.bbox': plot_cfg.bbox_inches,
        'savefig.pad_inches': plot_cfg.pad_inches,
        'pdf.fonttype': plot_cfg.plot_type_configs.get("decoding", {}).get("pdf_font_type", 42),
        'ps.fonttype': plot_cfg.plot_type_configs.get("decoding", {}).get("ps_font_type", 42),
        'axes.linewidth': plot_cfg.style.line.width_standard,
        'xtick.major.width': plot_cfg.style.line.width_standard,
        'ytick.major.width': plot_cfg.style.line.width_standard,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_palette('muted')

def despine(ax):
    sns.despine(ax=ax, trim=True)


###################################################################
# Helper Functions
###################################################################

def _calculate_axis_limits(values: np.ndarray, plot_cfg, margin_factor: Optional[float] = None) -> tuple[float, float]:
    if margin_factor is None:
        margin_factor = plot_cfg.plot_type_configs.get("decoding", {}).get("axis_margin_factor", 0.05)
    value_min = np.nanmin(values)
    value_max = np.nanmax(values)
    margin = (value_max - value_min) * margin_factor
    return value_min - margin, value_max + margin


def _calculate_shared_axis_limits(values1: np.ndarray, values2: np.ndarray, plot_cfg,
                                   margin_factor: Optional[float] = None) -> tuple[float, float]:
    combined_values = np.concatenate([values1, values2])
    return _calculate_axis_limits(combined_values, plot_cfg, margin_factor)

def _add_zero_reference_line(ax, plot_cfg, linewidth: Optional[float] = None, 
                             linestyle: str = '--', alpha: Optional[float] = None) -> None:
    if linewidth is None:
        linewidth = plot_cfg.style.line.width_thin
    if alpha is None:
        alpha = plot_cfg.style.line.alpha_zero
    ax.axhline(0, color=plot_cfg.style.colors.black, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

def _create_bar_plot(ax, x_positions: np.ndarray, values: np.ndarray, labels: list,
                     ylabel: str, plot_cfg, color: Optional[str] = None, 
                     alpha: Optional[float] = None, width: Optional[float] = None, 
                     add_zero_line: bool = True) -> None:
    if color is None:
        color = plot_cfg.style.colors.gray
    if alpha is None:
        alpha = plot_cfg.style.bar.alpha
    if width is None:
        width = plot_cfg.style.bar.width
    ax.bar(x_positions, values, color=color, alpha=alpha, width=width)
    if add_zero_line:
        _add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    despine(ax)



###################################################################
# Prediction Plots
###################################################################

def plot_prediction_scatter(pred_df: pd.DataFrame, model_name: str, pooled_metrics: dict, save_path: Path, config: Optional[Any] = None) -> None:
    if pred_df is None or len(pred_df) == 0:
        logger.warning(f"Empty prediction dataframe for {model_name}")
        return
    
    required_columns = ['y_true', 'y_pred']
    if not all(col in pred_df.columns for col in required_columns):
        logger.warning(f"Missing required columns in prediction dataframe for {model_name}")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("square", plot_type="decoding")
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="decoding")
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values
    
    ax.scatter(y_true, y_pred, s=marker_size, alpha=plot_cfg.style.scatter.alpha, 
               c=plot_cfg.style.colors.gray, edgecolors='none')
    
    lim_min, lim_max = _calculate_shared_axis_limits(y_true, y_pred, plot_cfg)
    axis_limits = [lim_min, lim_max]
    
    ax.plot(axis_limits, axis_limits, plot_cfg.style.colors.black, 
            linewidth=plot_cfg.style.line.width_standard, 
            alpha=plot_cfg.style.line.alpha_diagonal, zorder=1)
    
    min_samples = plot_cfg.validation.get("min_samples_for_fit", 2)
    y_true_finite, y_pred_finite, mask = extract_finite_mask(y_true, y_pred)
    if mask.sum() >= min_samples:
        fit_points = plot_cfg.plot_type_configs.get("decoding", {}).get("fit_points", 100)
        x_fit = np.linspace(axis_limits[0], axis_limits[1], fit_points)
        y_fit = fit_linear_regression(y_true_finite, y_pred_finite, x_fit, min_samples=min_samples)
        if np.any(np.isfinite(y_fit)):
            ax.plot(x_fit, y_fit, '-', color=plot_cfg.style.colors.red, 
                   linewidth=plot_cfg.style.line.width_thick, 
                   alpha=plot_cfg.style.line.alpha_fit, zorder=2)
    
    r = pooled_metrics.get('pearson_r', np.nan)
    r2 = pooled_metrics.get('r2', np.nan)
    p_val = pooled_metrics.get('p_value', np.nan)
    
    stats_text = f'r = {r:.3f}\nR² = {r2:.3f}\np = {p_val:.2e}'
    ax.text(plot_cfg.text_position.stats_x, plot_cfg.text_position.stats_y, stats_text, 
            transform=ax.transAxes, fontsize=plot_cfg.font.small, 
            verticalalignment='top', family='monospace')
    
    ax.set_xlabel('True Rating')
    ax.set_ylabel('Predicted Rating')
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect('equal')
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} prediction scatter: {save_path}")

def plot_per_subject_performance(per_subj_df: pd.DataFrame, model_name: str, save_path: Path, config: Optional[Any] = None) -> None:
    if per_subj_df is None or len(per_subj_df) == 0:
        logger.warning(f"Empty per-subject dataframe for {model_name}")
        return
    
    required_columns = ['group', 'pearson_r', 'r2']
    if not all(col in per_subj_df.columns for col in required_columns):
        logger.warning(f"Missing required columns in per-subject dataframe for {model_name}")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="decoding")
    
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    subjects = per_subj_df['group'].astype(str).values
    pearson_r = per_subj_df['pearson_r'].values
    r2 = per_subj_df['r2'].values
    
    x_positions = np.arange(len(subjects))
    
    _create_bar_plot(axes[0], x_positions, pearson_r, subjects.tolist(), "Pearson's r", plot_cfg)
    _create_bar_plot(axes[1], x_positions, r2, subjects.tolist(), 'R²', plot_cfg)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} per-subject performance: {save_path}")

def plot_residual_diagnostics(pred_df: pd.DataFrame, model_name: str, save_path: Path, config: Optional[Any] = None) -> None:
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
    _add_zero_reference_line(axes[0], plot_cfg, 
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
    
    lim_min, lim_max = _calculate_shared_axis_limits(theoretical_quantiles, residuals_sorted, plot_cfg)
    reference_limits = [lim_min, lim_max]
    axes[1].plot(reference_limits, reference_limits, color=plot_cfg.style.colors.black, linestyle='--',
                 linewidth=plot_cfg.style.line.width_standard, alpha=plot_cfg.style.line.alpha_reference)
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')
    despine(axes[1])
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} residual diagnostics: {save_path}")

def plot_model_comparison(models_dict: dict, save_path: Path, config: Optional[Any] = None) -> None:
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

def plot_calibration_curve(pred_df: pd.DataFrame, model_name: str, cal_metrics: dict, save_path: Path, config: Optional[Any] = None) -> None:
    if pred_df is None or len(pred_df) == 0:
        logger.warning(f"Empty prediction dataframe for {model_name} calibration curve")
        return
    
    required_columns = ['y_true', 'y_pred']
    if not all(col in pred_df.columns for col in required_columns):
        logger.warning(f"Missing required columns for {model_name} calibration curve")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("square", plot_type="decoding")
    
    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values
    y_true_finite, y_pred_finite, mask = extract_finite_mask(y_true, y_pred)
    
    min_samples = plot_cfg.validation.get("min_samples_for_calibration", 10)
    if len(y_true_finite) < min_samples:
        logger.warning(f"Insufficient samples ({len(y_true_finite)}) for {model_name} calibration curve")
        return
    
    max_bins = plot_cfg.validation.get("max_bins_for_calibration", 10)
    min_bins = plot_cfg.validation.get("min_bins_for_calibration", 3)
    samples_per_bin = plot_cfg.validation.get("samples_per_bin", 10)
    n_bins = min(max_bins, len(y_true_finite) // samples_per_bin)
    if n_bins < min_bins:
        logger.warning(f"Insufficient bins ({n_bins}) for {model_name} calibration curve")
        return
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    bin_centers, bin_means, bin_stds = compute_binned_statistics(y_pred_finite, y_true_finite, n_bins)
    
    if len(bin_centers) > 0:
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', 
                    markersize=plot_cfg.style.errorbar_markersize, color=plot_cfg.style.colors.red, 
                    linewidth=plot_cfg.style.line.width_standard, capsize=plot_cfg.style.errorbar_capsize)
    
    lim_min, lim_max = _calculate_shared_axis_limits(y_pred_finite, y_true_finite, plot_cfg)
    axis_limits = [lim_min, lim_max]
    
    ax.plot(axis_limits, axis_limits, color=plot_cfg.style.colors.black, linestyle='--',
            linewidth=plot_cfg.style.line.width_standard, 
            alpha=plot_cfg.style.line.alpha_diagonal, label='Perfect calibration')
    
    slope = cal_metrics.get('slope', np.nan)
    intercept = cal_metrics.get('intercept', np.nan)
    if np.isfinite(slope) and np.isfinite(intercept):
        x_fit = np.array(axis_limits)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '-', color=plot_cfg.style.colors.blue, 
               linewidth=plot_cfg.style.line.width_thick, 
               alpha=plot_cfg.style.line.alpha_fit, label='Fitted')
    
    stats_text = f'slope = {slope:.3f}\nintercept = {intercept:.3f}'
    ax.text(plot_cfg.text_position.stats_x, plot_cfg.text_position.stats_y, stats_text, 
            transform=ax.transAxes, fontsize=plot_cfg.font.small, 
            verticalalignment='top', family='monospace')
    
    ax.set_xlabel('Predicted Rating (binned)')
    ax.set_ylabel('True Rating (mean ± SE)')
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect('equal')
    ax.legend(fontsize=plot_cfg.font.small, loc='lower right', frameon=False)
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} calibration curve: {save_path}")

def _plot_bootstrap_metric(ax, point_value: float, ci: list, metric_name: str, plot_cfg, model_name: str = None):
    ax.axvline(point_value, color=plot_cfg.style.colors.red, linewidth=plot_cfg.style.line.width_bold, label='Observed')
    ax.axvline(ci[0], color=plot_cfg.style.colors.black, linewidth=plot_cfg.style.line.width_standard, 
               linestyle='--', alpha=plot_cfg.style.alpha_ci_line)
    ax.axvline(ci[1], color=plot_cfg.style.colors.black, linewidth=plot_cfg.style.line.width_standard, 
               linestyle='--', alpha=plot_cfg.style.alpha_ci_line)
    ax.axvspan(ci[0], ci[1], alpha=plot_cfg.style.alpha_ci, color=plot_cfg.style.colors.black, label='95% CI')
    ax.set_xlabel(metric_name)
    ax.set_ylabel('')
    if model_name:
        ax.text(plot_cfg.text_position.bootstrap_x, plot_cfg.text_position.bootstrap_y, model_name, 
                transform=ax.transAxes, fontsize=plot_cfg.font.medium, verticalalignment='top', weight='bold')
    ax.legend(fontsize=plot_cfg.font.small, loc='upper right', frameon=False)
    despine(ax)

def plot_bootstrap_distributions(bootstrap_results: dict, save_path: Path, config: Optional[Any] = None) -> None:
    if not bootstrap_results:
        logger.warning("Empty bootstrap results dictionary")
        return
    
    valid_models = [name for name, res in bootstrap_results.items() if res is not None]
    if len(valid_models) == 0:
        logger.warning("No valid bootstrap results found")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size_wide = plot_cfg.get_figure_size("wide", plot_type="decoding")
    
    n_models = len(valid_models)
    fig, axes = plt.subplots(n_models, 2, figsize=(fig_size_wide[0], fig_size_wide[1] * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for row, model_name in enumerate(valid_models):
        res = bootstrap_results[model_name]
        r_point = res.get('r_point', np.nan)
        r_ci = res.get('r_ci', [np.nan, np.nan])
        r2_point = res.get('r2_point', np.nan)
        r2_ci = res.get('r2_ci', [np.nan, np.nan])
        
        _plot_bootstrap_metric(axes[row, 0], r_point, r_ci, "Pearson's r", plot_cfg, model_name)
        _plot_bootstrap_metric(axes[row, 1], r2_point, r2_ci, 'R²', plot_cfg)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved bootstrap distributions: {save_path}")

def plot_permutation_null(null_rs: np.ndarray, observed_r: float, p_value: float, save_path: Path, config: Optional[Any] = None) -> None:
    if null_rs is None or len(null_rs) == 0:
        logger.warning("Empty null distribution array for permutation plot")
        return
    
    if not np.isfinite(observed_r) or not np.isfinite(p_value):
        logger.warning("Invalid observed_r or p_value for permutation plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("tall", plot_type="decoding")
    bins = plot_cfg.get_histogram_bins(plot_type="decoding")
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.hist(null_rs, bins=bins, color=plot_cfg.style.colors.light_gray, 
            alpha=plot_cfg.style.histogram.alpha, edgecolor='none', density=True)
    
    ax.axvline(observed_r, color=plot_cfg.style.colors.red, 
               linewidth=plot_cfg.style.line.width_bold, 
               label=f'Observed r = {observed_r:.3f}')
    
    p_text = format_p_value(p_value)
    ax.text(plot_cfg.text_position.p_value_x, plot_cfg.text_position.p_value_y, p_text, 
            transform=ax.transAxes, fontsize=plot_cfg.font.medium, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=plot_cfg.style.alpha_text_box, edgecolor='none'))
    
    ax.set_xlabel('Max |r| (null distribution)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=plot_cfg.font.small, loc='upper left', frameon=False)
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved permutation null distribution: {save_path}")



def plot_feature_importance_top_n(importance_df: pd.DataFrame, model_name: str, save_path: Path, top_n: int = 20, config: Optional[Any] = None) -> None:
    if importance_df is None or len(importance_df) == 0:
        logger.warning(f"Empty importance dataframe for {model_name}")
        return
    
    if 'feature_name' not in importance_df.columns:
        logger.warning(f"Missing 'feature_name' column for {model_name}")
        return
    
    values, ylabel = extract_importance_column(importance_df, top_n)
    if values is None:
        logger.warning(f"No importance column found in dataframe for {model_name}")
        return
    
    importance_column = 'mean_abs_shap' if 'mean_abs_shap' in importance_df.columns else 'importance'
    importance_df_sorted = importance_df.sort_values(importance_column, ascending=False).head(top_n)
    
    if len(importance_df_sorted) == 0:
        logger.warning(f"No features remaining after filtering for {model_name}")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size_tall = plot_cfg.get_figure_size("tall", plot_type="decoding")
    features = importance_df_sorted['feature_name'].values
    
    decoding_config = plot_cfg.plot_type_configs.get("decoding", {})
    feature_max_height = decoding_config.get("feature_plot_max_height", 6)
    feature_height_per_feature = decoding_config.get("feature_plot_height_per_feature", 0.3)
    fig_height = min(feature_max_height, feature_height_per_feature * len(features))
    fig, ax = plt.subplots(figsize=(fig_size_tall[0], fig_height))
    
    y_positions = np.arange(len(features))
    ax.barh(y_positions, values, color=plot_cfg.style.colors.gray, 
            alpha=plot_cfg.style.bar.alpha, height=plot_cfg.style.bar.width)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=plot_cfg.font.small)
    ax.set_xlabel(ylabel)
    ax.invert_yaxis()
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} top {top_n} features: {save_path}")

def plot_riemann_band_comparison(band_results: dict, save_path: Path, config: Optional[Any] = None) -> None:
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
    despine(axes[0])
    
    axes[1].plot(time_centers, r2_values, 'o-', color=plot_cfg.style.colors.gray, 
                 linewidth=plot_cfg.style.line.width_thick, markersize=marker_size)
    _add_zero_reference_line(axes[1], plot_cfg)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('R²')
    despine(axes[1])
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved Riemann sliding window: {save_path}")

def plot_incremental_validity(inc_summary: dict, save_path: Path, config: Optional[Any] = None) -> None:
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
    despine(ax)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved incremental validity: {save_path}")


def _plot_violin_stability(df: pd.DataFrame, channels_ordered: list, model_name: str, plot_cfg):
    decoding_config = plot_cfg.plot_type_configs.get("decoding", {})
    violin_min_width = decoding_config.get("violin_plot_min_width", 8)
    violin_width_per_channel = decoding_config.get("violin_plot_width_per_channel", 0.3)
    violin_height = decoding_config.get("violin_plot_height", 5)
    violin_width = decoding_config.get("violin_width", 0.7)
    
    fig_width = max(violin_min_width, len(channels_ordered) * violin_width_per_channel)
    fig, ax = plt.subplots(figsize=(fig_width, violin_height))
    
    positions = np.arange(len(channels_ordered))
    data_by_channel = [df[df['channel'] == ch]['importance'].values for ch in channels_ordered]
    
    parts = ax.violinplot(data_by_channel, positions=positions, widths=violin_width, 
                         showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(plot_cfg.style.colors.gray)
        pc.set_alpha(plot_cfg.style.alpha_violin_body)
    
    parts['cmeans'].set_color(plot_cfg.style.colors.red)
    parts['cmeans'].set_linewidth(plot_cfg.style.line.width_bold)
    parts['cmedians'].set_color(plot_cfg.style.colors.blue)
    parts['cmedians'].set_linewidth(plot_cfg.style.line.width_thick)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(channels_ordered, rotation=45, ha='right', fontsize=plot_cfg.font.small)
    ax.set_ylabel('Feature Importance (|coefficient|)', fontsize=plot_cfg.font.large)
    ax.set_title(f'{model_name}: Feature Importance Stability Across Folds', 
                 fontsize=plot_cfg.font.title, fontweight='bold')
    _add_zero_reference_line(ax, plot_cfg)
    despine(ax)
    
    return fig

def _plot_ridge_stability(df: pd.DataFrame, channels_ordered: list, model_name: str, plot_cfg):
    from scipy.stats import gaussian_kde
    
    decoding_config = plot_cfg.plot_type_configs.get("decoding", {})
    ridge_max_channels = decoding_config.get("ridge_plot_max_channels", 30)
    ridge_min_height = decoding_config.get("ridge_plot_min_height", 4)
    ridge_height_per_channel = decoding_config.get("ridge_plot_height_per_channel", 0.4)
    ridge_y_offset = decoding_config.get("ridge_plot_y_offset", 0.2)
    ridge_kde_points = decoding_config.get("ridge_kde_points", 200)
    
    n_channels = len(channels_ordered)
    if n_channels > ridge_max_channels:
        logger.info(f"Limiting ridge plot to top {ridge_max_channels} channels (requested {n_channels})")
        channels_ordered = channels_ordered[:ridge_max_channels]
        df = df[df['channel'].isin(channels_ordered)]
    
    fig_height = max(ridge_min_height, len(channels_ordered) * ridge_height_per_channel)
    tall_size = plot_cfg.get_figure_size("tall", plot_type="decoding")
    fig, axes = plt.subplots(len(channels_ordered), 1, figsize=(tall_size[0], fig_height), 
                            sharex=True, sharey=True)
    if len(channels_ordered) == 1:
        axes = [axes]
    
    x_range = np.linspace(df['importance'].min(), df['importance'].max(), ridge_kde_points)
    
    min_samples_for_fit = plot_cfg.validation.get("min_samples_for_fit", 2)
    for idx, channel in enumerate(channels_ordered):
        ch_data = df[df['channel'] == channel]['importance'].values
        ch_data = ch_data[np.isfinite(ch_data)]
        
        if len(ch_data) < min_samples_for_fit:
            ridge_text_x = decoding_config.get("ridge_text_x", 0.5)
            ridge_text_y = decoding_config.get("ridge_text_y", 0.5)
            axes[idx].text(ridge_text_x, ridge_text_y, 'Insufficient data', 
                         transform=axes[idx].transAxes, ha='center', va='center', 
                         fontsize=plot_cfg.font.small)
            axes[idx].set_yticks([])
            axes[idx].spines['left'].set_visible(False)
            continue
        
        try:
            kde = gaussian_kde(ch_data)
            y_kde = kde(x_range)
            y_kde = y_kde / y_kde.max()
        except Exception:
            y_kde = np.zeros_like(x_range)
        
        axes[idx].fill_between(x_range, idx, idx + y_kde, color=plot_cfg.style.colors.gray, 
                              alpha=plot_cfg.style.alpha_ridge_fill)
        axes[idx].axvline(np.mean(ch_data), color=plot_cfg.style.colors.red, 
                         linewidth=plot_cfg.style.line.width_thick, 
                         linestyle='--', alpha=plot_cfg.style.line.alpha_fit)
        axes[idx].axvline(np.median(ch_data), color=plot_cfg.style.colors.blue, 
                         linewidth=plot_cfg.style.line.width_standard, 
                         linestyle=':', alpha=plot_cfg.style.line.alpha_fit)
        
        axes[idx].set_yticks([idx + 0.5])
        axes[idx].set_yticklabels([channel], fontsize=plot_cfg.font.small)
        axes[idx].set_ylim([-ridge_y_offset, len(channels_ordered) + ridge_y_offset])
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['bottom'].set_visible(False)
        if idx < len(channels_ordered) - 1:
            axes[idx].spines['left'].set_visible(False)
            axes[idx].set_xticks([])
    
    axes[-1].set_xlabel('Feature Importance (|coefficient|)', fontsize=plot_cfg.font.large)
    axes[0].set_title(f'{model_name}: Feature Importance Stability Across Folds', 
                     fontsize=plot_cfg.font.title, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_feature_importance_stability(
    coef_matrix: np.ndarray,
    feature_names: list,
    model_name: str,
    save_path: Path,
    config: Optional[Any] = None,
    plot_type: str = "violin",
    top_n_channels: Optional[int] = None,
) -> None:
    if coef_matrix is None or coef_matrix.size == 0:
        logger.warning(f"No coefficients provided for {model_name} stability plot")
        return
    
    if len(feature_names) != coef_matrix.shape[1]:
        logger.warning(f"Feature name count ({len(feature_names)}) != coefficient matrix columns ({coef_matrix.shape[1]})")
        return
    
    plot_cfg = get_plot_config(config)
    n_folds = coef_matrix.shape[0]
    min_samples_for_fit = plot_cfg.validation.get("min_samples_for_fit", 2)
    if n_folds < min_samples_for_fit:
        logger.warning(f"Insufficient folds ({n_folds}) for stability visualization")
        return
    
    df = extract_channel_importance_from_coefficients(coef_matrix, feature_names)
    
    if len(df) == 0:
        logger.warning(f"No parseable power features found for {model_name} stability plot")
        return
    
    channel_means = df.groupby('channel')['importance'].mean().sort_values(ascending=False)
    if top_n_channels is not None and top_n_channels > 0:
        top_channels = channel_means.head(top_n_channels).index.tolist()
        df = df[df['channel'].isin(top_channels)]
    
    if len(df) == 0:
        logger.warning(f"No data remaining after filtering for {model_name} stability plot")
        return
    
    channels_ordered = df.groupby('channel')['importance'].mean().sort_values(ascending=False).index.tolist()
    
    if plot_type == "violin":
        fig = _plot_violin_stability(df, channels_ordered, model_name, plot_cfg)
    elif plot_type == "ridge":
        fig = _plot_ridge_stability(df, channels_ordered, model_name, plot_cfg)
    else:
        logger.warning(f"Unknown plot_type '{plot_type}', using violin plot")
        return plot_feature_importance_stability(coef_matrix, feature_names, model_name, save_path, config,
                                                 plot_type="violin", top_n_channels=top_n_channels)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} feature importance stability ({plot_type}): {save_path}")





