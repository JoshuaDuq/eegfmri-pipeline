from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.data.machine_learning import (
    extract_channel_importance_from_coefficients,
    extract_importance_column,
)
from eeg_pipeline.plotting.machine_learning.helpers import (
    despine,
    add_zero_reference_line,
    _get_machine_learning_config,
)

logger = logging.getLogger(__name__)


def _get_importance_column_name(importance_df: pd.DataFrame) -> str:
    """Get the name of the importance column."""
    if 'mean_abs_shap' in importance_df.columns:
        return 'mean_abs_shap'
    return 'importance'


def _calculate_feature_plot_height(
    n_features: int,
    ml_config: dict[str, Any],
) -> float:
    """Calculate figure height for feature importance plot."""
    max_height = ml_config.get("feature_plot_max_height", 6)
    height_per_feature = ml_config.get("feature_plot_height_per_feature", 0.3)
    calculated_height = height_per_feature * n_features
    return min(max_height, calculated_height)


def plot_feature_importance_top_n(
    importance_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
    top_n: int = 20,
    config: Optional[Any] = None,
) -> None:
    """Plot top N most important features from a feature importance dataframe."""
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
    
    importance_column = _get_importance_column_name(importance_df)
    importance_df_sorted = importance_df.sort_values(
        importance_column, ascending=False
    ).head(top_n)
    
    if len(importance_df_sorted) == 0:
        logger.warning(f"No features remaining after filtering for {model_name}")
        return
    
    plot_cfg = get_plot_config(config)
    ml_config = _get_machine_learning_config(plot_cfg)
    fig_width = plot_cfg.get_figure_size("tall", plot_type="machine_learning")[0]
    features = importance_df_sorted['feature_name'].values
    fig_height = _calculate_feature_plot_height(len(features), ml_config)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y_positions = np.arange(len(features))
    
    ax.barh(
        y_positions,
        values,
        color=plot_cfg.style.colors.gray,
        alpha=plot_cfg.style.bar.alpha,
        height=plot_cfg.style.bar.width,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=plot_cfg.font.small)
    ax.set_xlabel(ylabel)
    ax.invert_yaxis()
    despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} top {top_n} features: {save_path}")


def _calculate_violin_figure_size(
    n_channels: int,
    ml_config: dict[str, Any],
) -> tuple[float, float]:
    """Calculate figure size for violin plot."""
    min_width = ml_config.get("violin_plot_min_width", 8)
    width_per_channel = ml_config.get("violin_plot_width_per_channel", 0.3)
    height = ml_config.get("violin_plot_height", 5)
    calculated_width = n_channels * width_per_channel
    width = max(min_width, calculated_width)
    return width, height


def _extract_channel_data(
    df: pd.DataFrame,
    channels_ordered: list[str],
) -> list[np.ndarray]:
    """Extract importance values for each channel in order."""
    return [df[df['channel'] == ch]['importance'].values for ch in channels_ordered]


def _style_violin_plot(parts: dict, plot_cfg: Any) -> None:
    """Apply styling to violin plot components."""
    for body in parts['bodies']:
        body.set_facecolor(plot_cfg.style.colors.gray)
        body.set_alpha(plot_cfg.style.alpha_violin_body)
    
    parts['cmeans'].set_color(plot_cfg.style.colors.red)
    parts['cmeans'].set_linewidth(plot_cfg.style.line.width_bold)
    parts['cmedians'].set_color(plot_cfg.style.colors.blue)
    parts['cmedians'].set_linewidth(plot_cfg.style.line.width_thick)


def _plot_violin_stability(
    df: pd.DataFrame,
    channels_ordered: list[str],
    model_name: str,
    plot_cfg: Any,
) -> plt.Figure:
    """Create violin plot for feature importance stability."""
    ml_config = _get_machine_learning_config(plot_cfg)
    fig_width, fig_height = _calculate_violin_figure_size(
        len(channels_ordered), ml_config
    )
    violin_width = ml_config.get("violin_width", 0.7)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    positions = np.arange(len(channels_ordered))
    data_by_channel = _extract_channel_data(df, channels_ordered)
    
    parts = ax.violinplot(
        data_by_channel,
        positions=positions,
        widths=violin_width,
        showmeans=True,
        showmedians=True,
    )
    _style_violin_plot(parts, plot_cfg)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(
        channels_ordered,
        rotation=45,
        ha='right',
        fontsize=plot_cfg.font.small,
    )
    ax.set_ylabel(
        'Feature Importance (|coefficient|)',
        fontsize=plot_cfg.font.large,
    )
    ax.set_title(
        f'{model_name}: Feature Importance Stability Across Folds',
        fontsize=plot_cfg.font.title,
        fontweight='bold',
    )
    add_zero_reference_line(ax, plot_cfg)
    despine(ax)
    
    return fig


def _limit_channels_for_ridge_plot(
    channels_ordered: list[str],
    df: pd.DataFrame,
    max_channels: int,
) -> tuple[list[str], pd.DataFrame]:
    """Limit channels to maximum for ridge plot."""
    n_channels = len(channels_ordered)
    if n_channels <= max_channels:
        return channels_ordered, df
    
    logger.info(
        f"Limiting ridge plot to top {max_channels} channels "
        f"(requested {n_channels})"
    )
    limited_channels = channels_ordered[:max_channels]
    filtered_df = df[df['channel'].isin(limited_channels)]
    return limited_channels, filtered_df


def _calculate_ridge_figure_size(
    n_channels: int,
    ml_config: dict[str, Any],
    plot_cfg: Any,
) -> tuple[float, float]:
    """Calculate figure size for ridge plot."""
    min_height = ml_config.get("ridge_plot_min_height", 4)
    height_per_channel = ml_config.get("ridge_plot_height_per_channel", 0.4)
    width = plot_cfg.get_figure_size("tall", plot_type="machine_learning")[0]
    calculated_height = n_channels * height_per_channel
    height = max(min_height, calculated_height)
    return width, height


def _compute_kde_for_channel(
    channel_data: np.ndarray,
    x_range: np.ndarray,
) -> np.ndarray:
    """Compute KDE for channel data."""
    try:
        kde = gaussian_kde(channel_data)
        y_kde = kde(x_range)
        normalized_kde = y_kde / y_kde.max()
        return normalized_kde
    except (ValueError, np.linalg.LinAlgError):
        return np.zeros_like(x_range)


def _plot_insufficient_data_message(
    ax: plt.Axes,
    ml_config: dict[str, Any],
    plot_cfg: Any,
) -> None:
    """Display message when insufficient data for KDE fitting."""
    text_x = ml_config.get("ridge_text_x", 0.5)
    text_y = ml_config.get("ridge_text_y", 0.5)
    ax.text(
        text_x,
        text_y,
        'Insufficient data',
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=plot_cfg.font.small,
    )
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)


def _plot_ridge_channel(
    ax: plt.Axes,
    channel_data: np.ndarray,
    x_range: np.ndarray,
    channel_name: str,
    channel_idx: int,
    n_channels: int,
    ml_config: dict[str, Any],
    plot_cfg: Any,
) -> None:
    """Plot a single channel in ridge plot."""
    y_kde = _compute_kde_for_channel(channel_data, x_range)
    y_offset = ml_config.get("ridge_plot_y_offset", 0.2)
    
    ax.fill_between(
        x_range,
        channel_idx,
        channel_idx + y_kde,
        color=plot_cfg.style.colors.gray,
        alpha=plot_cfg.style.alpha_ridge_fill,
    )
    
    mean_value = np.mean(channel_data)
    median_value = np.median(channel_data)
    
    ax.axvline(
        mean_value,
        color=plot_cfg.style.colors.red,
        linewidth=plot_cfg.style.line.width_thick,
        linestyle='--',
        alpha=plot_cfg.style.line.alpha_fit,
    )
    ax.axvline(
        median_value,
        color=plot_cfg.style.colors.blue,
        linewidth=plot_cfg.style.line.width_standard,
        linestyle=':',
        alpha=plot_cfg.style.line.alpha_fit,
    )
    
    ax.set_yticks([channel_idx + 0.5])
    ax.set_yticklabels([channel_name], fontsize=plot_cfg.font.small)
    ax.set_ylim([-y_offset, n_channels + y_offset])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    if channel_idx < n_channels - 1:
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])


def _plot_ridge_stability(
    df: pd.DataFrame,
    channels_ordered: list[str],
    model_name: str,
    plot_cfg: Any,
) -> plt.Figure:
    """Create ridge plot for feature importance stability."""
    ml_config = _get_machine_learning_config(plot_cfg)
    max_channels = ml_config.get("ridge_plot_max_channels", 30)
    kde_points = ml_config.get("ridge_kde_points", 200)
    min_samples_for_fit = plot_cfg.validation.get("min_samples_for_fit", 2)
    
    channels_ordered, df = _limit_channels_for_ridge_plot(
        channels_ordered, df, max_channels
    )
    n_channels = len(channels_ordered)
    
    fig_width, fig_height = _calculate_ridge_figure_size(
        n_channels, ml_config, plot_cfg
    )
    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
    )
    if n_channels == 1:
        axes = [axes]
    
    importance_min = df['importance'].min()
    importance_max = df['importance'].max()
    x_range = np.linspace(importance_min, importance_max, kde_points)
    
    for idx, channel in enumerate(channels_ordered):
        channel_data = df[df['channel'] == channel]['importance'].values
        channel_data = channel_data[np.isfinite(channel_data)]
        
        if len(channel_data) < min_samples_for_fit:
            _plot_insufficient_data_message(axes[idx], ml_config, plot_cfg)
            continue
        
        _plot_ridge_channel(
            axes[idx],
            channel_data,
            x_range,
            channel,
            idx,
            n_channels,
            ml_config,
            plot_cfg,
        )
    
    axes[-1].set_xlabel(
        'Feature Importance (|coefficient|)',
        fontsize=plot_cfg.font.large,
    )
    axes[0].set_title(
        f'{model_name}: Feature Importance Stability Across Folds',
        fontsize=plot_cfg.font.title,
        fontweight='bold',
    )
    plt.tight_layout()
    
    return fig


def _validate_coefficient_matrix(
    coef_matrix: np.ndarray,
    feature_names: list[str],
    model_name: str,
    plot_cfg: Any,
) -> bool:
    """Validate coefficient matrix for stability plotting."""
    if coef_matrix is None or coef_matrix.size == 0:
        logger.warning(f"No coefficients provided for {model_name} stability plot")
        return False
    
    if len(feature_names) != coef_matrix.shape[1]:
        logger.warning(
            f"Feature name count ({len(feature_names)}) != "
            f"coefficient matrix columns ({coef_matrix.shape[1]})"
        )
        return False
    
    n_folds = coef_matrix.shape[0]
    min_samples_for_fit = plot_cfg.validation.get("min_samples_for_fit", 2)
    if n_folds < min_samples_for_fit:
        logger.warning(
            f"Insufficient folds ({n_folds}) for stability visualization"
        )
        return False
    
    return True


def _prepare_stability_data(
    coef_matrix: np.ndarray,
    feature_names: list[str],
    model_name: str,
    top_n_channels: Optional[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Extract and prepare data for stability plotting."""
    df = extract_channel_importance_from_coefficients(coef_matrix, feature_names)
    
    if len(df) == 0:
        logger.warning(
            f"No parseable power features found for {model_name} stability plot"
        )
        return pd.DataFrame(), []
    
    channel_means = df.groupby('channel')['importance'].mean().sort_values(
        ascending=False
    )
    
    if top_n_channels is not None and top_n_channels > 0:
        top_channels = channel_means.head(top_n_channels).index.tolist()
        df = df[df['channel'].isin(top_channels)]
        channel_means = df.groupby('channel')['importance'].mean().sort_values(
            ascending=False
        )
    
    if len(df) == 0:
        logger.warning(
            f"No data remaining after filtering for {model_name} stability plot"
        )
        return pd.DataFrame(), []
    
    channels_ordered = channel_means.index.tolist()
    
    return df, channels_ordered


def plot_feature_importance_stability_violin(
    coef_matrix: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_path: Path,
    config: Optional[Any] = None,
    top_n_channels: Optional[int] = None,
) -> None:
    """Plot feature importance stability as violin plot across CV folds."""
    plot_cfg = get_plot_config(config)
    if not _validate_coefficient_matrix(coef_matrix, feature_names, model_name, plot_cfg):
        return
    
    df, channels_ordered = _prepare_stability_data(
        coef_matrix, feature_names, model_name, top_n_channels
    )
    if len(df) == 0:
        return
    
    fig = _plot_violin_stability(df, channels_ordered, model_name, plot_cfg)
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} feature importance stability (violin): {save_path}")


def plot_feature_importance_stability_ridge(
    coef_matrix: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_path: Path,
    config: Optional[Any] = None,
    top_n_channels: Optional[int] = None,
) -> None:
    """Plot feature importance stability as ridge plot across CV folds."""
    plot_cfg = get_plot_config(config)
    if not _validate_coefficient_matrix(coef_matrix, feature_names, model_name, plot_cfg):
        return
    
    df, channels_ordered = _prepare_stability_data(
        coef_matrix, feature_names, model_name, top_n_channels
    )
    if len(df) == 0:
        return
    
    fig = _plot_ridge_stability(df, channels_ordered, model_name, plot_cfg)
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} feature importance stability (ridge): {save_path}")

