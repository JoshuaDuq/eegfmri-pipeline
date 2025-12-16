from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.data.decoding import extract_channel_importance_from_coefficients, extract_importance_column

logger = logging.getLogger(__name__)


###################################################################
# Helper Functions (imported from helpers module)
###################################################################

from eeg_pipeline.plotting.decoding.helpers import (
    _despine,
    _add_zero_reference_line,
)


###################################################################
# Feature Importance Plots
###################################################################

def plot_feature_importance_top_n(importance_df: pd.DataFrame, model_name: str, save_path: Path, top_n: int = 20, config: Optional[Any] = None) -> None:
    """
    Plot top N most important features from a feature importance dataframe.
    """
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
    _despine(ax)
    
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info(f"Saved {model_name} top {top_n} features: {save_path}")


def _plot_violin_stability(df: pd.DataFrame, channels_ordered: list, model_name: str, plot_cfg):
    """
    Helper function to create violin plot for feature importance stability.
    """
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
    _despine(ax)
    
    return fig


def _plot_ridge_stability(df: pd.DataFrame, channels_ordered: list, model_name: str, plot_cfg):
    """
    Helper function to create ridge plot for feature importance stability.
    """
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
    """
    Plot feature importance stability across cross-validation folds.
    
    Creates violin or ridge plots showing the distribution of feature importance
    values across folds for each channel/feature.
    """
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

