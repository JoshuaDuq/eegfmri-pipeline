"""
Moderation Visualization
========================

Simple slopes plots and Johnson-Neyman interval visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.utils.io.general import (
    save_fig,
    get_behavior_footer as _get_behavior_footer,
    get_default_logger as _get_default_logger,
    get_default_config as _get_default_config,
)
from eeg_pipeline.utils.analysis.stats.moderation import ModerationResult


###################################################################
# Simple Slopes Plot
###################################################################


def plot_simple_slopes(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    result: ModerationResult,
    output_path: Path,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    """Plot simple slopes showing effect of X on Y at different levels of W.
    
    Shows regression lines at W = mean - 1SD, mean, mean + 1SD.
    """
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    
    # Clean data
    mask = np.isfinite(X) & np.isfinite(W) & np.isfinite(Y)
    X_c = X[mask]
    W_c = W[mask]
    Y_c = Y[mask]
    
    W_mean = np.mean(W_c)
    W_std = np.std(W_c)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # X range for plotting regression lines
    x_range = np.linspace(np.min(X_c), np.max(X_c), 100)
    x_centered = x_range - np.mean(X_c)
    
    # Colors for low/mean/high W
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    labels = [f'Low {result.w_label} (-1SD)', f'Mean {result.w_label}', f'High {result.w_label} (+1SD)']
    w_values = [W_mean - W_std, W_mean, W_mean + W_std]
    slopes = [result.slope_low_w, result.slope_mean_w, result.slope_high_w]
    p_values = [result.p_slope_low, result.p_slope_mean, result.p_slope_high]
    
    # Plot reference: mean Y
    y_mean = np.mean(Y_c)
    
    for i, (color, label, w_val, slope, p_val) in enumerate(zip(colors, labels, w_values, slopes, p_values)):
        # Y at each x level: b0 + b1*x + b2*w + b3*x*w
        # With centering: Y = b0 + slope * x_centered  (at centered w)
        w_centered = w_val - W_mean
        y_line = result.b0 + slope * x_centered + result.b2 * w_centered
        
        linestyle = '-' if p_val < 0.05 else '--'
        sig_marker = '*' if p_val < 0.05 else ''
        
        ax.plot(x_range, y_line, color=color, linewidth=2.5, linestyle=linestyle,
                label=f'{label}: b={slope:.3f}{sig_marker}')
    
    # Add scatter points colored by W level
    w_groups = np.digitize(W_c, [W_mean - 0.5*W_std, W_mean + 0.5*W_std])
    group_colors = [colors[g] for g in w_groups]
    
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    ax.scatter(X_c, Y_c, c=group_colors, s=marker_size, alpha=0.4, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel(result.x_label, fontsize=plot_cfg.font.label)
    ax.set_ylabel(result.y_label, fontsize=plot_cfg.font.label)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotation box
    sig_text = "✓ Significant Interaction" if result.is_significant_moderation() else "✗ No Significant Interaction"
    sig_color = 'green' if result.is_significant_moderation() else 'red'
    
    annotation = (
        f"Interaction (b₃): {result.b3:.4f}, p={result.p_b3:.4f}\n"
        f"R²: {result.r_squared:.3f}, ΔR²: {result.r_squared_change:.3f}\n"
        f"n={result.n}"
    )
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.text(0.98, 0.02, sig_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            fontweight='bold', color=sig_color)
    
    if title is None:
        title = f"Simple Slopes: {result.x_label} × {result.w_label} → {result.y_label}"
    ax.set_title(title, fontsize=plot_cfg.font.title, fontweight='bold')
    
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
    logger.info(f"Simple slopes plot saved to {output_path}")


###################################################################
# Johnson-Neyman Plot
###################################################################


def plot_johnson_neyman(
    result: ModerationResult,
    W: np.ndarray,
    output_path: Path,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    """Plot Johnson-Neyman regions of significance.
    
    Shows the conditional effect of X on Y across the full range of W,
    highlighting regions where the effect is statistically significant.
    """
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    
    W_clean = W[np.isfinite(W)]
    W_mean = np.mean(W_clean)
    W_std = np.std(W_clean)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of W values to plot
    w_range = np.linspace(np.min(W_clean), np.max(W_clean), 200)
    w_centered = w_range - W_mean
    
    # Conditional slope at each W: b1 + b3 * (W - W_mean)
    slopes = result.b1 + result.b3 * w_centered
    
    # SE of conditional slope (approximate)
    se_slopes = np.sqrt(result.se_b1**2 + (w_centered**2) * result.se_b3**2)
    
    # 95% CI
    t_crit = 1.96
    ci_low = slopes - t_crit * se_slopes
    ci_high = slopes + t_crit * se_slopes
    
    # Plot conditional effect and CI band
    ax.plot(w_range, slopes, 'b-', linewidth=2, label='Conditional effect')
    ax.fill_between(w_range, ci_low, ci_high, alpha=0.2, color='blue', label='95% CI')
    
    # Zero reference line
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero effect')
    
    # Shade regions of significance
    if result.jn_type == "inside":
        # Significant inside the interval
        mask = (w_range >= result.jn_low) & (w_range <= result.jn_high)
        ax.fill_between(w_range, ax.get_ylim()[0], ax.get_ylim()[1],
                       where=mask, alpha=0.15, color='green', label='Significant region')
        ax.axvline(result.jn_low, color='green', linestyle=':', linewidth=1.5)
        ax.axvline(result.jn_high, color='green', linestyle=':', linewidth=1.5)
    elif result.jn_type == "outside":
        # Significant outside the interval
        mask_low = w_range < result.jn_low
        mask_high = w_range > result.jn_high
        ax.fill_between(w_range, ci_low, ci_high,
                       where=mask_low | mask_high, alpha=0.3, color='green', label='Significant region')
        ax.axvline(result.jn_low, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(result.jn_high, color='orange', linestyle=':', linewidth=1.5)
    
    # Mark W levels
    for w_val, label in [(W_mean - W_std, '-1SD'), (W_mean, 'Mean'), (W_mean + W_std, '+1SD')]:
        ax.axvline(w_val, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(w_val, ax.get_ylim()[1], label, ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel(f'{result.w_label} (Moderator)', fontsize=plot_cfg.font.label)
    ax.set_ylabel(f'Effect of {result.x_label} on {result.y_label}', fontsize=plot_cfg.font.label)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Annotation
    if result.jn_type in ["inside", "outside"]:
        jn_text = f"J-N Interval: [{result.jn_low:.2f}, {result.jn_high:.2f}]\n"
        if result.jn_type == "inside":
            jn_text += f"Effect significant when {result.w_label} is inside this range"
        else:
            jn_text += f"Effect significant when {result.w_label} is outside this range"
    elif result.jn_type == "always":
        jn_text = f"Effect of {result.x_label} is always significant across all {result.w_label}"
    else:
        jn_text = f"Effect of {result.x_label} is never significant across {result.w_label}"
    
    ax.text(0.02, 0.02, jn_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    if title is None:
        title = f"Johnson-Neyman Plot: {result.x_label} × {result.w_label}"
    ax.set_title(title, fontsize=plot_cfg.font.title, fontweight='bold')
    
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
    logger.info(f"Johnson-Neyman plot saved to {output_path}")


__all__ = [
    "plot_simple_slopes",
    "plot_johnson_neyman",
]
