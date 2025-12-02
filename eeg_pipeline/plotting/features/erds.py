"""
Power Dynamics Visualization
=============================

Clean, publication-quality visualizations for power dynamics features.
Uses violin/strip plots for distributions, shows individual data points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.utils.io.general import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import get_band_names, get_band_colors


###################################################################
# Power Dynamics Distribution Plots
###################################################################


def plot_erds_temporal_evolution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (12, 5),
    config: Any = None,
) -> plt.Figure:
    """Mean active power vs baseline logratio by band."""
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    plot_cfg = get_plot_config(config)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_mean_active" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("Mean Active Power")
    ax.set_title("Active Period Power")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_logratio" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("Log Ratio (Active/Baseline)")
    ax.set_title("Power Change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle("Power Dynamics by Band", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_erds_latency_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Burst rate and duration across frequency bands."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_burst_rate" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        for i, (pos, vals) in enumerate(zip(positions, data_list)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("Burst Count")
    ax.set_title("Burst Rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_burst_mean_duration" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values * 1000
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        for i, (pos, vals) in enumerate(zip(positions, data_list)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Burst Duration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle("Burst Characteristics", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_erds_erd_ers_separation(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Burst amplitude and power variability (Fano factor)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(BANDS):
        cols = [c for c in features_df.columns if f"dynamics_{band}_burst_mean_amplitude" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(BAND_COLORS[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([b.capitalize() for b in BANDS])
    ax.set_xlabel("Band")
    ax.set_ylabel("Amplitude")
    ax.set_title("Burst Amplitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_power_fano" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.axhline(1, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("Fano Factor")
    ax.set_title("Power Variability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle("Burst Amplitude and Variability", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_erds_global_summary(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Power change and GFP summary by band."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_logratio" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                pct = (np.power(10, vals) - 1) * 100
                data_list.append(pct)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("Power Change (%)")
    ax.set_title("Power Change vs Baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"gfp_{band}_mean_active" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel("GFP")
    ax.set_title("Global Field Power")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle("Power Dynamics Summary", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


__all__ = [
    "plot_erds_distribution",
    "plot_erds_temporal_evolution",
    "plot_erds_latency_distribution",
    "plot_erds_erd_ers_separation",
    "plot_erds_global_summary",
]
