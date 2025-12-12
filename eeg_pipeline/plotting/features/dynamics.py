"""
Temporal Dynamics Visualization
================================

Visualizations for autocorrelation, DFA, and multiscale entropy features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.plotting import save_fig
from eeg_pipeline.utils.io.paths import ensure_dir

if TYPE_CHECKING:
    import mne


def plot_autocorrelation_decay(
    dynamics_df: pd.DataFrame,
    save_path: Path,
    *,
    decay_col: str = "acf_decay_time_ms",
    figsize: Tuple[float, float] = (10, 5),
    by_condition: str = None,
    config: Any = None,
) -> plt.Figure:
    """Plot distribution of autocorrelation decay times."""
    plot_cfg = get_plot_config(config)
    
    if decay_col not in dynamics_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Column {decay_col} not found", ha="center", va="center")
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    values = dynamics_df[decay_col].dropna().values
    
    if by_condition and by_condition in dynamics_df.columns:
        conditions = dynamics_df[by_condition].unique()
        colors = ["#C42847", "#3B82F6"]
        for idx, cond in enumerate(conditions[:2]):
            cond_vals = dynamics_df[dynamics_df[by_condition] == cond][decay_col].dropna()
            ax1.hist(cond_vals, bins=20, alpha=0.6, label=str(cond), color=colors[idx])
        ax1.legend(fontsize=10)
    else:
        ax1.hist(values, bins=25, color="#3B82F6", alpha=0.7, edgecolor="white")
    
    ax1.set_xlabel("Decay Time (ms)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("ACF Decay Time Distribution", fontsize=12, fontweight="bold")
    ax1.axvline(np.median(values), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(values):.0f} ms")
    ax1.legend(fontsize=10)
    
    ax2 = axes[1]
    lag_cols = [c for c in dynamics_df.columns if "acf_lag_" in c]
    
    if lag_cols:
        lag_values = []
        lag_labels = []
        for col in sorted(lag_cols):
            lag_values.append(dynamics_df[col].mean())
            lag_labels.append(col.replace("acf_lag_", "").replace("ms", ""))
        
        ax2.bar(range(len(lag_values)), lag_values, color="#22C55E", alpha=0.7, edgecolor="white")
        ax2.set_xticks(range(len(lag_values)))
        ax2.set_xticklabels([f"{l} ms" for l in lag_labels], fontsize=9)
        ax2.set_xlabel("Lag", fontsize=11)
        ax2.set_ylabel("Mean ACF", fontsize=11)
        ax2.set_title("ACF at Fixed Lags", fontsize=12, fontweight="bold")
        ax2.axhline(1/np.e, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="1/e threshold")
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No lag columns", ha="center", va="center")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_mse_complexity_curves(
    dynamics_df: pd.DataFrame,
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (10, 5),
    by_condition: str = None,
) -> plt.Figure:
    """Plot multiscale entropy curves across scales."""
    mse_cols = sorted([c for c in dynamics_df.columns if c.startswith("mse_scale_")])
    
    if not mse_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No MSE scale columns found", ha="center", va="center")
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract scales
    scales = [int(c.replace("mse_scale_", "")) for c in mse_cols]
    
    ax1 = axes[0]
    
    if by_condition and by_condition in dynamics_df.columns:
        conditions = dynamics_df[by_condition].unique()
        colors = ["#C42847", "#3B82F6"]
        for idx, cond in enumerate(conditions[:2]):
            cond_df = dynamics_df[dynamics_df[by_condition] == cond]
            means = [cond_df[c].mean() for c in mse_cols]
            sems = [cond_df[c].sem() for c in mse_cols]
            ax1.errorbar(scales, means, yerr=sems, marker="o", label=str(cond), color=colors[idx], capsize=3)
    else:
        means = [dynamics_df[c].mean() for c in mse_cols]
        sems = [dynamics_df[c].sem() for c in mse_cols]
        ax1.errorbar(scales, means, yerr=sems, marker="o", color="#6366F1", capsize=3)
    
    ax1.set_xlabel("Scale", fontsize=plot_cfg.font.title)
    ax1.set_ylabel("Sample Entropy", fontsize=plot_cfg.font.title)
    ax1.set_title("Multiscale Entropy Curve", fontsize=plot_cfg.font.title, fontweight="bold")
    ax1.set_xscale("log")
    if by_condition:
        ax1.legend(fontsize=plot_cfg.font.medium)
    
    # Right: Complexity index distribution
    ax2 = axes[1]
    if "mse_complexity_index" in dynamics_df.columns:
        ci_values = dynamics_df["mse_complexity_index"].dropna().values
        ax2.hist(ci_values, bins=25, color="#22C55E", alpha=0.7, edgecolor="white")
        ax2.axvline(np.median(ci_values), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(ci_values):.2f}")
        ax2.set_xlabel("Complexity Index", fontsize=plot_cfg.font.title)
        ax2.set_ylabel("Count", fontsize=plot_cfg.font.title)
        ax2.set_title("MSE Complexity Index", fontsize=plot_cfg.font.title, fontweight="bold")
        ax2.legend(fontsize=plot_cfg.font.medium)
    else:
        ax2.text(0.5, 0.5, "No complexity index", ha="center", va="center")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_neural_timescale_comparison(
    dynamics_df: pd.DataFrame,
    save_path: Path,
    *,
    decay_col: str = "acf_decay_time_ms",
    dfa_col: str = "dfa_alpha_broadband",
    by_condition: str = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Scatter plot comparing different timescale measures."""
    if decay_col not in dynamics_df.columns or dfa_col not in dynamics_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = dynamics_df[decay_col].values
    y = dynamics_df[dfa_col].values
    
    valid = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid], y[valid]
    
    if by_condition and by_condition in dynamics_df.columns:
        conditions = dynamics_df[by_condition].values[valid]
        unique_conds = np.unique(conditions)
        colors = ["#C42847", "#3B82F6"]
        for idx, cond in enumerate(unique_conds[:2]):
            mask = conditions == cond
            ax.scatter(x_clean[mask], y_clean[mask], c=colors[idx], alpha=0.6, s=40, label=str(cond))
        ax.legend(fontsize=plot_cfg.font.medium)
    else:
        ax.scatter(x_clean, y_clean, c="#3B82F6", alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
    
    # Correlation
    r, p = stats.spearmanr(x_clean, y_clean)
    ax.set_xlabel("ACF Decay Time (ms)", fontsize=plot_cfg.font.title)
    ax.set_ylabel("DFA Exponent (α)", fontsize=plot_cfg.font.title)
    ax.set_title(f"Neural Timescale Comparison (r = {r:.2f})", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_dynamics_behavior_grid(
    dynamics_df: pd.DataFrame,
    behavior: np.ndarray,
    save_path: Path,
    *,
    dynamics_cols: List[str] = None,
    behavior_label: str = "Rating",
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Grid of dynamics features vs behavior correlations."""
    if dynamics_cols is None:
        dynamics_cols = [c for c in dynamics_df.columns 
                        if any(k in c for k in ["acf_", "dfa_", "mse_"]) and not c.endswith("_ms")]
    
    dynamics_cols = [c for c in dynamics_cols if c in dynamics_df.columns][:9]  # Max 3x3
    
    if not dynamics_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No dynamics columns", ha="center", va="center")
        return fig
    
    n_cols_grid = 3
    n_rows = (len(dynamics_cols) + n_cols_grid - 1) // n_cols_grid
    
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    y = np.asarray(behavior).flatten()
    min_len = min(len(dynamics_df), len(y))
    y = y[:min_len]
    
    for idx, col in enumerate(dynamics_cols):
        ax = axes[idx]
        
        x = dynamics_df[col].values[:min_len]
        valid = np.isfinite(x) & np.isfinite(y)
        
        if np.sum(valid) < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        x_clean, y_clean = x[valid], y[valid]
        
        ax.scatter(x_clean, y_clean, c="#3B82F6", alpha=0.5, s=20, edgecolors="white", linewidths=0.3)
        
        r, p = stats.spearmanr(x_clean, y_clean)
        color = "#C42847" if p < 0.05 else "#666666"
        
        # Regression line
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=1.5)
        
        ax.set_title(f"{col[:20]}... r={r:.2f}", fontsize=plot_cfg.font.medium, color=color)
        ax.tick_params(labelsize=7)
    
    for idx in range(len(dynamics_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Dynamics Features vs {behavior_label}", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig

