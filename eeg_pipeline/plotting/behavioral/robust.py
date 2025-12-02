"""
Robust Statistics Visualization
================================

Visualizations for comparing standard vs robust correlations,
outlier influence analysis, and method comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir


def plot_outlier_influence(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    *,
    title: str = "Outlier Influence Analysis",
    x_label: str = "Feature",
    y_label: str = "Behavior",
    figsize: Tuple[float, float] = (12, 5),
    z_threshold: float = 3.0,
) -> plt.Figure:
    """Show correlation before/after outlier removal with influence metrics."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    valid = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid], y[valid]
    n = len(x_clean)
    
    if n < 5:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig
    
    # Identify outliers using Mahalanobis-like distance
    z_x = (x_clean - np.median(x_clean)) / (np.median(np.abs(x_clean - np.median(x_clean))) * 1.4826 + 1e-12)
    z_y = (y_clean - np.median(y_clean)) / (np.median(np.abs(y_clean - np.median(y_clean))) * 1.4826 + 1e-12)
    
    outlier_mask = (np.abs(z_x) > z_threshold) | (np.abs(z_y) > z_threshold)
    n_outliers = np.sum(outlier_mask)
    
    # Correlations
    r_all, p_all = stats.pearsonr(x_clean, y_clean)
    
    if n_outliers > 0 and n_outliers < n - 5:
        r_clean, p_clean = stats.pearsonr(x_clean[~outlier_mask], y_clean[~outlier_mask])
    else:
        r_clean, p_clean = r_all, p_all
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: All data
    ax1 = axes[0]
    ax1.scatter(x_clean, y_clean, c="#3B82F6", alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
    
    # Regression line
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r = {r_all:.3f}")
    
    ax1.set_xlabel(x_label, fontsize=9)
    ax1.set_ylabel(y_label, fontsize=9)
    ax1.set_title(f"All Data (N={n})", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    
    # Panel 2: Highlight outliers
    ax2 = axes[1]
    ax2.scatter(x_clean[~outlier_mask], y_clean[~outlier_mask], c="#3B82F6", alpha=0.6, s=40, label="Inliers")
    ax2.scatter(x_clean[outlier_mask], y_clean[outlier_mask], c="#EF4444", alpha=0.8, s=60, 
                marker="x", linewidths=2, label=f"Outliers (n={n_outliers})")
    
    ax2.set_xlabel(x_label, fontsize=9)
    ax2.set_ylabel(y_label, fontsize=9)
    ax2.set_title(f"Outlier Detection (|z| > {z_threshold})", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    
    # Panel 3: After removal
    ax3 = axes[2]
    if n_outliers > 0:
        ax3.scatter(x_clean[~outlier_mask], y_clean[~outlier_mask], c="#22C55E", alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
        
        # Regression line
        slope_clean, intercept_clean = np.polyfit(x_clean[~outlier_mask], y_clean[~outlier_mask], 1)
        x_line_clean = np.linspace(x_clean[~outlier_mask].min(), x_clean[~outlier_mask].max(), 100)
        ax3.plot(x_line_clean, slope_clean * x_line_clean + intercept_clean, "r-", linewidth=2, label=f"r = {r_clean:.3f}")
    else:
        ax3.scatter(x_clean, y_clean, c="#22C55E", alpha=0.6, s=40)
        ax3.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r = {r_all:.3f}")
    
    ax3.set_xlabel(x_label, fontsize=9)
    ax3.set_ylabel(y_label, fontsize=9)
    ax3.set_title(f"After Removal (N={n - n_outliers})", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    
    # Summary annotation
    delta_r = r_clean - r_all
    fig.suptitle(f"{title}   |   Δr = {delta_r:+.3f}", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_correlation_methods_comparison(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    title: str = "Correlation Method Comparison",
    figsize: Tuple[float, float] = (10, 8),
    max_features: int = 25,
) -> plt.Figure:
    """Compare correlations from different methods (Pearson, Spearman, Robust)."""
    # Expected columns: feature, r_pearson, r_spearman, r_robust (or similar)
    method_cols = [c for c in stats_df.columns if c.startswith("r_") or c == "correlation"]
    
    if len(method_cols) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Need multiple correlation methods", ha="center", va="center")
        return fig
    
    df = stats_df.head(max_features)
    n_features = len(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_features)
    width = 0.25
    
    colors = {"r_pearson": "#3B82F6", "r_spearman": "#22C55E", "r_robust": "#F59E0B", "correlation": "#6366F1"}
    
    for idx, method_col in enumerate(method_cols[:3]):  # Max 3 methods
        if method_col not in df.columns:
            continue
        
        offset = (idx - 1) * width
        values = df[method_col].fillna(0).values
        
        label = method_col.replace("r_", "").replace("correlation", "default").title()
        color = colors.get(method_col, f"C{idx}")
        
        ax.barh(y_positions + offset, values, height=width, label=label, color=color, alpha=0.8)
    
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    ax.set_yticks(y_positions)
    feature_col = "feature" if "feature" in df.columns else "identifier"
    ax.set_yticklabels(df[feature_col].values if feature_col in df.columns else range(n_features), fontsize=8)
    ax.set_xlabel("Correlation (r)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_bootstrap_ci_comparison(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    r_col: str = "correlation",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    feature_col: str = "identifier",
    title: str = "Bootstrap Confidence Intervals",
    figsize: Tuple[float, float] = (10, 8),
    max_features: int = 25,
) -> plt.Figure:
    """Visualize bootstrap CIs with asymmetry indication."""
    # Map column names if needed
    df = stats_df.copy()
    
    # Find correlation column
    if r_col not in df.columns:
        if "r" in df.columns:
            r_col = "r"
        elif "correlation" in df.columns:
            r_col = "correlation"
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No correlation column found", ha="center", va="center")
            save_fig(fig, save_path)
            plt.close(fig)
            return fig
    
    # Find feature/identifier column
    if feature_col not in df.columns:
        for col in ["identifier", "channel", "feature", "roi", "name"]:
            if col in df.columns:
                feature_col = col
                break
    
    # Check for required columns
    required_cols = [r_col, ci_low_col, ci_high_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Missing columns: {missing_cols}", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    df = df.dropna(subset=required_cols).head(max_features)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No CI data", ha="center", va="center")
        return fig
    
    n_features = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_features)[::-1]
    
    r_values = df[r_col].values
    ci_low = df[ci_low_col].values
    ci_high = df[ci_high_col].values
    
    # Color by whether CI excludes zero (significant)
    colors = []
    for low, high in zip(ci_low, ci_high):
        if (low > 0 and high > 0) or (low < 0 and high < 0):
            colors.append("#C42847")  # Significant
        else:
            colors.append("#666666")  # Not significant
    
    # Plot CIs as horizontal lines
    for i, (r, low, high, color) in enumerate(zip(r_values, ci_low, ci_high, colors)):
        ax.plot([low, high], [y_positions[i], y_positions[i]], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([r], [y_positions[i]], c=color, s=50, zorder=3, edgecolors="white", linewidths=0.5)
    
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df[feature_col].values, fontsize=8)
    ax.set_xlabel("Correlation (r) with 95% CI", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_ylim(-0.5, n_features - 0.5)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_sensitivity_analysis(
    results: Dict[str, pd.DataFrame],
    save_path: Path,
    *,
    effect_col: str = "correlation",
    title: str = "Sensitivity Analysis",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Compare results across different analysis parameters."""
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    conditions = list(results.keys())
    n_conditions = len(conditions)
    
    positions = np.arange(n_conditions)
    
    all_effects = []
    means = []
    
    for cond, df in results.items():
        if effect_col in df.columns:
            effects = df[effect_col].dropna().values
            all_effects.append(effects)
            means.append(np.mean(effects))
        else:
            all_effects.append(np.array([]))
            means.append(np.nan)
    
    # Violin plots
    parts = ax.violinplot([e for e in all_effects if len(e) > 0], 
                          positions=[i for i, e in enumerate(all_effects) if len(e) > 0],
                          showmeans=True, showmedians=True)
    
    # Style
    for pc in parts["bodies"]:
        pc.set_facecolor("#3B82F6")
        pc.set_alpha(0.6)
    
    # Mean line across conditions
    valid_means = [(i, m) for i, m in enumerate(means) if np.isfinite(m)]
    if len(valid_means) > 1:
        ax.plot([v[0] for v in valid_means], [v[1] for v in valid_means], "r--", linewidth=2, alpha=0.7)
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(conditions, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Effect Size (r)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig

