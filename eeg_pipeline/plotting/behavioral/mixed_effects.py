"""
Mixed-Effects Model Visualization
==================================

Visualizations for multilevel modeling results including ICC,
random effects, and within/between subject variance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir


def plot_icc_bar_chart(
    icc_df: pd.DataFrame,
    save_path: Path,
    *,
    icc_col: str = "icc",
    feature_col: str = "feature",
    ci_low_col: str = "icc_ci_low",
    ci_high_col: str = "icc_ci_high",
    title: str = "Feature Reliability (ICC)",
    figsize: Tuple[float, float] = (10, 8),
    max_features: int = 30,
    sort: bool = True,
) -> plt.Figure:
    """Create bar chart of ICC values with confidence intervals.
    
    ICC interpretation:
    - < 0.5: Poor reliability
    - 0.5-0.75: Moderate reliability
    - 0.75-0.9: Good reliability
    - > 0.9: Excellent reliability
    """
    if icc_df.empty or icc_col not in icc_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No ICC data", ha="center", va="center")
        return fig
    
    df = icc_df.copy().dropna(subset=[icc_col])
    
    if sort:
        df = df.sort_values(icc_col, ascending=True)
    df = df.tail(max_features)
    
    n_features = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_features)
    icc_values = df[icc_col].values
    
    # Color by reliability level
    colors = []
    for icc in icc_values:
        if icc >= 0.9:
            colors.append("#22C55E")  # Excellent (green)
        elif icc >= 0.75:
            colors.append("#3B82F6")  # Good (blue)
        elif icc >= 0.5:
            colors.append("#F59E0B")  # Moderate (amber)
        else:
            colors.append("#EF4444")  # Poor (red)
    
    # Plot bars
    bars = ax.barh(y_positions, icc_values, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    
    # Add error bars if CI available
    if ci_low_col in df.columns and ci_high_col in df.columns:
        ci_low = df[ci_low_col].values
        ci_high = df[ci_high_col].values
        xerr = np.array([icc_values - ci_low, ci_high - icc_values])
        ax.errorbar(
            icc_values, y_positions, xerr=xerr, fmt="none",
            ecolor="black", elinewidth=1, capsize=2, capthick=1
        )
    
    # Reference lines
    for thresh, label in [(0.5, "Moderate"), (0.75, "Good"), (0.9, "Excellent")]:
        ax.axvline(thresh, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(thresh, n_features + 0.3, label, ha="center", fontsize=7, color="gray")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df[feature_col].values, fontsize=8)
    ax.set_xlabel("ICC", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Legend
    legend_elements = [
        Patch(facecolor="#22C55E", label="Excellent (≥0.9)"),
        Patch(facecolor="#3B82F6", label="Good (0.75-0.9)"),
        Patch(facecolor="#F59E0B", label="Moderate (0.5-0.75)"),
        Patch(facecolor="#EF4444", label="Poor (<0.5)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_variance_decomposition(
    results_df: pd.DataFrame,
    save_path: Path,
    *,
    random_var_col: str = "random_variance",
    residual_var_col: str = "residual_variance",
    feature_col: str = "feature",
    title: str = "Variance Decomposition",
    figsize: Tuple[float, float] = (10, 8),
    max_features: int = 25,
) -> plt.Figure:
    """Show within vs between subject variance for each feature."""
    if results_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    df = results_df.copy()
    
    # Compute total variance and ICC for sorting
    df["total_var"] = df[random_var_col] + df[residual_var_col]
    df["icc"] = df[random_var_col] / (df["total_var"] + 1e-12)
    df = df.sort_values("icc", ascending=True).tail(max_features)
    
    n_features = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_features)
    
    # Stacked horizontal bars
    between = df[random_var_col].values / (df["total_var"].values + 1e-12)
    within = df[residual_var_col].values / (df["total_var"].values + 1e-12)
    
    ax.barh(y_positions, between, color="#3B82F6", label="Between-subject", height=0.6)
    ax.barh(y_positions, within, left=between, color="#F59E0B", label="Within-subject", height=0.6)
    
    # Add ICC values as text
    for i, (b, icc) in enumerate(zip(between, df["icc"].values)):
        ax.text(1.02, i, f"ICC={icc:.2f}", va="center", fontsize=7, color="gray")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df[feature_col].values, fontsize=8)
    ax.set_xlabel("Proportion of Variance", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_mixed_effects_forest(
    results_df: pd.DataFrame,
    save_path: Path,
    *,
    effect_col: str = "fixed_effect_std",
    se_col: str = "fixed_se",
    p_col: str = "fixed_p",
    feature_col: str = "feature",
    title: str = "Fixed Effects (Standardized)",
    figsize: Tuple[float, float] = (8, 10),
    max_features: int = 30,
) -> plt.Figure:
    """Forest plot of fixed effects from mixed-effects models."""
    if results_df.empty or effect_col not in results_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    df = results_df.copy().dropna(subset=[effect_col])
    df["abs_effect"] = df[effect_col].abs()
    df = df.sort_values("abs_effect", ascending=False).head(max_features)
    
    n_features = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_features)[::-1]
    effects = df[effect_col].values
    
    # Colors by significance
    sig_color = "#C42847"
    nonsig_color = "#666666"
    
    if p_col in df.columns:
        sig_mask = df[p_col].values < 0.05
        # Check for FDR
        if "fixed_p_fdr" in df.columns:
            sig_mask = df["fixed_p_fdr"].values < 0.05
    else:
        sig_mask = np.ones(n_features, dtype=bool)
    
    colors = [sig_color if s else nonsig_color for s in sig_mask]
    
    # Error bars (±1.96 SE for 95% CI)
    if se_col in df.columns:
        se = df[se_col].values
        xerr = 1.96 * se
        ax.errorbar(
            effects, y_positions, xerr=xerr, fmt="none",
            ecolor=colors, elinewidth=1.5, capsize=3, capthick=1.5
        )
    
    ax.scatter(effects, y_positions, c=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df[feature_col].values, fontsize=8)
    ax.set_xlabel("Standardized Fixed Effect (β)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_ylim(-0.5, n_features - 0.5)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_subject_random_effects(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    subject_col: str,
    save_path: Path,
    *,
    title: str = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot feature values by subject showing random intercepts."""
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    subjects = df[subject_col].unique()
    n_subjects = len(subjects)
    colors = plt.cm.Set2(np.linspace(0, 1, n_subjects))
    
    # Plot individual points with jitter
    for idx, subj in enumerate(subjects):
        subj_data = df[df[subject_col] == subj][value_col].dropna()
        x_pos = np.random.normal(idx, 0.1, size=len(subj_data))
        ax.scatter(x_pos, subj_data, c=[colors[idx]], alpha=0.6, s=20)
        
        # Subject mean
        mean_val = subj_data.mean()
        ax.hlines(mean_val, idx - 0.3, idx + 0.3, colors=colors[idx], linewidth=2)
    
    # Grand mean
    grand_mean = df[value_col].mean()
    ax.axhline(grand_mean, color="black", linestyle="--", linewidth=1.5, label=f"Grand mean: {grand_mean:.2f}")
    
    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels(subjects, fontsize=9, rotation=45, ha="right")
    ax.set_xlabel("Subject", fontsize=10)
    ax.set_ylabel(feature_col, fontsize=10)
    
    if title is None:
        title = f"Subject Variability: {feature_col}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig

