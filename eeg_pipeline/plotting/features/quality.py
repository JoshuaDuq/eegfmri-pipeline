"""
Feature Quality Visualization
==============================

Visualizations for feature quality assessment, reliability,
outlier detection, and missing data patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.utils import get_numeric_feature_columns
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.utils.config.loader import get_config_value


def plot_feature_distribution_grid(
    df: pd.DataFrame,
    save_path: Path,
    *,
    feature_cols: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (3, 2.5),
    max_features: int = 16,
    config: Any = None,
) -> plt.Figure:
    """Grid of histograms for feature distributions with normality indicators."""
    from scipy import stats
    plot_cfg = get_plot_config(config)
    cfg = get_config_value(config, "plotting.plots.features.quality.distribution", {})
    n_cols = cfg.get("n_cols", n_cols)
    figsize_per_plot = tuple(cfg.get("figsize_per_plot", figsize_per_plot))
    max_features = cfg.get("max_features", max_features)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    if feature_cols is None:
        feature_cols = get_numeric_feature_columns(df)
    else:
        feature_cols = [c for c in feature_cols if c in df.columns] if feature_cols else []

    feature_cols = feature_cols[:max_features]
    n_features = len(feature_cols)
    
    if n_features == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric features", ha="center", va="center")
        return fig
    
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        values = df[col].dropna().values
        
        if len(values) < 5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(col[:15], fontsize=plot_cfg.font.medium)
            continue
        
        # Normality test
        try:
            if len(values) < 5000:
                _, p_norm = stats.shapiro(values[:5000])
            else:
                _, p_norm = stats.normaltest(values)
            is_normal = p_norm > 0.05
        except Exception:
            is_normal = None
            p_norm = np.nan
        
        color = "#22C55E" if is_normal else "#EF4444" if is_normal is not None else "#666666"
        
        ax.hist(values, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=1)
        ax.axvline(np.median(values), color="blue", linestyle=":", linewidth=1)
        
        # Title with normality indicator
        norm_text = "OK" if is_normal else "X" if is_normal is not None else "?"
        ax.set_title(f"{col[:12]}... {norm_text}", fontsize=plot_cfg.font.small, color=color)
        ax.tick_params(labelsize=6)
    
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    # Legend
    legend_elements = [
        Patch(facecolor="#22C55E", label="Normal (p>.05)"),
        Patch(facecolor="#EF4444", label="Non-normal"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=plot_cfg.font.medium)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_outlier_trials_heatmap(
    quality_df: pd.DataFrame,
    save_path: Path,
    *,
    z_threshold: float = 3.0,
    feature_cols: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8),
    max_features: int = 30,
    max_trials: int = 100,
    config: Any = None,
) -> plt.Figure:
    """Heatmap showing outlier status per trial and feature."""
    plot_cfg = get_plot_config(config)
    cfg = get_config_value(config, "plotting.plots.features.quality.outlier", {})
    z_threshold = cfg.get("z_threshold", z_threshold)
    figsize = tuple(cfg.get("figsize", figsize))
    max_features = cfg.get("max_features", max_features)
    max_trials = cfg.get("max_trials", max_trials)
    if quality_df is None or not isinstance(quality_df, pd.DataFrame) or quality_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    if feature_cols is None:
        feature_cols = get_numeric_feature_columns(quality_df)
    else:
        feature_cols = [c for c in feature_cols if c in quality_df.columns]
    
    feature_cols = feature_cols[:max_features]
    df = quality_df.head(max_trials)
    
    if not feature_cols or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    # Compute robust z-scores and outlier matrix
    outlier_matrix = np.zeros((len(df), len(feature_cols)))
    
    for j, col in enumerate(feature_cols):
        values = df[col].values
        valid = np.isfinite(values)
        if np.sum(valid) < 3:
            continue

        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        if not np.isfinite(mad) or mad < 1e-12:
            continue

        robust_z = 0.6745 * (values - median) / mad
        robust_z_abs = np.abs(robust_z)
        outlier_matrix[:, j] = np.where(valid & (robust_z_abs > z_threshold), 1, 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(outlier_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    
    ax.set_xlabel("Feature", fontsize=plot_cfg.font.title)
    ax.set_ylabel("Trial", fontsize=plot_cfg.font.title)
    ax.set_title(
        f"Outlier Detection (robust |z| > {z_threshold})",
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Normal", "Outlier"])
    
    # Summary annotation
    n_outliers = np.sum(outlier_matrix)
    total_cells = outlier_matrix.size
    pct = 100 * n_outliers / total_cells
    ax.set_xlabel(
        f"Feature ({n_outliers:.0f} outliers, {pct:.1f}% of data)",
        fontsize=plot_cfg.font.title,
    )
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_snr_distribution(
    quality_df: pd.DataFrame,
    save_path: Path,
    *,
    snr_col: str = "snr_db",
    threshold_db: float = 3.0,
    figsize: Tuple[float, float] = (8, 5),
    config: Any = None,
) -> plt.Figure:
    """Plot distribution of trial SNR values."""
    plot_cfg = get_plot_config(config)
    cfg = get_config_value(config, "plotting.plots.features.quality.snr", {})
    threshold_db = cfg.get("threshold_db", threshold_db)
    figsize = tuple(cfg.get("figsize", figsize))
    if snr_col not in quality_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Column {snr_col} not found", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    values = quality_df[snr_col].dropna().values
    
    # Color by threshold
    below = values < threshold_db
    
    ax.hist(values[~below], bins=25, color="#22C55E", alpha=0.7, label=f"Good (≥{threshold_db} dB)", edgecolor="white")
    ax.hist(values[below], bins=25, color="#EF4444", alpha=0.7, label=f"Poor (<{threshold_db} dB)", edgecolor="white")
    
    ax.axvline(threshold_db, color="black", linestyle="--", linewidth=2, label="Threshold")
    ax.axvline(np.median(values), color="blue", linestyle=":", linewidth=1.5, label=f"Median: {np.median(values):.1f} dB")
    
    ax.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=plot_cfg.font.title)
    ax.set_ylabel("Count", fontsize=plot_cfg.font.title)
    ax.set_title("Trial Quality: SNR Distribution", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    ax.legend(fontsize=plot_cfg.font.medium)
    
    # Summary
    n_poor = np.sum(below)
    n_total = len(values)
    ax.text(0.98, 0.98, f"{n_poor}/{n_total} ({100*n_poor/n_total:.1f}%) below threshold",
            transform=ax.transAxes, ha="right", va="top", fontsize=plot_cfg.font.large, 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig



def plot_reliability_summary(
    icc_df: pd.DataFrame,
    save_path: Path,
    *,
    icc_col: str = "icc",
    feature_col: str = "feature",
    figsize: Tuple[float, float] = (10, 6),
    config: Any = None,
) -> plt.Figure:
    """Summary plot of feature reliability with ICC categories."""
    plot_cfg = get_plot_config(config)
    if icc_df.empty or icc_col not in icc_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ICC data", ha="center", va="center")
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: ICC distribution
    ax1 = axes[0]
    icc_values = icc_df[icc_col].dropna().values
    
    ax1.hist(icc_values, bins=20, color="#3B82F6", alpha=0.7, edgecolor="white")
    
    # Thresholds
    for thresh, label in [(0.5, "Moderate"), (0.75, "Good"), (0.9, "Excellent")]:
        ax1.axvline(thresh, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    ax1.set_xlabel("ICC", fontsize=plot_cfg.font.title)
    ax1.set_ylabel("Count", fontsize=plot_cfg.font.title)
    ax1.set_title("ICC Distribution", fontsize=plot_cfg.font.title, fontweight="bold")
    ax1.set_xlim(0, 1)
    
    # Right: Category pie chart
    ax2 = axes[1]
    
    categories = ["Excellent (≥0.9)", "Good (0.75-0.9)", "Moderate (0.5-0.75)", "Poor (<0.5)"]
    counts = [
        np.sum(icc_values >= 0.9),
        np.sum((icc_values >= 0.75) & (icc_values < 0.9)),
        np.sum((icc_values >= 0.5) & (icc_values < 0.75)),
        np.sum(icc_values < 0.5),
    ]
    colors = ["#22C55E", "#3B82F6", "#F59E0B", "#EF4444"]
    
    # Only show non-zero slices
    valid = [(c, cat, col) for c, cat, col in zip(counts, categories, colors) if c > 0]
    if valid:
        counts_v, cats_v, colors_v = zip(*valid)
        ax2.pie(counts_v, labels=cats_v, colors=colors_v, autopct="%1.0f%%", startangle=90)
        ax2.set_title("Reliability Categories", fontsize=plot_cfg.font.title, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_quality_summary_dashboard(
    quality_report: Dict[str, Any],
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Create comprehensive quality dashboard from quality report."""
    fig = plt.figure(figsize=figsize)
    
    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top left: Summary stats
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    
    summary_text = f"""
    Total Trials: {quality_report.get('n_trials', 'N/A')}
    Total Features: {quality_report.get('n_features', 'N/A')}
    Subjects: {quality_report.get('n_subjects', 'N/A')}
    
    Missing Data:
      Total cells: {quality_report.get('missing_data', {}).get('total_missing', 0)}
      Features affected: {quality_report.get('missing_data', {}).get('features_with_missing', 0)}
    """
    ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes, fontsize=plot_cfg.font.title, va="top", family="monospace")
    ax1.set_title("Data Summary", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    # Top center: Issue counts
    ax2 = fig.add_subplot(gs[0, 1])
    summary = quality_report.get("summary", {})
    
    issues = ["Non-normal", "With outliers", "Floor effect", "Ceiling effect"]
    counts = [
        summary.get("non_normal_features", 0),
        summary.get("features_with_outliers", 0),
        summary.get("features_with_floor", 0),
        summary.get("features_with_ceiling", 0),
    ]
    colors = ["#EF4444", "#F59E0B", "#6366F1", "#8B5CF6"]
    
    ax2.barh(issues, counts, color=colors, edgecolor="white")
    ax2.set_xlabel("Count", fontsize=plot_cfg.font.title)
    ax2.set_title("Distribution Issues", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    # Top right: Problematic trials
    ax3 = fig.add_subplot(gs[0, 2])
    prob_trials = quality_report.get("problematic_trials", {})
    
    n_prob = prob_trials.get("count", 0)
    n_total = quality_report.get("n_trials", 1)
    
    sizes = [n_prob, n_total - n_prob]
    labels = [f"Problematic ({n_prob})", f"Good ({n_total - n_prob})"]
    colors = ["#EF4444", "#22C55E"]
    
    if n_total > 0:
        ax3.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax3.set_title("Trial Quality", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    # Bottom: Feature issue details (if available)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    
    dist_issues = quality_report.get("distribution_issues", {})
    if dist_issues:
        # Show top issues
        issues_list = []
        for feat, info in list(dist_issues.items())[:10]:
            if isinstance(info, dict) and info.get("valid", True):
                issue_str = []
                if info.get("n_outliers", 0) > 0:
                    issue_str.append(f"{info['n_outliers']} outliers")
                if info.get("is_normal") == False:
                    issue_str.append("non-normal")
                if issue_str:
                    issues_list.append(f"  {feat[:30]}: {', '.join(issue_str)}")
        
        if issues_list:
            issues_text = "Top Feature Issues:\n" + "\n".join(issues_list[:8])
            ax4.text(0.1, 0.9, issues_text, transform=ax4.transAxes, fontsize=plot_cfg.font.large, va="top", family="monospace")
    
    ax4.set_title("Feature Details", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    plt.suptitle("Feature Quality Report", fontsize=13, fontweight="bold")
    
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig
