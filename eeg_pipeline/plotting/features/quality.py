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
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.utils import get_numeric_feature_columns
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.utils.config.loader import get_config_value


# Constants
_MAD_NORMALIZATION_FACTOR = 0.6745
_MIN_SAMPLES_FOR_NORMALITY_TEST = 3
_MIN_SAMPLES_FOR_HISTOGRAM = 5
_MAX_SAMPLES_FOR_SHAPIRO = 5000
_NORMALITY_P_THRESHOLD = 0.05
_MIN_VALID_SAMPLES_FOR_OUTLIER = 3
_MAD_EPSILON = 1e-12

_ICC_THRESHOLD_MODERATE = 0.5
_ICC_THRESHOLD_GOOD = 0.75
_ICC_THRESHOLD_EXCELLENT = 0.9

_COLOR_NORMAL = "#22C55E"
_COLOR_NON_NORMAL = "#EF4444"
_COLOR_UNKNOWN = "#666666"
_COLOR_GOOD = "#3B82F6"
_COLOR_WARNING = "#F59E0B"
_COLOR_PROBLEM = "#EF4444"
_COLOR_EXCELLENT = "#22C55E"
_COLOR_MODERATE = "#6366F1"
_COLOR_CEILING = "#8B5CF6"


def _create_empty_plot(message: str) -> plt.Figure:
    """Create an empty plot with a message."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    return fig


def _validate_dataframe(df: Optional[pd.DataFrame]) -> bool:
    """Validate that dataframe is not None and not empty."""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def _test_normality(values: np.ndarray) -> Tuple[Optional[bool], float]:
    """Test normality of values using appropriate test.
    
    Returns:
        Tuple of (is_normal, p_value). is_normal is None if test failed.
    """
    try:
        if len(values) < _MAX_SAMPLES_FOR_SHAPIRO:
            _, p_value = stats.shapiro(values)
        else:
            _, p_value = stats.normaltest(values)
        is_normal = p_value > _NORMALITY_P_THRESHOLD
        return is_normal, p_value
    except Exception:
        return None, np.nan


def _get_normality_color(is_normal: Optional[bool]) -> str:
    """Get color code based on normality test result."""
    if is_normal is True:
        return _COLOR_NORMAL
    elif is_normal is False:
        return _COLOR_NON_NORMAL
    else:
        return _COLOR_UNKNOWN


def _get_normality_indicator(is_normal: Optional[bool]) -> str:
    """Get text indicator for normality status."""
    if is_normal is True:
        return "OK"
    elif is_normal is False:
        return "X"
    else:
        return "?"


def _compute_robust_z_scores(values: np.ndarray, z_threshold: float) -> np.ndarray:
    """Compute robust z-scores using median and MAD.
    
    Returns:
        Binary array where 1 indicates outlier, 0 indicates normal.
    """
    valid_mask = np.isfinite(values)
    if np.sum(valid_mask) < _MIN_VALID_SAMPLES_FOR_OUTLIER:
        return np.zeros_like(values, dtype=float)
    
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    
    if not np.isfinite(mad) or mad < _MAD_EPSILON:
        return np.zeros_like(values, dtype=float)
    
    robust_z = _MAD_NORMALIZATION_FACTOR * (values - median) / mad
    robust_z_abs = np.abs(robust_z)
    return np.where(valid_mask & (robust_z_abs > z_threshold), 1.0, 0.0)


def _count_icc_categories(icc_values: np.ndarray) -> Tuple[List[int], List[str], List[str]]:
    """Count ICC values in each reliability category.
    
    Returns:
        Tuple of (counts, labels, colors) for pie chart.
    """
    counts = [
        np.sum(icc_values >= _ICC_THRESHOLD_EXCELLENT),
        np.sum((icc_values >= _ICC_THRESHOLD_GOOD) & (icc_values < _ICC_THRESHOLD_EXCELLENT)),
        np.sum((icc_values >= _ICC_THRESHOLD_MODERATE) & (icc_values < _ICC_THRESHOLD_GOOD)),
        np.sum(icc_values < _ICC_THRESHOLD_MODERATE),
    ]
    labels = [
        f"Excellent (≥{_ICC_THRESHOLD_EXCELLENT})",
        f"Good ({_ICC_THRESHOLD_GOOD}-{_ICC_THRESHOLD_EXCELLENT})",
        f"Moderate ({_ICC_THRESHOLD_MODERATE}-{_ICC_THRESHOLD_GOOD})",
        f"Poor (<{_ICC_THRESHOLD_MODERATE})",
    ]
    colors = [_COLOR_EXCELLENT, _COLOR_GOOD, _COLOR_WARNING, _COLOR_PROBLEM]
    return counts, labels, colors


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
    plot_cfg = get_plot_config(config)
    cfg = get_config_value(config, "plotting.plots.features.quality.distribution", {})
    n_cols = cfg.get("n_cols", n_cols)
    figsize_per_plot = tuple(cfg.get("figsize_per_plot", figsize_per_plot))
    max_features = cfg.get("max_features", max_features)

    if not _validate_dataframe(df):
        return _create_empty_plot("No data")

    if feature_cols is None:
        feature_cols = get_numeric_feature_columns(df)
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]

    feature_cols = feature_cols[:max_features]
    n_features = len(feature_cols)
    
    if n_features == 0:
        return _create_empty_plot("No numeric features")
    
    n_rows = (n_features + n_cols - 1) // n_cols
    quality_config = plot_cfg.plot_type_configs.get("quality", {})
    width_per_plot = float(quality_config.get("width_per_plot", 3.0))
    height_per_plot = float(quality_config.get("height_per_plot", 2.5))
    figsize = (width_per_plot * n_cols, height_per_plot * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        values = df[col].dropna().values
        
        if len(values) < _MIN_SAMPLES_FOR_HISTOGRAM:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(col[:15], fontsize=plot_cfg.font.medium)
            continue
        
        is_normal, _ = _test_normality(values)
        color = _get_normality_color(is_normal)
        norm_indicator = _get_normality_indicator(is_normal)
        
        ax.hist(values, bins=20, color=color, alpha=0.7, edgecolor="white")
        mean_value = np.mean(values)
        median_value = np.median(values)
        ax.axvline(mean_value, color="black", linestyle="--", linewidth=1)
        ax.axvline(median_value, color="blue", linestyle=":", linewidth=1)
        
        title = f"{col[:12]}... {norm_indicator}"
        ax.set_title(title, fontsize=plot_cfg.font.small, color=color)
        ax.tick_params(labelsize=6)
    
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    legend_elements = [
        Patch(facecolor=_COLOR_NORMAL, label="Normal (p>.05)"),
        Patch(facecolor=_COLOR_NON_NORMAL, label="Non-normal"),
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
    max_features = cfg.get("max_features", max_features)
    max_trials = cfg.get("max_trials", max_trials)
    
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")
    
    if not _validate_dataframe(quality_df):
        return _create_empty_plot("No data")

    if feature_cols is None:
        feature_cols = get_numeric_feature_columns(quality_df)
    else:
        feature_cols = [c for c in feature_cols if c in quality_df.columns]
    
    feature_cols = feature_cols[:max_features]
    df = quality_df.head(max_trials)
    
    if not feature_cols or df.empty:
        return _create_empty_plot("No data")
    
    n_trials, n_features = len(df), len(feature_cols)
    outlier_matrix = np.zeros((n_trials, n_features))
    
    for j, col in enumerate(feature_cols):
        values = df[col].values
        outlier_matrix[:, j] = _compute_robust_z_scores(values, z_threshold)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(outlier_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    
    ax.set_ylabel("Trial", fontsize=plot_cfg.font.title)
    ax.set_title(
        f"Outlier Detection (robust |z| > {z_threshold})",
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
    )
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Normal", "Outlier"])
    
    n_outliers = int(np.sum(outlier_matrix))
    total_cells = outlier_matrix.size
    outlier_percentage = 100 * n_outliers / total_cells
    xlabel = f"Feature ({n_outliers:.0f} outliers, {outlier_percentage:.1f}% of data)"
    ax.set_xlabel(xlabel, fontsize=plot_cfg.font.title)
    
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
    
    if figsize is None:
        figsize = plot_cfg.get_figure_size("standard", plot_type="features")
    
    if snr_col not in quality_df.columns:
        return _create_empty_plot(f"Column {snr_col} not found")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    values = quality_df[snr_col].dropna().values
    below_threshold = values < threshold_db
    above_threshold = ~below_threshold
    
    ax.hist(
        values[above_threshold],
        bins=25,
        color=_COLOR_NORMAL,
        alpha=0.7,
        label=f"Good (≥{threshold_db} dB)",
        edgecolor="white",
    )
    ax.hist(
        values[below_threshold],
        bins=25,
        color=_COLOR_NON_NORMAL,
        alpha=0.7,
        label=f"Poor (<{threshold_db} dB)",
        edgecolor="white",
    )
    
    ax.axvline(threshold_db, color="black", linestyle="--", linewidth=2, label="Threshold")
    median_snr = np.median(values)
    ax.axvline(
        median_snr,
        color="blue",
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {median_snr:.1f} dB",
    )
    
    ax.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=plot_cfg.font.title)
    ax.set_ylabel("Count", fontsize=plot_cfg.font.title)
    ax.set_title("Trial Quality: SNR Distribution", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    ax.legend(fontsize=plot_cfg.font.medium)
    
    n_poor = int(np.sum(below_threshold))
    n_total = len(values)
    poor_percentage = 100 * n_poor / n_total
    summary_text = f"{n_poor}/{n_total} ({poor_percentage:.1f}%) below threshold"
    ax.text(
        0.98,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=plot_cfg.font.large,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
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
        return _create_empty_plot("No ICC data")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    icc_values = icc_df[icc_col].dropna().values
    
    ax1.hist(icc_values, bins=20, color=_COLOR_GOOD, alpha=0.7, edgecolor="white")
    
    thresholds = [
        (_ICC_THRESHOLD_MODERATE, "Moderate"),
        (_ICC_THRESHOLD_GOOD, "Good"),
        (_ICC_THRESHOLD_EXCELLENT, "Excellent"),
    ]
    for threshold, label in thresholds:
        ax1.axvline(threshold, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    ax1.set_xlabel("ICC", fontsize=plot_cfg.font.title)
    ax1.set_ylabel("Count", fontsize=plot_cfg.font.title)
    ax1.set_title("ICC Distribution", fontsize=plot_cfg.font.title, fontweight="bold")
    ax1.set_xlim(0, 1)
    
    ax2 = axes[1]
    counts, labels, colors = _count_icc_categories(icc_values)
    
    valid_slices = [(c, label, color) for c, label, color in zip(counts, labels, colors) if c > 0]
    if valid_slices:
        counts_valid, labels_valid, colors_valid = zip(*valid_slices)
        ax2.pie(counts_valid, labels=labels_valid, colors=colors_valid, autopct="%1.0f%%", startangle=90)
        ax2.set_title("Reliability Categories", fontsize=plot_cfg.font.title, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def _format_feature_issue(feature_name: str, info: Dict[str, Any]) -> Optional[str]:
    """Format a single feature issue description."""
    if not isinstance(info, dict) or not info.get("valid", True):
        return None
    
    issue_parts = []
    n_outliers = info.get("n_outliers", 0)
    if n_outliers > 0:
        issue_parts.append(f"{n_outliers} outliers")
    
    if info.get("is_normal") is False:
        issue_parts.append("non-normal")
    
    if not issue_parts:
        return None
    
    issue_text = ", ".join(issue_parts)
    return f"  {feature_name[:30]}: {issue_text}"


def plot_quality_summary_dashboard(
    quality_report: Dict[str, Any],
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (14, 8),
    config: Any = None,
) -> plt.Figure:
    """Create a comprehensive quality summary dashboard."""
    plot_cfg = get_plot_config(config)
    fig = plt.figure(figsize=figsize)
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    
    n_trials = quality_report.get("n_trials", "N/A")
    n_features = quality_report.get("n_features", "N/A")
    n_subjects = quality_report.get("n_subjects", "N/A")
    missing_data = quality_report.get("missing_data", {})
    total_missing = missing_data.get("total_missing", 0)
    features_with_missing = missing_data.get("features_with_missing", 0)
    
    summary_text = f"""
    Total Trials: {n_trials}
    Total Features: {n_features}
    Subjects: {n_subjects}
    
    Missing Data:
      Total cells: {total_missing}
      Features affected: {features_with_missing}
    """
    ax1.text(
        0.1,
        0.9,
        summary_text,
        transform=ax1.transAxes,
        fontsize=plot_cfg.font.title,
        va="top",
        family="monospace",
    )
    ax1.set_title("Data Summary", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    ax2 = fig.add_subplot(gs[0, 1])
    summary = quality_report.get("summary", {})
    
    issue_labels = ["Non-normal", "With outliers", "Floor effect", "Ceiling effect"]
    issue_keys = ["non_normal_features", "features_with_outliers", "features_with_floor", "features_with_ceiling"]
    issue_counts = [summary.get(key, 0) for key in issue_keys]
    issue_colors = [_COLOR_PROBLEM, _COLOR_WARNING, _COLOR_MODERATE, _COLOR_CEILING]
    
    ax2.barh(issue_labels, issue_counts, color=issue_colors, edgecolor="white")
    ax2.set_xlabel("Count", fontsize=plot_cfg.font.title)
    ax2.set_title("Distribution Issues", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    ax3 = fig.add_subplot(gs[0, 2])
    problematic_trials = quality_report.get("problematic_trials", {})
    
    n_problematic = problematic_trials.get("count", 0)
    n_total = quality_report.get("n_trials", 1)
    n_good = n_total - n_problematic
    
    if n_total > 0:
        sizes = [n_problematic, n_good]
        labels = [f"Problematic ({n_problematic})", f"Good ({n_good})"]
        colors = [_COLOR_PROBLEM, _COLOR_NORMAL]
        ax3.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax3.set_title("Trial Quality", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    
    distribution_issues = quality_report.get("distribution_issues", {})
    if distribution_issues:
        issues_list = []
        max_features_to_show = 10
        for feature_name, info in list(distribution_issues.items())[:max_features_to_show]:
            issue_text = _format_feature_issue(feature_name, info)
            if issue_text:
                issues_list.append(issue_text)
        
        if issues_list:
            max_issues_in_text = 8
            issues_to_show = issues_list[:max_issues_in_text]
            issues_text = "Top Feature Issues:\n" + "\n".join(issues_to_show)
            ax4.text(
                0.1,
                0.9,
                issues_text,
                transform=ax4.transAxes,
                fontsize=plot_cfg.font.large,
                va="top",
                family="monospace",
            )
    
    ax4.set_title("Feature Details", fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    plt.suptitle("Feature Quality Report", fontsize=13, fontweight="bold")
    
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig
