"""
Shared Plotting Utilities.

Common helper functions for creating standardized plots across the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.stats import fdr_bh


###################################################################
# Color Constants
###################################################################


def get_significance_colors(config: Any = None) -> Tuple[str, str]:
    """Get significant/non-significant colors from config.
    
    Returns:
        Tuple of (significant_color, nonsignificant_color)
    """
    plot_cfg = get_plot_config(config)
    sig_color = plot_cfg.get_color("significant", plot_type="behavioral")
    nonsig_color = plot_cfg.get_color("nonsignificant", plot_type="behavioral")
    return sig_color, nonsig_color


def get_band_colors() -> Dict[str, str]:
    """Get standard frequency band colors.
    
    Returns:
        Dictionary mapping band names to hex colors
    """
    return {
        "delta": "#1f77b4",
        "theta": "#2ca02c", 
        "alpha": "#ff7f0e",
        "beta": "#d62728",
        "gamma": "#9467bd",
    }


def get_correlation_cmap() -> str:
    """Get standard correlation colormap name."""
    return "RdBu_r"


def get_effect_cmap() -> str:
    """Get standard effect size colormap name."""
    return "PuOr_r"


def get_significance_color(
    p_value: float,
    alpha: float = 0.05,
    config: Any = None,
) -> str:
    """Get color based on significance.
    
    Args:
        p_value: P-value to check
        alpha: Significance threshold
        config: Pipeline config
    
    Returns:
        Color string
    """
    sig_color, nonsig_color = get_significance_colors(config)
    return sig_color if pd.notna(p_value) and p_value < alpha else nonsig_color


def get_direction_color(
    value: float,
    p_value: Optional[float] = None,
    alpha: float = 0.05,
    config: Any = None,
) -> str:
    """Get color based on value direction and significance.
    
    Args:
        value: Effect value (positive/negative)
        p_value: Optional p-value for significance
        alpha: Significance threshold
        config: Pipeline config
    
    Returns:
        Color string
    """
    sig_color, nonsig_color = get_significance_colors(config)
    
    if p_value is not None and (pd.isna(p_value) or p_value >= alpha):
        return nonsig_color
    
    return sig_color if value > 0 else "#4C72B0"  # Blue for negative


###################################################################
# Plotting Utilities
###################################################################


def add_significance_markers(
    ax: plt.Axes,
    p_values: pd.DataFrame,
    data_values: pd.DataFrame,
    alpha: float = 0.05,
    marker: str = "*",
    fontsize: int = 10,
    q_values: Optional[pd.DataFrame] = None,
    fdr_alpha: Optional[float] = None,
    fdr_marker: str = "●",
) -> None:
    """Add significance markers to a heatmap.
    
    Args:
        ax: Matplotlib axes with heatmap
        p_values: DataFrame of p-values (same shape as data)
        data_values: DataFrame of data values (for positioning)
        alpha: Significance threshold
        marker: Marker character for uncorrected significance
        fontsize: Marker font size
        q_values: Optional DataFrame of q-values (same shape as data)
        fdr_alpha: Optional FDR alpha threshold
        fdr_marker: Marker character for FDR-significant cells
    """
    if p_values is None or data_values is None:
        return
    if p_values.empty or data_values.empty:
        return
    
    use_q = q_values is not None and fdr_alpha is not None
    if use_q and not q_values.empty:
        q_values = q_values.reindex_like(p_values)
    
    for i, row in enumerate(p_values.index):
        for j, col in enumerate(p_values.columns):
            if row not in data_values.index or col not in data_values.columns:
                continue
            p_val = p_values.loc[row, col]
            marker_text = None
            if use_q and q_values is not None and row in q_values.index and col in q_values.columns:
                q_val = q_values.loc[row, col]
                if pd.notna(q_val) and q_val < fdr_alpha:
                    marker_text = fdr_marker
            if marker_text is None and pd.notna(p_val) and p_val < alpha:
                marker_text = marker
            if marker_text:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    marker_text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color="black",
                )


def compute_q_values_table(
    p_values: pd.DataFrame,
    alpha: float = 0.05,
    config: Any = None,
) -> Optional[pd.DataFrame]:
    """BH-FDR adjustment for a rectangular p-value table.
    
    Returns a DataFrame of q-values aligned to the input.
    """
    if p_values is None or p_values.empty:
        return None
    
    flat = pd.to_numeric(p_values.stack(), errors="coerce")
    if flat.empty:
        return None
    
    q_flat = fdr_bh(flat.values, alpha=alpha, config=config)
    q_series = pd.Series(q_flat, index=flat.index)
    return q_series.unstack().reindex_like(p_values)


def create_horizontal_bar_plot(
    ax: plt.Axes,
    values: np.ndarray,
    labels: List[str],
    colors: Optional[List[str]] = None,
    config: Any = None,
    alpha: float = 0.8,
    height: float = 0.7,
    add_zero_line: bool = True,
) -> None:
    """Create a styled horizontal bar plot.
    
    Args:
        ax: Matplotlib axes
        values: Bar values
        labels: Bar labels
        colors: Optional list of colors (defaults to significance-based)
        config: Pipeline config
        alpha: Bar transparency
        height: Bar height
        add_zero_line: Whether to add vertical line at x=0
    """
    y_pos = np.arange(len(values))
    
    if colors is None:
        sig_color, nonsig_color = get_significance_colors(config)
        colors = [sig_color if v > 0 else nonsig_color for v in values]
    
    ax.barh(y_pos, values, color=colors, alpha=alpha, height=height)
    
    if add_zero_line:
        ax.axvline(0, color="gray", linestyle="-", linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)


def create_correlation_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    annot: bool = True,
    fmt: str = ".2f",
    annot_fontsize: int = 9,
    cbar_label: str = "r",
    linewidths: float = 0.5,
    linecolor: str = "white",
) -> None:
    """Create a styled correlation heatmap.
    
    Args:
        ax: Matplotlib axes
        data: Pivot table of correlation values
        vmax: Maximum absolute value for color scale (auto-computed if None)
        cmap: Colormap name
        annot: Whether to annotate cells
        fmt: Annotation format string
        annot_fontsize: Annotation font size
        cbar_label: Colorbar label
        linewidths: Cell border width
        linecolor: Cell border color
    """
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    if vmax is None:
        vmax = max(0.3, data.abs().max().max())
    
    sns.heatmap(
        data, ax=ax, cmap=cmap, center=0,
        vmin=-vmax, vmax=vmax,
        annot=annot, fmt=fmt, annot_kws={"size": annot_fontsize},
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        linewidths=linewidths, linecolor=linecolor,
    )


def add_effect_size_guidelines(
    ax: plt.Axes,
    orientation: str = "vertical",
    small: float = 0.2,
    medium: float = 0.5,
    large: float = 0.8,
) -> None:
    """Add Cohen's effect size interpretation guidelines to a plot.
    
    Args:
        ax: Matplotlib axes
        orientation: "vertical" for vertical lines, "horizontal" for horizontal
        small: Small effect threshold
        medium: Medium effect threshold
        large: Large effect threshold
    """
    line_func = ax.axvline if orientation == "vertical" else ax.axhline
    
    for threshold in [-large, -medium, -small, small, medium, large]:
        line_func(threshold, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)


def create_figure_with_suptitle(
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    suptitle: str,
    subject: str,
    suptitle_fontsize: int = 13,
    suptitle_y: float = 1.02,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure with standardized suptitle formatting.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size
        suptitle: Main title text
        subject: Subject ID for title
        suptitle_fontsize: Title font size
        suptitle_y: Title y position
    
    Returns:
        Tuple of (figure, axes array)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(
        f"{suptitle} (sub-{subject})",
        fontsize=suptitle_fontsize,
        fontweight="bold",
        y=suptitle_y,
    )
    return fig, axes


###################################################################
# Statistical Annotation Utilities
###################################################################


def format_stats_summary(
    n_significant: int,
    n_total: int,
    effect_type: str = "effects",
) -> str:
    """Format a summary of significant results.
    
    Args:
        n_significant: Number of significant results
        n_total: Total number of tests
        effect_type: Description of what was tested
    
    Returns:
        Formatted summary string
    """
    pct = 100 * n_significant / n_total if n_total > 0 else 0
    return f"Significant {effect_type}: {n_significant}/{n_total} ({pct:.1f}%)"


###################################################################
# Legend Utilities
###################################################################


def add_significance_legend(
    ax: plt.Axes,
    config: Any = None,
    loc: str = "lower right",
    fontsize: int = 9,
    include_direction: bool = False,
) -> None:
    """Add a significance legend to a plot.
    
    Args:
        ax: Matplotlib axes
        config: Pipeline config
        loc: Legend location
        fontsize: Legend font size
        include_direction: Whether to include positive/negative labels
    """
    sig_color, nonsig_color = get_significance_colors(config)
    
    if include_direction:
        legend_elements = [
            Patch(facecolor=sig_color, label="Positive (p<0.05)"),
            Patch(facecolor="#4C72B0", label="Negative (p<0.05)"),
            Patch(facecolor=nonsig_color, label="Not significant"),
        ]
    else:
        legend_elements = [
            Patch(facecolor=sig_color, label="Significant (p<0.05)"),
            Patch(facecolor=nonsig_color, label="Not significant"),
        ]
    
    ax.legend(handles=legend_elements, loc=loc, fontsize=fontsize)


def add_band_legend(
    ax: plt.Axes,
    bands: Optional[List[str]] = None,
    loc: str = "lower right",
    fontsize: int = 9,
) -> None:
    """Add a frequency band legend to a plot.
    
    Args:
        ax: Matplotlib axes
        bands: List of bands to include (default: all)
        loc: Legend location
        fontsize: Legend font size
    """
    band_colors = get_band_colors()
    
    if bands is None:
        bands = list(band_colors.keys())
    
    legend_elements = [
        Patch(facecolor=band_colors.get(b, "gray"), label=b.title())
        for b in bands if b in band_colors
    ]
    
    ax.legend(handles=legend_elements, loc=loc, fontsize=fontsize)
