"""
Effect Size Visualization
==========================

Forest plots and effect size visualizations for publication-ready figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir


def plot_correlation_forest(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    sort_by: str = "correlation",
    ascending: bool = False,
    show_fdr: bool = True,
    max_features: int = 30,
    title: str = "Effect Sizes with 95% CI",
    figsize: Tuple[float, float] = (8, 10),
    config: Any = None,
) -> plt.Figure:
    """Create forest plot of correlations with confidence intervals.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Should contain columns for identifier and correlation.
        Accepts various naming conventions:
        - Identifier: 'identifier', 'channel', 'feature', 'roi', 'name'
        - Correlation: 'correlation', 'r', 'rho'
        Optional: p_fdr/q, p_value/p, ci_low, ci_high, band
    save_path : Path
        Output path
    sort_by : str
        Column to sort by
    show_fdr : bool
        Use FDR-corrected significance markers
    max_features : int
        Maximum features to show
    """
    # Validate input
    if stats_df is None or not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Normalize column names
    df = stats_df.copy()
    
    # Map identifier column
    id_cols = ["identifier", "channel", "feature", "roi", "name"]
    for col in id_cols:
        if col in df.columns:
            if col != "identifier":
                df["identifier"] = df[col]
            break
    else:
        raise ValueError(f"Missing identifier column. Available: {list(stats_df.columns)}")
    
    # Map correlation column
    r_cols = ["correlation", "r", "rho", "coef"]
    for col in r_cols:
        if col in df.columns:
            if col != "correlation":
                df["correlation"] = df[col]
            break
    else:
        raise ValueError(f"Missing correlation column. Available: {list(stats_df.columns)}")
    
    # Map p-value columns
    if "p" in df.columns and "p_value" not in df.columns:
        df["p_value"] = df["p"]
    if "q" in df.columns and "p_fdr" not in df.columns:
        df["p_fdr"] = df["q"]
    
    # Sort and limit
    sort_col = sort_by if sort_by in df.columns else "correlation"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)
    df = df.head(max_features)
    
    n_features = len(df)
    if n_features == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(n_features)[::-1]
    
    # Get plot config
    plot_cfg = get_plot_config(config) if config else None
    sig_color = "#C42847"
    nonsig_color = "#666666"
    
    # Determine significance
    if show_fdr and "p_fdr" in df.columns:
        sig_mask = df["p_fdr"].values < 0.05
    elif "p_value" in df.columns:
        sig_mask = df["p_value"].values < 0.05
    else:
        sig_mask = np.ones(n_features, dtype=bool)
    
    colors = [sig_color if s else nonsig_color for s in sig_mask]
    
    # Plot points and error bars
    correlations = df["correlation"].values
    
    if "ci_low" in df.columns and "ci_high" in df.columns:
        ci_low = df["ci_low"].values
        ci_high = df["ci_high"].values
        # Plot error bars individually to avoid RGBA issues with color lists
        for i in range(len(correlations)):
            ax.errorbar(
                correlations[i], y_positions[i], 
                xerr=[[correlations[i] - ci_low[i]], [ci_high[i] - correlations[i]]], 
                fmt="none",
                ecolor=colors[i], elinewidth=1.5, capsize=3, capthick=1.5, zorder=1
            )
    
    ax.scatter(correlations, y_positions, c=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    
    # Zero line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    # Labels
    labels = []
    for _, row in df.iterrows():
        label = row["identifier"]
        if "band" in row and pd.notna(row["band"]):
            label = f"{row['band']} | {label}"
        labels.append(label)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Correlation (r)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sig_color, markersize=8, label="Significant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=nonsig_color, markersize=8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    # Adjust limits
    ax.set_ylim(-0.5, n_features - 0.5)
    x_max = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-x_max * 1.1, x_max * 1.1)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_effect_size_comparison(
    stats_dfs: Dict[str, pd.DataFrame],
    save_path: Path,
    *,
    effect_col: str = "correlation",
    title: str = "Effect Size Comparison",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Compare effect sizes across conditions/groups."""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(stats_dfs)))
    
    for idx, (name, df) in enumerate(stats_dfs.items()):
        if df.empty or effect_col not in df.columns:
            continue
        
        effects = df[effect_col].dropna()
        positions = np.random.normal(idx, 0.1, size=len(effects))
        
        ax.scatter(positions, effects, alpha=0.6, s=30, c=[colors[idx]], label=name)
        
        # Box plot overlay
        bp = ax.boxplot(
            effects, positions=[idx], widths=0.3, patch_artist=True,
            showfliers=False, zorder=2
        )
        bp["boxes"][0].set_facecolor(colors[idx])
        bp["boxes"][0].set_alpha(0.3)
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(stats_dfs)))
    ax.set_xticklabels(list(stats_dfs.keys()), fontsize=10)
    ax.set_ylabel("Effect Size (r)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_effect_size_heatmap(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    row_col: str = "band",
    col_col: str = "identifier",
    value_col: str = "correlation",
    title: str = "Effect Size Heatmap",
    figsize: Tuple[float, float] = (12, 6),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Create heatmap of effect sizes (e.g., bands x ROIs)."""
    if stats_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    # Pivot data
    pivot = stats_df.pivot_table(
        index=row_col, columns=col_col, values=value_col, aggfunc="mean"
    )
    
    if pivot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data for pivot", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation (r)", fontsize=9)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_condition_effect_sizes(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    max_features: int = 40,
    title: str = "Pain vs Non-Pain Effect Sizes (Hedges' g)",
    figsize: Tuple[float, float] = (10, 12),
    config: Any = None,
) -> plt.Figure:
    """Create forest plot of condition effect sizes (Hedges' g).
    
    Visualizes effect sizes comparing pain vs non-pain feature distributions.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Should contain: feature, hedges_g, cohens_d, p_ttest
    save_path : Path
        Output path
    max_features : int
        Maximum features to show (sorted by absolute effect size)
    """
    if stats_df is None or stats_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No effect size data", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    df = stats_df.copy()
    
    # Ensure required columns
    if "feature" not in df.columns or "hedges_g" not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Missing required columns", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Sort by absolute effect size and limit
    df["abs_g"] = df["hedges_g"].abs()
    df = df.sort_values("abs_g", ascending=False).head(max_features)
    df = df.sort_values("hedges_g", ascending=True)  # Re-sort for display
    
    n_features = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(n_features)
    
    # Color by effect size magnitude
    effects = df["hedges_g"].values
    colors = []
    for g in effects:
        if abs(g) >= 0.8:
            colors.append("#C42847" if g > 0 else "#2563EB")  # Large
        elif abs(g) >= 0.5:
            colors.append("#F59E0B" if g > 0 else "#06B6D4")  # Medium
        else:
            colors.append("#6B7280")  # Small
    
    # Plot bars
    ax.barh(y_positions, effects, color=colors, alpha=0.8, height=0.7)
    
    # Add significance markers
    if "p_ttest" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            if row["p_ttest"] < 0.001:
                marker = "***"
            elif row["p_ttest"] < 0.01:
                marker = "**"
            elif row["p_ttest"] < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                x_pos = effects[i] + 0.05 if effects[i] >= 0 else effects[i] - 0.05
                ha = "left" if effects[i] >= 0 else "right"
                ax.text(x_pos, y_positions[i], marker, ha=ha, va="center", fontsize=10)
    
    # Reference lines for effect size thresholds
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(-0.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["feature"].values, fontsize=8)
    ax.set_xlabel("Hedges' g (Pain > Non-Pain)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C42847", label="Large (+)"),
        Patch(facecolor="#2563EB", label="Large (-)"),
        Patch(facecolor="#F59E0B", label="Medium (+)"),
        Patch(facecolor="#06B6D4", label="Medium (-)"),
        Patch(facecolor="#6B7280", label="Small"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Effect Size")
    
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_temperature_mediation(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    max_features: int = 30,
    title: str = "Temperature Mediation of Feature-Rating Correlations",
    figsize: Tuple[float, float] = (12, 10),
    config: Any = None,
) -> plt.Figure:
    """Visualize partial correlations controlling for temperature.
    
    Shows how correlations change when controlling for temperature,
    highlighting features where temperature mediates the relationship.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Should contain: feature, r, r_partial_temp, temp_mediated, condition
    """
    if stats_df is None or stats_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No partial correlation data", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    df = stats_df.copy()
    
    # Check required columns
    required = {"feature", "r", "r_partial_temp"}
    if not required.issubset(df.columns):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Missing columns: {required - set(df.columns)}", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Sort by change in correlation (mediation effect)
    df["r_change"] = df["r"].abs() - df["r_partial_temp"].abs()
    df = df.sort_values("r_change", ascending=False).head(max_features)
    
    n_features = len(df)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    y_positions = np.arange(n_features)[::-1]
    
    # Left panel: Simple vs Partial correlations
    ax1 = axes[0]
    
    # Plot simple correlations
    ax1.scatter(df["r"].values, y_positions, c="#3B82F6", s=60, label="Simple r", zorder=3, marker="o")
    
    # Plot partial correlations
    ax1.scatter(df["r_partial_temp"].values, y_positions, c="#10B981", s=60, label="Partial r (temp)", zorder=3, marker="s")
    
    # Connect with lines
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.plot([row["r"], row["r_partial_temp"]], [y_positions[i], y_positions[i]], 
                 color="#6B7280", linewidth=1, alpha=0.5, zorder=1)
    
    ax1.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Correlation (r)", fontsize=10)
    ax1.set_title("Simple vs Temperature-Controlled", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.3, linestyle=":")
    
    # Right panel: Mediation effect
    ax2 = axes[1]
    
    # Color by mediation status
    if "temp_mediated" in df.columns:
        colors = ["#EF4444" if m else "#6B7280" for m in df["temp_mediated"].values]
    else:
        colors = ["#6B7280"] * n_features
    
    ax2.barh(y_positions, df["r_change"].values, color=colors, alpha=0.8, height=0.6)
    
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Δr (Simple - Partial)", fontsize=10)
    ax2.set_title("Temperature Mediation Effect", fontsize=10, fontweight="bold")
    
    # Legend for mediation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EF4444", label="Mediated by temp"),
        Patch(facecolor="#6B7280", label="Not mediated"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax2.grid(axis="x", alpha=0.3, linestyle=":")
    
    # Y-axis labels
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(df["feature"].values, fontsize=8)
    
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig

