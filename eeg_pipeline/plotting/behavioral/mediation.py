"""
Mediation Analysis Visualization
=================================

Path diagrams and summary plots for mediation analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.mediation import MediationResult


def plot_mediation_diagram(
    result: "MediationResult",
    save_path: Path,
    *,
    x_label: str = "X",
    m_label: str = "M",
    y_label: str = "Y",
    title: str = "Mediation Analysis",
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Create mediation path diagram with coefficients.
    
    Displays:
    - Path a (X → M)
    - Path b (M → Y | X)
    - Path c (total: X → Y)
    - Path c' (direct: X → Y | M)
    - Indirect effect (a × b) with CI
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    
    # Box positions
    x_pos = (1.5, 2)
    m_pos = (5, 6)
    y_pos = (8.5, 2)
    
    box_width, box_height = 2, 1
    
    # Draw boxes
    def draw_box(pos, label, color="#E8E8E8"):
        box = FancyBboxPatch(
            (pos[0] - box_width/2, pos[1] - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color, edgecolor="black", linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, ha="center", va="center", fontsize=11, fontweight="bold")
    
    draw_box(x_pos, x_label, "#DBEAFE")  # Blue tint
    draw_box(m_pos, m_label, "#FEF3C7")  # Yellow tint
    draw_box(y_pos, y_label, "#DCFCE7")  # Green tint
    
    # Arrow styling
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=6"
    sig_color = "#C42847"
    nonsig_color = "#666666"
    
    def is_sig(p):
        return np.isfinite(p) and p < 0.05
    
    def format_coef(val, p):
        sig = "*" if is_sig(p) else ""
        return f"{val:.3f}{sig}"
    
    # Path a: X → M
    a_color = sig_color if is_sig(result.a_p) else nonsig_color
    ax.annotate(
        "", xy=(m_pos[0] - box_width/2, m_pos[1]), xytext=(x_pos[0] + box_width/2, x_pos[1] + 0.3),
        arrowprops=dict(arrowstyle=arrow_style, color=a_color, lw=2)
    )
    ax.text(
        (x_pos[0] + m_pos[0])/2 - 0.3, (x_pos[1] + m_pos[1])/2 + 0.5,
        f"a = {format_coef(result.a_path, result.a_p)}",
        fontsize=10, color=a_color, fontweight="bold"
    )
    
    # Path b: M → Y
    b_color = sig_color if is_sig(result.b_p) else nonsig_color
    ax.annotate(
        "", xy=(y_pos[0] - box_width/2, y_pos[1] + 0.3), xytext=(m_pos[0] + box_width/2, m_pos[1]),
        arrowprops=dict(arrowstyle=arrow_style, color=b_color, lw=2)
    )
    ax.text(
        (m_pos[0] + y_pos[0])/2 + 0.3, (m_pos[1] + y_pos[1])/2 + 0.5,
        f"b = {format_coef(result.b_path, result.b_p)}",
        fontsize=10, color=b_color, fontweight="bold"
    )
    
    # Path c' (direct): X → Y (dashed if not significant)
    c_prime_color = sig_color if is_sig(result.c_prime_p) else nonsig_color
    linestyle = "-" if is_sig(result.c_prime_p) else "--"
    ax.annotate(
        "", xy=(y_pos[0] - box_width/2, y_pos[1]), xytext=(x_pos[0] + box_width/2, x_pos[1]),
        arrowprops=dict(arrowstyle=arrow_style, color=c_prime_color, lw=2, linestyle=linestyle)
    )
    ax.text(
        (x_pos[0] + y_pos[0])/2, x_pos[1] - 0.8,
        f"c' = {format_coef(result.c_prime, result.c_prime_p)}",
        fontsize=10, color=c_prime_color, fontweight="bold", ha="center"
    )
    
    # Indirect effect box
    indirect_sig = result.significant
    indirect_color = sig_color if indirect_sig else nonsig_color
    
    ci_text = f"[{result.indirect_ci_low:.3f}, {result.indirect_ci_high:.3f}]"
    indirect_text = f"Indirect (a×b) = {result.indirect_effect:.3f}\n95% CI: {ci_text}"
    
    indirect_box = FancyBboxPatch(
        (3, 0.3), 4, 1.2,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor="white" if not indirect_sig else "#FEE2E2",
        edgecolor=indirect_color, linewidth=2
    )
    ax.add_patch(indirect_box)
    ax.text(5, 0.9, indirect_text, ha="center", va="center", fontsize=9, color=indirect_color)
    
    # Total effect (c) annotation
    ax.text(
        5, 7.5,
        f"Total effect (c) = {format_coef(result.c_path, result.c_p)}   |   N = {result.n}",
        ha="center", va="center", fontsize=10, style="italic"
    )
    
    # Title
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor=sig_color, linewidth=2, label="p < .05"),
        mpatches.Patch(facecolor="white", edgecolor=nonsig_color, linewidth=2, label="p ≥ .05"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_mediation_summary(
    results_df: pd.DataFrame,
    save_path: Path,
    *,
    title: str = "Mediation Analysis Summary",
    figsize: Tuple[float, float] = (10, 8),
    max_mediators: int = 20,
) -> plt.Figure:
    """Create summary plot of multiple mediators with indirect effects and CIs."""
    if results_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No mediation results", ha="center", va="center")
        return fig
    
    df = results_df.copy()
    
    # Sort by indirect effect magnitude
    df["abs_indirect"] = df["indirect_effect"].abs()
    df = df.sort_values("abs_indirect", ascending=False).head(max_mediators)
    
    n_mediators = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_mediators)[::-1]
    
    sig_color = "#C42847"
    nonsig_color = "#666666"
    
    colors = [sig_color if s else nonsig_color for s in df["significant"].values]
    
    # Plot indirect effects with CIs
    indirect = df["indirect_effect"].values
    ci_low = df["indirect_ci_low"].values
    ci_high = df["indirect_ci_high"].values
    
    xerr = np.array([indirect - ci_low, ci_high - indirect])
    ax.errorbar(
        indirect, y_positions, xerr=xerr, fmt="none",
        ecolor=colors, elinewidth=1.5, capsize=3, capthick=1.5
    )
    ax.scatter(indirect, y_positions, c=colors, s=80, zorder=3, edgecolors="white", linewidths=0.5)
    
    # Zero line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    # Labels with proportion mediated
    labels = []
    for _, row in df.iterrows():
        label = row["mediator"]
        if "proportion_mediated" in row and np.isfinite(row["proportion_mediated"]):
            prop = row["proportion_mediated"] * 100
            label = f"{label} ({prop:.1f}%)"
        labels.append(label)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Indirect Effect (a × b)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sig_color, markersize=10, label="Significant (CI excludes 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=nonsig_color, markersize=10, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    ax.set_ylim(-0.5, n_mediators - 0.5)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_mediation_paths_grid(
    results_df: pd.DataFrame,
    save_path: Path,
    *,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 3),
) -> plt.Figure:
    """Create grid of path coefficients for multiple mediators."""
    if results_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    df = results_df.head(9)  # Limit to 9 for 3x3 grid
    n_plots = len(df)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        
        # Bar plot of path coefficients
        paths = ["a", "b", "c'", "a×b"]
        values = [row["a_path"], row["b_path"], row["c_prime"], row["indirect_effect"]]
        p_values = [row["a_p"], row["b_p"], row["c_prime_p"], row["indirect_p"]]
        
        colors = ["#C42847" if p < 0.05 else "#666666" for p in p_values]
        
        bars = ax.barh(paths, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        
        ax.set_title(row["mediator"], fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)
        
        for bar, val, p in zip(bars, values, p_values):
            sig = "*" if p < 0.05 else ""
            x = val + 0.01 if val >= 0 else val - 0.01
            ha = "left" if val >= 0 else "right"
            ax.text(x, bar.get_y() + bar.get_height()/2, f"{val:.2f}{sig}", va="center", ha=ha, fontsize=7)
    
    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig

