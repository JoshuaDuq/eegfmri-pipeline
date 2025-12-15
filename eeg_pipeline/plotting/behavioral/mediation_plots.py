"""
Mediation Visualization
=======================

Path diagrams and summary visualizations for mediation analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.plotting.io.figures import (
    save_fig,
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
)
from eeg_pipeline.io.logging import get_default_logger as _get_default_logger
from eeg_pipeline.io.paths import ensure_dir
from eeg_pipeline.utils.analysis.stats.mediation import MediationResult


###################################################################
# Path Diagram
###################################################################


def _draw_path_arrow(
    ax: plt.Axes,
    start: tuple,
    end: tuple,
    coef: float,
    se: float,
    p: float,
    label: str,
    is_significant: bool,
    plot_cfg: PlotConfig,
    curved: bool = False,
    above: bool = True,
) -> None:
    """Draw an arrow representing a path coefficient."""
    color = 'green' if is_significant else 'gray'
    linewidth = 2.5 if is_significant else 1.5
    
    if curved:
        connectionstyle = "arc3,rad=0.3" if above else "arc3,rad=-0.3"
    else:
        connectionstyle = "arc3,rad=0"
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=15,
        color=color,
        linewidth=linewidth,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    
    # Label position
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    
    if curved:
        offset = 0.15 if above else -0.15
        mid_y += offset
    else:
        offset = 0.08
        mid_y += offset
    
    # Format coefficient text
    p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    text = f"{label}={coef:.3f}{p_str}"
    if np.isfinite(se):
        text += f"\n(SE={se:.3f})"
    
    ax.text(mid_x, mid_y, text, ha='center', va='center', 
            fontsize=10, fontweight='bold' if is_significant else 'normal',
            color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def _draw_variable_box(
    ax: plt.Axes,
    center: tuple,
    label: str,
    plot_cfg: PlotConfig,
    is_mediator: bool = False,
) -> None:
    """Draw a variable box (rectangle for observed, ellipse for latent)."""
    width, height = 0.25, 0.12
    
    color = '#E3F2FD' if is_mediator else '#FFF3E0'
    edgecolor = '#1976D2' if is_mediator else '#F57C00'
    
    box = FancyBboxPatch(
        (center[0] - width/2, center[1] - height/2),
        width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=2
    )
    ax.add_patch(box)
    
    ax.text(center[0], center[1], label, ha='center', va='center',
            fontsize=11, fontweight='bold')


def plot_mediation_path_diagram(
    result: MediationResult,
    output_path: Path,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    """Create a path diagram visualization for mediation analysis.
    
    Shows the classic mediation model:
    
              M
             /↑\
            a  b
           /    \
          X --c→ Y
          X -c'→ Y
    """
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Variable positions
    x_pos = (0.15, 0.3)   # X: left
    m_pos = (0.5, 0.85)   # M: top center
    y_pos = (0.85, 0.3)   # Y: right
    
    # Draw variable boxes
    _draw_variable_box(ax, x_pos, result.x_label, plot_cfg, is_mediator=False)
    _draw_variable_box(ax, m_pos, result.m_label, plot_cfg, is_mediator=True)
    _draw_variable_box(ax, y_pos, result.y_label, plot_cfg, is_mediator=False)
    
    # Draw paths
    # Path a: X → M
    _draw_path_arrow(
        ax,
        (x_pos[0] + 0.13, x_pos[1] + 0.06),
        (m_pos[0] - 0.10, m_pos[1] - 0.06),
        result.a, result.se_a, result.p_a, 'a',
        result.p_a < 0.05,
        plot_cfg
    )
    
    # Path b: M → Y
    _draw_path_arrow(
        ax,
        (m_pos[0] + 0.10, m_pos[1] - 0.06),
        (y_pos[0] - 0.13, y_pos[1] + 0.06),
        result.b, result.se_b, result.p_b, 'b',
        result.p_b < 0.05,
        plot_cfg
    )
    
    # Path c (total): X → Y (curved, above)
    _draw_path_arrow(
        ax,
        (x_pos[0] + 0.13, x_pos[1]),
        (y_pos[0] - 0.13, y_pos[1]),
        result.c, result.se_c, result.p_c, 'c',
        result.p_c < 0.05,
        plot_cfg,
        curved=True,
        above=False
    )
    
    # Path c' (direct): X → Y (straight, main)
    _draw_path_arrow(
        ax,
        (x_pos[0] + 0.13, x_pos[1] - 0.02),
        (y_pos[0] - 0.13, y_pos[1] - 0.02),
        result.c_prime, result.se_c_prime, result.p_c_prime, "c'",
        result.p_c_prime < 0.05,
        plot_cfg
    )
    
    # Summary annotation
    sig_text = "✓ Significant Mediation" if result.is_significant_mediation() else "✗ No Significant Mediation"
    sig_color = 'green' if result.is_significant_mediation() else 'red'
    
    summary = (
        f"Indirect Effect (a×b): {result.ab:.4f}\n"
        f"95% CI: [{result.ci_ab_low:.4f}, {result.ci_ab_high:.4f}]\n"
        f"Sobel p={result.p_ab:.4f}\n"
        f"Proportion Mediated: {result.proportion_mediated:.1%}\n"
        f"n={result.n}"
    )
    
    ax.text(0.5, 0.05, summary, ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.text(0.5, 0.95, sig_text, ha='center', va='top', fontsize=14, 
            fontweight='bold', color=sig_color)
    
    # Title
    if title is None:
        title = f"Mediation Analysis: {result.x_label} → {result.m_label} → {result.y_label}"
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches='tight',
        pad_inches=0.1,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)
    logger.info(f"Mediation path diagram saved to {output_path}")


def plot_indirect_effect_distribution(
    boot_distribution: np.ndarray,
    observed_ab: float,
    ci_low: float,
    ci_high: float,
    output_path: Path,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    """Plot bootstrap distribution of indirect effect with CIs."""
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histogram
    ax.hist(boot_distribution, bins=50, density=True, alpha=0.7,
            color=plot_cfg.get_color("primary", plot_type="behavioral"),
            edgecolor='white', label='Bootstrap distribution')
    
    # Observed value
    ax.axvline(observed_ab, color='red', linewidth=2, linestyle='-',
               label=f'Observed ab = {observed_ab:.4f}')
    
    # CI bounds
    ax.axvline(ci_low, color='orange', linewidth=1.5, linestyle='--',
               label=f'95% CI lower = {ci_low:.4f}')
    ax.axvline(ci_high, color='orange', linewidth=1.5, linestyle='--',
               label=f'95% CI upper = {ci_high:.4f}')
    
    # Zero reference
    ax.axvline(0, color='gray', linewidth=1, linestyle=':',
               label='Zero (null)')
    
    # Shade CI region
    ax.axvspan(ci_low, ci_high, alpha=0.2, color='orange')
    
    ax.set_xlabel("Indirect Effect (a × b)", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Density", fontsize=plot_cfg.font.label)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if title is None:
        title = "Bootstrap Distribution of Indirect Effect"
    ax.set_title(title, fontsize=plot_cfg.font.title, fontweight='bold')
    
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
    logger.info(f"Indirect effect distribution saved to {output_path}")


def plot_mediation_summary_table(
    results: List[MediationResult],
    output_path: Path,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    """Create a summary table of multiple mediation analyses."""
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    
    if not results:
        logger.warning("No mediation results to plot")
        return
    
    # Prepare table data
    columns = ['Mediator', 'a', 'b', 'c', "c'", 'ab', '95% CI', 'PM', 'Sig']
    rows = []
    
    for r in results:
        ci_str = f"[{r.ci_ab_low:.3f}, {r.ci_ab_high:.3f}]"
        pm_str = f"{r.proportion_mediated:.1%}" if np.isfinite(r.proportion_mediated) else "N/A"
        sig_str = "✓" if r.is_significant_mediation() else ""
        
        rows.append([
            r.m_label[:20],  # Truncate long names
            f"{r.a:.3f}{'*' if r.p_a < 0.05 else ''}",
            f"{r.b:.3f}{'*' if r.p_b < 0.05 else ''}",
            f"{r.c:.3f}{'*' if r.p_c < 0.05 else ''}",
            f"{r.c_prime:.3f}{'*' if r.p_c_prime < 0.05 else ''}",
            f"{r.ab:.4f}",
            ci_str,
            pm_str,
            sig_str,
        ])
    
    # Create figure with table
    n_rows = len(rows)
    fig_height = max(3, 1 + 0.4 * n_rows)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.08, 0.08, 0.08, 0.08, 0.10, 0.18, 0.08, 0.05]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#E0E0E0')
        table[(0, j)].set_text_props(fontweight='bold')
    
    # Color significant rows
    for i, r in enumerate(results):
        if r.is_significant_mediation():
            for j in range(len(columns)):
                table[(i + 1, j)].set_facecolor('#E8F5E9')
    
    if title is None:
        title = "Mediation Analysis Summary"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add footnote
    fig.text(0.5, 0.02, "* p < 0.05; PM = Proportion Mediated; Sig = Significant mediation (CI excludes 0)",
             ha='center', fontsize=8, style='italic')
    
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches='tight',
        pad_inches=0.2,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)
    logger.info(f"Mediation summary table saved to {output_path}")


__all__ = [
    "plot_mediation_path_diagram",
    "plot_indirect_effect_distribution",
    "plot_mediation_summary_table",
]
