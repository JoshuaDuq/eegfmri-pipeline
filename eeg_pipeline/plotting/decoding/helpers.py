"""
Decoding Plotting Helpers (Canonical)
=====================================

Shared helper functions for decoding visualization modules.
All decoding plotting modules should import these helpers from here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import seaborn as sns


def despine(ax) -> None:
    """Remove top and right spines from axis."""
    sns.despine(ax=ax, trim=True)


def calculate_axis_limits(
    values: np.ndarray, 
    plot_cfg, 
    margin_factor: Optional[float] = None
) -> tuple[float, float]:
    """Calculate axis limits with margin."""
    if margin_factor is None:
        margin_factor = plot_cfg.plot_type_configs.get("decoding", {}).get("axis_margin_factor", 0.05)
    value_min = np.nanmin(values)
    value_max = np.nanmax(values)
    margin = (value_max - value_min) * margin_factor
    return value_min - margin, value_max + margin


def calculate_shared_axis_limits(
    values1: np.ndarray, 
    values2: np.ndarray, 
    plot_cfg,
    margin_factor: Optional[float] = None
) -> tuple[float, float]:
    """Calculate shared axis limits for two value arrays."""
    combined_values = np.concatenate([values1, values2])
    return calculate_axis_limits(combined_values, plot_cfg, margin_factor)


def add_zero_reference_line(
    ax, 
    plot_cfg, 
    linewidth: Optional[float] = None, 
    linestyle: str = '--', 
    alpha: Optional[float] = None
) -> None:
    """Add horizontal zero reference line to axis."""
    if linewidth is None:
        linewidth = plot_cfg.style.line.width_thin
    if alpha is None:
        alpha = plot_cfg.style.line.alpha_zero
    ax.axhline(0, color=plot_cfg.style.colors.black, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def create_bar_plot(
    ax, 
    x_positions: np.ndarray, 
    values: np.ndarray, 
    labels: list,
    ylabel: str, 
    plot_cfg, 
    color: Optional[str] = None, 
    alpha: Optional[float] = None, 
    width: Optional[float] = None, 
    add_zero_line: bool = True
) -> None:
    """Create a bar plot with standard styling."""
    if color is None:
        color = plot_cfg.style.colors.gray
    if alpha is None:
        alpha = plot_cfg.style.bar.alpha
    if width is None:
        width = plot_cfg.style.bar.width
    ax.bar(x_positions, values, color=color, alpha=alpha, width=width)
    if add_zero_line:
        add_zero_reference_line(ax, plot_cfg)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    despine(ax)


# Backward compatibility aliases (private names)
_despine = despine
_calculate_axis_limits = calculate_axis_limits
_calculate_shared_axis_limits = calculate_shared_axis_limits
_add_zero_reference_line = add_zero_reference_line
_create_bar_plot = create_bar_plot
