"""
Core colorbar utilities.

Colorbar creation and styling functions for plotting.
"""

from __future__ import annotations

from typing import List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

from ..config import get_plot_config


###################################################################
# Normalized Colorbars
###################################################################


def add_normalized_colorbar(
    fig: plt.Figure,
    axes_list: Union[plt.Axes, List[plt.Axes]],
    vmin: float,
    vmax: float,
    cmap,
    config=None,
) -> None:
    """Add normalized colorbar to figure with multiple axes.
    
    Args:
        fig: Matplotlib figure
        axes_list: Single axes or list of axes to attach colorbar to
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap to use
        config: Optional config dictionary
    """
    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    colorbar_config = tfr_config.get("colorbar", {})
    cbar_fraction = colorbar_config.get("fraction", 0.045)
    cbar_pad = colorbar_config.get("pad", 0.06)
    cbar_shrink = colorbar_config.get("shrink", 0.9)
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_list, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)


###################################################################
# Difference Colorbars
###################################################################


def create_difference_colorbar(
    fig: plt.Figure,
    axes: Union[plt.Axes, List[plt.Axes]],
    vabs: float,
    cmap,
    label: Optional[str] = None,
    fraction: Optional[float] = None,
    pad: Optional[float] = None,
    shrink: Optional[float] = None,
    aspect: float = 20,
    fontsize: int = 9,
    config=None,
) -> Optional[plt.colorbar.Colorbar]:
    """Create difference colorbar with symmetric range around zero.
    
    Args:
        fig: Matplotlib figure
        axes: Single axes or list/array of axes to attach colorbar to
        vabs: Absolute value for symmetric range (-vabs to +vabs)
        cmap: Colormap to use
        label: Optional label for colorbar
        fraction: Optional colorbar fraction (overrides config)
        pad: Optional colorbar pad (overrides config)
        shrink: Optional colorbar shrink (overrides config)
        aspect: Colorbar aspect ratio
        fontsize: Font size for colorbar label
        config: Optional config dictionary
    
    Returns:
        Colorbar object, or None if vabs <= 0
    """
    if vabs <= 0:
        return None
    
    if fraction is None or pad is None or shrink is None:
        plot_cfg = get_plot_config(config) if config else None
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
        colorbar_config = tfr_config.get("colorbar", {})
        if fraction is None:
            fraction = colorbar_config.get("fraction", 0.045)
        if pad is None:
            pad = colorbar_config.get("pad", 0.06)
        if shrink is None:
            shrink = colorbar_config.get("shrink", 0.9)
    
    sm = ScalarMappable(
        norm=mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs),
        cmap=cmap
    )
    sm.set_array([])
    axes_list = axes.ravel().tolist() if hasattr(axes, 'ravel') else axes if isinstance(axes, list) else [axes]
    cbar = fig.colorbar(sm, ax=axes_list, fraction=fraction, pad=pad, shrink=shrink, aspect=aspect)
    if label:
        cbar.set_label(label, fontsize=fontsize)
    return cbar


def add_diff_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    vabs: float,
    cmap,
    config=None,
) -> None:
    """Add difference colorbar to single axes.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes to attach colorbar to
        vabs: Absolute value for symmetric range (-vabs to +vabs)
        cmap: Colormap to use
        config: Optional config dictionary
    """
    if vabs > 0:
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        colorbar_config = tfr_config.get("colorbar", {})
        cbar_fraction = colorbar_config.get("fraction", 0.045)
        cbar_pad = colorbar_config.get("pad", 0.06)
        cbar_shrink = colorbar_config.get("shrink", 0.9)
        sm = ScalarMappable(
            norm=mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs),
            cmap=cmap
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)


###################################################################
# Topomap Colorbars
###################################################################


def create_colorbar_for_topomaps(
    fig: plt.Figure,
    axes: Union[plt.Axes, List[plt.Axes]],
    vmin: float,
    vmax: float,
    cmap,
    colorbar_pad: float,
    colorbar_fraction: float,
    config=None,
) -> None:
    """Create colorbar for topomap plots with symmetric range.
    
    Args:
        fig: Matplotlib figure
        axes: Single axes or list of axes to attach colorbar to
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap to use
        colorbar_pad: Base padding for colorbar
        colorbar_fraction: Fraction for colorbar
        config: Optional config dictionary
    """
    sm = ScalarMappable(
        norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
        cmap=cmap
    )
    sm.set_array([])
    plot_cfg = get_plot_config(config) if config else None
    if plot_cfg:
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        colorbar_config = tfr_config.get("colorbar", {})
        colorbar_multiplier = colorbar_config.get("multiplier", 8.0)
    else:
        colorbar_multiplier = 8.0
    pad = colorbar_pad * colorbar_multiplier
    fig.colorbar(sm, ax=axes, fraction=colorbar_fraction, pad=pad)


###################################################################
# Behavioral Colorbars
###################################################################


def add_colorbar(
    fig: plt.Figure,
    axes: List[plt.Axes],
    successful_plots: List,
    config=None,
) -> None:
    """Add colorbar for behavioral correlation plots.
    
    Args:
        fig: Matplotlib figure
        axes: List of axes to position colorbar relative to
        successful_plots: List of plot objects (last one used for colorbar)
        config: Optional config dictionary
    """
    if not successful_plots:
        return
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    colorbar_config = behavioral_config.get("colorbar", {})
    
    width_fraction = colorbar_config.get("width_fraction", 0.55)
    left_offset_fraction = colorbar_config.get("left_offset_fraction", 0.225)
    bottom_offset = colorbar_config.get("bottom_offset", 0.12)
    min_bottom = colorbar_config.get("min_bottom", 0.04)
    height = colorbar_config.get("height", 0.028)
    label_fontsize = colorbar_config.get("label_fontsize", 11)
    tick_fontsize = colorbar_config.get("tick_fontsize", 9)
    tick_pad = colorbar_config.get("tick_pad", 2)
    
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    span = right - left
    cb_width = width_fraction * span
    cb_left = left + left_offset_fraction * span
    cb_bottom = max(min_bottom, bottom - bottom_offset)
    cax = fig.add_axes([cb_left, cb_bottom, cb_width, height])
    cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation='horizontal')
    cbar.set_label('Spearman correlation (ρ)', fontweight='bold', fontsize=label_fontsize)
    cbar.ax.tick_params(pad=tick_pad, labelsize=tick_fontsize)

