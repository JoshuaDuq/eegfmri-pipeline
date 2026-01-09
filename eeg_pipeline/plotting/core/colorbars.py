"""
Core colorbar utilities.

Colorbar creation and styling functions for plotting.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm

from ..config import get_plot_config


def _get_tfr_colorbar_config(config: Optional[Any] = None) -> dict:
    """Extract TFR colorbar configuration from plot config.
    
    Args:
        config: Optional config dictionary
        
    Returns:
        Dictionary with colorbar configuration values
    """
    plot_cfg = get_plot_config(config) if config else None
    if plot_cfg is None:
        return {}
    
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    return tfr_config.get("colorbar", {})


def _get_colorbar_params(
    config: Optional[Any] = None,
    fraction: Optional[float] = None,
    pad: Optional[float] = None,
    shrink: Optional[float] = None,
) -> tuple[float, float, float]:
    """Get colorbar parameters, using config defaults when not provided.
    
    Args:
        config: Optional config dictionary
        fraction: Optional fraction override
        pad: Optional pad override
        shrink: Optional shrink override
        
    Returns:
        Tuple of (fraction, pad, shrink) values
    """
    colorbar_config = _get_tfr_colorbar_config(config)
    
    default_fraction = colorbar_config.get("fraction", 0.045)
    default_pad = colorbar_config.get("pad", 0.06)
    default_shrink = colorbar_config.get("shrink", 0.9)
    
    return (
        fraction if fraction is not None else default_fraction,
        pad if pad is not None else default_pad,
        shrink if shrink is not None else default_shrink,
    )


def _normalize_axes_to_list(axes: Union[plt.Axes, List[plt.Axes]]) -> List[plt.Axes]:
    """Convert axes input to a list of axes objects.
    
    Handles single axes, lists, and numpy arrays.
    
    Args:
        axes: Single axes, list of axes, or array-like of axes
        
    Returns:
        List of axes objects
    """
    if hasattr(axes, "ravel"):
        return axes.ravel().tolist()
    if isinstance(axes, list):
        return axes
    return [axes]


def _create_scalar_mappable(
    vmin: float,
    vmax: float,
    cmap: Colormap,
    vcenter: Optional[float] = None,
) -> ScalarMappable:
    """Create ScalarMappable with appropriate normalization.
    
    Args:
        vmin: Minimum value
        vmax: Maximum value
        cmap: Colormap to use
        vcenter: Optional center value for TwoSlopeNorm
        
    Returns:
        ScalarMappable instance with empty array set
    """
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def add_normalized_colorbar(
    fig: plt.Figure,
    axes_list: Union[plt.Axes, List[plt.Axes]],
    vmin: float,
    vmax: float,
    cmap: Colormap,
    config: Optional[Any] = None,
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
    fraction, pad, shrink = _get_colorbar_params(config)
    sm = _create_scalar_mappable(vmin, vmax, cmap)
    fig.colorbar(sm, ax=axes_list, fraction=fraction, pad=pad, shrink=shrink)


def create_difference_colorbar(
    fig: plt.Figure,
    axes: Union[plt.Axes, List[plt.Axes]],
    vabs: float,
    cmap: Colormap,
    label: Optional[str] = None,
    fraction: Optional[float] = None,
    pad: Optional[float] = None,
    shrink: Optional[float] = None,
    aspect: float = 20,
    fontsize: int = 9,
    config: Optional[Any] = None,
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
    
    fraction, pad, shrink = _get_colorbar_params(config, fraction, pad, shrink)
    sm = _create_scalar_mappable(-vabs, vabs, cmap, vcenter=0.0)
    axes_list = _normalize_axes_to_list(axes)
    
    cbar = fig.colorbar(
        sm, ax=axes_list, fraction=fraction, pad=pad, shrink=shrink, aspect=aspect
    )
    
    if label:
        cbar.set_label(label, fontsize=fontsize)
    
    return cbar


def add_diff_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    vabs: float,
    cmap: Colormap,
    config: Optional[Any] = None,
) -> None:
    """Add difference colorbar to single axes.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes to attach colorbar to
        vabs: Absolute value for symmetric range (-vabs to +vabs)
        cmap: Colormap to use
        config: Optional config dictionary
    """
    if vabs <= 0:
        return
    
    fraction, pad, shrink = _get_colorbar_params(config)
    sm = _create_scalar_mappable(-vabs, vabs, cmap, vcenter=0.0)
    fig.colorbar(sm, ax=ax, fraction=fraction, pad=pad, shrink=shrink)


def create_colorbar_for_topomaps(
    fig: plt.Figure,
    axes: Union[plt.Axes, List[plt.Axes]],
    vmin: float,
    vmax: float,
    cmap: Colormap,
    colorbar_pad: float,
    colorbar_fraction: float,
    config: Optional[Any] = None,
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
    sm = _create_scalar_mappable(vmin, vmax, cmap, vcenter=0.0)
    
    colorbar_config = _get_tfr_colorbar_config(config)
    colorbar_multiplier = colorbar_config.get("multiplier", 8.0)
    pad = colorbar_pad * colorbar_multiplier
    
    fig.colorbar(sm, ax=axes, fraction=colorbar_fraction, pad=pad)


def add_colorbar(
    fig: plt.Figure,
    axes: List[plt.Axes],
    successful_plots: List,
    config: Optional[Any] = None,
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
    
    if not axes:
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
    cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation="horizontal")
    cbar.set_label("Spearman correlation (ρ)", fontweight="bold", fontsize=label_fontsize)
    cbar.ax.tick_params(pad=tick_pad, labelsize=tick_fontsize)

