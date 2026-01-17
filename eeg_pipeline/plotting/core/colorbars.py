"""
Core colorbar utilities.

Colorbar creation and styling functions for plotting.
"""

from __future__ import annotations

from typing import Any, Optional, Union

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
    if config is None:
        return {}
    
    plot_cfg = get_plot_config(config)
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
    
    return (
        fraction if fraction is not None else colorbar_config.get("fraction", 0.045),
        pad if pad is not None else colorbar_config.get("pad", 0.06),
        shrink if shrink is not None else colorbar_config.get("shrink", 0.9),
    )


def _normalize_axes_to_list(
    axes: Union[plt.Axes, list[plt.Axes], Any]
) -> list[plt.Axes]:
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
    axes_list: Union[plt.Axes, list[plt.Axes]],
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
    axes: Union[plt.Axes, list[plt.Axes]],
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

