"""
Core plotting utilities.

Generic utilities for font sizes, logging, and other common plotting operations.
These utilities have no dependencies on sibling plotting modules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Tuple

from ..config import get_plot_config
from .colors import get_band_colors as _get_band_colors
from .colors import get_significance_colors as _get_significance_colors


_LOGGER = logging.getLogger(__name__)
_VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


def get_font_sizes(plot_cfg: Optional[Any] = None) -> Dict[str, int]:
    """Get font size configuration dictionary.
    
    Args:
        plot_cfg: Optional PlotConfig instance. If None, loads default config.
    
    Returns:
        Dictionary mapping font size names to their values:
        - annotation: Small annotation font size
        - label: Axis label font size
        - title: Subplot title font size
        - ylabel: Y-axis label font size
        - suptitle: Figure suptitle font size
        - figure_title: Main figure title font size
    """
    if plot_cfg is None:
        plot_cfg = get_plot_config(None)
    return {
        "annotation": plot_cfg.font.annotation,
        "label": plot_cfg.font.label,
        "title": plot_cfg.font.title,
        "ylabel": plot_cfg.font.ylabel,
        "suptitle": plot_cfg.font.suptitle,
        "figure_title": plot_cfg.font.figure_title,
    }


def log(
    msg: str,
    logger: Optional[logging.Logger] = None,
    level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Log a message with optional logger and level.
    
    Args:
        msg: Message to log.
        logger: Optional logger instance. If None, uses module logger.
        level: Log level. Must be one of: "debug", "info", "warning", "error", "critical".
            Defaults to "info".
    
    Raises:
        ValueError: If level is not a valid log level.
        AttributeError: If logger does not have the requested log method.
    """
    if logger is None:
        logger = _LOGGER
    
    if level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{level}'. Must be one of: {_VALID_LOG_LEVELS}"
        )
    
    log_method = getattr(logger, level)
    log_method(msg)


def get_significance_colors(config: Optional[Any] = None) -> Tuple[str, str]:
    """Get significant and non-significant color pair from config.
    
    Args:
        config: Optional configuration object. If None, uses default config.
    
    Returns:
        Tuple of (significant_color, nonsignificant_color).
    """
    return _get_significance_colors(config)


def get_band_colors(config: Optional[Any] = None) -> Dict[str, str]:
    """Get color mapping for all frequency bands.
    
    Args:
        config: Optional configuration object. If None, uses default config.
    
    Returns:
        Dictionary mapping band names to their colors.
    """
    return _get_band_colors(config)
