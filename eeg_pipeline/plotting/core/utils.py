"""
Core plotting utilities.

Generic utilities for font sizes, logging, and other common plotting operations.
These utilities have no dependencies on sibling plotting modules.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from ..config import get_plot_config


###################################################################
# Font Utilities
###################################################################


def get_font_sizes(plot_cfg=None) -> Dict[str, int]:
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


###################################################################
# Logging Utilities
###################################################################


def log(msg: str, logger: Optional[logging.Logger] = None, level: str = "info") -> None:
    """Log a message with optional logger and level.
    
    Args:
        msg: Message to log
        logger: Optional logger instance. If None, creates logger for current module.
        level: Log level ("info", "warning", "error", "debug"). Defaults to "info".
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(msg)

