"""
Unified Plotting Style
======================

Centralizes aesthetic configuration for publication-quality figures.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Constants
# =============================================================================

DEFAULT_COLOR = "#333333"

COLORS: Dict[str, str] = {
    "pain": "#E31A1C",
    "nopain": "#1F78B4",
    "baseline": DEFAULT_COLOR,
    "grid": "#E0E0E0",
    "text": "#212121",
}

_PUBLICATION_RCPARAMS: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

# =============================================================================
# Public API
# =============================================================================


@contextmanager
def use_style(context: str = "paper", palette: str = "deep"):
    """
    Apply standardized plotting style within a context.

    Args:
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        palette: Color palette name

    Yields:
        None: Context manager yields control to the block

    Example:
        >>> with use_style(context="paper"):
        ...     plt.plot([1, 2, 3])
    """
    sns.set_theme(context=context, style="ticks", palette=palette)
    plt.rcParams.update(_PUBLICATION_RCPARAMS)

    try:
        yield
    finally:
        plt.rcParams.update(plt.rcParamsDefault)


def get_color(key: str, default: str = DEFAULT_COLOR) -> str:
    """
    Retrieve color from theme by key.

    Args:
        key: Color key (e.g., 'pain', 'nopain', 'baseline')
        default: Default color if key not found

    Returns:
        Hex color string
    """
    if not isinstance(key, str):
        raise TypeError(f"key must be str, got {type(key).__name__}")
    return COLORS.get(key, default)
