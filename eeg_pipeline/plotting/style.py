"""
Unified Plotting Style
======================

Centralizes aesthetic configuration for publication-quality figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from typing import Dict, Any

# =============================================================================
# Theme Definitions
# =============================================================================

COLORS = {
    "pain": "#E31A1C",      # Red
    "nopain": "#1F78B4",    # Blue
    "baseline": "#333333",  # Dark Grey
    "grid": "#E0E0E0",
    "text": "#212121",
    "bands": {
        "delta": "#1f77b4", 
        "theta": "#2ca02c", 
        "alpha": "#ff7f0e",
        "beta": "#d62728", 
        "gamma": "#9467bd"
    }
}

# Standard Figure Sizes (inches)
FIG_SIZES = {
    "scatter": (6, 5),
    "time_series": (10, 4),
    "topomap": (4, 4),
    "heatmap": (8, 6),
    "panel": (12, 8),
}

# =============================================================================
# Context Manager
# =============================================================================

@contextmanager
def use_style(context: str = "paper", palette: str = "deep"):
    """
    Context manager to apply standardized plotting style.
    
    Args:
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        palette: Color palette name
    """
    # 1. Base Style
    sns.set_theme(context=context, style="ticks", palette=palette)
    
    # 2. Custom Overrides (Publication Quality)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    
    try:
        yield
    finally:
        plt.rcParams.update(plt.rcParamsDefault)

def get_color(key: str, default: str = "#333333") -> str:
    """Retrieve color from theme."""
    return COLORS.get(key, default)

def get_band_color(band: str) -> str:
    """Retrieve color for frequency band."""
    return COLORS["bands"].get(band, "#333333")
