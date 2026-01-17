from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.config.loader import get_frequency_band_names


DEFAULT_BAND_COLORS = {
    "delta": "#1f77b4",
    "theta": "#2ca02c",
    "alpha": "#ff7f0e",
    "beta": "#d62728",
    "gamma": "#9467bd",
    "broadband": "#6B7280",
}

DEFAULT_NEGATIVE_DIRECTION_COLOR = "#4C72B0"

DEFAULT_SIGNIFICANCE_ALPHA = 0.05


def _to_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_significance_colors(config: Any = None) -> Tuple[str, str]:
    """Get significant and non-significant color pair from config."""
    plot_cfg = get_plot_config(config)
    sig_color = plot_cfg.get_color("significant", plot_type="behavioral")
    nonsig_color = plot_cfg.get_color("nonsignificant", plot_type="behavioral")
    return sig_color, nonsig_color


def get_significance_color(
    p_value: float,
    *,
    alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    config: Any = None,
) -> str:
    """Get color based on p-value significance."""
    sig_color, nonsig_color = get_significance_colors(config)
    p_value_float = _to_float(p_value)
    if p_value_float is None:
        return nonsig_color
    return sig_color if p_value_float < alpha else nonsig_color


def get_direction_color(
    value: float,
    *,
    p_value: Optional[float] = None,
    alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    config: Any = None,
) -> str:
    """Get color based on value direction and optional significance."""
    sig_color, nonsig_color = get_significance_colors(config)

    if p_value is not None:
        p_value_float = _to_float(p_value)
        if p_value_float is None or p_value_float >= alpha:
            return nonsig_color

    value_float = _to_float(value)
    if value_float is None:
        return nonsig_color

    return sig_color if value_float > 0 else DEFAULT_NEGATIVE_DIRECTION_COLOR


def get_band_color(band: str, config: Any = None) -> str:
    """Get color for a frequency band."""
    normalized_band = (band or "").lower()

    if normalized_band in DEFAULT_BAND_COLORS:
        return DEFAULT_BAND_COLORS[normalized_band]

    for band_name, color in DEFAULT_BAND_COLORS.items():
        if band_name in normalized_band:
            return color

    return DEFAULT_BAND_COLORS["broadband"]


def get_band_colors(config: Any = None) -> Dict[str, str]:
    """Get color mapping for all frequency bands."""
    bands = get_frequency_band_names(config)
    if not bands:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
    return {band: get_band_color(band, config) for band in bands}
