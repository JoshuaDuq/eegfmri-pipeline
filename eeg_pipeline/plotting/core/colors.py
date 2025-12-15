from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.config.loader import get_frequency_band_names


def get_significance_colors(config: Any = None) -> Tuple[str, str]:
    plot_cfg = get_plot_config(config)
    sig_color = plot_cfg.get_color("significant", plot_type="behavioral")
    nonsig_color = plot_cfg.get_color("nonsignificant", plot_type="behavioral")
    return sig_color, nonsig_color


def get_significance_color(
    p_value: float,
    *,
    alpha: float = 0.05,
    config: Any = None,
) -> str:
    sig_color, nonsig_color = get_significance_colors(config)
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return nonsig_color
    return sig_color if p < alpha else nonsig_color


def get_direction_color(
    value: float,
    *,
    p_value: Optional[float] = None,
    alpha: float = 0.05,
    config: Any = None,
) -> str:
    sig_color, nonsig_color = get_significance_colors(config)

    if p_value is not None:
        try:
            p = float(p_value)
        except (TypeError, ValueError):
            return nonsig_color
        if p >= alpha:
            return nonsig_color

    try:
        v = float(value)
    except (TypeError, ValueError):
        return nonsig_color

    return sig_color if v > 0 else "#4C72B0"


def get_band_color(band: str, config: Any = None) -> str:
    default = {
        "delta": "#1f77b4",
        "theta": "#2ca02c",
        "alpha": "#ff7f0e",
        "beta": "#d62728",
        "gamma": "#9467bd",
        "broadband": "#6B7280",
    }

    band_key = (band or "").lower()

    if config is not None and hasattr(config, "get"):
        colors = config.get("visualization.band_colors", None)
        if isinstance(colors, dict) and band_key in colors:
            return str(colors[band_key])

    if band_key in default:
        return default[band_key]

    for key, color in default.items():
        if key in band_key:
            return color

    return default["broadband"]


def get_band_colors(config: Any = None) -> Dict[str, str]:
    bands = get_frequency_band_names(config)
    if not bands:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
    return {band: get_band_color(band, config) for band in bands}
