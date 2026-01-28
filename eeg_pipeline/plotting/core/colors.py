from __future__ import annotations

from typing import Any

DEFAULT_BAND_COLORS = {
    "delta": "#1f77b4",
    "theta": "#2ca02c",
    "alpha": "#ff7f0e",
    "beta": "#d62728",
    "gamma": "#9467bd",
    "broadband": "#6B7280",
}


def get_band_color(band: str, config: Any = None) -> str:
    """Get color for a frequency band."""
    normalized_band = (band or "").lower()

    if normalized_band in DEFAULT_BAND_COLORS:
        return DEFAULT_BAND_COLORS[normalized_band]

    for band_name, color in DEFAULT_BAND_COLORS.items():
        if band_name in normalized_band:
            return color

    return DEFAULT_BAND_COLORS["broadband"]
