"""
Features plotting module.

Power, connectivity, phase, aperiodic, complexity, spectral, ratios, asymmetry,
bursts, and temporal feature visualizations.
"""

from __future__ import annotations

import importlib

__all__ = [
    "plot_power_spectral_density",
    "plot_cross_frequency_power_correlation",
    "plot_band_power_topomaps",
    "plot_connectivity_heatmap",
    "plot_connectivity_network",
    "plot_itpc_topomaps",
    "plot_aperiodic_topomaps",
    "FeaturePlotContext",
    "plot_complexity_by_condition",
    "plot_ratios_by_condition",
    "plot_asymmetry_by_condition",
]


def __getattr__(name: str):
    _module_map = {
        "plot_power_spectral_density": "power",
        "plot_cross_frequency_power_correlation": "power",
        "plot_band_power_topomaps": "power",
        "plot_connectivity_heatmap": "connectivity",
        "plot_connectivity_network": "connectivity",
        "plot_itpc_topomaps": "phase",
        "plot_aperiodic_topomaps": "aperiodic",
        "FeaturePlotContext": "context",
        "plot_complexity_by_condition": "complexity",
        "plot_ratios_by_condition": "ratios",
        "plot_asymmetry_by_condition": "asymmetry",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
