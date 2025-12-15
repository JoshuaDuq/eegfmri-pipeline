"""
Decoding plotting module.

Low-level plotting primitives live here. High-level orchestration/IO is defined in
`pipelines.viz.decoding` to keep responsibilities separated.
"""

from __future__ import annotations

import importlib

__all__ = [
    # Time-generalization plotting
    "plot_time_generalization_matrix",
    "plot_time_generalization_with_null",
    # Performance metric plotting
    "plot_prediction_scatter",
    "plot_per_subject_performance",
    "plot_decoding_null_hist",
    "plot_calibration_curve",
    "plot_bootstrap_distributions",
    "plot_permutation_null",
    # Residual diagnostics plotting
    "plot_residual_diagnostics",
    # Model comparison plotting
    "plot_model_comparison",
    "plot_riemann_band_comparison",
    "plot_riemann_sliding_window",
    "plot_incremental_validity",
    # Feature importance plotting
    "plot_feature_importance_top_n",
    "plot_feature_importance_stability",
]


def __getattr__(name: str):
    _module_map = {
        # Time-generalization plotting
        "plot_time_generalization_matrix": "time_generalization",
        "plot_time_generalization_with_null": "time_generalization",
        # Performance metric plotting
        "plot_prediction_scatter": "performance",
        "plot_per_subject_performance": "performance",
        "plot_decoding_null_hist": "performance",
        "plot_calibration_curve": "performance",
        "plot_bootstrap_distributions": "performance",
        "plot_permutation_null": "performance",
        # Residual diagnostics plotting
        "plot_residual_diagnostics": "diagnostics",
        # Model comparison plotting
        "plot_model_comparison": "comparisons",
        "plot_riemann_band_comparison": "comparisons",
        "plot_riemann_sliding_window": "comparisons",
        "plot_incremental_validity": "comparisons",
        # Feature importance plotting
        "plot_feature_importance_top_n": "importance",
        "plot_feature_importance_stability": "importance",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
