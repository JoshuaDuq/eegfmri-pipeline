"""
Machine learning plotting module.

Low-level plotting primitives live here. High-level orchestration/IO is defined in
`plotting.orchestration.machine_learning` to keep responsibilities separated.
"""

from __future__ import annotations

import importlib

__all__ = [
    # Time-generalization plotting
    "plot_time_generalization_with_null",
    # Performance metric plotting
    "plot_prediction_scatter",
    "plot_per_subject_performance",
    "plot_ml_null_hist",
    "plot_calibration_curve",
    "plot_permutation_null",
    # Residual diagnostics plotting
    "plot_residual_diagnostics",
]


def __getattr__(name: str):
    _module_map = {
        # Time-generalization plotting
        "plot_time_generalization_with_null": "time_generalization",
        # Performance metric plotting
        "plot_prediction_scatter": "performance",
        "plot_per_subject_performance": "performance",
        "plot_ml_null_hist": "performance",
        "plot_calibration_curve": "performance",
        "plot_permutation_null": "performance",
        # Residual diagnostics plotting
        "plot_residual_diagnostics": "helpers",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
