"""
Behavioral correlation plotting module.

Low-level plotting primitives live here. High-level orchestration/IO is defined in
`pipelines.viz.behavior` to keep responsibilities separated.
"""

from __future__ import annotations

import importlib


def visualize_subject_behavior(*args, **kwargs):
    """Lazy import wrapper to avoid circular imports."""
    from eeg_pipeline.plotting.orchestration.behavior import visualize_subject_behavior as _impl
    return _impl(*args, **kwargs)


def visualize_behavior_for_subjects(*args, **kwargs):
    """Lazy import wrapper to avoid circular imports."""
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects as _impl
    return _impl(*args, **kwargs)


def collect_significant_plots(*args, **kwargs):
    """Lazy import wrapper to avoid circular imports."""
    from eeg_pipeline.plotting.io.collections import collect_significant_plots as _impl
    return _impl(*args, **kwargs)


__all__ = [
    # Primary scatter API
    "plot_behavior_scatter",
    "AggregationMode",
    # Builders
    "generate_correlation_scatter",
    "plot_residual_qc",
    "plot_regression_residual_diagnostics",
    # Other scatter plots
    "plot_psychometrics",
    # Temporal
    "plot_temporal_correlation_topomaps_by_pain",
    "plot_significant_correlations_topomap",
    # Dose response
    "visualize_dose_response",
    # Orchestration
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
    "collect_significant_plots",
]


def __getattr__(name: str):
    """Lazy import for plotting functions."""
    _module_map = {
        # Primary API
        "plot_behavior_scatter": "scatter",
        "AggregationMode": "scatter",
        # Builders
        "generate_correlation_scatter": "builders",
        "plot_residual_qc": "builders",
        "plot_regression_residual_diagnostics": "builders",
        # Scatter
        "plot_psychometrics": "scatter",
        # Temporal
        "plot_temporal_correlation_topomaps_by_pain": "temporal",
        "plot_significant_correlations_topomap": "temporal",
        # Dose response
        "visualize_dose_response": "dose_response",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
