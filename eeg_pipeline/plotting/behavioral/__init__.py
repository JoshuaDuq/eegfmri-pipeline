"""
Behavioral correlation plotting module.

Low-level plotting primitives live here. High-level orchestration/IO is defined in
`pipelines.viz.behavior` to keep responsibilities separated.
"""

from __future__ import annotations

import importlib

# Visualization orchestration (pipeline-layer; re-exported via lightweight wrappers
# to avoid circular imports during module initialization)
def visualize_subject_behavior(*args, **kwargs):
    from eeg_pipeline.plotting.orchestration.behavior import visualize_subject_behavior as _impl

    return _impl(*args, **kwargs)


def visualize_behavior_for_subjects(*args, **kwargs):
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects as _impl

    return _impl(*args, **kwargs)


def collect_significant_plots(*args, **kwargs):
    from eeg_pipeline.plotting.io.collections import collect_significant_plots as _impl

    return _impl(*args, **kwargs)


__all__ = [
    # Low-level plot builders
    "generate_correlation_scatter",
    "plot_residual_qc",
    "plot_regression_residual_diagnostics",
    # Subject-level scatter and correlation plots
    "plot_psychometrics",
    "plot_power_roi_scatter",
    "plot_complexity_roi_scatter",
    "plot_aperiodic_roi_scatter",
    "plot_connectivity_roi_scatter",
    "plot_itpc_roi_scatter",
    "plot_behavioral_response_patterns",
    "plot_top_behavioral_predictors",
    # Temporal correlation topomaps
    "plot_temporal_correlation_topomaps_by_pain",
    "plot_pain_nonpain_clusters",
    "plot_significant_correlations_topomap",
    "visualize_dose_response",
    # Visualization orchestration (pipeline)
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
    # Significant collection
    "collect_significant_plots",
]


def __getattr__(name: str):
    _module_map = {
        # Low-level plot builders
        "generate_correlation_scatter": "builders",
        "plot_residual_qc": "builders",
        "plot_regression_residual_diagnostics": "builders",
        "plot_psychometrics": "scatter",
        "plot_power_roi_scatter": "scatter",
        "plot_complexity_roi_scatter": "scatter",
        "plot_aperiodic_roi_scatter": "scatter",
        "plot_connectivity_roi_scatter": "scatter",
        "plot_itpc_roi_scatter": "scatter",
        "plot_behavioral_response_patterns": "scatter",
        "plot_top_behavioral_predictors": "scatter",
        # Temporal correlation topomaps
        "plot_temporal_correlation_topomaps_by_pain": "temporal",
        "plot_pain_nonpain_clusters": "temporal",
        "plot_significant_correlations_topomap": "temporal",
        # Dose-response
        "visualize_dose_response": "dose_response",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
