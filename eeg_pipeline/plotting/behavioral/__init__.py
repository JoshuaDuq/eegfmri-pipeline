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
    # Comprehensive diagnostics
    "plot_comprehensive_diagnostics",
    "compute_leverage_and_cooks",
    "compute_normality_tests",
    "compute_vif",
    # Subject-level scatter and correlation plots
    "plot_psychometrics",
    "plot_power_roi_scatter",
    "plot_dynamics_roi_scatter",
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
    # Mediation analysis
    "plot_mediation_path_diagram",
    "plot_indirect_effect_distribution",
    "plot_mediation_summary_table",
    # Moderation analysis
    "plot_simple_slopes",
    "plot_johnson_neyman",
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
        # Comprehensive diagnostics
        "plot_comprehensive_diagnostics": "diagnostics",
        "compute_leverage_and_cooks": "diagnostics",
        "compute_normality_tests": "diagnostics",
        "compute_vif": "diagnostics",
        # Subject-level scatter and correlation plots
        "plot_psychometrics": "scatter",
        "plot_power_roi_scatter": "scatter",
        "plot_dynamics_roi_scatter": "scatter",
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
        # Mediation
        "plot_mediation_path_diagram": "mediation_plots",
        "plot_indirect_effect_distribution": "mediation_plots",
        "plot_mediation_summary_table": "mediation_plots",
        # Moderation
        "plot_simple_slopes": "moderation_plots",
        "plot_johnson_neyman": "moderation_plots",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)