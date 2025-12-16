"""
Features plotting module.

Power, microstates, connectivity, phase, and aperiodic feature visualizations.
"""

from __future__ import annotations

import importlib


def visualize_features(*args, **kwargs):
    from eeg_pipeline.plotting.orchestration.features import visualize_features as _impl

    return _impl(*args, **kwargs)


def visualize_features_for_subjects(*args, **kwargs):
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects as _impl

    return _impl(*args, **kwargs)

__all__ = [
    # Power plotting
    "plot_channel_power_heatmap",
    "plot_power_time_courses",
    "plot_power_spectral_density",
    "plot_power_spectral_density_by_pain",
    "plot_power_time_course_by_temperature",
    "plot_trial_power_variability",
    "plot_inter_band_spatial_power_correlation",
    "plot_power_variability_comprehensive",
    "plot_cross_frequency_power_correlation",
    "plot_feature_stability_heatmap",
    "plot_temporal_autocorrelation",
    "plot_feature_redundancy_matrix",
    "plot_band_power_topomaps",
    "plot_spectral_slope_topomap",
    "plot_feature_importance_ranking",
    # Microstate plotting
    "plot_microstate_templates",
    "plot_microstate_templates_by_pain",
    "plot_microstate_templates_by_temperature",
    "plot_microstate_coverage_by_pain",
    "plot_microstate_temporal_evolution",
    "plot_microstate_gfp_colored_by_state",
    "plot_microstate_gfp_by_temporal_bins",
    "plot_microstate_transition_network",
    "plot_microstate_duration_distributions",
    # PAC plotting
    "plot_pac_summary",
    # Connectivity plotting
    "plot_sliding_connectivity_trajectories",
    "plot_sliding_degree_heatmap",
    "plot_edge_significance_circle_from_stats",
    "plot_graph_metric_distributions",
    "plot_graph_metrics_bar",
    "plot_rsn_radar",
    "plot_connectivity_heatmap",
    "plot_connectivity_network",
    "plot_sliding_state_centroids",
    "plot_sliding_state_sequences",
    "plot_sliding_state_occupancy_boxplot",
    "plot_sliding_state_occupancy_ribbons",
    "plot_sliding_state_lagged_correlation_surfaces",
    # Phase plotting
    "plot_itpc_heatmap",
    "plot_itpc_topomaps",
    "plot_pac_comodulograms",
    "plot_pac_time_ribbons",
    # Aperiodic plotting
    "plot_aperiodic_residual_spectra",
    "plot_aperiodic_run_trajectories",
    "plot_aperiodic_topomaps",
    "plot_aperiodic_vs_pain",
    # Visualization orchestration
    "visualize_features",
    "visualize_features_for_subjects",
    "FeaturePlotContext",
    # CFC visualizations

    # Dynamics visualizations
    "plot_autocorrelation_decay",
    "plot_mse_complexity_curves",
    "plot_neural_timescale_comparison",
    "plot_dynamics_behavior_grid",
    # Quality visualizations
    "plot_feature_distribution_grid",
    "plot_outlier_trials_heatmap",
    "plot_snr_distribution",
    "plot_missing_data_matrix",
    "plot_reliability_summary",
    "plot_quality_summary_dashboard",
    # Complexity visualizations (new)
    "plot_hjorth_by_band",
    "plot_complexity_by_condition",
    # ERDS visualizations
    "plot_erds_temporal_evolution",
    "plot_erds_latency_distribution",
    "plot_erds_erd_ers_separation",
    "plot_erds_global_summary",
    # Burst/Dynamics visualizations (new)
    "plot_burst_duration_distribution",
    "plot_burst_amplitude_distribution",
    "plot_burst_summary_by_band",
    "plot_gfp_by_band",
    "plot_power_fano_factor",
    "plot_power_logratio",
    "plot_gamma_ramp_bursts",
    "plot_dynamics_by_condition",
]


def __getattr__(name: str):
    _module_map = {
        # Power plotting
        "plot_channel_power_heatmap": "power",
        "plot_power_time_courses": "power",
        "plot_power_spectral_density": "power",
        "plot_power_spectral_density_by_pain": "power",
        "plot_power_time_course_by_temperature": "power",
        "plot_trial_power_variability": "power",
        "plot_inter_band_spatial_power_correlation": "power",
        "plot_power_variability_comprehensive": "power",
        "plot_cross_frequency_power_correlation": "power",
        "plot_feature_stability_heatmap": "power",
        "plot_temporal_autocorrelation": "power",
        "plot_feature_redundancy_matrix": "power",
        "plot_band_power_topomaps": "power",
        "plot_spectral_slope_topomap": "power",
        "plot_feature_importance_ranking": "power",
        # Microstate plotting
        "plot_microstate_templates": "microstates",
        "plot_microstate_templates_by_pain": "microstates",
        "plot_microstate_templates_by_temperature": "microstates",
        "plot_microstate_coverage_by_pain": "microstates",
        "plot_microstate_temporal_evolution": "microstates",
        "plot_microstate_gfp_colored_by_state": "microstates",
        "plot_microstate_gfp_by_temporal_bins": "microstates",
        "plot_microstate_transition_network": "microstates",
        "plot_microstate_duration_distributions": "microstates",
        # PAC plotting
        "plot_pac_summary": "phase",
        # Connectivity plotting
        "plot_sliding_connectivity_trajectories": "connectivity",
        "plot_sliding_degree_heatmap": "connectivity",
        "plot_edge_significance_circle_from_stats": "connectivity",
        "plot_graph_metric_distributions": "connectivity",
        "plot_graph_metrics_bar": "connectivity",
        "plot_rsn_radar": "connectivity",
        "plot_connectivity_heatmap": "connectivity",
        "plot_connectivity_network": "connectivity",
        "plot_sliding_state_centroids": "connectivity",
        "plot_sliding_state_sequences": "connectivity",
        "plot_sliding_state_occupancy_boxplot": "connectivity",
        "plot_sliding_state_occupancy_ribbons": "connectivity",
        "plot_sliding_state_lagged_correlation_surfaces": "connectivity",
        # Phase plotting
        "plot_itpc_heatmap": "phase",
        "plot_itpc_topomaps": "phase",
        "plot_pac_comodulograms": "phase",
        "plot_pac_time_ribbons": "phase",
        # Aperiodic plotting
        "plot_aperiodic_residual_spectra": "aperiodic",
        "plot_aperiodic_run_trajectories": "aperiodic",
        "plot_aperiodic_topomaps": "aperiodic",
        "plot_aperiodic_vs_pain": "aperiodic",
        # Visualization orchestration is exposed as top-level wrappers above
        # Context
        "FeaturePlotContext": "context",
        # Dynamics visualizations
        "plot_autocorrelation_decay": "dynamics",
        "plot_mse_complexity_curves": "dynamics",
        "plot_neural_timescale_comparison": "dynamics",
        "plot_dynamics_behavior_grid": "dynamics",
        # Quality visualizations
        "plot_feature_distribution_grid": "quality",
        "plot_outlier_trials_heatmap": "quality",
        "plot_snr_distribution": "quality",
        "plot_missing_data_matrix": "quality",
        "plot_reliability_summary": "quality",
        "plot_quality_summary_dashboard": "quality",
        # Complexity visualizations
        "plot_hjorth_by_band": "complexity",
        "plot_complexity_by_condition": "complexity",
        # ERDS visualizations
        "plot_erds_temporal_evolution": "erds",
        "plot_erds_latency_distribution": "erds",
        "plot_erds_erd_ers_separation": "erds",
        "plot_erds_global_summary": "erds",
        # Burst/Dynamics visualizations
        "plot_burst_duration_distribution": "burst",
        "plot_burst_amplitude_distribution": "burst",
        "plot_burst_summary_by_band": "burst",
        "plot_gfp_by_band": "burst",
        "plot_power_fano_factor": "burst",
        "plot_power_logratio": "burst",
        "plot_gamma_ramp_bursts": "burst",
        "plot_dynamics_by_condition": "burst",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
