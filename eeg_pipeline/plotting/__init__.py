"""
EEG Pipeline Plotting Module.

Submodules:
- core: Shared utilities (colorbars, annotations, significance)
- erp: Event-related potential plots
- features: Feature distribution and microstate plots
- tfr: Time-frequency representation plots
- behavioral: Brain-behavior correlation plots
- decoding: ML decoding performance plots

Usage:
    # Import specific functions from submodules
    from eeg_pipeline.plotting.features import plot_power_distributions
    from eeg_pipeline.plotting.behavioral import visualize_behavior_for_subjects

    # Or use high-level visualizers
    from eeg_pipeline.plotting import visualize_erp_for_subjects

Lazy Imports:
    Many plotting functions use lazy imports to reduce startup time.
    Access them directly from this module - they will be loaded on first use:
    from eeg_pipeline.plotting import plot_power_distributions
"""

# Configuration
from .config import get_plot_config, PlotConfig

__all__ = [
    # Configuration
    "get_plot_config",
    "PlotConfig",
    # Core utilities
    "get_font_sizes",
    "add_colorbar",
    "compute_cluster_significance",
    "build_statistical_title",
    # High-level visualizers (lazy via __getattr__)
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
    "visualize_features_for_subjects",
    "visualize_subject_tfr",
    "visualize_tfr_for_subjects",
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
]


def __getattr__(name: str):
    """Lazy import for backward compatibility with all plotting functions."""
    # Map function names to their submodules
    _module_map = {
        # Core
        "log": "core",
        "format_cluster_significance_info": "core",
        "format_ttest_significance_info": "core",
        "build_significance_info": "core",
        "extract_valid_pvalues": "core",
        "apply_fdr_correction_to_roi_pvalues": "core",
        "find_annotation_x_position": "core",
        "build_roi_annotation_label": "core",
        "render_roi_annotations": "core",
        "add_roi_annotations": "core",
        "add_normalized_colorbar": "core",
        "create_difference_colorbar": "core",
        "add_diff_colorbar": "core",
        "create_colorbar_for_topomaps": "core",
        "build_topomap_diff_label": "core",
        "build_topomap_percentage_label": "core",
        "create_scalpmean_tfr_from_existing": "core",
        "get_strict_mode": "core",
        "compute_cluster_significance_from_combined": "core",
        "compute_significance_mask": "core",
        # ERP
        "erp_contrast_pain": "erp",
        "erp_by_temperature": "erp",
        "visualize_subject_erp": "erp",
        "visualize_erp_for_subjects": "erp",
        # Features
        "plot_power_distributions": "features",
        "plot_channel_power_heatmap": "features",
        "plot_power_time_courses": "features",
        "plot_power_spectral_density": "features",
        "plot_power_spectral_density_by_pain": "features",
        "plot_power_time_course_by_temperature": "features",
        "plot_trial_power_variability": "features",
        "plot_inter_band_spatial_power_correlation": "features",
        "plot_microstate_templates": "features",
        "plot_microstate_templates_by_pain": "features",
        "plot_microstate_templates_by_temperature": "features",
        "plot_microstate_coverage_by_pain": "features",
        "plot_microstate_temporal_evolution": "features",
        "plot_microstate_gfp_colored_by_state": "features",
        "plot_microstate_gfp_by_temporal_bins": "features",
        "plot_microstate_transition_network": "features",
        "plot_microstate_duration_distributions": "features",
        "plot_connectivity_circle_for_band": "features",
        "plot_sliding_connectivity_trajectories": "features",
        "plot_sliding_degree_heatmap": "features",
        "plot_edge_significance_circle_from_stats": "features",
        "plot_graph_metric_distributions": "features",
        "plot_graph_metrics_bar": "features",
        "plot_rsn_radar": "features",
        "plot_connectivity_heatmap": "features",
        "plot_connectivity_network": "features",
        "plot_sliding_state_centroids": "features",
        "plot_sliding_state_sequences": "features",
        "plot_sliding_state_occupancy_boxplot": "features",
        "plot_sliding_state_occupancy_ribbons": "features",
        "plot_sliding_state_lagged_correlation_surfaces": "features",
        "plot_itpc_heatmap": "features",
        "plot_itpc_topomaps": "features",
        "plot_pac_comodulograms": "features",
        "plot_pac_time_ribbons": "features",
        "plot_aperiodic_residual_spectra": "features",
        "plot_aperiodic_run_trajectories": "features",
        "plot_aperiodic_topomaps": "features",
        "plot_aperiodic_vs_pain": "features",
        # TFR
        "plot_cz_all_trials_raw": "tfr",
        "plot_cz_all_trials": "tfr",
        "plot_channels_all_trials": "tfr",
        "plot_scalpmean_all_trials": "tfr",
        "contrast_scalpmean_pain_nonpain": "tfr",
        "contrast_maxmin_temperature": "tfr",
        "plot_bands_pain_temp_contrasts": "tfr",
        "compute_roi_tfrs": "tfr",
        "plot_rois_all_trials": "tfr",
        "plot_topomap_grid_baseline_temps": "tfr",
        "plot_pain_nonpain_temporal_topomaps_diff_allbands": "tfr",
        "plot_temporal_topomaps_allbands_plateau": "tfr",
        "visualize_subject_tfr": "tfr",
        "visualize_tfr_for_subjects": "tfr",
        # Behavioral
        "generate_correlation_scatter": "behavioral",
        "plot_residual_qc": "behavioral",
        "plot_regression_residual_diagnostics": "behavioral",
        "plot_psychometrics": "behavioral",
        "plot_power_roi_scatter": "behavioral",
        "plot_temporal_correlation_topomaps_by_pain": "behavioral",
        "plot_pain_nonpain_clusters": "behavioral",
        "plot_regressor_distributions": "behavioral",
        "plot_pac_behavior_correlations": "behavioral",
        "plot_itpc_rating_scatter_grid": "behavioral",
        "visualize_subject_behavior": "behavioral",
        "visualize_behavior_for_subjects": "behavioral",
        # Decoding
        "plot_time_generalization_matrix": "decoding",
        "plot_time_generalization_with_null": "decoding",
        "plot_prediction_scatter": "decoding",
        "plot_per_subject_performance": "decoding",
        "plot_decoding_null_hist": "decoding",
        "plot_calibration_curve": "decoding",
        "plot_bootstrap_distributions": "decoding",
        "plot_permutation_null": "decoding",
        "plot_residual_diagnostics": "decoding",
        "plot_model_comparison": "decoding",
        "plot_riemann_band_comparison": "decoding",
        "plot_riemann_sliding_window": "decoding",
        "plot_incremental_validity": "decoding",
        "plot_feature_importance_top_n": "decoding",
        "plot_feature_importance_stability": "decoding",
        "visualize_features_for_subjects": "features",
        "visualize_tfr_for_subjects": "tfr",
        "visualize_behavior_for_subjects": "behavioral",
    }

    if name in _module_map:
        import importlib
        module = importlib.import_module(f".{_module_map[name]}", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

