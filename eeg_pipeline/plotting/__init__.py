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
"""

# Configuration
from .config import get_plot_config, PlotConfig

# High-level visualizers (most commonly used)
from .erp import visualize_erp_for_subjects
from .features import visualize_features_for_subjects
from .tfr import visualize_tfr_for_subjects
from .behavioral import visualize_behavior_for_subjects

# Core utilities
from .core import (
    get_font_sizes,
    add_colorbar,
    compute_cluster_significance,
    build_statistical_title,
)


__all__ = [
    # Configuration
    "get_plot_config",
    "PlotConfig",
    # Core utilities
    "get_font_sizes",
    "add_colorbar",
    "compute_cluster_significance",
    "build_statistical_title",
    # High-level visualizers
    "visualize_erp_for_subjects",
    "visualize_features_for_subjects",
    "visualize_tfr_for_subjects",
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
        "group_erp_contrast_pain": "erp",
        "erp_by_temperature": "erp",
        "group_erp_by_temperature": "erp",
        "visualize_subject_erp": "erp",
        "visualize_group_erp": "erp",
        # Features
        "plot_power_distributions": "features",
        "plot_channel_power_heatmap": "features",
        "plot_power_time_courses": "features",
        "plot_power_spectral_density": "features",
        "plot_power_spectral_density_by_pain": "features",
        "plot_power_time_course_by_temperature": "features",
        "plot_trial_power_variability": "features",
        "plot_inter_band_spatial_power_correlation": "features",
        "plot_group_power_plots": "features",
        "plot_group_band_power_time_courses": "features",
        "plot_microstate_templates": "features",
        "plot_microstate_templates_by_pain": "features",
        "plot_microstate_templates_by_temperature": "features",
        "plot_microstate_coverage_by_pain": "features",
        "plot_microstate_pain_correlation_heatmap": "features",
        "plot_microstate_temporal_evolution": "features",
        "plot_microstate_gfp_colored_by_state": "features",
        "plot_microstate_gfp_by_temporal_bins": "features",
        "plot_microstate_transition_network": "features",
        "plot_microstate_duration_distributions": "features",
        "plot_group_microstate_template_stability": "features",
        "plot_group_microstate_transition_summary": "features",
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
        "plot_itpc_behavior_scatter": "features",
        "plot_pac_comodulograms": "features",
        "plot_pac_time_ribbons": "features",
        "plot_pac_behavior_scatter": "features",
        "plot_aperiodic_r2_histogram": "features",
        "plot_aperiodic_residual_spectra": "features",
        "plot_aperiodic_run_trajectories": "features",
        "plot_aperiodic_topomaps": "features",
        "plot_aperiodic_vs_pain": "features",
        "visualize_subject_features": "features",
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
        "group_contrast_maxmin_temperature": "tfr",
        "group_rois_all_trials": "tfr",
        "group_contrast_pain_nonpain_rois": "tfr",
        "group_contrast_pain_nonpain_scalpmean": "tfr",
        "group_plot_bands_pain_temp_contrasts": "tfr",
        "group_plot_topomap_grid_baseline_temps": "tfr",
        "group_plot_pain_nonpain_temporal_topomaps_diff_allbands": "tfr",
        "group_plot_temporal_topomaps_allbands_plateau": "tfr",
        "group_tf_correlation": "tfr",
        "visualize_subject_tfr": "tfr",
        "visualize_group_tfr": "tfr",
        # Behavioral
        "generate_correlation_scatter": "behavioral",
        "plot_residual_qc": "behavioral",
        "plot_regression_residual_diagnostics": "behavioral",
        "plot_psychometrics": "behavioral",
        "plot_power_roi_scatter": "behavioral",
        "plot_group_power_roi_scatter": "behavioral",
        "plot_temporal_correlation_topomaps_by_temperature": "behavioral",
        "plot_temporal_correlation_topomaps_by_pain": "behavioral",
        "plot_pain_nonpain_clusters": "behavioral",
        "plot_regressor_distributions": "behavioral",
        "plot_pac_behavior_correlations": "behavioral",
        "plot_itpc_rating_scatter_grid": "behavioral",
        "plot_group_temporal_topomaps": "behavioral",
        "visualize_subject_behavior": "behavioral",
        "visualize_group_behavior": "behavioral",
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
        "visualize_regression_results": "decoding",
        "visualize_time_generalization": "decoding",
        "visualize_model_comparisons": "decoding",
        "visualize_riemann_analysis": "decoding",
        "visualize_incremental_validity": "decoding",
    }

    if name in _module_map:
        import importlib
        module = importlib.import_module(f".{_module_map[name]}", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

