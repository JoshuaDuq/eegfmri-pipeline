"""
Features plotting module.

Power, microstates, connectivity, phase, and aperiodic feature visualizations.
"""

from .power import (
    plot_channel_power_heatmap,
    plot_power_time_courses,
    plot_power_spectral_density,
    plot_power_spectral_density_by_pain,
    plot_power_time_course_by_temperature,
    plot_trial_power_variability,
    plot_inter_band_spatial_power_correlation,
    plot_power_variability_comprehensive,
    plot_cross_frequency_power_correlation,
    plot_feature_stability_heatmap,
    plot_temporal_autocorrelation,
    plot_feature_redundancy_matrix,
    plot_band_power_topomaps,
    plot_spectral_slope_topomap,
    plot_feature_importance_ranking,
)
from .dynamics import (
    plot_autocorrelation_decay,
)
from .microstates import (
    plot_microstate_templates,
    plot_microstate_templates_by_pain,
    plot_microstate_templates_by_temperature,
    plot_microstate_coverage_by_pain,
    plot_microstate_temporal_evolution,
    plot_microstate_gfp_colored_by_state,
    plot_microstate_gfp_by_temporal_bins,
    plot_microstate_transition_network,
    plot_microstate_duration_distributions,
)
from .phase import (
    plot_pac_summary,
)
from .connectivity import (
    plot_sliding_connectivity_trajectories,
    plot_sliding_degree_heatmap,
    plot_edge_significance_circle_from_stats,
    plot_graph_metric_distributions,
    plot_graph_metrics_bar,
    plot_rsn_radar,
    plot_connectivity_heatmap,
    plot_connectivity_network,
    plot_sliding_state_centroids,
    plot_sliding_state_sequences,
    plot_sliding_state_occupancy_boxplot,
    plot_sliding_state_occupancy_ribbons,
    plot_sliding_state_lagged_correlation_surfaces,
)
from .phase import (
    plot_itpc_heatmap,
    plot_itpc_topomaps,
    plot_pac_comodulograms,
    plot_pac_time_ribbons,
)
from .aperiodic import (
    plot_aperiodic_residual_spectra,
    plot_aperiodic_run_trajectories,
    plot_aperiodic_topomaps,
    plot_aperiodic_vs_pain,
)

from .dynamics import (
    plot_autocorrelation_decay,
    plot_mse_complexity_curves,
    plot_neural_timescale_comparison,
    plot_dynamics_behavior_grid,
)
from .quality import (
    plot_feature_distribution_grid,
    plot_outlier_trials_heatmap,
    plot_snr_distribution,
    plot_missing_data_matrix,
    plot_reliability_summary,
    plot_quality_summary_dashboard,
)
from .complexity import (
    plot_hjorth_by_band,
    plot_complexity_by_condition,
)
from .erds import (
    plot_erds_temporal_evolution,
    plot_erds_latency_distribution,
    plot_erds_erd_ers_separation,
    plot_erds_global_summary,
)
from .burst import (
    plot_burst_duration_distribution,
    plot_burst_amplitude_distribution,
    plot_burst_summary_by_band,
    plot_gfp_by_band,
    plot_power_fano_factor,
    plot_power_logratio,
    plot_gamma_ramp_bursts,
    plot_dynamics_by_condition,
)
from .visualize import (
    visualize_features,
    visualize_features_for_subjects,
    FeaturePlotContext,
)

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
    "visualize_features_for_subjects",
    "visualize_features",
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
