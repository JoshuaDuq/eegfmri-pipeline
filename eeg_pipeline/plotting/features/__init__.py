"""
Features plotting module.

Power, microstates, connectivity, phase, and aperiodic feature visualizations.
"""

from .power import (
    plot_power_distributions,
    plot_channel_power_heatmap,
    plot_power_time_courses,
    plot_power_spectral_density,
    plot_power_spectral_density_by_pain,
    plot_power_time_course_by_temperature,
    plot_trial_power_variability,
    plot_inter_band_spatial_power_correlation,
)
from .power_group import (
    plot_group_power_plots,
    plot_group_band_power_time_courses,
)
from .microstates import (
    plot_microstate_templates,
    plot_microstate_templates_by_pain,
    plot_microstate_templates_by_temperature,
    plot_microstate_coverage_by_pain,
    plot_microstate_pain_correlation_heatmap,
    plot_microstate_temporal_evolution,
    plot_microstate_gfp_colored_by_state,
    plot_microstate_gfp_by_temporal_bins,
    plot_microstate_transition_network,
    plot_microstate_duration_distributions,
    plot_microstate_transition_heatmaps,
    plot_group_microstate_template_stability,
    plot_group_microstate_transition_summary,
)
from .connectivity import (
    plot_connectivity_circle_for_band,
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
    plot_itpc_behavior_scatter,
    plot_pac_comodulograms,
    plot_pac_time_ribbons,
    plot_pac_behavior_scatter,
)
from .aperiodic import (
    plot_aperiodic_r2_histogram,
    plot_aperiodic_residual_spectra,
    plot_aperiodic_run_trajectories,
    plot_aperiodic_topomaps,
    plot_aperiodic_vs_pain,
)
from .viz import (
    visualize_subject_features,
    visualize_features_for_subjects,
)

__all__ = [
    # Power plotting
    "plot_power_distributions",
    "plot_channel_power_heatmap",
    "plot_power_time_courses",
    "plot_power_spectral_density",
    "plot_power_spectral_density_by_pain",
    "plot_power_time_course_by_temperature",
    "plot_trial_power_variability",
    "plot_inter_band_spatial_power_correlation",
    # Group power plotting
    "plot_group_power_plots",
    "plot_group_band_power_time_courses",
    # Microstate plotting
    "plot_microstate_templates",
    "plot_microstate_templates_by_pain",
    "plot_microstate_templates_by_temperature",
    "plot_microstate_coverage_by_pain",
    "plot_microstate_pain_correlation_heatmap",
    "plot_microstate_temporal_evolution",
    "plot_microstate_gfp_colored_by_state",
    "plot_microstate_gfp_by_temporal_bins",
    "plot_microstate_transition_network",
    "plot_microstate_duration_distributions",
    "plot_microstate_transition_heatmaps",
    "plot_group_microstate_template_stability",
    "plot_group_microstate_transition_summary",
    # Connectivity plotting
    "plot_connectivity_circle_for_band",
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
    "plot_itpc_behavior_scatter",
    "plot_pac_comodulograms",
    "plot_pac_time_ribbons",
    "plot_pac_behavior_scatter",
    # Aperiodic plotting
    "plot_aperiodic_r2_histogram",
    "plot_aperiodic_residual_spectra",
    "plot_aperiodic_run_trajectories",
    "plot_aperiodic_topomaps",
    "plot_aperiodic_vs_pain",
    # Visualization orchestration
    "visualize_subject_features",
    "visualize_features_for_subjects",
]
