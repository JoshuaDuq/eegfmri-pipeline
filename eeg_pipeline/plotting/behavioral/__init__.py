"""
Behavioral correlation plotting module.

This module provides functions for creating behavioral correlation visualizations
at multiple levels: scatter plots, temporal topomaps, group analysis, and
orchestration functions for comprehensive visualization workflows.
"""

from __future__ import annotations

# Low-level plot builders
from .builders import (
    generate_correlation_scatter,
    plot_residual_qc,
    plot_regression_residual_diagnostics,
)

# Subject-level scatter and correlation plots
from .scatter import (
    plot_psychometrics,
    plot_power_roi_scatter,
    plot_behavioral_response_patterns,
    plot_top_behavioral_predictors,
)

# Group-level scatter and correlation plots
from .group import (
    plot_group_power_roi_scatter,
)

# Temporal correlation topomaps
from .temporal import (
    plot_temporal_correlation_topomaps_by_temperature,
    plot_temporal_correlation_topomaps_by_pain,
    plot_pain_nonpain_clusters,
    plot_regressor_distributions,
    plot_pac_behavior_correlations,
    plot_itpc_rating_scatter_grid,
    plot_significant_correlations_topomap,
    plot_behavior_modulated_connectivity,
    plot_time_frequency_correlation_heatmap,
)

# Group temporal correlation topomaps
from .temporal_group import (
    plot_group_temporal_topomaps,
)

# Visualization orchestration
from .viz import (
    visualize_subject_behavior,
    visualize_group_behavior,
    visualize_behavior_for_subjects,
)


__all__ = [
    # Low-level plot builders
    "generate_correlation_scatter",
    "plot_residual_qc",
    "plot_regression_residual_diagnostics",
    # Subject-level scatter and correlation plots
    "plot_psychometrics",
    "plot_power_roi_scatter",
    "plot_behavioral_response_patterns",
    "plot_top_behavioral_predictors",
    # Group-level scatter and correlation plots
    "plot_group_power_roi_scatter",
    # Temporal correlation topomaps
    "plot_temporal_correlation_topomaps_by_temperature",
    "plot_temporal_correlation_topomaps_by_pain",
    "plot_pain_nonpain_clusters",
    "plot_regressor_distributions",
    "plot_pac_behavior_correlations",
    "plot_itpc_rating_scatter_grid",
    "plot_significant_correlations_topomap",
    "plot_behavior_modulated_connectivity",
    "plot_time_frequency_correlation_heatmap",
    # Group temporal correlation topomaps
    "plot_group_temporal_topomaps",
    # Visualization orchestration
    "visualize_subject_behavior",
    "visualize_group_behavior",
    "visualize_behavior_for_subjects",
]

