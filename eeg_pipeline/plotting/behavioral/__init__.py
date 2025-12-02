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

# Comprehensive diagnostics
from .diagnostics import (
    plot_comprehensive_diagnostics,
    compute_leverage_and_cooks,
    compute_normality_tests,
    compute_vif,
)

# Subject-level scatter and correlation plots
from .scatter import (
    plot_psychometrics,
    plot_power_roi_scatter,
    plot_dynamics_roi_scatter,
    plot_aperiodic_roi_scatter,
    plot_connectivity_roi_scatter,
    plot_itpc_roi_scatter,
    plot_behavioral_response_patterns,
    plot_top_behavioral_predictors,
)

# Temporal correlation topomaps
from .temporal import (
    plot_temporal_correlation_topomaps_by_pain,
    plot_pain_nonpain_clusters,
    plot_significant_correlations_topomap,
)
from .dose_response import visualize_dose_response

# Mediation analysis visualization
from .mediation_plots import (
    plot_mediation_path_diagram,
    plot_indirect_effect_distribution,
    plot_mediation_summary_table,
)

# Moderation analysis visualization
from .moderation_plots import (
    plot_simple_slopes,
    plot_johnson_neyman,
)

# Visualization orchestration
from .viz import (
    visualize_subject_behavior,
    visualize_behavior_for_subjects,
)

# Significant plot collector
from .collect_significant import collect_significant_plots


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
    # Visualization orchestration
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
    # Significant collection
    "collect_significant_plots",
]
