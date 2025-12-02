"""
Behavioral correlation plotting module.

This module provides functions for creating behavioral correlation visualizations
at multiple levels: scatter plots, temporal topomaps, group analysis, and
orchestration functions for comprehensive visualization workflows.

Modules:
    - builders: Low-level plot construction utilities
    - scatter: Subject-level scatter/correlation plots
    - temporal: Time-frequency and temporal correlation topomaps
    - group: Group-level aggregation plots
    - effect_sizes: Forest plots and effect size visualizations
    - mediation: Mediation analysis path diagrams
    - mixed_effects: Mixed-effects/multilevel model results
    - robust: Robust statistics and sensitivity analyses
    - distributions: Feature distribution summaries
    - summary: Dashboard and summary figures
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
    AVAILABLE_PLOTS,
)

# Effect size visualizations
from .effect_sizes import (
    plot_correlation_forest,
    plot_effect_size_comparison,
    plot_effect_size_heatmap,
    plot_condition_effect_sizes,
    plot_temperature_mediation,
)

# Mediation visualizations
from .mediation import (
    plot_mediation_diagram,
    plot_mediation_summary,
    plot_mediation_paths_grid,
)

# Mixed-effects visualizations
from .mixed_effects import (
    plot_icc_bar_chart,
    plot_variance_decomposition,
    plot_mixed_effects_forest,
    plot_subject_random_effects,
)

# Robust statistics visualizations
from .robust import (
    plot_outlier_influence,
    plot_correlation_methods_comparison,
    plot_bootstrap_ci_comparison,
    plot_sensitivity_analysis,
)

# Distribution visualizations (NEW)
from .distributions import (
    plot_feature_distributions,
    plot_raincloud,
    plot_behavioral_summary,
    plot_feature_by_condition,
    plot_feature_correlation_matrix,
    plot_top_predictors_summary,
)

# Summary dashboard visualizations (NEW)
from .summary import (
    plot_analysis_dashboard,
    plot_group_summary_dashboard,
    plot_quality_overview,
)

# Unified feature-behavior visualization
from .feature_behavior_plots import (
    visualize_feature_behavior_correlations,
    setup_behavior_plot_dirs,
    PlotContext,
    FEATURE_TYPE_COLORS,
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
    "AVAILABLE_PLOTS",
    # Effect size visualizations
    "plot_correlation_forest",
    "plot_effect_size_comparison",
    "plot_effect_size_heatmap",
    "plot_condition_effect_sizes",
    "plot_temperature_mediation",
    # Mediation visualizations
    "plot_mediation_diagram",
    "plot_mediation_summary",
    "plot_mediation_paths_grid",
    # Mixed-effects visualizations
    "plot_icc_bar_chart",
    "plot_variance_decomposition",
    "plot_mixed_effects_forest",
    "plot_subject_random_effects",
    # Robust statistics visualizations
    "plot_outlier_influence",
    "plot_correlation_methods_comparison",
    "plot_bootstrap_ci_comparison",
    "plot_sensitivity_analysis",
    # Distribution visualizations
    "plot_feature_distributions",
    "plot_raincloud",
    "plot_behavioral_summary",
    "plot_feature_by_condition",
    "plot_feature_correlation_matrix",
    "plot_top_predictors_summary",
    # Summary dashboard visualizations
    "plot_analysis_dashboard",
    "plot_group_summary_dashboard",
    "plot_quality_overview",
    # Unified feature-behavior visualization
    "visualize_feature_behavior_correlations",
    "setup_behavior_plot_dirs",
    "PlotContext",
    "FEATURE_TYPE_COLORS",
]

