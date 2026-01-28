"""
Behavior Analysis Module
========================

Correlate EEG features with behavioral measures (pain ratings, temperature).
Statistical entry points are consolidated in eeg_pipeline.analysis.behavior.api.
"""

from eeg_pipeline.analysis.behavior.api import (
    CorrelationConfig,
    CorrelationResult,
    FeatureBehaviorCorrelator,
    FeatureCorrelationResult,
    MixedEffectsResult,
    compute_change_features,
    compute_condition_effects,
    compute_multigroup_condition_effects,
    compute_icc,
    compute_two_condition_time_cluster_test,
    compute_pain_sensitivity_index,
    compute_split_half_reliability,
    compute_temporal_from_context,
    compute_time_frequency_from_context,
    fit_mixed_effects_model,
    run_cluster_test_from_context,
    run_mediation_analysis,
    run_pain_sensitivity_correlations,
    run_unified_feature_correlations,
    split_by_condition,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    interpret_effect_size,
    interpret_correlation,
)
from eeg_pipeline.utils.parallel import (
    get_n_jobs,
    parallel_condition_effects,
    parallel_feature_types,
)

# Note: BehaviorPipeline, BehaviorPipelineConfig, BehaviorPipelineResults
# should be imported directly from eeg_pipeline.pipelines.behavior when needed
# to avoid circular imports.


__all__ = [
    # Unified correlation pipeline
    "FeatureBehaviorCorrelator",
    "CorrelationConfig",
    "FeatureCorrelationResult",
    "run_unified_feature_correlations",
    "run_pain_sensitivity_correlations",
    "compute_pain_sensitivity_index",
    "compute_change_features",
    "compute_split_half_reliability",
    "CorrelationResult",
    # Condition and temporal stats
    "split_by_condition",
    "compute_condition_effects",
    "compute_multigroup_condition_effects",
    "compute_time_frequency_from_context",
    "compute_temporal_from_context",
    # Cluster helpers
    "compute_two_condition_time_cluster_test",
    "run_cluster_test_from_context",
    # Advanced models
    "fit_mixed_effects_model",
    "compute_icc",
    "MixedEffectsResult",
    "run_mediation_analysis",
    # Stats utilities
    "interpret_effect_size",
    "interpret_correlation",
    # Parallel utilities
    "get_n_jobs",
    "parallel_condition_effects",
    "parallel_feature_types",
]
