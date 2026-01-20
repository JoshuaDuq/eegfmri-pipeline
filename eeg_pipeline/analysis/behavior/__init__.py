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
    classify_feature,
    compute_change_features,
    compute_condition_effects,
    compute_multigroup_condition_effects,
    compute_icc,
    compute_pain_nonpain_time_cluster_test,
    compute_pain_sensitivity_index,
    compute_split_half_reliability,
    compute_temporal_correlations_by_condition,
    compute_temporal_from_context,
    compute_time_frequency_correlations,
    compute_time_frequency_from_context,
    fit_mixed_effects_model,
    get_feature_registry,
    run_cluster_test_from_context,
    run_mediation_analysis,
    run_multilevel_correlation_analysis,
    run_pain_sensitivity_correlations,
    run_power_topomap_correlations,
    run_unified_feature_correlations,
    split_by_condition,
    FeatureRegistry,
    FeatureRule,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    interpret_effect_size,
    interpret_correlation,
)
from eeg_pipeline.utils.analysis.stats.fdr import (
    apply_global_fdr,
    compute_effective_n,
    fdr_correction,
)
from eeg_pipeline.utils.parallel import (
    get_n_jobs,
    parallel_condition_effects,
    parallel_correlate_features,
    parallel_feature_types,
    parallel_map,
    parallel_subjects,
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
    "compute_time_frequency_correlations",
    "compute_temporal_correlations_by_condition",
    # Cluster and topomap helpers
    "compute_pain_nonpain_time_cluster_test",
    "run_cluster_test_from_context",
    "run_power_topomap_correlations",
    # Advanced models
    "fit_mixed_effects_model",
    "compute_icc",
    "run_multilevel_correlation_analysis",
    "MixedEffectsResult",
    "run_mediation_analysis",
    # Feature registry helpers
    "classify_feature",
    "get_feature_registry",
    "FeatureRegistry",
    "FeatureRule",
    # Stats utilities
    "interpret_effect_size",
    "interpret_correlation",
    "fdr_correction",
    "apply_global_fdr",
    "compute_effective_n",
    # Parallel utilities
    "get_n_jobs",
    "parallel_map",
    "parallel_correlate_features",
    "parallel_condition_effects",
    "parallel_feature_types",
    "parallel_subjects",
]
