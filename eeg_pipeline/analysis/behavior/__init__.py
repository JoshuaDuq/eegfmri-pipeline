"""Behavior Analysis Module
=========================

Correlate EEG features with behavioral measures (pain ratings, temperature).

Modules:
- correlations.py: Enhanced correlations with temperature/trial-order controls
- feature_correlator.py: Feature classification, registry, unified correlator
- cluster_tests.py: Cluster permutation tests with effect sizes
- mixed_effects.py: Linear mixed-effects models, ICC, mediation analysis
- temporal.py: Time-frequency correlations
- topomaps.py: Power topomap correlations
- condition.py: Pain vs non-pain comparison
- stats.py: FDR correction utilities

Usage:
    # Enhanced correlations with controls
    from eeg_pipeline.analysis.behavior import run_correlations
    results_df = run_correlations(features_df, targets, temperature=temp_series)
    
    # Unified feature correlator
    from eeg_pipeline.analysis.behavior import run_unified_feature_correlations
    run_unified_feature_correlations(ctx)
    
    # Cluster tests
    from eeg_pipeline.analysis.behavior import run_cluster_test_from_context
    run_cluster_test_from_context(ctx)
"""

# Core enhanced correlations
from eeg_pipeline.analysis.behavior.correlations import (
    run_correlations,
    run_pain_sensitivity_correlations,
    compute_pain_sensitivity_index,
    compute_change_features,
    compute_split_half_reliability,
    interpret_effect_size,
    interpret_correlation,
    CorrelationResult,
    EFFECT_SIZE_BENCHMARKS,
    CORRELATION_BENCHMARKS,
)

# Feature correlator (classification, registry, unified analysis)
from eeg_pipeline.analysis.behavior.feature_correlator import (
    FeatureBehaviorCorrelator,
    run_unified_feature_correlations,
    correlate_pain_relevant_features,
    classify_feature,
    get_feature_registry,
    FeatureRegistry,
    FeatureRule,
    CorrelationConfig,
)

# Cluster permutation tests
from eeg_pipeline.analysis.behavior.cluster_tests import (
    compute_pain_nonpain_time_cluster_test,
    run_cluster_test_from_context,
    get_cluster_test_config,
    compute_effect_size_map,
)

# Mixed-effects models
from eeg_pipeline.analysis.behavior.mixed_effects import (
    fit_mixed_effects_model,
    compute_icc,
    run_multilevel_correlation_analysis,
    MixedEffectsResult,
    run_mediation_analysis,
)

# Re-export MediationResult from utils for type hints
from eeg_pipeline.utils.analysis.stats.mediation import MediationResult

# Condition comparison
from eeg_pipeline.analysis.behavior.condition import (
    split_by_condition,
    compute_condition_effects,
)

# Temporal correlations
from eeg_pipeline.analysis.behavior.temporal import (
    compute_time_frequency_from_context,
    compute_temporal_from_context,
    compute_time_frequency_correlations,
    compute_temporal_correlations_by_condition,
)

# Topomap correlations
from eeg_pipeline.analysis.behavior.topomaps import (
    correlate_power_topomaps,
    correlate_power_topomaps_from_context,
)

# Statistics (import directly from canonical location)
from eeg_pipeline.utils.analysis.stats.fdr import (
    fdr_correction,
    apply_global_fdr,
    compute_effective_n,
)

# Parallelization
from eeg_pipeline.analysis.behavior.parallel import (
    get_n_jobs,
    parallel_map,
    parallel_correlate_features,
    parallel_condition_effects,
    parallel_feature_types,
    parallel_subjects,
)

# Pipeline orchestration - canonical location is now eeg_pipeline.pipelines.behavior
# Re-exported here for backward compatibility
from eeg_pipeline.pipelines.behavior import (
    run_pipeline,
    run_pipeline_batch,
    compute_behavior_correlations_for_subjects,
    BehaviorPipelineConfig as PipelineConfig,
    BehaviorPipelineResults as PipelineResults,
)


__all__ = [
    # Enhanced correlations
    "run_correlations",
    "run_pain_sensitivity_correlations",
    "compute_pain_sensitivity_index",
    "compute_change_features",
    "compute_split_half_reliability",
    "interpret_effect_size",
    "interpret_correlation",
    "CorrelationResult",
    "EFFECT_SIZE_BENCHMARKS",
    "CORRELATION_BENCHMARKS",
    # Feature correlator
    "FeatureBehaviorCorrelator",
    "run_unified_feature_correlations",
    "correlate_pain_relevant_features",
    "classify_feature",
    "get_feature_registry",
    "FeatureRegistry",
    "FeatureRule",
    "CorrelationConfig",
    # Cluster tests
    "compute_pain_nonpain_time_cluster_test",
    "run_cluster_test_from_context",
    "get_cluster_test_config",
    "compute_effect_size_map",
    # Mixed-effects
    "fit_mixed_effects_model",
    "compute_icc",
    "run_multilevel_correlation_analysis",
    "MixedEffectsResult",
    # Condition
    "split_by_condition",
    "compute_condition_effects",
    # Temporal
    "compute_time_frequency_from_context",
    "compute_temporal_from_context",
    "compute_time_frequency_correlations",
    "compute_temporal_correlations_by_condition",
    # Topomaps
    "correlate_power_topomaps",
    "correlate_power_topomaps_from_context",
    # Stats
    "fdr_correction",
    "apply_global_fdr",
    "compute_effective_n",
    # Mediation
    "run_mediation_analysis",
    "MediationResult",
    # Parallelization
    "get_n_jobs",
    "parallel_map",
    "parallel_correlate_features",
    "parallel_condition_effects",
    "parallel_feature_types",
    "parallel_subjects",
    # Pipeline orchestration
    "run_pipeline",
    "run_pipeline_batch",
    "compute_behavior_correlations_for_subjects",
    "PipelineConfig",
    "PipelineResults",
]
