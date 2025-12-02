"""Behavioral correlation analysis module."""

from eeg_pipeline.analysis.behavior.core import (
    BehaviorContext,
    CorrelationRecord,
    ComputationResult,
    ComputationStatus,
    AnalysisPipelineConfig,
    safe_correlation,
    build_correlation_record,
    correlate_features_loop,
    save_correlation_results,
    iterate_feature_columns,
    build_output_filename,
    get_min_samples,
    get_min_trials,
    run_behavior_analysis_pipeline,
)
from eeg_pipeline.utils.analysis.stats import fisher_z_test
from eeg_pipeline.analysis.behavior.fdr_correction import apply_global_fdr
from eeg_pipeline.analysis.behavior.condition_correlations import compute_condition_correlations


_LAZY = {
    # Unified Feature Correlator (primary entry point)
    "FeatureBehaviorCorrelator": "feature_correlator",
    "CorrelationConfig": "feature_correlator",
    "run_unified_feature_correlations": "feature_correlator",
    "correlate_pain_relevant_features": "feature_correlator",
    "compute_feature_importance_summary": "feature_correlator",
    "generate_feature_coverage_report": "feature_correlator",
    "FEATURE_CLASSIFIERS": "feature_correlator",
    # Precomputed (detailed pattern matching)
    "compute_precomputed_correlations": "precomputed_correlations",
    "correlate_precomputed_features": "precomputed_correlations",
    "FEATURE_PATTERNS": "precomputed_correlations",
    # Condition
    "split_data_by_condition": "condition_correlations",
    "compare_condition_correlations": "condition_correlations",
    "compute_condition_effect_sizes": "condition_correlations",
    "compute_partial_correlations_controlling_temperature": "condition_correlations",
    "FEATURE_CONFIGS": "condition_correlations",
    # Power ROI
    "compute_power_roi_stats": "power_roi",
    "compute_power_roi_stats_from_context": "power_roi",
    # Temporal
    "compute_time_frequency_correlations": "temporal",
    "compute_time_frequency_from_context": "temporal",
    "compute_temporal_from_context": "temporal",
    # Cluster
    "run_cluster_test_from_context": "cluster_tests",
    # Connectivity
    "correlate_connectivity_roi_from_context": "connectivity",
    "compute_sliding_state_metrics": "connectivity",
    # Topomaps
    "correlate_power_topomaps": "topomaps",
    "correlate_power_topomaps_from_context": "topomaps",
    # Exports
    "export_all_significant_predictors": "exports",
    "export_combined_power_corr_stats": "exports",
    "export_analysis_summary": "exports",
    "export_top_predictors": "exports",
    # Mixed Effects
    "fit_mixed_effects_model": "mixed_effects",
    "compute_icc": "mixed_effects",
    "run_multilevel_correlation_analysis": "mixed_effects",
    "MixedEffectsResult": "mixed_effects",
    # Mediation
    "test_mediation": "mediation",
    "run_mediation_analysis": "mediation",
    "MediationResult": "mediation",
}

__all__ = [
    "BehaviorContext", "CorrelationRecord", "ComputationResult", "ComputationStatus",
    "AnalysisPipelineConfig", "safe_correlation", "build_correlation_record",
    "correlate_features_loop", "save_correlation_results", "iterate_feature_columns",
    "fisher_z_test", "build_output_filename", "get_min_samples", "get_min_trials",
    "run_behavior_analysis_pipeline", "apply_global_fdr", "compute_condition_correlations",
    *_LAZY.keys(),
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{_LAZY[name]}", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
