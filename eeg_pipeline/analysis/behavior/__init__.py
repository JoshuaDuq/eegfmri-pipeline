"""
Behavioral correlation analysis.

Submodules:
- core: BehaviorContext, CorrelationRecord, utilities
- correlations: Correlation computation
- fdr_correction: FDR correction
- power_roi: Power ROI correlations
- connectivity: Connectivity correlations
- temporal: Time-frequency correlations
- precomputed_correlations: ERD/ERS, spectral, complexity
- condition_correlations: Pain vs non-pain
"""

# Core (commonly used)
from eeg_pipeline.analysis.behavior.core import (
    BehaviorContext,
    CorrelationRecord,
    ComputationResult,
    ComputationStatus,
    build_correlation_record,
    safe_correlation,
    correlate_features_loop,
    save_correlation_results,
)

# FDR
from eeg_pipeline.analysis.behavior.fdr_correction import apply_global_fdr

# Condition correlations
from eeg_pipeline.analysis.behavior.condition_correlations import compute_condition_correlations


###################################################################
# Lazy-import map
###################################################################

_MODULE_MAP = {
    # Core constants
    "MIN_SAMPLES_CHANNEL": "core",
    "MIN_SAMPLES_ROI": "core",
    "MIN_SAMPLES_DEFAULT": "core",
    "MIN_SAMPLES_EDGE": "core",
    "MIN_TRIALS_PER_CONDITION": "core",
    "DEFAULT_ALPHA": "core",
    "CORRELATION_METHODS": "core",
    "align_to_valid": "core",
    # Correlations
    "AnalysisConfig": "correlations",
    # Precomputed
    "compute_precomputed_correlations": "precomputed_correlations",
    "correlate_precomputed_features": "precomputed_correlations",
    "correlate_microstate_features": "precomputed_correlations",
    "classify_feature": "precomputed_correlations",
    "FEATURE_PATTERNS": "precomputed_correlations",
    # Condition
    "split_data_by_condition": "condition_correlations",
    "compare_condition_correlations": "condition_correlations",
    # Power ROI
    "compute_power_roi_stats": "power_roi",
    "compute_power_roi_stats_from_context": "power_roi",
    # Temporal
    "compute_time_frequency_correlations": "temporal",
    "compute_time_frequency_from_context": "temporal",
    "compute_temporal_from_context": "temporal",
    # Cluster tests
    "run_cluster_test_from_context": "cluster_tests",
    # Connectivity
    "correlate_connectivity_roi_from_context": "connectivity",
}


__all__ = [
    "BehaviorContext",
    "CorrelationRecord",
    "ComputationResult",
    "ComputationStatus",
    "build_correlation_record",
    "safe_correlation",
    "correlate_features_loop",
    "save_correlation_results",
    "apply_global_fdr",
    "compute_condition_correlations",
]


def __getattr__(name: str):
    """Lazy import for behavior analysis functions and constants."""
    if name in _MODULE_MAP:
        import importlib

        module = importlib.import_module(f".{_MODULE_MAP[name]}", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

