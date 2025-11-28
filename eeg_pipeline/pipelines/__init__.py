"""
Pipeline orchestration modules.

This package provides high-level pipeline functions for running
complete analysis workflows on EEG data.

Pipelines:
- behavior: Brain-behavior correlation analysis
- features: Feature extraction from epochs
- decoding: Machine learning decoding
- erp: Event-related potential analysis
"""

from eeg_pipeline.pipelines.behavior import (
    process_subject as process_behavior_subject,
    compute_behavior_correlations_for_subjects,
    create_context as create_behavior_context,
    run_computations as run_behavior_computations,
    apply_fdr_and_export,
    initialize_analysis_context,
    ALL_COMPUTATIONS as BEHAVIOR_COMPUTATIONS,
)

from eeg_pipeline.pipelines.features import (
    extract_all_features,
    process_subject as process_features_subject,
    extract_features_for_subjects,
)

from eeg_pipeline.pipelines.erp import (
    get_erp_config,
    load_and_prepare_epochs,
    extract_erp_stats,
    extract_erp_stats_for_subjects,
)

from eeg_pipeline.pipelines.decoding import (
    nested_loso_predictions,
    run_regression_decoding,
    run_time_generalization,
)

__all__ = [
    # Behavior
    "process_behavior_subject",
    "compute_behavior_correlations_for_subjects",
    "create_behavior_context",
    "run_behavior_computations",
    "apply_fdr_and_export",
    "initialize_analysis_context",
    "BEHAVIOR_COMPUTATIONS",
    # Features
    "extract_all_features",
    "process_features_subject",
    "extract_features_for_subjects",
    # ERP
    "get_erp_config",
    "load_and_prepare_epochs",
    "extract_erp_stats",
    "extract_erp_stats_for_subjects",
    # Decoding
    "nested_loso_predictions",
    "run_regression_decoding",
    "run_time_generalization",
]
