"""
Machine learning pipeline.

Submodules:
- cv: Cross-validation utilities and fold management
- pipelines: ML pipeline factories (ElasticNet, RF)
- orchestration: High-level CV orchestration (LOSO, within-subject CV)
- time_generalization: Temporal generalization analysis

Data loading: `eeg_pipeline.utils.data.machine_learning` (ML matrices),
`eeg_pipeline.utils.data.feature_io`, `eeg_pipeline.utils.data.epochs`.
"""

# CV utilities
from eeg_pipeline.analysis.machine_learning.cv import (
    create_loso_folds,
    create_inner_cv,
    create_block_aware_cv,
    create_within_subject_folds,
    create_block_aware_inner_cv,
    set_random_seeds,
    determine_inner_n_jobs,
    execute_folds_parallel,
    fit_with_warning_logging,
    grid_search_with_warning_logging,
    safe_pearsonr,
    make_pearsonr_scorer,
    create_scoring_dict,
    aggregate_fold_results,
    compute_subject_level_r,
    compute_metrics,
    create_best_params_record,
    get_inner_cv_splits,
    get_min_channels_required,
)

# Pipeline factories
from eeg_pipeline.analysis.machine_learning.pipelines import (
    create_elasticnet_pipeline,
    create_rf_pipeline,
    build_elasticnet_param_grid,
    build_rf_param_grid,
)

# Data loading (canonical location: eeg_pipeline.utils.data.machine_learning)
from eeg_pipeline.utils.data.machine_learning import load_active_matrix

# Compute-stage orchestration lives in eeg_pipeline.analysis.machine_learning.orchestration.
# Canonical end-user entry point is eeg_pipeline.pipelines.machine_learning.MLPipeline.

# Time generalization
from eeg_pipeline.analysis.machine_learning.time_generalization import (
    time_generalization_regression,
)

# Classification
from eeg_pipeline.analysis.machine_learning.classification import (
    create_svm_pipeline,
    create_logistic_pipeline,
    create_rf_classification_pipeline,
    create_ensemble_pipeline,
    decode_binary_outcome,
    nested_loso_classification,
    ClassificationResult,
)
from eeg_pipeline.analysis.machine_learning.cnn import (
    fit_predict_cnn_binary_classifier,
    nested_loso_cnn_classification,
)

# SHAP feature importance
from eeg_pipeline.analysis.machine_learning.shap_importance import (
    compute_shap_values,
    compute_shap_importance,
    compute_shap_for_cv_folds,
    SHAPResult,
)

# Uncertainty quantification
from eeg_pipeline.analysis.machine_learning.uncertainty import (
    compute_prediction_intervals,
    PredictionIntervalResult,
)

__all__ = [
    # CV utilities
    "create_loso_folds",
    "create_inner_cv",
    "create_block_aware_cv",
    "create_within_subject_folds",
    "create_block_aware_inner_cv",
    "set_random_seeds",
    "determine_inner_n_jobs",
    "execute_folds_parallel",
    "fit_with_warning_logging",
    "grid_search_with_warning_logging",
    "safe_pearsonr",
    "make_pearsonr_scorer",
    "create_scoring_dict",
    "aggregate_fold_results",
    "compute_subject_level_r",
    "compute_metrics",
    "create_best_params_record",
    "get_inner_cv_splits",
    "get_min_channels_required",
    # Pipelines
    "create_elasticnet_pipeline",
    "create_rf_pipeline",
    "build_elasticnet_param_grid",
    "build_rf_param_grid",
    # Data
    "load_active_matrix",
    # Time generalization
    "time_generalization_regression",
    # Classification
    "create_svm_pipeline",
    "create_logistic_pipeline",
    "create_rf_classification_pipeline",
    "create_ensemble_pipeline",
    "decode_binary_outcome",
    "nested_loso_classification",
    "ClassificationResult",
    "fit_predict_cnn_binary_classifier",
    "nested_loso_cnn_classification",
    # SHAP
    "compute_shap_values",
    "compute_shap_importance",
    "compute_shap_for_cv_folds",
    "SHAPResult",
    # Uncertainty
    "compute_prediction_intervals",
    "PredictionIntervalResult",
]
