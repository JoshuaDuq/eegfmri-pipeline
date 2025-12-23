"""
Machine learning decoding analysis.

Submodules:
- cv: Cross-validation utilities and fold management
- pipelines: ML pipeline factories (ElasticNet, RF)
- orchestration: High-level CV orchestration (LOSO, within-subject CV)
- time_generalization: Temporal generalization analysis
- permutation: Permutation importance

Data loading utilities live in `eeg_pipeline.utils.data.decoding` (decoding matrices)
and `eeg_pipeline.utils.data.epochs_loading` / `eeg_pipeline.utils.data.features_io`.
"""

# CV utilities
from eeg_pipeline.analysis.decoding.cv import (
    create_loso_folds,
    create_inner_cv,
    create_stratified_cv_by_binned_targets,
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
    _save_best_params,
    _fit_with_inner_cv,
    _fit_default_pipeline,
    _predict_and_log,
)

# Pipeline factories
from eeg_pipeline.analysis.decoding.pipelines import (
    create_elasticnet_pipeline,
    create_rf_pipeline,
    build_elasticnet_param_grid,
    build_rf_param_grid,
)

# Data loading (canonical location: eeg_pipeline.utils.data.decoding)
from eeg_pipeline.utils.data.decoding import load_epoch_windows, load_active_matrix

# Cross-validation orchestration (high-level CV strategies)
# Canonical pipeline class is in eeg_pipeline.pipelines.decoding
from eeg_pipeline.analysis.decoding.orchestration import (
    nested_loso_predictions,
    within_subject_kfold_predictions,
    loso_baseline_predictions,
    nested_loso_predictions_from_matrix,
)

# Time generalization
from eeg_pipeline.analysis.decoding.time_generalization import (
    time_generalization_regression,
)

# Classification
from eeg_pipeline.analysis.decoding.classification import (
    create_svm_pipeline,
    create_logistic_pipeline,
    create_rf_classification_pipeline,
    create_ensemble_pipeline,
    decode_pain_binary,
    nested_loso_classification,
    ClassificationResult,
    save_classification_results,
)

# SHAP feature importance
from eeg_pipeline.analysis.decoding.shap_importance import (
    compute_shap_values,
    compute_shap_importance,
    compute_shap_for_cv_folds,
    compute_shap_interactions,
    SHAPResult,
    plot_shap_summary,
    plot_shap_bar,
    save_shap_results,
)

# Uncertainty quantification
from eeg_pipeline.analysis.decoding.uncertainty import (
    compute_prediction_intervals,
    calibrate_classifier,
    PredictionIntervalResult,
    CalibrationResult,
    plot_prediction_intervals,
    plot_reliability_diagram,
    save_prediction_intervals,
)

__all__ = [
    # CV utilities
    "create_loso_folds",
    "create_inner_cv",
    "create_stratified_cv_by_binned_targets",
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
    "_save_best_params",
    "_fit_with_inner_cv",
    "_fit_default_pipeline",
    "_predict_and_log",
    # Pipelines
    "create_elasticnet_pipeline",
    "create_rf_pipeline",
    "build_elasticnet_param_grid",
    "build_rf_param_grid",
    # Data
    "load_active_matrix",
    "load_epoch_windows",
    # CV implementations
    "nested_loso_predictions",
    "within_subject_kfold_predictions",
    "loso_baseline_predictions",
    "nested_loso_predictions_from_matrix",
    # Time generalization
    "time_generalization_regression",
    # Classification
    "create_svm_pipeline",
    "create_logistic_pipeline",
    "create_rf_classification_pipeline",
    "create_ensemble_pipeline",
    "decode_pain_binary",
    "nested_loso_classification",
    "ClassificationResult",
    "save_classification_results",
    # SHAP
    "compute_shap_values",
    "compute_shap_importance",
    "compute_shap_for_cv_folds",
    "compute_shap_interactions",
    "SHAPResult",
    "plot_shap_summary",
    "plot_shap_bar",
    "save_shap_results",
    # Uncertainty
    "compute_prediction_intervals",
    "calibrate_classifier",
    "PredictionIntervalResult",
    "CalibrationResult",
    "plot_prediction_intervals",
    "plot_reliability_diagram",
    "save_prediction_intervals",
]

