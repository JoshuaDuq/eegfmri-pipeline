"""Machine learning pipeline exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "cv",
    "pipelines",
    "orchestration",
    "time_generalization",
    "classification",
    "cnn",
    "shap_importance",
    "uncertainty",
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
    "create_elasticnet_pipeline",
    "create_rf_pipeline",
    "build_elasticnet_param_grid",
    "build_rf_param_grid",
    "load_active_matrix",
    "time_generalization_regression",
    "create_svm_pipeline",
    "create_logistic_pipeline",
    "create_rf_classification_pipeline",
    "create_ensemble_pipeline",
    "decode_binary_outcome",
    "nested_loso_classification",
    "ClassificationResult",
    "fit_predict_cnn_binary_classifier",
    "nested_loso_cnn_classification",
    "compute_shap_values",
    "compute_shap_importance",
    "compute_shap_for_cv_folds",
    "SHAPResult",
    "compute_prediction_intervals",
    "PredictionIntervalResult",
]

_SUBMODULES = {
    "cv": "eeg_pipeline.analysis.machine_learning.cv",
    "pipelines": "eeg_pipeline.analysis.machine_learning.pipelines",
    "orchestration": "eeg_pipeline.analysis.machine_learning.orchestration",
    "time_generalization": "eeg_pipeline.analysis.machine_learning.time_generalization",
    "classification": "eeg_pipeline.analysis.machine_learning.classification",
    "cnn": "eeg_pipeline.analysis.machine_learning.cnn",
    "shap_importance": "eeg_pipeline.analysis.machine_learning.shap_importance",
    "uncertainty": "eeg_pipeline.analysis.machine_learning.uncertainty",
}

_EXPORTS = {
    "create_loso_folds": ("eeg_pipeline.analysis.machine_learning.cv", "create_loso_folds"),
    "create_inner_cv": ("eeg_pipeline.analysis.machine_learning.cv", "create_inner_cv"),
    "create_block_aware_cv": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "create_block_aware_cv",
    ),
    "create_within_subject_folds": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "create_within_subject_folds",
    ),
    "create_block_aware_inner_cv": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "create_block_aware_inner_cv",
    ),
    "set_random_seeds": ("eeg_pipeline.analysis.machine_learning.cv", "set_random_seeds"),
    "determine_inner_n_jobs": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "determine_inner_n_jobs",
    ),
    "execute_folds_parallel": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "execute_folds_parallel",
    ),
    "fit_with_warning_logging": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "fit_with_warning_logging",
    ),
    "grid_search_with_warning_logging": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "grid_search_with_warning_logging",
    ),
    "safe_pearsonr": ("eeg_pipeline.analysis.machine_learning.cv", "safe_pearsonr"),
    "make_pearsonr_scorer": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "make_pearsonr_scorer",
    ),
    "create_scoring_dict": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "create_scoring_dict",
    ),
    "aggregate_fold_results": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "aggregate_fold_results",
    ),
    "compute_subject_level_r": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "compute_subject_level_r",
    ),
    "compute_metrics": ("eeg_pipeline.analysis.machine_learning.cv", "compute_metrics"),
    "create_best_params_record": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "create_best_params_record",
    ),
    "get_inner_cv_splits": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "get_inner_cv_splits",
    ),
    "get_min_channels_required": (
        "eeg_pipeline.analysis.machine_learning.cv",
        "get_min_channels_required",
    ),
    "create_elasticnet_pipeline": (
        "eeg_pipeline.analysis.machine_learning.pipelines",
        "create_elasticnet_pipeline",
    ),
    "create_rf_pipeline": (
        "eeg_pipeline.analysis.machine_learning.pipelines",
        "create_rf_pipeline",
    ),
    "build_elasticnet_param_grid": (
        "eeg_pipeline.analysis.machine_learning.pipelines",
        "build_elasticnet_param_grid",
    ),
    "build_rf_param_grid": (
        "eeg_pipeline.analysis.machine_learning.pipelines",
        "build_rf_param_grid",
    ),
    "load_active_matrix": (
        "eeg_pipeline.utils.data.machine_learning",
        "load_active_matrix",
    ),
    "time_generalization_regression": (
        "eeg_pipeline.analysis.machine_learning.time_generalization",
        "time_generalization_regression",
    ),
    "create_svm_pipeline": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "create_svm_pipeline",
    ),
    "create_logistic_pipeline": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "create_logistic_pipeline",
    ),
    "create_rf_classification_pipeline": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "create_rf_classification_pipeline",
    ),
    "create_ensemble_pipeline": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "create_ensemble_pipeline",
    ),
    "decode_binary_outcome": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "decode_binary_outcome",
    ),
    "nested_loso_classification": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "nested_loso_classification",
    ),
    "ClassificationResult": (
        "eeg_pipeline.analysis.machine_learning.classification",
        "ClassificationResult",
    ),
    "fit_predict_cnn_binary_classifier": (
        "eeg_pipeline.analysis.machine_learning.cnn",
        "fit_predict_cnn_binary_classifier",
    ),
    "nested_loso_cnn_classification": (
        "eeg_pipeline.analysis.machine_learning.cnn",
        "nested_loso_cnn_classification",
    ),
    "compute_shap_values": (
        "eeg_pipeline.analysis.machine_learning.shap_importance",
        "compute_shap_values",
    ),
    "compute_shap_importance": (
        "eeg_pipeline.analysis.machine_learning.shap_importance",
        "compute_shap_importance",
    ),
    "compute_shap_for_cv_folds": (
        "eeg_pipeline.analysis.machine_learning.shap_importance",
        "compute_shap_for_cv_folds",
    ),
    "SHAPResult": (
        "eeg_pipeline.analysis.machine_learning.shap_importance",
        "SHAPResult",
    ),
    "compute_prediction_intervals": (
        "eeg_pipeline.analysis.machine_learning.uncertainty",
        "compute_prediction_intervals",
    ),
    "PredictionIntervalResult": (
        "eeg_pipeline.analysis.machine_learning.uncertainty",
        "PredictionIntervalResult",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(_SUBMODULES[name])
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_path)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
