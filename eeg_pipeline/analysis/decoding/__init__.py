"""
Machine learning decoding analysis.

Submodules:
- cv: Cross-validation utilities and fold management
- pipelines: ML pipeline factories (ElasticNet, RF)
- data: Data loading for decoding
- cross_validation: LOSO and within-subject CV
- time_generalization: Temporal generalization analysis
- permutation: Permutation importance
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

# Data loading
from eeg_pipeline.analysis.decoding.data import (
    load_plateau_matrix,
    load_epoch_windows,
)

# Cross-validation implementations
from eeg_pipeline.analysis.decoding.cross_validation import (
    nested_loso_predictions,
    within_subject_kfold_predictions,
    loso_baseline_predictions,
)

# Time generalization
from eeg_pipeline.analysis.decoding.time_generalization import (
    time_generalization_regression,
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
    "load_plateau_matrix",
    "load_epoch_windows",
    # CV implementations
    "nested_loso_predictions",
    "within_subject_kfold_predictions",
    "loso_baseline_predictions",
    # Time generalization
    "time_generalization_regression",
]

