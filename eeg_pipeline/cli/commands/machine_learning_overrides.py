"""Config override helpers for machine learning CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from eeg_pipeline.cli.common import apply_arg_overrides


def _update_path_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with path overrides from arguments."""
    if args.bids_root is not None:
        config.setdefault("paths", {})["bids_root"] = args.bids_root

    if args.deriv_root is not None:
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root


def _parse_max_depth_values(raw_values: List[str]) -> List[Optional[int]]:
    """Parse max_depth grid values, handling 'null'/'none' as None."""
    parsed = []
    for raw in raw_values:
        if raw.lower() in {"none", "null"}:
            parsed.append(None)
        else:
            parsed.append(int(raw))
    return parsed


_MODEL_CONFIG_OVERRIDES = [
    ("elasticnet_alpha_grid", "machine_learning.models.elasticnet.alpha_grid", lambda v: [float(x) for x in v]),
    ("elasticnet_l1_ratio_grid", "machine_learning.models.elasticnet.l1_ratio_grid", lambda v: [float(x) for x in v]),
    ("ridge_alpha_grid", "machine_learning.models.ridge.alpha_grid", lambda v: [float(x) for x in v]),
    ("rf_n_estimators", "machine_learning.models.random_forest.n_estimators", int),
    ("rf_min_samples_split_grid", "machine_learning.models.random_forest.min_samples_split_grid", lambda v: [int(float(x)) for x in v]),
    ("rf_min_samples_leaf_grid", "machine_learning.models.random_forest.min_samples_leaf_grid", lambda v: [int(float(x)) for x in v]),
    ("rf_bootstrap", "machine_learning.models.random_forest.bootstrap", bool),
    ("variance_threshold_grid", "machine_learning.preprocessing.variance_threshold_grid", lambda v: [float(x) for x in v]),
    ("imputer", "machine_learning.preprocessing.imputer_strategy", str),
    ("power_transformer_method", "machine_learning.preprocessing.power_transformer_method", str),
    ("power_transformer_standardize", "machine_learning.preprocessing.power_transformer_standardize", bool),
    ("pca_enabled", "machine_learning.preprocessing.pca.enabled", bool),
    ("pca_n_components", "machine_learning.preprocessing.pca.n_components", float),
    ("pca_whiten", "machine_learning.preprocessing.pca.whiten", bool),
    ("pca_svd_solver", "machine_learning.preprocessing.pca.svd_solver", str),
    ("pca_rng_seed", "machine_learning.preprocessing.pca.random_state", int),
    ("deconfound", "machine_learning.preprocessing.deconfound", bool),
    ("feature_selection_percentile", "machine_learning.preprocessing.feature_selection_percentile", float),
    ("ensemble_calibrate", "machine_learning.classification.calibrate_ensemble", bool),
    ("classification_resampler", "machine_learning.classification.resampler", str),
    ("classification_resampler_seed", "machine_learning.classification.resampler_seed", int),
    ("svm_kernel", "machine_learning.models.svm.kernel", str),
    ("svm_c_grid", "machine_learning.models.svm.C_grid", lambda v: [float(x) for x in v]),
    ("lr_penalty", "machine_learning.models.logistic_regression.penalty", str),
    ("lr_c_grid", "machine_learning.models.logistic_regression.C_grid", lambda v: [float(x) for x in v]),
    ("lr_max_iter", "machine_learning.models.logistic_regression.max_iter", int),
    ("cnn_filters1", "machine_learning.models.cnn.temporal_filters", int),
    ("cnn_filters2", "machine_learning.models.cnn.pointwise_filters", int),
    ("cnn_kernel_size1", "machine_learning.models.cnn.kernel_length", int),
    ("cnn_kernel_size2", "machine_learning.models.cnn.separable_kernel_length", int),
    ("cnn_pool_size", "machine_learning.models.cnn.pool_size", int),
    ("cnn_dense_units", "machine_learning.models.cnn.dense_units", int),
    ("cnn_dropout_conv", "machine_learning.models.cnn.dropout_conv", float),
    ("cnn_dropout_dense", "machine_learning.models.cnn.dropout", float),
    ("cnn_batch_size", "machine_learning.models.cnn.batch_size", int),
    ("cnn_epochs", "machine_learning.models.cnn.max_epochs", int),
    ("cnn_learning_rate", "machine_learning.models.cnn.learning_rate", float),
    ("cnn_patience", "machine_learning.models.cnn.patience", int),
    ("cnn_min_delta", "machine_learning.models.cnn.min_delta", float),
    ("cnn_l2_lambda", "machine_learning.models.cnn.weight_decay", float),
    ("cnn_random_seed", "project.random_state", int),
    ("cv_hygiene", "machine_learning.cv.hygiene_enabled", bool),
    ("cv_permutation_scheme", "machine_learning.cv.permutation_scheme", str),
    ("cv_min_valid_perm_fraction", "machine_learning.cv.min_valid_permutation_fraction", float),
    ("cv_default_n_bins", "machine_learning.cv.default_n_bins", int),
    ("eval_ci_method", "machine_learning.evaluation.ci_method", str),
    ("eval_bootstrap_iterations", "machine_learning.evaluation.bootstrap_iterations", int),
    ("data_covariates_strict", "machine_learning.data.covariates_strict", bool),
    ("data_max_excluded_subject_fraction", "machine_learning.data.max_excluded_subject_fraction", float),
    ("incremental_baseline_alpha", "machine_learning.incremental_validity.baseline_alpha", float),
    ("interpretability_grouped_outputs", "machine_learning.interpretability.grouped_outputs", bool),
    ("timegen_min_subjects", "machine_learning.analysis.time_generalization.min_subjects_per_cell", int),
    ("timegen_min_valid_perm_fraction", "machine_learning.analysis.time_generalization.min_valid_permutation_fraction", float),
    ("class_min_subjects_for_auc", "machine_learning.classification.min_subjects_with_auc_for_inference", int),
    ("class_max_failed_fold_fraction", "machine_learning.classification.max_failed_fold_fraction", float),
    ("strict_regression_continuous", "machine_learning.targets.strict_regression_target_continuous", bool),
]


def _update_model_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with model-specific hyperparameter overrides."""
    if args.rf_max_depth_grid is not None:
        config["machine_learning.models.random_forest.max_depth_grid"] = _parse_max_depth_values(args.rf_max_depth_grid)

    if getattr(args, "svm_gamma_grid", None) is not None:
        gamma_vals = []
        for value in args.svm_gamma_grid:
            try:
                gamma_vals.append(float(value))
            except (TypeError, ValueError):
                gamma_vals.append(str(value))
        config["machine_learning.models.svm.gamma_grid"] = gamma_vals

    if getattr(args, "svm_class_weight", None) is not None:
        config["machine_learning.models.svm.class_weight"] = None if args.svm_class_weight == "none" else args.svm_class_weight
    if getattr(args, "lr_class_weight", None) is not None:
        config["machine_learning.models.logistic_regression.class_weight"] = None if args.lr_class_weight == "none" else args.lr_class_weight
    if getattr(args, "rf_class_weight", None) is not None:
        config["machine_learning.models.random_forest.class_weight"] = None if args.rf_class_weight == "none" else args.rf_class_weight

    if getattr(args, "spatial_regions_allowed", None) is not None:
        config["machine_learning.preprocessing.spatial_regions_allowed"] = [
            str(v).strip() for v in args.spatial_regions_allowed if str(v).strip()
        ]

    apply_arg_overrides(args, config, _MODEL_CONFIG_OVERRIDES)


_ML_PLOT_CONFIG_OVERRIDES = [
    ("ml_plots", "machine_learning.plotting.enabled", bool),
    ("ml_plot_formats", "machine_learning.plotting.formats", lambda v: [str(x).strip().lower() for x in v]),
    ("ml_plot_dpi", "machine_learning.plotting.dpi", int),
    ("ml_plot_top_n_features", "machine_learning.plotting.top_n_features", int),
    ("ml_plot_diagnostics", "machine_learning.plotting.include_diagnostics", bool),
]

_FMRI_SIGNATURE_CONFIG_OVERRIDES = [
    ("fmri_signature_method", "machine_learning.fmri_signature.method", str),
    ("fmri_signature_contrast_name", "machine_learning.fmri_signature.contrast_name", str),
    ("fmri_signature_name", "machine_learning.fmri_signature.signature_name", str),
    ("fmri_signature_metric", "machine_learning.fmri_signature.metric", str),
    ("fmri_signature_normalization", "machine_learning.fmri_signature.normalization", str),
    ("fmri_signature_round_decimals", "machine_learning.fmri_signature.round_decimals", int),
]


def _update_ml_plot_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with ML plotting overrides."""
    apply_arg_overrides(args, config, _ML_PLOT_CONFIG_OVERRIDES)


def _update_fmri_signature_target_config(args: argparse.Namespace, config: Any) -> None:
    """Update config for fMRI signature target loading when requested."""
    apply_arg_overrides(args, config, _FMRI_SIGNATURE_CONFIG_OVERRIDES)
