"""Machine Learning CLI command."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    resolve_task,
    validate_subjects_not_empty,
    validate_min_subjects,
    create_progress_reporter,
    add_path_args,
    MIN_SUBJECTS_KEY,
    MIN_SUBJECTS_FOR_ML,
)


ML_STAGES = {
    "regression": "LOSO regression predicting pain intensity",
    "timegen": "Time-generalization analysis",
    "classify": "Binary pain classification",
    "model_comparison": "Compare ElasticNet vs Ridge vs RandomForest",
    "incremental_validity": "Quantify Δ performance when adding EEG over baseline",
    "uncertainty": "Conformal prediction intervals for uncertainty quantification",
    "shap": "SHAP-based feature importance",
    "permutation": "Permutation-based feature importance",
}


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model-specific argument overrides."""
    parser.add_argument(
        "--model",
        type=str,
        choices=["elasticnet", "ridge", "rf"],
        default="elasticnet",
        help="Model family: 'elasticnet' (default), 'ridge', or 'rf' (RandomForest).",
    )
    parser.add_argument(
        "--elasticnet-alpha-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override ElasticNet alpha grid (e.g., 0.01 0.1 1 10).",
    )
    parser.add_argument(
        "--elasticnet-l1-ratio-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override ElasticNet l1_ratio grid (e.g., 0.1 0.5 0.9).",
    )
    parser.add_argument(
        "--ridge-alpha-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override Ridge alpha grid (e.g., 0.01 0.1 1 10 100, only used when --model=ridge).",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=None,
        help="Override RandomForest n_estimators (only used when --model=rf).",
    )
    parser.add_argument(
        "--rf-max-depth-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override RandomForest max_depth grid (use 'null' for None, only used when --model=rf).",
    )
    parser.add_argument(
        "--variance-threshold-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override variance_threshold grid (e.g. 0.0 or 0.0 0.01). Use 0.0 only for small train folds (e.g. --cv-scope subject with few subjects).",
    )


def _add_ml_specific_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ML-specific arguments."""
    parser.add_argument(
        "--ml-plots",
        dest="ml_plots",
        action="store_true",
        default=None,
        help="Enable ML output plots (default: enabled).",
    )
    parser.add_argument(
        "--no-ml-plots",
        dest="ml_plots",
        action="store_false",
        help="Disable ML output plots.",
    )
    parser.add_argument(
        "--ml-plot-formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=None,
        help="ML plot output formats (default: png).",
    )
    parser.add_argument(
        "--ml-plot-dpi",
        type=int,
        default=None,
        help="ML plot DPI (default: 300).",
    )
    parser.add_argument(
        "--ml-plot-top-n-features",
        type=int,
        default=None,
        help="Top-N features to display in SHAP/permutation plots (default: 20).",
    )
    parser.add_argument(
        "--ml-plot-diagnostics",
        dest="ml_plot_diagnostics",
        action="store_true",
        default=None,
        help="Enable extended ML diagnostics panels (default: enabled).",
    )
    parser.add_argument(
        "--ml-plot-no-diagnostics",
        dest="ml_plot_diagnostics",
        action="store_false",
        help="Disable extended ML diagnostics panels.",
    )
    parser.add_argument(
        "--uncertainty-alpha",
        type=float,
        default=0.1,
        help="Significance level for prediction intervals (default: 0.1 = 90%% coverage).",
    )
    parser.add_argument(
        "--perm-n-repeats",
        type=int,
        default=10,
        help="Number of repeats for permutation importance (default: 10).",
    )
    parser.add_argument(
        "--cv-scope",
        type=str,
        choices=["group", "subject"],
        default="group",
        help="Cross-validation scope: 'group' (LOSO across selected subjects) or 'subject' (within-subject CV).",
    )
    parser.add_argument("--n-perm", type=int, default=0)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--outer-jobs", type=int, default=1)
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument(
        "--classification-model",
        type=str,
        choices=["svm", "lr", "rf", "cnn"],
        default=None,
        help=(
            "Classifier family for 'classify' stage. Overrides config "
            "'machine_learning.classification.model' when provided."
        ),
    )
    parser.add_argument(
        "--covariates",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional extra predictors appended to the EEG feature matrix (meta column names like "
            "'temperature', 'trial_index', 'block')."
        ),
    )
    parser.add_argument(
        "--require-trial-ml-safe",
        action="store_true",
        help=(
            "Fail fast unless config feature_engineering.analysis_mode='trial_ml_safe'. "
            "Use this to prevent CV leakage from cross-trial feature estimates."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Target to predict. Can be a logical name ('rating', 'temperature', 'pain_binary') "
            "or an explicit events.tsv column name. Use --target=fmri_signature to predict trial-wise "
            "NPS/SIIPS1 expression from fMRI beta-series/LSS. Defaults depend on stage."
        ),
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=None,
        help=(
            "Fixed threshold for binarizing a continuous target when running classification "
            "(e.g., --target=rating --binary-threshold=30). Median-split is intentionally disabled."
        ),
    )
    parser.add_argument(
        "--feature-families",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Feature families to load from derivatives (e.g., power spectral aperiodic connectivity itpc erp). "
            "If omitted, uses config machine_learning.data.feature_families."
        ),
    )
    parser.add_argument(
        "--feature-bands",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional restriction: only keep features whose NamingSchema band matches one of these values "
            "(e.g., alpha beta). If omitted, no band restriction."
        ),
    )
    parser.add_argument(
        "--feature-segments",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional restriction: only keep features whose NamingSchema segment matches one of these values "
            "(e.g., baseline active). If omitted, no segment restriction."
        ),
    )
    parser.add_argument(
        "--feature-scopes",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional restriction: only keep features whose NamingSchema scope matches one of these values "
            "(global roi ch chpair). If omitted, no scope restriction."
        ),
    )
    parser.add_argument(
        "--feature-stats",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional restriction: only keep features whose NamingSchema stat matches one of these values "
            "(e.g. wpli aec mean for connectivity). If omitted, no stat restriction."
        ),
    )
    parser.add_argument(
        "--feature-harmonization",
        type=str,
        choices=["intersection", "union_impute"],
        default=None,
        help=(
            "How to harmonize feature columns across subjects. "
            "'intersection' keeps only shared features (default). "
            "'union_impute' unions columns and relies on imputation for missing values."
        ),
    )
    parser.add_argument(
        "--baseline-predictors",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Baseline predictor columns for incremental_validity (standardized meta names like 'temperature', "
            "'trial_index', 'block'). Defaults to config machine_learning.incremental_validity.baseline_predictors."
        ),
    )
    parser.add_argument("--imputer", choices=["median", "mean", "most_frequent"], default=None)
    parser.add_argument("--power-transformer-method", choices=["yeo-johnson", "box-cox"], default=None)
    parser.add_argument("--power-transformer-standardize", action="store_true", default=None, dest="power_transformer_standardize")
    parser.add_argument("--no-power-transformer-standardize", action="store_false", dest="power_transformer_standardize")
    parser.add_argument("--pca-enabled", action="store_true", default=None, dest="pca_enabled")
    parser.add_argument("--pca-n-components", type=float, default=None, dest="pca_n_components")
    parser.add_argument("--pca-whiten", action="store_true", default=None, dest="pca_whiten")
    parser.add_argument("--pca-svd-solver", choices=["auto", "full", "randomized"], default=None, dest="pca_svd_solver")
    parser.add_argument("--pca-rng-seed", type=int, default=None, dest="pca_rng_seed")
    parser.add_argument("--deconfound", action="store_true", default=None, dest="deconfound")
    parser.add_argument("--feature-selection-percentile", type=float, default=None, dest="feature_selection_percentile")
    parser.add_argument("--ensemble-calibrate", action="store_true", default=None, dest="ensemble_calibrate")
    parser.add_argument("--spatial-regions-allowed", nargs="+", type=str, default=None)
    parser.add_argument(
        "--classification-resampler",
        choices=["none", "undersample", "smote"],
        default=None,
    )
    parser.add_argument("--classification-resampler-seed", type=int, default=None)
    parser.add_argument("--svm-kernel", choices=["rbf", "linear", "poly"], default=None)
    parser.add_argument("--svm-c-grid", nargs="+", type=str, default=None)
    parser.add_argument("--svm-gamma-grid", nargs="+", type=str, default=None)
    parser.add_argument("--svm-class-weight", choices=["balanced", "none"], default=None)
    parser.add_argument("--lr-penalty", choices=["l2", "l1", "elasticnet"], default=None)
    parser.add_argument("--lr-c-grid", nargs="+", type=str, default=None)
    parser.add_argument("--lr-max-iter", type=int, default=None)
    parser.add_argument("--lr-class-weight", choices=["balanced", "none"], default=None)
    parser.add_argument("--rf-min-samples-split-grid", nargs="+", type=str, default=None)
    parser.add_argument("--rf-min-samples-leaf-grid", nargs="+", type=str, default=None)
    parser.add_argument("--rf-bootstrap", action="store_true", default=None, dest="rf_bootstrap")
    parser.add_argument("--no-rf-bootstrap", action="store_false", dest="rf_bootstrap")
    parser.add_argument("--rf-class-weight", choices=["balanced", "balanced_subsample", "none"], default=None)
    parser.add_argument("--cnn-filters1", type=int, default=None)
    parser.add_argument("--cnn-filters2", type=int, default=None)
    parser.add_argument("--cnn-kernel-size1", type=int, default=None)
    parser.add_argument("--cnn-kernel-size2", type=int, default=None)
    parser.add_argument("--cnn-pool-size", type=int, default=None)
    parser.add_argument("--cnn-dense-units", type=int, default=None)
    parser.add_argument("--cnn-dropout-conv", type=float, default=None)
    parser.add_argument("--cnn-dropout-dense", type=float, default=None)
    parser.add_argument("--cnn-batch-size", type=int, default=None)
    parser.add_argument("--cnn-epochs", type=int, default=None)
    parser.add_argument("--cnn-learning-rate", type=float, default=None)
    parser.add_argument("--cnn-patience", type=int, default=None)
    parser.add_argument("--cnn-min-delta", type=float, default=None)
    parser.add_argument("--cnn-l2-lambda", type=float, default=None)
    parser.add_argument("--cnn-random-seed", type=int, default=None)
    parser.add_argument("--cv-hygiene", action="store_true", default=None, dest="cv_hygiene")
    parser.add_argument("--no-cv-hygiene", action="store_false", dest="cv_hygiene")
    parser.add_argument("--cv-permutation-scheme", choices=["within_subject", "within_subject_within_block"], default=None)
    parser.add_argument("--cv-min-valid-perm-fraction", type=float, default=None)
    parser.add_argument("--cv-default-n-bins", type=int, default=None)
    parser.add_argument("--eval-ci-method", choices=["bootstrap", "fixed_effects"], default=None)
    parser.add_argument("--eval-bootstrap-iterations", type=int, default=None)
    parser.add_argument("--data-covariates-strict", action="store_true", default=None, dest="data_covariates_strict")
    parser.add_argument("--no-data-covariates-strict", action="store_false", dest="data_covariates_strict")
    parser.add_argument("--data-max-excluded-subject-fraction", type=float, default=None)
    parser.add_argument("--incremental-baseline-alpha", type=float, default=None)
    parser.add_argument("--interpretability-grouped-outputs", action="store_true", default=None, dest="interpretability_grouped_outputs")
    parser.add_argument("--no-interpretability-grouped-outputs", action="store_false", dest="interpretability_grouped_outputs")
    parser.add_argument("--timegen-min-subjects", type=int, default=None)
    parser.add_argument("--timegen-min-valid-perm-fraction", type=float, default=None)
    parser.add_argument("--class-min-subjects-for-auc", type=int, default=None)
    parser.add_argument("--class-max-failed-fold-fraction", type=float, default=None)
    parser.add_argument("--strict-regression-continuous", action="store_true", default=None, dest="strict_regression_continuous")
    parser.add_argument("--no-strict-regression-continuous", action="store_false", dest="strict_regression_continuous")

    fmri_sig = parser.add_argument_group("fMRI signature target (when --target=fmri_signature)")
    fmri_sig.add_argument(
        "--fmri-signature-method",
        choices=["beta-series", "lss"],
        default=None,
        help="Which fMRI trial-beta estimation method to load signature targets from (default: beta-series).",
    )
    fmri_sig.add_argument(
        "--fmri-signature-contrast-name",
        type=str,
        default=None,
        help="Contrast name folder under fmri/(beta_series|lss)/task-*/contrast-*/ (default: pain_vs_nonpain).",
    )
    fmri_sig.add_argument(
        "--fmri-signature-name",
        choices=["NPS", "SIIPS1"],
        default=None,
        help="Which signature to use as the ML target (default: NPS).",
    )
    fmri_sig.add_argument(
        "--fmri-signature-metric",
        choices=["dot", "cosine", "pearson_r"],
        default=None,
        help="Which signature similarity metric to use (default: dot).",
    )
    fmri_sig.add_argument(
        "--fmri-signature-normalization",
        choices=[
            "none",
            "zscore_within_run",
            "zscore_within_subject",
            "robust_zscore_within_run",
            "robust_zscore_within_subject",
        ],
        default=None,
        help="Optional normalization applied to the signature target (default: none).",
    )
    fmri_sig.add_argument(
        "--fmri-signature-round-decimals",
        type=int,
        default=None,
        help="Rounding precision (decimals) for onset/duration matching across modalities (default: 3).",
    )


def setup_ml(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the ml command parser."""
    parser = subparsers.add_parser(
        "ml",
        help="Machine learning: run LOSO regression, time-generalization, or classification",
        description="Run machine learning pipeline (compute only; use plotting pipeline for visualization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "regression",
            "timegen",
            "classify",
            "model_comparison",
            "incremental_validity",
            "uncertainty",
            "shap",
            "permutation",
        ],
        default="regression",
        help=(
            "ML mode: 'regression' (LOSO regression), 'timegen' (time-generalization), "
            "'classify' (pain classification), 'model_comparison' (compare ElasticNet/Ridge/RF), "
            "'incremental_validity' (Δ performance from EEG over baseline), "
            "'uncertainty' (conformal prediction intervals), 'shap' (SHAP feature importance), "
            "'permutation' (permutation feature importance)"
        ),
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available ML stages and exit.",
    )
    
    _add_ml_specific_arguments(parser)
    _add_model_arguments(parser)
    add_path_args(parser)
    
    return parser


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


def _update_model_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with model-specific hyperparameter overrides."""
    if args.elasticnet_alpha_grid is not None:
        alpha_values = [float(v) for v in args.elasticnet_alpha_grid]
        config["machine_learning.models.elasticnet.alpha_grid"] = alpha_values
    
    if args.elasticnet_l1_ratio_grid is not None:
        l1_ratio_values = [float(v) for v in args.elasticnet_l1_ratio_grid]
        config["machine_learning.models.elasticnet.l1_ratio_grid"] = l1_ratio_values
    
    if args.rf_n_estimators is not None:
        config["machine_learning.models.random_forest.n_estimators"] = int(
            args.rf_n_estimators
        )
    
    if args.rf_max_depth_grid is not None:
        max_depth_values = _parse_max_depth_values(args.rf_max_depth_grid)
        config["machine_learning.models.random_forest.max_depth_grid"] = max_depth_values
    
    if args.ridge_alpha_grid is not None:
        alpha_values = [float(v) for v in args.ridge_alpha_grid]
        config["machine_learning.models.ridge.alpha_grid"] = alpha_values

    if args.variance_threshold_grid is not None:
        config["machine_learning.preprocessing.variance_threshold_grid"] = [
            float(v) for v in args.variance_threshold_grid
        ]
    if getattr(args, "imputer", None) is not None:
        config["machine_learning.preprocessing.imputer_strategy"] = str(args.imputer)
    if getattr(args, "power_transformer_method", None) is not None:
        config["machine_learning.preprocessing.power_transformer_method"] = str(args.power_transformer_method)
    if getattr(args, "power_transformer_standardize", None) is not None:
        config["machine_learning.preprocessing.power_transformer_standardize"] = bool(args.power_transformer_standardize)
    if getattr(args, "pca_enabled", None) is not None:
        config["machine_learning.preprocessing.pca.enabled"] = bool(args.pca_enabled)
    if getattr(args, "pca_n_components", None) is not None:
        config["machine_learning.preprocessing.pca.n_components"] = float(args.pca_n_components)
    if getattr(args, "pca_whiten", None) is not None:
        config["machine_learning.preprocessing.pca.whiten"] = bool(args.pca_whiten)
    if getattr(args, "pca_svd_solver", None) is not None:
        config["machine_learning.preprocessing.pca.svd_solver"] = str(args.pca_svd_solver)
    if getattr(args, "pca_rng_seed", None) is not None:
        config["machine_learning.preprocessing.pca.random_state"] = int(args.pca_rng_seed)
    if getattr(args, "deconfound", None) is not None:
        config["machine_learning.preprocessing.deconfound"] = bool(args.deconfound)
    if getattr(args, "feature_selection_percentile", None) is not None:
        config["machine_learning.preprocessing.feature_selection_percentile"] = float(args.feature_selection_percentile)
    if getattr(args, "ensemble_calibrate", None) is not None:
        config["machine_learning.classification.calibrate_ensemble"] = bool(args.ensemble_calibrate)
    if getattr(args, "spatial_regions_allowed", None) is not None:
        config["machine_learning.preprocessing.spatial_regions_allowed"] = [
            str(v).strip() for v in args.spatial_regions_allowed if str(v).strip()
        ]
    if getattr(args, "classification_resampler", None) is not None:
        config["machine_learning.classification.resampler"] = str(args.classification_resampler)
    if getattr(args, "classification_resampler_seed", None) is not None:
        config["machine_learning.classification.resampler_seed"] = int(args.classification_resampler_seed)
    if getattr(args, "svm_kernel", None) is not None:
        config["machine_learning.models.svm.kernel"] = str(args.svm_kernel)
    if getattr(args, "svm_c_grid", None) is not None:
        config["machine_learning.models.svm.C_grid"] = [float(v) for v in args.svm_c_grid]
    if getattr(args, "svm_gamma_grid", None) is not None:
        gamma_vals = []
        for v in args.svm_gamma_grid:
            try:
                gamma_vals.append(float(v))
            except (TypeError, ValueError):
                gamma_vals.append(str(v))
        config["machine_learning.models.svm.gamma_grid"] = gamma_vals
    if getattr(args, "svm_class_weight", None) is not None:
        config["machine_learning.models.svm.class_weight"] = None if args.svm_class_weight == "none" else args.svm_class_weight
    if getattr(args, "lr_penalty", None) is not None:
        config["machine_learning.models.logistic_regression.penalty"] = str(args.lr_penalty)
    if getattr(args, "lr_c_grid", None) is not None:
        config["machine_learning.models.logistic_regression.C_grid"] = [float(v) for v in args.lr_c_grid]
    if getattr(args, "lr_max_iter", None) is not None:
        config["machine_learning.models.logistic_regression.max_iter"] = int(args.lr_max_iter)
    if getattr(args, "lr_class_weight", None) is not None:
        config["machine_learning.models.logistic_regression.class_weight"] = None if args.lr_class_weight == "none" else args.lr_class_weight
    if getattr(args, "rf_min_samples_split_grid", None) is not None:
        config["machine_learning.models.random_forest.min_samples_split_grid"] = [int(float(v)) for v in args.rf_min_samples_split_grid]
    if getattr(args, "rf_min_samples_leaf_grid", None) is not None:
        config["machine_learning.models.random_forest.min_samples_leaf_grid"] = [int(float(v)) for v in args.rf_min_samples_leaf_grid]
    if getattr(args, "rf_bootstrap", None) is not None:
        config["machine_learning.models.random_forest.bootstrap"] = bool(args.rf_bootstrap)
    if getattr(args, "rf_class_weight", None) is not None:
        config["machine_learning.models.random_forest.class_weight"] = None if args.rf_class_weight == "none" else args.rf_class_weight
    if getattr(args, "cnn_filters1", None) is not None:
        config["machine_learning.models.cnn.temporal_filters"] = int(args.cnn_filters1)
    if getattr(args, "cnn_filters2", None) is not None:
        config["machine_learning.models.cnn.pointwise_filters"] = int(args.cnn_filters2)
    if getattr(args, "cnn_kernel_size1", None) is not None:
        config["machine_learning.models.cnn.kernel_length"] = int(args.cnn_kernel_size1)
    if getattr(args, "cnn_kernel_size2", None) is not None:
        config["machine_learning.models.cnn.separable_kernel_length"] = int(args.cnn_kernel_size2)
    if getattr(args, "cnn_pool_size", None) is not None:
        config["machine_learning.models.cnn.pool_size"] = int(args.cnn_pool_size)
    if getattr(args, "cnn_dense_units", None) is not None:
        config["machine_learning.models.cnn.dense_units"] = int(args.cnn_dense_units)
    if getattr(args, "cnn_dropout_conv", None) is not None:
        config["machine_learning.models.cnn.dropout_conv"] = float(args.cnn_dropout_conv)
    if getattr(args, "cnn_dropout_dense", None) is not None:
        config["machine_learning.models.cnn.dropout"] = float(args.cnn_dropout_dense)
    elif getattr(args, "cnn_dropout_conv", None) is not None:
        config["machine_learning.models.cnn.dropout"] = float(args.cnn_dropout_conv)
    if getattr(args, "cnn_batch_size", None) is not None:
        config["machine_learning.models.cnn.batch_size"] = int(args.cnn_batch_size)
    if getattr(args, "cnn_epochs", None) is not None:
        config["machine_learning.models.cnn.max_epochs"] = int(args.cnn_epochs)
    if getattr(args, "cnn_learning_rate", None) is not None:
        config["machine_learning.models.cnn.learning_rate"] = float(args.cnn_learning_rate)
    if getattr(args, "cnn_patience", None) is not None:
        config["machine_learning.models.cnn.patience"] = int(args.cnn_patience)
    if getattr(args, "cnn_min_delta", None) is not None:
        config["machine_learning.models.cnn.min_delta"] = float(args.cnn_min_delta)
    if getattr(args, "cnn_l2_lambda", None) is not None:
        config["machine_learning.models.cnn.weight_decay"] = float(args.cnn_l2_lambda)
    if getattr(args, "cnn_random_seed", None) is not None:
        config["project.random_state"] = int(args.cnn_random_seed)
    if getattr(args, "cv_hygiene", None) is not None:
        config["machine_learning.cv.hygiene_enabled"] = bool(args.cv_hygiene)
    if getattr(args, "cv_permutation_scheme", None) is not None:
        config["machine_learning.cv.permutation_scheme"] = str(args.cv_permutation_scheme)
    if getattr(args, "cv_min_valid_perm_fraction", None) is not None:
        config["machine_learning.cv.min_valid_permutation_fraction"] = float(args.cv_min_valid_perm_fraction)
    if getattr(args, "cv_default_n_bins", None) is not None:
        config["machine_learning.cv.default_n_bins"] = int(args.cv_default_n_bins)
    if getattr(args, "eval_ci_method", None) is not None:
        config["machine_learning.evaluation.ci_method"] = str(args.eval_ci_method)
    if getattr(args, "eval_bootstrap_iterations", None) is not None:
        config["machine_learning.evaluation.bootstrap_iterations"] = int(args.eval_bootstrap_iterations)
    if getattr(args, "data_covariates_strict", None) is not None:
        config["machine_learning.data.covariates_strict"] = bool(args.data_covariates_strict)
    if getattr(args, "data_max_excluded_subject_fraction", None) is not None:
        config["machine_learning.data.max_excluded_subject_fraction"] = float(args.data_max_excluded_subject_fraction)
    if getattr(args, "incremental_baseline_alpha", None) is not None:
        config["machine_learning.incremental_validity.baseline_alpha"] = float(args.incremental_baseline_alpha)
    if getattr(args, "interpretability_grouped_outputs", None) is not None:
        config["machine_learning.interpretability.grouped_outputs"] = bool(args.interpretability_grouped_outputs)
    if getattr(args, "timegen_min_subjects", None) is not None:
        config["machine_learning.analysis.time_generalization.min_subjects_per_cell"] = int(args.timegen_min_subjects)
    if getattr(args, "timegen_min_valid_perm_fraction", None) is not None:
        config["machine_learning.analysis.time_generalization.min_valid_permutation_fraction"] = float(args.timegen_min_valid_perm_fraction)
    if getattr(args, "class_min_subjects_for_auc", None) is not None:
        config["machine_learning.classification.min_subjects_with_auc_for_inference"] = int(args.class_min_subjects_for_auc)
    if getattr(args, "class_max_failed_fold_fraction", None) is not None:
        config["machine_learning.classification.max_failed_fold_fraction"] = float(args.class_max_failed_fold_fraction)
    if getattr(args, "strict_regression_continuous", None) is not None:
        config["machine_learning.targets.strict_regression_target_continuous"] = bool(args.strict_regression_continuous)


def _update_ml_plot_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with ML plotting overrides."""
    if getattr(args, "ml_plots", None) is not None:
        config["machine_learning.plotting.enabled"] = bool(args.ml_plots)
    if getattr(args, "ml_plot_formats", None) is not None:
        config["machine_learning.plotting.formats"] = [str(v).strip().lower() for v in args.ml_plot_formats]
    if getattr(args, "ml_plot_dpi", None) is not None:
        config["machine_learning.plotting.dpi"] = int(args.ml_plot_dpi)
    if getattr(args, "ml_plot_top_n_features", None) is not None:
        config["machine_learning.plotting.top_n_features"] = int(args.ml_plot_top_n_features)
    if getattr(args, "ml_plot_diagnostics", None) is not None:
        config["machine_learning.plotting.include_diagnostics"] = bool(args.ml_plot_diagnostics)


def _update_fmri_signature_target_config(args: argparse.Namespace, config: Any) -> None:
    """Update config for fMRI signature target loading when requested."""
    if getattr(args, "fmri_signature_method", None):
        config["machine_learning.fmri_signature.method"] = str(args.fmri_signature_method)
    if getattr(args, "fmri_signature_contrast_name", None):
        config["machine_learning.fmri_signature.contrast_name"] = str(args.fmri_signature_contrast_name)
    if getattr(args, "fmri_signature_name", None):
        config["machine_learning.fmri_signature.signature_name"] = str(args.fmri_signature_name)
    if getattr(args, "fmri_signature_metric", None):
        config["machine_learning.fmri_signature.metric"] = str(args.fmri_signature_metric)
    if getattr(args, "fmri_signature_normalization", None):
        config["machine_learning.fmri_signature.normalization"] = str(args.fmri_signature_normalization)
    if getattr(args, "fmri_signature_round_decimals", None) is not None:
        config["machine_learning.fmri_signature.round_decimals"] = int(args.fmri_signature_round_decimals)


def _print_stage_list() -> None:
    """Print available ML stages and exit."""
    print("\nAvailable ML stages:")
    print("=" * 50)
    for stage, desc in ML_STAGES.items():
        print(f"  {stage:22s} - {desc}")
    print("\nUsage: eeg ml <stage> --subjects 0001 0002 ...")


def _print_dry_run_info(args: argparse.Namespace, subjects: List[str]) -> None:
    """Print dry-run information and exit."""
    print(f"\n[DRY RUN] Would execute ML stage: {args.mode}")
    print(f"  Subjects: {subjects}")
    print(f"  CV scope: {args.cv_scope}")
    print(f"  Model: {args.model}")
    print(f"  n_perm: {args.n_perm}")
    print(f"  inner_splits: {args.inner_splits}")
    if args.target:
        print(f"  Target: {args.target}")
    if args.feature_families:
        print(f"  Feature families: {args.feature_families}")
    if args.feature_bands:
        print(f"  Feature bands: {args.feature_bands}")
    if args.feature_segments:
        print(f"  Feature segments: {args.feature_segments}")
    if args.feature_scopes:
        print(f"  Feature scopes: {args.feature_scopes}")
    if args.feature_stats:
        print(f"  Feature stats: {args.feature_stats}")
    if args.covariates:
        print(f"  Covariates: {args.covariates}")
    if args.require_trial_ml_safe:
        print("  Require trial_ml_safe: True")
    if args.ml_plots is not None:
        print(f"  ML plots enabled: {bool(args.ml_plots)}")
    if args.ml_plot_formats:
        print(f"  ML plot formats: {args.ml_plot_formats}")


def _validate_ml_requirements(
    subjects: List[str], cv_scope: str, config: Any
) -> None:
    """Validate subject requirements for ML pipeline."""
    validate_subjects_not_empty(subjects, "ml")
    
    if cv_scope == "group":
        min_subjects = config.get(MIN_SUBJECTS_KEY, MIN_SUBJECTS_FOR_ML)
        validate_min_subjects(subjects, min_subjects, "ML (group scope)")


def _build_pipeline_kwargs(args: argparse.Namespace, config: Any) -> Dict[str, Any]:
    """Build keyword arguments for pipeline.run_batch."""
    rng_seed = (
        args.rng_seed
        if args.rng_seed is not None
        else config.get("project.random_state")
    )
    
    return {
        "cv_scope": args.cv_scope,
        "n_perm": args.n_perm,
        "inner_splits": args.inner_splits,
        "outer_jobs": args.outer_jobs,
        "rng_seed": rng_seed,
        "model": args.model,
        "uncertainty_alpha": args.uncertainty_alpha,
        "perm_n_repeats": args.perm_n_repeats,
        "classification_model": args.classification_model,
        "target": args.target,
        "binary_threshold": args.binary_threshold,
        "feature_families": args.feature_families,
        "feature_bands": args.feature_bands,
        "feature_segments": args.feature_segments,
        "feature_scopes": args.feature_scopes,
        "feature_stats": args.feature_stats,
        "feature_harmonization": args.feature_harmonization,
        "baseline_predictors": args.baseline_predictors,
        "covariates": args.covariates,
    }


def run_ml(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the ml command."""
    if args.list_stages:
        _print_stage_list()
        return
    
    from eeg_pipeline.pipelines.machine_learning import MLPipeline
    
    if args.dry_run:
        _print_dry_run_info(args, subjects)
        return
    
    _validate_ml_requirements(subjects, args.cv_scope, config)
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    _update_path_config(args, config)
    _update_model_config(args, config)
    _update_ml_plot_config(args, config)
    _update_fmri_signature_target_config(args, config)
    # Enforce trial-level ML-safe feature computation by default to prevent CV leakage.
    # --require-trial-ml-safe is retained for backward-compatible CLI ergonomics.
    config["machine_learning.data.require_trial_ml_safe"] = True
    config["feature_engineering.analysis_mode"] = "trial_ml_safe"
    
    pipeline_kwargs = _build_pipeline_kwargs(args, config)
    pipeline = MLPipeline(config=config)
    
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        progress=progress,
        **pipeline_kwargs,
    )
