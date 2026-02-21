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
    apply_arg_overrides,
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


_MODEL_CONFIG_OVERRIDES = [
    ("elasticnet_alpha_grid",             "machine_learning.models.elasticnet.alpha_grid",                                    lambda v: [float(x) for x in v]),
    ("elasticnet_l1_ratio_grid",          "machine_learning.models.elasticnet.l1_ratio_grid",                                 lambda v: [float(x) for x in v]),
    ("ridge_alpha_grid",                  "machine_learning.models.ridge.alpha_grid",                                         lambda v: [float(x) for x in v]),
    ("rf_n_estimators",                   "machine_learning.models.random_forest.n_estimators",                               int),
    ("rf_min_samples_split_grid",         "machine_learning.models.random_forest.min_samples_split_grid",                     lambda v: [int(float(x)) for x in v]),
    ("rf_min_samples_leaf_grid",          "machine_learning.models.random_forest.min_samples_leaf_grid",                      lambda v: [int(float(x)) for x in v]),
    ("rf_bootstrap",                      "machine_learning.models.random_forest.bootstrap",                                  bool),
    ("variance_threshold_grid",           "machine_learning.preprocessing.variance_threshold_grid",                           lambda v: [float(x) for x in v]),
    ("imputer",                           "machine_learning.preprocessing.imputer_strategy",                                  str),
    ("power_transformer_method",          "machine_learning.preprocessing.power_transformer_method",                          str),
    ("power_transformer_standardize",     "machine_learning.preprocessing.power_transformer_standardize",                     bool),
    ("pca_enabled",                       "machine_learning.preprocessing.pca.enabled",                                       bool),
    ("pca_n_components",                  "machine_learning.preprocessing.pca.n_components",                                  float),
    ("pca_whiten",                        "machine_learning.preprocessing.pca.whiten",                                        bool),
    ("pca_svd_solver",                    "machine_learning.preprocessing.pca.svd_solver",                                    str),
    ("pca_rng_seed",                      "machine_learning.preprocessing.pca.random_state",                                  int),
    ("deconfound",                        "machine_learning.preprocessing.deconfound",                                        bool),
    ("feature_selection_percentile",      "machine_learning.preprocessing.feature_selection_percentile",                      float),
    ("ensemble_calibrate",                "machine_learning.classification.calibrate_ensemble",                               bool),
    ("classification_resampler",          "machine_learning.classification.resampler",                                        str),
    ("classification_resampler_seed",     "machine_learning.classification.resampler_seed",                                   int),
    ("svm_kernel",                        "machine_learning.models.svm.kernel",                                               str),
    ("svm_c_grid",                        "machine_learning.models.svm.C_grid",                                               lambda v: [float(x) for x in v]),
    ("lr_penalty",                        "machine_learning.models.logistic_regression.penalty",                              str),
    ("lr_c_grid",                         "machine_learning.models.logistic_regression.C_grid",                               lambda v: [float(x) for x in v]),
    ("lr_max_iter",                       "machine_learning.models.logistic_regression.max_iter",                             int),
    ("cnn_filters1",                      "machine_learning.models.cnn.temporal_filters",                                     int),
    ("cnn_filters2",                      "machine_learning.models.cnn.pointwise_filters",                                    int),
    ("cnn_kernel_size1",                  "machine_learning.models.cnn.kernel_length",                                        int),
    ("cnn_kernel_size2",                  "machine_learning.models.cnn.separable_kernel_length",                              int),
    ("cnn_pool_size",                     "machine_learning.models.cnn.pool_size",                                            int),
    ("cnn_dense_units",                   "machine_learning.models.cnn.dense_units",                                          int),
    ("cnn_dropout_conv",                  "machine_learning.models.cnn.dropout_conv",                                         float),
    ("cnn_dropout_dense",                 "machine_learning.models.cnn.dropout",                                              float),
    ("cnn_batch_size",                    "machine_learning.models.cnn.batch_size",                                           int),
    ("cnn_epochs",                        "machine_learning.models.cnn.max_epochs",                                           int),
    ("cnn_learning_rate",                 "machine_learning.models.cnn.learning_rate",                                        float),
    ("cnn_patience",                      "machine_learning.models.cnn.patience",                                             int),
    ("cnn_min_delta",                     "machine_learning.models.cnn.min_delta",                                            float),
    ("cnn_l2_lambda",                     "machine_learning.models.cnn.weight_decay",                                         float),
    ("cnn_random_seed",                   "project.random_state",                                                             int),
    ("cv_hygiene",                        "machine_learning.cv.hygiene_enabled",                                              bool),
    ("cv_permutation_scheme",             "machine_learning.cv.permutation_scheme",                                           str),
    ("cv_min_valid_perm_fraction",        "machine_learning.cv.min_valid_permutation_fraction",                               float),
    ("cv_default_n_bins",                 "machine_learning.cv.default_n_bins",                                               int),
    ("eval_ci_method",                    "machine_learning.evaluation.ci_method",                                            str),
    ("eval_bootstrap_iterations",         "machine_learning.evaluation.bootstrap_iterations",                                 int),
    ("data_covariates_strict",            "machine_learning.data.covariates_strict",                                          bool),
    ("data_max_excluded_subject_fraction","machine_learning.data.max_excluded_subject_fraction",                              float),
    ("incremental_baseline_alpha",        "machine_learning.incremental_validity.baseline_alpha",                             float),
    ("interpretability_grouped_outputs",  "machine_learning.interpretability.grouped_outputs",                                bool),
    ("timegen_min_subjects",              "machine_learning.analysis.time_generalization.min_subjects_per_cell",              int),
    ("timegen_min_valid_perm_fraction",   "machine_learning.analysis.time_generalization.min_valid_permutation_fraction",     float),
    ("class_min_subjects_for_auc",        "machine_learning.classification.min_subjects_with_auc_for_inference",              int),
    ("class_max_failed_fold_fraction",    "machine_learning.classification.max_failed_fold_fraction",                         float),
    ("strict_regression_continuous",      "machine_learning.targets.strict_regression_target_continuous",                     bool),
]


def _update_model_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with model-specific hyperparameter overrides."""
    if args.rf_max_depth_grid is not None:
        config["machine_learning.models.random_forest.max_depth_grid"] = _parse_max_depth_values(args.rf_max_depth_grid)

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
    ("ml_plots",              "machine_learning.plotting.enabled",           bool),
    ("ml_plot_formats",       "machine_learning.plotting.formats",           lambda v: [str(x).strip().lower() for x in v]),
    ("ml_plot_dpi",           "machine_learning.plotting.dpi",               int),
    ("ml_plot_top_n_features","machine_learning.plotting.top_n_features",    int),
    ("ml_plot_diagnostics",   "machine_learning.plotting.include_diagnostics",bool),
]

_FMRI_SIGNATURE_CONFIG_OVERRIDES = [
    ("fmri_signature_method",          "machine_learning.fmri_signature.method",          str),
    ("fmri_signature_contrast_name",   "machine_learning.fmri_signature.contrast_name",   str),
    ("fmri_signature_name",            "machine_learning.fmri_signature.signature_name",  str),
    ("fmri_signature_metric",          "machine_learning.fmri_signature.metric",          str),
    ("fmri_signature_normalization",   "machine_learning.fmri_signature.normalization",   str),
    ("fmri_signature_round_decimals",  "machine_learning.fmri_signature.round_decimals",  int),
]


def _update_ml_plot_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with ML plotting overrides."""
    apply_arg_overrides(args, config, _ML_PLOT_CONFIG_OVERRIDES)


def _update_fmri_signature_target_config(args: argparse.Namespace, config: Any) -> None:
    """Update config for fMRI signature target loading when requested."""
    apply_arg_overrides(args, config, _FMRI_SIGNATURE_CONFIG_OVERRIDES)


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
