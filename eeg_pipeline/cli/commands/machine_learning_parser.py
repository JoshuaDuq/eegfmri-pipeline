"""Parser construction for machine learning CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
)

ML_STAGES = {
    "regression": "LOSO regression predicting a continuous target",
    "timegen": "Time-generalization analysis",
    "classify": "Binary classification",
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
            "'predictor', 'trial_index', 'block')."
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
            "Target to predict. Can be a logical name ('outcome', 'predictor', 'binary_outcome') "
            "or an explicit events.tsv column name. Use --target=fmri_signature to predict trial-wise "
            "signature expression from fMRI beta-series/LSS. Defaults depend on stage."
        ),
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=None,
        help=(
            "Fixed threshold for binarizing a continuous target when running classification "
            "(e.g., --target=outcome --binary-threshold=30). Median-split is intentionally disabled."
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
            "Baseline predictor columns for incremental_validity (standardized meta names like 'predictor', "
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
    parser.add_argument("--classification-resampler", choices=["none", "undersample", "smote"], default=None)
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
        help="Contrast name folder under fmri/(beta_series|lss)/task-*/contrast-*/ (default: contrast).",
    )
    fmri_sig.add_argument(
        "--fmri-signature-name",
        default=None,
        help="Which signature to use as the ML target (default: config value, or auto-select first available signature).",
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
            "'classify' (binary classification), 'model_comparison' (compare ElasticNet/Ridge/RF), "
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
