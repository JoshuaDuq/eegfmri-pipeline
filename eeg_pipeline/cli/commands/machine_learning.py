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
        choices=["svm", "lr", "rf"],
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
    _update_fmri_signature_target_config(args, config)
    if args.require_trial_ml_safe:
        config["machine_learning.data.require_trial_ml_safe"] = True
    
    pipeline_kwargs = _build_pipeline_kwargs(args, config)
    pipeline = MLPipeline(config=config)
    
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        progress=progress,
        **pipeline_kwargs,
    )
