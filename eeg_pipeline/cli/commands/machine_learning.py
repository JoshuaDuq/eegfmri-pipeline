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


def _add_ml_specific_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ML-specific arguments."""
    parser.add_argument(
        "--uncertainty-alpha",
        type=float,
        default=0.1,
        help="Significance level for prediction intervals (default: 0.1 = 90%% coverage).",
    )
    parser.add_argument(
        "--shap-n-samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP (default: 100).",
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


def _extract_arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    """Extract argument value with default fallback."""
    return getattr(args, name, default)


def _update_path_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with path overrides from arguments."""
    bids_root = _extract_arg(args, "bids_root", None)
    if bids_root is not None:
        config.setdefault("paths", {})["bids_root"] = bids_root
    
    deriv_root = _extract_arg(args, "deriv_root", None)
    if deriv_root is not None:
        config.setdefault("paths", {})["deriv_root"] = deriv_root


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
    
    ridge_alpha_grid = _extract_arg(args, "ridge_alpha_grid", None)
    if ridge_alpha_grid is not None:
        alpha_values = [float(v) for v in ridge_alpha_grid]
        config["machine_learning.models.ridge.alpha_grid"] = alpha_values


def _print_stage_list() -> None:
    """Print available ML stages and exit."""
    print("\nAvailable ML stages:")
    print("=" * 50)
    for stage, desc in ML_STAGES.items():
        print(f"  {stage:22s} - {desc}")
    print("\nUsage: eeg ml <stage> --subjects 0001 0002 ...")


def _print_dry_run_info(args: argparse.Namespace, subjects: List[str]) -> None:
    """Print dry-run information and exit."""
    mode = _extract_arg(args, "mode", "regression")
    cv_scope = _extract_arg(args, "cv_scope", "group")
    model = _extract_arg(args, "model", "elasticnet")
    
    print(f"\n[DRY RUN] Would execute ML stage: {mode}")
    print(f"  Subjects: {subjects}")
    print(f"  CV scope: {cv_scope}")
    print(f"  Model: {model}")
    print(f"  n_perm: {args.n_perm}")
    print(f"  inner_splits: {args.inner_splits}")


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
        "cv_scope": _extract_arg(args, "cv_scope", "group"),
        "n_perm": args.n_perm,
        "inner_splits": args.inner_splits,
        "outer_jobs": args.outer_jobs,
        "rng_seed": rng_seed,
        "model": _extract_arg(args, "model", "elasticnet"),
        "uncertainty_alpha": _extract_arg(args, "uncertainty_alpha", 0.1),
        "perm_n_repeats": _extract_arg(args, "perm_n_repeats", 10),
    }


def run_ml(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the ml command."""
    if _extract_arg(args, "list_stages", False):
        _print_stage_list()
        return
    
    from eeg_pipeline.pipelines.machine_learning import MLPipeline
    
    mode = _extract_arg(args, "mode", "regression")
    cv_scope = _extract_arg(args, "cv_scope", "group")
    
    if _extract_arg(args, "dry_run", False):
        _print_dry_run_info(args, subjects)
        return
    
    _validate_ml_requirements(subjects, cv_scope, config)
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    _update_path_config(args, config)
    _update_model_config(args, config)
    
    pipeline_kwargs = _build_pipeline_kwargs(args, config)
    pipeline = MLPipeline(config=config)
    
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=mode,
        progress=progress,
        **pipeline_kwargs,
    )
