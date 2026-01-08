"""Machine Learning CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

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
            "regression", "timegen", "classify",
            "model_comparison", "incremental_validity",
            "uncertainty", "shap", "permutation",
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
        "--ridge-alpha-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override Ridge alpha grid (e.g., 0.01 0.1 1 10 100, only used when --model=ridge).",
    )
    add_path_args(parser)
    return parser


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


def run_ml(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the ml command."""
    if getattr(args, "list_stages", False):
        print("\nAvailable ML stages:")
        print("=" * 50)
        for stage, desc in ML_STAGES.items():
            print(f"  {stage:22s} - {desc}")
        print("\nUsage: eeg ml <stage> --subjects 0001 0002 ...")
        return
    
    from eeg_pipeline.pipelines.machine_learning import MLPipeline
    
    cv_scope = getattr(args, "cv_scope", "group")
    mode = getattr(args, "mode", "regression")
    
    if getattr(args, "dry_run", False):
        print(f"\n[DRY RUN] Would execute ML stage: {mode}")
        print(f"  Subjects: {subjects}")
        print(f"  CV scope: {cv_scope}")
        print(f"  Model: {getattr(args, 'model', 'elasticnet')}")
        print(f"  n_perm: {args.n_perm}")
        print(f"  inner_splits: {args.inner_splits}")
        return

    validate_subjects_not_empty(subjects, "ml")
    if cv_scope == "group":
        min_subjects = config.get(MIN_SUBJECTS_KEY, MIN_SUBJECTS_FOR_ML)
        validate_min_subjects(subjects, min_subjects, "ML (group scope)")
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    if args.elasticnet_alpha_grid is not None:
        config["machine_learning.models.elasticnet.alpha_grid"] = [float(v) for v in args.elasticnet_alpha_grid]
    if args.elasticnet_l1_ratio_grid is not None:
        config["machine_learning.models.elasticnet.l1_ratio_grid"] = [float(v) for v in args.elasticnet_l1_ratio_grid]
    if args.rf_n_estimators is not None:
        config["machine_learning.models.random_forest.n_estimators"] = int(args.rf_n_estimators)
    if args.rf_max_depth_grid is not None:
        parsed = []
        for raw in args.rf_max_depth_grid:
            if raw.lower() in {"none", "null"}:
                parsed.append(None)
            else:
                parsed.append(int(raw))
        config["machine_learning.models.random_forest.max_depth_grid"] = parsed
    if getattr(args, "ridge_alpha_grid", None) is not None:
        config["machine_learning.models.ridge.alpha_grid"] = [float(v) for v in args.ridge_alpha_grid]

    rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
    
    mode = getattr(args, "mode", "regression")
    model = getattr(args, "model", "elasticnet")
    
    pipeline = MLPipeline(config=config)
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=mode,
        cv_scope=cv_scope,
        n_perm=args.n_perm,
        inner_splits=args.inner_splits,
        outer_jobs=args.outer_jobs,
        rng_seed=rng_seed,
        progress=progress,
        model=model,
        uncertainty_alpha=getattr(args, "uncertainty_alpha", 0.1),
        perm_n_repeats=getattr(args, "perm_n_repeats", 10),
    )
