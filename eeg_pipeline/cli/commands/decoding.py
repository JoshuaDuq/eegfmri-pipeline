"""Decoding/ML CLI command."""

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
    MIN_SUBJECTS_FOR_DECODING,
)


def setup_decoding(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the decoding command parser."""
    parser = subparsers.add_parser(
        "decoding",
        help="Decoding analysis: run LOSO regression and time-generalization",
        description="Run EEG decoding (LOSO regression + time-generalization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
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
    parser.add_argument("--skip-time-gen", action="store_true")
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
        help="Override RandomForest n_estimators.",
    )
    parser.add_argument(
        "--rf-max-depth-grid",
        nargs="+",
        type=str,
        default=None,
        help="Override RandomForest max_depth grid (use 'null' for None).",
    )
    add_path_args(parser)
    return parser


def run_decoding(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the decoding command."""
    from eeg_pipeline.pipelines.decoding import DecodingPipeline
    
    cv_scope = getattr(args, "cv_scope", "group")

    validate_subjects_not_empty(subjects, "decoding")
    if cv_scope == "group":
        min_subjects = config.get(MIN_SUBJECTS_KEY, MIN_SUBJECTS_FOR_DECODING)
        validate_min_subjects(subjects, min_subjects, "Decoding (group scope)")
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    if args.elasticnet_alpha_grid is not None:
        config["decoding.models.elasticnet.alpha_grid"] = [float(v) for v in args.elasticnet_alpha_grid]
    if args.elasticnet_l1_ratio_grid is not None:
        config["decoding.models.elasticnet.l1_ratio_grid"] = [float(v) for v in args.elasticnet_l1_ratio_grid]
    if args.rf_n_estimators is not None:
        config["decoding.models.random_forest.n_estimators"] = int(args.rf_n_estimators)
    if args.rf_max_depth_grid is not None:
        parsed = []
        for raw in args.rf_max_depth_grid:
            if raw.lower() in {"none", "null"}:
                parsed.append(None)
            else:
                parsed.append(int(raw))
        config["decoding.models.random_forest.max_depth_grid"] = parsed

    rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
    
    pipeline = DecodingPipeline(config=config)
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        cv_scope=cv_scope,
        n_perm=args.n_perm,
        inner_splits=args.inner_splits,
        outer_jobs=args.outer_jobs,
        rng_seed=rng_seed,
        skip_time_gen=args.skip_time_gen,
        progress=progress,
    )
