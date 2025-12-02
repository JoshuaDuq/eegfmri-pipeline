"""
Decoding CLI Subcommand
=======================

Parser setup and run function for ML-based decoding analysis.
"""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    resolve_task,
    validate_subjects_not_empty,
    validate_min_subjects,
    get_deriv_root,
    MIN_SUBJECTS_KEY,
    MIN_SUBJECTS_FOR_DECODING,
)


def setup_decoding_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "decoding",
        help="Decoding analysis: run LOSO regression and time-generalization",
        description="Run EEG decoding (LOSO regression + time-generalization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument(
        "--n-perm", type=int, default=0,
        help="Number of label-shuffle permutations for nulls"
    )
    parser.add_argument(
        "--inner-splits", type=int, default=3,
        help="Inner CV splits for hyperparameter search"
    )
    parser.add_argument(
        "--outer-jobs", type=int, default=1,
        help="Parallel LOSO folds"
    )
    parser.add_argument(
        "--rng-seed", type=int, default=None,
        help="Random seed (default: project.random_state in config)"
    )
    parser.add_argument(
        "--skip-time-gen", action="store_true",
        help="Skip time-generalization decoding"
    )


def run_decoding(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.decoding import (
        run_regression_decoding,
        run_time_generalization,
    )
    from eeg_pipeline.utils.io.general import get_logger
    
    validate_subjects_not_empty(subjects, "decoding")
    min_subjects = config.get(MIN_SUBJECTS_KEY, MIN_SUBJECTS_FOR_DECODING)
    validate_min_subjects(subjects, min_subjects, "Decoding")
    
    task = resolve_task(args.task, config)
    logger = get_logger(__name__)
    deriv_root = get_deriv_root(config)
    results_root = deriv_root / "decoding"
    
    rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
    
    logger.info(
        f"Starting decoding: {len(subjects)} subjects, task={task}, n_perm={args.n_perm}, "
        f"inner_splits={args.inner_splits}, outer_jobs={args.outer_jobs}"
    )
    
    run_regression_decoding(
        subjects=subjects,
        task=task,
        deriv_root=deriv_root,
        config=config,
        n_perm=args.n_perm,
        inner_splits=args.inner_splits,
        outer_jobs=args.outer_jobs,
        rng_seed=rng_seed,
        results_root=results_root,
        logger=logger,
    )
    
    if not args.skip_time_gen:
        run_time_generalization(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            config=config,
            n_perm=args.n_perm,
            rng_seed=rng_seed,
            results_root=results_root,
            logger=logger,
        )
    
    logger.info("Decoding complete.")
