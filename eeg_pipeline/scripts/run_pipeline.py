"""Unified EEG Pipeline CLI.

Usage:
    python run_pipeline.py <command> <mode> [options]

Commands:
    behavior    Brain-behavior correlation analysis
    features    Feature extraction from epochs
    erp         Event-related potential analysis
    tfr         Time-frequency visualization
    decoding    ML-based prediction

Examples:
    python run_pipeline.py features compute --subject 0001
    python run_pipeline.py behavior compute --all-subjects
    python run_pipeline.py decoding --subject 0001 --subject 0002
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Any
import argparse

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import parse_subject_args


###################################################################
# Constants
###################################################################

DEFAULT_TASK_KEY = "project.task"
DEFAULT_TASK_VALUE = "thermalactive"
MIN_SUBJECTS_KEY = "analysis.min_subjects_for_group"
DEFAULT_MIN_SUBJECTS = 2
MIN_SUBJECTS_FOR_DECODING = 2
DEFAULT_RNG_SEED = 42
POOLING_STRATEGY_KEY = "behavior_analysis.pooling_strategy"
DEFAULT_POOLING_STRATEGY = "within_subject_centered"
FEATURE_CATEGORIES = ["power", "connectivity", "microstates", "aperiodic", "itpc", "pac", "precomputed"]
BEHAVIOR_COMPUTATIONS = [
    "power_roi", "connectivity_roi", "connectivity_heatmaps", "sliding_connectivity",
    "time_frequency", "temporal_correlations", "cluster_test", 
    "precomputed_correlations", "condition_correlations", "exports"
]


###################################################################
# Shared Utilities
###################################################################

def resolve_task(task: Optional[str], config: Any) -> str:
    return task or config.get(DEFAULT_TASK_KEY, DEFAULT_TASK_VALUE)


def validate_subjects_not_empty(subjects: List[str], operation: str) -> None:
    if not subjects:
        raise ValueError(f"No subjects specified for {operation}")


def validate_min_subjects(
    subjects: List[str],
    min_count: int,
    operation: str
) -> None:
    if len(subjects) < min_count:
        raise ValueError(
            f"{operation} requires at least {min_count} subjects, "
            f"got {len(subjects)}"
        )


def get_deriv_root(config: Any) -> Path:
    return Path(config.deriv_root)


###################################################################
# Shared Argument Parsing Utilities
###################################################################


def add_common_subject_args(parser: argparse.ArgumentParser) -> None:
    """Add standard subject selection arguments."""
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        "--group", type=str,
        help="Group of subjects: 'all' or comma-separated list"
    )
    subject_group.add_argument(
        "--subject", "-s", type=str, action="append",
        help="Subject label(s) without 'sub-' prefix"
    )
    subject_group.add_argument(
        "--all-subjects", action="store_true",
        help="Process all available subjects"
    )


def add_task_arg(parser: argparse.ArgumentParser) -> None:
    """Add standard task argument."""
    parser.add_argument(
        "--task", "-t", type=str, default=None,
        help="Task label (default from config)"
    )


###################################################################
# Behavior Subcommand
###################################################################

def setup_behavior_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "behavior",
        help="Behavior analysis: compute correlations, visualize, or aggregate",
        description="Behavior pipeline: compute correlations, visualize, or aggregate across subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["compute", "visualize", "aggregate"],
        help="Pipeline mode"
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    
    compute_group = parser.add_argument_group("Compute mode options")
    compute_group.add_argument(
        "--correlation-method", choices=["spearman", "pearson"], default=None,
        help="Correlation method (default from config)"
    )
    compute_group.add_argument(
        "--bootstrap", type=int, default=0,
        help="Number of bootstrap iterations (default: 0)"
    )
    compute_group.add_argument(
        "--n-perm", type=int, default=None,
        help="Number of permutations (default from config)"
    )
    compute_group.add_argument(
        "--rng-seed", type=int, default=DEFAULT_RNG_SEED,
        help=f"Random seed (default: {DEFAULT_RNG_SEED})"
    )
    compute_group.add_argument(
        "--computations", nargs="+",
        choices=BEHAVIOR_COMPUTATIONS,
        default=None,
        help=f"Specific behavior computations to run (default: all). Choices: {', '.join(BEHAVIOR_COMPUTATIONS)}"
    )
    
    visualize_group = parser.add_argument_group("Visualize mode options")
    plot_group = visualize_group.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plots", nargs="+",
        metavar="PLOT",
        help="Specific plot types to generate"
    )
    plot_group.add_argument(
        "--all-plots", action="store_true",
        help="Generate all available plots (default behavior)"
    )
    visualize_group.add_argument(
        "--skip-scatter", action="store_true",
        help="Skip ROI scatter plot generation"
    )
    visualize_group.add_argument(
        "--do-group", action="store_true",
        help="Generate group-level plots"
    )


def run_behavior(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.behavior import compute_behavior_correlations_for_subjects
    from eeg_pipeline.plotting.behavioral.viz import visualize_behavior_for_subjects
    from eeg_pipeline.analysis.group import aggregate_behavior_correlations
    from eeg_pipeline.utils.io.general import get_logger
    
    if args.mode == "compute":
        compute_behavior_correlations_for_subjects(
            subjects=subjects,
            task=args.task,
            correlation_method=args.correlation_method,
            bootstrap=args.bootstrap,
            n_perm=args.n_perm,
            rng_seed=args.rng_seed,
            computations=args.computations,
        )
    elif args.mode == "visualize":
        task = resolve_task(args.task, config)
        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            scatter_only=False,
            temporal_only=args.skip_scatter,
            group=args.do_group,
        )
    elif args.mode == "aggregate":
        deriv_root = get_deriv_root(config)
        validate_subjects_not_empty(subjects, "behavior aggregation")
        
        min_subjects = config.get(MIN_SUBJECTS_KEY, DEFAULT_MIN_SUBJECTS)
        validate_min_subjects(subjects, min_subjects, "Behavior aggregation")
        
        logger = get_logger("behavior_aggregation")
        logger.info(f"Starting behavior aggregation: {len(subjects)} subjects")
        
        aggregate_behavior_correlations(
            subjects=subjects,
            task=args.task,
            deriv_root=deriv_root,
            config=config,
            pooling_strategy=config.get(POOLING_STRATEGY_KEY, DEFAULT_POOLING_STRATEGY)
        )
        
        logger.info("Behavior aggregation complete")
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'compute', 'visualize', or 'aggregate'.")


###################################################################
# Features Subcommand
###################################################################

def setup_features_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "features",
        help="Features analysis: extract features, visualize, or aggregate",
        description="Features pipeline: extract features, visualize, or aggregate across subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["compute", "visualize", "aggregate"],
        help="Pipeline mode"
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument(
        "--fixed-templates", type=str,
        help="Path to .npz file containing fixed microstate templates"
    )
    parser.add_argument(
        "--feature-categories", nargs="+",
        choices=FEATURE_CATEGORIES,
        default=None,
        help=f"Specific feature categories to compute (default: all). Choices: {', '.join(FEATURE_CATEGORIES)}"
    )


def run_features(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.features import extract_features_for_subjects
    from eeg_pipeline.plotting.features import visualize_features_for_subjects
    from eeg_pipeline.analysis.group import aggregate_feature_stats
    from eeg_pipeline.utils.io.general import get_logger
    
    if args.mode == "compute":
        extract_features_for_subjects(
            subjects=subjects,
            task=args.task,
            fixed_templates_path=Path(args.fixed_templates) if args.fixed_templates else None,
            feature_categories=args.feature_categories,
        )
    elif args.mode == "visualize":
        visualize_features_for_subjects(subjects=subjects, task=args.task)
    elif args.mode == "aggregate":
        deriv_root = get_deriv_root(config)
        validate_subjects_not_empty(subjects, "feature aggregation")
        
        min_subjects = config.get(MIN_SUBJECTS_KEY, DEFAULT_MIN_SUBJECTS)
        validate_min_subjects(subjects, min_subjects, "Feature aggregation")
        
        logger = get_logger("feature_aggregation")
        logger.info(f"Starting feature aggregation: {len(subjects)} subjects")
        
        aggregate_feature_stats(subjects, args.task, deriv_root, config)
        
        logger.info("Feature aggregation complete")
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'compute', 'visualize', or 'aggregate'.")


###################################################################
# ERP Subcommand
###################################################################

def setup_erp_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "erp",
        help="ERP analysis: compute statistics or visualize",
        description="ERP pipeline: compute statistics or visualize ERPs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["compute", "visualize"],
        help="Pipeline mode"
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument(
        "--crop-tmin", type=float, default=None,
        help="ERP epoch crop start time (s)"
    )
    parser.add_argument(
        "--crop-tmax", type=float, default=None,
        help="ERP epoch crop end time (s)"
    )


def run_erp(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.erp import extract_erp_stats_for_subjects
    from eeg_pipeline.plotting.erp import visualize_erp_for_subjects
    
    if args.mode == "compute":
        extract_erp_stats_for_subjects(
            subjects=subjects,
            task=args.task,
            crop_tmin=args.crop_tmin,
            crop_tmax=args.crop_tmax,
        )
    elif args.mode == "visualize":
        visualize_erp_for_subjects(
            subjects=subjects,
            task=args.task,
            crop_tmin=args.crop_tmin,
            crop_tmax=args.crop_tmax,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'compute' or 'visualize'.")


###################################################################
# TFR Subcommand
###################################################################

def setup_tfr_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "tfr",
        help="TFR visualization: generate time-frequency representations",
        description="TFR pipeline: visualize time-frequency representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["visualize"],
        help="Pipeline mode (only visualize available)"
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument(
        "--do-group", action="store_true",
        help="Create group visualizations"
    )
    parser.add_argument(
        "--tfr-roi", action="store_true",
        help="Generate only ROI-level TFR visualizations"
    )
    parser.add_argument(
        "--tfr-topomaps-only", action="store_true",
        help="Generate only topomap visualizations"
    )


def run_tfr(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.plotting.tfr import visualize_tfr_for_subjects
    
    if args.mode == "visualize":
        visualize_tfr_for_subjects(
            subjects=subjects,
            task=args.task,
            group=args.do_group,
            tfr_roi_only=args.tfr_roi,
            tfr_topomaps_only=args.tfr_topomaps_only,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'visualize'.")


###################################################################
# Decoding Subcommand
###################################################################

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
        "--rng-seed", type=int, default=DEFAULT_RNG_SEED,
        help=f"Random seed (default: {DEFAULT_RNG_SEED})"
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
    validate_min_subjects(subjects, MIN_SUBJECTS_FOR_DECODING, "Decoding")
    
    task = resolve_task(args.task, config)
    logger = get_logger(__name__)
    deriv_root = get_deriv_root(config)
    results_root = deriv_root / "decoding"
    
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
        rng_seed=args.rng_seed,
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
            rng_seed=args.rng_seed,
            results_root=results_root,
            logger=logger,
        )
    
    logger.info("Decoding complete.")


###################################################################
# Main Entry Point
###################################################################

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(
        description="Unified EEG Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Behavior: compute correlations
  python run_pipeline.py behavior compute --subject 0001

  # Features: extract and visualize
  python run_pipeline.py features compute --subject 0001
  python run_pipeline.py features visualize --subject 0001

  # ERP: compute statistics
  python run_pipeline.py erp compute --subject 0001

  # TFR: visualize
  python run_pipeline.py tfr visualize --subject 0001

  # Decoding: run analysis
  python run_pipeline.py decoding --subject 0001 --subject 0002

For detailed help on each subcommand:
  python run_pipeline.py <subcommand> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Analysis type")
    
    setup_behavior_parser(subparsers)
    setup_features_parser(subparsers)
    setup_erp_parser(subparsers)
    setup_tfr_parser(subparsers)
    setup_decoding_parser(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    config = load_settings()
    deriv_root = get_deriv_root(config)
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        logging.error("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        return 2
    
    command_handlers = {
        "behavior": run_behavior,
        "features": run_features,
        "erp": run_erp,
        "tfr": run_tfr,
        "decoding": run_decoding,
    }
    
    handler = command_handlers.get(args.command)
    if not handler:
        logging.error("Unknown command: %s", args.command)
        return 1
    
    try:
        handler(args, subjects, config)
        return 0
    except Exception as e:
        logging.error("Error running %s: %s", args.command, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

