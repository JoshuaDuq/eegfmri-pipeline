import sys
from pathlib import Path
from typing import List, Optional
import argparse

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import parse_subject_args


###################################################################
# Shared Argument Parsing Utilities
###################################################################

def add_common_subject_args(parser: argparse.ArgumentParser) -> None:
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
        "--rng-seed", type=int, default=42,
        help="Random seed (default: 42)"
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


def run_behavior(args, subjects: List[str]) -> None:
    from eeg_pipeline.utils.pipelines.behavior import compute_behavior_correlations_for_subjects
    from eeg_pipeline.plotting.behavioral.viz import visualize_behavior_for_subjects
    from eeg_pipeline.analysis.group import aggregate_behavior_correlations
    from eeg_pipeline.utils.io.general import get_logger
    
    if args.mode == "compute":
        compute_behavior_correlations_for_subjects(
            subjects=subjects,
            task=args.task,
            correlation_method=args.correlation_method,
            partial_covars=None,
            bootstrap=args.bootstrap,
            n_perm=args.n_perm,
            rng_seed=args.rng_seed,
        )
    elif args.mode == "visualize":
        config = load_settings()
        task = args.task or config.get("project.task", "thermalactive")
        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            scatter_only=False,
            temporal_only=args.skip_scatter,
            group=args.do_group,
        )
    elif args.mode == "aggregate":
        config = load_settings()
        deriv_root = Path(config.deriv_root)
        
        if not subjects:
            raise ValueError("No subjects specified")
        
        min_subjects = config.get("analysis.min_subjects_for_group", 2)
        if len(subjects) < min_subjects:
            raise ValueError(f"Behavior aggregation requires at least {min_subjects} subjects, got {len(subjects)}")
        
        logger = get_logger("behavior_aggregation")
        logger.info(f"Starting behavior aggregation: {len(subjects)} subjects")
        
        aggregate_behavior_correlations(
            subjects=subjects,
            task=args.task,
            deriv_root=deriv_root,
            config=config,
            pooling_strategy=config.get("behavior_analysis.pooling_strategy", "within_subject_centered")
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


def run_features(args, subjects: List[str]) -> None:
    from eeg_pipeline.utils.pipelines.features import extract_features_for_subjects
    from eeg_pipeline.plotting.features import visualize_features_for_subjects
    from eeg_pipeline.analysis.group import aggregate_feature_stats
    from eeg_pipeline.utils.io.general import get_logger
    
    if args.mode == "compute":
        extract_features_for_subjects(
            subjects=subjects,
            task=args.task,
            fixed_templates_path=Path(args.fixed_templates) if args.fixed_templates else None,
        )
    elif args.mode == "visualize":
        visualize_features_for_subjects(subjects=subjects, task=args.task)
    elif args.mode == "aggregate":
        config = load_settings()
        deriv_root = Path(config.deriv_root)
        
        if not subjects:
            raise ValueError("No subjects specified")
        
        min_subjects = config.get("analysis.min_subjects_for_group", 2)
        if len(subjects) < min_subjects:
            raise ValueError(f"Feature aggregation requires at least {min_subjects} subjects, got {len(subjects)}")
        
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


def run_erp(args, subjects: List[str]) -> None:
    from eeg_pipeline.utils.pipelines.erp import extract_erp_stats_for_subjects
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


def run_tfr(args, subjects: List[str]) -> None:
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
        "--rng-seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip-time-gen", action="store_true",
        help="Skip time-generalization decoding"
    )


def run_decoding(args, subjects: List[str]) -> None:
    from eeg_pipeline.utils.pipelines.decoding import (
        run_regression_decoding,
        run_time_generalization,
    )
    from eeg_pipeline.utils.io.general import get_logger
    
    config = load_settings()
    task = args.task or config.get("project.task", "thermalactive")
    
    if not subjects:
        raise ValueError("No subjects specified.")
    if len(subjects) < 2:
        raise ValueError("Decoding requires at least 2 subjects for LOSO.")
    
    logger = get_logger(__name__)
    deriv_root = Path(config.deriv_root)
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
    deriv_root = Path(config.deriv_root)
    
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        return 2
    
    for subcommand_name, subcommand_func in [
        ("behavior", run_behavior),
        ("features", run_features),
        ("erp", run_erp),
        ("tfr", run_tfr),
        ("decoding", run_decoding),
    ]:
        if args.command == subcommand_name:
            try:
                subcommand_func(args, subjects)
                return 0
            except Exception as e:
                print(f"Error running {subcommand_name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1
    
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

