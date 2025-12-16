"""
CLI Command Registry
====================

Centralized registry of all CLI subcommands.
Each command is defined with its name, parser setup function, and run function.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

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
from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES, PRECOMPUTED_GROUP_CHOICES


@dataclass
class Command:
    """CLI command definition."""
    name: str
    help: str
    description: str
    setup: Callable[[argparse._SubParsersAction], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace, List[str], Any], None]
    requires_subjects: bool = True


BEHAVIOR_COMPUTATIONS = [
    # Canonical stage flags
    "correlations",
    "pain_sensitivity",
    "condition",
    "temporal",
    "cluster",
    "mediation",
    "mixed_effects",
    "export",
    # Legacy aliases (kept for backward compatibility)
    "power_roi",
    "connectivity_roi",
    "connectivity_heatmaps",
    "sliding_connectivity",
    "time_frequency",
    "temporal_correlations",
    "cluster_test",
    "precomputed_correlations",
    "condition_correlations",
    "exports",
]


FEATURE_VISUALIZE_CATEGORIES = [
    "power",
    "connectivity",
    "microstates",
    "aperiodic",
    "itpc",
    "pac",
    "dynamics",
    "burst",
    "erds",
    "complexity",
]


BEHAVIOR_VISUALIZE_CATEGORIES = [
    "psychometrics",
    "power",
    "dynamics",
    "aperiodic",
    "connectivity",
    "itpc",
    "temporal",
    "dose_response",
]


def _setup_behavior(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "behavior",
        help="Behavior analysis: compute correlations or visualize",
        description="Behavior pipeline: compute correlations or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=BEHAVIOR_VISUALIZE_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="Feature categories to process (e.g., power, connectivity, itpc)",
    )
    
    compute_group = parser.add_argument_group("Compute mode options")
    compute_group.add_argument("--correlation-method", choices=["spearman", "pearson"], default=None)
    compute_group.add_argument("--bootstrap", type=int, default=0)
    compute_group.add_argument("--n-perm", type=int, default=None)
    compute_group.add_argument("--rng-seed", type=int, default=None)
    compute_group.add_argument("--computations", nargs="+", choices=BEHAVIOR_COMPUTATIONS, default=None)
    
    visualize_group = parser.add_argument_group("Visualize mode options")
    plot_group = visualize_group.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", nargs="+", metavar="PLOT")
    plot_group.add_argument("--all-plots", action="store_true")
    visualize_group.add_argument("--skip-scatter", action="store_true")
    return parser


def _run_behavior(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.behavior import BehaviorPipeline
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    
    categories = getattr(args, "categories", None)
    
    if args.mode == "compute":
        rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
        if args.correlation_method:
            config.setdefault("behavior_analysis", {}).setdefault("statistics", {})["correlation_method"] = args.correlation_method
        if args.bootstrap and args.bootstrap > 0:
            config.setdefault("behavior_analysis", {})["bootstrap"] = args.bootstrap
        if args.n_perm and args.n_perm > 0:
            config.setdefault("behavior_analysis", {}).setdefault("statistics", {})["n_permutations"] = args.n_perm
        if rng_seed is not None:
            config.setdefault("project", {})["random_state"] = rng_seed
        
        pipeline = BehaviorPipeline(
            config=config,
            computations=args.computations,
            feature_categories=categories,
        )
        pipeline.run_batch(subjects, task=args.task)
    elif args.mode == "visualize":
        task = resolve_task(args.task, config)
        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            scatter_only=False,
            temporal_only=False,
            visualize_categories=categories,
        )


def _setup_features(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "features",
        help="Features analysis: extract features or visualize",
        description="Features pipeline: extract features or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument("--fixed-templates", type=str, help="Path to .npz file containing fixed microstate templates")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=FEATURE_VISUALIZE_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="Feature categories to process (e.g., power, connectivity, itpc)",
    )
    parser.add_argument(
        "--precomputed-groups",
        nargs="+",
        choices=PRECOMPUTED_GROUP_CHOICES,
        default=None,
        help="Override config precomputed groups for precomputed/cfc/dynamics features",
    )
    return parser


def _run_features(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.features import FeaturePipeline
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    
    categories = getattr(args, "categories", None)
    
    if args.mode == "compute":
        pipeline = FeaturePipeline(config=config)
        pipeline.run_batch(
            subjects=subjects,
            task=args.task,
            fixed_templates_path=Path(args.fixed_templates) if args.fixed_templates else None,
            feature_categories=categories,
            precomputed_groups=args.precomputed_groups,
        )
    elif args.mode == "visualize":
        visualize_features_for_subjects(
            subjects=subjects,
            task=args.task,
            config=config,
            visualize_categories=categories,
        )


def _setup_erp(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "erp",
        help="ERP analysis: compute statistics or visualize",
        description="ERP pipeline: compute statistics or visualize ERPs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument("--crop-tmin", type=float, default=None)
    parser.add_argument("--crop-tmax", type=float, default=None)
    return parser


def _run_erp(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.erp import ErpPipeline
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects
    
    if args.mode == "compute":
        pipeline = ErpPipeline(config=config)
        pipeline.set_crop_params(args.crop_tmin, args.crop_tmax)
        pipeline.run_batch(subjects, task=args.task)
    elif args.mode == "visualize":
        visualize_erp_for_subjects(subjects=subjects, task=args.task, crop_tmin=args.crop_tmin, crop_tmax=args.crop_tmax)


def _setup_tfr(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "tfr",
        help="TFR visualization: generate time-frequency representations",
        description="TFR pipeline: visualize time-frequency representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["visualize"], help="Pipeline mode (only visualize available)")
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument("--do-group", action="store_true")
    parser.add_argument("--tfr-roi", action="store_true")
    parser.add_argument("--tfr-topomaps-only", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser


def _run_tfr(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    
    if args.mode == "visualize":
        visualize_tfr_for_subjects(
            subjects=subjects,
            task=args.task,
            tfr_roi_only=args.tfr_roi,
            tfr_topomaps_only=args.tfr_topomaps_only,
            n_jobs=args.n_jobs,
        )


def _setup_decoding(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "decoding",
        help="Decoding analysis: run LOSO regression and time-generalization",
        description="Run EEG decoding (LOSO regression + time-generalization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument("--n-perm", type=int, default=0)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--outer-jobs", type=int, default=1)
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--skip-time-gen", action="store_true")
    return parser


def _run_decoding(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.decoding import DecodingPipeline
    
    validate_subjects_not_empty(subjects, "decoding")
    min_subjects = config.get(MIN_SUBJECTS_KEY, MIN_SUBJECTS_FOR_DECODING)
    validate_min_subjects(subjects, min_subjects, "Decoding")
    
    task = resolve_task(args.task, config)
    rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
    
    pipeline = DecodingPipeline(config=config)
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        n_perm=args.n_perm,
        inner_splits=args.inner_splits,
        outer_jobs=args.outer_jobs,
        rng_seed=rng_seed,
        skip_time_gen=args.skip_time_gen,
    )


def _setup_preprocessing(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "preprocessing",
        help="Preprocessing: raw-to-bids conversion or merge behavior",
        description="Preprocessing pipeline: convert raw EEG to BIDS or merge behavioral data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["raw-to-bids", "merge-behavior"], help="Preprocessing mode")
    add_task_arg(parser)
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--bids-root", type=str, default=None)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    
    raw_group = parser.add_argument_group("raw-to-bids options")
    raw_group.add_argument("--montage", type=str, default="easycap-M1")
    raw_group.add_argument("--line-freq", type=float, default=60.0)
    raw_group.add_argument("--overwrite", action="store_true")
    raw_group.add_argument("--zero-base-onsets", action="store_true")
    raw_group.add_argument("--trim-to-first-volume", action="store_true")
    raw_group.add_argument("--event-prefix", action="append", default=None)
    raw_group.add_argument("--keep-all-annotations", action="store_true")
    
    merge_group = parser.add_argument_group("merge-behavior options")
    merge_group.add_argument("--event-type", action="append", default=None)
    merge_group.add_argument("--dry-run", action="store_true")
    return parser


def _run_preprocessing(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    
    task = args.task or config.get("project.task", "thermalactive")
    
    if args.source_root:
        config.setdefault("paths", {})["source_data"] = args.source_root
    if args.bids_root:
        config._bids_root = Path(args.bids_root)
    
    pipeline = PreprocessingPipeline(config=config)
    
    if args.mode == "raw-to-bids":
        pipeline.run_raw_to_bids(
            task=task,
            subjects=args.subjects,
            montage=args.montage,
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            zero_base_onsets=args.zero_base_onsets,
            do_trim_to_first_volume=args.trim_to_first_volume,
            event_prefixes=args.event_prefix,
            keep_all_annotations=args.keep_all_annotations,
        )
    elif args.mode == "merge-behavior":
        pipeline.run_merge_behavior(
            task=task,
            event_prefixes=args.event_prefix,
            event_types=args.event_type,
            dry_run=args.dry_run,
        )


COMMANDS: List[Command] = [
    Command(
        name="behavior",
        help="Behavior analysis: compute correlations or visualize",
        description="Behavior pipeline: compute correlations or visualize",
        setup=_setup_behavior,
        run=_run_behavior,
    ),
    Command(
        name="features",
        help="Features analysis: extract features or visualize",
        description="Features pipeline: extract features or visualize",
        setup=_setup_features,
        run=_run_features,
    ),
    Command(
        name="erp",
        help="ERP analysis: compute statistics or visualize",
        description="ERP pipeline: compute statistics or visualize ERPs",
        setup=_setup_erp,
        run=_run_erp,
    ),
    Command(
        name="tfr",
        help="TFR visualization: generate time-frequency representations",
        description="TFR pipeline: visualize time-frequency representations",
        setup=_setup_tfr,
        run=_run_tfr,
    ),
    Command(
        name="decoding",
        help="Decoding analysis: run LOSO regression and time-generalization",
        description="Run EEG decoding (LOSO regression + time-generalization)",
        setup=_setup_decoding,
        run=_run_decoding,
    ),
    Command(
        name="preprocessing",
        help="Preprocessing: raw-to-bids conversion or merge behavior",
        description="Preprocessing pipeline: convert raw EEG to BIDS or merge behavioral data",
        setup=_setup_preprocessing,
        run=_run_preprocessing,
        requires_subjects=False,
    ),
]


def get_command(name: str) -> Optional[Command]:
    """Get command by name."""
    for cmd in COMMANDS:
        if cmd.name == name:
            return cmd
    return None


__all__ = ["Command", "COMMANDS", "get_command"]
