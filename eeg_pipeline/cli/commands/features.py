"""Features extraction CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    create_progress_reporter,
    resolve_task,
)
from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES, SPATIAL_MODES
from eeg_pipeline.cli.commands.base import FEATURE_VISUALIZE_CATEGORIES

FEATURE_CATEGORY_CHOICES = FEATURE_CATEGORIES + [
    category for category in FEATURE_VISUALIZE_CATEGORIES if category not in FEATURE_CATEGORIES
]


def setup_features(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the features command parser."""
    parser = subparsers.add_parser(
        "features",
        help="Features analysis: extract, combine, or visualize",
        description="Features pipeline: extract features, combine into single file, or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "combine", "visualize"], help="Pipeline mode (combine: merge feature files into features_all.tsv)")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=FEATURE_CATEGORY_CHOICES,
        default=None,
        metavar="CATEGORY",
        help="Feature categories to process (some are compute-only or visualize-only)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=["delta", "theta", "alpha", "beta", "gamma"],
        default=None,
        help="Frequency bands to compute (default: all)",
    )
    parser.add_argument(
        "--spatial",
        nargs="+",
        choices=SPATIAL_MODES,
        default=None,
        metavar="MODE",
        help="Spatial aggregation modes: roi, channels, global (default: roi, global)",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start time in seconds for feature extraction window",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="End time in seconds for feature extraction window (deprecated: use --time-range)",
    )
    parser.add_argument(
        "--time-range",
        nargs=3,
        action="append",
        metavar=("NAME", "TMIN", "TMAX"),
        help="Define a named time range (e.g. baseline 0 1). Can be specified multiple times.",
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method for spatial modes (default: mean)",
    )
    
    
    # Connectivity options
    parser.add_argument(
        "--connectivity-measures",
        nargs="+",
        choices=["wpli", "aec", "plv", "pli"],
        default=None,
        help="Connectivity measures to compute",
    )
    
    # PAC/CFC options
    parser.add_argument(
        "--pac-phase-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Phase frequency range for PAC/CFC (Hz)",
    )
    parser.add_argument(
        "--pac-amp-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Amplitude frequency range for PAC/CFC (Hz)",
    )
    
    # Aperiodic options
    parser.add_argument(
        "--aperiodic-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Frequency range for aperiodic fit (Hz)",
    )
    
    # Complexity options
    parser.add_argument(
        "--pe-order",
        type=int,
        default=None,
        help="Permutation entropy order (3-7, default: from config)",
    )
    
    return parser


def run_features(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the features command."""
    from eeg_pipeline.pipelines.features import FeaturePipeline
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if args.mode == "compute":
        # Apply feature-specific overrides to config
        if getattr(args, "connectivity_measures", None) is not None:
            config["feature_engineering.connectivity.measures"] = args.connectivity_measures
        
        if getattr(args, "pac_phase_range", None) is not None:
            config["feature_engineering.pac.phase_range"] = list(args.pac_phase_range)
        
        if getattr(args, "pac_amp_range", None) is not None:
            config["feature_engineering.pac.amp_range"] = list(args.pac_amp_range)
        
        if getattr(args, "aperiodic_range", None) is not None:
            config["feature_engineering.aperiodic.fmin"] = args.aperiodic_range[0]
            config["feature_engineering.aperiodic.fmax"] = args.aperiodic_range[1]
        
        if getattr(args, "pe_order", None) is not None:
            config["feature_engineering.complexity.pe_order"] = args.pe_order
        
        # Prepare time ranges
        time_ranges = []
        if getattr(args, "time_range", None):
            for name, tmin, tmax in args.time_range:
                time_ranges.append({
                    "name": name,
                    "tmin": float(tmin) if tmin.lower() != "none" and tmin != "" else None,
                    "tmax": float(tmax) if tmax.lower() != "none" and tmax != "" else None,
                })
        
        pipeline = FeaturePipeline(config=config)
        pipeline.run_batch(
            subjects=subjects,
            task=task,
            feature_categories=categories,
            bands=getattr(args, "bands", None),
            spatial_modes=getattr(args, "spatial", None),
            tmin=getattr(args, "tmin", None),
            tmax=getattr(args, "tmax", None),
            time_ranges=time_ranges or None,
            aggregation_method=getattr(args, "aggregation_method", "mean"),
            progress=progress,
        )
    elif args.mode == "combine":
        from eeg_pipeline.utils.data.feature_io import combine_all_features
        from eeg_pipeline.infra.paths import deriv_features_path, resolve_deriv_root
        
        progress.start("features_combine", subjects)
        deriv_root = resolve_deriv_root(config=config)
        
        for subject in subjects:
            features_dir = deriv_features_path(deriv_root, subject)
            if features_dir.exists():
                progress.subject_start(f"sub-{subject}")
                try:
                    combined_df = combine_all_features(features_dir, config)
                    if combined_df is not None:
                        progress.log("info", f"Combined {combined_df.shape[1]} columns into features_all.tsv")
                    else:
                        progress.log("warning", "No features found to combine")
                except Exception as e:
                    progress.log("error", f"Failed to combine: {e}")
                progress.subject_done(f"sub-{subject}", success=True)
            else:
                progress.log("warning", f"No features directory for sub-{subject}")
        
        progress.complete(success=True)
    elif args.mode == "visualize":
        visualize_features_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=categories,
        )
