"""Features extraction CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    create_progress_reporter,
    resolve_task,
)
from eeg_pipeline.pipelines.constants import (
    FEATURE_CATEGORIES,
    FREQUENCY_BANDS,
)
from eeg_pipeline.domain.features.constants import SPATIAL_MODES
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
        choices=FREQUENCY_BANDS,
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
        help="End time in seconds for feature extraction window",
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

    # ERP options
    parser.add_argument("--erp-baseline", action="store_true", default=None, help="Enable baseline correction for ERP")
    parser.add_argument("--no-erp-baseline", action="store_false", dest="erp_baseline", help="Disable baseline correction for ERP")

    # Burst options
    parser.add_argument("--burst-threshold", type=float, default=None, help="Z-score threshold for burst detection")

    # Power options
    parser.add_argument("--power-baseline-mode", choices=["logratio", "mean", "ratio", "zscore", "zlogratio"], default=None, help="Baseline normalization mode for power")

    # Spectral options
    parser.add_argument("--spectral-edge-percentile", type=float, default=None, help="Percentile for spectral edge frequency (0-1)")

    # Connectivity options (extend)
    parser.add_argument("--conn-output-level", choices=["full", "global_only"], default=None, help="Connectivity output level")
    parser.add_argument("--conn-graph-metrics", action="store_true", default=None, help="Enable graph metrics for connectivity")
    parser.add_argument("--no-conn-graph-metrics", action="store_false", dest="conn_graph_metrics", help="Disable graph metrics for connectivity")
    parser.add_argument("--conn-aec-mode", choices=["orth", "sym", "none"], default=None, help="AEC orthogonalization mode")
    
    # New Advanced Options
    
    parser.add_argument("--aperiodic-peak-z", type=float, default=None, help="Peak rejection Z-threshold for aperiodic fit")
    parser.add_argument("--aperiodic-min-r2", type=float, default=None, help="Minimum R2 for aperiodic fit")
    parser.add_argument("--aperiodic-min-points", type=int, default=None, help="Minimum fit points for aperiodic")
    
    parser.add_argument("--conn-graph-prop", type=float, default=None, help="Proportion of top edges to keep for graph metrics")
    parser.add_argument("--conn-window-len", type=float, default=None, help="Sliding window length (s) for connectivity")
    parser.add_argument("--conn-window-step", type=float, default=None, help="Sliding window step (s) for connectivity")
    
    parser.add_argument("--pac-method", choices=["mvl", "kl", "tort", "ozkurt"], default=None, help="PAC estimation method")
    parser.add_argument("--pac-min-epochs", type=int, default=None, help="Minimum epochs for PAC computation")
    
    parser.add_argument("--pe-delay", type=int, default=None, help="Permutation entropy delay")
    parser.add_argument("--burst-min-duration", type=int, default=None, help="Minimum burst duration (ms)")
    
    parser.add_argument("--min-epochs", type=int, default=None, help="Minimum epochs required for features")
    parser.add_argument("--export-all", action="store_true", default=None, help="Export all features into a single file")
    parser.add_argument("--no-export-all", action="store_false", dest="export_all", help="Don't export all features into a single file")
    
    add_path_args(parser)
    
    return parser


def run_features(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the features command."""
    from eeg_pipeline.pipelines.features import FeaturePipeline
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    
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

        if getattr(args, "erp_baseline", None) is not None:
            config["feature_engineering.erp.baseline_correction"] = args.erp_baseline
        
        if getattr(args, "burst_threshold", None) is not None:
            config["feature_engineering.bursts.threshold_z"] = args.burst_threshold

        if getattr(args, "power_baseline_mode", None) is not None:
            config["time_frequency_analysis.baseline_mode"] = args.power_baseline_mode
            
        if getattr(args, "spectral_edge_percentile", None) is not None:
            config["feature_engineering.spectral.edge_percentile"] = args.spectral_edge_percentile

        if getattr(args, "conn_output_level", None) is not None:
            config["feature_engineering.connectivity.output_level"] = args.conn_output_level
        if args.conn_graph_metrics is not None:
            config["feature_engineering.connectivity.enable_graph_metrics"] = args.conn_graph_metrics
        if getattr(args, "conn_aec_mode", None) is not None:
            config["feature_engineering.connectivity.aec_mode"] = args.conn_aec_mode
            
        # New Overrides
            
        if getattr(args, "aperiodic_peak_z", None) is not None:
            config["feature_engineering.aperiodic.peak_rejection_z"] = args.aperiodic_peak_z
        if getattr(args, "aperiodic_min_r2", None) is not None:
            config["feature_engineering.aperiodic.min_r2"] = args.aperiodic_min_r2
        if getattr(args, "aperiodic_min_points", None) is not None:
            config["feature_engineering.aperiodic.min_fit_points"] = args.aperiodic_min_points
            
        if getattr(args, "conn_graph_prop", None) is not None:
            config["feature_engineering.connectivity.graph_top_prop"] = args.conn_graph_prop
        if getattr(args, "conn_window_len", None) is not None:
            config["feature_engineering.connectivity.sliding_window_len"] = args.conn_window_len
        if getattr(args, "conn_window_step", None) is not None:
            config["feature_engineering.connectivity.sliding_window_step"] = args.conn_window_step
            
        if getattr(args, "pac_method", None) is not None:
            config["feature_engineering.pac.method"] = args.pac_method
        if getattr(args, "pac_min_epochs", None) is not None:
            config["feature_engineering.pac.min_epochs"] = args.pac_min_epochs
            
        if getattr(args, "pe_delay", None) is not None:
            config["feature_engineering.complexity.pe_delay"] = args.pe_delay
        if getattr(args, "burst_min_duration", None) is not None:
            config["feature_engineering.bursts.min_duration_ms"] = args.burst_min_duration
            
        if getattr(args, "min_epochs", None) is not None:
            config["feature_engineering.constants.min_epochs_for_features"] = args.min_epochs
            
        if args.export_all is not None:
            config["feature_engineering.create_combined_features"] = args.export_all
        
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
