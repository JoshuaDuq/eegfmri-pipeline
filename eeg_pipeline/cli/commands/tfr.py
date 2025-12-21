"""TFR visualization CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    create_progress_reporter,
)


def setup_tfr(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the TFR command parser."""
    parser = subparsers.add_parser(
        "tfr",
        help="TFR visualization: generate time-frequency representations",
        description="TFR pipeline: visualize time-frequency representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["visualize"], help="Pipeline mode (only visualize available)")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)

    # Channel selection
    channel_group = parser.add_argument_group("Channel selection")
    channel_group.add_argument("--tfr-roi", action="store_true", help="ROI-only visualization")
    channel_group.add_argument(
        "--rois",
        type=str,
        default=None,
        metavar="ROIS",
        help="Comma-separated list of ROIs (e.g., 'Frontal,Midline_ACC_MCC'). Default: all ROIs",
    )
    channel_group.add_argument("--all-channels", action="store_true", help="Plot all individual channels")
    channel_group.add_argument(
        "--channels",
        type=str,
        default=None,
        metavar="CHANNELS",
        help="Comma-separated list of specific channels (e.g., 'Cz,Fz,Pz')",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument("--tfr-topomaps-only", action="store_true", help="Topomaps only")
    viz_group.add_argument(
        "--bands",
        nargs="+",
        choices=["delta", "theta", "alpha", "beta", "gamma"],
        default=None,
        help="Frequency bands to visualize (default: all)",
    )
    viz_group.add_argument("--tmin", type=float, default=None, help="Start time in seconds")
    viz_group.add_argument("--tmax", type=float, default=None, help="End time in seconds")

    # Processing options
    parser.add_argument("--do-group", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser


def run_tfr(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the TFR command."""
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects

    progress = create_progress_reporter(args)

    if args.mode == "visualize":
        progress.start("tfr_visualize", subjects)
        progress.step("Rendering TFR plots", current=1, total=2)

        # Determine channel mode
        channels = None
        if args.channels:
            channels = [ch.strip() for ch in args.channels.split(",")]

        # Update config with band/time selections if provided
        if args.bands:
            config.setdefault("time_frequency_analysis", {})["selected_bands"] = args.bands
        if args.tmin is not None:
            config.setdefault("time_frequency_analysis", {})["tmin"] = args.tmin
        if args.tmax is not None:
            config.setdefault("time_frequency_analysis", {})["tmax"] = args.tmax

        visualize_tfr_for_subjects(
            subjects=subjects,
            task=args.task,
            tfr_roi_only=args.tfr_roi,
            tfr_topomaps_only=args.tfr_topomaps_only,
            n_jobs=args.n_jobs,
            config=config,
        )

        progress.step("Finalizing", current=2, total=2)
        progress.complete(success=True)
