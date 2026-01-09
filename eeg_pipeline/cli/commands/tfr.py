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


def _update_tfr_config(
    config: Any,
    bands: Optional[List[str]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> None:
    """Update config with TFR analysis parameters."""
    tfr_section = config.setdefault("time_frequency_analysis", {})
    if bands is not None:
        tfr_section["selected_bands"] = bands
    if tmin is not None:
        tfr_section["tmin"] = tmin
    if tmax is not None:
        tfr_section["tmax"] = tmax


def _validate_time_range(tmin: Optional[float], tmax: Optional[float]) -> None:
    """Validate that time range is logically consistent."""
    if tmin is not None and tmax is not None and tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")


def setup_tfr(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the TFR command parser."""
    parser = subparsers.add_parser(
        "tfr",
        help="TFR visualization: generate time-frequency representations",
        description="TFR pipeline: visualize time-frequency representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["visualize"],
        help="Pipeline mode (only visualize available)",
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)

    channel_group = parser.add_argument_group("Channel selection")
    channel_group.add_argument(
        "--tfr-roi",
        action="store_true",
        help="ROI-only visualization",
    )

    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument(
        "--tfr-topomaps-only",
        action="store_true",
        help="Topomaps only",
    )
    viz_group.add_argument(
        "--bands",
        nargs="+",
        choices=["delta", "theta", "alpha", "beta", "gamma"],
        default=None,
        help="Frequency bands to visualize (default: all)",
    )
    viz_group.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start time in seconds",
    )
    viz_group.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="End time in seconds",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: from config)",
    )
    return parser


def run_tfr(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the TFR command."""
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects

    if args.mode != "visualize":
        raise ValueError(f"Unsupported mode: {args.mode}")

    _validate_time_range(args.tmin, args.tmax)
    _update_tfr_config(config, args.bands, args.tmin, args.tmax)

    progress = create_progress_reporter(args)
    progress.start("tfr_visualize", subjects)
    progress.step("Rendering TFR plots", current=1, total=2)

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
