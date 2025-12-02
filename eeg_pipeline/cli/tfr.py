"""
TFR CLI Subcommand
==================

Parser setup and run function for time-frequency visualization.
"""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import add_common_subject_args, add_task_arg


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
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Number of parallel jobs (-1 = all cores minus 1, 1 = sequential). Defaults to config or auto."
    )


def run_tfr(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.plotting.tfr import visualize_tfr_for_subjects
    
    if args.mode == "visualize":
        visualize_tfr_for_subjects(
            subjects=subjects,
            task=args.task,
            tfr_roi_only=args.tfr_roi,
            tfr_topomaps_only=args.tfr_topomaps_only,
            n_jobs=args.n_jobs,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'visualize'.")
