"""
ERP CLI Subcommand
==================

Parser setup and run function for ERP analysis.
"""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import add_common_subject_args, add_task_arg


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
