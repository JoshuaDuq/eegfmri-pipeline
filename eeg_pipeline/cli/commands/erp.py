"""ERP analysis CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    create_progress_reporter,
    resolve_task,
)


def setup_erp(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the ERP command parser."""
    parser = subparsers.add_parser(
        "erp",
        help="ERP analysis: compute statistics or visualize",
        description="ERP pipeline: compute statistics or visualize ERPs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument("--crop-tmin", type=float, default=None)
    parser.add_argument("--crop-tmax", type=float, default=None)
    return parser


def run_erp(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the ERP command."""
    from eeg_pipeline.pipelines.erp import ErpPipeline
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if args.mode == "compute":
        pipeline = ErpPipeline(config=config)
        pipeline.set_crop_params(args.crop_tmin, args.crop_tmax)
        pipeline.run_batch(
            subjects=subjects,
            task=task,
            progress=progress,
        )
    elif args.mode == "visualize":
        visualize_erp_for_subjects(
            subjects=subjects,
            task=task,
            crop_tmin=args.crop_tmin,
            crop_tmax=args.crop_tmax,
        )
