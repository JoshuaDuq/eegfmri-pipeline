"""Utilities CLI command (raw-to-bids, merge-behavior)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    create_progress_reporter,
)


def setup_utilities(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the utilities command parser."""
    parser = subparsers.add_parser(
        "utilities",
        help="Utilities: raw-to-bids conversion or merge behavior",
        description="Utilities pipeline: convert raw EEG to BIDS or merge behavioral data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["raw-to-bids", "merge-behavior"], help="Preprocessing mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--bids-root", type=str, default=None)
    
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


    return parser


def run_utilities(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the utilities command."""
    from eeg_pipeline.pipelines.utilities import UtilityPipeline
    
    progress = create_progress_reporter(args)
    task = args.task or config.get("project.task", "thermalactive")
    
    if args.source_root:
        config.setdefault("paths", {})["source_data"] = args.source_root
    if args.bids_root:
        config.setdefault("paths", {})["bids_root"] = str(Path(args.bids_root))
    
    pipeline = UtilityPipeline(config=config)
    target_subjects = subjects or []
    
    if args.mode == "raw-to-bids":
        progress.start("utilities_raw_to_bids", target_subjects)
        progress.step("Converting to BIDS", current=1, total=2)
        
        pipeline.run_raw_to_bids(
            task=task,
            subjects=target_subjects,
            montage=args.montage,
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            zero_base_onsets=args.zero_base_onsets,
            do_trim_to_first_volume=args.trim_to_first_volume,
            event_prefixes=args.event_prefix,
            keep_all_annotations=args.keep_all_annotations,
        )
        
        progress.step("Finalizing", current=2, total=2)
        progress.complete(success=True)
    elif args.mode == "merge-behavior":
        progress.start("utilities_merge_behavior", target_subjects)
        progress.step("Merging behavior", current=1, total=2)
        
        pipeline.run_merge_behavior(
            task=task,
            event_prefixes=args.event_prefix,
            event_types=args.event_type,
            dry_run=getattr(args, "dry_run", False),
        )
        
        progress.step("Finalizing", current=2, total=2)
        progress.complete(success=True)
