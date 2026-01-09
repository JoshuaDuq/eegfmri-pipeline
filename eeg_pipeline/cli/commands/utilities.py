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
    resolve_task,
)


def setup_utilities(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the utilities command parser."""
    parser = subparsers.add_parser(
        "utilities",
        help="Utilities: raw-to-bids conversion or merge behavior",
        description="Utilities pipeline: convert raw EEG to BIDS or merge behavioral data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["raw-to-bids", "merge-behavior"],
        help="Utility operation mode"
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Override source data root path (default from config)"
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default=None,
        help="Override BIDS root path (default from config)"
    )
    
    raw_group = parser.add_argument_group("raw-to-bids options")
    raw_group.add_argument("--montage", type=str, default="easycap-M1")
    raw_group.add_argument("--line-freq", type=float, default=60.0)
    raw_group.add_argument("--overwrite", action="store_true")
    raw_group.add_argument("--zero-base-onsets", action="store_true")
    raw_group.add_argument("--trim-to-first-volume", action="store_true")
    raw_group.add_argument("--event-prefix", action="append")
    raw_group.add_argument("--keep-all-annotations", action="store_true")
    
    merge_group = parser.add_argument_group("merge-behavior options")
    merge_group.add_argument("--event-type", action="append")

    return parser


def _update_config_paths(config: Any, source_root: str | None, bids_root: str | None) -> None:
    """Update config with path overrides if provided."""
    if source_root:
        config.setdefault("paths", {})["source_data"] = source_root
    if bids_root:
        config.setdefault("paths", {})["bids_root"] = str(Path(bids_root))


def _run_raw_to_bids_mode(
    pipeline: Any,
    task: str,
    subjects: List[str],
    args: argparse.Namespace,
    progress: Any,
) -> None:
    """Execute raw-to-bids conversion mode."""
    progress.start("utilities_raw_to_bids", subjects)
    progress.step("Converting to BIDS", current=1, total=2)
    
    pipeline.run_raw_to_bids(
        task=task,
        subjects=subjects,
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


def _run_merge_behavior_mode(
    pipeline: Any,
    task: str,
    subjects: List[str],
    args: argparse.Namespace,
    progress: Any,
) -> None:
    """Execute merge-behavior mode."""
    progress.start("utilities_merge_behavior", subjects)
    progress.step("Merging behavior", current=1, total=2)
    
    dry_run = getattr(args, "dry_run", False)
    pipeline.run_merge_behavior(
        task=task,
        event_prefixes=args.event_prefix,
        event_types=args.event_type,
        dry_run=dry_run,
    )
    
    progress.step("Finalizing", current=2, total=2)
    progress.complete(success=True)


def run_utilities(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the utilities command."""
    from eeg_pipeline.pipelines.utilities import UtilityPipeline
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    target_subjects = subjects or []
    
    _update_config_paths(config, args.source_root, args.bids_root)
    pipeline = UtilityPipeline(config=config)
    
    if args.mode == "raw-to-bids":
        _run_raw_to_bids_mode(pipeline, task, target_subjects, args, progress)
    elif args.mode == "merge-behavior":
        _run_merge_behavior_mode(pipeline, task, target_subjects, args, progress)
