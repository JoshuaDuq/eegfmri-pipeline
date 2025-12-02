"""
Preprocessing CLI Subcommand
============================

Parser setup and run function for preprocessing (raw-to-BIDS and behavior merge).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eeg_pipeline.cli.common import add_task_arg


def setup_preprocessing_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "preprocessing",
        help="Preprocessing: raw-to-bids conversion or merge behavior",
        description="Preprocessing pipeline: convert raw EEG to BIDS or merge behavioral data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["raw-to-bids", "merge-behavior"],
        help="Preprocessing mode"
    )
    
    add_task_arg(parser)
    
    parser.add_argument(
        "--source-root", type=str, default=None,
        help="Source data root directory"
    )
    parser.add_argument(
        "--bids-root", type=str, default=None,
        help="BIDS output root directory"
    )
    parser.add_argument(
        "--subjects", type=str, nargs="*", default=None,
        help="Subject labels to process (e.g., 001 002)"
    )
    
    raw_group = parser.add_argument_group("raw-to-bids options")
    raw_group.add_argument(
        "--montage", type=str, default="easycap-M1",
        help="Standard montage name (e.g., easycap-M1)"
    )
    raw_group.add_argument(
        "--line-freq", type=float, default=60.0,
        help="Line noise frequency (Hz)"
    )
    raw_group.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing BIDS files"
    )
    raw_group.add_argument(
        "--zero-base-onsets", action="store_true",
        help="Zero-base kept annotation onsets"
    )
    raw_group.add_argument(
        "--trim-to-first-volume", action="store_true",
        help="Crop raw to start at first MRI volume trigger"
    )
    raw_group.add_argument(
        "--event-prefix", action="append", default=None,
        help="Keep annotations with this prefix (repeatable)"
    )
    raw_group.add_argument(
        "--keep-all-annotations", action="store_true",
        help="Keep all annotations without filtering"
    )
    
    merge_group = parser.add_argument_group("merge-behavior options")
    merge_group.add_argument(
        "--event-type", action="append", default=None,
        help="Keep events matching this exact type (repeatable)"
    )
    merge_group.add_argument(
        "--dry-run", action="store_true",
        help="Report planned changes without writing files"
    )


def run_preprocessing(args, config: Any) -> None:
    from eeg_pipeline.pipelines.preprocessing import run_raw_to_bids, run_merge_behavior
    
    task = args.task or config.get("project.task", "thermalactive")
    source_root = Path(args.source_root) if args.source_root else Path(config.get("paths.source_data", "data/source_data"))
    bids_root = Path(args.bids_root) if args.bids_root else config.bids_root
    
    if args.mode == "raw-to-bids":
        run_raw_to_bids(
            source_root=source_root,
            bids_root=bids_root,
            task=task,
            subjects=args.subjects,
            montage=args.montage,
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            zero_base_onsets=args.zero_base_onsets,
            trim_to_first_volume=args.trim_to_first_volume,
            event_prefixes=args.event_prefix,
            keep_all_annotations=args.keep_all_annotations,
        )
    
    elif args.mode == "merge-behavior":
        run_merge_behavior(
            bids_root=bids_root,
            source_root=source_root,
            task=task,
            event_prefixes=args.event_prefix,
            event_types=args.event_type,
            dry_run=args.dry_run,
        )
