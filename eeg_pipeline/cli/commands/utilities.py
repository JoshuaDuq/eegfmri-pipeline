"""Utilities CLI command (raw-to-bids, merge-behavior, clean)."""

from __future__ import annotations

import argparse
import json as json_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional

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
        help="Utilities: raw-to-bids conversion, merge behavior, clean up disk space",
        description="Utilities pipeline: convert raw EEG to BIDS, merge behavioral data, or clean up disk space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["raw-to-bids", "merge-behavior", "clean"],
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

    clean_group = parser.add_argument_group("clean options")
    clean_group.add_argument(
        "--target",
        choices=["plots", "cache", "logs", "old-features", "all", "preview"],
        default="preview",
        help="What to clean (default: preview shows what would be removed)",
    )
    clean_group.add_argument(
        "--older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="Only remove files older than N days",
    )
    clean_group.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Only clean specific subjects",
    )
    clean_group.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

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
    if args.mode == "clean":
        return _run_clean_mode(args, subjects, config)

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


def get_dir_size(root: Path) -> int:
    """Return total size in bytes of all files under the given directory."""
    total_size = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
    return total_size


def format_size(size_bytes: int) -> str:
    """Format a byte size into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def is_old_enough(path: Path, days: Optional[int]) -> bool:
    """Return True if the given path is older than the specified number of days."""
    if days is None:
        return True
    modification_time = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - modification_time > timedelta(days=days)


def _run_clean_mode(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute clean mode: clean up disk space."""
    from eeg_pipeline.infra.paths import resolve_deriv_root

    deriv_root = resolve_deriv_root(config=config)

    targets = {
        "plots": deriv_root / "plots",
        "cache": deriv_root / ".cache",
        "logs": deriv_root / "logs",
    }

    to_remove = []
    total_size = 0

    if args.target == "preview" or args.target == "all":
        print("CLEANUP PREVIEW")
        print("=" * 50)
        for name, path in targets.items():
            if path.exists():
                size = get_dir_size(path)
                print(f"  {name:15} {format_size(size):>10}")
                total_size += size

        if args.subjects:
            for subj in args.subjects:
                feat_dir = deriv_root / "features" / f"sub-{subj.replace('sub-', '')}"
                if feat_dir.exists():
                    size = get_dir_size(feat_dir)
                    total_size += size
                    print(f"  sub-{subj} features {format_size(size):>10}")

        print("-" * 50)
        print(f"  {'TOTAL':15} {format_size(total_size):>10}")
        print()
        print("Use 'utilities clean --target plots', 'utilities clean --target cache', etc. to remove specific items")
        print("Use 'utilities clean --target all' to remove everything")
        return

    target_path = targets.get(args.target)

    if args.target == "old-features":
        if args.older_than is None:
            print("Error: --older-than is required for old-features")
            return

        features_root = deriv_root / "features"
        if features_root.exists():
            for subj_dir in features_root.iterdir():
                if subj_dir.is_dir():
                    for tsv in subj_dir.glob("*.tsv"):
                        if is_old_enough(tsv, args.older_than):
                            to_remove.append(tsv)
                            total_size += tsv.stat().st_size

    elif target_path and target_path.exists():
        for item in target_path.rglob("*"):
            if item.is_file() and is_old_enough(item, args.older_than):
                to_remove.append(item)
                total_size += item.stat().st_size

    elif args.target == "all":
        for name, path in targets.items():
            if path.exists():
                for item in path.rglob("*"):
                    if item.is_file() and is_old_enough(item, args.older_than):
                        to_remove.append(item)
                        total_size += item.stat().st_size

    if not to_remove:
        print("Nothing to clean.")
        return

    if args.output_json:
        print(json_module.dumps({
            "files": [str(f) for f in to_remove[:50]],
            "total_files": len(to_remove),
            "total_size": total_size,
            "total_size_formatted": format_size(total_size),
        }, indent=2))
        return

    print(f"Found {len(to_remove)} files ({format_size(total_size)})")

    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        print("\nDry run - no files removed. Files that would be deleted:")
        for f in to_remove[:20]:
            print(f"  {f}")
        if len(to_remove) > 20:
            print(f"  ... and {len(to_remove) - 20} more")
        return

    if not args.force:
        response = input(f"\nDelete {len(to_remove)} files ({format_size(total_size)})? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    deleted = 0
    for f in to_remove:
        try:
            if f.is_file():
                f.unlink()
                deleted += 1
        except Exception as e:
            print(f"Warning: Could not delete {f}: {e}")

    print(f"✓ Deleted {deleted} files ({format_size(total_size)})")
