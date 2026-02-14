"""Utilities CLI command (raw-to-bids, merge-psychopy, clean)."""

from __future__ import annotations

import argparse
import json
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
from eeg_pipeline.utils.config.overrides import apply_runtime_overrides


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
        choices=["raw-to-bids", "fmri-raw-to-bids", "merge-psychopy", "clean"],
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
    parser.add_argument(
        "--bids-fmri-root",
        type=str,
        default=None,
        help="Override BIDS fMRI root path (default from config)"
    )

    raw_group = parser.add_argument_group("raw-to-bids options")
    raw_group.add_argument("--montage", type=str, default="easycap-M1")
    raw_group.add_argument("--line-freq", type=float, default=60.0)
    raw_group.add_argument("--overwrite", action="store_true")
    raw_group.add_argument("--trim-to-first-volume", action="store_true")
    raw_group.add_argument("--event-prefix", action="append")
    raw_group.add_argument("--keep-all-annotations", action="store_true")

    fmri_group = parser.add_argument_group("fmri-raw-to-bids options")
    fmri_group.add_argument("--session", type=str, default=None, help="Optional BIDS session label (without 'ses-')")
    fmri_group.add_argument("--rest-task", type=str, default="rest", help="Task label to use for resting-state BOLD")
    fmri_group.add_argument(
        "--no-rest",
        action="store_true",
        help="Do not convert resting-state series (if present)",
    )
    fmri_group.add_argument(
        "--no-fieldmaps",
        action="store_true",
        help="Do not convert fieldmaps (if present)",
    )
    fmri_group.add_argument(
        "--dicom-mode",
        choices=["symlink", "copy", "skip"],
        default="symlink",
        help="How to store original DICOMs under <bids_fmri_root>/sourcedata (default: symlink)",
    )
    fmri_group.add_argument(
        "--no-events",
        action="store_true",
        help="Do not generate BIDS *_events.tsv from PsychoPy TrialSummary.csv",
    )
    fmri_group.add_argument(
        "--event-granularity",
        choices=["trial", "phases"],
        default="phases",
        help="Events granularity for stimulation (default: phases)",
    )
    fmri_group.add_argument(
        "--onset-reference",
        choices=["as_is", "first_iti_start", "first_stim_start"],
        default="as_is",
        help="How to zero event onsets within each run (default: as_is)",
    )
    fmri_group.add_argument(
        "--onset-offset-s",
        type=float,
        default=0.0,
        help="Additive offset applied after onset-reference (seconds)",
    )
    fmri_group.add_argument(
        "--dcm2niix-path",
        type=str,
        default=None,
        help="Path to dcm2niix binary (defaults to PATH lookup)",
    )
    fmri_group.add_argument(
        "--dcm2niix-arg",
        action="append",
        default=None,
        help="Extra argument passed to dcm2niix (repeatable, inserted after executable)",
    )

    merge_group = parser.add_argument_group("merge-psychopy options")
    merge_group.add_argument("--event-type", action="append")
    merge_group.add_argument(
        "--qc-column",
        action="append",
        default=None,
        help="Cross-modal EEG↔fMRI QC column to compare (repeatable). If omitted, cross-modal column QC is skipped.",
    )
    merge_group.add_argument(
        "--allow-misaligned-trim",
        action="store_true",
        help="Allow behavioral/events count mismatch (trims/pads). Not recommended.",
    )

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


def _run_raw_to_bids_mode(
    task: str,
    subjects: List[str],
    args: argparse.Namespace,
    config: Any,
    progress: Any,
) -> None:
    """Execute raw-to-bids conversion mode."""
    progress.start("utilities_raw_to_bids", subjects)
    progress.step("Converting to BIDS", current=1, total=2)

    from eeg_pipeline.pipelines.eeg_raw_to_bids import EEGRawToBidsPipeline

    EEGRawToBidsPipeline(config=config).run_batch(
        subjects,
        task=task,
        montage=args.montage,
        line_freq=args.line_freq,
        overwrite=args.overwrite,
        do_trim_to_first_volume=args.trim_to_first_volume,
        event_prefixes=args.event_prefix,
        keep_all_annotations=args.keep_all_annotations,
    )
    
    progress.step("Finalizing", current=2, total=2)
    progress.complete(success=True)


def _run_merge_psychopy_mode(
    task: str,
    subjects: List[str],
    args: argparse.Namespace,
    config: Any,
    progress: Any,
) -> None:
    """Execute merge-psychopy mode."""
    progress.start("utilities_merge_psychopy", subjects)
    progress.step("Merging PsychoPy into events.tsv", current=1, total=2)

    from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

    MergePsychopyPipeline(config=config).run_batch(
        subjects,
        task=task,
        event_prefixes=args.event_prefix,
        event_types=args.event_type,
        qc_columns=getattr(args, "qc_column", None),
        dry_run=args.dry_run,
        allow_misaligned_trim=bool(getattr(args, "allow_misaligned_trim", False)),
    )
    
    progress.step("Finalizing", current=2, total=2)
    progress.complete(success=True)


def run_utilities(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the utilities command."""
    if args.mode == "clean":
        return _run_clean_mode(args, subjects, config)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    apply_runtime_overrides(
        config,
        source_root=getattr(args, "source_root", None),
        bids_root=getattr(args, "bids_root", None),
        bids_fmri_root=getattr(args, "bids_fmri_root", None),
    )

    if args.mode == "fmri-raw-to-bids":
        return _run_fmri_raw_to_bids_mode(task, subjects, args, config, progress)
    if args.mode == "raw-to-bids":
        return _run_raw_to_bids_mode(task, subjects, args, config, progress)
    if args.mode == "merge-psychopy":
        return _run_merge_psychopy_mode(task, subjects, args, config, progress)


def _run_fmri_raw_to_bids_mode(
    task: str,
    subjects: List[str],
    args: argparse.Namespace,
    config: Any,
    progress: Any,
) -> None:
    """Execute fMRI raw-to-bids conversion mode."""
    from fmri_pipeline.analysis.raw_to_bids import run_fmri_raw_to_bids

    source_root = Path(config.get("paths.source_data", "data/source_data"))
    bids_fmri_root = Path(config.get("paths.bids_fmri_root", "data/fMRI_data"))

    progress.start("utilities_fmri_raw_to_bids", subjects)
    progress.step("Converting DICOM to BIDS (fMRI)", current=1, total=2)

    run_fmri_raw_to_bids(
        source_root=source_root,
        bids_fmri_root=bids_fmri_root,
        task=task,
        subjects=subjects,
        session=args.session,
        rest_task=args.rest_task,
        include_rest=not args.no_rest,
        include_fieldmaps=not args.no_fieldmaps,
        dicom_mode=args.dicom_mode,
        overwrite=args.overwrite,
        create_events=not args.no_events,
        event_granularity=args.event_granularity,
        onset_reference=args.onset_reference,
        onset_offset_s=args.onset_offset_s,
        dcm2niix_path=args.dcm2niix_path,
        dcm2niix_extra_args=args.dcm2niix_arg,
        _logger=None,
    )

    progress.step("Finalizing", current=2, total=2)
    progress.complete(success=True)


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

    if args.target in ("preview", "all"):
        print("CLEANUP PREVIEW")
        print("=" * 50)
        for name, path in targets.items():
            if path.exists():
                size = get_dir_size(path)
                print(f"  {name:15} {format_size(size):>10}")
                total_size += size

        if args.subjects:
            for subj in args.subjects:
                subject_id = subj.replace("sub-", "") if subj.startswith("sub-") else subj
                feat_dir = deriv_root / "features" / f"sub-{subject_id}"
                if feat_dir.exists():
                    size = get_dir_size(feat_dir)
                    total_size += size
                    print(f"  sub-{subject_id} features {format_size(size):>10}")

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
        print(json.dumps({
            "files": [str(f) for f in to_remove[:50]],
            "total_files": len(to_remove),
            "total_size": total_size,
            "total_size_formatted": format_size(total_size),
        }, indent=2))
        return

    print(f"Found {len(to_remove)} files ({format_size(total_size)})")

    if args.dry_run:
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
            f.unlink()
            deleted += 1
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not delete {f}: {e}")

    print(f"✓ Deleted {deleted} files ({format_size(total_size)})")
