"""Cleanup CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import add_task_arg


def setup_clean(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the clean command parser."""
    parser = subparsers.add_parser(
        "clean",
        help="Clean up disk space: remove plots, cache, old files",
        description="Remove generated files to free disk space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        choices=["plots", "cache", "logs", "old-features", "all", "preview"],
        nargs="?",
        default="preview",
        help="What to clean (default: preview shows what would be removed)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="Only remove files older than N days",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Only clean specific subjects",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    return parser


def run_clean(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the clean command."""
    from eeg_pipeline.infra.paths import resolve_deriv_root
    
    deriv_root = resolve_deriv_root(config=config)
    
    def get_dir_size(path: Path) -> int:
        total = 0
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total
    
    def format_size(size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def is_old_enough(path: Path, days: int) -> bool:
        if days is None:
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime > timedelta(days=days)
    
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
        print("Use 'clean plots', 'clean cache', etc. to remove specific items")
        print("Use 'clean all' to remove everything")
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
            if f.is_file():
                f.unlink()
                deleted += 1
        except Exception as e:
            print(f"Warning: Could not delete {f}: {e}")
    
    print(f"✓ Deleted {deleted} files ({format_size(total_size)})")
