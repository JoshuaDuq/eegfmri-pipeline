"""Stats/analytics CLI command."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd

from eeg_pipeline.cli.common import add_task_arg, resolve_task
from eeg_pipeline.infra.paths import deriv_features_path, resolve_deriv_root
from eeg_pipeline.utils.data.subjects import (
    _collect_subjects_from_bids,
    _collect_subjects_from_derivatives_epochs,
    _collect_subjects_from_features,
)


# Constants
BAR_WIDTH = 20
BYTES_PER_KB = 1024
CATEGORY_DISPLAY_LIMIT = 3
HEADER_WIDTH = 50
INDENT_SIZE = 2
PERCENTAGE_PRECISION = 1
STORAGE_BAR_SCALE = 5
TIMELINE_DISPLAY_LIMIT = 20

FEATURE_CATEGORIES = [
    "power",
    "connectivity",
    "aperiodic",
    "erp",
    "bursts",
    "itpc",
    "pac",
    "complexity",
    "quality",
    "erds",
    "spectral",
    "ratios",
    "asymmetry",
]

METADATA_COLUMNS = {"subject", "epoch", "condition", "task"}


def get_dir_size(path: Path) -> int:
    """Calculate total size in bytes of all files under the given directory."""
    total_size = 0
    if path.exists():
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    return total_size


def format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string."""
    remaining_bytes = size_bytes
    for unit in ["B", "KB", "MB", "GB"]:
        if remaining_bytes < BYTES_PER_KB:
            return f"{remaining_bytes:.1f} {unit}"
        remaining_bytes /= BYTES_PER_KB
    return f"{remaining_bytes:.1f} TB"


def _collect_all_subjects(deriv_root: Path, task: str, config: Any) -> tuple[set[str], set[str], set[str]]:
    """Collect subjects from BIDS, epochs, and features."""
    try:
        bids_root = config.bids_root if hasattr(config, "bids_root") else config.get("paths.bids_root")
        if bids_root:
            bids_subjects = set(_collect_subjects_from_bids(Path(bids_root)))
        else:
            bids_subjects = set()
    except (OSError, ValueError, TypeError):
        bids_subjects = set()

    epochs_subjects = set(
        _collect_subjects_from_derivatives_epochs(deriv_root, task, config)
    )
    features_subjects = set(_collect_subjects_from_features(deriv_root))
    return bids_subjects, epochs_subjects, features_subjects



def _count_feature_categories(
    features_subjects: set[str], deriv_root: Path
) -> dict[str, int]:
    """Count how many subjects have each feature category."""
    category_counts = {category: 0 for category in FEATURE_CATEGORIES}
    for subject in features_subjects:
        features_dir = deriv_features_path(deriv_root, subject)
        if not features_dir.exists():
            continue
        for category in FEATURE_CATEGORIES:
            feature_files = list(features_dir.glob(f"features_{category}*"))
            if feature_files:
                category_counts[category] += 1
    return category_counts


def _print_summary_text(
    task: str,
    deriv_root: Path,
    n_total: int,
    n_bids: int,
    n_epochs: int,
    n_features: int,
    pct_epochs: float,
    pct_features: float,
    category_counts: dict[str, int],
) -> None:
    """Print human-readable summary statistics."""
    print("=" * HEADER_WIDTH)
    print("           EEG PIPELINE PROJECT STATS")
    print("=" * HEADER_WIDTH)
    print()
    print(f"  Task:           {task}")
    print(f"  Derivatives:    {deriv_root}")
    print()
    print("  SUBJECTS")
    print("  " + "-" * 30)
    print(f"  Total Discovered:   {n_total:>5}")
    print(f"  With BIDS:          {n_bids:>5}")
    print(f"  With Epochs:        {n_epochs:>5} ({pct_epochs:.0f}%)")
    print(f"  With Features:      {n_features:>5} ({pct_features:.0f}%)")
    print()
    print("  FEATURE CATEGORIES")
    print("  " + "-" * 30)
    for category, count in category_counts.items():
        bar_length = int((count / n_total) * BAR_WIDTH) if n_total > 0 else 0
        filled_bar = "█" * bar_length
        empty_bar = "░" * (BAR_WIDTH - bar_length)
        bar = filled_bar + empty_bar
        print(f"  {category:15} {bar} {count}/{n_total}")
    print()


def _print_subjects_text(
    bids_subjects: set[str],
    epochs_subjects: set[str],
    features_subjects: set[str],
    n_bids: int,
    n_epochs: int,
    n_features: int,
) -> None:
    """Print subject breakdown by pipeline stage."""
    print("SUBJECTS BY PIPELINE STAGE")
    print("=" * 40)

    complete_subjects = features_subjects
    epochs_only_subjects = epochs_subjects - features_subjects
    bids_only_subjects = bids_subjects - epochs_subjects

    if complete_subjects:
        print(f"\n✓ Complete ({len(complete_subjects)})")
        for subject in sorted(complete_subjects):
            print(f"    sub-{subject}")

    if epochs_only_subjects:
        print(f"\n◐ Epochs Only ({len(epochs_only_subjects)})")
        for subject in sorted(epochs_only_subjects):
            print(f"    sub-{subject}")

    if bids_only_subjects:
        print(f"\n○ BIDS Only ({len(bids_only_subjects)})")
        for subject in sorted(bids_only_subjects):
            print(f"    sub-{subject}")


def _handle_summary_mode(
    args: argparse.Namespace,
    task: str,
    deriv_root: Path,
    bids_subjects: set[str],
    epochs_subjects: set[str],
    features_subjects: set[str],
) -> None:
    """Handle summary statistics mode."""
    n_bids = len(bids_subjects)
    n_epochs = len(epochs_subjects)
    n_features = len(features_subjects)
    all_subjects = bids_subjects | epochs_subjects | features_subjects
    n_total = len(all_subjects)

    category_counts = _count_feature_categories(features_subjects, deriv_root)

    if n_total > 0:
        pct_epochs = (n_epochs / n_total) * 100
        pct_features = (n_features / n_total) * 100
    else:
        pct_epochs = 0.0
        pct_features = 0.0

    stats = {
        "total_subjects": n_total,
        "bids_subjects": n_bids,
        "epochs_subjects": n_epochs,
        "features_subjects": n_features,
        "epochs_pct": round(pct_epochs, PERCENTAGE_PRECISION),
        "features_pct": round(pct_features, PERCENTAGE_PRECISION),
        "feature_categories": category_counts,
        "task": task,
        "deriv_root": str(deriv_root),
    }

    if args.output_json:
        print(json.dumps(stats, indent=INDENT_SIZE))
    else:
        _print_summary_text(
            task,
            deriv_root,
            n_total,
            n_bids,
            n_epochs,
            n_features,
            pct_epochs,
            pct_features,
            category_counts,
        )


def _handle_subjects_mode(
    args: argparse.Namespace,
    bids_subjects: set[str],
    epochs_subjects: set[str],
    features_subjects: set[str],
) -> None:
    """Handle subjects breakdown mode."""
    n_bids = len(bids_subjects)
    n_epochs = len(epochs_subjects)
    n_features = len(features_subjects)

    if args.output_json:
        output = {
            "bids_only": list(bids_subjects - epochs_subjects),
            "epochs_only": list(epochs_subjects - features_subjects),
            "complete": list(features_subjects),
            "counts": {
                "bids": n_bids,
                "epochs": n_epochs,
                "features": n_features,
            },
        }
        print(json.dumps(output, indent=INDENT_SIZE))
    else:
        _print_subjects_text(
            bids_subjects,
            epochs_subjects,
            features_subjects,
            n_bids,
            n_epochs,
            n_features,
        )


def _extract_category_from_filename(filename: str) -> str:
    """Extract feature category from TSV filename."""
    stem = filename.replace("features_", "")
    return stem.split("_")[0]


def _count_feature_columns(dataframe: pd.DataFrame) -> int:
    """Count feature columns excluding metadata columns."""
    feature_columns = [
        column for column in dataframe.columns if column not in METADATA_COLUMNS
    ]
    return len(feature_columns)


def _handle_features_mode(
    args: argparse.Namespace,
    deriv_root: Path,
    features_subjects: set[str],
) -> None:
    """Handle feature statistics mode."""
    feature_stats = []
    for subject in sorted(features_subjects):
        features_dir = deriv_features_path(deriv_root, subject)
        if not features_dir.exists():
            continue

        subject_stats = {
            "subject": subject,
            "files": 0,
            "total_features": 0,
            "categories": [],
        }

        for tsv_file in features_dir.glob("features_*.tsv"):
            try:
                dataframe = pd.read_csv(tsv_file, sep="\t", nrows=1)
                feature_count = _count_feature_columns(dataframe)
                subject_stats["files"] += 1
                subject_stats["total_features"] += feature_count

                category = _extract_category_from_filename(tsv_file.stem)
                if category not in subject_stats["categories"]:
                    subject_stats["categories"].append(category)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
                continue

        feature_stats.append(subject_stats)

    if args.output_json:
        output = {"feature_stats": feature_stats}
        print(json.dumps(output, indent=INDENT_SIZE))
    else:
        print("FEATURE STATISTICS BY SUBJECT")
        print("=" * HEADER_WIDTH)
        print(f"{'Subject':15} {'Files':>8} {'Features':>10} {'Categories'}")
        print("-" * HEADER_WIDTH)
        for stats in feature_stats:
            categories = stats["categories"][:CATEGORY_DISPLAY_LIMIT]
            category_display = ", ".join(categories)
            remaining_count = len(stats["categories"]) - CATEGORY_DISPLAY_LIMIT
            if remaining_count > 0:
                category_display += f"... +{remaining_count}"
            print(
                f"sub-{stats['subject']:11} "
                f"{stats['files']:>8} "
                f"{stats['total_features']:>10} "
                f"{category_display}"
            )

        total_files = sum(stats["files"] for stats in feature_stats)
        total_features = sum(stats["total_features"] for stats in feature_stats)
        print("-" * HEADER_WIDTH)
        print(f"{'TOTAL':15} {total_files:>8} {total_features:>10}")


def _handle_storage_mode(
    args: argparse.Namespace,
    deriv_root: Path,
    features_subjects: set[str],
) -> None:
    """Handle storage statistics mode."""
    storage_stats = {}

    features_total = 0
    for subject in features_subjects:
        features_dir = deriv_features_path(deriv_root, subject)
        features_total += get_dir_size(features_dir)
    storage_stats["features"] = features_total

    epochs_dir = deriv_root / "epochs"
    storage_stats["epochs"] = (
        get_dir_size(epochs_dir) if epochs_dir.exists() else 0
    )

    plots_dir = deriv_root / "plots"
    storage_stats["plots"] = (
        get_dir_size(plots_dir) if plots_dir.exists() else 0
    )

    behavior_dir = deriv_root / "behavior"
    storage_stats["behavior"] = (
        get_dir_size(behavior_dir) if behavior_dir.exists() else 0
    )

    total_storage = sum(storage_stats.values())

    if args.output_json:
        output = {
            "storage_bytes": storage_stats,
            "storage_formatted": {
                key: format_size(value)
                for key, value in storage_stats.items()
            },
            "total_bytes": total_storage,
            "total_formatted": format_size(total_storage),
        }
        print(json.dumps(output, indent=INDENT_SIZE))
    else:
        print("STORAGE USAGE")
        print("=" * 40)
        sorted_items = sorted(
            storage_stats.items(), key=lambda item: -item[1]
        )
        for category, size in sorted_items:
            percentage = (
                (size / total_storage * 100) if total_storage > 0 else 0
            )
            bar_length = int(percentage / STORAGE_BAR_SCALE)
            bar = "█" * bar_length
            print(f"  {category:12} {format_size(size):>10}  {bar}")
        print("-" * 40)
        print(f"  {'TOTAL':12} {format_size(total_storage):>10}")


def _handle_timeline_mode(
    args: argparse.Namespace,
    deriv_root: Path,
    features_subjects: set[str],
) -> None:
    """Handle processing timeline mode."""
    feature_times = []
    for subject in features_subjects:
        features_dir = deriv_features_path(deriv_root, subject)
        if not features_dir.exists():
            continue

        tsv_files = list(features_dir.glob("*.tsv"))
        if not tsv_files:
            continue

        latest_mtime = max(file_path.stat().st_mtime for file_path in tsv_files)
        feature_times.append(
            {
                "subject": subject,
                "last_modified": datetime.fromtimestamp(
                    latest_mtime
                ).isoformat(),
                "n_files": len(tsv_files),
            }
        )

    feature_times.sort(key=lambda entry: entry["last_modified"], reverse=True)

    if args.output_json:
        output = {"timeline": feature_times}
        print(json.dumps(output, indent=INDENT_SIZE))
    else:
        print("PROCESSING TIMELINE (most recent first)")
        print("=" * HEADER_WIDTH)
        displayed_entries = feature_times[:TIMELINE_DISPLAY_LIMIT]
        for entry in displayed_entries:
            timestamp = entry["last_modified"][:19]
            print(
                f"  sub-{entry['subject']:11}  "
                f"{timestamp}  "
                f"({entry['n_files']} files)"
            )
        remaining_count = len(feature_times) - TIMELINE_DISPLAY_LIMIT
        if remaining_count > 0:
            print(f"  ... and {remaining_count} more")


def setup_stats(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the stats command parser."""
    parser = subparsers.add_parser(
        "stats",
        help="Project statistics: subjects, features, storage",
        description="Show project-wide statistics and analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["summary", "subjects", "features", "storage", "timeline"],
        nargs="?",
        default="summary",
        help="What to show (default: summary)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    return parser


def run_stats(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the stats command."""
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)

    bids_subjects, epochs_subjects, features_subjects = _collect_all_subjects(
        deriv_root, task, config
    )

    if args.mode == "summary":
        _handle_summary_mode(
            args,
            task,
            deriv_root,
            bids_subjects,
            epochs_subjects,
            features_subjects,
        )
    elif args.mode == "subjects":
        _handle_subjects_mode(
            args, bids_subjects, epochs_subjects, features_subjects
        )
    elif args.mode == "features":
        _handle_features_mode(args, deriv_root, features_subjects)
    elif args.mode == "storage":
        _handle_storage_mode(args, deriv_root, features_subjects)
    elif args.mode == "timeline":
        _handle_timeline_mode(args, deriv_root, features_subjects)
