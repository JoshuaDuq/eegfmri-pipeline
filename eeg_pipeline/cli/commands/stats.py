"""Stats/analytics CLI command."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from eeg_pipeline.cli.common import add_task_arg, resolve_task
from eeg_pipeline.infra.paths import deriv_features_path, resolve_deriv_root
from eeg_pipeline.pipelines.constants import FEATURE_CATEGORIES
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

METADATA_COLUMNS = {"subject", "epoch", "condition", "task"}



def _resolve_features_dir(deriv_root: Path, subject: str) -> Optional[Path]:
    """Return the first existing features directory for the subject.

    Checks: deriv_root/sub-XXX/eeg/features, then deriv_root/preprocessed/eeg/sub-XXX/features.
    """
    standard = deriv_features_path(deriv_root, subject)
    if standard.exists():
        return standard
    preproc = deriv_root / "preprocessed" / "eeg" / f"sub-{subject}" / "features"
    if preproc.exists():
        return preproc
    return None

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
    if size_bytes == 0:
        return "0.0 B"
    
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < BYTES_PER_KB:
            return f"{size:.1f} {unit}"
        size /= BYTES_PER_KB
    return f"{size:.1f} TB"


def _resolve_fmriprep_output_dir(deriv_root: Path, config: Any) -> Path:
    """Resolve fMRIPrep output parent dir: fmri_preprocessing.fmriprep.output_dir or deriv_root/preprocessed/fmri."""
    raw = config.get("fmri_preprocessing.fmriprep.output_dir")
    if raw and str(raw).strip():
        return Path(str(raw).strip()).expanduser().resolve()
    return deriv_root / "preprocessed" / "fmri"


def _collect_all_subjects(deriv_root: Path, task: str, config: Any) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Collect subjects from BIDS, epochs, features, and preprocessing directories.
    
    Returns:
        Tuple of (bids_subjects, epochs_subjects, features_subjects, eeg_prep_subjects, fmri_prep_subjects)
    """
    try:
        bids_root = config.get("paths.bids_root")
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

    # Collect EEG preprocessing subjects (preprocessed directory)
    eeg_prep_subjects = set()
    eeg_prep_dir = deriv_root / "preprocessed" / "eeg"
    if eeg_prep_dir.exists():
        for subj_dir in eeg_prep_dir.glob("sub-*"):
            if subj_dir.is_dir():
                eeg_dir = subj_dir / "eeg"
                if eeg_dir.exists():
                    # Check for ICA files as indicator of preprocessing
                    if list(eeg_dir.glob("*ica.fif")) or list(eeg_dir.glob("*_components.tsv")):
                        subj_id = subj_dir.name.replace("sub-", "")
                        eeg_prep_subjects.add(subj_id)

    # Collect fMRI preprocessing subjects (fMRIPrep output).
    # Path: output_dir from fmri_preprocessing.fmriprep.output_dir or deriv_root/preprocessed/fmri.
    # Layout: <output_dir>/sub-XXX/func/*preproc_bold* (subject dirs directly under output_dir).
    fmri_prep_subjects = set()
    fmri_output_dir = _resolve_fmriprep_output_dir(deriv_root, config)
    if fmri_output_dir.exists():
        for subj_dir in fmri_output_dir.glob("sub-*"):
            if not subj_dir.is_dir():
                continue
            func_dir = subj_dir / "func"
            if func_dir.exists() and list(func_dir.glob("*preproc_bold*")):
                subj_id = subj_dir.name.replace("sub-", "")
                fmri_prep_subjects.add(subj_id)

    return bids_subjects, epochs_subjects, features_subjects, eeg_prep_subjects, fmri_prep_subjects


def _collect_fmri_analysis_subjects(deriv_root: Path) -> tuple[set[str], set[str], set[str]]:
    """Collect subjects with fMRI analysis outputs.

    Paths (aligned with fmri_pipeline):
      - first_level: deriv_root/sub-XXX/fmri/first_level/
      - beta_series: deriv_root/sub-XXX/fmri/beta_series/
      - lss: deriv_root/sub-XXX/fmri/lss/

    Returns:
        (first_level_subjects, beta_series_subjects, lss_subjects)
    """
    first_level_subjects: set[str] = set()
    beta_series_subjects: set[str] = set()
    lss_subjects: set[str] = set()

    for subj_dir in deriv_root.glob("sub-*"):
        if not subj_dir.is_dir():
            continue
        subj_id = subj_dir.name.replace("sub-", "")
        fmri_dir = subj_dir / "fmri"

        fl_dir = fmri_dir / "first_level"
        if fl_dir.exists() and any(fl_dir.rglob("*")) and any(f.is_file() for f in fl_dir.rglob("*")):
            first_level_subjects.add(subj_id)

        bs_dir = fmri_dir / "beta_series"
        if bs_dir.exists() and any(bs_dir.rglob("*")) and any(f.is_file() for f in bs_dir.rglob("*")):
            beta_series_subjects.add(subj_id)

        lss_dir = fmri_dir / "lss"
        if lss_dir.exists() and any(lss_dir.rglob("*")) and any(f.is_file() for f in lss_dir.rglob("*")):
            lss_subjects.add(subj_id)

    return first_level_subjects, beta_series_subjects, lss_subjects


def _count_feature_categories(
    features_subjects: set[str], deriv_root: Path
) -> dict[str, int]:
    """Count how many subjects have each feature category.

    Looks for features_{category}* under sub-XXX/eeg/features (and subdirs)
    and under preprocessed/eeg/sub-XXX/features (and subdirs). Supports .tsv and .parquet.
    """
    category_counts = {category: 0 for category in FEATURE_CATEGORIES}
    for subject in features_subjects:
        features_dir = _resolve_features_dir(deriv_root, subject)
        if features_dir is None:
            continue
        for category in FEATURE_CATEGORIES:
            found = any(
                f.suffix in {".tsv", ".parquet"}
                for f in features_dir.rglob(f"features_{category}*")
                if f.is_file()
            )
            if found:
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
    eeg_prep_subjects: set[str],
    fmri_prep_subjects: set[str],
    fmri_first_level_subjects: set[str],
    fmri_beta_series_subjects: set[str],
    fmri_lss_subjects: set[str],
) -> None:
    """Handle summary statistics mode."""
    n_bids = len(bids_subjects)
    n_epochs = len(epochs_subjects)
    n_features = len(features_subjects)
    n_eeg_prep = len(eeg_prep_subjects)
    n_fmri_prep = len(fmri_prep_subjects)
    n_fmri_first_level = len(fmri_first_level_subjects)
    n_fmri_beta_series = len(fmri_beta_series_subjects)
    n_fmri_lss = len(fmri_lss_subjects)
    all_subjects = (
        bids_subjects | epochs_subjects | features_subjects
        | eeg_prep_subjects | fmri_prep_subjects
        | fmri_first_level_subjects | fmri_beta_series_subjects | fmri_lss_subjects
    )
    n_total = len(all_subjects)

    category_counts = _count_feature_categories(features_subjects, deriv_root)

    if n_total > 0:
        pct_epochs = (n_epochs / n_total) * 100
        pct_features = (n_features / n_total) * 100
        pct_eeg_prep = (n_eeg_prep / n_total) * 100
        pct_fmri_prep = (n_fmri_prep / n_total) * 100
        pct_fmri_first_level = (n_fmri_first_level / n_total) * 100
        pct_fmri_beta_series = (n_fmri_beta_series / n_total) * 100
        pct_fmri_lss = (n_fmri_lss / n_total) * 100
    else:
        pct_epochs = 0.0
        pct_features = 0.0
        pct_eeg_prep = 0.0
        pct_fmri_prep = 0.0
        pct_fmri_first_level = 0.0
        pct_fmri_beta_series = 0.0
        pct_fmri_lss = 0.0

    stats = {
        "total_subjects": n_total,
        "bids_subjects": n_bids,
        "eeg_prep_subjects": n_eeg_prep,
        "eeg_prep_pct": round(pct_eeg_prep, PERCENTAGE_PRECISION),
        "fmri_prep_subjects": n_fmri_prep,
        "fmri_prep_pct": round(pct_fmri_prep, PERCENTAGE_PRECISION),
        "fmri_first_level_subjects": n_fmri_first_level,
        "fmri_first_level_pct": round(pct_fmri_first_level, PERCENTAGE_PRECISION),
        "fmri_beta_series_subjects": n_fmri_beta_series,
        "fmri_beta_series_pct": round(pct_fmri_beta_series, PERCENTAGE_PRECISION),
        "fmri_lss_subjects": n_fmri_lss,
        "fmri_lss_pct": round(pct_fmri_lss, PERCENTAGE_PRECISION),
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
    if args.output_json:
        output = {
            "bids_only": list(bids_subjects - epochs_subjects),
            "epochs_only": list(epochs_subjects - features_subjects),
            "complete": list(features_subjects),
            "counts": {
                "bids": len(bids_subjects),
                "epochs": len(epochs_subjects),
                "features": len(features_subjects),
            },
        }
        print(json.dumps(output, indent=INDENT_SIZE))
    else:
        _print_subjects_text(
            bids_subjects,
            epochs_subjects,
            features_subjects,
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
        features_dir = _resolve_features_dir(deriv_root, subject)
        if features_dir is None:
            continue

        subject_stats = {
            "subject": subject,
            "files": 0,
            "total_features": 0,
            "categories": [],
        }

        for feat_file in list(features_dir.rglob("features_*.tsv")) + list(
            features_dir.rglob("features_*.parquet")
        ):
            if not feat_file.is_file():
                continue
            try:
                if feat_file.suffix == ".tsv":
                    dataframe = pd.read_csv(feat_file, sep="\t", nrows=1)
                else:
                    dataframe = pd.read_parquet(feat_file)
                    dataframe = dataframe.head(1)
                feature_count = _count_feature_columns(dataframe)
                subject_stats["files"] += 1
                subject_stats["total_features"] += feature_count

                category = _extract_category_from_filename(feat_file.stem)
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
        features_dir = _resolve_features_dir(deriv_root, subject)
        if features_dir is not None:
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
        features_dir = _resolve_features_dir(deriv_root, subject)
        if features_dir is None:
            continue

        tsv_files = list(features_dir.rglob("*.tsv")) + list(
            features_dir.rglob("*.parquet")
        )
        tsv_files = [f for f in tsv_files if f.is_file()]
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

    bids_subjects, epochs_subjects, features_subjects, eeg_prep_subjects, fmri_prep_subjects = _collect_all_subjects(
        deriv_root, task, config
    )

    fmri_first_level, fmri_beta_series, fmri_lss = _collect_fmri_analysis_subjects(deriv_root)

    if args.mode == "summary":
        _handle_summary_mode(
            args,
            task,
            deriv_root,
            bids_subjects,
            epochs_subjects,
            features_subjects,
            eeg_prep_subjects,
            fmri_prep_subjects,
            fmri_first_level,
            fmri_beta_series,
            fmri_lss,
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
