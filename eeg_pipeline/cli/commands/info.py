"""Info and discovery CLI command."""

from __future__ import annotations

import argparse
import json as json_module
import logging
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import add_task_arg, resolve_task
from eeg_pipeline.cli.commands.base import (
    detect_available_bands,
    detect_feature_availability,
    _empty_feature_availability,
)


MODE_SUBJECTS = "subjects"
MODE_FEATURES = "features"
MODE_CONFIG = "config"
MODE_VERSION = "version"
MODE_PLOTTERS = "plotters"
MODE_DISCOVER = "discover"
MODE_FMRI_CONDITIONS = "fmri-conditions"
MODE_FMRI_COLUMNS = "fmri-columns"

SOURCE_BIDS = "bids"
SOURCE_BIDS_FMRI = "bids_fmri"
SOURCE_EPOCHS = "epochs"
SOURCE_FEATURES = "features"
SOURCE_SOURCE_DATA = "source_data"
SOURCE_ALL = "all"

DISCOVERY_SOURCE_BIDS = "bids"
DISCOVERY_SOURCE_DERIVATIVES_EPOCHS = "derivatives_epochs"
DISCOVERY_SOURCE_FEATURES = "features"
DISCOVERY_SOURCE_SOURCE_DATA = "source_data"

EXTRACTION_CONFIG_FILENAME = "extraction_config.json"
EXTRACTION_CONFIG_PATTERN = "extraction_config_*.json"
FEATURES_FILE_PATTERN = "features_*.tsv"
STATS_FILE_PATTERNS = ("*.tsv", "*.npz", "*.csv", "*.json")

# Patterns to detect EEG preprocessing (ICA files from MNE-BIDS pipeline)
PREPROCESSING_EEG_PATTERNS = ("*ica.fif", "*_components.tsv")


def _get_available_channels(bids_root: Path, subject_id: str) -> List[str]:
    """Read available EEG channel names from BIDS electrodes.tsv.

    Looks for electrodes.tsv in `bids_root/sub-{subject_id}/eeg/`.
    Falls back to the first available subject if the specified subject has no file.
    """
    import pandas as pd

    pattern = f"sub-{subject_id}_*electrodes.tsv"
    sub_eeg_dir = bids_root / f"sub-{subject_id}" / "eeg"

    electrode_files = list(sub_eeg_dir.glob(pattern)) if sub_eeg_dir.exists() else []

    if not electrode_files:
        for sub_dir in sorted(bids_root.glob("sub-*")):
            eeg_dir = sub_dir / "eeg"
            if eeg_dir.exists():
                files = list(eeg_dir.glob("*electrodes.tsv"))
                if files:
                    electrode_files = files
                    break

    if not electrode_files:
        return []

    try:
        df = pd.read_csv(electrode_files[0], sep="\t")
        if "name" not in df.columns:
            return []
        names = df["name"].dropna().astype(str).tolist()
        return [n for n in names if n.lower() not in ("n/a", "nan", "ecg", "eog")]
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return []


def _get_unavailable_channels(deriv_root: Path, task: str) -> List[str]:
    """Extract bad channels from pyprep preprocessing log.

    Reads from `deriv_root/preprocessed/pyprep_task_{task}_log.csv` and
    aggregates the union of all bad channels across runs.
    """
    import ast
    import pandas as pd

    log_path = deriv_root / "preprocessed" / f"pyprep_task_{task}_log.csv"
    if not log_path.exists():
        return []

    try:
        df = pd.read_csv(log_path)
        if "bad_channels" not in df.columns:
            return []

        bad_set: set[str] = set()
        for val in df["bad_channels"].dropna():
            try:
                ch_list = ast.literal_eval(str(val))
                if isinstance(ch_list, list):
                    bad_set.update(ch_list)
            except (ValueError, SyntaxError):
                pass
        return sorted(bad_set)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return []


def setup_info(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the info command parser."""
    parser = subparsers.add_parser(
        "info",
        help="Discovery and status: list subjects, features, config",
        description="Show information about subjects, features, and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=[MODE_SUBJECTS, MODE_FEATURES, MODE_CONFIG, MODE_VERSION, MODE_PLOTTERS, MODE_DISCOVER, MODE_FMRI_CONDITIONS, MODE_FMRI_COLUMNS],
        help="What to show: subjects, features, config, version, discover columns, fmri-conditions, or fmri-columns",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Subject ID (for features mode) or config key (for config mode)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show processing status for each subject (subjects mode only)",
    )
    parser.add_argument(
        "--source",
        choices=[SOURCE_BIDS, SOURCE_BIDS_FMRI, SOURCE_EPOCHS, SOURCE_FEATURES, SOURCE_SOURCE_DATA, SOURCE_ALL],
        default=SOURCE_EPOCHS,
        help="Discovery source for subjects (default: epochs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Config keys to fetch (config mode only)",
    )

    discover_group = parser.add_argument_group("Discover options (mode: discover)")
    discover_group.add_argument(
        "--discover-source",
        choices=["events", "trial-table", "all"],
        default="all",
        help="Where to discover columns from (default: all)",
    )
    discover_group.add_argument(
        "--subject",
        default=None,
        help="Specific subject to scan (default: auto-detect from first available)",
    )
    discover_group.add_argument(
        "--column",
        default=None,
        help="Get values for a specific column only",
    )

    return parser


def _configure_logging_for_json_output() -> logging.Logger:
    """Configure logging to suppress output when JSON is requested."""
    logging.getLogger("mne").setLevel(logging.ERROR)
    logging.getLogger("nilearn").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)
    return logger


def _get_logger(output_json: bool) -> logging.Logger:
    """Get logger, suppressing output if JSON mode is enabled."""
    if output_json:
        return _configure_logging_for_json_output()
    from eeg_pipeline.infra.logging import get_logger
    return get_logger(__name__)


def _map_source_to_discovery_sources(source: str) -> List[str]:
    """Map source argument to list of discovery source names."""
    if source == SOURCE_ALL:
        return [
            DISCOVERY_SOURCE_BIDS,
            DISCOVERY_SOURCE_DERIVATIVES_EPOCHS,
            DISCOVERY_SOURCE_FEATURES,
            DISCOVERY_SOURCE_SOURCE_DATA,
        ]
    if source == SOURCE_BIDS:
        return [DISCOVERY_SOURCE_BIDS]
    if source == SOURCE_BIDS_FMRI:
        return [DISCOVERY_SOURCE_BIDS]
    if source == SOURCE_FEATURES:
        return [DISCOVERY_SOURCE_FEATURES]
    if source == SOURCE_SOURCE_DATA:
        return [DISCOVERY_SOURCE_SOURCE_DATA]
    return [DISCOVERY_SOURCE_DERIVATIVES_EPOCHS]


def _get_discovery_policy(source: str) -> str:
    """Get discovery policy based on source."""
    return "union" if source == SOURCE_ALL else "intersection"


def _extract_subject_id(subject_string: str) -> str:
    """Extract subject ID from subject string (removes 'sub-' prefix if present)."""
    return subject_string.replace("sub-", "")


def _print_json_output(data: dict) -> None:
    """Print data as formatted JSON."""
    print(json_module.dumps(data, indent=2))


def _handle_plotters_mode(output_json: bool) -> None:
    """Handle plotters mode: list available feature plotters."""
    from eeg_pipeline.plotting.features import registrations as _feature_plotters  # noqa: F401
    from eeg_pipeline.plotting.features.context import VisualizationRegistry

    feature_plotters = {}
    for category in sorted(VisualizationRegistry.get_categories()):
        plotters = []
        for name, _func in VisualizationRegistry.get_plotters(category):
            plotters.append(
                {
                    "id": f"{category}.{name}",
                    "category": category,
                    "name": name,
                }
            )
        feature_plotters[category] = plotters

    if output_json:
        _print_json_output({"feature_plotters": feature_plotters})
    else:
        for category, plotters in feature_plotters.items():
            print(f"{category}:")
            for plotter in plotters:
                print(f"  - {plotter['name']}")


def _get_available_time_windows(features_dir: Path, config: Any) -> List[str]:
    """Extract available time windows by scanning window-specific feature files.
    
    Detects windows from filenames matching pattern:
    features/{category}/features_{category}_{window}.{tsv,parquet}
    """
    if not features_dir.exists():
        return []

    windows = set()
    
    try:
        # Check window-specific files (both .tsv and .parquet) in all subdirectories
        for ext in ["tsv", "parquet"]:
            for fpath in features_dir.rglob(f"*/features_*.{ext}"):
                category = fpath.parent.name
                stem = fpath.stem
                prefix = f"features_{category}_"
                
                if stem.startswith(prefix):
                    window = stem[len(prefix):]
                    if window:
                        windows.add(window)
    except OSError:
        pass

    return sorted(windows)


def _get_available_event_columns(bids_root: Path, subject_id: str, task: str) -> List[str]:
    """Get available event columns from BIDS events file."""
    from eeg_pipeline.infra.paths import bids_events_path
    import pandas as pd

    events_path = bids_events_path(bids_root, subject_id, task)
    if not events_path.exists():
        return []

    try:
        df = pd.read_csv(events_path, sep="\t", nrows=1)
        return sorted(list(df.columns))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return []


def _process_single_subject(
    subj_id: str,
    has_epochs: bool,
    has_features: bool,
    deriv_root: Path,
    task: str,
    config: Any,
    global_epoch_metadata: dict,
) -> dict:
    """Process a single subject to build status information."""
    from eeg_pipeline.utils.data.subjects import get_epoch_metadata
    from eeg_pipeline.infra.paths import deriv_features_path, deriv_stats_path
    
    available_bands = []
    has_stats = False
    has_preprocessing = False

    if has_features or has_epochs:
        features_dir = deriv_features_path(deriv_root, subj_id)
        feature_availability = detect_feature_availability(features_dir)
        if features_dir.exists():
            available_bands = detect_available_bands(features_dir)
    else:
        feature_availability = _empty_feature_availability()

    # Check for EEG preprocessing (ICA files in preprocessed directory)
    eeg_prep_dir = deriv_root / "preprocessed" / f"sub-{subj_id}" / "eeg"
    if eeg_prep_dir.exists():
        for pattern in PREPROCESSING_EEG_PATTERNS:
            if any(eeg_prep_dir.glob(pattern)):
                has_preprocessing = True
                break

    stats_dir = deriv_stats_path(deriv_root, subj_id)
    if stats_dir.exists():
        for pattern in STATS_FILE_PATTERNS:
            if any(stats_dir.glob(pattern)):
                has_stats = True
                break

    metadata = {}
    if has_epochs:
        metadata = get_epoch_metadata(subj_id, task, deriv_root, config=config)
        if not metadata and global_epoch_metadata:
            metadata = global_epoch_metadata

    return {
        "id": subj_id,
        "has_epochs": has_epochs,
        "has_preprocessing": has_preprocessing,
        "has_features": has_features,
        "has_stats": has_stats,
        "epoch_metadata": metadata,
        "available_bands": available_bands,
        "feature_availability": feature_availability,
    }


def _build_subject_status_json(
    discovered_subjects: List[str],
    epochs_subjects: set,
    features_subjects: set,
    deriv_root: Path,
    task: str,
    config: Any,
) -> dict:
    """Build JSON output for subject status mode."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from eeg_pipeline.utils.data.subjects import get_epoch_metadata
    from eeg_pipeline.infra.paths import deriv_features_path

    results = []
    for subj in discovered_subjects:
        status = {
            "subject": f"sub-{subj}",
            "epochs": subj in epochs_subjects,
            "features": subj in features_subjects,
        }
        results.append(status)

    # Get global epoch metadata from first subject with epochs
    global_epoch_metadata = {}
    for subj in discovered_subjects:
        if subj in epochs_subjects:
            global_epoch_metadata = get_epoch_metadata(subj, task, deriv_root, config=config)
            if global_epoch_metadata:
                break

    # Get available windows from first subject with features
    available_windows = []
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        features_dir = deriv_features_path(deriv_root, subj_id)
        windows = _get_available_time_windows(features_dir, config)
        if windows:
            available_windows = windows
            break

    # Get available columns from first subject
    available_columns = []
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        columns = _get_available_event_columns(config.bids_root, subj_id, task)
        if columns:
            available_columns = columns
            break

    # Process subjects in parallel (I/O bound operations)
    json_results = []
    with ThreadPoolExecutor(max_workers=min(8, len(results))) as executor:
        futures = []
        for result in results:
            subj_id = _extract_subject_id(result["subject"])
            future = executor.submit(
                _process_single_subject,
                subj_id,
                result["epochs"],
                result["features"],
                deriv_root,
                task,
                config,
                global_epoch_metadata,
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                json_results.append(future.result())
            except Exception:
                # If processing fails for a subject, skip it
                pass
    
    # Sort results to maintain consistent order
    json_results.sort(key=lambda x: x["id"])

    # Discover available and unavailable channels (global for study)
    bids_root = Path(config.bids_root) if hasattr(config, "bids_root") else None
    first_subject = discovered_subjects[0] if discovered_subjects else ""
    available_channels = _get_available_channels(bids_root, first_subject) if bids_root else []
    unavailable_channels = _get_unavailable_channels(deriv_root, task)

    return {
        "subjects": json_results,
        "count": len(json_results),
        "available_windows": available_windows,
        "available_event_columns": available_columns,
        "available_channels": available_channels,
        "unavailable_channels": unavailable_channels,
    }


def _handle_subjects_mode(
    args: argparse.Namespace,
    deriv_root: Path,
    task: str,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Handle subjects mode: list available subjects."""
    from eeg_pipeline.utils.data.subjects import (
        get_available_subjects,
        _collect_subjects_from_bids,
        _collect_subjects_from_derivatives_epochs,
        _collect_subjects_from_features,
    )

    sources = _map_source_to_discovery_sources(args.source)
    policy = _get_discovery_policy(args.source)

    bids_root_override = None
    if args.source == SOURCE_BIDS_FMRI:
        bids_fmri_root = config.get("paths.bids_fmri_root")
        if bids_fmri_root:
            bids_root_override = Path(str(bids_fmri_root))

    discovered = get_available_subjects(
        config=config,
        deriv_root=deriv_root,
        bids_root=bids_root_override,
        task=task,
        discovery_sources=sources,
        subject_discovery_policy=policy,
        logger=logger,
    )

    if args.status:
        epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
        features_subjects = set(_collect_subjects_from_features(deriv_root))

        if args.output_json:
            output = _build_subject_status_json(
                discovered,
                epochs_subjects,
                features_subjects,
                deriv_root,
                task,
                config,
            )
            _print_json_output(output)
        else:
            for subj in discovered:
                epoch_mark = "x" if subj in epochs_subjects else " "
                feat_mark = "x" if subj in features_subjects else " "
                print(f"sub-{subj}  [{epoch_mark}]epochs  [{feat_mark}]features")
            print(f"\nTotal: {len(discovered)} subjects")
    else:
        if args.output_json:
            json_results = [
                {"id": subj_id, "has_epochs": False, "has_features": False}
                for subj_id in discovered
            ]
            _print_json_output({"subjects": json_results, "count": len(discovered)})
        else:
            for subj in discovered:
                print(f"sub-{subj}")
            print(f"\nTotal: {len(discovered)} subjects")


def _handle_features_mode(
    subject_id: str,
    deriv_root: Path,
    output_json: bool,
) -> None:
    """Handle features mode: list features for a subject."""
    from eeg_pipeline.infra.paths import deriv_features_path
    import pandas as pd

    features_dir = deriv_features_path(deriv_root, subject_id)

    if not features_dir.exists():
        print(f"No features directory found for sub-{subject_id}")
        return

    # Only look in subfolders for new organized structure
    feature_files = sorted(features_dir.glob("*/" + FEATURES_FILE_PATTERN))
    results = []

    for fpath in feature_files:
        try:
            df = pd.read_csv(fpath, sep="\t", nrows=1)
            results.append({"file": fpath.name, "columns": len(df.columns)})
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            results.append({"file": fpath.name, "columns": "error"})

    if output_json:
        _print_json_output({"subject": f"sub-{subject_id}", "features": results})
    else:
        print(f"Features for sub-{subject_id}:")
        for result in results:
            print(f"  {result['file']}: {result['columns']} columns")
        print(f"\nTotal: {len(results)} feature files")


def _handle_config_mode(args: argparse.Namespace, config: Any, deriv_root: Path, task: str) -> None:
    """Handle config mode: show configuration values."""
    if args.keys:
        values = {key: config.get(key) for key in args.keys}
        if args.output_json:
            _print_json_output(values)
        else:
            for key, value in values.items():
                print(f"{key}: {value}")
    elif args.target:
        value = config.get(args.target)
        if args.output_json:
            _print_json_output({args.target: value})
        else:
            print(f"{args.target}: {value}")
    else:
        summary = {
            "bids_root": str(config.bids_root) if hasattr(config, "bids_root") else None,
            "bids_fmri_root": config.get("paths.bids_fmri_root"),
            "deriv_root": str(deriv_root),
            "source_root": config.get("paths.source_data"),
            "task": task,
            "preprocessing_n_jobs": config.get("preprocessing.n_jobs", 1),
            "n_subjects": len(config.subjects) if hasattr(config, "subjects") and config.subjects else 0,
        }
        if args.output_json:
            _print_json_output(summary)
        else:
            print("Configuration Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")


def _handle_version_mode(output_json: bool) -> None:
    """Handle version mode: show package version."""
    import eeg_pipeline

    version = getattr(eeg_pipeline, "__version__", "unknown")
    if output_json:
        _print_json_output({"version": version})
    else:
        print(f"eeg-pipeline version: {version}")


def _print_discovery_report(result: dict) -> None:
    """Pretty-print discovery results to stdout."""
    print("=" * 50)
    print("       COLUMN DISCOVERY REPORT")
    print("=" * 50)
    print()

    if not result["columns"]:
        print("  No columns discovered.")
        print("  Make sure you have events files in your BIDS directory")
        print("  or have run behavior compute to create trial tables.")
        return

    print(f"  Source: {result.get('source', 'unknown')}")
    if result.get("file"):
        print(f"  File: {result['file']}")
    print()

    print("  AVAILABLE COLUMNS")
    print("  " + "-" * 30)
    for col in result["columns"]:
        has_values = col in result["values"]
        val_indicator = (
            f" ({len(result['values'][col])} values)" if has_values else ""
        )
        print(f"    • {col}{val_indicator}")
    print()

    if result["values"]:
        print("  COLUMN VALUES")
        print("  " + "-" * 30)
        for col, vals in sorted(result["values"].items()):
            if len(vals) <= 10:
                vals_str = ", ".join(str(v) for v in vals)
            else:
                vals_str = (
                    ", ".join(str(v) for v in vals[:8])
                    + f", ... (+{len(vals) - 8} more)"
                )
            print(f"    {col}: {vals_str}")
    print()


def _handle_fmri_conditions_mode(args: argparse.Namespace, config: Any) -> None:
    """Handle fmri-conditions mode: discover available trial_type conditions from fMRI events files."""
    from eeg_pipeline.analysis.features.fmri_contrast_builder import discover_available_conditions

    task = resolve_task(args.task, config)
    
    # Get fMRI root from config
    fmri_root = config.get("paths.bids_fmri_root")
    if not fmri_root:
        # Fallback to default location
        fmri_root = config.get("paths.bids_root", "").replace("bids_output", "fMRI_data")
    
    fmri_root = Path(fmri_root)
    if not fmri_root.is_absolute():
        # Make relative to project root
        from eeg_pipeline.utils.config.loader import get_project_root
        fmri_root = get_project_root() / fmri_root

    subject = args.subject
    if not subject:
        # Auto-detect first available subject
        if fmri_root.exists():
            for sub_dir in sorted(fmri_root.glob("sub-*")):
                if sub_dir.is_dir():
                    subject = sub_dir.name.replace("sub-", "")
                    break

    if not subject:
        result = {
            "conditions": [],
            "subject": None,
            "task": task,
            "error": "No subject specified and none found in fMRI directory",
        }
        if args.output_json:
            _print_json_output(result)
        else:
            print("Error: No subject specified and none found in fMRI directory")
        return

    # Map task name (thermalactive -> pain for fMRI)
    fmri_task = task.replace("thermal", "pain").replace("active", "")
    if not fmri_task:
        fmri_task = "pain"

    try:
        conditions = discover_available_conditions(fmri_root, subject, fmri_task)
    except Exception as e:
        conditions = []
        error_msg = str(e)
    else:
        error_msg = None

    result = {
        "conditions": conditions,
        "subject": subject,
        "task": fmri_task,
    }
    if error_msg:
        result["error"] = error_msg

    if args.output_json:
        _print_json_output(result)
    else:
        if conditions:
            print(f"Available fMRI conditions for sub-{subject}, task-{fmri_task}:")
            for cond in conditions:
                print(f"  - {cond}")
            print(f"\nTotal: {len(conditions)} conditions")
        else:
            print(f"No conditions found for sub-{subject}, task-{fmri_task}")
            if error_msg:
                print(f"Error: {error_msg}")


def _handle_discover_mode(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Handle discover mode: discover available columns and values from data files."""
    from eeg_pipeline.cli.commands.base import (
        discover_event_columns,
        discover_trial_table_columns,
    )
    from eeg_pipeline.infra.paths import resolve_deriv_root

    task = resolve_task(args.task, config)
    bids_root = Path(config.bids_root) if hasattr(config, "bids_root") else None
    deriv_root = resolve_deriv_root(config=config)

    result = {
        "columns": [],
        "values": {},
        "source": None,
        "sources_checked": [],
    }

    subject = args.subject
    if not subject and subjects:
        subject = subjects[0]

    if args.discover_source in ["events", "all"] and bids_root:
        events_data = discover_event_columns(bids_root, task=task, subject=subject)
        if events_data["columns"]:
            result["sources_checked"].append("events")
            if not result["columns"]:
                result["columns"] = events_data["columns"]
                result["values"] = events_data["values"]
                result["source"] = events_data["source"]
                result["file"] = events_data.get("file")
            else:
                for col, vals in events_data["values"].items():
                    if col not in result["values"]:
                        result["values"][col] = vals

    if args.discover_source in ["trial-table", "all"]:
        trial_data = discover_trial_table_columns(deriv_root, subject=subject)
        if trial_data["columns"]:
            result["sources_checked"].append("trial_table")
            if not result["columns"] or args.discover_source == "trial-table":
                result["columns"] = trial_data["columns"]
                result["values"] = trial_data["values"]
                result["source"] = trial_data["source"]
                result["file"] = trial_data.get("file")
            else:
                for col, vals in trial_data["values"].items():
                    if col not in result["values"]:
                        result["values"][col] = vals

    if args.column:
        if args.column in result["values"]:
            result["values"] = {args.column: result["values"][args.column]}
        else:
            result["values"] = {}

    if args.output_json:
        _print_json_output(result)
    else:
        _print_discovery_report(result)


def _handle_fmri_columns_mode(args: argparse.Namespace, config: Any) -> None:
    """Handle fmri-columns mode: discover columns from fMRI events files."""
    from eeg_pipeline.cli.commands.base import discover_fmri_event_columns

    task = resolve_task(args.task, config)

    fmri_root = config.get("paths.bids_fmri_root")
    if not fmri_root:
        fmri_root = config.get("paths.bids_root", "").replace("bids_output", "fMRI_data")

    fmri_root = Path(fmri_root)
    if not fmri_root.is_absolute():
        from eeg_pipeline.utils.config.loader import get_project_root
        fmri_root = get_project_root() / fmri_root

    fmri_task = task.replace("thermal", "pain").replace("active", "")
    if not fmri_task:
        fmri_task = "pain"

    subject = args.subject
    if not subject and fmri_root.exists():
        for sub_dir in sorted(fmri_root.glob("sub-*")):
            if sub_dir.is_dir():
                subject = sub_dir.name.replace("sub-", "")
                break

    result = discover_fmri_event_columns(fmri_root, task=fmri_task, subject=subject)

    if args.column:
        if args.column in result["values"]:
            result["values"] = {args.column: result["values"][args.column]}
        else:
            result["values"] = {}

    if args.output_json:
        _print_json_output(result)
    else:
        _print_discovery_report(result)


def run_info(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the info command."""
    from eeg_pipeline.infra.paths import resolve_deriv_root

    logger = _get_logger(args.output_json)
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)

    if args.mode == MODE_PLOTTERS:
        _handle_plotters_mode(args.output_json)
    elif args.mode == MODE_FMRI_CONDITIONS:
        _handle_fmri_conditions_mode(args, config)
    elif args.mode == MODE_FMRI_COLUMNS:
        _handle_fmri_columns_mode(args, config)
    elif args.mode == MODE_DISCOVER:
        _handle_discover_mode(args, subjects, config)
    elif args.mode == MODE_SUBJECTS:
        _handle_subjects_mode(args, deriv_root, task, config, logger)
    elif args.mode == MODE_FEATURES:
        if not args.target:
            print("Error: subject ID required for features mode")
            print("Usage: eeg-pipeline info features <SUBJECT_ID>")
            return
        subject_id = _extract_subject_id(args.target)
        _handle_features_mode(subject_id, deriv_root, args.output_json)
    elif args.mode == MODE_CONFIG:
        _handle_config_mode(args, config, deriv_root, task)
    elif args.mode == MODE_VERSION:
        _handle_version_mode(args.output_json)
