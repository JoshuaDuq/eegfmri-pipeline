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

SOURCE_BIDS = "bids"
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
        choices=[MODE_SUBJECTS, MODE_FEATURES, MODE_CONFIG, MODE_VERSION, MODE_PLOTTERS],
        help="What to show: subjects, features for a subject, config summary, or version",
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
        choices=[SOURCE_BIDS, SOURCE_EPOCHS, SOURCE_FEATURES, SOURCE_SOURCE_DATA, SOURCE_ALL],
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
    """Extract available time windows from extraction config or fallback to static config."""
    if not features_dir.exists():
        return []

    extraction_config_path = features_dir / EXTRACTION_CONFIG_FILENAME
    if extraction_config_path.exists():
        try:
            extraction_meta = json_module.loads(extraction_config_path.read_text())
            if extraction_meta.get("merged") and extraction_meta.get("time_ranges"):
                windows = [w for w in extraction_meta["time_ranges"] if w is not None]
                if windows:
                    return windows
        except (json_module.JSONDecodeError, OSError, KeyError):
            pass

    try:
        suffixed_configs = list(features_dir.glob(EXTRACTION_CONFIG_PATTERN))
        windows = []
        for cfg_path in suffixed_configs:
            stem = cfg_path.stem
            if stem.startswith("extraction_config_"):
                window_name = stem.replace("extraction_config_", "")
                if window_name and window_name not in windows:
                    windows.append(window_name)
        if windows:
            return sorted(windows)
    except OSError:
        pass

    return list(config.get("time_windows", {}).keys())


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


def _build_subject_status_json(
    discovered_subjects: List[str],
    epochs_subjects: set,
    features_subjects: set,
    deriv_root: Path,
    task: str,
    config: Any,
) -> dict:
    """Build JSON output for subject status mode."""
    from eeg_pipeline.utils.data.subjects import get_epoch_metadata
    from eeg_pipeline.infra.paths import deriv_features_path, deriv_stats_path

    results = []
    for subj in discovered_subjects:
        status = {
            "subject": f"sub-{subj}",
            "epochs": subj in epochs_subjects,
            "features": subj in features_subjects,
        }
        results.append(status)

    global_epoch_metadata = {}
    for subj in discovered_subjects:
        if subj in epochs_subjects:
            global_epoch_metadata = get_epoch_metadata(subj, task, deriv_root, config=config)
            if global_epoch_metadata:
                break

    available_windows = []
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        features_dir = deriv_features_path(deriv_root, subj_id)
        windows = _get_available_time_windows(features_dir, config)
        if windows:
            available_windows = windows
            break

    available_columns = []
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        columns = _get_available_event_columns(config.bids_root, subj_id, task)
        if columns:
            available_columns = columns
            break

    json_results = []
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        available_bands = []
        has_stats = False

        if result["features"] or result["epochs"]:
            features_dir = deriv_features_path(deriv_root, subj_id)
            feature_availability = detect_feature_availability(features_dir)
            if features_dir.exists():
                available_bands = detect_available_bands(features_dir)
        else:
            feature_availability = _empty_feature_availability()

        stats_dir = deriv_stats_path(deriv_root, subj_id)
        if stats_dir.exists():
            for pattern in STATS_FILE_PATTERNS:
                if any(stats_dir.glob(pattern)):
                    has_stats = True
                    break

        metadata = {}
        if result["epochs"]:
            metadata = get_epoch_metadata(subj_id, task, deriv_root, config=config)
            if not metadata and global_epoch_metadata:
                metadata = global_epoch_metadata

        json_results.append({
            "id": subj_id,
            "has_epochs": result["epochs"],
            "has_features": result["features"],
            "has_stats": has_stats,
            "epoch_metadata": metadata,
            "available_bands": available_bands,
            "feature_availability": feature_availability,
        })

    return {
        "subjects": json_results,
        "count": len(json_results),
        "available_windows": available_windows,
        "available_event_columns": available_columns,
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

    discovered = get_available_subjects(
        config=config,
        deriv_root=deriv_root,
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

    feature_files = sorted(features_dir.glob(FEATURES_FILE_PATTERN))
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


def run_info(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the info command."""
    from eeg_pipeline.infra.paths import resolve_deriv_root

    logger = _get_logger(args.output_json)
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)

    if args.mode == MODE_PLOTTERS:
        _handle_plotters_mode(args.output_json)
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
