"""Helper functions for info and discovery CLI command."""

from __future__ import annotations

import argparse
import json as json_module
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.cli.commands.base import (
    detect_available_bands,
    detect_feature_availability,
    _empty_feature_availability,
)
from eeg_pipeline.cli.common import resolve_task

MODE_SUBJECTS = "subjects"
MODE_FEATURES = "features"
MODE_CONFIG = "config"
MODE_VERSION = "version"
MODE_PLOTTERS = "plotters"
MODE_DISCOVER = "discover"
MODE_ROIS = "rois"
MODE_FMRI_CONDITIONS = "fmri-conditions"
MODE_FMRI_COLUMNS = "fmri-columns"
MODE_MULTIGROUP_STATS = "multigroup-stats"
MODE_ML_FEATURE_SPACE = "ml-feature-space"

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
    logger = logging.getLogger(__name__)

    log_path = deriv_root / "preprocessed" / "eeg" / f"pyprep_task_{task}_log.csv"
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
            except (ValueError, SyntaxError) as exc:
                logger.debug("Failed to parse bad_channels entry %r from %s: %s", val, log_path, exc)
        return sorted(bad_set)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return []


def _handle_ml_feature_space_mode(args: argparse.Namespace, config: Any, task: str) -> None:
    """Discover available (band, segment, scope) values from feature columns."""
    import pyarrow.parquet as pq

    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.infra.paths import resolve_deriv_root

    deriv_root = resolve_deriv_root(config=config)

    def _as_list(val: Any) -> Optional[List[str]]:
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            out = [str(v).strip() for v in val if str(v).strip()]
            return out or None
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return None

    families = _as_list(args.feature_families) or _as_list(config.get("machine_learning.data.feature_families")) or None
    if not families:
        # Broad default consistent with ML loader fallback.
        families = ["power"]

    subjects = _as_list(args.subjects)
    if not subjects:
        max_n = max(1, int(args.max_subjects or 3))
        detected = []
        for sub_dir in sorted(Path(deriv_root).glob("sub-*")):
            sid = _extract_subject_id(sub_dir.name)
            features_dir = Path(deriv_root) / f"sub-{sid}" / "eeg" / "features"
            if features_dir.exists():
                detected.append(sid)
            if len(detected) >= max_n:
                break
        subjects = detected

    bands: set[str] = set()
    segments: set[str] = set()
    scopes: set[str] = set()

    for subject in subjects or []:
        sid = _extract_subject_id(subject)
        features_dir = Path(deriv_root) / f"sub-{sid}" / "eeg" / "features"
        if not features_dir.exists():
            continue

        for fam in families:
            fname_override = config.get(f"machine_learning.data.feature_files.{fam}")
            filename = str(fname_override).strip() if isinstance(fname_override, str) and str(fname_override).strip() else f"features_{fam}.parquet"

            if str(fam).startswith("pac"):
                path = features_dir / "pac" / filename
            elif str(fam) in {"sourcelocalization", "source_localization"} or str(fam).startswith("sourcelocalization"):
                # Mirror ML loader behavior: sourcelocalization may live in nested subfolders.
                candidates = [
                    features_dir / "sourcelocalization" / "fmri_informed" / filename,
                    features_dir / "sourcelocalization" / "eeg_only" / filename,
                    features_dir / "sourcelocalization" / filename,
                ]
                path = None
                for cand in candidates:
                    if cand.exists():
                        path = cand
                        break
                if path is None:
                    path = candidates[0]
            else:
                # Default: features are stored in per-family subfolders.
                path = features_dir / str(fam) / filename

            if not path.exists():
                continue

            try:
                cols = pq.ParquetFile(path).schema_arrow.names
            except Exception:
                continue

            for col in cols:
                parsed = NamingSchema.parse(str(col))
                if not parsed.get("valid", False):
                    continue
                band = parsed.get("band")
                seg = parsed.get("segment")
                scope = parsed.get("scope")
                if band:
                    bands.add(str(band))
                if seg:
                    segments.add(str(seg))
                if scope:
                    scopes.add(str(scope))

    payload = {
        "subjects_scanned": subjects or [],
        "feature_families": families,
        "bands": sorted(bands),
        "segments": sorted(segments),
        "scopes": sorted(scopes),
        "task": task,
        "deriv_root": str(deriv_root),
        "error": "",
    }

    print(json_module.dumps(payload))


def _subjects_cache_dir(deriv_root: Path) -> Path:
    return Path(deriv_root) / ".cache" / "tui"


def _subjects_cache_path(deriv_root: Path, *, task: str, source: str) -> Path:
    safe_task = str(task or "unknown").strip().replace(os.sep, "_")
    safe_source = str(source or "unknown").strip().replace(os.sep, "_")
    return _subjects_cache_dir(deriv_root) / f"subjects_{safe_source}_{safe_task}.json"


def _scan_subject_dir_mtimes(root: Optional[Path]) -> Dict[str, int]:
    """Fast: list top-level sub-* dirs and their mtime_ns."""
    if root is None:
        return {}
    root = Path(root)
    if not root.exists():
        return {}
    out: Dict[str, int] = {}
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith("sub-"):
                continue
            try:
                out[name[4:]] = int(entry.stat().st_mtime_ns)
            except OSError:
                continue
    return out


def _safe_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _build_subjects_cache_stamp(
    *,
    deriv_root: Path,
    bids_root: Optional[Path],
) -> Dict[str, Any]:
    """Conservative, fast cache invalidation stamp (directory mtimes only)."""
    deriv_root = Path(deriv_root)
    bids_root = Path(bids_root) if bids_root is not None else None

    deriv_subject_dirs = _scan_subject_dir_mtimes(deriv_root)
    preproc_dir = deriv_root / "preprocessed" / "eeg"
    preproc_subject_dirs = _scan_subject_dir_mtimes(preproc_dir)

    # Per-subject “hot” directories where outputs are created.
    subjects = sorted(set(deriv_subject_dirs) | set(preproc_subject_dirs))
    per_subject: Dict[str, Dict[str, Optional[int]]] = {}
    for sid in subjects:
        subject_dir = deriv_root / f"sub-{sid}"
        per_subject[sid] = {
            "deriv_sub_mtime_ns": deriv_subject_dirs.get(sid),
            "preproc_sub_mtime_ns": preproc_subject_dirs.get(sid),
            "eeg_dir_mtime_ns": _safe_mtime_ns(subject_dir / "eeg"),
            "features_dir_mtime_ns": _safe_mtime_ns(subject_dir / "eeg" / "features"),
            "stats_dir_mtime_ns": _safe_mtime_ns(subject_dir / "eeg" / "stats"),
            "preprocessed_eeg_dir_mtime_ns": _safe_mtime_ns(preproc_dir / f"sub-{sid}"),
        }

    return {
        "bids_root": str(bids_root) if bids_root is not None else None,
        "deriv_root": str(deriv_root),
        "bids_subject_dirs_mtime_ns": _scan_subject_dir_mtimes(bids_root),
        "deriv_subject_dirs_mtime_ns": deriv_subject_dirs,
        "preproc_subject_dirs_mtime_ns": preproc_subject_dirs,
        "per_subject_dirs_mtime_ns": per_subject,
    }


def _read_subjects_cache(
    *,
    cache_path: Path,
    stamp: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None
    try:
        payload = json_module.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json_module.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != 1:
        return None
    if payload.get("stamp") != stamp:
        return None
    out = payload.get("payload")
    if not isinstance(out, dict):
        return None
    return out


def _write_subjects_cache_atomic(
    *,
    cache_path: Path,
    stamp: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(
            json_module.dumps(
                {"schema_version": 1, "stamp": stamp, "payload": payload},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(cache_path)
    except OSError:
        # Cache should never block subject discovery.
        return


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


def _get_available_time_windows(features_dir: Path, config: Any, feature_group: Optional[str] = None) -> List[str]:
    """Extract available time windows by scanning window-specific feature files and column names.
    
    Detects windows from:
    1. Filenames matching pattern: features/{category}/features_{category}_{window}.{tsv,parquet}
    2. Column names in feature files using NamingSchema (e.g., itpc_plateau_alpha_ch_Fz_val)
    
    Args:
        features_dir: Directory containing feature files
        config: Configuration object
        feature_group: Optional feature group to filter by (e.g., "itpc", "power", "connectivity")
                      If provided, only returns windows/segments for this feature group.
    """
    if not features_dir.exists():
        return []

    windows = set()
    
    # Method 1: Check window-specific files (both .tsv and .parquet) in all subdirectories
    for ext in ["tsv", "parquet"]:
        for fpath in features_dir.rglob(f"*/features_*.{ext}"):
            category = fpath.parent.name
            stem = fpath.stem
            prefix = f"features_{category}_"
            
            if stem.startswith(prefix):
                window = stem[len(prefix):]
                if window:
                    # If filtering by feature group, check if category matches
                    if feature_group is None or category == feature_group:
                        windows.add(window)
    
    # Method 2: Scan feature files and extract segments from column names
    from eeg_pipeline.domain.features.naming import NamingSchema
    import pandas as pd
    import pyarrow.parquet as pq
    
    for fpath in features_dir.rglob("features_*"):
        # Read only column names, not full data
        if fpath.suffix.lower() == ".parquet":
            parquet_file = pq.ParquetFile(fpath)
            columns = parquet_file.schema_arrow.names
        else:
            # For TSV, read just the header
            df = pd.read_csv(fpath, sep="\t", nrows=0)
            columns = df.columns.tolist()
        
        # Extract segments from column names
        for col in columns:
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid"):
                # If filtering by feature group, check if group matches
                if feature_group is not None and parsed.get("group") != feature_group:
                    continue
                segment = parsed.get("segment")
                if segment:
                    windows.add(str(segment))

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
    bids_root: Optional[Path] = None,
    source_root: Optional[Path] = None,
) -> dict:
    """Process a single subject to build status information."""
    from eeg_pipeline.utils.data.subjects import get_epoch_metadata
    from eeg_pipeline.infra.paths import deriv_features_path, deriv_stats_path
    
    available_bands = []
    has_stats = False
    has_preprocessing = False

    # Detect source data
    has_source_data = False
    if source_root:
        source_subj_dir = source_root / f"sub-{subj_id}"
        if not source_subj_dir.exists():
            # Try without sub- prefix
            source_subj_dir = source_root / subj_id
        has_source_data = source_subj_dir.exists() and source_subj_dir.is_dir()

    # Detect BIDS
    has_bids = False
    if bids_root:
        bids_subj_dir = bids_root / f"sub-{subj_id}"
        has_bids = bids_subj_dir.exists() and bids_subj_dir.is_dir()

    # Detect derivatives (any processed data)
    has_derivatives = False
    if deriv_root.exists():
        # Check for epochs
        from eeg_pipeline.infra.paths import find_clean_epochs_path
        epoch_path = find_clean_epochs_path(subj_id, task, deriv_root=deriv_root, config=config)
        if epoch_path and epoch_path.exists():
            has_derivatives = True
        
        # Check for features
        features_dir = deriv_features_path(deriv_root, subj_id)
        if features_dir.exists() and any(features_dir.rglob("features_*")):
            has_derivatives = True
        
        # Check for EEG preprocessing
        eeg_prep_dir = deriv_root / "preprocessed" / "eeg" / f"sub-{subj_id}"
        if eeg_prep_dir.exists():
            for pattern in PREPROCESSING_EEG_PATTERNS:
                if any(eeg_prep_dir.glob(pattern)):
                    has_derivatives = True
                    has_preprocessing = True
                    break
        
        # Check for fMRI preprocessing
        fmriprep_dir = deriv_root / "fmriprep" / f"sub-{subj_id}"
        if fmriprep_dir.exists():
            func_dir = fmriprep_dir / "func"
            if func_dir.exists() and any(func_dir.glob("*preproc_bold*")):
                has_derivatives = True
        
        # Check for stats
        stats_dir = deriv_stats_path(deriv_root, subj_id)
        if stats_dir.exists():
            for pattern in STATS_FILE_PATTERNS:
                if any(stats_dir.rglob(pattern)):
                    has_derivatives = True
                    has_stats = True
                    break

    if has_features or has_epochs:
        features_dir = deriv_features_path(deriv_root, subj_id)
        feature_availability = detect_feature_availability(features_dir)
        if features_dir.exists():
            available_bands = detect_available_bands(features_dir)
    else:
        feature_availability = _empty_feature_availability()

    metadata = {}
    if has_epochs:
        metadata = get_epoch_metadata(subj_id, task, deriv_root, config=config)
        if not metadata and global_epoch_metadata:
            metadata = global_epoch_metadata

    return {
        "id": subj_id,
        "has_source_data": has_source_data,
        "has_bids": has_bids,
        "has_derivatives": has_derivatives,
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
    bids_root_override: Optional[Path] = None,
) -> dict:
    """Build JSON output for subject status mode."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from eeg_pipeline.utils.data.subjects import get_epoch_metadata, _resolve_source_root
    from eeg_pipeline.infra.paths import deriv_features_path

    results = []
    for subj in discovered_subjects:
        status = {
            "subject": f"sub-{subj}",
            "epochs": subj in epochs_subjects,
            "features": subj in features_subjects,
        }
        results.append(status)

    # Resolve BIDS root
    bids_root = bids_root_override
    if bids_root is None:
        bids_root = Path(config.bids_root) if hasattr(config, "bids_root") else None

    # Resolve source root
    source_root = _resolve_source_root(config, bids_root)

    # Get global epoch metadata from first subject with epochs
    global_epoch_metadata = {}
    for subj in discovered_subjects:
        if subj in epochs_subjects:
            global_epoch_metadata = get_epoch_metadata(subj, task, deriv_root, config=config)
            if global_epoch_metadata:
                break

    # Get available windows from first subject with features
    available_windows = []
    available_windows_by_feature = {}
    feature_groups = ["itpc", "power", "connectivity", "aperiodic", "pac", "complexity", 
                      "ratios", "asymmetry", "erds", "spectral", "bursts", "erp"]
    
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        features_dir = deriv_features_path(deriv_root, subj_id)
        if not features_dir.exists():
            continue
            
        windows = _get_available_time_windows(features_dir, config)
        if windows:
            available_windows = windows
        
        for feature_group in feature_groups:
            feature_windows = _get_available_time_windows(features_dir, config, feature_group=feature_group)
            if feature_windows:
                available_windows_by_feature[feature_group] = feature_windows
        
        if available_windows:
            break

    # Get available columns from first subject
    available_columns = []
    for result in results:
        subj_id = _extract_subject_id(result["subject"])
        if bids_root:
            columns = _get_available_event_columns(bids_root, subj_id, task)
            if columns:
                available_columns = columns
                break

    # Process subjects in parallel (I/O bound operations)
    json_results = []
    if not results:
        return {
            "subjects": [],
            "count": 0,
            "available_windows": available_windows,
            "available_windows_by_feature": available_windows_by_feature,
            "available_event_columns": available_columns,
            "available_channels": [],
            "unavailable_channels": [],
        }
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
                bids_root,
                source_root,
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
    first_subject = discovered_subjects[0] if discovered_subjects else ""
    available_channels = _get_available_channels(bids_root, first_subject) if bids_root else []
    unavailable_channels = _get_unavailable_channels(deriv_root, task)

    return {
        "subjects": json_results,
        "count": len(json_results),
        "available_windows": available_windows,
        "available_windows_by_feature": available_windows_by_feature,
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
        _collect_subjects_from_derivatives_epochs,
        _collect_subjects_from_features,
    )

    sources = _map_source_to_discovery_sources(args.source)
    policy = _get_discovery_policy(args.source)
    if args.status and args.output_json:
        # TUI subject selection should show newly discovered subjects even when
        # project.subject_list is stale or narrower than on-disk data.
        policy = "union"

    bids_root_override = None
    if args.source == SOURCE_BIDS_FMRI:
        bids_fmri_root = config.get("paths.bids_fmri_root")
        if bids_fmri_root:
            bids_root_override = Path(str(bids_fmri_root))

    # Fast path: serve cached TUI payload without scanning epochs/features trees.
    if args.status and args.output_json and bool(getattr(args, "subjects_cache", False)) and not bool(getattr(args, "subjects_refresh", False)):
        bids_root_for_stamp = bids_root_override if bids_root_override is not None else getattr(config, "bids_root", None)
        cache_path = _subjects_cache_path(deriv_root, task=task, source=args.source)
        stamp = _build_subjects_cache_stamp(deriv_root=deriv_root, bids_root=bids_root_for_stamp)
        cached = _read_subjects_cache(cache_path=cache_path, stamp=stamp)
        if cached is not None:
            _print_json_output(cached)
            return

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
        if args.output_json:
            epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
            features_subjects = set(_collect_subjects_from_features(deriv_root))

            bids_root_for_stamp = bids_root_override if bids_root_override is not None else getattr(config, "bids_root", None)
            cache_enabled = bool(getattr(args, "subjects_cache", False))
            refresh = bool(getattr(args, "subjects_refresh", False))
            cache_path = _subjects_cache_path(deriv_root, task=task, source=args.source)

            if cache_enabled and not refresh:
                stamp = _build_subjects_cache_stamp(deriv_root=deriv_root, bids_root=bids_root_for_stamp)
                cached = _read_subjects_cache(cache_path=cache_path, stamp=stamp)
                if cached is not None:
                    _print_json_output(cached)
                    return

            output = _build_subject_status_json(
                discovered,
                epochs_subjects,
                features_subjects,
                deriv_root,
                task,
                config,
                bids_root_override,
            )
            if cache_enabled:
                stamp = _build_subjects_cache_stamp(deriv_root=deriv_root, bids_root=bids_root_for_stamp)
                _write_subjects_cache_atomic(cache_path=cache_path, stamp=stamp, payload=output)
            _print_json_output(output)
        else:
            epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
            features_subjects = set(_collect_subjects_from_features(deriv_root))
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
    from fmri_pipeline.analysis.contrast_builder import discover_available_conditions

    task = resolve_task(args.task, config)

    fmri_root = config.get("paths.bids_fmri_root")
    if not fmri_root:
        bids = config.get("paths.bids_root", "") or ""
        if bids:
            fmri_root = str(Path(bids).parent / "fmri")
        if not fmri_root:
            fmri_root = bids.replace("bids_output", "fMRI_data") if bids else "."

    fmri_root = Path(fmri_root)
    if not fmri_root.is_absolute():
        from eeg_pipeline.utils.config.loader import get_project_root
        fmri_root = get_project_root() / fmri_root

    subject = args.subject
    if not subject and fmri_root.exists():
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

    task_candidates = [task] if task else []

    conditions = []
    used_task = task or ""
    try:
        for t in task_candidates:
            conditions = discover_available_conditions(fmri_root, subject, t)
            if conditions:
                used_task = t
                break
    except Exception as e:
        conditions = []
        error_msg = str(e)
    else:
        error_msg = None

    result = {"conditions": conditions, "subject": subject, "task": used_task}
    if error_msg:
        result["error"] = error_msg

    if args.output_json:
        _print_json_output(result)
    else:
        if conditions:
            print(f"Available fMRI conditions for sub-{subject}, task-{used_task}:")
            for cond in conditions:
                print(f"  - {cond}")
            print(f"\nTotal: {len(conditions)} conditions")
        else:
            print(f"No conditions found for sub-{subject}, task-{used_task}")
            if error_msg:
                print(f"Error: {error_msg}")


def _handle_discover_mode(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Handle discover mode: discover available columns and values from data files."""
    from eeg_pipeline.cli.commands.base import (
        discover_event_columns,
        discover_trial_table_columns,
        discover_condition_effects_columns,
    )
    from eeg_pipeline.infra.paths import resolve_deriv_root

    task = resolve_task(args.task, config)
    bids_root = Path(config.bids_root) if hasattr(config, "bids_root") else None
    deriv_root = resolve_deriv_root(config=config)

    result = {
        "columns": [],
        "values": {},
        "windows": [],
        "source": None,
        "sources_checked": [],
    }

    subject = args.subject
    if not subject and subjects:
        subject = subjects[0]

    if args.discover_source in ["events", "all"] and bids_root:
        events_data = discover_event_columns(bids_root, task=task, subject=subject, deriv_root=deriv_root)
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

    if args.discover_source in ["condition-effects", "all"]:
        cond_effects_data = discover_condition_effects_columns(deriv_root, subject=subject)
        if cond_effects_data["columns"] or cond_effects_data.get("windows"):
            result["sources_checked"].append("condition_effects")
            if not result["columns"] or args.discover_source == "condition-effects":
                result["columns"] = cond_effects_data["columns"]
                result["values"] = cond_effects_data["values"]
                result["windows"] = cond_effects_data.get("windows", [])
                result["source"] = cond_effects_data["source"]
                result["files"] = cond_effects_data.get("files", [])
            else:
                for col, vals in cond_effects_data["values"].items():
                    if col not in result["values"]:
                        result["values"][col] = vals
                # Merge windows
                if cond_effects_data.get("windows"):
                    existing_windows = set(result.get("windows", []))
                    existing_windows.update(cond_effects_data["windows"])
                    result["windows"] = sorted(existing_windows)

    if args.column:
        if args.column in result["values"]:
            result["values"] = {args.column: result["values"][args.column]}
        else:
            result["values"] = {}

    if args.output_json:
        _print_json_output(result)
    else:
        _print_discovery_report(result)


def _handle_rois_mode(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Handle rois mode: discover available ROIs from feature parquet files."""
    import re
    import pandas as pd
    from eeg_pipeline.infra.paths import resolve_deriv_root, deriv_features_path

    deriv_root = resolve_deriv_root(config=config)

    subject = args.subject
    if not subject and subjects:
        subject = subjects[0]

    if not subject:
        # Try to find first available subject
        for sub_dir in sorted(deriv_root.glob("sub-*")):
            if sub_dir.is_dir():
                subject = sub_dir.name.replace("sub-", "")
                break

    result = {
        "rois": [],
        "subject": subject,
        "source": None,
    }

    if not subject:
        result["error"] = "No subject found"
        if args.output_json:
            _print_json_output(result)
        else:
            print("Error: No subject found for ROI discovery")
        return

    features_dir = deriv_features_path(deriv_root, subject)
    if not features_dir.exists():
        result["error"] = f"Features directory not found: {features_dir}"
        if args.output_json:
            _print_json_output(result)
        else:
            print(f"Error: Features directory not found: {features_dir}")
        return

    # Search for parquet files in category subdirectories
    rois = set()
    source_file = None

    for category_dir in sorted(features_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        for fpath in category_dir.glob("features_*.parquet"):
            if source_file is None:
                source_file = str(fpath)

            try:
                df = pd.read_parquet(fpath)
                roi_cols = [c for c in df.columns if "_roi_" in c]

                for col in roi_cols:
                    # Pattern: ..._roi_ROIName_metric_...
                    # ROI names are like: Frontal, Sensorimotor_Left, ParOccipital_Midline, Midline_FrontalCentral
                    match = re.search(r"_roi_([A-Za-z]+(?:_(?:Left|Right|Midline|FrontalCentral))?)_", col)
                    if match:
                        rois.add(match.group(1))
            except Exception:
                continue

    result["rois"] = sorted(rois)
    result["source"] = source_file

    if args.output_json:
        _print_json_output(result)
    else:
        if rois:
            print(f"Available ROIs for sub-{subject}:")
            for roi in sorted(rois):
                print(f"  - {roi}")
            print(f"\nTotal: {len(rois)} ROIs")
        else:
            print(f"No ROIs found for sub-{subject}")


def _handle_fmri_columns_mode(args: argparse.Namespace, config: Any) -> None:
    """Handle fmri-columns mode: discover columns from fMRI events files."""
    task = resolve_task(args.task, config)

    fmri_root = config.get("paths.bids_fmri_root")
    if not fmri_root:
        bids = config.get("paths.bids_root", "") or ""
        if bids:
            # Try bids_output/fmri when bids_root is e.g. .../bids_output/eeg
            fmri_root = str(Path(bids).parent / "fmri")
        if not fmri_root:
            fmri_root = bids.replace("bids_output", "fMRI_data") if bids else "."

    fmri_root = Path(fmri_root)
    if not fmri_root.is_absolute():
        from eeg_pipeline.utils.config.loader import get_project_root
        fmri_root = get_project_root() / fmri_root

    task_candidates = [task] if task else []

    subject = args.subject
    if not subject and fmri_root.exists():
        for sub_dir in sorted(fmri_root.glob("sub-*")):
            if sub_dir.is_dir():
                subject = sub_dir.name.replace("sub-", "")
                break

    # Discover columns/values from fMRI BIDS events files.
    result = {"columns": [], "values": {}, "source": None, "file": None}

    sub_label = None
    if subject:
        subj_id = subject.replace("sub-", "")
        sub_label = f"sub-{subj_id}"

    def glob_events(func_dir: Path, sub: str, t: str) -> list:
        # Prefer *_bold_events.tsv (time-aligned "analysis-ready" events) when present,
        # then fall back to standard *_events.tsv.
        out = sorted(func_dir.glob(f"{sub}_task-{t}_run-*_bold_events.tsv"))
        if not out:
            out = [
                p for p in sorted(func_dir.glob(f"{sub}_task-{t}_run-*_events.tsv"))
                if not p.name.endswith("_bold_events.tsv")
            ]
        if not out:
            out = sorted(func_dir.glob(f"{sub}_task-{t}_*events.tsv"))
        return out

    candidates = []
    if sub_label:
        func_dir = fmri_root / sub_label / "func"
        if func_dir.exists():
            for t in task_candidates:
                candidates = glob_events(func_dir, sub_label, t)
                if candidates:
                    break
    if not candidates and fmri_root.exists():
        for sub_dir in sorted(fmri_root.glob("sub-*"))[:5]:
            func_dir2 = sub_dir / "func"
            if not func_dir2.exists():
                continue
            sub2 = sub_dir.name
            for t in task_candidates:
                candidates = glob_events(func_dir2, sub2, t)
                if candidates:
                    break
            if candidates:
                break

    if candidates:
        import pandas as pd

        result["source"] = "fmri_events"
        result["file"] = str(candidates[0])

        columns = set()
        values: dict[str, set[str]] = {}
        for events_path in candidates[:10]:
            try:
                df = pd.read_csv(events_path, sep="\t")
            except Exception:
                continue

            for col in df.columns:
                columns.add(col)
                if col.lower() in {"onset", "duration", "sample"}:
                    continue
                uniq = df[col].dropna().unique()
                if len(uniq) > 50:
                    continue
                bucket = values.setdefault(col, set())
                for v in uniq:
                    if pd.isna(v):
                        continue
                    s = str(v).strip()
                    if s:
                        bucket.add(s)

        result["columns"] = sorted(columns)
        result["values"] = {k: sorted(v) for k, v in values.items()}

    if args.column:
        if args.column in result["values"]:
            result["values"] = {args.column: result["values"][args.column]}
        else:
            result["values"] = {}

    if args.output_json:
        _print_json_output(result)
    else:
        _print_discovery_report(result)


def _handle_multigroup_stats_mode(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Handle multigroup-stats mode: discover available multigroup comparisons from precomputed stats."""
    from eeg_pipeline.infra.paths import resolve_deriv_root
    from eeg_pipeline.infra.tsv import read_tsv
    
    deriv_root = resolve_deriv_root(config=config)
    
    subject = args.subject
    if not subject and subjects:
        subject = subjects[0]
    
    result = {
        "available": False,
        "groups": [],
        "n_features": 0,
        "n_significant": 0,
        "file": None,
        "subject": subject,
    }
    
    if not subject:
        for sub_dir in sorted(deriv_root.glob("sub-*")):
            if sub_dir.is_dir():
                subject = sub_dir.name.replace("sub-", "")
                result["subject"] = subject
                break
    
    if not subject:
        result["error"] = "No subject found"
        if args.output_json:
            _print_json_output(result)
        else:
            print("Error: No subject found for multigroup stats discovery")
        return
    
    stats_dir = deriv_root / f"sub-{subject}" / "stats" / "condition_effects"
    if not stats_dir.exists():
        stats_dir = deriv_root / f"sub-{subject}" / "stats"
    
    multigroup_files = list(stats_dir.glob("condition_effects_multigroup*.tsv")) if stats_dir.exists() else []
    
    if not multigroup_files:
        result["error"] = "No multigroup stats found. Run behavior pipeline with 3+ comparison values first."
        if args.output_json:
            _print_json_output(result)
        else:
            print("No multigroup stats found.")
            print("Run behavior pipeline with 3+ comparison values to generate multigroup stats.")
        return
    
    stats_file = multigroup_files[0]
    df = read_tsv(stats_file)
    
    if df is None or df.empty:
        result["error"] = f"Could not read stats file: {stats_file}"
        if args.output_json:
            _print_json_output(result)
        else:
            print(f"Error reading stats file: {stats_file}")
        return
    
    groups = set()
    if "group1" in df.columns:
        groups.update(df["group1"].dropna().unique())
    if "group2" in df.columns:
        groups.update(df["group2"].dropna().unique())
    
    result["available"] = True
    result["groups"] = sorted(list(groups))
    result["n_features"] = df["feature"].nunique() if "feature" in df.columns else len(df)
    result["n_significant"] = int(df["significant_fdr"].sum()) if "significant_fdr" in df.columns else 0
    result["file"] = str(stats_file)
    
    if args.output_json:
        _print_json_output(result)
    else:
        print("=" * 50)
        print("    MULTIGROUP STATS DISCOVERY")
        print("=" * 50)
        print(f"Subject: {subject}")
        print(f"File: {stats_file.name}")
        print(f"Groups: {', '.join(result['groups'])}")
        print(f"Features: {result['n_features']}")
        print(f"Significant (FDR): {result['n_significant']}")

