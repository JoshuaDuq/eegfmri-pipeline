"""Path utilities for BIDS and derivative paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from mne_bids import BIDSPath

from eeg_pipeline.utils.config.loader import ConfigDict


EEGConfig = ConfigDict

# BIDS naming constants
SUBJECT_PREFIX = "sub-"
CLEAN_PROCESSING_TOKENS = ("proc-clean", "proc-cleaned", "clean")
EPOCHS_SUFFIX = "epo.fif"
EVENTS_SUFFIX = "events.tsv"


def _normalize_subject_label(subject: str) -> str:
    """Ensure subject label has 'sub-' prefix."""
    if subject.startswith(SUBJECT_PREFIX):
        return subject
    return f"{SUBJECT_PREFIX}{subject}"


def _extract_subject_id(subject: str) -> str:
    """Extract subject ID by removing 'sub-' prefix if present."""
    if subject.startswith(SUBJECT_PREFIX):
        return subject.replace(SUBJECT_PREFIX, "", 1)
    return subject


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bids_sub_eeg_path(bids_root: Path, subject: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return bids_root / subject_label / "eeg"


def bids_events_path(bids_root: Path, subject: str, task: str) -> Path:
    """Construct BIDS events file path for subject and task."""
    subject_label = _normalize_subject_label(subject)
    return bids_sub_eeg_path(bids_root, subject) / f"{subject_label}_task-{task}_{EVENTS_SUFFIX}"


def deriv_sub_eeg_path(deriv_root: Path, subject: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return deriv_root / subject_label / "eeg"


def deriv_features_path(deriv_root: Path, subject: str) -> Path:
    return deriv_sub_eeg_path(deriv_root, subject) / "features"


def deriv_stats_path(deriv_root: Path, subject: str) -> Path:
    return deriv_sub_eeg_path(deriv_root, subject) / "stats"


def deriv_plots_path(deriv_root: Path, subject: str, subdir: Optional[str] = None) -> Path:
    plots_dir = deriv_sub_eeg_path(deriv_root, subject) / "plots"
    if subdir:
        return plots_dir / subdir
    return plots_dir


def deriv_group_eeg_path(deriv_root: Path) -> Path:
    return deriv_root / "group" / "eeg"


def deriv_group_stats_path(deriv_root: Path) -> Path:
    return deriv_group_eeg_path(deriv_root) / "stats"


def deriv_group_plots_path(deriv_root: Path, subdir: Optional[str] = None) -> Path:
    plots_root = deriv_group_eeg_path(deriv_root) / "plots"
    if subdir:
        return plots_root / subdir
    return plots_root


def find_connectivity_features_path(deriv_root: Path, subject: str) -> Path:
    sub = _normalize_subject_label(subject)
    base = deriv_root / sub / "eeg" / "features"
    return base / "connectivity" / "features_connectivity.parquet"


def _resolve_path_from_config(
    path_value: Optional[Path],
    config: Optional[EEGConfig],
    config_attr: Optional[str],
    config_key: Optional[str],
    constants: Optional[Dict[str, Any]],
    constants_key: Optional[str],
    error_message: str,
) -> Path:
    """Resolve path from multiple sources: direct value, config, or constants."""
    if path_value is not None:
        return Path(path_value)

    if config is not None:
        if config_attr is not None:
            config_path = getattr(config, config_attr, None)
            if config_path is not None:
                return Path(config_path)
        if config_key is not None:
            config_path = config.get(config_key)
            if config_path is not None:
                return Path(config_path)

    if constants is not None and constants_key is not None:
        if constants_key in constants:
            return Path(constants[constants_key])

    raise ValueError(error_message)


def _resolve_deriv_root(
    deriv_root: Optional[Path],
    config: Optional[EEGConfig] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> Path:
    """Resolve derivatives root path from multiple sources."""
    return _resolve_path_from_config(
        path_value=deriv_root,
        config=config,
        config_attr="deriv_root",
        config_key=None,
        constants=constants,
        constants_key="DERIV_ROOT",
        error_message="Either deriv_root, config, or constants must be provided to resolve derivatives root",
    )


def resolve_deriv_root(
    deriv_root: Optional[Path] = None,
    *,
    config: Optional[EEGConfig] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> Path:
    """Resolve derivatives root path from multiple sources (public API)."""
    return _resolve_deriv_root(deriv_root=deriv_root, config=config, constants=constants)


def _check_clean_tokens(filename: str) -> bool:
    """Check if filename contains clean processing tokens."""
    return any(token in filename for token in CLEAN_PROCESSING_TOKENS)


def _search_standard_bids_paths(root: Path, subject_id: str, task: str) -> Optional[Path]:
    """Search standard BIDS paths for clean epochs file."""
    bids_path = BIDSPath(
        subject=subject_id,
        task=task,
        datatype="eeg",
        processing="clean",
        suffix="epo",
        extension=".fif",
        root=root,
        check=False,
    )
    if bids_path.fpath is not None and bids_path.fpath.exists():
        return bids_path.fpath

    subject_label = f"{SUBJECT_PREFIX}{subject_id}"
    filename = f"{subject_label}_task-{task}_proc-clean_{EPOCHS_SUFFIX}"
    standard_paths = [
        root / subject_label / "eeg" / filename,
        root / "preprocessed" / "eeg" / subject_label / filename,
    ]

    for path in standard_paths:
        if path.exists():
            return path

    return None


def _search_directory_for_epochs(
    directory: Path,
    subject_id: str,
    task: str,
    prefer_clean: bool = True,
) -> Optional[Path]:
    """Search directory for epochs files matching subject and task."""
    if not directory.exists():
        return None

    subject_label = f"{SUBJECT_PREFIX}{subject_id}"
    pattern = f"{subject_label}_task-{task}*{EPOCHS_SUFFIX}"
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        # Common MNE-BIDS derivative layout nests files under datatype/session folders.
        candidates = sorted(directory.rglob(pattern))
        if not candidates:
            return None

    if prefer_clean:
        for candidate in candidates:
            if _check_clean_tokens(candidate.name):
                return candidate

    return candidates[0]


def _find_clean_epochs_path(
    subject: str,
    task: str,
    deriv_root: Optional[Path] = None,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
) -> Optional[Path]:
    """Find path to clean epochs file for subject and task."""
    root = _resolve_deriv_root(deriv_root, config, constants)
    subject_id = _extract_subject_id(subject)

    standard_path = _search_standard_bids_paths(root, subject_id, task)
    if standard_path:
        return standard_path

    subject_label = f"{SUBJECT_PREFIX}{subject_id}"
    search_directories = [
        (root / subject_label / "eeg", True),
        (root / "preprocessed" / "eeg" / subject_label, True),
        (root / subject_label, False),
        (root / "preprocessed", True),
    ]

    for directory, prefer_clean in search_directories:
        found_path = _search_directory_for_epochs(directory, subject_id, task, prefer_clean)
        if found_path:
            return found_path

    return None


def _derive_clean_events_from_epochs_path(epochs_path: Path) -> Path:
    name = epochs_path.name
    if name.endswith("_proc-clean_epo.fif"):
        return epochs_path.with_name(name.replace("_proc-clean_epo.fif", "_proc-clean_events.tsv"))
    if name.endswith("_proc-cleaned_epo.fif"):
        return epochs_path.with_name(name.replace("_proc-cleaned_epo.fif", "_proc-cleaned_events.tsv"))
    if name.endswith("_clean_epo.fif"):
        return epochs_path.with_name(name.replace("_clean_epo.fif", "_clean_events.tsv"))
    if name.endswith("_epo.fif"):
        # Best-effort fallback; prefer explicit proc-clean naming.
        return epochs_path.with_name(name.replace("_epo.fif", "_events.tsv"))
    return epochs_path.with_suffix(".tsv")


def _find_clean_events_path(
    subject: str,
    task: str,
    deriv_root: Optional[Path] = None,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
) -> Optional[Path]:
    """Find path to clean (post-rejection) events.tsv for subject and task."""
    epochs_path = _find_clean_epochs_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        constants=constants,
        config=config,
    )
    if epochs_path is not None:
        candidate = _derive_clean_events_from_epochs_path(epochs_path)
        if candidate.exists():
            return candidate

    root = _resolve_deriv_root(deriv_root, config, constants)
    subject_id = _extract_subject_id(subject)
    subject_label = f"{SUBJECT_PREFIX}{subject_id}"
    pattern = f"{subject_label}_task-{task}*proc-clean*_{EVENTS_SUFFIX}"

    for base in (
        root / subject_label,
        root / subject_label / "eeg",
        root / "preprocessed" / "eeg" / subject_label,
        root / "preprocessed" / "eeg" / subject_label / "eeg",
        root / "preprocessed" / "eeg",
    ):
        if not base.exists():
            continue
        matches = sorted(base.rglob(pattern))
        if matches:
            return matches[0]
    return None


def find_clean_epochs_path(
    subject: str,
    task: str,
    deriv_root: Optional[Path] = None,
    *,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
) -> Optional[Path]:
    return _find_clean_epochs_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        constants=constants,
        config=config,
    )


def find_clean_events_path(
    subject: str,
    task: str,
    deriv_root: Optional[Path] = None,
    *,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
) -> Optional[Path]:
    return _find_clean_events_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        constants=constants,
        config=config,
    )


def _resolve_bids_root(
    bids_root: Optional[Path],
    constants: Optional[Dict[str, Any]],
    config: Optional[EEGConfig],
) -> Path:
    """Resolve BIDS root path from multiple sources."""
    return _resolve_path_from_config(
        path_value=bids_root,
        config=config,
        config_attr="bids_root",
        config_key="paths.bids_root",
        constants=constants,
        constants_key="BIDS_ROOT",
        error_message="BIDS root not configured. Set paths.bids_root in eeg_config.yaml or pass bids_root parameter.",
    )


def _find_events_path(bids_root: Path, subject_id: str, task: str) -> Optional[Path]:
    """Find events file path for subject and task."""
    bids_path = BIDSPath(
        subject=subject_id,
        task=task,
        datatype="eeg",
        suffix="events",
        extension=".tsv",
        root=bids_root,
        check=False,
    )

    if bids_path.fpath is not None:
        return bids_path.fpath

    subject_label = f"{SUBJECT_PREFIX}{subject_id}"
    fallback_path = bids_root / subject_label / "eeg" / f"{subject_label}_task-{task}_{EVENTS_SUFFIX}"
    return fallback_path


def _load_events_df(
    subject: str,
    task: str,
    bids_root: Optional[Path] = None,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
    *,
    prefer_clean: bool = True,
) -> Optional[pd.DataFrame]:
    """Load events DataFrame (prefer cleaned derivative events when available)."""
    if prefer_clean:
        try:
            deriv_root = _resolve_deriv_root(None, config, constants)
        except Exception:
            deriv_root = None
        if deriv_root is not None:
            clean_path = _find_clean_events_path(
                subject=subject,
                task=task,
                deriv_root=deriv_root,
                constants=constants,
                config=config,
            )
            if clean_path is not None and clean_path.exists():
                return pd.read_csv(clean_path, sep="\t")

    root = _resolve_bids_root(bids_root, constants, config)
    subject_id = _extract_subject_id(subject)

    events_path = _find_events_path(root, subject_id, task)
    if events_path is None or not events_path.exists():
        return None

    return pd.read_csv(events_path, sep="\t")


def load_events_df(
    subject: str,
    task: str,
    bids_root: Optional[Path] = None,
    *,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
    prefer_clean: bool = True,
) -> Optional[pd.DataFrame]:
    """Load events DataFrame from BIDS events file (public API)."""
    return _load_events_df(
        subject=subject,
        task=task,
        bids_root=bids_root,
        constants=constants,
        config=config,
        prefer_clean=prefer_clean,
    )


def extract_subject_id_from_path(path: Path) -> Optional[str]:
    """Extract subject ID from path using BIDS naming convention."""
    path_str = str(path)
    pattern = rf"{SUBJECT_PREFIX}(\d+)"
    match = re.search(pattern, path_str)
    return match.group(1) if match else None


def ensure_derivatives_dataset_description(
    deriv_root: Optional[Path] = None,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
) -> None:
    """Ensure derivatives dataset_description.json exists with BIDS metadata."""
    root = _resolve_deriv_root(deriv_root, config, constants)

    desc_path = root / "dataset_description.json"
    if desc_path.exists():
        return

    metadata = {
        "Name": "EEG Pipeline Derivatives",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Thermal_Pain_EEG_Pipeline",
                "Version": "unknown",
                "Description": "Custom EEG analysis (ERP, TFR, features, machine learning)",
            }
        ],
    }
    ensure_dir(root)
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


__all__ = [
    "ensure_dir",
    "bids_sub_eeg_path",
    "bids_events_path",
    "deriv_sub_eeg_path",
    "deriv_features_path",
    "deriv_stats_path",
    "deriv_plots_path",
    "deriv_group_eeg_path",
    "deriv_group_stats_path",
    "deriv_group_plots_path",
    "find_connectivity_features_path",
    "resolve_deriv_root",
    "find_clean_epochs_path",
    "find_clean_events_path",
    "load_events_df",
    "extract_subject_id_from_path",
    "ensure_derivatives_dataset_description",
]
