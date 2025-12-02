"""
Path utilities for BIDS and derivative paths.

This module provides functions for constructing and resolving paths
in BIDS-compliant directory structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from mne_bids import BIDSPath

try:
    from ..config.loader import ConfigDict
except ImportError:
    ConfigDict = dict

EEGConfig = ConfigDict


def _normalize_subject_label(subject: str) -> str:
    if subject.startswith("sub-"):
        return subject
    return f"sub-{subject}"


def _normalize_subject_id(subject: str) -> str:
    if subject.startswith("sub-"):
        return subject.replace("sub-", "")
    return subject


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_first(pattern: str) -> Optional[Path]:
    import glob
    candidates = sorted(glob.glob(pattern))
    return Path(candidates[0]) if candidates else None


def bids_sub_eeg_path(bids_root: Path, subject: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return Path(bids_root) / subject_label / "eeg"


def bids_events_path(bids_root: Path, subject: str, task: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return bids_sub_eeg_path(bids_root, subject) / f"{subject_label}_task-{task}_events.tsv"


def deriv_sub_eeg_path(deriv_root: Path, subject: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return Path(deriv_root) / subject_label / "eeg"


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
    return Path(deriv_root) / "group" / "eeg"


def deriv_group_stats_path(deriv_root: Path) -> Path:
    return deriv_group_eeg_path(deriv_root) / "stats"


def deriv_group_plots_path(deriv_root: Path, subdir: Optional[str] = None) -> Path:
    plots_root = deriv_group_eeg_path(deriv_root) / "plots"
    if subdir:
        return plots_root / subdir
    return plots_root


def find_connectivity_features_path(deriv_root: Path, subject: str) -> Path:
    sub = _normalize_subject_label(subject)
    parquet_path = Path(deriv_root) / sub / "eeg" / "connectivity_features.parquet"
    if parquet_path.exists():
        return parquet_path
    return (
        Path(deriv_root)
        / sub
        / "eeg"
        / "features"
        / "features_connectivity.tsv"
    )


def _resolve_deriv_root(
    deriv_root: Optional[Path],
    config: Optional[EEGConfig] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> Path:
    if deriv_root is not None:
        return Path(deriv_root)
    
    if config is not None:
        try:
            deriv_path = config.deriv_root
            if deriv_path is not None:
                return Path(deriv_path)
        except AttributeError:
            pass
    
    if constants is not None and "DERIV_ROOT" in constants:
        return Path(constants["DERIV_ROOT"])
    
    raise ValueError(
        "Either deriv_root, config, or constants must be provided to resolve derivatives root"
    )


def _check_clean_tokens(filename: str) -> bool:
    clean_tokens = ("proc-clean", "proc-cleaned", "clean")
    return any(token in filename for token in clean_tokens)


def _search_standard_bids_paths(
    root: Path,
    subject_clean: str,
    task: str,
) -> Optional[Path]:
    bids_path = BIDSPath(
        subject=subject_clean,
        task=task,
        datatype="eeg",
        processing="clean",
        suffix="epo",
        extension=".fif",
        root=root,
        check=False,
    )
    if bids_path.fpath and bids_path.fpath.exists():
        return bids_path.fpath
    
    standard_paths = [
        root / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_proc-clean_epo.fif",
        root / "preprocessed" / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_proc-clean_epo.fif",
    ]
    
    for path in standard_paths:
        if path.exists():
            return path
    
    return None


def _search_directory_for_epochs(
    directory: Path,
    subject_clean: str,
    task: str,
    prefer_clean: bool = True,
) -> Optional[Path]:
    if not directory.exists():
        return None
    
    pattern = f"sub-{subject_clean}_task-{task}*epo.fif"
    candidates = sorted(directory.glob(pattern))
    
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
    root = _resolve_deriv_root(deriv_root, config, constants)
    subject_clean = _normalize_subject_id(subject)
    
    standard_path = _search_standard_bids_paths(root, subject_clean, task)
    if standard_path:
        return standard_path
    
    search_directories = [
        (root / f"sub-{subject_clean}" / "eeg", True),
        (root / "preprocessed" / f"sub-{subject_clean}" / "eeg", True),
        (root / f"sub-{subject_clean}", False),
        (root / "preprocessed", True),
    ]
    
    for directory, prefer_clean in search_directories:
        found_path = _search_directory_for_epochs(directory, subject_clean, task, prefer_clean)
        if found_path:
            return found_path
    
    return None


def _resolve_bids_root(
    bids_root: Optional[Path],
    constants: Optional[Dict[str, Any]],
    config: Optional[EEGConfig],
) -> Path:
    if bids_root is not None:
        return Path(bids_root)
    
    if config is not None:
        # Try to get bids_root from config, handling missing attribute
        try:
            cfg_bids_root = config.bids_root
            if cfg_bids_root is not None:
                return Path(cfg_bids_root)
        except AttributeError:
            pass
        
        # Fallback: try paths.bids_root via get()
        cfg_bids_root = config.get("paths.bids_root")
        if cfg_bids_root is not None:
            return Path(cfg_bids_root)
    
    if constants is not None and "BIDS_ROOT" in constants:
        return Path(constants["BIDS_ROOT"])
    
    raise ValueError("BIDS root not configured. Set paths.bids_root in eeg_config.yaml or pass bids_root parameter.")


def _find_events_path(bids_root: Path, subject_clean: str, task: str) -> Optional[Path]:
    bids_path = BIDSPath(
        subject=subject_clean,
        task=task,
        datatype="eeg",
        suffix="events",
        extension=".tsv",
        root=bids_root,
        check=False,
    )
    
    if bids_path.fpath is not None:
        return bids_path.fpath
    
    fallback_path = bids_root / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_events.tsv"
    return fallback_path


def _load_events_df(
    subject: str,
    task: str,
    bids_root: Optional[Path] = None,
    constants=None,
    config: Optional[EEGConfig] = None,
) -> Optional[Any]:
    import pandas as pd
    root = _resolve_bids_root(bids_root, constants, config)
    subject_clean = _normalize_subject_id(subject)
    
    events_path = _find_events_path(root, subject_clean, task)
    if events_path is None or not events_path.exists():
        return None
    
    return pd.read_csv(events_path, sep="\t")


def extract_subject_id_from_path(path: Path) -> Optional[str]:
    import re
    path_str = str(path)
    match = re.search(r'sub-(\d+)', path_str)
    return match.group(1) if match else None


__all__ = [
    "_normalize_subject_label",
    "_normalize_subject_id",
    "ensure_dir",
    "find_first",
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
    "_resolve_deriv_root",
    "_check_clean_tokens",
    "_search_standard_bids_paths",
    "_search_directory_for_epochs",
    "_find_clean_epochs_path",
    "_resolve_bids_root",
    "_find_events_path",
    "_load_events_df",
    "extract_subject_id_from_path",
]










