"""
Subject Discovery Utilities
===========================

Functions for discovering subjects and files in BIDS and derivatives directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import find_clean_epochs_path


def _collect_subjects_from_bids(bids_root: Path) -> List[str]:
    """Collect all subjects from a BIDS directory."""
    if not bids_root.exists():
        return []
    subjects = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if sub_dir.is_dir():
            subjects.append(sub_dir.name[4:])
    return subjects


def _collect_subjects_from_derivatives_epochs(
    deriv_root: Path, 
    task: str, 
    config: Optional[ConfigDict] = None, 
    constants: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Collect subjects that have clean epochs available."""
    if not deriv_root.exists():
        return []
        
    subjects = []
    for sub_dir in sorted(deriv_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name[4:]
        epo_path = find_clean_epochs_path(sub_id, task, deriv_root=deriv_root, config=config, constants=constants)
        if epo_path is not None and epo_path.exists():
            subjects.append(sub_id)
    return subjects


def _collect_subjects_from_features(deriv_root: Path) -> List[str]:
    """Collect subjects that have extracted features."""
    if not deriv_root.exists():
        return []
    subjects = []
    for sub_dir in sorted(deriv_root.glob("sub-*/eeg/features")):
        eeg_feat = sub_dir / "features_eeg_direct.tsv"
        y_tsv = sub_dir / "target_vas_ratings.tsv"
        if eeg_feat.exists() and y_tsv.exists():
            sub_id = sub_dir.parts[-3].replace("sub-", "")
            subjects.append(sub_id)
    return subjects


def get_available_subjects(
    bids_root: Path,
    deriv_root: Path,
    task: str,
    require_features: bool = False,
    require_epochs: bool = False,
    config: Optional[ConfigDict] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Get list of available subjects based on criteria.
    
    Parameters
    ----------
    bids_root : Path
        BIDS dataset root
    deriv_root : Path
        Derivatives root
    task : str
        Task name
    require_features : bool
        If True, require extracted features
    require_epochs : bool
        If True, require clean epochs
        
    Returns
    -------
    List[str]
        List of subject IDs
    """
    if require_features:
        return _collect_subjects_from_features(deriv_root)
        
    if require_epochs:
        return _collect_subjects_from_derivatives_epochs(deriv_root, task, config=config, constants=constants)
        
    return _collect_subjects_from_bids(bids_root)
