"""
Feature File Discovery
======================

Utilities for discovering and listing available feature files for a subject.
Used by the behavior pipeline and TUI to present users with selectable options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.infra.paths import deriv_features_path


STANDARD_FEATURE_FILES = {
    # Core spectral
    "power": "features_power.tsv",
    "spectral": "features_spectral.tsv",
    "aperiodic": "features_aperiodic.tsv",
    "erp": "features_erp.tsv",
    "erds": "features_erds.tsv",
    "ratios": "features_ratios.tsv",
    "asymmetry": "features_asymmetry.tsv",
    # Connectivity & phase
    "connectivity": "features_connectivity.parquet",
    "itpc": "features_itpc.tsv",
    "pac": "features_pac_trials.tsv",
    # Exploratory & QC
    "complexity": "features_complexity.tsv",
    "bursts": "features_bursts.tsv",
    "quality": "features_quality.tsv",
    "temporal": "features_temporal.tsv",
    # Aggregate
    "all": "features_all.tsv",
}

FEATURE_FILE_DISPLAY_NAMES = {
    # Core spectral
    "power": "Power Spectral Features",
    "spectral": "Spectral Peak Features",
    "aperiodic": "Aperiodic (1/f) Features",
    "erp": "ERP/LEP Time-Domain Features",
    "erds": "Event-Related Desync/Sync",
    "ratios": "Band Power Ratios",
    "asymmetry": "Hemispheric Asymmetry",
    # Connectivity & phase
    "connectivity": "Connectivity Features",
    "itpc": "Inter-Trial Phase Coherence",
    "pac": "Phase-Amplitude Coupling",
    # Exploratory & QC
    "complexity": "Complexity Features",
    "bursts": "Burst Dynamics",
    "quality": "Trial Quality Metrics",
    "temporal": "Temporal Binned Features",
    # Aggregate
    "all": "All Features Combined",
}


@dataclass
class FeatureFileInfo:
    """Information about a feature file."""
    key: str
    display_name: str
    filename: str
    path: Path
    exists: bool
    n_columns: int = 0
    n_rows: int = 0
    file_size_kb: float = 0.0


def _count_columns(path: Path) -> tuple[int, int]:
    """
    Quickly count columns and rows in a feature file.
    Returns (n_columns, n_rows).
    """
    if not path.exists():
        return 0, 0
    
    try:
        if path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            return len(df.columns), len(df)
        elif path.suffix == ".tsv":
            with open(path, 'r') as f:
                header = f.readline().strip()
                if not header:
                    return 0, 0
                n_cols = len(header.split('\t'))
                n_rows = sum(1 for _ in f)
                return n_cols, n_rows
    except Exception:
        pass
    
    return 0, 0


def discover_feature_files(
    subject: str,
    deriv_root: Path,
    include_empty: bool = False,
) -> Dict[str, FeatureFileInfo]:
    """
    Discover available feature files for a subject.
    
    Parameters
    ----------
    subject : str
        Subject ID (without 'sub-' prefix)
    deriv_root : Path
        Path to derivatives root
    include_empty : bool
        If True, include files that don't exist in the result
    
    Returns
    -------
    dict mapping file key to FeatureFileInfo
    """
    features_dir = deriv_features_path(deriv_root, subject)
    result: Dict[str, FeatureFileInfo] = {}
    
    for key, filename in STANDARD_FEATURE_FILES.items():
        path = features_dir / filename
        exists = path.exists()
        
        if not exists and not include_empty:
            continue
        
        n_cols, n_rows = 0, 0
        file_size_kb = 0.0
        
        if exists:
            n_cols, n_rows = _count_columns(path)
            file_size_kb = path.stat().st_size / 1024
        
        result[key] = FeatureFileInfo(
            key=key,
            display_name=FEATURE_FILE_DISPLAY_NAMES.get(key, key.title()),
            filename=filename,
            path=path,
            exists=exists,
            n_columns=n_cols,
            n_rows=n_rows,
            file_size_kb=file_size_kb,
        )
    
    return result


def get_available_feature_keys(subject: str, deriv_root: Path) -> List[str]:
    """Get list of available feature file keys for a subject."""
    files = discover_feature_files(subject, deriv_root, include_empty=False)
    return list(files.keys())


def validate_feature_file_selection(
    selected: List[str],
    subject: str,
    deriv_root: Path,
) -> tuple[List[str], List[str]]:
    """
    Validate selected feature files against what's available.
    
    Returns (valid_keys, missing_keys)
    """
    available = set(get_available_feature_keys(subject, deriv_root))
    valid = [k for k in selected if k in available]
    missing = [k for k in selected if k not in available]
    return valid, missing


def feature_files_to_json(files: Dict[str, FeatureFileInfo]) -> List[Dict[str, Any]]:
    """Convert feature file info to JSON-serializable format (for TUI)."""
    return [
        {
            "key": info.key,
            "display_name": info.display_name,
            "filename": info.filename,
            "exists": info.exists,
            "n_columns": info.n_columns,
            "n_rows": info.n_rows,
            "file_size_kb": round(info.file_size_kb, 1),
        }
        for info in files.values()
    ]
