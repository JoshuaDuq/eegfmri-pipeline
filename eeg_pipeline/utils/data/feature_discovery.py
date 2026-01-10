"""
Feature File Discovery
======================

Utilities for discovering and listing available feature files for a subject.
Used by the behavior pipeline and TUI to present users with selectable options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.paths import deriv_features_path


_BYTES_PER_KILOBYTE = 1024


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
    "connectivity": "features_connectivity.tsv",
    "itpc": "features_itpc.tsv",
    "pac": "features_pac_trials.tsv",
    # Exploratory & QC
    "complexity": "features_complexity.tsv",
    "bursts": "features_bursts.tsv",
    "quality": "features_quality.tsv",
    "temporal": "features_temporal.tsv",
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


def _count_columns_and_rows(path: Path) -> tuple[int, int]:
    """
    Count columns and rows in a feature file.
    
    Supports TSV and Parquet formats. Returns (0, 0) if file doesn't exist
    or cannot be read.
    
    Parameters
    ----------
    path : Path
        Path to the feature file
        
    Returns
    -------
    tuple[int, int]
        (column_count, row_count)
    """
    if not path.exists():
        return 0, 0
    
    try:
        if path.suffix == ".parquet":
            import pandas as pd
            dataframe = pd.read_parquet(path)
            column_count = len(dataframe.columns)
            row_count = len(dataframe)
            return column_count, row_count
        
        if path.suffix == ".tsv":
            with open(path, 'r', encoding='utf-8') as file:
                header_line = file.readline().strip()
                if not header_line:
                    return 0, 0
                
                tab_separated_values = header_line.split('\t')
                column_count = len(tab_separated_values)
                row_count = sum(1 for _ in file)
                return column_count, row_count
    except (OSError, ValueError, ImportError):
        return 0, 0
    
    return 0, 0


def _find_feature_file_path(features_dir: Path, key: str, filename: str) -> Path:
    """
    Find the feature file path, checking subfolder first then root.
    
    Features can be stored in two locations:
    - Subfolder: features/{key}/{filename} (new structure)
    - Root: features/{filename} (legacy structure)
    
    Parameters
    ----------
    features_dir : Path
        Root features directory
    key : str
        Feature key (e.g., 'power', 'connectivity')
    filename : str
        Feature filename (e.g., 'features_power.tsv')
        
    Returns
    -------
    Path
        Path to the feature file (subfolder if exists, otherwise root)
    """
    subfolder_path = features_dir / key / filename
    if subfolder_path.exists():
        return subfolder_path
    return features_dir / filename


def discover_feature_files(
    subject: str,
    deriv_root: Path,
    include_empty: bool = False,
) -> dict[str, FeatureFileInfo]:
    """
    Discover available feature files for a subject.
    
    Searches for feature files in both subfolder structure (features/{key}/)
    and root features directory for backwards compatibility.
    
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
    dict[str, FeatureFileInfo]
        Mapping from file key to FeatureFileInfo
    """
    if not subject:
        raise ValueError("Subject ID cannot be empty")
    if not deriv_root.exists():
        raise ValueError(f"Derivatives root does not exist: {deriv_root}")
    
    features_dir = deriv_features_path(deriv_root, subject)
    result: dict[str, FeatureFileInfo] = {}
    
    for key, filename in STANDARD_FEATURE_FILES.items():
        file_path = _find_feature_file_path(features_dir, key, filename)
        file_exists = file_path.exists()
        
        if not file_exists and not include_empty:
            continue
        
        column_count = 0
        row_count = 0
        file_size_kb = 0.0
        
        if file_exists:
            column_count, row_count = _count_columns_and_rows(file_path)
            file_size_bytes = file_path.stat().st_size
            file_size_kb = file_size_bytes / _BYTES_PER_KILOBYTE
        
        display_name = FEATURE_FILE_DISPLAY_NAMES.get(key, key.title())
        
        result[key] = FeatureFileInfo(
            key=key,
            display_name=display_name,
            filename=filename,
            path=file_path,
            exists=file_exists,
            n_columns=column_count,
            n_rows=row_count,
            file_size_kb=file_size_kb,
        )
    
    return result


def get_available_feature_keys(subject: str, deriv_root: Path) -> list[str]:
    """
    Get list of available feature file keys for a subject.
    
    Parameters
    ----------
    subject : str
        Subject ID (without 'sub-' prefix)
    deriv_root : Path
        Path to derivatives root
        
    Returns
    -------
    list[str]
        List of feature file keys that exist
    """
    feature_files = discover_feature_files(subject, deriv_root, include_empty=False)
    return list(feature_files.keys())


def validate_feature_file_selection(
    selected: list[str],
    subject: str,
    deriv_root: Path,
) -> tuple[list[str], list[str]]:
    """
    Validate selected feature files against what's available.
    
    Parameters
    ----------
    selected : list[str]
        List of feature file keys to validate
    subject : str
        Subject ID (without 'sub-' prefix)
    deriv_root : Path
        Path to derivatives root
        
    Returns
    -------
    tuple[list[str], list[str]]
        (valid_keys, missing_keys)
    """
    available_keys = set(get_available_feature_keys(subject, deriv_root))
    valid_keys = [key for key in selected if key in available_keys]
    missing_keys = [key for key in selected if key not in available_keys]
    return valid_keys, missing_keys


def feature_files_to_json(files: dict[str, FeatureFileInfo]) -> list[dict[str, Any]]:
    """
    Convert feature file info to JSON-serializable format.
    
    Used by TUI to display feature file information.
    
    Parameters
    ----------
    files : dict[str, FeatureFileInfo]
        Mapping from key to FeatureFileInfo
        
    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries with serialized feature file information
    """
    return [
        {
            "key": file_info.key,
            "display_name": file_info.display_name,
            "filename": file_info.filename,
            "exists": file_info.exists,
            "n_columns": file_info.n_columns,
            "n_rows": file_info.n_rows,
            "file_size_kb": round(file_info.file_size_kb, 1),
        }
        for file_info in files.values()
    ]
