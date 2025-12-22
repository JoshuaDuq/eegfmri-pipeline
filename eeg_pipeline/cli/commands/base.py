"""Shared utilities and constants for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import List

from eeg_pipeline.pipelines.constants import (
    BEHAVIOR_COMPUTATIONS,
    BEHAVIOR_VISUALIZE_CATEGORIES,
    FEATURE_CATEGORIES,
    FEATURE_VISUALIZE_CATEGORIES,
    FREQUENCY_BANDS,
)


def detect_available_bands(features_dir) -> List[str]:
    """Detect available frequency bands from feature file columns."""
    bands = {"delta", "theta", "alpha", "beta", "gamma"}
    found_bands = set()
    
    features_dir = Path(features_dir)
    for tsv_file in features_dir.glob("features_*.tsv"):
        try:
            with open(tsv_file, "r") as f:
                header = f.readline().strip()
            columns = header.split("\t")
            
            for col in columns:
                col_lower = col.lower()
                for band in bands:
                    if f"_{band}_" in col_lower or col_lower.endswith(f"_{band}"):
                        found_bands.add(band)
        except Exception:
            continue
    
    return sorted(found_bands)


def _empty_feature_availability() -> dict:
    """Return feature availability dict with all features marked as unavailable."""
    all_categories = [
        "power", "connectivity", "aperiodic", "erp", "bursts", "itpc", "pac",
        "complexity", "quality", "erds", "spectral", "ratios", "asymmetry",
        "temporal", "all"
    ]
    all_bands = ["delta", "theta", "alpha", "beta", "gamma"]
    
    return {
        "features": {cat: {"available": False, "last_modified": None} for cat in all_categories},
        "bands": {band: {"available": False, "last_modified": None} for band in all_bands},
    }


def detect_feature_availability(features_dir) -> dict:
    """Detect available feature categories and bands with modification timestamps."""
    from datetime import datetime
    
    features_dir = Path(features_dir)
    result = {
        "features": {},
        "bands": {},
    }
    
    category_patterns = {
        "power": ["features_power*.tsv", "features_all.tsv"],
        "connectivity": ["features_connectivity*"],
        "aperiodic": ["features_aperiodic*.tsv"],
        "erp": ["features_erp*.tsv"],
        "bursts": ["features_bursts*.tsv"],
        "itpc": ["features_itpc*.tsv"],
        "pac": ["features_pac*.tsv"],
        "complexity": ["features_complexity*.tsv"],
        "quality": ["features_quality*.tsv"],
        "erds": ["features_erds*.tsv"],
        "spectral": ["features_spectral*.tsv"],
        "ratios": ["features_ratios*.tsv"],
        "asymmetry": ["features_asymmetry*.tsv"],
        "temporal": ["features_temporal*.tsv"],
        "all": ["features_all.tsv"],
    }
    
    bands = {"delta", "theta", "alpha", "beta", "gamma"}
    band_times = {}
    
    for category, patterns in category_patterns.items():
        found_file = None
        for pattern in patterns:
            files = list(features_dir.glob(pattern))
            if files:
                found_file = max(files, key=lambda f: f.stat().st_mtime)
                break
        
        if found_file and found_file.exists():
            # Use UTC to avoid timezone issues with TUI
            mtime = datetime.fromtimestamp(found_file.stat().st_mtime, tz=None)
            mtime_utc = datetime.utcfromtimestamp(found_file.stat().st_mtime)
            mtime_str = mtime_utc.isoformat() + "Z"
            
            result["features"][category] = {
                "available": True,
                "last_modified": mtime_str,
            }
            
            try:
                with open(found_file, "r") as f:
                    header = f.readline().strip()
                for col in header.split("\t"):
                    col_lower = col.lower()
                    for band in bands:
                        if f"_{band}_" in col_lower or col_lower.endswith(f"_{band}"):
                            if band not in band_times or mtime_str > band_times[band]:
                                band_times[band] = mtime_str
            except Exception:
                pass
        else:
            result["features"][category] = {"available": False, "last_modified": None}
    
    for band in bands:
        if band in band_times:
            result["bands"][band] = {"available": True, "last_modified": band_times[band]}
        else:
            result["bands"][band] = {"available": False, "last_modified": None}
    
    return result


__all__ = [
    "detect_available_bands",
    "detect_feature_availability",
    "_empty_feature_availability",
    "BEHAVIOR_COMPUTATIONS",
    "FEATURE_VISUALIZE_CATEGORIES",
    "BEHAVIOR_VISUALIZE_CATEGORIES",
    "FREQUENCY_BANDS",
    "FEATURE_CATEGORIES",
]
