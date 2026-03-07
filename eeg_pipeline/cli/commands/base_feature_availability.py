"""Feature availability helpers shared across CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection, Iterable, List, Set, Union

from eeg_pipeline.pipelines.constants import (
    BEHAVIOR_COMPUTATIONS,
    FREQUENCY_BANDS,
)

logger = logging.getLogger(__name__)

###################################################################
# Shared helpers
###################################################################

def _find_bands_in_columns(
    columns: Iterable[str],
    candidate_bands: Collection[str],
) -> Set[str]:
    """Return bands whose name pattern appears in any of the columns."""
    found_bands: Set[str] = set()
    for column in columns:
        column_lower = column.lower()
        for band in candidate_bands:
            if f"_{band}_" in column_lower or column_lower.endswith(f"_{band}"):
                found_bands.add(band)
    return found_bands


_FEATURE_AVAILABILITY_CATEGORIES = [
    "power",
    "connectivity",
    "directedconnectivity",
    "sourcelocalization",
    "aperiodic",
    "erp",
    "bursts",
    "itpc",
    "pac",
    "complexity",
    "quality",
    "erds",
    "spectral",
    "ratios",
    "asymmetry",
    "microstates",
]


###################################################################
# Public API
###################################################################


def _read_parquet_columns_only(path: Path) -> List[str]:
    """Read only column names from parquet file without loading data."""
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(path)
    return [col for col in parquet_file.schema_arrow.names]


def detect_available_bands(features_dir: Union[str, Path]) -> List[str]:
    """Detect available frequency bands from feature file columns."""
    bands = set(FREQUENCY_BANDS)
    found_bands: Set[str] = set()

    features_path = Path(features_dir)
    for feature_file in features_path.rglob("features_*.parquet"):
        try:
            columns = _read_parquet_columns_only(feature_file)
            found_bands.update(_find_bands_in_columns(columns, bands))
        except (OSError, ValueError, ImportError):
            continue

    return sorted(found_bands)


def _empty_feature_availability() -> dict:
    """Return feature availability dict with all features marked as unavailable."""
    all_bands = FREQUENCY_BANDS
    all_computations = BEHAVIOR_COMPUTATIONS

    return {
        "features": {
            category: {"available": False, "last_modified": None}
            for category in _FEATURE_AVAILABILITY_CATEGORIES
        },
        "bands": {
            band: {"available": False, "last_modified": None}
            for band in all_bands
        },
        "computations": {
            computation: {"available": False, "last_modified": None}
            for computation in all_computations
        },
    }


def detect_feature_availability(features_dir: Union[str, Path]) -> dict:
    """Detect available feature categories, bands, and computations with modification timestamps."""
    from datetime import datetime
    
    features_path = Path(features_dir)
    result = {
        "features": {},
        "bands": {},
        "computations": {},
    }
    
    category_patterns = {
        "power": ["features_power*.parquet"],
        "connectivity": ["features_connectivity*.parquet"],
        "directedconnectivity": ["features_directedconnectivity*.parquet"],
        "sourcelocalization": ["features_sourcelocalization*.parquet"],
        "aperiodic": ["features_aperiodic*.parquet"],
        "erp": ["features_erp*.parquet"],
        "bursts": ["features_bursts*.parquet"],
        "itpc": ["features_itpc*.parquet"],
        "pac": ["features_pac*.parquet"],
        "complexity": ["features_complexity*.parquet"],
        "quality": ["features_quality*.parquet"],
        "erds": ["features_erds*.parquet"],
        "spectral": ["features_spectral*.parquet"],
        "ratios": ["features_ratios*.parquet"],
        "asymmetry": ["features_asymmetry*.parquet"],
        "microstates": ["features_microstates*.parquet"],
    }
    
    bands = set(FREQUENCY_BANDS)
    band_times = {}
    
    for category, patterns in category_patterns.items():
        found_file = None
        if features_path.exists():
            subfolder_path = features_path / category
            if subfolder_path.exists():
                for pattern in patterns:
                    files = list(subfolder_path.rglob(pattern))
                    if files:
                        found_file = max(files, key=lambda f: f.stat().st_mtime)
                        break
        
        if found_file:
            mtime_utc = datetime.utcfromtimestamp(found_file.stat().st_mtime)
            mtime_str = mtime_utc.isoformat() + "Z"
            
            result["features"][category] = {
                "available": True,
                "last_modified": mtime_str,
            }

            try:
                header_columns = _read_parquet_columns_only(found_file)
                bands_in_file = _find_bands_in_columns(header_columns, bands)
                for band in bands_in_file:
                    if band not in band_times or mtime_str > band_times[band]:
                        band_times[band] = mtime_str
            except (OSError, ValueError, ImportError) as exc:
                logger.debug(
                    "Failed to read columns from parquet feature file %s: %s",
                    found_file,
                    exc,
                )
        else:
            result["features"][category] = {"available": False, "last_modified": None}
    
    for band in bands:
        if band in band_times:
            result["bands"][band] = {"available": True, "last_modified": band_times[band]}
        else:
            result["bands"][band] = {"available": False, "last_modified": None}
    
    stats_dir = features_path.parent / "stats"
    computation_patterns = {
        "trial_table": ["trial_table*/*/trials_*.tsv", "trial_table*/*/trials_*.parquet"],
        "predictor_residual": ["predictor_residual*/*/trials_with_residual*.tsv", "predictor_residual*/*/*.metadata.json"],
        "regression": [
            "trialwise_regression*/*/regression_feature_effects*.parquet",
            "trialwise_regression*/*/regression_feature_effects*.tsv",
        ],
        "correlations": [
            "correlations*/*/correlations*.parquet",
            "correlations*/*/correlations*.tsv",
            "*_topomap_*_correlations_*.tsv",
        ],
        "condition": [
            "condition_effects*/*/condition_effects*.parquet",
            "condition_effects*/*/condition_effects*.tsv",
        ],
        "temporal": [
            "temporal_correlations*/*/temporal_correlations_*.parquet",
            "temporal_correlations*/*/temporal_correlations_*.tsv",
            "temporal_correlations*/*/normalized_results*.parquet",
            "temporal_correlations*/*/normalized_results*.tsv",
            "temporal_correlations*/*/corr_stats_temporal_*.tsv",
            "temporal_correlations*/*/corr_stats_temporal_combined*.tsv",
            "temporal_correlations*/*/corr_stats_tf_*.tsv",
            "temporal_correlations*/*/tf_grid_*.tsv",
            "temporal_correlations*/*/temporal_correlations_by_condition*.npz",
        ],
        "cluster": ["cluster*/*/cluster_results_*.tsv", "cluster*/*/null_distribution_*.json"],
    }
    
    for comp, patterns in computation_patterns.items():
        found_file = None
        if stats_dir.exists():
            for pattern in patterns:
                files = list(stats_dir.rglob(pattern))
                if comp == "correlations":
                    files = [
                        f for f in files
                        if "temporal" not in f.name.lower()
                        and not f.name.startswith("corr_stats_temporal")
                        and not f.name.startswith("temporal_correlations")
                    ]
                if files:
                    found_file = max(files, key=lambda f: f.stat().st_mtime)
                    break
        
        if found_file:
            mtime_utc = datetime.utcfromtimestamp(found_file.stat().st_mtime)
            mtime_str = mtime_utc.isoformat() + "Z"
            result["computations"][comp] = {
                "available": True,
                "last_modified": mtime_str,
            }
        else:
            result["computations"][comp] = {"available": False, "last_modified": None}
    
    return result
