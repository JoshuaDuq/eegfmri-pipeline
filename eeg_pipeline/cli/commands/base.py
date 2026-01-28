"""Shared utilities and constants for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterable, List, Optional, Set, Union

import pandas as pd

from eeg_pipeline.pipelines.constants import (
    BEHAVIOR_COMPUTATIONS,
    BEHAVIOR_VISUALIZE_CATEGORIES,
    FEATURE_VISUALIZE_CATEGORIES,
    FREQUENCY_BANDS,
)


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
            except (OSError, ValueError, ImportError):
                pass
        else:
            result["features"][category] = {"available": False, "last_modified": None}
    
    for band in bands:
        if band in band_times:
            result["bands"][band] = {"available": True, "last_modified": band_times[band]}
        else:
            result["bands"][band] = {"available": False, "last_modified": None}
    
    stats_dir = features_path.parent / "stats"
    computation_patterns = {
        "trial_table": ["trial_table*/*/trials*.tsv", "trial_table*/*/trials*.parquet"],
        "lag_features": ["lag_features*/*/trials_with_lags*.tsv", "lag_features*/*/*.metadata.json"],
        "pain_residual": ["pain_residual*/*/trials_with_residual*.tsv", "pain_residual*/*/*.metadata.json"],
        "temperature_models": [
            "temperature_models*/*/model_comparison*.parquet",
            "temperature_models*/*/model_comparison*.tsv",
            "temperature_models*/*/breakpoint_candidates*.parquet",
            "temperature_models*/*/breakpoint_candidates*.tsv",
        ],
        "regression": [
            "trialwise_regression*/*/regression_feature_effects*.parquet",
            "trialwise_regression*/*/regression_feature_effects*.tsv",
        ],
        "models": [
            "feature_models*/*/models_feature_effects*.parquet",
            "feature_models*/*/models_feature_effects*.tsv",
        ],
        "stability": [
            "stability_groupwise*/*/stability_groupwise*.parquet",
            "stability_groupwise*/*/stability_groupwise*.tsv",
        ],
        "consistency": [
            "consistency_summary*/*/consistency_summary*.parquet",
            "consistency_summary*/*/consistency_summary*.tsv",
        ],
        "influence": [
            "influence_diagnostics*/*/influence_diagnostics*.parquet",
            "influence_diagnostics*/*/influence_diagnostics*.tsv",
        ],
        "report": ["subject_report*/*/subject_report*.md", "subject_report*/*/subject_report*.html"],
        "correlations": [
            "correlations*/*/correlations*.parquet",
            "correlations*/*/correlations*.tsv",
            "*_topomap_*_correlations_*.tsv",
        ],
        "pain_sensitivity": [
            "pain_sensitivity*/*/pain_sensitivity*.parquet",
            "pain_sensitivity*/*/pain_sensitivity*.tsv",
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
        "mediation": ["mediation*/*/mediation*.tsv", "mediation*/*/mediation*.parquet"],
        "moderation": ["moderation*/*/moderation_results*.tsv", "moderation*/*/moderation_results*.parquet"],
        "mixed_effects": ["mixed_effects*/*/mixed_effects*.tsv", "mixed_effects*/*/mixed_effects*.parquet"],
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


def discover_event_columns(
    bids_root: Union[str, Path],
    task: Optional[str] = None,
    subject: Optional[str] = None,
    deriv_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Discover available columns and their unique values from clean events files.
    
    Only searches for clean events files (*proc-clean_events.tsv) in derivatives directory.
    Does not fall back to regular events files from BIDS root.
    
    Returns
    -------
    dict
        {
            "columns": ["onset", "duration", "trial_type", "condition", ...],
            "values": {
                "condition": ["pain", "nonpain"],
                "trial_type": ["stim", "rating"],
                ...
            },
            "source": "events" | "metadata",
            "file": "path/to/file"
        }
    """
    result = {"columns": [], "values": {}, "source": None, "file": None}
    
    if not deriv_root:
        return result
    
    deriv_root = Path(deriv_root)
    events_file = None
    
    clean_patterns = [
        f"*task-{task}*proc-clean*_events.tsv",
        f"*task-{task}*proc-clean_events.tsv",
        "*proc-clean*_events.tsv",
    ] if task else ["*proc-clean*_events.tsv"]
    
    if subject:
        subj_id = subject.replace("sub-", "")
        search_dirs = [
            deriv_root / "preprocessed" / "eeg" / f"sub-{subj_id}" / "eeg",
            deriv_root / "preprocessed" / "eeg" / f"sub-{subj_id}",
            deriv_root / f"sub-{subj_id}" / "eeg",
            deriv_root / f"sub-{subj_id}",
        ]
    else:
        search_dirs = [
            deriv_root / "preprocessed" / "eeg",
            deriv_root,
        ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in clean_patterns:
            files = list(search_dir.rglob(pattern))
            if files:
                events_file = files[0]
                break
        if events_file:
            break
    
    if not events_file:
        return result
    
    try:
        df = pd.read_csv(events_file, sep="\t")
        result["columns"] = df.columns.tolist()
        result["source"] = "events"
        result["file"] = str(events_file)
        
        skip_columns = {"onset", "duration", "sample", "value", "stim_file"}
        for col in df.columns:
            if col.lower() in skip_columns:
                continue
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:
                vals = [str(v) for v in unique_vals if pd.notna(v)]
                result["values"][col] = sorted(set(vals))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        pass
    
    return result


def discover_trial_table_columns(
    deriv_root: Union[str, Path],
    subject: Optional[str] = None,
) -> dict:
    """Discover columns from an existing trial table.
    
    This provides more detailed columns after behavior compute has run.
    """
    deriv_root = Path(deriv_root)
    result = {"columns": [], "values": {}, "source": None, "file": None}
    
    trial_file = None
    patterns = ["trial_table*/*/trials*.tsv", "trial_table*/*/trials*.parquet"]
    
    search_dirs = []
    if subject:
        subj_id = subject.replace("sub-", "")
        search_dirs.append(deriv_root / f"sub-{subj_id}" / "eeg" / "stats")
    else:
        search_dirs.extend(sorted(deriv_root.glob("sub-*/eeg/stats"))[:5])
    
    for stats_dir in search_dirs:
        if not stats_dir.exists():
            continue
        for pattern in patterns:
            files = list(stats_dir.glob(pattern))
            if files:
                trial_file = files[0]
                break
        if trial_file:
            break
    
    if not trial_file:
        return result
    
    try:
        if trial_file.suffix == ".parquet":
            import pyarrow.parquet as pq
            # Use increased thrift limit for large trial tables with many columns
            pf = pq.ParquetFile(
                trial_file,
                memory_map=False,
                thrift_string_size_limit=500_000_000,
            )
            # Read only first 500 rows for discovery
            table = pf.read_row_group(0) if pf.num_row_groups > 0 else pf.read()
            df = table.slice(0, 500).to_pandas()
        else:
            df = pd.read_csv(trial_file, sep="\t", nrows=500)
        result["columns"] = df.columns.tolist()
        result["source"] = "trial_table"
        result["file"] = str(trial_file)
        
        skip_columns = {"onset", "duration", "sample", "value", "epoch", "trial"}
        for col in df.columns:
            if col.lower() in skip_columns:
                continue
            if df[col].dtype in ["float64", "float32"] and df[col].nunique() > 20:
                continue
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:
                vals = [str(v) for v in unique_vals if pd.notna(v)]
                result["values"][col] = sorted(set(vals))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
        pass
    
    return result


def discover_condition_effects_columns(
    deriv_root: Union[str, Path],
    subject: Optional[str] = None,
) -> dict:
    """Discover columns and values from condition effects files.
    
    Reads condition_effects_column*.parquet and condition_effects_window*.parquet files
    to extract available condition columns and their values for plotting.
    
    Returns
    -------
    dict
        {
            "columns": ["condition_column1", "condition_column2", ...],
            "values": {
                "condition_column1": ["value1", "value2"],
                "condition_column2": ["value1", "value2"],
            },
            "windows": ["window1", "window2", ...],  # Available windows from window files
            "source": "condition_effects",
            "files": ["path/to/file1.parquet", ...]
        }
    """
    import pandas as pd
    from eeg_pipeline.infra.tsv import read_parquet
    
    deriv_root = Path(deriv_root)
    result = {"columns": [], "values": {}, "windows": [], "source": "condition_effects", "files": []}
    
    condition_effects_dirs: List[Path] = []
    if subject:
        subj_id = subject.replace("sub-", "")
        # Try multiple possible paths
        possible_paths = [
            deriv_root / f"sub-{subj_id}" / "eeg" / "stats" / "condition_effects",
            deriv_root / "stats" / f"sub-{subj_id}" / "eeg" / "condition_effects",
            deriv_root / f"sub-{subj_id}" / "stats" / "condition_effects",
        ]
        for stats_dir in possible_paths:
            if stats_dir.exists():
                # New layout nests feature folders under condition_effects*/<feature>/...
                subdirs = [p for p in stats_dir.glob("*") if p.is_dir()]
                if subdirs:
                    condition_effects_dirs.extend(subdirs)
                else:
                    condition_effects_dirs.append(stats_dir)
                break
    
    if not condition_effects_dirs:
        # Try to find any condition_effects directories
        for pattern in [
            "sub-*/eeg/stats/condition_effects",
            "stats/sub-*/eeg/condition_effects",
            "sub-*/stats/condition_effects",
        ]:
            for subj_dir in sorted(deriv_root.glob(pattern))[:5]:
                if subj_dir.exists():
                    subdirs = [p for p in subj_dir.glob("*") if p.is_dir()]
                    if subdirs:
                        condition_effects_dirs.extend(subdirs)
                    else:
                        condition_effects_dirs.append(subj_dir)
            if condition_effects_dirs:
                break
    
    if not condition_effects_dirs:
        return result
    
    seen_columns = set()
    seen_windows = set()
    
    # Process column files (parquet)
    for cond_dir in condition_effects_dirs:
        for file_path in cond_dir.glob("condition_effects_column*.parquet"):
            try:
                df = read_parquet(file_path)
                if df is None or df.empty:
                    continue
                
                if "condition_column" in df.columns:
                    cond_col = df["condition_column"].iloc[0]
                    if pd.notna(cond_col):
                        cond_col = str(cond_col).strip()
                        if cond_col and cond_col not in seen_columns:
                            seen_columns.add(cond_col)
                            result["columns"].append(cond_col)
                            
                            # Extract condition values
                            if "condition_value1" in df.columns and "condition_value2" in df.columns:
                                val1 = df["condition_value1"].iloc[0]
                                val2 = df["condition_value2"].iloc[0]
                                if pd.notna(val1) and pd.notna(val2):
                                    result["values"][cond_col] = [str(val1).strip(), str(val2).strip()]
                            elif "mean_pain" in df.columns and "mean_nonpain" in df.columns:
                                # Fallback: infer from column names
                                result["values"][cond_col] = ["pain", "nonpain"]
                            
                            if file_path not in result["files"]:
                                result["files"].append(str(file_path))
                
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
                continue
    
    # Process multigroup parquet files - also contain condition columns
    import re
    for cond_dir in condition_effects_dirs:
        for file_path in cond_dir.glob("condition_effects_multigroup*.parquet"):
            try:
                df = read_parquet(file_path)
                if df is None or df.empty:
                    continue
                
                # Multigroup files use "compare_column" instead of "condition_column"
                # Also check filename: condition_effects_multigroup*_{column}.parquet
                cond_col = None
                if "compare_column" in df.columns:
                    cond_col = df["compare_column"].iloc[0]
                elif "condition_column" in df.columns:
                    cond_col = df["condition_column"].iloc[0]
                
                # Extract from filename if not in data: condition_effects_multigroup*_{column}.parquet
                if cond_col is None or pd.isna(cond_col):
                    filename = file_path.stem
                    match = re.search(r'condition_effects_multigroup[^_]*_(.+)$', filename)
                    if match:
                        cond_col = match.group(1)
                
                if cond_col is not None and pd.notna(cond_col):
                    cond_col = str(cond_col).strip()
                    if cond_col and cond_col not in seen_columns:
                        seen_columns.add(cond_col)
                        result["columns"].append(cond_col)
                        
                        # Extract condition values from group1 and group2 columns (multigroup files store pairwise comparisons)
                        group_values = set()
                        if "group1" in df.columns:
                            group1_vals = df["group1"].dropna().unique()
                            for val in group1_vals:
                                if pd.notna(val):
                                    val_str = str(val).strip()
                                    if val_str:
                                        group_values.add(val_str)
                        if "group2" in df.columns:
                            group2_vals = df["group2"].dropna().unique()
                            for val in group2_vals:
                                if pd.notna(val):
                                    val_str = str(val).strip()
                                    if val_str:
                                        group_values.add(val_str)
                        
                        if group_values:
                            result["values"][cond_col] = sorted(group_values)
                        
                        if file_path not in result["files"]:
                            result["files"].append(str(file_path))
                
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
                continue
    
    # Process window files to extract available windows
    for cond_dir in condition_effects_dirs:
        for file_path in cond_dir.glob("condition_effects_window*.parquet"):
            try:
                df = read_parquet(file_path)
                if df is None or df.empty:
                    continue
                
                # Extract windows from window1 and window2 columns
                if "window1" in df.columns:
                    windows1 = df["window1"].dropna().unique()
                    for w in windows1:
                        w_str = str(w).strip()
                        if w_str and w_str not in seen_windows:
                            seen_windows.add(w_str)
                            result["windows"].append(w_str)
                
                if "window2" in df.columns:
                    windows2 = df["window2"].dropna().unique()
                    for w in windows2:
                        w_str = str(w).strip()
                        if w_str and w_str not in seen_windows:
                            seen_windows.add(w_str)
                            result["windows"].append(w_str)
                
                if file_path not in result["files"]:
                    result["files"].append(str(file_path))
                
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
                continue
    
    result["columns"] = sorted(set(result["columns"]))
    result["windows"] = sorted(set(result["windows"]))
    return result


__all__ = [
    "detect_available_bands",
    "detect_feature_availability",
    "_empty_feature_availability",
    "discover_event_columns",
    "discover_trial_table_columns",
    "BEHAVIOR_COMPUTATIONS",
    "FEATURE_VISUALIZE_CATEGORIES",
    "BEHAVIOR_VISUALIZE_CATEGORIES",
    "FREQUENCY_BANDS",
]
