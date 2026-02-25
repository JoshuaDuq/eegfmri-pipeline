"""Column discovery helpers shared across CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)

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
            "columns": ["onset", "duration", "condition", "event_type", ...],
            "values": {
                "condition": ["condition_1", "condition_2"],
                "event_type": ["stim", "outcome"],
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
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.debug("Failed to read events file for discovery (%s): %s", events_file, exc)
    
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
    
    trial_files: List[Path] = []
    
    search_dirs = []
    if subject:
        subj_id = subject.replace("sub-", "")
        search_dirs.append(deriv_root / f"sub-{subj_id}" / "eeg" / "stats")
    else:
        search_dirs.extend(sorted(deriv_root.glob("sub-*/eeg/stats"))[:5])
    
    for stats_dir in search_dirs:
        if not stats_dir.exists():
            continue
        from eeg_pipeline.utils.data.trial_table import (
            discover_trial_table_candidates,
            select_preferred_trial_tables,
        )

        found_in_dir = select_preferred_trial_tables(
            discover_trial_table_candidates(stats_dir)
        )

        if found_in_dir:
            trial_files = sorted(set(found_in_dir))
            break
    
    if not trial_files:
        return result
    
    columns: List[str] = []
    seen_columns: Set[str] = set()
    value_sets: dict[str, Set[str]] = {}
    skip_columns = {"onset", "duration", "sample", "value", "epoch", "trial"}

    for trial_file in trial_files:
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

            for col in df.columns:
                if col not in seen_columns:
                    seen_columns.add(col)
                    columns.append(col)

            for col in df.columns:
                if col.lower() in skip_columns:
                    continue
                if df[col].dtype in ["float64", "float32"] and df[col].nunique() > 20:
                    continue
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 50:
                    vals = {str(v) for v in unique_vals if pd.notna(v)}
                    if vals:
                        value_sets.setdefault(col, set()).update(vals)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.debug("Failed to read trial table for discovery (%s): %s", trial_file, exc)
        except Exception as exc:
            logger.warning("Unexpected error while discovering trial table columns (%s): %s", trial_file, exc)

    if columns:
        result["columns"] = columns
        result["values"] = {col: sorted(vals) for col, vals in value_sets.items()}
        result["source"] = "trial_table"
        result["file"] = str(trial_files[0])
    
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
