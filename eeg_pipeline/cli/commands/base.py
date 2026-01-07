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
    bands = set(FREQUENCY_BANDS)
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
        "complexity", "quality", "erds", "spectral", "ratios", "asymmetry"
    ]
    all_bands = FREQUENCY_BANDS
    all_computations = BEHAVIOR_COMPUTATIONS
    
    return {
        "features": {cat: {"available": False, "last_modified": None} for cat in all_categories},
        "bands": {band: {"available": False, "last_modified": None} for band in all_bands},
        "computations": {comp: {"available": False, "last_modified": None} for comp in all_computations},
    }


def detect_feature_availability(features_dir) -> dict:
    """Detect available feature categories, bands, and computations with modification timestamps."""
    from datetime import datetime
    
    features_dir = Path(features_dir)
    result = {
        "features": {},
        "bands": {},
        "computations": {},
    }
    
    category_patterns = {
        "power": ["features_power*.tsv"],
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
    }
    
    bands = set(FREQUENCY_BANDS)
    band_times = {}
    
    for category, patterns in category_patterns.items():
        found_file = None
        if features_dir.exists():
            for pattern in patterns:
                files = list(features_dir.glob(pattern))
                if files:
                    found_file = max(files, key=lambda f: f.stat().st_mtime)
                    break
        
        if found_file and found_file.exists():
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
    
    # Try to find stats adjacent to features
    stats_dir = features_dir.parent / "stats"
    computation_patterns = {
        "trial_table": ["trials*.tsv", "trials.parquet", "trial_table*.tsv"],
        "confounds": ["confounds_audit*.tsv"],
        "regression": ["regression_*.tsv", "trialwise_regression*.tsv", "regression_feature_effects*.tsv"],
        "models": ["model_*.tsv", "sensitivity_*.tsv", "temperature_model_comparison.tsv"],
        "stability": ["stability_*.tsv", "temperature_breakpoint_test.tsv"],
        "consistency": ["consistency_*.tsv"],
        "influence": ["influence_*.tsv", "cooks_*.tsv", "influence_diagnostics*.tsv"],
        "report": ["subject_report*.html", "subject_report*.pdf", "subject_report*.md"],
        "correlations": ["correlations*.tsv", "power_topomap_temperature_correlations*.tsv"],
        "pain_sensitivity": ["pain_sensitivity*.tsv"],
        "condition": ["condition_*.tsv", "condition_effects.tsv"],
        "temporal": ["temporal_*.tsv", "corr_stats_temporal*.tsv"],
        "cluster": ["cluster_*.tsv", "pain_nonpain_time_clusters*.tsv", "null_distribution_*.json"],
        "mediation": ["mediation*.tsv"],
        "mixed_effects": ["mixed_effects*.tsv", "lme_*.tsv"],
    }
    
    for comp, patterns in computation_patterns.items():
        found_file = None
        if stats_dir.exists():
            for pattern in patterns:
                # Use rglob to search recursively in subdirectories
                files = list(stats_dir.rglob(pattern))
                # Filter out temporal correlation files from regular correlations check
                if comp == "correlations":
                    files = [f for f in files if "temporal" not in f.name.lower() and not f.name.startswith("corr_stats_temporal")]
                if files:
                    found_file = max(files, key=lambda f: f.stat().st_mtime)
                    break
        
        if found_file and found_file.exists():
            mtime_utc = datetime.utcfromtimestamp(found_file.stat().st_mtime)
            mtime_str = mtime_utc.isoformat() + "Z"
            result["computations"][comp] = {
                "available": True,
                "last_modified": mtime_str,
            }
        else:
            result["computations"][comp] = {"available": False, "last_modified": None}
    
    return result


def discover_event_columns(bids_root, task: str = None, subject: str = None) -> dict:
    """Discover available columns and their unique values from events files.
    
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
    from pathlib import Path
    import pandas as pd
    
    bids_root = Path(bids_root)
    result = {"columns": [], "values": {}, "source": None, "file": None}
    
    events_file = None
    
    if subject:
        subj_id = subject.replace("sub-", "")
        subj_dir = bids_root / f"sub-{subj_id}"
        if subj_dir.exists():
            eeg_dir = subj_dir / "eeg"
            if eeg_dir.exists():
                patterns = [f"*task-{task}*_events.tsv", "*_events.tsv"] if task else ["*_events.tsv"]
                for pattern in patterns:
                    files = list(eeg_dir.glob(pattern))
                    if files:
                        events_file = files[0]
                        break
    
    if not events_file:
        for subj_dir in sorted(bids_root.glob("sub-*"))[:5]:
            eeg_dir = subj_dir / "eeg"
            if not eeg_dir.exists():
                continue
            patterns = [f"*task-{task}*_events.tsv", "*_events.tsv"] if task else ["*_events.tsv"]
            for pattern in patterns:
                files = list(eeg_dir.glob(pattern))
                if files:
                    events_file = files[0]
                    break
            if events_file:
                break
    
    if not events_file or not events_file.exists():
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
    except Exception:
        pass
    
    return result


def discover_trial_table_columns(deriv_root, subject: str = None) -> dict:
    """Discover columns from an existing trial table.
    
    This provides more detailed columns after behavior compute has run.
    """
    from pathlib import Path
    import pandas as pd
    
    deriv_root = Path(deriv_root)
    result = {"columns": [], "values": {}, "source": None, "file": None}
    
    trial_file = None
    
    if subject:
        subj_id = subject.replace("sub-", "")
        stats_dir = deriv_root / "stats" / f"sub-{subj_id}"
        if stats_dir.exists():
            for pattern in ["trials*.tsv", "trial_table*.tsv"]:
                files = list(stats_dir.glob(pattern))
                if files:
                    trial_file = files[0]
                    break
    
    if not trial_file:
        for stats_dir in sorted(deriv_root.glob("stats/sub-*"))[:5]:
            for pattern in ["trials*.tsv", "trial_table*.tsv"]:
                files = list(stats_dir.glob(pattern))
                if files:
                    trial_file = files[0]
                    break
            if trial_file:
                break
    
    if not trial_file or not trial_file.exists():
        return result
    
    try:
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
    except Exception:
        pass
    
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
    "FEATURE_CATEGORIES",
]
