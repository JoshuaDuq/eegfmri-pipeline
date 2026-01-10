"""Shared utilities and constants for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterable, List, Optional, Set, Union

from eeg_pipeline.pipelines.constants import (
    BEHAVIOR_COMPUTATIONS,
    BEHAVIOR_VISUALIZE_CATEGORIES,
    FEATURE_CATEGORIES,
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
    "directed_connectivity",
    "source_localization",
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


def detect_available_bands(features_dir: Union[str, Path]) -> List[str]:
    """Detect available frequency bands from feature file columns."""
    bands = set(FREQUENCY_BANDS)
    found_bands: Set[str] = set()

    features_path = Path(features_dir)
    for tsv_file in features_path.rglob("features_*.tsv"):
        try:
            with open(tsv_file, "r") as f:
                header = f.readline().strip()
            columns = header.split("\t")
            found_bands.update(_find_bands_in_columns(columns, bands))
        except Exception:
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
        "power": ["features_power*.tsv"],
        "connectivity": ["features_connectivity*"],
        "directed_connectivity": ["features_directed_connectivity*"],
        "source_localization": ["features_source_localization*"],
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
        if features_path.exists():
            for pattern in patterns:
                subfolder_path = features_path / category
                if subfolder_path.exists():
                    files = list(subfolder_path.glob(pattern))
                    if files:
                        found_file = max(files, key=lambda f: f.stat().st_mtime)
                        break
                if not found_file:
                    files = list(features_path.glob(pattern))
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
                header_columns = header.split("\t")
                bands_in_file = _find_bands_in_columns(header_columns, bands)
                for band in bands_in_file:
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
    
    stats_dir = features_path.parent / "stats"
    # Patterns are searched via rglob within stats_dir, so they can match nested subfolders.
    # Keep these aligned with outputs produced by eeg_pipeline.analysis.behavior.orchestration.
    # Subfolder structure: stats/<computation_type>/<filename>.tsv
    computation_patterns = {
        # trial_table/ subfolder
        "trial_table": ["trial_table/trials*.tsv", "trial_table/trials*.parquet", "trials*.tsv", "trials*.parquet"],
        # lag_features/ subfolder  
        "lag_features": ["lag_features/trials_with_lags*.tsv", "lag_features/*.metadata.json", "trials_with_lags*.tsv"],
        # pain_residual/ subfolder
        "pain_residual": ["pain_residual/trials_with_residual*.tsv", "pain_residual/*.metadata.json", "trials_with_residual*.tsv"],
        # temperature_models/ subfolder
        "temperature_models": [
            "temperature_models/model_comparison*.tsv",
            "temperature_models/breakpoint*.tsv",
            "model_comparison*.tsv",
            "breakpoint_candidates*.tsv",
        ],
        # confounds_audit/ subfolder
        "confounds": ["confounds_audit/confounds_audit*.tsv", "confounds_audit/*.metadata.json", "confounds_audit*.tsv"],
        # trialwise_regression/ subfolder
        "regression": ["trialwise_regression/regression_feature_effects*.tsv", "regression_feature_effects*.tsv"],
        # feature_models/ subfolder
        "models": ["feature_models/models_feature_effects*.tsv", "models_feature_effects*.tsv"],
        # stability_groupwise/ subfolder
        "stability": ["stability_groupwise/stability_groupwise*.tsv", "stability_groupwise*.tsv"],
        # consistency_summary/ subfolder
        "consistency": ["consistency_summary/consistency_summary*.tsv", "consistency_summary*.tsv"],
        # influence_diagnostics/ subfolder
        "influence": ["influence_diagnostics/influence_diagnostics*.tsv", "influence_diagnostics*.tsv"],
        # subject_report/ subfolder
        "report": ["subject_report/subject_report*.md", "subject_report*.md", "subject_report*.html"],
        # correlations/ subfolder
        "correlations": [
            "correlations/correlations*.tsv",
            "correlations*.tsv",
            "*_topomap_*_correlations_*.tsv",
        ],
        # pain_sensitivity/ subfolder
        "pain_sensitivity": ["pain_sensitivity/pain_sensitivity*.tsv", "pain_sensitivity*.tsv"],
        # condition_effects/ subfolder
        "condition": ["condition_effects/condition_effects*.tsv", "condition_effects*.tsv"],
        # temporal_correlations/ subfolder
        "temporal": [
            "temporal_correlations/corr_stats_temporal_*.tsv",
            "corr_stats_temporal_*.tsv",
            "corr_stats_temporal_combined*.tsv",
            "corr_stats_tf_*.tsv",
        ],
        # cluster/ subfolder
        "cluster": ["cluster/cluster_results_*.tsv", "cluster_results_*.tsv", "cluster_*.tsv", "null_distribution_*.json"],
        # mediation/ subfolder
        "mediation": ["mediation/mediation*.tsv", "mediation*.tsv"],
        # moderation/ subfolder
        "moderation": ["moderation/moderation_results*.tsv", "moderation_results*.tsv"],
        # mixed_effects/ subfolder
        "mixed_effects": ["mixed_effects/mixed_effects*.tsv", "mixed_effects*.tsv", "lme_*.tsv"],
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


def discover_event_columns(
    bids_root: Union[str, Path],
    task: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict:
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


def discover_trial_table_columns(
    deriv_root: Union[str, Path],
    subject: Optional[str] = None,
) -> dict:
    """Discover columns from an existing trial table.
    
    This provides more detailed columns after behavior compute has run.
    """
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
