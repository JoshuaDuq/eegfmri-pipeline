"""
Efficient Feature Extraction Pipeline
======================================

This module provides an efficient orchestrator for extracting multiple
feature types from EEG epochs. It achieves efficiency by:

1. **Precomputing expensive intermediates once** (band filtering, PSD, GFP)
2. **Passing precomputed data** to individual extractors
3. **Organizing extraction into logical groups** (erds, spectral, etc.)
4. **Supporting selective extraction** via feature_groups parameter

Architecture
------------
```
epochs
   │
   ▼
precompute_data()  ─────► PrecomputedData
   │                       ├── band_data (filtered + envelope + phase)
   │                       ├── psd_data (Welch PSD)
   │                       ├── gfp (global field power)
   │                       └── windows (time masks)
   │
   ▼
_extract_*_from_precomputed()  ─► FeatureSet
   │
   ▼
ExtractionResult (combined)
```

Usage
-----
```python
result = extract_precomputed_features(
    epochs, bands=["alpha", "beta"], config=config, logger=logger,
    feature_groups=["erds", "spectral", "roi"],
)
df = result.get_combined_df()
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.analysis.features.core import (
    PrecomputedData,
    FeatureExtractionContext,
    precompute_data,
    EPSILON_STD,
    MIN_CHANNELS_FOR_CONNECTIVITY,
    MIN_VALID_FRACTION,
    MIN_EPOCHS_FOR_FEATURES,
    MIN_EPOCHS_FOR_PLV,
    MIN_EDGE_SAMPLES,
    compute_psd,
)
from eeg_pipeline.analysis.features.aperiodic import extract_aperiodic_features
from eeg_pipeline.analysis.features.manifest import generate_manifest, save_features_organized
from eeg_pipeline.analysis.features.naming import make_power_name
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.arrays import (
    safe_nanmean,
    safe_nanstd,
    safe_divide,
    mask_valid,
    count_valid,
)


###################################################################
# Internal helpers
###################################################################

# Empty result constants for cleaner returns
_EMPTY_RESULT = (pd.DataFrame(), [])
_EMPTY_RESULT_WITH_QC = (pd.DataFrame(), [], {})


def _validate_window_masks(
    precomputed: PrecomputedData,
    logger: Any = None,
    *,
    require_baseline: bool = True,
    require_active: bool = True,
) -> bool:
    """
    Ensure baseline/active masks exist and contain samples.

    Returns False and logs a warning when masks are missing or empty so that
    feature extractors can fail fast instead of producing all-NaN outputs.
    """
    windows = precomputed.windows
    if windows is None:
        if logger:
            logger.warning("Time windows are missing; skipping feature extraction.")
        return False

    if require_baseline:
        baseline_mask = getattr(windows, "baseline_mask", None)
        if baseline_mask is None or not np.any(baseline_mask):
            if logger:
                logger.warning(
                    "Baseline window is empty; configured/used range: %s. Skipping feature extraction.",
                    getattr(windows, "baseline_range", None),
                )
            return False

    if require_active:
        active_mask = getattr(windows, "active_mask", None)
        if active_mask is None or not np.any(active_mask):
            if logger:
                logger.warning(
                    "Active window is empty; configured/used range: %s. Skipping feature extraction.",
                    getattr(windows, "active_range", None),
                )
            return False

    return True


def _nanmean_with_fraction(data: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int, int]:
    """Compute NaN-safe mean inside a mask and report finite fractions."""
    masked = data[mask]
    total = int(masked.size)
    finite_mask = np.isfinite(masked)
    valid = int(np.sum(finite_mask))
    mean_val = float(np.nanmean(masked)) if valid > 0 else np.nan
    frac = float(valid / total) if total > 0 else 0.0
    return mean_val, frac, valid, total


def _validate_precomputed_for_extraction(
    precomputed: PrecomputedData,
    *,
    require_bands: bool = True,
    require_baseline: bool = True,
    require_active: bool = True,
    min_epochs: int = 1,
    context: str = "",
) -> Optional[str]:
    """
    Validate precomputed data for feature extraction.
    
    Returns None if valid, or an error message string if invalid.
    """
    if precomputed is None:
        return f"{context}: precomputed data is None"
    
    if precomputed.data.size == 0:
        return f"{context}: empty data"
    
    n_epochs = precomputed.data.shape[0]
    if n_epochs < min_epochs:
        return f"{context}: only {n_epochs} epochs (min={min_epochs})"
    
    if require_bands and not precomputed.band_data:
        return f"{context}: no band data available"
    
    if precomputed.windows is None:
        return f"{context}: time windows missing"
    
    if require_baseline:
        if precomputed.windows.baseline_mask is None or not np.any(precomputed.windows.baseline_mask):
            return f"{context}: baseline window empty"
    
    if require_active:
        if precomputed.windows.active_mask is None or not np.any(precomputed.windows.active_mask):
            return f"{context}: active window empty"
    
    return None


def _build_records_to_df(records: List[Dict[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
    """Convert list of feature records to DataFrame with sorted columns."""
    if not records:
        return pd.DataFrame(), []
    columns = sorted(records[0].keys())
    return pd.DataFrame(records)[columns], columns


###################################################################
# Result Containers
###################################################################


@dataclass
class FeatureSet:
    """
    Container for a single feature group's extraction results.
    
    Attributes
    ----------
    df : pd.DataFrame
        Feature values, shape (n_epochs, n_features)
    columns : List[str]
        Column names for the features
    name : str
        Name of this feature group (e.g., "erds", "spectral")
    """
    df: pd.DataFrame
    columns: List[str]
    name: str


@dataclass
class ExtractionResult:
    """
    Container for all extracted feature groups, epoch-aligned with condition labels.
    
    Each feature group produces a DataFrame with one row per epoch.
    If events_df is provided, a 'condition' column is added indicating
    'pain' or 'nonpain' for each trial.
    
    Attributes
    ----------
    features : Dict[str, FeatureSet]
        Mapping from feature group name to FeatureSet
    precomputed : PrecomputedData
        The precomputed intermediates (can be reused)
    condition : Optional[np.ndarray]
        Array of condition labels ('pain' or 'nonpain') per epoch
    """
    features: Dict[str, FeatureSet] = field(default_factory=dict)
    precomputed: Optional[PrecomputedData] = None
    condition: Optional[np.ndarray] = None
    qc: Dict[str, Any] = field(default_factory=dict)
    
    def get_combined_df(self, include_condition: bool = True) -> pd.DataFrame:
        """
        Combine all feature sets into a single DataFrame.
        
        Parameters
        ----------
        include_condition : bool
            If True and condition labels exist, adds 'condition' column.
        
        Returns
        -------
        pd.DataFrame
            Combined features with one row per epoch.
        """
        if not self.features:
            return pd.DataFrame()
        
        dfs = [fs.df for fs in self.features.values() if not fs.df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, axis=1)
        
        # Add condition column if available
        if include_condition and self.condition is not None:
            combined.insert(0, "condition", self.condition)
        
        # Stable column order for reproducibility (condition first, then sorted)
        fixed_cols = ["condition"] if "condition" in combined.columns else []
        other_cols = sorted([c for c in combined.columns if c not in fixed_cols])
        return combined[fixed_cols + other_cols] if fixed_cols else combined[other_cols]
    
    def get_feature_group_df(self, group: str, include_condition: bool = True) -> pd.DataFrame:
        """Get DataFrame for a specific feature group."""
        if group not in self.features:
            return pd.DataFrame()
        
        df = self.features[group].df.copy()
        
        if include_condition and self.condition is not None:
            df.insert(0, "condition", self.condition)
        
        return df
    
    def get_all_columns(self) -> List[str]:
        """Get all column names across all feature groups."""
        cols = []
        for fs in self.features.values():
            cols.extend(fs.columns)
        return cols
    
    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        if self.features:
            first_fs = next(iter(self.features.values()))
            return len(first_fs.df)
        return 0
    
    @property
    def n_pain(self) -> int:
        """Number of pain trials."""
        if self.condition is not None:
            return int(np.sum(self.condition == "pain"))
        return 0
    
    @property
    def n_nonpain(self) -> int:
        """Number of non-pain trials."""
        if self.condition is not None:
            return int(np.sum(self.condition == "nonpain"))
        return 0
    
    def __repr__(self) -> str:
        n_features = sum(len(fs.columns) for fs in self.features.values())
        groups = list(self.features.keys())
        if self.condition is not None:
            condition_str = f" (pain={self.n_pain}, nonpain={self.n_nonpain})"
        else:
            condition_str = ""
        return f"ExtractionResult({self.n_epochs} epochs, {n_features} features from {groups}{condition_str})"

    def get_qc_summary(self) -> Dict[str, Any]:
        """
        Return aggregated QC metrics across all feature groups.
        
        Returns
        -------
        Dict[str, Any]
            Summary including:
            - n_feature_groups: number of successfully extracted groups
            - total_features: total feature count
            - groups_with_issues: list of groups that had QC issues or were skipped
            - per_group_status: dict mapping group name to success/skip status
        """
        summary: Dict[str, Any] = {
            "n_feature_groups": len(self.features),
            "total_features": sum(len(fs.columns) for fs in self.features.values()),
            "n_epochs": self.n_epochs,
            "groups_extracted": list(self.features.keys()),
            "groups_with_issues": [],
            "per_group_status": {},
        }
        
        for name, qc_data in self.qc.items():
            if name == "precomputed":
                continue
            if isinstance(qc_data, dict):
                if qc_data.get("skipped_reason"):
                    summary["groups_with_issues"].append(name)
                    summary["per_group_status"][name] = f"skipped: {qc_data['skipped_reason']}"
                elif qc_data.get("error"):
                    summary["groups_with_issues"].append(name)
                    summary["per_group_status"][name] = f"error: {qc_data['error']}"
                else:
                    summary["per_group_status"][name] = "ok"
        
        # Add condition summary if available
        if self.condition is not None:
            summary["n_pain"] = self.n_pain
            summary["n_nonpain"] = self.n_nonpain
        
        return summary

    def build_manifest(self, config: Any = None, subject: Optional[str] = None, task: Optional[str] = None) -> Dict[str, Any]:
        """Generate manifest for current feature columns."""
        feature_cols = [c for c in self.get_combined_df(include_condition=False).columns]
        return generate_manifest(feature_cols, config=config, subject=subject, task=task, qc=self.qc or None)

    def save_with_manifest(
        self,
        output_dir: Union[str, Path],
        subject: str,
        task: str,
        config: Any = None,
        include_condition: bool = True,
    ) -> Dict[str, Path]:
        """
        Save combined features and manifest in a reproducible, organized structure.
        """
        df = self.get_combined_df(include_condition=include_condition)
        return save_features_organized(df, Path(output_dir), subject, task, config=config, qc=self.qc or None)


###################################################################
# Feature Extractors (Using Precomputed Data)
###################################################################


def _extract_erds_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract comprehensive ERD/ERS features using precomputed band power.
    
    ERD/ERS = (Active - Baseline) / Baseline * 100
    Negative = ERD (desynchronization), Positive = ERS (synchronization)
    
    Features extracted per channel/band:
    - Mean ERD/ERS over full active period
    - ERD/ERS per coarse temporal window (early, mid, late)
    - ERD/ERS per fine temporal window (t1-t7) for HRF modeling
    - Temporal dynamics: slope, onset latency, peak latency, early-late diff
    - ERD vs ERS separation (magnitude of negative vs positive values)
    - Global (cross-channel) statistics per band
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), [], {}

    # Early bail: enforce minimum epoch count for stable ERDS estimation
    n_epochs = precomputed.data.shape[0]
    if n_epochs < MIN_EPOCHS_FOR_FEATURES:
        if precomputed.logger:
            precomputed.logger.warning(
                "ERDS extraction skipped: only %d epochs available (min=%d). "
                "Insufficient trials for stable ERD/ERS estimation.",
                n_epochs,
                MIN_EPOCHS_FOR_FEATURES,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_epochs", "n_epochs": n_epochs}

    if not _validate_window_masks(precomputed, precomputed.logger):
        return pd.DataFrame(), [], {}

    epsilon = EPSILON_STD
    config = precomputed.config or {}
    erds_cfg = config.get("feature_engineering.erds", {})
    min_baseline_power = float(
        erds_cfg.get(
            "min_baseline_power",
            config.get("feature_engineering.features.min_baseline_power", epsilon),
        )
    )
    min_active_power = float(erds_cfg.get("min_active_power", epsilon))
    use_log_ratio = bool(erds_cfg.get("use_log_ratio", False))
    clamped_baselines = 0
    windows = precomputed.windows
    
    records: List[Dict[str, float]] = []
    qc_payload: Dict[str, Dict[str, Any]] = {}
    n_epochs = precomputed.data.shape[0]
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            all_erds_full = []
            all_log_full = []
            clamped_channels_for_band = 0
            baseline_valid_count = 0  # Track how many channels have valid baselines
            n_channels = len(precomputed.ch_names)
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = _nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                baseline_std = float(np.nanstd(power[ch_idx, windows.baseline_mask]))
                baseline_ref = baseline_power if baseline_power >= min_baseline_power else min_baseline_power
                if baseline_power < min_baseline_power:
                    clamped_baselines += 1
                    clamped_channels_for_band += 1
                baseline_valid = baseline_ref > epsilon and baseline_frac >= MIN_VALID_FRACTION and baseline_total > 0
                if baseline_valid:
                    baseline_valid_count += 1
                active_power_trace = power[ch_idx, windows.active_mask]
                active_power_mean, active_frac, _, _ = _nanmean_with_fraction(power[ch_idx], windows.active_mask)
                safe_active_trace = np.maximum(active_power_trace, min_active_power)
                safe_active_mean = float(active_power_mean) if np.isfinite(active_power_mean) else min_active_power
                
                if baseline_valid:
                    erds_full = ((active_power_mean - baseline_ref) / baseline_ref) * 100
                    erds_trace = ((active_power_trace - baseline_ref) / baseline_ref) * 100
                    erds_full_db = 10 * np.log10(safe_active_mean / baseline_ref)
                    erds_trace_db = 10 * np.log10(safe_active_trace / baseline_ref)
                else:
                    erds_full = np.nan
                    erds_trace = np.full_like(active_power_trace, np.nan)
                    erds_full_db = np.nan
                    erds_trace_db = np.full_like(active_power_trace, np.nan)
                
                # Full period ERD/ERS
                record[f"erds_{band}_{ch_name}_full_percent"] = float(erds_full)
                if use_log_ratio:
                    record[f"erds_{band}_{ch_name}_full_db"] = float(erds_full_db)
                all_erds_full.append(erds_full)
                all_log_full.append(erds_full_db)
                
                # === Coarse temporal bins (early, mid, late) ===
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_valid:
                            erds_win = ((win_power - baseline_ref) / baseline_ref) * 100
                            erds_win_db = 10 * np.log10(max(win_power, min_active_power) / baseline_ref)
                        else:
                            erds_win = np.nan
                            erds_win_db = np.nan
                        record[f"erds_{band}_{ch_name}_{win_label}_percent"] = float(erds_win)
                        if use_log_ratio:
                            record[f"erds_{band}_{ch_name}_{win_label}_db"] = float(erds_win_db)
                        coarse_values[win_label] = erds_win
                
                # === Fine temporal bins (t1-t7) for HRF modeling ===
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_valid:
                            erds_win = ((win_power - baseline_ref) / baseline_ref) * 100
                            erds_win_db = 10 * np.log10(max(win_power, min_active_power) / baseline_ref)
                        else:
                            erds_win = np.nan
                            erds_win_db = np.nan
                        record[f"erds_{band}_{ch_name}_{win_label}_percent"] = float(erds_win)
                        if use_log_ratio:
                            record[f"erds_{band}_{ch_name}_{win_label}_db"] = float(erds_win_db)
                
                # === Temporal dynamics ===
                # Default ERD/ERS separation metrics to NaN for invalid baselines
                record[f"erds_{band}_{ch_name}_erd_magnitude"] = np.nan
                record[f"erds_{band}_{ch_name}_erd_duration"] = np.nan
                record[f"erds_{band}_{ch_name}_ers_magnitude"] = np.nan
                record[f"erds_{band}_{ch_name}_ers_duration"] = np.nan
                
                if np.any(np.isfinite(erds_trace)) and len(active_times) > 1:
                    valid_mask = np.isfinite(erds_trace)
                    
                    # Slope (linear trend over plateau)
                    if np.sum(valid_mask) > 2:
                        slope, _ = np.polyfit(active_times[valid_mask], erds_trace[valid_mask], 1)
                        record[f"erds_{band}_{ch_name}_slope"] = float(slope)
                    else:
                        record[f"erds_{band}_{ch_name}_slope"] = np.nan
                    
                    # Early-late difference
                    if "early" in coarse_values and "late" in coarse_values:
                        diff = coarse_values["late"] - coarse_values["early"]
                        record[f"erds_{band}_{ch_name}_early_late_diff"] = float(diff)
                    
                    # Peak latency
                    peak_idx = np.nanargmax(np.abs(erds_trace))
                    record[f"erds_{band}_{ch_name}_peak_latency"] = float(active_times[peak_idx])

                    # Onset latency
                    threshold = baseline_std / baseline_ref * 100 if baseline_ref > epsilon else np.inf
                    onset_mask = np.abs(erds_trace) > threshold
                    if np.any(onset_mask):
                        onset_idx = np.argmax(onset_mask)
                        record[f"erds_{band}_{ch_name}_onset_latency"] = float(active_times[onset_idx])
                    else:
                        record[f"erds_{band}_{ch_name}_onset_latency"] = np.nan
                    
                    # === ERD vs ERS separation ===
                    erd_vals = erds_trace[erds_trace < 0]
                    ers_vals = erds_trace[erds_trace > 0]
                    
                    if baseline_valid:
                        if len(erd_vals) > 0:
                            record[f"erds_{band}_{ch_name}_erd_magnitude"] = float(np.mean(np.abs(erd_vals)))
                            record[f"erds_{band}_{ch_name}_erd_duration"] = float(len(erd_vals) / precomputed.sfreq)
                        else:
                            record[f"erds_{band}_{ch_name}_erd_magnitude"] = 0.0
                            record[f"erds_{band}_{ch_name}_erd_duration"] = 0.0
                        
                        if len(ers_vals) > 0:
                            record[f"erds_{band}_{ch_name}_ers_magnitude"] = float(np.mean(ers_vals))
                            record[f"erds_{band}_{ch_name}_ers_duration"] = float(len(ers_vals) / precomputed.sfreq)
                        else:
                            record[f"erds_{band}_{ch_name}_ers_magnitude"] = 0.0
                            record[f"erds_{band}_{ch_name}_ers_duration"] = 0.0
            
            # === Global statistics per band ===
            valid_erds = [e for e in all_erds_full if np.isfinite(e)]
            valid_log = [e for e in all_log_full if np.isfinite(e)]
            record[f"erds_{band}_baseline_clamped_channels"] = int(clamped_channels_for_band)
            record[f"erds_{band}_baseline_valid_channels"] = int(baseline_valid_count)
            baseline_valid_fraction = baseline_valid_count / n_channels if n_channels > 0 else 0.0
            record[f"erds_{band}_baseline_valid_fraction"] = float(baseline_valid_fraction)
            
            if band not in qc_payload:
                qc_payload[band] = {"clamped_channels": [], "baseline_min_power": min_baseline_power, "valid_fractions": []}
            qc_payload[band]["clamped_channels"].append(int(clamped_channels_for_band))
            qc_payload[band]["valid_fractions"].append(float(baseline_valid_fraction))
            
            # Skip global summaries when baseline-valid fraction is too low to avoid mixing valid/invalid channels
            if baseline_valid_fraction < MIN_VALID_FRACTION:
                record[f"erds_{band}_global_full_mean"] = np.nan
                record[f"erds_{band}_global_full_std"] = np.nan
                for win_label in windows.coarse_labels:
                    record[f"erds_{band}_global_{win_label}_mean"] = np.nan
                    if use_log_ratio:
                        record[f"erds_{band}_global_{win_label}_db_mean"] = np.nan
                if use_log_ratio:
                    record[f"erds_{band}_global_full_db_mean"] = np.nan
                    record[f"erds_{band}_global_full_db_std"] = np.nan
            elif valid_erds:
                record[f"erds_{band}_global_full_mean"] = float(np.mean(valid_erds))
                record[f"erds_{band}_global_full_std"] = float(np.std(valid_erds))
                # Global per coarse bin
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_erds = []
                        win_log = []
                        for ch_idx in range(len(precomputed.ch_names)):
                            bp = np.mean(power[ch_idx, windows.baseline_mask])
                            bp_ref = bp if bp >= min_baseline_power else min_baseline_power
                            if bp_ref > epsilon:
                                wp = np.mean(power[ch_idx, win_mask])
                                win_erds.append(((wp - bp_ref) / bp_ref) * 100)
                                if use_log_ratio:
                                    win_log.append(10 * np.log10(max(wp, min_active_power) / bp_ref))
                        if win_erds:
                            record[f"erds_{band}_global_{win_label}_mean"] = float(np.mean(win_erds))
                        if use_log_ratio and win_log:
                            record[f"erds_{band}_global_{win_label}_db_mean"] = float(np.mean(win_log))
                if use_log_ratio and valid_log:
                    record[f"erds_{band}_global_full_db_mean"] = float(np.mean(valid_log))
                    record[f"erds_{band}_global_full_db_std"] = float(np.std(valid_log))

        records.append(record)

    if clamped_baselines > 0 and precomputed.logger:
        precomputed.logger.info(
            "Clamped %d baseline power values below min_baseline_power=%.3e to stabilize ERD/ERS ratios (use_log_ratio=%s).",
            clamped_baselines,
            min_baseline_power,
            use_log_ratio,
        )
    
    # QC summary per band
    qc_summary: Dict[str, Any] = {}
    for band, stats in qc_payload.items():
        clamp_list = stats.get("clamped_channels", [])
        valid_frac_list = stats.get("valid_fractions", [])
        qc_summary[band] = {
            "median_clamped_channels": float(np.median(clamp_list)) if clamp_list else 0.0,
            "max_clamped_channels": int(np.max(clamp_list)) if clamp_list else 0,
            "min_baseline_power": float(stats.get("baseline_min_power", min_baseline_power)),
            "median_baseline_valid_fraction": float(np.median(valid_frac_list)) if valid_frac_list else 0.0,
            "min_baseline_valid_fraction": float(np.min(valid_frac_list)) if valid_frac_list else 0.0,
            "n_epochs_low_validity": int(sum(1 for f in valid_frac_list if f < MIN_VALID_FRACTION)),
        }
    
    # Log warning if many epochs have low baseline validity
    for band, stats in qc_summary.items():
        if stats["n_epochs_low_validity"] > 0 and precomputed.logger:
            precomputed.logger.warning(
                "ERDS band '%s': %d/%d epochs had baseline_valid_fraction < %.1f%%; "
                "global summaries set to NaN for these epochs.",
                band,
                stats["n_epochs_low_validity"],
                n_epochs,
                MIN_VALID_FRACTION * 100,
            )

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, qc_summary


def _extract_power_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract time-resolved band power features using precomputed band data.
    
    Features extracted per channel/band:
    - Power per coarse temporal window (early, mid, late)
    - Power per fine temporal window (t1-t7) for HRF modeling
    - Temporal dynamics: slope, early-late diff
    - Baseline-normalized power (log-ratio)
    - Global (cross-channel) statistics per band
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), [], {}

    # Early bail: enforce minimum epoch count for stable power estimation
    n_epochs = precomputed.data.shape[0]
    if n_epochs < MIN_EPOCHS_FOR_FEATURES:
        if precomputed.logger:
            precomputed.logger.warning(
                "Power extraction skipped: only %d epochs available (min=%d). "
                "Insufficient trials for stable power estimation.",
                n_epochs,
                MIN_EPOCHS_FOR_FEATURES,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_epochs", "n_epochs": n_epochs}

    if not _validate_window_masks(precomputed, precomputed.logger):
        return pd.DataFrame(), [], {}
    
    epsilon = EPSILON_STD
    windows = precomputed.windows
    
    records: List[Dict[str, float]] = []
    qc_payload: Dict[str, Dict[str, Any]] = {}
    n_epochs = precomputed.data.shape[0]
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            baseline_valid_count = 0
            total_channels = len(precomputed.ch_names)
            all_power_full = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = _nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                active_power, active_frac, _, _ = _nanmean_with_fraction(
                    power[ch_idx], windows.active_mask
                )
                baseline_valid = baseline_power > epsilon and baseline_frac >= MIN_VALID_FRACTION and baseline_total > 0
                if baseline_valid:
                    baseline_valid_count += 1
                
                # Full period power (log-ratio normalized)
                if baseline_valid:
                    logratio = np.log10(active_power / baseline_power)
                else:
                    logratio = np.nan
                record[f"power_{band}_{ch_name}_full_logratio"] = float(logratio)
                all_power_full.append(logratio)
                
                # === Coarse temporal bins (early, mid, late) ===
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_power, win_frac, _, _ = _nanmean_with_fraction(power[ch_idx], win_mask)
                        if baseline_valid and win_frac >= MIN_VALID_FRACTION:
                            win_logratio = np.log10(win_power / baseline_power)
                        else:
                            win_logratio = np.nan
                        record[f"power_{band}_{ch_name}_{win_label}_logratio"] = float(win_logratio)
                        coarse_values[win_label] = win_logratio
                
                # === Fine temporal bins (t1-t7) for HRF modeling ===
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        win_power, win_frac, _, _ = _nanmean_with_fraction(power[ch_idx], win_mask)
                        if baseline_valid and win_frac >= MIN_VALID_FRACTION:
                            win_logratio = np.log10(win_power / baseline_power)
                        else:
                            win_logratio = np.nan
                        record[f"power_{band}_{ch_name}_{win_label}_logratio"] = float(win_logratio)
                
                # === Temporal dynamics ===
                if len(active_times) > 2:
                    active_power_trace = power[ch_idx, windows.active_mask]
                    if baseline_valid:
                        logratio_trace = np.log10(active_power_trace / baseline_power)
                        valid_mask = np.isfinite(logratio_trace)
                        if np.sum(valid_mask) > 2:
                            slope, _ = np.polyfit(active_times[valid_mask], logratio_trace[valid_mask], 1)
                            record[f"power_{band}_{ch_name}_slope"] = float(slope)
                        else:
                            record[f"power_{band}_{ch_name}_slope"] = np.nan
                    else:
                        record[f"power_{band}_{ch_name}_slope"] = np.nan
                    
                    # Early-late difference
                    if "early" in coarse_values and "late" in coarse_values:
                        diff = coarse_values["late"] - coarse_values["early"]
                        record[f"power_{band}_{ch_name}_early_late_diff"] = float(diff)
            
            # === Global statistics per band ===
            valid_power = [p for p in all_power_full if np.isfinite(p)]
            baseline_valid_fraction = baseline_valid_count / total_channels if total_channels > 0 else 0.0
            record[f"power_{band}_baseline_valid_fraction"] = float(baseline_valid_fraction)
            
            # Skip global summaries when baseline-valid fraction is too low
            if baseline_valid_fraction < MIN_VALID_FRACTION:
                record[f"power_{band}_global_full_mean"] = np.nan
                record[f"power_{band}_global_full_std"] = np.nan
                for win_label in windows.coarse_labels:
                    record[f"power_{band}_global_{win_label}_mean"] = np.nan
            elif valid_power:
                record[f"power_{band}_global_full_mean"] = float(np.mean(valid_power))
                record[f"power_{band}_global_full_std"] = float(np.std(valid_power))
                
                # Global per coarse bin
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_powers = []
                        for ch_idx in range(len(precomputed.ch_names)):
                            bp, bp_frac, _, bp_total = _nanmean_with_fraction(power[ch_idx], windows.baseline_mask)
                            if bp > epsilon and bp_frac >= MIN_VALID_FRACTION and bp_total > 0:
                                wp, wp_frac, _, _ = _nanmean_with_fraction(power[ch_idx], win_mask)
                                if wp_frac >= MIN_VALID_FRACTION:
                                    win_powers.append(np.log10(wp / bp))
                        if win_powers:
                            record[f"power_{band}_global_{win_label}_mean"] = float(np.mean(win_powers))
            
            # QC tracking per band
            if band not in qc_payload:
                qc_payload[band] = {"baseline_valid_fraction": [], "n_epochs_low_validity": 0}
            qc_payload[band]["baseline_valid_fraction"].append(float(baseline_valid_fraction))
            if baseline_valid_fraction < MIN_VALID_FRACTION:
                qc_payload[band]["n_epochs_low_validity"] += 1
        
        records.append(record)
    
    # Aggregate QC summaries per band
    qc_summary: Dict[str, Any] = {}
    for band, stats in qc_payload.items():
        baseline_vals = stats["baseline_valid_fraction"]
        n_low = stats.get("n_epochs_low_validity", 0)
        qc_summary[band] = {
            "baseline_valid_fraction_median": float(np.nanmedian(baseline_vals)) if baseline_vals else np.nan,
            "baseline_valid_fraction_min": float(np.nanmin(baseline_vals)) if baseline_vals else np.nan,
            "n_epochs_low_validity": n_low,
        }
        # Log warning if many epochs have low baseline validity
        if n_low > 0 and precomputed.logger:
            precomputed.logger.warning(
                "Power band '%s': %d/%d epochs had baseline_valid_fraction < %.1f%%; "
                "global summaries set to NaN for these epochs.",
                band,
                n_low,
                n_epochs,
                MIN_VALID_FRACTION * 100,
            )
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, qc_summary


def _extract_spectral_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract comprehensive spectral features using precomputed PSD.
    
    Features extracted per channel/band:
    - Absolute and relative power (mean, median, std)
    - Peak frequency and peak power
    - Spectral entropy (normalized)
    - Spectral edge frequencies (50%, 75%, 90%, 95%)
    - Spectral slope within band
    - Band power percentiles
    - Cross-band ratios (for ML discriminability)
    """
    if precomputed.psd_data is None:
        if precomputed.data.size > 0:
            precomputed.logger and precomputed.logger.warning(
                "PSD data missing for spectral features; computing PSD on the fly."
            )
            precomputed.psd_data = compute_psd(
                precomputed.data, precomputed.sfreq, config=precomputed.config, logger=precomputed.logger
            )
        if precomputed.psd_data is None:
            precomputed.logger and precomputed.logger.error(
                "PSD computation unavailable; skipping spectral features."
            )
            return pd.DataFrame(), []
    
    config = precomputed.config
    freq_bands = get_frequency_bands(config)
    freqs = precomputed.psd_data.freqs
    psd = precomputed.psd_data.psd  # (epochs, channels, freqs)
    missing_band_warned = set()
    
    records: List[Dict[str, float]] = []
    n_epochs = psd.shape[0]
    
    # Total power for relative calculations (ignore NaNs instead of zeroing entire epoch/channel)
    finite_mask = np.isfinite(psd) & np.isfinite(freqs)[None, None, :]
    psd_clean = np.where(finite_mask, psd, 0.0)
    total_power = np.trapz(psd_clean, freqs, axis=2)  # (epochs, channels)
    valid_bins = np.sum(finite_mask, axis=2)
    total_power = np.where(valid_bins >= 2, total_power, np.nan)
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        # Store band powers for ratio computation
        band_powers_per_ch: Dict[str, Dict[str, float]] = {ch: {} for ch in precomputed.ch_names}
        
        for band in bands:
            if band not in freq_bands:
                continue
            
            fmin, fmax = freq_bands[band]
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            
            if not np.any(freq_mask):
                if band not in missing_band_warned and precomputed.logger:
                    precomputed.logger.warning(
                        "Spectral band '%s' is outside PSD frequency grid [%.2f, %.2f]; skipping.",
                        band,
                        float(freqs.min()),
                        float(freqs.max()),
                    )
                    missing_band_warned.add(band)
                continue
            
            band_freqs = freqs[freq_mask]
            all_band_power = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                band_psd = psd[ep_idx, ch_idx, freq_mask]
                finite_mask = np.isfinite(band_psd) & np.isfinite(band_freqs)

                if not np.any(finite_mask):
                    record[f"pow_{band}_{ch_name}"] = np.nan
                    record[f"pow_{band}_{ch_name}_sum"] = np.nan
                    record[f"pow_{band}_{ch_name}_std"] = np.nan
                    record[f"pow_{band}_{ch_name}_median"] = np.nan
                    record[f"pow_{band}_{ch_name}_max"] = np.nan
                    record[f"logpow_{band}_{ch_name}"] = np.nan
                    record[f"relpow_{band}_{ch_name}"] = np.nan
                    record[f"peakfreq_{band}_{ch_name}"] = np.nan
                    record[f"peakpow_{band}_{ch_name}"] = np.nan
                    record[f"peakprom_{band}_{ch_name}"] = np.nan
                    record[f"se_{band}_{ch_name}"] = np.nan
                    record[f"slope_{band}_{ch_name}"] = np.nan
                    for edge_pct in [50, 75, 90, 95]:
                        record[f"sef{edge_pct}_{band}_{ch_name}"] = np.nan
                    continue

                band_freqs_clean = band_freqs[finite_mask]
                band_psd_clean = band_psd[finite_mask]

                band_power_area = float(np.trapz(band_psd_clean, band_freqs_clean))
                band_power_mean_psd = float(np.mean(band_psd_clean))

                band_powers_per_ch[ch_name][band] = band_power_area
                all_band_power.append(band_power_area)
                
                # === Absolute power statistics ===
                record[f"pow_{band}_{ch_name}"] = band_power_area
                record[f"pow_{band}_{ch_name}_sum"] = band_power_area
                record[f"pow_{band}_{ch_name}_std"] = float(np.std(band_psd_clean))
                record[f"pow_{band}_{ch_name}_median"] = float(np.median(band_psd_clean))
                record[f"pow_{band}_{ch_name}_max"] = float(np.max(band_psd_clean))
                
                # Log power (often more normally distributed)
                record[f"logpow_{band}_{ch_name}"] = float(np.log10(band_power_area + epsilon))
                
                # === Relative power ===
                if total_power[ep_idx, ch_idx] > epsilon:
                    rel_power = band_power_area / total_power[ep_idx, ch_idx]
                else:
                    rel_power = np.nan
                record[f"relpow_{band}_{ch_name}"] = float(rel_power)
                
                # === Peak frequency and power ===
                peak_idx = int(np.nanargmax(band_psd_clean))
                record[f"peakfreq_{band}_{ch_name}"] = float(band_freqs_clean[peak_idx])
                record[f"peakpow_{band}_{ch_name}"] = float(band_psd_clean[peak_idx])
                
                # Peak prominence (peak / mean PSD)
                if band_power_mean_psd > epsilon:
                    record[f"peakprom_{band}_{ch_name}"] = float(band_psd_clean[peak_idx] / band_power_mean_psd)
                else:
                    record[f"peakprom_{band}_{ch_name}"] = np.nan
                
                # === Spectral entropy (within band), frequency-weighted ===
                freq_weights = np.gradient(band_freqs_clean)
                power_weights = band_psd_clean * freq_weights
                total_band_area = float(np.nansum(power_weights))
                if np.isfinite(total_band_area) and total_band_area > 0:
                    p = power_weights / total_band_area
                    p = p[p > 0]
                    if p.size > 0:
                        se = -np.sum(p * np.log2(p))
                        se_norm = se / np.log2(p.size) if p.size > 1 else se
                    else:
                        se_norm = np.nan
                else:
                    se_norm = np.nan
                record[f"se_{band}_{ch_name}"] = float(se_norm)
                
                # === Spectral edge frequencies ===
                # Use the same frequency-weighted power used for spectral entropy
                band_area = power_weights
                if total_band_area > 0 and band_area.size > 0:
                    cumsum = np.cumsum(band_area)
                    for edge_pct in [50, 75, 90, 95]:
                        threshold = total_band_area * (edge_pct / 100.0)
                        edge_idx = np.searchsorted(cumsum, threshold)
                        edge_idx = min(edge_idx, len(band_freqs_clean) - 1)
                        record[f"sef{edge_pct}_{band}_{ch_name}"] = float(band_freqs_clean[edge_idx])
                else:
                    for edge_pct in [50, 75, 90, 95]:
                        record[f"sef{edge_pct}_{band}_{ch_name}"] = np.nan
                
                # === Spectral slope within band ===
                if len(band_freqs_clean) > 2:
                    log_freqs = np.log10(band_freqs_clean + epsilon)
                    log_psd = np.log10(band_psd_clean + epsilon)
                    try:
                        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
                        record[f"slope_{band}_{ch_name}"] = float(slope)
                    except (np.linalg.LinAlgError, ValueError):
                        record[f"slope_{band}_{ch_name}"] = np.nan
                else:
                    record[f"slope_{band}_{ch_name}"] = np.nan
            
            # === Global band statistics ===
            if all_band_power:
                record[f"pow_{band}_global_mean"] = float(np.mean(all_band_power))
                record[f"pow_{band}_global_std"] = float(np.std(all_band_power))
                record[f"pow_{band}_global_cv"] = float(np.std(all_band_power) / (np.mean(all_band_power) + epsilon))
        
        # === Cross-band ratios (important for ML) ===
        ratio_pairs = [
            ("theta", "beta"),    # Attention/ADHD marker
            ("alpha", "theta"),   # Alertness
            ("alpha", "beta"),    # Relaxation vs activation
            ("theta", "alpha"),   # Drowsiness
            ("beta", "gamma"),    # High-frequency activity
            ("delta", "beta"),    # Slow vs fast
        ]
        for ch_name in precomputed.ch_names:
            for num_band, denom_band in ratio_pairs:
                if num_band in band_powers_per_ch[ch_name] and denom_band in band_powers_per_ch[ch_name]:
                    num = band_powers_per_ch[ch_name][num_band]
                    denom = band_powers_per_ch[ch_name][denom_band]
                    if denom > epsilon:
                        record[f"spec_ratio_{num_band}_{denom_band}_{ch_name}"] = float(num / denom)
                        record[f"spec_logratio_{num_band}_{denom_band}_{ch_name}"] = float(np.log10((num + epsilon) / (denom + epsilon)))
                    else:
                        record[f"spec_ratio_{num_band}_{denom_band}_{ch_name}"] = np.nan
                        record[f"spec_logratio_{num_band}_{denom_band}_{ch_name}"] = np.nan
        
        # === IAF (Individual Alpha Frequency) with additional metrics ===
        alpha_range = freq_bands.get("alpha", (8.0, 13.0))
        alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
        if np.any(alpha_mask):
            alpha_freqs = freqs[alpha_mask]
            all_iaf = []
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                alpha_psd = psd[ep_idx, ch_idx, alpha_mask]
                finite_mask = np.isfinite(alpha_psd) & np.isfinite(alpha_freqs)
                if np.any(finite_mask):
                    alpha_freqs_clean = alpha_freqs[finite_mask]
                    alpha_psd_clean = alpha_psd[finite_mask]
                    freq_weights = np.gradient(alpha_freqs_clean)
                    alpha_area = np.nansum(alpha_psd_clean * freq_weights)
                    if alpha_area > 0:
                        iaf_cog = float(
                            np.nansum(alpha_freqs_clean * alpha_psd_clean * freq_weights) / alpha_area
                        )
                        iaf_peak_idx = int(np.nanargmax(alpha_psd_clean))
                        iaf_peak = float(alpha_freqs_clean[iaf_peak_idx])
                        iaf_power = float(alpha_psd_clean[iaf_peak_idx])
                    else:
                        iaf_cog = np.nan
                        iaf_peak = np.nan
                        iaf_power = np.nan
                else:
                    iaf_cog = np.nan
                    iaf_peak = np.nan
                    iaf_power = np.nan
                
                record[f"iaf_cog_{ch_name}"] = float(iaf_cog)
                record[f"iaf_peak_{ch_name}"] = float(iaf_peak)
                record[f"iaf_power_{ch_name}"] = float(iaf_power)
                if np.isfinite(iaf_cog):
                    all_iaf.append(iaf_cog)
            
            # Global IAF
            if all_iaf:
                record["iaf_global_mean"] = float(np.mean(all_iaf))
                record["iaf_global_std"] = float(np.std(all_iaf))
        
        records.append(record)
    
    return _build_records_to_df(records)


def _extract_gfp_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract GFP (Global Field Power) features: statistics, temporal dynamics,
    percentiles, baseline-normalized metrics, per-window features.
    """
    if precomputed.gfp is None:
        return _EMPTY_RESULT
    
    err = _validate_precomputed_for_extraction(
        precomputed, require_bands=False, context="GFP"
    )
    if err:
        if precomputed.logger:
            precomputed.logger.debug(err)
        return _EMPTY_RESULT
    
    from scipy.signal import find_peaks
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.gfp.shape[0]
    times = precomputed.times
    active_times = times[precomputed.windows.active_mask]
    epsilon = EPSILON_STD
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        # === Broadband GFP ===
        gfp = precomputed.gfp[ep_idx]
        gfp_baseline = gfp[precomputed.windows.baseline_mask]
        gfp_active = gfp[precomputed.windows.active_mask]
        
        # Basic statistics
        record["gfp_mean"] = float(np.mean(gfp_active))
        record["gfp_std"] = float(np.std(gfp_active))
        record["gfp_min"] = float(np.min(gfp_active))
        record["gfp_max"] = float(np.max(gfp_active))
        record["gfp_range"] = float(np.max(gfp_active) - np.min(gfp_active))
        record["gfp_cv"] = float(np.std(gfp_active) / (np.mean(gfp_active) + epsilon))
        record["gfp_median"] = float(np.median(gfp_active))
        
        # Percentiles
        for pct in [10, 25, 75, 90]:
            record[f"gfp_p{pct}"] = float(np.percentile(gfp_active, pct))
        
        # Baseline-normalized GFP
        baseline_mean = np.mean(gfp_baseline) if len(gfp_baseline) > 0 else epsilon
        if baseline_mean > epsilon:
            record["gfp_baseline_ratio"] = float(np.mean(gfp_active) / baseline_mean)
            record["gfp_baseline_change"] = float((np.mean(gfp_active) - baseline_mean) / baseline_mean * 100)
        else:
            record["gfp_baseline_ratio"] = np.nan
            record["gfp_baseline_change"] = np.nan
        
        # Temporal dynamics
        if len(active_times) > 2:
            # Slope
            try:
                slope, _ = np.polyfit(active_times, gfp_active, 1)
                record["gfp_slope"] = float(slope)
            except (np.linalg.LinAlgError, ValueError):
                record["gfp_slope"] = np.nan
            
            # Peak detection
            peaks, properties = find_peaks(gfp_active, distance=int(precomputed.sfreq * 0.1))  # min 100ms between peaks
            record["gfp_peak_count"] = float(len(peaks))
            
            if len(peaks) > 0:
                # Peak latency (time of first major peak)
                record["gfp_first_peak_latency"] = float(active_times[peaks[0]])
                # Max peak latency
                max_peak_idx = peaks[np.argmax(gfp_active[peaks])]
                record["gfp_max_peak_latency"] = float(active_times[max_peak_idx])
            else:
                record["gfp_first_peak_latency"] = np.nan
                record["gfp_max_peak_latency"] = np.nan
        else:
            record["gfp_slope"] = np.nan
            record["gfp_peak_count"] = np.nan
            record["gfp_first_peak_latency"] = np.nan
            record["gfp_max_peak_latency"] = np.nan
        
        # Per-window GFP (coarse bins)
        for win_mask, win_label in zip(
            precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
        ):
            if np.any(win_mask):
                gfp_win = gfp[win_mask]
                record[f"gfp_{win_label}_mean"] = float(np.mean(gfp_win))
                record[f"gfp_{win_label}_std"] = float(np.std(gfp_win))
        
        # Per-window GFP (fine bins for HRF)
        for win_mask, win_label in zip(
            precomputed.windows.fine_masks, precomputed.windows.fine_labels
        ):
            if np.any(win_mask):
                gfp_win = gfp[win_mask]
                record[f"gfp_{win_label}_mean"] = float(np.mean(gfp_win))
        
        # === Band-specific GFP ===
        for band in bands:
            if band not in precomputed.gfp_band:
                continue
            
            gfp_band = precomputed.gfp_band[band][ep_idx]
            gfp_band_baseline = gfp_band[precomputed.windows.baseline_mask]
            gfp_band_active = gfp_band[precomputed.windows.active_mask]
            
            # Basic statistics
            record[f"gfp_{band}_mean"] = float(np.mean(gfp_band_active))
            record[f"gfp_{band}_std"] = float(np.std(gfp_band_active))
            record[f"gfp_{band}_min"] = float(np.min(gfp_band_active))
            record[f"gfp_{band}_max"] = float(np.max(gfp_band_active))
            record[f"gfp_{band}_cv"] = float(np.std(gfp_band_active) / (np.mean(gfp_band_active) + epsilon))
            
            # Baseline-normalized
            band_baseline_mean = np.mean(gfp_band_baseline) if len(gfp_band_baseline) > 0 else epsilon
            if band_baseline_mean > epsilon:
                record[f"gfp_{band}_baseline_ratio"] = float(np.mean(gfp_band_active) / band_baseline_mean)
            else:
                record[f"gfp_{band}_baseline_ratio"] = np.nan
            
            # Percentiles
            for pct in [25, 50, 75]:
                record[f"gfp_{band}_p{pct}"] = float(np.percentile(gfp_band_active, pct))
            
            # Per-window band GFP (coarse bins)
            for win_mask, win_label in zip(
                precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
            ):
                if np.any(win_mask):
                    gfp_band_win = gfp_band[win_mask]
                    record[f"gfp_{band}_{win_label}_mean"] = float(np.mean(gfp_band_win))
            
            # Per-window band GFP (fine bins for HRF)
            for win_mask, win_label in zip(
                precomputed.windows.fine_masks, precomputed.windows.fine_labels
            ):
                if np.any(win_mask):
                    gfp_band_win = gfp_band[win_mask]
                    record[f"gfp_{band}_{win_label}_mean"] = float(np.mean(gfp_band_win))
        
        records.append(record)
    
    return _build_records_to_df(records)


def _extract_connectivity_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract connectivity features: wPLI, PLV, AEC, graph metrics (degree, density),
    statistics across channel pairs, per-window connectivity.
    """
    err = _validate_precomputed_for_extraction(
        precomputed, require_baseline=False, min_epochs=MIN_EPOCHS_FOR_PLV, context="Connectivity"
    )
    if err:
        if precomputed and precomputed.logger:
            precomputed.logger.debug(err)
        return _EMPTY_RESULT_WITH_QC
    
    # Additional connectivity-specific checks
    active_samples = int(np.sum(precomputed.windows.active_mask))
    if active_samples < MIN_EDGE_SAMPLES:
        if precomputed.logger:
            precomputed.logger.warning(
                "Connectivity extraction skipped: only %d samples in active window (min=%d). "
                "Insufficient samples for stable phase/envelope correlation.",
                active_samples,
                MIN_EDGE_SAMPLES,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_samples", "active_samples": active_samples}
    
    n_channels = len(precomputed.ch_names)
    if n_channels < MIN_CHANNELS_FOR_CONNECTIVITY:
        if precomputed.logger:
            precomputed.logger.warning(
                "Connectivity features skipped: need at least %d channels, found %d.",
                MIN_CHANNELS_FOR_CONNECTIVITY,
                n_channels,
            )
        return pd.DataFrame(), [], {}
    
    n_epochs = precomputed.data.shape[0]
    epsilon = 1e-12
    n_jobs_connectivity = int((precomputed.config or {}).get("feature_engineering.parallel.n_jobs_connectivity", 1))

    def _compute_connectivity_for_epoch(ep_idx: int) -> Tuple[Dict[str, float], Dict[str, Dict[str, List[float]]]]:
        """Compute connectivity metrics for a single epoch."""
        record: Dict[str, float] = {}
        band_qc_epoch: Dict[str, Dict[str, List[float]]] = {}

        for band in bands:
            if band not in precomputed.band_data:
                continue

            analytic = precomputed.band_data[band].analytic[ep_idx]  # (channels, times)
            envelope = precomputed.band_data[band].envelope[ep_idx]
            phases_full = precomputed.band_data[band].phase[ep_idx]  # (channels, times)

            active_mask = precomputed.windows.active_mask
            analytic_active_full = analytic[:, active_mask]
            envelope_active_full = envelope[:, active_mask]
            phases_active_full = phases_full[:, active_mask]

            # Per-channel validity to avoid penalizing all channels for a single bad sensor
            channel_valid = (
                np.isfinite(analytic_active_full)
                & np.isfinite(envelope_active_full)
                & np.isfinite(phases_active_full)
            )
            channel_valid_fraction = np.mean(channel_valid, axis=1)
            keep_channels = channel_valid_fraction >= MIN_VALID_FRACTION
            if np.sum(keep_channels) < MIN_CHANNELS_FOR_CONNECTIVITY:
                if precomputed.logger:
                    precomputed.logger.warning(
                        "Connectivity: insufficient valid channels for band '%s' (kept %d/%d) in epoch %d; skipping.",
                        band,
                        int(np.sum(keep_channels)),
                        n_channels,
                        ep_idx,
                    )
                continue

            analytic_active = analytic_active_full[keep_channels]
            envelope_active = envelope_active_full[keep_channels]
            phases_active = phases_active_full[keep_channels]
            kept_ch_names = [name for idx, name in enumerate(precomputed.ch_names) if keep_channels[idx]]

            channel_valid = channel_valid[keep_channels]
            valid_pairs = channel_valid[:, None, :] & channel_valid[None, :, :]
            pair_counts = np.sum(valid_pairs, axis=2)

            record[f"conn_{band}_n_pairs"] = int(len(kept_ch_names) * (len(kept_ch_names) - 1) / 2)
            record[f"conn_{band}_median_pair_valid_samples"] = float(np.nanmedian(pair_counts))
            record[f"conn_{band}_min_pair_valid_samples"] = float(np.nanmin(pair_counts))
            band_qc_epoch.setdefault(band, {
                "n_channels": [],
                "median_pair_valid_samples": [],
                "min_pair_valid_samples": [],
                "median_pair_valid_fraction": [],
            })
            band_qc_epoch[band]["n_channels"].append(len(kept_ch_names))
            n_time = float(analytic_active.shape[1])
            pair_valid_fraction = np.where(n_time > 0, pair_counts / n_time, np.nan)
            band_qc_epoch[band]["median_pair_valid_samples"].append(float(np.nanmedian(pair_counts)))
            band_qc_epoch[band]["min_pair_valid_samples"].append(float(np.nanmin(pair_counts)))
            band_qc_epoch[band]["median_pair_valid_fraction"].append(float(np.nanmedian(pair_valid_fraction)))

            # === wPLI computation with per-pair masking ===
            cross = analytic_active[:, None, :] * np.conj(analytic_active[None, :, :])
            cross_masked = np.where(valid_pairs, cross, np.nan)
            imag_cross = np.imag(cross_masked)
            denom = np.nanmean(np.abs(imag_cross), axis=-1)
            numer = np.abs(np.nanmean(imag_cross, axis=-1))

            with np.errstate(divide="ignore", invalid="ignore"):
                wpli = np.where(denom > 0, numer / denom, np.nan)
            wpli = 0.5 * (wpli + wpli.T)
            np.fill_diagonal(wpli, 0.0)
            triu_idx_epoch = np.triu_indices(len(kept_ch_names), k=1)
            wpli_values = wpli[triu_idx_epoch]
            wpli_valid = wpli_values[np.isfinite(wpli_values)]

            # wPLI statistics
            record[f"wpli_{band}_mean"] = float(np.nanmean(wpli_valid)) if wpli_valid.size else np.nan
            record[f"wpli_{band}_std"] = float(np.nanstd(wpli_valid)) if wpli_valid.size else np.nan
            record[f"wpli_{band}_max"] = float(np.nanmax(wpli_valid)) if wpli_valid.size else np.nan
            record[f"wpli_{band}_min"] = float(np.nanmin(wpli_valid)) if wpli_valid.size else np.nan
            record[f"wpli_{band}_median"] = float(np.nanmedian(wpli_valid)) if wpli_valid.size else np.nan
            for pct in [25, 75, 90]:
                record[f"wpli_{band}_p{pct}"] = float(np.nanpercentile(wpli_valid, pct)) if wpli_valid.size else np.nan

            # === PLV computation (vectorized with masking) ===
            phase_diff = phases_active[:, None, :] - phases_active[None, :, :]
            phase_diff_masked = np.where(valid_pairs, np.exp(1j * phase_diff), np.nan)
            plv_matrix = np.abs(np.nanmean(phase_diff_masked, axis=-1))
            plv_values = plv_matrix[triu_idx_epoch]
            plv_valid = plv_values[np.isfinite(plv_values)]

            # PLV statistics
            record[f"plv_{band}_mean"] = float(np.nanmean(plv_valid)) if plv_valid.size else np.nan
            record[f"plv_{band}_std"] = float(np.nanstd(plv_valid)) if plv_valid.size else np.nan
            record[f"plv_{band}_max"] = float(np.nanmax(plv_valid)) if plv_valid.size else np.nan
            record[f"plv_{band}_median"] = float(np.nanmedian(plv_valid)) if plv_valid.size else np.nan
            for pct in [25, 75, 90]:
                record[f"plv_{band}_p{pct}"] = float(np.nanpercentile(plv_valid, pct)) if plv_valid.size else np.nan

            # === AEC (Amplitude Envelope Correlation) ===
            n_kept = len(kept_ch_names)
            aec_matrix = np.full((n_kept, n_kept), np.nan)

            for i in range(n_kept):
                for j in range(i + 1, n_kept):
                    pair_mask = valid_pairs[i, j, :]
                    n_valid = np.sum(pair_mask)

                    if n_valid < 3:
                        continue

                    env_i = envelope_active[i, pair_mask]
                    env_j = envelope_active[j, pair_mask]

                    env_i_centered = env_i - np.mean(env_i)
                    env_j_centered = env_j - np.mean(env_j)
                    std_i = np.std(env_i)
                    std_j = np.std(env_j)

                    if std_i < epsilon or std_j < epsilon:
                        continue

                    corr = np.mean(env_i_centered * env_j_centered) / (std_i * std_j)
                    aec_matrix[i, j] = np.clip(corr, -1, 1)
                    aec_matrix[j, i] = aec_matrix[i, j]

            np.fill_diagonal(aec_matrix, 0.0)
            aec_values = aec_matrix[triu_idx_epoch]
            aec_values = np.clip(aec_values, -1, 1)

            # AEC statistics
            record[f"aec_{band}_mean"] = float(np.nanmean(aec_values)) if np.isfinite(aec_values).any() else np.nan
            record[f"aec_{band}_std"] = float(np.nanstd(aec_values)) if np.isfinite(aec_values).any() else np.nan
            record[f"aec_{band}_max"] = float(np.nanmax(aec_values)) if np.isfinite(aec_values).any() else np.nan
            record[f"aec_{band}_abs_mean"] = float(np.nanmean(np.abs(aec_values))) if np.isfinite(aec_values).any() else np.nan

            # === Graph metrics ===
            wpli_finite_counts = np.sum(np.isfinite(wpli), axis=1)
            plv_finite_counts = np.sum(np.isfinite(plv_matrix), axis=1)
            node_strength_wpli = np.where(
                wpli_finite_counts > 0,
                np.nansum(wpli, axis=1) / wpli_finite_counts,
                np.nan,
            )
            node_strength_plv = np.where(
                plv_finite_counts > 0,
                np.nansum(plv_matrix, axis=1) / plv_finite_counts,
                np.nan,
            )

            record[f"graph_{band}_wpli_mean_degree"] = float(np.nanmean(node_strength_wpli))
            record[f"graph_{band}_wpli_std_degree"] = float(np.nanstd(node_strength_wpli))
            record[f"graph_{band}_plv_mean_degree"] = float(np.nanmean(node_strength_plv))

            # Network density (proportion of strong connections)
            threshold = 0.3  # Common threshold for significant connectivity
            record[f"graph_{band}_wpli_density"] = float(np.nanmean(wpli_valid > threshold)) if wpli_valid.size else np.nan
            record[f"graph_{band}_plv_density"] = float(np.nanmean(plv_valid > threshold)) if plv_valid.size else np.nan

            # === Per-window connectivity (coarse bins) ===
            def _active_window_mask(win_mask: np.ndarray) -> Optional[np.ndarray]:
                """Mask window to valid, active samples only."""
                win_active = win_mask[active_mask]
                if win_active.shape[0] != valid_pairs.shape[2]:
                    return None
                return win_active

            def _compute_window_conn(win_mask_active: np.ndarray):
                """Helper to compute wPLI and PLV for a time window using cleaned data."""
                if win_mask_active is None or not np.any(win_mask_active):
                    return np.nan, np.nan

                valid_pairs_win = valid_pairs[:, :, win_mask_active]
                if not np.any(valid_pairs_win):
                    return np.nan, np.nan

                phases_win = phases_active[:, win_mask_active]
                phase_diff_win = phases_win[:, None, :] - phases_win[None, :, :]
                phase_diff_win_masked = np.where(valid_pairs_win, np.exp(1j * phase_diff_win), np.nan)
                plv_win = np.abs(np.nanmean(phase_diff_win_masked, axis=-1))
                plv_win_values = plv_win[triu_idx_epoch]

                analytic_win = analytic_active[:, win_mask_active]
                cross_win = analytic_win[:, None, :] * np.conj(analytic_win[None, :, :])
                cross_win_masked = np.where(valid_pairs_win, cross_win, np.nan)
                imag_cross_win = np.imag(cross_win_masked)
                denom_win = np.nanmean(np.abs(imag_cross_win), axis=-1)
                numer_win = np.abs(np.nanmean(imag_cross_win, axis=-1))
                with np.errstate(divide="ignore", invalid="ignore"):
                    wpli_win = np.where(denom_win > 0, numer_win / denom_win, np.nan)
                wpli_win = 0.5 * (wpli_win + wpli_win.T)
                np.fill_diagonal(wpli_win, 0.0)
                wpli_win_values = wpli_win[triu_idx_epoch]
                return np.nanmean(plv_win_values), np.nanmean(wpli_win_values)

            for win_mask, win_label in zip(
                precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
            ):
                win_mask_active = _active_window_mask(win_mask)
                if win_mask_active is not None and np.any(win_mask_active):
                    plv_mean, wpli_mean = _compute_window_conn(win_mask_active)
                    record[f"conn_plv_{band}_{win_label}_mean"] = float(plv_mean)
                    record[f"conn_wpli_{band}_{win_label}_mean"] = float(wpli_mean)

            # === Per-window connectivity (fine bins for HRF) ===
            for win_mask, win_label in zip(
                precomputed.windows.fine_masks, precomputed.windows.fine_labels
            ):
                win_mask_active = _active_window_mask(win_mask)
                if win_mask_active is not None and np.any(win_mask_active):
                    plv_mean, wpli_mean = _compute_window_conn(win_mask_active)
                    record[f"conn_plv_{band}_{win_label}_mean"] = float(plv_mean)
                    record[f"conn_wpli_{band}_{win_label}_mean"] = float(wpli_mean)

            # === Temporal dynamics ===
            if len(precomputed.windows.coarse_masks) >= 2:
                early_mask = precomputed.windows.coarse_masks[0]
                late_mask = precomputed.windows.coarse_masks[-1]
                if np.any(early_mask) and np.any(late_mask):
                    early_active = _active_window_mask(early_mask)
                    late_active = _active_window_mask(late_mask)
                    _, wpli_early = _compute_window_conn(early_active if early_active is not None else np.array([]))
                    _, wpli_late = _compute_window_conn(late_active if late_active is not None else np.array([]))
                    record[f"conn_wpli_{band}_early_late_diff"] = float(wpli_late - wpli_early)

        return record, band_qc_epoch

    try:
        if n_jobs_connectivity > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs_connectivity, prefer="processes")(
                delayed(_compute_connectivity_for_epoch)(ep_idx) for ep_idx in range(n_epochs)
            )
        else:
            results = [_compute_connectivity_for_epoch(ep_idx) for ep_idx in range(n_epochs)]
    except Exception as exc:  # pragma: no cover - defensive
        if precomputed.logger:
            precomputed.logger.warning(
                "Parallel connectivity extraction failed (%s); falling back to sequential.", exc
            )
        results = [_compute_connectivity_for_epoch(ep_idx) for ep_idx in range(n_epochs)]

    records: List[Dict[str, float]] = [rec for rec, _ in results]
    band_qc: Dict[str, Dict[str, List[float]]] = {}
    for _, qc_epoch in results:
        for band, stats in qc_epoch.items():
            if band not in band_qc:
                band_qc[band] = {k: [] for k in stats.keys()}
            for key, vals in stats.items():
                band_qc[band][key].extend(vals)
    
    columns = list(records[0].keys()) if records else []
    qc_payload: Dict[str, Any] = {}
    for band, stats in band_qc.items():
        qc_payload[band] = {
            "median_channels_used": float(np.nanmedian(stats["n_channels"])) if stats["n_channels"] else np.nan,
            "min_channels_used": float(np.nanmin(stats["n_channels"])) if stats["n_channels"] else np.nan,
            "median_pair_valid_samples": float(np.nanmedian(stats["median_pair_valid_samples"])) if stats["median_pair_valid_samples"] else np.nan,
            "min_pair_valid_samples": float(np.nanmin(stats["min_pair_valid_samples"])) if stats["min_pair_valid_samples"] else np.nan,
            "median_pair_valid_fraction": float(np.nanmedian(stats["median_pair_valid_fraction"])) if stats["median_pair_valid_fraction"] else np.nan,
        }

    return pd.DataFrame(records), columns, qc_payload


def _extract_roi_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract ROI-averaged features from precomputed data.
    
    Features per ROI per band:
    - Power (log-ratio normalized) for full period and each time bin
    - ERD/ERS (percent) for full period and each time bin
    - Temporal dynamics (slope, early-late diff)
    """
    from eeg_pipeline.analysis.features.core import build_roi_map
    
    config = precomputed.config
    roi_definitions = config.get("rois", {})
    if not roi_definitions:
        roi_definitions = config.get("time_frequency_analysis.rois", {})
    
    if not roi_definitions:
        return pd.DataFrame(), []
    
    roi_map = build_roi_map(precomputed.ch_names, roi_definitions)
    
    if not roi_map:
        return pd.DataFrame(), []
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    windows = precomputed.windows
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            
            for roi_name, ch_indices in roi_map.items():
                # ROI-averaged power across channels
                roi_power = np.mean(power[ch_indices], axis=0)  # (times,)
                roi_baseline = np.mean(roi_power[windows.baseline_mask])
                roi_active = np.mean(roi_power[windows.active_mask])
                
                # Full period features
                if roi_baseline > epsilon:
                    roi_logratio = np.log10(roi_active / roi_baseline)
                    roi_erds = ((roi_active - roi_baseline) / roi_baseline) * 100
                else:
                    roi_logratio = np.nan
                    roi_erds = np.nan
                
                record[f"roi_power_{band}_{roi_name}_full_logratio"] = float(roi_logratio)
                record[f"roi_erds_{band}_{roi_name}_full_percent"] = float(roi_erds)
                
                # Coarse temporal bins
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        roi_win = np.mean(roi_power[win_mask])
                        if roi_baseline > epsilon:
                            win_logratio = np.log10(roi_win / roi_baseline)
                            win_erds = ((roi_win - roi_baseline) / roi_baseline) * 100
                        else:
                            win_logratio = np.nan
                            win_erds = np.nan
                        record[f"roi_power_{band}_{roi_name}_{win_label}_logratio"] = float(win_logratio)
                        record[f"roi_erds_{band}_{roi_name}_{win_label}_percent"] = float(win_erds)
                        coarse_values[win_label] = win_erds
                
                # Fine temporal bins
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        roi_win = np.mean(roi_power[win_mask])
                        if roi_baseline > epsilon:
                            win_logratio = np.log10(roi_win / roi_baseline)
                        else:
                            win_logratio = np.nan
                        record[f"roi_power_{band}_{roi_name}_{win_label}_logratio"] = float(win_logratio)
                
                # Temporal dynamics
                if "early" in coarse_values and "late" in coarse_values:
                    diff = coarse_values["late"] - coarse_values["early"]
                    record[f"roi_erds_{band}_{roi_name}_early_late_diff"] = float(diff)
        
        records.append(record)
    
    return _build_records_to_df(records)


def _extract_temporal_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract time-domain features from precomputed band-filtered data.
    
    Features: var, std, skew, kurtosis, RMS, peak-to-peak, MAD,
    zero-crossings, line length, nonlinear energy per band/channel.
    """
    err = _validate_precomputed_for_extraction(
        precomputed, require_baseline=False, context="Temporal"
    )
    if err:
        if precomputed and precomputed.logger:
            precomputed.logger.debug(err)
        return _EMPTY_RESULT
    
    from scipy import stats as scipy_stats
    
    def _compute_temporal_for_epoch(ep_idx: int) -> Dict[str, float]:
        """Compute temporal features for a single epoch."""
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            band_filtered = precomputed.band_data[band].filtered[ep_idx]  # (channels, times)
            
            all_var = []
            all_rms = []
            all_skew = []
            all_kurt = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                ch_data = band_filtered[ch_idx, precomputed.windows.active_mask]
                
                # === Statistical moments ===
                ch_var = np.var(ch_data)
                ch_std = np.std(ch_data)
                ch_skew = scipy_stats.skew(ch_data)
                ch_kurt = scipy_stats.kurtosis(ch_data)
                
                record[f"var_{band}_{ch_name}"] = float(ch_var)
                record[f"std_{band}_{ch_name}"] = float(ch_std)
                record[f"skew_{band}_{ch_name}"] = float(ch_skew)
                record[f"kurt_{band}_{ch_name}"] = float(ch_kurt)
                
                all_var.append(ch_var)
                all_skew.append(ch_skew)
                all_kurt.append(ch_kurt)
                
                # === Amplitude features ===
                ch_rms = np.sqrt(np.mean(ch_data ** 2))
                ch_ptp = np.max(ch_data) - np.min(ch_data)
                ch_mad = np.mean(np.abs(ch_data - np.mean(ch_data)))
                
                record[f"rms_{band}_{ch_name}"] = float(ch_rms)
                record[f"ptp_{band}_{ch_name}"] = float(ch_ptp)
                record[f"mad_{band}_{ch_name}"] = float(ch_mad)
                
                all_rms.append(ch_rms)
                
                # === Waveform features ===
                signs = np.sign(ch_data)
                signs[signs == 0] = 1
                zc = np.sum(np.diff(signs) != 0)
                record[f"zerocross_{band}_{ch_name}"] = float(zc)
                
                duration = len(ch_data) / sfreq if sfreq > 0 else 1.0
                record[f"zerocross_rate_{band}_{ch_name}"] = float(zc / duration) if duration > 0 else np.nan
                
                line_len = np.sum(np.abs(np.diff(ch_data)))
                record[f"linelen_{band}_{ch_name}"] = float(line_len)
                
                if len(ch_data) >= 3:
                    nle = ch_data[1:-1] ** 2 - ch_data[:-2] * ch_data[2:]
                    record[f"nle_{band}_{ch_name}"] = float(np.mean(nle))
                else:
                    record[f"nle_{band}_{ch_name}"] = np.nan
                
                # Update naming for consistency
                record[f"temp_var_{band}_{ch_name}_full"] = record.pop(f"var_{band}_{ch_name}", np.nan)
                record[f"temp_rms_{band}_{ch_name}_full"] = record.pop(f"rms_{band}_{ch_name}", np.nan)
                record[f"temp_ptp_{band}_{ch_name}_full"] = record.pop(f"ptp_{band}_{ch_name}", np.nan)
                record[f"temp_mad_{band}_{ch_name}_full"] = record.pop(f"mad_{band}_{ch_name}", np.nan)
                record[f"temp_zerocross_{band}_{ch_name}_full"] = record.pop(f"zerocross_{band}_{ch_name}", np.nan)
                record[f"temp_zerocross_rate_{band}_{ch_name}_full"] = record.pop(f"zerocross_rate_{band}_{ch_name}", np.nan)
                record[f"temp_linelen_{band}_{ch_name}_full"] = record.pop(f"linelen_{band}_{ch_name}", np.nan)
                record[f"temp_nle_{band}_{ch_name}_full"] = record.pop(f"nle_{band}_{ch_name}", np.nan)
                record[f"temp_skew_{band}_{ch_name}_full"] = record.pop(f"skew_{band}_{ch_name}", np.nan)
                record[f"temp_kurt_{band}_{ch_name}_full"] = record.pop(f"kurt_{band}_{ch_name}", np.nan)
                
                # === Per-window features (coarse bins) ===
                for win_mask, win_label in zip(
                    precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
                ):
                    if np.any(win_mask):
                        ch_win = band_filtered[ch_idx, win_mask]
                        record[f"temp_var_{band}_{ch_name}_{win_label}"] = float(np.var(ch_win))
                        record[f"temp_rms_{band}_{ch_name}_{win_label}"] = float(np.sqrt(np.mean(ch_win ** 2)))
                
                # === Per-window features (fine bins for HRF) ===
                for win_mask, win_label in zip(
                    precomputed.windows.fine_masks, precomputed.windows.fine_labels
                ):
                    if np.any(win_mask):
                        ch_win = band_filtered[ch_idx, win_mask]
                        record[f"temp_var_{band}_{ch_name}_{win_label}"] = float(np.var(ch_win))
            
            # === Global statistics per band ===
            valid_var = [v for v in all_var if np.isfinite(v)]
            valid_rms = [v for v in all_rms if np.isfinite(v)]
            valid_skew = [v for v in all_skew if np.isfinite(v)]
            valid_kurt = [v for v in all_kurt if np.isfinite(v)]

            record[f"temp_var_{band}_global_full_mean"] = float(np.mean(valid_var)) if valid_var else np.nan
            record[f"temp_var_{band}_global_full_std"] = float(np.std(valid_var)) if len(valid_var) > 1 else np.nan
            record[f"temp_rms_{band}_global_full_mean"] = float(np.mean(valid_rms)) if valid_rms else np.nan
            record[f"temp_rms_{band}_global_full_std"] = float(np.std(valid_rms)) if len(valid_rms) > 1 else np.nan
            record[f"temp_skew_{band}_global_full_mean"] = float(np.mean(valid_skew)) if valid_skew else np.nan
            record[f"temp_kurt_{band}_global_full_mean"] = float(np.mean(valid_kurt)) if valid_kurt else np.nan
        
        return record
    
    n_epochs = precomputed.data.shape[0]
    sfreq = precomputed.sfreq
    config = precomputed.config or {}
    n_jobs_temporal = int(config.get("feature_engineering.parallel.n_jobs_temporal", 1))
    
    if n_jobs_temporal > 1:
        try:
            from joblib import Parallel, delayed
            if precomputed.logger:
                precomputed.logger.debug(f"Parallel temporal extraction: {n_epochs} epochs with {n_jobs_temporal} workers")
            records = Parallel(n_jobs=n_jobs_temporal, prefer="processes")(
                delayed(_compute_temporal_for_epoch)(ep_idx) for ep_idx in range(n_epochs)
            )
        except Exception as exc:
            if precomputed.logger:
                precomputed.logger.warning(f"Parallel temporal extraction failed ({exc}); falling back to sequential.")
            records = [_compute_temporal_for_epoch(ep_idx) for ep_idx in range(n_epochs)]
    else:
        records = [_compute_temporal_for_epoch(ep_idx) for ep_idx in range(n_epochs)]
    
    return _build_records_to_df(records)


def _extract_complexity_from_data(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract complexity features: permutation entropy, Hjorth parameters,
    Lempel-Ziv complexity per band/channel.
    """
    err = _validate_precomputed_for_extraction(
        precomputed, require_baseline=False, context="Complexity"
    )
    if err:
        if precomputed and precomputed.logger:
            precomputed.logger.debug(err)
        return _EMPTY_RESULT
    
    # Import from complexity module
    from eeg_pipeline.analysis.features.complexity import (
        _permutation_entropy,
        _hjorth_parameters,
        _lempel_ziv_complexity,
    )
    
    def _compute_complexity_for_epoch(ep_idx: int) -> Dict[str, float]:
        """Compute complexity features for a single epoch."""
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            band_filtered = precomputed.band_data[band].filtered[ep_idx]  # (channels, times)
            
            pe_vals = []
            mob_vals = []
            comp_vals = []
            act_vals = []
            lz_vals = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                ch_data = band_filtered[ch_idx, precomputed.windows.active_mask]
                
                # === Permutation Entropy ===
                pe = _permutation_entropy(ch_data, order=3, delay=1, normalize=True)
                record[f"pe_{band}_{ch_name}"] = float(pe) if np.isfinite(pe) else np.nan
                if np.isfinite(pe):
                    pe_vals.append(pe)
                
                # === Hjorth Parameters ===
                act, mob, comp = _hjorth_parameters(ch_data)
                record[f"hjorth_activity_{band}_{ch_name}"] = float(act) if np.isfinite(act) else np.nan
                record[f"hjorth_mobility_{band}_{ch_name}"] = float(mob) if np.isfinite(mob) else np.nan
                record[f"hjorth_complexity_{band}_{ch_name}"] = float(comp) if np.isfinite(comp) else np.nan
                
                if np.isfinite(act):
                    act_vals.append(act)
                if np.isfinite(mob):
                    mob_vals.append(mob)
                if np.isfinite(comp):
                    comp_vals.append(comp)
                
                # === Lempel-Ziv Complexity ===
                lz = _lempel_ziv_complexity(ch_data)
                record[f"comp_lzc_{band}_{ch_name}_full"] = float(lz) if np.isfinite(lz) else np.nan
                if np.isfinite(lz):
                    lz_vals.append(lz)
                
                # Update naming for consistency
                record[f"comp_pe_{band}_{ch_name}_full"] = record.pop(f"pe_{band}_{ch_name}", np.nan)
                record[f"comp_hjorth_act_{band}_{ch_name}_full"] = record.pop(f"hjorth_activity_{band}_{ch_name}", np.nan)
                record[f"comp_hjorth_mob_{band}_{ch_name}_full"] = record.pop(f"hjorth_mobility_{band}_{ch_name}", np.nan)
                record[f"comp_hjorth_comp_{band}_{ch_name}_full"] = record.pop(f"hjorth_complexity_{band}_{ch_name}", np.nan)
                
                # === Per-window complexity (coarse bins) ===
                for win_mask, win_label in zip(
                    precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
                ):
                    if np.any(win_mask):
                        ch_win = band_filtered[ch_idx, win_mask]
                        pe_win = _permutation_entropy(ch_win, order=3, delay=1, normalize=True)
                        record[f"comp_pe_{band}_{ch_name}_{win_label}"] = float(pe_win) if np.isfinite(pe_win) else np.nan
                
                # === Per-window complexity (fine bins for HRF) ===
                for win_mask, win_label in zip(
                    precomputed.windows.fine_masks, precomputed.windows.fine_labels
                ):
                    if np.any(win_mask):
                        ch_win = band_filtered[ch_idx, win_mask]
                        pe_win = _permutation_entropy(ch_win, order=3, delay=1, normalize=True)
                        record[f"comp_pe_{band}_{ch_name}_{win_label}"] = float(pe_win) if np.isfinite(pe_win) else np.nan
            
            # === Global summaries per band ===
            record[f"comp_pe_{band}_global_full_mean"] = float(np.mean(pe_vals)) if pe_vals else np.nan
            record[f"comp_pe_{band}_global_full_std"] = float(np.std(pe_vals)) if len(pe_vals) > 1 else np.nan
            record[f"comp_hjorth_act_{band}_global_full_mean"] = float(np.mean(act_vals)) if act_vals else np.nan
            record[f"comp_hjorth_mob_{band}_global_full_mean"] = float(np.mean(mob_vals)) if mob_vals else np.nan
            record[f"comp_hjorth_comp_{band}_global_full_mean"] = float(np.mean(comp_vals)) if comp_vals else np.nan
            record[f"comp_lzc_{band}_global_full_mean"] = float(np.mean(lz_vals)) if lz_vals else np.nan
            record[f"comp_lzc_{band}_global_full_std"] = float(np.std(lz_vals)) if len(lz_vals) > 1 else np.nan
        
        return record
    
    n_epochs = precomputed.data.shape[0]
    config = precomputed.config or {}
    n_jobs_complexity = int(config.get("feature_engineering.parallel.n_jobs_complexity", 1))
    
    if n_jobs_complexity > 1:
        try:
            from joblib import Parallel, delayed
            if precomputed.logger:
                precomputed.logger.debug(f"Parallel complexity extraction: {n_epochs} epochs with {n_jobs_complexity} workers")
            records = Parallel(n_jobs=n_jobs_complexity, prefer="processes")(
                delayed(_compute_complexity_for_epoch)(ep_idx) for ep_idx in range(n_epochs)
            )
        except Exception as exc:
            if precomputed.logger:
                precomputed.logger.warning(f"Parallel complexity extraction failed ({exc}); falling back to sequential.")
            records = [_compute_complexity_for_epoch(ep_idx) for ep_idx in range(n_epochs)]
    else:
        records = [_compute_complexity_for_epoch(ep_idx) for ep_idx in range(n_epochs)]
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_aperiodic_from_precomputed(
    precomputed: PrecomputedData,
    epochs: "mne.Epochs",
    bands: List[str],
    events_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract aperiodic (1/f) features using the robust fitter with peak rejection and QC.
    """
    if precomputed.windows is None or epochs is None:
        if precomputed.logger:
            precomputed.logger.warning("Aperiodic extraction skipped: missing windows or epochs.")
        return pd.DataFrame(), [], {}
    
    cfg = precomputed.config or {}
    tf_cfg = cfg.get("time_frequency_analysis", {})
    try:
        baseline_window = tuple(
            float(x) for x in tf_cfg.get("baseline_window", (-5.0, -0.01))[:2]
        )
    except Exception:
        baseline_window = (-5.0, -0.01)
    
    df, cols, qc_payload = extract_aperiodic_features(
        epochs=epochs,
        baseline_window=baseline_window,
        bands=bands,
        config=cfg,
        logger=precomputed.logger,
        events_df=events_df,
    )
    return df, cols, qc_payload


def _extract_ratios_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract band power ratios from precomputed data."""
    err = _validate_precomputed_for_extraction(
        precomputed, require_baseline=False, context="Ratios"
    )
    if err:
        if precomputed and precomputed.logger:
            precomputed.logger.debug(err)
        return _EMPTY_RESULT
    
    epsilon = EPSILON_STD
    
    # Define ratio pairs
    ratio_pairs = [
        ("theta", "beta"),    # Attention/ADHD marker
        ("alpha", "theta"),   # Alertness
        ("alpha", "beta"),    # Relaxation vs activation
        ("delta", "alpha"),   # Slow vs fast
        ("gamma", "alpha"),   # High-frequency vs alpha
    ]
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        # Get band powers for this epoch
        band_powers: Dict[str, np.ndarray] = {}
        for band in bands:
            if band in precomputed.band_data:
                power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
                active_mask = precomputed.windows.active_mask
                means = []
                for ch_idx in range(len(precomputed.ch_names)):
                    mean_val, frac, _, _ = _nanmean_with_fraction(power[ch_idx], active_mask)
                    means.append(mean_val if frac >= MIN_VALID_FRACTION else np.nan)
                band_powers[band] = np.array(means)
        
        # Compute ratios
        for num_band, denom_band in ratio_pairs:
            if num_band not in band_powers or denom_band not in band_powers:
                continue
            
            num_power = band_powers[num_band]
            denom_power = band_powers[denom_band]
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                if denom_power[ch_idx] > epsilon:
                    ratio = num_power[ch_idx] / denom_power[ch_idx]
                else:
                    ratio = np.nan
                record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = float(ratio)
            
            # Global ratio
            num_mean = float(np.nanmean(num_power))
            denom_mean = float(np.nanmean(denom_power))
            if denom_mean > epsilon:
                record[f"ratio_{num_band}_{denom_band}_global"] = float(num_mean / denom_mean)
            else:
                record[f"ratio_{num_band}_{denom_band}_global"] = np.nan
        
        records.append(record)
    
    return _build_records_to_df(records)


def _extract_effectsizes_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Compute baseline-vs-active effect sizes (Cohen's d) and log-ratios per band/channel."""
    err = _validate_precomputed_for_extraction(precomputed, context="EffectSizes")
    if err:
        if precomputed and precomputed.logger:
            precomputed.logger.debug(err)
        return _EMPTY_RESULT_WITH_QC

    records: List[Dict[str, float]] = []
    qc_payload: Dict[str, Dict[str, Any]] = {}
    epsilon = EPSILON_STD
    n_epochs = precomputed.data.shape[0]

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        for band in bands:
            if band not in precomputed.band_data:
                continue
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            baseline_std_zero = 0
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_vals = power[ch_idx, precomputed.windows.baseline_mask]
                active_vals = power[ch_idx, precomputed.windows.active_mask]
                logratio_name = make_power_name(band, ch_name, "full", "logratioES")
                cohend_name = make_power_name(band, ch_name, "full", "cohend")
                if baseline_vals.size == 0 or active_vals.size == 0:
                    record[logratio_name] = np.nan
                    record[cohend_name] = np.nan
                    continue

                baseline_mean = float(np.nanmean(baseline_vals))
                active_mean = float(np.nanmean(active_vals))
                baseline_std = float(np.nanstd(baseline_vals, ddof=1)) if baseline_vals.size > 1 else 0.0
                active_std = float(np.nanstd(active_vals, ddof=1)) if active_vals.size > 1 else 0.0
                baseline_finite = int(np.sum(np.isfinite(baseline_vals)))
                active_finite = int(np.sum(np.isfinite(active_vals)))
                if baseline_finite < max(1, int(MIN_VALID_FRACTION * max(1, baseline_vals.size))) or active_finite < max(1, int(MIN_VALID_FRACTION * max(1, active_vals.size))):
                    record[logratio_name] = np.nan
                    record[cohend_name] = np.nan
                    baseline_std_zero += 1
                    continue
                pooled_sd = np.sqrt(
                    ((baseline_finite - 1) * baseline_std ** 2 + (active_finite - 1) * active_std ** 2)
                    / max(baseline_finite + active_finite - 2, 1)
                )
                if pooled_sd < epsilon:
                    baseline_std_zero += 1
                    cohen_d = np.nan
                else:
                    cohen_d = (active_mean - baseline_mean) / pooled_sd

                logratio = np.nan
                if baseline_mean > epsilon:
                    logratio = np.log10(max(active_mean, epsilon) / baseline_mean)

                record[logratio_name] = float(logratio)
                record[cohend_name] = float(cohen_d)

            qc_payload.setdefault(band, {"baseline_zero_std_channels": []})
            qc_payload[band]["baseline_zero_std_channels"].append(baseline_std_zero)

        records.append(record)

    qc_summary: Dict[str, Any] = {}
    for band, stats in qc_payload.items():
        zeros = stats["baseline_zero_std_channels"]
        qc_summary[band] = {
            "median_zero_std_channels": float(np.median(zeros)) if zeros else 0.0,
            "max_zero_std_channels": int(np.max(zeros)) if zeros else 0,
        }

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, qc_summary


def _extract_asymmetry_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract hemispheric asymmetry features."""
    from eeg_pipeline.analysis.features.core import build_roi_map
    
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []
    
    config = precomputed.config
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    
    # Get ROI definitions
    roi_definitions = config.get("rois", {})
    if not roi_definitions:
        roi_definitions = config.get("time_frequency_analysis.rois", {})
    
    # Find left-right pairs - support both formats:
    # 1. Simple channel pairs: [["F3", "F4"], ["C3", "C4"], ...]
    # 2. ROI-based dicts: [{"left": "ROI_L", "right": "ROI_R", "name": "roi"}, ...]
    raw_pairs = config.get("feature_engineering.roi_features.asymmetry_pairs", [
        ["F3", "F4"], ["C3", "C4"], ["P3", "P4"], ["O1", "O2"]
    ])
    
    # Normalize to list of dicts
    asymmetry_pairs = []
    for pair in raw_pairs:
        if isinstance(pair, dict):
            # Already in dict format
            asymmetry_pairs.append(pair)
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            # Simple channel pair format: ["F3", "F4"]
            left_ch, right_ch = pair
            asymmetry_pairs.append({
                "left": left_ch,
                "right": right_ch,
                "name": f"{left_ch}_{right_ch}",
                "is_channel": True,  # Flag to indicate direct channel lookup
            })
    
    # Build ROI channel map using shared utility (for ROI-based pairs)
    roi_map = build_roi_map(precomputed.ch_names, roi_definitions)
    
    # Build channel name to index map (for direct channel pairs)
    ch_to_idx = {ch: idx for idx, ch in enumerate(precomputed.ch_names)}
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            
            for pair in asymmetry_pairs:
                left_key = pair.get("left", "")
                right_key = pair.get("right", "")
                pair_name = pair.get("name", f"{left_key}_{right_key}")
                is_channel = pair.get("is_channel", False)
                
                # Get indices based on whether this is a channel or ROI pair
                if is_channel:
                    left_indices = [ch_to_idx[left_key]] if left_key in ch_to_idx else []
                    right_indices = [ch_to_idx[right_key]] if right_key in ch_to_idx else []
                else:
                    left_indices = roi_map.get(left_key, [])
                    right_indices = roi_map.get(right_key, [])
                
                if not left_indices or not right_indices:
                    record[f"asym_{band}_{pair_name}_full_index"] = np.nan
                    continue
                
                # Full period asymmetry
                active_power = np.mean(power[:, precomputed.windows.active_mask], axis=1)
                left_power = np.mean(active_power[left_indices])
                right_power = np.mean(active_power[right_indices])
                
                denom = left_power + right_power
                if denom > epsilon:
                    asymmetry = (right_power - left_power) / denom
                else:
                    asymmetry = np.nan
                
                record[f"asym_{band}_{pair_name}_full_index"] = float(asymmetry)
                
                # Coarse temporal bins
                coarse_asym = {}
                for win_mask, win_label in zip(
                    precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
                ):
                    if np.any(win_mask):
                        win_power = np.mean(power[:, win_mask], axis=1)
                        left_win = np.mean(win_power[left_indices])
                        right_win = np.mean(win_power[right_indices])
                        denom_win = left_win + right_win
                        if denom_win > epsilon:
                            asym_win = (right_win - left_win) / denom_win
                        else:
                            asym_win = np.nan
                        record[f"asym_{band}_{pair_name}_{win_label}_index"] = float(asym_win)
                        coarse_asym[win_label] = asym_win
                
                # Fine temporal bins
                for win_mask, win_label in zip(
                    precomputed.windows.fine_masks, precomputed.windows.fine_labels
                ):
                    if np.any(win_mask):
                        win_power = np.mean(power[:, win_mask], axis=1)
                        left_win = np.mean(win_power[left_indices])
                        right_win = np.mean(win_power[right_indices])
                        denom_win = left_win + right_win
                        if denom_win > epsilon:
                            asym_win = (right_win - left_win) / denom_win
                        else:
                            asym_win = np.nan
                        record[f"asym_{band}_{pair_name}_{win_label}_index"] = float(asym_win)
                
                # Temporal dynamics
                if "early" in coarse_asym and "late" in coarse_asym:
                    if np.isfinite(coarse_asym["early"]) and np.isfinite(coarse_asym["late"]):
                        diff = coarse_asym["late"] - coarse_asym["early"]
                        record[f"asym_{band}_{pair_name}_early_late_diff"] = float(diff)
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_itpc_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Inter-Trial Phase Coherence from precomputed phase data.
    
    Features per channel/band:
    - ITPC for full active period
    - ITPC per coarse temporal window (early, mid, late)
    - ITPC per fine temporal window (t1-t7) for HRF modeling
    - Temporal dynamics: early-late diff, peak time
    - Global (cross-channel) statistics
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []

    if not _validate_window_masks(precomputed, precomputed.logger, require_baseline=False):
        return pd.DataFrame(), []
    
    n_epochs = precomputed.data.shape[0]
    if n_epochs < 5:
        return pd.DataFrame(), []
    
    windows = precomputed.windows
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    if not np.any(windows.active_mask):
        return pd.DataFrame(), []

    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    for band in bands:
        if band not in precomputed.band_data:
            continue

        phase = precomputed.band_data[band].phase  # (epochs, channels, times)
        if phase.shape[0] < 2:
            continue

        # Use unit vectors; mask non-finite values so they don't bias means
        unit_vectors = np.exp(1j * phase)
        unit_vectors = np.where(np.isfinite(unit_vectors), unit_vectors, np.nan)

        def _compute_window_itpc(win_mask: np.ndarray):
            if not np.any(win_mask):
                return None
            uv_win = unit_vectors[:, :, win_mask]  # (epochs, channels, times_win)
            valid = np.isfinite(uv_win)
            if not np.any(valid):
                return None
            sum_all = np.nansum(uv_win, axis=0)       # (channels, times_win)
            count_all = np.sum(valid, axis=0)         # (channels, times_win)
            per_epoch = []
            for ep_idx in range(n_epochs):
                sum_others = sum_all - np.where(valid[ep_idx], uv_win[ep_idx], 0.0)
                count_others = count_all - valid[ep_idx]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_others = np.where(count_others > 0, sum_others / count_others, np.nan)
                itpc = np.abs(mean_others)  # (channels, times_win)
                itpc_ch = np.nanmean(itpc, axis=1)  # (channels,)
                per_epoch.append((itpc, itpc_ch))
            return per_epoch

        # Full active window
        active_itpc = _compute_window_itpc(windows.active_mask)
        if active_itpc is None:
            continue

        for ep_idx, (itpc_mat, itpc_ch_mean) in enumerate(active_itpc):
            rec = records[ep_idx]
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                rec[f"phase_itpc_{band}_{ch_name}_full_mean"] = float(itpc_ch_mean[ch_idx])
            rec[f"phase_itpc_{band}_global_full_mean"] = float(np.nanmean(itpc_ch_mean))

            # Peak latency from per-epoch time course (mean over channels)
            if itpc_mat.shape[1] > 0:
                time_course = np.nanmean(itpc_mat, axis=0)
                if np.any(np.isfinite(time_course)):
                    peak_idx = int(np.nanargmax(time_course))
                    rec[f"phase_itpc_{band}_peak_latency"] = float(active_times[peak_idx])
                else:
                    rec[f"phase_itpc_{band}_peak_latency"] = np.nan
            else:
                rec[f"phase_itpc_{band}_peak_latency"] = np.nan

        # Coarse windows
        coarse_vals: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
        for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
            win_itpc = _compute_window_itpc(win_mask)
            if win_itpc is None:
                continue
            for ep_idx, (_, itpc_ch_mean) in enumerate(win_itpc):
                rec = records[ep_idx]
                for ch_idx, ch_name in enumerate(precomputed.ch_names):
                    rec[f"phase_itpc_{band}_{ch_name}_{win_label}_mean"] = float(itpc_ch_mean[ch_idx])
                coarse_vals[ep_idx][win_label] = float(np.nanmean(itpc_ch_mean))
                rec[f"phase_itpc_{band}_global_{win_label}_mean"] = coarse_vals[ep_idx][win_label]

        # Fine windows (global only)
        for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
            win_itpc = _compute_window_itpc(win_mask)
            if win_itpc is None:
                continue
            for ep_idx, (_, itpc_ch_mean) in enumerate(win_itpc):
                records[ep_idx][f"phase_itpc_{band}_global_{win_label}_mean"] = float(np.nanmean(itpc_ch_mean))

        # Temporal dynamics: early-late difference per epoch
        for ep_idx in range(n_epochs):
            if "early" in coarse_vals[ep_idx] and "late" in coarse_vals[ep_idx]:
                diff = coarse_vals[ep_idx]["late"] - coarse_vals[ep_idx]["early"]
                records[ep_idx][f"phase_itpc_{band}_global_early_late_diff"] = float(diff)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    columns = sorted({k for r in records for k in r.keys()})
    df = pd.DataFrame(records)
    return df, columns


def _extract_pac_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract simple phase–amplitude coupling features (MVL-based) per channel.
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), [], {}

    cfg = precomputed.config or {}
    pac_cfg = cfg.get("feature_engineering", {}).get("pac", {})
    default_pairs = [("theta", "gamma"), ("alpha", "gamma")]
    band_pairs = pac_cfg.get("pairs", default_pairs)

    records: List[Dict[str, float]] = []
    qc_payload: Dict[str, Dict[str, Any]] = {}
    active_mask = precomputed.windows.active_mask
    n_epochs = precomputed.data.shape[0]

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        for pair in band_pairs:
            if len(pair) != 2:
                continue
            phase_band, amp_band = pair
            if phase_band not in precomputed.band_data or amp_band not in precomputed.band_data:
                continue

            phase_data = precomputed.band_data[phase_band].phase[ep_idx]  # (channels, times)
            amp_env = precomputed.band_data[amp_band].envelope[ep_idx]

            valid_samples_per_ch = []
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                phase_ch = phase_data[ch_idx, active_mask]
                amp_ch = amp_env[ch_idx, active_mask]
                valid_mask = np.isfinite(phase_ch) & np.isfinite(amp_ch)
                valid_samples_per_ch.append(int(np.sum(valid_mask)))
                if not np.any(valid_mask):
                    record[f"pac_mvl_{phase_band}_{amp_band}_{ch_name}"] = np.nan
                    continue
                phase_valid = phase_ch[valid_mask]
                amp_valid = amp_ch[valid_mask]
                amp_norm = amp_valid / (np.mean(amp_valid) + 1e-12)
                mvl = np.abs(np.mean(amp_norm * np.exp(1j * phase_valid)))
                record[f"pac_mvl_{phase_band}_{amp_band}_{ch_name}"] = float(mvl)

            qc_payload.setdefault(f"{phase_band}->{amp_band}", {"median_valid_samples": [], "min_valid_samples": []})
            qc_payload[f"{phase_band}->{amp_band}"]["median_valid_samples"].append(float(np.median(valid_samples_per_ch)))
            qc_payload[f"{phase_band}->{amp_band}"]["min_valid_samples"].append(float(np.min(valid_samples_per_ch)))

        records.append(record)

    qc_summary: Dict[str, Any] = {}
    for pair_name, stats in qc_payload.items():
        qc_summary[pair_name] = {
            "median_valid_samples": float(np.nanmedian(stats["median_valid_samples"])) if stats["median_valid_samples"] else np.nan,
            "min_valid_samples": float(np.nanmin(stats["min_valid_samples"])) if stats["min_valid_samples"] else np.nan,
        }

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, qc_summary


def _extract_cfc_features(
    precomputed: PrecomputedData,
    bands: List[str],
    epochs: "mne.Epochs" = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract cross-frequency coupling features (MI-PAC, PPC)."""
    if epochs is None and precomputed is not None:
        # Cannot run CFC without epochs object for filtering
        return pd.DataFrame(), []
    
    from eeg_pipeline.analysis.features.cfc import (
        extract_modulation_index_pac,
        extract_phase_phase_coupling,
    )
    
    cfg = precomputed.config if precomputed else {}
    cfc_cfg = cfg.get("feature_engineering", {}).get("cross_frequency", {})
    
    phase_bands = cfc_cfg.get("phase_bands", ["theta", "alpha"])
    amp_bands = cfc_cfg.get("amp_bands", ["gamma"])
    ppc_pairs = cfc_cfg.get("phase_phase_pairs", [["theta", "alpha", 1, 2]])
    
    logger = precomputed.logger if precomputed else None
    all_dfs = []
    all_cols = []
    
    # MI-PAC
    if phase_bands and amp_bands:
        try:
            mi_df, mi_cols = extract_modulation_index_pac(
                epochs, phase_bands, amp_bands, cfg, logger
            )
            if not mi_df.empty:
                all_dfs.append(mi_df)
                all_cols.extend(mi_cols)
        except Exception:
            pass
    
    # Phase-phase coupling
    if ppc_pairs:
        try:
            ppc_tuples = [(p[0], p[1], p[2], p[3]) for p in ppc_pairs if len(p) == 4]
            ppc_df, ppc_cols = extract_phase_phase_coupling(
                epochs, ppc_tuples, cfg, logger
            )
            if not ppc_df.empty:
                all_dfs.append(ppc_df)
                all_cols.extend(ppc_cols)
        except Exception:
            pass
    
    if not all_dfs:
        return pd.DataFrame(), []
    
    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols


def _extract_microstates_features(
    precomputed: PrecomputedData,
    epochs: "mne.Epochs",
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract microstate features (coverage, duration, transitions, GEV).
    
    Microstates are quasi-stable topographic patterns that capture
    global brain state dynamics. Features include:
    - Coverage: fraction of time in each state
    - Duration: mean duration of each state
    - Occurrence: rate of state occurrences
    - GEV: global explained variance per state
    - Transitions: state transition probabilities
    """
    if epochs is None:
        if logger:
            logger.info("Microstates: skipped (no epochs provided)")
        return pd.DataFrame(), []
    
    from eeg_pipeline.analysis.features.microstates import extract_microstate_features
    
    cfg = config if config else (precomputed.config if precomputed else {})
    n_states = int(cfg.get("feature_engineering.microstates.n_states", 4))
    
    try:
        ms_df, ms_cols, templates = extract_microstate_features(
            epochs, n_states, cfg, logger
        )
        if ms_df.empty:
            if logger:
                logger.info("Microstates: returned empty DataFrame")
        return ms_df, ms_cols
    except Exception as exc:
        if logger:
            logger.warning(f"Microstates extraction failed: {exc}")
        return pd.DataFrame(), []


def _extract_quality_features(
    precomputed: PrecomputedData,
    epochs: "mne.Epochs" = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract trial quality metrics."""
    if epochs is None:
        return pd.DataFrame(), []
    
    from eeg_pipeline.analysis.features.quality import compute_trial_quality_metrics
    
    cfg = precomputed.config if precomputed else {}
    logger = precomputed.logger if precomputed else None
    
    try:
        quality_df = compute_trial_quality_metrics(epochs, cfg, logger)
        if quality_df.empty:
            return pd.DataFrame(), []
        cols = [c for c in quality_df.columns if c != "epoch"]
        return quality_df[cols], cols
    except Exception:
        return pd.DataFrame(), []


def extract_precomputed_features(
    epochs_or_ctx: Union["mne.Epochs", FeatureExtractionContext, PrecomputedData],
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    feature_groups: Optional[List[str]] = None,
    n_plateau_windows: int = 5,
    events_df: Optional[pd.DataFrame] = None,
) -> ExtractionResult:
    """
    Extract EEG features efficiently using precomputed intermediates.
    
    This function provides efficient extraction by precomputing expensive
    operations (filtering, PSD, GFP) once and reusing them. Use this for
    extracting multiple feature types from the same epochs.
    
    Output is epoch-aligned: one row per epoch with a 'condition' column
    indicating 'pain' or 'nonpain' for each trial (if events_df is provided).
    
    Parameters
    ----------
    epochs_or_ctx : mne.Epochs | FeatureExtractionContext | PrecomputedData
        Input epochs or a pre-built FeatureExtractionContext/PrecomputedData to reuse preprocessing
    bands : List[str]
        Frequency bands (e.g., ["delta", "theta", "alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance
        feature_groups : Optional[List[str]]
            Which feature groups to extract. Default: all.
            Options: "erds", "spectral", "gfp", "connectivity", "roi", "temporal", 
                     "complexity", "aperiodic", "ratios", "asymmetry", "itpc",
                     "effectsize", "pac"
    n_plateau_windows : int
        Number of temporal windows for windowed features
    events_df : Optional[pd.DataFrame]
        Events DataFrame with pain/condition labels. If provided, adds a
        'condition' column to the output indicating 'pain' or 'nonpain'.
        
    Returns
    -------
    ExtractionResult
        Container with epoch-aligned features:
        - result.get_combined_df() - all features with condition column
        - result.get_feature_group_df("erds") - specific feature group
    """
    ctx: Optional[FeatureExtractionContext] = None
    epochs: Optional["mne.Epochs"] = None
    precomputed: Optional[PrecomputedData] = None

    if isinstance(epochs_or_ctx, FeatureExtractionContext):
        ctx = epochs_or_ctx
        epochs = ctx.epochs
        config = ctx.config or config
        logger = ctx.logger or logger
        if not ctx.ensure_precomputed():
            logger.error("FeatureExtractionContext did not provide precomputed data; aborting extraction.")
            return ExtractionResult()
        precomputed = ctx.precomputed
    elif isinstance(epochs_or_ctx, PrecomputedData):
        precomputed = epochs_or_ctx
    else:
        epochs = epochs_or_ctx

    all_groups = [
        "power", "erds", "spectral", "gfp", "connectivity", "roi", 
        "temporal", "complexity", "aperiodic", "ratios", "asymmetry", "itpc",
        "effectsize", "pac", "cfc", "microstates", "quality",
    ]
    if feature_groups is None:
        feature_groups = all_groups
    else:
        unsupported = set(feature_groups) - set(all_groups)
        if unsupported:
            logger.warning(
                "The following feature groups are not supported in the precomputed pipeline and will be ignored as legacy: %s",
                sorted(unsupported),
            )
            feature_groups = [g for g in feature_groups if g in all_groups]
    
    logger.info(f"Extracting feature groups: {feature_groups}")
    
    # === Get condition labels from events (if provided) ===
    condition: Optional[np.ndarray] = None
    n_epochs_total = (
        len(epochs)
        if epochs is not None
        else (precomputed.data.shape[0] if precomputed is not None and precomputed.data.size > 0 else 0)
    )
    if events_df is not None and not events_df.empty and n_epochs_total > 0:
        from eeg_pipeline.utils.io.general import get_pain_column_from_config
        
        pain_col = get_pain_column_from_config(config, events_df)
        if pain_col is not None and pain_col in events_df.columns:
            pain_values = pd.to_numeric(events_df[pain_col], errors="coerce")
            pain_mask = (pain_values > 0).values
            
            if len(pain_mask) != n_epochs_total:
                logger.warning(
                    f"Events length ({len(pain_mask)}) doesn't match epochs ({n_epochs_total}); "
                    "condition column will not be added"
                )
            else:
                # Create condition labels array
                condition = np.where(pain_mask, "pain", "nonpain")
                n_pain = int(pain_mask.sum())
                n_nonpain = int((~pain_mask).sum())
                logger.info(f"Added condition labels: {n_pain} pain, {n_nonpain} non-pain trials")
        else:
            logger.warning("Pain column not found in events_df; condition column will not be added")
    
    # Determine what needs to be precomputed
    needs_bands = any(g in feature_groups for g in [
        "power", "erds", "connectivity", "roi", "gfp", "ratios", "asymmetry", "itpc", "effectsize", "pac"
    ])
    needs_psd = any(g in feature_groups for g in ["spectral", "aperiodic"])
    
    # Precompute all expensive intermediates ONCE
    if precomputed is None:
        if epochs is None:
            raise ValueError("Either epochs or FeatureExtractionContext/PrecomputedData must be provided.")
        logger.info("Precomputing intermediates (filtering, PSD, GFP)...")
        precomputed = precompute_data(
            epochs, bands, config, logger,
            compute_bands=needs_bands,
            compute_psd_data=needs_psd,
            n_plateau_windows=n_plateau_windows,
        )
    result = ExtractionResult(
        precomputed=precomputed,
        condition=condition,
        qc={"precomputed": precomputed.qc.as_dict() if precomputed is not None else {}},
    )
    
    # Extractor registry: (name, extractor_func, args_builder, has_qc)
    _EXTRACTORS = [
        ("power", _extract_power_from_precomputed, lambda: (precomputed, bands), True),
        ("erds", _extract_erds_from_precomputed, lambda: (precomputed, bands), True),
        ("spectral", _extract_spectral_from_precomputed, lambda: (precomputed, bands), False),
        ("gfp", _extract_gfp_from_precomputed, lambda: (precomputed, bands), False),
        ("connectivity", _extract_connectivity_from_precomputed, lambda: (precomputed, bands), True),
        ("roi", _extract_roi_from_precomputed, lambda: (precomputed, bands), False),
        ("temporal", _extract_temporal_from_precomputed, lambda: (precomputed, bands), False),
        ("complexity", _extract_complexity_from_data, lambda: (precomputed, bands), False),
        ("aperiodic", _extract_aperiodic_from_precomputed, lambda: (precomputed, epochs, bands, events_df), True),
        ("ratios", _extract_ratios_from_precomputed, lambda: (precomputed, bands), False),
        ("effectsize", _extract_effectsizes_from_precomputed, lambda: (precomputed, bands), True),
        ("asymmetry", _extract_asymmetry_from_precomputed, lambda: (precomputed, bands), False),
        ("itpc", _extract_itpc_from_precomputed, lambda: (precomputed, bands), False),
        ("pac", _extract_pac_from_precomputed, lambda: (precomputed, bands), True),
        ("cfc", _extract_cfc_features, lambda: (precomputed, bands, epochs), False),
        ("microstates", _extract_microstates_features, lambda: (precomputed, epochs, config, logger), False),
        ("quality", _extract_quality_features, lambda: (precomputed, epochs), False),
    ]
    
    requested = [(name, extractor, args_builder, has_qc) for name, extractor, args_builder, has_qc in _EXTRACTORS if name in feature_groups]
    n_jobs_groups = int(config.get("feature_engineering.parallel.n_jobs_feature_groups", 1))

    def _run_extractor(task):
        name, extractor, args_builder, has_qc = task
        logger.info(f"Extracting {name} features...")
        try:
            output = extractor(*args_builder())
            if has_qc:
                df, cols, qc_payload = output
            else:
                df, cols = output[:2]
                qc_payload = {}
            return {"name": name, "df": df, "cols": cols, "qc": qc_payload, "error": None}
        except Exception as exc:
            logger.warning(f"Extractor '{name}' failed: {exc}")
            return {"name": name, "df": pd.DataFrame(), "cols": [], "qc": {}, "error": str(exc)}

    if requested:
        if n_jobs_groups > 1:
            try:
                from joblib import Parallel, delayed
                results_list = Parallel(n_jobs=n_jobs_groups, prefer="processes")(
                    delayed(_run_extractor)(task) for task in requested
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Parallel extractor execution failed (%s); falling back to sequential.", exc
                )
                results_list = [_run_extractor(task) for task in requested]
        else:
            results_list = [_run_extractor(task) for task in requested]

        for res in results_list:
            if res["error"]:
                result.qc[res["name"]] = {"error": res["error"]}
                logger.debug(f"Extractor '{res['name']}' failed: {res['error']}")
                continue
            if res["qc"]:
                result.qc[res["name"]] = res["qc"]
            if not res["df"].empty:
                result.features[res["name"]] = FeatureSet(res["df"], res["cols"], res["name"])
                logger.debug(f"Extractor '{res['name']}': {len(res['cols'])} features extracted")
            else:
                logger.info(f"Extractor '{res['name']}' returned empty DataFrame (no features extracted)")
                result.qc[res["name"]] = result.qc.get(res["name"], {})
                if isinstance(result.qc[res["name"]], dict):
                    result.qc[res["name"]]["empty"] = True
    
    total_features = sum(len(fs.columns) for fs in result.features.values())
    logger.info(f"Extraction complete: {total_features} features from {len(result.features)} groups")
    
    return result


def extract_fmri_prediction_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    include_all_bands: bool = False,
    events_df: Optional[pd.DataFrame] = None,
) -> ExtractionResult:
    """
    Extract optimized feature set for EEG-fMRI prediction.
    
    This is a convenience function that extracts the most relevant
    features for predicting fMRI BOLD signals during pain.
    
    Key features for BOLD prediction:
    - ERD/ERS: Alpha/beta desync correlates with BOLD increase
    - ROI: Maps to fMRI brain regions
    - GFP: Global activity correlates with vigilance/arousal
    - Asymmetry: Contralateral activation for unilateral pain
    - Aperiodic: 1/f slope relates to E/I balance and BOLD
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    config : Any
        Configuration object
    logger : Any
        Logger instance
    include_all_bands : bool
        If True, includes all bands. If False (default), uses only
        alpha, beta, gamma which have strongest BOLD correlates.
    events_df : Optional[pd.DataFrame]
        Events DataFrame with pain labels. If provided, adds a 'condition'
        column to the output indicating 'pain' or 'nonpain'.
        
    Returns
    -------
    ExtractionResult
        Container with epoch-aligned features:
        - result.get_combined_df() - all features with condition column
        - result.get_feature_group_df("erds") - specific feature group
    """
    # Key bands for fMRI prediction
    if include_all_bands:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
    else:
        bands = ["alpha", "beta", "gamma"]
    
    # Feature groups most relevant for BOLD prediction
    # Prioritized by strength of EEG-BOLD correlation
    feature_groups = [
        "erds",       # Primary: ERD/ERS directly predicts BOLD
        "roi",        # ROI-averaged for interpretability + dimensionality reduction
        "gfp",        # Global activity
        "asymmetry",  # Lateralization (contralateral pain response)
        "aperiodic",  # 1/f slope (E/I balance, relates to BOLD)
        "spectral",   # Power + relative power
    ]
    
    return extract_precomputed_features(
        epochs, bands, config, logger,
        feature_groups=feature_groups,
        n_plateau_windows=5,  # Captures HRF dynamics (~2s windows)
        events_df=events_df,
    )


def get_feature_groups_for_ml() -> Dict[str, List[str]]:
    """
    Return recommended feature group combinations for different ML scenarios.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping scenario name to list of feature groups
    """
    return {
        # Minimal set for quick prototyping
        "minimal": ["erds", "roi", "gfp"],
        
        # Balanced set for fMRI prediction
        "fmri_prediction": ["erds", "roi", "gfp", "asymmetry", "aperiodic", "spectral"],
        
        # Full set for comprehensive analysis
        "comprehensive": [
            "erds", "spectral", "gfp", "connectivity", "roi",
            "temporal", "complexity", "aperiodic", "ratios", "asymmetry", "itpc",
            "effectsize", "pac", "microstates",
        ],
        
        # Connectivity-focused
        "connectivity": ["connectivity", "roi", "gfp"],
        
        # Pain-specific (emphasizes sensorimotor, asymmetry)
        "pain": ["erds", "roi", "asymmetry", "aperiodic", "spectral"],
        
        # Time-resolved (windowed features for HRF modeling)
        "temporal": ["erds", "roi", "gfp", "temporal"],
        
        # Microstate-focused (global brain state dynamics)
        "microstates": ["microstates", "gfp", "complexity"],
    }
