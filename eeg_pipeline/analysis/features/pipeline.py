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
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.analysis.features.core import (
    PrecomputedData,
    precompute_data,
    EPSILON_STD,
)
from eeg_pipeline.utils.config.loader import get_frequency_bands


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
        
        return combined
    
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


###################################################################
# Feature Extractors (Using Precomputed Data)
###################################################################


def _extract_erds_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
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
        return pd.DataFrame(), []
    
    epsilon = EPSILON_STD
    windows = precomputed.windows
    
    records: List[Dict[str, float]] = []
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
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power = np.mean(power[ch_idx, windows.baseline_mask])
                baseline_std = np.std(power[ch_idx, windows.baseline_mask])
                active_power_trace = power[ch_idx, windows.active_mask]
                active_power_mean = np.mean(active_power_trace)
                
                if baseline_power > epsilon:
                    erds_full = ((active_power_mean - baseline_power) / baseline_power) * 100
                    erds_trace = ((active_power_trace - baseline_power) / baseline_power) * 100
                else:
                    erds_full = np.nan
                    erds_trace = np.full_like(active_power_trace, np.nan)
                
                # Full period ERD/ERS
                record[f"erds_{band}_{ch_name}_full_percent"] = float(erds_full)
                all_erds_full.append(erds_full)
                
                # === Coarse temporal bins (early, mid, late) ===
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_power > epsilon:
                            erds_win = ((win_power - baseline_power) / baseline_power) * 100
                        else:
                            erds_win = np.nan
                        record[f"erds_{band}_{ch_name}_{win_label}_percent"] = float(erds_win)
                        coarse_values[win_label] = erds_win
                
                # === Fine temporal bins (t1-t7) for HRF modeling ===
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_power > epsilon:
                            erds_win = ((win_power - baseline_power) / baseline_power) * 100
                        else:
                            erds_win = np.nan
                        record[f"erds_{band}_{ch_name}_{win_label}_percent"] = float(erds_win)
                
                # === Temporal dynamics ===
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
                    threshold = baseline_std / baseline_power * 100 if baseline_power > epsilon else np.inf
                    onset_mask = np.abs(erds_trace) > threshold
                    if np.any(onset_mask):
                        onset_idx = np.argmax(onset_mask)
                        record[f"erds_{band}_{ch_name}_onset_latency"] = float(active_times[onset_idx])
                    else:
                        record[f"erds_{band}_{ch_name}_onset_latency"] = np.nan
                    
                    # === ERD vs ERS separation ===
                    erd_vals = erds_trace[erds_trace < 0]
                    ers_vals = erds_trace[erds_trace > 0]
                    
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
            if valid_erds:
                record[f"erds_{band}_global_full_mean"] = float(np.mean(valid_erds))
                record[f"erds_{band}_global_full_std"] = float(np.std(valid_erds))
                
                # Global per coarse bin
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_erds = []
                        for ch_idx in range(len(precomputed.ch_names)):
                            bp = np.mean(power[ch_idx, windows.baseline_mask])
                            if bp > epsilon:
                                wp = np.mean(power[ch_idx, win_mask])
                                win_erds.append(((wp - bp) / bp) * 100)
                        if win_erds:
                            record[f"erds_{band}_global_{win_label}_mean"] = float(np.mean(win_erds))
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_power_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
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
        return pd.DataFrame(), []
    
    epsilon = EPSILON_STD
    windows = precomputed.windows
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            all_power_full = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power = np.mean(power[ch_idx, windows.baseline_mask])
                active_power = np.mean(power[ch_idx, windows.active_mask])
                
                # Full period power (log-ratio normalized)
                if baseline_power > epsilon:
                    logratio = np.log10(active_power / baseline_power)
                else:
                    logratio = np.nan
                record[f"power_{band}_{ch_name}_full_logratio"] = float(logratio)
                all_power_full.append(logratio)
                
                # === Coarse temporal bins (early, mid, late) ===
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_power > epsilon:
                            win_logratio = np.log10(win_power / baseline_power)
                        else:
                            win_logratio = np.nan
                        record[f"power_{band}_{ch_name}_{win_label}_logratio"] = float(win_logratio)
                        coarse_values[win_label] = win_logratio
                
                # === Fine temporal bins (t1-t7) for HRF modeling ===
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_power > epsilon:
                            win_logratio = np.log10(win_power / baseline_power)
                        else:
                            win_logratio = np.nan
                        record[f"power_{band}_{ch_name}_{win_label}_logratio"] = float(win_logratio)
                
                # === Temporal dynamics ===
                if len(active_times) > 2:
                    active_power_trace = power[ch_idx, windows.active_mask]
                    if baseline_power > epsilon:
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
            if valid_power:
                record[f"power_{band}_global_full_mean"] = float(np.mean(valid_power))
                record[f"power_{band}_global_full_std"] = float(np.std(valid_power))
                
                # Global per coarse bin
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_powers = []
                        for ch_idx in range(len(precomputed.ch_names)):
                            bp = np.mean(power[ch_idx, windows.baseline_mask])
                            if bp > epsilon:
                                wp = np.mean(power[ch_idx, win_mask])
                                win_powers.append(np.log10(wp / bp))
                        if win_powers:
                            record[f"power_{band}_global_{win_label}_mean"] = float(np.mean(win_powers))
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


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
        return pd.DataFrame(), []
    
    config = precomputed.config
    freq_bands = get_frequency_bands(config)
    freqs = precomputed.psd_data.freqs
    psd = precomputed.psd_data.psd  # (epochs, channels, freqs)
    
    records: List[Dict[str, float]] = []
    n_epochs = psd.shape[0]
    
    # Total power for relative calculations
    total_power = np.trapz(psd, freqs, axis=2)  # (epochs, channels)
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
                        record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = float(num / denom)
                        record[f"logratio_{num_band}_{denom_band}_{ch_name}"] = float(np.log10((num + epsilon) / (denom + epsilon)))
                    else:
                        record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = np.nan
                        record[f"logratio_{num_band}_{denom_band}_{ch_name}"] = np.nan
        
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
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_gfp_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract comprehensive GFP (Global Field Power) features.
    
    GFP measures the spatial standard deviation across channels at each time point,
    reflecting overall brain activity level. Features include:
    - Basic statistics (mean, std, min, max, range)
    - Temporal dynamics (slope, peak count, peak latency)
    - Percentiles for robust statistics
    - Baseline-normalized metrics
    - Per-window features for temporal resolution
    """
    if precomputed.gfp is None or precomputed.windows is None:
        return pd.DataFrame(), []
    
    from scipy.signal import find_peaks
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.gfp.shape[0]
    times = precomputed.times
    active_times = times[precomputed.windows.active_mask]
    epsilon = 1e-12
    
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
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_connectivity_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract comprehensive connectivity features using precomputed analytic signal.
    
    Features include:
    - wPLI (weighted Phase Lag Index): robust to volume conduction
    - PLV (Phase Locking Value): overall phase synchronization
    - AEC (Amplitude Envelope Correlation): amplitude-based connectivity
    - Graph metrics: mean degree, clustering coefficient, network efficiency
    - Statistics: mean, std, max, percentiles across all pairs
    - Per-window connectivity for temporal dynamics
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    n_channels = len(precomputed.ch_names)
    triu_idx = np.triu_indices(n_channels, k=1)
    n_pairs = len(triu_idx[0])
    epsilon = 1e-12
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            analytic = precomputed.band_data[band].analytic[ep_idx]  # (channels, times)
            analytic_active = analytic[:, precomputed.windows.active_mask]
            envelope = precomputed.band_data[band].envelope[ep_idx]
            envelope_active = envelope[:, precomputed.windows.active_mask]
            phases = precomputed.band_data[band].phase[ep_idx, :, precomputed.windows.active_mask]
            
            # === wPLI computation ===
            cross = analytic_active[:, None, :] * np.conj(analytic_active[None, :, :])
            imag_cross = np.imag(cross)
            denom = np.mean(np.abs(imag_cross), axis=-1)
            numer = np.abs(np.mean(imag_cross, axis=-1))
            
            with np.errstate(divide="ignore", invalid="ignore"):
                wpli = np.where(denom > 0, numer / denom, 0.0)
            wpli = 0.5 * (wpli + wpli.T)
            np.fill_diagonal(wpli, 0.0)
            wpli_values = wpli[triu_idx]
            
            # wPLI statistics
            record[f"wpli_{band}_mean"] = float(np.mean(wpli_values))
            record[f"wpli_{band}_std"] = float(np.std(wpli_values))
            record[f"wpli_{band}_max"] = float(np.max(wpli_values))
            record[f"wpli_{band}_min"] = float(np.min(wpli_values))
            record[f"wpli_{band}_median"] = float(np.median(wpli_values))
            for pct in [25, 75, 90]:
                record[f"wpli_{band}_p{pct}"] = float(np.percentile(wpli_values, pct))
            
            # === PLV computation (vectorized) ===
            phase_diff = phases[:, None, :] - phases[None, :, :]
            plv_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
            plv_values = plv_matrix[triu_idx]
            
            # PLV statistics
            record[f"plv_{band}_mean"] = float(np.mean(plv_values))
            record[f"plv_{band}_std"] = float(np.std(plv_values))
            record[f"plv_{band}_max"] = float(np.max(plv_values))
            record[f"plv_{band}_median"] = float(np.median(plv_values))
            for pct in [25, 75, 90]:
                record[f"plv_{band}_p{pct}"] = float(np.percentile(plv_values, pct))
            
            # === AEC (Amplitude Envelope Correlation) ===
            # Orthogonalized AEC to reduce volume conduction effects
            env_centered = envelope_active - np.mean(envelope_active, axis=1, keepdims=True)
            env_std = np.std(envelope_active, axis=1, keepdims=True)
            env_std = np.where(env_std < epsilon, epsilon, env_std)
            env_norm = env_centered / env_std
            
            # Compute correlation matrix
            aec_matrix = np.corrcoef(env_norm)
            np.fill_diagonal(aec_matrix, 0.0)
            aec_values = aec_matrix[triu_idx]
            aec_values = np.clip(aec_values, -1, 1)  # Ensure valid correlation range
            
            # AEC statistics
            record[f"aec_{band}_mean"] = float(np.nanmean(aec_values))
            record[f"aec_{band}_std"] = float(np.nanstd(aec_values))
            record[f"aec_{band}_max"] = float(np.nanmax(aec_values))
            record[f"aec_{band}_abs_mean"] = float(np.nanmean(np.abs(aec_values)))
            
            # === Graph metrics ===
            # Mean degree (average connectivity strength per node)
            node_strength_wpli = np.sum(wpli, axis=1) / (n_channels - 1)
            node_strength_plv = np.sum(plv_matrix, axis=1) / (n_channels - 1)
            
            record[f"graph_{band}_wpli_mean_degree"] = float(np.mean(node_strength_wpli))
            record[f"graph_{band}_wpli_std_degree"] = float(np.std(node_strength_wpli))
            record[f"graph_{band}_plv_mean_degree"] = float(np.mean(node_strength_plv))
            
            # Network density (proportion of strong connections)
            threshold = 0.3  # Common threshold for significant connectivity
            record[f"graph_{band}_wpli_density"] = float(np.mean(wpli_values > threshold))
            record[f"graph_{band}_plv_density"] = float(np.mean(plv_values > threshold))
            
            # === Per-window connectivity (coarse bins) ===
            def _compute_window_conn(win_mask):
                """Helper to compute wPLI and PLV for a time window."""
                phases_win = precomputed.band_data[band].phase[ep_idx, :, win_mask]
                phase_diff_win = phases_win[:, None, :] - phases_win[None, :, :]
                plv_win = np.abs(np.mean(np.exp(1j * phase_diff_win), axis=-1))
                plv_win_values = plv_win[triu_idx]
                
                analytic_win = analytic[:, win_mask]
                cross_win = analytic_win[:, None, :] * np.conj(analytic_win[None, :, :])
                imag_cross_win = np.imag(cross_win)
                denom_win = np.mean(np.abs(imag_cross_win), axis=-1)
                numer_win = np.abs(np.mean(imag_cross_win, axis=-1))
                with np.errstate(divide="ignore", invalid="ignore"):
                    wpli_win = np.where(denom_win > 0, numer_win / denom_win, 0.0)
                wpli_win = 0.5 * (wpli_win + wpli_win.T)
                np.fill_diagonal(wpli_win, 0.0)
                wpli_win_values = wpli_win[triu_idx]
                return np.mean(plv_win_values), np.mean(wpli_win_values)
            
            for win_mask, win_label in zip(
                precomputed.windows.coarse_masks, precomputed.windows.coarse_labels
            ):
                if np.any(win_mask):
                    plv_mean, wpli_mean = _compute_window_conn(win_mask)
                    record[f"conn_plv_{band}_{win_label}_mean"] = float(plv_mean)
                    record[f"conn_wpli_{band}_{win_label}_mean"] = float(wpli_mean)
            
            # === Per-window connectivity (fine bins for HRF) ===
            for win_mask, win_label in zip(
                precomputed.windows.fine_masks, precomputed.windows.fine_labels
            ):
                if np.any(win_mask):
                    plv_mean, wpli_mean = _compute_window_conn(win_mask)
                    record[f"conn_plv_{band}_{win_label}_mean"] = float(plv_mean)
                    record[f"conn_wpli_{band}_{win_label}_mean"] = float(wpli_mean)
            
            # === Temporal dynamics ===
            if len(precomputed.windows.coarse_masks) >= 2:
                # Early-late connectivity difference
                early_mask = precomputed.windows.coarse_masks[0]
                late_mask = precomputed.windows.coarse_masks[-1]
                if np.any(early_mask) and np.any(late_mask):
                    _, wpli_early = _compute_window_conn(early_mask)
                    _, wpli_late = _compute_window_conn(late_mask)
                    record[f"conn_wpli_{band}_early_late_diff"] = float(wpli_late - wpli_early)
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


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
        roi_definitions = config.get("time_frequency_analysis", {}).get("rois", {})
    
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
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_temporal_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract comprehensive time-domain features from precomputed data.
    
    Features are computed per frequency band using precomputed band-filtered data.
    
    Features include:
    - Statistical moments (var, std, skew, kurtosis) per band
    - Amplitude features (RMS, peak-to-peak, MAD) per band
    - Waveform features (zero-crossings, line length, nonlinear energy) per band
    - Per-window temporal features per band
    """
    if precomputed.windows is None or not precomputed.band_data:
        return pd.DataFrame(), []
    
    from scipy import stats as scipy_stats
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    sfreq = precomputed.sfreq
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            # Use precomputed band-filtered data
            band_filtered = precomputed.band_data[band].filtered[ep_idx]  # (channels, times)
            
            # Collect for global stats per band
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
            record[f"temp_var_{band}_global_full_mean"] = float(np.mean(all_var))
            record[f"temp_var_{band}_global_full_std"] = float(np.std(all_var))
            record[f"temp_rms_{band}_global_full_mean"] = float(np.mean(all_rms))
            record[f"temp_rms_{band}_global_full_std"] = float(np.std(all_rms))
            record[f"temp_skew_{band}_global_full_mean"] = float(np.mean(all_skew))
            record[f"temp_kurt_{band}_global_full_mean"] = float(np.mean(all_kurt))
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_complexity_from_data(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract comprehensive complexity features per frequency band.
    
    Features are computed per frequency band using precomputed band-filtered data.
    
    Features include:
    - Permutation entropy per band
    - Hjorth parameters (activity, mobility, complexity) per band
    - Lempel-Ziv complexity per band
    - Per-window complexity for temporal dynamics per band
    """
    if precomputed.windows is None or not precomputed.band_data:
        return pd.DataFrame(), []
    
    # Import from complexity module to avoid code duplication
    from eeg_pipeline.analysis.features.complexity import (
        _permutation_entropy,
        _hjorth_parameters,
        _lempel_ziv_complexity,
    )
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            # Use precomputed band-filtered data
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
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_aperiodic_from_precomputed(
    precomputed: PrecomputedData,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract aperiodic (1/f) features from precomputed PSD."""
    if precomputed.psd_data is None or precomputed.windows is None:
        return pd.DataFrame(), []
    
    config = precomputed.config
    freqs = precomputed.psd_data.freqs
    psd = precomputed.psd_data.psd  # (epochs, channels, freqs)
    
    # Fit range
    fmin = float(config.get("feature_engineering.constants.aperiodic_fmin", 2.0))
    fmax = float(config.get("feature_engineering.constants.aperiodic_fmax", 40.0))
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    
    if not np.any(freq_mask):
        return pd.DataFrame(), []
    
    fit_freqs = freqs[freq_mask]
    log_freqs = np.log10(fit_freqs)
    epsilon = float(config.get("feature_engineering.constants.epsilon_psd", 1e-20))
    
    records: List[Dict[str, float]] = []
    n_epochs = psd.shape[0]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        slopes = []
        offsets = []
        
        for ch_idx, ch_name in enumerate(precomputed.ch_names):
            fit_psd = psd[ep_idx, ch_idx, freq_mask]
            log_psd = np.log10(np.maximum(fit_psd, epsilon))
            
            # Simple linear fit (log-log space = 1/f slope)
            try:
                slope, offset = np.polyfit(log_freqs, log_psd, 1)
                record[f"aperiodic_slope_{ch_name}"] = float(slope)
                record[f"aperiodic_offset_{ch_name}"] = float(offset)
                slopes.append(slope)
                offsets.append(offset)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                record[f"aperiodic_slope_{ch_name}"] = np.nan
                record[f"aperiodic_offset_{ch_name}"] = np.nan
        
        # Global summaries
        record["aperiodic_slope_global"] = float(np.mean(slopes)) if slopes else np.nan
        record["aperiodic_offset_global"] = float(np.mean(offsets)) if offsets else np.nan
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_ratios_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract band power ratios from precomputed data."""
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []
    
    config = precomputed.config
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    
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
                band_powers[band] = np.mean(power[:, precomputed.windows.active_mask], axis=1)
        
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
            num_mean = np.mean(num_power)
            denom_mean = np.mean(denom_power)
            if denom_mean > epsilon:
                record[f"ratio_{num_band}_{denom_band}_global"] = float(num_mean / denom_mean)
            else:
                record[f"ratio_{num_band}_{denom_band}_global"] = np.nan
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


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
        roi_definitions = config.get("time_frequency_analysis", {}).get("rois", {})
    
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
    
    n_epochs = precomputed.data.shape[0]
    if n_epochs < 5:
        return pd.DataFrame(), []
    
    windows = precomputed.windows
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    records: List[Dict[str, float]] = []
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            phase = precomputed.band_data[band].phase  # (epochs, channels, times)
            
            # Leave-one-out mask
            mask = np.ones(n_epochs, dtype=bool)
            mask[ep_idx] = False
            
            if mask.sum() < 2:
                continue
            
            # Full active period ITPC
            phase_active = phase[:, :, windows.active_mask]
            other_phases = phase_active[mask]
            unit_vectors = np.exp(1j * other_phases)
            itpc_full = np.abs(np.mean(unit_vectors, axis=0))  # (channels, times)
            itpc_ch_full = np.mean(itpc_full, axis=1)  # (channels,)
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                record[f"phase_itpc_{band}_{ch_name}_full_mean"] = float(itpc_ch_full[ch_idx])
            
            record[f"phase_itpc_{band}_global_full_mean"] = float(np.mean(itpc_ch_full))
            
            # Coarse temporal bins
            coarse_itpc = {}
            for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                if np.any(win_mask):
                    phase_win = phase[:, :, win_mask]
                    other_phases_win = phase_win[mask]
                    unit_vectors_win = np.exp(1j * other_phases_win)
                    itpc_win = np.abs(np.mean(unit_vectors_win, axis=0))
                    itpc_ch_win = np.mean(itpc_win, axis=1)
                    
                    for ch_idx, ch_name in enumerate(precomputed.ch_names):
                        record[f"phase_itpc_{band}_{ch_name}_{win_label}_mean"] = float(itpc_ch_win[ch_idx])
                    
                    coarse_itpc[win_label] = np.mean(itpc_ch_win)
                    record[f"phase_itpc_{band}_global_{win_label}_mean"] = float(coarse_itpc[win_label])
            
            # Fine temporal bins
            for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                if np.any(win_mask):
                    phase_win = phase[:, :, win_mask]
                    other_phases_win = phase_win[mask]
                    unit_vectors_win = np.exp(1j * other_phases_win)
                    itpc_win = np.abs(np.mean(unit_vectors_win, axis=0))
                    itpc_ch_win = np.mean(itpc_win, axis=1)
                    
                    record[f"phase_itpc_{band}_global_{win_label}_mean"] = float(np.mean(itpc_ch_win))
            
            # Temporal dynamics
            if "early" in coarse_itpc and "late" in coarse_itpc:
                diff = coarse_itpc["late"] - coarse_itpc["early"]
                record[f"phase_itpc_{band}_global_early_late_diff"] = float(diff)
            
            # Peak ITPC time (from full time course)
            if len(active_times) > 0:
                itpc_time_course = np.mean(itpc_full, axis=0)  # (times,)
                peak_idx = np.argmax(itpc_time_course)
                record[f"phase_itpc_{band}_peak_latency"] = float(active_times[peak_idx])
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def extract_precomputed_features(
    epochs: mne.Epochs,
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
    epochs : mne.Epochs
        Input epochs
    bands : List[str]
        Frequency bands (e.g., ["delta", "theta", "alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance
    feature_groups : Optional[List[str]]
        Which feature groups to extract. Default: all.
        Options: "erds", "spectral", "gfp", "connectivity", "roi", "temporal", 
                 "complexity", "aperiodic", "ratios", "asymmetry", "itpc"
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
    all_groups = [
        "power", "erds", "spectral", "gfp", "connectivity", "roi", 
        "temporal", "complexity", "aperiodic", "ratios", "asymmetry", "itpc"
    ]
    if feature_groups is None:
        feature_groups = all_groups
    
    logger.info(f"Extracting feature groups: {feature_groups}")
    
    # === Get condition labels from events (if provided) ===
    condition: Optional[np.ndarray] = None
    if events_df is not None and not events_df.empty:
        from eeg_pipeline.utils.io.general import get_pain_column_from_config
        
        pain_col = get_pain_column_from_config(config, events_df)
        if pain_col is not None and pain_col in events_df.columns:
            pain_values = pd.to_numeric(events_df[pain_col], errors="coerce")
            pain_mask = (pain_values > 0).values
            
            n_epochs = len(epochs)
            if len(pain_mask) != n_epochs:
                logger.warning(
                    f"Events length ({len(pain_mask)}) doesn't match epochs ({n_epochs}); "
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
        "power", "erds", "connectivity", "roi", "gfp", "ratios", "asymmetry", "itpc"
    ])
    needs_psd = any(g in feature_groups for g in ["spectral", "aperiodic"])
    
    # Precompute all expensive intermediates ONCE
    logger.info("Precomputing intermediates (filtering, PSD, GFP)...")
    precomputed = precompute_data(
        epochs, bands, config, logger,
        compute_bands=needs_bands,
        compute_psd_data=needs_psd,
        n_plateau_windows=n_plateau_windows,
    )
    
    result = ExtractionResult(precomputed=precomputed, condition=condition)
    
    # Extract each feature group
    if "power" in feature_groups:
        logger.info("Extracting time-resolved power features...")
        df, cols = _extract_power_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["power"] = FeatureSet(df, cols, "power")
    
    if "erds" in feature_groups:
        logger.info("Extracting ERD/ERS features...")
        df, cols = _extract_erds_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["erds"] = FeatureSet(df, cols, "erds")
    
    if "spectral" in feature_groups:
        logger.info("Extracting spectral features...")
        df, cols = _extract_spectral_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["spectral"] = FeatureSet(df, cols, "spectral")
    
    if "gfp" in feature_groups:
        logger.info("Extracting GFP features...")
        df, cols = _extract_gfp_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["gfp"] = FeatureSet(df, cols, "gfp")
    
    if "connectivity" in feature_groups:
        logger.info("Extracting connectivity features...")
        df, cols = _extract_connectivity_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["connectivity"] = FeatureSet(df, cols, "connectivity")
    
    if "roi" in feature_groups:
        logger.info("Extracting ROI features...")
        df, cols = _extract_roi_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["roi"] = FeatureSet(df, cols, "roi")
    
    if "temporal" in feature_groups:
        logger.info("Extracting temporal features...")
        df, cols = _extract_temporal_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["temporal"] = FeatureSet(df, cols, "temporal")
    
    if "complexity" in feature_groups:
        logger.info("Extracting complexity features...")
        df, cols = _extract_complexity_from_data(precomputed, bands)
        if not df.empty:
            result.features["complexity"] = FeatureSet(df, cols, "complexity")
    
    if "aperiodic" in feature_groups:
        logger.info("Extracting aperiodic (1/f) features...")
        df, cols = _extract_aperiodic_from_precomputed(precomputed)
        if not df.empty:
            result.features["aperiodic"] = FeatureSet(df, cols, "aperiodic")
    
    if "ratios" in feature_groups:
        logger.info("Extracting band power ratio features...")
        df, cols = _extract_ratios_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["ratios"] = FeatureSet(df, cols, "ratios")
    
    if "asymmetry" in feature_groups:
        logger.info("Extracting hemispheric asymmetry features...")
        df, cols = _extract_asymmetry_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["asymmetry"] = FeatureSet(df, cols, "asymmetry")
    
    if "itpc" in feature_groups:
        logger.info("Extracting ITPC features...")
        df, cols = _extract_itpc_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["itpc"] = FeatureSet(df, cols, "itpc")
    
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
            "temporal", "complexity", "aperiodic", "ratios", "asymmetry", "itpc"
        ],
        
        # Connectivity-focused
        "connectivity": ["connectivity", "roi", "gfp"],
        
        # Pain-specific (emphasizes sensorimotor, asymmetry)
        "pain": ["erds", "roi", "asymmetry", "aperiodic", "spectral"],
        
        # Time-resolved (windowed features for HRF modeling)
        "temporal": ["erds", "roi", "gfp", "temporal"],
    }
