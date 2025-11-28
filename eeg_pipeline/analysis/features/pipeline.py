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
    - Mean ERD/ERS over active period
    - ERD/ERS per temporal window
    - Temporal dynamics: slope, onset latency, peak latency
    - Statistics: std, min, max, range across time
    - Percentiles: 10th, 25th, 50th, 75th, 90th
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []
    
    epsilon = EPSILON_STD
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    times = precomputed.times
    active_times = times[precomputed.windows.active_mask]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            
            # Collect all channel ERD/ERS for global stats
            all_erds_mean = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power = np.mean(power[ch_idx, precomputed.windows.baseline_mask])
                baseline_std = np.std(power[ch_idx, precomputed.windows.baseline_mask])
                active_power_trace = power[ch_idx, precomputed.windows.active_mask]
                active_power_mean = np.mean(active_power_trace)
                
                if baseline_power > epsilon:
                    # Mean ERD/ERS
                    erds = ((active_power_mean - baseline_power) / baseline_power) * 100
                    # Full ERD/ERS time course
                    erds_trace = ((active_power_trace - baseline_power) / baseline_power) * 100
                else:
                    erds = np.nan
                    erds_trace = np.full_like(active_power_trace, np.nan)
                
                record[f"erds_{band}_{ch_name}"] = float(erds)
                all_erds_mean.append(erds)
                
                # === Temporal statistics ===
                if np.any(np.isfinite(erds_trace)):
                    record[f"erds_{band}_{ch_name}_std"] = float(np.nanstd(erds_trace))
                    record[f"erds_{band}_{ch_name}_min"] = float(np.nanmin(erds_trace))
                    record[f"erds_{band}_{ch_name}_max"] = float(np.nanmax(erds_trace))
                    record[f"erds_{band}_{ch_name}_range"] = float(np.nanmax(erds_trace) - np.nanmin(erds_trace))
                    
                    # Percentiles
                    for pct in [10, 25, 50, 75, 90]:
                        record[f"erds_{band}_{ch_name}_p{pct}"] = float(np.nanpercentile(erds_trace, pct))
                    
                    # Temporal dynamics
                    if len(active_times) > 1 and len(erds_trace) > 1:
                        # Slope (linear trend)
                        valid_mask = np.isfinite(erds_trace)
                        if np.sum(valid_mask) > 2:
                            slope, _ = np.polyfit(active_times[valid_mask], erds_trace[valid_mask], 1)
                            record[f"erds_{band}_{ch_name}_slope"] = float(slope)
                        else:
                            record[f"erds_{band}_{ch_name}_slope"] = np.nan
                        
                        # Peak latency (time of max absolute ERD/ERS)
                        peak_idx = np.nanargmax(np.abs(erds_trace))
                        record[f"erds_{band}_{ch_name}_peak_latency"] = float(active_times[peak_idx])
                        
                        # Onset latency (first time ERD/ERS exceeds 1 std from baseline)
                        threshold = baseline_std / baseline_power * 100 if baseline_power > epsilon else np.inf
                        onset_mask = np.abs(erds_trace) > threshold
                        if np.any(onset_mask):
                            onset_idx = np.argmax(onset_mask)
                            record[f"erds_{band}_{ch_name}_onset_latency"] = float(active_times[onset_idx])
                        else:
                            record[f"erds_{band}_{ch_name}_onset_latency"] = np.nan
                else:
                    for suffix in ["_std", "_min", "_max", "_range", "_slope", "_peak_latency", "_onset_latency"]:
                        record[f"erds_{band}_{ch_name}{suffix}"] = np.nan
                    for pct in [10, 25, 50, 75, 90]:
                        record[f"erds_{band}_{ch_name}_p{pct}"] = np.nan
                
                # === Temporal windows ===
                for win_idx, (win_mask, win_label) in enumerate(
                    zip(precomputed.windows.plateau_masks, precomputed.windows.window_labels)
                ):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        win_power_std = np.std(power[ch_idx, win_mask])
                        if baseline_power > epsilon:
                            erds_win = ((win_power - baseline_power) / baseline_power) * 100
                            erds_win_std = (win_power_std / baseline_power) * 100
                        else:
                            erds_win = np.nan
                            erds_win_std = np.nan
                        record[f"erds_{band}_{ch_name}_{win_label}"] = float(erds_win)
                        record[f"erds_{band}_{ch_name}_{win_label}_std"] = float(erds_win_std)
            
            # === Global (cross-channel) statistics per band ===
            valid_erds = [e for e in all_erds_mean if np.isfinite(e)]
            if valid_erds:
                record[f"erds_{band}_global_mean"] = float(np.mean(valid_erds))
                record[f"erds_{band}_global_std"] = float(np.std(valid_erds))
                record[f"erds_{band}_global_min"] = float(np.min(valid_erds))
                record[f"erds_{band}_global_max"] = float(np.max(valid_erds))
                record[f"erds_{band}_global_range"] = float(np.max(valid_erds) - np.min(valid_erds))
        
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
    total_power = np.sum(psd, axis=2)  # (epochs, channels)
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
                band_power_mean = np.mean(band_psd)
                band_power_sum = np.sum(band_psd)
                
                band_powers_per_ch[ch_name][band] = band_power_mean
                all_band_power.append(band_power_mean)
                
                # === Absolute power statistics ===
                record[f"pow_{band}_{ch_name}"] = float(band_power_mean)
                record[f"pow_{band}_{ch_name}_sum"] = float(band_power_sum)
                record[f"pow_{band}_{ch_name}_std"] = float(np.std(band_psd))
                record[f"pow_{band}_{ch_name}_median"] = float(np.median(band_psd))
                record[f"pow_{band}_{ch_name}_max"] = float(np.max(band_psd))
                
                # Log power (often more normally distributed)
                record[f"logpow_{band}_{ch_name}"] = float(np.log10(band_power_mean + epsilon))
                
                # === Relative power ===
                if total_power[ep_idx, ch_idx] > epsilon:
                    rel_power = band_power_mean / total_power[ep_idx, ch_idx]
                else:
                    rel_power = np.nan
                record[f"relpow_{band}_{ch_name}"] = float(rel_power)
                
                # === Peak frequency and power ===
                peak_idx = np.argmax(band_psd)
                record[f"peakfreq_{band}_{ch_name}"] = float(band_freqs[peak_idx])
                record[f"peakpow_{band}_{ch_name}"] = float(band_psd[peak_idx])
                
                # Peak prominence (peak / mean)
                if band_power_mean > epsilon:
                    record[f"peakprom_{band}_{ch_name}"] = float(band_psd[peak_idx] / band_power_mean)
                else:
                    record[f"peakprom_{band}_{ch_name}"] = np.nan
                
                # === Spectral entropy (within band) ===
                psd_norm = band_psd / (band_power_sum + epsilon)
                psd_norm = psd_norm[psd_norm > 0]
                if psd_norm.size > 0:
                    se = -np.sum(psd_norm * np.log2(psd_norm + epsilon))
                    se_norm = se / np.log2(len(band_psd)) if len(band_psd) > 1 else se
                else:
                    se_norm = np.nan
                record[f"se_{band}_{ch_name}"] = float(se_norm)
                
                # === Spectral edge frequencies ===
                cumsum = np.cumsum(band_psd)
                total_band = cumsum[-1] if len(cumsum) > 0 else 0
                if total_band > 0:
                    for edge_pct in [50, 75, 90, 95]:
                        threshold = total_band * (edge_pct / 100.0)
                        edge_idx = np.searchsorted(cumsum, threshold)
                        edge_idx = min(edge_idx, len(band_freqs) - 1)
                        record[f"sef{edge_pct}_{band}_{ch_name}"] = float(band_freqs[edge_idx])
                else:
                    for edge_pct in [50, 75, 90, 95]:
                        record[f"sef{edge_pct}_{band}_{ch_name}"] = np.nan
                
                # === Spectral slope within band ===
                if len(band_freqs) > 2:
                    log_freqs = np.log10(band_freqs + epsilon)
                    log_psd = np.log10(band_psd + epsilon)
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
                psd_sum = np.sum(alpha_psd)
                if psd_sum > 0:
                    # Center of gravity (CoG) IAF
                    iaf_cog = np.sum(alpha_freqs * alpha_psd) / psd_sum
                    # Peak IAF
                    iaf_peak = alpha_freqs[np.argmax(alpha_psd)]
                    # Alpha power at IAF
                    iaf_power = np.max(alpha_psd)
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
        
        # Per-window GFP
        for win_idx, (win_mask, win_label) in enumerate(
            zip(precomputed.windows.plateau_masks, precomputed.windows.window_labels)
        ):
            if np.any(win_mask):
                gfp_win = gfp[win_mask]
                record[f"gfp_{win_label}_mean"] = float(np.mean(gfp_win))
                record[f"gfp_{win_label}_std"] = float(np.std(gfp_win))
                record[f"gfp_{win_label}_max"] = float(np.max(gfp_win))
        
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
            
            # Per-window band GFP
            for win_idx, (win_mask, win_label) in enumerate(
                zip(precomputed.windows.plateau_masks, precomputed.windows.window_labels)
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
            
            # === Per-window connectivity ===
            for win_idx, (win_mask, win_label) in enumerate(
                zip(precomputed.windows.plateau_masks, precomputed.windows.window_labels)
            ):
                if np.any(win_mask):
                    phases_win = precomputed.band_data[band].phase[ep_idx, :, win_mask]
                    phase_diff_win = phases_win[:, None, :] - phases_win[None, :, :]
                    plv_win = np.abs(np.mean(np.exp(1j * phase_diff_win), axis=-1))
                    plv_win_values = plv_win[triu_idx]
                    
                    record[f"plv_{band}_{win_label}_mean"] = float(np.mean(plv_win_values))
                    # Compute window-specific wPLI
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
                    record[f"wpli_{band}_{win_label}_mean"] = float(np.mean(wpli_win_values))
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_roi_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract ROI-averaged features from precomputed data."""
    from eeg_pipeline.analysis.features.core import build_roi_map
    
    config = precomputed.config
    roi_definitions = config.get("rois", {})
    if not roi_definitions:
        roi_definitions = config.get("time_frequency_analysis", {}).get("rois", {})
    
    if not roi_definitions:
        return pd.DataFrame(), []
    
    # Build ROI channel map using shared utility
    roi_map = build_roi_map(precomputed.ch_names, roi_definitions)
    
    if not roi_map:
        return pd.DataFrame(), []
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    freq_bands = get_frequency_bands(config)
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            baseline_power = np.mean(power[:, precomputed.windows.baseline_mask], axis=1)
            active_power = np.mean(power[:, precomputed.windows.active_mask], axis=1)
            
            for roi_name, ch_indices in roi_map.items():
                roi_baseline = np.mean(baseline_power[ch_indices])
                roi_active = np.mean(active_power[ch_indices])
                
                record[f"roi_pow_{band}_{roi_name}"] = float(roi_active)
                
                if roi_baseline > epsilon:
                    roi_erds = ((roi_active - roi_baseline) / roi_baseline) * 100
                else:
                    roi_erds = np.nan
                record[f"roi_erds_{band}_{roi_name}"] = float(roi_erds)
        
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
                
                # === Per-window features ===
                for win_mask, win_label in zip(
                    precomputed.windows.plateau_masks, precomputed.windows.window_labels
                ):
                    if np.any(win_mask):
                        ch_win = band_filtered[ch_idx, win_mask]
                        record[f"var_{band}_{ch_name}_{win_label}"] = float(np.var(ch_win))
                        record[f"rms_{band}_{ch_name}_{win_label}"] = float(np.sqrt(np.mean(ch_win ** 2)))
            
            # === Global statistics per band ===
            record[f"var_{band}_global_mean"] = float(np.mean(all_var))
            record[f"var_{band}_global_std"] = float(np.std(all_var))
            record[f"rms_{band}_global_mean"] = float(np.mean(all_rms))
            record[f"rms_{band}_global_std"] = float(np.std(all_rms))
            record[f"skew_{band}_global_mean"] = float(np.mean(all_skew))
            record[f"kurt_{band}_global_mean"] = float(np.mean(all_kurt))
        
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
                record[f"lzc_{band}_{ch_name}"] = float(lz) if np.isfinite(lz) else np.nan
                if np.isfinite(lz):
                    lz_vals.append(lz)
                
                # === Per-window complexity ===
                for win_mask, win_label in zip(
                    precomputed.windows.plateau_masks, precomputed.windows.window_labels
                ):
                    if np.any(win_mask):
                        ch_win = band_filtered[ch_idx, win_mask]
                        pe_win = _permutation_entropy(ch_win, order=3, delay=1, normalize=True)
                        record[f"pe_{band}_{ch_name}_{win_label}"] = float(pe_win) if np.isfinite(pe_win) else np.nan
            
            # === Global summaries per band ===
            record[f"pe_{band}_global_mean"] = float(np.mean(pe_vals)) if pe_vals else np.nan
            record[f"pe_{band}_global_std"] = float(np.std(pe_vals)) if len(pe_vals) > 1 else np.nan
            record[f"hjorth_activity_{band}_global"] = float(np.mean(act_vals)) if act_vals else np.nan
            record[f"hjorth_mobility_{band}_global"] = float(np.mean(mob_vals)) if mob_vals else np.nan
            record[f"hjorth_complexity_{band}_global"] = float(np.mean(comp_vals)) if comp_vals else np.nan
            record[f"lzc_{band}_global_mean"] = float(np.mean(lz_vals)) if lz_vals else np.nan
            record[f"lzc_{band}_global_std"] = float(np.std(lz_vals)) if len(lz_vals) > 1 else np.nan
        
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
    
    # Find left-right pairs
    asymmetry_pairs = config.get("feature_engineering.roi_features.asymmetry_pairs", [
        {"left": "Sensorimotor_Ipsi_L", "right": "Sensorimotor_Contra_R", "name": "sensorimotor"},
        {"left": "Temporal_Ipsi_L", "right": "Temporal_Contra_R", "name": "temporal"},
        {"left": "ParOccipital_Ipsi_L", "right": "ParOccipital_Contra_R", "name": "paroccipital"},
    ])
    
    # Build ROI channel map using shared utility
    roi_map = build_roi_map(precomputed.ch_names, roi_definitions)
    
    records: List[Dict[str, float]] = []
    n_epochs = precomputed.data.shape[0]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            active_power = np.mean(power[:, precomputed.windows.active_mask], axis=1)
            
            for pair in asymmetry_pairs:
                left_roi = pair.get("left", "")
                right_roi = pair.get("right", "")
                pair_name = pair.get("name", f"{left_roi}_{right_roi}")
                
                left_indices = roi_map.get(left_roi, [])
                right_indices = roi_map.get(right_roi, [])
                
                if not left_indices or not right_indices:
                    record[f"asym_{band}_{pair_name}"] = np.nan
                    continue
                
                left_power = np.mean(active_power[left_indices])
                right_power = np.mean(active_power[right_indices])
                
                denom = left_power + right_power
                if denom > epsilon:
                    asymmetry = (right_power - left_power) / denom
                else:
                    asymmetry = np.nan
                
                record[f"asym_{band}_{pair_name}"] = float(asymmetry)
        
        records.append(record)
    
    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns


def _extract_itpc_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract Inter-Trial Phase Coherence from precomputed phase data."""
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []
    
    n_epochs = precomputed.data.shape[0]
    if n_epochs < 5:
        # Need multiple trials for ITPC
        return pd.DataFrame(), []
    
    records: List[Dict[str, float]] = []
    
    # For trial-level features, use leave-one-out ITPC
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            phase = precomputed.band_data[band].phase  # (epochs, channels, times)
            phase_active = phase[:, :, precomputed.windows.active_mask]
            
            # Leave-one-out: mean of all other trials
            mask = np.ones(n_epochs, dtype=bool)
            mask[ep_idx] = False
            other_phases = phase_active[mask]
            
            if other_phases.shape[0] < 2:
                for ch_idx, ch_name in enumerate(precomputed.ch_names):
                    record[f"itpc_{band}_{ch_name}"] = np.nan
                record[f"itpc_{band}_global"] = np.nan
                continue
            
            unit_vectors = np.exp(1j * other_phases)
            itpc_loo = np.abs(np.mean(unit_vectors, axis=0))  # (channels, times)
            itpc_ch = np.mean(itpc_loo, axis=1)  # (channels,)
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                record[f"itpc_{band}_{ch_name}"] = float(itpc_ch[ch_idx])
            
            record[f"itpc_{band}_global"] = float(np.mean(itpc_ch))
        
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
        "erds", "spectral", "gfp", "connectivity", "roi", 
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
        "erds", "connectivity", "roi", "gfp", "ratios", "asymmetry", "itpc"
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

