"""
Coupling Statistics
===================

Inter-band coupling and group power statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .band_stats import compute_band_spatial_correlation


def compute_consensus_labels(
    labels_all_trials: List[np.ndarray],
    n_timepoints: int,
) -> np.ndarray:
    """Compute consensus microstate labels across trials."""
    labels = np.zeros(n_timepoints, dtype=int)
    for t in range(n_timepoints):
        counts = np.bincount([labels_all_trials[trial][t] for trial in range(len(labels_all_trials))])
        labels[t] = np.argmax(counts)
    return labels


def compute_inter_band_coupling_matrix(
    tfr_avg,
    band_names: List[str],
    features_freq_bands: Dict[str, Tuple[float, float]],
    extract_band_channel_means_func,
) -> np.ndarray:
    """Compute inter-band coupling matrix."""
    n = len(band_names)
    mat = np.zeros((n, n))
    
    for i, b1 in enumerate(band_names):
        fmin1, fmax1 = features_freq_bands[b1]
        fm1 = (tfr_avg.freqs >= fmin1) & (tfr_avg.freqs <= fmax1)
        if not fm1.any():
            continue
        mat[i, i] = 1.0
        b1_ch = extract_band_channel_means_func(tfr_avg, fm1)
        
        for j in range(i + 1, n):
            b2 = band_names[j]
            fmin2, fmax2 = features_freq_bands[b2]
            fm2 = (tfr_avg.freqs >= fmin2) & (tfr_avg.freqs <= fmax2)
            if not fm2.any():
                continue
            b2_ch = extract_band_channel_means_func(tfr_avg, fm2)
            r = compute_band_spatial_correlation(b1_ch, b2_ch)
            mat[i, j] = mat[j, i] = r
    
    return mat


def compute_group_channel_power_statistics(
    subj_pow: Dict[str, pd.DataFrame],
    bands: List[str],
    all_channels: List[str],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Compute group channel power statistics."""
    heatmap_rows, stats_rows = [], []
    
    for band in bands:
        band_str = str(band)
        subj_means = []
        for _, df in subj_pow.items():
            vals = []
            for ch in all_channels:
                col = f"pow_{band_str}_{ch}"
                if col in df.columns:
                    vals.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
                else:
                    vals.append(np.nan)
            subj_means.append(vals)
        
        arr = np.asarray(subj_means, dtype=float)
        mean_across = np.nanmean(arr, axis=0)
        heatmap_rows.append(mean_across)
        
        n_eff = np.sum(np.isfinite(arr), axis=0)
        std_across = np.nanstd(arr, axis=0, ddof=1)
        
        for j, ch in enumerate(all_channels):
            stats_rows.append({
                "band": band_str, "channel": ch,
                "mean": float(mean_across[j]) if np.isfinite(mean_across[j]) else np.nan,
                "std": float(std_across[j]) if np.isfinite(std_across[j]) else np.nan,
                "n_subjects": int(n_eff[j]),
            })
    
    return heatmap_rows, stats_rows






