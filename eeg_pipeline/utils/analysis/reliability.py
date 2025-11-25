from __future__ import annotations

import hashlib
import logging
from typing import Tuple

import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


from .tfr import compute_adaptive_n_cycles, get_tfr_config, compute_tfr_morlet


###################################################################
# Reliability and Seeding Utilities
###################################################################

def get_subject_seed(base_seed: int, subject: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def compute_feature_split_half_reliability(
    feature_matrix: np.ndarray,
    ratings: np.ndarray,
    n_boot: int,
    use_spearman: bool,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Split-half reliability across feature-level correlations."""
    if feature_matrix.size == 0 or ratings.size == 0:
        return np.nan, np.nan, np.nan
    n_trials = feature_matrix.shape[0]
    if n_trials < 4:
        return np.nan, np.nan, np.nan

    rel_values = []
    for _ in range(int(n_boot)):
        idx = rng.permutation(n_trials)
        half = n_trials // 2
        idx_a, idx_b = idx[:half], idx[half:]
        Xa, Xb = feature_matrix[idx_a], feature_matrix[idx_b]
        ya, yb = ratings[idx_a], ratings[idx_b]

        ra = []
        rb = []
        for col in range(feature_matrix.shape[1]):
            fa = Xa[:, col]
            fb = Xb[:, col]
            if use_spearman:
                r_a, _ = spearmanr(fa, ya)
                r_b, _ = spearmanr(fb, yb)
            else:
                r_a, _ = pearsonr(fa, ya)
                r_b, _ = pearsonr(fb, yb)
            ra.append(r_a if np.isfinite(r_a) else np.nan)
            rb.append(r_b if np.isfinite(r_b) else np.nan)

        ra = np.asarray(ra, dtype=float)
        rb = np.asarray(rb, dtype=float)
        mask = np.isfinite(ra) & np.isfinite(rb)
        if not np.any(mask):
            continue
        r_half, _ = spearmanr(ra[mask], rb[mask]) if use_spearman else pearsonr(ra[mask], rb[mask])
        if np.isfinite(r_half):
            rel_values.append((2 * r_half) / (1 + r_half))  # Spearman-Brown

    if not rel_values:
        return np.nan, np.nan, np.nan
    rel_values = np.asarray(rel_values, dtype=float)
    return (
        float(np.nanmedian(rel_values)),
        float(np.nanpercentile(rel_values, 2.5)),
        float(np.nanpercentile(rel_values, 97.5)),
    )


def compute_tf_split_half_reliability(
    epochs: mne.Epochs,
    aligned_events: pd.DataFrame,
    y: pd.Series,
    config,
    use_spearman: bool,
    n_boot: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Tuple[float, float, float]:
    """Split-half reliability for time-frequency correlation map using a single TFR computation."""
    from eeg_pipeline.analysis.behavior.temporal import _compute_tf_correlations_for_bins
    from eeg_pipeline.utils.analysis.tfr import restrict_epochs_to_roi, apply_baseline_to_tfr

    heatmap_config = config.get("behavior_analysis", {}).get("time_frequency_heatmap", {})
    roi_selection = heatmap_config.get("roi_selection")

    epochs_for_tfr = restrict_epochs_to_roi(epochs, roi_selection, config, logger)

    tfr = compute_tfr_morlet(epochs_for_tfr, config, logger=logger)
    if tfr is None:
        return np.nan, np.nan, np.nan

    apply_baseline_to_tfr(tfr, config, logger)

    power = tfr.data.mean(axis=1)  # (epochs, freqs, times)
    times = tfr.times
    freq_vec = tfr.freqs
    y_array = y.to_numpy()

    time_window = heatmap_config.get("time_window")
    if time_window is not None:
        t_start, t_end = float(time_window[0]), float(time_window[1])
        time_mask = (times >= t_start) & (times <= t_end)
        if not np.any(time_mask):
            return np.nan, np.nan, np.nan
        times = times[time_mask]
        power = power[:, :, time_mask]

    freq_range = heatmap_config.get("freq_range")
    if freq_range is not None:
        f_min, f_max = float(freq_range[0]), float(freq_range[1])
        freq_mask = (freq_vec >= f_min) & (freq_vec <= f_max)
        if not np.any(freq_mask):
            return np.nan, np.nan, np.nan
        freq_vec = freq_vec[freq_mask]
        power = power[:, freq_mask, :]

    time_bin_width = float(heatmap_config.get("time_resolution"))
    time_bin_edges = np.arange(times[0], times[-1] + time_bin_width, time_bin_width)
    stats_config = config.get("behavior_analysis", {}).get("statistics", {})
    min_valid_points = int(stats_config.get("min_samples_roi"))

    rel_values = []
    for _ in range(int(n_boot)):
        idx = rng.permutation(len(y_array))
        half = len(idx) // 2
        idx_a, idx_b = idx[:half], idx[half:]
        if len(idx_a) < min_valid_points or len(idx_b) < min_valid_points:
            continue
        corr_a, _, _, _, info_a = _compute_tf_correlations_for_bins(
            power[idx_a], y_array[idx_a], times, freq_vec, time_bin_edges,
            min_valid_points, use_spearman, covariates_df=None
        )
        corr_b, _, _, _, info_b = _compute_tf_correlations_for_bins(
            power[idx_b], y_array[idx_b], times, freq_vec, time_bin_edges,
            min_valid_points, use_spearman, covariates_df=None
        )
        informative = set(map(tuple, info_a)) & set(map(tuple, info_b))
        if not informative:
            continue
        vals_a = []
        vals_b = []
        for f_idx, t_idx in informative:
            vals_a.append(corr_a[f_idx, t_idx])
            vals_b.append(corr_b[f_idx, t_idx])
        vals_a = np.asarray(vals_a, dtype=float)
        vals_b = np.asarray(vals_b, dtype=float)
        mask = np.isfinite(vals_a) & np.isfinite(vals_b)
        if not np.any(mask):
            continue
        r_half, _ = spearmanr(vals_a[mask], vals_b[mask]) if use_spearman else pearsonr(vals_a[mask], vals_b[mask])
        if np.isfinite(r_half):
            rel_values.append((2 * r_half) / (1 + r_half))

    if not rel_values:
        return np.nan, np.nan, np.nan
    rel_values = np.asarray(rel_values, dtype=float)
    return (
        float(np.nanmedian(rel_values)),
        float(np.nanpercentile(rel_values, 2.5)),
        float(np.nanpercentile(rel_values, 97.5)),
    )
