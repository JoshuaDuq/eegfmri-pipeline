"""
Group Correlation Statistics
============================

Functions for computing group-level correlation statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    import mne

from .base import get_config_value, ensure_config, get_statistics_constants
from .correlation import compute_correlation, fisher_z, inverse_fisher_z


def _normalize_pair(
    x_data: np.ndarray,
    y_data: np.ndarray,
    strategy: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Normalize x, y pair according to strategy."""
    x, y = x_data.copy(), y_data.copy()
    
    if strategy == "within_subject_centered":
        return x - np.nanmean(x), y - np.nanmean(y)
    
    if strategy == "within_subject_zscored":
        x_std, y_std = np.nanstd(x, ddof=1), np.nanstd(y, ddof=1)
        if x_std <= 0 or y_std <= 0:
            return None
        return (x - np.nanmean(x)) / x_std, (y - np.nanmean(y)) / y_std
    
    return x, y


def _build_valid_pairs(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    config: Optional[Any] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build list of valid (x, y) pairs."""
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    pairs = []
    for x_arr, y_arr in zip(x_lists, y_lists):
        x, y = np.asarray(x_arr), np.asarray(y_arr)
        if len(x) != len(y):
            raise ValueError(f"Mismatched lengths: {len(x)} vs {len(y)}")
        
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) >= min_samples:
            pairs.append((x[valid], y[valid]))
    
    return pairs


def _compute_correlation_by_method(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> Tuple[float, float]:
    """Compute correlation using specified method."""
    if method.lower() == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y, nan_policy="omit")
    
    def _sf(v):
        try:
            return float(v)
        except:
            return np.nan
    
    return _sf(r), _sf(p)


def _compute_bootstrap_ci_for_pairs(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    corr_func,
    n_boot: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute bootstrap CI for group correlation."""
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_stats", 3)
    
    if n_boot <= 0 or len(pairs) < min_samples:
        return np.nan, np.nan
    
    boots = []
    pair_idx = np.arange(len(pairs))
    
    for _ in range(n_boot):
        sampled = rng.choice(pair_idx, size=len(pairs), replace=True)
        x_samples, y_samples = [], []
        
        for idx in sampled:
            x_d, y_d = pairs[idx]
            norm = _normalize_pair(x_d, y_d, strategy)
            if norm is None:
                continue
            x_samples.append(norm[0])
            y_samples.append(norm[1])
        
        if not x_samples:
            continue
        
        x_cat = np.concatenate(x_samples)
        y_cat = np.concatenate(y_samples)
        r, _ = corr_func(x_cat, y_cat)
        if np.isfinite(r):
            boots.append(r)
    
    if not boots:
        return np.nan, np.nan
    
    ci_lo = constants.get("ci_percentile_low", 2.5)
    ci_hi = constants.get("ci_percentile_high", 97.5)
    
    def _sf(v):
        try:
            return float(v)
        except:
            return np.nan
    
    return _sf(np.percentile(boots, ci_lo)), _sf(np.percentile(boots, ci_hi))


def _get_ttest_pvalue(result) -> float:
    """Extract p-value from ttest result."""
    if result is None or not hasattr(result, "pvalue"):
        return np.nan
    try:
        return float(result.pvalue)
    except:
        return np.nan


def _compute_pooled_strategy_stats(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    method: str,
    n_boot: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    """Compute stats using pooled trial strategy."""
    constants = get_statistics_constants(config)
    min_stats = constants.get("min_samples_for_stats", 3)
    clip_min = constants.get("fisher_z_clip_min", -0.9999)
    clip_max = constants.get("fisher_z_clip_max", 0.9999)
    
    valid_pairs = []
    subject_rs = []
    
    for x, y in pairs:
        norm = _normalize_pair(x, y, strategy)
        if norm is None:
            continue
        x_n, y_n = norm
        valid_pairs.append((x_n, y_n))
        
        r, _ = _compute_correlation_by_method(x_n, y_n, method)
        if np.isfinite(r):
            subject_rs.append(float(np.clip(r, clip_min, clip_max)))
    
    if not valid_pairs:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan
    
    x_pooled = np.concatenate([x for x, _ in valid_pairs])
    y_pooled = np.concatenate([y for _, y in valid_pairs])
    
    r_obs, p_obs = _compute_correlation_by_method(x_pooled, y_pooled, method)
    n_trials = len(x_pooled)
    n_subj = len(valid_pairs)
    
    # Group p-value from Fisher z-transformed subject correlations
    p_group = np.nan
    if len(subject_rs) >= min_stats:
        z_scores = np.arctanh(np.array(subject_rs))
        result = stats.ttest_1samp(z_scores, popmean=0.0, nan_policy="omit")
        p_group = _get_ttest_pvalue(result)
    
    # Bootstrap CI
    ci = (np.nan, np.nan)
    if n_boot and n_subj >= min_stats:
        ci = _compute_bootstrap_ci_for_pairs(
            valid_pairs, strategy, lambda x, y: _compute_correlation_by_method(x, y, method),
            n_boot, rng, config
        )
    
    def _sf(v):
        try:
            return float(v)
        except:
            return np.nan
    
    return _sf(r_obs), _sf(p_group), n_trials, n_subj, ci, np.nan


def _compute_fisher_strategy_stats(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    method: str,
    n_boot: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    """Compute stats using Fisher z aggregation of subject correlations."""
    constants = get_statistics_constants(config)
    min_stats = constants.get("min_samples_for_stats", 3)
    clip_min = constants.get("fisher_z_clip_min", -0.9999)
    clip_max = constants.get("fisher_z_clip_max", 0.9999)
    ci_lo = constants.get("ci_percentile_low", 2.5)
    ci_hi = constants.get("ci_percentile_high", 97.5)
    
    subject_rs = []
    for x, y in pairs:
        r, _ = _compute_correlation_by_method(x, y, method)
        if np.isfinite(r):
            subject_rs.append(float(np.clip(r, clip_min, clip_max)))
    
    if not subject_rs:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan
    
    z_scores = np.arctanh(np.array(subject_rs))
    
    def _sf(v):
        try:
            return float(v)
        except:
            return np.nan
    
    r_group = _sf(np.tanh(np.nanmean(z_scores)))
    
    p_group = np.nan
    if len(z_scores) >= min_stats:
        result = stats.ttest_1samp(z_scores, popmean=0.0, nan_policy="omit")
        p_group = _get_ttest_pvalue(result)
    
    n_trials = int(sum(len(x) for x, _ in pairs))
    
    # Bootstrap CI on z-scores
    ci = (np.nan, np.nan)
    if n_boot and len(subject_rs) >= min_stats:
        boots = []
        idx = np.arange(len(subject_rs))
        for _ in range(n_boot):
            sampled = rng.choice(idx, size=len(subject_rs), replace=True)
            z_boot = np.mean(z_scores[sampled])
            boots.append(_sf(np.tanh(z_boot)))
        
        if boots:
            ci = (_sf(np.percentile(boots, ci_lo)), _sf(np.percentile(boots, ci_hi)))
    
    return r_group, _sf(p_group), n_trials, len(subject_rs), ci, np.nan


def compute_group_corr_stats(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    method: str,
    *,
    strategy: str,
    n_cluster_boot: int = 0,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    """
    Compute group correlation statistics.
    
    Parameters
    ----------
    x_lists : List[np.ndarray]
        List of x arrays, one per subject
    y_lists : List[np.ndarray]
        List of y arrays, one per subject
    method : str
        'pearson' or 'spearman'
    strategy : str
        'pooled_trials', 'within_subject_centered', 'within_subject_zscored', or 'fisher_by_subject'
    n_cluster_boot : int
        Number of bootstrap iterations
    rng : np.random.Generator
        Random generator
    config : Any
        Configuration object
        
    Returns
    -------
    r : float
        Group correlation
    p : float
        Group p-value
    n_trials : int
        Total trials
    n_subjects : int
        Number of subjects
    ci : Tuple[float, float]
        95% CI
    p_pooled : float
        Pooled parametric p-value (usually NaN)
    """
    pairs = _build_valid_pairs(x_lists, y_lists, config)
    if not pairs:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan
    
    if rng is None:
        seed = int(get_config_value(config, "project.random_state", 42)) if config else 42
        rng = np.random.default_rng(seed)
    
    if strategy in {"pooled_trials", "within_subject_centered", "within_subject_zscored"}:
        return _compute_pooled_strategy_stats(pairs, strategy, method, n_cluster_boot, rng, config)
    
    return _compute_fisher_strategy_stats(pairs, method, n_cluster_boot, rng, config)


# ============================================================================
# Channel Rating Correlations
# ============================================================================

def compute_channel_rating_correlations(
    channel_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    bootstrap: Optional[int],
    n_perm: Optional[int],
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, float, int, float, float]:
    """Compute channel-rating correlation with all statistics."""
    from .bootstrap import bootstrap_corr_ci
    from .correlation import partial_corr_xy_given_Z
    from .permutation import compute_perm_and_partial_perm
    from .eeg_stats import compute_bootstrap_ci
    
    if channel_values.empty or target_values.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan
    
    r, p = compute_correlation(channel_values.values, target_values.values, "spearman" if use_spearman else "pearson")
    
    ci_lo, ci_hi = compute_bootstrap_ci(
        channel_values.values, target_values.values, bootstrap or 0, 0.95,
        "spearman" if use_spearman else "pearson", rng
    )
    
    r_partial = p_partial = np.nan
    n_partial = 0
    if covariates_df is not None and not covariates_df.empty:
        r_partial, p_partial, n_partial = partial_corr_xy_given_Z(
            channel_values, target_values, covariates_df, method, config
        )
    
    p_perm, p_partial_perm = compute_perm_and_partial_perm(
        channel_values, target_values, covariates_df, method, n_perm, rng, groups=groups, config=config
    )
    
    return r, p, ci_lo, ci_hi, r_partial, p_partial, n_partial, p_perm, p_partial_perm


def compute_temp_correlations_for_roi(
    roi_values: pd.Series,
    temp_values: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: Optional[int],
    n_perm: Optional[int],
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    band: str = "",
    roi: str = "",
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Compute temperature correlations for ROI."""
    from .eeg_stats import compute_bootstrap_ci
    from .permutation import compute_temp_permutation_pvalues
    
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    # Align
    combined = pd.concat([roi_values.rename("x"), temp_values.rename("y")], axis=1)
    if covariates_without_temp_df is not None:
        combined = pd.concat([combined, covariates_without_temp_df], axis=1)
    clean = combined.dropna()
    
    if len(clean) < min_samples:
        return None
    
    x_aligned = clean["x"]
    temp_aligned = clean["y"]
    cov_aligned = clean[covariates_without_temp_df.columns] if covariates_without_temp_df is not None else None
    n_valid = len(clean)
    
    r, p = compute_correlation(x_aligned.values, temp_aligned.values, "spearman" if use_spearman else "pearson")
    
    ci_lo, ci_hi = compute_bootstrap_ci(
        x_aligned.values, temp_aligned.values, bootstrap or 0, 0.95,
        "spearman" if use_spearman else "pearson", rng
    )
    
    p_perm, p_partial_perm = compute_temp_permutation_pvalues(
        x_aligned, temp_aligned, cov_aligned, method, n_perm, rng,
        band=band, roi=roi, logger=logger, groups=groups, config=config
    )
    
    def _sf(v):
        try:
            return float(v)
        except:
            return np.nan
    
    return {
        "roi": roi,
        "band": band,
        "band_range": "",
        "r": r,
        "p": p,
        "n": n_valid,
        "method": method,
        "r_ci_low": ci_lo,
        "r_ci_high": ci_hi,
        "r_partial": np.nan,
        "p_partial": np.nan,
        "n_partial": 0,
        "partial_covars": "",
        "p_perm": _sf(p_perm),
        "p_partial_perm": _sf(p_partial_perm),
        "n_perm": n_perm,
    }


def compute_temp_correlation_for_roi_pair(
    xi: pd.Series,
    temp_series: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    measure_band: str,
    roi_i: str,
    roi_j: str,
    n_edges: int,
    rng: np.random.Generator,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Compute temperature correlation for ROI pair (connectivity)."""
    from .eeg_stats import compute_bootstrap_ci
    from .correlation import joint_valid_mask
    from .partial import compute_partial_correlation_for_roi_pair
    from .permutation import compute_permutation_pvalues_for_roi_pair
    
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    min_len = min(len(xi), len(temp_series))
    xi_al = xi.iloc[:min_len]
    temp_al = temp_series.iloc[:min_len]
    
    mask = joint_valid_mask(xi_al, temp_al)
    n_eff = int(mask.sum())
    
    if n_eff < min_samples:
        return None
    
    method = "spearman" if use_spearman else "pearson"
    r, p = compute_correlation(xi_al[mask].values, temp_al[mask].values, method)
    
    r_partial, p_partial, n_partial = compute_partial_correlation_for_roi_pair(
        xi_al[mask], temp_al[mask], covariates_without_temp_df, mask, method
    )
    
    ci_lo, ci_hi = compute_bootstrap_ci(
        xi_al.values, temp_al.values, bootstrap, 0.95, method, rng
    )
    
    p_perm, p_partial_perm = compute_permutation_pvalues_for_roi_pair(
        xi_al[mask], temp_al[mask], covariates_without_temp_df, mask,
        method, n_perm, n_eff, rng, min_samples
    )
    
    def _sf(v):
        try:
            return float(v)
        except:
            return np.nan
    
    return {
        "measure_band": measure_band,
        "roi_i": roi_i,
        "roi_j": roi_j,
        "summary_type": "within" if roi_i == roi_j else "between",
        "n_edges": n_edges,
        "r": r,
        "p": p,
        "n": n_eff,
        "method": method,
        "r_ci_low": ci_lo,
        "r_ci_high": ci_hi,
        "r_partial": _sf(r_partial),
        "p_partial": _sf(p_partial),
        "n_partial": n_partial,
        "partial_covars": "",
        "p_perm": _sf(p_perm),
        "p_partial_perm": _sf(p_partial_perm),
        "n_perm": n_perm,
    }


def compute_correlation_for_time_freq_bin(
    power: np.ndarray,
    y_array: np.ndarray,
    times: np.ndarray,
    f_idx: int,
    t_start: float,
    t_end: float,
    min_valid_points: int,
    use_spearman: bool,
) -> Tuple[Optional[float], Optional[float], int]:
    """Compute correlation for a time-frequency bin."""
    time_mask = (times >= t_start) & (times < t_end)
    if not np.any(time_mask):
        return None, None, 0
    
    mean_power = power[:, f_idx, time_mask].mean(axis=1)
    valid = np.isfinite(mean_power) & np.isfinite(y_array)
    n = int(valid.sum())
    
    if n < min_valid_points:
        return None, None, n
    
    r, p = compute_correlation(mean_power[valid], y_array[valid], "spearman" if use_spearman else "pearson")
    return r, p, n


def compute_correlation_from_vectors(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    min_valid_points: int,
    use_spearman: bool,
) -> Tuple[float, float]:
    """Compute correlation from vectors."""
    mask = np.isfinite(x_vec) & np.isfinite(y_vec)
    if mask.sum() < min_valid_points:
        return np.nan, np.nan
    return compute_correlation(x_vec[mask], y_vec[mask], "spearman" if use_spearman else "pearson")


# ============================================================================
# EEG Cluster Test (wrapping cluster.py)
# ============================================================================

def cluster_test_epochs(
    tfr_epochs: "mne.time_frequency.EpochsTFR",
    group_a_mask: np.ndarray,
    group_b_mask: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    paired: bool = False,
    alpha: Optional[float] = None,
    n_permutations: Optional[int] = None,
    restrict_picks: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Cluster test on EpochsTFR data."""
    import mne
    from .cluster import cluster_test_two_sample
    
    info = tfr_epochs.info
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        if logger:
            logger.warning("No EEG channels found")
        return None, None, None, None
    
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    t_mask = (times >= tmin) & (times < tmax)
    
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None, None, None, None
    
    data = np.asarray(tfr_epochs.data)[:, :, f_mask, :][:, :, :, t_mask]
    ch_power = data.mean(axis=(2, 3))
    
    if ch_power.shape[1] != len(info["ch_names"]):
        if logger:
            logger.error("Channel dimension mismatch")
        return None, None, None, None
    
    group_a = ch_power[np.asarray(group_a_mask, dtype=bool), :]
    group_b = ch_power[np.asarray(group_b_mask, dtype=bool), :]
    
    if group_a.shape[0] < 2 or group_b.shape[0] < 2:
        return None, None, None, None
    
    return cluster_test_two_sample(
        group_a, group_b, info, alpha=alpha, paired=paired,
        n_permutations=n_permutations, restrict_picks=restrict_picks,
        n_jobs=n_jobs, config=config
    )





