"""
Band Statistics
===============

Inter-band correlations and power statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_statistics_constants


def compute_band_spatial_correlation(
    band1_channels: np.ndarray,
    band2_channels: np.ndarray,
) -> float:
    """Compute spatial correlation between bands."""
    if len(band1_channels) <= 1 or len(band2_channels) <= 1:
        return np.nan
    return float(np.corrcoef(band1_channels, band2_channels)[0, 1])


def compute_band_pair_correlation(
    vec_i: Optional[Dict[str, float]],
    vec_j: Optional[Dict[str, float]],
) -> float:
    """Compute correlation between band topographies."""
    if vec_i is None or vec_j is None:
        return np.nan
    
    common = sorted(set(vec_i.keys()) & set(vec_j.keys()))
    if len(common) < 2:
        return np.nan
    
    vals_i = np.array([vec_i[ch] for ch in common])
    vals_j = np.array([vec_j[ch] for ch in common])
    
    if np.std(vals_i) < 1e-12 or np.std(vals_j) < 1e-12:
        return np.nan
    
    return float(np.corrcoef(vals_i, vals_j)[0, 1])


def compute_subject_band_correlation_matrix(
    band_vectors: Dict[str, Dict[str, float]],
    band_names: List[str],
) -> np.ndarray:
    """Compute inter-band correlation matrix for one subject."""
    n = len(band_names)
    mat = np.eye(n)
    for i, bi in enumerate(band_names):
        for j in range(i + 1, n):
            r = compute_band_pair_correlation(band_vectors.get(bi), band_vectors.get(band_names[j]))
            mat[i, j] = mat[j, i] = r
    return mat


def fisher_z_transform_mean(r_values: np.ndarray, config: Optional[Any] = None) -> float:
    """Compute Fisher z-transformed mean of correlations."""
    from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values
    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_clipped = np.clip(r_values, clip_min, clip_max)
    z_scores = np.arctanh(r_clipped)
    return float(np.tanh(np.mean(z_scores)))


def compute_group_band_correlation_matrix(
    per_subject_correlations: List[np.ndarray],
    n_bands: int,
) -> np.ndarray:
    """Aggregate inter-band correlations across subjects."""
    group_mat = np.eye(n_bands)
    arr = np.stack(per_subject_correlations, axis=0)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            r_vals = arr[:, i, j]
            r_vals = r_vals[np.isfinite(r_vals)]
            if r_vals.size == 0:
                group_mat[i, j] = group_mat[j, i] = np.nan
            else:
                group_mat[i, j] = group_mat[j, i] = fisher_z_transform_mean(r_vals)
    
    return group_mat


def compute_band_statistics_array(
    values: np.ndarray,
    ci_multiplier: float = 1.96,
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """Compute band statistics from array."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan, 0
    
    mean_val = float(np.mean(clean))
    n = len(clean)
    se = float(np.std(clean, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    delta = ci_multiplier * se if np.isfinite(se) else np.nan
    
    return mean_val, mean_val - delta if np.isfinite(delta) else np.nan, mean_val + delta if np.isfinite(delta) else np.nan, n


def compute_inter_band_correlation_statistics(
    per_subject_correlations: List[np.ndarray],
    band_names: List[str],
    ci_multiplier: float = 1.96,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Compute inter-band correlation statistics across subjects."""
    from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values
    clip_min, clip_max = get_fisher_z_clip_values(config)
    rows = []
    n_bands = len(band_names)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            r_vals = np.array([cm[i, j] for cm in per_subject_correlations])
            r_vals = r_vals[np.isfinite(r_vals)]
            if r_vals.size == 0:
                continue
            
            z_scores = np.arctanh(np.clip(r_vals, clip_min, clip_max))
            z_mean = float(np.mean(z_scores))
            n = len(z_scores)
            se = float(np.std(z_scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            
            rows.append({
                "band_i": band_names[i],
                "band_j": band_names[j],
                "r_group": float(np.tanh(z_mean)),
                "r_ci_low": float(np.tanh(z_mean - ci_multiplier * se)) if np.isfinite(se) else np.nan,
                "r_ci_high": float(np.tanh(z_mean + ci_multiplier * se)) if np.isfinite(se) else np.nan,
                "n_subjects": n,
            })
    
    return rows


def compute_correlation_ci_fisher(
    z_mean: float,
    se: float,
    ci_multiplier: float = 1.96,
) -> Tuple[float, float]:
    """Compute CI from Fisher z mean and SE."""
    if not np.isfinite(se):
        return np.nan, np.nan
    return float(np.tanh(z_mean - ci_multiplier * se)), float(np.tanh(z_mean + ci_multiplier * se))


def compute_band_correlations(
    pow_df: pd.DataFrame,
    y: pd.Series,
    band: str,
    power_prefix: str = "pow_",
    min_samples: int = 3,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute channel-wise correlations for a band."""
    band_l = str(band).lower()

    # Canonical NamingSchema columns:
    # power_{segment}_{band}_ch_{channel}_{stat}
    prefix = "power_"
    token = f"_{band_l}_ch_"
    candidate_cols = [
        c
        for c in pow_df.columns
        if str(c).lower().startswith(prefix) and token in str(c).lower()
    ]

    if not candidate_cols:
        return [], np.array([]), np.array([])

    import re

    # Example: power_plateau_alpha_ch_Fz_logratio
    pattern = re.compile(rf"^power_[^_]+_{re.escape(band_l)}_ch_(.+?)_", re.IGNORECASE)
    band_cols: List[str] = []
    ch_names: List[str] = []
    for col in candidate_cols:
        m = pattern.match(str(col))
        if m is None:
            continue
        band_cols.append(col)
        ch_names.append(m.group(1))

    if not band_cols:
        return [], np.array([]), np.array([])
    corrs, pvals = [], []
    
    for col in band_cols:
        valid = pd.concat([pow_df[col], y], axis=1).dropna()
        if len(valid) >= min_samples:
            r, p = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        else:
            r, p = np.nan, 1.0
        corrs.append(r)
        pvals.append(p)
    
    return ch_names, np.array(corrs), np.array(pvals)


def compute_connectivity_correlations(
    conn_df: pd.DataFrame,
    y: pd.Series,
    measure_cols: List[str],
    measure: str,
    band: str,
    min_samples: int = 3,
    min_correlation: float = 0.3,
    max_pvalue: float = 0.05,
) -> Tuple[List[float], List[str]]:
    """Compute significant connectivity correlations."""
    correlations, connections = [], []
    prefix = f'{measure}_{band}_'
    
    for col in measure_cols:
        valid = ~(conn_df[col].isna() | y.isna())
        if valid.sum() < min_samples:
            continue
        
        x_v, y_v = conn_df[col][valid].to_numpy(), y[valid].to_numpy()
        if np.std(x_v) <= 0 or np.std(y_v) <= 0:
            continue
        
        r, p = stats.spearmanr(x_v, y_v)
        if abs(r) > min_correlation and p < max_pvalue:
            correlations.append(r)
            connections.append(col.replace(prefix, '').replace('conn_', ''))
    
    return correlations, connections


def compute_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: Optional[np.random.Generator],
    min_samples: int = 3,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute correlation with optional bootstrap CI."""
    from .bootstrap import bootstrap_corr_ci
    
    mask = np.isfinite(x) & np.isfinite(y)
    n_eff = int(mask.sum())
    
    if n_eff < min_samples:
        return np.nan, np.nan, n_eff, (np.nan, np.nan)
    
    x_v, y_v = x[mask], y[mask]
    r, p = stats.spearmanr(x_v, y_v, nan_policy="omit")
    
    ci = (np.nan, np.nan)
    if bootstrap_ci > 0:
        ci = bootstrap_corr_ci(x_v, y_v, method_code, n_boot=bootstrap_ci, rng=rng)
    
    return float(r), float(p), n_eff, ci


def compute_partial_residuals_stats(
    x_res: pd.Series,
    y_res: pd.Series,
    stats_df: Optional[pd.Series],
    n_res: int,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute partial correlation statistics from residuals."""
    from .bootstrap import bootstrap_corr_ci
    
    def _sf(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(v)
    
    r_resid = p_resid = np.nan
    n_partial = n_res
    
    if stats_df is not None:
        r_resid = _sf(stats_df.get("r_partial", np.nan))
        p_resid = _sf(stats_df.get("p_partial", np.nan))
        n_partial = int(stats_df.get("n_partial", n_partial))
    
    if not np.isfinite(r_resid):
        r_resid, p_resid = stats.spearmanr(x_res, y_res, nan_policy="omit")
    
    ci = (np.nan, np.nan)
    if bootstrap_ci > 0:
        ci = bootstrap_corr_ci(x_res, y_res, method_code, n_boot=bootstrap_ci, rng=rng)
    
    return float(r_resid), float(p_resid), n_partial, ci






