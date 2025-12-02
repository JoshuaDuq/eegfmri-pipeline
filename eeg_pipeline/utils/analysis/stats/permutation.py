"""
Permutation Testing
===================

Permutation tests for partial correlations and group comparisons.
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
from .correlation import compute_correlation


def _permute_within_groups(
    n: int,
    rng: np.random.Generator,
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Permute indices within groups."""
    if groups is None:
        return rng.permutation(n)
    perm = []
    for g in np.unique(groups):
        g_idx = np.where(groups == g)[0]
        perm.extend(g_idx[rng.permutation(len(g_idx))])
    return np.asarray(perm, dtype=int)


def perm_pval_partial_freedman_lane(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    *,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> float:
    """Freedman-Lane permutation test for partial correlation."""
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    if len(df) < min_samples or n_perm is None or n_perm <= 0:
        return np.nan
    
    intercept = np.ones(len(df))
    
    if method == "spearman":
        x_vals = stats.rankdata(df["x"].to_numpy())
        y_vals = stats.rankdata(df["y"].to_numpy())
        Z_vals = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
    else:
        x_vals = df["x"].to_numpy()
        y_vals = df["y"].to_numpy()
        Z_vals = df[Z.columns].to_numpy() if len(Z.columns) else np.empty((len(df), 0))
    
    design = np.column_stack([intercept, Z_vals])
    cond = np.linalg.cond(design)
    if not np.isfinite(cond) or cond > 1e8 or np.linalg.matrix_rank(design, tol=1e-10) < design.shape[1]:
        return np.nan
    
    x_coef = np.linalg.lstsq(design, x_vals, rcond=None)[0]
    y_coef = np.linalg.lstsq(design, y_vals, rcond=None)[0]
    x_res = x_vals - design @ x_coef
    y_res = y_vals - design @ y_coef
    y_hat = design @ y_coef
    
    var_tol = 1e-12 * max(np.var(x_vals), np.var(y_vals), 1.0)
    if np.var(x_res, ddof=1) < var_tol or np.var(y_res, ddof=1) < var_tol:
        return np.nan
    
    obs_r, _ = stats.pearsonr(x_res, y_res)
    exceed = 1
    
    groups_arr = np.asarray(groups) if groups is not None else None
    if groups_arr is not None and len(groups_arr) != len(df):
        raise ValueError("groups length mismatch")
    
    for _ in range(n_perm):
        perm_idx = _permute_within_groups(len(y_res), rng, groups_arr)
        y_perm = y_hat + y_res[perm_idx]
        
        try:
            y_perm_coef = np.linalg.lstsq(design, y_perm, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        
        y_perm_res = y_perm - design @ y_perm_coef
        if np.var(y_perm_res, ddof=1) < var_tol:
            continue
        
        perm_r, _ = stats.pearsonr(x_res, y_perm_res)
        if np.abs(perm_r) >= np.abs(obs_r):
            exceed += 1
    
    return exceed / (n_perm + 1)


def compute_perm_and_partial_perm(
    x: pd.Series,
    y: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    *,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute permutation p-values for simple and partial correlation."""
    from .bootstrap import perm_pval_simple
    
    p_perm = p_partial_perm = np.nan
    if n_perm is None or n_perm <= 0:
        return p_perm, p_partial_perm
    
    p_perm = perm_pval_simple(x, y, method, n_perm, rng, groups=groups, config=config)
    
    if covariates_df is not None and not covariates_df.empty:
        p_partial_perm = perm_pval_partial_freedman_lane(x, y, covariates_df, method, n_perm, rng, groups=groups, config=config)
    
    return p_perm, p_partial_perm


def compute_permutation_pvalue_partial(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: pd.DataFrame,
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    context: str = "",
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> float:
    """Compute permutation p-value for partial correlation."""
    return perm_pval_partial_freedman_lane(x_aligned, y_aligned, covariates_df, method, n_perm, rng, groups=groups, config=config)


def compute_permutation_pvalues(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    method: str,
    n_perm: Optional[int],
    n_eff: int,
    rng: np.random.Generator,
    band: str = "",
    roi: str = "",
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute all permutation p-values for ROI analysis."""
    from .bootstrap import perm_pval_simple
    
    p_perm = p_partial_perm = p_temp_perm = np.nan
    
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    if n_perm is None or n_perm <= 0 or n_eff < min_samples:
        return p_perm, p_partial_perm, p_temp_perm
    
    p_perm = perm_pval_simple(x_aligned, y_aligned, method, n_perm, rng, groups=groups, config=config)
    
    if covariates_df is not None and not covariates_df.empty:
        p_partial_perm = perm_pval_partial_freedman_lane(x_aligned, y_aligned, covariates_df, method, n_perm, rng, groups=groups, config=config)
    
    if temp_series is not None and not temp_series.empty:
        temp_cov = pd.DataFrame({"temp": temp_series})
        p_temp_perm = perm_pval_partial_freedman_lane(x_aligned, y_aligned, temp_cov, method, n_perm, rng, config=config)
    
    return p_perm, p_partial_perm, p_temp_perm


def compute_temp_permutation_pvalues(
    roi_values: pd.Series,
    temp_values: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    band: str = "",
    roi: str = "",
    logger: Optional[logging.Logger] = None,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute temperature permutation p-values."""
    from .bootstrap import perm_pval_simple
    
    p_temp_perm = p_partial_perm = np.nan
    
    if n_perm is None or n_perm <= 0:
        return p_temp_perm, p_partial_perm
    
    p_temp_perm = perm_pval_simple(roi_values, temp_values, method, n_perm, rng, groups=groups, config=config)
    
    if covariates_without_temp_df is not None and not covariates_without_temp_df.empty:
        p_partial_perm = perm_pval_partial_freedman_lane(roi_values, temp_values, covariates_without_temp_df, method, n_perm, rng, groups=groups, config=config)
    
    return p_temp_perm, p_partial_perm


def compute_permutation_pvalues_for_roi_pair(
    x_masked: pd.Series,
    y_masked: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    mask: pd.Series,
    method: str,
    n_perm: Optional[int],
    n_eff: int,
    rng: np.random.Generator,
    min_samples: int = 5,
) -> Tuple[float, float]:
    """Compute permutation p-values for ROI pair."""
    p_perm = p_partial_perm = np.nan
    
    if n_perm is None or n_perm <= 0 or n_eff < min_samples:
        return p_perm, p_partial_perm
    
    cov_valid = None
    groups = None
    
    if covariates_df is not None and not covariates_df.empty:
        cov_valid = covariates_df.iloc[mask] if hasattr(mask, '__iter__') else covariates_df[mask]
        if cov_valid.empty:
            cov_valid = None
        
        # Try to extract groups
        for col in cov_valid.columns if cov_valid is not None else []:
            if str(col).lower() in ["run", "run_id", "block"]:
                candidate = pd.to_numeric(cov_valid[col], errors="coerce").to_numpy()
                if np.unique(candidate[~np.isnan(candidate)]).size < len(candidate):
                    groups = candidate
                    break
    
    p_perm, p_partial_perm = compute_perm_and_partial_perm(x_masked, y_masked, cov_valid, method, n_perm, rng, groups=groups)
    return p_perm, p_partial_perm





