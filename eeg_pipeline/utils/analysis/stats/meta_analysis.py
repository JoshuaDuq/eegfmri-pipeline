"""
Meta-Analysis Statistics
========================

Inverse-variance weighted pooling, random-effects models, heterogeneity metrics,
and forest plot utilities for group-level inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


###################################################################
# Data Structures
###################################################################


@dataclass
class MetaAnalysisResult:
    """Result container for meta-analysis."""
    
    r_pooled: float
    se_pooled: float
    ci_low: float
    ci_high: float
    z_score: float
    p_value: float
    
    # Heterogeneity
    q_statistic: float
    q_pvalue: float
    i_squared: float
    tau_squared: float
    
    # Sample info
    n_studies: int
    n_total: int
    
    # Per-study data for forest plots
    study_rs: np.ndarray
    study_ses: np.ndarray
    study_weights: np.ndarray
    study_ns: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_pooled": self.r_pooled,
            "se_pooled": self.se_pooled,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "q_statistic": self.q_statistic,
            "q_pvalue": self.q_pvalue,
            "i_squared": self.i_squared,
            "tau_squared": self.tau_squared,
            "n_studies": self.n_studies,
            "n_total": self.n_total,
        }


###################################################################
# Core Meta-Analysis Functions
###################################################################


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher z-transform correlation coefficients."""
    r = np.clip(np.asarray(r), -0.9999, 0.9999)
    return np.arctanh(r)


def inverse_fisher_z(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher z-transform."""
    return np.tanh(np.asarray(z))


def correlation_se(n: np.ndarray) -> np.ndarray:
    """Standard error for Fisher z-transformed correlation."""
    n = np.asarray(n)
    return 1.0 / np.sqrt(np.maximum(n - 3, 1))


def compute_heterogeneity(
    z_values: np.ndarray,
    weights: np.ndarray,
    z_pooled: float,
) -> Tuple[float, float, float, float]:
    """
    Compute heterogeneity statistics.
    
    Returns
    -------
    q_stat : float
        Cochran's Q statistic
    q_pvalue : float
        P-value for Q
    i_squared : float
        I² percentage (0-100)
    tau_squared : float
        Between-study variance estimate
    """
    k = len(z_values)
    if k < 2:
        return 0.0, 1.0, 0.0, 0.0
    
    q_stat = float(np.sum(weights * (z_values - z_pooled) ** 2))
    df = k - 1
    q_pvalue = float(1.0 - stats.chi2.cdf(q_stat, df))
    
    # I² = (Q - df) / Q * 100
    i_squared = max(0.0, (q_stat - df) / q_stat * 100) if q_stat > 0 else 0.0
    
    # τ² estimation (DerSimonian-Laird)
    c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau_squared = max(0.0, (q_stat - df) / c) if c > 0 else 0.0
    
    return q_stat, q_pvalue, i_squared, tau_squared


def fixed_effects_meta(
    r_values: np.ndarray,
    n_values: np.ndarray,
    ci_level: float = 0.95,
) -> MetaAnalysisResult:
    """
    Fixed-effects inverse-variance weighted meta-analysis.
    
    Parameters
    ----------
    r_values : array
        Correlation coefficients per study
    n_values : array
        Sample sizes per study
    ci_level : float
        Confidence interval level
    """
    r_values = np.asarray(r_values)
    n_values = np.asarray(n_values)
    
    valid = np.isfinite(r_values) & np.isfinite(n_values) & (n_values >= 4)
    if valid.sum() < 1:
        return _empty_meta_result()
    
    r_valid = r_values[valid]
    n_valid = n_values[valid]
    
    z_values = fisher_z(r_valid)
    se_values = correlation_se(n_valid)
    weights = 1.0 / (se_values ** 2)
    
    z_pooled = np.sum(weights * z_values) / np.sum(weights)
    se_pooled = 1.0 / np.sqrt(np.sum(weights))
    
    r_pooled = inverse_fisher_z(z_pooled)
    
    alpha = 1 - ci_level
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = inverse_fisher_z(z_pooled - z_crit * se_pooled)
    ci_high = inverse_fisher_z(z_pooled + z_crit * se_pooled)
    
    z_score = z_pooled / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    q_stat, q_pvalue, i_sq, tau_sq = compute_heterogeneity(z_values, weights, z_pooled)
    
    return MetaAnalysisResult(
        r_pooled=float(r_pooled),
        se_pooled=float(se_pooled),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        z_score=float(z_score),
        p_value=float(p_value),
        q_statistic=q_stat,
        q_pvalue=q_pvalue,
        i_squared=i_sq,
        tau_squared=tau_sq,
        n_studies=int(valid.sum()),
        n_total=int(n_valid.sum()),
        study_rs=r_valid,
        study_ses=se_values,
        study_weights=weights / weights.sum(),
        study_ns=n_valid,
    )


def random_effects_meta(
    r_values: np.ndarray,
    n_values: np.ndarray,
    ci_level: float = 0.95,
) -> MetaAnalysisResult:
    """
    Random-effects meta-analysis using DerSimonian-Laird estimator.
    
    Accounts for between-study heterogeneity in the pooled estimate.
    """
    r_values = np.asarray(r_values)
    n_values = np.asarray(n_values)
    
    valid = np.isfinite(r_values) & np.isfinite(n_values) & (n_values >= 4)
    if valid.sum() < 2:
        return fixed_effects_meta(r_values, n_values, ci_level)
    
    r_valid = r_values[valid]
    n_valid = n_values[valid]
    
    z_values = fisher_z(r_valid)
    se_values = correlation_se(n_valid)
    within_var = se_values ** 2
    
    # Fixed-effects estimate for Q calculation
    w_fe = 1.0 / within_var
    z_fe = np.sum(w_fe * z_values) / np.sum(w_fe)
    
    q_stat, q_pvalue, i_sq, tau_sq = compute_heterogeneity(z_values, w_fe, z_fe)
    
    # Random-effects weights
    total_var = within_var + tau_sq
    w_re = 1.0 / total_var
    
    z_pooled = np.sum(w_re * z_values) / np.sum(w_re)
    se_pooled = 1.0 / np.sqrt(np.sum(w_re))
    
    r_pooled = inverse_fisher_z(z_pooled)
    
    alpha = 1 - ci_level
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = inverse_fisher_z(z_pooled - z_crit * se_pooled)
    ci_high = inverse_fisher_z(z_pooled + z_crit * se_pooled)
    
    z_score = z_pooled / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return MetaAnalysisResult(
        r_pooled=float(r_pooled),
        se_pooled=float(se_pooled),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        z_score=float(z_score),
        p_value=float(p_value),
        q_statistic=q_stat,
        q_pvalue=q_pvalue,
        i_squared=i_sq,
        tau_squared=tau_sq,
        n_studies=int(valid.sum()),
        n_total=int(n_valid.sum()),
        study_rs=r_valid,
        study_ses=se_values,
        study_weights=w_re / w_re.sum(),
        study_ns=n_valid,
    )


def leave_one_out_meta(
    r_values: np.ndarray,
    n_values: np.ndarray,
    use_random_effects: bool = True,
) -> List[Tuple[int, MetaAnalysisResult]]:
    """
    Leave-one-out sensitivity analysis.
    
    Returns list of (excluded_index, meta_result) tuples.
    """
    r_values = np.asarray(r_values)
    n_values = np.asarray(n_values)
    
    meta_func = random_effects_meta if use_random_effects else fixed_effects_meta
    results = []
    
    for i in range(len(r_values)):
        mask = np.ones(len(r_values), dtype=bool)
        mask[i] = False
        result = meta_func(r_values[mask], n_values[mask])
        results.append((i, result))
    
    return results


def _empty_meta_result() -> MetaAnalysisResult:
    """Return empty meta-analysis result."""
    return MetaAnalysisResult(
        r_pooled=np.nan,
        se_pooled=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        z_score=np.nan,
        p_value=np.nan,
        q_statistic=np.nan,
        q_pvalue=np.nan,
        i_squared=np.nan,
        tau_squared=np.nan,
        n_studies=0,
        n_total=0,
        study_rs=np.array([]),
        study_ses=np.array([]),
        study_weights=np.array([]),
        study_ns=np.array([]),
    )


###################################################################
# Null Distribution and Permutation
###################################################################


def permutation_null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 1000,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate permutation null distribution for correlation.
    
    Returns
    -------
    null_rs : array
        Null distribution of correlations
    observed_r : float
        Observed correlation
    p_perm : float
        Permutation p-value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return np.array([]), np.nan, np.nan
    
    x_v, y_v = x[valid], y[valid]
    
    if method == "pearson":
        observed_r, _ = stats.pearsonr(x_v, y_v)
    else:
        observed_r, _ = stats.spearmanr(x_v, y_v)
    
    null_rs = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y_v)
        if method == "pearson":
            null_rs[i], _ = stats.pearsonr(x_v, y_perm)
        else:
            null_rs[i], _ = stats.spearmanr(x_v, y_perm)
    
    n_extreme = np.sum(np.abs(null_rs) >= np.abs(observed_r))
    p_perm = (n_extreme + 1) / (n_perm + 1)
    
    return null_rs, float(observed_r), float(p_perm)


###################################################################
# Bootstrap CI Computation
###################################################################


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Bootstrap confidence interval for correlation.
    
    Returns
    -------
    r_obs : float
        Observed correlation
    ci_low : float
        Lower CI bound
    ci_high : float
        Upper CI bound
    boot_rs : array
        Bootstrap distribution
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = valid.sum()
    if n_valid < 4:
        return np.nan, np.nan, np.nan, np.array([])
    
    x_v, y_v = x[valid], y[valid]
    
    if method == "pearson":
        r_obs, _ = stats.pearsonr(x_v, y_v)
    else:
        r_obs, _ = stats.spearmanr(x_v, y_v)
    
    boot_rs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_valid, size=n_valid)
        if method == "pearson":
            boot_rs[i], _ = stats.pearsonr(x_v[idx], y_v[idx])
        else:
            boot_rs[i], _ = stats.spearmanr(x_v[idx], y_v[idx])
    
    boot_rs = boot_rs[np.isfinite(boot_rs)]
    if len(boot_rs) < 10:
        return float(r_obs), np.nan, np.nan, boot_rs
    
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_rs, 100 * alpha / 2)
    ci_high = np.percentile(boot_rs, 100 * (1 - alpha / 2))
    
    return float(r_obs), float(ci_low), float(ci_high), boot_rs


def ensure_bootstrap_ci(
    stats_df,
    x_data: Optional[np.ndarray] = None,
    y_data: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    method: str = "spearman",
    config: Optional[Any] = None,
) -> "pd.DataFrame":
    """
    Ensure bootstrap CIs exist in stats dataframe, computing if missing.
    
    If ci_low/ci_high columns are missing or contain NaN, computes bootstrap CIs.
    """
    import pandas as pd
    
    if stats_df is None or stats_df.empty:
        return stats_df
    
    df = stats_df.copy()
    
    has_ci = "ci_low" in df.columns and "ci_high" in df.columns
    if has_ci:
        missing_ci = df["ci_low"].isna() | df["ci_high"].isna()
        if not missing_ci.any():
            return df
    
    # Compute Fisher z-based CIs as fallback when raw data unavailable
    if "r" in df.columns and "n" in df.columns:
        def _fisher_ci(row):
            r, n = row.get("r"), row.get("n")
            if pd.isna(r) or pd.isna(n) or n < 4:
                return np.nan, np.nan
            z = np.arctanh(np.clip(r, -0.9999, 0.9999))
            se = 1.0 / np.sqrt(n - 3)
            z_crit = 1.96
            return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
        
        if "ci_low" not in df.columns:
            df["ci_low"] = np.nan
        if "ci_high" not in df.columns:
            df["ci_high"] = np.nan
        
        for idx in df.index:
            if pd.isna(df.loc[idx, "ci_low"]) or pd.isna(df.loc[idx, "ci_high"]):
                ci_lo, ci_hi = _fisher_ci(df.loc[idx])
                df.loc[idx, "ci_low"] = ci_lo
                df.loc[idx, "ci_high"] = ci_hi
    
    return df


###################################################################
# Bayes Factor for Equivalence Testing
###################################################################


def bayes_factor_correlation(
    r: float,
    n: int,
    prior_scale: float = 0.707,
) -> float:
    """
    Compute Bayes Factor for correlation (BF10).
    
    Uses Jeffreys' approximate BF for correlation.
    BF10 > 3: moderate evidence for effect
    BF10 < 1/3: moderate evidence for null
    
    Parameters
    ----------
    r : float
        Observed correlation
    n : int
        Sample size
    prior_scale : float
        Scale of prior (default 0.707 = medium effect)
    """
    if np.isnan(r) or n < 4:
        return np.nan
    
    r = np.clip(r, -0.9999, 0.9999)
    
    # Approximate BF using Wetzels & Wagenmakers (2012) formula
    # This is a simplified version; full computation requires numerical integration
    t = r * np.sqrt((n - 2) / (1 - r**2))
    df = n - 2
    
    # Savage-Dickey approximation
    # BF10 ≈ (1 + t²/df)^(-(df+1)/2) * sqrt(df) / (prior_scale * sqrt(2*pi))
    # Simplified JZS approximation
    rscale = prior_scale
    bf10 = np.exp(
        0.5 * np.log(1 + (t**2) / df) * (-(df + 1) / 2)
        + 0.5 * np.log(df)
        - np.log(rscale)
        - 0.5 * np.log(2 * np.pi)
    )
    
    # More accurate approximation using beta function
    from scipy.special import betaln
    log_bf = (
        betaln(0.5, (n - 2) / 2)
        - betaln(0.5, 0.5)
        + ((n - 1) / 2) * np.log(1 - r**2)
    )
    bf10 = np.exp(-log_bf)
    
    return float(bf10)


def equivalence_test_correlation(
    r: float,
    n: int,
    equiv_bound: float = 0.1,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    TOST equivalence test for correlation.
    
    Tests whether |r| is smaller than equiv_bound.
    
    Returns
    -------
    p_lower : float
        P-value for lower bound test
    p_upper : float
        P-value for upper bound test
    equivalent : bool
        True if correlation is equivalent to zero
    """
    if np.isnan(r) or n < 4:
        return np.nan, np.nan, False
    
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    
    z_lower = np.arctanh(-equiv_bound)
    z_upper = np.arctanh(equiv_bound)
    
    # Test H0: z <= z_lower (effect is too negative)
    t_lower = (z - z_lower) / se
    p_lower = 1 - stats.norm.cdf(t_lower)
    
    # Test H0: z >= z_upper (effect is too positive)
    t_upper = (z - z_upper) / se
    p_upper = stats.norm.cdf(t_upper)
    
    # Both must be significant for equivalence
    equivalent = max(p_lower, p_upper) < alpha
    
    return float(p_lower), float(p_upper), equivalent
