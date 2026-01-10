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
import pandas as pd
from scipy import stats
from scipy.special import betaln


_MIN_SAMPLE_SIZE = 4
_MIN_CORRELATION = -0.9999
_MAX_CORRELATION = 0.9999
_DEFAULT_PRIOR_SCALE = 0.707
_DEFAULT_Z_CRITICAL = 1.96


@dataclass
class MetaAnalysisResult:
    """Result container for meta-analysis."""
    
    r_pooled: float
    se_pooled: float
    ci_low: float
    ci_high: float
    z_score: float
    p_value: float
    q_statistic: float
    q_pvalue: float
    i_squared: float
    tau_squared: float
    n_studies: int
    n_total: int
    study_rs: np.ndarray
    study_ses: np.ndarray
    study_weights: np.ndarray
    study_ns: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
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


# Import unified Fisher z functions from correlation module
from .correlation import fisher_z, inverse_fisher_z

# Import ensure_bootstrap_ci from bootstrap module to avoid duplication
from .bootstrap import ensure_bootstrap_ci

# Re-export for backward compatibility (aliases for array-focused usage)
# The functions in correlation.py now handle both scalars and arrays


def correlation_se(n: np.ndarray) -> np.ndarray:
    """Standard error for Fisher z-transformed correlation."""
    n_array = np.asarray(n)
    degrees_of_freedom = np.maximum(n_array - 3, 1)
    return 1.0 / np.sqrt(degrees_of_freedom)


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
    n_studies = len(z_values)
    if n_studies < 2:
        return 0.0, 1.0, 0.0, 0.0
    
    squared_deviations = (z_values - z_pooled) ** 2
    q_stat = float(np.sum(weights * squared_deviations))
    degrees_of_freedom = n_studies - 1
    q_pvalue = float(1.0 - stats.chi2.cdf(q_stat, degrees_of_freedom))
    
    i_squared = 0.0
    if q_stat > 0:
        i_squared = max(0.0, (q_stat - degrees_of_freedom) / q_stat * 100)
    
    sum_weights = np.sum(weights)
    sum_squared_weights = np.sum(weights ** 2)
    denominator = sum_weights - sum_squared_weights / sum_weights
    tau_squared = 0.0
    if denominator > 0:
        tau_squared = max(0.0, (q_stat - degrees_of_freedom) / denominator)
    
    return q_stat, q_pvalue, i_squared, tau_squared


def _compute_confidence_interval(
    z_pooled: float,
    se_pooled: float,
    ci_level: float,
) -> Tuple[float, float]:
    """Compute confidence interval for pooled z-value."""
    from .base import get_z_critical_value
    z_critical = get_z_critical_value(ci_level)
    z_lower = z_pooled - z_critical * se_pooled
    z_upper = z_pooled + z_critical * se_pooled
    ci_low = inverse_fisher_z(z_lower)
    ci_high = inverse_fisher_z(z_upper)
    return float(ci_low), float(ci_high)


def _compute_p_value(z_score: float) -> float:
    """Compute two-tailed p-value from z-score."""
    return float(2 * (1 - stats.norm.cdf(abs(z_score))))


def _validate_inputs(
    r_values: np.ndarray,
    n_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and filter inputs for meta-analysis."""
    r_array = np.asarray(r_values)
    n_array = np.asarray(n_values)
    
    is_finite_r = np.isfinite(r_array)
    is_finite_n = np.isfinite(n_array)
    has_sufficient_n = n_array >= _MIN_SAMPLE_SIZE
    valid_mask = is_finite_r & is_finite_n & has_sufficient_n
    
    return r_array, n_array, valid_mask


def _build_meta_result(
    z_pooled: float,
    se_pooled: float,
    ci_level: float,
    z_values: np.ndarray,
    weights: np.ndarray,
    r_valid: np.ndarray,
    se_values: np.ndarray,
    n_valid: np.ndarray,
    n_studies: int,
) -> MetaAnalysisResult:
    """Build MetaAnalysisResult from computed values."""
    r_pooled = inverse_fisher_z(z_pooled)
    ci_low, ci_high = _compute_confidence_interval(z_pooled, se_pooled, ci_level)
    z_score = z_pooled / se_pooled
    p_value = _compute_p_value(z_score)
    q_stat, q_pvalue, i_squared, tau_squared = compute_heterogeneity(
        z_values, weights, z_pooled
    )
    normalized_weights = weights / weights.sum()
    
    return MetaAnalysisResult(
        r_pooled=float(r_pooled),
        se_pooled=float(se_pooled),
        ci_low=ci_low,
        ci_high=ci_high,
        z_score=float(z_score),
        p_value=p_value,
        q_statistic=q_stat,
        q_pvalue=q_pvalue,
        i_squared=i_squared,
        tau_squared=tau_squared,
        n_studies=n_studies,
        n_total=int(n_valid.sum()),
        study_rs=r_valid,
        study_ses=se_values,
        study_weights=normalized_weights,
        study_ns=n_valid,
    )


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
    r_array, n_array, valid_mask = _validate_inputs(r_values, n_values)
    
    n_valid_studies = valid_mask.sum()
    if n_valid_studies < 1:
        return _empty_meta_result()
    
    r_valid = r_array[valid_mask]
    n_valid = n_array[valid_mask]
    
    z_values = fisher_z(r_valid)
    se_values = correlation_se(n_valid)
    weights = 1.0 / (se_values ** 2)
    
    sum_weights = np.sum(weights)
    z_pooled = np.sum(weights * z_values) / sum_weights
    se_pooled = 1.0 / np.sqrt(sum_weights)
    
    return _build_meta_result(
        z_pooled=z_pooled,
        se_pooled=se_pooled,
        ci_level=ci_level,
        z_values=z_values,
        weights=weights,
        r_valid=r_valid,
        se_values=se_values,
        n_valid=n_valid,
        n_studies=n_valid_studies,
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
    r_array, n_array, valid_mask = _validate_inputs(r_values, n_values)
    
    n_valid_studies = valid_mask.sum()
    if n_valid_studies < 2:
        return fixed_effects_meta(r_values, n_values, ci_level)
    
    r_valid = r_array[valid_mask]
    n_valid = n_array[valid_mask]
    
    z_values = fisher_z(r_valid)
    se_values = correlation_se(n_valid)
    within_study_variance = se_values ** 2
    
    fixed_effects_weights = 1.0 / within_study_variance
    sum_fe_weights = np.sum(fixed_effects_weights)
    z_fixed_effects = np.sum(fixed_effects_weights * z_values) / sum_fe_weights
    
    q_stat, q_pvalue, i_squared, tau_squared = compute_heterogeneity(
        z_values, fixed_effects_weights, z_fixed_effects
    )
    
    total_variance = within_study_variance + tau_squared
    random_effects_weights = 1.0 / total_variance
    
    sum_re_weights = np.sum(random_effects_weights)
    z_pooled = np.sum(random_effects_weights * z_values) / sum_re_weights
    se_pooled = 1.0 / np.sqrt(sum_re_weights)
    
    return _build_meta_result(
        z_pooled=z_pooled,
        se_pooled=se_pooled,
        ci_level=ci_level,
        z_values=z_values,
        weights=random_effects_weights,
        r_valid=r_valid,
        se_values=se_values,
        n_valid=n_valid,
        n_studies=n_valid_studies,
    )


def _leave_one_out_analysis(
    r_values: np.ndarray,
    n_values: np.ndarray,
    meta_function,
) -> List[Tuple[int, MetaAnalysisResult]]:
    """Perform leave-one-out analysis with specified meta-analysis function."""
    n_studies = len(r_values)
    results = []
    
    for excluded_index in range(n_studies):
        mask = np.ones(n_studies, dtype=bool)
        mask[excluded_index] = False
        result = meta_function(r_values[mask], n_values[mask])
        results.append((excluded_index, result))
    
    return results


def leave_one_out_meta_fixed(
    r_values: np.ndarray,
    n_values: np.ndarray,
) -> List[Tuple[int, MetaAnalysisResult]]:
    """
    Leave-one-out sensitivity analysis using fixed-effects model.
    
    Returns list of (excluded_index, meta_result) tuples.
    """
    r_array = np.asarray(r_values)
    n_array = np.asarray(n_values)
    return _leave_one_out_analysis(r_array, n_array, fixed_effects_meta)


def leave_one_out_meta_random(
    r_values: np.ndarray,
    n_values: np.ndarray,
) -> List[Tuple[int, MetaAnalysisResult]]:
    """
    Leave-one-out sensitivity analysis using random-effects model.
    
    Returns list of (excluded_index, meta_result) tuples.
    """
    r_array = np.asarray(r_values)
    n_array = np.asarray(n_values)
    return _leave_one_out_analysis(r_array, n_array, random_effects_meta)


def leave_one_out_meta(
    r_values: np.ndarray,
    n_values: np.ndarray,
    use_random_effects: bool = True,
) -> List[Tuple[int, MetaAnalysisResult]]:
    """
    Leave-one-out sensitivity analysis.
    
    Deprecated: Use leave_one_out_meta_fixed or leave_one_out_meta_random instead.
    
    Returns list of (excluded_index, meta_result) tuples.
    """
    if use_random_effects:
        return leave_one_out_meta_random(r_values, n_values)
    return leave_one_out_meta_fixed(r_values, n_values)


def _empty_meta_result() -> MetaAnalysisResult:
    """Return empty meta-analysis result."""
    empty_array = np.array([])
    nan_value = np.nan
    return MetaAnalysisResult(
        r_pooled=nan_value,
        se_pooled=nan_value,
        ci_low=nan_value,
        ci_high=nan_value,
        z_score=nan_value,
        p_value=nan_value,
        q_statistic=nan_value,
        q_pvalue=nan_value,
        i_squared=nan_value,
        tau_squared=nan_value,
        n_studies=0,
        n_total=0,
        study_rs=empty_array,
        study_ses=empty_array,
        study_weights=empty_array,
        study_ns=empty_array,
    )


# Import ensure_bootstrap_ci from bootstrap module to avoid duplication
from .bootstrap import ensure_bootstrap_ci


def bayes_factor_correlation(
    r: float,
    n: int,
    prior_scale: float = _DEFAULT_PRIOR_SCALE,
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
    if np.isnan(r) or n < _MIN_SAMPLE_SIZE:
        return np.nan
    
    r_clipped = np.clip(r, _MIN_CORRELATION, _MAX_CORRELATION)
    r_squared = r_clipped ** 2
    
    degrees_of_freedom = n - 2
    log_bf = (
        betaln(0.5, degrees_of_freedom / 2)
        - betaln(0.5, 0.5)
        + ((n - 1) / 2) * np.log(1 - r_squared)
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
    if np.isnan(r) or n < _MIN_SAMPLE_SIZE:
        return np.nan, np.nan, False
    
    r_clipped = np.clip(r, _MIN_CORRELATION, _MAX_CORRELATION)
    z = np.arctanh(r_clipped)
    se = 1.0 / np.sqrt(n - 3)
    
    z_lower_bound = np.arctanh(-equiv_bound)
    z_upper_bound = np.arctanh(equiv_bound)
    
    t_statistic_lower = (z - z_lower_bound) / se
    p_lower = 1 - stats.norm.cdf(t_statistic_lower)
    
    t_statistic_upper = (z - z_upper_bound) / se
    p_upper = stats.norm.cdf(t_statistic_upper)
    
    max_p_value = max(p_lower, p_upper)
    equivalent = max_p_value < alpha
    
    return float(p_lower), float(p_upper), equivalent
