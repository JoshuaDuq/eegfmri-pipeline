"""
Mediation Analysis
==================

Statistical mediation analysis for the Temperature → Neural Feature → Pain pathway.

Implements Baron & Kenny (1986) approach with modern extensions:
- Bootstrap confidence intervals for indirect effect
- Sobel test for indirect effect significance
- Proportion mediated

Theoretical Framework:
    X = Temperature (stimulus intensity)
    M = Neural feature (e.g., alpha power)
    Y = Pain rating (subjective report)

Path Model:
    X → M (path a)
    M → Y controlling for X (path b)  
    X → Y (path c, total effect)
    X → Y controlling for M (path c', direct effect)
    
    Indirect effect = a × b
    Total effect c = c' + ab
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._regression_utils import _ols_regression


# Constants
MIN_SAMPLE_SIZE = 10
MIN_SAMPLE_SIZE_FEATURE = 20
MIN_BOOTSTRAP_SUCCESSES = 100
MIN_TOTAL_EFFECT_THRESHOLD = 1e-10
PARALLEL_THRESHOLD_BOOTSTRAP = 100
PARALLEL_THRESHOLD_FEATURES = 5
DEFAULT_ALPHA = 0.05


@dataclass
class MediationResult:
    """Container for mediation analysis results."""
    
    # Path coefficients
    a: float = np.nan          # X → M
    se_a: float = np.nan
    p_a: float = np.nan
    
    b: float = np.nan          # M → Y | X
    se_b: float = np.nan
    p_b: float = np.nan
    
    c: float = np.nan          # X → Y (total)
    se_c: float = np.nan
    p_c: float = np.nan
    
    c_prime: float = np.nan    # X → Y | M (direct)
    se_c_prime: float = np.nan
    p_c_prime: float = np.nan
    
    # Indirect effect
    ab: float = np.nan         # a × b
    se_ab: float = np.nan      # Sobel SE
    p_ab: float = np.nan       # Sobel test p-value
    ci_ab_low: float = np.nan  # Bootstrap CI
    ci_ab_high: float = np.nan
    
    # Effect proportions
    proportion_mediated: float = np.nan  # ab / c
    
    # Sample info
    n: int = 0
    
    # Variable labels
    x_label: str = "X"
    m_label: str = "M"
    y_label: str = "Y"
    
    def is_significant_mediation(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """Check if there is significant mediation.
        
        Requires both paths (a and b) to be significant and the indirect
        effect confidence interval to exclude zero.
        """
        both_paths_significant = self.p_a < alpha and self.p_b < alpha
        indirect_effect_positive = self.ci_ab_low > 0
        indirect_effect_negative = self.ci_ab_high < 0
        ci_excludes_zero = indirect_effect_positive or indirect_effect_negative
        
        return both_paths_significant and ci_excludes_zero
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary."""
        return {
            "path_a": self.a,
            "path_b": self.b,
            "path_c_total": self.c,
            "path_c_prime_direct": self.c_prime,
            "indirect_effect_ab": self.ab,
            "sobel_p": self.p_ab,
            "ci_ab_low": self.ci_ab_low,
            "ci_ab_high": self.ci_ab_high,
            "proportion_mediated": self.proportion_mediated,
            "n": self.n,
            "significant": self.is_significant_mediation(),
        }


###################################################################
# Core Computation
###################################################################


def _compute_t_statistic_p_value(
    coefficient: float,
    standard_error: float,
    degrees_of_freedom: int,
) -> float:
    """Compute two-tailed t-test p-value for a regression coefficient.
    
    Parameters
    ----------
    coefficient : float
        Regression coefficient
    standard_error : float
        Standard error of coefficient
    degrees_of_freedom : int
        Degrees of freedom for t-distribution
        
    Returns
    -------
    float
        Two-tailed p-value, or np.nan if invalid
    """
    if standard_error <= 0 or not np.isfinite(coefficient):
        return np.nan
    
    t_statistic = coefficient / standard_error
    if not np.isfinite(t_statistic):
        return np.nan
    
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=degrees_of_freedom))
    return p_value if np.isfinite(p_value) else np.nan




def compute_mediation_paths(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
) -> MediationResult:
    """Compute mediation path coefficients using OLS.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable (temperature)
    M : np.ndarray
        Mediator (neural feature)
    Y : np.ndarray
        Dependent variable (pain rating)
        
    Returns
    -------
    MediationResult
        Path coefficients and statistics
    """
    if not isinstance(X, np.ndarray) or not isinstance(M, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("All inputs must be numpy arrays")
    
    if X.shape != M.shape or X.shape != Y.shape:
        raise ValueError("X, M, and Y must have the same shape")
    
    finite_mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    sample_size = np.sum(finite_mask)
    
    result = MediationResult(n=sample_size)
    
    if sample_size < MIN_SAMPLE_SIZE:
        return result
    
    independent_var = X[finite_mask]
    mediator = M[finite_mask]
    dependent_var = Y[finite_mask]
    
    # Path a: X → M
    design_matrix_x = np.column_stack([np.ones(sample_size), independent_var])
    coefficients_a, standard_errors_a, _, _ = _ols_regression(mediator, design_matrix_x)
    result.a = coefficients_a[1]
    result.se_a = standard_errors_a[1]
    result.p_a = _compute_t_statistic_p_value(result.a, result.se_a, sample_size - 2)
    
    # Path c: X → Y (total effect)
    coefficients_c, standard_errors_c, _, _ = _ols_regression(dependent_var, design_matrix_x)
    result.c = coefficients_c[1]
    result.se_c = standard_errors_c[1]
    result.p_c = _compute_t_statistic_p_value(result.c, result.se_c, sample_size - 2)
    
    # Paths b and c': Y ~ X + M
    design_matrix_xm = np.column_stack([np.ones(sample_size), independent_var, mediator])
    coefficients_bc, standard_errors_bc, _, _ = _ols_regression(dependent_var, design_matrix_xm)
    
    result.c_prime = coefficients_bc[1]
    result.se_c_prime = standard_errors_bc[1]
    result.p_c_prime = _compute_t_statistic_p_value(
        result.c_prime, result.se_c_prime, sample_size - 3
    )
    
    result.b = coefficients_bc[2]
    result.se_b = standard_errors_bc[2]
    result.p_b = _compute_t_statistic_p_value(result.b, result.se_b, sample_size - 3)
    
    # Indirect effect
    result.ab = result.a * result.b
    
    # Sobel test
    if np.isfinite(result.a) and np.isfinite(result.b):
        variance_ab = (
            result.a**2 * result.se_b**2 + result.b**2 * result.se_a**2
        )
        result.se_ab = np.sqrt(variance_ab)
        
        if result.se_ab > 0 and np.isfinite(result.ab):
            z_sobel = result.ab / result.se_ab
            if np.isfinite(z_sobel):
                result.p_ab = 2 * (1 - stats.norm.cdf(np.abs(z_sobel)))
    
    # Proportion mediated
    if np.isfinite(result.c) and abs(result.c) > MIN_TOTAL_EFFECT_THRESHOLD:
        result.proportion_mediated = result.ab / result.c
    
    return result


def _single_bootstrap_mediation(
    bootstrap_seed: int,
    independent_var: np.ndarray,
    mediator: np.ndarray,
    dependent_var: np.ndarray,
    sample_size: int,
) -> float:
    """Single bootstrap iteration for mediation analysis.
    
    Parameters
    ----------
    bootstrap_seed : int
        Random seed for this bootstrap iteration
    independent_var : np.ndarray
        Independent variable values
    mediator : np.ndarray
        Mediator variable values
    dependent_var : np.ndarray
        Dependent variable values
    sample_size : int
        Number of observations to resample
        
    Returns
    -------
    float
        Indirect effect (a×b) from bootstrap sample, or np.nan if invalid
    """
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(0, sample_size, size=sample_size)
    
    x_boot = independent_var[bootstrap_indices]
    m_boot = mediator[bootstrap_indices]
    y_boot = dependent_var[bootstrap_indices]
    
    result = compute_mediation_paths(x_boot, m_boot, y_boot)
    return result.ab if np.isfinite(result.ab) else np.nan


def bootstrap_indirect_effect(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 5000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap confidence interval for indirect effect (a×b).
    
    Uses percentile method which is more robust than Sobel for small samples.
    Parallel processing with loky backend for speed.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable
    M : np.ndarray
        Mediator variable
    Y : np.ndarray
        Dependent variable
    n_boot : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs minus one)
        
    Returns
    -------
    ci_low : float
        Lower bound of confidence interval
    ci_high : float
        Upper bound of confidence interval
    boot_distribution : np.ndarray
        Array of bootstrap indirect effect values
    """
    from joblib import Parallel, delayed, cpu_count
    
    if rng is None:
        rng = np.random.default_rng()
    
    finite_mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    sample_size = np.sum(finite_mask)
    
    if sample_size < MIN_SAMPLE_SIZE:
        return np.nan, np.nan, np.array([])
    
    independent_var = X[finite_mask]
    mediator = M[finite_mask]
    dependent_var = Y[finite_mask]
    
    if n_jobs == -1:
        n_jobs_actual = max(1, cpu_count() - 1)
    else:
        n_jobs_actual = n_jobs
    
    max_seed_value = 2**31
    base_seed = int(rng.integers(0, max_seed_value))
    
    should_parallelize = n_jobs_actual > 1 and n_boot > PARALLEL_THRESHOLD_BOOTSTRAP
    
    if should_parallelize:
        bootstrap_results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_single_bootstrap_mediation)(
                base_seed + i, independent_var, mediator, dependent_var, sample_size
            )
            for i in range(n_boot)
        )
    else:
        bootstrap_results = [
            _single_bootstrap_mediation(
                base_seed + i, independent_var, mediator, dependent_var, sample_size
            )
            for i in range(n_boot)
        ]
    
    valid_indirect_effects = [
        effect for effect in bootstrap_results if np.isfinite(effect)
    ]
    
    if len(valid_indirect_effects) < MIN_BOOTSTRAP_SUCCESSES:
        return np.nan, np.nan, np.array([])
    
    indirect_effects_array = np.array(valid_indirect_effects)
    alpha_level = 1 - ci_level
    percentile_lower = 100 * alpha_level / 2
    percentile_upper = 100 * (1 - alpha_level / 2)
    
    ci_low = np.percentile(indirect_effects_array, percentile_lower)
    ci_high = np.percentile(indirect_effects_array, percentile_upper)
    
    return float(ci_low), float(ci_high), indirect_effects_array


def run_full_mediation_analysis(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 5000,
    x_label: str = "Temperature",
    m_label: str = "Power",
    y_label: str = "Pain Rating",
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
) -> MediationResult:
    """Run complete mediation analysis with bootstrap CIs.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable (temperature)
    M : np.ndarray
        Mediator (neural feature)  
    Y : np.ndarray
        Dependent variable (pain rating)
    n_boot : int
        Number of bootstrap iterations
    x_label, m_label, y_label : str
        Labels for variables
    rng : np.random.Generator, optional
        Random generator for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)
        
    Returns
    -------
    MediationResult
        Complete mediation results
    """
    # Compute paths
    result = compute_mediation_paths(X, M, Y)
    result.x_label = x_label
    result.m_label = m_label
    result.y_label = y_label
    
    # Bootstrap CI for indirect effect
    ci_low, ci_high, _ = bootstrap_indirect_effect(
        X, M, Y, n_boot=n_boot, rng=rng, n_jobs=n_jobs
    )
    result.ci_ab_low = ci_low
    result.ci_ab_high = ci_high
    
    return result


###################################################################
# Batch Processing
###################################################################


def _analyze_single_feature_mediation(
    feature_name: str,
    mediator_values: np.ndarray,
    independent_var: np.ndarray,
    dependent_var: np.ndarray,
    n_boot: int,
    x_label: str,
    y_label: str,
    rng_seed: int,
) -> Optional[MediationResult]:
    """Analyze mediation for a single feature.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature being analyzed
    mediator_values : np.ndarray
        Mediator variable values for this feature
    independent_var : np.ndarray
        Independent variable values
    dependent_var : np.ndarray
        Dependent variable values
    n_boot : int
        Number of bootstrap iterations
    x_label : str
        Label for independent variable
    y_label : str
        Label for dependent variable
    rng_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Optional[MediationResult]
        Mediation results if sufficient data, None otherwise
    """
    finite_mask = (
        np.isfinite(independent_var)
        & np.isfinite(mediator_values)
        & np.isfinite(dependent_var)
    )
    valid_sample_size = np.sum(finite_mask)
    
    if valid_sample_size < MIN_SAMPLE_SIZE_FEATURE:
        return None
    
    rng = np.random.default_rng(rng_seed)
    result = run_full_mediation_analysis(
        independent_var,
        mediator_values,
        dependent_var,
        n_boot=n_boot,
        x_label=x_label,
        m_label=feature_name,
        y_label=y_label,
        rng=rng,
        n_jobs=1,  # Avoid nested parallelism
    )
    
    return result


def analyze_mediation_for_features(
    X: np.ndarray,
    feature_df: pd.DataFrame,
    Y: np.ndarray,
    n_boot: int = 2000,
    x_label: str = "Temperature",
    y_label: str = "Pain Rating",
    logger: Optional[logging.Logger] = None,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
) -> List[MediationResult]:
    """Run mediation analysis for multiple neural features.
    
    Tests whether each feature mediates the X → Y relationship.
    Uses parallel processing with loky backend for speed.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable
    feature_df : pd.DataFrame
        DataFrame where each column is a potential mediator
    Y : np.ndarray
        Dependent variable
    n_boot : int
        Bootstrap iterations per feature
    x_label : str
        Label for independent variable
    y_label : str
        Label for dependent variable
    logger : logging.Logger, optional
        Logger instance for output messages
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs minus one)
        
    Returns
    -------
    List[MediationResult]
        Results for features with sufficient data for analysis
    """
    from joblib import Parallel, delayed, cpu_count
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if rng is None:
        rng = np.random.default_rng()
    
    if n_jobs == -1:
        n_jobs_actual = max(1, cpu_count() - 1)
    else:
        n_jobs_actual = n_jobs
    
    max_seed_value = 2**31
    base_seed = int(rng.integers(0, max_seed_value))
    n_features = len(feature_df.columns)
    
    should_parallelize = n_jobs_actual > 1 and n_features > PARALLEL_THRESHOLD_FEATURES
    
    if should_parallelize:
        raw_results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_analyze_single_feature_mediation)(
                col_name,
                feature_df[col_name].values,
                X,
                Y,
                n_boot,
                x_label,
                y_label,
                base_seed + i,
            )
            for i, col_name in enumerate(feature_df.columns)
        )
    else:
        raw_results = [
            _analyze_single_feature_mediation(
                col_name,
                feature_df[col_name].values,
                X,
                Y,
                n_boot,
                x_label,
                y_label,
                base_seed + i,
            )
            for i, col_name in enumerate(feature_df.columns)
        ]
    
    valid_results = [result for result in raw_results if result is not None]
    
    significant_count = sum(
        1 for result in valid_results if result.is_significant_mediation()
    )
    
    for result in valid_results:
        if result.is_significant_mediation():
            logger.info(
                f"Significant mediation: {result.m_label} "
                f"(ab={result.ab:.3f}, "
                f"CI=[{result.ci_ab_low:.3f}, {result.ci_ab_high:.3f}])"
            )
    
    logger.info(
        f"Obtained results for {len(valid_results)} mediators "
        f"({significant_count} significant) out of {n_features} tested"
    )
    
    return valid_results


__all__ = [
    "MediationResult",
    "compute_mediation_paths",
    "bootstrap_indirect_effect",
    "run_full_mediation_analysis",
    "analyze_mediation_for_features",
]
