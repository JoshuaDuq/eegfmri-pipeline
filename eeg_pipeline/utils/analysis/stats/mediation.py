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
    p_ab_perm: float = np.nan  # Permutation p-value for indirect effect
    ci_ab_low: float = np.nan  # Bootstrap CI
    ci_ab_high: float = np.nan
    
    # Effect proportions
    proportion_mediated: float = np.nan  # ab / c
    
    # Permutation info
    n_permutations: int = 0
    
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
            "p_ab_perm": self.p_ab_perm,
            "ci_ab_low": self.ci_ab_low,
            "ci_ab_high": self.ci_ab_high,
            "proportion_mediated": self.proportion_mediated,
            "n_permutations": self.n_permutations,
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
        Independent variable (predictor)
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
    groups: Optional[np.ndarray] = None,
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
    if groups is None:
        bootstrap_indices = rng.integers(0, sample_size, size=sample_size)
    else:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != sample_size:
            return np.nan
        sampled_chunks = []
        for group in np.unique(groups_arr):
            group_idx = np.where(groups_arr == group)[0]
            if group_idx.size == 0:
                continue
            sampled_chunks.append(rng.choice(group_idx, size=group_idx.size, replace=True))
        if not sampled_chunks:
            return np.nan
        bootstrap_indices = np.concatenate(sampled_chunks)
        rng.shuffle(bootstrap_indices)
    
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
    groups: Optional[np.ndarray] = None,
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
    groups_valid = None
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != len(X):
            raise ValueError("groups length must match X length for grouped bootstrap.")
        groups_valid = groups_arr[finite_mask]
    
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
                base_seed + i, independent_var, mediator, dependent_var, sample_size, groups_valid
            )
            for i in range(n_boot)
        )
    else:
        bootstrap_results = [
            _single_bootstrap_mediation(
                base_seed + i, independent_var, mediator, dependent_var, sample_size, groups_valid
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


def _single_permutation_mediation(
    perm_seed: int,
    independent_var: np.ndarray,
    mediator: np.ndarray,
    dependent_var: np.ndarray,
    groups: Optional[np.ndarray] = None,
    scheme: str = "shuffle",
) -> float:
    """Single permutation iteration for mediation null distribution.
    
    Shuffles the mediator to break the M→Y relationship while preserving
    the X→M relationship structure, creating a null distribution for
    the indirect effect.
    
    Parameters
    ----------
    perm_seed : int
        Random seed for this permutation iteration
    independent_var : np.ndarray
        Independent variable values
    mediator : np.ndarray
        Mediator variable values (will be shuffled)
    dependent_var : np.ndarray
        Dependent variable values
        
    Returns
    -------
    float
        Indirect effect (a×b) from permuted sample, or np.nan if invalid
    """
    from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

    rng = np.random.default_rng(perm_seed)
    groups_arr = np.asarray(groups) if groups is not None else None
    try:
        shuffle_indices = permute_within_groups(
            len(mediator),
            rng,
            groups_arr,
            scheme=scheme,
            strict=True,
        )
    except ValueError:
        return np.nan
    m_shuffled = mediator[shuffle_indices]
    
    result = compute_mediation_paths(independent_var, m_shuffled, dependent_var)
    return result.ab if np.isfinite(result.ab) else np.nan


def permutation_indirect_effect(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    observed_ab: float,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
    groups: Optional[np.ndarray] = None,
    scheme: str = "shuffle",
) -> float:
    """Compute permutation p-value for indirect effect (a×b).
    
    Creates a null distribution by shuffling the mediator to break the
    M→Y relationship, then computes the proportion of null effects that
    are as extreme as the observed effect (two-tailed).
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable
    M : np.ndarray
        Mediator variable
    Y : np.ndarray
        Dependent variable
    observed_ab : float
        Observed indirect effect from the original data
    n_perm : int
        Number of permutation iterations
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs minus one)
        
    Returns
    -------
    float
        Two-tailed permutation p-value
    """
    from joblib import Parallel, delayed, cpu_count
    
    if not np.isfinite(observed_ab) or n_perm <= 0:
        return np.nan
    
    if rng is None:
        rng = np.random.default_rng()
    
    finite_mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    sample_size = np.sum(finite_mask)
    
    if sample_size < MIN_SAMPLE_SIZE:
        return np.nan
    
    independent_var = X[finite_mask]
    mediator = M[finite_mask]
    dependent_var = Y[finite_mask]
    groups_valid = None
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != len(X):
            raise ValueError("groups length must match X length for grouped permutation.")
        groups_valid = groups_arr[finite_mask]
    
    if n_jobs == -1:
        n_jobs_actual = max(1, cpu_count() - 1)
    else:
        n_jobs_actual = n_jobs
    
    max_seed_value = 2**31
    base_seed = int(rng.integers(0, max_seed_value))
    
    should_parallelize = n_jobs_actual > 1 and n_perm > PARALLEL_THRESHOLD_BOOTSTRAP
    
    if should_parallelize:
        null_effects = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_single_permutation_mediation)(
                base_seed + i, independent_var, mediator, dependent_var, groups_valid, scheme
            )
            for i in range(n_perm)
        )
    else:
        null_effects = [
            _single_permutation_mediation(
                base_seed + i, independent_var, mediator, dependent_var, groups_valid, scheme
            )
            for i in range(n_perm)
        ]
    
    valid_null_effects = np.array([e for e in null_effects if np.isfinite(e)])
    
    if len(valid_null_effects) < MIN_BOOTSTRAP_SUCCESSES:
        return np.nan
    
    # Two-tailed p-value: count effects as extreme or more extreme than observed
    n_extreme = np.sum(np.abs(valid_null_effects) >= np.abs(observed_ab))
    p_perm = (n_extreme + 1) / (len(valid_null_effects) + 1)
    
    return float(p_perm)


def run_full_mediation_analysis(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 5000,
    n_perm: int = 0,
    x_label: str = "Temperature",
    m_label: str = "Power",
    y_label: str = "Pain Rating",
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
    groups: Optional[np.ndarray] = None,
    permutation_scheme: str = "shuffle",
) -> MediationResult:
    """Run complete mediation analysis with bootstrap CIs and permutation test.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable (predictor)
    M : np.ndarray
        Mediator (neural feature)  
    Y : np.ndarray
        Dependent variable (pain rating)
    n_boot : int
        Number of bootstrap iterations for confidence intervals
    n_perm : int
        Number of permutations for p-value (0 to skip permutation test)
    x_label, m_label, y_label : str
        Labels for variables
    rng : np.random.Generator, optional
        Random generator for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)
        
    Returns
    -------
    MediationResult
        Complete mediation results including permutation p-value if requested
    """
    # Compute paths
    result = compute_mediation_paths(X, M, Y)
    result.x_label = x_label
    result.m_label = m_label
    result.y_label = y_label
    
    # Bootstrap CI for indirect effect
    ci_low, ci_high, _ = bootstrap_indirect_effect(
        X, M, Y, n_boot=n_boot, rng=rng, n_jobs=n_jobs, groups=groups
    )
    result.ci_ab_low = ci_low
    result.ci_ab_high = ci_high
    
    # Permutation test for indirect effect
    if n_perm > 0 and np.isfinite(result.ab):
        result.p_ab_perm = permutation_indirect_effect(
            X, M, Y, result.ab, n_perm=n_perm, rng=rng, n_jobs=n_jobs, groups=groups, scheme=permutation_scheme
        )
        result.n_permutations = n_perm
    
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
    n_perm: int,
    x_label: str,
    y_label: str,
    rng_seed: int,
    groups: Optional[np.ndarray] = None,
    permutation_scheme: str = "shuffle",
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
    n_perm : int
        Number of permutations for p-value (0 to skip)
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
        n_perm=n_perm,
        x_label=x_label,
        m_label=feature_name,
        y_label=y_label,
        rng=rng,
        n_jobs=1,  # Avoid nested parallelism
        groups=groups,
        permutation_scheme=permutation_scheme,
    )
    
    return result


def analyze_mediation_for_features(
    X: np.ndarray,
    feature_df: pd.DataFrame,
    Y: np.ndarray,
    n_boot: int = 2000,
    n_perm: int = 0,
    x_label: str = "Temperature",
    y_label: str = "Pain Rating",
    logger: Optional[logging.Logger] = None,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
    min_effect_size: float = 0.0,
    groups: Optional[np.ndarray] = None,
    permutation_scheme: str = "shuffle",
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
        Bootstrap iterations per feature for confidence intervals
    n_perm : int
        Permutation iterations per feature for p-values (0 to skip)
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
    min_effect_size : float
        Minimum absolute indirect effect (ab) or proportion mediated to include.
        Results below this threshold are filtered out. Default 0.0 (no filtering).
        
    Returns
    -------
    List[MediationResult]
        Results for features with sufficient data for analysis and effect size
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
    groups_array = None
    if groups is not None:
        groups_array = np.asarray(groups)
        if len(groups_array) != len(X):
            raise ValueError("groups length must match X length for mediation analysis.")
    
    should_parallelize = n_jobs_actual > 1 and n_features > PARALLEL_THRESHOLD_FEATURES
    
    if should_parallelize:
        raw_results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_analyze_single_feature_mediation)(
                col_name,
                feature_df[col_name].values,
                X,
                Y,
                n_boot,
                n_perm,
                x_label,
                y_label,
                base_seed + i,
                groups_array,
                permutation_scheme,
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
                n_perm,
                x_label,
                y_label,
                base_seed + i,
                groups_array,
                permutation_scheme,
            )
            for i, col_name in enumerate(feature_df.columns)
        ]
    
    valid_results = [result for result in raw_results if result is not None]
    
    # Filter by minimum effect size if specified
    if min_effect_size > 0.0:
        filtered_results = []
        for result in valid_results:
            abs_indirect = abs(result.ab) if np.isfinite(result.ab) else 0.0
            abs_proportion = abs(result.proportion_mediated) if np.isfinite(result.proportion_mediated) else 0.0
            if abs_indirect >= min_effect_size or abs_proportion >= min_effect_size:
                filtered_results.append(result)
        valid_results = filtered_results
    
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
    "permutation_indirect_effect",
    "run_full_mediation_analysis",
    "analyze_mediation_for_features",
]
