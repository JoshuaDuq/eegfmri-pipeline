"""
Moderation Analysis
===================

Statistical moderation analysis to test whether the relationship between
X and Y depends on a third variable (moderator W).

Moderation Model:
    Y = b0 + b1*X + b2*W + b3*(X*W) + e
    
    b3 is the interaction term - if significant, W moderates the X→Y relationship.
    
Simple Slopes Analysis:
    Effect of X on Y at different levels of W:
    - Low W (mean - 1 SD)
    - Mean W
    - High W (mean + 1 SD)
    
Johnson-Neyman Interval:
    The range of W values where the effect of X on Y is significant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

from ._regression_utils import _ols_regression


# Numerical constants
MIN_SAMPLE_SIZE_FOR_MODERATION = 15
DEFAULT_ALPHA = 0.05
SIMPLE_SLOPE_SD_OFFSET = 1.0
PARALLEL_THRESHOLD_PERM = 100
MIN_PERM_SUCCESSES = 100


@dataclass
class ModerationResult:
    """Container for moderation analysis results."""
    
    # Main effects
    b0: float = np.nan           # Intercept
    b1: float = np.nan           # X main effect
    b2: float = np.nan           # W (moderator) main effect
    b3: float = np.nan           # X*W interaction
    
    # Standard errors
    se_b1: float = np.nan
    se_b2: float = np.nan
    se_b3: float = np.nan
    
    # P-values
    p_b1: float = np.nan
    p_b2: float = np.nan
    p_b3: float = np.nan         # Key: is moderation significant?
    
    # Simple slopes at W levels
    slope_low_w: float = np.nan  # Effect at W = mean - 1SD
    slope_mean_w: float = np.nan # Effect at W = mean
    slope_high_w: float = np.nan # Effect at W = mean + 1SD
    
    p_slope_low: float = np.nan
    p_slope_mean: float = np.nan
    p_slope_high: float = np.nan
    
    # Johnson-Neyman interval (if applicable)
    jn_low: float = np.nan       # Lower bound of significance region
    jn_high: float = np.nan      # Upper bound of significance region
    jn_type: str = "none"        # "inside", "outside", "always", "never"
    
    # Model fit
    r_squared: float = np.nan
    r_squared_change: float = np.nan  # R² change due to interaction
    f_interaction: float = np.nan
    p_f_interaction: float = np.nan
    
    # Sample info
    n: int = 0
    
    # Labels
    x_label: str = "X"
    w_label: str = "W"
    y_label: str = "Y"
    
    # Permutation info
    p_b3_perm: float = np.nan  # Permutation p-value for interaction
    n_permutations: int = 0
    
    def is_significant_moderation(self, alpha: float = 0.05) -> bool:
        """Check if the interaction term is significant."""
        return np.isfinite(self.p_b3) and self.p_b3 < alpha
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary."""
        return {
            "b1_x_effect": self.b1,
            "b2_w_effect": self.b2,
            "b3_interaction": self.b3,
            "p_interaction": self.p_b3,
            "p_interaction_perm": self.p_b3_perm,
            "slope_low_w": self.slope_low_w,
            "slope_mean_w": self.slope_mean_w,
            "slope_high_w": self.slope_high_w,
            "r_squared": self.r_squared,
            "r_squared_change": self.r_squared_change,
            "jn_interval": (self.jn_low, self.jn_high),
            "jn_type": self.jn_type,
            "n_permutations": self.n_permutations,
            "n": self.n,
            "significant": self.is_significant_moderation(),
        }


###################################################################
# Core Computation
###################################################################


def compute_moderation_effect(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    center_predictors: bool = True,
) -> ModerationResult:
    """Compute moderation analysis with interaction term.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor variable (e.g., neural feature)
    W : np.ndarray
        Moderator variable (e.g., connectivity strength)
    Y : np.ndarray
        Outcome variable (e.g., pain rating)
    center_predictors : bool
        Whether to mean-center X and W (recommended for interpretation)
        
    Returns
    -------
    ModerationResult
        Full moderation analysis results
    """
    valid_mask = np.isfinite(X) & np.isfinite(W) & np.isfinite(Y)
    n_valid = np.sum(valid_mask)
    
    result = ModerationResult(n=n_valid)
    
    if n_valid < MIN_SAMPLE_SIZE_FOR_MODERATION:
        return result
    
    X_clean = X[valid_mask].copy()
    W_clean = W[valid_mask].copy()
    Y_clean = Y[valid_mask].copy()
    
    X_mean = np.mean(X_clean)
    W_mean = np.mean(W_clean)
    W_std = np.std(W_clean)
    
    if center_predictors:
        X_centered = X_clean - X_mean
        W_centered = W_clean - W_mean
    else:
        X_centered = X_clean
        W_centered = W_clean
    
    interaction_term = X_centered * W_centered
    
    design_matrix_full = np.column_stack([
        np.ones(n_valid), 
        X_centered, 
        W_centered, 
        interaction_term
    ])
    coefficients_full, se_full, _, r2_full = _ols_regression(
        Y_clean, design_matrix_full, compute_r2=True
    )
    
    design_matrix_reduced = np.column_stack([
        np.ones(n_valid), 
        X_centered, 
        W_centered
    ])
    _, _, _, r2_reduced = _ols_regression(Y_clean, design_matrix_reduced, compute_r2=True)
    
    result.b0 = coefficients_full[0]
    result.b1 = coefficients_full[1]
    result.b2 = coefficients_full[2]
    result.b3 = coefficients_full[3]
    
    result.se_b1 = se_full[1]
    result.se_b2 = se_full[2]
    result.se_b3 = se_full[3]
    
    degrees_of_freedom = n_valid - 4
    _compute_p_values(result, degrees_of_freedom)
    
    result.r_squared = r2_full
    result.r_squared_change = (
        r2_full - r2_reduced if np.isfinite(r2_reduced) else np.nan
    )
    
    _compute_interaction_f_test(result, r2_full, r2_reduced, n_valid)
    
    result = _compute_simple_slopes(result, W_std)
    result = _compute_johnson_neyman(result, n_valid, W_mean, W_std)
    
    return result


def _compute_p_values(result: ModerationResult, degrees_of_freedom: int) -> None:
    """Compute p-values for main effects and interaction term."""
    coefficient_se_pairs = [
        (result.b1, result.se_b1, 'p_b1'),
        (result.b2, result.se_b2, 'p_b2'),
        (result.b3, result.se_b3, 'p_b3'),
    ]
    
    for coefficient, standard_error, p_attr_name in coefficient_se_pairs:
        if not (np.isfinite(coefficient) and np.isfinite(standard_error) and standard_error > 0):
            continue
        
        t_statistic = coefficient / standard_error
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), degrees_of_freedom))
        setattr(result, p_attr_name, p_value)


def _compute_interaction_f_test(
    result: ModerationResult,
    r2_full: float,
    r2_reduced: float,
    n_samples: int,
) -> None:
    """Compute F-test for interaction term significance."""
    if not (np.isfinite(r2_full) and np.isfinite(r2_reduced)):
        return
    
    degrees_of_freedom_interaction = 1
    degrees_of_freedom_residual = n_samples - 4
    
    numerator = (r2_full - r2_reduced) / degrees_of_freedom_interaction
    denominator = (1 - r2_full) / degrees_of_freedom_residual
    
    if denominator > 0:
        result.f_interaction = numerator / denominator
        result.p_f_interaction = 1 - stats.f.cdf(
            result.f_interaction, 
            degrees_of_freedom_interaction, 
            degrees_of_freedom_residual
        )


def _compute_simple_slopes(
    result: ModerationResult,
    W_std: float,
) -> ModerationResult:
    """Compute simple slopes at W = mean ± 1SD.
    
    Simple slopes represent the effect of X on Y at different levels of W.
    """
    degrees_of_freedom = result.n - 4
    
    w_levels = [
        (-SIMPLE_SLOPE_SD_OFFSET * W_std, 'slope_low_w', 'p_slope_low'),
        (0.0, 'slope_mean_w', 'p_slope_mean'),
        (SIMPLE_SLOPE_SD_OFFSET * W_std, 'slope_high_w', 'p_slope_high'),
    ]
    
    for w_value, slope_attr, p_attr in w_levels:
        conditional_slope = result.b1 + result.b3 * w_value
        
        # Variance of conditional slope: Var(b1 + b3*w) ≈ Var(b1) + w²*Var(b3)
        # Approximation assumes Cov(b1, b3) ≈ 0
        se_conditional_slope = np.sqrt(
            result.se_b1**2 + (w_value**2) * result.se_b3**2
        )
        
        if se_conditional_slope > 0 and np.isfinite(conditional_slope):
            t_statistic = conditional_slope / se_conditional_slope
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), degrees_of_freedom))
        else:
            p_value = np.nan
        
        setattr(result, slope_attr, conditional_slope)
        setattr(result, p_attr, p_value)
    
    return result


def _compute_johnson_neyman(
    result: ModerationResult,
    n_samples: int,
    W_mean: float,
    W_std: float,
) -> ModerationResult:
    """Compute Johnson-Neyman regions of significance.
    
    The Johnson-Neyman technique identifies the range of moderator values
    where the effect of X on Y is statistically significant.
    """
    if not (np.isfinite(result.b3) and np.isfinite(result.se_b3)):
        return result
    
    degrees_of_freedom = n_samples - 4
    t_critical = stats.t.ppf(1 - DEFAULT_ALPHA / 2, degrees_of_freedom)
    
    # Conditional effect: slope(W) = b1 + b3*W
    # Standard error: SE(slope) ≈ sqrt(se_b1² + W²*se_b3²)
    # Significance when: |b1 + b3*W| / SE(slope) = t_critical
    # This yields a quadratic equation in W: a*W² + b*W + c = 0
    
    quadratic_coefficient = result.b3**2 - t_critical**2 * result.se_b3**2
    linear_coefficient = 2 * result.b1 * result.b3
    constant_coefficient = result.b1**2 - t_critical**2 * result.se_b1**2
    
    discriminant = linear_coefficient**2 - 4 * quadratic_coefficient * constant_coefficient
    
    if discriminant < 0:
        # No real roots: effect is either always or never significant
        t_at_mean = abs(result.b1) / result.se_b1 if result.se_b1 > 0 else 0.0
        result.jn_type = "always" if t_at_mean > t_critical else "never"
    elif abs(quadratic_coefficient) < 1e-10:
        # Linear case: interaction effect is negligible
        result.jn_type = "never"
    else:
        # Two real roots
        sqrt_discriminant = np.sqrt(discriminant)
        root_lower = (-linear_coefficient - sqrt_discriminant) / (2 * quadratic_coefficient)
        root_upper = (-linear_coefficient + sqrt_discriminant) / (2 * quadratic_coefficient)
        
        # Convert from centered scale back to original scale
        jn_lower_bound = root_lower + W_mean
        jn_upper_bound = root_upper + W_mean
        
        result.jn_low = min(jn_lower_bound, jn_upper_bound)
        result.jn_high = max(jn_lower_bound, jn_upper_bound)
        
        # Determine significance region type
        if quadratic_coefficient > 0:
            result.jn_type = "outside"  # Significant outside [jn_low, jn_high]
        else:
            result.jn_type = "inside"   # Significant inside [jn_low, jn_high]
    
    return result


###################################################################
# Permutation Testing
###################################################################


def _single_permutation_moderation(
    perm_seed: int,
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    center_predictors: bool,
) -> float:
    """Single permutation iteration for moderation null distribution.
    
    Shuffles Y to break all relationships while preserving the X-W
    covariance structure.
    
    Parameters
    ----------
    perm_seed : int
        Random seed for this permutation iteration
    X : np.ndarray
        Predictor variable values
    W : np.ndarray
        Moderator variable values
    Y : np.ndarray
        Outcome variable values (will be shuffled)
    center_predictors : bool
        Whether to center predictors
        
    Returns
    -------
    float
        Interaction coefficient (b3) from permuted sample, or np.nan if invalid
    """
    rng = np.random.default_rng(perm_seed)
    shuffle_indices = rng.permutation(len(Y))
    Y_shuffled = Y[shuffle_indices]
    
    result = compute_moderation_effect(X, W, Y_shuffled, center_predictors)
    return result.b3 if np.isfinite(result.b3) else np.nan


def permutation_moderation_pvalue(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    observed_b3: float,
    n_perm: int = 1000,
    center_predictors: bool = True,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
) -> float:
    """Compute permutation p-value for moderation interaction effect.
    
    Creates a null distribution by shuffling Y to break the outcome
    relationships, then computes the proportion of null effects that
    are as extreme as the observed effect (two-tailed).
    
    Parameters
    ----------
    X : np.ndarray
        Predictor variable
    W : np.ndarray
        Moderator variable
    Y : np.ndarray
        Outcome variable
    observed_b3 : float
        Observed interaction coefficient from the original data
    n_perm : int
        Number of permutation iterations
    center_predictors : bool
        Whether to center predictors
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
    
    if not np.isfinite(observed_b3) or n_perm <= 0:
        return np.nan
    
    if rng is None:
        rng = np.random.default_rng()
    
    valid_mask = np.isfinite(X) & np.isfinite(W) & np.isfinite(Y)
    sample_size = np.sum(valid_mask)
    
    if sample_size < MIN_SAMPLE_SIZE_FOR_MODERATION:
        return np.nan
    
    X_clean = X[valid_mask]
    W_clean = W[valid_mask]
    Y_clean = Y[valid_mask]
    
    if n_jobs == -1:
        n_jobs_actual = max(1, cpu_count() - 1)
    else:
        n_jobs_actual = n_jobs
    
    max_seed_value = 2**31
    base_seed = int(rng.integers(0, max_seed_value))
    
    should_parallelize = n_jobs_actual > 1 and n_perm > PARALLEL_THRESHOLD_PERM
    
    if should_parallelize:
        null_effects = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_single_permutation_moderation)(
                base_seed + i, X_clean, W_clean, Y_clean, center_predictors
            )
            for i in range(n_perm)
        )
    else:
        null_effects = [
            _single_permutation_moderation(
                base_seed + i, X_clean, W_clean, Y_clean, center_predictors
            )
            for i in range(n_perm)
        ]
    
    valid_null_effects = np.array([e for e in null_effects if np.isfinite(e)])
    
    if len(valid_null_effects) < MIN_PERM_SUCCESSES:
        return np.nan
    
    # Two-tailed p-value: count effects as extreme or more extreme than observed
    n_extreme = np.sum(np.abs(valid_null_effects) >= np.abs(observed_b3))
    p_perm = (n_extreme + 1) / (len(valid_null_effects) + 1)
    
    return float(p_perm)


def run_moderation_analysis(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 0,
    x_label: str = "Predictor",
    w_label: str = "Moderator",
    y_label: str = "Outcome",
    center_predictors: bool = True,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
) -> ModerationResult:
    """Run complete moderation analysis with optional permutation test.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor variable
    W : np.ndarray
        Moderator variable
    Y : np.ndarray
        Outcome variable
    n_perm : int
        Number of permutations for p-value (0 to skip permutation test)
    x_label, w_label, y_label : str
        Labels for variables
    center_predictors : bool
        Whether to mean-center X and W
    rng : np.random.Generator, optional
        Random generator for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)
        
    Returns
    -------
    ModerationResult
        Complete moderation analysis results including permutation p-value if requested
    """
    result = compute_moderation_effect(X, W, Y, center_predictors=center_predictors)
    result.x_label = x_label
    result.w_label = w_label
    result.y_label = y_label
    
    # Permutation test for interaction effect
    if n_perm > 0 and np.isfinite(result.b3):
        result.p_b3_perm = permutation_moderation_pvalue(
            X, W, Y, result.b3, n_perm=n_perm,
            center_predictors=center_predictors, rng=rng, n_jobs=n_jobs
        )
        result.n_permutations = n_perm
    
    return result


__all__ = [
    "ModerationResult",
    "compute_moderation_effect",
    "permutation_moderation_pvalue",
    "run_moderation_analysis",
]
