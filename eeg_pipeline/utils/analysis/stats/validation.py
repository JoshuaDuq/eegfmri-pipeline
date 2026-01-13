"""
Data Validation
===============

Validation functions for EEG data integrity, statistical assumption checks,
permutation/randomization validation, and family-wise error control.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Tuple, Union, List, Dict, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_config_value, ensure_config, filter_finite_values


###################################################################
# Constants
###################################################################

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_RANDOM_SEED = 42
SHAPIRO_WILK_MAX_SAMPLES = 5000
MIN_SAMPLES_FOR_SHAPIRO = 3
MIN_SAMPLES_FOR_DAGOSTINO = 20
MIN_SAMPLES_FOR_VARIANCE_TEST = 2
MIN_SAMPLES_FOR_GROUP_COMPARISON = 3
MIN_PERMUTATIONS_FOR_ALPHA = 10
MAX_SKEW_FOR_NULL_DISTRIBUTION = 2.0
MAX_MONTE_CARLO_SE = 0.01
MIN_P_VALUE_FOR_MC_WARNING = 0.1
MIN_CORRELATION_FOR_POWER_CALC = 0.01
EFFECTIVELY_INFINITE_SAMPLE_SIZE = 999999


###################################################################
# Helper Functions
###################################################################


def _clean_finite_groups(*groups: np.ndarray) -> list[np.ndarray]:
    """Clean and filter groups, keeping only those with sufficient samples."""
    clean_groups = []
    for group in groups:
        cleaned = filter_finite_values(group, flatten=True)
        if len(cleaned) >= MIN_SAMPLES_FOR_VARIANCE_TEST:
            clean_groups.append(cleaned)
    return clean_groups


###################################################################
# Assumption Check Data Structures
###################################################################


@dataclass
class AssumptionCheckResult:
    """Result container for statistical assumption checks."""
    
    test_name: str
    passed: bool
    statistic: float
    p_value: float
    warning_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Aggregated validation report for statistical analyses."""
    
    normality_checks: List[AssumptionCheckResult] = field(default_factory=list)
    variance_checks: List[AssumptionCheckResult] = field(default_factory=list)
    permutation_checks: List[AssumptionCheckResult] = field(default_factory=list)
    fwer_checks: List[AssumptionCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        
    def set_flag(self, flag_name: str, value: bool) -> None:
        self.flags[flag_name] = value
    
    def has_violations(self) -> bool:
        all_checks = self.normality_checks + self.variance_checks + self.permutation_checks
        return any(not c.passed for c in all_checks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "normality_checks": [c.to_dict() for c in self.normality_checks],
            "variance_checks": [c.to_dict() for c in self.variance_checks],
            "permutation_checks": [c.to_dict() for c in self.permutation_checks],
            "fwer_checks": [c.to_dict() for c in self.fwer_checks],
            "warnings": self.warnings,
            "flags": self.flags,
            "has_violations": self.has_violations(),
        }
    
    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def validate_sample_size_for_correlation(
    n: int,
    target_r: float = 0.3,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
) -> AssumptionCheckResult:
    """Check if sample size is adequate for detecting target correlation.
    
    Uses Fisher z-transform power calculation.
    
    Parameters
    ----------
    n : int
        Actual sample size
    target_r : float
        Target/expected correlation effect size
    power : float
        Desired statistical power
    alpha : float
        Significance level
        
    Returns
    -------
    AssumptionCheckResult
        Result with passed=True if n >= required_n
    """
    if abs(target_r) < MIN_CORRELATION_FOR_POWER_CALC:
        required_n = EFFECTIVELY_INFINITE_SAMPLE_SIZE
    else:
        correlation_clipped = np.clip(target_r, -0.999, 0.999)
        z_correlation = np.arctanh(correlation_clipped)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        required_n = int(np.ceil(((z_alpha + z_beta) / z_correlation) ** 2 + 3))
    
    passed = n >= required_n
    warning_message = (
        "" if passed 
        else f"Underpowered: n={n}, need n>={required_n} for power={power} to detect r={target_r}"
    )
    
    return AssumptionCheckResult(
        test_name="Sample Size Adequacy",
        passed=passed,
        statistic=float(n),
        p_value=float(required_n),
        warning_message=warning_message,
        details={"target_r": target_r, "power": power, "alpha": alpha, "required_n": required_n},
    )


###################################################################
# Normality Checks
###################################################################


def check_normality_shapiro(
    data: np.ndarray,
    group_name: str = "data",
    alpha: float = DEFAULT_ALPHA,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check normality using Shapiro-Wilk test.
    
    Appropriate for n < 5000. For larger samples, use check_normality_dagostino.
    """
    cleaned_data = filter_finite_values(data, flatten=True)
    
    if len(cleaned_data) < MIN_SAMPLES_FOR_SHAPIRO:
        return AssumptionCheckResult(
            test_name="Shapiro-Wilk",
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message=f"{group_name}: Insufficient data for normality test (n={len(cleaned_data)})",
        )
    
    # Shapiro-Wilk limited to 5000 samples
    if len(cleaned_data) > SHAPIRO_WILK_MAX_SAMPLES:
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        cleaned_data = rng.choice(cleaned_data, size=SHAPIRO_WILK_MAX_SAMPLES, replace=False)
    
    statistic, p_value = stats.shapiro(cleaned_data)
    passed = p_value >= alpha
    
    warning_message = ""
    if not passed:
        warning_message = (
            f"{group_name}: Non-normal distribution "
            f"(Shapiro-Wilk W={statistic:.4f}, p={p_value:.4f})"
        )
        if logger:
            logger.warning(warning_message)
    
    return AssumptionCheckResult(
        test_name="Shapiro-Wilk",
        passed=passed,
        statistic=float(statistic),
        p_value=float(p_value),
        warning_message=warning_message,
        details={"n": len(cleaned_data), "group": group_name},
    )


def check_normality_dagostino(
    data: np.ndarray,
    group_name: str = "data",
    alpha: float = DEFAULT_ALPHA,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check normality using D'Agostino-Pearson test.
    
    Better for larger samples (n > 20).
    """
    cleaned_data = filter_finite_values(data, flatten=True)
    
    if len(cleaned_data) < MIN_SAMPLES_FOR_DAGOSTINO:
        return AssumptionCheckResult(
            test_name="D'Agostino-Pearson",
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message=(
                f"{group_name}: Insufficient data for D'Agostino test "
                f"(n={len(cleaned_data)}, need {MIN_SAMPLES_FOR_DAGOSTINO}+)"
            ),
        )
    
    statistic, p_value = stats.normaltest(cleaned_data)
    passed = p_value >= alpha
    
    warning_message = ""
    if not passed:
        warning_message = (
            f"{group_name}: Non-normal distribution "
            f"(D'Agostino K²={statistic:.4f}, p={p_value:.4f})"
        )
        if logger:
            logger.warning(warning_message)
    
    return AssumptionCheckResult(
        test_name="D'Agostino-Pearson",
        passed=passed,
        statistic=float(statistic),
        p_value=float(p_value),
        warning_message=warning_message,
        details={"n": len(cleaned_data), "group": group_name},
    )


def compute_qq_data(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute QQ plot data for normality assessment.
    
    Returns (theoretical_quantiles, sample_quantiles, slope, intercept).
    """
    cleaned_data = filter_finite_values(data, flatten=True)
    sample_quantiles = np.sort(cleaned_data)
    
    sample_size = len(sample_quantiles)
    quantile_positions = (np.arange(1, sample_size + 1) - 0.5) / sample_size
    theoretical_quantiles = stats.norm.ppf(quantile_positions)
    
    slope, intercept, _, _, _ = stats.linregress(theoretical_quantiles, sample_quantiles)
    
    return theoretical_quantiles, sample_quantiles, float(slope), float(intercept)


###################################################################
# Variance Homogeneity Checks
###################################################################


def _check_variance_homogeneity(
    *groups: np.ndarray,
    group_names: Optional[List[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    test_name: str = "Variance",
    test_function: Callable[..., Tuple[float, float]] = stats.levene,
    test_kwargs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Common implementation for variance homogeneity tests."""
    clean_groups = _clean_finite_groups(*groups)
    
    if len(clean_groups) < 2:
        return AssumptionCheckResult(
            test_name=test_name,
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message=f"Insufficient groups for {test_name} test",
        )
    
    kwargs = test_kwargs or {}
    statistic, p_value = test_function(*clean_groups, **kwargs)
    passed = p_value >= alpha
    
    default_names = [f"Group {i+1}" for i in range(len(clean_groups))]
    names = group_names or default_names
    variances = {
        name: float(np.var(group, ddof=1)) 
        for name, group in zip(names, clean_groups)
    }
    
    warning_message = ""
    if not passed:
        warning_message = (
            f"Heterogeneous variances ({test_name} statistic={statistic:.4f}, "
            f"p={p_value:.4f}): {variances}"
        )
        if logger:
            logger.warning(warning_message)
    
    return AssumptionCheckResult(
        test_name=test_name,
        passed=passed,
        statistic=float(statistic),
        p_value=float(p_value),
        warning_message=warning_message,
        details={"variances": variances, "n_groups": len(clean_groups)},
    )


def check_variance_levene(
    *groups: np.ndarray,
    group_names: Optional[List[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    center: str = "median",
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check homogeneity of variance using Levene's test.
    
    Parameters
    ----------
    *groups : arrays
        Two or more groups to compare
    group_names : list of str, optional
        Names for the groups
    alpha : float
        Significance threshold
    center : str
        'median' (default, robust) or 'mean'
    """
    return _check_variance_homogeneity(
        *groups,
        group_names=group_names,
        alpha=alpha,
        test_name="Levene",
        test_function=stats.levene,
        test_kwargs={"center": center},
        logger=logger,
    )


def check_variance_bartlett(
    *groups: np.ndarray,
    group_names: Optional[List[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check homogeneity of variance using Bartlett's test.
    
    More powerful than Levene but assumes normality.
    """
    return _check_variance_homogeneity(
        *groups,
        group_names=group_names,
        alpha=alpha,
        test_name="Bartlett",
        test_function=stats.bartlett,
        test_kwargs=None,
        logger=logger,
    )


###################################################################
# Permutation/Randomization Validation
###################################################################


def validate_permutation_distribution(
    null_distribution: np.ndarray,
    observed_statistic: float,
    n_permutations: int,
    test_name: str = "permutation",
    alpha: float = DEFAULT_ALPHA,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Validate permutation test assumptions and distribution quality.
    
    Checks:
    - Sufficient permutations for desired precision
    - Null distribution symmetry (for two-tailed tests)
    - No extreme outliers suggesting computational issues
    """
    cleaned_null = filter_finite_values(null_distribution, flatten=True)
    
    warnings = []
    passed = True
    details = {}
    
    # Check sufficient permutations
    min_permutations_required = int(1 / alpha) * MIN_PERMUTATIONS_FOR_ALPHA
    if n_permutations < min_permutations_required:
        warnings.append(
            f"Low permutation count ({n_permutations}) for α={alpha}; "
            f"recommend ≥{min_permutations_required}"
        )
        passed = False
    
    # Check null distribution properties
    null_mean = np.mean(cleaned_null)
    null_std = np.std(cleaned_null)
    min_samples_for_moments = 8
    has_sufficient_samples = len(cleaned_null) > min_samples_for_moments
    null_skew = stats.skew(cleaned_null) if has_sufficient_samples else 0.0
    null_kurtosis = stats.kurtosis(cleaned_null) if has_sufficient_samples else 0.0
    
    details["null_mean"] = float(null_mean)
    details["null_std"] = float(null_std)
    details["null_skew"] = float(null_skew)
    details["null_kurtosis"] = float(null_kurtosis)
    
    # Check for extreme skewness (may indicate issues)
    if abs(null_skew) > MAX_SKEW_FOR_NULL_DISTRIBUTION:
        warnings.append(f"Highly skewed null distribution (skew={null_skew:.2f})")
    
    # Compute permutation p-value
    observed_abs = np.abs(observed_statistic)
    n_extreme = np.sum(np.abs(cleaned_null) >= observed_abs)
    p_permutation = (n_extreme + 1) / (len(cleaned_null) + 1)
    details["p_permutation"] = float(p_permutation)
    details["n_extreme"] = int(n_extreme)
    
    # Check Monte Carlo error
    monte_carlo_se = np.sqrt(p_permutation * (1 - p_permutation) / n_permutations)
    details["mc_standard_error"] = float(monte_carlo_se)
    
    if monte_carlo_se > MAX_MONTE_CARLO_SE and p_permutation < MIN_P_VALUE_FOR_MC_WARNING:
        warnings.append(
            f"High Monte Carlo SE ({monte_carlo_se:.4f}); "
            f"increase permutations for stable p-value"
        )
    
    warning_message = "; ".join(warnings) if warnings else ""
    if warning_message and logger:
        logger.warning(f"{test_name}: {warning_message}")
    
    return AssumptionCheckResult(
        test_name=f"Permutation validation ({test_name})",
        passed=passed,
        statistic=float(observed_statistic),
        p_value=float(p_permutation),
        warning_message=warning_message,
        details=details,
    )


def check_randomization_balance(
    group_assignments: np.ndarray,
    expected_ratio: float = 0.5,
    tolerance: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check that randomization produced balanced groups.
    
    Parameters
    ----------
    group_assignments : array
        Binary array of group assignments (0/1)
    expected_ratio : float
        Expected proportion of group 1
    tolerance : float
        Acceptable deviation from expected ratio
    """
    cleaned_assignments = filter_finite_values(group_assignments, flatten=True)
    
    observed_ratio = np.mean(cleaned_assignments)
    deviation = abs(observed_ratio - expected_ratio)
    passed = deviation <= tolerance
    
    # Binomial test for significant imbalance
    total_samples = len(cleaned_assignments)
    n_group1 = int(np.sum(cleaned_assignments))
    
    # Use modern binomtest if available, fallback to binom_test
    if hasattr(stats, 'binomtest'):
        binom_result = stats.binomtest(n_group1, total_samples, expected_ratio)
        binom_p_value = binom_result.pvalue
    else:
        binom_p_value = stats.binom_test(n_group1, total_samples, expected_ratio)
    
    warning_message = ""
    if not passed:
        warning_message = (
            f"Imbalanced groups: observed ratio={observed_ratio:.3f}, "
            f"expected={expected_ratio:.3f}"
        )
        if logger:
            logger.warning(warning_message)
    
    return AssumptionCheckResult(
        test_name="Randomization balance",
        passed=passed,
        statistic=float(observed_ratio),
        p_value=float(binom_p_value),
        warning_message=warning_message,
        details={
            "n_total": total_samples,
            "n_group1": n_group1,
            "expected_ratio": expected_ratio,
            "deviation": float(deviation),
        },
    )


###################################################################
# Family-Wise Error Rate Control
###################################################################


def compute_fwer_bonferroni(
    p_values: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Bonferroni correction for FWER control.
    
    Returns (adjusted_p, reject_mask, corrected_alpha).
    """
    p_flat = np.asarray(p_values).ravel()
    n_tests = len(p_flat)
    corrected_alpha = alpha / n_tests
    adjusted_p = np.minimum(p_flat * n_tests, 1.0)
    reject_mask = p_flat < corrected_alpha
    return adjusted_p, reject_mask, corrected_alpha


def compute_fwer_holm(
    p_values: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction for FWER control.
    
    More powerful than Bonferroni while still controlling FWER.
    
    Returns (adjusted_p, reject_mask).
    """
    p_flat = np.asarray(p_values).ravel()
    n_tests = len(p_flat)
    
    sorted_indices = np.argsort(p_flat)
    sorted_p_values = p_flat[sorted_indices]
    
    # Holm adjustment
    adjusted_p = np.zeros(n_tests)
    for i, (original_idx, p_value) in enumerate(zip(sorted_indices, sorted_p_values)):
        adjusted_p[original_idx] = p_value * (n_tests - i)
    
    # Enforce monotonicity (adjusted p-values must be non-decreasing)
    adjusted_sorted = adjusted_p[sorted_indices]
    for i in range(1, n_tests):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted_p[sorted_indices] = adjusted_sorted
    
    adjusted_p = np.minimum(adjusted_p, 1.0)
    reject_mask = adjusted_p < alpha
    
    return adjusted_p, reject_mask


def compute_fwer_sidak(
    p_values: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Šidák correction for FWER control.
    
    Slightly more powerful than Bonferroni for independent tests.
    
    Returns (adjusted_p, reject_mask, corrected_alpha).
    """
    p_flat = np.asarray(p_values).ravel()
    n_tests = len(p_flat)
    
    corrected_alpha = 1 - (1 - alpha) ** (1 / n_tests)
    adjusted_p = 1 - (1 - p_flat) ** n_tests
    reject_mask = p_flat < corrected_alpha
    
    return adjusted_p, reject_mask, corrected_alpha


def validate_fwer_control(
    p_values: np.ndarray,
    method: str = "holm",
    alpha: float = DEFAULT_ALPHA,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Validate and report FWER correction results.
    
    Parameters
    ----------
    p_values : array
        Raw p-values
    method : str
        'bonferroni', 'holm', or 'sidak'
    alpha : float
        Family-wise error rate
    """
    p_flat = filter_finite_values(p_values, flatten=True)
    n_tests = len(p_flat)
    
    if method == "bonferroni":
        adjusted_p, reject_mask, corrected_alpha = compute_fwer_bonferroni(p_flat, alpha)
    elif method == "sidak":
        adjusted_p, reject_mask, corrected_alpha = compute_fwer_sidak(p_flat, alpha)
    else:  # holm
        adjusted_p, reject_mask = compute_fwer_holm(p_flat, alpha)
        corrected_alpha = alpha / n_tests  # Approximate for reporting
    
    n_significant_raw = np.sum(p_flat < alpha)
    n_significant_corrected = np.sum(reject_mask)
    
    reduction_percentage = (
        (n_significant_raw - n_significant_corrected) / max(n_significant_raw, 1) * 100
    )
    
    details = {
        "n_tests": n_tests,
        "method": method,
        "alpha": alpha,
        "corrected_alpha": float(corrected_alpha),
        "n_significant_raw": int(n_significant_raw),
        "n_significant_corrected": int(n_significant_corrected),
        "reduction_pct": float(reduction_percentage),
    }
    
    warning_message = ""
    if n_significant_raw > 0 and n_significant_corrected == 0:
        warning_message = (
            f"All {n_significant_raw} raw significant results lost "
            f"after {method} correction"
        )
        if logger:
            logger.warning(warning_message)
    
    return AssumptionCheckResult(
        test_name=f"FWER control ({method})",
        passed=True,  # Correction always "passes" - it's informational
        statistic=float(n_significant_corrected),
        p_value=float(corrected_alpha),
        warning_message=warning_message,
        details=details,
    )


###################################################################
# Comprehensive Validation for Behavioral Contrasts
###################################################################


def _generate_permutation_null_distribution(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[np.ndarray, float]:
    """Generate permutation null distribution for two-group comparison.
    
    Returns (null_distribution, observed_statistic).
    """
    combined = np.concatenate([group1, group2])
    n_group1 = len(group1)
    
    observed_statistic = np.mean(group1) - np.mean(group2)
    
    rng = np.random.default_rng(random_seed)
    null_distribution = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(combined)
        perm_group1 = permuted[:n_group1]
        perm_group2 = permuted[n_group1:]
        null_distribution[i] = np.mean(perm_group1) - np.mean(perm_group2)
    
    return null_distribution, observed_statistic


def validate_behavioral_contrast(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Pain",
    group2_name: str = "Non-pain",
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = 1000,
    logger: Optional[logging.Logger] = None,
) -> ValidationReport:
    """Comprehensive validation for two-group behavioral contrast.
    
    Performs:
    - Normality checks (Shapiro-Wilk, QQ data)
    - Variance homogeneity (Levene's test)
    - Permutation test validation
    - Effect size with CI
    
    Returns ValidationReport with all checks and warnings.
    """
    report = ValidationReport()
    
    cleaned_group1 = filter_finite_values(group1, flatten=True)
    cleaned_group2 = filter_finite_values(group2, flatten=True)
    
    # Sample size check
    if (len(cleaned_group1) < MIN_SAMPLES_FOR_GROUP_COMPARISON or 
        len(cleaned_group2) < MIN_SAMPLES_FOR_GROUP_COMPARISON):
        report.add_warning(
            f"Very small sample sizes: {group1_name}={len(cleaned_group1)}, "
            f"{group2_name}={len(cleaned_group2)}"
        )
        report.set_flag("small_sample", True)
    
    # Normality checks
    normality_check1 = check_normality_shapiro(
        cleaned_group1, group1_name, alpha, logger
    )
    normality_check2 = check_normality_shapiro(
        cleaned_group2, group2_name, alpha, logger
    )
    report.normality_checks.extend([normality_check1, normality_check2])
    
    if not normality_check1.passed or not normality_check2.passed:
        report.set_flag("normality_violated", True)
        report.add_warning("Non-parametric tests recommended due to normality violation")
    
    # Variance check
    variance_check = check_variance_levene(
        cleaned_group1, 
        cleaned_group2, 
        group_names=[group1_name, group2_name], 
        alpha=alpha, 
        logger=logger
    )
    report.variance_checks.append(variance_check)
    
    if not variance_check.passed:
        report.set_flag("variance_heterogeneous", True)
        report.add_warning("Welch's t-test or Mann-Whitney U recommended")
    
    # Permutation validation
    null_distribution, observed_statistic = _generate_permutation_null_distribution(
        cleaned_group1, cleaned_group2, n_permutations
    )
    
    test_name = f"{group1_name} vs {group2_name}"
    permutation_check = validate_permutation_distribution(
        null_distribution, 
        observed_statistic, 
        n_permutations, 
        test_name, 
        alpha, 
        logger
    )
    report.permutation_checks.append(permutation_check)
    
    # Store null distribution for plotting
    permutation_check.details["null_distribution"] = null_distribution.tolist()
    
    return report


###################################################################
# Original Validation Functions (preserved)
###################################################################


def validate_baseline_window_pre_stimulus(
    baseline_window: Union[Tuple[float, float], List[float], float],
    baseline_end: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    *,
    strict: bool = False,
) -> Union[bool, Tuple[float, float]]:
    """Check baseline window ends before stimulus onset.
    
    Parameters
    ----------
    baseline_window : tuple, list, or float
        If tuple/list: (tmin, baseline_end). If float: baseline_end (tmin assumed to be before this).
    baseline_end : float, optional
        If baseline_window is a single float, this is ignored. If baseline_window is a tuple,
        this parameter is ignored and baseline_end is extracted from the tuple.
    logger : Logger, optional
        Logger for warnings.
    strict : bool, optional
        If True, raise ValueError when baseline extends past stimulus onset (t=0).
        If False (default), only log a warning. For scientifically valid baseline
        normalization, strict=True is recommended.
        
    Returns
    -------
    bool or tuple
        If baseline_window is a tuple/list, returns the validated tuple (tmin, baseline_end).
        If baseline_window is a float, returns True if valid, False otherwise.
        
    Raises
    ------
    ValueError
        If strict=True and baseline extends past stimulus onset.
    """
    STIMULUS_ONSET = 0.0
    
    def _handle_invalid_baseline(baseline_end_value: float) -> None:
        """Handle baseline that extends past stimulus."""
        if baseline_end_value > STIMULUS_ONSET:
            msg = (
                f"Baseline window extends past stimulus onset: baseline_end={baseline_end_value:.3f}s > 0. "
                "This contaminates baseline with stimulus-evoked activity, invalidating "
                "baseline normalization. Adjust baseline_window to end at or before t=0."
            )
            if strict:
                raise ValueError(msg)
            elif logger:
                logger.warning(msg)
    
    # Handle tuple/list input (new calling convention)
    if isinstance(baseline_window, (tuple, list)) and len(baseline_window) >= 2:
        tmin = float(baseline_window[0])
        baseline_end_value = float(baseline_window[1])
        _handle_invalid_baseline(baseline_end_value)
        return (tmin, baseline_end_value)
    
    # Handle old calling convention (two separate arguments)
    if baseline_end is not None:
        baseline_end_value = float(baseline_end)
        _handle_invalid_baseline(baseline_end_value)
        return baseline_end_value <= STIMULUS_ONSET
    
    # Handle single float input (treat as baseline_end)
    if isinstance(baseline_window, (int, float)):
        baseline_end_value = float(baseline_window)
        _handle_invalid_baseline(baseline_end_value)
        return baseline_end_value <= STIMULUS_ONSET
    
    raise ValueError(f"Invalid baseline_window type: {type(baseline_window)}")

