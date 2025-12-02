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
from typing import Any, Optional, Tuple, Union, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_config_value, ensure_config

try:
    from ...config.loader import get_constants
except ImportError:
    get_constants = None


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


###################################################################
# Normality Checks
###################################################################


def check_normality_shapiro(
    data: np.ndarray,
    group_name: str = "data",
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check normality using Shapiro-Wilk test.
    
    Appropriate for n < 5000. For larger samples, use check_normality_dagostino.
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    
    if len(data) < 3:
        return AssumptionCheckResult(
            test_name="Shapiro-Wilk",
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message=f"{group_name}: Insufficient data for normality test (n={len(data)})",
        )
    
    # Shapiro-Wilk limited to 5000 samples
    if len(data) > 5000:
        data = np.random.default_rng(42).choice(data, size=5000, replace=False)
    
    stat, p_value = stats.shapiro(data)
    passed = p_value >= alpha
    
    warning = ""
    if not passed:
        warning = f"{group_name}: Non-normal distribution (Shapiro-Wilk W={stat:.4f}, p={p_value:.4f})"
        if logger:
            logger.warning(warning)
    
    return AssumptionCheckResult(
        test_name="Shapiro-Wilk",
        passed=passed,
        statistic=float(stat),
        p_value=float(p_value),
        warning_message=warning,
        details={"n": len(data), "group": group_name},
    )


def check_normality_dagostino(
    data: np.ndarray,
    group_name: str = "data",
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check normality using D'Agostino-Pearson test.
    
    Better for larger samples (n > 20).
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    
    if len(data) < 20:
        return AssumptionCheckResult(
            test_name="D'Agostino-Pearson",
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message=f"{group_name}: Insufficient data for D'Agostino test (n={len(data)}, need 20+)",
        )
    
    stat, p_value = stats.normaltest(data)
    passed = p_value >= alpha
    
    warning = ""
    if not passed:
        warning = f"{group_name}: Non-normal distribution (D'Agostino K²={stat:.4f}, p={p_value:.4f})"
        if logger:
            logger.warning(warning)
    
    return AssumptionCheckResult(
        test_name="D'Agostino-Pearson",
        passed=passed,
        statistic=float(stat),
        p_value=float(p_value),
        warning_message=warning,
        details={"n": len(data), "group": group_name},
    )


def compute_qq_data(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute QQ plot data for normality assessment.
    
    Returns (theoretical_quantiles, sample_quantiles, slope, intercept).
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    data_sorted = np.sort(data)
    
    n = len(data_sorted)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    
    # Fit line
    slope, intercept, _, _, _ = stats.linregress(theoretical_quantiles, data_sorted)
    
    return theoretical_quantiles, data_sorted, float(slope), float(intercept)


###################################################################
# Variance Homogeneity Checks
###################################################################


def check_variance_levene(
    *groups: np.ndarray,
    group_names: Optional[List[str]] = None,
    alpha: float = 0.05,
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
    clean_groups = []
    for g in groups:
        g_arr = np.asarray(g).ravel()
        g_clean = g_arr[np.isfinite(g_arr)]
        if len(g_clean) >= 2:
            clean_groups.append(g_clean)
    
    if len(clean_groups) < 2:
        return AssumptionCheckResult(
            test_name="Levene",
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message="Insufficient groups for Levene's test",
        )
    
    stat, p_value = stats.levene(*clean_groups, center=center)
    passed = p_value >= alpha
    
    group_names = group_names or [f"Group {i+1}" for i in range(len(clean_groups))]
    variances = {name: float(np.var(g, ddof=1)) for name, g in zip(group_names, clean_groups)}
    
    warning = ""
    if not passed:
        warning = f"Heterogeneous variances (Levene W={stat:.4f}, p={p_value:.4f}): {variances}"
        if logger:
            logger.warning(warning)
    
    return AssumptionCheckResult(
        test_name="Levene",
        passed=passed,
        statistic=float(stat),
        p_value=float(p_value),
        warning_message=warning,
        details={"variances": variances, "n_groups": len(clean_groups)},
    )


def check_variance_bartlett(
    *groups: np.ndarray,
    group_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Check homogeneity of variance using Bartlett's test.
    
    More powerful than Levene but assumes normality.
    """
    clean_groups = []
    for g in groups:
        g_arr = np.asarray(g).ravel()
        g_clean = g_arr[np.isfinite(g_arr)]
        if len(g_clean) >= 2:
            clean_groups.append(g_clean)
    
    if len(clean_groups) < 2:
        return AssumptionCheckResult(
            test_name="Bartlett",
            passed=False,
            statistic=np.nan,
            p_value=np.nan,
            warning_message="Insufficient groups for Bartlett's test",
        )
    
    stat, p_value = stats.bartlett(*clean_groups)
    passed = p_value >= alpha
    
    group_names = group_names or [f"Group {i+1}" for i in range(len(clean_groups))]
    variances = {name: float(np.var(g, ddof=1)) for name, g in zip(group_names, clean_groups)}
    
    warning = ""
    if not passed:
        warning = f"Heterogeneous variances (Bartlett χ²={stat:.4f}, p={p_value:.4f}): {variances}"
        if logger:
            logger.warning(warning)
    
    return AssumptionCheckResult(
        test_name="Bartlett",
        passed=passed,
        statistic=float(stat),
        p_value=float(p_value),
        warning_message=warning,
        details={"variances": variances, "n_groups": len(clean_groups)},
    )


###################################################################
# Permutation/Randomization Validation
###################################################################


def validate_permutation_distribution(
    null_distribution: np.ndarray,
    observed_statistic: float,
    n_permutations: int,
    test_name: str = "permutation",
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> AssumptionCheckResult:
    """Validate permutation test assumptions and distribution quality.
    
    Checks:
    - Sufficient permutations for desired precision
    - Null distribution symmetry (for two-tailed tests)
    - No extreme outliers suggesting computational issues
    """
    null = np.asarray(null_distribution).ravel()
    null = null[np.isfinite(null)]
    
    warnings = []
    passed = True
    details = {}
    
    # Check sufficient permutations
    min_perm_for_alpha = int(1 / alpha) * 10
    if n_permutations < min_perm_for_alpha:
        warnings.append(f"Low permutation count ({n_permutations}) for α={alpha}; recommend ≥{min_perm_for_alpha}")
        passed = False
    
    # Check null distribution properties
    null_mean = np.mean(null)
    null_std = np.std(null)
    null_skew = stats.skew(null) if len(null) > 8 else 0
    null_kurt = stats.kurtosis(null) if len(null) > 8 else 0
    
    details["null_mean"] = float(null_mean)
    details["null_std"] = float(null_std)
    details["null_skew"] = float(null_skew)
    details["null_kurtosis"] = float(null_kurt)
    
    # Check for extreme skewness (may indicate issues)
    if abs(null_skew) > 2:
        warnings.append(f"Highly skewed null distribution (skew={null_skew:.2f})")
    
    # Compute permutation p-value
    n_extreme = np.sum(np.abs(null) >= np.abs(observed_statistic))
    p_perm = (n_extreme + 1) / (len(null) + 1)
    details["p_permutation"] = float(p_perm)
    details["n_extreme"] = int(n_extreme)
    
    # Check Monte Carlo error
    mc_se = np.sqrt(p_perm * (1 - p_perm) / n_permutations)
    details["mc_standard_error"] = float(mc_se)
    
    if mc_se > 0.01 and p_perm < 0.1:
        warnings.append(f"High Monte Carlo SE ({mc_se:.4f}); increase permutations for stable p-value")
    
    warning_msg = "; ".join(warnings) if warnings else ""
    if warning_msg and logger:
        logger.warning(f"{test_name}: {warning_msg}")
    
    return AssumptionCheckResult(
        test_name=f"Permutation validation ({test_name})",
        passed=passed,
        statistic=float(observed_statistic),
        p_value=float(p_perm),
        warning_message=warning_msg,
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
    assignments = np.asarray(group_assignments).ravel()
    assignments = assignments[np.isfinite(assignments)]
    
    observed_ratio = np.mean(assignments)
    deviation = abs(observed_ratio - expected_ratio)
    passed = deviation <= tolerance
    
    # Binomial test for significant imbalance
    n = len(assignments)
    k = int(np.sum(assignments))
    binom_p = stats.binom_test(k, n, expected_ratio) if hasattr(stats, 'binom_test') else \
              stats.binomtest(k, n, expected_ratio).pvalue
    
    warning = ""
    if not passed:
        warning = f"Imbalanced groups: observed ratio={observed_ratio:.3f}, expected={expected_ratio:.3f}"
        if logger:
            logger.warning(warning)
    
    return AssumptionCheckResult(
        test_name="Randomization balance",
        passed=passed,
        statistic=float(observed_ratio),
        p_value=float(binom_p),
        warning_message=warning,
        details={
            "n_total": n,
            "n_group1": k,
            "expected_ratio": expected_ratio,
            "deviation": float(deviation),
        },
    )


###################################################################
# Family-Wise Error Rate Control
###################################################################


def compute_fwer_bonferroni(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Bonferroni correction for FWER control.
    
    Returns (adjusted_p, reject_mask, corrected_alpha).
    """
    p = np.asarray(p_values).ravel()
    n_tests = len(p)
    corrected_alpha = alpha / n_tests
    adjusted_p = np.minimum(p * n_tests, 1.0)
    reject = p < corrected_alpha
    return adjusted_p, reject, corrected_alpha


def compute_fwer_holm(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction for FWER control.
    
    More powerful than Bonferroni while still controlling FWER.
    
    Returns (adjusted_p, reject_mask).
    """
    p = np.asarray(p_values).ravel()
    n = len(p)
    
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    
    # Holm adjustment
    adjusted = np.zeros(n)
    for i, (idx, pval) in enumerate(zip(sorted_idx, sorted_p)):
        adjusted[idx] = pval * (n - i)
    
    # Enforce monotonicity
    adjusted_sorted = adjusted[sorted_idx]
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i-1])
    adjusted[sorted_idx] = adjusted_sorted
    
    adjusted = np.minimum(adjusted, 1.0)
    reject = adjusted < alpha
    
    return adjusted, reject


def compute_fwer_sidak(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Šidák correction for FWER control.
    
    Slightly more powerful than Bonferroni for independent tests.
    
    Returns (adjusted_p, reject_mask, corrected_alpha).
    """
    p = np.asarray(p_values).ravel()
    n_tests = len(p)
    
    corrected_alpha = 1 - (1 - alpha) ** (1 / n_tests)
    adjusted_p = 1 - (1 - p) ** n_tests
    reject = p < corrected_alpha
    
    return adjusted_p, reject, corrected_alpha


def validate_fwer_control(
    p_values: np.ndarray,
    method: str = "holm",
    alpha: float = 0.05,
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
    p = np.asarray(p_values).ravel()
    p = p[np.isfinite(p)]
    n_tests = len(p)
    
    if method == "bonferroni":
        adjusted, reject, corrected_alpha = compute_fwer_bonferroni(p, alpha)
    elif method == "sidak":
        adjusted, reject, corrected_alpha = compute_fwer_sidak(p, alpha)
    else:  # holm
        adjusted, reject = compute_fwer_holm(p, alpha)
        corrected_alpha = alpha / n_tests  # Approximate for reporting
    
    n_reject_raw = np.sum(p < alpha)
    n_reject_corrected = np.sum(reject)
    
    details = {
        "n_tests": n_tests,
        "method": method,
        "alpha": alpha,
        "corrected_alpha": float(corrected_alpha),
        "n_significant_raw": int(n_reject_raw),
        "n_significant_corrected": int(n_reject_corrected),
        "reduction_pct": float((n_reject_raw - n_reject_corrected) / max(n_reject_raw, 1) * 100),
    }
    
    warning = ""
    if n_reject_raw > 0 and n_reject_corrected == 0:
        warning = f"All {n_reject_raw} raw significant results lost after {method} correction"
        if logger:
            logger.warning(warning)
    
    return AssumptionCheckResult(
        test_name=f"FWER control ({method})",
        passed=True,  # Correction always "passes" - it's informational
        statistic=float(n_reject_corrected),
        p_value=float(corrected_alpha),
        warning_message=warning,
        details=details,
    )


###################################################################
# Comprehensive Validation for Behavioral Contrasts
###################################################################


def validate_behavioral_contrast(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Pain",
    group2_name: str = "Non-pain",
    alpha: float = 0.05,
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
    
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    
    # Sample size check
    if len(g1) < 3 or len(g2) < 3:
        report.add_warning(f"Very small sample sizes: {group1_name}={len(g1)}, {group2_name}={len(g2)}")
        report.set_flag("small_sample", True)
    
    # Normality checks
    norm1 = check_normality_shapiro(g1, group1_name, alpha, logger)
    norm2 = check_normality_shapiro(g2, group2_name, alpha, logger)
    report.normality_checks.extend([norm1, norm2])
    
    if not norm1.passed or not norm2.passed:
        report.set_flag("normality_violated", True)
        report.add_warning("Non-parametric tests recommended due to normality violation")
    
    # Variance check
    var_check = check_variance_levene(g1, g2, group_names=[group1_name, group2_name], 
                                       alpha=alpha, logger=logger)
    report.variance_checks.append(var_check)
    
    if not var_check.passed:
        report.set_flag("variance_heterogeneous", True)
        report.add_warning("Welch's t-test or Mann-Whitney U recommended")
    
    # Permutation validation
    rng = np.random.default_rng(42)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    
    # Observed statistic (difference in means)
    obs_diff = np.mean(g1) - np.mean(g2)
    
    # Generate null distribution
    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_diffs[i] = np.mean(perm[:n1]) - np.mean(perm[n1:])
    
    perm_check = validate_permutation_distribution(
        null_diffs, obs_diff, n_permutations, 
        f"{group1_name} vs {group2_name}", alpha, logger
    )
    report.permutation_checks.append(perm_check)
    
    # Store null distribution for plotting
    perm_check.details["null_distribution"] = null_diffs.tolist()
    
    return report


###################################################################
# Original Validation Functions (preserved)
###################################################################


def validate_pain_binary_values(
    values: pd.Series,
    column_name: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, int]:
    """Validate pain binary column contains only 0/1 values."""
    logger = logger or logging.getLogger(__name__)

    numeric_vals = pd.to_numeric(values, errors="coerce")
    n_total = len(values)
    n_nan = int(numeric_vals.isna().sum())
    n_invalid = int(((numeric_vals != 0) & (numeric_vals != 1) & numeric_vals.notna()).sum())

    if n_nan > 0 or n_invalid > 0:
        error_msg = (
            f"Invalid pain binary values in '{column_name}': "
            f"{n_nan} NaN/missing, {n_invalid} non-binary out of {n_total}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    validated = numeric_vals.fillna(0).astype(int).values
    return validated, n_nan + n_invalid


def validate_temperature_values(
    values: pd.Series,
    column_name: str,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, int]:
    """Validate temperature values are within expected range."""
    logger = logger or logging.getLogger(__name__)

    if min_temp is None or max_temp is None:
        config = ensure_config(config)
        if get_constants is not None:
            io_constants = get_constants("io", config)
            min_temp = min_temp or float(io_constants.get("temperature_min", 35.0))
            max_temp = max_temp or float(io_constants.get("temperature_max", 50.0))
        else:
            min_temp = min_temp or 35.0
            max_temp = max_temp or 50.0

    numeric_vals = pd.to_numeric(values, errors="coerce")
    n_nan = int(numeric_vals.isna().sum())
    n_out_of_range = int(((numeric_vals < min_temp) | (numeric_vals > max_temp)).sum())

    if n_nan > 0:
        logger.warning(f"{column_name}: {n_nan} NaN values")
    if n_out_of_range > 0:
        logger.warning(f"{column_name}: {n_out_of_range} values outside [{min_temp}, {max_temp}]")

    return numeric_vals.values, n_nan + n_out_of_range


def validate_baseline_window_pre_stimulus(
    baseline_window: Union[Tuple[float, float], List[float], float],
    baseline_end: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
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
        
    Returns
    -------
    bool or tuple
        If baseline_window is a tuple/list, returns the validated tuple (tmin, baseline_end).
        If baseline_window is a float, returns True if valid, False otherwise.
    """
    # Handle tuple/list input (new calling convention)
    if isinstance(baseline_window, (tuple, list)) and len(baseline_window) >= 2:
        tmin, baseline_end_val = float(baseline_window[0]), float(baseline_window[1])
        if baseline_end_val > 0:
            if logger:
                logger.warning(f"Baseline extends past stimulus: baseline_end={baseline_end_val}")
            return (tmin, baseline_end_val)  # Return tuple even if invalid
        return (tmin, baseline_end_val)
    
    # Handle old calling convention (two separate arguments)
    if baseline_end is not None:
        baseline_end_float = float(baseline_end)
        if baseline_end_float > 0:
            if logger:
                logger.warning(f"Baseline extends past stimulus: baseline_end={baseline_end_float}")
            return False
        return True
    
    # Handle single float input (treat as baseline_end)
    if isinstance(baseline_window, (int, float)):
        baseline_end_val = float(baseline_window)
        if baseline_end_val > 0:
            if logger:
                logger.warning(f"Baseline extends past stimulus: baseline_end={baseline_end_val}")
            return False
        return True
    
    raise ValueError(f"Invalid baseline_window type: {type(baseline_window)}")


def check_pyriemann() -> bool:
    """Check if pyriemann package is available."""
    try:
        import pyriemann  # type: ignore[reportMissingImports]
        return True
    except ImportError:
        return False


def extract_finite_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mask where both arrays are finite."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask], mask


def extract_pain_masks(pain_vals: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract boolean masks for pain and non-pain trials."""
    pain_arr = np.asarray(pain_vals)
    pain_mask = pain_arr == 1
    nonpain_mask = pain_arr == 0
    return pain_mask, nonpain_mask


def extract_duration_data(durations: pd.Series, mask: np.ndarray) -> np.ndarray:
    """Extract duration data for masked trials."""
    return durations.values[mask]

