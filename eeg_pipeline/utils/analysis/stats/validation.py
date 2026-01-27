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
from scipy import stats

from .base import filter_finite_values


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


###################################################################
# Normality Checks
###################################################################




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


###################################################################
# Permutation/Randomization Validation
###################################################################


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


###################################################################
# Baseline Window Validation
###################################################################


def validate_baseline_window_pre_stimulus(
    baseline_window: Union[Tuple[float, float], List[float]],
    logger: Optional[logging.Logger] = None,
    *,
    strict: bool = False,
) -> Tuple[float, float]:
    """Check baseline window ends before stimulus onset.
    
    Parameters
    ----------
    baseline_window : tuple or list
        Baseline window as (tmin, baseline_end)
    logger : Logger, optional
        Logger for warnings
    strict : bool, optional
        If True, raise ValueError when baseline extends past stimulus onset (t=0).
        If False (default), only log a warning. For scientifically valid baseline
        normalization, strict=True is recommended.
        
    Returns
    -------
    tuple
        Validated tuple (tmin, baseline_end)
        
    Raises
    ------
    ValueError
        If strict=True and baseline extends past stimulus onset, or if baseline_window
        is not a tuple/list with at least 2 elements.
    """
    STIMULUS_ONSET = 0.0
    
    if not isinstance(baseline_window, (tuple, list)) or len(baseline_window) < 2:
        raise ValueError(
            f"baseline_window must be a tuple or list with at least 2 elements, "
            f"got {type(baseline_window)}"
        )
    
    tmin = float(baseline_window[0])
    baseline_end_value = float(baseline_window[1])
    
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
    
    return (tmin, baseline_end_value)

