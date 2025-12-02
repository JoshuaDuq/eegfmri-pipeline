"""
Effect Size Statistics
======================

Cohen's d, correlation difference effects, and Fisher z-tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
) -> float:
    """
    Compute Cohen's d effect size.
    
    Uses pooled SD by default; set pooled=False for Cohen's d_s.
    """
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()

    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    n1, n2 = len(g1), len(g2)

    if pooled:
        pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        sd = np.sqrt(pooled_var)
    else:
        sd = np.sqrt((s1**2 + s2**2) / 2)

    if sd < 1e-12:
        return np.nan

    return float((m1 - m2) / sd)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Hedges' g (bias-corrected Cohen's d)."""
    d = cohens_d(group1, group2, pooled=True)
    if not np.isfinite(d):
        return np.nan

    n1 = np.sum(np.isfinite(group1))
    n2 = np.sum(np.isfinite(group2))
    df = n1 + n2 - 2

    if df < 2:
        return d

    # Approximate correction factor
    correction = 1 - 3 / (4 * df - 1)
    return float(d * correction)


def glass_delta(group1: np.ndarray, group2: np.ndarray, control: int = 2) -> float:
    """Glass' delta using control group SD."""
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()

    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    m1, m2 = np.mean(g1), np.mean(g2)

    if control == 1:
        sd = np.std(g1, ddof=1)
    else:
        sd = np.std(g2, ddof=1)

    if sd < 1e-12:
        return np.nan

    return float((m1 - m2) / sd)


def fisher_z_test(
    r1: float,
    r2: float,
    n1: int,
    n2: int,
) -> Tuple[float, float]:
    """
    Fisher z-test for difference between two correlations.
    
    Returns (z_statistic, p_value).
    """
    if n1 < 4 or n2 < 4:
        return np.nan, np.nan

    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan, np.nan

    r1 = np.clip(r1, -0.9999, 0.9999)
    r2 = np.clip(r2, -0.9999, 0.9999)

    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se

    p = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return float(z_stat), float(p)


def cohens_q(r1: float, r2: float) -> float:
    """Cohen's q for difference between correlations."""
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan

    r1 = np.clip(r1, -0.9999, 0.9999)
    r2 = np.clip(r2, -0.9999, 0.9999)

    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    return float(z1 - z2)


def correlation_difference_effect(
    r1: float,
    r2: float,
    n1: int,
    n2: int,
) -> Dict[str, float]:
    """
    Comprehensive effect statistics for correlation difference.
    
    Returns dict with z_stat, p_value, cohens_q, r_diff.
    """
    z_stat, p_val = fisher_z_test(r1, r2, n1, n2)
    q = cohens_q(r1, r2)

    return {
        "r_diff": float(r1 - r2) if np.isfinite(r1) and np.isfinite(r2) else np.nan,
        "z_stat": z_stat,
        "p_value": p_val,
        "cohens_q": q,
    }


def r_to_d(r: float) -> float:
    """Convert correlation to Cohen's d approximation."""
    if not np.isfinite(r) or np.abs(r) >= 1:
        return np.nan
    return 2 * r / np.sqrt(1 - r**2)


def d_to_r(d: float) -> float:
    """Convert Cohen's d to correlation approximation."""
    if not np.isfinite(d):
        return np.nan
    return d / np.sqrt(d**2 + 4)


def compute_effect_sizes(
    r_val: float,
    p_val: float,
    n_samples: int,
    group1_data: Optional[np.ndarray] = None,
    group2_data: Optional[np.ndarray] = None,
    effect_size_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multiple effect size metrics.
    
    Supported metrics: r, r_squared, d_from_r, cohens_d, hedges_g.
    """
    if effect_size_metrics is None:
        effect_size_metrics = ["r", "r_squared", "d_from_r"]

    results = {}

    if "r" in effect_size_metrics:
        results["r"] = float(r_val) if np.isfinite(r_val) else np.nan

    if "r_squared" in effect_size_metrics:
        results["r_squared"] = float(r_val**2) if np.isfinite(r_val) else np.nan

    if "d_from_r" in effect_size_metrics:
        results["d_from_r"] = r_to_d(r_val)

    if group1_data is not None and group2_data is not None:
        if "cohens_d" in effect_size_metrics:
            results["cohens_d"] = cohens_d(group1_data, group2_data)
        if "hedges_g" in effect_size_metrics:
            results["hedges_g"] = hedges_g(group1_data, group2_data)

    return results

