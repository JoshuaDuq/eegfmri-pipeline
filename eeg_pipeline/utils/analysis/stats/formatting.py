"""
Statistical Formatting
======================

Functions for formatting statistical results for display.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import stats as scipy_stats

from .base import get_ci_level, get_fdr_alpha
from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p is None or not isinstance(p, (int, float)):
        return "p=N/A"
    if p < 0.001:
        return "p<.001"
    elif p < 0.01:
        return f"p={p:.3f}"
    else:
        return f"p={p:.2f}"


def format_correlation_text(r_val: float, p_val: Optional[float] = None) -> str:
    """Format correlation coefficient for display."""
    if r_val is None or not isinstance(r_val, (int, float)):
        return "r=N/A"
    text = f"r={r_val:.2f}"
    if p_val is not None:
        text += f" ({format_p_value(p_val)})"
    return text


def format_cluster_ann(
    p: float,
    k: Optional[int] = None,
    mass: Optional[float] = None,
    config: Optional[Any] = None,
) -> str:
    """Format cluster test annotation."""
    parts = []

    if k is not None:
        parts.append(f"k={k}")
    if mass is not None:
        parts.append(f"mass={mass:.1f}")

    p_str = format_p_value(p)
    parts.append(p_str)

    alpha = get_fdr_alpha(config)
    if p is not None and p <= alpha:
        parts.append("*")

    return " ".join(parts)


def format_fdr_ann(
    q_min: Optional[float],
    k_rej: Optional[int],
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> str:
    """Format FDR correction annotation."""
    if alpha is None:
        alpha = get_fdr_alpha(config)

    if q_min is None:
        return "FDR: no tests"

    if k_rej is None or k_rej == 0:
        return f"FDR q={q_min:.3f} (none sig)"

    return f"FDR q={q_min:.3f} ({k_rej} sig at α={alpha})"


def format_correlation_stats_text(
    r: float,
    p: float,
    n: int,
    ci: Optional[tuple[float, float]] = None,
    stats_tag: Optional[str] = None,
    config: Optional[Any] = None,
    include_r_squared: bool = True,
    include_bayes_factor: bool = False,
) -> str:
    """Format full correlation statistics with R², CI, and optional Bayes Factor.

    Parameters
    ----------
    r : float
        Correlation coefficient
    p : float
        P-value
    n : int
        Sample size
    ci : tuple, optional
        (ci_low, ci_high) confidence interval. If None, Fisher Z CI is computed.
    stats_tag : str, optional
        Additional tag to append (e.g., "precomputed")
    include_r_squared : bool
        Whether to include R² (variance explained)
    include_bayes_factor : bool
        Whether to include Bayes Factor interpretation

    Returns
    -------
    str
        Formatted statistics text
    """
    ci_low, ci_high = _extract_confidence_interval(ci)
    main_text = _format_correlation_main_text(r, p, n, include_r_squared)
    ci_text = _format_confidence_interval_text(r, n, ci_low, ci_high, config)
    bf_text = _format_bayes_factor_text(r, n, include_bayes_factor)
    tag_text = _format_stats_tag(stats_tag)

    return main_text + ci_text + bf_text + tag_text


def _extract_confidence_interval(
    ci: Optional[tuple[float, float]],
) -> tuple[Optional[float], Optional[float]]:
    """Extract confidence interval bounds from tuple."""
    if ci is None:
        return None, None
    if isinstance(ci, tuple) and len(ci) == 2:
        return ci[0], ci[1]
    return None, None


def _format_correlation_main_text(
    r: float, p: float, n: int, include_r_squared: bool
) -> str:
    """Format main correlation statistics text."""
    if include_r_squared and np.isfinite(r):
        r_squared = r ** 2
        return f"r={r:.3f}, R²={r_squared:.3f}, {format_p_value(p)}, n={n}"
    return f"r={r:.3f}, {format_p_value(p)}, n={n}"


def _format_confidence_interval_text(
    r: float,
    n: int,
    ci_low: Optional[float],
    ci_high: Optional[float],
    config: Optional[Any],
) -> str:
    """Format confidence interval text, computing Fisher Z CI if needed."""
    if ci_low is None or ci_high is None:
        if np.isfinite(r) and n > 3:
            ci_level = get_ci_level(config)
            ci_low, ci_high = _compute_fisher_z_ci(
                r, n, ci_level=ci_level, config=config
            )

    if ci_low is None or ci_high is None:
        return ""

    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return ""

    ci_level = get_ci_level(config)
    ci_percent = int(round(ci_level * 100))
    return f", {ci_percent}% CI [{ci_low:.3f}, {ci_high:.3f}]"


def _format_bayes_factor_text(r: float, n: int, include_bayes_factor: bool) -> str:
    """Format Bayes Factor text if requested."""
    if not include_bayes_factor:
        return ""
    if not (np.isfinite(r) and n > 3):
        return ""

    bf10 = _compute_bf10_correlation(r, n)
    if not np.isfinite(bf10):
        return ""

    interpretation = _interpret_bayes_factor(bf10)
    formatted_bf = _format_bayes_factor_value(bf10)
    return f", BF₁₀={formatted_bf} ({interpretation})"


def _format_bayes_factor_value(bf10: float) -> str:
    """Format Bayes Factor value with appropriate precision."""
    if bf10 >= 1000:
        return ">1000"
    elif bf10 >= 100:
        return f"{bf10:.0f}"
    elif bf10 >= 1:
        return f"{bf10:.1f}"
    else:
        return f"{bf10:.2f}"


def _format_stats_tag(stats_tag: Optional[str]) -> str:
    """Format optional statistics tag."""
    if stats_tag:
        return f" [{stats_tag}]"
    return ""


def _compute_fisher_z_ci(
    r: float,
    n: int,
    *,
    ci_level: float = 0.95,
    config: Optional[Any] = None,
) -> tuple[float, float]:
    """Compute Fisher Z-transformed confidence interval for correlation.

    Uses the Fisher transformation: z = arctanh(r)
    SE(z) = 1 / sqrt(n - 3)
    """
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan

    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_clipped = np.clip(r, clip_min, clip_max)

    z = np.arctanh(r_clipped)
    se_z = 1.0 / np.sqrt(n - 3)

    alpha = 1 - ci_level
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)

    z_low = z - z_crit * se_z
    z_high = z + z_crit * se_z

    ci_low = float(np.tanh(z_low))
    ci_high = float(np.tanh(z_high))
    return ci_low, ci_high


def _compute_bf10_correlation(r: float, n: int) -> float:
    """Compute Bayes Factor for correlation (H1: ρ ≠ 0 vs H0: ρ = 0).

    Uses the minimum Bayes Factor bound from Sellke, Bayarri & Berger (2001):
    BF₁₀ ≥ -1/(e * p * log(p))

    This is a conservative lower bound that is well-calibrated and does not
    over-state evidence.

    For more accurate computation, consider using the pingouin package:
    pingouin.bayesfactor_pearson(r, n)

    Reference:
    Sellke, T., Bayarri, M. J., & Berger, J. O. (2001). Calibration of p values
    for testing precise null hypotheses. The American Statistician.
    """
    if n <= 3 or not np.isfinite(r):
        return np.nan

    r_clipped = np.clip(r, -0.9999, 0.9999)
    r_squared = r_clipped ** 2

    degrees_of_freedom = n - 2
    denominator = 1 - r_squared + 1e-10
    t_statistic = r_clipped * np.sqrt(degrees_of_freedom / denominator)
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_statistic), degrees_of_freedom))

    if not np.isfinite(p_value) or p_value <= 0 or p_value >= 1:
        return np.nan

    bf10 = _compute_bayes_factor_from_p_value(p_value)
    bf10 = _adjust_bayes_factor_for_effect_size(bf10, r)
    bf10 = np.clip(bf10, 1e-4, 1e6)

    return float(bf10)


def _compute_bayes_factor_from_p_value(p_value: float) -> float:
    """Compute Bayes Factor from p-value using Sellke et al. (2001) bound."""
    p_threshold = 1 / np.e
    if p_value < p_threshold:
        bf10 = -1 / (np.e * p_value * np.log(p_value))
    else:
        bf10 = 1 / (1 + np.sqrt(p_value) * 2)
    return bf10


def _adjust_bayes_factor_for_effect_size(bf10: float, r: float) -> float:
    """Apply effect size adjustment to Bayes Factor for small correlations."""
    small_effect_threshold = 0.1
    if abs(r) < small_effect_threshold:
        penalty_factor = abs(r) / small_effect_threshold
        bf10 = bf10 * penalty_factor
    return bf10


def _interpret_bayes_factor(bf10: float) -> str:
    """Interpret Bayes Factor using modified Jeffreys scale.

    Jeffreys (1961) / Lee & Wagenmakers (2013) scale:
    BF > 100: Extreme evidence for H1
    BF 30-100: Very strong evidence for H1
    BF 10-30: Strong evidence for H1
    BF 3-10: Moderate evidence for H1
    BF 1-3: Anecdotal evidence for H1
    BF 1: No evidence
    BF 1/3-1: Anecdotal evidence for H0
    BF < 1/3: Moderate+ evidence for H0
    """
    if not np.isfinite(bf10):
        return "N/A"

    if bf10 >= 100:
        return "extreme"
    elif bf10 >= 30:
        return "very strong"
    elif bf10 >= 10:
        return "strong"
    elif bf10 >= 3:
        return "moderate"
    elif bf10 >= 1:
        return "anecdotal"
    elif bf10 >= 1 / 3:
        return "anecdotal H₀"
    elif bf10 >= 1 / 10:
        return "moderate H₀"
    elif bf10 >= 1 / 30:
        return "strong H₀"
    else:
        return "very strong H₀"


def _safe_float(value: Any) -> float:
    """Safely convert value to float, returning NaN on failure."""
    try:
        f = float(value)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")
