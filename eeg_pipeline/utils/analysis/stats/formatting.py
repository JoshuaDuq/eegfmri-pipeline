"""
Statistical Formatting
======================

Functions for formatting statistical results for display.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import get_fdr_alpha


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
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
) -> str:
    """Format full correlation statistics."""
    text = f"r={r:.3f}, {format_p_value(p)}, n={n}"
    if ci_low is not None and ci_high is not None:
        text += f", 95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    return text


def _safe_float(value: Any) -> float:
    """Safely convert value to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

