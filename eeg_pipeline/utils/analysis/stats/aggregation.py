"""
Statistical Aggregation
=======================

Functions for aggregating statistics across subjects/groups.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def compute_error_bars_from_ci_dicts(
    values: List[float],
    ci_dicts: List[Optional[Dict[str, List[float]]]],
) -> Tuple[List[float], List[float]]:
    """Convert confidence interval dictionaries to error bar arrays.
    
    Parameters
    ----------
    values : List[float]
        List of mean values
    ci_dicts : List[Optional[Dict[str, List[float]]]]
        List of CI dictionaries with "low" and "high" keys
        
    Returns
    -------
    Tuple[List[float], List[float]]
        (lower_errors, upper_errors) where errors are distances from mean
    """
    lower_errors = []
    upper_errors = []
    
    for mean_value, ci_dict in zip(values, ci_dicts):
        has_valid_ci = (
            ci_dict is not None
            and "low" in ci_dict
            and "high" in ci_dict
        )
        
        if not has_valid_ci:
            lower_errors.append(0.0)
            upper_errors.append(0.0)
            continue
        
        ci_lower_value = ci_dict["low"][0] if ci_dict["low"] else mean_value
        ci_upper_value = ci_dict["high"][0] if ci_dict["high"] else mean_value
        
        lower_errors.append(mean_value - ci_lower_value)
        upper_errors.append(ci_upper_value - mean_value)
    
    return lower_errors, upper_errors



