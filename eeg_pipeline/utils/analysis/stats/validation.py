"""
Data Validation
===============

Validation functions for EEG data integrity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

from .base import get_config_value, ensure_config

try:
    from ...config.loader import get_constants
except ImportError:
    get_constants = None


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

