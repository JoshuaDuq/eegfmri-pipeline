"""
Numeric Data Utilities
======================

Centralized utilities for handling numeric conversions and transformations
across the EEG pipeline. This module provides a single source of truth for
numeric data handling to avoid redundant computations.

Functions:
- ensure_numeric: Convert DataFrame columns to numeric types efficiently
- ensure_numeric_array: Convert arrays to numeric with error handling
- broadcast_values: Efficient broadcasting for granularity operations
"""

from __future__ import annotations

from typing import Optional, List, Union

import numpy as np
import pandas as pd


def ensure_numeric(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Ensure specified columns are numeric, converting efficiently.
    
    This is a centralized replacement for repeated pd.to_numeric calls
    throughout the pipeline. Uses vectorized operations for better performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : Optional[List[str]]
        Columns to convert. If None, converts all columns.
    inplace : bool
        If True, modifies DataFrame in place. Otherwise returns a copy.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with numeric columns
    
    Examples
    --------
    >>> df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    >>> numeric_df = ensure_numeric(df, columns=["a"])
    >>> numeric_df["a"].dtype
    dtype('float64')
    """
    if columns is None:
        columns = df.columns.tolist()
    
    target_df = df if inplace else df.copy()
    
    for col in columns:
        if col in target_df.columns:
            target_df[col] = pd.to_numeric(target_df[col], errors="coerce")
    
    return target_df


def ensure_numeric_array(
    arr: Union[np.ndarray, pd.Series],
    dtype: type = float,
) -> np.ndarray:
    """
    Convert array-like to numeric array with error handling.
    
    Parameters
    ----------
    arr : Union[np.ndarray, pd.Series]
        Input array
    dtype : type
        Target dtype (default: float)
    
    Returns
    -------
    np.ndarray
        Numeric array with NaN for non-convertible values
    """
    if isinstance(arr, pd.Series):
        return pd.to_numeric(arr, errors="coerce").to_numpy(dtype=dtype)
    
    try:
        return np.asarray(arr, dtype=dtype)
    except (ValueError, TypeError):
        result = np.empty_like(arr, dtype=dtype)
        result[:] = np.nan
        return result


def broadcast_values(
    values: np.ndarray,
    n_repeats: int,
) -> np.ndarray:
    """
    Efficiently broadcast values to n_repeats rows.
    
    This replaces np.tile for pandas DataFrame row broadcasting with
    better memory efficiency and clearer intent.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of values to broadcast
    n_repeats : int
        Number of times to repeat values
    
    Returns
    -------
    np.ndarray
        2D array with shape (n_repeats, len(values))
    
    Examples
    --------
    >>> values = np.array([1.0, 2.0, 3.0])
    >>> broadcast_values(values, 3)
    array([[1., 2., 3.],
           [1., 2., 3.],
           [1., 2., 3.]])
    """
    return np.vstack([values] * n_repeats)


def compute_scaling_params(
    df: pd.DataFrame,
    mask: Optional[np.ndarray] = None,
    min_std: float = 1e-12,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute mean and std for scaling, using vectorized operations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    mask : Optional[np.ndarray]
        Boolean mask for subset (e.g., training trials)
    min_std : float
        Minimum standard deviation to avoid division by zero
    
    Returns
    -------
    tuple[dict, dict]
        (means, stds) dictionaries mapping column names to values
    """
    if mask is not None:
        subset = df.iloc[mask]
    else:
        subset = df
    
    numeric = subset.apply(pd.to_numeric, errors="coerce")
    
    means = numeric.mean().to_dict()
    stds = (
        numeric.std()
        .replace(0, min_std)
        .replace([np.inf, -np.inf], min_std)
        .to_dict()
    )
    
    return means, stds


def apply_scaling(
    df: pd.DataFrame,
    means: dict[str, float],
    stds: dict[str, float],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply scaling parameters to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to scale
    means : dict[str, float]
        Mean values per column
    stds : dict[str, float]
        Standard deviation values per column
    inplace : bool
        If True, modifies DataFrame in place
    
    Returns
    -------
    pd.DataFrame
        Scaled DataFrame
    """
    target_df = df if inplace else df.copy()
    
    for col in means:
        if col in target_df.columns and col in stds:
            numeric = pd.to_numeric(target_df[col], errors="coerce")
            target_df[col] = (numeric - means[col]) / stds[col]
    
    return target_df
