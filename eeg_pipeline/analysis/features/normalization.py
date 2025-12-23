"""
Feature Normalization Utilities
================================

Standardized normalization methods for EEG features:
- Z-score normalization
- Robust normalization (median/MAD)
- Min-max scaling
- Rank transformation
- Condition-aware normalization
- Run-wise normalization (for scanner drift)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple, Literal

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_feature_constant


NormMethod = Literal["zscore", "robust", "minmax", "rank", "log", "none"]


def zscore_normalize(
    values: np.ndarray,
    *,
    reference_values: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    Z-score normalization: (x - mean) / std.
    
    Parameters
    ----------
    values : np.ndarray
        Values to normalize
    reference_values : Optional[np.ndarray]
        Values to compute mean/std from (for train/test split)
    epsilon : float
        Minimum std to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Normalized values
    """
    ref = reference_values if reference_values is not None else values
    ref_finite = ref[np.isfinite(ref)]
    
    if len(ref_finite) < 2:
        return np.full_like(values, np.nan, dtype=float)
    
    mean = np.mean(ref_finite)
    std = np.std(ref_finite, ddof=1)
    if epsilon is None:
        # Get epsilon from config if available
        try:
            from eeg_pipeline.utils.config.loader import load_config, get_config_value
            config = load_config()
            epsilon = get_config_value(config, "feature_engineering.constants.epsilon_normalization", 1e-12)
        except Exception:
            epsilon = 1e-12  # Fallback default epsilon
    std = max(std, epsilon)
    
    return (values - mean) / std


def robust_normalize(
    values: np.ndarray,
    *,
    reference_values: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    Robust normalization using median and MAD.
    
    More robust to outliers than z-score.
    
    Parameters
    ----------
    values : np.ndarray
        Values to normalize
    reference_values : Optional[np.ndarray]
        Values to compute median/MAD from
    epsilon : float
        Minimum MAD to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Normalized values
    """
    ref = reference_values if reference_values is not None else values
    ref_finite = ref[np.isfinite(ref)]
    
    if len(ref_finite) < 2:
        return np.full_like(values, np.nan, dtype=float)
    
    median = np.median(ref_finite)
    mad = stats.median_abs_deviation(ref_finite, scale="normal")
    if epsilon is None:
        epsilon = 1e-12  # Default epsilon
    mad = max(mad, epsilon)
    
    return (values - median) / mad


def minmax_normalize(
    values: np.ndarray,
    *,
    reference_values: Optional[np.ndarray] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Min-max normalization to a specified range.
    
    Parameters
    ----------
    values : np.ndarray
        Values to normalize
    reference_values : Optional[np.ndarray]
        Values to compute min/max from
    feature_range : Tuple[float, float]
        Target range (min, max)
    
    Returns
    -------
    np.ndarray
        Normalized values
    """
    ref = reference_values if reference_values is not None else values
    ref_finite = ref[np.isfinite(ref)]
    
    if len(ref_finite) < 2:
        return np.full_like(values, np.nan, dtype=float)
    
    min_val = np.min(ref_finite)
    max_val = np.max(ref_finite)
    
    if max_val == min_val:
        return np.full_like(values, feature_range[0], dtype=float)
    
    scaled = (values - min_val) / (max_val - min_val)
    return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]


def rank_normalize(
    values: np.ndarray,
    *,
    method: str = "average",
) -> np.ndarray:
    """
    Rank transformation normalization.
    
    Converts values to their ranks, robust to outliers and
    produces uniform distribution.
    
    Parameters
    ----------
    values : np.ndarray
        Values to normalize
    method : str
        Ranking method ("average", "min", "max", "dense", "ordinal")
    
    Returns
    -------
    np.ndarray
        Rank-normalized values (0 to 1)
    """
    finite_mask = np.isfinite(values)
    result = np.full_like(values, np.nan, dtype=float)
    
    if np.sum(finite_mask) < 2:
        return result
    
    finite_values = values[finite_mask]
    ranks = stats.rankdata(finite_values, method=method)
    
    # Normalize ranks to 0-1
    n = len(ranks)
    result[finite_mask] = (ranks - 1) / (n - 1) if n > 1 else 0.5
    
    return result


def log_normalize(
    values: np.ndarray,
    *,
    epsilon: Optional[float] = None,
    base: float = np.e,
) -> np.ndarray:
    """
    Log transformation for positively skewed data.
    
    Parameters
    ----------
    values : np.ndarray
        Values to transform (should be positive)
    epsilon : float
        Small value to add before log
    base : float
        Log base (e for natural log, 10 for log10)
    
    Returns
    -------
    np.ndarray
        Log-transformed values
    """
    if epsilon is None:
        epsilon = 1e-12  # Default epsilon
    safe_values = np.maximum(values, epsilon)
    if base == np.e:
        return np.log(safe_values)
    elif base == 10:
        return np.log10(safe_values)
    else:
        return np.log(safe_values) / np.log(base)


def normalize_features(
    df: pd.DataFrame,
    method: NormMethod = "zscore",
    *,
    reference: str = "all",
    condition_column: Optional[str] = "condition",
    run_column: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Normalize all numeric features in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to normalize
    method : NormMethod
        Normalization method: "zscore", "robust", "minmax", "rank", "log", "none"
    reference : str
        Reference for computing normalization parameters:
        - "all": Use all data
        - "condition": Normalize within each condition separately
        - "run": Normalize within each run (for scanner drift)
    condition_column : Optional[str]
        Column name for condition (if reference="condition")
    run_column : Optional[str]
        Column name for run (if reference="run")
    exclude_columns : Optional[List[str]]
        Columns to exclude from normalization
    reference_df : Optional[pd.DataFrame]
        Separate DataFrame to compute normalization parameters from
        (e.g., training set for train/test split)
    
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame
    """
    if method == "none":
        return df.copy()
    
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "run", "run_id"]
    
    result = df.copy()
    
    # Get numeric columns
    numeric_cols = [c for c in df.columns 
                   if c not in exclude_columns 
                   and pd.api.types.is_numeric_dtype(df[c])]
    
    if not numeric_cols:
        return result
    
    # Select normalization function
    norm_funcs = {
        "zscore": zscore_normalize,
        "robust": robust_normalize,
        "minmax": minmax_normalize,
        "rank": rank_normalize,
        "log": log_normalize,
    }
    norm_fn = norm_funcs.get(method, zscore_normalize)
    
    if reference == "all":
        # Normalize using all data
        for col in numeric_cols:
            values = df[col].to_numpy(dtype=float)
            ref_values = None
            if reference_df is not None and col in reference_df.columns:
                ref_values = reference_df[col].to_numpy(dtype=float)
            
            if method == "rank":
                result[col] = norm_fn(values)
            elif method == "log":
                result[col] = norm_fn(values)
            else:
                result[col] = norm_fn(values, reference_values=ref_values)
    
    elif reference == "condition" and condition_column and condition_column in df.columns:
        # Normalize within each condition
        conditions = df[condition_column].unique()
        
        for col in numeric_cols:
            for cond in conditions:
                mask = df[condition_column] == cond
                values = df.loc[mask, col].to_numpy(dtype=float)
                
                if method == "rank":
                    result.loc[mask, col] = norm_fn(values)
                elif method == "log":
                    result.loc[mask, col] = norm_fn(values)
                else:
                    result.loc[mask, col] = norm_fn(values)
    
    elif reference == "run" and run_column and run_column in df.columns:
        # Normalize within each run (for scanner drift)
        runs = df[run_column].unique()
        
        for col in numeric_cols:
            for run in runs:
                mask = df[run_column] == run
                values = df.loc[mask, col].to_numpy(dtype=float)
                
                if method == "rank":
                    result.loc[mask, col] = norm_fn(values)
                elif method == "log":
                    result.loc[mask, col] = norm_fn(values)
                else:
                    result.loc[mask, col] = norm_fn(values)
    
    else:
        # Fallback to all
        for col in numeric_cols:
            values = df[col].to_numpy(dtype=float)
            if method == "rank":
                result[col] = norm_fn(values)
            elif method == "log":
                result[col] = norm_fn(values)
            else:
                result[col] = norm_fn(values)
    
    return result


def normalize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: NormMethod = "zscore",
    *,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize train and test DataFrames using training statistics only.
    
    This prevents data leakage from test set into normalization.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    method : NormMethod
        Normalization method
    exclude_columns : Optional[List[str]]
        Columns to exclude
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Normalized (train, test) DataFrames
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "run", "run_id"]
    
    # Normalize train
    train_norm = normalize_features(
        train_df, method=method, reference="all", exclude_columns=exclude_columns
    )
    
    # Normalize test using train statistics
    test_norm = normalize_features(
        test_df, method=method, reference="all", 
        exclude_columns=exclude_columns, reference_df=train_df
    )
    
    return train_norm, test_norm


class FeatureNormalizer:
    """
    Fitted normalizer that can be applied to new data.
    
    Use this when you need to normalize new data with parameters
    learned from training data.
    """
    
    def __init__(
        self,
        method: NormMethod = "zscore",
        exclude_columns: Optional[List[str]] = None,
    ):
        self.method = method
        self.exclude_columns = exclude_columns or [
            "condition", "epoch", "trial", "subject", "run", "run_id"
        ]
        self.params_: Dict[str, Dict[str, float]] = {}
        self.fitted_ = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """Fit normalizer to training data."""
        numeric_cols = [c for c in df.columns 
                       if c not in self.exclude_columns 
                       and pd.api.types.is_numeric_dtype(df[c])]
        
        for col in numeric_cols:
            values = df[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            
            if len(finite) < 2:
                self.params_[col] = {"valid": False}
                continue
            
            if self.method == "zscore":
                self.params_[col] = {
                    "valid": True,
                    "mean": float(np.mean(finite)),
                    "std": float(max(np.std(finite, ddof=1), 1e-12)),  # Default epsilon_std
                }
            elif self.method == "robust":
                self.params_[col] = {
                    "valid": True,
                    "median": float(np.median(finite)),
                    "mad": float(max(stats.median_abs_deviation(finite, scale="normal"), 1e-12)),  # Default epsilon_std
                }
            elif self.method == "minmax":
                self.params_[col] = {
                    "valid": True,
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                }
            else:
                self.params_[col] = {"valid": True}
        
        self.fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if not self.fitted_:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        result = df.copy()
        
        for col, params in self.params_.items():
            if col not in df.columns:
                continue
            if not params.get("valid", False):
                result[col] = np.nan
                continue
            
            values = df[col].to_numpy(dtype=float)
            
            if self.method == "zscore":
                result[col] = (values - params["mean"]) / params["std"]
            elif self.method == "robust":
                result[col] = (values - params["median"]) / params["mad"]
            elif self.method == "minmax":
                range_val = params["max"] - params["min"]
                if range_val > 0:
                    result[col] = (values - params["min"]) / range_val
                else:
                    result[col] = 0.0
            elif self.method == "rank":
                result[col] = rank_normalize(values)
            elif self.method == "log":
                result[col] = log_normalize(values)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

