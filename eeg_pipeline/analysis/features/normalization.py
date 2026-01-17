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

from typing import Optional, List, Dict, Tuple, Literal, Any

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_feature_constant, load_config


NormMethod = Literal["zscore", "robust", "minmax", "rank", "log", "none"]

DEFAULT_EPSILON = 1e-12
MIN_SAMPLES_FOR_NORMALIZATION = 2
DEFAULT_EXCLUDE_COLUMNS = ["condition", "epoch", "trial", "subject", "run", "run_id"]


def _get_epsilon(config: Optional[Any] = None) -> float:
    """Get epsilon constant from config with fallback."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            return DEFAULT_EPSILON
    
    try:
        return float(get_feature_constant(config, "EPSILON_STD", DEFAULT_EPSILON))
    except Exception:
        return DEFAULT_EPSILON


def _extract_finite_values(
    values: np.ndarray,
    reference_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract finite values from array or reference array."""
    source = reference_values if reference_values is not None else values
    return source[np.isfinite(source)]


def _validate_normalization_input(values: np.ndarray) -> bool:
    """Validate that array has sufficient finite values for normalization."""
    finite_values = _extract_finite_values(values)
    return len(finite_values) >= MIN_SAMPLES_FOR_NORMALIZATION


def zscore_normalize(
    values: np.ndarray,
    *,
    reference_values: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    config: Optional[Any] = None,
) -> np.ndarray:
    """
    Z-score normalization: (x - mean) / std.
    
    Parameters
    ----------
    values : np.ndarray
        Values to normalize
    reference_values : Optional[np.ndarray]
        Values to compute mean/std from (for train/test split)
    epsilon : Optional[float]
        Minimum std to avoid division by zero. If None, uses config.
    config : Optional[Any]
        Configuration object for epsilon lookup
    
    Returns
    -------
    np.ndarray
        Normalized values
    """
    if not _validate_normalization_input(values):
        return np.full_like(values, np.nan, dtype=float)
    
    reference_finite = _extract_finite_values(values, reference_values)
    mean = np.mean(reference_finite)
    std = np.std(reference_finite, ddof=1)
    
    if epsilon is None:
        epsilon = _get_epsilon(config)
    
    std = max(std, epsilon)
    return (values - mean) / std


def robust_normalize(
    values: np.ndarray,
    *,
    reference_values: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    config: Optional[Any] = None,
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
    epsilon : Optional[float]
        Minimum MAD to avoid division by zero. If None, uses config.
    config : Optional[Any]
        Configuration object for epsilon lookup
    
    Returns
    -------
    np.ndarray
        Normalized values
    """
    if not _validate_normalization_input(values):
        return np.full_like(values, np.nan, dtype=float)
    
    reference_finite = _extract_finite_values(values, reference_values)
    median = np.median(reference_finite)
    mad = stats.median_abs_deviation(reference_finite, scale="normal")
    
    if epsilon is None:
        epsilon = _get_epsilon(config)
    
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
    if not _validate_normalization_input(values):
        return np.full_like(values, np.nan, dtype=float)
    
    reference_finite = _extract_finite_values(values, reference_values)
    min_val = np.min(reference_finite)
    max_val = np.max(reference_finite)
    
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
    
    finite_count = np.sum(finite_mask)
    if finite_count < MIN_SAMPLES_FOR_NORMALIZATION:
        return result
    
    finite_values = values[finite_mask]
    ranks = stats.rankdata(finite_values, method=method)
    
    normalized_ranks = (ranks - 1) / (finite_count - 1) if finite_count > 1 else 0.5
    result[finite_mask] = normalized_ranks
    
    return result


def log_normalize(
    values: np.ndarray,
    *,
    epsilon: Optional[float] = None,
    base: float = np.e,
    config: Optional[Any] = None,
) -> np.ndarray:
    """
    Log transformation for positively skewed data.
    
    Parameters
    ----------
    values : np.ndarray
        Values to transform (should be positive)
    epsilon : Optional[float]
        Small value to add before log. If None, uses config.
    base : float
        Log base (e for natural log, 10 for log10)
    config : Optional[Any]
        Configuration object for epsilon lookup
    
    Returns
    -------
    np.ndarray
        Log-transformed values
    """
    if epsilon is None:
        epsilon = _get_epsilon(config)
    
    safe_values = np.maximum(values, epsilon)
    
    if base == np.e:
        return np.log(safe_values)
    if base == 10:
        return np.log10(safe_values)
    
    return np.log(safe_values) / np.log(base)


def _get_numeric_columns(
    df: pd.DataFrame,
    exclude_columns: List[str],
) -> List[str]:
    """Extract numeric columns excluding specified columns."""
    return [
        col
        for col in df.columns
        if col not in exclude_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def _get_normalization_function(method: NormMethod):
    """Get normalization function for given method."""
    norm_functions = {
        "zscore": zscore_normalize,
        "robust": robust_normalize,
        "minmax": minmax_normalize,
        "rank": rank_normalize,
        "log": log_normalize,
    }
    return norm_functions.get(method, zscore_normalize)


def _normalize_column_all_data(
    df: pd.DataFrame,
    col: str,
    norm_fn,
    method: NormMethod,
    reference_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
) -> np.ndarray:
    """Normalize a column using all data."""
    values = df[col].to_numpy(dtype=float)
    
    if method in ("rank", "log"):
        return norm_fn(values, config=config)
    
    ref_values = None
    if reference_df is not None and col in reference_df.columns:
        ref_values = reference_df[col].to_numpy(dtype=float)
    
    return norm_fn(values, reference_values=ref_values, config=config)


def _normalize_column_by_group(
    df: pd.DataFrame,
    col: str,
    group_column: str,
    norm_fn,
    config: Optional[Any] = None,
) -> np.ndarray:
    """Normalize a column within each group."""
    result = np.full(len(df), np.nan, dtype=float)
    groups = df[group_column].unique()
    
    for group in groups:
        mask = df[group_column] == group
        group_values = df.loc[mask, col].to_numpy(dtype=float)
        normalized = norm_fn(group_values, config=config)
        result[mask] = normalized
    
    return result


def normalize_features(
    df: pd.DataFrame,
    method: NormMethod = "zscore",
    *,
    reference: str = "all",
    condition_column: Optional[str] = "condition",
    run_column: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
    reference_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
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
    config : Optional[Any]
        Configuration object for epsilon lookup
    
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame
    """
    if method == "none":
        return df.copy()
    
    if exclude_columns is None:
        exclude_columns = DEFAULT_EXCLUDE_COLUMNS.copy()
    
    numeric_columns = _get_numeric_columns(df, exclude_columns)
    if not numeric_columns:
        return df.copy()
    
    norm_fn = _get_normalization_function(method)
    result = df.copy()
    
    if reference == "all":
        for col in numeric_columns:
            result[col] = _normalize_column_all_data(
                df, col, norm_fn, method, reference_df, config
            )
    elif reference == "condition" and condition_column and condition_column in df.columns:
        for col in numeric_columns:
            result[col] = _normalize_column_by_group(
                df, col, condition_column, norm_fn, config
            )
    elif reference == "run" and run_column and run_column in df.columns:
        for col in numeric_columns:
            result[col] = _normalize_column_by_group(
                df, col, run_column, norm_fn, config
            )
    else:
        for col in numeric_columns:
            result[col] = _normalize_column_all_data(
                df, col, norm_fn, method, reference_df, config
            )
    
    return result


def normalize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: NormMethod = "zscore",
    *,
    exclude_columns: Optional[List[str]] = None,
    config: Optional[Any] = None,
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
    config : Optional[Any]
        Configuration object for epsilon lookup
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Normalized (train, test) DataFrames
    """
    if exclude_columns is None:
        exclude_columns = DEFAULT_EXCLUDE_COLUMNS.copy()
    
    train_norm = normalize_features(
        train_df,
        method=method,
        reference="all",
        exclude_columns=exclude_columns,
        config=config,
    )
    
    test_norm = normalize_features(
        test_df,
        method=method,
        reference="all",
        exclude_columns=exclude_columns,
        reference_df=train_df,
        config=config,
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
        config: Optional[Any] = None,
    ):
        self.method = method
        self.exclude_columns = exclude_columns or DEFAULT_EXCLUDE_COLUMNS.copy()
        self.config = config
        self.params_: Dict[str, Dict[str, float]] = {}
        self.fitted_ = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """Fit normalizer to training data."""
        numeric_columns = _get_numeric_columns(df, self.exclude_columns)
        epsilon = _get_epsilon(self.config)
        
        for col in numeric_columns:
            values = df[col].to_numpy(dtype=float)
            finite_values = _extract_finite_values(values)
            
            if not _validate_normalization_input(values):
                self.params_[col] = {"valid": False}
                continue
            
            if self.method == "zscore":
                self.params_[col] = {
                    "valid": True,
                    "mean": float(np.mean(finite_values)),
                    "std": float(max(np.std(finite_values, ddof=1), epsilon)),
                }
            elif self.method == "robust":
                self.params_[col] = {
                    "valid": True,
                    "median": float(np.median(finite_values)),
                    "mad": float(max(stats.median_abs_deviation(finite_values, scale="normal"), epsilon)),
                }
            elif self.method == "minmax":
                self.params_[col] = {
                    "valid": True,
                    "min": float(np.min(finite_values)),
                    "max": float(np.max(finite_values)),
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
                result[col] = log_normalize(values, config=self.config)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
