"""
Feature Reliability Assessment
==============================

Methods for assessing the reliability (reproducibility) of EEG features:
- Split-half reliability
- Odd-even reliability
- Bootstrap reliability
- Intraclass Correlation Coefficient (ICC)

Reliable features are essential for robust ML models and scientific reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.analysis.features.core import (
    ConfigLike,
    ProgressCallback,
    null_progress,
)


@dataclass
class ReliabilityResult:
    """Result of reliability computation for a single feature."""
    name: str
    reliability: float  # Correlation/ICC value
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n_samples: int  # Number of samples used
    method: str  # Method used
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if reliability meets threshold."""
        return self.reliability >= threshold
    
    def is_good(self, threshold: float = 0.8) -> bool:
        """Check if reliability is good."""
        return self.reliability >= threshold
    
    def is_excellent(self, threshold: float = 0.9) -> bool:
        """Check if reliability is excellent."""
        return self.reliability >= threshold


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Compute Pearson correlation with 95% CI."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    n = len(x_clean)
    if n < 3:
        return np.nan, np.nan, np.nan
    
    r, _ = stats.pearsonr(x_clean, y_clean)
    
    # Fisher z-transformation for CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return float(r), float(ci_lower), float(ci_upper)


def _spearman_brown_correction(r: float) -> float:
    """Apply Spearman-Brown prophecy formula for split-half."""
    if np.isnan(r) or r <= -1:
        return np.nan
    return (2 * r) / (1 + r)


def compute_split_half_reliability(
    epochs_data: np.ndarray,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    *,
    n_iterations: int = 100,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute split-half reliability with Spearman-Brown correction.
    
    Randomly splits epochs into two halves, extracts features from each,
    and correlates them. Repeats multiple times and averages.
    
    Parameters
    ----------
    epochs_data : np.ndarray
        Shape (n_epochs, n_channels, n_times)
    feature_extractor : Callable
        Function that takes epochs array and returns feature values
    n_iterations : int
        Number of random splits
    random_state : int
        Random seed
    
    Returns
    -------
    Tuple[float, float, float]
        (reliability, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_state)
    n_epochs = epochs_data.shape[0]
    
    if n_epochs < 4:
        return np.nan, np.nan, np.nan
    
    correlations = []
    
    for _ in range(n_iterations):
        # Random split
        indices = rng.permutation(n_epochs)
        half = n_epochs // 2
        idx1 = indices[:half]
        idx2 = indices[half:2*half]
        
        # Extract features for each half
        try:
            features1 = feature_extractor(epochs_data[idx1])
            features2 = feature_extractor(epochs_data[idx2])
        except (ValueError, IndexError):
            continue
        
        if features1.size == 0 or features2.size == 0:
            continue
        
        # Correlate
        r, _, _ = _pearson_correlation(features1, features2)
        if np.isfinite(r):
            correlations.append(r)
    
    if len(correlations) < 5:
        return np.nan, np.nan, np.nan
    
    # Average correlation and apply Spearman-Brown correction
    mean_r = np.mean(correlations)
    reliability = _spearman_brown_correction(mean_r)
    
    # Bootstrap CI
    corrected = [_spearman_brown_correction(r) for r in correlations]
    corrected = [c for c in corrected if np.isfinite(c)]
    
    if len(corrected) < 5:
        return reliability, np.nan, np.nan
    
    ci_lower = float(np.percentile(corrected, 2.5))
    ci_upper = float(np.percentile(corrected, 97.5))
    
    return float(reliability), ci_lower, ci_upper


def compute_odd_even_reliability(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    *,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, ReliabilityResult]:
    """
    Compute odd-even split reliability for DataFrame features.
    
    Splits trials into odd and even numbered, then correlates
    mean features between halves.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with one row per trial/epoch
    feature_columns : Optional[List[str]]
        Columns to compute reliability for (if None, uses all numeric)
    exclude_columns : Optional[List[str]]
        Columns to exclude
    
    Returns
    -------
    Dict[str, ReliabilityResult]
        Reliability results per feature
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "run"]
    
    if feature_columns is None:
        feature_columns = [c for c in df.columns 
                         if c not in exclude_columns
                         and pd.api.types.is_numeric_dtype(df[c])]
    
    n_rows = len(df)
    if n_rows < 4:
        return {}
    
    # Split into odd and even
    odd_mask = np.arange(n_rows) % 2 == 1
    even_mask = ~odd_mask
    
    results = {}
    
    for col in feature_columns:
        values = df[col].to_numpy(dtype=float)
        
        odd_mean = np.nanmean(values[odd_mask])
        even_mean = np.nanmean(values[even_mask])
        
        # For single-value comparison, use individual values
        odd_vals = values[odd_mask]
        even_vals = values[even_mask]
        
        # Pad to same length
        min_len = min(len(odd_vals), len(even_vals))
        odd_vals = odd_vals[:min_len]
        even_vals = even_vals[:min_len]
        
        r, ci_lower, ci_upper = _pearson_correlation(odd_vals, even_vals)
        reliability = _spearman_brown_correction(r)
        
        # Adjust CI with Spearman-Brown
        ci_lower_sb = _spearman_brown_correction(ci_lower) if np.isfinite(ci_lower) else np.nan
        ci_upper_sb = _spearman_brown_correction(ci_upper) if np.isfinite(ci_upper) else np.nan
        
        results[col] = ReliabilityResult(
            name=col,
            reliability=reliability,
            ci_lower=ci_lower_sb,
            ci_upper=ci_upper_sb,
            n_samples=min_len,
            method="odd_even",
        )
    
    return results


def compute_bootstrap_reliability(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    *,
    n_iterations: int = 1000,
    sample_fraction: float = 0.5,
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
    progress: ProgressCallback = null_progress,
) -> Dict[str, ReliabilityResult]:
    """
    Compute bootstrap reliability estimates.
    
    Repeatedly samples subsets and computes mean features, then
    correlates between bootstrap samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_columns : Optional[List[str]]
        Columns to analyze
    n_iterations : int
        Number of bootstrap iterations
    sample_fraction : float
        Fraction of data to sample each iteration
    random_state : int
        Random seed
    exclude_columns : Optional[List[str]]
        Columns to exclude
    progress : ProgressCallback
        Progress callback
    
    Returns
    -------
    Dict[str, ReliabilityResult]
        Reliability results per feature
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "run"]
    
    if feature_columns is None:
        feature_columns = [c for c in df.columns 
                         if c not in exclude_columns
                         and pd.api.types.is_numeric_dtype(df[c])]
    
    n_rows = len(df)
    sample_size = int(n_rows * sample_fraction)
    
    if sample_size < 2:
        return {}
    
    rng = np.random.default_rng(random_state)
    
    # Generate bootstrap samples
    bootstrap_means: Dict[str, List[float]] = {col: [] for col in feature_columns}
    
    for i in range(n_iterations):
        if i % 100 == 0:
            progress("Bootstrap", i / n_iterations)
        
        idx = rng.choice(n_rows, size=sample_size, replace=True)
        sample = df.iloc[idx]
        
        for col in feature_columns:
            mean_val = np.nanmean(sample[col].to_numpy(dtype=float))
            bootstrap_means[col].append(mean_val)
    
    progress("Computing reliability", 0.9)
    
    results = {}
    
    for col in feature_columns:
        means = np.array(bootstrap_means[col])
        finite_means = means[np.isfinite(means)]
        
        if len(finite_means) < 10:
            results[col] = ReliabilityResult(
                name=col,
                reliability=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                n_samples=len(finite_means),
                method="bootstrap",
            )
            continue
        
        # Split bootstrap samples and correlate
        half = len(finite_means) // 2
        r, ci_lower, ci_upper = _pearson_correlation(
            finite_means[:half], finite_means[half:2*half]
        )
        
        reliability = _spearman_brown_correction(r)
        ci_lower_sb = _spearman_brown_correction(ci_lower) if np.isfinite(ci_lower) else np.nan
        ci_upper_sb = _spearman_brown_correction(ci_upper) if np.isfinite(ci_upper) else np.nan
        
        results[col] = ReliabilityResult(
            name=col,
            reliability=reliability,
            ci_lower=ci_lower_sb,
            ci_upper=ci_upper_sb,
            n_samples=len(finite_means),
            method="bootstrap",
        )
    
    progress("Complete", 1.0)
    
    return results


def compute_icc(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    icc_type: str = "ICC(3,1)",
) -> Tuple[float, float, float]:
    """
    Compute Intraclass Correlation Coefficient.
    
    ICC measures consistency/agreement between measurements.
    
    Parameters
    ----------
    values : np.ndarray
        Measurement values
    groups : np.ndarray
        Group labels (e.g., subject IDs for test-retest)
    icc_type : str
        ICC type: "ICC(1,1)", "ICC(2,1)", "ICC(3,1)"
    
    Returns
    -------
    Tuple[float, float, float]
        (ICC, ci_lower, ci_upper)
    """
    # Clean data
    mask = np.isfinite(values)
    values_clean = values[mask]
    groups_clean = groups[mask]
    
    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        return np.nan, np.nan, np.nan
    
    # Organize into groups
    group_data = []
    for g in unique_groups:
        g_vals = values_clean[groups_clean == g]
        if len(g_vals) > 0:
            group_data.append(g_vals)
    
    if len(group_data) < 2:
        return np.nan, np.nan, np.nan
    
    # Compute variances using ANOVA-like decomposition
    # Between-group variance
    group_means = np.array([np.mean(g) for g in group_data])
    grand_mean = np.mean(values_clean)
    n_per_group = np.array([len(g) for g in group_data])
    
    ss_between = np.sum(n_per_group * (group_means - grand_mean) ** 2)
    df_between = n_groups - 1
    
    # Within-group variance
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in group_data)
    df_within = sum(len(g) - 1 for g in group_data)
    
    if df_between == 0 or df_within == 0:
        return np.nan, np.nan, np.nan
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    # ICC(3,1) - Two-way mixed, consistency
    k = np.mean(n_per_group)  # Average group size
    
    if ms_within == 0:
        icc = 1.0 if ms_between > 0 else np.nan
    else:
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    
    # Confidence interval approximation
    f_value = ms_between / ms_within if ms_within > 0 else np.inf
    
    if np.isfinite(f_value) and f_value > 0:
        # F-based CI
        try:
            f_lower = f_value / stats.f.ppf(0.975, df_between, df_within)
            f_upper = f_value * stats.f.ppf(0.975, df_within, df_between)
            
            ci_lower = (f_lower - 1) / (f_lower + k - 1)
            ci_upper = (f_upper - 1) / (f_upper + k - 1)
        except (ValueError, ZeroDivisionError):
            ci_lower, ci_upper = np.nan, np.nan
    else:
        ci_lower, ci_upper = np.nan, np.nan
    
    return float(icc), float(ci_lower), float(ci_upper)


def compute_feature_reliability(
    df: pd.DataFrame,
    method: str = "odd_even",
    *,
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    n_iterations: int = 100,
    random_state: int = 42,
    progress: ProgressCallback = null_progress,
) -> pd.DataFrame:
    """
    Compute reliability for all features in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    method : str
        Method: "odd_even", "bootstrap"
    feature_columns : Optional[List[str]]
        Columns to analyze
    exclude_columns : Optional[List[str]]
        Columns to exclude
    n_iterations : int
        Number of iterations (for bootstrap)
    random_state : int
        Random seed
    progress : ProgressCallback
        Progress callback
    
    Returns
    -------
    pd.DataFrame
        Reliability results with columns:
        [name, reliability, ci_lower, ci_upper, n_samples, method, is_acceptable, is_good]
    """
    if method == "odd_even":
        results = compute_odd_even_reliability(
            df, feature_columns, exclude_columns=exclude_columns
        )
    elif method == "bootstrap":
        results = compute_bootstrap_reliability(
            df, feature_columns,
            n_iterations=n_iterations,
            random_state=random_state,
            exclude_columns=exclude_columns,
            progress=progress,
        )
    else:
        raise ValueError(f"Unknown reliability method: {method}")
    
    records = []
    for name, result in results.items():
        records.append({
            "name": result.name,
            "reliability": result.reliability,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "n_samples": result.n_samples,
            "method": result.method,
            "is_acceptable": result.is_acceptable(),
            "is_good": result.is_good(),
            "is_excellent": result.is_excellent(),
        })
    
    return pd.DataFrame(records)


def filter_reliable_features(
    df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    threshold: float = 0.7,
    *,
    keep_metadata_cols: bool = True,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter features to keep only those meeting reliability threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original feature DataFrame
    reliability_df : pd.DataFrame
        Reliability results from compute_feature_reliability
    threshold : float
        Minimum reliability to keep
    keep_metadata_cols : bool
        Keep metadata columns
    metadata_cols : Optional[List[str]]
        Metadata columns to keep
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if metadata_cols is None:
        metadata_cols = ["condition", "epoch", "trial", "subject", "run"]
    
    reliable = reliability_df[reliability_df["reliability"] >= threshold]["name"].tolist()
    
    if keep_metadata_cols:
        cols_to_keep = [c for c in metadata_cols if c in df.columns] + reliable
    else:
        cols_to_keep = reliable
    
    return df[cols_to_keep].copy()

