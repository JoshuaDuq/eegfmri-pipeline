"""
Reliability and Validity Statistics
====================================

Functions for assessing measurement reliability and predictive validity:
- ICC (Intraclass Correlation Coefficient)
- Split-half reliability
- Hierarchical FDR correction
- Cross-validated predictive modeling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.types import ProgressCallback, null_progress
from .base import get_subject_seed


###################################################################
# Constants
###################################################################

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_FOR_CORRELATION = 3
MIN_SAMPLES_FOR_SPLIT_HALF = 4
MIN_SAMPLES_PER_SPLIT = 10
MIN_SAMPLES_FOR_POWER = 4
MIN_GROUPS_FOR_ICC = 2
MIN_PIVOT_ROWS_FOR_ICC = 2
MIN_PIVOT_COLS_FOR_ICC = 2
RELIABILITY_THRESHOLD_ACCEPTABLE = 0.7
RELIABILITY_THRESHOLD_GOOD = 0.8
RELIABILITY_THRESHOLD_EXCELLENT = 0.9
DEFAULT_N_SPLITS = 100
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_FOLDS = 5
DEFAULT_N_PERMUTATIONS = 100
DEFAULT_N_BINS = 10
DEFAULT_SAMPLE_FRACTION = 0.5
MIN_CORRELATION_FOR_POWER = 0.001
EFFECTIVELY_INFINITE_N = 999999
DEFAULT_ALPHA_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.9]
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_MAX_DEPTH = 5
MIN_SAMPLES_FOR_CV = 10
MIN_FOLDS = 2
MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF = 20


###################################################################
# Intraclass Correlation Coefficient (ICC)
###################################################################


def _compute_icc_one_way_single(ms_rows: float, ms_within: float, k: int) -> float:
    """Compute ICC(1,1): One-way random, single rater."""
    return (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)


def _compute_icc_two_way_random_single(ms_rows: float, ms_error: float, ms_cols: float, n: int, k: int) -> float:
    """Compute ICC(2,1): Two-way random, single rater."""
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return (ms_rows - ms_error) / denominator


def _compute_icc_two_way_mixed_single(ms_rows: float, ms_error: float, k: int) -> float:
    """Compute ICC(3,1): Two-way mixed, single rater."""
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)


def _compute_icc_one_way_average(ms_rows: float, ms_within: float) -> float:
    """Compute ICC(1,k): One-way random, average of k raters."""
    return (ms_rows - ms_within) / ms_rows


def _compute_icc_two_way_random_average(ms_rows: float, ms_error: float, ms_cols: float, n: int) -> float:
    """Compute ICC(2,k): Two-way random, average of k raters."""
    return (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)


def _compute_icc_two_way_mixed_average(ms_rows: float, ms_error: float) -> float:
    """Compute ICC(3,k): Two-way mixed, average of k raters."""
    return (ms_rows - ms_error) / ms_rows


def _compute_icc_confidence_intervals(
    ms_rows: float,
    ms_error: float,
    n: int,
    k: int,
) -> Tuple[float, float]:
    """Compute 95% confidence intervals for ICC using F-distribution."""
    if ms_error <= 0:
        return np.nan, np.nan
    
    f_value = ms_rows / ms_error
    if not (np.isfinite(f_value) and f_value > 0):
        return np.nan, np.nan
    
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_critical_upper = stats.f.ppf(0.975, df1, df2)
    f_critical_lower = stats.f.ppf(0.975, df2, df1)
    
    f_low = f_value / f_critical_upper
    f_high = f_value * f_critical_lower
    
    ci_low = (f_low - 1) / (f_low + k - 1)
    ci_high = (f_high - 1) / (f_high + k - 1)
    
    return float(ci_low), float(ci_high)


def compute_icc(
    data: np.ndarray,
    icc_type: str = "ICC(2,1)",
) -> Tuple[float, float, float]:
    """Compute Intraclass Correlation Coefficient.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_subjects, n_raters/n_sessions).
        Each row is a subject, each column is a rater/session.
    icc_type : str
        Type of ICC to compute:
        - "ICC(1,1)": One-way random, single rater
        - "ICC(2,1)": Two-way random, single rater (default)
        - "ICC(3,1)": Two-way mixed, single rater
        - "ICC(1,k)": One-way random, average of k raters
        - "ICC(2,k)": Two-way random, average of k raters
        - "ICC(3,k)": Two-way mixed, average of k raters
    
    Returns
    -------
    icc : float
        ICC value
    ci_low : float
        Lower 95% CI bound
    ci_high : float
        Upper 95% CI bound
    """
    data = np.asarray(data)
    if data.ndim != 2:
        return np.nan, np.nan, np.nan
    
    n_subjects, n_raters = data.shape
    if n_subjects < 2 or n_raters < 2:
        return np.nan, np.nan, np.nan
    
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = n_raters * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n_subjects * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    
    ms_rows = ss_rows / (n_subjects - 1)
    ms_cols = ss_cols / (n_raters - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
    
    icc_type_upper = icc_type.upper()
    if icc_type_upper in ["ICC(1,1)", "ICC1"]:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_single(ms_rows, ms_within, n_raters)
    elif icc_type_upper in ["ICC(2,1)", "ICC2"]:
        icc = _compute_icc_two_way_random_single(ms_rows, ms_error, ms_cols, n_subjects, n_raters)
    elif icc_type_upper in ["ICC(3,1)", "ICC3"]:
        icc = _compute_icc_two_way_mixed_single(ms_rows, ms_error, n_raters)
    elif icc_type_upper in ["ICC(1,K)", "ICC1K"]:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_average(ms_rows, ms_within)
    elif icc_type_upper in ["ICC(2,K)", "ICC2K"]:
        icc = _compute_icc_two_way_random_average(ms_rows, ms_error, ms_cols, n_subjects)
    elif icc_type_upper in ["ICC(3,K)", "ICC3K"]:
        icc = _compute_icc_two_way_mixed_average(ms_rows, ms_error)
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")
    
    ci_low, ci_high = _compute_icc_confidence_intervals(ms_rows, ms_error, n_subjects, n_raters)
    icc_clipped = float(np.clip(icc, -1, 1))
    
    return icc_clipped, ci_low, ci_high


def _apply_spearman_brown(r: float) -> float:
    """Apply Spearman-Brown prophecy formula."""
    if r <= -1 or not np.isfinite(r):
        return np.nan
    return (2 * r) / (1 + r)


def compute_split_half_reliability(
    data: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Compute split-half reliability with Spearman-Brown correction.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_trials, n_features) or 1D array of values.
    n_splits : int
        Number of random splits to average over.
    method : str
        Correlation method ('spearman' or 'pearson').
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    reliability : float
        Spearman-Brown corrected reliability coefficient.
    ci_low : float
        Lower 95% CI bound.
    ci_high : float
        Upper 95% CI bound.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_trials = data.shape[0]
    if n_trials < MIN_SAMPLES_FOR_SPLIT_HALF:
        return np.nan, np.nan, np.nan
    
    correlations = []
    half_size = n_trials // 2
    
    for _ in range(n_splits):
        indices = rng.permutation(n_trials)
        half1_indices = indices[:half_size]
        half2_indices = indices[half_size:2 * half_size]
        
        half1_means = data[half1_indices].mean(axis=0)
        half2_means = data[half2_indices].mean(axis=0)
        
        from .correlation import compute_correlation
        if len(half1_means) == 1:
            half1_values = data[half1_indices, 0]
            half2_values = data[half2_indices, 0]
            r, _ = compute_correlation(half1_values, half2_values, method)
        else:
            r, _ = compute_correlation(half1_means, half2_means, method)
        r = r if np.isfinite(r) else np.nan
        
        if np.isfinite(r):
            correlations.append(r)
    
    if not correlations:
        return np.nan, np.nan, np.nan
    
    mean_correlation = np.mean(correlations)
    reliability = _apply_spearman_brown(mean_correlation)
    
    boot_reliabilities = [
        _apply_spearman_brown(r) for r in correlations
        if np.isfinite(_apply_spearman_brown(r))
    ]
    
    min_samples_for_ci = 10
    if len(boot_reliabilities) > min_samples_for_ci:
        ci_low = float(np.percentile(boot_reliabilities, 2.5))
        ci_high = float(np.percentile(boot_reliabilities, 97.5))
    else:
        ci_low, ci_high = np.nan, np.nan
    
    return float(reliability), ci_low, ci_high


def compute_feature_reliability(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    groupby_col: str = "trial",
    session_col: Optional[str] = None,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Compute reliability metrics for each feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with features and values.
    feature_col : str
        Column containing feature names.
    value_col : str
        Column containing feature values.
    groupby_col : str
        Column to group by (e.g., 'trial', 'block').
    session_col : str, optional
        Column for session/run (for ICC).
    min_observations : int
        Minimum observations required.
    
    Returns
    -------
    pd.DataFrame
        Reliability metrics per feature.
    """
    results = []
    
    for feature in df[feature_col].unique():
        feat_df = df[df[feature_col] == feature]
        
        if len(feat_df) < min_observations:
            continue
        
        values = feat_df[value_col].values
        
        # Split-half reliability
        sh_rel, sh_low, sh_high = compute_split_half_reliability(values)
        
        result = {
            "feature": feature,
            "n": len(values),
            "split_half_reliability": sh_rel,
            "sh_ci_low": sh_low,
            "sh_ci_high": sh_high,
        }
        
        # ICC if session info available
        if session_col and session_col in feat_df.columns:
            sessions = feat_df[session_col].unique()
            if len(sessions) >= 2:
                # Reshape for ICC
                pivot = feat_df.pivot_table(
                    index=groupby_col, 
                    columns=session_col, 
                    values=value_col,
                    aggfunc="mean"
                ).dropna()
                
                if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
                    icc, icc_low, icc_high = compute_icc(pivot.values)
                    result["icc"] = icc
                    result["icc_ci_low"] = icc_low
                    result["icc_ci_high"] = icc_high
        
        results.append(result)
    
    return pd.DataFrame(results)


###################################################################
# Hierarchical FDR Correction
###################################################################


def hierarchical_fdr_dict(
    p_values: Dict[str, np.ndarray],
    alpha: float = DEFAULT_ALPHA,
    method: str = "bh",
) -> Dict[str, Dict[str, Any]]:
    """Apply hierarchical FDR correction across multiple families (dict API).
    
    Two-stage procedure:
    1. Apply FDR within each family
    2. Apply FDR across family-level summary statistics
    
    Parameters
    ----------
    p_values : Dict[str, np.ndarray]
        Dictionary mapping family names to arrays of p-values.
    alpha : float
        FDR threshold.
    method : str
        FDR method ('bh' for Benjamini-Hochberg).
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results per family with:
        - q_values: FDR-corrected q-values within family
        - reject: Boolean rejection mask within family
        - q_global: Global q-value for the family
        - reject_global: Whether family passes global FDR
        - n_tests: Number of tests in family
        - n_reject: Number of rejections within family
    """
    from eeg_pipeline.utils.analysis.stats import fdr_bh
    
    results = {}
    family_min_p = []
    family_names = []
    
    # Stage 1: Within-family FDR
    for family_name, p_arr in p_values.items():
        p_arr = np.asarray(p_arr)
        valid_mask = np.isfinite(p_arr)
        
        if not valid_mask.any():
            results[family_name] = {
                "q_values": np.full_like(p_arr, np.nan),
                "reject": np.zeros_like(p_arr, dtype=bool),
                "q_global": np.nan,
                "reject_global": False,
                "n_tests": 0,
                "n_reject": 0,
            }
            continue
        
        # Apply BH-FDR within family
        q_arr = np.full_like(p_arr, np.nan)
        q_arr[valid_mask] = fdr_bh(p_arr[valid_mask], alpha=alpha)
        reject = q_arr < alpha
        
        results[family_name] = {
            "q_values": q_arr,
            "reject": reject,
            "n_tests": int(valid_mask.sum()),
            "n_reject": int(reject.sum()),
        }
        
        # Collect minimum p-value per family for global correction
        min_p = np.nanmin(p_arr[valid_mask])
        family_min_p.append(min_p)
        family_names.append(family_name)
    
    # Stage 2: Global FDR across families
    if family_min_p:
        family_min_p = np.array(family_min_p)
        global_q = fdr_bh(family_min_p, alpha=alpha)
        global_reject = global_q < alpha
        
        for i, family_name in enumerate(family_names):
            results[family_name]["q_global"] = float(global_q[i])
            results[family_name]["reject_global"] = bool(global_reject[i])
    
    return results


def compute_hierarchical_fdr_summary(
    stats_dir,
    alpha: float = DEFAULT_ALPHA,
    config: Optional[Any] = None,
    include_glob: Union[str, Iterable[str]] = "corr_stats_*.tsv",
) -> pd.DataFrame:
    """Compute hierarchical FDR summary from stats directory.
    
    Groups tests by analysis type and applies two-stage FDR.
    
    Parameters
    ----------
    stats_dir : Path
        Directory containing stats TSV files.
    alpha : float
        FDR threshold.
    config : Any, optional
        Pipeline configuration.
    include_glob : str or Iterable[str]
        Glob pattern(s) for files to include.
    
    Returns
    -------
    pd.DataFrame
        Summary with hierarchical FDR results.
    """
    from pathlib import Path
    from eeg_pipeline.infra.tsv import read_tsv
    from eeg_pipeline.utils.analysis.stats.fdr import infer_fdr_family, select_p_column_for_fdr
    
    stats_dir = Path(stats_dir)
    
    if isinstance(include_glob, str):
        files = list(stats_dir.glob(include_glob))
    else:
        files = []
        for pat in include_glob:
            files.extend(list(stats_dir.glob(pat)))
        seen = set()
        files = [f for f in files if not (f in seen or seen.add(f))]

    def _extract_feature_family(family: str) -> str:
        if "|features:" in family:
            return str(family.split("|features:", 1)[1]).strip()
        return str(family)

    # Group files by feature family, using the same inference as apply_global_fdr
    analysis_groups: Dict[str, List[Path]] = {}
    for fpath in files:
        df = read_tsv(fpath)
        if df is None or df.empty:
            continue

        family = infer_fdr_family(fpath, df)
        group_name = _extract_feature_family(family)
        analysis_groups.setdefault(group_name, []).append(fpath)
    
    # Collect p-values by group
    p_by_group = {}
    file_refs = {}
    
    for group_name, files in analysis_groups.items():
        all_p = []
        refs = []
        
        for fpath in files:
            df = read_tsv(fpath)
            if df is None or df.empty:
                continue
            
            p_col = select_p_column_for_fdr(df)
            if p_col is None:
                continue
            
            p_vals = pd.to_numeric(df[p_col], errors="coerce").values
            for i, p in enumerate(p_vals):
                if np.isfinite(p):
                    all_p.append(p)
                    refs.append((fpath, i))
        
        if all_p:
            p_by_group[group_name] = np.array(all_p)
            file_refs[group_name] = refs
    
    # Apply hierarchical FDR
    if not p_by_group:
        return pd.DataFrame()
    
    hier_results = hierarchical_fdr_dict(p_by_group, alpha=alpha)
    
    # Build summary dataframe
    summary_rows = []
    for group_name, results in hier_results.items():
        summary_rows.append({
            "analysis_type": group_name,
            "n_tests": results["n_tests"],
            "n_reject_within": results["n_reject"],
            "pct_reject_within": 100 * results["n_reject"] / max(results["n_tests"], 1),
            "q_global": results.get("q_global", np.nan),
            "reject_global": results.get("reject_global", False),
        })
    
    return pd.DataFrame(summary_rows)


###################################################################
# Feature-Extractor-Based Split-Half Reliability
###################################################################


def compute_correlation_split_half_reliability(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_splits: int = DEFAULT_N_SPLITS,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute split-half reliability for a single correlation with Spearman-Brown correction.
    
    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable (e.g., ratings)
    method : str
        Correlation method ('spearman' or 'pearson')
    n_splits : int
        Number of random splits
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    float
        Spearman-Brown corrected reliability
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid_mask.sum())
    
    if n_valid < MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF:
        return np.nan
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    indices = np.arange(n_valid)
    
    from .correlation import compute_correlation
    
    correlations = []
    half_size = n_valid // 2
    
    for _ in range(n_splits):
        rng.shuffle(indices)
        idx1 = indices[:half_size]
        idx2 = indices[half_size:2 * half_size]
        
        r1, _ = compute_correlation(x_valid[idx1], y_valid[idx1], method)
        r2, _ = compute_correlation(x_valid[idx2], y_valid[idx2], method)
        r1 = r1 if np.isfinite(r1) else np.nan
        r2 = r2 if np.isfinite(r2) else np.nan
        
        if np.isfinite(r1) and np.isfinite(r2):
            mean_correlation = (r1 + r2) / 2
            correlations.append(mean_correlation)
    
    if not correlations:
        return np.nan
    
    mean_half_correlation = np.mean(correlations)
    return float(_apply_spearman_brown(mean_half_correlation))


###################################################################
# DataFrame-Based Reliability (Wide Format)
###################################################################


@dataclass
class ReliabilityResult:
    """Result of reliability computation for a single feature."""
    name: str
    reliability: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    method: str
    
    def is_acceptable(self, threshold: float = RELIABILITY_THRESHOLD_ACCEPTABLE) -> bool:
        return self.reliability >= threshold
    
    def is_good(self, threshold: float = RELIABILITY_THRESHOLD_GOOD) -> bool:
        return self.reliability >= threshold
    
    def is_excellent(self, threshold: float = RELIABILITY_THRESHOLD_EXCELLENT) -> bool:
        return self.reliability >= threshold






###################################################################
# Exports
###################################################################

__all__ = [
    # ICC and reliability
    "compute_icc",
    "compute_split_half_reliability",
    "compute_feature_reliability",
    "compute_correlation_split_half_reliability",
    "get_subject_seed",
    # DataFrame-based reliability (wide format)
    "ReliabilityResult",
    # Hierarchical FDR
    "hierarchical_fdr_dict",
    "compute_hierarchical_fdr_summary",
]
