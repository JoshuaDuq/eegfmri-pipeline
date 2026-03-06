"""
Reliability statistics used by the active behavior pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_subject_seed


DEFAULT_ALPHA = 0.05
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 100
MIN_SAMPLES_FOR_SPLIT_HALF = 4
MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF = 20
RELIABILITY_THRESHOLD_ACCEPTABLE = 0.7
RELIABILITY_THRESHOLD_GOOD = 0.8
RELIABILITY_THRESHOLD_EXCELLENT = 0.9


def _compute_icc_one_way_single(ms_rows: float, ms_within: float, k: int) -> float:
    return (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)


def _compute_icc_two_way_random_single(
    ms_rows: float,
    ms_error: float,
    ms_cols: float,
    n: int,
    k: int,
) -> float:
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return (ms_rows - ms_error) / denominator


def _compute_icc_two_way_mixed_single(ms_rows: float, ms_error: float, k: int) -> float:
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)


def _compute_icc_one_way_average(ms_rows: float, ms_within: float) -> float:
    return (ms_rows - ms_within) / ms_rows


def _compute_icc_two_way_random_average(
    ms_rows: float,
    ms_error: float,
    ms_cols: float,
    n: int,
) -> float:
    return (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)


def _compute_icc_two_way_mixed_average(ms_rows: float, ms_error: float) -> float:
    return (ms_rows - ms_error) / ms_rows


def _compute_icc_confidence_intervals(
    ms_rows: float,
    ms_error: float,
    n: int,
    k: int,
) -> Tuple[float, float]:
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


def compute_icc(data: np.ndarray, icc_type: str = "ICC(2,1)") -> Tuple[float, float, float]:
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
    if icc_type_upper in {"ICC(1,1)", "ICC1"}:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_single(ms_rows, ms_within, n_raters)
    elif icc_type_upper in {"ICC(2,1)", "ICC2"}:
        icc = _compute_icc_two_way_random_single(ms_rows, ms_error, ms_cols, n_subjects, n_raters)
    elif icc_type_upper in {"ICC(3,1)", "ICC3"}:
        icc = _compute_icc_two_way_mixed_single(ms_rows, ms_error, n_raters)
    elif icc_type_upper in {"ICC(1,K)", "ICC1K"}:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_average(ms_rows, ms_within)
    elif icc_type_upper in {"ICC(2,K)", "ICC2K"}:
        icc = _compute_icc_two_way_random_average(ms_rows, ms_error, ms_cols, n_subjects)
    elif icc_type_upper in {"ICC(3,K)", "ICC3K"}:
        icc = _compute_icc_two_way_mixed_average(ms_rows, ms_error)
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")

    ci_low, ci_high = _compute_icc_confidence_intervals(ms_rows, ms_error, n_subjects, n_raters)
    return float(np.clip(icc, -1, 1)), ci_low, ci_high


def _apply_spearman_brown(r_value: float) -> float:
    if r_value <= -1 or not np.isfinite(r_value):
        return np.nan
    return (2 * r_value) / (1 + r_value)


def compute_split_half_reliability(
    data: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    from .correlation import compute_correlation

    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_trials = data.shape[0]
    if n_trials < MIN_SAMPLES_FOR_SPLIT_HALF:
        return np.nan, np.nan, np.nan

    half_size = n_trials // 2
    correlations: List[float] = []

    for _ in range(n_splits):
        indices = rng.permutation(n_trials)
        half1_indices = indices[:half_size]
        half2_indices = indices[half_size : 2 * half_size]

        half1_means = data[half1_indices].mean(axis=0)
        half2_means = data[half2_indices].mean(axis=0)

        if len(half1_means) == 1:
            r_value, _ = compute_correlation(data[half1_indices, 0], data[half2_indices, 0], method)
        else:
            r_value, _ = compute_correlation(half1_means, half2_means, method)

        if np.isfinite(r_value):
            correlations.append(float(r_value))

    if not correlations:
        return np.nan, np.nan, np.nan

    reliability = _apply_spearman_brown(float(np.mean(correlations)))
    boot = [_apply_spearman_brown(r_value) for r_value in correlations]
    boot = [value for value in boot if np.isfinite(value)]
    if len(boot) > 10:
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))
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
    results: List[Dict[str, float | str | int]] = []

    for feature in df[feature_col].unique():
        feat_df = df[df[feature_col] == feature]
        if len(feat_df) < min_observations:
            continue

        values = feat_df[value_col].values
        sh_rel, sh_low, sh_high = compute_split_half_reliability(values)
        result: Dict[str, float | str | int] = {
            "feature": feature,
            "n": len(values),
            "split_half_reliability": sh_rel,
            "sh_ci_low": sh_low,
            "sh_ci_high": sh_high,
        }

        if session_col and session_col in feat_df.columns:
            pivot = feat_df.pivot_table(
                index=groupby_col,
                columns=session_col,
                values=value_col,
                aggfunc="mean",
            ).dropna()
            if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
                icc, icc_low, icc_high = compute_icc(pivot.values)
                result["icc"] = icc
                result["icc_ci_low"] = icc_low
                result["icc_ci_high"] = icc_high

        results.append(result)

    return pd.DataFrame(results)


def compute_correlation_split_half_reliability(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_splits: int = DEFAULT_N_SPLITS,
    rng: Optional[np.random.Generator] = None,
) -> float:
    from .correlation import compute_correlation

    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid_mask.sum())
    if n_valid < MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF:
        return np.nan

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    indices = np.arange(n_valid)
    half_size = n_valid // 2
    correlations: List[float] = []

    for _ in range(n_splits):
        rng.shuffle(indices)
        idx1 = indices[:half_size]
        idx2 = indices[half_size : 2 * half_size]
        r1, _ = compute_correlation(x_valid[idx1], y_valid[idx1], method)
        r2, _ = compute_correlation(x_valid[idx2], y_valid[idx2], method)
        if np.isfinite(r1) and np.isfinite(r2):
            correlations.append(float((r1 + r2) / 2))

    if not correlations:
        return np.nan

    return float(_apply_spearman_brown(float(np.mean(correlations))))


@dataclass
class ReliabilityResult:
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


__all__ = [
    "compute_correlation_split_half_reliability",
    "compute_feature_reliability",
    "compute_icc",
    "compute_split_half_reliability",
    "DEFAULT_ALPHA",
    "ReliabilityResult",
    "get_subject_seed",
]
