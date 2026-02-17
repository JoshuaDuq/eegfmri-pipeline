"""
Prediction Uncertainty Quantification
======================================

Conformal prediction for distribution-free prediction intervals with coverage guarantees.

Usage:
    from eeg_pipeline.analysis.machine_learning.uncertainty import (
        compute_prediction_intervals,
        PredictionIntervalResult,
    )
    
    result = compute_prediction_intervals(model, X_train, y_train, X_test, alpha=0.1)
    print(f"Coverage: {result.coverage:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone


###################################################################
# Conformal Prediction
###################################################################


@dataclass
class PredictionIntervalResult:
    """Container for prediction interval results."""
    
    y_pred: np.ndarray  # Point predictions
    lower: np.ndarray  # Lower bounds
    upper: np.ndarray  # Upper bounds
    alpha: float  # Significance level (1 - coverage)
    
    # Computed metrics
    coverage: float = np.nan
    mean_width: float = np.nan
    median_width: float = np.nan
    
    # Per-sample confidence
    widths: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Compute interval width-based summary metrics."""
        self.widths = self.upper - self.lower
        self.mean_width = float(np.mean(self.widths))
        self.median_width = float(np.median(self.widths))
    
    def compute_coverage(self, y_true: np.ndarray) -> float:
        """Compute empirical coverage given true values."""
        y_true = np.asarray(y_true)
        in_interval = (y_true >= self.lower) & (y_true <= self.upper)
        self.coverage = float(np.mean(in_interval))
        return self.coverage
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "y_pred": self.y_pred,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.widths,
        })
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Prediction Intervals (α={self.alpha:.2f}):\n"
            f"  Target coverage: {1 - self.alpha:.1%}\n"
            f"  Empirical coverage: {self.coverage:.1%}\n"
            f"  Mean width: {self.mean_width:.3f}\n"
            f"  Median width: {self.median_width:.3f}"
        )


def compute_prediction_intervals(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    method: str = "cv_plus",
    cv_splits: int = 5,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> PredictionIntervalResult:
    """
    Compute conformal prediction intervals.
    
    Provides distribution-free prediction intervals with guaranteed
    coverage probability of at least (1 - alpha) for exchangeable data.
    
    Parameters
    ----------
    model : Any
        Sklearn-compatible regressor
    X_train : np.ndarray
        Training features (for calibration)
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    alpha : float
        Significance level. Default 0.1 gives 90% prediction intervals.
    method : str
        Conformal method:
        - "split": Simple split conformal (fast, less efficient)
        - "cv_plus": CV+ conformal (more efficient, uses all data)
        - "cqr": Conformalized Quantile Regression (for heteroscedastic data)
    cv_splits : int
        Number of CV splits for cv_plus method
    seed : int
        Random seed
    
    Returns
    -------
    PredictionIntervalResult
        Prediction intervals with coverage metrics
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    rng = np.random.default_rng(seed)
    
    if method == "split":
        return _conformal_split(model, X_train, y_train, X_test, alpha, rng)
    elif method == "cv_plus":
        return _conformal_cv_plus(model, X_train, y_train, X_test, alpha, cv_splits, seed, groups)
    elif method == "cqr":
        return _conformalized_quantile_regression(model, X_train, y_train, X_test, alpha, cv_splits, seed, groups)
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Compute conformal quantile from residuals."""
    n_cal = len(residuals)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    return float(np.quantile(residuals, q_level))


def _order_stat_quantile(values: np.ndarray, q: float, *, tail: str) -> float:
    """Conservative finite-sample order-statistic quantile."""
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan

    q_clamped = float(np.clip(q, 0.0, 1.0))
    if tail == "upper":
        # Conservative upper-tail quantile (ceiling rank).
        rank = int(np.ceil(q_clamped * arr.size)) - 1
    else:
        # Conservative lower-tail quantile (floor rank).
        rank = int(np.floor(q_clamped * arr.size))
    rank = int(np.clip(rank, 0, arr.size - 1))
    return float(arr[rank])


def _get_cv_splitter(
    cv_splits: int,
    seed: int,
    groups: Optional[np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """Get cross-validation splitter based on groups."""
    from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
    
    if groups is not None:
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        if n_groups >= cv_splits:
            cv = GroupKFold(n_splits=cv_splits)
            return cv.split(X_train, y_train, groups)
        else:
            cv = LeaveOneGroupOut()
            return cv.split(X_train, y_train, groups)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        return cv.split(X_train)


def _conformal_split(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> PredictionIntervalResult:
    """
    Split conformal prediction.
    
    Split training data into proper training and calibration sets.
    """
    n = len(X_train)
    if n < 5:
        raise ValueError("Split conformal requires at least 5 training samples.")
    # Use ~20% for calibration (minimum 2), while preserving at least 2 proper-train samples.
    n_cal = min(max(int(0.2 * n), 2), n - 2)
    
    indices = rng.permutation(n)
    cal_idx = indices[:n_cal]
    train_idx = indices[n_cal:]
    
    X_proper = X_train[train_idx]
    y_proper = y_train[train_idx]
    X_cal = X_train[cal_idx]
    y_cal = y_train[cal_idx]
    
    # Fit on proper training set
    model_fit = clone(model)
    model_fit.fit(X_proper, y_proper)
    
    y_cal_pred = model_fit.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred)
    q_hat = _compute_conformal_quantile(residuals, alpha)
    
    # Predict on test set
    y_test_pred = model_fit.predict(X_test)
    
    return PredictionIntervalResult(
        y_pred=y_test_pred,
        lower=y_test_pred - q_hat,
        upper=y_test_pred + q_hat,
        alpha=alpha,
    )


def _conformal_cv_plus(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    cv_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> PredictionIntervalResult:
    """
    CV+ conformal prediction (Jackknife+).
    
    Uses cross-validation residuals for more efficient calibration.
    Supports group-aware CV via GroupKFold or LeaveOneGroupOut.
    """
    split_iter = list(_get_cv_splitter(cv_splits, seed, groups, X_train, y_train))
    n_test = int(len(X_test))
    lower_chunks: List[np.ndarray] = []
    upper_chunks: List[np.ndarray] = []

    for train_idx, val_idx in split_iter:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model_fold = clone(model)
        model_fold.fit(X_train[train_idx], y_train[train_idx])

        y_val_pred = np.asarray(model_fold.predict(X_train[val_idx]), dtype=float)
        residuals = np.abs(np.asarray(y_train[val_idx], dtype=float) - y_val_pred)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            continue

        y_test_pred_fold = np.asarray(model_fold.predict(X_test), dtype=float)
        if y_test_pred_fold.size != n_test:
            continue

        lower_chunks.append(y_test_pred_fold[:, None] - residuals[None, :])
        upper_chunks.append(y_test_pred_fold[:, None] + residuals[None, :])

    if not lower_chunks or not upper_chunks:
        # Safety fallback when fold-wise calibration is impossible.
        rng = np.random.default_rng(seed)
        return _conformal_split(model, X_train, y_train, X_test, alpha, rng)

    lower_candidates = np.concatenate(lower_chunks, axis=1)
    upper_candidates = np.concatenate(upper_chunks, axis=1)

    lower = np.full(n_test, np.nan, dtype=float)
    upper = np.full(n_test, np.nan, dtype=float)
    for i in range(n_test):
        lower[i] = _order_stat_quantile(lower_candidates[i, :], alpha, tail="lower")
        upper[i] = _order_stat_quantile(upper_candidates[i, :], 1.0 - alpha, tail="upper")

    model_full = clone(model)
    model_full.fit(X_train, y_train)
    y_test_pred = np.asarray(model_full.predict(X_test), dtype=float)

    return PredictionIntervalResult(
        y_pred=y_test_pred,
        lower=lower,
        upper=upper,
        alpha=alpha,
    )


def _conformalized_quantile_regression(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    cv_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> PredictionIntervalResult:
    """
    Conformalized Quantile Regression (CQR).
    
    Better for heteroscedastic data where prediction uncertainty
    varies across the feature space.
    Supports group-aware CV via GroupKFold or LeaveOneGroupOut.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return _conformal_cv_plus(model, X_train, y_train, X_test, alpha, cv_splits, seed, groups)
    
    n = len(X_train)
    alpha_lo = alpha / 2
    alpha_hi = 1 - alpha / 2
    split_iter = _get_cv_splitter(cv_splits, seed, groups, X_train, y_train)
    
    loo_lower = np.zeros(n)
    loo_upper = np.zeros(n)
    
    for train_idx, val_idx in split_iter:
        qr_low = GradientBoostingRegressor(
            loss="quantile", alpha=alpha_lo, random_state=seed
        )
        qr_low.fit(X_train[train_idx], y_train[train_idx])
        loo_lower[val_idx] = qr_low.predict(X_train[val_idx])
        
        qr_high = GradientBoostingRegressor(
            loss="quantile", alpha=alpha_hi, random_state=seed
        )
        qr_high.fit(X_train[train_idx], y_train[train_idx])
        loo_upper[val_idx] = qr_high.predict(X_train[val_idx])
    
    E_lo = loo_lower - y_train
    E_hi = y_train - loo_upper
    scores = np.maximum(E_lo, E_hi)
    Q_hat = _compute_conformal_quantile(scores, alpha)
    
    qr_low_full = GradientBoostingRegressor(
        loss="quantile", alpha=alpha_lo, random_state=seed
    )
    qr_low_full.fit(X_train, y_train)
    
    qr_high_full = GradientBoostingRegressor(
        loss="quantile", alpha=alpha_hi, random_state=seed
    )
    qr_high_full.fit(X_train, y_train)
    
    lower_pred = qr_low_full.predict(X_test) - Q_hat
    upper_pred = qr_high_full.predict(X_test) + Q_hat
    
    model_full = clone(model)
    model_full.fit(X_train, y_train)
    y_test_pred = model_full.predict(X_test)
    
    return PredictionIntervalResult(
        y_pred=y_test_pred,
        lower=lower_pred,
        upper=upper_pred,
        alpha=alpha,
    )
