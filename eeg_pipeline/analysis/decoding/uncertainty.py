"""
Prediction Uncertainty Quantification
======================================

Conformal prediction and calibrated uncertainty for regression and classification.

Key methods:
- Conformal Prediction: Distribution-free prediction intervals with coverage guarantees
- Calibration: Ensure predicted probabilities match true frequencies

Usage:
    from eeg_pipeline.analysis.decoding.uncertainty import (
        compute_prediction_intervals,
        calibrate_classifier,
        PredictionIntervalResult,
    )
    
    # Get prediction intervals with 90% coverage
    result = compute_prediction_intervals(model, X_cal, y_cal, X_test, alpha=0.1)
    print(f"Coverage: {result.coverage:.1%}")
    
    # Calibrate classifier
    calibrated_probs = calibrate_classifier(model, X_cal, y_cal, X_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import cross_val_predict


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
        """Compute metrics if y_true is available."""
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
        return _conformal_cv_plus(model, X_train, y_train, X_test, alpha, cv_splits, seed)
    elif method == "cqr":
        return _conformalized_quantile_regression(model, X_train, y_train, X_test, alpha, cv_splits, seed)
    else:
        raise ValueError(f"Unknown method: {method}")


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
    n_cal = max(int(0.2 * n), 50)  # Use 20% for calibration, min 50
    
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
    
    # Compute residuals on calibration set
    y_cal_pred = model_fit.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred)
    
    # Compute conformal quantile
    n_cal = len(residuals)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(residuals, q_level)
    
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
) -> PredictionIntervalResult:
    """
    CV+ conformal prediction (Jackknife+).
    
    Uses cross-validation residuals for more efficient calibration.
    """
    from sklearn.model_selection import KFold
    
    n = len(X_train)
    
    # Get out-of-fold predictions via cross-validation
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    
    # Store residuals and which model was used for each point
    loo_residuals = np.zeros(n)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        model_fold = clone(model)
        model_fold.fit(X_train[train_idx], y_train[train_idx])
        
        y_val_pred = model_fold.predict(X_train[val_idx])
        loo_residuals[val_idx] = np.abs(y_train[val_idx] - y_val_pred)
    
    # Fit final model on all training data
    model_full = clone(model)
    model_full.fit(X_train, y_train)
    y_test_pred = model_full.predict(X_test)
    
    # Compute conformal quantile
    n_cal = len(loo_residuals)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(loo_residuals, q_level)
    
    return PredictionIntervalResult(
        y_pred=y_test_pred,
        lower=y_test_pred - q_hat,
        upper=y_test_pred + q_hat,
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
) -> PredictionIntervalResult:
    """
    Conformalized Quantile Regression (CQR).
    
    Better for heteroscedastic data where prediction uncertainty
    varies across the feature space.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        # Fall back to CV+ if no quantile regressor available
        return _conformal_cv_plus(model, X_train, y_train, X_test, alpha, cv_splits, seed)
    
    from sklearn.model_selection import KFold
    
    n = len(X_train)
    alpha_lo = alpha / 2
    alpha_hi = 1 - alpha / 2
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    
    # Get out-of-fold quantile predictions
    loo_lower = np.zeros(n)
    loo_upper = np.zeros(n)
    
    for train_idx, val_idx in kf.split(X_train):
        # Fit lower quantile model
        qr_low = GradientBoostingRegressor(
            loss="quantile", alpha=alpha_lo, random_state=seed
        )
        qr_low.fit(X_train[train_idx], y_train[train_idx])
        loo_lower[val_idx] = qr_low.predict(X_train[val_idx])
        
        # Fit upper quantile model
        qr_high = GradientBoostingRegressor(
            loss="quantile", alpha=alpha_hi, random_state=seed
        )
        qr_high.fit(X_train[train_idx], y_train[train_idx])
        loo_upper[val_idx] = qr_high.predict(X_train[val_idx])
    
    # Compute conformity scores
    E_lo = loo_lower - y_train
    E_hi = y_train - loo_upper
    scores = np.maximum(E_lo, E_hi)
    
    # Compute conformal quantile
    n_cal = len(scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    Q_hat = np.quantile(scores, q_level)
    
    # Fit final quantile models on all data
    qr_low_full = GradientBoostingRegressor(
        loss="quantile", alpha=alpha_lo, random_state=seed
    )
    qr_low_full.fit(X_train, y_train)
    
    qr_high_full = GradientBoostingRegressor(
        loss="quantile", alpha=alpha_hi, random_state=seed
    )
    qr_high_full.fit(X_train, y_train)
    
    # Predict on test set
    lower_pred = qr_low_full.predict(X_test) - Q_hat
    upper_pred = qr_high_full.predict(X_test) + Q_hat
    
    # Point prediction from main model
    model_full = clone(model)
    model_full.fit(X_train, y_train)
    y_test_pred = model_full.predict(X_test)
    
    return PredictionIntervalResult(
        y_pred=y_test_pred,
        lower=lower_pred,
        upper=upper_pred,
        alpha=alpha,
    )


###################################################################
# Classifier Calibration
###################################################################


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    
    probabilities: np.ndarray  # Calibrated probabilities
    original_probs: np.ndarray  # Original uncalibrated probabilities
    
    # Calibration metrics
    ece: float = np.nan  # Expected Calibration Error
    mce: float = np.nan  # Maximum Calibration Error
    brier_score: float = np.nan
    
    # Reliability diagram data
    bin_centers: Optional[np.ndarray] = None
    bin_accuracies: Optional[np.ndarray] = None
    bin_confidences: Optional[np.ndarray] = None
    bin_counts: Optional[np.ndarray] = None
    
    def compute_metrics(self, y_true: np.ndarray, n_bins: int = 10):
        """Compute calibration metrics given true labels."""
        y_true = np.asarray(y_true)
        
        # Brier score
        self.brier_score = float(np.mean((self.probabilities - y_true) ** 2))
        
        # Binned calibration metrics
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(self.probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        self.bin_accuracies = np.zeros(n_bins)
        self.bin_confidences = np.zeros(n_bins)
        self.bin_counts = np.zeros(n_bins, dtype=int)
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                acc = np.mean(y_true[mask])
                conf = np.mean(self.probabilities[mask])
                count = int(mask.sum())
                
                self.bin_accuracies[i] = acc
                self.bin_confidences[i] = conf
                self.bin_counts[i] = count
                
                gap = abs(acc - conf)
                ece += count * gap
                mce = max(mce, gap)
        
        self.ece = ece / len(y_true)
        self.mce = mce
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Calibration Results:\n"
            f"  Expected Calibration Error (ECE): {self.ece:.4f}\n"
            f"  Maximum Calibration Error (MCE): {self.mce:.4f}\n"
            f"  Brier Score: {self.brier_score:.4f}"
        )


def calibrate_classifier(
    model: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    method: str = "isotonic",
) -> CalibrationResult:
    """
    Calibrate classifier probabilities.
    
    Calibration ensures that when the model says 70% probability,
    the outcome is true ~70% of the time.
    
    Parameters
    ----------
    model : Any
        Fitted classifier with predict_proba method
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration labels
    X_test : np.ndarray
        Test features
    method : str
        Calibration method:
        - "isotonic": Isotonic regression (flexible, no assumptions)
        - "sigmoid": Platt scaling (assumes sigmoid relationship)
        - "beta": Beta calibration (handles over/under-confidence)
    
    Returns
    -------
    CalibrationResult
        Calibrated probabilities with metrics
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    X_test = np.asarray(X_test)
    
    # Get original probabilities
    if hasattr(model, "predict_proba"):
        original_probs = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must have predict_proba method")
    
    # Calibrate
    if method == "beta":
        # Beta calibration (if available)
        try:
            from betacal import BetaCalibration
            cal = BetaCalibration()
            cal_probs_train = model.predict_proba(X_cal)[:, 1]
            cal.fit(cal_probs_train.reshape(-1, 1), y_cal)
            test_probs = model.predict_proba(X_test)[:, 1]
            calibrated_probs = cal.predict(test_probs.reshape(-1, 1)).ravel()
        except ImportError:
            # Fall back to isotonic
            method = "isotonic"
    
    if method in ("isotonic", "sigmoid"):
        calibrated_clf = CalibratedClassifierCV(
            model, method=method, cv="prefit"
        )
        calibrated_clf.fit(X_cal, y_cal)
        calibrated_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    
    return CalibrationResult(
        probabilities=calibrated_probs,
        original_probs=original_probs,
    )


###################################################################
# Visualization
###################################################################


def plot_prediction_intervals(
    result: PredictionIntervalResult,
    y_true: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    max_points: int = 100,
) -> Any:
    """
    Plot prediction intervals.
    
    Parameters
    ----------
    result : PredictionIntervalResult
        Prediction interval result
    y_true : np.ndarray, optional
        True values to overlay
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    max_points : int
        Maximum points to display
    
    Returns
    -------
    matplotlib.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    
    n = len(result.y_pred)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
    else:
        idx = np.arange(n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(idx))
    
    # Plot intervals as error bars
    ax.errorbar(
        x, result.y_pred[idx],
        yerr=[
            result.y_pred[idx] - result.lower[idx],
            result.upper[idx] - result.y_pred[idx],
        ],
        fmt="none",
        color="#1f77b4",
        alpha=0.3,
        capsize=2,
        label=f"Prediction Interval ({1 - result.alpha:.0%})",
    )
    
    # Plot predictions
    ax.scatter(x, result.y_pred[idx], color="#1f77b4", s=20, label="Prediction", zorder=3)
    
    # Plot true values if available
    if y_true is not None:
        y_true = np.asarray(y_true)
        in_interval = (y_true[idx] >= result.lower[idx]) & (y_true[idx] <= result.upper[idx])
        
        ax.scatter(
            x[in_interval], y_true[idx][in_interval],
            color="#2ca02c", s=20, marker="x", label="True (in interval)",
            zorder=4
        )
        ax.scatter(
            x[~in_interval], y_true[idx][~in_interval],
            color="#d62728", s=20, marker="x", label="True (outside)",
            zorder=4
        )
        
        coverage = np.mean(in_interval)
        ax.set_title(f"Prediction Intervals (Coverage: {coverage:.1%})")
    else:
        ax.set_title("Prediction Intervals")
    
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    return fig


def plot_reliability_diagram(
    result: CalibrationResult,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Any:
    """
    Plot reliability diagram (calibration curve).
    
    Parameters
    ----------
    result : CalibrationResult
        Calibration result with computed metrics
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    
    if result.bin_accuracies is None:
        raise ValueError("Call result.compute_metrics() first")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    
    mask = result.bin_counts > 0
    ax1.bar(
        result.bin_centers[mask],
        result.bin_accuracies[mask],
        width=0.1,
        alpha=0.5,
        edgecolor="black",
        label="Model",
    )
    
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Reliability Diagram (ECE={result.ece:.3f})")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predictions
    ax2.bar(
        result.bin_centers,
        result.bin_counts,
        width=0.1,
        alpha=0.5,
        edgecolor="black",
    )
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    return fig


###################################################################
# Save/Load
###################################################################


def save_prediction_intervals(
    result: PredictionIntervalResult,
    output_path: Path,
    prefix: str = "prediction_intervals",
) -> Dict[str, Path]:
    """Save prediction interval results."""
    from eeg_pipeline.utils.io.tsv import write_tsv
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = {}
    
    # Main results
    df = result.to_dataframe()
    path = output_path / f"{prefix}_results.tsv"
    write_tsv(df, path)
    saved["results"] = path
    
    # Summary metrics
    metrics = {
        "alpha": result.alpha,
        "target_coverage": 1 - result.alpha,
        "empirical_coverage": result.coverage,
        "mean_width": result.mean_width,
        "median_width": result.median_width,
        "n_samples": len(result.y_pred),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_path / f"{prefix}_metrics.tsv"
    write_tsv(metrics_df, metrics_path)
    saved["metrics"] = metrics_path
    
    return saved

















