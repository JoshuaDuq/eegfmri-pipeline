"""Machine learning visualization orchestration (pipeline-level).

Plot primitives live in `eeg_pipeline.plotting.machine_learning.*`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, r2_score, roc_curve

from eeg_pipeline.infra.logging import get_module_logger
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import read_tsv

from eeg_pipeline.plotting.machine_learning.helpers import plot_residual_diagnostics
from eeg_pipeline.plotting.machine_learning.performance import (
    plot_calibration_curve,
    plot_ml_null_hist,
    plot_per_subject_performance,
    plot_permutation_null,
    plot_prediction_scatter,
)
from eeg_pipeline.plotting.machine_learning.time_generalization import (
    plot_time_generalization_with_null,
)


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Get logger instance, creating one if needed."""
    return logger if logger is not None else get_module_logger()


def _load_predictions_file(results_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Find and return path to predictions file, or None if not found."""
    for filename in ["loso_predictions.tsv", "cv_predictions.tsv"]:
        pred_path = results_dir / filename
        if pred_path.exists():
            return pred_path
    
    logger.warning(f"Predictions file not found in {results_dir}")
    return None


def _load_metrics_file(results_dir: Path) -> Dict[str, float]:
    """Load pooled metrics from JSON file if it exists."""
    metrics_path = results_dir / "pooled_metrics.json"
    if not metrics_path.exists():
        return {}
    
    with open(metrics_path) as f:
        return json.load(f)


def _find_group_column(pred_df: pd.DataFrame) -> Optional[str]:
    """Find the appropriate group column in predictions dataframe."""
    for col in ["subject_id", "group"]:
        if col in pred_df.columns:
            return col
    return None


def _compute_per_subject_metrics(
    pred_df: pd.DataFrame, group_column: str
) -> Optional[pd.DataFrame]:
    """Compute per-subject performance metrics from predictions."""
    per_subject_records = []
    
    for subject_id, subject_df in pred_df.groupby(group_column):
        y_true = pd.to_numeric(subject_df["y_true"], errors="coerce").values
        y_pred = pd.to_numeric(subject_df["y_pred"], errors="coerce").values
        
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite_mask.sum() < 2:
            continue
        
        y_true_finite = y_true[finite_mask]
        y_pred_finite = y_pred[finite_mask]
        
        r_value, _ = pearsonr(y_true_finite, y_pred_finite)
        r2_value = r2_score(y_true_finite, y_pred_finite)
        
        per_subject_records.append({
            "group": str(subject_id),
            "pearson_r": r_value,
            "r2": r2_value,
        })
    
    return pd.DataFrame(per_subject_records) if per_subject_records else None


def _extract_model_prefix(pred_path: Path) -> str:
    """Extract model prefix from predictions file path."""
    if pred_path.name == "loso_predictions.tsv":
        return "loso"
    return pred_path.stem.replace("_predictions", "") or "cv"


def _compute_p_value(null_distribution: np.ndarray, observed_value: float) -> float:
    """Compute p-value from null distribution and observed value."""
    if null_distribution.size == 0:
        return 1.0
    return (null_distribution >= observed_value).mean()


def _plot_roc_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    model_name: str,
    save_path: Path,
    logger: logging.Logger,
) -> None:
    """Plot ROC curve for classification results."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved ROC curve to {save_path}")
    except (ValueError, KeyError) as exc:
        logger.warning(f"ROC curve plot failed: {exc}")


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    save_path: Path,
    logger: logging.Logger,
) -> None:
    """Plot confusion matrix for classification results."""
    try:
        combined = pd.concat([y_true, y_pred], ignore_index=True)
        labels = pd.Index(combined.dropna()).unique().tolist()
        display_labels = [str(v) for v in labels]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {model_name}")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion matrix to {save_path}")
    except (ValueError, KeyError) as exc:
        logger.warning(f"Confusion matrix plot failed: {exc}")


def _plot_null_distribution_if_available(
    results_dir: Path,
    pooled_metrics: Dict[str, float],
    prefix: str,
    config: Optional[Any],
    logger: logging.Logger,
) -> None:
    """Plot null distribution histogram if available."""
    null_path = results_dir / "loso_null_elasticnet.npz"
    if not null_path.exists():
        return

    data = np.load(null_path)
    null_rs = data.get("null_r")
    empirical_r = pooled_metrics.get("pearson_r", np.nan)

    if null_rs is None or null_rs.size == 0 or not np.isfinite(empirical_r):
        return

    plots_dir = results_dir / "plots"
    plot_ml_null_hist(
        null_r=null_rs,
        empirical_r=empirical_r,
        save_path=plots_dir / f"{prefix}_null_distribution",
        config=config,
    )


def _plot_classification_roc_curve(
    pred_df: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plot ROC curve for classification results."""
    required_cols = ["y_true", "y_prob"]
    if not all(col in pred_df.columns for col in required_cols):
        return

    save_path = plots_dir / f"roc_curve_{model_name}.png"
    _plot_roc_curve(pred_df["y_true"], pred_df["y_prob"], model_name, save_path, logger)


def _plot_classification_confusion_matrix(
    pred_df: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plot confusion matrix for classification results."""
    required_cols = ["y_true", "y_pred"]
    if not all(col in pred_df.columns for col in required_cols):
        return

    save_path = plots_dir / f"confusion_matrix_{model_name}.png"
    _plot_confusion_matrix(pred_df["y_true"], pred_df["y_pred"], model_name, save_path, logger)


def _plot_classification_calibration(
    pred_df: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    config: Optional[Any],
    logger: logging.Logger,
) -> None:
    """Plot calibration curve for classification results."""
    required_cols = ["y_true", "y_prob"]
    if not all(col in pred_df.columns for col in required_cols):
        return

    try:
        plot_calibration_curve(
            pred_df=pred_df,
            model_name=model_name,
            cal_metrics={"model": model_name},
            save_path=plots_dir / f"calibration_{model_name}",
            config=config,
        )
    except (ValueError, KeyError) as exc:
        logger.warning(f"Calibration curve failed: {exc}")


def _plot_classification_null_distribution(
    results_dir: Path,
    pooled_metrics: Dict[str, Any],
    model_name: str,
    plots_dir: Path,
    config: Optional[Any],
    logger: logging.Logger,
) -> None:
    """Plot null distribution for classification results if available."""
    null_paths = list(results_dir.glob("loso_null_*.npz"))
    if not null_paths:
        return

    try:
        data = np.load(null_paths[0])
        null_auc = data.get("null_auc")
        if null_auc is None or null_auc.size == 0:
            return

        observed_auc = pooled_metrics.get("auc", 0.0)
        p_value = _compute_p_value(null_auc, observed_auc)

        plot_permutation_null(
            null_rs=null_auc,
            observed_r=observed_auc,
            p_value=p_value,
            save_path=plots_dir / f"null_distribution_{model_name}.png",
            config=config,
        )
    except (ValueError, KeyError, OSError) as exc:
        logger.warning(f"Null distribution plot failed: {exc}")


def visualize_regression_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate regression machine learning plots from saved results on disk.
    
    This is the pipeline-layer entrypoint that reads predictions, metrics,
    and null distributions from disk and generates all standard plots.
    """
    logger = _get_logger(logger)
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)

    pred_path = _load_predictions_file(results_dir, logger)
    if pred_path is None:
        return

    pred_df = read_tsv(pred_path)
    pooled_metrics = _load_metrics_file(results_dir)
    model_name = "elasticnet"
    prefix = _extract_model_prefix(pred_path)

    plot_prediction_scatter(
        pred_df=pred_df,
        model_name=model_name,
        pooled_metrics=pooled_metrics,
        save_path=plots_dir / f"{prefix}_prediction_scatter",
        config=config,
    )

    plot_residual_diagnostics(
        pred_df=pred_df,
        model_name=model_name,
        save_path=plots_dir / f"{prefix}_residual_diagnostics",
        config=config,
    )

    group_column = _find_group_column(pred_df)
    if group_column is not None:
        per_subj_df = _compute_per_subject_metrics(pred_df, group_column)
        if per_subj_df is not None:
            plot_per_subject_performance(
                per_subj_df=per_subj_df,
                model_name=model_name,
                save_path=plots_dir / f"{prefix}_per_subject_performance",
                config=config,
            )

    _plot_null_distribution_if_available(
        results_dir, pooled_metrics, prefix, config, logger
    )

    logger.info(f"Regression machine learning visualizations saved to {plots_dir}")


def visualize_time_generalization_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate time-generalization plots from saved results on disk."""
    logger = _get_logger(logger)
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)

    tg_path = results_dir / "time_generalization_regression.npz"
    if not tg_path.exists():
        logger.warning(f"Time-generalization results not found: {tg_path}")
        return

    data = np.load(tg_path)
    tg_r = data.get("r_matrix")
    tg_r2 = data.get("r2_matrix")
    window_centers = data.get("window_centers")
    null_r = data.get("null_r")
    null_r2 = data.get("null_r2")

    if tg_r is None or window_centers is None:
        logger.warning("Missing required arrays in time-generalization results")
        return

    plot_time_generalization_with_null(
        tg_matrix=tg_r,
        null_matrix=null_r,
        window_centers=window_centers,
        save_path=plots_dir / "time_generalization_r",
        metric="r",
        config=config,
    )

    if tg_r2 is not None:
        plot_time_generalization_with_null(
            tg_matrix=tg_r2,
            null_matrix=null_r2,
            window_centers=window_centers,
            save_path=plots_dir / "time_generalization_r2",
            metric="r2",
            config=config,
        )

    logger.info(f"Time-generalization visualizations saved to {plots_dir}")


def visualize_classification_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize classification ML results from disk.
    
    Single responsibility: Read classification results contract and create plots.
    """
    logger = _get_logger(logger)
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)

    pred_path = results_dir / "loso_predictions.tsv"
    if not pred_path.exists():
        logger.warning(f"Classification predictions not found: {pred_path}")
        return

    pred_df = read_tsv(pred_path)
    pooled_metrics = _load_metrics_file(results_dir)
    model_name = pooled_metrics.get("model", "classification")

    _plot_classification_roc_curve(pred_df, model_name, plots_dir, logger)
    _plot_classification_confusion_matrix(pred_df, model_name, plots_dir, logger)
    _plot_classification_calibration(pred_df, model_name, plots_dir, config, logger)
    _plot_classification_null_distribution(
        results_dir, pooled_metrics, model_name, plots_dir, config, logger
    )

    logger.info(f"Classification visualizations saved to {plots_dir}")


__all__ = [
    "visualize_regression_from_disk",
    "visualize_time_generalization_from_disk",
    "visualize_classification_from_disk",
]
