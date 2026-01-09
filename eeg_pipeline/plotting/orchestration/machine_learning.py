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

from eeg_pipeline.plotting.machine_learning.comparisons import (
    plot_incremental_validity,
    plot_model_comparison,
    plot_riemann_band_comparison,
    plot_riemann_sliding_window,
)
from eeg_pipeline.plotting.machine_learning.helpers import plot_residual_diagnostics
from eeg_pipeline.plotting.machine_learning.performance import (
    plot_bootstrap_distributions,
    plot_calibration_curve,
    plot_ml_null_hist,
    plot_per_subject_performance,
    plot_permutation_null,
    plot_prediction_scatter,
)
from eeg_pipeline.plotting.machine_learning.time_generalization import (
    plot_time_generalization_matrix,
    plot_time_generalization_with_null,
)


###################################################################
# Helper Functions
###################################################################


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Get logger instance, creating one if needed."""
    return logger if logger is not None else get_module_logger()


def _ensure_plots_directory(plots_dir: Path, logger: logging.Logger) -> None:
    """Ensure plots directory exists."""
    ensure_dir(plots_dir)


def _load_predictions_file(results_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Find and return path to predictions file, or None if not found."""
    pred_path = results_dir / "loso_predictions.tsv"
    if pred_path.exists():
        return pred_path
    
    alt_path = results_dir / "cv_predictions.tsv"
    if alt_path.exists():
        return alt_path
    
    logger.warning(f"Predictions file not found in {results_dir}")
    return None


def _load_metrics_file(results_dir: Path) -> Dict[str, float]:
    """Load pooled metrics from JSON file if it exists."""
    metrics_path = results_dir / "pooled_metrics.json"
    if not metrics_path.exists():
        return {}
    
    with open(metrics_path) as f:
        return json.load(f)


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
    
    if not per_subject_records:
        return None
    
    return pd.DataFrame(per_subject_records)


def _extract_model_prefix(pred_path: Path) -> str:
    """Extract model prefix from predictions file path."""
    if pred_path.name == "loso_predictions.tsv":
        return "loso"
    return pred_path.stem.replace("_predictions", "") or "cv"


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
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Pain", "Pain"])
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {model_name}")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    except (ValueError, KeyError) as exc:
        logger.warning(f"Confusion matrix plot failed: {exc}")


###################################################################
# Regression ML Visualization
###################################################################


def visualize_regression_results(
    pred_df: pd.DataFrame,
    per_subj_df: pd.DataFrame,
    pooled_metrics: Dict[str, float],
    model_name: str,
    plots_dir: Path,
    config: Optional[Any] = None,
    null_r: Optional[np.ndarray] = None,
    empirical_r: Optional[float] = None,
    bootstrap_results: Optional[Dict[str, Any]] = None,
    cal_metrics: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate regression machine learning visualizations from in-memory results."""
    logger = _get_logger(logger)
    _ensure_plots_directory(plots_dir, logger)

    logger.info(f"Creating regression ML visualizations for {model_name}...")

    plot_prediction_scatter(
        pred_df=pred_df,
        model_name=model_name,
        pooled_metrics=pooled_metrics,
        save_path=plots_dir / f"{model_name}_prediction_scatter",
        config=config,
    )

    plot_per_subject_performance(
        per_subj_df=per_subj_df,
        model_name=model_name,
        save_path=plots_dir / f"{model_name}_per_subject_performance",
        config=config,
    )

    plot_residual_diagnostics(
        pred_df=pred_df,
        model_name=model_name,
        save_path=plots_dir / f"{model_name}_residual_diagnostics",
        config=config,
    )

    if null_r is not None and empirical_r is not None:
        plot_ml_null_hist(
            null_r=null_r,
            empirical_r=empirical_r,
            save_path=plots_dir / f"{model_name}_null_histogram",
            config=config,
        )

    if cal_metrics is not None:
        plot_calibration_curve(
            pred_df=pred_df,
            model_name=model_name,
            cal_metrics=cal_metrics,
            save_path=plots_dir / f"{model_name}_calibration_curve",
            config=config,
        )

    if bootstrap_results is not None:
        plot_bootstrap_distributions(
            bootstrap_results=bootstrap_results,
            save_path=plots_dir / f"{model_name}_bootstrap_distributions",
            config=config,
        )

    logger.info(f"Regression machine learning visualizations saved to {plots_dir}")


###################################################################
# Time Generalization Visualization
###################################################################


def visualize_time_generalization(
    tg_matrix: np.ndarray,
    window_centers: np.ndarray,
    plots_dir: Path,
    metric: str = "r",
    null_matrix: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate time-generalization visualizations from in-memory results."""
    logger = _get_logger(logger)
    _ensure_plots_directory(plots_dir, logger)

    logger.info("Creating time-generalization visualizations...")

    if null_matrix is not None:
        plot_time_generalization_with_null(
            tg_matrix=tg_matrix,
            null_matrix=null_matrix,
            window_centers=window_centers,
            save_path=plots_dir / "time_generalization_with_null",
            metric=metric,
            config=config,
        )
    else:
        plot_time_generalization_matrix(
            tg_matrix=tg_matrix,
            window_centers=window_centers,
            save_path=plots_dir / "time_generalization_matrix",
            metric=metric,
            config=config,
        )

    logger.info(f"Time-generalization visualizations saved to {plots_dir}")


###################################################################
# Model Comparison Visualization
###################################################################


def visualize_model_comparisons(
    models_dict: Dict[str, Dict[str, float]],
    plots_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate model comparison visualizations."""
    logger = _get_logger(logger)
    _ensure_plots_directory(plots_dir, logger)

    logger.info("Creating model comparison visualizations...")

    plot_model_comparison(
        models_dict=models_dict,
        save_path=plots_dir / "model_comparison",
        config=config,
    )

    logger.info(f"Model comparison visualizations saved to {plots_dir}")


###################################################################
# Riemann Analysis Visualization
###################################################################


def visualize_riemann_analysis(
    band_results: Optional[Dict[str, Dict[str, float]]] = None,
    sliding_df: Optional[pd.DataFrame] = None,
    plots_dir: Optional[Path] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate Riemann analysis visualizations."""
    logger = _get_logger(logger)

    if plots_dir is None:
        logger.warning("No plots directory provided for Riemann visualizations")
        return

    _ensure_plots_directory(plots_dir, logger)
    logger.info("Creating Riemann analysis visualizations...")

    if band_results is not None:
        plot_riemann_band_comparison(
            band_results=band_results,
            save_path=plots_dir / "riemann_band_comparison",
            config=config,
        )

    if sliding_df is not None:
        plot_riemann_sliding_window(
            sliding_df=sliding_df,
            save_path=plots_dir / "riemann_sliding_window",
            config=config,
        )

    logger.info(f"Riemann analysis visualizations saved to {plots_dir}")


###################################################################
# Incremental Validity Visualization
###################################################################


def visualize_incremental_validity(
    inc_summary: Dict[str, Any],
    plots_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate incremental validity visualizations."""
    logger = _get_logger(logger)
    _ensure_plots_directory(plots_dir, logger)

    logger.info("Creating incremental validity visualizations...")

    plot_incremental_validity(
        inc_summary=inc_summary,
        save_path=plots_dir / "incremental_validity",
        config=config,
    )

    logger.info(f"Incremental validity visualizations saved to {plots_dir}")


###################################################################
# Post-Analysis Visualization (reads saved results)
###################################################################


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
    _ensure_plots_directory(plots_dir, logger)

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


def _find_group_column(pred_df: pd.DataFrame) -> Optional[str]:
    """Find the appropriate group column in predictions dataframe."""
    if "subject_id" in pred_df.columns:
        return "subject_id"
    if "group" in pred_df.columns:
        return "group"
    return None


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

    if null_rs is None or null_rs.size == 0:
        return

    if not np.isfinite(empirical_r):
        return

    plot_ml_null_hist(
        null_r=null_rs,
        empirical_r=empirical_r,
        save_path=results_dir / "plots" / f"{prefix}_null_distribution",
        config=config,
    )


def visualize_time_generalization_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate time-generalization plots from saved results on disk."""
    logger = _get_logger(logger)

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
        save_path=results_dir / "time_generalization_r",
        metric="r",
        config=config,
    )

    if tg_r2 is not None:
        plot_time_generalization_with_null(
            tg_matrix=tg_r2,
            null_matrix=null_r2,
            window_centers=window_centers,
            save_path=results_dir / "time_generalization_r2",
            metric="r2",
            config=config,
        )

    logger.info(f"Time-generalization visualizations saved to {results_dir}")


###################################################################
# Classification ML Visualization
###################################################################


def visualize_classification_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize classification ML results from disk.
    
    Single responsibility: Read classification results contract and create plots.
    """
    logger = _get_logger(logger)
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    _ensure_plots_directory(plots_dir, logger)

    pred_path = results_dir / "loso_predictions.tsv"
    if not pred_path.exists():
        logger.warning(f"Classification predictions not found: {pred_path}")
        return

    pred_df = pd.read_csv(pred_path, sep="\t")
    pooled_metrics = _load_metrics_file(results_dir)
    model_name = pooled_metrics.get("model", "classification")

    _plot_classification_roc_curve(pred_df, model_name, plots_dir, logger)
    _plot_classification_confusion_matrix(pred_df, model_name, plots_dir, logger)
    _plot_classification_calibration(pred_df, model_name, plots_dir, config, logger)
    _plot_classification_null_distribution(
        results_dir, pooled_metrics, model_name, plots_dir, logger
    )

    logger.info(f"Classification visualizations saved to {plots_dir}")


def _plot_classification_roc_curve(
    pred_df: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plot ROC curve for classification results."""
    if "y_true" not in pred_df.columns or "y_prob" not in pred_df.columns:
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
    if "y_true" not in pred_df.columns or "y_pred" not in pred_df.columns:
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
    if "y_true" not in pred_df.columns or "y_prob" not in pred_df.columns:
        return

    try:
        cal_metrics = {"model": model_name}
        plot_calibration_curve(
            pred_df=pred_df,
            model_name=model_name,
            cal_metrics=cal_metrics,
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
            config=None,
        )
    except (ValueError, KeyError, OSError) as exc:
        logger.warning(f"Null distribution plot failed: {exc}")


def _compute_p_value(null_distribution: np.ndarray, observed_value: float) -> float:
    """Compute p-value from null distribution and observed value."""
    if null_distribution.size == 0:
        return 1.0
    return (null_distribution >= observed_value).mean()


__all__ = [
    "visualize_regression_results",
    "visualize_time_generalization",
    "visualize_model_comparisons",
    "visualize_riemann_analysis",
    "visualize_incremental_validity",
    "visualize_regression_from_disk",
    "visualize_time_generalization_from_disk",
    "visualize_classification_from_disk",
]
