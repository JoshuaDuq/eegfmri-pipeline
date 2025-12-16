"""Decoding visualization orchestration (pipeline-level).

Plot primitives live in `eeg_pipeline.plotting.decoding.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from eeg_pipeline.infra.paths import ensure_dir

from eeg_pipeline.plotting.decoding.time_generalization import (
    plot_time_generalization_matrix,
    plot_time_generalization_with_null,
)
from eeg_pipeline.plotting.decoding.performance import (
    plot_prediction_scatter,
    plot_per_subject_performance,
    plot_decoding_null_hist,
    plot_calibration_curve,
    plot_bootstrap_distributions,
    plot_permutation_null,
)
from eeg_pipeline.plotting.decoding.helpers import plot_residual_diagnostics
from eeg_pipeline.plotting.decoding.comparisons import (
    plot_model_comparison,
    plot_riemann_band_comparison,
    plot_riemann_sliding_window,
    plot_incremental_validity,
)


###################################################################
# Regression Decoding Visualization
###################################################################


def visualize_regression_results(
    pred_df: Any,
    per_subj_df: Any,
    pooled_metrics: Dict[str, float],
    model_name: str,
    plots_dir: Path,
    config: Optional[Any] = None,
    null_r: Optional[Any] = None,
    empirical_r: Optional[float] = None,
    bootstrap_results: Optional[Dict[str, Any]] = None,
    cal_metrics: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(plots_dir)

    logger.info(f"Creating regression decoding visualizations for {model_name}...")

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
        plot_decoding_null_hist(
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

    logger.info(f"Regression decoding visualizations saved to {plots_dir}")


###################################################################
# Time Generalization Visualization
###################################################################


def visualize_time_generalization(
    tg_matrix: Any,
    window_centers: Any,
    plots_dir: Path,
    metric: str = "r",
    null_matrix: Optional[Any] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(plots_dir)

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
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(plots_dir)

    logger.info("Creating model comparison visualizations...")

    plot_model_comparison(models_dict=models_dict, save_path=plots_dir / "model_comparison", config=config)

    logger.info(f"Model comparison visualizations saved to {plots_dir}")


###################################################################
# Riemann Analysis Visualization
###################################################################


def visualize_riemann_analysis(
    band_results: Optional[Dict[str, Dict[str, float]]] = None,
    sliding_df: Optional[Any] = None,
    plots_dir: Optional[Path] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    if plots_dir is None:
        logger.warning("No plots directory provided for Riemann visualizations")
        return

    ensure_dir(plots_dir)

    logger.info("Creating Riemann analysis visualizations...")

    if band_results is not None:
        plot_riemann_band_comparison(band_results=band_results, save_path=plots_dir / "riemann_band_comparison", config=config)

    if sliding_df is not None:
        plot_riemann_sliding_window(sliding_df=sliding_df, save_path=plots_dir / "riemann_sliding_window", config=config)

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
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(plots_dir)

    logger.info("Creating incremental validity visualizations...")

    plot_incremental_validity(inc_summary=inc_summary, save_path=plots_dir / "incremental_validity", config=config)

    logger.info(f"Incremental validity visualizations saved to {plots_dir}")


###################################################################
# Post-Analysis Visualization (reads saved results)
###################################################################


def visualize_regression_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate regression decoding plots from saved results on disk.
    
    This is the pipeline-layer entrypoint that reads predictions, metrics,
    and null distributions from disk and generates all standard plots.
    """
    import json
    import pandas as pd
    import numpy as np
    from eeg_pipeline.infra.tsv import read_tsv
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)
    
    pred_path = results_dir / "loso_predictions.tsv"
    metrics_path = results_dir / "pooled_metrics.json"
    null_path = results_dir / "loso_null_elasticnet.npz"
    
    if not pred_path.exists():
        logger.warning(f"Predictions file not found: {pred_path}")
        return
    
    pred_df = read_tsv(pred_path)
    
    pooled_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            pooled_metrics = json.load(f)
    
    model_name = "elasticnet"
    
    plot_prediction_scatter(
        pred_df=pred_df,
        model_name=model_name,
        pooled_metrics=pooled_metrics,
        save_path=plots_dir / "loso_prediction_scatter",
        config=config,
    )
    
    plot_residual_diagnostics(
        pred_df=pred_df,
        model_name=model_name,
        save_path=plots_dir / "loso_residual_diagnostics",
        config=config,
    )
    
    if "subject_id" in pred_df.columns or "group" in pred_df.columns:
        group_col = "subject_id" if "subject_id" in pred_df.columns else "group"
        per_subj_records = []
        for subj, df_sub in pred_df.groupby(group_col):
            yt = pd.to_numeric(df_sub["y_true"], errors="coerce").values
            yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").values
            finite = np.isfinite(yt) & np.isfinite(yp)
            if finite.sum() < 2:
                continue
            from scipy.stats import pearsonr
            r_val, _ = pearsonr(yt[finite], yp[finite])
            from sklearn.metrics import r2_score
            r2_val = r2_score(yt[finite], yp[finite])
            per_subj_records.append({"group": str(subj), "pearson_r": r_val, "r2": r2_val})
        
        if per_subj_records:
            per_subj_df = pd.DataFrame(per_subj_records)
            plot_per_subject_performance(
                per_subj_df=per_subj_df,
                model_name=model_name,
                save_path=plots_dir / "loso_per_subject_performance",
                config=config,
            )
    
    if null_path.exists():
        data = np.load(null_path)
        null_rs = data.get("null_r")
        empirical_r = pooled_metrics.get("pearson_r", np.nan)
        if null_rs is not None and null_rs.size > 0 and np.isfinite(empirical_r):
            plot_decoding_null_hist(
                null_r=null_rs,
                empirical_r=empirical_r,
                save_path=plots_dir / "loso_null_distribution",
                config=config,
            )
    
    logger.info(f"Regression decoding visualizations saved to {plots_dir}")


def visualize_time_generalization_from_disk(
    results_dir: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate time-generalization plots from saved results on disk."""
    import numpy as np
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
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


__all__ = [
    "visualize_regression_results",
    "visualize_time_generalization",
    "visualize_model_comparisons",
    "visualize_riemann_analysis",
    "visualize_incremental_validity",
    "visualize_regression_from_disk",
    "visualize_time_generalization_from_disk",
]

