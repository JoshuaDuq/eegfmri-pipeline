from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.behavioral.builders import generate_correlation_scatter
from eeg_pipeline.utils.analysis.stats import (
    compute_correlation_stats,
    joint_valid_mask,
)


@dataclass
class SubjectScatterData:
    temporal_df: pd.DataFrame
    features_df: pd.DataFrame
    outcome_series: pd.Series
    info: mne.Info
    temp_series: Optional[pd.Series]
    covariate_df_full: Optional[pd.DataFrame]
    covariate_df_temp: Optional[pd.DataFrame]
    roi_map: Dict[str, List[str]]
    stats_dir: Path
    plots_dir: Path
    conn_df: Optional[pd.DataFrame] = None




@dataclass
class ScatterPlotConfig:
    """Configuration for scatter plot generation."""

    method_code: str
    bootstrap_ci: int
    rng: np.random.Generator
    min_samples_for_plot: int
    significance_threshold: float
    target_rating: str


@dataclass
class ScatterPlotParams:
    """Parameters for generating a single scatter plot."""

    roi_vals: pd.Series
    target_vals: pd.Series
    roi: str
    band: str
    band_title: str
    band_color: str
    metric: Optional[str]
    target_type: str
    title: str
    x_label: str
    y_label: str
    output_path: Path
    roi_channels: List[str]
    feature_name: str
    stats_tag: Optional[str] = None
    is_partial_residuals: bool = False


def _get_scatter_plot_config_from_config(config: Any) -> ScatterPlotConfig:
    """Extract scatter plot configuration from main config object."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    validation_config = plot_cfg.validation

    return ScatterPlotConfig(
        method_code="",  # Set by caller
        bootstrap_ci=0,  # Set by caller
        rng=None,  # Set by caller
        min_samples_for_plot=validation_config.get("min_samples_for_plot", 5),
        significance_threshold=float(
            behavioral_config.get("significance_threshold", 0.05)
        ),
        target_rating=behavioral_config.get("target_rating", "outcome"),
    )


def _compute_correlation_statistics(
    roi_vals: pd.Series,
    target_vals: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    precomp_stats: Optional[Dict[str, Any]],
    *,
    is_partial_residuals: bool = False,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute correlation statistics from data or use precomputed values."""
    if precomp_stats:
        r_val = precomp_stats["r"]
        p_val = precomp_stats["p"]
        n_eff = precomp_stats["n"]
        ci_val = (precomp_stats.get("ci_low"), precomp_stats.get("ci_high"))
    else:
        # For Spearman partial residuals, the statistically correct correlation is
        # Pearson on rank-residuals (not re-ranking residuals and re-running Spearman).
        method_for_stats = method_code
        if is_partial_residuals and str(method_code).lower() == "spearman":
            method_for_stats = "pearson"
        r_val, p_val, n_eff, ci_val = compute_correlation_stats(
            roi_vals, target_vals, method_for_stats, bootstrap_ci, rng
        )
    return r_val, p_val, n_eff, ci_val


def _record_scatter_result(
    feature_name: str,
    roi: str,
    target_type: str,
    r_val: float,
    p_val: float,
    n_eff: int,
    output_path: Path,
    results: Dict[str, List],
    significance_threshold: float,
) -> None:
    """Record scatter plot result in results dictionary."""
    record = {
        "feature": feature_name,
        "roi": roi,
        "target": target_type,
        "r": r_val,
        "p": p_val,
        "n": n_eff,
        "path": str(output_path),
    }
    results["all"].append(record)
    if np.isfinite(p_val) and p_val < significance_threshold:
        results["significant"].append(record)


def _generate_single_scatter(
    *,
    params: ScatterPlotParams,
    plot_config: ScatterPlotConfig,
    precomp_stats: Optional[Dict[str, Any]],
    logger: logging.Logger,
    config: Any,
    results: Dict[str, List],
) -> None:
    """Generate a single scatter plot and record results."""
    if params.roi_vals.empty or params.target_vals.empty:
        logger.debug(
            f"Skipping scatter plot for {params.roi} {params.band}: empty data"
        )
        return

    valid_mask = joint_valid_mask(params.roi_vals, params.target_vals)
    n_valid = int(valid_mask.sum())

    r_val, p_val, n_eff, ci_val = _compute_correlation_statistics(
        params.roi_vals,
        params.target_vals,
        plot_config.method_code,
        plot_config.bootstrap_ci,
        plot_config.rng,
        precomp_stats,
        is_partial_residuals=params.is_partial_residuals,
    )

    if n_valid >= plot_config.min_samples_for_plot:
        generate_correlation_scatter(
            x_data=params.roi_vals,
            y_data=params.target_vals,
            x_label=params.x_label,
            y_label=params.y_label,
            title_prefix=params.title,
            band_color=params.band_color,
            output_path=params.output_path,
            method_code=plot_config.method_code,
            bootstrap_ci=0,
            rng=plot_config.rng,
            is_partial_residuals=params.is_partial_residuals,
            roi_channels=params.roi_channels,
            logger=logger,
            annotated_stats=(r_val, p_val, n_eff),
            annot_ci=ci_val,
            stats_tag=params.stats_tag,
            config=config,
        )

        _record_scatter_result(
            params.feature_name,
            params.roi,
            params.target_type,
            r_val,
            p_val,
            n_eff,
            params.output_path,
            results,
            plot_config.significance_threshold,
        )


__all__ = [
    "SubjectScatterData",
    "ScatterPlotConfig",
    "ScatterPlotParams",
]
