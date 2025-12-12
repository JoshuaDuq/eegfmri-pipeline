from __future__ import annotations

import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.plotting.behavioral.builders import (
    generate_correlation_scatter,
    plot_residual_qc,
    plot_regression_residual_diagnostics,
)
from eeg_pipeline.utils.data import (
    _build_covariate_matrices,
    _load_features_and_targets,
    _pick_first_column,
    load_epochs_for_analysis,
    load_precomputed_correlations,
    get_precomputed_stats_for_roi_band,
    load_subject_scatter_data,
)
from eeg_pipeline.utils.io.paths import deriv_plots_path, deriv_stats_path, ensure_dir, _load_events_df
from eeg_pipeline.utils.io.plotting import (
    save_fig,
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
)
from eeg_pipeline.utils.io.formatting import (
    sanitize_label,
    get_residual_labels,
    get_target_labels,
    get_temporal_xlabel,
    format_time_suffix,
)
from eeg_pipeline.utils.io.logging import get_subject_logger, get_default_logger as _get_default_logger
from eeg_pipeline.utils.analysis.stats import (
    compute_partial_residuals as _compute_partial_residuals,
    bootstrap_corr_ci as _bootstrap_corr_ci,
    joint_valid_mask,
    compute_correlation_stats,
    compute_partial_residuals_stats,
    compute_kde_scale,
    extract_roi_statistics,
    extract_overall_statistics,
    update_stats_from_dataframe,
)
from eeg_pipeline.utils.analysis.tfr import build_rois_from_info as _build_rois
from eeg_pipeline.plotting.behavioral.registry import BehaviorPlotRegistry


###################################################################
# Data Containers
###################################################################


@dataclass
class SubjectScatterData:
    """Container for loaded subject data used in ROI scatter plotting."""
    temporal_df: pd.DataFrame
    features_df: pd.DataFrame
    y: pd.Series
    info: mne.Info
    temp_series: Optional[pd.Series]
    Z_df_full: Optional[pd.DataFrame]
    Z_df_temp: Optional[pd.DataFrame]
    roi_map: Dict[str, List[str]]
    stats_dir: Path
    plots_dir: Path
    conn_df: Optional[pd.DataFrame] = None


class FeatureColumnExtractor(Protocol):
    """Protocol for feature-specific column extraction."""
    
    def __call__(
        self, 
        features_df: pd.DataFrame, 
        band: str, 
        roi_channels: List[str],
        metric: Optional[str] = None,
    ) -> Tuple[pd.Series, bool]:
        ...


###################################################################
# Generic ROI Scatter Infrastructure
###################################################################


def _setup_scatter_context(
    subject: str,
    deriv_root: Path,
    task: Optional[str],
    plots_dir: Optional[Path],
    feature_subdir: str,
    config,
    logger: logging.Logger,
) -> Optional[SubjectScatterData]:
    """Set up scatter plotting context and load subject data."""
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    
    if plots_dir is None:
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    
    feature_dir = plots_dir / feature_subdir
    ensure_dir(feature_dir)
    
    if task is None:
        task = config.task
    
    # Load full data including connectivity
    result = load_subject_scatter_data(subject, task, deriv_root, config, logger, None)
    temporal_df, features_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map, conn_df = result
    
    if temporal_df is None:
        return None
    
    stats_dir = deriv_stats_path(deriv_root, subject)
    
    return SubjectScatterData(
        temporal_df=temporal_df,
        features_df=features_df,
        y=y,
        info=info,
        temp_series=temp_series,
        Z_df_full=Z_df_full,
        Z_df_temp=Z_df_temp,
        roi_map=roi_map,
        stats_dir=stats_dir,
        plots_dir=feature_dir,
        conn_df=conn_df,
    )


def _generate_single_scatter(
    roi_vals: pd.Series,
    target_vals: pd.Series,
    roi: str,
    band: str,
    band_title: str,
    band_color: str,
    metric: Optional[str],
    target_type: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    chs: List[str],
    precomp_stats: Optional[Dict[str, Any]],
    logger: logging.Logger,
    config: Any,
    results: Dict[str, List],
    feature_name: str,
) -> None:
    """Generate a single scatter plot and record results."""
    n_valid = len(joint_valid_mask(roi_vals, target_vals))
    
    if precomp_stats:
        r_val = precomp_stats["r"]
        p_val = precomp_stats["p"]
        n_eff = precomp_stats["n"]
        ci_val = (precomp_stats.get("ci_low"), precomp_stats.get("ci_high"))
    else:
        r_val, p_val, n_eff, ci_val = compute_correlation_stats(
            roi_vals, target_vals, method_code, bootstrap_ci, rng
        )
    
    if n_valid > 5:
        generate_correlation_scatter(
            x_data=roi_vals,
            y_data=target_vals,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title,
            band_color=band_color,
            output_path=output_path,
            method_code=method_code,
            bootstrap_ci=0,
            rng=rng,
            roi_channels=chs,
            logger=logger,
            annotated_stats=(r_val, p_val, n_eff),
            annot_ci=ci_val,
            config=config,
        )
        
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
        if p_val < 0.05:
            results["significant"].append(record)


def _create_roi_scatter_plots(
    data: SubjectScatterData,
    feature_type: str,
    column_extractor: FeatureColumnExtractor,
    title_formatter: Callable[[str, str, str, Optional[str]], str],
    x_label_formatter: Callable[[str, Optional[str]], str],
    filename_formatter: Callable[[str, str, Optional[str]], str],
    feature_name_formatter: Callable[[str, Optional[str]], str],
    bands: List[str],
    metrics: Optional[List[str]],
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    do_temp: bool,
    rating_stats: Optional[pd.DataFrame],
    temp_stats: Optional[pd.DataFrame],
    logger: logging.Logger,
    config: Any,
) -> Dict[str, Any]:
    """Generic ROI scatter plotting with strategy pattern."""
    results = {"significant": [], "all": []}
    
    # Select appropriate dataframe based on feature type
    if feature_type == "connectivity":
        source_df = data.conn_df
        if source_df is None or source_df.empty:
            logger.warning("No connectivity data available for scatter plots")
            return results
    else:
        source_df = data.features_df
    
    metric_list = metrics if metrics else [None]
    
    for metric in metric_list:
        for band in bands:
            band_title = band.capitalize()
            band_color = get_band_color(band, config)
            
            for roi, chs in data.roi_map.items():
                roi_vals, is_valid = column_extractor(source_df, band, chs, metric)
                if not is_valid:
                    continue
                
                roi_plots_dir = data.plots_dir / sanitize_label(roi)
                ensure_dir(roi_plots_dir)
                
                title_rating = title_formatter(band_title, roi, "Rating", metric)
                x_label = x_label_formatter(band_title, metric)
                output_path = roi_plots_dir / filename_formatter(band, "rating", metric)
                feature_name = feature_name_formatter(band, metric)
                
                precomp = get_precomputed_stats_for_roi_band(rating_stats, roi, band, logger)
                
                _generate_single_scatter(
                    roi_vals=roi_vals,
                    target_vals=data.y,
                    roi=roi,
                    band=band,
                    band_title=band_title,
                    band_color=band_color,
                    metric=metric,
                    target_type="rating",
                    title=title_rating,
                    x_label=x_label,
                    y_label="Rating",
                    output_path=output_path,
                    method_code=method_code,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    chs=chs,
                    precomp_stats=precomp,
                    logger=logger,
                    config=config,
                    results=results,
                    feature_name=feature_name,
                )
                
                if do_temp and data.temp_series is not None and not data.temp_series.empty:
                    title_temp = title_formatter(band_title, roi, "Temp", metric)
                    output_path_temp = roi_plots_dir / filename_formatter(band, "temp", metric)
                    
                    precomp_t = get_precomputed_stats_for_roi_band(temp_stats, roi, band, logger)
                    
                    _generate_single_scatter(
                        roi_vals=roi_vals,
                        target_vals=data.temp_series,
                        roi=roi,
                        band=band,
                        band_title=band_title,
                        band_color=band_color,
                        metric=metric,
                        target_type="temp",
                        title=title_temp,
                        x_label=x_label,
                        y_label="Temperature (°C)",
                        output_path=output_path_temp,
                        method_code=method_code,
                        bootstrap_ci=bootstrap_ci,
                        rng=rng,
                        chs=chs,
                        precomp_stats=precomp_t,
                        logger=logger,
                        config=config,
                        results=results,
                        feature_name=feature_name,
                    )
    
    return results


###################################################################
# Helper Functions
###################################################################


def _get_title_components(band_title: str, target_type: str, roi_name: str, time_label: Optional[str] = None) -> str:
    roi_display = "Overall" if roi_name == "Overall" else roi_name
    time_suffix = format_time_suffix(time_label)
    return f"{band_title} power vs {target_type} — {roi_display}{time_suffix}"


def _get_target_suffix(target_type: str, config=None) -> str:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    target_rating = behavioral_config.get("target_rating", "rating")
    return "rating" if target_type == target_rating else "temp"


def _get_time_suffix(time_label: Optional[str]) -> str:
    return f"_{time_label}" if time_label else "_plateau"


def _get_base_filename(band: str, target_type: str, roi_name: str, config=None) -> str:
    band_safe = sanitize_label(band)
    target_suffix = _get_target_suffix(target_type, config)
    
    if roi_name == "All" or roi_name == "Overall":
        return f"scatter_pow_overall_{band_safe}_vs_{target_suffix}"
    return f"scatter_pow_{band_safe}_vs_{target_suffix}"


def _get_output_filename(band: str, target_type: str, roi_name: str, plot_type: str, time_label: Optional[str] = None, config: Optional[Any] = None) -> str:
    base = _get_base_filename(band, target_type, roi_name, config)
    time_suffix = _get_time_suffix(time_label)
    
    plot_type_map = {
        "scatter": f"{base}{time_suffix}",
        "residual_qc": f"residual_qc_{base}{time_suffix}",
        "residual_diagnostics": f"residual_diagnostics_{base}",
        "partial": f"{base}_partial",
    }
    
    return plot_type_map.get(plot_type, base)


###################################################################
# Partial Residuals Plotting
###################################################################


def _plot_partial_residuals(
    x_data: pd.Series,
    y_data: pd.Series,
    Z_data: pd.DataFrame,
    method_code: str,
    band_title: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    band_color: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    stats_df: Optional[pd.Series],
    logger: logging.Logger,
    config,
    roi_channels: Optional[List[str]] = None,
) -> None:
    mask = joint_valid_mask(x_data, y_data)
    x_part = x_data.iloc[mask]
    y_part = y_data.iloc[mask]
    Z_part = Z_data.iloc[mask]
    
    x_res, y_res, n_res = _compute_partial_residuals(
        x_part, y_part, Z_part, method_code, logger=logger,
        context=f"Partial residuals {roi_name} {target_type} {band_title}"
    )
    
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    if n_res < min_samples_for_plot:
        return
    
    r_resid, p_resid, n_partial, ci_resid = compute_partial_residuals_stats(
        x_res, y_res, stats_df, n_res, method_code, bootstrap_ci, rng
    )
    
    residual_xlabel, residual_ylabel = get_residual_labels(method_code, target_type)
    title = f"Partial residuals — {band_title} vs {target_type} — {roi_name}"
    output_path = output_dir / _get_output_filename(
        band_title.lower(), target_type, roi_name, "partial"
    )
    
    generate_correlation_scatter(
        x_data=x_res,
        y_data=y_res,
        x_label=residual_xlabel,
        y_label=residual_ylabel,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        bootstrap_ci=0,
        rng=rng,
        is_partial_residuals=True,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=(r_resid, p_resid, n_partial),
        annot_ci=ci_resid,
        config=config,
    )


###################################################################
# Temporal Correlation Plotting
###################################################################


def _get_temporal_columns(temporal_df: pd.DataFrame, band: str, time_label: str, config: Optional[Any] = None) -> List[str]:
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    v2_cols = [
        c for c in temporal_df.columns
        if str(c).startswith(f"power_{time_label}_{band}_ch_")
    ]
    if v2_cols:
        return v2_cols

    power_prefix = behavioral_config.get("power_prefix", "pow_")
    return [
        c for c in temporal_df.columns
        if c.startswith(f"{power_prefix}{band}_") and c.endswith(f"_{time_label}")
    ]


def _plot_single_temporal_correlation(
    temporal_vals: pd.Series,
    y_data: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    time_label: str,
    output_dir: Path,
    method_code: str,
    Z_covars: Optional[pd.DataFrame],
    covar_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    config,
    roi_channels: Optional[List[str]],
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    target_rating = behavioral_config.get("target_rating", "rating")
    
    has_covariates = Z_covars is not None and not Z_covars.empty
    if has_covariates:
        annotated_stats = None
        annot_ci = None
        bootstrap_ci_for_plot = 0
    else:
        r_temporal, p_temporal, n_eff_temporal, ci_temporal = compute_correlation_stats(
            temporal_vals, y_data, method_code, bootstrap_ci, rng, min_samples=min_samples_for_plot
        )
        annotated_stats = (r_temporal, p_temporal, n_eff_temporal)
        annot_ci = ci_temporal
        bootstrap_ci_for_plot = 0
    
    x_label = get_temporal_xlabel(time_label)
    y_label = "Rating" if target_type == target_rating else "Temperature (°C)"
    title = _get_title_components(band_title, target_type, roi_name, time_label)
    
    output_path = output_dir / _get_output_filename(
        band, target_type, roi_name, "scatter", time_label
    )
    
    generate_correlation_scatter(
        x_data=temporal_vals,
        y_data=y_data,
        x_label=x_label,
        y_label=y_label,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        Z_covars=Z_covars,
        covar_names=covar_names,
        bootstrap_ci=bootstrap_ci_for_plot,
        rng=rng,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=annotated_stats,
        annot_ci=annot_ci,
        config=config,
    )
    
    qc_output_path = output_dir / _get_output_filename(
        band, target_type, roi_name, "residual_qc", time_label
    )
    
    plot_residual_qc(
        x_data=temporal_vals,
        y_data=y_data,
        title_prefix=title,
        output_path=qc_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )


def _plot_temporal_correlations(
    temporal_df: pd.DataFrame,
    y_data: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    method_code: str,
    Z_covars: Optional[pd.DataFrame],
    covar_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    config,
    roi_channels: Optional[List[str]] = None,
) -> None:
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    time_labels = behavioral_config.get("time_labels", ["early", "mid", "late"])
    for time_label in time_labels:
        temporal_cols = _get_temporal_columns(temporal_df, band, time_label, config)
        if not temporal_cols:
            continue
        
        temporal_vals = temporal_df[temporal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        
        _plot_single_temporal_correlation(
            temporal_vals,
            y_data,
            band,
            band_title,
            band_color,
            target_type,
            roi_name,
            time_label,
            output_dir,
            method_code,
            Z_covars,
            covar_names,
            bootstrap_ci,
            rng,
            logger,
            config,
            roi_channels,
        )


###################################################################
# Target Correlation Plotting
###################################################################


def _plot_target_correlations(
    power_vals: pd.Series,
    target_vals: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    method_code: str,
    Z_covars: Optional[pd.DataFrame],
    covar_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    stats_df: Optional[pd.Series],
    logger: logging.Logger,
    config,
    temporal_df: Optional[pd.DataFrame] = None,
    roi_channels: Optional[List[str]] = None,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    
    has_covariates = Z_covars is not None and not Z_covars.empty
    if has_covariates:
        annotated_stats = None
        annot_ci = None
        bootstrap_ci_for_plot = 0
    else:
        r_val, p_val, n_eff, ci_val = compute_correlation_stats(
            power_vals, target_vals, method_code, bootstrap_ci, rng, min_samples=min_samples_for_plot
        )
        
        r_val, p_val, n_eff, ci_val = update_stats_from_dataframe(
            stats_df, r_val, p_val, n_eff, ci_val
        )
        
        annotated_stats = (r_val, p_val, n_eff)
        annot_ci = ci_val
        bootstrap_ci_for_plot = 0
    
    x_label, y_label = get_target_labels(target_type)
    title = _get_title_components(band_title, target_type, roi_name)
    
    output_path = output_dir / _get_output_filename(band, target_type, roi_name, "scatter")
    
    generate_correlation_scatter(
        x_data=power_vals,
        y_data=target_vals,
        x_label=x_label,
        y_label=y_label,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        Z_covars=Z_covars,
        covar_names=covar_names,
        bootstrap_ci=bootstrap_ci_for_plot,
        rng=rng,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=annotated_stats,
        annot_ci=annot_ci,
        config=config,
    )
    
    qc_output_path = output_dir / _get_output_filename(
        band, target_type, roi_name, "residual_qc"
    )
    
    plot_residual_qc(
        x_data=power_vals,
        y_data=target_vals,
        title_prefix=title,
        output_path=qc_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )
    
    if temporal_df is not None:
        _plot_temporal_correlations(
            temporal_df,
            target_vals,
            band,
            band_title,
            band_color,
            target_type,
            roi_name,
            output_dir,
            method_code,
            Z_covars,
            covar_names,
            bootstrap_ci,
            rng,
            logger,
            config,
            roi_channels,
        )
    
    diagnostics_output_path = output_dir / _get_output_filename(
        band, target_type, roi_name, "residual_diagnostics"
    )
    
    diagnostics_title = _get_title_components(band_title, target_type, roi_name, None).replace(" (plateau)", "")
    plot_regression_residual_diagnostics(
        x_data=power_vals,
        y_data=target_vals,
        title_prefix=diagnostics_title,
        output_path=diagnostics_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )
    
    if Z_covars is not None and not Z_covars.empty:
        _plot_partial_residuals(
            power_vals,
            target_vals,
            Z_covars,
            method_code,
            band_title,
            target_type,
            roi_name,
            output_dir,
            band_color,
            bootstrap_ci,
            rng,
            stats_df,
            logger,
            config,
            roi_channels,
        )


###################################################################
# Subject-Level Plotting Functions
###################################################################


def _plot_distribution_histogram(
    data: pd.Series,
    x_label: str,
    title: str,
    output_path: Path,
    plot_cfg: PlotConfig,
    config,
    logger: logging.Logger,
) -> None:
    if data.empty or data.notna().sum() < 3:
        logger.warning(f"Insufficient data for histogram: {title}")
        return
    
    fig_size = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    fig, ax = plt.subplots(figsize=fig_size)
    
    band_color = get_band_color("alpha", config)
    behavioral_config = plot_cfg.get_behavioral_config()
    default_bins = behavioral_config.get("histogram_default_bins", 30)
    bins = plot_cfg.style.histogram.bins if hasattr(plot_cfg.style.histogram, "bins") else default_bins
    
    ax.hist(
        data,
        bins=bins,
        color=band_color,
        alpha=plot_cfg.style.scatter.alpha,
        edgecolor=plot_cfg.style.histogram.edgecolor,
        linewidth=plot_cfg.style.histogram.edgewidth,
    )
    
    if len(data) > plot_cfg.validation.get("min_samples_for_kde", 3):
        kde = gaussian_kde(data)
        data_range = np.linspace(data.min(), data.max(), plot_cfg.style.kde_points)
        kde_vals = kde(data_range)
        kde_scale = compute_kde_scale(data, hist_bins=bins, kde_points=plot_cfg.style.kde_points)
        scaled_kde = kde_vals * kde_scale
        ax.plot(
            data_range,
            scaled_kde,
            color=plot_cfg.style.kde_color,
            linewidth=plot_cfg.style.kde_linewidth,
            alpha=plot_cfg.style.kde_alpha,
        )
    
    ax.set_xlabel(x_label, fontsize=plot_cfg.font.label)
    ax.set_ylabel("Frequency", fontsize=plot_cfg.font.label)
    ax.set_title(title, fontsize=plot_cfg.font.title, fontweight="bold")
    
    stats_text = f"n = {len(data)}\nMean = {data.mean():.2f}\nSD = {data.std():.2f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_cfg.font.title,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=plot_cfg.style.alpha_text_box),
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()
    
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)


def _load_subject_data(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
    partial_covars: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, pd.Series, mne.Info, Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, List[str]]]:
    """Wrapper for load_subject_scatter_data for backward compatibility.
    
    Returns 8-tuple (drops conn_df for backward compatibility with existing callers).
    Use load_subject_scatter_data directly if you need connectivity data.
    """
    result = load_subject_scatter_data(subject, task, deriv_root, config, logger, partial_covars)
    # Return first 8 elements for backward compatibility
    return result[:8]


def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    events = _load_events_df(subject, task, config=config)
    if events is None or len(events) == 0:
        logger.warning(f"No events for psychometrics: sub-{subject}")
        return

    psych_temp_columns = config.get("event_columns.temperature", [])
    rating_columns = config.get("event_columns.rating", [])
    pain_binary_columns = config.get("event_columns.pain_binary", [])
    
    temp_col = _pick_first_column(events, psych_temp_columns)
    rating_col = _pick_first_column(events, rating_columns)
    pain_col = _pick_first_column(events, pain_binary_columns)

    if temp_col is None:
        logger.warning(f"Psychometrics: no temperature column found; skipping for sub-{subject}.")
        return

    temp = pd.to_numeric(events[temp_col], errors="coerce")
    
    psychometrics_dir = plots_dir / "psychometrics"
    ensure_dir(psychometrics_dir)
    
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    
    valid_mask = temp.notna()
    if rating_col is not None:
        rating = pd.to_numeric(events[rating_col], errors="coerce")
        valid_mask = valid_mask & rating.notna()
    else:
        rating = None
    
    if not valid_mask.sum() >= min_samples_for_plot:
        logger.warning(f"Insufficient valid data for psychometrics (n={valid_mask.sum()} < {min_samples_for_plot}); skipping for sub-{subject}")
        return
    
    temp_valid = temp[valid_mask]
    
    if rating is not None:
        rating_valid = rating[valid_mask]
        
        method_code = behavioral_config.get("method_spearman", "spearman")
        default_rng_seed = behavioral_config.get("default_rng_seed", 42)
        rng = np.random.default_rng(default_rng_seed)
        
        x_label = "Temperature (°C)"
        y_label = "Rating"
        
        output_path = psychometrics_dir / f"psychometrics_temp_vs_rating_sub-{subject}"
        
        generate_correlation_scatter(
            x_data=temp_valid,
            y_data=rating_valid,
            x_label=x_label,
            y_label=y_label,
            title_prefix=f"Psychometrics — Temperature vs Rating — sub-{subject}",
            band_color=get_band_color("alpha", config),
            output_path=output_path,
            method_code=method_code,
            Z_covars=None,
            covar_names=None,
            bootstrap_ci=0,
            rng=rng,
            roi_channels=None,
            logger=logger,
            annotated_stats=None,
            annot_ci=None,
            config=config,
        )
    
    _plot_distribution_histogram(
        data=temp_valid,
        x_label="Temperature (°C)",
        title=f"Temperature Distribution — sub-{subject}",
        output_path=psychometrics_dir / f"psychometrics_temp_distribution_sub-{subject}",
        plot_cfg=plot_cfg,
        config=config,
        logger=logger,
    )
    
    logger.info(f"Completed psychometrics plotting for sub-{subject}")


def plot_power_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    rating_stats: Optional[pd.DataFrame] = None,
    temp_stats: Optional[pd.DataFrame] = None,
    plots_dir: Optional[Path] = None,
    config=None,
) -> None:
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ROI power scatter plotting for sub-{subject}")
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    
    if plots_dir is None:
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    if task is None:
        task = config.task

    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    
    temporal_df, pow_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map = _load_subject_data(
        subject, task, deriv_root, config, logger, partial_covars
    )
    if temporal_df is None:
        return
    
    if rating_stats is None:
        logger.warning("ROI rating statistics not provided; plots will use empirical correlations only.")
    if do_temp and temp_stats is None:
        logger.warning("ROI temperature statistics not provided; temperature annotations will be omitted.")

    if use_spearman:
        method_code = "spearman"
    else:
        method_code = "pearson"
    covar_names = list(Z_df_full.columns) if Z_df_full is not None and not Z_df_full.empty else None
    covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None and not Z_df_temp.empty else None

    power_prefix = behavioral_config.get("power_prefix", "pow_")
    overall_roi_keys = behavioral_config.get("overall_roi_keys", ["overall", "all", "global"])
    target_rating = behavioral_config.get("target_rating", "rating")
    target_temperature = behavioral_config.get("target_temperature", "temperature")
    power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    
    for band in power_bands_to_use:
        band_cols = {
            c for c in pow_df.columns
            if str(c).startswith(f"power_plateau_{band}_ch_")
        }
        using_v2 = True
        if not band_cols:
            using_v2 = False
            band_cols = {c for c in pow_df.columns if str(c).startswith(f"{power_prefix}{band}_")}
        if not band_cols:
            continue
        
        band_title = band.capitalize()
        band_color = get_band_color(band, config)
        overall_vals = pow_df[list(band_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        overall_plots_dir = plots_dir / "overall"
        ensure_dir(overall_plots_dir)
        
        stats_overall_rating = extract_overall_statistics(rating_stats, band, overall_keys=overall_roi_keys)
        _plot_target_correlations(
            overall_vals, y, band, band_title, band_color, target_rating, "Overall",
            overall_plots_dir, method_code, Z_df_full, covar_names, bootstrap_ci, rng,
            stats_overall_rating, logger, config, temporal_df
        )

        if do_temp and temp_series is not None and not temp_series.empty:
            stats_overall_temp = extract_overall_statistics(temp_stats, band, overall_keys=overall_roi_keys)
            _plot_target_correlations(
                overall_vals, temp_series, band, band_title, band_color, target_temperature, "Overall",
                overall_plots_dir, method_code, Z_df_temp, covar_names_temp, bootstrap_ci, rng,
                stats_overall_temp, logger, config, temporal_df
            )

        for roi, chs in roi_map.items():
            if using_v2:
                roi_cols = []
                for ch in chs:
                    candidates = [
                        f"power_plateau_{band}_ch_{ch}_logratio",
                        f"power_plateau_{band}_ch_{ch}_log10raw",
                    ]
                    col = next((c for c in candidates if c in band_cols), None)
                    if col is not None:
                        roi_cols.append(col)
            else:
                roi_cols = [
                    f"{power_prefix}{band}_{ch}"
                    for ch in chs
                    if f"{power_prefix}{band}_{ch}" in band_cols
                ]
            if not roi_cols:
                continue
            
            roi_vals = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            roi_plots_dir = plots_dir / "roi_scatters" / sanitize_label(roi)
            ensure_dir(roi_plots_dir)

            stats_roi_rating = extract_roi_statistics(rating_stats, roi, band)
            _plot_target_correlations(
                roi_vals, y, band, band_title, band_color, target_rating, roi,
                roi_plots_dir, method_code, Z_df_full, covar_names, bootstrap_ci, rng,
                stats_roi_rating, logger, config, temporal_df, roi_channels=chs
            )

            if do_temp and temp_series is not None and not temp_series.empty:
                stats_roi_temp = extract_roi_statistics(temp_stats, roi, band)
                _plot_target_correlations(
                    roi_vals, temp_series, band, band_title, band_color, target_temperature, roi,
                    roi_plots_dir, method_code, Z_df_temp, covar_names_temp, bootstrap_ci, rng,
                    stats_roi_temp, logger, config, temporal_df, roi_channels=chs
                )


###################################################################
# Behavioral Response Patterns
###################################################################


def _get_behavioral_config(plot_cfg):
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _create_rating_distribution_plot(y: pd.Series, config=None) -> Tuple[plt.Figure, plt.Axes]:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    rating_config = behavioral_config.get("rating_distribution", {})
    
    figsize_width = rating_config.get("figsize_width", 8)
    figsize_height = rating_config.get("figsize_height", 6)
    bins = rating_config.get("bins", 20)
    alpha = rating_config.get("alpha", 0.7)
    edgecolor = rating_config.get("edgecolor", "black")
    
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    ax.hist(y.dropna(), bins=bins, alpha=alpha, edgecolor=edgecolor)
    grid_alpha = rating_config.get("grid_alpha", 0.3)
    ax.set_xlabel('Pain Rating')
    ax.set_ylabel('Frequency')
    ax.set_title('Rating Distribution')
    ax.grid(True, alpha=grid_alpha)
    
    mean_rating = y.mean()
    std_rating = y.std()
    text_box_alpha = rating_config.get("text_box_alpha", 0.8)
    ax.text(
        0.02, 0.98, f'Mean: {mean_rating:.2f} ± {std_rating:.2f}',
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=text_box_alpha)
    )
    
    return fig, ax


def plot_behavioral_response_patterns(
    y: pd.Series,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config=None,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    fig, ax = _create_rating_distribution_plot(y, config)
    
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f'sub-{subject}_rating_distribution',
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config)
    )
    plt.close(fig)


###################################################################
# Top Behavioral Predictors
###################################################################


def _load_correlation_stats(stats_dir: Path, logger: logging.Logger, config: Optional[Any] = None) -> Optional[pd.DataFrame]:
    if config is None:
        target_rating = "rating"
        target_temperature = "temperature"
    else:
        plot_cfg = get_plot_config(config)
        behavioral_config = _get_behavioral_config(plot_cfg)
        target_rating = behavioral_config.get("target_rating", "rating")
        target_temperature = behavioral_config.get("target_temperature", "temperature")
    candidate_files = [
        (target_rating, stats_dir / "corr_stats_pow_combined_vs_rating.tsv"),
        (target_rating, stats_dir / "corr_stats_power_combined_vs_rating.tsv"),
        (target_temperature, stats_dir / "corr_stats_pow_combined_vs_temp.tsv"),
        (target_temperature, stats_dir / "corr_stats_power_combined_vs_temp.tsv"),
    ]
    
    frames = []
    for target_label, path in candidate_files:
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            df["target"] = target_label
            frames.append(df)
    
    if not frames:
        logger.warning(
            "No combined correlation stats found for rating or temperature. Expected one of: "
            + ", ".join(str(p) for _, p in candidate_files)
        )
        return None
    
    return pd.concat(frames, axis=0, ignore_index=True)


def _filter_significant_predictors(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return df[
        (df['p'] <= alpha) &
        df['r'].notna() &
        df['p'].notna() &
        df['channel'].notna() &
        df['band'].notna()
    ].copy()


def _create_predictor_labels(df: pd.DataFrame) -> pd.Series:
    if 'target' in df.columns:
        return df['channel'] + ' (' + df['band'] + ') [' + df['target'].astype(str) + ']'
    return df['channel'] + ' (' + df['band'] + ')'


def _create_predictors_plot(df_top: pd.DataFrame, top_n: int, alpha: float, config) -> Tuple[plt.Figure, plt.Axes]:
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    predictors_config = behavioral_config.get("predictors_plot", {})
    
    figsize_width = predictors_config.get("figsize_width", 10)
    figsize_height_base = predictors_config.get("figsize_height_base", 8)
    figsize_height_per_predictor = predictors_config.get("figsize_height_per_predictor", 0.4)
    fig, ax = plt.subplots(figsize=(figsize_width, max(figsize_height_base, top_n * figsize_height_per_predictor)))
    
    band_colors = {str(band): get_band_color(band, config) for band in df_top['band'].unique()}
    colors = [band_colors[band] for band in df_top['band']]
    
    bar_alpha = predictors_config.get("bar_alpha", 0.8)
    bar_edgecolor = predictors_config.get("bar_edgecolor", "black")
    bar_linewidth = predictors_config.get("bar_linewidth", 0.5)
    y_pos = np.arange(len(df_top))
    ax.barh(y_pos, df_top['abs_r'], color=colors, alpha=bar_alpha, edgecolor=bar_edgecolor, linewidth=bar_linewidth)
    
    ax.set_yticks(y_pos)
    label_fontsize = predictors_config.get("label_fontsize", 11)
    ax.set_yticklabels(df_top['predictor'], fontsize=label_fontsize)
    xlabel_fontsize = predictors_config.get("xlabel_fontsize", 12)
    ax.set_xlabel(f"|Spearman ρ| with Behavior (p < {alpha})", fontweight='bold', fontsize=xlabel_fontsize)
    title_fontsize = predictors_config.get("title_fontsize", 14)
    title_pad = predictors_config.get("title_pad", 20)
    ax.set_title(f'Top {top_n} Significant Behavioral Predictors', fontweight='bold', fontsize=title_fontsize, pad=title_pad)
    
    value_fontsize = predictors_config.get("value_fontsize", 10)
    value_x_offset = predictors_config.get("value_x_offset", 0.01)
    for i, (_, row) in enumerate(df_top.iterrows()):
        r_val, p_val, abs_r_val = row['r'], row['p'], row['abs_r']
        sign_str = '(+)' if r_val >= 0 else '(-)'
        x_pos = abs_r_val + value_x_offset
        ax.text(x_pos, i, f'{abs_r_val:.3f} {sign_str} (p={p_val:.3f})',
               va='center', ha='left', fontsize=value_fontsize, fontweight='normal')
    
    max_r = df_top['abs_r'].max()
    xlim_multiplier = predictors_config.get("xlim_multiplier", 1.25)
    ax.set_xlim(0, max_r * xlim_multiplier)
    grid_alpha = predictors_config.get("grid_alpha", 0.3)
    grid_linestyle = predictors_config.get("grid_linestyle", "-")
    grid_linewidth = predictors_config.get("grid_linewidth", 0.5)
    ax.grid(True, axis='x', alpha=grid_alpha, linestyle=grid_linestyle, linewidth=grid_linewidth)
    ax.set_axisbelow(True)
    
    return fig, ax


def _export_top_predictors(df_top: pd.DataFrame, stats_dir: Path, top_n: int, logger: logging.Logger) -> None:
    top_predictors_file = stats_dir / f"top_{top_n}_behavioral_predictors.tsv"
    export_cols = ['predictor', 'channel', 'band', 'r', 'abs_r', 'p', 'n']
    if 'target' in df_top.columns and 'target' not in export_cols:
        export_cols = ['target'] + export_cols
    
    df_top_export = df_top[export_cols].copy()
    df_top_export = df_top_export.sort_values('abs_r', ascending=False)
    df_top_export.to_csv(top_predictors_file, sep="\t", index=False)
    logger.info(f"Exported top predictors data: {top_predictors_file}")


def plot_top_behavioral_predictors(subject: str, task: Optional[str] = None, alpha: float = None, top_n: int = None, plots_dir: Optional[Path] = None) -> None:
    config = load_settings()
    if task is None:
        task = config.task
    
    alpha = alpha or config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    top_n = top_n or int(config.get("behavior_analysis.predictors.top_n", 20))
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Creating top {top_n} behavioral predictors plot for sub-{subject}")
    
    if plots_dir is None:
        deriv_root = Path(config.deriv_root)
        plot_cfg = get_plot_config(config) if config else None
        behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {}) if plot_cfg else {}
        plot_subdir = behavioral_config.get("plot_subdir", "behavior") if plot_cfg else "behavior"
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    ensure_dir(plots_dir)
    
    df = _load_correlation_stats(stats_dir, logger)
    if df is None:
        return
    
    df_sig = _filter_significant_predictors(df, alpha)
    if df_sig.empty:
        logger.warning(f"No significant correlations found (p <= {alpha})")
        return
    
    df_sig['abs_r'] = df_sig['r'].abs()
    df_top = df_sig.nlargest(top_n, 'abs_r')
    
    if df_top.empty:
        logger.warning("No top correlations to plot")
        return
    
    df_top['predictor'] = _create_predictor_labels(df_top)
    df_top = df_top.sort_values('abs_r', ascending=True)
    
    fig, ax = _create_predictors_plot(df_top, top_n, alpha, config)
    plt.tight_layout()
    
    output_path = plots_dir / f'sub-{subject}_top_{top_n}_behavioral_predictors'
    if plot_cfg is None:
        plot_cfg = get_plot_config(config)
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi, bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, footer=_get_behavior_footer(config))
    plt.close(fig)
    
    logger.info(f"Saved top {top_n} behavioral predictors plot: {output_path}.png")
    if 'target' in df.columns:
        counts_by_tgt = df_sig['target'].value_counts().to_dict() if len(df_sig) else {}
        logger.info(f"Found {len(df_top)} significant predictors across targets {counts_by_tgt} (out of {len(df)} total correlations)")
    else:
        logger.info(f"Found {len(df_top)} significant predictors (out of {len(df)} total correlations)")
    
    _export_top_predictors(df_top, stats_dir, top_n, logger)


###################################################################
# Dynamics ROI Scatter (LZC, PE, Hjorth)
###################################################################


def _get_dynamics_metrics(config) -> List[str]:
    return config.get("dynamics.metrics", ["lzc", "pe", "hjorth_mobility", "hjorth_complexity"])


def _get_dynamics_metric_title(metric: str) -> str:
    titles = {
        "lzc": "LZC",
        "pe": "PE", 
        "hjorth_mobility": "Hjorth Mob.",
        "hjorth_complexity": "Hjorth Comp.",
    }
    return titles.get(metric, metric.upper())


def _extract_dynamics_columns(features_df: pd.DataFrame, band: str, metric: str, roi_channels: List[str]) -> List[str]:
    cols = []
    for col in features_df.columns:
        if f"dynamics_plateau_{band}_" not in col:
            continue
        if f"_{metric}" not in col:
            continue
        for ch in roi_channels:
            if f"_ch_{ch}_" in col or col.endswith(f"_ch_{ch}"):
                cols.append(col)
                break
    return cols


def _extract_dynamics_values(
    features_df: pd.DataFrame, 
    band: str, 
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    """Extract dynamics feature values for a ROI."""
    if metric is None:
        return pd.Series(dtype=float), False
    cols = _extract_dynamics_columns(features_df, band, metric, roi_channels)
    if not cols:
        return pd.Series(dtype=float), False
    vals = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return vals, True


def _format_dynamics_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    """Format title for dynamics scatter plot."""
    metric_title = _get_dynamics_metric_title(metric) if metric else "Dynamics"
    return f"{metric_title} ({band_title}) vs {target} — {roi}"


def _format_dynamics_x_label(band_title: str, metric: Optional[str]) -> str:
    """Format x-axis label for dynamics scatter plot."""
    metric_title = _get_dynamics_metric_title(metric) if metric else "Dynamics"
    return f"{metric_title} Power"


def _format_dynamics_filename(band: str, target: str, metric: Optional[str]) -> str:
    """Format filename for dynamics scatter plot."""
    metric_safe = metric if metric else "dynamics"
    return f"scatter_dynamics_{metric_safe}_{band}_vs_{target}"


def _format_dynamics_feature_name(band: str, metric: Optional[str]) -> str:
    """Format feature name for dynamics results."""
    return f"dynamics_{metric}_{band}" if metric else f"dynamics_{band}"


def _extract_aperiodic_values(
    features_df: pd.DataFrame, 
    band: str, 
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    """Extract aperiodic feature values for a ROI."""
    if metric is None:
        return pd.Series(dtype=float), False
    roi_cols = []
    for col in features_df.columns:
        if "aperiodic_plateau_broadband_ch_" not in col:
            continue
        if f"_{metric}" not in col:
            continue
        for ch in roi_channels:
            if f"_ch_{ch}_" in col:
                roi_cols.append(col)
                break
    if not roi_cols:
        return pd.Series(dtype=float), False
    vals = features_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return vals, True


def _format_aperiodic_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    """Format title for aperiodic scatter plot."""
    metric_title = f"1/f {metric.capitalize()}" if metric else "1/f"
    return f"{metric_title} vs {target} — {roi}"


def _format_aperiodic_x_label(band_title: str, metric: Optional[str]) -> str:
    """Format x-axis label for aperiodic scatter plot."""
    return f"1/f {metric.capitalize()}" if metric else "1/f"


def _format_aperiodic_filename(band: str, target: str, metric: Optional[str]) -> str:
    """Format filename for aperiodic scatter plot."""
    metric_safe = metric if metric else "aperiodic"
    return f"scatter_aperiodic_{metric_safe}_vs_{target}"


def _format_aperiodic_feature_name(band: str, metric: Optional[str]) -> str:
    """Format feature name for aperiodic results."""
    return f"aperiodic_{metric}" if metric else "aperiodic"


def _extract_within_roi_connectivity(features_df: pd.DataFrame, band: str, roi_channels: List[str]) -> pd.Series:
    import re
    cols = []
    for col in features_df.columns:
        if f"conn_plateau_{band}_" not in col:
            continue
        match = re.search(r'_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_', col)
        if match:
            ch1, ch2 = match.group(1), match.group(2)
            if ch1 in roi_channels and ch2 in roi_channels:
                cols.append(col)
    
    if not cols:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)
    return features_df[cols].mean(axis=1)


def _extract_connectivity_values(
    features_df: pd.DataFrame, 
    band: str, 
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    """Extract connectivity feature values for a ROI."""
    vals = _extract_within_roi_connectivity(features_df, band, roi_channels)
    if vals.isna().all():
        return vals, False
    return vals, True


def _format_connectivity_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    """Format title for connectivity scatter plot."""
    return f"wPLI ({band_title}) vs {target} — {roi}"


def _format_connectivity_x_label(band_title: str, metric: Optional[str]) -> str:
    """Format x-axis label for connectivity scatter plot."""
    return f"wPLI ({band_title})"


def _format_connectivity_filename(band: str, target: str, metric: Optional[str]) -> str:
    """Format filename for connectivity scatter plot."""
    return f"scatter_conn_{band}_vs_{target}"


def _format_connectivity_feature_name(band: str, metric: Optional[str]) -> str:
    """Format feature name for connectivity results."""
    return f"conn_{band}"


def _extract_itpc_columns(features_df: pd.DataFrame, band: str, roi_channels: List[str]) -> List[str]:
    cols = []
    for col in features_df.columns:
        if f"itpc_plateau_{band}_ch_" not in col:
            continue
        for ch in roi_channels:
            if f"_ch_{ch}_" in col or col.endswith(f"_ch_{ch}"):
                cols.append(col)
                break
    return cols


def _extract_itpc_values(
    features_df: pd.DataFrame, 
    band: str, 
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    """Extract ITPC feature values for a ROI."""
    cols = _extract_itpc_columns(features_df, band, roi_channels)
    if not cols:
        return pd.Series(dtype=float), False
    vals = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return vals, True


def _format_itpc_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    """Format title for ITPC scatter plot."""
    return f"ITPC ({band_title}) vs {target} — {roi}"


def _format_itpc_x_label(band_title: str, metric: Optional[str]) -> str:
    """Format x-axis label for ITPC scatter plot."""
    return f"ITPC ({band_title})"


def _format_itpc_filename(band: str, target: str, metric: Optional[str]) -> str:
    """Format filename for ITPC scatter plot."""
    return f"scatter_itpc_{band}_vs_{target}"


def _format_itpc_feature_name(band: str, metric: Optional[str]) -> str:
    """Format feature name for ITPC results."""
    return f"itpc_{band}"


def plot_dynamics_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    plots_dir: Optional[Path] = None,
    config=None,
) -> Dict[str, Any]:
    """Generate scatter plots for dynamics features vs behavioral outcomes."""
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting dynamics ROI scatter plotting for sub-{subject}")
    
    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    
    data = _setup_scatter_context(subject, deriv_root, task, plots_dir, "dynamics", config, logger)
    if data is None:
        return {"significant": [], "all": []}
    
    rating_stats = load_precomputed_correlations(data.stats_dir, "dynamics", "rating", logger)
    temp_stats = load_precomputed_correlations(data.stats_dir, "dynamics", "temperature", logger) if do_temp else None
    
    results = _create_roi_scatter_plots(
        data=data,
        feature_type="dynamics",
        column_extractor=_extract_dynamics_values,
        title_formatter=_format_dynamics_title,
        x_label_formatter=_format_dynamics_x_label,
        filename_formatter=_format_dynamics_filename,
        feature_name_formatter=_format_dynamics_feature_name,
        bands=config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"]),
        metrics=_get_dynamics_metrics(config),
        method_code="spearman" if use_spearman else "pearson",
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        do_temp=do_temp,
        rating_stats=rating_stats,
        temp_stats=temp_stats,
        logger=logger,
        config=config,
    )
    
    logger.info(f"Dynamics scatter: {len(results['significant'])} significant of {len(results['all'])} total")
    return results


###################################################################
# Aperiodic ROI Scatter (slope, offset)
###################################################################


def plot_aperiodic_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    plots_dir: Optional[Path] = None,
    config=None,
) -> Dict[str, Any]:
    """Generate scatter plots for aperiodic (1/f) features vs behavioral outcomes."""
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting aperiodic ROI scatter plotting for sub-{subject}")
    
    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    
    data = _setup_scatter_context(subject, deriv_root, task, plots_dir, "aperiodic", config, logger)
    if data is None:
        return {"significant": [], "all": []}
    
    rating_stats = load_precomputed_correlations(data.stats_dir, "aperiodic", "rating", logger)
    temp_stats = load_precomputed_correlations(data.stats_dir, "aperiodic", "temperature", logger) if do_temp else None
    
    results = _create_roi_scatter_plots(
        data=data,
        feature_type="aperiodic",
        column_extractor=_extract_aperiodic_values,
        title_formatter=_format_aperiodic_title,
        x_label_formatter=_format_aperiodic_x_label,
        filename_formatter=_format_aperiodic_filename,
        feature_name_formatter=_format_aperiodic_feature_name,
        bands=["broadband"],
        metrics=["slope", "offset"],
        method_code="spearman" if use_spearman else "pearson",
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        do_temp=do_temp,
        rating_stats=rating_stats,
        temp_stats=temp_stats,
        logger=logger,
        config=config,
    )
    
    logger.info(f"Aperiodic scatter: {len(results['significant'])} significant of {len(results['all'])} total")
    return results


###################################################################
# Connectivity ROI Scatter (wPLI, AEC)
###################################################################


def plot_connectivity_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    plots_dir: Optional[Path] = None,
    config=None,
) -> Dict[str, Any]:
    """Generate scatter plots for connectivity features vs behavioral outcomes."""
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting connectivity ROI scatter plotting for sub-{subject}")
    
    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    
    data = _setup_scatter_context(subject, deriv_root, task, plots_dir, "connectivity", config, logger)
    if data is None:
        return {"significant": [], "all": []}
    
    rating_stats = load_precomputed_correlations(data.stats_dir, "connectivity", "rating", logger)
    temp_stats = load_precomputed_correlations(data.stats_dir, "connectivity", "temperature", logger) if do_temp else None
    
    results = _create_roi_scatter_plots(
        data=data,
        feature_type="connectivity",
        column_extractor=_extract_connectivity_values,
        title_formatter=_format_connectivity_title,
        x_label_formatter=_format_connectivity_x_label,
        filename_formatter=_format_connectivity_filename,
        feature_name_formatter=_format_connectivity_feature_name,
        bands=config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"]),
        metrics=None,
        method_code="spearman" if use_spearman else "pearson",
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        do_temp=do_temp,
        rating_stats=rating_stats,
        temp_stats=temp_stats,
        logger=logger,
        config=config,
    )
    
    logger.info(f"Connectivity scatter: {len(results['significant'])} significant of {len(results['all'])} total")
    return results


###################################################################
# ITPC ROI Scatter
###################################################################


def plot_itpc_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    plots_dir: Optional[Path] = None,
    config=None,
) -> Dict[str, Any]:
    """Generate scatter plots for ITPC features vs behavioral outcomes."""
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ITPC ROI scatter plotting for sub-{subject}")
    
    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    
    data = _setup_scatter_context(subject, deriv_root, task, plots_dir, "itpc", config, logger)
    if data is None:
        return {"significant": [], "all": []}
    
    rating_stats = load_precomputed_correlations(data.stats_dir, "itpc", "rating", logger)
    temp_stats = load_precomputed_correlations(data.stats_dir, "itpc", "temperature", logger) if do_temp else None
    
    results = _create_roi_scatter_plots(
        data=data,
        feature_type="itpc",
        column_extractor=_extract_itpc_values,
        title_formatter=_format_itpc_title,
        x_label_formatter=_format_itpc_x_label,
        filename_formatter=_format_itpc_filename,
        feature_name_formatter=_format_itpc_feature_name,
        bands=config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"]),
        metrics=None,
        method_code="spearman" if use_spearman else "pearson",
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        do_temp=do_temp,
        rating_stats=rating_stats,
        temp_stats=temp_stats,
        logger=logger,
        config=config,
    )
    
    logger.info(f"ITPC scatter: {len(results['significant'])} significant of {len(results['all'])} total")
    return results


###################################################################
# Registry adapters
###################################################################


def _record_results(ctx, result):
    if isinstance(result, dict) and "all" in result:
        ctx.all_results.append(result)


@BehaviorPlotRegistry.register("psychometrics", name="psychometrics")
def run_psychometrics(ctx, saved_plots):
    plot_psychometrics(ctx.subject, ctx.deriv_root, ctx.task, ctx.config)
    saved_plots["psychometrics"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="power_roi_scatter")
def run_power_scatter(ctx, saved_plots):
    result = plot_power_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
        rating_stats=ctx.rating_stats,
        temp_stats=ctx.temp_stats,
    )
    _record_results(ctx, result)
    saved_plots["power_roi_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="dynamics_scatter")
def run_dynamics_scatter(ctx, saved_plots):
    result = plot_dynamics_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["dynamics_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="aperiodic_scatter")
def run_aperiodic_scatter(ctx, saved_plots):
    result = plot_aperiodic_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["aperiodic_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="connectivity_scatter")
def run_connectivity_scatter(ctx, saved_plots):
    result = plot_connectivity_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["connectivity_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="itpc_scatter")
def run_itpc_scatter(ctx, saved_plots):
    result = plot_itpc_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["itpc_scatter"] = ctx.plots_dir

