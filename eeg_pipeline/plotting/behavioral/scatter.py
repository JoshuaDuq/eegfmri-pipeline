from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from eeg_pipeline.utils.data.loading import (
    _build_covariate_matrices,
    _load_features_and_targets,
    _pick_first_column,
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io.general import (
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    save_fig,
    _load_events_df,
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    sanitize_label,
    get_subject_logger,
    get_default_logger as _get_default_logger,
    get_default_config as _get_default_config,
    get_residual_labels,
    get_target_labels,
    get_temporal_xlabel,
    format_time_suffix,
)
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
    temporal_df, pow_df, _conn_df, y, info = _load_features_and_targets(subject, task, deriv_root, config)
    y = pd.to_numeric(y, errors="coerce")
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=deriv_root, bids_root=config.bids_root, config=config, logger=logger
    )
    if epochs is None:
        logger.error(f"Could not find epochs for ROI scatter plots: sub-{subject}")
        return None, pow_df, y, info, None, None, None, {}
    
    temp_series = None
    temp_col = None
    psych_temp_columns = config.get("event_columns.temperature", [])
    if aligned_events is not None:
        temp_col = _pick_first_column(aligned_events, psych_temp_columns)
        if temp_col:
            temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    
    Z_df_full, Z_df_temp = _build_covariate_matrices(aligned_events, partial_covars, temp_col, config=config)
    roi_map = _build_rois(info, config=config)
    
    return temporal_df, pow_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map


def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations")
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
        plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations")
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
        band_cols = {c for c in pow_df.columns if c.startswith(f"{power_prefix}{band}_")}
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
            roi_cols = [f"{power_prefix}{band}_{ch}" for ch in chs if f"{power_prefix}{band}_{ch}" in band_cols]
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
        (target_temperature, stats_dir / "corr_stats_pow_combined_vs_temp.tsv"),
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
        plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations") if plot_cfg else "04_behavior_correlations"
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

