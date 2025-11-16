from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, StrMethodFormatter
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    _build_covariate_matrices,
    _load_features_and_targets,
    _pick_first_column,
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io_utils import (
    deriv_group_plots_path,
    deriv_group_stats_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_aligned_lengths,
    ensure_dir,
    fdr_bh_reject,
    find_connectivity_features_path,
    save_fig,
    _load_events_df,
)
from eeg_pipeline.utils.plotting_config import get_plot_config
from eeg_pipeline.utils.io_utils import get_group_logger, get_subject_logger
from eeg_pipeline.utils.io_utils import (
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    logratio_to_pct as _logratio_to_pct,
    pct_to_logratio as _pct_to_logratio,
    sanitize_label,
    get_default_logger as _get_default_logger,
    get_default_config as _get_default_config,
    format_channel_list_for_display,
    format_roi_description,
    get_residual_labels,
    get_target_labels,
    get_temporal_xlabel,
    format_time_suffix,
)
from eeg_pipeline.utils.stats_utils import (
    format_correlation_stats_text,
    compute_partial_residuals as _compute_partial_residuals,
    bootstrap_corr_ci as _bootstrap_corr_ci,
    compute_group_corr_stats as _compute_group_corr_stats,
    partial_corr_xy_given_Z as _partial_corr_xy_given_Z,
    partial_residuals_xy_given_Z as _partial_residuals_xy_given_Z,
    joint_valid_mask,
    compute_linear_residuals,
    center_series,
    zscore_series,
    apply_pooling_strategy,
    compute_correlation_stats,
    compute_partial_residuals_stats,
    compute_band_correlations,
    compute_connectivity_correlations,
    compute_kde_scale,
    compute_correlation_vmax,
    prepare_data_for_plotting,
    prepare_data_without_validation,
    prepare_group_data,
    extract_roi_statistics,
)
from eeg_pipeline.utils.data_loading import (
    prepare_partial_correlation_data,
    extract_common_dataframe_columns,
    prepare_group_partial_residuals_data,
    prepare_group_band_roi_data,
    prepare_topomap_correlation_data,
)
from eeg_pipeline.utils.stats_utils import (
    extract_overall_statistics,
    update_stats_from_dataframe,
)
from eeg_pipeline.utils.tfr_utils import build_rois_from_info as _build_rois


###################################################################
# Plot Builders
###################################################################


def _get_behavioral_config(plot_cfg):
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _create_scatter_figure(plot_cfg) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig_size = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    
    gridspec_params = plot_cfg.get_gridspec_params()
    width_ratios = gridspec_params["width_ratios"]
    height_ratios = gridspec_params["height_ratios"]
    hspace = gridspec_params["hspace"]
    wspace = gridspec_params["wspace"]
    left = gridspec_params["left"]
    right = gridspec_params["right"]
    top = gridspec_params["top"]
    bottom = gridspec_params["bottom"]
    
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)
    return fig, ax_main, ax_histx, ax_histy




def _plot_histogram_with_kde(
    ax: plt.Axes,
    data: pd.Series,
    band_color: str,
    plot_cfg,
    orientation: str = "vertical",
) -> None:
    bins = plot_cfg.get_histogram_bins(plot_type="behavioral")
    hist_kwargs = {
        "bins": bins,
        "color": band_color,
        "alpha": plot_cfg.style.scatter.alpha,
        "edgecolor": plot_cfg.style.histogram.edgecolor,
        "linewidth": plot_cfg.style.histogram.edgewidth,
    }
    if orientation == "horizontal":
        hist_kwargs["orientation"] = "horizontal"
    ax.hist(data, **hist_kwargs)
    
    min_samples = plot_cfg.validation.get("min_samples_for_kde", 3)
    if len(data) > min_samples:
        kde = gaussian_kde(data)
        data_range = np.linspace(data.min(), data.max(), plot_cfg.style.kde_points)
        kde_vals = kde(data_range)
        kde_scale = compute_kde_scale(data, hist_bins=bins, kde_points=plot_cfg.style.kde_points)
        scaled_kde = kde_vals * kde_scale
        if orientation == "horizontal":
            ax.plot(scaled_kde, data_range, color=plot_cfg.style.kde_color, 
                    linewidth=plot_cfg.style.kde_linewidth, alpha=plot_cfg.style.kde_alpha)
        else:
            ax.plot(data_range, scaled_kde, color=plot_cfg.style.kde_color, 
                    linewidth=plot_cfg.style.kde_linewidth, alpha=plot_cfg.style.kde_alpha)


def _should_show_percentage_axis(x_label: str) -> bool:
    power_indicators = ["log10(power", "residuals of log10(power"]
    return any(indicator in x_label for indicator in power_indicators)


def _add_percentage_axis(ax: plt.Axes, x_label: str, plot_cfg) -> None:
    if not _should_show_percentage_axis(x_label):
        return
    
    behavioral_config = _get_behavioral_config(plot_cfg)
    percentage_axis_nbins = behavioral_config.get("percentage_axis_nbins", 7)
    percentage_axis_minor_divisor = behavioral_config.get("percentage_axis_minor_divisor", 2)
    
    try:
        ax_pct = ax.secondary_xaxis("top", functions=(_logratio_to_pct, _pct_to_logratio))
        ax_pct.set_xlabel("Power Change (%)", fontsize=plot_cfg.font.large)
        ax_pct.xaxis.set_major_locator(MaxNLocator(nbins=percentage_axis_nbins))
        ax_pct.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax_pct.xaxis.set_minor_locator(AutoMinorLocator(percentage_axis_minor_divisor))
    except (AttributeError, TypeError, ValueError):
        pass




def _add_channel_annotation(fig: plt.Figure, roi_channels: List[str], plot_cfg) -> None:
    if not roi_channels:
        return
    
    behavioral_config = _get_behavioral_config(plot_cfg)
    max_channels_to_display = behavioral_config.get("max_channels_to_display", 10)
    channel_text = format_channel_list_for_display(roi_channels, max_channels=max_channels_to_display)
    fig.text(
        plot_cfg.text_position.channel_annotation_x,
        plot_cfg.text_position.channel_annotation_y,
        channel_text,
        fontsize=plot_cfg.font.label,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=plot_cfg.style.alpha_text_box),
    )


def _get_line_color(p_value: float, plot_cfg) -> str:
    behavioral_config = _get_behavioral_config(plot_cfg)
    significance_threshold = behavioral_config.get("significance_threshold", 0.05)
    is_significant = np.isfinite(p_value) and p_value < significance_threshold
    return plot_cfg.get_color("significant", plot_type="behavioral") if is_significant else plot_cfg.get_color("nonsignificant", plot_type="behavioral")


def _extract_band_info(title_prefix: str) -> str:
    if "power" not in title_prefix.lower():
        return ""
    
    frequency_bands = ["delta", "theta", "alpha", "beta", "gamma"]
    title_lower = title_prefix.lower()
    for band in frequency_bands:
        if band in title_lower:
            return f" ({band.upper()})"
    return ""


def _extract_target_info(title_prefix: str) -> str:
    title_lower = title_prefix.lower()
    if "rating" in title_lower:
        return " vs rating"
    if "temp" in title_lower or "temperature" in title_lower:
        return " vs temperature"
    return ""




def _log_plot_details(
    logger: logging.Logger,
    title_prefix: str,
    roi_channels: Optional[List[str]],
) -> None:
    roi_description = format_roi_description(roi_channels)
    band_info = _extract_band_info(title_prefix)
    target_info = _extract_target_info(title_prefix)
    plot_description = f"Scatter plot{band_info}{target_info}: {roi_description}"
    logger.info(plot_description)


def generate_correlation_scatter(
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str,
    y_label: str,
    title_prefix: str,
    band_color: str,
    output_path: Path,
    *,
    method_code: str = "spearman",
    Z_covars: Optional[pd.DataFrame] = None,
    covar_names: Optional[List[str]] = None,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    is_partial_residuals: bool = False,
    roi_channels: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    annotated_stats: Optional[Tuple[float, float, int]] = None,
    annot_ci: Optional[Tuple[float, float]] = None,
    stats_tag: Optional[str] = None,
    config = None,
) -> None:
    logger = logger or _get_default_logger()
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)

    ensure_aligned_lengths(x_data, y_data, context="Correlation scatter inputs", strict=True)

    prepare_func = prepare_data_without_validation if is_partial_residuals else prepare_data_for_plotting
    x_clean, y_clean, n_eff = prepare_func(x_data, y_data)

    min_samples = plot_cfg.validation.get("min_samples_for_plot", 5)
    if n_eff < min_samples:
        logger.warning(f"Insufficient data for scatter plot (n={n_eff} < {min_samples}), skipping {output_path}")
        return

    if annotated_stats is None:
        if Z_covars is not None and not Z_covars.empty:
            r_disp, p_disp, n_disp = _partial_corr_xy_given_Z(
                x_clean, y_clean, Z_covars, method_code, config=config
            )
        else:
            mask = joint_valid_mask(x_clean, y_clean)
            n_valid = int(mask.sum())
            if n_valid >= min_samples:
                x_vals = x_clean.iloc[mask] if isinstance(x_clean, pd.Series) else x_clean[mask]
                y_vals = y_clean.iloc[mask] if isinstance(y_clean, pd.Series) else y_clean[mask]
                use_spearman = method_code.lower() == "spearman"
                if use_spearman:
                    r_disp, p_disp = stats.spearmanr(x_vals, y_vals, nan_policy="omit")
                else:
                    r_disp, p_disp = stats.pearsonr(x_vals, y_vals)
                n_disp = n_valid
            else:
                r_disp, p_disp, n_disp = np.nan, np.nan, n_valid
    else:
        r_disp, p_disp, n_disp = annotated_stats
    
    if annot_ci is None:
        if bootstrap_ci > 0:
            ci_disp = _bootstrap_corr_ci(
                x_clean, y_clean, method_code, n_boot=bootstrap_ci, rng=rng, config=config
            )
        else:
            ci_disp = (np.nan, np.nan)
    else:
        ci_disp = annot_ci

    fig, ax_main, ax_histx, ax_histy = _create_scatter_figure(plot_cfg)

    line_color = _get_line_color(p_disp, plot_cfg)
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    line_width = plot_cfg.get_line_width("bold", plot_type="behavioral")
    
    behavioral_config = _get_behavioral_config(plot_cfg)
    sns.regplot(
        x=x_clean,
        y=y_clean,
        ax=ax_main,
        ci=behavioral_config.get("correlation_ci_level", 95),
        scatter_kws={
            "s": marker_size,
            "alpha": plot_cfg.style.scatter.alpha,
            "color": band_color,
            "edgecolor": plot_cfg.style.scatter.edgecolor,
            "linewidths": plot_cfg.style.scatter.edgewidth,
        },
        line_kws={"color": line_color, "lw": line_width},
    )

    _plot_histogram_with_kde(ax_histx, x_clean, band_color, plot_cfg, orientation="vertical")
    _plot_histogram_with_kde(ax_histy, y_clean, band_color, plot_cfg, orientation="horizontal")

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    _add_percentage_axis(ax_histx, x_label, plot_cfg)

    stats_text = format_correlation_stats_text(r_disp, p_disp, n_disp, ci_disp, stats_tag)
    fig.text(
        plot_cfg.text_position.p_value_x,
        plot_cfg.text_position.p_value_y,
        stats_text,
        fontsize=plot_cfg.font.title,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=plot_cfg.style.alpha_text_box),
    )

    if roi_channels:
        _add_channel_annotation(fig, roi_channels, plot_cfg)

    fig.suptitle(title_prefix, fontsize=plot_cfg.font.figure_title, fontweight="bold", 
                 y=plot_cfg.text_position.title_y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()

    save_formats = plot_cfg.formats if plot_cfg.formats else config.get("output.save_formats", ["svg"])
    _log_plot_details(logger, title_prefix, roi_channels)
    save_fig(fig, output_path, formats=save_formats, dpi=plot_cfg.dpi, 
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches,
             footer=_get_behavior_footer(config), logger=logger)
    plt.close(fig)




def _plot_residuals_vs_fitted(
    ax: plt.Axes,
    fitted: np.ndarray,
    residuals: np.ndarray,
    band_color: str,
    plot_cfg,
) -> None:
    sort_order = np.argsort(fitted)
    fitted_sorted = fitted[sort_order]
    residuals_sorted = residuals[sort_order]
    
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    ax.scatter(
        fitted_sorted,
        residuals_sorted,
        s=marker_size,
        alpha=plot_cfg.style.scatter.alpha,
        color=band_color,
        edgecolor=plot_cfg.style.scatter.edgecolor,
        linewidth=plot_cfg.style.scatter.edgewidth,
    )
    ax.axhline(0, color=plot_cfg.style.colors.black, linestyle="--", 
               linewidth=plot_cfg.style.line.regression_width)
    ax.set_xlabel("Fitted values", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Residuals", fontsize=plot_cfg.font.ylabel)
    ax.set_title("Residuals vs Fitted", fontsize=plot_cfg.font.figure_title, fontweight="bold")
    ax.grid(True, alpha=plot_cfg.style.alpha_grid)


def _plot_qq_plot(ax: plt.Axes, residuals: np.ndarray, band_color: str, plot_cfg) -> None:
    qq_data = stats.probplot(residuals, dist="norm")
    theoretical_quantiles = qq_data[0][0]
    sample_quantiles = qq_data[0][1]
    fit_params = qq_data[1]
    
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    ax.scatter(
        theoretical_quantiles,
        sample_quantiles,
        color=band_color,
        alpha=plot_cfg.style.scatter.alpha,
        s=marker_size,
        edgecolor=plot_cfg.style.scatter.edgecolor,
        linewidth=plot_cfg.style.scatter.edgewidth,
    )
    fitted_line = fit_params[0] + fit_params[1] * theoretical_quantiles
    ax.plot(
        theoretical_quantiles,
        fitted_line,
        color=plot_cfg.style.colors.black,
        linestyle="--",
        linewidth=plot_cfg.style.line.qq_width,
    )
    ax.set_title("Normal Q-Q Plot", fontsize=plot_cfg.font.figure_title, fontweight="bold")
    ax.set_xlabel("Theoretical quantiles", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Sample quantiles", fontsize=plot_cfg.font.ylabel)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid)


def plot_residual_qc(
    x_data: pd.Series,
    y_data: pd.Series,
    *,
    title_prefix: str,
    output_path: Path,
    band_color: str,
    logger: Optional[logging.Logger] = None,
    config = None,
) -> None:
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)

    x_series = pd.to_numeric(x_data, errors="coerce")
    y_series = pd.to_numeric(y_data, errors="coerce")
    mask = x_series.notna() & y_series.notna()
    n_obs = int(mask.sum())
    min_samples = plot_cfg.validation.get("min_samples_for_plot", 5)
    if n_obs < min_samples:
        logger.warning(f"Residual QC skipped: insufficient paired samples (<{min_samples}).")
        return

    fitted, residuals, _ = compute_linear_residuals(x_data, y_data)

    behavioral_config = _get_behavioral_config(plot_cfg)
    residual_qc_figsize = tuple(behavioral_config.get("residual_qc_figsize", [12, 5]))
    fig, axes = plt.subplots(1, 2, figsize=residual_qc_figsize)
    _plot_residuals_vs_fitted(axes[0], fitted, residuals, band_color, plot_cfg)
    _plot_qq_plot(axes[1], residuals, band_color, plot_cfg)

    fig.suptitle(f"{title_prefix} — Residual QC", fontsize=plot_cfg.font.figure_title, 
                 fontweight="bold", y=plot_cfg.text_position.residual_qc_title_y)
    fig.tight_layout()

    save_formats = plot_cfg.formats if plot_cfg.formats else config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        output_path,
        formats=save_formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)


def _plot_residual_histogram(ax: plt.Axes, residuals: np.ndarray, band_color: str, plot_cfg) -> None:
    bins = plot_cfg.get_histogram_bins(plot_type="residual")
    ax.hist(
        residuals,
        bins=bins,
        color=band_color,
        alpha=plot_cfg.style.histogram.alpha_residual,
        edgecolor=plot_cfg.style.histogram.edgecolor,
        linewidth=plot_cfg.style.histogram.edgewidth,
    )
    ax.set_title("Residual distribution", fontsize=plot_cfg.font.title)
    ax.set_xlabel("Residual", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Count", fontsize=plot_cfg.font.label)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid)


def _plot_residuals_vs_predictor(ax: plt.Axes, x_clean: np.ndarray, residuals: np.ndarray, band_color: str, plot_cfg) -> None:
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    ax.scatter(
        x_clean,
        residuals,
        s=marker_size,
        alpha=plot_cfg.style.scatter.alpha,
        color=band_color,
        edgecolor=plot_cfg.style.scatter.edgecolor,
        linewidth=plot_cfg.style.scatter.edgewidth,
    )
    ax.axhline(0, color=plot_cfg.style.colors.black, linestyle="--", 
               linewidth=plot_cfg.style.line.residual_width)
    ax.set_xlabel("Predictor", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Residuals", fontsize=plot_cfg.font.label)
    ax.set_title("Residuals vs Predictor", fontsize=plot_cfg.font.title)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid)


def plot_regression_residual_diagnostics(
    x_data: pd.Series,
    y_data: pd.Series,
    *,
    title_prefix: str,
    output_path: Path,
    band_color: str,
    logger: Optional[logging.Logger] = None,
    config = None,
) -> None:
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)

    x_series = pd.to_numeric(x_data, errors="coerce")
    y_series = pd.to_numeric(y_data, errors="coerce")
    mask = x_series.notna() & y_series.notna()
    n_obs = int(mask.sum())
    min_samples = plot_cfg.validation.get("min_samples_for_plot", 5)
    if n_obs < min_samples:
        logger.warning(f"Residual diagnostics skipped: insufficient paired samples (<{min_samples}).")
        return

    fitted, residuals, x_clean = compute_linear_residuals(x_data, y_data)

    behavioral_config = _get_behavioral_config(plot_cfg)
    diagnostics_figsize = tuple(behavioral_config.get("diagnostics_figsize", [10, 8]))
    fig, axes = plt.subplots(2, 2, figsize=diagnostics_figsize)
    _plot_residuals_vs_fitted(axes[0, 0], fitted, residuals, band_color, plot_cfg)
    _plot_qq_plot(axes[0, 1], residuals, band_color, plot_cfg)
    _plot_residual_histogram(axes[1, 0], residuals, band_color, plot_cfg)
    _plot_residuals_vs_predictor(axes[1, 1], x_clean, residuals, band_color, plot_cfg)

    fig.suptitle(f"{title_prefix} — Residual Diagnostics", fontsize=plot_cfg.font.figure_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    save_formats = plot_cfg.formats if plot_cfg.formats else config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        output_path,
        formats=save_formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)
    logger.info(f"Residual diagnostics saved to {output_path}")


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
    behavioral_config = _get_behavioral_config(plot_cfg)
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


def _get_temporal_columns(temporal_df: pd.DataFrame, band: str, time_label: str, config: Optional[Any] = None) -> List[str]:
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
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
    behavioral_config = _get_behavioral_config(plot_cfg)
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
    behavioral_config = _get_behavioral_config(plot_cfg)
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
# Subject Plots
###################################################################

def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
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


def _plot_distribution_histogram(
    data: pd.Series,
    x_label: str,
    title: str,
    output_path: Path,
    plot_cfg,
    config,
    logger: logging.Logger,
) -> None:
    if data.empty or data.notna().sum() < 3:
        logger.warning(f"Insufficient data for histogram: {title}")
        return
    
    fig_size = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    fig, ax = plt.subplots(figsize=fig_size)
    
    band_color = get_band_color("alpha", config)
    behavioral_config = _get_behavioral_config(plot_cfg)
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
    
    save_formats = plot_cfg.formats if plot_cfg.formats else config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        output_path,
        formats=save_formats,
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
    config=None,
) -> None:
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ROI power scatter plotting for sub-{subject}")
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {})
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
# Group Plot Helpers
###################################################################


def _add_subject_dummies_if_needed(
    Z_all_vis: pd.DataFrame, partial_subj_ids: List[str]
) -> pd.DataFrame:
    if "__subject_id__" not in Z_all_vis.columns:
        dummies = pd.get_dummies(
            pd.Series(partial_subj_ids, name="__subject_id__").astype(str),
            prefix="sub",
            drop_first=True,
        )
        Z_all_vis = pd.concat([Z_all_vis.reset_index(drop=True), dummies], axis=1)
    
    return Z_all_vis


def _prepare_partial_residuals_data(
    x_lists: List,
    y_lists: List,
    Z_lists: List,
    has_Z_flags: List[bool],
    subj_order: List[str],
    pooling_strategy: str,
    subject_fixed_effects: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    if not any(has_Z_flags):
        return None, None, None
    
    partial_x: List[pd.Series] = []
    partial_y: List[pd.Series] = []
    partial_Z: List[pd.DataFrame] = []
    partial_subj_ids: List[str] = []
    
    for idx, (has_cov, Z_df, x_arr, y_arr) in enumerate(zip(has_Z_flags, Z_lists, x_lists, y_lists)):
        if not has_cov or Z_df is None:
            continue
        
        xi, yi, Zi = prepare_partial_correlation_data(x_arr, y_arr, Z_df, pooling_strategy)
        if xi is None or yi is None:
            continue
        
        partial_x.append(xi)
        partial_y.append(yi)
        partial_Z.append(Zi)
        subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
        partial_subj_ids.extend([subj_id] * len(xi))

    if not partial_Z:
        return None, None, None

    common_cols = extract_common_dataframe_columns(partial_Z)
    if common_cols:
        partial_Z = [df[common_cols] for df in partial_Z]
    
    Z_all_vis = pd.concat(partial_Z, ignore_index=True)
    x_all_partial = pd.concat(partial_x, ignore_index=True)
    y_all_partial = pd.concat(partial_y, ignore_index=True)
    
    if subject_fixed_effects:
        Z_all_vis = _add_subject_dummies_if_needed(Z_all_vis, partial_subj_ids)
    
    return Z_all_vis, x_all_partial, y_all_partial


def _get_y_label(target_type: str, pooling_strategy: str, config: Optional[Any] = None) -> str:
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    target_rating = behavioral_config.get("target_rating", "rating")
    target_temperature = behavioral_config.get("target_temperature", "temperature")
    
    label_map = {
        target_rating: {
            "pooled_trials": "Rating",
            "within_subject_centered": "Rating (centered)",
            "within_subject_zscored": "Rating (z-scored)",
        },
        target_temperature: {
            "pooled_trials": "Temperature (°C)",
            "within_subject_centered": "Temperature (°C, centered)",
            "within_subject_zscored": "Temperature (z-scored)",
        },
    }
    
    base_labels = {
        target_rating: "Rating",
        target_temperature: "Temperature",
    }
    
    return label_map.get(target_type, {}).get(pooling_strategy, base_labels.get(target_type, target_type))


def _format_band_title(band: str, freq_bands_cfg: Dict) -> str:
    freq_range = freq_bands_cfg.get(band)
    if not freq_range:
        return band.capitalize()
    
    freq_range = tuple(freq_range)
    return f"{band.capitalize()} ({freq_range[0]:g}–{freq_range[1]:g} Hz)"


def _get_output_directory_and_path(plots_dir: Path, roi: str, band: str, target_type: str, config: Optional[Any] = None) -> Tuple[Path, Path]:
    is_overall = roi == "All"
    title_roi = "Overall" if is_overall else roi
    
    if is_overall:
        out_dir = plots_dir / "overall"
        base_name = "scatter_pow_overall"
    else:
        out_dir = plots_dir / "roi_scatters" / sanitize_label(roi)
        base_name = "scatter_pow"
    
    ensure_dir(out_dir)
    target_suffix = _get_target_suffix(target_type, config)
    out_path = out_dir / f"{base_name}_{sanitize_label(band)}_vs_{target_suffix}"
    return out_dir, out_path


def _plot_partial_residuals_if_available(
    Z_all_vis: Optional[pd.DataFrame],
    x_all_partial: Optional[pd.Series],
    y_all_partial: Optional[pd.Series],
    method_code: str,
    band_title: str,
    target_type: str,
    title_roi: str,
    out_dir: Path,
    band: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    config,
    logger: logging.Logger,
) -> None:
    if Z_all_vis is None or x_all_partial is None or y_all_partial is None:
        return
    
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    if len(x_all_partial) < min_samples_for_plot:
        return
    
    x_res, y_res, n_res = _partial_residuals_xy_given_Z(x_all_partial, y_all_partial, Z_all_vis, method_code)
    if n_res < min_samples_for_plot:
        return
    
    if method_code.lower() == "spearman":
        r_resid, p_resid = stats.spearmanr(x_res, y_res, nan_policy="omit")
    else:
        r_resid, p_resid = stats.pearsonr(x_res, y_res)
    ci_resid = _bootstrap_corr_ci(x_res, y_res, method_code, n_boot=bootstrap_ci, rng=rng) if bootstrap_ci > 0 else (np.nan, np.nan)
    
    residual_xlabel, residual_ylabel = get_residual_labels(method_code, target_type)
    is_overall = title_roi == "Overall"
    base_name = "scatter_pow_overall" if is_overall else "scatter_pow"
    target_suffix = _get_target_suffix(target_type, config)
    
    generate_correlation_scatter(
        x_data=x_res,
        y_data=y_res,
        x_label=residual_xlabel,
        y_label=residual_ylabel,
        title_prefix=f"Partial residuals — {band_title} vs {target_type} — {title_roi}",
        band_color=get_band_color(band, config),
        output_path=out_dir / f"{base_name}_{sanitize_label(band)}_vs_{target_suffix}_partial",
        method_code=method_code,
        bootstrap_ci=0,
        rng=rng,
        is_partial_residuals=True,
        roi_channels=None,
        logger=logger,
        annotated_stats=(r_resid, p_resid, n_res),
        annot_ci=ci_resid,
        config=config,
    )


def _save_group_stats(
    records: List[Dict[str, Any]],
    target_type: str,
    stats_dir: Path,
    config,
    logger: logging.Logger,
) -> None:
    if not records:
        return
    
    df = pd.DataFrame(records)
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    rej, crit = fdr_bh_reject(df["p_group"].to_numpy(), alpha=fdr_alpha)
    df["fdr_reject"] = rej
    df["fdr_crit_p"] = crit
    
    target_suffix = _get_target_suffix(target_type, config)
    out_stats = stats_dir / f"group_pooled_corr_pow_roi_vs_{target_suffix}.tsv"
    df.to_csv(out_stats, sep="\t", index=False)
    logger.info("Wrote pooled ROI vs %s stats: %s", target_type, out_stats)




def _compute_group_statistics(
    x_list, y_list, method_code, pooling_strategy, cluster_bootstrap, rng
):
    return _compute_group_corr_stats(
        [np.asarray(v) for v in x_list],
        [np.asarray(v) for v in y_list],
        method_code,
        strategy=pooling_strategy,
        n_cluster_boot=cluster_bootstrap,
        rng=rng,
    )


def _create_group_plots(
    x_all, y_all, band, band_title, roi, target_type, target_suffix,
    title_roi, out_dir, out_path, base_name, y_label, tag,
    r_g, p_g, n_trials, ci95, method_code, rng, logger, config
):
    generate_correlation_scatter(
        x_data=x_all,
        y_data=y_all,
        x_label="log10(power/baseline [-5–0 s])",
        y_label=y_label,
        title_prefix=f"{band_title} power vs {target_type} — {title_roi}",
        band_color=get_band_color(band, config),
        output_path=out_path,
        method_code=method_code,
        Z_covars=None,
        covar_names=None,
        bootstrap_ci=0,
        rng=rng,
        logger=logger,
        annotated_stats=(r_g, p_g, n_trials),
        annot_ci=ci95,
        stats_tag=tag,
        config=config,
    )
    
    plot_residual_qc(
        x_data=pd.Series(x_all),
        y_data=pd.Series(y_all),
        title_prefix=f"{band_title} power vs {target_type} — {title_roi}",
        output_path=out_dir / f"residual_qc_{base_name}_{sanitize_label(band)}_vs_{target_suffix}",
        band_color=get_band_color(band, config),
        logger=logger,
        config=config,
    )


def _build_statistics_record(
    roi, band, r_g, p_g, p_pooled, n_trials, n_subj, pooling_strategy, ci95
):
    return {
        "roi": roi,
        "band": band,
        "r_pooled": r_g,
        "p_group": p_g,
        "p_trials": p_pooled,
        "n_total": n_trials,
        "n_subjects": n_subj,
        "pooling_strategy": pooling_strategy,
        "ci_low": ci95[0],
        "ci_high": ci95[1],
    }


def _process_group_target(
    x_lists: Dict,
    y_lists: Dict,
    Z_lists: Dict,
    has_Z_flags: Dict,
    subj_order: Dict,
    target_type: str,
    plots_dir: Path,
    stats_dir: Path,
    config,
    pooling_strategy: str,
    cluster_bootstrap: int,
    subject_fixed_effects: bool,
    bootstrap_ci: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    tag_map: Dict[str, str],
    freq_bands_cfg: Dict,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    method_code = behavioral_config.get("method_spearman", "spearman")
    
    for (band, roi), x_list in x_lists.items():
        y_list = y_lists.get((band, roi))
        if not y_list:
            continue

        Z_list = Z_lists.get((band, roi), [])
        has_Z_flag = has_Z_flags.get((band, roi), [])
        subj_ord = subj_order.get((band, roi), [])

        result = prepare_group_band_roi_data(
            x_list, y_list, Z_list, has_Z_flag, subj_ord,
            pooling_strategy, subject_fixed_effects
        )
        if result[0] is None:
            continue
        
        x_all, y_all, Z_all_vis, x_all_partial, y_all_partial, vis_subj_ids = result

        r_g, p_g, n_trials, n_subj, ci95, p_pooled = _compute_group_statistics(
            x_list, y_list, method_code, pooling_strategy, cluster_bootstrap, rng
        )

        tag = tag_map.get(pooling_strategy)
        band_title = _format_band_title(band, freq_bands_cfg)
        title_roi = "Overall" if roi == "All" else roi
        out_dir, out_path = _get_output_directory_and_path(plots_dir, roi, band, target_type, config)
        base_name = "scatter_pow_overall" if roi == "All" else "scatter_pow"
        target_suffix = _get_target_suffix(target_type, config)
        y_label = _get_y_label(target_type, pooling_strategy, config)

        _create_group_plots(
            x_all, y_all, band, band_title, roi, target_type, target_suffix,
            title_roi, out_dir, out_path, base_name, y_label, tag,
            r_g, p_g, n_trials, ci95, method_code, rng, logger, config
        )

        records.append(_build_statistics_record(
            roi, band, r_g, p_g, p_pooled, n_trials, n_subj, pooling_strategy, ci95
        ))

        _plot_partial_residuals_if_available(
            Z_all_vis, x_all_partial, y_all_partial, method_code, band_title,
            target_type, title_roi, out_dir, band, bootstrap_ci, rng, config, logger
        )

    return records


###################################################################
# Group Plots
###################################################################

def _get_attr_safe(obj, name: str, default: Any) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _extract_scatter_inputs(scatter_inputs, do_temp: bool) -> Dict[str, Any]:
    return {
        "rating_x": _get_attr_safe(scatter_inputs, "rating_x", {}),
        "rating_y": _get_attr_safe(scatter_inputs, "rating_y", {}),
        "rating_Z": _get_attr_safe(scatter_inputs, "rating_Z", {}),
        "rating_hasZ": _get_attr_safe(scatter_inputs, "rating_hasZ", {}),
        "rating_subjects": _get_attr_safe(scatter_inputs, "rating_subjects", {}),
        "temp_x": _get_attr_safe(scatter_inputs, "temp_x", {}),
        "temp_y": _get_attr_safe(scatter_inputs, "temp_y", {}),
        "temp_Z": _get_attr_safe(scatter_inputs, "temp_Z", {}),
        "temp_hasZ": _get_attr_safe(scatter_inputs, "temp_hasZ", {}),
        "temp_subjects": _get_attr_safe(scatter_inputs, "temp_subjects", {}),
        "have_temp": bool(_get_attr_safe(scatter_inputs, "have_temp", False)) and do_temp,
    }


def plot_group_power_roi_scatter(
    scatter_inputs,
    *,
    config=None,
    pooling_strategy: str = "within_subject_centered",
    cluster_bootstrap: int = 0,
    subject_fixed_effects: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    deriv_root = Path(config.deriv_root)
    plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations")
    plots_dir = deriv_group_plots_path(deriv_root, subdir=plot_subdir)
    stats_dir = deriv_group_stats_path(deriv_root)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_group_logger("behavior_analysis", log_name, config=config)

    inputs = _extract_scatter_inputs(scatter_inputs, do_temp)
    rating_x = inputs["rating_x"]
    rating_y = inputs["rating_y"]
    rating_Z = inputs["rating_Z"]
    rating_hasZ = inputs["rating_hasZ"]
    rating_subjects = inputs["rating_subjects"]
    temp_x = inputs["temp_x"]
    temp_y = inputs["temp_y"]
    temp_Z = inputs["temp_Z"]
    temp_hasZ = inputs["temp_hasZ"]
    temp_subjects = inputs["temp_subjects"]
    have_temp = inputs["have_temp"]

    subject_set = set()
    for subj_lists in rating_subjects.values():
        subject_set.update(subj_lists)
    n_subjects = len(subject_set)

    logger.info(
        "Starting group pooled ROI scatters for %d subjects (strategy=%s, FE=%s)",
        n_subjects,
        pooling_strategy,
        "on" if subject_fixed_effects else "off",
    )

    allowed_pool = {"pooled_trials", "within_subject_centered", "within_subject_zscored", "fisher_by_subject"}
    if pooling_strategy not in allowed_pool:
        logger.warning(
            "Unknown pooling_strategy '%s', falling back to 'pooled_trials'",
            pooling_strategy,
        )
        pooling_strategy = "pooled_trials"

    tag_map = {
        "pooled_trials": "[Pooled]",
        "within_subject_centered": "[Centered]",
        "within_subject_zscored": "[Z-scored]",
        "fisher_by_subject": "[Fisher]",
    }

    freq_bands_cfg = config.get("time_frequency_analysis.bands", {})

    target_rating_group = behavioral_config.get("target_rating", "rating")
    rating_records = _process_group_target(
        rating_x, rating_y, rating_Z, rating_hasZ, rating_subjects,
        target_rating_group, plots_dir, stats_dir, config, pooling_strategy,
        cluster_bootstrap, subject_fixed_effects, bootstrap_ci, rng,
        logger, tag_map, freq_bands_cfg,
    )

    target_rating = behavioral_config.get("target_rating", "rating")
    _save_group_stats(rating_records, target_rating, stats_dir, config, logger)

    if not have_temp:
        return

    target_temperature_group = behavioral_config.get("target_temperature", "temperature")
    temp_records = _process_group_target(
        temp_x, temp_y, temp_Z, temp_hasZ, temp_subjects,
        target_temperature_group, plots_dir, stats_dir, config, pooling_strategy,
        cluster_bootstrap, subject_fixed_effects, bootstrap_ci, rng,
        logger, tag_map, freq_bands_cfg,
    )

    target_temperature_save = behavioral_config.get("target_temperature", "temperature")
    _save_group_stats(temp_records, target_temperature_save, stats_dir, config, logger)







def _add_colorbar(fig, axes, successful_plots, config=None):
    if not successful_plots:
        return
    
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    colorbar_config = behavioral_config.get("colorbar", {})
    
    width_fraction = colorbar_config.get("width_fraction", 0.55)
    left_offset_fraction = colorbar_config.get("left_offset_fraction", 0.225)
    bottom_offset = colorbar_config.get("bottom_offset", 0.12)
    min_bottom = colorbar_config.get("min_bottom", 0.04)
    height = colorbar_config.get("height", 0.028)
    label_fontsize = colorbar_config.get("label_fontsize", 11)
    tick_fontsize = colorbar_config.get("tick_fontsize", 9)
    tick_pad = colorbar_config.get("tick_pad", 2)
    
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    span = right - left
    cb_width = width_fraction * span
    cb_left = left + left_offset_fraction * span
    cb_bottom = max(min_bottom, bottom - bottom_offset)
    cax = fig.add_axes([cb_left, cb_bottom, cb_width, height])
    cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation='horizontal')
    cbar.set_label('Spearman correlation (ρ)', fontweight='bold', fontsize=label_fontsize)
    cbar.ax.tick_params(pad=tick_pad, labelsize=tick_fontsize)


def plot_significant_correlations_topomap(
    pow_df: pd.DataFrame,
    y: pd.Series,
    bands: List[str],
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config=None,
    alpha: float = 0.05,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    
    bands_with_data = []
    for band in bands:
        ch_names, correlations, p_values = compute_band_correlations(
            pow_df, y, band, power_prefix=power_prefix, min_samples=min_samples_for_plot
        )
        if len(ch_names) == 0:
            continue
        
        sig_mask = p_values < alpha
        bands_with_data.append({
            'band': band,
            'channels': ch_names,
            'correlations': correlations,
            'p_values': p_values,
            'significant_mask': sig_mask
        })
    
    if not bands_with_data:
        logger.warning("No significant correlations found across any frequency band")
        return
    
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    topomap_config = behavioral_config.get("topomap", {})
    
    n_bands = len(bands_with_data)
    figsize_per_band_width = topomap_config.get("figsize_per_band_width", 4.8)
    figsize_per_band_height = topomap_config.get("figsize_per_band_height", 4.8)
    fig, axes = plt.subplots(1, n_bands, figsize=(figsize_per_band_width * n_bands, figsize_per_band_height))
    if n_bands == 1:
        axes = [axes]
    
    subplots_left = topomap_config.get("subplots_left", 0.06)
    subplots_right = topomap_config.get("subplots_right", 0.98)
    subplots_top = topomap_config.get("subplots_top", 0.83)
    subplots_bottom = topomap_config.get("subplots_bottom", 0.25)
    subplots_wspace = topomap_config.get("subplots_wspace", 0.08)
    plt.subplots_adjust(left=subplots_left, right=subplots_right, top=subplots_top, bottom=subplots_bottom, wspace=subplots_wspace)
    
    vmax = compute_correlation_vmax(bands_with_data)
    successful_plots = []
    
    for i, band_data in enumerate(bands_with_data):
        ax = axes[i]
        topo_data, topo_mask = prepare_topomap_correlation_data(band_data, info)
        
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
        if len(picks) == 0:
            continue
        
        plot_cfg = get_plot_config(config)
        topomap_plot_config = plot_cfg.plot_type_configs.get("topomap", {})
        colormap = topomap_plot_config.get("colormap", "RdBu_r")
        contours = topomap_plot_config.get("contours", 6)
        
        im, _ = mne.viz.plot_topomap(
            topo_data[picks],
            mne.pick_info(info, picks),
            axes=ax,
            show=False,
            cmap=colormap,
            vlim=(-vmax, vmax),
            contours=contours,
            mask=topo_mask[picks],
            mask_params=dict(
                marker=topomap_config.get("mask_marker", "o"),
                markerfacecolor=topomap_config.get("mask_markerfacecolor", "white"),
                markeredgecolor=topomap_config.get("mask_markeredgecolor", "black"),
                linewidth=topomap_config.get("mask_linewidth", 1),
                markersize=topomap_config.get("mask_markersize", 6)
            )
        )
        
        successful_plots.append(im)
        
        n_sig = topo_mask[picks].sum()
        n_total = len([ch for ch in band_data['channels'] if ch in info['ch_names']])
        title_fontsize = topomap_config.get("title_fontsize", 12)
        title_pad = topomap_config.get("title_pad", 10)
        ax.set_title(
            f'{band_data["band"].upper()}\n{n_sig}/{n_total} significant',
            fontweight='bold', fontsize=title_fontsize, pad=title_pad
        )
    
    suptitle_fontsize = topomap_config.get("suptitle_fontsize", 14)
    suptitle_y = topomap_config.get("suptitle_y", 1.02)
    plt.suptitle(
        f'Significant EEG-Pain Correlations (p < {alpha})\nSubject {subject}',
        fontweight='bold', fontsize=suptitle_fontsize, y=suptitle_y
    )
    
    _add_colorbar(fig, axes, successful_plots, config)
    
    save_formats = config.get("output.save_formats", ["svg"])
    tight_layout_rect = topomap_config.get("tight_layout_rect", [0, 0.15, 1, 1])
    save_fig(
        fig,
        save_dir / f'sub-{subject}_significant_correlations_topomap',
        formats=save_formats,
        bbox_inches="tight",
        footer=_get_behavior_footer(config),
        tight_layout_rect=tight_layout_rect
    )
    plt.close(fig)
    
    logger.info(f"Created topomaps for {len(bands_with_data)} frequency bands: {[bd['band'] for bd in bands_with_data]}")


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
    fig, ax = _create_rating_distribution_plot(y, config)
    
    plt.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        save_dir / f'sub-{subject}_rating_distribution',
        formats=save_formats,
        bbox_inches="tight",
        footer=_get_behavior_footer(config)
    )
    plt.close(fig)


def _load_connectivity_data(conn_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not conn_path.exists():
        logger.warning(f"No connectivity data found at {conn_path}")
        return None
    
    if conn_path.suffix == '.parquet':
        return pd.read_parquet(conn_path)
    elif conn_path.suffix == '.tsv':
        return pd.read_csv(conn_path, sep='\t')
    else:
        logger.warning(f"Unsupported connectivity file format: {conn_path.suffix}")
        return None


def _find_available_connectivity_measure(conn_df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    conn_measures = ['coh', 'plv', 'pli', 'wpli', 'aec']
    available_measures = [m for m in conn_measures if any(m in col for col in conn_df.columns)]
    
    if not available_measures:
        logger.warning("No connectivity measures found")
        return None
    
    return 'coh' if 'coh' in available_measures else available_measures[0]




def _build_connectivity_graph(connections: List[str], correlations: List[float]) -> Optional[Any]:
    import networkx as nx
    
    G = nx.Graph()
    for conn, corr in zip(connections, correlations):
        if '__' in conn:
            ch1, ch2 = conn.split('__', 1)
            G.add_edge(ch1, ch2, weight=abs(corr), correlation=corr)
    
    if G.number_of_nodes() == 0:
        return None
    
    return G


def _plot_connectivity_network(
    G: Any,
    measure: str,
    band: str,
    subject: str,
    save_dir: Path,
    config,
) -> None:
    import networkx as nx
    
    base_seed = config.get("random.seed", 42)
    layout_key = f"{subject}_{measure}_{band}"
    layout_bytes = layout_key.encode('utf-8')
    layout_hash = int(hashlib.md5(layout_bytes).hexdigest()[:8], 16) & 0x7FFFFFFF
    layout_seed = (base_seed + layout_hash) % (2**31)
    
    plot_cfg = get_plot_config(config)
    fig_size_standard = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    fig, ax = plt.subplots(figsize=fig_size_standard)
    behavioral_config = _get_behavioral_config(plot_cfg)
    conn_config = behavioral_config.get("connectivity_network", {})
    
    spring_k = conn_config.get("spring_layout_k", 3)
    spring_iterations = conn_config.get("spring_layout_iterations", 50)
    pos = nx.spring_layout(G, k=spring_k, iterations=spring_iterations, seed=layout_seed)
    
    node_size_multiplier = conn_config.get("node_size_multiplier", 100)
    node_sizes = [G.degree(node) * node_size_multiplier for node in G.nodes()]
    node_color = conn_config.get("node_color", "lightblue")
    node_alpha = conn_config.get("node_alpha", 0.7)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color, alpha=node_alpha, ax=ax)
    
    edges = G.edges()
    weights = [G[u][v]['correlation'] for u, v in edges]
    max_weight = max(abs(w) for w in weights) if weights else 1.0
    
    edge_width = conn_config.get("edge_width", 2)
    edge_colormap = conn_config.get("edge_colormap", "RdBu_r")
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=weights,
        edge_cmap=plt.cm.get_cmap(edge_colormap), edge_vmin=-max_weight, edge_vmax=max_weight, width=edge_width, ax=ax
    )
    
    label_fontsize = conn_config.get("label_fontsize", 8)
    label_fontweight = conn_config.get("label_fontweight", "bold")
    nx.draw_networkx_labels(G, pos, font_size=label_fontsize, font_weight=label_fontweight, ax=ax)
    
    colorbar_colormap = conn_config.get("colorbar_colormap", "RdBu_r")
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap(colorbar_colormap),
        norm=plt.Normalize(vmin=-max_weight, vmax=max_weight)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Correlation with Behavior', fontweight='bold')
    
    conn_title_fontsize = behavioral_config.get("connectivity_network_title_fontsize", 14)
    ax.set_title(
        f'Behavior-Modulated {measure.upper()} Connectivity\n{band.capitalize()} Band - Subject {subject}',
        fontweight='bold', fontsize=conn_title_fontsize
    )
    ax.axis('off')
    
    plt.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        save_dir / f'sub-{subject}_connectivity_network_{measure}_{band}',
        formats=save_formats,
        bbox_inches="tight",
        footer=_get_behavior_footer(config)
    )
    plt.close(fig)


def plot_behavior_modulated_connectivity(
    subject: str,
    task: str,
    y: pd.Series,
    save_dir: Path,
    logger: logging.Logger,
    config=None,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    
    deriv_root = Path(config.deriv_root)
    conn_path = find_connectivity_features_path(deriv_root, subject)
    
    conn_df = _load_connectivity_data(conn_path, logger)
    if conn_df is None:
        return
    
    measure = _find_available_connectivity_measure(conn_df, logger)
    if measure is None:
        return
    
    bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    
    for band in bands:
        measure_cols = [col for col in conn_df.columns if f'{measure}_{band}' in col]
        if not measure_cols:
            continue
        correlations, connections = compute_connectivity_correlations(
            conn_df, y, measure_cols, measure, band, min_samples=min_samples_for_plot
        )
        
        if len(connections) < 3:
            continue
        
        G = _build_connectivity_graph(connections, correlations)
        if G is None:
            continue
        
        _plot_connectivity_network(G, measure, band, subject, save_dir, config)
    
    logger.info(f"Saved behavior-modulated connectivity networks")


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
        ("temp", stats_dir / "corr_stats_pow_combined_vs_temp.tsv"),
        (target_temperature, stats_dir / "corr_stats_pow_combined_vs_temperature.tsv"),
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


def plot_top_behavioral_predictors(subject: str, task: Optional[str] = None, alpha: float = None, top_n: int = None) -> None:
    config = load_settings()
    if task is None:
        task = config.task
    
    alpha = alpha or config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    top_n = top_n or int(config.get("behavior_analysis.predictors.top_n", 20))
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Creating top {top_n} behavioral predictors plot for sub-{subject}")
    
    deriv_root = Path(config.deriv_root)
    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {}) if plot_cfg else {}
    plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations") if plot_cfg else "04_behavior_correlations"
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    stats_dir = deriv_stats_path(deriv_root, subject)
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
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(fig, output_path, formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
    plt.close(fig)
    
    logger.info(f"Saved top {top_n} behavioral predictors plot: {output_path}.png")
    if 'target' in df.columns:
        counts_by_tgt = df_sig['target'].value_counts().to_dict() if len(df_sig) else {}
        logger.info(f"Found {len(df_top)} significant predictors across targets {counts_by_tgt} (out of {len(df)} total correlations)")
    else:
        logger.info(f"Found {len(df_top)} significant predictors (out of {len(df)} total correlations)")
    
    _export_top_predictors(df_top, stats_dir, top_n, logger)


def _load_tf_correlation_data(data_path: Path, config) -> Dict[str, Any]:
    with np.load(data_path, allow_pickle=True) as data:
        return {
            "correlations": data["correlations"],
            "p_values": data["p_values"],
            "p_corrected": data.get("p_corrected", np.full_like(data["correlations"], np.nan)),
            "significant_mask": data.get("significant_mask", np.zeros_like(data["correlations"], dtype=bool)),
            "cluster_labels": data.get("cluster_labels"),
            "cluster_pvals": data.get("cluster_pvals"),
            "cluster_sig_mask": data.get("cluster_sig_mask"),
            "freqs": data["freqs"],
            "time_bin_centers": data["time_bin_centers"],
            "time_bin_edges": data.get("time_bin_edges"),
            "n_valid": data.get("n_valid", np.zeros_like(data["correlations"], dtype=int)),
            "freq_range": tuple(data.get("freq_range", (float(data["freqs"][0]), float(data["freqs"][-1])))),
            "baseline_applied": bool(data.get("baseline_applied", False)),
            "baseline_window": data.get("baseline_window", np.array([])),
            "time_resolution": float(data.get("time_resolution", 0.1)),
            "alpha": float(data.get("alpha", config.get("behavior_analysis.statistics.fdr_alpha", 0.05))),
            "cluster_alpha": float(data.get("cluster_alpha", data.get("alpha", config.get("behavior_analysis.statistics.fdr_alpha", 0.05)))),
            "cluster_n_perm": int(data.get("cluster_n_perm", 0)),
            "method": str(data.get("method", "spearman")),
        }


def _normalize_cluster_data(cluster_labels, cluster_pvals, cluster_sig_mask):
    if cluster_labels is not None and cluster_labels.size == 0:
        cluster_labels = None
    if cluster_pvals is not None and cluster_pvals.size == 0:
        cluster_pvals = None
    if cluster_sig_mask is not None and cluster_sig_mask.size == 0:
        cluster_sig_mask = None
    return cluster_labels, cluster_pvals, cluster_sig_mask


def _compute_time_bin_edges(time_bin_centers, time_bin_edges, time_resolution):
    if time_bin_edges is None or len(time_bin_edges) != len(time_bin_centers) + 1:
        half_step = time_resolution / 2.0
        time_bin_edges = np.concatenate(([time_bin_centers[0] - half_step], time_bin_centers + half_step))
    return time_bin_edges


def _extract_baseline_window(baseline_window_raw):
    if baseline_window_raw.size == 2:
        return (float(baseline_window_raw[0]), float(baseline_window_raw[1]))
    return None


def _create_tf_heatmap_plot(
    correlations,
    time_bin_edges,
    freqs,
    cluster_sig_mask,
    time_bin_centers,
    correlation_vmin,
    correlation_vmax,
    subject,
    method_name,
    roi_name,
    baseline_applied,
    baseline_window_used,
    config: Optional[Any] = None,
):
    extent = [
        float(time_bin_edges[0]),
        float(time_bin_edges[-1]),
        float(freqs[0]),
        float(freqs[-1]),
    ]

    plot_cfg = get_plot_config(config)
    fig_size_standard = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    fig, ax = plt.subplots(figsize=fig_size_standard)
    im = ax.imshow(
        correlations,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=correlation_vmin,
        vmax=correlation_vmax,
    )

    if cluster_sig_mask is not None and np.any(cluster_sig_mask):
        x_vals = time_bin_centers
        y_vals = freqs
        xx, yy = np.meshgrid(x_vals, y_vals)
        ax.contour(
            xx,
            yy,
            cluster_sig_mask.astype(float),
            levels=[0.5],
            colors="gold",
            linewidths=1.0,
            linestyles="--",
        )

    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontweight="bold")

    metric = "log10(power/baseline)" if baseline_applied else "raw power"
    baseline_text = ""
    if baseline_applied and baseline_window_used is not None:
        baseline_text = f" | BL: [{baseline_window_used[0]:.2f}, {baseline_window_used[1]:.2f}] s"

    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = _get_behavioral_config(plot_cfg) if plot_cfg else {}
    tf_heatmap_config = behavioral_config.get("time_frequency_heatmap", {})
    
    title_text = (
        "Time-Frequency Power-Behavior Correlations\n"
        f"Subject: {subject} | Method: {method_name} | ROI: {roi_name} | Metric: {metric}{baseline_text}"
    )
    title_fontsize = tf_heatmap_config.get("title_fontsize", 14)
    ax.set_title(title_text, fontsize=title_fontsize, fontweight="bold")
    
    zero_line_color = tf_heatmap_config.get("zero_line_color", "black")
    zero_line_linestyle = tf_heatmap_config.get("zero_line_linestyle", "--")
    zero_line_alpha = tf_heatmap_config.get("zero_line_alpha", 0.5)
    ax.axvline(0, color=zero_line_color, linestyle=zero_line_linestyle, alpha=zero_line_alpha)
    
    cbar = plt.colorbar(im, ax=ax, label="Correlation (r)")
    colorbar_label_fontsize = tf_heatmap_config.get("colorbar_label_fontsize", 12)
    cbar.ax.tick_params(labelsize=colorbar_label_fontsize)

    return fig, ax


def _add_frequency_band_markers(ax, freq_bands, freq_range, config=None):
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    tf_heatmap_config = behavioral_config.get("time_frequency_heatmap", {})
    
    marker_color = tf_heatmap_config.get("frequency_band_marker_color", "white")
    marker_linestyle = tf_heatmap_config.get("frequency_band_marker_linestyle", "-")
    marker_alpha = tf_heatmap_config.get("frequency_band_marker_alpha", 0.3)
    marker_linewidth = tf_heatmap_config.get("frequency_band_marker_linewidth", 0.5)
    
    for band, band_range in freq_bands.items():
        if not isinstance(band_range, (list, tuple)) or len(band_range) != 2:
            continue
        fmin, fmax = tuple(band_range)
        if (
            fmin is not None
            and fmax is not None
            and fmin >= freq_range[0]
            and fmax <= freq_range[1]
        ):
            ax.axhline(fmin, color=marker_color, linestyle=marker_linestyle, alpha=marker_alpha, linewidth=marker_linewidth)
            ax.axhline(fmax, color=marker_color, linestyle=marker_linestyle, alpha=marker_alpha, linewidth=marker_linewidth)


def _create_summary_dataframe(
    freqs,
    time_bin_centers,
    correlations,
    p_corrected,
    significant_mask,
    n_valid,
    cluster_labels,
    cluster_sig_mask,
    cluster_pvals,
):
    n_freqs = len(freqs)
    n_time_bins = len(time_bin_centers)
    n_points = n_freqs * n_time_bins

    cluster_flat = cluster_labels.flatten() if cluster_labels is not None else np.zeros(n_points, dtype=int)
    cluster_sig_flat = cluster_sig_mask.flatten() if cluster_sig_mask is not None else np.zeros(n_points, dtype=bool)
    cluster_p_flat = cluster_pvals.flatten() if cluster_pvals is not None else np.full(n_points, np.nan)

    return pd.DataFrame(
        {
            "frequency": np.repeat(freqs, n_time_bins),
            "time": np.tile(time_bin_centers, n_freqs),
            "correlation": correlations.flatten(),
            "p_corrected": p_corrected.flatten(),
            "significant": significant_mask.flatten(),
            "n_valid": n_valid.flatten(),
            "cluster_id": cluster_flat,
            "cluster_significant": cluster_sig_flat,
            "cluster_p": cluster_p_flat,
        }
    )


def _log_tf_correlation_summary(
    summary_df,
    alpha,
    cluster_labels,
    cluster_sig_mask,
    cluster_alpha,
    cluster_n_perm,
    logger,
):
    if summary_df.empty:
        logger.warning("No valid TF correlations available for summary")
        return

    n_significant = int(np.nansum(summary_df.get("significant", False)))
    max_r = float(np.nanmax(np.abs(summary_df["correlation"])))
    max_idx = np.nanargmax(np.abs(summary_df["correlation"]))
    best_row = summary_df.iloc[max_idx]

    logger.info("Time-frequency correlation summary:")
    logger.info("  - Total TF points: %d", len(summary_df))
    logger.info("  - Significant correlations (FDR < %.3f): %d", alpha, n_significant)

    if cluster_labels is not None and cluster_sig_mask is not None:
        sig_clusters = np.unique(cluster_labels[cluster_sig_mask])
        sig_clusters = [int(cid) for cid in sig_clusters if cid != 0]
        logger.info(
            "  - Significant clusters (cluster α=%.3f, n_perm=%d): %d",
            cluster_alpha,
            cluster_n_perm,
            len(sig_clusters),
        )

    logger.info(
        "  - Strongest correlation: r=%.3f at %.1f Hz, %.2f s",
        best_row["correlation"],
        best_row["frequency"],
        best_row["time"],
    )


def plot_time_frequency_correlation_heatmap(
    subject: str,
    task: Optional[str] = None,
    data_path: Optional[Path] = None,
) -> None:
    config = load_settings()
    if task is None:
        task = config.task

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Rendering time-frequency correlation heatmap for sub-{subject}")

    behavior_config = config.get("behavior_analysis", {})
    heatmap_config = behavior_config.get("time_frequency_heatmap", {})
    viz_config = behavior_config.get("visualization", {})

    roi_selection = heatmap_config.get("roi_selection")
    if roi_selection == "null":
        roi_selection = None
    roi_suffix = f"_{roi_selection.lower()}" if roi_selection else ""

    use_spearman = bool(config.get("statistics.use_spearman_default", True))
    method_suffix = "_spearman" if use_spearman else "_pearson"

    deriv_root = Path(config.deriv_root)
    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {}) if plot_cfg else {}
    plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations") if plot_cfg else "04_behavior_correlations"
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)

    if data_path is None:
        data_path = stats_dir / f"time_frequency_correlation_data{roi_suffix}{method_suffix}.npz"

    if not data_path.exists():
        logger.error(f"Precomputed TF correlation data not found: {data_path}")
        return

    logger.info(f"Loading TF correlation data from {data_path}")
    tf_data = _load_tf_correlation_data(data_path, config)

    correlations = tf_data["correlations"]
    cluster_labels, cluster_pvals, cluster_sig_mask = _normalize_cluster_data(
        tf_data["cluster_labels"],
        tf_data["cluster_pvals"],
        tf_data["cluster_sig_mask"],
    )

    time_bin_edges = _compute_time_bin_edges(
        tf_data["time_bin_centers"],
        tf_data["time_bin_edges"],
        tf_data["time_resolution"],
    )

    baseline_window_used = _extract_baseline_window(tf_data["baseline_window"])

    n_freqs, n_time_bins = correlations.shape
    if n_freqs == 0 or n_time_bins == 0:
        logger.warning("TF correlation data is empty; nothing to plot")
        return

    correlation_vmin = viz_config.get("correlation_vmin", -0.6)
    correlation_vmax = viz_config.get("correlation_vmax", 0.6)

    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    method_spearman = behavioral_config.get("method_spearman", "spearman")
    method_name = "Spearman" if tf_data["method"].lower() == method_spearman else "Pearson"
    roi_name = roi_selection or "All Channels"

    fig, ax = _create_tf_heatmap_plot(
        correlations,
        time_bin_edges,
        tf_data["freqs"],
        cluster_sig_mask,
        tf_data["time_bin_centers"],
        correlation_vmin,
        correlation_vmax,
        subject,
        method_name,
        roi_name,
        tf_data["baseline_applied"],
        baseline_window_used,
        config,
    )

    freq_bands = config.get("time_frequency_analysis.bands", {})
    _add_frequency_band_markers(ax, freq_bands, tf_data["freq_range"], config)

    plt.tight_layout()
    fig_name = f"time_frequency_correlation_heatmap{roi_suffix}{method_suffix}"
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        plots_dir / fig_name,
        formats=save_formats,
        bbox_inches="tight",
        footer=_get_behavior_footer(config),
    )
    plt.close(fig)

    stats_file = stats_dir / f"time_frequency_correlation_stats{roi_suffix}{method_suffix}.tsv"
    summary_df: Optional[pd.DataFrame] = None
    if stats_file.exists():
        try:
            summary_df = pd.read_csv(stats_file, sep="\t")
        except Exception as exc:
            logger.warning(f"Failed to read TF correlation summary ({exc})")

    if summary_df is None or summary_df.empty:
        summary_df = _create_summary_dataframe(
            tf_data["freqs"],
            tf_data["time_bin_centers"],
            correlations,
            tf_data["p_corrected"],
            tf_data["significant_mask"],
            tf_data["n_valid"],
            cluster_labels,
            cluster_sig_mask,
            cluster_pvals,
        )

    _log_tf_correlation_summary(
        summary_df,
        tf_data["alpha"],
        cluster_labels,
        cluster_sig_mask,
        tf_data["cluster_alpha"],
        tf_data["cluster_n_perm"],
        logger,
    )
