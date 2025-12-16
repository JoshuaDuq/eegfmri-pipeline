from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.utils.validation import ensure_aligned_lengths
from eeg_pipeline.plotting.io.figures import (
    save_fig,
    get_behavior_footer as _get_behavior_footer,
    logratio_to_pct as _logratio_to_pct,
    pct_to_logratio as _pct_to_logratio,
    get_default_config as _get_default_config,
)
from eeg_pipeline.utils.formatting import format_channel_list_for_display, format_roi_description
from eeg_pipeline.infra.logging import get_default_logger as _get_default_logger
from eeg_pipeline.utils.analysis.stats import (
    format_correlation_stats_text,
    bootstrap_corr_ci as _bootstrap_corr_ci,
    partial_corr_xy_given_Z as _partial_corr_xy_given_Z,
    joint_valid_mask,
    compute_linear_residuals,
    prepare_data_for_plotting,
    prepare_data_without_validation,
    compute_kde_scale,
)


###################################################################
# Figure Creation
###################################################################


def _create_scatter_figure(plot_cfg: PlotConfig) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
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


###################################################################
# Histogram and KDE Plotting
###################################################################


def _plot_histogram_with_kde(
    ax: plt.Axes,
    data: pd.Series,
    band_color: str,
    plot_cfg: PlotConfig,
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


###################################################################
# Axis and Annotation Helpers
###################################################################


def _should_show_percentage_axis(x_label: str) -> bool:
    power_indicators = ["log10(power", "residuals of log10(power"]
    return any(indicator in x_label for indicator in power_indicators)


def _add_percentage_axis(ax: plt.Axes, x_label: str, plot_cfg: PlotConfig) -> None:
    if not _should_show_percentage_axis(x_label):
        return
    
    behavioral_config = plot_cfg.get_behavioral_config()
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


def _add_channel_annotation(fig: plt.Figure, roi_channels: List[str], plot_cfg: PlotConfig) -> None:
    if not roi_channels:
        return
    
    behavioral_config = plot_cfg.get_behavioral_config()
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


def _get_line_color(p_value: float, plot_cfg: PlotConfig) -> str:
    behavioral_config = plot_cfg.get_behavioral_config()
    significance_threshold = behavioral_config.get("significance_threshold", 0.05)
    is_significant = np.isfinite(p_value) and p_value < significance_threshold
    return plot_cfg.get_color("significant", plot_type="behavioral") if is_significant else plot_cfg.get_color("nonsignificant", plot_type="behavioral")


###################################################################
# Title and Logging Helpers
###################################################################


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


###################################################################
# Correlation Scatter Plot
###################################################################


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
    
    behavioral_config = plot_cfg.get_behavioral_config()
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

    _log_plot_details(logger, title_prefix, roi_channels)
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi, 
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches,
             footer=_get_behavior_footer(config), logger=logger)
    plt.close(fig)


###################################################################
# Residual Diagnostics
###################################################################


def _plot_residuals_vs_fitted(
    ax: plt.Axes,
    fitted: np.ndarray,
    residuals: np.ndarray,
    band_color: str,
    plot_cfg: PlotConfig,
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


def _plot_qq_plot(ax: plt.Axes, residuals: np.ndarray, band_color: str, plot_cfg: PlotConfig) -> None:
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

    behavioral_config = plot_cfg.get_behavioral_config()
    residual_qc_figsize = tuple(behavioral_config.get("residual_qc_figsize", [12, 5]))
    fig, axes = plt.subplots(1, 2, figsize=residual_qc_figsize)
    _plot_residuals_vs_fitted(axes[0], fitted, residuals, band_color, plot_cfg)
    _plot_qq_plot(axes[1], residuals, band_color, plot_cfg)

    fig.suptitle(f"{title_prefix} — Residual QC", fontsize=plot_cfg.font.figure_title, 
                 fontweight="bold", y=plot_cfg.text_position.residual_qc_title_y)
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


def _plot_residual_histogram(ax: plt.Axes, residuals: np.ndarray, band_color: str, plot_cfg: PlotConfig) -> None:
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


def _plot_residuals_vs_predictor(ax: plt.Axes, x_clean: np.ndarray, residuals: np.ndarray, band_color: str, plot_cfg: PlotConfig) -> None:
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

    behavioral_config = plot_cfg.get_behavioral_config()
    diagnostics_figsize = tuple(behavioral_config.get("diagnostics_figsize", [10, 8]))
    fig, axes = plt.subplots(2, 2, figsize=diagnostics_figsize)
    _plot_residuals_vs_fitted(axes[0, 0], fitted, residuals, band_color, plot_cfg)
    _plot_qq_plot(axes[0, 1], residuals, band_color, plot_cfg)
    _plot_residual_histogram(axes[1, 0], residuals, band_color, plot_cfg)
    _plot_residuals_vs_predictor(axes[1, 1], x_clean, residuals, band_color, plot_cfg)

    fig.suptitle(f"{title_prefix} — Residual Diagnostics", fontsize=plot_cfg.font.figure_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

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
    logger.info(f"Residual diagnostics saved to {output_path}")
