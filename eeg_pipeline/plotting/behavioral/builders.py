from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
)
from eeg_pipeline.utils.formatting import format_channel_list_for_display, format_roi_description
from eeg_pipeline.infra.logging import get_default_logger as _get_default_logger
from eeg_pipeline.utils.analysis.stats import (
    format_correlation_stats_text,
    bootstrap_corr_ci as _bootstrap_corr_ci,
    partial_corr_xy_given_Z as _partial_corr_xy_given_Z,
    joint_valid_mask,
    prepare_data_for_plotting,
    compute_kde_scale,
)


###################################################################
# Figure Creation
###################################################################


def _create_scatter_figure(plot_cfg: PlotConfig) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig_size = plot_cfg.get_figure_size("standard", plot_type="behavioral")
    gridspec_params = plot_cfg.get_gridspec_params()
    
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=gridspec_params["width_ratios"],
        height_ratios=gridspec_params["height_ratios"],
        hspace=gridspec_params["hspace"],
        wspace=gridspec_params["wspace"],
        left=gridspec_params["left"],
        right=gridspec_params["right"],
        top=gridspec_params["top"],
        bottom=gridspec_params["bottom"],
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
    is_horizontal = orientation == "horizontal"
    
    hist_kwargs = {
        "bins": bins,
        "color": band_color,
        "alpha": plot_cfg.style.scatter.alpha,
        "edgecolor": plot_cfg.style.histogram.edgecolor,
        "linewidth": plot_cfg.style.histogram.edgewidth,
    }
    if is_horizontal:
        hist_kwargs["orientation"] = "horizontal"
    ax.hist(data, **hist_kwargs)
    
    min_samples_for_kde = plot_cfg.validation.get("min_samples_for_kde", 3)
    if len(data) <= min_samples_for_kde:
        return
    
    kde = gaussian_kde(data)
    data_min = data.min()
    data_max = data.max()
    data_range = np.linspace(data_min, data_max, plot_cfg.style.kde_points)
    kde_values = kde(data_range)
    kde_scale = compute_kde_scale(data, hist_bins=bins, kde_points=plot_cfg.style.kde_points)
    scaled_kde = kde_values * kde_scale
    
    kde_kwargs = {
        "color": plot_cfg.style.kde_color,
        "linewidth": plot_cfg.style.kde_linewidth,
        "alpha": plot_cfg.style.kde_alpha,
    }
    if is_horizontal:
        ax.plot(scaled_kde, data_range, **kde_kwargs)
    else:
        ax.plot(data_range, scaled_kde, **kde_kwargs)


###################################################################
# Axis and Annotation Helpers
###################################################################


def _should_show_percentage_axis(x_label: str) -> bool:
    if "(ranked)" in x_label.lower():
        return False
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
    
    if is_significant:
        return plot_cfg.get_color("significant", plot_type="behavioral")
    return plot_cfg.get_color("nonsignificant", plot_type="behavioral")


###################################################################
# Title and Logging Helpers
###################################################################


def _extract_band_info(title_prefix: str) -> str:
    title_lower = title_prefix.lower()
    if "power" not in title_lower:
        return ""
    
    frequency_bands = ["delta", "theta", "alpha", "beta", "gamma"]
    for band in frequency_bands:
        if band in title_lower:
            return f" ({band.upper()})"
    return ""


def _extract_target_info(title_prefix: str) -> str:
    title_lower = title_prefix.lower()
    if "outcome" in title_lower:
        return " vs rating"
    if "predictor" in title_lower or "temp" in title_lower:
        return " vs predictor"
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


def _compute_correlation_statistics(
    x_clean: pd.Series,
    y_clean: pd.Series,
    method_code: str,
    plot_cfg: PlotConfig,
    *,
    Z_covars: Optional[pd.DataFrame] = None,
    is_partial_residuals: bool = False,
    annotated_stats: Optional[Tuple[float, float, int]] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    if annotated_stats is not None:
        return annotated_stats
    
    if Z_covars is not None and not Z_covars.empty:
        return _partial_corr_xy_given_Z(x_clean, y_clean, Z_covars, method_code)
    
    mask = joint_valid_mask(x_clean, y_clean)
    n_valid = int(mask.sum())
    min_samples = plot_cfg.validation.get("min_samples_for_plot", 5)
    
    if n_valid < min_samples:
        return np.nan, np.nan, n_valid
    
    x_vals = x_clean.iloc[mask] if isinstance(x_clean, pd.Series) else x_clean[mask]
    y_vals = y_clean.iloc[mask] if isinstance(y_clean, pd.Series) else y_clean[mask]
    
    use_spearman = method_code.lower() == "spearman" and not is_partial_residuals
    if use_spearman:
        correlation, p_value = stats.spearmanr(x_vals, y_vals, nan_policy="omit")
    else:
        correlation, p_value = stats.pearsonr(x_vals, y_vals)
    
    return correlation, p_value, n_valid


def _compute_confidence_interval(
    x_clean: pd.Series,
    y_clean: pd.Series,
    method_code: str,
    *,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    annot_ci: Optional[Tuple[float, float]] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    if annot_ci is not None:
        return annot_ci
    
    if bootstrap_ci > 0:
        return _bootstrap_corr_ci(
            x_clean, y_clean, method_code, n_boot=bootstrap_ci, rng=rng, config=config
        )
    
    return np.nan, np.nan


def _plot_regression_with_histograms(
    fig: plt.Figure,
    ax_main: plt.Axes,
    ax_histx: plt.Axes,
    ax_histy: plt.Axes,
    x_clean: pd.Series,
    y_clean: pd.Series,
    x_label: str,
    y_label: str,
    band_color: str,
    line_color: str,
    plot_cfg: PlotConfig,
) -> None:
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    line_width = plot_cfg.get_line_width("bold", plot_type="behavioral")
    behavioral_config = plot_cfg.get_behavioral_config()
    
    show_regression_ci = bool(behavioral_config.get("show_regression_ci", False))
    regression_ci_level = int(behavioral_config.get("correlation_ci_level", 95)) if show_regression_ci else None
    
    sns.regplot(
        x=x_clean,
        y=y_clean,
        ax=ax_main,
        ci=regression_ci_level,
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


def _add_statistics_annotation(
    fig: plt.Figure,
    r_value: float,
    p_value: float,
    n_samples: int,
    ci: Tuple[float, float],
    stats_tag: Optional[str],
    plot_cfg: PlotConfig,
    config: Optional[Any],
) -> None:
    stats_text = format_correlation_stats_text(
        r_value, p_value, n_samples, ci, stats_tag, config=config
    )
    fig.text(
        plot_cfg.text_position.p_value_x,
        plot_cfg.text_position.p_value_y,
        stats_text,
        fontsize=plot_cfg.font.title,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=plot_cfg.style.alpha_text_box),
    )


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
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    is_partial_residuals: bool = False,
    roi_channels: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    annotated_stats: Optional[Tuple[float, float, int]] = None,
    annot_ci: Optional[Tuple[float, float]] = None,
    stats_tag: Optional[str] = None,
    config: Optional[Any] = None,
) -> None:
    logger = logger or _get_default_logger()
    if config is None:
        raise ValueError("config is required for behavioral plotting")
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    
    ensure_aligned_lengths(x_data, y_data, context="Correlation scatter inputs", strict=True)

    x_clean, y_clean, n_eff = prepare_data_for_plotting(x_data, y_data)

    min_samples = plot_cfg.validation.get("min_samples_for_plot", 5)
    if n_eff < min_samples:
        logger.warning(
            f"Insufficient data for scatter plot (n={n_eff} < {min_samples}), "
            f"skipping {output_path}"
        )
        return

    r_value, p_value, n_samples = _compute_correlation_statistics(
        x_clean,
        y_clean,
        method_code,
        plot_cfg,
        Z_covars=Z_covars,
        is_partial_residuals=is_partial_residuals,
        annotated_stats=annotated_stats,
        config=config,
    )
    
    ci = _compute_confidence_interval(
        x_clean,
        y_clean,
        method_code,
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        annot_ci=annot_ci,
        config=config,
    )

    fig, ax_main, ax_histx, ax_histy = _create_scatter_figure(plot_cfg)
    line_color = _get_line_color(p_value, plot_cfg)

    _plot_regression_with_histograms(
        fig, ax_main, ax_histx, ax_histy,
        x_clean, y_clean, x_label, y_label,
        band_color, line_color, plot_cfg,
    )

    _add_statistics_annotation(
        fig, r_value, p_value, n_samples, ci, stats_tag, plot_cfg, config
    )

    if roi_channels:
        _add_channel_annotation(fig, roi_channels, plot_cfg)

    fig.suptitle(
        title_prefix,
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=plot_cfg.text_position.title_y,
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()

    _log_plot_details(logger, title_prefix, roi_channels)

    significance_threshold = float(behavioral_config.get("significance_threshold", 0.05))
    has_covariates = Z_covars is not None and not Z_covars.empty
    has_partial = has_covariates or bool(is_partial_residuals)
    inference = "Displayed: partial-correlation p-values (uncorrected)" if has_partial else "Displayed: correlation p-values (uncorrected)"

    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config, inference=inference, alpha=significance_threshold),
        logger=logger,
    )
    plt.close(fig)
