from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, StrMethodFormatter
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from scipy import interpolate

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    _build_covariate_matrices,
    _load_features_and_targets,
    _pick_first_column,
    get_available_subjects,
    load_epochs_for_analysis,
    pick_event_columns,
)
from eeg_pipeline.utils.io_utils import (
    deriv_group_plots_path,
    deriv_group_stats_path,
    deriv_features_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_aligned_lengths,
    ensure_dir,
    fdr_bh_reject,
    find_connectivity_features_path,
    save_fig,
    _load_events_df,
)
from eeg_pipeline.utils.io_utils import get_group_logger, get_subject_logger
from eeg_pipeline.utils.io_utils import (
    get_band_color as _get_band_color,
    get_behavior_footer as _get_behavior_footer,
    logratio_to_pct as _logratio_to_pct,
    pct_to_logratio as _pct_to_logratio,
    sanitize_label,
)
from eeg_pipeline.utils.stats_utils import (
    bh_adjust as _bh_adjust,
    compute_partial_corr as _compute_partial_corr,
    compute_partial_residuals as _compute_partial_residuals,
    fisher_ci as _fisher_ci,
    _get_ttest_pvalue,
    _safe_float,
    bootstrap_corr_ci as _bootstrap_corr_ci,
    compute_group_corr_stats as _compute_group_corr_stats,
    partial_corr_xy_given_Z as _partial_corr_xy_given_Z,
    partial_residuals_xy_given_Z as _partial_residuals_xy_given_Z,
)
from eeg_pipeline.utils.tfr_utils import build_rois_from_info as _build_rois, validate_baseline_window

PLOT_SUBDIR = "04_behavior_correlation_analysis"




###################################################################
# Plot Builders
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
    if logger is None:
        logger = logging.getLogger(__name__)
    if config is None:
        from eeg_pipeline.utils.config_loader import load_settings
        config = load_settings()

    from eeg_pipeline.utils.io_utils import ensure_aligned_lengths
    ensure_aligned_lengths(x_data, y_data, context="Correlation scatter inputs", strict=True)
    x, y = x_data, y_data

    if is_partial_residuals:
        m = pd.Series([True] * len(x), index=x.index if hasattr(x, "index") else range(len(x)))
        n_eff = len(x)
        x_clean = x
        y_clean = y
    else:
        m = x.notna() & y.notna()
        n_eff = m.sum()
        x_clean = x[m]
        y_clean = y[m]

    if n_eff < 5:
        logger.warning(f"Insufficient data for scatter plot (n={n_eff} < 5), skipping {output_path}")
        return

    if annotated_stats is not None:
        r_disp, p_disp, n_disp = annotated_stats
    else:
        r_disp, p_disp, n_disp = np.nan, np.nan, n_eff

    ci_disp = annot_ci if annot_ci is not None else (np.nan, np.nan)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.15,
        wspace=0.15,
        left=0.1,
        right=0.95,
        top=0.80,
        bottom=0.12,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)

    line_color = "#C42847" if (np.isfinite(p_disp) and p_disp < 0.05) else "#666666"
    sns.regplot(
        x=x_clean,
        y=y_clean,
        ax=ax_main,
        ci=95,
        scatter_kws={"s": 30, "alpha": 0.7, "color": band_color, "edgecolor": "white", "linewidths": 0.3},
        line_kws={"color": line_color, "lw": 1.5},
    )

    ax_histx.hist(x_clean, bins=15, color=band_color, alpha=0.7, edgecolor="white", linewidth=0.5)
    if len(x_clean) > 3:
        kde_x = gaussian_kde(x_clean)
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
        kde_vals = kde_x(x_range)
        hist_counts, _ = np.histogram(x_clean, bins=15)
        kde_scale = hist_counts.max() / kde_vals.max() if kde_vals.max() > 0 else 1
        ax_histx.plot(x_range, kde_vals * kde_scale, color="darkblue", linewidth=1.5, alpha=0.8)

    ax_histy.hist(
        y_clean,
        bins=15,
        orientation="horizontal",
        color=band_color,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    if len(y_clean) > 3:
        kde_y = gaussian_kde(y_clean)
        y_range = np.linspace(y_clean.min(), y_clean.max(), 100)
        kde_vals_y = kde_y(y_range)
        hist_counts_y, _ = np.histogram(y_clean, bins=15)
        kde_scale_y = hist_counts_y.max() / kde_vals_y.max() if kde_vals_y.max() > 0 else 1
        ax_histy.plot(kde_vals_y * kde_scale_y, y_range, color="darkblue", linewidth=1.5, alpha=0.8)

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    show_pct_axis = ("log10(power" in x_label) if not is_partial_residuals else ("residuals of log10(power" in x_label)

    if show_pct_axis:
        try:
            ax_pct = ax_histx.secondary_xaxis("top", functions=(_logratio_to_pct, _pct_to_logratio))
            ax_pct.set_xlabel("Power Change (%)", fontsize=9)
            # Use a few more major ticks and clean integer labels; add minor ticks for readability
            ax_pct.xaxis.set_major_locator(MaxNLocator(nbins=7))
            ax_pct.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax_pct.xaxis.set_minor_locator(AutoMinorLocator(2))
        except (AttributeError, TypeError, ValueError):
            pass

    label = "Spearman \u03c1"
    ci_str = ""
    if ci_disp is not None and np.all(np.isfinite(ci_disp)):
        ci_str = f"\nCI [{ci_disp[0]:.2f}, {ci_disp[1]:.2f}]"
    tag_str = f" {stats_tag}" if stats_tag else ""
    stats_text = f"{label}{tag_str} = {r_disp:.3f}\np = {p_disp:.3f}\nn = {n_disp}{ci_str}"
    fig.text(
        0.98,
        0.94,
        stats_text,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    # Optional ROI channels annotation (if provided)
    if roi_channels:
        chan_text = "Channels: " + ", ".join(roi_channels[:10])
        fig.text(
            0.02,
            0.94,
            chan_text,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    # True figure-level title at top
    fig.suptitle(title_prefix, fontsize=12, fontweight="bold", y=0.975)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    
    # Extract plot details for logging
    plot_details = []
    if roi_channels:
        if len(roi_channels) == 1:
            plot_details.append(f"Channel: {roi_channels[0]}")
        else:
            plot_details.append(f"ROI: {len(roi_channels)} channels")
    else:
        plot_details.append("Overall")
    
    # Extract band from title_prefix if available
    band_info = ""
    if "power" in title_prefix.lower():
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            if band in title_prefix.lower():
                band_info = f" ({band.upper()})"
                break
    
    # Extract target from title_prefix
    target_info = ""
    if "rating" in title_prefix.lower():
        target_info = " vs rating"
    elif "temp" in title_prefix.lower() or "temperature" in title_prefix.lower():
        target_info = " vs temperature"
    
    plot_desc = f"Scatter plot{band_info}{target_info}: {' | '.join(plot_details)}"
    if logger:
        logger.info(plot_desc)
    
    save_fig(fig, output_path, formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config), logger=logger)
    plt.close(fig)


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
    if config is None:
        config = load_settings()
    if logger is None:
        logger = logging.getLogger(__name__)

    x_series = pd.to_numeric(x_data, errors="coerce")
    y_series = pd.to_numeric(y_data, errors="coerce")
    mask = x_series.notna() & y_series.notna()
    n_obs = int(mask.sum())
    if n_obs < 5:
        logger.warning("Residual diagnostics skipped: insufficient paired samples (<5).")
        return

    x_clean = x_series[mask].to_numpy(dtype=float)
    y_clean = y_series[mask].to_numpy(dtype=float)

    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
    fitted = intercept + slope * x_clean
    residuals = y_clean - fitted

    order = np.argsort(fitted)
    fitted_ord = fitted[order]
    residual_ord = residuals[order]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_res_vs_fit = axes[0, 0]
    ax_qq = axes[0, 1]
    ax_hist = axes[1, 0]
    ax_res_vs_x = axes[1, 1]

    ax_res_vs_fit.scatter(fitted_ord, residual_ord, s=25, alpha=0.7, color=band_color, edgecolor="white", linewidth=0.4)
    ax_res_vs_fit.axhline(0, color="k", linestyle="--", linewidth=1)
    ax_res_vs_fit.set_xlabel("Fitted values")
    ax_res_vs_fit.set_ylabel("Residuals")
    ax_res_vs_fit.set_title("Residuals vs Fitted")
    ax_res_vs_fit.grid(True, alpha=0.3)

    qq_data = stats.probplot(residuals, dist="norm")
    ax_qq.scatter(qq_data[0][0], qq_data[0][1], color=band_color, alpha=0.7, s=25, edgecolor="white", linewidth=0.4)
    ax_qq.plot(qq_data[0][0], qq_data[1][0] + qq_data[1][1] * qq_data[0][0], color="k", linestyle="--", linewidth=1)
    ax_qq.set_title("Normal Q-Q")
    ax_qq.set_xlabel("Theoretical quantiles")
    ax_qq.set_ylabel("Sample quantiles")
    ax_qq.grid(True, alpha=0.3)

    ax_hist.hist(residuals, bins=20, color=band_color, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax_hist.set_title("Residual distribution")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.3)

    ax_res_vs_x.scatter(x_clean, residuals, s=25, alpha=0.7, color=band_color, edgecolor="white", linewidth=0.4)
    ax_res_vs_x.axhline(0, color="k", linestyle="--", linewidth=1)
    ax_res_vs_x.set_xlabel("Predictor")
    ax_res_vs_x.set_ylabel("Residuals")
    ax_res_vs_x.set_title("Residuals vs Predictor")
    ax_res_vs_x.grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix} — Residual Diagnostics", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(
        fig,
        output_path,
        formats=save_formats,
        bbox_inches="tight",
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)
    logger.info(f"Residual diagnostics saved to {output_path}")


###################################################################
# Subject Plots
###################################################################

def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
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

    # Clean columns
    temp = pd.to_numeric(events[temp_col], errors="coerce")
    # Plot continuous rating vs temperature if available
    if rating_col is not None:
        rating = pd.to_numeric(events[rating_col], errors="coerce")
        mask = temp.notna() & rating.notna()
        if mask.sum() >= 5:
            t = temp[mask]
            r = rating[mask]
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.regplot(x=t, y=r, scatter_kws={"s": 25, "alpha": 0.7}, line_kws={"color": "k"}, ax=ax)
            ax.set_xlabel(f"Temperature")
            ax.set_ylabel(f"Rating")
            ax.set_title("Rating vs Temperature")
            ax.grid(True, alpha=0.3)
            sr, sp = stats.spearmanr(t, r, nan_policy="omit")
            if sp < 0.001:
                p_text = "p < .001"
            elif sp < 0.01:
                p_text = f"p < .01"
            elif sp < 0.05:
                p_text = f"p < .05"
            else:
                p_text = f"p = {sp:.3f}"
            stats_text = f'ρ = {sr:.3f}, {p_text}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontweight='bold', fontsize=10)

            try:
                t_vals = np.asarray(t, dtype=float)
                uniq = np.unique(np.round(t_vals, 6))
                # If temperatures take on a small number of discrete values, use them as ticks
                if uniq.size <= 12:
                    ax.set_xticks(uniq)
                    # If all are integers, force integer locator to prevent half-step misalignment
                    if np.allclose(uniq, uniq.astype(int)):
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
                    else:
                        # Otherwise show up to 2 decimals for readability
                        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
                # Set tight x-limits with a small margin to avoid tick/point edge clipping
                if np.isfinite(t_vals).any():
                    xmin = np.nanmin(t_vals)
                    xmax = np.nanmax(t_vals)
                    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                        pad = 0.02 * (xmax - xmin) if xmax - xmin > 0 else 0.5
                        ax.set_xlim(xmin - pad, xmax + pad)
            except Exception:
                pass
            
            save_formats = config.get("output.save_formats", ["png"])
            save_fig(fig, plots_dir / "psychometric_rating_vs_temp", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
            plt.close(fig)
            
            # Save Spearman only (consistent metric)
            pd.DataFrame({
                "metric": ["spearman_r", "spearman_p"],
                "value": [sr, sp],
            }).to_csv(stats_dir / "psychometric_rating_vs_temp_stats.tsv", sep="\t", index=False)


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
) -> None:
    config = load_settings()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ROI power scatter plotting for sub-{subject}")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    ensure_dir(plots_dir)

    if task is None:
        task = config.task

    rng = rng or np.random.default_rng(42)
    temporal_df, pow_df, _conn_df, y, info = _load_features_and_targets(subject, task, deriv_root, config)
    y = pd.to_numeric(y, errors="coerce")
    epochs, aligned_events = load_epochs_for_analysis(subject, task, align="strict", preload=False, deriv_root=deriv_root, bids_root=config.bids_root, config=config, logger=logger)
    if epochs is None:
        logger.error(f"Could not find epochs for ROI scatter plots: sub-{subject}")
        return
    if rating_stats is None:
        logger.warning("ROI rating statistics not provided; plots will use empirical correlations only.")
    if do_temp and temp_stats is None:
        logger.warning("ROI temperature statistics not provided; temperature annotations will be omitted.")

    temp_series = None
    temp_col = None
    psych_temp_columns = config.get("event_columns.temperature", [])
    if aligned_events is not None:
        temp_col = _pick_first_column(aligned_events, psych_temp_columns)
        if temp_col:
            temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
            
    Z_df_full, Z_df_temp = _build_covariate_matrices(aligned_events, partial_covars, temp_col, config=config)
    roi_map = _build_rois(info, config=config)
    def _fetch_roi_stats(df: Optional[pd.DataFrame], roi_name: str, band_name: str) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        if "roi" not in df.columns or "band" not in df.columns:
            return None
        mask = (
            df["band"].astype(str).str.lower() == band_name.lower()
        ) & (
            df["roi"].astype(str).str.lower() == roi_name.lower()
        )
        if mask.any():
            return df.loc[mask].iloc[0]
        return None

    def _fetch_overall_stats(df: Optional[pd.DataFrame], band_name: str) -> Optional[pd.Series]:
        for key in ["overall", "all", "global"]:
            row = _fetch_roi_stats(df, key, band_name)
            if row is not None:
                return row
        return None

    power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    for band in power_bands_to_use:
        band_cols = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols:
            continue
        freq_bands = config.get("time_frequency_analysis.bands", {})
        band_rng = freq_bands.get(band)
        if band_rng:
            band_rng = tuple(band_rng)
        band_title = band.capitalize()
        band_color = _get_band_color(band)

        overall_vals = pow_df[list(band_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        method_code = "spearman"
        covar_names = list(Z_df_full.columns) if Z_df_full is not None and not Z_df_full.empty else None

        overall_plots_dir = plots_dir / "overall"
        ensure_dir(overall_plots_dir)

        m = overall_vals.notna() & y.notna()
        n_eff = int(m.sum())
        r_direct, p_direct = np.nan, np.nan
        ci_direct = (np.nan, np.nan)
        if n_eff >= 5:
            r_direct, p_direct = stats.spearmanr(overall_vals[m], y[m], nan_policy="omit")
            if bootstrap_ci > 0:
                ci_direct = _bootstrap_corr_ci(overall_vals[m], y[m], method_code, n_boot=bootstrap_ci, rng=rng)

        stats_overall = _fetch_overall_stats(rating_stats, band)
        if stats_overall is not None:
            n_eff = int(stats_overall.get("n", n_eff))
            r_direct = _safe_float(stats_overall.get("r", r_direct))
            p_direct = _safe_float(stats_overall.get("p", p_direct))
            ci_direct = (
                _safe_float(stats_overall.get("r_ci_low", ci_direct[0])),
                _safe_float(stats_overall.get("r_ci_high", ci_direct[1])),
            )

        generate_correlation_scatter(
            x_data=overall_vals, y_data=y,
            x_label="log10(power/baseline [-5–0 s])", y_label="Rating",
            title_prefix=f"{band_title} power vs rating — Overall (plateau)",
            band_color=band_color,
            output_path=overall_plots_dir / f"scatter_pow_overall_{sanitize_label(band)}_vs_rating_plateau",
            method_code=method_code, Z_covars=Z_df_full, covar_names=covar_names,
            bootstrap_ci=0, rng=rng, logger=logger,
            annotated_stats=(r_direct, p_direct, n_eff),
            annot_ci=ci_direct,
            config=config
        )
        
        if temporal_df is None:
            logger.info("No group temporal features available; skipping.")
        else:
            for time_label in ["early", "mid", "late"]:
                temporal_band_cols = [c for c in temporal_df.columns if c.startswith(f"pow_{band}_") and c.endswith(f"_{time_label}")]
                if not temporal_band_cols:
                    continue
                temporal_overall_vals = temporal_df[temporal_band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                m_temporal = temporal_overall_vals.notna() & y.notna()
                n_eff_temporal = int(m_temporal.sum())
                r_temporal, p_temporal = np.nan, np.nan
                ci_temporal = (np.nan, np.nan)
                if n_eff_temporal >= 5:
                    r_temporal, p_temporal = stats.spearmanr(temporal_overall_vals[m_temporal], y[m_temporal], nan_policy="omit")
                    if bootstrap_ci > 0:
                        ci_temporal = _bootstrap_corr_ci(temporal_overall_vals[m_temporal], y[m_temporal], method_code, n_boot=bootstrap_ci, rng=rng)
                
                generate_correlation_scatter(
                    x_data=temporal_overall_vals, y_data=y,
                    x_label=f"log10(power/baseline) [{time_label} window]", y_label="Rating",
                    title_prefix=f"{band_title} power vs rating — Overall ({time_label})",
                    band_color=band_color,
                    output_path=overall_plots_dir / f"scatter_pow_overall_{sanitize_label(band)}_vs_rating_{time_label}",
                    method_code=method_code, Z_covars=Z_df_full, covar_names=covar_names,
                    bootstrap_ci=0, rng=rng, logger=logger,
                    annotated_stats=(r_temporal, p_temporal, n_eff_temporal),
                    annot_ci=ci_temporal,
                    config=config
                )
        
        plot_regression_residual_diagnostics(
            x_data=overall_vals, y_data=y,
            title_prefix=f"{band_title} power vs rating — Overall",
            output_path=overall_plots_dir / f"residual_diagnostics_pow_overall_{sanitize_label(band)}_vs_rating",
            band_color=band_color, logger=logger, config=config
        )
        
        
        if Z_df_full is not None and not Z_df_full.empty:
            n_len_pt = min(len(overall_vals), len(y), len(Z_df_full))
            x_part, y_part = overall_vals.iloc[:n_len_pt], y.iloc[:n_len_pt] 
            Z_part = Z_df_full.iloc[:n_len_pt]
            x_res_sr, y_res_sr, n_res = _compute_partial_residuals(
                x_part, y_part, Z_part, method_code, logger=logger, context=f"Partial residuals overall {band}"
            )
            if n_res >= 5:
                r_resid = np.nan
                p_resid = np.nan
                n_partial = n_res
                if stats_overall is not None:
                    r_resid = _safe_float(stats_overall.get("r_partial", r_resid))
                    p_resid = _safe_float(stats_overall.get("p_partial", p_resid))
                    n_partial = int(stats_overall.get("n_partial", n_partial))
                if not np.isfinite(r_resid) or not np.isfinite(p_resid):
                    r_resid, p_resid = stats.spearmanr(x_res_sr, y_res_sr, nan_policy="omit")
                ci_resid = (np.nan, np.nan)
                if bootstrap_ci > 0:
                    ci_resid = _bootstrap_corr_ci(x_res_sr, y_res_sr, method_code, n_boot=bootstrap_ci, rng=rng)
                residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)" if method_code == "spearman" else "Partial residuals of log10(power/baseline)"
                residual_ylabel = "Partial residuals (ranked) of rating" if method_code == "spearman" else "Partial residuals of rating"
                
                generate_correlation_scatter(
                    x_data=x_res_sr, y_data=y_res_sr,
                    x_label=residual_xlabel, y_label=residual_ylabel,
                    title_prefix=f"Partial residuals — {band_title} vs rating — Overall",
                    band_color=band_color,
                    output_path=overall_plots_dir / f"scatter_pow_overall_{sanitize_label(band)}_vs_rating_partial",
                    method_code=method_code, bootstrap_ci=0, rng=rng,
                    is_partial_residuals=True, logger=logger,
                    annotated_stats=(r_resid, p_resid, n_partial),
                    annot_ci=ci_resid,
                    config=config
                )

        if do_temp and temp_series is not None and not temp_series.empty:
            method2_code = "spearman"
            covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None and not Z_df_temp.empty else None
            
            m_temp = overall_vals.notna() & temp_series.notna()
            n_eff_temp = int(m_temp.sum())
            r_temp, p_temp = np.nan, np.nan
            ci_temp = (np.nan, np.nan)
            if n_eff_temp >= 5:
                r_temp, p_temp = stats.spearmanr(overall_vals[m_temp], temp_series[m_temp], nan_policy="omit")
                if bootstrap_ci > 0:
                    ci_temp = _bootstrap_corr_ci(overall_vals[m_temp], temp_series[m_temp], method2_code, n_boot=bootstrap_ci, rng=rng)

            stats_overall_temp = _fetch_overall_stats(temp_stats, band)
            if stats_overall_temp is not None:
                n_eff_temp = int(stats_overall_temp.get("n", n_eff_temp))
                r_temp = _safe_float(stats_overall_temp.get("r", r_temp))
                p_temp = _safe_float(stats_overall_temp.get("p", p_temp))
            
            generate_correlation_scatter(
                x_data=overall_vals, y_data=temp_series,
                x_label="log10(power/baseline [-5–0 s])", y_label="Temperature (°C)",
                title_prefix=f"{band_title} power vs temperature — Overall (plateau)",
                band_color=band_color,
                output_path=overall_plots_dir / f"scatter_pow_overall_{sanitize_label(band)}_vs_temp_plateau",
                method_code=method2_code, Z_covars=Z_df_temp, covar_names=covar_names_temp,
                bootstrap_ci=0, rng=rng, logger=logger,
                annotated_stats=(r_temp, p_temp, n_eff_temp),
                annot_ci=ci_temp,
                config=config
            )
            
            if temporal_df is not None:
                for time_label in ["early", "mid", "late"]:
                    temporal_band_cols = [c for c in temporal_df.columns if c.startswith(f"pow_{band}_") and c.endswith(f"_{time_label}")]
                    if not temporal_band_cols:
                        continue
                    temporal_overall_vals = temporal_df[temporal_band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    
                    m_temp_temporal = temporal_overall_vals.notna() & temp_series.notna()
                    n_eff_temp_temporal = int(m_temp_temporal.sum())
                    r_temp_temporal, p_temp_temporal = np.nan, np.nan
                    ci_temp_temporal = (np.nan, np.nan)
                    if n_eff_temp_temporal >= 5:
                        r_temp_temporal, p_temp_temporal = stats.spearmanr(temporal_overall_vals[m_temp_temporal], temp_series[m_temp_temporal], nan_policy="omit")
                        if bootstrap_ci > 0:
                            ci_temp_temporal = _bootstrap_corr_ci(temporal_overall_vals[m_temp_temporal], temp_series[m_temp_temporal], method2_code, n_boot=bootstrap_ci, rng=rng)
                    
                    generate_correlation_scatter(
                        x_data=temporal_overall_vals, y_data=temp_series,
                        x_label=f"log10(power/baseline) [{time_label} window]", y_label="Temperature (°C)",
                        title_prefix=f"{band_title} power vs temperature — Overall ({time_label})",
                        band_color=band_color,
                        output_path=overall_plots_dir / f"scatter_pow_overall_{sanitize_label(band)}_vs_temp_{time_label}",
                        method_code=method2_code, Z_covars=Z_df_temp, covar_names=covar_names_temp,
                        bootstrap_ci=0, rng=rng, logger=logger,
                        annotated_stats=(r_temp_temporal, p_temp_temporal, n_eff_temp_temporal),
                        annot_ci=ci_temp_temporal,
                        config=config
                    )
            
            plot_regression_residual_diagnostics(
                x_data=overall_vals, y_data=temp_series,
                title_prefix=f"{band_title} power vs temperature — Overall",
                output_path=overall_plots_dir / f"residual_diagnostics_pow_overall_{sanitize_label(band)}_vs_temp",
                band_color=band_color, logger=logger, config=config
            )
            
            if Z_df_temp is not None and not Z_df_temp.empty:
                n_len_pt2 = min(len(overall_vals), len(temp_series), len(Z_df_temp))
                x_part2, y_part2 = overall_vals.iloc[:n_len_pt2], temp_series.iloc[:n_len_pt2]
                Z_part2 = Z_df_temp.iloc[:n_len_pt2]
                x2_res_sr, y2_res_sr, n2_res = _compute_partial_residuals(
                    x_part2, y_part2, Z_part2, method2_code, logger=logger, context=f"Partial residuals overall temp {band}"
                )
                if n2_res >= 5:
                    r_resid2, p_resid2 = stats.spearmanr(x2_res_sr, y2_res_sr, nan_policy="omit")
                    ci_resid2 = (np.nan, np.nan)
                    if bootstrap_ci > 0:
                        ci_resid2 = _bootstrap_corr_ci(x2_res_sr, y2_res_sr, method2_code, n_boot=bootstrap_ci, rng=rng)
                    residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)"
                    residual_ylabel = "Partial residuals (ranked) of temperature (°C)"
                    
                    generate_correlation_scatter(
                        x_data=x2_res_sr, y_data=y2_res_sr,
                        x_label=residual_xlabel, y_label=residual_ylabel,
                        title_prefix=f"Partial residuals — {band_title} vs temperature — Overall",
                        band_color=band_color,
                        output_path=overall_plots_dir / f"scatter_pow_overall_{sanitize_label(band)}_vs_temp_partial",
                        method_code=method2_code, bootstrap_ci=0, rng=rng,
                        is_partial_residuals=True, logger=logger,
                        annotated_stats=(r_resid2, p_resid2, n2_res),
                        annot_ci=ci_resid2,
                        config=config
                    )

        for roi, chs in roi_map.items():
            roi_cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in band_cols]
            if not roi_cols:
                continue
            roi_vals = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

            roi_plots_dir = plots_dir / "roi_scatters" / sanitize_label(roi)
            ensure_dir(roi_plots_dir)

            m_roi = roi_vals.notna() & y.notna()
            n_eff_roi = int(m_roi.sum())
            r_roi, p_roi = np.nan, np.nan
            ci_roi = (np.nan, np.nan)
            if n_eff_roi >= 5:
                r_roi, p_roi = stats.spearmanr(roi_vals[m_roi], y[m_roi], nan_policy="omit")
                if bootstrap_ci > 0:
                    ci_roi = _bootstrap_corr_ci(roi_vals[m_roi], y[m_roi], method_code, n_boot=bootstrap_ci, rng=rng)

            stats_roi = _fetch_roi_stats(rating_stats, roi, band)
            if stats_roi is not None:
                n_eff_roi = int(stats_roi.get("n", n_eff_roi))
                r_roi = _safe_float(stats_roi.get("r", r_roi))
                p_roi = _safe_float(stats_roi.get("p", p_roi))
                ci_roi = (
                    _safe_float(stats_roi.get("r_ci_low", ci_roi[0])),
                    _safe_float(stats_roi.get("r_ci_high", ci_roi[1])),
                )

            generate_correlation_scatter(
                x_data=roi_vals, y_data=y,
                x_label="log10(power/baseline [-5–0 s])", y_label="Rating",
                title_prefix=f"{band_title} power vs rating — {roi} (plateau)",
                band_color=band_color,
                output_path=roi_plots_dir / f"scatter_pow_{sanitize_label(band)}_vs_rating_plateau",
                method_code=method_code, Z_covars=Z_df_full, covar_names=covar_names,
                bootstrap_ci=0, rng=rng, roi_channels=chs, logger=logger,
                annotated_stats=(r_roi, p_roi, n_eff_roi),
                annot_ci=ci_roi,
                config=config
            )
            
            if temporal_df is not None:
                for time_label in ["early", "mid", "late"]:
                    temporal_band_cols = [c for c in temporal_df.columns if c.startswith(f"pow_{band}_") and c.endswith(f"_{time_label}")]
                    if not temporal_band_cols:
                        continue
                    temporal_vals = temporal_df[temporal_band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    
                    m_roi_temporal = temporal_vals.notna() & y.notna()
                    n_eff_roi_temporal = int(m_roi_temporal.sum())
                    r_roi_temporal, p_roi_temporal = np.nan, np.nan
                    ci_roi_temporal = (np.nan, np.nan)
                    if n_eff_roi_temporal >= 5:
                        r_roi_temporal, p_roi_temporal = stats.spearmanr(temporal_vals[m_roi_temporal], y[m_roi_temporal], nan_policy="omit")
                        if bootstrap_ci > 0:
                            ci_roi_temporal = _bootstrap_corr_ci(temporal_vals[m_roi_temporal], y[m_roi_temporal], method_code, n_boot=bootstrap_ci, rng=rng)
                    
                    generate_correlation_scatter(
                        x_data=temporal_vals, y_data=y,
                        x_label=f"log10(power/baseline) [{time_label} window]", y_label="Rating",
                        title_prefix=f"{band_title} power vs rating — {roi} ({time_label})",
                        band_color=band_color,
                        output_path=roi_plots_dir / f"scatter_pow_{sanitize_label(band)}_vs_rating_{time_label}",
                        method_code=method_code, Z_covars=Z_df_full, covar_names=covar_names,
                        bootstrap_ci=0, rng=rng, roi_channels=chs, logger=logger,
                        annotated_stats=(r_roi_temporal, p_roi_temporal, n_eff_roi_temporal),
                        annot_ci=ci_roi_temporal,
                        config=config
                    )
            
            plot_regression_residual_diagnostics(
                x_data=roi_vals, y_data=y,
                title_prefix=f"{band_title} power vs rating — {roi}",
                output_path=roi_plots_dir / f"residual_diagnostics_pow_{sanitize_label(band)}_vs_rating",
                band_color=band_color, logger=logger, config=config
            )
            
            
            if Z_df_full is not None and not Z_df_full.empty:
                n_len_pt = min(len(roi_vals), len(y), len(Z_df_full))
                x_part, y_part = roi_vals.iloc[:n_len_pt], y.iloc[:n_len_pt]
                Z_part = Z_df_full.iloc[:n_len_pt]
                x_res_sr, y_res_sr, n_res = _compute_partial_residuals(
                    x_part, y_part, Z_part, method_code, logger=logger, context=f"Partial residuals {roi} rating {band}"
                )
                if n_res >= 5:
                    r_resid_roi = np.nan
                    p_resid_roi = np.nan
                    n_partial_roi = n_res
                    if stats_roi is not None:
                        r_resid_roi = _safe_float(stats_roi.get("r_partial", r_resid_roi))
                        p_resid_roi = _safe_float(stats_roi.get("p_partial", p_resid_roi))
                        n_partial_roi = int(stats_roi.get("n_partial", n_partial_roi))
                    if not np.isfinite(r_resid_roi) or not np.isfinite(p_resid_roi):
                        r_resid_roi, p_resid_roi = stats.spearmanr(x_res_sr, y_res_sr, nan_policy="omit")
                    ci_resid_roi = (np.nan, np.nan)
                    if bootstrap_ci > 0:
                        ci_resid_roi = _bootstrap_corr_ci(x_res_sr, y_res_sr, method_code, n_boot=bootstrap_ci, rng=rng)
                    residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)" if method_code == "spearman" else "Partial residuals of log10(power/baseline)"
                    residual_ylabel = "Partial residuals (ranked) of rating" if method_code == "spearman" else "Partial residuals of rating"
                    
                    generate_correlation_scatter(
                        x_data=x_res_sr, y_data=y_res_sr,
                        x_label=residual_xlabel, y_label=residual_ylabel,
                        title_prefix=f"Partial residuals — {band_title} vs rating — {roi}",
                        band_color=band_color,
                        output_path=roi_plots_dir / f"scatter_pow_{sanitize_label(band)}_vs_rating_partial",
                        method_code=method_code, bootstrap_ci=0, rng=rng,
                        is_partial_residuals=True, roi_channels=chs, logger=logger,
                        annotated_stats=(r_resid_roi, p_resid_roi, n_partial_roi),
                        annot_ci=ci_resid_roi,
                        config=config
                    )

            if do_temp and temp_series is not None and not temp_series.empty:
                method2_code = "spearman"
                covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None and not Z_df_temp.empty else None
                
                m_roi_temp = roi_vals.notna() & temp_series.notna()
                n_eff_roi_temp = int(m_roi_temp.sum())
                r_roi_temp, p_roi_temp = np.nan, np.nan
                ci_roi_temp = (np.nan, np.nan)
                if n_eff_roi_temp >= 5:
                    r_roi_temp, p_roi_temp = stats.spearmanr(roi_vals[m_roi_temp], temp_series[m_roi_temp], nan_policy="omit")
                    if bootstrap_ci > 0:
                        ci_roi_temp = _bootstrap_corr_ci(roi_vals[m_roi_temp], temp_series[m_roi_temp], method2_code, n_boot=bootstrap_ci, rng=rng)

                stats_roi_temp = _fetch_roi_stats(temp_stats, roi, band)
                if stats_roi_temp is not None:
                    n_eff_roi_temp = int(stats_roi_temp.get("n", n_eff_roi_temp))
                    r_roi_temp = _safe_float(stats_roi_temp.get("r", r_roi_temp))
                    p_roi_temp = _safe_float(stats_roi_temp.get("p", p_roi_temp))
                
                generate_correlation_scatter(
                    x_data=roi_vals, y_data=temp_series,
                    x_label="log10(power/baseline [-5–0 s])", y_label="Temperature (°C)",
                    title_prefix=f"{band_title} power vs temperature — {roi} (plateau)",
                    band_color=band_color,
                    output_path=roi_plots_dir / f"scatter_pow_{sanitize_label(band)}_vs_temp_plateau",
                    method_code=method2_code, Z_covars=Z_df_temp, covar_names=covar_names_temp,
                    bootstrap_ci=0, rng=rng, roi_channels=chs, logger=logger,
                    annotated_stats=(r_roi_temp, p_roi_temp, n_eff_roi_temp),
                    annot_ci=ci_roi_temp,
                    config=config
                )
                
                if temporal_df is not None:
                    for time_label in ["early", "mid", "late"]:
                        temporal_band_cols = [c for c in temporal_df.columns if c.startswith(f"pow_{band}_") and c.endswith(f"_{time_label}")]
                        if not temporal_band_cols:
                            continue
                        temporal_vals = temporal_df[temporal_band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                        
                        m_roi_temp_temporal = temporal_vals.notna() & temp_series.notna()
                        n_eff_roi_temp_temporal = int(m_roi_temp_temporal.sum())
                        r_roi_temp_temporal, p_roi_temp_temporal = np.nan, np.nan
                        ci_roi_temp_temporal = (np.nan, np.nan)
                        if n_eff_roi_temp_temporal >= 5:
                            r_roi_temp_temporal, p_roi_temp_temporal = stats.spearmanr(temporal_vals[m_roi_temp_temporal], temp_series[m_roi_temp_temporal], nan_policy="omit")
                            if bootstrap_ci > 0:
                                ci_roi_temp_temporal = _bootstrap_corr_ci(temporal_vals[m_roi_temp_temporal], temp_series[m_roi_temp_temporal], method2_code, n_boot=bootstrap_ci, rng=rng)
                        
                        generate_correlation_scatter(
                            x_data=temporal_vals, y_data=temp_series,
                            x_label=f"log10(power/baseline) [{time_label} window]", y_label="Temperature (°C)",
                            title_prefix=f"{band_title} power vs temperature — {roi} ({time_label})",
                            band_color=band_color,
                            output_path=roi_plots_dir / f"scatter_pow_{sanitize_label(band)}_vs_temp_{time_label}",
                            method_code=method2_code, Z_covars=Z_df_temp, covar_names=covar_names_temp,
                            bootstrap_ci=0, rng=rng, roi_channels=chs, logger=logger,
                            annotated_stats=(r_roi_temp_temporal, p_roi_temp_temporal, n_eff_roi_temp_temporal),
                            annot_ci=ci_roi_temp_temporal,
                            config=config
                        )
                
                plot_regression_residual_diagnostics(
                    x_data=roi_vals, y_data=temp_series,
                    title_prefix=f"{band_title} power vs temperature — {roi}",
                    output_path=roi_plots_dir / f"residual_diagnostics_pow_{sanitize_label(band)}_vs_temp",
                    band_color=band_color, logger=logger, config=config
                )
                
                if Z_df_temp is not None and not Z_df_temp.empty:
                    n_len_pt2 = min(len(roi_vals), len(temp_series), len(Z_df_temp))
                    x_part2, y_part2 = roi_vals.iloc[:n_len_pt2], temp_series.iloc[:n_len_pt2]
                    Z_part2 = Z_df_temp.iloc[:n_len_pt2]
                    x2_res_sr, y2_res_sr, n2_res = _compute_partial_residuals(
                        x_part2, y_part2, Z_part2, method2_code, logger=logger, context=f"Partial residuals {roi} temp {band}"
                    )
                    if n2_res >= 5:
                        r_resid_roi2 = np.nan
                        p_resid_roi2 = np.nan
                        n_partial_roi2 = n2_res
                        if stats_roi_temp is not None:
                            r_resid_roi2 = _safe_float(stats_roi_temp.get("r_partial", r_resid_roi2))
                            p_resid_roi2 = _safe_float(stats_roi_temp.get("p_partial", p_resid_roi2))
                            n_partial_roi2 = int(stats_roi_temp.get("n_partial", n_partial_roi2))
                        if not np.isfinite(r_resid_roi2) or not np.isfinite(p_resid_roi2):
                            r_resid_roi2, p_resid_roi2 = stats.spearmanr(x2_res_sr, y2_res_sr, nan_policy="omit")
                        ci_resid_roi2 = (np.nan, np.nan)
                        if bootstrap_ci > 0:
                            ci_resid_roi2 = _bootstrap_corr_ci(x2_res_sr, y2_res_sr, method2_code, n_boot=bootstrap_ci, rng=rng)
                        residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)"
                        residual_ylabel = "Partial residuals (ranked) of temperature (°C)"
                        generate_correlation_scatter(
                            x_data=x2_res_sr, y_data=y2_res_sr,
                            x_label=residual_xlabel, y_label=residual_ylabel,
                            title_prefix=f"Partial residuals — {band_title} vs temperature — {roi}",
                            band_color=band_color,
                            output_path=roi_plots_dir / f"scatter_pow_{sanitize_label(band)}_vs_temp_partial",
                            method_code=method2_code, bootstrap_ci=0, rng=rng,
                            is_partial_residuals=True, roi_channels=chs, logger=logger,
                            annotated_stats=(r_resid_roi2, p_resid_roi2, n_partial_roi2),
                            annot_ci=ci_resid_roi2,
                            config=config
                    )
            

###################################################################
# Group Plots
###################################################################

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
    if config is None:
        config = load_settings()
    rng = rng or np.random.default_rng(42)

    deriv_root = Path(config.deriv_root)
    plots_dir = deriv_group_plots_path(deriv_root, subdir=PLOT_SUBDIR)
    stats_dir = deriv_group_stats_path(deriv_root)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_group_logger("behavior_analysis", log_name, config=config)

    def _get_attr(obj, name, default):
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    rating_x = _get_attr(scatter_inputs, "rating_x", {})
    rating_y = _get_attr(scatter_inputs, "rating_y", {})
    rating_Z = _get_attr(scatter_inputs, "rating_Z", {})
    rating_hasZ = _get_attr(scatter_inputs, "rating_hasZ", {})
    rating_subjects = _get_attr(scatter_inputs, "rating_subjects", {})

    temp_x = _get_attr(scatter_inputs, "temp_x", {})
    temp_y = _get_attr(scatter_inputs, "temp_y", {})
    temp_Z = _get_attr(scatter_inputs, "temp_Z", {})
    temp_hasZ = _get_attr(scatter_inputs, "temp_hasZ", {})
    temp_subjects = _get_attr(scatter_inputs, "temp_subjects", {})
    have_temp = bool(_get_attr(scatter_inputs, "have_temp", False)) and do_temp

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

    allowed_pool = {"pooled_trials", "within_subject_centered", "fisher_by_subject"}
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

    rating_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []
    freq_bands_cfg = config.get("time_frequency_analysis.bands", {})

    for (band, roi), x_lists in rating_x.items():
        y_lists = rating_y.get((band, roi))
        if not y_lists:
            continue

        Z_lists = rating_Z.get((band, roi), [])
        has_Z_flags = rating_hasZ.get((band, roi), [])
        subj_order = rating_subjects.get((band, roi), [])

        vis_x: List[pd.Series] = []
        vis_y: List[pd.Series] = []
        vis_subj_ids: List[str] = []

        for idx, (x_arr, y_arr) in enumerate(zip(x_lists, y_lists)):
            xi = pd.Series(np.asarray(x_arr))
            yi = pd.Series(np.asarray(y_arr))
            n = min(len(xi), len(yi))
            xi = xi.iloc[:n]
            yi = yi.iloc[:n]
            mask = xi.notna() & yi.notna()
            xi = xi[mask]
            yi = yi[mask]
            if xi.empty or yi.empty:
                continue
            if pooling_strategy == "within_subject_centered":
                xi = xi - xi.mean()
                yi = yi - yi.mean()
            elif pooling_strategy == "within_subject_zscored":
                sx = xi.std(ddof=1)
                sy = yi.std(ddof=1)
                if sx <= 0 or sy <= 0:
                    continue
                xi = (xi - xi.mean()) / sx
                yi = (yi - yi.mean()) / sy
            subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
            vis_subj_ids.extend([subj_id] * len(xi))
            vis_x.append(xi.reset_index(drop=True))
            vis_y.append(yi.reset_index(drop=True))

        if not vis_x:
            continue

        x_all = pd.concat(vis_x, ignore_index=True)
        y_all = pd.concat(vis_y, ignore_index=True)

        Z_all_vis = None
        x_all_partial = None
        y_all_partial = None
        partial_subj_ids: List[str] = []

        if any(has_Z_flags):
            partial_x: List[pd.Series] = []
            partial_y: List[pd.Series] = []
            partial_Z: List[pd.DataFrame] = []
            for idx, (has_cov, Z_df, x_arr, y_arr) in enumerate(zip(has_Z_flags, Z_lists, x_lists, y_lists)):
                if not has_cov or Z_df is None:
                    continue
                xi = pd.Series(np.asarray(x_arr))
                yi = pd.Series(np.asarray(y_arr))
                Zi = Z_df.copy()
                n = min(len(xi), len(yi), len(Zi))
                xi = xi.iloc[:n]
                yi = yi.iloc[:n]
                Zi = Zi.iloc[:n].copy()
                mask = xi.notna() & yi.notna()
                xi = xi[mask]
                yi = yi[mask]
                Zi = Zi.loc[mask]
                if xi.empty or yi.empty:
                    continue
                if pooling_strategy == "within_subject_centered":
                    xi = xi - xi.mean()
                    yi = yi - yi.mean()
                elif pooling_strategy == "within_subject_zscored":
                    sx = xi.std(ddof=1)
                    sy = yi.std(ddof=1)
                    if sx <= 0 or sy <= 0:
                        continue
                    xi = (xi - xi.mean()) / sx
                    yi = (yi - yi.mean()) / sy
                partial_x.append(xi.reset_index(drop=True))
                partial_y.append(yi.reset_index(drop=True))
                partial_Z.append(Zi.reset_index(drop=True))
                subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
                partial_subj_ids.extend([subj_id] * len(xi))

            if partial_Z:
                common_cols = set(partial_Z[0].columns)
                for df in partial_Z[1:]:
                    common_cols &= set(df.columns)
                common_cols = sorted(common_cols)
                if common_cols:
                    partial_Z = [df[common_cols] for df in partial_Z]
                Z_all_vis = pd.concat(partial_Z, ignore_index=True)
                x_all_partial = pd.concat(partial_x, ignore_index=True)
                y_all_partial = pd.concat(partial_y, ignore_index=True)
                if subject_fixed_effects and "__subject_id__" not in Z_all_vis.columns:
                    dummies = pd.get_dummies(
                        pd.Series(partial_subj_ids, name="__subject_id__").astype(str),
                        prefix="sub",
                        drop_first=True,
                    )
                    Z_all_vis = pd.concat([Z_all_vis.reset_index(drop=True), dummies], axis=1)

        if Z_all_vis is None and subject_fixed_effects and len(vis_subj_ids) == len(x_all):
            dummies = pd.get_dummies(pd.Series(vis_subj_ids, name="__subject_id__").astype(str), prefix="sub", drop_first=True)
            Z_all_vis = dummies
            x_all_partial = x_all
            y_all_partial = y_all

        method_code = "spearman"
        r_g, p_g, n_trials, n_subj, ci95, p_pooled = _compute_group_corr_stats(
            [np.asarray(v) for v in x_lists],
            [np.asarray(v) for v in y_lists],
            method_code,
            strategy=pooling_strategy,
            n_cluster_boot=cluster_bootstrap,
            rng=rng,
        )

        tag = tag_map.get(pooling_strategy)
        freq_range = freq_bands_cfg.get(band)
        if freq_range:
            freq_range = tuple(freq_range)
        band_title = f"{band.capitalize()} ({freq_range[0]:g}–{freq_range[1]:g} Hz)" if freq_range else band.capitalize()
        title_roi = "Overall" if roi == "All" else roi
        out_dir = plots_dir / ("overall" if roi == "All" else Path("roi_scatters") / sanitize_label(roi))
        ensure_dir(out_dir)
        base_name = "scatter_pow_overall" if roi == "All" else "scatter_pow"
        out_path = out_dir / f"{base_name}_{sanitize_label(band)}_vs_rating"

        generate_correlation_scatter(
            x_data=x_all,
            y_data=y_all,
            x_label="log10(power/baseline [-5–0 s])",
            y_label="Rating" if pooling_strategy == "pooled_trials" else ("Rating (centered)" if pooling_strategy == "within_subject_centered" else "Rating (z-scored)"),
            title_prefix=f"{band_title} power vs rating — {title_roi}",
            band_color=_get_band_color(band),
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

        rating_records.append({
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
        })

        if Z_all_vis is not None and x_all_partial is not None and y_all_partial is not None and len(x_all_partial) >= 5:
            x_res, y_res, n_res = _partial_residuals_xy_given_Z(x_all_partial, y_all_partial, Z_all_vis, method_code)
            if n_res >= 5:
                r_resid, p_resid = stats.spearmanr(x_res, y_res, nan_policy="omit")
                ci_resid = (np.nan, np.nan)
                if bootstrap_ci > 0:
                    ci_resid = _bootstrap_corr_ci(x_res, y_res, method_code, n_boot=bootstrap_ci, rng=rng)
                generate_correlation_scatter(
                    x_data=x_res,
                    y_data=y_res,
                    x_label="Partial residuals (ranked) of log10(power/baseline)",
                    y_label="Partial residuals (ranked) of rating",
                    title_prefix=f"Partial residuals — {band_title} vs rating — {title_roi}",
                    band_color=_get_band_color(band),
                    output_path=out_dir / f"{base_name}_{sanitize_label(band)}_vs_rating_partial",
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

    if rating_records:
        df = pd.DataFrame(rating_records)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        rej, crit = fdr_bh_reject(df["p_group"].to_numpy(), alpha=fdr_alpha)
        df["fdr_reject"] = rej
        df["fdr_crit_p"] = crit
        out_stats = stats_dir / "group_pooled_corr_pow_roi_vs_rating.tsv"
        df.to_csv(out_stats, sep="	", index=False)
        logger.info("Wrote pooled ROI vs rating stats: %s", out_stats)

    if not have_temp:
        return

    for (band, roi), x_lists in temp_x.items():
        y_lists = temp_y.get((band, roi))
        if not y_lists:
            continue

        Z_lists = temp_Z.get((band, roi), [])
        has_Z_flags = temp_hasZ.get((band, roi), [])
        subj_order = temp_subjects.get((band, roi), [])

        vis_x: List[pd.Series] = []
        vis_y: List[pd.Series] = []
        vis_subj_ids: List[str] = []

        for idx, (x_arr, y_arr) in enumerate(zip(x_lists, y_lists)):
            xi = pd.Series(np.asarray(x_arr))
            yi = pd.Series(np.asarray(y_arr))
            n = min(len(xi), len(yi))
            xi = xi.iloc[:n]
            yi = yi.iloc[:n]
            mask = xi.notna() & yi.notna()
            xi = xi[mask]
            yi = yi[mask]
            if xi.empty or yi.empty:
                continue
            if pooling_strategy == "within_subject_centered":
                xi = xi - xi.mean()
                yi = yi - yi.mean()
            elif pooling_strategy == "within_subject_zscored":
                sx = xi.std(ddof=1)
                sy = yi.std(ddof=1)
                if sx <= 0 or sy <= 0:
                    continue
                xi = (xi - xi.mean()) / sx
                yi = (yi - yi.mean()) / sy
            subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
            vis_subj_ids.extend([subj_id] * len(xi))
            vis_x.append(xi.reset_index(drop=True))
            vis_y.append(yi.reset_index(drop=True))

        if not vis_x:
            continue

        x_all = pd.concat(vis_x, ignore_index=True)
        y_all = pd.concat(vis_y, ignore_index=True)

        Z_all_vis = None
        x_all_partial = None
        y_all_partial = None
        partial_subj_ids: List[str] = []

        if any(has_Z_flags):
            partial_x: List[pd.Series] = []
            partial_y: List[pd.Series] = []
            partial_Z: List[pd.DataFrame] = []
            for idx, (has_cov, Z_df, x_arr, y_arr) in enumerate(zip(has_Z_flags, Z_lists, x_lists, y_lists)):
                if not has_cov or Z_df is None:
                    continue
                xi = pd.Series(np.asarray(x_arr))
                yi = pd.Series(np.asarray(y_arr))
                Zi = Z_df.copy()
                n = min(len(xi), len(yi), len(Zi))
                xi = xi.iloc[:n]
                yi = yi.iloc[:n]
                Zi = Zi.iloc[:n].copy()
                mask = xi.notna() & yi.notna()
                xi = xi[mask]
                yi = yi[mask]
                Zi = Zi.loc[mask]
                if xi.empty or yi.empty:
                    continue
                if pooling_strategy == "within_subject_centered":
                    xi = xi - xi.mean()
                    yi = yi - yi.mean()
                elif pooling_strategy == "within_subject_zscored":
                    sx = xi.std(ddof=1)
                    sy = yi.std(ddof=1)
                    if sx <= 0 or sy <= 0:
                        continue
                    xi = (xi - xi.mean()) / sx
                    yi = (yi - yi.mean()) / sy
                partial_x.append(xi.reset_index(drop=True))
                partial_y.append(yi.reset_index(drop=True))
                partial_Z.append(Zi.reset_index(drop=True))
                subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
                partial_subj_ids.extend([subj_id] * len(xi))

            if partial_Z:
                common_cols = set(partial_Z[0].columns)
                for df in partial_Z[1:]:
                    common_cols &= set(df.columns)
                common_cols = sorted(common_cols)
                if common_cols:
                    partial_Z = [df[common_cols] for df in partial_Z]
                Z_all_vis = pd.concat(partial_Z, ignore_index=True)
                x_all_partial = pd.concat(partial_x, ignore_index=True)
                y_all_partial = pd.concat(partial_y, ignore_index=True)
                if subject_fixed_effects and "__subject_id__" not in Z_all_vis.columns:
                    dummies = pd.get_dummies(
                        pd.Series(partial_subj_ids, name="__subject_id__").astype(str),
                        prefix="sub",
                        drop_first=True,
                    )
                    Z_all_vis = pd.concat([Z_all_vis.reset_index(drop=True), dummies], axis=1)

        if Z_all_vis is None and subject_fixed_effects and len(vis_subj_ids) == len(x_all):
            dummies = pd.get_dummies(pd.Series(vis_subj_ids, name="__subject_id__").astype(str), prefix="sub", drop_first=True)
            Z_all_vis = dummies
            x_all_partial = x_all
            y_all_partial = y_all

        method_code = "spearman"
        r_g, p_g, n_trials, n_subj, ci95, p_pooled = _compute_group_corr_stats(
            [np.asarray(v) for v in x_lists],
            [np.asarray(v) for v in y_lists],
            method_code,
            strategy=pooling_strategy,
            n_cluster_boot=cluster_bootstrap,
            rng=rng,
        )

        tag = tag_map.get(pooling_strategy)
        freq_range = freq_bands_cfg.get(band)
        if freq_range:
            freq_range = tuple(freq_range)
        band_title = f"{band.capitalize()} ({freq_range[0]:g}–{freq_range[1]:g} Hz)" if freq_range else band.capitalize()
        title_roi = "Overall" if roi == "All" else roi
        out_dir = plots_dir / ("overall" if roi == "All" else Path("roi_scatters") / sanitize_label(roi))
        ensure_dir(out_dir)
        base_name = "scatter_pow_overall" if roi == "All" else "scatter_pow"
        out_path = out_dir / f"{base_name}_{sanitize_label(band)}_vs_temp"

        generate_correlation_scatter(
            x_data=x_all,
            y_data=y_all,
            x_label="log10(power/baseline [-5–0 s])",
            y_label="Temperature (°C)" if pooling_strategy == "pooled_trials" else ("Temperature (°C, centered)" if pooling_strategy == "within_subject_centered" else "Temperature (z-scored)"),
            title_prefix=f"{band_title} power vs temperature — {title_roi}",
            band_color=_get_band_color(band),
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

        temp_records.append({
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
        })

        if Z_all_vis is not None and x_all_partial is not None and y_all_partial is not None and len(x_all_partial) >= 5:
            x_res, y_res, n_res = _partial_residuals_xy_given_Z(x_all_partial, y_all_partial, Z_all_vis, method_code)
            if n_res >= 5:
                r_resid, p_resid = stats.spearmanr(x_res, y_res, nan_policy="omit")
                ci_resid = (np.nan, np.nan)
                if bootstrap_ci > 0:
                    ci_resid = _bootstrap_corr_ci(x_res, y_res, method_code, n_boot=bootstrap_ci, rng=rng)
                generate_correlation_scatter(
                    x_data=x_res,
                    y_data=y_res,
                    x_label="Partial residuals (ranked) of log10(power/baseline)",
                    y_label="Partial residuals (ranked) of temperature",
                    title_prefix=f"Partial residuals — {band_title} vs temperature — {title_roi}",
                    band_color=_get_band_color(band),
                    output_path=out_dir / f"{base_name}_{sanitize_label(band)}_vs_temp_partial",
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

    if temp_records:
        df_t = pd.DataFrame(temp_records)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        rej_t, crit_t = fdr_bh_reject(df_t["p_group"].to_numpy(), alpha=fdr_alpha)
        df_t["fdr_reject"] = rej_t
        df_t["fdr_crit_p"] = crit_t
        out_stats_t = stats_dir / "group_pooled_corr_pow_roi_vs_temp.tsv"
        df_t.to_csv(out_stats_t, sep="	", index=False)
        logger.info("Wrote pooled ROI vs temperature stats: %s", out_stats_t)

def plot_power_behavior_correlation(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger, config=None):
    if config is None:
        config = load_settings()
    if y is None or len(y) == 0 or y.isna().all():
        logger.warning("No valid behavioral data for correlation plots")
        return
        
    n_bands = len(bands)
    n_cols = 2
    n_rows = (n_bands + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_bands == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, band in enumerate(bands):
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if not band_cols:
            continue
            
        band_power_avg = pow_df[band_cols].mean(axis=1)
        valid_mask = ~(band_power_avg.isna() | y.isna())
        if valid_mask.sum() < 5:
            logger.info("No valid data after filtering; skipping plot.")
            continue
            
        x_valid, y_valid = band_power_avg[valid_mask], y[valid_mask]
        
        axes[i].scatter(x_valid, y_valid, alpha=0.6, s=30, color=_get_band_color(band, config=config))
        
        z = np.polyfit(x_valid, y_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
        axes[i].plot(x_line, p(x_line), 'r--', alpha=0.8)
        
        axes[i].set_xlabel(f'{band.capitalize()} Power\nlog10(power/baseline)')
        axes[i].set_ylabel('Behavioral Rating')
        axes[i].set_title(f'{band.capitalize()} Power vs Behavior')
        axes[i].grid(True, alpha=0.3)
        
        rho, p_spear = stats.spearmanr(x_valid, y_valid)
        axes[i].text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p_spear:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(fig, save_dir / f'power_behavior_correlation_{subject}', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
    plt.close(fig)


def plot_power_behavior_correlation_matrix(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger, config=None):
    if config is None:
        config = load_settings()
    for band in bands:
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if not band_cols:
            continue
        
        correlations, p_values, channel_names = [], [], []
        
        for col in band_cols:
            valid_mask = ~(pow_df[col].isna() | y.isna())
            if valid_mask.sum() > 5:
                r, p = stats.spearmanr(pow_df[col][valid_mask], y[valid_mask])
                correlations.append(r)
                p_values.append(p)
            else:
                correlations.append(0)
                p_values.append(1.0)
            channel_names.append(col.replace(f'pow_{band}_', ''))
        
        if correlations:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bars = ax.bar(range(len(correlations)), correlations, 
                         color=['red' if p < 0.05 else 'lightblue' for p in p_values])
            
            ax.set_xlabel('Channel', fontweight='bold')
            ax.set_ylabel('Spearman ρ', fontweight='bold')
            ax.set_title(f'{band.upper()} Band - Channel-wise Correlations with Behavior\nSubject {subject}', 
                       fontweight='bold', fontsize=14)
            ax.set_xticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Moderate correlation')
            ax.axhline(y=-0.3, color='green', linestyle='--', alpha=0.7)
            
            significant_count = sum(1 for p in p_values if p < 0.05)
            ax.text(0.02, 0.98, f'Significant channels: {significant_count}/{len(correlations)}', 
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_formats = config.get("output.save_formats", ["svg"])
            save_fig(fig, save_dir / f'sub-{subject}_power_behavior_correlation_{band}', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
            plt.close(fig)
    
    logger.info(f"Saved separate power-behavior correlation plots for {len(bands)} bands in: {save_dir}")


def plot_significant_correlations_topomap(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], info: mne.Info, subject: str, save_dir: Path, logger: logging.Logger, config=None, alpha: float = 0.05):
    if config is None:
        config = load_settings()
    bands_with_data = []
    
    for band in bands:
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if not band_cols:
            continue
            
        ch_names = [col.replace(f'pow_{band}_', '') for col in band_cols]
        correlations, p_values = [], []
        
        for col in band_cols:
            valid_data = pd.concat([pow_df[col], y], axis=1).dropna()
            if len(valid_data) >= 5:
                r, p = stats.spearmanr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
            else:
                r, p = np.nan, 1.0
            correlations.append(r)
            p_values.append(p)
        
        sig_mask = np.array(p_values) < alpha
        bands_with_data.append({
            'band': band,
            'channels': ch_names,
            'correlations': np.array(correlations),
            'p_values': np.array(p_values),
            'significant_mask': sig_mask
        })
    
    if not bands_with_data:
        logger.warning("No significant correlations found across any frequency band")
        return
    
    n_bands = len(bands_with_data)
    fig, axes = plt.subplots(1, n_bands, figsize=(4.8 * n_bands, 4.8))
    if n_bands == 1:
        axes = [axes]
    plt.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.20, wspace=0.08)
    
    all_sig_corrs = []
    for band_data in bands_with_data:
        sig_corrs = band_data['correlations'][band_data['significant_mask']]
        all_sig_corrs.extend(sig_corrs[np.isfinite(sig_corrs)])
    if all_sig_corrs:
        vmax = max(abs(np.min(all_sig_corrs)), abs(np.max(all_sig_corrs)))
    else:
        all_corrs = []
        for band_data in bands_with_data:
            all_corrs.extend(band_data['correlations'][np.isfinite(band_data['correlations'])])
        vmax = max(abs(np.min(all_corrs)), abs(np.max(all_corrs))) if all_corrs else 0.5
    
    successful_plots = []
    
    for i, band_data in enumerate(bands_with_data):
        ax = axes[i]
        
        n_info_chs = len(info['ch_names'])
        topo_data = np.zeros(n_info_chs)
        topo_mask = np.zeros(n_info_chs, dtype=bool)
        
        for j, info_ch in enumerate(info['ch_names']):
            if info_ch in band_data['channels']:
                ch_idx = band_data['channels'].index(info_ch)
                topo_data[j] = band_data['correlations'][ch_idx] if np.isfinite(band_data['correlations'][ch_idx]) else 0
                topo_mask[j] = band_data['significant_mask'][ch_idx]
        
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
        if len(picks) == 0:
            continue
        
        im, _ = mne.viz.plot_topomap(
            topo_data[picks],
            mne.pick_info(info, picks),
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(-vmax, vmax),
            contours=6,
            mask=topo_mask[picks],
            mask_params=dict(
                marker='o', 
                markerfacecolor='white', 
                markeredgecolor='black', 
                linewidth=1, 
                markersize=6
            )
        )
        
        successful_plots.append(im)
        
        n_sig = topo_mask[picks].sum()
        n_total = len([ch for ch in band_data['channels'] if ch in info['ch_names']])
        ax.set_title(
            f'{band_data["band"].upper()}\n{n_sig}/{n_total} significant',
            fontweight='bold', fontsize=12, pad=10
        )
    
    plt.suptitle(
        f'Significant EEG-Pain Correlations (p < {alpha})\nSubject {subject}',
        fontweight='bold', fontsize=14, y=1.02
    )
    
    if successful_plots:
        left = min(ax.get_position().x0 for ax in axes)
        right = max(ax.get_position().x1 for ax in axes)
        bottom = min(ax.get_position().y0 for ax in axes)
        span = right - left
        cb_width = 0.55 * span
        cb_left = left + 0.225 * span
        cb_bottom = max(0.04, bottom - 0.06)
        cax = fig.add_axes([cb_left, cb_bottom, cb_width, 0.028])
        cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation='horizontal')
        cbar.set_label('Spearman correlation (ρ)', fontweight='bold', fontsize=11)
        cbar.ax.tick_params(pad=2, labelsize=9)
    
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(fig, save_dir / f'sub-{subject}_significant_correlations_topomap', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
    plt.close(fig)
    
    logger.info(f"Created topomaps for {len(bands_with_data)} frequency bands: {[bd['band'] for bd in bands_with_data]}")


def plot_behavioral_response_patterns(y: pd.Series, aligned_events: Optional[pd.DataFrame], subject: str, save_dir: Path, logger: logging.Logger, config=None):
    if config is None:
        config = load_settings()
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(y.dropna(), bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Pain Rating')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Rating Distribution')
    ax1.grid(True, alpha=0.3)
    
    mean_rating = y.mean()
    std_rating = y.std()
    ax1.text(0.02, 0.98, f'Mean: {mean_rating:.2f} ± {std_rating:.2f}', 
            transform=ax1.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(fig1, save_dir / f'sub-{subject}_rating_distribution', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
    plt.close(fig1)


def plot_power_spectrogram_with_behavior(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger, config=None):
    if config is None:
        config = load_settings()
    band_powers = {}
    for band in bands:
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if band_cols:
            band_powers[band] = pow_df[band_cols].mean(axis=1)
    
    if not band_powers:
        logger.warning("No band power data found for spectrogram")
        return
    
    n_trials = len(y)
    trial_indices = np.arange(n_trials)
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
    
    for band in bands:
        if band not in band_powers:
            continue
            
        power = band_powers[band]
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(trial_indices, power, 'k-', alpha=0.3, linewidth=1, label='EEG Power')
        
        scatter = ax.scatter(trial_indices, power, c=y_norm, cmap='RdYlBu_r', 
                           s=40, alpha=0.8, edgecolor='black', linewidth=0.5,
                           label='Power (colored by rating)')
        
        if len(power) > 10:
            power_smooth = interpolate.interp1d(trial_indices, power, kind='cubic')(trial_indices)
            ax.plot(trial_indices, power_smooth, 'b-', alpha=0.6, linewidth=2, label='Power trend')
        
        ax.set_ylabel(f'{band.capitalize()} Power (log)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Trial Number', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{band.capitalize()} Band Power Dynamics with Behavioral Ratings\nSubject {subject}', 
                    fontweight='bold', fontsize=14)
        
        valid_mask = ~(power.isna() | y.isna())
        if valid_mask.sum() > 5:
            r, p = stats.spearmanr(power[valid_mask], y[valid_mask])
            ax.text(0.02, 0.95, f'ρ = {r:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Pain Rating (normalized)', fontweight='bold')
        
        y_min, y_max = y.min(), y.max()
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f'{y_min:.1f}', f'{(y_min+y_max)/2:.1f}', f'{y_max:.1f}'])
        
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, save_dir / f'sub-{subject}_power_spectrogram_behavior_{band}', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)
    
    logger.info(f"Saved EEG power spectrograms with behavior for {len(bands)} bands in: {save_dir}")


def plot_power_spectrogram_temperature_band(pow_df: pd.DataFrame, aligned_events: pd.DataFrame, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger, config=None):
    if config is None:
        config = load_settings()
    psych_temp_columns = config.get("event_columns.temperature", [])
    temp_col = _pick_first_column(aligned_events, psych_temp_columns) if aligned_events is not None else None
    if temp_col is None or aligned_events is None:
        logger.warning("No temperature data found for spectrogram")
        return
        
    temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    valid_mask = ~temp_series.isna()
    if valid_mask.sum() < 5:
        logger.warning("Insufficient valid temperature data for spectrogram")
        return
        
    pow_df_filtered = pow_df[valid_mask]
    temp_filtered = temp_series[valid_mask]
    
    band_powers = {}
    for band in bands:
        band_cols = [col for col in pow_df_filtered.columns if col.startswith(f'pow_{band}_')]
        if band_cols:
            band_powers[band] = pow_df_filtered[band_cols].mean(axis=1)
    
    if not band_powers:
        logger.warning("No band power data found for temperature spectrogram")
        return
    
    n_trials = len(temp_filtered)
    trial_indices = np.arange(n_trials)
    temp_norm = (temp_filtered - temp_filtered.min()) / (temp_filtered.max() - temp_filtered.min()) if temp_filtered.max() > temp_filtered.min() else np.zeros_like(temp_filtered)
    
    for band in bands:
        if band not in band_powers:
            continue
            
        power = band_powers[band]
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(trial_indices, power, 'k-', alpha=0.3, linewidth=1, label='EEG Power')
        
        scatter = ax.scatter(trial_indices, power, c=temp_norm, cmap='coolwarm', 
                           s=40, alpha=0.8, edgecolor='black', linewidth=0.5,
                           label='Power (colored by temperature)')
        
        if len(power) > 10:
            power_smooth = interpolate.interp1d(trial_indices, power, kind='cubic')(trial_indices)
            ax.plot(trial_indices, power_smooth, 'b-', alpha=0.6, linewidth=2, label='Power trend')
        
        ax.set_ylabel(f'{band.capitalize()} Power (log)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Trial Number', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{band.capitalize()} Band Power Dynamics with Temperature\nSubject {subject}', 
                    fontweight='bold', fontsize=14)
        
        valid_corr_mask = ~(power.isna() | temp_filtered.isna())
        if valid_corr_mask.sum() > 5:
            r, p = stats.spearmanr(power[valid_corr_mask], temp_filtered[valid_corr_mask])
            ax.text(0.02, 0.95, f'ρ = {r:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Temperature (°C)', fontweight='bold')
        
        temp_min, temp_max = temp_filtered.min(), temp_filtered.max()
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f'{temp_min:.1f}', f'{(temp_min+temp_max)/2:.1f}', f'{temp_max:.1f}'])
        
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, save_dir / f'sub-{subject}_power_spectrogram_temperature_{band}', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)
    
    logger.info(f"Saved EEG power spectrograms with temperature for {len(bands)} bands in: {save_dir}")


def plot_behavior_modulated_connectivity(subject: str, task: str, y: pd.Series, save_dir: Path, logger: logging.Logger, config=None):
    if config is None:
        config = load_settings()
    deriv_root = Path(config.deriv_root)
    conn_path = find_connectivity_features_path(deriv_root, subject)
    if not conn_path.exists():
        logger.warning(f"No connectivity data found for {subject}")
        return
    
    if conn_path.suffix == '.parquet':
        conn_df = pd.read_parquet(conn_path)
    elif conn_path.suffix == '.tsv':
        conn_df = pd.read_csv(conn_path, sep='\t')
    else:
        logger.warning(f"Unsupported connectivity file format: {conn_path.suffix}")
        return
    
    conn_measures = ['coh', 'plv', 'pli', 'wpli']
    available_measures = [m for m in conn_measures if any(m in col for col in conn_df.columns)]
    
    if not available_measures:
        logger.warning("No connectivity measures found")
        return
    
    measure = 'coh' if 'coh' in available_measures else available_measures[0]
    bands = ['alpha', 'beta', 'gamma']
    
    for band in bands:
        measure_cols = [col for col in conn_df.columns if f'{measure}_{band}' in col]
        if not measure_cols:
            continue
        
        correlations, connections = [], []
        
        for col in measure_cols:
            valid_mask = ~(conn_df[col].isna() | y.isna())
            if valid_mask.sum() > 5:
                r, p = stats.spearmanr(conn_df[col][valid_mask], y[valid_mask])
                if abs(r) > 0.3 and p < 0.05:
                    correlations.append(r)
                    pair = col.replace(f'{measure}_{band}_', '').replace('conn_', '')
                    connections.append(pair)
        
        if len(connections) < 3:
            continue
        
        # Create network visualization
        import networkx as nx
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges
        edge_weights = []
        for i, (conn, corr) in enumerate(zip(connections, correlations)):
            if '-' in conn:
                ch1, ch2 = conn.split('-')
                G.add_edge(ch1, ch2, weight=abs(corr), correlation=corr)
                edge_weights.append(abs(corr))
        
        if G.number_of_nodes() == 0:
            plt.close(fig)
            continue
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color='lightblue', alpha=0.7, ax=ax)
        
        # Draw edges colored by correlation strength
        edges = G.edges()
        weights = [G[u][v]['correlation'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights,
                             edge_cmap=plt.cm.RdBu_r, edge_vmin=-max(abs(w) for w in weights),
                             edge_vmax=max(abs(w) for w in weights), width=2, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, 
                                 norm=plt.Normalize(vmin=-max(abs(w) for w in weights),
                                                  vmax=max(abs(w) for w in weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Correlation with Behavior', fontweight='bold')
        
        ax.set_title(f'Behavior-Modulated {measure.upper()} Connectivity\n{band.capitalize()} Band - Subject {subject}',
                    fontweight='bold', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, save_dir / f'sub-{subject}_connectivity_network_{measure}_{band}', formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)
    
    logger.info(f"Saved behavior-modulated connectivity networks")


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
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    
    candidate_files = [
        ("rating", stats_dir / "corr_stats_pow_combined_vs_rating.tsv"),
        ("temp", stats_dir / "corr_stats_pow_combined_vs_temp.tsv"),
        ("temperature", stats_dir / "corr_stats_pow_combined_vs_temperature.tsv"),
    ]
    frames = []
    for target_label, path in candidate_files:
        if path.exists():
            _df = pd.read_csv(path, sep="\t")
            _df["target"] = target_label
            frames.append(_df)
    if not frames:
        logger.warning(
            "No combined correlation stats found for rating or temperature. Expected one of: "
            + ", ".join(str(p) for _, p in candidate_files)
        )
        return
    
    df = pd.concat(frames, axis=0, ignore_index=True)
    
    df_sig = df[
        (df['p'] <= alpha) & 
        df['r'].notna() & 
        df['p'].notna() &
        df['channel'].notna() &
        df['band'].notna()
    ].copy()
    
    if len(df_sig) == 0:
        logger.warning(f"No significant correlations found (p <= {alpha})")
        return
    
    df_sig['abs_r'] = df_sig['r'].abs()
    df_top = df_sig.nlargest(top_n, 'abs_r')
    
    if len(df_top) == 0:
        logger.warning("No top correlations to plot")
        return
    
    if 'target' in df_top.columns:
        df_top['predictor'] = df_top['channel'] + ' (' + df_top['band'] + ') [' + df_top['target'].astype(str) + ']'
    else:
        df_top['predictor'] = df_top['channel'] + ' (' + df_top['band'] + ')'
    
    df_top = df_top.sort_values('abs_r', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    band_colors = {}
    try:
        from eeg_pipeline.utils.config_loader import get_config
        config_temp = get_config()
        config_colors = config_temp.get("visualization.band_colors", {})
    except Exception:
        config_colors = {}
    
    for band in df_top['band'].unique():
        if band in config_colors:
            band_colors[band] = config_colors[band]
        else:
            band_colors[band] = _get_band_color(band, config_temp if 'config_temp' in locals() else None)
    
    colors = [band_colors[band] for band in df_top['band']]
    
    y_pos = np.arange(len(df_top))
    bars = ax.barh(y_pos, df_top['abs_r'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['predictor'], fontsize=11)
    ax.set_xlabel(f"|Spearman ρ| with Behavior (p < {alpha})", fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} Significant Behavioral Predictors', fontweight='bold', fontsize=14, pad=20)
    
    for i, (_, row) in enumerate(df_top.iterrows()):
        r_val, p_val, abs_r_val = row['r'], row['p'], row['abs_r']
        sign_str = '(+)' if r_val >= 0 else '(-)'
        x_pos = abs_r_val + 0.01
        ax.text(x_pos, i, f'{abs_r_val:.3f} {sign_str} (p={p_val:.3f})', 
               va='center', ha='left', fontsize=10, fontweight='normal')
    
    max_r = df_top['abs_r'].max()
    ax.set_xlim(0, max_r * 1.25)
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
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
    
    top_predictors_file = stats_dir / f"top_{top_n}_behavioral_predictors.tsv"
    export_cols = ['predictor', 'channel', 'band', 'r', 'abs_r', 'p', 'n']
    if 'target' in df_top.columns and 'target' not in export_cols:
        export_cols = ['target'] + export_cols
    df_top_export = df_top[export_cols].copy()
    df_top_export = df_top_export.sort_values('abs_r', ascending=False)
    df_top_export.to_csv(top_predictors_file, sep="\t", index=False)
    logger.info(f"Exported top predictors data: {top_predictors_file}")


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
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)

    if data_path is None:
        data_path = stats_dir / f"time_frequency_correlation_data{roi_suffix}{method_suffix}.npz"

    if not data_path.exists():
        logger.error(f"Precomputed TF correlation data not found: {data_path}")
        return

    logger.info(f"Loading TF correlation data from {data_path}")
    with np.load(data_path, allow_pickle=True) as data:
        correlations = data["correlations"]
        p_values = data["p_values"]
        p_corrected = data.get("p_corrected", np.full_like(correlations, np.nan))
        significant_mask = data.get("significant_mask", np.zeros_like(correlations, dtype=bool))
        freqs = data["freqs"]
        time_bin_centers = data["time_bin_centers"]
        time_bin_edges = data.get("time_bin_edges")
        n_valid = data.get("n_valid", np.zeros_like(correlations, dtype=int))
        freq_range = tuple(data.get("freq_range", (float(freqs[0]), float(freqs[-1]))))
        baseline_applied = bool(data.get("baseline_applied", False))
        baseline_window_raw = data.get("baseline_window", np.array([]))
        time_resolution = float(data.get("time_resolution", 0.1))
        alpha = float(data.get("alpha", config.get("behavior_analysis.statistics.fdr_alpha", 0.05)))
        tf_method = str(data.get("method", "spearman"))

    if time_bin_edges is None or len(time_bin_edges) != len(time_bin_centers) + 1:
        half_step = time_resolution / 2.0
        time_bin_edges = np.concatenate(( [time_bin_centers[0] - half_step], time_bin_centers + half_step ))

    if baseline_window_raw.size == 2:
        baseline_window_used: Optional[Tuple[float, float]] = (
            float(baseline_window_raw[0]),
            float(baseline_window_raw[1]),
        )
    else:
        baseline_window_used = None

    n_freqs, n_time_bins = correlations.shape
    if n_freqs == 0 or n_time_bins == 0:
        logger.warning("TF correlation data is empty; nothing to plot")
        return

    correlation_vmin = viz_config.get("correlation_vmin", -0.6)
    correlation_vmax = viz_config.get("correlation_vmax", 0.6)

    extent = [
        float(time_bin_edges[0]),
        float(time_bin_edges[-1]),
        float(freqs[0]),
        float(freqs[-1]),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        correlations,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=correlation_vmin,
        vmax=correlation_vmax,
    )
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontweight="bold")

    method_name = "Spearman" if tf_method.lower() == "spearman" else "Pearson"
    roi_name = roi_selection or "All Channels"
    metric = "log10(power/baseline)" if baseline_applied else "raw power"
    bl_txt = ""
    if baseline_applied and baseline_window_used is not None:
        bl_txt = f" | BL: [{baseline_window_used[0]:.2f}, {baseline_window_used[1]:.2f}] s"
    title_text = (
        "Time-Frequency Power-Behavior Correlations\n"
        f"Subject: {subject} | Method: {method_name} | ROI: {roi_name} | Metric: {metric}{bl_txt}"
    )
    ax.set_title(title_text, fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    cbar = plt.colorbar(im, ax=ax, label="Correlation (r)")
    cbar.ax.tick_params(labelsize=12)

    freq_bands = config.get("time_frequency_analysis.bands", {})
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
            ax.axhline(fmin, color="white", linestyle="-", alpha=0.3, linewidth=0.5)
            ax.axhline(fmax, color="white", linestyle="-", alpha=0.3, linewidth=0.5)

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
            summary_df = pd.read_csv(stats_file, sep="	")
        except Exception as exc:
            logger.warning(f"Failed to read TF correlation summary ({exc})")

    if summary_df is None or summary_df.empty:
        summary_df = pd.DataFrame(
            {
                "frequency": np.repeat(freqs, n_time_bins),
                "time": np.tile(time_bin_centers, n_freqs),
                "correlation": correlations.flatten(),
                "p_corrected": p_corrected.flatten(),
                "significant": significant_mask.flatten(),
                "n_valid": n_valid.flatten(),
            }
        )

    if not summary_df.empty:
        n_significant = int(np.nansum(summary_df.get("significant", False)))
        max_r = float(np.nanmax(np.abs(summary_df["correlation"])))
        max_idx = np.nanargmax(np.abs(summary_df["correlation"]))
        best_row = summary_df.iloc[max_idx]
        logger.info("Time-frequency correlation summary:")
        logger.info("  - Total TF points: %d", len(summary_df))
        logger.info("  - Significant correlations (FDR < %.3f): %d", alpha, n_significant)
        logger.info(
            "  - Strongest correlation: r=%.3f at %.1f Hz, %.2f s",
            best_row["correlation"],
            best_row["frequency"],
            best_row["time"],
        )
    else:
        logger.warning("No valid TF correlations available for summary")

def plot_power_behavior_evolution_across_trials(subject: str, task: Optional[str] = None, window_size: int = 20, bands_to_plot: Optional[List[str]] = None) -> None:
    config = load_settings()
    if task is None:
        task = config.task
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Creating power-behavior evolution analysis for sub-{subject}")
    
    deriv_root = Path(config.deriv_root)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    ensure_dir(plots_dir)
    
    _temporal_df, pow_df, _, y, info = _load_features_and_targets(subject, task, deriv_root, config)
    y = pd.to_numeric(y, errors="coerce")
    
    if bands_to_plot is None:
        power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        bands_to_plot = power_bands_to_use[:3]
    
    roi_map = _build_rois(info, config=config)
    key_rois = ['Frontal', 'Central', 'Parietal', 'Occipital']
    available_rois = [roi for roi in key_rois if roi in roi_map]
    
    if not available_rois:
        logger.warning(f"No key ROIs available for evolution analysis: sub-{subject}")
        return
        
    n_trials = len(y)
    if n_trials < window_size * 2:
        logger.warning(f"Not enough trials ({n_trials}) for evolution analysis (need at least {window_size * 2})")
        return
        
    trial_numbers = np.arange(1, n_trials + 1)
    
    fig_behav, ax_behav = plt.subplots(1, 1, figsize=(10, 6))
    
    y_clean = y.fillna(y.mean())
    smoothing_sigma = config.get("behavior_analysis.visualization.smoothing_sigma", 2.0)
    y_smooth = gaussian_filter1d(y_clean, sigma=smoothing_sigma)
    
    ax_behav.scatter(trial_numbers, y, alpha=0.4, s=15, color='gray', label='Raw ratings')
    ax_behav.plot(trial_numbers, y_smooth, color='red', linewidth=3, 
                 label='Smoothed trend', alpha=0.8)
    ax_behav.set_xlabel('Trial Number', fontweight='bold')
    ax_behav.set_ylabel('Pain Rating', fontweight='bold')
    ax_behav.set_title(f'Behavioral Response Evolution - sub-{subject}', fontweight='bold', fontsize=12)
    ax_behav.legend(fontsize=10)
    ax_behav.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(fig_behav, plots_dir / f"behavioral_response_evolution", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
    plt.close(fig_behav)
    
    for band in bands_to_plot:
        try:
            from eeg_pipeline.utils.config_loader import get_config
            config_temp = get_config()
            config_colors = config_temp.get("visualization.band_colors", {})
            band_color = config_colors.get(band, _get_band_color(band, config_temp))
        except Exception:
            band_color = _get_band_color(band)
        freq_bands = config.get("time_frequency_analysis.bands", {})
        band_range = freq_bands.get(band)
        if band_range:
            band_range = tuple(band_range)
        band_label = f"{band.title()} ({band_range[0]:g}–{band_range[1]:g} Hz)" if band_range else band.title()
        
        roi_power = {}
        for roi in available_rois:
            roi_cols = [f"pow_{band}_{ch}" for ch in roi_map[roi] 
                      if f"pow_{band}_{ch}" in pow_df.columns]
            if roi_cols:
                roi_power[roi] = pow_df[roi_cols].apply(
                    pd.to_numeric, errors='coerce').mean(axis=1)
        
        if not roi_power:
            logger.warning(f"No power data available for {band} band")
            continue
        
        fig, (ax_corr, ax_power) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Compute rolling correlations
        mid_points = []
        correlations = {roi: [] for roi in roi_power.keys()}
        
        for start in range(n_trials - window_size + 1):
            end = start + window_size
            mid_points.append(start + window_size // 2 + 1)
            y_window = y.iloc[start:end]
            
            for roi in roi_power.keys():
                x_window = roi_power[roi].iloc[start:end]
                mask = x_window.notna() & y_window.notna()
                r = stats.spearmanr(x_window[mask], y_window[mask])[0] if mask.sum() >= 5 else np.nan
                correlations[roi].append(r)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, roi in enumerate(roi_power.keys()):
            color = colors[i % len(colors)]
            ax_corr.plot(mid_points, correlations[roi], label=roi, linewidth=2.5, alpha=0.8, color=color)
            smoothing_sigma = config.get("behavior_analysis.visualization.smoothing_sigma", 2.0)
            power_smooth = gaussian_filter1d(roi_power[roi].fillna(roi_power[roi].mean()), sigma=smoothing_sigma)
            ax_power.plot(trial_numbers, power_smooth, label=roi, linewidth=2.5, alpha=0.8, color=color)
        
        ax_corr.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_corr.set_ylabel('Running Correlation (r)', fontweight='bold')
        ax_corr.set_title(f'{band_label}: Power-Behavior Correlation Evolution', fontweight='bold', fontsize=12)
        ax_corr.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax_corr.grid(True, alpha=0.3)
        ax_corr.set_ylim(-1, 1)
        
        ax_power.set_xlabel('Trial Number', fontweight='bold')
        ax_power.set_ylabel('log10(power/baseline)', fontweight='bold')
        ax_power.set_title(f'{band_label} Power Evolution', fontweight='bold', fontsize=12)
        ax_power.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax_power.grid(True, alpha=0.3)
        
        plt.suptitle(f'Power Evolution: {band_label} - sub-{subject}', fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, plots_dir / f"power_evolution_{band}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)
    
    logger.info(f"Saved behavioral evolution figure and power evolution analysis for {len(bands_to_plot)} bands for sub-{subject}")
