"""
Dose-Response Relationship Visualizations
==========================================

Scientific Question: Is there a monotonic relationship between
stimulus intensity and neural response?

Plots for each feature type (power, connectivity, aperiodic, microstates):
1. Dose-response curves (temperature vs feature)
2. Nonlinearity test (linear vs polynomial fit)
3. Threshold detection (inflection points, derivatives)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.infra.paths import ensure_dir, deriv_plots_path, deriv_features_path
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.plotting.core.utils import get_band_colors, get_significance_colors
from eeg_pipeline.utils.data.features import (
    get_aperiodic_columns,
    get_connectivity_columns_by_band,
    get_itpc_columns_by_band,
    get_microstate_columns,
    get_power_columns_by_band,
)
from eeg_pipeline.utils.data.manipulation import find_column
from eeg_pipeline.utils.data import load_subject_scatter_data


# =============================================================================
# Config Helpers
# =============================================================================


def _get_bands_from_config(config) -> list:
    """Get frequency bands from config.
    
    Raises ValueError if config is None or frequency bands are not configured.
    """
    if config is None:
        raise ValueError("Config is required for dose-response visualization")
    bands = config.get("power.bands_to_use", None)
    if bands is None:
        bands = list(config.get("frequency_bands", {}).keys())
    if not bands:
        raise ValueError(
            "Frequency bands not configured. Set 'power.bands_to_use' or "
            "'frequency_bands' in eeg_config.yaml"
        )
    return bands


# =============================================================================
# Helper Functions
# =============================================================================





def _load_additional_features(deriv_root: Path, subject: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Load additional feature files (aperiodic, microstates, ITPC)."""
    features_dir = deriv_features_path(deriv_root, subject)
    additional = {}
    
    feature_files = {
        "aperiodic": "features_aperiodic.tsv",
        "microstates": "features_microstates.tsv",
        "itpc": "features_itpc.tsv",
    }
    
    for name, filename in feature_files.items():
        path = features_dir / filename
        if path.exists():
            try:
                df = read_tsv(path)
                if df is not None and not df.empty:
                    additional[name] = df
                    logger.debug(f"Loaded {name} features: {len(df)} trials, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"Failed to load {name} features: {e}")
    
    return additional


# =============================================================================
# Main Entry Point
# =============================================================================


def visualize_dose_response(
    subject: str,
    deriv_root: Path,
    task: str,
    config,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Create dose-response relationship visualizations.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "0001")
    deriv_root : Path
        Path to derivatives root directory
    task : str
        Task name
    config : Config
        Pipeline configuration object
    logger : Logger
        Logger instance
        
    Returns
    -------
    Dict[str, Path]
        Mapping of plot names to saved file paths
    """
    saved_files: Dict[str, Path] = {}
    
    # Use config from behavioral plotting if available to get subplot dir
    from eeg_pipeline.plotting.config import get_plot_config
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    output_dir = plots_dir / "dose_response"
    ensure_dir(output_dir)
    
    # Load data using shared loader
    # Returns 9-tuple: temporal_df, plateau_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map, conn_df
    _, pow_df, _, _, temp_series, _, _, _, conn_df = load_subject_scatter_data(
        subject, task, deriv_root, config, logger
    )
    
    if pow_df is None or pow_df.empty:
        logger.debug("No power features found for dose-response analysis")
        return saved_files
        
    if temp_series is None or temp_series.empty:
        logger.debug("No temperature data found for dose-response analysis")
        return saved_files

    temp_col = "temperature"
    
    # Load additional feature files
    additional_features = _load_additional_features(deriv_root, subject, logger)
    
    # =================================================================
    # Power dose-response
    # =================================================================
    bands = _get_bands_from_config(config)
    band_colors = get_band_colors()
    
    df_power = pow_df.copy()
    df_power[temp_col] = temp_series.values
    power_cols = get_power_columns_by_band(df_power, bands=bands)
    
    if power_cols:
        power_dir = output_dir / "power"
        ensure_dir(power_dir)
        _plot_dose_response_curves(
            df_power, temp_col, power_cols, subject, power_dir, saved_files, logger,
            feature_type="Power", y_label="Mean Power (a.u.)", bands=bands, band_colors=band_colors
        )
        _plot_nonlinearity_test(
            df_power, temp_col, power_cols, subject, power_dir, saved_files, logger,
            feature_type="power", bands=bands, band_colors=band_colors
        )
        _plot_threshold_detection(
            df_power, temp_col, power_cols, subject, power_dir, saved_files, logger,
            feature_type="power", bands=bands, band_colors=band_colors
        )
    
    # =================================================================
    # Connectivity dose-response
    # =================================================================
    if conn_df is not None and not conn_df.empty:
        df_conn = conn_df.copy()
        df_conn[temp_col] = temp_series.values
        conn_cols = get_connectivity_columns_by_band(df_conn, bands=bands)
        
        if conn_cols:
            conn_dir = output_dir / "connectivity"
            ensure_dir(conn_dir)
            _plot_dose_response_curves(
                df_conn, temp_col, conn_cols, subject, conn_dir, saved_files, logger,
                feature_type="Connectivity", y_label="Mean wPLI", bands=bands, band_colors=band_colors
            )
            _plot_nonlinearity_test(
                df_conn, temp_col, conn_cols, subject, conn_dir, saved_files, logger,
                feature_type="connectivity", bands=bands, band_colors=band_colors
            )
            _plot_threshold_detection(
                df_conn, temp_col, conn_cols, subject, conn_dir, saved_files, logger,
                feature_type="connectivity", bands=bands, band_colors=band_colors
            )
    
    # =================================================================
    # Aperiodic dose-response
    # =================================================================
    if "aperiodic" in additional_features:
        df_aper = additional_features["aperiodic"].copy()
        if len(df_aper) == len(temp_series):
            df_aper[temp_col] = temp_series.values
            aper_cols = get_aperiodic_columns(df_aper)
            
            if aper_cols:
                aper_dir = output_dir / "aperiodic"
                ensure_dir(aper_dir)
                _plot_aperiodic_dose_response(
                    df_aper, temp_col, aper_cols, subject, aper_dir, saved_files, logger
                )
    
    # =================================================================
    # Microstate dose-response
    # =================================================================
    if "microstates" in additional_features:
        df_ms = additional_features["microstates"].copy()
        if len(df_ms) == len(temp_series):
            df_ms[temp_col] = temp_series.values
            ms_cols = get_microstate_columns(df_ms)
            
            if ms_cols:
                ms_dir = output_dir / "microstates"
                ensure_dir(ms_dir)
                _plot_microstate_dose_response(
                    df_ms, temp_col, ms_cols, subject, ms_dir, saved_files, logger
                )
    
    # =================================================================
    # ITPC dose-response
    # =================================================================
    if "itpc" in additional_features:
        df_itpc = additional_features["itpc"].copy()
        if len(df_itpc) == len(temp_series):
            df_itpc[temp_col] = temp_series.values
            itpc_cols = get_itpc_columns_by_band(df_itpc, bands=bands)
            
            if itpc_cols:
                itpc_dir = output_dir / "itpc"
                ensure_dir(itpc_dir)
                _plot_dose_response_curves(
                    df_itpc, temp_col, itpc_cols, subject, itpc_dir, saved_files, logger,
                    feature_type="ITPC", y_label="Mean ITPC", bands=bands, band_colors=band_colors
                )
                _plot_nonlinearity_test(
                    df_itpc, temp_col, itpc_cols, subject, itpc_dir, saved_files, logger,
                    feature_type="itpc", bands=bands, band_colors=band_colors
                )
    
    logger.info(f"Created {len(saved_files)} dose-response plots")
    return saved_files


###################################################################
# Registry adapters
###################################################################
# =============================================================================
# Plot Functions
# =============================================================================


def _plot_dose_response_curves(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
    feature_type: str = "Power",
    y_label: str = "Mean Power (a.u.)",
    bands: Optional[List[str]] = None,
    band_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Plot dose-response curves for each frequency band."""
    if bands is None:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
    if band_colors is None:
        band_colors = get_band_colors()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    temps = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temps.dropna().unique())
    
    if len(unique_temps) < 2:
        plt.close(fig)
        logger.debug("Insufficient temperature levels for dose-response")
        return
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        
        if band not in feature_cols:
            ax.text(0.5, 0.5, f"No {band} data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{band.title()} Band")
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        
        temp_means = []
        temp_sems = []
        for t in unique_temps:
            mask = temps == t
            if mask.sum() > 0:
                temp_means.append(band_values[mask].mean())
                temp_sems.append(band_values[mask].std() / np.sqrt(mask.sum()))
            else:
                temp_means.append(np.nan)
                temp_sems.append(np.nan)
        
        temp_means = np.array(temp_means)
        temp_sems = np.array(temp_sems)
        
        ax.errorbar(
            unique_temps, temp_means, yerr=temp_sems,
            fmt='o-', color=band_colors.get(band, "#333333"), capsize=4, capthick=1.5,
            markersize=8, linewidth=2, label="Observed"
        )
        
        valid = ~np.isnan(temp_means)
        if valid.sum() >= 3:
            x_valid = unique_temps[valid]
            y_valid = temp_means[valid]
            
            slope, intercept, r_lin, p_lin, _ = stats.linregress(x_valid, y_valid)
            x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
            y_lin = slope * x_fit + intercept
            
            coeffs = np.polyfit(x_valid, y_valid, 2)
            y_quad = np.polyval(coeffs, x_fit)
            r_quad = np.corrcoef(y_valid, np.polyval(coeffs, x_valid))[0, 1]
            
            ax.plot(x_fit, y_lin, '--', color='gray', alpha=0.7, linewidth=1.5,
                   label=f"Linear (r²={r_lin**2:.2f})")
            ax.plot(x_fit, y_quad, ':', color='darkgray', alpha=0.7, linewidth=1.5,
                   label=f"Quadratic (r²={r_quad**2:.2f})")
        
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(f"{band.title()} Band", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    
    ax = axes[5]
    ax.axis("off")
    
    summary_text = (
        f"Dose-Response Summary\n"
        f"{'─' * 25}\n\n"
        f"Question: Is there a monotonic\n"
        f"relationship between stimulus\n"
        f"intensity and neural response?\n\n"
        f"Temperature range: {unique_temps.min():.1f}–{unique_temps.max():.1f}°C\n"
        f"N temperatures: {len(unique_temps)}\n"
        f"N trials: {len(df)}"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.suptitle(f"Dose-Response Curves: Temperature vs {feature_type} (sub-{subject})",
                fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    feature_key = feature_type.lower().replace(" ", "_")
    path = output_dir / f"sub-{subject}_{feature_key}_dose_response_curves.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files[f"{feature_key}_dose_response_curves"] = path
    logger.info(f"Created {feature_type} dose-response curves plot")


def _plot_nonlinearity_test(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
    feature_type: str = "power",
    bands: Optional[List[str]] = None,
    band_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Test for nonlinear dose-response relationships."""
    if bands is None:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
    if band_colors is None:
        band_colors = get_band_colors()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    temps = pd.to_numeric(df[temp_col], errors="coerce")
    results = []
    
    for band in bands:
        if band not in feature_cols:
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        valid = ~(temps.isna() | band_values.isna())
        
        if valid.sum() < 10:
            continue
        
        x = temps[valid].values
        y = band_values[valid].values
        
        slope, intercept, r_lin, p_lin, _ = stats.linregress(x, y)
        ss_res_lin = np.sum((y - (slope * x + intercept)) ** 2)
        
        coeffs = np.polyfit(x, y, 2)
        y_pred_quad = np.polyval(coeffs, x)
        ss_res_quad = np.sum((y - y_pred_quad) ** 2)
        r_quad = np.corrcoef(y, y_pred_quad)[0, 1]
        
        df_lin = len(x) - 2
        df_quad = len(x) - 3
        if df_quad > 0 and ss_res_quad > 0:
            f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / df_quad)
            p_improvement = 1 - stats.f.cdf(f_stat, 1, df_quad)
        else:
            f_stat = np.nan
            p_improvement = np.nan
        
        results.append({
            "band": band,
            "r2_linear": r_lin ** 2,
            "r2_quadratic": r_quad ** 2,
            "improvement": r_quad ** 2 - r_lin ** 2,
            "f_stat": f_stat,
            "p_improvement": p_improvement,
        })
    
    if results:
        results_df = pd.DataFrame(results)
        x_pos = np.arange(len(results_df))
        width = 0.35
        
        ax.bar(x_pos - width/2, results_df["r2_linear"], width,
               label="Linear R²", color="#4C72B0", alpha=0.8)
        ax.bar(x_pos + width/2, results_df["r2_quadratic"], width,
               label="Quadratic R²", color="#55A868", alpha=0.8)
        
        for i, row in results_df.iterrows():
            if pd.notna(row["p_improvement"]) and row["p_improvement"] < 0.05:
                ax.annotate("*", (i + width/2, row["r2_quadratic"] + 0.02),
                           ha="center", fontsize=14, fontweight="bold")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([r["band"].title() for r in results], fontsize=11)
        ax.set_ylabel("Variance Explained (R²)", fontsize=11)
        ax.set_xlabel("Frequency Band", fontsize=11)
        ax.set_title(f"{feature_type.title()} Nonlinearity Test (sub-{subject})",
                    fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        ax.set_ylim(0, max(results_df["r2_quadratic"].max() * 1.2, 0.1))
        
        ax.text(0.02, 0.98, "* p < 0.05 for quadratic improvement",
               transform=ax.transAxes, fontsize=9, verticalalignment="top", style="italic")
    else:
        ax.text(0.5, 0.5, "Insufficient data for nonlinearity test",
               ha="center", va="center", transform=ax.transAxes)
    
    plt.tight_layout()
    path = output_dir / f"sub-{subject}_{feature_type}_nonlinearity_test.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files[f"{feature_type}_nonlinearity_test"] = path
    logger.info(f"Created {feature_type} nonlinearity test plot")


def _plot_threshold_detection(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
    feature_type: str = "power",
    bands: Optional[List[str]] = None,
    band_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Detect response thresholds via derivative and normalized response analysis."""
    if bands is None:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
    if band_colors is None:
        band_colors = get_band_colors()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    temps = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temps.dropna().unique())
    
    if len(unique_temps) < 3:
        plt.close(fig)
        return
    
    ax1 = axes[0]
    for band in bands:
        if band not in feature_cols:
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        
        temp_means = []
        for t in unique_temps:
            mask = temps == t
            if mask.sum() > 0:
                temp_means.append(band_values[mask].mean())
            else:
                temp_means.append(np.nan)
        
        temp_means = np.array(temp_means)
        
        if len(unique_temps) > 1:
            derivative = np.gradient(temp_means, unique_temps)
            ax1.plot(unique_temps, derivative, 'o-', color=band_colors.get(band, "#333333"),
                    label=band.title(), markersize=6, linewidth=1.5)
    
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Temperature (°C)", fontsize=11)
    ax1.set_ylabel(f"Rate of Change (d{feature_type.title()}/dTemp)", fontsize=11)
    ax1.set_title("Derivative Analysis: Where Does Response Accelerate?",
                 fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for band in bands:
        if band not in feature_cols:
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        
        temp_means = []
        for t in unique_temps:
            mask = temps == t
            if mask.sum() > 0:
                temp_means.append(band_values[mask].mean())
            else:
                temp_means.append(np.nan)
        
        temp_means = np.array(temp_means)
        valid = ~np.isnan(temp_means)
        
        if valid.sum() > 0:
            y_norm = temp_means[valid]
            y_range = y_norm.max() - y_norm.min()
            if y_range > 0:
                y_norm = (y_norm - y_norm.min()) / y_range
            else:
                y_norm = np.zeros_like(y_norm)
            ax2.plot(unique_temps[valid], y_norm, 'o-', color=band_colors.get(band, "#333333"),
                    label=band.title(), markersize=6, linewidth=1.5)
    
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(unique_temps.max(), 0.52, "50% threshold", fontsize=9, ha="right")
    
    ax2.set_xlabel("Temperature (°C)", fontsize=11)
    ax2.set_ylabel("Normalized Response (0-1)", fontsize=11)
    ax2.set_title("Threshold Detection: Normalized Response Curves",
                 fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    fig.suptitle(f"{feature_type.title()} Threshold Analysis (sub-{subject})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = output_dir / f"sub-{subject}_{feature_type}_threshold_detection.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files[f"{feature_type}_threshold_detection"] = path
    logger.info(f"Created {feature_type} threshold detection plot")


# =============================================================================
# Aperiodic and Microstate Specific Plots
# =============================================================================


def _plot_aperiodic_dose_response(
    df: pd.DataFrame,
    temp_col: str,
    aper_cols: Dict[str, List[str]],
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """Plot dose-response curves for aperiodic parameters (slope, offset)."""
    n_metrics = len(aper_cols)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(5 * min(n_metrics, 3), 5))
    if n_metrics == 1:
        axes = [axes]
    
    temps = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temps.dropna().unique())
    
    if len(unique_temps) < 2:
        plt.close(fig)
        return
    
    metric_colors = {"slope": "#E24A33", "offset": "#348ABD", "exponent": "#988ED5"}
    
    for idx, (metric, cols) in enumerate(aper_cols.items()):
        if idx >= 3:
            break
        ax = axes[idx]
        
        metric_values = df[cols].mean(axis=1)
        
        temp_means = []
        temp_sems = []
        for t in unique_temps:
            mask = temps == t
            if mask.sum() > 0:
                temp_means.append(metric_values[mask].mean())
                temp_sems.append(metric_values[mask].std() / np.sqrt(mask.sum()))
            else:
                temp_means.append(np.nan)
                temp_sems.append(np.nan)
        
        temp_means = np.array(temp_means)
        temp_sems = np.array(temp_sems)
        
        color = metric_colors.get(metric, "#333333")
        ax.errorbar(
            unique_temps, temp_means, yerr=temp_sems,
            fmt='o-', color=color, capsize=4, capthick=1.5,
            markersize=8, linewidth=2
        )
        
        valid = ~np.isnan(temp_means)
        if valid.sum() >= 3:
            x_valid = unique_temps[valid]
            y_valid = temp_means[valid]
            slope, intercept, r_val, p_val, _ = stats.linregress(x_valid, y_valid)
            x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, linewidth=1.5)
            ax.text(0.05, 0.95, f"r = {r_val:.3f}\np = {p_val:.3f}",
                   transform=ax.transAxes, fontsize=9, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.set_ylabel(f"Mean {metric.title()}", fontsize=10)
        ax.set_title(f"Aperiodic {metric.title()}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Aperiodic Dose-Response (sub-{subject})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = output_dir / f"sub-{subject}_aperiodic_dose_response.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files["aperiodic_dose_response"] = path
    logger.info("Created aperiodic dose-response plot")


def _plot_microstate_dose_response(
    df: pd.DataFrame,
    temp_col: str,
    ms_cols: Dict[str, List[str]],
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """Plot dose-response curves for microstate metrics (coverage, duration)."""
    n_metrics = len(ms_cols)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, min(n_metrics, 4), figsize=(4 * min(n_metrics, 4), 5))
    if n_metrics == 1:
        axes = [axes]
    
    temps = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temps.dropna().unique())
    
    if len(unique_temps) < 2:
        plt.close(fig)
        return
    
    metric_colors = {"coverage": "#55A868", "duration": "#C44E52", "occurrence": "#8172B2", "gev": "#CCB974"}
    
    for idx, (metric, cols) in enumerate(ms_cols.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        
        metric_values = df[cols].mean(axis=1)
        
        temp_means = []
        temp_sems = []
        for t in unique_temps:
            mask = temps == t
            if mask.sum() > 0:
                temp_means.append(metric_values[mask].mean())
                temp_sems.append(metric_values[mask].std() / np.sqrt(mask.sum()))
            else:
                temp_means.append(np.nan)
                temp_sems.append(np.nan)
        
        temp_means = np.array(temp_means)
        temp_sems = np.array(temp_sems)
        
        color = metric_colors.get(metric, "#333333")
        ax.errorbar(
            unique_temps, temp_means, yerr=temp_sems,
            fmt='o-', color=color, capsize=4, capthick=1.5,
            markersize=8, linewidth=2
        )
        
        valid = ~np.isnan(temp_means)
        if valid.sum() >= 3:
            x_valid = unique_temps[valid]
            y_valid = temp_means[valid]
            slope, intercept, r_val, p_val, _ = stats.linregress(x_valid, y_valid)
            x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, linewidth=1.5)
            ax.text(0.05, 0.95, f"r = {r_val:.3f}\np = {p_val:.3f}",
                   transform=ax.transAxes, fontsize=9, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.set_ylabel(f"Mean {metric.title()}", fontsize=10)
        ax.set_title(f"Microstate {metric.title()}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Microstate Dose-Response (sub-{subject})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = output_dir / f"sub-{subject}_microstate_dose_response.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files["microstate_dose_response"] = path
    logger.info("Created microstate dose-response plot")
