"""
Dose-Response Relationship Visualizations
==========================================

Scientific Question: Is there a monotonic relationship between
stimulus intensity and neural response?

Plots for each feature type (power, connectivity, aperiodic, ITPC):
1. Dose-response curves (temperature vs feature)
2. Nonlinearity test (linear vs polynomial fit)
3. Threshold detection (inflection points, derivatives)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.infra.paths import ensure_dir, deriv_plots_path, deriv_features_path
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.plotting.core.colors import get_band_colors
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.data.features import (
    get_aperiodic_columns,
    get_connectivity_columns_by_band,
    get_itpc_columns_by_band,
    get_power_columns_by_band,
)
from eeg_pipeline.utils.data import load_subject_scatter_data


# Constants
MIN_TEMPERATURE_LEVELS_FOR_CURVES = 2
MIN_TEMPERATURE_LEVELS_FOR_THRESHOLD = 3
MIN_DATA_POINTS_FOR_FIT = 3
MIN_DATA_POINTS_FOR_NONLINEARITY = 10
POLYNOMIAL_DEGREE = 2
FIT_POINTS = 100
SIGNIFICANCE_THRESHOLD = 0.05
MAX_APERIODIC_METRICS = 3
NORMALIZED_THRESHOLD = 0.5

TEMPERATURE_COLUMN = "temperature"
DEFAULT_COLOR = "#333333"


class TemperatureAggregation(NamedTuple):
    """Aggregated feature values by temperature level."""
    unique_temperatures: np.ndarray
    means: np.ndarray
    standard_errors: np.ndarray


class PlotConfig(NamedTuple):
    """Configuration for plotting dose-response relationships."""
    feature_type: str
    y_label: str
    bands: List[str]
    band_colors: Dict[str, str]


def _get_bands_from_config(config) -> List[str]:
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


def _load_additional_features(
    deriv_root: Path, subject: str, logger: logging.Logger
) -> Dict[str, pd.DataFrame]:
    """Load additional feature files (aperiodic, ITPC)."""
    features_dir = deriv_features_path(deriv_root, subject)
    additional = {}
    
    feature_files = {
        "aperiodic": "features_aperiodic.tsv",
        "itpc": "features_itpc.tsv",
    }
    
    for name, filename in feature_files.items():
        path = features_dir / filename
        if path.exists():
            try:
                df = read_tsv(path)
                if df is not None and not df.empty:
                    additional[name] = df
                    logger.debug(
                        f"Loaded {name} features: {len(df)} trials, "
                        f"{len(df.columns)} columns"
                    )
            except Exception as e:
                logger.warning(f"Failed to load {name} features: {e}")
    
    return additional


def _aggregate_by_temperature(
    temperatures: pd.Series, feature_values: pd.Series
) -> TemperatureAggregation:
    """Aggregate feature values by temperature level.
    
    Returns means and standard errors for each unique temperature.
    """
    unique_temps = np.sort(temperatures.dropna().unique())
    means = []
    sems = []
    
    for temp in unique_temps:
        mask = temperatures == temp
        n_samples = mask.sum()
        
        if n_samples > 0:
            subset = feature_values[mask]
            means.append(subset.mean())
            sems.append(subset.std() / np.sqrt(n_samples))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    
    return TemperatureAggregation(
        unique_temperatures=np.array(unique_temps),
        means=np.array(means),
        standard_errors=np.array(sems),
    )


def _compute_linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute linear regression fit statistics."""
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
    }


def _compute_quadratic_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute quadratic polynomial fit."""
    coefficients = np.polyfit(x, y, POLYNOMIAL_DEGREE)
    y_predicted = np.polyval(coefficients, x)
    r_squared = np.corrcoef(y, y_predicted)[0, 1] ** 2
    return {
        "coefficients": coefficients,
        "r_squared": r_squared,
        "y_predicted": y_predicted,
    }


def _compute_nonlinearity_statistics(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    """Compare linear vs quadratic fits using F-test."""
    linear_fit = _compute_linear_fit(x, y)
    quadratic_fit = _compute_quadratic_fit(x, y)
    
    y_linear_pred = linear_fit["slope"] * x + linear_fit["intercept"]
    ss_residual_linear = np.sum((y - y_linear_pred) ** 2)
    ss_residual_quadratic = np.sum((y - quadratic_fit["y_predicted"]) ** 2)
    
    degrees_freedom_linear = len(x) - 2
    degrees_freedom_quadratic = len(x) - 3
    
    if degrees_freedom_quadratic > 0 and ss_residual_quadratic > 0:
        f_statistic = (
            (ss_residual_linear - ss_residual_quadratic) / 1
        ) / (ss_residual_quadratic / degrees_freedom_quadratic)
        p_improvement = 1 - stats.f.cdf(f_statistic, 1, degrees_freedom_quadratic)
    else:
        f_statistic = np.nan
        p_improvement = np.nan
    
    return {
        "r2_linear": linear_fit["r_squared"],
        "r2_quadratic": quadratic_fit["r_squared"],
        "improvement": quadratic_fit["r_squared"] - linear_fit["r_squared"],
        "f_statistic": f_statistic,
        "p_improvement": p_improvement,
    }


def _plot_band_dose_response_curve(
    ax: plt.Axes,
    aggregation: TemperatureAggregation,
    band: str,
    band_color: str,
    y_label: str,
) -> None:
    """Plot dose-response curve for a single frequency band."""
    unique_temps = aggregation.unique_temperatures
    means = aggregation.means
    sems = aggregation.standard_errors
    
    ax.errorbar(
        unique_temps,
        means,
        yerr=sems,
        fmt="o-",
        color=band_color,
        capsize=4,
        capthick=1.5,
        markersize=8,
        linewidth=2,
        label="Observed",
    )
    
    valid_mask = ~np.isnan(means)
    n_valid = valid_mask.sum()
    
    if n_valid >= MIN_DATA_POINTS_FOR_FIT:
        x_valid = unique_temps[valid_mask]
        y_valid = means[valid_mask]
        
        linear_fit = _compute_linear_fit(x_valid, y_valid)
        quadratic_fit = _compute_quadratic_fit(x_valid, y_valid)
        
        x_fit = np.linspace(x_valid.min(), x_valid.max(), FIT_POINTS)
        y_linear = linear_fit["slope"] * x_fit + linear_fit["intercept"]
        y_quadratic = np.polyval(quadratic_fit["coefficients"], x_fit)
        
        ax.plot(
            x_fit,
            y_linear,
            "--",
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label=f"Linear (r²={linear_fit['r_squared']:.2f})",
        )
        ax.plot(
            x_fit,
            y_quadratic,
            ":",
            color="darkgray",
            alpha=0.7,
            linewidth=1.5,
            label=f"Quadratic (r²={quadratic_fit['r_squared']:.2f})",
        )
    
    ax.set_xlabel("Temperature (°C)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{band.title()} Band", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def _plot_dose_response_curves(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    plot_config: PlotConfig,
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """Plot dose-response curves for each frequency band."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    temperatures = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temperatures.dropna().unique())
    
    if len(unique_temps) < MIN_TEMPERATURE_LEVELS_FOR_CURVES:
        plt.close(fig)
        logger.debug("Insufficient temperature levels for dose-response")
        return
    
    for idx, band in enumerate(plot_config.bands):
        ax = axes[idx]
        
        if band not in feature_cols:
            ax.text(
                0.5,
                0.5,
                f"No {band} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{band.title()} Band")
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        aggregation = _aggregate_by_temperature(temperatures, band_values)
        band_color = plot_config.band_colors.get(band, DEFAULT_COLOR)
        
        _plot_band_dose_response_curve(
            ax, aggregation, band, band_color, plot_config.y_label
        )
    
    summary_ax = axes[5]
    summary_ax.axis("off")
    
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
    summary_ax.text(
        0.1,
        0.9,
        summary_text,
        transform=summary_ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    fig.suptitle(
        f"Dose-Response Curves: Temperature vs {plot_config.feature_type} "
        f"(sub-{subject})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    feature_key = plot_config.feature_type.lower().replace(" ", "_")
    path = output_dir / f"sub-{subject}_{feature_key}_dose_response_curves.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files[f"{feature_key}_dose_response_curves"] = path
    logger.info(f"Created {plot_config.feature_type} dose-response curves plot")


def _plot_nonlinearity_test(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    plot_config: PlotConfig,
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """Test for nonlinear dose-response relationships."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    temperatures = pd.to_numeric(df[temp_col], errors="coerce")
    results = []
    
    for band in plot_config.bands:
        if band not in feature_cols:
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        valid_mask = ~(temperatures.isna() | band_values.isna())
        
        if valid_mask.sum() < MIN_DATA_POINTS_FOR_NONLINEARITY:
            continue
        
        x = temperatures[valid_mask].values
        y = band_values[valid_mask].values
        
        stats_dict = _compute_nonlinearity_statistics(x, y)
        results.append({"band": band, **stats_dict})
    
    if results:
        results_df = pd.DataFrame(results)
        x_positions = np.arange(len(results_df))
        bar_width = 0.35
        
        ax.bar(
            x_positions - bar_width / 2,
            results_df["r2_linear"],
            bar_width,
            label="Linear R²",
            color="#4C72B0",
            alpha=0.8,
        )
        ax.bar(
            x_positions + bar_width / 2,
            results_df["r2_quadratic"],
            bar_width,
            label="Quadratic R²",
            color="#55A868",
            alpha=0.8,
        )
        
        for idx, row in results_df.iterrows():
            p_value = row["p_improvement"]
            if pd.notna(p_value) and p_value < SIGNIFICANCE_THRESHOLD:
                y_position = row["r2_quadratic"] + 0.02
                ax.annotate(
                    "*",
                    (idx + bar_width / 2, y_position),
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [r["band"].title() for r in results], fontsize=11
        )
        ax.set_ylabel("Variance Explained (R²)", fontsize=11)
        ax.set_xlabel("Frequency Band", fontsize=11)
        ax.set_title(
            f"{plot_config.feature_type.title()} Nonlinearity Test "
            f"(sub-{subject})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper right")
        max_r2 = results_df["r2_quadratic"].max()
        ax.set_ylim(0, max(max_r2 * 1.2, 0.1))
        
        ax.text(
            0.02,
            0.98,
            "* p < 0.05 for quadratic improvement",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            style="italic",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for nonlinearity test",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    
    plt.tight_layout()
    feature_key = plot_config.feature_type.lower()
    path = output_dir / f"sub-{subject}_{feature_key}_nonlinearity_test.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files[f"{feature_key}_nonlinearity_test"] = path
    logger.info(f"Created {plot_config.feature_type} nonlinearity test plot")


def _plot_derivative_analysis(
    ax: plt.Axes,
    temperatures: pd.Series,
    feature_cols: Dict[str, List[str]],
    df: pd.DataFrame,
    plot_config: PlotConfig,
) -> None:
    """Plot derivative analysis for threshold detection."""
    unique_temps = np.sort(temperatures.dropna().unique())
    
    for band in plot_config.bands:
        if band not in feature_cols:
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        aggregation = _aggregate_by_temperature(temperatures, band_values)
        
        if len(unique_temps) > 1:
            derivative = np.gradient(aggregation.means, unique_temps)
            band_color = plot_config.band_colors.get(band, DEFAULT_COLOR)
            ax.plot(
                unique_temps,
                derivative,
                "o-",
                color=band_color,
                label=band.title(),
                markersize=6,
                linewidth=1.5,
            )
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel(
        f"Rate of Change (d{plot_config.feature_type.title()}/dTemp)",
        fontsize=11,
    )
    ax.set_title(
        "Derivative Analysis: Where Does Response Accelerate?",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)


def _plot_normalized_response(
    ax: plt.Axes,
    temperatures: pd.Series,
    feature_cols: Dict[str, List[str]],
    df: pd.DataFrame,
    plot_config: PlotConfig,
) -> None:
    """Plot normalized response curves for threshold detection."""
    unique_temps = np.sort(temperatures.dropna().unique())
    
    for band in plot_config.bands:
        if band not in feature_cols:
            continue
        
        band_values = df[feature_cols[band]].mean(axis=1)
        aggregation = _aggregate_by_temperature(temperatures, band_values)
        
        valid_mask = ~np.isnan(aggregation.means)
        if valid_mask.sum() > 0:
            y_values = aggregation.means[valid_mask]
            y_range = y_values.max() - y_values.min()
            
            if y_range > 0:
                y_normalized = (y_values - y_values.min()) / y_range
            else:
                y_normalized = np.zeros_like(y_values)
            
            band_color = plot_config.band_colors.get(band, DEFAULT_COLOR)
            ax.plot(
                unique_temps[valid_mask],
                y_normalized,
                "o-",
                color=band_color,
                label=band.title(),
                markersize=6,
                linewidth=1.5,
            )
    
    ax.axhline(
        NORMALIZED_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.text(
        unique_temps.max(),
        NORMALIZED_THRESHOLD + 0.02,
        "50% threshold",
        fontsize=9,
        ha="right",
    )
    
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Normalized Response (0-1)", fontsize=11)
    ax.set_title(
        "Threshold Detection: Normalized Response Curves",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def _plot_threshold_detection(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    plot_config: PlotConfig,
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """Detect response thresholds via derivative and normalized response analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    temperatures = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temperatures.dropna().unique())
    
    if len(unique_temps) < MIN_TEMPERATURE_LEVELS_FOR_THRESHOLD:
        plt.close(fig)
        return
    
    _plot_derivative_analysis(axes[0], temperatures, feature_cols, df, plot_config)
    _plot_normalized_response(axes[1], temperatures, feature_cols, df, plot_config)
    
    fig.suptitle(
        f"{plot_config.feature_type.title()} Threshold Analysis (sub-{subject})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    
    feature_key = plot_config.feature_type.lower()
    path = output_dir / f"sub-{subject}_{feature_key}_threshold_detection.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files[f"{feature_key}_threshold_detection"] = path
    logger.info(f"Created {plot_config.feature_type} threshold detection plot")


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
    
    n_subplots = min(n_metrics, MAX_APERIODIC_METRICS)
    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]
    
    temperatures = pd.to_numeric(df[temp_col], errors="coerce")
    unique_temps = np.sort(temperatures.dropna().unique())
    
    if len(unique_temps) < MIN_TEMPERATURE_LEVELS_FOR_CURVES:
        plt.close(fig)
        return
    
    metric_colors = {"slope": "#E24A33", "offset": "#348ABD", "exponent": "#988ED5"}
    
    for idx, (metric, cols) in enumerate(aper_cols.items()):
        if idx >= MAX_APERIODIC_METRICS:
            break
        
        ax = axes[idx]
        metric_values = df[cols].mean(axis=1)
        aggregation = _aggregate_by_temperature(temperatures, metric_values)
        
        color = metric_colors.get(metric, DEFAULT_COLOR)
        ax.errorbar(
            aggregation.unique_temperatures,
            aggregation.means,
            yerr=aggregation.standard_errors,
            fmt="o-",
            color=color,
            capsize=4,
            capthick=1.5,
            markersize=8,
            linewidth=2,
        )
        
        valid_mask = ~np.isnan(aggregation.means)
        if valid_mask.sum() >= MIN_DATA_POINTS_FOR_FIT:
            x_valid = aggregation.unique_temperatures[valid_mask]
            y_valid = aggregation.means[valid_mask]
            
            linear_fit = _compute_linear_fit(x_valid, y_valid)
            x_fit = np.linspace(x_valid.min(), x_valid.max(), FIT_POINTS)
            y_fit = linear_fit["slope"] * x_fit + linear_fit["intercept"]
            
            ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.7, linewidth=1.5)
            ax.text(
                0.05,
                0.95,
                f"r = {linear_fit['r_squared']**0.5:.3f}\n"
                f"p = {linear_fit['p_value']:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.set_ylabel(f"Mean {metric.title()}", fontsize=10)
        ax.set_title(f"Aperiodic {metric.title()}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"Aperiodic Dose-Response (sub-{subject})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    
    path = output_dir / f"sub-{subject}_aperiodic_dose_response.png"
    save_fig(fig, path)
    plt.close(fig)
    saved_files["aperiodic_dose_response"] = path
    logger.info("Created aperiodic dose-response plot")


def _create_plot_config(
    feature_type: str,
    y_label: str,
    bands: List[str],
    band_colors: Dict[str, str],
) -> PlotConfig:
    """Create plot configuration."""
    return PlotConfig(
        feature_type=feature_type,
        y_label=y_label,
        bands=bands,
        band_colors=band_colors,
    )


def _plot_feature_dose_response(
    df: pd.DataFrame,
    temp_col: str,
    feature_cols: Dict[str, List[str]],
    plot_config: PlotConfig,
    subject: str,
    output_dir: Path,
    saved_files: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """Create all dose-response plots for a feature type."""
    _plot_dose_response_curves(
        df, temp_col, feature_cols, plot_config, subject, output_dir, saved_files, logger
    )
    _plot_nonlinearity_test(
        df, temp_col, feature_cols, plot_config, subject, output_dir, saved_files, logger
    )
    _plot_threshold_detection(
        df, temp_col, feature_cols, plot_config, subject, output_dir, saved_files, logger
    )


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
    if not isinstance(subject, str) or not subject:
        raise ValueError("Subject must be a non-empty string")
    if not isinstance(deriv_root, Path):
        raise ValueError("deriv_root must be a Path object")
    if not isinstance(task, str) or not task:
        raise ValueError("Task must be a non-empty string")
    
    saved_files: Dict[str, Path] = {}
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    output_dir = plots_dir / "dose_response"
    ensure_dir(output_dir)
    
    _, pow_df, _, _, temp_series, _, _, _, conn_df = load_subject_scatter_data(
        subject, task, deriv_root, config, logger
    )
    
    if pow_df is None or pow_df.empty:
        logger.debug("No power features found for dose-response analysis")
        return saved_files
    
    if temp_series is None or temp_series.empty:
        logger.debug("No temperature data found for dose-response analysis")
        return saved_files
    
    bands = _get_bands_from_config(config)
    band_colors = get_band_colors()
    
    df_power = pow_df.copy()
    df_power[TEMPERATURE_COLUMN] = temp_series.values
    power_cols = get_power_columns_by_band(df_power, bands=bands)
    
    if power_cols:
        power_dir = output_dir / "power"
        ensure_dir(power_dir)
        power_config = _create_plot_config(
            "Power", "Mean Power (a.u.)", bands, band_colors
        )
        _plot_feature_dose_response(
            df_power,
            TEMPERATURE_COLUMN,
            power_cols,
            power_config,
            subject,
            power_dir,
            saved_files,
            logger,
        )
    
    if conn_df is not None and not conn_df.empty:
        df_conn = conn_df.copy()
        df_conn[TEMPERATURE_COLUMN] = temp_series.values
        conn_cols = get_connectivity_columns_by_band(df_conn, bands=bands)
        
        if conn_cols:
            conn_dir = output_dir / "connectivity"
            ensure_dir(conn_dir)
            conn_config = _create_plot_config(
                "Connectivity", "Mean wPLI", bands, band_colors
            )
            _plot_feature_dose_response(
                df_conn,
                TEMPERATURE_COLUMN,
                conn_cols,
                conn_config,
                subject,
                conn_dir,
                saved_files,
                logger,
            )
    
    additional_features = _load_additional_features(deriv_root, subject, logger)
    
    if "aperiodic" in additional_features:
        df_aper = additional_features["aperiodic"].copy()
        if len(df_aper) == len(temp_series):
            df_aper[TEMPERATURE_COLUMN] = temp_series.values
            aper_cols = get_aperiodic_columns(df_aper)
            
            if aper_cols:
                aper_dir = output_dir / "aperiodic"
                ensure_dir(aper_dir)
                _plot_aperiodic_dose_response(
                    df_aper,
                    TEMPERATURE_COLUMN,
                    aper_cols,
                    subject,
                    aper_dir,
                    saved_files,
                    logger,
                )
    
    if "itpc" in additional_features:
        df_itpc = additional_features["itpc"].copy()
        if len(df_itpc) == len(temp_series):
            df_itpc[TEMPERATURE_COLUMN] = temp_series.values
            itpc_cols = get_itpc_columns_by_band(df_itpc, bands=bands)
            
            if itpc_cols:
                itpc_dir = output_dir / "itpc"
                ensure_dir(itpc_dir)
                itpc_config = _create_plot_config("ITPC", "Mean ITPC", bands, band_colors)
                _plot_feature_dose_response(
                    df_itpc,
                    TEMPERATURE_COLUMN,
                    itpc_cols,
                    itpc_config,
                    subject,
                    itpc_dir,
                    saved_files,
                    logger,
                )
    
    logger.info(f"Created {len(saved_files)} dose-response plots")
    return saved_files
