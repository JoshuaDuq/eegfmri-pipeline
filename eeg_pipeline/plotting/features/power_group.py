"""
Group-level power plotting functions.

Functions for creating group-level power plots across multiple subjects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ...utils.io.general import (
    get_band_color,
    save_fig,
)
from ...utils.analysis.tfr import validate_baseline_indices
from ...utils.analysis.stats import (
    compute_group_channel_power_statistics,
    compute_group_band_statistics,
    compute_error_bars_from_arrays,
    compute_subject_band_correlation_matrix,
    compute_group_band_correlation_matrix,
    compute_inter_band_correlation_statistics,
)
from ...utils.data.loading import extract_band_channel_vectors
from ..config import get_plot_config


###################################################################
# Helper Functions
###################################################################


def _collect_channel_names_by_band(subj_pow: Dict[str, pd.DataFrame], bands: List[str], config: Optional[Any] = None) -> Dict[str, List[str]]:
    """Collect channel names by band across subjects.
    
    Args:
        subj_pow: Dictionary mapping subject IDs to power DataFrames
        bands: List of frequency band names
        config: Optional configuration object
    
    Returns:
        Dictionary mapping band names to lists of channel names
    """
    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = plot_cfg.get_behavioral_config() if plot_cfg else {}
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    band_channels = {}
    for band in bands:
        band_str = str(band)
        channel_union = set()
        for _, df in subj_pow.items():
            cols = [c for c in df.columns if str(c).startswith(f"{power_prefix}{band_str}_")]
            channel_union.update([str(c).replace(f"{power_prefix}{band_str}_", "") for c in cols])
        band_channels[band_str] = sorted(channel_union)
    return band_channels


def _plot_group_channel_power_heatmap(
    heatmap_data: np.ndarray,
    channels: List[str],
    bands: List[str],
    output_path: Path,
    config: Optional[Any]
) -> None:
    """Plot group channel power heatmap.
    
    Args:
        heatmap_data: 2D array of power values (bands x channels)
        channels: List of channel names
        bands: List of frequency band names
        output_path: Path to save plot
        config: Optional configuration object
    """
    if heatmap_data.size == 0:
        return
    plot_cfg = get_plot_config(config)
    features_config = plot_cfg.plot_type_configs.get("features", {})
    correlation_config = features_config.get("correlation", {})
    
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    vmin = correlation_config.get("vmin", -0.6)
    vmax = correlation_config.get("vmax", 0.6)
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=plot_cfg.font.small)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([b.capitalize() for b in bands], fontsize=plot_cfg.font.medium)
    ax.set_title("Group Mean Power per Channel and Band\nlog10(power/baseline)", fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel("Channel", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Frequency Band", fontsize=plot_cfg.font.ylabel)
    plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
    plt.tight_layout()
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)


def _collect_band_power_records(subj_pow: Dict[str, pd.DataFrame], bands: List[str], config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Collect band power records across subjects.
    
    Args:
        subj_pow: Dictionary mapping subject IDs to power DataFrames
        bands: List of frequency band names
        config: Optional configuration object
    
    Returns:
        List of dictionaries with subject, band, and mean_power
    """
    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = plot_cfg.get_behavioral_config() if plot_cfg else {}
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    records = []
    for band in bands:
        band_str = str(band)
        for subject, df in subj_pow.items():
            cols = [c for c in df.columns if str(c).startswith(f"{power_prefix}{band_str}_")]
            if not cols:
                continue
            values = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            records.append({
                "subject": subject,
                "band": band_str,
                "mean_power": float(np.mean(values))
            })
    return records


def _plot_subject_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    bands_present: List[str],
    config: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None
) -> None:
    """Plot subject scatter points on group summary plot.
    
    Args:
        ax: Matplotlib axes
        df: DataFrame with band and mean_power columns
        bands_present: List of band names present in data
        config: Optional configuration object
        rng: Optional random number generator
    """
    if rng is None:
        if config is None:
            rng_seed = 42
        else:
            rng_seed = config.get("project.random_state", 42)
        rng = np.random.default_rng(rng_seed)
    
    jitter_range = 0.2
    if config is not None:
        jitter_range = config.get("behavior_analysis.group_aggregation.jitter_range", 0.2)
    
    for i, band in enumerate(bands_present):
        values = df[df["band"] == band]["mean_power"].to_numpy(dtype=float)
        jitter = (rng.random(len(values)) - 0.5) * jitter_range
        ax.scatter(
            np.full_like(values, i, dtype=float) + jitter,
            values, color='k', s=12, alpha=0.6
        )


def _plot_group_band_power_summary(
    bands_present: List[str],
    means: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    n_subjects: List[int],
    df: pd.DataFrame,
    output_path: Path,
    stats_path: Path,
    config: Optional[Any],
    logger: logging.Logger
) -> None:
    """Plot group band power summary with error bars.
    
    Args:
        bands_present: List of band names present in data
        means: List of mean values per band
        ci_lower: List of lower CI bounds per band
        ci_upper: List of upper CI bounds per band
        n_subjects: List of subject counts per band
        df: DataFrame with subject data
        output_path: Path to save plot
        stats_path: Path to save statistics CSV
        config: Optional configuration object
        logger: Logger instance
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(bands_present))
    
    ax.bar(x_positions, means, color='steelblue', alpha=0.8)
    yerr = compute_error_bars_from_arrays(means, ci_lower, ci_upper)
    ax.errorbar(x_positions, means, yerr=yerr, fmt='none', ecolor='k', capsize=3)
    _plot_subject_scatter(ax, df, bands_present, config=config)
    
    plot_cfg = get_plot_config(config)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([b.capitalize() for b in bands_present], fontsize=plot_cfg.font.medium)
    ax.set_ylabel('Mean log10(power/baseline) across subjects', fontsize=plot_cfg.font.ylabel)
    ax.set_title('Group Band Power Summary (subject means, 95% CI)', fontsize=plot_cfg.font.figure_title)
    ax.axhline(0, color=plot_cfg.style.colors.black, linewidth=plot_cfg.style.line.width_standard)
    
    plt.tight_layout()
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    output_df = pd.DataFrame({
        "band": bands_present,
        "group_mean": means,
        "ci_low": ci_lower,
        "ci_high": ci_upper,
        "n_subjects": n_subjects
    })
    output_df.to_csv(stats_path, sep="\t", index=False)
    logger.info("Saved group band power distributions and stats.")


def _plot_group_inter_band_correlation(
    group_correlation: np.ndarray,
    band_names: List[str],
    output_path: Path,
    config: Optional[Any]
) -> None:
    """Plot group inter-band correlation heatmap.
    
    Args:
        group_correlation: 2D correlation matrix
        band_names: List of frequency band names
        output_path: Path to save plot
        config: Optional configuration object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(group_correlation, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(band_names)))
    ax.set_yticks(range(len(band_names)))
    ax.set_xticklabels([b.capitalize() for b in band_names], rotation=45, ha='right')
    ax.set_yticklabels([b.capitalize() for b in band_names])
    for i in range(len(band_names)):
        for j in range(len(band_names)):
            if np.isfinite(group_correlation[i, j]):
                value = group_correlation[i, j]
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f"{value:.2f}", ha='center', va='center', color=text_color)
    plot_cfg = get_plot_config(config)
    ax.set_title('Group Inter Band Spatial Power Correlation', fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    cbar = plt.colorbar(im, ax=ax, label='Correlation (r)')
    cbar.set_label('Correlation (r)', fontsize=plot_cfg.font.title)
    plt.tight_layout()
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)


def _add_baseline_region_to_axis(ax: plt.Axes, times: np.ndarray, config: Optional[Any]) -> None:
    """Add baseline region shading to axis.
    
    Args:
        ax: Matplotlib axes
        times: Time array
        config: Optional configuration object
    """
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]) if config else [-2.0, 0.0])
    min_baseline_samples = int(config.get("time_frequency_analysis.min_baseline_samples", 5) if config else 5)
    b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline, min_samples=min_baseline_samples)
    baseline_start = max(float(times.min()), float(b_start))
    baseline_end = min(float(times.max()), float(b_end))
    if baseline_end > baseline_start:
        ax.axvspan(baseline_start, baseline_end, alpha=0.1, color='gray')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)


def _plot_group_band_time_course(
    valid_bands: List[str],
    band_data: Dict[str, List[np.ndarray]],
    times: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
    config: Optional[Any],
    logger: logging.Logger
) -> None:
    """Plot group band time course with confidence intervals.
    
    Args:
        valid_bands: List of band names to plot
        band_data: Dictionary mapping band names to lists of time series arrays
        times: Time array
        ylabel: Y-axis label
        title: Plot title
        output_path: Path to save plot
        config: Optional configuration object
        logger: Logger instance
    """
    nrows = len(valid_bands)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    
    for i, band in enumerate(valid_bands):
        ax = axes[i]
        series_list = band_data.get(band, [])
        if len(series_list) < 2:
            continue
        array = np.vstack(series_list)
        mean_values = np.nanmean(array, axis=0)
        se = np.nanstd(array, axis=0, ddof=1) / np.sqrt(array.shape[0])
        ci = 1.96 * se
        band_color = get_band_color(band, config)
        ax.plot(times, mean_values, color=band_color, label=str(band))
        ax.fill_between(times, mean_values - ci, mean_values + ci, color=band_color, alpha=0.2)
        _add_baseline_region_to_axis(ax, times, config)
        ax.set_title(f"{band.capitalize()} (group mean ±95% CI)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    plot_cfg = get_plot_config(config)
    axes[-1].set_xlabel("Time (s)", fontsize=plot_cfg.font.label)
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info(f"Saved {title.lower()}")


###################################################################
# Group Power Plotting Functions
###################################################################


def plot_group_power_plots(
    subj_pow: Dict[str, pd.DataFrame],
    bands: List[str],
    gplots: Path,
    gstats: Path,
    config: Optional[Any],
    logger: logging.Logger
) -> None:
    """Create group-level power plots across subjects.
    
    Args:
        subj_pow: Dictionary mapping subject IDs to power DataFrames
        bands: List of frequency band names
        gplots: Directory to save plots
        gstats: Directory to save statistics
        config: Optional configuration object
        logger: Logger instance
    """
    band_channels = _collect_channel_names_by_band(subj_pow, bands, config)
    all_channels = sorted(set().union(*band_channels.values())) if band_channels else []
    
    heatmap_rows, statistics_rows = compute_group_channel_power_statistics(
        subj_pow, bands, all_channels
    )
    heatmap_data = np.vstack(heatmap_rows) if heatmap_rows else np.zeros((0, 0))
    
    _plot_group_channel_power_heatmap(
        heatmap_data, all_channels, bands,
        gplots / "group_channel_power_heatmap", config
    )
    pd.DataFrame(statistics_rows).to_csv(
        gstats / "group_channel_power_means.tsv", sep="\t", index=False
    )
    logger.info("Saved group channel power heatmap and stats.")
    
    records = _collect_band_power_records(subj_pow, bands, config)
    df_records = pd.DataFrame(records)
    if not df_records.empty:
        bands_present, means, ci_lower, ci_upper, n_subjects = compute_group_band_statistics(
            df_records, bands
        )
        _plot_group_band_power_summary(
            bands_present, means, ci_lower, ci_upper, n_subjects, df_records,
            gplots / "group_power_distributions_per_band_across_subjects",
            gstats / "group_band_power_subject_means.tsv",
            config, logger
        )
    
    default_freq_bands = {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    }
    freq_bands = config.get("time_frequency_analysis.bands", default_freq_bands) if config else default_freq_bands
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    per_subject_correlations = []
    
    for _, df in subj_pow.items():
        band_vectors = extract_band_channel_vectors(df, band_names)
        if len(band_vectors) < 2:
            continue
        correlation_matrix = compute_subject_band_correlation_matrix(band_vectors, band_names)
        per_subject_correlations.append(correlation_matrix)
    
    if len(per_subject_correlations) >= 2:
        group_correlation = compute_group_band_correlation_matrix(
            per_subject_correlations, n_bands
        )
        _plot_group_inter_band_correlation(
            group_correlation, band_names,
            gplots / "group_inter_band_spatial_power_correlation", config
        )
        correlation_statistics = compute_inter_band_correlation_statistics(
            per_subject_correlations, band_names
        )
        if correlation_statistics:
            pd.DataFrame(correlation_statistics).to_csv(
                gstats / "group_inter_band_correlation.tsv", sep="\t", index=False
            )
            logger.info("Saved group inter-band correlation heatmap and stats.")


def plot_group_band_power_time_courses(
    valid_bands: List[str],
    band_tc: Dict[str, List[np.ndarray]],
    band_tc_pct: Dict[str, List[np.ndarray]],
    tref: np.ndarray,
    gplots: Path,
    config: Optional[Any],
    logger: logging.Logger
) -> None:
    """Plot group band power time courses.
    
    Args:
        valid_bands: List of band names to plot
        band_tc: Dictionary mapping band names to lists of time course arrays (log ratio)
        band_tc_pct: Dictionary mapping band names to lists of time course arrays (percent change)
        tref: Time reference array
        gplots: Directory to save plots
        config: Optional configuration object
        logger: Logger instance
    """
    _plot_group_band_time_course(
        valid_bands, band_tc, tref,
        "log10(power/baseline)", "Group Band Power Time Courses",
        gplots / "group_band_power_time_courses", config, logger
    )
    _plot_group_band_time_course(
        valid_bands, band_tc_pct, tref,
        "Percent change from baseline (%)",
        "Group Band Power Time Courses (percent change, ratio-domain averaging)",
        gplots / "group_band_power_time_courses_percent_change", config, logger
    )

