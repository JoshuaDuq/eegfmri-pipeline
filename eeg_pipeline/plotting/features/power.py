"""
Power distribution and PSD plotting functions.

Extracted from plot_features.py for modular organization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from eeg_pipeline.utils.io.general import (
    get_band_color,
    save_fig,
    find_pain_column_in_events,
    find_temperature_column_in_events,
)
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.tfr import (
    apply_baseline_and_crop,
    validate_baseline_indices,
    extract_band_channel_means,
)
from eeg_pipeline.utils.analysis.stats import (
    compute_inter_band_coupling_matrix,
)


###################################################################
# Helper Functions
###################################################################


def _setup_subplot_grid(n_items: int, n_cols: int = 2) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a subplot grid for multiple plots.
    
    Args:
        n_items: Number of subplots needed
        n_cols: Number of columns (default: 2)
    
    Returns:
        Tuple of (figure, list of axes)
    """
    n_rows = (n_items + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    if n_items == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    return fig, axes


def _validate_epochs_tfr(tfr: Any, function_name: str, logger: logging.Logger) -> bool:
    """Validate that TFR is EpochsTFR (4D) and raise if AverageTFR (3D).
    
    Args:
        tfr: TFR object to validate
        function_name: Name of calling function for error messages
        logger: Logger instance
    
    Returns:
        True if valid
    
    Raises:
        TypeError: If TFR is not EpochsTFR
        ValueError: If TFR data shape is incorrect
    """
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if isinstance(tfr, mne.time_frequency.AverageTFR):
            error_msg = (
                f"{function_name} requires EpochsTFR (4D: n_epochs, n_channels, n_freqs, n_times), "
                f"but received AverageTFR (3D: n_channels, n_freqs, n_times). "
                f"Cannot split by epochs/conditions with averaged data."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        else:
            error_msg = (
                f"{function_name} requires EpochsTFR, but received {type(tfr).__name__}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
    
    if len(tfr.data.shape) != 4:
        error_msg = (
            f"{function_name} requires 4D TFR data (n_epochs, n_channels, n_freqs, n_times), "
            f"but received shape {tfr.data.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True


def _get_plateau_window(config: Any) -> List[float]:
    """Get plateau window from config.
    
    Args:
        config: Configuration object
    
    Returns:
        List of [start, end] times for plateau window
    """
    if config:
        return config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    return [3.0, 10.5]


def _crop_tfr_to_plateau(tfr: Any, plateau_window: List[float], logger: logging.Logger) -> Optional[Any]:
    """Crop TFR to plateau window.
    
    Args:
        tfr: TFR object to crop
        plateau_window: List of [start, end] times
        logger: Logger instance
    
    Returns:
        Cropped TFR or None if window is invalid
    """
    times = np.asarray(tfr.times)
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    tmin = max(times.min(), plateau_start)
    tmax = min(times.max(), plateau_end)
    
    if tmax <= tmin:
        logger.warning("Invalid plateau window; skipping PSD")
        return None
    
    return tfr.copy().crop(tmin, tmax)


def _validate_temperature_data(
    tfr: Any, events_df: Optional[pd.DataFrame], subject: str, logger: logging.Logger
) -> Optional[pd.Series]:
    """Validate and extract temperature data from events DataFrame.
    
    Args:
        tfr: TFR object
        events_df: Events DataFrame
        subject: Subject identifier
        logger: Logger instance
    
    Returns:
        Series of temperature values or None if validation fails
    """
    if events_df is None or events_df.empty:
        logger.warning("No events DataFrame provided for temperature analysis")
        return None
        
    temp_col = find_temperature_column_in_events(events_df)
    if temp_col is None:
        logger.warning("No temperature column found in events")
        return None
        
    temps = pd.to_numeric(events_df[temp_col], errors="coerce")
    if len(tfr) != len(temps):
        logger.warning(
            f"TFR window ({len(tfr)} epochs) and events "
            f"({len(temps)} rows) length mismatch for subject {subject}"
        )
        return None
        
    return temps


def _get_band_frequency_mask(tfr: Any, band: str, config: Any, logger: logging.Logger) -> Optional[np.ndarray]:
    """Get frequency mask for a given band.
    
    Args:
        tfr: TFR object
        band: Band name (e.g., 'alpha')
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Boolean mask array or None if band not found
    """
    if config is None:
        logger.warning("Config is required to get band frequency mask")
        return None
        
    freq_bands = config.get("time_frequency_analysis.bands")
    if not freq_bands:
        freq_bands = {
            "delta": [1.0, 3.9],
            "theta": [4.0, 7.9],
            "alpha": [8.0, 12.9],
            "beta": [13.0, 30.0],
            "gamma": [30.1, 80.0],
        }
        
    if band not in freq_bands:
        logger.warning(f"Band '{band}' not found in configuration")
        return None
        
    fmin, fmax = freq_bands[band]
    mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    
    if not mask.any():
        logger.warning(f"No frequencies found for band '{band}' ({fmin}-{fmax} Hz)")
        return None
        
    return mask


###################################################################
# Power Distribution Plotting
###################################################################


def plot_power_distributions(
    pow_df: pd.DataFrame,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot power distributions per frequency band.
    
    Args:
        pow_df: DataFrame with power columns
        bands: List of frequency band names
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    n_bands = len(bands)
    fig, axes = _setup_subplot_grid(n_bands)
    
    for i, band in enumerate(bands):
        band = str(band)
        band_cols = [col for col in pow_df.columns if str(col).startswith(f'{power_prefix}{band}_')]
        if not band_cols:
            logger.warning(f"No columns found for band '{band}'")
            continue
        
        band_data = pow_df[band_cols].values.flatten()
        band_data = band_data[~np.isnan(band_data)]
        if len(band_data) == 0:
            logger.warning(f"No valid data for band '{band}'")
            continue
        
        parts = axes[i].violinplot(
            [band_data], positions=[1], showmeans=True, showmedians=True
        )
        band_color = get_band_color(band, config)
        for pc in parts['bodies']:
            pc.set_facecolor(band_color)
            pc.set_alpha(plot_cfg.style.alpha_violin_body)
        
        axes[i].axhline(y=0, color=plot_cfg.style.colors.red, linestyle='--', 
                       alpha=plot_cfg.style.line.alpha_reference, label='Baseline')
        axes[i].set_title(f'{band.capitalize()} Power Distribution\n(All channels, all trials)',
                         fontsize=plot_cfg.font.title)
        axes[i].set_ylabel('log10(power/baseline)', fontsize=plot_cfg.font.ylabel)
        axes[i].set_xticks([])
        axes[i].grid(True, alpha=plot_cfg.style.alpha_grid)
        
        mean_val = np.mean(band_data)
        std_val = np.std(band_data)
        median_val = np.median(band_data)
        stats_text = (
            f'μ={mean_val:.3f}\nσ={std_val:.3f}\n'
            f'Mdn={median_val:.3f}\nn={len(band_data)}'
        )
        axes[i].text(
            0.7, 0.95, stats_text, transform=axes[i].transAxes,
            verticalalignment='top', fontsize=plot_cfg.font.small,
            bbox=dict(boxstyle='round', facecolor='white', alpha=plot_cfg.style.alpha_text_box)
        )
    
    for j in range(len(bands), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f'sub-{subject}_power_distributions_per_band',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    logger.info("Saved power distributions")


def plot_channel_power_heatmap(
    pow_df: pd.DataFrame,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot heatmap of mean power per channel and band.
    
    Args:
        pow_df: DataFrame with power columns
        bands: List of frequency band names
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    band_means = []
    channel_names = []
    valid_bands = []
    
    for band in bands:
        band = str(band)
        band_cols = [col for col in pow_df.columns if str(col).startswith(f'{power_prefix}{band}_')]
        if band_cols:
            band_data = pow_df[band_cols].mean(axis=0)
            band_means.append(band_data.values)
            valid_bands.append(band)
            if not channel_names:
                channel_names = [col.replace(f'{power_prefix}{band}_', '') for col in band_cols]
    
    if not band_means:
        logger.warning("No valid band data for heatmap")
        return
    
    features_config = plot_cfg.plot_type_configs.get("features", {})
    power_config = features_config.get("power", {})
    
    heatmap_data = np.array(band_means)
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    vmin = power_config.get("vmin")
    vmax = power_config.get("vmax")
    if vmin is None or vmax is None:
        data_min = np.nanmin(heatmap_data)
        data_max = np.nanmax(heatmap_data)
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=plot_cfg.font.small)
    ax.set_yticks(range(len(valid_bands)))
    ax.set_yticklabels([b.capitalize() for b in valid_bands], fontsize=plot_cfg.font.medium)
    ax.set_title('Mean Power per Channel and Band\nlog10(power/baseline)', fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel('Channel', fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
    
    heatmap_threshold = features_config.get("heatmap_text_threshold", 200)
    if len(channel_names) * len(valid_bands) <= heatmap_threshold:
        std_threshold = np.std(heatmap_data)
        for i in range(len(valid_bands)):
            for j in range(len(channel_names)):
                value = heatmap_data[i, j]
                text = f'{value:.2f}'
                color = 'white' if abs(value) > std_threshold else 'black'
                fontsize = max(6, min(10, heatmap_threshold/len(channel_names)))
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=fontsize)
    
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f'sub-{subject}_channel_power_heatmap',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    logger.info("Saved channel power heatmap")


def plot_power_time_courses(
    tfr_raw: Any,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot power time courses for multiple bands.
    
    Args:
        tfr_raw: TFR object (raw, not averaged)
        bands: List of frequency band names
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    times = tfr_raw.times
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    plot_cfg = get_plot_config(config)
    
    n_epochs = tfr_raw.data.shape[0]
    n_channels = tfr_raw.data.shape[1]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    for band in bands:
        if band not in features_freq_bands:
            logger.warning(f"Band '{band}' not in config; skipping time course.")
            continue
        
        fmin, fmax = features_freq_bands[band]
        freq_mask = (tfr_raw.freqs >= fmin) & (tfr_raw.freqs <= fmax)
        if not freq_mask.any():
            logger.warning(f"No frequencies found for {band} band ({fmin}-{fmax} Hz)")
            continue
        
        band_power_log = tfr_raw.data[:, :, freq_mask, :].mean(axis=(0, 1, 2))
        ax.plot(times, band_power_log, linewidth=2, color=get_band_color(band, config), label=band.capitalize())
    
    b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline)
    bs = max(float(times.min()), float(b_start))
    be = min(float(times.max()), float(b_end))
    if be > bs:
        ax.axvspan(bs, be, alpha=0.2, color='gray', label='Baseline')
    ax.axvspan(0, times[-1], alpha=0.2, color='orange', label='Stimulus')
    
    ax.set_ylabel('log10(power/baseline)', fontsize=plot_cfg.font.ylabel)
    ax.set_xlabel('Time (s)', fontsize=plot_cfg.font.label)
    ax.set_title(f'Band Power Time Courses (sub-{subject})', fontsize=plot_cfg.font.title)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid)
    ax.legend(fontsize=plot_cfg.font.small, loc='best')
    
    footer_text = (
        f"n={n_epochs} trials, {n_channels} channels | "
        f"Baseline: [{b_start:.2f}, {b_end:.2f}]s (logratio)"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_dir / f'sub-{subject}_power_time_courses_all_bands'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved combined power time courses for all bands")


###################################################################
# Power Spectral Density Plotting
###################################################################


def _plot_psd_by_temperature(
    tfr_epochs: Any,
    temps: pd.Series,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> bool:
    """Plot PSD by temperature condition (internal helper).
    
    Args:
        tfr_epochs: EpochsTFR object
        temps: Series of temperature values
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    
    Returns:
        True if plot was created, False otherwise
    """
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < 2:
        return False
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    temp_colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(unique_temps)))
    
    plateau_window = _get_plateau_window(config)
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    
    trial_counts = []
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        n_trials_temp = int(temp_mask.sum())
        if n_trials_temp < 1:
            continue
        trial_counts.append(f"{temp:.0f}°C: n={n_trials_temp}")
            
        tfr_temp_avg = tfr_epochs[temp_mask].average()
        apply_baseline_and_crop(tfr_temp_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_temp_win = _crop_tfr_to_plateau(tfr_temp_avg, plateau_window, logger)
        
        if tfr_temp_win is None:
            continue
            
        psd_avg = tfr_temp_win.data.mean(axis=(0, 2))
        
        if len(psd_avg) != len(tfr_temp_win.freqs):
            logger.warning(f"Frequency dimension mismatch: {len(psd_avg)} vs {len(tfr_temp_win.freqs)}")
            continue
            
        ax.plot(
            tfr_temp_win.freqs, psd_avg,
            color=temp_colors[idx], linewidth=1.5,
            label=f'{temp:.0f}°C (n={n_trials_temp})', alpha=0.9
        )
    
    ax.axhline(0, color=plot_cfg.style.colors.gray, 
               linewidth=plot_cfg.style.line.width_standard, 
               alpha=plot_cfg.style.line.alpha_dim, linestyle='--')
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Power spectral density (log10 ratio to baseline)", fontsize=plot_cfg.font.ylabel)
    ax.legend(loc='upper left', fontsize=plot_cfg.font.title, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, linestyle=':', linewidth=0.5)
    
    footer_text = (
        f"Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}]s | "
        f"Plateau: [{plateau_window[0]:.1f}, {plateau_window[1]:.1f}]s | "
        f"Total: n={len(tfr_epochs)} trials"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_temperature'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved PSD by temperature (Induced)")
    return True


def _plot_psd_overall(
    tfr_avg_win: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot overall PSD (internal helper).
    
    Args:
        tfr_avg_win: AverageTFR object (already averaged, baselined, and cropped)
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    psd_avg = tfr_avg_win.data.mean(axis=(0, 2))
    
    fig, ax = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    ax.plot(tfr_avg_win.freqs, psd_avg, color="0.2", linewidth=1.0)
    ax.axhline(0, color="0.7", linewidth=0.5, alpha=0.6)
    
    default_freq_bands = {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    }
    freq_bands = config.get("time_frequency_analysis.bands", default_freq_bands)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    for band, (fmin, fmax) in features_freq_bands.items():
        if fmin < tfr_avg_win.freqs.max():
            fmax_clipped = min(fmax, tfr_avg_win.freqs.max())
            ax.axvspan(fmin, fmax_clipped, alpha=0.08, color="0.5", linewidth=0)
            mid = (fmin + fmax_clipped) / 2
            if mid < tfr_avg_win.freqs.max():
                y_max = ax.get_ylim()[1]
                ax.text(
                    mid, y_max * 0.95, band[0].upper(),
                    fontsize=7, ha="center", va="top", color="0.4"
                )
    
    plot_cfg = get_plot_config(config)
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.medium)
    ax.set_ylabel("log10(power/baseline)", fontsize=plot_cfg.font.medium)
    ax.tick_params(labelsize=plot_cfg.font.small)
    sns.despine(ax=ax, trim=True)
    
    output_path = save_dir / f'sub-{subject}_power_spectral_density'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved PSD (Induced)")


def plot_power_spectral_density(
    tfr: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    events_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None
) -> None:
    """Plot power spectral density.
    
    If events_df with temperature column is provided, plots by temperature.
    Otherwise, plots overall induced PSD.
    
    Args:
        tfr: EpochsTFR object
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        events_df: Optional events DataFrame with temperature column
        config: Configuration object
    """
    _validate_epochs_tfr(tfr, "plot_power_spectral_density", logger)
    
    plateau_window = _get_plateau_window(config)
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    
    if events_df is not None and not events_df.empty:
        temp_col = find_temperature_column_in_events(events_df)
        if temp_col is not None:
            temps = pd.to_numeric(events_df[temp_col], errors="coerce")
            if len(tfr) != len(temps):
                raise ValueError(
                    f"TFR window ({len(tfr)} epochs) and events "
                    f"({len(temps)} rows) length mismatch for subject {subject}"
                )
            if _plot_psd_by_temperature(tfr, temps, subject, save_dir, logger, config):
                return
    
    tfr_avg = tfr.copy().average()
    apply_baseline_and_crop(tfr_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
    tfr_win = _crop_tfr_to_plateau(tfr_avg, plateau_window, logger)
    
    if tfr_win is not None:
        _plot_psd_overall(tfr_win, subject, save_dir, logger, config)


def plot_power_spectral_density_by_pain(
    tfr: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    events_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None
) -> None:
    """Plot PSD by pain condition.
    
    Args:
        tfr: EpochsTFR object
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        events_df: Events DataFrame with pain column
        config: Configuration object
    """
    _validate_epochs_tfr(tfr, "plot_power_spectral_density_by_pain", logger)
    
    if events_df is None or events_df.empty:
        logger.warning("No events for PSD by pain")
        return
    
    pain_col = find_pain_column_in_events(events_df)
    if pain_col is None:
        logger.warning("No pain binary column found")
        return
    
    plateau_window = _get_plateau_window(config)
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    
    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    
    if len(tfr) != len(pain_vals):
        raise ValueError(
            f"TFR window ({len(tfr)} epochs) and events "
            f"({len(pain_vals)} rows) length mismatch for subject {subject}"
        )
    
    nonpain_mask = (pain_vals == 0).to_numpy()
    pain_mask = (pain_vals == 1).to_numpy()
    
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1:
        logger.warning("Insufficient trials for pain comparison")
        return
    
    n_nonpain = int(nonpain_mask.sum())
    n_pain = int(pain_mask.sum())
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mask, label, color, n_trials in [
        (nonpain_mask, 'Non-pain', 'steelblue', n_nonpain),
        (pain_mask, 'Pain', 'orangered', n_pain)
    ]:
        if mask.sum() < 1:
            continue
            
        tfr_cond_avg = tfr[mask].average()
        apply_baseline_and_crop(tfr_cond_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_cond_win = _crop_tfr_to_plateau(tfr_cond_avg, plateau_window, logger)
        
        if tfr_cond_win is None:
            continue
            
        psd_mean = tfr_cond_win.data.mean(axis=(0, 2))
        
        if len(psd_mean) != len(tfr_cond_win.freqs):
            logger.warning(f"Frequency dimension mismatch for {label}")
            continue
            
        ax.plot(
            tfr_cond_win.freqs, psd_mean,
            color=color, linewidth=1.5, label=f'{label} (n={n_trials})', alpha=0.9
        )
    
    ax.axhline(0, color=plot_cfg.style.colors.gray, 
               linewidth=plot_cfg.style.line.width_standard, 
               alpha=plot_cfg.style.line.alpha_dim, linestyle='--')
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Power spectral density (log10 ratio to baseline)", fontsize=plot_cfg.font.ylabel)
    ax.legend(loc='upper left', fontsize=plot_cfg.font.title, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, linestyle=':', linewidth=0.5)
    
    footer_text = (
        f"Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}]s | "
        f"Plateau: [{plateau_window[0]:.1f}, {plateau_window[1]:.1f}]s | "
        f"Total: n={len(tfr)} trials"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_pain'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved PSD by pain condition (Induced)")


def plot_power_time_course_by_temperature(
    tfr: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    events_df: Optional[pd.DataFrame] = None,
    band: str = 'alpha',
    config: Optional[Any] = None
) -> None:
    """Plot power time course by temperature for a specific band.
    
    Args:
        tfr: EpochsTFR object
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        events_df: Events DataFrame with temperature column
        band: Frequency band name (default: 'alpha')
        config: Configuration object
    """
    _validate_epochs_tfr(tfr, "plot_power_time_course_by_temperature", logger)
    
    band = str(band)
    temps = _validate_temperature_data(tfr, events_df, subject, logger)
    if temps is None:
        return
    
    freq_mask = _get_band_frequency_mask(tfr, band, config, logger)
    if freq_mask is None:
        return
    
    unique_temps = sorted(temps.dropna().unique())
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(unique_temps)))
    fig, ax = plt.subplots(figsize=(10, 5))
    
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    freq_bands = config.get("time_frequency_analysis.bands", {})
    band_range = freq_bands.get(band, [None, None])
    
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        n_trials_temp = int(temp_mask.sum())
        if n_trials_temp < 1:
            continue
        
        tfr_temp_avg = tfr[temp_mask].average()
        apply_baseline_and_crop(tfr_temp_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        
        band_power = tfr_temp_avg.data[:, freq_mask, :].mean(axis=(0, 1))
        
        ax.plot(
            tfr_temp_avg.times, band_power,
            color=colors[idx], linewidth=1.8, alpha=0.9,
            label=f"{temp:.0f}°C (n={n_trials_temp})"
        )
    
    plot_cfg = get_plot_config(config)
    ax.axhline(0, color=plot_cfg.style.colors.gray, 
               linewidth=plot_cfg.style.line.width_standard, 
               alpha=plot_cfg.style.line.alpha_dim, linestyle='--')
    ax.set_xlabel("Time (s)", fontsize=plot_cfg.font.ylabel)
    band_label = f"{band.capitalize()}"
    if band_range[0] is not None and band_range[1] is not None:
        band_label += f" ({band_range[0]:.0f}-{band_range[1]:.0f} Hz)"
    ax.set_ylabel(f"{band_label} power (log10 ratio)", fontsize=plot_cfg.font.ylabel)
    ax.legend(loc='upper left', fontsize=plot_cfg.font.title, frameon=False)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, linestyle=':', linewidth=0.5)
    
    footer_text = (
        f"Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}]s | "
        f"Total: n={len(tfr)} trials"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_dir / f'sub-{subject}_time_course_by_temperature_{band}'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved band time course by temperature (Induced)")


def plot_trial_power_variability(pow_df, bands, subject, save_dir, logger, config):
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    n_bands = len(bands)
    n_trials = len(pow_df)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3 * n_bands))
    if n_bands == 1:
        axes = [axes]
    
    for i, band in enumerate(bands):
        band_str = str(band)
        band_cols = [col for col in pow_df.columns if str(col).startswith(f'{power_prefix}{band_str}_')]
        if not band_cols:
            continue
        
        n_channels = len(band_cols)
        band_power_trials = pow_df[band_cols].mean(axis=1)
        trial_numbers = range(1, len(band_power_trials) + 1)
        band_color = get_band_color(band_str, config)
        axes[i].plot(
            trial_numbers, band_power_trials, 'o-',
            alpha=0.7, linewidth=1, color=band_color
        )
        
        mean_power = band_power_trials.mean()
        std_power = band_power_trials.std()
        coefficient_of_variation = (
            std_power / abs(mean_power) if abs(mean_power) > 1e-10 else np.nan
        )
        
        axes[i].axhline(
            mean_power, color='red', linestyle='--', alpha=0.8,
            label=f'Mean = {mean_power:.3f}'
        )
        axes[i].fill_between(
            trial_numbers, mean_power - std_power, mean_power + std_power,
            alpha=0.2, color='red', label=f'±1 SD = ±{std_power:.3f}'
        )
        axes[i].set_ylabel(f'{band_str.capitalize()}\nlog10(power/baseline)')
        axes[i].set_title(
            f'{band_str.capitalize()} Band Power Variability (n={n_trials} trials, {n_channels} channels, CV={coefficient_of_variation:.3f})'
        )
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plot_cfg = get_plot_config(config)
    axes[-1].set_xlabel('Trial Number', fontsize=plot_cfg.font.label)
    
    footer_text = f"n={n_trials} trials | Units: log10(power/baseline)"
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig(
        fig, save_dir / f'sub-{subject}_trial_power_variability',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    logger.info("Saved trial power variability")


def plot_inter_band_spatial_power_correlation(tfr, subject, save_dir, logger, config):
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    
    times = np.asarray(tfr.times)
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    tmin_clip = float(max(times.min(), plateau_start))
    tmax_clip = float(min(times.max(), plateau_end))
    
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        logger.warning(
            f"Skipping inter-band spatial power correlation: invalid plateau within data range "
            f"(requested [{plateau_start}, {plateau_end}] s, "
            f"available [{times.min():.2f}, {times.max():.2f}] s)"
        )
        return
    
    tfr_windowed = tfr.copy().crop(tmin_clip, tmax_clip)
    tfr_avg = tfr_windowed.average()
    coupling_matrix = compute_inter_band_coupling_matrix(
        tfr_avg,
        band_names,
        features_freq_bands,
        extract_band_channel_means
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(coupling_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_bands))
    ax.set_yticks(range(n_bands))
    ax.set_xticklabels([band.capitalize() for band in band_names], rotation=45, ha='right')
    ax.set_yticklabels([band.capitalize() for band in band_names])
    
    for i in range(n_bands):
        for j in range(n_bands):
            value = coupling_matrix[i, j]
            text_color = "black" if abs(value) < 0.5 else "white"
            ax.text(
                j, i, f'{value:.2f}',
                ha="center", va="center", color=text_color
            )
    
    plot_cfg = get_plot_config(config)
    ax.set_title('Inter Band Spatial Power Correlation', fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation (r)', fontsize=plot_cfg.font.title)
    
    plt.tight_layout()
    output_path = save_dir / f'sub-{subject}_inter_band_spatial_power_correlation'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved inter-band spatial power correlation")

