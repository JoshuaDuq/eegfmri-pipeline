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
from mne.viz import plot_topomap

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.io.figures import get_band_color, save_fig
from eeg_pipeline.utils.data.columns import (
    find_pain_column_in_events,
    find_temperature_column_in_events,
)
from eeg_pipeline.utils.analysis.events import extract_pain_mask
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.roi import (
    get_roi_definitions,
    get_roi_channels,
    extract_channels_from_columns,
)
from eeg_pipeline.utils.analysis.tfr import (
    apply_baseline_and_crop,
    validate_baseline_indices,
    extract_band_channel_means,
)
from eeg_pipeline.utils.analysis.stats import (
    compute_inter_band_coupling_matrix,
)
from eeg_pipeline.plotting.features.utils import compute_cohens_d
from eeg_pipeline.plotting import utils as plot_utils
from eeg_pipeline.utils.data.features import get_power_columns_by_band
from scipy.stats import mannwhitneyu


###################################################################
# Helper Functions
###################################################################


def plot_power_by_condition(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare power between conditions per band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if power_df is None or power_df.empty or events_df is None:
        return

    from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from .utils import (
        plot_paired_comparison,
        apply_fdr_correction,
        get_named_segments,
    )

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    # Get segments from config or auto-detect from data
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = get_named_segments(power_df, group="power")
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                logger.info(f"Auto-detected segments for comparison: {segments}")
    
    bands = list(get_frequency_band_names(config) or ["delta", "theta", "alpha", "beta", "gamma"])
    
    rois = get_roi_definitions(config)
    all_channels = extract_channels_from_columns(list(power_df.columns))
    
    # Determine which ROIs to plot
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif r in rois:
                roi_names.append(r)
    else:
        roi_names = ["all"]
        if rois:
            roi_names.extend(list(rois.keys()))
    
    if logger:
        logger.info(f"Power comparison: segments={segments}, ROIs={roi_names}, compare_windows={compare_wins}, compare_columns={compare_cols}")
    
    # Window comparison (paired) - use unified helper
    if compare_wins and len(segments) >= 2:

        seg1, seg2 = segments[0], segments[1]
        
        for roi_name in roi_names:
            if roi_name == "all":
                roi_channels = all_channels
            else:
                roi_channels = get_roi_channels(rois[roi_name], all_channels)
            
            if not roi_channels:
                continue
            
            roi_set = set(roi_channels)
            
            # Collect data by band
            data_by_band = {}
            for band in bands:
                cols1, cols2 = [], []
                for c in power_df.columns:
                    parsed = NamingSchema.parse(str(c))
                    if not (parsed.get("valid") and parsed.get("group") == "power"):
                        continue
                    channel_id = str(parsed.get("identifier") or "")
                    if channel_id and channel_id not in roi_set:
                        continue
                    if str(parsed.get("segment") or "") == seg1 and str(parsed.get("band") or "") == band:
                        cols1.append(c)
                    if str(parsed.get("segment") or "") == seg2 and str(parsed.get("band") or "") == band:
                        cols2.append(c)
                
                if cols1 and cols2:
                    s1 = power_df[cols1].mean(axis=1)
                    s2 = power_df[cols2].mean(axis=1)
                    valid_mask = s1.notna() & s2.notna()
                    v1, v2 = s1[valid_mask].values, s2[valid_mask].values
                    if len(v1) > 0:
                        data_by_band[band] = (v1, v2)
            
            if data_by_band:
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_power_by_condition{suffix}_window"
                
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="Band Power",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )

    # Column comparison (unpaired)
    if compare_cols:
        comp_mask_info = extract_comparison_mask(events_df, config)
        if not comp_mask_info:
            if logger:
                logger.debug("Column comparison requested but config incomplete")
        else:
            m1, m2, label1, label2 = comp_mask_info
            seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
            plot_cfg = get_plot_config(config)
            segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
            band_colors = {band: get_band_color(band, config) for band in bands}

            # Try to load pre-computed column comparison stats
            from .utils import load_precomputed_paired_stats, get_precomputed_qvalues
            
            precomputed_column_stats = None
            if stats_dir is not None:
                precomputed_column_stats = load_precomputed_paired_stats(
                    stats_dir=stats_dir,
                    feature_type="power",
                    comparison_type="column",
                    condition1=label1.lower(),
                    condition2=label2.lower(),
                    roi_name=None,
                )
                if precomputed_column_stats is not None and not precomputed_column_stats.empty:
                    if logger:
                        logger.info(f"Using pre-computed column comparison stats ({len(precomputed_column_stats)} entries)")

            for roi_name in roi_names:
                if roi_name == "all":
                    roi_channels = all_channels
                else:
                    roi_channels = get_roi_channels(rois[roi_name], all_channels)
                
                if not roi_channels:
                    continue
                
                roi_set = set(roi_channels)
                all_pvals, pvalue_keys, cell_data = [], [], {}
                
                # Use pre-computed stats if available
                use_precomputed = precomputed_column_stats is not None and not precomputed_column_stats.empty
                
                if use_precomputed:
                    qvalues = get_precomputed_qvalues(precomputed_column_stats, bands, roi_name or "all")
                    n_significant = sum(1 for v in qvalues.values() if v[3])
                    
                    # Still need cell_data for plotting the actual values
                    for col_idx, band in enumerate(bands):
                        cols = []
                        for c in power_df.columns:
                            parsed = NamingSchema.parse(str(c))
                            if not (parsed.get("valid") and parsed.get("group") == "power"):
                                continue
                            channel_id = str(parsed.get("identifier") or "")
                            if channel_id and channel_id not in roi_set:
                                continue
                            if str(parsed.get("segment") or "") == seg_name and str(parsed.get("band") or "") == band:
                                cols.append(c)
                        
                        if not cols:
                            cell_data[col_idx] = None
                            continue
                        
                        val_series = power_df[cols].mean(axis=1)
                        v1 = val_series[m1].dropna().values
                        v2 = val_series[m2].dropna().values
                        cell_data[col_idx] = {"v1": v1, "v2": v2}
                        
                        # Map band to col_idx for q-value lookup
                        if band in qvalues:
                            p, q, d, sig = qvalues[band]
                            qvalues[col_idx] = (p, q, d, sig)
                else:
                    # Compute on-the-fly
                    for col_idx, band in enumerate(bands):
                        cols = []
                        for c in power_df.columns:
                            parsed = NamingSchema.parse(str(c))
                            if not (parsed.get("valid") and parsed.get("group") == "power"):
                                continue
                            channel_id = str(parsed.get("identifier") or "")
                            if channel_id and channel_id not in roi_set:
                                continue
                            if str(parsed.get("segment") or "") == seg_name and str(parsed.get("band") or "") == band:
                                cols.append(c)
                        
                        if not cols:
                            cell_data[col_idx] = None
                            continue
                        
                        val_series = power_df[cols].mean(axis=1)
                        v1 = val_series[m1].dropna().values
                        v2 = val_series[m2].dropna().values
                        
                        cell_data[col_idx] = {"v1": v1, "v2": v2}
                        
                        if len(v1) >= 3 and len(v2) >= 3:
                            from scipy.stats import mannwhitneyu
                            try:
                                _, p = mannwhitneyu(v1, v2, alternative="two-sided")
                                diff = np.mean(v2) - np.mean(v1)
                                pooled_std = np.sqrt(((len(v1)-1)*np.var(v1, ddof=1) + (len(v2)-1)*np.var(v2, ddof=1)) / (len(v1)+len(v2)-2))
                                d = diff / pooled_std if pooled_std > 0 else 0
                                all_pvals.append(p)
                                pvalue_keys.append((col_idx, p, d))
                            except Exception:
                                pass
                    
                    qvalues = {}
                    n_significant = 0
                    if all_pvals:
                        rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
                        for i, (key, p, d) in enumerate(pvalue_keys):
                            qvalues[key] = (p, qvals[i], d, rejected[i])
                        n_significant = int(np.sum(rejected))
                
                fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 5), squeeze=False)
                
                for col_idx, band in enumerate(bands):
                    ax = axes.flatten()[col_idx]
                    data = cell_data.get(col_idx)
                    
                    if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                        ax.set_xticks([])
                        continue
                    
                    v1, v2 = data["v1"], data["v2"]
                    
                    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=0.4, patch_artist=True)
                    bp["boxes"][0].set_facecolor(segment_colors["v1"])
                    bp["boxes"][0].set_alpha(0.6)
                    bp["boxes"][1].set_facecolor(segment_colors["v2"])
                    bp["boxes"][1].set_alpha(0.6)
                    
                    ax.scatter(np.random.uniform(-0.08, 0.08, len(v1)), v1, c=segment_colors["v1"], alpha=0.3, s=6)
                    ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(v2)), v2, c=segment_colors["v2"], alpha=0.3, s=6)
                    
                    all_vals = np.concatenate([v1, v2])
                    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                    yrange = ymax - ymin if ymax > ymin else 0.1
                    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
                    
                    if col_idx in qvalues:
                        _, q, d, sig = qvalues[col_idx]
                        sig_marker = "†" if sig else ""
                        sig_color = "#d62728" if sig else "#333333"
                        ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", xy=(0.5, ymax + 0.05 * yrange),
                                   ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                                   fontweight="bold" if sig else "normal")
                    
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label1, label2], fontsize=9)
                    ax.set_title(band.capitalize(), fontweight="bold", color=band_colors[band])
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                
                n_trials = len(power_df)
                roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
                n_tests = len(qvalues)
                
                stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
                title = (f"Band Power: {label1} vs {label2} (Column Comparison)\n"
                         f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | {stats_source} | "
                         f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                
                plt.tight_layout()
                
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                filename = f"sub-{subject}_power_by_condition{suffix}_column"
                
                from eeg_pipeline.plotting.io.figures import save_fig
                save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                         bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                plt.close(fig)
                
                if logger:
                    logger.info(f"Saved power column comparison for ROI {roi_display} ({n_significant}/{n_tests} FDR significant, {stats_source})")





def _setup_subplot_grid(n_items: int, n_cols: int = 2, config: Any = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a subplot grid for multiple plots.
    
    Args:
        n_items: Number of subplots needed
        n_cols: Number of columns (default: 2)
        config: Configuration object
    
    Returns:
        Tuple of (figure, list of axes)
    """
    plot_cfg = get_plot_config(config)
    n_rows = (n_items + n_cols - 1) // n_cols
    width_per_col = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_col", 6.0))
    height_per_row = float(plot_cfg.plot_type_configs.get("power", {}).get("height_per_row", 4.0))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_col * n_cols, height_per_row * n_rows))
    
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


def _get_active_window(config: Any) -> List[float]:
    """Get active window from config."""
    from eeg_pipeline.utils.config.loader import get_config_value
    return get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])


def _get_plotting_tfr_baseline_window(config: Any) -> tuple[float, float]:
    """Resolve baseline window for plotting.

    Plotting can override the baseline window without changing the analysis
    baseline via `plotting.tfr.default_baseline_window`.
    """
    from eeg_pipeline.utils.config.loader import get_config_value

    override = get_config_value(config, "plotting.tfr.default_baseline_window", None)
    if isinstance(override, (list, tuple)) and len(override) == 2:
        try:
            return float(override[0]), float(override[1])
        except (TypeError, ValueError):
            pass

    baseline = get_config_value(config, "time_frequency_analysis.baseline_window", [-3.0, -0.5])
    if isinstance(baseline, (list, tuple)) and len(baseline) == 2:
        try:
            return float(baseline[0]), float(baseline[1])
        except (TypeError, ValueError):
            pass

    return -3.0, -0.5


def _crop_tfr_to_active(tfr: Any, active_window: List[float], logger: logging.Logger) -> Optional[Any]:
    """Crop TFR to active window.
    
    Args:
        tfr: TFR object to crop
        active_window: List of [start, end] times
        logger: Logger instance
    
    Returns:
        Cropped TFR or None if window is invalid
    """
    times = np.asarray(tfr.times)
    active_start = float(active_window[0])
    active_end = float(active_window[1])
    tmin = max(times.min(), active_start)
    tmax = min(times.max(), active_end)
    
    if tmax <= tmin:
        logger.warning("Invalid active window; skipping PSD")
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
        
    temp_col = find_temperature_column_in_events(events_df, config=None)
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

    power_cols_by_band = get_power_columns_by_band(pow_df, bands=[str(b) for b in bands])
        
    band_means = []
    channel_names = []
    valid_bands = []
    
    for band in bands:
        band = str(band)
        band_cols = [
            c for c in power_cols_by_band.get(band, [])
            if NamingSchema.parse(str(c)).get("scope") == "ch"
        ]
        if band_cols:
            band_data = pow_df[band_cols].mean(axis=0)
            band_means.append(band_data.values)
            valid_bands.append(band)
            if not channel_names:
                extracted: List[str] = []
                for col in band_cols:
                    parsed = NamingSchema.parse(str(col))
                    if parsed.get("valid") and parsed.get("group") == "power":
                        ident = parsed.get("identifier")
                        if ident:
                            extracted.append(str(ident))
                            continue
                    extracted.append(str(col))
                channel_names = extracted
    
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
        data_min = float(np.nanmin(heatmap_data))
        data_max = float(np.nanmax(heatmap_data))
        vmax_abs = max(abs(data_min), abs(data_max))
        if vmin is None:
            vmin = -vmax_abs
        if vmax is None:
            vmax = vmax_abs
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=plot_cfg.font.small)
    ax.set_yticks(range(len(valid_bands)))
    ax.set_yticklabels([b.capitalize() for b in valid_bands], fontsize=plot_cfg.font.medium)
    n_trials = int(len(pow_df))
    ax.set_title(
        f"Mean Power per Channel and Band (n={n_trials} trials)\n"
        "Baseline-normalized log10(power/baseline)",
        fontsize=plot_cfg.font.figure_title,
    )
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
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    plot_cfg = get_plot_config(config)
    
    n_epochs = int(tfr_raw.data.shape[0])
    n_channels = int(tfr_raw.data.shape[1])
    
    fig_size = plot_cfg.get_figure_size("wide", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    for band in bands:
        if band not in features_freq_bands:
            logger.warning(f"Band '{band}' not in config; skipping time course.")
            continue
        
        fmin, fmax = features_freq_bands[band]
        freq_mask = (tfr_raw.freqs >= fmin) & (tfr_raw.freqs <= fmax)
        if not freq_mask.any():
            logger.warning(f"No frequencies found for {band} band ({fmin}-{fmax} Hz)")
            continue
        
        epoch_time_courses = tfr_raw.data[:, :, freq_mask, :].mean(axis=(1, 2))
        mean_tc = np.nanmean(epoch_time_courses, axis=0)
        if n_epochs >= 2:
            sem_tc = np.nanstd(epoch_time_courses, axis=0, ddof=1) / np.sqrt(n_epochs)
        else:
            sem_tc = np.full_like(mean_tc, np.nan, dtype=float)

        color = get_band_color(band, config)
        ax.plot(times, mean_tc, linewidth=2, color=color, label=band.capitalize())
        ax.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc, color=color, alpha=0.15, linewidth=0)
    
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
        f"Baseline: [{b_start:.2f}, {b_end:.2f}]s | "
        "Shading: ±SEM across trials"
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
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    
    trial_counts = []
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        n_trials_temp = int(temp_mask.sum())
        if n_trials_temp < 1:
            continue
        trial_counts.append(f"{temp:.0f}°C: n={n_trials_temp}")
            
        tfr_temp_avg = tfr_epochs[temp_mask].average()
        apply_baseline_and_crop(tfr_temp_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_temp_win = _crop_tfr_to_active(tfr_temp_avg, active_window, logger)
        
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
        f"Active: [{active_window[0]:.1f}, {active_window[1]:.1f}]s | "
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
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    
    if events_df is not None and not events_df.empty:
        temp_col = None
        if config is None:
            logger.warning("Config is required to identify temperature column; skipping temperature-based PSD")
        else:
            temp_col = find_temperature_column_in_events(events_df, config)
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
    tfr_win = _crop_tfr_to_active(tfr_avg, active_window, logger)
    
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
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.utils.config.loader import get_config_value

    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)

    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", True) # Default to True for this plot
    comp_mask_info = None
    if compare_cols:
        comp_mask_info = extract_comparison_mask(events_df, config)
    
    if comp_mask_info:
        m1, m2, label1, label2 = comp_mask_info
        v1_mask, v2_mask = m1, m2
    else:
        # Fallback for PSD might still want pain if nothing defined, 
        # but let's be strict if that's what's requested
        logger.warning("No valid comparison defined for PSD comparison")
        return

    if len(tfr) != len(events_df):
        raise ValueError(
            f"TFR ({len(tfr)} epochs) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )
    
    if v1_mask.sum() < 1 or v2_mask.sum() < 1:
        logger.warning(f"Insufficient trials for {label1} vs {label2} comparison")
        return
    
    n1, n2 = int(v1_mask.sum()), int(v2_mask.sum())
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mask, label, color, n_trials in [
        (v1_mask, label1, 'steelblue', n1),
        (v2_mask, label2, 'orangered', n2)
    ]:
        if mask.sum() < 1:
            continue
            
        tfr_cond_avg = tfr[mask].average()
        apply_baseline_and_crop(tfr_cond_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_cond_win = _crop_tfr_to_active(tfr_cond_avg, active_window, logger)
        
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
        f"Active: [{active_window[0]:.1f}, {active_window[1]:.1f}]s | "
        f"Total: n={len(tfr)} trials"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    comp_type = "Window Comparison" if compare_cols else "Comparison"
    fig.suptitle(f"PSD Comparison: {label1} vs {label2} ({comp_type})", fontsize=plot_cfg.font.figure_title, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = save_dir / f'sub-{subject}_power_spectral_density_condition'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info(f"Saved PSD comparison plot ({label1} vs {label2})")


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
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    from eeg_pipeline.utils.config.loader import get_config_value
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    freq_bands = get_config_value(config, "time_frequency_analysis.bands", {})
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


def plot_inter_band_spatial_power_correlation(tfr, subject, save_dir, logger, config):
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    
    times = np.asarray(tfr.times)
    active_window = config.get("time_frequency_analysis.active_window", [3.0, 10.5])
    active_start = float(active_window[0])
    active_end = float(active_window[1])
    tmin_clip = float(max(times.min(), active_start))
    tmax_clip = float(min(times.max(), active_end))
    
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        logger.warning(
            f"Skipping inter-band spatial power correlation: invalid active within data range "
            f"(requested [{active_start}, {active_end}] s, "
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


def plot_power_variability_comprehensive(
    pow_df: pd.DataFrame,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Trial-to-trial power variability with CV, Fano factor, and normality tests.
    
    Creates a comprehensive variability dashboard showing:
    - CV (Coefficient of Variation) per band
    - Fano factor per band (for positive-shifted data)
    - Normality test results (Shapiro-Wilk)
    - Distribution shapes with normality indicators
    """
    if pow_df is None or pow_df.empty:
        return
    
    from .utils import compute_variability_metrics, test_normality
    
    plot_cfg = get_plot_config(config)
    n_bands = len(bands)
    power_cols_by_band = get_power_columns_by_band(pow_df, bands=[str(b) for b in bands])
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, n_bands, height_ratios=[1.5, 1, 1], hspace=0.35, wspace=0.3)
    
    variability_data = []
    
    for i, band in enumerate(bands):
        band_str = str(band)
        cols = power_cols_by_band.get(band_str, [])
        
        if not cols:
            continue
        
        trial_means = pow_df[cols].mean(axis=1).dropna().values
        
        if len(trial_means) < 5:
            continue
        
        metrics = compute_variability_metrics(trial_means)
        is_normal, p_norm, norm_interp = test_normality(trial_means)
        
        variability_data.append({
            "band": band_str,
            "cv": metrics["cv"],
            "fano": metrics["fano"],
            "std": metrics["std"],
            "iqr": metrics["iqr"],
            "mad": metrics["mad"],
            "is_normal": is_normal,
            "p_norm": p_norm,
            "norm_interp": norm_interp,
            "n_trials": len(trial_means),
            "values": trial_means,
        })
    
    if not variability_data:
        plt.close(fig)
        return
    
    ax_cv = fig.add_subplot(gs[0, :len(variability_data)//2])
    ax_fano = fig.add_subplot(gs[0, len(variability_data)//2:])
    
    band_names = [d["band"] for d in variability_data]
    cvs = [d["cv"] for d in variability_data]
    fanos = [d["fano"] for d in variability_data]
    colors = [get_band_color(b, config) for b in band_names]
    
    x = np.arange(len(band_names))
    bars_cv = ax_cv.bar(x, cvs, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels([b.upper() for b in band_names], fontsize=plot_cfg.font.medium)
    ax_cv.set_ylabel("Coefficient of Variation\n(std / |mean|)", fontsize=plot_cfg.font.label)
    ax_cv.set_title("Trial-to-Trial Variability: CV", fontsize=plot_cfg.font.title, fontweight="bold")
    ax_cv.spines["top"].set_visible(False)
    ax_cv.spines["right"].set_visible(False)
    ax_cv.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="CV=0.5 (moderate)")
    ax_cv.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="CV=1.0 (high)")
    ax_cv.legend(fontsize=plot_cfg.font.small, loc="upper right")
    
    for j, (bar, cv_val) in enumerate(zip(bars_cv, cvs)):
        if np.isfinite(cv_val):
            ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f"{cv_val:.2f}", ha="center", va="bottom", fontsize=plot_cfg.font.small)
    
    valid_fanos = [(i, f) for i, f in enumerate(fanos) if np.isfinite(f) and f > 0]
    if valid_fanos:
        fano_x = [v[0] for v in valid_fanos]
        fano_y = [v[1] for v in valid_fanos]
        fano_colors = [colors[i] for i in fano_x]
        bars_fano = ax_fano.bar(range(len(fano_x)), fano_y, color=fano_colors, alpha=0.8, 
                                edgecolor="white", linewidth=1.5)
        ax_fano.set_xticks(range(len(fano_x)))
        ax_fano.set_xticklabels([band_names[i].upper() for i in fano_x], fontsize=plot_cfg.font.medium)
        ax_fano.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Fano=1 (Poisson)")
        ax_fano.legend(fontsize=plot_cfg.font.small, loc="upper right")
        
        for bar, fano_val in zip(bars_fano, fano_y):
            ax_fano.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{fano_val:.2f}", ha="center", va="bottom", fontsize=plot_cfg.font.small)
    else:
        ax_fano.text(0.5, 0.5, "Fano factor requires\npositive mean values", 
                    ha="center", va="center", transform=ax_fano.transAxes, fontsize=plot_cfg.font.medium)
    
    ax_fano.set_ylabel("Fano Factor\n(var / mean)", fontsize=plot_cfg.font.label)
    ax_fano.set_title("Trial-to-Trial Variability: Fano Factor", fontsize=plot_cfg.font.title, fontweight="bold")
    ax_fano.spines["top"].set_visible(False)
    ax_fano.spines["right"].set_visible(False)
    
    for i, d in enumerate(variability_data):
        if i >= n_bands:
            break
        ax_dist = fig.add_subplot(gs[1, i])
        
        vals = d["values"]
        color = get_band_color(d["band"], config)
        norm_color = "#22C55E" if d["is_normal"] else "#EF4444"
        
        ax_dist.hist(vals, bins=15, color=color, alpha=0.7, edgecolor="white", density=True)
        ax_dist.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1.5, label="Mean")
        ax_dist.axvline(np.median(vals), color="blue", linestyle=":", linewidth=1.5, label="Median")
        
        ax_dist.set_xlabel("Power (dB)", fontsize=plot_cfg.font.small)
        if i == 0:
            ax_dist.set_ylabel("Density", fontsize=plot_cfg.font.small)
        
        norm_symbol = "✓" if d["is_normal"] else "✗"
        ax_dist.set_title(f"{d['band'].upper()} {norm_symbol}", fontsize=plot_cfg.font.medium, 
                         fontweight="bold", color=norm_color)
        
        ax_dist.text(0.95, 0.95, f"W p={d['p_norm']:.3f}" if np.isfinite(d['p_norm']) else "W p=N/A",
                    transform=ax_dist.transAxes, ha="right", va="top", fontsize=plot_cfg.font.small,
                    color=norm_color, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)
    
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    
    table_data = []
    for d in variability_data:
        norm_status = "Normal" if d["is_normal"] else "Non-normal"
        table_data.append([
            d["band"].upper(),
            f"{d['n_trials']}",
            f"{d['cv']:.3f}" if np.isfinite(d['cv']) else "N/A",
            f"{d['fano']:.3f}" if np.isfinite(d['fano']) else "N/A",
            f"{d['std']:.3f}" if np.isfinite(d['std']) else "N/A",
            f"{d['iqr']:.3f}" if np.isfinite(d['iqr']) else "N/A",
            f"{d['mad']:.3f}" if np.isfinite(d['mad']) else "N/A",
            norm_status,
        ])
    
    col_labels = ["Band", "N", "CV", "Fano", "SD", "IQR", "MAD", "Normality"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(plot_cfg.font.small)
    table.scale(1.2, 1.5)
    
    for i, d in enumerate(variability_data):
        cell = table[(i + 1, 7)]
        cell.set_facecolor("#d4edda" if d["is_normal"] else "#f8d7da")
    
    fig.suptitle(f"Power Variability Analysis (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.98)
    
    footer = f"n={len(pow_df)} trials | Shapiro-Wilk normality test α=0.05 | ✓=Normal, ✗=Non-normal"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, save_dir / f"sub-{subject}_power_variability_comprehensive",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info("Saved comprehensive power variability plot")


def plot_cross_frequency_power_correlation(
    pow_df: pd.DataFrame,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Cross-frequency power correlation matrix with significance testing.
    
    Shows how power in different frequency bands co-varies across trials.
    Includes raw p-values and FDR-corrected significance markers.
    """
    if pow_df is None or pow_df.empty:
        return
    
    from .utils import apply_fdr_correction, get_fdr_alpha, test_normality
    from scipy.stats import spearmanr
    
    plot_cfg = get_plot_config(config)
    
    band_power = {}
    for band in bands:
        band_str = str(band)
        cols = [c for c in pow_df.columns 
                if c.startswith("power_") and f"_{band_str}_" in c and "_ch_" in c]
        if cols:
            vals = pow_df[cols].mean(axis=1).dropna().values
            if len(vals) >= 5:
                band_power[band_str] = vals
    
    valid_bands = list(band_power.keys())
    n_bands = len(valid_bands)
    
    if n_bands < 2:
        return
    
    corr_matrix = np.zeros((n_bands, n_bands))
    pval_matrix = np.zeros((n_bands, n_bands))
    
    for i, band1 in enumerate(valid_bands):
        for j, band2 in enumerate(valid_bands):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                vals1 = band_power[band1]
                vals2 = band_power[band2]
                min_len = min(len(vals1), len(vals2))
                r, p = spearmanr(vals1[:min_len], vals2[:min_len])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p
    
    upper_tri_idx = np.triu_indices(n_bands, k=1)
    pvals_upper = pval_matrix[upper_tri_idx]
    
    alpha_fdr = get_fdr_alpha(config)

    if len(pvals_upper) > 0:
        rejected, qvals, alpha_fdr_arr = apply_fdr_correction(list(pvals_upper), config=config)
        try:
            alpha_fdr = float(np.asarray(alpha_fdr_arr).ravel()[0])
        except Exception:
            alpha_fdr = get_fdr_alpha(config)
        qval_matrix = np.zeros((n_bands, n_bands))
        qval_matrix[upper_tri_idx] = qvals
        qval_matrix = qval_matrix + qval_matrix.T
        np.fill_diagonal(qval_matrix, 0)
        n_significant = int(np.sum(rejected))
    else:
        qval_matrix = pval_matrix.copy()
        n_significant = 0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    sig_mask = np.isfinite(qval_matrix) & (qval_matrix < alpha_fdr)
    sig_mask |= np.eye(n_bands, dtype=bool)
    corr_sig_only = np.where(sig_mask, corr_matrix, np.nan)
    im1 = ax1.imshow(corr_sig_only, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax1.set_xticks(range(n_bands))
    ax1.set_yticks(range(n_bands))
    ax1.set_xticklabels([b.capitalize() for b in valid_bands], rotation=45, ha="right", 
                        fontsize=plot_cfg.font.medium)
    ax1.set_yticklabels([b.capitalize() for b in valid_bands], fontsize=plot_cfg.font.medium)
    ax1.set_title("Spearman Correlation", fontsize=plot_cfg.font.title, fontweight="bold")
    
    for i in range(n_bands):
        for j in range(n_bands):
            r = corr_matrix[i, j]
            p = pval_matrix[i, j]
            q = qval_matrix[i, j]
            
            text_color = "white" if abs(r) > 0.5 else "black"
            
            if i != j:
                sig_marker = "†" if q < 0.05 else ("*" if p < 0.05 else "")
                text = f"{r:.2f}{sig_marker}"
            else:
                text = "1.00"
            
            ax1.text(j, i, text, ha="center", va="center", color=text_color, 
                    fontsize=plot_cfg.font.small, fontweight="bold" if q < 0.05 else "normal")
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Spearman ρ", fontsize=plot_cfg.font.label)
    
    ax2 = axes[1]
    log_pvals = -np.log10(pval_matrix + 1e-10)
    np.fill_diagonal(log_pvals, 0)
    
    im2 = ax2.imshow(log_pvals, cmap="YlOrRd", vmin=0, vmax=max(3, np.max(log_pvals)), aspect="equal")
    ax2.set_xticks(range(n_bands))
    ax2.set_yticks(range(n_bands))
    ax2.set_xticklabels([b.capitalize() for b in valid_bands], rotation=45, ha="right",
                        fontsize=plot_cfg.font.medium)
    ax2.set_yticklabels([b.capitalize() for b in valid_bands], fontsize=plot_cfg.font.medium)
    ax2.set_title("Significance (-log₁₀ p)", fontsize=plot_cfg.font.title, fontweight="bold")
    
    for i in range(n_bands):
        for j in range(n_bands):
            if i != j:
                p = pval_matrix[i, j]
                q = qval_matrix[i, j]
                
                if p < 0.001:
                    p_text = "<.001"
                else:
                    p_text = f"{p:.3f}"
                
                fdr_marker = "†" if q < 0.05 else ""
                text_color = "white" if log_pvals[i, j] > 1.5 else "black"
                
                ax2.text(j, i, f"{p_text}{fdr_marker}", ha="center", va="center", 
                        color=text_color, fontsize=plot_cfg.font.small - 1)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("-log₁₀(p)", fontsize=plot_cfg.font.label)
    
    ax2.axhline(-0.5, color="gray", linewidth=0.5)
    ax2.axhline(n_bands - 0.5, color="gray", linewidth=0.5)
    ax2.axvline(-0.5, color="gray", linewidth=0.5)
    ax2.axvline(n_bands - 0.5, color="gray", linewidth=0.5)
    
    fig.suptitle(f"Cross-Frequency Power Correlations (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.98)
    
    n_tests = len(pvals_upper)
    footer = (f"n={len(pow_df)} trials | {n_tests} tests | FDR-BH α=0.05 | "
              f"{n_significant} significant | *p<.05 uncorrected, †q<.05 FDR")
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, save_dir / f"sub-{subject}_cross_frequency_power_correlation",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved cross-frequency power correlation ({n_significant}/{n_tests} FDR significant)")


def plot_power_topomaps_from_df(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot spatial distribution of power using DataFrame values (aggregated)."""
    if pow_df is None or epochs_info is None:
        return

    plot_cfg = get_plot_config(config)
    power_cols_by_band = get_power_columns_by_band(pow_df, bands=[str(b) for b in bands])
    bands_to_plot = [b for b in bands if power_cols_by_band.get(str(b))]
    if not bands_to_plot: return
    
    fig, axes = _setup_subplot_grid(len(bands_to_plot), n_cols=max(1, min(3, len(bands_to_plot))), config=config)
    # Ensure iterable list for downstream indexing
    if not isinstance(axes, list):
        axes = list(np.ravel(axes))
    
    features_found = False
    
    for i, band in enumerate(bands_to_plot):
        ax = axes[i]
        band_cols = []
        for c in power_cols_by_band.get(str(band), []):
            parsed = NamingSchema.parse(str(c))
            if parsed.get("valid") and parsed.get("group") == "power":
                if str(parsed.get("scope") or "") == "ch":
                    band_cols.append(c)
                continue
        if not band_cols:
            ax.axis('off')
            continue
            
        mean_power = pow_df[band_cols].mean(axis=0)
        
        data_map = {}
        for col, val in mean_power.items():
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid") and parsed.get("group") == "power":
                ident = parsed.get("identifier")
                if ident:
                    data_map[str(ident)] = val
                    continue
        
        data_array = np.full(len(epochs_info.ch_names), np.nan)
        mask = np.zeros(len(epochs_info.ch_names), dtype=bool)
        for ch_idx, ch_name in enumerate(epochs_info.ch_names):
            if ch_name in data_map:
                data_array[ch_idx] = data_map[ch_name]
                mask[ch_idx] = True
        
        if mask.sum() > 3:
            im, _ = plot_topomap(
                data_array[mask], 
                mne.pick_info(epochs_info, np.where(mask)[0]),
                axes=ax, show=False, cmap='viridis', contours=0
            )
            plt.colorbar(im, ax=ax, shrink=0.6)
            features_found = True
        ax.set_title(f"{band.upper()}")
        
    if features_found:
        fig.suptitle(f"Spatial Distribution of Spectral Power (sub-{subject})", y=1.05, fontweight='bold')
        plt.tight_layout()
        save_fig(fig, save_dir / f"sub-{subject}_power_topomaps_agg", 
                 formats=plot_cfg.formats, dpi=plot_cfg.dpi)
    else:
        plt.close(fig)
        return

    if logger:
        logger.info("Saved aggregated power topomaps")


def plot_spectral_slope_topomap(
    aperiodic_df: pd.DataFrame,
    epochs_info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Spectral slope (1/f) topomap showing spatial distribution.
    
    Answers: "Where is the 1/f slope steepest/shallowest?"
    """
    if aperiodic_df is None or epochs_info is None:
        return
    
    plot_cfg = get_plot_config(config)
    
    slope_cols = [c for c in aperiodic_df.columns if c.startswith("aper_slope_")]
    
    if not slope_cols:
        return
    
    ch_slopes = {}
    for col in slope_cols:
        parsed = NamingSchema.parse(col)
        if parsed["valid"] and "identifier" in parsed:
            ch_name = parsed["identifier"]
            ch_slopes[ch_name] = aperiodic_df[col].mean()
    
    if not ch_slopes:
        return
    
    data_array = np.full(len(epochs_info.ch_names), np.nan)
    mask = np.zeros(len(epochs_info.ch_names), dtype=bool)
    
    for ch_idx, ch_name in enumerate(epochs_info.ch_names):
        if ch_name in ch_slopes:
            data_array[ch_idx] = ch_slopes[ch_name]
            mask[ch_idx] = True
    
    if mask.sum() < 3:
        return
    
    fig_size = plot_cfg.get_figure_size("wide", plot_type="features")
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    ax1 = axes[0]
    valid_data = data_array[mask]
    valid_info = mne.pick_info(epochs_info, np.where(mask)[0])
    
    im, _ = plot_topomap(
        valid_data, valid_info,
        axes=ax1, show=False, cmap="RdBu_r", contours=6
    )
    plt.colorbar(im, ax=ax1, shrink=0.6, label="Slope")
    ax1.set_title("1/f Spectral Slope", fontweight="bold")
    
    ax2 = axes[1]
    ax2.hist(valid_data, bins=20, color=plot_cfg.style.colors.blue, alpha=0.7, edgecolor="white")
    ax2.axvline(np.median(valid_data), color="red", linestyle="--", linewidth=2, 
                label=f"Median: {np.median(valid_data):.2f}")
    ax2.set_xlabel("Slope")
    ax2.set_ylabel("Count")
    ax2.set_title("Slope Distribution", fontweight="bold")
    ax2.legend()
    
    fig.suptitle(f"Spectral Slope (1/f) Spatial Distribution (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    footer = f"n={len(aperiodic_df)} trials | Steeper slope = more 1/f noise"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_fig(fig, save_dir / f"sub-{subject}_spectral_slope_topomap",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info("Saved spectral slope topomap")


def plot_feature_importance_ranking(
    features_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    top_n: int = 30,
) -> None:
    """Feature importance ranking based on variance.
    
    Answers: "Which features have the most information content?"
    """
    if features_df is None or features_df.empty:
        return
    
    plot_cfg = get_plot_config(config)
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return
    
    variances = {}
    for col in numeric_cols:
        vals = features_df[col].dropna().values
        if len(vals) >= 5:
            variances[col] = np.var(vals)
    
    if not variances:
        return
    
    sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_features) * 0.25)))
    
    feature_names = [f[0][:30] + "..." if len(f[0]) > 30 else f[0] for f in sorted_features]
    feature_vars = [f[1] for f in sorted_features]
    
    from eeg_pipeline.domain.features.naming import NamingSchema

    group_map = {
        "power": "power",
        "conn": "connectivity",
        "aperiodic": "aperiodic",
        "erds": "erds",
        "itpc": "itpc",
        "pac": "pac",
        "comp": "complexity",
        "quality": "quality",
        "spectral": "spectral",
        "temporal": "temporal",
        "ratio": "ratios",
        "asym": "asymmetry",
        "bursts": "bursts",
    }

    feature_types = []
    for f in sorted_features:
        name = f[0]
        parsed = NamingSchema.parse(str(name))
        if parsed.get("valid"):
            feature_types.append(group_map.get(parsed.get("group"), "other"))
            continue
        if name.startswith("power_"):
            feature_types.append("power")
        elif name.startswith("conn_") or "wpli_" in name or "aec_" in name:
            feature_types.append("connectivity")
        elif name.startswith("aper_") or "aperiodic" in name:
            feature_types.append("aperiodic")
        elif name.startswith("itpc_"):
            feature_types.append("itpc")
        elif name.startswith("pac_"):
            feature_types.append("pac")
        else:
            feature_types.append("other")
    
    type_colors = {
        "power": "#3B82F6",
        "connectivity": "#22C55E",
        "aperiodic": "#8B5CF6",
        "erds": "#F97316",
        "itpc": "#EC4899",
        "pac": "#14B8A6",
        "complexity": "#F59E0B",
        "quality": "#0EA5E9",
        "spectral": "#10B981",
        "temporal": "#6366F1",
        "ratios": "#A855F7",
        "asymmetry": "#F43F5E",
        "bursts": "#D97706",
        "other": "#6B7280",
    }
    colors = [type_colors.get(t, "#6B7280") for t in feature_types]
    
    y_pos = range(len(sorted_features))
    ax.barh(y_pos, feature_vars, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Variance")
    ax.set_title("Feature Importance (by Variance)", fontweight="bold")
    ax.invert_yaxis()
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.capitalize()) 
                      for t, c in type_colors.items() if t in feature_types]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    fig.suptitle(f"Feature Importance Ranking (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.98)
    
    footer = f"n={len(features_df)} trials | Top {len(sorted_features)} features by variance"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, save_dir / f"sub-{subject}_feature_importance_ranking",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved feature importance ranking (top {len(sorted_features)})")


def plot_band_power_topomaps(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str = "active",
) -> None:
    """Band power topomaps showing spatial distribution per frequency band.
    
    Creates MNE topomaps for each frequency band.
    """
    if pow_df is None or epochs_info is None:
        return
    
    plot_cfg = get_plot_config(config)
    
    valid_bands = []
    band_data = {}
    
    for band in bands:
        cols = [c for c in pow_df.columns 
                if c.startswith("power_") and f"_{segment}_" in c and f"_{band}_" in c and "_ch_" in c]
        
        if not cols:
            continue
        
        ch_power = {}
        for col in cols:
            parsed = NamingSchema.parse(col)
            if parsed["valid"] and "identifier" in parsed:
                ch_name = parsed["identifier"]
                ch_power[ch_name] = pow_df[col].mean()
        
        if ch_power:
            valid_bands.append(band)
            band_data[band] = ch_power
    
    if not valid_bands:
        return
    
    n_bands = len(valid_bands)
    width_per_band = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_band", 3.5))
    fig, axes = plt.subplots(1, n_bands, figsize=(width_per_band * n_bands, 4))
    if n_bands == 1:
        axes = [axes]
    
    for i, band in enumerate(valid_bands):
        ax = axes[i]
        ch_power = band_data[band]
        
        data_array = np.full(len(epochs_info.ch_names), np.nan)
        mask = np.zeros(len(epochs_info.ch_names), dtype=bool)
        
        for ch_idx, ch_name in enumerate(epochs_info.ch_names):
            if ch_name in ch_power:
                data_array[ch_idx] = ch_power[ch_name]
                mask[ch_idx] = True
        
        if mask.sum() > 3:
            valid_data = data_array[mask]
            valid_info = mne.pick_info(epochs_info, np.where(mask)[0])
            
            im, _ = plot_topomap(
                valid_data, valid_info,
                axes=ax, show=False, cmap="RdBu_r", contours=6
            )
            plt.colorbar(im, ax=ax, shrink=0.6, label="dB")
        
        band_color = get_band_color(band, config)
        ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)
    
    fig.suptitle(f"Band Power Topomaps - {segment.capitalize()} (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    footer = f"n={len(pow_df)} trials | Segment: {segment} | Values: mean dB (log ratio)"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_fig(fig, save_dir / f"sub-{subject}_band_power_topomaps_{segment}",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved band power topomaps ({segment})")
