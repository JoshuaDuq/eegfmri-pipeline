from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.plotting.core.utils import get_font_sizes
from eeg_pipeline.plotting.core.colorbars import create_difference_colorbar
from eeg_pipeline.plotting.core.annotations import find_annotation_x_position, get_sig_marker_text
from eeg_pipeline.utils.analysis.tfr import (
    build_rois_from_info,
    build_roi_channel_mask,
)
from eeg_pipeline.utils.io.general import (
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    find_connectivity_features_path,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    save_fig,
    read_tsv,
    get_behavior_footer as _get_behavior_footer,
    get_subject_logger,
    get_default_logger as _get_default_logger,
    get_default_config as _get_default_config,
    log_if_present as _log_if_present,
)
from eeg_pipeline.utils.config.loader import load_settings, get_frequency_band_names
from eeg_pipeline.utils.analysis.stats import (
    compute_correlation_vmax,
    compute_band_correlations,
    compute_connectivity_correlations,
)
from eeg_pipeline.utils.data.loading import prepare_topomap_correlation_data


###################################################################
# Helper Functions
###################################################################




def _add_correlation_roi_annotations(
    ax, corr_data, p_uncorr, p_fdr, info, config=None, roi_map=None, fdr_alpha=0.05
):
    if config is None and roi_map is None:
        return
    
    if roi_map is None and config is not None:
        roi_map = build_rois_from_info(info, config=config)
    if not roi_map:
        return
    
    ch_names = info["ch_names"]
    if len(corr_data) != len(ch_names):
        return
    
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    font_sizes = get_font_sizes(plot_cfg)
    annotation_fontsize = font_sizes["annotation"]
    
    x_pos_ax = find_annotation_x_position(ax, plot_cfg)
    annotation_y_start = tfr_config.get("annotation_y_start", 0.98)
    y_pos_ax = annotation_y_start
    annotation_line_height = tfr_config.get("annotation_line_height", 0.045)
    annotation_min_spacing = tfr_config.get("annotation_min_spacing", 0.03)
    annotation_spacing_multiplier = tfr_config.get("annotation_spacing_multiplier", 0.3)
    
    annotations = []
    for roi, roi_chs in roi_map.items():
        mask_vec = build_roi_channel_mask(ch_names, roi_chs)
        if not mask_vec.any():
            continue
        
        roi_corrs = corr_data[mask_vec]
        roi_corrs_finite = roi_corrs[np.isfinite(roi_corrs)]
        if len(roi_corrs_finite) == 0:
            continue
        
        mean_corr = np.nanmean(roi_corrs_finite)
        
        roi_p_uncorr = None
        roi_p_fdr = None
        if p_uncorr is not None:
            roi_p_uncorr_vals = p_uncorr[mask_vec]
            roi_p_uncorr_finite = roi_p_uncorr_vals[np.isfinite(roi_p_uncorr_vals)]
            if len(roi_p_uncorr_finite) > 0:
                roi_p_uncorr = np.nanmin(roi_p_uncorr_finite)
        
        if p_fdr is not None:
            roi_p_fdr_vals = p_fdr[mask_vec]
            roi_p_fdr_finite = roi_p_fdr_vals[np.isfinite(roi_p_fdr_vals)]
            if len(roi_p_fdr_finite) > 0:
                roi_p_fdr = np.nanmin(roi_p_fdr_finite)
        
        annotations.append((roi, mean_corr, roi_p_uncorr, roi_p_fdr))
    
    for i, (roi, mean_corr, roi_p_uncorr, roi_p_fdr) in enumerate(annotations):
        if not np.isfinite(mean_corr):
            continue
        
        label = f"{roi}: r={mean_corr:+.2f}"
        
        if roi_p_fdr is not None and np.isfinite(roi_p_fdr) and roi_p_fdr < fdr_alpha:
            label += f" (q={roi_p_fdr:.3f})"
        elif roi_p_uncorr is not None and np.isfinite(roi_p_uncorr) and roi_p_uncorr < 0.05:
            label += f" (p={roi_p_uncorr:.3f})"
        
        ax.text(x_pos_ax, y_pos_ax, label, transform=ax.transAxes, 
               ha="left", va="top", fontsize=annotation_fontsize)
        
        if i < len(annotations) - 1:
            spacing_ax = annotation_min_spacing + (annotation_line_height * annotation_spacing_multiplier)
            y_pos_ax -= (annotation_line_height + spacing_ax)


def _load_global_fdr_for_temporal_correlations(
    stats_dir: Path,
    use_spearman: bool,
    logger: logging.Logger,
) -> Optional[Dict[Tuple[str, str, float, float, str], bool]]:
    """
    Load global FDR rejection status from consolidated TSV file for temporal correlations.
    
    Returns a dictionary mapping (condition, band, time_start, time_end, channel) -> fdr_reject_global.
    Returns None if TSV file doesn't exist or doesn't have global FDR columns.
    """
    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    tsv_path = stats_dir / f"corr_stats_temporal_all{use_spearman_suffix}.tsv"
    
    if not tsv_path.exists():
        logger.debug(f"TSV file not found for global FDR: {tsv_path}")
        return None
    
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        logger.warning(f"Failed to load TSV for global FDR: {e}")
        return None
    
    if df.empty:
        logger.warning(f"TSV file is empty: {tsv_path}")
        return None
    
    required_cols = ["condition", "band", "time_start", "time_end", "channel", "fdr_reject_global"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"TSV file missing required columns: {missing_cols}")
        return None
    
    global_fdr_map = {}
    n_valid = 0
    n_invalid = 0
    
    for _, row in df.iterrows():
        try:
            condition = str(row["condition"]).strip()
            band = str(row["band"]).strip()
            channel = str(row["channel"]).strip()
            
            if pd.isna(row["time_start"]) or pd.isna(row["time_end"]):
                n_invalid += 1
                continue
            
            time_start = float(row["time_start"])
            time_end = float(row["time_end"])
            
            if pd.isna(row.get("fdr_reject_global")):
                fdr_reject = False
            else:
                fdr_reject = bool(row["fdr_reject_global"])
            
            if not condition or not band or not channel:
                n_invalid += 1
                continue
            
            key = (condition, band, time_start, time_end, channel)
            global_fdr_map[key] = fdr_reject
            n_valid += 1
        except (ValueError, TypeError) as e:
            n_invalid += 1
            logger.debug(f"Skipping invalid row in global FDR TSV: {e}")
            continue
    
    if n_invalid > 0:
        logger.debug(f"Skipped {n_invalid} invalid rows when loading global FDR")
    
    if not global_fdr_map:
        logger.warning(f"No valid global FDR entries found in {tsv_path.name}")
        return None
    
    logger.debug(f"Loaded global FDR for {len(global_fdr_map)} entries from {tsv_path.name}")
    return global_fdr_map


###################################################################
# Temporal Correlation Topomaps
###################################################################


def plot_temporal_correlation_topomaps_by_temperature(
    subject: str,
    task: str,
    plots_dir: Path,
    stats_dir: Path,
    config,
    logger: logging.Logger,
    use_spearman: bool = True,
) -> None:
    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    data_path = stats_dir / f"temporal_correlations_by_temperature{use_spearman_suffix}.npz"
    
    if not data_path.exists():
        logger.warning(f"Temporal correlation data not found: {data_path}")
        return
    
    logger.info("Plotting temporal correlation topomaps by temperature...")
    
    global_fdr_map = _load_global_fdr_for_temporal_correlations(
        stats_dir, use_spearman, logger
    )
    use_global_fdr = global_fdr_map is not None
    if use_global_fdr:
        logger.info("Using global FDR correction for significance masking")
    else:
        logger.info("Using per-analysis FDR correction (global FDR not available)")
    
    data = np.load(data_path, allow_pickle=True)
    info = data.get("info", None)
    if info is None:
        logger.warning("Info not found in data file")
        return
    
    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()
    
    ch_names = data.get("ch_names", None)
    if ch_names is None:
        logger.warning("Channel names not found in data file")
        return
    
    if isinstance(ch_names, np.ndarray):
        ch_names = ch_names.tolist()
    
    temp_keys = [k for k in data.keys() if k.startswith("temp_")]
    if not temp_keys:
        logger.info("No temperature data found in temporal correlation file; skipping temperature topomaps.")
        return
    
    info_ch_names = info["ch_names"]
    if len(ch_names) != len(info_ch_names) or set(ch_names) != set(info_ch_names):
        picks = mne.pick_channels(info_ch_names, include=ch_names, exclude=[])
        info = mne.pick_info(info, picks)
        logger.debug(f"Picked {len(picks)} channels from Info to match data channels")
    
    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    sig_text = get_sig_marker_text(config)
    
    temp_results = {}
    for temp_key in temp_keys:
        temp_str = temp_key.replace("temp_", "")
        temp_results[temp_str] = data[temp_key].item()
    
    band_names = temp_results[list(temp_results.keys())[0]]["band_names"]
    band_ranges = temp_results[list(temp_results.keys())[0]]["band_ranges"]
    window_starts = temp_results[list(temp_results.keys())[0]]["window_starts"]
    window_ends = temp_results[list(temp_results.keys())[0]]["window_ends"]
    n_windows = len(window_starts)
    
    temp_keys_sorted = sorted(temp_results.keys(), key=lambda x: float(x.replace("_", ".")))
    temp_rows = temp_keys_sorted
    n_rows = len(temp_rows)
    
    if n_rows == 0:
        logger.warning("No valid temperature conditions found")
        return
    
    all_corr_data = []
    for temp_str in temp_rows:
        for band_idx in range(len(band_names)):
            corr_data = temp_results[temp_str]["correlations"][band_idx]
            all_corr_data.extend([c for c in corr_data.flatten() if np.isfinite(c)])
    
    vabs_corr = robust_sym_vlim(all_corr_data) if all_corr_data else 0.6
    
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    topomap_config = tfr_config.get("topomap", {})
    tfr_specific = topomap_config.get("tfr_specific", {})
    hspace = tfr_specific.get("hspace", 0.25)
    wspace = tfr_specific.get("wspace", 1.2)
    
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0] if plot_cfg else 10.0
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1] if plot_cfg else 10.0
    
    for band_idx, band_name in enumerate(band_names):
        fmin, fmax = band_ranges[band_idx]
        freq_label = f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)"
        
        fig, axes = plt.subplots(
            n_rows, n_windows, 
            figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows), 
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace}
        )
        
        for row_idx, temp_str in enumerate(temp_rows):
            result = temp_results[temp_str]
            correlations = result["correlations"][band_idx]
            p_values = result["p_values"][band_idx]
            p_corrected = result["p_corrected"][band_idx]
            
            temp_display = temp_str.replace("_", ".")
            if row_idx == 0:
                axes[row_idx, 0].set_ylabel(f"{temp_display}°C\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
            else:
                axes[row_idx, 0].set_ylabel(f"{temp_display}°C", fontsize=font_sizes["ylabel"], labelpad=10)
            
            for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
                if row_idx == 0:
                    time_label = f"{tmin_win:.2f}s"
                    axes[row_idx, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)
                
                corr_data = correlations[col, :]
                p_uncorr = p_values[col, :]
                p_fdr = p_corrected[col, :]
                
                sig_mask_uncorr = (p_uncorr < 0.05) & np.isfinite(p_uncorr)
                
                if use_global_fdr:
                    sig_mask_fdr = np.zeros(len(ch_names), dtype=bool)
                    n_matched = 0
                    for ch_idx, ch_name in enumerate(ch_names):
                        key = (temp_str, band_name, tmin_win, tmax_win, ch_name)
                        if key in global_fdr_map:
                            sig_mask_fdr[ch_idx] = global_fdr_map[key]
                            n_matched += 1
                    if n_matched == 0 and len(ch_names) > 0:
                        logger.debug(
                            f"No global FDR matches for condition={temp_str}, band={band_name}, "
                            f"window=[{tmin_win:.3f}, {tmax_win:.3f}]"
                        )
                else:
                    sig_mask_fdr = (p_fdr < 0.05) & np.isfinite(p_fdr)
                
                plot_topomap_on_ax(
                    axes[row_idx, col], corr_data, info,
                    vmin=-vabs_corr, vmax=+vabs_corr,
                    mask=sig_mask_fdr,
                    mask_params=dict(marker="o", markerfacecolor="green", markeredgecolor="green", markersize=4),
                    config=config
                )
                
                if sig_mask_uncorr.sum() > 0:
                    uncorr_chs = np.where(sig_mask_uncorr & ~sig_mask_fdr)[0]
                    if len(uncorr_chs) > 0:
                        try:
                            from mne.channels.layout import _find_topomap_coords
                            pos = _find_topomap_coords(info, picks=None)
                            axes[row_idx, col].plot(
                                pos[uncorr_chs, 0], pos[uncorr_chs, 1],
                                "o", markerfacecolor="white", markeredgecolor="black",
                                markersize=4, markeredgewidth=1, zorder=10
                            )
                        except Exception:
                            pass
                
                _add_correlation_roi_annotations(
                    axes[row_idx, col], corr_data, p_uncorr, p_fdr, info, config=config, fdr_alpha=0.05
                )
        
        create_difference_colorbar(
            fig, axes, vabs_corr, viz_params["topo_cmap"],
            label="Correlation coefficient"
        )
        
        window_label = f"{window_starts[0]:.1f}–{window_ends[-1]:.1f}s; {n_windows} windows"
        method_name = "Spearman" if use_spearman else "Pearson"
        fig.suptitle(
            f"Temporal correlation topomaps by temperature ({band_name}, {window_label})\n{method_name} correlation, vlim ±{vabs_corr:.2f}{sig_text}\n",
            fontsize=font_sizes["suptitle"], y=0.995
        )
        
        topomap_dir = plots_dir / "topomaps"
        ensure_dir(topomap_dir)
        filename = f"sub-{subject}_temporal_correlations_by_temperature_{band_name}.png"
        save_fig(fig, topomap_dir / filename, formats=plot_cfg.formats if plot_cfg else ["png", "svg"], dpi=plot_cfg.dpi if plot_cfg else None, bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight", pad_inches=plot_cfg.pad_inches if plot_cfg else None, footer=_get_behavior_footer(config))
        plt.close(fig)
    
    logger.info(f"Created temporal correlation topomaps by temperature for {len(band_names)} bands")


def plot_temporal_correlation_topomaps_by_pain(
    subject: str,
    task: str,
    plots_dir: Path,
    stats_dir: Path,
    config,
    logger: logging.Logger,
    use_spearman: bool = True,
) -> None:
    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    data_path = stats_dir / f"temporal_correlations_by_pain{use_spearman_suffix}.npz"
    
    if not data_path.exists():
        logger.warning(f"Temporal correlation data not found: {data_path}")
        return
    
    logger.info("Plotting temporal correlation topomaps by pain condition...")
    
    global_fdr_map = _load_global_fdr_for_temporal_correlations(
        stats_dir, use_spearman, logger
    )
    use_global_fdr = global_fdr_map is not None
    if use_global_fdr:
        logger.info("Using global FDR correction for significance masking")
    else:
        logger.info("Using per-analysis FDR correction (global FDR not available)")
    
    data = np.load(data_path, allow_pickle=True)
    info = data.get("info", None)
    if info is None:
        logger.warning("Info not found in data file")
        return
    
    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()
    
    ch_names = data.get("ch_names", None)
    if ch_names is None:
        logger.warning("Channel names not found in data file")
        return
    
    if isinstance(ch_names, np.ndarray):
        ch_names = ch_names.tolist()
    
    if "pain" not in data or "non_pain" not in data:
        logger.warning("Pain/non-pain data not found in file")
        return
    
    info_ch_names = info["ch_names"]
    if len(ch_names) != len(info_ch_names) or set(ch_names) != set(info_ch_names):
        picks = mne.pick_channels(info_ch_names, include=ch_names, exclude=[])
        info = mne.pick_info(info, picks)
        logger.debug(f"Picked {len(picks)} channels from Info to match data channels")
    
    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    sig_text = get_sig_marker_text(config)
    
    result_pain = data["pain"].item()
    result_non = data["non_pain"].item()
    
    band_names = result_pain["band_names"]
    band_ranges = result_pain["band_ranges"]
    window_starts = result_pain["window_starts"]
    window_ends = result_pain["window_ends"]
    n_windows = len(window_starts)
    
    n_rows = 2
    
    all_corr_data = []
    for result in [result_pain, result_non]:
        for band_idx in range(len(band_names)):
            corr_data = result["correlations"][band_idx]
            all_corr_data.extend([c for c in corr_data.flatten() if np.isfinite(c)])
    
    vabs_corr = robust_sym_vlim(all_corr_data) if all_corr_data else 0.6
    
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    topomap_config = tfr_config.get("topomap", {})
    tfr_specific = topomap_config.get("tfr_specific", {})
    hspace = tfr_specific.get("hspace", 0.25)
    wspace = tfr_specific.get("wspace", 1.2)
    
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0] if plot_cfg else 10.0
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1] if plot_cfg else 10.0
    
    for band_idx, band_name in enumerate(band_names):
        fmin, fmax = band_ranges[band_idx]
        freq_label = f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)"
        
        fig, axes = plt.subplots(
            n_rows, n_windows, 
            figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows), 
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace}
        )
        
        row_labels = ["Non-Pain", "Pain"]
        results = [result_non, result_pain]
        
        for row_idx, (row_label, result) in enumerate(zip(row_labels, results)):
            correlations = result["correlations"][band_idx]
            p_values = result["p_values"][band_idx]
            p_corrected = result["p_corrected"][band_idx]
            
            condition_name = "non_pain" if row_idx == 0 else "pain"
            
            axes[row_idx, 0].set_ylabel(f"{row_label}\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
            
            for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
                if row_idx == 0:
                    time_label = f"{tmin_win:.2f}s"
                    axes[row_idx, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)
                
                corr_data = correlations[col, :]
                p_uncorr = p_values[col, :]
                p_fdr = p_corrected[col, :]
                
                sig_mask_uncorr = (p_uncorr < 0.05) & np.isfinite(p_uncorr)
                
                if use_global_fdr:
                    sig_mask_fdr = np.zeros(len(ch_names), dtype=bool)
                    n_matched = 0
                    for ch_idx, ch_name in enumerate(ch_names):
                        key = (condition_name, band_name, tmin_win, tmax_win, ch_name)
                        if key in global_fdr_map:
                            sig_mask_fdr[ch_idx] = global_fdr_map[key]
                            n_matched += 1
                    if n_matched == 0 and len(ch_names) > 0:
                        logger.debug(
                            f"No global FDR matches for condition={condition_name}, band={band_name}, "
                            f"window=[{tmin_win:.3f}, {tmax_win:.3f}]"
                        )
                else:
                    sig_mask_fdr = (p_fdr < 0.05) & np.isfinite(p_fdr)
                
                plot_topomap_on_ax(
                    axes[row_idx, col], corr_data, info,
                    vmin=-vabs_corr, vmax=+vabs_corr,
                    mask=sig_mask_fdr,
                    mask_params=dict(marker="o", markerfacecolor="green", markeredgecolor="green", markersize=4),
                    config=config
                )
                
                if sig_mask_uncorr.sum() > 0:
                    uncorr_chs = np.where(sig_mask_uncorr & ~sig_mask_fdr)[0]
                    if len(uncorr_chs) > 0:
                        try:
                            from mne.channels.layout import _find_topomap_coords
                            pos = _find_topomap_coords(info, picks=None)
                            axes[row_idx, col].plot(
                                pos[uncorr_chs, 0], pos[uncorr_chs, 1],
                                "o", markerfacecolor="white", markeredgecolor="black",
                                markersize=4, markeredgewidth=1, zorder=10
                            )
                        except Exception:
                            pass
                
                _add_correlation_roi_annotations(
                    axes[row_idx, col], corr_data, p_uncorr, p_fdr, info, config=config, fdr_alpha=0.05
                )
        
        create_difference_colorbar(
            fig, axes, vabs_corr, viz_params["topo_cmap"],
            label="Correlation coefficient"
        )
        
        window_label = f"{window_starts[0]:.1f}–{window_ends[-1]:.1f}s; {n_windows} windows"
        method_name = "Spearman" if use_spearman else "Pearson"
        fig.suptitle(
            f"Temporal correlation topomaps by pain condition ({band_name}, {window_label})\n{method_name} correlation, vlim ±{vabs_corr:.2f}{sig_text}\n",
            fontsize=font_sizes["suptitle"], y=0.995
        )
        
        topomap_dir = plots_dir / "topomaps"
        ensure_dir(topomap_dir)
        filename = f"sub-{subject}_temporal_correlations_by_pain_{band_name}.png"
        save_fig(fig, topomap_dir / filename, formats=plot_cfg.formats if plot_cfg else ["png", "svg"], dpi=plot_cfg.dpi if plot_cfg else None, bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight", pad_inches=plot_cfg.pad_inches if plot_cfg else None, footer=_get_behavior_footer(config))
        plt.close(fig)
    
    logger.info(f"Created temporal correlation topomaps by pain for {len(band_names)} bands")
    
    logger.info("Creating combined figure with all bands...")
    n_bands = len(band_names)
    mid_window_idx = n_windows // 2
    
    fig, axes = plt.subplots(
        n_rows, n_bands, 
        figsize=(fig_size_per_col_large * n_bands, fig_size_per_row_large * n_rows), 
        squeeze=False,
        gridspec_kw={"hspace": hspace, "wspace": wspace}
    )
    
    row_labels = ["Non-Pain", "Pain"]
    results = [result_non, result_pain]
    
    for row_idx, (row_label, result) in enumerate(zip(row_labels, results)):
        for band_idx, band_name in enumerate(band_names):
            fmin, fmax = band_ranges[band_idx]
            freq_label = f"{band_name}\n({fmin:.0f}-{fmax:.0f}Hz)"
            
            correlations = result["correlations"][band_idx]
            p_values = result["p_values"][band_idx]
            p_corrected = result["p_corrected"][band_idx]
            
            if row_idx == 0:
                axes[row_idx, band_idx].set_title(freq_label, fontsize=font_sizes["title"], pad=12, y=1.07)
            
            if band_idx == 0:
                axes[row_idx, band_idx].set_ylabel(row_label, fontsize=font_sizes["ylabel"], labelpad=10)
            
            corr_data = correlations[mid_window_idx, :]
            p_uncorr = p_values[mid_window_idx, :]
            p_fdr = p_corrected[mid_window_idx, :]
            
            sig_mask_uncorr = (p_uncorr < 0.05) & np.isfinite(p_uncorr)
            sig_mask_fdr = (p_fdr < 0.05) & np.isfinite(p_fdr)
            
            plot_topomap_on_ax(
                axes[row_idx, band_idx], corr_data, info,
                vmin=-vabs_corr, vmax=+vabs_corr,
                mask=sig_mask_fdr,
                mask_params=dict(marker="o", markerfacecolor="green", markeredgecolor="green", markersize=4),
                config=config
            )
            
            if sig_mask_uncorr.sum() > 0:
                uncorr_chs = np.where(sig_mask_uncorr & ~sig_mask_fdr)[0]
                if len(uncorr_chs) > 0:
                    try:
                        from mne.channels.layout import _find_topomap_coords
                        pos = _find_topomap_coords(info, picks=None)
                        axes[row_idx, band_idx].plot(
                            pos[uncorr_chs, 0], pos[uncorr_chs, 1],
                            "o", markerfacecolor="white", markeredgecolor="black",
                            markersize=4, markeredgewidth=1, zorder=10
                        )
                    except Exception:
                        pass
            
            _add_correlation_roi_annotations(
                axes[row_idx, band_idx], corr_data, p_uncorr, p_fdr, info, config=config, fdr_alpha=0.05
            )
    
    create_difference_colorbar(
        fig, axes, vabs_corr, viz_params["topo_cmap"],
        label="Correlation coefficient"
    )
    
    tmin_win, tmax_win = window_starts[mid_window_idx], window_ends[mid_window_idx]
    method_name = "Spearman" if use_spearman else "Pearson"
    fig.suptitle(
        f"Temporal correlation topomaps by pain condition - All bands ({tmin_win:.1f}-{tmax_win:.1f}s)\n{method_name} correlation, vlim ±{vabs_corr:.2f}{sig_text}\n",
        fontsize=font_sizes["suptitle"], y=0.995
    )
    
    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    filename = f"sub-{subject}_temporal_correlations_by_pain_allbands.png"
    save_fig(fig, topomap_dir / filename, formats=plot_cfg.formats if plot_cfg else ["png", "svg"], dpi=plot_cfg.dpi if plot_cfg else None, bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight", pad_inches=plot_cfg.pad_inches if plot_cfg else None, footer=_get_behavior_footer(config))
    plt.close(fig)
    
    logger.info("Created combined temporal correlation topomap with all bands")
    
    logger.info("Creating combined figure with all bands averaged...")
    mid_window_idx = n_windows // 2
    
    fig, axes = plt.subplots(
        n_rows, 1, 
        figsize=(fig_size_per_col_large, fig_size_per_row_large * n_rows), 
        squeeze=False,
        gridspec_kw={"hspace": hspace, "wspace": wspace}
    )
    
    row_labels = ["Non-Pain", "Pain"]
    results = [result_non, result_pain]
    
    for row_idx, (row_label, result) in enumerate(zip(row_labels, results)):
        all_band_correlations = []
        all_band_p_uncorr = []
        all_band_p_fdr = []
        
        for band_idx in range(len(band_names)):
            correlations = result["correlations"][band_idx]
            p_values = result["p_values"][band_idx]
            p_corrected = result["p_corrected"][band_idx]
            
            all_band_correlations.append(correlations[mid_window_idx, :])
            all_band_p_uncorr.append(p_values[mid_window_idx, :])
            all_band_p_fdr.append(p_corrected[mid_window_idx, :])
        
        combined_corr = np.nanmean(all_band_correlations, axis=0)
        combined_p_uncorr = np.nanmin(all_band_p_uncorr, axis=0)
        combined_p_fdr = np.nanmin(all_band_p_fdr, axis=0)
        
        sig_mask_fdr = (combined_p_fdr < 0.05) & np.isfinite(combined_p_fdr)
        
        axes[row_idx, 0].set_ylabel(row_label, fontsize=font_sizes["ylabel"], labelpad=10)
        
        tmin_win, tmax_win = window_starts[mid_window_idx], window_ends[mid_window_idx]
        time_label = f"{tmin_win:.1f}-{tmax_win:.1f}s"
        if row_idx == 0:
            axes[row_idx, 0].set_title(f"All bands combined\n{time_label}", fontsize=font_sizes["title"], pad=12, y=1.07)
        
        plot_topomap_on_ax(
            axes[row_idx, 0], combined_corr, info,
            vmin=-vabs_corr, vmax=+vabs_corr,
            mask=sig_mask_fdr,
            mask_params=dict(marker="o", markerfacecolor="green", markeredgecolor="green", markersize=4),
            config=config
        )
        
        _add_correlation_roi_annotations(
            axes[row_idx, 0], combined_corr, combined_p_uncorr, combined_p_fdr, info, config=config, fdr_alpha=0.05
        )
    
    create_difference_colorbar(
        fig, axes, vabs_corr, viz_params["topo_cmap"],
        label="Correlation coefficient"
    )
    
    method_name = "Spearman" if use_spearman else "Pearson"
    fig.suptitle(
        f"Temporal correlation topomaps by pain condition - All bands combined ({time_label})\n{method_name} correlation (averaged across bands), vlim ±{vabs_corr:.2f}{sig_text}\n",
        fontsize=font_sizes["suptitle"], y=0.995
    )
    
    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    filename = f"sub-{subject}_temporal_correlations_by_pain_allbands_combined.png"
    save_fig(fig, topomap_dir / filename, formats=plot_cfg.formats if plot_cfg else ["png", "svg"], dpi=plot_cfg.dpi if plot_cfg else None, bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight", pad_inches=plot_cfg.pad_inches if plot_cfg else None, footer=_get_behavior_footer(config))
    plt.close(fig)
    
    logger.info("Created combined temporal correlation topomap with all bands averaged")


###################################################################
# Additional Visualization Functions
###################################################################


def plot_pain_nonpain_clusters(
    subject: str,
    stats_dir: Path,
    plots_dir: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    alpha: Optional[float] = None,
) -> None:
    """
    Visualize pain vs. non-pain cluster permutation results as band-wise ribbons and channel-time masks.
    """
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    ensure_dir(plots_dir)
    # Get alpha value safely
    if alpha is not None:
        alpha_used = float(alpha)
    else:
        try:
            alpha_used = float(config.get("behavior_analysis.statistics.fdr_alpha", config.get("statistics.sig_alpha", 0.05)))
        except (ValueError, TypeError):
            alpha_used = 0.05
    
    if not np.isfinite(alpha_used) or alpha_used <= 0:
        alpha_used = 0.05

    files = sorted(stats_dir.glob("pain_nonpain_time_clusters_*.tsv"))
    if not files:
        _log_if_present(logger, "warning", f"No pain/non-pain cluster stats found in {stats_dir}")
        return

    frames: List[pd.DataFrame] = []
    for fpath in files:
        df = read_tsv(fpath)
        if df is None or df.empty:
            continue
        band = df["band"].iloc[0] if "band" in df.columns else fpath.stem.replace("pain_nonpain_time_clusters_", "")
        df["band"] = df.get("band", band)
        df["p_value"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")
        if "fdr_reject_global" in df.columns:
            df["significant"] = df["fdr_reject_global"].fillna(False)
        else:
            df["significant"] = df["p_value"] < alpha_used
        df["t_start"] = pd.to_numeric(df.get("t_start", np.nan), errors="coerce")
        df["t_end"] = pd.to_numeric(df.get("t_end", np.nan), errors="coerce")
        df["cluster_index"] = pd.to_numeric(df.get("cluster_index", np.nan), errors="coerce")
        frames.append(df)

    if not frames:
        _log_if_present(logger, "warning", "Pain/non-pain cluster TSVs were found but empty.")
        return

    all_clusters = pd.concat(frames, ignore_index=True)
    sig_clusters = all_clusters[all_clusters["significant"]]
    if sig_clusters.empty:
        _log_if_present(logger, "warning", "No significant pain vs. non-pain clusters to plot.")
        return

    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    sig_out = topomap_dir / "pain_nonpain_clusters_significant.tsv"
    sig_clusters.to_csv(sig_out, sep="\t", index=False)
    _log_if_present(logger, "info", f"Saved significant cluster table to {sig_out}")

    band_order = get_frequency_band_names(config)
    if not band_order:
        band_order = sorted(all_clusters["band"].unique())
    band_order = [b for b in band_order if not all_clusters[all_clusters["band"] == b].empty]
    if not band_order:
        _log_if_present(logger, "warning", "No clusters available after filtering bands.")
        return

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))
    for idx, band in enumerate(band_order):
        band_df = all_clusters[all_clusters["band"] == band]
        if band_df.empty:
            continue
        t_start_min = band_df["t_start"].dropna().min() if not band_df["t_start"].dropna().empty else 0
        t_end_max = band_df["t_end"].dropna().max() if not band_df["t_end"].dropna().empty else 0
        if np.isfinite(t_start_min) and np.isfinite(t_end_max):
            ax.hlines(idx, t_start_min, t_end_max, colors=plot_cfg.get_color("light_gray"), linestyles="--", linewidth=0.5, alpha=0.5)
        
        for _, row in band_df.iterrows():
            t_start_val = row.get("t_start")
            t_end_val = row.get("t_end")
            
            # Check for None/NaN values
            if pd.isna(t_start_val) or pd.isna(t_end_val):
                continue
            
            try:
                t_start = float(t_start_val)
                t_end = float(t_end_val)
            except (ValueError, TypeError):
                continue
            
            if not np.isfinite(t_start) or not np.isfinite(t_end):
                continue
            
            width = t_end - t_start
            if width <= 0:
                continue
            
            color = plot_cfg.get_color("significant") if row.get("significant", False) else plot_cfg.get_color("nonsignificant")
            ax.broken_barh([(t_start, width)], (idx - 0.35, 0.7), facecolors=color, edgecolors="none", alpha=0.8)

    ax.set_yticks(range(len(band_order)))
    ax.set_yticklabels(band_order)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band")
    ax.set_title(f"Pain vs. non-pain cluster summary (sub-{subject})")
    legend_patches = [
        mpatches.Patch(color=plot_cfg.get_color("significant"), label="Significant"),
        mpatches.Patch(color=plot_cfg.get_color("nonsignificant"), label="Non-significant"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=False)
    plt.tight_layout()
    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    save_fig(fig, topomap_dir / f"sub-{subject}_pain_nonpain_cluster_ribbons", formats=plot_cfg.formats, dpi=plot_cfg.dpi, bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)

    for band in band_order:
        band_df = sig_clusters[sig_clusters["band"] == band]
        if band_df.empty:
            continue
        
        # Get t_min and t_max safely
        t_start_vals = band_df["t_start"].dropna()
        t_end_vals = band_df["t_end"].dropna()
        
        if t_start_vals.empty or t_end_vals.empty:
            continue
        
        try:
            t_min = float(t_start_vals.min())
            t_max = float(t_end_vals.max())
        except (ValueError, TypeError):
            continue
        
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue
        time_grid = np.linspace(t_min, t_max, 200)
        channels: List[str] = []
        for ch_list in band_df.get("channels", []):
            if pd.isna(ch_list):
                continue
            channels.extend([ch.strip() for ch in str(ch_list).split(",") if ch.strip()])
        channels = sorted(set(channels))
        if not channels:
            continue
        mask = np.zeros((len(channels), len(time_grid) - 1), dtype=int)
        for _, row in band_df.iterrows():
            ch_list = row.get("channels", "")
            if pd.isna(ch_list):
                continue
            row_channels = [ch.strip() for ch in str(ch_list).split(",") if ch.strip()]
            
            t_start_val = row.get("t_start")
            t_end_val = row.get("t_end")
            
            # Check for None/NaN values
            if pd.isna(t_start_val) or pd.isna(t_end_val):
                continue
            
            try:
                t_start = float(t_start_val)
                t_end = float(t_end_val)
            except (ValueError, TypeError):
                continue
            
            if not np.isfinite(t_start) or not np.isfinite(t_end):
                continue
            overlaps = (time_grid[:-1] < t_end) & (time_grid[1:] > t_start)
            if not np.any(overlaps):
                continue
            for ch in row_channels:
                try:
                    ch_idx = channels.index(ch)
                except ValueError:
                    continue
                mask[ch_idx, overlaps] = 1

        fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("standard", plot_type="behavioral"))
        mesh = ax.pcolormesh(time_grid, np.arange(len(channels) + 1), mask, cmap="Reds", shading="auto", vmin=0, vmax=1)
        ax.set_title(f"Pain vs. non-pain clusters (band: {band}, sub-{subject})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        ax.set_yticks(np.arange(len(channels)) + 0.5)
        ax.set_yticklabels(channels)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Significant cluster coverage")
        plt.tight_layout()
        topomap_dir = plots_dir / "topomaps"
        ensure_dir(topomap_dir)
        save_fig(fig, topomap_dir / f"sub-{subject}_pain_nonpain_cluster_mask_{band}", formats=plot_cfg.formats, dpi=plot_cfg.dpi, bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
        plt.close(fig)


def plot_regressor_distributions(
    regressor_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    max_regressors: int = 12,
) -> None:
    """
    QC scatter plots of fMRI regressors vs. pain ratings.
    """
    if regressor_df is None or regressor_df.empty:
        _log_if_present(logger, "warning", "Regressor dataframe is empty; skipping QC plot.")
        return

    rating_col = "rating"
    if rating_col not in regressor_df.columns:
        _log_if_present(logger, "warning", "Rating column missing in regressor dataframe; skipping QC plot.")
        return

    reg_cols = [c for c in regressor_df.columns if c not in ("onset", "duration", rating_col)]
    if not reg_cols:
        _log_if_present(logger, "warning", "No regressors available for QC plot.")
        return

    reg_cols = reg_cols[:max_regressors]
    n_reg = len(reg_cols)
    n_cols = min(3, n_reg)
    n_rows = int(np.ceil(n_reg / n_cols))

    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), squeeze=False)

    ratings = pd.to_numeric(regressor_df[rating_col], errors="coerce")

    for idx, col in enumerate(reg_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        reg_vals = pd.to_numeric(regressor_df[col], errors="coerce")
        ax.scatter(ratings, reg_vals, alpha=0.65, s=22, color="teal", edgecolor="none")
        ax.set_xlabel("VAS rating")
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)
        try:
            rho, pval = stats.spearmanr(ratings, reg_vals, nan_policy="omit")
            if np.isfinite(rho) and np.isfinite(pval):
                ax.set_title(f"{col} (rho={rho:.2f}, p={pval:.3f})", fontsize=10)
        except Exception as exc:
            _log_if_present(logger, "warning", f"Failed correlation for {col}: {exc}")

    total_axes = n_rows * n_cols
    for j in range(n_reg, total_axes):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"fMRI regressor QC (sub-{subject})", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = save_dir / f"sub-{subject}_fmri_regressor_qc"
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    _log_if_present(logger, "info", f"Saved fMRI regressor QC plot to {output_path}")


def plot_pac_behavior_correlations(
    subject: str,
    stats_dir: Path,
    plots_dir: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Plot PAC-behavior correlations as comodulograms (phase x amplitude) per ROI.
    """
    logger = logger or _get_default_logger()
    stats_path = stats_dir / "corr_stats_pac_vs_rating.tsv"
    if not stats_path.exists():
        _log_if_present(logger, "warning", f"PAC stats not found: {stats_path}")
        return

    try:
        df = read_tsv(stats_path)
    except Exception as exc:
        _log_if_present(logger, "warning", f"Failed to read PAC stats: {exc}")
        return
    if df is None or df.empty:
        _log_if_present(logger, "warning", "PAC stats empty; skipping PAC plots.")
        return

    alpha = float(config.get("statistics.sig_alpha", 0.05))
    roi_list = sorted(df["roi"].dropna().unique().tolist())
    plot_cfg = get_plot_config(config)
    ensure_dir(plots_dir)

    for roi in roi_list:
        sub_df = df[df["roi"] == roi].copy()
        if sub_df.empty:
            continue

        phase_vals = pd.to_numeric(sub_df["phase_freq"], errors="coerce")
        amp_vals = pd.to_numeric(sub_df["amp_freq"], errors="coerce")
        r_vals = pd.to_numeric(sub_df["r"], errors="coerce")
        p_vals = pd.to_numeric(
            sub_df.get("p_used_for_global_fdr", sub_df.get("p", np.nan)),
            errors="coerce",
        )

        signif = None
        if "fdr_reject_global" in sub_df.columns:
            signif = sub_df["fdr_reject_global"].fillna(False).to_numpy(dtype=bool)
        elif "fdr_reject" in sub_df.columns:
            signif = sub_df["fdr_reject"].fillna(False).to_numpy(dtype=bool)
        else:
            signif = p_vals < alpha

        sub_df["phase_freq"] = phase_vals
        sub_df["amp_freq"] = amp_vals
        sub_df["r"] = r_vals
        sub_df["p_plot"] = p_vals
        sub_df["signif"] = signif

        pivot_r = sub_df.pivot_table(
            index="amp_freq",
            columns="phase_freq",
            values="r",
            aggfunc="mean",
        )
        if pivot_r.empty:
            continue

        vmax = compute_correlation_vmax(pivot_r.values)
        fig, ax = plt.subplots(
            figsize=plot_cfg.get_figure_size("standard", plot_type="behavioral")
        )
        sns.heatmap(
            pivot_r.sort_index(ascending=True),
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": "r"},
            linewidths=0.1,
            linecolor="white",
        )

        sig_rows = []
        for _, row in sub_df.iterrows():
            if not row.get("signif", False):
                continue
            sig_rows.append((row["amp_freq"], row["phase_freq"]))
        if sig_rows:
            amps, phases = zip(*sig_rows)
            ax.scatter(
                phases,
                amps,
                s=22,
                color=plot_cfg.get_color("significant", plot_type="behavioral"),
                marker="o",
                edgecolor="white",
                linewidth=0.3,
            )

        ax.set_xlabel("Phase frequency (Hz)")
        ax.set_ylabel("Amplitude frequency (Hz)")
        ax.set_title(f"PAC vs rating (ROI: {roi}, sub-{subject})")
        plt.tight_layout()
        pac_dir = plots_dir / "topomaps"
        ensure_dir(pac_dir)
        out_path = pac_dir / f"sub-{subject}_pac_comod_{roi}.png"
        save_fig(
            fig,
            out_path,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        plt.close(fig)
        _log_if_present(logger, "info", f"Saved PAC comodulogram for ROI {roi} to {out_path}")


def plot_itpc_rating_scatter_grid(
    subject: str,
    pow_df: pd.DataFrame,
    ratings: pd.Series,
    stats_dir: Path,
    plots_dir: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    top_n: int = 9,
) -> None:
    """
    Plot ITPC vs rating scatter for top significant bins (band x channel x time_bin).
    """
    logger = logger or _get_default_logger()
    stats_path = stats_dir / "corr_stats_itpc_vs_rating.tsv"
    if not stats_path.exists():
        _log_if_present(logger, "warning", f"ITPC stats not found: {stats_path}")
        return

    try:
        stats_df = read_tsv(stats_path)
    except Exception as exc:
        _log_if_present(logger, "warning", f"Failed to read ITPC stats: {exc}")
        return
    if stats_df is None or stats_df.empty:
        _log_if_present(logger, "warning", "ITPC stats empty; skipping ITPC scatter plots.")
        return

    alpha = float(config.get("statistics.sig_alpha", 0.05))
    stats_df = stats_df.copy()
    stats_df["abs_r"] = stats_df["r"].abs()

    if "fdr_reject_global" in stats_df.columns:
        stats_df = stats_df[stats_df["fdr_reject_global"].fillna(False)]
    elif "fdr_reject" in stats_df.columns:
        stats_df = stats_df[stats_df["fdr_reject"].fillna(False)]
    else:
        stats_df = stats_df[pd.to_numeric(stats_df.get("p", np.nan), errors="coerce") < alpha]

    if stats_df.empty:
        _log_if_present(logger, "warning", "No significant ITPC correlations to plot.")
        return

    stats_df = stats_df.sort_values("abs_r", ascending=False).head(top_n)
    plot_cfg = get_plot_config(config)
    ensure_dir(plots_dir)

    n = len(stats_df)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for idx, (_, row) in enumerate(stats_df.iterrows()):
        ax = axes[idx // n_cols][idx % n_cols]
        band = str(row["band"])
        channel = str(row["channel"])
        time_bin = str(row["time_bin"])
        col_name = f"itpc_{band}_{channel}_{time_bin}"
        if col_name not in pow_df.columns:
            _log_if_present(logger, "warning", f"Missing ITPC column {col_name}; skipping.")
            ax.axis("off")
            continue
        x_vals = pd.to_numeric(pow_df[col_name], errors="coerce")
        y_vals = pd.to_numeric(ratings, errors="coerce")
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask.sum() < 3:
            ax.axis("off")
            continue

        ax.scatter(x_vals[mask], y_vals[mask], s=14, alpha=0.7, color=plot_cfg.get_color("blue"))
        ax.set_xlabel(f"{col_name}")
        ax.set_ylabel("Rating")
        ax.grid(alpha=0.2)
        try:
            rho, pval = stats.spearmanr(x_vals[mask], y_vals[mask], nan_policy="omit")
            ax.set_title(f"{channel} ({band}, {time_bin})\nr={rho:.2f}, p={pval:.3f}")
        except Exception:
            ax.set_title(f"{channel} ({band}, {time_bin})")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"ITPC vs rating (sub-{subject})", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    scatter_dir = plots_dir / "scatter"
    ensure_dir(scatter_dir)
    out_path = scatter_dir / f"sub-{subject}_itpc_rating_scatter.png"
    save_fig(
        fig,
        out_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    _log_if_present(logger, "info", f"Saved ITPC scatter grid to {out_path}")


###################################################################
# Significant Correlations Topomap
###################################################################


def _get_behavioral_config(plot_cfg):
    return plot_cfg.plot_type_configs.get("behavioral", {})


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
    
    topomap_config = behavioral_config.get("topomap", {})
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    topomap_config_tfr = tfr_config.get("topomap", {})
    tfr_specific = topomap_config_tfr.get("tfr_specific", {})
    wspace = tfr_specific.get("wspace", 1.2)
    
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0] if plot_cfg else 10.0
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1] if plot_cfg else 10.0
    
    n_bands = len(bands_with_data)
    fig, axes = plt.subplots(
        1, n_bands, 
        figsize=(fig_size_per_col_large * n_bands, fig_size_per_row_large), 
        squeeze=False,
        gridspec_kw={"wspace": wspace}
    )
    axes = axes[0]
    
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
    
    tight_layout_rect = topomap_config.get("tight_layout_rect", [0, 0.15, 1, 1])
    topomap_dir = save_dir / "topomaps"
    ensure_dir(topomap_dir)
    save_fig(
        fig,
        topomap_dir / f'sub-{subject}_significant_correlations_topomap',
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        tight_layout_rect=tight_layout_rect
    )
    plt.close(fig)
    
    logger.info(f"Created topomaps for {len(bands_with_data)} frequency bands: {[bd['band'] for bd in bands_with_data]}")


###################################################################
# Behavior-Modulated Connectivity
###################################################################


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
    
    base_seed = config.get("project.random_state", 42)
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
    save_fig(
        fig,
        save_dir / f'sub-{subject}_connectivity_network_{measure}_{band}',
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
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
    
    bands = get_frequency_band_names(config)
    
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


###################################################################
# Time-Frequency Correlation Heatmap
###################################################################


def _load_tf_correlation_data(data_path: Path, config) -> Dict[str, Any]:
    with np.load(data_path, allow_pickle=True) as data:
        correlations = data["correlations"]
        expected_n_time_bins = correlations.shape[1] if len(correlations.shape) == 2 else correlations.shape[0]
        
        time_bin_edges = data.get("time_bin_edges")
        if "time_bin_centers" in data:
            time_bin_centers = data["time_bin_centers"]
            if len(time_bin_centers) != expected_n_time_bins:
                if time_bin_edges is not None and len(time_bin_edges) > 1:
                    time_bin_centers = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2.0
                else:
                    time_bin_centers = time_bin_centers[:expected_n_time_bins]
        elif "times" in data:
            times = data["times"]
            if len(times) == expected_n_time_bins:
                time_bin_centers = times
            elif time_bin_edges is not None and len(time_bin_edges) > 1:
                time_bin_centers = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2.0
            else:
                time_bin_centers = times[:expected_n_time_bins]
        elif time_bin_edges is not None and len(time_bin_edges) > 1:
            time_bin_centers = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2.0
        else:
            raise ValueError(f"Cannot determine time_bin_centers from {data_path}")
        
        if len(time_bin_centers) != expected_n_time_bins:
            raise ValueError(
                f"Time bin centers length ({len(time_bin_centers)}) does not match "
                f"correlation matrix time dimension ({expected_n_time_bins})"
            )
        
        method = data.get("method")
        if method is None:
            method = "spearman" if data.get("use_spearman", True) else "pearson"
        
        covariates_raw = data.get("covariates_used", np.array([], dtype=str))
        covariates_used = list(covariates_raw) if covariates_raw.size > 0 else []
        
        return {
            "correlations": correlations,
            "p_values": data["p_values"],
            "p_corrected": data.get("p_corrected", np.full_like(correlations, np.nan)),
            "significant_mask": data.get("significant_mask", np.zeros_like(correlations, dtype=bool)),
            "cluster_labels": data.get("cluster_labels"),
            "cluster_pvals": data.get("cluster_pvals"),
            "cluster_sig_mask": data.get("cluster_sig_mask"),
            "freqs": data["freqs"],
            "time_bin_centers": time_bin_centers,
            "time_bin_edges": time_bin_edges,
            "n_valid": data.get("n_valid", np.zeros_like(correlations, dtype=int)),
            "freq_range": tuple(data.get("freq_range", (float(data["freqs"][0]), float(data["freqs"][-1])))),
            "baseline_applied": bool(data.get("baseline_applied", False)),
            "baseline_window": data.get("baseline_window", np.array([])),
            "time_resolution": float(data.get("time_resolution", 0.1)),
            "alpha": float(data.get("alpha", config.get("behavior_analysis.statistics.fdr_alpha", 0.05))),
            "cluster_alpha": float(data.get("cluster_alpha", data.get("alpha", config.get("behavior_analysis.statistics.fdr_alpha", 0.05)))),
            "cluster_n_perm": int(data.get("cluster_n_perm", 0)),
            "method": str(method),
            "covariates_used": covariates_used,
            "n_trials": int(data.get("n_trials", 0)),
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
    plots_dir: Optional[Path] = None,
) -> None:
    config = load_settings()
    if task is None:
        task = config.get("project.task", "thermalactive")

    log_name = config.get("logging.log_file_name", "behavior_analysis.log")
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
    if plots_dir is None:
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
    
    footer_text = _get_behavior_footer(config)
    n_trials = tf_data.get("n_trials", 0)
    covariates_used = tf_data.get("covariates_used", [])
    if n_trials > 0:
        footer_text = f"{footer_text} | n={n_trials} trials"
    if covariates_used:
        covar_str = ", ".join(covariates_used)
        footer_text = f"{footer_text} | Partial corr. controlling: {covar_str}"
    
    save_fig(
        fig,
        plots_dir / fig_name,
        formats=plot_cfg.formats if plot_cfg else ["png", "svg"],
        dpi=plot_cfg.dpi if plot_cfg else None,
        bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight",
        pad_inches=plot_cfg.pad_inches if plot_cfg else None,
        footer=footer_text,
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
