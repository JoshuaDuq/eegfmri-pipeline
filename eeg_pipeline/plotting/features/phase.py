"""
Phase analysis visualization plotting functions.

Functions for creating phase-related plots including ITPC (Inter-Trial Phase Coherence)
heatmaps, topomaps, behavior scatter plots, and PAC (Phase-Amplitude Coupling)
comodulograms and time ribbons.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import save_fig, log_if_present
from ..config import get_plot_config
from ...utils.analysis.stats import fdr_bh
from eeg_pipeline.utils.formatting import sanitize_label

# Constants
ITPC_MIN = 0.0
ITPC_MAX = 1.0
PAC_DEFAULT_MIN = 0.0
PAC_DEFAULT_MAX = 0.5
PERCENTILE_LOW = 5
PERCENTILE_HIGH = 95
MIN_COLOR_RANGE = 0.01
COLOR_RANGE_ADJUSTMENT = 0.005
SCATTER_JITTER_RANGE = 0.08
BOXPLOT_WIDTH = 0.4
Y_RANGE_PADDING_LOW = 0.1
Y_RANGE_PADDING_HIGH = 0.3
STATS_ANNOTATION_OFFSET = 0.05
DEFAULT_ALPHA_SIG = 0.05
ROI_NAME_MAX_LENGTH = 20


def _validate_itpc_data(itpc_map: Optional[np.ndarray], freqs: Optional[np.ndarray], 
                        times: Optional[np.ndarray], logger: Optional[logging.Logger]) -> bool:
    """Validate ITPC data inputs.
    
    Args:
        itpc_map: ITPC map array
        freqs: Frequency array
        times: Time array
        logger: Logger instance
        
    Returns:
        True if data is valid, False otherwise
    """
    if itpc_map is None or freqs is None or times is None:
        log_if_present(logger, "warning", "ITPC map missing; skipping heatmap")
        return False
    
    if itpc_map.ndim != 3:
        log_if_present(logger, "warning", 
                      f"Unexpected ITPC map shape {itpc_map.shape}; skipping heatmap")
        return False
    
    return True


def _compute_color_limits(values: np.ndarray, default_min: float = 0.0, 
                         default_max: float = 1.0) -> Tuple[float, float]:
    """Compute color limits from data values using percentiles.
    
    Args:
        values: Array of values
        default_min: Default minimum if computation fails
        default_max: Default maximum if computation fails
        
    Returns:
        Tuple of (vmin, vmax)
    """
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return default_min, default_max
    
    vmin = np.percentile(finite_values, PERCENTILE_LOW)
    vmax = np.percentile(finite_values, PERCENTILE_HIGH)
    
    if np.isclose(vmin, vmax):
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
    
    if vmax - vmin < MIN_COLOR_RANGE:
        center = (vmin + vmax) / 2
        vmin = center - COLOR_RANGE_ADJUSTMENT
        vmax = center + COLOR_RANGE_ADJUSTMENT
    
    return vmin, vmax


def _convert_itpc_wide_to_long(itpc_df: pd.DataFrame, 
                                logger: Optional[logging.Logger]) -> Optional[pd.DataFrame]:
    """Convert ITPC DataFrame from wide to long format.
    
    Args:
        itpc_df: DataFrame in wide format
        logger: Logger instance
        
    Returns:
        DataFrame in long format or None if conversion fails
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    
    rows = []
    for col in itpc_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "itpc":
            continue
        if parsed.get("scope") != "ch":
            continue
        if parsed.get("stat") not in ("val", "mean", "avg", "value"):
            continue
        
        band = parsed.get("band")
        segment = parsed.get("segment")
        channel = parsed.get("identifier")
        if not band or not segment or not channel:
            continue
        
        itpc_val = pd.to_numeric(itpc_df[col], errors="coerce").mean()
        rows.append({
            "channel": str(channel),
            "band": str(band),
            "time_bin": str(segment),
            "itpc": itpc_val,
        })
    
    if not rows:
        log_if_present(logger, "warning", 
                      "Could not parse ITPC columns; skipping topomaps")
        return None
    
    return pd.DataFrame(rows)


def _prepare_topomap_data(df_long: pd.DataFrame, info: mne.Info) -> Dict[Tuple[str, str], Tuple[List[float], mne.Info]]:
    """Prepare topomap data from long-format DataFrame.
    
    Args:
        df_long: DataFrame in long format
        info: MNE Info object
        
    Returns:
        Dictionary mapping (band, time_bin) to (values_ordered, info_subset)
    """
    topomap_data = {}
    
    for (band, time_bin), df_sub in df_long.groupby(["band", "time_bin"]):
        ch_names = df_sub["channel"].astype(str).tolist()
        values = df_sub["itpc"].to_numpy(dtype=float)
        
        picks = mne.pick_channels(info.ch_names, include=ch_names, ordered=True)
        if len(picks) == 0:
            continue
        
        info_subset = mne.pick_info(info, picks)
        values_ordered = []
        for ch in info_subset.ch_names:
            try:
                idx = ch_names.index(ch)
                values_ordered.append(values[idx])
            except ValueError:
                values_ordered.append(np.nan)
        
        if np.any(np.isfinite(values_ordered)):
            topomap_data[(band, time_bin)] = (values_ordered, info_subset)
    
    return topomap_data


def plot_itpc_heatmap(
    itpc_map: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot ITPC heatmap averaged across channels.
    
    Args:
        itpc_map: ITPC map array (n_channels, n_freqs, n_times)
        freqs: Frequency array
        times: Time array
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if not _validate_itpc_data(itpc_map, freqs, times, logger):
        return

    plot_cfg = get_plot_config(config)
    itpc_avg = np.nanmean(itpc_map, axis=0)

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("standard", plot_type="tfr"))
    im = ax.imshow(
        itpc_avg,
        origin="lower",
        aspect="auto",
        extent=[float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])],
        cmap="magma",
        vmin=ITPC_MIN,
        vmax=ITPC_MAX,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"ITPC (channel-mean, sub-{subject})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ITPC")

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_itpc_heatmap"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved ITPC heatmap")


def _ensure_long_format(itpc_df: pd.DataFrame, 
                        logger: Optional[logging.Logger]) -> Optional[pd.DataFrame]:
    """Ensure DataFrame is in long format, converting if necessary.
    
    Args:
        itpc_df: DataFrame in either long or wide format
        logger: Logger instance
        
    Returns:
        DataFrame in long format or None if conversion fails
    """
    required_cols = {"channel", "band", "time_bin", "itpc"}
    if required_cols.issubset(set(itpc_df.columns)):
        return itpc_df
    
    return _convert_itpc_wide_to_long(itpc_df, logger)


def _plot_topomap_grid(axes: np.ndarray, bands: List[str], time_bins: List[str],
                       topomap_data: Dict[Tuple[str, str], Tuple[List[float], mne.Info]],
                       shared_colorbar: bool, all_values: List[float],
                       plot_cfg: Any) -> Tuple[Optional[Any], float, float]:
    """Plot topomaps in a grid layout.
    
    Args:
        axes: 2D array of axes
        bands: List of band names
        time_bins: List of time bin names
        topomap_data: Dictionary mapping (band, time_bin) to (values, info)
        shared_colorbar: Whether to use shared colorbar
        all_values: All values for shared colorbar computation
        plot_cfg: Plot configuration
        
    Returns:
        Tuple of (image object for colorbar or None, vmin, vmax)
    """
    if all_values and shared_colorbar:
        vmin, vmax = _compute_color_limits(np.array(all_values), ITPC_MIN, ITPC_MAX)
    else:
        vmin, vmax = ITPC_MIN, ITPC_MAX
    
    im = None
    for row_idx, band in enumerate(bands):
        for col_idx, time_bin in enumerate(time_bins):
            ax = axes[row_idx, col_idx]
            
            if (band, time_bin) in topomap_data:
                values_ordered, info_subset = topomap_data[(band, time_bin)]
                
                if shared_colorbar:
                    vmin_use, vmax_use = vmin, vmax
                else:
                    finite_values = np.array([v for v in values_ordered if np.isfinite(v)])
                    if len(finite_values) > 0:
                        vmin_use, vmax_use = _compute_color_limits(finite_values, ITPC_MIN, ITPC_MAX)
                    else:
                        vmin_use, vmax_use = ITPC_MIN, ITPC_MAX

                im, _ = mne.viz.plot_topomap(
                    values_ordered,
                    info_subset,
                    axes=ax,
                    show=False,
                    cmap="viridis",
                    vlim=(vmin_use, vmax_use),
                    contours=6,
                )
                ax.set_title(f"{band} | {time_bin}")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes)
    
    return im, vmin, vmax


def plot_itpc_topomaps(
    itpc_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot ITPC topomaps in a grid organized by band and time bin.
    
    Args:
        itpc_df: DataFrame with either:
            - Long format columns: channel, band, time_bin, itpc
            - Wide format columns: itpc_{segment}_{band}_ch_{channel}_val
        info: MNE Info object
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if itpc_df is None or len(itpc_df) == 0:
        log_if_present(logger, "warning", "ITPC dataframe empty; skipping topomaps")
        return

    df_long = _ensure_long_format(itpc_df, logger)
    if df_long is None:
        return

    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)

    bands = sorted(df_long["band"].unique())
    time_bins = sorted(df_long["time_bin"].unique())
    
    if not bands or not time_bins:
        log_if_present(logger, "warning", "No bands or time bins found in ITPC data")
        return

    n_rows = len(bands)
    n_cols = len(time_bins)
    
    itpc_config = plot_cfg.plot_type_configs.get("itpc", {})
    width_per_bin = float(itpc_config.get("width_per_bin", 4.5))
    height_per_band = float(itpc_config.get("height_per_band", 4.0))
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(width_per_bin * n_cols, height_per_band * n_rows), 
        squeeze=False
    )
    
    topomap_data = _prepare_topomap_data(df_long, info)
    all_values = []
    for values_ordered, _ in topomap_data.values():
        all_values.extend([v for v in values_ordered if np.isfinite(v)])
    
    shared_colorbar = bool(config.get("plotting.plots.itpc.shared_colorbar", True))
    im, vmin, vmax = _plot_topomap_grid(
        axes, bands, time_bins, topomap_data, shared_colorbar, all_values, plot_cfg
    )
    
    for row_idx, band in enumerate(bands):
        axes[row_idx, 0].set_ylabel(
            band.capitalize(), 
            fontsize=plot_cfg.font.figure_title, 
            fontweight="bold"
        )
    
    for col_idx, time_bin in enumerate(time_bins):
        axes[0, col_idx].set_title(
            f"{time_bin}\n{axes[0, col_idx].get_title()}", 
            fontweight="bold"
        )
    
    if shared_colorbar and im is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=f"ITPC\\n[{vmin:.3f}, {vmax:.3f}]")
    
    fig.suptitle(f"ITPC Topomaps (sub-{subject})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    output_name = save_dir / f"sub-{subject}_itpc_topomaps_grid"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", 
                  f"Saved ITPC topomap grid ({n_rows} bands × {n_cols} time bins)")


def _get_itpc_columns_for_roi(itpc_df: pd.DataFrame, segment: str, band: str, 
                              roi_name: str, rois: Dict, all_channels: List[str]) -> List[str]:
    """Get ITPC columns filtered by segment, band, and ROI.
    
    Args:
        itpc_df: DataFrame with ITPC columns
        segment: Segment name
        band: Band name
        roi_name: ROI name
        rois: ROI definitions dictionary
        all_channels: List of all available channels
        
    Returns:
        List of matching column names
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.plotting.features.roi import get_roi_channels
    
    cols = []
    roi_channels = (all_channels if roi_name == "all" 
                   else get_roi_channels(rois.get(roi_name, []), all_channels))
    roi_set = set(roi_channels) if roi_channels else set(all_channels)
    
    for col in itpc_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "itpc":
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != band:
            continue
        
        scope = parsed.get("scope") or ""
        if scope in ("global", "roi"):
            cols.append(col)
        elif scope == "ch":
            ch_id = str(parsed.get("identifier") or "")
            if ch_id in roi_set:
                cols.append(col)
    
    return cols


def _extract_bands_from_dataframe(df: pd.DataFrame, config: Optional[Any] = None) -> List[str]:
    """Extract available bands from DataFrame columns.
    
    Args:
        df: DataFrame with feature columns
        config: Optional configuration object for band ordering
        
    Returns:
        Sorted list of unique band names
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.plotting.features.utils import get_band_names
    
    band_set = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "itpc":
            band = parsed.get("band")
            if band:
                band_set.add(str(band))
    
    band_order = get_band_names(config) if config else []
    bands = [b for b in band_order if b in band_set]
    bands += [b for b in sorted(band_set) if b not in bands]
    
    return bands


def _get_roi_names_for_comparison(config: Any, rois: Dict) -> List[str]:
    """Get ROI names to use for comparison plots.
    
    Args:
        config: Configuration object
        rois: ROI definitions dictionary
        
    Returns:
        List of ROI names
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif r in rois:
                roi_names.append(r)
        return roi_names
    
    roi_names = ["all"]
    if rois:
        roi_names.extend(list(rois.keys()))
    
    return roi_names


def plot_itpc_by_condition(
    itpc_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare ITPC between conditions per band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if itpc_df is None or itpc_df.empty or events_df is None:
        return

    if len(itpc_df) != len(events_df):
        log_if_present(
            logger,
            "warning",
            f"ITPC by condition skipped: {len(itpc_df)} rows vs {len(events_df)} events",
        )
        return

    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import (
        plot_paired_comparison,
        get_named_segments,
        get_band_color,
    )
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    from eeg_pipeline.domain.features.naming import NamingSchema

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = get_named_segments(itpc_df, group="itpc")
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                logger.info(f"Auto-detected segments for ITPC comparison: {segments}")
    
    bands = _extract_bands_from_dataframe(itpc_df, config)
    if not bands:
        return

    rois = get_roi_definitions(config)
    all_channels = []
    for col in itpc_df.columns:
        parsed = NamingSchema.parse(str(col))
        if (parsed.get("valid") and parsed.get("group") == "itpc" 
            and parsed.get("scope") == "ch"):
            ch = parsed.get("identifier")
            if ch:
                all_channels.append(str(ch))
    
    roi_names = _get_roi_names_for_comparison(config, rois)
    
    if logger:
        logger.info(
            f"ITPC comparison: segments={segments}, ROIs={roi_names}, "
            f"bands={bands}, compare_windows={compare_wins}, "
            f"compare_columns={compare_cols}"
        )
    
    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    
    if compare_wins and len(segments) >= 2:
        seg1, seg2 = segments[0], segments[1]
        
        for roi_name in roi_names:
            data_by_band = {}
            for band in bands:
                cols1 = _get_itpc_columns_for_roi(itpc_df, seg1, band, roi_name, rois, all_channels)
                cols2 = _get_itpc_columns_for_roi(itpc_df, seg2, band, roi_name, rois, all_channels)
                
                if not cols1 or not cols2:
                    continue
                
                s1 = itpc_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                s2 = itpc_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                valid_mask = s1.notna() & s2.notna()
                v1, v2 = s1[valid_mask].values, s2[valid_mask].values
                
                if len(v1) > 0:
                    data_by_band[band] = (v1, v2)
            
            if data_by_band:
                from eeg_pipeline.utils.formatting import sanitize_label
                roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_itpc_by_condition{suffix}_window"
                
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="ITPC",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        
        log_if_present(logger, "info", 
                      f"Saved ITPC paired comparison plots for {len(roi_names)} ROIs")

    if compare_cols:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=False)
        if not comp_mask_info:
            if logger:
                logger.warning(
                    "Column comparison enabled but config incomplete. "
                    "Set plotting.comparisons.comparison_column and comparison_values."
                )
        else:
            m1, m2, label1, label2 = comp_mask_info
            seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
            
            from eeg_pipeline.plotting.features.utils import compute_or_load_column_stats
            
            segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
            band_colors = {band: get_band_color(band, config) for band in bands}
            n_bands = len(bands)
            n_trials = len(itpc_df)
            
            for roi_name in roi_names:
                cell_data = {}
                for col_idx, band in enumerate(bands):
                    cols = _get_itpc_columns_for_roi(itpc_df, seg_name, band, roi_name, rois, all_channels)
                    
                    if not cols:
                        cell_data[col_idx] = None
                        continue
                    
                    val_series = itpc_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    v1 = val_series[m1].dropna().values
                    v2 = val_series[m2].dropna().values
                    
                    cell_data[col_idx] = {"v1": v1, "v2": v2}
                
                qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                    stats_dir=stats_dir,
                    feature_type="itpc",
                    feature_keys=bands,
                    cell_data=cell_data,
                    config=config,
                    logger=logger,
                )
                
                fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5), squeeze=False)
                
                for col_idx, band in enumerate(bands):
                    ax = axes.flatten()[col_idx]
                    data = cell_data.get(col_idx)
                    
                    if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                        ax.set_xticks([])
                        continue
                    
                    v1, v2 = data["v1"], data["v2"]
                    
                    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=BOXPLOT_WIDTH, patch_artist=True)
                    bp["boxes"][0].set_facecolor(segment_colors["v1"])
                    bp["boxes"][0].set_alpha(0.6)
                    bp["boxes"][1].set_facecolor(segment_colors["v2"])
                    bp["boxes"][1].set_alpha(0.6)
                    
                    jitter_range = SCATTER_JITTER_RANGE
                    ax.scatter(np.random.uniform(-jitter_range, jitter_range, len(v1)), v1, 
                              c=segment_colors["v1"], alpha=0.3, s=6)
                    ax.scatter(1 + np.random.uniform(-jitter_range, jitter_range, len(v2)), v2, 
                              c=segment_colors["v2"], alpha=0.3, s=6)
                    
                    all_vals = np.concatenate([v1, v2])
                    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                    yrange = ymax - ymin if ymax > ymin else 0.1
                    ax.set_ylim(ymin - Y_RANGE_PADDING_LOW * yrange, 
                               ymax + Y_RANGE_PADDING_HIGH * yrange)
                    
                    if col_idx in qvalues:
                        _, q, d, sig = qvalues[col_idx]
                        sig_marker = "†" if sig else ""
                        sig_color = "#d62728" if sig else "#333333"
                        ax.annotate(
                            f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                            xy=(0.5, ymax + STATS_ANNOTATION_OFFSET * yrange),
                            ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                            fontweight="bold" if sig else "normal"
                        )
                    
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label1, label2], fontsize=9)
                    ax.set_title(band.capitalize(), fontweight="bold", 
                               color=band_colors.get(band, "gray"))
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                
                n_tests = len(qvalues)
                roi_display = (roi_name.replace("_", " ").title() 
                              if roi_name != "all" else "All Channels")
                
                stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
                title = (f"ITPC: {label1} vs {label2} (Column Comparison)\n"
                         f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
                         f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                
                plt.tight_layout()
                
                from eeg_pipeline.utils.formatting import sanitize_label
                roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                filename = f"sub-{subject}_itpc_by_condition{suffix}_column"
                
                save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                         bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                plt.close(fig)
            
            log_if_present(logger, "info", 
                          f"Saved ITPC column comparison plots for {len(roi_names)} ROIs")


def _get_pac_plot_cfg(config):
    """Get PAC-specific plot configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (plot_cfg, pac_cfg)
    """
    plot_cfg = get_plot_config(config)
    pac_cfg = plot_cfg.plot_type_configs.get("pac", {})
    return plot_cfg, pac_cfg


def plot_pac_comodulograms(
    pac_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot PAC comodulograms (phase-amplitude coupling matrices) as a single combined figure.
    
    Creates a grid of comodulograms for all ROIs in a single figure to reduce file count
    and provide a comprehensive overview.
    
    Args:
        pac_df: DataFrame with columns: roi, phase_freq, amp_freq, pac, (optional) p_perm, q_perm
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    if pac_df is None or pac_df.empty:
        log_if_present(logger, "warning", "PAC dataframe empty; skipping comodulograms")
        return

    required_cols = {"roi", "phase_freq", "amp_freq", "pac"}
    if not required_cols.issubset(set(pac_df.columns)):
        log_if_present(logger, "warning", 
                      f"PAC dataframe missing required columns {required_cols}; skipping")
        return

    from ...utils.config.loader import get_config_value, ensure_config
    config = ensure_config(config)
    plot_cfg, pac_plot_cfg = _get_pac_plot_cfg(config)
    cmap = pac_plot_cfg.get("cmap", "magma")
    alpha_sig = pac_plot_cfg.get("alpha_sig", 
                                 get_config_value(config, "statistics.sig_alpha", DEFAULT_ALPHA_SIG))
    
    all_pac_values = pac_df["pac"].dropna().values
    if len(all_pac_values) > 0:
        vmin, vmax = _compute_color_limits(all_pac_values, PAC_DEFAULT_MIN, PAC_DEFAULT_MAX)
        vmin = max(0, vmin)
    else:
        vmin, vmax = PAC_DEFAULT_MIN, PAC_DEFAULT_MAX

    ensure_dir(save_dir)
    
    # Get unique ROIs
    rois = sorted(pac_df["roi"].unique())
    n_rois = len(rois)
    
    if n_rois == 0:
        log_if_present(logger, "warning", "No ROIs found in PAC data")
        return
    
    # Create grid layout
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    width_per_col = float(pac_plot_cfg.get("width_per_col", 4.0))
    height_per_row = float(pac_plot_cfg.get("height_per_row", 3.5))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_col * n_cols, height_per_row * n_rows), squeeze=False)
    
    for idx, roi in enumerate(rois):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        df_roi = pac_df[pac_df["roi"] == roi].copy()
        
        if "q_perm" not in df_roi.columns and "p_perm" in df_roi.columns:
            pvals = pd.to_numeric(df_roi["p_perm"], errors="coerce")
            if pvals.notna().any():
                qvals = fdr_bh(pvals.to_numpy(dtype=float), config=config)
                df_roi["q_perm"] = qvals

        phase_freqs = np.sort(df_roi["phase_freq"].unique())
        amp_freqs = np.sort(df_roi["amp_freq"].unique())
        
        if len(phase_freqs) == 0 or len(amp_freqs) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(roi[:20], fontsize=plot_cfg.font.title)
            continue
            
        grid = df_roi.pivot(index="amp_freq", columns="phase_freq", values="pac").reindex(index=amp_freqs, columns=phase_freqs)

        c = ax.pcolormesh(
            phase_freqs,
            amp_freqs,
            grid.to_numpy(),
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        
        # Truncate long ROI names
        roi_short = (roi[:ROI_NAME_MAX_LENGTH] + "..." 
                    if len(roi) > ROI_NAME_MAX_LENGTH else roi)
        ax.set_title(roi_short, fontsize=plot_cfg.font.title, fontweight='bold')
        
        if row == n_rows - 1:
            ax.set_xlabel("Phase (Hz)", fontsize=plot_cfg.font.large)
        if col == 0:
            ax.set_ylabel("Amp (Hz)", fontsize=plot_cfg.font.large)
        
        ax.tick_params(labelsize=8)

        if "q_perm" in df_roi.columns:
            q_grid = df_roi.pivot(index="amp_freq", columns="phase_freq", values="q_perm").reindex(index=amp_freqs, columns=phase_freqs)
            sig_mask = (q_grid.to_numpy() <= alpha_sig)
            if np.any(sig_mask):
                ax.contour(phase_freqs, amp_freqs, sig_mask, colors="white", linewidths=0.8, linestyles="--")
    
    # Hide empty subplots
    for idx in range(n_rois, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(c, cax=cbar_ax)
    cbar.set_label("PAC (MI)", fontsize=plot_cfg.font.title)
    
    fig.suptitle(f"Phase-Amplitude Coupling Comodulograms\nsub-{subject} | {n_rois} ROIs",
                fontsize=plot_cfg.font.figure_title, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    output_name = save_dir / f"sub-{subject}_pac_comodulograms_grid"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved PAC comodulograms grid ({n_rois} ROIs)")


def plot_pac_time_ribbons(
    pac_time_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot PAC over time (time x amp_freq) for each ROI and phase frequency.
    
    Args:
        pac_time_df: DataFrame with columns: roi, phase_freq, amp_freq, time, pac
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    if pac_time_df is None or pac_time_df.empty:
        return

    required_cols = {"roi", "phase_freq", "amp_freq", "time", "pac"}
    if not required_cols.issubset(set(pac_time_df.columns)):
        return

    plot_cfg, pac_plot_cfg = _get_pac_plot_cfg(config)
    cmap = pac_plot_cfg.get("cmap", "magma")
    ensure_dir(save_dir)

    group_cols = ["roi", "phase_freq"]
    has_segment = "segment" in pac_time_df.columns
    if has_segment:
        group_cols = ["segment"] + group_cols

    for key, df_sub in pac_time_df.groupby(group_cols):
        if has_segment:
            segment, roi, phase_f = key
        else:
            segment, roi, phase_f = None, key[0], key[1]
        times = np.sort(df_sub["time"].unique())
        amp_freqs = np.sort(df_sub["amp_freq"].unique())
        grid = df_sub.pivot(index="amp_freq", columns="time", values="pac").reindex(index=amp_freqs, columns=times)

        fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="pac"))
        grid_values = grid.to_numpy().flatten()
        if np.any(np.isfinite(grid_values)):
            vmin, vmax = _compute_color_limits(grid_values, 0.0, 1.0)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-3
        else:
            vmin, vmax = 0.0, 1.0
        c = ax.pcolormesh(times, amp_freqs, grid.to_numpy(), cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude frequency (Hz)")
        title = f"PAC over time (ROI: {roi}, phase {phase_f:.1f} Hz)"
        if segment is not None:
            title = f"{title} | Segment: {segment}"
        ax.set_title(title)
        plt.colorbar(c, ax=ax, label="PAC")

        out_name = f"sub-{subject}_pac_time_roi-{roi}_phase-{phase_f:.1f}"
        if segment is not None:
            out_name = f"{out_name}_segment-{sanitize_label(segment)}"
        out = save_dir / out_name
        save_fig(
            fig,
            out,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        plt.close(fig)
        log_if_present(logger, "info", f"Saved PAC time ribbon for ROI {roi}, phase {phase_f:.1f}")


def _get_pac_columns_for_roi(pac_df: pd.DataFrame, segment: str, pair: str, 
                             roi_name: str, rois: Dict, all_channels: List[str]) -> List[str]:
    """Get PAC columns filtered by segment, pair, and ROI.
    
    Args:
        pac_df: DataFrame with PAC columns
        segment: Segment name
        pair: Phase-amplitude pair name
        roi_name: ROI name
        rois: ROI definitions dictionary
        all_channels: List of all available channels
        
    Returns:
        List of matching column names
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.plotting.features.roi import get_roi_channels
    
    cols = []
    roi_channels = (all_channels if roi_name == "all" 
                   else get_roi_channels(rois.get(roi_name, []), all_channels))
    roi_set = set(roi_channels) if roi_channels else set(all_channels)
    
    for col in pac_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "pac":
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != pair:
            continue
        
        scope = parsed.get("scope") or ""
        if scope in ("global", "roi"):
            cols.append(col)
        elif scope == "ch":
            ch_id = str(parsed.get("identifier") or "")
            if ch_id in roi_set:
                cols.append(col)
    
    return cols


def _extract_pac_pairs_from_dataframe(pac_df: pd.DataFrame, config: Any) -> List[str]:
    """Extract available PAC pairs from DataFrame columns.
    
    Args:
        pac_df: DataFrame with PAC columns
        config: Configuration object
        
    Returns:
        Sorted list of unique pair names
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.utils.config.loader import get_config_value
    
    all_pairs = set()
    for col in pac_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "pac":
            band = parsed.get("band")
            if band:
                all_pairs.add(str(band))
    
    cfg_pairs = get_config_value(config, "plotting.plots.features.pac_pairs", None)
    if cfg_pairs is None:
        cfg_pairs = get_config_value(config, "feature_engineering.pac.pairs", None)
    
    ordered_pairs = []
    for entry in cfg_pairs or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            ordered_pairs.append(f"{entry[0]}_{entry[1]}")
        elif isinstance(entry, str):
            ordered_pairs.append(entry)
    
    if ordered_pairs:
        pairs = ([p for p in ordered_pairs if p in all_pairs] + 
                [p for p in sorted(all_pairs) if p not in ordered_pairs])
    else:
        pairs = sorted(all_pairs)
    
    return pairs


def plot_pac_by_condition(
    pac_trials_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare PAC between conditions per phase-amplitude pair.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if pac_trials_df is None or pac_trials_df.empty or events_df is None:
        return
    if len(pac_trials_df) != len(events_df):
        log_if_present(
            logger,
            "warning",
            f"PAC by condition skipped: {len(pac_trials_df)} rows vs {len(events_df)} events",
        )
        return

    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import plot_paired_comparison, get_named_segments
    from eeg_pipeline.plotting.features.roi import get_roi_definitions

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = get_named_segments(pac_trials_df, group="pac")
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                logger.info(f"Auto-detected segments for PAC comparison: {segments}")
    
    pairs = _extract_pac_pairs_from_dataframe(pac_trials_df, config)
    if not pairs:
        return
    
    rois = get_roi_definitions(config)
    all_channels = []
    for col in pac_trials_df.columns:
        parsed = NamingSchema.parse(str(col))
        if (parsed.get("valid") and parsed.get("group") == "pac" 
            and parsed.get("scope") == "ch"):
            ch = parsed.get("identifier")
            if ch:
                all_channels.append(str(ch))
    
    roi_names = _get_roi_names_for_comparison(config, rois)
    
    if logger:
        logger.info(
            f"PAC comparison: segments={segments}, ROIs={roi_names}, pairs={pairs}, "
            f"compare_windows={compare_wins}, compare_columns={compare_cols}"
        )
    
    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    
    if compare_wins and len(segments) >= 2:
        seg1, seg2 = segments[0], segments[1]
        
        for roi_name in roi_names:
            data_by_band = {}
            for pair in pairs:
                cols1 = _get_pac_columns_for_roi(pac_trials_df, seg1, pair, roi_name, rois, all_channels)
                cols2 = _get_pac_columns_for_roi(pac_trials_df, seg2, pair, roi_name, rois, all_channels)
                
                if not cols1 or not cols2:
                    continue
                
                s1 = pac_trials_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                s2 = pac_trials_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                valid_mask = s1.notna() & s2.notna()
                v1, v2 = s1[valid_mask].values, s2[valid_mask].values
                
                if len(v1) > 0:
                    data_by_band[pair] = (v1, v2)
            
            if data_by_band:
                from eeg_pipeline.utils.formatting import sanitize_label
                roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_pac_by_condition{suffix}_window"
                
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="PAC",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        
        log_if_present(logger, "info", 
                      f"Saved PAC paired comparison plots for {len(roi_names)} ROIs")

    if compare_cols:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=False)
        if not comp_mask_info:
            if logger:
                logger.warning(
                    "Column comparison enabled but config incomplete. "
                    "Set plotting.comparisons.comparison_column and comparison_values."
                )
        else:
            m1, m2, label1, label2 = comp_mask_info
            seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
            
            from eeg_pipeline.plotting.features.utils import compute_or_load_column_stats
            
            segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
            n_pairs = len(pairs)
            n_trials = len(pac_trials_df)
            
            for roi_name in roi_names:
                cell_data = {}
                for col_idx, pair in enumerate(pairs):
                    cols = _get_pac_columns_for_roi(pac_trials_df, seg_name, pair, roi_name, rois, all_channels)
                    
                    if not cols:
                        cell_data[col_idx] = None
                        continue
                    
                    val_series = pac_trials_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    v1 = val_series[m1].dropna().values
                    v2 = val_series[m2].dropna().values
                    
                    cell_data[col_idx] = {"v1": v1, "v2": v2}
                
                qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                    stats_dir=stats_dir,
                    feature_type="pac",
                    feature_keys=pairs,
                    cell_data=cell_data,
                    config=config,
                    logger=logger,
                )
                
                fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 5), squeeze=False)
                
                for col_idx, pair in enumerate(pairs):
                    ax = axes.flatten()[col_idx]
                    data = cell_data.get(col_idx)
                    
                    if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                        ax.set_xticks([])
                        continue
                    
                    v1, v2 = data["v1"], data["v2"]
                    
                    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=BOXPLOT_WIDTH, patch_artist=True)
                    bp["boxes"][0].set_facecolor(segment_colors["v1"])
                    bp["boxes"][0].set_alpha(0.6)
                    bp["boxes"][1].set_facecolor(segment_colors["v2"])
                    bp["boxes"][1].set_alpha(0.6)
                    
                    jitter_range = SCATTER_JITTER_RANGE
                    ax.scatter(np.random.uniform(-jitter_range, jitter_range, len(v1)), v1, 
                              c=segment_colors["v1"], alpha=0.3, s=6)
                    ax.scatter(1 + np.random.uniform(-jitter_range, jitter_range, len(v2)), v2, 
                              c=segment_colors["v2"], alpha=0.3, s=6)
                    
                    all_vals = np.concatenate([v1, v2])
                    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                    yrange = ymax - ymin if ymax > ymin else 0.1
                    ax.set_ylim(ymin - Y_RANGE_PADDING_LOW * yrange, 
                               ymax + Y_RANGE_PADDING_HIGH * yrange)
                    
                    if col_idx in qvalues:
                        _, q, d, sig = qvalues[col_idx]
                        sig_marker = "†" if sig else ""
                        sig_color = "#d62728" if sig else "#333333"
                        ax.annotate(
                            f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                            xy=(0.5, ymax + STATS_ANNOTATION_OFFSET * yrange),
                            ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                            fontweight="bold" if sig else "normal"
                        )
                    
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label1, label2], fontsize=9)
                    ax.set_title(pair.replace("_", "→"), fontweight="bold", color="#8E44AD")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                
                n_tests = len(qvalues)
                roi_display = (roi_name.replace("_", " ").title() 
                              if roi_name != "all" else "All Channels")
                
                stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
                title = (f"PAC: {label1} vs {label2} (Column Comparison)\n"
                         f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
                         f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                
                plt.tight_layout()
                
                roi_safe = roi_name.replace("_", " ").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                filename = f"sub-{subject}_pac_by_condition{suffix}_column"
                
                save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                         bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                plt.close(fig)
            
            log_if_present(logger, "info", 
                          f"Saved PAC column comparison plots for {len(roi_names)} ROIs")


def plot_pac_summary(
    pac_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """PAC summary showing aggregate coupling across frequency pairs.
    
    Answers: "What is the overall PAC profile?"
    """
    if pac_df is None or pac_df.empty:
        return
    
    plot_cfg = get_plot_config(config)
    
    pac_cols = [c for c in pac_df.columns if c.startswith("pac_") or "pac" in c.lower()]
    
    if not pac_cols:
        return
    
    phase_bands = ["delta", "theta", "alpha"]
    amp_bands = ["alpha", "beta", "gamma", "highgamma"]
    
    pac_matrix = np.zeros((len(phase_bands), len(amp_bands)))
    pac_counts = np.zeros((len(phase_bands), len(amp_bands)))
    
    for col in pac_cols:
        col_lower = col.lower()
        for pi, phase in enumerate(phase_bands):
            for ai, amp in enumerate(amp_bands):
                if phase in col_lower and amp in col_lower:
                    vals = pac_df[col].dropna().values
                    if len(vals) > 0:
                        pac_matrix[pi, ai] += np.mean(vals)
                        pac_counts[pi, ai] += 1
    
    pac_counts[pac_counts == 0] = 1
    pac_matrix = pac_matrix / pac_counts
    
    if pac_matrix.sum() == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ax1 = axes[0]
    im = ax1.imshow(pac_matrix, cmap="hot", aspect="auto")
    ax1.set_xticks(range(len(amp_bands)))
    ax1.set_xticklabels([b.capitalize() for b in amp_bands])
    ax1.set_yticks(range(len(phase_bands)))
    ax1.set_yticklabels([b.capitalize() for b in phase_bands])
    ax1.set_xlabel("Amplitude Band")
    ax1.set_ylabel("Phase Band")
    ax1.set_title("Mean PAC Matrix", fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="PAC")
    
    for i in range(len(phase_bands)):
        for j in range(len(amp_bands)):
            if pac_matrix[i, j] > 0:
                ax1.text(j, i, f"{pac_matrix[i, j]:.3f}", ha="center", va="center", 
                        fontsize=8, color="white" if pac_matrix[i, j] > pac_matrix.max()/2 else "black")
    
    ax2 = axes[1]
    all_pac_vals = []
    for col in pac_cols:
        vals = pac_df[col].dropna().values
        all_pac_vals.extend(vals)
    
    if all_pac_vals:
        ax2.hist(all_pac_vals, bins=30, color="#14B8A6", alpha=0.7, edgecolor="white")
        ax2.axvline(np.mean(all_pac_vals), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {np.mean(all_pac_vals):.3f}")
        ax2.axvline(np.median(all_pac_vals), color="orange", linestyle=":", linewidth=2,
                   label=f"Median: {np.median(all_pac_vals):.3f}")
        ax2.set_xlabel("PAC Value")
        ax2.set_ylabel("Count")
        ax2.set_title("PAC Distribution", fontweight="bold")
        ax2.legend()
    
    ax3 = axes[2]
    phase_means = pac_matrix.mean(axis=1)
    amp_means = pac_matrix.mean(axis=0)
    
    x = np.arange(max(len(phase_bands), len(amp_bands)))
    width = 0.35
    
    ax3.bar(x[:len(phase_bands)] - width/2, phase_means, width, label="Phase", color="#3B82F6", alpha=0.8)
    ax3.bar(x[:len(amp_bands)] + width/2, amp_means, width, label="Amplitude", color="#22C55E", alpha=0.8)
    
    all_labels = phase_bands + amp_bands[len(phase_bands):]
    ax3.set_xticks(x[:len(all_labels)])
    ax3.set_xticklabels([b[:3].capitalize() for b in all_labels])
    ax3.set_ylabel("Mean PAC")
    ax3.set_title("PAC by Band Role", fontweight="bold")
    ax3.legend()
    
    fig.suptitle(f"Phase-Amplitude Coupling Summary (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    max_pac = pac_matrix.max()
    max_idx = np.unravel_index(pac_matrix.argmax(), pac_matrix.shape)
    max_pair = f"{phase_bands[max_idx[0]]}-{amp_bands[max_idx[1]]}"
    footer = f"n={len(pac_df)} trials | Strongest coupling: {max_pair} (PAC={max_pac:.3f})"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    ensure_dir(save_dir)
    save_fig(fig, save_dir / f"sub-{subject}_pac_summary",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved PAC summary (max: {max_pair}={max_pac:.3f})")
