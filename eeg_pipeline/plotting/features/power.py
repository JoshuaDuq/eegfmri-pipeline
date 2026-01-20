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
from eeg_pipeline.utils.data.features import get_power_columns_by_band
from eeg_pipeline.utils.config.loader import get_frequency_bands
from scipy.stats import mannwhitneyu


###################################################################
# Constants
###################################################################

FDR_ALPHA_DEFAULT = 0.05
MIN_TRIALS_FOR_STATISTICS = 3
MIN_CHANNELS_FOR_TOPO = 3
HEATMAP_TEXT_THRESHOLD = 200
MIN_TRIALS_FOR_VARIABILITY = 5
MIN_EPOCHS_FOR_SEM = 2
BAR_LABEL_OFFSET = 0.02
HISTOGRAM_BINS = 15
MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 10


###################################################################
# Helper Functions
###################################################################


def _extract_channel_names_from_columns(band_cols: List[str]) -> List[str]:
    """Extract channel names from power column names."""
    channel_names = []
    for col in band_cols:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "power":
            ident = parsed.get("identifier")
            if ident:
                channel_names.append(str(ident))
                continue
        channel_names.append(str(col))
    return channel_names


def _build_channel_data_map(mean_power: pd.Series) -> Dict[str, float]:
    """Build a mapping from channel names to power values."""
    data_map = {}
    for col, val in mean_power.items():
        parsed = NamingSchema.parse(str(col))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        ident = parsed.get("identifier")
        if ident:
            data_map[str(ident)] = val
    return data_map


def _compute_heatmap_color_limits(
    heatmap_data: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[float, float]:
    """Compute color limits for heatmap display."""
    if vmin is None or vmax is None:
        data_min = float(np.nanmin(heatmap_data))
        data_max = float(np.nanmax(heatmap_data))
        vmax_abs = max(abs(data_min), abs(data_max))
        if vmin is None:
            vmin = -vmax_abs
        if vmax is None:
            vmax = vmax_abs
    return vmin, vmax


def _get_comparison_segments(
    power_df: pd.DataFrame,
    config: Any,
    logger: Optional[logging.Logger],
) -> List[str]:
    """Extract comparison segments from config or auto-detect from data."""
    from eeg_pipeline.utils.config.loader import get_config_value
    from .utils import get_named_segments
    
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = get_named_segments(power_df, group="power")
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                logger.info(f"Auto-detected segments for comparison: {segments}")
    return segments


def _get_comparison_rois(
    config: Any,
    rois: Dict[str, Any],
) -> List[str]:
    """Determine which ROIs to plot for comparisons."""
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


def _get_power_columns_for_roi(
    power_df: pd.DataFrame,
    segment: str,
    band: str,
    roi_channels: List[str],
) -> List[str]:
    """Get power columns for a specific segment, band, and ROI channels."""
    roi_set = set(roi_channels)
    cols = []
    for c in power_df.columns:
        parsed = NamingSchema.parse(str(c))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != band:
            continue
        channel_id = str(parsed.get("identifier") or "")
        if channel_id and channel_id not in roi_set:
            continue
        cols.append(c)
    return cols


def _extract_band_data_for_roi(
    power_df: pd.DataFrame,
    bands: List[str],
    segments: List[str],
    roi_channels: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract power data by band for two segments within an ROI."""
    roi_set = set(roi_channels)
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
            if str(parsed.get("segment") or "") == segments[0] and str(parsed.get("band") or "") == band:
                cols1.append(c)
            if str(parsed.get("segment") or "") == segments[1] and str(parsed.get("band") or "") == band:
                cols2.append(c)
        
        if cols1 and cols2:
            s1 = power_df[cols1].mean(axis=1)
            s2 = power_df[cols2].mean(axis=1)
            valid_mask = s1.notna() & s2.notna()
            v1, v2 = s1[valid_mask].values, s2[valid_mask].values
            if len(v1) > 0:
                data_by_band[band] = (v1, v2)
    
    return data_by_band


def _extract_multi_segment_data_for_roi(
    power_df: pd.DataFrame,
    bands: List[str],
    segments: List[str],
    roi_channels: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract power data by band for multiple segments within an ROI.
    
    Returns:
        Dict mapping band -> {segment_name -> values array}
    """
    from .utils import extract_multi_segment_data
    return extract_multi_segment_data(
        df=power_df,
        group="power",
        bands=bands,
        segments=segments,
        identifiers=roi_channels,
    )


def _extract_column_comparison_data(
    power_df: pd.DataFrame,
    bands: List[str],
    seg_name: str,
    roi_channels: List[str],
) -> Dict[int, pd.Series]:
    """Extract power data by band for column comparison within an ROI."""
    roi_set = set(roi_channels)
    cell_data = {}
    
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
        else:
            val_series = power_df[cols].mean(axis=1)
            cell_data[col_idx] = val_series
    
    return cell_data


def _compute_column_comparison_statistics(
    cell_data: Dict[int, Optional[pd.Series]],
    m1: np.ndarray,
    m2: np.ndarray,
    bands: List[str],
    config: Any,
) -> Tuple[Dict[int, Tuple[float, float, float, bool]], int]:
    """Compute Mann-Whitney U statistics for column comparison."""
    from .utils import apply_fdr_correction
    
    all_pvals = []
    pvalue_keys = []
    
    for col_idx, band in enumerate(bands):
        val_series = cell_data.get(col_idx)
        if val_series is None:
            continue
        
        v1 = val_series[m1].dropna().values
        v2 = val_series[m2].dropna().values
        
        if len(v1) < MIN_TRIALS_FOR_STATISTICS or len(v2) < MIN_TRIALS_FOR_STATISTICS:
            continue
        
        try:
            _, p = mannwhitneyu(v1, v2, alternative="two-sided")
            mean_diff = np.mean(v2) - np.mean(v1)
            pooled_std = np.sqrt(
                ((len(v1) - 1) * np.var(v1, ddof=1) + (len(v2) - 1) * np.var(v2, ddof=1))
                / (len(v1) + len(v2) - 2)
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            all_pvals.append(p)
            pvalue_keys.append((col_idx, p, cohens_d))
        except Exception:
            pass
    
    qvalues = {}
    n_significant = 0
    if all_pvals:
        rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
        n_significant = int(np.sum(rejected))
    
    return qvalues, n_significant


def _plot_window_comparison(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segments: List[str],
    bands: List[str],
    roi_names: List[str],
    rois: Dict[str, Any],
    all_channels: List[str],
    stats_dir: Optional[Path],
) -> None:
    """Plot window comparison (paired) for power by condition.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from .utils import plot_paired_comparison, plot_multi_window_comparison
    from eeg_pipeline.utils.formatting import sanitize_label
    
    use_multi_window = len(segments) > 2
    
    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)
        
        if not roi_channels:
            continue
        
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        suffix = f"_roi-{roi_safe}" if roi_safe else ""
        
        if use_multi_window:
            data_by_band = _extract_multi_segment_data_for_roi(
                power_df, bands, segments, roi_channels
            )
            
            if not data_by_band:
                continue
            
            save_path = save_dir / f"sub-{subject}_power_by_condition{suffix}_multiwindow"
            
            plot_multi_window_comparison(
                data_by_band=data_by_band,
                subject=subject,
                save_path=save_path,
                feature_label="Band Power",
                segments=segments,
                config=config,
                logger=logger,
                roi_name=roi_name,
                stats_dir=stats_dir,
            )
        else:
            seg1, seg2 = segments[0], segments[1]
            data_by_band = _extract_band_data_for_roi(
                power_df, bands, [seg1, seg2], roi_channels
            )
            
            if not data_by_band:
                continue
            
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


def _plot_column_comparison(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    bands: List[str],
    roi_names: List[str],
    rois: Dict[str, Any],
    all_channels: List[str],
    stats_dir: Optional[Path],
) -> None:
    """Plot column comparison (unpaired) for power by condition.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.utils.config.loader import get_config_value
    from .utils import load_precomputed_paired_stats, get_precomputed_qvalues, apply_fdr_correction, plot_multi_group_column_comparison, get_named_segments
    
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=False)
        if not multi_group_info:
            if logger:
                logger.warning("Multi-group column comparison enabled but config incomplete.")
            return
        
        masks_dict, group_labels = multi_group_info
        seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", None)
        if not seg_name:
            if logger:
                logger.error("Column comparison requires plotting.comparisons.comparison_segment.")
            return
        
        for roi_name in roi_names:
            if roi_name == "all":
                roi_channels = all_channels
            else:
                roi_channels = get_roi_channels(rois[roi_name], all_channels)
            
            if not roi_channels:
                continue
            
            data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                cols = _get_power_columns_for_roi(power_df, seg_name, band, roi_channels)
                if not cols:
                    continue
                
                val_series = power_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                group_values = {}
                for label, mask in masks_dict.items():
                    vals = val_series[mask].dropna().values
                    if len(vals) > 0:
                        group_values[label] = vals
                
                if len(group_values) >= 2:
                    data_by_band[band] = group_values
            
            if data_by_band:
                from eeg_pipeline.utils.formatting import sanitize_label
                roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_power_by_condition{suffix}_multigroup"
                
                plot_multi_group_column_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="Band Power",
                    groups=group_labels,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        
        if logger:
            logger.info(f"Saved power multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=False)
    if not comp_mask_info:
        if logger:
            logger.warning(
                "Column comparison enabled but config incomplete. "
                "Set plotting.comparisons.comparison_column and comparison_values."
            )
        return
    
    m1, m2, label1, label2 = comp_mask_info
    
    seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", None)
    if not seg_name or seg_name.strip() == "":
        if logger:
            logger.error(
                "Column comparison requires plotting.comparisons.comparison_segment to be set. "
                "No fallback will be used."
            )
        return
    
    available_segments = get_named_segments(power_df, group="power")
    if seg_name not in available_segments:
        if logger:
            logger.error(
                f"Configured segment '{seg_name}' not found in data. "
                f"Available segments: {available_segments}. "
                "Column comparison will not be performed."
            )
        return
    
    plot_cfg = get_plot_config(config)
    segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
    band_colors = {band: get_band_color(band, config) for band in bands}
    
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
    
    use_precomputed = precomputed_column_stats is not None and not precomputed_column_stats.empty
    
    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)
        
        if not roi_channels:
            continue
        
        cell_data = _extract_column_comparison_data(
            power_df, bands, seg_name, roi_channels
        )
        
        bands_with_data = [band for col_idx, band in enumerate(bands) if cell_data.get(col_idx) is not None]
        if not bands_with_data:
            if logger:
                logger.error(
                    f"No power data found for segment '{seg_name}' in ROI {roi_name}. "
                    f"Skipping column comparison plot for this ROI."
                )
            continue
        
        plot_data = {}
        for col_idx, band in enumerate(bands):
            val_series = cell_data.get(col_idx)
            if val_series is None:
                plot_data[col_idx] = None
                continue
            v1 = val_series[m1].dropna().values
            v2 = val_series[m2].dropna().values
            if len(v1) == 0 or len(v2) == 0:
                if logger:
                    logger.warning(
                        f"No data for band {band} after filtering by conditions. "
                        f"v1: {len(v1)} trials, v2: {len(v2)} trials"
                    )
                plot_data[col_idx] = None
                continue
            plot_data[col_idx] = {"v1": v1, "v2": v2}
        
        if use_precomputed:
            qvalues = get_precomputed_qvalues(precomputed_column_stats, bands, roi_name or "all")
            n_significant = sum(1 for v in qvalues.values() if v[3])
            
            for col_idx, band in enumerate(bands):
                if band in qvalues:
                    p, q, d, sig = qvalues[band]
                    qvalues[col_idx] = (p, q, d, sig)
        else:
            qvalues, n_significant = _compute_column_comparison_statistics(
                cell_data, m1, m2, bands, config
            )
        
        fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 5), squeeze=False)
        
        for col_idx, band in enumerate(bands):
            ax = axes.flatten()[col_idx]
            data = plot_data.get(col_idx)
            
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
            ymin = np.nanmin(all_vals)
            ymax = np.nanmax(all_vals)
            yrange = ymax - ymin if ymax > ymin else 0.1
            y_padding_bottom = 0.1 * yrange
            y_padding_top = 0.3 * yrange
            ax.set_ylim(ymin - y_padding_bottom, ymax + y_padding_top)
            
            if col_idx in qvalues:
                _, q, d, sig = qvalues[col_idx]
                sig_marker = "†" if sig else ""
                sig_color = "#d62728" if sig else "#333333"
                annotation_y = ymax + 0.05 * yrange
                ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", xy=(0.5, annotation_y),
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
        
        from eeg_pipeline.utils.formatting import sanitize_label
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        suffix = f"_roi-{roi_safe}" if roi_safe else ""
        filename = f"sub-{subject}_power_by_condition{suffix}_column"
        
        save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                 bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
        plt.close(fig)
        
        if logger:
            logger.info(f"Saved power column comparison for ROI {roi_display} ({n_significant}/{n_tests} FDR significant, {stats_source})")


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

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = _get_comparison_segments(power_df, config, logger)
    bands = list(get_frequency_band_names(config) or ["delta", "theta", "alpha", "beta", "gamma"])
    
    rois = get_roi_definitions(config)
    all_channels = extract_channels_from_columns(list(power_df.columns))
    roi_names = _get_comparison_rois(config, rois)
    
    if logger:
        logger.info(f"Power comparison: segments={segments}, ROIs={roi_names}, compare_windows={compare_wins}, compare_columns={compare_cols}")
    
    if compare_wins and len(segments) >= 2:
        _plot_window_comparison(
            power_df, events_df, subject, save_dir, logger, config,
            segments, bands, roi_names, rois, all_channels, stats_dir
        )

    if compare_cols:
        _plot_column_comparison(
            power_df, events_df, subject, save_dir, logger, config,
            bands, roi_names, rois, all_channels, stats_dir
        )





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
    tfr: Any,
    events_df: Optional[pd.DataFrame],
    *,
    config: Any,
    subject: str,
    logger: logging.Logger,
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
    if config is None:
        logger.warning("Config is required for temperature plotting; skipping.")
        return None

    if events_df is None or events_df.empty:
        logger.warning("No events DataFrame provided for temperature analysis")
        return None
        
    temp_col = find_temperature_column_in_events(events_df, config)
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
        
    freq_bands = get_frequency_bands(config)
    if not freq_bands or band not in freq_bands:
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
        band_str = str(band)
        band_cols = [
            c for c in power_cols_by_band.get(band_str, [])
            if NamingSchema.parse(str(c)).get("scope") == "ch"
        ]
        if not band_cols:
            continue
        
        band_data = pow_df[band_cols].mean(axis=0)
        band_means.append(band_data.values)
        valid_bands.append(band_str)
        
        if not channel_names:
            channel_names = _extract_channel_names_from_columns(band_cols)
    
    if not band_means:
        logger.warning("No valid band data for heatmap")
        return
    
    features_config = plot_cfg.plot_type_configs.get("features", {})
    power_config = features_config.get("power", {})
    
    heatmap_data = np.array(band_means)
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    vmin_config = power_config.get("vmin")
    vmax_config = power_config.get("vmax")
    vmin, vmax = _compute_heatmap_color_limits(heatmap_data, vmin_config, vmax_config)
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
    
    heatmap_threshold = features_config.get("heatmap_text_threshold", HEATMAP_TEXT_THRESHOLD)
    n_cells = len(channel_names) * len(valid_bands)
    if n_cells <= heatmap_threshold:
        std_threshold = np.std(heatmap_data)
        fontsize_base = heatmap_threshold / len(channel_names)
        fontsize = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, fontsize_base))
        for i in range(len(valid_bands)):
            for j in range(len(channel_names)):
                value = heatmap_data[i, j]
                text = f'{value:.2f}'
                is_high_value = abs(value) > std_threshold
                text_color = 'white' if is_high_value else 'black'
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=fontsize)
    
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f'sub-{subject}_channel_power_heatmap',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config
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
    features_freq_bands = get_frequency_bands(config)
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
        if n_epochs >= MIN_EPOCHS_FOR_SEM:
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
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
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
    MIN_TEMPERATURES_FOR_COMPARISON = 2
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < MIN_TEMPERATURES_FOR_COMPARISON:
        return False
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    COLOR_MAP_START = 0.15
    COLOR_MAP_END = 0.85
    temp_colors = plt.cm.coolwarm(np.linspace(COLOR_MAP_START, COLOR_MAP_END, len(unique_temps)))
    
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
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
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
    
    freq_bands = get_frequency_bands(config)
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
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
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

    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", True)
    comp_mask_info = extract_comparison_mask(events_df, config) if compare_cols else None
    
    if not comp_mask_info:
        logger.warning("No valid comparison defined for PSD comparison")
        return
    
    m1, m2, label1, label2 = comp_mask_info

    if len(tfr) != len(events_df):
        raise ValueError(
            f"TFR ({len(tfr)} epochs) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )
    
    if m1.sum() < 1 or m2.sum() < 1:
        logger.warning(f"Insufficient trials for {label1} vs {label2} comparison")
        return
    
    n1, n2 = int(m1.sum()), int(m2.sum())
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mask, label, color, n_trials in [
        (m1, label1, 'steelblue', n1),
        (m2, label2, 'orangered', n2)
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
    
    fig.suptitle(f"PSD Comparison: {label1} vs {label2}", fontsize=plot_cfg.font.figure_title, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = save_dir / f'sub-{subject}_power_spectral_density_condition'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
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
    temps = _validate_temperature_data(tfr, events_df, config=config, subject=subject, logger=logger)
    if temps is None:
        return
    
    freq_mask = _get_band_frequency_mask(tfr, band, config, logger)
    if freq_mask is None:
        return
    
    COLOR_MAP_START = 0.15
    COLOR_MAP_END = 0.85
    unique_temps = sorted(temps.dropna().unique())
    colors = plt.cm.coolwarm(np.linspace(COLOR_MAP_START, COLOR_MAP_END, len(unique_temps)))
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    freq_bands = get_frequency_bands(config)
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
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    logger.info("Saved band time course by temperature (Induced)")


def plot_inter_band_spatial_power_correlation(
    tfr: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot inter-band spatial power correlation matrix."""
    features_freq_bands = get_frequency_bands(config)
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    
    times = np.asarray(tfr.times)
    active_window = config.get("time_frequency_analysis.active_window", [3.0, 10.5])
    active_start = float(active_window[0])
    active_end = float(active_window[1])
    tmin_clip = float(max(times.min(), active_start))
    tmax_clip = float(min(times.max(), active_end))
    
    is_valid_window = np.isfinite(tmin_clip) and np.isfinite(tmax_clip) and (tmax_clip > tmin_clip)
    if not is_valid_window:
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
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    logger.info("Saved inter-band spatial power correlation")
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
    
    from .utils import apply_fdr_correction, get_fdr_alpha
    from scipy.stats import spearmanr
    
    plot_cfg = get_plot_config(config)
    
    band_power = {}
    for band in bands:
        band_str = str(band)
        cols = [c for c in pow_df.columns 
                if c.startswith("power_") and f"_{band_str}_" in c and "_ch_" in c]
        if not cols:
            continue
        vals = pow_df[cols].mean(axis=1).dropna().values
        if len(vals) >= MIN_TRIALS_FOR_VARIABILITY:
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
                correlation, p_value = spearmanr(vals1[:min_len], vals2[:min_len])
                corr_matrix[i, j] = correlation
                pval_matrix[i, j] = p_value
    
    upper_tri_idx = np.triu_indices(n_bands, k=1)
    pvals_upper = pval_matrix[upper_tri_idx]
    
    alpha_fdr = get_fdr_alpha(config)

    if len(pvals_upper) > 0:
        rejected, qvals, alpha_fdr_arr = apply_fdr_correction(list(pvals_upper), config=config)
        try:
            alpha_fdr_flat = np.asarray(alpha_fdr_arr).ravel()
            alpha_fdr = float(alpha_fdr_flat[0])
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
    is_significant = np.isfinite(qval_matrix) & (qval_matrix < alpha_fdr)
    diagonal_mask = np.eye(n_bands, dtype=bool)
    sig_mask = is_significant | diagonal_mask
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
            correlation = corr_matrix[i, j]
            p_value = pval_matrix[i, j]
            q_value = qval_matrix[i, j]
            
            is_high_correlation = abs(correlation) > 0.5
            text_color = "white" if is_high_correlation else "black"
            
            if i != j:
                is_fdr_significant = q_value < FDR_ALPHA_DEFAULT
                is_uncorrected_significant = p_value < FDR_ALPHA_DEFAULT
                sig_marker = "†" if is_fdr_significant else ("*" if is_uncorrected_significant else "")
                text = f"{correlation:.2f}{sig_marker}"
            else:
                text = "1.00"
                is_fdr_significant = False
            
            fontweight = "bold" if is_fdr_significant else "normal"
            ax1.text(j, i, text, ha="center", va="center", color=text_color, 
                    fontsize=plot_cfg.font.small, fontweight=fontweight)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Spearman ρ", fontsize=plot_cfg.font.label)
    
    ax2 = axes[1]
    EPSILON = 1e-10
    MIN_LOG_PVAL_DISPLAY = 3
    log_pvals = -np.log10(pval_matrix + EPSILON)
    np.fill_diagonal(log_pvals, 0)
    
    max_log_pval = np.max(log_pvals)
    vmax_log_pval = max(MIN_LOG_PVAL_DISPLAY, max_log_pval)
    im2 = ax2.imshow(log_pvals, cmap="YlOrRd", vmin=0, vmax=vmax_log_pval, aspect="equal")
    ax2.set_xticks(range(n_bands))
    ax2.set_yticks(range(n_bands))
    ax2.set_xticklabels([b.capitalize() for b in valid_bands], rotation=45, ha="right",
                        fontsize=plot_cfg.font.medium)
    ax2.set_yticklabels([b.capitalize() for b in valid_bands], fontsize=plot_cfg.font.medium)
    ax2.set_title("Significance (-log₁₀ p)", fontsize=plot_cfg.font.title, fontweight="bold")
    
    P_VALUE_THRESHOLD_DISPLAY = 0.001
    LOG_PVAL_TEXT_THRESHOLD = 1.5
    
    for i in range(n_bands):
        for j in range(n_bands):
            if i == j:
                continue
            
            p_value = pval_matrix[i, j]
            q_value = qval_matrix[i, j]
            log_pval = log_pvals[i, j]
            
            if p_value < P_VALUE_THRESHOLD_DISPLAY:
                p_text = "<.001"
            else:
                p_text = f"{p_value:.3f}"
            
            is_fdr_significant = q_value < FDR_ALPHA_DEFAULT
            fdr_marker = "†" if is_fdr_significant else ""
            is_high_log_pval = log_pval > LOG_PVAL_TEXT_THRESHOLD
            text_color = "white" if is_high_log_pval else "black"
            
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
    footer = (f"n={len(pow_df)} trials | {n_tests} tests | FDR-BH α={FDR_ALPHA_DEFAULT} | "
              f"{n_significant} significant | *p<{FDR_ALPHA_DEFAULT} uncorrected, †q<{FDR_ALPHA_DEFAULT} FDR")
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, save_dir / f"sub-{subject}_cross_frequency_power_correlation",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved cross-frequency power correlation ({n_significant}/{n_tests} FDR significant)")


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
        if len(vals) >= MIN_TRIALS_FOR_VARIABILITY:
            variances[col] = np.var(vals)
    
    if not variances:
        return
    
    sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_features) * 0.25)))
    
    MAX_FEATURE_NAME_LENGTH = 30
    feature_names = [
        f[0][:MAX_FEATURE_NAME_LENGTH] + "..." if len(f[0]) > MAX_FEATURE_NAME_LENGTH else f[0] 
        for f in sorted_features
    ]
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
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
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
    segment: str,
) -> None:
    """Band power topomaps showing spatial distribution per frequency band.
    
    Creates MNE topomaps for each frequency band.
    
    Args:
        segment: Time window segment name (required, no fallback).
    """
    if pow_df is None or epochs_info is None:
        return
    
    if not segment or segment.strip() == "":
        if logger:
            logger.error("plot_band_power_topomaps requires segment parameter. No fallback will be used.")
        return
    
    plot_cfg = get_plot_config(config)
    
    valid_bands = []
    band_data = {}
    detected_stats = set()
    
    for band in bands:
        cols = []
        for c in pow_df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment:
                continue
            if parsed.get("band") != band:
                continue
            if parsed.get("scope") != "ch":
                continue
            cols.append(c)
            stat = parsed.get("stat", "")
            if stat:
                detected_stats.add(stat)
        
        if not cols:
            continue
        
        ch_power = {}
        for col in cols:
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid") and parsed.get("identifier"):
                ch_name = parsed["identifier"]
                ch_power[ch_name] = pow_df[col].mean()
        
        if ch_power:
            valid_bands.append(band)
            band_data[band] = ch_power
    
    if not valid_bands:
        if logger:
            power_cols = [c for c in pow_df.columns if c.startswith("power_")]
            sample_cols = power_cols[:5] if len(power_cols) > 0 else []
            available_segments = set()
            for c in power_cols:
                parsed = NamingSchema.parse(str(c))
                if parsed.get("valid") and parsed.get("group") == "power":
                    seg = parsed.get("segment")
                    if seg:
                        available_segments.add(seg)
            logger.warning(
                f"No valid band power columns found for segment '{segment}'. "
                f"Available segments: {sorted(available_segments)}. "
                f"Available power columns: {len(power_cols)}. "
                f"Sample columns: {sample_cols}"
            )
        return
    
    # Determine unit label based on detected stat types
    # Use LaTeX math mode for proper rendering of subscripts
    STAT_TO_LABEL = {
        "logratio": r"$\log_{10}$(ratio)",
        "mean": r"power ($\mu$V²)",
        "baselined": "power (baseline-corrected)",
        "log10raw": r"$\log_{10}$(power)",
    }
    
    # Use the most common stat, or default to logratio if multiple stats found
    primary_stat = None
    if detected_stats:
        # Prefer logratio > mean > baselined > log10raw
        priority_order = ["logratio", "mean", "baselined", "log10raw"]
        for stat in priority_order:
            if stat in detected_stats:
                primary_stat = stat
                break
        if primary_stat is None:
            primary_stat = sorted(detected_stats)[0]
    
    unit_label = STAT_TO_LABEL.get(primary_stat, "power")
    value_label = f"mean {unit_label}" if primary_stat else "mean power"
    
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
        
        if mask.sum() > MIN_CHANNELS_FOR_TOPO:
            valid_data = data_array[mask]
            valid_info = mne.pick_info(epochs_info, np.where(mask)[0])
            
            im, _ = plot_topomap(
                valid_data, valid_info,
                axes=ax, show=False, cmap="RdBu_r", contours=6
            )
            plt.colorbar(im, ax=ax, shrink=0.6, label=unit_label)
        
        band_color = get_band_color(band, config)
        ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)
    
    fig.suptitle(f"Band Power Topomaps - {segment.capitalize()} (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    footer = f"n={len(pow_df)} trials | Segment: {segment} | Values: {value_label}"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_fig(fig, save_dir / f"sub-{subject}_band_power_topomaps_{segment}",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved band power topomaps ({segment})")
