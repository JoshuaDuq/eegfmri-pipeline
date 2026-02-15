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
from eeg_pipeline.plotting.io.figures import get_viz_params
from eeg_pipeline.utils.data.columns import (
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
)
from eeg_pipeline.utils.config.loader import get_frequency_bands, require_config_value
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


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


def _get_comparison_segments(
    power_df: pd.DataFrame,
    config: Any,
    logger: Optional[logging.Logger],
) -> List[str]:
    """Extract comparison segments from config (no auto-detection)."""
    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least 2 window names "
            f"(got {segments!r})"
        )
    return [str(s) for s in segments]


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
        except Exception as exc:
            logger.warning(
                "Failed Mann-Whitney computation for band=%s (column index %s): %s",
                band,
                col_idx,
                exc,
            )
    
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
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError("Multi-group column comparison requested but could not resolve group masks.")
        
        masks_dict, group_labels = multi_group_info
        seg_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        if seg_name == "":
            raise ValueError("plotting.comparisons.comparison_segment must be a non-empty string")
        
        from .utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
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
                    multigroup_stats=multigroup_stats,
                )
        
        if logger:
            logger.info(f"Saved power multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comp_mask_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")
    
    m1, m2, label1, label2 = comp_mask_info
    
    seg_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
    if seg_name == "":
        raise ValueError("plotting.comparisons.comparison_segment must be a non-empty string")
    
    available_segments = get_named_segments(power_df, group="power")
    if seg_name not in available_segments:
        raise ValueError(
            f"Configured segment '{seg_name}' not found in data. Available segments: {available_segments}"
        )
    
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
    active_window = require_config_value(config, "time_frequency_analysis.active_window")
    if not isinstance(active_window, (list, tuple)) or len(active_window) < 2:
        raise ValueError(
            "time_frequency_analysis.active_window must be a list/tuple of length 2 "
            f"(got {active_window!r})"
        )
    return [float(active_window[0]), float(active_window[1])]


def _get_plotting_tfr_baseline_window(config: Any) -> tuple[float, float]:
    """Resolve baseline window for plotting."""
    baseline = require_config_value(config, "time_frequency_analysis.baseline_window")
    if not isinstance(baseline, (list, tuple)) or len(baseline) < 2:
        raise ValueError(
            "time_frequency_analysis.baseline_window must be a list/tuple of length 2 "
            f"(got {baseline!r})"
        )
    return float(baseline[0]), float(baseline[1])


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
# Power Spectral Density Plotting
###################################################################


def _plot_psd_by_conditions(
    tfr_epochs: Any,
    conditions: List[Tuple[str, np.ndarray]],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    roi_suffix: str = "",
    roi_name: Optional[str] = None,
) -> bool:
    """Plot PSD by condition with uncertainty visualization and frequency band annotations.
    
    Computes PSD per trial, then averages across channels and time windows.
    Shows mean ± SEM with shaded confidence intervals for scientific rigor.
    Includes frequency band annotations and optional statistical comparison.
    
    Args:
        tfr_epochs: EpochsTFR object
        conditions: List of (label, mask) tuples
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    
    Returns:
        True if plot was created
    
    Raises:
        ValueError: If insufficient conditions or no valid data
    """
    if len(conditions) < 1:
        raise ValueError(
            f"power_spectral_density requires at least 1 condition, got {len(conditions)}"
        )
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    
    condition_colors = plt.cm.Set2(np.linspace(0.2, 0.8, len(conditions)))
    
    freq_bands = get_frequency_bands(config)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    psd_data_by_condition = []
    
    for idx, (label, mask) in enumerate(conditions):
        n_trials_cond = int(mask.sum())
        if n_trials_cond < 1:
            continue
        
        tfr_cond = tfr_epochs[mask]
        if len(tfr_cond) == 0:
            continue
        
        tfr_cond_avg = tfr_cond.average()
        apply_baseline_and_crop(tfr_cond_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_cond_win = _crop_tfr_to_active(tfr_cond_avg, active_window, logger)
        
        if tfr_cond_win is None:
            continue
        
        psd_mean = tfr_cond_win.data.mean(axis=(0, 2))
        
        if len(psd_mean) != len(tfr_cond_win.freqs):
            logger.warning(f"Frequency dimension mismatch: {len(psd_mean)} vs {len(tfr_cond_win.freqs)}")
            continue
        
        freqs = tfr_cond_win.freqs
        
        if len(tfr_cond) >= MIN_EPOCHS_FOR_SEM:
            psd_per_trial = []
            for trial_idx in range(len(tfr_cond)):
                tfr_trial = tfr_cond[[trial_idx]]
                tfr_trial_avg = tfr_trial.average()
                apply_baseline_and_crop(tfr_trial_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
                tfr_trial_win = _crop_tfr_to_active(tfr_trial_avg, active_window, logger)
                if tfr_trial_win is not None:
                    psd_trial = tfr_trial_win.data.mean(axis=(0, 2))
                    if len(psd_trial) == len(freqs):
                        psd_per_trial.append(psd_trial)
            
            if len(psd_per_trial) >= MIN_EPOCHS_FOR_SEM:
                psd_per_trial = np.array(psd_per_trial)
                psd_sem = psd_per_trial.std(axis=0, ddof=1) / np.sqrt(len(psd_per_trial))
                ci_multiplier = 1.96
                psd_ci_lower = psd_mean - ci_multiplier * psd_sem
                psd_ci_upper = psd_mean + ci_multiplier * psd_sem
            else:
                psd_sem = np.zeros_like(psd_mean)
                psd_ci_lower = psd_mean
                psd_ci_upper = psd_mean
        else:
            psd_sem = np.zeros_like(psd_mean)
            psd_ci_lower = psd_mean
            psd_ci_upper = psd_mean
        
        psd_data_by_condition.append({
            'label': label,
            'freqs': freqs,
            'mean': psd_mean,
            'sem': psd_sem,
            'ci_lower': psd_ci_lower,
            'ci_upper': psd_ci_upper,
            'n_trials': n_trials_cond,
            'color': condition_colors[idx],
        })
    
    if not psd_data_by_condition:
        plt.close(fig)
        raise ValueError(
            "power_spectral_density plot failed: no conditions had valid trials. "
            "Check that conditions have sufficient data."
        )
    
    for psd_data in psd_data_by_condition:
        has_uncertainty = np.any(psd_data['sem'] > 0)
        
        if has_uncertainty:
            ax.fill_between(
                psd_data['freqs'],
                psd_data['ci_lower'],
                psd_data['ci_upper'],
                color=psd_data['color'],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
        
        ax.plot(
            psd_data['freqs'],
            psd_data['mean'],
            color=psd_data['color'],
            linewidth=2.0,
            label=f"{psd_data['label']} (n={psd_data['n_trials']})",
            zorder=3,
        )
    
    for band, (fmin, fmax) in features_freq_bands.items():
        if fmin < psd_data_by_condition[0]['freqs'].max():
            fmax_clipped = min(fmax, psd_data_by_condition[0]['freqs'].max())
            ax.axvspan(fmin, fmax_clipped, alpha=0.06, color="0.6", linewidth=0, zorder=0)
            mid = (fmin + fmax_clipped) / 2
            if mid < psd_data_by_condition[0]['freqs'].max():
                y_max = ax.get_ylim()[1]
                ax.text(
                    mid, y_max * 0.96, band[0].upper(),
                    fontsize=8, ha="center", va="top", color="0.5", zorder=2,
                    fontweight='medium'
                )
    
    ax.axhline(0, color="0.4", linewidth=1.0, alpha=0.5, linestyle='--', zorder=2)
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.set_ylabel(r"$\log_{10}$(power / baseline)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.legend(loc='best', fontsize=plot_cfg.font.medium, frameon=False, handlelength=1.5)
    
    if roi_name:
        roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        title = f"Power Spectral Density (sub-{subject}) | ROI: {roi_display}"
        fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.98)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=plot_cfg.font.small)
    
    footer_text = (
        f"Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}]s | "
        f"Window: [{active_window[0]:.1f}, {active_window[1]:.1f}]s | "
        f"n={len(tfr_epochs)} trials | 95% CI"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small - 1,
        color='0.5',
        alpha=0.7
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_condition{roi_suffix}'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    logger.info(f"Saved PSD by condition (Induced) with uncertainty visualization{roi_suffix}")
    return True


def _plot_psd_by_temperature(
    tfr_epochs: Any,
    temps: pd.Series,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> bool:
    """Plot PSD by temperature condition with uncertainty visualization and frequency band annotations.
    
    Computes PSD per trial, then averages across channels and time windows.
    Shows mean ± SEM with shaded confidence intervals for scientific rigor.
    Includes frequency band annotations.
    
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
    temp_colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(unique_temps)))
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    
    freq_bands = get_frequency_bands(config)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    psd_data_by_temp = []
    
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        n_trials_temp = int(temp_mask.sum())
        if n_trials_temp < 1:
            continue
        
        tfr_temp = tfr_epochs[temp_mask]
        if len(tfr_temp) == 0:
            continue
        
        tfr_temp_avg = tfr_temp.average()
        apply_baseline_and_crop(tfr_temp_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_temp_win = _crop_tfr_to_active(tfr_temp_avg, active_window, logger)
        
        if tfr_temp_win is None:
            continue
        
        psd_mean = tfr_temp_win.data.mean(axis=(0, 2))
        
        if len(psd_mean) != len(tfr_temp_win.freqs):
            logger.warning(f"Frequency dimension mismatch: {len(psd_mean)} vs {len(tfr_temp_win.freqs)}")
            continue
        
        freqs = tfr_temp_win.freqs
        
        if len(tfr_temp) >= MIN_EPOCHS_FOR_SEM:
            psd_per_trial = []
            for trial_idx in range(len(tfr_temp)):
                tfr_trial = tfr_temp[[trial_idx]]
                tfr_trial_avg = tfr_trial.average()
                apply_baseline_and_crop(tfr_trial_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
                tfr_trial_win = _crop_tfr_to_active(tfr_trial_avg, active_window, logger)
                if tfr_trial_win is not None:
                    psd_trial = tfr_trial_win.data.mean(axis=(0, 2))
                    if len(psd_trial) == len(freqs):
                        psd_per_trial.append(psd_trial)
            
            if len(psd_per_trial) >= MIN_EPOCHS_FOR_SEM:
                psd_per_trial = np.array(psd_per_trial)
                psd_sem = psd_per_trial.std(axis=0, ddof=1) / np.sqrt(len(psd_per_trial))
                ci_multiplier = 1.96
                psd_ci_lower = psd_mean - ci_multiplier * psd_sem
                psd_ci_upper = psd_mean + ci_multiplier * psd_sem
            else:
                psd_sem = np.zeros_like(psd_mean)
                psd_ci_lower = psd_mean
                psd_ci_upper = psd_mean
        else:
            psd_sem = np.zeros_like(psd_mean)
            psd_ci_lower = psd_mean
            psd_ci_upper = psd_mean
        
        psd_data_by_temp.append({
            'label': f'{temp:.0f}°C',
            'freqs': freqs,
            'mean': psd_mean,
            'sem': psd_sem,
            'ci_lower': psd_ci_lower,
            'ci_upper': psd_ci_upper,
            'n_trials': n_trials_temp,
            'color': temp_colors[idx],
        })
    
    if not psd_data_by_temp:
        plt.close(fig)
        return False
    
    for psd_data in psd_data_by_temp:
        has_uncertainty = np.any(psd_data['sem'] > 0)
        
        if has_uncertainty:
            ax.fill_between(
                psd_data['freqs'],
                psd_data['ci_lower'],
                psd_data['ci_upper'],
                color=psd_data['color'],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
        
        ax.plot(
            psd_data['freqs'],
            psd_data['mean'],
            color=psd_data['color'],
            linewidth=2.0,
            label=f"{psd_data['label']} (n={psd_data['n_trials']})",
            zorder=3,
        )
    
    for band, (fmin, fmax) in features_freq_bands.items():
        if fmin < psd_data_by_temp[0]['freqs'].max():
            fmax_clipped = min(fmax, psd_data_by_temp[0]['freqs'].max())
            ax.axvspan(fmin, fmax_clipped, alpha=0.06, color="0.6", linewidth=0, zorder=0)
            mid = (fmin + fmax_clipped) / 2
            if mid < psd_data_by_temp[0]['freqs'].max():
                y_max = ax.get_ylim()[1]
                ax.text(
                    mid, y_max * 0.96, band[0].upper(),
                    fontsize=8, ha="center", va="top", color="0.5", zorder=2,
                    fontweight='medium'
                )
    
    ax.axhline(0, color="0.4", linewidth=1.0, alpha=0.5, linestyle='--', zorder=2)
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.set_ylabel(r"$\log_{10}$(power / baseline)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.legend(loc='best', fontsize=plot_cfg.font.medium, frameon=False, handlelength=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=plot_cfg.font.small)
    
    footer_text = (
        f"Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}]s | "
        f"Window: [{active_window[0]:.1f}, {active_window[1]:.1f}]s | "
        f"n={len(tfr_epochs)} trials | 95% CI"
    )
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small - 1,
        color='0.5',
        alpha=0.7
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_temperature'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    logger.info("Saved PSD by temperature (Induced) with uncertainty visualization")
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
    ax.set_ylabel(r"$\log_{10}$(power/baseline)", fontsize=plot_cfg.font.medium)
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
    """Plot power spectral density by condition, one plot per ROI.
    
    Requires condition selection to be configured via plotting.comparisons.
    Creates one plot per ROI specified in the TUI.
    No fallbacks - errors will surface if conditions cannot be extracted.
    
    Args:
        tfr: EpochsTFR object
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        events_df: Events DataFrame (required)
        config: Configuration object (required)
    
    Raises:
        ValueError: If events_df or config is missing, or if conditions cannot be extracted
    """
    _validate_epochs_tfr(tfr, "plot_power_spectral_density", logger)
    
    if events_df is None or events_df.empty:
        raise ValueError(
            "plot_power_spectral_density requires events_df. "
            "No fallback will be used."
        )
    
    if config is None:
        raise ValueError(
            "plot_power_spectral_density requires config. "
            "No fallback will be used."
        )
    
    if len(tfr) != len(events_df):
        raise ValueError(
            f"TFR window ({len(tfr)} epochs) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
    from eeg_pipeline.utils.formatting import sanitize_label
    
    rois = get_roi_definitions(config)
    all_channels = tfr.ch_names
    roi_names = _get_comparison_rois(config, rois)
    
    if logger:
        logger.info(f"PSD plotting: ROIs={roi_names}")
    
    column = require_config_value(config, "plotting.comparisons.comparison_column")
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    
    if not isinstance(values_spec, (list, tuple)) or len(values_spec) < 1:
        raise ValueError(
            "power_spectral_density requires plotting.comparisons.comparison_values with at least 1 value. "
            "Configure via TUI plot-specific settings or CLI."
        )
    
    if len(values_spec) == 1:
        val = values_spec[0]
        if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 1:
            label = str(labels_spec[0]).strip()
        else:
            label = str(val)
        
        column_values = events_df[column]
        try:
            numeric_val = float(val)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_val).values
        except (ValueError, TypeError):
            val_str = str(val).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == val_str).values
        
        if int(mask.sum()) == 0:
            raise ValueError(
                f"power_spectral_density: no trials found for value {val!r} in column {column!r}"
            )
        
        conditions = [(label, mask)]
    elif len(values_spec) == 2:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
        if not comp_mask_info:
            raise ValueError(
                "power_spectral_density plot requested but could not resolve comparison masks. "
                "Configure plotting.comparisons.comparison_column and comparison_values."
            )
        mask1, mask2, label1, label2 = comp_mask_info
        conditions = [(label1, mask1), (label2, mask2)]
    else:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError(
                "power_spectral_density plot requested but could not resolve multi-group masks. "
                "Configure plotting.comparisons.comparison_column and comparison_values."
            )
        masks_dict, group_labels = multi_group_info
        conditions = [(label, masks_dict[label]) for label in group_labels]
    
    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)
        
        if not roi_channels:
            if logger:
                logger.warning(f"No channels found for ROI {roi_name}, skipping PSD plot")
            continue
        
        tfr_roi = tfr.copy().pick_channels(roi_channels)
        if len(tfr_roi.ch_names) == 0:
            if logger:
                logger.warning(f"No valid channels after filtering for ROI {roi_name}, skipping PSD plot")
            continue
        
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        roi_suffix = f"_roi-{roi_safe}" if roi_safe else ""
        
        _plot_psd_by_conditions(tfr_roi, conditions, subject, save_dir, logger, config, roi_suffix=roi_suffix, roi_name=roi_name)


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
    ax2.set_title(r"Significance ($-\log_{10}$ p)", fontsize=plot_cfg.font.title, fontweight="bold")
    
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
    cbar2.set_label(r"$-\log_{10}$(p)", fontsize=plot_cfg.font.label)
    
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


def plot_band_power_topomaps(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str,
    events_df: Optional[pd.DataFrame] = None,
    *,
    sample_unit: str = "trials",
    label_suffix: Optional[str] = None,
) -> None:
    """Band power topomaps showing spatial distribution per frequency band.
    
    Creates MNE topomaps for each frequency band for a single segment.
    Supports condition-based filtering and optional column-based contrasts.
    
    Args:
        segment: Time window segment name (required, no fallback).
        events_df: Optional events DataFrame for condition-based filtering.
        label_suffix: Optional label to append to the title/filename when not using
            condition-based filtering (e.g., group mean for a specific condition).
    """
    if pow_df is None or epochs_info is None:
        return
    
    if not segment or segment.strip() == "":
        if logger:
            logger.error("plot_band_power_topomaps requires segment parameter. No fallback will be used.")
        return
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.formatting import sanitize_label

    all_channels = epochs_info.ch_names

    compare_columns = bool(get_config_value(config, "plotting.comparisons.compare_columns", True))
    comparison_column = str(get_config_value(config, "plotting.comparisons.comparison_column", "") or "").strip()
    comparison_values = get_config_value(config, "plotting.comparisons.comparison_values", [])

    has_column_spec = (
        comparison_column != ""
        and isinstance(comparison_values, (list, tuple))
        and len(comparison_values) >= 2
    )

    if has_column_spec and len(comparison_values) != 2:
        raise ValueError(
            f"band_power_topomaps compare_columns requires exactly 2 comparison_values, "
            f"got {len(comparison_values)}: {comparison_values}"
        )

    conditions: Optional[List[Tuple[str, np.ndarray]]] = None
    if events_df is not None and has_column_spec:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=False)
        if not comp_mask_info:
            raise ValueError(
                "band_power_topomaps column comparison requested but could not resolve "
                f"comparison masks for column={comparison_column!r}, values={comparison_values!r}"
            )
        mask1, mask2, label1, label2 = comp_mask_info
        conditions = [(label1, mask1), (label2, mask2)]
    
    _plot_band_power_topomaps_single_segment(
        pow_df,
        epochs_info,
        bands,
        subject,
        save_dir,
        logger,
        config,
        segment,
        conditions,
        compare_columns,
        events_df,
        sample_unit,
        label_suffix,
    )
    
    
def _plot_band_power_topomaps_single_segment(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str,
    conditions: Optional[List[Tuple[str, np.ndarray]]],
    compare_columns: bool,
    events_df: Optional[pd.DataFrame],
    sample_unit: str,
    label_suffix: Optional[str] = None,
) -> None:
    """Internal helper to plot topomaps."""
    plot_cfg = get_plot_config(config)
    
    def extract_band_data(power_subset: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict[str, float]], set]:
        """Extract band power data from a power DataFrame subset."""
        valid_bands = []
        band_data = {}
        detected_stats = set()
        
        for band in bands:
            cols = []
            for c in power_subset.columns:
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
                    ch_power[ch_name] = power_subset[col].mean()
            
            if ch_power:
                valid_bands.append(band)
                band_data[band] = ch_power
        
        return valid_bands, band_data, detected_stats
    
    if conditions:
        valid_bands, band_data, detected_stats = extract_band_data(pow_df)
        if not valid_bands:
            if logger:
                logger.warning("No valid band power columns found for condition-based topomaps")
            return
    else:
        valid_bands, band_data, detected_stats = extract_band_data(pow_df)
    
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
    
    from eeg_pipeline.utils.formatting import sanitize_label
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha
    from scipy.stats import mannwhitneyu
    
    def save_topomap_plot(
        condition_pow_df: pd.DataFrame,
        condition_label: Optional[str],
        condition_color: Optional[str],
        n_samples: int,
        unit: str,
        value: str,
    ) -> None:
        """Create and save a topomap plot for a condition."""
        cond_valid_bands, cond_band_data, _ = extract_band_data(condition_pow_df)
        if not cond_valid_bands:
            return
        
        fig, axes = plt.subplots(1, len(cond_valid_bands), figsize=(width_per_band * len(cond_valid_bands), 4))
        if len(cond_valid_bands) == 1:
            axes = [axes]
        
        for i, band in enumerate(cond_valid_bands):
            ax = axes[i]
            ch_power = cond_band_data[band]
            
            data_array = np.full(len(epochs_info.ch_names), np.nan)
            mask = np.zeros(len(epochs_info.ch_names), dtype=bool)
            
            for ch_idx, ch_name in enumerate(epochs_info.ch_names):
                if ch_name in ch_power:
                    val = ch_power[ch_name]
                    try:
                        is_finite = np.isfinite(val)
                    except TypeError:
                        is_finite = False
                    if is_finite:
                        data_array[ch_idx] = float(val)
                        mask[ch_idx] = True
            
            if mask.sum() > MIN_CHANNELS_FOR_TOPO:
                valid_data = data_array[mask]
                valid_info = mne.pick_info(epochs_info, np.where(mask)[0])
                
                im, _ = plot_topomap(
                    valid_data, valid_info,
                    axes=ax, show=False, cmap="RdBu_r", contours=6
                )
                plt.colorbar(im, ax=ax, shrink=0.6, label=unit)
            
            band_color = get_band_color(band, config)
            ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)
        
        title_text = f"Band Power Topomaps - {segment.capitalize()} (sub-{subject})"
        effective_label = condition_label or (str(label_suffix).strip() if label_suffix is not None else "")
        if effective_label:
            title_text += f" | {effective_label}"
        fig.suptitle(title_text, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
        
        unit_label = str(sample_unit).strip() or "trials"
        footer = f"n={n_samples} {unit_label} | Segment: {segment} | Values: {value}"
        fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        if effective_label:
            condition_safe = sanitize_label(effective_label).lower().replace(" ", "_")
            filename = f"sub-{subject}_band_power_topomaps_{segment}_{condition_safe}"
        else:
            filename = f"sub-{subject}_band_power_topomaps_{segment}"
        
        save_fig(fig, save_dir / filename,
                 formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                 bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
        plt.close(fig)

    def _band_channel_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Map band -> channel -> column for this segment."""
        mapping: Dict[str, Dict[str, str]] = {str(b): {} for b in bands}
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment:
                continue
            if parsed.get("scope") != "ch":
                continue
            band = str(parsed.get("band") or "")
            ch = str(parsed.get("identifier") or "")
            if band in mapping and ch:
                mapping[band][ch] = str(c)
        return mapping

    def save_column_comparison_topomaps() -> None:
        """If compare_columns, save statistical contrast topomaps with significance masks."""
        if not conditions:
            return
        if events_df is None:
            raise ValueError("band_power_topomaps compare_columns requires events_df.")

        band_to_ch_col = _band_channel_columns(pow_df)
        alpha = float(get_fdr_alpha(config))

        # Collect p-values across all band×channel tests for global FDR within this plot.
        tests: List[Tuple[str, str, float]] = []  # (band, ch, p)
        effect_map: Dict[Tuple[str, str], float] = {}

        for band in [str(b) for b in bands]:
            ch_to_col = band_to_ch_col.get(band, {})
            for ch_name in epochs_info.ch_names:
                col = ch_to_col.get(ch_name)
                if not col:
                    continue

                group_arrays: List[np.ndarray] = []
                group_means: List[float] = []
                for label, mask in conditions:
                    n_samples = min(len(pow_df), len(mask))
                    mask_array = np.asarray(mask[:n_samples], dtype=bool)
                    series = pd.to_numeric(pow_df.iloc[:n_samples][col], errors="coerce")
                    vals = series[mask_array].dropna().values
                    group_arrays.append(vals)
                    group_means.append(float(np.nanmean(vals)) if len(vals) else np.nan)

                if len(group_arrays) != 2:
                    continue

                v1, v2 = group_arrays[0], group_arrays[1]
                if len(v1) < MIN_TRIALS_FOR_STATISTICS or len(v2) < MIN_TRIALS_FOR_STATISTICS:
                    p = 1.0
                else:
                    p = float(mannwhitneyu(v1, v2, alternative="two-sided").pvalue)
                tests.append((band, ch_name, p))
                effect_map[(band, ch_name)] = (group_means[1] - group_means[0])

        if not tests:
            return

        rejected, qvals, _ = apply_fdr_correction([p for _, _, p in tests], config=config)
        q_map: Dict[Tuple[str, str], float] = {}
        sig_map: Dict[Tuple[str, str], bool] = {}
        for (band, ch, _), q, rej in zip(tests, qvals, rejected):
            q_map[(band, ch)] = float(q)
            sig_map[(band, ch)] = bool(rej) and float(q) < alpha

        n_bands = len([str(b) for b in bands])
        fig, axes = plt.subplots(1, n_bands, figsize=(width_per_band * n_bands, 4))
        if n_bands == 1:
            axes = [axes]

        total_sig = 0
        total_tests = 0

        for i, band in enumerate([str(b) for b in bands]):
            ax = axes[i]
            data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
            sig_mask_full = np.zeros(len(epochs_info.ch_names), dtype=bool)

            for ch_idx, ch_name in enumerate(epochs_info.ch_names):
                key = (band, ch_name)
                if key not in effect_map:
                    continue
                data_array[ch_idx] = float(effect_map[key])
                if sig_map.get(key, False):
                    sig_mask_full[ch_idx] = True

            present = np.isfinite(data_array)
            if present.sum() <= MIN_CHANNELS_FOR_TOPO:
                ax.set_axis_off()
                continue

            valid_data = data_array[present]
            valid_info = mne.pick_info(epochs_info, np.where(present)[0])
            valid_sig = sig_mask_full[present]

            vmax = float(np.nanmax(np.abs(valid_data))) if np.isfinite(valid_data).any() else 1.0
            vmin = -vmax

            im, _ = plot_topomap(
                valid_data,
                valid_info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                contours=6,
                vlim=(vmin, vmax),
                mask=valid_sig,
                mask_params=dict(markersize=2, markerfacecolor="none", markeredgecolor="k"),
            )
            plt.colorbar(im, ax=ax, shrink=0.6, label=unit_label)

            band_color = get_band_color(band, config)
            ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

            # Count stats for footer
            for ch_name in epochs_info.ch_names:
                key = (band, ch_name)
                if key in q_map:
                    total_tests += 1
                    if sig_map.get(key, False):
                        total_sig += 1

        label1 = str(conditions[0][0])
        label2 = str(conditions[1][0])
        title = f"Band Power Contrast - {segment.capitalize()} (sub-{subject}) | {label2} − {label1}"
        fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
        fig.text(
            0.5,
            0.01,
            f"n={len(pow_df)} trials | FDR α={alpha:.3f} | sig={total_sig}/{max(total_tests,1)}",
            ha="center",
            va="bottom",
            fontsize=plot_cfg.font.small,
            color="gray",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        label1_safe = sanitize_label(str(conditions[0][0])).lower().replace(" ", "_")
        label2_safe = sanitize_label(str(conditions[1][0])).lower().replace(" ", "_")
        out = save_dir / f"sub-{subject}_band_power_topomaps_{segment}_contrast_{label2_safe}_minus_{label1_safe}"
        save_fig(
            fig,
            out,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            config=config,
        )
        plt.close(fig)
    
    if conditions:
        group_colors = plt.cm.Set2(np.linspace(0, 1, max(len(conditions), 3)))
        condition_colors = {label: group_colors[i] for i, (label, _) in enumerate(conditions)}
        
        for condition_label, condition_mask in conditions:
            n_samples = min(len(pow_df), len(condition_mask))
            if n_samples <= 0:
                continue
            
            condition_mask_array = np.asarray(condition_mask[:n_samples], dtype=bool)
            if int(condition_mask_array.sum()) == 0:
                continue
            
            condition_pow_df = pow_df.loc[condition_mask_array]
            n_trials = int(condition_mask_array.sum())
            condition_color = condition_colors.get(condition_label, plot_cfg.get_color("blue"))
            
            save_topomap_plot(condition_pow_df, condition_label, condition_color, n_trials, unit_label, value_label)
        
        if logger:
            logger.info(f"Saved band power topomaps ({segment}) for {len(conditions)} conditions")
        if compare_columns:
            save_column_comparison_topomaps()
    else:
        save_topomap_plot(pow_df, None, None, len(pow_df), unit_label, value_label)
        if logger:
            label_for_log = str(label_suffix).strip() if label_suffix is not None else ""
            if label_for_log:
                logger.info(f"Saved band power topomaps ({segment}) | {label_for_log}")
            else:
                logger.info(f"Saved band power topomaps ({segment})")


def plot_band_power_topomaps_window_contrast(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    *,
    window1: str,
    window2: str,
) -> None:
    """Paired window contrast topomaps (window2 − window1) with FDR significance marks."""
    if pow_df is None or pow_df.empty or epochs_info is None:
        return

    if not window1 or not window2:
        raise ValueError("plot_band_power_topomaps_window_contrast requires window1 and window2.")

    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha
    from scipy.stats import wilcoxon

    plot_cfg = get_plot_config(config)
    alpha = float(get_fdr_alpha(config))

    def _band_channel_columns(df: pd.DataFrame, segment_name: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {str(b): {} for b in bands}
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment_name:
                continue
            if parsed.get("scope") != "ch":
                continue
            band = str(parsed.get("band") or "")
            ch = str(parsed.get("identifier") or "")
            if band in mapping and ch:
                mapping[band][ch] = str(c)
        return mapping

    cols_w1 = _band_channel_columns(pow_df, window1)
    cols_w2 = _band_channel_columns(pow_df, window2)

    tests: List[Tuple[str, str, float]] = []  # (band, ch, p)
    diff_map: Dict[Tuple[str, str], float] = {}

    for band in [str(b) for b in bands]:
        ch_to_col1 = cols_w1.get(band, {})
        ch_to_col2 = cols_w2.get(band, {})
        for ch_name in epochs_info.ch_names:
            col1 = ch_to_col1.get(ch_name)
            col2 = ch_to_col2.get(ch_name)
            if not col1 or not col2:
                continue
            s1 = pd.to_numeric(pow_df[col1], errors="coerce")
            s2 = pd.to_numeric(pow_df[col2], errors="coerce")
            valid = s1.notna() & s2.notna()
            if int(valid.sum()) < MIN_TRIALS_FOR_STATISTICS:
                p = 1.0
                mean_diff = float(np.nanmean((s2 - s1).values))
            else:
                diffs = (s2[valid] - s1[valid]).values
                mean_diff = float(np.nanmean(diffs)) if diffs.size else np.nan
                if diffs.size == 0 or not np.isfinite(diffs).any() or np.allclose(diffs, 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
            tests.append((band, ch_name, p))
            diff_map[(band, ch_name)] = mean_diff

    if not tests:
        return

    rejected, qvals, _ = apply_fdr_correction([p for _, _, p in tests], config=config)
    q_map: Dict[Tuple[str, str], float] = {}
    sig_map: Dict[Tuple[str, str], bool] = {}
    for (band, ch, _), q, rej in zip(tests, qvals, rejected):
        q_map[(band, ch)] = float(q)
        sig_map[(band, ch)] = bool(rej) and float(q) < alpha

    width_per_band = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_band", 3.5))
    fig, axes = plt.subplots(1, len(bands), figsize=(width_per_band * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]

    total_sig = 0
    total_tests = 0

    for i, band in enumerate([str(b) for b in bands]):
        ax = axes[i]
        data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
        sig_mask_full = np.zeros(len(epochs_info.ch_names), dtype=bool)

        for ch_idx, ch_name in enumerate(epochs_info.ch_names):
            key = (band, ch_name)
            if key not in diff_map:
                continue
            data_array[ch_idx] = float(diff_map[key])
            if sig_map.get(key, False):
                sig_mask_full[ch_idx] = True
            if key in q_map:
                total_tests += 1
                if sig_map.get(key, False):
                    total_sig += 1

        present = np.isfinite(data_array)
        if present.sum() <= MIN_CHANNELS_FOR_TOPO:
            ax.set_axis_off()
            continue

        valid_data = data_array[present]
        valid_info = mne.pick_info(epochs_info, np.where(present)[0])
        valid_sig = sig_mask_full[present]

        vmax = float(np.nanmax(np.abs(valid_data))) if np.isfinite(valid_data).any() else 1.0
        vmin = -vmax

        im, _ = plot_topomap(
            valid_data,
            valid_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=6,
            vlim=(vmin, vmax),
            mask=valid_sig,
            mask_params=dict(markersize=2, markerfacecolor="none", markeredgecolor="k"),
        )
        plt.colorbar(im, ax=ax, shrink=0.6, label="Δ power")

        band_color = get_band_color(band, config)
        ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

    fig.suptitle(
        f"Band Power Window Contrast (sub-{subject}) | {window2} − {window1}",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        f"n={len(pow_df)} trials | FDR α={alpha:.3f} | sig={total_sig}/{max(total_tests,1)}",
        ha="center",
        va="bottom",
        fontsize=plot_cfg.font.small,
        color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_band_power_topomaps_contrast_{window2}_minus_{window1}",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )
    plt.close(fig)


def plot_band_power_topomaps_group_condition_contrast(
    *,
    pow_df_condition1: pd.DataFrame,
    pow_df_condition2: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str,
    label1: str,
    label2: str,
    sample_unit: str = "subjects",
) -> None:
    """Paired condition contrast topomaps (label2 − label1) across samples (typically subjects).

    This is intended for group-level plotting, where each row in pow_df_condition*
    represents a subject-level summary (e.g., mean across trials within a condition).

    Statistical testing:
    - Paired Wilcoxon signed-rank test per band×channel.
    - Global FDR correction across all band×channel tests within this (segment) contrast.
    - If fewer than MIN_TRIALS_FOR_STATISTICS paired samples exist for a given test,
      p-values default to 1.0 and no significance markers will be shown.
    """
    if pow_df_condition1 is None or pow_df_condition2 is None:
        return
    if pow_df_condition1.empty or pow_df_condition2.empty or epochs_info is None:
        return

    if not segment or segment.strip() == "":
        if logger:
            logger.error(
                "plot_band_power_topomaps_group_condition_contrast requires segment parameter. No fallback will be used."
            )
        return

    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha
    from eeg_pipeline.utils.formatting import sanitize_label

    plot_cfg = get_plot_config(config)
    viz_params = get_viz_params(config)
    alpha = float(get_fdr_alpha(config))

    def _band_channel_columns(df: pd.DataFrame, segment_name: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {str(b): {} for b in bands}
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment_name:
                continue
            if parsed.get("scope") != "ch":
                continue
            band = str(parsed.get("band") or "")
            ch = str(parsed.get("identifier") or "")
            if band in mapping and ch:
                mapping[band][ch] = str(c)
        return mapping

    cols1 = _band_channel_columns(pow_df_condition1, segment)
    cols2 = _band_channel_columns(pow_df_condition2, segment)

    tests: List[Tuple[str, str, float]] = []  # (band, ch, p)
    diff_map: Dict[Tuple[str, str], float] = {}

    for band in [str(b) for b in bands]:
        ch_to_col1 = cols1.get(band, {})
        ch_to_col2 = cols2.get(band, {})
        for ch_name in epochs_info.ch_names:
            col1 = ch_to_col1.get(ch_name)
            col2 = ch_to_col2.get(ch_name)
            if not col1 or not col2:
                continue

            s1 = pd.to_numeric(pow_df_condition1[col1], errors="coerce")
            s2 = pd.to_numeric(pow_df_condition2[col2], errors="coerce")
            valid = s1.notna() & s2.notna()
            diffs = (s2[valid] - s1[valid]).values

            mean_diff = float(np.nanmean(diffs)) if diffs.size else np.nan
            diff_map[(band, ch_name)] = mean_diff

            if int(valid.sum()) < MIN_TRIALS_FOR_STATISTICS:
                p = 1.0
            else:
                # Wilcoxon cannot run on all-zero diffs.
                if diffs.size == 0 or not np.isfinite(diffs).any() or np.allclose(diffs, 0):
                    p = 1.0
                else:
                    try:
                        p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
                    except Exception:
                        p = 1.0
            tests.append((band, ch_name, p))

    if not tests:
        return

    rejected, qvals, _ = apply_fdr_correction([p for _, _, p in tests], config=config)
    q_map: Dict[Tuple[str, str], float] = {}
    sig_map: Dict[Tuple[str, str], bool] = {}
    for (band, ch, _), q, rej in zip(tests, qvals, rejected):
        q_map[(band, ch)] = float(q)
        sig_map[(band, ch)] = bool(rej) and float(q) < alpha

    width_per_band = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_band", 3.5))
    fig, axes = plt.subplots(1, len(bands), figsize=(width_per_band * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]

    total_sig = 0
    total_tests = 0

    for i, band in enumerate([str(b) for b in bands]):
        ax = axes[i]
        data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
        sig_mask_full = np.zeros(len(epochs_info.ch_names), dtype=bool)

        for ch_idx, ch_name in enumerate(epochs_info.ch_names):
            key = (band, ch_name)
            if key not in diff_map:
                continue
            data_array[ch_idx] = float(diff_map[key])
            if sig_map.get(key, False):
                sig_mask_full[ch_idx] = True

        present = np.isfinite(data_array)
        if present.sum() <= MIN_CHANNELS_FOR_TOPO:
            ax.set_axis_off()
            continue

        valid_data = data_array[present]
        valid_info = mne.pick_info(epochs_info, np.where(present)[0])
        valid_sig = sig_mask_full[present]

        vmax = float(np.nanmax(np.abs(valid_data))) if np.isfinite(valid_data).any() else 1.0
        vmin = -vmax

        im, _ = plot_topomap(
            valid_data,
            valid_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=6,
            vlim=(vmin, vmax),
            mask=valid_sig,
            mask_params=viz_params.get("sig_mask_params"),
        )
        plt.colorbar(im, ax=ax, shrink=0.6, label="Δ power")

        band_color = get_band_color(band, config)
        ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

        for ch_name in epochs_info.ch_names:
            key = (band, ch_name)
            if key in q_map:
                total_tests += 1
                if sig_map.get(key, False):
                    total_sig += 1

    fig.suptitle(
        f"Band Power Contrast - {segment.capitalize()} (sub-{subject}) | {label2} − {label1}",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        f"n={len(pow_df_condition1)} {sample_unit} | FDR α={alpha:.3f} | sig={total_sig}/{max(total_tests,1)}",
        ha="center",
        va="bottom",
        fontsize=plot_cfg.font.small,
        color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    label1_safe = sanitize_label(str(label1)).lower().replace(" ", "_")
    label2_safe = sanitize_label(str(label2)).lower().replace(" ", "_")
    out = save_dir / f"sub-{subject}_band_power_topomaps_{segment}_contrast_{label2_safe}_minus_{label1_safe}"
    save_fig(
        fig,
        out,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )
    plt.close(fig)
