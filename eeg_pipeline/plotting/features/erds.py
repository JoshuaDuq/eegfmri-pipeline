"""
ERDS Visualization
==================

Clean, publication-quality visualizations for ERD/ERS features.
Uses violin/strip plots for distributions and summary comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import get_band_names


###################################################################
# Constants
###################################################################

_VIOLIN_WIDTH = 0.7
_VIOLIN_ALPHA = 0.6
_SCATTER_ALPHA = 0.3
_SCATTER_SIZE = 8
_JITTER_RANGE = 0.1
_SECONDS_TO_MILLISECONDS = 1000.0
_ZERO_LINE_COLOR = "black"
_ZERO_LINE_WIDTH = 1
_NO_DATA_MESSAGE = "No data"
_UNKNOWN_SEGMENT_LABEL = "unknown"


###################################################################
# Helper Functions
###################################################################

def _get_erds_segments(features_df: pd.DataFrame) -> List[str]:
    """Extract unique ERDS segment names from feature columns."""
    segments = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "erds":
            continue
        segment = str(parsed.get("segment") or "")
        if segment:
            segments.add(segment)
    return sorted(segments)


def _select_erds_segment(
    features_df: pd.DataFrame, preferred: str = "active"
) -> Optional[str]:
    """Select segment from available ERDS segments, preferring specified one."""
    segments = _get_erds_segments(features_df)
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _matches_erds_criteria(
    parsed: Dict[str, Any],
    segment: str,
    band: str,
    stat: str,
    scope: str,
) -> bool:
    """Check if parsed column name matches ERDS selection criteria."""
    if not parsed.get("valid"):
        return False
    if parsed.get("group") != "erds":
        return False
    if str(parsed.get("segment") or "") != str(segment):
        return False
    if str(parsed.get("band") or "") != str(band):
        return False
    if scope and str(parsed.get("scope") or "") != str(scope):
        return False
    if str(parsed.get("stat") or "") != str(stat):
        return False
    return True


def _collect_erds_values(
    features_df: pd.DataFrame,
    *,
    band: str,
    segment: str,
    stat: str,
    scope: str = "global",
) -> np.ndarray:
    """Collect ERDS values matching specified criteria."""
    matching_columns = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if _matches_erds_criteria(parsed, segment, band, stat, scope):
            matching_columns.append(str(col))

    if not matching_columns:
        return np.array([])

    if len(matching_columns) == 1:
        series = pd.to_numeric(
            features_df[matching_columns[0]], errors="coerce"
        )
    else:
        series = (
            features_df[matching_columns]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
        )
    values = series.dropna().values
    return values[np.isfinite(values)]


def _collect_band_data(
    features_df: pd.DataFrame,
    bands: List[str],
    band_colors: Dict[str, str],
    segment: Optional[str],
    stat: str,
    scope: str,
    scale_factor: float = 1.0,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Collect ERDS values for each band, returning data, positions, and colors."""
    data_list = []
    positions = []
    colors = []

    if segment is not None:
        for position, band in enumerate(bands):
            values = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat=stat,
                scope=scope,
            )
            if values.size > 0:
                scaled_values = values * scale_factor
                data_list.append(scaled_values)
                positions.append(position)
                colors.append(band_colors[band])

    return data_list, positions, colors


def _create_violin_plot(
    ax: plt.Axes,
    data_list: List[np.ndarray],
    positions: List[int],
    colors: List[str],
    band_labels: List[str],
) -> None:
    """Create violin plot with specified data, positions, and colors."""
    if not data_list:
        ax.text(
            0.5,
            0.5,
            _NO_DATA_MESSAGE,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        return

    violin_parts = ax.violinplot(
        data_list,
        positions=positions,
        showmedians=True,
        widths=_VIOLIN_WIDTH,
    )
    for index, body in enumerate(violin_parts.get("bodies", [])):
        body.set_facecolor(colors[index])
        body.set_alpha(_VIOLIN_ALPHA)

    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels([band.capitalize() for band in band_labels])


def _add_scatter_overlay(
    ax: plt.Axes,
    positions: List[int],
    data_list: List[np.ndarray],
    colors: List[str],
    seed: Optional[int] = None,
) -> None:
    """Add jittered scatter points over violin plots for better data visibility."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    for position, values, color in zip(positions, data_list, colors):
        jitter = rng.uniform(-_JITTER_RANGE, _JITTER_RANGE, len(values))
        ax.scatter(
            position + jitter,
            values,
            c=color,
            alpha=_SCATTER_ALPHA,
            s=_SCATTER_SIZE,
        )


def _format_axis_style(ax: plt.Axes) -> None:
    """Apply standard axis styling: remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _create_figure_title(
    fig: plt.Figure,
    title: str,
    segment: Optional[str],
    plot_cfg: Any,
) -> None:
    """Add suptitle to figure with segment information."""
    segment_label = segment if segment is not None else _UNKNOWN_SEGMENT_LABEL
    fig.suptitle(
        f"{title} ({segment_label})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )


def _save_and_close_figure(fig: plt.Figure, save_path: Path, config: Any = None) -> None:
    """Save figure and close to free memory."""
    plt.tight_layout()
    save_fig(fig, save_path, config=config)
    plt.close(fig)


###################################################################
# ERDS Distribution Plots
###################################################################


###################################################################
# ERDS Condition Comparison
###################################################################

def _extract_channel_identifiers(features_df: pd.DataFrame) -> List[str]:
    """Extract all channel identifiers from ERDS channel-scoped columns."""
    channel_identifiers = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if (
            parsed.get("valid")
            and parsed.get("group") == "erds"
            and parsed.get("scope") == "ch"
        ):
            channel_id = parsed.get("identifier")
            if channel_id:
                channel_identifiers.add(str(channel_id))
    return list(channel_identifiers)


def _determine_roi_names(
    config: Any, rois: Dict[str, Any]
) -> List[str]:
    """Determine which ROIs to plot based on config."""
    from eeg_pipeline.utils.config.loader import get_config_value

    comp_rois = get_config_value(
        config, "plotting.comparisons.comparison_rois", []
    )
    if comp_rois:
        roi_names = []
        for roi in comp_rois:
            if roi.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif roi in rois:
                roi_names.append(roi)
        return roi_names if roi_names else ["all"]

    roi_names = ["all"]
    if rois:
        roi_names.extend(list(rois.keys()))
    return roi_names


def _get_erds_columns_for_roi(
    features_df: pd.DataFrame,
    segment: str,
    band: str,
    roi_name: str,
    all_channels: List[str],
    rois: Dict[str, Any],
) -> List[str]:
    """Get ERDS columns filtered by segment, band, and ROI."""
    from eeg_pipeline.plotting.features.roi import get_roi_channels

    if roi_name == "all":
        roi_channels = all_channels
    else:
        roi_channels = get_roi_channels(
            rois.get(roi_name, []), all_channels
        )
    roi_channel_set = set(roi_channels) if roi_channels else set()

    matching_columns = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "erds":
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != band:
            continue

        scope = parsed.get("scope") or ""
        stat = str(parsed.get("stat") or "")
        if stat not in ("percent_mean", "db_mean", "percent", "db"):
            continue

        if scope in ("global", "roi"):
            matching_columns.append(col)
        elif scope == "ch":
            channel_id = str(parsed.get("identifier") or "")
            if channel_id in roi_channel_set:
                matching_columns.append(col)

    return matching_columns


def _create_window_comparison_plots(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    roi_names: List[str],
    all_channels: List[str],
    rois: Dict[str, Any],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Create paired window comparison plots for ERDS.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.plotting.features.utils import plot_paired_comparison, plot_multi_window_comparison
    from eeg_pipeline.plotting.io.figures import log_if_present
    from eeg_pipeline.utils.formatting import sanitize_label

    if len(segments) < 2:
        return

    use_multi_window = len(segments) > 2

    for roi_name in roi_names:
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        suffix = f"_roi-{roi_safe}" if roi_safe else ""
        
        if use_multi_window:
            data_by_band_multi: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                segment_series = {}
                for seg in segments:
                    cols = _get_erds_columns_for_roi(
                        features_df, seg, band, roi_name, all_channels, rois
                    )
                    if cols:
                        segment_series[seg] = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                if len(segment_series) < 2:
                    continue
                
                valid_mask = pd.Series(True, index=features_df.index)
                for series in segment_series.values():
                    valid_mask &= series.notna()
                
                segment_values = {}
                for seg, series in segment_series.items():
                    vals = series[valid_mask].values
                    if len(vals) > 0:
                        segment_values[seg] = vals
                
                if len(segment_values) >= 2:
                    data_by_band_multi[band] = segment_values
            
            if data_by_band_multi:
                save_path = save_dir / f"sub-{subject}_erds_by_condition{suffix}_multiwindow"
                plot_multi_window_comparison(
                    data_by_band=data_by_band_multi,
                    subject=subject,
                    save_path=save_path,
                    feature_label="ERDS",
                    segments=segments,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        else:
            segment1, segment2 = segments[0], segments[1]
            data_by_band = {}
            for band in bands:
                cols1 = _get_erds_columns_for_roi(
                    features_df, segment1, band, roi_name, all_channels, rois
                )
                cols2 = _get_erds_columns_for_roi(
                    features_df, segment2, band, roi_name, all_channels, rois
                )

                if not cols1 or not cols2:
                    continue

                series1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                series2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)

                valid_mask = series1.notna() & series2.notna()
                values1 = series1[valid_mask].values
                values2 = series2[valid_mask].values

                if len(values1) > 0:
                    data_by_band[band] = (values1, values2)

            if data_by_band:
                save_path = save_dir / f"sub-{subject}_erds_by_condition{suffix}_window"
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="ERDS",
                    config=config,
                    logger=logger,
                    label1=segment1.capitalize(),
                    label2=segment2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )

    plot_type = "multi-window" if use_multi_window else "paired"
    log_if_present(
        logger,
        "info",
        f"Saved ERDS {plot_type} comparison plots for {len(roi_names)} ROIs",
    )


def _create_column_comparison_plots(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    bands: List[str],
    roi_names: List[str],
    all_channels: List[str],
    rois: Dict[str, Any],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Create unpaired column comparison plots for ERDS.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.plotting.features.utils import (
        compute_or_load_column_stats,
        get_band_color,
        plot_multi_group_column_comparison,
    )
    from eeg_pipeline.plotting.io.figures import log_if_present
    from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
    from eeg_pipeline.utils.formatting import sanitize_label

    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError("Multi-group column comparison requested but could not resolve group masks.")
        
        masks_dict, group_labels = multi_group_info
        segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        
        from eeg_pipeline.plotting.features.utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
        for roi_name in roi_names:
            data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                cols = _get_erds_columns_for_roi(features_df, segment_name, band, roi_name, all_channels, rois)
                if not cols:
                    continue
                
                val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                group_values = {}
                for label, mask in masks_dict.items():
                    vals = val_series[mask].dropna().values
                    if len(vals) > 0:
                        group_values[label] = vals
                
                if len(group_values) >= 2:
                    data_by_band[band] = group_values
            
            if data_by_band:
                roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_erds_by_condition{suffix}_multigroup"
                
                plot_multi_group_column_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="ERDS",
                    groups=group_labels,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                    multigroup_stats=multigroup_stats,
                )
        
        log_if_present(logger, "info", f"Saved ERDS multi-group column comparison for {len(roi_names)} ROIs")
        return

    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comp_mask_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")

    mask1, mask2, label1, label2 = comp_mask_info
    segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()

    condition_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
    band_colors = {band: get_band_color(band, config) for band in bands}
    plot_cfg = get_plot_config(config)
    n_bands = len(bands)
    n_trials = len(features_df)

    for roi_name in roi_names:
        cell_data = {}
        for col_idx, band in enumerate(bands):
            cols = _get_erds_columns_for_roi(
                features_df,
                segment_name,
                band,
                roi_name,
                all_channels,
                rois,
            )

            if not cols:
                cell_data[col_idx] = None
                continue

            value_series = (
                features_df[cols]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1)
            )
            values1 = value_series[mask1].dropna().values
            values2 = value_series[mask2].dropna().values

            cell_data[col_idx] = {"v1": values1, "v2": values2}

        qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
            stats_dir=stats_dir,
            feature_type="erds",
            feature_keys=bands,
            cell_data=cell_data,
            config=config,
            logger=logger,
        )

        fig, axes = plt.subplots(
            1, n_bands, figsize=(3 * n_bands, 5), squeeze=False
        )

        for col_idx, band in enumerate(bands):
            ax = axes.flatten()[col_idx]
            data = cell_data.get(col_idx)

            if (
                data is None
                or len(data.get("v1", [])) == 0
                or len(data.get("v2", [])) == 0
            ):
                ax.text(
                    0.5,
                    0.5,
                    _NO_DATA_MESSAGE,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.title,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            values1, values2 = data["v1"], data["v2"]

            boxplot = ax.boxplot(
                [values1, values2],
                positions=[0, 1],
                widths=0.4,
                patch_artist=True,
            )
            boxplot["boxes"][0].set_facecolor(condition_colors["v1"])
            boxplot["boxes"][0].set_alpha(_VIOLIN_ALPHA)
            boxplot["boxes"][1].set_facecolor(condition_colors["v2"])
            boxplot["boxes"][1].set_alpha(_VIOLIN_ALPHA)

            rng = np.random.RandomState(seed=42)
            jitter1 = rng.uniform(-0.08, 0.08, len(values1))
            jitter2 = rng.uniform(-0.08, 0.08, len(values2))
            ax.scatter(
                jitter1,
                values1,
                c=condition_colors["v1"],
                alpha=_SCATTER_ALPHA,
                s=6,
            )
            ax.scatter(
                1 + jitter2,
                values2,
                c=condition_colors["v2"],
                alpha=_SCATTER_ALPHA,
                s=6,
            )

            all_values = np.concatenate([values1, values2])
            ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
            yrange = ymax - ymin if ymax > ymin else 0.1
            ax.set_ylim(
                ymin - 0.1 * yrange, ymax + 0.3 * yrange
            )

            if col_idx in qvalues:
                _, qvalue, cohens_d, is_significant = qvalues[col_idx]
                significance_marker = "†" if is_significant else ""
                significance_color = "#d62728" if is_significant else "#333333"
                ax.annotate(
                    f"q={qvalue:.3f}{significance_marker}\nd={cohens_d:.2f}",
                    xy=(0.5, ymax + 0.05 * yrange),
                    ha="center",
                    fontsize=plot_cfg.font.medium,
                    color=significance_color,
                    fontweight="bold" if is_significant else "normal",
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels([label1, label2], fontsize=9)
            ax.set_title(
                band.capitalize(),
                fontweight="bold",
                color=band_colors.get(band, "gray"),
            )
            _format_axis_style(ax)

        n_tests = len(qvalues)
        roi_display = (
            roi_name.replace("_", " ").title()
            if roi_name != "all"
            else "All Channels"
        )

        stats_source = (
            "pre-computed" if use_precomputed else "Mann-Whitney U"
        )
        title = (
            f"ERDS: {label1} vs {label2} (Column Comparison)\n"
            f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
            f"{stats_source} | FDR: {n_significant}/{n_tests} significant "
            f"(†=q<0.05)"
        )
        fig.suptitle(
            title,
            fontsize=plot_cfg.font.suptitle,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        from eeg_pipeline.utils.formatting import sanitize_label
        roi_safe = (
            sanitize_label(roi_name).lower()
            if roi_name != "all"
            else ""
        )
        suffix = f"_roi-{roi_safe}" if roi_safe else ""
        filename = f"sub-{subject}_erds_by_condition{suffix}_column"

        save_fig(
            fig,
            save_dir / filename,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            config=config,
        )
        plt.close(fig)

    log_if_present(
        logger,
        "info",
        f"Saved ERDS column comparison plots for {len(roi_names)} ROIs",
    )


def plot_erds_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare ERDS between conditions per band.

    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.

    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    from eeg_pipeline.plotting.io.figures import log_if_present

    if features_df is None or features_df.empty or events_df is None:
        return

    compare_windows = get_config_value(
        config, "plotting.comparisons.compare_windows", True
    )
    compare_columns = get_config_value(
        config, "plotting.comparisons.compare_columns", False
    )

    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least 2 window names "
            f"(got {segments!r})"
        )
    segments = [str(s) for s in segments]

    bands = get_band_names(config)
    if not bands:
        return

    rois = get_roi_definitions(config)
    all_channels = _extract_channel_identifiers(features_df)
    roi_names = _determine_roi_names(config, rois)

    if logger:
        log_if_present(
            logger,
            "info",
            f"ERDS comparison: segments={segments}, ROIs={roi_names}, "
            f"bands={bands}, compare_windows={compare_windows}, "
            f"compare_columns={compare_columns}",
        )

    ensure_dir(save_dir)

    if compare_windows and len(segments) >= 2:
        _create_window_comparison_plots(
            features_df,
            segments,
            bands,
            roi_names,
            all_channels,
            rois,
            subject,
            save_dir,
            config,
            logger,
            stats_dir,
        )

    if compare_columns:
        _create_column_comparison_plots(
            features_df,
            events_df,
            bands,
            roi_names,
            all_channels,
            rois,
            subject,
            save_dir,
            config,
            logger,
            stats_dir,
        )


__all__ = [
    "plot_erds_by_condition",
]
