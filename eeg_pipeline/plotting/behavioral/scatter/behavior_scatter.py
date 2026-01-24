"""
Unified Behavior Scatter Plot
=============================

Configurable scatter plot for correlating any EEG feature with any behavioral column.
Supports multiple aggregation modes: ROI, Global, Per-Channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import (
    _find_clean_events_path,
    deriv_features_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
)
from eeg_pipeline.infra.tsv import read_table
from eeg_pipeline.plotting.behavioral.builders import generate_correlation_scatter
from eeg_pipeline.plotting.behavioral.scatter.core import (
    SubjectScatterData,
    _get_scatter_plot_config_from_config,
    _generate_single_scatter,
    ScatterPlotConfig,
    ScatterPlotParams,
)
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import get_named_segments
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.utils.analysis.stats import joint_valid_mask
from eeg_pipeline.utils.analysis.stats.correlation import format_correlation_method_label
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names
from eeg_pipeline.utils.data import load_precomputed_correlations, load_subject_scatter_data
from eeg_pipeline.utils.formatting import sanitize_label


class AggregationMode(Enum):
    """Aggregation modes for feature extraction."""
    ROI = "roi"
    GLOBAL = "global"
    CHANNEL = "channel"


@dataclass
class BehaviorScatterConfig:
    """Configuration for behavior scatter plotting."""
    features: List[str]
    columns: List[str]
    aggregation_modes: List[AggregationMode]
    bands: List[str]
    rois: Dict[str, List[str]]
    use_spearman: bool = True
    bootstrap_ci: int = 0


FEATURE_GROUP_MAP = {
    "power": "power",
    "connectivity": "conn",
    "aperiodic": "aperiodic",
    "complexity": "comp",
    "itpc": "itpc",
    "pac": "pac",
    "erds": "erds",
    "spectral": "spectral",
    "ratios": "ratios",
    "asymmetry": "asym",
}

FEATURE_FILE_MAP = {
    "power": ("power", "features_power"),
    "connectivity": ("connectivity", "features_connectivity"),
    "aperiodic": ("aperiodic", "features_aperiodic"),
    "complexity": ("complexity", "features_complexity"),
    "itpc": ("itpc", "features_itpc"),
    "pac": ("pac", "features_pac"),
    "erds": ("erds", "features_erds"),
    "spectral": ("spectral", "features_spectral"),
    "ratios": ("ratios", "features_ratios"),
    "asymmetry": ("asymmetry", "features_asymmetry"),
}

FEATURE_METRICS = {
    "aperiodic": ["slope", "offset"],
    "complexity": ["lzc", "pe"],
}


def _load_feature_df(
    deriv_root: Path,
    subject: str,
    feature_type: str,
    logger,
) -> Optional[pd.DataFrame]:
    """Load feature DataFrame for the given feature type."""
    features_dir = deriv_features_path(deriv_root, subject)
    feature_info = FEATURE_FILE_MAP.get(feature_type)
    if feature_info is None:
        logger.warning(f"Unknown feature type: {feature_type}")
        return None

    folder, base_name = feature_info
    
    candidates = [
        features_dir / folder / f"{base_name}.parquet",
        features_dir / folder / f"{base_name}.tsv",
        features_dir / f"{base_name}.parquet",
        features_dir / f"{base_name}.tsv",
    ]
    
    for feature_path in candidates:
        if feature_path.exists():
            try:
                df = read_table(feature_path)
                if df.empty:
                    logger.debug(f"Feature file is empty: {feature_path}")
                    return None
                return df
            except Exception as e:
                logger.debug(f"Failed to read {feature_path}: {e}")
                continue
    
    logger.debug(f"Feature file not found for {feature_type}. Tried: {[str(c) for c in candidates]}")
    return None


def _get_feature_group_name(feature_type: str) -> str:
    """Get the NamingSchema group name for a feature type."""
    return FEATURE_GROUP_MAP.get(feature_type, feature_type)


def _matches_feature_criteria(
    parsed: Dict[str, Any],
    group: str,
    band: str,
    segment: Optional[str],
    roi_set: Set[str],
    metric: Optional[str] = None,
) -> bool:
    """Check if parsed column matches feature criteria for ROI/channel extraction."""
    if not parsed.get("valid"):
        return False
    if parsed.get("group") != group:
        return False
    if parsed.get("scope") != "ch":
        return False

    parsed_band = parsed.get("band")
    if group == "aperiodic":
        if parsed_band != "broadband":
            return False
        if segment is not None and parsed.get("segment") != segment:
            return False
    else:
        if parsed_band != band:
            return False
        if segment is not None:
            parsed_segment = parsed.get("segment")
            if parsed_segment and parsed_segment != segment:
                return False

    if metric and parsed.get("stat") != metric:
        return False

    channel = parsed.get("identifier")
    return bool(channel and channel in roi_set)


def _extract_matching_columns(
    features_df: pd.DataFrame,
    group: str,
    band: str,
    segment: Optional[str],
    roi_channels: List[str],
    metric: Optional[str] = None,
    logger=None,
) -> List[str]:
    """Extract column names matching the feature criteria."""
    roi_set = set(roi_channels)
    matching = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if _matches_feature_criteria(parsed, group, band, segment, roi_set, metric):
            matching.append(str(col))
    
    if not matching and segment is not None and logger:
        logger.info(
            f"No columns matched for group={group}, band={band}, segment={segment}, "
            f"metric={metric} in {len(roi_channels)} channels. "
            f"Trying without segment constraint..."
        )
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if _matches_feature_criteria(parsed, group, band, None, roi_set, metric):
                matching.append(str(col))
    
    return matching


def _extract_feature_values_roi(
    features_df: pd.DataFrame,
    feature_type: str,
    band: str,
    segment: Optional[str],
    roi_channels: List[str],
    metric: Optional[str] = None,
    logger=None,
) -> Tuple[pd.Series, bool]:
    """Extract feature values averaged across ROI channels."""
    group = _get_feature_group_name(feature_type)
    columns = _extract_matching_columns(
        features_df, group, band, segment, roi_channels, metric, logger
    )
    if not columns:
        if logger:
            segment_str = segment if segment is not None else "any"
            logger.debug(
                f"No matching columns for {feature_type} {band} segment={segment_str} "
                f"metric={metric} in {len(roi_channels)} channels"
            )
        return pd.Series(dtype=float), False

    values = features_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return values, True


def _extract_feature_values_global(
    features_df: pd.DataFrame,
    feature_type: str,
    band: str,
    segment: Optional[str],
    all_channels: List[str],
    metric: Optional[str] = None,
    logger=None,
) -> Tuple[pd.Series, bool]:
    """Extract feature values averaged across all channels (global)."""
    return _extract_feature_values_roi(
        features_df, feature_type, band, segment, all_channels, metric, logger
    )


def _extract_feature_values_channel(
    features_df: pd.DataFrame,
    feature_type: str,
    band: str,
    segment: Optional[str],
    channel: str,
    metric: Optional[str] = None,
    logger=None,
) -> Tuple[pd.Series, bool]:
    """Extract feature values for a single channel."""
    return _extract_feature_values_roi(
        features_df, feature_type, band, segment, [channel], metric, logger
    )


def _get_behavioral_column(
    events_df: pd.DataFrame,
    column_name: str,
) -> Tuple[pd.Series, bool]:
    """Extract behavioral column from events DataFrame."""
    if column_name not in events_df.columns:
        return pd.Series(dtype=float), False

    values = pd.to_numeric(events_df[column_name], errors="coerce")
    if values.isna().all():
        return pd.Series(dtype=float), False

    return values, True


def _format_title(
    feature_type: str,
    band: str,
    target_col: str,
    location: str,
    metric: Optional[str] = None,
) -> str:
    """Format scatter plot title."""
    feature_label = feature_type.capitalize()
    if metric:
        feature_label = f"{feature_label} ({metric})"

    band_label = band.capitalize()
    return f"{feature_label} {band_label} vs {target_col} — {location}"


def _format_x_label(
    feature_type: str,
    band: str,
    metric: Optional[str] = None,
) -> str:
    """Format x-axis label."""
    feature_label = feature_type.capitalize()
    if metric:
        feature_label = f"{feature_label} ({metric})"
    return f"{feature_label} ({band.capitalize()})"


def _format_filename(
    feature_type: str,
    band: str,
    target_col: str,
    location: str,
    metric: Optional[str] = None,
) -> str:
    """Format output filename."""
    parts = [
        "scatter",
        sanitize_label(feature_type),
        sanitize_label(band),
    ]
    if metric:
        parts.append(sanitize_label(metric))
    parts.extend([
        "vs",
        sanitize_label(target_col),
        sanitize_label(location),
    ])
    return "_".join(parts)


def _get_all_channels_from_roi_map(roi_map: Dict[str, List[str]]) -> List[str]:
    """Get all unique channels from ROI map."""
    all_channels = set()
    for channels in roi_map.values():
        all_channels.update(channels)
    return sorted(all_channels)


def _plot_for_aggregation_mode(
    *,
    mode: AggregationMode,
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    feature_type: str,
    band: str,
    segment: Optional[str],
    target_col: str,
    target_values: pd.Series,
    roi_map: Dict[str, List[str]],
    all_channels: List[str],
    metric: Optional[str],
    output_dir: Path,
    plot_config: ScatterPlotConfig,
    band_color: str,
    logger,
    config,
    results: Dict[str, List],
) -> None:
    """Generate scatter plots for a specific aggregation mode."""
    if mode == AggregationMode.GLOBAL:
        feature_vals, valid = _extract_feature_values_global(
            features_df, feature_type, band, segment, all_channels, metric, logger
        )
        if not valid:
            return

        location = "Global"
        global_dir = output_dir / "global"
        ensure_dir(global_dir)

        _generate_scatter_plot(
            feature_vals=feature_vals,
            target_vals=target_values,
            feature_type=feature_type,
            band=band,
            target_col=target_col,
            location=location,
            metric=metric,
            output_dir=global_dir,
            plot_config=plot_config,
            band_color=band_color,
            logger=logger,
            config=config,
            results=results,
            channels=all_channels,
        )

    elif mode == AggregationMode.ROI:
        for roi_name, roi_channels in roi_map.items():
            feature_vals, valid = _extract_feature_values_roi(
                features_df, feature_type, band, segment, roi_channels, metric, logger
            )
            if not valid:
                continue

            roi_dir = output_dir / "roi" / sanitize_label(roi_name)
            ensure_dir(roi_dir)

            _generate_scatter_plot(
                feature_vals=feature_vals,
                target_vals=target_values,
                feature_type=feature_type,
                band=band,
                target_col=target_col,
                location=roi_name,
                metric=metric,
                output_dir=roi_dir,
                plot_config=plot_config,
                band_color=band_color,
                logger=logger,
                config=config,
                results=results,
                channels=roi_channels,
            )

    elif mode == AggregationMode.CHANNEL:
        channel_dir = output_dir / "channel"
        ensure_dir(channel_dir)

        for channel in all_channels:
            feature_vals, valid = _extract_feature_values_channel(
                features_df, feature_type, band, segment, channel, metric, logger
            )
            if not valid:
                continue

            _generate_scatter_plot(
                feature_vals=feature_vals,
                target_vals=target_values,
                feature_type=feature_type,
                band=band,
                target_col=target_col,
                location=channel,
                metric=metric,
                output_dir=channel_dir,
                plot_config=plot_config,
                band_color=band_color,
                logger=logger,
                config=config,
                results=results,
                channels=[channel],
            )


def _generate_scatter_plot(
    *,
    feature_vals: pd.Series,
    target_vals: pd.Series,
    feature_type: str,
    band: str,
    target_col: str,
    location: str,
    metric: Optional[str],
    output_dir: Path,
    plot_config: ScatterPlotConfig,
    band_color: str,
    logger,
    config,
    results: Dict[str, List],
    channels: List[str],
) -> None:
    """Generate a single scatter plot."""
    valid_mask = joint_valid_mask(feature_vals, target_vals)
    n_valid = int(valid_mask.sum())

    if n_valid < plot_config.min_samples_for_plot:
        return

    title = _format_title(feature_type, band, target_col, location, metric)
    x_label = _format_x_label(feature_type, band, metric)
    y_label = target_col
    filename = _format_filename(feature_type, band, target_col, location, metric)
    output_path = output_dir / filename

    params = ScatterPlotParams(
        roi_vals=feature_vals,
        target_vals=target_vals,
        roi=location,
        band=band,
        band_title=band.capitalize(),
        band_color=band_color,
        metric=metric,
        target_type=target_col,
        title=title,
        x_label=x_label,
        y_label=y_label,
        output_path=output_path,
        roi_channels=channels,
        feature_name=f"{feature_type}_{band}" + (f"_{metric}" if metric else ""),
    )

    _generate_single_scatter(
        params=params,
        plot_config=plot_config,
        precomp_stats=None,
        logger=logger,
        config=config,
        results=results,
    )


def plot_behavior_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    *,
    features: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    aggregation_modes: Optional[List[str]] = None,
    use_spearman: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    plots_dir: Optional[Path] = None,
    config=None,
) -> Dict[str, Any]:
    """
    Generate configurable behavior scatter plots.

    Parameters
    ----------
    subject : str
        Subject identifier.
    deriv_root : Path
        Root directory for derivatives.
    task : str, optional
        Task name.
    features : list of str, optional
        Feature types to plot (e.g., ["power", "complexity"]).
        If None, uses config default or all available.
    columns : list of str, optional
        Behavioral columns to correlate with (from events.tsv).
        If None, uses config default (e.g., ["rating", "temperature"]).
    aggregation_modes : list of str, optional
        Aggregation modes: "roi", "global", "channel".
        If None, defaults to ["roi", "global"].
    use_spearman : bool
        Use Spearman correlation (default True).
    bootstrap_ci : int
        Bootstrap iterations for CI (default 0).
    rng : np.random.Generator, optional
        Random number generator.
    plots_dir : Path, optional
        Output directory for plots.
    config : Any
        Configuration object (required).

    Returns
    -------
    dict
        Results dictionary with "significant" and "all" lists.
    """
    if config is None:
        raise ValueError("config is required for behavior scatter plotting")

    logger = get_subject_logger("behavior_scatter", subject)
    logger.info(f"Starting behavior scatter plotting for sub-{subject}")

    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()

    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    if task is None:
        task = config.task

    if plots_dir is None:
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    scatter_dir = plots_dir / "scatter"
    ensure_dir(scatter_dir)

    scatter_config = get_config_value(config, "plotting.plots.behavior.scatter", {})
    if features is None:
        features = scatter_config.get("features", ["power"])
    if columns is None:
        columns = scatter_config.get("columns", ["rating"])
    if aggregation_modes is None:
        agg_str = scatter_config.get("aggregation_modes", ["roi", "global"])
        aggregation_modes = agg_str if isinstance(agg_str, list) else [agg_str]

    modes = []
    for mode_str in aggregation_modes:
        try:
            modes.append(AggregationMode(mode_str.lower()))
        except ValueError:
            logger.warning(f"Unknown aggregation mode: {mode_str}")

    if not modes:
        modes = [AggregationMode.ROI, AggregationMode.GLOBAL]

    bands = get_frequency_band_names(config)
    segment = scatter_config.get("segment")

    result = load_subject_scatter_data(subject, task, deriv_root, config, logger, None)
    (
        _,
        _,
        _,
        info,
        _,
        _,
        _,
        roi_map,
        _,
    ) = result

    if not roi_map:
        logger.warning(f"No ROI map found for sub-{subject}")
        return {"significant": [], "all": []}

    all_channels = _get_all_channels_from_roi_map(roi_map)

    events_path = _find_clean_events_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        config=config,
    )
    if events_path is None or not events_path.exists():
        logger.warning(f"Events file not found for sub-{subject}, task-{task}")
        return {"significant": [], "all": []}

    events_df = read_table(events_path)
    if events_df.empty:
        logger.warning(f"Events file is empty: {events_path}")
        return {"significant": [], "all": []}

    method_code = "spearman" if use_spearman else "pearson"
    scatter_plot_config = _get_scatter_plot_config_from_config(config)
    scatter_plot_config.method_code = method_code
    scatter_plot_config.bootstrap_ci = bootstrap_ci
    scatter_plot_config.rng = rng

    results = {"significant": [], "all": []}
    plots_attempted = 0
    feature_files_found = 0
    feature_files_missing = []

    for feature_type in features:
        features_df = _load_feature_df(deriv_root, subject, feature_type, logger)
        if features_df is None:
            feature_files_missing.append(feature_type)
            continue
        feature_files_found += 1

        if len(features_df) != len(events_df):
            logger.warning(
                f"Length mismatch for {feature_type}: features={len(features_df)}, events={len(events_df)}"
            )
            continue

        available_segments = set()
        available_groups = set()
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid"):
                seg = parsed.get("segment")
                if seg:
                    available_segments.add(seg)
                grp = parsed.get("group")
                if grp:
                    available_groups.add(grp)
        
        if segment is None and available_segments:
            detected_segments = get_named_segments(features_df)
            if detected_segments:
                segment = detected_segments[0]
                logger.info(
                    f"Auto-detected segment '{segment}' from {feature_type} features. "
                    f"Available segments: {sorted(available_segments)}"
                )
        
        if segment is not None and available_segments and segment not in available_segments:
            logger.info(
                f"Requested segment '{segment}' not found in {feature_type} features. "
                f"Available segments: {sorted(available_segments)}. Will try without segment constraint."
            )

        feature_dir = scatter_dir / sanitize_label(feature_type)
        ensure_dir(feature_dir)

        metrics_list = FEATURE_METRICS.get(feature_type, [None])

        for column in columns:
            target_values, valid = _get_behavioral_column(events_df, column)
            if not valid:
                logger.info(f"Column '{column}' not found or invalid in events.tsv, skipping")
                continue

            for band in bands:
                band_color = get_band_color(band, config)
                band_dir = feature_dir / sanitize_label(band)
                ensure_dir(band_dir)

                for metric in metrics_list:
                    for mode in modes:
                        plots_attempted += 1
                        plots_before = len(results["all"])
                        _plot_for_aggregation_mode(
                            mode=mode,
                            features_df=features_df,
                            events_df=events_df,
                            feature_type=feature_type,
                            band=band,
                            segment=segment,
                            target_col=column,
                            target_values=target_values,
                            roi_map=roi_map,
                            all_channels=all_channels,
                            metric=metric,
                            output_dir=band_dir,
                            plot_config=scatter_plot_config,
                            band_color=band_color,
                            logger=logger,
                            config=config,
                            results=results,
                        )
                        plots_after = len(results["all"])
                        if plots_after > plots_before:
                            logger.debug(
                                f"Generated {plots_after - plots_before} plot(s) for "
                                f"{feature_type} {band} {mode.value} vs {column}"
                            )

    logger.info(
        f"Behavior scatter: {len(results['significant'])} significant of {len(results['all'])} total "
        f"(attempted {plots_attempted} plot combinations, found {feature_files_found}/{len(features)} feature files)"
    )
    if feature_files_missing:
        logger.warning(
            f"Feature files not found for: {', '.join(feature_files_missing)}"
        )
    if plots_attempted > 0 and len(results['all']) == 0:
        logger.warning(
            f"No plots generated despite {plots_attempted} attempts. "
            f"This may indicate feature column matching issues (e.g., segment/band/channel mismatches)."
        )
    return results


__all__ = [
    "AggregationMode",
    "BehaviorScatterConfig",
    "plot_behavior_scatter",
]
