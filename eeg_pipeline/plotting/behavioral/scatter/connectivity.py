from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.plotting.behavioral.scatter.core import (
    create_roi_scatter_plots,
    setup_scatter_context,
)
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.stats.correlation import format_correlation_method_label
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data import load_precomputed_correlations


def _matches_connectivity_criteria(
    parsed: Dict[str, Any],
    segment: str,
    band: str,
    metric: str,
) -> bool:
    """Check if parsed column name matches connectivity feature criteria."""
    if not parsed.get("valid"):
        return False

    group = str(parsed.get("group") or "")
    if group not in {"conn", "connectivity"}:
        return False
    if str(parsed.get("segment") or "") != str(segment):
        return False
    if str(parsed.get("band") or "") != str(band):
        return False
    if str(parsed.get("scope") or "") != "global":
        return False
    if str(parsed.get("stat") or "") != str(metric):
        return False

    return True


def _extract_connectivity_values_for_segment(
    features_df: pd.DataFrame,
    segment: str,
    band: str,
    metric: Optional[str],
) -> Tuple[pd.Series, bool]:
    """Extract connectivity values for given segment, band, and metric."""
    if metric is None:
        return pd.Series(dtype=float), False

    matching_columns = [
        str(col)
        for col in features_df.columns
        if _matches_connectivity_criteria(
            NamingSchema.parse(str(col)), segment, band, metric
        )
    ]

    if not matching_columns:
        return pd.Series(dtype=float), False

    values = (
        features_df[matching_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )
    return values, True


def _get_correlation_method_label(use_spearman: bool, config: Any) -> str:
    """Get formatted correlation method label from config."""
    robust_method = get_config_value(
        config, "behavior_analysis.robust_correlation", None
    )
    if robust_method is not None:
        robust_method = str(robust_method).strip().lower() or None

    method_name = "spearman" if use_spearman else "pearson"
    return format_correlation_method_label(method_name, robust_method)


def _load_precomputed_stats(
    stats_dir: Path,
    logger: Any,
    method_label: str,
    include_temperature: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load precomputed correlation statistics."""
    rating_stats = load_precomputed_correlations(
        stats_dir, "connectivity", "rating", logger, method_label=method_label
    )

    temp_stats = None
    if include_temperature:
        temp_stats = load_precomputed_correlations(
            stats_dir, "connectivity", "temperature", logger, method_label=method_label
        )

    return rating_stats, temp_stats


def _validate_feature_target_length(
    features_df: pd.DataFrame, target_series: pd.Series
) -> None:
    """Validate that features and targets have matching lengths."""
    if len(features_df) != len(target_series):
        raise ValueError(
            f"Length mismatch: connectivity features ({len(features_df)}) "
            f"!= targets ({len(target_series)})"
        )


def _format_connectivity_metric_label(metric: Optional[str]) -> str:
    """Get display label for a connectivity metric."""
    if metric is None:
        return "Connectivity"
    metric_str = str(metric)
    if metric_str.startswith("wpli"):
        return "wPLI"
    if metric_str.startswith("aec"):
        return "AEC"
    return metric_str


def _format_connectivity_title(
    band_title: str, roi: str, target: str, metric: Optional[str]
) -> str:
    """Format plot title for connectivity scatter plot."""
    metric_label = _format_connectivity_metric_label(metric)
    return f"{metric_label} ({band_title}) vs {target} — {roi}"


def _format_connectivity_x_label(band_title: str, metric: Optional[str]) -> str:
    """Format x-axis label for connectivity scatter plot."""
    metric_label = _format_connectivity_metric_label(metric)
    return f"{metric_label} ({band_title})"


def _format_connectivity_filename(band: str, target: str, metric: Optional[str]) -> str:
    """Format output filename for connectivity scatter plot."""
    metric_safe = str(metric) if metric is not None else "connectivity"
    return f"scatter_conn_{metric_safe}_{band}_vs_{target}"


def _format_connectivity_feature_name(band: str, metric: Optional[str]) -> str:
    """Format feature name for connectivity metric."""
    metric_safe = str(metric) if metric is not None else "connectivity"
    return f"conn_{metric_safe}_{band}"


def plot_connectivity_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    plots_dir: Optional[Path] = None,
    config=None,
) -> Dict[str, Any]:
    """Generate ROI scatter plots for connectivity features vs behavioral targets."""
    if config is None:
        raise ValueError(
            "config is required for behavioral connectivity ROI scatter plotting"
        )

    logger = get_subject_logger("behavior_analysis", subject)
    logger.info(f"Starting connectivity ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    method_label = _get_correlation_method_label(use_spearman, config)
    method_code = "spearman" if use_spearman else "pearson"

    conn_plot_cfg = behavioral_config.get("connectivity", {})
    segment = str(conn_plot_cfg.get("segment", "active"))
    metrics = conn_plot_cfg.get("metrics", ["wpli_mean", "aec_mean"])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(
            "plotting.behavioral.connectivity.metrics must be a non-empty list"
        )

    bands = (
        config.get("power.bands_to_use")
        or list(config.get("frequency_bands", {}).keys())
    )
    if not bands:
        raise ValueError("No frequency bands configured for connectivity scatter")

    data = setup_scatter_context(
        subject, deriv_root, task, plots_dir, "connectivity", config, logger
    )
    if data is None:
        return {"significant": [], "all": []}

    if data.conn_df is None or data.conn_df.empty:
        logger.warning(
            f"No connectivity data available for scatter plots (sub-{subject})"
        )
        return {"significant": [], "all": []}

    _validate_feature_target_length(data.conn_df, data.rating_series)

    data.roi_map = {"Overall": []}

    rating_stats, temp_stats = _load_precomputed_stats(
        data.stats_dir, logger, method_label, do_temp
    )

    column_extractor = (
        lambda df, band, roi_channels, metric=None: _extract_connectivity_values_for_segment(
            df, segment=segment, band=band, metric=metric
        )
    )

    results = create_roi_scatter_plots(
        data=data,
        feature_type="connectivity",
        column_extractor=column_extractor,
        title_formatter=_format_connectivity_title,
        x_label_formatter=_format_connectivity_x_label,
        filename_formatter=_format_connectivity_filename,
        feature_name_formatter=_format_connectivity_feature_name,
        bands=[str(b) for b in bands],
        metrics=metrics,
        method_code=method_code,
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        include_temperature=do_temp,
        rating_stats=rating_stats,
        temp_stats=temp_stats,
        logger=logger,
        config=config,
    )

    logger.info(
        "Connectivity scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
