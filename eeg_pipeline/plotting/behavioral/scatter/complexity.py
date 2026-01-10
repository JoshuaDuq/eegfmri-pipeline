from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import deriv_features_path
from eeg_pipeline.infra.tsv import read_table
from eeg_pipeline.plotting.behavioral.scatter.core import (
    create_roi_scatter_plots,
    setup_scatter_context,
)
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.stats.correlation import format_correlation_method_label
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data import load_precomputed_correlations


def _get_complexity_metrics(config: Any) -> List[str]:
    """Get list of complexity metrics from config."""
    return config.get("complexity.metrics", ["lzc", "pe"])


def _get_complexity_metric_title(metric: str) -> str:
    """Get display title for a complexity metric."""
    titles = {
        "lzc": "LZC",
        "pe": "PE",
    }
    return titles.get(metric, metric.upper())


def _matches_complexity_criteria(
    parsed: Dict[str, Any],
    segment: str,
    band: str,
    metric: str,
    roi_set: Set[str],
) -> bool:
    """Check if parsed column name matches complexity feature criteria."""
    if not parsed.get("valid"):
        return False
    if parsed.get("group") != "comp":
        return False
    if parsed.get("segment") != segment:
        return False
    if parsed.get("band") != band:
        return False
    if parsed.get("scope") != "ch":
        return False
    if parsed.get("stat") != metric:
        return False
    channel = parsed.get("identifier") or ""
    return bool(channel and channel in roi_set)


def _extract_complexity_columns(
    features_df: pd.DataFrame,
    *,
    segment: str,
    band: str,
    metric: str,
    roi_channels: List[str],
) -> List[str]:
    """Extract column names matching complexity criteria."""
    roi_set = set(roi_channels)
    matching_columns = [
        str(col)
        for col in features_df.columns
        if _matches_complexity_criteria(
            NamingSchema.parse(str(col)), segment, band, metric, roi_set
        )
    ]
    return matching_columns


def _extract_complexity_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
    *,
    segment: str,
) -> Tuple[pd.Series, bool]:
    """Extract complexity values for given band, ROI channels, and metric."""
    if metric is None:
        return pd.Series(dtype=float), False

    matching_columns = _extract_complexity_columns(
        features_df,
        segment=segment,
        band=band,
        metric=metric,
        roi_channels=roi_channels,
    )

    if not matching_columns:
        return pd.Series(dtype=float), False

    values = (
        features_df[matching_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )
    return values, True


def _format_complexity_title(
    band_title: str, roi: str, target: str, metric: Optional[str]
) -> str:
    """Format plot title for complexity scatter plot."""
    metric_title = _get_complexity_metric_title(metric) if metric else "Complexity"
    return f"{metric_title} ({band_title}) vs {target} — {roi}"


def _format_complexity_x_label(band_title: str, metric: Optional[str]) -> str:
    """Format x-axis label for complexity scatter plot."""
    metric_title = _get_complexity_metric_title(metric) if metric else "Complexity"
    return f"{metric_title} ({band_title})"


def _format_complexity_filename(band: str, target: str, metric: Optional[str]) -> str:
    """Format output filename for complexity scatter plot."""
    metric_safe = metric if metric else "complexity"
    return f"scatter_complexity_{metric_safe}_{band}_vs_{target}"


def _format_complexity_feature_name(band: str, metric: Optional[str]) -> str:
    """Format feature name for complexity metric."""
    metric_safe = metric if metric else "complexity"
    return f"comp_{metric_safe}_{band}"


def _load_complexity_features(
    deriv_root: Path, subject: str, feature_file: str, logger: Any
) -> Optional[pd.DataFrame]:
    """Load complexity features table for the subject."""
    features_dir = deriv_features_path(deriv_root, subject)
    comp_path = features_dir / "complexity" / feature_file
    if not comp_path.exists():
        comp_path = features_dir / feature_file
    if not comp_path.exists():
        logger.warning("Complexity features not found at %s", comp_path)
        return None

    features_df = read_table(comp_path)
    if features_df.empty:
        logger.warning("Complexity features table is empty at %s", comp_path)
        return None

    return features_df


def _validate_feature_target_length(
    features_df: pd.DataFrame, target_series: pd.Series
) -> None:
    """Validate that features and targets have matching lengths."""
    if len(features_df) != len(target_series):
        raise ValueError(
            f"Length mismatch: complexity features ({len(features_df)}) "
            f"!= targets ({len(target_series)})"
        )


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
        stats_dir, "complexity", "rating", logger, method_label=method_label
    )

    temp_stats = None
    if include_temperature:
        temp_stats = load_precomputed_correlations(
            stats_dir, "complexity", "temperature", logger, method_label=method_label
        )

    return rating_stats, temp_stats


def plot_complexity_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    plots_dir: Optional[Path] = None,
    config: Any = None,
) -> Dict[str, Any]:
    """Generate ROI scatter plots for complexity features vs behavioral targets."""
    if config is None:
        raise ValueError(
            "config is required for behavioral complexity ROI scatter plotting"
        )

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting complexity ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    method_label = _get_correlation_method_label(use_spearman, config)
    method_code = "spearman" if use_spearman else "pearson"

    comp_plot_cfg = behavioral_config.get("complexity", {})
    feature_file = str(comp_plot_cfg.get("feature_file", "features_complexity.tsv"))
    segment = str(comp_plot_cfg.get("segment", "active"))

    data = setup_scatter_context(
        subject, deriv_root, task, plots_dir, "complexity", config, logger
    )
    if data is None:
        return {"significant": [], "all": []}

    comp_df = _load_complexity_features(deriv_root, subject, feature_file, logger)
    if comp_df is None:
        return {"significant": [], "all": []}

    _validate_feature_target_length(comp_df, data.rating_series)
    data.features_df = comp_df

    rating_stats, temp_stats = _load_precomputed_stats(
        data.stats_dir, logger, method_label, do_temp
    )

    column_extractor = partial(
        _extract_complexity_values, segment=segment
    )

    bands = (
        config.get("power.bands_to_use")
        or list(config.get("frequency_bands", {}).keys())
    )
    metrics = _get_complexity_metrics(config)

    results = create_roi_scatter_plots(
        data=data,
        feature_type="complexity",
        column_extractor=column_extractor,
        title_formatter=_format_complexity_title,
        x_label_formatter=_format_complexity_x_label,
        filename_formatter=_format_complexity_filename,
        feature_name_formatter=_format_complexity_feature_name,
        bands=bands,
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
        "Complexity scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
