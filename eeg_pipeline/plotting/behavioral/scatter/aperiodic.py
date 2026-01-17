from __future__ import annotations

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


def _matches_aperiodic_criteria(
    parsed: Dict[str, Any],
    band: str,
    metric: str,
    roi_set: Set[str],
) -> bool:
    """Check if parsed column name matches aperiodic feature criteria."""
    if not parsed.get("valid"):
        return False
    if parsed.get("group") != "aperiodic":
        return False
    if parsed.get("segment") != band:
        return False
    if parsed.get("band") != "broadband":
        return False
    if parsed.get("scope") != "ch":
        return False
    if parsed.get("stat") != metric:
        return False
    channel = str(parsed.get("identifier") or "")
    return bool(channel and channel in roi_set)


def _extract_aperiodic_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    if metric is None:
        return pd.Series(dtype=float), False

    roi_set = set(roi_channels)
    matching_columns = [
        str(col)
        for col in features_df.columns
        if _matches_aperiodic_criteria(
            NamingSchema.parse(str(col)), band, metric, roi_set
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


def _format_aperiodic_title(
    _band_title: str, roi: str, target: str, metric: Optional[str]
) -> str:
    metric_title = f"1/f {metric.capitalize()}" if metric else "1/f"
    return f"{metric_title} vs {target} — {roi}"


def _format_aperiodic_x_label(_band_title: str, metric: Optional[str]) -> str:
    if metric:
        return f"1/f {metric.capitalize()}"
    return "1/f"


def _format_aperiodic_filename(_band: str, target: str, metric: Optional[str]) -> str:
    metric_safe = metric if metric else "aperiodic"
    return f"scatter_aperiodic_{metric_safe}_vs_{target}"


def _format_aperiodic_feature_name(_band: str, metric: Optional[str]) -> str:
    if metric:
        return f"aperiodic_{metric}"
    return "aperiodic"


def _load_aperiodic_features(
    deriv_root: Path, subject: str, logger: Any
) -> Optional[pd.DataFrame]:
    """Load aperiodic features table for the subject."""
    features_dir = deriv_features_path(deriv_root, subject)
    aperiodic_path = features_dir / "aperiodic" / "features_aperiodic.tsv"
    
    if not aperiodic_path.exists():
        logger.warning("Aperiodic features not found at %s", aperiodic_path)
        return None

    features_df = read_table(aperiodic_path)
    if features_df.empty:
        logger.warning("Aperiodic features table is empty at %s", aperiodic_path)
        return None

    return features_df


def _validate_feature_target_length(
    features_df: pd.DataFrame, target_series: pd.Series
) -> None:
    """Validate that features and targets have matching lengths."""
    if len(features_df) != len(target_series):
        raise ValueError(
            f"Length mismatch: aperiodic features ({len(features_df)}) "
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
        stats_dir, "aperiodic", "rating", logger, method_label=method_label
    )

    temp_stats = None
    if include_temperature:
        temp_stats = load_precomputed_correlations(
            stats_dir, "aperiodic", "temperature", logger, method_label=method_label
        )

    return rating_stats, temp_stats


def plot_aperiodic_roi_scatter(
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
    if config is None:
        raise ValueError(
            "config is required for behavioral aperiodic ROI scatter plotting"
        )

    logger = get_subject_logger("behavior_analysis", subject)
    logger.info(f"Starting aperiodic ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    method_label = _get_correlation_method_label(use_spearman, config)
    method_code = "spearman" if use_spearman else "pearson"

    aperiodic_plot_cfg = behavioral_config.get("aperiodic", {})
    segment = str(aperiodic_plot_cfg.get("segment", "baseline"))

    data = setup_scatter_context(
        subject, deriv_root, task, plots_dir, "aperiodic", config, logger
    )
    if data is None:
        return {"significant": [], "all": []}

    aperiodic_df = _load_aperiodic_features(deriv_root, subject, logger)
    if aperiodic_df is None:
        return {"significant": [], "all": []}

    _validate_feature_target_length(aperiodic_df, data.rating_series)
    data.features_df = aperiodic_df

    rating_stats, temp_stats = _load_precomputed_stats(
        data.stats_dir, logger, method_label, do_temp
    )

    results = create_roi_scatter_plots(
        data=data,
        feature_type="aperiodic",
        column_extractor=_extract_aperiodic_values,
        title_formatter=_format_aperiodic_title,
        x_label_formatter=_format_aperiodic_x_label,
        filename_formatter=_format_aperiodic_filename,
        feature_name_formatter=_format_aperiodic_feature_name,
        bands=[segment],
        metrics=["slope", "offset"],
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
        "Aperiodic scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
