from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.behavioral.scatter.core import (
    create_roi_scatter_plots,
    setup_scatter_context,
)
from eeg_pipeline.infra.paths import deriv_features_path
from eeg_pipeline.infra.tsv import read_table
from eeg_pipeline.utils.data import load_precomputed_correlations
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.utils.analysis.stats.correlation import format_correlation_method_label
from eeg_pipeline.utils.config.loader import get_config_value


def _extract_connectivity_values_for_segment(
    features_df: pd.DataFrame,
    segment: str,
    band: str,
    metric: Optional[str],
) -> Tuple[pd.Series, bool]:
    if metric is None:
        return pd.Series(dtype=float), False

    cols: List[str] = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue

        group = str(parsed.get("group") or "")
        if group not in {"conn", "connectivity"}:
            continue
        if str(parsed.get("segment") or "") != str(segment):
            continue
        if str(parsed.get("band") or "") != str(band):
            continue
        if str(parsed.get("scope") or "") != "global":
            continue
        if str(parsed.get("stat") or "") != str(metric):
            continue

        cols.append(str(col))

    if not cols:
        return pd.Series(dtype=float), False

    vals = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return vals, True


def _extract_connectivity_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    raise RuntimeError(
        "Connectivity extractor must be provided via closure with a fixed segment. "
        "Use plot_connectivity_roi_scatter(...), which injects the correct extractor."
    )


def _format_connectivity_metric_label(metric: Optional[str]) -> str:
    if metric is None:
        return "Connectivity"
    metric_str = str(metric)
    if metric_str.startswith("wpli"):
        return "wPLI"
    if metric_str.startswith("aec"):
        return "AEC"
    return metric_str


def _format_connectivity_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    metric_label = _format_connectivity_metric_label(metric)
    return f"{metric_label} ({band_title}) vs {target} — {roi}"


def _format_connectivity_x_label(band_title: str, metric: Optional[str]) -> str:
    metric_label = _format_connectivity_metric_label(metric)
    return f"{metric_label} ({band_title})"


def _format_connectivity_filename(band: str, target: str, metric: Optional[str]) -> str:
    metric_safe = str(metric) if metric is not None else "connectivity"
    return f"scatter_conn_{metric_safe}_{band}_vs_{target}"


def _format_connectivity_feature_name(band: str, metric: Optional[str]) -> str:
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
    if config is None:
        raise ValueError("config is required for behavioral connectivity ROI scatter plotting")
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting connectivity ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
    if robust_method is not None:
        robust_method = str(robust_method).strip().lower() or None
    method_label = format_correlation_method_label(
        "spearman" if use_spearman else "pearson",
        robust_method,
    )

    conn_plot_cfg = behavioral_config.get("connectivity", {})
    segment = str(conn_plot_cfg.get("segment", "plateau"))
    metrics = conn_plot_cfg.get("metrics", ["wpli_mean", "aec_mean"])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("plotting.behavioral.connectivity.metrics must be a non-empty list")
    bands = config.get("power.bands_to_use") or list(config.get("frequency_bands", {}).keys())
    if not bands:
        raise ValueError("No frequency bands configured for connectivity scatter")

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "connectivity", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    conn_path = deriv_features_path(deriv_root, subject) / "features_connectivity.parquet"
    conn_df = read_table(conn_path) if conn_path.exists() else None
    if conn_df is None or conn_df.empty:
        logger.warning("Connectivity features not found at %s", conn_path)
        return {"significant": [], "all": []}
    if len(conn_df) != len(data.y):
        raise ValueError(
            f"Length mismatch: connectivity features ({len(conn_df)}) != targets ({len(data.y)})"
        )
    data.conn_df = conn_df
    data.roi_map = {"Overall": []}

    rating_stats = load_precomputed_correlations(
        data.stats_dir,
        "connectivity",
        "rating",
        logger,
        method_label=method_label,
    )
    temp_stats = (
        load_precomputed_correlations(
            data.stats_dir,
            "connectivity",
            "temperature",
            logger,
            method_label=method_label,
        )
        if do_temp
        else None
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
        method_code="spearman" if use_spearman else "pearson",
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        do_temp=do_temp,
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
