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


def _get_complexity_metrics(config) -> List[str]:
    return config.get("complexity.metrics", ["lzc", "pe"])


def _get_complexity_metric_title(metric: str) -> str:
    titles = {
        "lzc": "LZC",
        "pe": "PE",
    }
    return titles.get(metric, metric.upper())


def _extract_complexity_columns(
    features_df: pd.DataFrame,
    *,
    segment: str,
    band: str,
    metric: str,
    roi_channels: List[str],
) -> List[str]:
    cols: List[str] = []
    roi_set = set(roi_channels)
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "comp":
            continue
        if str(parsed.get("segment") or "") != str(segment):
            continue
        if str(parsed.get("band") or "") != str(band):
            continue
        if str(parsed.get("scope") or "") != "ch":
            continue
        if str(parsed.get("stat") or "") != str(metric):
            continue
        ch = str(parsed.get("identifier") or "")
        if ch in roi_set:
            cols.append(str(col))
    return cols


def _extract_complexity_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    if metric is None:
        return pd.Series(dtype=float), False
    raise RuntimeError(
        "Complexity extractor must be provided via closure with a fixed segment. "
        "Use plot_complexity_roi_scatter(...), which injects the correct extractor."
    )


def _format_complexity_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    metric_title = _get_complexity_metric_title(metric) if metric else "Complexity"
    return f"{metric_title} ({band_title}) vs {target} — {roi}"


def _format_complexity_x_label(band_title: str, metric: Optional[str]) -> str:
    metric_title = _get_complexity_metric_title(metric) if metric else "Complexity"
    return f"{metric_title} ({band_title})"


def _format_complexity_filename(band: str, target: str, metric: Optional[str]) -> str:
    metric_safe = metric if metric else "complexity"
    return f"scatter_complexity_{metric_safe}_{band}_vs_{target}"


def _format_complexity_feature_name(band: str, metric: Optional[str]) -> str:
    metric_safe = metric if metric else "complexity"
    return f"comp_{metric_safe}_{band}"


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
    config=None,
) -> Dict[str, Any]:
    if config is None:
        raise ValueError("config is required for behavioral complexity ROI scatter plotting")
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting complexity ROI scatter plotting for sub-{subject}")

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

    comp_plot_cfg = behavioral_config.get("complexity", {})
    feature_file = str(comp_plot_cfg.get("feature_file", "features_complexity.tsv"))
    segment = str(comp_plot_cfg.get("segment", "active"))

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "complexity", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    comp_path = deriv_features_path(deriv_root, subject) / feature_file
    comp_df = read_table(comp_path) if comp_path.exists() else None
    if comp_df is None or comp_df.empty:
        logger.warning("Complexity features not found at %s", comp_path)
        return {"significant": [], "all": []}
    if len(comp_df) != len(data.y):
        raise ValueError(
            f"Length mismatch: complexity features ({len(comp_df)}) != targets ({len(data.y)})"
        )
    data.features_df = comp_df

    rating_stats = load_precomputed_correlations(
        data.stats_dir,
        "complexity",
        "rating",
        logger,
        method_label=method_label,
    )
    temp_stats = (
        load_precomputed_correlations(
            data.stats_dir,
            "complexity",
            "temperature",
            logger,
            method_label=method_label,
        )
        if do_temp
        else None
    )

    column_extractor = (
        lambda df, band, roi_channels, metric=None: (
            (lambda cols: (
                (df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1), True)
                if cols
                else (pd.Series(dtype=float), False)
            ))(
                _extract_complexity_columns(
                    df,
                    segment=segment,
                    band=str(band),
                    metric=str(metric) if metric is not None else "",
                    roi_channels=roi_channels,
                )
            )
        )
    )

    results = create_roi_scatter_plots(
        data=data,
        feature_type="complexity",
        column_extractor=column_extractor,
        title_formatter=_format_complexity_title,
        x_label_formatter=_format_complexity_x_label,
        filename_formatter=_format_complexity_filename,
        feature_name_formatter=_format_complexity_feature_name,
        bands=config.get("power.bands_to_use") or list(config.get("frequency_bands", {}).keys()),
        metrics=_get_complexity_metrics(config),
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
        "Complexity scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
