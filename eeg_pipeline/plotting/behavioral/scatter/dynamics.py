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


def _get_dynamics_metrics(config) -> List[str]:
    return config.get("dynamics.metrics", ["lzc", "pe", "hjorth_mobility", "hjorth_complexity"])


def _get_dynamics_metric_title(metric: str) -> str:
    titles = {
        "lzc": "LZC",
        "pe": "PE",
        "hjorth_mobility": "Hjorth Mob.",
        "hjorth_complexity": "Hjorth Comp.",
    }
    return titles.get(metric, metric.upper())


def _extract_dynamics_columns(
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
        if parsed.get("group") != "dynamics":
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


def _extract_dynamics_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    if metric is None:
        return pd.Series(dtype=float), False
    raise RuntimeError(
        "Dynamics extractor must be provided via closure with a fixed segment. "
        "Use plot_dynamics_roi_scatter(...), which injects the correct extractor."
    )


def _format_dynamics_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    metric_title = _get_dynamics_metric_title(metric) if metric else "Dynamics"
    return f"{metric_title} ({band_title}) vs {target} — {roi}"


def _format_dynamics_x_label(band_title: str, metric: Optional[str]) -> str:
    metric_title = _get_dynamics_metric_title(metric) if metric else "Dynamics"
    return f"{metric_title} ({band_title})"


def _format_dynamics_filename(band: str, target: str, metric: Optional[str]) -> str:
    metric_safe = metric if metric else "dynamics"
    return f"scatter_dynamics_{metric_safe}_{band}_vs_{target}"


def _format_dynamics_feature_name(band: str, metric: Optional[str]) -> str:
    return f"dynamics_{metric}_{band}" if metric else f"dynamics_{band}"


def plot_dynamics_roi_scatter(
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
        raise ValueError("config is required for behavioral dynamics ROI scatter plotting")
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting dynamics ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    dyn_plot_cfg = behavioral_config.get("dynamics", {})
    feature_file = str(dyn_plot_cfg.get("feature_file", "features_complexity.tsv"))
    segment = str(dyn_plot_cfg.get("segment", "baseline"))

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "dynamics", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    dyn_path = deriv_features_path(deriv_root, subject) / feature_file
    dyn_df = read_table(dyn_path) if dyn_path.exists() else None
    if dyn_df is None or dyn_df.empty:
        logger.warning("Dynamics features not found at %s", dyn_path)
        return {"significant": [], "all": []}
    if len(dyn_df) != len(data.y):
        raise ValueError(
            f"Length mismatch: dynamics features ({len(dyn_df)}) != targets ({len(data.y)})"
        )
    data.features_df = dyn_df

    rating_stats = load_precomputed_correlations(data.stats_dir, "dynamics", "rating", logger)
    temp_stats = (
        load_precomputed_correlations(data.stats_dir, "dynamics", "temperature", logger)
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
                _extract_dynamics_columns(
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
        feature_type="dynamics",
        column_extractor=column_extractor,
        title_formatter=_format_dynamics_title,
        x_label_formatter=_format_dynamics_x_label,
        filename_formatter=_format_dynamics_filename,
        feature_name_formatter=_format_dynamics_feature_name,
        bands=config.get("power.bands_to_use") or list(config.get("frequency_bands", {}).keys()),
        metrics=_get_dynamics_metrics(config),
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
        "Dynamics scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
