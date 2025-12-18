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


def _extract_aperiodic_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    if metric is None:
        return pd.Series(dtype=float), False

    roi_set = set(roi_channels)
    cols: List[str] = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "aperiodic":
            continue
        if parsed.get("segment") != band:
            continue
        if parsed.get("band") != "broadband":
            continue
        if parsed.get("scope") != "ch":
            continue
        if parsed.get("stat") != metric:
            continue
        ch = str(parsed.get("identifier") or "")
        if ch and ch in roi_set:
            cols.append(str(col))

    if not cols:
        return pd.Series(dtype=float), False

    vals = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return vals, True


def _format_aperiodic_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    metric_title = f"1/f {metric.capitalize()}" if metric else "1/f"
    return f"{metric_title} vs {target} — {roi}"


def _format_aperiodic_x_label(band_title: str, metric: Optional[str]) -> str:
    return f"1/f {metric.capitalize()}" if metric else "1/f"


def _format_aperiodic_filename(band: str, target: str, metric: Optional[str]) -> str:
    metric_safe = metric if metric else "aperiodic"
    return f"scatter_aperiodic_{metric_safe}_vs_{target}"


def _format_aperiodic_feature_name(band: str, metric: Optional[str]) -> str:
    return f"aperiodic_{metric}" if metric else "aperiodic"


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
        raise ValueError("config is required for behavioral aperiodic ROI scatter plotting")
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting aperiodic ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    aperiodic_plot_cfg = behavioral_config.get("aperiodic", {})
    segment = str(aperiodic_plot_cfg.get("segment", "baseline"))

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "aperiodic", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    aperiodic_path = deriv_features_path(deriv_root, subject) / "features_aperiodic.tsv"
    aperiodic_df = read_table(aperiodic_path) if aperiodic_path.exists() else None
    if aperiodic_df is None or aperiodic_df.empty:
        logger.warning("Aperiodic features not found at %s", aperiodic_path)
        return {"significant": [], "all": []}
    if len(aperiodic_df) != len(data.y):
        raise ValueError(
            f"Length mismatch: aperiodic features ({len(aperiodic_df)}) != targets ({len(data.y)})"
        )
    data.features_df = aperiodic_df

    rating_stats = load_precomputed_correlations(data.stats_dir, "aperiodic", "rating", logger)
    temp_stats = (
        load_precomputed_correlations(data.stats_dir, "aperiodic", "temperature", logger)
        if do_temp
        else None
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
        "Aperiodic scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
