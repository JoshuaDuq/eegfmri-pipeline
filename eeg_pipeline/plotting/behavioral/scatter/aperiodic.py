from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.behavioral.scatter.core import (
    create_roi_scatter_plots,
    setup_scatter_context,
)
from eeg_pipeline.utils.data import load_precomputed_correlations
from eeg_pipeline.plotting.io.figures import get_default_config as _get_default_config
from eeg_pipeline.infra.logging import get_subject_logger


def _extract_aperiodic_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    if metric is None:
        return pd.Series(dtype=float), False

    roi_cols = []
    for col in features_df.columns:
        if "aperiodic_plateau_broadband_ch_" not in col:
            continue
        if f"_{metric}" not in col:
            continue
        for ch in roi_channels:
            if f"_ch_{ch}_" in col:
                roi_cols.append(col)
                break

    if not roi_cols:
        return pd.Series(dtype=float), False

    vals = features_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
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
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting aperiodic ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "aperiodic", config, logger)
    if data is None:
        return {"significant": [], "all": []}

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
        bands=["broadband"],
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
