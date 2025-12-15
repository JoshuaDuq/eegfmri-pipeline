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
from eeg_pipeline.io.logging import get_subject_logger


def _extract_itpc_columns(features_df: pd.DataFrame, band: str, roi_channels: List[str]) -> List[str]:
    cols = []
    for col in features_df.columns:
        if f"itpc_plateau_{band}_ch_" not in col:
            continue
        for ch in roi_channels:
            if f"_ch_{ch}_" in col or col.endswith(f"_ch_{ch}"):
                cols.append(col)
                break
    return cols


def _extract_itpc_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    cols = _extract_itpc_columns(features_df, band, roi_channels)
    if not cols:
        return pd.Series(dtype=float), False
    vals = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return vals, True


def _format_itpc_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    return f"ITPC ({band_title}) vs {target} — {roi}"


def _format_itpc_x_label(band_title: str, metric: Optional[str]) -> str:
    return f"ITPC ({band_title})"


def _format_itpc_filename(band: str, target: str, metric: Optional[str]) -> str:
    return f"scatter_itpc_{band}_vs_{target}"


def _format_itpc_feature_name(band: str, metric: Optional[str]) -> str:
    return f"itpc_{band}"


def plot_itpc_roi_scatter(
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
    logger.info(f"Starting ITPC ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "itpc", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    rating_stats = load_precomputed_correlations(data.stats_dir, "itpc", "rating", logger)
    temp_stats = (
        load_precomputed_correlations(data.stats_dir, "itpc", "temperature", logger)
        if do_temp
        else None
    )

    results = create_roi_scatter_plots(
        data=data,
        feature_type="itpc",
        column_extractor=_extract_itpc_values,
        title_formatter=_format_itpc_title,
        x_label_formatter=_format_itpc_x_label,
        filename_formatter=_format_itpc_filename,
        feature_name_formatter=_format_itpc_feature_name,
        bands=config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"]),
        metrics=None,
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
        "ITPC scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
