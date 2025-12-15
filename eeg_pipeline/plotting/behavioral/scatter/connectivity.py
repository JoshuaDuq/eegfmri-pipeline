from __future__ import annotations

import re
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


def _extract_within_roi_connectivity(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
) -> pd.Series:
    cols = []
    for col in features_df.columns:
        if f"conn_plateau_{band}_" not in col:
            continue
        match = re.search(r"_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_", col)
        if match:
            ch1, ch2 = match.group(1), match.group(2)
            if ch1 in roi_channels and ch2 in roi_channels:
                cols.append(col)

    if not cols:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    return features_df[cols].mean(axis=1)


def _extract_connectivity_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    vals = _extract_within_roi_connectivity(features_df, band, roi_channels)
    if vals.isna().all():
        return vals, False
    return vals, True


def _format_connectivity_title(band_title: str, roi: str, target: str, metric: Optional[str]) -> str:
    return f"wPLI ({band_title}) vs {target} — {roi}"


def _format_connectivity_x_label(band_title: str, metric: Optional[str]) -> str:
    return f"wPLI ({band_title})"


def _format_connectivity_filename(band: str, target: str, metric: Optional[str]) -> str:
    return f"scatter_conn_{band}_vs_{target}"


def _format_connectivity_feature_name(band: str, metric: Optional[str]) -> str:
    return f"conn_{band}"


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
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting connectivity ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "connectivity", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    rating_stats = load_precomputed_correlations(data.stats_dir, "connectivity", "rating", logger)
    temp_stats = (
        load_precomputed_correlations(data.stats_dir, "connectivity", "temperature", logger)
        if do_temp
        else None
    )

    results = create_roi_scatter_plots(
        data=data,
        feature_type="connectivity",
        column_extractor=_extract_connectivity_values,
        title_formatter=_format_connectivity_title,
        x_label_formatter=_format_connectivity_x_label,
        filename_formatter=_format_connectivity_filename,
        feature_name_formatter=_format_connectivity_feature_name,
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
        "Connectivity scatter: %d significant of %d total",
        len(results["significant"]),
        len(results["all"]),
    )
    return results
