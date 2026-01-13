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


def _extract_itpc_columns(features_df: pd.DataFrame, band: str, roi_channels: List[str]) -> List[str]:
    """Extract ITPC column names matching band and ROI channels."""
    roi_set = set(roi_channels)
    matching_columns = []
    
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        
        is_itpc_feature = (
            parsed.get("group") == "itpc"
            and parsed.get("segment") == "active"
            and parsed.get("band") == band
            and parsed.get("scope") == "ch"
        )
        if not is_itpc_feature:
            continue
        
        channel_identifier = parsed.get("identifier")
        if channel_identifier in roi_set:
            matching_columns.append(str(col))
    
    return matching_columns


def _extract_itpc_values(
    features_df: pd.DataFrame,
    band: str,
    roi_channels: List[str],
    metric: Optional[str] = None,
) -> Tuple[pd.Series, bool]:
    """Extract ITPC values by averaging across ROI channels for the given band."""
    matching_columns = _extract_itpc_columns(features_df, band, roi_channels)
    if not matching_columns:
        return pd.Series(dtype=float), False
    
    roi_data = features_df[matching_columns]
    numeric_data = roi_data.apply(pd.to_numeric, errors="coerce")
    averaged_values = numeric_data.mean(axis=1)
    
    return averaged_values, True


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
    if config is None:
        raise ValueError("config is required for behavioral ITPC ROI scatter plotting")
    logger = get_subject_logger("behavior_analysis", subject)
    logger.info(f"Starting ITPC ROI scatter plotting for sub-{subject}")

    behavioral_config = get_plot_config(config).get_behavioral_config()
    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)
    
    raw_robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
    robust_method = None
    if raw_robust_method is not None:
        normalized_method = str(raw_robust_method).strip().lower()
        robust_method = normalized_method if normalized_method else None
    
    correlation_method = "spearman" if use_spearman else "pearson"
    method_label = format_correlation_method_label(correlation_method, robust_method)

    data = setup_scatter_context(subject, deriv_root, task, plots_dir, "itpc", config, logger)
    if data is None:
        return {"significant": [], "all": []}

    features_dir = deriv_features_path(deriv_root, subject)
    itpc_path = features_dir / "itpc" / "features_itpc.tsv"
    if not itpc_path.exists():
        itpc_path = features_dir / "features_itpc.tsv"
    itpc_df = read_table(itpc_path) if itpc_path.exists() else None
    if itpc_df is None or itpc_df.empty:
        logger.warning("ITPC features not found at %s", itpc_path)
        return {"significant": [], "all": []}
    
    num_itpc_samples = len(itpc_df)
    num_target_samples = len(data.rating_series)
    if num_itpc_samples != num_target_samples:
        raise ValueError(
            f"Length mismatch: ITPC features ({num_itpc_samples}) != targets ({num_target_samples})"
        )
    data.features_df = itpc_df

    rating_stats = load_precomputed_correlations(
        data.stats_dir,
        "itpc",
        "rating",
        logger,
        method_label=method_label,
    )
    temp_stats = (
        load_precomputed_correlations(
            data.stats_dir,
            "itpc",
            "temperature",
            logger,
            method_label=method_label,
        )
        if do_temp
        else None
    )

    frequency_bands_config = config.get("frequency_bands", {})
    bands_to_use = config.get("power.bands_to_use") or list(frequency_bands_config.keys())
    
    results = create_roi_scatter_plots(
        data=data,
        feature_type="itpc",
        column_extractor=_extract_itpc_values,
        title_formatter=_format_itpc_title,
        x_label_formatter=_format_itpc_x_label,
        filename_formatter=_format_itpc_filename,
        feature_name_formatter=_format_itpc_feature_name,
        bands=bands_to_use,
        metrics=None,
        method_code=correlation_method,
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        include_temperature=do_temp,
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
