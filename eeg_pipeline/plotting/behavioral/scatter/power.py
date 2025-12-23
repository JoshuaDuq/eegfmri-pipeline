from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.behavioral.scatter.core import load_subject_data, plot_target_correlations
from eeg_pipeline.utils.analysis.stats import extract_overall_statistics, extract_roi_statistics
from eeg_pipeline.utils.formatting import sanitize_label
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.utils.data.features import get_power_columns_by_band


def plot_power_roi_scatter(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    *,
    rating_stats: Optional[pd.DataFrame] = None,
    temp_stats: Optional[pd.DataFrame] = None,
    plots_dir: Optional[Path] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    if config is None:
        raise ValueError("config is required for behavioral power ROI scatter plotting")
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ROI power scatter plotting for sub-{subject}")

    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()

    if plots_dir is None:
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    if task is None:
        task = config.task

    default_rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    temporal_df, pow_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map = load_subject_data(
        subject, task, deriv_root, config, logger, partial_covars
    )
    if temporal_df is None:
        return {"significant": [], "all": []}

    if rating_stats is None:
        logger.debug("ROI rating statistics not provided; using empirical correlations")
    if do_temp and temp_stats is None:
        logger.debug("ROI temperature statistics not provided; using empirical correlations")

    method_code = "spearman" if use_spearman else "pearson"
    covar_names = list(Z_df_full.columns) if Z_df_full is not None and not Z_df_full.empty else None
    covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None and not Z_df_temp.empty else None

    overall_roi_keys = behavioral_config.get("overall_roi_keys", ["overall", "all", "global"])
    target_rating = behavioral_config.get("target_rating", "rating")
    target_temperature = behavioral_config.get("target_temperature", "temperature")
    power_bands_to_use = config.get("power.bands_to_use") or list(config.get("frequency_bands", {}).keys())
    if not power_bands_to_use:
        logger.error("Frequency bands not configured in eeg_config.yaml")
        return {"significant": [], "all": []}

    results = {"significant": [], "all": []}
    power_cols_by_band = get_power_columns_by_band(pow_df, bands=[str(b) for b in power_bands_to_use])

    for band in power_bands_to_use:
        band_cols = set(power_cols_by_band.get(str(band), []))
        if not band_cols:
            continue

        band_title = band.capitalize()
        band_color = get_band_color(band, config)
        overall_vals = pow_df[list(band_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        overall_plots_dir = plots_dir / "overall"
        ensure_dir(overall_plots_dir)

        stats_overall_rating = extract_overall_statistics(
            rating_stats, band, overall_keys=overall_roi_keys
        )
        plot_target_correlations(
            power_vals=overall_vals,
            target_vals=y,
            band=band,
            band_title=band_title,
            band_color=band_color,
            target_type=target_rating,
            roi_name="Overall",
            output_dir=overall_plots_dir,
            method_code=method_code,
            Z_covars=Z_df_full,
            covar_names=covar_names,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
            stats_df=stats_overall_rating,
            logger=logger,
            config=config,
            results=results,
            feature_name=f"power_{band}",
            temporal_df=temporal_df,
        )

        if do_temp and temp_series is not None and not temp_series.empty:
            stats_overall_temp = extract_overall_statistics(
                temp_stats, band, overall_keys=overall_roi_keys
            )
            plot_target_correlations(
                power_vals=overall_vals,
                target_vals=temp_series,
                band=band,
                band_title=band_title,
                band_color=band_color,
                target_type=target_temperature,
                roi_name="Overall",
                output_dir=overall_plots_dir,
                method_code=method_code,
                Z_covars=Z_df_temp,
                covar_names=covar_names_temp,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
                stats_df=stats_overall_temp,
                logger=logger,
                config=config,
                results=results,
                feature_name=f"power_{band}",
                temporal_df=temporal_df,
            )

        for roi, chs in roi_map.items():
            roi_cols = []
            for ch in chs:
                for c in band_cols:
                    parsed = NamingSchema.parse(str(c))
                    if not parsed.get("valid") or parsed.get("group") != "power":
                        continue
                    if str(parsed.get("band") or "") != str(band):
                        continue
                    if str(parsed.get("segment") or "") != "active":
                        continue
                    if str(parsed.get("scope") or "") != "ch":
                        continue
                    if str(parsed.get("identifier") or "") == str(ch):
                        roi_cols.append(str(c))
            if not roi_cols:
                continue

            roi_vals = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            roi_plots_dir = plots_dir / "roi_scatters" / sanitize_label(roi)
            ensure_dir(roi_plots_dir)

            stats_roi_rating = extract_roi_statistics(rating_stats, roi, band)
            plot_target_correlations(
                power_vals=roi_vals,
                target_vals=y,
                band=band,
                band_title=band_title,
                band_color=band_color,
                target_type=target_rating,
                roi_name=roi,
                output_dir=roi_plots_dir,
                method_code=method_code,
                Z_covars=Z_df_full,
                covar_names=covar_names,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
                stats_df=stats_roi_rating,
                logger=logger,
                config=config,
                results=results,
                feature_name=f"power_{band}",
                temporal_df=temporal_df,
                roi_channels=chs,
            )

            if do_temp and temp_series is not None and not temp_series.empty:
                stats_roi_temp = extract_roi_statistics(temp_stats, roi, band)
                plot_target_correlations(
                    power_vals=roi_vals,
                    target_vals=temp_series,
                    band=band,
                    band_title=band_title,
                    band_color=band_color,
                    target_type=target_temperature,
                    roi_name=roi,
                    output_dir=roi_plots_dir,
                    method_code=method_code,
                    Z_covars=Z_df_temp,
                    covar_names=covar_names_temp,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    stats_df=stats_roi_temp,
                    logger=logger,
                    config=config,
                    results=results,
                    feature_name=f"power_{band}",
                    temporal_df=temporal_df,
                    roi_channels=chs,
                )

    return results
