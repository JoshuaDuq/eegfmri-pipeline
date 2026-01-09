from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.behavioral.scatter.core import (
    load_subject_data,
    plot_target_correlations,
)
from eeg_pipeline.utils.analysis.stats import (
    extract_overall_statistics,
    extract_roi_statistics,
)
from eeg_pipeline.utils.formatting import sanitize_label
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.utils.data.features import get_power_columns_by_band


def _extract_roi_power_columns(
    band_cols: set[str], roi_channels: List[str], band: str
) -> List[str]:
    """Extract power columns matching ROI channels for a specific band.

    Args:
        band_cols: Set of all power column names for the band.
        roi_channels: List of channel identifiers in the ROI.
        band: Frequency band name.

    Returns:
        List of column names matching the ROI channels for the band.
    """
    roi_columns = []
    for channel in roi_channels:
        for column_name in band_cols:
            parsed = NamingSchema.parse(str(column_name))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if str(parsed.get("band") or "") != str(band):
                continue
            if str(parsed.get("segment") or "") != "active":
                continue
            if str(parsed.get("scope") or "") != "ch":
                continue
            if str(parsed.get("identifier") or "") == str(channel):
                roi_columns.append(str(column_name))
    return roi_columns


def _plot_correlation(
    power_values: pd.Series,
    target_values: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    method_code: str,
    covariate_data: Optional[pd.DataFrame],
    covariate_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    stats_series: Optional[pd.Series],
    logger: Any,
    config: Any,
    results: Dict[str, List[Dict[str, Any]]],
    temporal_df: Optional[pd.DataFrame],
    roi_channels: Optional[List[str]] = None,
) -> None:
    """Plot correlation between power values and target variable.

    Args:
        power_values: Power feature values.
        target_values: Target variable values (rating or temperature).
        band: Frequency band identifier.
        band_title: Display title for the band.
        band_color: Color for the band.
        target_type: Type of target variable.
        roi_name: Name of the ROI.
        output_dir: Directory to save plots.
        method_code: Correlation method ("spearman" or "pearson").
        covariate_data: Optional covariate DataFrame.
        covariate_names: Optional list of covariate column names.
        bootstrap_ci: Bootstrap confidence interval iterations.
        rng: Random number generator.
        stats_series: Optional pre-computed statistics.
        logger: Logger instance.
        config: Configuration object.
        results: Results dictionary to update.
        temporal_df: Optional temporal DataFrame.
        roi_channels: Optional list of ROI channel identifiers.
    """
    plot_target_correlations(
        power_vals=power_values,
        target_vals=target_values,
        band=band,
        band_title=band_title,
        band_color=band_color,
        target_type=target_type,
        roi_name=roi_name,
        output_dir=output_dir,
        method_code=method_code,
        covariate_data=covariate_data,
        covariate_names=covariate_names,
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        stats_df=stats_series,
        logger=logger,
        config=config,
        results=results,
        feature_name=f"power_{band}",
        temporal_df=temporal_df,
        roi_channels=roi_channels,
    )


def _plot_overall_correlations(
    overall_power_values: pd.Series,
    rating_series: pd.Series,
    temperature_series: Optional[pd.Series],
    band: str,
    band_title: str,
    band_color: str,
    target_rating: str,
    target_temperature: str,
    overall_plots_dir: Path,
    method_code: str,
    covariate_data_full: Optional[pd.DataFrame],
    covariate_names_full: Optional[List[str]],
    covariate_data_temp: Optional[pd.DataFrame],
    covariate_names_temp: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    rating_stats: Optional[pd.DataFrame],
    temp_stats: Optional[pd.DataFrame],
    overall_roi_keys: List[str],
    temporal_df: Optional[pd.DataFrame],
    logger: Any,
    config: Any,
    results: Dict[str, List[Dict[str, Any]]],
    do_temperature: bool,
) -> None:
    """Plot overall (across all channels) power correlations.

    Args:
        overall_power_values: Mean power values across all channels.
        rating_series: Rating target values.
        temperature_series: Optional temperature target values.
        band: Frequency band identifier.
        band_title: Display title for the band.
        band_color: Color for the band.
        target_rating: Name of rating target variable.
        target_temperature: Name of temperature target variable.
        overall_plots_dir: Directory to save overall plots.
        method_code: Correlation method.
        covariate_data_full: Covariates for rating analysis.
        covariate_names_full: Covariate names for rating analysis.
        covariate_data_temp: Covariates for temperature analysis.
        covariate_names_temp: Covariate names for temperature analysis.
        bootstrap_ci: Bootstrap confidence interval iterations.
        rng: Random number generator.
        rating_stats: Optional pre-computed rating statistics.
        temp_stats: Optional pre-computed temperature statistics.
        overall_roi_keys: Keys to use for extracting overall statistics.
        temporal_df: Optional temporal DataFrame.
        logger: Logger instance.
        config: Configuration object.
        results: Results dictionary to update.
        do_temperature: Whether to plot temperature correlations.
    """
    rating_stats_series = extract_overall_statistics(
        rating_stats, band, overall_keys=overall_roi_keys
    )
    _plot_correlation(
        power_values=overall_power_values,
        target_values=rating_series,
        band=band,
        band_title=band_title,
        band_color=band_color,
        target_type=target_rating,
        roi_name="Overall",
        output_dir=overall_plots_dir,
        method_code=method_code,
        covariate_data=covariate_data_full,
        covariate_names=covariate_names_full,
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        stats_series=rating_stats_series,
        logger=logger,
        config=config,
        results=results,
        temporal_df=temporal_df,
    )

    if do_temperature and temperature_series is not None and not temperature_series.empty:
        temp_stats_series = extract_overall_statistics(
            temp_stats, band, overall_keys=overall_roi_keys
        )
        _plot_correlation(
            power_values=overall_power_values,
            target_values=temperature_series,
            band=band,
            band_title=band_title,
            band_color=band_color,
            target_type=target_temperature,
            roi_name="Overall",
            output_dir=overall_plots_dir,
            method_code=method_code,
            covariate_data=covariate_data_temp,
            covariate_names=covariate_names_temp,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
            stats_series=temp_stats_series,
            logger=logger,
            config=config,
            results=results,
            temporal_df=temporal_df,
        )


def _plot_roi_correlations(
    roi_power_values: pd.Series,
    roi_name: str,
    roi_channels: List[str],
    rating_series: pd.Series,
    temperature_series: Optional[pd.Series],
    band: str,
    band_title: str,
    band_color: str,
    target_rating: str,
    target_temperature: str,
    roi_plots_dir: Path,
    method_code: str,
    covariate_data_full: Optional[pd.DataFrame],
    covariate_names_full: Optional[List[str]],
    covariate_data_temp: Optional[pd.DataFrame],
    covariate_names_temp: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    rating_stats: Optional[pd.DataFrame],
    temp_stats: Optional[pd.DataFrame],
    temporal_df: Optional[pd.DataFrame],
    logger: Any,
    config: Any,
    results: Dict[str, List[Dict[str, Any]]],
    do_temperature: bool,
) -> None:
    """Plot ROI-specific power correlations.

    Args:
        roi_power_values: Mean power values for the ROI.
        roi_name: Name of the ROI.
        roi_channels: List of channel identifiers in the ROI.
        rating_series: Rating target values.
        temperature_series: Optional temperature target values.
        band: Frequency band identifier.
        band_title: Display title for the band.
        band_color: Color for the band.
        target_rating: Name of rating target variable.
        target_temperature: Name of temperature target variable.
        roi_plots_dir: Directory to save ROI plots.
        method_code: Correlation method.
        covariate_data_full: Covariates for rating analysis.
        covariate_names_full: Covariate names for rating analysis.
        covariate_data_temp: Covariates for temperature analysis.
        covariate_names_temp: Covariate names for temperature analysis.
        bootstrap_ci: Bootstrap confidence interval iterations.
        rng: Random number generator.
        rating_stats: Optional pre-computed rating statistics.
        temp_stats: Optional pre-computed temperature statistics.
        temporal_df: Optional temporal DataFrame.
        logger: Logger instance.
        config: Configuration object.
        results: Results dictionary to update.
        do_temperature: Whether to plot temperature correlations.
    """
    rating_stats_series = extract_roi_statistics(rating_stats, roi_name, band)
    _plot_correlation(
        power_values=roi_power_values,
        target_values=rating_series,
        band=band,
        band_title=band_title,
        band_color=band_color,
        target_type=target_rating,
        roi_name=roi_name,
        output_dir=roi_plots_dir,
        method_code=method_code,
        covariate_data=covariate_data_full,
        covariate_names=covariate_names_full,
        bootstrap_ci=bootstrap_ci,
        rng=rng,
        stats_series=rating_stats_series,
        logger=logger,
        config=config,
        results=results,
        temporal_df=temporal_df,
        roi_channels=roi_channels,
    )

    if do_temperature and temperature_series is not None and not temperature_series.empty:
        temp_stats_series = extract_roi_statistics(temp_stats, roi_name, band)
        _plot_correlation(
            power_values=roi_power_values,
            target_values=temperature_series,
            band=band,
            band_title=band_title,
            band_color=band_color,
            target_type=target_temperature,
            roi_name=roi_name,
            output_dir=roi_plots_dir,
            method_code=method_code,
            covariate_data=covariate_data_temp,
            covariate_names=covariate_names_temp,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
            stats_series=temp_stats_series,
            logger=logger,
            config=config,
            results=results,
            temporal_df=temporal_df,
            roi_channels=roi_channels,
        )


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
    """Plot power ROI scatter plots for behavioral correlations.

    Args:
        subject: Subject identifier.
        deriv_root: Root directory for derivatives.
        task: Optional task name.
        use_spearman: Whether to use Spearman correlation (default: True).
        partial_covars: Optional list of partial correlation covariates.
        do_temp: Whether to plot temperature correlations (default: True).
        bootstrap_ci: Number of bootstrap iterations for CI (default: 0).
        rng: Optional random number generator.
        rating_stats: Optional pre-computed rating statistics.
        temp_stats: Optional pre-computed temperature statistics.
        plots_dir: Optional plots directory path.
        config: Configuration object (required).

    Returns:
        Dictionary with "significant" and "all" result lists.
    """
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

    temporal_df, pow_df, rating_series, info, temp_series, Z_df_full, Z_df_temp, roi_map = (
        load_subject_data(subject, task, deriv_root, config, logger, partial_covars)
    )
    if temporal_df is None:
        return {"significant": [], "all": []}

    if rating_stats is None:
        logger.debug("ROI rating statistics not provided; using empirical correlations")
    if do_temp and temp_stats is None:
        logger.debug("ROI temperature statistics not provided; using empirical correlations")

    method_code = "spearman" if use_spearman else "pearson"
    covariate_names_full = (
        list(Z_df_full.columns) if Z_df_full is not None and not Z_df_full.empty else None
    )
    covariate_names_temp = (
        list(Z_df_temp.columns) if Z_df_temp is not None and not Z_df_temp.empty else None
    )

    overall_roi_keys = behavioral_config.get("overall_roi_keys", ["overall", "all", "global"])
    target_rating = behavioral_config.get("target_rating", "rating")
    target_temperature = behavioral_config.get("target_temperature", "temperature")
    power_bands_to_use = (
        config.get("power.bands_to_use") or list(config.get("frequency_bands", {}).keys())
    )
    if not power_bands_to_use:
        logger.error("Frequency bands not configured in eeg_config.yaml")
        return {"significant": [], "all": []}

    results = {"significant": [], "all": []}
    power_cols_by_band = get_power_columns_by_band(
        pow_df, bands=[str(band) for band in power_bands_to_use]
    )

    for band in power_bands_to_use:
        band_str = str(band)
        band_columns = set(power_cols_by_band.get(band_str, []))
        if not band_columns:
            continue

        band_title = band.capitalize()
        band_color = get_band_color(band, config)
        overall_power_values = (
            pow_df[list(band_columns)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )
        overall_plots_dir = plots_dir / "overall"
        ensure_dir(overall_plots_dir)

        _plot_overall_correlations(
            overall_power_values=overall_power_values,
            rating_series=rating_series,
            temperature_series=temp_series,
            band=band,
            band_title=band_title,
            band_color=band_color,
            target_rating=target_rating,
            target_temperature=target_temperature,
            overall_plots_dir=overall_plots_dir,
            method_code=method_code,
            covariate_data_full=Z_df_full,
            covariate_names_full=covariate_names_full,
            covariate_data_temp=Z_df_temp,
            covariate_names_temp=covariate_names_temp,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
            rating_stats=rating_stats,
            temp_stats=temp_stats,
            overall_roi_keys=overall_roi_keys,
            temporal_df=temporal_df,
            logger=logger,
            config=config,
            results=results,
            do_temperature=do_temp,
        )

        for roi_name, roi_channels in roi_map.items():
            roi_columns = _extract_roi_power_columns(band_columns, roi_channels, band)
            if not roi_columns:
                continue

            roi_power_values = (
                pow_df[roi_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            )
            roi_plots_dir = plots_dir / "roi_scatters" / sanitize_label(roi_name)
            ensure_dir(roi_plots_dir)

            _plot_roi_correlations(
                roi_power_values=roi_power_values,
                roi_name=roi_name,
                roi_channels=roi_channels,
                rating_series=rating_series,
                temperature_series=temp_series,
                band=band,
                band_title=band_title,
                band_color=band_color,
                target_rating=target_rating,
                target_temperature=target_temperature,
                roi_plots_dir=roi_plots_dir,
                method_code=method_code,
                covariate_data_full=Z_df_full,
                covariate_names_full=covariate_names_full,
                covariate_data_temp=Z_df_temp,
                covariate_names_temp=covariate_names_temp,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
                rating_stats=rating_stats,
                temp_stats=temp_stats,
                temporal_df=temporal_df,
                logger=logger,
                config=config,
                results=results,
                do_temperature=do_temp,
            )

    return results
