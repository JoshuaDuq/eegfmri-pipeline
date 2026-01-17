from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.config import PlotConfig, get_plot_config
from eeg_pipeline.plotting.behavioral.builders import (
    generate_correlation_scatter,
    plot_regression_residual_diagnostics,
    plot_residual_qc,
)
from eeg_pipeline.utils.data import load_subject_scatter_data
from eeg_pipeline.utils.formatting import (
    format_time_suffix,
    get_residual_labels,
    get_target_labels,
    get_temporal_xlabel,
    sanitize_label,
)
from eeg_pipeline.infra.paths import deriv_plots_path, deriv_stats_path, ensure_dir
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.utils.analysis.stats import (
    compute_correlation_stats,
    partial_corr_xy_given_Z,
    compute_partial_residuals as _compute_partial_residuals,
    compute_partial_residuals_stats,
    joint_valid_mask,
    update_stats_from_dataframe,
)


@dataclass
class SubjectScatterData:
    temporal_df: pd.DataFrame
    features_df: pd.DataFrame
    rating_series: pd.Series
    info: mne.Info
    temp_series: Optional[pd.Series]
    covariate_df_full: Optional[pd.DataFrame]
    covariate_df_temp: Optional[pd.DataFrame]
    roi_map: Dict[str, List[str]]
    stats_dir: Path
    plots_dir: Path
    conn_df: Optional[pd.DataFrame] = None


class FeatureColumnExtractor(Protocol):
    """Protocol for extracting feature columns from a DataFrame."""

    def __call__(
        self,
        features_df: pd.DataFrame,
        band: str,
        roi_channels: List[str],
        metric: Optional[str] = None,
    ) -> Tuple[pd.Series, bool]:
        """Extract feature column for given band, ROI channels, and optional metric."""
        ...


@dataclass
class ScatterPlotConfig:
    """Configuration for scatter plot generation."""

    method_code: str
    bootstrap_ci: int
    rng: np.random.Generator
    min_samples_for_plot: int
    significance_threshold: float
    target_rating: str


@dataclass
class ScatterPlotParams:
    """Parameters for generating a single scatter plot."""

    roi_vals: pd.Series
    target_vals: pd.Series
    roi: str
    band: str
    band_title: str
    band_color: str
    metric: Optional[str]
    target_type: str
    title: str
    x_label: str
    y_label: str
    output_path: Path
    roi_channels: List[str]
    feature_name: str


def _get_scatter_plot_config_from_config(config: Any) -> ScatterPlotConfig:
    """Extract scatter plot configuration from main config object."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    validation_config = plot_cfg.validation

    return ScatterPlotConfig(
        method_code="",  # Set by caller
        bootstrap_ci=0,  # Set by caller
        rng=None,  # Set by caller
        min_samples_for_plot=validation_config.get("min_samples_for_plot", 5),
        significance_threshold=float(
            behavioral_config.get("significance_threshold", 0.05)
        ),
        target_rating=behavioral_config.get("target_rating", "rating"),
    )


def setup_scatter_context(
    subject: str,
    deriv_root: Path,
    task: Optional[str],
    plots_dir: Optional[Path],
    feature_subdir: str,
    config: Any,
    logger: logging.Logger,
) -> Optional[SubjectScatterData]:
    """Set up context for scatter plot generation."""
    if config is None:
        raise ValueError("config is required for scatter plot context setup")
    if not subject:
        raise ValueError("subject must be non-empty")
    if not isinstance(deriv_root, Path):
        raise TypeError("deriv_root must be a Path object")

    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()

    if plots_dir is None:
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)

    feature_dir = plots_dir / feature_subdir
    ensure_dir(feature_dir)

    if task is None:
        task = config.task

    result = load_subject_scatter_data(
        subject, task, deriv_root, config, logger, None
    )
    (
        temporal_df,
        features_df,
        rating_series,
        info,
        temp_series,
        covariate_df_full,
        covariate_df_temp,
        roi_map,
        conn_df,
    ) = result

    if temporal_df is None:
        return None

    stats_dir = deriv_stats_path(deriv_root, subject)

    return SubjectScatterData(
        temporal_df=temporal_df,
        features_df=features_df,
        rating_series=rating_series,
        info=info,
        temp_series=temp_series,
        covariate_df_full=covariate_df_full,
        covariate_df_temp=covariate_df_temp,
        roi_map=roi_map,
        stats_dir=stats_dir,
        plots_dir=feature_dir,
        conn_df=conn_df,
    )


def _compute_correlation_statistics(
    roi_vals: pd.Series,
    target_vals: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    precomp_stats: Optional[Dict[str, Any]],
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute correlation statistics from data or use precomputed values."""
    if precomp_stats:
        r_val = precomp_stats["r"]
        p_val = precomp_stats["p"]
        n_eff = precomp_stats["n"]
        ci_val = (precomp_stats.get("ci_low"), precomp_stats.get("ci_high"))
    else:
        r_val, p_val, n_eff, ci_val = compute_correlation_stats(
            roi_vals, target_vals, method_code, bootstrap_ci, rng
        )
    return r_val, p_val, n_eff, ci_val


def _record_scatter_result(
    feature_name: str,
    roi: str,
    target_type: str,
    r_val: float,
    p_val: float,
    n_eff: int,
    output_path: Path,
    results: Dict[str, List],
    significance_threshold: float,
) -> None:
    """Record scatter plot result in results dictionary."""
    record = {
        "feature": feature_name,
        "roi": roi,
        "target": target_type,
        "r": r_val,
        "p": p_val,
        "n": n_eff,
        "path": str(output_path),
    }
    results["all"].append(record)
    if np.isfinite(p_val) and p_val < significance_threshold:
        results["significant"].append(record)


def _generate_single_scatter(
    *,
    params: ScatterPlotParams,
    plot_config: ScatterPlotConfig,
    precomp_stats: Optional[Dict[str, Any]],
    logger: logging.Logger,
    config: Any,
    results: Dict[str, List],
) -> None:
    """Generate a single scatter plot and record results."""
    if params.roi_vals.empty or params.target_vals.empty:
        logger.debug(
            f"Skipping scatter plot for {params.roi} {params.band}: empty data"
        )
        return

    valid_mask = joint_valid_mask(params.roi_vals, params.target_vals)
    n_valid = int(valid_mask.sum())

    r_val, p_val, n_eff, ci_val = _compute_correlation_statistics(
        params.roi_vals,
        params.target_vals,
        plot_config.method_code,
        plot_config.bootstrap_ci,
        plot_config.rng,
        precomp_stats,
    )

    if n_valid >= plot_config.min_samples_for_plot:
        generate_correlation_scatter(
            x_data=params.roi_vals,
            y_data=params.target_vals,
            x_label=params.x_label,
            y_label=params.y_label,
            title_prefix=params.title,
            band_color=params.band_color,
            output_path=params.output_path,
            method_code=plot_config.method_code,
            bootstrap_ci=0,
            rng=plot_config.rng,
            roi_channels=params.roi_channels,
            logger=logger,
            annotated_stats=(r_val, p_val, n_eff),
            annot_ci=ci_val,
            config=config,
        )

        _record_scatter_result(
            params.feature_name,
            params.roi,
            params.target_type,
            r_val,
            p_val,
            n_eff,
            params.output_path,
            results,
            plot_config.significance_threshold,
        )


def _get_source_dataframe(
    data: SubjectScatterData, feature_type: str, logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """Get the appropriate source DataFrame based on feature type."""
    if feature_type == "connectivity":
        if data.conn_df is None or data.conn_df.empty:
            logger.warning("No connectivity data available for scatter plots")
            return None
        return data.conn_df
    return data.features_df


def _plot_rating_scatter(
    *,
    roi_vals: pd.Series,
    data: SubjectScatterData,
    roi: str,
    band: str,
    band_title: str,
    band_color: str,
    metric: Optional[str],
    title_formatter: Callable[[str, str, str, Optional[str]], str],
    x_label_formatter: Callable[[str, Optional[str]], str],
    filename_formatter: Callable[[str, str, Optional[str]], str],
    feature_name_formatter: Callable[[str, Optional[str]], str],
    roi_plots_dir: Path,
    rating_stats: Optional[pd.DataFrame],
    plot_config: PlotConfig,
    logger: logging.Logger,
    config: Any,
    results: Dict[str, List],
    roi_channels: List[str],
) -> None:
    """Generate scatter plot for rating target."""
    title_rating = title_formatter(band_title, roi, "Rating", metric)
    x_label = x_label_formatter(band_title, metric)
    output_path = roi_plots_dir / filename_formatter(band, "rating", metric)
    feature_name = feature_name_formatter(band, metric)

    precomp_stats = _get_precomputed_stats_for_roi_band(
        rating_stats, roi, band, logger
    )

    params = ScatterPlotParams(
        roi_vals=roi_vals,
        target_vals=data.rating_series,
        roi=roi,
        band=band,
        band_title=band_title,
        band_color=band_color,
        metric=metric,
        target_type="rating",
        title=title_rating,
        x_label=x_label,
        y_label="Rating",
        output_path=output_path,
        roi_channels=roi_channels,
        feature_name=feature_name,
    )

    _generate_single_scatter(
        params=params,
        plot_config=plot_config,
        precomp_stats=precomp_stats,
        logger=logger,
        config=config,
        results=results,
    )


def _plot_temperature_scatter(
    *,
    roi_vals: pd.Series,
    data: SubjectScatterData,
    roi: str,
    band: str,
    band_title: str,
    band_color: str,
    metric: Optional[str],
    title_formatter: Callable[[str, str, str, Optional[str]], str],
    x_label_formatter: Callable[[str, Optional[str]], str],
    filename_formatter: Callable[[str, str, Optional[str]], str],
    feature_name_formatter: Callable[[str, Optional[str]], str],
    roi_plots_dir: Path,
    temp_stats: Optional[pd.DataFrame],
    plot_config: PlotConfig,
    logger: logging.Logger,
    config: Any,
    results: Dict[str, List],
    roi_channels: List[str],
) -> None:
    """Generate scatter plot for temperature target."""
    if data.temp_series is None or data.temp_series.empty:
        return

    title_temp = title_formatter(band_title, roi, "Temperature", metric)
    x_label = x_label_formatter(band_title, metric)
    output_path_temp = roi_plots_dir / filename_formatter(band, "temp", metric)
    feature_name = feature_name_formatter(band, metric)

    precomp_stats = _get_precomputed_stats_for_roi_band(temp_stats, roi, band, logger)

    params = ScatterPlotParams(
        roi_vals=roi_vals,
        target_vals=data.temp_series,
        roi=roi,
        band=band,
        band_title=band_title,
        band_color=band_color,
        metric=metric,
        target_type="temp",
        title=title_temp,
        x_label=x_label,
        y_label="Temperature (°C)",
        output_path=output_path_temp,
        roi_channels=roi_channels,
        feature_name=feature_name,
    )

    _generate_single_scatter(
        params=params,
        plot_config=plot_config,
        precomp_stats=precomp_stats,
        logger=logger,
        config=config,
        results=results,
    )


def create_roi_scatter_plots(
    *,
    data: SubjectScatterData,
    feature_type: str,
    column_extractor: FeatureColumnExtractor,
    title_formatter: Callable[[str, str, str, Optional[str]], str],
    x_label_formatter: Callable[[str, Optional[str]], str],
    filename_formatter: Callable[[str, str, Optional[str]], str],
    feature_name_formatter: Callable[[str, Optional[str]], str],
    bands: List[str],
    metrics: Optional[List[str]],
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    include_temperature: bool,
    rating_stats: Optional[pd.DataFrame],
    temp_stats: Optional[pd.DataFrame],
    logger: logging.Logger,
    config: Any,
) -> Dict[str, Any]:
    """Create ROI scatter plots for rating and optionally temperature targets."""
    if config is None:
        raise ValueError("config is required for ROI scatter plots")
    if not bands:
        raise ValueError("bands must be a non-empty list")
    if method_code not in ("pearson", "spearman"):
        raise ValueError(f"method_code must be 'pearson' or 'spearman', got '{method_code}'")

    results = {"significant": [], "all": []}

    source_df = _get_source_dataframe(data, feature_type, logger)
    if source_df is None:
        return results

    plot_cfg = _get_scatter_plot_config_from_config(config)
    plot_cfg.method_code = method_code
    plot_cfg.bootstrap_ci = bootstrap_ci
    plot_cfg.rng = rng

    metric_list = metrics if metrics else [None]

    for metric in metric_list:
        for band in bands:
            band_title = band.capitalize()
            band_color = get_band_color(band, config)

            for roi, roi_channels in data.roi_map.items():
                roi_vals, is_valid = column_extractor(
                    source_df, band, roi_channels, metric
                )
                if not is_valid:
                    continue

                roi_plots_dir = data.plots_dir / sanitize_label(roi)
                ensure_dir(roi_plots_dir)

                _plot_rating_scatter(
                    roi_vals=roi_vals,
                    data=data,
                    roi=roi,
                    band=band,
                    band_title=band_title,
                    band_color=band_color,
                    metric=metric,
                    title_formatter=title_formatter,
                    x_label_formatter=x_label_formatter,
                    filename_formatter=filename_formatter,
                    feature_name_formatter=feature_name_formatter,
                    roi_plots_dir=roi_plots_dir,
                    rating_stats=rating_stats,
                    plot_config=plot_cfg,
                    logger=logger,
                    config=config,
                    results=results,
                    roi_channels=roi_channels,
                )

                if include_temperature:
                    _plot_temperature_scatter(
                        roi_vals=roi_vals,
                        data=data,
                        roi=roi,
                        band=band,
                        band_title=band_title,
                        band_color=band_color,
                        metric=metric,
                        title_formatter=title_formatter,
                        x_label_formatter=x_label_formatter,
                        filename_formatter=filename_formatter,
                        feature_name_formatter=feature_name_formatter,
                        roi_plots_dir=roi_plots_dir,
                        temp_stats=temp_stats,
                        plot_config=plot_cfg,
                        logger=logger,
                        config=config,
                        results=results,
                        roi_channels=roi_channels,
                    )

    return results


def _get_precomputed_stats_for_roi_band(
    stats: Optional[pd.DataFrame],
    roi: str,
    band: str,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Extract precomputed statistics for a specific ROI and band."""
    if stats is None or stats.empty:
        return None

    try:
        has_roi_column = "roi" in stats.columns
        has_band_column = "band" in stats.columns
        if not (has_roi_column and has_band_column):
            return None

        roi_mask = stats["roi"] == roi
        band_mask = stats["band"] == band
        matching_rows = stats[roi_mask & band_mask]

        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]
        r = float(row["r"])
        p = float(row["p"])
        n = int(row["n"]) if "n" in row.index else int(row.get("n_eff", np.nan))
        ci_low = row.get("ci_low", np.nan)
        ci_high = row.get("ci_high", np.nan)

        return {"r": r, "p": p, "n": n, "ci_low": ci_low, "ci_high": ci_high}
    except Exception as exc:
        logger.debug("Failed to lookup precomputed stats: %s", exc)
        return None


def _get_title_components(
    band_title: str,
    target_type: str,
    roi_name: str,
    time_label: Optional[str] = None,
) -> str:
    roi_display = "Overall" if roi_name == "Overall" else roi_name
    time_suffix = format_time_suffix(time_label)
    return f"{band_title} power vs {target_type} — {roi_display}{time_suffix}"


def _get_target_suffix(target_type: str, target_rating: str) -> str:
    """Get filename suffix for target type."""
    return "rating" if target_type == target_rating else "temp"


def _get_time_suffix(time_label: Optional[str]) -> str:
    return f"_{time_label}" if time_label else "_active"


def _get_base_filename(
    band: str, target_type: str, roi_name: str, target_rating: str
) -> str:
    """Generate base filename for scatter plot."""
    band_safe = sanitize_label(band)
    target_suffix = _get_target_suffix(target_type, target_rating)

    is_overall_roi = roi_name in ("All", "Overall")
    if is_overall_roi:
        return f"scatter_pow_overall_{band_safe}_vs_{target_suffix}"
    return f"scatter_pow_{band_safe}_vs_{target_suffix}"


def get_output_filename(
    band: str,
    target_type: str,
    roi_name: str,
    plot_type: str,
    time_label: Optional[str] = None,
    config: Optional[Any] = None,
) -> str:
    """Generate output filename for scatter plot."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    target_rating = behavioral_config.get("target_rating", "rating")

    base = _get_base_filename(band, target_type, roi_name, target_rating)
    time_suffix = _get_time_suffix(time_label)

    plot_type_map = {
        "scatter": f"{base}{time_suffix}",
        "residual_qc": f"residual_qc_{base}{time_suffix}",
        "residual_diagnostics": f"residual_diagnostics_{base}",
        "partial": f"{base}_partial",
    }

    return plot_type_map.get(plot_type, base)


def _get_temporal_columns(
    temporal_df: pd.DataFrame,
    band: str,
    time_label: str,
) -> List[str]:
    columns = []
    for c in temporal_df.columns:
        parsed = NamingSchema.parse(str(c))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        if str(parsed.get("segment") or "") != str(time_label):
            continue
        if str(parsed.get("band") or "") != str(band):
            continue
        if str(parsed.get("scope") or "") != "ch":
            continue
        columns.append(str(c))
    return columns


def _plot_partial_residuals(
    x_data: pd.Series,
    y_data: pd.Series,
    covariate_data: pd.DataFrame,
    method_code: str,
    band_title: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    band_color: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    stats_df: Optional[pd.Series],
    min_samples_for_plot: int,
    logger: logging.Logger,
    config: Any,
    roi_channels: Optional[List[str]] = None,
) -> None:
    """Plot partial residuals after controlling for covariates."""
    valid_mask = joint_valid_mask(x_data, y_data)
    x_valid = x_data.iloc[valid_mask]
    y_valid = y_data.iloc[valid_mask]
    covariate_valid = covariate_data.iloc[valid_mask]

    x_residuals, y_residuals, n_residuals = _compute_partial_residuals(
        x_valid,
        y_valid,
        covariate_valid,
        method_code,
        logger=logger,
        context=f"Partial residuals {roi_name} {target_type} {band_title}",
    )

    if n_residuals < min_samples_for_plot:
        return

    r_resid, p_resid, n_partial, ci_resid = compute_partial_residuals_stats(
        x_residuals, y_residuals, stats_df, n_residuals, method_code, bootstrap_ci, rng
    )

    residual_xlabel, residual_ylabel = get_residual_labels(method_code, target_type)
    title = f"Partial residuals — {band_title} vs {target_type} — {roi_name}"
    output_path = output_dir / get_output_filename(
        band_title.lower(), target_type, roi_name, "partial", config=config
    )

    generate_correlation_scatter(
        x_data=x_residuals,
        y_data=y_residuals,
        x_label=residual_xlabel,
        y_label=residual_ylabel,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        bootstrap_ci=0,
        rng=rng,
        is_partial_residuals=True,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=(r_resid, p_resid, n_partial),
        annot_ci=ci_resid,
        config=config,
    )


def _plot_single_temporal_correlation(
    temporal_vals: pd.Series,
    target_data: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    time_label: str,
    output_dir: Path,
    method_code: str,
    covariate_data: Optional[pd.DataFrame],
    covariate_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    min_samples_for_plot: int,
    target_rating: str,
    logger: logging.Logger,
    config: Any,
    roi_channels: Optional[List[str]],
) -> None:
    """Plot correlation for a single temporal segment."""
    has_covariates = covariate_data is not None and not covariate_data.empty
    if has_covariates:
        annotated_stats = None
        annot_ci = None
    else:
        r_temporal, p_temporal, n_eff_temporal, ci_temporal = compute_correlation_stats(
            temporal_vals,
            target_data,
            method_code,
            bootstrap_ci,
            rng,
            min_samples=min_samples_for_plot,
        )
        annotated_stats = (r_temporal, p_temporal, n_eff_temporal)
        annot_ci = ci_temporal

    x_label = get_temporal_xlabel(time_label)
    y_label = "Rating" if target_type == target_rating else "Temperature (°C)"
    title = _get_title_components(band_title, target_type, roi_name, time_label)

    output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "scatter", time_label, config=config
    )

    generate_correlation_scatter(
        x_data=temporal_vals,
        y_data=target_data,
        x_label=x_label,
        y_label=y_label,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        Z_covars=covariate_data,
        bootstrap_ci=0,
        rng=rng,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=annotated_stats,
        annot_ci=annot_ci,
        config=config,
    )

    qc_output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "residual_qc", time_label, config=config
    )

    plot_residual_qc(
        x_data=temporal_vals,
        y_data=target_data,
        title_prefix=title,
        output_path=qc_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )


def _plot_temporal_correlations(
    temporal_df: pd.DataFrame,
    target_data: pd.Series,
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
    min_samples_for_plot: int,
    target_rating: str,
    time_labels: List[str],
    logger: logging.Logger,
    config: Any,
    roi_channels: Optional[List[str]] = None,
) -> None:
    """Plot correlations for multiple temporal segments."""
    for time_label in time_labels:
        temporal_cols = _get_temporal_columns(temporal_df, band, time_label)
        if not temporal_cols:
            continue

        temporal_vals = (
            temporal_df[temporal_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
        )

        _plot_single_temporal_correlation(
            temporal_vals,
            target_data,
            band,
            band_title,
            band_color,
            target_type,
            roi_name,
            time_label,
            output_dir,
            method_code,
            covariate_data,
            covariate_names,
            bootstrap_ci,
            rng,
            min_samples_for_plot,
            target_rating,
            logger,
            config,
            roi_channels,
        )


def _compute_correlation_with_covariates(
    power_vals: pd.Series,
    target_vals: pd.Series,
    covariate_data: pd.DataFrame,
    method_code: str,
    config: Any,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute partial correlation controlling for covariates."""
    r_val, p_val, n_eff = partial_corr_xy_given_Z(
        power_vals, target_vals, covariate_data, method_code, config=config
    )
    ci_val = (np.nan, np.nan)
    return r_val, p_val, n_eff, ci_val


def _compute_correlation_without_covariates(
    power_vals: pd.Series,
    target_vals: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    min_samples_for_plot: int,
    stats_df: Optional[pd.Series],
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute standard correlation statistics."""
    r_val, p_val, n_eff, ci_val = compute_correlation_stats(
        power_vals, target_vals, method_code, bootstrap_ci, rng, min_samples=min_samples_for_plot
    )
    r_val, p_val, n_eff, ci_val = update_stats_from_dataframe(
        stats_df, r_val, p_val, n_eff, ci_val
    )
    return r_val, p_val, n_eff, ci_val


def plot_target_correlations(
    *,
    power_vals: pd.Series,
    target_vals: pd.Series,
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
    stats_df: Optional[pd.Series],
    logger: logging.Logger,
    config: Any,
    results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    feature_name: Optional[str] = None,
    temporal_df: Optional[pd.DataFrame] = None,
    roi_channels: Optional[List[str]] = None,
) -> None:
    """Plot correlations between power values and target variables."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")

    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    target_rating = behavioral_config.get("target_rating", "rating")
    time_labels = behavioral_config.get("time_labels", ["early", "mid", "late"])

    has_covariates = covariate_data is not None and not covariate_data.empty
    if has_covariates:
        r_val, p_val, n_eff, ci_val = _compute_correlation_with_covariates(
            power_vals, target_vals, covariate_data, method_code, config
        )
    else:
        r_val, p_val, n_eff, ci_val = _compute_correlation_without_covariates(
            power_vals,
            target_vals,
            method_code,
            bootstrap_ci,
            rng,
            min_samples_for_plot,
            stats_df,
        )

    annotated_stats = (r_val, p_val, n_eff)
    annot_ci = ci_val

    x_label, y_label = get_target_labels(target_type)
    title = _get_title_components(band_title, target_type, roi_name)

    output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "scatter", config=config
    )

    generate_correlation_scatter(
        x_data=power_vals,
        y_data=target_vals,
        x_label=x_label,
        y_label=y_label,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        Z_covars=covariate_data,
        bootstrap_ci=0,
        rng=rng,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=annotated_stats,
        annot_ci=annot_ci,
        config=config,
    )

    qc_output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "residual_qc", config=config
    )

    plot_residual_qc(
        x_data=power_vals,
        y_data=target_vals,
        title_prefix=title,
        output_path=qc_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )

    if temporal_df is not None:
        _plot_temporal_correlations(
            temporal_df,
            target_vals,
            band,
            band_title,
            band_color,
            target_type,
            roi_name,
            output_dir,
            method_code,
            covariate_data,
            covariate_names,
            bootstrap_ci,
            rng,
            min_samples_for_plot,
            target_rating,
            time_labels,
            logger,
            config,
            roi_channels,
        )

    diagnostics_output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "residual_diagnostics", config=config
    )

    diagnostics_title = _get_title_components(
        band_title, target_type, roi_name, None
    ).replace(" (active)", "")
    plot_regression_residual_diagnostics(
        x_data=power_vals,
        y_data=target_vals,
        title_prefix=diagnostics_title,
        output_path=diagnostics_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )

    if has_covariates:
        _plot_partial_residuals(
            power_vals,
            target_vals,
            covariate_data,
            method_code,
            band_title,
            target_type,
            roi_name,
            output_dir,
            band_color,
            bootstrap_ci,
            rng,
            stats_df,
            min_samples_for_plot,
            logger,
            config,
            roi_channels,
        )


def load_subject_data(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
    partial_covars: Optional[List[str]] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    pd.DataFrame,
    pd.Series,
    mne.Info,
    Optional[pd.Series],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, List[str]],
]:
    result = load_subject_scatter_data(
        subject, task, deriv_root, config, logger, partial_covars
    )
    return result[:8]


__all__ = [
    "SubjectScatterData",
    "FeatureColumnExtractor",
    "setup_scatter_context",
    "create_roi_scatter_plots",
    "plot_target_correlations",
    "get_output_filename",
    "load_subject_data",
]
