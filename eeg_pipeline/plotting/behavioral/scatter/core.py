from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
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
from eeg_pipeline.utils.data import (
    load_subject_scatter_data,
)
from eeg_pipeline.utils.formatting import (
    format_time_suffix,
    get_residual_labels,
    get_target_labels,
    get_temporal_xlabel,
    sanitize_label,
)
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import deriv_plots_path, deriv_stats_path, ensure_dir
from eeg_pipeline.plotting.io.figures import (
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    save_fig,
)
from eeg_pipeline.utils.analysis.stats import (
    bootstrap_corr_ci as _bootstrap_corr_ci,
    compute_correlation_stats,
    partial_corr_xy_given_Z,
    compute_partial_residuals as _compute_partial_residuals,
    compute_partial_residuals_stats,
    compute_kde_scale,
    joint_valid_mask,
    update_stats_from_dataframe,
)
from eeg_pipeline.utils.data.features import infer_power_band


@dataclass
class SubjectScatterData:
    temporal_df: pd.DataFrame
    features_df: pd.DataFrame
    y: pd.Series
    info: mne.Info
    temp_series: Optional[pd.Series]
    Z_df_full: Optional[pd.DataFrame]
    Z_df_temp: Optional[pd.DataFrame]
    roi_map: Dict[str, List[str]]
    stats_dir: Path
    plots_dir: Path
    conn_df: Optional[pd.DataFrame] = None


class FeatureColumnExtractor(Protocol):
    def __call__(
        self,
        features_df: pd.DataFrame,
        band: str,
        roi_channels: List[str],
        metric: Optional[str] = None,
    ) -> Tuple[pd.Series, bool]:
        ...


def setup_scatter_context(
    subject: str,
    deriv_root: Path,
    task: Optional[str],
    plots_dir: Optional[Path],
    feature_subdir: str,
    config,
    logger: logging.Logger,
) -> Optional[SubjectScatterData]:
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()

    if plots_dir is None:
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)

    feature_dir = plots_dir / feature_subdir
    ensure_dir(feature_dir)

    if task is None:
        task = config.task

    result = load_subject_scatter_data(subject, task, deriv_root, config, logger, None)
    temporal_df, features_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map, conn_df = result

    if temporal_df is None:
        return None

    stats_dir = deriv_stats_path(deriv_root, subject)

    return SubjectScatterData(
        temporal_df=temporal_df,
        features_df=features_df,
        y=y,
        info=info,
        temp_series=temp_series,
        Z_df_full=Z_df_full,
        Z_df_temp=Z_df_temp,
        roi_map=roi_map,
        stats_dir=stats_dir,
        plots_dir=feature_dir,
        conn_df=conn_df,
    )


def _generate_single_scatter(
    *,
    roi_vals: pd.Series,
    target_vals: pd.Series,
    roi: str,
    band: str,
    band_title: str,
    band_color: str,
    metric: Optional[str],
    target_type: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    chs: List[str],
    precomp_stats: Optional[Dict[str, Any]],
    logger: logging.Logger,
    config: Any,
    results: Dict[str, List],
    feature_name: str,
) -> None:
    valid_mask = joint_valid_mask(roi_vals, target_vals)
    n_valid = int(valid_mask.sum())

    if precomp_stats:
        r_val = precomp_stats["r"]
        p_val = precomp_stats["p"]
        n_eff = precomp_stats["n"]
        ci_val = (precomp_stats.get("ci_low"), precomp_stats.get("ci_high"))
    else:
        r_val, p_val, n_eff, ci_val = compute_correlation_stats(
            roi_vals, target_vals, method_code, bootstrap_ci, rng
        )

    if n_valid > 5:
        generate_correlation_scatter(
            x_data=roi_vals,
            y_data=target_vals,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title,
            band_color=band_color,
            output_path=output_path,
            method_code=method_code,
            bootstrap_ci=0,
            rng=rng,
            roi_channels=chs,
            logger=logger,
            annotated_stats=(r_val, p_val, n_eff),
            annot_ci=ci_val,
            config=config,
        )

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
        plot_cfg = get_plot_config(config)
        behavioral_config = plot_cfg.get_behavioral_config()
        sig_thr = float(behavioral_config.get("significance_threshold", 0.05))
        if np.isfinite(p_val) and p_val < sig_thr:
            results["significant"].append(record)


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
    do_temp: bool,
    rating_stats: Optional[pd.DataFrame],
    temp_stats: Optional[pd.DataFrame],
    logger: logging.Logger,
    config: Any,
) -> Dict[str, Any]:
    results = {"significant": [], "all": []}

    if feature_type == "connectivity":
        source_df = data.conn_df
        if source_df is None or source_df.empty:
            logger.warning("No connectivity data available for scatter plots")
            return results
    else:
        source_df = data.features_df

    metric_list = metrics if metrics else [None]

    for metric in metric_list:
        for band in bands:
            band_title = band.capitalize()
            band_color = get_band_color(band, config)

            for roi, chs in data.roi_map.items():
                roi_vals, is_valid = column_extractor(source_df, band, chs, metric)
                if not is_valid:
                    continue

                roi_plots_dir = data.plots_dir / sanitize_label(roi)
                ensure_dir(roi_plots_dir)

                title_rating = title_formatter(band_title, roi, "Rating", metric)
                x_label = x_label_formatter(band_title, metric)
                output_path = roi_plots_dir / filename_formatter(band, "rating", metric)
                feature_name = feature_name_formatter(band, metric)

                precomp = _get_precomputed_stats_for_roi_band(rating_stats, roi, band, logger)

                _generate_single_scatter(
                    roi_vals=roi_vals,
                    target_vals=data.y,
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
                    method_code=method_code,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    chs=chs,
                    precomp_stats=precomp,
                    logger=logger,
                    config=config,
                    results=results,
                    feature_name=feature_name,
                )

                if do_temp and data.temp_series is not None and not data.temp_series.empty:
                    title_temp = title_formatter(band_title, roi, "Temperature", metric)
                    output_path_temp = roi_plots_dir / filename_formatter(band, "temp", metric)

                    precomp_t = _get_precomputed_stats_for_roi_band(temp_stats, roi, band, logger)

                    _generate_single_scatter(
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
                        method_code=method_code,
                        bootstrap_ci=bootstrap_ci,
                        rng=rng,
                        chs=chs,
                        precomp_stats=precomp_t,
                        logger=logger,
                        config=config,
                        results=results,
                        feature_name=feature_name,
                    )

    return results


def _get_precomputed_stats_for_roi_band(
    stats: Optional[pd.DataFrame],
    roi: str,
    band: str,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    if stats is None or stats.empty:
        return None

    try:
        roi_mask = stats.get("roi") == roi if "roi" in stats.columns else None
        band_mask = stats.get("band") == band if "band" in stats.columns else None
        if roi_mask is None or band_mask is None:
            return None

        row = stats[roi_mask & band_mask]
        if row.empty:
            return None
        r = float(row["r"].iloc[0])
        p = float(row["p"].iloc[0])
        n = int(row["n"].iloc[0]) if "n" in row.columns else int(row.get("n_eff", np.nan))
        ci_low = row["ci_low"].iloc[0] if "ci_low" in row.columns else np.nan
        ci_high = row["ci_high"].iloc[0] if "ci_high" in row.columns else np.nan
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


def _get_target_suffix(target_type: str, config=None) -> str:
    if config is None:
        raise ValueError("config is required for behavioral plotting")
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    target_rating = behavioral_config.get("target_rating", "rating")
    return "rating" if target_type == target_rating else "temp"


def _get_time_suffix(time_label: Optional[str]) -> str:
    return f"_{time_label}" if time_label else "_plateau"


def _get_base_filename(band: str, target_type: str, roi_name: str, config=None) -> str:
    band_safe = sanitize_label(band)
    target_suffix = _get_target_suffix(target_type, config)

    if roi_name == "All" or roi_name == "Overall":
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
    base = _get_base_filename(band, target_type, roi_name, config)
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
    config: Optional[Any] = None,
) -> List[str]:
    v2_cols = []
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
        v2_cols.append(str(c))
    if v2_cols:
        return v2_cols

    return [
        c
        for c in temporal_df.columns
        if infer_power_band(str(c), bands=[str(band)]) == str(band).lower()
        and str(c).endswith(f"_{time_label}")
    ]


def _plot_partial_residuals(
    x_data: pd.Series,
    y_data: pd.Series,
    Z_data: pd.DataFrame,
    method_code: str,
    band_title: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    band_color: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    stats_df: Optional[pd.Series],
    logger: logging.Logger,
    config,
    roi_channels: Optional[List[str]] = None,
) -> None:
    mask = joint_valid_mask(x_data, y_data)
    x_part = x_data.iloc[mask]
    y_part = y_data.iloc[mask]
    Z_part = Z_data.iloc[mask]

    x_res, y_res, n_res = _compute_partial_residuals(
        x_part,
        y_part,
        Z_part,
        method_code,
        logger=logger,
        context=f"Partial residuals {roi_name} {target_type} {band_title}",
    )

    if config is None:
        raise ValueError("config is required for behavioral plotting")
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    if n_res < min_samples_for_plot:
        return

    r_resid, p_resid, n_partial, ci_resid = compute_partial_residuals_stats(
        x_res, y_res, stats_df, n_res, method_code, bootstrap_ci, rng
    )

    residual_xlabel, residual_ylabel = get_residual_labels(method_code, target_type)
    title = f"Partial residuals — {band_title} vs {target_type} — {roi_name}"
    output_path = output_dir / get_output_filename(
        band_title.lower(), target_type, roi_name, "partial"
    )

    generate_correlation_scatter(
        x_data=x_res,
        y_data=y_res,
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
    y_data: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    time_label: str,
    output_dir: Path,
    method_code: str,
    Z_covars: Optional[pd.DataFrame],
    covar_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    config,
    roi_channels: Optional[List[str]],
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    target_rating = behavioral_config.get("target_rating", "rating")

    has_covariates = Z_covars is not None and not Z_covars.empty
    if has_covariates:
        annotated_stats = None
        annot_ci = None
        bootstrap_ci_for_plot = 0
    else:
        r_temporal, p_temporal, n_eff_temporal, ci_temporal = compute_correlation_stats(
            temporal_vals,
            y_data,
            method_code,
            bootstrap_ci,
            rng,
            min_samples=min_samples_for_plot,
        )
        annotated_stats = (r_temporal, p_temporal, n_eff_temporal)
        annot_ci = ci_temporal
        bootstrap_ci_for_plot = 0

    x_label = get_temporal_xlabel(time_label)
    y_label = "Rating" if target_type == target_rating else "Temperature (°C)"
    title = _get_title_components(band_title, target_type, roi_name, time_label)

    output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "scatter", time_label
    )

    generate_correlation_scatter(
        x_data=temporal_vals,
        y_data=y_data,
        x_label=x_label,
        y_label=y_label,
        title_prefix=title,
        band_color=band_color,
        output_path=output_path,
        method_code=method_code,
        Z_covars=Z_covars,
        covar_names=covar_names,
        bootstrap_ci=bootstrap_ci_for_plot,
        rng=rng,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=annotated_stats,
        annot_ci=annot_ci,
        config=config,
    )

    qc_output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "residual_qc", time_label
    )

    plot_residual_qc(
        x_data=temporal_vals,
        y_data=y_data,
        title_prefix=title,
        output_path=qc_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )


def _plot_temporal_correlations(
    temporal_df: pd.DataFrame,
    y_data: pd.Series,
    band: str,
    band_title: str,
    band_color: str,
    target_type: str,
    roi_name: str,
    output_dir: Path,
    method_code: str,
    Z_covars: Optional[pd.DataFrame],
    covar_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    config,
    roi_channels: Optional[List[str]] = None,
) -> None:
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    time_labels = behavioral_config.get("time_labels", ["early", "mid", "late"])
    for time_label in time_labels:
        temporal_cols = _get_temporal_columns(temporal_df, band, time_label, config)
        if not temporal_cols:
            continue

        temporal_vals = (
            temporal_df[temporal_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
        )

        _plot_single_temporal_correlation(
            temporal_vals,
            y_data,
            band,
            band_title,
            band_color,
            target_type,
            roi_name,
            time_label,
            output_dir,
            method_code,
            Z_covars,
            covar_names,
            bootstrap_ci,
            rng,
            logger,
            config,
            roi_channels,
        )


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
    Z_covars: Optional[pd.DataFrame],
    covar_names: Optional[List[str]],
    bootstrap_ci: int,
    rng: np.random.Generator,
    stats_df: Optional[pd.Series],
    logger: logging.Logger,
    config,
    results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    feature_name: Optional[str] = None,
    temporal_df: Optional[pd.DataFrame] = None,
    roi_channels: Optional[List[str]] = None,
) -> None:
    if config is None:
        raise ValueError("config is required for behavioral plotting")
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)

    has_covariates = Z_covars is not None and not Z_covars.empty
    if has_covariates:
        r_val, p_val, n_eff = partial_corr_xy_given_Z(
            power_vals,
            target_vals,
            Z_covars,
            method_code,
            config=config,
        )
        ci_val = (np.nan, np.nan)
        annotated_stats = (r_val, p_val, n_eff)
        annot_ci = ci_val
        bootstrap_ci_for_plot = 0
    else:
        r_val, p_val, n_eff, ci_val = compute_correlation_stats(
            power_vals,
            target_vals,
            method_code,
            bootstrap_ci,
            rng,
            min_samples=min_samples_for_plot,
        )

        r_val, p_val, n_eff, ci_val = update_stats_from_dataframe(
            stats_df, r_val, p_val, n_eff, ci_val
        )

        annotated_stats = (r_val, p_val, n_eff)
        annot_ci = ci_val
        bootstrap_ci_for_plot = 0

    x_label, y_label = get_target_labels(target_type)
    title = _get_title_components(band_title, target_type, roi_name)

    output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "scatter"
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
        Z_covars=Z_covars,
        covar_names=covar_names,
        bootstrap_ci=bootstrap_ci_for_plot,
        rng=rng,
        roi_channels=roi_channels,
        logger=logger,
        annotated_stats=annotated_stats,
        annot_ci=annot_ci,
        config=config,
    )

    qc_output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "residual_qc"
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
            Z_covars,
            covar_names,
            bootstrap_ci,
            rng,
            logger,
            config,
            roi_channels,
        )

    diagnostics_output_path = output_dir / get_output_filename(
        band, target_type, roi_name, "residual_diagnostics"
    )

    diagnostics_title = _get_title_components(
        band_title, target_type, roi_name, None
    ).replace(" (plateau)", "")
    plot_regression_residual_diagnostics(
        x_data=power_vals,
        y_data=target_vals,
        title_prefix=diagnostics_title,
        output_path=diagnostics_output_path,
        band_color=band_color,
        logger=logger,
        config=config,
    )

    if Z_covars is not None and not Z_covars.empty:
        _plot_partial_residuals(
            power_vals,
            target_vals,
            Z_covars,
            method_code,
            band_title,
            target_type,
            roi_name,
            output_dir,
            band_color,
            bootstrap_ci,
            rng,
            stats_df,
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
