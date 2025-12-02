from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.plotting.behavioral.builders import (
    generate_correlation_scatter,
    plot_residual_qc,
)
from eeg_pipeline.plotting.behavioral.scatter import _get_target_suffix
from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import (
    prepare_partial_correlation_data,
    extract_common_dataframe_columns,
    prepare_group_band_roi_data,
    prepare_topomap_correlation_data,
)
from eeg_pipeline.utils.io.general import (
    deriv_group_plots_path,
    deriv_group_stats_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    fdr_bh_reject,
    find_connectivity_features_path,
    save_fig,
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    sanitize_label,
    get_group_logger,
    get_subject_logger,
    get_default_logger as _get_default_logger,
    get_default_config as _get_default_config,
    get_residual_labels,
)
from eeg_pipeline.utils.analysis.stats import (
    bootstrap_corr_ci as _bootstrap_corr_ci,
    compute_group_corr_stats as _compute_group_corr_stats,
    partial_residuals_xy_given_Z as _partial_residuals_xy_given_Z,
    compute_band_correlations,
    compute_connectivity_correlations,
    compute_correlation_vmax,
)


###################################################################
# Group Plot Helper Functions
###################################################################


def _add_subject_dummies_if_needed(
    Z_all_vis: pd.DataFrame, partial_subj_ids: List[str]
) -> pd.DataFrame:
    if "__subject_id__" not in Z_all_vis.columns:
        dummies = pd.get_dummies(
            pd.Series(partial_subj_ids, name="__subject_id__").astype(str),
            prefix="sub",
            drop_first=True,
        )
        Z_all_vis = pd.concat([Z_all_vis.reset_index(drop=True), dummies], axis=1)
    
    return Z_all_vis


def _prepare_partial_residuals_data(
    x_lists: List,
    y_lists: List,
    Z_lists: List,
    has_Z_flags: List[bool],
    subj_order: List[str],
    pooling_strategy: str,
    subject_fixed_effects: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    if not any(has_Z_flags):
        return None, None, None
    
    partial_x: List[pd.Series] = []
    partial_y: List[pd.Series] = []
    partial_Z: List[pd.DataFrame] = []
    partial_subj_ids: List[str] = []
    
    for idx, (has_cov, Z_df, x_arr, y_arr) in enumerate(zip(has_Z_flags, Z_lists, x_lists, y_lists)):
        if not has_cov or Z_df is None:
            continue
        
        xi, yi, Zi = prepare_partial_correlation_data(x_arr, y_arr, Z_df, pooling_strategy)
        if xi is None or yi is None:
            continue
        
        partial_x.append(xi)
        partial_y.append(yi)
        partial_Z.append(Zi)
        subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
        partial_subj_ids.extend([subj_id] * len(xi))

    if not partial_Z:
        return None, None, None

    common_cols = extract_common_dataframe_columns(partial_Z)
    if common_cols:
        partial_Z = [df[common_cols] for df in partial_Z]
    
    Z_all_vis = pd.concat(partial_Z, ignore_index=True)
    x_all_partial = pd.concat(partial_x, ignore_index=True)
    y_all_partial = pd.concat(partial_y, ignore_index=True)
    
    if subject_fixed_effects:
        Z_all_vis = _add_subject_dummies_if_needed(Z_all_vis, partial_subj_ids)
    
    return Z_all_vis, x_all_partial, y_all_partial


def _get_y_label(target_type: str, pooling_strategy: str, config: Optional[Any] = None) -> str:
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    target_rating = behavioral_config.get("target_rating", "rating")
    target_temperature = behavioral_config.get("target_temperature", "temperature")
    
    label_map = {
        target_rating: {
            "pooled_trials": "Rating",
            "within_subject_centered": "Rating (centered)",
            "within_subject_zscored": "Rating (z-scored)",
        },
        target_temperature: {
            "pooled_trials": "Temperature (°C)",
            "within_subject_centered": "Temperature (°C, centered)",
            "within_subject_zscored": "Temperature (z-scored)",
        },
    }
    
    base_labels = {
        target_rating: "Rating",
        target_temperature: "Temperature",
    }
    
    return label_map.get(target_type, {}).get(pooling_strategy, base_labels.get(target_type, target_type))


def _format_band_title(band: str, freq_bands_cfg: Dict) -> str:
    freq_range = freq_bands_cfg.get(band)
    if not freq_range:
        return band.capitalize()
    
    freq_range = tuple(freq_range)
    return f"{band.capitalize()} ({freq_range[0]:g}–{freq_range[1]:g} Hz)"


def _get_output_directory_and_path(plots_dir: Path, roi: str, band: str, target_type: str, config: Optional[Any] = None) -> Tuple[Path, Path]:
    is_overall = roi == "All"
    title_roi = "Overall" if is_overall else roi
    
    if is_overall:
        out_dir = plots_dir / "overall"
        base_name = "scatter_pow_overall"
    else:
        out_dir = plots_dir / "roi_scatters" / sanitize_label(roi)
        base_name = "scatter_pow"
    
    ensure_dir(out_dir)
    target_suffix = _get_target_suffix(target_type, config)
    out_path = out_dir / f"{base_name}_{sanitize_label(band)}_vs_{target_suffix}"
    return out_dir, out_path


def _plot_partial_residuals_if_available(
    Z_all_vis: Optional[pd.DataFrame],
    x_all_partial: Optional[pd.Series],
    y_all_partial: Optional[pd.Series],
    method_code: str,
    band_title: str,
    target_type: str,
    title_roi: str,
    out_dir: Path,
    band: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
    config,
    logger: logging.Logger,
) -> None:
    if Z_all_vis is None or x_all_partial is None or y_all_partial is None:
        return
    
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)
    if len(x_all_partial) < min_samples_for_plot:
        return
    
    x_res, y_res, n_res = _partial_residuals_xy_given_Z(x_all_partial, y_all_partial, Z_all_vis, method_code)
    if n_res < min_samples_for_plot:
        return
    
    if method_code.lower() == "spearman":
        r_resid, p_resid = stats.spearmanr(x_res, y_res, nan_policy="omit")
    else:
        r_resid, p_resid = stats.pearsonr(x_res, y_res)
    ci_resid = _bootstrap_corr_ci(x_res, y_res, method_code, n_boot=bootstrap_ci, rng=rng) if bootstrap_ci > 0 else (np.nan, np.nan)
    
    residual_xlabel, residual_ylabel = get_residual_labels(method_code, target_type)
    is_overall = title_roi == "Overall"
    base_name = "scatter_pow_overall" if is_overall else "scatter_pow"
    target_suffix = _get_target_suffix(target_type, config)
    
    generate_correlation_scatter(
        x_data=x_res,
        y_data=y_res,
        x_label=residual_xlabel,
        y_label=residual_ylabel,
        title_prefix=f"Partial residuals — {band_title} vs {target_type} — {title_roi}",
        band_color=get_band_color(band, config),
        output_path=out_dir / f"{base_name}_{sanitize_label(band)}_vs_{target_suffix}_partial",
        method_code=method_code,
        bootstrap_ci=0,
        rng=rng,
        is_partial_residuals=True,
        roi_channels=None,
        logger=logger,
        annotated_stats=(r_resid, p_resid, n_res),
        annot_ci=ci_resid,
        config=config,
    )


def _save_group_stats(
    records: List[Dict[str, Any]],
    target_type: str,
    stats_dir: Path,
    config,
    logger: logging.Logger,
) -> None:
    if not records:
        return
    
    df = pd.DataFrame(records)
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha") or config.get("statistics.fdr_alpha", 0.05)
    rej, crit = fdr_bh_reject(df["p_group"].to_numpy(), alpha=fdr_alpha)
    df["fdr_reject"] = rej
    df["fdr_crit_p"] = crit
    
    target_suffix = _get_target_suffix(target_type, config)
    out_stats = stats_dir / f"group_pooled_corr_pow_roi_vs_{target_suffix}.tsv"
    df.to_csv(out_stats, sep="\t", index=False)
    logger.info("Wrote pooled ROI vs %s stats: %s", target_type, out_stats)


def _compute_group_statistics(
    x_list, y_list, method_code, pooling_strategy, cluster_bootstrap, rng
):
    return _compute_group_corr_stats(
        [np.asarray(v) for v in x_list],
        [np.asarray(v) for v in y_list],
        method_code,
        strategy=pooling_strategy,
        n_cluster_boot=cluster_bootstrap,
        rng=rng,
    )


def _create_group_plots(
    x_all, y_all, band, band_title, roi, target_type, target_suffix,
    title_roi, out_dir, out_path, base_name, y_label, tag,
    r_g, p_g, n_trials, ci95, method_code, rng, logger, config
):
    generate_correlation_scatter(
        x_data=x_all,
        y_data=y_all,
        x_label="log10(power/baseline [-5–0 s])",
        y_label=y_label,
        title_prefix=f"{band_title} power vs {target_type} — {title_roi}",
        band_color=get_band_color(band, config),
        output_path=out_path,
        method_code=method_code,
        Z_covars=None,
        covar_names=None,
        bootstrap_ci=0,
        rng=rng,
        logger=logger,
        annotated_stats=(r_g, p_g, n_trials),
        annot_ci=ci95,
        stats_tag=tag,
        config=config,
    )
    
    plot_residual_qc(
        x_data=pd.Series(x_all),
        y_data=pd.Series(y_all),
        title_prefix=f"{band_title} power vs {target_type} — {title_roi}",
        output_path=out_dir / f"residual_qc_{base_name}_{sanitize_label(band)}_vs_{target_suffix}",
        band_color=get_band_color(band, config),
        logger=logger,
        config=config,
    )


def _build_statistics_record(
    roi, band, r_g, p_g, p_pooled, n_trials, n_subj, pooling_strategy, ci95
):
    return {
        "roi": roi,
        "band": band,
        "r_pooled": r_g,
        "p_group": p_g,
        "p_trials": p_pooled,
        "n_total": n_trials,
        "n_subjects": n_subj,
        "pooling_strategy": pooling_strategy,
        "ci_low": ci95[0],
        "ci_high": ci95[1],
    }


def _process_group_target(
    x_lists: Dict,
    y_lists: Dict,
    Z_lists: Dict,
    has_Z_flags: Dict,
    subj_order: Dict,
    target_type: str,
    plots_dir: Path,
    stats_dir: Path,
    config,
    pooling_strategy: str,
    cluster_bootstrap: int,
    subject_fixed_effects: bool,
    bootstrap_ci: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    tag_map: Dict[str, str],
    freq_bands_cfg: Dict,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    method_code = behavioral_config.get("method_spearman", "spearman")
    
    for (band, roi), x_list in x_lists.items():
        y_list = y_lists.get((band, roi))
        if not y_list:
            continue

        Z_list = Z_lists.get((band, roi), [])
        has_Z_flag = has_Z_flags.get((band, roi), [])
        subj_ord = subj_order.get((band, roi), [])

        result = prepare_group_band_roi_data(
            x_list, y_list, Z_list, has_Z_flag, subj_ord,
            pooling_strategy, subject_fixed_effects
        )
        if result[0] is None:
            continue
        
        x_all, y_all, Z_all_vis, x_all_partial, y_all_partial, vis_subj_ids = result

        r_g, p_g, n_trials, n_subj, ci95, p_pooled = _compute_group_statistics(
            x_list, y_list, method_code, pooling_strategy, cluster_bootstrap, rng
        )

        tag = tag_map.get(pooling_strategy)
        band_title = _format_band_title(band, freq_bands_cfg)
        title_roi = "Overall" if roi == "All" else roi
        out_dir, out_path = _get_output_directory_and_path(plots_dir, roi, band, target_type, config)
        base_name = "scatter_pow_overall" if roi == "All" else "scatter_pow"
        target_suffix = _get_target_suffix(target_type, config)
        y_label = _get_y_label(target_type, pooling_strategy, config)

        _create_group_plots(
            x_all, y_all, band, band_title, roi, target_type, target_suffix,
            title_roi, out_dir, out_path, base_name, y_label, tag,
            r_g, p_g, n_trials, ci95, method_code, rng, logger, config
        )

        records.append(_build_statistics_record(
            roi, band, r_g, p_g, p_pooled, n_trials, n_subj, pooling_strategy, ci95
        ))

        _plot_partial_residuals_if_available(
            Z_all_vis, x_all_partial, y_all_partial, method_code, band_title,
            target_type, title_roi, out_dir, band, bootstrap_ci, rng, config, logger
        )

    return records


###################################################################
# Group Plot Functions
###################################################################


def _get_attr_safe(obj, name: str, default: Any) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _extract_scatter_inputs(scatter_inputs, do_temp: bool) -> Dict[str, Any]:
    return {
        "rating_x": _get_attr_safe(scatter_inputs, "rating_x", {}),
        "rating_y": _get_attr_safe(scatter_inputs, "rating_y", {}),
        "rating_Z": _get_attr_safe(scatter_inputs, "rating_Z", {}),
        "rating_hasZ": _get_attr_safe(scatter_inputs, "rating_hasZ", {}),
        "rating_subjects": _get_attr_safe(scatter_inputs, "rating_subjects", {}),
        "temp_x": _get_attr_safe(scatter_inputs, "temp_x", {}),
        "temp_y": _get_attr_safe(scatter_inputs, "temp_y", {}),
        "temp_Z": _get_attr_safe(scatter_inputs, "temp_Z", {}),
        "temp_hasZ": _get_attr_safe(scatter_inputs, "temp_hasZ", {}),
        "temp_subjects": _get_attr_safe(scatter_inputs, "temp_subjects", {}),
        "have_temp": bool(_get_attr_safe(scatter_inputs, "have_temp", False)) and do_temp,
    }


def plot_group_power_roi_scatter(
    scatter_inputs,
    *,
    config=None,
    pooling_strategy: str = "within_subject_centered",
    cluster_bootstrap: int = 0,
    subject_fixed_effects: bool = True,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    default_rng_seed = config.get("project.random_state", 42)
    rng = rng or np.random.default_rng(default_rng_seed)

    deriv_root = Path(config.deriv_root)
    plot_subdir = behavioral_config.get("plot_subdir", "04_behavior_correlations")
    plots_dir = deriv_group_plots_path(deriv_root, subdir=plot_subdir)
    stats_dir = deriv_group_stats_path(deriv_root)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    log_name = config.get("logging.log_file_name", "behavior_analysis.log")
    logger = get_group_logger("behavior_analysis", log_name, config=config)

    inputs = _extract_scatter_inputs(scatter_inputs, do_temp)
    rating_x = inputs["rating_x"]
    rating_y = inputs["rating_y"]
    rating_Z = inputs["rating_Z"]
    rating_hasZ = inputs["rating_hasZ"]
    rating_subjects = inputs["rating_subjects"]
    temp_x = inputs["temp_x"]
    temp_y = inputs["temp_y"]
    temp_Z = inputs["temp_Z"]
    temp_hasZ = inputs["temp_hasZ"]
    temp_subjects = inputs["temp_subjects"]
    have_temp = inputs["have_temp"]

    subject_set = set()
    for subj_lists in rating_subjects.values():
        subject_set.update(subj_lists)
    n_subjects = len(subject_set)

    logger.info(
        "Starting group pooled ROI scatters for %d subjects (strategy=%s, FE=%s)",
        n_subjects,
        pooling_strategy,
        "on" if subject_fixed_effects else "off",
    )

    allowed_pool = {"pooled_trials", "within_subject_centered", "within_subject_zscored", "fisher_by_subject"}
    if pooling_strategy not in allowed_pool:
        logger.warning(
            "Unknown pooling_strategy '%s', falling back to 'pooled_trials'",
            pooling_strategy,
        )
        pooling_strategy = "pooled_trials"

    tag_map = {
        "pooled_trials": "[Pooled]",
        "within_subject_centered": "[Centered]",
        "within_subject_zscored": "[Z-scored]",
        "fisher_by_subject": "[Fisher]",
    }

    freq_bands_cfg = config.get("time_frequency_analysis.bands", {})

    target_rating_group = behavioral_config.get("target_rating", "rating")
    rating_records = _process_group_target(
        rating_x, rating_y, rating_Z, rating_hasZ, rating_subjects,
        target_rating_group, plots_dir, stats_dir, config, pooling_strategy,
        cluster_bootstrap, subject_fixed_effects, bootstrap_ci, rng,
        logger, tag_map, freq_bands_cfg,
    )

    target_rating = behavioral_config.get("target_rating", "rating")
    _save_group_stats(rating_records, target_rating, stats_dir, config, logger)

    if not have_temp:
        return

    target_temperature_group = behavioral_config.get("target_temperature", "temperature")
    temp_records = _process_group_target(
        temp_x, temp_y, temp_Z, temp_hasZ, temp_subjects,
        target_temperature_group, plots_dir, stats_dir, config, pooling_strategy,
        cluster_bootstrap, subject_fixed_effects, bootstrap_ci, rng,
        logger, tag_map, freq_bands_cfg,
    )

    target_temperature_save = behavioral_config.get("target_temperature", "temperature")
    _save_group_stats(temp_records, target_temperature_save, stats_dir, config, logger)

