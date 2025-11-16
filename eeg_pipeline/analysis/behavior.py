import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    _build_covariate_matrices,
    _load_features_and_targets,
    _pick_first_column,
    load_epochs_for_analysis,
    extract_temperature_data,
    build_covariate_matrix,
    build_covariates_without_temp,
    extract_roi_columns,
    extract_measure_prefixes,
    extract_node_names_from_prefix,
    build_summary_map_for_prefix,
)
from eeg_pipeline.utils.io_utils import (
    deriv_features_path,
    deriv_stats_path,
    ensure_dir,
    fdr_bh_reject,
    get_subject_logger,
    sanitize_label,
    deriv_plots_path,
    parse_analysis_type_from_filename,
    parse_target_from_filename,
    parse_measure_band_from_filename,
    build_partial_covars_string,
    format_band_range,
    validate_predictor_file,
    build_file_updates_dict,
    build_predictor_column_mapping,
    build_predictor_name,
    build_connectivity_heatmap_records,
    build_meta_for_row,
    read_tsv,
    write_tsv,
)
from eeg_pipeline.utils.stats_utils import (
    apply_fdr_correction_and_save,
    should_apply_fisher_transform,
    get_pvalue_series,
    extract_pvalue_from_dataframe,
    compute_fdr_rejections_for_heatmap,
    build_correlation_matrices_for_prefix,
)
from eeg_pipeline.utils.tfr_utils import (
    validate_baseline_window,
    build_rois_from_info as _build_rois,
    get_summary_type,
)
from eeg_pipeline.utils.stats_utils import (
    _safe_float,
    bh_adjust as _bh_adjust,
    joint_valid_mask,
    prepare_aligned_data,
    get_correlation_method,
    compute_correlation,
    compute_bootstrap_ci,
    get_fdr_alpha_from_config,
    filter_significant_predictors,
    compute_fisher_transformed_mean,
    compute_partial_correlations,
    compute_permutation_pvalues,
    compute_temp_permutation_pvalues,
    compute_channel_rating_correlations,
    compute_partial_correlation_for_roi_pair,
    compute_permutation_pvalues_for_roi_pair,
    compute_temp_correlations_for_roi,
    compute_temp_correlation_for_roi_pair,
    compute_correlation_for_time_freq_bin,
    compute_cluster_masses_1d,
    compute_topomap_permutation_masses,
    compute_cluster_pvalues_1d,
)


@dataclass
class GroupScatterInputs:
    rating_x: Dict[Tuple[str, str], List[np.ndarray]]
    rating_y: Dict[Tuple[str, str], List[np.ndarray]]
    rating_Z: Dict[Tuple[str, str], List[pd.DataFrame]]
    rating_hasZ: Dict[Tuple[str, str], List[bool]]
    rating_subjects: Dict[Tuple[str, str], List[str]]
    temp_x: Dict[Tuple[str, str], List[np.ndarray]]
    temp_y: Dict[Tuple[str, str], List[np.ndarray]]
    temp_Z: Dict[Tuple[str, str], List[pd.DataFrame]]
    temp_hasZ: Dict[Tuple[str, str], List[bool]]
    temp_subjects: Dict[Tuple[str, str], List[str]]
    have_temp: bool = False


@dataclass
class CorrelationStats:
    correlation: float
    p_value: float
    ci_low: float
    ci_high: float
    r_partial: float
    p_partial: float
    n_partial: int
    r_partial_temp: float
    p_partial_temp: float
    n_partial_temp: int
    p_perm: float
    p_partial_perm: float
    p_partial_temp_perm: float


###################################################################
# Helper Functions
###################################################################


def _build_base_correlation_record(
    identifier: str,
    identifier_key: str,
    band: str,
    correlation: float,
    p_value: float,
    n_valid: int,
    method: str,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    r_partial: float = np.nan,
    p_partial: float = np.nan,
    n_partial: int = 0,
    p_perm: float = np.nan,
    p_partial_perm: float = np.nan,
    **extra_fields,
) -> Dict[str, Any]:
    record = {
        identifier_key: identifier,
        "band": band,
        "r": correlation,
        "p": p_value,
        "n": n_valid,
        "method": method,
        "r_partial": _safe_float(r_partial),
        "p_partial": _safe_float(p_partial),
        "n_partial": n_partial,
        "p_perm": _safe_float(p_perm),
        "p_partial_perm": _safe_float(p_partial_perm),
    }
    
    if identifier_key == "channel":
        record["ci_lo"] = ci_low
        record["ci_hi"] = ci_high
    else:
        record["r_ci_low"] = ci_low
        record["r_ci_high"] = ci_high
    
    record.update(extra_fields)
    return record


def _build_temp_record_unified(
    x_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    identifier: str,
    identifier_key: str,
    band: str,
    use_spearman: bool,
    method: str,
    min_samples: int,
    logger,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
    config=None,
    **extra_fields,
) -> Optional[Dict[str, Any]]:
    if temp_series is None or temp_series.empty:
        return None
    
    context_temp = f"temperature {identifier_key} {identifier} ({band})"
    x_aligned, temp_aligned, _, _, _ = prepare_aligned_data(
        x_values,
        temp_series,
        covariates_without_temp_df,
        min_samples=min_samples,
        logger=logger,
        context=context_temp,
    )
    
    if x_aligned is None or temp_aligned is None:
        return None
    
    correlation_temp, p_value_temp = compute_correlation(
        x_aligned, temp_aligned, use_spearman
    )
    
    ci_low = ci_high = np.nan
    p_perm_temp = np.nan
    
    if bootstrap > 0 and rng is not None:
        ci_low, ci_high = compute_bootstrap_ci(
            x_values, temp_series, bootstrap, use_spearman, rng,
            min_samples, logger=logger, config=config
        )
    
    if n_perm > 0 and rng is not None:
        p_perm_temp, _ = compute_temp_permutation_pvalues(
            x_values, temp_series, covariates_without_temp_df, method,
            n_perm, rng, band, identifier, logger
        )
    
    record = _build_base_correlation_record(
        identifier=identifier,
        identifier_key=identifier_key,
        band=band,
        correlation=correlation_temp,
        p_value=p_value_temp,
        n_valid=int(len(x_aligned)),
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        p_perm=p_perm_temp,
        **extra_fields,
    )
    
    return record


def _load_connectivity_data(
    subject: str,
    deriv_root: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    feats_dir = deriv_features_path(deriv_root, subject)
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"
    
    if not conn_path.exists() or not y_path.exists():
        return None, None
    
    connectivity_df = read_tsv(conn_path)
    target_df = read_tsv(y_path)
    
    if connectivity_df is None or target_df is None or target_df.empty:
        return None, None
    
    target_values = pd.to_numeric(target_df.iloc[:, 0], errors="coerce")
    
    return connectivity_df, target_values


def _initialize_analysis_context(
    subject: str,
    task: Optional[str],
    config: Any,
    logger_name: str = "behavior_analysis",
) -> Tuple[Any, str, Path, Path, Any]:
    if not subject:
        raise ValueError("Subject must be provided")
    
    if config is None:
        config = load_settings()
    
    if task is None:
        task = config.task
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger(logger_name, subject, log_name, config=config)
    
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    
    return config, task, deriv_root, stats_dir, logger



def _build_roi_pair_rating_record(
    measure_band: str,
    roi_i: str,
    roi_j: str,
    n_edges: int,
    correlation: float,
    p_value: float,
    n_eff: int,
    method: str,
    ci_low: float,
    ci_high: float,
    r_partial: float,
    p_partial: float,
    n_partial: int,
    covariates_df: Optional[pd.DataFrame],
    p_perm: float,
    p_partial_perm: float,
    n_perm: int,
) -> Dict[str, Any]:
    partial_covars_str = build_partial_covars_string(covariates_df)
    return _build_base_correlation_record(
        identifier=f"{roi_i}_{roi_j}",
        identifier_key="roi_pair",
        band=measure_band,
        correlation=correlation,
        p_value=p_value,
        n_valid=n_eff,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_partial,
        p_partial=p_partial,
        n_partial=n_partial,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        measure_band=measure_band,
        roi_i=roi_i,
        roi_j=roi_j,
        summary_type=get_summary_type(roi_i, roi_j),
        n_edges=n_edges,
        partial_covars=partial_covars_str,
        n_perm=n_perm,
    )


###################################################################
# Power ROI Statistics
###################################################################


def _build_roi_rating_record(
    roi: str,
    band: str,
    band_range_str: str,
    correlation: float,
    p_value: float,
    n_eff: int,
    method: str,
    ci_low: float,
    ci_high: float,
    r_part: float,
    p_part: float,
    n_part: int,
    partial_covars_str: str,
    r_part_temp: float,
    p_part_temp: float,
    n_part_temp: int,
    p_perm: float,
    p_partial_perm: float,
    p_partial_temp_perm: float,
    n_perm: int,
) -> Dict[str, Any]:
    return _build_base_correlation_record(
        identifier=roi,
        identifier_key="roi",
        band=band,
        correlation=correlation,
        p_value=p_value,
        n_valid=n_eff,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_part,
        p_partial=p_part,
        n_partial=n_part,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        band_range=band_range_str,
        partial_covars=partial_covars_str,
        r_partial_given_temp=_safe_float(r_part_temp),
        p_partial_given_temp=_safe_float(p_part_temp),
        n_partial_given_temp=n_part_temp,
        p_partial_given_temp_perm=_safe_float(p_partial_temp_perm),
        n_perm=n_perm,
    )


def _build_channel_rating_record(
    channel: str,
    band: str,
    correlation: float,
    p_value: float,
    ci_low: float,
    ci_high: float,
    r_partial: float,
    p_partial: float,
    n_partial: int,
    p_perm: float,
    p_partial_perm: float,
    n_valid: int,
    method: str,
) -> Dict[str, Any]:
    return _build_base_correlation_record(
        identifier=channel,
        identifier_key="channel",
        band=band,
        correlation=correlation,
        p_value=p_value,
        n_valid=n_valid,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_partial,
        p_partial=p_partial,
        n_partial=n_partial,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
    )


def _process_single_channel_for_band(
    column_name: str,
    band: str,
    power_df: pd.DataFrame,
    target_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    logger,
    config=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    channel_name = column_name.replace(f"pow_{band}_", "")
    channel_values = pd.to_numeric(power_df[column_name], errors="coerce")
    context = f"channel {channel_name} ({band})"
    
    x_aligned, y_aligned, covariates_aligned, _, _ = prepare_aligned_data(
        channel_values,
        target_values,
        covariates_df,
        min_samples=config.get("behavior_analysis.statistics.min_samples_channel", 10),
        logger=logger,
        context=context,
    )
    
    if x_aligned is None or y_aligned is None:
        return None, None
    
    (
        correlation,
        p_value,
        ci_low,
        ci_high,
        r_partial,
        p_partial,
        n_partial,
        p_perm,
        p_partial_perm,
    ) = compute_channel_rating_correlations(
        x_aligned,
        y_aligned,
        covariates_aligned,
        bootstrap,
        n_perm,
        use_spearman,
        method,
        rng,
        logger=logger,
        config=config,
    )
    
    rating_record = _build_channel_rating_record(
        channel_name,
        band,
        correlation,
        p_value,
        ci_low,
        ci_high,
        r_partial,
        p_partial,
        n_partial,
        p_perm,
        p_partial_perm,
        int(len(x_aligned)),
        method,
    )
    
    temp_record = _build_temp_record_for_channel(
        channel_name,
        band,
        channel_values,
        temp_series,
        covariates_without_temp_df,
        use_spearman,
        method,
        logger,
        config=config,
    )
    
    return rating_record, temp_record


def _build_temp_record_for_channel(
    channel_name: str,
    band: str,
    channel_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    use_spearman: bool,
    method: str,
    logger,
    config=None,
) -> Optional[Dict[str, Any]]:
    if config is None:
        config = load_settings()
    return _build_temp_record_unified(
        x_values=channel_values,
        temp_series=temp_series,
        covariates_without_temp_df=covariates_without_temp_df,
        identifier=channel_name,
        identifier_key="channel",
        band=band,
        use_spearman=use_spearman,
        method=method,
        min_samples=config.get("behavior_analysis.statistics.min_samples_channel", 10),
        logger=logger,
        config=config,
    )


def _process_channel_level_correlations(
    power_bands: List[str],
    power_df: pd.DataFrame,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    stats_dir: Path,
    config,
    logger,
) -> None:
    for band in power_bands:
        band_columns = [
            col for col in power_df.columns if col.startswith(f"pow_{band}_")
        ]
        if not band_columns:
            logger.debug(f"No channel-level power columns for band '{band}'")
            continue

        rating_records: List[Dict[str, Any]] = []
        temp_records: List[Dict[str, Any]] = []

        for column_name in band_columns:
            rating_record, temp_record = _process_single_channel_for_band(
                column_name,
                band,
                power_df,
                target_values,
                temp_series,
                covariates_df,
                covariates_without_temp_df,
                bootstrap,
                n_perm,
                use_spearman,
                method,
                rng,
                logger,
                config=config,
            )
            if rating_record is not None:
                rating_records.append(rating_record)
            if temp_record is not None:
                temp_records.append(temp_record)

        if rating_records:
            rating_df = pd.DataFrame(rating_records)
            apply_fdr_correction_and_save(
                rating_df,
                stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv",
                config,
                logger,
            )

        if temp_records:
            temp_df = pd.DataFrame(temp_records)
            apply_fdr_correction_and_save(
                temp_df,
                stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv",
                config,
                logger,
                use_permutation_p=False,
            )


def _load_analysis_data(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Any,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[mne.Info],
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[str],
]:
    if not subject or not task:
        logger.error("Subject and task must be provided")
        return None, None, None, None, None, None
    
    _, power_df, _, target_values, info = _load_features_and_targets(
        subject, task, deriv_root, config
    )
    target_values = pd.to_numeric(target_values, errors="coerce")

    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )
    if epochs is None:
        logger.error(f"Could not find epochs for ROI correlations: sub-{subject}")
        return None, None, None, None, None, None

    temp_series, temp_col = extract_temperature_data(aligned_events, config)

    return power_df, target_values, info, aligned_events, temp_series, temp_col


def _compute_correlation_statistics(
    x_values: pd.Series,
    y_values: pd.Series,
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    covariates_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    n_eff: int,
    min_samples: int,
    logger,
    config=None,
    temp_series: Optional[pd.Series] = None,
    context: str = "",
    band: str = "",
    identifier: str = "",
) -> CorrelationStats:
    correlation, p_value = compute_correlation(x_aligned, y_aligned, use_spearman)
    
    (
        r_partial,
        p_partial,
        n_partial,
        r_partial_temp,
        p_partial_temp,
        n_partial_temp,
    ) = compute_partial_correlations(
        x_values, y_values, covariates_df, temp_series, method, context, logger, min_samples
    )
    
    ci_low, ci_high = compute_bootstrap_ci(
        x_values, y_values, bootstrap, use_spearman, rng,
        min_samples, logger=logger, config=config
    )
    
    p_perm, p_partial_perm, p_partial_temp_perm = compute_permutation_pvalues(
        x_aligned, y_aligned, covariates_df, temp_series, method,
        n_perm, n_eff, rng, band, identifier,
    )
    
    return CorrelationStats(
        correlation=correlation,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_partial,
        p_partial=p_partial,
        n_partial=n_partial,
        r_partial_temp=r_partial_temp,
        p_partial_temp=p_partial_temp,
        n_partial_temp=n_partial_temp,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        p_partial_temp_perm=p_partial_temp_perm,
    )


def _compute_roi_correlation_stats(
    roi_values: pd.Series,
    target_values: pd.Series,
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    n_eff: int,
    band: str,
    roi: str,
    context: str,
    logger,
    config=None,
) -> CorrelationStats:
    return _compute_correlation_statistics(
        roi_values, target_values, x_aligned, y_aligned,
        covariates_df, bootstrap, n_perm, use_spearman, method,
        rng, n_eff, config.get("behavior_analysis.statistics.min_samples_roi", 5), logger, config,
        temp_series, context, band, roi
    )


def _process_single_roi_for_band(
    roi: str,
    channels: List[str],
    band: str,
    band_columns: set,
    power_df: pd.DataFrame,
    target_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    band_range_str: str,
    logger,
    config=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if power_df is None or target_values is None or power_df.empty or target_values.empty:
        return None, None
    
    roi_columns = extract_roi_columns(roi, channels, band, band_columns)
    if roi_columns is None:
        return None, None

    roi_values = power_df[roi_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    context = f"ROI {roi} ({band})"
    x_aligned, y_aligned, _, _, n_eff = prepare_aligned_data(
        roi_values, target_values, None,
        min_samples=config.get("behavior_analysis.statistics.min_samples_roi", 5), logger=logger, context=context,
    )
    
    if x_aligned is None or y_aligned is None:
        return None, None

    stats = _compute_roi_correlation_stats(
        roi_values, target_values, x_aligned, y_aligned,
        covariates_df, temp_series, bootstrap, n_perm, use_spearman,
        method, rng, n_eff, band, roi, context, logger, config=config,
    )

    partial_covars_str = build_partial_covars_string(covariates_df)

    rating_record = _build_roi_rating_record(
        roi, band, band_range_str, stats.correlation, stats.p_value, n_eff, method,
        stats.ci_low, stats.ci_high, stats.r_partial, stats.p_partial, stats.n_partial, partial_covars_str,
        stats.r_partial_temp, stats.p_partial_temp, stats.n_partial_temp,
        stats.p_perm, stats.p_partial_perm, stats.p_partial_temp_perm, n_perm,
    )

    temp_record = _build_temp_record_for_roi(
        roi_values, temp_series, covariates_without_temp_df,
        bootstrap, n_perm, use_spearman, method, rng,
        band, roi, band_range_str, logger, config=config,
    )

    return rating_record, temp_record


def _build_temp_record_for_roi(
    roi_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    band: str,
    roi: str,
    band_range_str: str,
    logger,
    config=None,
) -> Optional[Dict[str, Any]]:
    if bootstrap > 0 or n_perm > 0:
        temp_record = compute_temp_correlations_for_roi(
            roi_values,
            temp_series,
            covariates_without_temp_df,
            bootstrap,
            n_perm,
            use_spearman,
            method,
            rng,
            band,
            roi,
            logger,
            config=config,
        )
        if temp_record is not None:
            temp_record["band_range"] = band_range_str
        return temp_record
    
    temp_record = _build_temp_record_unified(
        x_values=roi_values,
        temp_series=temp_series,
        covariates_without_temp_df=covariates_without_temp_df,
        identifier=roi,
        identifier_key="roi",
        band=band,
        use_spearman=use_spearman,
        method=method,
        min_samples=config.get("behavior_analysis.statistics.min_samples_roi", 5),
        logger=logger,
        bootstrap=bootstrap,
        n_perm=n_perm,
        rng=rng,
        band_range=band_range_str,
    )
    
    return temp_record


def _save_roi_results(
    recs_rating: List[Dict[str, Any]],
    recs_temp: List[Dict[str, Any]],
    stats_dir: Path,
    config,
    logger,
) -> None:
    if recs_rating:
        df_rating = pd.DataFrame(recs_rating)
        apply_fdr_correction_and_save(
            df_rating,
            stats_dir / "corr_stats_pow_roi_vs_rating.tsv",
            config,
            logger,
        )
        write_tsv(df_rating, stats_dir / "corr_stats_pow_combined_vs_rating.tsv")

    if recs_temp:
        df_temp = pd.DataFrame(recs_temp)
        apply_fdr_correction_and_save(
            df_temp, stats_dir / "corr_stats_pow_roi_vs_temp.tsv", config, logger
        )
        write_tsv(df_temp, stats_dir / "corr_stats_pow_combined_vs_temp.tsv")
        write_tsv(df_temp, stats_dir / "corr_stats_pow_combined_vs_temperature.tsv")


def compute_power_roi_stats(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    if not subject:
        return
    
    config = load_settings()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ROI power correlation analysis for sub-{subject}")

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    if task is None:
        task = config.task

    rng_seed = config.get("random.seed", 42)
    rng = rng or np.random.default_rng(rng_seed)

    pow_df, y, info, aligned_events, temp_series, temp_col = _load_analysis_data(
        subject, task, deriv_root, config, logger
    )
    if pow_df is None or y is None:
        return

    roi_map = _build_rois(info, config=config)
    if not roi_map:
        logger.warning(f"No ROI definitions found; skipping ROI stats for sub-{subject}")
        roi_map = {}

    covariates_df = build_covariate_matrix(aligned_events, partial_covars, config)
    covariates_without_temp_df = build_covariates_without_temp(covariates_df, temp_col)

    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    freq_bands = config.get("time_frequency_analysis.bands", {})

    rating_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []
    missing_roi_by_band: Dict[str, List[str]] = defaultdict(list)
    method = get_correlation_method(use_spearman)

    for band in power_bands:
        band_columns = {col for col in pow_df.columns if col.startswith(f"pow_{band}_")}
        if not band_columns:
            continue

        band_range_str = format_band_range(band, freq_bands)

        for roi, channels in roi_map.items():
            rating_record, temp_record = _process_single_roi_for_band(
                roi, channels, band, band_columns, pow_df, y, temp_series,
                covariates_df, covariates_without_temp_df, bootstrap, n_perm, use_spearman,
                method, rng, band_range_str, logger, config=config
            )
            if rating_record is None:
                missing_roi_by_band[band].append(roi)
                continue
            rating_records.append(rating_record)
            if temp_record is not None:
                temp_records.append(temp_record)

    if missing_roi_by_band:
        detail_str = "; ".join(
            f"{band}: {', '.join(sorted(set(rois)))}" for band, rois in missing_roi_by_band.items()
        )
        logger.warning(
            "Skipped ROI statistics for missing channel groups (band -> ROI list): %s",
            detail_str,
        )

    _save_roi_results(rating_records, temp_records, stats_dir, config, logger)

    _process_channel_level_correlations(
        power_bands,
        pow_df,
        y,
        covariates_df,
        temp_series,
        covariates_without_temp_df,
        bootstrap,
        n_perm,
        use_spearman,
        method,
        rng,
        stats_dir,
        config,
        logger,
    )


###################################################################
# Time-Frequency Correlations
###################################################################
def _restrict_epochs_to_roi(
    epochs: mne.Epochs,
    roi_selection: Optional[str],
    config,
    logger,
) -> mne.Epochs:
    if roi_selection is None:
        return epochs
    
    roi_map = _build_rois(epochs.info, config=config)
    if roi_selection not in roi_map:
        logger.warning(f"ROI '{roi_selection}' not found; using all channels")
        return epochs
    
    channels = roi_map[roi_selection]
    epochs_restricted = epochs.pick_channels(channels)
    logger.info(f"Restricted TF computation to ROI '{roi_selection}' ({len(channels)} channels)")
    return epochs_restricted


def _apply_baseline_to_tfr(
    tfr, config, logger
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    baseline_applied = False
    baseline_window_used = None
    baseline_window = config.get(
        "time_frequency_analysis.baseline_window", [-5.0, -0.01]
    )
    min_samples_roi = config.get("behavior_analysis.statistics.min_samples_roi", 5)
    min_baseline_samples = int(
        config.get("time_frequency_analysis.min_baseline_samples", min_samples_roi)
    )
    
    try:
        b_start, b_end, _ = validate_baseline_window(
            tfr.times,
            tuple(baseline_window),
            min_samples=min_baseline_samples,
        )
        tfr.apply_baseline(baseline=(b_start, b_end), mode="logratio")
        baseline_applied = True
        baseline_window_used = (b_start, b_end)
    except (ValueError, RuntimeError) as err:
        logger.error(
            f"Baseline validation failed ({err}); raising error"
        )
        raise
    
    return baseline_applied, baseline_window_used


def _compute_tf_correlations_for_bins(
    power: np.ndarray,
    y_array: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    time_bin_edges: np.ndarray,
    min_valid_points: int,
    use_spearman: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    n_time_bins = len(time_bin_edges) - 1
    correlations = np.full((len(freqs), n_time_bins), np.nan)
    p_values = np.full_like(correlations, np.nan)
    n_valid = np.zeros_like(correlations, dtype=int)
    bin_data = np.full((len(freqs), n_time_bins, len(y_array)), np.nan, dtype=float)
    informative_bins: List[Tuple[int, int]] = []

    for f_idx, freq in enumerate(freqs):
        for t_idx in range(n_time_bins):
            t_start, t_end = time_bin_edges[t_idx], time_bin_edges[t_idx + 1]
            time_mask = (times >= t_start) & (times < t_end)
            if np.any(time_mask):
                vals = power[:, f_idx, time_mask].mean(axis=1)
                bin_data[f_idx, t_idx, :] = vals
            
            correlation, p_value, n_obs = compute_correlation_for_time_freq_bin(
                power,
                y_array,
                times,
                f_idx,
                t_start,
                t_end,
                min_valid_points,
                use_spearman,
            )
            
            n_valid[f_idx, t_idx] = n_obs
            if correlation is not None and p_value is not None:
                correlations[f_idx, t_idx] = correlation
                p_values[f_idx, t_idx] = p_value
                informative_bins.append((f_idx, t_idx))

    return correlations, p_values, n_valid, bin_data, informative_bins


###################################################################
# Power Topomap Correlations
###################################################################


def _build_channel_record(
    band: str,
    channel: str,
    correlation: float,
    p_value: float,
    p_corrected: float,
    significant: bool,
    cluster_id: int,
    cluster_p: float,
    cluster_significant: bool,
    n_valid: int,
    method: str,
) -> Dict[str, Any]:
    return {
        "band": band,
        "channel": channel,
        "correlation": _safe_float(correlation),
        "p_value": _safe_float(p_value),
        "p_corrected": _safe_float(p_corrected),
        "significant": bool(significant),
        "cluster_id": int(cluster_id),
        "cluster_p": _safe_float(cluster_p),
        "cluster_significant": bool(cluster_significant),
        "n_valid": int(n_valid),
        "method": method,
    }


def _append_channel_records_with_clusters(
    records: List[Dict[str, Any]],
    band: str,
    ch_names: List[str],
    correlations: np.ndarray,
    p_values: np.ndarray,
    p_corrected: np.ndarray,
    significant_mask: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_pvals: np.ndarray,
    cluster_sig_mask: np.ndarray,
    n_valid: np.ndarray,
    method: str,
    n_channels: int,
) -> None:
    for ch_idx in range(n_channels):
        records.append(_build_channel_record(
            band, ch_names[ch_idx], correlations[ch_idx],
            p_values[ch_idx], p_corrected[ch_idx], significant_mask[ch_idx],
            cluster_labels[ch_idx], cluster_pvals[ch_idx], cluster_sig_mask[ch_idx],
            n_valid[ch_idx], method
        ))


def correlate_power_topomaps(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    if not subject:
        return
    
    config = load_settings()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Computing power topomap correlations with temperature for sub-{subject}")

    if task is None:
        task = config.task

    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    rng_seed = config.get("random.seed", 42)
    rng = rng or np.random.default_rng(rng_seed)

    _, pow_df, _, y, info = _load_features_and_targets(
        subject, task, deriv_root, config
    )

    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )
    if epochs is None:
        logger.warning("Could not load epochs; skipping topomap correlations")
        return

    temp_series: Optional[pd.Series] = None
    if aligned_events is not None:
        psych_temp_columns = config.get("event_columns.temperature", [])
        temp_col = _pick_first_column(aligned_events, psych_temp_columns)
        if temp_col is not None:
            temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")

    if temp_series is None or temp_series.isna().all():
        logger.warning("No temperature data available; skipping topomap correlations")
        return

    from eeg_pipeline.utils.stats_utils import get_eeg_adjacency

    adjacency, eeg_picks, _ = get_eeg_adjacency(info)
    if adjacency is None or eeg_picks is None:
        logger.warning("Could not compute EEG adjacency; skipping cluster-corrected topomap correlations")
        return

    power_bands = config.get("features.frequency_bands", ["delta", "theta", "alpha", "beta", "gamma"])
    alpha = get_fdr_alpha_from_config(config)
    
    cluster_cfg = config.get("behavior_analysis.cluster_correction", {})
    cluster_alpha = float(cluster_cfg.get("alpha", alpha))
    cluster_n_perm_config = cluster_cfg.get("n_permutations", 500)
    n_cluster_perm = int(cluster_n_perm_config) if cluster_n_perm_config > 0 else 0
    cluster_rng_seed = int(cluster_cfg.get("rng_seed", config.get("random.seed", 42)))
    cluster_rng = np.random.default_rng(cluster_rng_seed)

    method = get_correlation_method(use_spearman)
    min_valid_points = config.get("behavior_analysis.statistics.min_samples_roi", 5)

    all_records: List[Dict[str, Any]] = []

    for band in power_bands:
        band_cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
        if not band_cols:
            logger.debug(f"No channel-level power columns for band '{band}'")
            continue

        ch_names = [col.replace(f"pow_{band}_", "") for col in band_cols]
        n_channels = len(ch_names)

        correlations = np.full(n_channels, np.nan)
        p_values = np.full(n_channels, np.nan)
        n_valid = np.zeros(n_channels, dtype=int)

        channel_data = np.full((n_channels, len(pow_df)), np.nan, dtype=float)

        for ch_idx, col in enumerate(band_cols):
            series = pd.to_numeric(pow_df[col], errors="coerce")
            channel_data[ch_idx, :] = series.values

            valid_mask = series.notna() & temp_series.notna()
            n_obs = int(valid_mask.sum())
            n_valid[ch_idx] = n_obs

            if n_obs < min_valid_points:
                continue

            x_valid = series[valid_mask]
            temp_valid = temp_series[valid_mask]

            correlation, p_value = compute_correlation(x_valid, temp_valid, use_spearman)
            correlations[ch_idx] = correlation
            p_values[ch_idx] = p_value

        valid_mask = np.isfinite(p_values) & (n_valid >= min_valid_points)
        if not np.any(valid_mask):
            logger.debug(f"No valid correlations for band '{band}'")
            continue

        p_corrected = np.full_like(p_values, np.nan)
        if np.any(valid_mask):
            p_corrected[valid_mask] = _bh_adjust(p_values[valid_mask])
        significant_mask = (p_corrected < alpha) & valid_mask

        cluster_labels = np.zeros(n_channels, dtype=int)
        cluster_pvals = np.full(n_channels, np.nan)
        cluster_sig_mask = np.zeros(n_channels, dtype=bool)
        cluster_records: List[Dict[str, Any]] = []
        perm_max_masses: List[float] = []

        eeg_ch_names = [info['ch_names'][i] for i in eeg_picks]
        ch_to_eeg_idx = {
            ch_idx: eeg_ch_names.index(ch_name)
            for ch_idx, ch_name in enumerate(ch_names)
            if ch_name in eeg_ch_names
        }

        min_channels_for_adjacency = config.get("behavior_analysis.statistics.min_channels_for_adjacency", 2)
        if len(ch_to_eeg_idx) < min_channels_for_adjacency:
            logger.debug(f"Insufficient channels with adjacency for band '{band}'")
            cluster_labels_empty = np.zeros(n_channels, dtype=int)
            cluster_pvals_empty = np.full(n_channels, np.nan)
            cluster_sig_mask_empty = np.zeros(n_channels, dtype=bool)
            _append_channel_records_with_clusters(
                all_records, band, ch_names, correlations, p_values,
                p_corrected, significant_mask, cluster_labels_empty,
                cluster_pvals_empty, cluster_sig_mask_empty, n_valid, method, n_channels
            )
            continue

        cluster_labels_obs, cluster_masses = compute_cluster_masses_1d(
            correlations,
            p_values,
            cluster_alpha,
            ch_to_eeg_idx,
            eeg_picks,
            adjacency,
        )
        if cluster_masses:
            cluster_labels = cluster_labels_obs
            perm_max_masses = compute_topomap_permutation_masses(
                channel_data,
                temp_series,
                n_channels,
                n_cluster_perm,
                cluster_alpha,
                min_valid_points,
                use_spearman,
                ch_to_eeg_idx,
                eeg_picks,
                adjacency,
                cluster_rng,
            )
            cluster_pvals, cluster_sig_mask, cluster_records = (
                compute_cluster_pvalues_1d(
                    cluster_labels_obs, cluster_masses, perm_max_masses, alpha
                )
            )

        _append_channel_records_with_clusters(
            all_records, band, ch_names, correlations, p_values,
            p_corrected, significant_mask, cluster_labels, cluster_pvals,
            cluster_sig_mask, n_valid, method, n_channels
        )

        if cluster_records:
            n_significant = sum(
                1 for r in cluster_records if r["p_value"] <= alpha
            )
            logger.info(
                f"Band {band}: {len(cluster_records)} clusters, "
                f"{n_significant} significant"
            )

    if not all_records:
        logger.warning("No topomap correlation records generated")
        return

    df_results = pd.DataFrame(all_records)
    method_suffix = "_spearman" if use_spearman else "_pearson"
    stats_file = stats_dir / f"power_topomap_temperature_correlations{method_suffix}.tsv"
    write_tsv(df_results, stats_file)
    logger.info(f"Saved {len(df_results)} topomap temperature correlation records to {stats_file}")

    cluster_summary = {
        "subject": subject,
        "task": task,
        "method": method,
        "n_bands": len(power_bands),
        "n_channels_total": len(set(df_results["channel"])),
        "alpha_bh": float(alpha),
        "cluster_alpha": float(cluster_alpha),
        "n_cluster_permutations": int(n_cluster_perm),
        "cluster_rng_seed": int(cluster_rng_seed),
        "n_bh_significant_channels": int(df_results["significant"].sum()),
        "n_cluster_significant_channels": int(df_results["cluster_significant"].sum()),
    }
    meta_path = stats_dir / f"power_topomap_temperature_meta{method_suffix}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(cluster_summary, fh, indent=2)
    logger.info(f"Saved topomap temperature correlation metadata to {meta_path}")


###################################################################
# Connectivity ROI Summary Correlations
###################################################################

def correlate_connectivity_roi_summaries(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None
) -> None:
    try:
        config, task, deriv_root, stats_dir, logger = _initialize_analysis_context(
            subject, task, None
        )
    except ValueError:
        return

    rng_seed = config.get("random.seed", 42)
    rng = rng or np.random.default_rng(rng_seed)

    X, y = _load_connectivity_data(subject, deriv_root)
    if X is None or y is None:
        return

    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )
    if epochs is None:
        return
    info = epochs.info
    roi_map = _build_rois(info, config=config)
    
    temp_series, temp_col = extract_temperature_data(aligned_events, config)

    covariates_df = build_covariate_matrix(aligned_events, partial_covars, config)
    covariates_without_temp_df = build_covariates_without_temp(covariates_df, temp_col)

    prefixes = extract_measure_prefixes(X.columns)
    
    for prefix in prefixes:
        prefix_columns = [c for c in X.columns if c.startswith(prefix + "_")]
        if not prefix_columns:
            continue
        
        summary_map = build_summary_map_for_prefix(
            prefix, prefix_columns, roi_map
        )
        if not summary_map:
            continue
        
        apply_fisher_transform = should_apply_fisher_transform(prefix)

        recs: List[Dict[str, object]] = []
        recs_temp: List[Dict[str, object]] = []
        method = get_correlation_method(use_spearman)
        
        for (roi_i, roi_j), cols_list in summary_map.items():
            edge_df = X[cols_list].apply(pd.to_numeric, errors="coerce")
            xi = (
                compute_fisher_transformed_mean(edge_df)
                if apply_fisher_transform
                else edge_df.mean(axis=1)
            )
            
            mask = joint_valid_mask(xi, y)
            n_eff = int(mask.sum())
            min_samples_roi = config.get("behavior_analysis.statistics.min_samples_roi", 5)
            if n_eff < min_samples_roi:
                continue
            
            xi_masked = xi.iloc[mask]
            y_masked = y.iloc[mask]
            
            correlation, p_value = compute_correlation(xi_masked, y_masked, use_spearman)
            
            r_partial, p_partial, n_partial = compute_partial_correlation_for_roi_pair(
                xi_masked, y_masked, covariates_df, mask, method
            )

            ci_low, ci_high = compute_bootstrap_ci(
                xi, y, bootstrap, use_spearman, rng,
                min_samples_roi, logger=logger, config=config
            )

            p_perm, p_partial_perm = compute_permutation_pvalues_for_roi_pair(
                xi_masked, y_masked, covariates_df, mask, method, n_perm, n_eff, rng,
            )

            rating_record = _build_roi_pair_rating_record(
                prefix,
                roi_i,
                roi_j,
                len(cols_list),
                correlation,
                p_value,
                n_eff,
                method,
                ci_low,
                ci_high,
                r_partial,
                p_partial,
                n_partial,
                covariates_df,
                p_perm,
                p_partial_perm,
                n_perm,
            )
            recs.append(rating_record)

            if temp_series is not None and not temp_series.empty:
                temp_record = compute_temp_correlation_for_roi_pair(
                    xi, temp_series, covariates_without_temp_df,
                    bootstrap, n_perm, use_spearman, prefix,
                    roi_i, roi_j, len(cols_list), rng, logger, config=config,
                )
                if temp_record is not None:
                    recs_temp.append(temp_record)

        if recs:
            df = pd.DataFrame(recs)
            apply_fdr_correction_and_save(
                df,
                stats_dir / f"corr_stats_conn_roi_summary_{sanitize_label(prefix)}_vs_rating.tsv",
                config,
                logger,
            )

        if recs_temp:
            df_t = pd.DataFrame(recs_temp)
            apply_fdr_correction_and_save(
                df_t,
                stats_dir / f"corr_stats_conn_roi_summary_{sanitize_label(prefix)}_vs_temp.tsv",
                config,
                logger,
            )


###################################################################
# Connectivity Heatmap Correlations
###################################################################


def correlate_connectivity_heatmaps(subject: str, task: Optional[str] = None, use_spearman: bool = True) -> None:
    try:
        config, task, deriv_root, stats_dir, logger = _initialize_analysis_context(
            subject, task, None
        )
    except ValueError:
        return
    
    logger.info(f"Starting connectivity correlation analysis for sub-{subject}")
    plot_subdir = config.get("plotting.behavioral.plot_subdir", "04_behavior_correlations")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    connectivity_dataframe, target_values = _load_connectivity_data(subject, deriv_root)
    if connectivity_dataframe is None or target_values is None:
        logger.warning(
            f"Connectivity features or targets missing for sub-{subject}; "
            f"skipping connectivity correlations."
        )
        return

    column_names = list(connectivity_dataframe.columns)
    measure_prefixes = sorted({"_".join(col.split("_")[:2]) for col in column_names})

    for prefix in measure_prefixes:
        _process_connectivity_prefix(
            prefix,
            column_names,
            connectivity_dataframe,
            target_values,
            use_spearman,
            stats_dir,
            config,
            logger,
        )


def _process_connectivity_prefix(
    prefix: str,
    column_names: List[str],
    connectivity_dataframe: pd.DataFrame,
    target_values: pd.Series,
    use_spearman: bool,
    stats_dir: Path,
    config: Any,
    logger: Any,
) -> None:
    prefix_columns = [col for col in column_names if col.startswith(prefix + "_")]
    if not prefix_columns:
        return
    
    node_info = extract_node_names_from_prefix(prefix, prefix_columns)
    if node_info is None:
        logger.warning(f"Could not infer nodes for {prefix}; skipping heatmap.")
        return
    
    node_names, node_to_index = node_info
    
    correlation_matrix, p_value_matrix = build_correlation_matrices_for_prefix(
        prefix,
        prefix_columns,
        connectivity_dataframe,
        target_values,
        node_to_index,
        use_spearman,
    )
    
    rejection_map, critical_value = compute_fdr_rejections_for_heatmap(
        p_value_matrix, len(node_names), config
    )

    records = build_connectivity_heatmap_records(
        len(node_names),
        node_names,
        correlation_matrix,
        p_value_matrix,
        rejection_map,
        critical_value,
    )
    
    if records:
        results_dataframe = pd.DataFrame(records)
        output_path = stats_dir / f"corr_stats_edges_{sanitize_label(prefix)}_vs_rating.tsv"
        write_tsv(results_dataframe, output_path)
        logger.info(f"Saved connectivity heatmap correlations for {prefix}")


def _apply_fdr_updates_to_files(
    file_updates: Dict[Path, List[Tuple[int, float, bool, float]]],
    critical_p: float,
    stats_dir: Path,
    logger: Any,
) -> None:
    for file_path, update_items in file_updates.items():
        try:
            dataframe = read_tsv(file_path)
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        
        if dataframe is None or dataframe.empty:
            continue
        
        n_rows = len(dataframe)
        q_column = np.full(n_rows, np.nan, dtype=float)
        rejection_column = np.zeros(n_rows, dtype=bool)
        p_used_column = np.full(n_rows, np.nan, dtype=float)
        
        for row_index, q_value, is_rejected, p_used in update_items:
            row_idx = int(row_index)
            if 0 <= row_idx < n_rows:
                q_column[row_idx] = q_value
                rejection_column[row_idx] = is_rejected
                p_used_column[row_idx] = p_used
        
        dataframe["p_used_for_global_fdr"] = p_used_column
        dataframe["q_fdr_global"] = q_column
        dataframe["fdr_reject_global"] = rejection_column
        dataframe["fdr_crit_p_global"] = _safe_float(critical_p)
        
        try:
            write_tsv(dataframe, file_path)
            logger.info(f"Updated {file_path.name} with global FDR correction")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to update {file_path.name} with global FDR correction: {e}")


###################################################################
# Export Functions
###################################################################

def _add_fdr_columns_to_mapping(cols: Dict[str, str], df: pd.DataFrame) -> Dict[str, str]:
    if "fdr_reject" in df.columns:
        cols["fdr_reject"] = "fdr_significant"
    if "fdr_crit_p" in df.columns:
        cols["fdr_crit_p"] = "fdr_critical_p"
    return cols


def _process_predictor_file(
    file_path: Path,
    target: str,
    predictor_type: str,
    use_fdr: bool,
    alpha: float,
    logger,
) -> Optional[pd.DataFrame]:
    if not file_path.exists():
        return None

    dataframe = read_tsv(file_path)
    if not validate_predictor_file(dataframe, predictor_type, target, logger):
        return None

    significant_predictors = filter_significant_predictors(dataframe, use_fdr, alpha)
    if len(significant_predictors) == 0:
        return None

    significant_predictors["predictor_type"] = predictor_type
    significant_predictors["target"] = target
    significant_predictors["predictor"] = build_predictor_name(
        significant_predictors, predictor_type
    )
    
    column_mapping = build_predictor_column_mapping(predictor_type)
    column_mapping = _add_fdr_columns_to_mapping(column_mapping, significant_predictors)
    result_subset = significant_predictors[list(column_mapping.keys())].rename(
        columns=column_mapping
    )
    logger.info(
        f"Found {len(significant_predictors)} significant {predictor_type} "
        f"predictors for target '{target}'"
    )
    return result_subset


def export_all_significant_predictors(
    subject: str, alpha: float = None, use_fdr: bool = True
) -> None:
    if not subject:
        return
    
    config = load_settings()
    if alpha is None:
        alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Exporting all significant predictors for sub-{subject} (alpha={alpha})")

    all_predictors = []
    target_names = ("rating", "temp", "temperature")

    for target_name in target_names:
        roi_file_path = stats_dir / f"corr_stats_pow_roi_vs_{target_name}.tsv"
        roi_predictors = _process_predictor_file(
            roi_file_path, target_name, "ROI", use_fdr, alpha, logger
        )
        if roi_predictors is not None:
            all_predictors.append(roi_predictors)
        
        combined_file_path = stats_dir / f"corr_stats_pow_combined_vs_{target_name}.tsv"
        channel_predictors = _process_predictor_file(
            combined_file_path, target_name, "Channel", use_fdr, alpha, logger
        )
        if channel_predictors is not None:
            all_predictors.append(channel_predictors)

    output_file_path = stats_dir / "all_significant_predictors.csv"
    if all_predictors:
        combined_dataframe = pd.concat(all_predictors, ignore_index=True)
        combined_dataframe["abs_r"] = combined_dataframe["r"].abs()
        combined_dataframe = combined_dataframe.sort_values("p", ascending=True)
        combined_dataframe.to_csv(output_file_path, index=False)

        logger.info(
            f"Exported {len(combined_dataframe)} total significant predictors to: "
            f"{output_file_path}"
        )
        n_roi_predictors = len(
            combined_dataframe[combined_dataframe["type"] == "ROI"]
        )
        n_channel_predictors = len(
            combined_dataframe[combined_dataframe["type"] == "Channel"]
        )
        strongest_predictor = combined_dataframe.iloc[0]
        logger.info(
            f"Summary: {n_roi_predictors} ROI + {n_channel_predictors} channel predictors"
        )
        logger.info(
            f"Strongest predictor: {strongest_predictor['predictor']} "
            f"(r={strongest_predictor['r']:.3f})"
        )
    else:
        logger.warning("No significant predictors found")
        empty_dataframe = pd.DataFrame(
            columns=[
                "predictor",
                "region",
                "band",
                "r",
                "p",
                "n",
                "type",
                "target",
                "abs_r",
            ]
        )
        empty_dataframe.to_csv(output_file_path, index=False)


def export_combined_power_corr_stats(subject: str) -> None:
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    for target in ("rating", "temp"):
        frames: List[pd.DataFrame] = []
        for band in bands:
            f = stats_dir / f"corr_stats_pow_{band}_vs_{target}.tsv"
            if not f.exists():
                continue
            df = read_tsv(f)
            if df is None or df.empty:
                continue
            if "band" not in df.columns:
                df["band"] = band
            else:
                df["band"] = df["band"].fillna(band)
            frames.append(df)

        if frames:
            cat = pd.concat(frames, ignore_index=True)
            out_base = stats_dir / f"corr_stats_pow_combined_vs_{target}"
            cat.to_csv(out_base.with_suffix(".tsv"), sep="\t", index=False)
            cat.to_csv(out_base.with_suffix(".csv"), index=False)


###################################################################
# Global FDR
###################################################################


def apply_global_fdr(subject: str, alpha: float = None) -> None:
    """
    Apply global FDR correction across ALL analysis types for valid statistical inference.
    
    This function is CRITICAL for maintaining statistical validity. It applies Benjamini-Hochberg
    FDR correction across all tests from different analysis types (power ROI correlations,
    connectivity ROI summaries, connectivity edges, etc.) to control the family-wise error rate.
    
    Per-analysis-type FDR correction (applied in _apply_fdr_correction_and_save) is insufficient
    when interpreting results across multiple analysis types, as it inflates the false positive rate.
    
    The global FDR correction adds columns to each stats file:
    - p_used_for_global_fdr: The p-value used in global FDR correction
    - q_fdr_global: The FDR-adjusted q-value from global correction
    - fdr_reject_global: Boolean indicating rejection at global FDR level
    - fdr_crit_p_global: The critical p-value threshold for global FDR
    
    Parameters
    ----------
    subject : str
        Subject identifier (without 'sub-' prefix)
    alpha : float, optional
        FDR significance level (default: from config behavior_analysis.statistics.fdr_alpha, or 0.05)
    """
    if not subject:
        return
    
    config = load_settings()
    if alpha is None:
        alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    logger.info(
        f"Applying GLOBAL FDR correction (alpha={alpha}) across all analysis types. "
        f"This is CRITICAL for valid statistical inference when interpreting results across "
        f"multiple analysis types (power ROI, connectivity ROI, edges, etc.)."
    )

    patterns = [
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_roi_vs_temp.tsv", 
        "corr_stats_conn_roi_summary_*_vs_rating.tsv",
        "corr_stats_conn_roi_summary_*_vs_temp.tsv",
        "corr_stats_edges_*_vs_rating.tsv",
        "corr_stats_edges_*_vs_temp.tsv",
    ]
    
    files = [f for pat in patterns for f in sorted(stats_dir.glob(pat))]
    if not files:
        logger.warning(
            f"No stats TSVs found for global FDR in {stats_dir}. "
            f"Global FDR correction cannot be applied. Statistical validity may be compromised."
        )
        return

    p_values = []
    file_references = []
    metadata_records = []

    for file_path in files:
        try:
            dataframe = read_tsv(file_path)
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if dataframe is None or dataframe.empty:
            continue

        filename = file_path.name
        analysis_type = parse_analysis_type_from_filename(filename)
        target = parse_target_from_filename(filename)
        measure_band = parse_measure_band_from_filename(analysis_type, filename)

        p_permutation_series, p_raw_series = get_pvalue_series(dataframe)
        p_combined_series = p_permutation_series.where(np.isfinite(p_permutation_series), p_raw_series)
        valid_mask = np.isfinite(p_combined_series.to_numpy())
        if not np.any(valid_mask):
            continue

        for row_index, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            p_value, p_source = extract_pvalue_from_dataframe(dataframe, row_index)
            if not np.isfinite(p_value):
                continue
            p_values.append(p_value)
            file_references.append((file_path, row_index))
            metadata = build_meta_for_row(
                dataframe, row_index, filename, analysis_type, target, measure_band, p_source
            )
            metadata_records.append(metadata)

    if not p_values:
        logger.warning(
            "No valid p-values found for global FDR; skipping. "
            "Statistical validity may be compromised without global FDR correction."
        )
        return

    p_array = np.asarray(p_values, dtype=float)
    n_tests = len(p_array)
    logger.info(f"Applying global FDR correction to {n_tests} tests across {len(files)} files")
    
    q_array = _bh_adjust(p_array)
    rejections_array, critical_p = fdr_bh_reject(p_array, alpha=_safe_float(alpha))
    
    n_rejected = int(rejections_array.sum())
    logger.info(
        f"Global FDR correction results: {n_rejected}/{n_tests} tests significant "
        f"at global FDR alpha={alpha} (critical p={critical_p:.6f})"
    )

    file_updates = build_file_updates_dict(file_references, q_array, rejections_array, p_array)
    _apply_fdr_updates_to_files(file_updates, critical_p, stats_dir, logger)

    summary_rows = []
    for index, metadata in enumerate(metadata_records):
        row = dict(metadata)
        row["p_used_for_global_fdr"] = _safe_float(p_array[index])
        row["q_fdr_global"] = _safe_float(q_array[index])
        row["fdr_reject_global"] = bool(rejections_array[index])
        row["fdr_crit_p_global"] = _safe_float(critical_p)
        summary_rows.append(row)

    try:
        summary_dataframe = pd.DataFrame(summary_rows)
        write_tsv(summary_dataframe, stats_dir / "global_fdr_summary.tsv")
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to write global FDR summary: {e}")


###################################################################
# Group Input Collection
###################################################################

def _add_group_data_for_key(
    key: Tuple[str, str],
    x_values: np.ndarray,
    y_values: np.ndarray,
    subject: str,
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    do_temp: bool,
    rating_x: Dict[Tuple[str, str], List[np.ndarray]],
    rating_y: Dict[Tuple[str, str], List[np.ndarray]],
    rating_subjects: Dict[Tuple[str, str], List[str]],
    rating_Z: Dict[Tuple[str, str], List[pd.DataFrame]],
    rating_hasZ: Dict[Tuple[str, str], List[bool]],
    temp_x: Dict[Tuple[str, str], List[np.ndarray]],
    temp_y: Dict[Tuple[str, str], List[np.ndarray]],
    temp_subjects: Dict[Tuple[str, str], List[str]],
    temp_Z: Dict[Tuple[str, str], List[pd.DataFrame]],
    temp_hasZ: Dict[Tuple[str, str], List[bool]],
) -> None:
    rating_x[key].append(x_values)
    rating_y[key].append(y_values)
    rating_subjects[key].append(subject)
    has_covariates = covariates_df is not None and not covariates_df.empty
    rating_Z[key].append(covariates_df if has_covariates else None)
    rating_hasZ[key].append(has_covariates)

    if do_temp and temp_series is not None:
        temp_values = temp_series.to_numpy()
        temp_x[key].append(x_values)
        temp_y[key].append(temp_values)
        temp_subjects[key].append(subject)
        has_temp_covariates = covariates_without_temp_df is not None and not covariates_without_temp_df.empty
        temp_Z[key].append(covariates_without_temp_df if has_temp_covariates else None)
        temp_hasZ[key].append(has_temp_covariates)


def collect_group_power_roi_inputs(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    *,
    partial_covars: Optional[List[str]] = None,
    do_temp: bool = True,
) -> GroupScatterInputs:
    rating_x = defaultdict(list)
    rating_y = defaultdict(list)
    rating_Z = defaultdict(list)
    rating_hasZ = defaultdict(list)
    rating_subjects = defaultdict(list)

    temp_x = defaultdict(list)
    temp_y = defaultdict(list)
    temp_Z = defaultdict(list)
    temp_hasZ = defaultdict(list)
    temp_subjects = defaultdict(list)

    have_temp = False

    for sub in subjects:
        epochs, aligned_events = load_epochs_for_analysis(
            sub,
            task,
            align="strict",
            preload=False,
            deriv_root=deriv_root,
            bids_root=config.bids_root,
            config=config,
            logger=None,
        )
        if epochs is None:
            continue

        try:
            _, pow_df, _, y, info = _load_features_and_targets(
                sub, task, deriv_root, config, epochs=epochs
            )
        except FileNotFoundError:
            continue

        y = pd.to_numeric(y, errors="coerce")

        temp_series, temp_col = extract_temperature_data(aligned_events, config)
        if do_temp and temp_series is not None:
            have_temp = True

        covariates_df, covariates_without_temp_df = _build_covariate_matrices(aligned_events, partial_covars, temp_col, config=config)

        roi_map = _build_rois(info, config=config)
        power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])

        for band in power_bands_to_use:
            band_columns = [col for col in pow_df.columns if col.startswith(f"pow_{band}_")]
            if not band_columns:
                continue
            band_values = pow_df[band_columns].apply(pd.to_numeric, errors="coerce")
            overall_values = band_values.mean(axis=1).to_numpy()
            key_overall = (band, "All")

            _add_group_data_for_key(
                key_overall, overall_values, y.to_numpy(), sub,
                covariates_df, covariates_without_temp_df,
                temp_series, do_temp,
                rating_x, rating_y, rating_subjects, rating_Z, rating_hasZ,
                temp_x, temp_y, temp_subjects, temp_Z, temp_hasZ
            )

            for roi, channels in roi_map.items():
                roi_columns = [f"pow_{band}_{ch}" for ch in channels if f"pow_{band}_{ch}" in pow_df.columns]
                if not roi_columns:
                    continue
                roi_values = pow_df[roi_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
                key_roi = (band, roi)

                _add_group_data_for_key(
                    key_roi, roi_values, y.to_numpy(), sub,
                    covariates_df, covariates_without_temp_df,
                    temp_series, do_temp,
                    rating_x, rating_y, rating_subjects, rating_Z, rating_hasZ,
                    temp_x, temp_y, temp_subjects, temp_Z, temp_hasZ
                )

    return GroupScatterInputs(
        rating_x=dict(rating_x),
        rating_y=dict(rating_y),
        rating_Z=dict(rating_Z),
        rating_hasZ=dict(rating_hasZ),
        rating_subjects=dict(rating_subjects),
        temp_x=dict(temp_x),
        temp_y=dict(temp_y),
        temp_Z=dict(temp_Z),
        temp_hasZ=dict(temp_hasZ),
        temp_subjects=dict(temp_subjects),
        have_temp=have_temp,
    )
