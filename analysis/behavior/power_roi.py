import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import (
    _load_features_and_targets,
    load_epochs_for_analysis,
    extract_temperature_data,
    extract_roi_columns,
    build_covariate_matrix,
    build_covariates_without_temp,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    deriv_features_path,
    ensure_dir,
    get_subject_logger,
    format_band_range,
    write_tsv,
    read_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    apply_fdr_correction_and_save,
    get_correlation_method,
    compute_temp_correlations_for_roi,
    _safe_float,
    compute_channel_rating_correlations,
)
from eeg_pipeline.utils.analysis.tfr import (
    build_rois_from_info as _build_rois,
    get_summary_type,
)
from eeg_pipeline.utils.config.loader import get_frequency_bands_for_aperiodic
from eeg_pipeline.analysis.behavior.correlations import (
    AnalysisConfig,
    _align_groups_to_series,
    _build_base_correlation_record,
    _build_temp_record_unified,
    _compute_roi_correlation_stats,
)
from eeg_pipeline.utils.analysis.stats import (
    prepare_aligned_data,
)
from eeg_pipeline.utils.io.general import build_partial_covars_string


def _load_analysis_data(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Any,
) -> Tuple[
    Optional[pd.DataFrame],  # power_df
    Optional[pd.DataFrame],  # conn_df
    Optional[pd.Series],
    Optional[mne.Info],
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[str],
]:
    if not subject or not task:
        logger.error("Subject and task must be provided")
        return None, None, None, None, None, None, None
    
    _, power_df, conn_df, target_values, info = _load_features_and_targets(
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
        return None, None, None, None, None, None, None

    temp_series, temp_col = extract_temperature_data(aligned_events, config)

    return power_df, conn_df, target_values, info, aligned_events, temp_series, temp_col


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


def _build_temp_record_for_roi(
    roi_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    band: str,
    roi: str,
    band_range_str: str,
    analysis_cfg: AnalysisConfig,
    groups: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    if (analysis_cfg.bootstrap is not None and analysis_cfg.bootstrap > 0) or (analysis_cfg.n_perm is not None and analysis_cfg.n_perm > 0):
        temp_record = compute_temp_correlations_for_roi(
            roi_values,
            temp_series,
            covariates_without_temp_df,
            analysis_cfg.bootstrap,
            analysis_cfg.n_perm,
            analysis_cfg.use_spearman,
            analysis_cfg.method,
            analysis_cfg.rng,
            band,
            roi,
            analysis_cfg.logger,
            config=analysis_cfg.config,
            groups=groups,
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
        analysis_cfg=analysis_cfg,
        groups=analysis_cfg.groups,
        band_range=band_range_str,
    )
    
    return temp_record


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
    band_range_str: str,
    analysis_cfg: AnalysisConfig,
    mixed_effects_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if power_df is None or target_values is None or power_df.empty or target_values.empty:
        return None, None
    
    roi_columns = extract_roi_columns(roi, channels, band, band_columns)
    if roi_columns is None:
        return None, None

    roi_values = power_df[roi_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    context = f"ROI {roi} ({band})"
    x_aligned, y_aligned, covariates_aligned, _, n_eff = prepare_aligned_data(
        roi_values, target_values, covariates_df,
        min_samples=analysis_cfg.min_samples_roi, logger=analysis_cfg.logger, context=context,
    )

    if x_aligned is None or y_aligned is None:
        return None, None

    groups_aligned = _align_groups_to_series(x_aligned, analysis_cfg.groups)

    stats = _compute_roi_correlation_stats(
        x_aligned, y_aligned, x_aligned, y_aligned,
        covariates_aligned, temp_series, n_eff, band, roi, context, analysis_cfg,
        groups=groups_aligned, mixed_effects_records=mixed_effects_records,
    )

    partial_covars_str = build_partial_covars_string(covariates_df)

    rating_record = _build_roi_rating_record(
        roi, band, band_range_str, stats.correlation, stats.p_value, n_eff, analysis_cfg.method,
        stats.ci_low, stats.ci_high, stats.r_partial, stats.p_partial, stats.n_partial, partial_covars_str,
        stats.r_partial_temp, stats.p_partial_temp, stats.n_partial_temp,
        stats.p_perm, stats.p_partial_perm, stats.p_partial_temp_perm, analysis_cfg.n_perm,
    )

    temp_record = _build_temp_record_for_roi(
        roi_values, temp_series, covariates_without_temp_df,
        band, roi, band_range_str, analysis_cfg, groups=groups_aligned,
    )

    return rating_record, temp_record


def _save_roi_results(
    recs_rating: List[Dict[str, Any]],
    recs_temp: List[Dict[str, Any]],
    analysis_cfg: AnalysisConfig,
) -> None:
    if recs_rating:
        df_rating = pd.DataFrame(recs_rating)
        analysis_cfg.logger.info(
            "Saving ROI stats. NOTE: These files use local FDR correction. "
            "Refer to group_aggregation outputs for Global FDR corrected results."
        )
        apply_fdr_correction_and_save(
            df_rating,
            analysis_cfg.stats_dir / "corr_stats_pow_roi_vs_rating.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
        )
        write_tsv(df_rating, analysis_cfg.stats_dir / "corr_stats_pow_combined_vs_rating.tsv")

    if recs_temp:
        df_temp = pd.DataFrame(recs_temp)
        apply_fdr_correction_and_save(
            df_temp, analysis_cfg.stats_dir / "corr_stats_pow_roi_vs_temp.tsv", analysis_cfg.config, analysis_cfg.logger
        )
        write_tsv(df_temp, analysis_cfg.stats_dir / "corr_stats_pow_combined_vs_temp.tsv")


def _save_mixed_effects_results(
    mixed_effects_records: List[Dict[str, Any]],
    analysis_cfg: AnalysisConfig,
) -> None:
    if not mixed_effects_records:
        return
    try:
        df = pd.DataFrame(mixed_effects_records)
        write_tsv(df, analysis_cfg.stats_dir / "mixed_effects_fixed_effects.tsv")
        analysis_cfg.logger.info("Saved mixed-effects fixed effects table: %d rows", len(df))
    except Exception as exc:
        analysis_cfg.logger.warning("Failed to save mixed-effects results: %s", exc)


###################################################################
# Channel-Level Processing
###################################################################


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
    """
    Build correlation record for channel-level power analysis.
    
    Parameters
    ----------
    channel : str
        Channel name
    band : str
        Frequency band
    correlation : float
        Correlation coefficient
    p_value : float
        P-value
    ci_low : float
        Lower confidence interval bound
    ci_high : float
        Upper confidence interval bound
    r_partial : float
        Partial correlation coefficient
    p_partial : float
        Partial correlation p-value
    n_partial : int
        Sample size for partial correlation
    p_perm : float
        Permutation p-value
    p_partial_perm : float
        Permutation p-value for partial correlation
    n_valid : int
        Number of valid samples
    method : str
        Correlation method used
        
    Returns
    -------
    Dict[str, Any]
        Correlation record dictionary
    """
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


def _build_temp_record_for_channel(
    channel_name: str,
    band: str,
    channel_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    analysis_cfg: AnalysisConfig,
    groups: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """
    Build temperature correlation record for channel-level power.
    
    Parameters
    ----------
    channel_name : str
        Channel name
    band : str
        Frequency band
    channel_values : pd.Series
        Channel power values
    temp_series : Optional[pd.Series]
        Temperature series
    covariates_without_temp_df : Optional[pd.DataFrame]
        Covariates excluding temperature
    analysis_cfg : AnalysisConfig
        Analysis configuration
    groups : Optional[np.ndarray]
        Grouping variable for mixed-effects
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Temperature correlation record, or None if temp_series is None
    """
    return _build_temp_record_unified(
        x_values=channel_values,
        temp_series=temp_series,
        covariates_without_temp_df=covariates_without_temp_df,
        identifier=channel_name,
        identifier_key="channel",
        band=band,
        analysis_cfg=analysis_cfg,
        groups=groups,
    )


def _process_single_channel_for_band(
    column_name: str,
    band: str,
    power_df: pd.DataFrame,
    target_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
    analysis_cfg: AnalysisConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process correlations for a single channel within a frequency band.
    
    Parameters
    ----------
    column_name : str
        Column name in power_df (e.g., "pow_alpha_Cz")
    band : str
        Frequency band name
    power_df : pd.DataFrame
        Power features dataframe
    target_values : pd.Series
        Target variable (rating)
    temp_series : Optional[pd.Series]
        Temperature series
    covariates_df : Optional[pd.DataFrame]
        Covariates dataframe
    covariates_without_temp_df : Optional[pd.DataFrame]
        Covariates excluding temperature
    analysis_cfg : AnalysisConfig
        Analysis configuration
        
    Returns
    -------
    Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]
        (rating_record, temp_record) tuple
    """
    channel_name = column_name.replace(f"pow_{band}_", "")
    channel_values = pd.to_numeric(power_df[column_name], errors="coerce")
    context = f"channel {channel_name} ({band})"
    
    x_aligned, y_aligned, covariates_aligned, _, _ = prepare_aligned_data(
        channel_values,
        target_values,
        covariates_df,
        min_samples=analysis_cfg.min_samples_channel,
        logger=analysis_cfg.logger,
        context=context,
    )
    
    if x_aligned is None or y_aligned is None:
        return None, None
    
    groups_aligned = _align_groups_to_series(x_aligned, analysis_cfg.groups)

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
        analysis_cfg.bootstrap,
        analysis_cfg.n_perm,
        analysis_cfg.use_spearman,
        analysis_cfg.method,
        analysis_cfg.rng,
        logger=analysis_cfg.logger,
        config=analysis_cfg.config,
        groups=groups_aligned,
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
        analysis_cfg.method,
    )
    
    temp_record = _build_temp_record_for_channel(
        channel_name,
        band,
        channel_values,
        temp_series,
        covariates_without_temp_df,
        analysis_cfg,
        groups=groups_aligned,
    )
    
    return rating_record, temp_record


def _process_channel_level_correlations(
    power_bands: List[str],
    power_df: pd.DataFrame,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    analysis_cfg: AnalysisConfig,
) -> None:
    """
    Process channel-level power correlations for all bands.
    
    Computes correlations between channel-level power and rating/temperature,
    applies FDR correction, and saves results per band.
    
    Parameters
    ----------
    power_bands : List[str]
        List of frequency bands to process
    power_df : pd.DataFrame
        Power features dataframe
    target_values : pd.Series
        Target variable (rating)
    covariates_df : Optional[pd.DataFrame]
        Covariates dataframe
    temp_series : Optional[pd.Series]
        Temperature series
    covariates_without_temp_df : Optional[pd.DataFrame]
        Covariates excluding temperature
    analysis_cfg : AnalysisConfig
        Analysis configuration
    """
    for band in power_bands:
        band_columns = [
            col for col in power_df.columns if col.startswith(f"pow_{band}_")
        ]
        if not band_columns:
            analysis_cfg.logger.debug(f"No channel-level power columns for band '{band}'")
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
                analysis_cfg,
            )
            if rating_record is not None:
                rating_records.append(rating_record)
            if temp_record is not None:
                temp_records.append(temp_record)

        if rating_records:
            rating_df = pd.DataFrame(rating_records)
            apply_fdr_correction_and_save(
                rating_df,
                analysis_cfg.stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv",
                analysis_cfg.config,
                analysis_cfg.logger,
            )

        if temp_records:
            temp_df = pd.DataFrame(temp_records)
            apply_fdr_correction_and_save(
                temp_df,
                analysis_cfg.stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv",
                analysis_cfg.config,
                analysis_cfg.logger,
                use_permutation_p=False,
            )


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
    from eeg_pipeline.analysis.behavior.connectivity import compute_sliding_state_metrics
    from eeg_pipeline.analysis.behavior.specialized_features import (
        _process_itpc_correlations,
        _process_aperiodic_correlations,
        _process_pac_correlations,
    )
    
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

    pow_df, conn_df, y, info, aligned_events, temp_series, temp_col = _load_analysis_data(
        subject, task, deriv_root, config, logger
    )
    if pow_df is None or y is None:
        return

    itpc_columns = [c for c in pow_df.columns if str(c).startswith("itpc_")]
    itpc_df = pow_df[itpc_columns].copy() if itpc_columns else pd.DataFrame()

    aper_columns = [c for c in pow_df.columns if str(c).startswith(("aper_slope_", "aper_offset_", "powcorr_"))]
    aper_df = pow_df[aper_columns].copy() if aper_columns else pd.DataFrame()

    pow_df = pow_df[[c for c in pow_df.columns if str(c).startswith("pow_")]]
    
    group_col = next((c for c in ("run_id", "run", "block") if aligned_events is not None and c in aligned_events.columns), None)
    groups = aligned_events[group_col] if group_col else None
    
    method = get_correlation_method(use_spearman)
    analysis_cfg = AnalysisConfig(
        subject=subject,
        config=config,
        logger=logger,
        rng=rng,
        bootstrap=bootstrap,
        n_perm=n_perm,
        use_spearman=use_spearman,
        method=method,
        min_samples_channel=config.get("behavior_analysis.statistics.min_samples_channel", 10),
        min_samples_roi=config.get("behavior_analysis.statistics.min_samples_roi", 20),
        groups=groups,
    )
    analysis_cfg.stats_dir = stats_dir

    roi_map = _build_rois(info, config=config)
    if not roi_map:
        logger.warning(f"No ROI definitions found; skipping ROI stats for sub-{subject}")
        roi_map = {}

    covariates_df = build_covariate_matrix(aligned_events, partial_covars, config)
    covariates_without_temp_df = build_covariates_without_temp(covariates_df, temp_col)

    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    freq_bands = config.get("time_frequency_analysis.bands", {})
    freq_bands_aper = get_frequency_bands_for_aperiodic(config)

    rating_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []
    missing_roi_by_band: Dict[str, List[str]] = defaultdict(list)
    mixed_effects_records: List[Dict[str, Any]] = []

    for band in power_bands:
        band_columns = {col for col in pow_df.columns if col.startswith(f"pow_{band}_")}
        if not band_columns:
            continue

        band_range_str = format_band_range(band, freq_bands)

        for roi, channels in roi_map.items():
            rating_record, temp_record = _process_single_roi_for_band(
                roi, channels, band, band_columns, pow_df, y, temp_series,
                covariates_df, covariates_without_temp_df, band_range_str, analysis_cfg,
                mixed_effects_records=mixed_effects_records,
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

    _save_roi_results(rating_records, temp_records, analysis_cfg)
    _save_mixed_effects_results(mixed_effects_records, analysis_cfg)

    _process_channel_level_correlations(
        power_bands,
        pow_df,
        y,
        covariates_df,
        temp_series,
        covariates_without_temp_df,
        analysis_cfg,
    )

    _process_itpc_correlations(
        itpc_df,
        y,
        covariates_df,
        temp_series,
        covariates_without_temp_df,
        analysis_cfg,
    )

    _process_aperiodic_correlations(
        aper_df,
        roi_map,
        y,
        temp_series,
        covariates_df,
        covariates_without_temp_df,
        freq_bands_aper,
        analysis_cfg,
    )

    pac_path = deriv_features_path(deriv_root, subject) / "features_pac_trials.tsv"
    pac_trials_df = read_tsv(pac_path) if pac_path.exists() else None
    _process_pac_correlations(
        pac_trials_df,
        aligned_events,
        analysis_cfg,
        temp_series,
        covariates_df,
        covariates_without_temp_df,
    )

    compute_sliding_state_metrics(
        subject=subject,
        task=task,
        conn_df=conn_df,
        aligned_events=aligned_events,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
    )

