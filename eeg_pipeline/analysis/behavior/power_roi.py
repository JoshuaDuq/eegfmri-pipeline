import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import (
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
    get_correlation_method,
    compute_temp_correlations_for_roi,
    _safe_float,
    compute_channel_rating_correlations,
)
from eeg_pipeline.analysis.behavior.core import save_correlation_results
from eeg_pipeline.utils.analysis.tfr import (
    build_rois_from_info as _build_rois,
    get_summary_type,
)
from eeg_pipeline.utils.config.loader import get_frequency_bands_for_aperiodic
from eeg_pipeline.analysis.behavior.correlations import (
    AnalysisConfig,
    _align_groups_to_series,
    _build_temp_record_unified,
    _compute_roi_correlation_stats,
)
from eeg_pipeline.analysis.behavior.core import build_correlation_record
from eeg_pipeline.utils.analysis.stats import (
    prepare_aligned_data,
)
from eeg_pipeline.utils.io.general import build_partial_covars_string




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
    record = build_correlation_record(
        identifier=roi,
        band=band,
        r=correlation,
        p=p_value,
        n=n_eff,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_part,
        p_partial=p_part,
        n_partial=n_part,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        r_partial_temp=r_part_temp,
        p_partial_temp=p_part_temp,
        n_partial_temp=n_part_temp,
        p_partial_temp_perm=p_partial_temp_perm,
        identifier_type="roi",
        analysis_type="power",
        band_range=band_range_str,
        partial_covars=partial_covars_str,
        n_perm=n_perm,
    )
    return record.to_dict()


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
        save_correlation_results(
            df_rating,
            analysis_cfg.stats_dir / "corr_stats_pow_roi_vs_rating.tsv",
            apply_fdr=True,
            config=analysis_cfg.config,
            logger=analysis_cfg.logger,
            use_permutation_p=True,
            add_fdr_reject=True,
        )
        save_correlation_results(
            df_rating,
            analysis_cfg.stats_dir / "corr_stats_pow_combined_vs_rating.tsv",
            apply_fdr=False,
            config=analysis_cfg.config,
            logger=analysis_cfg.logger,
        )

    if recs_temp:
        df_temp = pd.DataFrame(recs_temp)
        save_correlation_results(
            df_temp,
            analysis_cfg.stats_dir / "corr_stats_pow_roi_vs_temp.tsv",
            apply_fdr=True,
            config=analysis_cfg.config,
            logger=analysis_cfg.logger,
            use_permutation_p=False,
            add_fdr_reject=True,
        )
        save_correlation_results(
            df_temp,
            analysis_cfg.stats_dir / "corr_stats_pow_combined_vs_temp.tsv",
            apply_fdr=False,
            config=analysis_cfg.config,
            logger=analysis_cfg.logger,
        )


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
    """Build correlation record for channel-level power analysis."""
    record = build_correlation_record(
        identifier=channel,
        band=band,
        r=correlation,
        p=p_value,
        n=n_valid,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_partial,
        p_partial=p_partial,
        n_partial=n_partial,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        identifier_type="channel",
        analysis_type="power",
    )
    return record.to_dict()


def _build_temp_record_for_channel(
    channel_name: str,
    band: str,
    channel_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    analysis_cfg: AnalysisConfig,
    groups: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Build temperature correlation record for channel-level power."""
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
    """Process correlations for a single channel within a frequency band."""
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
    """Process channel-level power correlations for all bands."""
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
            save_correlation_results(
                rating_df,
                analysis_cfg.stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv",
                apply_fdr=True,
                config=analysis_cfg.config,
                logger=analysis_cfg.logger,
                use_permutation_p=True,
                add_fdr_reject=True,
            )

        if temp_records:
            temp_df = pd.DataFrame(temp_records)
            save_correlation_results(
                temp_df,
                analysis_cfg.stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv",
                apply_fdr=True,
                config=analysis_cfg.config,
                logger=analysis_cfg.logger,
                use_permutation_p=False,
                add_fdr_reject=True,
            )


def _run_power_roi_correlations(
    subject: str,
    task: str,
    config: Any,
    logger: Any,
    deriv_root: Path,
    stats_dir: Path,
    pow_df: pd.DataFrame,
    conn_df: Optional[pd.DataFrame],
    y: pd.Series,
    info: Optional[mne.Info],
    aligned_events: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    temp_col: Optional[str],
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
    use_spearman: bool,
    bootstrap: int,
    n_perm: int,
    rng: np.random.Generator,
) -> None:
    """Core implementation for power ROI correlations."""
    from eeg_pipeline.analysis.behavior.connectivity import compute_sliding_state_metrics
    from eeg_pipeline.analysis.behavior.specialized_features import (
        _process_itpc_correlations,
        _process_aperiodic_correlations,
        _process_pac_correlations,
    )
    
    logger.info(f"Starting ROI power correlation analysis for sub-{subject}")
    ensure_dir(stats_dir)
    
    if pow_df is None or y is None:
        logger.warning("No power features or targets")
        return
    
    # Extract sub-dataframes
    itpc_cols = [c for c in pow_df.columns if str(c).startswith("itpc_")]
    itpc_df = pow_df[itpc_cols].copy() if itpc_cols else pd.DataFrame()
    
    aper_cols = [c for c in pow_df.columns if str(c).startswith(("aper_slope_", "aper_offset_", "powcorr_"))]
    aper_df = pow_df[aper_cols].copy() if aper_cols else pd.DataFrame()
    
    pow_df = pow_df[[c for c in pow_df.columns if str(c).startswith("pow_")]]
    
    # Get groups for mixed effects
    group_col = next((c for c in ("run_id", "run", "block") if aligned_events is not None and c in aligned_events.columns), None)
    groups = aligned_events[group_col] if group_col else None
    
    analysis_cfg = AnalysisConfig(
        subject=subject,
        config=config,
        logger=logger,
        rng=rng,
        bootstrap=bootstrap,
        n_perm=n_perm,
        use_spearman=use_spearman,
        method=get_correlation_method(use_spearman),
        min_samples_channel=config.get("behavior_analysis.statistics.min_samples_channel", 10),
        min_samples_roi=config.get("behavior_analysis.statistics.min_samples_roi", 20),
        groups=groups,
    )
    analysis_cfg.stats_dir = stats_dir
    
    roi_map = _build_rois(info, config=config) if info else {}
    if not roi_map:
        logger.warning(f"No ROI definitions found; skipping ROI stats for sub-{subject}")
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    freq_bands = config.get("time_frequency_analysis.bands", {})
    freq_bands_aper = get_frequency_bands_for_aperiodic(config)
    
    rating_records, temp_records, mixed_effects_records = [], [], []
    missing_roi_by_band: Dict[str, List[str]] = defaultdict(list)
    
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
        detail = "; ".join(f"{b}: {', '.join(sorted(set(r)))}" for b, r in missing_roi_by_band.items())
        logger.warning(f"Skipped ROI statistics for missing channel groups: {detail}")
    
    _save_roi_results(rating_records, temp_records, analysis_cfg)
    _save_mixed_effects_results(mixed_effects_records, analysis_cfg)
    _process_channel_level_correlations(power_bands, pow_df, y, covariates_df, temp_series, covariates_without_temp_df, analysis_cfg)
    _process_itpc_correlations(itpc_df, y, covariates_df, temp_series, covariates_without_temp_df, analysis_cfg)
    _process_aperiodic_correlations(aper_df, roi_map, y, temp_series, covariates_df, covariates_without_temp_df, freq_bands_aper, analysis_cfg)
    
    pac_path = deriv_features_path(deriv_root, subject) / "features_pac_trials.tsv"
    pac_trials_df = read_tsv(pac_path) if pac_path.exists() else None
    _process_pac_correlations(pac_trials_df, aligned_events, analysis_cfg, temp_series, covariates_df, covariates_without_temp_df)
    
    compute_sliding_state_metrics(
        subject=subject, task=task, conn_df=conn_df, aligned_events=aligned_events,
        deriv_root=deriv_root, config=config, logger=logger,
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
    """Compute ROI power correlations using BehaviorContext for data loading."""
    from eeg_pipeline.pipelines.behavior import initialize_analysis_context
    from eeg_pipeline.analysis.behavior.core import BehaviorContext
    
    if not subject:
        return
    
    try:
        config, task, deriv_root, stats_dir, logger = initialize_analysis_context(
            subject, task, None
        )
    except ValueError:
        return
    
    rng_seed = config.get("random.seed", 42)
    rng = rng or np.random.default_rng(rng_seed)
    
    ctx = BehaviorContext(
        subject=subject,
        task=task,
        config=config,
        logger=logger,
        deriv_root=deriv_root,
        stats_dir=stats_dir,
        use_spearman=use_spearman,
        bootstrap=bootstrap,
        n_perm=n_perm,
        rng=rng,
        partial_covars=partial_covars,
    )
    
    if not ctx.load_data():
        return
    
    compute_power_roi_stats_from_context(ctx)


def compute_power_roi_stats_from_context(ctx: "BehaviorContext") -> None:
    """Compute ROI power correlations using pre-loaded data from context."""
    _run_power_roi_correlations(
        ctx.subject, ctx.task, ctx.config, ctx.logger, ctx.deriv_root, ctx.stats_dir,
        ctx.power_df, ctx.connectivity_df, ctx.targets, ctx.epochs_info,
        ctx.aligned_events, ctx.temperature, ctx.temperature_column,
        ctx.covariates_df, ctx.covariates_without_temp_df,
        ctx.use_spearman, ctx.bootstrap, ctx.n_perm, ctx.rng,
    )

