import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import (
    deriv_features_path,
    read_tsv,
    build_partial_covars_string,
    format_band_range,
    get_column_from_config,
)
from eeg_pipeline.utils.analysis.stats import (
    apply_fdr_correction_and_save,
    prepare_aligned_data,
    compute_channel_rating_correlations,
)
from eeg_pipeline.analysis.behavior.correlations import (
    AnalysisConfig,
    _align_groups_to_series,
    _build_base_correlation_record,
    _build_temp_record_unified,
    _compute_roi_correlation_stats,
)


###################################################################
# Aperiodic Feature Processing
###################################################################


def _parse_powcorr_column(col: str) -> Optional[Tuple[str, str]]:
    """
    Parse aperiodic power correlation column name to extract band and channel.
    
    Parameters
    ----------
    col : str
        Column name (e.g., "powcorr_alpha_Cz")
        
    Returns
    -------
    Optional[Tuple[str, str]]
        (band, channel) tuple if valid, None otherwise
    """
    if not isinstance(col, str) or not col.startswith("powcorr_"):
        return None
    parts = col.split("_")
    if len(parts) < 3:
        return None
    band = parts[1]
    channel = "_".join(parts[2:])
    return band, channel


def _process_aperiodic_correlations(
    aper_df: pd.DataFrame,
    roi_map: Dict[str, List[str]],
    target_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
    freq_bands: Dict[str, List[float]],
    analysis_cfg: AnalysisConfig,
) -> None:
    """
    Process aperiodic feature correlations (slope, offset, powcorr) with behavioral data.
    
    Processes channel-level and ROI-level aperiodic features, computing correlations
    with rating and temperature, applying FDR correction, and saving results.
    """
    if aper_df is None or aper_df.empty:
        return

    from eeg_pipeline.analysis.behavior.power_roi import _build_channel_rating_record
    from eeg_pipeline.analysis.behavior.power_roi import (
        _build_roi_rating_record,
        _build_temp_record_for_roi,
    )

    # Channel-level slopes/offsets
    for metric in ("aper_slope", "aper_offset"):
        cols = [c for c in aper_df.columns if str(c).startswith(f"{metric}_")]
        if not cols:
            continue

        rating_records: List[Dict[str, Any]] = []
        temp_records: List[Dict[str, Any]] = []

        for col in cols:
            channel = col.replace(f"{metric}_", "")
            values = pd.to_numeric(aper_df[col], errors="coerce")
            context = f"{metric} {channel}"
            x_aligned, y_aligned, cov_aligned, _, _ = prepare_aligned_data(
                values, target_values, covariates_df,
                min_samples=analysis_cfg.min_samples_channel, logger=analysis_cfg.logger, context=context,
            )
            if x_aligned is None or y_aligned is None:
                continue
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
                cov_aligned,
                analysis_cfg.bootstrap,
                analysis_cfg.n_perm,
                analysis_cfg.use_spearman,
                analysis_cfg.method,
                analysis_cfg.rng,
                logger=analysis_cfg.logger,
                config=analysis_cfg.config,
                groups=groups_aligned,
            )
            rating_records.append(
                _build_channel_rating_record(
                    channel,
                    metric,
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
            )

            temp_record = _build_temp_record_unified(
                x_values=values,
                temp_series=temp_series,
                covariates_without_temp_df=covariates_without_temp_df,
                identifier=channel,
                identifier_key="channel",
                band=metric,
                analysis_cfg=analysis_cfg,
                groups=groups_aligned,
            )
            if temp_record is not None:
                temp_records.append(temp_record)

        if rating_records:
            rating_df = pd.DataFrame(rating_records)
            apply_fdr_correction_and_save(
                rating_df,
                analysis_cfg.stats_dir / f"corr_stats_{metric}_vs_rating.tsv",
                analysis_cfg.config,
                analysis_cfg.logger,
            )
        if temp_records:
            temp_df = pd.DataFrame(temp_records)
            apply_fdr_correction_and_save(
                temp_df,
                analysis_cfg.stats_dir / f"corr_stats_{metric}_vs_temp.tsv",
                analysis_cfg.config,
                analysis_cfg.logger,
                use_permutation_p=False,
            )

    # Channel-level residual bands
    powcorr_cols = [c for c in aper_df.columns if str(c).startswith("powcorr_")]
    bands_available = {b for b, _ in (_parse_powcorr_column(c) or (None, None) for c in powcorr_cols) if b}
    rating_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []

    for col in powcorr_cols:
        parsed = _parse_powcorr_column(col)
        if parsed is None:
            continue
        band, channel = parsed
        values = pd.to_numeric(aper_df[col], errors="coerce")
        context = f"powcorr {band} {channel}"
        x_aligned, y_aligned, cov_aligned, _, _ = prepare_aligned_data(
            values, target_values, covariates_df,
            min_samples=analysis_cfg.min_samples_channel, logger=analysis_cfg.logger, context=context,
        )
        if x_aligned is None or y_aligned is None:
            continue
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
            cov_aligned,
            analysis_cfg.bootstrap,
            analysis_cfg.n_perm,
            analysis_cfg.use_spearman,
            analysis_cfg.method,
            analysis_cfg.rng,
            logger=analysis_cfg.logger,
            config=analysis_cfg.config,
            groups=groups_aligned,
        )
        rating_records.append(
            _build_channel_rating_record(
                channel,
                f"powcorr_{band}",
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
        )
        temp_record = _build_temp_record_unified(
            x_values=values,
            temp_series=temp_series,
            covariates_without_temp_df=covariates_without_temp_df,
            identifier=channel,
            identifier_key="channel",
            band=f"powcorr_{band}",
            analysis_cfg=analysis_cfg,
            groups=groups_aligned,
        )
        if temp_record is not None:
            temp_records.append(temp_record)

    if rating_records:
        rating_df = pd.DataFrame(rating_records)
        apply_fdr_correction_and_save(
            rating_df,
            analysis_cfg.stats_dir / "corr_stats_powcorr_vs_rating.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
        )
    if temp_records:
        temp_df = pd.DataFrame(temp_records)
        apply_fdr_correction_and_save(
            temp_df,
            analysis_cfg.stats_dir / "corr_stats_powcorr_vs_temp.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
            use_permutation_p=False,
        )

    # ROI-level residual bands
    if roi_map:
        freq_bands_labels = freq_bands or {}
        rating_roi_records: List[Dict[str, Any]] = []
        temp_roi_records: List[Dict[str, Any]] = []
        for band in sorted(bands_available):
            band_range_str = format_band_range(band, freq_bands_labels)
            for roi, channels in roi_map.items():
                roi_cols = [f"powcorr_{band}_{ch}" for ch in channels if f"powcorr_{band}_{ch}" in aper_df.columns]
                if not roi_cols:
                    continue
                roi_values = aper_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                context = f"powcorr {band} ROI {roi}"
                x_aligned, y_aligned, cov_aligned, _, n_eff = prepare_aligned_data(
                    roi_values, target_values, covariates_df,
                    min_samples=analysis_cfg.min_samples_roi, logger=analysis_cfg.logger, context=context,
                )
                if x_aligned is None or y_aligned is None:
                    continue
                groups_aligned = _align_groups_to_series(x_aligned, analysis_cfg.groups)
                stats = _compute_roi_correlation_stats(
                    x_aligned, y_aligned, x_aligned, y_aligned,
                    cov_aligned, temp_series, n_eff, band, roi, context, analysis_cfg,
                    groups=groups_aligned,
                )
                partial_covars_str = build_partial_covars_string(covariates_df)
                rating_roi_records.append(
                    _build_roi_rating_record(
                        roi, band, band_range_str,
                        stats.correlation, stats.p_value, n_eff, analysis_cfg.method,
                        stats.ci_low, stats.ci_high, stats.r_partial, stats.p_partial, stats.n_partial, partial_covars_str,
                        stats.r_partial_temp, stats.p_partial_temp, stats.n_partial_temp,
                        stats.p_perm, stats.p_partial_perm, stats.p_partial_temp_perm, analysis_cfg.n_perm,
                    )
                )
                temp_record = _build_temp_record_for_roi(
                    roi_values, temp_series, covariates_without_temp_df,
                    band, roi, band_range_str, analysis_cfg, groups=groups_aligned,
                )
                if temp_record is not None:
                    temp_roi_records.append(temp_record)

        if rating_roi_records:
            rating_df = pd.DataFrame(rating_roi_records)
            apply_fdr_correction_and_save(
                rating_df,
                analysis_cfg.stats_dir / "corr_stats_powcorr_roi_vs_rating.tsv",
                analysis_cfg.config,
                analysis_cfg.logger,
            )
        if temp_roi_records:
            temp_df = pd.DataFrame(temp_roi_records)
            apply_fdr_correction_and_save(
                temp_df,
                analysis_cfg.stats_dir / "corr_stats_powcorr_roi_vs_temp.tsv",
                analysis_cfg.config,
                analysis_cfg.logger,
                use_permutation_p=False,
            )


###################################################################
# ITPC Feature Processing
###################################################################


def _parse_itpc_column(col: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse ITPC column name to extract band, time bin, and channel.
    
    Parameters
    ----------
    col : str
        Column name (e.g., "itpc_alpha_pre_Cz")
        
    Returns
    -------
    Optional[Tuple[str, str, str]]
        (band, time_bin, channel) tuple if valid, None otherwise
    """
    if not isinstance(col, str) or not col.startswith("itpc_"):
        return None
    parts = col.split("_")
    if len(parts) < 4:
        return None
    band = parts[1]
    time_bin = parts[-1]
    channel = "_".join(parts[2:-1])
    return band, time_bin, channel


def _build_itpc_rating_record(
    channel: str,
    band: str,
    time_bin: str,
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
    Build correlation record for ITPC feature.
    
    Parameters
    ----------
    channel : str
        Channel name
    band : str
        Frequency band
    time_bin : str
        Time bin identifier
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
        time_bin=time_bin,
    )


def _process_itpc_correlations(
    itpc_df: pd.DataFrame,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    analysis_cfg: AnalysisConfig,
) -> None:
    """
    Process ITPC (Inter-Trial Phase Coherence) correlations with behavioral data.
    
    Computes correlations between ITPC features and rating/temperature, applies
    FDR correction, and saves results.
    """
    if itpc_df is None or itpc_df.empty:
        return

    rating_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []

    for col in itpc_df.columns:
        parsed = _parse_itpc_column(str(col))
        if parsed is None:
            continue
        band, time_bin, channel_name = parsed
        channel_values = pd.to_numeric(itpc_df[col], errors="coerce")
        context = f"ITPC {channel_name} ({band}, {time_bin})"

        x_aligned, y_aligned, covariates_aligned, _, _ = prepare_aligned_data(
            channel_values,
            target_values,
            covariates_df,
            min_samples=analysis_cfg.min_samples_channel,
            logger=analysis_cfg.logger,
            context=context,
        )
        if x_aligned is None or y_aligned is None:
            continue

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

        rating_records.append(
            _build_itpc_rating_record(
                channel_name,
                band,
                time_bin,
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
        )

        temp_record = _build_temp_record_unified(
            x_values=channel_values,
            temp_series=temp_series,
            covariates_without_temp_df=covariates_without_temp_df,
            identifier=channel_name,
            identifier_key="channel",
            band=band,
            analysis_cfg=analysis_cfg,
            groups=groups_aligned,
            time_bin=time_bin,
        )
        if temp_record is not None:
            temp_records.append(temp_record)

    if rating_records:
        rating_df = pd.DataFrame(rating_records)
        apply_fdr_correction_and_save(
            rating_df,
            analysis_cfg.stats_dir / "corr_stats_itpc_vs_rating.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
        )

    if temp_records:
        temp_df = pd.DataFrame(temp_records)
        apply_fdr_correction_and_save(
            temp_df,
            analysis_cfg.stats_dir / "corr_stats_itpc_vs_temp.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
            use_permutation_p=False,
        )


###################################################################
# PAC Feature Processing
###################################################################


def _load_pac_trials(subject: str, deriv_root: Path) -> Optional[pd.DataFrame]:
    """
    Load PAC (Phase-Amplitude Coupling) trial-level features.
    
    Parameters
    ----------
    subject : str
        Subject identifier (without 'sub-' prefix)
    deriv_root : Path
        Derivatives root directory
        
    Returns
    -------
    Optional[pd.DataFrame]
        PAC features dataframe, or None if file not found
    """
    pac_path = deriv_features_path(deriv_root, subject) / "features_pac_trials.tsv"
    if not pac_path.exists():
        return None
    try:
        return read_tsv(pac_path)
    except Exception:
        return None


def _process_pac_correlations(
    pac_trials_df: pd.DataFrame,
    events_df: pd.DataFrame,
    analysis_cfg: AnalysisConfig,
    temp_series: Optional[pd.Series],
    covariates_df: Optional[pd.DataFrame],
    covariates_without_temp_df: Optional[pd.DataFrame],
) -> None:
    """
    Process PAC (Phase-Amplitude Coupling) correlations with behavioral data.
    
    Computes correlations between PAC features and rating/temperature, applies
    FDR correction, and saves results.
    """
    if pac_trials_df is None or pac_trials_df.empty or events_df is None or events_df.empty:
        return

    rating_col = get_column_from_config(analysis_cfg.config, "event_columns.rating", events_df)
    if rating_col is None or rating_col not in events_df.columns:
        return

    rating = pd.to_numeric(events_df[rating_col], errors="coerce")
    if len(rating) != len(pac_trials_df):
        return

    min_samples = analysis_cfg.min_samples_roi
    rating_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []

    for (roi, phase_f, amp_f), df_sub in pac_trials_df.groupby(["roi", "phase_freq", "amp_freq"]):
        pac_vals = pd.to_numeric(df_sub["pac"], errors="coerce")
        context = f"PAC {roi} {phase_f:.1f}-{amp_f:.1f}"
        x_aligned, y_aligned, cov_aligned, _, _ = prepare_aligned_data(
            pac_vals,
            rating,
            covariates_df,
            min_samples=min_samples,
            logger=analysis_cfg.logger,
            context=context,
        )
        if x_aligned is None or y_aligned is None:
            continue

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
            cov_aligned,
            analysis_cfg.bootstrap,
            analysis_cfg.n_perm,
            analysis_cfg.use_spearman,
            analysis_cfg.method,
            analysis_cfg.rng,
            logger=analysis_cfg.logger,
            config=analysis_cfg.config,
            groups=groups_aligned,
        )

        rating_records.append(
            _build_base_correlation_record(
                identifier=roi,
                identifier_key="roi",
                band=f"pac_{phase_f:.1f}_{amp_f:.1f}",
                correlation=correlation,
                p_value=p_value,
                n_valid=int(len(x_aligned)),
                method=analysis_cfg.method,
                ci_low=ci_low,
                ci_high=ci_high,
                r_partial=r_partial,
                p_partial=p_partial,
                n_partial=n_partial,
                p_perm=p_perm,
                p_partial_perm=p_partial_perm,
                phase_freq=float(phase_f),
                amp_freq=float(amp_f),
            )
        )

        temp_record = _build_temp_record_unified(
            x_values=pac_vals,
            temp_series=temp_series,
            covariates_without_temp_df=covariates_without_temp_df,
            identifier=roi,
            identifier_key="roi",
            band=f"pac_{phase_f:.1f}_{amp_f:.1f}",
            analysis_cfg=analysis_cfg,
            groups=groups_aligned,
            phase_freq=float(phase_f),
            amp_freq=float(amp_f),
        )
        if temp_record is not None:
            temp_records.append(temp_record)

    if rating_records:
        rating_df = pd.DataFrame(rating_records)
        apply_fdr_correction_and_save(
            rating_df,
            analysis_cfg.stats_dir / "corr_stats_pac_vs_rating.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
        )

    if temp_records:
        temp_df = pd.DataFrame(temp_records)
        apply_fdr_correction_and_save(
            temp_df,
            analysis_cfg.stats_dir / "corr_stats_pac_vs_temp.tsv",
            analysis_cfg.config,
            analysis_cfg.logger,
            use_permutation_p=False,
        )

