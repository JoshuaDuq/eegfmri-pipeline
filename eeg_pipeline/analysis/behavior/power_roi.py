"""Power ROI correlation analysis."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from collections import defaultdict

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import load_settings, get_frequency_band_names, get_frequency_bands_for_aperiodic
from eeg_pipeline.utils.data.loading import extract_roi_columns, build_covariate_matrix, build_covariates_without_temp
from eeg_pipeline.utils.io.general import (
    deriv_stats_path, deriv_features_path, ensure_dir, get_subject_logger,
    format_band_range, write_tsv, read_tsv, build_partial_covars_string,
)
from eeg_pipeline.utils.analysis.stats import (
    get_correlation_method, compute_temp_correlations_for_roi, _safe_float,
    compute_channel_rating_correlations, prepare_aligned_data,
)
from eeg_pipeline.analysis.behavior.core import save_correlation_results, build_correlation_record
from eeg_pipeline.utils.analysis.tfr import build_rois_from_info as _build_rois, get_summary_type
from eeg_pipeline.analysis.behavior.correlations import (
    AnalysisConfig, _align_groups_to_series, _build_temp_record_unified, _compute_roi_correlation_stats,
)

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext


def _build_roi_rating_record(roi: str, band: str, band_range_str: str, r: float, p: float,
                             n: int, method: str, ci_low: float, ci_high: float,
                             r_part: float, p_part: float, n_part: int, partial_str: str,
                             r_part_temp: float, p_part_temp: float, n_part_temp: int,
                             p_perm: float, p_partial_perm: float, p_partial_temp_perm: float, n_perm: int) -> Dict:
    return build_correlation_record(
        roi, band, r, p, n, method, ci_low=ci_low, ci_high=ci_high,
        r_partial=r_part, p_partial=p_part, n_partial=n_part, p_perm=p_perm,
        p_partial_perm=p_partial_perm, r_partial_temp=r_part_temp, p_partial_temp=p_part_temp,
        n_partial_temp=n_part_temp, p_partial_temp_perm=p_partial_temp_perm,
        identifier_type="roi", analysis_type="power", band_range=band_range_str,
        partial_covars=partial_str, n_perm=n_perm,
    ).to_dict()


def _build_channel_rating_record(ch: str, band: str, r: float, p: float, ci_low: float,
                                  ci_high: float, r_part: float, p_part: float, n_part: int,
                                  p_perm: float, p_partial_perm: float, n: int, method: str) -> Dict:
    return build_correlation_record(
        ch, band, r, p, n, method, ci_low=ci_low, ci_high=ci_high,
        r_partial=r_part, p_partial=p_part, n_partial=n_part,
        p_perm=p_perm, p_partial_perm=p_partial_perm,
        identifier_type="channel", analysis_type="power",
    ).to_dict()


def _build_temp_record_for_roi(roi_vals, temp_series, cov_no_temp, band, roi, band_range_str, cfg, groups=None):
    if (cfg.bootstrap and cfg.bootstrap > 0) or (cfg.n_perm and cfg.n_perm > 0):
        rec = compute_temp_correlations_for_roi(
            roi_vals, temp_series, cov_no_temp, cfg.bootstrap, cfg.n_perm, cfg.use_spearman,
            cfg.method, cfg.rng, band, roi, cfg.logger, config=cfg.config, groups=groups,
        )
        if rec:
            rec["band_range"] = band_range_str
        return rec
    return _build_temp_record_unified(roi_vals, temp_series, cov_no_temp, roi, "roi", band, cfg, groups, band_range=band_range_str)


def _process_single_roi_for_band(roi, channels, band, band_cols, pow_df, target, temp_series,
                                  cov_df, cov_no_temp, band_range_str, cfg, me_recs=None):
    if pow_df is None or target is None:
        return None, None
    if hasattr(pow_df, 'empty') and pow_df.empty:
        return None, None
    if hasattr(target, 'empty') and target.empty:
        return None, None
    
    roi_cols = extract_roi_columns(roi, channels, band, band_cols)
    if roi_cols is None:
        return None, None
    
    roi_vals = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    x, y, cov, _, n = prepare_aligned_data(roi_vals, target, cov_df)
    if len(x) == 0 or len(y) == 0:
        return None, None
    
    groups = _align_groups_to_series(x, cfg.groups)
    stats = _compute_roi_correlation_stats(x, y, x, y, cov, temp_series, n, band, roi,
                                           f"ROI {roi} ({band})", cfg, groups=groups, me_records=me_recs)
    
    rating_rec = _build_roi_rating_record(
        roi, band, band_range_str, stats.correlation, stats.p_value, n, cfg.method,
        stats.ci_low, stats.ci_high, stats.r_partial, stats.p_partial, stats.n_partial,
        build_partial_covars_string(cov_df), stats.r_partial_temp, stats.p_partial_temp,
        stats.n_partial_temp, stats.p_perm, stats.p_partial_perm, stats.p_partial_temp_perm, cfg.n_perm,
    )
    temp_rec = _build_temp_record_for_roi(roi_vals, temp_series, cov_no_temp, band, roi, band_range_str, cfg, groups)
    return rating_rec, temp_rec


def _save_roi_results(rating_recs, temp_recs, cfg):
    if rating_recs:
        df = pd.DataFrame(rating_recs)
        save_correlation_results(df, cfg.stats_dir / "corr_stats_pow_roi_vs_rating.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger,
                                use_permutation_p=True)
        save_correlation_results(df, cfg.stats_dir / "corr_stats_pow_combined_vs_rating.tsv",
                                apply_fdr=False, config=cfg.config, logger=cfg.logger)
    if temp_recs:
        df = pd.DataFrame(temp_recs)
        save_correlation_results(df, cfg.stats_dir / "corr_stats_pow_roi_vs_temp.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger)
        save_correlation_results(df, cfg.stats_dir / "corr_stats_pow_combined_vs_temp.tsv",
                                apply_fdr=False, config=cfg.config, logger=cfg.logger)


def _process_channel_level(power_bands, pow_df, target, cov_df, temp_series, cov_no_temp, cfg):
    for band in power_bands:
        cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
        if not cols:
            continue
        rating_recs, temp_recs = [], []
        for col in cols:
            ch = col.replace(f"pow_{band}_", "")
            vals = pd.to_numeric(pow_df[col], errors="coerce")
            x, y, cov, _, _ = prepare_aligned_data(vals, target, cov_df)
            if len(x) == 0 or len(y) == 0:
                continue
            groups = _align_groups_to_series(x, cfg.groups)
            r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm = compute_channel_rating_correlations(
                x, y, cov, cfg.bootstrap, cfg.n_perm, cfg.use_spearman, cfg.method, cfg.rng,
                logger=cfg.logger, config=cfg.config, groups=groups,
            )
            rating_recs.append(_build_channel_rating_record(ch, band, r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm, len(x), cfg.method))
            temp_rec = _build_temp_record_unified(vals, temp_series, cov_no_temp, ch, "channel", band, cfg, groups)
            if temp_rec:
                temp_recs.append(temp_rec)
        
        if rating_recs:
            save_correlation_results(pd.DataFrame(rating_recs), cfg.stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv",
                                    apply_fdr=True, config=cfg.config, logger=cfg.logger, use_permutation_p=True)
        if temp_recs:
            save_correlation_results(pd.DataFrame(temp_recs), cfg.stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv",
                                    apply_fdr=True, config=cfg.config, logger=cfg.logger)


def _run_power_roi_correlations(subject, task, config, logger, deriv_root, stats_dir, pow_df, conn_df,
                                y, info, events, temp_series, temp_col, cov_df, cov_no_temp, use_spearman,
                                bootstrap, n_perm, rng):
    from eeg_pipeline.analysis.behavior.connectivity import compute_sliding_state_metrics
    from eeg_pipeline.analysis.behavior.specialized_features import (
        _process_itpc_correlations, _process_aperiodic_correlations, _process_pac_correlations,
    )
    
    logger.info(f"Starting ROI power correlation analysis for sub-{subject}")
    ensure_dir(stats_dir)
    if pow_df is None or y is None:
        return
    
    # Extract sub-dataframes
    itpc_df = pow_df[[c for c in pow_df.columns if str(c).startswith("itpc_")]].copy()
    aper_df = pow_df[[c for c in pow_df.columns if str(c).startswith(("aper_slope_", "aper_offset_", "powcorr_"))]].copy()
    pow_df = pow_df[[c for c in pow_df.columns if str(c).startswith("pow_")]]
    
    group_col = next((c for c in ("run_id", "run", "block") if events is not None and c in events.columns), None)
    groups = events[group_col] if group_col else None
    
    cfg = AnalysisConfig(
        subject=subject, config=config, logger=logger, rng=rng, bootstrap=bootstrap, n_perm=n_perm,
        use_spearman=use_spearman, method=get_correlation_method(use_spearman),
        min_samples_channel=config.get("behavior_analysis.statistics.min_samples_channel", 10),
        min_samples_roi=config.get("behavior_analysis.statistics.min_samples_roi", 20), groups=groups,
    )
    cfg.stats_dir = stats_dir
    
    roi_map = _build_rois(info, config=config) if info else {}
    power_bands = get_frequency_band_names(config)
    freq_bands = config.get("time_frequency_analysis.bands", {})
    freq_bands_aper = get_frequency_bands_for_aperiodic(config)
    
    rating_recs, temp_recs, me_recs = [], [], []
    missing = defaultdict(list)
    
    for band in power_bands:
        band_cols = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols:
            continue
        for roi, channels in roi_map.items():
            rat, tmp = _process_single_roi_for_band(roi, channels, band, band_cols, pow_df, y, temp_series,
                                                     cov_df, cov_no_temp, format_band_range(band, freq_bands), cfg, me_recs)
            if rat is None:
                missing[band].append(roi)
            else:
                rating_recs.append(rat)
                if tmp:
                    temp_recs.append(tmp)
    
    if missing:
        logger.warning(f"Skipped ROIs: {'; '.join(f'{b}: {sorted(set(r))}' for b, r in missing.items())}")
    
    _save_roi_results(rating_recs, temp_recs, cfg)
    if me_recs:
        write_tsv(pd.DataFrame(me_recs), cfg.stats_dir / "mixed_effects_fixed_effects.tsv")
    
    _process_channel_level(power_bands, pow_df, y, cov_df, temp_series, cov_no_temp, cfg)
    _process_itpc_correlations(itpc_df, y, cov_df, temp_series, cov_no_temp, cfg)
    _process_aperiodic_correlations(aper_df, roi_map, y, temp_series, cov_df, cov_no_temp, freq_bands_aper, cfg)
    
    pac_path = deriv_features_path(deriv_root, subject) / "features_pac_trials.tsv"
    pac_df = read_tsv(pac_path) if pac_path.exists() else None
    _process_pac_correlations(pac_df, events, cfg, temp_series, cov_df, cov_no_temp)
    
    compute_sliding_state_metrics(subject, task, conn_df, events, deriv_root, config, logger)


def compute_power_roi_stats(subject: str, deriv_root: Path, task: Optional[str] = None,
                            use_spearman: bool = True, partial_covars: Optional[List[str]] = None,
                            bootstrap: int = 0, n_perm: int = 0, rng: Optional[np.random.Generator] = None) -> None:
    from eeg_pipeline.pipelines.behavior import initialize_analysis_context
    from eeg_pipeline.analysis.behavior.core import BehaviorContext
    
    if not subject:
        return
    try:
        config, task, deriv_root, stats_dir, logger = initialize_analysis_context(subject, task, None)
    except ValueError:
        return
    
    rng = rng or np.random.default_rng(config.get("project.random_state", 42))
    ctx = BehaviorContext(subject=subject, task=task, config=config, logger=logger, deriv_root=deriv_root,
                          stats_dir=stats_dir, use_spearman=use_spearman, bootstrap=bootstrap,
                          n_perm=n_perm, rng=rng, partial_covars=partial_covars)
    if ctx.load_data():
        compute_power_roi_stats_from_context(ctx)


def compute_power_roi_stats_from_context(ctx: "BehaviorContext") -> None:
    _run_power_roi_correlations(
        ctx.subject, ctx.task, ctx.config, ctx.logger, ctx.deriv_root, ctx.stats_dir,
        ctx.power_df, ctx.connectivity_df, ctx.targets, ctx.epochs_info, ctx.aligned_events,
        ctx.temperature, ctx.temperature_column, ctx.covariates_df, ctx.covariates_without_temp_df,
        ctx.use_spearman, ctx.bootstrap, ctx.n_perm, ctx.rng,
    )
