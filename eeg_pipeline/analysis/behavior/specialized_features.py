"""Specialized feature correlations: ITPC, aperiodic, PAC."""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import deriv_features_path, read_tsv, build_partial_covars_string, format_band_range, get_column_from_config
from eeg_pipeline.utils.analysis.stats import prepare_aligned_data, compute_channel_rating_correlations
from eeg_pipeline.analysis.behavior.core import save_correlation_results, build_correlation_record
from eeg_pipeline.analysis.behavior.correlations import AnalysisConfig, _align_groups_to_series, _build_temp_record_unified, _compute_roi_correlation_stats


def _parse_powcorr_column(col: str) -> Optional[Tuple[str, str]]:
    if not isinstance(col, str) or not col.startswith("powcorr_"):
        return None
    parts = col.split("_")
    return (parts[1], "_".join(parts[2:])) if len(parts) >= 3 else None


def _parse_itpc_column(col: str) -> Optional[Tuple[str, str, str]]:
    if not isinstance(col, str) or not col.startswith("itpc_"):
        return None
    parts = col.split("_")
    return (parts[1], parts[-1], "_".join(parts[2:-1])) if len(parts) >= 4 else None


def _build_channel_rec(ch: str, band: str, r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm, n, method, **extra):
    return build_correlation_record(ch, band, r, p, n, method, ci_low=ci_lo, ci_high=ci_hi,
                                    r_partial=r_p, p_partial=p_p, n_partial=n_p,
                                    p_perm=p_perm, p_partial_perm=p_p_perm,
                                    identifier_type="channel", analysis_type=extra.get("analysis", "power"), **extra).to_dict()


def _process_aperiodic_correlations(aper_df, roi_map, target, temp_series, cov_df, cov_no_temp, freq_bands, cfg):
    if aper_df is None or aper_df.empty:
        return

    from eeg_pipeline.analysis.behavior.power_roi import _build_roi_rating_record, _build_temp_record_for_roi

    # Channel-level slopes/offsets
    for metric in ("aper_slope", "aper_offset"):
        cols = [c for c in aper_df.columns if str(c).startswith(f"{metric}_")]
        if not cols:
            continue
        rating_recs, temp_recs = [], []
        for col in cols:
            ch = col.replace(f"{metric}_", "")
            vals = pd.to_numeric(aper_df[col], errors="coerce")
            x, y, cov, _, _ = prepare_aligned_data(vals, target, cov_df)
            if len(x) == 0 or len(y) == 0:
                continue
            groups = _align_groups_to_series(x, cfg.groups)
            r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm = compute_channel_rating_correlations(
                x, y, cov, cfg.bootstrap, cfg.n_perm, cfg.use_spearman, cfg.method, cfg.rng,
                logger=cfg.logger, config=cfg.config, groups=groups,
            )
            rating_recs.append(_build_channel_rec(ch, metric, r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm, len(x), cfg.method))
            tr = _build_temp_record_unified(vals, temp_series, cov_no_temp, ch, "channel", metric, cfg, groups)
            if tr:
                temp_recs.append(tr)
        
        if rating_recs:
            save_correlation_results(pd.DataFrame(rating_recs), cfg.stats_dir / f"corr_stats_{metric}_vs_rating.tsv",
                                    apply_fdr=True, config=cfg.config, logger=cfg.logger, use_permutation_p=True)
        if temp_recs:
            save_correlation_results(pd.DataFrame(temp_recs), cfg.stats_dir / f"corr_stats_{metric}_vs_temp.tsv",
                                    apply_fdr=True, config=cfg.config, logger=cfg.logger)

    # Channel-level residual bands (powcorr)
    powcorr_cols = [c for c in aper_df.columns if str(c).startswith("powcorr_")]
    bands_avail = {b for b, _ in (_parse_powcorr_column(c) or (None, None) for c in powcorr_cols) if b}
    rating_recs, temp_recs = [], []
    
    for col in powcorr_cols:
        parsed = _parse_powcorr_column(col)
        if not parsed:
            continue
        band, ch = parsed
        vals = pd.to_numeric(aper_df[col], errors="coerce")
        x, y, cov, _, _ = prepare_aligned_data(vals, target, cov_df)
        if len(x) == 0 or len(y) == 0:
            continue
        groups = _align_groups_to_series(x, cfg.groups)
        r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm = compute_channel_rating_correlations(
            x, y, cov, cfg.bootstrap, cfg.n_perm, cfg.use_spearman, cfg.method, cfg.rng,
            logger=cfg.logger, config=cfg.config, groups=groups,
        )
        rating_recs.append(_build_channel_rec(ch, f"powcorr_{band}", r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm, len(x), cfg.method))
        tr = _build_temp_record_unified(vals, temp_series, cov_no_temp, ch, "channel", f"powcorr_{band}", cfg, groups)
        if tr:
            temp_recs.append(tr)
    
    if rating_recs:
        save_correlation_results(pd.DataFrame(rating_recs), cfg.stats_dir / "corr_stats_powcorr_vs_rating.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger, use_permutation_p=True)
    if temp_recs:
        save_correlation_results(pd.DataFrame(temp_recs), cfg.stats_dir / "corr_stats_powcorr_vs_temp.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger)

    # ROI-level residual bands
    if roi_map:
        rating_recs, temp_recs = [], []
        for band in sorted(bands_avail):
            band_range_str = format_band_range(band, freq_bands or {})
            for roi, channels in roi_map.items():
                roi_cols = [f"powcorr_{band}_{ch}" for ch in channels if f"powcorr_{band}_{ch}" in aper_df.columns]
                if not roi_cols:
                    continue
                roi_vals = aper_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                x, y, cov, _, n = prepare_aligned_data(roi_vals, target, cov_df)
                if len(x) == 0 or len(y) == 0:
                    continue
                groups = _align_groups_to_series(x, cfg.groups)
                stats = _compute_roi_correlation_stats(x, y, x, y, cov, temp_series, n, band, roi,
                                                       f"powcorr {band} ROI {roi}", cfg, groups=groups)
                rating_recs.append(_build_roi_rating_record(
                    roi, band, band_range_str, stats.correlation, stats.p_value, n, cfg.method,
                    stats.ci_low, stats.ci_high, stats.r_partial, stats.p_partial, stats.n_partial,
                    build_partial_covars_string(cov_df), stats.r_partial_temp, stats.p_partial_temp,
                    stats.n_partial_temp, stats.p_perm, stats.p_partial_perm, stats.p_partial_temp_perm, cfg.n_perm,
                ))
                tr = _build_temp_record_for_roi(roi_vals, temp_series, cov_no_temp, band, roi, band_range_str, cfg, groups)
                if tr:
                    temp_recs.append(tr)
        
        if rating_recs:
            save_correlation_results(pd.DataFrame(rating_recs), cfg.stats_dir / "corr_stats_powcorr_roi_vs_rating.tsv",
                                    apply_fdr=True, config=cfg.config, logger=cfg.logger, use_permutation_p=True)
        if temp_recs:
            save_correlation_results(pd.DataFrame(temp_recs), cfg.stats_dir / "corr_stats_powcorr_roi_vs_temp.tsv",
                                    apply_fdr=True, config=cfg.config, logger=cfg.logger)


def _process_itpc_correlations(itpc_df, target, cov_df, temp_series, cov_no_temp, cfg):
    if itpc_df is None or itpc_df.empty:
        return
    
    rating_recs, temp_recs = [], []
    for col in itpc_df.columns:
        parsed = _parse_itpc_column(str(col))
        if not parsed:
            continue
        band, time_bin, ch = parsed
        vals = pd.to_numeric(itpc_df[col], errors="coerce")
        x, y, cov, _, _ = prepare_aligned_data(vals, target, cov_df)
        if len(x) == 0 or len(y) == 0:
            continue
        groups = _align_groups_to_series(x, cfg.groups)
        r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm = compute_channel_rating_correlations(
            x, y, cov, cfg.bootstrap, cfg.n_perm, cfg.use_spearman, cfg.method, cfg.rng,
            logger=cfg.logger, config=cfg.config, groups=groups,
        )
        rec = build_correlation_record(
            ch, band, r, p, len(x), cfg.method, ci_low=ci_lo, ci_high=ci_hi,
            r_partial=r_p, p_partial=p_p, n_partial=n_p, p_perm=p_perm, p_partial_perm=p_p_perm,
            identifier_type="channel", analysis_type="itpc", time_bin=time_bin,
        ).to_dict()
        rating_recs.append(rec)
        tr = _build_temp_record_unified(vals, temp_series, cov_no_temp, ch, "channel", band, cfg, groups, time_bin=time_bin)
        if tr:
            temp_recs.append(tr)
    
    if rating_recs:
        save_correlation_results(pd.DataFrame(rating_recs), cfg.stats_dir / "corr_stats_itpc_vs_rating.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger, use_permutation_p=True)
    if temp_recs:
        save_correlation_results(pd.DataFrame(temp_recs), cfg.stats_dir / "corr_stats_itpc_vs_temp.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger)


def _process_pac_correlations(pac_df, events_df, cfg, temp_series, cov_df, cov_no_temp):
    if pac_df is None or pac_df.empty or events_df is None or events_df.empty:
        return
    
    rating_col = get_column_from_config(cfg.config, "event_columns.rating", events_df)
    if rating_col is None or rating_col not in events_df.columns:
        return
    
    rating = pd.to_numeric(events_df[rating_col], errors="coerce")
    n_trials = pac_df["trial"].nunique() if "trial" in pac_df.columns else 0
    if n_trials == 0 or len(rating) != n_trials:
        return
    
    rating_recs, temp_recs = [], []
    for (roi, phase_f, amp_f), df_sub in pac_df.groupby(["roi", "phase_freq", "amp_freq"]):
        df_sub = df_sub.sort_values("trial")
        pac_vals = pd.to_numeric(df_sub["pac"], errors="coerce")
        pac_vals.index = df_sub["trial"].values
        
        x, y, cov, _, _ = prepare_aligned_data(pac_vals, rating.reset_index(drop=True), cov_df)
        if len(x) == 0 or len(y) == 0:
            continue
        groups = _align_groups_to_series(x, cfg.groups)
        r, p, ci_lo, ci_hi, r_p, p_p, n_p, p_perm, p_p_perm = compute_channel_rating_correlations(
            x, y, cov, cfg.bootstrap, cfg.n_perm, cfg.use_spearman, cfg.method, cfg.rng,
            logger=cfg.logger, config=cfg.config, groups=groups,
        )
        rec = build_correlation_record(
            roi, f"pac_{phase_f:.1f}_{amp_f:.1f}", r, p, len(x), cfg.method,
            ci_low=ci_lo, ci_high=ci_hi, r_partial=r_p, p_partial=p_p, n_partial=n_p,
            p_perm=p_perm, p_partial_perm=p_p_perm, identifier_type="roi", analysis_type="pac",
            phase_freq=float(phase_f), amp_freq=float(amp_f),
        ).to_dict()
        rating_recs.append(rec)
        tr = _build_temp_record_unified(pac_vals, temp_series, cov_no_temp, roi, "roi",
                                        f"pac_{phase_f:.1f}_{amp_f:.1f}", cfg, groups,
                                        phase_freq=float(phase_f), amp_freq=float(amp_f))
        if tr:
            temp_recs.append(tr)
    
    if rating_recs:
        save_correlation_results(pd.DataFrame(rating_recs), cfg.stats_dir / "corr_stats_pac_vs_rating.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger, use_permutation_p=True)
    if temp_recs:
        save_correlation_results(pd.DataFrame(temp_recs), cfg.stats_dir / "corr_stats_pac_vs_temp.tsv",
                                apply_fdr=True, config=cfg.config, logger=cfg.logger)
