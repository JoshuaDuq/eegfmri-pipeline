"""Precomputed feature correlations with behavioral measures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import deriv_features_path, read_tsv
from eeg_pipeline.analysis.behavior.core import (
    BehaviorContext, ComputationResult, ComputationStatus, CorrelationRecord,
    build_correlation_record, safe_correlation, save_correlation_results,
    correlate_features_loop, MIN_SAMPLES_DEFAULT,
)


FEATURE_PATTERNS = {
    "erds": re.compile(r"^erds_(\w+)_(.+)$"),
    "erds_windowed": re.compile(r"^erds_(\w+)_(\w+)_(early|mid|late|t\d+)$"),
    "iaf": re.compile(r"^iaf_(.+)$"),
    "iaf_power": re.compile(r"^iaf_power_(.+)$"),
    "relative_power": re.compile(r"^relative_(\w+)_(.+)$"),
    "band_ratio": re.compile(r"^ratio_(\w+)_(\w+)_(.+)$"),
    "spectral_entropy": re.compile(r"^spectral_entropy_(.+)$"),
    "spectral_edge": re.compile(r"^spectral_edge_(\d+)_(.+)$"),
    "peak_freq": re.compile(r"^peak_freq_(\w+)_(.+)$"),
    "bandwidth": re.compile(r"^bandwidth_(\w+)_(.+)$"),
    "temporal_stat": re.compile(r"^(mean|var|std|skew|kurt|median|iqr)_(.+)$"),
    "amplitude": re.compile(r"^(rms|p2p|line_length|nle)_(.+)$"),
    "zero_cross": re.compile(r"^zero_cross_(.+)$"),
    "slope": re.compile(r"^slope_(.+)$"),
    "permutation_entropy": re.compile(r"^pe_(.+)$"),
    "sample_entropy": re.compile(r"^sampen_(.+)$"),
    "hjorth": re.compile(r"^hjorth_(activity|mobility|complexity)_(.+)$"),
    "lzc": re.compile(r"^lzc_(.+)$"),
    "hurst": re.compile(r"^hurst_(.+)$"),
    "dfa": re.compile(r"^dfa_(.+)$"),
    "gfp": re.compile(r"^gfp_(mean|std|max|peak_count|peak_rate)$"),
    "global_plv": re.compile(r"^global_plv_(\w+)$"),
    "variance_explained": re.compile(r"^var_explained_(.+)$"),
    "roi_power": re.compile(r"^roi_pow_(\w+)_(.+)$"),
    "roi_asymmetry": re.compile(r"^asymmetry_(\w+)_(.+)$"),
    "roi_erds": re.compile(r"^roi_erds_(\w+)_(.+)$"),
    "roi_laterality": re.compile(r"^laterality_(\w+)_(.+)$"),
    "ms_coverage": re.compile(r"^ms_coverage_(\w+)$"),
    "ms_duration": re.compile(r"^ms_duration_(\w+)$"),
    "ms_occurrence": re.compile(r"^ms_occurrence_(\w+)$"),
    "ms_transition": re.compile(r"^ms_transition_(\w+)_(\w+)$"),
    "ms_gev": re.compile(r"^ms_gev_(\w+)$"),
    "ms_entropy": re.compile(r"^ms_entropy$"),
    "itpc": re.compile(r"^itpc_(\w+)_(.+)_(\w+)$"),
    "plv": re.compile(r"^plv_(\w+)_(.+)_(.+)$"),
    "pac": re.compile(r"^pac_(.+)$"),
    "aperiodic": re.compile(r"^aper_(slope|offset|knee)_(.+)$"),
    "powcorr": re.compile(r"^powcorr_(\w+)_(.+)$"),
    "conn_graph": re.compile(r"^(wpli|aec|imcoh|pli)_(\w+)_(geff|clust|pc|smallworld)$"),
}


def classify_feature(column_name: str) -> Tuple[str, Dict[str, str]]:
    """Classify feature column into type and metadata."""
    for ftype, pattern in FEATURE_PATTERNS.items():
        m = pattern.match(column_name)
        if not m:
            continue
        g = m.groups()
        
        if ftype == "erds":
            return ftype, {"band": g[0], "identifier": g[1], "channel": g[1]}
        elif ftype == "erds_windowed":
            return "erds", {"band": g[0], "channel": g[1], "window": g[2], "identifier": f"{g[1]}_{g[2]}"}
        elif ftype == "relative_power":
            return ftype, {"band": g[0], "channel": g[1], "identifier": g[1]}
        elif ftype == "band_ratio":
            return ftype, {"band": f"{g[0]}/{g[1]}", "channel": g[2], "identifier": g[2]}
        elif ftype in ("temporal_stat", "amplitude"):
            return ftype, {"stat": g[0], "channel": g[1], "identifier": g[1], "band": "N/A"}
        elif ftype == "hjorth":
            return ftype, {"param": g[0], "channel": g[1], "identifier": g[1], "band": "N/A"}
        elif ftype == "roi_power":
            return ftype, {"band": g[0], "roi": g[1], "identifier": g[1]}
        elif ftype in ("roi_asymmetry", "roi_laterality"):
            return ftype, {"band": g[0], "pair": g[1], "identifier": g[1]}
        elif ftype == "ms_transition":
            return "microstate", {"from_state": g[0], "to_state": g[1], "identifier": f"{g[0]}->{g[1]}", "band": "N/A"}
        elif ftype.startswith("ms_"):
            return "microstate", {"state": g[0], "identifier": g[0], "band": "N/A"}
        elif ftype == "itpc":
            return ftype, {"band": g[0], "channel": g[1], "time_bin": g[2], "identifier": g[1]}
        elif ftype == "aperiodic":
            return ftype, {"param": g[0], "channel": g[1], "identifier": g[1], "band": "aperiodic"}
        elif ftype == "powcorr":
            return ftype, {"band": g[0], "channel": g[1], "identifier": g[1]}
        elif ftype == "conn_graph":
            return "connectivity", {"measure": g[0], "band": g[1], "metric": g[2], "identifier": f"{g[0]}_{g[2]}"}
        elif ftype == "gfp":
            return ftype, {"metric": g[0], "identifier": g[0], "band": "global"}
        else:
            return ftype, {"identifier": g[0] if g else column_name, "band": "N/A"}

    # Fallback for pow_* columns
    if column_name.startswith("pow_"):
        parts = column_name.split("_")
        if len(parts) >= 3:
            return "power", {"band": parts[1], "channel": "_".join(parts[2:]), "identifier": "_".join(parts[2:])}
    return "unknown", {"identifier": column_name, "band": "N/A"}


def _correlate_df(df: pd.DataFrame, target: pd.Series, config: Any, logger, use_spearman: bool,
                  source: str) -> Tuple[pd.DataFrame, List[CorrelationRecord]]:
    """Correlate all columns with target."""
    method = "spearman" if use_spearman else "pearson"
    min_samples = int(config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    # Coerce to numeric to avoid type issues in correlation (non-numeric -> NaN)
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    records, df_out = correlate_features_loop(df_numeric, target, method, min_samples, logger,
                                   identifier_type="feature", analysis_type=source,
                                   feature_classifier=classify_feature)
    if not df_out.empty:
        # Rename correlation/p columns for plotting compatibility
        df_out = df_out.rename(columns={"r": "correlation", "p": "p_value"})
    return df_out, records


def correlate_precomputed_features(subject: str, deriv_root: Path, target: pd.Series,
                                   config: Any, logger, use_spearman: bool = True) -> Tuple[pd.DataFrame, List]:
    """Correlate precomputed features with behavior."""
    path = deriv_features_path(deriv_root, subject) / "features_precomputed.tsv"
    if not path.exists():
        return pd.DataFrame(), []
    df = read_tsv(path)
    if df.empty:
        return pd.DataFrame(), []
    
    n = min(len(df), len(target))
    df, target = df.iloc[:n], target.iloc[:n]
    
    method = "spearman" if use_spearman else "pearson"
    min_samples = int(config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    target_arr = target.to_numpy()
    records = []
    
    logger.info(f"Correlating {len(df.columns)} precomputed features...")
    for col in df.columns:
        ftype, meta = classify_feature(col)
        r, p, n_val = safe_correlation(df[col].to_numpy(), target_arr, method, min_samples)
        try:
            r_float = float(r) if r is not None else np.nan
        except (ValueError, TypeError):
            r_float = np.nan
        if np.isfinite(r_float):
            records.append(build_correlation_record(
                meta.get("identifier", col), meta.get("band", "N/A"), r_float, p, n_val, method,
                identifier_type=ftype, analysis_type=ftype
            ))
    
    logger.info(f"  {len(records)} features, {sum(1 for r in records if r.is_significant)} sig")
    return pd.DataFrame([r.to_dict() for r in records]) if records else pd.DataFrame(), records


def correlate_microstate_features(subject: str, deriv_root: Path, target: pd.Series,
                                  config: Any, logger, use_spearman: bool = True) -> Tuple[pd.DataFrame, List]:
    """Correlate microstate features with behavior."""
    path = deriv_features_path(deriv_root, subject) / "features_microstates.tsv"
    if not path.exists():
        return pd.DataFrame(), []
    df = read_tsv(path)
    if df.empty:
        return pd.DataFrame(), []
    
    n = min(len(df), len(target))
    df, target = df.iloc[:n], target.iloc[:n]
    
    method = "spearman" if use_spearman else "pearson"
    min_samples = int(config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    target_arr = target.to_numpy()
    records = []
    
    for col in df.columns:
        ftype, meta = classify_feature(col)
        r, p, n_val = safe_correlation(df[col].to_numpy(), target_arr, method, min_samples)
        try:
            r_float = float(r) if r is not None else np.nan
        except (ValueError, TypeError):
            r_float = np.nan
        if np.isfinite(r_float):
            records.append(build_correlation_record(
                meta.get("identifier", col), meta.get("band", "N/A"), r_float, p, n_val, method,
                identifier_type="microstate", analysis_type="microstate"
            ))
    
    logger.info(f"  Microstates: {len(records)} features, {sum(1 for r in records if r.is_significant)} sig")
    return pd.DataFrame([r.to_dict() for r in records]) if records else pd.DataFrame(), records


def compute_precomputed_correlations(ctx: BehaviorContext) -> ComputationResult:
    """Compute correlations for all precomputed features."""
    if ctx.targets is None:
        return ComputationResult(name="precomputed", status=ComputationStatus.SKIPPED,
                                 metadata={"reason": "No target"})
    
    method_suffix = "_spearman" if ctx.use_spearman else "_pearson"
    
    try:
        # Precomputed features
        if ctx.precomputed_df is not None and not ctx.precomputed_df.empty:
            precomp_df, precomp_recs = _correlate_df(ctx.precomputed_df, ctx.targets, ctx.config,
                                                      ctx.logger, ctx.use_spearman, "precomputed")
        else:
            precomp_df, precomp_recs = correlate_precomputed_features(
                ctx.subject, ctx.deriv_root, ctx.targets, ctx.config, ctx.logger, ctx.use_spearman)
        
        # Microstates
        if ctx.microstates_df is not None and not ctx.microstates_df.empty:
            ms_df, ms_recs = _correlate_df(ctx.microstates_df, ctx.targets, ctx.config,
                                           ctx.logger, ctx.use_spearman, "microstates")
        else:
            ms_df, ms_recs = correlate_microstate_features(
                ctx.subject, ctx.deriv_root, ctx.targets, ctx.config, ctx.logger, ctx.use_spearman)
        
        # Save
        if precomp_df is not None and not precomp_df.empty:
            save_correlation_results(precomp_df, ctx.stats_dir / f"corr_stats_precomputed_vs_rating{method_suffix}.tsv",
                                    apply_fdr=True, config=ctx.config, logger=ctx.logger, use_permutation_p=False)
        if not ms_df.empty:
            save_correlation_results(ms_df, ctx.stats_dir / f"corr_stats_microstates_vs_rating{method_suffix}.tsv",
                                    apply_fdr=True, config=ctx.config, logger=ctx.logger)
        
        # Temperature
        if ctx.temperature is not None and len(ctx.temperature.dropna()) >= MIN_SAMPLES_DEFAULT:
            ctx.logger.info("Computing temperature correlations...")
            if ctx.precomputed_df is not None and not ctx.precomputed_df.empty:
                temp_df, _ = _correlate_df(ctx.precomputed_df, ctx.temperature, ctx.config,
                                           ctx.logger, ctx.use_spearman, "precomputed")
                if temp_df is not None and not temp_df.empty:
                    save_correlation_results(temp_df, ctx.stats_dir / f"corr_stats_precomputed_vs_temp{method_suffix}.tsv",
                                            apply_fdr=True, config=ctx.config, logger=ctx.logger, use_permutation_p=False)
            if ctx.microstates_df is not None and not ctx.microstates_df.empty:
                temp_ms_df, _ = _correlate_df(ctx.microstates_df, ctx.temperature, ctx.config,
                                              ctx.logger, ctx.use_spearman, "microstates")
                if not temp_ms_df.empty:
                    save_correlation_results(temp_ms_df, ctx.stats_dir / f"corr_stats_microstates_vs_temp{method_suffix}.tsv",
                                            apply_fdr=True, config=ctx.config, logger=ctx.logger)
        
        all_records = precomp_recs + ms_recs
        combined_parts = [df for df in (precomp_df, ms_df) if df is not None and not df.empty]
        combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
        
        return ComputationResult(
            name="precomputed", status=ComputationStatus.SUCCESS, records=all_records, dataframe=combined,
            metadata={"n_precomputed": len(precomp_recs), "n_microstates": len(ms_recs)}
        )
    except Exception as e:
        ctx.logger.error(f"precomputed failed: {e}")
        return ComputationResult(name="precomputed", status=ComputationStatus.FAILED, error=str(e))
