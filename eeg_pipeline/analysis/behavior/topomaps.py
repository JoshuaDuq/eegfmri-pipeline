import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import (
    _load_features_and_targets,
    _pick_first_column,
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    ensure_dir,
    get_subject_logger,
    write_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    get_correlation_method,
    compute_correlation,
    get_fdr_alpha_from_config,
    get_eeg_adjacency,
    compute_cluster_masses_1d,
    compute_topomap_permutation_masses,
    compute_cluster_pvalues_1d,
    bh_adjust as _bh_adjust,
    _safe_float,
)
from eeg_pipeline.analysis.behavior.correlations import AnalysisConfig


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


def _process_single_topomap_band(
    band: str,
    pow_df: pd.DataFrame,
    temp_series: pd.Series,
    analysis_cfg: AnalysisConfig,
    cluster_params: Dict[str, Any],
    eeg_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    band_cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
    if not band_cols:
        analysis_cfg.logger.debug(f"No channel-level power columns for band '{band}'")
        return []

    ch_names = [col.replace(f"pow_{band}_", "") for col in band_cols]
    n_channels = len(ch_names)

    correlations = np.full(n_channels, np.nan)
    p_values = np.full(n_channels, np.nan)
    n_valid = np.zeros(n_channels, dtype=int)
    channel_data = np.full((n_channels, len(pow_df)), np.nan, dtype=float)

    min_valid_points = analysis_cfg.min_samples_roi

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
        
        correlation, p_value = compute_correlation(x_valid, temp_valid, analysis_cfg.use_spearman)
        correlations[ch_idx] = correlation
        p_values[ch_idx] = p_value

    valid_mask = np.isfinite(p_values) & (n_valid >= min_valid_points)
    if not np.any(valid_mask):
        analysis_cfg.logger.debug(f"No valid correlations for band '{band}' (need >= {min_valid_points} valid points per channel)")
        return []

    p_corrected = np.full_like(p_values, np.nan)
    if np.any(valid_mask):
        p_corrected[valid_mask] = _bh_adjust(p_values[valid_mask])
    significant_mask = (p_corrected < cluster_params["alpha"]) & valid_mask

    # Cluster correction
    cluster_labels = np.zeros(n_channels, dtype=int)
    cluster_pvals = np.full(n_channels, np.nan)
    cluster_sig_mask = np.zeros(n_channels, dtype=bool)
    cluster_records = []
    
    eeg_ch_names = [eeg_info["info"]['ch_names'][i] for i in eeg_info["picks"]]
    ch_to_eeg_idx = {
        ch_idx: eeg_ch_names.index(ch_name)
        for ch_idx, ch_name in enumerate(ch_names)
        if ch_name in eeg_ch_names
    }

    min_channels_adj = cluster_params["min_channels_for_adjacency"]
    if len(ch_to_eeg_idx) < min_channels_adj:
        analysis_cfg.logger.debug(f"Insufficient channels with adjacency for band '{band}' ({len(ch_to_eeg_idx)} < {min_channels_adj})")
    else:
        cluster_labels_obs, cluster_masses = compute_cluster_masses_1d(
            correlations,
            p_values,
            cluster_params["cluster_alpha"],
            ch_to_eeg_idx,
            eeg_info["picks"],
            eeg_info["adjacency"],
        )
        if cluster_masses:
            cluster_labels = cluster_labels_obs
            perm_max_masses = compute_topomap_permutation_masses(
                channel_data,
                temp_series,
                n_channels,
                cluster_params["n_cluster_perm"],
                cluster_params["cluster_alpha"],
                min_valid_points,
                analysis_cfg.use_spearman,
                ch_to_eeg_idx,
                eeg_info["picks"],
                eeg_info["adjacency"],
                cluster_params["cluster_rng"],
            )
            cluster_pvals, cluster_sig_mask, cluster_records = (
                compute_cluster_pvalues_1d(
                    cluster_labels_obs, cluster_masses, perm_max_masses, cluster_params["alpha"]
                )
            )

    records = []
    _append_channel_records_with_clusters(
        records, band, ch_names, correlations, p_values,
        p_corrected, significant_mask, cluster_labels, cluster_pvals,
        cluster_sig_mask, n_valid, analysis_cfg.method, n_channels
    )
    
    if cluster_records:
        n_significant = sum(1 for r in cluster_records if r["p_value"] <= cluster_params["alpha"])
        analysis_cfg.logger.info(f"Band {band}: {len(cluster_records)} clusters, {n_significant} significant")

    return records


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

    adjacency, eeg_picks, _ = get_eeg_adjacency(info)
    if adjacency is None or eeg_picks is None:
        logger.warning("Could not compute EEG adjacency; skipping cluster-corrected topomap correlations")
        return

    power_bands = config.get("features.frequency_bands", ["delta", "theta", "alpha", "beta", "gamma"])
    alpha = get_fdr_alpha_from_config(config)
    
    cluster_cfg = config.get("behavior_analysis.cluster_correction", {})
    cluster_n_perm_config = cluster_cfg.get("n_permutations", 100)
    cluster_params = {
        "alpha": float(alpha),
        "cluster_alpha": float(cluster_cfg.get("alpha", alpha)),
        "n_cluster_perm": int(cluster_n_perm_config) if cluster_n_perm_config > 0 else 0,
        "cluster_rng": np.random.default_rng(int(cluster_cfg.get("rng_seed", config.get("random.seed", 42)))),
        "min_channels_for_adjacency": config.get("behavior_analysis.statistics.min_channels_for_adjacency", 2)
    }

    method = get_correlation_method(use_spearman)
    min_valid_points = config.get("behavior_analysis.statistics.min_samples_roi", 5)
    
    analysis_cfg = AnalysisConfig(
        subject=subject,
        config=config,
        logger=logger,
        rng=rng,
        stats_dir=stats_dir,
        bootstrap=bootstrap,
        n_perm=n_perm,
        use_spearman=use_spearman,
        method=method,
        min_samples_roi=min_valid_points
    )

    eeg_info = {"info": info, "adjacency": adjacency, "picks": eeg_picks}
    all_records: List[Dict[str, Any]] = []

    for band in power_bands:
        band_records = _process_single_topomap_band(band, pow_df, temp_series, analysis_cfg, cluster_params, eeg_info)
        all_records.extend(band_records)

    if not all_records:
        logger.warning(
            f"No topomap correlation records generated for sub-{subject}. "
            f"Possible reasons: no power columns found for bands {power_bands}, "
            f"or insufficient valid data points (need >= {min_valid_points} per channel)"
        )
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
        "alpha_bh": cluster_params["alpha"],
        "cluster_alpha": cluster_params["cluster_alpha"],
        "n_cluster_permutations": cluster_params["n_cluster_perm"],
        "cluster_rng_seed": int(cluster_cfg.get("rng_seed", config.get("random.seed", 42))),
        "n_bh_significant_channels": int(df_results["significant"].sum()),
        "n_cluster_significant_channels": int(df_results["cluster_significant"].sum()),
    }
    meta_path = stats_dir / f"power_topomap_temperature_meta{method_suffix}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(cluster_summary, fh, indent=2)
    logger.info(f"Saved topomap temperature correlation metadata to {meta_path}")

