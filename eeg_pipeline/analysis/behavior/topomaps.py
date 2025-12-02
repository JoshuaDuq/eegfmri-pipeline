"""Power topomap correlations with temperature and cluster correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings, get_frequency_band_names
from eeg_pipeline.utils.data.loading import _load_features_and_targets, _pick_first_column, load_epochs_for_analysis
from eeg_pipeline.utils.io.general import deriv_stats_path, ensure_dir, get_subject_logger, write_tsv
from eeg_pipeline.utils.analysis.stats import (
    get_correlation_method, compute_correlation, get_fdr_alpha_from_config, get_eeg_adjacency,
    compute_cluster_masses_1d, compute_topomap_permutation_masses, compute_cluster_pvalues_1d,
    bh_adjust as _bh_adjust, _safe_float,
)
from eeg_pipeline.analysis.behavior.correlations import AnalysisConfig

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext


def _process_band(band: str, pow_df: pd.DataFrame, temp: pd.Series, cfg: AnalysisConfig,
                  cluster_params: Dict, eeg_info: Dict) -> List[Dict]:
    cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
    if not cols:
        return []
    
    ch_names = [c.replace(f"pow_{band}_", "") for c in cols]
    n_ch = len(ch_names)
    min_pts = cfg.min_samples_roi
    
    corrs, pvals, n_valid = np.full(n_ch, np.nan), np.full(n_ch, np.nan), np.zeros(n_ch, dtype=int)
    ch_data = np.full((n_ch, len(pow_df)), np.nan)
    
    for i, col in enumerate(cols):
        s = pd.to_numeric(pow_df[col], errors="coerce")
        ch_data[i] = s.values
        mask = s.notna() & temp.notna()
        n_valid[i] = int(mask.sum())
        if n_valid[i] >= min_pts:
            corrs[i], pvals[i] = compute_correlation(s[mask], temp[mask], cfg.use_spearman)
    
    valid = np.isfinite(pvals) & (n_valid >= min_pts)
    if not valid.any():
        return []
    
    p_corr = np.full_like(pvals, np.nan)
    p_corr[valid] = _bh_adjust(pvals[valid])
    sig = (p_corr < cluster_params["alpha"]) & valid
    
    c_labels, c_pvals, c_sig = np.zeros(n_ch, dtype=int), np.full(n_ch, np.nan), np.zeros(n_ch, dtype=bool)
    
    eeg_names = [eeg_info["info"]['ch_names'][i] for i in eeg_info["picks"]]
    ch_map = {i: eeg_names.index(n) for i, n in enumerate(ch_names) if n in eeg_names}
    
    if len(ch_map) >= cluster_params["min_channels_for_adjacency"]:
        labels, masses = compute_cluster_masses_1d(corrs, pvals, cluster_params["cluster_alpha"],
                                                   ch_map, eeg_info["picks"], eeg_info["adjacency"])
        if masses:
            c_labels = labels
            perm_masses = compute_topomap_permutation_masses(
                ch_data, temp, n_ch, cluster_params["n_cluster_perm"], cluster_params["cluster_alpha"],
                min_pts, cfg.use_spearman, ch_map, eeg_info["picks"], eeg_info["adjacency"], cluster_params["cluster_rng"]
            )
            c_pvals, c_sig, _ = compute_cluster_pvalues_1d(labels, masses, perm_masses, cluster_params["alpha"])
    
    return [{
        "band": band, "channel": ch_names[i], "correlation": _safe_float(corrs[i]),
        "p_value": _safe_float(pvals[i]), "p_corrected": _safe_float(p_corr[i]), "significant": bool(sig[i]),
        "cluster_id": int(c_labels[i]), "cluster_p": _safe_float(c_pvals[i]), "cluster_significant": bool(c_sig[i]),
        "n_valid": int(n_valid[i]), "method": cfg.method,
    } for i in range(n_ch)]


def _run_topomap_correlations(subject: str, task: str, pow_df: pd.DataFrame, temp: pd.Series,
                              info, stats_dir: Path, config, logger, use_spearman: bool, rng,
                              bootstrap: int = 0, n_perm: int = 0) -> None:
    if pow_df is None or pow_df.empty or temp is None or temp.isna().all():
        logger.warning("Missing power or temperature data; skipping topomap correlations")
        return
    
    adj, picks, _ = get_eeg_adjacency(info)
    if adj is None:
        logger.warning("No EEG adjacency; skipping topomap correlations")
        return
    
    bands = get_frequency_band_names(config)
    alpha = get_fdr_alpha_from_config(config)
    c_cfg = config.get("behavior_analysis.cluster_correction", {})
    
    cluster_params = {
        "alpha": float(alpha),
        "cluster_alpha": float(c_cfg.get("alpha", alpha)),
        "n_cluster_perm": int(c_cfg.get("n_permutations", 100)),
        "cluster_rng": rng or np.random.default_rng(config.get("project.random_state", 42)),
        "min_channels_for_adjacency": config.get("behavior_analysis.statistics.min_channels_for_adjacency", 2),
    }
    
    cfg = AnalysisConfig(subject=subject, config=config, logger=logger, rng=rng, stats_dir=stats_dir,
                         bootstrap=bootstrap, n_perm=n_perm, use_spearman=use_spearman,
                         method=get_correlation_method(use_spearman),
                         min_samples_roi=config.get("behavior_analysis.statistics.min_samples_roi", 5))
    
    eeg_info = {"info": info, "adjacency": adj, "picks": picks}
    records = []
    for band in bands:
        records.extend(_process_band(band, pow_df, temp, cfg, cluster_params, eeg_info))
    
    if not records:
        logger.warning(f"No topomap correlations for sub-{subject}")
        return
    
    df = pd.DataFrame(records)
    sfx = "_spearman" if use_spearman else "_pearson"
    write_tsv(df, stats_dir / f"power_topomap_temperature_correlations{sfx}.tsv")
    
    summary = {"subject": subject, "task": task, "method": cfg.method, "n_bands": len(bands),
               "n_channels": len(set(df["channel"])), "n_bh_sig": int(df["significant"].sum()),
               "n_cluster_sig": int(df["cluster_significant"].sum())}
    with open(stats_dir / f"power_topomap_temperature_meta{sfx}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {len(df)} topomap correlations")


def correlate_power_topomaps(subject: str, task: Optional[str] = None, use_spearman: bool = True,
                             bootstrap: int = 0, n_perm: int = 0, rng=None) -> None:
    """Correlate power topomaps with temperature (standalone entry point)."""
    if not subject:
        return
    config = load_settings()
    task = task or config.get("project.task", "thermalactive")
    logger = get_subject_logger("behavior_analysis", subject, config.get("logging.log_file_name"), config=config)
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    rng = rng or np.random.default_rng(config.get("project.random_state", 42))
    
    _, pow_df, _, _, info = _load_features_and_targets(subject, task, deriv_root, config)
    epochs, events = load_epochs_for_analysis(subject, task, align="strict", preload=False,
                                              deriv_root=deriv_root, bids_root=config.bids_root, config=config, logger=logger)
    temp = None
    if events is not None:
        t_col = _pick_first_column(events, config.get("event_columns.temperature", []))
        if t_col:
            temp = pd.to_numeric(events[t_col], errors="coerce")
    
    _run_topomap_correlations(subject, task, pow_df, temp, info, stats_dir, config, logger, use_spearman, rng, bootstrap, n_perm)


def correlate_power_topomaps_from_context(ctx: "BehaviorContext") -> None:
    """Correlate power topomaps using pre-loaded context data."""
    _run_topomap_correlations(ctx.subject, ctx.task, ctx.power_df, ctx.temperature, ctx.epochs_info,
                              ctx.stats_dir, ctx.config, ctx.logger, ctx.use_spearman, ctx.rng, ctx.bootstrap, ctx.n_perm)
