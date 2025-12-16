"""
Power topomap correlations with temperature and cluster correction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema, parse_legacy_power_feature_name
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
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import write_tsv


def _process_band(
    band: str,
    pow_df: pd.DataFrame,
    temp: pd.Series,
    min_samples: int,
    cluster_params: Dict[str, Any],
    eeg_info: Dict[str, Any],
    use_spearman: bool,
) -> List[Dict[str, Any]]:
    """
    Compute per-channel correlations for a single frequency band.
    """
    cols: List[str] = []
    ch_names: List[str] = []
    band_l = str(band).lower()
    for col in pow_df.columns:
        name = str(col)
        parsed = NamingSchema.parse(name)
        if parsed.get("valid") and parsed.get("group") == "power":
            if str(parsed.get("band") or "").lower() != band_l:
                continue
            identifier = parsed.get("identifier")
            if identifier is None:
                continue
            cols.append(name)
            ch_names.append(str(identifier))
            continue

        legacy = parse_legacy_power_feature_name(name)
        if legacy is None:
            continue
        legacy_band, legacy_ch = legacy
        if str(legacy_band).lower() != band_l:
            continue
        cols.append(name)
        ch_names.append(str(legacy_ch))

    if not cols:
        return []

    n_ch = len(ch_names)

    corrs = np.full(n_ch, np.nan)
    pvals = np.full(n_ch, np.nan)
    n_valid = np.zeros(n_ch, dtype=int)
    ch_data = np.full((n_ch, len(pow_df)), np.nan)

    for i, col in enumerate(cols):
        s = pd.to_numeric(pow_df[col], errors="coerce")
        ch_data[i] = s.values
        mask = s.notna() & temp.notna()
        n_valid[i] = int(mask.sum())
        if n_valid[i] >= min_samples:
            corrs[i], pvals[i] = compute_correlation(
                s[mask],
                temp[mask],
                "spearman" if use_spearman else "pearson",
            )

    valid = np.isfinite(pvals) & (n_valid >= min_samples)
    if not valid.any():
        return []

    p_corr = np.full_like(pvals, np.nan)
    p_corr[valid] = _bh_adjust(pvals[valid])
    sig = (p_corr < cluster_params["alpha"]) & valid

    c_labels = np.zeros(n_ch, dtype=int)
    c_pvals = np.full(n_ch, np.nan)
    c_sig = np.zeros(n_ch, dtype=bool)

    eeg_names = [eeg_info["info"]["ch_names"][i] for i in eeg_info["picks"]]
    eeg_name_to_idx = {name: idx for idx, name in enumerate(eeg_names)}
    ch_map = {i: eeg_name_to_idx[n] for i, n in enumerate(ch_names) if n in eeg_name_to_idx}

    if len(ch_map) >= cluster_params["min_channels_for_adjacency"]:
        labels, masses = compute_cluster_masses_1d(
            corrs,
            pvals,
            cluster_params["cluster_alpha"],
            ch_map,
            eeg_info["picks"],
            eeg_info["adjacency"],
        )
        if masses:
            c_labels = labels
            perm_masses = compute_topomap_permutation_masses(
                ch_data,
                temp,
                n_ch,
                cluster_params["n_cluster_perm"],
                cluster_params["cluster_alpha"],
                min_samples,
                use_spearman,
                ch_map,
                eeg_info["picks"],
                eeg_info["adjacency"],
                cluster_params["cluster_rng"],
            )
            c_pvals, c_sig, _ = compute_cluster_pvalues_1d(
                labels, masses, perm_masses, cluster_params["alpha"]
            )

    return [
        {
            "band": band,
            "channel": ch_names[i],
            "correlation": _safe_float(corrs[i]),
            "p_value": _safe_float(pvals[i]),
            "p_corrected": _safe_float(p_corr[i]),
            "significant": bool(sig[i]),
            "cluster_id": int(c_labels[i]),
            "cluster_p": _safe_float(c_pvals[i]),
            "cluster_significant": bool(c_sig[i]),
            "n_valid": int(n_valid[i]),
        }
        for i in range(n_ch)
    ]


def run_power_topomap_correlations(
    subject: str,
    task: Optional[str],
    power_df: Optional[pd.DataFrame],
    temperature: Optional[pd.Series],
    epochs_info: Any,
    stats_dir: Path,
    config: Any,
    logger,
    use_spearman: bool = True,
    rng=None,
    bootstrap: int = 0,
    n_perm: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Correlate power topomaps with temperature using pre-loaded data.

    Parameters
    ----------
    subject : str
        Subject label.
    task : str
        Task label.
    power_df : pd.DataFrame
        Power features with columns pow_<band>_<channel>.
    temperature : pd.Series
        Temperature/target series aligned to power_df rows.
    epochs_info : mne.Info
        EEG info object used to derive adjacency.
    stats_dir : Path
        Directory for saving outputs.
    config : Any
        Configuration object/dict.
    logger : logging.Logger
        Logger for status messages.
    use_spearman : bool
        Whether to use Spearman correlations (default True).
    rng : np.random.Generator, optional
        Random generator for cluster permutations.
    bootstrap : int
        Bootstrap iterations (retained for API compatibility).
    n_perm : int
        Permutation count (retained for API compatibility).
    """
    if power_df is None or power_df.empty or temperature is None or temperature.isna().all():
        logger.warning("Missing power or temperature data; skipping topomap correlations")
        return None

    if epochs_info is None:
        logger.warning("Missing EEG info; skipping topomap correlations")
        return None

    ensure_dir(stats_dir)

    adjacency, picks, _ = get_eeg_adjacency(epochs_info)
    if adjacency is None:
        logger.warning("No EEG adjacency; skipping topomap correlations")
        return None

    bands = get_frequency_band_names(config)
    alpha = get_fdr_alpha_from_config(config)
    cluster_cfg = config.get("behavior_analysis.cluster_correction", {}) if config is not None else {}

    cluster_params = {
        "alpha": float(alpha),
        "cluster_alpha": float(cluster_cfg.get("alpha", alpha)),
        "n_cluster_perm": int(cluster_cfg.get("n_permutations", 100)),
        "cluster_rng": rng or np.random.default_rng(config.get("project.random_state", 42) if config else 42),
        "min_channels_for_adjacency": config.get("behavior_analysis.statistics.min_channels_for_adjacency", 2)
        if config
        else 2,
    }

    min_samples = (
        int(config.get("behavior_analysis.statistics.min_samples_roi", 5))
        if config is not None
        else 5
    )
    method = get_correlation_method(use_spearman)

    eeg_info = {"info": epochs_info, "adjacency": adjacency, "picks": picks}
    records: List[Dict[str, Any]] = []
    for band in bands:
        records.extend(
            _process_band(
                band,
                power_df,
                temperature,
                min_samples,
                cluster_params,
                eeg_info,
                use_spearman,
            )
        )

    if not records:
        logger.warning(f"No topomap correlations for sub-{subject}")
        return None

    df = pd.DataFrame(records)
    sfx = "_spearman" if use_spearman else "_pearson"
    write_tsv(df, stats_dir / f"power_topomap_temperature_correlations{sfx}.tsv")

    summary = {
        "subject": subject,
        "task": task,
        "method": method,
        "n_bands": len(bands),
        "n_channels": len(set(df["channel"])),
        "n_bh_sig": int(df["significant"].sum()),
        "n_cluster_sig": int(df["cluster_significant"].sum()),
        "bootstrap": bootstrap,
        "n_perm": n_perm,
    }
    with open(stats_dir / f"power_topomap_temperature_meta{sfx}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {len(df)} topomap correlations to {stats_dir}")
    return df


__all__ = [
    "run_power_topomap_correlations",
]
