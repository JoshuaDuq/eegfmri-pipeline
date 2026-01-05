"""
Temporal correlation helpers shared across behavior analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr

from eeg_pipeline.utils.analysis.stats import (
    compute_correlation,
    compute_partial_corr,
    compute_cluster_correction_2d,
)
from eeg_pipeline.utils.analysis.stats.correlation import format_correlation_method_label
from eeg_pipeline.utils.analysis.tfr import (
    clip_time_range,
    compute_tfr_morlet,
    extract_trial_band_power,
    get_bands_for_tfr,
    restrict_epochs_to_roi,
    apply_baseline_to_tfr,
)
from eeg_pipeline.utils.analysis.windowing import build_time_windows_fixed_size_clamped
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.feature_io import _load_features_and_targets
from eeg_pipeline.utils.data.tfr_alignment import compute_aligned_data_length, extract_pain_vector_array
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.columns import get_pain_column_from_config
from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
from eeg_pipeline.infra.tsv import write_tsv

if TYPE_CHECKING:
    from eeg_pipeline.context.behavior import BehaviorContext


def _compute_single_bin_correlation(
    f_i: int, t_i: int, power: np.ndarray, y: np.ndarray, times: np.ndarray,
    time_edges: np.ndarray, min_pts: int, use_spearman: bool,
    cov_mat: Optional[np.ndarray], cov_df: Optional[pd.DataFrame], n_cov: int,
    config: Any, min_samples_per_cov: int = 5, partial_corr_base: int = 5, min_dof: int = 2
) -> Tuple[int, int, np.ndarray, float, float, int, bool]:
    """Compute correlation for a single frequency-time bin."""
    t_mask = (times >= time_edges[t_i]) & (times < time_edges[t_i + 1])
    bin_vals = np.full(len(y), np.nan)
    if t_mask.any():
        bin_vals = power[:, f_i, t_mask].mean(axis=1)

    valid = np.isfinite(bin_vals) & np.isfinite(y)
    n_obs = int(valid.sum())
    r, p = np.nan, np.nan

    if cov_mat is not None:
        cov_valid = valid & np.all(np.isfinite(cov_mat), axis=1)
        req = max(min_pts, n_cov * min_samples_per_cov + partial_corr_base)
        n_obs = int(cov_valid.sum())
        if n_obs >= req:
            try:
                r, p, n_obs = compute_partial_corr(
                    pd.Series(bin_vals[cov_valid]), pd.Series(y[cov_valid]),
                    pd.DataFrame(cov_mat[cov_valid], columns=list(cov_df.columns)),
                    method="spearman" if use_spearman else "pearson"
                )
            except (ValueError, RuntimeWarning) as err:
                logging.getLogger(__name__).debug(
                    "Partial correlation failed for bin (f=%s, t=%s): %s", f_i, t_i, err
                )
                r, p = np.nan, np.nan
    elif n_obs >= min_pts:
        r, p = compute_correlation(bin_vals[valid], y[valid], "spearman" if use_spearman else "pearson")

    dof = n_obs - n_cov - 2
    is_valid = np.isfinite(r) and np.isfinite(p) and dof >= min_dof and abs(r) < 1

    return f_i, t_i, bin_vals, r, p, n_obs, is_valid


def _compute_tf_correlations_for_bins(
    power: np.ndarray, y: np.ndarray, times: np.ndarray, freqs: np.ndarray,
    time_edges: np.ndarray, min_pts: int, use_spearman: bool, cov_df: Optional[pd.DataFrame] = None,
    n_jobs: int = 1, config: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Compute correlations for all time-frequency bins."""
    n_bins = len(time_edges) - 1
    corrs = np.full((len(freqs), n_bins), np.nan)
    pvals = np.full_like(corrs, np.nan)
    n_valid = np.zeros_like(corrs, dtype=int)
    bin_data = np.full((len(freqs), n_bins, len(y)), np.nan)
    info_bins = []

    cov_mat, n_cov = None, 0
    if cov_df is not None and not cov_df.empty:
        cov_mat = cov_df.apply(pd.to_numeric, errors="coerce").to_numpy()
        n_cov = cov_mat.shape[1]

    min_samples_per_cov = int(get_config_value(config, "behavior_analysis.statistics.min_samples_per_covariate", 5))
    partial_corr_base = int(get_config_value(config, "behavior_analysis.statistics.partial_corr_base_samples", 5))
    min_dof = int(get_config_value(config, "behavior_analysis.statistics.min_dof_for_correlation", 2))

    tasks = [(f_i, t_i) for f_i in range(len(freqs)) for t_i in range(n_bins)]

    if n_jobs == 1:
        results = [
            _compute_single_bin_correlation(
                f_i, t_i, power, y, times, time_edges, min_pts, use_spearman,
                cov_mat, cov_df, n_cov, config, min_samples_per_cov, partial_corr_base, min_dof
            )
            for f_i, t_i in tasks
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_single_bin_correlation)(
                f_i, t_i, power, y, times, time_edges, min_pts, use_spearman,
                cov_mat, cov_df, n_cov, config, min_samples_per_cov, partial_corr_base, min_dof
            )
            for f_i, t_i in tasks
        )

    for f_i, t_i, bin_vals, r, p, n_obs, is_valid in results:
        bin_data[f_i, t_i, :] = bin_vals
        n_valid[f_i, t_i] = n_obs
        if is_valid:
            corrs[f_i, t_i] = r
            pvals[f_i, t_i] = p
            info_bins.append((f_i, t_i))

    return corrs, pvals, n_valid, bin_data, info_bins


def _compute_single_condition_channel(
    b_i: int, w_i: int, c_i: int, bp: np.ndarray, y_c: np.ndarray,
    cov_vals: Optional[np.ndarray], cov_valid_mask: Optional[np.ndarray],
    req_samples: int, method: str, use_spearman: bool,
) -> Tuple[int, int, int, float, float, int]:
    """Compute correlation for a single band/window/channel combination."""
    ch_p = bp[:, c_i]
    valid = np.isfinite(ch_p) & np.isfinite(y_c)
    if cov_valid_mask is not None:
        valid &= cov_valid_mask

    if valid.sum() < req_samples:
        return b_i, w_i, c_i, np.nan, np.nan, 0

    ch_valid = ch_p[valid]
    if np.std(ch_valid) < 1e-12:
        return b_i, w_i, c_i, np.nan, np.nan, 0

    try:
        if cov_vals is not None and cov_vals.shape[1] > 0:
            r, p, n = compute_partial_corr(
                pd.Series(ch_valid), pd.Series(y_c[valid]),
                pd.DataFrame(cov_vals[valid]), method=method
            )
            if n < req_samples:
                return b_i, w_i, c_i, np.nan, np.nan, 0
        else:
            corr_fn = spearmanr if use_spearman else pearsonr
            r, p = corr_fn(ch_valid, y_c[valid])

        if np.isfinite(r) and np.isfinite(p):
            return b_i, w_i, c_i, float(r), float(p), int(valid.sum())
    except (ValueError, RuntimeWarning):
        pass

    return b_i, w_i, c_i, np.nan, np.nan, 0


def _compute_correlations_for_condition(
    tfr, y: np.ndarray, mask: np.ndarray, name: str, bands: Dict, win_s: np.ndarray, win_e: np.ndarray,
    fmax: float, min_trials: int, corr_fn, alpha: float, logger, cov_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None, n_jobs: int = 1,
) -> Optional[Dict[str, Any]]:
    """Compute channel/band/window correlations for a single condition."""
    if mask is None or mask.sum() < min_trials:
        return None

    idx = np.where(mask)[0] if mask.dtype == bool else mask
    tfr_c, y_c = tfr[idx], y[mask]
    use_spearman = corr_fn == spearmanr
    method = "spearman" if use_spearman else "pearson"
    cov_vals = cov_df.iloc[idx].apply(pd.to_numeric, errors="coerce").to_numpy() if cov_df is not None and not cov_df.empty else None

    band_names = list(bands.keys())
    n_ch, n_b, n_w = len(tfr.ch_names), len(band_names), len(win_s)
    n_trials = len(y_c)
    corrs = np.full((n_b, n_w, n_ch), np.nan)
    pvals = np.full_like(corrs, np.nan)
    n_valid = np.zeros_like(corrs, dtype=int)
    bin_data = np.full((n_b, n_w, n_ch, n_trials), np.nan)
    informative_bins = [[] for _ in range(n_b)]

    if cov_vals is not None:
        min_samples_per_cov = int(get_config_value(config, "behavior_analysis.statistics.min_samples_per_covariate", 5))
        partial_corr_base = int(get_config_value(config, "behavior_analysis.statistics.partial_corr_base_samples", 5))
        req_samples = max(min_trials, cov_vals.shape[1] * min_samples_per_cov + partial_corr_base)
    else:
        req_samples = min_trials

    tasks = []

    for b_i, (bn, (fmin, fmax_b)) in enumerate(bands.items()):
        fmax_eff = min(fmax_b, fmax)
        if fmin >= fmax_eff:
            continue
        for w_i, (t0, t1) in enumerate(zip(win_s, win_e)):
            bp = extract_trial_band_power(tfr_c, fmin, fmax_eff, t0, t1)
            if bp is None:
                continue
            cov_valid_mask = np.all(np.isfinite(cov_vals), axis=1) if cov_vals is not None else None

            # Cache bin-level data for permutation-based cluster correction
            if bp.shape[0] == n_trials and bp.shape[1] == n_ch:
                bin_data[b_i, w_i] = bp.T
            else:
                logger.debug(
                    "Skipping bin data caching for band=%s, window=%s due to shape mismatch (%s != %s)",
                    bn, w_i, bp.shape, (n_trials, n_ch),
                )

            for c_i in range(n_ch):
                tasks.append((b_i, w_i, c_i, bp, y_c, cov_vals, cov_valid_mask, req_samples, method, use_spearman))

    if n_jobs == 1 or len(tasks) < 50:
        results = [
            _compute_single_condition_channel(*task)
            for task in tasks
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_single_condition_channel)(
                *task
            )
            for task in tasks
        )

    for b_i, w_i, c_i, r, p, n_obs in results:
        if not np.isfinite(r) or not np.isfinite(p) or n_obs < req_samples:
            continue
        corrs[b_i, w_i, c_i] = r
        pvals[b_i, w_i, c_i] = p
        n_valid[b_i, w_i, c_i] = n_obs
        informative_bins[b_i].append((w_i, c_i))

    valid = np.isfinite(pvals) & (n_valid >= min_trials)
    if not valid.any():
        return None

    from eeg_pipeline.utils.analysis.stats.cluster_2d import (
        compute_cluster_correction_2d, compute_cluster_masses_2d
    )

    # Support both legacy `behavior_analysis.cluster_correction.*` and current
    # `behavior_analysis.cluster.*` keys from utils/config/eeg_config.yaml.
    cluster_cfg = {}
    if config is not None:
        cluster_cfg = config.get("behavior_analysis.cluster_correction", {}) or {}
        if not cluster_cfg:
            cluster_cfg = config.get("behavior_analysis.cluster", {}) or {}
    c_alpha = float(cluster_cfg.get("alpha", alpha))
    n_cluster_perm = int(cluster_cfg.get("n_permutations", 0))
    cluster_forming_threshold = cluster_cfg.get("cluster_forming_threshold", cluster_cfg.get("forming_threshold"))
    seed = int(get_config_value(config, "project.random_state", 42)) if config is not None else 42
    cluster_rng = np.random.default_rng(seed)

    cluster_labels = np.zeros_like(corrs, dtype=int)
    p_corrected = np.full_like(corrs, np.nan)
    cluster_sig = np.zeros_like(corrs, dtype=bool)
    cluster_records: List[Dict[str, Any]] = []
    cluster_perm_max: List[Dict[str, Any]] = []
    cluster_thresholds: List[Dict[str, Any]] = []
    cluster_masses: List[Dict[str, Any]] = []

    for b_i, bn in enumerate(band_names):
        band_corrs = corrs[b_i]
        band_pvals = pvals[b_i]
        band_data = bin_data[b_i]
        band_info_bins = informative_bins[b_i]

        if not band_info_bins:
            continue

        labels, pvals_corr, sig_mask, records, perm_max, c_thresh = compute_cluster_correction_2d(
            correlations=band_corrs,
            p_values=band_pvals,
            bin_data=band_data,
            informative_bins=band_info_bins,
            y_array=y_c,
            cluster_alpha=c_alpha,
            n_cluster_perm=n_cluster_perm,
            alpha=c_alpha,
            min_valid_points=req_samples,
            use_spearman=use_spearman,
            cluster_rng=cluster_rng,
            covariates_matrix=cov_vals,
            groups=None,
            cluster_forming_threshold=cluster_forming_threshold,
        )

        cluster_labels[b_i] = labels
        p_corrected[b_i] = pvals_corr
        cluster_sig[b_i] = sig_mask
        cluster_records.append({"band": bn, "clusters": records})
        cluster_perm_max.append({"band": bn, "perm_max_masses": perm_max})
        cluster_thresholds.append({"band": bn, "cluster_forming_threshold": float(c_thresh)})

        _, masses = compute_cluster_masses_2d(
            band_corrs,
            band_pvals,
            cluster_alpha=c_alpha,
            cluster_forming_threshold=cluster_forming_threshold,
            config=config,
        )
        cluster_masses.append({"band": bn, "masses": masses})

    return {
        "name": name,
        "correlations": corrs,
        "p_values": pvals,
        "n_valid": n_valid,
        "mask": cluster_sig,
        "p_corrected": p_corrected,
        "cluster_labels": cluster_labels,
        "cluster_records": cluster_records,
        "cluster_perm_max_masses": cluster_perm_max,
        "cluster_forming_thresholds": cluster_thresholds,
        "cluster_masses": cluster_masses,
        "band_names": band_names,
        "band_ranges": [bands[bn] for bn in band_names],  # List of (fmin, fmax) tuples
        "window_starts": win_s,
        "window_ends": win_e,
    }


###################################################################
# Orchestration Helpers
###################################################################


def _run_tf_correlations_core(
    subject: str,
    epochs,
    events: pd.DataFrame,
    y: pd.Series,
    stats_dir: Path,
    config,
    use_spearman: bool,
    cov_df,
    logger,
) -> None:
    """Run time-frequency correlations and save outputs."""
    logger.info("Computing time-frequency correlations...")
    if epochs is None or events is None or y is None:
        return
    if not epochs.preload:
        epochs.load_data()

    hm_cfg = config.get("behavior_analysis.time_frequency_heatmap", {})
    epochs_tfr = restrict_epochs_to_roi(epochs, hm_cfg.get("roi_selection"), config, logger)
    tfr = compute_tfr_morlet(epochs_tfr, config, logger=logger)
    if tfr is None:
        return

    baseline_applied, baseline_window = apply_baseline_to_tfr(tfr, config, logger)
    power, times, freqs = tfr.data.mean(axis=1), tfr.times, tfr.freqs
    y_arr = y.to_numpy()

    # Apply windowing
    tw = hm_cfg.get("time_window")
    if tw:
        tm = (times >= float(tw[0])) & (times <= float(tw[1]))
        if not tm.any():
            return
        times, power = times[tm], power[:, :, tm]
    fr = hm_cfg.get("freq_range")
    if fr:
        fm = (freqs >= float(fr[0])) & (freqs <= float(fr[1]))
        if not fm.any():
            return
        freqs, power = freqs[fm], power[:, fm, :]

    time_res = hm_cfg.get("time_resolution", 0.1)
    time_edges = np.arange(times[0], times[-1] + float(time_res), float(time_res))
    min_pts = int(config.get("behavior_analysis.statistics.min_samples_roi", 20))

    tf_jobs_cfg = config.get("behavior_analysis.time_frequency_heatmap.n_jobs", None)
    global_jobs_cfg = config.get("behavior_analysis.n_jobs", 1)
    n_jobs_corr = tf_jobs_cfg if tf_jobs_cfg is not None else global_jobs_cfg
    try:
        n_jobs_corr = int(n_jobs_corr)
    except (TypeError, ValueError):
        n_jobs_corr = 1
    if n_jobs_corr == 0:
        n_jobs_corr = 1
    backend_label = "loky" if n_jobs_corr != 1 else "sequential"
    logger.info(
        "TF correlations backend=%s, n_jobs=%s, bins=%s, freqs=%s",
        backend_label, n_jobs_corr, len(time_edges) - 1, len(freqs),
    )

    corrs, pvals, n_valid, bin_data, info_bins = _compute_tf_correlations_for_bins(
        power, y_arr, times, freqs, time_edges, min_pts, use_spearman, cov_df, n_jobs=n_jobs_corr, config=config
    )

    # Cluster correction
    n_cov = cov_df.shape[1] if cov_df is not None and not cov_df.empty else 0
    cluster_stat = np.full_like(corrs, np.nan)
    for f_i, t_i in info_bins:
        r, n = corrs[f_i, t_i], int(n_valid[f_i, t_i])
        dof = n - n_cov - 2
        if np.isfinite(r) and dof > 0 and abs(r) < 1:
            cluster_stat[f_i, t_i] = r * np.sqrt(dof / max(1e-15, 1.0 - r**2))

    cluster_cfg = config.get("behavior_analysis.cluster_correction", {})
    n_perm = max(int(cluster_cfg.get("n_permutations", 100)), int(hm_cfg.get("n_cluster_perm", 0)))
    c_alpha = float(cluster_cfg.get("alpha", config.get("statistics.sig_alpha", 0.05)))
    rng = np.random.default_rng(int(config.get("project.random_state", 42)))
    cov_mat = cov_df.to_numpy() if cov_df is not None and not cov_df.empty else None

    c_labels, c_pvals, c_sig, c_recs, perm_masses, c_thresh = (
        np.zeros_like(corrs, dtype=int),
        np.full_like(corrs, np.nan),
        np.zeros_like(corrs, dtype=bool),
        [],
        [],
        np.nan,
    )
    if n_perm > 0:
        c_labels, c_pvals, c_sig, c_recs, perm_masses, c_thresh = compute_cluster_correction_2d(
            correlations=cluster_stat, p_values=pvals, bin_data=bin_data,
            informative_bins=info_bins, y_array=y_arr, cluster_alpha=c_alpha,
            n_cluster_perm=n_perm, alpha=c_alpha, min_valid_points=min_pts,
            use_spearman=use_spearman, cluster_rng=rng, covariates_matrix=cov_mat,
            groups=events["run_id"].to_numpy() if "run_id" in events.columns else None,
        )

    p_used = np.where(np.isfinite(c_pvals), c_pvals, pvals) if n_perm > 0 else pvals
    ensure_dir(stats_dir)

    sfx = "_spearman" if use_spearman else "_pearson"
    roi = hm_cfg.get("roi_selection")
    roi_sfx = f"_{roi.lower()}" if roi and roi != "null" else ""
    roi_label = roi or "all"

    np.savez_compressed(
        stats_dir / f"time_frequency_correlation_data{roi_sfx}{sfx}.npz",
        correlations=corrs, p_values_raw=pvals, p_values_cluster=c_pvals, p_values=p_used,
        n_valid=n_valid, bin_data=bin_data, times=times, freqs=freqs, time_bin_edges=time_edges,
        informative_bins=np.array(info_bins), baseline_applied=baseline_applied,
        roi_selection=roi_label, use_spearman=use_spearman, cluster_labels=c_labels,
        cluster_sig_mask=c_sig, cluster_records=c_recs, cluster_perm_max_masses=perm_masses,
        cluster_forming_threshold=float(c_thresh) if n_perm > 0 else np.nan,
        cluster_alpha=c_alpha, n_cluster_perm=n_perm, n_trials=len(y_arr),
        **({"baseline_window": baseline_window} if baseline_window else {}),
    )

    method_label = format_correlation_method_label("spearman" if use_spearman else "pearson", None)
    # Save TSV
    recs = []
    for f_i, freq in enumerate(freqs):
        for t_i in range(len(time_edges) - 1):
            p_raw, p_c = pvals[f_i, t_i], c_pvals[f_i, t_i] if n_perm > 0 else np.nan
            if not (np.isfinite(p_raw) or np.isfinite(p_c)):
                continue
            recs.append({
                "roi": roi_label, "freq": float(freq), "time_start": float(time_edges[t_i]),
                "time_end": float(time_edges[t_i+1]), "r": float(corrs[f_i, t_i]),
                "beta_std": float(corrs[f_i, t_i]),
                "beta_kind": "standardized",
                "p": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_cluster": float(p_c) if np.isfinite(p_c) else np.nan,
                "cluster_id": int(c_labels[f_i, t_i]) if n_perm > 0 else 0,
                "cluster_significant": bool(c_sig[f_i, t_i]) if n_perm > 0 else False,
                "n": int(n_valid[f_i, t_i]),
                "method": "spearman" if use_spearman else "pearson",
                "method_label": method_label,
            })
    if recs:
        write_tsv(pd.DataFrame(recs), stats_dir / f"corr_stats_tf_{roi_label.lower()}{sfx}.tsv")

    logger.info(f"Saved TF correlations: shape={corrs.shape}, info_bins={len(info_bins)}")
    
    return {
        "n_tests": len(info_bins),
        "n_sig_raw": int((pvals[np.isfinite(pvals)] < 0.05).sum()) if pvals is not None else 0,
    }


def compute_time_frequency_correlations(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    use_spearman: bool,
    logger,
) -> None:
    """Load data from disk and run time-frequency correlations."""
    epochs, events = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=deriv_root, bids_root=config.bids_root,
        config=config, logger=logger,
    )
    _, _, _, y, _ = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)

    stats_cfg = config.get("behavior_analysis.statistics", {})
    partial_covars = stats_cfg.get("partial_covariates", [])
    cov_df = None
    if partial_covars and events is not None:
        avail = [c for c in partial_covars if c in events.columns]
        if avail:
            cov_df = events[avail].apply(pd.to_numeric, errors="coerce")

    return _run_tf_correlations_core(
        subject, epochs, events, y, deriv_stats_path(deriv_root, subject),
        config, use_spearman, cov_df, logger,
    )


def _build_temporal_tsv_records(
    res: Dict[str, Any],
    condition: str,
    ch_names: List[str],
    method: str,
) -> List[Dict[str, Any]]:
    """Build TSV records from temporal correlation results for global FDR."""
    method_label = format_correlation_method_label(method, None)
    records = []
    corrs = res["correlations"]  # shape: (n_bands, n_windows, n_channels)
    pvals = res["p_values"]
    n_valid = res["n_valid"]
    band_names = res["band_names"]
    win_s = res["window_starts"]
    win_e = res["window_ends"]

    n_bands, n_windows, n_channels = corrs.shape
    for b_i in range(n_bands):
        for w_i in range(n_windows):
            for c_i in range(n_channels):
                r = corrs[b_i, w_i, c_i]
                p = pvals[b_i, w_i, c_i]
                if not np.isfinite(r) or not np.isfinite(p):
                    continue
                records.append({
                    "condition": condition,
                    "band": band_names[b_i],
                    "time_start": float(win_s[w_i]),
                    "time_end": float(win_e[w_i]),
                    "channel": ch_names[c_i] if c_i < len(ch_names) else f"ch_{c_i}",
                    "r": float(r),
                    # Regression analogue: for standardized residualized variables,
                    # the slope equals the (partial) correlation.
                    "beta_std": float(r),
                    "beta_kind": "standardized",
                    "p": float(p),
                    "n": int(n_valid[b_i, w_i, c_i]),
                    "method": method,
                    "method_label": method_label,
                })
    return records


def _run_temporal_by_condition_core(
    epochs,
    events,
    y,
    stats_dir: Path,
    config,
    use_spearman: bool,
    cov_df,
    logger,
) -> None:
    """Core implementation for temporal correlations by condition."""
    if epochs is None or events is None or y is None:
        return
    if not epochs.preload:
        epochs.load_data()

    win_ms = float(get_config_value(
        config, "behavior_analysis.temporal_correlation_topomaps.window_size_ms", 500.0
    ))
    min_trials = int(get_config_value(
        config, "behavior_analysis.temporal_correlation_topomaps.min_trials_per_condition", 5
    ))
    active = tuple(config.get("time_frequency_analysis.active_window"))

    tfr = compute_tfr_morlet(epochs, config, logger=logger)
    if tfr is None:
        return
    apply_baseline_to_tfr(tfr, config, logger)

    times = np.asarray(tfr.times)
    clipped = clip_time_range(times, active[0], active[1])
    if clipped is None:
        return
    win_s, win_e = build_time_windows_fixed_size_clamped(clipped[0], clipped[1], win_ms / 1000.0)
    if len(win_s) == 0:
        return

    y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    n = compute_aligned_data_length(tfr, events)
    pain_col = get_pain_column_from_config(config, events)
    pain_vec = extract_pain_vector_array(tfr, events, pain_col, n) if pain_col else None

    pain_m, non_m = ((pain_vec == 1), (pain_vec == 0)) if pain_vec is not None else (None, None)

    fmax = float(np.max(tfr.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    corr_fn = spearmanr if use_spearman else pearsonr
    alpha = float(get_config_value(config, "statistics.sig_alpha", 0.05))
    sfx = "_spearman" if use_spearman else "_pearson"
    method = "spearman" if use_spearman else "pearson"
    ensure_dir(stats_dir)

    all_tsv_records = []
    ch_names = tfr.ch_names

    info_dict = {"ch_names": ch_names}
    if hasattr(tfr, "info"):
        info_dict["info"] = tfr.info

    pain_results = {}
    for name, mask in [("pain", pain_m), ("non_pain", non_m)]:
        if mask is None:
            continue
        n_jobs = int(get_config_value(config, "behavior_analysis.n_jobs", -1))
        res = _compute_correlations_for_condition(
            tfr, y_arr, mask, name, bands, win_s, win_e, fmax, min_trials, corr_fn, alpha, logger, cov_df, config, n_jobs
        )
        if res:
            pain_results[name] = res
            np.savez_compressed(
                stats_dir / f"temporal_correlations_{name}{sfx}.npz",
                **res, times=times, ch_names=ch_names,
            )
            all_tsv_records.extend(_build_temporal_tsv_records(res, name, ch_names, method))

    if pain_results:
        combined_pain_data = dict(pain_results)
        combined_pain_data.update(info_dict)
        combined_pain_data["times"] = times
        np.savez_compressed(stats_dir / f"temporal_correlations_by_pain{sfx}.npz", **combined_pain_data)
        logger.info(f"Saved combined pain temporal correlations: {len(pain_results)} conditions")

    if all_tsv_records:
        tsv_path = stats_dir / f"corr_stats_temporal_all{sfx}.tsv"
        write_tsv(pd.DataFrame(all_tsv_records), tsv_path)
        logger.info(f"Saved temporal correlation TSV for global FDR: {len(all_tsv_records)} tests -> {tsv_path.name}")

    logger.info("Temporal correlations by condition completed")
    
    n_tests = len(all_tsv_records)
    n_sig = 0
    if all_tsv_records:
        n_sig = sum(1 for r in all_tsv_records if r.get("p", 1.0) < 0.05)
        
    return {
        "n_tests": n_tests,
        "n_sig_raw": n_sig,
    }


def compute_temporal_correlations_by_condition(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    use_spearman: bool,
    logger,
) -> None:
    """Compute temporal correlations by condition by loading data from disk."""
    logger.info("Computing temporal correlations by condition...")
    epochs, events = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=deriv_root, bids_root=config.bids_root,
        config=config, logger=logger,
    )
    _, _, _, y, _ = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    stats_cfg = config.get("behavior_analysis.statistics", {})
    partial_covars = stats_cfg.get("partial_covariates", [])
    cov_df = None
    if partial_covars and events is not None:
        avail = [c for c in partial_covars if c in events.columns]
        if avail:
            cov_df = events[avail].apply(pd.to_numeric, errors="coerce")
    return _run_temporal_by_condition_core(
        epochs, events, y, deriv_stats_path(deriv_root, subject), config, use_spearman, cov_df, logger
    )


def compute_time_frequency_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Run time-frequency correlations using pre-loaded context data."""
    if ctx.computation_features and "temporal" in ctx.computation_features:
        allowed = ctx.computation_features["temporal"]
        if "power" not in allowed and "spectral" not in allowed:
            ctx.logger.info("Skipping time-frequency correlations: feature filter %s excludes 'power'/'spectral'", allowed)
            return None

    return _run_tf_correlations_core(
        ctx.subject,
        ctx.epochs,
        ctx.aligned_events,
        ctx.targets,
        ctx.stats_dir,
        ctx.config,
        ctx.use_spearman,
        ctx.covariates_df,
        ctx.logger,
    )


def compute_temporal_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Compute temporal correlations by condition using pre-loaded data."""
    if ctx.computation_features and "temporal" in ctx.computation_features:
        allowed = ctx.computation_features["temporal"]
        if "power" not in allowed and "spectral" not in allowed:
            ctx.logger.info("Skipping temporal correlations: feature filter %s excludes 'power'/'spectral'", allowed)
            return None

    return _run_temporal_by_condition_core(
        ctx.epochs,
        ctx.aligned_events,
        ctx.targets,
        ctx.stats_dir,
        ctx.config,
        ctx.use_spearman,
        ctx.covariates_df,
        ctx.logger,
    )





