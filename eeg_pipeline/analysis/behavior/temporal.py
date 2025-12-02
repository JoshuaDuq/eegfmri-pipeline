"""Time-frequency and temporal correlation analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from joblib import Parallel, delayed

from eeg_pipeline.utils.data.loading import (
    _load_features_and_targets, load_epochs_for_analysis, compute_aligned_data_length,
    extract_pain_vector_array, extract_temperature_series,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path, ensure_dir, get_pain_column_from_config,
    get_temperature_column_from_config, write_tsv,
)
from eeg_pipeline.utils.analysis.stats import compute_correlation, compute_partial_corr, _safe_float
from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_morlet, extract_trial_band_power, get_bands_for_tfr,
    clip_time_range, create_time_windows_fixed_size,
)

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext

_MIN_SAMPLES_PER_COVARIATE = 5
_PARTIAL_CORR_BASE_SAMPLES = 5


def _create_temperature_masks(temp_series: Optional[pd.Series], min_trials: int, logger) -> Dict[str, np.ndarray]:
    if temp_series is None:
        return {}
    temp_arr = temp_series.values if isinstance(temp_series, pd.Series) else np.asarray(temp_series)
    if temp_arr.size == 0:
        return {}
    masks = {}
    for t in np.unique(temp_arr[np.isfinite(temp_arr)]):
        m = (temp_arr == t)
        if m.sum() >= min_trials:
            masks[f"{t:.1f}".replace(".", "_")] = m
            logger.info(f"Temp {t:.1f}°C: {m.sum()} trials")
    return masks


def _compute_single_bin_correlation(
    f_i: int, t_i: int, power: np.ndarray, y: np.ndarray, times: np.ndarray,
    time_edges: np.ndarray, min_pts: int, use_spearman: bool,
    cov_mat: Optional[np.ndarray], cov_df: Optional[pd.DataFrame], n_cov: int
) -> Tuple[int, int, np.ndarray, float, float, int, bool]:
    """Compute correlation for a single frequency-time bin. Returns (f_i, t_i, bin_data, r, p, n_obs, is_valid)."""
    t_mask = (times >= time_edges[t_i]) & (times < time_edges[t_i + 1])
    bin_vals = np.full(len(y), np.nan)
    if t_mask.any():
        bin_vals = power[:, f_i, t_mask].mean(axis=1)
    
    valid = np.isfinite(bin_vals) & np.isfinite(y)
    n_obs = int(valid.sum())
    r, p = np.nan, np.nan
    
    if cov_mat is not None:
        cov_valid = valid & np.all(np.isfinite(cov_mat), axis=1)
        req = max(min_pts, n_cov * _MIN_SAMPLES_PER_COVARIATE + _PARTIAL_CORR_BASE_SAMPLES)
        n_obs = int(cov_valid.sum())
        if n_obs >= req:
            try:
                r, p, n_obs = compute_partial_corr(
                    pd.Series(bin_vals[cov_valid]), pd.Series(y[cov_valid]),
                    pd.DataFrame(cov_mat[cov_valid], columns=list(cov_df.columns)),
                    method="spearman" if use_spearman else "pearson"
                )
            except Exception:
                pass
    elif n_obs >= min_pts:
        r, p = compute_correlation(bin_vals[valid], y[valid], "spearman" if use_spearman else "pearson")
    
    dof = n_obs - n_cov - 2
    is_valid = np.isfinite(r) and np.isfinite(p) and dof >= 2 and abs(r) < 1
    
    return f_i, t_i, bin_vals, r, p, n_obs, is_valid


def _compute_tf_correlations_for_bins(
    power: np.ndarray, y: np.ndarray, times: np.ndarray, freqs: np.ndarray,
    time_edges: np.ndarray, min_pts: int, use_spearman: bool, cov_df: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
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

    # Create list of all (frequency, time_bin) pairs to process
    tasks = [(f_i, t_i) for f_i in range(len(freqs)) for t_i in range(n_bins)]
    
    # Process in parallel if n_jobs > 1, otherwise sequential
    if n_jobs == 1:
        results = [
            _compute_single_bin_correlation(
                f_i, t_i, power, y, times, time_edges, min_pts, use_spearman,
                cov_mat, cov_df, n_cov
            )
            for f_i, t_i in tasks
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_compute_single_bin_correlation)(
                f_i, t_i, power, y, times, time_edges, min_pts, use_spearman,
                cov_mat, cov_df, n_cov
            )
            for f_i, t_i in tasks
        )
    
    # Aggregate results
    for f_i, t_i, bin_vals, r, p, n_obs, is_valid in results:
        bin_data[f_i, t_i, :] = bin_vals
        n_valid[f_i, t_i] = n_obs
        if is_valid:
            corrs[f_i, t_i] = r
            pvals[f_i, t_i] = p
            info_bins.append((f_i, t_i))

    return corrs, pvals, n_valid, bin_data, info_bins


def _compute_correlations_for_condition(
    tfr, y: np.ndarray, mask: np.ndarray, name: str, bands: Dict, win_s: np.ndarray, win_e: np.ndarray,
    fmax: float, min_trials: int, corr_fn, alpha: float, logger, cov_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    if mask is None or mask.sum() < min_trials:
        return None
    
    # Convert boolean mask to integer indices for MNE object indexing
    idx = np.where(mask)[0] if mask.dtype == bool else mask
    tfr_c, y_c = tfr[idx], y[mask]
    method = "spearman" if corr_fn == spearmanr else "pearson"
    cov_vals = cov_df.iloc[idx].apply(pd.to_numeric, errors="coerce").to_numpy() if cov_df is not None and not cov_df.empty else None
    
    n_ch, n_b, n_w = len(tfr.ch_names), len(bands), len(win_s)
    corrs = np.full((n_b, n_w, n_ch), np.nan)
    pvals = np.full_like(corrs, np.nan)
    n_valid = np.zeros_like(corrs, dtype=int)

    for b_i, (bn, (fmin, fmax_b)) in enumerate(bands.items()):
        fmax_eff = min(fmax_b, fmax)
        if fmin >= fmax_eff:
            continue
        for w_i, (t0, t1) in enumerate(zip(win_s, win_e)):
            bp = extract_trial_band_power(tfr_c, fmin, fmax_eff, t0, t1)
            if bp is None:
                continue
            for c_i in range(n_ch):
                ch_p = bp[:, c_i]
                valid = np.isfinite(ch_p) & np.isfinite(y_c)
                if cov_vals is not None:
                    valid &= np.all(np.isfinite(cov_vals), axis=1)
                    req = max(min_trials, cov_vals.shape[1] * _MIN_SAMPLES_PER_COVARIATE + _PARTIAL_CORR_BASE_SAMPLES)
                else:
                    req = min_trials
                if valid.sum() < req:
                    continue
                try:
                    if cov_vals is not None and cov_vals.shape[1] > 0:
                        r, p, n = compute_partial_corr(
                            pd.Series(ch_p[valid]), pd.Series(y_c[valid]),
                            pd.DataFrame(cov_vals[valid]), method=method
                        )
                        if n < req:
                            continue
                    else:
                        r, p = corr_fn(ch_p[valid], y_c[valid])
                    if np.isfinite(r) and np.isfinite(p):
                        corrs[b_i, w_i, c_i], pvals[b_i, w_i, c_i] = r, p
                        n_valid[b_i, w_i, c_i] = int(valid.sum())
                except Exception:
                    pass

    return {
        "correlations": corrs, "p_values": pvals, "p_corrected": np.full_like(pvals, np.nan),
        "n_valid": n_valid, "band_names": list(bands.keys()),
        "band_ranges": [(fmin, min(fmax_b, fmax)) for fmin, fmax_b in bands.values()],
        "window_starts": win_s, "window_ends": win_e, "condition_name": name,
    }


def _run_tf_correlations_core(subject: str, epochs, events: pd.DataFrame, y: pd.Series,
                               stats_dir: Path, config, use_spearman: bool, cov_df, logger) -> None:
    from eeg_pipeline.utils.analysis.tfr import restrict_epochs_to_roi, apply_baseline_to_tfr
    from eeg_pipeline.utils.analysis.stats import compute_cluster_correction_2d
    
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
    
    time_edges = np.arange(times[0], times[-1] + float(hm_cfg.get("time_resolution")), float(hm_cfg.get("time_resolution")))
    min_pts = int(config.get("behavior_analysis.statistics.min_samples_roi", 20))
    
    # Get number of parallel jobs for correlation computation (default: 1 for safety)
    n_jobs_corr = int(config.get("behavior_analysis.time_frequency_heatmap.n_jobs", 1))
    
    corrs, pvals, n_valid, bin_data, info_bins = _compute_tf_correlations_for_bins(
        power, y_arr, times, freqs, time_edges, min_pts, use_spearman, cov_df, n_jobs=n_jobs_corr
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
    
    c_labels, c_pvals, c_sig, c_recs, perm_masses, c_thresh = np.zeros_like(corrs, dtype=int), np.full_like(corrs, np.nan), np.zeros_like(corrs, dtype=bool), [], [], np.nan
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
                "p": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_cluster": float(p_c) if np.isfinite(p_c) else np.nan,
                "cluster_id": int(c_labels[f_i, t_i]) if n_perm > 0 else 0,
                "cluster_significant": bool(c_sig[f_i, t_i]) if n_perm > 0 else False,
                "n": int(n_valid[f_i, t_i]), "method": "spearman" if use_spearman else "pearson",
            })
    if recs:
        write_tsv(pd.DataFrame(recs), stats_dir / f"corr_stats_tf_{roi_label.lower()}{sfx}.tsv")
    
    logger.info(f"Saved TF correlations: shape={corrs.shape}, info_bins={len(info_bins)}")


def compute_time_frequency_correlations(subject: str, task: str, deriv_root: Path,
                                        config, use_spearman: bool, logger) -> None:
    epochs, events = load_epochs_for_analysis(subject, task, align="strict", preload=True,
                                              deriv_root=deriv_root, bids_root=config.bids_root,
                                              config=config, logger=logger)
    _, _, _, y, _ = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    
    stats_cfg = config.get("behavior_analysis.statistics", {})
    partial_covars = stats_cfg.get("partial_covariates", [])
    cov_df = None
    if partial_covars and events is not None:
        avail = [c for c in partial_covars if c in events.columns]
        if avail:
            cov_df = events[avail].apply(pd.to_numeric, errors="coerce")
    
    _run_tf_correlations_core(subject, epochs, events, y, deriv_stats_path(deriv_root, subject),
                              config, use_spearman, cov_df, logger)


def _run_temporal_by_condition_core(epochs, events, y, stats_dir: Path, config, use_spearman: bool, cov_df, logger) -> None:
    """Core implementation for temporal correlations by condition."""
    from eeg_pipeline.utils.analysis.tfr import apply_baseline_to_tfr
    
    if epochs is None or events is None or y is None:
        return
    if not epochs.preload:
        epochs.load_data()
    
    topo_cfg = config.get("behavior_analysis.temporal_correlation_topomaps", {})
    win_ms = float(topo_cfg.get("window_size_ms", 100.0))
    min_trials = int(topo_cfg.get("min_trials_per_condition", 5))
    plateau = tuple(config.get("time_frequency_analysis.plateau_window"))
    
    tfr = compute_tfr_morlet(epochs, config, logger=logger)
    if tfr is None:
        return
    apply_baseline_to_tfr(tfr, config, logger)
    
    times = np.asarray(tfr.times)
    clipped = clip_time_range(times, plateau[0], plateau[1])
    if clipped is None:
        return
    win_s, win_e = create_time_windows_fixed_size(clipped[0], clipped[1], win_ms)
    if len(win_s) == 0:
        return
    
    y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    n = compute_aligned_data_length(tfr, events)
    pain_col = get_pain_column_from_config(config, events)
    temp_col = get_temperature_column_from_config(config, events)
    pain_vec = extract_pain_vector_array(tfr, events, pain_col, n) if pain_col else None
    temp_series = extract_temperature_series(tfr, events, temp_col, n) if temp_col else None
    
    pain_m, non_m = ((pain_vec == 1), (pain_vec == 0)) if pain_vec is not None else (None, None)
    temp_masks = _create_temperature_masks(temp_series, min_trials, logger)
    
    fmax = float(np.max(tfr.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    corr_fn = spearmanr if use_spearman else pearsonr
    alpha = float(config.get("statistics.sig_alpha", 0.05))
    sfx = "_spearman" if use_spearman else "_pearson"
    ensure_dir(stats_dir)
    
    # Save temperature data in combined file (for plotting function)
    temp_results = {}
    info_dict = {"ch_names": tfr.ch_names}
    if hasattr(tfr, "info"):
        info_dict["info"] = tfr.info
    
    for t_str, t_mask in temp_masks.items():
        res = _compute_correlations_for_condition(tfr, y_arr, t_mask, f"temp_{t_str}", bands, win_s, win_e, fmax, min_trials, corr_fn, alpha, logger, cov_df)
        if res:
            # Store in combined dict for single file (use temp_ prefix as expected by plotting function)
            temp_results[f"temp_{t_str}"] = np.array([res], dtype=object)  # Wrap in array for npz compatibility
            # Also save individual file for backwards compatibility
            np.savez_compressed(stats_dir / f"temporal_correlations_temp_{t_str}{sfx}.npz", **res, times=times, ch_names=tfr.ch_names)
    
    # Save combined temperature file if we have any temperature data
    if temp_results:
        # Unwrap the results for the combined file
        combined_temp_data = {}
        for key, res_array in temp_results.items():
            combined_temp_data[key] = res_array.item()  # Unwrap from array
        combined_temp_data.update(info_dict)
        combined_temp_data["times"] = times
        np.savez_compressed(stats_dir / f"temporal_correlations_by_temperature{sfx}.npz", **combined_temp_data)
        logger.info(f"Saved combined temperature temporal correlations: {len(temp_results)} temperature levels")
    
    for name, mask in [("pain", pain_m), ("non_pain", non_m)]:
        if mask is None:
            continue
        res = _compute_correlations_for_condition(tfr, y_arr, mask, name, bands, win_s, win_e, fmax, min_trials, corr_fn, alpha, logger, cov_df)
        if res:
            np.savez_compressed(stats_dir / f"temporal_correlations_{name}{sfx}.npz", **res, times=times, ch_names=tfr.ch_names)
    
    logger.info("Temporal correlations by condition completed")


def compute_temporal_correlations_by_condition(subject: str, task: str, deriv_root: Path,
                                               config, use_spearman: bool, logger) -> None:
    """Compute temporal correlations by condition by loading data from disk."""
    logger.info("Computing temporal correlations by condition...")
    epochs, events = load_epochs_for_analysis(subject, task, align="strict", preload=True,
                                              deriv_root=deriv_root, bids_root=config.bids_root,
                                              config=config, logger=logger)
    _, _, _, y, _ = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    stats_cfg = config.get("behavior_analysis.statistics", {})
    partial_covars = stats_cfg.get("partial_covariates", [])
    cov_df = None
    if partial_covars and events is not None:
        avail = [c for c in partial_covars if c in events.columns]
        if avail:
            cov_df = events[avail].apply(pd.to_numeric, errors="coerce")
    _run_temporal_by_condition_core(epochs, events, y, deriv_stats_path(deriv_root, subject), config, use_spearman, cov_df, logger)


def compute_time_frequency_from_context(ctx: "BehaviorContext") -> None:
    _run_tf_correlations_core(ctx.subject, ctx.epochs, ctx.aligned_events, ctx.targets,
                              ctx.stats_dir, ctx.config, ctx.use_spearman, ctx.covariates_df, ctx.logger)


def compute_temporal_from_context(ctx: "BehaviorContext") -> None:
    """Compute temporal correlations by condition using pre-loaded data."""
    _run_temporal_by_condition_core(ctx.epochs, ctx.aligned_events, ctx.targets, ctx.stats_dir,
                                    ctx.config, ctx.use_spearman, ctx.covariates_df, ctx.logger)
