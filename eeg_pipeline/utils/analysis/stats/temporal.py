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


# Numerical stability constants
MIN_VARIANCE_THRESHOLD = 1e-12
MIN_DENOMINATOR_THRESHOLD = 1e-15
MIN_OBSERVATIONS_FOR_CORRELATION = 10
PARALLELIZATION_TASK_THRESHOLD = 50


def _compute_single_bin_correlation(
    freq_idx: int,
    time_idx: int,
    power: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    time_edges: np.ndarray,
    min_pts: int,
    use_spearman: bool,
    cov_mat: Optional[np.ndarray],
    cov_df: Optional[pd.DataFrame],
    n_cov: int,
    config: Any,
    min_samples_per_cov: int = 5,
    partial_corr_base: int = 5,
    min_dof: int = 2,
) -> Tuple[int, int, np.ndarray, float, float, int, bool]:
    """Compute correlation for a single frequency-time bin."""
    t_mask = (times >= time_edges[time_idx]) & (times < time_edges[time_idx + 1])
    bin_vals = np.full(len(y), np.nan)
    if t_mask.any():
        bin_vals = power[:, freq_idx, t_mask].mean(axis=1)

    valid = np.isfinite(bin_vals) & np.isfinite(y)
    n_obs = int(valid.sum())
    r, p = np.nan, np.nan

    if cov_mat is not None:
        cov_valid = valid & np.all(np.isfinite(cov_mat), axis=1)
        min_required = max(min_pts, n_cov * min_samples_per_cov + partial_corr_base)
        n_obs = int(cov_valid.sum())
        if n_obs >= min_required:
            try:
                r, p, n_obs = compute_partial_corr(
                    pd.Series(bin_vals[cov_valid]),
                    pd.Series(y[cov_valid]),
                    pd.DataFrame(cov_mat[cov_valid], columns=list(cov_df.columns)),
                    method="spearman" if use_spearman else "pearson",
                )
            except (ValueError, RuntimeWarning) as err:
                logging.getLogger(__name__).debug(
                    "Partial correlation failed for bin (f=%s, t=%s): %s",
                    freq_idx,
                    time_idx,
                    err,
                )
                r, p = np.nan, np.nan
    elif n_obs >= min_pts:
        r, p = compute_correlation(
            bin_vals[valid], y[valid], "spearman" if use_spearman else "pearson"
        )

    dof = n_obs - n_cov - 2
    is_valid = (
        np.isfinite(r) and np.isfinite(p) and dof >= min_dof and abs(r) < 1
    )

    return freq_idx, time_idx, bin_vals, r, p, n_obs, is_valid


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

    tasks = [
        (freq_idx, time_idx)
        for freq_idx in range(len(freqs))
        for time_idx in range(n_bins)
    ]

    should_parallelize = n_jobs != 1 and len(tasks) >= PARALLELIZATION_TASK_THRESHOLD
    if should_parallelize:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_single_bin_correlation)(
                freq_idx,
                time_idx,
                power,
                y,
                times,
                time_edges,
                min_pts,
                use_spearman,
                cov_mat,
                cov_df,
                n_cov,
                config,
                min_samples_per_cov,
                partial_corr_base,
                min_dof,
            )
            for freq_idx, time_idx in tasks
        )
    else:
        results = [
            _compute_single_bin_correlation(
                freq_idx,
                time_idx,
                power,
                y,
                times,
                time_edges,
                min_pts,
                use_spearman,
                cov_mat,
                cov_df,
                n_cov,
                config,
                min_samples_per_cov,
                partial_corr_base,
                min_dof,
            )
            for freq_idx, time_idx in tasks
        ]

    for freq_idx, time_idx, bin_vals, r, p, n_obs, is_valid in results:
        bin_data[freq_idx, time_idx, :] = bin_vals
        n_valid[freq_idx, time_idx] = n_obs
        if is_valid:
            corrs[freq_idx, time_idx] = r
            pvals[freq_idx, time_idx] = p
            info_bins.append((freq_idx, time_idx))

    return corrs, pvals, n_valid, bin_data, info_bins


def _compute_single_condition_channel(
    band_idx: int,
    window_idx: int,
    channel_idx: int,
    band_power: np.ndarray,
    y_condition: np.ndarray,
    cov_vals: Optional[np.ndarray],
    cov_valid_mask: Optional[np.ndarray],
    req_samples: int,
    method: str,
    use_spearman: bool,
) -> Tuple[int, int, int, float, float, int]:
    """Compute correlation for a single band/window/channel combination."""
    channel_power = band_power[:, channel_idx]
    valid = np.isfinite(channel_power) & np.isfinite(y_condition)
    if cov_valid_mask is not None:
        valid &= cov_valid_mask

    if valid.sum() < req_samples:
        return band_idx, window_idx, channel_idx, np.nan, np.nan, 0

    channel_valid = channel_power[valid]
    if np.std(channel_valid) < MIN_VARIANCE_THRESHOLD:
        return band_idx, window_idx, channel_idx, np.nan, np.nan, 0

    try:
        if cov_vals is not None and cov_vals.shape[1] > 0:
            r, p, n_obs = compute_partial_corr(
                pd.Series(channel_valid),
                pd.Series(y_condition[valid]),
                pd.DataFrame(cov_vals[valid]),
                method=method,
            )
            if n_obs < req_samples:
                return band_idx, window_idx, channel_idx, np.nan, np.nan, 0
        else:
            corr_fn = spearmanr if use_spearman else pearsonr
            r, p = corr_fn(channel_valid, y_condition[valid])

        if np.isfinite(r) and np.isfinite(p):
            return (
                band_idx,
                window_idx,
                channel_idx,
                float(r),
                float(p),
                int(valid.sum()),
            )
    except (ValueError, RuntimeWarning):
        pass

    return band_idx, window_idx, channel_idx, np.nan, np.nan, 0


def _compute_correlations_for_condition(
    tfr, y: np.ndarray, mask: np.ndarray, name: str, bands: Dict, win_s: np.ndarray, win_e: np.ndarray,
    fmax: float, corr_fn, alpha: float, logger, cov_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None, n_jobs: int = 1,
) -> Optional[Dict[str, Any]]:
    """Compute channel/band/window correlations for a single condition."""
    if mask is None:
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
        req_samples = cov_vals.shape[1] * min_samples_per_cov + partial_corr_base
    else:
        req_samples = 0

    tasks = []

    for band_idx, (band_name, (fmin, fmax_band)) in enumerate(bands.items()):
        fmax_effective = min(fmax_band, fmax)
        if fmin >= fmax_effective:
            continue
        for window_idx, (t0, t1) in enumerate(zip(win_s, win_e)):
            band_power = extract_trial_band_power(tfr_c, fmin, fmax_effective, t0, t1)
            if band_power is None:
                continue
            cov_valid_mask = (
                np.all(np.isfinite(cov_vals), axis=1) if cov_vals is not None else None
            )

            # Cache bin-level data for permutation-based cluster correction
            if band_power.shape[0] == n_trials and band_power.shape[1] == n_ch:
                bin_data[band_idx, window_idx] = band_power.T
            else:
                logger.debug(
                    "Skipping bin data caching for band=%s, window=%s due to shape mismatch (%s != %s)",
                    band_name,
                    window_idx,
                    band_power.shape,
                    (n_trials, n_ch),
                )

            for channel_idx in range(n_ch):
                tasks.append(
                    (
                        band_idx,
                        window_idx,
                        channel_idx,
                        band_power,
                        y_c,
                        cov_vals,
                        cov_valid_mask,
                        req_samples,
                        method,
                        use_spearman,
                    )
                )

    should_parallelize = n_jobs != 1 and len(tasks) >= PARALLELIZATION_TASK_THRESHOLD
    if should_parallelize:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_single_condition_channel)(*task) for task in tasks
        )
    else:
        results = [
            _compute_single_condition_channel(*task) for task in tasks
        ]

    for band_idx, window_idx, channel_idx, r, p, n_obs in results:
        if not np.isfinite(r) or not np.isfinite(p) or n_obs < req_samples:
            continue
        corrs[band_idx, window_idx, channel_idx] = r
        pvals[band_idx, window_idx, channel_idx] = p
        n_valid[band_idx, window_idx, channel_idx] = n_obs
        informative_bins[band_idx].append((window_idx, channel_idx))

    valid = np.isfinite(pvals)
    if not valid.any():
        return None

    from eeg_pipeline.utils.analysis.stats.cluster import (
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

    for band_idx, band_name in enumerate(band_names):
        band_corrs = corrs[band_idx]
        band_pvals = pvals[band_idx]
        band_data = bin_data[band_idx]
        band_info_bins = informative_bins[band_idx]

        if not band_info_bins:
            continue

        labels, pvals_corr, sig_mask, records, perm_max, c_thresh = (
            compute_cluster_correction_2d(
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
        )

        cluster_labels[band_idx] = labels
        p_corrected[band_idx] = pvals_corr
        cluster_sig[band_idx] = sig_mask
        cluster_records.append({"band": band_name, "clusters": records})
        cluster_perm_max.append({"band": band_name, "perm_max_masses": perm_max})
        cluster_thresholds.append(
            {"band": band_name, "cluster_forming_threshold": float(c_thresh)}
        )

        _, masses = compute_cluster_masses_2d(
            band_corrs,
            band_pvals,
            cluster_alpha=c_alpha,
            cluster_forming_threshold=cluster_forming_threshold,
            config=config,
        )
        cluster_masses.append({"band": band_name, "masses": masses})

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
# Condition Splitting Helpers
###################################################################


def _determine_condition_values(
    events: pd.DataFrame,
    n_trials: int,
    temporal_cfg: Dict[str, Any],
    config: Any,
    logger: logging.Logger,
    analysis_name: str = "temporal",
) -> Tuple[List[Any], np.ndarray]:
    """Determine condition values and vector for splitting analysis.
    
    Returns:
        condition_values: List of condition values to process
        condition_vec: Array of condition values for each trial (length n_trials)
    """
    split_by_condition = bool(temporal_cfg.get("split_by_condition", True))
    condition_column = temporal_cfg.get("condition_column", "") or ""
    filter_value = temporal_cfg.get("filter_value", "") or ""
    requested_values = temporal_cfg.get("condition_values", []) or []

    if not condition_column:
        condition_column = get_pain_column_from_config(config, events)

    condition_vec = None
    condition_values = []

    if split_by_condition and condition_column and condition_column in events.columns:
        condition_vec = (
            events[condition_column].to_numpy()[:n_trials]
            if len(events) >= n_trials
            else events[condition_column].to_numpy()
        )

        if filter_value:
            try:
                filter_val = (
                    type(condition_vec[0])(filter_value)
                    if len(condition_vec) > 0
                    else filter_value
                )
            except (ValueError, TypeError):
                filter_val = filter_value
            condition_values = [filter_val]
            logger.info(
                f"{analysis_name}: filtering to condition '{condition_column}' == '{filter_val}'"
            )
        elif isinstance(requested_values, (list, tuple)) and len(requested_values) > 0:
            exemplar = pd.Series(condition_vec).dropna()
            exemplar_val = exemplar.iloc[0] if len(exemplar) > 0 else None

            def _cast_value(value: Any) -> Any:
                if exemplar_val is None:
                    return value
                try:
                    if isinstance(exemplar_val, (np.integer, int)):
                        return int(float(value))
                    if isinstance(exemplar_val, (np.floating, float)):
                        return float(value)
                    return type(exemplar_val)(value)
                except (ValueError, TypeError):
                    return value

            condition_values = [_cast_value(v) for v in requested_values]
            logger.info(
                f"{analysis_name}: using configured values for '{condition_column}': {condition_values}"
            )
        else:
            condition_values = list(pd.Series(condition_vec).dropna().unique())
            logger.info(
                f"{analysis_name}: splitting by '{condition_column}' with values {condition_values}"
            )
    elif not split_by_condition:
        logger.info(f"{analysis_name}: split_by_condition=False, computing over all trials")
        condition_values = ["all"]
        condition_vec = np.array(["all"] * n_trials)
    else:
        logger.warning(
            f"{analysis_name}: condition column '{condition_column}' not found, computing over all trials"
        )
        condition_values = ["all"]
        condition_vec = np.array(["all"] * n_trials)

    return condition_values, condition_vec


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
    for freq_idx, time_idx in info_bins:
        r = corrs[freq_idx, time_idx]
        n_obs = int(n_valid[freq_idx, time_idx])
        dof = n_obs - n_cov - 2
        if np.isfinite(r) and dof > 0 and abs(r) < 1:
            denominator = max(MIN_DENOMINATOR_THRESHOLD, 1.0 - r**2)
            cluster_stat[freq_idx, time_idx] = r * np.sqrt(dof / denominator)

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
    for freq_idx, freq in enumerate(freqs):
        for time_idx in range(len(time_edges) - 1):
            p_raw = pvals[freq_idx, time_idx]
            p_cluster = c_pvals[freq_idx, time_idx] if n_perm > 0 else np.nan
            if not (np.isfinite(p_raw) or np.isfinite(p_cluster)):
                continue
            recs.append({
                "roi": roi_label,
                "freq": float(freq),
                "time_start": float(time_edges[time_idx]),
                "time_end": float(time_edges[time_idx + 1]),
                "r": float(corrs[freq_idx, time_idx]),
                "beta_std": float(corrs[freq_idx, time_idx]),
                "beta_kind": "standardized",
                "p": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_cluster": float(p_cluster) if np.isfinite(p_cluster) else np.nan,
                "cluster_id": int(c_labels[freq_idx, time_idx]) if n_perm > 0 else 0,
                "cluster_significant": bool(c_sig[freq_idx, time_idx]) if n_perm > 0 else False,
                "n": int(n_valid[freq_idx, time_idx]),
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
    for band_idx in range(n_bands):
        for window_idx in range(n_windows):
            for channel_idx in range(n_channels):
                r = corrs[band_idx, window_idx, channel_idx]
                p = pvals[band_idx, window_idx, channel_idx]
                if not np.isfinite(r) or not np.isfinite(p):
                    continue
                records.append({
                    "condition": condition,
                    "band": band_names[band_idx],
                    "time_start": float(win_s[window_idx]),
                    "time_end": float(win_e[window_idx]),
                    "channel": (
                        ch_names[channel_idx]
                        if channel_idx < len(ch_names)
                        else f"ch_{channel_idx}"
                    ),
                    "r": float(r),
                    # Regression analogue: for standardized residualized variables,
                    # the slope equals the (partial) correlation.
                    "beta_std": float(r),
                    "beta_kind": "standardized",
                    "p": float(p),
                    "n": int(n_valid[band_idx, window_idx, channel_idx]),
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
    selected_bands: Optional[List[str]] = None,
) -> None:
    """Core implementation for temporal correlations by condition."""
    if epochs is None or events is None or y is None:
        return
    if not epochs.preload:
        epochs.load_data()

    win_ms = float(get_config_value(
        config, "behavior_analysis.temporal_correlation_topomaps.window_size_ms", 500.0
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

    temporal_cfg = config.get("behavior_analysis.temporal", {}) or {}
    condition_values, condition_vec = _determine_condition_values(
        events, n, temporal_cfg, config, logger, analysis_name="Temporal correlations"
    )

    fmax = float(np.max(tfr.freqs))
    all_bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    if selected_bands:
        bands = {k: v for k, v in all_bands.items() if k.lower() in [b.lower() for b in selected_bands]}
        if not bands:
            logger.warning(f"Temporal correlations: no matching bands in {selected_bands}, using all bands")
            bands = all_bands
        else:
            logger.info(f"Temporal correlations: using selected bands {list(bands.keys())}")
    else:
        bands = all_bands
    
    corr_fn = spearmanr if use_spearman else pearsonr
    alpha = float(get_config_value(config, "statistics.sig_alpha", 0.05))
    sfx = "_spearman" if use_spearman else "_pearson"
    method = "spearman" if use_spearman else "pearson"
    
    out_dir = stats_dir / "temporal_correlations"
    ensure_dir(out_dir)

    all_tsv_records = []
    ch_names = tfr.ch_names

    for cond_val in condition_values:
        if cond_val == "all":
            mask = np.ones(n, dtype=bool)
        else:
            mask = condition_vec == cond_val
        
        if not np.any(mask):
            logger.info(f"Temporal correlations: no trials for condition '{cond_val}', skipping")
            continue
        
        safe_name = str(cond_val).replace(" ", "_").replace("/", "_")
        n_jobs = int(get_config_value(config, "behavior_analysis.n_jobs", -1))
        res = _compute_correlations_for_condition(
            tfr, y_arr, mask, safe_name, bands, win_s, win_e, fmax, corr_fn, alpha, logger, cov_df, config, n_jobs
        )
        if res:
            cond_records = _build_temporal_tsv_records(res, safe_name, ch_names, method)
            for rec in cond_records:
                rec["feature"] = "power"
            all_tsv_records.extend(cond_records)
            logger.info(f"Computed power temporal for condition '{safe_name}': {len(cond_records)} tests")

    logger.info("Temporal correlations by condition completed")
    
    n_tests = len(all_tsv_records)
    n_sig = 0
    if all_tsv_records:
        n_sig = sum(1 for r in all_tsv_records if r.get("p", 1.0) < 0.05)
        
    return {
        "n_tests": n_tests,
        "n_sig_raw": n_sig,
        "feature": "power",
        "records": all_tsv_records,
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
        selected_bands=ctx.selected_bands,
    )


###################################################################
# ITPC Temporal Correlations
###################################################################


def _compute_itpc_for_window(
    tfr_complex: np.ndarray,
    fmin: float,
    fmax: float,
    t0: float,
    t1: float,
    times: np.ndarray,
    freqs: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute ITPC for a specific time window and frequency band.
    
    ITPC is computed as the magnitude of the mean complex-valued phase unit vectors:
        ITPC = |mean(exp(1j * phase))|
    
    Returns shape (n_channels,) with mean ITPC across trials for each channel.
    """
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    time_mask = (times >= t0) & (times < t1)
    
    if not freq_mask.any() or not time_mask.any():
        return None
    
    # tfr_complex shape: (n_trials, n_channels, n_freqs, n_times)
    # Extract relevant frequency and time bins
    data_sub = tfr_complex[:, :, freq_mask, :][:, :, :, time_mask]
    
    if data_sub.size == 0:
        return None
    
    # Extract phase from complex TFR
    phase = np.angle(data_sub)
    
    # Compute phase unit vectors and average across trials
    # ITPC = |mean_trials(exp(1j*phase))|
    phase_vectors = np.exp(1j * phase)  # (n_trials, n_ch, n_freq_sub, n_time_sub)
    mean_vector = np.mean(phase_vectors, axis=0)  # (n_ch, n_freq_sub, n_time_sub)
    
    # Take magnitude (ITPC) and average across freq and time bins
    itpc = np.abs(mean_vector).mean(axis=(1, 2))  # (n_ch,)
    
    return itpc


def _extract_trial_itpc(
    tfr_complex: np.ndarray,
    fmin: float,
    fmax: float,
    t0: float,
    t1: float,
    times: np.ndarray,
    freqs: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract per-trial ITPC approximation for correlation analysis.
    
    For correlation with behavior, we need a per-trial metric.
    We use the trial's contribution to the overall phase coherence:
        trial_itpc = cos(phase_trial - circular_mean_phase)
    
    This gives higher values for trials whose phase aligns with the mean.
    
    Returns shape (n_trials, n_channels).
    """
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    time_mask = (times >= t0) & (times < t1)
    
    if not freq_mask.any() or not time_mask.any():
        return None
    
    # tfr_complex shape: (n_trials, n_channels, n_freqs, n_times)
    data_sub = tfr_complex[:, :, freq_mask, :][:, :, :, time_mask]
    
    if data_sub.size == 0:
        return None
    
    n_trials, n_ch = data_sub.shape[0], data_sub.shape[1]
    
    # Extract phase
    phase = np.angle(data_sub)  # (n_trials, n_ch, n_freq_sub, n_time_sub)
    
    # Compute circular mean phase across trials for each (ch, freq, time)
    phase_vectors = np.exp(1j * phase)
    mean_vector = np.mean(phase_vectors, axis=0)  # (n_ch, n_freq_sub, n_time_sub)
    mean_phase = np.angle(mean_vector)  # circular mean phase
    
    # Compute each trial's alignment with the mean phase
    # cos(phase_trial - mean_phase) gives +1 if aligned, -1 if anti-aligned
    phase_diff = phase - mean_phase[np.newaxis, :, :, :]
    alignment = np.cos(phase_diff)  # (n_trials, n_ch, n_freq_sub, n_time_sub)
    
    # Average across freq and time bins
    trial_itpc = alignment.mean(axis=(2, 3))  # (n_trials, n_ch)
    
    return trial_itpc


def _run_itpc_temporal_by_condition_core(
    epochs,
    events,
    y,
    stats_dir: Path,
    config,
    use_spearman: bool,
    cov_df,
    logger,
    selected_bands: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Core implementation for ITPC temporal correlations by condition.
    
    Uses user-configurable:
    - condition_column: which events column to split by
    - filter_value: optionally compute only for a specific value  
    - time windows: computed from config time_range_ms and resolution
    """
    from eeg_pipeline.utils.analysis.tfr import compute_complex_tfr
    
    if epochs is None or events is None or y is None:
        return None
    if not epochs.preload:
        epochs.load_data()
    
    # Get temporal config
    temporal_cfg = config.get("behavior_analysis.temporal", {}) or {}
    itpc_cfg = temporal_cfg.get("itpc", {}) or {}
    
    baseline_correction = bool(itpc_cfg.get("baseline_correction", True))
    baseline_window = itpc_cfg.get("baseline_window", [-0.5, -0.01])
    
    # Time window configuration from user settings
    time_range_ms = temporal_cfg.get("time_range_ms", [-200, 1000])
    time_resolution_ms = temporal_cfg.get("time_resolution_ms", 50)
    tmin_s = time_range_ms[0] / 1000.0
    tmax_s = time_range_ms[1] / 1000.0
    win_size_s = time_resolution_ms / 1000.0
    
    # Compute complex TFR for phase extraction
    tfr_complex = compute_complex_tfr(epochs, config, logger=logger)
    if tfr_complex is None:
        return None
    
    tfr_data = tfr_complex.data  # (n_trials, n_ch, n_freqs, n_times)
    times = np.asarray(tfr_complex.times)
    freqs = np.asarray(tfr_complex.freqs)
    ch_names = tfr_complex.ch_names
    
    # Use user-configured time range, clamped to available data
    clipped = clip_time_range(times, tmin_s, tmax_s)
    if clipped is None:
        logger.warning("ITPC temporal: no valid time range after clipping")
        return None
    win_s, win_e = build_time_windows_fixed_size_clamped(clipped[0], clipped[1], win_size_s)
    if len(win_s) == 0:
        logger.warning("ITPC temporal: no time windows generated")
        return None
    
    y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    n = compute_aligned_data_length(tfr_complex, events)
    
    condition_values, condition_vec = _determine_condition_values(
        events, n, temporal_cfg, config, logger, analysis_name="ITPC temporal"
    )
    
    fmax = float(np.max(freqs))
    all_bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    
    # Filter bands if user selected specific ones
    if selected_bands:
        bands = {k: v for k, v in all_bands.items() if k.lower() in [b.lower() for b in selected_bands]}
        if not bands:
            logger.warning(f"ITPC temporal: no matching bands in {selected_bands}, using all bands")
            bands = all_bands
        else:
            logger.info(f"ITPC temporal: using selected bands {list(bands.keys())}")
    else:
        bands = all_bands
    
    corr_fn = spearmanr if use_spearman else pearsonr
    sfx = "_spearman" if use_spearman else "_pearson"
    method = "spearman" if use_spearman else "pearson"
    
    out_dir = stats_dir / "temporal_correlations"
    ensure_dir(out_dir)
    
    all_tsv_records = []
    itpc_results = {}
    
    for cond_val in condition_values:
        if cond_val == "all":
            mask = np.ones(n, dtype=bool)
        else:
            mask = condition_vec == cond_val
        
        if not np.any(mask):
            logger.info(f"ITPC temporal: no trials for condition '{cond_val}', skipping")
            continue
        
        idx = np.where(mask)[0]
        
        safe_name = str(cond_val).replace(" ", "_").replace("/", "_")
        tfr_c = tfr_data[idx]
        y_c = y_arr[mask]
        
        band_names = list(bands.keys())
        n_ch, n_b, n_w = len(ch_names), len(band_names), len(win_s)
        
        corrs = np.full((n_b, n_w, n_ch), np.nan)
        pvals = np.full_like(corrs, np.nan)
        n_valid = np.zeros_like(corrs, dtype=int)
        
        for band_idx, (band_name, (fmin, fmax_band)) in enumerate(bands.items()):
            fmax_effective = min(fmax_band, fmax)
            if fmin >= fmax_effective:
                continue
            
            for window_idx, (t0, t1) in enumerate(zip(win_s, win_e)):
                trial_itpc = _extract_trial_itpc(
                    tfr_c, fmin, fmax_effective, t0, t1, times, freqs
                )
                if trial_itpc is None:
                    continue
                
                if baseline_correction and baseline_window:
                    baseline_itpc = _extract_trial_itpc(
                        tfr_c,
                        fmin,
                        fmax_effective,
                        baseline_window[0],
                        baseline_window[1],
                        times,
                        freqs,
                    )
                    if baseline_itpc is not None:
                        trial_itpc = trial_itpc - baseline_itpc
                
                for channel_idx in range(n_ch):
                    channel_vals = trial_itpc[:, channel_idx]
                    valid = np.isfinite(channel_vals) & np.isfinite(y_c)
                    n_obs = int(valid.sum())
                    
                    if n_obs < MIN_OBSERVATIONS_FOR_CORRELATION:
                        continue
                    
                    if np.std(channel_vals[valid]) < MIN_VARIANCE_THRESHOLD:
                        continue
                    
                    try:
                        r, p = corr_fn(channel_vals[valid], y_c[valid])
                        if np.isfinite(r) and np.isfinite(p):
                            corrs[band_idx, window_idx, channel_idx] = float(r)
                            pvals[band_idx, window_idx, channel_idx] = float(p)
                            n_valid[band_idx, window_idx, channel_idx] = n_obs
                    except (ValueError, RuntimeWarning):
                        pass
        
        res = {
            "name": safe_name,
            "feature": "itpc",
            "correlations": corrs,
            "p_values": pvals,
            "n_valid": n_valid,
            "band_names": band_names,
            "window_starts": win_s,
            "window_ends": win_e,
        }
        itpc_results[safe_name] = res
        
        np.savez_compressed(
            out_dir / f"temporal_itpc_{safe_name}{sfx}.npz",
            **res, times=times, ch_names=ch_names,
        )
        
        cond_records = []
        for band_idx, band_name in enumerate(band_names):
            for window_idx in range(n_w):
                for channel_idx in range(n_ch):
                    r = corrs[band_idx, window_idx, channel_idx]
                    p = pvals[band_idx, window_idx, channel_idx]
                    if not np.isfinite(r) or not np.isfinite(p):
                        continue
                    cond_records.append({
                        "condition": safe_name,
                        "feature": "itpc",
                        "band": band_name,
                        "time_start": float(win_s[window_idx]),
                        "time_end": float(win_e[window_idx]),
                        "channel": ch_names[channel_idx],
                        "r": float(r),
                        "beta_std": float(r),
                        "beta_kind": "standardized",
                        "p": float(p),
                        "n": int(n_valid[band_idx, window_idx, channel_idx]),
                        "method": method,
                        "method_label": format_correlation_method_label(method, None),
                    })
        
        all_tsv_records.extend(cond_records)
        logger.info(f"Computed ITPC temporal for condition '{safe_name}': {len(cond_records)} tests")
    
    if itpc_results:
        combined_data = dict(itpc_results)
        combined_data["times"] = times
        combined_data["ch_names"] = ch_names
        np.savez_compressed(out_dir / f"temporal_itpc_combined{sfx}.npz", **combined_data)
        logger.info(f"Saved ITPC temporal arrays: {len(itpc_results)} conditions")
    
    logger.info("ITPC temporal correlations completed")
    
    n_tests = len(all_tsv_records)
    n_sig = sum(1 for r in all_tsv_records if r.get("p", 1.0) < 0.05) if all_tsv_records else 0
    
    return {
        "n_tests": n_tests,
        "n_sig_raw": n_sig,
        "feature": "itpc",
        "records": all_tsv_records,
    }


def compute_itpc_temporal_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Compute ITPC temporal correlations by condition using pre-loaded data.
    
    Note: Feature selection is handled by the orchestration layer based on
    which feature files the user selected in step 3 (feature selection).
    """
    return _run_itpc_temporal_by_condition_core(
        ctx.epochs,
        ctx.aligned_events,
        ctx.targets,
        ctx.stats_dir,
        ctx.config,
        ctx.use_spearman,
        ctx.covariates_df,
        ctx.logger,
        selected_bands=ctx.selected_bands,
    )


###################################################################
# ERDS Temporal Correlations
###################################################################


def _extract_trial_erds(
    tfr_power: np.ndarray,
    fmin: float,
    fmax: float,
    t0: float,
    t1: float,
    bl_start: float,
    bl_end: float,
    times: np.ndarray,
    freqs: np.ndarray,
    method: str = "percent",
) -> Optional[np.ndarray]:
    """Extract per-trial ERDS (event-related desync/sync) for correlation analysis.
    
    ERDS = (active_power - baseline_power) / baseline_power * 100  [percent method]
    ERDS = (active_power - baseline_power) / baseline_std           [zscore method]
    
    Returns shape (n_trials, n_channels).
    """
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    time_mask = (times >= t0) & (times < t1)
    bl_mask = (times >= bl_start) & (times < bl_end)
    
    if not freq_mask.any() or not time_mask.any() or not bl_mask.any():
        return None
    
    # tfr_power shape: (n_trials, n_channels, n_freqs, n_times)
    active_data = tfr_power[:, :, freq_mask, :][:, :, :, time_mask]
    baseline_data = tfr_power[:, :, freq_mask, :][:, :, :, bl_mask]
    
    if active_data.size == 0 or baseline_data.size == 0:
        return None
    
    # Average power across freq and time bins
    active_mean = active_data.mean(axis=(2, 3))  # (n_trials, n_ch)
    baseline_mean = baseline_data.mean(axis=(2, 3))  # (n_trials, n_ch)
    
    # Compute ERDS normalization
    if method == "zscore":
        baseline_std = baseline_data.std(axis=(2, 3))  # (n_trials, n_ch)
        baseline_std = np.where(
            baseline_std < MIN_VARIANCE_THRESHOLD,
            MIN_VARIANCE_THRESHOLD,
            baseline_std,
        )
        erds = (active_mean - baseline_mean) / baseline_std
    else:  # percent (default)
        baseline_mean_safe = np.where(
            np.abs(baseline_mean) < MIN_VARIANCE_THRESHOLD,
            MIN_VARIANCE_THRESHOLD,
            baseline_mean,
        )
        erds = (active_mean - baseline_mean) / baseline_mean_safe * 100.0
    
    return erds


def _run_erds_temporal_by_condition_core(
    epochs,
    events,
    y,
    stats_dir: Path,
    config,
    use_spearman: bool,
    cov_df,
    logger,
    selected_bands: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Core implementation for ERDS temporal correlations by condition."""
    if epochs is None or events is None or y is None:
        return None
    if not epochs.preload:
        epochs.load_data()
    
    temporal_cfg = config.get("behavior_analysis.temporal", {}) or {}
    erds_cfg = temporal_cfg.get("erds", {}) or {}
    
    baseline_window = erds_cfg.get("baseline_window", [-0.5, -0.1])
    method = str(erds_cfg.get("method", "percent")).lower()
    
    time_range_ms = temporal_cfg.get("time_range_ms", [-200, 1000])
    time_resolution_ms = temporal_cfg.get("time_resolution_ms", 50)
    tmin_s = time_range_ms[0] / 1000.0
    tmax_s = time_range_ms[1] / 1000.0
    win_size_s = time_resolution_ms / 1000.0
    
    tfr = compute_tfr_morlet(epochs, config, logger=logger)
    if tfr is None:
        return None
    # NOTE: Do NOT apply baseline correction here - ERDS computes its own 
    # per-trial baseline normalization. Applying baseline correction would
    # result in double normalization which is scientifically incorrect.
    
    tfr_data = tfr.data
    times = np.asarray(tfr.times)
    freqs = np.asarray(tfr.freqs)
    ch_names = tfr.ch_names
    
    clipped = clip_time_range(times, tmin_s, tmax_s)
    if clipped is None:
        return None
    win_s, win_e = build_time_windows_fixed_size_clamped(clipped[0], clipped[1], win_size_s)
    if len(win_s) == 0:
        return None
    
    y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    n = compute_aligned_data_length(tfr, events)
    
    condition_values, condition_vec = _determine_condition_values(
        events, n, temporal_cfg, config, logger, analysis_name="ERDS temporal"
    )
    
    fmax = float(np.max(freqs))
    all_bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    if selected_bands:
        selected_bands_lower = [b.lower() for b in selected_bands]
        bands = {
            k: v
            for k, v in all_bands.items()
            if k.lower() in selected_bands_lower
        } or all_bands
    else:
        bands = all_bands
    
    corr_fn = spearmanr if use_spearman else pearsonr
    sfx = "_spearman" if use_spearman else "_pearson"
    corr_method = "spearman" if use_spearman else "pearson"
    
    out_dir = stats_dir / "temporal_correlations"
    ensure_dir(out_dir)
    
    all_tsv_records = []
    erds_results = {}
    
    for cond_val in condition_values:
        mask = np.ones(n, dtype=bool) if cond_val == "all" else (condition_vec == cond_val)
        if not np.any(mask):
            continue
        
        idx = np.where(mask)[0]
        
        safe_name = str(cond_val).replace(" ", "_").replace("/", "_")
        tfr_c = tfr_data[idx]
        y_c = y_arr[mask]
        
        band_names = list(bands.keys())
        n_ch, n_b, n_w = len(ch_names), len(band_names), len(win_s)
        
        corrs = np.full((n_b, n_w, n_ch), np.nan)
        pvals = np.full_like(corrs, np.nan)
        n_valid = np.zeros_like(corrs, dtype=int)
        
        for band_idx, (band_name, (fmin, fmax_band)) in enumerate(bands.items()):
            fmax_effective = min(fmax_band, fmax)
            if fmin >= fmax_effective:
                continue
            
            for window_idx, (t0, t1) in enumerate(zip(win_s, win_e)):
                trial_erds = _extract_trial_erds(
                    tfr_c,
                    fmin,
                    fmax_effective,
                    t0,
                    t1,
                    baseline_window[0],
                    baseline_window[1],
                    times,
                    freqs,
                    method,
                )
                if trial_erds is None:
                    continue
                
                for channel_idx in range(n_ch):
                    channel_vals = trial_erds[:, channel_idx]
                    valid = np.isfinite(channel_vals) & np.isfinite(y_c)
                    n_obs = int(valid.sum())
                    if (
                        n_obs < MIN_OBSERVATIONS_FOR_CORRELATION
                        or np.std(channel_vals[valid]) < MIN_VARIANCE_THRESHOLD
                    ):
                        continue
                    try:
                        r, p = corr_fn(channel_vals[valid], y_c[valid])
                        if np.isfinite(r) and np.isfinite(p):
                            corrs[band_idx, window_idx, channel_idx] = float(r)
                            pvals[band_idx, window_idx, channel_idx] = float(p)
                            n_valid[band_idx, window_idx, channel_idx] = n_obs
                    except (ValueError, RuntimeWarning):
                        pass
        
        res = {
            "name": safe_name,
            "feature": "erds",
            "correlations": corrs,
            "p_values": pvals,
            "n_valid": n_valid,
            "band_names": band_names,
            "window_starts": win_s,
            "window_ends": win_e,
        }
        erds_results[safe_name] = res
        np.savez_compressed(
            out_dir / f"temporal_erds_{safe_name}{sfx}.npz",
            **res,
            times=times,
            ch_names=ch_names,
        )
        
        cond_records = []
        for band_idx, band_name in enumerate(band_names):
            for window_idx in range(n_w):
                for channel_idx in range(n_ch):
                    r = corrs[band_idx, window_idx, channel_idx]
                    p = pvals[band_idx, window_idx, channel_idx]
                    if not np.isfinite(r) or not np.isfinite(p):
                        continue
                    cond_records.append({
                        "condition": safe_name,
                        "feature": "erds",
                        "band": band_name,
                        "time_start": float(win_s[window_idx]),
                        "time_end": float(win_e[window_idx]),
                        "channel": ch_names[channel_idx],
                        "r": float(r),
                        "beta_std": float(r),
                        "beta_kind": "standardized",
                        "p": float(p),
                        "n": int(n_valid[band_idx, window_idx, channel_idx]),
                        "method": corr_method,
                        "method_label": format_correlation_method_label(corr_method, None),
                    })
        all_tsv_records.extend(cond_records)
        logger.info(f"Computed ERDS temporal for condition '{safe_name}': {len(cond_records)} tests")
    
    if erds_results:
        np.savez_compressed(out_dir / f"temporal_erds_combined{sfx}.npz", **erds_results, times=times, ch_names=ch_names)
    
    logger.info(f"ERDS temporal correlations: {len(all_tsv_records)} tests")
    return {"n_tests": len(all_tsv_records), "n_sig_raw": sum(1 for r in all_tsv_records if r.get("p", 1.0) < 0.05), "feature": "erds", "records": all_tsv_records}


def compute_erds_temporal_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Compute ERDS temporal correlations by condition using pre-loaded data.
    
    Note: Feature selection is handled by the orchestration layer based on
    which feature files the user selected in step 3 (feature selection).
    """
    return _run_erds_temporal_by_condition_core(
        ctx.epochs, ctx.aligned_events, ctx.targets, ctx.stats_dir, ctx.config,
        ctx.use_spearman, ctx.covariates_df, ctx.logger, selected_bands=ctx.selected_bands,
    )
