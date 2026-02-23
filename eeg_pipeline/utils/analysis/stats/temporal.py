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
from eeg_pipeline.utils.data.tfr_alignment import compute_aligned_data_length
from eeg_pipeline.utils.data.columns import get_pain_column_from_config
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import write_tsv

if TYPE_CHECKING:
    from eeg_pipeline.context.behavior import BehaviorContext


# Numerical stability constants
MIN_VARIANCE_THRESHOLD = 1e-12
MIN_DENOMINATOR_THRESHOLD = 1e-15
MIN_OBSERVATIONS_FOR_CORRELATION = 10
PARALLELIZATION_TASK_THRESHOLD = 50


def _get_temporal_target_column(config: Any) -> Optional[str]:
    """Return explicit temporal target column override (events column name) if configured."""
    raw = get_config_value(config, "behavior_analysis.temporal.target_column", None)
    if raw is None:
        return None
    col = str(raw).strip()
    return col or None


def _get_temporal_targets_from_events(
    events: pd.DataFrame,
    *,
    config: Any,
    logger: logging.Logger,
    analysis_name: str,
) -> pd.Series:
    """Resolve and extract the behavioral target series for temporal analyses."""
    if events is None or not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")

    target_col = _get_temporal_target_column(config)
    if target_col is not None:
        if target_col not in events.columns:
            raise ValueError(
                f"{analysis_name}: temporal target_column='{target_col}' not found in events. "
                f"Available columns: {list(events.columns)}"
            )
        logger.info("%s: using temporal target column '%s'", analysis_name, target_col)
        return pd.to_numeric(events[target_col], errors="coerce")

    rating_columns = (
        list(config.get("event_columns.rating", []) or [])
        if config is not None and hasattr(config, "get")
        else []
    )
    from eeg_pipeline.utils.data.columns import pick_target_column

    rating_col = pick_target_column(events, target_columns=rating_columns)
    if rating_col is None:
        raise ValueError(
            f"{analysis_name}: no rating column found. Configure event_columns.rating or set "
            f"behavior_analysis.temporal.target_column. Available columns: {list(events.columns)}"
        )
    logger.info("%s: using rating column '%s'", analysis_name, rating_col)
    return pd.to_numeric(events[rating_col], errors="coerce")


def _to_numpy_array(y: Any) -> np.ndarray:
    """Convert y to numpy array, handling pandas Series."""
    return y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)


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
    groups: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Compute channel/band/window correlations for a single condition."""
    if mask is None:
        return None

    is_bool_mask = mask.dtype == bool
    idx = np.where(mask)[0] if is_bool_mask else np.asarray(mask)
    tfr_c, y_c = tfr[idx], (y[mask] if is_bool_mask else y[idx])
    if groups is not None:
        try:
            groups_arr = np.asarray(groups)
            groups_c = groups_arr[mask] if groups_arr.shape[0] == y.shape[0] else groups_arr
        except Exception:
            groups_c = None
    else:
        groups_c = None
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

    # Always require a minimum number of observations for stable inference.
    req_samples = max(int(req_samples), int(MIN_OBSERVATIONS_FOR_CORRELATION))

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

    c_alpha = float(
        get_config_value(config, "behavior_analysis.cluster.alpha", None) or
        get_config_value(config, "behavior_analysis.cluster_correction.alpha", alpha)
    )
    n_cluster_perm = int(
        get_config_value(config, "behavior_analysis.cluster.n_permutations", None) or
        get_config_value(config, "behavior_analysis.cluster_correction.n_permutations", 0)
    )
    cluster_forming_threshold = (
        get_config_value(config, "behavior_analysis.cluster.forming_threshold", None) or
        get_config_value(config, "behavior_analysis.cluster_correction.cluster_forming_threshold", None) or
        get_config_value(config, "behavior_analysis.cluster_correction.forming_threshold", None)
    )
    # Cluster engine derives thresholds in t-stat space; if a user config provides
    # a very small value (often a p-value like 0.05), ignore it and derive instead.
    if cluster_forming_threshold is not None:
        try:
            threshold_float = float(cluster_forming_threshold)
            if threshold_float < 1.0:
                if logger:
                    logger.warning(
                        "%s: cluster_forming_threshold=%.4g looks like a p-value/r threshold; deriving threshold by permutation instead.",
                        name,
                        threshold_float,
                    )
                cluster_forming_threshold = None
        except (ValueError, TypeError):
            cluster_forming_threshold = None
    seed = int(get_config_value(config, "project.random_state", 42))
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

        # Convert observed r -> t for cluster-forming and mass computation.
        n_cov = int(cov_vals.shape[1]) if cov_vals is not None else 0
        band_stat = np.full_like(band_corrs, np.nan, dtype=float)
        for window_idx, channel_idx in band_info_bins:
            r_val = band_corrs[window_idx, channel_idx]
            n_obs = int(n_valid[band_idx, window_idx, channel_idx])
            dof = n_obs - n_cov - 2
            if np.isfinite(r_val) and dof > 0 and abs(r_val) < 1:
                denom = max(MIN_DENOMINATOR_THRESHOLD, 1.0 - float(r_val) ** 2)
                band_stat[window_idx, channel_idx] = float(r_val) * np.sqrt(float(dof) / denom)

        # IMPORTANT validity fix:
        # The band grid is (window_idx x channel_idx). There is no scientifically valid
        # adjacency defined across channel index here, so cluster correction must NOT
        # cluster across channels. We instead perform time-only clustering separately
        # for each channel by reshaping (n_windows,) -> (n_windows, 1).

        band_record: Dict[str, Any] = {"band": band_name, "channels": []}
        band_perm_record: Dict[str, Any] = {"band": band_name, "channels": []}
        band_thresh_record: Dict[str, Any] = {"band": band_name, "channels": []}
        band_mass_record: Dict[str, Any] = {"band": band_name, "channels": []}

        n_ch = int(band_stat.shape[1])
        for channel_idx in range(n_ch):
            ch_bins = [(w_idx, 0) for (w_idx, c_idx) in band_info_bins if c_idx == channel_idx]
            if not ch_bins:
                continue

            ch_stat = band_stat[:, channel_idx][:, None]  # (n_windows, 1)
            ch_pvals = band_pvals[:, channel_idx][:, None]
            ch_data = band_data[:, channel_idx, :][:, None, :]  # (n_windows, 1, n_trials)

            labels, pvals_corr, sig_mask, records, perm_max, c_thresh = compute_cluster_correction_2d(
                correlations=ch_stat,
                p_values=ch_pvals,
                bin_data=ch_data,
                informative_bins=ch_bins,
                y_array=y_c,
                cluster_alpha=c_alpha,
                n_cluster_perm=n_cluster_perm,
                alpha=c_alpha,
                min_valid_points=req_samples,
                use_spearman=use_spearman,
                cluster_rng=cluster_rng,
                covariates_matrix=cov_vals,
                groups=groups_c,
                cluster_forming_threshold=cluster_forming_threshold,
                config=config,
            )

            cluster_labels[band_idx, :, channel_idx] = labels[:, 0]
            p_corrected[band_idx, :, channel_idx] = pvals_corr[:, 0]
            cluster_sig[band_idx, :, channel_idx] = sig_mask[:, 0]

            band_record["channels"].append(
                {"channel_idx": int(channel_idx), "clusters": records}
            )
            band_perm_record["channels"].append(
                {"channel_idx": int(channel_idx), "perm_max_masses": perm_max}
            )
            band_thresh_record["channels"].append(
                {"channel_idx": int(channel_idx), "cluster_forming_threshold": float(c_thresh)}
            )

            _, masses = compute_cluster_masses_2d(
                ch_stat,
                ch_pvals,
                cluster_alpha=c_alpha,
                cluster_forming_threshold=float(c_thresh),
                config=config,
            )
            band_mass_record["channels"].append(
                {"channel_idx": int(channel_idx), "masses": masses}
            )

        cluster_records.append(band_record)
        cluster_perm_max.append(band_perm_record)
        cluster_thresholds.append(band_thresh_record)
        cluster_masses.append(band_mass_record)

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
    requested_values = temporal_cfg.get("condition_values", []) or []

    if not condition_column:
        condition_column = get_pain_column_from_config(config, events)

    if split_by_condition and condition_column and condition_column in events.columns:
        condition_vec = events[condition_column].to_numpy()[:n_trials]

        if isinstance(requested_values, (list, tuple)) and len(requested_values) > 0:
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
    else:
        if not split_by_condition:
            logger.info(f"{analysis_name}: split_by_condition=False, computing over all trials")
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

    temporal_cfg = config.get("behavior_analysis.temporal", {}) or {}
    roi_selection = temporal_cfg.get("roi_selection")
    epochs_tfr = restrict_epochs_to_roi(epochs, roi_selection, config, logger)
    tfr = compute_tfr_morlet(epochs_tfr, config, logger=logger)
    if tfr is None:
        return

    apply_baseline_to_tfr(tfr, config, logger)
    power, times, freqs = tfr.data.mean(axis=1), tfr.times, tfr.freqs
    y_arr = y.to_numpy()

    # Apply windowing
    tw = temporal_cfg.get("time_window")
    if tw is None:
        range_ms = temporal_cfg.get("time_range_ms", None)
        if isinstance(range_ms, (list, tuple)) and len(range_ms) >= 2:
            tw = [float(range_ms[0]) / 1000.0, float(range_ms[1]) / 1000.0]
    if tw:
        tm = (times >= float(tw[0])) & (times <= float(tw[1]))
        if not tm.any():
            return
        times, power = times[tm], power[:, :, tm]
    fr = temporal_cfg.get("freq_range")
    if fr:
        fm = (freqs >= float(fr[0])) & (freqs <= float(fr[1]))
        if not fm.any():
            return
        freqs, power = freqs[fm], power[:, fm, :]

    time_res = None
    if temporal_cfg.get("time_resolution_ms", None) is not None:
        time_res = float(temporal_cfg.get("time_resolution_ms")) / 1000.0
    if time_res is None:
        time_res = 0.1
    time_edges = np.arange(times[0], times[-1] + float(time_res), float(time_res))
    min_pts = int(config.get("behavior_analysis.statistics.min_samples_roi", 20))

    tf_jobs_cfg = temporal_cfg.get("n_jobs", None)
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

    n_perm_cfg = int(
        get_config_value(config, "behavior_analysis.cluster.n_permutations", None) or
        get_config_value(config, "behavior_analysis.cluster_correction.n_permutations", 100)
    )
    n_perm = max(n_perm_cfg, int(temporal_cfg.get("n_cluster_perm", 0)))
    c_alpha = float(
        get_config_value(config, "behavior_analysis.cluster.alpha", None) or
        get_config_value(config, "behavior_analysis.cluster_correction.alpha", None) or
        get_config_value(config, "statistics.sig_alpha", 0.05)
    )
    rng = np.random.default_rng(int(get_config_value(config, "project.random_state", 42)))
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
        run_col = str(get_config_value(config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
        groups = events[run_col].to_numpy() if run_col and events is not None and run_col in events.columns else None
        c_labels, c_pvals, c_sig, c_recs, perm_masses, c_thresh = compute_cluster_correction_2d(
            correlations=cluster_stat,
            p_values=pvals,
            bin_data=bin_data,
            informative_bins=info_bins,
            y_array=y_arr,
            cluster_alpha=c_alpha,
            n_cluster_perm=n_perm,
            alpha=c_alpha,
            min_valid_points=min_pts,
            use_spearman=use_spearman,
            cluster_rng=rng,
            covariates_matrix=cov_mat,
            groups=groups,
            config=config,
        )

    ensure_dir(stats_dir)

    sfx = "_spearman" if use_spearman else "_pearson"
    roi = roi_selection
    f"_{roi.lower()}" if roi and roi != "null" else ""
    roi_label = roi or "all"

    method = "spearman" if use_spearman else "pearson"
    
    recs = _build_tf_grid_records(
        corrs, pvals, n_valid, times, freqs, time_edges, method, roi_label,
        c_pvals if n_perm > 0 else None,
        c_labels if n_perm > 0 else None,
        c_sig if n_perm > 0 else None,
    )
    
    temporal_dir = stats_dir
    ensure_dir(temporal_dir)
    if recs:
        tf_df = pd.DataFrame(recs)
        write_tsv(tf_df, temporal_dir / f"tf_grid_{roi_label.lower()}{sfx}.tsv")

    logger.info(f"Saved TF correlations: shape={corrs.shape}, info_bins={len(info_bins)}")
    
    return {
        "n_tests": len(info_bins),
        "n_sig_raw": int((pvals[np.isfinite(pvals)] < 0.05).sum()) if pvals is not None else 0,
        "records": recs,
    }


def _build_temporal_tsv_records(
    res: Dict[str, Any],
    condition: str,
    ch_names: List[str],
    method: str,
) -> List[Dict[str, Any]]:
    """Build TSV records from temporal correlation results for global FDR.
    
    Includes cluster correction fields when available for compatibility with
    unified temporal correlations output format.
    """
    method_label = format_correlation_method_label(method, None)
    records = []
    corrs = res["correlations"]  # shape: (n_bands, n_windows, n_channels)
    pvals = res["p_values"]
    n_valid = res["n_valid"]
    band_names = res["band_names"]
    win_s = res["window_starts"]
    win_e = res["window_ends"]
    
    # Cluster correction fields (optional)
    p_corrected = res.get("p_corrected")
    cluster_labels = res.get("cluster_labels")
    cluster_sig = res.get("mask")

    n_bands, n_windows, n_channels = corrs.shape
    for band_idx in range(n_bands):
        for window_idx in range(n_windows):
            for channel_idx in range(n_channels):
                r = corrs[band_idx, window_idx, channel_idx]
                p = pvals[band_idx, window_idx, channel_idx]
                if not np.isfinite(r) or not np.isfinite(p):
                    continue
                
                p_cluster = np.nan
                cluster_id = 0
                is_cluster_sig = False
                if p_corrected is not None:
                    p_cluster = float(p_corrected[band_idx, window_idx, channel_idx])
                if cluster_labels is not None:
                    cluster_id = int(cluster_labels[band_idx, window_idx, channel_idx])
                if cluster_sig is not None:
                    is_cluster_sig = bool(cluster_sig[band_idx, window_idx, channel_idx])
                
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
                    "beta_std": float(r),
                    "beta_kind": "standardized",
                    "p": float(p),
                    "p_cluster": p_cluster if np.isfinite(p_cluster) else np.nan,
                    "cluster_id": cluster_id,
                    "cluster_significant": is_cluster_sig,
                    "n": int(n_valid[band_idx, window_idx, channel_idx]),
                    "method": method,
                    "method_label": method_label,
                })
    return records


def _compute_roi_correlations_for_condition(
    tfr,
    y: np.ndarray,
    mask: np.ndarray,
    condition_name: str,
    bands: Dict[str, Tuple[float, float]],
    win_s: np.ndarray,
    win_e: np.ndarray,
    *,
    fmax_available: float,
    corr_fn,
    logger,
    cov_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
    roi_definitions: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Compute ROI-level temporal correlations from ROI-averaged trial-wise power.

    Scientific validity:
    - Computes the correlation of a *single ROI summary per trial* with behavior,
      rather than meta-combining channel-wise correlations/p-values.
    - Supports partial correlation when covariates are provided.
    - Optionally applies time-only cluster correction across windows (for each ROI/band),
      using the same permutation machinery as channel-level temporal correlations.
    """
    if mask is None:
        return []

    is_bool_mask = getattr(mask, "dtype", None) == bool
    idx = np.where(mask)[0] if is_bool_mask else np.asarray(mask)
    if idx.size == 0:
        return []

    tfr_c = tfr[idx]
    y_c = y[mask] if is_bool_mask else y[idx]
    use_spearman = corr_fn == spearmanr
    method = "spearman" if use_spearman else "pearson"
    method_label = format_correlation_method_label(method, None)

    cov_vals = None
    cov_cols: Optional[List[str]] = None
    if cov_df is not None and not cov_df.empty:
        cov_df_c = cov_df.iloc[idx].apply(pd.to_numeric, errors="coerce")
        cov_vals = cov_df_c.to_numpy()
        cov_cols = list(cov_df_c.columns)

    # Minimum samples for stable correlation / valid t-stat conversion
    min_samples = int(
        get_config_value(
            config, "behavior_analysis.statistics.min_observations_for_correlation", MIN_OBSERVATIONS_FOR_CORRELATION
        )
    )
    min_samples = max(min_samples, MIN_OBSERVATIONS_FOR_CORRELATION)

    if cov_vals is not None:
        min_samples_per_cov = int(get_config_value(config, "behavior_analysis.statistics.min_samples_per_covariate", 5))
        partial_corr_base = int(get_config_value(config, "behavior_analysis.statistics.partial_corr_base_samples", 5))
        req_samples = int(cov_vals.shape[1]) * int(min_samples_per_cov) + int(partial_corr_base)
        req_samples = max(req_samples, min_samples)
    else:
        req_samples = min_samples

    # Cluster correction config
    n_cluster_perm = int(
        get_config_value(config, "behavior_analysis.cluster.n_permutations", None)
        or get_config_value(config, "behavior_analysis.cluster_correction.n_permutations", 0)
    )
    c_alpha = float(
        get_config_value(config, "behavior_analysis.cluster.alpha", None)
        or get_config_value(config, "behavior_analysis.cluster_correction.alpha", get_config_value(config, "statistics.sig_alpha", 0.05))
    )
    cluster_forming_threshold = (
        get_config_value(config, "behavior_analysis.cluster.forming_threshold", None)
        or get_config_value(config, "behavior_analysis.cluster_correction.cluster_forming_threshold", None)
        or get_config_value(config, "behavior_analysis.cluster_correction.forming_threshold", None)
    )
    if cluster_forming_threshold is not None:
        try:
            threshold_float = float(cluster_forming_threshold)
            if threshold_float < 1.0:
                if logger:
                    logger.warning(
                        "%s: cluster_forming_threshold=%.4g looks like a p-value/r threshold; deriving threshold by permutation instead.",
                        condition_name,
                        threshold_float,
                    )
                cluster_forming_threshold = None
        except (ValueError, TypeError):
            cluster_forming_threshold = None

    seed = int(get_config_value(config, "project.random_state", 42))
    cluster_rng = np.random.default_rng(seed)

    if groups is not None:
        try:
            groups_arr = np.asarray(groups)
            groups_c = groups_arr
        except Exception:
            groups_c = None
    else:
        groups_c = None

    ch_names = list(getattr(tfr, "ch_names", []))
    n_ch = len(ch_names)
    if n_ch == 0:
        return []

    # Build ROI map: include an explicit "all" ROI.
    roi_defs: Dict[str, List[str]] = {}
    if roi_definitions:
        roi_defs.update({str(k): list(v) for k, v in roi_definitions.items() if isinstance(v, (list, tuple))})
    roi_defs.setdefault("all", ch_names)

    roi_to_indices: Dict[str, List[int]] = {}
    for roi_name, roi_channels in roi_defs.items():
        indices = [i for i, ch in enumerate(ch_names) if ch in set(roi_channels)]
        if indices:
            roi_to_indices[str(roi_name)] = indices

    if not roi_to_indices:
        return []

    records: List[Dict[str, Any]] = []

    from eeg_pipeline.utils.analysis.stats.cluster import compute_cluster_correction_2d

    band_names = list(bands.keys())
    for band_name in band_names:
        fmin, fmax_band = bands[band_name]
        fmax_effective = min(float(fmax_band), float(fmax_available))
        if float(fmin) >= float(fmax_effective):
            continue

        # Cache per-window per-trial per-channel power once per band.
        band_power_by_window: List[Optional[np.ndarray]] = []
        for t0, t1 in zip(win_s, win_e):
            band_power_by_window.append(
                extract_trial_band_power(tfr_c, float(fmin), float(fmax_effective), float(t0), float(t1))
            )

        for roi_name, ch_indices in roi_to_indices.items():
            n_windows = int(len(win_s))
            r_vec = np.full((n_windows,), np.nan)
            p_vec = np.full((n_windows,), np.nan)
            n_vec = np.zeros((n_windows,), dtype=int)
            roi_bin_data = np.full((n_windows, 1, len(y_c)), np.nan)
            informative_bins: List[Tuple[int, int]] = []

            for window_idx, band_power in enumerate(band_power_by_window):
                if band_power is None or band_power.size == 0:
                    continue

                if band_power.shape[0] != len(y_c) or band_power.shape[1] != n_ch:
                    continue

                roi_trial_vals = np.nanmean(band_power[:, ch_indices], axis=1)
                roi_bin_data[window_idx, 0, :] = roi_trial_vals

                valid = np.isfinite(roi_trial_vals) & np.isfinite(y_c)
                if cov_vals is not None:
                    valid &= np.all(np.isfinite(cov_vals), axis=1)

                n_obs = int(np.sum(valid))
                if n_obs < req_samples:
                    continue

                if np.std(roi_trial_vals[valid]) < MIN_VARIANCE_THRESHOLD:
                    continue

                try:
                    if cov_vals is not None and cov_vals.shape[1] > 0:
                        r, p, n_out = compute_partial_corr(
                            pd.Series(roi_trial_vals[valid]),
                            pd.Series(y_c[valid]),
                            pd.DataFrame(cov_vals[valid], columns=cov_cols),
                            method=method,
                        )
                        if int(n_out) < req_samples:
                            continue
                        r_vec[window_idx] = float(r)
                        p_vec[window_idx] = float(p)
                        n_vec[window_idx] = int(n_out)
                    else:
                        r, p = corr_fn(roi_trial_vals[valid], y_c[valid])
                        if not (np.isfinite(r) and np.isfinite(p)):
                            continue
                        r_vec[window_idx] = float(r)
                        p_vec[window_idx] = float(p)
                        n_vec[window_idx] = n_obs
                    informative_bins.append((window_idx, 0))
                except (ValueError, RuntimeWarning):
                    continue

            if not informative_bins:
                continue

            # Cluster correction across time windows (time-only adjacency via (n_windows, 1) shape).
            p_cluster_vec = np.full_like(p_vec, np.nan, dtype=float)
            cluster_id_vec = np.zeros_like(n_vec, dtype=int)
            cluster_sig_vec = np.zeros_like(n_vec, dtype=bool)

            if n_cluster_perm > 0:
                n_cov = int(cov_vals.shape[1]) if cov_vals is not None else 0
                t_stat = np.full((n_windows,), np.nan, dtype=float)
                for w_idx, _ in informative_bins:
                    r_val = float(r_vec[w_idx])
                    n_obs = int(n_vec[w_idx])
                    dof = n_obs - n_cov - 2
                    if np.isfinite(r_val) and dof > 0 and abs(r_val) < 1:
                        denom = max(MIN_DENOMINATOR_THRESHOLD, 1.0 - r_val**2)
                        t_stat[w_idx] = r_val * np.sqrt(float(dof) / denom)

                try:
                    labels, p_corr, sig_mask, _records, _perm_max, _thresh = compute_cluster_correction_2d(
                        correlations=t_stat[:, None],
                        p_values=p_vec[:, None],
                        bin_data=roi_bin_data,
                        informative_bins=informative_bins,
                        y_array=y_c,
                        cluster_alpha=c_alpha,
                        n_cluster_perm=n_cluster_perm,
                        alpha=c_alpha,
                        min_valid_points=req_samples,
                        use_spearman=use_spearman,
                        cluster_rng=cluster_rng,
                        covariates_matrix=cov_vals,
                        groups=groups_c,
                        cluster_forming_threshold=cluster_forming_threshold,
                        config=config,
                    )
                    cluster_id_vec = labels[:, 0].astype(int)
                    p_cluster_vec = p_corr[:, 0].astype(float)
                    cluster_sig_vec = sig_mask[:, 0].astype(bool)
                except Exception:
                    # Fall back to uncorrected output if cluster correction fails.
                    pass

            for window_idx in range(n_windows):
                if not (np.isfinite(r_vec[window_idx]) and np.isfinite(p_vec[window_idx])):
                    continue
                records.append(
                    {
                        "condition": condition_name,
                        "band": band_name,
                        "time_start": float(win_s[window_idx]),
                        "time_end": float(win_e[window_idx]),
                        "channel": f"roi_{roi_name}",
                        "r": float(r_vec[window_idx]),
                        "beta_std": float(r_vec[window_idx]),
                        "beta_kind": "standardized",
                        "p": float(p_vec[window_idx]),
                        "p_cluster": float(p_cluster_vec[window_idx]) if np.isfinite(p_cluster_vec[window_idx]) else np.nan,
                        "cluster_id": int(cluster_id_vec[window_idx]),
                        "cluster_significant": bool(cluster_sig_vec[window_idx]),
                        "n": int(n_vec[window_idx]),
                        "method": method,
                        "method_label": method_label,
                        "n_channels": int(len(ch_indices)),
                    }
                )

    return records


def _build_tf_grid_records(
    corrs: np.ndarray,
    pvals: np.ndarray,
    n_valid: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    time_edges: np.ndarray,
    method: str,
    roi_label: str,
    c_pvals: Optional[np.ndarray] = None,
    c_labels: Optional[np.ndarray] = None,
    c_sig: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Build TSV records from time-frequency grid correlations.
    
    Creates records at individual frequency resolution (not bands) for detailed
    time-frequency analysis. Used by TF heatmap computation.
    """
    method_label = format_correlation_method_label(method, None)
    records = []
    
    for freq_idx, freq in enumerate(freqs):
        for time_idx in range(len(time_edges) - 1):
            p_raw = pvals[freq_idx, time_idx]
            if not np.isfinite(p_raw):
                continue
                
            p_cluster = np.nan
            cluster_id = 0
            is_cluster_sig = False
            if c_pvals is not None and np.isfinite(c_pvals[freq_idx, time_idx]):
                p_cluster = float(c_pvals[freq_idx, time_idx])
            if c_labels is not None:
                cluster_id = int(c_labels[freq_idx, time_idx])
            if c_sig is not None:
                is_cluster_sig = bool(c_sig[freq_idx, time_idx])
                
            records.append({
                "condition": "all",
                "band": f"freq_{freq:.1f}Hz",
                "freq": float(freq),
                "time_start": float(time_edges[time_idx]),
                "time_end": float(time_edges[time_idx + 1]),
                "channel": f"roi_{roi_label}",
                "r": float(corrs[freq_idx, time_idx]),
                "beta_std": float(corrs[freq_idx, time_idx]),
                "beta_kind": "standardized",
                "p": float(p_raw),
                "p_cluster": p_cluster,
                "cluster_id": cluster_id,
                "cluster_significant": is_cluster_sig,
                "n": int(n_valid[freq_idx, time_idx]),
                "method": method,
                "method_label": method_label,
                "feature": "tf_grid",
            })
    return records


def _save_temporal_topomap_npz(
    condition_results: Dict[str, Dict[str, Any]],
    ch_names: List[str],
    info: Any,
    out_dir: Path,
    suffix: str,
    logger: logging.Logger,
) -> None:
    """Save temporal correlation results as NPZ for topomap plotting.
    
    Creates a NPZ file with one key per condition, plus metadata for plotting.
    The file is saved as temporal_correlations_by_condition{suffix}.npz.
    """
    if not condition_results:
        logger.info("No condition results to save for topomap NPZ")
        return
    
    ensure_dir(out_dir)
    
    npz_payload = {
        "ch_names": np.array(ch_names, dtype=object),
        "info": info,
        "condition_names": np.array(list(condition_results.keys()), dtype=object),
    }
    
    for cond_name, res in condition_results.items():
        npz_payload[cond_name] = res
    
    npz_path = out_dir / f"temporal_correlations_by_condition{suffix}.npz"
    np.savez_compressed(npz_path, **npz_payload)
    logger.info(f"Saved temporal topomap NPZ: {npz_path.name} ({len(condition_results)} conditions)")


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

    y_arr = _to_numpy_array(y)
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
    
    out_dir = stats_dir
    ensure_dir(out_dir)

    all_tsv_records = []
    ch_names = tfr.ch_names

    include_roi_averages = bool(temporal_cfg.get("include_roi_averages", True))
    roi_definitions = config.get("channel_rois", {}) or {} if include_roi_averages else None
    
    condition_results = {}

    run_col = str(get_config_value(config, "behavior_analysis.run_adjustment.column", "run_id")).strip()
    groups_all = events[run_col].to_numpy()[:n] if run_col and run_col in events.columns else None
    
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
        groups_cond = groups_all[mask] if groups_all is not None else None
        res = _compute_correlations_for_condition(
            tfr, y_arr, mask, safe_name, bands, win_s, win_e, fmax, corr_fn, alpha, logger,
            cov_df, config, n_jobs, groups=groups_cond,
        )
        if res:
            condition_results[safe_name] = res
            
            cond_records = _build_temporal_tsv_records(res, safe_name, ch_names, method)
            for rec in cond_records:
                rec["feature"] = "power"
            all_tsv_records.extend(cond_records)
            logger.info(f"Computed power temporal for condition '{safe_name}': {len(cond_records)} tests")
            
            # ROI-averaged records
            if include_roi_averages:
                roi_records = _compute_roi_correlations_for_condition(
                    tfr,
                    y_arr,
                    mask,
                    safe_name,
                    bands,
                    win_s,
                    win_e,
                    fmax_available=fmax,
                    corr_fn=corr_fn,
                    logger=logger,
                    cov_df=cov_df,
                    config=config,
                    groups=groups_cond,
                    roi_definitions=roi_definitions,
                )
                for rec in roi_records:
                    rec["feature"] = "power_roi"
                all_tsv_records.extend(roi_records)
                if roi_records:
                    logger.info(f"Added {len(roi_records)} ROI-averaged records for condition '{safe_name}'")

    _save_temporal_topomap_npz(
        condition_results, ch_names, epochs.info, out_dir, sfx, logger
    )

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
        "condition_results": condition_results,
    }


def compute_time_frequency_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Run time-frequency correlations using pre-loaded context data."""
    if ctx.computation_features and "temporal" in ctx.computation_features:
        allowed = ctx.computation_features["temporal"]
        if "power" not in allowed and "spectral" not in allowed:
            ctx.logger.info("Skipping time-frequency correlations: feature filter %s excludes 'power'/'spectral'", allowed)
            return None

    if ctx.aligned_events is None:
        ctx.logger.warning("No events available for time-frequency correlations")
        return None
    targets = _get_temporal_targets_from_events(
        ctx.aligned_events,
        config=ctx.config,
        logger=ctx.logger,
        analysis_name="Time-frequency correlations",
    )

    from eeg_pipeline.analysis.behavior.orchestration import get_behavior_output_dir
    out_dir = get_behavior_output_dir(ctx, "temporal_correlations", ensure=True)

    return _run_tf_correlations_core(
        ctx.subject,
        ctx.epochs,
        ctx.aligned_events,
        targets,
        out_dir,
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

    if ctx.aligned_events is None:
        ctx.logger.warning("No events available for temporal correlations")
        return None
    targets = _get_temporal_targets_from_events(
        ctx.aligned_events,
        config=ctx.config,
        logger=ctx.logger,
        analysis_name="Temporal correlations",
    )

    from eeg_pipeline.analysis.behavior.orchestration import get_behavior_output_dir
    out_dir = get_behavior_output_dir(ctx, "temporal_correlations", ensure=True)

    return _run_temporal_by_condition_core(
        ctx.epochs,
        ctx.aligned_events,
        targets,
        out_dir,
        ctx.config,
        ctx.use_spearman,
        ctx.covariates_df,
        ctx.logger,
        selected_bands=ctx.selected_bands,
    )


###################################################################
# ITPC Temporal Correlations
###################################################################


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
    
    _n_trials, _n_ch = data_sub.shape[0], data_sub.shape[1]
    
    phase = np.angle(data_sub)
    phase_vectors = np.exp(1j * phase)
    mean_vector = np.mean(phase_vectors, axis=0)
    mean_phase = np.angle(mean_vector)
    
    phase_diff = phase - mean_phase[np.newaxis, :, :, :]
    alignment = np.cos(phase_diff)
    trial_itpc = alignment.mean(axis=(2, 3))
    
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
    - condition_values: optionally compute only for specific values
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
    
    time_range_ms = temporal_cfg.get("time_range_ms", [-200, 1000])
    time_resolution_ms = temporal_cfg.get("time_resolution_ms", 50)
    tmin_s = time_range_ms[0] / 1000.0
    tmax_s = time_range_ms[1] / 1000.0
    win_size_s = time_resolution_ms / 1000.0
    
    tfr_complex = compute_complex_tfr(epochs, config, logger=logger)
    if tfr_complex is None:
        return None
    
    tfr_data = tfr_complex.data  # (n_trials, n_ch, n_freqs, n_times)
    times = np.asarray(tfr_complex.times)
    freqs = np.asarray(tfr_complex.freqs)
    ch_names = tfr_complex.ch_names
    
    clipped = clip_time_range(times, tmin_s, tmax_s)
    if clipped is None:
        logger.warning("ITPC temporal: no valid time range after clipping")
        return None
    win_s, win_e = build_time_windows_fixed_size_clamped(clipped[0], clipped[1], win_size_s)
    if len(win_s) == 0:
        logger.warning("ITPC temporal: no time windows generated")
        return None
    
    y_arr = _to_numpy_array(y)
    n = compute_aligned_data_length(tfr_complex, events)
    
    condition_values, condition_vec = _determine_condition_values(
        events, n, temporal_cfg, config, logger, analysis_name="ITPC temporal"
    )
    
    fmax = float(np.max(freqs))
    all_bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    
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
    method = "spearman" if use_spearman else "pearson"
    
    out_dir = stats_dir
    ensure_dir(out_dir)
    
    all_tsv_records = []
    
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
    if ctx.aligned_events is None:
        ctx.logger.warning("No events available for ITPC temporal correlations")
        return None
    targets = _get_temporal_targets_from_events(
        ctx.aligned_events,
        config=ctx.config,
        logger=ctx.logger,
        analysis_name="ITPC temporal correlations",
    )

    from eeg_pipeline.analysis.behavior.orchestration import get_behavior_output_dir
    out_dir = get_behavior_output_dir(ctx, "temporal_correlations", ensure=True)

    return _run_itpc_temporal_by_condition_core(
        ctx.epochs,
        ctx.aligned_events,
        targets,
        out_dir,
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
    
    active_data = tfr_power[:, :, freq_mask, :][:, :, :, time_mask]
    baseline_data = tfr_power[:, :, freq_mask, :][:, :, :, bl_mask]
    
    if active_data.size == 0 or baseline_data.size == 0:
        return None
    
    active_mean = active_data.mean(axis=(2, 3))
    baseline_mean = baseline_data.mean(axis=(2, 3))
    
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
    
    y_arr = _to_numpy_array(y)
    n = compute_aligned_data_length(tfr, events)
    
    condition_values, condition_vec = _determine_condition_values(
        events, n, temporal_cfg, config, logger, analysis_name="ERDS temporal"
    )
    
    fmax = float(np.max(freqs))
    all_bands = get_bands_for_tfr(max_freq_available=fmax, config=config)
    if selected_bands:
        bands = {k: v for k, v in all_bands.items() if k.lower() in [b.lower() for b in selected_bands]}
        if not bands:
            logger.warning(f"ERDS temporal: no matching bands in {selected_bands}, using all bands")
            bands = all_bands
        else:
            logger.info(f"ERDS temporal: using selected bands {list(bands.keys())}")
    else:
        bands = all_bands
    
    corr_fn = spearmanr if use_spearman else pearsonr
    corr_method = "spearman" if use_spearman else "pearson"
    
    out_dir = stats_dir
    ensure_dir(out_dir)
    
    all_tsv_records = []

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
    
    logger.info(f"ERDS temporal correlations: {len(all_tsv_records)} tests")
    return {"n_tests": len(all_tsv_records), "n_sig_raw": sum(1 for r in all_tsv_records if r.get("p", 1.0) < 0.05), "feature": "erds", "records": all_tsv_records}


def compute_erds_temporal_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Compute ERDS temporal correlations by condition using pre-loaded data.
    
    Note: Feature selection is handled by the orchestration layer based on
    which feature files the user selected in step 3 (feature selection).
    """
    if ctx.aligned_events is None:
        ctx.logger.warning("No events available for ERDS temporal correlations")
        return None
    targets = _get_temporal_targets_from_events(
        ctx.aligned_events,
        config=ctx.config,
        logger=ctx.logger,
        analysis_name="ERDS temporal correlations",
    )

    from eeg_pipeline.analysis.behavior.orchestration import get_behavior_output_dir
    out_dir = get_behavior_output_dir(ctx, "temporal_correlations", ensure=True)

    return _run_erds_temporal_by_condition_core(
        ctx.epochs, ctx.aligned_events, targets, out_dir, ctx.config,
        ctx.use_spearman, ctx.covariates_df, ctx.logger, selected_bands=ctx.selected_bands,
    )
