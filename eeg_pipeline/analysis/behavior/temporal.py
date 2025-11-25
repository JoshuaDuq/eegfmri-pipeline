import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from eeg_pipeline.utils.data.loading import (
    _load_features_and_targets,
    load_epochs_for_analysis,
    compute_aligned_data_length,
    extract_pain_vector_array,
    extract_temperature_series,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    ensure_dir,
    get_pain_column_from_config,
    get_temperature_column_from_config,
    write_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    compute_correlation,
    compute_partial_corr,
    fdr_bh_values,
    _safe_float,
)
from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_morlet,
    extract_trial_band_power,
    get_bands_for_tfr,
    clip_time_range,
    create_time_windows_fixed_size,
)


def _create_temperature_masks(temp_series, min_trials_per_condition: int, logger: logging.Logger) -> dict:
    if temp_series is None:
        return {}

    temp_array = temp_series.values if isinstance(temp_series, pd.Series) else np.asarray(temp_series)
    if temp_array.size == 0:
        return {}

    unique_temps = np.unique(temp_array[np.isfinite(temp_array)])
    logger.info(f"Found temperature values: {unique_temps.tolist()}")

    temp_masks = {}
    for temp_val in unique_temps:
        temp_mask = (temp_array == temp_val)
        n_trials = temp_mask.sum()

        if n_trials >= min_trials_per_condition:
            temp_str = f"{temp_val:.1f}".replace(".", "_")
            temp_masks[temp_str] = temp_mask
            logger.info(f"Temperature {temp_val:.1f}°C: {n_trials} trials")
        else:
            logger.debug(f"Temperature {temp_val:.1f}°C: {n_trials} trials (insufficient, need >= {min_trials_per_condition})")

    return temp_masks


def _compute_tf_correlations_for_bins(
    power: np.ndarray,
    y_array: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    time_bin_edges: np.ndarray,
    min_valid_points: int,
    use_spearman: bool,
    covariates_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    n_time_bins = len(time_bin_edges) - 1
    correlations = np.full((len(freqs), n_time_bins), np.nan)
    p_values = np.full_like(correlations, np.nan)
    n_valid = np.zeros_like(correlations, dtype=int)
    bin_data = np.full((len(freqs), n_time_bins, len(y_array)), np.nan, dtype=float)
    informative_bins: List[Tuple[int, int]] = []
    cov_matrix: Optional[np.ndarray] = None
    covariate_columns: List[str] = []
    covariate_count: int = 0
    if covariates_df is not None and not covariates_df.empty:
        covariate_columns = list(covariates_df.columns)
        cov_matrix = covariates_df.apply(pd.to_numeric, errors="coerce").to_numpy()
        covariate_count = cov_matrix.shape[1]

    for f_idx, freq in enumerate(freqs):
        for t_idx in range(n_time_bins):
            t_start, t_end = time_bin_edges[t_idx], time_bin_edges[t_idx + 1]
            time_mask = (times >= t_start) & (times < t_end)
            if np.any(time_mask):
                vals = power[:, f_idx, time_mask].mean(axis=1)
                bin_data[f_idx, t_idx, :] = vals
            
            ch_power = bin_data[f_idx, t_idx, :]
            valid_mask = np.isfinite(ch_power) & np.isfinite(y_array)
            n_obs = int(valid_mask.sum())

            correlation: Optional[float] = None
            p_value: Optional[float] = None

            if cov_matrix is not None:
                cov_mask = valid_mask & np.all(np.isfinite(cov_matrix), axis=1)
                required = max(min_valid_points, covariate_count + 2)
                n_obs = int(cov_mask.sum())
                if n_obs >= required:
                    x_series = pd.Series(ch_power[cov_mask])
                    y_series = pd.Series(y_array[cov_mask])
                    cov_df_local = pd.DataFrame(cov_matrix[cov_mask], columns=covariate_columns)
                    try:
                        correlation, p_value, n_partial = compute_partial_corr(
                            x_series, y_series, cov_df_local,
                            method="spearman" if use_spearman else "pearson"
                        )
                    except Exception:
                        correlation, p_value, n_partial = np.nan, np.nan, 0
                    n_obs = n_partial
            else:
                required = min_valid_points
                if n_obs >= required:
                    correlation, p_value = compute_correlation(ch_power[valid_mask], y_array[valid_mask], use_spearman)

            n_valid[f_idx, t_idx] = n_obs
            dof = n_obs - covariate_count - 2
            if (
                correlation is not None
                and p_value is not None
                and np.isfinite(correlation)
                and np.isfinite(p_value)
                and n_obs >= required
                and dof >= 2  # Require DoF >= 2 for meaningful t-distribution approximation
                and abs(correlation) < 1
            ):
                correlations[f_idx, t_idx] = correlation
                p_values[f_idx, t_idx] = p_value
                informative_bins.append((f_idx, t_idx))

    return correlations, p_values, n_valid, bin_data, informative_bins


def _save_temporal_correlations(
    results: dict,
    output_path: Path,
    times: np.ndarray,
    baseline_applied: bool,
    baseline_window_used: Optional[tuple],
    use_spearman: bool,
    ch_names: List[str],
    info,
    logger: logging.Logger,
) -> None:
    save_dict = {
        **results,
        "times": times,
        "baseline_applied": baseline_applied,
        "use_spearman": use_spearman,
        "ch_names": ch_names,
        "info": info,
    }
    if baseline_window_used:
        save_dict["baseline_window"] = baseline_window_used

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Saved temporal correlations to {output_path}")


def _compute_correlations_for_condition(
    tfr,
    y_array: np.ndarray,
    condition_mask: np.ndarray,
    condition_name: str,
    bands_dict: dict,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    fmax_available: float,
    min_trials_per_condition: int,
    correlation_func,
    sig_alpha: float,
    logger: logging.Logger,
    covariates_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    if condition_mask is None or condition_mask.sum() < min_trials_per_condition:
        return None

    tfr_condition = tfr[condition_mask]
    y_condition = y_array[condition_mask]
    method = "spearman" if correlation_func == spearmanr else "pearson"
    cov_values = None
    if covariates_df is not None and not covariates_df.empty:
        cov_columns = list(covariates_df.columns)
        cov_values = (
            covariates_df.loc[condition_mask, cov_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )

    n_channels = len(tfr.ch_names)
    n_bands = len(bands_dict)
    n_windows = len(window_starts)

    correlations = np.full((n_bands, n_windows, n_channels), np.nan)
    p_values = np.full((n_bands, n_windows, n_channels), np.nan)
    n_valid = np.zeros((n_bands, n_windows, n_channels), dtype=int)

    for band_idx, (band_name, (fmin, fmax)) in enumerate(bands_dict.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue

        for win_idx, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            band_power = extract_trial_band_power(tfr_condition, fmin, fmax_eff, tmin_win, tmax_win)
            if band_power is None:
                continue

            for ch_idx in range(n_channels):
                ch_power = band_power[:, ch_idx]
                valid_mask = np.isfinite(ch_power) & np.isfinite(y_condition)
                if cov_values is not None:
                    cov_valid = np.all(np.isfinite(cov_values), axis=1)
                    valid_mask = valid_mask & cov_valid
                    required = max(min_trials_per_condition, cov_values.shape[1] + 2)
                else:
                    required = min_trials_per_condition

                n_valid_trials = valid_mask.sum()
                if n_valid_trials < required:
                    continue

                try:
                    if cov_values is not None and cov_values.shape[1] > 0:
                        cov_df_local = pd.DataFrame(cov_values[valid_mask], columns=cov_columns)
                        corr, pval, n_partial = compute_partial_corr(
                            pd.Series(ch_power[valid_mask]),
                            pd.Series(y_condition[valid_mask]),
                            cov_df_local,
                            method=method,
                        )
                        n_valid_trials = n_partial
                        if n_valid_trials < required or not np.isfinite(corr) or not np.isfinite(pval):
                            continue
                    else:
                        corr, pval = correlation_func(ch_power[valid_mask], y_condition[valid_mask])
                        if not np.isfinite(corr) or not np.isfinite(pval):
                            continue
                    correlations[band_idx, win_idx, ch_idx] = corr
                    p_values[band_idx, win_idx, ch_idx] = pval
                    n_valid[band_idx, win_idx, ch_idx] = n_valid_trials
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    logger.debug(f"Correlation failed: {condition_name}, {band_name}, win{win_idx}, ch{ch_idx}: {e}")

    # No local FDR correction - global FDR is applied across all analyses
    # via fdr_correction.apply_global_fdr() after all stats files are generated

    return {
        "correlations": correlations,
        "p_values": p_values,
        "p_corrected": np.full_like(p_values, np.nan),  # Placeholder; filled by global FDR
        "n_valid": n_valid,
        "band_names": list(bands_dict.keys()),
        "band_ranges": [(fmin, min(fmax, fmax_available)) for fmin, fmax in bands_dict.values()],
        "window_starts": window_starts,
        "window_ends": window_ends,
        "condition_name": condition_name,
    }


def compute_time_frequency_correlations(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    use_spearman: bool,
    logger: logging.Logger,
) -> None:
    from eeg_pipeline.utils.analysis.tfr import restrict_epochs_to_roi, apply_baseline_to_tfr
    
    logger.info("Computing time-frequency correlations...")

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )

    _, _, _, y, _ = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    if epochs is None or aligned_events is None or y is None:
        logger.warning("Cannot compute TF correlations: epochs, events, or target variable not found")
        return

    heatmap_config = config.get("behavior_analysis", {}).get("time_frequency_heatmap", {})
    roi_selection = heatmap_config.get("roi_selection")

    epochs_for_tfr = restrict_epochs_to_roi(epochs, roi_selection, config, logger)

    tfr = compute_tfr_morlet(epochs_for_tfr, config, logger=logger)

    if tfr is None:
        logger.warning("TFR computation failed")
        return

    baseline_applied, baseline_window_used = apply_baseline_to_tfr(tfr, config, logger)

    power = tfr.data.mean(axis=1)
    times = tfr.times
    freqs = tfr.freqs
    y_array = y.to_numpy()
    time_window = heatmap_config.get("time_window")
    if time_window is not None:
        t_start, t_end = float(time_window[0]), float(time_window[1])
        time_mask = (times >= t_start) & (times <= t_end)
        if not np.any(time_mask):
            logger.warning(
                "Time-frequency heatmap window [%s, %s] s has no samples (available [%s, %s]); skipping TF correlations.",
                t_start, t_end, float(times.min()), float(times.max()),
            )
            return
        times = times[time_mask]
        power = power[:, :, time_mask]
    freq_range = heatmap_config.get("freq_range")
    if freq_range is not None:
        f_min, f_max = float(freq_range[0]), float(freq_range[1])
        freq_mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(freq_mask):
            logger.warning(
                "Time-frequency heatmap frequency range [%s, %s] Hz has no bins (available [%s, %s]); skipping TF correlations.",
                f_min, f_max, float(freqs.min()), float(freqs.max()),
            )
            return
        freqs = freqs[freq_mask]
        power = power[:, freq_mask, :]

    time_bin_width = float(heatmap_config.get("time_resolution"))
    time_bin_edges = np.arange(times[0], times[-1] + time_bin_width, time_bin_width)
    stats_config = config.get("behavior_analysis", {}).get("statistics", {})
    min_valid_points = int(stats_config.get("min_samples_roi"))
    partial_covars = stats_config.get("partial_covariates", [])
    cov_df = None
    if partial_covars:
        covars_available = [c for c in partial_covars if c in aligned_events.columns]
        if covars_available:
            cov_df = aligned_events[covars_available].apply(pd.to_numeric, errors="coerce")

    correlations, p_values, n_valid, bin_data, informative_bins = _compute_tf_correlations_for_bins(
        power, y_array, times, freqs, time_bin_edges, min_valid_points, use_spearman, covariates_df=cov_df
    )

    covariate_count = 0
    if cov_df is not None and not cov_df.empty:
        covariate_count = int(cov_df.shape[1])

    cluster_stat = np.full_like(correlations, np.nan, dtype=float)
    for f_idx, t_idx in informative_bins:
        r_val = correlations[f_idx, t_idx]
        n_obs = int(n_valid[f_idx, t_idx])
        if not np.isfinite(r_val):
            continue
        dof = n_obs - covariate_count - 2
        if dof <= 0 or abs(r_val) >= 1:
            continue
        t_stat = r_val * np.sqrt(dof / max(1e-15, 1.0 - r_val**2))
        cluster_stat[f_idx, t_idx] = t_stat

    cluster_cfg = config.get("behavior_analysis", {}).get("cluster_correction", {})
    cluster_perm_cfg = int(cluster_cfg.get("n_permutations", 5000))
    heatmap_override = int(heatmap_config.get("n_cluster_perm", 0))
    n_cluster_perm = cluster_perm_cfg
    if heatmap_override > 0:
        n_cluster_perm = max(heatmap_override, cluster_perm_cfg)
    # Minimum 5000 permutations for reliable p-value estimation at alpha=0.05
    min_cluster_perm = max(5000, cluster_perm_cfg)
    if 0 < n_cluster_perm < min_cluster_perm:
        logger.warning(
            "Time-frequency cluster permutations increased from %d to %d to ensure valid p-values.",
            n_cluster_perm, min_cluster_perm,
        )
        n_cluster_perm = min_cluster_perm
    cluster_alpha = float(cluster_cfg.get("alpha", config.get("statistics.sig_alpha", 0.05)))
    cluster_rng_seed = int(cluster_cfg.get("rng_seed", config.get("random.seed", 42)))
    cluster_rng = np.random.default_rng(cluster_rng_seed)

    cov_matrix = None
    if cov_df is not None and not cov_df.empty:
        cov_matrix = cov_df.apply(pd.to_numeric, errors="coerce").to_numpy()

    cluster_labels = np.zeros_like(correlations, dtype=int)
    cluster_pvals = np.full_like(correlations, np.nan)
    cluster_sig_mask = np.zeros_like(correlations, dtype=bool)
    cluster_records = []
    perm_max_masses: List[float] = []
    cluster_forming_threshold = np.nan

    from eeg_pipeline.utils.analysis.stats import compute_cluster_correction_2d

    if n_cluster_perm > 0:
        (
            cluster_labels,
            cluster_pvals,
            cluster_sig_mask,
            cluster_records,
            perm_max_masses,
            cluster_forming_threshold,
        ) = compute_cluster_correction_2d(
            correlations=cluster_stat,
            p_values=p_values,
            bin_data=bin_data,
            informative_bins=informative_bins,
            y_array=y_array,
            cluster_alpha=cluster_alpha,
            n_cluster_perm=n_cluster_perm,
            alpha=cluster_alpha,
            min_valid_points=min_valid_points,
            use_spearman=use_spearman,
            cluster_rng=cluster_rng,
            covariates_matrix=cov_matrix,
            groups=aligned_events["run_id"].to_numpy() if "run_id" in aligned_events.columns else None,
        )

    if n_cluster_perm > 0:
        p_used = np.where(np.isfinite(cluster_pvals), cluster_pvals, p_values)
    else:
        p_used = p_values

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    roi_suffix = f"_{roi_selection.lower()}" if roi_selection and roi_selection != "null" else ""
    output_path = stats_dir / f"time_frequency_correlation_data{roi_suffix}{use_spearman_suffix}.npz"

    save_dict = {
        "correlations": correlations,
        "p_values_raw": p_values,
        "p_values_cluster": cluster_pvals,
        "p_values_used": p_used,
        "p_values": p_used,
        "n_valid": n_valid,
        "bin_data": bin_data,
        "times": times,
        "freqs": freqs,
        "time_bin_edges": time_bin_edges,
        "informative_bins": np.array(informative_bins),
        "baseline_applied": baseline_applied,
        "roi_selection": roi_selection if roi_selection else "all",
        "use_spearman": use_spearman,
        "cluster_labels": cluster_labels,
        "cluster_sig_mask": cluster_sig_mask,
        "cluster_records": cluster_records,
        "cluster_perm_max_masses": perm_max_masses,
        "cluster_forming_threshold": float(cluster_forming_threshold) if n_cluster_perm > 0 else np.nan,
        "cluster_alpha": cluster_alpha,
        "n_cluster_perm": n_cluster_perm,
    }
    if baseline_window_used:
        save_dict["baseline_window"] = baseline_window_used

    np.savez_compressed(output_path, **save_dict)

    tf_records = []
    n_time_bins = len(time_bin_edges) - 1
    roi_label = roi_selection if roi_selection else "all"

    for f_idx, freq in enumerate(freqs):
        for t_idx in range(n_time_bins):
            p_raw = p_values[f_idx, t_idx]
            p_cluster = cluster_pvals[f_idx, t_idx] if n_cluster_perm > 0 else np.nan
            if not (np.isfinite(p_raw) or np.isfinite(p_cluster)):
                continue

            tf_records.append({
                "roi": roi_label,
                "freq": float(freq),
                "time_start": float(time_bin_edges[t_idx]),
                "time_end": float(time_bin_edges[t_idx+1]),
                "r": float(correlations[f_idx, t_idx]),
                "p": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_cluster": float(p_cluster) if np.isfinite(p_cluster) else np.nan,
                "cluster_id": int(cluster_labels[f_idx, t_idx]) if n_cluster_perm > 0 else 0,
                "cluster_significant": bool(cluster_sig_mask[f_idx, t_idx]) if n_cluster_perm > 0 else False,
                "n": int(n_valid[f_idx, t_idx]),
                "method": "spearman" if use_spearman else "pearson"
            })

    if tf_records:
        df_tf = pd.DataFrame(tf_records)
        tsv_path = stats_dir / f"corr_stats_tf_{roi_label.lower()}{use_spearman_suffix}.tsv"
        write_tsv(df_tf, tsv_path)

    if n_cluster_perm > 0 and cluster_records:
        cluster_rows = []
        cluster_map = {rec.get("cluster_id", idx + 1): rec for idx, rec in enumerate(cluster_records)}
        for cid in sorted(np.unique(cluster_labels[cluster_labels > 0])):
            region = (cluster_labels == cid)
            freq_inds = np.where(region.any(axis=1))[0]
            time_inds = np.where(region.any(axis=0))[0]
            if freq_inds.size == 0 or time_inds.size == 0:
                continue
            rec = cluster_map.get(int(cid), {})
            cluster_rows.append({
                "roi": roi_label,
                "cluster_id": int(cid),
                "p": _safe_float(rec.get("p_value", np.nan)),
                "p_cluster": _safe_float(rec.get("p_value", np.nan)),
                "mass": _safe_float(rec.get("mass", np.nan)),
                "size": int(rec.get("size", region.sum())),
                "freq_min": float(freqs[freq_inds.min()]),
                "freq_max": float(freqs[freq_inds.max()]),
                "time_start": float(time_bin_edges[time_inds.min()]),
                "time_end": float(time_bin_edges[time_inds.max() + 1]),
                "n_freq_bins": int(freq_inds.size),
                "n_time_bins": int(time_inds.size),
                "cluster_forming_threshold": float(cluster_forming_threshold),
                "method": "spearman" if use_spearman else "pearson",
            })
        if cluster_rows:
            df_clusters = pd.DataFrame(cluster_rows)
            write_tsv(
                df_clusters,
                stats_dir / f"corr_stats_tf_clusters_{roi_label.lower()}{use_spearman_suffix}.tsv"
            )

    logger.info(f"Saved time-frequency correlations to {output_path}")
    logger.info(f"  Shape: {correlations.shape}, Informative bins: {len(informative_bins)}")


def compute_temporal_correlations_by_condition(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    use_spearman: bool,
    logger: logging.Logger,
) -> None:
    from eeg_pipeline.utils.analysis.tfr import apply_baseline_to_tfr
    
    logger.info("Computing temporal correlations by condition...")

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )

    _, _, _, y, info = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    if epochs is None or aligned_events is None or y is None:
        logger.warning("Cannot compute temporal correlations: epochs, events, or target variable not found")
        return

    topomap_config = config.get("behavior_analysis", {}).get("temporal_correlation_topomaps", {})
    window_size_ms = float(topomap_config.get("window_size_ms", 100.0))
    min_trials_per_condition = int(topomap_config.get("min_trials_per_condition", 5))
    plateau_window = tuple(config.get("time_frequency_analysis.plateau_window"))

    tfr = compute_tfr_morlet(epochs, config, logger=logger)

    if tfr is None:
        logger.warning("TFR computation failed")
        return

    baseline_applied, baseline_window_used = apply_baseline_to_tfr(tfr, config, logger)

    times = np.asarray(tfr.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        logger.warning(f"No valid time interval within data range; skipping temporal correlations (available [{times.min():.2f}, {times.max():.2f}] s).")
        return
    tmin_clip, tmax_clip = clipped

    window_starts, window_ends = create_time_windows_fixed_size(tmin_clip, tmax_clip, window_size_ms)
    n_windows = len(window_starts)

    if n_windows == 0:
        logger.warning("No valid windows created; skipping temporal correlations.")
        return

    logger.info(f"Computing correlations for {n_windows} time windows from {tmin_clip:.2f} to {tmax_clip:.2f} s")

    y_array = y.to_numpy()
    stats_config = config.get("behavior_analysis", {}).get("statistics", {})
    partial_covars = stats_config.get("partial_covariates", [])
    cov_df = None
    if partial_covars:
        covars_available = [c for c in partial_covars if c in aligned_events.columns]
        if covars_available:
            cov_df = aligned_events[covars_available].apply(pd.to_numeric, errors="coerce")

    n = compute_aligned_data_length(tfr, aligned_events)
    pain_col = get_pain_column_from_config(config, aligned_events)
    temp_col = get_temperature_column_from_config(config, aligned_events)

    pain_vec = extract_pain_vector_array(tfr, aligned_events, pain_col, n) if pain_col else None
    temp_series = extract_temperature_series(tfr, aligned_events, temp_col, n) if temp_col else None

    pain_mask, non_mask = None, None
    if pain_vec is not None:
        pain_mask = np.asarray(pain_vec == 1, dtype=bool)
        non_mask = np.asarray(pain_vec == 0, dtype=bool)

    temp_masks = _create_temperature_masks(temp_series, min_trials_per_condition, logger)

    fmax_available = float(np.max(tfr.freqs))
    bands_dict = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    correlation_func = spearmanr if use_spearman else pearsonr
    sig_alpha = config.get("statistics.sig_alpha")
    if sig_alpha is None:
        raise ValueError("sig_alpha not found in config: statistics.sig_alpha")
    sig_alpha = float(sig_alpha)

    results_by_temp = {}
    for temp_str, temp_mask in temp_masks.items():
        result = _compute_correlations_for_condition(
            tfr, y_array, temp_mask, f"temp_{temp_str}",
            bands_dict, window_starts, window_ends, fmax_available,
            min_trials_per_condition, correlation_func, sig_alpha, logger,
            covariates_df=cov_df,
        )
        if result is not None:
            results_by_temp[temp_str] = result

    results_by_pain = {}
    if pain_mask is not None and non_mask is not None:
        for condition_name, mask in [("pain", pain_mask), ("non_pain", non_mask)]:
            result = _compute_correlations_for_condition(
                tfr, y_array, mask, condition_name,
                bands_dict, window_starts, window_ends, fmax_available,
                min_trials_per_condition, correlation_func, sig_alpha, logger,
                covariates_df=cov_df,
            )
            if result is not None:
                results_by_pain[condition_name] = result

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"

    all_temporal_records = []

    if results_by_temp:
        output_path = stats_dir / f"temporal_correlations_by_temperature{use_spearman_suffix}.npz"
        _save_temporal_correlations(
            {f"temp_{k}": v for k, v in results_by_temp.items()},
            output_path, times, baseline_applied, baseline_window_used,
            use_spearman, tfr.ch_names, tfr.info, logger
        )

        for cond_name, res in results_by_temp.items():
            p_vals = res["p_values"]
            corrs = res["correlations"]
            n_vals = res["n_valid"]
            band_names = res["band_names"]
            starts = res["window_starts"]
            ends = res["window_ends"]

            for b_idx, band in enumerate(band_names):
                for w_idx, (t_start, t_end) in enumerate(zip(starts, ends)):
                    for ch_idx, ch_name in enumerate(tfr.ch_names):
                        p_raw = p_vals[b_idx, w_idx, ch_idx]
                        if np.isfinite(p_raw):
                            all_temporal_records.append({
                                "condition": cond_name,
                                "band": band,
                                "time_start": float(t_start),
                                "time_end": float(t_end),
                                "channel": ch_name,
                                "r": float(corrs[b_idx, w_idx, ch_idx]),
                                "p": float(p_raw),
                                "n": int(n_vals[b_idx, w_idx, ch_idx]),
                                "method": "spearman" if use_spearman else "pearson"
                            })

    if results_by_pain:
        output_path = stats_dir / f"temporal_correlations_by_pain{use_spearman_suffix}.npz"
        _save_temporal_correlations(
            results_by_pain,
            output_path, times, baseline_applied, baseline_window_used,
            use_spearman, tfr.ch_names, tfr.info, logger
        )

        for cond_name, res in results_by_pain.items():
            p_vals = res["p_values"]
            corrs = res["correlations"]
            n_vals = res["n_valid"]
            band_names = res["band_names"]
            starts = res["window_starts"]
            ends = res["window_ends"]

            for b_idx, band in enumerate(band_names):
                for w_idx, (t_start, t_end) in enumerate(zip(starts, ends)):
                    for ch_idx, ch_name in enumerate(tfr.ch_names):
                        p_raw = p_vals[b_idx, w_idx, ch_idx]
                        if np.isfinite(p_raw):
                            all_temporal_records.append({
                                "condition": cond_name,
                                "band": band,
                                "time_start": float(t_start),
                                "time_end": float(t_end),
                                "channel": ch_name,
                                "r": float(corrs[b_idx, w_idx, ch_idx]),
                                "p": float(p_raw),
                                "n": int(n_vals[b_idx, w_idx, ch_idx]),
                                "method": "spearman" if use_spearman else "pearson"
                            })

    if all_temporal_records:
        df_all = pd.DataFrame(all_temporal_records)
        write_tsv(df_all, stats_dir / f"corr_stats_temporal_all{use_spearman_suffix}.tsv")
        logger.info(f"Exported {len(all_temporal_records)} temporal correlation records to consolidated TSV")

