from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    _build_covariate_matrices,
    _load_features_and_targets,
    _pick_first_column,
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io_utils import (
    deriv_features_path,
    deriv_stats_path,
    ensure_aligned_lengths,
    ensure_dir,
    fdr_bh_reject,
    get_subject_logger,
)
from eeg_pipeline.utils.tfr_utils import (
    validate_baseline_window,
    build_rois_from_info as _build_rois,
)
from eeg_pipeline.utils.stats_utils import (
    _safe_float,
    bh_adjust as _bh_adjust,
    partial_corr_xy_given_Z as _partial_corr_xy_given_Z,
    compute_partial_corr as _compute_partial_corr,
    perm_pval_simple as _perm_pval_simple,
    perm_pval_partial_freedman_lane as _perm_pval_partial_FL,
)
from eeg_pipeline.utils.io_utils import sanitize_label

PLOT_SUBDIR = "04_behavior_correlation_analysis"


@dataclass
class GroupScatterInputs:
    rating_x: Dict[Tuple[str, str], List[np.ndarray]]
    rating_y: Dict[Tuple[str, str], List[np.ndarray]]
    rating_Z: Dict[Tuple[str, str], List[pd.DataFrame]]
    rating_hasZ: Dict[Tuple[str, str], List[bool]]
    rating_subjects: Dict[Tuple[str, str], List[str]]
    temp_x: Dict[Tuple[str, str], List[np.ndarray]]
    temp_y: Dict[Tuple[str, str], List[np.ndarray]]
    temp_Z: Dict[Tuple[str, str], List[pd.DataFrame]]
    temp_hasZ: Dict[Tuple[str, str], List[bool]]
    temp_subjects: Dict[Tuple[str, str], List[str]]
    have_temp: bool = False


###################################################################
# Power ROI Statistics
###################################################################

def compute_power_roi_stats(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    config = load_settings()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting ROI power correlation analysis for sub-{subject}")

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    if task is None:
        task = config.task

    rng = rng or np.random.default_rng(42)

    _temporal_df, pow_df, _conn_df, y, info = _load_features_and_targets(subject, task, deriv_root, config)
    y = pd.to_numeric(y, errors="coerce")

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
        logger.error(f"Could not find epochs for ROI correlations: sub-{subject}")
        return

    temp_series = None
    temp_col = None
    psych_temp_columns = config.get("event_columns.temperature", [])
    if aligned_events is not None:
        temp_col = _pick_first_column(aligned_events, psych_temp_columns)
        if temp_col:
            temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")

    roi_map = _build_rois(info, config=config)
    if not roi_map:
        logger.warning(f"No ROI definitions found; skipping ROI stats for sub-{subject}")
        roi_map = {}

    def _build_Z(df_events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_events is None:
            return None
        covars = list(partial_covars) if partial_covars else []
        if not covars:
            tcol = _pick_first_column(df_events, psych_temp_columns)
            if tcol:
                covars.append(tcol)
            for cand in ["trial", "trial_number", "trial_index", "run", "block"]:
                if cand in df_events.columns:
                    covars.append(cand)
                    break
        if not covars:
            return None
        Z = pd.DataFrame()
        for cov in covars:
            if cov in df_events.columns:
                Z[cov] = pd.to_numeric(df_events[cov], errors="coerce")
        return Z if not Z.empty else None

    Z_df_full = _build_Z(aligned_events)
    Z_df_temp = None
    if Z_df_full is not None:
        Z_df_temp = Z_df_full.drop(columns=[temp_col], errors="ignore") if temp_col else Z_df_full.copy()
        if Z_df_temp.shape[1] == 0:
            Z_df_temp = None

    def _get_fdr_alpha(default: float = 0.05) -> float:
        direct = config.get("behavior_analysis.statistics.fdr_alpha")
        if direct is not None:
            return float(direct)
        return (
            config.get("analysis", {})
            .get("behavior_analysis", {})
            .get("statistics", {})
            .get("fdr_alpha", default)
        )

    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    freq_bands = config.get("time_frequency_analysis.bands", {})

    recs_rating: List[Dict[str, Any]] = []
    recs_temp: List[Dict[str, Any]] = []

    for band in power_bands:
        band_cols = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols:
            continue
        band_rng = freq_bands.get(band)
        if band_rng:
            band_rng = tuple(band_rng)
        band_range_str = f"{band_rng[0]:g}–{band_rng[1]:g} Hz" if band_rng else ""
        method = "spearman" if use_spearman else "pearson"

        for roi, channels in roi_map.items():
            roi_cols = [f"pow_{band}_{ch}" for ch in channels if f"pow_{band}_{ch}" in band_cols]
            if not roi_cols:
                continue

            roi_vals = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            ensure_aligned_lengths(roi_vals, y, context=f"ROI stats (band={band}, roi={roi})", strict=True)
            x = roi_vals
            y_r = y
            mask = x.notna() & y_r.notna()
            n_eff = int(mask.sum())

            r = p = np.nan
            if n_eff >= 5:
                if use_spearman:
                    r, p = stats.spearmanr(x[mask], y_r[mask], nan_policy="omit")
                else:
                    r, p = stats.pearsonr(x[mask], y_r[mask])

            r_part = p_part = np.nan
            n_part = 0
            if Z_df_full is not None and not Z_df_full.empty:
                ensure_aligned_lengths(roi_vals, y, Z_df_full, context=f"ROI partial covariates (band={band}, roi={roi})", strict=True)
                r_part, p_part, n_part = _compute_partial_corr(
                    roi_vals,
                    y,
                    Z_df_full,
                    method,
                    logger=logger,
                    context=f"ROI {roi} {band} rating partial",
                )

            r_part_temp = p_part_temp = np.nan
            n_part_temp = 0
            if temp_series is not None and not temp_series.empty:
                ensure_aligned_lengths(roi_vals, y, temp_series, context=f"ROI temp covariate (band={band}, roi={roi})", strict=True)
                df_temp_cov = pd.DataFrame({"temp": temp_series})
                r_part_temp, p_part_temp, n_part_temp = _compute_partial_corr(
                    roi_vals,
                    y,
                    df_temp_cov,
                    method,
                    logger=logger,
                    context=f"ROI {roi} {band} rating|temp",
                )

            ci_low = ci_high = np.nan
            if bootstrap > 0 and n_eff >= 5:
                idx = np.where(mask.to_numpy())[0]
                boots = []
                for _ in range(bootstrap):
                    bidx = rng.choice(idx, size=len(idx), replace=True)
                    if use_spearman:
                        val = stats.spearmanr(x.iloc[bidx], y_r.iloc[bidx], nan_policy="omit")[0]
                    else:
                        val = stats.pearsonr(x.iloc[bidx], y_r.iloc[bidx])[0]
                    if np.isfinite(val):
                        boots.append(val)
                if boots:
                    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

            p_perm = p_partial_perm = p_partial_temp_perm = np.nan
            if n_perm > 0 and n_eff >= 5:
                p_perm = _perm_pval_simple(x, y_r, method, n_perm, rng)
                if Z_df_full is not None and not Z_df_full.empty:
                    ensure_aligned_lengths(x, y_r, Z_df_full, context=f"ROI perm partial (band={band}, roi={roi})", strict=True)
                    p_partial_perm = _perm_pval_partial_FL(x, y_r, Z_df_full, method, n_perm, rng)
                if temp_series is not None and not temp_series.empty:
                    df_temp_cov = pd.DataFrame({"temp": temp_series})
                    ensure_aligned_lengths(x, y_r, df_temp_cov, context=f"ROI perm temp (band={band}, roi={roi})", strict=True)
                    p_partial_temp_perm = _perm_pval_partial_FL(x, y_r, df_temp_cov, method, n_perm, rng)

            recs_rating.append(
                {
                    "roi": roi,
                    "band": band,
                    "band_range": band_range_str,
                    "r": _safe_float(r),
                    "p": _safe_float(p),
                    "n": n_eff,
                    "method": method,
                    "r_ci_low": _safe_float(ci_low),
                    "r_ci_high": _safe_float(ci_high),
                    "r_partial": _safe_float(r_part),
                    "p_partial": _safe_float(p_part),
                    "n_partial": n_part,
                    "partial_covars": ",".join(Z_df_full.columns.tolist()) if Z_df_full is not None and not Z_df_full.empty else "",
                    "r_partial_given_temp": _safe_float(r_part_temp),
                    "p_partial_given_temp": _safe_float(p_part_temp),
                    "n_partial_given_temp": n_part_temp,
                    "p_perm": _safe_float(p_perm),
                    "p_partial_perm": _safe_float(p_partial_perm),
                    "p_partial_given_temp_perm": _safe_float(p_partial_temp_perm),
                    "n_perm": n_perm,
                }
            )

            if temp_series is not None and not temp_series.empty:
                ensure_aligned_lengths(roi_vals, temp_series, context=f"ROI temperature stats (band={band}, roi={roi})", strict=True)
                x_temp = roi_vals
                mask_temp = x_temp.notna() & temp_series.notna()
                n_temp = int(mask_temp.sum())
                r_temp = p_temp = np.nan
                if n_temp >= 5:
                    if use_spearman:
                        r_temp, p_temp = stats.spearmanr(x_temp[mask_temp], temp_series[mask_temp], nan_policy="omit")
                    else:
                        r_temp, p_temp = stats.pearsonr(x_temp[mask_temp], temp_series[mask_temp])

                ci_temp_low = ci_temp_high = np.nan
                if bootstrap > 0 and n_temp >= 5:
                    idx_temp = np.where(mask_temp.to_numpy())[0]
                    boots_temp = []
                    for _ in range(bootstrap):
                        bidx_temp = rng.choice(idx_temp, size=len(idx_temp), replace=True)
                        if use_spearman:
                            val = stats.spearmanr(x_temp.iloc[bidx_temp], temp_series.iloc[bidx_temp], nan_policy="omit")[0]
                        else:
                            val = stats.pearsonr(x_temp.iloc[bidx_temp], temp_series.iloc[bidx_temp])[0]
                        if np.isfinite(val):
                            boots_temp.append(val)
                    if boots_temp:
                        ci_temp_low, ci_temp_high = np.percentile(boots_temp, [2.5, 97.5])

                p_temp_perm = p_temp_partial_perm = np.nan
                if n_perm > 0 and n_temp >= 5:
                    p_temp_perm = _perm_pval_simple(x_temp, temp_series, method, n_perm, rng)
                    if Z_df_temp is not None and not Z_df_temp.empty:
                        ensure_aligned_lengths(x_temp, temp_series, Z_df_temp, context=f"ROI temp perm partial (band={band}, roi={roi})", strict=True)
                        p_temp_partial_perm = _perm_pval_partial_FL(x_temp, temp_series, Z_df_temp, method, n_perm, rng)

                recs_temp.append(
                    {
                        "roi": roi,
                        "band": band,
                        "band_range": band_range_str,
                        "r": _safe_float(r_temp),
                        "p": _safe_float(p_temp),
                        "n": n_temp,
                        "method": method,
                        "r_ci_low": _safe_float(ci_temp_low),
                        "r_ci_high": _safe_float(ci_temp_high),
                        "r_partial": np.nan,
                        "p_partial": np.nan,
                        "n_partial": 0,
                        "partial_covars": ",".join(Z_df_temp.columns.tolist()) if Z_df_temp is not None and not Z_df_temp.empty else "",
                        "p_perm": _safe_float(p_temp_perm),
                        "p_partial_perm": _safe_float(p_temp_partial_perm),
                        "n_perm": n_perm,
                    }
                )

    if recs_rating:
        df_rating = pd.DataFrame(recs_rating)
        alpha = _get_fdr_alpha()
        use_perm = "p_perm" in df_rating.columns and np.isfinite(df_rating["p_perm"].to_numpy()).any()
        pvec = df_rating["p_perm"].to_numpy() if use_perm else df_rating["p"].to_numpy()
        rej, crit = fdr_bh_reject(pvec, alpha=alpha)
        df_rating["fdr_reject"] = rej
        df_rating["fdr_crit_p"] = crit
        df_rating.to_csv(stats_dir / "corr_stats_pow_roi_vs_rating.tsv", sep="\t", index=False)
        df_rating.to_csv(stats_dir / "corr_stats_pow_combined_vs_rating.tsv", sep="\t", index=False)
        logger.info(f"Saved {len(df_rating)} ROI correlations vs rating")

    if recs_temp:
        df_temp = pd.DataFrame(recs_temp)
        alpha_t = _get_fdr_alpha()
        use_perm_t = "p_perm" in df_temp.columns and np.isfinite(df_temp["p_perm"].to_numpy()).any()
        pvec_t = df_temp["p_perm"].to_numpy() if use_perm_t else df_temp["p"].to_numpy()
        rej_t, crit_t = fdr_bh_reject(pvec_t, alpha=alpha_t)
        df_temp["fdr_reject"] = rej_t
        df_temp["fdr_crit_p"] = crit_t
        df_temp.to_csv(stats_dir / "corr_stats_pow_roi_vs_temp.tsv", sep="\t", index=False)
        df_temp.to_csv(stats_dir / "corr_stats_pow_combined_vs_temp.tsv", sep="\t", index=False)
        df_temp.to_csv(stats_dir / "corr_stats_pow_combined_vs_temperature.tsv", sep="\t", index=False)
        logger.info(f"Saved {len(df_temp)} ROI correlations vs temperature")

    for band in power_bands:
        band_cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
        if not band_cols:
            logger.debug(f"No channel-level power columns for band '{band}'")
            continue

        chan_recs_rating: List[Dict[str, Any]] = []
        chan_recs_temp: List[Dict[str, Any]] = []

        for col in band_cols:
            channel = col.replace(f"pow_{band}_", "")
            series = pd.to_numeric(pow_df[col], errors="coerce")

            valid_mask = series.notna() & y.notna()
            if valid_mask.sum() < 10:
                continue

            x_valid = series[valid_mask]
            y_valid = y[valid_mask]
            Z_valid = Z_df_full.loc[valid_mask] if Z_df_full is not None and not Z_df_full.empty else None

            if use_spearman:
                method = "spearman"
                r, p = stats.spearmanr(x_valid, y_valid, nan_policy="omit")
            else:
                method = "pearson"
                r, p = stats.pearsonr(x_valid, y_valid)

            ci_lo = ci_hi = np.nan
            if bootstrap > 0:
                x_arr = x_valid.to_numpy()
                y_arr = y_valid.to_numpy()
                boots = []
                for _ in range(bootstrap):
                    idx = rng.integers(0, len(x_arr), size=len(x_arr))
                    if use_spearman:
                        val = stats.spearmanr(x_arr[idx], y_arr[idx])[0]
                    else:
                        val = stats.pearsonr(x_arr[idx], y_arr[idx])[0]
                    if np.isfinite(val):
                        boots.append(val)
                if boots:
                    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

            r_partial = p_partial = np.nan
            n_partial = 0
            if Z_valid is not None and not Z_valid.empty:
                r_partial, p_partial, n_partial = _partial_corr_xy_given_Z(x_valid, y_valid, Z_valid, use_spearman)

            p_perm = np.nan
            p_partial_perm = np.nan
            if n_perm > 0:
                null_rs = []
                for _ in range(n_perm):
                    shuffled = rng.permutation(y_valid)
                    if use_spearman:
                        null_r = stats.spearmanr(x_valid, shuffled)[0]
                    else:
                        null_r = stats.pearsonr(x_valid, shuffled)[0]
                    if np.isfinite(null_r):
                        null_rs.append(null_r)
                if null_rs:
                    p_perm = (np.sum(np.abs(null_rs) >= np.abs(r)) + 1) / (len(null_rs) + 1)
                if Z_valid is not None and not Z_valid.empty:
                    p_partial_perm = _perm_pval_partial_FL(x_valid, y_valid, Z_valid, method, n_perm, rng)

            chan_recs_rating.append({
                "channel": channel,
                "band": band,
                "r": _safe_float(r),
                "p": _safe_float(p),
                "ci_lo": _safe_float(ci_lo),
                "ci_hi": _safe_float(ci_hi),
                "r_partial": _safe_float(r_partial),
                "p_partial": _safe_float(p_partial),
                "n_partial": n_partial,
                "p_perm": _safe_float(p_perm),
                "p_partial_perm": _safe_float(p_partial_perm),
                "n": int(valid_mask.sum()),
                "method": method,
            })

            if temp_series is not None:
                temp_valid = temp_series[valid_mask]
                temp_valid = pd.to_numeric(temp_valid, errors="coerce")
                mask_temp = temp_valid.notna() & x_valid.notna()
                if mask_temp.sum() >= 10:
                    if use_spearman:
                        r_temp, p_temp = stats.spearmanr(x_valid[mask_temp], temp_valid[mask_temp])
                    else:
                        r_temp, p_temp = stats.pearsonr(x_valid[mask_temp], temp_valid[mask_temp])
                    chan_recs_temp.append({
                        "channel": channel,
                        "band": band,
                        "r": _safe_float(r_temp),
                        "p": _safe_float(p_temp),
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "n": int(mask_temp.sum()),
                        "method": method,
                    })

        if chan_recs_rating:
            df_chan_rating = pd.DataFrame(chan_recs_rating)
            use_perm = "p_perm" in df_chan_rating.columns and np.isfinite(df_chan_rating["p_perm"].to_numpy()).any()
            pvec = df_chan_rating["p_perm"].to_numpy() if use_perm else df_chan_rating["p"].to_numpy()
            alpha_chan = _get_fdr_alpha()
            rej, crit = fdr_bh_reject(pvec, alpha=alpha_chan)
            df_chan_rating["fdr_reject"] = rej
            df_chan_rating["fdr_crit_p"] = crit
            df_chan_rating.to_csv(stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv", sep="\t", index=False)
            logger.info(f"Saved {len(df_chan_rating)} channel correlations vs rating for band {band}")

        if chan_recs_temp:
            df_chan_temp = pd.DataFrame(chan_recs_temp)
            pvec_temp = df_chan_temp["p"].to_numpy()
            alpha_chan_t = _get_fdr_alpha()
            rej_t, crit_t = fdr_bh_reject(pvec_temp, alpha=alpha_chan_t)
            df_chan_temp["fdr_reject"] = rej_t
            df_chan_temp["fdr_crit_p"] = crit_t
            df_chan_temp.to_csv(stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv", sep="\t", index=False)
            logger.info(f"Saved {len(df_chan_temp)} channel correlations vs temperature for band {band}")


###################################################################
# Time-Frequency Correlations
###################################################################

def compute_time_frequency_correlations(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    *,
    use_spearman: bool = True,
) -> Optional[Path]:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Computing time-frequency correlations for sub-{subject}")

    behavior_config = config.get("behavior_analysis", {})
    heatmap_config = behavior_config.get("time_frequency_heatmap", {})
    spline_config = behavior_config.get("spline", {})

    time_resolution = heatmap_config.get("time_resolution", 0.1)
    freq_resolution = heatmap_config.get("freq_resolution", 2.0)
    time_window = tuple(heatmap_config.get("time_window", [-0.5, 2.0]))
    freq_range = tuple(heatmap_config.get("freq_range", [4.0, 40.0]))
    alpha = heatmap_config.get("alpha", config.get("behavior_analysis.statistics.fdr_alpha", 0.05))
    roi_selection = heatmap_config.get("roi_selection")
    if roi_selection == "null":
        roi_selection = None
    n_cycles_factor = heatmap_config.get("n_cycles_factor", 2.0)
    decim = heatmap_config.get("decim", 3)
    min_valid_points = heatmap_config.get("min_valid_points", 5)

    deriv_root = Path(deriv_root)
    from eeg_pipeline.utils.io_utils import deriv_plots_path
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=True,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )
    if epochs is None:
        logger.error("Could not load epochs; skipping TF correlation computation")
        return None
    if aligned_events is None:
        logger.error("Aligned events missing; skipping TF correlation computation")
        return None

    rating_col = _pick_first_column(aligned_events, config.get("event_columns.rating", []))
    if rating_col is None:
        logger.error("No rating column found for TF correlation computation")
        return None

    y = pd.to_numeric(aligned_events[rating_col], errors="coerce")
    if y.isna().all():
        logger.error("All behavioral ratings are NaN; skipping TF correlation computation")
        return None

    if roi_selection is not None:
        roi_map = _build_rois(epochs.info, config=config)
        if roi_selection in roi_map:
            channels = roi_map[roi_selection]
            epochs = epochs.pick_channels(channels)
            logger.info(f"Restricted TF computation to ROI '{roi_selection}' ({len(channels)} channels)")
        else:
            logger.warning(f"ROI '{roi_selection}' not found; using all channels")

    freqs = np.arange(freq_range[0], freq_range[1] + freq_resolution, freq_resolution)
    n_cycles = freqs / n_cycles_factor

    logger.info(f"Computing Morlet TFR: n_freqs={len(freqs)}, n_epochs={len(epochs)}")
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=decim,
        n_jobs=1,
        average=False,
        verbose=False,
    )

    if getattr(getattr(tfr, "data", None), "ndim", None) != 4:
        logger.error("Expected epoch-resolved TFR with shape (epochs, channels, freqs, times)")
        return None

    baseline_applied = False
    baseline_window_used: Optional[Tuple[float, float]] = None
    bl = config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01])
    min_bl_samp = int(config.get("time_frequency_analysis.min_baseline_samples", 5))
    try:
        b_start, b_end, _ = validate_baseline_window(
            tfr.times,
            tuple(bl),
            min_samples=min_bl_samp,
            strict_mode=False,
        )
        tfr.apply_baseline(baseline=(b_start, b_end), mode="logratio")
        baseline_applied = True
        baseline_window_used = (b_start, b_end)
    except (ValueError, RuntimeError) as err:
        logger.warning(f"Baseline validation failed ({err}); continuing without baseline correction")

    tmin_req, tmax_req = _safe_float(time_window[0]), _safe_float(time_window[1])
    tmin_avail, tmax_avail = float(tfr.times[0]), float(tfr.times[-1])
    tmin_clip = max(tmin_req, tmin_avail)
    tmax_clip = min(tmax_req, tmax_avail)
    if tmin_clip > tmax_clip:
        logger.warning(
            f"Requested window [{tmin_req}, {tmax_req}] outside TFR range "
            f"[{tmin_avail}, {tmax_avail}]; aborting TF correlation computation"
        )
        return None

    time_mask = (tfr.times >= tmin_clip) & (tfr.times <= tmax_clip)
    data = tfr.data[:, :, :, time_mask]
    times = tfr.times[time_mask]

    logger.info(f"TFR cropped to {data.shape[0]} epochs, {data.shape[1]} channels, "
                f"{data.shape[2]} freqs, {data.shape[3]} time points")

    power = np.mean(data.real ** 2 + data.imag ** 2, axis=1)
    smoothing_sigma = spline_config.get("gaussian_sigma", 1.0)
    if smoothing_sigma and smoothing_sigma > 0:
        power = gaussian_filter1d(power, sigma=smoothing_sigma, axis=-1)

    time_bin_edges = np.arange(times[0], times[-1] + time_resolution, time_resolution)
    if len(time_bin_edges) < 2:
        logger.error("Insufficient time bins for TF correlation computation")
        return None
    n_time_bins = len(time_bin_edges) - 1
    time_bin_centers = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2

    correlations = np.full((len(freqs), n_time_bins), np.nan)
    p_values = np.full_like(correlations, np.nan)
    n_valid = np.zeros_like(correlations, dtype=int)

    for f_idx, freq in enumerate(freqs):
        for t_idx in range(n_time_bins):
            t_start, t_end = time_bin_edges[t_idx], time_bin_edges[t_idx + 1]
            mask = (times >= t_start) & (times < t_end)
            if not np.any(mask):
                continue
            vals = power[:, f_idx, mask].mean(axis=1)
            valid_mask = np.isfinite(vals) & np.isfinite(y.to_numpy())
            if valid_mask.sum() < min_valid_points:
                continue

            x_valid = vals[valid_mask]
            y_valid = y.to_numpy()[valid_mask]

            if use_spearman:
                r, p = stats.spearmanr(x_valid, y_valid, nan_policy="omit")
            else:
                r, p = stats.pearsonr(x_valid, y_valid)

            correlations[f_idx, t_idx] = _safe_float(r)
            p_values[f_idx, t_idx] = _safe_float(p)
            n_valid[f_idx, t_idx] = int(valid_mask.sum())

    valid_mask = np.isfinite(p_values)
    p_corrected = np.full_like(p_values, np.nan)
    if np.any(valid_mask):
        p_flat = p_values[valid_mask]
        p_corrected_flat = _bh_adjust(p_flat)
        p_corrected[valid_mask] = p_corrected_flat
        significant_mask = p_corrected < alpha
    else:
        significant_mask = np.zeros_like(p_values, dtype=bool)

    time_bin_centers = time_bin_centers.astype(float)
    n_time_bins = len(time_bin_centers)
    stats_file_suffix = ""
    data_file_suffix = ""
    if roi_selection:
        stats_file_suffix += f"_{roi_selection.lower()}"
        data_file_suffix += f"_{roi_selection.lower()}"
    method_suffix = "_spearman" if use_spearman else "_pearson"
    stats_file_suffix += method_suffix
    data_file_suffix += method_suffix

    stats_file = stats_dir / f"time_frequency_correlation_stats{stats_file_suffix}.tsv"
    results_df = pd.DataFrame(
        {
            "frequency": np.repeat(freqs, n_time_bins),
            "time": np.tile(time_bin_centers, len(freqs)),
            "correlation": correlations.flatten(),
            "p_value": p_values.flatten(),
            "p_corrected": p_corrected.flatten(),
            "significant": significant_mask.flatten(),
            "n_valid": n_valid.flatten(),
        }
    ).dropna(subset=["correlation"])
    results_df.to_csv(stats_file, sep="\t", index=False)

    data_path = stats_dir / f"time_frequency_correlation_data{data_file_suffix}.npz"
    np.savez_compressed(
        data_path,
        correlations=correlations,
        p_values=p_values,
        p_corrected=p_corrected,
        significant_mask=significant_mask,
        freqs=freqs,
        time_bin_centers=time_bin_centers,
        time_bin_edges=time_bin_edges,
        n_valid=n_valid,
        baseline_applied=baseline_applied,
        baseline_window=baseline_window_used if baseline_window_used is not None else np.array([]),
        roi_selection=roi_selection if roi_selection is not None else "",
        method="spearman" if use_spearman else "pearson",
        time_window_clipped=(tmin_clip, tmax_clip),
        freq_range=freq_range,
        time_resolution=time_resolution,
        freq_resolution=freq_resolution,
        alpha=alpha,
        min_valid_points=min_valid_points,
    )

    logger.info(f"Saved TF correlation data to {data_path}")
    logger.info(f"Saved TF correlation summary to {stats_file}")
    return data_path


###################################################################
# Power Topomap Correlations
###################################################################

def correlate_power_topomaps(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    config = load_settings()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(
        "correlate_power_topomaps not implemented in refactored pipeline; skipping topomap correlations"
    )


###################################################################
# Connectivity ROI Summary Correlations
###################################################################

def correlate_connectivity_roi_summaries(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None
) -> None:
    config = load_settings()
    if task is None:
        task = config.task

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    
    deriv_root = Path(config.deriv_root)
    from eeg_pipeline.utils.io_utils import deriv_plots_path
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    rng = rng or np.random.default_rng(42)

    feats_dir = deriv_features_path(deriv_root, subject)
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"
    if not conn_path.exists() or not y_path.exists():
        return

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=Path(config.deriv_root), bids_root=config.bids_root, config=config
    )
    if epochs is None:
        return
    info = epochs.info
    roi_map = _build_rois(info, config=config)
    
    temp_series: Optional[pd.Series] = None
    temp_col: Optional[str] = None
    if aligned_events is not None:
        psych_temp_columns = config.get("event_columns.temperature", [])
        tcol = _pick_first_column(aligned_events, psych_temp_columns)
        if tcol is not None:
            temp_col = tcol
            temp_series = pd.to_numeric(aligned_events[tcol], errors="coerce")

    def _build_Z(df_events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_events is None:
            return None
        covars = list(partial_covars) if partial_covars else []
        if not covars:
            psych_temp_columns = config.get("event_columns.temperature", [])
            tcol = _pick_first_column(df_events, psych_temp_columns)
            if tcol is not None:
                covars.append(tcol)
            for c in ["trial", "trial_number", "trial_index", "run", "block"]:
                if c in df_events.columns:
                    covars.append(c)
                    break
        if not covars:
            return None
        Z = pd.DataFrame()
        for c in covars:
            if c in df_events.columns:
                Z[c] = pd.to_numeric(df_events[c], errors="coerce")
        return Z if not Z.empty else None

    Z_df_full = _build_Z(aligned_events)
    Z_df_temp = None
    if Z_df_full is not None:
        Z_df_temp = Z_df_full.drop(columns=[temp_col], errors="ignore") if temp_col else Z_df_full.copy()
        if Z_df_temp.shape[1] == 0:
            Z_df_temp = None

    X = pd.read_csv(conn_path, sep="\t")
    y_df = pd.read_csv(y_path, sep="\t")
    y = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")

    cols = list(X.columns)
    prefixes = sorted({"_".join(c.split("_")[:2]) for c in cols})

    for pref in prefixes:
        cols_pref = [c for c in cols if c.startswith(pref + "_")]
        if not cols_pref:
            continue
        pair_names = [c.split(pref + "_", 1)[-1] for c in cols_pref]
        nodes = sorted({nm for pair in pair_names for nm in pair.split("__")})
        node_to_idx = {nm: i for i, nm in enumerate(nodes)}

        meas_name = pref.split("_", 1)[0].lower()
        apply_fisher_edges = meas_name in ("aec", "aec_orth", "corr", "pearsonr")
        
        def _build_atlas_rois_from_nodes(node_list: List[str], hemisphere_split: bool = True) -> Dict[str, List[str]]:
            import re
            systems = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}
            roi_nodes: Dict[str, List[str]] = {}
            for nm in node_list:
                toks = nm.split("_")
                hemi = None
                system = None
                for t in toks:
                    if t in ("LH", "RH"):
                        hemi = t
                    if t in systems:
                        system = t
                if system is None:
                    m = re.search(r"(?:^|_)(LH|RH)_([A-Za-z]+)", nm)
                    if m:
                        hemi = m.group(1)
                        cand = m.group(2)
                        if cand in systems:
                            system = cand
                if system is None:
                    continue
                roi = system + (f"_{hemi}" if hemisphere_split and hemi else "")
                roi_nodes.setdefault(roi, []).append(nm)
            return roi_nodes

        def _build_summary_map_from_roi_nodes(roi_nodes: Dict[str, List[str]]) -> Dict[Tuple[str, str], List[str]]:
            summary: Dict[Tuple[str, str], List[str]] = {}
            for roi_name, members in roi_nodes.items():
                ch_set = set(members)
                key = (roi_name, roi_name)
                cols_within: List[str] = []
                for col in cols_pref:
                    pair = col.split(pref + "_", 1)[-1]
                    try:
                        a, b = pair.split("__")
                    except ValueError:
                        continue
                    if a in ch_set and b in ch_set and a != b:
                        cols_within.append(col)
                if cols_within:
                    summary[key] = cols_within
            rois_local = sorted(roi_nodes.keys())
            for i in range(len(rois_local)):
                for j in range(i + 1, len(rois_local)):
                    r1, r2 = rois_local[i], rois_local[j]
                    set1, set2 = set(roi_nodes[r1]), set(roi_nodes[r2])
                    cols_between: List[str] = []
                    for col in cols_pref:
                        pair = col.split(pref + "_", 1)[-1]
                        try:
                            a, b = pair.split("__")
                        except ValueError:
                            continue
                        if (a in set1 and b in set2) or (a in set2 and b in set1):
                            cols_between.append(col)
                    if cols_between:
                        summary[(r1, r2)] = cols_between
            return summary

        atlas_roi_map = _build_atlas_rois_from_nodes(nodes, hemisphere_split=True)
        summary_map: Dict[Tuple[str, str], List[str]] = {}
        if atlas_roi_map:
            summary_map = _build_summary_map_from_roi_nodes(atlas_roi_map)

        if not summary_map and roi_map:
            summary_map = _build_summary_map_from_roi_nodes(roi_map)

        if not summary_map:
            continue

        recs: List[Dict[str, object]] = []
        recs_temp: List[Dict[str, object]] = []
        for (roi_i, roi_j), cols_list in summary_map.items():
            edge_df = X[cols_list].apply(pd.to_numeric, errors="coerce")
            if apply_fisher_edges:
                arr = edge_df.to_numpy(dtype=float)
                arr = np.clip(arr, -0.999999, 0.999999)
                z = np.arctanh(arr)
                z_mean = np.nanmean(z, axis=1)
                xi = pd.Series(np.tanh(z_mean), index=edge_df.index)
            else:
                xi = edge_df.mean(axis=1)
            mask = xi.notna() & y.notna()
            n_eff = int(mask.sum())
            if n_eff < 5:
                continue
            r, p = stats.spearmanr(xi[mask], y[mask], nan_policy="omit")
            method = "spearman"

            r_part = np.nan
            p_part = np.nan
            n_part = 0
            if Z_df_full is not None and not Z_df_full.empty:
                n_len_pt = min(len(xi), len(y), len(Z_df_full))
                r_part, p_part, n_part = _partial_corr_xy_given_Z(
                    xi.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_df_full.iloc[:n_len_pt], method
                )

            ci_low = np.nan
            ci_high = np.nan
            if bootstrap and n_eff >= 5:
                idx = np.where(mask.to_numpy())[0]
                boots: List[float] = []
                for _ in range(bootstrap):
                    bidx = rng.choice(idx, size=len(idx), replace=True)
                    xb = xi.iloc[bidx]
                    yb = y.iloc[bidx]
                    rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
                    boots.append(rb)
                if boots:
                    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

            p_perm = np.nan
            p_partial_perm = np.nan
            if n_perm and n_eff >= 5:
                p_perm = _perm_pval_simple(xi, y, method, int(n_perm), rng)
                if Z_df_full is not None and not Z_df_full.empty:
                    n_len_pt = min(len(xi), len(y), len(Z_df_full))
                    p_partial_perm = _perm_pval_partial_FL(
                        xi.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_df_full.iloc[:n_len_pt], method, int(n_perm), rng
                    )

            recs.append({
                "measure_band": pref,
                "roi_i": roi_i,
                "roi_j": roi_j,
                "summary_type": "within" if roi_i == roi_j else "between",
                "n_edges": len(cols_list),
                "r": _safe_float(r),
                "p": _safe_float(p),
                "n": n_eff,
                "method": method,
                "r_ci_low": _safe_float(ci_low),
                "r_ci_high": _safe_float(ci_high),
                "r_partial": _safe_float(r_part),
                "p_partial": _safe_float(p_part),
                "n_partial": n_part,
                "partial_covars": ",".join(Z_df_full.columns.tolist()) if Z_df_full is not None and not Z_df_full.empty else "",
                "p_perm": _safe_float(p_perm),
                "p_partial_perm": _safe_float(p_partial_perm),
                "n_perm": n_perm,
            })

            if temp_series is not None and not temp_series.empty:
                if len(xi) != len(temp_series):
                    logger.warning(f"Channel vs temp length mismatch: power={len(xi)}, temp={len(temp_series)}. Using overlap.")
                n_len_t = min(len(xi), len(temp_series))
                xt, tt = xi.iloc[:n_len_t], temp_series.iloc[:n_len_t]
                m2 = xt.notna() & tt.notna()
                n_eff2 = m2.sum()
                if n_eff2 >= 5:
                    r2, p2 = stats.spearmanr(xt[m2], tt[m2], nan_policy="omit")
                    method2 = "spearman"
                    
                    r2_part = np.nan
                    p2_part = np.nan
                    n2_part = 0
                    if Z_df_temp is not None and not Z_df_temp.empty:
                        n_len_pt2 = min(len(xt), len(tt), len(Z_df_temp))
                        r2_part, p2_part, n2_part = _partial_corr_xy_given_Z(
                            xt.iloc[:n_len_pt2], tt.iloc[:n_len_pt2], Z_df_temp.iloc[:n_len_pt2], method2
                        )
                        
                        ci2_low = np.nan
                        ci2_high = np.nan
                    if bootstrap and n_eff2 >= 5:
                        idx2 = np.where(m2.to_numpy())[0]
                        boots2: List[float] = []
                        for _ in range(bootstrap):
                            bidx2 = rng.choice(idx2, size=len(idx2), replace=True)
                            xb = xt.iloc[bidx2]
                            tb = tt.iloc[bidx2]
                            rb, _ = stats.spearmanr(xb, tb, nan_policy="omit")
                            boots2.append(rb)
                        if boots2:
                            ci2_low, ci2_high = np.percentile(boots2, [2.5, 97.5])
                    
                    p2_perm = np.nan
                    p2_partial_perm = np.nan
                    if n_perm and n_eff2 >= 5:
                        p2_perm = _perm_pval_simple(xt, tt, method2, int(n_perm), rng)
                        if Z_df_temp is not None and not Z_df_temp.empty:
                            n_len_pt2 = min(len(xt), len(tt), len(Z_df_temp))
                            p2_partial_perm = _perm_pval_partial_FL(
                                xt.iloc[:n_len_pt2], tt.iloc[:n_len_pt2], Z_df_temp.iloc[:n_len_pt2], method2, int(n_perm), rng
                            )

                    recs_temp.append({
                        "measure_band": pref,
                        "roi_i": roi_i,
                        "roi_j": roi_j,
                        "summary_type": "within" if roi_i == roi_j else "between",
                        "n_edges": len(cols_list),
                        "r": _safe_float(r2),
                        "p": _safe_float(p2),
                        "n": n_eff2,
                        "method": method2,
                        "r_ci_low": _safe_float(ci2_low),
                        "r_ci_high": _safe_float(ci2_high),
                        "r_partial": _safe_float(r2_part),
                        "p_partial": _safe_float(p2_part),
                        "n_partial": n2_part,
                        "partial_covars": ",".join(Z_df_temp.columns.tolist()) if Z_df_temp is not None and not Z_df_temp.empty else "",
                        "p_perm": _safe_float(p2_perm),
                        "p_partial_perm": _safe_float(p2_partial_perm),
                        "n_perm": n_perm,
                    })

        if recs:
            df = pd.DataFrame(recs)
            pvec = df["p_perm"].to_numpy() if "p_perm" in df.columns and np.isfinite(df["p_perm"]).any() else df["p"].to_numpy()
            fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
            rej, crit = fdr_bh_reject(pvec, alpha=fdr_alpha)
            df["fdr_reject"] = rej
            df["fdr_crit_p"] = crit
            df.to_csv(stats_dir / f"corr_stats_conn_roi_summary_{sanitize_label(pref)}_vs_rating.tsv", sep="\t", index=False)

        if recs_temp:
            df_t = pd.DataFrame(recs_temp)
            pvec_t = df_t["p_perm"].to_numpy() if "p_perm" in df_t.columns and np.isfinite(df_t["p_perm"]).any() else df_t["p"].to_numpy()
            fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
            rej_t, crit_t = fdr_bh_reject(pvec_t, alpha=fdr_alpha)
            df_t["fdr_reject"] = rej_t
            df_t["fdr_crit_p"] = crit_t
            df_t.to_csv(stats_dir / f"corr_stats_conn_roi_summary_{sanitize_label(pref)}_vs_temp.tsv", sep="\t", index=False)


###################################################################
# Connectivity Heatmap Correlations
###################################################################

def correlate_connectivity_heatmaps(subject: str, task: Optional[str] = None, use_spearman: bool = True) -> None:
    config = load_settings()
    if task is None:
        task = config.task
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Starting connectivity correlation analysis for sub-{subject}")
    deriv_root = Path(config.deriv_root)
    from eeg_pipeline.utils.io_utils import deriv_plots_path
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    feats_dir = deriv_features_path(deriv_root, subject)
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"
    if not conn_path.exists() or not y_path.exists():
        logger.warning(f"Connectivity features or targets missing for sub-{subject}; skipping connectivity correlations.")
        return

    X = pd.read_csv(conn_path, sep="\t")
    y_df = pd.read_csv(y_path, sep="\t")
    y = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")

    cols = list(X.columns)
    prefixes = sorted({"_".join(c.split("_")[:2]) for c in cols})

    for pref in prefixes:
        cols_pref = [c for c in cols if c.startswith(pref + "_")]
        pair_names = [c.split(pref + "_", 1)[-1] for c in cols_pref]
        nodes = sorted({nm for pair in pair_names for nm in pair.split("__")})
        n_nodes = len(nodes)
        if n_nodes < 3:
            logger.warning(f"Could not infer nodes for {pref}; skipping heatmap.")
            continue
        node_idx = {nm: i for i, nm in enumerate(nodes)}

        rvals = np.full((n_nodes, n_nodes), np.nan, float)
        pvals = np.full((n_nodes, n_nodes), np.nan, float)
        for col in cols_pref:
            pair = col.split(pref + "_", 1)[-1]
            try:
                a, b = pair.split("__")
            except ValueError:
                continue
            i, j = node_idx[a], node_idx[b]
            xi = pd.to_numeric(X[col], errors="coerce")
            mask = xi.notna() & y.notna()
            if mask.sum() < 5:
                continue
            r, p = stats.spearmanr(xi[mask], y[mask], nan_policy="omit")
            rvals[i, j] = r
            rvals[j, i] = r
            pvals[i, j] = p
            pvals[j, i] = p
        
        iu = np.triu_indices(n_nodes, k=1)
        p_flat = pvals[iu]
        valid_idx = np.isfinite(p_flat)
        p_valid = p_flat[valid_idx]
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        from eeg_pipeline.utils.io_utils import fdr_bh_reject
        rej_valid, crit_p = fdr_bh_reject(p_valid, alpha=fdr_alpha)
        valid_pairs = [(iu[0][k], iu[1][k]) for k in np.where(valid_idx)[0]]
        reject_map = {pair: bool(rej_valid[k]) for k, pair in enumerate(valid_pairs)}
        crit_val = _safe_float(np.max(p_valid[rej_valid])) if np.any(rej_valid) else np.nan

        recs: List[Dict[str, object]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                pair_key = (i, j)
                r = rvals[i, j]
                p = pvals[i, j]
                is_sig = reject_map.get(pair_key, False)
                recs.append({
                    "node_i": nodes[i],
                    "node_j": nodes[j],
                    "r": _safe_float(r),
                    "p": _safe_float(p),
                    "fdr_reject": is_sig,
                    "fdr_crit_p": crit_val,
                })
        
        if recs:
            df = pd.DataFrame(recs)
            df.to_csv(stats_dir / f"corr_stats_edges_{sanitize_label(pref)}_vs_rating.tsv", sep="\t", index=False)
            logger.info(f"Saved connectivity heatmap correlations for {pref}")


###################################################################
# Export Functions
###################################################################

def export_all_significant_predictors(subject: str, alpha: float = 0.05, use_fdr: bool = True) -> None:
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Exporting all significant predictors for sub-{subject} (alpha={alpha})")
    
    all_predictors = []
    
    for target in ("rating", "temp", "temperature"):
        roi_file = stats_dir / f"corr_stats_pow_roi_vs_{target}.tsv"
        if not roi_file.exists():
            continue
        roi_df = pd.read_csv(roi_file, sep="\t")
        if use_fdr and "fdr_reject" in roi_df.columns:
            significant_roi = roi_df[roi_df["fdr_reject"] == True].copy()
        elif use_fdr and "fdr_crit_p" in roi_df.columns and "p" in roi_df.columns:
            significant_roi = roi_df[roi_df["p"] <= roi_df["fdr_crit_p"]].copy()
        else:
            significant_roi = roi_df[roi_df["p"] <= alpha].copy()
        if len(significant_roi) > 0:
            significant_roi["predictor_type"] = "ROI"
            significant_roi["target"] = target
            significant_roi["predictor"] = significant_roi["roi"] + " (" + significant_roi["band"] + ")"
            roi_cols = {
                "predictor": "predictor",
                "roi": "region",
                "band": "band",
                "r": "r",
                "p": "p",
                "n": "n",
                "predictor_type": "type",
                "target": "target",
            }
            if "fdr_reject" in significant_roi.columns:
                roi_cols["fdr_reject"] = "fdr_significant"
            if "fdr_crit_p" in significant_roi.columns:
                roi_cols["fdr_crit_p"] = "fdr_critical_p"
            roi_subset = significant_roi[list(roi_cols.keys())].rename(columns=roi_cols)
            all_predictors.append(roi_subset)
            logger.info(f"Found {len(significant_roi)} significant ROI predictors for target '{target}'")
    
    for target in ("rating", "temp", "temperature"):
        combined_file = stats_dir / f"corr_stats_pow_combined_vs_{target}.tsv"
        if not combined_file.exists():
            continue
        chan_df = pd.read_csv(combined_file, sep="\t")
        
        if "channel" not in chan_df.columns or "band" not in chan_df.columns:
            logger.debug(f"Skipping combined file for target '{target}' - missing required columns (expected 'channel' and 'band')")
            continue
            
        if use_fdr and "fdr_reject" in chan_df.columns:
            significant_chan = chan_df[chan_df["fdr_reject"] == True].copy()
        elif use_fdr and "fdr_crit_p" in chan_df.columns and "p" in chan_df.columns:
            significant_chan = chan_df[chan_df["p"] <= chan_df["fdr_crit_p"]].copy()
        else:
            significant_chan = chan_df[chan_df["p"] <= alpha].copy()
        if len(significant_chan) > 0:
            significant_chan["predictor_type"] = "Channel"
            significant_chan["target"] = target
            significant_chan["predictor"] = significant_chan["channel"] + " (" + significant_chan["band"] + ")"
            chan_cols = {
                "predictor": "predictor",
                "channel": "region",
                "band": "band",
                "r": "r",
                "p": "p",
                "n": "n",
                "predictor_type": "type",
                "target": "target",
            }
            if "fdr_reject" in significant_chan.columns:
                chan_cols["fdr_reject"] = "fdr_significant"
            if "fdr_crit_p" in significant_chan.columns:
                chan_cols["fdr_crit_p"] = "fdr_critical_p"
            chan_subset = significant_chan[list(chan_cols.keys())].rename(columns=chan_cols)
            all_predictors.append(chan_subset)
            logger.info(f"Found {len(significant_chan)} significant channel predictors for target '{target}'")
    
    if all_predictors:
        combined_df = pd.concat(all_predictors, ignore_index=True)
        combined_df["abs_r"] = combined_df["r"].abs()
        combined_df = combined_df.sort_values("p", ascending=True)
        
        output_file = stats_dir / "all_significant_predictors.csv"
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(combined_df)} total significant predictors to: {output_file}")
        
        n_roi = len(combined_df[combined_df["type"] == "ROI"])
        n_chan = len(combined_df[combined_df["type"] == "Channel"])
        max_r = combined_df["abs_r"].max()
        strongest = combined_df.iloc[0]
        
        logger.info(f"Summary: {n_roi} ROI + {n_chan} channel predictors")
        logger.info(f"Strongest predictor: {strongest['predictor']} (r={strongest['r']:.3f})")
        
    else:
        logger.warning("No significant predictors found")
        empty_df = pd.DataFrame(columns=["predictor", "region", "band", "r", "p", "n", "type", "target", "abs_r"])
        output_file = stats_dir / "all_significant_predictors.csv"
        empty_df.to_csv(output_file, index=False)


def export_combined_power_corr_stats(subject: str) -> None:
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    for target in ("rating", "temp"):
        frames: List[pd.DataFrame] = []
        for band in bands:
            f = stats_dir / f"corr_stats_pow_{band}_vs_{target}.tsv"
            if not f.exists():
                continue
            df = pd.read_csv(f, sep="\t")
            if df is None or df.empty:
                continue
            if "band" not in df.columns:
                df["band"] = band
            else:
                df["band"] = df["band"].fillna(band)
            frames.append(df)

        if frames:
            cat = pd.concat(frames, ignore_index=True)
            out_base = stats_dir / f"corr_stats_pow_combined_vs_{target}"
            cat.to_csv(out_base.with_suffix(".tsv"), sep="\t", index=False)
            cat.to_csv(out_base.with_suffix(".csv"), index=False)


###################################################################
# Global FDR
###################################################################

def apply_global_fdr(subject: str, alpha: float = 0.05) -> None:
    config = load_settings()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    patterns = [
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_roi_vs_temp.tsv", 
        "corr_stats_conn_roi_summary_*_vs_rating.tsv",
        "corr_stats_conn_roi_summary_*_vs_temp.tsv",
        "corr_stats_edges_*_vs_rating.tsv",
        "corr_stats_edges_*_vs_temp.tsv",
    ]
    
    files = [f for pat in patterns for f in sorted(stats_dir.glob(pat))]
    if not files:
        logger.info(f"No stats TSVs found for global FDR in {stats_dir}")
        return

    def _parse_analysis_type(name: str) -> str:
        if name.startswith("corr_stats_pow_roi"): return "pow_roi"
        if name.startswith("corr_stats_conn_roi_summary"): return "conn_roi_summary" 
        if name.startswith("corr_stats_edges"): return "conn_edges"
        return "other"

    def _parse_target(name: str) -> str:
        return name.split("_vs_", 1)[1].split(".", 1)[0] if "_vs_" in name else ""

    def _parse_measure_band(analysis_type: str, name: str) -> str:
        if analysis_type == "conn_edges" and name.startswith("corr_stats_edges_"):
            return name[len("corr_stats_edges_"):].split("_vs_", 1)[0]
        if analysis_type == "conn_roi_summary" and name.startswith("corr_stats_conn_roi_summary_"):
            return name[len("corr_stats_conn_roi_summary_"):].split("_vs_", 1)[0]
        return ""

    all_p = []
    refs = []
    metas = []

    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df is None or df.empty:
            continue

        name = f.name
        analysis_type = _parse_analysis_type(name)
        target = _parse_target(name)
        measure_band = _parse_measure_band(analysis_type, name)

        p_perm_ser = pd.to_numeric(df["p_perm"], errors="coerce") if "p_perm" in df.columns else pd.Series(np.nan, index=df.index)
        p_raw_ser = pd.to_numeric(df["p"], errors="coerce") if "p" in df.columns else pd.Series(np.nan, index=df.index)
        pser = p_perm_ser.where(np.isfinite(p_perm_ser), p_raw_ser)
        mask = np.isfinite(pser.to_numpy())
        if not np.any(mask):
            continue

        for idx, used in enumerate(mask):
            if not used:
                continue
            pval = _safe_float(pser.iloc[idx])
            all_p.append(pval)
            refs.append((f, idx))

            meta = {
                "source_file": f.name,
                "analysis_type": analysis_type,
                "target": target,
                "measure_band": measure_band,
                "row_index": int(idx),
            }
            try:
                src = "p_perm" if np.isfinite(p_perm_ser.iloc[idx]) else ("p" if np.isfinite(p_raw_ser.iloc[idx]) else "")
            except (IndexError, KeyError):
                src = ""
            if src:
                meta["p_used_source"] = src
            try:
                if analysis_type == "pow_roi":
                    roi = df.get("roi", pd.Series([""] * len(df))).iloc[idx]
                    band = df.get("band", pd.Series([""] * len(df))).iloc[idx]
                    meta.update({"roi": roi, "band": band})
                    meta["test_label"] = f"pow_{band}_ROI {roi} vs {target}"
                elif analysis_type == "conn_roi_summary":
                    roi_i = df.get("roi_i", pd.Series([""] * len(df))).iloc[idx]
                    roi_j = df.get("roi_j", pd.Series([""] * len(df))).iloc[idx]
                    summ = df.get("summary_type", pd.Series([""] * len(df))).iloc[idx]
                    meas = df.get("measure_band", pd.Series([measure_band] * len(df))).iloc[idx]
                    meta.update({"roi_i": roi_i, "roi_j": roi_j, "summary_type": summ, "measure_band": meas})
                    meta["test_label"] = f"{meas} {roi_i}-{roi_j} ({summ}) vs {target}"
                elif analysis_type == "conn_edges":
                    ni = df.get("node_i", pd.Series([""] * len(df))).iloc[idx]
                    nj = df.get("node_j", pd.Series([""] * len(df))).iloc[idx]
                    meta.update({"node_i": ni, "node_j": nj})
                    meta["test_label"] = f"{measure_band} edge {ni}-{nj} vs {target}"
                else:
                    meta["test_label"] = f"{name}[{idx}]"
            except (IndexError, KeyError, ValueError):
                meta["test_label"] = f"{name}[{idx}]"
            metas.append(meta)

    if not all_p:
        logger.info("No valid p-values found for global FDR; skipping.")
        return

    p_arr = np.asarray(all_p, dtype=float)
    q_arr = _bh_adjust(p_arr)
    rej_arr, crit_p = fdr_bh_reject(p_arr, alpha=_safe_float(alpha))

    updates = {}
    for k, (f, row_idx) in enumerate(refs):
        updates.setdefault(f, []).append((row_idx, _safe_float(q_arr[k]), bool(rej_arr[k]), _safe_float(p_arr[k])))

    for f, items in updates.items():
        try:
            df = pd.read_csv(f, sep="\t")
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df is None or df.empty:
            continue
        
        nrows = len(df)
        qcol = np.full(nrows, np.nan, dtype=float)
        rcol = np.zeros(nrows, dtype=bool)
        pused = np.full(nrows, np.nan, dtype=float)
        
        for (row_idx, qv, rj, pu) in items:
            if 0 <= int(row_idx) < nrows:
                qcol[int(row_idx)] = qv
                rcol[int(row_idx)] = rj
                pused[int(row_idx)] = pu
        
        df["p_used_for_global_fdr"] = pused
        df["q_fdr_global"] = qcol
        df["fdr_reject_global"] = rcol
        df["fdr_crit_p_global"] = _safe_float(crit_p)
        
        try:
            df.to_csv(f, sep="\t", index=False)
        except (OSError, PermissionError):
            pass

    summary_rows = []
    for k, meta in enumerate(metas):
        row = dict(meta)
        row["p_used_for_global_fdr"] = _safe_float(p_arr[k])
        row["q_fdr_global"] = _safe_float(q_arr[k])
        row["fdr_reject_global"] = bool(rej_arr[k])
        row["fdr_crit_p_global"] = _safe_float(crit_p)
        summary_rows.append(row)

    try:
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(stats_dir / "global_fdr_summary.tsv", sep="\t", index=False)
    except (OSError, PermissionError):
        pass


###################################################################
# Group Input Collection
###################################################################

def collect_group_power_roi_inputs(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    *,
    partial_covars: Optional[List[str]] = None,
    do_temp: bool = True,
) -> GroupScatterInputs:
    rating_x = defaultdict(list)
    rating_y = defaultdict(list)
    rating_Z = defaultdict(list)
    rating_hasZ = defaultdict(list)
    rating_subjects = defaultdict(list)

    temp_x = defaultdict(list)
    temp_y = defaultdict(list)
    temp_Z = defaultdict(list)
    temp_hasZ = defaultdict(list)
    temp_subjects = defaultdict(list)

    have_temp = False

    for sub in subjects:
        try:
            _temporal_df, pow_df, _conn_df, y, info = _load_features_and_targets(sub, task, deriv_root, config)
        except FileNotFoundError:
            continue

        y = pd.to_numeric(y, errors="coerce")

        epochs, aligned_events = load_epochs_for_analysis(
            sub,
            task,
            align="strict",
            preload=False,
            deriv_root=deriv_root,
            bids_root=config.bids_root,
            config=config,
        )
        if epochs is None:
            continue

        temp_series = None
        temp_col = None
        psych_temp_columns = config.get("event_columns.temperature", [])
        if aligned_events is not None:
            temp_col = _pick_first_column(aligned_events, psych_temp_columns)
            if temp_col:
                temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
                if do_temp:
                    have_temp = True

        Z_df_full, Z_df_temp = _build_covariate_matrices(aligned_events, partial_covars, temp_col, config=config)

        roi_map = _build_rois(info, config=config)
        power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])

        for band in power_bands_to_use:
            band_cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
            if not band_cols:
                continue
            band_vals = pow_df[band_cols].apply(pd.to_numeric, errors="coerce")
            overall_vals = band_vals.mean(axis=1).to_numpy()
            key_overall = (band, "All")

            rating_x[key_overall].append(overall_vals)
            rating_y[key_overall].append(y.to_numpy())
            rating_subjects[key_overall].append(sub)
            has_full = Z_df_full is not None and not Z_df_full.empty
            rating_Z[key_overall].append(Z_df_full if has_full else None)
            rating_hasZ[key_overall].append(has_full)

            if do_temp and temp_series is not None:
                temp_vals = temp_series.to_numpy()
                temp_x[key_overall].append(overall_vals)
                temp_y[key_overall].append(temp_vals)
                temp_subjects[key_overall].append(sub)
                has_temp = Z_df_temp is not None and not Z_df_temp.empty
                temp_Z[key_overall].append(Z_df_temp if has_temp else None)
                temp_hasZ[key_overall].append(has_temp)

            for roi, chs in roi_map.items():
                cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in pow_df.columns]
                if not cols:
                    continue
                roi_vals = pow_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
                key_roi = (band, roi)

                rating_x[key_roi].append(roi_vals)
                rating_y[key_roi].append(y.to_numpy())
                rating_subjects[key_roi].append(sub)
                has_full = Z_df_full is not None and not Z_df_full.empty
                rating_Z[key_roi].append(Z_df_full if has_full else None)
                rating_hasZ[key_roi].append(has_full)

                if do_temp and temp_series is not None:
                    temp_vals = temp_series.to_numpy()
                    temp_x[key_roi].append(roi_vals)
                    temp_y[key_roi].append(temp_vals)
                    temp_subjects[key_roi].append(sub)
                    has_temp = Z_df_temp is not None and not Z_df_temp.empty
                    temp_Z[key_roi].append(Z_df_temp if has_temp else None)
                    temp_hasZ[key_roi].append(has_temp)

    return GroupScatterInputs(
        rating_x=dict(rating_x),
        rating_y=dict(rating_y),
        rating_Z=dict(rating_Z),
        rating_hasZ=dict(rating_hasZ),
        rating_subjects=dict(rating_subjects),
        temp_x=dict(temp_x),
        temp_y=dict(temp_y),
        temp_Z=dict(temp_Z),
        temp_hasZ=dict(temp_hasZ),
        temp_subjects=dict(temp_subjects),
        have_temp=have_temp,
    )

