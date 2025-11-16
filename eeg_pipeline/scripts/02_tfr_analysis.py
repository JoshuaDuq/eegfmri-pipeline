from __future__ import annotations

# Standard library
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import logging

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Third-party
import numpy as np
import pandas as pd
import mne

# Local - config and data
from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    get_available_subjects,
    load_epochs_for_analysis,
    parse_subject_args,
)
from eeg_pipeline.utils.io_utils import (
    _find_clean_epochs_path,
    deriv_plots_path,
    ensure_derivatives_dataset_description,
    get_group_logger,
    get_subject_logger,
    setup_matplotlib,
    find_temperature_column_in_events,
    find_pain_column_in_events,
    sanitize_label,
    format_temperature_label,
    write_group_trial_counts,
)
from eeg_pipeline.utils.tfr_utils import (
    compute_adaptive_n_cycles,
    compute_subject_tfr,
    canonicalize_ch_name as _canonicalize_ch_name,
    find_roi_channels as _find_roi_channels,
    build_rois_from_info as _build_rois,
    avg_alltrials_to_avg_tfr,
    avg_by_mask_to_avg_tfr,
    align_avg_tfrs,
    collect_group_temperatures,
    get_rois,
    get_bands_for_tfr,
    run_tfr_morlet,
    validate_baseline_indices,
    apply_baseline_and_crop,
    average_tfr_band,
    extract_epochwise_channel_values,
)
from eeg_pipeline.utils.stats_utils import (
    validate_pain_binary_values,
    validate_temperature_values,
    validate_baseline_window_pre_stimulus,
)

# Local - plotting
from eeg_pipeline.plotting.plot_tfr import (
    plot_scalpmean_all_trials,
    contrast_scalpmean_pain_nonpain,
    plot_channels_all_trials,
    contrast_channels_pain_nonpain,
    qc_baseline_plateau_power,
    contrast_maxmin_temperature,
    plot_rois_all_trials,
    group_contrast_maxmin_temperature,
    group_rois_all_trials,
    group_contrast_pain_nonpain_rois,
    group_contrast_pain_nonpain_scalpmean,
    contrast_pain_nonpain_rois,
    group_tf_correlation,
    group_plot_bands_pain_temp_contrasts,
    group_plot_pain_nonpain_temporal_topomaps_diff_allbands,
    group_plot_temporal_topomaps_allbands_plateau,
    group_plot_topomap_grid_baseline_temps,
    plot_bands_pain_temp_contrasts,
    plot_pain_nonpain_temporal_topomaps_diff_allbands,
    plot_temporal_topomaps_allbands_plateau,
    plot_topomap_grid_baseline_temps,
)


###################################################################
# Core TFR Utilities
###################################################################

def _resolve_tfr_workers(config):
    raw = os.getenv("EEG_TFR_WORKERS")
    default_workers = config.get("tfr_topography_pipeline.tfr.workers", -1)
    if not raw or raw.strip().lower() in {"auto", ""}:
        return default_workers
    
    try:
        return max(1, int(raw))
    except ValueError:
        logger = logging.getLogger(__name__)
        logger.warning(f"EEG_TFR_WORKERS={raw} invalid; using {default_workers}")
        return default_workers


###################################################################
# Configuration Extraction
###################################################################

def _extract_tfr_config(config) -> Dict[str, Any]:
    return {
        "freq_min": config.get("tfr_topography_pipeline.tfr.freq_min", 1.0),
        "freq_max": config.get("tfr_topography_pipeline.tfr.freq_max", 100.0),
        "n_freqs": config.get("tfr_topography_pipeline.tfr.n_freqs", 40),
        "n_cycles_factor": config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0),
        "tfr_decim": config.get("tfr_topography_pipeline.tfr.decim", 4),
        "tfr_picks": config.get("tfr_topography_pipeline.tfr.picks", "eeg"),
        "workers": _resolve_tfr_workers(config),
        "baseline_window": tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0])),
        "plateau_window": tuple(config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])),
        "temperature_strategy": config.get("tfr_topography_pipeline.temperature_strategy", "pooled"),
        "min_baseline_samples": int(config.get("tfr_topography_pipeline.min_baseline_samples", 5)),
    }


def _clip_time_window(times: np.ndarray, window: Tuple[float, float]) -> Tuple[float, float]:
    tmin_req, tmax_req = window
    tmin_eff = float(max(float(np.min(times)), float(tmin_req)))
    tmax_eff = float(min(float(np.max(times)), float(tmax_req)))
    return tmin_eff, tmax_eff


def _nanmean_eeg(vec, info):
    picks = mne.pick_types(info, eeg=True, exclude=[])
    return float(np.nanmean(vec[picks] if len(picks) > 0 else vec))


def _is_tail_trim_consistent(events_df: pd.DataFrame, epochs: "mne.Epochs", n_keep: int, logger=None) -> bool:
    if logger is None:
        logger = logging.getLogger(__name__)
    if n_keep <= 0:
        return False
    sample_col = "sample" if "sample" in events_df.columns else None
    epoch_events = getattr(epochs, "events", None)
    if sample_col is None or not isinstance(epoch_events, np.ndarray):
        logger.error(
            "Cannot verify trial alignment during trim — missing 'sample' column or epoch events."
        )
        return False

    event_samples = pd.to_numeric(events_df[sample_col], errors="coerce").to_numpy()
    epoch_samples = np.asarray(epoch_events[:, 0], dtype=float)
    if len(event_samples) < n_keep or len(epoch_samples) < n_keep:
        return False

    head_events = event_samples[:n_keep]
    head_epochs = epoch_samples[:n_keep]
    if not np.array_equal(head_epochs, head_events):
        logger.error(
            f"Head mismatch detected: events and epochs do not align in the first {n_keep} samples. "
            f"Event samples (head): {head_events[:min(5, len(head_events))]}, "
            f"Epoch samples (head): {head_epochs[:min(5, len(head_epochs))]}. "
            f"This indicates a misalignment between events TSV and epochs."
        )
        raise ValueError(
            f"Head alignment check failed: events and epochs samples do not match. "
            f"This indicates a critical misalignment between events TSV and epochs data."
        )

    if len(event_samples) > n_keep:
        tail_events = event_samples[n_keep:]
        if not np.all(tail_events > head_events[-1]):
            return False
    if len(epoch_samples) > n_keep:
        tail_epochs = epoch_samples[n_keep:]
        if not np.all(tail_epochs > head_epochs[-1]):
            return False

    return True


def _log_baseline_qc(tfr_obj, baseline, config=None, logger=None, output_dir=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    times = np.asarray(tfr_obj.times)
    min_samples = config.get("tfr_topography_pipeline.min_baseline_samples", 5) if config else 5
    b_start, b_end, idx = validate_baseline_indices(times, baseline, min_samples=min_samples, logger=logger)
    
    base = tfr_obj.data[:, :, :, idx]
    temporal_std = np.nanstd(base, axis=-1)
    med_temporal_std = float(np.nanmedian(temporal_std))
    epoch_means = np.nanmean(base, axis=(1, 2, 3))
    med = float(np.nanmedian(epoch_means))
    mad = float(np.nanmedian(np.abs(epoch_means - med)))
    rcv = float(1.4826 * mad / (abs(med) if abs(med) > 1e-12 else 1e-12))
    n_time = int(len(idx))
    n_epochs = int(base.shape[0])
    n_channels = int(base.shape[1])
    n_freqs = int(base.shape[2])
    
    max_rcv_threshold = float(config.get("tfr_topography_pipeline.max_baseline_rcv", 0.5)) if config else 0.5
    min_temporal_std_threshold = float(config.get("tfr_topography_pipeline.min_baseline_temporal_std", 0.01)) if config else 0.01
    
    rcv_passed = rcv <= max_rcv_threshold
    temporal_std_passed = med_temporal_std >= min_temporal_std_threshold
    
    msg = (
        f"Baseline QC: n_time={n_time}, median_temporal_std={med_temporal_std:.3g}, "
        f"epoch_MAD={mad:.3g}, RCV={rcv:.3g}"
    )
    logger.info(msg)
    
    if not np.isfinite(med_temporal_std) or not np.isfinite(rcv):
        error_msg = (
            f"Baseline QC FAILED: non-finite metrics detected (med_temporal_std={med_temporal_std}, "
            f"RCV={rcv}). Baseline is unstable and all TFR results are invalid."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if rcv > max_rcv_threshold:
        warning_msg = (
            f"Baseline QC WARNING: RCV={rcv:.3g} exceeds maximum threshold ({max_rcv_threshold}). "
            f"Baseline is too unstable (high variability across epochs). "
            f"This may invalidate TFR baseline correction. "
            f"Check data quality and preprocessing steps."
        )
        logger.warning(warning_msg)
    
    if med_temporal_std < min_temporal_std_threshold:
        warning_msg = (
            f"Baseline QC WARNING: median_temporal_std={med_temporal_std:.3g} below minimum threshold ({min_temporal_std_threshold}). "
            f"Baseline shows insufficient temporal variability, suggesting potential data issues. "
            f"This may indicate flatlined or corrupted baseline data."
        )
        logger.warning(warning_msg)
    
    if rcv_passed and temporal_std_passed:
        logger.info(
            f"Baseline QC PASSED: RCV={rcv:.3g} <= {max_rcv_threshold}, "
            f"temporal_std={med_temporal_std:.3g} >= {min_temporal_std_threshold}"
        )
    
    if output_dir is not None:
        qc_dir = Path(output_dir) / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        
        qc_stats = {
            "baseline_start": float(b_start),
            "baseline_end": float(b_end),
            "n_time_samples": n_time,
            "n_epochs": n_epochs,
            "n_channels": n_channels,
            "n_frequencies": n_freqs,
            "median_temporal_std": med_temporal_std,
            "epoch_median": med,
            "epoch_mad": mad,
            "rcv": rcv,
            "max_rcv_threshold": max_rcv_threshold,
            "min_temporal_std_threshold": min_temporal_std_threshold,
            "rcv_passed": bool(rcv_passed),
            "temporal_std_passed": bool(temporal_std_passed),
            "overall_passed": bool(rcv_passed and temporal_std_passed),
        }
        
        qc_df = pd.DataFrame([qc_stats])
        qc_path = qc_dir / "baseline_qc_stats.tsv"
        qc_df.to_csv(qc_path, sep="\t", index=False, float_format="%.6e")
        logger.info(f"Saved baseline QC stats: {qc_path}")




def _discover_subjects_with_tf(roi_suffix, method_suffix, config, allowed_subjects=None):
    subs = []
    for sd in sorted(config.deriv_root.glob("sub-*")):
        if not sd.is_dir():
            continue
        sub = sd.name[4:]
        if allowed_subjects is not None and sub not in allowed_subjects:
            continue
        cand = sd / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
        if cand.exists():
            subs.append(sub)
    return subs


def _load_subject_tf(sub, roi_suffix, method_suffix, config):
    p = config.deriv_root / f"sub-{sub}" / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None




def _build_group_roi_map_from_channels(ch_names, config=None):
    from eeg_pipeline.utils.tfr_utils import build_rois_from_info
    info = mne.create_info(ch_names, sfreq=100.0, ch_types=['eeg'] * len(ch_names))
    roi_map = build_rois_from_info(info, config=config)
    return roi_map




def _pick_central_channel(info, preferred="Cz", logger=None, group_mode=False):
    ch_names = info['ch_names']
    if preferred in ch_names:
        return preferred
    for nm in ch_names:
        if nm.lower() == preferred.lower():
            return nm
    picks = mne.pick_types(info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    fallback = ch_names[picks[0]]
    if group_mode:
        raise ValueError(
            f"Channel '{preferred}' not found in group-level analysis. "
            f"Available channels: {ch_names[:10]}{'...' if len(ch_names) > 10 else ''}. "
            f"Falling back to different channels across subjects would create inconsistent topographies. "
            f"Ensure all subjects have '{preferred}' channel."
        )
    if logger:
        logger.warning(f"Channel '{preferred}' not found; using '{fallback}' instead.")
    return fallback


def _format_baseline_in_filename(baseline_used: Tuple[float, float]) -> str:
    b_start, b_end = baseline_used
    return f"bl{abs(b_start):.1f}to{abs(b_end):.2f}"


def _get_figure_footer_text(baseline_used: Tuple[float, float], config) -> str:
    b_start, b_end = baseline_used
    plot_styling = config.get("tfr_topography_pipeline.plot_styling", {})
    footer_config = plot_styling.get("footer", {})
    template = footer_config.get("template", "Units: log10(power/baseline) | Baseline: [{baseline_start:.2f}, {baseline_end:.2f}] s")
    return template.format(baseline_start=b_start, baseline_end=b_end)




def _compute_power_and_events(
    subject: str,
    task: str,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional["mne.time_frequency.EpochsTFR"], Optional[pd.DataFrame]]:
    freq_min = config.get("tfr_topography_pipeline.tfr.freq_min", 1.0)
    freq_max = config.get("tfr_topography_pipeline.tfr.freq_max", 100.0)
    n_freqs = config.get("tfr_topography_pipeline.tfr.n_freqs", 40)
    n_cycles_factor = config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0)
    tfr_decim = config.get("tfr_topography_pipeline.tfr.decim", 4)
    tfr_picks = config.get("tfr_topography_pipeline.tfr.picks", "eeg")
    workers = _resolve_tfr_workers(config)
    return compute_subject_tfr(
        subject=subject,
        task=task,
        freq_min=freq_min,
        freq_max=freq_max,
        n_freqs=n_freqs,
        n_cycles_factor=n_cycles_factor,
        tfr_decim=tfr_decim,
        tfr_picks=tfr_picks,
        workers=workers,
        logger=logger,
    )




def _epochs_mean_roi(epochs: mne.Epochs, roi_name: str, roi_chs: list[str]) -> Optional[mne.Epochs]:
    if len(roi_chs) == 0:
        return None
    picks = mne.pick_channels(epochs.ch_names, include=roi_chs, ordered=True)
    if len(picks) == 0:
        return None
    data = epochs.get_data()
    roi_data = data[:, picks, :].mean(axis=1, keepdims=True)
    info = mne.create_info([roi_name], sfreq=epochs.info['sfreq'], ch_types='eeg')
    epo_roi = mne.EpochsArray(
        roi_data,
        info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin,
        metadata=epochs.metadata,
        verbose=False,
    )
    return epo_roi


def compute_roi_tfrs(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    config,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, mne.time_frequency.EpochsTFR]:
    if roi_map is None:
        roi_map = _build_rois(epochs.info, config=config)
    roi_tfrs = {}
    for roi, chs in roi_map.items():
        epo_roi = _epochs_mean_roi(epochs, roi, chs)
        if epo_roi is None:
            continue
        decim = config.get("tfr_topography_pipeline.tfr.decim", 4)
        picks = config.get("tfr_topography_pipeline.tfr.picks", "eeg")
        workers = _resolve_tfr_workers(config)
        power = run_tfr_morlet(
            epo_roi,
            freqs,
            n_cycles,
            decim=decim,
            picks=picks,
            workers=workers,
            config=config,
        )
        roi_tfrs[roi] = power
    return roi_tfrs


def _avg_tfr_to_roi_average(
    tfr_avg: "mne.time_frequency.AverageTFR",
    roi: str,
    roi_map_override: Optional[Dict[str, list[str]]] = None,
    config=None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    if roi_map_override is not None:
        roi_map = roi_map_override
        chs_override = roi_map.get(roi)
        if chs_override:
            chs_all = list(chs_override)
        else:
            if config is None:
                raise ValueError("config is required when roi_map_override does not contain roi")
            roi_defs = get_rois(config)
            pats = roi_defs.get(roi, [])
            chs_all = _find_roi_channels(tfr_avg.info, pats)
    else:
        if config is None:
            raise ValueError("config is required for _avg_tfr_to_roi_average")
        roi_defs = get_rois(config)
        pats = roi_defs.get(roi, [])
        chs_all = _find_roi_channels(tfr_avg.info, pats)
    
    subj_chs = tfr_avg.info['ch_names']
    canon_subj = {_canonicalize_ch_name(ch).upper(): ch for ch in subj_chs}
    want = {_canonicalize_ch_name(ch).upper() for ch in chs_all}
    chs = [canon_subj[_canonicalize_ch_name(ch).upper()] for ch in subj_chs if _canonicalize_ch_name(ch).upper() in want]
    
    if len(chs) == 0:
        subj_roi_map = _build_group_roi_map_from_channels(list(subj_chs), config=config)
        chs_subj_heur = list(subj_roi_map.get(roi, []))
        if chs_subj_heur:
            chs = [ch for ch in subj_chs if ch in set(chs_subj_heur)]
    
    if len(chs) == 0:
        if config is None:
            raise ValueError("config is required for _avg_tfr_to_roi_average")
        roi_defs = get_rois(config)
        pats = roi_defs.get(roi, [])
        chs_rx = _find_roi_channels(tfr_avg.info, pats)
        if chs_rx:
            chs = [ch for ch in subj_chs if ch in set(chs_rx)]
    
    if len(chs) == 0:
        return None
    
    picks = mne.pick_channels(subj_chs, include=chs, exclude=[])
    if len(picks) == 0:
        return None
    
    data = np.asarray(tfr_avg.data)[picks, :, :].mean(axis=0, keepdims=True)
    roi_tfr = tfr_avg.copy()
    roi_tfr.data = data
    roi_tfr.info = mne.create_info([f"ROI:{roi}"], sfreq=tfr_avg.info['sfreq'], ch_types='eeg')
    roi_tfr.nave = int(getattr(tfr_avg, 'nave', 1))
    roi_tfr.comment = f"ROI:{roi}"
    return roi_tfr


###################################################################
# Group Summary Functions
###################################################################



def _write_group_band_summary(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    rows: list[dict] = []
    
    def add_row(cond: str, label: str, band: str, mu: float, n_subj: int):
        rows.append({
            "condition": cond,
            "label": label,
            "band": band,
            "mu_log10": float(mu),
            "pct_change": float((10.0 ** mu - 1.0) * 100.0),
            "n_subjects": int(n_subj),
        })

    avg_list = []
    for p in powers:
        t = p.copy()
        baseline_used = apply_baseline_and_crop(t, baseline=baseline, mode="logratio", logger=logger)
        avg_list.append(t.average())
    info_all, data_all = align_avg_tfrs(avg_list, logger=logger)
    if info_all is not None and data_all is not None:
        freqs = np.asarray(avg_list[0].freqs)
        times = np.asarray(avg_list[0].times)
        fmax_available = float(freqs.max())
        bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
        tmin_req, tmax_req = plateau_window
        tmin = float(max(times.min(), tmin_req))
        tmax = float(min(times.max(), tmax_req))
        mean_all = data_all.mean(axis=0)
        eeg_picks = mne.pick_types(info_all, eeg=True, exclude=[])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            vec = mean_all[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
            add_row("All", "All trials", band, mu, data_all.shape[0])

    pain_cols = config.get("event_columns.pain_binary", [])
    avg_pain, avg_non = [], []
    for p, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = next((c for c in pain_cols if c in ev.columns), None)
        if pain_col is None:
            continue
        vals, n_coerced = validate_pain_binary_values(
            ev[pain_col], pain_col, logger=logger
        )
        mask_p, mask_n = vals == 1, vals == 0
        if mask_p.sum() == 0 or mask_n.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(p, mask_p, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(p, mask_n, baseline=baseline, logger=logger)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    
    info_p, data_p = align_avg_tfrs(avg_pain, logger=logger)
    info_n, data_n = align_avg_tfrs(avg_non, logger=logger)
    if info_p is not None and info_n is not None and data_p is not None and data_n is not None:
        freqs = np.asarray(avg_pain[0].freqs)
        times = np.asarray(avg_pain[0].times)
        fmax_available = float(freqs.max())
        bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
        tmin_req, tmax_req = plateau_window
        tmin = float(max(times.min(), tmin_req))
        tmax = float(min(times.max(), tmax_req))
        mean_p = data_p.mean(axis=0)
        mean_n = data_n.mean(axis=0)
        eeg_picks_p = mne.pick_types(info_p, eeg=True, exclude=[])
        eeg_picks_n = mne.pick_types(info_n, eeg=True, exclude=[])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            mu_p = float(np.nanmean(v_p[eeg_picks_p])) if len(eeg_picks_p) > 0 else float(np.nanmean(v_p))
            mu_n = float(np.nanmean(v_n[eeg_picks_n])) if len(eeg_picks_n) > 0 else float(np.nanmean(v_n))
            mu_d = float(np.nanmean(v_p - v_n))
            add_row("Pain", "pain", band, mu_p, data_p.shape[0])
            add_row("Non-pain", "non-pain", band, mu_n, data_n.shape[0])
            add_row("Diff", "pain-minus-non", band, mu_d, min(data_p.shape[0], data_n.shape[0]))

    temp_cols = config.get("event_columns.temperature", [])
    temps = collect_group_temperatures(events_by_subj, temp_cols)
    for tval in temps:
        avg_list = []
        ns = 0
        for p, ev in zip(powers, events_by_subj):
            if ev is None:
                continue
            temp_cols = config.get("event_columns.temperature", [])
            tcol = next((c for c in temp_cols if c in ev.columns), None)
            if tcol is None:
                continue
            temp_values, n_dropped = validate_temperature_values(
                ev[tcol], tcol, min_temp=35.0, max_temp=55.0, logger=logger
            )
            mask = np.abs(temp_values - float(tval)) < 0.05
            mask = np.asarray(mask, dtype=bool) & np.isfinite(temp_values)
            if mask.sum() == 0:
                continue
            a = avg_by_mask_to_avg_tfr(p, mask, baseline=baseline, logger=logger)
            if a is not None:
                avg_list.append(a)
                ns += 1
        info_t, data_t = align_avg_tfrs(avg_list, logger=logger)
        if info_t is None or data_t is None:
            continue
        freqs = np.asarray(avg_list[0].freqs)
        times = np.asarray(avg_list[0].times)
        fmax_available = float(freqs.max())
        bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
        tmin_req, tmax_req = plateau_window
        tmin = float(max(times.min(), tmin_req))
        tmax = float(min(times.max(), tmax_req))
        mean_t = data_t.mean(axis=0)
        eeg_picks = mne.pick_types(info_t, eeg=True, exclude=[])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            vec = mean_t[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
            add_row("Temperature", f"{float(tval):.1f}°C", band, mu, data_t.shape[0])

    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "group_band_summary.tsv"
        pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
        if logger:
            logger.info(f"Saved band summary: {out_path}")


def _write_group_roi_summary(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    roi_map: Optional[Dict[str, list[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    rows: list[dict] = []
    
    def add_row(roi: str, cond: str, label: str, band: str, mu: float, n_subj: int):
        rows.append({
            "roi": roi,
            "condition": cond,
            "label": label,
            "band": band,
            "mu_log10": float(mu),
            "pct_change": float((10.0 ** mu - 1.0) * 100.0),
            "n_subjects": int(n_subj),
        })

    if not powers:
        return

    avg_all = []
    for p in powers:
        avg_all.append(avg_alltrials_to_avg_tfr(p, baseline=(-5.0, -0.01), logger=logger))
    if not avg_all:
        return

    freqs_ref = np.asarray(avg_all[0].freqs)
    times_ref = np.asarray(avg_all[0].times)
    fmax_available = float(freqs_ref.max())
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_ref.min(), tmin_req))
    tmax = float(min(times_ref.max(), tmax_req))

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for _write_group_roi_summary")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())

    def group_mean_roi(avgs: List["mne.time_frequency.AverageTFR"], roi_name: str) -> Tuple[Optional[np.ndarray], int]:
        rois_list: List["mne.time_frequency.AverageTFR"] = []
        for a in avgs:
            r = _avg_tfr_to_roi_average(a, roi_name, roi_map_override=roi_map, config=config)
            if r is not None:
                rois_list.append(r)
        if not rois_list:
            return None, 0
        info_c, data_c = align_avg_tfrs(rois_list, logger=logger)
        if info_c is None or data_c is None:
            return None, 0
        return data_c.mean(axis=0), data_c.shape[0]

    for roi in rois:
        mean_roi, nsub = group_mean_roi(avg_all, roi)
        if mean_roi is None or nsub == 0:
            continue
        arr = np.asarray(mean_roi[0])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs_ref >= fmin) & (freqs_ref <= fmax_eff)
            tmask = (times_ref >= tmin) & (times_ref < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            mu = float(np.nanmean(arr[fmask, :][:, tmask]))
            add_row(roi, "All", "All trials", band, mu, nsub)

    pain_cols = config.get("event_columns.pain_binary", [])
    avg_pain = []
    avg_non = []
    for p, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = next((c for c in pain_cols if c in ev.columns), None)
        if pain_col is None:
            continue
        vals, n_coerced = validate_pain_binary_values(
            ev[pain_col], pain_col, logger=logger
        )
        mask_p = np.asarray(vals == 1, dtype=bool)
        mask_n = np.asarray(vals == 0, dtype=bool)
        if mask_p.sum() == 0 or mask_n.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(p, mask_p, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(p, mask_n, baseline=baseline, logger=logger)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    
    if avg_pain and avg_non:
        freqs_p = np.asarray(avg_pain[0].freqs)
        times_p = np.asarray(avg_pain[0].times)
        fmax_available_p = float(freqs_p.max())
        bands_p = get_bands_for_tfr(max_freq_available=fmax_available_p, config=config)
        tmin_p = float(max(times_p.min(), tmin_req))
        tmax_p = float(min(times_p.max(), tmax_req))
        for roi in rois:
            mean_p, n_p = group_mean_roi(avg_pain, roi)
            mean_n, n_n = group_mean_roi(avg_non, roi)
            if mean_p is None or mean_n is None or n_p == 0 or n_n == 0:
                continue
            arr_p = np.asarray(mean_p[0])
            arr_n = np.asarray(mean_n[0])
            for band, (fmin, fmax) in bands_p.items():
                fmax_eff = min(fmax, fmax_available_p)
                fmask = (freqs_p >= fmin) & (freqs_p <= fmax_eff)
                tmask = (times_p >= tmin_p) & (times_p < tmax_p)
                if fmask.sum() == 0 or tmask.sum() == 0:
                    continue
                mu_p = float(np.nanmean(arr_p[fmask, :][:, tmask]))
                mu_n = float(np.nanmean(arr_n[fmask, :][:, tmask]))
                mu_d = float(np.nanmean((arr_p - arr_n)[fmask, :][:, tmask]))
                add_row(roi, "Pain", "pain", band, mu_p, n_p)
                add_row(roi, "Non-pain", "non-pain", band, mu_n, n_n)
                add_row(roi, "Diff", "pain-minus-non", band, mu_d, min(n_p, n_n))

    temp_cols = config.get("event_columns.temperature", [])
    temps = collect_group_temperatures(events_by_subj, temp_cols)
    if temps:
        for roi in rois:
            for tval in temps:
                avgs_t = []
                for p, ev in zip(powers, events_by_subj):
                    if ev is None:
                        continue
                    temp_cols = config.get("event_columns.temperature", [])
                    tcol = next((c for c in temp_cols if c in ev.columns), None)
                    if tcol is None:
                        continue
                    temp_values, n_dropped = validate_temperature_values(
                        ev[tcol], tcol, min_temp=35.0, max_temp=55.0, logger=logger
                    )
                    mask = np.abs(temp_values - float(tval)) < 0.05
                    mask = np.asarray(mask, dtype=bool) & np.isfinite(temp_values)
                    if mask.sum() == 0:
                        continue
                    a = avg_by_mask_to_avg_tfr(p, mask, baseline=baseline, logger=logger)
                    if a is not None:
                        avgs_t.append(a)
                if not avgs_t:
                    continue
                freqs_t = np.asarray(avgs_t[0].freqs)
                times_t = np.asarray(avgs_t[0].times)
                fmax_available_t = float(freqs_t.max())
                bands_t = get_bands_for_tfr(max_freq_available=fmax_available_t, config=config)
                tmin_t = float(max(times_t.min(), tmin_req))
                tmax_t = float(min(times_t.max(), tmax_req))
                mean_t, n_t = group_mean_roi(avgs_t, roi)
                if mean_t is None or n_t == 0:
                    continue
                arr_t = np.asarray(mean_t[0])
                for band, (fmin, fmax) in bands_t.items():
                    fmax_eff = min(fmax, fmax_available_t)
                    fmask = (freqs_t >= fmin) & (freqs_t <= fmax_eff)
                    tmask = (times_t >= tmin_t) & (times_t < tmax_t)
                    if fmask.sum() == 0 or tmask.sum() == 0:
                        continue
                    mu = float(np.nanmean(arr_t[fmask, :][:, tmask]))
                    add_row(roi, "Temperature", f"{float(tval):.1f}°C", band, mu, n_t)

    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "group_roi_summary.tsv"
        pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
        if logger:
            logger.info(f"Saved ROI summary: {out_path}")


###################################################################
# Single-Subject Processing
###################################################################

def _compute_subject_tfr(epochs, config, logger):
    freq_min = config.get("tfr_topography_pipeline.tfr.freq_min", 1.0)
    freq_max = config.get("tfr_topography_pipeline.tfr.freq_max", 100.0)
    n_freqs = config.get("tfr_topography_pipeline.tfr.n_freqs", 40)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles_factor = config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    tfr_decim = config.get("tfr_topography_pipeline.tfr.decim", 4)
    tfr_picks = config.get("tfr_topography_pipeline.tfr.picks", "eeg")
    workers = _resolve_tfr_workers(config)
    
    if logger:
        logger.info("Computing per-trial TFR (Morlet)...")
    power = run_tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=tfr_decim,
        picks=tfr_picks,
        workers=workers,
        logger=logger,
        config=config
    )
    return power


def _run_overall_analysis(power, events_df, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger):
    plot_scalpmean_all_trials(
        power, plots_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), subject=subject, task=task, logger=logger
    )
    contrast_scalpmean_pain_nonpain(
        power, events_df, plots_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), logger=logger, subject=subject
    )


def _run_channel_specific_analysis(power, events_df, plots_dir, config, baseline_window, subject, task, logger, channels=None):
    if logger:
        if channels:
            logger.info(f"Generating channel-level TFR plots for specified channels: {channels}")
        else:
            logger.info("Generating comprehensive channel-level TFR plots for all channels...")
    plot_channels_all_trials(
        power, plots_dir, config=config, baseline=baseline_window,
        logger=logger, subject=subject, task=task, channels=channels
    )
    contrast_channels_pain_nonpain(
        power, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger, channels=channels
    )


def _run_channel_analysis(power, events_df, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger, run_overall=True, run_channel_specific=True, channels=None):
    if run_channel_specific:
        _run_channel_specific_analysis(power, events_df, plots_dir, config, baseline_window, subject, task, logger, channels=channels)


def _run_roi_analysis(power, events_df, epochs, plots_dir, config, baseline_window, logger):
    if logger:
        logger.info("Building ROIs and computing ROI TFRs (pooled)...")
    roi_map = _build_rois(epochs.info, config=config)
    if len(roi_map) == 0:
        if logger:
            logger.warning("No ROI channels found in montage; skipping ROI analysis.")
        return
    
    freq_min = config.get("tfr_topography_pipeline.tfr.freq_min", 1.0)
    freq_max = config.get("tfr_topography_pipeline.tfr.freq_max", 100.0)
    n_freqs = config.get("tfr_topography_pipeline.tfr.n_freqs", 40)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles_factor = config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    roi_tfrs = compute_roi_tfrs(epochs, freqs=freqs, n_cycles=n_cycles, config=config, roi_map=roi_map)
    plot_rois_all_trials(roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger)
    contrast_pain_nonpain_rois(roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger)


def _run_pooled_analysis(power, events_df, epochs, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger, run_overall=True, run_channels=True, run_rois=True, channels=None):
    if run_overall:
        _run_overall_analysis(power, events_df, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger)
    
    if run_channels:
        _run_channel_analysis(power, events_df, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger, run_overall=False, run_channel_specific=run_channels, channels=channels)
    
    if run_rois:
        _run_roi_analysis(power, events_df, epochs, plots_dir, config, baseline_window, logger)


def _run_per_temperature_analysis(epochs, events_df, temp_col, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger, run_overall=True, run_channels=True, run_rois=True):
    temp_values_valid, n_dropped = validate_temperature_values(
        events_df[temp_col], temp_col, min_temp=35.0, max_temp=55.0, logger=logger
    )
    temps = sorted(map(float, np.unique(np.round(temp_values_valid[np.isfinite(temp_values_valid)], 1))))
    if len(temps) == 0:
        if logger:
            logger.warning("No valid temperatures found in events; skipping per-temperature plots.")
        return
    
    if logger:
        logger.info(f"Running per-temperature analysis for {len(temps)} level(s): {temps}")
    roi_map_all = _build_rois(epochs.info, config=config) if run_rois else {}
    
    freq_min = config.get("tfr_topography_pipeline.tfr.freq_min", 1.0)
    freq_max = config.get("tfr_topography_pipeline.tfr.freq_max", 100.0)
    n_freqs = config.get("tfr_topography_pipeline.tfr.n_freqs", 40)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles_factor = config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    
    for tval in temps:
        temp_values_valid, _ = validate_temperature_values(
            events_df[temp_col], temp_col, min_temp=35.0, max_temp=55.0, logger=logger
        )
        mask = np.abs(temp_values_valid - float(tval)) < 0.05
        mask = np.asarray(mask, dtype=bool) & np.isfinite(temp_values_valid)
        n_sel = int(mask.sum())
        if n_sel == 0:
            continue
        
        epochs_t = epochs.copy()[mask.to_numpy()]
        events_t = events_df.loc[mask].reset_index(drop=True)
        
        t_label = format_temperature_label(float(events_t[temp_col].iloc[0]))
        plots_dir_t = plots_dir / f"temperature" / f"temp-{t_label}"
        plots_dir_t.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Computing TFR for temperature {tval} ({n_sel} trials)...")
        tfr_decim = config.get("tfr_topography_pipeline.tfr.decim", 4)
        tfr_picks = config.get("tfr_topography_pipeline.tfr.picks", "eeg")
        workers = _resolve_tfr_workers(config)
        power_t = run_tfr_morlet(
            epochs_t,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=tfr_decim,
            picks=tfr_picks,
            workers=workers,
            logger=logger,
            config=config
        )
        
        task_t = f"{task}_temp{t_label}"
        if run_overall:
            plot_scalpmean_all_trials(
                power_t, plots_dir_t, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), subject=subject, task=task_t, logger=logger
            )
            contrast_scalpmean_pain_nonpain(
                power_t, events_t, plots_dir_t, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), logger=logger, subject=subject, task=task_t
            )
        
        if run_rois and len(roi_map_all) > 0:
            roi_tfrs_t = compute_roi_tfrs(epochs_t, freqs=freqs, n_cycles=n_cycles, config=config, roi_map=roi_map_all)
            plot_rois_all_trials(roi_tfrs_t, plots_dir_t, config=config, baseline=baseline_window, logger=logger)
            contrast_pain_nonpain_rois(roi_tfrs_t, events_t, plots_dir_t, config=config, baseline=baseline_window, logger=logger)


def _run_tfr_topomaps(
    power: "mne.time_frequency.EpochsTFR",
    events_df: pd.DataFrame,
    epochs: mne.Epochs,
    plots_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    plateau_tmin: float,
    plateau_tmax: float,
    temperature_strategy: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    temperature_columns = config.get("event_columns.temperature", []) if config else []
    temperature_column = next((col for col in temperature_columns if col in events_df.columns), None) if temperature_columns else None
    
    if temperature_strategy in ("pooled", "both"):
        if power is not None:
            if logger:
                logger.info("Generating topomap plots...")
            
            roi_map = _build_rois(epochs.info, config=config)
            if len(roi_map) > 0:
                plot_bands_pain_temp_contrasts(
                    power, events_df, plots_dir, config=config, baseline=baseline_window,
                    plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                )
            
            if temperature_column is not None:
                plot_topomap_grid_baseline_temps(
                    power, events_df, plots_dir, config=config, baseline=baseline_window,
                    plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                )
            
            if logger:
                logger.info("Generating temporal window topomap plots...")
            topomap_windows_config = config.get("foundational_analysis.topomap_windows", {})
            
            pain_nonpain_temporal_diff_allbands_window_size_ms = topomap_windows_config.get("pain_nonpain_temporal_diff_allbands", {}).get("window_size_ms", 100.0)
            plot_pain_nonpain_temporal_topomaps_diff_allbands(
                power, events_df, plots_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), window_size_ms=pain_nonpain_temporal_diff_allbands_window_size_ms, logger=logger
            )
            
            temporal_allbands_plateau_window_count = topomap_windows_config.get("temporal_allbands_plateau", {}).get("window_count", 5)
            plot_temporal_topomaps_allbands_plateau(
                power, events_df, plots_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), window_count=temporal_allbands_plateau_window_count, logger=logger
            )
    
    if temperature_strategy in ("per", "both") and temperature_column is not None:
        temp_values_valid, n_dropped = validate_temperature_values(
            events_df[temperature_column], temperature_column, min_temp=35.0, max_temp=55.0, logger=logger
        )
        temps = sorted(map(float, np.unique(np.round(temp_values_valid[np.isfinite(temp_values_valid)], 1))))
        if len(temps) > 0:
            if logger:
                logger.info(f"Generating per-temperature topomaps for {len(temps)} level(s)...")
            roi_map_all = _build_rois(epochs.info, config=config)
            
            for tval in temps:
                temp_values_valid_iter, _ = validate_temperature_values(
                    events_df[temperature_column], temperature_column, min_temp=35.0, max_temp=55.0, logger=logger
                )
                mask = np.abs(temp_values_valid_iter - float(tval)) < 0.05
                mask = np.asarray(mask, dtype=bool) & np.isfinite(temp_values_valid_iter)
                n_sel = int(mask.sum())
                if n_sel == 0:
                    continue
                epochs_t = epochs.copy()[mask.to_numpy()]
                events_t = events_df.loc[mask].reset_index(drop=True)
                
                t_label = format_temperature_label(float(events_t[temperature_column].iloc[0]))
                plots_dir_t = plots_dir / f"temperature" / f"temp-{t_label}"
                plots_dir_t.mkdir(parents=True, exist_ok=True)
                
                if not getattr(epochs_t, "preload", False):
                    epochs_t.load_data()
                
                freq_min = config.get("tfr_topography_pipeline.tfr.freq_min", 1.0)
                freq_max = config.get("tfr_topography_pipeline.tfr.freq_max", 100.0)
                n_freqs = config.get("tfr_topography_pipeline.tfr.n_freqs", 40)
                freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
                n_cycles_factor = config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0)
                n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
                tfr_decim = config.get("tfr_topography_pipeline.tfr.decim", 4)
                tfr_picks = config.get("tfr_topography_pipeline.tfr.picks", "eeg")
                workers = _resolve_tfr_workers(config)
                
                power_t = run_tfr_morlet(
                    epochs_t,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    decim=tfr_decim,
                    picks=tfr_picks,
                    workers=workers,
                    logger=logger,
                    config=config
                )
                
                if power_t is not None:
                    if len(roi_map_all) > 0:
                        plot_bands_pain_temp_contrasts(
                            power_t, events_t, plots_dir_t, config=config, baseline=baseline_window,
                            plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                        )
                    
                    plot_topomap_grid_baseline_temps(
                        power_t, events_t, plots_dir_t, config=config, baseline=baseline_window,
                        plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                    )
                    
                    topomap_windows_config = config.get("foundational_analysis.topomap_windows", {})
                    pain_nonpain_temporal_diff_allbands_window_size_ms = topomap_windows_config.get("pain_nonpain_temporal_diff_allbands", {}).get("window_size_ms", 100.0)
                    plot_pain_nonpain_temporal_topomaps_diff_allbands(
                        power_t, events_t, plots_dir_t, config=config, baseline=baseline_window,
                        plateau_window=(plateau_tmin, plateau_tmax), window_size_ms=pain_nonpain_temporal_diff_allbands_window_size_ms, logger=logger
                    )
                    
                    temporal_allbands_plateau_window_count = topomap_windows_config.get("temporal_allbands_plateau", {}).get("window_count", 5)
                    plot_temporal_topomaps_allbands_plateau(
                        power_t, events_t, plots_dir_t, config=config, baseline=baseline_window,
                        plateau_window=(plateau_tmin, plateau_tmax), window_count=temporal_allbands_plateau_window_count, logger=logger
                    )


def process_single_subject(
    subject: str = "001",
    task: Optional[str] = None,
    plateau_tmin: Optional[float] = None,
    plateau_tmax: Optional[float] = None,
    temperature_strategy: Optional[str] = None,
    topomaps_only: bool = False,
    run_overall: bool = True,
    run_channels: bool = True,
    run_rois: bool = True,
    channels: Optional[List[str]] = None,
    no_plots: bool = False,
) -> None:
    config = load_settings(script_name=Path(__file__).name)
    setup_matplotlib(config)
    ensure_derivatives_dataset_description(deriv_root=config.deriv_root)
    
    if task is None:
        task = config.task
    if plateau_tmin is None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        plateau_tmin = float(plateau_window[0])
    if plateau_tmax is None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        plateau_tmax = float(plateau_window[1])
    if temperature_strategy is None:
        temperature_strategy = config.get("tfr_topography_pipeline.temperature_strategy", "pooled")
    
    log_file_name = config.get("logging.file_names.time_frequency", "02_tfr_analysis.log")
    logger = get_subject_logger("tfr_analysis", subject, log_file_name, config=config)
    logger.info(f"=== Time-frequency analysis: sub-{subject}, task-{task} ===")
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="02_tfr_analysis")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    epochs, events_df = load_epochs_for_analysis(
        subject, task, align="strict", preload=True, deriv_root=config.deriv_root,
        bids_root=config.bids_root, config=config, logger=logger
    )
    
    if epochs is None:
        msg = f"Error: cleaned epochs file not found for sub-{subject}, task-{task} under {config.deriv_root}."
        if logger:
            logger.error(msg)
        raise RuntimeError(f"Failed to load epochs for sub-{subject}, task-{task}; cannot proceed")
    
    if events_df is None:
        msg = "Warning: events.tsv missing; contrasts will be skipped if needed."
        if logger:
            logger.warning(msg)
    
    power = _compute_subject_tfr(epochs, config, logger)
    if logger:
        logger.info(f"Computed TFR: type={type(power).__name__}, n_epochs={power.data.shape[0]}, n_freqs={len(power.freqs)}")
    
    baseline_window_raw = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    baseline_window = validate_baseline_window_pre_stimulus(
        baseline_window_raw, logger=logger
    )
    
    if no_plots:
        if logger:
            logger.info("Skipping all plotting operations (--no-plots enabled)")
        return
    
    if topomaps_only:
        if logger:
            logger.info("Topomaps-only mode: skipping other analyses and generating only topomap plots.")
        _run_tfr_topomaps(
            power, events_df, epochs, plots_dir, config, baseline_window,
            plateau_tmin, plateau_tmax, temperature_strategy, logger
        )
        if logger:
            logger.info("Done.")
        return
    
    _log_baseline_qc(power, baseline_window, config=config, logger=logger, output_dir=plots_dir)
    qc_baseline_plateau_power(
        power, plots_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), logger=logger
    )
    
    if temperature_strategy in ("pooled", "both"):
        _run_pooled_analysis(power, events_df, epochs, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger, run_overall=run_overall, run_channels=run_channels, run_rois=run_rois, channels=channels)
        
        temp_col = find_temperature_column_in_events(events_df) if events_df is not None else None
        if temp_col is not None:
            contrast_maxmin_temperature(
                power, events_df, plots_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), logger=logger
            )
        
        _run_tfr_topomaps(
            power, events_df, epochs, plots_dir, config, baseline_window,
            plateau_tmin, plateau_tmax, temperature_strategy, logger
        )
    
    if temperature_strategy in ("per", "both"):
        temp_col = find_temperature_column_in_events(events_df) if events_df is not None else None
        if temp_col is None:
            if logger:
                logger.warning("Per-temperature analysis requested, but no temperature column found; skipping per-temperature plots.")
        else:
            _run_per_temperature_analysis(epochs, events_df, temp_col, plots_dir, config, baseline_window, plateau_tmin, plateau_tmax, subject, task, logger, run_overall=run_overall, run_channels=run_channels, run_rois=run_rois)
            _run_tfr_topomaps(
                power, events_df, epochs, plots_dir, config, baseline_window,
                plateau_tmin, plateau_tmax, temperature_strategy, logger
            )
    
    if logger:
        logger.info("Done.")


###################################################################
# Group-Level Processing
###################################################################

def process_group(
    subjects: List[str],
    task: Optional[str] = None,
    plateau_tmin: Optional[float] = None,
    plateau_tmax: Optional[float] = None,
    temperature_strategy: Optional[str] = None,
    config=None,
    no_plots: bool = False,
) -> None:
    if config is None:
        config = load_settings(script_name=Path(__file__).name)
    
    if task is None:
        task = config.task
    if plateau_tmin is None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        plateau_tmin = float(plateau_window[0])
    if plateau_tmax is None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        plateau_tmax = float(plateau_window[1])
    if temperature_strategy is None:
        temperature_strategy = config.get("tfr_topography_pipeline.temperature_strategy", "pooled")
    
    log_file_name = config.get("logging.file_names.time_frequency", "02_tfr_analysis.log")
    logger = get_group_logger("tfr_analysis", log_file_name, config=config)
    logger.info(f"=== Time-frequency group analysis: {len(subjects)} subjects, task-{task} ===")
    logger.info(f"Subjects: {', '.join(subjects)}")
    out_dir = config.deriv_root / "group" / "eeg" / "plots" / "02_tfr_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    powers: List[mne.time_frequency.EpochsTFR] = []
    events_list: List[Optional[pd.DataFrame]] = []
    ok_subjects: List[str] = []
    for s in subjects:
        logger.info(f"--- Computing TFR for subject {s} ---")
        power, ev = _compute_power_and_events(s, task, config, logger)
        if power is not None:
            powers.append(power)
            events_list.append(ev)
            ok_subjects.append(s)
        else:
            logger.warning(f"Skipping subject {s} due to errors")

    if len(powers) < 2:
        logger.warning(f"Only {len(powers)} subjects valid; skipping group-level plots")
        return

    all_chs: list[str] = []
    seen = set()
    for p in powers:
        for ch in p.info['ch_names']:
            if ch not in seen:
                seen.add(ch)
                all_chs.append(ch)
    group_roi_map = _build_group_roi_map_from_channels(all_chs, config=config)
    counts = {roi: len(chs) for roi, chs in group_roi_map.items()}
    counts_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
    logger.info(f"Built group ROI map from union of subject channels; counts: {counts_str}")

    baseline_window_raw = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    baseline_window = validate_baseline_window_pre_stimulus(
        baseline_window_raw, logger=logger
    )
    
    if no_plots:
        if logger:
            logger.info("Skipping all group plotting operations (--no-plots enabled)")
    else:
        if temperature_strategy in ("pooled", "both"):
            group_contrast_maxmin_temperature(
                powers, events_list, out_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), logger=logger
            )

        group_rois_all_trials(
            powers, out_dir, config=config, baseline=baseline_window, logger=logger, roi_map=group_roi_map
        )
        group_contrast_pain_nonpain_rois(
            powers, events_list, out_dir, config=config, baseline=baseline_window, logger=logger, roi_map=group_roi_map
        )

        group_contrast_pain_nonpain_scalpmean(
            powers, events_list, out_dir, config=config, baseline=baseline_window, logger=logger
        )
        
        if temperature_strategy in ("pooled", "both") and len(powers) >= 2:
            if logger:
                logger.info("Generating group TFR topomap plots...")
            
            roi_map = _build_group_roi_map_from_channels(all_chs, config=config)
            if len(roi_map) > 0:
                group_plot_bands_pain_temp_contrasts(
                    powers, events_list, out_dir, config=config, baseline=baseline_window,
                    plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                )
            
            group_plot_topomap_grid_baseline_temps(
                powers, events_list, out_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), logger=logger
            )
            
            if logger:
                logger.info("Generating group temporal window topomap plots...")
            topomap_windows_config = config.get("foundational_analysis.topomap_windows", {})
            
            group_pain_nonpain_temporal_diff_allbands_window_size_ms = topomap_windows_config.get("group_pain_nonpain_temporal_diff_allbands", {}).get("window_size_ms", 100.0)
            group_plot_pain_nonpain_temporal_topomaps_diff_allbands(
                powers, events_list, out_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), window_size_ms=group_pain_nonpain_temporal_diff_allbands_window_size_ms, logger=logger
            )
            
            group_temporal_allbands_plateau_window_count = topomap_windows_config.get("group_temporal_allbands_plateau", {}).get("window_count", 5)
            group_plot_temporal_topomaps_allbands_plateau(
                powers, events_list, out_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), window_count=group_temporal_allbands_plateau_window_count, logger=logger
            )

    pain_counts_list = []
    for ev in events_list:
        n_pain = n_non = 0
        if ev is not None:
            pain_col = find_pain_column_in_events(ev)
            if pain_col is not None:
                vals, n_coerced = validate_pain_binary_values(
                    ev[pain_col], pain_col, logger=logger
                )
                n_pain = int((vals == 1).sum())
                n_non = int((vals == 0).sum())
        pain_counts_list.append((n_pain, n_non))
    
    write_group_trial_counts(ok_subjects, out_dir, "counts_pain.tsv", pain_counts_list, logger)
    _write_group_band_summary(
        powers, events_list, out_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), logger=logger
    )
    _write_group_roi_summary(
        powers, events_list, out_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), roi_map=group_roi_map, logger=logger
    )

    tf_targets: List[Optional[str]] = [None]
    if group_roi_map:
        tf_targets.extend(sorted(group_roi_map.keys()))
    for roi_name in tf_targets:
        min_subjects = int(config.get("analysis.min_subjects_for_topomaps", 3))
        group_tf_correlation(
            subjects=ok_subjects, roi=roi_name, method="auto",
            alpha=config.get("statistics.sig_alpha", 0.05),
            min_subjects=min_subjects, config=config, logger=logger
        )
    logger.info(f"Group analysis completed. Results saved to: {out_dir}")


###################################################################
# Command-Line Interface
###################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFR analysis supporting single and multiple subjects")
    sel = parser.add_mutually_exclusive_group(required=False)
    sel.add_argument(
        "--group", type=str,
        help="Group to process: 'all' or comma/space-separated subject labels without 'sub-' (e.g., '0001,0002,0003')."
    )
    sel.add_argument(
        "--subject", "-s", type=str, action="append",
        help="BIDS subject label(s) without 'sub-' prefix (e.g., 0001). Can be specified multiple times."
    )
    sel.add_argument(
        "--all-subjects", action="store_true",
        help="Process all available subjects with cleaned epochs files"
    )
    parser.add_argument("--task", "-t", type=str, default=None, help="BIDS task label (default from config)")
    parser.add_argument(
        "--plateau_tmin", type=float, default=None,
        help="Plateau window start time in seconds (for topomaps and summaries)"
    )
    parser.add_argument(
        "--plateau_tmax", type=float, default=None,
        help="Plateau window end time in seconds (for topomaps and summaries)"
    )
    parser.add_argument(
        "--temperature_strategy", "-T", type=str, choices=["pooled", "per", "both"], default=None,
        help="Temperature analysis strategy: pooled/per/both (default from config)"
    )
    parser.add_argument(
        "--topomaps-only", action="store_true",
        help="Generate only topomap plots, skipping other analyses (QC, contrasts, etc.)"
    )
    parser.add_argument(
        "--tfr-channel", type=str, action="append",
        help="Channel(s) to generate TFR plots for. Can be specified multiple times or comma-separated (e.g., --tfr-channel FT8 --tfr-channel FT9 or --tfr-channel FT8,FT9,FT10). If not specified, all channels are processed."
    )
    
    tfr_analysis_group = parser.add_mutually_exclusive_group(required=False)
    tfr_analysis_group.add_argument(
        "--tfr-all", action="store_true",
        help="Run all TFR analyses (overall, channels, and ROIs) - this is the default"
    )
    tfr_analysis_group.add_argument(
        "--tfr-overall", action="store_true",
        help="Run only overall/averaged TFR analyses (Cz plots, overall contrasts)"
    )
    tfr_analysis_group.add_argument(
        "--tfr-channels", action="store_true",
        help="Run only channel-level TFR analyses (skip overall and ROI analyses)"
    )
    tfr_analysis_group.add_argument(
        "--tfr-rois", action="store_true",
        help="Run only ROI-level TFR analyses (skip overall and channel analyses)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting operations"
    )

    args = parser.parse_args()

    config = load_settings(script_name=Path(__file__).name)
    deriv_root = config.deriv_root
    task = args.task or config.task
    
    subjects = parse_subject_args(args, config, task=task, deriv_root=deriv_root)
    
    if not subjects:
        logger = logging.getLogger(__name__)
        logger.error("No subjects provided via --group/--all-subjects/--subject. For single subject, pass --subject <ID>.")
        sys.exit(2)

    if args.tfr_overall:
        run_overall = True
        run_channels = False
        run_rois = False
    elif args.tfr_rois:
        run_overall = False
        run_channels = False
        run_rois = True
    elif args.tfr_channels:
        run_overall = False
        run_channels = True
        run_rois = False
    else:
        run_overall = True
        run_channels = True
        run_rois = True

    channels = None
    if args.tfr_channel:
        channels = []
        for ch_arg in args.tfr_channel:
            channels.extend([ch.strip() for ch in ch_arg.split(",") if ch.strip()])
        if channels:
            channels = list(set(channels))

    if len(subjects) == 1:
        try:
            process_single_subject(
                subject=subjects[0], task=args.task, plateau_tmin=args.plateau_tmin,
                plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy,
                topomaps_only=args.topomaps_only, run_overall=run_overall, run_channels=run_channels, run_rois=run_rois,
                channels=channels, no_plots=args.no_plots
            )
        except (RuntimeError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing subject {subjects[0]}: {e}")
            sys.exit(1)
    else:
        topomaps_only = args.topomaps_only
        if topomaps_only:
            logger = logging.getLogger(__name__)
            logger.info(f"Topomaps-only mode: processing {len(subjects)} subjects (skipping group analysis)...")
        else:
            logger = logging.getLogger(__name__)
            logger.info(f"Processing {len(subjects)} subjects: per-subject analysis first, then group analysis...")
        
        for subject in subjects:
            logger = logging.getLogger(__name__)
            logger.info(f"--- Processing per-subject plots for {subject} ---")
            try:
                process_single_subject(
                    subject=subject, task=args.task, plateau_tmin=args.plateau_tmin,
                    plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy,
                    topomaps_only=topomaps_only, run_overall=run_overall, run_channels=run_channels, run_rois=run_rois,
                    channels=channels, no_plots=args.no_plots
                )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error processing subject {subject}: {e}")
                continue
        
        if not topomaps_only:
            logger = logging.getLogger(__name__)
            logger.info(f"--- Processing group-level plots for {len(subjects)} subjects ---")
            process_group(
                subjects=subjects, task=args.task, plateau_tmin=args.plateau_tmin,
                plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy,
                no_plots=args.no_plots
            )
