from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import get_available_subjects, load_epochs_for_analysis, parse_subject_args
from eeg_pipeline.utils.io_utils import (
    _find_clean_epochs_path,
    deriv_plots_path,
    ensure_derivatives_dataset_description,
    get_group_logger,
    get_subject_logger,
    setup_matplotlib,
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
)
from eeg_pipeline.utils.io_utils import (
    get_viz_params as __get_viz_params,
    unwrap_figure as __unwrap_figure,
    plot_topomap_on_ax as __plot_topomap_on_ax,
    format_p_value as __format_p_value,
    format_cluster_ann as __format_cluster_ann,
    format_fdr_ann as __format_fdr_ann,
    robust_sym_vlim as __robust_sym_vlim,
)
from eeg_pipeline.plotting.plot_tfr import (
    plot_cz_all_trials_raw,
    plot_cz_all_trials,
    plot_channels_all_trials,
    contrast_channels_pain_nonpain,
    qc_baseline_plateau_power,
    contrast_pain_nonpain,
    contrast_maxmin_temperature,
    contrast_pain_nonpain_topomaps_rois,
    plot_rois_all_trials,
    plot_topomaps_bands_all_trials,
    group_topomaps_bands_all_trials,
    group_topomaps_pain_nonpain,
    group_pain_nonpain_temporal_topomaps,
    group_maxmin_temperature_temporal_topomaps,
    group_topomap_grid_baseline_temps,
    group_contrast_maxmin_temperature,
    group_rois_all_trials,
    group_contrast_pain_nonpain_rois,
    group_contrast_pain_nonpain_scalpmean,
    plot_topomap_grid_baseline_temps,
    plot_pain_nonpain_temporal_topomaps,
    contrast_pain_nonpain_rois,
    group_tf_correlation,
)
import argparse


###################################################################
# Helper Functions
###################################################################

def _get_viz_params(config=None):
    return __get_viz_params(config)


def _log(msg, logger=None, level="info"):
    if logger is None:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(msg)


def _fig(obj):
    return __unwrap_figure(obj)


def _resolve_tfr_workers(config):
    raw = os.getenv("EEG_TFR_WORKERS")
    default_workers = config.get("tfr_topography_pipeline.tfr.workers", -1)
    if not raw or raw.strip().lower() in {"auto", ""}:
        return default_workers
    
    try:
        return max(1, int(raw))
    except ValueError:
        _log(f"EEG_TFR_WORKERS={raw} invalid; using {default_workers}", level="warning")
        return default_workers


def _clip_time_window(times: np.ndarray, window: Tuple[float, float]) -> Tuple[float, float]:
    tmin_req, tmax_req = window
    tmin_eff = float(max(float(np.min(times)), float(tmin_req)))
    tmax_eff = float(min(float(np.max(times)), float(tmax_req)))
    return tmin_eff, tmax_eff


def _nanmean_eeg(vec, info):
    picks = mne.pick_types(info, eeg=True, exclude=[])
    return float(np.nanmean(vec[picks] if len(picks) > 0 else vec))


def _is_tail_trim_consistent(events_df: pd.DataFrame, epochs: "mne.Epochs", n_keep: int, logger=None) -> bool:
    if n_keep <= 0:
        return False
    sample_col = "sample" if "sample" in events_df.columns else None
    epoch_events = getattr(epochs, "events", None)
    if sample_col is None or not isinstance(epoch_events, np.ndarray):
        _log(
            "Cannot verify trial alignment during trim — missing 'sample' column or epoch events.",
            logger,
            level="error",
        )
        return False

    event_samples = pd.to_numeric(events_df[sample_col], errors="coerce").to_numpy()
    epoch_samples = np.asarray(epoch_events[:, 0], dtype=float)
    if len(event_samples) < n_keep or len(epoch_samples) < n_keep:
        return False

    head_events = event_samples[:n_keep]
    head_epochs = epoch_samples[:n_keep]
    if not np.array_equal(head_epochs, head_events):
        _log(
            f"Head mismatch detected: events and epochs do not align in the first {n_keep} samples. "
            f"Event samples (head): {head_events[:min(5, len(head_events))]}, "
            f"Epoch samples (head): {head_epochs[:min(5, len(head_epochs))]}. "
            f"This indicates a misalignment between events TSV and epochs.",
            logger,
            level="error",
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


def _log_baseline_qc(tfr_obj, baseline, config=None, logger=None):
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
    
    msg = (
        f"Baseline QC: n_time={n_time}, median_temporal_std={med_temporal_std:.3g}, "
        f"epoch_MAD={mad:.3g}, RCV={rcv:.3g}"
    )
    _log(msg, logger)
    
    if not np.isfinite(med_temporal_std) or not np.isfinite(rcv):
        _log("Baseline QC: non-finite metrics detected; baseline may be unstable.", logger, "warning")




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


def _sanitize(name):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _build_group_roi_map_from_channels(ch_names, config=None):
    from eeg_pipeline.utils.tfr_utils import build_rois_from_info
    info = mne.create_info(ch_names, sfreq=100.0, ch_types=['eeg'] * len(ch_names))
    roi_map = build_rois_from_info(info, config=config)
    return roi_map


def _find_temperature_column(events_df, config):
    if events_df is None:
        return None
    temp_cols = config.get("event_columns.temperature", [])
    if not temp_cols:
        raise KeyError("event_columns.temperature not found in config. Required for temperature-based analysis.")
    return next((c for c in temp_cols if c in events_df.columns), None)


def _find_pain_binary_column(events_df, config):
    if events_df is None:
        return None
    pain_cols = config.get("event_columns.pain_binary", [])
    if not pain_cols:
        raise KeyError("event_columns.pain_binary not found in config. Required for pain contrast analysis.")
    return next((c for c in pain_cols if c in events_df.columns), None)


def _format_temp_label(val):
    import re
    try:
        return f"{float(val):.1f}".replace(".", "p")
    except Exception:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(val))


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
    _log(f"Channel '{preferred}' not found; using '{fallback}' instead.", logger, "warning")
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


def _plot_topomap_on_ax(ax, data, info, mask=None, mask_params=None, vmin=None, vmax=None, config=None):
    return __plot_topomap_on_ax(ax, data, info, mask=mask, mask_params=mask_params, vmin=vmin, vmax=vmax, config=config)


def _format_p_value(p):
    return __format_p_value(p)


def _format_cluster_ann(p, k=None, mass=None, config=None):
    return __format_cluster_ann(p, k=k, mass=mass, config=config)


def _extract_epochwise_channel_values(tfr_epochs, fmin, fmax, tmin, tmax):
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    fmask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    tmask = (times >= float(tmin)) & (times < float(tmax))
    if fmask.sum() == 0 or tmask.sum() == 0:
        return None
    data = np.asarray(tfr_epochs.data)[:, :, fmask, :][:, :, :, tmask]
    return data.mean(axis=(2, 3))


def _format_fdr_ann(q_min: Optional[float], k_rej: Optional[int], alpha: float = 0.05) -> str:
    return __format_fdr_ann(q_min, k_rej, alpha)


def _robust_sym_vlim(arrs, q_low: float = 0.02, q_high: float = 0.98, cap: float = 0.25, min_v: float = 1e-6) -> float:
    return __robust_sym_vlim(arrs, q_low=q_low, q_high=q_high, cap=cap, min_v=min_v)


def _average_tfr_band(tfr_avg, fmin, fmax, tmin, tmax):
    freqs = np.asarray(tfr_avg.freqs)
    times = np.asarray(tfr_avg.times)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    t_mask = (times >= tmin) & (times < tmax)
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    sel = tfr_avg.data[:, f_mask, :][:, :, t_mask]
    return sel.mean(axis=(1, 2))


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
    strict_mode = config.get("analysis.strict_mode", True)
    return compute_subject_tfr(
        subject=subject,
        task=task,
        freq_min=freq_min,
        freq_max=freq_max,
        n_freqs=n_freqs,
        n_cycles_factor=n_cycles_factor,
        tfr_decim=tfr_decim,
        tfr_picks=tfr_picks,
        strict_mode=strict_mode,
        workers=-1,
        logger=logger,
    )


def _align_avg_tfrs(
    tfr_list: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Info], Optional[np.ndarray]]:
    return align_avg_tfrs(tfr_list, logger=logger)


def _avg_alltrials_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    logger: Optional[logging.Logger] = None
) -> "mne.time_frequency.AverageTFR":
    if baseline is None:
        baseline = (-5.0, -0.01)
    return avg_alltrials_to_avg_tfr(power, baseline=baseline, logger=logger)


def _avg_by_mask_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    mask: np.ndarray,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    if baseline is None:
        baseline = (-5.0, -0.01)
    return avg_by_mask_to_avg_tfr(power, mask, baseline=baseline, logger=logger)


def _collect_group_temperatures(events_by_subj: List[Optional[pd.DataFrame]], config) -> list[float]:
    temp_cols = config.get("event_columns.temperature", [])
    return collect_group_temperatures(events_by_subj, temp_cols)


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

def _write_group_pain_counts_from_events(
    subjects: List[str],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    logger: Optional[logging.Logger] = None,
) -> None:
    rows = []
    for subj, ev in zip(subjects, events_list):
        n_pain = n_non = 0
        if ev is not None:
            pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c in ev.columns), None)
            if pain_col is not None:
                vals = pd.to_numeric(ev[pain_col], errors="coerce")
                n_pain = int((vals == 1).sum())
                n_non = int((vals == 0).sum())
        rows.append({
            "subject": subj,
            "n_pain": n_pain,
            "n_nonpain": n_non,
            "n_total": n_pain + n_non,
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    total = df[["n_pain", "n_nonpain", "n_total"]].sum()
    total_row = {"subject": "TOTAL", **{k: int(v) for k, v in total.to_dict().items()}}
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "counts_pain.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    _log(f"Saved counts: {out_path}", logger)


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
    info_all, data_all = _align_avg_tfrs(avg_list, logger=logger)
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
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        mask_p, mask_n = vals == 1, vals == 0
        if mask_p.sum() == 0 or mask_n.sum() == 0:
            continue
        a_p = _avg_by_mask_to_avg_tfr(p, mask_p, baseline=baseline)
        a_n = _avg_by_mask_to_avg_tfr(p, mask_n, baseline=baseline)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    
    info_p, data_p = _align_avg_tfrs(avg_pain, logger=logger)
    info_n, data_n = _align_avg_tfrs(avg_non, logger=logger)
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

    temps = _collect_group_temperatures(events_by_subj, config)
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
            mask = pd.to_numeric(ev[tcol], errors="coerce").round(1) == round(float(tval), 1)
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            a = _avg_by_mask_to_avg_tfr(p, mask)
            if a is not None:
                avg_list.append(a)
                ns += 1
        info_t, data_t = _align_avg_tfrs(avg_list, logger=logger)
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
        _log(f"Saved band summary: {out_path}", logger)


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
        avg_all.append(_avg_alltrials_to_avg_tfr(p, logger=logger))
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
        info_c, data_c = _align_avg_tfrs(rois_list, logger=logger)
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
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        mask_p = np.asarray(vals == 1, dtype=bool)
        mask_n = np.asarray(vals == 0, dtype=bool)
        if mask_p.sum() == 0 or mask_n.sum() == 0:
            continue
        a_p = _avg_by_mask_to_avg_tfr(p, mask_p, baseline=baseline)
        a_n = _avg_by_mask_to_avg_tfr(p, mask_n, baseline=baseline)
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

    temps = _collect_group_temperatures(events_by_subj, config)
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
                    mask = pd.to_numeric(ev[tcol], errors="coerce").round(1) == round(float(tval), 1)
                    mask = np.asarray(mask, dtype=bool)
                    if mask.sum() == 0:
                        continue
                    a = _avg_by_mask_to_avg_tfr(p, mask)
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
        _log(f"Saved ROI summary: {out_path}", logger)


###################################################################
# Single-Subject Processing
###################################################################

def process_single_subject(
    subject: str = "001",
    task: Optional[str] = None,
    plateau_tmin: Optional[float] = None,
    plateau_tmax: Optional[float] = None,
    temperature_strategy: Optional[str] = None
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
    
    log_file_name = config.get("logging.file_names.time_frequency", "03_tfr_topography_pipeline.log")
    logger = get_subject_logger("tfr_topography_pipeline", subject, log_file_name, config=config)
    logger.info(f"=== Time-frequency analysis: sub-{subject}, task-{task} ===")
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="03_tfr_topography_pipeline")
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs, events_df = load_epochs_for_analysis(
        subject, task, align="strict", preload=True, deriv_root=config.deriv_root,
        bids_root=config.bids_root, config=config, logger=logger
    )
    
    if epochs is None:
        msg = f"Error: cleaned epochs file not found for sub-{subject}, task-{task} under {config.deriv_root}."
        _log(msg, logger, "error")
        sys.exit(1)
    
    if events_df is None:
        msg = "Warning: events.tsv missing; contrasts will be skipped if needed."
        _log(msg, logger, "warning")

    freq_min = config.get("tfr_topography_pipeline.tfr.freq_min", 1.0)
    freq_max = config.get("tfr_topography_pipeline.tfr.freq_max", 100.0)
    n_freqs = config.get("tfr_topography_pipeline.tfr.n_freqs", 40)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles_factor = config.get("tfr_topography_pipeline.tfr.n_cycles_factor", 2.0)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, min_cycles=3.0)
    _log("Computing per-trial TFR (Morlet)...", logger)
    tfr_decim = config.get("tfr_topography_pipeline.tfr.decim", 4)
    tfr_picks = config.get("tfr_topography_pipeline.tfr.picks", "eeg")
    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False, average=False, decim=tfr_decim,
        n_jobs=-1, picks=tfr_picks, verbose=False
    )
    _log(f"Computed TFR: type={type(power).__name__}, n_epochs={power.data.shape[0]}, n_freqs={len(power.freqs)}", logger)

    baseline_window = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    _log_baseline_qc(power, baseline_window, config=config, logger=logger)
    plot_cz_all_trials_raw(power, plots_dir, config=config, logger=logger)
    qc_baseline_plateau_power(
        power, plots_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), logger=logger
    )

    if temperature_strategy in ("pooled", "both"):
        plot_cz_all_trials(
            power, plots_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), subject=subject, task=task, logger=logger
        )
        contrast_pain_nonpain(
            power, events_df, plots_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), logger=logger, subject=subject
        )
        plot_pain_nonpain_temporal_topomaps(
            power, events_df, plots_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), window_count=5, logger=logger
        )
        
        _log("Generating comprehensive channel-level TFR plots...", logger)
        plot_channels_all_trials(
            power, plots_dir, config=config, baseline=baseline_window,
            logger=logger, subject=subject, task=task
        )
        contrast_channels_pain_nonpain(
            power, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger
        )

        _log("Building ROIs and computing ROI TFRs (pooled)...", logger)
        roi_map = _build_rois(epochs.info, config=config)
        if len(roi_map) == 0:
            _log("No ROI channels found in montage; skipping ROI analysis.", logger, "warning")
        else:
            roi_tfrs = compute_roi_tfrs(epochs, freqs=freqs, n_cycles=n_cycles, config=config, roi_map=roi_map)
            plot_rois_all_trials(roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger)
            contrast_pain_nonpain_rois(roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger)
            plot_topomaps_bands_all_trials(
                power, plots_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), logger=logger
            )
            contrast_pain_nonpain_topomaps_rois(
                power, events_df, roi_map, plots_dir, config=config, baseline=baseline_window,
                plateau_window=(plateau_tmin, plateau_tmax), logger=logger
            )

    temp_col = _find_temperature_column(events_df, config)
    if temperature_strategy in ("pooled", "both") and events_df is not None and temp_col is not None:
        plot_topomap_grid_baseline_temps(
            power, events_df, plots_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), logger=logger
        )
        contrast_maxmin_temperature(
            power, events_df, plots_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), logger=logger
        )
    
    if temperature_strategy in ("per", "both"):
        if events_df is None or temp_col is None:
            _log("Per-temperature analysis requested, but no temperature column found; skipping per-temperature plots.", logger, "warning")
        else:
            temps = sorted(map(float, pd.to_numeric(events_df[temp_col], errors="coerce").round(1).dropna().unique()))
            if len(temps) == 0:
                _log("No temperatures found in events; skipping per-temperature plots.", logger, "warning")
            else:
                _log(f"Running per-temperature analysis for {len(temps)} level(s): {temps}", logger, "info")
                roi_map_all = _build_rois(epochs.info, config=config)
                for tval in temps:
                    mask = pd.to_numeric(events_df[temp_col], errors="coerce").round(1) == round(float(tval), 1)
                    n_sel = int(mask.sum())
                    if n_sel == 0:
                        continue
                    epochs_t = epochs.copy()[mask.to_numpy()]
                    events_t = events_df.loc[mask].reset_index(drop=True)

                    t_label = _format_temp_label(float(events_t[temp_col].iloc[0]))
                    plots_dir_t = plots_dir / f"temperature" / f"temp-{t_label}"
                    plots_dir_t.mkdir(parents=True, exist_ok=True)

                    _log(f"Computing TFR for temperature {tval} ({n_sel} trials)...", logger)
                    power_t = mne.time_frequency.tfr_morlet(
                        epochs_t, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, average=False, decim=config.get("tfr_topography_pipeline.tfr.decim", 4),
                        picks=config.get("tfr_topography_pipeline.tfr.picks", "eeg"), n_jobs=-1
                    )

                    task_t = f"{task}_temp{t_label}"
                    plot_cz_all_trials(
                        power_t, plots_dir_t, config=config, baseline=baseline_window,
                        plateau_window=(plateau_tmin, plateau_tmax), subject=subject, task=task_t, logger=logger
                    )
                    contrast_pain_nonpain(
                        power_t, events_t, plots_dir_t, config=config, baseline=baseline_window,
                        plateau_window=(plateau_tmin, plateau_tmax), logger=logger, subject=subject, task=task_t
                    )

                    if len(roi_map_all) == 0:
                        _log("No ROI channels found; skipping ROI analyses for temperature subset.", logger, "warning")
                    else:
                        roi_tfrs_t = compute_roi_tfrs(epochs_t, freqs=freqs, n_cycles=n_cycles, config=config, roi_map=roi_map_all)
                        plot_rois_all_trials(roi_tfrs_t, plots_dir_t, config=config, baseline=baseline_window, logger=logger)
                        contrast_pain_nonpain_rois(roi_tfrs_t, events_t, plots_dir_t, config=config, baseline=baseline_window, logger=logger)
                        plot_topomaps_bands_all_trials(
                            power_t, plots_dir_t, config=config, baseline=baseline_window,
                            plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                        )
                        contrast_pain_nonpain_topomaps_rois(
                            power_t, events_t, roi_map_all, plots_dir_t, config=config,
                            baseline=baseline_window, plateau_window=(plateau_tmin, plateau_tmax), logger=logger
                        )
    
    _log("Done.", logger)


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
    
    log_file_name = config.get("logging.file_names.time_frequency", "03_tfr_topography_pipeline.log")
    logger = get_group_logger("tfr_topography_pipeline", log_file_name, config=config)
    logger.info(f"=== Time-frequency group analysis: {len(subjects)} subjects, task-{task} ===")
    logger.info(f"Subjects: {', '.join(subjects)}")
    out_dir = config.deriv_root / "group" / "eeg" / "plots" / "03_tfr_topography_pipeline"
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

    baseline_window = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    group_topomaps_bands_all_trials(
        powers, out_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), logger=logger
    )

    group_topomaps_pain_nonpain(
        powers, events_list, out_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), logger=logger
    )

    if temperature_strategy in ("pooled", "both"):
        group_topomap_grid_baseline_temps(
            powers, events_list, out_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), logger=logger
        )
        group_contrast_maxmin_temperature(
            powers, events_list, out_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), logger=logger
        )
        group_maxmin_temperature_temporal_topomaps(
            powers, events_list, out_dir, config=config, baseline=baseline_window,
            plateau_window=(plateau_tmin, plateau_tmax), window_count=5, logger=logger
        )

    group_rois_all_trials(
        powers, out_dir, config=config, baseline=baseline_window, logger=logger, roi_map=group_roi_map
    )
    group_contrast_pain_nonpain_rois(
        powers, events_list, out_dir, config=config, baseline=baseline_window, logger=logger, roi_map=group_roi_map
    )

    group_pain_nonpain_temporal_topomaps(
        powers, events_list, out_dir, config=config, baseline=baseline_window,
        plateau_window=(plateau_tmin, plateau_tmax), window_count=5, logger=logger
    )

    group_contrast_pain_nonpain_scalpmean(
        powers, events_list, out_dir, config=config, baseline=baseline_window, logger=logger
    )

    _write_group_pain_counts_from_events(ok_subjects, events_list, out_dir, config, logger)
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
        group_tf_correlation(
            subjects=ok_subjects, roi=roi_name, method="auto",
            alpha=config.get("statistics.sig_alpha", 0.05),
            min_subjects=max(3, min(len(ok_subjects), 3)), config=config, logger=logger
        )
    logger.info(f"Group analysis completed. Results saved to: {out_dir}")


###################################################################
# Command-Line Interface
###################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-frequency analysis supporting single and multiple subjects")
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

    args = parser.parse_args()

    config = load_settings(script_name=Path(__file__).name)
    deriv_root = config.deriv_root
    task = args.task or config.task
    
    subjects = parse_subject_args(args, config, task=task, deriv_root=deriv_root)
    
    if not subjects:
        _log("No subjects provided via --group/--all-subjects/--subject. For single subject, pass --subject <ID>.")
        sys.exit(2)

    if len(subjects) == 1:
        process_single_subject(
            subject=subjects[0], task=args.task, plateau_tmin=args.plateau_tmin,
            plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy
        )
    else:
        _log(f"Processing {len(subjects)} subjects: per-subject analysis first, then group analysis...")
        for subject in subjects:
            _log(f"--- Processing per-subject plots for {subject} ---")
            try:
                process_single_subject(
                    subject=subject, task=args.task, plateau_tmin=args.plateau_tmin,
                    plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy
                )
            except Exception as e:
                _log(f"Error processing subject {subject}: {e}", logger=None, level="error")
                continue
        
        _log(f"--- Processing group-level plots for {len(subjects)} subjects ---")
        process_group(
            subjects=subjects, task=args.task, plateau_tmin=args.plateau_tmin,
            plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy
        )
