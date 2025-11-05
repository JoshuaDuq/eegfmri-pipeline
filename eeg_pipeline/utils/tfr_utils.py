from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mne
import numpy as np
import pandas as pd

from .config_loader import load_settings
from .data_loading import load_epochs_for_analysis


###################################################################
# ROI Channel Operations
###################################################################

def canonicalize_ch_name(ch: str) -> str:
    s = ch.strip()
    try:
        s = re.sub(r"^(EEG[ \-_]*)", "", s, flags=re.IGNORECASE)
        s = re.split(r"[-/]", s)[0]
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"(Ref|LE|RE|M1|M2|A1|A2|AVG|AVE)$", "", s, flags=re.IGNORECASE)
    except Exception:
        return ch
    return s


def get_rois(config) -> Dict[str, List[str]]:
    if config is None:
        raise ValueError("config is required for get_rois")
    rois = config.get("time_frequency_analysis.rois")
    if rois is None:
        raise ValueError("time_frequency_analysis.rois not found in config")
    return dict(rois)


def find_roi_channels(info: mne.Info, patterns: List[str]) -> List[str]:
    chs = info["ch_names"]
    out: List[str] = []
    canon_map = {ch: canonicalize_ch_name(ch) for ch in chs}
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for ch in chs:
            cn = canon_map.get(ch, ch)
            if rx.match(ch) or rx.match(cn):
                out.append(ch)
    seen = set()
    ordered: List[str] = []
    for ch in chs:
        if ch in out and ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return ordered


def build_rois_from_info(info: mne.Info, config=None) -> Dict[str, List[str]]:
    rois = {}
    roi_defs = get_rois(config)
    for roi, pats in roi_defs.items():
        chans = find_roi_channels(info, pats)
        if chans:
            rois[roi] = chans
    return rois


###################################################################
# TFR Parameter Computation
###################################################################

def compute_adaptive_n_cycles(
    freqs: Union[np.ndarray, list],
    cycles_factor: float = 2.0,
    min_cycles: float = 3.0,
    max_cycles: Optional[float] = None
) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    base_cycles = freqs / cycles_factor
    n_cycles = np.maximum(base_cycles, min_cycles)
    if max_cycles is not None:
        n_cycles = np.minimum(n_cycles, max_cycles)
    return n_cycles


def log_tfr_resolution(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    logger: Optional[logging.Logger] = None
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    time_res = n_cycles / freqs
    freq_res = freqs / n_cycles

    logger.info("TFR Resolution Summary:")
    logger.info(f"  Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    logger.info(f"  n_cycles range: {n_cycles.min():.1f} - {n_cycles.max():.1f}")
    logger.info(f"  Time resolution: {time_res.min():.3f} - {time_res.max():.3f} s")
    logger.info(f"  Frequency resolution: {freq_res.min():.2f} - {freq_res.max():.2f} Hz")

    low_freq_mask = freqs <= 8
    if np.any(low_freq_mask):
        low_cycles = n_cycles[low_freq_mask]
        if np.any(low_cycles < 2.5):
            logger.warning(f"Low n_cycles detected in theta/alpha: min={low_cycles.min():.1f}")
        else:
            logger.info(f"Good low-frequency resolution: min n_cycles={low_cycles.min():.1f}")


def validate_tfr_parameters(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    logger: Optional[logging.Logger] = None
) -> bool:
    if logger is None:
        logger = logging.getLogger(__name__)

    issues = []
    if np.any(n_cycles < 2.0):
        issues.append(f"n_cycles too low: min={n_cycles.min():.1f}")
    if np.any(freqs >= sfreq / 2):
        issues.append(f"Frequencies above Nyquist: max_freq={freqs.max():.1f}, Nyquist={sfreq/2:.1f}")
    max_time_res = np.max(n_cycles / freqs)
    min_freq = np.min(freqs)
    time_res_threshold = 5.0 if min_freq <= 2.0 else 2.0
    if max_time_res > time_res_threshold:
        issues.append(f"Excessive time resolution: max={max_time_res:.1f}s (threshold={time_res_threshold:.1f}s for min_freq={min_freq:.1f}Hz)")

    if issues:
        for issue in issues:
            logger.warning(f"TFR parameter issue: {issue}")
        return False

    logger.info("TFR parameters validation passed")
    return True


def resolve_tfr_workers(workers_default: int = -1) -> int:
    raw = os.getenv("EEG_TFR_WORKERS")
    if not raw or raw.strip().lower() in {"auto", ""}:
        return workers_default

    try:
        return max(1, int(raw))
    except ValueError:
        logger = logging.getLogger(__name__)
        logger.warning(f"EEG_TFR_WORKERS={raw} invalid; using {workers_default}")
        return workers_default


def get_bands_for_tfr(
    tfr=None,
    max_freq_available: Optional[float] = None,
    band_bounds: Optional[Dict[str, Tuple[float, Optional[float]]]] = None,
    config=None,
) -> Dict[str, Tuple[float, float]]:
    if band_bounds is None:
        if config is None:
            config = load_settings()
        band_bounds = {
            k: (v[0], (v[1] if v[1] is not None else None))
            for k, v in dict(config.get("tfr_topography_pipeline.bands", {
                "theta": [4.0, 7.9],
                "alpha": [8.0, 12.9],
                "beta": [13.0, 30.0],
                "gamma": [30.1, 80.0],
            })).items()
        }

    max_freq = max_freq_available or (float(np.max(tfr.freqs)) if tfr else 80.0)

    bands = {k: v for k, v in band_bounds.items() if k in ["delta", "theta", "alpha", "beta"]}

    gamma_lower, gamma_upper = band_bounds.get("gamma", (30.1, 80.0))
    bands["gamma"] = (gamma_lower, min(gamma_upper or 80.0, max_freq))
    return bands


###################################################################
# TFR I/O with Unit Standardization
###################################################################

def read_tfr_average_with_logratio(
    tfr_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    min_baseline_samples: int = 5,
) -> Optional["mne.time_frequency.AverageTFR"]:
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        read = getattr(mne.time_frequency, "read_tfrs", None)
        if read is None:
            logger.warning(f"read_tfrs unavailable for {tfr_path}")
            return None

        tfrs = read(str(tfr_path))
        if tfrs is None:
            logger.warning(f"read_tfrs returned None for {tfr_path}")
            return None
            
        if not isinstance(tfrs, list):
            tfrs = [tfrs]
            
        if len(tfrs) == 0:
            logger.warning(f"No TFRs found in {tfr_path}")
            return None

        t = tfrs[0]
    except Exception as e:
        logger.warning(f"Error reading TFR from {tfr_path}: {e}")
        return None
    data = getattr(t, "data", None)
    if data is not None and getattr(data, "ndim", 0) == 4:
        t = t.average()

    mode_detected, detected_by_sidecar = _detect_baseline_mode(
        t, tfr_path, logger
    )

    if mode_detected is None:
        return _apply_baseline_if_needed(t, baseline_window, min_baseline_samples, logger, tfr_path)

    if mode_detected == "logratio":
        return t
    
    if mode_detected == "ratio":
        if not detected_by_sidecar:
            logger.error(
                f"{tfr_path}: detected 'ratio' but sidecar verification failed. "
                f"Refusing conversion to prevent invalid results."
            )
            return None
        t.data = np.log10(np.maximum(t.data, 1e-20))
        if hasattr(t, "comment"):
            t.comment = (t.comment or "") + " | converted ratio->log10ratio"
        return t
    
    if mode_detected == "percent":
        if not detected_by_sidecar:
            logger.error(
                f"{tfr_path}: detected 'percent' but sidecar verification failed. "
                f"Refusing conversion to prevent invalid results."
            )
            return None
        ratio = 1.0 + (t.data / 100.0)
        t.data = np.log10(np.clip(ratio, 1e-20, np.inf))
        if hasattr(t, "comment"):
            t.comment = (t.comment or "") + " | converted percent->log10ratio"
        return t

    return None


def _detect_baseline_mode(
    t,
    tfr_path: Union[str, "os.PathLike[str]"],
    logger: logging.Logger
) -> Tuple[Optional[str], bool]:
    sidecar_path = str(tfr_path).rsplit(".", 1)[0] + ".json"
    mode_detected = None
    detected_by_sidecar = False

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if bool(meta.get("baseline_applied", False)):
            mode_detected = str(meta.get("baseline_mode", "")).strip().lower() or None
            detected_by_sidecar = mode_detected is not None
            if logger:
                logger.info(
                    f"{getattr(t, 'comment', '') or tfr_path}: "
                    f"Sidecar found. Baseline mode={mode_detected}"
                )
    except FileNotFoundError:
        logger.error(
            f"CRITICAL: Missing TFR sidecar for {tfr_path}. "
            f"Cannot determine baseline units. To prevent invalid results, "
            f"this file will not be loaded. Re-run feature extraction."
        )
        return None, False
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed reading TFR sidecar for {tfr_path}: {exc}")
        return None, False

    if mode_detected not in {"logratio", "ratio", "percent"}:
        logger.warning(f"Unsupported baseline mode '{mode_detected}' "
                       f"from sidecar for {tfr_path}; skipping.")
        return None, False
    
    return mode_detected, detected_by_sidecar


def _apply_baseline_if_needed(
    t,
    baseline_window: Tuple[float, float],
    min_baseline_samples: int,
    logger: logging.Logger,
    tfr_path: Union[str, "os.PathLike[str]"]
) -> Optional["mne.time_frequency.AverageTFR"]:
    times = np.asarray(t.times)
    b_start, b_end = float(baseline_window[0]), float(baseline_window[1])
    if b_end > 0:
        b_end = 0.0
    
    mask_n = int(((times >= b_start) & (times < b_end)).sum())
    if mask_n < int(min_baseline_samples):
        logger.warning(
            f"Insufficient baseline samples ({mask_n}) in window "
            f"{baseline_window} for {tfr_path}."
        )
        return None
    
    t.apply_baseline(baseline=(b_start, b_end), mode="logratio")
    if hasattr(t, "comment"):
        t.comment = (t.comment or "") + " | baseline(logratio) applied"
    return t


def save_tfr_with_sidecar(
    tfr: Union["mne.time_frequency.EpochsTFR", "mne.time_frequency.AverageTFR"],
    out_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    from pathlib import Path

    if logger is None:
        logger = logging.getLogger(__name__)

    if mode is None:
        if config is None:
            config = load_settings()
        mode = str(config.get("tfr_processing.baseline_mode", "logratio"))

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tfr.save(str(p), overwrite=True)
    
    sidecar = {
        "baseline_applied": True,
        "baseline_mode": str(mode),
        "units": ("log10ratio" if str(mode).lower() == "logratio" else str(mode)),
        "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
        "created_by": "tfr_utils.save_tfr_with_sidecar",
        "comment": getattr(tfr, "comment", None),
    }
    with open(p.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    
    logger.info(f"Saved TFR and sidecar: {p} (+ .json)")


###################################################################
# TFR Baseline Operations
###################################################################

def validate_baseline_window(
    times: np.ndarray,
    baseline: Tuple[float, float],
    min_samples: int = 5,
    strict_mode: Optional[bool] = None,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    if strict_mode is None:
        if config is None:
            config = load_settings()
        strict_mode = bool(config.get("analysis.strict_mode", True))
    
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)
    
    if b_end > 0:
        raise ValueError(f"Baseline window must end at or before 0 s, got [{b_start}, {b_end}]")
    
    mask = (times >= b_start) & (times < b_end)
    n_samples = int(mask.sum())
    
    if n_samples < min_samples:
        msg = (
            f"Baseline window [{b_start:.3f}, {b_end:.3f}] s has {n_samples} samples; "
            f"at least {min_samples} required"
        )
        if strict_mode:
            raise ValueError(msg)
        logger = logging.getLogger(__name__)
        logger.warning(msg)
    
    return b_start, b_end, mask


def validate_baseline_indices(
    times: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, np.ndarray]:
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)

    if b_end > 0:
        raise ValueError("Baseline window must end at or before 0 s (stimulus onset)")

    baseline_mask = (times >= b_start) & (times < b_end)
    idx = np.where(baseline_mask)[0]

    if len(idx) < min_samples:
        raise ValueError(
            f"Baseline window contains only {len(idx)} samples "
            f"(minimum {min_samples} required)"
        )

    if logger is not None:
        timespan = (float(times[idx[0]]), float(times[idx[-1]]))
        logger.info(
            f"Baseline indices: window [{b_start:.3f}, {b_end:.3f}] s maps to "
            f"indices [{idx[0]}, {idx[-1]}] with actual timespan [{timespan[0]:.3f}, {timespan[1]:.3f}] s "
            f"(n_samples={len(idx)})"
        )

    return b_start, b_end, idx


def validate_baseline_window_for_times(
    times: np.ndarray,
    baseline_window: Optional[Tuple[float, float]] = None,
    min_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    if baseline_window is None:
        if config is None:
            config = load_settings()
        baseline_window = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    if min_samples is None:
        if config is None:
            config = load_settings()
        min_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples", 5))
    
    return validate_baseline_indices(
        times,
        baseline_window,
        min_samples=min_samples,
        logger=logger
    )


def validate_plateau_window_for_times(
    times: np.ndarray,
    plateau_window: Optional[Tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    if plateau_window is None:
        if config is None:
            config = load_settings()
        plateau_window = tuple(config.get("time_frequency_analysis.plateau_window", [3.0, 10.5]))
    
    plateau_start, plateau_end = plateau_window
    
    if plateau_start >= plateau_end:
        raise ValueError(
            f"Plateau window invalid: start ({plateau_start}) must be < end ({plateau_end})"
        )
    
    if plateau_start < 0:
        raise ValueError(
            f"Plateau window must start at or after stimulus onset (0 s), got start={plateau_start}"
        )
    
    mask = (times >= plateau_start) & (times < plateau_end)
    idx = np.where(mask)[0]
    
    if len(idx) < 1:
        raise ValueError(
            f"Plateau window [{plateau_start:.3f}, {plateau_end:.3f}] s contains no samples "
            f"for time range [{times.min():.3f}, {times.max():.3f}] s"
        )
    
    if logger is not None:
        actual_tmin = float(times[idx[0]])
        actual_tmax = float(times[idx[-1]])
        logger.info(
            f"Plateau window [{plateau_start:.3f}, {plateau_end:.3f}] s maps to "
            f"indices [{idx[0]}, {idx[-1]}] with actual timespan [{actual_tmin:.3f}, {actual_tmax:.3f}] s "
            f"(n_samples={len(idx)})"
        )
    
    return plateau_start, plateau_end, idx


def apply_baseline_safe(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]],
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
    min_samples: Optional[int] = None,
    strict_mode: Optional[bool] = None,
    config=None,
) -> bool:
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        config = load_settings()

    if mode is None:
        mode = str(config.get("tfr_processing.baseline_mode", "logratio"))

    sentinel = "BASELINED:"
    comment = getattr(tfr_obj, "comment", None)
    if not force and isinstance(comment, str) and sentinel in comment:
        logger.debug("Detected baseline-corrected TFR by sentinel; skipping re-application.")
        return True

    if isinstance(tfr_obj, mne.time_frequency.AverageTFR):
        logger.warning("Applying baseline to AverageTFR (averaged) — prefer per-epoch baseline")

    times = np.asarray(tfr_obj.times)
    
    if min_samples is None:
        min_samples = int(config.get("time_frequency_analysis.min_baseline_samples", 5))
    
    if strict_mode is None:
        strict_mode = bool(config.get("analysis.strict_mode", True))
    
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)
    
    if b_end > 0:
        logger.warning(f"Baseline window must end at or before 0 s, got [{b_start}, {b_end}]; skipping baseline.")
        return False
    
    tmin_avail, tmax_avail = float(times[0]), float(times[-1])
    b_start_clip = max(b_start, tmin_avail)
    b_end_clip = min(b_end, 0.0, tmax_avail)
    
    times_arr = np.asarray(times)
    bl_mask = (times_arr >= b_start_clip) & (times_arr <= b_end_clip)
    n_samples = int(bl_mask.sum())
    
    if b_start_clip >= b_end_clip or n_samples < max(1, min_samples):
        logger.warning(
            f"Baseline window [{b_start}, {b_end}] invalid/insufficient for available times "
            f"[{tmin_avail}, {tmax_avail}] (samples={n_samples}); skipping baseline."
        )
        if strict_mode:
            raise ValueError(
                f"Baseline window [{b_start}, {b_end}] has {n_samples} samples; "
                f"at least {min_samples} required"
            )
        return False
    
    if (b_start_clip != b_start) or (b_end_clip != b_end):
        logger.info(
            f"Clipped baseline window from [{b_start}, {b_end}] to "
            f"[{b_start_clip}, {b_end_clip}] to fit data range."
        )
    
    tfr_obj.apply_baseline(baseline=(b_start_clip, b_end_clip), mode=mode)

    prev = getattr(tfr_obj, "comment", "")
    tag = f"{sentinel}mode={mode};win=({b_start_clip:.3f},{b_end_clip:.3f})"
    tfr_obj.comment = f"{prev} | {tag}" if prev else tag

    logger.info(f"Applied baseline {(b_start_clip, b_end_clip)} with mode='{mode}'.")
    return True


def log_baseline_qc(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: int = 5,
    logger: Optional[logging.Logger] = None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    times = np.asarray(tfr_obj.times)
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
    logger.info(msg)
    
    if not np.isfinite(med_temporal_std) or not np.isfinite(rcv):
        logger.warning("Baseline QC: non-finite metrics detected; baseline may be unstable.")


def apply_baseline_and_crop(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]],
    crop_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force_baseline: bool = False,
    min_samples: Optional[int] = None,
    strict_mode: Optional[bool] = None,
) -> Tuple[float, float]:
    if logger is None:
        logger = logging.getLogger(__name__)

    baseline_applied = apply_baseline_safe(
        tfr_obj,
        baseline=baseline,
        mode=mode,
        logger=logger,
        force=force_baseline,
        min_samples=min_samples,
        strict_mode=strict_mode,
    )

    baseline_used = baseline
    if baseline_applied and hasattr(tfr_obj, "comment"):
        import re
        comment = str(tfr_obj.comment)
        match = re.search(r"BASELINED:.*?win=\(([^,]+),([^)]+)\)", comment)
        if match:
            baseline_used = (float(match.group(1)), float(match.group(2)))

    if crop_window is not None:
        times = np.asarray(tfr_obj.times)
        tmin_req, tmax_req = crop_window
        tmin_avail, tmax_avail = float(times[0]), float(times[-1])
        
        tmin_req = float(times.min()) if tmin_req is None else float(tmin_req)
        tmax_req = float(times.max()) if tmax_req is None else float(tmax_req)
        
        tmin_clip = max(tmin_req, tmin_avail)
        tmax_clip = min(tmax_req, tmax_avail)
        
        if tmin_clip > tmax_clip:
            logger.warning(
                f"Requested crop window [{tmin_req}, {tmax_req}] invalid for available times "
                f"[{tmin_avail}, {tmax_avail}]; using full range."
            )
            tmin_clip, tmax_clip = tmin_avail, tmax_avail
        elif tmin_clip != tmin_req or tmax_clip != tmax_req:
            logger.info(
                f"Clipped crop window from [{tmin_req}, {tmax_req}] to "
                f"[{tmin_clip}, {tmax_clip}] to fit data range."
            )
        
        tfr_obj.crop(tmin=tmin_clip, tmax=tmax_clip)
    
    return baseline_used


###################################################################
# TFR Computation and Processing
###################################################################

def run_tfr_morlet(
    epochs,
    freqs,
    n_cycles,
    *,
    decim: int = 4,
    picks: Union[str, list] = "eeg",
    workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
):
    if workers is None:
        if config is None:
            config = load_settings()
        workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
        workers = resolve_tfr_workers(workers_default=workers_default)
    
    return mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=decim,
        n_jobs=workers,
        picks=picks,
        verbose=False
    )


###################################################################
# TFR Data Extraction and Masking
###################################################################

def average_tfr_band(tfr_avg, fmin: float, fmax: float, tmin: float, tmax: float):
    freqs = np.asarray(tfr_avg.freqs)
    times = np.asarray(tfr_avg.times)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    t_mask = (times >= tmin) & (times < tmax)
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    sel = tfr_avg.data[:, f_mask, :][:, :, t_mask]
    return sel.mean(axis=(1, 2))


def extract_epochwise_channel_values(tfr_epochs, fmin: float, fmax: float, tmin: float, tmax: float):
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    fmask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    tmask = (times >= float(tmin)) & (times < float(tmax))
    if fmask.sum() == 0 or tmask.sum() == 0:
        return None
    data = np.asarray(tfr_epochs.data)[:, :, fmask, :][:, :, :, tmask]
    return data.mean(axis=(2, 3))


def effective_plateau_window(
    times: np.ndarray,
    requested: Tuple[float, float]
) -> Tuple[Optional[float], Optional[float], np.ndarray]:
    t_arr = np.asarray(times)
    tmin_req, tmax_req = float(requested[0]), float(requested[1])
    tmin_eff = float(max(t_arr.min(), tmin_req))
    tmax_eff = float(min(t_arr.max(), tmax_req))
    if not np.isfinite(tmin_eff) or not np.isfinite(tmax_eff) or tmax_eff <= tmin_eff:
        return None, None, np.zeros_like(t_arr, dtype=bool)
    tmask = (t_arr >= tmin_eff) & (t_arr < tmax_eff)
    return tmin_eff, tmax_eff, tmask


def band_time_masks(
    freqs: np.ndarray,
    times: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    f_arr = np.asarray(freqs)
    t_arr = np.asarray(times)
    fmask = (f_arr >= float(fmin)) & (f_arr <= float(fmax))
    tmask = (t_arr >= float(tmin)) & (t_arr < float(tmax))
    return fmask, tmask


def time_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return (times >= tmin) & (times < tmax)


def freq_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= fmax)


def find_tfr_path(subject: str, task: str, deriv_root: Path) -> Optional[Path]:
    primary_path = deriv_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
    if primary_path.exists():
        return primary_path
    
    eeg_dir = deriv_root / f"sub-{subject}" / "eeg"
    if eeg_dir.exists():
        candidates = sorted(eeg_dir.glob(f"sub-{subject}_task-{task}*_epo-tfr.h5"))
        if candidates:
            return candidates[0]
    
    subj_dir = deriv_root / f"sub-{subject}"
    if subj_dir.exists():
        candidates = sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*_epo-tfr.h5"))
        if candidates:
            return candidates[0]
    
    return None


###################################################################
# Subject-Level TFR Computation
###################################################################

def compute_subject_tfr(
    subject: str,
    task: str,
    freq_min: float,
    freq_max: float,
    n_freqs: int,
    n_cycles_factor: float,
    tfr_decim: int,
    tfr_picks: Union[str, list],
    strict_mode: bool = True,
    workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional["mne.time_frequency.EpochsTFR"], Optional[pd.DataFrame]]:
    config = load_settings()
    deriv_root = config.deriv_root
    epochs, events_df = load_epochs_for_analysis(
        subject, task, align="strict", preload=True, deriv_root=deriv_root, bids_root=config.bids_root, config=config, logger=logger
    )
    
    if epochs is None:
        if logger:
            logger.error(f"No cleaned epochs for sub-{subject}, task-{task}")
        return None, None
    
    if events_df is None:
        if logger:
            logger.warning("Events missing; contrasts will be skipped for this subject.")
    
    if events_df is not None and strict_mode:
        trial_cols = ["trial", "trial_number", "trial_index"]
        trial_col = next((c for c in trial_cols if c in events_df.columns), None)
        if trial_col is not None:
            trial_vals = pd.to_numeric(events_df[trial_col], errors="coerce")
            
            # Check if we have multi-run data (trial numbers reset per run)
            run_col = next((c for c in ["run_id", "run", "run_number"] if c in events_df.columns), None)
            
            if run_col is not None:
                # Multi-run: check duplicates within each run
                grouped = events_df.groupby(run_col)[trial_col]
                within_run_dups = grouped.apply(lambda x: pd.to_numeric(x, errors="coerce").duplicated().sum()).sum()
                
                if within_run_dups > 0:
                    if logger:
                        logger.error(
                            f"CRITICAL: Trial index column '{trial_col}' has {within_run_dups} duplicates within runs. "
                            f"This indicates a serious alignment issue."
                        )
                else:
                    if logger:
                        logger.debug(
                            f"Trial indices validated: {len(events_df)} trials across {events_df[run_col].nunique()} runs "
                            f"(trial numbers reset per run, which is expected)."
                        )
            else:
                # Single run: check global duplicates and monotonicity
                if trial_vals.duplicated().any():
                    n_dup = trial_vals.duplicated().sum()
                    if logger:
                        logger.error(
                            f"CRITICAL: Trial index column '{trial_col}' has {n_dup} duplicates. "
                            f"This indicates a serious alignment issue where multiple epochs map to the same trial."
                        )
                
                elif not trial_vals.is_monotonic_increasing and not trial_vals.is_monotonic_decreasing:
                    n_gaps = (trial_vals.diff() != 1).sum() - 1  # -1 for first NaN
                    if logger:
                        logger.debug(
                            f"Trial index column '{trial_col}' is non-monotonic ({n_gaps} gaps). "
                            f"This is expected when trials were dropped during preprocessing."
                        )
    
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, min_cycles=3.0)
    power = run_tfr_morlet(
        epochs,
        freqs,
        n_cycles,
        decim=tfr_decim,
        picks=tfr_picks,
        workers=workers,
        logger=logger,
    )
    
    return power, events_df


###################################################################
# Group ROI Helpers
###################################################################

def collect_group_temperatures(
    events_by_subj: List[Optional[pd.DataFrame]],
    temperature_columns: List[str],
) -> List[float]:
    temps = set()
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = None
        for c in temperature_columns:
            if c in ev.columns:
                tcol = c
                break
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1).dropna().unique()
        for v in vals:
            temps.add(float(v))
    return sorted(temps)


def avg_alltrials_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> "mne.time_frequency.AverageTFR":
    tfr_all = power.copy()
    apply_baseline_and_crop(tfr_all, baseline=baseline, mode="logratio", logger=logger)
    return tfr_all.average()


def avg_by_mask_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    mask: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    t = power.copy()[mask]
    apply_baseline_and_crop(t, baseline=baseline, mode="logratio", logger=logger)
    return t.average()


def align_avg_tfrs(
    tfr_list: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Info], Optional[np.ndarray]]:
    if not tfr_list:
        return None, None
    tfr_list = [t for t in tfr_list if t is not None]
    if not tfr_list:
        return None, None
    base = tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    keep: List[Tuple[str, "mne.time_frequency.AverageTFR"]] = [("S0", base)]
    for i, tfr in enumerate(tfr_list[1:], start=1):
        ok = np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs)
        if not ok:
            if logger:
                logger.warning(f"Skipping subject {i}: times/freqs mismatch for group alignment")
            continue
        keep.append((f"S{i}", tfr))
    if len(keep) == 0:
        return None, None
    ch_sets = [set(t.info["ch_names"]) for _, t in keep]
    common = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common) == 0:
        if logger:
            logger.warning("No common channels across subjects; cannot align")
        return None, None
    arrs = []
    for tag, t in keep:
        idxs = [t.info["ch_names"].index(ch) for ch in common]
        arrs.append(np.asarray(t.data)[idxs, :, :])
    data = np.stack(arrs, axis=0)
    pick_inds = [base_chs.index(ch) for ch in common]
    info_common = mne.pick_info(base.info, pick_inds)
    return info_common, data


def align_paired_avg_tfrs(
    list_a: List["mne.time_frequency.AverageTFR"],
    list_b: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Info], Optional[np.ndarray], Optional[np.ndarray]]:
    if not list_a or not list_b:
        return None, None, None
    n = min(len(list_a), len(list_b))
    pairs = []
    for i in range(n):
        a = list_a[i]
        b = list_b[i]
        if a is None or b is None:
            continue
        pairs.append((a, b))
    if not pairs:
        return None, None, None
    base_a, base_b = pairs[0]
    base_times = np.asarray(base_a.times)
    base_freqs = np.asarray(base_a.freqs)
    keep: list[tuple["mne.time_frequency.AverageTFR", "mne.time_frequency.AverageTFR"]] = []
    for idx, (a, b) in enumerate(pairs):
        ok = (
            np.allclose(a.times, base_times) and np.allclose(a.freqs, base_freqs)
            and np.allclose(b.times, base_times) and np.allclose(b.freqs, base_freqs)
        )
        if not ok:
            if logger:
                logger.warning(f"Skipping subject pair {idx}: times/freqs mismatch for paired alignment")
            continue
        keep.append((a, b))
    if not keep:
        return None, None, None
    per_pair_common = []
    for a, b in keep:
        per_pair_common.append(set(a.info["ch_names"]) & set(b.info["ch_names"]))
    common = list(sorted(set.intersection(*per_pair_common))) if per_pair_common else []
    if len(common) == 0:
        if logger:
            logger.warning("Paired alignment: no common channels across retained pairs")
        return None, None, None
    
    data_a = []
    data_b = []
    for a, b in keep:
        idx_a = [a.info["ch_names"].index(ch) for ch in common]
        idx_b = [b.info["ch_names"].index(ch) for ch in common]
        data_a.append(np.asarray(a.data)[idx_a, :, :])
        data_b.append(np.asarray(b.data)[idx_b, :, :])
    data_a_arr = np.stack(data_a, axis=0)
    data_b_arr = np.stack(data_b, axis=0)
    pick_inds = [base_a.info["ch_names"].index(ch) for ch in common]
    info_common = mne.pick_info(base_a.info, pick_inds)
    return info_common, data_a_arr, data_b_arr


__all__ = [
    "canonicalize_ch_name",
    "find_roi_channels",
    "build_rois_from_info",
    "get_rois",
    "compute_adaptive_n_cycles",
    "log_tfr_resolution",
    "validate_tfr_parameters",
    "resolve_tfr_workers",
    "get_bands_for_tfr",
    "read_tfr_average_with_logratio",
    "save_tfr_with_sidecar",
    "validate_baseline_window",
    "validate_baseline_indices",
    "apply_baseline_safe",
    "apply_baseline_and_crop",
    "log_baseline_qc",
    "run_tfr_morlet",
    "average_tfr_band",
    "extract_epochwise_channel_values",
    "effective_plateau_window",
    "band_time_masks",
    "time_mask",
    "freq_mask",
    "compute_subject_tfr",
    "collect_group_temperatures",
    "avg_alltrials_to_avg_tfr",
    "avg_by_mask_to_avg_tfr",
    "align_avg_tfrs",
    "align_paired_avg_tfrs",
]
