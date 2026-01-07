"""
Precomputation Module
=====================

Functions for precomputing expensive intermediate data (bands, PSD, GFP)
shared across multiple feature extractors.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import mne

from eeg_pipeline.types import PrecomputedData, PrecomputedQC, TimeWindows
from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec
from eeg_pipeline.utils.analysis.spectral import compute_band_data, compute_psd
from eeg_pipeline.utils.analysis.signal_metrics import compute_gfp
from eeg_pipeline.utils.config.loader import get_frequency_bands


def _compute_single_band(
    data: np.ndarray,
    sfreq: float,
    band_name: str,
    fmin: float,
    fmax: float,
    *,
    pad_sec: float,
    pad_cycles: float,
):
    """Compute band data for a single band (parallel worker)."""
    try:
        # Note: We don't pass logger to parallel worker to avoid pickling issues
        bd = compute_band_data(
            data,
            sfreq,
            band_name,
            fmin,
            fmax,
            logger=None,
            pad_sec=pad_sec,
            pad_cycles=pad_cycles,
        )
        if bd is None:
            return None
        gfp_band = compute_gfp(bd.filtered)
        band_power = bd.power
        qc_entry = {}
        if band_power.size > 0:
            qc_entry = {
                "finite_fraction": float(np.isfinite(band_power).sum() / band_power.size),
                "median_power": float(np.nanmedian(band_power)),
                "fmin": fmin,
                "fmax": fmax,
            }
        return band_name, bd, gfp_band, qc_entry
    except Exception as exc:
        return None, str(exc)


def precompute_data(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    compute_bands: bool = True,
    compute_psd_data: bool = True,
    windows_spec: Any = None,
    frequency_bands_override: Any = None,
) -> PrecomputedData:
    """
    Precompute all intermediate data needed by feature extraction modules.
    
    Call this once at the start, then pass the result to feature functions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    bands : List[str]
        Frequency bands to precompute
    config : Any
        Configuration object
    logger : Any
        Logger instance
    compute_bands : bool
        Whether to precompute band-filtered data
    compute_psd : bool
        Whether to precompute PSD
        
    Returns
    -------
    PrecomputedData
        Container with all precomputed data
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    
    if len(picks) == 0:
        logger.warning("No EEG channels available")
        return PrecomputedData(
            data=np.array([]),
            times=epochs.times,
            sfreq=float(epochs.info["sfreq"]),
            ch_names=[],
            picks=picks,
            config=config,
            logger=logger,
        )
    
    spatial_transform = str(config.get("feature_engineering.spatial_transform", "none")).strip().lower()
    if spatial_transform not in {"none", "csd", "laplacian"}:
        spatial_transform = "none"

    epochs_picked = epochs.copy().pick(picks)
    if spatial_transform in {"csd", "laplacian"}:
        try:
            lambda2 = float(config.get("feature_engineering.spatial_transform_params.lambda2", 1e-5))
            stiffness = float(config.get("feature_engineering.spatial_transform_params.stiffness", 4.0))
        except Exception:
            lambda2 = 1e-5
            stiffness = 4.0

        try:
            epochs_picked = mne.preprocessing.compute_current_source_density(
                epochs_picked,
                lambda2=lambda2,
                stiffness=stiffness,
                verbose=False,
            )
            if logger:
                logger.info(
                    "Applied spatial transform=%s (CSD) to epochs for feature precomputation (lambda2=%s, stiffness=%s).",
                    spatial_transform,
                    str(lambda2),
                    str(stiffness),
                )
        except Exception as exc:
            if logger:
                logger.warning(
                    "Failed to apply spatial transform=%s; proceeding without it (%s).",
                    spatial_transform,
                    exc,
                )

    data = epochs_picked.get_data()
    times = epochs_picked.times
    sfreq = float(epochs_picked.info["sfreq"])
    ch_names = list(epochs_picked.ch_names)
    
    # Create container
    precomputed = PrecomputedData(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=ch_names,
        picks=picks,
        config=config,
        logger=logger,
    )

    # Lightweight QC for downstream inspection
    if data.size > 0:
        precomputed.qc.data_finite_fraction = float(np.isfinite(data).sum() / data.size)
        precomputed.qc.n_epochs = data.shape[0]
        precomputed.qc.n_channels = data.shape[1]
        precomputed.qc.n_times = data.shape[2]
        precomputed.qc.sfreq = sfreq
    
    # Compute time windows
    try:
        spec = windows_spec
        if isinstance(spec, TimeWindows):
            precomputed.windows = spec
        else:
            if spec is None:
                spec = TimeWindowSpec(times=times, config=config, sampling_rate=sfreq, logger=logger)
            precomputed.windows = time_windows_from_spec(
                spec,
                logger=logger,
                strict=True,
            )

        precomputed.qc.time_windows = {
            "baseline_samples": int(np.sum(precomputed.windows.baseline_mask)),
            "active_samples": int(np.sum(precomputed.windows.active_mask)),
            "baseline_range": getattr(precomputed.windows, "baseline_range", (np.nan, np.nan)),
            "active_range": getattr(precomputed.windows, "active_range", (np.nan, np.nan)),
            "clamped": getattr(precomputed.windows, "clamped", False),
            "errors": list(getattr(precomputed.windows, "errors", [])),
        }
        if logger and precomputed.windows:
            logger.info(
                "Using time windows baseline=%s, active=%s (clamped=%s)",
                getattr(precomputed.windows, "baseline_range", None),
                getattr(precomputed.windows, "active_range", None),
                getattr(precomputed.windows, "clamped", False),
            )
    except ValueError as exc:
        precomputed.windows = None
        precomputed.qc.time_windows = {
            "baseline_samples": 0,
            "active_samples": 0,
            "baseline_range": (np.nan, np.nan),
            "active_range": (np.nan, np.nan),
            "clamped": False,
            "errors": [str(exc)],
        }
        precomputed.qc.errors.append(f"time_windows: {exc}")
        if logger:
            logger.error("Time window computation failed; downstream features will be skipped: %s", exc)
    
    # Compute GFP
    precomputed.gfp = compute_gfp(data)
    if precomputed.gfp.size > 0:
        finite_fraction = float(np.isfinite(precomputed.gfp).sum() / precomputed.gfp.size)
        precomputed.qc.gfp = {
            "finite_fraction": finite_fraction,
            "median": float(np.nanmedian(precomputed.gfp)),
        }
    
    # Optionally derive individualized band definitions from IAF (baseline PSD).
    use_iaf = bool(config.get("feature_engineering.bands.use_iaf", False)) if hasattr(config, "get") else False
    freq_bands_base = frequency_bands_override or get_frequency_bands(config)
    freq_bands_use = dict(freq_bands_base) if isinstance(freq_bands_base, dict) else {}

    if use_iaf and precomputed.windows is not None:
        try:
            from scipy.signal import find_peaks
            from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
            from eeg_pipeline.utils.analysis.channels import build_roi_map

            iaf_cfg = config.get("feature_engineering.bands", {}) if hasattr(config, "get") else {}
            alpha_range = iaf_cfg.get("iaf_search_range_hz", [7.0, 13.0])
            alpha_fmin = float(alpha_range[0])
            alpha_fmax = float(alpha_range[1])
            prom = float(iaf_cfg.get("iaf_min_prominence", 0.05))

            rois = iaf_cfg.get("iaf_rois", ["ParOccipital_Midline", "ParOccipital_Ipsi_L", "ParOccipital_Contra_R"])
            roi_defs = get_roi_definitions(config)
            roi_map = build_roi_map(ch_names, roi_defs) if roi_defs else {}
            roi_idxs: List[int] = []
            if isinstance(rois, (list, tuple)) and roi_map:
                for roi in rois:
                    idxs = roi_map.get(str(roi), [])
                    roi_idxs.extend(list(idxs))
            roi_idxs = sorted(set(int(i) for i in roi_idxs if i is not None))

            baseline_mask = getattr(precomputed.windows, "baseline_mask", None)
            if baseline_mask is not None and np.any(baseline_mask):
                x = data[:, :, baseline_mask]
            else:
                x = data

            psds, freqs = mne.time_frequency.psd_array_multitaper(
                x,
                sfreq=sfreq,
                fmin=max(1.0, alpha_fmin - 4.0),
                fmax=min(40.0, sfreq / 2.0 - 0.5),
                adaptive=True,
                normalization="full",
                verbose=False,
            )
            psds = np.asarray(psds, dtype=float)
            freqs = np.asarray(freqs, dtype=float)
            if psds.ndim == 3 and freqs.size > 0:
                idx_use = roi_idxs if roi_idxs else list(range(psds.shape[1]))
                mean_psd = np.nanmean(psds[:, idx_use, :], axis=(0, 1))
                log_f = np.log10(np.maximum(freqs, 1e-6))
                log_p = np.log10(np.maximum(mean_psd, 1e-20))
                # Remove 1/f trend with robust linear fit (broadband 2-40).
                fit_mask = (freqs >= 2.0) & (freqs <= 40.0) & np.isfinite(log_p)
                if np.sum(fit_mask) >= 10:
                    slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
                    resid = log_p - (intercept + slope * log_f)
                else:
                    resid = log_p

                a_mask = (freqs >= alpha_fmin) & (freqs <= alpha_fmax) & np.isfinite(resid)
                iaf = np.nan
                if np.any(a_mask):
                    y = resid[a_mask]
                    peaks, props = find_peaks(y, prominence=prom)
                    if peaks.size:
                        best = int(peaks[np.argmax(props.get("prominences", np.ones_like(peaks)))])
                        iaf = float(freqs[a_mask][best])
                    else:
                        y_pos = np.maximum(y, 0.0)
                        denom = float(np.sum(y_pos))
                        if denom > 0:
                            iaf = float(np.sum(freqs[a_mask] * y_pos) / denom)

                if np.isfinite(iaf):
                    precomputed.qc.time_windows["iaf_hz"] = float(iaf)
                    width = float(iaf_cfg.get("alpha_width_hz", 2.0))
                    alpha_min = max(6.0, iaf - width)
                    alpha_max = min(14.0, iaf + width)
                    freq_bands_use["alpha"] = [float(alpha_min), float(alpha_max)]
                    # Theta ends at alpha_min; beta starts at alpha_max.
                    freq_bands_use.setdefault("theta", [4.0, 8.0])
                    freq_bands_use.setdefault("beta", [13.0, 30.0])
                    freq_bands_use["theta"] = [max(3.0, float(iaf - 6.0)), max(4.0, float(alpha_min))]
                    beta_min_default = float(freq_bands_use["beta"][0]) if isinstance(freq_bands_use.get("beta"), (list, tuple)) and len(freq_bands_use["beta"]) >= 2 else 13.0
                    beta_max_default = float(freq_bands_use["beta"][1]) if isinstance(freq_bands_use.get("beta"), (list, tuple)) and len(freq_bands_use["beta"]) >= 2 else 30.0
                    freq_bands_use["beta"] = [max(beta_min_default, float(alpha_max)), beta_max_default]
        except Exception as exc:
            if logger:
                logger.warning("IAF estimation failed; using config bands (%s).", exc)

    precomputed.frequency_bands = freq_bands_use or None

    # Compute band data (optionally in parallel)
    if compute_bands and bands:
        freq_bands = freq_bands_use or get_frequency_bands(config)
        
        band_defs = [(band, freq_bands[band]) for band in bands if band in freq_bands]

        if not band_defs and logger:
            logger.warning("No valid bands found in config; skipping band precomputation.")
        else:
            n_jobs_bands = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
            pad_sec = float(config.get("feature_engineering.band_envelope.pad_sec", 0.5))
            pad_cycles = float(config.get("feature_engineering.band_envelope.pad_cycles", 3.0))

            results = []
            # Handle parallel execution (n_jobs != 1, including -1 for all CPUs)
            if n_jobs_bands != 1:
                try:
                    from joblib import Parallel, delayed

                    results = Parallel(n_jobs=n_jobs_bands, prefer="processes")(
                        delayed(_compute_single_band)(
                            data,
                            sfreq,
                            band,
                            fmin,
                            fmax,
                            pad_sec=pad_sec,
                            pad_cycles=pad_cycles,
                        )
                        for band, (fmin, fmax) in band_defs
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    if logger:
                        logger.warning(
                            "Parallel band computation failed (%s); falling back to sequential.", exc
                        )
                    n_jobs_bands = 1

            # Fallback or configured sequential
            if n_jobs_bands == 1:
                results = [
                    _compute_single_band(
                        data,
                        sfreq,
                        band,
                        fmin,
                        fmax,
                        pad_sec=pad_sec,
                        pad_cycles=pad_cycles,
                    )
                    for band, (fmin, fmax) in band_defs
                ]

            for res in results:
                if res is None:
                    continue
                if len(res) == 2 and res[0] is None:
                    # Error case
                    if logger:
                        logger.warning(f"Band computation failed: {res[1]}")
                    continue
                    
                band_name, bd, gfp_band, qc_entry = res
                precomputed.band_data[band_name] = bd
                precomputed.gfp_band[band_name] = gfp_band
                if qc_entry:
                    precomputed.qc.bands[band_name] = qc_entry

            logger.info(f"Precomputed band data for: {list(precomputed.band_data.keys())}")
    
    # Compute PSD
    if compute_psd_data:
        psd_input = data
        psd_window = "full"
        try:
            baseline_mask = getattr(precomputed.windows, "baseline_mask", None) if precomputed.windows is not None else None
            if baseline_mask is not None and isinstance(baseline_mask, np.ndarray) and np.any(baseline_mask):
                psd_input = data[:, :, baseline_mask]
                psd_window = "baseline"
        except Exception:
            psd_input = data
            psd_window = "full"

        precomputed.psd_data = compute_psd(psd_input, sfreq, config=config, logger=logger)
        if precomputed.psd_data is not None:
            logger.info(f"Precomputed PSD: {len(precomputed.psd_data.freqs)} freq bins")
            psd_arr = precomputed.psd_data.psd
            precomputed.qc.psd = {
                "n_freq_bins": int(len(precomputed.psd_data.freqs)),
                "finite_fraction": float(np.isfinite(psd_arr).sum() / psd_arr.size),
                "window": psd_window,
                "freq_range": (
                    float(precomputed.psd_data.freqs[0]),
                    float(precomputed.psd_data.freqs[-1]),
                )
                if len(precomputed.psd_data.freqs) > 1
                else (np.nan, np.nan),
            }
    
    return precomputed
