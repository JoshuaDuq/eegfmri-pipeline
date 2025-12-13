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

from eeg_pipeline.types import PrecomputedData, PrecomputedQC
from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec
from eeg_pipeline.utils.analysis.spectral import compute_band_data, compute_psd
from eeg_pipeline.utils.analysis.signal_metrics import compute_gfp
from eeg_pipeline.utils.config.loader import get_frequency_bands


def _compute_single_band(data: np.ndarray, sfreq: float, band_name: str, fmin: float, fmax: float):
    """Compute band data for a single band (parallel worker)."""
    try:
        # Note: We don't pass logger to parallel worker to avoid pickling issues
        bd = compute_band_data(data, sfreq, band_name, fmin, fmax, logger=None)
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
    n_plateau_windows: int = 5,
    windows_spec: Any = None,
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
    n_plateau_windows : int
        Number of temporal windows in plateau period
        
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
    
    data = epochs.get_data(picks=picks)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    
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
        if spec is None:
            spec = TimeWindowSpec(times=times, config=config, sampling_rate=sfreq, logger=logger)
        precomputed.windows = time_windows_from_spec(
            spec,
            n_plateau_windows=n_plateau_windows,
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
    
    # Compute band data (optionally in parallel)
    if compute_bands and bands:
        freq_bands = get_frequency_bands(config)
        
        band_defs = [(band, freq_bands[band]) for band in bands if band in freq_bands]

        if not band_defs and logger:
            logger.warning("No valid bands found in config; skipping band precomputation.")
        else:
            n_jobs_bands = int(config.get("feature_engineering.parallel.n_jobs_bands", 1))

            results = []
            # Handle parallel execution (n_jobs != 1, including -1 for all CPUs)
            if n_jobs_bands != 1:
                try:
                    from joblib import Parallel, delayed

                    results = Parallel(n_jobs=n_jobs_bands, prefer="processes")(
                        delayed(_compute_single_band)(data, sfreq, band, fmin, fmax) for band, (fmin, fmax) in band_defs
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    if logger:
                        logger.warning(
                            "Parallel band computation failed (%s); falling back to sequential.", exc
                        )
                    n_jobs_bands = 1

            # Fallback or configured sequential
            if n_jobs_bands == 1:
                results = [_compute_single_band(data, sfreq, band, fmin, fmax) for band, (fmin, fmax) in band_defs]

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
