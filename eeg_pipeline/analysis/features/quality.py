"""
Feature Quality Assessment (Signal QC)
=======================================

Quality metrics for evaluating signal integrity per trial:
- Variance
- SNR estimates (Low/High frequency ratio)
- Muscle artifact ratio (High Gamma / Total)
- Peak-to-Peak Amplitude
- Finite Fraction (Missing data)

Computed on 'baseline' and 'active' windows.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import EPSILON_STD


MIN_SAMPLES_FOR_SPECTRAL = 10
DEFAULT_N_FFT = 256
DEFAULT_FMIN = 1.0
DEFAULT_FMAX = 100.0
DEFAULT_LINE_NOISE_FREQ = 50.0
DEFAULT_LINE_NOISE_WIDTH = 1.0
DEFAULT_LINE_NOISE_HARMONICS = 3
DEFAULT_SNR_SIGNAL_BAND = [1.0, 30.0]
DEFAULT_SNR_NOISE_BAND = [40.0, 80.0]
DEFAULT_MUSCLE_BAND = [30.0, 80.0]
SNR_DB_MULTIPLIER = 10


def _extract_quality_config(config: Any) -> Dict[str, Any]:
    """Extract quality feature configuration from config object."""
    if config is None or not hasattr(config, "get"):
        return {}
    quality_cfg = dict(config.get("feature_engineering.quality", {}) or {})
    if "line_noise_freqs" not in quality_cfg:
        line_freq = config.get("preprocessing.line_freq", DEFAULT_LINE_NOISE_FREQ)
        try:
            line_freq = float(line_freq)
            if np.isfinite(line_freq) and line_freq > 0:
                quality_cfg["line_noise_freqs"] = [line_freq]
        except (TypeError, ValueError):
            pass
    return quality_cfg


def _get_psd_method(config: Dict[str, Any]) -> str:
    """Get PSD computation method, defaulting to welch."""
    method = str(config.get("psd_method", "welch")).strip().lower()
    if method not in {"welch", "multitaper"}:
        raise ValueError(f"feature_engineering.quality.psd_method must be 'welch' or 'multitaper' (got '{method}').")
    return method


def _get_frequency_range(config: Dict[str, Any], sfreq: float) -> Tuple[float, float]:
    """Get frequency range for PSD computation."""
    fmin = float(config.get("fmin", DEFAULT_FMIN))
    fmax = float(config.get("fmax", min(DEFAULT_FMAX, float(sfreq) / 2.0 - 0.5)))
    if not (np.isfinite(fmin) and np.isfinite(fmax) and fmax > fmin and fmin >= 0):
        raise ValueError(f"Invalid feature_engineering.quality fmin/fmax: fmin={fmin}, fmax={fmax}.")
    return fmin, fmax


def _get_line_noise_parameters(config: Dict[str, Any]) -> Tuple[List[float], float, int]:
    """Extract line noise exclusion parameters."""
    line_freqs_raw = config.get("line_noise_freqs", [DEFAULT_LINE_NOISE_FREQ])
    if line_freqs_raw is None:
        line_freqs_raw = []
    if not isinstance(line_freqs_raw, (list, tuple)):
        raise TypeError(
            "feature_engineering.quality.line_noise_freqs must be a list/tuple of numbers "
            f"(got {type(line_freqs_raw).__name__})."
        )
    line_freqs = [float(f) for f in line_freqs_raw]
    
    width = float(config.get("line_noise_width_hz", DEFAULT_LINE_NOISE_WIDTH))
    n_harmonics = int(config.get("line_noise_harmonics", DEFAULT_LINE_NOISE_HARMONICS))
    if not (np.isfinite(width) and width > 0):
        raise ValueError(f"feature_engineering.quality.line_noise_width_hz must be > 0 (got {width}).")
    if n_harmonics <= 0:
        raise ValueError(f"feature_engineering.quality.line_noise_harmonics must be > 0 (got {n_harmonics}).")
    
    return line_freqs, width, n_harmonics


def _exclude_line_noise_frequencies(
    freqs: np.ndarray,
    psds: np.ndarray,
    line_freqs: List[float],
    width: float,
    n_harmonics: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exclude line noise frequencies and their harmonics from PSD."""
    if not line_freqs or width <= 0 or n_harmonics <= 0:
        return freqs, psds
    
    keep_mask = np.ones_like(freqs, dtype=bool)
    
    for base_freq in line_freqs:
        if not np.isfinite(base_freq) or base_freq <= 0:
            continue
        for harmonic in range(1, n_harmonics + 1):
            harmonic_freq = base_freq * harmonic
            lower_bound = harmonic_freq - width
            upper_bound = harmonic_freq + width
            keep_mask &= ~((freqs >= lower_bound) & (freqs <= upper_bound))
    
    if np.any(~keep_mask):
        return freqs[keep_mask], psds[:, keep_mask]
    
    return freqs, psds


def _compute_psd(
    data: np.ndarray,
    sfreq: float,
    config: Dict[str, Any],
    *,
    logger: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using specified method."""
    method = _get_psd_method(config)
    fmin, fmax = _get_frequency_range(config, sfreq)
    n_times = int(data.shape[1])
    default_n_per_seg = min(int(float(sfreq) * 2.0), n_times)
    n_per_seg = int(config.get("n_per_seg", default_n_per_seg))
    n_per_seg = max(2, min(n_per_seg, n_times))
    n_fft = int(config.get("n_fft", max(DEFAULT_N_FFT, n_per_seg)))
    n_fft = max(2, min(n_fft, n_times))
    
    if method == "multitaper":
        multitaper_adaptive = bool(config.get("multitaper_adaptive", False))
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            data,
            sfreq=float(sfreq),
            fmin=fmin,
            fmax=fmax,
            adaptive=multitaper_adaptive,
            normalization="full",
            verbose=False,
        )
    else:
        n_overlap_raw = config.get("n_overlap", n_per_seg // 2)
        try:
            n_overlap_raw = int(n_overlap_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "feature_engineering.quality.n_overlap must be an int (samples) or None "
                f"(got {type(n_overlap_raw).__name__})."
            ) from exc

        # Fail-safe: MNE requires n_overlap < n_per_seg (and <= n_fft).
        # User request: warn and continue rather than crashing quality extraction.
        n_overlap = max(0, min(n_overlap_raw, n_per_seg - 1, n_fft - 1))
        if logger is not None and n_overlap != n_overlap_raw:
            logger.warning(
                "Quality PSD: clamped Welch n_overlap from %d → %d to satisfy "
                "n_overlap < n_per_seg (%d) and n_overlap < n_fft (%d).",
                int(n_overlap_raw),
                int(n_overlap),
                int(n_per_seg),
                int(n_fft),
            )

        try:
            psds, freqs = mne.time_frequency.psd_array_welch(
                data,
                sfreq=float(sfreq),
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                n_per_seg=n_per_seg,
                n_overlap=n_overlap,
                verbose=False,
            )
        except ValueError as exc:
            msg = str(exc).lower()
            if (
                "n_overlap cannot be greater than n_per_seg" in msg
                or "n_overlap" in msg and "n_per_seg" in msg
            ):
                if logger is not None:
                    logger.warning(
                        "Quality PSD: Welch parameter error (%s). "
                        "Skipping spectral quality metrics for this segment.",
                        str(exc).strip(),
                    )
                raise
            raise
    
    freqs = np.asarray(freqs, dtype=float)
    psds = np.asarray(psds, dtype=float)
    
    exclude_line_noise = bool(config.get("exclude_line_noise", True))
    if exclude_line_noise:
        line_freqs, width, n_harmonics = _get_line_noise_parameters(config)
        freqs, psds = _exclude_line_noise_frequencies(
            freqs, psds, line_freqs, width, n_harmonics
        )
    
    return psds, freqs


def _compute_snr_from_psd(
    psds: np.ndarray,
    freqs: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Compute SNR as ratio of signal band to noise band power."""
    signal_band = config.get("snr_signal_band", DEFAULT_SNR_SIGNAL_BAND)
    noise_band = config.get("snr_noise_band", DEFAULT_SNR_NOISE_BAND)
    
    try:
        signal_low, signal_high = float(signal_band[0]), float(signal_band[1])
        noise_low, noise_high = float(noise_band[0]), float(noise_band[1])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Invalid snr_signal_band/snr_noise_band: {signal_band!r} / {noise_band!r}"
        ) from exc
    
    signal_mask = (freqs >= signal_low) & (freqs <= signal_high)
    noise_mask = (freqs >= noise_low) & (freqs <= noise_high)
    df = np.gradient(freqs) if freqs.size > 1 else np.ones_like(freqs, dtype=float)

    signal_bandwidth = float(np.sum(df[signal_mask])) if np.any(signal_mask) else np.nan
    noise_bandwidth = float(np.sum(df[noise_mask])) if np.any(noise_mask) else np.nan
    signal_power = np.sum(psds[:, signal_mask] * df[signal_mask], axis=1) if np.any(signal_mask) else np.full(psds.shape[0], np.nan)
    noise_power = np.sum(psds[:, noise_mask] * df[noise_mask], axis=1) if np.any(noise_mask) else np.full(psds.shape[0], np.nan)

    signal_density = signal_power / max(signal_bandwidth, EPSILON_STD) if np.isfinite(signal_bandwidth) else np.full(psds.shape[0], np.nan)
    noise_density = noise_power / max(noise_bandwidth, EPSILON_STD) if np.isfinite(noise_bandwidth) else np.full(psds.shape[0], np.nan)

    snr_linear = signal_density / (noise_density + EPSILON_STD)
    snr_db = SNR_DB_MULTIPLIER * np.log10(snr_linear)
    
    return snr_db


def _compute_muscle_ratio_from_psd(
    psds: np.ndarray,
    freqs: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Compute muscle artifact ratio as high-frequency power fraction."""
    muscle_band = config.get("muscle_band", DEFAULT_MUSCLE_BAND)
    
    try:
        muscle_low, muscle_high = float(muscle_band[0]), float(muscle_band[1])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"Invalid muscle_band: {muscle_band!r}") from exc
    
    muscle_mask = (freqs >= muscle_low) & (freqs <= muscle_high)
    df = np.gradient(freqs) if freqs.size > 1 else np.ones_like(freqs, dtype=float)
    muscle_power = np.sum(psds[:, muscle_mask] * df[muscle_mask], axis=1) if np.any(muscle_mask) else np.full(psds.shape[0], np.nan)
    total_power = np.sum(psds * df[None, :], axis=1)
    
    muscle_ratio = muscle_power / (total_power + EPSILON_STD)
    return muscle_ratio


def _compute_basic_metrics(data: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute basic time-domain quality metrics."""
    variance = np.var(data, axis=1)
    peak_to_peak = np.ptp(data, axis=1)
    finite_fraction = np.mean(np.isfinite(data), axis=1)
    
    return {
        "variance": variance,
        "ptp": peak_to_peak,
        "finite": finite_fraction,
    }


def _compute_spectral_metrics(
    data: np.ndarray,
    sfreq: float,
    config: Dict[str, Any],
    *,
    logger: Any = None,
) -> Dict[str, np.ndarray]:
    """Compute spectral quality metrics from PSD."""
    try:
        psds, freqs = _compute_psd(data, sfreq, config, logger=logger)
    except ValueError as exc:
        # User request: warn and continue, do not crash feature extraction.
        # We keep time-domain quality metrics and mark spectral metrics as NaN.
        if logger is not None:
            logger.warning(
                "Quality: PSD computation failed (%s). Setting SNR/muscle metrics to NaN.",
                str(exc).strip(),
            )
        n_channels = int(data.shape[0])
        return {
            "snr": np.full(n_channels, np.nan),
            "muscle": np.full(n_channels, np.nan),
        }
    snr = _compute_snr_from_psd(psds, freqs, config)
    muscle_ratio = _compute_muscle_ratio_from_psd(psds, freqs, config)
    return {"snr": snr, "muscle": muscle_ratio}


def _compute_signal_metrics(
    data: np.ndarray,
    sfreq: float,
    config: Any = None,
    *,
    logger: Any = None,
) -> Dict[str, np.ndarray]:
    """Compute all quality metrics for signal data.
    
    Args:
        data: Channel data array of shape (n_channels, n_times)
        sfreq: Sampling frequency in Hz
        config: Configuration object
        
    Returns:
        Dictionary with metric names as keys and per-channel arrays as values
    """
    metrics = _compute_basic_metrics(data)
    
    n_times = data.shape[1]
    if n_times < MIN_SAMPLES_FOR_SPECTRAL:
        n_channels = data.shape[0]
        metrics["snr"] = np.full(n_channels, np.nan)
        metrics["muscle"] = np.full(n_channels, np.nan)
        return metrics
    
    quality_config = _extract_quality_config(config)
    spectral_metrics = _compute_spectral_metrics(data, sfreq, quality_config, logger=logger)
    metrics.update(spectral_metrics)
    
    return metrics


def _store_metric_values(
    results: Dict[str, List[float]],
    metrics: Dict[str, np.ndarray],
    segment: str,
    channel_names: List[str],
    epoch_idx: int,
    n_epochs: int,
) -> None:
    """Store metric values for per-channel and global aggregations."""
    for metric_name, channel_values in metrics.items():
        for channel_idx, channel_name in enumerate(channel_names):
            column_name = NamingSchema.build(
                "quality", segment, "broadband", "ch", metric_name, channel=channel_name
            )
            if column_name not in results:
                results[column_name] = [np.nan] * n_epochs
            results[column_name][epoch_idx] = channel_values[channel_idx]
        
        global_value = np.nan
        if np.isfinite(channel_values).any():
            global_value = float(np.nanmean(channel_values))
        global_column = NamingSchema.build(
            "quality", segment, "broadband", "global", metric_name
        )
        if global_column not in results:
            results[global_column] = [np.nan] * n_epochs
        results[global_column][epoch_idx] = global_value


def extract_quality_features(
    ctx: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract quality features from epochs for baseline and active segments.
    
    Args:
        ctx: FeatureContext with epochs, windows, and optional config
        
    Returns:
        Tuple of (DataFrame with quality metrics, list of column names)
    """
    epochs = ctx.epochs
    picks, channel_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        raise ValueError("Quality: no EEG channels available.")
    
    full_data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]
    n_epochs = len(full_data)
    config = getattr(ctx, "config", None)
    
    results = {}
    windows = ctx.windows
    target_name = getattr(ctx, "name", None)
    logger = getattr(ctx, "logger", None)
    
    config is not None and hasattr(config, "get")
    
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            masks = {target_name: mask}
        else:
            if logger:
                logger.error(
                    "Quality: targeted window '%s' has no valid mask; skipping.",
                    target_name,
                )
            return pd.DataFrame(), []
    else:
        masks = get_segment_masks(epochs.times, windows, config)
    
    if not masks:
        if logger:
            logger.error("Quality: no valid time window masks available; skipping.")
        return pd.DataFrame(), []

    for segment, mask in masks.items():
        if not np.any(mask):
            continue
        
        segment_data = full_data[..., mask]
        
        for epoch_idx in range(n_epochs):
            epoch_data = segment_data[epoch_idx]
            metrics = _compute_signal_metrics(epoch_data, sfreq, config, logger=logger)
            _store_metric_values(
                results, metrics, segment, channel_names, epoch_idx, n_epochs
            )

    if not results:
        if logger:
            logger.error("Quality: no metrics were produced; skipping.")
        return pd.DataFrame(), []
    
    df = pd.DataFrame(results)
    return df, list(df.columns)


def compute_trial_quality_metrics(
    epochs: mne.Epochs,
    config: Any = None,
) -> pd.DataFrame:
    """Compute trial-level quality metrics from epochs.
    
    This is a standalone wrapper for use in pipelines that don't have
    a full FeatureContext. For full feature extraction, use extract_quality_features.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs
    config : Any
        Configuration object
        
    Returns
    -------
    pd.DataFrame
        Quality metrics per trial with global aggregations
    """
    picks, _ = pick_eeg_channels(epochs)
    if len(picks) == 0:
        raise ValueError("Quality: no EEG channels available.")
    
    full_data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]
    n_epochs = len(full_data)
    
    results = {}
    
    for epoch_idx in range(n_epochs):
        epoch_data = full_data[epoch_idx]
        metrics = _compute_signal_metrics(epoch_data, sfreq, config)
        
        for metric_name, channel_values in metrics.items():
            global_value = np.nanmean(channel_values)
            column_name = f"quality_global_{metric_name}"
            if column_name not in results:
                results[column_name] = [np.nan] * n_epochs
            results[column_name][epoch_idx] = global_value
    
    if not results:
        raise ValueError("Quality: no metrics were produced.")
    
    return pd.DataFrame(results)
