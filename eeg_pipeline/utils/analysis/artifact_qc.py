"""Shared trial-level artifact metric helpers."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
from mne.filter import filter_data
from scipy.signal import hilbert


def window_mask(times: np.ndarray, window: Tuple[float, float], *, path: str) -> np.ndarray:
    start, end = window
    mask = (times >= float(start)) & (times <= float(end))
    if not np.any(mask):
        raise ValueError(f"{path} window {window} does not overlap epoch times.")
    return mask


def pick_channels(info: Any, names: Sequence[str], *, path: str) -> List[int]:
    channel_names = list(info["ch_names"])
    picks: List[int] = []
    missing: List[str] = []
    for name in names:
        if name not in channel_names:
            missing.append(str(name))
            continue
        picks.append(channel_names.index(str(name)))
    if missing:
        raise ValueError(f"{path} references missing channels: {missing}")
    if not picks:
        raise ValueError(f"{path} did not resolve any channels.")
    return picks


def metric_rms(data: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(data), axis=(1, 2)))


def metric_variance(data: np.ndarray) -> np.ndarray:
    return np.var(data, axis=(1, 2), ddof=0)


def mean_abs_correlation_metric(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    if primary.ndim != 3 or secondary.ndim != 3:
        raise ValueError("Correlation metric expects arrays shaped (epochs, channels, times).")
    if primary.shape[0] != secondary.shape[0] or primary.shape[2] != secondary.shape[2]:
        raise ValueError("Primary and secondary arrays must align on epoch and time dimensions.")
    primary_centered = primary - np.mean(primary, axis=2, keepdims=True)
    secondary_centered = secondary - np.mean(secondary, axis=2, keepdims=True)
    primary_scale = np.std(primary_centered, axis=2, ddof=0, keepdims=True)
    secondary_scale = np.std(secondary_centered, axis=2, ddof=0, keepdims=True)
    primary_norm = np.divide(
        primary_centered,
        primary_scale,
        out=np.zeros_like(primary_centered),
        where=primary_scale > 0,
    )
    secondary_norm = np.divide(
        secondary_centered,
        secondary_scale,
        out=np.zeros_like(secondary_centered),
        where=secondary_scale > 0,
    )
    denom = float(primary.shape[2])
    if denom <= 0:
        raise ValueError("Correlation metric requires at least one time sample.")
    correlation = np.einsum("ect,edt->ecd", primary_norm, secondary_norm) / denom
    return np.mean(np.abs(correlation), axis=(1, 2))


def band_power_metric(
    *,
    epochs: Any,
    channels: Sequence[str],
    band: Tuple[float, float],
    mask: np.ndarray,
    path: str,
) -> np.ndarray:
    picks = pick_channels(epochs.info, channels, path=path)
    data = np.asarray(epochs.get_data(picks=picks), dtype=float)
    n_epochs, n_channels, n_times = data.shape
    reshaped = data.reshape(n_epochs * n_channels, n_times)
    filtered = filter_data(
        reshaped,
        sfreq=float(epochs.info["sfreq"]),
        l_freq=float(band[0]),
        h_freq=float(band[1]),
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        copy=True,
        verbose=False,
    )
    analytic = hilbert(filtered, axis=-1)
    power = np.abs(analytic) ** 2
    power = power.reshape(n_epochs, n_channels, n_times)
    return np.mean(power[:, :, mask], axis=(1, 2))


__all__ = [
    "band_power_metric",
    "mean_abs_correlation_metric",
    "metric_rms",
    "metric_variance",
    "pick_channels",
    "window_mask",
]
