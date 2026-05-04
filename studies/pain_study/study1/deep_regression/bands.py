"""Band-limited tensor construction for Study 1 deep regression."""

from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np

from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_bands


def _validate_bands(config: Any, bands: list[str]) -> dict[str, list[float]]:
    if not bands:
        raise ValueError("Deep regression requires at least one frequency band.")

    available = get_frequency_bands(config)
    missing = [band for band in bands if band not in available]
    if missing:
        raise ValueError(f"Unknown frequency bands requested for deep regression: {missing}")
    return available


def _deep_regression_time_window(epochs: mne.Epochs, config: Any) -> tuple[float, float] | None:
    raw_window = get_config_value(config, "study1.deep_regression.time_window", None)
    if raw_window is None:
        return None
    if not isinstance(raw_window, (list, tuple)) or len(raw_window) != 2:
        raise ValueError("study1.deep_regression.time_window must be a two-element [start, end] range.")

    try:
        start = float(raw_window[0])
        end = float(raw_window[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"study1.deep_regression.time_window must contain finite numeric bounds, got {raw_window!r}."
        ) from exc
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError(
            f"study1.deep_regression.time_window must satisfy finite start < end, got {raw_window!r}."
        )

    mask = (epochs.times >= start) & (epochs.times <= end)
    if not np.any(mask):
        raise ValueError(
            "study1.deep_regression.time_window does not overlap the clean epoch time axis: "
            f"window=[{start}, {end}], epoch_span=[{epochs.times[0]}, {epochs.times[-1]}]."
        )
    return start, end


def build_band_tensor(
    *,
    epochs: mne.Epochs,
    config: Any,
    bands: list[str],
    channels: list[str],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    if logger is None:
        logger = logging.getLogger(__name__)

    band_definitions = _validate_bands(config, bands)
    if not channels:
        raise ValueError("Deep regression requires at least one common EEG channel.")

    time_window = _deep_regression_time_window(epochs, config)
    tensors: list[np.ndarray] = []
    for band_name in bands:
        fmin, fmax = band_definitions[band_name]
        working_epochs = epochs.copy().pick(channels)
        filtered = working_epochs.filter(
            l_freq=float(fmin),
            h_freq=float(fmax),
            picks="eeg",
            verbose=False,
        )
        filtered.apply_hilbert(envelope=True)
        if time_window is not None:
            filtered.crop(tmin=time_window[0], tmax=time_window[1])
        band_data = filtered.get_data(picks="eeg").astype(float)
        tensors.append(band_data)
        logger.info(
            "Built %s band tensor: %d trials, %d channels, %d timepoints",
            band_name,
            band_data.shape[0],
            band_data.shape[1],
            band_data.shape[2],
        )

    return np.stack(tensors, axis=1)


__all__ = ["build_band_tensor"]
