"""Band-limited tensor construction for Study 1 deep regression."""

from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np

from eeg_pipeline.utils.config.loader import get_frequency_bands


def _validate_bands(config: Any, bands: list[str]) -> dict[str, list[float]]:
    if not bands:
        raise ValueError("Deep regression requires at least one frequency band.")

    available = get_frequency_bands(config)
    missing = [band for band in bands if band not in available]
    if missing:
        raise ValueError(f"Unknown frequency bands requested for deep regression: {missing}")
    return available


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

    tensors: list[np.ndarray] = []
    for band_name in bands:
        fmin, fmax = band_definitions[band_name]
        filtered = epochs.copy().pick(channels).filter(
            l_freq=float(fmin),
            h_freq=float(fmax),
            picks="eeg",
            verbose=False,
        )
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
