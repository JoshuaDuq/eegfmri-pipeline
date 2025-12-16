from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd


def apply_baseline(
    epochs: mne.Epochs,
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    config=None,
) -> mne.Epochs:
    from ..analysis.tfr import validate_baseline_indices

    times = np.asarray(epochs.times)
    validate_baseline_indices(times, baseline_window, logger=logger, config=config)
    baseline_start = float(baseline_window[0])
    baseline_end = float(baseline_window[1])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return epochs.copy().apply_baseline((baseline_start, min(baseline_end, 0.0)))


def crop_epochs(
    epochs: mne.Epochs,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    logger: Optional[logging.Logger] = None,
) -> mne.Epochs:
    if crop_tmin is None and crop_tmax is None:
        return epochs

    time_min = epochs.tmin if crop_tmin is None else float(crop_tmin)
    time_max = epochs.tmax if crop_tmax is None else float(crop_tmax)

    if time_max <= time_min:
        raise ValueError(f"Invalid crop window: tmin={time_min}, tmax={time_max}")

    epochs_copy = epochs.copy()
    if not getattr(epochs_copy, "preload", False):
        epochs_copy.load_data()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return epochs_copy.crop(tmin=time_min, tmax=time_max, include_tmax=include_tmax_in_crop)


def process_temperature_levels(
    epochs: mne.Epochs,
    temperature_column: str,
) -> Tuple[List, Dict, bool]:
    temperature_series = epochs.metadata[temperature_column]
    numeric_values = pd.to_numeric(temperature_series, errors="coerce")

    if numeric_values.notna().all():
        levels = np.sort(numeric_values.unique())
        labels = {
            value: str(int(value)) if float(value).is_integer() else str(value)
            for value in levels
        }
        return levels, labels, True

    levels = sorted(temperature_series.astype(str).unique())
    return levels, {}, False


def build_epoch_query_string(
    column: str,
    level,
    is_numeric: bool,
    labels: Optional[Dict] = None,
) -> Tuple[str, str]:
    if labels is None:
        labels = {}

    if is_numeric:
        try:
            value = float(level)
        except (TypeError, ValueError):
            value = level
        label = labels.get(level)
        if label is None:
            try:
                label = str(int(value)) if float(value).is_integer() else str(value)
            except Exception:
                label = str(level)
        query = f"{column} == {value}"
        return query, str(label)

    value_str = str(level)
    label = labels.get(level, value_str)
    query = f"{column} == '{value_str}'"
    return query, str(label)


def select_epochs_by_value(epochs: mne.Epochs, column: str, value) -> mne.Epochs:
    if epochs.metadata is None or column not in epochs.metadata.columns:
        raise ValueError(f"Column '{column}' not found in epochs.metadata")

    val_expr = value if isinstance(value, (int, float)) else f"'{value}'"
    selected = epochs[f"{column} == {val_expr}"]
    if len(selected) == 0:
        available_values = (
            sorted(epochs.metadata[column].unique().tolist())
            if column in epochs.metadata.columns
            else []
        )
        raise ValueError(
            f"No epochs found for {column} == {value}. "
            f"Available values: {available_values}"
        )
    return selected
