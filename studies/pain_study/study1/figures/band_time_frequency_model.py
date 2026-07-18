"""Temperature-slope estimators for Study 1 Hanning time-frequency maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, signal


@dataclass(frozen=True)
class TemperatureModel:
    """Validated trial selection and OLS contrast for delivered temperature."""

    epoch_indices: np.ndarray
    excluded_epoch_indices: np.ndarray
    temperature_weights: np.ndarray
    n_clean_epochs: int
    n_model_trials: int
    design_rank: int
    design_columns: int
    condition_number: float


@dataclass(frozen=True)
class VariableHanningPower:
    """Trial/channel power from frequency-dependent Hanning windows."""

    power: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    window_durations_s: np.ndarray


@dataclass(frozen=True)
class TemperatureSlopeBatch:
    """Weighted temperature-slope contribution from one trial batch."""

    slope_sum: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    n_trials: int
    n_channels: int
    channel_names: tuple[str, ...]


def build_temperature_model(
    events: pd.DataFrame,
    *,
    n_epochs: int,
    temperatures: Sequence[float],
) -> TemperatureModel:
    """Build an OLS temperature contrast adjusted for run, surface, and trial order."""

    required = (
        "epoch_index",
        "stimulus_temp",
        "run_id",
        "selected_surface",
        "trial_number",
    )
    missing = [column for column in required if column not in events]
    if missing:
        raise ValueError(f"Study 1 temperature TFR events missing columns: {missing}.")
    if isinstance(n_epochs, bool) or int(n_epochs) != n_epochs or n_epochs < 1:
        raise ValueError("Study 1 temperature TFR requires a positive epoch count.")
    if len(events) != int(n_epochs):
        raise ValueError("Study 1 temperature TFR events must match the clean epoch count.")

    epoch_indices = _integer_values(events["epoch_index"], "epoch_index")
    expected_indices = np.arange(int(n_epochs), dtype=int)
    if not np.array_equal(epoch_indices, expected_indices):
        raise ValueError("Study 1 temperature TFR events must be in exact epoch-index order.")

    configured_temperatures = np.asarray(tuple(float(value) for value in temperatures))
    if (
        len(configured_temperatures) < 2
        or not np.isfinite(configured_temperatures).all()
        or np.any(np.diff(configured_temperatures) <= 0.0)
    ):
        raise ValueError("Study 1 temperature TFR temperatures must be finite and increasing.")

    numeric = events.loc[:, required[1:]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    finite_temperatures = numeric.loc[np.isfinite(numeric["stimulus_temp"]), "stimulus_temp"]
    unknown = np.setdiff1d(finite_temperatures.unique(), configured_temperatures)
    if len(unknown):
        raise ValueError(f"Study 1 temperature TFR events contain unknown temperatures: {unknown}.")
    model_rows = numeric.loc[finite].copy()
    if len(model_rows) < 2:
        raise ValueError("Study 1 temperature TFR requires at least two modelled trials.")
    observed_temperatures = np.sort(model_rows["stimulus_temp"].unique())
    if not np.array_equal(observed_temperatures, configured_temperatures):
        raise ValueError("Study 1 temperature TFR requires every configured temperature.")

    design = _temperature_design(model_rows)
    design_rank = int(np.linalg.matrix_rank(design))
    design_columns = int(design.shape[1])
    condition_number = float(np.linalg.cond(design))
    if design_rank != design_columns:
        raise ValueError("Study 1 temperature TFR nuisance design is rank deficient.")
    if len(model_rows) <= design_columns:
        raise ValueError(
            "Study 1 temperature TFR nuisance design has no residual degrees of freedom."
        )
    if not np.isfinite(condition_number):
        raise ValueError("Study 1 temperature TFR nuisance design condition number is non-finite.")

    selected_indices = epoch_indices[finite]
    orthogonal, triangular = linalg.qr(design, mode="economic")
    coefficient_operator = linalg.solve_triangular(triangular, orthogonal.T)
    return TemperatureModel(
        epoch_indices=selected_indices,
        excluded_epoch_indices=epoch_indices[~finite],
        temperature_weights=coefficient_operator[1],
        n_clean_epochs=int(n_epochs),
        n_model_trials=len(model_rows),
        design_rank=design_rank,
        design_columns=design_columns,
        condition_number=condition_number,
    )


def compute_variable_cycle_hanning_power(
    data: np.ndarray,
    *,
    sampling_frequency_hz: float,
    epoch_start_s: float,
    frequencies: np.ndarray,
    n_cycles: float,
    time_step_s: float,
    time_window: tuple[float, float],
) -> VariableHanningPower:
    """Compute FieldTrip-style single-taper power with cycles/frequency windows."""

    samples = np.asarray(data, dtype=float)
    frequency_array = np.asarray(frequencies, dtype=float)
    sampling_frequency = _positive_float(sampling_frequency_hz, "sampling frequency")
    cycles = _positive_float(n_cycles, "cycle count")
    step = _positive_float(time_step_s, "time step")
    epoch_start = float(epoch_start_s)
    if samples.ndim != 3 or any(size < 1 for size in samples.shape):
        raise ValueError("Study 1 Hanning TFR data must have trial/channel/time axes.")
    if not np.isfinite(samples).all() or not np.isfinite(epoch_start):
        raise ValueError("Study 1 Hanning TFR data and epoch start must be finite.")
    if (
        frequency_array.ndim != 1
        or len(frequency_array) < 1
        or not np.isfinite(frequency_array).all()
        or np.any(frequency_array <= 0.0)
        or np.any(np.diff(frequency_array) <= 0.0)
    ):
        raise ValueError("Study 1 Hanning TFR frequencies must be positive and increasing.")

    times = _sample_aligned_times(
        time_window,
        step_s=step,
        sampling_frequency_hz=sampling_frequency,
        epoch_start_s=epoch_start,
        n_samples=samples.shape[-1],
    )
    center_indices = np.rint((times - epoch_start) * sampling_frequency).astype(int)
    demeaned = samples - samples.mean(axis=-1, keepdims=True)
    window_samples = np.rint(cycles / frequency_array * sampling_frequency).astype(int)
    if np.any(window_samples < 3):
        raise ValueError("Study 1 Hanning TFR windows require at least three samples.")

    power = np.empty(
        (*samples.shape[:2], len(frequency_array), len(times)),
        dtype=float,
    )
    for frequency_index, (frequency, length) in enumerate(
        zip(frequency_array, window_samples, strict=True)
    ):
        half_window = length // 2
        starts = center_indices - half_window
        if starts.min() < 0 or (starts + length).max() > samples.shape[-1]:
            raise ValueError("Study 1 Hanning TFR window extends beyond the recorded epoch.")
        taper = signal.windows.hann(length, sym=True)
        relative_samples = np.arange(length) - (length - 1) / 2.0
        wave = taper * np.exp(-2j * np.pi * frequency * relative_samples / sampling_frequency)
        kernel = wave[::-1] / np.linalg.norm(taper)
        convolution = signal.fftconvolve(
            demeaned,
            kernel[None, None, :],
            mode="full",
            axes=-1,
        )
        convolution_indices = center_indices + length - 1 - half_window
        power[:, :, frequency_index] = np.abs(convolution[..., convolution_indices]) ** 2
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError("Study 1 Hanning TFR power must be finite and non-negative.")
    return VariableHanningPower(
        power=power,
        times=times,
        frequencies=frequency_array,
        window_durations_s=window_samples / sampling_frequency,
    )


def summarize_temperature_slope_batch(
    *,
    power: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    window_durations_s: np.ndarray,
    channel_names: Sequence[str],
    baseline_window: tuple[float, float],
    display_window: tuple[float, float],
    temperature_weights: np.ndarray,
) -> TemperatureSlopeBatch:
    """Normalize trial/channel power and apply the participant temperature contrast."""

    values = np.asarray(power, dtype=float)
    times_array = np.asarray(times, dtype=float)
    frequency_array = np.asarray(frequencies, dtype=float)
    durations = np.asarray(window_durations_s, dtype=float)
    weights = np.asarray(temperature_weights, dtype=float)
    names = tuple(str(channel).strip() for channel in channel_names)
    expected_shape = (len(weights), len(names), len(frequency_array), len(times_array))
    if values.shape != expected_shape or values.ndim != 4:
        raise ValueError("Study 1 temperature TFR power does not match its coordinates.")
    if durations.shape != frequency_array.shape or np.any(durations <= 0.0):
        raise ValueError("Study 1 temperature TFR window durations are invalid.")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Study 1 temperature TFR channel names must be unique and non-empty.")
    if not np.isfinite(values).all() or np.any(values <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("Study 1 temperature TFR power and weights must be finite.")

    display_mask = _center_mask(times_array, display_window)
    normalized = np.empty(
        (*values.shape[:3], int(display_mask.sum())),
        dtype=float,
    )
    for frequency_index, duration in enumerate(durations):
        baseline_mask = _contained_window_mask(
            times_array,
            baseline_window,
            window_duration_s=float(duration),
        )
        baseline = values[:, :, frequency_index, baseline_mask].mean(axis=-1)
        normalized[:, :, frequency_index] = 10.0 * np.log10(
            values[:, :, frequency_index, display_mask] / baseline[:, :, None]
        )
    channel_mean = normalized.mean(axis=1)
    slope_sum = np.tensordot(weights, channel_mean, axes=(0, 0))
    if not np.isfinite(slope_sum).all():
        raise ValueError("Study 1 temperature TFR slope contribution must be finite.")
    return TemperatureSlopeBatch(
        slope_sum=slope_sum,
        times=times_array[display_mask],
        frequencies=frequency_array,
        n_trials=len(weights),
        n_channels=len(names),
        channel_names=names,
    )


def _temperature_design(rows: pd.DataFrame) -> np.ndarray:
    temperature = rows["stimulus_temp"].to_numpy(dtype=float)
    trial_number = rows["trial_number"].to_numpy(dtype=float)
    trial_standard_deviation = float(np.std(trial_number, ddof=1))
    if trial_standard_deviation <= np.finfo(float).eps:
        raise ValueError("Study 1 temperature TFR trial order must vary.")
    columns = [
        np.ones(len(rows), dtype=float),
        temperature - temperature.mean(),
    ]
    for column in ("run_id", "selected_surface"):
        categorical = pd.Categorical(rows[column], categories=sorted(rows[column].unique()))
        columns.extend(
            pd.get_dummies(categorical, drop_first=True, dtype=float).to_numpy(dtype=float).T
        )
    columns.append((trial_number - trial_number.mean()) / trial_standard_deviation)
    return np.column_stack(columns)


def _sample_aligned_times(
    window: tuple[float, float],
    *,
    step_s: float,
    sampling_frequency_hz: float,
    epoch_start_s: float,
    n_samples: int,
) -> np.ndarray:
    start, end = _window(window, "analysis")
    step_samples = step_s * sampling_frequency_hz
    if not np.isclose(step_samples, round(step_samples), rtol=0.0, atol=1e-9):
        raise ValueError("Study 1 Hanning TFR time step must resolve to whole samples.")
    count = int(round((end - start) / step_s))
    times = start + np.arange(count + 1) * step_s
    if not np.isclose(times[-1], end, rtol=0.0, atol=1e-9):
        raise ValueError("Study 1 Hanning TFR time window must resolve to whole steps.")
    indices = (times - epoch_start_s) * sampling_frequency_hz
    if not np.allclose(indices, np.rint(indices), rtol=0.0, atol=1e-9):
        raise ValueError("Study 1 Hanning TFR centers must align with samples.")
    if indices.min() < 0 or indices.max() >= n_samples:
        raise ValueError("Study 1 Hanning TFR centers extend beyond the recorded epoch.")
    return times


def _contained_window_mask(
    times: np.ndarray,
    window: tuple[float, float],
    *,
    window_duration_s: float,
) -> np.ndarray:
    start, end = _window(window, "baseline")
    half_window = window_duration_s / 2.0
    tolerance = 1e-9
    mask = (times - half_window >= start - tolerance) & (times + half_window <= end + tolerance)
    if not mask.any():
        raise ValueError("Study 1 temperature TFR baseline contains no complete Hanning windows.")
    return mask


def _center_mask(times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    start, end = _window(window, "display")
    mask = (times >= start) & (times <= end)
    if not mask.any():
        raise ValueError("Study 1 temperature TFR display window contains no centers.")
    return mask


def _window(value: tuple[float, float], label: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"Study 1 temperature TFR {label} window requires two values.")
    start, end = (float(item) for item in value)
    if not np.isfinite((start, end)).all() or start >= end:
        raise ValueError(f"Study 1 temperature TFR {label} window must be finite and increasing.")
    return start, end


def _positive_float(value: object, label: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"Study 1 Hanning TFR {label} must be positive and finite.")
    return parsed


def _integer_values(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.allclose(numeric, np.rint(numeric)):
        raise ValueError(f"Study 1 temperature TFR {label} must contain finite integers.")
    return np.rint(numeric).astype(int)


__all__ = [
    "TemperatureModel",
    "TemperatureSlopeBatch",
    "VariableHanningPower",
    "build_temperature_model",
    "compute_variable_cycle_hanning_power",
    "summarize_temperature_slope_batch",
]
