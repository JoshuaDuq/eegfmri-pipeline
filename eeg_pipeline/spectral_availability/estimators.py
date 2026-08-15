from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window


_WELCH_DTFT_OVERSAMPLING = 64


def _positive_float(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(f"{name} must contain real numbers")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numbers")

    result = np.asarray(array, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must contain only finite positive values")
    return result


def _spectral_geometry(
    frequencies: ArrayLike,
    n_cycles: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    frequency_array = _positive_array("frequencies", frequencies)
    cycle_array = _positive_array("n_cycles", n_cycles)
    if frequency_array.shape != cycle_array.shape:
        raise ValueError("frequencies and n_cycles must have the same shape")
    return frequency_array, cycle_array


def multitaper_half_support(full_bandwidth_hz: Real) -> float:
    return _positive_float("full_bandwidth_hz", full_bandwidth_hz) / 2.0


def morlet_half_support(
    frequencies: ArrayLike,
    n_cycles: ArrayLike,
) -> NDArray[np.float64]:
    frequency_array, cycle_array = _spectral_geometry(frequencies, n_cycles)
    return frequency_array / cycle_array * np.sqrt(np.log(2.0))


def multitaper_tfr_half_support(
    frequencies: ArrayLike,
    n_cycles: ArrayLike,
    time_bandwidth: Real,
) -> NDArray[np.float64]:
    frequency_array, cycle_array = _spectral_geometry(frequencies, n_cycles)
    time_bandwidth_value = _positive_float("time_bandwidth", time_bandwidth)
    if time_bandwidth_value < 2.0:
        raise ValueError("time_bandwidth must be at least 2.0")
    full_bandwidth = time_bandwidth_value * frequency_array / cycle_array
    return full_bandwidth / 2.0


def welch_half_support(
    sfreq: Real,
    n_per_seg: Integral,
    window: str | float | tuple[object, ...],
) -> float:
    sampling_frequency = _positive_float("sfreq", sfreq)
    if isinstance(n_per_seg, bool) or not isinstance(n_per_seg, Integral):
        raise TypeError("n_per_seg must be an integer")
    segment_length = int(n_per_seg)
    if segment_length < 2:
        raise ValueError("n_per_seg must be at least two")

    window_values = np.asarray(
        get_window(window, segment_length, fftbins=True),
        dtype=float,
    )
    if window_values.shape != (segment_length,) or not np.all(np.isfinite(window_values)):
        raise ValueError("window must produce one finite value per segment sample")

    transform_length = 1 << (segment_length * _WELCH_DTFT_OVERSAMPLING - 1).bit_length()
    power = np.abs(np.fft.rfft(window_values, n=transform_length)) ** 2
    half_power = power[0] / 2.0
    if not np.isfinite(half_power) or half_power <= 0:
        raise ValueError("window has no finite positive zero-frequency power")

    crossing_candidates = np.flatnonzero(power[1:] <= half_power)
    if crossing_candidates.size == 0:
        raise ValueError("window has no measurable half-power main-lobe crossing")

    upper_index = int(crossing_candidates[0] + 1)
    lower_index = upper_index - 1
    lower_power = power[lower_index]
    upper_power = power[upper_index]
    crossing_fraction = (half_power - lower_power) / (upper_power - lower_power)
    crossing_index = lower_index + crossing_fraction
    return float(crossing_index * sampling_frequency / transform_length)
