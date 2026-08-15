from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


_BIDS_ENTITY_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _validate_bids_entity(name: str, value: str | None) -> None:
    if value is None and name == "session":
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _BIDS_ENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty canonical BIDS entity value")


def _finite_float(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _numeric_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(f"{name} must contain real numbers")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numbers")
    return np.asarray(array, dtype=float)


def _frequency_grid(name: str, values: ArrayLike) -> NDArray[np.float64]:
    frequencies = _numeric_array(name, values)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return frequencies


def _half_support(
    half_support_hz: ArrayLike,
    frequency_count: int,
) -> NDArray[np.float64]:
    support = _numeric_array("half_support_hz", half_support_hz)
    if support.ndim == 0:
        support = np.full(frequency_count, support.item())
    elif support.ndim != 1 or support.shape != (frequency_count,):
        raise ValueError(
            "half_support_hz must be a scalar or match the frequency axis"
        )
    if not np.all(np.isfinite(support)) or np.any(support < 0):
        raise ValueError("half_support_hz must contain finite non-negative values")
    return support


def _band_edges(fmin: Real, fmax: Real) -> tuple[float, float]:
    low_hz = _finite_float("fmin", fmin)
    high_hz = _finite_float("fmax", fmax)
    if high_hz <= low_hz:
        raise ValueError("fmax must be greater than fmin")
    return low_hz, high_hz


@dataclass(frozen=True)
class RecordingKey:
    subject: str
    task: str
    run: str
    session: str | None = None

    def __post_init__(self) -> None:
        _validate_bids_entity("subject", self.subject)
        _validate_bids_entity("task", self.task)
        _validate_bids_entity("run", self.run)
        _validate_bids_entity("session", self.session)


@dataclass(frozen=True)
class FrequencyInterval:
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        low_hz = _finite_float("low_hz", self.low_hz)
        high_hz = _finite_float("high_hz", self.high_hz)
        if low_hz < 0:
            raise ValueError("low_hz must be non-negative")
        if high_hz <= low_hz:
            raise ValueError("high_hz must be greater than low_hz")
        object.__setattr__(self, "low_hz", low_hz)
        object.__setattr__(self, "high_hz", high_hz)


def merge_frequency_intervals(
    intervals: Iterable[FrequencyInterval],
) -> tuple[FrequencyInterval, ...]:
    interval_values = tuple(intervals)
    if any(not isinstance(interval, FrequencyInterval) for interval in interval_values):
        raise TypeError("intervals must contain FrequencyInterval values")
    ordered = sorted(interval_values, key=lambda interval: interval.low_hz)
    if not ordered:
        return ()

    merged = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.low_hz <= previous.high_hz:
            merged[-1] = FrequencyInterval(
                previous.low_hz,
                max(previous.high_hz, interval.high_hz),
            )
        else:
            merged.append(interval)
    return tuple(merged)


@dataclass(frozen=True)
class RecordingExclusions:
    key: RecordingKey
    intervals: tuple[FrequencyInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, RecordingKey):
            raise TypeError("key must be a RecordingKey")
        object.__setattr__(self, "intervals", merge_frequency_intervals(self.intervals))


@dataclass(frozen=True)
class EpochSpectralAvailability:
    recording_keys: tuple[RecordingKey, ...]
    exclusions_by_epoch: tuple[tuple[FrequencyInterval, ...], ...]

    def __post_init__(self) -> None:
        recording_keys = tuple(self.recording_keys)
        if any(not isinstance(key, RecordingKey) for key in recording_keys):
            raise TypeError("recording_keys must contain RecordingKey values")

        exclusions_by_epoch = tuple(
            merge_frequency_intervals(intervals)
            for intervals in self.exclusions_by_epoch
        )
        if len(recording_keys) != len(exclusions_by_epoch):
            raise ValueError(
                "recording_keys and exclusions_by_epoch must have the same length"
            )

        object.__setattr__(self, "recording_keys", recording_keys)
        object.__setattr__(self, "exclusions_by_epoch", exclusions_by_epoch)

    def valid_frequency_mask(
        self,
        centre_frequencies: ArrayLike,
        half_support_hz: ArrayLike,
    ) -> NDArray[np.bool_]:
        centres = _frequency_grid("centre_frequencies", centre_frequencies)
        half_support = _half_support(half_support_hz, centres.size)
        support_lows = centres - half_support
        support_highs = centres + half_support
        valid = np.ones((len(self.recording_keys), centres.size), dtype=bool)

        for epoch_index, intervals in enumerate(self.exclusions_by_epoch):
            for interval in intervals:
                overlaps = (support_highs >= interval.low_hz) & (
                    support_lows <= interval.high_hz
                )
                valid[epoch_index, overlaps] = False
        return valid

    def contiguous_band_eligible(
        self,
        fmin: Real,
        fmax: Real,
    ) -> NDArray[np.bool_]:
        return np.array(
            [not intervals for intervals in self.intersections(fmin, fmax)],
            dtype=bool,
        )

    def intersections(
        self,
        fmin: Real,
        fmax: Real,
    ) -> tuple[tuple[FrequencyInterval, ...], ...]:
        low_hz, high_hz = _band_edges(fmin, fmax)
        return tuple(
            tuple(
                interval
                for interval in intervals
                if interval.high_hz >= low_hz and interval.low_hz <= high_hz
            )
            for intervals in self.exclusions_by_epoch
        )

    def retained_bandwidth(
        self,
        frequencies: ArrayLike,
        weights: ArrayLike,
        half_support_hz: ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        frequency_array = _frequency_grid("frequencies", frequencies)
        weight_array = _numeric_array("weights", weights)
        if weight_array.ndim != 1 or weight_array.shape != frequency_array.shape:
            raise ValueError("weights must match the one-dimensional frequency axis")
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0):
            raise ValueError("weights must contain only finite positive values")

        valid = self.valid_frequency_mask(frequency_array, half_support_hz)
        return valid @ weight_array
