"""Detect stable plateau samples at volume-locked ECG troughs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PlateauWindow:
    """Half-open sample interval surrounding one refined trough."""

    start: int
    stop: int
    trough_index: int
    threshold: float

    @property
    def sample_count(self) -> int:
        """Return the number of selected samples."""
        return self.stop - self.start


def refine_trough_indices(
    signal: ArrayLike,
    *,
    reference_indices: ArrayLike,
    search_radius_samples: int,
) -> NDArray[np.int64]:
    """Refine reference troughs to run-specific minima within a fixed radius."""
    values = _validate_signal(signal)
    references = _validate_indices(reference_indices, values.size, "reference_indices")
    if search_radius_samples < 0:
        raise ValueError("search_radius_samples must be non-negative.")

    refined = np.empty(references.size, dtype=np.int64)
    for index, reference in enumerate(references):
        start = max(0, int(reference) - search_radius_samples)
        stop = min(values.size, int(reference) + search_radius_samples + 1)
        refined[index] = start + int(np.argmin(values[start:stop]))
    if np.any(np.diff(refined) <= 0):
        raise ValueError("Refined trough indices must be strictly increasing.")
    return refined


def derive_plateau_windows(
    signal: ArrayLike,
    *,
    trough_indices: ArrayLike,
    depth_fraction: float,
    minimum_samples: int,
    maximum_samples: int,
) -> tuple[PlateauWindow, ...]:
    """Select each contiguous trough bottom below a depth-relative threshold."""
    values = _validate_signal(signal)
    troughs = _validate_indices(trough_indices, values.size, "trough_indices")
    if not 0.0 < depth_fraction < 1.0:
        raise ValueError("depth_fraction must be in (0, 1).")
    if minimum_samples < 1 or maximum_samples < minimum_samples:
        raise ValueError("Plateau sample limits must be positive and increasing.")

    boundaries = _phase_boundaries(troughs, values.size)
    windows = tuple(
        _derive_one_window(
            values,
            trough_index=int(trough),
            phase_start=int(boundaries[index]),
            phase_stop=int(boundaries[index + 1]),
            depth_fraction=depth_fraction,
            minimum_samples=minimum_samples,
            maximum_samples=maximum_samples,
        )
        for index, trough in enumerate(troughs)
    )
    return windows


def _derive_one_window(
    signal: NDArray[np.float64],
    *,
    trough_index: int,
    phase_start: int,
    phase_stop: int,
    depth_fraction: float,
    minimum_samples: int,
    maximum_samples: int,
) -> PlateauWindow:
    left_shoulder = float(np.max(signal[phase_start : trough_index + 1]))
    right_shoulder = float(np.max(signal[trough_index:phase_stop]))
    conservative_shoulder = min(left_shoulder, right_shoulder)
    trough_value = float(signal[trough_index])
    if conservative_shoulder <= trough_value:
        raise ValueError(f"Trough at sample {trough_index} has no positive depth.")
    threshold = trough_value + depth_fraction * (conservative_shoulder - trough_value)

    start = trough_index
    while start > phase_start and signal[start - 1] <= threshold:
        start -= 1
    stop = trough_index + 1
    while stop < phase_stop and signal[stop] <= threshold:
        stop += 1

    sample_count = stop - start
    if sample_count < minimum_samples:
        raise ValueError(
            f"Plateau at sample {trough_index} is shorter than {minimum_samples} samples."
        )
    if sample_count > maximum_samples:
        raise ValueError(
            f"Plateau at sample {trough_index} is longer than {maximum_samples} samples."
        )
    return PlateauWindow(start, stop, trough_index, threshold)


def _phase_boundaries(troughs: NDArray[np.int64], signal_size: int) -> NDArray[np.int64]:
    boundaries = np.empty(troughs.size + 1, dtype=np.int64)
    boundaries[0] = 0
    boundaries[-1] = signal_size
    boundaries[1:-1] = (troughs[:-1] + troughs[1:] + 1) // 2
    return boundaries


def _validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values.")
    return values


def _validate_indices(
    indices: ArrayLike,
    signal_size: int,
    name: str,
) -> NDArray[np.int64]:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    integer = raw.astype(np.int64)
    if not np.array_equal(raw, integer):
        raise ValueError(f"{name} must contain integer sample indices.")
    if integer[0] < 0 or integer[-1] >= signal_size or np.any(np.diff(integer) <= 0):
        raise ValueError(f"{name} must be strictly increasing and within the signal.")
    return integer
