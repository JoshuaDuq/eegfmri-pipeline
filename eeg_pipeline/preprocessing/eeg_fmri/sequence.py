"""Run-specific multiband acquisition timing for gradient correction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MultibandSliceSchedule:
    """Exact BIDS slice timing collapsed into simultaneous acquisition groups."""

    repetition_time_seconds: float
    slice_times_seconds: Sequence[float] | np.ndarray
    multiband_factor: int
    group_times_seconds: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        repetition_time = float(self.repetition_time_seconds)
        if not np.isfinite(repetition_time) or repetition_time <= 0:
            raise ValueError("repetition_time_seconds must be finite and positive")
        if type(self.multiband_factor) is not int or self.multiband_factor < 1:
            raise ValueError("multiband_factor must be a positive integer")

        slice_times = np.asarray(self.slice_times_seconds, dtype=float)
        if slice_times.ndim != 1 or slice_times.size == 0:
            raise ValueError("slice_times_seconds must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(slice_times)):
            raise ValueError("slice_times_seconds contains non-finite values")
        if np.any(slice_times < 0) or np.any(slice_times >= repetition_time):
            raise ValueError("slice times must fall within the repetition time")

        group_times, group_counts = np.unique(slice_times, return_counts=True)
        if np.any(group_counts != self.multiband_factor):
            raise ValueError(
                "Each acquisition group must contain exactly " f"{self.multiband_factor} slices"
            )
        if not np.isclose(group_times[0], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("The first acquisition group must occur at volume onset")

        immutable_slice_times = slice_times.copy()
        immutable_group_times = group_times.copy()
        immutable_slice_times.setflags(write=False)
        immutable_group_times.setflags(write=False)
        object.__setattr__(self, "slice_times_seconds", immutable_slice_times)
        object.__setattr__(self, "group_times_seconds", immutable_group_times)

    @property
    def slice_count(self) -> int:
        return int(np.asarray(self.slice_times_seconds).size)

    @property
    def group_count(self) -> int:
        return int(self.group_times_seconds.size)

    def group_boundaries_samples(self, sampling_frequency: float) -> np.ndarray:
        """Return integer group starts plus the exclusive volume stop sample."""
        if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be finite and positive")
        volume_samples = self.repetition_time_seconds * sampling_frequency
        rounded_volume_samples = round(volume_samples)
        if not np.isclose(volume_samples, rounded_volume_samples, rtol=0.0, atol=1e-9):
            raise ValueError("Repetition time must map to an integer acquisition sample count")

        fractional_starts = self.group_times_seconds * sampling_frequency
        group_starts = np.rint(fractional_starts).astype(int)
        boundaries = np.concatenate((group_starts, [int(rounded_volume_samples)]))
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("Acquisition groups collapse at this sampling frequency")
        return boundaries


def load_multiband_slice_schedule(path: str | Path) -> MultibandSliceSchedule:
    """Load the required multiband timing fields from one BIDS BOLD sidecar."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"BOLD metadata does not exist: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("BOLD metadata root must be an object")
    required = {"RepetitionTime", "SliceTiming", "MultibandAccelerationFactor"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"BOLD metadata is missing required fields: {missing}")
    if not isinstance(payload["SliceTiming"], list):
        raise TypeError("BOLD SliceTiming must be a list")
    if type(payload["MultibandAccelerationFactor"]) is not int:
        raise TypeError("BOLD MultibandAccelerationFactor must be an integer")
    return MultibandSliceSchedule(
        repetition_time_seconds=float(payload["RepetitionTime"]),
        slice_times_seconds=payload["SliceTiming"],
        multiband_factor=payload["MultibandAccelerationFactor"],
    )
