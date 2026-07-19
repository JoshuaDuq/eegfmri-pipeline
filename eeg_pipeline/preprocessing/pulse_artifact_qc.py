"""Validation of BrainVision Analyzer pulse-artifact markers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np

PULSE_MARKER_DESCRIPTION = "Pulse Artifact/R"


@dataclass(frozen=True)
class PulseMarkerCriteria:
    """Physiological acceptance criteria for Analyzer R markers."""

    minimum_bpm: float
    maximum_bpm: float
    minimum_marker_fraction: float
    minimum_recording_coverage: float

    def __post_init__(self) -> None:
        if not 0 < self.minimum_bpm < self.maximum_bpm:
            raise ValueError("Pulse-marker BPM bounds must satisfy 0 < minimum < maximum.")
        for name in ("minimum_marker_fraction", "minimum_recording_coverage"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")


@dataclass(frozen=True)
class PulseMarkerMetrics:
    """Validated run-level pulse-marker measurements."""

    recording_id: str
    marker_count: int
    median_bpm: float
    marker_fraction: float
    recording_coverage: float


def _pulse_onsets(raw: mne.io.BaseRaw) -> np.ndarray:
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    return np.asarray(
        raw.annotations.onset[descriptions == PULSE_MARKER_DESCRIPTION],
        dtype=float,
    )


def validate_pulse_markers(
    raw: mne.io.BaseRaw,
    criteria: PulseMarkerCriteria,
    *,
    recording_id: str,
) -> PulseMarkerMetrics:
    """Validate that Analyzer R markers cover a run at a physiological rate."""
    if not recording_id.strip():
        raise ValueError("recording_id must not be empty.")

    onsets = _pulse_onsets(raw)
    if len(onsets) < 3:
        raise ValueError(f"{recording_id}: pulse marker count {len(onsets)} is insufficient.")

    intervals = np.diff(onsets)
    if np.any(intervals <= 0):
        raise ValueError(f"{recording_id}: pulse marker onsets must be strictly increasing.")

    median_bpm = 60.0 / float(np.median(intervals))
    if not criteria.minimum_bpm <= median_bpm <= criteria.maximum_bpm:
        raise ValueError(
            f"{recording_id}: median pulse-marker heart rate {median_bpm:.1f} bpm "
            f"is outside {criteria.minimum_bpm:.1f}-{criteria.maximum_bpm:.1f} bpm."
        )

    duration_seconds = raw.n_times / float(raw.info["sfreq"])
    minimum_marker_count = int(
        np.ceil(duration_seconds * criteria.minimum_bpm / 60.0 * criteria.minimum_marker_fraction)
    )
    if len(onsets) < minimum_marker_count:
        raise ValueError(
            f"{recording_id}: pulse marker count {len(onsets)} is below the required "
            f"minimum of {minimum_marker_count}."
        )

    representative_interval = float(np.quantile(intervals, 0.2, method="lower"))
    marker_span = float(onsets[-1] - onsets[0])
    expected_marker_count = int(np.floor(marker_span / representative_interval)) + 1
    marker_fraction = min(1.0, len(onsets) / expected_marker_count)
    if marker_fraction < criteria.minimum_marker_fraction:
        raise ValueError(
            f"{recording_id}: pulse marker fraction {marker_fraction:.3f} is below "
            f"{criteria.minimum_marker_fraction:.3f}."
        )

    recording_coverage = float((onsets[-1] - onsets[0]) / duration_seconds)
    if recording_coverage < criteria.minimum_recording_coverage:
        raise ValueError(
            f"{recording_id}: pulse-marker recording coverage {recording_coverage:.3f} "
            f"is below {criteria.minimum_recording_coverage:.3f}."
        )

    return PulseMarkerMetrics(
        recording_id=recording_id,
        marker_count=len(onsets),
        median_bpm=median_bpm,
        marker_fraction=marker_fraction,
        recording_coverage=recording_coverage,
    )


def validate_pulse_marker_recordings(
    recordings: Iterable[tuple[str, mne.io.BaseRaw]],
    criteria: PulseMarkerCriteria,
    *,
    output_path: Path,
) -> Path:
    """Validate runs, write their QC table, then surface any invalid inputs."""
    rows = []
    errors = []
    for recording_id, raw in recordings:
        try:
            metrics = validate_pulse_markers(
                raw,
                criteria,
                recording_id=recording_id,
            )
            rows.append(
                {
                    "recording_id": metrics.recording_id,
                    "marker_count": metrics.marker_count,
                    "median_bpm": metrics.median_bpm,
                    "marker_fraction": metrics.marker_fraction,
                    "recording_coverage": metrics.recording_coverage,
                    "status": "pass",
                    "error": "",
                }
            )
        except ValueError as error:
            errors.append(str(error))
            rows.append(
                {
                    "recording_id": recording_id,
                    "marker_count": "",
                    "median_bpm": "",
                    "marker_fraction": "",
                    "recording_coverage": "",
                    "status": "fail",
                    "error": str(error),
                }
            )

    if not rows:
        raise ValueError("No EEG recordings were provided for pulse-marker QC.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id",
        "marker_count",
        "median_bpm",
        "marker_fraction",
        "recording_coverage",
        "status",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    if errors:
        raise ValueError("Invalid BrainVision Analyzer pulse markers: " + " | ".join(errors))
    return output_path


__all__ = [
    "PULSE_MARKER_DESCRIPTION",
    "PulseMarkerCriteria",
    "PulseMarkerMetrics",
    "validate_pulse_marker_recordings",
    "validate_pulse_markers",
]
