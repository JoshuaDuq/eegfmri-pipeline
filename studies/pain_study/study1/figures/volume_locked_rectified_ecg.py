"""Volume-marker-locked rectified ECG analysis for Study 1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mne
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.signal import find_peaks

from eeg_pipeline.utils.config.loader import require_config_value


@dataclass(frozen=True)
class VolumeLockedEcgSpecification:
    """Validated epoch and artifact-peak settings."""

    marker_description: str
    epoch_duration_s: float
    minimum_complete_epochs: int
    peak_prominence_fraction: float
    peak_minimum_distance_ms: float
    peak_search_radius_ms: float
    peak_maximum_latency_sd_ms: float

    def __post_init__(self) -> None:
        if not self.marker_description:
            raise ValueError("marker_description must be non-empty.")
        if not math.isfinite(self.epoch_duration_s) or self.epoch_duration_s <= 0.0:
            raise ValueError("epoch_duration_s must be positive and finite.")
        if self.minimum_complete_epochs < 2:
            raise ValueError("minimum_complete_epochs must be at least 2.")
        if not 0.0 < self.peak_prominence_fraction < 1.0:
            raise ValueError("peak_prominence_fraction must be in (0, 1).")
        for name in (
            "peak_minimum_distance_ms",
            "peak_search_radius_ms",
            "peak_maximum_latency_sd_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")


@dataclass(frozen=True)
class VolumeLockedEcgRun:
    """Mean rectified ECG for one run at its native sampling rate."""

    subject_id: str
    run_id: str
    stage: str
    source_file: str
    sampling_frequency_hz: float
    n_complete_epochs: int
    times_ms: NDArray[np.float64]
    mean_rectified_ecg_uv: NDArray[np.float64]


@dataclass(frozen=True)
class VolumeLockedEcgParticipant:
    """Equal-run participant mean and its contributing run traces."""

    subject_id: str
    stage: str
    sampling_frequency_hz: float
    n_runs: int
    n_complete_epochs: int
    times_ms: NDArray[np.float64]
    mean_rectified_ecg_uv: NDArray[np.float64]
    runs: tuple[VolumeLockedEcgRun, ...]


@dataclass(frozen=True)
class VolumeLockedEcgSummary:
    """Paired participant-stage traces and stable artifact peaks."""

    runs: tuple[VolumeLockedEcgRun, ...]
    participants: tuple[VolumeLockedEcgParticipant, ...]
    peaks: pd.DataFrame
    troughs: pd.DataFrame


def volume_locked_ecg_specification(config: Any) -> VolumeLockedEcgSpecification:
    """Load the Study 1 volume-locked ECG analysis settings."""
    configured = require_config_value(
        config,
        "study1.figures.volume_locked_rectified_ecg",
    )
    peak = configured["peak_detection"]
    marker_segments = configured["marker_segments"]
    if not isinstance(marker_segments, list) or len(marker_segments) != 2:
        raise ValueError("volume_locked_rectified_ecg.marker_segments must contain two values.")
    return VolumeLockedEcgSpecification(
        marker_description="/".join(str(value) for value in marker_segments),
        epoch_duration_s=float(configured["epoch_duration_s"]),
        minimum_complete_epochs=int(configured["minimum_complete_epochs"]),
        peak_prominence_fraction=float(peak["prominence_fraction"]),
        peak_minimum_distance_ms=float(peak["minimum_distance_ms"]),
        peak_search_radius_ms=float(peak["search_radius_ms"]),
        peak_maximum_latency_sd_ms=float(peak["maximum_run_latency_sd_ms"]),
    )


def summarize_volume_locked_ecg_run(
    raw: mne.io.BaseRaw,
    *,
    subject_id: str,
    run_id: str,
    stage: str,
    source_file: str,
    expected_sampling_frequency_hz: float,
    specification: VolumeLockedEcgSpecification,
) -> VolumeLockedEcgRun:
    """Rectify complete ECG volume epochs before averaging them."""
    sampling_frequency_hz = float(raw.info["sfreq"])
    if sampling_frequency_hz != expected_sampling_frequency_hz:
        raise ValueError(
            f"Expected {expected_sampling_frequency_hz:g} Hz for {stage}, "
            f"got {sampling_frequency_hz:g} Hz."
        )
    if raw.ch_names.count("ECG") != 1:
        raise ValueError("Expected exactly one channel named 'ECG'.")

    epoch_samples_float = specification.epoch_duration_s * sampling_frequency_hz
    if not epoch_samples_float.is_integer():
        raise ValueError("epoch_duration_s must yield an integer number of samples.")
    epoch_samples = int(epoch_samples_float)

    descriptions = np.asarray(raw.annotations.description, dtype=str)
    marker_onsets = raw.annotations.onset[descriptions == specification.marker_description]
    starts = raw.time_as_index(marker_onsets, use_rounding=True).astype(np.int64)
    if starts.size and np.any(np.diff(starts) <= 0):
        raise ValueError("Volume-marker samples must be strictly increasing.")
    if starts.size > 1 and np.any(np.diff(starts) < epoch_samples):
        raise ValueError("Volume-marker intervals cannot overlap ECG epochs.")
    starts = starts[starts + epoch_samples <= raw.n_times]
    if starts.size < specification.minimum_complete_epochs:
        raise ValueError(
            f"Expected at least {specification.minimum_complete_epochs} complete ECG epochs, "
            f"got {starts.size}."
        )

    ecg_v = raw.get_data(picks=[raw.ch_names.index("ECG")])[0]
    offsets = np.arange(epoch_samples, dtype=np.int64)
    sample_matrix = starts[:, np.newaxis] + offsets[np.newaxis, :]
    mean_rectified_ecg_uv = np.mean(np.abs(ecg_v[sample_matrix]), axis=0) * 1e6
    times_ms = offsets.astype(float) * 1_000.0 / sampling_frequency_hz
    times_ms.setflags(write=False)
    mean_rectified_ecg_uv.setflags(write=False)
    return VolumeLockedEcgRun(
        subject_id=subject_id,
        run_id=run_id,
        stage=stage,
        source_file=source_file,
        sampling_frequency_hz=sampling_frequency_hz,
        n_complete_epochs=int(starts.size),
        times_ms=times_ms,
        mean_rectified_ecg_uv=mean_rectified_ecg_uv,
    )


def average_volume_locked_ecg_runs(
    runs: tuple[VolumeLockedEcgRun, ...],
) -> VolumeLockedEcgParticipant:
    """Average run means equally within one participant and stage."""
    if not runs:
        raise ValueError("At least one volume-locked ECG run is required.")
    first = runs[0]
    for run in runs[1:]:
        if run.subject_id != first.subject_id or run.stage != first.stage:
            raise ValueError("All ECG runs must belong to one participant and stage.")
        if run.sampling_frequency_hz != first.sampling_frequency_hz:
            raise ValueError("All ECG runs must have the same sampling frequency.")
        if not np.array_equal(run.times_ms, first.times_ms):
            raise ValueError("All ECG runs must have the same native time grid.")

    mean_rectified_ecg_uv = np.mean(
        np.stack([run.mean_rectified_ecg_uv for run in runs]),
        axis=0,
    )
    mean_rectified_ecg_uv.setflags(write=False)
    return VolumeLockedEcgParticipant(
        subject_id=first.subject_id,
        stage=first.stage,
        sampling_frequency_hz=first.sampling_frequency_hz,
        n_runs=len(runs),
        n_complete_epochs=sum(run.n_complete_epochs for run in runs),
        times_ms=first.times_ms,
        mean_rectified_ecg_uv=mean_rectified_ecg_uv,
        runs=runs,
    )


def detect_stable_artifact_peaks(
    participant: VolumeLockedEcgParticipant,
    specification: VolumeLockedEcgSpecification,
) -> pd.DataFrame:
    """Locate prominent participant peaks with stable run-level timing."""
    return _detect_stable_artifact_extrema(
        participant,
        specification,
        polarity=1.0,
        identifier_column="peak_id",
        identifier_prefix="P",
        run_mean_column="run_mean_peak_uv",
        run_sd_column="run_peak_sd_uv",
    )


def detect_stable_artifact_troughs(
    participant: VolumeLockedEcgParticipant,
    specification: VolumeLockedEcgSpecification,
) -> pd.DataFrame:
    """Locate Analyzer-display troughs with stable run-level timing."""
    return _detect_stable_artifact_extrema(
        participant,
        specification,
        polarity=-1.0,
        identifier_column="trough_id",
        identifier_prefix="T",
        run_mean_column="run_mean_trough_uv",
        run_sd_column="run_trough_sd_uv",
    )


def _detect_stable_artifact_extrema(
    participant: VolumeLockedEcgParticipant,
    specification: VolumeLockedEcgSpecification,
    *,
    polarity: float,
    identifier_column: str,
    identifier_prefix: str,
    run_mean_column: str,
    run_sd_column: str,
) -> pd.DataFrame:
    signal = participant.mean_rectified_ecg_uv
    signal_range_uv = float(np.ptp(signal))
    columns = [
        "subject_id",
        "stage",
        identifier_column,
        "latency_ms",
        "mean_rectified_ecg_uv",
        "prominence_uv",
        "run_latency_sd_ms",
        run_mean_column,
        run_sd_column,
        "contributing_runs",
        "sampling_frequency_hz",
    ]
    if signal_range_uv == 0.0:
        return pd.DataFrame(columns=columns)

    minimum_distance_samples = max(
        1,
        int(
            round(
                specification.peak_minimum_distance_ms * participant.sampling_frequency_hz / 1_000.0
            )
        ),
    )
    extrema_indices, properties = find_peaks(
        polarity * signal,
        prominence=specification.peak_prominence_fraction * signal_range_uv,
        distance=minimum_distance_samples,
    )
    rows: list[dict[str, float | int | str]] = []
    for extremum_index, prominence_uv in zip(
        extrema_indices,
        properties["prominences"],
        strict=True,
    ):
        latency_ms = float(participant.times_ms[extremum_index])
        run_latencies_ms = []
        run_amplitudes_uv = []
        for run in participant.runs:
            search_mask = np.abs(run.times_ms - latency_ms) <= specification.peak_search_radius_ms
            search_indices = np.flatnonzero(search_mask)
            if not search_indices.size:
                raise ValueError("Extremum search radius contains no native samples.")
            local_index = int(
                search_indices[np.argmax(polarity * run.mean_rectified_ecg_uv[search_indices])]
            )
            run_latencies_ms.append(float(run.times_ms[local_index]))
            run_amplitudes_uv.append(float(run.mean_rectified_ecg_uv[local_index]))

        latency_sd_ms = (
            float(np.std(run_latencies_ms, ddof=1)) if len(run_latencies_ms) > 1 else 0.0
        )
        if latency_sd_ms > specification.peak_maximum_latency_sd_ms:
            continue
        rows.append(
            {
                "subject_id": participant.subject_id,
                "stage": participant.stage,
                "latency_ms": latency_ms,
                "mean_rectified_ecg_uv": float(signal[extremum_index]),
                "prominence_uv": float(prominence_uv),
                "run_latency_sd_ms": latency_sd_ms,
                run_mean_column: float(np.mean(run_amplitudes_uv)),
                run_sd_column: float(np.std(run_amplitudes_uv, ddof=1)),
                "contributing_runs": len(run_latencies_ms),
                "sampling_frequency_hz": participant.sampling_frequency_hz,
            }
        )

    for index, row in enumerate(rows, start=1):
        row[identifier_column] = f"{identifier_prefix}{index:02d}"
    return pd.DataFrame(rows, columns=columns)


def build_volume_locked_ecg_summary(
    runs: tuple[VolumeLockedEcgRun, ...],
    specification: VolumeLockedEcgSpecification,
    *,
    expected_runs_per_participant: int,
) -> VolumeLockedEcgSummary:
    """Validate stage pairing, average runs, and identify stable peaks."""
    if expected_runs_per_participant < 1:
        raise ValueError("expected_runs_per_participant must be positive.")
    if not runs:
        raise ValueError("At least one volume-locked ECG run is required.")
    identities_by_stage = {
        stage: {(run.subject_id, run.run_id) for run in runs if run.stage == stage}
        for stage in ("raw", "processed")
    }
    observed_stages = {run.stage for run in runs}
    if observed_stages != {"raw", "processed"}:
        raise ValueError("Volume-locked ECG summary requires raw and processed stages.")
    if identities_by_stage["raw"] != identities_by_stage["processed"]:
        raise ValueError("Volume-locked ECG requires paired raw and processed run identities.")

    grouped: dict[tuple[str, str], list[VolumeLockedEcgRun]] = {}
    for run in runs:
        grouped.setdefault((run.subject_id, run.stage), []).append(run)
    for (subject_id, stage), stage_runs in grouped.items():
        if len(stage_runs) != expected_runs_per_participant:
            raise ValueError(
                f"Expected {expected_runs_per_participant} {stage} runs for {subject_id}, "
                f"got {len(stage_runs)}."
            )

    stage_order = {"raw": 0, "processed": 1}
    participants = tuple(
        average_volume_locked_ecg_runs(tuple(sorted(stage_runs, key=lambda run: int(run.run_id))))
        for (subject_id, stage), stage_runs in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], stage_order[item[0][1]]),
        )
    )
    peak_frames = [
        detect_stable_artifact_peaks(participant, specification) for participant in participants
    ]
    peaks = pd.concat(peak_frames, ignore_index=True)
    trough_frames = [
        detect_stable_artifact_troughs(participant, specification) for participant in participants
    ]
    troughs = pd.concat(trough_frames, ignore_index=True)
    return VolumeLockedEcgSummary(
        runs=tuple(sorted(runs, key=lambda run: (run.subject_id, int(run.run_id), run.stage))),
        participants=participants,
        peaks=peaks,
        troughs=troughs,
    )


__all__ = [
    "VolumeLockedEcgParticipant",
    "VolumeLockedEcgRun",
    "VolumeLockedEcgSpecification",
    "VolumeLockedEcgSummary",
    "average_volume_locked_ecg_runs",
    "build_volume_locked_ecg_summary",
    "detect_stable_artifact_peaks",
    "detect_stable_artifact_troughs",
    "summarize_volume_locked_ecg_run",
    "volume_locked_ecg_specification",
]
