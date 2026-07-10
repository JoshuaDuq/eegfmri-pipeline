"""Outcome-blind scanner-harmonic spectral QC for Study 1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    DEFAULT_HARMONIC_WINDOWS,
    FrequencyWindow,
)
from eeg_pipeline.utils.config.loader import require_config_value

NUMBERED_SUBJECT_PATTERN = re.compile(r"^sub-\d+$")


@dataclass(frozen=True)
class ScannerHarmonicSpecification:
    """Fixed acquisition and peak-detection settings for the QC figure."""

    frequency_range_hz: tuple[float, float]
    n_fft: int
    n_overlap: int
    sampling_frequency_hz: float
    peak_prominence_db: float
    peak_distance_bins: int
    volume_repetition_time_s: float
    harmonic_orders: tuple[int, ...]
    excluded_subjects: tuple[str, ...]
    harmonic_windows: tuple[FrequencyWindow, ...] = DEFAULT_HARMONIC_WINDOWS


@dataclass(frozen=True)
class HarmonicPeak:
    """Strongest qualifying peak in one prespecified frequency window."""

    window_name: str
    peak_frequency_hz: float
    prominence_db: float


def scanner_harmonic_specification(config: Any) -> ScannerHarmonicSpecification:
    """Load the fixed scanner-harmonic analysis settings."""
    scanner = require_config_value(config, "study1.figures.scanner_harmonics")
    frequency_range = tuple(float(value) for value in scanner["frequency_range_hz"])
    harmonic_orders = tuple(int(value) for value in scanner["harmonic_orders"])
    if len(frequency_range) != 2:
        raise ValueError("Scanner-harmonic frequency_range_hz must contain two values.")
    if len(harmonic_orders) != len(DEFAULT_HARMONIC_WINDOWS):
        raise ValueError(
            "Scanner-harmonic order count must match the fixed harmonic-window count."
        )
    return ScannerHarmonicSpecification(
        frequency_range_hz=(frequency_range[0], frequency_range[1]),
        n_fft=int(scanner["n_fft"]),
        n_overlap=int(scanner["n_overlap"]),
        sampling_frequency_hz=float(scanner["sampling_frequency_hz"]),
        peak_prominence_db=float(scanner["peak_prominence_db"]),
        peak_distance_bins=int(scanner["peak_distance_bins"]),
        volume_repetition_time_s=float(scanner["volume_repetition_time_s"]),
        harmonic_orders=harmonic_orders,
        excluded_subjects=tuple(str(value) for value in scanner["excluded_subjects"]),
    )


def discover_final_clean_runs(
    derivative_root: Path,
    *,
    task: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Discover final-clean FIF runs for numbered, eligible participants."""
    excluded = set(excluded_subjects)
    requested = set(requested_subjects)
    candidates = Path(derivative_root).glob(
        f"sub-*/eeg/sub-*_task-{task}_run-*_proc-clean_raw.fif"
    )
    selected: list[Path] = []
    for path in candidates:
        subject = path.parts[-3]
        if NUMBERED_SUBJECT_PATTERN.fullmatch(subject) is None or subject in excluded:
            continue
        if requested and subject not in requested:
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            "No numbered-participant final-clean FIF files found for task "
            f"{task!r} in {derivative_root}."
        )
    return tuple(sorted(selected))


def select_scanner_harmonic_peaks(
    frequencies: np.ndarray,
    spectrum_db: np.ndarray,
    specification: ScannerHarmonicSpecification,
) -> tuple[HarmonicPeak, ...]:
    """Select the greatest-prominence eligible peak from each fixed window."""
    frequency_values = np.asarray(frequencies, dtype=float)
    spectrum_values = np.asarray(spectrum_db, dtype=float)
    if frequency_values.ndim != 1 or spectrum_values.shape != frequency_values.shape:
        raise ValueError("Scanner-harmonic frequencies and spectrum must be aligned vectors.")
    peak_indices, properties = find_peaks(
        spectrum_values,
        prominence=specification.peak_prominence_db,
        distance=specification.peak_distance_bins,
    )
    peak_frequencies = frequency_values[peak_indices]
    prominences = properties["prominences"]

    selected: list[HarmonicPeak] = []
    for window in specification.harmonic_windows:
        eligible = np.flatnonzero(
            (peak_frequencies >= window.low_hz)
            & (peak_frequencies <= window.high_hz)
        )
        if eligible.size == 0:
            raise ValueError(
                "No qualifying spectral peak in scanner-harmonic window "
                f"{window.label} Hz."
            )
        strongest = int(eligible[np.argmax(prominences[eligible])])
        selected.append(
            HarmonicPeak(
                window_name=window.name,
                peak_frequency_hz=float(peak_frequencies[strongest]),
                prominence_db=float(prominences[strongest]),
            )
        )
    return tuple(selected)


__all__ = [
    "HarmonicPeak",
    "ScannerHarmonicSpecification",
    "discover_final_clean_runs",
    "scanner_harmonic_specification",
    "select_scanner_harmonic_peaks",
]
