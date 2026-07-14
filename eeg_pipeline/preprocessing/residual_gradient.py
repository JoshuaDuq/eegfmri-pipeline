"""Residual scanner-gradient modeling after BrainVision artifact correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ResidualObsSettings:
    """Fixed acquisition contract for residual OBS."""

    expected_sfreq_hz: float = 1_000.0
    volume_marker: str = "Volume/V  1"
    tr_s: float = 0.9
    min_complete_epochs: int = 50
    n_folds: int = 5

    def __post_init__(self) -> None:
        if self.expected_sfreq_hz <= 0:
            raise ValueError("expected_sfreq_hz must be positive.")
        if not self.volume_marker:
            raise ValueError("volume_marker must be non-empty.")
        if self.tr_s <= 0:
            raise ValueError("tr_s must be positive.")
        if self.min_complete_epochs < 2:
            raise ValueError("min_complete_epochs must be at least 2.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2.")


@dataclass(frozen=True)
class VolumeLayout:
    """Complete, non-overlapping scanner-volume epochs."""

    starts: NDArray[np.int64]
    block_ids: NDArray[np.int64]
    epoch_samples: int

    @property
    def n_epochs(self) -> int:
        return int(self.starts.size)


def validate_brainvision_source(vhdr_path: str | Path) -> Path:
    """Validate one Analyzer-corrected BrainVision triplet."""
    path = Path(vhdr_path)
    if path.suffix.lower() != ".vhdr":
        raise ValueError(f"Expected a .vhdr BrainVision header, got: {path}")
    if not path.name.endswith("_scannerpulse_corrected.vhdr"):
        raise ValueError(f"Expected a _scannerpulse_corrected.vhdr input, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"BrainVision header does not exist: {path}")

    entries = _read_header_entries(path)
    _require_referenced_file(path, entries, "DataFile", "data")
    _require_referenced_file(path, entries, "MarkerFile", "marker")
    return path


def validate_residual_obs_raw(raw: mne.io.BaseRaw, settings: ResidualObsSettings) -> None:
    """Validate an in-memory recording at the residual-OBS boundary."""
    sfreq = float(raw.info["sfreq"])
    if sfreq != settings.expected_sfreq_hz:
        raise ValueError(
            f"Expected sampling frequency {settings.expected_sfreq_hz} Hz, got {sfreq} Hz."
        )
    if settings.volume_marker not in set(map(str, raw.annotations.description)):
        raise ValueError(f"Required annotation is absent: {settings.volume_marker!r}")
    if not mne.pick_types(raw.info, eeg=True, exclude=[]).size:
        raise ValueError("Residual OBS requires at least one EEG channel.")


def build_volume_layout(
    raw: mne.io.BaseRaw,
    settings: ResidualObsSettings,
) -> VolumeLayout:
    """Return complete volume epochs, splitting acquisition blocks at long gaps."""
    validate_residual_obs_raw(raw, settings)
    expected_samples = settings.tr_s * settings.expected_sfreq_hz
    epoch_samples = int(round(expected_samples))
    if not np.isclose(expected_samples, epoch_samples, atol=1e-12):
        raise ValueError("tr_s must map to an integer number of samples.")

    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets = raw.annotations.onset[descriptions == settings.volume_marker]
    starts = raw.time_as_index(onsets, use_rounding=True).astype(np.int64)
    intervals = np.diff(starts)
    if np.any(intervals <= 0):
        raise ValueError("Volume marker samples must be strictly increasing.")
    if np.any(intervals < epoch_samples):
        interval = int(intervals[intervals < epoch_samples][0])
        raise ValueError(
            f"Volume marker interval {interval} is shorter than the "
            f"{epoch_samples}-sample TR."
        )

    starts = starts[starts + epoch_samples <= raw.n_times]
    if starts.size < settings.min_complete_epochs:
        raise ValueError(
            f"Expected at least {settings.min_complete_epochs} complete volume epochs, "
            f"got {starts.size}."
        )

    retained_intervals = np.diff(starts)
    block_ids = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(retained_intervals > epoch_samples)]
    ).astype(np.int64)
    starts.setflags(write=False)
    block_ids.setflags(write=False)
    return VolumeLayout(starts=starts, block_ids=block_ids, epoch_samples=epoch_samples)


def _read_header_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        entries[key.strip()] = value.strip()
    return entries


def _require_referenced_file(
    header_path: Path,
    entries: dict[str, str],
    key: str,
    label: str,
) -> None:
    if key not in entries:
        raise ValueError(f"BrainVision header has no {key} entry: {header_path}")
    referenced_path = header_path.parent / entries[key]
    if not referenced_path.is_file():
        raise FileNotFoundError(
            f"BrainVision referenced {label} file does not exist: {referenced_path}"
        )
