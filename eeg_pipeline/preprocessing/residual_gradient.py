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


@dataclass(frozen=True)
class ResidualObsResult:
    """Corrected recording and fold-level component audit records."""

    raw: mne.io.BaseRaw
    component_rows: tuple[dict[str, float | int | str], ...]


def validate_brainvision_source(vhdr_path: str | Path) -> Path:
    """Validate one Analyzer-corrected BrainVision triplet."""
    return brainvision_source_files(vhdr_path)[0]


def brainvision_source_files(vhdr_path: str | Path) -> tuple[Path, Path, Path]:
    """Return the validated header, data, and marker files in source order."""
    path = Path(vhdr_path)
    if path.suffix.lower() != ".vhdr":
        raise ValueError(f"Expected a .vhdr BrainVision header, got: {path}")
    if not path.name.endswith("_scannerpulse_corrected.vhdr"):
        raise ValueError(f"Expected a _scannerpulse_corrected.vhdr input, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"BrainVision header does not exist: {path}")

    entries = _read_header_entries(path)
    data_path = _require_referenced_file(path, entries, "DataFile", "data")
    marker_path = _require_referenced_file(path, entries, "MarkerFile", "marker")
    return path, data_path, marker_path


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
            f"Volume marker interval {interval} is shorter than the {epoch_samples}-sample TR."
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


def apply_residual_obs(
    raw: mne.io.BaseRaw,
    layout: VolumeLayout,
    *,
    n_components: int,
    n_folds: int,
) -> ResidualObsResult:
    """Remove held-out temporal PCA projections from complete EEG volume epochs."""
    return apply_residual_obs_grid(
        raw,
        layout,
        component_counts=(n_components,),
        n_folds=n_folds,
    )[n_components]


def apply_residual_obs_grid(
    raw: mne.io.BaseRaw,
    layout: VolumeLayout,
    *,
    component_counts: tuple[int, ...],
    n_folds: int,
) -> dict[int, ResidualObsResult]:
    """Derive multiple OBS orders from one decomposition per channel and fold."""
    if not component_counts:
        raise ValueError("component_counts must be non-empty.")
    if any(type(count) is not int or count < 0 for count in component_counts):
        raise ValueError("component_counts must contain non-negative integers.")
    if len(set(component_counts)) != len(component_counts):
        raise ValueError("component_counts must not contain duplicates.")
    if n_folds < 2 or n_folds > layout.n_epochs:
        raise ValueError("n_folds must be between 2 and the number of volume epochs.")

    source = raw.copy().load_data()
    corrected_by_count = {
        count: source if index == 0 else source.copy()
        for index, count in enumerate(component_counts)
    }
    positive_counts = tuple(sorted(count for count in component_counts if count > 0))
    if not positive_counts:
        return {
            count: ResidualObsResult(raw=corrected_by_count[count], component_rows=())
            for count in component_counts
        }

    picks = mne.pick_types(source.info, eeg=True, exclude=[])
    offsets = np.arange(layout.epoch_samples, dtype=np.int64)
    sample_matrix = layout.starts[:, np.newaxis] + offsets[np.newaxis, :]
    fold_ids = np.arange(layout.n_epochs, dtype=np.int64) % n_folds
    rows_by_count: dict[int, list[dict[str, float | int | str]]] = {
        count: [] for count in positive_counts
    }
    maximum_count = max(positive_counts)

    for pick in picks:
        channel_epochs = source._data[pick, sample_matrix].copy()
        centered_epochs = channel_epochs - channel_epochs.mean(axis=1, keepdims=True)
        corrected_epochs_by_count = {count: channel_epochs.copy() for count in positive_counts}

        for fold in range(n_folds):
            held_mask = fold_ids == fold
            train = centered_epochs[~held_mask]
            basis_rank = min(train.shape)
            if maximum_count > basis_rank:
                raise ValueError(f"n_components={maximum_count} exceeds basis rank {basis_rank}.")

            _, singular_values, right_vectors = np.linalg.svd(train, full_matrices=False)
            total_variance = float(np.sum(singular_values**2))
            if total_variance == 0.0:
                channel = source.ch_names[pick]
                raise ValueError(f"Training data have zero variance for channel {channel!r}.")

            basis = _normalize_component_signs(right_vectors[:maximum_count])
            held = centered_epochs[held_mask]
            for count in positive_counts:
                count_basis = basis[:count]
                scores = np.einsum("ij,kj->ik", held, count_basis, optimize=True)
                fitted = np.einsum("ik,kj->ij", scores, count_basis, optimize=True)
                corrected_epochs_by_count[count][held_mask] -= fitted
                explained_variance = float(np.sum(singular_values[:count] ** 2) / total_variance)
                rows_by_count[count].append(
                    {
                        "channel": source.ch_names[pick],
                        "fold": fold,
                        "n_components": count,
                        "training_variance_explained": explained_variance,
                        "held_removed_rms_v": float(np.sqrt(np.mean(fitted**2))),
                    }
                )

        for count in positive_counts:
            corrected_by_count[count]._data[pick, sample_matrix] = corrected_epochs_by_count[count]

    return {
        count: ResidualObsResult(
            raw=corrected_by_count[count],
            component_rows=tuple(rows_by_count.get(count, ())),
        )
        for count in component_counts
    }


def _normalize_component_signs(components: NDArray[np.float64]) -> NDArray[np.float64]:
    normalized = components.copy()
    maxima = np.argmax(np.abs(normalized), axis=1)
    signs = np.sign(normalized[np.arange(normalized.shape[0]), maxima])
    signs[signs == 0] = 1
    return normalized * signs[:, np.newaxis]


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
) -> Path:
    if key not in entries:
        raise ValueError(f"BrainVision header has no {key} entry: {header_path}")
    referenced_path = header_path.parent / entries[key]
    if not referenced_path.is_file():
        raise FileNotFoundError(
            f"BrainVision referenced {label} file does not exist: {referenced_path}"
        )
    return referenced_path
