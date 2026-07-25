"""Validated MRI, BIDS EEG, and YAML inputs for T1 electrode localization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from studies.pain_study.scripts.t1_electrode_localization import LocalizationParameters


@dataclass(frozen=True)
class SubjectLocalizationInput:
    """Participant-specific localization inputs."""

    subject_id: str
    t1w_path: Path
    eeg_bids_subject_directory: Path


@dataclass(frozen=True)
class RunConfiguration:
    """Validated configuration for a localization run."""

    output_root: Path
    montage_name: str
    parameters: LocalizationParameters
    participants: tuple[SubjectLocalizationInput, ...]


def _resolve_config_path(value: object, config_directory: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field {field!r} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_directory / path


def load_run_configuration(config_path: str | Path) -> RunConfiguration:
    """Load a strict YAML configuration for T1 electrode localization."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Localization configuration does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Localization configuration must be a YAML mapping.")
    required_keys = {"output_root", "montage", "parameters", "participants"}
    if set(raw) != required_keys:
        raise ValueError(
            "Localization configuration must contain exactly "
            f"{sorted(required_keys)}, got {sorted(raw)}."
        )
    if not isinstance(raw["montage"], str) or not raw["montage"].strip():
        raise ValueError("Configuration field 'montage' must be a non-empty string.")
    if not isinstance(raw["parameters"], dict):
        raise ValueError("Configuration field 'parameters' must be a mapping.")
    try:
        parameters = LocalizationParameters(**raw["parameters"])
    except TypeError as error:
        raise ValueError(f"Invalid localization parameters: {error}") from error
    participant_mapping = raw["participants"]
    if not isinstance(participant_mapping, dict) or not participant_mapping:
        raise ValueError("Configuration field 'participants' must be a non-empty mapping.")

    config_directory = path.parent
    participants: list[SubjectLocalizationInput] = []
    participant_fields = {"t1w", "eeg_bids_subject_directory"}
    for subject_id, values in participant_mapping.items():
        if not isinstance(subject_id, str) or not subject_id.startswith("sub-"):
            raise ValueError(f"Invalid BIDS participant identifier: {subject_id!r}.")
        if not isinstance(values, dict) or set(values) != participant_fields:
            raise ValueError(
                f"Participant {subject_id} must contain exactly " f"{sorted(participant_fields)}."
            )
        participants.append(
            SubjectLocalizationInput(
                subject_id=subject_id,
                t1w_path=_resolve_config_path(
                    values["t1w"],
                    config_directory,
                    f"participants.{subject_id}.t1w",
                ),
                eeg_bids_subject_directory=_resolve_config_path(
                    values["eeg_bids_subject_directory"],
                    config_directory,
                    f"participants.{subject_id}.eeg_bids_subject_directory",
                ),
            )
        )
    return RunConfiguration(
        output_root=_resolve_config_path(
            raw["output_root"],
            config_directory,
            "output_root",
        ),
        montage_name=raw["montage"],
        parameters=parameters,
        participants=tuple(participants),
    )


def load_canonical_t1(t1w_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a full-head T1 image in canonical scanner RAS voxel order."""
    path = Path(t1w_path)
    if not path.is_file():
        raise FileNotFoundError(f"T1w image does not exist: {path}")
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"T1w image must be three-dimensional, got {image.shape}.")
    canonical = nib.as_closest_canonical(image)
    affine = np.asarray(canonical.affine, dtype=float)
    voxel_sizes = nib.affines.voxel_sizes(affine)
    if np.any(voxel_sizes < 0.5) or np.any(voxel_sizes > 2.0):
        raise ValueError(
            "T1w voxel sizes must lie between 0.5 and 2.0 mm for electrode "
            f"localization, got {voxel_sizes.tolist()} mm."
        )
    data = canonical.get_fdata(dtype=np.float32)
    return data, affine


def _read_eeg_channel_names(channels_path: Path) -> tuple[str, ...]:
    with channels_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not {"name", "type"}.issubset(reader.fieldnames):
            raise ValueError(f"BIDS channels sidecar lacks name/type columns: {channels_path}")
        names = tuple(row["name"].strip() for row in reader if row["type"].strip().upper() == "EEG")
    if not names:
        raise ValueError(f"No EEG channels were found in {channels_path}.")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate EEG channel names were found in {channels_path}.")
    return names


def discover_eeg_channel_names(subject_directory: str | Path) -> tuple[str, ...]:
    """Read the common EEG channel set across all BIDS channel sidecars."""
    directory = Path(subject_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"EEG BIDS subject directory does not exist: {directory}")
    sidecars = sorted(
        path for path in directory.rglob("*_channels.tsv") if not path.name.startswith("._")
    )
    if not sidecars:
        raise FileNotFoundError(f"No BIDS channels.tsv sidecar exists below {directory}.")
    channel_sets = [_read_eeg_channel_names(path) for path in sidecars]
    reference_set = set(channel_sets[0])
    if any(set(names) != reference_set for names in channel_sets[1:]):
        raise ValueError(
            "All BIDS channels.tsv files must contain identical EEG channel sets: " f"{sidecars}."
        )
    return channel_sets[0]


def make_template_positions(
    channel_names: tuple[str, ...],
    montage_name: str,
) -> dict[str, np.ndarray]:
    """Return standard-cap head-coordinate positions for recorded EEG channels."""
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    montage_positions = montage.get_positions()["ch_pos"]
    missing = sorted(set(channel_names) - set(montage_positions))
    if missing:
        raise ValueError(f"EEG channels are absent from MNE montage {montage_name!r}: {missing}.")
    return {name: np.asarray(montage_positions[name], dtype=float) for name in channel_names}


__all__ = [
    "RunConfiguration",
    "SubjectLocalizationInput",
    "discover_eeg_channel_names",
    "load_canonical_t1",
    "load_run_configuration",
    "make_template_positions",
]
