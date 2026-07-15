"""End-of-preprocessing scanner-harmonic cohort QC orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
from mne_bids import find_matching_paths, read_raw_bids

from eeg_pipeline.analysis.qc.scanner_harmonic_comb import (
    ParticipantSpectrum,
    ScannerCombParameters,
    combine_participant_runs,
    compute_epoch_comb_spectrum,
    compute_raw_comb_spectrum,
    summarize_scanner_comb,
)
from eeg_pipeline.infra.paths import find_clean_epochs_path
from eeg_pipeline.plotting.scanner_harmonic_comb import write_scanner_harmonic_comb


@dataclass(frozen=True)
class ScannerHarmonicQcOutputs:
    """Paths written by cohort scanner-harmonic QC."""

    png_path: Path
    tsv_path: Path


def scanner_comb_parameters_from_config(config: Any) -> ScannerCombParameters:
    """Build validated scanner-comb parameters from the pipeline config."""
    values = config.get("preprocessing.scanner_harmonic_qc")
    if not isinstance(values, Mapping):
        raise ValueError("Missing required config mapping: preprocessing.scanner_harmonic_qc")

    expected_keys = {
        "frequency_range_hz",
        "welch_duration_seconds",
        "frequency_resolution_hz",
        "bootstrap_resamples",
        "confidence_level",
    }
    unknown_keys = sorted(set(values) - expected_keys)
    if unknown_keys:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(f"Unknown scanner harmonic QC config keys: {joined_keys}")

    missing_keys = sorted(expected_keys - set(values))
    if missing_keys:
        joined_keys = ", ".join(missing_keys)
        raise ValueError(f"Missing scanner harmonic QC config keys: {joined_keys}")

    frequency_range = values["frequency_range_hz"]
    if not isinstance(frequency_range, (list, tuple)) or len(frequency_range) != 2:
        raise ValueError("frequency_range_hz must contain exactly two values.")

    random_seed = config.get("project.random_state")
    if random_seed is None:
        raise ValueError("Missing required config value: project.random_state")

    return ScannerCombParameters(
        frequency_min_hz=float(frequency_range[0]),
        frequency_max_hz=float(frequency_range[1]),
        welch_duration_seconds=float(values["welch_duration_seconds"]),
        frequency_resolution_hz=float(values["frequency_resolution_hz"]),
        bootstrap_resamples=values["bootstrap_resamples"],
        confidence_level=float(values["confidence_level"]),
        random_seed=random_seed,
    )


def run_scanner_harmonic_qc(
    *,
    subjects: Sequence[str],
    task: str,
    bids_root: Path,
    deriv_root: Path,
    input_extension: str,
    parameters: ScannerCombParameters,
) -> ScannerHarmonicQcOutputs:
    """Compute and write the selected cohort's input-versus-final comb."""
    participants = _normalize_subjects(subjects)
    bids_path = _require_directory(Path(bids_root), "BIDS root")
    derivative_path = _require_directory(Path(deriv_root), "Derivative root")

    input_spectra = [
        _load_input_participant(
            participant=participant,
            task=task,
            bids_root=bids_path,
            input_extension=input_extension,
            parameters=parameters,
        )
        for participant in participants
    ]
    final_spectra = [
        _load_final_participant(
            participant=participant,
            task=task,
            deriv_root=derivative_path,
            parameters=parameters,
        )
        for participant in participants
    ]
    summary = summarize_scanner_comb(input_spectra, final_spectra, parameters)
    png_path, tsv_path = write_scanner_harmonic_comb(
        summary,
        output_dir=derivative_path / "preprocessed" / "eeg" / "qc",
        task=task,
    )
    return ScannerHarmonicQcOutputs(png_path=png_path, tsv_path=tsv_path)


def _load_input_participant(
    *,
    participant: str,
    task: str,
    bids_root: Path,
    input_extension: str,
    parameters: ScannerCombParameters,
) -> ParticipantSpectrum:
    extension = _validate_extension(input_extension)
    bids_paths = find_matching_paths(
        bids_root,
        subjects=participant,
        tasks=task,
        suffixes="eeg",
        extensions=extension,
        datatypes="eeg",
        check=True,
        ignore_json=True,
    )
    visible_paths = [
        path for path in bids_paths if path.fpath.is_file() and not path.fpath.name.startswith("._")
    ]
    if not visible_paths:
        raise FileNotFoundError(
            f"No task-{task} {extension} BIDS EEG runs found for sub-{participant}."
        )
    _validate_unique_runs(visible_paths, participant)

    run_spectra = []
    for bids_path in sorted(visible_paths, key=_run_sort_key):
        raw = read_raw_bids(
            bids_path,
            extra_params={"preload": False},
            verbose=False,
        )
        run_spectra.append(compute_raw_comb_spectrum(raw, parameters))
    return combine_participant_runs(participant, run_spectra)


def _load_final_participant(
    *,
    participant: str,
    task: str,
    deriv_root: Path,
    parameters: ScannerCombParameters,
) -> ParticipantSpectrum:
    epochs_path = find_clean_epochs_path(
        participant,
        task,
        deriv_root=deriv_root,
    )
    if epochs_path is None or not epochs_path.is_file():
        raise FileNotFoundError(f"sub-{participant}, task-{task} is missing final clean epochs.")
    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)
    spectrum = compute_epoch_comb_spectrum(epochs, parameters)
    return ParticipantSpectrum(
        participant,
        spectrum.frequencies_hz,
        spectrum.power_db,
    )


def _normalize_subjects(subjects: Sequence[str]) -> tuple[str, ...]:
    participants = tuple(str(subject).removeprefix("sub-").strip() for subject in subjects)
    if not participants or any(not participant for participant in participants):
        raise ValueError("subjects must contain at least one non-empty participant ID.")
    if len(set(participants)) != len(participants):
        raise ValueError("subjects must be unique.")
    return participants


def _require_directory(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _validate_extension(extension: str) -> str:
    value = str(extension).strip().lower()
    if not value.startswith(".") or len(value) < 2:
        raise ValueError("input_extension must include a leading dot.")
    return value


def _validate_unique_runs(bids_paths: Sequence[object], participant: str) -> None:
    run_labels = [str(path.run or "") for path in bids_paths]
    if len(run_labels) != len(set(run_labels)):
        raise ValueError(f"Duplicate BIDS EEG run labels found for sub-{participant}.")


def _run_sort_key(bids_path: object) -> tuple[int, str]:
    run = str(bids_path.run or "")
    return (int(run), run) if run.isdigit() else (0, run)


__all__ = [
    "ScannerHarmonicQcOutputs",
    "run_scanner_harmonic_qc",
    "scanner_comb_parameters_from_config",
]
