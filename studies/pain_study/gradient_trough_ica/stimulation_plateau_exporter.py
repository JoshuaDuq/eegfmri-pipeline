"""Restrict gradient-trough ICA training data to stimulation plateaus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mne
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.io import loadmat
import yaml

from .configuration import GradientTroughIcaConfig, REPOSITORY_ROOT
from .exporter import (
    TASK,
    _aligned_trial_samples,
    _fieldtrip_data,
    _require_events_path,
    _require_run_headers,
    _write_matlab,
    concatenate_selected_intervals,
    select_thermal_trials,
    write_runtime_manifest,
)

CONFIG_PATH = Path(__file__).resolve().parent / "stimulation_plateau_ica_config.yaml"


@dataclass(frozen=True)
class StimulationPlateauExportConfig:
    """Configuration for deriving a plateau-restricted ICA export."""

    base: GradientTroughIcaConfig
    plateau_start_s: float
    plateau_stop_s: float
    source_export_root: Path

    @classmethod
    def load(
        cls,
        path: Path = CONFIG_PATH,
        *,
        validate_paths: bool = True,
    ) -> StimulationPlateauExportConfig:
        """Load the base workflow and explicit plateau-export overrides."""
        with path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("Stimulation-plateau configuration must be a mapping.")
        required = {
            "base_config",
            "stimulation_plateau_s",
            "source_export_root",
            "output_root",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"Stimulation-plateau configuration lacks values: {missing}")

        base_path = path.parent / str(values["base_config"])
        base = GradientTroughIcaConfig.load(base_path, validate_paths=validate_paths)
        plateau = tuple(float(value) for value in values["stimulation_plateau_s"])
        if len(plateau) != 2 or plateau[0] < 0.0 or plateau[0] >= plateau[1]:
            raise ValueError("stimulation_plateau_s must contain increasing non-negative bounds.")
        if plateau[1] > base.trial_tmax_s:
            raise ValueError("Stimulation plateau extends beyond the exported thermal epoch.")

        source_export_root = _repository_path(str(values["source_export_root"]))
        output_root = _repository_path(str(values["output_root"]))
        if source_export_root == output_root:
            raise ValueError("Source and plateau-restricted output roots must differ.")
        if validate_paths and not source_export_root.is_dir():
            raise FileNotFoundError(f"Source export root does not exist: {source_export_root}")
        return cls(
            base=replace(base, output_root=output_root),
            plateau_start_s=plateau[0],
            plateau_stop_s=plateau[1],
            source_export_root=source_export_root,
        )


@dataclass(frozen=True)
class RestrictedPlateauInterval:
    """One trough interval attributed to a thermal stimulation plateau."""

    source_interval_index: int
    start: int
    stop: int
    volume_number: int
    trough_number: int
    trial_number: int
    temperature_c: float
    trial_trigger_sample: int
    stimulation_plateau_start: int
    stimulation_plateau_stop: int


def restrict_intervals_to_stimulation_plateaus(
    intervals: ArrayLike,
    *,
    trial_trigger_samples: ArrayLike,
    trial_numbers: ArrayLike,
    temperatures_c: ArrayLike,
    plateau_offsets_samples: tuple[int, int],
) -> tuple[RestrictedPlateauInterval, ...]:
    """Retain trough intervals fully contained in exactly one trial plateau."""
    interval_values = np.asarray(intervals)
    if interval_values.ndim != 2 or interval_values.shape[1] != 4:
        raise ValueError("intervals must be an N-by-4 array.")
    interval_values = interval_values.astype(np.int64)
    if interval_values.size == 0:
        raise ValueError("At least one trough interval is required.")
    if np.any(interval_values[:, 1] <= interval_values[:, 0]):
        raise ValueError("Every trough interval must have positive duration.")
    if np.any(interval_values[1:, 0] < interval_values[:-1, 1]):
        raise ValueError("Trough intervals must be chronological and non-overlapping.")

    triggers = _integer_vector(trial_trigger_samples, "trial_trigger_samples")
    numbers = _integer_vector(trial_numbers, "trial_numbers")
    temperatures = np.asarray(temperatures_c, dtype=float)
    if temperatures.ndim != 1 or not np.isfinite(temperatures).all():
        raise ValueError("temperatures_c must be a finite one-dimensional array.")
    if not (triggers.size == numbers.size == temperatures.size):
        raise ValueError("Trial trigger, number, and temperature arrays must align.")
    if np.any(np.diff(triggers) <= 0) or np.any(np.diff(numbers) <= 0):
        raise ValueError("Trial triggers and numbers must be strictly increasing.")

    plateau_start_offset, plateau_stop_offset = plateau_offsets_samples
    if plateau_start_offset < 0 or plateau_stop_offset <= plateau_start_offset:
        raise ValueError("Plateau sample offsets must be non-negative and increasing.")
    plateau_starts = triggers + plateau_start_offset
    plateau_stops = triggers + plateau_stop_offset
    if np.any(plateau_starts[1:] < plateau_stops[:-1]):
        raise ValueError("Stimulation plateau intervals overlap.")

    selected: list[RestrictedPlateauInterval] = []
    trial_counts = np.zeros(triggers.size, dtype=np.int64)
    for interval_index, (start, stop, volume_number, trough_number) in enumerate(interval_values):
        containing = np.flatnonzero((start >= plateau_starts) & (stop <= plateau_stops))
        if containing.size > 1:
            raise ValueError("A trough interval is contained in multiple stimulation plateaus.")
        if containing.size == 0:
            continue
        trial_index = int(containing[0])
        selected.append(
            RestrictedPlateauInterval(
                source_interval_index=interval_index,
                start=int(start),
                stop=int(stop),
                volume_number=int(volume_number),
                trough_number=int(trough_number),
                trial_number=int(numbers[trial_index]),
                temperature_c=float(temperatures[trial_index]),
                trial_trigger_sample=int(triggers[trial_index]),
                stimulation_plateau_start=int(plateau_starts[trial_index]),
                stimulation_plateau_stop=int(plateau_stops[trial_index]),
            )
        )
        trial_counts[trial_index] += 1

    missing_trials = np.flatnonzero(trial_counts == 0)
    if missing_trials.size:
        trial_number = int(numbers[missing_trials[0]])
        raise ValueError(f"Trial {trial_number} has no fully contained trough interval.")
    return tuple(selected)


def export_participant(
    config: StimulationPlateauExportConfig,
    participant: str,
) -> Path:
    """Derive one stimulation-plateau-restricted MATLAB export."""
    base = config.base
    if participant not in base.participants:
        raise ValueError(f"Participant is not configured: {participant}")
    source_path = (
        config.source_export_root
        / "exports"
        / participant
        / f"{participant}_task-{TASK}_desc-troughica_data.mat"
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Whole-run trough ICA export does not exist: {source_path}")
    output_path = (
        base.output_root
        / "exports"
        / participant
        / f"{participant}_task-{TASK}_desc-troughica_data.mat"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_path}")

    gradient_trough = loadmat(source_path, simplify_cells=True)["gradient_trough"]
    _validate_source_package(gradient_trough, participant, base.expected_runs)
    audit_values, audit_columns = _audit_matrix(gradient_trough["selection_audit"])
    source_ica_trials = _trial_list(gradient_trough["ica_data"]["trial"])
    channel_names = tuple(
        str(value) for value in np.atleast_1d(gradient_trough["ica_data"]["label"])
    )
    run_headers = _require_run_headers(base.corrected_data_roots[participant], base.expected_runs)

    selected_runs: list[NDArray[np.float32]] = []
    restricted_audit_rows: list[tuple[float, ...]] = []
    selection_summary_rows: list[tuple[float, ...]] = []
    plateau_offsets = (
        _samples(config.plateau_start_s, base.sampling_frequency_hz),
        _samples(config.plateau_stop_s, base.sampling_frequency_hz),
    )
    for run_index, (run_id, header_path) in enumerate(
        zip(base.expected_runs, run_headers, strict=True)
    ):
        events_path = _require_events_path(base.events_roots[participant], participant, run_id)
        events = pd.read_csv(events_path, sep="\t")
        trials = select_thermal_trials(
            events,
            run_id=run_id,
            thermal_marker=base.thermal_marker,
        )
        raw = mne.io.read_raw_brainvision(header_path, preload=False, verbose="ERROR")
        trial_samples = _aligned_trial_samples(
            raw,
            trials,
            events=events,
            thermal_marker=base.thermal_marker,
            volume_marker=base.volume_marker,
        )
        run_rows = audit_values[audit_values[:, audit_columns["run_id"]] == run_id]
        intervals = np.column_stack(
            (
                run_rows[:, audit_columns["source_start_sample_matlab"]] - 1,
                run_rows[:, audit_columns["source_stop_sample_matlab"]],
                run_rows[:, audit_columns["volume_number"]],
                run_rows[:, audit_columns["trough_number"]],
            )
        ).astype(np.int64)
        restricted = restrict_intervals_to_stimulation_plateaus(
            intervals,
            trial_trigger_samples=trial_samples,
            trial_numbers=trials["trial_number"].to_numpy(dtype=np.int64),
            temperatures_c=trials["stimulus_temp"].to_numpy(dtype=float),
            plateau_offsets_samples=plateau_offsets,
        )
        selected_run, run_audit, run_summary = _extract_restricted_run(
            source_ica_trials[run_index],
            run_rows,
            audit_columns,
            restricted,
            run_id,
            base.sampling_frequency_hz,
        )
        selected_runs.append(selected_run)
        restricted_audit_rows.extend(run_audit)
        selection_summary_rows.extend(run_summary)

    package = {
        "participant": participant,
        "ica_data": _fieldtrip_data(
            selected_runs,
            channel_names,
            base.sampling_frequency_hz,
        ),
        "broadband_data": gradient_trough["broadband_data"],
        "elec": gradient_trough["elec"],
        "trial_metadata": gradient_trough["trial_metadata"],
        "selection_audit": {
            "values": np.asarray(restricted_audit_rows, dtype=np.float64),
            "columns": _matlab_strings(RESTRICTED_AUDIT_COLUMNS),
        },
        "stimulation_plateau_selection": {
            "values": np.asarray(selection_summary_rows, dtype=np.float64),
            "columns": _matlab_strings(SELECTION_SUMMARY_COLUMNS),
        },
        "selection_qc": gradient_trough["selection_qc"],
        "provenance": {
            "source_stage": "BrainVision Analyzer corrected 1 kHz",
            "source_whole_run_export": str(source_path),
            "ica_training_scope": "gradient troughs fully within stimulation plateau",
            "stimulation_plateau_start_s": config.plateau_start_s,
            "stimulation_plateau_stop_s": config.plateau_stop_s,
            "ica_selection_filter": "40 Hz zero-phase FIR high-pass before trough extraction",
            "broadband_filter": "none",
            "data_unit": "V",
            "configuration_file": str(CONFIG_PATH),
        },
    }
    _write_matlab(output_path, "gradient_trough", package)
    return output_path


RESTRICTED_AUDIT_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "volume_number",
    "trough_number",
    "source_start_sample_matlab",
    "source_stop_sample_matlab",
    "selected_run_start_sample_matlab",
    "selected_run_stop_sample_matlab",
    "trial_trigger_sample_matlab",
    "stimulation_plateau_start_sample_matlab",
    "stimulation_plateau_stop_sample_matlab",
    "interval_start_relative_to_trigger_ms",
    "interval_stop_relative_to_trigger_ms",
)

SELECTION_SUMMARY_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "selected_interval_count",
    "selected_sample_count",
)


def _extract_restricted_run(
    source_ica_trial: NDArray[np.floating[Any]],
    source_audit: NDArray[np.float64],
    columns: dict[str, int],
    restricted: tuple[RestrictedPlateauInterval, ...],
    run_id: int,
    sampling_frequency_hz: float,
) -> tuple[NDArray[np.float32], list[tuple[float, ...]], list[tuple[float, ...]]]:
    source_slices = [
        (
            int(
                source_audit[
                    item.source_interval_index, columns["selected_run_start_sample_matlab"]
                ]
            )
            - 1,
            int(
                source_audit[item.source_interval_index, columns["selected_run_stop_sample_matlab"]]
            ),
        )
        for item in restricted
    ]
    selected = concatenate_selected_intervals(source_ica_trial, source_slices).astype(np.float32)
    audit_rows: list[tuple[float, ...]] = []
    selected_offset = 0
    for item in restricted:
        duration = item.stop - item.start
        audit_rows.append(
            (
                run_id,
                item.trial_number,
                item.temperature_c,
                item.volume_number,
                item.trough_number,
                item.start + 1,
                item.stop,
                selected_offset + 1,
                selected_offset + duration,
                item.trial_trigger_sample + 1,
                item.stimulation_plateau_start + 1,
                item.stimulation_plateau_stop,
                (item.start - item.trial_trigger_sample) * 1_000.0 / sampling_frequency_hz,
                (item.stop - item.trial_trigger_sample) * 1_000.0 / sampling_frequency_hz,
            )
        )
        selected_offset += duration
    summary = []
    trial_numbers = sorted({item.trial_number for item in restricted})
    for trial_number in trial_numbers:
        trial_intervals = [item for item in restricted if item.trial_number == trial_number]
        summary.append(
            (
                run_id,
                trial_number,
                trial_intervals[0].temperature_c,
                len(trial_intervals),
                sum(item.stop - item.start for item in trial_intervals),
            )
        )
    return selected, audit_rows, summary


def _validate_source_package(
    package: dict[str, Any],
    participant: str,
    run_ids: tuple[int, ...],
) -> None:
    required = {
        "participant",
        "ica_data",
        "broadband_data",
        "elec",
        "trial_metadata",
        "selection_audit",
        "selection_qc",
    }
    missing = sorted(required - set(package))
    if missing:
        raise ValueError(f"Whole-run export lacks fields: {missing}")
    if str(package["participant"]) != participant:
        raise ValueError("Whole-run export participant does not match the requested participant.")
    if len(_trial_list(package["ica_data"]["trial"])) != len(run_ids):
        raise ValueError("Whole-run ICA trial count does not match configured runs.")


def _audit_matrix(audit: dict[str, Any]) -> tuple[NDArray[np.float64], dict[str, int]]:
    values = np.asarray(audit["values"], dtype=float)
    labels = tuple(str(value) for value in np.atleast_1d(audit["columns"]))
    required = {
        "run_id",
        "volume_number",
        "trough_number",
        "source_start_sample_matlab",
        "source_stop_sample_matlab",
        "selected_run_start_sample_matlab",
        "selected_run_stop_sample_matlab",
    }
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("Whole-run selection audit values and columns are inconsistent.")
    missing = sorted(required - set(labels))
    if missing:
        raise ValueError(f"Whole-run selection audit lacks columns: {missing}")
    return values, {label: index for index, label in enumerate(labels)}


def _trial_list(value: Any) -> list[NDArray[np.float32]]:
    trials = np.atleast_1d(value).tolist()
    return [np.asarray(trial) for trial in trials]


def _integer_vector(values: ArrayLike, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    integer = raw.astype(np.int64)
    if not np.array_equal(raw, integer):
        raise ValueError(f"{name} must contain integers.")
    return integer


def _samples(duration_s: float, sampling_frequency_hz: float) -> int:
    value = duration_s * sampling_frequency_hz
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError(f"Duration {duration_s:g} s does not map to an integer sample count.")
    return int(rounded)


def _matlab_strings(values: tuple[str, ...]) -> NDArray[np.object_]:
    return np.asarray(values, dtype=object).reshape(-1, 1)


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def export_all(config: StimulationPlateauExportConfig) -> tuple[Path, ...]:
    """Export all participants and write the separate MATLAB runtime manifest."""
    exports = tuple(
        export_participant(config, participant) for participant in config.base.participants
    )
    write_runtime_manifest(
        config.base,
        dict(zip(config.base.participants, exports, strict=True)),
    )
    return exports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--participant", action="append")
    return parser.parse_args()


def main() -> None:
    """Create plateau-restricted exports for configured participants."""
    arguments = _parse_args()
    config = StimulationPlateauExportConfig.load(arguments.config)
    participants = tuple(arguments.participant or config.base.participants)
    unknown = sorted(set(participants) - set(config.base.participants))
    if unknown:
        raise ValueError(f"Participants are not configured: {unknown}")
    exports = tuple(export_participant(config, participant) for participant in participants)
    configured_exports = {
        participant: config.base.output_root
        / "exports"
        / participant
        / f"{participant}_task-{TASK}_desc-troughica_data.mat"
        for participant in config.base.participants
    }
    if all(path.is_file() for path in configured_exports.values()):
        write_runtime_manifest(config.base, configured_exports)
    for path in exports:
        print(path.resolve())


if __name__ == "__main__":
    main()
