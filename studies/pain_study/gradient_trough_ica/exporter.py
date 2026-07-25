"""Export plateau-selected ICA data and unfiltered trials for FieldTrip."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

import mne
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.io import savemat

from .configuration import CONFIG_PATH, GradientTroughIcaConfig
from .trough_selection import PlateauWindow, derive_plateau_windows, refine_trough_indices

TASK = "thermalactive"
EXPECTED_TRIALS_PER_RUN = 11
TRIALINFO_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
)
RUN_PATTERN = re.compile(r"_run(?P<run>\d+)_")


def select_thermal_trials(
    events: pd.DataFrame,
    *,
    run_id: int,
    thermal_marker: str,
    expected_trial_count: int = EXPECTED_TRIALS_PER_RUN,
) -> pd.DataFrame:
    """Return one complete, ordered run of thermal trials."""
    required = {"onset", "trial_type", *TRIALINFO_COLUMNS}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Events table lacks required columns: {missing}")
    trials = events.loc[
        events["trial_type"].eq(thermal_marker) & events["stimulus_temp"].notna()
    ].copy()
    if len(trials) != expected_trial_count:
        raise ValueError(
            f"Run {run_id} has {len(trials)} thermal trials; expected {expected_trial_count}."
        )
    trials["run_id"] = pd.to_numeric(trials["run_id"], errors="raise").astype(int)
    trials["trial_number"] = pd.to_numeric(trials["trial_number"], errors="raise").astype(int)
    if not trials["run_id"].eq(run_id).all():
        raise ValueError(f"Run {run_id} events contain a different run_id.")
    expected_numbers = list(range(1, expected_trial_count + 1))
    if trials["trial_number"].tolist() != expected_numbers:
        raise ValueError(f"Run {run_id} trial numbers must be ordered {expected_numbers}.")
    for column in TRIALINFO_COLUMNS[2:]:
        trials[column] = pd.to_numeric(trials[column], errors="raise")
    return trials.reset_index(drop=True)


def concatenate_selected_intervals(
    data: NDArray[np.floating[Any]],
    intervals: Sequence[tuple[int, int]],
) -> NDArray[np.floating[Any]]:
    """Concatenate strictly ordered half-open sample intervals along time."""
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("data must be a non-empty channels-by-samples array.")
    if not intervals:
        raise ValueError("At least one selected interval is required.")
    previous_stop = 0
    selections: list[NDArray[np.floating[Any]]] = []
    for start, stop in intervals:
        if start < previous_stop or start < 0 or stop <= start or stop > values.shape[1]:
            raise ValueError(
                "Selected intervals must be strictly ordered and non-overlapping within data."
            )
        selections.append(values[:, start:stop])
        previous_stop = stop
    return np.concatenate(selections, axis=1)


def export_participant(
    config: GradientTroughIcaConfig,
    participant: str,
) -> Path:
    """Create one participant's complete MATLAB analysis package."""
    if participant not in config.participants:
        raise ValueError(f"Participant is not configured: {participant}")
    output_directory = config.output_root / "exports" / participant
    output_path = output_directory / f"{participant}_task-{TASK}_desc-troughica_data.mat"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_path}")

    run_headers = _require_run_headers(
        config.corrected_data_roots[participant],
        config.expected_runs,
    )
    selected_runs: list[NDArray[np.float32]] = []
    broadband_trials: list[NDArray[np.float32]] = []
    metadata_frames: list[pd.DataFrame] = []
    audit_rows: list[tuple[float, ...]] = []
    qc_runs: list[dict[str, Any]] = []
    reference_channels: tuple[str, ...] | None = None
    electrode_positions: NDArray[np.float64] | None = None

    for run_id, header_path in zip(config.expected_runs, run_headers, strict=True):
        events_path = _require_events_path(config.events_roots[participant], participant, run_id)
        events = pd.read_csv(events_path, sep="\t")
        trials = select_thermal_trials(
            events,
            run_id=run_id,
            thermal_marker=config.thermal_marker,
        )
        raw = mne.io.read_raw_brainvision(header_path, preload=False, verbose="ERROR")
        _validate_raw(raw, config, run_id)
        eeg_indices = np.asarray(
            [
                index
                for index, channel_type in enumerate(raw.get_channel_types())
                if channel_type == "eeg"
            ],
            dtype=np.int64,
        )
        channels = tuple(raw.ch_names[index] for index in eeg_indices)
        if reference_channels is None:
            reference_channels = channels
            electrode_positions = _electrode_positions(raw, eeg_indices)
        elif channels != reference_channels:
            raise ValueError(f"Run {run_id} EEG channels or ordering differ from run 1.")

        volume_samples = _annotation_samples(raw, config.volume_marker)
        complete_volume_samples = volume_samples[
            volume_samples + _samples(config.volume_epoch_s, config.sampling_frequency_hz)
            <= raw.n_times
        ]
        mean_rectified_ecg_uv = _mean_rectified_volume_ecg(
            raw,
            complete_volume_samples,
            config,
        )
        plateau_windows = _run_plateau_windows(mean_rectified_ecg_uv, config, participant)
        source_intervals = _source_plateau_intervals(
            complete_volume_samples,
            plateau_windows,
            raw.n_times,
        )

        trial_samples = _aligned_trial_samples(
            raw,
            trials,
            events=events,
            thermal_marker=config.thermal_marker,
            volume_marker=config.volume_marker,
        )
        broadband_trials.extend(_extract_broadband_trials(raw, eeg_indices, trial_samples, config))
        trials["source_event_file"] = str(events_path)
        trials["source_brainvision_header"] = str(header_path)
        metadata_frames.append(trials)

        eeg_data = raw.get_data(picks=eeg_indices)
        mne.filter.filter_data(
            eeg_data,
            sfreq=config.sampling_frequency_hz,
            l_freq=config.highpass_hz,
            h_freq=None,
            l_trans_bandwidth=config.highpass_transition_hz,
            method="fir",
            phase="zero",
            fir_design="firwin",
            copy=False,
            verbose="ERROR",
        )
        selected_run = concatenate_selected_intervals(
            eeg_data,
            [(start, stop) for start, stop, _, _ in source_intervals],
        ).astype(np.float32)
        selected_runs.append(selected_run)
        run_audit, _ = _selection_audit(
            run_id,
            source_intervals,
            plateau_windows,
            0,
        )
        audit_rows.extend(run_audit)
        qc_runs.append(
            _selection_qc_run(
                run_id,
                mean_rectified_ecg_uv,
                plateau_windows,
                config.sampling_frequency_hz,
            )
        )

    if reference_channels is None or electrode_positions is None:
        raise RuntimeError(f"No runs were loaded for {participant}.")
    metadata = pd.concat(metadata_frames, ignore_index=True)
    expected_trials = len(config.expected_runs) * EXPECTED_TRIALS_PER_RUN
    if len(broadband_trials) != expected_trials or len(metadata) != expected_trials:
        raise ValueError(
            f"Expected {expected_trials} trials, got {len(broadband_trials)} data trials "
            f"and {len(metadata)} metadata rows."
        )

    export_timestamp = datetime.now(UTC).isoformat()
    package = {
        "participant": participant,
        "ica_data": _fieldtrip_data(
            selected_runs,
            reference_channels,
            config.sampling_frequency_hz,
        ),
        "broadband_data": _fieldtrip_data(
            broadband_trials,
            reference_channels,
            config.sampling_frequency_hz,
            time_start_s=config.trial_tmin_s,
            trialinfo=_numeric_trialinfo(metadata),
            trialinfo_labels=TRIALINFO_COLUMNS,
        ),
        "elec": _fieldtrip_electrodes(reference_channels, electrode_positions),
        "trial_metadata": _dataframe_structure(metadata),
        "selection_audit": {
            "values": np.asarray(audit_rows, dtype=np.float64),
            "columns": _matlab_strings(
                (
                    "run_id",
                    "volume_number",
                    "trough_number",
                    "source_start_sample_matlab",
                    "source_stop_sample_matlab",
                    "selected_run_start_sample_matlab",
                    "selected_run_stop_sample_matlab",
                    "trough_latency_ms",
                    "threshold_uv",
                )
            ),
        },
        "selection_qc": _qc_structure(qc_runs),
        "provenance": {
            "source_stage": "BrainVision Analyzer corrected 1 kHz",
            "ica_selection_filter": (
                f"{config.highpass_hz:g} Hz zero-phase FIR high-pass applied before selection"
            ),
            "broadband_filter": "none",
            "data_unit": "V",
            "export_timestamp_utc": export_timestamp,
            "configuration_file": str(CONFIG_PATH),
        },
    }
    _write_matlab(output_path, "gradient_trough", package)
    return output_path


def export_all(config: GradientTroughIcaConfig) -> tuple[Path, ...]:
    """Export every configured participant and write the MATLAB runtime manifest."""
    exports = tuple(export_participant(config, participant) for participant in config.participants)
    write_runtime_manifest(config, dict(zip(config.participants, exports, strict=True)))
    return exports


def write_runtime_manifest(
    config: GradientTroughIcaConfig,
    exports: dict[str, Path],
) -> Path:
    """Write the MATLAB runtime only after all participant exports exist."""
    if set(exports) != set(config.participants):
        raise ValueError("Runtime exports must contain every configured participant.")
    missing = [path for path in exports.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Participant exports do not exist: {missing}")
    manifest_path = config.output_root / "gradient_trough_ica_runtime.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite runtime manifest: {manifest_path}")
    manifest = {
        "participants": list(config.participants),
        "exports": {participant: str(exports[participant]) for participant in config.participants},
        "output_root": str(config.output_root),
        "ica": {"method": "runica", "extended": True, "seed": config.ica_seed},
        "tfr": {
            "frequencies_hz": list(config.tfr_frequencies_hz),
            "times_s": list(config.tfr_times_s),
            "window_s": config.tfr_window_s,
            "smoothing_hz": config.tfr_smoothing_hz,
            "baseline_s": list(config.tfr_baseline_s),
            "padding_s": config.tfr_padding_s,
            "low_temperatures_c": list(config.low_temperatures_c),
            "high_temperatures_c": list(config.high_temperatures_c),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _require_run_headers(source_root: Path, run_ids: tuple[int, ...]) -> tuple[Path, ...]:
    headers = tuple(sorted(source_root.glob("*.vhdr")))
    by_run: dict[int, list[Path]] = {run_id: [] for run_id in run_ids}
    for header in headers:
        match = RUN_PATTERN.search(header.name)
        if match and int(match.group("run")) in by_run:
            by_run[int(match.group("run"))].append(header)
    resolved: list[Path] = []
    for run_id in run_ids:
        candidates = by_run[run_id]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one corrected header for run {run_id}, found {len(candidates)}."
            )
        header = candidates[0]
        for suffix in (".eeg", ".vmrk"):
            if not header.with_suffix(suffix).is_file():
                raise FileNotFoundError(
                    f"Incomplete BrainVision triplet: {header.with_suffix(suffix)}"
                )
        resolved.append(header)
    return tuple(resolved)


def _require_events_path(events_root: Path, participant: str, run_id: int) -> Path:
    path = events_root / f"{participant}_task-{TASK}_run-{run_id}_events.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Events table does not exist: {path}")
    return path


def _validate_raw(raw: mne.io.BaseRaw, config: GradientTroughIcaConfig, run_id: int) -> None:
    sfreq = float(raw.info["sfreq"])
    if sfreq != config.sampling_frequency_hz:
        raise ValueError(f"Run {run_id} has {sfreq:g} Hz data; expected 1000 Hz.")
    if raw.ch_names.count(config.ecg_channel) != 1:
        raise ValueError(f"Run {run_id} must contain exactly one {config.ecg_channel} channel.")
    if raw.get_channel_types().count("eeg") < 2:
        raise ValueError(f"Run {run_id} contains fewer than two EEG channels.")


def _annotation_samples(raw: mne.io.BaseRaw, description: str) -> NDArray[np.int64]:
    onsets = raw.annotations.onset[
        np.asarray(raw.annotations.description, dtype=str) == description
    ]
    if onsets.size == 0:
        raise ValueError(f"Recording contains no '{description}' annotations.")
    samples = raw.time_as_index(onsets, use_rounding=True).astype(np.int64)
    if np.any(np.diff(samples) <= 0):
        raise ValueError(f"'{description}' annotations are not strictly increasing.")
    return samples


def _mean_rectified_volume_ecg(
    raw: mne.io.BaseRaw,
    volume_samples: NDArray[np.int64],
    config: GradientTroughIcaConfig,
) -> NDArray[np.float64]:
    if volume_samples.size < 2:
        raise ValueError("At least two complete volume epochs are required.")
    epoch_samples = _samples(config.volume_epoch_s, config.sampling_frequency_hz)
    offsets = np.arange(epoch_samples, dtype=np.int64)
    ecg = raw.get_data(picks=[config.ecg_channel])[0]
    return np.mean(np.abs(ecg[volume_samples[:, None] + offsets[None, :]]), axis=0) * 1e6


def _run_plateau_windows(
    trace_uv: NDArray[np.float64],
    config: GradientTroughIcaConfig,
    participant: str,
) -> tuple[PlateauWindow, ...]:
    references = np.rint(
        np.asarray(config.trough_reference_ms[participant]) * config.sampling_frequency_hz / 1_000.0
    ).astype(np.int64)
    troughs = refine_trough_indices(
        trace_uv,
        reference_indices=references,
        search_radius_samples=_samples(
            config.trough_refinement_radius_ms / 1_000.0,
            config.sampling_frequency_hz,
        ),
    )
    return derive_plateau_windows(
        trace_uv,
        trough_indices=troughs,
        depth_fraction=config.plateau_depth_fraction,
        minimum_samples=_samples(
            config.plateau_minimum_duration_ms / 1_000.0,
            config.sampling_frequency_hz,
        ),
        maximum_samples=_samples(
            config.plateau_maximum_duration_ms / 1_000.0,
            config.sampling_frequency_hz,
        ),
    )


def _source_plateau_intervals(
    volume_samples: NDArray[np.int64],
    windows: tuple[PlateauWindow, ...],
    recording_samples: int,
) -> tuple[tuple[int, int, int, int], ...]:
    intervals = tuple(
        (int(volume + window.start), int(volume + window.stop), volume_number, trough_number)
        for volume_number, volume in enumerate(volume_samples, start=1)
        for trough_number, window in enumerate(windows, start=1)
    )
    if not intervals or intervals[-1][1] > recording_samples:
        raise ValueError("A selected plateau interval extends outside the recording.")
    if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
        raise ValueError("Selected plateau intervals overlap or are not chronological.")
    return intervals


def match_relative_event_onsets(
    event_onsets_s: NDArray[np.float64],
    annotation_onsets_s: NDArray[np.float64],
    *,
    first_volume_onset_s: float,
    event_volume_onset_s: float,
    tolerance_s: float,
) -> NDArray[np.int64]:
    """Match first-volume-relative event times to source-clock annotations."""
    event_onsets = np.asarray(event_onsets_s, dtype=float)
    annotation_onsets = np.asarray(annotation_onsets_s, dtype=float)
    if event_onsets.ndim != 1 or annotation_onsets.ndim != 1:
        raise ValueError("Event and annotation onsets must be one-dimensional.")
    if event_onsets.size == 0 or annotation_onsets.size == 0:
        raise ValueError("Event and annotation onsets must be non-empty.")
    if not np.isfinite(first_volume_onset_s) or not np.isfinite(event_volume_onset_s):
        raise ValueError("Source and event clock zeros must be finite.")
    if tolerance_s <= 0.0:
        raise ValueError("Onset matching tolerance must be positive.")

    source_clock_onsets = event_onsets - event_volume_onset_s + first_volume_onset_s
    matches = np.empty(event_onsets.size, dtype=np.int64)
    used: set[int] = set()
    for index, expected_onset in enumerate(source_clock_onsets):
        candidate = int(np.argmin(np.abs(annotation_onsets - expected_onset)))
        difference = abs(float(annotation_onsets[candidate] - expected_onset))
        if difference > tolerance_s:
            raise ValueError(
                f"Thermal trial {index + 1} has no annotation within "
                f"{tolerance_s * 1_000:g} ms; nearest difference is {difference:g} s."
            )
        if candidate in used:
            raise ValueError("Multiple thermal trials matched one annotation.")
        used.add(candidate)
        matches[index] = candidate
    return matches


def _aligned_trial_samples(
    raw: mne.io.BaseRaw,
    trials: pd.DataFrame,
    *,
    events: pd.DataFrame,
    thermal_marker: str,
    volume_marker: str,
) -> NDArray[np.int64]:
    annotation_samples = _annotation_samples(raw, thermal_marker)
    annotation_times = raw.times[annotation_samples]
    event_onsets = pd.to_numeric(trials["onset"], errors="raise").to_numpy(dtype=float)
    event_volume_onsets = pd.to_numeric(
        events.loc[events["trial_type"].eq(volume_marker), "onset"],
        errors="raise",
    ).to_numpy(dtype=float)
    if event_volume_onsets.size == 0:
        raise ValueError(f"Events table contains no '{volume_marker}' markers.")
    first_volume_sample = _annotation_samples(raw, volume_marker)[0]
    matches = match_relative_event_onsets(
        event_onsets,
        annotation_times,
        first_volume_onset_s=float(raw.times[first_volume_sample]),
        event_volume_onset_s=float(event_volume_onsets[0]),
        tolerance_s=0.002,
    )
    return annotation_samples[matches]


def _extract_broadband_trials(
    raw: mne.io.BaseRaw,
    eeg_indices: NDArray[np.int64],
    trigger_samples: NDArray[np.int64],
    config: GradientTroughIcaConfig,
) -> list[NDArray[np.float32]]:
    start_offset = _samples(config.trial_tmin_s, config.sampling_frequency_hz)
    stop_offset = _samples(config.trial_tmax_s, config.sampling_frequency_hz) + 1
    starts = trigger_samples + start_offset
    stops = trigger_samples + stop_offset
    invalid = np.flatnonzero((starts < 0) | (stops > raw.n_times))
    if invalid.size:
        raise ValueError(f"Thermal epochs extend outside the recording: {invalid.tolist()}")
    trials = [
        raw.get_data(picks=eeg_indices, start=int(start), stop=int(stop)).astype(np.float32)
        for start, stop in zip(starts, stops, strict=True)
    ]
    if not all(np.isfinite(trial).all() for trial in trials):
        raise ValueError("A broadband EEG trial contains non-finite samples.")
    return trials


def _selection_audit(
    run_id: int,
    intervals: tuple[tuple[int, int, int, int], ...],
    windows: tuple[PlateauWindow, ...],
    selected_offset: int,
) -> tuple[list[tuple[float, ...]], int]:
    rows: list[tuple[float, ...]] = []
    offset = selected_offset
    for start, stop, volume_number, trough_number in intervals:
        window = windows[trough_number - 1]
        rows.append(
            (
                run_id,
                volume_number,
                trough_number,
                start + 1,
                stop,
                offset + 1,
                offset + stop - start,
                window.trough_index,
                window.threshold,
            )
        )
        offset += stop - start
    return rows, offset


def _selection_qc_run(
    run_id: int,
    trace_uv: NDArray[np.float64],
    windows: tuple[PlateauWindow, ...],
    sampling_frequency_hz: float,
) -> dict[str, Any]:
    scale = 1_000.0 / sampling_frequency_hz
    return {
        "run_id": run_id,
        "times_ms": np.arange(trace_uv.size, dtype=float) * scale,
        "mean_rectified_ecg_uv": trace_uv,
        "plateau_start_ms": np.asarray([window.start * scale for window in windows]),
        "plateau_stop_ms": np.asarray([window.stop * scale for window in windows]),
        "trough_latency_ms": np.asarray([window.trough_index * scale for window in windows]),
        "threshold_uv": np.asarray([window.threshold for window in windows]),
    }


def _electrode_positions(
    raw: mne.io.BaseRaw,
    eeg_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    positions = np.vstack([raw.info["chs"][int(index)]["loc"][:3] for index in eeg_indices])
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0.0):
        raise ValueError("Every EEG channel must have a finite non-zero electrode position.")
    return positions


def _fieldtrip_data(
    trials: Sequence[NDArray[np.floating[Any]]],
    channel_names: tuple[str, ...],
    sampling_frequency_hz: float,
    *,
    time_start_s: float = 0.0,
    trialinfo: NDArray[np.float64] | None = None,
    trialinfo_labels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not trials:
        raise ValueError("FieldTrip data requires at least one trial.")
    trial_cells = np.empty((1, len(trials)), dtype=object)
    time_cells = np.empty((1, len(trials)), dtype=object)
    sampleinfo = np.empty((len(trials), 2), dtype=np.int64)
    logical_offset = 0
    for index, trial in enumerate(trials):
        if trial.ndim != 2 or trial.shape[0] != len(channel_names):
            raise ValueError(f"Unexpected FieldTrip trial shape: {trial.shape}")
        trial_cells[0, index] = trial
        time_cells[0, index] = time_start_s + np.arange(trial.shape[1]) / sampling_frequency_hz
        sampleinfo[index] = (logical_offset + 1, logical_offset + trial.shape[1])
        logical_offset += trial.shape[1]
    result: dict[str, Any] = {
        "label": _matlab_strings(channel_names),
        "trial": trial_cells,
        "time": time_cells,
        "fsample": sampling_frequency_hz,
        "sampleinfo": sampleinfo,
    }
    if trialinfo is not None:
        if trialinfo.shape[0] != len(trials) or trialinfo_labels is None:
            raise ValueError("FieldTrip trialinfo must align with all data trials.")
        result["trialinfo"] = trialinfo
        result["trialinfo_labels"] = _matlab_strings(trialinfo_labels)
    return result


def _fieldtrip_electrodes(
    channel_names: tuple[str, ...],
    positions: NDArray[np.float64],
) -> dict[str, Any]:
    return {
        "label": _matlab_strings(channel_names),
        "chanpos": positions,
        "elecpos": positions,
        "unit": "m",
        "coordsys": "ctf",
    }


def _numeric_trialinfo(metadata: pd.DataFrame) -> NDArray[np.float64]:
    return np.column_stack(
        [pd.to_numeric(metadata[column], errors="raise") for column in TRIALINFO_COLUMNS]
    ).astype(np.float64)


def _dataframe_structure(dataframe: pd.DataFrame) -> dict[str, NDArray[Any]]:
    result: dict[str, NDArray[Any]] = {}
    for column in dataframe.columns:
        series = dataframe[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            result[column] = pd.to_numeric(series, errors="raise").to_numpy().reshape(-1, 1)
        else:
            result[column] = series.fillna("").astype(str).to_numpy(dtype=object).reshape(-1, 1)
    return result


def _qc_structure(qc_runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"run_id": np.asarray([run["run_id"] for run in qc_runs])}
    for key in qc_runs[0]:
        if key == "run_id":
            continue
        cells = np.empty((1, len(qc_runs)), dtype=object)
        for index, run in enumerate(qc_runs):
            cells[0, index] = run[key]
        result[key] = cells
    return result


def _matlab_strings(values: Sequence[str]) -> NDArray[np.object_]:
    return np.asarray(values, dtype=object).reshape(-1, 1)


def _samples(duration_s: float, sampling_frequency_hz: float) -> int:
    samples = duration_s * sampling_frequency_hz
    rounded = round(samples)
    if abs(samples - rounded) > 1e-9:
        raise ValueError(f"Duration {duration_s:g} s does not map to an integer sample count.")
    return int(rounded)


def _write_matlab(path: Path, variable_name: str, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        savemat(
            temporary_path,
            {variable_name: value},
            do_compression=False,
            long_field_names=True,
            oned_as="column",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--participant", action="append")
    return parser.parse_args()


def main() -> None:
    """Run the strict export for configured or explicitly selected participants."""
    arguments = _parse_args()
    config = GradientTroughIcaConfig.load(arguments.config)
    participants = tuple(arguments.participant or config.participants)
    unknown = sorted(set(participants) - set(config.participants))
    if unknown:
        raise ValueError(f"Participants are not configured: {unknown}")
    exports = tuple(export_participant(config, participant) for participant in participants)
    configured_exports = {
        configured_participant: config.output_root
        / "exports"
        / configured_participant
        / f"{configured_participant}_task-{TASK}_desc-troughica_data.mat"
        for configured_participant in config.participants
    }
    if all(path.is_file() for path in configured_exports.values()):
        write_runtime_manifest(config, configured_exports)
    for path in exports:
        print(path.resolve())


if __name__ == "__main__":
    main()
