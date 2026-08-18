"""Export sub-0015 BrainVision trials and aligned metadata for MATLAB."""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd
from scipy.io import savemat

from studies.pain_study.scripts.conversion.brainvision_markers import validate_unambiguous_vas_markers

SUBJECT = "sub-0015"
TASK = "thermalactive"
RUN_IDS = tuple(range(1, 7))
TRIALS_PER_RUN = 11
THERMAL_TRIGGER = "Trig_therm/T  1"
SAMPLING_FREQUENCY = 1_000.0
EPOCH_TMIN_S = -7.0
EPOCH_TMAX_S = 15.0
ONSET_TOLERANCE_S = 0.002
EXPECTED_CHANNEL_COUNT = 64
TRIALINFO_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
)
FIELDTRIP_TRIALINFO_COLUMNS = (
    *TRIALINFO_COLUMNS,
    "trial_onset_relative_to_first_volume_s",
)
PROCESSED_EXPORT_SUFFIXES = ("scannerpulse_corrected", "scanner_artifact_step2")
"""Analyzer's names for the final node of a processed export, oldest first.

The export that recovers Analyzer's missed beats ends at `scanner_artifact_step2`; the one
before it ended at `scannerpulse_corrected`. Both are searched, and finding a run under
more than one of them is still an error rather than a preference order.
"""
DEFAULT_SOURCE_DIRECTORY = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v2/"
    "sub-0015/eeg/brainvision_processed_1khz"
)
DEFAULT_EVENTS_DIRECTORY = Path("/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg/sub-0015/eeg")
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/matlab_exports/sub-0015/brainvision_processed_1khz")
DEFAULT_CLEAN_MNE_DIRECTORY = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/"
    "brainvision_analyzer_mne_preprocessing_sub-0015/preprocessed/eeg/sub-0015/eeg"
)


def select_trials(events: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Select and validate one run's ordered thermal trials."""
    required_columns = {"trial_type", *TRIALINFO_COLUMNS}
    missing_columns = sorted(required_columns - set(events.columns))
    if missing_columns:
        raise ValueError(f"Events table lacks required columns: {missing_columns}")

    thermal_mask = events["trial_type"].eq(THERMAL_TRIGGER)
    temperature_mask = events["stimulus_temp"].notna()
    trials = events.loc[thermal_mask & temperature_mask].copy()
    if len(trials) != TRIALS_PER_RUN:
        raise ValueError(
            f"Run {run_id} contains {len(trials)} thermal trials; expected 11 thermal trials."
        )

    trials["run_id"] = pd.to_numeric(trials["run_id"], errors="raise").astype(int)
    trials["trial_number"] = pd.to_numeric(trials["trial_number"], errors="raise").astype(int)
    if not trials["run_id"].eq(run_id).all():
        raise ValueError(f"Run {run_id} events contain a different run_id.")

    expected_trial_numbers = list(range(1, TRIALS_PER_RUN + 1))
    if trials["trial_number"].tolist() != expected_trial_numbers:
        raise ValueError(
            f"Run {run_id} trial_number values must be ordered {expected_trial_numbers}."
        )
    return trials.reset_index(drop=True)


def epoch_bounds(
    trigger_samples: np.ndarray,
    sampling_frequency: float,
    recording_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive epoch bounds as half-open sample intervals."""
    if not np.isclose(sampling_frequency, SAMPLING_FREQUENCY, atol=1e-12):
        raise ValueError(f"Expected {SAMPLING_FREQUENCY:g} Hz, got {sampling_frequency:g} Hz.")

    start_offset = int(round(EPOCH_TMIN_S * sampling_frequency))
    stop_offset = int(round(EPOCH_TMAX_S * sampling_frequency)) + 1
    starts = np.asarray(trigger_samples, dtype=np.int64) + start_offset
    stops = np.asarray(trigger_samples, dtype=np.int64) + stop_offset
    invalid = np.flatnonzero((starts < 0) | (stops > recording_samples))
    if invalid.size:
        raise ValueError(
            f"Epochs extend outside the recording at trial indices {invalid.tolist()}."
        )
    return starts, stops


def align_trial_onsets(
    bids_onsets: np.ndarray,
    annotation_onsets: np.ndarray,
) -> np.ndarray:
    """Match BIDS trial onsets to annotations on the source recording clock."""
    expected_onsets = np.asarray(bids_onsets, dtype=float)
    source_onsets = np.asarray(annotation_onsets, dtype=float)
    matched_indices: set[int] = set()
    aligned_onsets = np.empty(expected_onsets.size, dtype=float)
    for trial_index, expected_onset in enumerate(expected_onsets):
        differences = np.abs(source_onsets - expected_onset)
        annotation_index = int(np.argmin(differences))
        difference = float(differences[annotation_index])
        if difference > ONSET_TOLERANCE_S:
            raise ValueError(
                f"Trial {trial_index + 1} has no thermode annotation within "
                f"{ONSET_TOLERANCE_S:g} s; nearest difference is {difference:g} s."
            )
        if annotation_index in matched_indices:
            raise ValueError("Multiple trials matched the same thermode annotation.")
        matched_indices.add(annotation_index)
        aligned_onsets[trial_index] = source_onsets[annotation_index]
    return aligned_onsets


def relative_volume_times(volume_onsets: np.ndarray) -> np.ndarray:
    """Express ordered volume onsets relative to the first volume marker."""
    onsets = np.asarray(volume_onsets, dtype=float)
    if onsets.ndim != 1 or onsets.size == 0:
        raise ValueError("Volume onsets must be a non-empty one-dimensional array.")
    if not np.isfinite(onsets).all():
        raise ValueError("Volume onsets contain non-finite values.")
    if np.any(np.diff(onsets) <= 0):
        raise ValueError("Volume onsets must be strictly increasing.")
    return onsets - onsets[0]


def _require_run_header(source_directory: Path, run_id: int) -> Path:
    headers = sorted(
        header
        for suffix in PROCESSED_EXPORT_SUFFIXES
        for header in source_directory.glob(f"ThermalPainEEGFMRI_run{run_id}_*_{suffix}.vhdr")
        if not header.name.startswith("._")
    )
    if len(headers) != 1:
        raise FileNotFoundError(
            f"Expected one processed BrainVision header for run {run_id}, found {len(headers)}."
        )
    header = headers[0]
    for suffix in (".eeg", ".vmrk"):
        referenced_file = header.with_suffix(suffix)
        if not referenced_file.is_file():
            raise FileNotFoundError(f"BrainVision triplet is incomplete: {referenced_file}")
    return header


def _require_events_path(events_directory: Path, run_id: int) -> Path:
    path = events_directory / f"{SUBJECT}_task-{TASK}_run-{run_id}_events.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Run {run_id} events table does not exist: {path}")
    return path


def _match_trial_samples(raw: mne.io.BaseRaw, trials: pd.DataFrame) -> np.ndarray:
    validate_unambiguous_vas_markers(raw)
    annotation_onsets = np.asarray(
        [
            annotation["onset"]
            for annotation in raw.annotations
            if annotation["description"] == THERMAL_TRIGGER
        ],
        dtype=float,
    )
    if annotation_onsets.size < len(trials):
        raise ValueError(
            f"Recording contains {annotation_onsets.size} thermode annotations for "
            f"{len(trials)} trials."
        )

    volume_onsets = np.asarray(
        [
            annotation["onset"]
            for annotation in raw.annotations
            if annotation["description"] == "Volume/V  1"
        ],
        dtype=float,
    )
    if volume_onsets.size == 0:
        raise ValueError("Recording contains no scanner-volume annotation.")

    bids_onsets = pd.to_numeric(trials["onset"], errors="raise").to_numpy(dtype=float)
    aligned_onsets = align_trial_onsets(
        bids_onsets,
        annotation_onsets,
    )
    return raw.time_as_index(aligned_onsets, use_rounding=True).astype(np.int64)


def _extract_run_epochs(
    raw: mne.io.BaseRaw,
    trigger_samples: np.ndarray,
) -> list[np.ndarray]:
    starts, stops = epoch_bounds(
        trigger_samples,
        sampling_frequency=float(raw.info["sfreq"]),
        recording_samples=raw.n_times,
    )
    epochs = []
    for start, stop in zip(starts, stops, strict=True):
        epoch = raw.get_data(start=int(start), stop=int(stop)).astype(np.float32)
        if epoch.shape != (EXPECTED_CHANNEL_COUNT, 22_001):
            raise ValueError(f"Unexpected epoch shape: {epoch.shape}")
        if not np.isfinite(epoch).all():
            raise ValueError("EEG epoch contains non-finite samples.")
        epochs.append(epoch)
    return epochs


def _volume_timing_frame(
    raw: mne.io.BaseRaw,
    run_id: int,
    header_path: Path,
) -> pd.DataFrame:
    volume_onsets = np.asarray(
        [
            annotation["onset"]
            for annotation in raw.annotations
            if annotation["description"] == "Volume/V  1"
        ],
        dtype=float,
    )
    relative_times = relative_volume_times(volume_onsets)
    zero_based_samples = raw.time_as_index(volume_onsets, use_rounding=True).astype(np.int64)
    return pd.DataFrame(
        {
            "run_id": run_id,
            "volume_number": np.arange(1, volume_onsets.size + 1, dtype=np.int64),
            "source_sample_zero_based": zero_based_samples,
            "source_sample_matlab": zero_based_samples + 1,
            "source_recording_time_s": volume_onsets,
            "relative_time_to_first_volume_s": relative_times,
            "source_brainvision_header": str(header_path),
        }
    )


def numeric_trialinfo(metadata: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the concise analysis columns used as FieldTrip trialinfo."""
    labels = FIELDTRIP_TRIALINFO_COLUMNS
    missing_columns = sorted(set(labels) - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"Trial metadata lacks FieldTrip columns: {missing_columns}")
    values = np.column_stack(
        [pd.to_numeric(metadata[column], errors="raise") for column in labels]
    ).astype(np.float64)
    return values, labels


def _fieldtrip_data(
    epochs: Sequence[np.ndarray],
    channel_names: Sequence[str],
    metadata: pd.DataFrame,
    times: np.ndarray,
    sampling_frequency: float,
    volume_timing: pd.DataFrame,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    trial_cells = np.empty((1, len(epochs)), dtype=object)
    time_cells = np.empty((1, len(epochs)), dtype=object)
    times = np.asarray(times, dtype=np.float64)
    for trial_index, epoch in enumerate(epochs):
        if epoch.shape != (len(channel_names), times.size):
            raise ValueError(f"Unexpected FieldTrip epoch shape: {epoch.shape}")
        if not np.isfinite(epoch).all():
            raise ValueError(f"FieldTrip epoch {trial_index} contains non-finite samples.")
        trial_cells[0, trial_index] = epoch
        time_cells[0, trial_index] = times

    epoch_samples = times.size
    sampleinfo = np.column_stack(
        (
            np.arange(len(epochs), dtype=np.int64) * epoch_samples + 1,
            np.arange(1, len(epochs) + 1, dtype=np.int64) * epoch_samples,
        )
    )
    channel_units = np.asarray(["V"] * len(channel_names), dtype=object).reshape(-1, 1)
    channel_types = np.asarray(
        ["ecg" if name == "ECG" else "eeg" for name in channel_names], dtype=object
    ).reshape(-1, 1)
    trialinfo, trialinfo_labels = numeric_trialinfo(metadata)
    return {
        "label": np.asarray(channel_names, dtype=object).reshape(-1, 1),
        "trial": trial_cells,
        "time": time_cells,
        "fsample": sampling_frequency,
        "sampleinfo": sampleinfo,
        "trialinfo": trialinfo,
        "trialinfo_labels": np.asarray(trialinfo_labels, dtype=object).reshape(-1, 1),
        "volume_timing": _dataframe_structure(volume_timing),
        "provenance": provenance,
        "hdr": {
            "nChans": len(channel_names),
            "nSamples": epoch_samples,
            "nSamplesPre": int(round(max(0.0, -times[0]) * sampling_frequency)),
            "nTrials": len(epochs),
            "Fs": sampling_frequency,
            "label": np.asarray(channel_names, dtype=object).reshape(-1, 1),
            "chantype": channel_types,
            "chanunit": channel_units,
        },
    }


def _matlab_column(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64).reshape(-1, 1)
    return series.fillna("").astype(str).to_numpy(dtype=object).reshape(-1, 1)


def _dataframe_structure(dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
    return {column: _matlab_column(dataframe[column]) for column in dataframe.columns}


def _trial_info(
    metadata: pd.DataFrame,
    volume_timing: pd.DataFrame,
    export_timestamp: str,
) -> dict[str, Any]:
    trial_info = _dataframe_structure(metadata)
    trial_info.update(
        {
            "volume_timing": _dataframe_structure(volume_timing),
            "subject": SUBJECT,
            "task": TASK,
            "event_name": THERMAL_TRIGGER,
            "epoch_tmin_s": EPOCH_TMIN_S,
            "epoch_tmax_s": EPOCH_TMAX_S,
            "sampling_frequency_hz": SAMPLING_FREQUENCY,
            "data_unit": "V",
            "export_timestamp_utc": export_timestamp,
        }
    )
    return trial_info


def _write_matlab(path: Path, variable_name: str, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {path}")

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


def _export_clean_mne_epochs(
    clean_directory: Path,
    output_directory: Path,
    volume_timing: pd.DataFrame,
    export_timestamp: str,
) -> Path:
    epochs_path = clean_directory / f"{SUBJECT}_task-{TASK}_proc-clean_epo.fif"
    events_path = clean_directory / f"{SUBJECT}_task-{TASK}_proc-clean_events.tsv"
    for path in (epochs_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(f"Clean MNE derivative does not exist: {path}")

    clean_epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
    clean_events = pd.read_csv(events_path, sep="\t")
    if len(clean_epochs) != len(clean_events):
        raise ValueError(
            f"Clean MNE epochs/events mismatch: {len(clean_epochs)} and {len(clean_events)}."
        )
    expected_epoch_indices = np.arange(len(clean_events))
    epoch_indices = pd.to_numeric(clean_events["epoch_index"], errors="raise").to_numpy()
    if not np.array_equal(epoch_indices, expected_epoch_indices):
        raise ValueError("Clean events epoch_index is not ordered from zero.")
    if not clean_events["trial_type"].eq(THERMAL_TRIGGER).all():
        raise ValueError("Clean events contain a non-thermal trial_type.")

    first_volume_by_run = (
        volume_timing.loc[volume_timing["volume_number"].eq(1)]
        .set_index("run_id")["source_recording_time_s"]
        .to_dict()
    )
    run_ids = pd.to_numeric(clean_events["run_id"], errors="raise").astype(int)
    clean_events["trial_onset_relative_to_first_volume_s"] = pd.to_numeric(
        clean_events["onset"], errors="raise"
    ) - run_ids.map(first_volume_by_run)
    if clean_events["trial_onset_relative_to_first_volume_s"].isna().any():
        raise ValueError("Clean events contain a run without first-volume timing.")

    data = clean_epochs.get_data(copy=False).astype(np.float32, copy=False)
    output_path = output_directory / (f"{SUBJECT}_task-{TASK}_desc-icaclean_fieldtrip.mat")
    _write_matlab(
        output_path,
        "data",
        _fieldtrip_data(
            data,
            clean_epochs.ch_names,
            clean_events,
            clean_epochs.times,
            float(clean_epochs.info["sfreq"]),
            volume_timing,
            {
                "processing_stage": "ICA-cleaned and AutoReject-retained MNE epochs",
                "source_mne_epochs": str(epochs_path),
                "source_clean_events": str(events_path),
                "export_timestamp_utc": export_timestamp,
                "data_unit": "V",
            },
        ),
    )
    return output_path


def export_subject(
    source_directory: Path,
    events_directory: Path,
    output_directory: Path,
    clean_mne_directory: Path,
) -> tuple[Path, Path, Path]:
    """Export all six validated runs for sub-0015."""
    if not source_directory.is_dir():
        raise FileNotFoundError(f"BrainVision source directory does not exist: {source_directory}")
    if not events_directory.is_dir():
        raise FileNotFoundError(f"Events directory does not exist: {events_directory}")

    epochs: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []
    volume_timing_frames: list[pd.DataFrame] = []
    reference_channels: tuple[str, ...] | None = None
    for run_id in RUN_IDS:
        header_path = _require_run_header(source_directory, run_id)
        events_path = _require_events_path(events_directory, run_id)
        events = pd.read_csv(events_path, sep="\t")
        trials = select_trials(events, run_id)

        raw = mne.io.read_raw_brainvision(header_path, preload=False, verbose="ERROR")
        channels = tuple(raw.ch_names)
        if len(channels) != EXPECTED_CHANNEL_COUNT:
            raise ValueError(
                f"Run {run_id} contains {len(channels)} channels; expected 64 channels."
            )
        if reference_channels is None:
            reference_channels = channels
        elif channels != reference_channels:
            raise ValueError(f"Run {run_id} channel names or ordering differ from run 1.")

        trigger_samples = _match_trial_samples(raw, trials)
        epochs.extend(_extract_run_epochs(raw, trigger_samples))
        volume_timing = _volume_timing_frame(raw, run_id, header_path)
        first_volume_onset = float(volume_timing["source_recording_time_s"].iloc[0])
        trials["trial_onset_relative_to_first_volume_s"] = (
            raw.times[trigger_samples] - first_volume_onset
        )
        trials["source_event_file"] = str(events_path)
        trials["source_brainvision_header"] = str(header_path)
        metadata_frames.append(trials)
        volume_timing_frames.append(volume_timing)

    metadata = pd.concat(metadata_frames, ignore_index=True)
    volume_timing = pd.concat(volume_timing_frames, ignore_index=True)
    if len(epochs) != len(metadata) or len(epochs) != 66:
        raise ValueError(
            f"Expected 66 aligned epochs and rows, got {len(epochs)} epochs and "
            f"{len(metadata)} rows."
        )
    identifiers = metadata[["run_id", "trial_number"]]
    if identifiers.duplicated().any():
        raise ValueError("Trial metadata contains duplicated run/trial identifiers.")
    if reference_channels is None:
        raise RuntimeError("No BrainVision runs were loaded.")

    export_timestamp = datetime.now(UTC).isoformat()
    data_path = output_directory / (f"{SUBJECT}_task-{TASK}_desc-brainvisionprocessed_epochs.mat")
    trial_info_path = output_directory / f"{SUBJECT}_task-{TASK}_trial_info.mat"
    _write_matlab(
        data_path,
        "data",
        _fieldtrip_data(
            epochs,
            reference_channels,
            metadata,
            np.arange(-7_000, 15_001, dtype=np.float64) / SAMPLING_FREQUENCY,
            SAMPLING_FREQUENCY,
            volume_timing,
            {
                "processing_stage": "BrainVision Analyzer scanner/pulse corrected",
                "export_timestamp_utc": export_timestamp,
                "data_unit": "V",
            },
        ),
    )
    _write_matlab(
        trial_info_path,
        "trial_info",
        _trial_info(metadata, volume_timing, export_timestamp),
    )
    clean_data_path = _export_clean_mne_epochs(
        clean_mne_directory,
        output_directory,
        volume_timing,
        export_timestamp,
    )
    return data_path, trial_info_path, clean_data_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--events-directory", type=Path, default=DEFAULT_EVENTS_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--clean-mne-directory", type=Path, default=DEFAULT_CLEAN_MNE_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    """Run the strict sub-0015 MATLAB export."""
    arguments = _parse_args()
    data_path, trial_info_path, clean_data_path = export_subject(
        arguments.source_directory,
        arguments.events_directory,
        arguments.output_directory,
        arguments.clean_mne_directory,
    )
    print(data_path.resolve())
    print(trial_info_path.resolve())
    print(clean_data_path.resolve())


if __name__ == "__main__":
    main()
