"""Export validated pre-ICA MNE data to FieldTrip MATLAB structures."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd
import yaml
from scipy.io import savemat

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "config" / "fieldtrip_tfr.yaml"
REQUIRED_EVENT_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
)
TRIALINFO_COLUMNS = REQUIRED_EVENT_COLUMNS


@dataclass(frozen=True)
class ExportConfig:
    """Validated exporter configuration."""

    raw: dict[str, Any]
    fieldtrip_path: Path
    bids_root: Path
    derivatives_root: Path
    output_root: Path
    task: str
    excluded_subjects: frozenset[str]
    expected_runs: tuple[int, ...]
    trials_per_run: int
    event_name: str
    onset_tolerance_s: float
    tmin_s: float
    tmax_s: float
    ica_highpass_hz: float

    @classmethod
    def load(cls, path: Path) -> "ExportConfig":
        """Load and validate the YAML configuration."""
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration must contain a YAML mapping: {path}")

        paths = _require_mapping(raw, "paths")
        study = _require_mapping(raw, "study")
        epochs = _require_mapping(raw, "epochs")
        ica = _require_mapping(raw, "ica")
        _require_mapping(raw, "tfr")

        config = cls(
            raw=raw,
            fieldtrip_path=Path(_require_text(paths, "fieldtrip")).expanduser(),
            bids_root=Path(_require_text(paths, "bids_eeg")).expanduser(),
            derivatives_root=Path(_require_text(paths, "mne_derivatives")).expanduser(),
            output_root=Path(_require_text(paths, "output")).expanduser(),
            task=_require_text(study, "task"),
            excluded_subjects=frozenset(str(value) for value in study["excluded_subjects"]),
            expected_runs=tuple(int(value) for value in study["expected_runs"]),
            trials_per_run=int(study["trials_per_run"]),
            event_name=_require_text(study, "event_name"),
            onset_tolerance_s=float(study["event_onset_tolerance_s"]),
            tmin_s=float(epochs["tmin_s"]),
            tmax_s=float(epochs["tmax_s"]),
            ica_highpass_hz=float(ica["highpass_hz"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate paths, ranges, and fixed study assumptions."""
        for label, path in (
            ("FieldTrip", self.fieldtrip_path),
            ("BIDS EEG", self.bids_root),
            ("MNE derivatives", self.derivatives_root),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"{label} directory does not exist: {path}")
        if not (self.fieldtrip_path / "ft_defaults.m").is_file():
            raise FileNotFoundError(
                f"FieldTrip ft_defaults.m does not exist under {self.fieldtrip_path}."
            )
        if not self.expected_runs:
            raise ValueError("study.expected_runs must contain at least one run.")
        if self.trials_per_run < 1:
            raise ValueError("study.trials_per_run must be positive.")
        if self.tmin_s >= self.tmax_s:
            raise ValueError("epochs.tmin_s must be earlier than epochs.tmax_s.")
        if self.ica_highpass_hz <= 0:
            raise ValueError("ica.highpass_hz must be positive.")


@dataclass(frozen=True)
class SubjectExport:
    """Prepared data and metadata for one participant."""

    subject: str
    export_id: str
    broadband_epochs: mne.Epochs
    ica_epochs: mne.Epochs
    trial_metadata: pd.DataFrame
    bad_channels: tuple[str, ...]
    source_files: tuple[Path, ...]


def discover_subjects(config: ExportConfig) -> list[str]:
    """Return every eligible participant with a derivatives directory."""
    subjects = sorted(
        path.name
        for path in config.derivatives_root.glob("sub-*")
        if path.is_dir() and path.name not in config.excluded_subjects
    )
    if not subjects:
        raise FileNotFoundError(
            f"No eligible subject directories exist under {config.derivatives_root}."
        )
    return subjects


def prepare_subject(config: ExportConfig, subject: str) -> SubjectExport:
    """Create aligned broadband and ICA-fit epochs for one participant."""
    subject = _normalize_subject(subject)
    if subject in config.excluded_subjects:
        raise ValueError(f"Participant is excluded by configuration: {subject}")

    run_epochs: list[mne.Epochs] = []
    metadata_frames: list[pd.DataFrame] = []
    source_files: list[Path] = []
    reference_channels: tuple[str, ...] | None = None
    reference_bads: tuple[str, ...] | None = None

    for run in config.expected_runs:
        raw_path = _resolve_raw_path(config, subject, run)
        events_path = _resolve_events_path(config, subject, run)
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        metadata = _load_run_metadata(config, events_path, run)
        run_event_samples = _match_event_samples(config, raw, metadata, raw_path)
        epochs = _epoch_run(config, raw, run_event_samples, metadata)

        channels = tuple(epochs.ch_names)
        bads = tuple(sorted(raw.info["bads"]))
        if reference_channels is None:
            reference_channels = channels
            reference_bads = bads
        else:
            _require_equal(channels, reference_channels, f"channel order for {raw_path}")
            _require_equal(bads, reference_bads, f"bad-channel set for {raw_path}")

        run_epochs.append(epochs)
        metadata_frames.append(metadata)
        source_files.extend((raw_path, events_path))

    broadband_epochs = mne.concatenate_epochs(run_epochs, add_offset=True, verbose=False)
    trial_metadata = pd.concat(metadata_frames, ignore_index=True)
    broadband_epochs.metadata = trial_metadata.copy()
    _validate_subject_epochs(config, broadband_epochs, trial_metadata)

    ica_epochs = broadband_epochs.copy().filter(
        l_freq=config.ica_highpass_hz,
        h_freq=None,
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_design="firwin",
        verbose=False,
    )
    _require_finite(ica_epochs.get_data(copy=False), f"{subject} ICA-fit epochs")

    return SubjectExport(
        subject=subject,
        export_id=str(uuid.uuid4()),
        broadband_epochs=broadband_epochs,
        ica_epochs=ica_epochs,
        trial_metadata=trial_metadata,
        bad_channels=reference_bads or (),
        source_files=tuple(source_files),
    )


def export_subject(config: ExportConfig, prepared: SubjectExport, *, overwrite: bool) -> Path:
    """Write one participant package atomically."""
    output_dir = config.output_root / "exports" / prepared.subject
    output_path = output_dir / f"{prepared.subject}_task-{config.task}_desc-preica_fieldtrip.mat"
    provenance_path = output_path.with_suffix(".json")
    if (output_path.exists() or provenance_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Export already exists for {prepared.subject}: {output_path}. Use --overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    package = _build_matlab_package(config, prepared)
    provenance = _build_provenance(config, prepared, output_path)

    temporary_path = _temporary_output_path(output_dir, output_path.name)
    temporary_provenance = temporary_path.with_suffix(".json")
    try:
        savemat(
            temporary_path,
            package,
            appendmat=False,
            do_compression=True,
            long_field_names=True,
            oned_as="row",
        )
        temporary_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary_path, output_path)
        os.replace(temporary_provenance, provenance_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)

    return output_path


def write_runtime_config(config: ExportConfig) -> Path:
    """Write the resolved configuration consumed by MATLAB."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    runtime_path = config.output_root / "fieldtrip_tfr_runtime.json"
    resolved = dict(config.raw)
    resolved_paths = dict(resolved["paths"])
    resolved_paths.update({
        "fieldtrip": str(config.fieldtrip_path.resolve()),
        "bids_eeg": str(config.bids_root.resolve()),
        "mne_derivatives": str(config.derivatives_root.resolve()),
        "output": str(config.output_root.resolve()),
    })
    resolved["paths"] = resolved_paths
    runtime_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return runtime_path


def _resolve_raw_path(config: ExportConfig, subject: str, run: int) -> Path:
    subject_eeg = config.derivatives_root / subject / "eeg"
    pattern = f"{subject}_task-{config.task}_run-{run}_proc-filt_raw.fif"
    return _require_file(subject_eeg / pattern, "pre-ICA filtered raw")


def _resolve_events_path(config: ExportConfig, subject: str, run: int) -> Path:
    subject_eeg = config.bids_root / subject / "eeg"
    pattern = f"{subject}_task-{config.task}_run-{run}_events.tsv"
    return _require_file(subject_eeg / pattern, "BIDS events")


def _load_run_metadata(config: ExportConfig, path: Path, run: int) -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t")
    missing_columns = sorted(set(REQUIRED_EVENT_COLUMNS) - set(events.columns))
    if missing_columns:
        raise ValueError(f"{path} lacks required columns: {missing_columns}")

    trial_mask = events["trial_type"].eq(config.event_name) & events["stimulus_temp"].notna()
    trials = events.loc[trial_mask].copy()
    if len(trials) != config.trials_per_run:
        raise ValueError(
            f"{path} contains {len(trials)} experimental thermal trials; expected "
            f"{config.trials_per_run}."
        )
    trials["run_id"] = pd.to_numeric(trials["run_id"], errors="raise").astype(int)
    trials["trial_number"] = pd.to_numeric(
        trials["trial_number"], errors="raise"
    ).astype(int)
    if not trials["run_id"].eq(run).all():
        raise ValueError(f"{path} contains a run_id other than {run}.")
    if trials["trial_number"].duplicated().any():
        raise ValueError(f"{path} contains duplicated trial_number values.")
    expected_trial_numbers = list(range(1, config.trials_per_run + 1))
    if trials["trial_number"].tolist() != expected_trial_numbers:
        raise ValueError(
            f"{path} trial_number values must be ordered {expected_trial_numbers}."
        )
    return trials.reset_index(drop=True)


def _match_event_samples(
    config: ExportConfig,
    raw: mne.io.BaseRaw,
    metadata: pd.DataFrame,
    raw_path: Path,
) -> np.ndarray:
    annotation_onsets = np.array(
        [
            annotation["onset"]
            for annotation in raw.annotations
            if annotation["description"] == config.event_name
        ],
        dtype=float,
    )
    if annotation_onsets.size < len(metadata):
        raise ValueError(
            f"{raw_path} contains only {annotation_onsets.size} {config.event_name!r} annotations "
            f"for {len(metadata)} BIDS trials."
        )

    bids_onsets = pd.to_numeric(metadata["onset"], errors="raise").to_numpy(dtype=float)
    samples = np.empty(len(bids_onsets), dtype=int)
    matched_indices: set[int] = set()
    for trial_index, onset in enumerate(bids_onsets):
        differences = np.abs(annotation_onsets - onset)
        annotation_index = int(np.argmin(differences))
        difference = float(differences[annotation_index])
        if difference > config.onset_tolerance_s:
            raise ValueError(
                f"{raw_path} has no {config.event_name!r} annotation within "
                f"{config.onset_tolerance_s:.6f} s of BIDS onset {onset:.6f} s."
            )
        if annotation_index in matched_indices:
            raise ValueError(f"{raw_path} maps multiple BIDS trials to one annotation.")
        matched_indices.add(annotation_index)
        samples[trial_index] = raw.time_as_index(annotation_onsets[annotation_index])[0]
    return samples


def _epoch_run(
    config: ExportConfig,
    raw: mne.io.BaseRaw,
    samples: np.ndarray,
    metadata: pd.DataFrame,
) -> mne.Epochs:
    events = np.column_stack(
        [samples + raw.first_samp, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)]
    )
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={config.event_name: 1},
        tmin=config.tmin_s,
        tmax=config.tmax_s,
        baseline=None,
        picks=mne.pick_types(raw.info, eeg=True, exclude=[]),
        preload=True,
        reject_by_annotation=True,
        metadata=metadata,
        on_missing="raise",
        verbose=False,
    )
    if len(epochs) != len(metadata):
        dropped = [index for index, reason in enumerate(epochs.drop_log) if reason]
        raise ValueError(
            f"Epoch construction retained {len(epochs)}/{len(metadata)} trials; dropped indices: "
            f"{dropped}."
        )
    return epochs


def _validate_subject_epochs(
    config: ExportConfig,
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
) -> None:
    expected_trials = len(config.expected_runs) * config.trials_per_run
    if len(epochs) != expected_trials:
        raise ValueError(f"Prepared {len(epochs)} trials; expected {expected_trials}.")
    identifiers = metadata[["run_id", "trial_number"]]
    if identifiers.duplicated().any():
        raise ValueError("Prepared trial metadata contains duplicate run/trial identifiers.")
    _require_finite(epochs.get_data(copy=False), "broadband epochs")


def _build_matlab_package(
    config: ExportConfig,
    prepared: SubjectExport,
) -> dict[str, Any]:
    metadata = prepared.trial_metadata
    trialinfo = np.column_stack(
        [pd.to_numeric(metadata[column], errors="coerce") for column in TRIALINFO_COLUMNS]
    )
    common = {
        "subject": prepared.subject,
        "export_id": prepared.export_id,
        "trialinfo": trialinfo,
        "trialinfo_labels": np.asarray(TRIALINFO_COLUMNS, dtype=object),
        "bad_channels": np.asarray(prepared.bad_channels, dtype=object),
        "ica_channels": np.asarray(
            [
                channel
                for channel in prepared.broadband_epochs.ch_names
                if channel not in prepared.bad_channels
            ],
            dtype=object,
        ),
    }
    return {
        "broadband_data": _epochs_to_fieldtrip(prepared.broadband_epochs),
        "ica_data": _epochs_to_fieldtrip(prepared.ica_epochs),
        "metadata": common,
        "runtime_config_json": json.dumps(config.raw),
    }


def _epochs_to_fieldtrip(epochs: mne.Epochs) -> dict[str, Any]:
    data = epochs.get_data(copy=False).astype(np.float32, copy=False)
    trial_cells = np.empty((1, len(epochs)), dtype=object)
    time_cells = np.empty((1, len(epochs)), dtype=object)
    times = epochs.times.astype(np.float64, copy=False)
    for trial_index in range(len(epochs)):
        trial_cells[0, trial_index] = data[trial_index]
        time_cells[0, trial_index] = times

    channel_positions = np.vstack(
        [epochs.info["chs"][index]["loc"][:3] for index in range(len(epochs.ch_names))]
    )
    if not np.isfinite(channel_positions).all():
        raise ValueError("EEG channel positions contain non-finite values.")
    elec = {
        "label": np.asarray(epochs.ch_names, dtype=object),
        "chanpos": channel_positions,
        "elecpos": channel_positions,
        "unit": "m",
    }
    sample_count = len(times)
    sampleinfo = np.column_stack(
        [
            np.arange(0, len(epochs) * sample_count, sample_count) + 1,
            np.arange(1, len(epochs) + 1) * sample_count,
        ]
    )
    return {
        "label": np.asarray(epochs.ch_names, dtype=object),
        "trial": trial_cells,
        "time": time_cells,
        "fsample": float(epochs.info["sfreq"]),
        "sampleinfo": sampleinfo,
        "elec": elec,
    }


def _build_provenance(
    config: ExportConfig,
    prepared: SubjectExport,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "subject": prepared.subject,
        "task": config.task,
        "export_id": prepared.export_id,
        "output": str(output_path),
        "trial_count": len(prepared.broadband_epochs),
        "channel_count": len(prepared.broadband_epochs.ch_names),
        "bad_channels": list(prepared.bad_channels),
        "ica_channels": [
            channel
            for channel in prepared.broadband_epochs.ch_names
            if channel not in prepared.bad_channels
        ],
        "sampling_frequency_hz": prepared.broadband_epochs.info["sfreq"],
        "epoch_window_s": [prepared.broadband_epochs.tmin, prepared.broadband_epochs.tmax],
        "source_files": [str(path) for path in prepared.source_files],
    }


def _temporary_output_path(directory: Path, filename: str) -> Path:
    descriptor, path = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    temporary_path = Path(path)
    temporary_path.unlink()
    return temporary_path


def _normalize_subject(subject: str) -> str:
    value = subject.strip()
    return value if value.startswith("sub-") else f"sub-{value}"


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field {key!r} must be a mapping.")
    return value


def _require_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field {key!r} must be non-empty text.")
    return value.strip()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} file does not exist: {path}")
    return path


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Inconsistent {label}: {actual!r} != {expected!r}")


def _require_finite(data: np.ndarray, label: str) -> None:
    if not np.isfinite(data).all():
        raise ValueError(f"{label} contain non-finite samples.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pre-ICA Study 1 EEG data to FieldTrip MATLAB files."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--subject", action="append", dest="subjects")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation or export for the selected study cohort."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    config = ExportConfig.load(args.config.resolve())
    subjects = (
        [_normalize_subject(value) for value in args.subjects]
        if args.subjects
        else discover_subjects(config)
    )
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subjects must not be repeated.")

    for subject in subjects:
        LOGGER.info("Preparing %s", subject)
        prepared = prepare_subject(config, subject)
        if args.validate_only:
            LOGGER.info(
                "Validated %s: %d trials, %d channels, %d ICA channels",
                subject,
                len(prepared.broadband_epochs),
                len(prepared.broadband_epochs.ch_names),
                len(prepared.broadband_epochs.ch_names) - len(prepared.bad_channels),
            )
            continue
        output = export_subject(config, prepared, overwrite=args.overwrite)
        LOGGER.info("Exported %s", output)

    if not args.validate_only:
        runtime_path = write_runtime_config(config)
        LOGGER.info("Wrote MATLAB runtime configuration %s", runtime_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
