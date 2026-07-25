"""Export exact MNE ICA sources and rejection decisions for FieldTrip review."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd
import yaml
from scipy.io import savemat

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "config" / "mne_ica_review.yaml"
TRIAL_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
)


@dataclass(frozen=True)
class ReviewConfig:
    """Validated MNE ICA review configuration."""

    raw: dict[str, Any]
    path: Path
    fieldtrip_path: Path
    bids_root: Path
    derivatives_roots: tuple[Path, ...]
    output_root: Path
    task: str
    participants: tuple[str, ...] | None
    excluded_participants: frozenset[str]
    expected_runs: tuple[int, ...]
    trials_per_run: int
    event_name: str
    tmin_s: float
    tmax_s: float
    onset_tolerance_s: float
    resample_hz: float
    overwrite_exports: bool

    @classmethod
    def load(cls, path: Path) -> "ReviewConfig":
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")

        paths = _mapping(raw, "paths")
        study = _mapping(raw, "study")
        execution = _mapping(raw, "execution")
        _mapping(raw, "epochs")
        _mapping(raw, "tfr")
        _mapping(raw, "plots")

        roots_value = paths.get("mne_derivatives")
        if not isinstance(roots_value, list) or not roots_value:
            raise ValueError("paths.mne_derivatives must be a non-empty list.")
        participants = study.get("participants")
        if participants == "all":
            participant_ids = None
        elif isinstance(participants, list) and participants:
            participant_ids = tuple(_subject_id(value) for value in participants)
        else:
            raise ValueError("study.participants must be 'all' or a non-empty list.")

        config = cls(
            raw=raw,
            path=path,
            fieldtrip_path=Path(_text(paths, "fieldtrip")).expanduser().resolve(),
            bids_root=Path(_text(paths, "bids_eeg")).expanduser().resolve(),
            derivatives_roots=tuple(
                Path(str(value)).expanduser().resolve() for value in roots_value
            ),
            output_root=Path(_text(paths, "output")).expanduser().resolve(),
            task=_text(study, "task"),
            participants=participant_ids,
            excluded_participants=frozenset(
                _subject_id(value) for value in study["excluded_participants"]
            ),
            expected_runs=tuple(int(value) for value in study["expected_runs"]),
            trials_per_run=int(study["trials_per_run"]),
            event_name=_text(study, "event_name"),
            tmin_s=float(_mapping(raw, "epochs")["tmin_s"]),
            tmax_s=float(_mapping(raw, "epochs")["tmax_s"]),
            onset_tolerance_s=float(
                _mapping(raw, "epochs")["event_onset_tolerance_s"]
            ),
            resample_hz=float(_mapping(raw, "epochs")["component_resample_hz"]),
            overwrite_exports=bool(execution["overwrite_exports"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for label, directory in (
            ("FieldTrip", self.fieldtrip_path),
            ("BIDS EEG", self.bids_root),
        ):
            if not directory.is_dir():
                raise FileNotFoundError(f"{label} directory does not exist: {directory}")
        if not (self.fieldtrip_path / "ft_defaults.m").is_file():
            raise FileNotFoundError(
                f"FieldTrip ft_defaults.m does not exist: {self.fieldtrip_path}"
            )
        for root in self.derivatives_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"MNE derivatives root does not exist: {root}")
        if not self.expected_runs or self.trials_per_run < 1:
            raise ValueError("Expected runs and trials_per_run must be positive.")
        if self.tmin_s >= self.tmax_s:
            raise ValueError("epochs.tmin_s must be earlier than epochs.tmax_s.")
        if self.onset_tolerance_s <= 0:
            raise ValueError("epochs.event_onset_tolerance_s must be positive.")
        if self.resample_hz <= 2 * float(_mapping(self.raw, "tfr")["frequency_max_hz"]):
            raise ValueError(
                "epochs.component_resample_hz must exceed twice tfr.frequency_max_hz."
            )


def discover_subject_directories(config: ReviewConfig) -> dict[str, Path]:
    """Find exactly one MNE EEG directory for every configured participant."""
    discovered: dict[str, Path] = {}
    for root in config.derivatives_roots:
        for subject_directory in sorted(root.glob("sub-*")):
            eeg_directory = subject_directory / "eeg"
            if not eeg_directory.is_dir():
                continue
            subject = _subject_id(subject_directory.name)
            if subject in discovered:
                raise ValueError(
                    f"Participant {subject} occurs in multiple configured derivatives roots."
                )
            discovered[subject] = eeg_directory

    requested = sorted(discovered) if config.participants is None else list(config.participants)
    requested = [
        subject for subject in requested if subject not in config.excluded_participants
    ]
    if not requested:
        raise ValueError("No participants remain after applying the configuration.")
    missing = sorted(set(requested) - set(discovered))
    if missing:
        raise FileNotFoundError(
            f"Configured participants lack MNE derivative directories: {missing}"
        )
    return {subject: discovered[subject] for subject in requested}


def export_subject(config: ReviewConfig, subject: str, eeg_directory: Path) -> Path:
    """Export one exact MNE decomposition with trial-rejected pre-ICA sources."""
    paths = _subject_paths(config, subject, eeg_directory)
    ica = mne.preprocessing.read_ica(paths["ica"], verbose=False)
    metadata = _load_trial_metadata(config, subject)
    review_epochs = _build_review_epochs(config, metadata, paths, ica.ch_names)
    sources = ica.get_sources(review_epochs)
    sources.resample(config.resample_hz, npad="auto", verbose=False)
    source_data = sources.get_data(copy=False)
    _finite(source_data, f"{subject} ICA source epochs")

    topographies = np.asarray(ica.get_components(), dtype=float)
    if topographies.shape != (len(ica.ch_names), ica.n_components_):
        raise ValueError(f"Unexpected ICA topography shape for {subject}.")
    _finite(topographies, f"{subject} ICA topographies")

    component_proposals = _load_component_proposals(
        paths["components"], ica.n_components_, ica.exclude
    )
    component = _component_structure(
        ica,
        review_epochs,
        sources,
        topographies,
        component_proposals,
    )
    trialinfo = np.column_stack(
        [pd.to_numeric(metadata[column], errors="raise") for column in TRIAL_COLUMNS]
    )
    package = {
        "component": component,
        "metadata": {
            "subject": subject,
            "trialinfo": trialinfo,
            "trialinfo_labels": np.asarray(TRIAL_COLUMNS, dtype=object),
            "source_files": np.asarray([str(path) for path in paths.values()], dtype=object),
        },
    }

    output_directory = config.output_root / "exports" / subject
    output_path = output_directory / (
        f"{subject}_task-{config.task}_desc-mneica_components.mat"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not config.overwrite_exports:
        LOGGER.info("Keeping existing export for %s: %s", subject, output_path)
        return output_path
    _save_mat_atomic(output_path, package)
    LOGGER.info("Exported exact MNE ICA review data for %s: %s", subject, output_path)
    return output_path


def write_runtime_config(config: ReviewConfig) -> Path:
    """Write resolved JSON consumed by the MATLAB functions."""
    runtime = json.loads(json.dumps(config.raw))
    runtime["paths"].update(
        {
            "fieldtrip": str(config.fieldtrip_path),
            "bids_eeg": str(config.bids_root),
            "mne_derivatives": [str(path) for path in config.derivatives_roots],
            "output": str(config.output_root),
        }
    )
    runtime["resolved_participants"] = list(discover_subject_directories(config))
    config.output_root.mkdir(parents=True, exist_ok=True)
    runtime_path = config.output_root / "mne_ica_review_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return runtime_path


def _subject_paths(
    config: ReviewConfig, subject: str, eeg_directory: Path
) -> dict[str, Path]:
    paths = {
        "ica": eeg_directory / f"{subject}_proc-ica_ica.fif",
        "components": eeg_directory / f"{subject}_proc-ica_components.tsv",
    }
    for run in config.expected_runs:
        paths[f"filtered_raw_run_{run}"] = eeg_directory / (
            f"{subject}_task-{config.task}_run-{run}_proc-filt_raw.fif"
        )
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} input for {subject}: {path}")
    return paths


def _load_component_proposals(
    path: Path,
    component_count: int,
    ica_exclude: Sequence[int],
) -> pd.DataFrame:
    proposals = pd.read_csv(path, sep="\t", keep_default_na=False)
    required_columns = {"component", "status", "status_description"}
    missing_columns = sorted(required_columns - set(proposals.columns))
    if missing_columns:
        raise ValueError(f"{path} lacks component proposal columns: {missing_columns}")
    components = pd.to_numeric(proposals["component"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(components, np.arange(component_count)):
        raise ValueError(f"{path} components do not match the saved ICA decomposition.")
    statuses = proposals["status"].astype(str).str.strip().str.lower()
    if not statuses.isin(("good", "bad")).all():
        raise ValueError(f"{path} statuses must contain only 'good' or 'bad'.")
    proposed_bad = components[statuses.eq("bad")]
    excluded = np.asarray(sorted(int(index) for index in ica_exclude), dtype=int)
    if not np.array_equal(proposed_bad, excluded):
        raise ValueError(
            f"{path} automatic bad proposals do not match the saved ICA exclusions."
        )
    return proposals


def _build_review_epochs(
    config: ReviewConfig,
    metadata: pd.DataFrame,
    paths: dict[str, Path],
    ica_channels: Sequence[str],
) -> mne.Epochs:
    run_epochs = []
    for run in config.expected_runs:
        raw_path = paths[f"filtered_raw_run_{run}"]
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        run_metadata = metadata.loc[metadata["run_id"].eq(run)].reset_index(drop=True)
        samples = _match_event_samples(config, raw, run_metadata, raw_path)
        events = np.column_stack(
            (
                samples + raw.first_samp,
                np.zeros(len(samples), dtype=int),
                np.ones(len(samples), dtype=int),
            )
        )
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id={config.event_name: 1},
            tmin=config.tmin_s,
            tmax=config.tmax_s,
            baseline=None,
            picks=list(ica_channels),
            preload=True,
            reject_by_annotation=True,
            metadata=run_metadata,
            on_missing="raise",
            verbose=False,
        )
        if len(epochs) != config.trials_per_run:
            raise ValueError(
                f"{raw_path} retained {len(epochs)} thermal epochs; "
                f"expected {config.trials_per_run}."
            )
        run_epochs.append(epochs)

    review_epochs = mne.concatenate_epochs(run_epochs, add_offset=True, verbose=False)
    expected_trials = len(config.expected_runs) * config.trials_per_run
    if len(review_epochs) != expected_trials:
        raise ValueError(
            f"Constructed {len(review_epochs)} review epochs; expected {expected_trials}."
        )
    if review_epochs.ch_names != list(ica_channels):
        raise ValueError("Review epoch channels do not match the saved ICA channel order.")
    _finite(review_epochs.get_data(copy=False), "pre-ICA review epochs")
    return review_epochs


def _match_event_samples(
    config: ReviewConfig,
    raw: mne.io.BaseRaw,
    metadata: pd.DataFrame,
    raw_path: Path,
) -> np.ndarray:
    annotation_onsets = np.asarray(
        [
            annotation["onset"]
            for annotation in raw.annotations
            if annotation["description"] == config.event_name
        ],
        dtype=float,
    )
    if annotation_onsets.size < len(metadata):
        raise ValueError(
            f"{raw_path} contains {annotation_onsets.size} matching annotations for "
            f"{len(metadata)} thermal trials."
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
                f"{raw_path} has no thermal annotation within "
                f"{config.onset_tolerance_s:.6f} s of BIDS onset {onset:.6f} s."
            )
        if annotation_index in matched_indices:
            raise ValueError(f"{raw_path} maps multiple trials to one annotation.")
        matched_indices.add(annotation_index)
        samples[trial_index] = raw.time_as_index(annotation_onsets[annotation_index])[0]
    return samples


def _load_trial_metadata(config: ReviewConfig, subject: str) -> pd.DataFrame:
    frames = []
    for run in config.expected_runs:
        path = (
            config.bids_root
            / subject
            / "eeg"
            / f"{subject}_task-{config.task}_run-{run}_events.tsv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing BIDS events for {subject}, run {run}: {path}")
        events = pd.read_csv(path, sep="\t")
        required_columns = set(TRIAL_COLUMNS) | {"onset", "trial_type"}
        missing = sorted(required_columns - set(events.columns))
        if missing:
            raise ValueError(f"{path} lacks required trial columns: {missing}")
        mask = events["trial_type"].eq(config.event_name) & events["stimulus_temp"].notna()
        trials = events.loc[mask].copy()
        if len(trials) != config.trials_per_run:
            raise ValueError(
                f"{path} has {len(trials)} thermal trials; expected {config.trials_per_run}."
            )
        if not pd.to_numeric(trials["run_id"], errors="raise").eq(run).all():
            raise ValueError(f"{path} contains a run_id other than {run}.")
        frames.append(trials)
    metadata = pd.concat(frames, ignore_index=True)
    if metadata[["run_id", "trial_number"]].duplicated().any():
        raise ValueError(f"BIDS trial identifiers are duplicated for {subject}.")
    return metadata


def _component_structure(
    ica: mne.preprocessing.ICA,
    epochs: mne.Epochs,
    sources: mne.Epochs,
    topographies: np.ndarray,
    proposals: pd.DataFrame,
) -> dict[str, Any]:
    trials = np.empty((1, len(sources)), dtype=object)
    times = np.empty((1, len(sources)), dtype=object)
    source_data = sources.get_data(copy=False).astype(np.float32, copy=False)
    for index in range(len(sources)):
        trials[0, index] = source_data[index]
        times[0, index] = sources.times.astype(float, copy=False)

    channel_positions = np.vstack(
        [epochs.info["chs"][epochs.ch_names.index(name)]["loc"][:3] for name in ica.ch_names]
    )
    _finite(channel_positions, "ICA channel positions")
    component_labels = np.asarray(
        [f"ic{index + 1:03d}" for index in range(ica.n_components_)], dtype=object
    )
    proposed_bad_indices = np.asarray(
        sorted(int(index) + 1 for index in ica.exclude), dtype=int
    )
    proposed_bad_mask = np.zeros(ica.n_components_, dtype=bool)
    proposed_bad_mask[proposed_bad_indices - 1] = True
    sample_count = len(sources.times)
    return {
        "label": component_labels,
        "trial": trials,
        "time": times,
        "fsample": float(sources.info["sfreq"]),
        "sampleinfo": np.column_stack(
            (
                np.arange(0, len(sources) * sample_count, sample_count) + 1,
                np.arange(sample_count, (len(sources) + 1) * sample_count, sample_count),
            )
        ),
        "topo": topographies,
        "topolabel": np.asarray(ica.ch_names, dtype=object),
        "unmixing": np.linalg.pinv(topographies),
        "elec": {
            "label": np.asarray(ica.ch_names, dtype=object),
            "chanpos": channel_positions,
            "elecpos": channel_positions,
            "unit": "m",
        },
        "mne_proposed_bad_indices": proposed_bad_indices,
        "mne_proposed_bad_mask": proposed_bad_mask,
        "mne_status": np.asarray(proposals["status"].astype(str), dtype=object),
        "mne_status_description": np.asarray(
            proposals["status_description"].astype(str), dtype=object
        ),
    }


def _save_mat_atomic(path: Path, package: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".mat", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        savemat(
            temporary_path,
            package,
            appendmat=False,
            do_compression=True,
            long_field_names=True,
            oned_as="row",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration '{key}' must be a mapping.")
    return value


def _text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration '{key}' must be non-empty text.")
    return value.strip()


def _subject_id(value: Any) -> str:
    subject = str(value).strip()
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"
    if subject == "sub-":
        raise ValueError("Participant identifiers cannot be empty.")
    return subject


def _finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contain non-finite values.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runtime-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    config = ReviewConfig.load(args.config.expanduser().resolve())
    if not args.runtime_only:
        for subject, eeg_directory in discover_subject_directories(config).items():
            export_subject(config, subject, eeg_directory)
    runtime_path = write_runtime_config(config)
    print(f"RUNTIME_CONFIG={runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
