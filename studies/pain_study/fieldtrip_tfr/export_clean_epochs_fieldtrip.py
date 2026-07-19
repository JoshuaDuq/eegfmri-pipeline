"""Export pre-ICA MNE epochs after applying the final trial-rejection mask."""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import Sequence

import mne
import numpy as np
import pandas as pd

from studies.pain_study.fieldtrip_tfr.export_fieldtrip import (
    CONFIG_PATH,
    ExportConfig,
    SubjectExport,
    _load_run_metadata,
    _normalize_subject,
    _require_finite,
    _resolve_events_path,
    export_subject,
    write_runtime_config,
)

LOGGER = logging.getLogger(__name__)


def prepare_trial_rejected_preica_subject(
    config: ExportConfig,
    subject: str,
    pre_ica_epochs_path: Path,
    trial_rejected_epochs_path: Path,
) -> SubjectExport:
    """Retain rejected-trial selection while preserving pre-ICA signals."""
    subject = _normalize_subject(subject)
    pre_ica_epochs_path = pre_ica_epochs_path.expanduser().resolve()
    trial_rejected_epochs_path = trial_rejected_epochs_path.expanduser().resolve()
    if not pre_ica_epochs_path.is_file():
        raise FileNotFoundError(
            f"Pre-ICA epochs file does not exist: {pre_ica_epochs_path}"
        )
    if not trial_rejected_epochs_path.is_file():
        raise FileNotFoundError(
            "Trial-rejected epochs file does not exist: "
            f"{trial_rejected_epochs_path}"
        )

    metadata_frames = []
    events_paths = []
    for run in config.expected_runs:
        events_path = _resolve_events_path(config, subject, run)
        metadata_frames.append(_load_run_metadata(config, events_path, run))
        events_paths.append(events_path)
    original_metadata = pd.concat(metadata_frames, ignore_index=True)

    pre_ica_epochs = mne.read_epochs(
        pre_ica_epochs_path, preload=True, verbose=False
    ).pick("eeg")
    trial_rejected_epochs = mne.read_epochs(
        trial_rejected_epochs_path, preload=False, verbose=False
    )
    selection = np.asarray(trial_rejected_epochs.selection, dtype=int)
    if len(selection) != len(trial_rejected_epochs):
        raise ValueError("Trial-rejection selection count is inconsistent.")
    if len(np.unique(selection)) != len(selection):
        raise ValueError("Clean epoch selection contains duplicate original indices.")
    if np.any(selection < 0) or np.any(selection >= len(original_metadata)):
        raise ValueError(
            "Clean epoch selection contains an index outside the original trial table."
        )
    if not np.all(np.diff(selection) > 0):
        raise ValueError("Clean epoch selection must preserve original trial order.")
    expected_selection = np.arange(len(original_metadata))
    if not np.array_equal(pre_ica_epochs.selection, expected_selection):
        raise ValueError(
            "Pre-ICA epochs must contain every original trial in original order."
        )
    if not np.isclose(pre_ica_epochs.tmin, config.tmin_s) or not np.isclose(
        pre_ica_epochs.tmax, config.tmax_s
    ):
        raise ValueError(
            f"Pre-ICA epoch window [{pre_ica_epochs.tmin}, "
            f"{pre_ica_epochs.tmax}] does not match "
            f"configured [{config.tmin_s}, {config.tmax_s}]."
        )
    if not np.isclose(trial_rejected_epochs.tmin, pre_ica_epochs.tmin) or not np.isclose(
        trial_rejected_epochs.tmax, pre_ica_epochs.tmax
    ):
        raise ValueError("Trial-rejected and pre-ICA epoch windows do not match.")

    trial_metadata = original_metadata.iloc[selection].reset_index(drop=True)
    epochs = pre_ica_epochs[selection]
    epochs.metadata = trial_metadata.copy()
    _require_finite(epochs.get_data(copy=False), f"{subject} retained pre-ICA epochs")

    ica_epochs = epochs.copy().filter(
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
        broadband_epochs=epochs,
        ica_epochs=ica_epochs,
        trial_metadata=trial_metadata,
        bad_channels=tuple(sorted(epochs.info["bads"])),
        source_files=(pre_ica_epochs_path, trial_rejected_epochs_path, *events_paths),
    )


def _configured_path(config: ExportConfig, name: str) -> Path:
    paths = config.raw.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration paths must be a mapping.")
    value = paths.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration paths.{name} must be a non-empty path.")
    return Path(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trial-rejected pre-ICA epochs to FieldTrip."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Export one clean-epochs participant package and runtime configuration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    config = ExportConfig.load(args.config.resolve())
    prepared = prepare_trial_rejected_preica_subject(
        config,
        args.subject,
        _configured_path(config, "pre_ica_epochs"),
        _configured_path(config, "trial_rejected_epochs"),
    )
    output = export_subject(config, prepared, overwrite=args.overwrite)
    runtime_path = write_runtime_config(config)
    LOGGER.info(
        "Exported %s trial-rejected pre-ICA trials from %s to %s",
        len(prepared.broadband_epochs),
        prepared.source_files[0],
        output,
    )
    LOGGER.info("Wrote MATLAB runtime configuration %s", runtime_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
