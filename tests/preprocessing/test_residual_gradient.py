from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.residual_gradient import (
    ResidualObsSettings,
    build_volume_layout,
    validate_brainvision_source,
    validate_residual_obs_raw,
)


def _make_raw(
    *,
    sfreq: float = 1_000.0,
    n_times: int = 54_000,
    marker_samples: np.ndarray | None = None,
) -> mne.io.RawArray:
    info = mne.create_info(["Fz", "Cz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((3, n_times)), info, verbose="ERROR")
    samples = marker_samples if marker_samples is not None else np.arange(0, 45_000, 900)
    raw.set_annotations(
        mne.Annotations(
            onset=samples / sfreq,
            duration=np.zeros(samples.size),
            description=["Volume/V  1"] * samples.size,
        )
    )
    return raw


def test_build_volume_layout_accepts_complete_epochs_and_long_gap() -> None:
    samples = np.concatenate([np.arange(0, 27_000, 900), np.arange(30_000, 48_000, 900)])
    raw = _make_raw(n_times=49_000, marker_samples=samples)

    layout = build_volume_layout(raw, ResidualObsSettings())

    assert layout.epoch_samples == 900
    assert layout.starts.tolist() == samples.tolist()
    assert layout.block_ids.tolist() == [0] * 30 + [1] * 20


def test_validate_raw_rejects_unexpected_sampling_frequency() -> None:
    with pytest.raises(ValueError, match="Expected sampling frequency 1000.0 Hz"):
        validate_residual_obs_raw(_make_raw(sfreq=500.0), ResidualObsSettings())


def test_validate_raw_rejects_missing_volume_marker() -> None:
    raw = _make_raw()
    raw.set_annotations(mne.Annotations([], [], []))

    with pytest.raises(ValueError, match="Required annotation is absent"):
        validate_residual_obs_raw(raw, ResidualObsSettings())


def test_build_volume_layout_rejects_short_interval() -> None:
    raw = _make_raw(marker_samples=np.array([0, 900, 1_799, 2_700]))

    with pytest.raises(ValueError, match="shorter than the 900-sample TR"):
        build_volume_layout(raw, ResidualObsSettings(min_complete_epochs=2))


def test_validate_brainvision_source_requires_referenced_files(tmp_path: Path) -> None:
    header = tmp_path / "run_scannerpulse_corrected.vhdr"
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "DataFile=run_scannerpulse_corrected.eeg\n"
        "MarkerFile=run_scannerpulse_corrected.vmrk\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="referenced data file"):
        validate_brainvision_source(header)


def test_validate_brainvision_source_rejects_uncorrected_name(tmp_path: Path) -> None:
    header = tmp_path / "run.vhdr"
    header.touch()

    with pytest.raises(ValueError, match="_scannerpulse_corrected.vhdr"):
        validate_brainvision_source(header)
