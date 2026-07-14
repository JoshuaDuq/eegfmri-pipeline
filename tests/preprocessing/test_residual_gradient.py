from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.residual_gradient import (
    ResidualObsSettings,
    apply_residual_obs,
    brainvision_source_files,
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


def _make_artifact_raw(*, extra_samples: int = 0) -> mne.io.RawArray:
    sfreq = 1_000.0
    epoch_samples = 900
    n_epochs = 50
    time = np.arange(epoch_samples) / sfreq
    artifact = np.sin(2 * np.pi * 20 * time) + 0.7 * np.sin(2 * np.pi * 41 * time)
    artifact -= artifact.mean()
    amplitudes = np.linspace(0.8, 1.2, n_epochs)
    eeg = np.concatenate([amplitude * artifact for amplitude in amplitudes]) * 1e-6
    eeg = np.pad(eeg, (0, extra_samples))
    ecg = np.arange(eeg.size, dtype=float) * 1e-9
    data = np.vstack([eeg, 0.5 * eeg, ecg])
    info = mne.create_info(["Fz", "Cz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    samples = np.arange(n_epochs) * epoch_samples
    raw.set_annotations(
        mne.Annotations(
            onset=samples / sfreq,
            duration=np.zeros(n_epochs),
            description=["Volume/V  1"] * n_epochs,
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


def test_brainvision_source_files_returns_validated_triplet(tmp_path: Path) -> None:
    header = tmp_path / "run_scannerpulse_corrected.vhdr"
    data = tmp_path / "samples.eeg"
    markers = tmp_path / "events.vmrk"
    data.touch()
    markers.touch()
    header.write_text(
        "DataFile=samples.eeg\nMarkerFile=events.vmrk\n",
        encoding="utf-8",
    )

    assert brainvision_source_files(header) == (header, data, markers)


def test_validate_brainvision_source_rejects_uncorrected_name(tmp_path: Path) -> None:
    header = tmp_path / "run.vhdr"
    header.touch()

    with pytest.raises(ValueError, match="_scannerpulse_corrected.vhdr"):
        validate_brainvision_source(header)


def test_zero_components_is_exact_identity() -> None:
    raw = _make_artifact_raw()
    layout = build_volume_layout(raw, ResidualObsSettings())

    result = apply_residual_obs(raw, layout, n_components=0, n_folds=5)

    np.testing.assert_array_equal(result.raw.get_data(), raw.get_data())
    assert result.component_rows == ()


def test_one_component_removes_known_rank_one_residual() -> None:
    raw = _make_artifact_raw()
    layout = build_volume_layout(raw, ResidualObsSettings())

    result = apply_residual_obs(raw, layout, n_components=1, n_folds=5)

    assert np.sqrt(np.mean(result.raw.get_data(picks="eeg") ** 2)) < 1e-10


def test_obs_preserves_non_eeg_and_samples_outside_epochs() -> None:
    raw = _make_artifact_raw(extra_samples=100)
    layout = build_volume_layout(raw, ResidualObsSettings())

    result = apply_residual_obs(raw, layout, n_components=1, n_folds=5)

    ecg_pick = raw.ch_names.index("ECG")
    np.testing.assert_array_equal(result.raw.get_data([ecg_pick]), raw.get_data([ecg_pick]))
    np.testing.assert_array_equal(result.raw.get_data()[:, -100:], raw.get_data()[:, -100:])
    np.testing.assert_array_equal(result.raw.annotations.onset, raw.annotations.onset)
    np.testing.assert_array_equal(result.raw.annotations.duration, raw.annotations.duration)
    np.testing.assert_array_equal(result.raw.annotations.description, raw.annotations.description)
    assert result.raw.annotations.orig_time == raw.annotations.orig_time
