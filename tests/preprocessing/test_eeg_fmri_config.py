from __future__ import annotations

from pathlib import Path

import pytest

from eeg_pipeline.preprocessing.eeg_fmri.config import load_native_eeg_fmri_parameters
from tests import REPO_ROOT

CONFIG_PATH = (
    REPO_ROOT
    / "studies"
    / "pain_study"
    / "scripts"
    / "config"
    / "native_eeg_fmri_artifact_correction.yaml"
)


def test_native_eeg_fmri_config_encodes_the_fixed_study_pipeline() -> None:
    parameters = load_native_eeg_fmri_parameters(CONFIG_PATH)

    assert parameters.acquisition_sampling_frequency_hz == 5_000.0
    assert parameters.repetition_time_seconds == 0.9
    assert parameters.expected_channel_count == 64
    assert parameters.ecg_channel == "ECG"
    assert parameters.volume_annotation == "Volume/V  1"
    assert parameters.maximum_marker_deviation_samples == 1
    assert parameters.gradient.moving_average_volumes == 21
    assert parameters.gradient.alignment_upsampling == 4
    assert parameters.gradient.residual_obs_components == 0
    assert parameters.gradient.residual_obs_folds == 5
    assert parameters.gradient.residual_obs_seed == 42
    assert parameters.low_pass_frequency_hz == 100.0
    assert parameters.output_sampling_frequency_hz == 1_000.0
    assert parameters.cardiac.obs_components == 4
    assert parameters.cardiac.detection.sampling_frequency_hz == 250.0
    assert parameters.cardiac.detection.low_frequency_hz == 0.5
    assert parameters.cardiac.detection.high_frequency_hz == 30.0
    assert parameters.cardiac.detection.window_stride_samples == 50
    assert parameters.cardiac.detection.probability_threshold == 0.05
    assert parameters.cardiac.detection.minimum_support_samples == 5
    assert parameters.cardiac.detection.refractory_period_seconds == 0.4
    assert parameters.cardiac.detection.edge_margin_seconds == 0.1
    assert parameters.cardiac.minimum_temporal_coverage == 0.9
    assert parameters.cardiac.minimum_warning_rr_seconds == 0.375
    assert parameters.cardiac.maximum_warning_rr_seconds == 1.5
    assert parameters.qc_channels == (
        "Fp1",
        "F3",
        "C3",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Oz",
        "POz",
        "FC3",
        "PO3",
        "PO7",
    )
    assert parameters.qc_welch_duration_seconds == 16.384
    assert parameters.qc_bootstrap_iterations == 10_000
    assert parameters.qc_bootstrap_confidence_level == 0.95
    assert parameters.qc_bootstrap_seed == 42


def test_native_eeg_fmri_config_rejects_unknown_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_native_eeg_fmri_parameters(config_path)


def test_native_eeg_fmri_config_rejects_version_one(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace("version: 3", "version: 1", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported"):
        load_native_eeg_fmri_parameters(config_path)
