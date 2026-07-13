from __future__ import annotations

import pytest

from studies.pain_study.study1.config.loader import load_study1_config


@pytest.mark.parametrize(
    ("stage", "label", "sampling_frequency_hz", "n_fft", "n_overlap"),
    (
        ("raw", "Original BrainVision", 5000.0, 81_920, 40_960),
        ("processed", "BrainVision processed", 1000.0, 16_384, 8_192),
        ("mne", "Final MNE processed", 500.0, 8_192, 4_096),
    ),
)
def test_preprocessing_stage_psd_specification_uses_equal_time_windows(
    stage: str,
    label: str,
    sampling_frequency_hz: float,
    n_fft: int,
    n_overlap: int,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        preprocessing_stage_psd_specification,
    )

    specification = preprocessing_stage_psd_specification(load_study1_config(), stage)

    assert specification.stage.identifier == stage
    assert specification.stage.label == label
    assert specification.spectrum.sampling_frequency_hz == sampling_frequency_hz
    assert specification.spectrum.n_fft == n_fft
    assert specification.spectrum.n_overlap == n_overlap
    assert specification.segment_duration_s == 16.384
    assert specification.overlap_fraction == 0.5


def test_preprocessing_stage_psd_specification_rejects_unknown_stage() -> None:
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        preprocessing_stage_psd_specification,
    )

    with pytest.raises(ValueError, match="Unknown preprocessing PSD stage"):
        preprocessing_stage_psd_specification(load_study1_config(), "corrected")
