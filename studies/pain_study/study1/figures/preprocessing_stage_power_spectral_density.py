"""Study 1 preprocessing-stage cohort power spectral density definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.continuous_spectrum import (
    ContinuousSpectrumSpecification,
)


@dataclass(frozen=True)
class PreprocessingStage:
    """Identity and acquisition rate for one stored EEG checkpoint."""

    identifier: str
    label: str
    sampling_frequency_hz: float


@dataclass(frozen=True)
class PreprocessingStagePsdSpecification:
    """Validated spectral settings for one preprocessing checkpoint."""

    stage: PreprocessingStage
    spectrum: ContinuousSpectrumSpecification
    segment_duration_s: float
    overlap_fraction: float
    excluded_subjects: tuple[str, ...]


def preprocessing_stage_psd_specification(
    config: Any,
    stage_identifier: str,
) -> PreprocessingStagePsdSpecification:
    """Load one stage and derive equal-duration Welch sample counts."""
    stage_report = require_config_value(
        config,
        "study1.figures.preprocessing_stage_power_spectral_density",
    )
    stages = stage_report["stages"]
    if not isinstance(stages, Mapping) or stage_identifier not in stages:
        raise ValueError(f"Unknown preprocessing PSD stage: {stage_identifier!r}.")
    stage_config = stages[stage_identifier]
    if not isinstance(stage_config, Mapping):
        raise ValueError(f"Preprocessing PSD stage {stage_identifier!r} must be a mapping.")

    segment_duration_s = float(stage_report["segment_duration_s"])
    overlap_fraction = float(stage_report["overlap_fraction"])
    sampling_frequency_hz = float(stage_config["sampling_frequency_hz"])
    if not math.isfinite(segment_duration_s) or segment_duration_s <= 0.0:
        raise ValueError("Preprocessing PSD segment_duration_s must be positive and finite.")
    if not math.isfinite(overlap_fraction) or not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("Preprocessing PSD overlap_fraction must be in [0, 1).")

    n_fft = _integral_samples(segment_duration_s, sampling_frequency_hz, "segment")
    n_overlap = _integral_samples(
        segment_duration_s * overlap_fraction,
        sampling_frequency_hz,
        "overlap",
    )
    stage = PreprocessingStage(
        identifier=stage_identifier,
        label=str(stage_config["label"]),
        sampling_frequency_hz=sampling_frequency_hz,
    )
    cohort = require_config_value(config, "study1.figures.cohort_power_spectral_density")
    continuous = require_config_value(config, "study1.figures.continuous_spectrum")
    frequency_range = tuple(float(value) for value in cohort["frequency_range_hz"])
    if len(frequency_range) != 2:
        raise ValueError("Cohort PSD frequency_range_hz must contain two values.")
    return PreprocessingStagePsdSpecification(
        stage=stage,
        spectrum=ContinuousSpectrumSpecification(
            frequency_range_hz=(frequency_range[0], frequency_range[1]),
            n_fft=n_fft,
            n_overlap=n_overlap,
            sampling_frequency_hz=sampling_frequency_hz,
        ),
        segment_duration_s=segment_duration_s,
        overlap_fraction=overlap_fraction,
        excluded_subjects=tuple(str(value) for value in continuous["excluded_subjects"]),
    )


def _integral_samples(duration_s: float, sampling_frequency_hz: float, name: str) -> int:
    sample_count = duration_s * sampling_frequency_hz
    if not math.isfinite(sample_count) or not sample_count.is_integer():
        raise ValueError(f"Preprocessing PSD {name} duration must yield an integer sample count.")
    return int(sample_count)


__all__ = [
    "PreprocessingStage",
    "PreprocessingStagePsdSpecification",
    "preprocessing_stage_psd_specification",
]
