"""Study 1 preprocessing-stage cohort power spectral density definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.cohort_power_spectral_density import (
    CohortPsdSpecification,
    CohortPsdSummary,
    build_cohort_psd_summary,
)
from studies.pain_study.study1.figures.continuous_spectrum import (
    ContinuousSpectrumSpecification,
)
from studies.pain_study.study1.figures.preprocessing_psd_sources import (
    BrainVisionSourceCorrection,
    BrainVisionSourceExclusion,
    EegRunSource,
    estimate_source_spectrum,
)
from studies.pain_study.study1.figures.spectral_statistics import (
    ParticipantBootstrapSpecification,
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
    source_corrections: tuple[BrainVisionSourceCorrection, ...]
    source_exclusions: tuple[BrainVisionSourceExclusion, ...]


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
        source_corrections=_source_corrections(stage_config),
        source_exclusions=_source_exclusions(stage_config),
    )


def build_preprocessing_stage_psd_summary(
    sources: tuple[EegRunSource, ...],
    specification: PreprocessingStagePsdSpecification,
    *,
    bootstrap: ParticipantBootstrapSpecification,
) -> CohortPsdSummary:
    """Estimate and summarize all runs for one preprocessing stage."""
    run_spectra = tuple(
        estimate_source_spectrum(source, specification.spectrum) for source in sources
    )
    summary = build_cohort_psd_summary(
        run_spectra,
        CohortPsdSpecification(
            spectrum=specification.spectrum,
            excluded_subjects=specification.excluded_subjects,
        ),
        bootstrap=bootstrap,
    )
    return label_preprocessing_stage_summary(summary, sources, specification)


def label_preprocessing_stage_summary(
    summary: CohortPsdSummary,
    sources: tuple[EegRunSource, ...],
    specification: PreprocessingStagePsdSpecification,
) -> CohortPsdSummary:
    """Return a stage-labeled summary with exact source provenance."""
    representations = {source.source_path: source.representation for source in sources}
    run_audit = summary.run_audit.copy()
    run_audit["run"] = pd.to_numeric(run_audit["run"], errors="raise").astype(int)
    run_audit.insert(0, "stage", specification.stage.identifier)
    run_audit.insert(
        1,
        "source_representation",
        run_audit["source_file"].astype(str).map(representations),
    )
    corrections = {source.source_path: source.source_correction for source in sources}
    run_audit.insert(
        2,
        "source_correction",
        run_audit["source_file"].astype(str).map(corrections),
    )
    if run_audit["source_representation"].isna().any():
        missing = run_audit.loc[
            run_audit["source_representation"].isna(),
            "source_file",
        ].tolist()
        raise ValueError(f"Missing preprocessing-stage source provenance: {missing}")
    run_audit["segment_duration_s"] = specification.segment_duration_s
    run_audit["overlap_fraction"] = specification.overlap_fraction
    return CohortPsdSummary(
        participant_spectra=_prepend_stage(
            summary.participant_spectra,
            specification.stage.identifier,
        ),
        cohort_spectrum=_prepend_stage(
            summary.cohort_spectrum,
            specification.stage.identifier,
        ),
        run_audit=run_audit,
    )


def _prepend_stage(frame: pd.DataFrame, stage_identifier: str) -> pd.DataFrame:
    labeled = frame.copy()
    labeled.insert(0, "stage", stage_identifier)
    return labeled


def _source_corrections(
    stage_config: Mapping[str, Any],
) -> tuple[BrainVisionSourceCorrection, ...]:
    configured = stage_config.get("source_corrections", ())
    if not isinstance(configured, list | tuple):
        raise ValueError("Preprocessing PSD source_corrections must be a sequence.")
    corrections = []
    for correction in configured:
        if not isinstance(correction, Mapping):
            raise ValueError("Each preprocessing PSD source correction must be a mapping.")
        corrections.append(
            BrainVisionSourceCorrection(
                header_filename=str(correction["header_filename"]),
                subject_id=str(correction["subject_id"]),
                run_id=str(correction["run_id"]),
                data_filename=str(correction["data_filename"]),
                marker_filename=str(correction["marker_filename"]),
                expected_data_reference=str(correction["expected_data_reference"]),
                expected_marker_reference=str(correction["expected_marker_reference"]),
                reason=str(correction["reason"]),
            )
        )
    return tuple(corrections)


def _source_exclusions(
    stage_config: Mapping[str, Any],
) -> tuple[BrainVisionSourceExclusion, ...]:
    configured = stage_config.get("source_exclusions", ())
    if not isinstance(configured, list | tuple):
        raise ValueError("Preprocessing PSD source_exclusions must be a sequence.")
    exclusions = []
    for exclusion in configured:
        if not isinstance(exclusion, Mapping):
            raise ValueError("Each preprocessing PSD source exclusion must be a mapping.")
        exclusions.append(
            BrainVisionSourceExclusion(
                header_filename=str(exclusion["header_filename"]),
                reason=str(exclusion["reason"]),
            )
        )
    return tuple(exclusions)


def _integral_samples(duration_s: float, sampling_frequency_hz: float, name: str) -> int:
    sample_count = duration_s * sampling_frequency_hz
    if not math.isfinite(sample_count) or not sample_count.is_integer():
        raise ValueError(f"Preprocessing PSD {name} duration must yield an integer sample count.")
    return int(sample_count)


__all__ = [
    "PreprocessingStage",
    "PreprocessingStagePsdSpecification",
    "build_preprocessing_stage_psd_summary",
    "label_preprocessing_stage_summary",
    "preprocessing_stage_psd_specification",
]
