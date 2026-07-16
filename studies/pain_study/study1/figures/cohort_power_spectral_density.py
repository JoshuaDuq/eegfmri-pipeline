"""Participant-first cohort power spectral density analysis for Study 1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.participant_bootstrap import paired_participant_bootstrap
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.continuous_spectrum import (
    ContinuousRunSpectrum,
    ContinuousSpectrumSpecification,
)
from studies.pain_study.study1.figures.spectral_statistics import (
    ParticipantBootstrapSpecification,
)

VOLT_SQUARED_TO_MICROVOLT_SQUARED = 1e12


@dataclass(frozen=True)
class CohortPsdSpecification:
    """Continuous-spectrum settings and participant exclusions."""

    spectrum: ContinuousSpectrumSpecification
    excluded_subjects: tuple[str, ...]


@dataclass(frozen=True)
class CohortPsdSummary:
    """Run audits, participant spectra, and cohort uncertainty."""

    participant_spectra: pd.DataFrame
    cohort_spectrum: pd.DataFrame
    run_audit: pd.DataFrame

    @property
    def n_subjects(self) -> int:
        return int(self.participant_spectra["subject_id"].nunique())

    @property
    def n_runs(self) -> int:
        return int(len(self.run_audit))


def cohort_psd_specification(config: Any) -> CohortPsdSpecification:
    """Load the fixed cohort-PSD settings."""
    cohort = require_config_value(config, "study1.figures.cohort_power_spectral_density")
    continuous = require_config_value(config, "study1.figures.continuous_spectrum")
    frequency_range = tuple(float(value) for value in cohort["frequency_range_hz"])
    if len(frequency_range) != 2:
        raise ValueError("Cohort PSD frequency_range_hz must contain two values.")
    return CohortPsdSpecification(
        spectrum=ContinuousSpectrumSpecification(
            frequency_range_hz=(frequency_range[0], frequency_range[1]),
            n_fft=int(cohort["n_fft"]),
            n_overlap=int(cohort["n_overlap"]),
            sampling_frequency_hz=float(cohort["sampling_frequency_hz"]),
        ),
        excluded_subjects=tuple(str(value) for value in continuous["excluded_subjects"]),
    )


def build_cohort_psd_summary(
    run_spectra: Sequence[ContinuousRunSpectrum],
    specification: CohortPsdSpecification,
    *,
    bootstrap: ParticipantBootstrapSpecification,
) -> CohortPsdSummary:
    """Aggregate linear run PSDs within participants before cohort estimation."""
    runs = tuple(run_spectra)
    if not runs:
        raise ValueError("Cohort PSD summary requires at least one run spectrum.")
    frequencies = _validate_run_spectra(runs, specification)
    subjects = sorted({run.subject_id for run in runs})

    participant_rows: list[dict[str, float | int | str]] = []
    participant_matrix = np.empty((len(subjects), frequencies.size), dtype=float)
    for subject_index, subject_id in enumerate(subjects):
        subject_runs = tuple(run for run in runs if run.subject_id == subject_id)
        participant_linear_psd = np.median(
            np.stack([run.median_psd_v2_hz for run in subject_runs]),
            axis=0,
        )
        participant_psd_db = 10.0 * np.log10(
            participant_linear_psd * VOLT_SQUARED_TO_MICROVOLT_SQUARED
        )
        participant_matrix[subject_index] = participant_psd_db
        participant_rows.extend(
            {
                "subject_id": subject_id,
                "frequency_hz": float(frequency),
                "psd_db_uv2_hz": float(power),
                "n_runs": len(subject_runs),
            }
            for frequency, power in zip(frequencies, participant_psd_db, strict=True)
        )

    cohort_median, ci_low, ci_high = paired_participant_bootstrap(
        participant_matrix,
        iterations=bootstrap.iterations,
        confidence_level=bootstrap.confidence_level,
        seed=bootstrap.seed,
    )
    cohort_spectrum = pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "median_psd_db_uv2_hz": cohort_median,
            "ci_low_psd_db_uv2_hz": ci_low,
            "ci_high_psd_db_uv2_hz": ci_high,
            "n_subjects": len(subjects),
        }
    )
    return CohortPsdSummary(
        participant_spectra=pd.DataFrame(
            participant_rows,
            columns=("subject_id", "frequency_hz", "psd_db_uv2_hz", "n_runs"),
        ),
        cohort_spectrum=cohort_spectrum,
        run_audit=_build_run_audit(runs, specification.spectrum),
    )


def _validate_run_spectra(
    runs: Sequence[ContinuousRunSpectrum],
    specification: CohortPsdSpecification,
) -> np.ndarray:
    frequencies = np.asarray(runs[0].frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2 or not np.isfinite(frequencies).all():
        raise ValueError("Cohort PSD requires a finite one-dimensional frequency axis.")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Cohort PSD frequencies must be strictly increasing.")
    for run in runs:
        if not np.array_equal(run.frequencies_hz, frequencies):
            raise ValueError("Cohort PSD run spectra use inconsistent frequency bins.")
        power = np.asarray(run.median_psd_v2_hz, dtype=float)
        if power.shape != frequencies.shape:
            raise ValueError("Cohort PSD power and frequency vectors must align.")
        if not np.isfinite(power).all() or np.any(power <= 0.0):
            raise ValueError("Cohort PSD runs require strictly positive finite power.")
        if run.sampling_frequency_hz != specification.spectrum.sampling_frequency_hz:
            raise ValueError("Cohort PSD run sampling frequencies do not match configuration.")
    return frequencies


def _build_run_audit(
    runs: Sequence[ContinuousRunSpectrum],
    specification: ContinuousSpectrumSpecification,
) -> pd.DataFrame:
    lower_frequency, upper_frequency = specification.frequency_range_hz
    rows = [
        {
            "subject_id": run.subject_id,
            "run": run.run_id,
            "source_file": str(run.source_file),
            "n_channels": run.n_channels,
            "sampling_frequency_hz": run.sampling_frequency_hz,
            "n_samples": run.n_samples,
            "recording_duration_s": run.recording_duration_s,
            "bad_annotation_duration_s": run.bad_annotation_duration_s,
            "analyzed_duration_s": run.analyzed_duration_s,
            "frequency_min_hz": lower_frequency,
            "frequency_max_hz": upper_frequency,
            "n_fft": specification.n_fft,
            "n_overlap": specification.n_overlap,
            "frequency_resolution_hz": float(np.median(np.diff(run.frequencies_hz))),
        }
        for run in runs
    ]
    return (
        pd.DataFrame(rows).sort_values(["subject_id", "run"], kind="stable").reset_index(drop=True)
    )


__all__ = [
    "CohortPsdSpecification",
    "CohortPsdSummary",
    "build_cohort_psd_summary",
    "cohort_psd_specification",
]
