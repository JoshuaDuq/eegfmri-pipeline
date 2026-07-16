"""Participant-first scanner-spectrum QC for native EEG-fMRI correction."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eeg_pipeline.analysis.participant_bootstrap import paired_participant_bootstrap
from eeg_pipeline.preprocessing.eeg_fmri.qc import HarmonicSpectrum, HarmonicStageQc

SCANNER_SPECTRUM_LOW_HZ = 15.0
SCANNER_SPECTRUM_HIGH_HZ = 90.0
SCANNER_SPECTRUM_STAGES = ("raw", "gradient_corrected", "final")


@dataclass(frozen=True)
class RunScannerSpectra:
    """Cropped channel-median spectra retained from one completed run."""

    subject: str
    run: int
    channel_count: int
    raw: HarmonicSpectrum
    gradient_corrected: HarmonicSpectrum
    final: HarmonicSpectrum

    @property
    def stages(self) -> tuple[tuple[str, HarmonicSpectrum], ...]:
        """Return spectra in the fixed correction order."""
        return (
            ("raw", self.raw),
            ("gradient_corrected", self.gradient_corrected),
            ("final", self.final),
        )


@dataclass(frozen=True)
class CohortStageSpectrum:
    """Participant-first cohort spectrum and bootstrap interval for one stage."""

    frequencies_hz: np.ndarray
    median_power_db: np.ndarray
    confidence_low_power_db: np.ndarray
    confidence_high_power_db: np.ndarray


@dataclass(frozen=True)
class CohortScannerSpectra:
    """Equally weighted participant spectra for all native correction stages."""

    participant_count: int
    run_count: int
    channel_count: int
    raw: CohortStageSpectrum
    gradient_corrected: CohortStageSpectrum
    final: CohortStageSpectrum

    @property
    def stages(self) -> tuple[tuple[str, CohortStageSpectrum], ...]:
        """Return cohort stages in the fixed correction order."""
        return (
            ("raw", self.raw),
            ("gradient_corrected", self.gradient_corrected),
            ("final", self.final),
        )


def _immutable_float(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _crop_spectrum(spectrum: HarmonicSpectrum, stage: str) -> HarmonicSpectrum:
    frequencies = np.asarray(spectrum.frequencies_hz, dtype=float)
    power = np.asarray(spectrum.median_power_db, dtype=float)
    if frequencies.ndim != 1 or power.shape != frequencies.shape:
        raise ValueError(f"{stage} scanner spectrum arrays must be aligned and one-dimensional")
    if frequencies.size < 2 or np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"{stage} scanner spectrum frequencies must be strictly increasing")
    if not np.isfinite(frequencies).all() or not np.isfinite(power).all():
        raise ValueError(f"{stage} scanner spectrum must contain only finite values")
    selected = (frequencies >= SCANNER_SPECTRUM_LOW_HZ) & (frequencies <= SCANNER_SPECTRUM_HIGH_HZ)
    if np.count_nonzero(selected) < 2:
        raise ValueError(f"{stage} scanner spectrum does not cover 15–90 Hz")
    return HarmonicSpectrum(
        frequencies_hz=_immutable_float(frequencies[selected]),
        median_power_db=_immutable_float(power[selected]),
    )


def extract_run_scanner_spectra(
    *,
    subject: str,
    run: int,
    harmonic_stages: Mapping[str, HarmonicStageQc],
) -> RunScannerSpectra:
    """Extract the small common-frequency spectrum record needed for cohort QC."""
    if not subject:
        raise ValueError("Scanner spectrum subject must be non-empty")
    if run < 1:
        raise ValueError("Scanner spectrum run must be positive")
    if set(harmonic_stages) != set(SCANNER_SPECTRUM_STAGES):
        raise ValueError(
            "Scanner spectrum stages must be exactly " f"{list(SCANNER_SPECTRUM_STAGES)}"
        )

    channel_counts = {
        int(harmonic_stages[stage].summary["n_channels"]) for stage in SCANNER_SPECTRUM_STAGES
    }
    if len(channel_counts) != 1:
        raise ValueError("Scanner spectrum stage channel counts do not match")
    channel_count = channel_counts.pop()
    if channel_count < 1:
        raise ValueError("Scanner spectrum channel count must be positive")

    spectra = {
        stage: _crop_spectrum(harmonic_stages[stage].spectrum, stage)
        for stage in SCANNER_SPECTRUM_STAGES
    }
    reference_frequencies = spectra["raw"].frequencies_hz
    for stage in SCANNER_SPECTRUM_STAGES[1:]:
        if not np.array_equal(spectra[stage].frequencies_hz, reference_frequencies):
            raise ValueError("Scanner spectrum stage frequency bins do not match")
    return RunScannerSpectra(
        subject=subject,
        run=run,
        channel_count=channel_count,
        raw=spectra["raw"],
        gradient_corrected=spectra["gradient_corrected"],
        final=spectra["final"],
    )


def _validate_runs(runs: Sequence[RunScannerSpectra]) -> np.ndarray:
    if not runs:
        raise ValueError("Cohort scanner spectra require at least one completed run")
    identities = [(run.subject, run.run) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate scanner spectrum run identity")

    channel_counts = {run.channel_count for run in runs}
    if len(channel_counts) != 1:
        raise ValueError("Cohort scanner spectrum channel counts do not match")
    frequencies = runs[0].raw.frequencies_hz
    for run in runs:
        for _, spectrum in run.stages:
            if not np.array_equal(spectrum.frequencies_hz, frequencies):
                raise ValueError("Cohort scanner spectrum frequency bins do not match")
            if not np.isfinite(spectrum.median_power_db).all():
                raise ValueError("Cohort scanner spectrum power contains non-finite values")
    return frequencies


def _aggregate_stage(
    runs: Sequence[RunScannerSpectra],
    *,
    stage: str,
    frequencies_hz: np.ndarray,
    bootstrap_iterations: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> CohortStageSpectrum:
    subject_runs: dict[str, list[np.ndarray]] = defaultdict(list)
    for run in runs:
        spectra = dict(run.stages)
        subject_runs[run.subject].append(spectra[stage].median_power_db)
    participant_matrix = np.stack(
        [np.median(np.stack(subject_runs[subject]), axis=0) for subject in sorted(subject_runs)]
    )
    median, confidence_low, confidence_high = paired_participant_bootstrap(
        participant_matrix,
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )
    return CohortStageSpectrum(
        frequencies_hz=_immutable_float(frequencies_hz),
        median_power_db=_immutable_float(median),
        confidence_low_power_db=_immutable_float(confidence_low),
        confidence_high_power_db=_immutable_float(confidence_high),
    )


def aggregate_cohort_scanner_spectra(
    runs: Sequence[RunScannerSpectra],
    *,
    bootstrap_iterations: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> CohortScannerSpectra:
    """Aggregate channel-median run spectra with participants equally weighted."""
    completed_runs = tuple(runs)
    frequencies = _validate_runs(completed_runs)
    stage_spectra = {
        stage: _aggregate_stage(
            completed_runs,
            stage=stage,
            frequencies_hz=frequencies,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
        for stage in SCANNER_SPECTRUM_STAGES
    }
    return CohortScannerSpectra(
        participant_count=len({run.subject for run in completed_runs}),
        run_count=len(completed_runs),
        channel_count=completed_runs[0].channel_count,
        raw=stage_spectra["raw"],
        gradient_corrected=stage_spectra["gradient_corrected"],
        final=stage_spectra["final"],
    )


def write_cohort_scanner_spectra_tsv(
    cohort: CohortScannerSpectra,
    path: str | Path,
) -> None:
    """Write every plotted cohort spectrum value to a deterministic TSV."""
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "stage",
                "frequency_hz",
                "median_psd_db_v2_hz",
                "ci_low_psd_db_v2_hz",
                "ci_high_psd_db_v2_hz",
                "n_participants",
                "n_runs",
            ]
        )
        for stage, spectrum in cohort.stages:
            for frequency, median, confidence_low, confidence_high in zip(
                spectrum.frequencies_hz,
                spectrum.median_power_db,
                spectrum.confidence_low_power_db,
                spectrum.confidence_high_power_db,
                strict=True,
            ):
                writer.writerow(
                    [
                        stage,
                        float(frequency),
                        float(median),
                        float(confidence_low),
                        float(confidence_high),
                        cohort.participant_count,
                        cohort.run_count,
                    ]
                )


__all__ = [
    "CohortScannerSpectra",
    "CohortStageSpectrum",
    "RunScannerSpectra",
    "aggregate_cohort_scanner_spectra",
    "extract_run_scanner_spectra",
    "write_cohort_scanner_spectra_tsv",
]
