"""Single-pass per-run evidence for the subject report.

Four panels describe each continuous run: the sensor spectra, the gradient comb, the
volume-locked residual, and the time-resolved amplitude. Each one needs the run both
before and after the ICA exclusions, and ``ICA.apply`` on a full-length run is the most
expensive operation in the report by a wide margin.

This module exists so that cost is paid once per run instead of once per panel. Each run
is read, cleaned, measured for everything, and released before the next is read, so peak
memory stays at one run rather than the whole session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mne

from eeg_pipeline.preprocessing.report.analyzer_qc import (
    RrIntervals,
    add_rr_interval_section,
    compute_rr_intervals,
)
from eeg_pipeline.preprocessing.report.continuity import (
    RunContinuity,
    add_continuity_section,
    compute_run_continuity,
)
from eeg_pipeline.preprocessing.report.scanner import (
    VOLUME_MARKER_DESCRIPTION,
    CombResidual,
    VolumeLockedAverage,
    add_scanner_residual_section,
    compute_comb_residual,
    compute_volume_locked_average,
    measure_volume_timing,
)
from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.preprocessing.report.spectra import (
    RunSpectra,
    add_spectra_section,
    compute_run_spectra,
)

#: Gradient harmonics marked on the sensor-spectra figure. The comb has tens of teeth in
#: the plotted range, and drawing all of them would bury the spectrum under vertical
#: lines. The dedicated comb panel measures every harmonic properly; these few are here
#: only so a reviewer can see where the comb sits relative to everything else.
MARKED_GRADIENT_HARMONICS = 3


@dataclass
class RunEvidence:
    """Everything measured about the continuous runs of one report."""

    spectra: list[RunSpectra] = field(default_factory=list)
    combs: list[CombResidual] = field(default_factory=list)
    locked_averages: list[VolumeLockedAverage] = field(default_factory=list)
    continuity: list[RunContinuity] = field(default_factory=list)
    rr_intervals: list[RrIntervals] = field(default_factory=list)
    gradient_fundamentals_hz: list[float] = field(default_factory=list)

    @property
    def has_scanner_evidence(self) -> bool:
        return bool(self.combs or self.locked_averages)

    @property
    def gradient_marks_hz(self) -> tuple[float, ...]:
        """The few gradient harmonics worth drawing on the sensor-spectra figure."""
        if not self.gradient_fundamentals_hz:
            return ()
        fundamental = min(self.gradient_fundamentals_hz)
        return tuple(fundamental * order for order in range(1, MARKED_GRADIENT_HARMONICS + 1))


def _measure_gradient(
    raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    *,
    recording_id: str,
    settings: ReportSettings,
    volume_description: str,
    evidence: RunEvidence,
) -> None:
    """Add whichever gradient measurements this run's volume markers support."""
    timing = measure_volume_timing(raw, description=volume_description)
    if timing is None:
        return
    evidence.gradient_fundamentals_hz.append(timing.fundamental_hz)

    comb = compute_comb_residual(
        raw,
        cleaned,
        timing=timing,
        recording_id=recording_id,
        band_hz=settings.comb_frequency_range_hz,
        welch_seconds=settings.comb_welch_seconds,
    )
    if comb is not None:
        evidence.combs.append(comb)

    locked = compute_volume_locked_average(
        raw,
        cleaned,
        timing=timing,
        recording_id=recording_id,
        description=volume_description,
    )
    if locked is not None:
        evidence.locked_averages.append(locked)


def measure_runs(
    *,
    filtered_raw_paths: Sequence[Path],
    ica: mne.preprocessing.ICA,
    settings: ReportSettings,
    volume_description: str = VOLUME_MARKER_DESCRIPTION,
) -> RunEvidence:
    """Measure every per-run panel, reading and cleaning each run exactly once."""
    if not filtered_raw_paths:
        raise ValueError("Per-run evidence requires at least one filtered run.")

    evidence = RunEvidence()
    for path in filtered_raw_paths:
        recording_id = path.name.removesuffix("_proc-filt_raw.fif")
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        cleaned = ica.apply(raw.copy(), exclude=ica.exclude, verbose="ERROR")

        evidence.spectra.append(
            compute_run_spectra(
                raw,
                cleaned,
                recording_id=recording_id,
                fmax=settings.spectra_fmax,
                line_frequency=settings.spectra_line_frequency,
            )
        )
        evidence.continuity.append(
            compute_run_continuity(
                raw,
                recording_id=recording_id,
                window_seconds=settings.continuity_window_seconds,
                volume_description=volume_description,
            )
        )
        intervals = compute_rr_intervals(raw, recording_id=recording_id)
        if intervals is not None:
            evidence.rr_intervals.append(intervals)

        _measure_gradient(
            raw,
            cleaned,
            recording_id=recording_id,
            settings=settings,
            volume_description=volume_description,
            evidence=evidence,
        )
        # Release both copies before the next run is read.
        del raw, cleaned
    return evidence


def add_run_evidence_sections(
    *,
    report: mne.Report,
    evidence: RunEvidence,
    settings: ReportSettings,
) -> None:
    """Append every per-run section that has something to show."""
    if evidence.spectra:
        add_spectra_section(
            report=report,
            spectra=evidence.spectra,
            line_frequency=settings.spectra_line_frequency,
            marked_frequencies=(
                tuple(settings.spectra_marked_frequencies) + evidence.gradient_marks_hz
            ),
        )
    if evidence.has_scanner_evidence:
        add_scanner_residual_section(
            report=report,
            combs=evidence.combs,
            averages=evidence.locked_averages,
        )
    if evidence.continuity:
        add_continuity_section(report=report, runs=evidence.continuity)
    if evidence.rr_intervals:
        add_rr_interval_section(report=report, series=evidence.rr_intervals)


def add_run_evidence_review(
    *,
    report: mne.Report,
    filtered_raw_paths: Sequence[Path],
    ica: mne.preprocessing.ICA,
    settings: ReportSettings,
    volume_description: str = VOLUME_MARKER_DESCRIPTION,
) -> RunEvidence:
    """Measure every run once and append all the per-run sections."""
    evidence = measure_runs(
        filtered_raw_paths=filtered_raw_paths,
        ica=ica,
        settings=settings,
        volume_description=volume_description,
    )
    add_run_evidence_sections(report=report, evidence=evidence, settings=settings)
    return evidence


__all__ = [
    "MARKED_GRADIENT_HARMONICS",
    "RunEvidence",
    "add_run_evidence_review",
    "add_run_evidence_sections",
    "measure_runs",
]
