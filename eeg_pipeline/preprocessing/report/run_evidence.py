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
import numpy as np

from eeg_pipeline.preprocessing.report.analyzer_qc import (
    CardiacResidual,
    MarkerAgreement,
    RrIntervals,
    add_marker_agreement_section,
    add_rr_interval_section,
    compute_cardiac_residual,
    compute_rr_intervals,
    compute_run_marker_agreement,
)
from eeg_pipeline.preprocessing.report.continuity import (
    RunContinuity,
    add_continuity_section,
    compute_run_continuity,
)
from eeg_pipeline.preprocessing.report.scanner import (
    CombResidual,
    VolumeLockedAverage,
    VolumeTiming,
    CombNotMeasured,
    add_scanner_residual_section,
    compute_comb_residual,
    compute_volume_locked_average,
    measure_volume_timing,
)
from eeg_pipeline.preprocessing.report.preservation import (
    PosteriorAlpha,
    compute_posterior_alpha,
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

_FINITE_CHECK_SECONDS = 10.0


@dataclass
class RunEvidence:
    """Everything measured about the continuous runs of one report."""

    spectra: list[RunSpectra] = field(default_factory=list)
    combs: list[CombResidual] = field(default_factory=list)
    #: Runs carrying volume markers that the comb measurement declined, with the reason.
    #: Reported rather than dropped: a run absent from the comb table has not been
    #: measured and found clean, it has not been measured.
    declined_combs: list[CombNotMeasured] = field(default_factory=list)
    locked_averages: list[VolumeLockedAverage] = field(default_factory=list)
    #: Volume timing per run, keyed by recording. Kept per run rather than pooled because
    #: the repetition time sets the frequency of every comb harmonic, and a cohort that
    #: mixed repetition times would otherwise pool different harmonics into one bin.
    timings: dict[str, VolumeTiming] = field(default_factory=dict)
    continuity: list[RunContinuity] = field(default_factory=list)
    rr_intervals: list[RrIntervals] = field(default_factory=list)
    #: Runs whose R-marker train was too short to build an interval series from.
    #:
    #: Recorded rather than discarded so the beat-detection panel can name them. A run
    #: that silently drops out of that figure is indistinguishable from one that was
    #: never acquired, and the two have opposite implications for the pulse correction.
    rr_missing: list[str] = field(default_factory=list)
    #: Analyzer's marker train measured against R peaks detected from the ECG signal.
    #:
    #: Empty when no run carried an ECG channel, which is the case for a montage that
    #: recorded none: there is then one beat detector rather than two, and nothing to
    #: reconcile.
    marker_agreements: list[MarkerAgreement] = field(default_factory=list)
    #: Beat-locked EEG residual per run, measured before the ICA exclusions.
    #:
    #: What the *upstream* pulse correction left behind. A run whose Analyzer R detection
    #: failed carries no marker train, so no subtraction was possible and the
    #: ballistocardiogram is still there; this is the measurement that says how much.
    cardiac_residuals: list[CardiacResidual] = field(default_factory=list)
    gradient_fundamentals_hz: list[float] = field(default_factory=list)
    #: Where each EEG sensor sat, in head coordinates, taken from the recording itself.
    #:
    #: Captured here because this is the one place the montage is already in memory. A
    #: cohort topography needs the positions the electrodes actually had; recovering them
    #: later from a montage name would place them plausibly and, wherever the name did not
    #: match the cap, silently wrongly -- which is the one failure such a figure must not
    #: have.
    channel_positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: Channels marked bad, per run, so a cohort can take the union the sync policy implies.
    bad_channels_by_run: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Acquisition dates seen across the runs, which index cap ageing and electrode wear.
    #: Deliberately not the processing date, which indexes pipeline change instead.
    measurement_dates: list[str] = field(default_factory=list)
    #: Posterior alpha measured either side of the exclusions, per run.
    #:
    #: Measured here, on the continuous run, because this is the only point in the pipeline
    #: where the same data exists both before and after the exclusions. The preservation
    #: section's own alpha is measured on the cleaned epochs and has no counterpart: there
    #: is no pre-ICA epochs file, so "did cleaning cost this participant their rhythm" --
    #: the question the cohort's paired panel exists to answer -- is unanswerable from it.
    #:
    #: One method applied to both stages rather than two methods compared, so the pairing
    #: is a difference between stages instead of a difference between estimators.
    posterior_alpha_before: list[PosteriorAlpha] = field(default_factory=list)
    posterior_alpha_after: list[PosteriorAlpha] = field(default_factory=list)

    @property
    def has_scanner_evidence(self) -> bool:
        return bool(self.combs or self.locked_averages)

    @property
    def acquisition_date(self) -> str | None:
        """The earliest date any run was recorded on, or ``None`` for an anonymised set.

        The earliest rather than the latest, because a session split across midnight is one
        session and the date a study means by it is the day it started.
        """
        return min(self.measurement_dates) if self.measurement_dates else None

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
    timing: VolumeTiming | None,
) -> None:
    """Add whichever gradient measurements this run's volume markers support.

    ``timing`` is measured by the caller and passed in, because the aperiodic fit in the
    spectra needs it too and measuring the marker train twice would be two chances to
    disagree about where the comb is.
    """
    if timing is None:
        return
    evidence.gradient_fundamentals_hz.append(timing.fundamental_hz)
    evidence.timings[recording_id] = timing

    comb = compute_comb_residual(
        raw,
        cleaned,
        timing=timing,
        recording_id=recording_id,
        band_hz=settings.comb_frequency_range_hz,
        welch_seconds=settings.comb_welch_seconds,
        line_frequency=settings.spectra_line_frequency,
        notch_half_width_hz=settings.notch_exclusion_half_width_hz,
        unavailable_intervals=settings.unavailable_intervals_by_recording.get(recording_id),
    )
    if isinstance(comb, CombNotMeasured):
        evidence.declined_combs.append(comb)
    else:
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


def _validate_finite(raw: mne.io.BaseRaw, *, recording_id: str) -> None:
    """Reject invalid ICA output without allocating another full recording."""
    block_samples = max(1, int(round(_FINITE_CHECK_SECONDS * raw.info["sfreq"])))
    for start in range(0, raw.n_times, block_samples):
        data = raw.get_data(start=start, stop=min(start + block_samples, raw.n_times))
        if not np.isfinite(data).all():
            raise ValueError(f"{recording_id}: non-finite values in ICA output.")


def measure_runs(
    *,
    filtered_raw_paths: Sequence[Path],
    ica: mne.preprocessing.ICA,
    settings: ReportSettings,
    edge_support_seconds: float = 0.0,
) -> RunEvidence:
    """Measure every per-run panel, reading and cleaning each run exactly once."""
    if not filtered_raw_paths:
        raise ValueError("Per-run evidence requires at least one filtered run.")

    evidence = RunEvidence()
    for path in filtered_raw_paths:
        recording_id = path.name.removesuffix("_proc-filt_raw.fif")
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        cleaned = ica.apply(raw.copy(), exclude=ica.exclude, verbose="ERROR")
        _validate_finite(cleaned, recording_id=recording_id)

        # Measured before the spectra, because the aperiodic fit inside them has to know
        # where the comb is: the harmonics run through the fit range, and a line fitted
        # across them is fitted partly to the scanner. Costs nothing extra -- the timing is
        # measured from the marker train, which the gradient section needs anyway.
        timing = measure_volume_timing(
            raw,
            description=settings.volume_marker_description,
        )

        evidence.spectra.append(
            compute_run_spectra(
                raw,
                cleaned,
                recording_id=recording_id,
                fmax=settings.spectra_fmax,
                line_frequency=settings.spectra_line_frequency,
                gradient_fundamental_hz=None if timing is None else timing.fundamental_hz,
                aperiodic_fit_range_hz=settings.aperiodic_fit_range_hz,
                notch_half_width_hz=settings.notch_exclusion_half_width_hz,
                aperiodic_exclude_hz=settings.aperiodic_exclude_hz,
                unavailable_intervals=settings.unavailable_intervals_by_recording.get(
                    recording_id
                ),
            )
        )
        evidence.continuity.append(
            compute_run_continuity(
                raw,
                recording_id=recording_id,
                window_seconds=settings.continuity_window_seconds,
                edge_support_seconds=edge_support_seconds,
                volume_description=settings.volume_marker_description,
                pulse_description=settings.pulse_marker_description,
                non_event_prefixes=settings.non_event_prefixes,
            )
        )
        intervals = compute_rr_intervals(
            raw,
            recording_id=recording_id,
            description=settings.pulse_marker_description,
        )
        if intervals is not None:
            evidence.rr_intervals.append(intervals)
        else:
            evidence.rr_missing.append(recording_id)

        agreement = compute_run_marker_agreement(
            raw,
            recording_id=recording_id,
            description=settings.pulse_marker_description,
            tolerance_s=settings.marker_agreement_tolerance_s,
        )
        if agreement is not None:
            evidence.marker_agreements.append(agreement)

        # Measured on ``raw`` rather than ``cleaned``: the question is what the upstream
        # pulse correction left, and measuring after the exclusions would credit Analyzer
        # for whatever MNE's decomposition removed. Costs one epoching pass over a
        # recording already in memory.
        evidence.cardiac_residuals.append(
            compute_cardiac_residual(
                raw,
                recording_id=recording_id,
                marker_description=settings.pulse_marker_description,
                window_s=settings.bcg_residual_window_s,
                baseline_s=settings.bcg_residual_baseline_s,
                measurement_s=settings.bcg_residual_measurement_s,
            )
        )

        _measure_gradient(
            raw,
            cleaned,
            recording_id=recording_id,
            settings=settings,
            volume_description=settings.volume_marker_description,
            evidence=evidence,
            timing=timing,
        )
        # Both stages, on the same run, by the same estimator: the only paired measurement
        # of the rhythm the pipeline can make. A stage that measured nothing contributes
        # nothing rather than a zero, and the pair is only used where both sides exist.
        before_alpha = compute_posterior_alpha(
            raw,
            band_hz=settings.alpha_band_hz,
            reference_band_hz=settings.alpha_reference_band_hz,
            pattern=settings.posterior_channel_pattern,
        )
        after_alpha = compute_posterior_alpha(
            cleaned,
            band_hz=settings.alpha_band_hz,
            reference_band_hz=settings.alpha_reference_band_hz,
            pattern=settings.posterior_channel_pattern,
        )
        if before_alpha is not None and after_alpha is not None:
            evidence.posterior_alpha_before.append(before_alpha)
            evidence.posterior_alpha_after.append(after_alpha)

        _record_montage(raw, recording_id=recording_id, evidence=evidence)
        # Release both copies before the next run is read.
        del raw, cleaned
    return evidence


def _record_montage(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    evidence: RunEvidence,
) -> None:
    """Note where the sensors were, which were bad, and when the run was recorded.

    None of this is a measurement, and all of it is needed by a cohort. It is taken here
    because the recording is already open: reading a gigabyte of filtered raw a second time
    to recover a sensor position would cost more than every measurement above put together.

    Positions that are absent or non-finite are skipped rather than stored as the origin.
    A sensor recorded at (0, 0, 0) is not at the centre of the head; it is a sensor whose
    position was never digitised, and a topography that believes otherwise draws every one
    of them on top of each other.
    """
    for channel in mne.pick_types(raw.info, eeg=True, exclude=()):
        entry = raw.info["chs"][channel]
        position = tuple(float(value) for value in entry["loc"][:3])
        if not all(map(np.isfinite, position)) or not any(position):
            continue
        evidence.channel_positions.setdefault(str(entry["ch_name"]), position)

    evidence.bad_channels_by_run[recording_id] = tuple(
        str(name) for name in raw.info.get("bads", ())
    )
    measured = raw.info.get("meas_date")
    if measured is not None:
        evidence.measurement_dates.append(measured.date().isoformat())


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
            notch_half_width_hz=settings.notch_exclusion_half_width_hz,
            unavailable_intervals_by_recording=settings.unavailable_intervals_by_recording,
        )
    if evidence.has_scanner_evidence:
        add_scanner_residual_section(
            report=report,
            combs=evidence.combs,
            averages=evidence.locked_averages,
            declined=evidence.declined_combs,
        )
    if evidence.continuity:
        add_continuity_section(report=report, runs=evidence.continuity)
    if evidence.rr_intervals:
        add_rr_interval_section(
            report=report,
            series=evidence.rr_intervals,
            missing=evidence.rr_missing,
            plausible_rr_range_s=settings.plausible_rr_range_s,
        )
    if evidence.marker_agreements:
        add_marker_agreement_section(
            report=report,
            agreements=evidence.marker_agreements,
        )


def add_run_evidence_review(
    *,
    report: mne.Report,
    filtered_raw_paths: Sequence[Path],
    ica: mne.preprocessing.ICA,
    settings: ReportSettings,
    edge_support_seconds: float = 0.0,
) -> RunEvidence:
    """Measure every run once and append all the per-run sections."""
    evidence = measure_runs(
        filtered_raw_paths=filtered_raw_paths,
        ica=ica,
        settings=settings,
        edge_support_seconds=edge_support_seconds,
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
