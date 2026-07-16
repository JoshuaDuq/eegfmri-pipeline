"""Participant-equal scanner-harmonic spectra for MNE preprocessing QC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mne
import numpy as np

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    DEFAULT_HARMONIC_WINDOWS,
    select_harmonic_peak,
)


@dataclass(frozen=True)
class ScannerCombParameters:
    """Validated numerical settings for matched scanner-comb spectra."""

    frequency_min_hz: float = 15.0
    frequency_max_hz: float = 90.0
    welch_duration_seconds: float = 4.0
    frequency_resolution_hz: float = 0.25
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    random_seed: int = 42

    def __post_init__(self) -> None:
        numeric_values = (
            self.frequency_min_hz,
            self.frequency_max_hz,
            self.welch_duration_seconds,
            self.frequency_resolution_hz,
            self.confidence_level,
        )
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError("Scanner-comb parameters must be finite.")
        if self.frequency_min_hz <= 0:
            raise ValueError("frequency_min_hz must be positive.")
        if self.frequency_min_hz >= self.frequency_max_hz:
            raise ValueError("frequency_min_hz must be below frequency_max_hz.")
        if self.welch_duration_seconds <= 0:
            raise ValueError("welch_duration_seconds must be positive.")
        if self.frequency_resolution_hz <= 0:
            raise ValueError("frequency_resolution_hz must be positive.")

        frequency_span_bins = (
            self.frequency_max_hz - self.frequency_min_hz
        ) / self.frequency_resolution_hz
        if not np.isclose(frequency_span_bins, round(frequency_span_bins)):
            raise ValueError("The frequency span must contain an integer number of bins.")
        expected_resolution = 1.0 / self.welch_duration_seconds
        if not np.isclose(self.frequency_resolution_hz, expected_resolution):
            raise ValueError("frequency_resolution_hz must equal 1 / welch_duration_seconds.")
        if isinstance(self.bootstrap_resamples, bool) or self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be a positive integer.")
        if int(self.bootstrap_resamples) != self.bootstrap_resamples:
            raise ValueError("bootstrap_resamples must be a positive integer.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        if isinstance(self.random_seed, bool) or int(self.random_seed) != self.random_seed:
            raise ValueError("random_seed must be an integer.")


@dataclass(frozen=True)
class Spectrum:
    """One stage spectrum on the shared frequency grid."""

    frequencies_hz: np.ndarray
    power_db: np.ndarray

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        power = np.asarray(self.power_db, dtype=float)
        if frequencies.ndim != 1 or frequencies.size < 2:
            raise ValueError("frequencies_hz must be a one-dimensional grid.")
        if power.shape != frequencies.shape:
            raise ValueError("power_db must match frequencies_hz.")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(power)):
            raise ValueError("Spectrum values must be finite.")
        if not np.all(np.diff(frequencies) > 0):
            raise ValueError("frequencies_hz must be strictly increasing.")


@dataclass(frozen=True)
class ParticipantSpectrum(Spectrum):
    """One participant spectrum after within-participant aggregation."""

    participant: str

    def __init__(
        self,
        participant: str,
        frequencies_hz: np.ndarray,
        power_db: np.ndarray,
    ) -> None:
        if not str(participant).strip():
            raise ValueError("participant must be non-empty.")
        object.__setattr__(self, "participant", str(participant))
        object.__setattr__(self, "frequencies_hz", np.asarray(frequencies_hz, dtype=float))
        object.__setattr__(self, "power_db", np.asarray(power_db, dtype=float))
        Spectrum.__post_init__(self)


@dataclass(frozen=True)
class ScannerCombSummary:
    """Cohort spectrum and participant-bootstrap confidence limits."""

    participant_ids: tuple[str, ...]
    frequencies_hz: np.ndarray
    input_median_db: np.ndarray
    input_ci_low_db: np.ndarray
    input_ci_high_db: np.ndarray
    final_median_db: np.ndarray
    final_ci_low_db: np.ndarray
    final_ci_high_db: np.ndarray
    harmonic_frequencies_hz: tuple[float, ...]

    @property
    def participant_count(self) -> int:
        return len(self.participant_ids)


def compute_raw_comb_spectrum(
    raw: mne.io.BaseRaw,
    parameters: ScannerCombParameters,
) -> Spectrum:
    """Compute a channel-median Welch spectrum from continuous EEG."""
    return _compute_welch_spectrum(raw, parameters)


def compute_epoch_comb_spectrum(
    epochs: mne.BaseEpochs,
    parameters: ScannerCombParameters,
) -> Spectrum:
    """Compute Welch spectra within epochs before aggregating epochs."""
    if len(epochs) == 0:
        raise ValueError("Scanner-comb QC requires at least one retained epoch.")
    return _compute_welch_spectrum(epochs, parameters)


def _compute_welch_spectrum(
    instance: mne.io.BaseRaw | mne.BaseEpochs,
    parameters: ScannerCombParameters,
) -> Spectrum:
    sampling_frequency = float(instance.info["sfreq"])
    if parameters.frequency_max_hz > sampling_frequency / 2.0:
        raise ValueError("frequency_max_hz exceeds the data Nyquist frequency.")

    n_per_segment = _integer_samples(
        parameters.welch_duration_seconds * sampling_frequency,
        "Welch duration",
    )
    n_fft = _integer_samples(
        sampling_frequency / parameters.frequency_resolution_hz,
        "Frequency resolution",
    )
    available_samples = len(instance.times)
    if available_samples < n_per_segment:
        raise ValueError(
            "Scanner-comb Welch duration exceeds the available samples: "
            f"{n_per_segment} > {available_samples}."
        )

    picks = mne.pick_types(instance.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("Scanner-comb QC requires at least one non-bad EEG channel.")
    spectrum = instance.compute_psd(
        method="welch",
        fmin=parameters.frequency_min_hz,
        fmax=parameters.frequency_max_hz,
        picks=picks,
        n_fft=n_fft,
        n_per_seg=n_per_segment,
        n_overlap=n_per_segment // 2,
        average="median",
        window="hamming",
        remove_dc=True,
        verbose=False,
    )
    power, frequencies = spectrum.get_data(return_freqs=True)
    power = np.asarray(power, dtype=float)
    if power.shape[-1] != frequencies.size:
        raise RuntimeError("PSD frequency axis does not match the returned frequency grid.")
    if not np.all(np.isfinite(power)) or np.any(power <= 0):
        raise ValueError("Scanner-comb PSD must contain finite positive power values.")

    aggregate_axes = tuple(range(power.ndim - 1))
    median_power = np.median(power, axis=aggregate_axes)
    power_db = 10.0 * np.log10(median_power)
    return Spectrum(np.asarray(frequencies, dtype=float), power_db)


def _integer_samples(value: float, name: str) -> int:
    rounded = int(round(value))
    if rounded < 1 or not np.isclose(value, rounded):
        raise ValueError(f"{name} must map to an integer number of samples.")
    return rounded


def combine_participant_runs(
    participant: str,
    run_spectra: Sequence[Spectrum],
) -> ParticipantSpectrum:
    """Aggregate runs equally within one participant."""
    if not run_spectra:
        raise ValueError(f"No input run spectra were provided for sub-{participant}.")
    frequencies = _require_identical_grids(run_spectra)
    power = np.median(np.stack([item.power_db for item in run_spectra]), axis=0)
    return ParticipantSpectrum(participant, frequencies, power)


def summarize_scanner_comb(
    input_spectra: Sequence[ParticipantSpectrum],
    final_spectra: Sequence[ParticipantSpectrum],
    parameters: ScannerCombParameters,
) -> ScannerCombSummary:
    """Build an equal-participant cohort summary with paired bootstrap draws."""
    input_by_participant = _index_participants(input_spectra, "input")
    final_by_participant = _index_participants(final_spectra, "final")
    if input_by_participant.keys() != final_by_participant.keys():
        raise ValueError("Input and final spectra must contain identical participants.")

    participant_ids = tuple(sorted(input_by_participant))
    if not participant_ids:
        raise ValueError("Scanner-comb cohort requires at least one participant.")
    ordered_spectra = [input_by_participant[participant] for participant in participant_ids] + [
        final_by_participant[participant] for participant in participant_ids
    ]
    frequencies = _require_identical_grids(ordered_spectra)
    input_values = np.stack(
        [input_by_participant[participant].power_db for participant in participant_ids]
    )
    final_values = np.stack(
        [final_by_participant[participant].power_db for participant in participant_ids]
    )
    input_low, input_high, final_low, final_high = _paired_bootstrap_intervals(
        input_values,
        final_values,
        parameters,
    )
    input_median = np.median(input_values, axis=0)
    final_median = np.median(final_values, axis=0)
    harmonic_frequencies = tuple(
        select_harmonic_peak(frequencies, input_median, window)[0]
        for window in DEFAULT_HARMONIC_WINDOWS
    )
    return ScannerCombSummary(
        participant_ids=participant_ids,
        frequencies_hz=frequencies,
        input_median_db=input_median,
        input_ci_low_db=input_low,
        input_ci_high_db=input_high,
        final_median_db=final_median,
        final_ci_low_db=final_low,
        final_ci_high_db=final_high,
        harmonic_frequencies_hz=harmonic_frequencies,
    )


def _index_participants(
    spectra: Sequence[ParticipantSpectrum],
    stage: str,
) -> dict[str, ParticipantSpectrum]:
    indexed: dict[str, ParticipantSpectrum] = {}
    for spectrum in spectra:
        if spectrum.participant in indexed:
            raise ValueError(f"Duplicate {stage} spectrum for sub-{spectrum.participant}.")
        indexed[spectrum.participant] = spectrum
    return indexed


def _require_identical_grids(spectra: Sequence[Spectrum]) -> np.ndarray:
    if not spectra:
        raise ValueError("At least one spectrum is required.")
    reference = spectra[0].frequencies_hz
    for spectrum in spectra[1:]:
        if not np.array_equal(reference, spectrum.frequencies_hz):
            raise ValueError("Scanner-comb spectra must use identical frequency grids.")
    return reference.copy()


def _paired_bootstrap_intervals(
    input_values: np.ndarray,
    final_values: np.ndarray,
    parameters: ScannerCombParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    participant_count, frequency_count = input_values.shape
    rng = np.random.default_rng(parameters.random_seed)
    input_bootstrap = np.empty(
        (parameters.bootstrap_resamples, frequency_count),
        dtype=float,
    )
    final_bootstrap = np.empty_like(input_bootstrap)
    batch_size = 256
    for start in range(0, parameters.bootstrap_resamples, batch_size):
        stop = min(start + batch_size, parameters.bootstrap_resamples)
        indices = rng.integers(
            0,
            participant_count,
            size=(stop - start, participant_count),
        )
        input_bootstrap[start:stop] = np.median(input_values[indices], axis=1)
        final_bootstrap[start:stop] = np.median(final_values[indices], axis=1)

    alpha = 100.0 * (1.0 - parameters.confidence_level) / 2.0
    input_low, input_high = np.percentile(
        input_bootstrap,
        [alpha, 100.0 - alpha],
        axis=0,
    )
    final_low, final_high = np.percentile(
        final_bootstrap,
        [alpha, 100.0 - alpha],
        axis=0,
    )
    return input_low, input_high, final_low, final_high


__all__ = [
    "ParticipantSpectrum",
    "ScannerCombParameters",
    "ScannerCombSummary",
    "Spectrum",
    "combine_participant_runs",
    "compute_epoch_comb_spectrum",
    "compute_raw_comb_spectrum",
    "summarize_scanner_comb",
]
