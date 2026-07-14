"""Outcome-blind QC for residual scanner-gradient correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    FrequencyWindow,
    summarize_scanner_harmonics,
)
from eeg_pipeline.preprocessing.residual_gradient import VolumeLayout

HARMONIC_LABELS = ("18_23", "38_43", "56_67", "77_85")


@dataclass(frozen=True)
class InjectionSettings:
    """Fixed non-volume-locked validation signals."""

    frequencies_hz: tuple[float, ...] = (15.5, 26.0, 34.0, 49.5, 72.0)
    sinusoid_amplitude_uv: float = 1.0
    transient_amplitude_uv: float = 2.0
    transient_spacing_s: float = 13.7
    transient_width_s: float = 0.08

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a non-empty one-dimensional sequence.")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("frequencies_hz must contain finite positive values.")
        if np.unique(frequencies).size != frequencies.size:
            raise ValueError("frequencies_hz must not contain duplicates.")
        positive_values = (
            self.sinusoid_amplitude_uv,
            self.transient_amplitude_uv,
            self.transient_spacing_s,
            self.transient_width_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive_values):
            raise ValueError(
                "Injection amplitudes, spacing, and width must be finite and positive."
            )


@dataclass(frozen=True)
class ValidationInjection:
    signal_v: NDArray[np.float64]
    sinusoid_v: NDArray[np.float64]
    transient_v: NDArray[np.float64]
    transient_samples: tuple[int, ...]
    frequencies_hz: tuple[float, ...]


@dataclass(frozen=True)
class RecoveryMetrics:
    minimum_sinusoid_amplitude_ratio: float
    maximum_phase_error_deg: float
    transient_peak_ratio: float


@dataclass(frozen=True)
class PsdChangeMetrics:
    median_change_db: float
    maximum_channel_absolute_change_db: float


@dataclass(frozen=True)
class SelectionThresholds:
    minimum_sinusoid_amplitude_ratio: float = 0.95
    maximum_phase_error_deg: float = 5.0
    minimum_transient_peak_ratio: float = 0.95
    maximum_outside_harmonic_psd_change_db: float = 0.5
    maximum_run_prominence_increase_db: float = 1.0

    def __post_init__(self) -> None:
        ratios = (
            self.minimum_sinusoid_amplitude_ratio,
            self.minimum_transient_peak_ratio,
        )
        if not all(0 < ratio <= 1 for ratio in ratios):
            raise ValueError("Minimum preservation ratios must be in (0, 1].")
        upper_bounds = (
            self.maximum_phase_error_deg,
            self.maximum_outside_harmonic_psd_change_db,
            self.maximum_run_prominence_increase_db,
        )
        if not all(np.isfinite(value) and value >= 0 for value in upper_bounds):
            raise ValueError("Maximum preservation thresholds must be finite and non-negative.")


@dataclass(frozen=True)
class ComponentDecision:
    status: str
    selected_components: int | None
    evaluated_components: tuple[int, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_validation_injection(
    n_times: int,
    sfreq: float,
    settings: InjectionSettings,
) -> ValidationInjection:
    """Build deterministic sinusoids and non-volume-locked Gaussian transients."""
    if n_times <= 0 or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("n_times and sfreq must be positive.")
    if max(settings.frequencies_hz) >= sfreq / 2:
        raise ValueError("Injection frequencies must be below the Nyquist frequency.")

    time = np.arange(n_times, dtype=float) / sfreq
    phases = np.linspace(0.17, 1.31, len(settings.frequencies_hz))
    components = [
        np.sin(2 * np.pi * frequency * time + phase)
        for frequency, phase in zip(settings.frequencies_hz, phases, strict=True)
    ]
    sinusoid = settings.sinusoid_amplitude_uv * 1e-6 * np.mean(components, axis=0)

    event_times = np.arange(5.35, time[-1] - 1.0, settings.transient_spacing_s)
    transient_samples = tuple(np.rint(event_times * sfreq).astype(int))
    if not transient_samples:
        raise ValueError("Recording is too short to contain a validation transient.")
    transient = np.zeros(n_times, dtype=float)
    width_samples = settings.transient_width_s * sfreq
    support = np.arange(n_times)
    for sample in transient_samples:
        transient += np.exp(-0.5 * ((support - sample) / width_samples) ** 2)
    transient *= settings.transient_amplitude_uv * 1e-6

    return ValidationInjection(
        signal_v=sinusoid + transient,
        sinusoid_v=sinusoid,
        transient_v=transient,
        transient_samples=transient_samples,
        frequencies_hz=settings.frequencies_hz,
    )


def evaluate_injection_recovery(
    *,
    expected: ValidationInjection,
    recovered_v: NDArray[np.float64],
    sfreq: float,
) -> RecoveryMetrics:
    """Measure sinusoid and transient preservation after correction."""
    recovered = np.asarray(recovered_v, dtype=float)
    if recovered.shape != expected.signal_v.shape:
        raise ValueError("recovered_v shape must match the validation injection.")
    if not np.isfinite(recovered).all():
        raise ValueError("recovered_v must contain only finite values.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive.")

    time = np.arange(recovered.size, dtype=float) / sfreq
    design_columns: list[NDArray[np.float64]] = []
    for frequency in expected.frequencies_hz:
        angle = 2 * np.pi * frequency * time
        design_columns.extend([np.sin(angle), np.cos(angle)])
    design = np.column_stack(design_columns)
    expected_coefficients = np.linalg.lstsq(design, expected.signal_v, rcond=None)[0]
    recovered_coefficients = np.linalg.lstsq(design, recovered, rcond=None)[0]

    amplitude_ratios: list[float] = []
    phase_errors: list[float] = []
    for index in range(len(expected.frequencies_hz)):
        pair = slice(2 * index, 2 * index + 2)
        expected_pair = expected_coefficients[pair]
        recovered_pair = recovered_coefficients[pair]
        amplitude_ratios.append(
            float(np.linalg.norm(recovered_pair) / np.linalg.norm(expected_pair))
        )
        expected_phase = np.arctan2(expected_pair[1], expected_pair[0])
        recovered_phase = np.arctan2(recovered_pair[1], recovered_pair[0])
        phase_error = np.angle(np.exp(1j * (recovered_phase - expected_phase)), deg=True)
        phase_errors.append(abs(float(phase_error)))

    expected_residual = expected.signal_v - design @ expected_coefficients
    recovered_residual = recovered - design @ recovered_coefficients
    expected_peak = max(expected_residual[sample] for sample in expected.transient_samples)
    recovered_peak = max(recovered_residual[sample] for sample in expected.transient_samples)
    return RecoveryMetrics(
        minimum_sinusoid_amplitude_ratio=min(amplitude_ratios),
        maximum_phase_error_deg=max(phase_errors),
        transient_peak_ratio=float(recovered_peak / expected_peak),
    )


def compute_volume_locked_rms(
    data_v: NDArray[np.float64],
    starts: NDArray[np.int64],
    epoch_samples: int,
) -> float:
    """Return RMS of the across-volume mean waveform."""
    data = np.asarray(data_v, dtype=float)
    starts_array = np.asarray(starts, dtype=np.int64)
    if data.ndim != 2:
        raise ValueError("data_v must have shape (channels, samples).")
    if starts_array.ndim != 1 or starts_array.size == 0:
        raise ValueError("starts must be a non-empty one-dimensional array.")
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be positive.")
    if np.any(starts_array < 0) or np.any(starts_array + epoch_samples > data.shape[1]):
        raise ValueError("Volume epochs must fall within data_v.")

    offsets = np.arange(epoch_samples, dtype=np.int64)
    epochs = data[:, starts_array[:, np.newaxis] + offsets[np.newaxis, :]]
    phase_locked_mean = epochs.mean(axis=1)
    return float(np.sqrt(np.mean(phase_locked_mean**2)))


def compute_outside_harmonic_psd_change(
    before_v: NDArray[np.float64],
    after_v: NDArray[np.float64],
    *,
    sfreq: float,
    nperseg: int,
    overlap_fraction: float,
    harmonic_windows_hz: Sequence[tuple[float, float]],
) -> PsdChangeMetrics:
    """Measure channel-wise median PSD change outside scanner windows."""
    before = np.asarray(before_v, dtype=float)
    after = np.asarray(after_v, dtype=float)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("before_v and after_v must share shape (channels, samples).")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("PSD inputs must contain only finite values.")
    if nperseg <= 0 or nperseg > before.shape[1]:
        raise ValueError("nperseg must be positive and no greater than the sample count.")
    noverlap = _welch_overlap(nperseg, overlap_fraction)

    frequencies, before_psd = welch(
        before,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )
    _, after_psd = welch(
        after,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )
    retained = (frequencies >= 13.0) & (frequencies <= 95.0)
    for low_hz, high_hz in harmonic_windows_hz:
        retained &= ~((frequencies >= low_hz) & (frequencies <= high_hz))
    if not np.any(retained):
        raise ValueError("No PSD frequencies remain outside the harmonic windows.")

    floor = np.finfo(float).tiny
    before_db = 10 * np.log10(np.maximum(before_psd, floor))
    after_db = 10 * np.log10(np.maximum(after_psd, floor))
    channel_changes = np.median(after_db[:, retained] - before_db[:, retained], axis=1)
    return PsdChangeMetrics(
        median_change_db=float(np.median(channel_changes)),
        maximum_channel_absolute_change_db=float(np.max(np.abs(channel_changes))),
    )


def summarize_candidate(
    raw: mne.io.BaseRaw,
    *,
    layout: VolumeLayout,
    source_file: str | Path,
    run: int,
    n_components: int,
    nperseg: int,
    overlap_fraction: float,
    harmonic_windows_hz: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Summarize harmonic peaks and volume-locked RMS for one candidate."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks)
    if nperseg <= 0 or nperseg > raw.n_times:
        raise ValueError("nperseg must be positive and no greater than the sample count.")
    noverlap = _welch_overlap(nperseg, overlap_fraction)
    frequencies, psd = welch(
        data,
        fs=raw.info["sfreq"],
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )
    windows = tuple(
        FrequencyWindow(f"scanner_{low_hz:g}_{high_hz:g}", low_hz, high_hz)
        for low_hz, high_hz in harmonic_windows_hz
    )
    row = summarize_scanner_harmonics(
        freqs=frequencies,
        psd=psd,
        source_file=source_file,
        sfreq=float(raw.info["sfreq"]),
        n_samples=raw.n_times,
        channel_names=[raw.ch_names[pick] for pick in picks],
        harmonic_windows=windows,
    )
    row["run"] = run
    row["n_components"] = n_components
    row["volume_locked_rms"] = compute_volume_locked_rms(
        data,
        layout.starts,
        layout.epoch_samples,
    )
    return row


def select_component_count(
    run_rows: Sequence[dict[str, Any]],
    preservation_rows: Sequence[dict[str, Any]],
    component_counts: Sequence[int],
    thresholds: SelectionThresholds,
) -> ComponentDecision:
    """Select the smallest nonzero order that passes every fixed gate."""
    evaluated = tuple(component_counts)
    if not evaluated or evaluated[0] != 0 or len(set(evaluated)) != len(evaluated):
        raise ValueError("component_counts must start with zero and contain unique values.")

    references = [row for row in run_rows if row["n_components"] == 0]
    if not references:
        raise ValueError("At least one zero-component reference row is required.")
    reference_by_run = {int(row["run"]): row for row in references}
    if len(reference_by_run) != len(references):
        raise ValueError("Reference run rows must be unique.")

    reasons: list[str] = []
    for count in sorted(value for value in evaluated if value > 0):
        candidates = [row for row in run_rows if row["n_components"] == count]
        if {int(row["run"]) for row in candidates} != set(reference_by_run):
            raise ValueError(f"Candidate runs do not match references for components={count}.")
        preservation = [row for row in preservation_rows if row["n_components"] == count]
        if not preservation:
            raise ValueError(f"No signal-preservation rows for components={count}.")

        eligible = _passes_preservation_gates(preservation, thresholds)
        for label in HARMONIC_LABELS:
            eligible &= _improves_harmonic_window(
                candidates,
                references,
                reference_by_run,
                label,
                thresholds.maximum_run_prominence_increase_db,
            )
        eligible &= np.median([row["volume_locked_rms"] for row in candidates]) < np.median(
            [row["volume_locked_rms"] for row in references]
        )
        if eligible:
            return ComponentDecision("accepted", count, evaluated, ())
        reasons.append(f"components={count} failed one or more fixed gates")

    return ComponentDecision("rejected", None, evaluated, tuple(reasons))


def _passes_preservation_gates(
    rows: Sequence[dict[str, Any]],
    thresholds: SelectionThresholds,
) -> bool:
    return bool(
        min(row["minimum_sinusoid_amplitude_ratio"] for row in rows)
        >= thresholds.minimum_sinusoid_amplitude_ratio
        and max(row["maximum_phase_error_deg"] for row in rows)
        <= thresholds.maximum_phase_error_deg
        and min(row["transient_peak_ratio"] for row in rows)
        >= thresholds.minimum_transient_peak_ratio
        and max(abs(row["outside_harmonic_psd_change_db"]) for row in rows)
        <= thresholds.maximum_outside_harmonic_psd_change_db
    )


def _welch_overlap(nperseg: int, overlap_fraction: float) -> int:
    if not np.isfinite(overlap_fraction) or not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1).")
    return int(nperseg * overlap_fraction)


def _improves_harmonic_window(
    candidates: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
    reference_by_run: dict[int, dict[str, Any]],
    label: str,
    maximum_run_increase_db: float,
) -> bool:
    peak_column = f"harmonic_{label}_peak_power_db"
    prominence_column = f"harmonic_{label}_prominence_db"
    candidate_power = np.median([row[peak_column] for row in candidates])
    reference_power = np.median([row[peak_column] for row in references])
    candidate_prominence = np.median([row[prominence_column] for row in candidates])
    reference_prominence = np.median([row[prominence_column] for row in references])
    no_run_worsened = all(
        row[prominence_column]
        <= reference_by_run[int(row["run"])][prominence_column] + maximum_run_increase_db
        for row in candidates
    )
    return bool(
        candidate_power < reference_power
        and candidate_prominence < reference_prominence
        and no_run_worsened
    )
