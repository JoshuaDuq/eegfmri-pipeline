"""Qualification metrics for native post-AAS residual gradient OBS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class NativeObsSelectionThresholds:
    """Prespecified suppression and signal-preservation gates."""

    minimum_line_power_reduction_db: float = 1.0
    minimum_line_prominence_reduction_db: float = 1.0
    maximum_recording_prominence_increase_db: float = 1.0
    minimum_volume_locked_rms_reduction_fraction: float = 0.01
    minimum_sinusoid_amplitude_ratio: float = 0.95
    maximum_phase_error_deg: float = 5.0
    minimum_transient_peak_ratio: float = 0.95
    maximum_outside_line_psd_change_db: float = 0.5
    maximum_channel_outside_line_psd_change_db: float = 1.0

    def __post_init__(self) -> None:
        nonnegative = (
            self.minimum_line_power_reduction_db,
            self.minimum_line_prominence_reduction_db,
            self.maximum_recording_prominence_increase_db,
            self.minimum_volume_locked_rms_reduction_fraction,
            self.maximum_phase_error_deg,
            self.maximum_outside_line_psd_change_db,
            self.maximum_channel_outside_line_psd_change_db,
        )
        if not all(np.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError("OBS thresholds must be finite and non-negative")
        if self.minimum_volume_locked_rms_reduction_fraction >= 1:
            raise ValueError("minimum_volume_locked_rms_reduction_fraction must be below 1")
        ratios = (
            self.minimum_sinusoid_amplitude_ratio,
            self.minimum_transient_peak_ratio,
        )
        if not all(np.isfinite(value) and 0 < value <= 1 for value in ratios):
            raise ValueError("Signal-preservation ratios must be in (0, 1]")


@dataclass(frozen=True)
class NativeObsDecision:
    """Outcome of the fixed component-order qualification."""

    status: str
    selected_components: int | None
    evaluated_components: tuple[int, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_fixed_frequency_lines(
    data_v: np.ndarray,
    *,
    sampling_frequency: float,
    reference_frequencies_hz: Sequence[float],
    welch_duration_seconds: float,
    overlap_fraction: float,
    background_inner_hz: float,
    background_outer_hz: float,
) -> tuple[dict[str, float], ...]:
    """Measure fixed scanner-line power and prominence on a median EEG PSD."""
    data = np.asarray(data_v, dtype=float)
    references = np.asarray(reference_frequencies_hz, dtype=float)
    if data.ndim != 2 or min(data.shape) < 1:
        raise ValueError("data_v must have shape (channels, samples)")
    if not np.all(np.isfinite(data)):
        raise ValueError("data_v contains non-finite samples")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if references.ndim != 1 or references.size == 0:
        raise ValueError("reference_frequencies_hz must be a non-empty sequence")
    if not np.all(np.isfinite(references)) or np.any(references <= 0):
        raise ValueError("reference frequencies must be finite and positive")
    if np.unique(references).size != references.size:
        raise ValueError("reference frequencies must be unique")
    if np.any(references >= sampling_frequency / 2):
        raise ValueError("reference frequencies must be below Nyquist")
    if not np.isfinite(welch_duration_seconds) or welch_duration_seconds <= 0:
        raise ValueError("welch_duration_seconds must be finite and positive")
    if not np.isfinite(overlap_fraction) or not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    if not 0 < background_inner_hz < background_outer_hz:
        raise ValueError("background widths must satisfy 0 < inner < outer")

    segment_samples = int(round(welch_duration_seconds * sampling_frequency))
    if segment_samples > data.shape[1]:
        raise ValueError("Welch duration exceeds the recording duration")
    frequencies, channel_psd = welch(
        data,
        fs=sampling_frequency,
        nperseg=segment_samples,
        noverlap=int(segment_samples * overlap_fraction),
        axis=-1,
    )
    median_psd = np.median(channel_psd, axis=0)
    median_power_db = 10.0 * np.log10(np.maximum(median_psd, np.finfo(float).tiny))

    rows = []
    for reference in references:
        line_index = int(np.argmin(np.abs(frequencies - reference)))
        distance = np.abs(frequencies - reference)
        background = (distance >= background_inner_hz) & (distance <= background_outer_hz)
        if np.count_nonzero(background) < 2:
            raise ValueError(f"Insufficient background bins around {reference:g} Hz")
        power_db = float(median_power_db[line_index])
        rows.append(
            {
                "reference_frequency_hz": float(reference),
                "evaluated_frequency_hz": float(frequencies[line_index]),
                "power_db": power_db,
                "local_prominence_db": power_db - float(np.median(median_power_db[background])),
            }
        )
    return tuple(rows)


def select_native_obs_component_count(
    *,
    line_rows: Sequence[dict[str, Any]],
    run_rows: Sequence[dict[str, Any]],
    preservation_rows: Sequence[dict[str, Any]],
    component_counts: Sequence[int],
    thresholds: NativeObsSelectionThresholds,
) -> NativeObsDecision:
    """Select the smallest nonzero order satisfying every qualification gate."""
    evaluated = tuple(int(value) for value in component_counts)
    if not evaluated or evaluated[0] != 0 or len(set(evaluated)) != len(evaluated):
        raise ValueError("component_counts must start with zero and contain unique values")
    if any(value < 0 for value in evaluated):
        raise ValueError("component_counts must be non-negative")

    reference_lines = _unique_rows(
        [row for row in line_rows if int(row["n_components"]) == 0],
        key_fields=("recording_id", "reference_frequency_hz"),
        context="zero-component line",
    )
    reference_runs = _unique_rows(
        [row for row in run_rows if int(row["n_components"]) == 0],
        key_fields=("recording_id",),
        context="zero-component run",
    )
    if not reference_lines or not reference_runs:
        raise ValueError("Zero-component line and run references are required")

    expected_line_keys = set(reference_lines)
    expected_recordings = set(reference_runs)
    reference_frequencies = sorted({float(key[1]) for key in expected_line_keys})
    reasons: list[str] = []
    for components in sorted(value for value in evaluated if value > 0):
        candidate_lines = _unique_rows(
            [row for row in line_rows if int(row["n_components"]) == components],
            key_fields=("recording_id", "reference_frequency_hz"),
            context=f"components={components} line",
        )
        candidate_runs = _unique_rows(
            [row for row in run_rows if int(row["n_components"]) == components],
            key_fields=("recording_id",),
            context=f"components={components} run",
        )
        candidate_preservation = _unique_rows(
            [row for row in preservation_rows if int(row["n_components"]) == components],
            key_fields=("recording_id",),
            context=f"components={components} preservation",
        )
        if set(candidate_lines) != expected_line_keys:
            raise ValueError(f"Line rows do not match references for components={components}")
        if set(candidate_runs) != expected_recordings:
            raise ValueError(f"Run rows do not match references for components={components}")
        if set(candidate_preservation) != expected_recordings:
            raise ValueError(
                f"Preservation rows do not match references for components={components}"
            )

        failures = _line_failures(
            components=components,
            reference_frequencies=reference_frequencies,
            references=reference_lines,
            candidates=candidate_lines,
            thresholds=thresholds,
        )
        failures.extend(
            _run_failures(
                components=components,
                references=reference_runs,
                candidates=candidate_runs,
                thresholds=thresholds,
            )
        )
        failures.extend(
            _preservation_failures(
                components,
                tuple(candidate_preservation.values()),
                thresholds,
            )
        )
        if not failures:
            return NativeObsDecision("accepted", components, evaluated, ())
        reasons.extend(failures)

    return NativeObsDecision("rejected", None, evaluated, tuple(reasons))


def _unique_rows(
    rows: Sequence[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    context: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    keyed = {tuple(row[field] for field in key_fields): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate {context} rows")
    return keyed


def _line_failures(
    *,
    components: int,
    reference_frequencies: Sequence[float],
    references: dict[tuple[Any, ...], dict[str, Any]],
    candidates: dict[tuple[Any, ...], dict[str, Any]],
    thresholds: NativeObsSelectionThresholds,
) -> list[str]:
    failures = []
    for frequency in reference_frequencies:
        keys = [key for key in references if float(key[1]) == frequency]
        power_reductions = np.asarray(
            [references[key]["power_db"] - candidates[key]["power_db"] for key in keys],
            dtype=float,
        )
        prominence_reductions = np.asarray(
            [
                references[key]["local_prominence_db"] - candidates[key]["local_prominence_db"]
                for key in keys
            ],
            dtype=float,
        )
        median_power_reduction = float(np.median(power_reductions))
        median_prominence_reduction = float(np.median(prominence_reductions))
        if median_power_reduction < thresholds.minimum_line_power_reduction_db:
            failures.append(
                f"components={components}: {frequency:.2f}-Hz median line power reduction "
                f"{median_power_reduction:.3f} dB is below "
                f"{thresholds.minimum_line_power_reduction_db:.3f} dB"
            )
        if median_prominence_reduction < thresholds.minimum_line_prominence_reduction_db:
            failures.append(
                f"components={components}: {frequency:.2f}-Hz median prominence reduction "
                f"{median_prominence_reduction:.3f} dB is below "
                f"{thresholds.minimum_line_prominence_reduction_db:.3f} dB"
            )
        maximum_increase = float(np.max(-prominence_reductions))
        if maximum_increase > thresholds.maximum_recording_prominence_increase_db:
            failures.append(
                f"components={components}: {frequency:.2f}-Hz maximum recording prominence "
                f"increase {maximum_increase:.3f} dB exceeds "
                f"{thresholds.maximum_recording_prominence_increase_db:.3f} dB"
            )
    return failures


def _run_failures(
    *,
    components: int,
    references: dict[tuple[Any, ...], dict[str, Any]],
    candidates: dict[tuple[Any, ...], dict[str, Any]],
    thresholds: NativeObsSelectionThresholds,
) -> list[str]:
    reductions = []
    for key, reference in references.items():
        reference_rms = float(reference["volume_locked_rms_v"])
        candidate_rms = float(candidates[key]["volume_locked_rms_v"])
        if not np.isfinite(reference_rms) or reference_rms <= 0:
            raise ValueError("Reference volume_locked_rms_v must be finite and positive")
        reductions.append((reference_rms - candidate_rms) / reference_rms)
    median_reduction = float(np.median(reductions))
    if median_reduction >= thresholds.minimum_volume_locked_rms_reduction_fraction:
        return []
    return [
        f"components={components}: median volume-locked RMS reduction "
        f"{median_reduction:.4f} is below "
        f"{thresholds.minimum_volume_locked_rms_reduction_fraction:.4f}"
    ]


def _preservation_failures(
    components: int,
    rows: Sequence[dict[str, Any]],
    thresholds: NativeObsSelectionThresholds,
) -> list[str]:
    failures = []
    minimum_amplitude = min(float(row["minimum_sinusoid_amplitude_ratio"]) for row in rows)
    if minimum_amplitude < thresholds.minimum_sinusoid_amplitude_ratio:
        failures.append(
            f"components={components}: minimum sinusoid amplitude ratio "
            f"{minimum_amplitude:.4f} is below "
            f"{thresholds.minimum_sinusoid_amplitude_ratio:.4f}"
        )
    maximum_phase = max(float(row["maximum_phase_error_deg"]) for row in rows)
    if maximum_phase > thresholds.maximum_phase_error_deg:
        failures.append(
            f"components={components}: maximum phase error {maximum_phase:.3f} deg exceeds "
            f"{thresholds.maximum_phase_error_deg:.3f} deg"
        )
    minimum_transient = min(float(row["transient_peak_ratio"]) for row in rows)
    if minimum_transient < thresholds.minimum_transient_peak_ratio:
        failures.append(
            f"components={components}: minimum transient peak ratio "
            f"{minimum_transient:.4f} is below "
            f"{thresholds.minimum_transient_peak_ratio:.4f}"
        )
    maximum_psd_change = max(abs(float(row["outside_line_psd_change_db"])) for row in rows)
    if maximum_psd_change > thresholds.maximum_outside_line_psd_change_db:
        failures.append(
            f"components={components}: maximum median outside-line PSD change "
            f"{maximum_psd_change:.3f} dB exceeds "
            f"{thresholds.maximum_outside_line_psd_change_db:.3f} dB"
        )
    maximum_channel_change = max(
        float(row["maximum_channel_outside_line_psd_change_db"]) for row in rows
    )
    if maximum_channel_change > thresholds.maximum_channel_outside_line_psd_change_db:
        failures.append(
            f"components={components}: maximum channel outside-line PSD change "
            f"{maximum_channel_change:.3f} dB exceeds "
            f"{thresholds.maximum_channel_outside_line_psd_change_db:.3f} dB"
        )
    return failures
