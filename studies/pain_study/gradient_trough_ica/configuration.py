"""Configuration loading for the gradient-trough ICA workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CONFIG_PATH = PACKAGE_ROOT / "gradient_trough_ica_config.yaml"


@dataclass(frozen=True)
class GradientTroughIcaConfig:
    """Validated settings shared by the Python and MATLAB stages."""

    participants: tuple[str, ...]
    expected_runs: tuple[int, ...]
    sampling_frequency_hz: float
    volume_marker: str
    volume_epoch_s: float
    ecg_channel: str
    highpass_hz: float
    highpass_transition_hz: float
    trough_reference_ms: dict[str, tuple[float, ...]]
    trough_refinement_radius_ms: float
    plateau_depth_fraction: float
    plateau_minimum_duration_ms: float
    plateau_maximum_duration_ms: float
    thermal_marker: str
    trial_tmin_s: float
    trial_tmax_s: float
    low_temperatures_c: tuple[float, ...]
    high_temperatures_c: tuple[float, ...]
    ica_seed: int
    tfr_window_s: float
    tfr_smoothing_hz: float
    tfr_frequencies_hz: tuple[float, ...]
    tfr_times_s: tuple[float, ...]
    tfr_baseline_s: tuple[float, float]
    tfr_padding_s: float
    corrected_data_roots: dict[str, Path]
    events_roots: dict[str, Path]
    output_root: Path

    @classmethod
    def load(
        cls,
        path: Path = CONFIG_PATH,
        *,
        validate_paths: bool = True,
    ) -> GradientTroughIcaConfig:
        """Read, validate, and resolve the workflow YAML configuration."""
        with path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("Gradient-trough ICA configuration must be a mapping.")

        selection = _mapping(values, "selection")
        trials = _mapping(values, "trials")
        ica = _mapping(values, "ica")
        tfr = _mapping(values, "tfr")
        paths = _mapping(values, "paths")
        participants = tuple(str(value) for value in values["participants"])

        config = cls(
            participants=participants,
            expected_runs=tuple(int(value) for value in values["expected_runs"]),
            sampling_frequency_hz=float(values["sampling_frequency_hz"]),
            volume_marker=str(selection["volume_marker"]),
            volume_epoch_s=float(selection["volume_epoch_s"]),
            ecg_channel=str(selection["ecg_channel"]),
            highpass_hz=float(selection["highpass_hz"]),
            highpass_transition_hz=float(selection["highpass_transition_hz"]),
            trough_reference_ms={
                participant: tuple(float(item) for item in timings)
                for participant, timings in _mapping(selection, "trough_reference_ms").items()
            },
            trough_refinement_radius_ms=float(selection["trough_refinement_radius_ms"]),
            plateau_depth_fraction=float(selection["plateau_depth_fraction"]),
            plateau_minimum_duration_ms=float(selection["plateau_minimum_duration_ms"]),
            plateau_maximum_duration_ms=float(selection["plateau_maximum_duration_ms"]),
            thermal_marker=str(trials["thermal_marker"]),
            trial_tmin_s=float(trials["tmin_s"]),
            trial_tmax_s=float(trials["tmax_s"]),
            low_temperatures_c=tuple(float(value) for value in trials["low_temperatures_c"]),
            high_temperatures_c=tuple(float(value) for value in trials["high_temperatures_c"]),
            ica_seed=int(ica["seed"]),
            tfr_window_s=float(tfr["window_s"]),
            tfr_smoothing_hz=float(tfr["smoothing_hz"]),
            tfr_frequencies_hz=_inclusive_grid(tfr, "frequency_hz"),
            tfr_times_s=_inclusive_grid(tfr, "time_s"),
            tfr_baseline_s=tuple(float(value) for value in tfr["baseline_s"]),
            tfr_padding_s=float(tfr["padding_s"]),
            corrected_data_roots=_participant_paths(
                paths,
                "corrected_data_roots",
                participants,
            ),
            events_roots=_participant_paths(paths, "events_roots", participants),
            output_root=_resolve_path(str(paths["output_root"])),
        )
        config.validate(validate_paths=validate_paths)
        return config

    def validate(self, *, validate_paths: bool) -> None:
        """Fail fast when assumptions required by the workflow are violated."""
        if not self.participants or set(self.participants) != set(self.trough_reference_ms):
            raise ValueError("Every participant must have trough reference timings.")
        if not self.expected_runs or min(self.expected_runs) < 1:
            raise ValueError("expected_runs must contain positive run numbers.")
        if self.sampling_frequency_hz != 1_000.0:
            raise ValueError("This workflow requires corrected 1 kHz data.")
        if not 0.0 < self.plateau_depth_fraction < 1.0:
            raise ValueError("plateau_depth_fraction must be in (0, 1).")
        if self.plateau_minimum_duration_ms > self.plateau_maximum_duration_ms:
            raise ValueError("Plateau minimum duration exceeds maximum duration.")
        if self.highpass_hz <= 0.0 or self.highpass_hz >= self.sampling_frequency_hz / 2.0:
            raise ValueError("highpass_hz must be between 0 and Nyquist.")
        if self.trial_tmin_s >= self.trial_tmax_s:
            raise ValueError("Trial start must precede trial end.")
        if len(self.tfr_baseline_s) != 2 or self.tfr_baseline_s[0] >= self.tfr_baseline_s[1]:
            raise ValueError("TFR baseline must contain increasing start and stop times.")
        if set(self.low_temperatures_c) & set(self.high_temperatures_c):
            raise ValueError("Low and high temperature groups must be disjoint.")
        if validate_paths:
            for source_path in (*self.corrected_data_roots.values(), *self.events_roots.values()):
                if not source_path.is_dir():
                    raise FileNotFoundError(
                        f"Configured source directory does not exist: {source_path}"
                    )


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    result = values[key]
    if not isinstance(result, dict):
        raise ValueError(f"Configuration value '{key}' must be a mapping.")
    return result


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _participant_paths(
    paths: dict[str, Any],
    key: str,
    participants: tuple[str, ...],
) -> dict[str, Path]:
    configured = _mapping(paths, key)
    if set(configured) != set(participants):
        raise ValueError(f"paths.{key} must define every configured participant exactly once.")
    return {
        participant: _resolve_path(str(configured[participant])) for participant in participants
    }


def _inclusive_grid(values: dict[str, Any], key: str) -> tuple[float, ...]:
    start, stop, step = (float(value) for value in values[key])
    if step <= 0.0 or start > stop:
        raise ValueError(f"tfr.{key} must contain an increasing [start, stop, step] grid.")
    count = round((stop - start) / step)
    if abs(start + count * step - stop) > 1e-9:
        raise ValueError(f"tfr.{key} stop must lie on its configured grid.")
    return tuple(start + index * step for index in range(count + 1))
