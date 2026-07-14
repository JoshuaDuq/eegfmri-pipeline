"""Benchmark residual scanner-gradient OBS after BrainVision preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eeg_pipeline.analysis.qc.residual_gradient import (
    InjectionSettings,
    SelectionThresholds,
)
from eeg_pipeline.preprocessing.residual_gradient import ResidualObsSettings


@dataclass(frozen=True)
class WelchSettings:
    nperseg: int
    overlap_fraction: float

    def __post_init__(self) -> None:
        if self.nperseg <= 0:
            raise ValueError("welch.nperseg must be positive.")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("welch.overlap_fraction must be in [0, 1).")


@dataclass(frozen=True)
class BenchmarkConfig:
    pilot_subject: str
    expected_runs: int
    component_counts: tuple[int, ...]
    obs: ResidualObsSettings
    harmonic_windows_hz: tuple[tuple[float, float], ...]
    welch: WelchSettings
    injection: InjectionSettings
    acceptance: SelectionThresholds

    def __post_init__(self) -> None:
        if self.pilot_subject != "0006":
            raise ValueError("pilot_subject must be '0006' for the prespecified excluded pilot.")
        if self.expected_runs != 6:
            raise ValueError("expected_runs must be exactly 6 for the prespecified pilot.")
        if self.component_counts != (0, 1, 2, 3, 4):
            raise ValueError("component_counts must be exactly [0, 1, 2, 3, 4].")
        if not self.harmonic_windows_hz:
            raise ValueError("harmonic_windows_hz must be non-empty.")
        previous_high = 0.0
        for low_hz, high_hz in self.harmonic_windows_hz:
            if not 0 < low_hz < high_hz:
                raise ValueError("Each harmonic window must have 0 < low_hz < high_hz.")
            if low_hz <= previous_high:
                raise ValueError("harmonic_windows_hz must be sorted and non-overlapping.")
            previous_high = high_hz


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load the benchmark's exact, fail-fast YAML schema."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark configuration does not exist: {config_path}")
    payload = _require_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "benchmark configuration root",
    )
    _require_exact_keys(
        payload,
        {
            "pilot_subject",
            "expected_runs",
            "component_counts",
            "obs",
            "harmonic_windows_hz",
            "welch",
            "injection",
            "acceptance",
        },
        "benchmark",
    )

    obs_values = _require_mapping(payload["obs"], "obs")
    _require_exact_keys(
        obs_values,
        {
            "expected_sfreq_hz",
            "volume_marker",
            "tr_s",
            "min_complete_epochs",
            "n_folds",
        },
        "obs",
    )
    welch_values = _require_mapping(payload["welch"], "welch")
    _require_exact_keys(welch_values, {"nperseg", "overlap_fraction"}, "welch")
    injection_values = _require_mapping(payload["injection"], "injection")
    _require_exact_keys(
        injection_values,
        {
            "frequencies_hz",
            "sinusoid_amplitude_uv",
            "transient_amplitude_uv",
            "transient_spacing_s",
            "transient_width_s",
        },
        "injection",
    )
    acceptance_values = _require_mapping(payload["acceptance"], "acceptance")
    _require_exact_keys(
        acceptance_values,
        {
            "minimum_sinusoid_amplitude_ratio",
            "maximum_phase_error_deg",
            "minimum_transient_peak_ratio",
            "maximum_outside_harmonic_psd_change_db",
            "maximum_run_prominence_increase_db",
        },
        "acceptance",
    )

    return BenchmarkConfig(
        pilot_subject=_require_string(payload["pilot_subject"], "pilot_subject"),
        expected_runs=_require_integer(payload["expected_runs"], "expected_runs"),
        component_counts=_integer_tuple(payload["component_counts"], "component_counts"),
        obs=ResidualObsSettings(
            expected_sfreq_hz=_require_number(
                obs_values["expected_sfreq_hz"], "obs.expected_sfreq_hz"
            ),
            volume_marker=_require_string(obs_values["volume_marker"], "obs.volume_marker"),
            tr_s=_require_number(obs_values["tr_s"], "obs.tr_s"),
            min_complete_epochs=_require_integer(
                obs_values["min_complete_epochs"], "obs.min_complete_epochs"
            ),
            n_folds=_require_integer(obs_values["n_folds"], "obs.n_folds"),
        ),
        harmonic_windows_hz=_frequency_windows(payload["harmonic_windows_hz"]),
        welch=WelchSettings(
            nperseg=_require_integer(welch_values["nperseg"], "welch.nperseg"),
            overlap_fraction=_require_number(
                welch_values["overlap_fraction"], "welch.overlap_fraction"
            ),
        ),
        injection=InjectionSettings(
            frequencies_hz=_number_tuple(
                injection_values["frequencies_hz"], "injection.frequencies_hz"
            ),
            sinusoid_amplitude_uv=_require_number(
                injection_values["sinusoid_amplitude_uv"],
                "injection.sinusoid_amplitude_uv",
            ),
            transient_amplitude_uv=_require_number(
                injection_values["transient_amplitude_uv"],
                "injection.transient_amplitude_uv",
            ),
            transient_spacing_s=_require_number(
                injection_values["transient_spacing_s"],
                "injection.transient_spacing_s",
            ),
            transient_width_s=_require_number(
                injection_values["transient_width_s"],
                "injection.transient_width_s",
            ),
        ),
        acceptance=SelectionThresholds(
            minimum_sinusoid_amplitude_ratio=_require_number(
                acceptance_values["minimum_sinusoid_amplitude_ratio"],
                "acceptance.minimum_sinusoid_amplitude_ratio",
            ),
            maximum_phase_error_deg=_require_number(
                acceptance_values["maximum_phase_error_deg"],
                "acceptance.maximum_phase_error_deg",
            ),
            minimum_transient_peak_ratio=_require_number(
                acceptance_values["minimum_transient_peak_ratio"],
                "acceptance.minimum_transient_peak_ratio",
            ),
            maximum_outside_harmonic_psd_change_db=_require_number(
                acceptance_values["maximum_outside_harmonic_psd_change_db"],
                "acceptance.maximum_outside_harmonic_psd_change_db",
            ),
            maximum_run_prominence_increase_db=_require_number(
                acceptance_values["maximum_run_prominence_increase_db"],
                "acceptance.maximum_run_prominence_increase_db",
            ),
        ),
    )


def discover_pilot_files(source_root: str | Path, subject: str) -> list[Path]:
    """Discover only Analyzer-corrected headers for the prespecified pilot."""
    root = Path(source_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {root}")
    eeg_dir = root / f"sub-{subject}" / "eeg"
    paths = sorted(
        path
        for path in eeg_dir.glob("*_scannerpulse_corrected.vhdr")
        if path.is_file() and not path.name.startswith("._")
    )
    if not paths:
        raise FileNotFoundError(
            f"No _scannerpulse_corrected.vhdr files found for sub-{subject}: {eeg_dir}"
        )
    return paths


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    extra = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if extra:
        raise ValueError(f"Unknown {label} configuration keys: {extra}")
    if missing:
        raise ValueError(f"Missing {label} configuration keys: {missing}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys.")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def _require_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer.")
    return value


def _require_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError(f"{label} must be a finite number.")
    return float(value)


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list.")
    return value


def _integer_tuple(value: Any, label: str) -> tuple[int, ...]:
    values = _require_list(value, label)
    return tuple(_require_integer(item, f"{label} item") for item in values)


def _number_tuple(value: Any, label: str) -> tuple[float, ...]:
    values = _require_list(value, label)
    return tuple(_require_number(item, f"{label} item") for item in values)


def _frequency_windows(value: Any) -> tuple[tuple[float, float], ...]:
    windows = _require_list(value, "harmonic_windows_hz")
    parsed: list[tuple[float, float]] = []
    for index, window in enumerate(windows):
        bounds = _number_tuple(window, f"harmonic_windows_hz[{index}]")
        if len(bounds) != 2:
            raise ValueError(f"harmonic_windows_hz[{index}] must contain exactly two bounds.")
        parsed.append(bounds)
    return tuple(parsed)
