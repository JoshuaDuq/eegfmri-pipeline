"""Strict configuration for native EEG-fMRI artifact correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from eeg_pipeline.preprocessing.eeg_fmri.cardiac import CardiacArtifactParameters
from eeg_pipeline.preprocessing.eeg_fmri.gradient import GradientArtifactParameters
from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import (
    NeuXusQrsDetectionParameters,
)


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected_keys: set[str],
    context: str,
) -> None:
    missing = sorted(expected_keys - set(mapping))
    unexpected = sorted(set(mapping) - expected_keys)
    if missing or unexpected:
        raise ValueError(f"Invalid {context} keys; missing={missing}, unexpected={unexpected}")


@dataclass(frozen=True)
class NativeEegFmriParameters:
    """Complete fixed parameters for the native correction sequence."""

    acquisition_sampling_frequency_hz: float
    repetition_time_seconds: float
    expected_channel_count: int
    ecg_channel: str
    volume_annotation: str
    maximum_marker_deviation_samples: int
    gradient: GradientArtifactParameters
    low_pass_frequency_hz: float
    output_sampling_frequency_hz: float
    cardiac: CardiacArtifactParameters
    qc_channels: tuple[str, ...]
    qc_welch_duration_seconds: float
    qc_minimum_duration_seconds: float
    qc_bootstrap_iterations: int
    qc_bootstrap_confidence_level: float
    qc_bootstrap_seed: int

    def __post_init__(self) -> None:
        if self.acquisition_sampling_frequency_hz <= 0:
            raise ValueError("acquisition_sampling_frequency_hz must be positive")
        if self.expected_channel_count < 2:
            raise ValueError("expected_channel_count must be at least 2")
        if not self.ecg_channel:
            raise ValueError("ecg_channel must be non-empty")
        if not self.volume_annotation:
            raise ValueError("volume_annotation must be non-empty")
        if self.maximum_marker_deviation_samples < 0:
            raise ValueError("maximum_marker_deviation_samples must be non-negative")
        if not np.isfinite(self.low_pass_frequency_hz):
            raise ValueError("low_pass_frequency_hz must be finite")
        if self.low_pass_frequency_hz <= 0:
            raise ValueError("low_pass_frequency_hz must be positive")
        if self.output_sampling_frequency_hz <= 0:
            raise ValueError("output_sampling_frequency_hz must be positive")
        if self.output_sampling_frequency_hz >= self.acquisition_sampling_frequency_hz:
            raise ValueError("output_sampling_frequency_hz must be below the acquisition rate")
        if self.low_pass_frequency_hz >= self.output_sampling_frequency_hz / 2:
            raise ValueError("low_pass_frequency_hz must be below the output Nyquist frequency")
        if not self.qc_channels:
            raise ValueError("qc_channels must be non-empty")
        if len(set(self.qc_channels)) != len(self.qc_channels):
            raise ValueError("qc_channels must be unique")
        if any(not channel for channel in self.qc_channels):
            raise ValueError("qc_channels cannot contain empty names")
        if self.qc_welch_duration_seconds <= 0:
            raise ValueError("qc_welch_duration_seconds must be positive")
        if self.qc_minimum_duration_seconds <= 0:
            raise ValueError("qc_minimum_duration_seconds must be positive")
        if self.qc_bootstrap_iterations < 1:
            raise ValueError("qc_bootstrap_iterations must be positive")
        if not 0.0 < self.qc_bootstrap_confidence_level < 1.0:
            raise ValueError("qc_bootstrap_confidence_level must be between 0 and 1")
        if self.qc_bootstrap_seed < 0:
            raise ValueError("qc_bootstrap_seed must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeEegFmriParameters:
        """Construct parameters from one exact configuration mapping."""
        _require_exact_keys(
            value,
            {"version", "acquisition", "gradient", "resampling", "cardiac", "qc"},
            "top-level configuration",
        )
        if value["version"] != 2:
            raise ValueError(f"Unsupported native EEG-fMRI config version: {value['version']!r}")

        acquisition = _require_mapping(value["acquisition"], "acquisition")
        _require_exact_keys(
            acquisition,
            {
                "sampling_frequency_hz",
                "repetition_time_seconds",
                "expected_channel_count",
                "ecg_channel",
                "volume_annotation",
                "maximum_marker_deviation_samples",
            },
            "acquisition",
        )
        gradient = _require_mapping(value["gradient"], "gradient")
        _require_exact_keys(
            gradient,
            {
                "moving_average_volumes",
                "alignment_upsampling",
                "maximum_alignment_shift_samples",
            },
            "gradient",
        )
        resampling = _require_mapping(value["resampling"], "resampling")
        _require_exact_keys(
            resampling,
            {"low_pass_frequency_hz", "output_sampling_frequency_hz"},
            "resampling",
        )
        cardiac = _require_mapping(value["cardiac"], "cardiac")
        _require_exact_keys(
            cardiac,
            {
                "obs_components",
                "detection",
                "minimum_heart_rate_bpm",
                "maximum_heart_rate_bpm",
                "minimum_qrs_count",
                "minimum_temporal_coverage",
                "minimum_warning_rr_seconds",
                "maximum_warning_rr_seconds",
            },
            "cardiac",
        )
        detection = _require_mapping(cardiac["detection"], "cardiac.detection")
        _require_exact_keys(
            detection,
            {
                "sampling_frequency_hz",
                "low_frequency_hz",
                "high_frequency_hz",
                "window_stride_samples",
                "probability_threshold",
                "minimum_support_samples",
                "refractory_period_seconds",
                "edge_margin_seconds",
            },
            "cardiac.detection",
        )
        qc = _require_mapping(value["qc"], "qc")
        _require_exact_keys(
            qc,
            {
                "channels",
                "welch_duration_seconds",
                "minimum_duration_seconds",
                "bootstrap",
            },
            "qc",
        )
        channels = qc["channels"]
        if not isinstance(channels, list):
            raise TypeError("qc.channels must be a list")
        bootstrap = _require_mapping(qc["bootstrap"], "qc.bootstrap")
        _require_exact_keys(
            bootstrap,
            {"iterations", "confidence_level", "seed"},
            "qc.bootstrap",
        )

        repetition_time = float(acquisition["repetition_time_seconds"])
        return cls(
            acquisition_sampling_frequency_hz=float(acquisition["sampling_frequency_hz"]),
            repetition_time_seconds=repetition_time,
            expected_channel_count=int(acquisition["expected_channel_count"]),
            ecg_channel=str(acquisition["ecg_channel"]),
            volume_annotation=str(acquisition["volume_annotation"]),
            maximum_marker_deviation_samples=int(acquisition["maximum_marker_deviation_samples"]),
            gradient=GradientArtifactParameters(
                repetition_time_seconds=repetition_time,
                moving_average_volumes=int(gradient["moving_average_volumes"]),
                alignment_upsampling=int(gradient["alignment_upsampling"]),
                maximum_alignment_shift_samples=float(gradient["maximum_alignment_shift_samples"]),
            ),
            low_pass_frequency_hz=float(resampling["low_pass_frequency_hz"]),
            output_sampling_frequency_hz=float(resampling["output_sampling_frequency_hz"]),
            cardiac=CardiacArtifactParameters(
                obs_components=int(cardiac["obs_components"]),
                detection=NeuXusQrsDetectionParameters(
                    sampling_frequency_hz=float(detection["sampling_frequency_hz"]),
                    low_frequency_hz=float(detection["low_frequency_hz"]),
                    high_frequency_hz=float(detection["high_frequency_hz"]),
                    window_stride_samples=int(detection["window_stride_samples"]),
                    probability_threshold=float(detection["probability_threshold"]),
                    minimum_support_samples=int(detection["minimum_support_samples"]),
                    refractory_period_seconds=float(detection["refractory_period_seconds"]),
                    edge_margin_seconds=float(detection["edge_margin_seconds"]),
                ),
                minimum_heart_rate_bpm=float(cardiac["minimum_heart_rate_bpm"]),
                maximum_heart_rate_bpm=float(cardiac["maximum_heart_rate_bpm"]),
                minimum_qrs_count=int(cardiac["minimum_qrs_count"]),
                minimum_temporal_coverage=float(cardiac["minimum_temporal_coverage"]),
                minimum_warning_rr_seconds=float(cardiac["minimum_warning_rr_seconds"]),
                maximum_warning_rr_seconds=float(cardiac["maximum_warning_rr_seconds"]),
            ),
            qc_channels=tuple(str(channel) for channel in channels),
            qc_welch_duration_seconds=float(qc["welch_duration_seconds"]),
            qc_minimum_duration_seconds=float(qc["minimum_duration_seconds"]),
            qc_bootstrap_iterations=int(bootstrap["iterations"]),
            qc_bootstrap_confidence_level=float(bootstrap["confidence_level"]),
            qc_bootstrap_seed=int(bootstrap["seed"]),
        )


def load_native_eeg_fmri_parameters(path: str | Path) -> NativeEegFmriParameters:
    """Load one exact versioned YAML configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Native EEG-fMRI config does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = _require_mapping(loaded, "native EEG-fMRI configuration")
    return NativeEegFmriParameters.from_mapping(mapping)
