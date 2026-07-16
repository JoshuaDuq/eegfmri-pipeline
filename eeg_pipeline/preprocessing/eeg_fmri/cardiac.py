"""ECG-driven optimal-basis pulse-artifact correction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mne
import numpy as np
from collections.abc import Sequence

from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import (
    NeuXusQrsDetection,
    NeuXusQrsDetectionParameters,
    QrsDetector,
)


@dataclass(frozen=True)
class CardiacArtifactParameters:
    """Parameters for automatic NeuXus detection and MNE PCA-OBS."""

    obs_components: int = 4
    detection: NeuXusQrsDetectionParameters = field(default_factory=NeuXusQrsDetectionParameters)
    minimum_heart_rate_bpm: float = 40.0
    maximum_heart_rate_bpm: float = 160.0
    minimum_qrs_count: int = 10
    minimum_temporal_coverage: float = 0.9
    minimum_warning_rr_seconds: float = 0.375
    maximum_warning_rr_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.obs_components < 1:
            raise ValueError("obs_components must be positive")
        if self.minimum_heart_rate_bpm <= 0:
            raise ValueError("minimum_heart_rate_bpm must be positive")
        if self.maximum_heart_rate_bpm <= self.minimum_heart_rate_bpm:
            raise ValueError("maximum_heart_rate_bpm must exceed minimum_heart_rate_bpm")
        if self.minimum_qrs_count < 3:
            raise ValueError("minimum_qrs_count must be at least 3")
        if not 0.0 < self.minimum_temporal_coverage <= 1.0:
            raise ValueError("minimum_temporal_coverage must be between zero and one")
        if self.minimum_warning_rr_seconds <= 0:
            raise ValueError("minimum_warning_rr_seconds must be positive")
        if self.maximum_warning_rr_seconds <= self.minimum_warning_rr_seconds:
            raise ValueError("maximum_warning_rr_seconds must exceed minimum_warning_rr_seconds")


@dataclass(frozen=True)
class QrsQuality:
    """Physiological and temporal quality of one automatic QRS detection."""

    qrs_count: int
    median_heart_rate_bpm: float
    minimum_rr_seconds: float
    median_rr_seconds: float
    maximum_rr_seconds: float
    abnormal_rr_count: int
    abnormal_rr_fraction: float
    temporal_coverage: float
    warnings: tuple[str, ...]
    correction_permitted: bool


@dataclass(frozen=True)
class QrsDetection:
    """Accepted QRS times, quality metrics, and detector diagnostics."""

    times: np.ndarray
    quality: QrsQuality
    diagnostics: NeuXusQrsDetection


def _validate_ecg_channel(raw: mne.io.BaseRaw, ecg_channel: str) -> None:
    if raw.ch_names.count(ecg_channel) != 1:
        raise ValueError(f"Expected exactly one channel named {ecg_channel!r}")
    channel_index = raw.ch_names.index(ecg_channel)
    if raw.get_channel_types()[channel_index] != "ecg":
        raise ValueError(f"Channel {ecg_channel!r} must have MNE channel type 'ecg'")


def classify_qrs_quality(
    times: np.ndarray,
    *,
    duration_seconds: float,
    parameters: CardiacArtifactParameters,
) -> QrsQuality:
    """Validate correction usability and report non-blocking RR abnormalities."""
    values = np.asarray(times, dtype=float)
    if values.ndim != 1:
        raise ValueError("QRS times must be one-dimensional")
    if values.size < parameters.minimum_qrs_count:
        raise ValueError(
            f"QRS detection requires at least {parameters.minimum_qrs_count} peaks, "
            f"found {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("QRS times must contain only finite values")
    rr_intervals = np.diff(values)
    if np.any(rr_intervals <= 0):
        raise ValueError("QRS times must be strictly increasing")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if values[0] < 0 or values[-1] > duration_seconds:
        raise ValueError("QRS times must lie within the data interval")

    median_rr = float(np.median(rr_intervals))
    median_heart_rate = 60.0 / median_rr
    if not (
        parameters.minimum_heart_rate_bpm <= median_heart_rate <= parameters.maximum_heart_rate_bpm
    ):
        raise ValueError(
            f"Detected median heart rate {median_heart_rate:.2f} bpm is outside "
            f"{parameters.minimum_heart_rate_bpm:.2f}-"
            f"{parameters.maximum_heart_rate_bpm:.2f} bpm"
        )

    temporal_coverage = float((values[-1] - values[0]) / duration_seconds)
    if temporal_coverage + np.finfo(float).eps < parameters.minimum_temporal_coverage:
        raise ValueError(
            f"QRS temporal coverage {temporal_coverage:.3f} is below "
            f"{parameters.minimum_temporal_coverage:.3f}"
        )
    abnormal = (rr_intervals < parameters.minimum_warning_rr_seconds) | (
        rr_intervals > parameters.maximum_warning_rr_seconds
    )
    abnormal_count = int(np.count_nonzero(abnormal))
    warnings: tuple[str, ...] = ()
    if abnormal_count:
        warnings = (
            f"{abnormal_count} of {rr_intervals.size} RR intervals fall outside "
            f"{parameters.minimum_warning_rr_seconds:.3f}-"
            f"{parameters.maximum_warning_rr_seconds:.3f} seconds",
        )
    return QrsQuality(
        qrs_count=int(values.size),
        median_heart_rate_bpm=float(median_heart_rate),
        minimum_rr_seconds=float(np.min(rr_intervals)),
        median_rr_seconds=median_rr,
        maximum_rr_seconds=float(np.max(rr_intervals)),
        abnormal_rr_count=abnormal_count,
        abnormal_rr_fraction=float(abnormal_count / rr_intervals.size),
        temporal_coverage=temporal_coverage,
        warnings=warnings,
        correction_permitted=True,
    )


def detect_qrs(
    raw: mne.io.BaseRaw,
    *,
    ecg_channel: str,
    detector: QrsDetector,
    fallback_detectors: Sequence[QrsDetector] = (),
    parameters: CardiacArtifactParameters,
) -> QrsDetection:
    """Detect R-peaks with NeuXus and classify their correction quality."""
    _validate_ecg_channel(raw, ecg_channel)
    ecg = raw.get_data(picks=[ecg_channel])[0]
    duration_seconds = float(raw.times[-1])
    rejection_messages = []
    for detector_index, candidate_detector in enumerate((detector, *fallback_detectors)):
        diagnostics = candidate_detector.detect(
            ecg,
            sampling_frequency_hz=float(raw.info["sfreq"]),
        )
        try:
            quality = classify_qrs_quality(
                diagnostics.times,
                duration_seconds=duration_seconds,
                parameters=parameters,
            )
        except ValueError as error:
            rejection_messages.append(str(error))
            continue
        if detector_index:
            quality = replace(
                quality,
                warnings=(
                    "Primary QRS detection rejected; validated fallback used: "
                    + " | ".join(rejection_messages),
                    *quality.warnings,
                ),
            )
        times = np.array(diagnostics.times, dtype=float, copy=True)
        times.setflags(write=False)
        return QrsDetection(times=times, quality=quality, diagnostics=diagnostics)

    raise ValueError("All QRS detectors failed validation: " + " | ".join(rejection_messages))


def apply_cardiac_obs_in_place(
    raw: mne.io.BaseRaw,
    *,
    qrs: QrsDetection,
    parameters: CardiacArtifactParameters,
) -> None:
    """Apply MNE PCA-OBS to EEG channels using accepted automatic QRS times."""
    if not raw.preload:
        raise ValueError("PCA-OBS requires a preloaded Raw object")
    if not qrs.quality.correction_permitted:
        raise ValueError("PCA-OBS requires QRS detection permitted for correction")
    eeg_channels = [
        channel_name
        for channel_name, channel_type in zip(
            raw.ch_names,
            raw.get_channel_types(),
            strict=True,
        )
        if channel_type == "eeg"
    ]
    if not eeg_channels:
        raise ValueError("PCA-OBS requires at least one EEG channel")
    mne.preprocessing.apply_pca_obs(
        raw,
        picks=eeg_channels,
        qrs_times=qrs.times,
        n_components=parameters.obs_components,
        n_jobs=None,
        copy=False,
        verbose=False,
    )
