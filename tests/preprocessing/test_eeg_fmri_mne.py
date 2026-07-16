from __future__ import annotations

import mne
import numpy as np
import pytest

import eeg_pipeline.preprocessing.eeg_fmri.cardiac as cardiac
from eeg_pipeline.preprocessing.eeg_fmri.cardiac import (
    CardiacArtifactParameters,
    apply_cardiac_obs_in_place,
    detect_qrs,
)
from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import (
    MneQrsDetector,
    NeuXusQrsDetection,
    NeuXusQrsDetectionParameters,
    PanTompkinsQrsDetector,
)
from eeg_pipeline.preprocessing.eeg_fmri.mne_io import (
    extract_volume_samples,
    validate_acquisition,
)


def _raw_with_annotations(descriptions: list[str]) -> mne.io.RawArray:
    info = mne.create_info(["Cz", "ECG"], 1_000.0, ["eeg", "eeg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    onsets = 0.1 + np.arange(len(descriptions)) * 0.1
    raw.set_annotations(mne.Annotations(onsets, 0.0, descriptions))
    return raw


def _raw_with_cardiac_artifact() -> tuple[mne.io.RawArray, np.ndarray]:
    sampling_frequency = 200.0
    duration_seconds = 30.0
    n_samples = int(sampling_frequency * duration_seconds)
    time = np.arange(n_samples) / sampling_frequency
    qrs_times = np.arange(1.0, duration_seconds - 1.0, 1.0)

    ecg = np.zeros(n_samples)
    pulse_artifact = np.zeros(n_samples)
    sample_offsets = np.arange(-8, 9)
    qrs_waveform = np.exp(-0.5 * (sample_offsets / 1.5) ** 2)
    artifact_waveform = np.exp(-0.5 * ((sample_offsets - 3) / 3.0) ** 2)
    for qrs_time in qrs_times:
        center = int(round(qrs_time * sampling_frequency))
        indices = center + sample_offsets
        ecg[indices] += qrs_waveform
        pulse_artifact[indices] += artifact_waveform

    eeg_1 = 0.02 * np.sin(2 * np.pi * 10.0 * time) + 0.4 * pulse_artifact
    eeg_2 = 0.02 * np.cos(2 * np.pi * 8.0 * time) - 0.25 * pulse_artifact
    info = mne.create_info(["C3", "C4", "ECG"], sampling_frequency, ["eeg", "eeg", "ecg"])
    return mne.io.RawArray(np.vstack((eeg_1, eeg_2, ecg)), info, verbose=False), qrs_times


class _FixedDetector:
    def __init__(self, times: np.ndarray) -> None:
        self.times = times

    def detect(self, ecg: np.ndarray, *, sampling_frequency_hz: float) -> NeuXusQrsDetection:
        detection_samples = np.rint(self.times * 250.0).astype(int)
        n_samples = int(np.ceil(ecg.size * 250.0 / sampling_frequency_hz))
        return NeuXusQrsDetection(
            times=self.times,
            peak_samples=detection_samples,
            filtered_ecg=np.zeros(n_samples),
            probabilities=np.zeros(n_samples),
            probability_support=np.ones(n_samples, dtype=int),
            sampling_frequency_hz=250.0,
            model_sha256="test-model",
        )


def _cardiac_parameters() -> CardiacArtifactParameters:
    return CardiacArtifactParameters(
        obs_components=2,
        detection=NeuXusQrsDetectionParameters(),
        minimum_heart_rate_bpm=40.0,
        maximum_heart_rate_bpm=160.0,
        minimum_qrs_count=10,
        minimum_temporal_coverage=0.9,
        minimum_warning_rr_seconds=0.375,
        maximum_warning_rr_seconds=1.5,
    )


def test_extract_volume_samples_selects_only_exact_volume_annotations() -> None:
    raw = _raw_with_annotations(["Volume/V  1", "Stim_on/S  1", "Volume/V  1", "Vas_on/VAS_ON"])

    samples = extract_volume_samples(raw, annotation_description="Volume/V  1")

    np.testing.assert_array_equal(samples, np.array([100, 300]))


def test_extract_volume_samples_rejects_remaining_v1_collision() -> None:
    raw = _raw_with_annotations(["Volume/V  1", "Vas_on/V  1", "Volume/V  1"])

    with pytest.raises(ValueError, match="Vas_on/V  1"):
        extract_volume_samples(raw, annotation_description="Volume/V  1")


def test_validate_acquisition_assigns_the_ecg_channel_type() -> None:
    raw = _raw_with_annotations(["Volume/V  1", "Volume/V  1"])

    validate_acquisition(
        raw,
        expected_sampling_frequency=1_000.0,
        expected_channel_count=2,
        ecg_channel="ECG",
    )

    assert raw.get_channel_types() == ["eeg", "ecg"]


def test_validate_acquisition_rejects_wrong_sampling_frequency() -> None:
    raw = _raw_with_annotations(["Volume/V  1", "Volume/V  1"])

    with pytest.raises(ValueError, match="sampling frequency"):
        validate_acquisition(
            raw,
            expected_sampling_frequency=5_000.0,
            expected_channel_count=2,
            ecg_channel="ECG",
        )


def test_detect_qrs_finds_physiologic_heart_rate() -> None:
    raw, expected_times = _raw_with_cardiac_artifact()

    detection = detect_qrs(
        raw,
        ecg_channel="ECG",
        detector=_FixedDetector(expected_times),
        parameters=_cardiac_parameters(),
    )

    assert detection.quality.median_heart_rate_bpm == pytest.approx(60.0)
    assert detection.quality.correction_permitted
    assert detection.quality.warnings == ()
    assert detection.times.size == expected_times.size
    np.testing.assert_array_equal(detection.times, expected_times)


def test_detect_qrs_uses_validated_fallback_when_primary_is_unusable() -> None:
    raw, expected_times = _raw_with_cardiac_artifact()
    missed_alternate_peaks = expected_times[::2]

    detection = detect_qrs(
        raw,
        ecg_channel="ECG",
        detector=_FixedDetector(missed_alternate_peaks),
        fallback_detectors=(
            _FixedDetector(missed_alternate_peaks),
            _FixedDetector(expected_times),
        ),
        parameters=_cardiac_parameters(),
    )

    np.testing.assert_array_equal(detection.times, expected_times)
    assert detection.quality.median_heart_rate_bpm == pytest.approx(60.0)
    assert any("fallback" in warning.lower() for warning in detection.quality.warnings)


def test_mne_qrs_fallback_enforces_refractory_period() -> None:
    sampling_frequency = 1_000.0
    time = np.arange(int(30 * sampling_frequency)) / sampling_frequency
    ecg = 0.01 * np.sin(2 * np.pi * 1.0 * time)
    for peak_time in np.arange(1.0, 29.0, 1.0):
        center = int(peak_time * sampling_frequency)
        offsets = np.arange(-12, 13)
        ecg[center + offsets] += np.exp(-0.5 * (offsets / 3.0) ** 2)

    detection = MneQrsDetector(NeuXusQrsDetectionParameters()).detect(
        ecg,
        sampling_frequency_hz=sampling_frequency,
    )

    assert detection.model_sha256.startswith("mne.preprocessing.ecg.qrs_detector")
    assert detection.times.size == 28
    assert np.min(np.diff(detection.times)) >= 0.4


def test_pan_tompkins_fallback_detects_regular_qrs_energy() -> None:
    sampling_frequency = 1_000.0
    time = np.arange(int(30 * sampling_frequency)) / sampling_frequency
    ecg = 0.01 * np.sin(2 * np.pi * 1.0 * time)
    for peak_time in np.arange(1.0, 29.0, 1.0):
        center = int(peak_time * sampling_frequency)
        offsets = np.arange(-12, 13)
        ecg[center + offsets] += np.exp(-0.5 * (offsets / 3.0) ** 2)

    detection = PanTompkinsQrsDetector(NeuXusQrsDetectionParameters()).detect(
        ecg,
        sampling_frequency_hz=sampling_frequency,
    )

    assert detection.model_sha256.startswith("pan-tompkins")
    assert detection.times.size == 28
    assert np.min(np.diff(detection.times)) >= 0.4


def test_qrs_quality_warns_without_blocking_isolated_bad_intervals() -> None:
    times = np.array([0.2, 1.2, 2.2, 2.21, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2])

    quality = cardiac.classify_qrs_quality(
        times,
        duration_seconds=10.0,
        parameters=_cardiac_parameters(),
    )

    assert quality.correction_permitted
    assert quality.abnormal_rr_count == 2
    assert quality.abnormal_rr_fraction == pytest.approx(2 / 9)
    assert quality.maximum_rr_seconds == pytest.approx(1.99)
    assert quality.temporal_coverage == pytest.approx(0.9)
    assert len(quality.warnings) == 1


@pytest.mark.parametrize(
    "times, duration, message",
    [
        (np.array([1.0, 2.0]), 10.0, "at least"),
        (np.array([0.0, 1.0, np.nan] + list(np.arange(3.0, 10.0))), 10.0, "finite"),
        (np.array([0.0, 1.0, 3.0, 2.0] + list(np.arange(4.0, 10.0))), 10.0, "increasing"),
        (np.arange(-1.0, 10.0), 10.0, "data interval"),
        (np.arange(0.0, 2.1, 0.2), 2.1, "heart rate"),
        (np.arange(3.0, 13.0), 15.0, "coverage"),
    ],
)
def test_qrs_quality_rejects_unusable_detections(
    times: np.ndarray,
    duration: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cardiac.classify_qrs_quality(
            times,
            duration_seconds=duration,
            parameters=_cardiac_parameters(),
        )


def test_cardiac_obs_changes_only_eeg_channels() -> None:
    raw, expected_times = _raw_with_cardiac_artifact()
    original = raw.get_data().copy()
    detection = detect_qrs(
        raw,
        ecg_channel="ECG",
        detector=_FixedDetector(expected_times),
        parameters=_cardiac_parameters(),
    )

    apply_cardiac_obs_in_place(
        raw,
        qrs=detection,
        parameters=_cardiac_parameters(),
    )

    assert detection.times.size == 28
    assert not np.array_equal(raw.get_data(picks="eeg"), original[:2])
    np.testing.assert_array_equal(raw.get_data(picks=["ECG"])[0], original[2])
