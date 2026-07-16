from __future__ import annotations

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.config import NativeEegFmriParameters
from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import NeuXusQrsDetection
from eeg_pipeline.preprocessing.eeg_fmri.pipeline import preprocess_raw_in_place
from eeg_pipeline.preprocessing.eeg_fmri.qc import summarize_cardiac_locked_eeg
from eeg_pipeline.preprocessing.eeg_fmri.sequence import MultibandSliceSchedule


class _FixedQrsDetector:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def detect(
        self,
        ecg: np.ndarray,
        *,
        sampling_frequency_hz: float,
    ) -> NeuXusQrsDetection:
        self.calls.append(sampling_frequency_hz)
        times = np.array([0.1, 0.6, 1.1, 1.6, 1.9, 2.5, 3.0])
        peak_samples = np.rint(times * 250.0).astype(int)
        detection_samples = int(np.ceil(ecg.size * 250.0 / sampling_frequency_hz))
        return NeuXusQrsDetection(
            times=times,
            peak_samples=peak_samples,
            filtered_ecg=np.zeros(detection_samples),
            probabilities=np.zeros(detection_samples),
            probability_support=np.ones(detection_samples, dtype=int),
            sampling_frequency_hz=250.0,
            model_sha256="test-model-sha256",
        )


def _parameters() -> NativeEegFmriParameters:
    return NativeEegFmriParameters.from_mapping(
        {
            "version": 3,
            "acquisition": {
                "sampling_frequency_hz": 1_000.0,
                "repetition_time_seconds": 0.1,
                "expected_channel_count": 3,
                "ecg_channel": "ECG",
                "volume_annotation": "Volume/V  1",
                "maximum_marker_deviation_samples": 1,
            },
            "gradient": {
                "moving_average_volumes": 11,
                "alignment_upsampling": 4,
                "maximum_alignment_shift_samples": 1.0,
                "residual_obs_components": 0,
                "residual_obs_folds": 5,
                "residual_obs_seed": 42,
            },
            "resampling": {
                "low_pass_frequency_hz": 80.0,
                "output_sampling_frequency_hz": 500.0,
            },
            "cardiac": {
                "obs_components": 2,
                "detection": {
                    "sampling_frequency_hz": 250.0,
                    "low_frequency_hz": 0.5,
                    "high_frequency_hz": 30.0,
                    "window_stride_samples": 50,
                    "probability_threshold": 0.05,
                    "minimum_support_samples": 5,
                    "refractory_period_seconds": 0.4,
                    "edge_margin_seconds": 0.1,
                },
                "minimum_heart_rate_bpm": 40.0,
                "maximum_heart_rate_bpm": 160.0,
                "minimum_qrs_count": 3,
                "minimum_temporal_coverage": 0.9,
                "minimum_warning_rr_seconds": 0.375,
                "maximum_warning_rr_seconds": 1.5,
            },
            "qc": {
                "bootstrap": {
                    "iterations": 100,
                    "confidence_level": 0.95,
                    "seed": 42,
                },
                "channels": ["C3", "C4"],
                "welch_duration_seconds": 2.048,
                "minimum_duration_seconds": 0.256,
            },
        }
    )


def _schedule() -> MultibandSliceSchedule:
    return MultibandSliceSchedule(
        repetition_time_seconds=0.1,
        slice_times_seconds=np.array([0.0, 0.025, 0.05, 0.075]),
        multiband_factor=1,
    )


def _raw(
    *,
    incomplete_terminal_volume: bool,
    internal_scanner_restart: bool = False,
) -> mne.io.RawArray:
    sampling_frequency = 1_000.0
    volume_samples = 25 + np.arange(31) * 100
    if internal_scanner_restart:
        volume_samples[16:] += 50
    terminal_samples = 50 if incomplete_terminal_volume else 125
    n_samples = int(volume_samples[-1] + terminal_samples)
    time = np.arange(n_samples) / sampling_frequency
    data = np.vstack(
        (
            np.sin(2 * np.pi * 10.0 * time),
            np.cos(2 * np.pi * 8.0 * time),
            np.sin(2 * np.pi * 1.0 * time),
        )
    )
    template = 20.0 * np.sin(2 * np.pi * np.arange(100) / 9)
    complete_volume_count = 30 if incomplete_terminal_volume else 31
    for start in volume_samples[:complete_volume_count]:
        data[:, start : start + 100] += np.array([1.0, -0.6, 0.4])[:, None] * template

    info = mne.create_info(["C3", "C4", "ECG"], sampling_frequency, ["eeg"] * 3)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            volume_samples / sampling_frequency,
            0.0,
            ["Volume/V  1"] * volume_samples.size,
        )
    )
    return raw


def test_preprocess_raw_runs_detection_and_cardiac_qc_around_obs(monkeypatch) -> None:
    raw = _raw(incomplete_terminal_volume=False)
    detector = _FixedQrsDetector()
    call_order: list[str] = []
    ecg_before_obs: list[np.ndarray] = []

    def record_summary(raw, *, qrs_times):
        call_order.append("qc")
        return summarize_cardiac_locked_eeg(raw, qrs_times=qrs_times)

    def fake_obs(raw, *, qrs, parameters):
        call_order.append("obs")
        assert parameters.obs_components == 2
        ecg_before_obs.append(raw.get_data(picks=["ECG"])[0].copy())
        raw._data[:2] *= 0.25

    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.eeg_fmri.pipeline.apply_cardiac_obs_in_place",
        fake_obs,
    )
    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.eeg_fmri.pipeline.summarize_cardiac_locked_eeg",
        record_summary,
    )

    result = preprocess_raw_in_place(
        raw,
        parameters=_parameters(),
        qrs_detector=detector,
        slice_schedule=_schedule(),
    )

    assert result.raw is raw
    assert raw.info["sfreq"] == 500.0
    assert raw.get_channel_types() == ["eeg", "eeg", "ecg"]
    assert detector.calls == [500.0]
    assert call_order == ["qc", "obs", "qc"]
    np.testing.assert_array_equal(raw.get_data(picks=["ECG"])[0], ecg_before_obs[0])
    assert result.complete_volume_count == 31
    assert result.discarded_terminal_samples == 0
    assert result.group_shifts_samples.shape == (31, 4)
    assert result.residual_obs_removed_rms == 0.0
    assert result.qrs.quality.correction_permitted
    assert result.qrs.quality.warnings
    assert result.qrs.diagnostics.model_sha256 == "test-model-sha256"
    assert result.cardiac_qc.rms_attenuation_db == pytest.approx(20.0 * np.log10(4.0))
    assert result.cardiac_qc.peak_to_peak_attenuation_db == pytest.approx(20.0 * np.log10(4.0))
    assert set(result.harmonic_stages) == {"raw", "gradient_corrected", "final"}
    for stage in result.harmonic_stages.values():
        assert stage.summary["stage"] in {"raw", "gradient_corrected", "final"}
        assert stage.spectrum.frequencies_hz.shape == stage.spectrum.median_power_db.shape


def test_preprocess_raw_crops_incomplete_terminal_scanner_interval(monkeypatch) -> None:
    raw = _raw(incomplete_terminal_volume=True)
    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.eeg_fmri.pipeline.apply_cardiac_obs_in_place",
        lambda raw, **kwargs: raw._data.__setitem__(slice(0, 2), raw._data[:2] * 0.5),
    )

    result = preprocess_raw_in_place(
        raw,
        parameters=_parameters(),
        qrs_detector=_FixedQrsDetector(),
        slice_schedule=_schedule(),
    )

    assert result.complete_volume_count == 30
    assert result.discarded_terminal_samples == 50
    assert raw.n_times == pytest.approx(3_025 * 500 / 1_000, abs=1)


def test_preprocess_raw_corrects_contiguous_blocks_across_scanner_restart(monkeypatch) -> None:
    raw = _raw(
        incomplete_terminal_volume=False,
        internal_scanner_restart=True,
    )
    gap_before = raw.get_data(start=1_625, stop=1_675).copy()
    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.eeg_fmri.pipeline.apply_cardiac_obs_in_place",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.eeg_fmri.pipeline._low_pass_and_resample",
        lambda *_args, **_kwargs: None,
    )

    result = preprocess_raw_in_place(
        raw,
        parameters=_parameters(),
        qrs_detector=_FixedQrsDetector(),
        slice_schedule=_schedule(),
    )

    assert result.complete_volume_count == 31
    np.testing.assert_allclose(
        result.raw.get_data(start=1_625, stop=1_675),
        gap_before,
        atol=1e-10,
    )


def test_preprocess_raw_rejects_task_event_inside_discarded_terminal_interval() -> None:
    raw = _raw(incomplete_terminal_volume=True)
    terminal_onset = 3.04
    raw.set_annotations(
        raw.annotations + mne.Annotations([terminal_onset], [0.0], ["Stim_on/S  1"])
    )

    with pytest.raises(ValueError, match="Stim_on/S  1"):
        preprocess_raw_in_place(
            raw,
            parameters=_parameters(),
            qrs_detector=_FixedQrsDetector(),
            slice_schedule=_schedule(),
        )
