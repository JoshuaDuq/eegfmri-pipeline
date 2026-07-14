from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

MODULE_NAME = "eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs"


def _module():
    return importlib.import_module(MODULE_NAME)


def _valid_weights(module) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in module.NEUXUS_WEIGHT_SHAPES.items()
    }


def _write_archive(path: Path, weights: dict[str, np.ndarray]) -> str:
    np.savez(path, **weights)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_loader_validates_and_freezes_weights(tmp_path: Path) -> None:
    module = _module()
    archive_path = tmp_path / "weights.npz"
    digest = _write_archive(archive_path, _valid_weights(module))

    model = module.load_neuxus_qrs_model(archive_path, expected_sha256=digest)

    assert model.window_samples == 500
    assert model.hidden_units == 64
    assert model.sha256 == digest
    assert set(model.weights) == set(module.NEUXUS_WEIGHT_SHAPES)
    assert all(not weight.flags.writeable for weight in model.weights.values())
    with pytest.raises(TypeError):
        model.weights["wd"] = np.zeros((128, 1), dtype=np.float32)


def test_model_loader_rejects_checksum_mismatch(tmp_path: Path) -> None:
    module = _module()
    archive_path = tmp_path / "weights.npz"
    _write_archive(archive_path, _valid_weights(module))

    with pytest.raises(ValueError, match="SHA-256"):
        module.load_neuxus_qrs_model(archive_path, expected_sha256="0" * 64)


@pytest.mark.parametrize("defect", ["missing", "shape", "dtype"])
def test_model_loader_rejects_invalid_parameters(tmp_path: Path, defect: str) -> None:
    module = _module()
    weights = _valid_weights(module)
    if defect == "missing":
        del weights["wd"]
        expected_message = "keys"
    elif defect == "shape":
        weights["wd"] = np.zeros((127, 1), dtype=np.float32)
        expected_message = "shape"
    else:
        weights["wd"] = np.zeros((128, 1), dtype=np.float64)
        expected_message = "float32"
    archive_path = tmp_path / "weights.npz"
    digest = _write_archive(archive_path, weights)

    with pytest.raises(ValueError, match=expected_message):
        module.load_neuxus_qrs_model(archive_path, expected_sha256=digest)


def test_packaged_model_loads_with_recorded_identity() -> None:
    module = _module()

    model = module.load_packaged_neuxus_qrs_model()

    assert model.sha256 == "b8b6514e8150d92d42af149137524ded2429053124bdfdb43e8d3b5e8f4f78fc"


def test_inference_matches_pinned_neuxus_reference() -> None:
    module = _module()
    model = module.load_packaged_neuxus_qrs_model()
    window = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    reference_indices = np.array([0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499])
    reference_probabilities = np.array(
        [
            0.0021473477,
            0.0000063115,
            0.0000093780,
            0.0000628504,
            0.0003273452,
            0.0006898817,
            0.0008298367,
            0.0004884555,
            0.0003746858,
            0.0001492661,
            0.0003185509,
            0.2542915344,
        ],
        dtype=np.float32,
    )

    probabilities = module.predict_neuxus_qrs(window, model)

    assert probabilities.shape == (500,)
    assert probabilities.dtype == np.float32
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(
        probabilities[reference_indices],
        reference_probabilities,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_array_equal(probabilities, module.predict_neuxus_qrs(window, model))


@pytest.mark.parametrize(
    "window, message",
    [
        (np.zeros(499, dtype=np.float32), "shape"),
        (np.zeros((500, 1), dtype=np.float32), "shape"),
        (np.full(500, np.nan, dtype=np.float32), "finite"),
    ],
)
def test_inference_rejects_invalid_windows(window: np.ndarray, message: str) -> None:
    module = _module()
    model = module.load_packaged_neuxus_qrs_model()

    with pytest.raises(ValueError, match=message):
        module.predict_neuxus_qrs(window, model)


class _ThresholdPredictor:
    window_samples = 500
    model_sha256 = "test-model"

    def predict(self, window: np.ndarray) -> np.ndarray:
        probabilities = np.zeros(window.size, dtype=np.float32)
        probabilities[window > 0.25] = 0.9
        return probabilities


def _synthetic_ecg(
    peak_times: np.ndarray,
    *,
    amplitudes: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    sampling_frequency = 1_000.0
    time = np.arange(7_000) / sampling_frequency
    if amplitudes is None:
        amplitudes = np.ones(peak_times.size)
    ecg = 0.01 * np.sin(2 * np.pi * 2.0 * time)
    for peak_time, amplitude in zip(peak_times, amplitudes, strict=True):
        ecg += amplitude * np.exp(-0.5 * ((time - peak_time) / 0.025) ** 2)
    return ecg, sampling_frequency


def _detection_parameters(module):
    return module.NeuXusQrsDetectionParameters(
        sampling_frequency_hz=250.0,
        low_frequency_hz=0.5,
        high_frequency_hz=30.0,
        window_stride_samples=50,
        probability_threshold=0.05,
        minimum_support_samples=5,
        refractory_period_seconds=0.4,
        edge_margin_seconds=0.1,
    )


def test_detection_recovers_peaks_from_overlapping_windows() -> None:
    module = _module()
    expected_times = np.arange(1.0, 6.0)
    ecg, sampling_frequency = _synthetic_ecg(expected_times)
    detector = module.NeuXusQrsDetector(
        predictor=_ThresholdPredictor(),
        parameters=_detection_parameters(module),
    )

    detection = detector.detect(ecg, sampling_frequency_hz=sampling_frequency)

    np.testing.assert_allclose(detection.times, expected_times, atol=0.02)
    assert detection.sampling_frequency_hz == 250.0
    assert detection.probabilities.shape == detection.filtered_ecg.shape
    assert np.all(detection.probability_support[detection.probability_support > 0] >= 1)
    assert detection.model_sha256 == "test-model"


def test_detection_refractory_period_keeps_the_stronger_peak() -> None:
    module = _module()
    peak_times = np.array([1.0, 1.2, 2.0, 3.0, 4.0, 5.0])
    amplitudes = np.array([1.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ecg, sampling_frequency = _synthetic_ecg(peak_times, amplitudes=amplitudes)
    detector = module.NeuXusQrsDetector(
        predictor=_ThresholdPredictor(),
        parameters=_detection_parameters(module),
    )

    detection = detector.detect(ecg, sampling_frequency_hz=sampling_frequency)

    assert np.min(np.abs(detection.times - 1.0)) <= 0.02
    assert np.min(np.abs(detection.times - 1.2)) > 0.1


@pytest.mark.parametrize(
    "ecg, sampling_frequency, message",
    [
        (np.zeros(7_000), 1_000.0, "flat"),
        (np.full(7_000, np.nan), 1_000.0, "finite"),
        (np.zeros((1, 7_000)), 1_000.0, "one-dimensional"),
        (np.zeros(7_000), 999.0, "integer multiple"),
    ],
)
def test_detection_rejects_invalid_ecg(
    ecg: np.ndarray,
    sampling_frequency: float,
    message: str,
) -> None:
    module = _module()
    detector = module.NeuXusQrsDetector(
        predictor=_ThresholdPredictor(),
        parameters=_detection_parameters(module),
    )

    with pytest.raises(ValueError, match=message):
        detector.detect(ecg, sampling_frequency_hz=sampling_frequency)
