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
