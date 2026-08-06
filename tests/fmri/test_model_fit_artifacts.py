"""Exact fitted-model series persisted for report diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.model_fit import (
    ModelFitImages,
    extract_model_fit_images,
    write_model_fit_images,
)


def _image(
    value: float,
    *,
    frames: int = 5,
    affine: np.ndarray | None = None,
) -> nib.Nifti1Image:
    data = np.full((4, 4, 4, frames), value, dtype=np.float32)
    return nib.Nifti1Image(data, np.eye(4) if affine is None else affine)


def _model(*, runs: int = 2, frames: int = 5) -> SimpleNamespace:
    designs = [pd.DataFrame({"constant": np.ones(frames)}) for _ in range(runs)]
    labels = [np.zeros(64) for _ in range(runs)]
    results = [
        {
            0.0: SimpleNamespace(
                theta=np.full((1, 64), 1.5),
                Y=np.full((frames, 64), 1.75),
            )
        }
        for _ in range(runs)
    ]
    return SimpleNamespace(
        design_matrices_=designs,
        labels_=labels,
        results_=results,
        masker_=_VoxelMasker(),
    )


class _VoxelMasker:
    def __init__(
        self,
        *,
        frame_limit: int | None = None,
        prediction_frame_limit: int | None = None,
        prediction_affine: np.ndarray | None = None,
    ) -> None:
        self.frame_limit = frame_limit
        self.prediction_frame_limit = prediction_frame_limit
        self.prediction_affine = prediction_affine
        self.calls = 0

    def inverse_transform(self, series: np.ndarray) -> nib.Nifti1Image:
        self.calls += 1
        frame_limit = self.frame_limit
        affine = np.eye(4)
        if self.calls % 2 == 0:
            frame_limit = self.prediction_frame_limit or frame_limit
            affine = self.prediction_affine if self.prediction_affine is not None else affine
        values = np.asarray(series, dtype=np.float32)
        if frame_limit is not None:
            values = values[:frame_limit]
        data = values.T.reshape(4, 4, 4, -1)
        return nib.Nifti1Image(data, affine)


class _SingleVoxelMasker:
    def inverse_transform(self, series: np.ndarray) -> nib.Nifti1Image:
        data = np.asarray(series, dtype=np.float32).T.reshape(1, 1, 1, -1)
        return nib.Nifti1Image(data, np.eye(4))


def test_extracts_prediction_and_residual_in_the_unwhitened_model_response_space() -> None:
    design = pd.DataFrame(
        {
            "constant": np.ones(4),
            "condition": np.arange(4, dtype=float),
        }
    )
    coefficients = np.array([[2.0], [3.0]])
    expected_prediction = design.to_numpy() @ coefficients
    expected_residual = np.array([[0.5], [-0.5], [1.0], [-1.0]])
    response = expected_prediction + expected_residual

    model = SimpleNamespace(
        design_matrices_=[design],
        labels_=[np.array([0.0])],
        results_=[{0.0: SimpleNamespace(theta=coefficients, Y=response)}],
        masker_=_SingleVoxelMasker(),
        # These public Nilearn properties are deliberately inconsistent with X beta
        # under prewhitening and must not define the saved model-response artifacts.
        predicted_=[_image(-20.0, frames=4)],
        residuals_=[_image(30.0, frames=4)],
    )

    images = extract_model_fit_images(model)

    np.testing.assert_allclose(
        images.predicted[0].get_fdata().reshape(-1),
        expected_prediction.reshape(-1),
    )
    np.testing.assert_allclose(
        images.residuals[0].get_fdata().reshape(-1),
        expected_residual.reshape(-1),
    )


def test_extracts_one_residual_and_prediction_series_per_design() -> None:
    images = extract_model_fit_images(_model(runs=2))

    assert len(images.residuals) == 2
    assert len(images.predicted) == 2
    assert images.residuals[0].shape == (4, 4, 4, 5)


def test_rejects_a_missing_model_series() -> None:
    model = _model()
    del model.results_

    with pytest.raises(AttributeError, match="results_"):
        extract_model_fit_images(model)


def test_rejects_series_that_do_not_align_with_the_designs() -> None:
    model = _model(runs=2)
    model.results_ = model.results_[:1]

    with pytest.raises(ValueError, match="one label and regression-result set per design"):
        extract_model_fit_images(model)


def test_rejects_residual_and_prediction_shape_disagreement() -> None:
    model = _model(runs=1)
    model.masker_ = _VoxelMasker(prediction_frame_limit=4)

    with pytest.raises(ValueError, match="matching shapes"):
        extract_model_fit_images(model)


def test_rejects_residual_and_prediction_affine_disagreement() -> None:
    model = _model(runs=1)
    model.masker_ = _VoxelMasker(prediction_affine=np.diag([2.0, 2.0, 2.0, 1.0]))

    with pytest.raises(ValueError, match="matching affines"):
        extract_model_fit_images(model)


def test_rejects_a_time_axis_that_does_not_match_the_fitted_design() -> None:
    model = _model(runs=1, frames=5)
    model.masker_ = _VoxelMasker(frame_limit=4)

    with pytest.raises(ValueError, match="design rows"):
        extract_model_fit_images(model)


def test_writes_explicit_model_response_series_for_each_run(tmp_path: Path) -> None:
    images = extract_model_fit_images(_model(runs=2))
    paths = write_model_fit_images(
        images,
        out_dir=tmp_path,
        stem="sub-01_task-heat_contrast-pain",
        cfg_hash="abc123",
        run_labels=("run-01", "run-02"),
    )

    assert len(paths.residuals) == 2
    assert len(paths.predicted) == 2
    assert all(path.is_file() for path in (*paths.residuals, *paths.predicted))
    assert all("modelResponseResidual" in path.name for path in paths.residuals)
    assert all("modelResponsePredicted" in path.name for path in paths.predicted)


def test_writing_rejects_run_labels_that_do_not_align(tmp_path: Path) -> None:
    images = extract_model_fit_images(_model(runs=2))

    with pytest.raises(ValueError, match="run_labels"):
        write_model_fit_images(
            images,
            out_dir=tmp_path,
            stem="sub-01_task-heat_contrast-pain",
            cfg_hash="abc123",
            run_labels=("run-01",),
        )


def test_writing_rejects_residual_and_prediction_count_disagreement(tmp_path: Path) -> None:
    images = ModelFitImages(
        residuals=(_image(0.25), _image(0.5)),
        predicted=(_image(1.5),),
    )

    with pytest.raises(ValueError, match="residuals and predicted"):
        write_model_fit_images(
            images,
            out_dir=tmp_path,
            stem="sub-01_task-heat_contrast-pain",
            cfg_hash="abc123",
            run_labels=("run-01", "run-02"),
        )
