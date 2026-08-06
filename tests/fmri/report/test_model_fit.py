from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import model_fit as model_fit_figures
from fmri_pipeline.analysis.report.figures.model_fit import (
    collect_residual_carpet,
    model_fit_table,
    summarize_model_fit,
    write_model_fit_tsv,
)


def _image(path: Path, data: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    predicted = np.array(
        [
            [[[0.0, 1.0, 2.0, 3.0]]],
            [[[0.0, 2.0, 4.0, 6.0]]],
        ]
    )
    residual = np.array(
        [
            [[[1.0, -1.0, 1.0, -1.0]]],
            [[[2.0, -2.0, 2.0, -2.0]]],
        ]
    )
    mask = np.ones((2, 1, 1), dtype=np.uint8)
    return (
        _image(tmp_path / "residual.nii.gz", residual),
        _image(tmp_path / "predicted.nii.gz", predicted),
        _image(tmp_path / "mask.nii.gz", mask),
    )


def test_model_fit_measurements_use_the_recorded_prediction_residual_decomposition(
    tmp_path: Path,
) -> None:
    residual, predicted, mask = _artifacts(tmp_path)

    measurements = summarize_model_fit(
        run_labels=("run-01",),
        residual_paths=(residual,),
        predicted_paths=(predicted,),
        mask_path=mask,
    )

    assert len(measurements) == 1
    run = measurements[0]
    assert run.label == "run-01"
    assert run.retained_frames == 4
    assert run.r_squared == pytest.approx((0.2, 0.2, 0.2))
    assert run.residual_standard_deviation == pytest.approx((1.25, 1.5, 1.75))
    assert run.residual_autocorrelation_lag_1 == pytest.approx((-0.75, -0.75, -0.75))


def test_model_fit_table_and_tsv_carry_the_same_exact_measurements(
    tmp_path: Path,
) -> None:
    residual, predicted, mask = _artifacts(tmp_path)
    measurements = summarize_model_fit(
        run_labels=("run-01",),
        residual_paths=(residual,),
        predicted_paths=(predicted,),
        mask_path=mask,
    )

    table_html, rows = model_fit_table(
        measurements,
        response_units="% signal change",
    )
    tsv_path = write_model_fit_tsv(
        measurements,
        path=tmp_path / "model_fit_measurements.tsv",
        response_units="% signal change",
    )

    assert "Median R²" in table_html
    assert "Residual SD (% signal change)" in table_html
    assert "0.200000" in table_html
    assert "-0.750000" in table_html
    assert tsv_path.read_text(encoding="utf-8").splitlines() == rows


def test_model_fit_measurements_reject_undefined_voxelwise_statistics(
    tmp_path: Path,
) -> None:
    constant = np.ones((1, 1, 1, 4), dtype=np.float32)
    zero = np.zeros_like(constant)
    residual = _image(tmp_path / "residual.nii.gz", zero)
    predicted = _image(tmp_path / "predicted.nii.gz", constant)
    mask = _image(tmp_path / "mask.nii.gz", np.ones((1, 1, 1), dtype=np.uint8))

    with pytest.raises(ValueError, match="undefined voxelwise R²"):
        summarize_model_fit(
            run_labels=("run-01",),
            residual_paths=(residual,),
            predicted_paths=(predicted,),
            mask_path=mask,
        )


def test_residual_carpet_reconstructs_the_exact_acquired_frame_axis(
    tmp_path: Path,
) -> None:
    mask = _image(
        tmp_path / "mask.nii.gz",
        np.ones((2, 1, 1), dtype=np.uint8),
    )
    first = _image(
        tmp_path / "run-01_residual.nii.gz",
        np.array(
            [
                [[[1.0, 3.0]]],
                [[[2.0, 6.0]]],
            ]
        ),
    )
    second = _image(
        tmp_path / "run-02_residual.nii.gz",
        np.array(
            [
                [[[2.0, 4.0, 6.0]]],
                [[[1.0, 2.0, 3.0]]],
            ]
        ),
    )

    carpet = collect_residual_carpet(
        residual_paths=(first, second),
        retained_frame_indices=((1, 3), (0, 1, 2)),
        acquired_frame_counts=(4, 3),
        mask_path=mask,
        tissue_codes=None,
    )

    assert carpet.values.shape == (2, 7)
    assert carpet.run_boundaries == (4,)
    assert carpet.not_retained.tolist() == [True, False, True, False, False, False, False]
    assert np.isnan(carpet.values[:, [0, 2]]).all()
    np.testing.assert_allclose(carpet.values[0, [1, 3]], [-1.0, 1.0])
    np.testing.assert_allclose(carpet.values[0, 4:], [-1.22474487, 0.0, 1.22474487])


def test_pooled_residual_sd_uses_every_retained_sample_inside_the_fitted_mask(
    tmp_path: Path,
) -> None:
    mask = _image(
        tmp_path / "mask.nii.gz",
        np.array([[[1]], [[1]], [[0]]], dtype=np.uint8),
    )
    first_values = np.array(
        [
            [[[-1.0, 1.0]]],
            [[[-2.0, 2.0]]],
            [[[100.0, 100.0]]],
        ]
    )
    second_values = np.array(
        [
            [[[-3.0, 0.0, 3.0]]],
            [[[-1.0, 0.0, 1.0]]],
            [[[200.0, 200.0, 200.0]]],
        ]
    )
    first = _image(tmp_path / "run-01_residual.nii.gz", first_values)
    second = _image(tmp_path / "run-02_residual.nii.gz", second_values)

    result = model_fit_figures.pooled_residual_standard_deviation(
        residual_paths=(first, second),
        mask_path=mask,
    )

    expected = np.std(
        np.concatenate([first_values, second_values], axis=3),
        axis=3,
        ddof=0,
    )
    actual = np.asarray(result.image.dataobj)
    np.testing.assert_allclose(actual[:2], expected[:2])
    assert actual[2, 0, 0] == 0.0
    assert result.run_count == 2
    assert result.retained_frames == 5
    assert result.voxel_count == 2
