from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import coverage


def _smooth_noise(sigma: float, shape=(24, 24, 24)) -> nib.Nifti1Image:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    data = gaussian_filter(rng.standard_normal(shape), sigma=sigma)
    return nib.Nifti1Image(data.astype(np.float32), np.eye(4))


def test_smoother_data_yields_a_larger_estimated_fwhm() -> None:
    rough = coverage.estimate_fwhm(_smooth_noise(0.5))
    smooth = coverage.estimate_fwhm(_smooth_noise(3.0))
    assert np.mean(smooth) > np.mean(rough)


def test_fwhm_is_reported_in_millimetres_using_the_voxel_size() -> None:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    data = gaussian_filter(rng.standard_normal((24, 24, 24)), sigma=2.0)
    unit = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    coarse = nib.Nifti1Image(data.astype(np.float32), np.diag([3.0, 3.0, 3.0, 1.0]))
    assert np.mean(coverage.estimate_fwhm(coarse)) == pytest.approx(
        3.0 * np.mean(coverage.estimate_fwhm(unit)), rel=0.05
    )


def test_fwhm_recovers_a_known_smoothing_kernel() -> None:
    # Guards the estimator itself, not just its monotonicity.
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    sigma = 2.0
    data = gaussian_filter(rng.standard_normal((40, 40, 40)), sigma=sigma)
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    expected = sigma * np.sqrt(8.0 * np.log(2.0))
    assert np.mean(coverage.estimate_fwhm(img)) == pytest.approx(expected, rel=0.10)


def test_fwhm_estimation_honours_a_mask() -> None:
    img = _smooth_noise(2.0)
    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[4:20, 4:20, 4:20] = True
    assert all(np.isfinite(coverage.estimate_fwhm(img, mask=mask)))


def test_fwhm_rejects_a_map_with_too_few_voxels_to_estimate() -> None:
    tiny = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="too small"):
        coverage.estimate_fwhm(tiny)


def test_fwhm_rejects_a_constant_map() -> None:
    flat = nib.Nifti1Image(np.ones((12, 12, 12), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="constant"):
        coverage.estimate_fwhm(flat)


def test_coverage_figure_states_the_modelled_voxel_count() -> None:
    mask = np.zeros((12, 12, 12), dtype=np.float32)
    mask[2:10, 2:10, 2:10] = 1.0
    figure = coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "512" in text  # 8 * 8 * 8
    plt.close(figure)


def test_coverage_figure_says_untested_voxels_were_not_tested() -> None:
    # The whole point: an unmodelled voxel is not a null result.
    mask = np.ones((12, 12, 12), dtype=np.float32)
    figure = coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "not tested" in text.lower()
    plt.close(figure)


def test_coverage_figure_returns_a_figure() -> None:
    mask = np.ones((12, 12, 12), dtype=np.float32)
    figure = coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
    assert figure is not None
    plt.close(figure)


def test_coverage_figure_rejects_an_empty_mask() -> None:
    empty = nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="no voxels"):
        coverage.coverage_figure(empty)
