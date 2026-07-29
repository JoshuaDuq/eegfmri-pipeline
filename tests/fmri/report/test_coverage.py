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


# --- extent, volume, and smoothness reporting -----------------------------


def test_mask_volume_scales_with_the_voxel_size() -> None:
    # The point of reporting mm^3: the same brain on a coarser grid has a third the
    # voxels and the same volume.
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 1
    fine = nib.Nifti1Image(mask, np.diag([1.0, 1.0, 1.0, 1.0]))
    coarse = nib.Nifti1Image(mask, np.diag([3.0, 1.0, 1.0, 1.0]))
    _, fine_volume = coverage.mask_volume_mm3(fine)
    _, coarse_volume = coverage.mask_volume_mm3(coarse)
    assert coarse_volume == pytest.approx(3.0 * fine_volume)


def test_an_extent_threshold_in_resels_falls_as_smoothness_rises() -> None:
    # The same voxel count is a strong constraint on unsmoothed data and almost none
    # at 8 mm, which is exactly what a bare count fails to say.
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    rough = coverage.extent_in_resels(50, mask_img=mask, fwhm=(2.0, 2.0, 2.0))
    smooth = coverage.extent_in_resels(50, mask_img=mask, fwhm=(8.0, 8.0, 8.0))
    assert rough > smooth
    assert smooth == pytest.approx(50.0 / 8.0**3)


def test_the_smoothness_note_names_where_the_estimate_came_from() -> None:
    # Estimated from a statistic map this overestimates wherever signal sits; the
    # same number from residuals would not, and the reader cannot tell them apart.
    note = coverage.smoothness_note((5.0, 5.2, 6.1), source="statistic map")
    assert "statistic map" in note and "FWHM" in note


def test_the_coverage_panel_reports_volume_not_a_field_of_view_ratio() -> None:
    # The old line counted air: the ratio measured how much empty space the
    # acquisition box contained.
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    figure = coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "cm³" in text
    assert "field of view" not in text
    plt.close(figure)


def test_the_coverage_panel_states_only_the_extent_it_was_told() -> None:
    # It previously asserted "intersection across N runs" over whatever mask it was
    # handed, which was false whenever that mask was a single run's.
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    figure = coverage.coverage_figure(
        nib.Nifti1Image(mask, np.eye(4)), extent_note="intersection across 6 runs"
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "intersection across 6 runs" in text
    plt.close(figure)


def test_the_coverage_panel_makes_no_extent_claim_by_default() -> None:
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    figure = coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "intersection" not in text
    plt.close(figure)


def test_the_coverage_panel_leaves_voxels_outside_the_mask_transparent() -> None:
    # Handed a continuous colormap, nilearn maps the mask's zeros to its low colour
    # and paints them, so the panel drew a solid block over the whole field-of-view
    # box -- covering the anatomy that shows which regions fell outside the model,
    # which is the only thing this panel is read for.
    from unittest.mock import patch

    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    with patch("nilearn.plotting.plot_roi") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    cmap = kwargs["cmap"]
    assert cmap(0.0)[3] == 0.0, "the zero colour must be fully transparent"
    assert cmap(1.0)[3] > 0.0, "the mask colour must be visible"
    assert kwargs["vmin"] == 0
    assert kwargs["vmax"] == 1
    # A scalar alpha overrides the colormap's per-colour alpha and would paint the
    # transparent entry too.
    assert "alpha" not in kwargs


def test_the_coverage_panel_does_not_let_nilearn_brighten_the_background() -> None:
    # At nilearn's "auto" dimming, air outside the head is lifted to mid grey and the
    # anatomy loses the contrast that shows which regions fell outside the model.
    from unittest.mock import patch

    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    with patch("nilearn.plotting.plot_roi") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["dim"] == 0


def test_a_binary_mask_gets_no_colour_scale() -> None:
    # A mask is in or out. A 0-to-1 scale beside it invites a reading of degree that
    # the two states do not carry.
    from unittest.mock import patch

    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    with patch("nilearn.plotting.plot_roi") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            coverage.coverage_figure(nib.Nifti1Image(mask, np.eye(4)))
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["colorbar"] is False
