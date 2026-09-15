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
    reference = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    rough = coverage.extent_in_resels(50, reference_img=reference, fwhm=(2.0, 2.0, 2.0))
    smooth = coverage.extent_in_resels(50, reference_img=reference, fwhm=(8.0, 8.0, 8.0))
    assert rough > smooth
    assert smooth == pytest.approx(50.0 / 8.0**3)


def test_the_search_volume_in_resels_counts_independent_tests_not_voxels() -> None:
    # The gap this exists to show: smoothing makes neighbouring voxels one
    # measurement, while every corrected height in the report divides alpha across
    # the voxel count.
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[2:18, 2:18, 2:18] = 1
    img = nib.Nifti1Image(mask, np.diag([3.0, 3.0, 3.0, 1.0]))
    voxels, _volume = coverage.mask_volume_mm3(img)
    resels = coverage.search_volume_resels(img, fwhm=(6.0, 6.0, 6.0))
    assert resels < voxels
    # 16^3 voxels of 27 mm^3 in resels of 216 mm^3.
    assert resels == pytest.approx(16**3 * 27.0 / 216.0)


def test_the_search_volume_in_resels_equals_the_voxel_count_at_one_voxel_smoothness() -> None:
    # The degenerate case that fixes the scale: with no smoothing every voxel is its
    # own resel, so the two counts must coincide.
    mask = np.ones((8, 8, 8), dtype=np.uint8)
    img = nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0]))
    assert coverage.search_volume_resels(img, fwhm=(2.0, 2.0, 2.0)) == pytest.approx(8**3)


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


def _run_img(shape=(12, 12, 12), *, filled: slice, seed: int = 0) -> nib.Nifti1Image:
    """A single-volume EPI whose signal occupies ``filled`` along x."""
    rng = np.random.default_rng(seed)
    data = np.zeros(shape + (4,), dtype=float)
    data[filled] = 800.0 + rng.normal(0, 5.0, data[filled].shape)
    return nib.Nifti1Image(data, np.eye(4))


def test_run_contribution_counts_how_many_runs_reach_each_voxel() -> None:
    # The analysis mask is an intersection, so the voxels some runs hold and others do
    # not are thrown away before the panel is drawn -- and those are exactly the
    # voxels a coverage panel is consulted for.
    runs = [
        _run_img(filled=slice(2, 10), seed=0),
        _run_img(filled=slice(2, 10), seed=1),
        _run_img(filled=slice(4, 10), seed=2),
    ]
    counts = coverage.run_contribution_map(runs)
    data = np.asanyarray(counts.dataobj)

    assert data.max() == 3
    # The band only two runs reach must be distinguishable from the band all three do.
    assert 0 < data[3, 6, 6] < 3
    assert data[6, 6, 6] == 3


def test_run_contribution_needs_more_than_one_run() -> None:
    with pytest.raises(ValueError, match="more than one run"):
        coverage.run_contribution_map([_run_img(filled=slice(2, 10))])


def test_the_coverage_panel_draws_the_contribution_map_when_it_is_given_one(
    caplog,
) -> None:
    # Asserting on the strip alone passed over a panel whose every row had failed:
    # mosaic_figure catches a row that will not render so the other two survive, so a
    # broken overlay produces an empty picture and a warning, not an exception. The
    # overlay itself has to be checked.
    import logging

    runs = [
        _run_img(filled=slice(2, 10), seed=0),
        _run_img(filled=slice(4, 10), seed=1),
    ]
    counts = coverage.run_contribution_map(runs)
    binary = nib.Nifti1Image(
        (np.asanyarray(counts.dataobj) == 2).astype(np.uint8), np.eye(4)
    )

    with caplog.at_level(logging.WARNING):
        figure = coverage.coverage_figure(
            binary,
            bg_img=binary,
            contribution_img=counts,
            n_runs=2,
            title="Analysis mask",
        )
    try:
        text = " ".join(artist.get_text() for artist in figure.texts)
        # One image per tile for the underlay, a second for the count overlay.
        overlays = [axis for axis in figure.axes if len(axis.images) >= 2]
    finally:
        plt.close(figure)

    assert "Could not draw" not in caplog.text, caplog.text
    assert overlays, "the contribution overlay was never drawn"
    assert "run" in text.lower()


def test_a_single_run_falls_back_to_the_plain_mask() -> None:
    # With one run every modelled voxel is reached by every run, so the count map is
    # uniform and says nothing the mask does not.
    mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.uint8), np.eye(4))
    figure = coverage.coverage_figure(mask, bg_img=mask, contribution_img=None, n_runs=1)
    try:
        text = " ".join(artist.get_text() for artist in figure.texts)
    finally:
        plt.close(figure)

    assert "how many of the" not in text
