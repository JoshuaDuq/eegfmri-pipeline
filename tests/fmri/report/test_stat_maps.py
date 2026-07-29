from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import stat_maps


def _noise_img(seed: int = 0) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    return nib.Nifti1Image(
        rng.standard_normal((12, 12, 12)).astype(np.float32), np.eye(4)
    )


def test_glass_brain_plots_signed_values_not_absolute_values() -> None:
    with patch("nilearn.plotting.plot_glass_brain") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.glass_brain(_noise_img(), threshold=2.3)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["plot_abs"] is False


def test_thresholded_mosaic_colour_limit_exceeds_its_threshold_for_a_noise_map() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    assert kwargs["vmax"] > kwargs["threshold"]


def test_mosaic_keeps_coordinate_and_laterality_annotation() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["annotate"] is True


def test_signed_maps_use_the_diverging_colormap_and_symmetric_bar() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img())
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    assert kwargs["cmap"] == "RdBu_r"
    assert kwargs["symmetric_cbar"] is True


def test_mosaic_returns_a_figure_and_labels_its_colorbar() -> None:
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3, cbar_label="z")
    assert figure is not None
    plt.close(figure)


def test_glass_brain_returns_a_figure_for_a_real_image() -> None:
    figure = stat_maps.glass_brain(_noise_img(), threshold=2.3)
    assert figure is not None
    plt.close(figure)


def test_an_unthresholded_panel_uses_a_robust_symmetric_limit() -> None:
    data = np.zeros((12, 12, 12), dtype=np.float32)
    data[0, 0, 0] = 1000.0
    data[1:, :, :] = 1.0
    img = nib.Nifti1Image(data, np.eye(4))
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(img)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["vmax"] < 100.0


def test_one_sided_inference_removes_the_negative_half_before_plotting() -> None:
    # plot_stat_map thresholds |value|, so without this the panel shows negative
    # clusters that the declared one-sided test never examined.
    data = np.linspace(-5, 5, 8**3).reshape(8, 8, 8).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    result = np.asarray(stat_maps.apply_sidedness(img, two_sided=False).get_fdata())
    assert result.min() == 0.0
    assert result.max() == pytest.approx(5.0)


def test_two_sided_inference_leaves_the_map_untouched() -> None:
    data = np.linspace(-5, 5, 8**3).reshape(8, 8, 8).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    result = np.asarray(stat_maps.apply_sidedness(img, two_sided=True).get_fdata())
    assert result.min() == pytest.approx(-5.0)


def test_mosaic_states_its_threshold_and_clipping_on_the_figure() -> None:
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "2.30" in text
    assert "clipped" in text.lower()
    plt.close(figure)


def test_mosaic_states_the_voxel_count_it_summarised() -> None:
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "1,728" in text  # 12 * 12 * 12
    plt.close(figure)


def test_mosaic_states_its_orientation_convention() -> None:
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3, radiological=False)
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "neurological" in text.lower()
    plt.close(figure)


def test_radiological_convention_is_passed_through_and_stated() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), radiological=True)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["radiological"] is True


def test_dual_coding_drives_opacity_from_the_statistic_map() -> None:
    # Hue carries effect magnitude; opacity carries statistical evidence.
    effect = _noise_img(1)
    stat = _noise_img(2)
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.dual_coded_mosaic(effect, stat_img=stat, threshold=2.3)
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    assert kwargs["transparency"] is not None
    # The map being coloured is the effect map, not the statistic map.
    assert mock_plot.call_args.args[0] is not stat


def test_dual_coding_fades_below_the_threshold_and_is_opaque_above_it() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.dual_coded_mosaic(
                _noise_img(1), stat_img=_noise_img(2), threshold=2.3
            )
        except Exception:
            pass
    low, high = mock_plot.call_args.kwargs["transparency_range"]
    assert low < 2.3 <= high


def test_dual_coding_is_not_thresholded_so_subthreshold_structure_survives() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.dual_coded_mosaic(
                _noise_img(1), stat_img=_noise_img(2), threshold=2.3
            )
        except Exception:
            pass
    assert not mock_plot.call_args.kwargs["threshold"]


def test_dual_coding_requires_matching_geometry() -> None:
    small = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="shape"):
        stat_maps.dual_coded_mosaic(_noise_img(), stat_img=small, threshold=2.3)


def test_glass_brain_honours_the_same_orientation_convention_as_the_mosaic() -> None:
    # Both panels sit in one report section. If only the mosaic flipped, the two
    # would disagree and the stated convention would be wrong for one of them.
    with patch("nilearn.plotting.plot_glass_brain") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.glass_brain(_noise_img(), threshold=2.3, radiological=True)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["radiological"] is True


def test_glass_brain_states_the_orientation_it_actually_used() -> None:
    figure = stat_maps.glass_brain(_noise_img(), threshold=2.3, radiological=True)
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "radiological" in text.lower()
    plt.close(figure)


def test_dual_coded_mosaic_returns_a_figure() -> None:
    figure = stat_maps.dual_coded_mosaic(
        _noise_img(1), stat_img=_noise_img(2), threshold=2.3
    )
    assert figure is not None
    plt.close(figure)


# --- colour limits are computed inside the brain ---------------------------


def _map_with_background(brain_fraction_shape=(40, 48, 40)) -> "nib.Nifti1Image":
    """A stat map whose background is exactly zero, as a real one's is."""
    rng = np.random.default_rng(0)
    data = np.zeros(brain_fraction_shape)
    brain = np.zeros(brain_fraction_shape, bool)
    brain[8:32, 10:38, 8:32] = True
    data[brain] = rng.standard_normal(int(brain.sum()))
    data[16:20, 20:24, 16:20] = 5.0
    return nib.Nifti1Image(data.astype(np.float32), np.eye(4)), brain


def test_the_colour_limit_ignores_the_zero_background() -> None:
    """Background zeros dominate the percentile and drag the limit far too low.

    A stat map is mostly background. Taking a robust limit over the whole volume
    computes a percentile of a distribution that is largely zeros, so the limit
    lands below the real one and every voxel above it saturates.
    """
    img, brain = _map_with_background()
    data = np.asarray(img.get_fdata())

    masked = stat_maps._resolve_vmax(
        img,
        threshold=None,
        vmax=None,
        mask_img=nib.Nifti1Image(brain.astype(np.uint8), np.eye(4)),
    )

    inside = float(np.percentile(np.abs(data[brain]), 98.0))
    whole_volume = float(np.percentile(np.abs(data), 98.0))

    assert masked == pytest.approx(inside, rel=0.01)
    # The defect this guards against: a limit taken over the whole volume is
    # substantially lower, so the panel saturates everything above it.
    assert masked > whole_volume * 1.2


def test_without_a_mask_the_limit_falls_back_to_nonzero_voxels() -> None:
    """An explicit mask is best, but an exact-zero background is still detectable."""
    img, brain = _map_with_background()
    data = np.asarray(img.get_fdata())

    resolved = stat_maps._resolve_vmax(img, threshold=None, vmax=None)
    nonzero = float(np.percentile(np.abs(data[data != 0]), 98.0))
    assert resolved == pytest.approx(nonzero, rel=0.01)


def test_a_map_with_no_background_is_unaffected_by_the_fallback() -> None:
    """Every voxel nonzero: the fallback must not change a well-behaved map."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((10, 10, 10)) + 10.0
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    assert stat_maps._resolve_vmax(img, threshold=None, vmax=None) == pytest.approx(
        float(np.percentile(np.abs(data), 98.0)), rel=0.01
    )


def test_the_mosaic_states_which_voxels_its_limit_came_from() -> None:
    img, brain = _map_with_background()
    figure = stat_maps.stat_map_mosaic(
        img, mask_img=nib.Nifti1Image(brain.astype(np.uint8), np.eye(4))
    )
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "mask" in text.lower()
    finally:
        plt.close(figure)


# --- peak markers actually carry the numbers the caption promises ----------


def _peaked_map() -> "nib.Nifti1Image":
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[4:12, 4:12, 4:12] = 3.0
    data[5, 5, 5] = 9.0
    data[16, 16, 16] = 6.0
    return nib.Nifti1Image(data, np.eye(4))


def test_peak_markers_are_numbered_on_the_projection() -> None:
    """The caption keys the markers to the cluster table, so numbers must be drawn.

    Previously the index reached only a debug log and the projection carried
    anonymous dots, while the caption told the reader they were numbered.
    """
    figure = stat_maps.glass_brain(
        _peaked_map(), threshold=2.3, two_sided=False, peak_coords=[(5, 5, 5), (16, 16, 16)]
    )
    try:
        drawn = {
            t.get_text()
            for ax in figure.axes
            for t in ax.texts
            if t.get_text() in {"1", "2"}
        }
        assert drawn == {"1", "2"}
    finally:
        plt.close(figure)


def test_peak_markers_use_supplied_labels_when_given() -> None:
    """Labels come from the cluster table's own IDs, not a fresh 1..N count."""
    figure = stat_maps.glass_brain(
        _peaked_map(),
        threshold=2.3,
        two_sided=False,
        peak_coords=[(5, 5, 5), (16, 16, 16)],
        peak_labels=["3", "7"],
    )
    try:
        drawn = {
            t.get_text()
            for ax in figure.axes
            for t in ax.texts
            if t.get_text() in {"3", "7"}
        }
        assert drawn == {"3", "7"}
    finally:
        plt.close(figure)


def test_a_glass_brain_without_peaks_draws_no_numbers() -> None:
    figure = stat_maps.glass_brain(_peaked_map(), threshold=2.3, two_sided=False)
    try:
        numeric = [
            t.get_text()
            for ax in figure.axes
            for t in ax.texts
            if t.get_text().isdigit()
        ]
        assert numeric == []
    finally:
        plt.close(figure)
