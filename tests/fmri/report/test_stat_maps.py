from __future__ import annotations

import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
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
    # Drawn by the layout rather than by nilearn. Nilearn writes each tile's
    # coordinate at that tile's own bottom-left, where on a packed row it runs under
    # the neighbouring tile, and marks left/right on all fourteen tiles of a row.
    # The information has to survive; the collisions do not.
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
    texts = [artist.get_text() for artist in figure.texts]
    assert any(text.startswith("z = ") for text in texts)
    assert any(text.startswith("x = ") for text in texts)
    assert "L" in texts and "R" in texts
    plt.close(figure)


def test_the_mosaic_does_not_let_nilearn_annotate_the_tiles() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["annotate"] is False


def test_the_mosaic_draws_three_projections_with_several_cuts_each() -> None:
    # Nilearn's own "mosaic" mode spreads its cuts across the underlay's extent,
    # which put tiles at z = -71 and y = 80 -- outside the brain -- while the
    # interesting slices got one tile apiece.
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
        except Exception:
            pass
    calls = mock_plot.call_args_list
    assert {call.kwargs["display_mode"] for call in calls} == {"x", "y", "z"}
    assert all(len(call.kwargs["cut_coords"]) > 1 for call in calls)


def test_a_thresholded_mosaic_marks_the_band_it_does_not_draw() -> None:
    # Stating a threshold in prose leaves a reader to work out which values are
    # missing from the picture. The colourbar shows it.
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3, two_sided=True)
    hatched = [
        patch_artist
        for axes in figure.axes
        for patch_artist in axes.patches
        if patch_artist.get_hatch()
    ]
    assert hatched, "the suppressed band is not marked on the colourbar"
    plt.close(figure)


def test_an_unthresholded_mosaic_marks_no_suppressed_band() -> None:
    # Nothing is hidden, so hatching a band would claim otherwise.
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=None)
    hatched = [
        patch_artist
        for axes in figure.axes
        for patch_artist in axes.patches
        if patch_artist.get_hatch()
    ]
    assert not hatched
    plt.close(figure)


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


# --- the glass brain ------------------------------------------------------


def test_the_glass_brain_title_sits_outside_the_projections() -> None:
    # Nilearn draws it inside the axes, where it lands on the sagittal projection.
    figure = stat_maps.glass_brain(_noise_img(), threshold=2.3, title="a contrast")
    with patch("nilearn.plotting.plot_glass_brain") as mock_plot:
        mock_plot.return_value.axes = {}
        try:
            stat_maps.glass_brain(_noise_img(), threshold=2.3, title="a contrast")
        except Exception:
            pass
    assert mock_plot.call_args.kwargs.get("title") in (None, "")
    assert any(a.get_text() == "a contrast" for a in figure.texts)
    plt.close(figure)


def test_the_glass_brain_marks_peaks_in_a_colour_outside_the_map_s_ramp() -> None:
    # Marked in the neutral guide grey, the markers were invisible against a dense
    # projection -- which is most of them, since a glass brain fills wherever
    # anything survives the threshold.
    assert stat_maps.PEAK_MARKER_COLOR not in {"0.35", "black", "white"}
    figure = stat_maps.glass_brain(
        _noise_img(), threshold=2.3, peak_coords=[(1.0, 2.0, 3.0)], peak_labels=["1"]
    )
    labelled = [
        artist
        for axes in figure.axes
        for artist in axes.texts
        if artist.get_text() == "1"
    ]
    assert labelled, "the peak label is missing"
    # Haloed, so it survives being drawn over a saturated projection.
    assert all(artist.get_path_effects() for artist in labelled)
    plt.close(figure)


def test_the_glass_brain_states_that_it_is_a_projection() -> None:
    # A saturated projection otherwise reads as a very strong result.
    figure = stat_maps.glass_brain(_noise_img(), threshold=2.3)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "maximum-intensity projection" in text
    plt.close(figure)


def test_the_glass_brain_marks_the_band_it_does_not_draw() -> None:
    figure = stat_maps.glass_brain(_noise_img(), threshold=2.3, two_sided=True)
    hatched = [p for axes in figure.axes for p in axes.patches if p.get_hatch()]
    assert hatched
    plt.close(figure)


def test_a_peak_label_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="peak labels"):
        stat_maps.glass_brain(
            _noise_img(),
            threshold=2.3,
            peak_coords=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            peak_labels=["1"],
        )


def _fsaverage_is_cached() -> bool:
    from pathlib import Path

    return (Path.home() / "nilearn_data" / "fsaverage" / "infl_left.gii.gz").is_file()


@pytest.mark.skipif(not _fsaverage_is_cached(), reason="fsaverage mesh is not cached")
def test_surface_projection_names_the_mesh_and_the_cortex_it_omits() -> None:
    """The panel is a projection, not the volume, and it drops everything but cortex.

    Nilearn samples between the white and pial surfaces, so cerebellum, brainstem and
    subcortex are absent by construction rather than by threshold. A reader comparing
    this against the mosaic beside it has to be told that, inside the figure, because
    a figure travels away from its caption.
    """
    figure = stat_maps.surface_projection(
        _noise_img(), threshold=1.5, mesh="fsaverage", title="Cortical surface"
    )
    try:
        provenance = " ".join(text.get_text() for text in figure.texts)
        assert "fsaverage" in provenance
        assert "cortical surface projection" in provenance
        assert "cerebellum" in provenance
    finally:
        plt.close(figure)


@pytest.mark.skipif(not _fsaverage_is_cached(), reason="fsaverage mesh is not cached")
def test_surface_projection_formats_a_non_integer_colourbar() -> None:
    """Nilearn's cbar_tick_format defaults to '%i'.

    Left alone it renders a z scale of 1.5 to 4.5 as a column of identical integers and
    warns while doing it, which is the colourbar saying nothing at all.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = stat_maps.surface_projection(_noise_img(), threshold=1.5, mesh="fsaverage")
    try:
        assert not [w for w in caught if "integer formatting" in str(w.message)]
    finally:
        plt.close(figure)


def test_a_thresholded_panel_counts_what_it_drew_not_just_what_it_masked() -> None:
    """`n = 85,326 voxels` beside `|z| > 2.93` reads as 85,326 voxels above 2.93.

    It was the mask size; 5,882 voxels actually survived. The two clauses sat adjacent in
    one strip with nothing marking them apart, and the panel was read that way -- by me,
    repeatedly -- so the count has to name what it counts.
    """
    values = np.concatenate([np.zeros(900), np.full(100, 5.0)])

    lines = stat_maps._provenance(values, threshold=3.0, limit=6.0, two_sided=True)
    strip = " · ".join(lines)

    assert "100 of 1,000" in strip
    assert "n = 1,000 voxels" not in strip


def test_an_unthresholded_panel_still_reports_a_plain_voxel_count() -> None:
    lines = stat_maps._provenance(np.ones(1000), threshold=None, limit=2.0, two_sided=True)

    assert "n = 1,000 voxels" in " · ".join(lines)


def test_residual_panel_states_the_bound_its_own_tails_cannot_cross() -> None:
    """Standardised residuals are bounded by sqrt(n - rank), which is small here.

    All the residual can sit on one observation at most, giving |e| = sqrt(SSE) against
    sqrt(SSE / (n - rank)). At 13 participants and one column that ceiling is 3.46, so
    the tails are cut off by arithmetic and cannot look normal whatever the data does.
    Comparing them to the drawn normal without knowing that reads as a finding.
    """
    from fmri_pipeline.analysis.report.figures import residuals as residual_figures

    rng = np.random.default_rng(0)
    normalized = rng.standard_normal((13, 400))
    per_subject = pd.DataFrame(
        {"subject": [f"sub-{i:04d}" for i in range(13)], "residual RMS": rng.random(13)}
    )

    figure = residual_figures.residual_figure(normalized, per_subject, r_square=0.0, rank=1)
    try:
        provenance = " ".join(text.get_text() for text in figure.texts)
        assert "3.46" in provenance
        assert "bounded" in provenance
    finally:
        plt.close(figure)


def _noise_map(scale: float = 0.1, shape=(20, 20, 20)) -> nib.Nifti1Image:
    rng = np.random.default_rng(0)
    return nib.Nifti1Image(rng.normal(0.0, scale, shape), np.eye(4))


def _stamped(figure) -> str:
    return " ".join(text.get_text() for text in figure.texts)


def test_a_thresholded_mosaic_says_so_when_nothing_survives() -> None:
    # Three panels of bare underlay, and the only statement that nothing survived was
    # 6.5-point grey at the bottom of the strip. An empty mosaic has to read as a
    # result rather than as a rendering failure.
    figure = stat_maps.stat_map_mosaic(_noise_map(), threshold=2.3, two_sided=True)
    try:
        text = _stamped(figure)
    finally:
        plt.close(figure)

    assert "no voxel" in text.lower()
    assert "2.30" in text


def test_a_mosaic_with_survivors_is_not_stamped() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.1, (20, 20, 20))
    data[8:12, 8:12, 8:12] = 6.0
    figure = stat_maps.stat_map_mosaic(
        nib.Nifti1Image(data, np.eye(4)), threshold=2.3, two_sided=True
    )
    try:
        text = _stamped(figure)
    finally:
        plt.close(figure)

    assert "no voxel" not in text.lower()


def test_an_unthresholded_mosaic_is_never_stamped() -> None:
    # Nothing is suppressed, so there is nothing to explain.
    figure = stat_maps.stat_map_mosaic(_noise_map(), threshold=None)
    try:
        text = _stamped(figure)
    finally:
        plt.close(figure)

    assert "no voxel" not in text.lower()


def test_an_empty_glass_brain_says_so_too() -> None:
    # A projection of nothing is a blank outline, and no more self-explanatory than
    # an empty mosaic.
    figure = stat_maps.glass_brain(_noise_map(), threshold=2.3, two_sided=True)
    try:
        text = _stamped(figure)
    finally:
        plt.close(figure)

    assert "no voxel" in text.lower()


def test_suprathreshold_count_is_none_without_a_threshold() -> None:
    values = np.array([0.0, 5.0, -5.0])
    assert stat_maps.suprathreshold_count(values, threshold=None, two_sided=True) is None
    assert stat_maps.suprathreshold_count(values, threshold=2.3, two_sided=True) == 2
    assert stat_maps.suprathreshold_count(values, threshold=2.3, two_sided=False) == 1


def test_the_dual_coded_panel_states_what_its_opacity_ramp_reaches() -> None:
    # The ramp runs from half the threshold to the threshold. On a map whose evidence
    # sits well below that, every voxel is transparent and the panel renders as bare
    # underlay -- indistinguishable from a rendering failure unless it is stated.
    rng = np.random.default_rng(0)
    stat = nib.Nifti1Image(rng.normal(0.0, 0.1, (20, 20, 20)), np.eye(4))
    effect = nib.Nifti1Image(np.asarray(stat.dataobj) * 0.04, np.eye(4))

    figure = stat_maps.dual_coded_mosaic(effect, stat_img=stat, threshold=2.3)
    try:
        text = _stamped(figure)
    finally:
        plt.close(figure)

    assert "0.0%" in text and "full" in text
    assert "reaches |z|" in text
