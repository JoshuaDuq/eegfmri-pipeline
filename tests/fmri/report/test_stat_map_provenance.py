"""What a map panel says about itself, where saying it wrongly is invisible."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report import style
from fmri_pipeline.analysis.report.figures import stat_maps


def _provenance(figure: plt.Figure) -> str:
    return " ".join(artist.get_text() for artist in figure.texts)


def _map_with_background(seed: int = 0) -> tuple:
    """A map whose brain occupies a corner, the rest exact zero."""
    rng = np.random.default_rng(seed)
    data = np.zeros((16, 16, 16), dtype=np.float32)
    brain = np.zeros((16, 16, 16), dtype=bool)
    brain[2:10, 2:10, 2:10] = True
    data[brain] = rng.standard_normal(int(brain.sum())).astype(np.float32) * 1.4
    return (
        nib.Nifti1Image(data, np.eye(4)),
        nib.Nifti1Image(brain.astype(np.uint8), np.eye(4)),
    )


def test_a_thresholded_panel_clips_against_the_voxels_it_draws() -> None:
    # Over the whole mask the sub-threshold voxels dominate and the clipped fraction
    # collapses toward zero, describing saturation the panel does not have.
    stat_img, mask_img = _map_with_background()
    figure = stat_maps.stat_map_mosaic(stat_img, mask_img=mask_img, threshold=1.5)
    text = _provenance(figure)
    assert "of drawn voxels" in text
    plt.close(figure)


def test_an_unthresholded_panel_clips_against_every_voxel_it_draws() -> None:
    stat_img, mask_img = _map_with_background()
    figure = stat_maps.stat_map_mosaic(stat_img, mask_img=mask_img, threshold=None)
    assert "of drawn voxels" not in _provenance(figure)
    plt.close(figure)


def test_the_clipped_fraction_of_a_thresholded_panel_is_the_suprathreshold_one() -> None:
    stat_img, mask_img = _map_with_background(seed=1)
    data = np.asarray(stat_img.get_fdata())
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    values = data[mask]

    # The limit is recomputed rather than parsed back out of the caption: the caption
    # rounds to two decimals, and at this sample size one boundary voxel is 0.8%.
    limit = style.suprathreshold_limit(values, threshold=1.5)
    surviving = values[np.abs(values) > 1.5]
    expected = float(np.mean(np.abs(surviving) > limit))
    assert expected > 0  # otherwise the test passes for the wrong reason

    figure = stat_maps.stat_map_mosaic(stat_img, mask_img=mask_img, threshold=1.5)
    reported = float(_provenance(figure).split("(")[1].split("%")[0]) / 100.0
    assert reported == pytest.approx(expected, abs=0.001)
    plt.close(figure)


def test_a_mask_that_does_not_fit_the_map_is_named_as_such() -> None:
    # Reporting this as "no mask supplied" sends the reader to the manifest to look
    # for a mask that is in fact recorded, and present, and the wrong shape.
    stat_img, _mask = _map_with_background()
    wrong = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    figure = stat_maps.stat_map_mosaic(stat_img, mask_img=wrong, threshold=None)
    assert "did not fit the map" in _provenance(figure)
    plt.close(figure)


def test_an_absent_mask_is_still_named_an_absent_mask() -> None:
    stat_img, _mask = _map_with_background()
    figure = stat_maps.stat_map_mosaic(stat_img, mask_img=None, threshold=None)
    assert "no mask supplied" in _provenance(figure)
    plt.close(figure)


def test_a_thresholded_panel_survives_nothing_surviving() -> None:
    stat_img, mask_img = _map_with_background()
    figure = stat_maps.stat_map_mosaic(stat_img, mask_img=mask_img, threshold=99.0)
    assert "0.0% clipped" in _provenance(figure)
    plt.close(figure)


# --- unsigned magnitudes --------------------------------------------------


def _magnitude_img(seed: int = 0):
    """A standard-error-like map: zero outside the brain, positive inside."""
    rng = np.random.default_rng(seed)
    data = np.zeros((16, 16, 16), dtype=np.float32)
    brain = np.zeros((16, 16, 16), dtype=bool)
    brain[2:12, 2:12, 2:12] = True
    data[brain] = (0.4 + rng.random(int(brain.sum()))).astype(np.float32)
    return (
        nib.Nifti1Image(data, np.eye(4)),
        nib.Nifti1Image(brain.astype(np.uint8), np.eye(4)),
    )


def test_a_magnitude_scale_starts_at_zero_and_is_not_symmetric() -> None:
    # Through the signed path a standard error spanning 0 to 1.09 got limits of
    # +/-1.09: half the ramp went to values that cannot occur and every voxel landed
    # in the top quarter of the colours, so the panel rendered as a flat wash.
    img, mask = _magnitude_img()
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.magnitude_mosaic(img, mask_img=mask)
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    assert kwargs["vmin"] == 0.0
    assert kwargs["symmetric_cbar"] is False
    assert kwargs["vmax"] > 0


def test_a_magnitude_panel_uses_the_perceptually_uniform_ramp_by_default() -> None:
    img, mask = _magnitude_img()
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.magnitude_mosaic(img, mask_img=mask)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["cmap"] == "cividis"


def test_a_magnitude_panel_leaves_the_zero_background_transparent() -> None:
    # Exact zeros outside the brain otherwise land on the ramp and draw a solid block
    # over the anatomy.
    img, mask = _magnitude_img()
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.magnitude_mosaic(img, mask_img=mask)
        except Exception:
            pass
    threshold = mock_plot.call_args.kwargs["threshold"]
    assert 0 < threshold < 1e-30


def test_a_magnitude_panel_declares_that_its_scale_is_unsigned() -> None:
    img, mask = _magnitude_img()
    figure = stat_maps.magnitude_mosaic(img, mask_img=mask)
    assert "not symmetric" in _provenance(figure)
    plt.close(figure)


def test_a_magnitude_panel_refuses_a_map_with_nothing_positive() -> None:
    empty = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="positive"):
        stat_maps.magnitude_mosaic(empty)


def test_the_magnitude_limit_comes_from_the_positive_values_only() -> None:
    from fmri_pipeline.analysis.report import style as style_mod

    img, mask = _magnitude_img(seed=3)
    data = np.asarray(img.get_fdata())
    positive = data[np.asanyarray(mask.dataobj).astype(bool) & (data > 0)]
    expected = style_mod.robust_upper_limit(positive)

    figure = stat_maps.magnitude_mosaic(img, mask_img=mask)
    reported = float(_provenance(figure).split("0–")[1].split(" ")[0])
    assert reported == pytest.approx(expected, rel=0.01)
    plt.close(figure)
