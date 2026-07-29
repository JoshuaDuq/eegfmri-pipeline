"""What a map panel says about itself, where saying it wrongly is invisible."""

from __future__ import annotations

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
