"""Where in the brain a result actually lives.

A map whose suprathreshold voxels sit disproportionately in white matter or the
ventricles is showing residual motion, a coregistration shift, or pulsatility -- and
it reaches the cluster table looking exactly like a result.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import tissue


SHAPE = (10, 10, 10)


def _codes() -> np.ndarray:
    """Grey, white, and CSF in three slabs."""
    codes = np.zeros(SHAPE, dtype=int)
    codes[:4] = 1  # GM
    codes[4:8] = 2  # WM
    codes[8:] = 3  # CSF
    return codes


def _map(gm: float = 3.0, wm: float = 0.0, csf: float = 0.0) -> nib.Nifti1Image:
    """A statistic map with a chosen mean offset per tissue slab."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(SHAPE).astype(np.float32)
    data[:4] += gm
    data[4:8] += wm
    data[8:] += csf
    return nib.Nifti1Image(data, np.eye(4))


# --- splitting -------------------------------------------------------------


def test_each_class_gets_its_own_voxels() -> None:
    slices = tissue.split_by_tissue(_map(), tissue_codes=_codes())
    assert [item.name for item in slices] == ["GM", "WM", "CSF"]
    assert [item.n_voxels for item in slices] == [400, 400, 200]


def test_the_analysis_mask_restricts_the_split() -> None:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[:2] = 1
    slices = tissue.split_by_tissue(
        _map(), tissue_codes=_codes(), mask_img=nib.Nifti1Image(mask, np.eye(4))
    )
    assert [item.name for item in slices] == ["GM"]
    assert slices[0].n_voxels == 200


def test_unclassified_voxels_are_not_assigned_to_a_class() -> None:
    codes = _codes()
    codes[:4] = 0
    slices = tissue.split_by_tissue(_map(), tissue_codes=codes)
    assert [item.name for item in slices] == ["WM", "CSF"]


def test_codes_of_the_wrong_shape_are_rejected() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        tissue.split_by_tissue(_map(), tissue_codes=np.zeros((4, 4, 4), dtype=int))


# --- enrichment ------------------------------------------------------------


def test_the_class_carrying_the_effect_has_the_highest_survival_rate() -> None:
    slices = tissue.split_by_tissue(_map(gm=4.0), tissue_codes=_codes())
    rates = dict(
        (name, share)
        for name, share, *_rest in tissue.enrichment(
            slices, threshold=2.3, two_sided=True
        )
    )
    assert rates["GM"] > rates["WM"]
    assert rates["GM"] > rates["CSF"]


def test_the_rate_is_within_class_not_a_raw_count() -> None:
    # Grey matter is the largest class in any brain mask, so a raw count puts it
    # first whatever the map does. Given the same effect in a class of 400 voxels
    # and one of 200, the counts differ twofold and the rates agree -- and the rate
    # is the quantity that answers "is this result grey matter".
    slices = tissue.split_by_tissue(_map(gm=4.0, csf=4.0), tissue_codes=_codes())
    measured = tissue.enrichment(slices, threshold=2.3, two_sided=True)
    rates = {name: share for name, share, *_rest in measured}
    counts = {name: survivors for name, _share, survivors, _total in measured}

    assert rates["GM"] == pytest.approx(rates["CSF"], abs=0.05)
    assert counts["GM"] > 1.5 * counts["CSF"]


def test_a_one_sided_contrast_ignores_the_negative_tail() -> None:
    slices = tissue.split_by_tissue(_map(gm=-4.0), tissue_codes=_codes())
    two_sided = tissue.enrichment(slices, threshold=2.3, two_sided=True)[0][1]
    one_sided = tissue.enrichment(slices, threshold=2.3, two_sided=False)[0][1]
    assert two_sided > one_sided


# --- the figure ------------------------------------------------------------


def test_the_panel_draws_a_distribution_and_a_survival_rate() -> None:
    slices = tissue.split_by_tissue(_map(), tissue_codes=_codes())
    figure = tissue.tissue_distribution_figure(slices, threshold=2.3)
    assert len(figure.axes) == 2
    assert "density" in figure.axes[0].get_ylabel()
    assert "above threshold" in figure.axes[1].get_ylabel()
    plt.close(figure)


def test_the_panel_reports_every_class_rate_not_just_grey_over_white() -> None:
    # Quoting one grey-to-white ratio hides the reading that matters most: on this
    # study's own contrast CSF survives at a higher rate than grey matter, which a
    # GM/WM ratio of 1.77 reports as healthy enrichment.
    slices = tissue.split_by_tissue(_map(gm=1.0, csf=4.0), tissue_codes=_codes())
    figure = tissue.tissue_distribution_figure(slices, threshold=2.3)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "survival rate by class" in text
    assert "CSF" in text and "GM" in text and "WM" in text
    plt.close(figure)


def test_the_panel_applies_no_criterion() -> None:
    slices = tissue.split_by_tissue(_map(), tissue_codes=_codes())
    figure = tissue.tissue_distribution_figure(slices, threshold=2.3)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "no criterion is applied" in text
    plt.close(figure)


def test_the_panel_names_where_the_segmentation_came_from() -> None:
    slices = tissue.split_by_tissue(_map(), tissue_codes=_codes())
    figure = tissue.tissue_distribution_figure(
        slices, threshold=2.3, tissue_source="probseg"
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "probseg" in text
    plt.close(figure)


def test_without_a_threshold_the_survival_panel_says_so() -> None:
    slices = tissue.split_by_tissue(_map(), tissue_codes=_codes())
    figure = tissue.tissue_distribution_figure(slices, threshold=None)
    text = " ".join(artist.get_text() for artist in figure.axes[1].texts)
    assert "nothing survives" in text
    plt.close(figure)


def test_an_empty_split_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one populated class"):
        tissue.tissue_distribution_figure([], threshold=2.3)


def _slices(spread: float = 0.09, n: int = 4_000) -> list[tissue.TissueSlice]:
    rng = np.random.default_rng(0)
    return [
        tissue.TissueSlice(name=name, values=rng.normal(0.0, spread, n))
        for name in ("GM", "WM", "CSF")
    ]


def test_the_density_axis_is_not_stretched_by_an_out_of_range_threshold() -> None:
    # The bins span the data's robust range; drawing the threshold with axvline let a
    # line carrying no data set the axis. Measured: bins over +-0.25 against an axis
    # reaching +-2.4, so all three densities occupied 8% of the panel width.
    figure = tissue.tissue_distribution_figure(_slices(), threshold=2.3, two_sided=True)
    try:
        low, high = figure.axes[0].get_xlim()
    finally:
        plt.close(figure)

    assert high < 1.0, f"axis reaches {high:.2f} for data inside +-0.4"
    assert low > -1.0


def test_an_out_of_range_threshold_is_still_named_on_the_panel() -> None:
    # Clipping the axis must not silently drop the threshold: a reader has to know the
    # rejection region lies beyond the drawn range.
    figure = tissue.tissue_distribution_figure(_slices(), threshold=2.3, two_sided=True)
    try:
        drawn = " ".join(
            text.get_text() for text in figure.axes[0].texts
        ) + " ".join(text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)

    assert "2.30" in drawn


def test_an_in_range_threshold_is_drawn_as_a_line() -> None:
    figure = tissue.tissue_distribution_figure(_slices(spread=2.0), threshold=2.3)
    try:
        positions = [
            float(line.get_xdata()[0])
            for line in figure.axes[0].lines
            if len(set(np.asarray(line.get_xdata(), dtype=float))) == 1
        ]
    finally:
        plt.close(figure)

    assert any(abs(p - 2.3) < 1e-6 for p in positions)


def test_an_all_zero_survival_panel_keeps_a_real_axis() -> None:
    # With nothing above threshold every bar is zero and the autoscaler invented a
    # +-0.04 axis around three zeros, which reads as measured precision.
    figure = tissue.tissue_distribution_figure(_slices(), threshold=2.3)
    try:
        bottom, top = figure.axes[1].get_ylim()
    finally:
        plt.close(figure)

    assert bottom == 0.0
    assert top >= 1.0
