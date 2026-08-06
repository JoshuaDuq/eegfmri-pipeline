"""Layout of the volume panels.

Nilearn's own mosaic drew the title on top of the first slice tile, spread its cuts
across the underlay rather than the mask, and parked a colourbar with clipped ticks at
the far right. The layout is owned here so those four failures are fixed in one place;
these tests are what keep them fixed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import _mosaic


def _mask(shape=(20, 24, 22), voxel=3.0) -> nib.Nifti1Image:
    """A mask that tapers at both ends of every axis, like a brain."""
    data = np.zeros(shape, dtype=np.uint8)
    centre = np.array(shape) / 2.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                offset = (np.array([i, j, k]) - centre) / centre
                if np.sum(offset**2) <= 1.0:
                    data[i, j, k] = 1
    return nib.Nifti1Image(data, np.diag([voxel, voxel, voxel, 1.0]))


def _record_rows():
    """A plot_row callable that records what the layout asked it to draw."""
    calls = []

    def plot_row(figure, rect, direction, cuts):
        calls.append({"rect": rect, "direction": direction, "cuts": list(cuts)})
        figure.add_axes(rect).set_axis_off()

    return plot_row, calls


# --- the grid -------------------------------------------------------------


def test_the_layout_draws_one_row_per_projection() -> None:
    plot_row, calls = _record_rows()
    figure = _mosaic.mosaic_figure(plot_row, reference_img=_mask(), mask_img=_mask())
    assert [call["direction"] for call in calls] == ["z", "y", "x"]
    plt.close(figure)


def test_rows_do_not_overlap_one_another() -> None:
    # Overlapping bands would draw one row's tiles over the previous row's labels.
    plot_row, calls = _record_rows()
    figure = _mosaic.mosaic_figure(plot_row, reference_img=_mask(), mask_img=_mask())
    bands = sorted((call["rect"][1], call["rect"][1] + call["rect"][3]) for call in calls)
    for (_low, high), (next_low, _next_high) in zip(bands, bands[1:]):
        assert next_low >= high
    plt.close(figure)


def test_every_row_gets_the_requested_number_of_cuts() -> None:
    plot_row, calls = _record_rows()
    figure = _mosaic.mosaic_figure(
        plot_row, reference_img=_mask(), mask_img=_mask(), n_cuts=5
    )
    assert all(len(call["cuts"]) == 5 for call in calls)
    plt.close(figure)


def test_a_row_that_will_not_draw_does_not_cost_the_others() -> None:
    # A panel missing its coronal band is still a panel; an exception here is a
    # section of the report that silently disappears.
    def plot_row(figure, rect, direction, cuts):
        if direction == "y":
            raise RuntimeError("this projection refuses to render")
        figure.add_axes(rect).set_axis_off()

    figure = _mosaic.mosaic_figure(plot_row, reference_img=_mask(), mask_img=_mask())
    texts = [artist.get_text() for artist in figure.texts]
    assert "Axial" in texts and "Sagittal" in texts
    assert "Coronal" not in texts
    plt.close(figure)


def test_no_row_directions_is_an_error() -> None:
    plot_row, _calls = _record_rows()
    with pytest.raises(ValueError, match="at least one row"):
        _mosaic.mosaic_figure(plot_row, reference_img=_mask(), directions=())


# --- annotation -----------------------------------------------------------


def test_the_title_sits_above_the_tiles_rather_than_on_them() -> None:
    # Nilearn draws the title inside the axes, where on a mosaic it lands on the
    # first slice.
    plot_row, calls = _record_rows()
    figure = _mosaic.mosaic_figure(
        plot_row, reference_img=_mask(), mask_img=_mask(), title="a contrast"
    )
    title = next(a for a in figure.texts if a.get_text() == "a contrast")
    highest_row_top = max(call["rect"][1] + call["rect"][3] for call in calls)
    assert title.get_position()[1] > highest_row_top
    plt.close(figure)


def test_each_tile_is_labelled_with_its_own_coordinate() -> None:
    plot_row, calls = _record_rows()
    figure = _mosaic.mosaic_figure(
        plot_row, reference_img=_mask(), mask_img=_mask(), n_cuts=4
    )
    texts = [artist.get_text() for artist in figure.texts]
    assert sum(text.startswith("z = ") for text in texts) == 4
    assert sum(text.startswith("y = ") for text in texts) == 4
    assert sum(text.startswith("x = ") for text in texts) == 4
    plt.close(figure)


def test_laterality_is_keyed_once_per_row_and_never_on_the_sagittal() -> None:
    # A sagittal tile has no left or right: both hemispheres project onto it.
    plot_row, _calls = _record_rows()
    figure = _mosaic.mosaic_figure(plot_row, reference_img=_mask(), mask_img=_mask())
    texts = [artist.get_text() for artist in figure.texts]
    assert texts.count("L") == 2 and texts.count("R") == 2
    plt.close(figure)


def test_the_radiological_convention_swaps_the_laterality_key() -> None:
    # A left/right error leaves no trace in the image, so the key has to follow the
    # convention the panel was actually drawn with.
    plot_row, _calls = _record_rows()
    positions = {}
    for radiological in (False, True):
        figure = _mosaic.mosaic_figure(
            plot_row,
            reference_img=_mask(),
            mask_img=_mask(),
            radiological=radiological,
        )
        left = next(a for a in figure.texts if a.get_text() == "L")
        positions[radiological] = left.get_position()[0]
        plt.close(figure)
    assert positions[False] < positions[True]


def test_provenance_is_written_below_the_tiles() -> None:
    plot_row, calls = _record_rows()
    figure = _mosaic.mosaic_figure(
        plot_row,
        reference_img=_mask(),
        mask_img=_mask(),
        provenance=["n = 10 voxels", "neurological"],
    )
    strip = next(a for a in figure.texts if "n = 10 voxels" in a.get_text())
    assert "neurological" in strip.get_text()
    assert strip.get_position()[1] < min(call["rect"][1] for call in calls)
    plt.close(figure)


# --- the colourbar --------------------------------------------------------


def test_the_colourbar_marks_the_band_the_panel_suppressed() -> None:
    plot_row, _calls = _record_rows()
    figure = _mosaic.mosaic_figure(
        plot_row,
        reference_img=_mask(),
        mask_img=_mask(),
        colorbar=_mosaic.ColorbarSpec(
            cmap="RdBu_r", vmin=-5.0, vmax=5.0, label="z", suppressed=(-2.3, 2.3)
        ),
    )
    hatched = [p for axes in figure.axes for p in axes.patches if p.get_hatch()]
    assert len(hatched) == 1
    # In data coordinates. A vertical ColorbarBase gives its axes a y range of
    # (vmin, vmax), and normalised fractions drew the band as a sliver around zero.
    low, high = hatched[0].get_xy()[1], hatched[0].get_xy()[1] + hatched[0].get_height()
    assert low == pytest.approx(-2.3, abs=0.05)
    assert high == pytest.approx(2.3, abs=0.05)
    plt.close(figure)


def test_a_colourbar_with_nothing_suppressed_is_unhatched() -> None:
    plot_row, _calls = _record_rows()
    figure = _mosaic.mosaic_figure(
        plot_row,
        reference_img=_mask(),
        mask_img=_mask(),
        colorbar=_mosaic.ColorbarSpec(cmap="cividis", vmin=0.0, vmax=100.0, label="tSNR"),
    )
    assert not [p for axes in figure.axes for p in axes.patches if p.get_hatch()]
    plt.close(figure)


def test_a_one_sided_panel_suppresses_everything_below_its_threshold() -> None:
    # apply_sidedness has already zeroed the negative half, so it is not merely
    # unthresholded -- it is absent.
    band = _mosaic.suppressed_band(2.3, two_sided=False)
    assert band == (-np.inf, 2.3)


def test_a_two_sided_panel_suppresses_a_band_straddling_zero() -> None:
    assert _mosaic.suppressed_band(2.3, two_sided=True) == (-2.3, 2.3)


def test_an_unthresholded_panel_suppresses_nothing() -> None:
    assert _mosaic.suppressed_band(None, two_sided=True) is None
    assert _mosaic.suppressed_band(0.0, two_sided=True) is None


def test_the_laterality_key_is_legible_over_a_dark_tile() -> None:
    # The key sits over the outermost tile rather than beside it, and those tiles are
    # mostly black. Drawn as plain text it was black on black and vanished from every
    # panel whose first slice reached the frame edge -- and a key a reader cannot see
    # is worse than none, since the panel still looks labelled.
    plot_row, _calls = _record_rows()
    figure = _mosaic.mosaic_figure(plot_row, reference_img=_mask(), mask_img=_mask())
    keys = [artist for artist in figure.texts if artist.get_text() in {"L", "R"}]
    assert keys
    assert all(key.get_path_effects() for key in keys)
    plt.close(figure)
