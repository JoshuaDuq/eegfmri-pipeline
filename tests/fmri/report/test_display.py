"""The underlay decides how much of each panel is brain.

Nilearn chooses its slice positions across the extent of the image it draws over, so
an untrimmed whole-head T1w spends every mosaic on anatomy the model never saw. On
this study the underlay spanned 176 x 256 x 256 mm against a 135 x 165 x 138 mm
analysis mask.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from fmri_pipeline.analysis.report.figures import _display


def _head(shape=(200, 200, 200), voxel=1.0):
    """A whole-head underlay on a 1 mm grid, larger than the mask in every direction."""
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    return nib.Nifti1Image(data, np.diag([voxel, voxel, voxel, 1.0]))


def _mask(shape=(20, 25, 25), voxel=4.0, origin=(40.0, 40.0, 40.0)):
    """A functional-resolution mask sitting well inside the head, in world coordinates.

    Spans 4 x (n - 1) mm per axis between the outermost voxel centres: 76 x 96 x 96 mm.
    """
    affine = np.diag([voxel, voxel, voxel, 1.0])
    affine[:3, 3] = origin
    return nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine)


# --- the crop -------------------------------------------------------------


def test_the_underlay_is_trimmed_to_the_mask() -> None:
    head, mask = _head(), _mask()
    cropped = _display.crop_to_mask(head, mask)
    assert np.prod(cropped.shape[:3]) < np.prod(head.shape[:3])


def test_the_crop_keeps_the_mask_entirely_inside_the_frame() -> None:
    # A mask boundary on the edge of the panel cannot be told from a mask that was
    # cut off by the crop, which is the whole reading of the coverage figure.
    head, mask = _head(), _mask()
    cropped = _display.crop_to_mask(head, mask)

    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (0, mask.shape[0] - 1)
            for y in (0, mask.shape[1] - 1)
            for z in (0, mask.shape[2] - 1)
        ]
    )
    world = corners @ np.asarray(mask.affine).T
    in_cropped = world @ np.linalg.inv(np.asarray(cropped.affine)).T
    assert (in_cropped[:, :3] >= 0).all()
    assert (in_cropped[:, :3] <= np.array(cropped.shape[:3]) - 1).all()


def test_the_crop_keeps_a_margin_of_anatomy_outside_the_mask() -> None:
    # Tissue outside the mask is what says a missing region was dropout rather than a
    # region the picture stops short of.
    head, mask = _head(), _mask()
    cropped = _display.crop_to_mask(head, mask, margin_mm=12.0)
    # The mask spans 76 mm in x between its outermost voxel centres; on a 1 mm underlay
    # the crop must exceed that by the margin on both sides.
    assert cropped.shape[0] >= 76 + 2 * 12


def test_a_larger_margin_keeps_more_anatomy() -> None:
    head, mask = _head(), _mask()
    tight = _display.crop_to_mask(head, mask, margin_mm=2.0)
    loose = _display.crop_to_mask(head, mask, margin_mm=20.0)
    assert np.prod(loose.shape[:3]) > np.prod(tight.shape[:3])


def test_the_crop_does_not_resample_the_anatomy() -> None:
    # Exact voxels, taken as a slice. Interpolating an underlay to fit a box would
    # blur the anatomy a cluster is being located against.
    head, mask = _head(), _mask()
    cropped = _display.crop_to_mask(head, mask)
    original = np.asarray(head.dataobj)
    kept = np.asarray(cropped.dataobj)
    assert set(np.unique(kept)) <= set(np.unique(original))
    voxel_of = lambda img: np.sqrt((np.asarray(img.affine)[:3, :3] ** 2).sum(axis=0))
    np.testing.assert_allclose(voxel_of(cropped), voxel_of(head))


def test_the_crop_preserves_world_coordinates() -> None:
    # A slice that did not carry its affine would move every cluster in the report.
    head, mask = _head(), _mask()
    cropped = _display.crop_to_mask(head, mask)
    original = np.asarray(head.dataobj)
    kept = np.asarray(cropped.dataobj)

    index = np.array([5, 6, 7, 1.0])
    world = index @ np.asarray(cropped.affine).T
    back = world @ np.linalg.inv(np.asarray(head.affine)).T
    i, j, k = np.rint(back[:3]).astype(int)
    assert kept[5, 6, 7] == pytest.approx(original[i, j, k])


def test_a_rotated_underlay_is_cropped_around_the_whole_mask() -> None:
    # Two opposite corners are not the extremes once the affines differ in
    # orientation, so all eight are mapped.
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    head = nib.Nifti1Image(
        np.random.default_rng(1).random((200, 200, 200)).astype(np.float32), rotation
    )
    # World coordinates chosen so all eight mask corners land inside the rotated head,
    # with room for the margin: the crop can only keep anatomy that exists.
    mask = _mask(origin=(10.0, 85.0, 52.0))
    cropped = _display.crop_to_mask(head, mask)

    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (0, mask.shape[0] - 1)
            for y in (0, mask.shape[1] - 1)
            for z in (0, mask.shape[2] - 1)
        ]
    )
    in_cropped = (
        corners @ np.asarray(mask.affine).T @ np.linalg.inv(np.asarray(cropped.affine)).T
    )
    assert (in_cropped[:, :3] >= 0).all()
    assert (in_cropped[:, :3] <= np.array(cropped.shape[:3]) - 1).all()


# --- declining to crop ----------------------------------------------------


def test_no_mask_leaves_the_underlay_alone() -> None:
    head = _head()
    assert _display.crop_to_mask(head, None) is head


def test_no_underlay_stays_absent() -> None:
    assert _display.crop_to_mask(None, _mask()) is None


def test_an_empty_mask_leaves_the_underlay_alone() -> None:
    # Nothing to crop to. A box computed from no voxels would be degenerate.
    head = _head()
    empty = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4))
    assert _display.crop_to_mask(head, empty) is head


def test_a_mask_outside_the_underlay_costs_the_crop_and_not_the_panel() -> None:
    # A smaller panel is not worth an exception on the figure carrying the result.
    head = _head()
    far = _mask(origin=(5_000.0, 5_000.0, 5_000.0))
    cropped = _display.crop_to_mask(head, far)
    assert cropped is head or np.prod(cropped.shape[:3]) > 0


def test_a_mask_covering_the_whole_underlay_changes_nothing_material() -> None:
    head = _head(shape=(40, 40, 40))
    whole = nib.Nifti1Image(np.ones((40, 40, 40), dtype=np.uint8), np.eye(4))
    cropped = _display.crop_to_mask(head, whole)
    assert cropped.shape[:3] == head.shape[:3]


# --- the report underlay --------------------------------------------------
#
# This study's T1w is 10.7 degrees oblique against an axis-aligned analysis mask.
# Nilearn draws in world coordinates, so the head rendered tilted and every tile lost
# its corners to black wedges.


def _oblique_head(degrees: float = 10.7, shape=(180, 200, 200)) -> nib.Nifti1Image:
    angle = np.deg2rad(degrees)
    affine = np.eye(4)
    affine[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    affine[:3, 3] = [-90.0, -100.0, -100.0]
    data = np.random.default_rng(3).random(shape).astype(np.float32)
    return nib.Nifti1Image(data, affine)


def _centred_mask(voxel=3.0, shape=(30, 34, 30)) -> nib.Nifti1Image:
    affine = np.diag([voxel, voxel, voxel, 1.0])
    affine[:3, 3] = -voxel * np.array(shape) / 2.0
    return nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine)


def test_the_report_underlay_is_axis_aligned() -> None:
    # The whole point: an oblique grid is what tilts the head and wedges the tiles.
    aligned = _display.report_underlay(_oblique_head(), _centred_mask())
    rotation = np.asarray(aligned.affine)[:3, :3]
    off_diagonal = rotation - np.diag(np.diag(rotation))
    assert np.allclose(off_diagonal, 0.0, atol=1e-9)


def test_the_report_underlay_is_bounded_by_the_mask() -> None:
    head, mask = _oblique_head(), _centred_mask()
    aligned = _display.report_underlay(head, mask)
    assert np.prod(aligned.shape[:3]) < np.prod(head.shape[:3])


def test_the_report_underlay_keeps_the_whole_mask_in_frame() -> None:
    # A mask boundary on the edge of the panel cannot be told from one the crop cut
    # off, which is the whole reading of the coverage figure.
    head, mask = _oblique_head(), _centred_mask()
    aligned = _display.report_underlay(head, mask, margin_mm=12.0)
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (0, mask.shape[0] - 1)
            for y in (0, mask.shape[1] - 1)
            for z in (0, mask.shape[2] - 1)
        ]
    )
    in_underlay = (
        corners @ np.asarray(mask.affine).T @ np.linalg.inv(np.asarray(aligned.affine)).T
    )
    assert (in_underlay[:, :3] >= 0).all()
    assert (in_underlay[:, :3] <= np.array(aligned.shape[:3]) - 1).all()


def test_the_report_underlay_does_not_upsample_the_anatomy() -> None:
    # Interpolating a 1 mm T1w onto a finer grid invents detail it does not have.
    aligned = _display.report_underlay(_oblique_head(), _centred_mask())
    voxel = np.sqrt((np.asarray(aligned.affine)[:3, :3] ** 2).sum(axis=0))
    assert np.allclose(voxel, 1.0)


def test_a_larger_margin_keeps_more_anatomy_in_the_underlay() -> None:
    head, mask = _oblique_head(), _centred_mask()
    tight = _display.report_underlay(head, mask, margin_mm=2.0)
    loose = _display.report_underlay(head, mask, margin_mm=20.0)
    assert np.prod(loose.shape[:3]) > np.prod(tight.shape[:3])


def test_no_mask_leaves_the_underlay_alone_for_the_report_too() -> None:
    head = _oblique_head()
    assert _display.report_underlay(head, None) is head
    assert _display.report_underlay(None, _centred_mask()) is None


def test_an_unusable_mask_falls_back_to_a_crop_rather_than_failing() -> None:
    # A tilted panel is worth more than a missing one.
    head = _oblique_head()
    empty = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4))
    assert _display.report_underlay(head, empty) is head


# --- choosing cuts --------------------------------------------------------


def _tapered_mask(shape=(24, 24, 24), voxel=3.0) -> nib.Nifti1Image:
    """A sphere: full in the middle, tapering to a speck at both ends of every axis."""
    grid = np.indices(shape).astype(float)
    centre = (np.array(shape) - 1) / 2.0
    radius = np.sqrt(sum((grid[i] - centre[i]) ** 2 for i in range(3)))
    data = (radius <= centre.min()).astype(np.uint8)
    affine = np.diag([voxel, voxel, voxel, 1.0])
    affine[:3, 3] = -voxel * centre
    return nib.Nifti1Image(data, affine)


def test_cuts_avoid_the_slices_where_the_mask_has_tapered_away() -> None:
    # Spanning the raw extent spent the end tiles on under 6% of the peak in-plane
    # area, which rendered as specks: a seventh of the panel on nothing readable.
    mask = _tapered_mask()
    data = np.asanyarray(mask.dataobj).astype(bool)
    cuts = _display.mask_cut_coords(mask, "z", 7)

    inverse = np.linalg.inv(np.asarray(mask.affine))
    areas = data.sum(axis=(0, 1))
    for cut in cuts:
        index = int(round((np.array([0.0, 0.0, cut, 1.0]) @ inverse.T)[2]))
        assert areas[index] >= 0.25 * areas.max()


def test_cuts_span_the_mask_rather_than_clustering_on_the_signal() -> None:
    # Even spacing, not nilearn's find_cut_slices: positions chosen from the data
    # make an absence of effect unshowable, because no tile is ever spent where
    # nothing happened.
    cuts = _display.mask_cut_coords(_tapered_mask(), "z", 7)
    gaps = np.diff(cuts)
    assert np.allclose(gaps, gaps[0])


def test_cuts_are_returned_in_world_coordinates() -> None:
    mask = _tapered_mask(voxel=3.0)
    cuts = _display.mask_cut_coords(mask, "x", 5)
    low, high = min(cuts), max(cuts)
    # The sphere is centred on the origin, so its cuts straddle it.
    assert low < 0.0 < high


def test_an_empty_mask_cannot_choose_cuts() -> None:
    empty = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="no voxels"):
        _display.mask_cut_coords(empty, "z", 5)


def test_an_unknown_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="x, y, z"):
        _display.mask_cut_coords(_tapered_mask(), "a", 5)


def test_without_a_mask_cuts_come_from_the_image_extent() -> None:
    head = _head(shape=(40, 40, 40))
    cuts = _display.cut_coords_for(head, "z", 5, mask_img=None)
    assert len(cuts) == 5
    assert min(cuts) > 0.0  # the outermost slices of an acquisition box are air


def test_an_unusable_mask_falls_back_to_the_image_extent() -> None:
    head = _head(shape=(40, 40, 40))
    empty = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4))
    assert len(_display.cut_coords_for(head, "z", 5, mask_img=empty)) == 5
