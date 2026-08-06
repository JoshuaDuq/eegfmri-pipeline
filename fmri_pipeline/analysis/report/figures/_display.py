"""Glue for nilearn display objects.

Nilearn's plotters return a slicer, not a figure, and different slicers expose the
underlying figure differently: ``OrthoSlicer`` carries only ``frame_axes``, while
others set ``figure`` or ``_fig``. Neither the colorbar nor the figure has a public
accessor, so every module that draws through nilearn needs the same two workarounds.
They live here once rather than being reimplemented per figure module.

:func:`report_underlay` and :func:`crop_to_mask` are here for the same reason:
nilearn chooses its slice positions across the *background's* extent, so a whole-head
T1w decides how much of each panel is brain.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Anatomy kept around the analysis mask when cropping, in millimetres.
#:
#: Enough that the mask's own boundary is always drawn against tissue outside it --
#: which is what the coverage panel is read for, and what tells a reader whether a
#: missing region was dropout or was never acquired. A tight crop would put the
#: boundary on the edge of the frame and make an absent region look like a cropped one.
CROP_MARGIN_MM = 12.0

#: Smallest share of the fullest slice's in-plane mask area that still earns a tile.
#:
#: Cuts spread across the mask's raw extent spend the outermost tiles on the few
#: voxels where the mask tapers away. Measured on this study's own mask, the end tiles
#: held under 6% of the peak in-plane area and rendered as specks -- a seventh of the
#: panel spent on nothing a reader can read. Selecting on area rather than on position
#: is what keeps every tile a slice worth looking at.
MIN_SLICE_AREA_FRACTION = 0.25

_AXIS_OF = {"x": 0, "y": 1, "z": 2}


def _world_bounds(img: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the world-coordinate corners spanned by ``img``'s voxel grid.

    All eight corners are mapped, not two opposite ones: under a rotated affine the
    box's extremes are not the corners you started from.
    """
    shape = np.asarray(img.shape[:3], dtype=float) - 1.0
    corners = np.array(
        [
            [shape[0] if a else 0.0, shape[1] if b else 0.0, shape[2] if c else 0.0]
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        ]
    )
    world = np.column_stack([corners, np.ones(len(corners))]) @ np.asarray(img.affine).T
    return world[:, :3].min(axis=0), world[:, :3].max(axis=0)


def _mask_world_bounds(mask_img: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the world-coordinate box enclosing the non-zero voxels of ``mask_img``."""
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    if not mask.any():
        raise ValueError("The mask has no voxels; no bounding box can be computed.")
    indices = np.array(np.nonzero(mask), dtype=float)
    low, high = indices.min(axis=1), indices.max(axis=1)
    corners = np.array(
        [
            [low[0] if a else high[0], low[1] if b else high[1], low[2] if c else high[2]]
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        ]
    )
    world = np.column_stack([corners, np.ones(len(corners))]) @ np.asarray(
        mask_img.affine
    ).T
    return world[:, :3].min(axis=0), world[:, :3].max(axis=0)


def report_underlay(
    background: Any,
    mask_img: Any,
    *,
    margin_mm: float = CROP_MARGIN_MM,
    resolution_mm: Optional[float] = None,
) -> Any:
    """Put the anatomical underlay on an axis-aligned grid enclosing the mask.

    Two separate problems, one resample.

    *Obliquity.* This study's T1w carries a 10.7 degree oblique affine while the
    analysis mask and every statistic map are axis-aligned at 3 mm. Nilearn draws the
    underlay in world coordinates, so the head rendered visibly tilted and each tile
    lost its corners to black wedges. Resampling onto a diagonal grid derived from the
    mask puts the anatomy in the same frame the model was fitted in, which is the frame
    every coordinate in this report is quoted in.

    *Extent.* Nilearn picks slice positions across the extent of the image it draws
    over. Measured here, the underlay spanned 176 x 256 x 256 mm against a mask of
    135 x 165 x 138 mm, so slices ran from the vertex through the neck. Deriving the
    target box from the mask is what stops the panel spending tiles on anatomy the
    model never saw: 176 x 256 x 256 becomes 145 x 175 x 148.

    This adds no interpolation. Nilearn's own ``_show_im`` calls ``reorder_img`` with
    continuous interpolation on any non-diagonal affine, so an oblique underlay is
    already resampled once before it is drawn; doing it here takes control of the
    bounding box rather than adding a pass. The grid keeps the underlay's own finest
    voxel size unless ``resolution_mm`` says otherwise, so the anatomy is never
    upsampled into detail it does not have.

    ``margin_mm`` keeps tissue outside the mask in frame, which is what lets the
    coverage panel show that a region fell outside the model rather than outside the
    picture.

    Returns ``background`` unchanged when there is nothing to align to, and falls back
    to :func:`crop_to_mask` if the resample fails -- a tilted panel is worth more than
    a missing one.
    """
    if background is None or mask_img is None:
        return background

    try:
        from nilearn import image as nilearn_image

        low, high = _mask_world_bounds(mask_img)
        low = low - float(margin_mm)
        high = high + float(margin_mm)

        voxel_sizes = np.sqrt((np.asarray(background.affine)[:3, :3] ** 2).sum(axis=0))
        spacing = (
            float(resolution_mm)
            if resolution_mm is not None
            else float(np.min(voxel_sizes[voxel_sizes > 0]))
        )
        if not np.isfinite(spacing) or spacing <= 0:
            return crop_to_mask(background, mask_img, margin_mm=margin_mm)

        target_affine = np.eye(4)
        target_affine[:3, :3] = np.diag([spacing, spacing, spacing])
        target_affine[:3, 3] = low
        target_shape = tuple(
            int(n) for n in np.ceil((high - low) / spacing).astype(int) + 1
        )
        if min(target_shape) < 2:
            return crop_to_mask(background, mask_img, margin_mm=margin_mm)

        # Continuous interpolation on a binary image invents intermediate values,
        # which on a mask used as an underlay renders a soft halo where there are
        # only two states. Nilearn warns about exactly this; nearest is correct there.
        interpolation = "continuous"
        try:
            from nilearn._utils.niimg import is_binary_niimg

            if is_binary_niimg(background):
                interpolation = "nearest"
        except Exception:
            logger.debug("Could not test the underlay for binarity; assuming continuous.")

        return nilearn_image.resample_img(
            background,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True,
        )
    except Exception as exc:  # pragma: no cover - depends on a malformed affine
        logger.info(
            "Could not align the underlay to the analysis mask (%s); cropping instead.",
            exc,
        )
        return crop_to_mask(background, mask_img, margin_mm=margin_mm)


def mask_cut_coords(
    mask_img: Any,
    direction: str,
    n_cuts: int,
    *,
    min_area_fraction: float = MIN_SLICE_AREA_FRACTION,
) -> List[float]:
    """Return ``n_cuts`` world coordinates spanning the slices that carry brain.

    Evenly spaced across the slices holding at least ``min_area_fraction`` of the
    fullest slice's in-plane mask area, rather than across the mask's raw extent --
    see :data:`MIN_SLICE_AREA_FRACTION` for what the difference costs.

    Even spacing rather than nilearn's ``find_cut_slices``, which places cuts where the
    map is most active. That is the right choice for finding a result and the wrong one
    for reporting it: slice positions chosen from the data make an absence of effect
    unshowable, because no tile is ever spent where nothing happened.
    """
    if direction not in _AXIS_OF:
        raise ValueError(f"Direction must be one of x, y, z; got {direction!r}.")
    if n_cuts < 1:
        raise ValueError(f"n_cuts must be at least 1, got {n_cuts}.")

    axis = _AXIS_OF[direction]
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    area = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    if not area.any():
        raise ValueError("The mask has no voxels; no cut positions can be chosen.")

    keep = np.nonzero(area >= float(min_area_fraction) * area.max())[0]
    low, high = float(keep.min()), float(keep.max())

    affine = np.asarray(mask_img.affine)
    coords: List[float] = []
    for position in np.linspace(low, high, n_cuts):
        voxel = np.zeros(3)
        voxel[axis] = position
        coords.append(float((np.append(voxel, 1.0) @ affine.T)[axis]))
    return coords


def fallback_cut_coords(img: Any, direction: str, n_cuts: int) -> List[float]:
    """Cut positions spanning an image's own extent, for when there is no mask.

    The middle 80% of the image, because the outermost slices of any acquisition box
    are air. Cruder than :func:`mask_cut_coords` and only reached when the manifest
    records no analysis mask.
    """
    if direction not in _AXIS_OF:
        raise ValueError(f"Direction must be one of x, y, z; got {direction!r}.")
    axis = _AXIS_OF[direction]
    low, high = _world_bounds(img)
    span = high[axis] - low[axis]
    return [
        float(c)
        for c in np.linspace(low[axis] + 0.1 * span, high[axis] - 0.1 * span, n_cuts)
    ]


def cut_coords_for(
    img: Any,
    direction: str,
    n_cuts: int,
    *,
    mask_img: Any = None,
    min_area_fraction: float = MIN_SLICE_AREA_FRACTION,
) -> List[float]:
    """Cut positions from the mask when there is one, from the image otherwise."""
    if mask_img is not None:
        try:
            return mask_cut_coords(
                mask_img, direction, n_cuts, min_area_fraction=min_area_fraction
            )
        except Exception as exc:
            logger.info(
                "Could not choose cuts from the mask (%s); using the image extent.", exc
            )
    return fallback_cut_coords(img, direction, n_cuts)


def figure_of(display: Any) -> Any:
    """Return the Matplotlib figure behind a nilearn display.

    Tries every attribute nilearn's slicers use, because which one exists depends on
    the display class: ``plot_img``'s ``OrthoSlicer`` exposes ``frame_axes`` alone.
    """
    figure = getattr(display, "figure", None) or getattr(display, "_fig", None)
    if figure is None:
        frame_axes = getattr(display, "frame_axes", None)
        figure = getattr(frame_axes, "figure", None)
    if figure is None:
        raise RuntimeError(
            "Could not resolve a Matplotlib figure from the nilearn display "
            f"{type(display).__name__}."
        )
    return figure


def label_colorbar(display: Any, label: str) -> None:
    """Name the units on a nilearn colorbar.

    Nilearn exposes no parameter for this, so the private attribute is the only
    route. Degrades to a debug log rather than failing: an unlabelled colorbar is a
    worse figure, not a broken pipeline.
    """
    colorbar = getattr(display, "_cbar", None)
    if colorbar is None:
        logger.debug(
            "Nilearn display %s exposed no colorbar to label.", type(display).__name__
        )
        return
    colorbar.set_label(label, rotation=90, labelpad=6)


def crop_to_mask(
    background: Any, mask_img: Any, *, margin_mm: float = CROP_MARGIN_MM
) -> Any:
    """Trim an anatomical underlay to the analysis mask's extent, plus a margin.

    Nilearn picks its slice positions across the extent of the image it is drawing
    over, so a whole-head T1w spends the panel on anatomy the model never saw.
    Measured on this study: the underlay spans 176 x 256 x 256 mm against a mask of
    135 x 165 x 138 mm, so slices ran from the vertex through the neck and the brain
    occupied barely half of each tile in the through-plane direction. Nothing below
    the cerebellum was tested, and nothing there was drawn on -- the panels simply
    rendered the neck at the same resolution as the cortex.

    Trimming is exact, not resampling: the bounding box is mapped through the two
    affines and taken as a slice, so no voxel value is interpolated and the anatomy is
    the anatomy. ``margin_mm`` keeps tissue outside the mask in frame, which is what
    lets the coverage panel show that a region fell outside the model rather than
    outside the picture.

    Returns ``background`` unchanged when there is nothing to crop to, or when the
    computed box is degenerate -- a smaller panel is not worth an exception on the
    figure that carries the result.
    """
    if background is None or mask_img is None:
        return background

    try:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)
        if not mask.any():
            return background

        indices = np.array(np.nonzero(mask), dtype=float)
        low = indices.min(axis=1)
        high = indices.max(axis=1)

        # Every corner of the mask's voxel box, in world coordinates. Taking only the
        # two opposite corners is wrong the moment the two images differ in
        # orientation: a rotated affine sends the box's extremes to corners that are
        # not the ones you started from.
        corners = np.array(
            [
                [low[0] if a else high[0], low[1] if b else high[1], low[2] if c else high[2]]
                for a in (0, 1)
                for b in (0, 1)
                for c in (0, 1)
            ]
        )
        world = np.column_stack([corners, np.ones(len(corners))]) @ np.asarray(
            mask_img.affine
        ).T

        inverse = np.linalg.inv(np.asarray(background.affine))
        in_background = world @ inverse.T
        box_low = in_background[:, :3].min(axis=0)
        box_high = in_background[:, :3].max(axis=0)

        voxel_sizes = np.sqrt((np.asarray(background.affine)[:3, :3] ** 2).sum(axis=0))
        pad = float(margin_mm) / np.where(voxel_sizes > 0, voxel_sizes, 1.0)

        shape = np.asarray(background.shape[:3])
        start = np.clip(np.floor(box_low - pad).astype(int), 0, shape - 1)
        stop = np.clip(np.ceil(box_high + pad).astype(int) + 1, 1, shape)
        if np.any(stop - start < 2):
            return background

        return background.slicer[
            start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]
        ]
    except Exception as exc:  # pragma: no cover - depends on a malformed affine
        logger.info("Could not crop the underlay to the analysis mask (%s)", exc)
        return background


__all__ = [
    "CROP_MARGIN_MM",
    "MIN_SLICE_AREA_FRACTION",
    "crop_to_mask",
    "cut_coords_for",
    "fallback_cut_coords",
    "figure_of",
    "label_colorbar",
    "mask_cut_coords",
    "report_underlay",
]
