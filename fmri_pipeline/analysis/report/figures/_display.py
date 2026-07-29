"""Glue for nilearn display objects.

Nilearn's plotters return a slicer, not a figure, and different slicers expose the
underlying figure differently: ``OrthoSlicer`` carries only ``frame_axes``, while
others set ``figure`` or ``_fig``. Neither the colorbar nor the figure has a public
accessor, so every module that draws through nilearn needs the same two workarounds.
They live here once rather than being reimplemented per figure module.

:func:`crop_to_mask` is here for the same reason: nilearn chooses its slice positions
across the *background's* extent, so a whole-head T1w decides how much of each panel
is brain.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Anatomy kept around the analysis mask when cropping, in millimetres.
#:
#: Enough that the mask's own boundary is always drawn against tissue outside it --
#: which is what the coverage panel is read for, and what tells a reader whether a
#: missing region was dropout or was never acquired. A tight crop would put the
#: boundary on the edge of the frame and make an absent region look like a cropped one.
CROP_MARGIN_MM = 12.0


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


__all__ = ["CROP_MARGIN_MM", "crop_to_mask", "figure_of", "label_colorbar"]
