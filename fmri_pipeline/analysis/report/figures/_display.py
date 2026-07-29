"""Glue for nilearn display objects.

Nilearn's plotters return a slicer, not a figure, and different slicers expose the
underlying figure differently: ``OrthoSlicer`` carries only ``frame_axes``, while
others set ``figure`` or ``_fig``. Neither the colorbar nor the figure has a public
accessor, so every module that draws through nilearn needs the same two workarounds.
They live here once rather than being reimplemented per figure module.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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


__all__ = ["figure_of", "label_colorbar"]
