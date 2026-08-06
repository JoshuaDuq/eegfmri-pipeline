"""Layout for the report's volume panels.

Every volume panel in this report used to hand its layout to nilearn's
``display_mode="mosaic"`` or ``"ortho"``, which owns the title, the tile positions,
and the colourbar. Measured on this study's own output, that cost four things at once:

* the title is drawn *inside* the axes, so on a mosaic it landed on top of the first
  slice tile in every panel;
* cuts are spread across the underlay's extent, so tiles rendered at ``z = -71`` and
  ``y = 80`` -- outside the brain -- while the interesting slices got one tile each;
* the colourbar floats at the far right with its tick labels clipped by the figure
  edge, separated from the tiles by a band of dead canvas;
* ``ortho`` spends 40% of the canvas on nothing, which is what the tSNR and coverage
  panels were drawn with.

None of that is reachable through nilearn's parameters, so this module owns the
layout and calls nilearn once per row into an axes rectangle it controls. Nilearn
still performs the resampling, the overlay, and the orientation handling: a
left/right error there would be invisible in the output, and reimplementing it to win
a layout would be a poor trade.

The colourbar is drawn here rather than by nilearn so that a thresholded panel can
mark the band of values it suppressed. A reader otherwise has to take the caption's
word for which values are missing from the picture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.figures._display import cut_coords_for
from fmri_pipeline.analysis.report.style import GUIDE_COLOR

logger = logging.getLogger(__name__)

#: Row order, and what each projection is called.
#:
#: Axial first because it is the view a reader orients to fastest, sagittal last
#: because it is the one that carries the midline structures a summary reads for.
DEFAULT_DIRECTIONS: Tuple[Tuple[str, str], ...] = (
    ("z", "Axial"),
    ("y", "Coronal"),
    ("x", "Sagittal"),
)

#: Tiles per row. Seven spans a brain at roughly 20 mm between cuts on this study's
#: mask, which is close enough that a cluster cannot fall entirely between two tiles.
DEFAULT_CUTS_PER_ROW = 7


@dataclass(frozen=True)
class ColorbarSpec:
    """What a shared colourbar should show.

    ``suppressed`` is the ``(low, high)`` band a thresholded panel does not draw. It is
    marked on the bar rather than described in prose because the whole point of stating
    a threshold is to let a reader see which values are absent from the picture.
    """

    cmap: str
    vmin: float
    vmax: float
    label: str = ""
    suppressed: Optional[Tuple[float, float]] = None


#: Figure geometry, in figure fractions. Named rather than inlined because the
#: colourbar, the title, and the provenance strip all have to agree about where the
#: tile grid ends.
_LEFT = 0.045
_RIGHT = 0.885
_TOP = 0.905
_BOTTOM = 0.055
_TITLE_Y = 0.962
_PROVENANCE_Y = 0.016


def _row_label(figure: plt.Figure, y_centre: float, text: str) -> None:
    figure.text(
        _LEFT - 0.030,
        y_centre,
        text,
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
        color=GUIDE_COLOR,
    )


def _tile_labels(
    figure: plt.Figure, *, direction: str, cuts: Sequence[float], y_baseline: float
) -> None:
    """Write each tile's coordinate centred beneath it.

    Nilearn's own ``annotate`` writes the coordinate at each tile's bottom-left, where
    on a packed row it runs under the neighbouring tile -- visible in every mosaic this
    report produced. Centred beneath its own tile, the label cannot collide.
    """
    width = (_RIGHT - _LEFT) / max(len(cuts), 1)
    for index, cut in enumerate(cuts):
        figure.text(
            _LEFT + (index + 0.5) * width,
            y_baseline,
            f"{direction} = {cut:+.0f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=GUIDE_COLOR,
        )


def _laterality_key(figure: plt.Figure, *, y_top: float, radiological: bool) -> None:
    """Mark left and right once per row.

    Nilearn marks every tile, which on a seven-tile row is fourteen labels crowding
    the tile boundaries. The convention is a property of the whole row, so one key
    says the same thing with a thirteenth of the ink -- and the convention is named in
    full in the provenance strip regardless, because a left/right error leaves no
    trace in the image.

    Haloed, because the key sits over the outermost tile rather than beside it and
    those tiles are mostly black: drawn as plain text it was black on black and
    vanished from every panel whose first slice reached the frame edge. A left/right
    key a reader cannot see is worse than none, since the panel still looks labelled.
    """
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    left_text, right_text = ("R", "L") if radiological else ("L", "R")
    for x, text, align in (
        (_LEFT + 0.004, left_text, "left"),
        (_RIGHT - 0.004, right_text, "right"),
    ):
        figure.text(
            x,
            y_top,
            text,
            ha=align,
            va="top",
            fontsize=7.5,
            color="#111111",
            path_effects=halo,
        )


def draw_colorbar(
    figure: plt.Figure,
    spec: ColorbarSpec,
    rect: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Draw a colourbar, marking any band the panel suppressed.

    Public because the glass brain needs the same bar and the same hatched band, and
    it is not a mosaic: nilearn's own colourbar is what leaves the ticks clipped
    against the figure edge on both.
    """
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    if rect is None:
        height = (_TOP - _BOTTOM) * 0.52
        rect = (
            _RIGHT + 0.030,
            _BOTTOM + (_TOP - _BOTTOM - height) / 2.0,
            0.015,
            height,
        )
    axis = figure.add_axes(rect)
    norm = Normalize(vmin=spec.vmin, vmax=spec.vmax)
    bar = ColorbarBase(
        axis, cmap=plt.get_cmap(spec.cmap), norm=norm, orientation="vertical"
    )
    if spec.label:
        bar.set_label(spec.label, rotation=90, labelpad=5, fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_linewidth(0.6)

    if spec.suppressed is not None:
        low, high = spec.suppressed
        # In data coordinates. A vertical ColorbarBase gives its axes a y range of
        # (vmin, vmax), not (0, 1) -- normalised fractions drew the band as a sliver
        # around zero, understating what a threshold of 2.3 on a +-5.55 scale hides.
        bottom, top = axis.get_ylim()
        span_low = max(float(low), min(bottom, top))
        span_high = min(float(high), max(bottom, top))
        if span_high > span_low:
            # Hatched rather than blanked: the colours in this band are real, the
            # panel simply does not draw voxels that fall in it.
            axis.axhspan(
                span_low,
                span_high,
                facecolor="none",
                edgecolor="0.2",
                hatch="///",
                linewidth=0.0,
                zorder=5,
            )
            # Left of the bar. The ticks and the unit label are both on the right,
            # where this ran straight through them.
            axis.annotate(
                "not drawn",
                xy=(0.0, (span_low + span_high) / 2.0),
                xycoords=("axes fraction", "data"),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=6,
                color=GUIDE_COLOR,
                va="center",
                ha="center",
                rotation=90,
            )


def mosaic_figure(
    plot_row: Callable[[plt.Figure, Tuple[float, float, float, float], str, Sequence[float]], None],
    *,
    reference_img: Any,
    mask_img: Any = None,
    directions: Sequence[Tuple[str, str]] = DEFAULT_DIRECTIONS,
    n_cuts: int = DEFAULT_CUTS_PER_ROW,
    title: str = "",
    provenance: Sequence[str] = (),
    colorbar: Optional[ColorbarSpec] = None,
    radiological: bool = False,
    figwidth: float = 11.0,
    row_height: float = 1.72,
) -> plt.Figure:
    """Lay out one volume panel and return its figure.

    ``plot_row`` is called once per row with ``(figure, axes_rect, direction, cuts)``
    and is expected to draw into ``axes_rect`` -- in practice by passing ``figure=``
    and ``axes=`` through to a nilearn plotter. Everything around the tiles, including
    the title and the colourbar, belongs to this function.

    Splitting it this way is what lets the statistic, magnitude, dual-coded, and mask
    panels share one layout while each keeps its own colour rules: the four differ
    only in what they draw inside a tile.
    """
    rows = list(directions)
    if not rows:
        raise ValueError("A mosaic needs at least one row direction.")

    figure = plt.figure(figsize=(figwidth, row_height * len(rows) + 0.95))
    figure.patch.set_facecolor("white")
    band = (_TOP - _BOTTOM) / len(rows)

    for index, (direction, name) in enumerate(rows):
        y0 = _TOP - (index + 1) * band
        cuts = cut_coords_for(reference_img, direction, n_cuts, mask_img=mask_img)
        rect = (_LEFT, y0 + 0.030, _RIGHT - _LEFT, band - 0.042)
        try:
            plot_row(figure, rect, direction, cuts)
        except Exception as exc:
            # One row that will not render must not cost the other two. A panel
            # missing its coronal band is still a panel; an exception here is a
            # section of the report that silently disappears.
            logger.warning("Could not draw the %s row of a mosaic (%s)", direction, exc)
            continue
        _row_label(figure, y0 + band / 2.0, name)
        _tile_labels(figure, direction=direction, cuts=cuts, y_baseline=y0 + 0.010)
        if direction != "x":
            # A sagittal tile has no left or right: both hemispheres project onto it.
            _laterality_key(
                figure, y_top=y0 + band - 0.028, radiological=radiological
            )

    if colorbar is not None:
        draw_colorbar(figure, colorbar)

    if title:
        figure.text(
            _LEFT - 0.008,
            _TITLE_Y,
            title,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    if provenance:
        # Placed inside the canvas rather than through annotate_provenance's
        # below-canvas strip: this figure sets its own axes rectangles, so there is
        # already a reserved band and no tight bounding box to grow into it.
        figure.text(
            _LEFT - 0.008,
            _PROVENANCE_Y,
            "  ·  ".join(str(line) for line in provenance if line),
            ha="left",
            va="center",
            fontsize=6.5,
            color=GUIDE_COLOR,
        )
    return figure


def suppressed_band(
    threshold: Optional[float], *, two_sided: bool
) -> Optional[Tuple[float, float]]:
    """The colour range a thresholded panel does not draw.

    Two-sided inference hides a band straddling zero; one-sided inference hides
    everything below the threshold, negative half included, because
    :func:`~fmri_pipeline.analysis.report.figures.stat_maps.apply_sidedness` has
    already zeroed it.
    """
    if not threshold or threshold <= 0:
        return None
    value = float(threshold)
    return (-value, value) if two_sided else (-np.inf, value)


__all__ = [
    "DEFAULT_CUTS_PER_ROW",
    "DEFAULT_DIRECTIONS",
    "ColorbarSpec",
    "draw_colorbar",
    "mosaic_figure",
    "suppressed_band",
]
