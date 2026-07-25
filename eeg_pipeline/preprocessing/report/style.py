"""Colour, colormap, and scaling conventions shared by every report figure.

Every module that appends figures to the subject HTML report renders through these
constants so that one colour means one thing across the whole document.

Conventions
-----------
Categorical colours come from the Okabe-Ito palette, which stays distinguishable
under the common forms of colour vision deficiency.

Zero-centred power in decibels is drawn with a diverging, colour-vision-safe
colormap on a symmetric scale, so that the neutral colour always marks "no change
from baseline". Sequential rainbow colormaps such as ``jet`` and ``turbo`` are not
used: their non-monotonic lightness introduces visual boundaries that do not exist
in the data.

Embedding format follows the dominant content. Report figures are viewed in a browser
at a width the author does not control, so line-based figures are embedded as SVG and
keep legible text at any zoom. Figures dominated by a dense image layer stay raster,
because wrapping the same pixels in base64 inside an SVG only inflates the report. Any
dense layer inside a vector figure must still be drawn with ``rasterized=True`` so that
a vector frame carries a raster interior.
"""

from __future__ import annotations

import numpy as np

from eeg_pipeline.infra.matplotlib import setup_matplotlib

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

#: Colour for a signal, source, or condition shown on its own.
PRIMARY_COLOR = OKABE_ITO["blue"]
#: Colour for the state before a correction step.
BEFORE_COLOR = OKABE_ITO["vermillion"]
#: Colour for the state after a correction step.
AFTER_COLOR = OKABE_ITO["blue"]
#: Colour for a physiological reference trace (ECG, EOG) shown as context.
REFERENCE_COLOR = OKABE_ITO["reddish_purple"]
#: Colour reserved for flags, exclusions, and threshold crossings.
FLAG_COLOR = OKABE_ITO["vermillion"]
#: Neutral colour for guides, medians, and zero lines.
GUIDE_COLOR = "0.35"

#: Per-run colours, cycled. Shared so that a run keeps one colour across every panel.
RUN_COLORS = (
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
)

#: Diverging colormap for baseline-relative power and other zero-centred decibels.
DIVERGING_POWER_COLORMAP = "RdBu_r"

#: Format for figures built from lines, scatters, and bars.
#:
#: Vector text stays legible at any zoom, and the path data for these figures is small.
REPORT_IMAGE_FORMAT = "svg"

#: Format for raster figures: those dominated by a dense image layer, and every list.
#:
#: A rasterized mesh inside an SVG is the same pixel data carried as base64, so it costs
#: more bytes while only sharpening the axis text. Measured on a three-row component
#: dossier at MNE's embedding resolution: 399 KB as WebP, 500 KB as PNG, 714 KB as SVG.
#: Across the ~310 dossiers in a five-band review that is 121 MB against 216 MB, so
#: these figures stay raster and use the smallest raster MNE supports.
REPORT_RASTER_IMAGE_FORMAT = "webp"


def report_image_format(*, has_dense_image: bool = False, is_figure_list: bool = False) -> str:
    """Return a safe embedding format for one ``Report.add_figure`` call.

    ``is_figure_list`` must be true whenever a list of figures is passed, because MNE
    renders a list through its slider template. That template has no SVG branch and
    always emits ``data:image/{format};base64``, which for SVG produces the invalid MIME
    type ``image/svg`` instead of ``image/svg+xml`` — browsers silently refuse to draw
    it, so every figure in the slider disappears. Only the single-figure template
    special-cases SVG and inlines the markup. Figure lists therefore stay raster
    regardless of their content.
    """
    if is_figure_list or has_dense_image:
        return REPORT_RASTER_IMAGE_FORMAT
    return REPORT_IMAGE_FORMAT


#: Percentile of the absolute values that defines a robust symmetric colour limit.
COLOR_LIMIT_PERCENTILE = 98.0


def apply_report_style() -> None:
    """Configure the non-interactive backend and shared render defaults."""
    setup_matplotlib()


def robust_symmetric_limit(
    *values: np.ndarray,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return a symmetric colour limit that a few extreme samples cannot dominate.

    A shared colour scale lets a reviewer compare components against each other, but
    taking the scale from the single largest absolute value lets one extreme
    component flatten every other component to the neutral colour. The limit is
    therefore taken from a high percentile of the pooled absolute values, and the
    caller is expected to state the resulting limit on the figure so that clipped
    samples are declared rather than hidden.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"Colour limit percentile must lie in (0, 100], got {percentile!r}.")
    pooled = np.concatenate([np.abs(np.asarray(value, dtype=float)).ravel() for value in values])
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        raise ValueError("Colour limits require at least one finite value.")
    limit = float(np.percentile(finite, percentile))
    if limit <= 0.0:
        raise ValueError("Colour limits require a non-zero spread of values.")
    return limit


def power_colorbar_label(limit: float, *, quantity: str = "Baseline-relative power") -> str:
    """Return a colourbar label that declares the symmetric clipping limit."""
    return f"{quantity} (dB, clipped at ±{limit:.1f})"


__all__ = [
    "AFTER_COLOR",
    "BEFORE_COLOR",
    "COLOR_LIMIT_PERCENTILE",
    "DIVERGING_POWER_COLORMAP",
    "FLAG_COLOR",
    "GUIDE_COLOR",
    "OKABE_ITO",
    "PRIMARY_COLOR",
    "REFERENCE_COLOR",
    "RUN_COLORS",
    "REPORT_IMAGE_FORMAT",
    "REPORT_RASTER_IMAGE_FORMAT",
    "apply_report_style",
    "report_image_format",
    "power_colorbar_label",
    "robust_symmetric_limit",
]
