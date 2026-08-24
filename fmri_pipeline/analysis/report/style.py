"""Render conventions shared by every fMRI report figure.

Colour policy
-------------
Signed quantities -- z statistics, effect sizes -- are drawn with a diverging
colormap on a symmetric scale, so the neutral colour always marks zero. Unsigned
magnitudes -- tSNR, standard error -- are drawn with a single-hue perceptually
uniform ramp. Rainbow and multi-hue sequential colormaps are not used: their
non-monotonic lightness introduces boundaries that are not in the data.

``cold_hot`` is deliberately absent even though nilearn offers it. Its midpoint is
dark, which is correct only against ``black_bg=True``; these figures are drawn on a
white background, where the midpoint must be light for zero to read as neutral.

Matplotlib 3.10 added Crameri's perceptually uniform diverging maps -- ``berlin``,
``managua``, ``vanimo`` -- and they were evaluated and rejected here for the same
reason. Measured CIE L* at their midpoints: berlin 4.4, vanimo 7.1, managua 24.0,
against RdBu_r's 97.1. All three are built for a dark canvas; on a white report
surface their midpoint is the heaviest ink on the page, so "no effect" becomes the
most visually salient value in the figure. RdBu_r's lightness is monotonic across
each half (verified), which is the property that actually matters for a diverging
scale. Revisit only if these figures move to a dark canvas, or add ``cmcrameri`` for
``vik``, which is perceptually uniform *and* light-centred.

Scoping
-------
Style is applied through :func:`plot_context` rather than by mutating
``plt.rcParams`` at import. The EEG pipeline's ``setup_matplotlib`` mutates global
state via ``seaborn.set_theme``; when both pipelines run in one process, whichever
ran last silently restyles the other. A context manager cannot do that.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from eeg_pipeline.preprocessing.report.style import OKABE_ITO

#: Diverging colormap for signed maps. Light neutral at zero on a white background.
SIGNED_CMAP = "RdBu_r"

#: Single-hue perceptually uniform ramp for unsigned magnitude.
MAGNITUDE_CMAP = "cividis"

#: Pipeline decisions -- retained versus censored frames, ROI usable versus not --
#: get a neutral ramp. A hue would imply the quantity was measured rather than chosen.
SEQUENTIAL_DECISION_CMAP = "Greys"

#: Neutral colour for guides, thresholds, and reference curves.
GUIDE_COLOR = "0.35"

#: False is the neurological convention: subject left on the viewer's left.
#:
#: Passed explicitly to every volume plotter and named in every figure's provenance
#: line. Nilearn's own default happens to agree, but a figure that relies on a
#: library default states nothing, and a left/right error is invisible in the image.
RADIOLOGICAL = False


def orientation_label(radiological: bool = RADIOLOGICAL) -> str:
    """Name the convention a panel was actually drawn with.

    Takes an argument rather than only reading :data:`RADIOLOGICAL`, so a panel drawn
    against a non-default convention still describes itself truthfully.
    """
    return "radiological (R on viewer left)" if radiological else "neurological (L on viewer left)"


ORIENTATION_LABEL = orientation_label()

#: Dense figures embedded in the HTML report. At the report's 1180 px content width,
#: 200 dpi keeps slice labels and fine carpet structure sharp on high-density displays
#: without paying the size of manuscript-resolution raster output. Line figures are
#: SVG and therefore resolution-independent.
HTML_FIGURE_DPI = 200
#: Figures written to disk for manuscript use, where the resolution is wanted.
PRINT_FIGURE_DPI = 300

#: Percentile defining a robust colour limit a few extreme voxels cannot dominate.
COLOR_LIMIT_PERCENTILE = 98.0

#: Smallest ratio of colour limit to threshold that leaves a panel usable range.
#:
#: Without a floor, a thresholded panel drawn from noise gets a limit barely above
#: its own threshold -- p99(|z|) of standard normal noise is about 2.58 against a
#: 2.3 threshold -- and every surviving voxel saturates to one colour.
SUPRATHRESHOLD_HEADROOM = 1.5

FMRI_RC: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "figure.dpi": HTML_FIGURE_DPI,
    "savefig.dpi": HTML_FIGURE_DPI,
    "savefig.bbox": "tight",
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "image.interpolation": "nearest",
    # Element ids in an SVG are salted from this. Left unset, Matplotlib salts from
    # the process, so two renders of identical data produce different files.
    "svg.hashsalt": "fmri-report",
    # Text becomes paths. Costs bytes, but the figure then renders identically on a
    # machine without Arial -- including a journal's typesetting system, which is
    # where a report figure eventually ends up.
    "svg.fonttype": "path",
}


def plot_context() -> AbstractContextManager:
    """Return a context in which this package's render defaults apply."""
    return plt.rc_context(FMRI_RC)


def savefig_kwargs(path: Path) -> dict[str, Any]:
    """Return ``savefig`` arguments that make the output byte-reproducible.

    Matplotlib stamps a creation date into SVG and a ``Software`` tag into PNG.
    Either makes two renders of identical data differ, so a figure cannot be diffed
    against its predecessor, content-addressed, or cached. Measured on this
    installation: PNG is stable either way, SVG is not.

    ``bbox_inches`` is passed explicitly rather than left to the rcParam. The
    provenance strip is drawn below the canvas and is only included in the output
    because the bounding box is tight; a caller saving under different rc state would
    otherwise silently crop the line off every figure.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".svg":
        return {"bbox_inches": "tight", "metadata": {"Date": None}}
    if suffix == ".png":
        return {"bbox_inches": "tight", "metadata": {"Software": None}}
    return {"bbox_inches": "tight"}


#: Where the provenance strip sits, in figure coordinates.
#:
#: Below the canvas, not on it. ``savefig`` runs with a tight bounding box, which
#: expands the saved image to include every artist, so the strip gets a band of its
#: own and the figure above it is untouched.
PROVENANCE_Y = -0.045


def annotate_provenance(figure: plt.Figure, lines: Sequence[str]) -> None:
    """Print the numbers a reader needs to trust the figure, inside the figure.

    A figure travels: it gets pulled out of the report into a slide, a manuscript,
    an email. Everything needed to interpret it -- how many samples, what threshold,
    what colour limit, how much the limit clipped -- has to survive that trip, and a
    caption in the surrounding HTML does not.

    Stating the clipped fraction matters most. A robust colour limit deliberately
    saturates the extreme voxels; unstated, the figure silently claims it did not.

    The strip sits below the canvas rather than in its bottom corner. On a nilearn
    mosaic the axes run to the figure edge and each slice writes its coordinate just
    under its own box, so text at the corner lands on top of those labels -- and a
    provenance line that cannot be read is worse than none, since the figure still
    looks as though it documents itself.

    Placing it below and letting the tight bounding box grow to include it is what
    survives nilearn's layout. Reserving space by moving the axes does not: the
    slicers reposition themselves at draw time and take the reserved band back.
    """
    if not lines:
        return
    figure.text(
        0.005,
        PROVENANCE_Y,
        "  ·  ".join(lines),
        fontsize=6.5,
        color=GUIDE_COLOR,
        va="center",
        ha="left",
    )


def colour_limit_note(limit: float, clipped: float) -> str:
    """One-line description of a colour limit and how much of the data it hid.

    A robust colour limit deliberately saturates the extreme values. Left unstated,
    the figure silently claims it did not.
    """
    return f"colour limit ±{limit:.2f} · {clipped * 100:.2f}% clipped"


def panel_label(ax: Any, letter: str) -> None:
    """Put a bold panel letter above the top-left corner of an axes, journal style."""
    ax.text(
        -0.02,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _finite(*values: np.ndarray) -> np.ndarray:
    pooled = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        raise ValueError("Colour limits require at least one finite value.")
    return finite


def robust_symmetric_limit(
    *values: np.ndarray,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return a symmetric colour limit a few extreme samples cannot dominate.

    Taking the limit from the single largest absolute value lets one extreme voxel
    flatten the rest of the map to the neutral colour. The limit therefore comes
    from a high percentile of the pooled absolute values, and callers state it on
    the figure so that clipped voxels are declared rather than hidden.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"Percentile must lie in (0, 100], got {percentile!r}.")
    return float(np.percentile(np.abs(_finite(*values)), percentile))


def robust_upper_limit(
    values: np.ndarray,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return an upper colour limit for an unsigned magnitude.

    Separate from :func:`robust_symmetric_limit` because tSNR and standard error
    have no negative half; taking a symmetric limit for them wastes half the ramp
    and implies a sign the quantity does not have.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"Percentile must lie in (0, 100], got {percentile!r}.")
    return float(np.percentile(_finite(values), percentile))


def clipped_fraction(values: np.ndarray, *, limit: float) -> float:
    """Return the fraction of finite values a colour limit saturates.

    Reported on every figure that uses a robust limit, so that saturation is a
    declared property of the panel rather than something a reader has to suspect.
    """
    finite = _finite(values)
    return float(np.mean(np.abs(finite) > abs(float(limit))))


def suprathreshold_limit(
    values: np.ndarray,
    *,
    threshold: float,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return a colour limit for a thresholded panel.

    Computed over only the voxels that survive ``threshold``, because a limit taken
    from the whole map is dominated by the sub-threshold voxels the panel does not
    show. Floored at ``threshold * SUPRATHRESHOLD_HEADROOM`` so the panel keeps
    usable dynamic range even when nothing meaningfully exceeds the threshold.
    """
    if threshold <= 0:
        raise ValueError(f"Threshold must be > 0, got {threshold!r}.")
    floor = float(threshold) * SUPRATHRESHOLD_HEADROOM
    finite = _finite(values)
    surviving = np.abs(finite)[np.abs(finite) > float(threshold)]
    if surviving.size == 0:
        return floor
    return max(floor, float(np.percentile(surviving, percentile)))


def figure_format(*, dense: bool) -> str:
    """Return the embedding format for one figure.

    Figures dominated by a dense image layer -- brain mosaics, carpets -- stay
    raster: wrapping the same pixels in base64 inside an SVG costs more bytes while
    only sharpening the axis text. Line and bar figures are vector, so their text
    stays legible at any zoom in a browser whose width the author does not control.
    """
    return "png" if dense else "svg"


def save_report_figure(
    figure: Any,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dense: bool = True,
) -> Path:
    """Save one report figure and return its preferred HTML artifact."""
    preferred = figure_format(dense=dense)
    wanted = [preferred, *(fmt for fmt in formats if fmt != preferred)]
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary: Path | None = None
        with plot_context():
            for fmt in wanted:
                path = out_dir / f"{stem}.{fmt}"
                figure.savefig(path, **savefig_kwargs(path))
                if primary is None:
                    primary = path
        if primary is None:  # pragma: no cover - preferred always supplies one
            raise RuntimeError(f"No figure artifact was written for {stem!r}.")
        return primary
    finally:
        plt.close(figure)


__all__ = [
    "COLOR_LIMIT_PERCENTILE",
    "FMRI_RC",
    "GUIDE_COLOR",
    "HTML_FIGURE_DPI",
    "MAGNITUDE_CMAP",
    "OKABE_ITO",
    "ORIENTATION_LABEL",
    "PRINT_FIGURE_DPI",
    "RADIOLOGICAL",
    "SEQUENTIAL_DECISION_CMAP",
    "SIGNED_CMAP",
    "annotate_provenance",
    "clipped_fraction",
    "colour_limit_note",
    "figure_format",
    "orientation_label",
    "panel_label",
    "plot_context",
    "robust_symmetric_limit",
    "robust_upper_limit",
    "save_report_figure",
    "savefig_kwargs",
    "suprathreshold_limit",
]
