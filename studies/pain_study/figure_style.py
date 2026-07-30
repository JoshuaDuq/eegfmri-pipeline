"""Shared publication style for the pain-study figure suite."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

MILLIMETERS_PER_INCH = 25.4
TEXT_COLOR = "#1A1A1A"
MUTED_TEXT_COLOR = "#4A4A4A"
AXIS_COLOR = "#333333"
REFERENCE_COLOR = "#5C5C5C"
GRID_COLOR = "#D9D9D9"


def figure_size_inches(dimensions_mm: Mapping[str, float]) -> tuple[float, float]:
    """Convert positive physical figure dimensions to inches."""

    width = float(dimensions_mm["width"])
    height = float(dimensions_mm["height"])
    if not all(isfinite(value) and value > 0.0 for value in (width, height)):
        raise ValueError("Publication figure dimensions must be finite and positive.")
    return width / MILLIMETERS_PER_INCH, height / MILLIMETERS_PER_INCH


def require_font_family(font_family: str) -> str:
    """Require an installed publication font without substitution."""

    try:
        font_manager.findfont(
            font_manager.FontProperties(family=font_family),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise ValueError(f"Required figure font '{font_family}' is unavailable.") from exc
    return font_family


def publication_rc_params(
    font_family: str,
    *,
    svg_hash_salt: str,
) -> dict[str, Any]:
    """Return the common final-size Matplotlib publication contract."""

    require_font_family(font_family)
    return {
        "font.family": font_family,
        "font.size": 6.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": TEXT_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.0,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": svg_hash_salt,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def outside_top_legend(figure: Figure, axis: Axes) -> Legend:
    """Add one layout-managed figure legend above a single data axis."""

    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        raise ValueError("Publication figure legend requires labeled artists.")
    return figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(handles),
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )


__all__ = [
    "AXIS_COLOR",
    "GRID_COLOR",
    "MUTED_TEXT_COLOR",
    "REFERENCE_COLOR",
    "TEXT_COLOR",
    "figure_size_inches",
    "outside_top_legend",
    "publication_rc_params",
    "require_font_family",
]
