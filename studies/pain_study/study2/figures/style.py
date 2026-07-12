"""Study 2 publication style and deterministic SVG output."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure

MILLIMETERS_PER_INCH = 25.4
EMBEDDED_RASTER_DPI = 600


def figure_size_inches(dimensions_mm: Mapping[str, float]) -> tuple[float, float]:
    return (
        float(dimensions_mm["width"]) / MILLIMETERS_PER_INCH,
        float(dimensions_mm["height"]) / MILLIMETERS_PER_INCH,
    )


@contextmanager
def publication_style(font_family: str) -> Iterator[None]:
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=font_family),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise ValueError(f"Required figure font {font_family!r} is unavailable.") from exc
    with mpl.rc_context(
        {
            "font.family": font_family,
            "font.size": 6.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "svg.hashsalt": "study2-haufe-forward-patterns",
        }
    ):
        yield


def save_publication_svg(
    figure: Figure,
    output_path: Path,
    *,
    dimensions_mm: Mapping[str, float],
    font_family: str,
) -> Path:
    if output_path.suffix != ".svg":
        raise ValueError(f"Study 2 publication figures require an .svg path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(*figure_size_inches(dimensions_mm), forward=False)
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".svg",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        with publication_style(font_family):
            figure.savefig(
                temporary_path,
                format="svg",
                dpi=EMBEDDED_RASTER_DPI,
                metadata={"Date": None},
            )
        temporary_path.replace(output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)
    return output_path


def save_publication_png(
    figure: Figure,
    output_path: Path,
    *,
    dimensions_mm: Mapping[str, float],
    font_family: str,
    dpi: int,
) -> Path:
    if output_path.suffix != ".png":
        raise ValueError(f"Study 2 raster figure output requires a .png path: {output_path}")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 300:
        raise ValueError(
            "Study 2 publication PNG resolution must be an integer of at least 300 dpi."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(*figure_size_inches(dimensions_mm), forward=False)
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".png",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        with publication_style(font_family):
            figure.savefig(
                temporary_path,
                format="png",
                dpi=dpi,
                facecolor=figure.get_facecolor(),
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


__all__ = [
    "figure_size_inches",
    "publication_style",
    "save_publication_png",
    "save_publication_svg",
]
