"""Study 2 publication style and deterministic SVG output."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from studies.pain_study.figure_style import figure_size_inches, publication_rc_params

EMBEDDED_RASTER_DPI = 600
SVG_HASH_SALT = "study2-publication"


def study2_diverging_color_map() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "study2_source_association",
        ("#2166AC", "#F7F7F7", "#D95F0E"),
        N=256,
    )


@contextmanager
def publication_style(font_family: str) -> Iterator[None]:
    rc_params = publication_rc_params(font_family, svg_hash_salt=SVG_HASH_SALT)
    with mpl.rc_context(rc_params):
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
    "study2_diverging_color_map",
]
