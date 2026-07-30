"""Publication style and SVG/PNG output for Study 1 validity figures."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.figure_style import (
    figure_size_inches,
    publication_rc_params,
    require_font_family,
)
from studies.pain_study.study1.cohort import study1_output_root

EMBEDDED_RASTER_DPI = 600
SVG_HASH_SALT = "study1-validity"


def validity_output_dir(config: Any) -> Path:
    parts = require_config_value(config, "study1.figures.validity.output_parts")
    return study1_output_root(config).joinpath(*(str(part) for part in parts))


def require_configured_font(config: Any) -> str:
    family = str(require_config_value(config, "study1.figures.validity.font.family"))
    return require_font_family(family)


@contextmanager
def publication_style(config: Any) -> Iterator[None]:
    family = require_configured_font(config)
    font = require_config_value(config, "study1.figures.validity.font")
    style = require_config_value(config, "study1.figures.validity.style")
    rc_params = publication_rc_params(family, svg_hash_salt=SVG_HASH_SALT)
    rc_params.update(
        {
            "font.size": float(font["tick_label_pt"]),
            "axes.labelsize": float(font["axis_label_pt"]),
            "xtick.labelsize": float(font["tick_label_pt"]),
            "ytick.labelsize": float(font["tick_label_pt"]),
            "legend.fontsize": float(font["legend_pt"]),
            "axes.linewidth": float(style["axis_line_width_pt"]),
            "xtick.major.width": float(style["axis_line_width_pt"]),
            "ytick.major.width": float(style["axis_line_width_pt"]),
        }
    )
    with mpl.rc_context(rc_params):
        yield


def configured_figure_size(config: Any) -> tuple[float, float]:
    dimensions = require_config_value(config, "study1.figures.validity.dimensions_mm")
    return figure_size_inches(dimensions)


def save_publication_svg(
    figure: Figure,
    output_path: Path,
    config: Any,
    *,
    dimensions_mm: Mapping[str, float],
) -> Path:
    if output_path.suffix != ".svg":
        raise ValueError(f"Study 1 publication figures require an .svg path: {output_path}")
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
        with publication_style(config):
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
    config: Any,
    *,
    dimensions_mm: Mapping[str, float],
    dpi: int,
) -> Path:
    if output_path.suffix != ".png":
        raise ValueError(f"Study 1 publication figures require an .png path: {output_path}")
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise TypeError("Study 1 publication PNG DPI must be an integer.")
    if dpi < 300:
        raise ValueError("Study 1 publication PNG DPI must be at least 300.")

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
        with publication_style(config):
            figure.savefig(
                temporary_path,
                format="png",
                dpi=dpi,
                metadata={"Software": "EEG_fMRI_Pipeline"},
            )
        temporary_path.replace(output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)
    return output_path


def save_validity_svg(figure: Figure, output_path: Path, config: Any) -> Path:
    dimensions = require_config_value(config, "study1.figures.validity.dimensions_mm")
    return save_publication_svg(
        figure,
        output_path,
        config,
        dimensions_mm=dimensions,
    )


__all__ = [
    "configured_figure_size",
    "figure_size_inches",
    "publication_style",
    "require_configured_font",
    "save_publication_png",
    "save_publication_svg",
    "save_validity_svg",
    "validity_output_dir",
]
