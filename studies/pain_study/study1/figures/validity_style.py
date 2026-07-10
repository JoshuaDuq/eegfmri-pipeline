"""Publication style and SVG output for Study 1 validity figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.cohort import study1_output_root

MILLIMETERS_PER_INCH = 25.4


def validity_output_dir(config: Any) -> Path:
    parts = require_config_value(config, "study1.figures.validity.output_parts")
    return study1_output_root(config).joinpath(*(str(part) for part in parts))


def require_configured_font(config: Any) -> str:
    family = str(require_config_value(config, "study1.figures.validity.font.family"))
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise ValueError(f"Required figure font '{family}' is unavailable.") from exc
    return family


@contextmanager
def publication_style(config: Any) -> Iterator[None]:
    family = require_configured_font(config)
    font = require_config_value(config, "study1.figures.validity.font")
    style = require_config_value(config, "study1.figures.validity.style")
    rc_params = {
        "font.family": family,
        "font.size": float(font["tick_label_pt"]),
        "axes.labelsize": float(font["axis_label_pt"]),
        "xtick.labelsize": float(font["tick_label_pt"]),
        "ytick.labelsize": float(font["tick_label_pt"]),
        "legend.fontsize": float(font["legend_pt"]),
        "axes.linewidth": float(style["axis_line_width_pt"]),
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": float(style["axis_line_width_pt"]),
        "ytick.major.width": float(style["axis_line_width_pt"]),
        "svg.fonttype": "none",
    }
    with mpl.rc_context(rc_params):
        yield


def configured_figure_size(config: Any) -> tuple[float, float]:
    dimensions = require_config_value(config, "study1.figures.validity.dimensions_mm")
    return (
        float(dimensions["width"]) / MILLIMETERS_PER_INCH,
        float(dimensions["height"]) / MILLIMETERS_PER_INCH,
    )


def save_validity_svg(figure: Figure, output_path: Path, config: Any) -> Path:
    if output_path.suffix != ".svg":
        raise ValueError(f"Study 1 validity figures require an .svg path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(*configured_figure_size(config), forward=True)
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
                metadata={"Date": None},
            )
        temporary_path.replace(output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)
    return output_path


__all__ = [
    "configured_figure_size",
    "publication_style",
    "require_configured_font",
    "save_validity_svg",
    "validity_output_dir",
]
