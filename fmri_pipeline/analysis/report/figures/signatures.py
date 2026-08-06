"""Signature expression, as a table and -- once there are enough of them -- a figure.

What a reader does with these numbers is compare signatures against one another. Past
a handful that comparison is an ordering, which a plot shows and a table makes the
reader do by arithmetic. Below it the plot carries nothing the table does not: two
marks on an axis are two numbers, and the report printed those same two numbers
immediately beneath the figure.

The table is therefore always present and the figure is conditional. See
:data:`MIN_SIGNATURES_FOR_A_PLOT`.
"""

from __future__ import annotations

from html import escape
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)


def _escape(value: object) -> str:
    """Escape a cell before it is interpolated into markup.

    Cells carry values from the data: a rejection region reads "z < -6.57",
    and a regressor name is whatever the design called it. Interpolated raw,
    the first of those opened a tag and swallowed the cell that contained it.
    """
    return escape("" if value is None else str(value))


#: Metrics this panel can draw. The label states each one's comparability, because
#: that is the property a reader is most likely to assume wrongly.
_METRIC_LABELS = {
    "cosine": "Cosine similarity (unitless)",
    "pearson_r": "Pearson r",
    "dot": "Dot product (effect-map units; not comparable across subjects)",
}


@dataclass(frozen=True)
class SignaturePoint:
    """One signature's expression.

    A plain record rather than the analysis module's ``SignatureResult``, so this
    package keeps taking arrays and plain data and never imports the model path.
    """

    name: str
    dot: float
    cosine: Optional[float]
    pearson_r: Optional[float]
    n_voxels: int


def read_expression_tsv(path: Any) -> List[SignaturePoint]:
    """Read the expression table the analysis run wrote.

    The report reads derivatives rather than recomputing them: signature expression
    needs the weight maps and the study's signature configuration, neither of which
    the render path should have to reach for.

    An absent file means no signatures were configured, which is the stock
    configuration and not a fault. A blank metric field means that metric could not
    be computed for that signature, which is not the same as a value of zero.
    """
    from pathlib import Path

    file_path = Path(path)
    if not file_path.exists():
        return []

    def _optional(text: str) -> Optional[float]:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    # Columns are located by name, not by position. The writer lives in the analysis
    # package and this reader in the report package, coupled only by the order of a
    # TSV header; inserting a column there would have made this silently read the
    # wrong field and plot a number that is not the one it names. A missing column
    # now yields no points rather than a wrong figure.
    header = [name.strip() for name in lines[0].split("\t")]
    try:
        index = {name: header.index(name) for name in ("signature", "dot", "cosine", "pearson_r", "n_voxels")}
    except ValueError:
        return []

    def _field(fields: List[str], name: str) -> str:
        position = index[name]
        return fields[position] if position < len(fields) else ""

    points: List[SignaturePoint] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        dot = _optional(_field(fields, "dot"))
        try:
            n_voxels = int(float(_field(fields, "n_voxels") or 0))
        except ValueError:
            n_voxels = 0
        points.append(
            SignaturePoint(
                name=_field(fields, "signature").strip(),
                dot=0.0 if dot is None else dot,
                cosine=_optional(_field(fields, "cosine")),
                pearson_r=_optional(_field(fields, "pearson_r")),
                n_voxels=n_voxels,
            )
        )
    return points


#: Signatures below which a dot plot conveys nothing a table does not.
#:
#: A lollipop chart is read for the ordering and the spread across many marks. This
#: study configures two signatures, and two marks on an axis carry exactly the two
#: numbers a table carries -- while the report also printed those numbers immediately
#: below, so the same two values appeared twice on one screen. Above this count the
#: ordering starts to be the point and the plot earns its space.
MIN_SIGNATURES_FOR_A_PLOT = 5


def signature_table(
    results: Sequence[SignaturePoint], *, metric: str = "cosine"
) -> Tuple[str, List[str]]:
    """Signature expression as an HTML table and its TSV rows.

    Carries every quantity, not just the plotted one. Cosine similarity is bounded and
    unitless, so it is what compares across subjects; the dot product scales with the
    effect map's units and is what a reader reproducing the score needs; the voxel
    count says how much of the signature the map actually covered, which decides
    whether either number means anything.
    """
    headers = ["Signature", "Cosine", "Dot product", "Pearson r", "Voxels"]

    def _cell(value: Optional[float], digits: int = 3) -> str:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):+.{digits}f}"

    rows: List[List[str]] = []
    for point in results:
        rows.append(
            [
                point.name,
                _cell(point.cosine),
                f"{point.dot:+.4g}",
                _cell(point.pearson_r),
                f"{point.n_voxels:,}",
            ]
        )

    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    tsv = ["\t".join(headers)]
    tsv.extend("\t".join(row) for row in rows)
    return table_html, tsv


def signature_dot_plot(
    results: Sequence[SignaturePoint],
    *,
    metric: str = "cosine",
    title: str = "",
) -> Any:
    """Draw each signature's expression on one shared axis.

    ``cosine`` is the default because the dot product scales with the effect map's
    units: two subjects cannot be compared on it, and nothing about the number tells
    a reader that. Cosine similarity is bounded and unitless.

    A signature whose metric is missing is dropped rather than drawn at zero -- an
    unmeasured similarity is not a similarity of zero, and plotting it at the origin
    would place it exactly where "no relationship" lives.
    """
    if metric not in _METRIC_LABELS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {sorted(_METRIC_LABELS)}."
        )

    usable: List[SignaturePoint] = [
        point
        for point in results
        if getattr(point, metric) is not None
        and np.isfinite(float(getattr(point, metric)))
    ]
    if not usable:
        raise ValueError(f"No signature carried a finite {metric} to plot.")

    names = [point.name for point in usable]
    values = np.array([float(getattr(point, metric)) for point in usable])
    positions = np.arange(len(usable))

    with plot_context():
        # Row pitch, not a fixed canvas. At 0.42 inches per signature a two-signature
        # panel spent most of its height on empty axis between two lollipops, and the
        # marks a reader is comparing sat further apart than the numbers warrant.
        figure, ax = plt.subplots(figsize=(6.4, 0.34 * len(usable) + 1.5))

        # Sign is the whole interpretation of a signature score, so zero is drawn
        # first and each mark is coloured by which side of it it falls on.
        ax.axvline(0.0, color=GUIDE_COLOR, linewidth=0.9, zorder=1)
        colours = [
            OKABE_ITO["vermillion"] if value >= 0 else OKABE_ITO["blue"]
            for value in values
        ]
        # A stem to zero, so the distance from "no expression" is a length rather
        # than something the reader estimates against the axis.
        ax.hlines(
            positions, 0.0, values, color=colours, linewidth=1.4, alpha=0.6, zorder=2
        )
        ax.scatter(values, positions, s=46, c=colours, zorder=3)

        # The value on the mark. It is the number a reader takes away, and reading it
        # off the axis to three decimals is not something a lollipop supports -- so it
        # was only ever available in a separate block further down the page.
        span = float(np.max(np.abs(values))) or 1.0
        for position, value in zip(positions, values):
            ax.annotate(
                f"{value:+.3f}",
                xy=(value, position),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7.5,
                color=GUIDE_COLOR,
            )
        # Room for the labels, which otherwise run off the axis at the extremes.
        ax.set_xlim(
            min(0.0, float(np.min(values))) - 0.22 * span,
            max(0.0, float(np.max(values))) + 0.22 * span,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(_METRIC_LABELS[metric])
        if title:
            ax.set_title(title)

        voxels = {point.n_voxels for point in usable}
        voxel_note = (
            f"n = {next(iter(voxels)):,} voxels"
            if len(voxels) == 1
            else f"n = {min(voxels):,}-{max(voxels):,} voxels"
        )
        provenance = [voxel_note, f"metric: {metric}"]
        dropped = len(results) - len(usable)
        if dropped:
            provenance.append(f"{dropped} signature(s) carried no {metric}")
        annotate_provenance(figure, provenance)
        figure.tight_layout()
        return figure


__all__ = [
    "MIN_SIGNATURES_FOR_A_PLOT",
    "SignaturePoint",
    "read_expression_tsv",
    "signature_dot_plot",
    "signature_table",
]
