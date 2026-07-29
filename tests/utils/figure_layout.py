"""Geometry assertions for report figures, measured after a real render.

A figure that overlaps its own annotations still builds, still passes every test that
inspects titles and limits, and is only wrong once someone looks at it. These helpers let
a test ask the question the reader asks: after layout has run, does this text land on top
of something else?

Rendered rather than declared coordinates, because the whole class of bug being pinned is
an artist placed in coordinates the layout engine never consulted. Only the renderer knows
where anything finally sat.
"""

from __future__ import annotations

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox


def rendered(figure: Figure):
    """Draw ``figure`` on an Agg canvas and return the renderer that placed it.

    Figures built for the report are closed by their plotting function and carry no
    canvas of their own, so asking one for a renderer raises. Attaching a canvas here
    keeps that detail out of every test that needs a bounding box.
    """
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return canvas.get_renderer()


def overlaps(first: Bbox, second: Bbox) -> bool:
    """Whether two rendered bounding boxes share any area."""
    return bool(
        first.x0 < second.x1
        and second.x0 < first.x1
        and first.y0 < second.y1
        and second.y0 < first.y1
    )


def colliding_text(figure: Figure) -> list[tuple[str, str]]:
    """Every pair of visible strings in ``figure`` that were drawn over each other.

    The invariant a reader depends on, stated once for any figure: two sentences printed
    in the same place leave both unreadable, and it does not matter which artist put them
    there. Asserting on the pairs rather than on a count makes the failure name the two
    strings that collided.

    Empty strings and whitespace are skipped -- matplotlib creates a text artist for every
    unlabelled tick and offset, and a zero-width box at the origin "overlaps" its
    neighbours without anything being wrong.
    """
    renderer = rendered(figure)
    drawn: list[tuple[str, Bbox]] = []
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        drawn.append((text.get_text(), box))
    return [
        (first_text, second_text)
        for index, (first_text, first_box) in enumerate(drawn)
        for second_text, second_box in drawn[index + 1 :]
        if overlaps(first_box, second_box)
    ]


__all__ = ["colliding_text", "overlaps", "rendered"]
