"""Report figures have to render every character their labels contain.

Matplotlib does not fail when a glyph is missing from the selected font. It warns and
draws a replacement box, so the defect reaches the HTML report rather than the console:
the Poincare axes read ``RR[] (s)`` and every band dossier title reads
``stimulus_temp [] [48.3, 49.3]``. These tests pin the render, not the configuration,
because the configuration that produced the boxes looked correct.
"""

from __future__ import annotations

import warnings

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eeg_pipeline.infra.matplotlib import setup_matplotlib  # noqa: E402

#: Characters the report's own figure labels use, with the label each comes from.
#:
#: The subscripts belong to the Poincare axes in ``report.analyzer_qc``, which name the
#: interval pair ``RRn`` against ``RRn+1``. The set symbol belongs to the condition
#: labels in ``band_ica_report``, which state the values a comparison group holds.
REPORT_LABEL_GLYPHS = "ₙ₊₁∈"


def _missing_glyph_warnings(label: str) -> list[str]:
    """Return matplotlib's missing-glyph warnings from rendering ``label``."""
    figure, axis = plt.subplots()
    axis.set_xlabel(label)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure.canvas.draw()
        return [str(entry.message) for entry in caught if "missing from font" in str(entry.message)]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("glyph", list(REPORT_LABEL_GLYPHS))
def test_a_character_used_in_a_figure_label_is_drawn_rather_than_boxed(glyph: str) -> None:
    """Arial carries none of these, and naming it alone gave no font that does.

    ``font.family="sans-serif"`` with a ``font.sans-serif`` preference list resolves to
    one font and stops there, so a glyph Arial lacks has nowhere to fall back to.
    """
    setup_matplotlib()

    assert _missing_glyph_warnings(f"RR{glyph} (s)") == []
