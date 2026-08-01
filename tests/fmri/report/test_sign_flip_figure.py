"""The one Phase 2 measurement that earns a picture.

Height, survivors and p are numbers and live in the threshold table. What the table
cannot show is whether the observed maximum stands apart from the null or sits inside
it, which is a position in a distribution.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from fmri_pipeline.analysis.report.figures import sign_flip
from fmri_pipeline.analysis.report.inference import SignFlipSummary


def _summary(**overrides) -> SignFlipSummary:
    params = dict(
        height=7.02,
        survivors=38,
        global_p=0.0606,
        p_floor=0.0606,
        n_runs=6,
        n_patterns=32,
        observed_max=8.87,
    )
    params.update(overrides)
    return SignFlipSummary(**params)


def _texts(figure) -> str:
    return " ".join(
        artist.get_text()
        for artist in figure.findobj(match=lambda o: hasattr(o, "get_text"))
    )


def test_the_observed_value_and_the_height_are_both_marked():
    figure = sign_flip.sign_flip_figure([8.87, 5.7, 6.1, 5.2], summary=_summary())
    text = _texts(figure)
    assert "8.87" in text
    assert "7.02" in text
    plt.close(figure)


def test_every_pattern_is_drawn_rather_than_binned():
    """32 values binned into a density invents a shape the data does not have."""
    values = [8.87, 5.7, 6.1, 5.2, 6.4, 5.9]
    figure = sign_flip.sign_flip_figure(values, summary=_summary())
    plotted = [
        float(x)
        for collection in figure.axes[0].collections
        for x, _y in collection.get_offsets()
    ]
    assert sorted(plotted) == pytest.approx(sorted(values))
    plt.close(figure)


def test_each_pattern_gets_its_own_row():
    """Ranked rather than piled: no marker hides another and no jitter is invented."""
    values = [8.87, 5.7, 6.1, 5.2, 6.4, 5.9]
    figure = sign_flip.sign_flip_figure(values, summary=_summary())
    rows = [
        float(y)
        for collection in figure.axes[0].collections
        for _x, y in collection.get_offsets()
    ]
    assert sorted(rows) == list(range(1, len(values) + 1))
    plt.close(figure)


def test_the_observed_value_is_distinguished_from_the_rest():
    values = [8.87, 5.7, 6.1, 5.2]
    figure = sign_flip.sign_flip_figure(values, summary=_summary())
    collections = figure.axes[0].collections
    highlighted = [c for c in collections if len(c.get_offsets()) == 1]
    assert highlighted, "the observed maximum is not drawn apart from the null"
    assert float(highlighted[0].get_offsets()[0][0]) == pytest.approx(8.87)
    plt.close(figure)


def test_the_floor_is_named_when_it_binds():
    """The figure states that the p is at its floor; the caption explains why.

    Provenance lines are joined into a single 6.5pt strip, so a sentence here becomes
    an unreadable ribbon across the figure.
    """
    figure = sign_flip.sign_flip_figure([8.87, 5.7], summary=_summary())
    text = _texts(figure).lower()
    assert "0.061" in text
    assert "its floor for 6 runs" in text
    plt.close(figure)


def test_the_provenance_strip_fits_the_figure():
    """The helper joins every note into one 6.5pt line, which can outrun the canvas.

    Measured rather than counted in characters: what matters is whether the rendered
    strip fits, and a character budget is a guess at that.
    """
    figure = sign_flip.sign_flip_figure([8.87, 5.7], summary=_summary())
    figure.canvas.draw()
    strip = next(
        (t for t in figure.texts if "sign patterns over" in t.get_text()), None
    )
    assert strip is not None, "no provenance strip was drawn"

    rendered = strip.get_window_extent(figure.canvas.get_renderer()).width
    assert rendered <= figure.get_window_extent().width, (
        f"the provenance strip is {rendered:.0f}px wide against a "
        f"{figure.get_window_extent().width:.0f}px figure"
    )
    plt.close(figure)


def test_the_floor_is_stated_plainly_when_it_does_not_bind():
    figure = sign_flip.sign_flip_figure(
        [9.1, 5.0, 5.4],
        summary=_summary(n_runs=8, n_patterns=128, global_p=0.008, p_floor=0.0155),
    )
    text = _texts(figure).lower()
    assert "smallest this test can return" not in text
    assert "0.016" in text or "0.015" in text
    plt.close(figure)


def test_nothing_is_scored_against_a_criterion():
    figure = sign_flip.sign_flip_figure([8.87, 5.7], summary=_summary())
    text = _texts(figure).lower()
    for word in ("significant", "pass", "fail", "reject the null", "recommended"):
        assert word not in text
    plt.close(figure)


def test_an_empty_null_is_refused():
    with pytest.raises(ValueError, match="at least one enumerated maximum"):
        sign_flip.sign_flip_figure([], summary=_summary())
