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
        global_p=0.03125,
        p_floor=0.03125,
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


def test_every_pattern_is_a_step_on_the_curve():
    """32 values binned into a density invents a shape the data does not have."""
    values = [8.87, 5.7, 6.1, 5.2, 6.4, 5.9]
    figure = sign_flip.sign_flip_figure(values, summary=_summary())
    marked = [
        float(x)
        for line in figure.axes[0].lines
        if line.get_linestyle() == "None"
        for x, _y in line.get_xydata()
    ]
    assert sorted(marked) == pytest.approx(sorted(values))
    plt.close(figure)


def test_the_vertical_axis_is_a_probability_not_an_index():
    """Rank carries no information: nothing follows from a pattern being 17th."""
    values = [8.87, 5.7, 6.1, 5.2]
    figure = sign_flip.sign_flip_figure(values, summary=_summary())
    axis = figure.axes[0]
    heights = sorted(
        float(y)
        for line in axis.lines
        if line.get_linestyle() == "None"
        for _x, y in line.get_xydata()
    )
    assert heights == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert axis.get_ylim()[1] <= 1.1
    assert "proportion" in axis.get_ylabel()


def test_the_curve_is_monotone_and_reaches_one():
    """An ECDF that does not reach 1 is missing patterns from its own null."""
    values = [8.87, 5.7, 6.1, 5.2, 6.4, 5.9]
    figure = sign_flip.sign_flip_figure(values, summary=_summary())
    steps = [
        line for line in figure.axes[0].lines if line.get_drawstyle() != "default"
    ]
    assert steps, "no step curve was drawn"
    heights = steps[0].get_ydata()
    assert list(heights) == sorted(heights)
    assert heights[-1] == pytest.approx(1.0)
    plt.close(figure)


def test_the_familywise_level_is_drawn_so_the_height_can_be_read_off_it():
    """The height is the 0.95 quantile; without that line it cannot be verified."""
    figure = sign_flip.sign_flip_figure([8.87, 5.7, 6.1], summary=_summary())
    axis = figure.axes[0]
    horizontals = [
        float(line.get_ydata()[0])
        for line in axis.lines
        if len(set(line.get_ydata())) == 1 and len(line.get_ydata()) > 1
    ]
    assert any(abs(value - 0.95) < 1e-9 for value in horizontals)
    plt.close(figure)


def test_the_floor_is_named_when_it_binds():
    """The figure states that the p is at its floor; the caption explains why.

    Provenance lines are joined into a single 6.5pt strip, so a sentence here becomes
    an unreadable ribbon across the figure.
    """
    figure = sign_flip.sign_flip_figure(
        [8.87, 5.7], summary=_summary(global_p=0.125, p_floor=0.0625, n_runs=5)
    )
    text = _texts(figure).lower()
    assert "0.125" in text
    assert "floor 0.062" in text
    assert "its floor" not in text
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


def test_the_familywise_label_sits_clear_of_the_step_curve() -> None:
    # Anchored below-left of the crossing, the label was drawn straight through the
    # curve: the ECDF rises toward that corner, so down-left of it is exactly where
    # the steps are. Above the alpha line is empty until the curve reaches it.
    summary = _summary(height=5.18, observed_max=6.03)
    figure = sign_flip.sign_flip_figure(
        [2.8, 3.2, 3.6, 4.1, 4.7, 5.0, 5.4, 6.03], summary=summary
    )
    try:
        axis = figure.axes[0]
        label = next(child for child in axis.texts if "familywise" in child.get_text())
        _x, y = label.get_position()
        alignment = label.get_va()
    finally:
        plt.close(figure)

    assert y >= 0.95, "the label is anchored below the alpha line, over the curve"
    assert alignment == "bottom"
