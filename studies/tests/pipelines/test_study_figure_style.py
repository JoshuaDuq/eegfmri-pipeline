from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager

from studies.pain_study.figure_style import (
    outside_top_legend,
    publication_rc_params,
    require_font_family,
)


def test_publication_rc_params_define_editable_neutral_artwork() -> None:
    params = publication_rc_params("Arial", svg_hash_salt="study-figures")

    assert params["font.family"] == "Arial"
    assert params["font.size"] == 6.0
    assert params["axes.labelsize"] == 7.0
    assert params["axes.titlesize"] == 7.0
    assert params["legend.frameon"] is False
    assert params["svg.fonttype"] == "none"
    assert params["pdf.fonttype"] == 42
    assert params["figure.facecolor"] == "white"
    assert params["text.color"] == "#1A1A1A"


def test_require_font_family_rejects_missing_font(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_font(*args, **kwargs):
        raise ValueError("missing")

    monkeypatch.setattr(font_manager, "findfont", raise_missing_font)

    with pytest.raises(
        ValueError,
        match="Required figure font 'Missing' is unavailable",
    ):
        require_font_family("Missing")


def test_outside_top_legend_requires_and_reserves_labeled_artists() -> None:
    figure, axis = plt.subplots(layout="constrained")
    axis.plot([0, 1], [0, 1], label="Estimate")

    legend = outside_top_legend(figure, axis)
    figure.canvas.draw()

    assert axis.get_legend() is None
    assert figure.legends == [legend]
    assert legend.get_window_extent().y0 >= axis.get_window_extent().y1
    plt.close(figure)


def test_outside_top_legend_rejects_unlabeled_axes() -> None:
    figure, axis = plt.subplots(layout="constrained")

    with pytest.raises(ValueError, match="requires labeled artists"):
        outside_top_legend(figure, axis)

    plt.close(figure)
