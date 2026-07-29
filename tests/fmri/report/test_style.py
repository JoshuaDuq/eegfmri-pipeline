from __future__ import annotations

import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report import style


def test_plot_context_does_not_leak_into_global_rcparams() -> None:
    before = plt.rcParams["axes.spines.top"]
    with style.plot_context():
        assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.top"] == before


def test_robust_symmetric_limit_is_not_dominated_by_a_single_outlier() -> None:
    values = np.concatenate([np.full(999, 1.0), np.array([1000.0])])
    assert style.robust_symmetric_limit(values) == pytest.approx(1.0, abs=0.01)


def test_robust_symmetric_limit_rejects_an_all_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        style.robust_symmetric_limit(np.array([np.nan, np.inf]))


def test_suprathreshold_limit_uses_only_surviving_voxels() -> None:
    # 990 sub-threshold voxels must not drag the limit down toward the threshold.
    values = np.concatenate([np.full(990, 0.1), np.full(10, 8.0)])
    assert style.suprathreshold_limit(values, threshold=2.3) == pytest.approx(8.0, abs=0.01)


def test_suprathreshold_limit_floors_above_the_threshold_for_a_noise_map() -> None:
    # p99(|z|) of standard normal noise is ~2.58, barely above a 2.3 threshold.
    rng = np.random.default_rng(0)
    limit = style.suprathreshold_limit(rng.standard_normal(100_000), threshold=2.3)
    assert limit >= 2.3 * 1.5


def test_suprathreshold_limit_floors_when_nothing_survives() -> None:
    assert style.suprathreshold_limit(np.zeros(100), threshold=2.3) == pytest.approx(3.45)


def test_dense_figures_are_raster_and_line_figures_are_vector() -> None:
    assert style.figure_format(dense=True) == "png"
    assert style.figure_format(dense=False) == "svg"


def test_signed_and_magnitude_colormaps_are_not_rainbows() -> None:
    assert style.SIGNED_CMAP == "RdBu_r"
    assert style.MAGNITUDE_CMAP == "cividis"


def test_robust_upper_limit_ignores_sign_conventions_of_symmetric_data() -> None:
    # An unsigned magnitude gets an upper bound, not a symmetric one.
    assert style.robust_upper_limit(np.arange(101.0)) == pytest.approx(98.0, abs=0.5)


def test_clipped_fraction_reports_what_a_colour_limit_hides() -> None:
    values = np.concatenate([np.zeros(90), np.full(10, 100.0)])
    assert style.clipped_fraction(values, limit=50.0) == pytest.approx(0.10)


def test_clipped_fraction_is_zero_when_the_limit_covers_everything() -> None:
    assert style.clipped_fraction(np.arange(10.0), limit=100.0) == 0.0


def test_annotate_provenance_writes_its_lines_into_the_figure() -> None:
    figure, _ = plt.subplots()
    style.annotate_provenance(figure, ["n = 1,024 voxels", "|z| > 2.30"])
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "n = 1,024 voxels" in text and "|z| > 2.30" in text
    plt.close(figure)


def _render(tmp_path, name: str) -> str:
    with style.plot_context():
        figure, axis = plt.subplots()
        axis.plot([1, 2, 3])
        path = tmp_path / name
        figure.savefig(path, **style.savefig_kwargs(path))
        plt.close(figure)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_svg_rendering_is_byte_stable_across_repeated_renders(tmp_path) -> None:
    # SVG embeds a timestamp by default, so figures churn and cannot be diffed.
    first = _render(tmp_path, "a.svg")
    time.sleep(1.1)
    assert _render(tmp_path, "b.svg") == first


def test_png_rendering_is_byte_stable_across_repeated_renders(tmp_path) -> None:
    first = _render(tmp_path, "a.png")
    time.sleep(1.1)
    assert _render(tmp_path, "b.png") == first


def test_orientation_is_a_stated_convention_not_an_inferred_one() -> None:
    """A left/right error is invisible in the image, so the convention must be named."""
    assert style.RADIOLOGICAL is False
    assert "neurological" in style.ORIENTATION_LABEL
    assert "L on viewer left" in style.ORIENTATION_LABEL


def test_a_panel_drawn_radiologically_says_so() -> None:
    """The helper takes an argument so a non-default panel describes itself truthfully."""
    assert "radiological" in style.orientation_label(True)
    assert "R on viewer left" in style.orientation_label(True)
    assert style.orientation_label(False) == style.ORIENTATION_LABEL


def test_pipeline_decisions_get_a_neutral_ramp_not_a_hue() -> None:
    """Retained-vs-censored is a pipeline decision, not a measured quantity."""
    assert style.SEQUENTIAL_DECISION_CMAP == "Greys"


def test_embedded_figures_are_lighter_than_print_figures() -> None:
    """300 dpi at a 1180 px layout width embeds resolution no reader sees."""
    assert style.HTML_FIGURE_DPI == 150
    assert style.PRINT_FIGURE_DPI == 300
    assert style.FMRI_RC["savefig.dpi"] == style.HTML_FIGURE_DPI


def test_the_robust_percentile_stays_conservative() -> None:
    assert style.COLOR_LIMIT_PERCENTILE == 98.0


def test_the_colour_limit_note_states_what_was_clipped() -> None:
    note = style.colour_limit_note(3.5, 0.012)
    assert "3.5" in note
    assert "1.2" in note


def test_a_panel_letter_lands_outside_the_axes() -> None:
    figure, ax = plt.subplots()
    try:
        style.panel_label(ax, "A")
        texts = [t for t in ax.texts if t.get_text() == "A"]
        assert len(texts) == 1
        assert texts[0].get_fontweight() == "bold"
        assert texts[0].get_position()[1] > 1.0
    finally:
        plt.close(figure)


# --- the provenance strip -------------------------------------------------


def test_the_provenance_strip_sits_below_the_canvas() -> None:
    # On a nilearn mosaic each slice writes its coordinate just under its own box, so
    # text placed in the figure's bottom corner lands on top of those labels. The
    # strip goes below the canvas and the tight bounding box grows to include it.
    #
    # Reserving space by moving the axes was tried and does not survive: the slicers
    # reposition themselves at draw time and take the reserved band back.
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    style.annotate_provenance(figure, ["n = 10 voxels", "|z| > 2.30"])
    assert figure.texts[0].get_position()[1] < 0
    assert figure.texts[0].get_position()[1] < axis.get_position().y0
    plt.close(figure)


def test_saving_always_uses_a_tight_bounding_box() -> None:
    # The strip is off-canvas, so a save without a tight box crops it off every
    # figure -- silently, since the figure itself still renders.
    for name in ("panel.png", "panel.svg", "panel.pdf"):
        assert style.savefig_kwargs(Path(name))["bbox_inches"] == "tight"


def test_the_strip_is_written_as_one_line() -> None:
    import matplotlib.pyplot as plt

    figure, _axis = plt.subplots(figsize=(6.0, 4.0))
    style.annotate_provenance(figure, ["a", "b", "c"])
    assert figure.texts[0].get_text().count("\n") == 0
    assert "a" in figure.texts[0].get_text() and "c" in figure.texts[0].get_text()
    plt.close(figure)


def test_no_lines_leaves_the_figure_untouched() -> None:
    import matplotlib.pyplot as plt

    figure, _axis = plt.subplots(figsize=(6.0, 4.0))
    style.annotate_provenance(figure, [])
    assert not figure.texts
    plt.close(figure)
