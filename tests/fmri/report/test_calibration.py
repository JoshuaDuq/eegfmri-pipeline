from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report import inference
from fmri_pipeline.analysis.report.figures import distributions


def _values(n: int = 40_000, seed: int = 0, signal: int = 800) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.standard_normal(n), 5.0 + rng.standard_normal(signal)])


def _context(values: np.ndarray, **kwargs) -> inference.ThresholdContext:
    params = dict(applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=True)
    params.update(kwargs)
    return inference.threshold_context(values, **params)


def _figure(values: np.ndarray, **kwargs) -> plt.Figure:
    return distributions.null_calibration_figure(
        values, context=_context(values, **kwargs), mask_source="analysis mask"
    )


def _legend_text(figure: plt.Figure) -> str:
    legend = figure.axes[0].get_legend()
    assert legend is not None
    return " | ".join(entry.get_text() for entry in legend.get_texts())


def _figure_text(figure: plt.Figure) -> str:
    return " | ".join(artist.get_text() for artist in figure.texts)


def _vertical_positions(figure: plt.Figure) -> list[float]:
    return sorted(
        float(line.get_xdata()[0])
        for line in figure.axes[0].lines
        if len(set(np.asarray(line.get_xdata(), dtype=float))) == 1
    )


# --- the two nulls --------------------------------------------------------


def test_the_panel_draws_the_theoretical_and_the_empirical_null() -> None:
    # Both, because the whole diagnostic is the gap between them.
    figure = _figure(_values())
    text = _legend_text(figure)
    assert "N(0, 1)" in text
    assert "empirical null" in text
    plt.close(figure)


def test_the_empirical_null_curve_reports_its_fitted_width() -> None:
    values = 1.5 * np.random.default_rng(1).standard_normal(50_000)
    figure = _figure(values)
    entry = next(t for t in _legend_text(figure).split(" | ") if "empirical null" in t)
    width = float(re.search(r",\s*([0-9.]+)", entry).group(1))
    assert width == pytest.approx(1.5, abs=0.05)
    plt.close(figure)


def test_the_panel_states_that_both_nulls_assume_every_voxel_is_null() -> None:
    # Scaling both curves to the full voxel count is only right when nothing is
    # active. Unstated, the reader reads the heights as fitted.
    figure = _figure(_values())
    assert "every voxel" in _figure_text(figure).lower()
    plt.close(figure)


def test_the_panel_survives_a_map_whose_null_cannot_be_fitted() -> None:
    values = np.concatenate([np.zeros(4_000), np.array([9.0])])
    figure = distributions.null_calibration_figure(
        values, context=_context(values), mask_source="analysis mask"
    )
    assert "empirical null" not in _legend_text(figure)
    plt.close(figure)


# --- thresholds -----------------------------------------------------------


def test_the_panel_draws_all_three_thresholds() -> None:
    values = _values()
    context = _context(values)
    figure = _figure(values)
    positions = _vertical_positions(figure)
    for threshold in (context.applied, context.fdr, context.bonferroni):
        assert threshold is not None
        assert any(abs(p - threshold) < 1e-6 for p in positions)
    plt.close(figure)


def test_a_two_sided_test_mirrors_every_threshold() -> None:
    values = _values()
    context = _context(values)
    positions = _vertical_positions(_figure(values))
    assert any(abs(p + context.applied) < 1e-6 for p in positions)


def test_a_one_sided_test_draws_only_the_tail_it_examined() -> None:
    # Mirroring here would mark a threshold on the half of the map the test never
    # looked at.
    values = _values()
    positions = _vertical_positions(_figure(values, two_sided=False))
    assert all(p > -1e-6 for p in positions)


def test_the_legend_states_how_many_voxels_survive_each_threshold() -> None:
    values = _values()
    context = _context(values)
    text = _legend_text(_figure(values))
    assert f"{context.applied_survivors:,}" in text
    assert f"{context.bonferroni_survivors:,}" in text


def test_the_panel_states_the_survivors_expected_under_the_null() -> None:
    # The number that makes an uncorrected threshold legible.
    figure = _figure(_values())
    assert "expected" in _legend_text(figure).lower()
    plt.close(figure)


def test_the_panel_says_so_when_fdr_rejects_nothing() -> None:
    # Absence of an FDR line would be indistinguishable from a rendering failure.
    values = np.random.default_rng(2).standard_normal(20_000)
    context = _context(values)
    assert context.fdr is None
    figure = distributions.null_calibration_figure(
        values, context=context, mask_source="analysis mask"
    )
    assert "no voxel" in _legend_text(figure).lower()
    plt.close(figure)


def test_every_drawn_threshold_stays_inside_the_axis_limits() -> None:
    # Bonferroni over many voxels can sit beyond the observed data range; a line
    # drawn outside the view is a threshold the reader never sees.
    values = np.random.default_rng(3).standard_normal(200_000)
    context = _context(values)
    figure = distributions.null_calibration_figure(
        values, context=context, mask_source="analysis mask"
    )
    low, high = figure.axes[0].get_xlim()
    assert low <= -context.bonferroni and high >= context.bonferroni
    plt.close(figure)


# --- axes and provenance --------------------------------------------------


def test_the_panel_uses_a_log_count_axis() -> None:
    # On a linear axis the null peak is the only visible feature and the tails --
    # which are the entire question -- lie flat against the axis.
    figure = _figure(_values())
    assert figure.axes[0].get_yscale() == "log"
    plt.close(figure)


def test_the_panel_names_the_voxels_it_was_given() -> None:
    figure = _figure(_values())
    text = _figure_text(figure)
    assert "analysis mask" in text
    assert "40,800 voxels" in text
    plt.close(figure)


def test_the_panel_reports_the_threshold_in_empirical_null_units() -> None:
    # 2.3 against a null of width 1.5 is 1.53 sigma, and that is the number a reader
    # cannot get from anywhere else in the document.
    values = 1.5 * np.random.default_rng(4).standard_normal(50_000)
    figure = _figure(values)
    assert "1.5" in _figure_text(figure)
    plt.close(figure)


def test_the_panel_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        values = np.array([1.0, 2.0])
        distributions.null_calibration_figure(
            np.array([]), context=_context(values), mask_source=""
        )


def test_the_removed_histogram_is_gone() -> None:
    # Replaced rather than supplemented: it drew a theoretical null over an
    # unmasked volume and offered no way to see the map's own null.
    assert not hasattr(distributions, "z_histogram")


def test_the_panel_draws_without_an_applied_threshold() -> None:
    values = _values()
    context = _context(values, applied_threshold=None)
    figure = distributions.null_calibration_figure(
        values, context=context, mask_source="analysis mask"
    )
    text = _legend_text(figure)
    assert "no height threshold applied" in text
    assert "Bonferroni" in text  # the corrected heights still belong on the axis
    plt.close(figure)
