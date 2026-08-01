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


def test_the_table_states_how_many_voxels_survive_each_threshold() -> None:
    # Moved out of the figure's legend, where four sentences of 7-point type took a
    # third of the canvas. The counts have to survive the move.
    values = _values()
    context = _context(values)
    table, _rows = distributions.threshold_table(context)
    assert f"{context.applied_survivors:,}" in table
    assert f"{context.bonferroni_survivors:,}" in table


def test_the_table_states_the_survivors_expected_under_the_null() -> None:
    # The number that makes an uncorrected threshold legible.
    table, _rows = distributions.threshold_table(_context(_values()))
    assert "expected" in table.lower()


def test_the_panel_says_so_when_fdr_rejects_nothing() -> None:
    # Absence of an FDR line would be indistinguishable from a rendering failure.
    values = np.random.default_rng(2).standard_normal(20_000)
    context = _context(values)
    assert context.fdr is None
    table, _rows = distributions.threshold_table(context)
    assert "none survives" in table.lower()


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


def test_the_panel_reports_what_the_applied_height_is_worth_against_the_fitted_null() -> None:
    # The number a reader cannot get anywhere else. A width of 1.5 puts |z| > 2.3 at
    # p = 0.06 in each tail against a nominal 0.021.
    values = 1.5 * np.random.default_rng(4).standard_normal(50_000)
    text = _figure_text(_figure(values))
    assert "fitted null" in text
    assert "0.06" in text


def test_the_panel_separates_the_two_tails_on_a_shifted_null() -> None:
    # A symmetric cut on a shifted null buys different evidence in each direction, and
    # the negative clusters on the map are then the weaker half. One number hides that.
    values = -0.6 + 1.5 * np.random.default_rng(5).standard_normal(100_000)
    text = _figure_text(_figure(values))
    assert "upward" in text and "downward" in text


def test_the_panel_states_the_nominal_p_the_applied_height_claims() -> None:
    # Without it the reader has the corrected figure but not the one it corrects.
    assert "nominal p" in _figure_text(_figure(_values()))


def test_a_one_sided_panel_reports_only_the_tail_it_examined() -> None:
    text = _figure_text(_figure(_values(), two_sided=False))
    assert "fitted null" in text
    assert "downward" not in text


# --- the fitted null carried into every count -----------------------------


def test_the_table_states_survivors_expected_under_the_fitted_null() -> None:
    # The comparison the panel exists for: an observed count that its own map's noise
    # fully explains must not read as enrichment over N(0, 1).
    values = 1.5 * np.random.default_rng(6).standard_normal(100_000)
    context = _context(values)
    table, _rows = distributions.threshold_table(context)
    assert "fitted null" in table
    assert f"{context.calibration.expected_survivors:,.0f}" in table


def test_the_table_gives_the_empirical_null_fdr_its_own_row() -> None:
    # Efron's correction, reported rather than only implied by the fitted curve.
    values = 1.5 * np.random.default_rng(7).standard_normal(100_000)
    table, _rows = distributions.threshold_table(_context(values))
    assert "vs N(0,1)" in table
    assert "vs fitted null" in table


def test_the_empirical_null_fdr_bounds_are_drawn_on_the_axis() -> None:
    rng = np.random.default_rng(8)
    values = np.concatenate(
        [-0.6 + 1.5 * rng.standard_normal(100_000), 12.0 + rng.standard_normal(500)]
    )
    context = _context(values)
    positions = _vertical_positions(_figure(values))
    for bound in (context.calibration.fdr_upper, context.calibration.fdr_lower):
        assert bound is not None
        assert any(abs(p - bound) < 1e-6 for p in positions)


def test_the_table_says_so_when_the_empirical_null_fdr_rejects_nothing() -> None:
    # Pure noise, however wide. A dropped row would read as a build failure.
    values = 1.5 * np.random.default_rng(9).standard_normal(50_000)
    table, _rows = distributions.threshold_table(_context(values))
    assert "vs fitted null" in table
    assert "none survives" in table.lower()


def test_a_panel_without_a_fitted_null_omits_its_fdr_rather_than_guessing() -> None:
    values = np.concatenate([np.zeros(4_000), np.array([9.0])])
    figure = distributions.null_calibration_figure(
        values, context=_context(values), mask_source="analysis mask"
    )
    assert "fitted null" not in _legend_text(figure)
    table, _rows = distributions.threshold_table(_context(values))
    assert "vs fitted null" not in table
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
    # The corrected heights still belong on the axis, and the table still says the
    # applied one was absent rather than omitting its row.
    labels = {artist.get_text() for artist in figure.axes[0].texts}
    assert any("Bonferroni" in label for label in labels)
    table, _rows = distributions.threshold_table(context)
    assert "no height applied" in table
    plt.close(figure)


# --- effect against evidence ----------------------------------------------
#
# A thresholded map answers "where is the evidence" and says nothing about how large
# the effects are, which is the quantity a result is reported in. The two come apart
# in both directions and neither case is visible in either map alone.


def _paired(n: int = 5_000, seed: int = 0):
    """Effects, statistics, and standard errors describing the same voxels."""
    rng = np.random.default_rng(seed)
    error = np.abs(rng.normal(0.05, 0.01, size=n))
    effect = rng.normal(0.0, 0.1, size=n)
    return effect, effect / error, error


def test_the_panel_relates_effect_to_evidence() -> None:
    effect, stat, error = _paired()
    figure = distributions.effect_versus_evidence_figure(
        effect, stat, standard_error=error, threshold=2.3, effect_units="% signal change"
    )
    assert "|z|" in figure.axes[0].get_xlabel()
    assert "% signal change" in figure.axes[0].get_ylabel()
    plt.close(figure)


def test_the_panel_states_how_many_voxels_it_drew() -> None:
    effect, stat, _error = _paired(n=3_000)
    figure = distributions.effect_versus_evidence_figure(effect, stat)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "3,000 voxels in the mask" in text
    plt.close(figure)


def test_a_large_map_is_subsampled_and_says_so() -> None:
    # 50,000 points is a solid block of ink that hides its own density, and the file
    # it produces dominates the report's size.
    effect, stat, _error = _paired(n=60_000)
    figure = distributions.effect_versus_evidence_figure(effect, stat)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "random sample" in text
    drawn = figure.axes[0].collections[0].get_offsets()
    assert len(drawn) < 60_000
    plt.close(figure)


def test_the_sample_is_reproducible() -> None:
    effect, stat, _error = _paired(n=60_000)
    first = distributions.effect_versus_evidence_figure(effect, stat)
    second = distributions.effect_versus_evidence_figure(effect, stat)
    np.testing.assert_array_equal(
        first.axes[0].collections[0].get_offsets(),
        second.axes[0].collections[0].get_offsets(),
    )
    plt.close(first)
    plt.close(second)


def test_effect_and_statistic_must_describe_the_same_voxels() -> None:
    with pytest.raises(ValueError, match="same voxels"):
        distributions.effect_versus_evidence_figure(np.zeros(10), np.zeros(11))


def test_non_finite_voxels_are_dropped_from_every_array_together() -> None:
    # Filtering each array independently leaves them of different lengths and pairs
    # every voxel after the first gap with a different voxel's statistic -- a
    # scatter that looks entirely normal and is scrambled.
    effect = np.array([1.0, np.nan, 3.0, 4.0])
    stat = np.array([1.0, 2.0, np.inf, 4.0])
    error = np.array([0.1, 0.1, 0.1, 0.1])
    figure = distributions.effect_versus_evidence_figure(
        effect, stat, standard_error=error
    )
    drawn = figure.axes[0].collections[0].get_offsets()
    assert len(drawn) == 2
    np.testing.assert_allclose(sorted(point[1] for point in drawn), [1.0, 4.0])
    plt.close(figure)


def test_a_standard_error_of_the_wrong_length_is_ignored_rather_than_paired() -> None:
    effect, stat, _error = _paired(n=100)
    figure = distributions.effect_versus_evidence_figure(
        effect, stat, standard_error=np.zeros(7)
    )
    assert len(figure.axes[0].collections) == 1
    plt.close(figure)


def test_the_colour_limit_is_robust_to_a_few_dropout_voxels() -> None:
    # Scaled to the maximum, a handful of very large standard errors take the top of
    # the ramp and every remaining point lands in its bottom tenth.
    effect, stat, error = _paired(n=2_000)
    error[:5] = 50.0
    figure = distributions.effect_versus_evidence_figure(
        effect, stat, standard_error=error
    )
    _low, high = figure.axes[0].collections[0].get_clim()
    assert high < 1.0
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "colour limit" in text and "clipped" in text
    plt.close(figure)


def test_the_panel_marks_the_height_the_maps_were_drawn_at() -> None:
    effect, stat, _error = _paired()
    figure = distributions.effect_versus_evidence_figure(effect, stat, threshold=2.3)
    marked = [line.get_xdata()[0] for line in figure.axes[0].lines]
    assert any(abs(float(x) - 2.3) < 1e-9 for x in marked)
    plt.close(figure)


def test_the_panel_scores_no_voxel() -> None:
    effect, stat, _error = _paired()
    figure = distributions.effect_versus_evidence_figure(effect, stat, threshold=2.3)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "no voxel is scored against a criterion here" in text
    plt.close(figure)


def test_an_empty_map_is_refused_rather_than_drawn_blank() -> None:
    with pytest.raises(ValueError, match="at least one voxel"):
        distributions.effect_versus_evidence_figure(
            np.array([np.nan]), np.array([np.nan])
        )


# --- counts belong in a table ---------------------------------------------
#
# These rode in the figure's legend as four sentences of 7-point type occupying a
# third of the canvas: a results table drawn in the wrong medium, beside the very
# lines it described.


def test_the_table_carries_every_threshold() -> None:
    context = _context(_values())
    table, rows = distributions.threshold_table(context)
    assert "Applied" in table
    assert "FDR" in table and "Bonferroni" in table
    # Header plus applied, FDR vs N(0,1), FDR vs fitted, Bonferroni.
    assert len(rows) == 5


def test_the_applied_row_states_both_expectations() -> None:
    # The comparison between them is the reading: a survivor count its own map's
    # noise fully explains reads as enrichment against N(0, 1) alone.
    context = _context(_values())
    table, _rows = distributions.threshold_table(context)
    assert "Expected under N(0,1)" in table
    assert "Expected under fitted null" in table


def test_an_asymmetric_region_is_not_collapsed_to_one_height() -> None:
    # The empirical-null bounds sit at different distances from zero whenever the
    # fitted null is shifted, and a single |z| > figure reintroduces exactly the
    # error the correction removes.
    context = _context(_values())
    table, _rows = distributions.threshold_table(context)
    if context.calibration is not None and context.calibration.fdr_lower is not None:
        assert "z &lt;" in table or "z <" in table


def test_a_threshold_nothing_survives_keeps_its_row() -> None:
    # "No voxel survives correction" is a finding, and dropping the row would make it
    # indistinguishable from a table that failed to build it.
    rng = np.random.default_rng(3)
    context = _context(rng.standard_normal(4_000))
    table, _rows = distributions.threshold_table(context)
    assert "FDR" in table


def test_the_tsv_matches_the_table() -> None:
    context = _context(_values())
    _table, rows = distributions.threshold_table(context)
    assert rows[0].startswith("Threshold\tRejection region\tVoxels surviving")
    assert all(len(row.split("\t")) == 5 for row in rows)


# --- the figure keeps only what a picture shows ---------------------------


def test_the_figure_legend_is_short_enough_to_read() -> None:
    # Three entries: the observed distribution and the two nulls it is read against.
    figure = _figure(_values())
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert len(legend.get_texts()) <= 3
    plt.close(figure)


def test_each_threshold_is_named_on_the_axis() -> None:
    figure = _figure(_values())
    labels = {artist.get_text() for artist in figure.axes[0].texts}
    assert "applied" in labels
    assert any("Bonferroni" in label for label in labels)
    plt.close(figure)


def test_a_rejection_region_containing_a_less_than_sign_is_escaped() -> None:
    # The empirical-null region reads "z < -6.57 or z > 5.35". Interpolated raw into
    # a <td>, the "<" opened a tag and the browser swallowed the cell -- the row
    # rendered with one column missing and nothing said so.
    rng = np.random.default_rng(8)
    values = np.concatenate(
        [-0.6 + 1.5 * rng.standard_normal(100_000), 12.0 + rng.standard_normal(500)]
    )
    context = _context(values)
    assert context.calibration.fdr_lower is not None
    table, rows = distributions.threshold_table(context)
    assert "&lt;" in table
    # Every row still carries a cell per header.
    import re

    header_count = len(re.findall(r"<th>", table))
    for body_row in re.findall(r"<tr>((?:<td>.*?</td>)+)</tr>", table):
        assert len(re.findall(r"<td>", body_row)) == header_count
    # The TSV keeps the raw text: it is read by scripts, not browsers.
    assert any("<" in row for row in rows)


def _summary(**overrides) -> inference.SignFlipSummary:
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
    return inference.SignFlipSummary(**params)


def test_p_floor_accounts_for_the_identity_tie():
    """The unflipped pattern is always in the null and always ties the observed max."""
    assert inference.sign_flip_p_floor(6) == pytest.approx(2 / 33)
    assert inference.sign_flip_p_floor(3) == pytest.approx(2 / 5)


def test_six_runs_cannot_reach_a_map_level_p_below_0_05():
    """Stated as a property, because it decides whether the p is worth printing."""
    assert inference.sign_flip_p_floor(6) > 0.05
    assert inference.sign_flip_p_floor(7) < 0.05
    assert _summary().floor_limited is True
    assert _summary(p_floor=0.031, n_runs=7).floor_limited is False


def test_threshold_context_carries_a_sign_flip_summary():
    context = _context(_values(), sign_flip=_summary())
    assert context.sign_flip.height == 7.02
    assert context.sign_flip.survivors == 38


def test_sign_flip_is_optional():
    """A single-run contrast has no null; the table must still build."""
    assert _context(_values()).sign_flip is None


def test_threshold_table_gains_a_sign_flip_row():
    table, rows = distributions.threshold_table(_context(_values(), sign_flip=_summary()))
    assert "sign-flip" in table.lower()
    assert "7.02" in table
    assert any("sign-flip" in row.lower() and "38" in row for row in rows)


def test_threshold_table_omits_the_row_without_a_null():
    table, rows = distributions.threshold_table(_context(_values()))
    assert "sign-flip" not in table.lower()


def test_sign_flip_row_is_not_scored_against_the_others():
    """The row states a height and a count. It must not rank or recommend."""
    table, _ = distributions.threshold_table(_context(_values(), sign_flip=_summary()))
    for word in ("recommended", "preferred", "correct choice", "should use", "best"):
        assert word not in table.lower()
