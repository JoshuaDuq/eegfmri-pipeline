"""Contracts for the design-estimability figures.

Merged from the two parallel implementations: every behaviour either side asserted is
asserted here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.figures import design


def _design_frame(n_scans: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    time = np.arange(n_scans)
    return pd.DataFrame(
        {
            "cond_a": np.sin(time / 9.0),
            "cond_b": np.cos(time / 9.0),
            "trans_x": rng.normal(scale=0.05, size=n_scans),
            "rot_z": rng.normal(scale=0.02, size=n_scans),
            "drift_1": np.cos(np.pi * time / n_scans),
            "drift_2": np.cos(2 * np.pi * time / n_scans),
            "constant": np.ones(n_scans),
        }
    )


# --------------------------------------------------------------------------- #
# Regressor roles
# --------------------------------------------------------------------------- #


def test_regressors_are_grouped_by_role_not_left_interleaved() -> None:
    columns = ["drift_1", "cond_a", "trans_x", "constant", "cond_b", "a_comp_cor_00"]
    ordered, groups = design.classify_regressors(columns)

    assert ordered[:2] == ["cond_a", "cond_b"]
    assert [g.name for g in groups] == ["Task", "Confound", "Drift", "Constant"]
    assert [(g.start, g.stop) for g in groups] == [(0, 2), (2, 4), (4, 5), (5, 6)]


def test_every_column_lands_in_exactly_one_group() -> None:
    columns = ["trans_x", "heat", "drift_1", "warm", "constant", "a_comp_cor_00"]
    ordered, groups = design.classify_regressors(columns)
    assert sorted(ordered) == sorted(columns)
    assert sum(g.size for g in groups) == len(columns)


def test_confound_and_drift_naming_conventions_are_recognised() -> None:
    columns = [
        "trans_x_derivative1_power2",
        "framewise_displacement",
        "a_comp_cor_03",
        "non_steady_state_outlier00",
        "white_matter",
        "csf",
        "drift_7",
        "cosine03",
        "stimulation",
    ]
    _ordered, groups = design.classify_regressors(columns)
    sizes = {g.name: g.size for g in groups}
    assert sizes["Confound"] == 6
    assert sizes["Drift"] == 2
    assert sizes["Task"] == 1


# --------------------------------------------------------------------------- #
# Variance inflation
# --------------------------------------------------------------------------- #


def test_vif_is_near_one_for_independent_regressors() -> None:
    rng = np.random.default_rng(1)
    independent = rng.normal(size=(500, 4))
    vifs = design.variance_inflation_factors(independent)
    assert np.all(vifs < 1.2)


def test_vif_is_infinite_for_an_exactly_collinear_column() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(200, 2))
    collinear = np.column_stack([base, base[:, 0] + base[:, 1]])
    vifs = design.variance_inflation_factors(collinear)
    assert np.isinf(vifs).any()


def test_vif_rises_with_correlation() -> None:
    rng = np.random.default_rng(3)
    first = rng.normal(size=1000)
    mild = np.column_stack([first, first * 0.3 + rng.normal(size=1000)])
    severe = np.column_stack([first, first * 3.0 + rng.normal(scale=0.1, size=1000)])
    assert design.variance_inflation_factors(
        severe
    ).max() > design.variance_inflation_factors(mild).max()


def test_vif_returns_empty_for_a_single_column() -> None:
    """A lone column has nothing to be inflated by."""
    assert design.variance_inflation_factors(np.ones((10, 1))).size == 0


def test_vif_does_not_overflow_on_a_realistic_design_with_an_intercept(recwarn) -> None:
    # The design's own constant column plus the intercept added by the regression
    # makes the system singular: lstsq returns an enormous beta and the residual
    # sum overflows. Only shows up with a realistic number of regressors.
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([rng.standard_normal((n, 8)), np.linspace(0, 1, n), np.ones(n)])
    vif = design.variance_inflation_factors(X)
    numeric = [
        w for w in recwarn if "overflow" in str(w.message) or "divide" in str(w.message)
    ]
    assert numeric == []
    assert np.isinf(vif[-1])  # the constant column
    assert np.all(np.isfinite(vif[:-1]))


# --------------------------------------------------------------------------- #
# Contrast efficiency
# --------------------------------------------------------------------------- #


def test_efficiency_falls_when_the_compared_conditions_become_collinear() -> None:
    """Two conditions that share variance estimate their difference less precisely."""
    n = 300
    rng = np.random.default_rng(4)
    independent_a = rng.normal(size=n)
    independent_b = rng.normal(size=n)
    shared = independent_a * 0.98 + rng.normal(scale=0.05, size=n)

    contrast = np.array([1.0, -1.0, 0.0])
    separable = np.column_stack([independent_a, independent_b, np.ones(n)])
    entangled = np.column_stack([independent_a, shared, np.ones(n)])

    assert design.contrast_efficiency(entangled, contrast) < design.contrast_efficiency(
        separable, contrast
    )


def test_efficiency_is_none_for_a_mismatched_contrast_length() -> None:
    assert design.contrast_efficiency(np.eye(4), np.array([1.0, -1.0])) is None


def test_an_inestimable_contrast_has_no_efficiency_rather_than_a_flattering_one() -> None:
    """Two identical regressors cannot have their difference estimated at all.

    The contrast lies in the null space of the design, so returning a finite number
    here would advertise precision for a comparison the data cannot make.
    """
    a = np.linspace(-1, 1, 40)
    X = np.column_stack([a, a, np.ones(40)])
    assert design.contrast_efficiency(X, np.array([1.0, -1.0, 0.0])) is None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def test_design_matrix_figure_carries_the_contrast_on_a_shared_axis() -> None:
    """A weight only means something once you can see which regressor it lands on."""
    figure = design.design_matrix_figure(
        _design_frame(),
        contrast={"cond_a": 1.0, "cond_b": -1.0},
        tr_seconds=2.0,
        run_label="run-01",
    )
    try:
        data_axes = [ax for ax in figure.axes if not hasattr(ax, "_colorbar_info")]
        assert len(data_axes) == 2, "expected the matrix and the contrast strip"
        matrix_ax, contrast_ax = data_axes
        assert matrix_ax.get_xlim() == contrast_ax.get_xlim()
        # The weight sits on its cell; the column it lands on is named by the tick
        # label beneath that same cell. Both on the shared axis is what makes the
        # pairing readable without any matching up.
        figure.canvas.draw()
        weights = {t.get_text(): round(t.get_position()[0]) for t in contrast_ax.texts}
        assert "+1" in weights
        ticks = {
            round(t.get_position()[0]): t.get_text()
            for t in contrast_ax.get_xticklabels()
            if t.get_text()
        }
        assert ticks[weights["+1"]] == "cond_a"
    finally:
        plt.close(figure)


def test_design_matrix_figure_labels_every_regressor() -> None:
    matrix = _design_frame()
    figure = design.design_matrix_figure(matrix)
    try:
        # The matrix and contrast strip share an x axis, so the tick labels live on
        # whichever axes matplotlib designates as the shared bottom one.
        labels = {
            t.get_text() for ax in figure.axes for t in ax.get_xticklabels()
        }
        assert set(matrix.columns) <= labels
    finally:
        plt.close(figure)


def test_design_matrix_figure_labels_regressor_groups() -> None:
    figure = design.design_matrix_figure(_design_frame())
    try:
        annotations = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
        assert "Task (2)" in annotations
        assert "Confound (2)" in annotations
        assert "Drift (2)" in annotations
    finally:
        plt.close(figure)


def test_design_matrix_figure_renders_without_a_contrast() -> None:
    figure = design.design_matrix_figure(_design_frame())
    try:
        rendered = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
        assert "no contrast supplied" in rendered
    finally:
        plt.close(figure)


def test_a_contrast_naming_no_real_regressor_is_refused() -> None:
    """A dict contrast maps by name, so a typo would silently weight nothing.

    The array-based predecessor could not express this mistake -- a wrong length
    raised. Mapping by name removes that check, so it has to be reinstated: an
    all-zero contrast strip drawn from a misspelt key looks exactly like a
    legitimately empty contrast.
    """
    with pytest.raises(ValueError, match="cond_typo"):
        design.design_matrix_figure(
            _design_frame(), contrast={"cond_typo": 1.0, "cond_b": -1.0}
        )


def test_correlation_figure_states_the_worst_pair() -> None:
    figure = design.regressor_correlation_figure(_design_frame(), run_label="run-01")
    try:
        footer = " ".join(t.get_text() for t in figure.texts)
        assert "largest |r| off the diagonal" in footer
        assert "constant term excluded" in footer
    finally:
        plt.close(figure)


def test_correlation_does_not_emit_nan_warnings_for_the_constant(recwarn) -> None:
    # Every design has a constant column, and np.corrcoef divides by its zero
    # standard deviation. Unhandled, that renders as blank cells that look like
    # data which failed to load.
    design.regressor_correlation_figure(_design_frame())
    divides = [w for w in recwarn if "divide" in str(w.message)]
    assert divides == []
    plt.close("all")


def test_vif_figure_renders_and_explains_its_measure() -> None:
    figure = design.variance_inflation_figure(_design_frame(), run_label="run-01")
    try:
        footer = " ".join(t.get_text() for t in figure.texts)
        assert "VIF = 1/(1 - R" in footer
        assert "constant term excluded" in footer
    finally:
        plt.close(figure)


def test_vif_figure_uses_a_log_axis_so_a_huge_vif_stays_visible() -> None:
    figure = design.variance_inflation_figure(_design_frame())
    try:
        assert figure.axes[0].get_yscale() == "log"
    finally:
        plt.close(figure)


def test_vif_figure_draws_one_bar_per_modelled_regressor() -> None:
    frame = _design_frame()
    figure = design.variance_inflation_figure(frame)
    try:
        # The constant is excluded from the modelled set, hence the -1.
        assert len(figure.axes[0].patches) == len(frame.columns) - 1
    finally:
        plt.close(figure)


def test_figures_drop_label_clutter_only_past_a_screen_width() -> None:
    # These panels used to drop every name past a couple of dozen columns. This
    # pipeline's designs carry 47, so in practice the VIF panel drew 46 unlabelled
    # bars: a reader could see something inflated by a factor of 130 and had no way
    # to find out what. The panels widen instead, and only fall back on role bands
    # past a width no screen carries.
    rng = np.random.default_rng(5)
    n_over = design.MAX_LABELLED_REGRESSORS + 10
    wide = pd.DataFrame(
        {f"a_comp_cor_{i:03d}": rng.normal(size=n_over + 20) for i in range(n_over)}
    )
    wide["constant"] = 1.0

    matrix_fig = design.design_matrix_figure(wide)
    corr_fig = design.regressor_correlation_figure(wide)
    try:
        assert len(matrix_fig.axes[0].get_xticks()) == 0
        assert "modelled regressors" in corr_fig.axes[0].get_xlabel()
    finally:
        plt.close(matrix_fig)
        plt.close(corr_fig)


def test_a_realistic_confound_design_keeps_every_regressor_name() -> None:
    # 47 regressors is this study's own design: six task columns, a motion-24 block,
    # eight drift terms, and a constant.
    rng = np.random.default_rng(6)
    columns = {f"task_{i}": rng.normal(size=120) for i in range(6)}
    columns.update({f"trans_{i:02d}": rng.normal(size=120) for i in range(32)})
    columns.update({f"drift_{i}": rng.normal(size=120) for i in range(8)})
    columns["constant"] = np.ones(120)
    frame = pd.DataFrame(columns)
    assert frame.shape[1] == 47

    figure = design.variance_inflation_figure(frame)
    figure.canvas.draw()
    labels = {t.get_text() for t in figure.axes[0].get_xticklabels() if t.get_text()}
    assert "trans_31" in labels and "drift_7" in labels and "task_0" in labels
    plt.close(figure)


def test_the_vif_panel_marks_the_contrast_s_own_regressors_in_its_labels() -> None:
    # Inflation on a regressor the contrast weights is what costs the comparison its
    # precision; inflation on a motion derivative's square is ordinary. A reader
    # scanning names should not have to match a bar back to a colour swatch.
    frame = _wide_frame()
    figure = design.variance_inflation_figure(frame, contrast={"task_0": 1.0})
    figure.canvas.draw()
    weighted = [
        t for t in figure.axes[0].get_xticklabels() if t.get_text() == "task_0"
    ]
    assert weighted, "the weighted regressor lost its label"
    assert weighted[0].get_fontweight() == "bold"
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def test_summary_reports_conditioning_and_efficiency() -> None:
    summary = design.summarize_design(
        _design_frame(), contrast={"cond_a": 1.0, "cond_b": -1.0}
    )
    assert summary.n_scans == 120
    assert summary.n_regressors == 7
    assert summary.condition_number > 1
    assert summary.max_vif is not None
    assert summary.efficiency is not None


def test_summary_survives_a_singular_design() -> None:
    """A rank-deficient design is exactly the case these figures exist to expose."""
    frame = _design_frame()
    frame["duplicate"] = frame["cond_a"]
    summary = design.summarize_design(frame, contrast={"cond_a": 1.0, "cond_b": -1.0})
    assert summary.max_vif is not None
    assert not np.isfinite(summary.max_vif) or summary.max_vif > 100


def test_summary_reports_the_residual_degrees_of_freedom() -> None:
    # The denominator of every t this design produces. A map can look decisive on very
    # few, and nothing else in the report reveals it.
    summary = design.summarize_design(_design_frame())
    assert summary.rank == 7
    assert summary.residual_dof == 120 - 7


def test_the_residual_dof_follows_the_rank_not_the_column_count() -> None:
    # A duplicated column adds no parameter, so it costs no degree of freedom. Taking
    # the column count would understate what is left to estimate the variance with.
    frame = _design_frame()
    frame["duplicate"] = frame["cond_a"]
    summary = design.summarize_design(frame)
    assert summary.n_regressors == 8
    assert summary.rank == 7
    assert summary.residual_dof == 120 - 7


def test_display_scaling_is_per_column() -> None:
    """Motion in millimetres and task in HRF units cannot share one colour scale."""
    matrix = np.column_stack([np.array([0.0, 1.0, 2.0]), np.array([0.0, 100.0, 200.0])])
    scaled = design._display_scaled(matrix)
    np.testing.assert_allclose(scaled[:, 0], scaled[:, 1])
    assert scaled.max() == pytest.approx(1.0)


# --- label legibility -----------------------------------------------------


def _visible_tick_labels(figure):
    """Tick labels a reader actually sees, with their rotation."""
    figure.canvas.draw()
    seen = []
    for axes in figure.axes:
        for label in axes.get_xticklabels():
            if label.get_visible() and label.get_text():
                seen.append((label.get_text(), label.get_rotation()))
    return seen


def test_the_regressor_labels_a_reader_sees_are_rotated() -> None:
    # Under sharex the upper axes' labels are hidden and the lower axes renders its
    # own from the shared locator -- without the rotation, which belongs to the Text
    # objects on the axes it was set on. Set on the wrong axes these came out
    # horizontal and overlapped into an unreadable smear.
    frame = pd.DataFrame(
        {
            "pain": np.linspace(0, 1, 30),
            "nonpain": np.linspace(1, 0, 30),
            "trans_x": np.random.default_rng(0).standard_normal(30),
            "constant": np.ones(30),
        }
    )
    figure = design.design_matrix_figure(frame, contrast={"pain": 1.0, "nonpain": -1.0})
    labels = _visible_tick_labels(figure)
    assert labels, "the panel showed no regressor labels at all"
    assert all(rotation == 90 for _text, rotation in labels)
    assert {"pain", "nonpain"} <= {text for text, _rotation in labels}
    plt.close(figure)


def test_a_wide_design_names_its_nuisance_block_too() -> None:
    # The panel exists to say which weight lands on which regressor. A confound the
    # reader cannot name is one they cannot check the model for, so the figure
    # widens rather than dropping the nuisance labels.
    columns = {f"task_{i}": np.linspace(0, 1, 40) for i in range(3)}
    columns.update({f"trans_{i}": np.linspace(0, 1, 40) for i in range(30)})
    columns["constant"] = np.ones(40)
    figure = design.design_matrix_figure(pd.DataFrame(columns))
    texts = {text for text, _rotation in _visible_tick_labels(figure)}
    assert {"task_0", "task_1", "task_2"} <= texts
    assert any(text.startswith("trans_") for text in texts)
    plt.close(figure)


def test_a_wider_design_gets_a_wider_figure() -> None:
    # Width is what buys the labels; without it they overlap into a smear.
    def frame(n_confound: int) -> pd.DataFrame:
        columns = {"task_0": np.linspace(0, 1, 40)}
        columns.update({f"trans_{i}": np.linspace(0, 1, 40) for i in range(n_confound)})
        columns["constant"] = np.ones(40)
        return pd.DataFrame(columns)

    narrow = design.design_matrix_figure(frame(4))
    wide = design.design_matrix_figure(frame(44))
    try:
        assert wide.get_size_inches()[0] > narrow.get_size_inches()[0]
    finally:
        plt.close(narrow)
        plt.close(wide)


def test_the_contrast_weight_is_written_on_the_cell_it_belongs_to() -> None:
    # Not in a caption below the strip: the column is already named by the tick label
    # underneath, so a caption repeated the name and collided with it.
    frame = pd.DataFrame(
        {
            "pain": np.linspace(0, 1, 30),
            "nonpain": np.linspace(1, 0, 30),
            "constant": np.ones(30),
        }
    )
    figure = design.design_matrix_figure(frame, contrast={"pain": 1.0, "nonpain": -1.0})
    strip = figure.axes[1]
    weights = {t.get_text() for t in strip.texts}
    assert weights == {"+1", "-1"}
    # On the cell, vertically centred in the strip.
    assert all(t.get_position()[1] == pytest.approx(0.5) for t in strip.texts)
    plt.close(figure)


def test_a_contrast_over_many_regressors_is_left_to_its_colour() -> None:
    # Past a point the cells are narrower than the digits, and a number rendered over
    # the wrong cell is worse than none.
    columns = {f"c{i}": np.random.default_rng(i).standard_normal(60) for i in range(40)}
    frame = pd.DataFrame(columns)
    contrast = {f"c{i}": 1.0 for i in range(40)}
    figure = design.design_matrix_figure(frame, contrast=contrast)
    assert not figure.axes[1].texts
    plt.close(figure)


# --- naming which regressor is the problem --------------------------------


def _wide_frame(n_task: int = 3, n_confound: int = 30, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    columns = {f"task_{i}": rng.standard_normal(n) for i in range(n_task)}
    columns.update({f"trans_{i}": rng.standard_normal(n) for i in range(n_confound)})
    columns["constant"] = np.ones(n)
    return pd.DataFrame(columns)


def _provenance(figure) -> str:
    return " ".join(artist.get_text() for artist in figure.texts)


def test_the_vif_panel_names_its_worst_regressor() -> None:
    # Unlabelled bars reduce the panel to "some regressor is inflated", which is not
    # actionable: whether it matters depends entirely on which.
    frame = _wide_frame()
    frame["trans_7"] = frame["trans_3"] * 2.0 + 1e-9 * np.arange(len(frame))
    figure = design.variance_inflation_figure(frame)
    assert "largest VIF:" in _provenance(figure)
    assert "trans_" in _provenance(figure)
    plt.close(figure)


def test_the_vif_panel_bands_a_wide_design_by_role() -> None:
    # A VIF of 130 on a motion derivative's square is ordinary; the same number on a
    # task regressor is not.
    figure = design.variance_inflation_figure(_wide_frame())
    labels = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "Task" in labels and "Confound" in labels
    plt.close(figure)


def test_the_vif_panel_marks_the_regressors_this_contrast_weights() -> None:
    frame = _wide_frame()
    figure = design.variance_inflation_figure(
        frame, contrast={"task_0": 1.0, "task_1": -1.0}
    )
    assert "weighted by this contrast" in " ".join(
        t.get_text() for t in figure.axes[0].texts
    )
    plt.close(figure)


def test_the_vif_panel_reports_the_worst_among_the_weighted_regressors() -> None:
    # The one that actually costs this contrast its precision.
    frame = _wide_frame()
    frame["task_1"] = frame["task_0"] * 3.0 + 1e-9 * np.arange(len(frame))
    figure = design.variance_inflation_figure(
        frame, contrast={"task_0": 1.0, "task_1": -1.0}
    )
    assert "largest among weighted:" in _provenance(figure)
    plt.close(figure)


def test_the_vif_panel_makes_no_contrast_claim_without_a_contrast() -> None:
    figure = design.variance_inflation_figure(_wide_frame())
    assert "weighted" not in _provenance(figure)
    plt.close(figure)


def test_the_correlation_panel_names_the_pair_behind_its_worst_r() -> None:
    # "Largest |r| off the diagonal: 0.98" says two columns duplicate each other but
    # not which two, and a design too wide to label has nowhere else to say it.
    frame = _wide_frame()
    frame["trans_9"] = frame["trans_2"] * -1.0
    figure = design.regressor_correlation_figure(frame)
    text = _provenance(figure)
    assert "1.00" in text
    assert "trans_2" in text and "trans_9" in text
    plt.close(figure)


# --- degenerate designs ---------------------------------------------------


def test_the_correlation_panel_survives_an_intercept_only_design() -> None:
    # A one-sample second-level design is intercept-only, so excluding the constant
    # leaves no columns, and np.corrcoef of that returns a 0-d array imshow rejects.
    # That is the commonest group analysis there is.
    figure = design.regressor_correlation_figure(pd.DataFrame({"intercept": np.ones(10)}))
    text = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "nothing to correlate" in text
    plt.close(figure)


def test_the_correlation_panel_survives_a_single_modelled_regressor() -> None:
    frame = pd.DataFrame({"intercept": np.ones(10), "group": np.r_[np.ones(5), -np.ones(5)]})
    figure = design.regressor_correlation_figure(frame)
    text = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "nothing to correlate" in text
    plt.close(figure)


def test_two_modelled_regressors_still_get_a_real_matrix() -> None:
    frame = pd.DataFrame(
        {
            "intercept": np.ones(20),
            "group": np.r_[np.ones(10), -np.ones(10)],
            "age": np.linspace(20, 60, 20),
        }
    )
    figure = design.regressor_correlation_figure(frame)
    assert figure.axes[0].images, "expected a correlation matrix to be drawn"
    plt.close(figure)


def test_the_vif_panel_survives_an_intercept_only_design() -> None:
    figure = design.variance_inflation_figure(pd.DataFrame({"intercept": np.ones(10)}))
    assert figure is not None
    plt.close(figure)


def test_the_correlation_panel_bands_a_design_too_wide_to_label() -> None:
    # A hot off-diagonal block among the confounds is ordinary -- a motion parameter
    # and its own square are correlated by construction. The same block reaching the
    # task regressors is what costs the contrast its variance, and unbanded the two
    # are indistinguishable. Reached only once the design outgrows its labels.
    figure = design.regressor_correlation_figure(
        _wide_frame(n_confound=design.MAX_LABELLED_REGRESSORS + 10, n=140)
    )
    labels = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "Task" in labels and "Confound" in labels
    plt.close(figure)


def test_the_correlation_panel_names_a_realistic_design() -> None:
    figure = design.regressor_correlation_figure(_wide_frame())
    figure.canvas.draw()
    ticks = {t.get_text() for t in figure.axes[0].get_yticklabels() if t.get_text()}
    assert "task_0" in ticks and "trans_29" in ticks
    plt.close(figure)


def test_a_narrow_design_keeps_its_regressor_names_instead_of_bands() -> None:
    frame = pd.DataFrame(
        {
            "cond_a": np.linspace(0, 1, 40),
            "cond_b": np.linspace(1, 0, 40),
            "trans_x": np.random.default_rng(0).standard_normal(40),
            "constant": np.ones(40),
        }
    )
    figure = design.regressor_correlation_figure(frame)
    figure.canvas.draw()
    ticks = {t.get_text() for t in figure.axes[0].get_yticklabels() if t.get_text()}
    assert {"cond_a", "cond_b", "trans_x"} <= ticks
    plt.close(figure)


# --- the per-run summary as one table -------------------------------------
#
# Six runs produced six stacked key-value blocks of the same seven labels, and
# comparing a condition number across runs meant scrolling between them -- while
# comparison across runs is the entire reason those numbers are reported per run.


def _summaries(n: int = 3):
    frame = _design_frame()
    return [
        design.summarize_design(frame, contrast={"cond_a": 1.0, "cond_b": -1.0})
        for _ in range(n)
    ]


def test_one_row_per_run() -> None:
    table, rows = design.design_summary_table(
        _summaries(3), run_labels=["run-01", "run-02", "run-03"]
    )
    assert table.count("<tr>") == 4  # header plus three runs
    assert len(rows) == 4
    assert "run-02" in table


def test_the_table_carries_the_conditioning_and_the_efficiency() -> None:
    table, _rows = design.design_summary_table(_summaries(1), run_labels=["run-01"])
    for column in ("Condition number", "Largest VIF", "Efficiency", "Residual dof"):
        assert column in table


def test_a_rank_deficient_design_is_named_as_one() -> None:
    # A design whose columns are linearly dependent carries fewer parameters than it
    # appears to, and no other line in the report says so.
    summary = design.DesignSummary(
        n_scans=100,
        n_regressors=5,
        condition_number=1e9,
        max_vif=None,
        max_vif_regressor="",
        efficiency=None,
        rank=4,
        residual_dof=96,
    )
    table, _rows = design.design_summary_table([summary], run_labels=["run-01"])
    assert "4 (deficient)" in table


def test_an_inestimable_quantity_says_so_rather_than_showing_zero() -> None:
    summary = design.DesignSummary(
        n_scans=100,
        n_regressors=5,
        condition_number=float("inf"),
        max_vif=None,
        max_vif_regressor="",
        efficiency=None,
        rank=5,
        residual_dof=95,
    )
    table, _rows = design.design_summary_table([summary], run_labels=["run-01"])
    assert "not estimable" in table
    assert "∞" in table


# --- how much data the contrast rests on ----------------------------------


def test_events_are_counted_from_the_convolved_regressor() -> None:
    # The report described the model in every other respect and never said how much
    # data the contrast rested on.
    frame = pd.DataFrame(
        {
            "cond_a": np.concatenate([np.zeros(5), np.ones(4), np.zeros(6), np.ones(4), np.zeros(5)]),
            "cond_b": np.concatenate([np.zeros(12), np.ones(4), np.zeros(8)]),
            "constant": np.ones(24),
        }
    )
    counts = design.count_events(frame, ["cond_a", "cond_b"])
    assert counts == {"cond_a": 2, "cond_b": 1}


def test_a_condition_absent_from_a_run_is_not_counted_as_zero() -> None:
    # A run that never presented a condition is a different fact from a run that
    # presented it zero times, and the table shows "n/a" for the first.
    frame = pd.DataFrame({"cond_a": np.ones(10), "constant": np.ones(10)})
    assert design.count_events(frame, ["cond_a", "cond_b"]) == {"cond_a": 1}

    table, _rows = design.design_summary_table(
        _summaries(1),
        run_labels=["run-01"],
        event_counts=[{"cond_a": 5}],
        condition_names=["cond_a", "cond_b"],
    )
    assert "Events: cond_a" in table and "Events: cond_b" in table
    assert "n/a" in table


def test_event_counts_join_the_summary_table() -> None:
    table, rows = design.design_summary_table(
        _summaries(2),
        run_labels=["run-01", "run-02"],
        event_counts=[{"cond_a": 6}, {"cond_a": 5}],
        condition_names=["cond_a"],
    )
    assert "Events: cond_a" in table
    assert rows[1].endswith("\t6")
    assert rows[2].endswith("\t5")


def test_a_condition_high_from_the_first_frame_still_counts_as_an_event() -> None:
    # Such an event began at or before the first modelled frame: the design drops
    # non-steady-state volumes, so a run whose first trial starts immediately loses
    # its rising edge with them. Reporting no event would say the run contained none.
    assert list(design.onset_rows(np.concatenate([np.ones(4), np.zeros(6)]))) == [0]
    assert list(
        design.onset_rows(np.concatenate([np.ones(4), np.zeros(6), np.ones(4)]))
    ) == [0, 10]


# --- event timing ----------------------------------------------------------
#
# The summary table counts the events; it cannot show where they fell. Timing decides
# whether two conditions are separable at all, and no count or condition number
# reveals it.


def test_the_raster_draws_one_lane_per_run() -> None:
    onsets = [
        {"cond_a": np.array([10, 50]), "cond_b": np.array([30, 70])},
        {"cond_a": np.array([12, 52]), "cond_b": np.array([32])},
    ]
    figure = design.event_raster_figure(
        onsets, run_labels=["run-01", "run-02"], condition_names=["cond_a", "cond_b"]
    )
    figure.canvas.draw()
    ticks = [t.get_text() for t in figure.axes[0].get_yticklabels() if t.get_text()]
    assert ticks == ["run-01", "run-02"]
    plt.close(figure)


def test_the_raster_totals_each_condition() -> None:
    onsets = [
        {"cond_a": np.array([10, 50]), "cond_b": np.array([30])},
        {"cond_a": np.array([12]), "cond_b": np.array([32, 72])},
    ]
    figure = design.event_raster_figure(
        onsets, run_labels=["run-01", "run-02"], condition_names=["cond_a", "cond_b"]
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "cond_a: 3" in text and "cond_b: 3" in text
    plt.close(figure)


def test_the_raster_puts_its_axis_in_seconds_when_it_knows_the_tr() -> None:
    onsets = [{"cond_a": np.array([10])}]
    with_tr = design.event_raster_figure(
        onsets, run_labels=["run-01"], condition_names=["cond_a"], tr_seconds=2.0
    )
    without = design.event_raster_figure(
        onsets, run_labels=["run-01"], condition_names=["cond_a"]
    )
    assert "Time (s)" in with_tr.get_axes()[0].get_xlabel()
    assert "row" in without.get_axes()[0].get_xlabel().lower()
    plt.close(with_tr)
    plt.close(without)


def test_a_run_missing_a_condition_still_draws() -> None:
    onsets = [{"cond_a": np.array([10])}, {"cond_b": np.array([30])}]
    figure = design.event_raster_figure(
        onsets, run_labels=["run-01", "run-02"], condition_names=["cond_a", "cond_b"]
    )
    assert figure.axes
    plt.close(figure)


def test_a_raster_without_conditions_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one condition"):
        design.event_raster_figure([{}], run_labels=["run-01"], condition_names=[])


# --- one panel over all runs, not one per run ------------------------------
#
# Six runs drew six near-identical bar charts and six correlation matrices, and a
# reader comparing a regressor between them had to hold six pictures in mind. The
# between-run comparison is the reading, so it belongs on one axis -- and the spread
# across runs, which distinguishes a property of the design from a property of one
# run, was never shown at all.


def _run_frames(n: int = 4, seed: int = 11):
    rng = np.random.default_rng(seed)
    frames = []
    # Conditions as separate boxcars rather than a ramp and its reverse: the latter
    # are perfectly anti-correlated by construction, so they, not the pair injected
    # below, would always be the worst pair on the panel.
    cond_a = np.zeros(80)
    cond_a[5:15] = cond_a[35:45] = 1.0
    cond_b = np.zeros(80)
    cond_b[20:30] = cond_b[55:65] = 1.0
    for index in range(n):
        columns = {
            "cond_a": cond_a.copy(),
            "cond_b": cond_b.copy(),
            "trans_x": rng.standard_normal(80),
            "trans_y": rng.standard_normal(80),
            "constant": np.ones(80),
        }
        # One run alone carries a near-duplicate pair.
        if index == 2:
            columns["trans_y"] = columns["trans_x"] * 1.0 + 1e-6 * rng.standard_normal(80)
        frames.append(pd.DataFrame(columns))
    return frames


def test_the_vif_panel_covers_every_run_on_one_axis() -> None:
    figure = design.variance_inflation_across_runs_figure(
        _run_frames(), contrast={"cond_a": 1.0, "cond_b": -1.0}
    )
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "4 run(s)" in text
    assert "median across runs" in text and "range" in text
    plt.close(figure)


def test_the_vif_panel_shows_the_spread_across_runs() -> None:
    # A regressor inflated in one run only must be distinguishable from one inflated
    # in all of them, which a per-run panel could never show.
    figure = design.variance_inflation_across_runs_figure(_run_frames())
    spans = [
        line.get_ydata()
        for line in figure.axes[0].lines
        if len(line.get_ydata()) == 2
    ]
    assert any(high > 10 * low for low, high in spans)
    plt.close(figure)


def test_the_vif_across_runs_panel_names_its_worst_regressor() -> None:
    figure = design.variance_inflation_across_runs_figure(_run_frames())
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "largest median VIF:" in text
    plt.close(figure)


def test_only_regressors_present_in_every_run_are_compared() -> None:
    # A regressor one run lacks has no value to compare across runs, and padding it
    # would put a gap in a panel whose entire reading is between-run variation.
    frames = _run_frames(n=2)
    frames[1] = frames[1].drop(columns=["trans_y"])
    figure = design.variance_inflation_across_runs_figure(frames)
    figure.canvas.draw()
    labels = {t.get_text() for t in figure.axes[0].get_xticklabels() if t.get_text()}
    assert "trans_x" in labels and "trans_y" not in labels
    plt.close(figure)


def test_the_correlation_panel_takes_the_strongest_across_runs() -> None:
    # A pair collinear in a single run costs the contrast its precision in that run,
    # and an average across runs would dilute exactly that away.
    figure = design.regressor_correlation_across_runs_figure(_run_frames())
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "largest |r| off the diagonal: 1.00" in text
    assert "trans_x" in text and "trans_y" in text
    plt.close(figure)


def test_the_correlation_panel_is_a_magnitude_not_a_signed_value() -> None:
    # A sign taken from whichever run was most extreme would name a number no single
    # run holds, so the scale runs from zero.
    figure = design.regressor_correlation_across_runs_figure(_run_frames())
    image = figure.axes[0].get_images()[0]
    assert image.get_clim() == (0.0, 1.0)
    plt.close(figure)


def test_a_single_shared_regressor_leaves_nothing_to_correlate() -> None:
    frames = [pd.DataFrame({"cond_a": np.linspace(0, 1, 30), "constant": np.ones(30)})]
    figure = design.regressor_correlation_across_runs_figure(frames)
    text = " ".join(a.get_text() for ax in figure.axes for a in ax.texts)
    assert "nothing to correlate" in text
    plt.close(figure)


def test_no_shared_regressor_is_refused_rather_than_drawn_empty() -> None:
    frames = [
        pd.DataFrame({"a": np.linspace(0, 1, 20), "constant": np.ones(20)}),
        pd.DataFrame({"b": np.linspace(0, 1, 20), "constant": np.ones(20)}),
    ]
    with pytest.raises(ValueError, match="present in every run"):
        design.variance_inflation_across_runs_figure(frames)


def _vif_axis(figure, label):
    return next((ax for ax in figure.axes if ax.get_label() == label), None)


def test_the_contrast_regressors_get_their_own_axis() -> None:
    """VIF on the two regressors the contrast weights is what costs it precision.

    On sub-0001 those two sit at VIF 8-9 among 45 other bars, indistinguishable from
    regressors whose inflation costs the comparison nothing.
    """
    figure = design.variance_inflation_across_runs_figure(
        _run_frames(), contrast={"cond_a": 1.0, "cond_b": -1.0}
    )
    figure.canvas.draw()
    contrast_axis = _vif_axis(figure, "vif-contrast")
    assert contrast_axis is not None
    labels = {t.get_text() for t in contrast_axis.get_xticklabels() if t.get_text()}
    assert labels == {"cond_a", "cond_b"}
    plt.close(figure)


def test_the_weighted_regressors_are_absent_from_the_nuisance_axis() -> None:
    figure = design.variance_inflation_across_runs_figure(
        _run_frames(), contrast={"cond_a": 1.0, "cond_b": -1.0}
    )
    figure.canvas.draw()
    labels = {
        t.get_text() for t in _vif_axis(figure, "vif-rest").get_xticklabels() if t.get_text()
    }
    assert "cond_a" not in labels and "cond_b" not in labels
    assert "trans_x" in labels
    plt.close(figure)


def test_both_vif_axes_share_one_scale() -> None:
    """Split across two axes with different scales, the numbers stop being comparable."""
    figure = design.variance_inflation_across_runs_figure(
        _run_frames(), contrast={"cond_a": 1.0, "cond_b": -1.0}
    )
    figure.canvas.draw()
    top = _vif_axis(figure, "vif-contrast")
    rest = _vif_axis(figure, "vif-rest")
    assert top.get_yscale() == rest.get_yscale() == "log"
    assert top.get_ylim() == rest.get_ylim()
    plt.close(figure)


def test_no_split_without_a_contrast() -> None:
    """Unchanged single-axis panel when the contrast's columns are unknown."""
    figure = design.variance_inflation_across_runs_figure(_run_frames())
    assert _vif_axis(figure, "vif-contrast") is None
    plt.close(figure)


def test_no_split_when_the_contrast_weights_every_regressor() -> None:
    """An empty remainder axis would be a blank panel with a role band over it."""
    frames = _run_frames()
    contrast = {name: 1.0 for name in frames[0].columns}
    figure = design.variance_inflation_across_runs_figure(frames, contrast=contrast)
    assert _vif_axis(figure, "vif-contrast") is None
    plt.close(figure)


def test_contrast_confound_detects_time_on_task_correlation() -> None:
    """A contrast whose positive condition sits early correlates with elapsed time."""
    import numpy as np
    import pandas as pd

    n = 100
    early = np.zeros(n); early[:40] = 1.0
    late = np.zeros(n); late[60:] = 1.0
    frame = pd.DataFrame(
        {"cond_a": early, "cond_b": late, "drift_1": np.linspace(-1, 1, n)}
    )
    row = design.contrast_confounding(frame, {"cond_a": 1.0, "cond_b": -1.0})
    assert row["r_with_time"] < -0.5


def test_an_interleaved_design_has_little_time_correlation() -> None:
    import numpy as np
    import pandas as pd

    n = 100
    a = np.zeros(n); a[::10] = 1.0
    b = np.zeros(n); b[5::10] = 1.0
    frame = pd.DataFrame(
        {"cond_a": a, "cond_b": b, "drift_1": np.linspace(-1, 1, n)}
    )
    row = design.contrast_confounding(frame, {"cond_a": 1.0, "cond_b": -1.0})
    assert abs(row["r_with_time"]) < 0.2


def test_drift_correlation_is_the_strongest_over_the_basis() -> None:
    """One drift column absorbing the contrast is what costs it, not the average."""
    import numpy as np
    import pandas as pd

    n = 120
    ramp = np.linspace(-1, 1, n)
    frame = pd.DataFrame(
        {
            "cond_a": ramp,
            "drift_1": np.cos(np.pi * np.arange(n) / n),
            "drift_2": ramp,
        }
    )
    row = design.contrast_confounding(frame, {"cond_a": 1.0})
    assert row["r_with_drift_max"] > 0.99


def test_a_constant_contrast_regressor_yields_no_correlation() -> None:
    """No variance means no correlation; nan is the honest answer, not zero."""
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"cond_a": np.ones(50), "drift_1": np.linspace(-1, 1, 50)})
    row = design.contrast_confounding(frame, {"cond_a": 1.0})
    assert np.isnan(row["r_with_time"])


def test_a_design_without_drift_columns_reports_no_drift_correlation() -> None:
    import numpy as np
    import pandas as pd

    n = 60
    a = np.zeros(n); a[::6] = 1.0
    frame = pd.DataFrame({"cond_a": a})
    row = design.contrast_confounding(frame, {"cond_a": 1.0})
    assert np.isnan(row["r_with_drift_max"])


def test_the_summary_table_carries_the_confound_columns() -> None:
    frames = _run_frames(n=2)
    summaries = [design.summarize_design(f, contrast=None) for f in frames]
    table, rows = design.design_summary_table(
        summaries,
        run_labels=["run-01", "run-02"],
        confounding=[
            design.contrast_confounding(f, {"cond_a": 1.0, "cond_b": -1.0})
            for f in frames
        ],
    )
    assert "r with elapsed time" in table
    assert "r with drift" in table
    assert any("r with elapsed time" in row for row in rows)
