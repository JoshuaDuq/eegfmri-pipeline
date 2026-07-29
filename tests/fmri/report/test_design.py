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


def test_figures_drop_label_clutter_for_wide_designs() -> None:
    rng = np.random.default_rng(5)
    wide = pd.DataFrame({f"a_comp_cor_{i:02d}": rng.normal(size=80) for i in range(70)})
    wide["constant"] = 1.0

    matrix_fig = design.design_matrix_figure(wide)
    corr_fig = design.regressor_correlation_figure(wide)
    try:
        assert len(matrix_fig.axes[0].get_xticks()) == 0
        assert "modelled regressors" in corr_fig.axes[0].get_xlabel()
    finally:
        plt.close(matrix_fig)
        plt.close(corr_fig)


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


def test_a_wide_design_labels_its_task_regressors_and_not_its_nuisance_block() -> None:
    columns = {f"task_{i}": np.linspace(0, 1, 40) for i in range(3)}
    columns.update({f"trans_{i}": np.linspace(0, 1, 40) for i in range(30)})
    columns["constant"] = np.ones(40)
    figure = design.design_matrix_figure(pd.DataFrame(columns))
    texts = {text for text, _rotation in _visible_tick_labels(figure)}
    assert {"task_0", "task_1", "task_2"} <= texts
    assert not any(text.startswith("trans_") for text in texts)
    plt.close(figure)


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
