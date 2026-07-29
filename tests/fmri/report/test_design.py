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
        labels = " ".join(t.get_text() for t in contrast_ax.texts)
        assert "cond_a" in labels and "+1" in labels
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
