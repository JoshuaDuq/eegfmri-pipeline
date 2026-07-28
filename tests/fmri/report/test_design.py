from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.figures import design


def _design(n_frames: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "task_a": rng.standard_normal(n_frames),
            "task_b": rng.standard_normal(n_frames),
            "trans_x": rng.standard_normal(n_frames),
            "drift_1": np.linspace(0, 1, n_frames),
            "constant": np.ones(n_frames),
        }
    )


def test_vif_is_infinite_for_a_perfectly_collinear_column() -> None:
    base = np.random.default_rng(0).standard_normal((40, 2))
    X = np.column_stack([base, base[:, 0]])
    assert np.isinf(design.vif_from_design(X)[2])


def test_vif_is_near_one_for_independent_columns() -> None:
    X = np.random.default_rng(0).standard_normal((400, 3))
    assert np.all(design.vif_from_design(X) < 1.5)


def test_vif_returns_empty_for_a_single_column() -> None:
    assert design.vif_from_design(np.ones((10, 1))).size == 0


def test_vif_does_not_overflow_on_a_realistic_design_with_an_intercept(recwarn) -> None:
    # The design's own constant column plus the intercept added by the regression
    # makes the system singular: lstsq returns an enormous beta and the residual
    # sum overflows. Only shows up with a realistic number of regressors.
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack(
        [rng.standard_normal((n, 8)), np.linspace(0, 1, n), np.ones(n)]
    )
    vif = design.vif_from_design(X)
    numeric = [w for w in recwarn if "overflow" in str(w.message) or "divide" in str(w.message)]
    assert numeric == []
    assert np.isinf(vif[-1])  # the intercept
    assert np.all(np.isfinite(vif[:-1]))


def test_design_matrix_figure_labels_every_regressor() -> None:
    matrix = _design()
    figure = design.design_matrix_figure(matrix)
    labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
    assert set(matrix.columns) <= set(labels)
    plt.close(figure)


def test_design_matrix_figure_draws_a_contrast_strip_when_given_one() -> None:
    matrix = _design()
    figure = design.design_matrix_figure(
        matrix, contrast=np.array([1.0, -1.0, 0.0, 0.0, 0.0]), contrast_name="a - b"
    )
    assert len(figure.axes) >= 2
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "a - b" in text
    plt.close(figure)


def test_design_matrix_figure_rejects_a_contrast_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="columns"):
        design.design_matrix_figure(_design(), contrast=np.array([1.0, -1.0]))


def test_collinearity_figure_reports_a_vif_bar_per_regressor() -> None:
    matrix = _design()
    figure = design.collinearity_figure(matrix)
    vif_axis = figure.axes[0]
    assert len(vif_axis.patches) == len(matrix.columns)
    plt.close(figure)


def test_collinearity_figure_uses_a_log_axis_so_infinite_vif_is_visible() -> None:
    figure = design.collinearity_figure(_design())
    assert figure.axes[0].get_xscale() == "log"
    plt.close(figure)


def test_correlation_does_not_emit_nan_warnings_for_the_intercept(recwarn) -> None:
    # Every design has a constant column, and np.corrcoef divides by its zero
    # standard deviation. Unhandled, that renders as blank cells that look like
    # data which failed to load.
    design.collinearity_figure(_design())
    divides = [w for w in recwarn if "divide" in str(w.message)]
    assert divides == []
    plt.close("all")


def test_constant_regressors_are_named_as_undefined_rather_than_left_blank() -> None:
    figure = design.collinearity_figure(_design())
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "undefined" in text and "constant" in text
    plt.close(figure)


def test_correlation_is_still_computed_for_the_varying_regressors() -> None:
    matrix = _design()
    correlation, constant = design._regressor_correlation(matrix.to_numpy(dtype=float))
    assert constant.tolist() == [False, False, False, False, True]
    # The 4x4 varying block is fully populated; only the constant row/column is NaN.
    assert np.all(np.isfinite(correlation[:4, :4]))
    assert np.all(np.isnan(correlation[4, :]))
