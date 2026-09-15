"""The staged residual estimator must be fitted inside each training split it serves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from eeg_pipeline.analysis.machine_learning.orchestration import (
    model_comparison_cv_predictions,
)
from tests.utils.pipelines_test_utils import DotConfig


def _staged_config() -> DotConfig:
    return DotConfig(
        {
            "machine_learning": {
                "target_residualization": {"strategy": "staged_residual_learning"},
                "preprocessing": {"imputer_strategy": "median"},
            }
        }
    )


def _dataset(*, n_subjects: int = 5, n_trials: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_subjects * n_trials
    nuisance = rng.normal(size=n)
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    y = 10.0 + 2.0 * nuisance + 3.0 * first + 1.5 * second + rng.normal(scale=0.1, size=n)
    groups = np.repeat([f"sub-{i:04d}" for i in range(n_subjects)], n_trials).astype(object)
    meta = pd.DataFrame({"nuisance": nuisance})
    X = np.column_stack([first, second])
    return X, y, groups, meta


def _loso_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.flatnonzero(groups != held), np.flatnonzero(groups == held))
        for held in np.unique(groups)
    ]


def _mean_delta_r2(**kwargs) -> float:
    result = model_comparison_cv_predictions(**kwargs)
    return float(np.mean([record["delta_r2"] for record in result.records]))


def test_one_missing_value_does_not_delete_the_feature(monkeypatch) -> None:
    """A single NaN must cost one cell, not an entire documented feature.

    The nuisance regression is a least-squares solve: a NaN anywhere in a column makes
    every fitted coefficient and every residual in that column NaN, and intersection
    harmonization then drops the column outright. The missingness policy and the
    training-median imputation have to run before the solve, not after it.
    """
    from eeg_pipeline.analysis.machine_learning import orchestration

    X, y, groups, meta = _dataset()
    folds = _loso_folds(groups)

    widths: list[int] = []
    original = orchestration._fit_subject_weighted_inner_cv_estimator

    def _record_width(**kwargs):
        widths.append(int(np.asarray(kwargs["X_train"]).shape[1]))
        return original(**kwargs)

    monkeypatch.setattr(orchestration, "_fit_subject_weighted_inner_cv_estimator", _record_width)

    call = dict(
        model_name="linear",
        pipe=LinearRegression(),
        param_grid={},
        y=y,
        groups=groups,
        meta=meta,
        outer_folds=folds,
        inner_splits=2,
        outer_jobs=1,
        config=_staged_config(),
        harmonization_mode="intersection",
        covariates=None,
        target_residualization_columns=("nuisance",),
        collect_records=True,
    )

    complete = _mean_delta_r2(X=X, **call)
    complete_widths = list(widths)

    widths.clear()
    holed = X.copy()
    holed[0, 0] = np.nan  # 1 value of 250, inside the documented 5% allowance
    with_hole = _mean_delta_r2(X=holed, **call)

    assert complete_widths == widths
    assert all(width == 2 for width in widths)
    assert with_hole == pytest.approx(complete, abs=0.05)


def test_inner_hyperparameter_selection_does_not_see_its_validation_targets(
    monkeypatch,
) -> None:
    """Nuisance coefficients and the Yeo-Johnson fit belong to the inner training split.

    Learning them from the whole outer training cohort lets each inner validation
    participant's own target shape the transformed targets that select regularization.
    """
    from eeg_pipeline.analysis.machine_learning import orchestration

    X, y, groups, meta = _dataset()
    folds = _loso_folds(groups)

    fitted_row_counts: list[int] = []
    original = orchestration._fit_staged_residual_preprocessor

    def _record_rows(**kwargs):
        fitted_row_counts.append(int(np.asarray(kwargs["rows"]).size))
        return original(**kwargs)

    monkeypatch.setattr(orchestration, "_fit_staged_residual_preprocessor", _record_rows)

    model_comparison_cv_predictions(
        model_name="linear",
        pipe=LinearRegression(),
        param_grid={"fit_intercept": [True, False]},
        X=X,
        y=y,
        groups=groups,
        meta=meta,
        outer_folds=folds[:1],
        inner_splits=2,
        outer_jobs=1,
        config=_staged_config(),
        harmonization_mode="none",
        covariates=None,
        target_residualization_columns=("nuisance",),
        collect_records=True,
    )

    outer_train_rows = int(folds[0][0].size)
    # One fit per inner training split per candidate, then one on the whole outer split.
    assert any(count < outer_train_rows for count in fitted_row_counts)
    assert fitted_row_counts.count(outer_train_rows) == 1


@pytest.mark.parametrize("missing_subject", ["sub-0000", "sub-0001"])
def test_staged_imputation_rejects_excessive_subject_missingness(missing_subject) -> None:
    """A subject cannot pass eligibility just because other subjects have complete EEG."""
    X, y, groups, meta = _dataset(n_subjects=30, n_trials=10)
    X[groups == missing_subject] = np.nan

    with pytest.raises(ValueError, match=f"Missingness limit exceeded: subject {missing_subject}"):
        model_comparison_cv_predictions(
            model_name="linear",
            pipe=LinearRegression(),
            param_grid={},
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            outer_folds=_loso_folds(groups)[:1],
            inner_splits=2,
            outer_jobs=1,
            config=_staged_config(),
            harmonization_mode="intersection",
            covariates=None,
            target_residualization_columns=("nuisance",),
            collect_records=True,
            fixed_params={},
        )
