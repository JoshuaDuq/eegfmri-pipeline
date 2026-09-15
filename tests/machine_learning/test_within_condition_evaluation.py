from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.machine_learning.orchestration import (
    ModelComparisonPredictions,
    _within_condition_prediction_metrics,
    _within_subject_centered_prediction_metrics,
)

TEMPERATURES = np.asarray([45.3, 45.3, 45.3, 49.3, 49.3, 49.3], dtype=float)


def _predictions(
    target: np.ndarray, full: np.ndarray, nuisance: np.ndarray
) -> ModelComparisonPredictions:
    return ModelComparisonPredictions(
        evaluation_target=target,
        full_prediction=full,
        nuisance_prediction=nuisance,
        residual_prediction=full - nuisance,
        records=(),
    )


def _cohort(n_subjects: int = 3) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    groups = np.repeat([f"sub-{index:04d}" for index in range(n_subjects)], len(TEMPERATURES))
    temperatures = np.tile(TEMPERATURES, n_subjects)
    return groups, pd.DataFrame({"stimulus_temp": temperatures}), temperatures


def test_participant_temperature_slope_scores_within_subject_but_not_within_condition() -> None:
    groups, meta, temperatures = _cohort()
    rng = np.random.default_rng(7)
    # Participants differ sharply in how steeply they respond to temperature.
    slope = np.repeat([1.0, 5.0, 9.0], len(TEMPERATURES))
    offset = np.repeat([-4.0, 0.0, 4.0], len(TEMPERATURES))
    fluctuation = rng.normal(scale=0.5, size=len(groups))
    target = offset + slope * (temperatures - 45.3) + fluctuation
    # A prediction that only knows this participant's temperature response.
    full = offset + slope * (temperatures - 45.3)
    # The nuisance design carries only the pooled temperature effect.
    nuisance = offset + float(np.mean(slope)) * (temperatures - 45.3)

    centered = _within_subject_centered_prediction_metrics(
        _predictions(target, full, nuisance), groups
    )
    conditioned = _within_condition_prediction_metrics(
        _predictions(target, full, nuisance), groups, meta, {}
    )

    assert centered["within_subject_centered_delta_r2"] > 0.2
    assert conditioned["within_condition_centered_delta_r2"] == pytest.approx(0.0)
    assert conditioned["within_condition_centered_n_subjects"] == 3
    assert conditioned["within_condition_centered_n_trials"] == len(groups)


def test_within_condition_credits_prediction_of_fixed_temperature_fluctuations() -> None:
    groups, meta, temperatures = _cohort()
    rng = np.random.default_rng(11)
    fluctuation = rng.normal(size=len(groups))
    nuisance = 2.0 * (temperatures - 45.3)
    target = nuisance + fluctuation
    full = nuisance + fluctuation

    conditioned = _within_condition_prediction_metrics(
        _predictions(target, full, nuisance), groups, meta, {}
    )

    assert conditioned["within_condition_centered_delta_r2"] == pytest.approx(1.0)


def test_within_condition_ignores_conditions_a_participant_saw_once() -> None:
    groups = np.asarray(["sub-0001"] * 5, dtype=object)
    meta = pd.DataFrame({"stimulus_temp": [45.3, 45.3, 45.3, 49.3, 47.3]})
    target = np.asarray([1.0, 2.0, 3.0, 9.0, 4.0], dtype=float)
    nuisance = np.zeros(5, dtype=float)

    conditioned = _within_condition_prediction_metrics(
        _predictions(target, target, nuisance), groups, meta, {}
    )

    assert conditioned["within_condition_centered_n_trials"] == 3


def test_within_condition_reports_nothing_without_the_condition_column() -> None:
    groups, _meta, _temperatures = _cohort()
    values = np.arange(len(groups), dtype=float)

    conditioned = _within_condition_prediction_metrics(
        _predictions(values, values, np.zeros_like(values)),
        groups,
        pd.DataFrame({"run": np.ones(len(groups))}),
        {},
    )

    assert np.isnan(conditioned["within_condition_centered_delta_r2"])
    assert conditioned["within_condition_centered_n_subjects"] == 0
