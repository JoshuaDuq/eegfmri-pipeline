"""Tests for the Study 2 target-retrained null driver."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from unittest.mock import patch

from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.target_permutations import InvalidPermutationDraw
from studies.pain_study.study2.target_retrained_null import (
    Study1ModelContext,
    assemble_permuted_scores,
    build_target_retrained_null_maps,
    stacked_permuted_targets,
    standardize_scores_within_subject,
)
from studies.tests.pipelines.test_study2_source_stage import _source_stage_frame

BANDS = ("alpha", "beta", "gamma")
NUISANCE_COLUMNS = ("onset",)


def _aligned_context(frame, config):
    rng = np.random.default_rng(0)
    groups = frame["subject_id"].to_numpy(dtype=object)
    onset = frame["onset"].to_numpy(dtype=float)
    feature = rng.normal(size=len(frame))
    y = 2.0 * onset + 1.5 * feature + rng.normal(scale=0.1, size=len(frame))
    subjects = list(dict.fromkeys(groups.tolist()))
    outer_folds = tuple(
        (
            np.flatnonzero(groups != subject),
            np.flatnonzero(groups == subject),
        )
        for subject in subjects
    )
    config["machine_learning"] = {
        "target_residualization": {"strategy": "staged_residual_learning"}
    }
    return Study1ModelContext(
        X=feature.reshape(-1, 1),
        feature_names=("feature",),
        y=y,
        groups=groups,
        meta=frame,
        outer_folds=outer_folds,
        fixed_params_by_fold=tuple({} for _ in outer_folds),
        pipe=LinearRegression(),
        param_grid={},
        config=config,
        target_residualization_columns=NUISANCE_COLUMNS,
        runs=None,
        trial_indices=None,
        scheme="within_subject",
        inner_splits=2,
        harmonization_mode="none",
        covariates=None,
    )


def test_stacked_permuted_targets_returns_one_row_per_fold() -> None:
    frame = _cohort_frame()
    context = _aligned_context(frame, load_study2_config())

    stacked = stacked_permuted_targets(context, np.random.default_rng(1))

    assert stacked.shape == (len(context.outer_folds), len(context.y))
    assert np.all(np.isfinite(stacked))


def test_stacked_permuted_targets_reuses_one_mapping_across_folds() -> None:
    frame = _cohort_frame()
    context = _aligned_context(frame, load_study2_config())
    permutation_indices = np.arange(len(context.y), dtype=int)

    with (
        patch(
            "studies.pain_study.study2.target_retrained_null._permutation_indices_by_scheme",
            return_value=permutation_indices,
        ) as sample_indices,
        patch(
            "studies.pain_study.study2.target_retrained_null.reconstruct_staged_permutation_target_for_fold",
            return_value=context.y,
        ) as reconstruct,
    ):
        stacked_permuted_targets(context, np.random.default_rng(1))

    assert sample_indices.call_count == 1
    assert reconstruct.call_count == len(context.outer_folds)
    for call in reconstruct.call_args_list:
        np.testing.assert_array_equal(call.kwargs["permutation_indices"], permutation_indices)


def test_standardize_scores_within_subject_raises_on_zero_variance() -> None:
    scores = np.array([1.0, 1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    groups = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)

    with pytest.raises(InvalidPermutationDraw, match="zero variance"):
        standardize_scores_within_subject(scores, groups)


def test_assemble_permuted_scores_uses_eeg_residual_prediction() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        ModelComparisonPredictions,
    )

    frame = _cohort_frame()
    context = _aligned_context(frame, load_study2_config())
    residual_prediction = np.arange(len(context.y), dtype=float)
    prediction_result = ModelComparisonPredictions(
        evaluation_target=context.y,
        full_prediction=100.0 + residual_prediction,
        nuisance_prediction=np.full(len(context.y), 100.0, dtype=float),
        residual_prediction=residual_prediction,
        records=tuple(),
    )
    stacked_targets = np.tile(context.y, (len(context.outer_folds), 1))

    with patch(
        "studies.pain_study.study2.target_retrained_null.model_comparison_cv_predictions",
        return_value=prediction_result,
    ):
        scores = assemble_permuted_scores(context, stacked_targets)

    np.testing.assert_array_equal(scores, residual_prediction)


def test_standardize_scores_within_subject_unit_variance_per_subject() -> None:
    scores = np.array([0.0, 2.0, 4.0, 10.0, 20.0, 30.0], dtype=float)
    groups = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)

    standardized = standardize_scores_within_subject(scores, groups)

    for subject in ("a", "b"):
        block = standardized[groups == subject]
        assert float(np.mean(block)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.std(block, ddof=0)) == pytest.approx(1.0)


def test_build_target_retrained_null_maps_produces_per_band_null_arrays() -> None:
    frame = _cohort_frame()
    config = load_study2_config()
    context = _aligned_context(frame, config)
    source_power = _all_valid_source_power(frame)
    source_power_by_band = {band: source_power for band in BANDS}

    null_by_band, result = build_target_retrained_null_maps(
        context=context,
        source_power_by_band=source_power_by_band,
        score_frame_template=frame,
        bands=BANDS,
        expected_subject_ids=("sub-0001", "sub-0002"),
        n_valid_draws=3,
        max_invalid_fraction=0.8,
        random_state=7,
    )

    assert result.n_valid_draws == 3
    assert set(null_by_band) == set(BANDS)
    for band in BANDS:
        maps = null_by_band[band]
        assert maps.shape == (3, 2, 3)
        assert np.all(np.isfinite(maps))


def _all_valid_source_power(frame) -> dict[str, np.ndarray]:
    """Per-subject source power with every vertex carrying variance (realistic case)."""
    rng = np.random.default_rng(5)
    return {
        str(subject_id): rng.normal(size=(len(subject_frame), 3))
        for subject_id, subject_frame in frame.groupby("subject_id", sort=True)
    }


def _cohort_frame():
    import pandas as pd

    return pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame().assign(subject_id="sub-0002"),
        ],
        ignore_index=True,
    )
