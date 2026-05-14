from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from studies.tests.test_support import DotConfig


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, prediction: float = 0.0) -> None:
        self.prediction = prediction

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.full(len(X), self.prediction, dtype=float)


class FixedShiftRng:
    def choice(self, values):
        assert 5 in values
        return 5


def test_circular_shift_permutation_orders_trials_by_original_index() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    trial_indices = np.asarray([3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int)
    y = trial_indices.astype(float)
    groups = np.repeat("sub-0001", len(y)).astype(object)
    blocks = np.repeat(1, len(y)).astype(float)

    permuted = _permute_labels_by_scheme(
        y,
        groups,
        blocks=blocks,
        trial_indices=trial_indices,
        rng=FixedShiftRng(),
        scheme="circular_shift_within_run",
    )

    expected = np.asarray([9, 7, 8, 10, 11, 1, 2, 3, 4, 5, 6], dtype=float)
    assert np.array_equal(permuted, expected)


def test_model_comparison_inner_tuning_uses_equal_subject_weighted_r2() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        _fit_subject_weighted_inner_cv_estimator,
    )

    group_a = np.repeat("sub-a", 100)
    group_b = np.repeat("sub-b", 2)
    group_c = np.repeat("sub-c", 2)
    groups = np.concatenate([group_a, group_b, group_c]).astype(object)
    y = np.concatenate(
        [
            np.zeros(len(group_a), dtype=float),
            np.full(len(group_b), 10.0, dtype=float),
            np.full(len(group_c), 10.0, dtype=float),
        ]
    )
    X = np.zeros((len(y), 1), dtype=float)

    estimator = _fit_subject_weighted_inner_cv_estimator(
        base_estimator=ConstantRegressor(),
        param_grid={"prediction": [0.0, 10.0]},
        X_train=X,
        y_train=y,
        groups_train=groups,
        inner_splits=2,
    )

    assert estimator.prediction == 10.0


def test_model_comparison_inner_tuning_enforces_subject_missingness_limit() -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    from eeg_pipeline.analysis.machine_learning.orchestration import (
        _fit_subject_weighted_inner_cv_estimator,
    )
    from eeg_pipeline.analysis.machine_learning.preprocessing import MissingnessThreshold

    rows_per_subject = 20
    n_features = 50
    groups = np.repeat(
        ["sub-a", "sub-b", "sub-c", "sub-d", "sub-e", "sub-f"],
        rows_per_subject,
    ).astype(object)
    y = np.linspace(0.0, 1.0, len(groups))
    X = np.ones((len(groups), n_features), dtype=float)
    sub_a_rows = np.flatnonzero(groups == "sub-a")
    X[sub_a_rows[:3], :] = np.nan

    estimator = Pipeline(
        [
            (
                "missingness",
                MissingnessThreshold(
                    max_feature_missingness=0.05,
                    max_subject_missingness=0.10,
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("regressor", ConstantRegressor()),
        ]
    )

    with pytest.raises(ValueError, match="subject sub-a"):
        _fit_subject_weighted_inner_cv_estimator(
            base_estimator=estimator,
            param_grid={"regressor__prediction": [0.0]},
            X_train=X,
            y_train=y,
            groups_train=groups,
            inner_splits=2,
        )


def test_model_comparison_inner_tuning_enforces_subject_missingness_limit_in_wrapped_pipeline() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        _fit_subject_weighted_inner_cv_estimator,
    )
    from eeg_pipeline.analysis.machine_learning.pipelines import create_ridge_pipeline
    from studies.tests.test_support import DotConfig

    rows_per_subject = 20
    n_features = 50
    groups = np.repeat(
        ["sub-a", "sub-b", "sub-c", "sub-d", "sub-e", "sub-f"],
        rows_per_subject,
    ).astype(object)
    y = np.linspace(0.0, 1.0, len(groups))
    X = np.ones((len(groups), n_features), dtype=float)
    sub_a_rows = np.flatnonzero(groups == "sub-a")
    X[sub_a_rows[:3], :] = np.nan

    estimator = create_ridge_pipeline(
        config=DotConfig(
            {
                "machine_learning": {
                    "models": {"ridge": {"alpha_grid": [0.1]}},
                    "preprocessing": {
                        "max_feature_missingness": 0.05,
                        "max_subject_missingness": 0.10,
                        "variance_threshold_grid": [0.0],
                    },
                }
            }
        )
    )

    with pytest.raises(ValueError, match="subject sub-a"):
        _fit_subject_weighted_inner_cv_estimator(
            base_estimator=estimator,
            param_grid={"regressor__regressor__alpha": [0.1]},
            X_train=X,
            y_train=y,
            groups_train=groups,
            inner_splits=2,
        )


def test_feature_metadata_filter_excludes_prespecified_channels() -> None:
    from eeg_pipeline.utils.data.machine_learning import filter_feature_columns_by_metadata

    columns = [
        "power_plateau_alpha_ch_Fp1_mean",
        "power_plateau_alpha_ch_Fp2_mean",
        "power_plateau_alpha_ch_Cz_mean",
        "power_plateau_beta_ch_Cz_mean",
        "power_plateau_alpha_roi_frontal_mean",
    ]

    kept = filter_feature_columns_by_metadata(
        columns,
        bands=["alpha"],
        segments=["plateau"],
        scopes=None,
        stats=["mean"],
        excluded_channels=["Fp1", "Fp2"],
    )

    assert kept == [
        "power_plateau_alpha_ch_Cz_mean",
        "power_plateau_alpha_roi_frontal_mean",
    ]


def test_reproducibility_info_records_full_config_and_input_hashes(tmp_path: Path) -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import write_reproducibility_info

    path = write_reproducibility_info(
        tmp_path,
        subjects=["sub-0001", "sub-0002"],
        config=DotConfig(
            {
                "study1": {"targets": {"names": ["NPS", "SIIPS1"]}},
                "machine_learning": {"cv": {"permutation_scheme": "circular_shift_within_run"}},
            }
        ),
        rng_seed=7,
        input_hashes={
            "feature_matrix_sha256": "feature-hash",
            "target_vector_sha256": "target-hash",
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["config_snapshot"]["study1"]["targets"]["names"] == ["NPS", "SIIPS1"]
    assert payload["input_hashes"]["feature_matrix_sha256"] == "feature-hash"
    assert payload["input_hashes"]["target_vector_sha256"] == "target-hash"
    assert payload["command"]
    assert payload["git"]["commit"]
