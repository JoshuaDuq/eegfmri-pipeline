from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tests.pipelines_test_utils import DotConfig


class TestMachineLearningValidityFixes(unittest.TestCase):
    def test_execute_folds_parallel_handles_userwarning_filter(self):
        from eeg_pipeline.analysis.machine_learning import cv

        folds = [(0, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int))]

        def _fake_fold_func(fold, train_idx, test_idx):
            return {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }

        out = cv.execute_folds_parallel(folds=folds, fold_func=_fake_fold_func, outer_n_jobs=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["fold"], 0)

    def test_create_inner_cv_requires_two_groups(self):
        from eeg_pipeline.analysis.machine_learning.cv import create_inner_cv

        train_groups = np.array(["sub-0001", "sub-0001"], dtype=object)
        with self.assertRaisesRegex(ValueError, "at least 2 unique groups"):
            create_inner_cv(train_groups, inner_cv_splits=3)

    def test_create_within_subject_folds_respects_outer_cv_splits(self):
        from eeg_pipeline.analysis.machine_learning.cv import create_within_subject_folds

        groups = np.array(["sub-0001"] * 6, dtype=object)
        blocks = np.array([0, 0, 1, 1, 2, 2], dtype=float)

        folds = create_within_subject_folds(
            groups=groups,
            blocks_all=blocks,
            inner_cv_splits=5,
            outer_cv_splits=2,
            seed=42,
            config=DotConfig({}),
            epochs=None,
            apply_hygiene=False,
        )
        self.assertEqual(len(folds), 2)

    def test_split_conformal_handles_small_training_sets(self):
        from sklearn.dummy import DummyRegressor

        from eeg_pipeline.analysis.machine_learning.uncertainty import _conformal_split

        rng = np.random.default_rng(7)
        X_train = rng.standard_normal((5, 3))
        y_train = rng.standard_normal(5)
        X_test = rng.standard_normal((3, 3))

        result = _conformal_split(
            model=DummyRegressor(strategy="mean"),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            alpha=0.1,
            rng=np.random.default_rng(99),
        )

        self.assertEqual(len(result.y_pred), 3)
        self.assertEqual(len(result.lower), 3)
        self.assertEqual(len(result.upper), 3)
        self.assertTrue(np.all(np.isfinite(result.y_pred)))

    def test_classification_result_handles_single_class_confusion_matrix(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult

        y_true = np.zeros(6, dtype=int)
        y_pred = np.zeros(6, dtype=int)
        y_prob = np.zeros(6, dtype=float)

        result = ClassificationResult(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        self.assertEqual(result.confusion.shape, (2, 2))
        self.assertTrue(np.isfinite(result.specificity))

    def test_load_active_matrix_blocks_target_covariate_leakage(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        config = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
            }
        )

        x_df = pd.DataFrame({"power_feature": [1.0, 2.0]})
        meta_df = pd.DataFrame(
            {
                "subject_id": ["sub-0001", "sub-0001"],
                "rating": [10.0, 20.0],
                "temperature": [44.0, 45.0],
                "trial_index": [0, 1],
            }
        )

        def _fake_load_subject(*_args, **_kwargs):
            return x_df.copy(), np.array([10.0, 20.0], dtype=float), "rating", meta_df.copy()

        with patch.object(ml_data, "_load_subject_ml_from_features", side_effect=_fake_load_subject):
            with self.assertRaisesRegex(ValueError, "Covariates include the selected target"):
                ml_data.load_active_matrix(
                    subjects=["0001"],
                    task="thermalactive",
                    deriv_root=Path("."),
                    config=config,
                    feature_families=["power"],
                    target="rating",
                    covariates=["rating"],
                )

    def test_incremental_validity_blocks_target_baseline_predictor(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        config = DotConfig({})
        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "rating": y,
                "temperature": [44.0, 45.0, 46.0, 47.0],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)):
                with self.assertRaisesRegex(
                    ValueError, "Baseline predictors include the selected target"
                ):
                    orch.run_incremental_validity_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=config,
                        n_perm=0,
                        inner_splits=3,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        target="rating",
                        baseline_predictors=["rating"],
                    )

    def test_time_generalization_raises_on_stage_error(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "time_generalization_regression", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "Time-generalization stage failed"):
                    orch.run_time_generalization(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=DotConfig({}),
                        n_perm=0,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                    )

    def test_time_generalization_raises_on_empty_outputs(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch,
                "time_generalization_regression",
                return_value=(np.array([]), np.array([]), np.array([])),
            ):
                with self.assertRaisesRegex(RuntimeError, "produced no valid outputs"):
                    orch.run_time_generalization(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=DotConfig({}),
                        n_perm=0,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                    )

    def test_time_generalization_skips_folds_with_no_evaluable_cells(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        class _FakeEpochs:
            def __init__(self):
                self.times = np.array([0.0, 0.1, 0.2], dtype=float)
                self.metadata = pd.DataFrame({"block": [1]})

            def copy(self):
                return _FakeEpochs()

            def pick(self, _chs):
                return self

        trial_records = [("sub-0001", 0), ("sub-0002", 0)]
        y_arr = np.array([1.0, 2.0], dtype=float)
        groups_arr = np.array(["sub-0001", "sub-0002"], dtype=object)
        subj_to_epochs = {"sub-0001": _FakeEpochs(), "sub-0002": _FakeEpochs()}

        def _fake_prepare(_tuples):
            return trial_records, y_arr, groups_arr, subj_to_epochs, None

        def _fake_extract_epoch_data_block(indices, *_args, **_kwargs):
            return np.full((len(indices), 1, 3), np.nan, dtype=float)

        cfg = DotConfig(
            {
                "analysis": {"min_subjects_for_group": 2},
                "machine_learning": {
                    "analysis": {
                        "time_generalization": {
                            "active_window": [0.0, 0.1],
                            "window_len": 0.1,
                            "step": 0.1,
                            "min_samples_per_window": 1,
                            "min_samples_for_corr": 1,
                        }
                    }
                },
            }
        )

        with patch.object(tg, "load_epochs_with_targets", return_value=([], None)), patch.object(
            tg, "prepare_trial_records_from_epochs", side_effect=_fake_prepare
        ), patch.object(
            tg, "find_common_channels_train_test", return_value=["Cz"]
        ), patch.object(
            tg, "get_min_channels_required", return_value=1
        ), patch.object(
            tg, "extract_epoch_data_block", side_effect=_fake_extract_epoch_data_block
        ), patch.object(
            tg, "build_time_windows", return_value=[(0.0, 0.1)]
        ):
            tg_r, tg_r2, window_centers = tg.time_generalization_regression(
                deriv_root=Path("."),
                subjects=["0001", "0002"],
                task="thermalactive",
                results_dir=None,
                config_dict=cfg,
                n_perm=0,
                seed=42,
            )
        self.assertEqual(tg_r.size, 0)
        self.assertEqual(tg_r2.size, 0)
        self.assertEqual(window_centers.size, 0)

    def test_uncertainty_requires_minimum_valid_fold_fraction(self):
        from sklearn.dummy import DummyRegressor

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        class _FakeInterval:
            def __init__(self, y_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray):
                self.y_pred = y_pred
                self.lower = lower
                self.upper = upper
                self.coverage = np.nan
                self.mean_width = np.nan

            def compute_coverage(self, y_test: np.ndarray) -> None:
                self.coverage = float(np.mean((y_test >= self.lower) & (y_test <= self.upper)))
                self.mean_width = float(np.mean(self.upper - self.lower))

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)

        call_counter = {"n": 0}

        def _fake_compute_prediction_intervals(**kwargs):
            call_counter["n"] += 1
            x_test = np.asarray(kwargs["X_test"], dtype=float)
            if call_counter["n"] == 1:
                return _FakeInterval(
                    y_pred=np.zeros(len(x_test), dtype=float),
                    lower=np.full(len(x_test), -1.0, dtype=float),
                    upper=np.full(len(x_test), 1.0, dtype=float),
                )
            raise RuntimeError("synthetic fold failure")

        cfg = DotConfig(
            {
                "machine_learning": {
                    "cv": {"inner_splits": 2},
                    "analysis": {"uncertainty": {"min_valid_fold_fraction": 1.0}},
                }
            }
        )
        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch,
                "_build_regression_model_spec",
                return_value=("elasticnet", DummyRegressor(strategy="mean"), {}),
            ), patch.object(
                orch, "_fit_tuned_regression_estimator", side_effect=lambda **kwargs: kwargs["base_pipe"]
            ), patch(
                "eeg_pipeline.analysis.machine_learning.uncertainty.compute_prediction_intervals",
                side_effect=_fake_compute_prediction_intervals,
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid uncertainty folds"):
                    orch._run_uncertainty_stage(
                        X=X,
                        y=y,
                        groups=groups,
                        config=cfg,
                        seed=42,
                        alpha=0.1,
                        results_dir=Path(td),
                        logger=Mock(),
                        model_name="elasticnet",
                    )

    def test_shap_kernel_uses_estimator_predict_fn_for_transformed_features(self):
        from eeg_pipeline.analysis.machine_learning import shap_importance as si

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        feature_names = ["f1", "f2"]
        estimator = object()

        class _FakeKernelExplainer:
            def __init__(self, predict_fn, background):
                self.expected_value = 0.0
                self._predict_fn = predict_fn
                self._background = background

            def shap_values(self, X_input, nsamples=100):
                _ = (self._predict_fn, self._background, nsamples)
                return np.zeros_like(X_input, dtype=float)

        fake_shap = types.SimpleNamespace(KernelExplainer=_FakeKernelExplainer)
        captured = {}

        def _fake_create_predict_fn(model_obj):
            captured["predict_arg"] = model_obj
            return lambda X_input: np.zeros(len(X_input), dtype=float)

        with patch.object(si, "_check_shap_available", return_value=True), patch.object(
            si,
            "_extract_estimator_transform_and_feature_names",
            return_value=(estimator, X, feature_names),
        ), patch.object(si, "_create_predict_fn", side_effect=_fake_create_predict_fn), patch.dict(
            sys.modules, {"shap": fake_shap}
        ):
            si.compute_shap_values(
                model=object(),
                X=X,
                feature_names=feature_names,
                background_samples=2,
            )
        self.assertIs(captured["predict_arg"], estimator)

    def test_classification_calibration_uses_finite_probabilities_only(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult

        X = np.array(
            [
                [0.1, 1.0],
                [0.2, 1.1],
                [0.3, 1.2],
                [0.4, 1.3],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "block": [0, 1, 0, 1],
            }
        )
        result = ClassificationResult(
            y_true=y,
            y_pred=y.copy(),
            y_prob=np.array([0.1, 0.9, np.nan, np.nan], dtype=float),
            groups=groups,
            failed_fold_count=0,
            n_folds_total=2,
        )
        cfg = DotConfig({"machine_learning": {"classification": {"min_subjects_with_auc_for_inference": 1}}})
        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
                return_value=(result, pd.DataFrame()),
            ), patch.object(
                orch, "export_subject_selection_report", return_value={}
            ), patch.object(
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ):
                out_dir = orch.run_classification_ml(
                    subjects=["0001", "0002"],
                    task="thermalactive",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    classification_model="svm",
                )
            with open(out_dir / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
        self.assertAlmostEqual(float(metrics["brier_score"]), 0.01, places=6)
        self.assertAlmostEqual(float(metrics["expected_calibration_error"]), 0.1, places=6)

    def test_regression_permutation_completion_threshold_uses_config(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import cv

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        cfg = DotConfig({"machine_learning": {"cv": {"min_valid_permutation_fraction": 0.75}}})
        call_counter = {"n": 0}

        def _fake_nested_loso_predictions_matrix(**_kwargs):
            call_counter["n"] += 1
            y_true = np.array([0.0, 1.0], dtype=float)
            groups_ordered = ["sub-0001", "sub-0002"]
            if call_counter["n"] <= 2:
                y_pred = np.array([0.0, 1.0], dtype=float)
            else:
                y_pred = np.array([0.5, 0.5], dtype=float)
            return y_true, y_pred, groups_ordered, [0, 1], [1, 1]

        with tempfile.TemporaryDirectory() as td:
            with patch.object(cv, "nested_loso_predictions_matrix", side_effect=_fake_nested_loso_predictions_matrix):
                with self.assertRaisesRegex(RuntimeError, "required 0.750"):
                    cv.run_permutation_test(
                        X=X,
                        y=y,
                        groups=groups,
                        blocks=None,
                        pipe=Pipeline([("regressor", DummyRegressor(strategy="mean"))]),
                        param_grid={},
                        inner_cv_splits=2,
                        inner_n_jobs=1,
                        seed=42,
                        model_name="elasticnet",
                        null_n_perm=4,
                        null_output_path=Path(td) / "null.npz",
                        config=cfg,
                    )

    def test_model_comparison_rejects_nonzero_permutations(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "does not implement permutation inference"):
                orch.run_model_comparison_ml(
                    subjects=["0001", "0002"],
                    task="thermalactive",
                    deriv_root=Path(td),
                    config=DotConfig({}),
                    n_perm=10,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                )

    def test_incremental_validity_rejects_nonzero_permutations(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch,
                "load_active_matrix",
                side_effect=AssertionError("load_active_matrix should not be called for unsupported n_perm"),
            ):
                with self.assertRaisesRegex(ValueError, "does not implement permutation inference"):
                    orch.run_incremental_validity_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=DotConfig({}),
                        n_perm=10,
                        inner_splits=2,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                    )

    def test_model_comparison_blocks_binary_like_regression_target(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame({"subject_id": groups, "trial_id": [0, 1, 2, 3]})
        cfg = DotConfig({"machine_learning": {"targets": {"strict_regression_target_continuous": True}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)):
                with self.assertRaisesRegex(ValueError, "regression target appears binary-like"):
                    orch.run_model_comparison_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=0,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        target="rating",
                    )

    def test_incremental_validity_blocks_binary_like_regression_target(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "temperature": [44.0, 45.0, 46.0, 47.0],
            }
        )
        cfg = DotConfig({"machine_learning": {"targets": {"strict_regression_target_continuous": True}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)):
                with self.assertRaisesRegex(ValueError, "regression target appears binary-like"):
                    orch.run_incremental_validity_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=0,
                        inner_splits=2,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        target="rating",
                    )

    def test_within_subject_regression_permutation_completion_threshold_uses_config(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "block": [0, 1, 0, 1],
            }
        )

        folds = [
            (
                0,
                np.array([0, 2], dtype=int),
                np.array([1, 3], dtype=int),
                "sub-0001",
                {},
            )
        ]

        class _FakePipe:
            def __init__(self):
                self.named_steps = {"regressor": types.SimpleNamespace(param_grid={})}

        class _FakeEstimator:
            def predict(self, X_in: np.ndarray) -> np.ndarray:
                return np.asarray(X_in[:, 0], dtype=float)

        fit_calls = {"n": 0}

        def _fake_fit_within_subject_fold(**_kwargs):
            fit_calls["n"] += 1
            # 1 empirical fold + 4 permutations = 5 calls.
            # Fail the last 2 permutation fits so completion=2/4 (0.5).
            if fit_calls["n"] >= 4:
                raise RuntimeError("synthetic permutation failure")
            return _FakeEstimator()

        def _fake_export_predictions(y_true, y_pred, groups_ordered, *_args, **_kwargs):
            return pd.DataFrame(
                {
                    "y_true": np.asarray(y_true, dtype=float),
                    "y_pred": np.asarray(y_pred, dtype=float),
                    "subject_id": np.asarray(groups_ordered, dtype=object),
                }
            )

        cfg = DotConfig({"machine_learning": {"cv": {"min_valid_permutation_fraction": 0.75}}})
        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch.object(
                orch, "create_elasticnet_pipeline", side_effect=lambda **_kwargs: _FakePipe()
            ), patch.object(
                orch, "build_elasticnet_param_grid", return_value={}
            ), patch.object(
                orch, "_fit_within_subject_fold", side_effect=_fake_fit_within_subject_fold
            ), patch.object(
                orch, "export_predictions", side_effect=_fake_export_predictions
            ), patch.object(
                orch, "export_indices", return_value=None
            ), patch.object(
                orch, "compute_subject_level_r", return_value=(0.2, [("sub-0001", 0.2)], 0.1, 0.3)
            ), patch.object(
                orch,
                "compute_subject_level_errors",
                return_value={
                    "mean_mae": 0.1,
                    "ci_low_mae": 0.05,
                    "ci_high_mae": 0.2,
                    "mean_rmse": 0.2,
                    "ci_low_rmse": 0.1,
                    "ci_high_rmse": 0.3,
                    "per_subject": [],
                },
            ), patch.object(
                orch, "export_baseline_predictions", return_value={}
            ), patch.object(
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ), patch.object(
                orch, "export_subject_selection_report", return_value={}
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid within-subject regression permutations"):
                    orch.run_within_subject_regression_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=4,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                    )

    def test_regression_permutation_test_handles_nonfinite_predictions(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import cv

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        cfg = DotConfig({"machine_learning": {"cv": {"min_valid_permutation_fraction": 0.75}}})
        call_counter = {"n": 0}

        def _fake_nested_loso_predictions_matrix(**_kwargs):
            call_counter["n"] += 1
            y_true = np.array([0.0, 1.0], dtype=float)
            y_pred = np.array([0.0, 1.0], dtype=float)
            if call_counter["n"] == 2:
                y_pred = np.array([0.0, np.nan], dtype=float)
            return y_true, y_pred, ["sub-0001", "sub-0002"], [0, 1], [1, 1]

        with tempfile.TemporaryDirectory() as td:
            with patch.object(cv, "nested_loso_predictions_matrix", side_effect=_fake_nested_loso_predictions_matrix), patch.object(
                cv, "compute_subject_level_r", return_value=(0.2, [("sub-0001", 0.2)], 0.1, 0.3)
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid permutations"):
                    cv.run_permutation_test(
                        X=X,
                        y=y,
                        groups=groups,
                        blocks=None,
                        pipe=Pipeline([("regressor", DummyRegressor(strategy="mean"))]),
                        param_grid={},
                        inner_cv_splits=2,
                        inner_n_jobs=1,
                        seed=42,
                        model_name="elasticnet",
                        null_n_perm=2,
                        null_output_path=Path(td) / "null.npz",
                        config=cfg,
                    )

    def test_time_generalization_requires_min_valid_fold_fraction(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        class _FakeEpochs:
            def __init__(self):
                self.times = np.array([0.0, 0.1], dtype=float)
                self.metadata = pd.DataFrame({"block": [0, 1]})

            def copy(self):
                return _FakeEpochs()

            def pick(self, _chs):
                return self

        trial_records = [
            ("sub-0001", 0),
            ("sub-0001", 1),
            ("sub-0002", 0),
            ("sub-0002", 1),
            ("sub-0003", 0),
            ("sub-0003", 1),
        ]
        y_arr = np.array([0.0, 1.0, 0.5, 1.5, 0.25, 1.25], dtype=float)
        groups_arr = np.array([r[0] for r in trial_records], dtype=object)
        subj_to_epochs = {
            "sub-0001": _FakeEpochs(),
            "sub-0002": _FakeEpochs(),
            "sub-0003": _FakeEpochs(),
        }

        def _fake_prepare(_tuples):
            return trial_records, y_arr, groups_arr, subj_to_epochs, None

        def _fake_extract_epoch_data_block(indices, *_args, **_kwargs):
            return np.ones((len(indices), 1, 2), dtype=float)

        def _fake_extract_window_features(_data, records, *_args, **_kwargs):
            subs = {str(r[0]) for r in records}
            n_trials = len(records)
            if len(subs) == 1 and "sub-0001" not in subs:
                return np.full((n_trials, 1, 1), np.nan, dtype=float)
            vals = np.linspace(0.0, 1.0, max(n_trials, 1), dtype=float)
            return vals.reshape(n_trials, 1, 1)

        cfg = DotConfig(
            {
                "analysis": {"min_subjects_for_group": 2},
                "machine_learning": {
                    "analysis": {
                        "time_generalization": {
                            "active_window": [0.0, 0.1],
                            "window_len": 0.1,
                            "step": 0.1,
                            "min_samples_per_window": 1,
                            "min_samples_for_corr": 1,
                            "min_valid_fold_fraction": 0.8,
                        }
                    }
                },
            }
        )

        with patch.object(tg, "load_epochs_with_targets", return_value=([], None)), patch.object(
            tg, "prepare_trial_records_from_epochs", side_effect=_fake_prepare
        ), patch.object(
            tg, "find_common_channels_train_test", return_value=["Cz"]
        ), patch.object(
            tg, "get_min_channels_required", return_value=1
        ), patch.object(
            tg, "extract_epoch_data_block", side_effect=_fake_extract_epoch_data_block
        ), patch.object(
            tg, "_extract_window_features", side_effect=_fake_extract_window_features
        ), patch.object(
            tg, "build_time_windows", return_value=[(0.0, 0.1)]
        ), patch.object(
            tg, "safe_pearsonr", return_value=(0.5, 0.01)
        ), patch.object(
            tg, "r2_score", return_value=0.2
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient valid time-generalization fold coverage"):
                tg.time_generalization_regression(
                    deriv_root=Path("."),
                    subjects=["0001", "0002", "0003"],
                    task="thermalactive",
                    results_dir=None,
                    config_dict=cfg,
                    n_perm=0,
                    seed=42,
                )

    def test_shap_stage_requires_min_valid_fold_fraction(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        feature_names = ["f1"]
        shap_df = pd.DataFrame(
            {
                "feature": ["f1"],
                "shap_importance": [0.5],
                "shap_std_across_folds": [0.1],
                "n_folds_present": [1],
                "n_folds_used": [1],
                "n_folds_attempted": [3],
            }
        )
        cfg = DotConfig({"machine_learning": {"analysis": {"shap": {"min_valid_fold_fraction": 0.8}}}})

        with tempfile.TemporaryDirectory() as td:
            with patch(
                "eeg_pipeline.analysis.machine_learning.shap_importance.compute_shap_for_cv_folds",
                return_value=shap_df,
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid SHAP folds"):
                    orch._run_shap_importance_stage(
                        X=X,
                        y=y,
                        groups=groups,
                        feature_names=feature_names,
                        config=cfg,
                        seed=42,
                        results_dir=Path(td),
                        logger=Mock(),
                        model_name="elasticnet",
                    )

    def test_permutation_importance_stage_requires_min_valid_fold_fraction(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.1, 1.0],
                [0.2, 1.1],
                [0.3, 1.2],
                [0.4, 1.3],
                [0.5, 1.4],
                [0.6, 1.5],
            ],
            dtype=float,
        )
        y = np.array([1.0, 1.1, 2.0, 2.1, 3.0, 3.1], dtype=float)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        feature_names = ["f1", "f2"]
        cfg = DotConfig(
            {
                "machine_learning": {
                    "cv": {"inner_splits": 2},
                    "analysis": {
                        "permutation_importance": {"min_valid_fold_fraction": 0.8},
                    },
                }
            }
        )

        class _FakePermResult:
            def __init__(self, values):
                self.importances_mean = np.asarray(values, dtype=float)

        calls = {"n": 0}

        def _fake_permutation_importance(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakePermResult([0.2, 0.1])
            raise RuntimeError("synthetic fold failure")

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "_fit_tuned_regression_estimator", side_effect=lambda **kwargs: kwargs["base_pipe"]
            ), patch(
                "sklearn.inspection.permutation_importance",
                side_effect=_fake_permutation_importance,
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid permutation-importance folds"):
                    orch._run_permutation_importance_stage(
                        X=X,
                        y=y,
                        groups=groups,
                        feature_names=feature_names,
                        config=cfg,
                        seed=42,
                        n_repeats=3,
                        results_dir=Path(td),
                        logger=Mock(),
                        model_name="elasticnet",
                    )

    def test_within_subject_regression_skips_partial_fold_permutations(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "block": [0, 1, 0, 1],
            }
        )
        folds = [
            (0, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int), "sub-0001", {}),
            (1, np.array([2, 3], dtype=int), np.array([0, 1], dtype=int), "sub-0002", {}),
        ]

        class _FakePipe:
            def __init__(self):
                self.named_steps = {"regressor": types.SimpleNamespace(param_grid={})}

        class _FakeEstimator:
            def predict(self, X_in: np.ndarray) -> np.ndarray:
                return np.asarray(X_in[:, 0], dtype=float)

        fit_calls = {"n": 0}

        def _fake_fit_within_subject_fold(**_kwargs):
            fit_calls["n"] += 1
            if fit_calls["n"] <= 2:
                return _FakeEstimator()
            # Permutation calls: fail second fold in each permutation (calls 4 and 6).
            if fit_calls["n"] in {4, 6}:
                raise RuntimeError("synthetic permutation fold failure")
            return _FakeEstimator()

        def _fake_export_predictions(y_true, y_pred, groups_ordered, *_args, **_kwargs):
            return pd.DataFrame(
                {
                    "y_true": np.asarray(y_true, dtype=float),
                    "y_pred": np.asarray(y_pred, dtype=float),
                    "subject_id": np.asarray(groups_ordered, dtype=object),
                }
            )

        cfg = DotConfig({"machine_learning": {"cv": {"min_valid_permutation_fraction": 0.75}}})
        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch.object(
                orch, "create_elasticnet_pipeline", side_effect=lambda **_kwargs: _FakePipe()
            ), patch.object(
                orch, "build_elasticnet_param_grid", return_value={}
            ), patch.object(
                orch, "_fit_within_subject_fold", side_effect=_fake_fit_within_subject_fold
            ), patch.object(
                orch, "export_predictions", side_effect=_fake_export_predictions
            ), patch.object(
                orch, "export_indices", return_value=None
            ), patch.object(
                orch, "compute_subject_level_r", return_value=(0.2, [("sub-0001", 0.2)], 0.1, 0.3)
            ), patch.object(
                orch,
                "compute_subject_level_errors",
                return_value={
                    "mean_mae": 0.1,
                    "ci_low_mae": 0.05,
                    "ci_high_mae": 0.2,
                    "mean_rmse": 0.2,
                    "ci_low_rmse": 0.1,
                    "ci_high_rmse": 0.3,
                    "per_subject": [],
                },
            ), patch.object(
                orch, "export_baseline_predictions", return_value={}
            ), patch.object(
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ), patch.object(
                orch, "export_subject_selection_report", return_value={}
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid within-subject regression permutations"):
                    orch.run_within_subject_regression_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=2,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                    )

    def test_within_subject_classification_permutations_enforce_failed_fold_fraction(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.ones((8, 2, 3), dtype=float)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0002", "sub-0002"],
            dtype=object,
        )
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "block": np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
            }
        )
        folds = [
            (0, np.array([0, 1, 4, 5], dtype=int), np.array([2, 3, 6, 7], dtype=int), "sub-0001", {}),
            (1, np.array([2, 3, 6, 7], dtype=int), np.array([0, 1, 4, 5], dtype=int), "sub-0002", {}),
        ]
        fit_calls = {"n": 0}

        def _fake_fit_predict_cnn_binary_classifier(*, X_test, **_kwargs):
            fit_calls["n"] += 1
            # Empirical folds (first 2 calls) succeed.
            # During permutations, fail every second fold to induce failed_fold_fraction=0.5.
            if fit_calls["n"] > 2 and fit_calls["n"] % 2 == 0:
                raise RuntimeError("synthetic cnn permutation fold failure")
            n_test = int(len(X_test))
            y_pred = np.array(([0, 1] * ((n_test + 1) // 2))[:n_test], dtype=int)
            y_prob = np.array(([0.1, 0.9] * ((n_test + 1) // 2))[:n_test], dtype=float)
            return y_pred, y_prob

        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {"max_failed_fold_fraction": 0.25, "min_subjects_with_auc_for_inference": 1},
                    "cv": {"min_valid_permutation_fraction": 0.75},
                }
            }
        )
        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_epoch_tensor_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch(
                "eeg_pipeline.analysis.machine_learning.cnn.fit_predict_cnn_binary_classifier",
                side_effect=_fake_fit_predict_cnn_binary_classifier,
            ), patch.object(
                orch, "export_predictions", side_effect=lambda y_true, y_pred, groups_ordered, *_a, **_k: pd.DataFrame(
                    {"y_true": y_true, "y_pred": y_pred, "subject_id": groups_ordered}
                )
            ), patch.object(
                orch, "export_indices", return_value=None
            ), patch.object(
                orch, "export_subject_selection_report", return_value={}
            ), patch.object(
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient valid within-subject classification permutations"):
                    orch.run_within_subject_classification_ml(
                        subjects=["0001", "0002"],
                        task="thermalactive",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=2,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        classification_model="cnn",
                    )

    def test_group_classification_permutations_enforce_failed_fold_fraction(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        bad_result = ClassificationResult(
            y_true=y,
            y_pred=y.copy(),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8], dtype=float),
            groups=groups,
            failed_fold_count=1,
            n_folds_total=2,
        )
        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {"max_failed_fold_fraction": 0.25},
                    "cv": {"min_valid_permutation_fraction": 0.75},
                }
            }
        )
        with patch(
            "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
            return_value=(bad_result, pd.DataFrame()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient valid classification permutations"):
                orch._run_classification_permutations(
                    X=X,
                    y=y,
                    groups=groups,
                    blocks=None,
                    model="svm",
                    inner_splits=2,
                    seed=42,
                    n_perm=2,
                    config=cfg,
                    logger=Mock(),
                    harmonization_mode="intersection",
                )

    def test_group_classification_permutations_do_not_fallback_to_pooled_auc(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {
                        "max_failed_fold_fraction": 1.0,
                        "min_subjects_with_auc_for_inference": 2,
                    },
                    "cv": {"min_valid_permutation_fraction": 0.75},
                }
            }
        )

        fake_result = types.SimpleNamespace(
            failed_fold_count=0,
            n_folds_total=2,
            per_subject_metrics={
                "sub-0001": {"auc": 0.6},
                "sub-0002": {"accuracy": 1.0},
            },
            auc=0.95,
        )

        with patch(
            "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
            return_value=(fake_result, pd.DataFrame()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient valid classification permutations"):
                orch._run_classification_permutations(
                    X=X,
                    y=y,
                    groups=groups,
                    blocks=None,
                    model="svm",
                    inner_splits=2,
                    seed=42,
                    n_perm=2,
                    config=cfg,
                    logger=Mock(),
                    harmonization_mode="intersection",
                )
