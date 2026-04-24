from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from builtins import __import__ as _real_import
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tests.pipelines_test_utils import DotConfig


class TestMachineLearningValidityFixes(unittest.TestCase):
    def _import_ml_data(self):
        mne_home = Path(tempfile.mkdtemp())
        with patch.dict(
            os.environ,
            {"HOME": str(mne_home), "MNE_DONTWRITE_HOME": "true"},
            clear=False,
        ), patch.dict(
            sys.modules,
            {
                "eeg_pipeline.utils.config": types.SimpleNamespace(__path__=[]),
                "eeg_pipeline.utils.config.loader": types.SimpleNamespace(
                    ConfigDict=dict,
                    get_config_value=lambda config, key, default=None: config.get(key, default)
                    if hasattr(config, "get")
                    else default,
                    load_config=lambda *args, **kwargs: DotConfig({}),
                ),
                "mne": types.SimpleNamespace(
                    pick_types=lambda *_args, **_kwargs: np.array([], dtype=int),
                    Epochs=object,
                ),
                "eeg_pipeline.domain.features": types.ModuleType("eeg_pipeline.domain.features"),
                "eeg_pipeline.domain.features.naming": types.SimpleNamespace(
                    NamingSchema=type(
                        "NamingSchema",
                        (),
                        {"parse": staticmethod(lambda *_args, **_kwargs: {"valid": False})},
                    )
                ),
                "eeg_pipeline.infra": types.SimpleNamespace(__path__=[]),
                "eeg_pipeline.infra.tsv": types.SimpleNamespace(
                    read_table=lambda *_args, **_kwargs: pd.DataFrame(),
                    read_tsv=lambda *_args, **_kwargs: pd.DataFrame(),
                ),
                "eeg_pipeline.infra.paths": types.SimpleNamespace(
                    _find_clean_events_path=lambda *_args, **_kwargs: None,
                    deriv_features_path=lambda deriv_root, subject: Path(deriv_root)
                    / f"sub-{subject}"
                    / "eeg"
                    / "features",
                    load_events_df=lambda *_args, **_kwargs: pd.DataFrame(),
                ),
                "eeg_pipeline.utils.data.epochs": types.SimpleNamespace(
                    load_epochs_for_analysis=lambda *_args, **_kwargs: (None, None),
                ),
                "eeg_pipeline.utils.data.fmri_signature_targets": types.SimpleNamespace(
                    find_block_column=lambda *_args, **_kwargs: None,
                    load_fmri_signature_target_for_subject=lambda **_kwargs: (_ for _ in ()).throw(
                        AssertionError("fMRI signature target helper should not be used in this test")
                    ),
                ),
                "mne_bids": types.SimpleNamespace(
                    BIDSPath=type(
                        "BIDSPath",
                        (),
                        {"__init__": lambda self, *args, **kwargs: setattr(self, "fpath", None)},
                    )
                )
            },
        ):
            from eeg_pipeline.utils.data import machine_learning as ml_data

        return ml_data

    def test_spatial_feature_selector_requires_feature_names_when_regions_are_requested(self):
        from eeg_pipeline.analysis.machine_learning.preprocessing import SpatialFeatureSelector

        selector = SpatialFeatureSelector(allowed_regions=["insula"], config=DotConfig({}))

        with self.assertRaisesRegex(ValueError, "feature names"):
            selector.fit(np.ones((4, 3), dtype=float))

    def test_resolve_feature_families_surfaces_standard_catalog_import_failure(self):
        from eeg_pipeline.utils.data.machine_learning import _resolve_feature_families

        def _import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "eeg_pipeline.utils.data.feature_discovery":
                raise ImportError("catalog-missing")
            return _real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_import_with_failure):
            with self.assertRaisesRegex(ImportError, "catalog-missing"):
                _resolve_feature_families(
                    feature_families=None,
                    feature_set="combined",
                    config=DotConfig({}),
                )

    def test_resolve_feature_families_prefers_configured_feature_categories_for_combined_defaults(
        self,
    ):
        ml_data = self._import_ml_data()

        def _import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "eeg_pipeline.utils.data.feature_discovery":
                raise ImportError("feature-discovery-should-not-load")
            return _real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_import_with_failure):
            resolved = ml_data._resolve_feature_families(
                feature_families=None,
                feature_set="combined",
                config=DotConfig(
                    {
                        "feature_engineering": {
                            "feature_categories": ["power", "aperiodic", "erp"],
                        }
                    }
                ),
            )

        self.assertEqual(resolved, ["power", "aperiodic", "erp"])

    def test_cnn_loso_surfaces_fold_failures(self):
        from eeg_pipeline.analysis.machine_learning import cnn

        X = np.random.default_rng(17).standard_normal((4, 2, 16))
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)

        with patch.object(
            cnn,
            "fit_predict_cnn_binary_classifier",
            side_effect=RuntimeError("synthetic fold failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CNN fold 1 failed"):
                cnn.nested_loso_cnn_classification(
                    X=X,
                    y=y,
                    groups=groups,
                    seed=42,
                    config=DotConfig({}),
                )

    def test_cnn_validation_split_stays_group_disjoint_when_group_splitter_fails(self):
        from eeg_pipeline.analysis.machine_learning import cnn

        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        y = np.array([0, 1, 0, 1, 0, 1], dtype=int)

        with patch.object(cnn.GroupShuffleSplit, "split", side_effect=RuntimeError("synthetic failure")):
            train_idx, val_idx = cnn._split_train_val_indices(
                groups_train=groups,
                y_train=y,
                seed=42,
                val_fraction=0.34,
            )

        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        self.assertTrue(train_groups)
        self.assertTrue(val_groups)
        self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_cnn_validation_split_is_disjoint_for_small_sample_regimes(self):
        from eeg_pipeline.analysis.machine_learning import cnn

        groups = np.array(["sub-0001", "sub-0001", "sub-0001", "sub-0001"], dtype=object)
        y = np.array([0, 1, 0, 1], dtype=int)

        train_idx, val_idx = cnn._split_train_val_indices(
            groups_train=groups,
            y_train=y,
            seed=13,
            val_fraction=0.25,
        )

        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertEqual(len(np.intersect1d(train_idx, val_idx)), 0)

    def test_decode_binary_outcome_uses_stratified_group_kfold_for_grouped_numeric_cv(self):
        from eeg_pipeline.analysis.machine_learning import classification as clf

        X = np.array(
            [
                [0.0, 1.0],
                [0.1, 1.1],
                [0.2, 1.2],
                [0.3, 1.3],
                [0.4, 1.4],
                [0.5, 1.5],
                [0.6, 1.6],
                [0.7, 1.7],
                [0.8, 1.8],
                [0.9, 1.9],
                [1.0, 2.0],
                [1.1, 2.1],
            ],
            dtype=float,
        )
        y = np.array([0, 1] * 6, dtype=int)
        groups = np.array(
            [
                "sub-0001",
                "sub-0001",
                "sub-0002",
                "sub-0002",
                "sub-0003",
                "sub-0003",
                "sub-0004",
                "sub-0004",
                "sub-0005",
                "sub-0005",
                "sub-0006",
                "sub-0006",
            ],
            dtype=object,
        )
        called = {"group_kfold": False}

        original_group_kfold = clf.StratifiedGroupKFold

        class _TrackingStratifiedGroupKFold:
            def __init__(self, *args, **kwargs):
                self._inner = original_group_kfold(*args, **kwargs)

            def split(self, X_in, y_in, groups_in):
                called["group_kfold"] = True
                return self._inner.split(X_in, y_in, groups_in)

        with patch.object(clf, "StratifiedGroupKFold", _TrackingStratifiedGroupKFold):
            result = clf.decode_binary_outcome(
                X=X,
                y=y,
                cv=3,
                groups=groups,
                model="lr",
                seed=42,
                config=DotConfig({}),
            )

        self.assertTrue(called["group_kfold"])
        self.assertEqual(result.n_folds_total, 3)

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

    def test_create_within_subject_folds_requires_run_blocks(self):
        from eeg_pipeline.analysis.machine_learning.cv import create_within_subject_folds

        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)

        with self.assertRaisesRegex(ValueError, "run_id"):
            create_within_subject_folds(
                groups=groups,
                blocks_all=None,
                inner_cv_splits=2,
                outer_cv_splits=2,
                seed=42,
                config=DotConfig({}),
                epochs=None,
                apply_hygiene=False,
            )

    def test_create_within_subject_folds_rejects_insufficient_subject_blocks(self):
        from eeg_pipeline.analysis.machine_learning.cv import create_within_subject_folds

        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([1, 1, 1, 2], dtype=int)

        with self.assertRaisesRegex(ValueError, "insufficient blocks"):
            create_within_subject_folds(
                groups=groups,
                blocks_all=blocks,
                inner_cv_splits=2,
                outer_cv_splits=2,
                seed=42,
                config=DotConfig({}),
                epochs=None,
                apply_hygiene=False,
            )

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

    def test_cv_plus_uses_fold_specific_prediction_bands(self):
        from sklearn.base import BaseEstimator, RegressorMixin

        from eeg_pipeline.analysis.machine_learning.uncertainty import _conformal_cv_plus

        class _MeanRegressor(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                _ = X
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_, dtype=float)

        X_train = np.arange(4, dtype=float).reshape(-1, 1)
        y_train = np.array([0.0, 0.0, 10.0, 10.0], dtype=float)
        X_test = np.array([[100.0]], dtype=float)
        fold_splits = [
            (np.array([0, 1], dtype=int), np.array([2, 3], dtype=int)),
            (np.array([2, 3], dtype=int), np.array([0, 1], dtype=int)),
        ]

        with patch(
            "eeg_pipeline.analysis.machine_learning.uncertainty._get_cv_splitter",
            return_value=iter(fold_splits),
        ):
            result = _conformal_cv_plus(
                model=_MeanRegressor(),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                alpha=0.1,
                cv_splits=2,
                seed=42,
                groups=None,
            )

        # Proper CV+ lower/upper candidates are built from each fold model:
        # fold-1 mean=0 with residuals 10, fold-2 mean=10 with residuals 10.
        self.assertAlmostEqual(float(result.y_pred[0]), 5.0, places=8)
        self.assertAlmostEqual(float(result.lower[0]), -10.0, places=8)
        self.assertAlmostEqual(float(result.upper[0]), 20.0, places=8)

    def test_cv_plus_raises_when_no_calibration_chunks_survive(self):
        from sklearn.base import BaseEstimator, RegressorMixin

        from eeg_pipeline.analysis.machine_learning.uncertainty import _conformal_cv_plus

        class _MeanRegressor(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                _ = X
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_, dtype=float)

        X_train = np.arange(6, dtype=float).reshape(-1, 1)
        y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        X_test = np.array([[10.0]], dtype=float)
        groups = np.array(["sub-0001"] * len(y_train), dtype=object)

        with patch(
            "eeg_pipeline.analysis.machine_learning.uncertainty._get_cv_splitter",
            return_value=iter([]),
        ):
            with self.assertRaisesRegex(RuntimeError, "CV\\+ calibration failed"):
                _conformal_cv_plus(
                    model=_MeanRegressor(),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    alpha=0.1,
                    cv_splits=3,
                    seed=42,
                    groups=groups,
                )

    def test_classification_result_handles_single_class_confusion_matrix(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult

        y_true = np.zeros(6, dtype=int)
        y_pred = np.zeros(6, dtype=int)
        y_prob = np.zeros(6, dtype=float)

        result = ClassificationResult(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        self.assertEqual(result.confusion.shape, (2, 2))
        self.assertTrue(np.isfinite(result.specificity))

    def test_classification_result_marks_single_class_subject_balanced_accuracy_nan(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult

        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_pred = np.array([0, 1, 1, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)

        result = ClassificationResult(y_true=y_true, y_pred=y_pred, groups=groups)
        self.assertIn("balanced_accuracy", result.per_subject_metrics["sub-0001"])
        self.assertIn("balanced_accuracy", result.per_subject_metrics["sub-0002"])
        self.assertTrue(np.isnan(result.per_subject_metrics["sub-0001"]["balanced_accuracy"]))
        self.assertTrue(np.isnan(result.per_subject_metrics["sub-0002"]["balanced_accuracy"]))

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
                "vas_final_rating": [10.0, 20.0],
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
                    task="task",
                    deriv_root=Path("."),
                    config=config,
                    feature_families=["power"],
                    target="rating",
                    covariates=["rating"],
                )

    def test_load_active_matrix_preserves_canonical_trial_ids(self):
        ml_data = self._import_ml_data()

        config = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
            }
        )

        subject_payloads = {
            "0001": (
                pd.DataFrame({"power_feature": [1.0, 2.0]}),
                np.array([10.0, 20.0], dtype=float),
                "rating",
                pd.DataFrame(
                    {
                        "subject_id": ["sub-0001", "sub-0001"],
                        "trial_id": [10, 11],
                        "trial_index": [0, 1],
                    }
                ),
            ),
            "0002": (
                pd.DataFrame({"power_feature": [3.0]}),
                np.array([30.0], dtype=float),
                "rating",
                pd.DataFrame(
                    {
                        "subject_id": ["sub-0002"],
                        "trial_id": [21],
                        "trial_index": [0],
                    }
                ),
            ),
        }

        def _fake_load_subject(subject, *_args, **_kwargs):
            return subject_payloads[str(subject)]

        with patch.object(ml_data, "_load_subject_ml_from_features", side_effect=_fake_load_subject):
            _X, _y, groups, _feature_names, meta = ml_data.load_active_matrix(
                subjects=["0001", "0002"],
                task="task",
                deriv_root=Path("."),
                config=config,
                feature_families=["power"],
            )

        self.assertEqual(groups.tolist(), ["sub-0001", "sub-0001", "sub-0002"])
        self.assertEqual(meta["trial_id"].tolist(), [10, 11, 21])

    def test_load_subject_ml_from_features_passes_deriv_root_to_clean_event_lookup(self):
        ml_data = self._import_ml_data()
        deriv_root = Path(tempfile.mkdtemp())
        captured: dict[str, object] = {}

        def _load_events_df(subject, task, *, deriv_root=None, config=None, prefer_clean=True):
            _ = (subject, task, config, prefer_clean)
            captured["deriv_root"] = deriv_root
            return pd.DataFrame({"trial_id": [7], "rating": [10.0]})

        feature_df = pd.DataFrame({"power_feature": [1.0]})
        feature_df.attrs["trial_id"] = np.array([7], dtype=int)

        with patch.object(ml_data, "load_events_df", side_effect=_load_events_df), patch.object(
            ml_data,
            "_load_subject_feature_table",
            return_value=(feature_df, ["power_feature"]),
        ):
            _X_df, y, y_col, meta = ml_data._load_subject_ml_from_features(
                subject="0001",
                task="task",
                deriv_root=deriv_root,
                config=DotConfig({"event_columns": {"outcome": ["rating"]}}),
                feature_families=["power"],
                feature_input_root=None,
                target=None,
                target_kind="continuous",
                binary_threshold=None,
                logger=Mock(),
            )

        self.assertEqual(captured["deriv_root"], deriv_root)
        self.assertEqual(y.tolist(), [10.0])
        self.assertEqual(y_col, "rating")
        self.assertEqual(meta["trial_id"].tolist(), [7])

    def test_load_subject_feature_table_raises_when_any_requested_family_is_missing(self):
        ml_data = self._import_ml_data()
        cfg = DotConfig({"machine_learning": {"data": {"require_trial_ml_safe": False}}})

        with tempfile.TemporaryDirectory() as td:
            deriv_root = Path(td)
            power_dir = deriv_root / "sub-0001" / "eeg" / "features" / "power"
            power_dir.mkdir(parents=True, exist_ok=True)
            (power_dir / "features_power.parquet").touch()

            with patch.object(
                ml_data,
                "read_table",
                return_value=pd.DataFrame(
                    {
                        "trial_id": [1, 2],
                        "power_alpha_global_mean": [1.0, 2.0],
                    }
                ),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "connectivity"):
                    ml_data._load_subject_feature_table(
                        subject="0001",
                        task="task",
                        deriv_root=deriv_root,
                        config=cfg,
                        feature_families=["power", "connectivity"],
                        feature_input_root=None,
                        logger=Mock(),
                    )

    def test_load_active_matrix_rejects_invalid_feature_harmonization(self):
        ml_data = self._import_ml_data()

        cfg = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
                "machine_learning": {"data": {"feature_harmonization": "zscore"}},
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid machine_learning.data.feature_harmonization",
        ):
            ml_data.load_active_matrix(
                subjects=["0001"],
                task="task",
                deriv_root=Path("/tmp/deriv"),
                config=cfg,
                feature_families=["power"],
            )

    def test_load_active_matrix_rejects_feature_families_with_channels_mean_feature_set(self):
        ml_data = self._import_ml_data()

        cfg = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
                "machine_learning": {"data": {"feature_set": "channels_mean"}},
            }
        )

        with self.assertRaisesRegex(ValueError, "channels_mean"):
            ml_data.load_active_matrix(
                subjects=["0001"],
                task="task",
                deriv_root=Path("/tmp/deriv"),
                config=cfg,
                    feature_families=["power"],
            )

    def test_load_active_matrix_does_not_require_runtime_trial_safe_mode_when_subject_tables_load(
        self,
    ):
        ml_data = self._import_ml_data()

        cfg = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "group_stats"},
                "machine_learning": {"data": {"require_trial_ml_safe": True}},
            }
        )

        payload = (
            pd.DataFrame({"power_alpha_global_mean": [1.0, 2.0]}),
            np.array([10.0, 20.0], dtype=float),
            "rating",
            pd.DataFrame(
                {
                    "subject_id": ["sub-0001", "sub-0001"],
                    "trial_id": [1, 2],
                }
            ),
        )

        with patch.object(ml_data, "_load_subject_ml_from_features", return_value=payload):
            X, y, groups, feature_names, meta = ml_data.load_active_matrix(
                subjects=["0001"],
                task="task",
                deriv_root=Path("/tmp/deriv"),
                config=cfg,
                feature_families=["power"],
            )

        self.assertEqual(X.shape, (2, 1))
        np.testing.assert_array_equal(y, np.array([10.0, 20.0], dtype=float))
        np.testing.assert_array_equal(groups, np.array(["sub-0001", "sub-0001"], dtype=object))
        self.assertEqual(feature_names, ["power_alpha_global_mean"])
        self.assertEqual(meta["trial_id"].tolist(), [1, 2])

    def test_load_active_matrix_raises_when_requested_covariates_are_missing_in_strict_mode(self):
        ml_data = self._import_ml_data()

        cfg = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
                "machine_learning": {"data": {"covariates_strict": True}},
            }
        )

        def _fake_load_subject(*_args, **_kwargs):
            return (
                pd.DataFrame({"power_alpha_global_mean": [1.0, 2.0]}),
                np.array([10.0, 20.0], dtype=float),
                "rating",
                pd.DataFrame(
                    {
                        "subject_id": ["sub-0001", "sub-0001"],
                        "trial_id": [1, 2],
                    }
                ),
            )

        with patch.object(ml_data, "_load_subject_ml_from_features", side_effect=_fake_load_subject):
            with self.assertRaisesRegex(ValueError, "Requested covariates missing from meta"):
                ml_data.load_active_matrix(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path("/tmp/deriv"),
                    config=cfg,
                    feature_families=["power"],
                    covariates=["age"],
                )

    def test_load_active_matrix_drops_missing_covariates_when_not_strict(self):
        ml_data = self._import_ml_data()

        cfg = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
                "machine_learning": {"data": {"covariates_strict": False}},
            }
        )

        def _fake_load_subject(*_args, **_kwargs):
            return (
                pd.DataFrame({"power_alpha_global_mean": [1.0, 2.0]}),
                np.array([10.0, 20.0], dtype=float),
                "rating",
                pd.DataFrame(
                    {
                        "subject_id": ["sub-0001", "sub-0001"],
                        "trial_id": [1, 2],
                    }
                ),
            )

        with patch.object(ml_data, "_load_subject_ml_from_features", side_effect=_fake_load_subject):
            X, y, groups, feature_names, meta = ml_data.load_active_matrix(
                subjects=["0001"],
                task="task",
                deriv_root=Path("/tmp/deriv"),
                config=cfg,
                feature_families=["power"],
                covariates=["age"],
            )

        self.assertEqual(X.shape, (2, 1))
        np.testing.assert_array_equal(y, np.array([10.0, 20.0], dtype=float))
        np.testing.assert_array_equal(groups, np.array(["sub-0001", "sub-0001"], dtype=object))
        self.assertEqual(feature_names, ["power_alpha_global_mean"])
        self.assertEqual(meta["trial_id"].tolist(), [1, 2])

    def test_load_active_matrix_surfaces_subject_loading_errors(self):
        ml_data = self._import_ml_data()

        cfg = DotConfig({"feature_engineering": {"analysis_mode": "trial_ml_safe"}})
        payload = (
            pd.DataFrame({"power_alpha_global_mean": [1.0, 2.0]}),
            np.array([10.0, 20.0], dtype=float),
            "rating",
            pd.DataFrame(
                {
                    "subject_id": ["sub-0001", "sub-0001"],
                    "trial_id": [1, 2],
                }
            ),
        )

        with patch.object(
            ml_data,
            "_load_subject_ml_from_features",
            side_effect=[payload, FileNotFoundError("missing requested feature table")],
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing requested feature table"):
                ml_data.load_active_matrix(
                    subjects=["0001", "0002"],
                    task="task",
                    deriv_root=Path("/tmp/deriv"),
                    config=cfg,
                    feature_families=["power"],
                )

    def test_load_subject_ml_from_features_rejects_trial_id_order_mismatch(self):
        ml_data = self._import_ml_data()
        cfg = DotConfig(
            {
                "event_columns": {"outcome": ["rating"]},
                "machine_learning": {"data": {"require_trial_ml_safe": False}},
            }
        )

        with tempfile.TemporaryDirectory() as td:
            deriv_root = Path(td)
            power_dir = deriv_root / "sub-0001" / "eeg" / "features" / "power"
            power_dir.mkdir(parents=True, exist_ok=True)
            (power_dir / "features_power.parquet").touch()

            feature_df = pd.DataFrame(
                {
                    "trial_id": [2, 1],
                    "power_alpha_global_mean": [1.0, 2.0],
                }
            )
            events_df = pd.DataFrame(
                {
                    "trial_id": [1, 2],
                    "rating": [10.0, 20.0],
                }
            )

            with patch.object(ml_data, "read_table", return_value=feature_df), patch.object(
                ml_data,
                "load_events_df",
                return_value=events_df,
            ):
                with self.assertRaisesRegex(ValueError, "trial_id"):
                    ml_data._load_subject_ml_from_features(
                        subject="0001",
                        task="task",
                        deriv_root=deriv_root,
                        config=cfg,
                        feature_families=["power"],
                        feature_input_root=None,
                        target="rating",
                        target_kind="continuous",
                        binary_threshold=None,
                        logger=Mock(),
                    )

    def test_load_epochs_with_targets_raises_when_requested_subject_has_no_epochs(self):
        ml_data = self._import_ml_data()

        fake_mne = types.SimpleNamespace(
            channels=types.SimpleNamespace(make_standard_montage=lambda name: f"montage:{name}")
        )

        with patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(None, None),
        ), patch.object(
            ml_data,
            "mne",
            fake_mne,
        ):
            with self.assertRaisesRegex(RuntimeError, "No aligned epochs/events for sub-0001"):
                ml_data.load_epochs_with_targets(
                    deriv_root=Path("."),
                    config=DotConfig({"event_columns": {"outcome": ["rating"]}}),
                    subjects=["0001"],
                    task="task",
                    target="rating",
                    logger=Mock(),
                )

    def test_load_epochs_with_targets_discovers_subjects_from_epochs_when_subjects_omitted(self):
        ml_data = self._import_ml_data()
        captured: dict[str, object] = {}

        def _fake_discovery(**kwargs):
            captured["discovery_sources"] = kwargs.get("discovery_sources")
            return ["0001"]

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.utils.data.subjects": types.SimpleNamespace(
                    get_available_subjects=_fake_discovery
                )
            },
        ), patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(None, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "No aligned epochs/events for sub-0001"):
                ml_data.load_epochs_with_targets(
                    deriv_root=Path("."),
                    config=DotConfig({"event_columns": {"outcome": ["rating"]}}),
                    subjects=None,
                    task="task",
                    target="rating",
                    bids_root=Path("."),
                    logger=Mock(),
                )

        self.assertEqual(captured["discovery_sources"], ["derivatives_epochs"])

    def test_load_channels_mean_matrix_rejects_empty_active_window(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        class _EpochsStub:
            def __init__(self):
                self.info = {"bads": []}
                self.times = np.array([-0.2, 0.0, 0.2], dtype=float)
                self.ch_names = ["C3", "C4"]
                self.baseline = None

            def get_data(self, picks=None):
                _ = picks
                return np.ones((2, 2, 3), dtype=float)

        aligned_events = pd.DataFrame({"rating": [10.0, 20.0]})
        config = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-0.2, 0.0],
                    "active_window": [1.0, 2.0],
                }
            }
        )

        with patch(
            "eeg_pipeline.utils.data.epochs.load_epochs_for_analysis",
            return_value=(_EpochsStub(), aligned_events),
        ), patch.object(ml_data.mne, "pick_types", return_value=np.array([0, 1], dtype=int)):
            with self.assertRaisesRegex(ValueError, "Active window empty"):
                ml_data.load_channels_mean_matrix(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path("."),
                    config=config,
                    target="rating",
                )

    def test_load_channels_mean_matrix_rejects_empty_baseline_window_when_epochs_are_not_baselined(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        class _EpochsStub:
            def __init__(self):
                self.info = {"bads": []}
                self.times = np.array([0.0, 0.1, 0.2], dtype=float)
                self.ch_names = ["C3", "C4"]
                self.baseline = None

            def get_data(self, picks=None):
                _ = picks
                return np.ones((2, 2, 3), dtype=float)

        aligned_events = pd.DataFrame({"rating": [10.0, 20.0]})
        config = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-2.0, -1.0],
                    "active_window": [0.0, 0.2],
                }
            }
        )

        with patch(
            "eeg_pipeline.utils.data.epochs.load_epochs_for_analysis",
            return_value=(_EpochsStub(), aligned_events),
        ), patch.object(ml_data.mne, "pick_types", return_value=np.array([0, 1], dtype=int)):
            with self.assertRaisesRegex(ValueError, "Baseline window empty"):
                ml_data.load_channels_mean_matrix(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path("."),
                    config=config,
                    target="rating",
                )

    def test_load_channels_mean_matrix_preserves_canonical_trial_ids(self):
        ml_data = self._import_ml_data()

        class _EpochsStub:
            def __init__(self):
                self.info = {"bads": []}
                self.times = np.array([-0.2, 0.0, 0.2], dtype=float)
                self.ch_names = ["C3", "C4"]
                self.baseline = None

            def get_data(self, picks=None):
                _ = picks
                return np.ones((2, 2, 3), dtype=float)

        aligned_events = pd.DataFrame(
            {
                "trial_id": [7, 9],
                "rating": [10.0, 20.0],
            }
        )
        config = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-0.2, 0.0],
                    "active_window": [0.0, 0.2],
                }
            }
        )

        with patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(_EpochsStub(), aligned_events),
        ), patch.object(ml_data.mne, "pick_types", return_value=np.array([0, 1], dtype=int)):
            _X, _y, _groups, _feature_names, meta = ml_data.load_channels_mean_matrix(
                subjects=["0001"],
                task="task",
                deriv_root=Path("."),
                config=config,
                target="rating",
            )

        self.assertEqual(meta["trial_id"].tolist(), [7, 9])

    def test_load_channels_mean_matrix_raises_when_requested_subject_has_no_epochs(self):
        ml_data = self._import_ml_data()

        config = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-0.2, 0.0],
                    "active_window": [0.0, 0.2],
                }
            }
        )

        with patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(None, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "No epochs for sub-0001"):
                ml_data.load_channels_mean_matrix(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path("."),
                    config=config,
                    target="rating",
                )

    def test_load_active_matrix_blocks_target_covariate_leakage_for_explicit_target_column(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        config = DotConfig(
            {
                "feature_engineering": {"analysis_mode": "trial_ml_safe"},
                "event_columns": {"outcome": ["vas_final_rating", "rating"]},
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
            return x_df.copy(), np.array([10.0, 20.0], dtype=float), "vas_final_rating", meta_df.copy()

        with patch.object(ml_data, "_load_subject_ml_from_features", side_effect=_fake_load_subject):
            with self.assertRaisesRegex(ValueError, "Covariates include the selected target"):
                ml_data.load_active_matrix(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path("."),
                    config=config,
                    feature_families=["power"],
                    target="vas_final_rating",
                    covariates=["vas_final_rating"],
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
                "vas_final_rating": y,
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
                        task="task",
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

    def test_incremental_validity_blocks_target_baseline_predictor_for_explicit_target_column(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        config = DotConfig({"event_columns": {"outcome": ["vas_final_rating", "rating"]}})
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
                        task="task",
                        deriv_root=Path(td),
                        config=config,
                        n_perm=0,
                        inner_splits=3,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        target="vas_final_rating",
                        baseline_predictors=["vas_final_rating"],
                    )

    def test_time_generalization_raises_on_stage_error(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "time_generalization_regression", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "Time-generalization stage failed"):
                    orch.run_time_generalization(
                        subjects=["0001", "0002"],
                        task="task",
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
                        task="task",
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
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                        "time_generalization": {
                            "active_window": [0.0, 0.1],
                            "window_len": 0.1,
                            "step": 0.1,
                            "min_samples_per_window": 1,
                            "min_samples_for_corr": 1,
                            "min_valid_fold_fraction": 0.8,
                            "min_subjects_per_cell": 2,
                            "min_count_per_cell": 1,
                            "cluster_threshold": 0.05,
                            "use_ridgecv": False,
                            "alpha_grid": [0.01, 0.1, 1.0],
                            "default_alpha": 1.0,
                        },
                    },
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
                task="task",
                results_dir=None,
                config_dict=cfg,
                n_perm=0,
                seed=42,
            )
        self.assertEqual(tg_r.size, 0)
        self.assertEqual(tg_r2.size, 0)
        self.assertEqual(window_centers.size, 0)

    def test_time_generalization_passes_explicit_target_to_epoch_loader(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        captured: dict[str, object] = {}

        def _fake_loader(*_args, **kwargs):
            captured["target"] = kwargs.get("target")
            raise RuntimeError("stop")

        with patch.object(tg, "load_epochs_with_targets", side_effect=_fake_loader):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                tg.time_generalization_regression(
                    deriv_root=Path("."),
                    subjects=["0001"],
                    task="task",
                    results_dir=None,
                    config_dict=DotConfig({}),
                    n_perm=0,
                    seed=42,
                    target="temperature",
                )

        self.assertEqual(captured.get("target"), "temperature")

    def test_time_generalization_passes_runtime_config_to_epoch_loader(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        captured: dict[str, object] = {}
        cfg = DotConfig({"event_columns": {"outcome": ["rating"]}})

        def _fake_loader(*_args, **kwargs):
            captured["config"] = kwargs.get("config")
            raise RuntimeError("stop")

        with patch.object(tg, "load_epochs_with_targets", side_effect=_fake_loader):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                tg.time_generalization_regression(
                    deriv_root=Path("."),
                    subjects=["0001"],
                    task="task",
                    results_dir=None,
                    config_dict=cfg,
                    n_perm=0,
                    seed=42,
                )

        self.assertIs(captured.get("config"), cfg)

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

    def test_uncertainty_exports_fold_and_subject_provenance(self):
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

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        cfg = DotConfig(
            {
                "machine_learning": {
                    "cv": {"inner_splits": 2},
                    "analysis": {"uncertainty": {"min_valid_fold_fraction": 0.5}},
                }
            }
        )

        def _fake_compute_prediction_intervals(**kwargs):
            x_test = np.asarray(kwargs["X_test"], dtype=float)
            return _FakeInterval(
                y_pred=np.zeros(len(x_test), dtype=float),
                lower=np.full(len(x_test), -1.0, dtype=float),
                upper=np.full(len(x_test), 1.0, dtype=float),
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
                output_path = orch._run_uncertainty_stage(
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
            self.assertIsNotNone(output_path)
            pred_df = pd.read_csv(Path(td) / "prediction_intervals.tsv", sep="\t")
            self.assertIn("fold", pred_df.columns)
            self.assertIn("subject_id", pred_df.columns)
            self.assertIn("test_index", pred_df.columns)
            self.assertIn("in_interval", pred_df.columns)

            with open(Path(td) / "metrics" / "uncertainty_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
            self.assertIn("subject_level", metrics)
            self.assertIn("mean_coverage", metrics["subject_level"])
            self.assertIn("n_subjects", metrics["subject_level"])

    def test_subject_level_r_uses_equal_subject_weighting_by_default(self):
        from eeg_pipeline.analysis.machine_learning.cv import compute_subject_level_r

        subj_a_true = np.arange(50, dtype=float)
        subj_a_pred = subj_a_true.copy()
        subj_b_true = np.arange(5, dtype=float)
        subj_b_pred = subj_b_true[::-1].copy()

        pred_df = pd.DataFrame(
            {
                "subject_id": (["sub-0001"] * len(subj_a_true)) + (["sub-0002"] * len(subj_b_true)),
                "y_true": np.concatenate([subj_a_true, subj_b_true]),
                "y_pred": np.concatenate([subj_a_pred, subj_b_pred]),
            }
        )

        agg_r, per_subject, _, _ = compute_subject_level_r(pred_df, config=DotConfig({}), ci_method="fixed_effects")
        self.assertEqual(len(per_subject), 2)
        self.assertLess(abs(float(agg_r)), 0.2)

    def test_subject_level_r_rejects_invalid_subjects(self):
        from eeg_pipeline.analysis.machine_learning.cv import compute_subject_level_r

        pred_df = pd.DataFrame(
            {
                "subject_id": ["sub-0001"] * 4 + ["sub-0002"] * 4,
                "y_true": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                "y_pred": [0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Invalid subject-level correlation"):
            compute_subject_level_r(pred_df, config=DotConfig({}), ci_method="fixed_effects")

    def test_subject_level_errors_reject_invalid_subjects(self):
        from eeg_pipeline.analysis.machine_learning.cv import compute_subject_level_errors

        pred_df = pd.DataFrame(
            {
                "subject_id": ["sub-0001"] * 3 + ["sub-0002"] * 3,
                "y_true": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "y_pred": [0.1, 1.1, 2.1, np.nan, np.nan, np.nan],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Invalid subject-level error inputs"):
            compute_subject_level_errors(pred_df, config=DotConfig({}), ci_method="fixed_effects")

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

    def test_cv_shap_raises_when_inner_grid_search_fails(self):
        from eeg_pipeline.analysis.machine_learning import shap_importance as si

        class _Model:
            def fit(self, X, y):
                _ = (X, y)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        class _FailingGridSearch:
            def __init__(self, *args, **kwargs):
                _ = (args, kwargs)

            def fit(self, *args, **kwargs):
                _ = (args, kwargs)
                raise RuntimeError("inner cv failure")

        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.linspace(0.0, 1.0, 6)
        groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
        splits = [(np.array([0, 1, 2, 3], dtype=int), np.array([4, 5], dtype=int))]
        shap_result = si.SHAPResult(
            shap_values=np.ones((2, 2), dtype=float),
            expected_value=0.0,
            feature_names=["f1", "f2"],
            X=X[4:6],
        )

        with patch.object(si, "_check_shap_available", return_value=True), patch.object(
            si, "GridSearchCV", _FailingGridSearch
        ), patch.object(si, "compute_shap_values", return_value=shap_result):
            with self.assertRaisesRegex(RuntimeError, "inner CV failed"):
                si.compute_shap_for_cv_folds(
                    model_factory=_Model,
                    X=X,
                    y=y,
                    cv_splits=splits,
                    feature_names=["f1", "f2"],
                    groups=groups,
                    param_grid={"alpha": [1.0]},
                    inner_cv_splits=2,
                )

    def test_cv_shap_raises_when_fold_explanation_fails(self):
        from eeg_pipeline.analysis.machine_learning import shap_importance as si

        class _Model:
            def fit(self, X, y):
                _ = (X, y)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.linspace(0.0, 1.0, 4)
        splits = [(np.array([0, 1], dtype=int), np.array([2, 3], dtype=int))]

        with patch.object(si, "_check_shap_available", return_value=True), patch.object(
            si, "compute_shap_values", side_effect=RuntimeError("shap failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "SHAP computation failed"):
                si.compute_shap_for_cv_folds(
                    model_factory=_Model,
                    X=X,
                    y=y,
                    cv_splits=splits,
                    feature_names=["f1", "f2"],
                )

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
                    task="task",
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
            with open(out_dir / "metrics" / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
        self.assertAlmostEqual(float(metrics["brier_score"]), 0.01, places=6)
        self.assertAlmostEqual(float(metrics["expected_calibration_error"]), 0.1, places=6)

    def test_classification_primary_balanced_accuracy_stays_subject_level_only(self):
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
        y = np.array([0, 0, 1, 1], dtype=int)
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
            y_prob=np.array([0.1, 0.2, 0.8, 0.9], dtype=float),
            groups=groups,
            failed_fold_count=0,
            n_folds_total=2,
        )
        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {"min_subjects_with_auc_for_inference": 1},
                    "evaluation": {"bootstrap_iterations": 200},
                },
                "project": {"random_state": 7},
            }
        )
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
                    task="task",
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
            with open(out_dir / "metrics" / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
        self.assertTrue(np.isnan(float(metrics["balanced_accuracy"])))
        self.assertTrue(np.isnan(float(metrics["subject_level"]["balanced_accuracy_mean"])))

    def test_classification_primary_precision_recall_f1_are_subject_level(self):
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
            y_pred=np.array([0, 1, 0, 0], dtype=int),
            y_prob=np.array([0.1, 0.9, 0.2, 0.4], dtype=float),
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
                    task="task",
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
            with open(out_dir / "metrics" / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

        self.assertAlmostEqual(float(metrics["precision"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["recall"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["f1"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["pooled_trials"]["precision"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["pooled_trials"]["recall"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["pooled_trials"]["f1"]), 2.0 / 3.0, places=6)

    def test_group_classification_exports_fold_and_trial_provenance_files(self):
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
                "trial_id": [10, 11, 12, 13],
                "block": [0, 1, 0, 1],
            }
        )
        result = ClassificationResult(
            y_true=y,
            y_pred=np.array([0, 1, 1, 0], dtype=int),
            y_prob=np.array([0.1, 0.9, 0.8, 0.2], dtype=float),
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
                    task="task",
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

            pred_df = pd.read_csv(out_dir / "data" / "loso_predictions.tsv", sep="\t")
            self.assertIn("trial_id", pred_df.columns)
            self.assertIn("fold", pred_df.columns)
            self.assertIn("model", pred_df.columns)
            self.assertTrue((out_dir / "loso_indices.tsv").exists())

    def test_regression_permutation_completion_threshold_uses_config(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import cv

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        cfg = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                    },
                    "cv": {
                        "min_valid_permutation_fraction": 0.75,
                        "min_valid_permutation_fold_fraction": 0.75,
                        "permutation_scheme": "within_subject",
                    },
                }
            }
        )
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

    def test_model_comparison_supports_pairwise_inference_with_permutations(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            X = np.array(
                [
                    [0.0, 1.0],
                    [0.1, 1.1],
                    [0.2, 1.2],
                    [0.3, 1.3],
                    [0.4, 1.4],
                    [0.5, 1.5],
                ],
                dtype=float,
            )
            y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
            groups = np.array(
                ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
                dtype=object,
            )
            meta = pd.DataFrame({"subject_id": groups, "trial_id": np.arange(len(groups), dtype=int)})

            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)):
                out_dir = orch.run_model_comparison_ml(
                    subjects=["0001", "0002"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig(
                        {
                            "machine_learning": {
                                "preprocessing": {"variance_threshold_grid": [0.0]}
                            }
                        }
                    ),
                    n_perm=8,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                )
            with open(out_dir / "metrics" / "model_comparison_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertIn("pairwise_inference", summary)
            self.assertTrue(summary["pairwise_inference"])

    def test_incremental_validity_supports_delta_inference_with_permutations(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch,
                "load_active_matrix",
                return_value=(
                    np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float),
                    np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
                    np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object),
                    ["f1"],
                    pd.DataFrame(
                        {
                            "subject_id": np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object),
                            "trial_id": [0, 1, 2, 3],
                            "predictor": [44.0, 45.0, 46.0, 47.0],
                            "temperature": [44.0, 45.0, 46.0, 47.0],
                        }
                    ),
                ),
            ):
                out_dir = orch.run_incremental_validity_ml(
                    subjects=["0001", "0002"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig({}),
                    n_perm=8,
                    inner_splits=2,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    baseline_predictors=["predictor"],
                )
            with open(out_dir / "metrics" / "incremental_validity_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertIn("delta_r2_inference", summary)
            self.assertIn("p_value", summary["delta_r2_inference"])

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
                        task="task",
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
                        task="task",
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

        cfg = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                    },
                    "cv": {
                        "min_valid_permutation_fraction": 0.75,
                        "min_valid_permutation_fold_fraction": 0.75,
                        "permutation_scheme": "within_subject",
                    },
                }
            }
        )
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
                        task="task",
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
        cfg = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                    },
                    "cv": {
                        "min_valid_permutation_fraction": 0.75,
                        "min_valid_permutation_fold_fraction": 0.75,
                        "permutation_scheme": "within_subject",
                    },
                }
            }
        )
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
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                        "time_generalization": {
                            "active_window": [0.0, 0.1],
                            "window_len": 0.1,
                            "step": 0.1,
                            "min_samples_per_window": 1,
                            "min_samples_for_corr": 1,
                            "min_valid_fold_fraction": 0.8,
                            "min_subjects_per_cell": 2,
                            "min_count_per_cell": 1,
                            "cluster_threshold": 0.05,
                            "use_ridgecv": False,
                            "alpha_grid": [0.01, 0.1, 1.0],
                            "default_alpha": 1.0,
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
                    task="task",
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

        cfg = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                    },
                    "cv": {
                        "min_valid_permutation_fraction": 0.75,
                        "min_valid_permutation_fold_fraction": 0.75,
                        "permutation_scheme": "within_subject",
                    },
                }
            }
        )
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
                        task="task",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=2,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                    )

    def test_within_subject_regression_uses_fold_baseline_and_effective_subjects(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "block": [0, 1, 0, 0],
            }
        )
        # Only sub-0001 contributes evaluable folds.
        folds = [
            (0, np.array([0], dtype=int), np.array([1], dtype=int), "sub-0001", {}),
        ]

        class _FakePipe:
            def __init__(self):
                self.named_steps = {"regressor": types.SimpleNamespace(param_grid={})}

        class _FakeEstimator:
            def predict(self, X_in: np.ndarray) -> np.ndarray:
                return np.asarray(X_in[:, 0], dtype=float)

        def _fake_export_predictions(y_true, y_pred, groups_ordered, *_args, **_kwargs):
            return pd.DataFrame(
                {
                    "y_true": np.asarray(y_true, dtype=float),
                    "y_pred": np.asarray(y_pred, dtype=float),
                    "subject_id": np.asarray(groups_ordered, dtype=object),
                }
            )

        cfg = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                    }
                }
            }
        )
        with tempfile.TemporaryDirectory() as td:
            captured_groups_used = {}

            def _capture_subject_selection(_results_dir, _subjects, groups_used, _meta, _config):
                captured_groups_used["values"] = np.asarray(groups_used, dtype=object).tolist()
                return {}

            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch.object(
                orch, "create_elasticnet_pipeline", side_effect=lambda **_kwargs: _FakePipe()
            ), patch.object(
                orch, "build_elasticnet_param_grid", return_value={}
            ), patch.object(
                orch, "_fit_within_subject_fold", return_value=_FakeEstimator()
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
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ), patch.object(
                orch, "export_subject_selection_report", side_effect=_capture_subject_selection
            ):
                out_dir = orch.run_within_subject_regression_ml(
                    subjects=["0001", "0002"],
                    task="task",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                )

            with open(out_dir / "metrics" / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
            baseline_df = pd.read_csv(out_dir / "baseline_predictions.tsv", sep="\t")

        self.assertEqual(metrics["n_subjects"], 1)
        self.assertEqual(captured_groups_used["values"], ["sub-0001"])
        self.assertEqual(float(baseline_df.loc[0, "y_true"]), 2.0)
        self.assertEqual(float(baseline_df.loc[0, "y_pred_baseline"]), 1.0)

    def test_within_subject_regression_passes_rf_param_grid_to_inner_cv_helper(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0001", "sub-0001"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "block": [0, 0, 1, 1],
            }
        )
        folds = [
            (1, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int), "sub-0001", {}),
        ]

        class _FakeEstimator:
            def predict(self, X_in: np.ndarray) -> np.ndarray:
                return np.asarray(X_in[:, 0], dtype=float)

        captured: dict[str, object] = {}

        def _fake_fit_within_subject_fold(*, param_grid=None, **_kwargs):
            captured["param_grid"] = param_grid
            return _FakeEstimator()

        cfg = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "permutation_importance": {"enabled": False},
                        "shap": {"enabled": False},
                    }
                }
            }
        )
        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch.object(
                orch, "_fit_within_subject_fold", side_effect=_fake_fit_within_subject_fold
            ), patch.object(
                orch, "export_subject_selection_report", return_value={}
            ), patch.object(
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ):
                orch.run_within_subject_regression_ml(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    model="rf",
                )

        self.assertIsInstance(captured.get("param_grid"), dict)
        self.assertIn("rf__max_depth", captured.get("param_grid", {}))

    def test_tuned_loso_regression_requires_inner_cv_groups(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        pipe = Pipeline([("reg", DummyRegressor(strategy="mean"))])
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y_train = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
        groups_train = np.array(["sub-0001"] * 4, dtype=object)

        with self.assertRaisesRegex(RuntimeError, "inner CV requires at least 2 training groups"):
            orch._fit_tuned_regression_estimator(
                base_pipe=pipe,
                param_grid={"reg__strategy": ["mean"]},
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                inner_splits=3,
                seed=42,
                logger=Mock(),
                fold_info="fold 1",
            )

    def test_within_subject_regression_applies_fold_feature_harmonization(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
                [0.4, 4.0],
            ],
            dtype=float,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0001", "sub-0001"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "block": [0, 0, 1, 1],
            }
        )
        folds = [
            (1, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int), "sub-0001", {}),
        ]

        captured = {"harmonize_calls": 0, "n_test_features": None}

        class _FakeEstimator:
            def predict(self, X_in: np.ndarray) -> np.ndarray:
                captured["n_test_features"] = int(X_in.shape[1])
                return np.asarray(X_in[:, 0], dtype=float)

        def _fake_harmonize(X_train, X_test, groups_train, harmonization_mode, n_covariates=0):
            _ = (groups_train, harmonization_mode, n_covariates)
            captured["harmonize_calls"] += 1
            keep = np.array([False, True], dtype=bool)
            return X_train[:, keep], X_test[:, keep], keep

        def _fake_export_predictions(y_true, y_pred, groups_ordered, *_args, **_kwargs):
            return pd.DataFrame(
                {
                    "y_true": np.asarray(y_true, dtype=float),
                    "y_pred": np.asarray(y_pred, dtype=float),
                    "subject_id": np.asarray(groups_ordered, dtype=object),
                }
            )

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch.object(
                orch, "apply_fold_feature_harmonization", side_effect=_fake_harmonize
            ), patch.object(
                orch, "_fit_within_subject_fold", return_value=_FakeEstimator()
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
                orch, "write_reproducibility_info", return_value=Path(td) / "reproducibility_info.json"
            ), patch.object(
                orch, "export_subject_selection_report", return_value={}
            ):
                orch.run_within_subject_regression_ml(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig(
                        {
                            "machine_learning": {
                                "analysis": {
                                    "permutation_importance": {"enabled": False},
                                    "shap": {"enabled": False},
                                }
                            }
                        }
                    ),
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    feature_harmonization="intersection",
                )

        self.assertEqual(captured["harmonize_calls"], 1)
        self.assertEqual(captured["n_test_features"], 1)

    def test_within_subject_classification_permutation_failures_do_not_abort_inference(self):
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
            if fit_calls["n"] > 2:
                raise RuntimeError("synthetic cnn permutation crash")
            n_test = int(len(X_test))
            y_pred = np.array(([0, 1] * ((n_test + 1) // 2))[:n_test], dtype=int)
            y_prob = np.array(([0.1, 0.9] * ((n_test + 1) // 2))[:n_test], dtype=float)
            return y_pred, y_prob

        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {"max_failed_fold_fraction": 0.25, "min_subjects_with_auc_for_inference": 1},
                    "cv": {"min_valid_permutation_fraction": 0.0, "permutation_scheme": "within_subject"},
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
            ), patch.object(
                orch, "_maybe_generate_mode_plots", return_value=None
            ):
                results_dir = orch.run_within_subject_classification_ml(
                    subjects=["0001", "0002"],
                    task="task",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=1,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    classification_model="cnn",
                )

        self.assertEqual(results_dir, Path(td) / "within_subject_classification")

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
                    "cv": {
                        "min_valid_permutation_fraction": 0.75,
                        "permutation_scheme": "within_subject",
                    },
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
                    "cv": {
                        "min_valid_permutation_fraction": 0.75,
                        "permutation_scheme": "within_subject",
                    },
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

    def test_group_classification_permutation_failures_do_not_abort_inference(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {"max_failed_fold_fraction": 0.25},
                    "cv": {
                        "min_valid_permutation_fraction": 0.0,
                        "permutation_scheme": "within_subject",
                    },
                }
            }
        )

        with patch(
            "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
            side_effect=RuntimeError("synthetic permutation crash"),
        ):
            out = orch._run_classification_permutations(
                X=X,
                y=y,
                groups=groups,
                blocks=None,
                model="svm",
                inner_splits=2,
                seed=42,
                n_perm=1,
                config=cfg,
                logger=Mock(),
                harmonization_mode="intersection",
            )

        self.assertIsNone(out)

    def test_load_subject_feature_table_requires_explicit_trial_safe_provenance_when_strict(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        cfg = DotConfig({"machine_learning": {"data": {"require_trial_ml_safe": True}}})

        with tempfile.TemporaryDirectory() as td:
            deriv_root = Path(td)
            power_dir = deriv_root / "sub-0001" / "eeg" / "features" / "power"
            meta_dir = power_dir / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (power_dir / "features_power.parquet").touch()
            (meta_dir / "extraction_config.json").write_text(
                json.dumps({"cli_command": "eeg-pipeline features compute"}),
                encoding="utf-8",
            )

            with patch.object(
                ml_data,
                "read_table",
                return_value=pd.DataFrame(
                    {
                        "trial_id": [1, 2],
                        "power_alpha_global_mean": [1.0, 2.0],
                    }
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Cannot verify trial_ml_safe provenance"):
                    ml_data._load_subject_feature_table(
                        subject="0001",
                        task="task",
                        deriv_root=deriv_root,
                        config=cfg,
                        feature_families=["power"],
                        feature_input_root=None,
                        logger=Mock(),
                    )

    def test_load_subject_feature_table_accepts_explicit_trial_safe_analysis_mode(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        cfg = DotConfig({"machine_learning": {"data": {"require_trial_ml_safe": True}}})

        with tempfile.TemporaryDirectory() as td:
            deriv_root = Path(td)
            power_dir = deriv_root / "sub-0001" / "eeg" / "features" / "power"
            meta_dir = power_dir / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (power_dir / "features_power.parquet").touch()
            (meta_dir / "extraction_config.json").write_text(
                json.dumps(
                    {
                        "analysis_mode": "trial_ml_safe",
                        "connectivity_granularity": "trial",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                ml_data,
                "read_table",
                return_value=pd.DataFrame(
                    {
                        "trial_id": [1, 2],
                        "power_alpha_global_mean": [1.0, 2.0],
                    }
                ),
            ):
                df, cols = ml_data._load_subject_feature_table(
                    subject="0001",
                    task="task",
                    deriv_root=deriv_root,
                    config=cfg,
                    feature_families=["power"],
                    feature_input_root=None,
                    logger=Mock(),
                )

        self.assertEqual(df.shape, (2, 1))
        self.assertEqual(cols, ["power_alpha_global_mean"])

    def test_load_subject_feature_table_connectivity_requires_explicit_trial_granularity_when_strict(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data

        cfg = DotConfig({"machine_learning": {"data": {"require_trial_ml_safe": True}}})

        with tempfile.TemporaryDirectory() as td:
            deriv_root = Path(td)
            conn_dir = deriv_root / "sub-0001" / "eeg" / "features" / "connectivity"
            meta_dir = conn_dir / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (conn_dir / "features_connectivity.parquet").touch()
            (meta_dir / "extraction_config.json").write_text(
                json.dumps({"analysis_mode": "trial_ml_safe"}),
                encoding="utf-8",
            )

            with patch.object(
                ml_data,
                "read_table",
                return_value=pd.DataFrame(
                    {
                        "trial_id": [1, 2],
                        "conn_alpha_global_wpli_mean": [0.1, 0.2],
                    }
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Cannot verify connectivity granularity"):
                    ml_data._load_subject_feature_table(
                        subject="0001",
                        task="task",
                        deriv_root=deriv_root,
                        config=cfg,
                        feature_families=["connectivity"],
                        feature_input_root=None,
                        logger=Mock(),
                    )

    def test_incremental_validity_raises_when_harmonization_removes_all_baseline_predictors(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        cfg = DotConfig({})
        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
                "temperature": [44.0, 45.0, 46.0, 47.0],
            }
        )

        def _drop_baseline_columns(X_train, X_test, groups_train, harmonization_mode, n_covariates=0):
            _ = (groups_train, harmonization_mode, n_covariates)
            keep = np.array([True, False], dtype=bool)
            return X_train[:, keep], X_test[:, keep], keep

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "apply_fold_feature_harmonization", side_effect=_drop_baseline_columns
            ):
                with self.assertRaisesRegex(ValueError, "removed all baseline predictors"):
                    orch.run_incremental_validity_ml(
                        subjects=["0001", "0002"],
                        task="task",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=0,
                        inner_splits=2,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        target="rating",
                        baseline_predictors=["temperature"],
                    )

    def test_split_conformal_requires_minimum_training_samples(self):
        from sklearn.dummy import DummyRegressor

        from eeg_pipeline.analysis.machine_learning.uncertainty import _conformal_split

        X_train = np.arange(4.0).reshape(-1, 1)
        y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        X_test = np.array([[5.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "at least 5 training samples"):
            _conformal_split(
                model=DummyRegressor(strategy="mean"),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                alpha=0.1,
                rng=np.random.default_rng(7),
            )

    def test_model_comparison_uses_correlation_refit_objective(self):
        from sklearn.dummy import DummyRegressor

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.0, 1.0],
                [0.1, 1.1],
                [0.2, 1.2],
                [0.3, 1.3],
                [0.4, 1.4],
                [0.5, 1.5],
            ],
            dtype=float,
        )
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        meta = pd.DataFrame({"subject_id": groups, "trial_id": np.arange(len(groups), dtype=int)})
        calls = []

        class _CaptureGrid:
            def __init__(self, estimator, param_grid, cv, scoring, n_jobs, error_score, refit=None):
                _ = (estimator, param_grid, cv, n_jobs, error_score)
                calls.append({"scoring": scoring, "refit": refit})
                self.best_params_ = {}

            def fit(self, X_fit, y_fit, groups=None):
                _ = (X_fit, y_fit, groups)
                return self

            def predict(self, X_pred):
                return np.zeros(len(X_pred), dtype=float)

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)), patch.object(
                orch, "GridSearchCV", _CaptureGrid
            ), patch.object(
                orch, "create_elasticnet_pipeline", return_value=DummyRegressor(strategy="mean")
            ), patch.object(
                orch, "create_ridge_pipeline", return_value=DummyRegressor(strategy="mean")
            ), patch.object(
                orch, "create_rf_pipeline", return_value=DummyRegressor(strategy="mean")
            ), patch.object(
                orch, "build_elasticnet_param_grid", return_value={}
            ), patch.object(
                orch, "build_ridge_param_grid", return_value={}
            ), patch.object(
                orch, "build_rf_param_grid", return_value={}
            ):
                orch.run_model_comparison_ml(
                    subjects=["0001", "0002", "0003"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig({}),
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                )

        self.assertTrue(calls)
        for rec in calls:
            self.assertIsInstance(rec["scoring"], dict)
            self.assertIn("r", rec["scoring"])
            self.assertEqual(rec["refit"], "r")

    def test_load_epoch_tensor_matrix_preserves_canonical_trial_ids(self):
        ml_data = self._import_ml_data()

        class _EpochsStub:
            def __init__(self):
                self.info = {"bads": []}
                self.ch_names = ["C3", "C4"]

            def __getitem__(self, item):
                if isinstance(item, slice):
                    return self
                raise TypeError("Unexpected index type")

            def copy(self):
                return self

            def pick(self, channels):
                _ = channels
                return self

            def get_data(self, picks=None):
                _ = picks
                return np.ones((2, 2, 4), dtype=float)

            def __len__(self):
                return 2

        aligned_events = pd.DataFrame(
            {
                "trial_id": [14, 18],
                "rating": [1.0, 2.0],
            }
        )

        with patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(_EpochsStub(), aligned_events),
        ), patch.object(ml_data.mne, "pick_types", return_value=np.array([0, 1], dtype=int)):
            _X, _y, _groups, _channels, meta = ml_data.load_epoch_tensor_matrix(
                subjects=["0001"],
                task="task",
                deriv_root=Path("."),
                config=DotConfig({}),
                target="rating",
            )

        self.assertEqual(meta["trial_id"].tolist(), [14, 18])

    def test_load_epoch_tensor_matrix_raises_when_requested_subject_has_no_epochs(self):
        ml_data = self._import_ml_data()

        with patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(None, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "No epochs for sub-0001"):
                ml_data.load_epoch_tensor_matrix(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path("."),
                    config=DotConfig({}),
                    target="rating",
                )

    def test_load_epochs_with_targets_supports_resting_state_without_clean_events_file(self):
        ml_data = self._import_ml_data()

        class _EpochsStub:
            def __init__(self):
                self.info = {"bads": [], "ch_names": ["C3", "C4"]}
                self.metadata = None
                self.montage = None
                self.interpolated = False

            def set_montage(self, montage):
                self.montage = montage

            def interpolate_bads(self, reset_bads=True):
                _ = reset_bads
                self.interpolated = True

            def get_channel_types(self, picks=None):
                _ = picks
                return ["eeg"]

            def __len__(self):
                return 2

        epochs = _EpochsStub()
        aligned = pd.DataFrame({"trial_id": [1, 2], "rating": [10.0, 20.0]})
        fake_mne = types.SimpleNamespace(
            channels=types.SimpleNamespace(make_standard_montage=lambda name: f"montage:{name}")
        )

        with patch.object(
            ml_data,
            "load_epochs_for_analysis",
            return_value=(epochs, aligned),
        ), patch.object(
            ml_data,
            "mne",
            fake_mne,
        ):
            tuples, common_channels = ml_data.load_epochs_with_targets(
                deriv_root=Path("."),
                config=DotConfig(
                    {
                        "event_columns": {"outcome": ["rating"]},
                    }
                ),
                subjects=["0001"],
                task="rest",
                target="rating",
                logger=Mock(),
            )

        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuples[0][0], "sub-0001")
        self.assertEqual(tuples[0][2].tolist(), [10.0, 20.0])
        self.assertEqual(common_channels, ["C3", "C4"])
        self.assertEqual(epochs.metadata["trial_id"].tolist(), [1, 2])
        self.assertEqual(epochs.montage, "montage:standard_1005")

    def test_incremental_validity_uses_correlation_refit_objective(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.0],
                [0.1],
                [0.2],
                [0.3],
                [0.4],
                [0.5],
            ],
            dtype=float,
        )
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "predictor": np.linspace(44.0, 46.5, len(groups)),
                "temperature": np.linspace(44.0, 46.5, len(groups)),
            }
        )
        calls = []

        class _CaptureGrid:
            def __init__(self, estimator, param_grid, cv, scoring, n_jobs, error_score, refit=None):
                _ = (estimator, param_grid, cv, n_jobs, error_score)
                calls.append({"scoring": scoring, "refit": refit})
                self.best_params_ = {}

            def fit(self, X_fit, y_fit, groups=None):
                _ = (X_fit, y_fit, groups)
                return self

            def predict(self, X_pred):
                if X_pred.shape[1] == 1:
                    return np.full(len(X_pred), -1.0, dtype=float)
                return np.full(len(X_pred), 1.0, dtype=float)

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "GridSearchCV", _CaptureGrid
            ), patch(
                "sklearn.metrics.r2_score", return_value=0.0
            ):
                orch.run_incremental_validity_ml(
                    subjects=["0001", "0002", "0003"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig({}),
                    n_perm=0,
                    inner_splits=2,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    baseline_predictors=["predictor"],
                )

        self.assertTrue(calls)
        for rec in calls:
            self.assertIsInstance(rec["scoring"], dict)
            self.assertIn("r", rec["scoring"])
            self.assertEqual(rec["refit"], "r")

    def test_incremental_validity_uses_subject_level_delta_as_primary_metric(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.0],
                [0.1],
                [0.2],
                [0.3],
                [0.4],
                [0.5],
            ],
            dtype=float,
        )
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "predictor": np.linspace(44.0, 46.5, len(groups)),
                "temperature": np.linspace(44.0, 46.5, len(groups)),
            }
        )
        n_total = int(len(y))

        class _FakeGrid:
            def __init__(self, estimator, param_grid, cv, scoring, n_jobs, error_score, refit=None):
                _ = (estimator, param_grid, cv, scoring, n_jobs, error_score, refit)
                self.best_params_ = {}

            def fit(self, X_fit, y_fit, groups=None):
                _ = (X_fit, y_fit, groups)
                return self

            def predict(self, X_pred):
                if X_pred.shape[1] == 1:
                    return np.full(len(X_pred), -1.0, dtype=float)
                return np.full(len(X_pred), 1.0, dtype=float)

        def _fake_r2_score(y_true_in, y_pred_in):
            y_pred_arr = np.asarray(y_pred_in, dtype=float)
            if len(y_true_in) == n_total:
                return 0.9 if np.all(y_pred_arr == -1.0) else 0.95
            return 0.2 if np.all(y_pred_arr == -1.0) else 0.6

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)), patch.object(
                orch, "GridSearchCV", _FakeGrid
            ), patch(
                "sklearn.metrics.r2_score", side_effect=_fake_r2_score
            ):
                out_dir = orch.run_incremental_validity_ml(
                    subjects=["0001", "0002", "0003"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig({}),
                    n_perm=0,
                    inner_splits=2,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    baseline_predictors=["predictor"],
                )

            with open(out_dir / "metrics" / "incremental_validity_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)

        self.assertAlmostEqual(float(summary["mean_fold_delta_r2"]), 0.4, places=8)
        self.assertAlmostEqual(float(summary["delta_r2"]), 0.4, places=8)
        self.assertIn("pooled_trials", summary)
        self.assertAlmostEqual(float(summary["pooled_trials"]["delta_r2"]), 0.05, places=8)

    def test_regression_permutation_requires_effective_label_shuffling(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import cv

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([1.0, 2.0, 1.0, 2.0], dtype=float)
        cfg = DotConfig(
            {
                "machine_learning": {
                    "cv": {
                        "permutation_scheme": "within_subject_within_block",
                        "min_label_shuffle_fraction": 0.1,
                        "min_valid_permutation_fraction": 0.0,
                    }
                }
            }
        )

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "No effective permutations"):
                cv.run_permutation_test(
                    X=X,
                    y=y,
                    groups=groups,
                    blocks=blocks,
                    pipe=Pipeline([("regressor", DummyRegressor(strategy="mean"))]),
                    param_grid={},
                    inner_cv_splits=2,
                    inner_n_jobs=1,
                    seed=42,
                    model_name="elasticnet",
                    null_n_perm=3,
                    null_output_path=Path(td) / "null.npz",
                    config=cfg,
                )

    def test_resolve_permutation_scheme_rejects_invalid_value(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        cfg = DotConfig(
            {
                "machine_learning": {
                    "cv": {"permutation_scheme": "not-a-scheme"},
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "Invalid machine_learning.cv.permutation_scheme"):
            orch._resolve_permutation_scheme(cfg)

    def test_resolve_permutation_scheme_defaults_to_within_subject(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        self.assertEqual(orch._resolve_permutation_scheme(DotConfig({})), "within_subject")

    def test_generate_effective_permutation_requires_blocks_for_blockwise_scheme(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)

        with self.assertRaisesRegex(ValueError, "requires block labels"):
            orch._generate_effective_permutation(
                y,
                groups,
                blocks=None,
                rng=np.random.default_rng(42),
                requested_scheme="within_subject_within_block",
                min_changed_fraction=0.1,
            )

    def test_generate_effective_permutation_rejects_block_length_mismatch(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([1.0, 2.0], dtype=float)

        with self.assertRaisesRegex(ValueError, "same length as y"):
            orch._generate_effective_permutation(
                y,
                groups,
                blocks=blocks,
                rng=np.random.default_rng(42),
                requested_scheme="within_subject_within_block",
                min_changed_fraction=0.1,
            )

    def test_generate_effective_permutation_rejects_all_missing_blocks(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

        with self.assertRaisesRegex(ValueError, "requires block labels"):
            orch._generate_effective_permutation(
                y,
                groups,
                blocks=blocks,
                rng=np.random.default_rng(42),
                requested_scheme="within_subject_within_block",
                min_changed_fraction=0.1,
            )

    def test_run_permutation_test_rejects_all_missing_blocks_for_blockwise_scheme(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import cv

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
        cfg = DotConfig(
            {
                "machine_learning": {
                    "cv": {
                        "permutation_scheme": "within_subject_within_block",
                        "min_valid_permutation_fraction": 0.0,
                    }
                }
            }
        )

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "requires block labels"):
                cv.run_permutation_test(
                    X=X,
                    y=y,
                    groups=groups,
                    blocks=blocks,
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

    def test_generate_effective_permutation_does_not_downgrade_blockwise_scheme(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([1.0, 2.0, 1.0, 2.0], dtype=float)

        y_perm, effective, changed_fraction, used_scheme = orch._generate_effective_permutation(
            y,
            groups,
            blocks=blocks,
            rng=np.random.default_rng(42),
            requested_scheme="within_subject_within_block",
            min_changed_fraction=0.1,
        )

        self.assertFalse(effective)
        self.assertEqual(used_scheme, "within_subject_within_block")
        self.assertAlmostEqual(changed_fraction, 0.0, places=8)
        np.testing.assert_array_equal(y_perm, y)

    def test_nested_loso_classification_raises_when_training_fold_has_one_class(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import classification as clf

        X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
        y = np.array([0, 0, 1, 1], dtype=int)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
            dtype=object,
        )
        pipe = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        cfg = DotConfig({"machine_learning": {"classification": {"scoring": "accuracy"}}})

        with patch.object(clf, "create_svm_pipeline", return_value=pipe), patch.object(
            clf, "build_svm_param_grid", return_value={}
        ):
            with self.assertRaisesRegex(RuntimeError, "only one class in training"):
                clf.nested_loso_classification(
                    X=X,
                    y=y,
                    groups=groups,
                    model="svm",
                    inner_splits=2,
                    seed=42,
                    config=cfg,
                    logger=Mock(),
                )

    def test_nested_loso_classification_requires_inner_cv_groups_for_tuning(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import classification as clf

        X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
            dtype=object,
        )
        pipe = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        cfg = DotConfig({"machine_learning": {"classification": {"scoring": "accuracy"}}})

        with patch.object(clf, "create_svm_pipeline", return_value=pipe), patch.object(
            clf, "build_svm_param_grid", return_value={"clf__strategy": ["most_frequent"]}
        ):
            with self.assertRaisesRegex(RuntimeError, "inner CV requires at least 2 training groups"):
                clf.nested_loso_classification(
                    X=X,
                    y=y,
                    groups=groups,
                    model="svm",
                    inner_splits=2,
                    seed=42,
                    config=cfg,
                    logger=Mock(),
                )

    def test_nested_loso_classification_requires_stratified_inner_cv_support(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import classification as clf

        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array([0, 1, 0, 1, 0, 0], dtype=int)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        pipe = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        cfg = DotConfig({"machine_learning": {"classification": {"scoring": "accuracy"}}})

        with patch.object(clf, "create_svm_pipeline", return_value=pipe), patch.object(
            clf, "build_svm_param_grid", return_value={"clf__strategy": ["most_frequent"]}
        ):
            with self.assertRaisesRegex(RuntimeError, "StratifiedGroupKFold"):
                clf.nested_loso_classification(
                    X=X,
                    y=y,
                    groups=groups,
                    model="svm",
                    inner_splits=2,
                    seed=42,
                    config=cfg,
                    logger=Mock(),
                )

    def test_create_within_subject_folds_supports_forward_ordering(self):
        from eeg_pipeline.analysis.machine_learning.cv import create_within_subject_folds

        groups = np.array(["sub-0001"] * 6, dtype=object)
        blocks = np.array([1, 1, 2, 2, 3, 3], dtype=float)
        cfg = DotConfig({"machine_learning": {"cv": {"within_subject_ordered_blocks": True}}})

        folds = create_within_subject_folds(
            groups=groups,
            blocks_all=blocks,
            inner_cv_splits=3,
            outer_cv_splits=2,
            seed=42,
            config=cfg,
            epochs=None,
            apply_hygiene=False,
        )

        self.assertEqual(len(folds), 2)
        for _fold_id, train_idx, test_idx, _subject, _params in folds:
            train_blocks = blocks[np.asarray(train_idx, dtype=int)]
            test_blocks = blocks[np.asarray(test_idx, dtype=int)]
            self.assertLess(np.max(train_blocks), np.min(test_blocks))

    def test_create_within_subject_folds_raises_when_ordered_blocks_cannot_be_formed(self):
        from eeg_pipeline.analysis.machine_learning.cv import create_within_subject_folds

        groups = np.array(["sub-0001"] * 4, dtype=object)
        blocks = np.array(["run-a", "run-a", "run-b", "run-b"], dtype=object)
        cfg = DotConfig({"machine_learning": {"cv": {"within_subject_ordered_blocks": True}}})

        with self.assertRaisesRegex(ValueError, "ordered within-subject CV requested"):
            create_within_subject_folds(
                groups=groups,
                blocks_all=blocks,
                inner_cv_splits=2,
                outer_cv_splits=2,
                seed=42,
                config=cfg,
                epochs=None,
                apply_hygiene=False,
            )

    def test_run_within_subject_classification_raises_when_training_fold_has_one_class(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.ones((4, 1, 2), dtype=float)
        y = np.array([0, 0, 1, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "block": [0, 1, 0, 1],
            }
        )
        folds = [(1, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int), "sub-0001", None)]

        cfg = DotConfig({"machine_learning": {"classification": {"min_subjects_with_auc_for_inference": 1}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_epoch_tensor_matrix", return_value=(X, y, groups, ["f1"], meta)
            ), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ):
                with self.assertRaisesRegex(RuntimeError, "only one class in training"):
                    orch.run_within_subject_classification_ml(
                        subjects=["0001", "0002"],
                        task="task",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=0,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        classification_model="cnn",
                    )

    def test_run_within_subject_classification_raises_when_cnn_fold_fit_fails(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.ones((4, 1, 2), dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "block": [0, 1, 0, 1],
            }
        )
        folds = [(1, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int), "sub-0001", None)]

        cfg = DotConfig({"machine_learning": {"classification": {"min_subjects_with_auc_for_inference": 1}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_epoch_tensor_matrix", return_value=(X, y, groups, ["f1"], meta)
            ), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch(
                "eeg_pipeline.analysis.machine_learning.cnn.fit_predict_cnn_binary_classifier",
                side_effect=RuntimeError("synthetic cnn fold failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic cnn fold failure"):
                    orch.run_within_subject_classification_ml(
                        subjects=["0001", "0002"],
                        task="task",
                        deriv_root=Path(td),
                        config=cfg,
                        n_perm=0,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        classification_model="cnn",
                    )

    def test_run_within_subject_classification_raises_when_inner_cv_fails(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "block": [0, 1, 0, 1],
            }
        )
        folds = [(1, np.array([0, 1], dtype=int), np.array([2, 3], dtype=int), "sub-0001", None)]
        pipe = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        cfg = DotConfig({"machine_learning": {"classification": {"min_subjects_with_auc_for_inference": 1}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)
            ), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.create_svm_pipeline",
                return_value=pipe,
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.build_svm_param_grid",
                return_value={},
            ), patch(
                "eeg_pipeline.analysis.machine_learning.orchestration.GridSearchCV.fit",
                side_effect=RuntimeError("synthetic inner cv failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic inner cv failure"):
                    orch.run_within_subject_classification_ml(
                        subjects=["0001", "0002"],
                        task="task",
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

    def test_time_generalization_permutation_helper_rejects_invalid_scheme(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        y = np.array([0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001"], dtype=object)

        with self.assertRaisesRegex(ValueError, "Unsupported permutation scheme"):
            tg._permute_labels_within_subject_structure(
                y,
                groups,
                blocks=np.array([0.0, 1.0], dtype=float),
                rng=np.random.default_rng(42),
                scheme="not-a-scheme",
            )

    def test_time_generalization_permutation_helper_requires_blocks_for_blockwise_scheme(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        y = np.array([0.0, 1.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001"], dtype=object)

        with self.assertRaisesRegex(ValueError, "requires block labels"):
            tg._permute_labels_within_subject_structure(
                y,
                groups,
                blocks=None,
                rng=np.random.default_rng(42),
                scheme="within_subject_within_block",
            )

    def test_time_generalization_regression_raises_when_blockwise_permutations_lack_blocks(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        class _FakeEpochs:
            def __init__(self):
                self.times = np.array([0.0, 0.1], dtype=float)
                self.metadata = pd.DataFrame({"block": [np.nan, np.nan]})

            def copy(self):
                return _FakeEpochs()

            def pick(self, _chs):
                return self

        trial_records = [
            ("sub-0001", 0),
            ("sub-0001", 1),
            ("sub-0002", 0),
            ("sub-0002", 1),
        ]
        y_arr = np.array([0.0, 1.0, 0.5, 1.5], dtype=float)
        groups_arr = np.array([r[0] for r in trial_records], dtype=object)
        subj_to_epochs = {"sub-0001": _FakeEpochs(), "sub-0002": _FakeEpochs()}

        def _fake_prepare(_tuples):
            return trial_records, y_arr, groups_arr, subj_to_epochs, None

        def _fake_extract_epoch_data_block(indices, *_args, **_kwargs):
            return np.ones((len(indices), 1, 2), dtype=float)

        def _fake_extract_window_features(_data, records, *_args, **_kwargs):
            vals = np.linspace(0.0, 1.0, max(len(records), 1), dtype=float)
            return vals.reshape(len(records), 1, 1)

        cfg = DotConfig(
            {
                "analysis": {"min_subjects_for_group": 2},
                "machine_learning": {
                    "cv": {"permutation_scheme": "within_subject_within_block"},
                    "analysis": {
                        "time_generalization": {
                            "active_window": [0.0, 0.1],
                            "window_len": 0.1,
                            "step": 0.1,
                            "min_samples_per_window": 1,
                            "min_samples_for_corr": 1,
                            "min_valid_fold_fraction": 0.8,
                            "min_valid_permutation_fraction": 0.0,
                            "min_subjects_per_cell": 1,
                            "min_count_per_cell": 1,
                            "cluster_threshold": 0.05,
                            "use_ridgecv": False,
                            "alpha_grid": [0.01, 0.1, 1.0],
                            "default_alpha": 1.0,
                        }
                    },
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
            with self.assertRaisesRegex(ValueError, "requires block labels"):
                tg.time_generalization_regression(
                    deriv_root=Path("."),
                    subjects=["0001", "0002"],
                    task="task",
                    results_dir=None,
                    config_dict=cfg,
                    n_perm=1,
                    seed=42,
                )

    def test_classification_reports_subject_level_confidence_intervals(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult

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
        y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
            dtype=object,
        )
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": np.arange(len(groups), dtype=int),
                "block": [0, 1, 0, 1, 0, 1],
            }
        )
        result = ClassificationResult(
            y_true=y,
            y_pred=np.array([0, 1, 1, 1, 0, 0], dtype=int),
            y_prob=np.array([0.1, 0.9, 0.6, 0.8, 0.4, 0.3], dtype=float),
            groups=groups,
            failed_fold_count=0,
            n_folds_total=3,
        )
        cfg = DotConfig(
            {
                "machine_learning": {
                    "classification": {"min_subjects_with_auc_for_inference": 2},
                    "evaluation": {"bootstrap_iterations": 200},
                },
                "project": {"random_state": 7},
            }
        )

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
                    subjects=["0001", "0002", "0003"],
                    task="task",
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

            with open(out_dir / "metrics" / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

        subject_level = metrics.get("subject_level", {})
        self.assertIn("auc_ci_low", subject_level)
        self.assertIn("auc_ci_high", subject_level)
        self.assertIn("balanced_accuracy_ci_low", subject_level)
        self.assertIn("balanced_accuracy_ci_high", subject_level)

    def test_incremental_validity_missing_baseline_predictors_fail_fast_by_default(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)):
                with self.assertRaisesRegex(ValueError, "Missing baseline predictors"):
                    orch.run_incremental_validity_ml(
                        subjects=["0001", "0002"],
                        task="task",
                        deriv_root=Path(td),
                        config=DotConfig({}),
                        n_perm=0,
                        inner_splits=2,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        baseline_predictors=["temperature"],
                    )

    def test_incremental_validity_requires_explicit_baseline_predictors(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)):
                with self.assertRaisesRegex(ValueError, "requires explicit baseline predictors"):
                    orch.run_incremental_validity_ml(
                        subjects=["0001", "0002"],
                        task="task",
                        deriv_root=Path(td),
                        config=DotConfig({}),
                        n_perm=0,
                        inner_splits=2,
                        rng_seed=42,
                        results_root=Path(td),
                        logger=Mock(),
                        baseline_predictors=None,
                    )

    def test_incremental_validity_can_use_intercept_fallback_when_explicitly_enabled(self):
        from sklearn.dummy import DummyRegressor

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [0, 1, 2, 3],
            }
        )
        cfg = DotConfig({"machine_learning": {"incremental_validity": {"require_baseline_predictors": False}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)
            ), patch.object(
                orch, "create_elasticnet_pipeline", return_value=DummyRegressor(strategy="mean")
            ), patch.object(
                orch, "build_elasticnet_param_grid", return_value={}
            ):
                out_dir = orch.run_incremental_validity_ml(
                    subjects=["0001", "0002"],
                    task="task",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=0,
                    inner_splits=2,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    baseline_predictors=["temperature"],
                )

            with open(out_dir / "metrics" / "incremental_validity_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)

        self.assertEqual(summary["data"]["baseline_predictors"], ["intercept_only"])

    def test_model_comparison_reports_holm_corrected_pairwise_p_values(self):
        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        with tempfile.TemporaryDirectory() as td:
            X = np.array(
                [
                    [0.0, 1.0],
                    [0.1, 1.1],
                    [0.2, 1.2],
                    [0.3, 1.3],
                    [0.4, 1.4],
                    [0.5, 1.5],
                ],
                dtype=float,
            )
            y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
            groups = np.array(
                ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
                dtype=object,
            )
            meta = pd.DataFrame({"subject_id": groups, "trial_id": np.arange(len(groups), dtype=int)})

            with patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)):
                out_dir = orch.run_model_comparison_ml(
                    subjects=["0001", "0002", "0003"],
                    task="task",
                    deriv_root=Path(td),
                    config=DotConfig(
                        {
                            "machine_learning": {
                                "preprocessing": {"variance_threshold_grid": [0.0]}
                            }
                        }
                    ),
                    n_perm=8,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                )
            with open(out_dir / "metrics" / "model_comparison_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)

        pairwise = summary.get("pairwise_inference", {})
        self.assertTrue(pairwise)
        for rec in pairwise.values():
            self.assertIn("p_value_delta_r2_holm", rec)
            self.assertIn("p_value_delta_mae_holm", rec)

    def test_within_subject_classification_exports_probabilities_when_available(self):
        from sklearn.dummy import DummyClassifier

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.0, 1.0],
                [0.1, 1.1],
                [0.2, 1.2],
                [0.3, 1.3],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0001", "sub-0001"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [10, 11, 12, 13],
                "block": [0, 0, 1, 1],
            }
        )
        folds = [
            (
                1,
                np.array([0, 1], dtype=int),
                np.array([2, 3], dtype=int),
                "sub-0001",
                None,
            )
        ]
        cfg = DotConfig({"machine_learning": {"classification": {"min_subjects_with_auc_for_inference": 1}}})

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)
            ), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.create_logistic_pipeline",
                return_value=DummyClassifier(strategy="most_frequent"),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.build_logistic_param_grid",
                return_value={},
            ):
                out_dir = orch.run_within_subject_classification_ml(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    classification_model="lr",
                )

            pred_df = pd.read_csv(out_dir / "data" / "cv_predictions.tsv", sep="\t")

        self.assertIn("y_prob", pred_df.columns)
        self.assertTrue(np.all(np.isfinite(pred_df["y_prob"].to_numpy(dtype=float))))

    def test_within_subject_classification_applies_fold_feature_harmonization(self):
        from sklearn.dummy import DummyClassifier

        from eeg_pipeline.analysis.machine_learning import orchestration as orch

        X = np.array(
            [
                [0.0, 1.0],
                [0.1, 1.1],
                [0.2, 1.2],
                [0.3, 1.3],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0001", "sub-0001"], dtype=object)
        meta = pd.DataFrame(
            {
                "subject_id": groups,
                "trial_id": [10, 11, 12, 13],
                "block": [0, 0, 1, 1],
            }
        )
        folds = [
            (
                1,
                np.array([0, 1], dtype=int),
                np.array([2, 3], dtype=int),
                "sub-0001",
                None,
            )
        ]
        cfg = DotConfig({"machine_learning": {"classification": {"min_subjects_with_auc_for_inference": 1}}})

        calls = {"harmonize": 0}

        def _fake_harmonize(X_train, X_test, groups_train, harmonization_mode, n_covariates=0):
            _ = (groups_train, harmonization_mode, n_covariates)
            calls["harmonize"] += 1
            keep = np.array([False, True], dtype=bool)
            return X_train[:, keep], X_test[:, keep], keep

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                orch, "load_active_matrix", return_value=(X, y, groups, ["f1", "f2"], meta)
            ), patch.object(
                orch, "create_within_subject_folds", return_value=folds
            ), patch.object(
                orch, "apply_fold_feature_harmonization", side_effect=_fake_harmonize
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.create_logistic_pipeline",
                return_value=DummyClassifier(strategy="most_frequent"),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.build_logistic_param_grid",
                return_value={},
            ):
                orch.run_within_subject_classification_ml(
                    subjects=["0001"],
                    task="task",
                    deriv_root=Path(td),
                    config=cfg,
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=Path(td),
                    logger=Mock(),
                    classification_model="lr",
                    feature_harmonization="intersection",
                )

        self.assertEqual(calls["harmonize"], 1)

    def test_time_generalization_surfaces_inner_cv_failure(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        class FailingGridSearch:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, *args, **kwargs):
                raise RuntimeError("inner cv failure")

        config = DotConfig(
            {
                "machine_learning": {
                    "analysis": {
                        "time_generalization": {
                            "use_ridgecv": True,
                            "alpha_grid": [0.1, 1.0],
                            "default_alpha": 1.0,
                        }
                    },
                    "preprocessing": {
                        "power_transformer_method": "yeo-johnson",
                        "power_transformer_standardize": True,
                    },
                }
            }
        )
        X_train = np.arange(12, dtype=float).reshape(6, 2)
        y_train = np.linspace(0.0, 1.0, 6)
        groups_train = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)

        with patch.object(tg, "GridSearchCV", FailingGridSearch):
            with self.assertRaisesRegex(RuntimeError, "inner CV failed"):
                tg._fit_time_generalization_model(
                    X_train,
                    y_train,
                    groups_train,
                    config,
                    fold=1,
                    train_window=0,
                )

    def test_time_generalization_surfaces_cell_prediction_failure(self):
        from eeg_pipeline.analysis.machine_learning import time_generalization as tg

        class FailingModel:
            def predict(self, X):
                raise RuntimeError("prediction failure")

        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            tg._evaluate_time_generalization_cell(
                model=FailingModel(),
                test_features=np.ones((4, 2), dtype=float),
                col_mask=np.array([True, True], dtype=bool),
                y_test=np.linspace(0.0, 1.0, 4),
                min_samples_per_window=2,
                min_samples_for_corr=2,
                fold=1,
                train_window=0,
                test_window=0,
            )

    def test_grid_search_wrapper_rejects_nonfinite_test_scores(self):
        from eeg_pipeline.analysis.machine_learning.cv import grid_search_with_warning_logging

        class FakeGrid:
            cv_results_ = None

            def fit(self, X, y, **fit_params):
                self.cv_results_ = {
                    "mean_test_score": np.array([np.nan, 0.2], dtype=float),
                    "params": [{"alpha": 0.1}, {"alpha": 1.0}],
                }
                return self

        with self.assertRaisesRegex(RuntimeError, "non-finite GridSearchCV test scores"):
            grid_search_with_warning_logging(
                FakeGrid(),
                np.ones((4, 2), dtype=float),
                np.arange(4, dtype=float),
                fold_info="synthetic fold",
                log=Mock(),
            )

    def test_find_block_column_parses_run_prefixed_labels(self):
        from eeg_pipeline.utils.data.machine_learning import _find_block_column

        events = pd.DataFrame({"run_id": ["run-01", "run-01", "run-02", "run-02"]})
        block = _find_block_column(events)
        self.assertIsNotNone(block)
        vals = pd.to_numeric(block, errors="coerce").to_numpy(dtype=float)
        np.testing.assert_allclose(vals, np.array([1.0, 1.0, 2.0, 2.0], dtype=float), atol=1e-12)
