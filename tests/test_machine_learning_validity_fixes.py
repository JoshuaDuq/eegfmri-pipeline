import json
import logging
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestMachineLearningValidityFixes(unittest.TestCase):
    def test_classification_permutation_pvalue_uses_subject_level_auc(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult
        from eeg_pipeline.analysis.machine_learning.orchestration import run_classification_ml

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"])
        feature_names = ["f0"]
        meta = pd.DataFrame({"block": [1, 2, 1, 2]})

        fake_result = ClassificationResult(
            y_true=y,
            y_pred=y.copy(),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8], dtype=float),
            groups=groups,
        )
        # Deliberately make pooled AUC differ from subject-level mean AUC.
        fake_result.auc = 0.90
        fake_result.per_subject_metrics = {
            "sub-0001": {"auc": 0.60, "n_trials": 2},
            "sub-0002": {"auc": 0.60, "n_trials": 2},
        }

        config = {
            "machine_learning": {
                "targets": {"classification": "pain_binary"},
                "classification": {"model": "svm"},
            },
            "project": {"random_state": 42},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            with patch(
                "eeg_pipeline.analysis.machine_learning.orchestration.load_active_matrix",
                return_value=(X, y, groups, feature_names, meta),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
                return_value=(fake_result, pd.DataFrame()),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.orchestration._run_classification_permutations",
                return_value=np.array([0.65, 0.70], dtype=float),
            ):
                out_dir = run_classification_ml(
                    subjects=["0001", "0002"],
                    task="pain",
                    deriv_root=results_root,
                    config=config,
                    n_perm=2,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=results_root,
                    logger=logging.getLogger(__name__),
                    classification_model="svm",
                )

            with open(out_dir / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

        # Expected from subject-level AUC=0.60 against null [0.65, 0.70]:
        # p = ((2 >= 0.60) + 1) / (2 + 1) = 1.0
        self.assertAlmostEqual(float(metrics["p_value"]), 1.0, places=7)

    def test_load_active_matrix_fails_when_exclusion_fraction_is_too_high(self):
        from eeg_pipeline.utils.data import machine_learning as ml_data
        from eeg_pipeline.utils.config.loader import load_config

        def _loader(subject, *args, **kwargs):
            if str(subject) == "003":
                X_df = pd.DataFrame({"power_a": [1.0, 2.0]})
                y = np.array([3.0, 4.0], dtype=float)
                meta = pd.DataFrame({"subject_id": ["sub-003", "sub-003"], "block": [1, 2]})
                return X_df, y, "rating", meta
            raise FileNotFoundError(f"missing subject {subject}")

        config = load_config()
        config["feature_engineering.analysis_mode"] = "trial_ml_safe"
        config["machine_learning.data.feature_set"] = "combined"
        config["machine_learning.data.feature_harmonization"] = "intersection"
        config["machine_learning.data.max_excluded_subject_fraction"] = 0.20

        with patch.object(ml_data, "_load_subject_ml_from_features", side_effect=_loader):
            with self.assertRaises(RuntimeError):
                ml_data.load_active_matrix(
                    subjects=["001", "002", "003"],
                    task="pain",
                    deriv_root=Path("."),
                    config=config,
                    log=logging.getLogger(__name__),
                    feature_families=["power"],
                    target="rating",
                    target_kind="continuous",
                )

    def test_default_permutation_scheme_is_block_aware(self):
        from eeg_pipeline.utils.config.loader import get_config_value, load_config

        cfg = load_config()
        scheme = str(
            get_config_value(cfg, "machine_learning.cv.permutation_scheme", "within_subject")
        ).strip()
        self.assertEqual(scheme, "within_subject_within_block")

    def test_fold_specific_intersection_harmonization_is_train_only(self):
        from eeg_pipeline.analysis.machine_learning.cv import apply_fold_feature_harmonization

        # Feature 1 is missing in all rows for group B and should be removed by train-only intersection.
        X_train = np.array(
            [
                [1.0, 10.0],
                [2.0, 11.0],   # group A
                [3.0, np.nan],
                [4.0, np.nan], # group B
            ],
            dtype=float,
        )
        X_test = np.array([[5.0, 12.0]], dtype=float)
        groups_train = np.array(["A", "A", "B", "B"], dtype=object)

        Xtr_h, Xte_h, keep = apply_fold_feature_harmonization(
            X_train, X_test, groups_train, "intersection"
        )

        self.assertTrue(np.array_equal(keep, np.array([True, False])))
        self.assertEqual(Xtr_h.shape[1], 1)
        self.assertEqual(Xte_h.shape[1], 1)

    def test_classification_permutation_guard_raises_on_low_completion(self):
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_classification_permutations

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([1, 2, 1, 2], dtype=float)

        config = {
            "machine_learning": {
                "cv": {
                    "permutation_scheme": "within_subject_within_block",
                    "min_valid_permutation_fraction": 0.5,
                }
            }
        }

        with patch(
            "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
            side_effect=RuntimeError("perm failed"),
        ):
            with self.assertRaises(RuntimeError):
                _run_classification_permutations(
                    X=X,
                    y=y,
                    groups=groups,
                    blocks=blocks,
                    model="svm",
                    inner_splits=2,
                    seed=42,
                    n_perm=4,
                    config=config,
                    logger=logging.getLogger(__name__),
                    harmonization_mode="intersection",
                )

    def test_nested_loso_classification_uses_nan_probabilities_on_failed_folds(self):
        from eeg_pipeline.analysis.machine_learning.classification import nested_loso_classification

        X = np.array(
            [
                [0.0], [1.0], [2.0], [3.0],
                [4.0], [5.0], [6.0], [7.0],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        groups = np.array(
            ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003", "sub-0004", "sub-0004"],
            dtype=object,
        )

        with patch("sklearn.model_selection.GridSearchCV.fit", side_effect=RuntimeError("grid failed")):
            result, _ = nested_loso_classification(
                X=X,
                y=y,
                groups=groups,
                model="svm",
                inner_splits=2,
                seed=42,
                config=None,
                logger=logging.getLogger(__name__),
                harmonization_mode="intersection",
            )

        # All folds failed before predict_proba; y_prob should be treated as unavailable, not zeros.
        self.assertIsNone(result.y_prob)

    def test_time_generalization_block_permutation_respects_subject_blocks(self):
        from eeg_pipeline.analysis.machine_learning.time_generalization import (
            _permute_labels_within_subject_structure,
        )

        y = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0], dtype=float)
        groups = np.array(["sub-1", "sub-1", "sub-1", "sub-1", "sub-2", "sub-2"], dtype=object)
        blocks = np.array(["A", "A", "B", "B", "A", "B"], dtype=object)
        rng = np.random.default_rng(123)

        y_perm = _permute_labels_within_subject_structure(
            y,
            groups,
            blocks,
            rng=rng,
            scheme="within_subject_within_block",
        )

        mask_a = (groups == "sub-1") & (blocks == "A")
        mask_b = (groups == "sub-1") & (blocks == "B")
        self.assertCountEqual(y_perm[mask_a].tolist(), y[mask_a].tolist())
        self.assertCountEqual(y_perm[mask_b].tolist(), y[mask_b].tolist())

    def test_time_generalization_uses_equal_subject_weight_for_r2(self):
        from eeg_pipeline.analysis.machine_learning.time_generalization import (
            _aggregate_time_generalization_matrices,
        )

        stacked_r = np.array(
            [
                [[0.10]],
                [[0.20]],
            ],
            dtype=float,
        )
        stacked_r2 = np.array(
            [
                [[0.90]],
                [[0.10]],
            ],
            dtype=float,
        )
        stacked_counts = np.array(
            [
                [[100]],
                [[1]],
            ],
            dtype=int,
        )
        config = {
            "machine_learning": {
                "analysis": {
                    "time_generalization": {
                        "min_subjects_per_cell": 2,
                        "min_count_per_cell": 0,
                    }
                }
            }
        }

        tg_r, tg_r2, coverage_map, subject_coverage_map, tested_mask = _aggregate_time_generalization_matrices(
            stacked_r,
            stacked_r2,
            stacked_counts,
            config,
        )

        expected_r = float(np.tanh(np.mean(np.arctanh(np.array([0.10, 0.20])))))
        self.assertAlmostEqual(float(tg_r[0, 0]), expected_r, places=7)
        self.assertAlmostEqual(float(tg_r2[0, 0]), 0.50, places=7)
        self.assertEqual(int(coverage_map[0, 0]), 101)
        self.assertEqual(int(subject_coverage_map[0, 0]), 2)
        self.assertTrue(bool(tested_mask[0, 0]))

    def test_time_generalization_significance_uses_tested_mask_only(self):
        from eeg_pipeline.analysis.machine_learning.time_generalization import (
            _compute_time_generalization_significance,
        )

        tg_r = np.array(
            [
                [0.30, np.nan],
                [np.nan, 0.25],
            ],
            dtype=float,
        )
        null_r = np.array(
            [
                [[0.05, 0.02], [0.01, 0.03]],
                [[0.04, 0.03], [0.02, 0.01]],
                [[0.03, 0.01], [0.01, 0.02]],
                [[0.06, 0.01], [0.03, 0.04]],
            ],
            dtype=float,
        )
        config = {"machine_learning": {"analysis": {"time_generalization": {"cluster_threshold": 0.05}}}}

        p_matrix, sig_fdr, sig_maxstat, sig_cluster, tested_mask = _compute_time_generalization_significance(
            tg_r=tg_r,
            null_r=null_r,
            config=config,
        )

        self.assertTrue(bool(tested_mask[0, 0]))
        self.assertFalse(bool(tested_mask[0, 1]))
        self.assertTrue(np.isnan(p_matrix[0, 1]))
        self.assertFalse(bool(sig_fdr[0, 1]))
        self.assertFalse(bool(sig_maxstat[0, 1]))
        self.assertFalse(bool(sig_cluster[0, 1]))

    def test_classification_run_raises_when_failed_fold_fraction_too_high(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult
        from eeg_pipeline.analysis.machine_learning.orchestration import run_classification_ml

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        meta = pd.DataFrame({"block": [1, 2, 1, 2]})
        feature_names = ["f0"]

        fake_result = ClassificationResult(
            y_true=y,
            y_pred=y.copy(),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8], dtype=float),
            groups=groups,
            failed_fold_count=2,
            n_folds_total=4,
        )

        config = {
            "machine_learning": {
                "targets": {"classification": "pain_binary"},
                "classification": {
                    "model": "svm",
                    "max_failed_fold_fraction": 0.25,
                },
            },
            "project": {"random_state": 42},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            with patch(
                "eeg_pipeline.analysis.machine_learning.orchestration.load_active_matrix",
                return_value=(X, y, groups, feature_names, meta),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.classification.nested_loso_classification",
                return_value=(fake_result, pd.DataFrame()),
            ):
                with self.assertRaises(RuntimeError):
                    run_classification_ml(
                        subjects=["0001", "0002"],
                        task="pain",
                        deriv_root=results_root,
                        config=config,
                        n_perm=0,
                        inner_splits=2,
                        outer_jobs=1,
                        rng_seed=42,
                        results_root=results_root,
                        logger=logging.getLogger(__name__),
                        classification_model="svm",
                    )

    def test_ml_cli_accepts_cnn_classification_model(self):
        from eeg_pipeline.cli.commands.machine_learning import setup_ml

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_ml(subparsers)

        args = parser.parse_args(
            [
                "ml",
                "classify",
                "--subject",
                "0001",
                "--subject",
                "0002",
                "--classification-model",
                "cnn",
            ]
        )
        self.assertEqual(args.classification_model, "cnn")

    def test_run_classification_ml_uses_epoch_tensor_loader_for_cnn(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult
        from eeg_pipeline.analysis.machine_learning.orchestration import run_classification_ml

        X_epoch = np.random.RandomState(0).randn(4, 3, 16).astype(float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        feature_names = ["C3", "Cz", "C4"]
        meta = pd.DataFrame({"block": [1, 2, 1, 2]})

        fake_result = ClassificationResult(
            y_true=y,
            y_pred=y.copy(),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8], dtype=float),
            groups=groups,
        )
        fake_result.per_subject_metrics = {
            "sub-0001": {"auc": 0.75, "balanced_accuracy": 0.5, "accuracy": 1.0, "n_trials": 2},
            "sub-0002": {"auc": 0.75, "balanced_accuracy": 0.5, "accuracy": 1.0, "n_trials": 2},
        }

        config = {
            "machine_learning": {
                "targets": {"classification": "pain_binary"},
                "classification": {"model": "cnn"},
            },
            "project": {"random_state": 42},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            with patch(
                "eeg_pipeline.analysis.machine_learning.orchestration.load_epoch_tensor_matrix",
                return_value=(X_epoch, y, groups, feature_names, meta),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.orchestration.load_active_matrix",
                side_effect=AssertionError("CNN path should not use tabular feature loader"),
            ), patch(
                "eeg_pipeline.analysis.machine_learning.cnn.nested_loso_cnn_classification",
                return_value=(fake_result, pd.DataFrame()),
            ):
                out_dir = run_classification_ml(
                    subjects=["0001", "0002"],
                    task="pain",
                    deriv_root=results_root,
                    config=config,
                    n_perm=0,
                    inner_splits=2,
                    outer_jobs=1,
                    rng_seed=42,
                    results_root=results_root,
                    logger=logging.getLogger(__name__),
                    classification_model="cnn",
                )

            with open(out_dir / "pooled_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

        self.assertEqual(metrics["model"], "cnn")
        self.assertEqual(int(metrics.get("n_channels", 0)), 3)
        self.assertEqual(int(metrics.get("n_timepoints", 0)), 16)

    def test_classification_permutations_support_cnn_model(self):
        from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_classification_permutations

        X_epoch = np.random.RandomState(1).randn(4, 2, 8).astype(float)
        y = np.array([0, 1, 0, 1], dtype=int)
        groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
        blocks = np.array([1, 2, 1, 2], dtype=float)
        config = {
            "machine_learning": {
                "cv": {
                    "permutation_scheme": "within_subject_within_block",
                    "min_valid_permutation_fraction": 0.5,
                }
            }
        }

        fake_result = ClassificationResult(
            y_true=y,
            y_pred=y.copy(),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8], dtype=float),
            groups=groups,
        )
        fake_result.per_subject_metrics = {
            "sub-0001": {"auc": 0.6},
            "sub-0002": {"auc": 0.7},
        }

        with patch(
            "eeg_pipeline.analysis.machine_learning.cnn.nested_loso_cnn_classification",
            return_value=(fake_result, pd.DataFrame()),
        ):
            null_aucs = _run_classification_permutations(
                X=X_epoch,
                y=y,
                groups=groups,
                blocks=blocks,
                model="cnn",
                inner_splits=2,
                seed=42,
                n_perm=3,
                config=config,
                logger=logging.getLogger(__name__),
                harmonization_mode="intersection",
            )

        self.assertIsNotNone(null_aucs)
        self.assertEqual(len(null_aucs), 3)
