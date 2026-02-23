import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


class TestMachineLearningCompletion(unittest.TestCase):
    def test_ml_init_extract_params_process_subject(self):
        from eeg_pipeline.pipelines.machine_learning import MLPipeline

        cfg = DotConfig({"project": {"random_state": 99}})
        with patch("eeg_pipeline.pipelines.machine_learning.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "config", config or cfg), setattr(self, "deriv_root", Path("/tmp/deriv")), setattr(self, "logger", Mock()))):
            p = MLPipeline(config=cfg)
        self.assertEqual(p.results_root, Path("/tmp/deriv") / "machine_learning")

        fake_cli = types.SimpleNamespace(ProgressReporter=lambda enabled=False: _NoopProgress())
        with patch.dict(sys.modules, {"eeg_pipeline.cli.common": fake_cli}):
            params = p._extract_ml_parameters({})
        self.assertEqual(params["rng_seed"], 99)

        with self.assertRaises(NotImplementedError):
            p.process_subject("0001")

    def test_ml_run_internal_stages(self):
        from eeg_pipeline.pipelines.machine_learning import MLPipeline

        p = object.__new__(MLPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.results_root = p.deriv_root / "ml"
        p.config = DotConfig({})
        p.logger = Mock()

        fake_data = types.SimpleNamespace(load_active_matrix=lambda *a, **k: (np.array([[1.0]]), np.array([1.0]), np.array([1]), ["f1"], None))
        fake_orch = types.SimpleNamespace(
            _run_uncertainty_stage=lambda **k: Path("/tmp/u"),
            _run_shap_importance_stage=lambda **k: Path("/tmp/s"),
            _run_permutation_importance_stage=lambda **k: Path("/tmp/p"),
        )

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.utils.data.machine_learning": fake_data,
                "eeg_pipeline.analysis.machine_learning.orchestration": fake_orch,
            },
        ):
            self.assertEqual(p._run_uncertainty(["0001"], "t", 42), Path("/tmp/u"))
            self.assertEqual(p._run_shap(["0001"], "t", 42), Path("/tmp/s"))
            self.assertEqual(p._run_permutation_importance(["0001"], "t", 42), Path("/tmp/p"))

class TestMachineLearningDeep(unittest.TestCase):
        def test_ml_pipeline_mode_executors(self):
            from eeg_pipeline.pipelines.machine_learning import MLPipeline

            p = object.__new__(MLPipeline)
            p.deriv_root = Path(tempfile.mkdtemp())
            p.results_root = p.deriv_root / "machine_learning"
            p.config = DotConfig({})
            p.logger = Mock()
            progress = SimpleNamespace(step=lambda *a, **k: None)
            params = {
                "n_perm": 0,
                "inner_splits": 3,
                "outer_jobs": 1,
                "rng_seed": 42,
                "model": "elasticnet",
                "classification_model": None,
                "target": None,
                "binary_threshold": None,
                "feature_families": None,
                "feature_bands": None,
                "feature_segments": None,
                "feature_scopes": None,
                "feature_stats": None,
                "feature_harmonization": None,
                "covariates": None,
                "perm_n_repeats": 5,
                "uncertainty_alpha": 0.1,
            }

            with patch("eeg_pipeline.pipelines.machine_learning.run_regression_ml", return_value=Path("/tmp/r")), patch(
                "eeg_pipeline.pipelines.machine_learning.run_within_subject_regression_ml", return_value=Path("/tmp/wsr")
            ), patch(
                "eeg_pipeline.pipelines.machine_learning.run_time_generalization", return_value=Path("/tmp/tg")
            ), patch(
                "eeg_pipeline.pipelines.machine_learning.run_classification_ml", return_value=Path("/tmp/c")
            ), patch(
                "eeg_pipeline.pipelines.machine_learning.run_within_subject_classification_ml", return_value=Path("/tmp/wsc")
            ), patch(
                "eeg_pipeline.pipelines.machine_learning.run_model_comparison_ml", return_value=Path("/tmp/m")
            ), patch(
                "eeg_pipeline.pipelines.machine_learning.run_incremental_validity_ml", return_value=Path("/tmp/i")
            ), patch.object(
                MLPipeline, "_run_uncertainty", return_value=Path("/tmp/u")
            ), patch.object(
                MLPipeline, "_run_shap", return_value=Path("/tmp/s")
            ), patch.object(
                MLPipeline, "_run_permutation_importance", return_value=Path("/tmp/p")
            ):
                self.assertEqual(p._execute_regression(["0001", "0002"], "t", "group", params, progress), Path("/tmp/r"))
                self.assertEqual(p._execute_regression(["0001"], "t", "subject", params, progress), Path("/tmp/wsr"))
                self.assertEqual(p._execute_timegen(["0001", "0002"], "t", "group", params, progress), Path("/tmp/tg"))
                with self.assertRaises(ValueError):
                    p._execute_timegen(["0001"], "t", "subject", params, progress)
                self.assertEqual(p._execute_classify(["0001", "0002"], "t", "group", params, progress), Path("/tmp/c"))
                self.assertEqual(p._execute_classify(["0001"], "t", "subject", params, progress), Path("/tmp/wsc"))
                self.assertEqual(p._execute_model_comparison(["0001"], "t", "group", params, progress), Path("/tmp/m"))
                self.assertEqual(p._execute_incremental_validity(["0001"], "t", "group", params, progress), Path("/tmp/i"))
                self.assertEqual(p._execute_uncertainty(["0001"], "t", "group", params, progress), Path("/tmp/u"))
                self.assertEqual(p._execute_shap(["0001"], "t", "group", params, progress), Path("/tmp/s"))
                self.assertEqual(p._execute_permutation(["0001"], "t", "group", params, progress), Path("/tmp/p"))

class TestMachineLearningGapfill(unittest.TestCase):
        def test_ml_missing_task_and_visualize(self):
            from eeg_pipeline.pipelines.machine_learning import MLPipeline

            p = object.__new__(MLPipeline)
            p.config = DotConfig({})
            with self.assertRaises(ValueError):
                p._validate_inputs(["0001", "0002"], None, "group")
            with self.assertRaises(NotImplementedError):
                p.visualize(Path(tempfile.mkdtemp()))

        def test_run_batch_raises_when_mode_returns_none(self):
            from eeg_pipeline.pipelines.machine_learning import MLPipeline

            p = object.__new__(MLPipeline)
            p.config = DotConfig({})
            p.logger = Mock()

            progress = Mock()
            params = {
                "progress": progress,
                "cv_scope": "group",
                "model": "elasticnet",
                "n_perm": 0,
                "inner_splits": 3,
            }

            with patch.object(MLPipeline, "_extract_ml_parameters", return_value=params), patch.object(
                MLPipeline, "_validate_inputs", return_value="task"
            ), patch.object(
                MLPipeline, "_get_mode_dispatcher", return_value={"regression": (lambda **kwargs: None)}
            ):
                with self.assertRaisesRegex(RuntimeError, "produced no output"):
                    p.run_batch(["0001", "0002"], task="task", mode="regression")
            progress.complete.assert_called_once_with(success=False)

        def test_run_batch_writes_reproducibility_metadata(self):
            from eeg_pipeline.pipelines.machine_learning import MLPipeline

            p = object.__new__(MLPipeline)
            p.name = "machine_learning"
            p.config = DotConfig({})
            p.logger = Mock()
            p.deriv_root = Path(tempfile.mkdtemp())
            p.results_root = p.deriv_root / "machine_learning"

            progress = _NoopProgress()
            params = {
                "progress": progress,
                "cv_scope": "group",
                "model": "elasticnet",
                "n_perm": 0,
                "inner_splits": 3,
            }
            out_dir = Path(tempfile.mkdtemp())

            with patch.object(MLPipeline, "_extract_ml_parameters", return_value=params), patch.object(
                MLPipeline, "_validate_inputs", return_value="task"
            ), patch.object(
                MLPipeline,
                "_get_mode_dispatcher",
                return_value={"regression": (lambda **kwargs: out_dir)},
            ):
                out = p.run_batch(["0001"], task="task", mode="regression")

            self.assertEqual(out[0]["status"], "success")
            metadata_dir = p.deriv_root / "logs" / "run_metadata" / "machine_learning"
            metadata_files = sorted(metadata_dir.glob("run_*.json"))
            self.assertTrue(metadata_files)
            payload = json.loads(metadata_files[-1].read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["specifications"]["mode"], "regression")
