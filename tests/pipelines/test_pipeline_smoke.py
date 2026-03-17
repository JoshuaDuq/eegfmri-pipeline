import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


class TestAllPipelines(unittest.TestCase):
    def test_pipeline_base_validate_batch_inputs_errors(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class DummyPipeline(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
                self.deriv_root.mkdir(parents=True, exist_ok=True)

            def process_subject(self, subject, task, **kwargs):
                return None

        pipeline = DummyPipeline()

        with self.assertRaises(ValueError):
            pipeline._validate_batch_inputs([], task=None)
        with self.assertRaises(ValueError):
            pipeline._validate_batch_inputs(["0001"], task=None)

    def test_pipeline_base_write_ledger_sanitizes_newlines(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class DummyPipeline(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "task"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
                self.deriv_root.mkdir(parents=True, exist_ok=True)

            def process_subject(self, subject, task, **kwargs):
                return None

        pipeline = DummyPipeline()
        ledger_path = Path(tempfile.mkdtemp()) / "ledger.tsv"
        pipeline._write_ledger(
            [
                {
                    "subject": "0001",
                    "status": "failed",
                    "duration_s": 1.23,
                    "error": "line1\nline2",
                    "traceback_path": "/tmp/tb.log",
                }
            ],
            ledger_path,
        )
        text = ledger_path.read_text(encoding="utf-8")
        self.assertIn("line1 | line2", text)

    def test_pipeline_base_handle_batch_failures_all_failed_raises(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class DummyPipeline(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "task"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
                self.deriv_root.mkdir(parents=True, exist_ok=True)

            def process_subject(self, subject, task, **kwargs):
                return None

        pipeline = DummyPipeline()
        progress = Mock()
        with self.assertRaises(RuntimeError):
            pipeline._handle_batch_failures(
                ledger=[{"subject": "0001", "status": "failed"}],
                subjects=["0001"],
                ledger_path=Path("/tmp/ledger.tsv"),
                progress=progress,
            )
        progress.complete.assert_called_once_with(success=False)

    def test_pipeline_base_run_batch_success_and_group_level(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class DummyPipeline(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "task"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
                self.deriv_root.mkdir(parents=True, exist_ok=True)
                self.group_called = False

            def process_subject(self, subject, task, **kwargs):
                return None

            def run_group_level(self, subjects, task, **kwargs):
                self.group_called = True

        pipeline = DummyPipeline()

        with patch("eeg_pipeline.pipelines.base.BatchProgress", _NoopBatchProgress):
            ledger = pipeline.run_batch(["0001", "0002"], task="task")

        self.assertEqual(len(ledger), 2)
        self.assertTrue(all(item["status"] == "success" for item in ledger))
        self.assertTrue(pipeline.group_called)

    def test_preprocessing_pipeline_run_batch_dispatches_steps(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        pipeline = object.__new__(PreprocessingPipeline)
        pipeline.name = "preprocessing"
        pipeline.config = DotConfig({"project": {"task": "task"}})
        pipeline.logger = Mock()
        pipeline.deriv_root = Path(tempfile.mkdtemp())
        progress = _DummyProgress()

        with patch.object(
            PreprocessingPipeline,
            "_extract_preprocessing_params",
            return_value=("task", "ica", True, True, False, 2, progress),
        ), patch.object(
            PreprocessingPipeline,
            "_get_steps_for_mode",
            return_value=["ica-fit", "ica-label"],
        ), patch.object(PreprocessingPipeline, "_execute_steps") as mock_exec:
            result = pipeline.run_batch(["0001", "0002"])

        self.assertEqual(result[0]["status"], "success")
        mock_exec.assert_called_once()

    def test_preprocessing_pipeline_get_steps_for_mode_and_invalid(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        pipeline = object.__new__(PreprocessingPipeline)
        self.assertEqual(pipeline._get_steps_for_mode("bad-channels"), ["bad-channels"])
        self.assertEqual(pipeline._get_steps_for_mode("ica"), ["ica-fit", "ica-label"])
        with self.assertRaises(ValueError):
            pipeline._get_steps_for_mode("invalid")

    def test_preprocessing_pipeline_normalize_subjects(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        pipeline = object.__new__(PreprocessingPipeline)
        self.assertEqual(pipeline._normalize_subjects(["all"]), "all")
        self.assertEqual(pipeline._normalize_subjects(["0001"]), ["0001"])

    def test_feature_pipeline_requires_task(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        pipeline = object.__new__(FeaturePipeline)
        pipeline.config = DotConfig({})
        pipeline.logger = Mock()

        with self.assertRaises(ValueError):
            pipeline.process_subject("0001", task=None)

    def test_feature_pipeline_helpers(self):
        from eeg_pipeline.pipelines.features import _resolve_time_ranges, _calculate_total_steps

        self.assertEqual(
            _resolve_time_ranges([{"name": "a", "tmin": 0.0, "tmax": 1.0}], None, None),
            [{"name": "a", "tmin": 0.0, "tmax": 1.0}],
        )
        self.assertEqual(
            _resolve_time_ranges(None, 0.1, 0.5),
            [{"name": None, "tmin": 0.1, "tmax": 0.5}],
        )
        self.assertEqual(_calculate_total_steps(1), 4)
        self.assertEqual(_calculate_total_steps(3), 10)

    def test_behavior_pipeline_group_level_noop_when_disabled(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipeline

        pipeline = object.__new__(BehaviorPipeline)
        pipeline.pipeline_config = SimpleNamespace(
            run_multilevel_correlations=False,
        )
        pipeline.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)
        pipeline.config = DotConfig({})
        pipeline.logger = Mock()
        pipeline.feature_files = []
        pipeline.feature_categories = []

        result = pipeline.run_group_level(["0001", "0002"])
        self.assertIsNone(result)

    def test_behavior_flag_resolution_and_optional_int(self):
        from eeg_pipeline.pipelines.behavior import _resolve_behavior_computation_flags, _get_optional_int

        flags = _resolve_behavior_computation_flags(["validation", "icc", "unknown"], logger=Mock())
        self.assertTrue(flags["icc"])
        self.assertFalse(flags["regression"])

        cfg = DotConfig({"a": {"b": "7"}, "x": {"y": None}})
        self.assertEqual(_get_optional_int(cfg, "a.b", None), 7)
        self.assertIsNone(_get_optional_int(cfg, "x.y", 9))

    def test_behavior_pipeline_group_level_calls_orchestration_when_enabled(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipeline

        pipeline = object.__new__(BehaviorPipeline)
        pipeline.pipeline_config = SimpleNamespace(
            run_multilevel_correlations=True,
        )
        pipeline.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)
        pipeline.config = DotConfig({})
        pipeline.logger = Mock()
        pipeline.feature_files = []
        pipeline.feature_categories = []

        fake_result = SimpleNamespace(multilevel_correlations=None)
        with patch(
            "eeg_pipeline.analysis.behavior.orchestration.run_group_level_analysis",
            return_value=fake_result,
        ), patch(
            "eeg_pipeline.analysis.behavior.trial_table_helpers.find_trial_table_path",
            return_value=Path("/tmp/trials.parquet"),
        ):
            out = pipeline.run_group_level(["0001", "0002"])
        self.assertIs(out, fake_result)

    def test_ml_pipeline_rejects_unknown_mode(self):
        from eeg_pipeline.pipelines.machine_learning import MLPipeline

        pipeline = object.__new__(MLPipeline)
        pipeline.name = "machine_learning"
        pipeline.config = DotConfig({"project": {"task": "task"}})
        pipeline.logger = Mock()
        pipeline.deriv_root = Path(tempfile.mkdtemp())

        params = {
            "cv_scope": "group",
            "progress": _DummyProgress(),
            "n_perm": 0,
            "inner_splits": 3,
            "outer_jobs": 1,
            "rng_seed": 42,
            "model": "elasticnet",
            "uncertainty_alpha": 0.1,
            "perm_n_repeats": 10,
            "classification_model": None,
            "feature_families": None,
            "feature_bands": None,
            "feature_segments": None,
            "feature_scopes": None,
            "feature_stats": None,
            "feature_harmonization": None,
            "target": None,
            "binary_threshold": None,
            "baseline_predictors": None,
            "covariates": None,
        }

        with patch.object(MLPipeline, "_extract_ml_parameters", return_value=params):
            with self.assertRaises(ValueError):
                pipeline.run_batch(["0001", "0002"], mode="unknown")

    def test_ml_pipeline_validate_inputs(self):
        from eeg_pipeline.pipelines.machine_learning import MLPipeline

        pipeline = object.__new__(MLPipeline)
        pipeline.config = DotConfig(
            {"project": {"task": "task"}, "analysis": {"min_subjects_for_group": 2}}
        )
        with self.assertRaises(ValueError):
            pipeline._validate_inputs([], None, "group")
        with self.assertRaises(ValueError):
            pipeline._validate_inputs(["0001"], "task", "bad")
        with self.assertRaises(ValueError):
            pipeline._validate_inputs(["0001"], "task", "group")
        resolved = pipeline._validate_inputs(["0001"], "task", "subject")
        self.assertEqual(resolved, "task")

    def test_ml_pipeline_run_batch_dispatches_mode(self):
        from eeg_pipeline.pipelines.machine_learning import MLPipeline

        pipeline = object.__new__(MLPipeline)
        pipeline.name = "machine_learning"
        pipeline.config = DotConfig({"project": {"task": "task"}})
        pipeline.logger = Mock()
        pipeline.deriv_root = Path(tempfile.mkdtemp())

        progress = _DummyProgress()
        params = {
            "cv_scope": "group",
            "progress": progress,
            "n_perm": 0,
            "inner_splits": 3,
            "outer_jobs": 1,
            "rng_seed": 42,
            "model": "elasticnet",
            "uncertainty_alpha": 0.1,
            "perm_n_repeats": 10,
            "classification_model": None,
            "feature_families": None,
            "feature_bands": None,
            "feature_segments": None,
            "feature_scopes": None,
            "feature_stats": None,
            "feature_harmonization": None,
            "target": None,
            "binary_threshold": None,
            "baseline_predictors": None,
            "covariates": None,
        }

        fake_executor = Mock(return_value=Path("/tmp/results"))
        with patch.object(MLPipeline, "_extract_ml_parameters", return_value=params), patch.object(
            MLPipeline, "_validate_inputs", return_value="task"
        ), patch.object(MLPipeline, "_get_mode_dispatcher", return_value={"regression": fake_executor}):
            out = pipeline.run_batch(["0001", "0002"], mode="regression")
        self.assertEqual(out[0]["status"], "success")
        self.assertIn("/tmp/results", out[0]["results_dir"])
        fake_executor.assert_called_once()

    def test_fmri_preprocessing_pipeline_dry_run(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids_fmri"
        bids_root.mkdir(parents=True, exist_ok=True)
        (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)
        fs_license = tmp / "license.txt"
        fs_license.write_text("dummy", encoding="utf-8")

        pipeline = object.__new__(FmriPreprocessingPipeline)
        pipeline.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids_root)},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {"fs_license_file": str(fs_license)},
                },
            }
        )
        pipeline.logger = Mock()
        pipeline.deriv_root = tmp / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)

        with patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"):
            pipeline.process_subject("0001", task="", progress=_DummyProgress(), dry_run=True)

    def test_fmri_preprocessing_helpers(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import _resolve_path, _require_executable

        self.assertIsNone(_resolve_path(None))
        self.assertIsNone(_resolve_path("   "))
        self.assertTrue(isinstance(_resolve_path("."), Path))

        with patch("fmri_pipeline.pipelines.fmri_preprocessing.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                _require_executable("docker")

    def test_fmri_analysis_pipeline_dry_run(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids_fmri"
        bids_root.mkdir(parents=True, exist_ok=True)

        pipeline = object.__new__(FmriAnalysisPipeline)
        pipeline.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        pipeline.logger = Mock()
        pipeline.deriv_root = tmp / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)

        fake_nib = types.SimpleNamespace(save=lambda *_args, **_kwargs: None, load=lambda *_args, **_kwargs: None)
        with patch.dict(sys.modules, {"nibabel": fake_nib}):
            pipeline.process_subject(
                "0001",
                task="task",
                contrast_cfg=SimpleNamespace(name="contrast", output_type="z-score"),
                dry_run=True,
            )

    def test_fmri_analysis_helpers(self):
        from fmri_pipeline.pipelines.fmri_analysis import _safe_slug, _contrast_hash, FmriAnalysisPipeline

        self.assertEqual(_safe_slug("condition a vs b"), "condition_a_vs_b")
        self.assertTrue(len(_contrast_hash(SimpleNamespace(a=1))) == 8)

        tmp = Path(tempfile.mkdtemp())
        explicit = tmp / "sig"
        explicit.mkdir(parents=True, exist_ok=True)

        pipeline = object.__new__(FmriAnalysisPipeline)
        pipeline.config = DotConfig({"paths": {"signature_dir": str(explicit)}})
        pipeline.deriv_root = tmp / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)
        sig_root, _sig_specs = pipeline._discover_signature_root_and_specs()
        self.assertEqual(sig_root, explicit)

    def test_fmri_trial_signature_pipeline_dry_run(self):
        from fmri_pipeline.pipelines.fmri_trial_signatures import FmriTrialSignaturePipeline

        tmp = Path(tempfile.mkdtemp())

        pipeline = object.__new__(FmriTrialSignaturePipeline)
        pipeline.config = DotConfig({})
        pipeline.logger = Mock()
        pipeline.deriv_root = tmp / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)

        class TrialSignatureExtractionConfig:
            def __init__(self, method="lss"):
                self.method = method

        fake_module = types.SimpleNamespace(
            TrialSignatureExtractionConfig=TrialSignatureExtractionConfig,
            run_trial_signature_extraction_for_subject=lambda **_kwargs: {"output_dir": str(tmp / "out")},
        )

        with patch.dict(sys.modules, {"fmri_pipeline.analysis.trial_signatures": fake_module}):
            pipeline.process_subject(
                "0001",
                task="task",
                bids_fmri_root=tmp,
                trial_cfg=TrialSignatureExtractionConfig(method="lss"),
                dry_run=True,
            )

    def test_fmri_trial_signature_pipeline_requires_config_type(self):
        from fmri_pipeline.pipelines.fmri_trial_signatures import FmriTrialSignaturePipeline

        tmp = Path(tempfile.mkdtemp())
        pipeline = object.__new__(FmriTrialSignaturePipeline)
        pipeline.config = DotConfig({})
        pipeline.logger = Mock()
        pipeline.deriv_root = tmp / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)

        class TrialSignatureExtractionConfig:
            def __init__(self, method="lss"):
                self.method = method

        fake_module = types.SimpleNamespace(
            TrialSignatureExtractionConfig=TrialSignatureExtractionConfig,
            run_trial_signature_extraction_for_subject=lambda **_kwargs: {"output_dir": str(tmp / "out")},
        )

        with patch.dict(sys.modules, {"fmri_pipeline.analysis.trial_signatures": fake_module}):
            with self.assertRaises(TypeError):
                pipeline.process_subject(
                    "0001",
                    task="task",
                    bids_fmri_root=tmp,
                    trial_cfg=object(),
                    dry_run=False,
                )
