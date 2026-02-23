import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


class TestBaseCompletion(unittest.TestCase):
    def test_base_init_setup_and_subject_logger(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class Dummy(PipelineBase):
            def process_subject(self, subject: str, task: str, **kwargs):
                return None

        fake_cfg = DotConfig({"project": {"task": "task"}})
        fake_root = Path(tempfile.mkdtemp())

        with patch("eeg_pipeline.pipelines.base.load_config", return_value=fake_cfg), patch(
            "eeg_pipeline.pipelines.base.get_logger", return_value=Mock()
        ), patch("eeg_pipeline.pipelines.base.setup_matplotlib"), patch(
            "eeg_pipeline.pipelines.base.resolve_deriv_root", return_value=fake_root
        ), patch("eeg_pipeline.pipelines.base.ensure_derivatives_dataset_description"), patch(
            "eeg_pipeline.pipelines.base.get_subject_logger", return_value=Mock()
        ) as mock_subj:
            d = Dummy("dummy", config=None)
            d.get_subject_logger("0001")

        self.assertEqual(d.deriv_root, fake_root)
        mock_subj.assert_called_once_with("dummy", "0001")

    def test_base_process_single_subject_failure(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class Dummy(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "x"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp())

            def process_subject(self, subject, task, **kwargs):
                raise RuntimeError("boom")

        d = Dummy()
        ledger = []
        ledger_dir = Path(tempfile.mkdtemp())

        with self.assertRaises(RuntimeError):
            d._process_single_subject("0001", "task", 0.0, ledger, ledger_dir)
        self.assertEqual(ledger[0]["status"], "failed")
        self.assertTrue(Path(ledger[0]["traceback_path"]).exists())

    def test_base_run_batch_fail_fast(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class Dummy(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "x"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp())

            def process_subject(self, subject, task, **kwargs):
                if subject == "0001":
                    raise RuntimeError("fail")

        d = Dummy()
        with patch("eeg_pipeline.pipelines.base.BatchProgress", _NoopBatchProgress):
            with self.assertRaises(RuntimeError):
                d.run_batch(["0001", "0002"], task="x", fail_fast=True, progress=_NoopProgress())

    def test_base_handle_batch_failures_partial_and_super_pass_paths(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class Dummy(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "x"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp())

            def process_subject(self, subject, task, **kwargs):
                super().process_subject(subject, task, **kwargs)

        d = Dummy()
        failed = d._handle_batch_failures(
            ledger=[{"subject": "0001", "status": "failed"}, {"subject": "0002", "status": "success"}],
            subjects=["0001", "0002"],
            ledger_path=Path("/tmp/l.tsv"),
            progress=None,
        )
        self.assertEqual(failed, ["0001"])
        self.assertIsNone(d.run_group_level(["0001"], task="x"))
        self.assertIsNone(d.process_subject("0001", "x"))

    def test_base_run_batch_writes_reproducibility_metadata(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class Dummy(PipelineBase):
            def __init__(self):
                self.name = "dummy"
                self.config = DotConfig({"project": {"task": "x"}, "paths": {"bids_root": "/tmp/bids"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp())

            def process_subject(self, subject: str, task: str, **kwargs):
                return None

        d = Dummy()
        with patch("eeg_pipeline.pipelines.base.BatchProgress", _NoopBatchProgress):
            ledger = d.run_batch(["0001"], task="x", progress=_NoopProgress(), example_flag=True)
        self.assertEqual(ledger[0]["status"], "success")

        metadata_dir = d.deriv_root / "logs" / "run_metadata" / "dummy"
        metadata_files = sorted(metadata_dir.glob("run_*.json"))
        self.assertTrue(metadata_files)

        payload = json.loads(metadata_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(payload["pipeline"], "dummy")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["subjects"], ["0001"])
        self.assertEqual(payload["task"], "x")
        self.assertEqual(payload["specifications"]["example_flag"], True)
        self.assertEqual(payload["config"]["project"]["task"], "x")

    def test_base_run_batch_partial_success_records_partial_status(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        class Dummy(PipelineBase):
            def __init__(self):
                self.name = "dummy_partial"
                self.config = DotConfig({"project": {"task": "x"}})
                self.logger = Mock()
                self.deriv_root = Path(tempfile.mkdtemp())

            def process_subject(self, subject: str, task: str, **kwargs):
                if subject == "0001":
                    raise RuntimeError("boom")
                return None

        d = Dummy()
        with patch("eeg_pipeline.pipelines.base.BatchProgress", _NoopBatchProgress):
            ledger = d.run_batch(["0001", "0002"], task="x", progress=_NoopProgress())

        self.assertEqual(len(ledger), 2)
        self.assertEqual(sum(1 for row in ledger if row.get("status") == "failed"), 1)

        metadata_dir = d.deriv_root / "logs" / "run_metadata" / "dummy_partial"
        metadata_files = sorted(metadata_dir.glob("run_*.json"))
        self.assertTrue(metadata_files)

        payload = json.loads(metadata_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "partial_success")
