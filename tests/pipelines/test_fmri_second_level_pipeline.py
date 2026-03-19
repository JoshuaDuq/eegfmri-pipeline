from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from tests.pipelines_test_utils import DotConfig


@dataclass(frozen=True)
class _FakeSecondLevelConfig:
    model: str
    contrast_names: tuple[str, ...]


class TestFmriSecondLevelPipeline(unittest.TestCase):
    def _import_module(self):
        fake_run = Mock(return_value={"status": "ok"})

        fake_base = types.ModuleType("eeg_pipeline.pipelines.base")

        class PipelineBase:
            def _validate_batch_inputs(self, subjects, task):
                if not subjects:
                    raise ValueError("No subjects specified")
                return task or self.config.get("project.task")

            def _create_run_metadata_context(self, *, subjects, task, kwargs):
                return {
                    "run_id": "test-run",
                    "started_at": datetime.now(timezone.utc),
                    "task": task,
                    "subjects": list(subjects),
                    "specifications": kwargs,
                }

            def _write_run_metadata(self, *args, **kwargs):
                return Path("/tmp/run.json")

        fake_base.PipelineBase = PipelineBase

        fake_analysis = types.ModuleType("fmri_pipeline.analysis.second_level")
        fake_analysis.SecondLevelConfig = _FakeSecondLevelConfig
        fake_analysis.run_second_level_analysis = fake_run

        module_name = "fmri_pipeline.pipelines.fmri_second_level"
        sys.modules.pop(module_name, None)

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.pipelines.base": fake_base,
                "fmri_pipeline.analysis.second_level": fake_analysis,
            },
        ):
            module = importlib.import_module(module_name)

        return module, fake_run

    def _build_pipeline(self):
        module, _ = self._import_module()
        pipeline = object.__new__(module.FmriSecondLevelPipeline)
        pipeline.name = "fmri_second_level"
        pipeline.config = DotConfig({"project": {"task": "pain"}})
        pipeline.deriv_root = Path(tempfile.mkdtemp()) / "derivatives"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)
        pipeline.logger = Mock()
        return pipeline, module

    def test_process_subject_is_not_implemented(self) -> None:
        pipeline, _ = self._build_pipeline()

        with self.assertRaises(NotImplementedError):
            pipeline.process_subject("0001", task="pain")

    def test_run_batch_rejects_single_subject(self) -> None:
        pipeline, module = self._build_pipeline()
        cfg = module.SecondLevelConfig(model="one-sample", contrast_names=("pain",))

        with self.assertRaisesRegex(
            ValueError,
            "requires at least two selected subjects",
        ):
            pipeline.run_batch(["0001"], task="pain", second_level_cfg=cfg)

    def test_run_group_level_rejects_invalid_config_type(self) -> None:
        pipeline, _ = self._build_pipeline()

        with self.assertRaisesRegex(
            TypeError,
            "second_level_cfg must be a SecondLevelConfig instance",
        ):
            pipeline.run_group_level(["0001", "0002"], task="pain", second_level_cfg={})

    def test_run_group_level_forwards_to_second_level_analysis(self) -> None:
        pipeline, module = self._build_pipeline()
        cfg = module.SecondLevelConfig(model="one-sample", contrast_names=("pain",))

        with patch.object(module, "run_second_level_analysis", return_value={"status": "ok"}) as mock_run:
            result = pipeline.run_group_level(
                ["0001", "0002"],
                task="pain",
                second_level_cfg=cfg,
                dry_run=True,
                progress=Mock(),
            )

        self.assertEqual(result, {"status": "ok"})
        mock_run.assert_called_once()

    def test_run_batch_calls_group_level_and_writes_metadata(self) -> None:
        pipeline, module = self._build_pipeline()
        cfg = module.SecondLevelConfig(model="one-sample", contrast_names=("pain",))
        progress = Mock()

        with patch.object(
            pipeline,
            "run_group_level",
            return_value={"n_subjects": 2},
        ) as mock_run_group, patch.object(
            pipeline,
            "_write_run_metadata",
            return_value=Path("/tmp/run.json"),
        ) as mock_write_metadata:
            result = pipeline.run_batch(
                ["0001", "0002"],
                task="pain",
                second_level_cfg=cfg,
                dry_run=False,
                progress=progress,
            )

        self.assertEqual(result, {"n_subjects": 2})
        progress.start.assert_called_once_with("fmri_second_level", ["0001", "0002"])
        progress.complete.assert_called_once_with(success=True)
        mock_run_group.assert_called_once()
        mock_write_metadata.assert_called_once()
