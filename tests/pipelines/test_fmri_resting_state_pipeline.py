from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.pipelines_test_utils import DotConfig


class _FakeRestingStateConfig:
    def __init__(self, atlas_labels_img: str):
        self.atlas_labels_img = atlas_labels_img

    def normalized(self):
        return self


class TestFmriRestingStatePipeline(unittest.TestCase):
    def _import_module(self, *, run_side_effect=None):
        fake_run = Mock(return_value={"connectivity_path": "/tmp/connectivity.tsv"})
        if run_side_effect is not None:
            fake_run.side_effect = run_side_effect

        fake_base = types.ModuleType("eeg_pipeline.pipelines.base")

        class PipelineBase:
            def __init__(self, name, config=None):
                self.name = name
                self.config = config or DotConfig({})
                self.logger = Mock()

        fake_base.PipelineBase = PipelineBase

        fake_roots = types.ModuleType("eeg_pipeline.utils.config.roots")
        fake_roots.resolve_fmri_bids_root = Mock(return_value=Path("/tmp/bids-rest"))
        fake_roots.resolve_fmri_deriv_root = Mock(return_value=Path("/tmp/deriv-rest"))

        fake_analysis = types.ModuleType("fmri_pipeline.analysis.resting_state")
        fake_analysis.RestingStateAnalysisConfig = _FakeRestingStateConfig
        fake_analysis.atlas_output_name = Mock(return_value="fake-atlas")
        fake_analysis.run_resting_state_analysis_for_subject = fake_run

        module_name = "fmri_pipeline.pipelines.fmri_resting_state"
        sys.modules.pop(module_name, None)

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.pipelines.base": fake_base,
                "eeg_pipeline.utils.config.roots": fake_roots,
                "fmri_pipeline.analysis.resting_state": fake_analysis,
            },
        ):
            module = importlib.import_module(module_name)

        return module, fake_roots, fake_analysis, fake_run

    def test_init_and_deriv_root_resolution_use_rest_mode(self):
        module, fake_roots, _, _ = self._import_module()

        pipeline = module.FmriRestingStatePipeline(config=DotConfig({}))

        self.assertEqual(pipeline.name, "fmri_resting_state")
        self.assertTrue(pipeline._resolve_task_is_rest())
        self.assertEqual(pipeline._resolve_pipeline_deriv_root(), Path("/tmp/deriv-rest"))
        fake_roots.resolve_fmri_deriv_root.assert_called_with(
            pipeline.config,
            task_is_rest=True,
        )

    def test_process_subject_dry_run_creates_default_output_dir_and_reports_success(self):
        module, _, fake_analysis, fake_run = self._import_module()
        pipeline = module.FmriRestingStatePipeline(config=DotConfig({}))
        progress = Mock()
        temp_root = Path(tempfile.mkdtemp())

        with (
            patch.object(module, "resolve_fmri_bids_root", return_value=temp_root / "bids"),
            patch.object(
                module,
                "resolve_fmri_deriv_root",
                return_value=temp_root / "derivatives",
            ),
        ):
            cfg = module.RestingStateAnalysisConfig(atlas_labels_img="/tmp/atlas.nii.gz")
            pipeline.process_subject("0001", "rest", rest_cfg=cfg, dry_run=True, progress=progress)

        expected_output_dir = (
            temp_root
            / "derivatives"
            / "sub-0001"
            / "fmri"
            / "rest"
            / "task-rest"
            / "atlas-fake-atlas"
        )
        self.assertTrue(expected_output_dir.exists())
        progress.subject_start.assert_called_once_with("sub-0001")
        progress.subject_done.assert_called_once_with("sub-0001", success=True)
        fake_run.assert_not_called()
        fake_analysis.atlas_output_name.assert_called_once_with("/tmp/atlas.nii.gz")

    def test_process_subject_forwards_resolved_paths_and_explicit_output_dir(self):
        module, _, _, fake_run = self._import_module()
        pipeline = module.FmriRestingStatePipeline(config=DotConfig({}))
        progress = Mock()
        temp_root = Path(tempfile.mkdtemp())
        explicit_output_dir = temp_root / "custom-out"

        with (
            patch.object(module, "resolve_fmri_bids_root", return_value=temp_root / "bids"),
            patch.object(
                module,
                "resolve_fmri_deriv_root",
                return_value=temp_root / "derivatives",
            ),
        ):
            cfg = module.RestingStateAnalysisConfig(atlas_labels_img="/tmp/atlas.nii.gz")
            pipeline.process_subject(
                "sub-0002",
                "rest",
                rest_cfg=cfg,
                output_dir=explicit_output_dir,
                dry_run=False,
                progress=progress,
            )

        fake_run.assert_called_once_with(
            bids_fmri_root=(temp_root / "bids").resolve(),
            bids_derivatives=(temp_root / "derivatives").resolve(),
            deriv_root=(temp_root / "derivatives").resolve(),
            subject="sub-0002",
            task="rest",
            cfg=cfg,
            output_dir=explicit_output_dir.resolve(),
        )
        progress.subject_start.assert_called_once_with("sub-0002")
        progress.subject_done.assert_called_once_with("sub-0002", success=True)

    def test_process_subject_marks_progress_failed_when_analysis_raises(self):
        module, _, _, _ = self._import_module(run_side_effect=RuntimeError("rest-fail"))
        pipeline = module.FmriRestingStatePipeline(config=DotConfig({}))
        progress = Mock()
        temp_root = Path(tempfile.mkdtemp())

        with (
            patch.object(module, "resolve_fmri_bids_root", return_value=temp_root / "bids"),
            patch.object(
                module,
                "resolve_fmri_deriv_root",
                return_value=temp_root / "derivatives",
            ),
        ):
            cfg = module.RestingStateAnalysisConfig(atlas_labels_img="/tmp/atlas.nii.gz")
            with self.assertRaisesRegex(RuntimeError, "rest-fail"):
                pipeline.process_subject("0003", "rest", rest_cfg=cfg, progress=progress)

        progress.subject_start.assert_called_once_with("sub-0003")
        progress.subject_done.assert_called_once_with("sub-0003", success=False)


if __name__ == "__main__":
    unittest.main()
