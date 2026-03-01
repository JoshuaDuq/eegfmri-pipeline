from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock, patch

from eeg_pipeline.cli.commands.features_orchestrator import run_features


class TestFeaturesOrchestratorFailures(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            mode="compute",
            task="thermalactive",
            categories=["sourcelocalization"],
            time_range=None,
            bands=None,
            spatial=None,
            tmin=None,
            tmax=None,
            aggregation_method="mean",
            fixed_templates_path=None,
            bids_root=None,
            deriv_root=None,
            freesurfer_dir=None,
            set_overrides=None,
        )

    def test_run_features_raises_when_any_subject_fails(self):
        args = self._args()
        config = {}
        subjects = ["0000", "0001"]

        pipeline_instance = MagicMock()
        pipeline_instance.run_batch.return_value = [
            {"subject": "0000", "status": "success"},
            {"subject": "0001", "status": "failed"},
        ]

        with patch(
            "eeg_pipeline.cli.commands.features_orchestrator.create_progress_reporter",
            return_value=None,
        ), patch(
            "eeg_pipeline.cli.commands.features_orchestrator._apply_feature_config_overrides"
        ), patch(
            "eeg_pipeline.cli.commands.features_orchestrator.apply_set_overrides"
        ), patch(
            "eeg_pipeline.cli.commands.features_orchestrator.resolve_task",
            return_value="thermalactive",
        ), patch(
            "eeg_pipeline.pipelines.features.FeaturePipeline",
            return_value=pipeline_instance,
        ):
            with self.assertRaisesRegex(RuntimeError, "subjects failed"):
                run_features(args, subjects, config)

