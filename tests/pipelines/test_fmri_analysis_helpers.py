from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


class TestFmriAnalysisHelpers(unittest.TestCase):
    def _import_module(self):
        fake_base = types.ModuleType("eeg_pipeline.pipelines.base")
        fake_base.PipelineBase = type("PipelineBase", (), {})

        sys.modules.pop("fmri_pipeline.pipelines.fmri_analysis", None)

        with patch.dict(
            sys.modules,
            {"eeg_pipeline.pipelines.base": fake_base},
        ):
            return importlib.import_module("fmri_pipeline.pipelines.fmri_analysis")

    def test_contrast_arg_for_model_runs_passes_through_lists_and_tuples(self) -> None:
        module = self._import_module()
        flm = types.SimpleNamespace(design_matrices_=[object(), object()])

        values = [
            ["A", "B"],
            ("A", "B"),
            {"name": "A"},
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertIs(module._contrast_arg_for_model_runs(flm, value), value)

    def test_contrast_arg_for_model_runs_broadcasts_single_contrast_across_runs(self) -> None:
        module = self._import_module()
        flm = types.SimpleNamespace(design_matrices_=[object(), object(), object()])

        result = module._contrast_arg_for_model_runs(flm, "pain")

        self.assertEqual(result, ["pain", "pain", "pain"])

    def test_contrast_arg_for_model_runs_keeps_single_run_scalar(self) -> None:
        module = self._import_module()
        flm = types.SimpleNamespace(design_matrices_=[object()])

        self.assertEqual(module._contrast_arg_for_model_runs(flm, "pain"), "pain")
