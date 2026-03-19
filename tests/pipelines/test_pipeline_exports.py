from __future__ import annotations

import types
import unittest
from unittest.mock import patch


class TestPipelineExports(unittest.TestCase):
    def test_getattr_lazy_imports_known_exports(self) -> None:
        import eeg_pipeline.pipelines as pipelines

        fake_module = types.SimpleNamespace(PipelineBase=object())

        with patch(
            "eeg_pipeline.pipelines.import_module",
            return_value=fake_module,
        ) as mock_import:
            exported = pipelines.PipelineBase

        self.assertIs(exported, fake_module.PipelineBase)
        mock_import.assert_called_once_with("eeg_pipeline.pipelines.base")

    def test_getattr_raises_for_unknown_export(self) -> None:
        import eeg_pipeline.pipelines as pipelines

        with self.assertRaises(AttributeError):
            _ = pipelines.not_an_export

    def test_dir_includes_public_exports(self) -> None:
        import eeg_pipeline.pipelines as pipelines

        names = pipelines.__dir__()
        for expected in pipelines.__all__:
            self.assertIn(expected, names)
