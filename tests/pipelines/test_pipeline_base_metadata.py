from __future__ import annotations

import importlib
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch


class TestPipelineBaseMetadata(unittest.TestCase):
    def _import_base_module(self):
        fake_loader = types.ModuleType("eeg_pipeline.utils.config.loader")
        fake_loader.load_config = lambda: {}

        fake_logging = types.ModuleType("eeg_pipeline.infra.logging")
        fake_logging.get_logger = lambda *_args, **_kwargs: Mock()
        fake_logging.get_subject_logger = lambda *_args, **_kwargs: Mock()

        fake_figures = types.ModuleType("eeg_pipeline.plotting.io.figures")
        fake_figures.setup_matplotlib = lambda *_args, **_kwargs: None

        fake_paths = types.ModuleType("eeg_pipeline.infra.paths")
        fake_paths.ensure_derivatives_dataset_description = lambda **_kwargs: None
        fake_paths.resolve_deriv_root = lambda **_kwargs: Path("/tmp/derivatives")

        fake_progress = types.ModuleType("eeg_pipeline.utils.progress")

        class BatchProgress:
            pass

        fake_progress.BatchProgress = BatchProgress

        fake_packages = {
            "eeg_pipeline.utils": types.ModuleType("eeg_pipeline.utils"),
            "eeg_pipeline.utils.config": types.ModuleType("eeg_pipeline.utils.config"),
            "eeg_pipeline.infra": types.ModuleType("eeg_pipeline.infra"),
            "eeg_pipeline.plotting": types.ModuleType("eeg_pipeline.plotting"),
            "eeg_pipeline.plotting.io": types.ModuleType("eeg_pipeline.plotting.io"),
        }
        for module in fake_packages.values():
            module.__path__ = []  # type: ignore[attr-defined]

        module_name = "eeg_pipeline.pipelines.base"
        sys.modules.pop(module_name, None)

        with patch.dict(
            sys.modules,
            {
                **fake_packages,
                "eeg_pipeline.utils.config.loader": fake_loader,
                "eeg_pipeline.infra.logging": fake_logging,
                "eeg_pipeline.plotting.io.figures": fake_figures,
                "eeg_pipeline.infra.paths": fake_paths,
                "eeg_pipeline.utils.progress": fake_progress,
            },
        ):
            return importlib.import_module(module_name)

    def test_sanitize_metadata_value_handles_common_python_types(self) -> None:
        module = self._import_base_module()

        @dataclass
        class Example:
            path: Path
            tags: list[Path]

        class Dummy(module.PipelineBase):
            def process_subject(self, subject: str, task: str, **kwargs):
                return None

        helper = object.__new__(Dummy)

        sample = Example(path=Path("/tmp/example"), tags=[Path("/tmp/a"), Path("/tmp/b")])
        self.assertEqual(
            helper._sanitize_metadata_value(sample),
            {"path": "/tmp/example", "tags": ["/tmp/a", "/tmp/b"]},
        )

    def test_sanitize_metadata_value_handles_array_like_and_object_dicts(self) -> None:
        module = self._import_base_module()

        class Dummy(module.PipelineBase):
            def process_subject(self, subject: str, task: str, **kwargs):
                return None

        class ArrayLike:
            shape = (2, 3)
            dtype = "float32"

        class Holder:
            def __init__(self):
                self.path = Path("/tmp/inner")
                self.nested = {"value": 3}

        helper = object.__new__(Dummy)

        self.assertEqual(
            helper._sanitize_metadata_value(ArrayLike()),
            {"__type__": "ArrayLike", "shape": [2, 3], "dtype": "float32"},
        )
        self.assertEqual(
            helper._sanitize_metadata_value(Holder()),
            {"path": "/tmp/inner", "nested": {"value": 3}},
        )

    def test_sanitize_metadata_value_falls_back_past_max_depth(self) -> None:
        module = self._import_base_module()

        class Dummy(module.PipelineBase):
            def process_subject(self, subject: str, task: str, **kwargs):
                return None

        helper = object.__new__(Dummy)

        nested: object = "leaf"
        for _ in range(10):
            nested = {"child": nested}

        rendered = helper._sanitize_metadata_value(nested)
        self.assertIsInstance(rendered, dict)
        self.assertIn("child", rendered)
