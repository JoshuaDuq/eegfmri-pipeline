from __future__ import annotations

import importlib
import sys
import unittest
from builtins import __import__ as _orig_import
from unittest.mock import patch


class TestCliProgress(unittest.TestCase):
    def test_module_imports_without_resource_module(self) -> None:
        original_resource = sys.modules.pop("resource", None)
        try:
            with patch("builtins.__import__", side_effect=self._import_without_resource):
                sys.modules.pop("eeg_pipeline.cli.progress", None)
                module = importlib.import_module("eeg_pipeline.cli.progress")
            self.assertIsNone(module.resource)
        finally:
            sys.modules.pop("eeg_pipeline.cli.progress", None)
            if original_resource is not None:
                sys.modules["resource"] = original_resource

    def test_get_resource_usage_falls_back_to_windows_memory(self) -> None:
        module = importlib.import_module("eeg_pipeline.cli.progress")
        reporter = module.ProgressReporter(enabled=True)

        with patch.object(module, "resource", None), patch.object(module.sys, "platform", "win32"), patch.object(
            reporter,
            "_get_windows_memory_usage_gb",
            return_value=1.5,
        ), patch.object(module.time, "process_time", return_value=0.0):
            usage = reporter._get_resource_usage()

        self.assertEqual(usage["memory"], 1.5)
        self.assertEqual(usage["cpu"], 0.0)

    @staticmethod
    def _import_without_resource(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource":
            raise ImportError("No module named 'resource'")
        return _orig_import(name, globals, locals, fromlist, level)


if __name__ == "__main__":
    unittest.main()
