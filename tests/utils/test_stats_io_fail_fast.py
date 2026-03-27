from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _stats_io_import_stubs() -> dict[str, types.ModuleType]:
    return {
        "eeg_pipeline.analysis.features.rest": _make_module(
            "eeg_pipeline.analysis.features.rest",
            is_resting_state_feature_mode=lambda config: False,
        ),
        "eeg_pipeline.utils.data.covariates": _make_module(
            "eeg_pipeline.utils.data.covariates",
            _build_covariate_matrices=lambda *_args, **_kwargs: (None, None),
        ),
    }


class TestStatsIoFailFast(unittest.TestCase):
    def setUp(self):
        self._patcher = patch.dict(sys.modules, _stats_io_import_stubs())
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        sys.modules.pop("eeg_pipeline.utils.data.stats_io", None)
        self.stats_io = importlib.import_module("eeg_pipeline.utils.data.stats_io")
        self.addCleanup(sys.modules.pop, "eeg_pipeline.utils.data.stats_io", None)

    def test_load_subject_scatter_data_surfaces_loading_errors(self):
        logger = Mock()

        with patch.object(
            self.stats_io,
            "_load_epochs_for_subject",
            side_effect=ValueError("missing epochs"),
        ):
            with self.assertRaisesRegex(ValueError, "missing epochs"):
                self.stats_io.load_subject_scatter_data(
                    "0001",
                    "pain",
                    Path("/tmp"),
                    {},
                    logger,
                )
