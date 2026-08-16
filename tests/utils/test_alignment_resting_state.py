from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.utils.pipelines_test_utils import DotConfig


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _alignment_import_stubs() -> dict[str, types.ModuleType]:
    return {
        "mne": _make_module("mne", Epochs=object),
        "eeg_pipeline.utils.data.feature_alignment": _make_module(
            "eeg_pipeline.utils.data.feature_alignment",
            require_trial_id_column=lambda frame, *, context: frame["trial_id"],
        ),
    }


class _EpochsStub:
    def __init__(self, n_epochs: int):
        self._n_epochs = int(n_epochs)

    def __len__(self) -> int:
        return self._n_epochs


class TestAlignmentRestingState(unittest.TestCase):
    def setUp(self):
        self._patcher = patch.dict(sys.modules, _alignment_import_stubs())
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        sys.modules.pop("eeg_pipeline.utils.data.alignment", None)
        self.alignment = importlib.import_module("eeg_pipeline.utils.data.alignment")
        self.addCleanup(sys.modules.pop, "eeg_pipeline.utils.data.alignment", None)

    def test_get_aligned_events_synthesizes_rest_alignment_without_clean_events(self):
        deriv_root = Path(tempfile.mkdtemp())
        config = DotConfig({"preprocessing": {"task_is_rest": True}})

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": _make_module(
                    "eeg_pipeline.infra.paths",
                    _find_clean_events_path=lambda **_kwargs: None,
                    _resolve_deriv_root=lambda *_args, **_kwargs: deriv_root,
                )
            },
        ):
            aligned = self.alignment.get_aligned_events(
                _EpochsStub(3),
                "0001",
                "rest",
                strict=True,
                config=config,
            )

        self.assertEqual(aligned["trial_id"].tolist(), [1, 2, 3])

    def test_get_aligned_events_surfaces_deriv_root_resolution_errors(self):
        config = DotConfig({"preprocessing": {"task_is_rest": False}})

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": _make_module(
                    "eeg_pipeline.infra.paths",
                    _find_clean_events_path=lambda **_kwargs: None,
                    _resolve_deriv_root=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        ValueError("deriv-root-missing")
                    ),
                )
            },
        ):
            with self.assertRaisesRegex(ValueError, "deriv-root-missing"):
                self.alignment.get_aligned_events(
                    _EpochsStub(2),
                    "0001",
                    "task",
                    strict=True,
                    config=config,
                )

    def test_get_aligned_events_uses_explicit_deriv_root_without_resolving_config_root(self):
        deriv_root = Path(tempfile.mkdtemp())
        config = DotConfig({"preprocessing": {"task_is_rest": True}})
        recorded: dict[str, Path] = {}

        def _find_clean_events_path(**kwargs):
            recorded["deriv_root"] = kwargs["deriv_root"]
            return None

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": _make_module(
                    "eeg_pipeline.infra.paths",
                    _find_clean_events_path=_find_clean_events_path,
                    _resolve_deriv_root=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("_resolve_deriv_root should not be called")
                    ),
                )
            },
        ):
            aligned = self.alignment.get_aligned_events(
                _EpochsStub(2),
                "0001",
                "rest",
                strict=True,
                config=config,
                deriv_root=deriv_root,
            )

        self.assertEqual(recorded["deriv_root"], deriv_root)
        self.assertEqual(aligned["trial_id"].tolist(), [1, 2])
