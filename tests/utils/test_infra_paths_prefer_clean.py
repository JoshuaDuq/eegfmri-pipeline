from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _paths_import_stubs() -> dict[str, types.ModuleType]:
    infra_package = _make_module("eeg_pipeline.infra")
    infra_package.__path__ = [str(Path(__file__).resolve().parents[2] / "eeg_pipeline" / "infra")]
    return {
        "eeg_pipeline.infra": infra_package,
        "mne_bids": _make_module(
            "mne_bids",
            BIDSPath=type(
                "BIDSPath",
                (),
                {"__init__": lambda self, *args, **kwargs: setattr(self, "fpath", None)},
            ),
        ),
        "eeg_pipeline.utils.config.loader": _make_module(
            "eeg_pipeline.utils.config.loader",
            ConfigDict=dict,
        ),
    }


class TestInfraPathsPreferClean(unittest.TestCase):
    def setUp(self):
        self._patcher = patch.dict(sys.modules, _paths_import_stubs())
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        sys.modules.pop("eeg_pipeline.infra.paths", None)
        self.paths = importlib.import_module("eeg_pipeline.infra.paths")
        self.addCleanup(sys.modules.pop, "eeg_pipeline.infra.paths", None)

    def test_load_events_df_falls_back_to_bids_when_clean_events_are_missing(self):
        bids_root = Path(tempfile.mkdtemp())
        bids_events_path = bids_root / "sub-0001" / "eeg" / "sub-0001_task-task_events.tsv"
        bids_events_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"marker": ["bids"]}).to_csv(bids_events_path, sep="\t", index=False)

        loaded = self.paths.load_events_df(
            "0001",
            "task",
            constants={
                "DERIV_ROOT": str(Path(tempfile.mkdtemp())),
                "BIDS_ROOT": str(bids_root),
            },
            prefer_clean=True,
        )

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["marker"].tolist(), ["bids"])

    def test_find_clean_epochs_path_ignores_non_clean_epochs_files(self):
        deriv_root = Path(tempfile.mkdtemp())
        epochs_path = deriv_root / "sub-0001" / "eeg" / "sub-0001_task-task_epo.fif"
        epochs_path.parent.mkdir(parents=True, exist_ok=True)
        epochs_path.write_text("epochs", encoding="utf-8")

        found = self.paths.find_clean_epochs_path("0001", "task", deriv_root=deriv_root)

        self.assertIsNone(found)

    def test_find_clean_events_path_does_not_treat_raw_events_as_clean(self):
        deriv_root = Path(tempfile.mkdtemp())
        epochs_path = deriv_root / "sub-0001" / "eeg" / "sub-0001_task-task_epo.fif"
        raw_events_path = epochs_path.with_name("sub-0001_task-task_events.tsv")
        epochs_path.parent.mkdir(parents=True, exist_ok=True)
        epochs_path.write_text("epochs", encoding="utf-8")
        pd.DataFrame({"onset": [0.0]}).to_csv(raw_events_path, sep="\t", index=False)

        found = self.paths.find_clean_events_path("0001", "task", deriv_root=deriv_root)

        self.assertIsNone(found)
