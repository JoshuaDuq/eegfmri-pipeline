from __future__ import annotations

import contextlib
import importlib
import sys
import types
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _ica_import_stubs() -> dict[str, types.ModuleType]:
    return {
        "mne": _make_module(
            "mne",
            utils=types.SimpleNamespace(
                use_log_level=lambda *_args, **_kwargs: contextlib.nullcontext()
            ),
        ),
        "mne_icalabel": _make_module(
            "mne_icalabel",
            label_components=lambda *_args, **_kwargs: {
                "labels": ["brain"],
                "y_pred_proba": np.array([0.1]),
            },
        ),
        "mne_bids": _make_module(
            "mne_bids",
            get_entities_from_fname=lambda _path: {"subject": "0001"},
            BIDSPath=Mock(),
        ),
        "mne_bids_pipeline._logging": _make_module(
            "mne_bids_pipeline._logging",
            gen_log_kwargs=lambda **kwargs: kwargs,
            logger=types.SimpleNamespace(
                info=Mock(),
                error=Mock(),
                warning=Mock(),
                title=Mock(),
            ),
        ),
        "eeg_pipeline.preprocessing.pipeline.utils": _make_module(
            "eeg_pipeline.preprocessing.pipeline.utils",
            get_derived_path=lambda path, *_args, **_kwargs: path.replace(
                "_proc-icafit_ica.fif",
                "_proc-ica_components.tsv",
            ),
        ),
        "eeg_pipeline.preprocessing.pipeline.io": _make_module(
            "eeg_pipeline.preprocessing.pipeline.io",
            load_ica=lambda _path: types.SimpleNamespace(n_components_=1, exclude=[]),
            load_epochs=lambda _path: types.SimpleNamespace(
                set_eeg_reference=lambda *_args, **_kwargs: None
            ),
            read_components_tsv=lambda _path: pd.DataFrame(
                {"component": [0], "status": ["good"], "status_description": [""]}
            ),
            create_empty_components_tsv=lambda n_components: pd.DataFrame(
                {
                    "component": list(range(n_components)),
                    "status": ["good"] * n_components,
                    "status_description": [""] * n_components,
                }
            ),
            write_components_tsv=Mock(),
            save_ica=Mock(),
        ),
        "eeg_pipeline.preprocessing.pipeline.preprocess": _make_module(
            "eeg_pipeline.preprocessing.pipeline.preprocess"
        ),
        "eeg_pipeline.preprocessing.pipeline.tfr": _make_module(
            "eeg_pipeline.preprocessing.pipeline.tfr"
        ),
        "eeg_pipeline.preprocessing.pipeline.stats": _make_module(
            "eeg_pipeline.preprocessing.pipeline.stats"
        ),
    }


class TestIcaFailFast(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.dict(sys.modules, _ica_import_stubs())
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("eeg_pipeline.preprocessing.pipeline.ica", None)
        self.ica = importlib.import_module("eeg_pipeline.preprocessing.pipeline.ica")
        self.addCleanup(sys.modules.pop, "eeg_pipeline.preprocessing.pipeline.ica", None)

    def test_run_ica_label_single_file_raises_when_component_write_fails(self):
        self.ica.io.write_components_tsv.side_effect = RuntimeError("write boom")

        with self.assertRaisesRegex(RuntimeError, "write boom"):
            self.ica.run_ica_label_single_file("/tmp/sub-0001_task-task_proc-icafit_ica.fif")
