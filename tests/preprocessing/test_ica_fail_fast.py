from __future__ import annotations

import contextlib
import importlib
import sys
import tempfile
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
        stubs = _ica_import_stubs()
        patcher = unittest.mock.patch.dict(sys.modules, stubs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._patch_cached_pipeline_submodules(stubs)
        sys.modules.pop("eeg_pipeline.preprocessing.pipeline.ica", None)
        self.ica = importlib.import_module("eeg_pipeline.preprocessing.pipeline.ica")
        self.addCleanup(sys.modules.pop, "eeg_pipeline.preprocessing.pipeline.ica", None)

    def _patch_cached_pipeline_submodules(self, stubs: dict[str, types.ModuleType]) -> None:
        package = sys.modules.get("eeg_pipeline.preprocessing.pipeline")
        if package is None:
            return

        names = ("utils", "io", "preprocess", "tfr", "stats")
        previous = {
            name: getattr(package, name)
            for name in names
            if hasattr(package, name)
        }
        missing = [name for name in names if not hasattr(package, name)]

        for name in names:
            setattr(package, name, stubs[f"eeg_pipeline.preprocessing.pipeline.{name}"])

        def restore() -> None:
            for name in names:
                if name in previous:
                    setattr(package, name, previous[name])
                elif name in missing and hasattr(package, name):
                    delattr(package, name)

        self.addCleanup(restore)

    def test_run_ica_label_single_file_raises_when_component_write_fails(self):
        self.ica.io.write_components_tsv.side_effect = RuntimeError("write boom")

        with self.assertRaisesRegex(RuntimeError, "write boom"):
            self.ica.run_ica_label_single_file("/tmp/sub-0001_task-task_proc-icafit_ica.fif")

    def test_run_ica_label_rebuilds_stale_component_table(self):
        fitted_ica = types.SimpleNamespace(n_components_=2, exclude=[])
        self.ica.io.load_ica = Mock(return_value=fitted_ica)
        self.ica.io.read_components_tsv = Mock(
            return_value=pd.DataFrame(
                {
                    "component": [0, 1, 2],
                    "status": ["bad", "bad", "bad"],
                    "status_description": ["old", "old", "old"],
                }
            )
        )
        self.ica.io.create_empty_components_tsv = Mock(
            return_value=pd.DataFrame(
                {
                    "component": [0, 1],
                    "status": [None, None],
                    "status_description": [None, None],
                }
            )
        )

        with unittest.mock.patch.object(
            self.ica,
            "label_components",
            return_value={
                "labels": ["brain", "eye blink"],
                "y_pred_proba": np.array([0.1, 0.95]),
            },
        ):
            self.ica.run_ica_label_single_file(
                "/tmp/sub-0001_task-task_proc-icafit_ica.fif",
                keep_mnebids_bads=True,
            )

        written = self.ica.io.write_components_tsv.call_args.args[0]
        self.assertEqual(written["component"].tolist(), [0, 1])
        self.assertEqual(
            written["mne_icalabel_labels"].tolist(), ["brain", "eye blink"]
        )
        self.assertEqual(written["status"].tolist(), ["good", "bad"])
        self.assertEqual(
            written["status_description"].tolist(),
            ["", "Bad component detected by mne_icalabel"],
        )
        self.assertEqual(fitted_ica.exclude, [1])

    def test_run_ica_label_discovers_subject_level_ica_when_task_is_selected(self):
        with tempfile.TemporaryDirectory() as pipeline_path:
            ica_path = f"{pipeline_path}/sub-0001/eeg/sub-0001_proc-icafit_ica.fif"

            def bids_path(**entities):
                matches = [ica_path] if entities.get("task") is None else []
                return types.SimpleNamespace(match=lambda: matches)

            labeled = pd.DataFrame({"participant_id": ["0001"]})
            mne_bids = sys.modules["mne_bids"]
            with (
                unittest.mock.patch.object(mne_bids, "BIDSPath", side_effect=bids_path),
                unittest.mock.patch.object(
                    self.ica,
                    "run_ica_label_single_file",
                    return_value=labeled,
                ) as run_single,
            ):
                self.ica.run_ica_label(
                    pipeline_path=pipeline_path,
                    task="thermalactive",
                    subjects=["0001"],
                )

            run_single.assert_called_once_with(
                ica_path,
                prob_threshold=0.8,
                labels_to_keep=["brain", "other"],
                keep_mnebids_bads=False,
            )

    def test_run_ica_label_raises_when_no_subject_level_ica_exists(self):
        mne_bids = sys.modules["mne_bids"]
        bids_path = Mock(return_value=types.SimpleNamespace(match=lambda: []))

        with (
            tempfile.TemporaryDirectory() as pipeline_path,
            unittest.mock.patch.object(mne_bids, "BIDSPath", bids_path),
            self.assertRaisesRegex(FileNotFoundError, "No subject-level ICA files found"),
        ):
            self.ica.run_ica_label(
                pipeline_path=pipeline_path,
                task="thermalactive",
                subjects=["0001"],
            )
