from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.pipelines_test_utils import DotConfig


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _EpochsStub:
    def __init__(self) -> None:
        self.event_id = {"stim": 1}
        self.selection = [0, 1]

    def __len__(self) -> int:
        return 2


class TestPreprocessingCleanEvents(unittest.TestCase):
    def test_write_clean_events_assigns_one_based_trial_ids(self):
        mne_home = Path(tempfile.mkdtemp())
        with patch.dict(
            os.environ,
            {"HOME": str(mne_home), "MNE_DONTWRITE_HOME": "true"},
            clear=False,
        ), patch.dict(
            sys.modules,
            {
                "mne": _make_module(
                    "mne",
                    read_epochs=lambda *_args, **_kwargs: _EpochsStub(),
                    BaseEpochs=object,
                ),
                "eeg_pipeline.utils.analysis.artifact_qc": _make_module(
                    "eeg_pipeline.utils.analysis.artifact_qc",
                    band_power_metric=lambda *_args, **_kwargs: np.array([], dtype=float),
                    mean_abs_correlation_metric=lambda *_args, **_kwargs: np.array([], dtype=float),
                    pick_channels=lambda *_args, **_kwargs: np.array([], dtype=int),
                    window_mask=lambda *_args, **_kwargs: np.array([], dtype=bool),
                ),
                "eeg_pipeline.infra": _make_module(
                    "eeg_pipeline.infra",
                    __path__=[],
                ),
                "eeg_pipeline.infra.tsv": _make_module(
                    "eeg_pipeline.infra.tsv",
                    read_tsv=lambda *_args, **_kwargs: pd.DataFrame(),
                ),
                "yaml": _make_module(
                    "yaml",
                    safe_load=lambda *_args, **_kwargs: {},
                    YAMLError=Exception,
                ),
                "mne_bids": _make_module(
                    "mne_bids",
                    BIDSPath=type(
                        "BIDSPath",
                        (),
                        {"__init__": lambda self, *args, **kwargs: setattr(self, "fpath", None)},
                    ),
                )
            },
        ):
            from eeg_pipeline.utils.data import preprocessing as preproc

        bids_root = Path(tempfile.mkdtemp())
        epochs_path = Path(tempfile.mkdtemp()) / "sub-0001_task-task_proc-clean_epo.fif"
        epochs_path.write_text("epochs", encoding="utf-8")
        (bids_root / "sub-0001" / "eeg").mkdir(parents=True, exist_ok=True)

        events_df = pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "onset": [0.0, 1.0],
                "duration": [0.5, 0.5],
            }
        )
        sidecar_calls: list[tuple[Path, list[str]]] = []

        with patch.object(preproc, "mne", _make_module("mne", read_epochs=lambda *_args, **_kwargs: _EpochsStub())), patch.object(
            preproc,
            "_load_subject_events_for_epochs",
            return_value=events_df,
        ), patch.object(
            preproc,
            "_compute_clean_events_qc_table",
            return_value=pd.DataFrame(index=np.arange(2, dtype=int)),
        ), patch.dict(
            sys.modules,
            {
                "eeg_pipeline.analysis.utilities.bids_metadata": _make_module(
                    "eeg_pipeline.analysis.utilities.bids_metadata",
                    ensure_events_sidecar=lambda path, columns: sidecar_calls.append(
                        (Path(path), list(columns))
                    ),
                )
            },
        ):
            out_path = preproc.write_clean_events_tsv_for_epochs(
                subject="0001",
                task="task",
                bids_root=bids_root,
                epochs_path=epochs_path,
                config=DotConfig({}),
                conditions=["stim"],
            )

        written = pd.read_csv(out_path, sep="\t")
        self.assertEqual(written["trial_id"].tolist(), [1, 2])
        self.assertEqual(written["epoch_index"].tolist(), [0, 1])
        self.assertEqual(len(sidecar_calls), 1)
