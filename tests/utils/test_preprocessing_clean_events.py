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
import pytest

from tests.utils.pipelines_test_utils import DotConfig


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _EpochsStub:
    """Two condition events, both kept.

    ``drop_log`` carries one entry per event MNE was originally given: ``('IGNORED',)``
    for events outside the requested conditions, an empty tuple for a kept epoch, and a
    reason tuple for a dropped one. The extra ``IGNORED`` entry stands in for the scanner
    markers that outnumber the trials in a real recording.
    """

    def __init__(self) -> None:
        self.event_id = {"stim": 1}
        self.selection = [0, 2]
        self.drop_log = ((), ("IGNORED",), ())

    def __len__(self) -> int:
        return 2


class _ThreeEpochsStub:
    def __init__(self) -> None:
        self.event_id = {"stim": 1}
        self.selection = [0, 1, 3]
        self.drop_log = ((), (), ("IGNORED",), ())

    def __len__(self) -> int:
        return 3


class TestPreprocessingCleanEvents(unittest.TestCase):
    def test_presented_events_keep_the_rejection_denominator(self):
        from eeg_pipeline.utils.data import preprocessing as preproc

        bids_root = Path(tempfile.mkdtemp())
        (bids_root / "sub-0001" / "eeg").mkdir(parents=True)
        events = pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "run_id": [1, 2],
            }
        )

        with patch.object(
            preproc,
            "_load_subject_events_for_epochs",
            return_value=events,
        ):
            presented = preproc.presented_events_for_epochs(
                subject="0001",
                task="task",
                bids_root=bids_root,
                epochs=_EpochsStub(),
            )

        self.assertEqual(presented["event_index"].tolist(), [0, 1])
        self.assertEqual(presented["run_id"].tolist(), [1, 2])

    def test_write_clean_events_assigns_one_based_trial_ids(self):
        mne_home = Path(tempfile.mkdtemp())
        with (
            patch.dict(
                os.environ,
                {"HOME": str(mne_home), "MNE_DONTWRITE_HOME": "true"},
                clear=False,
            ),
            patch.dict(
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
                        mean_abs_correlation_metric=lambda *_args, **_kwargs: np.array(
                            [], dtype=float
                        ),
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
                            {
                                "__init__": lambda self, *args, **kwargs: setattr(
                                    self, "fpath", None
                                )
                            },
                        ),
                    ),
                },
            ),
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

        with (
            patch.object(
                preproc,
                "mne",
                _make_module("mne", read_epochs=lambda *_args, **_kwargs: _EpochsStub()),
            ),
            patch.object(
                preproc,
                "_load_subject_events_for_epochs",
                return_value=events_df,
            ),
            patch.object(
                preproc,
                "_compute_clean_events_qc_table",
                return_value=pd.DataFrame(index=np.arange(2, dtype=int)),
            ),
            patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.analysis.utilities.bids_metadata": _make_module(
                        "eeg_pipeline.analysis.utilities.bids_metadata",
                        ensure_events_sidecar=lambda path, columns: sidecar_calls.append(
                            (Path(path), list(columns))
                        ),
                    )
                },
            ),
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

    def test_write_clean_events_aligns_epochs_with_bids_trial_type(self):
        from eeg_pipeline.utils.data import preprocessing as preproc

        bids_root = Path(tempfile.mkdtemp())
        epochs_path = Path(tempfile.mkdtemp()) / "sub-0001_task-task_proc-clean_epo.fif"
        epochs_path.write_text("epochs", encoding="utf-8")
        (bids_root / "sub-0001" / "eeg").mkdir(parents=True, exist_ok=True)

        events_df = pd.DataFrame(
            {
                "trial_type": ["stim", "stim", "stim"],
                "condition": ["keep", "drop", "drop"],
                "onset": [0.0, 1.0, 2.0],
                "duration": [0.5, 0.5, 0.5],
            }
        )

        with (
            patch.object(
                preproc,
                "mne",
                _make_module("mne", read_epochs=lambda *_args, **_kwargs: _ThreeEpochsStub()),
            ),
            patch.object(
                preproc,
                "_load_subject_events_for_epochs",
                return_value=events_df,
            ),
            patch.object(
                preproc,
                "_compute_clean_events_qc_table",
                return_value=pd.DataFrame(index=np.arange(3, dtype=int)),
            ),
            patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.analysis.utilities.bids_metadata": _make_module(
                        "eeg_pipeline.analysis.utilities.bids_metadata",
                        ensure_events_sidecar=lambda *_args, **_kwargs: None,
                    )
                },
            ),
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
        self.assertEqual(written["event_index"].tolist(), [0, 1, 2])
        self.assertEqual(written["condition"].tolist(), ["keep", "drop", "drop"])

    def test_write_clean_events_uses_run_level_events_when_available(self):
        from eeg_pipeline.utils.data import preprocessing as preproc

        bids_root = Path(tempfile.mkdtemp())
        epochs_path = Path(tempfile.mkdtemp()) / "sub-0001_task-task_proc-clean_epo.fif"
        epochs_path.write_text("epochs", encoding="utf-8")
        eeg_dir = bids_root / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "onset": [0.0, 1.0],
                "source": ["combined", "combined"],
            }
        ).to_csv(eeg_dir / "sub-0001_task-task_events.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [0.0],
                "source": ["run-1"],
            }
        ).to_csv(eeg_dir / "sub-0001_task-task_run-1_events.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "onset": [0.0, 1.0],
                "source": ["run-2", "run-2"],
            }
        ).to_csv(eeg_dir / "sub-0001_task-task_run-2_events.tsv", sep="\t", index=False)

        with (
            patch.object(
                preproc,
                "mne",
                _make_module("mne", read_epochs=lambda *_args, **_kwargs: _ThreeEpochsStub()),
            ),
            patch.object(preproc, "read_tsv", lambda path: pd.read_csv(path, sep="\t")),
            patch.object(
                preproc,
                "_compute_clean_events_qc_table",
                return_value=pd.DataFrame(index=np.arange(3, dtype=int)),
            ),
            patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.analysis.utilities.bids_metadata": _make_module(
                        "eeg_pipeline.analysis.utilities.bids_metadata",
                        ensure_events_sidecar=lambda *_args, **_kwargs: None,
                    )
                },
            ),
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
        self.assertEqual(written["source"].tolist(), ["run-1", "run-2", "run-2"])
        self.assertEqual(written["run_id"].tolist(), [1, 2, 2])

    def test_clean_events_carry_the_autoreject_repair_counts(self):
        """Which channels in a trial are spline estimates has to reach the trial table.

        Otherwise a downstream connectivity or CSD feature has no way to know that one of
        the two channels it is relating was reconstructed from the other's neighbourhood.
        """
        from eeg_pipeline.preprocessing.autoreject_log import (
            AutorejectLog,
            autoreject_log_path_for_epochs,
            write_autoreject_log,
        )
        from eeg_pipeline.utils.data import preprocessing as preproc

        bids_root = Path(tempfile.mkdtemp())
        epochs_path = Path(tempfile.mkdtemp()) / "sub-0001_task-task_proc-clean_epo.fif"
        epochs_path.write_text("epochs", encoding="utf-8")
        (bids_root / "sub-0001" / "eeg").mkdir(parents=True, exist_ok=True)

        # Three pre-rejection epochs; the middle one was dropped, so the clean file holds
        # the two the stub reports.
        write_autoreject_log(
            AutorejectLog(
                ch_names=("C3", "Cz", "C4"),
                labels=np.array([[2, 2, 1], [1, 1, 1], [0, 0, 0]]),
                bad_epochs=np.array([False, True, False]),
                n_interpolate=2,
                consensus=0.8,
            ),
            autoreject_log_path_for_epochs(epochs_path),
        )

        events_df = pd.DataFrame(
            {"trial_type": ["stim", "stim"], "onset": [0.0, 1.0], "duration": [0.5, 0.5]}
        )

        with (
            patch.object(
                preproc,
                "mne",
                _make_module("mne", read_epochs=lambda *_args, **_kwargs: _EpochsStub()),
            ),
            patch.object(preproc, "_load_subject_events_for_epochs", return_value=events_df),
            patch.object(
                preproc,
                "_compute_clean_events_qc_table",
                return_value=pd.DataFrame(index=np.arange(2, dtype=int)),
            ),
            patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.analysis.utilities.bids_metadata": _make_module(
                        "eeg_pipeline.analysis.utilities.bids_metadata",
                        ensure_events_sidecar=lambda *_args, **_kwargs: None,
                    )
                },
            ),
        ):
            out_path = preproc.write_clean_events_tsv_for_epochs(
                subject="0001",
                task="task",
                bids_root=bids_root,
                epochs_path=epochs_path,
                config=DotConfig({"preprocessing": {"autoreject_log": True}}),
                conditions=["stim"],
            )

        written = pd.read_csv(out_path, sep="\t")
        self.assertEqual(written["n_channels_interpolated"].tolist(), [2, 0])
        self.assertEqual(written["n_channels_bad_not_interpolated"].tolist(), [1, 0])

    def test_enabled_autoreject_counts_without_a_log_is_an_error(self):
        from eeg_pipeline.utils.data import preprocessing as preproc

        bids_root = Path(tempfile.mkdtemp())
        epochs_path = Path(tempfile.mkdtemp()) / "sub-0001_task-task_proc-clean_epo.fif"
        epochs_path.write_text("epochs", encoding="utf-8")
        (bids_root / "sub-0001" / "eeg").mkdir(parents=True, exist_ok=True)

        events_df = pd.DataFrame(
            {"trial_type": ["stim", "stim"], "onset": [0.0, 1.0], "duration": [0.5, 0.5]}
        )

        with (
            patch.object(
                preproc,
                "mne",
                _make_module("mne", read_epochs=lambda *_args, **_kwargs: _EpochsStub()),
            ),
            patch.object(preproc, "_load_subject_events_for_epochs", return_value=events_df),
            patch.object(
                preproc,
                "_compute_clean_events_qc_table",
                return_value=pd.DataFrame(index=np.arange(2, dtype=int)),
            ),
            patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.analysis.utilities.bids_metadata": _make_module(
                        "eeg_pipeline.analysis.utilities.bids_metadata",
                        ensure_events_sidecar=lambda *_args, **_kwargs: None,
                    )
                },
            ),
            self.assertRaises(FileNotFoundError),
        ):
            preproc.write_clean_events_tsv_for_epochs(
                subject="0001",
                task="task",
                bids_root=bids_root,
                epochs_path=epochs_path,
                config=DotConfig({"preprocessing": {"autoreject_log": True}}),
                conditions=["stim"],
            )


# --------------------------------------------------------------------------------------
# Pre-rejection callers must not be asked for a record that cannot exist yet
# --------------------------------------------------------------------------------------


def _minimal_clean_events_inputs(tmp_path):
    """A BIDS events file and matching epochs, with no AutoReject log written."""
    import mne
    import numpy as np

    bids_root = tmp_path / "bids"
    eeg_dir = bids_root / "sub-0001" / "eeg"
    eeg_dir.mkdir(parents=True)
    onsets = [1.0, 3.0, 5.0]
    lines = ["onset\tduration\ttrial_type\tstimulus_temp"]
    lines += [f"{o}\t0.001\tTrig_therm/T  1\t48.3" for o in onsets]
    (eeg_dir / "sub-0001_task-thermalactive_run-1_events.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    sfreq = 100.0
    info = mne.create_info(["Cz", "Pz"], sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((2, int(sfreq * 10))), info, verbose="ERROR")
    events = np.array([[int(o * sfreq), 0, 1] for o in onsets])
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"Trig_therm/T  1": 1},
        tmin=-0.2,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    deriv = tmp_path / "deriv"
    deriv.mkdir()
    epochs_path = deriv / "sub-0001_task-thermalactive_epo.fif"
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")
    return bids_root, epochs_path


def test_pre_rejection_events_do_not_require_an_autoreject_log(tmp_path) -> None:
    """AutoReject is fitted in the rejection step, so before it runs there is no per-trial
    repair record to attach. The provisional band-ICA comparisons write their events table
    at ICA-fitting time and must not be asked for one."""
    from eeg_pipeline.utils.data.preprocessing import write_clean_events_tsv_for_epochs

    bids_root, epochs_path = _minimal_clean_events_inputs(tmp_path)
    config = {
        "preprocessing": {
            "autoreject_log": True,
            "clean_events_qc": {"enabled": False},
        }
    }

    out = write_clean_events_tsv_for_epochs(
        subject="0001",
        task="thermalactive",
        bids_root=bids_root,
        epochs_path=epochs_path,
        config=config,
        conditions=["Trig_therm/T  1"],
        after_rejection=False,
    )

    written = pd.read_csv(out, sep="\t")
    assert len(written) == 3
    assert not [c for c in written.columns if "autoreject" in c.lower()]


def test_post_rejection_events_still_demand_the_autoreject_log(tmp_path) -> None:
    """The default must keep failing loudly: that log is the only record of which
    channel-in-trial samples are spline estimates rather than measurements."""
    from eeg_pipeline.utils.data.preprocessing import write_clean_events_tsv_for_epochs

    bids_root, epochs_path = _minimal_clean_events_inputs(tmp_path)
    config = {
        "preprocessing": {
            "autoreject_log": True,
            "clean_events_qc": {"enabled": False},
        }
    }

    with pytest.raises(FileNotFoundError, match="autoreject_log is enabled"):
        write_clean_events_tsv_for_epochs(
            subject="0001",
            task="thermalactive",
            bids_root=bids_root,
            epochs_path=epochs_path,
            config=config,
            conditions=["Trig_therm/T  1"],
        )
