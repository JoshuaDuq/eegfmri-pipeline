from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from eeg_pipeline.preprocessing.pipeline.preprocess import (
    run_bads_detection,
    run_bads_detection_single_file,
)


def test_bads_detection_surfaces_bids_read_errors(tmp_path: Path) -> None:
    eeg_path = tmp_path / "sub-0001_ses-01_task-pain_eeg.vhdr"
    eeg_path.write_text("", encoding="utf-8")
    channels_path = tmp_path / "sub-0001_ses-01_task-pain_channels.tsv"
    channels_path.write_text("name\ttype\tstatus\tdescription\nCz\tEEG\tgood\t\n", encoding="utf-8")
    channels_df = pd.DataFrame(
        {
            "name": ["Cz"],
            "type": ["EEG"],
            "status": ["good"],
            "description": [""],
        }
    )

    with patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
        return_value={"subject": "0001", "session": "01", "task": "pain"},
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
        return_value=str(channels_path),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
        return_value=channels_df,
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
        side_effect=RuntimeError("BIDS metadata broken"),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.read_raw",
        side_effect=AssertionError("raw-reader fallback should not be used"),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
        Mock(),
    ):
        with pytest.raises(RuntimeError, match="BIDS metadata broken"):
            run_bads_detection_single_file(
                str(eeg_path),
                bids_path=tmp_path,
            )


def test_bads_detection_surfaces_montage_application_errors(tmp_path: Path) -> None:
    class FakeRaw:
        def __init__(self) -> None:
            self.info = {"dig": None, "bads": []}

        def load_data(self) -> None:
            return None

        def get_montage(self) -> None:
            return None

        def set_montage(self, _montage: str) -> None:
            raise RuntimeError("montage missing")

    eeg_path = tmp_path / "sub-0001_ses-01_task-pain_eeg.vhdr"
    eeg_path.write_text("", encoding="utf-8")
    channels_path = tmp_path / "sub-0001_ses-01_task-pain_channels.tsv"
    channels_path.write_text("name\ttype\tstatus\tdescription\nCz\tEEG\tgood\t\n", encoding="utf-8")
    channels_df = pd.DataFrame(
        {
            "name": ["Cz"],
            "type": ["EEG"],
            "status": ["good"],
            "description": [""],
        }
    )

    with patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
        return_value={"subject": "0001", "session": "01", "task": "pain"},
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
        return_value=str(channels_path),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
        return_value=channels_df,
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
        return_value=FakeRaw(),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.BaseRaw",
        FakeRaw,
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
        side_effect=AssertionError("PyPREP should not be reached after montage failure"),
    ):
        with pytest.raises(RuntimeError, match="montage missing"):
            run_bads_detection_single_file(
                str(eeg_path),
                bids_path=tmp_path,
                l_pass=None,
            )


def test_bads_detection_uses_independent_pyprep_repeats_with_majority_vote(tmp_path: Path) -> None:
    class FakeRaw:
        def __init__(self) -> None:
            self.info = {"dig": "present", "bads": []}

        def load_data(self) -> None:
            return None

        def get_montage(self) -> str:
            return "existing"

    class FakeNoisyChannels:
        outputs = iter([["Cz"], [], []])

        def __init__(self, raw, random_state=None) -> None:
            random_states.append(random_state)
            raw_bad_snapshots.append(list(raw.info["bads"]))

        def find_bad_by_deviation(self) -> None:
            return None

        def find_bad_by_correlation(self) -> None:
            return None

        def get_bads(self) -> list[str]:
            return next(self.outputs)

    raw_bad_snapshots: list[list[str]] = []
    random_states: list[int | None] = []
    written: dict[str, pd.DataFrame] = {}
    eeg_path = tmp_path / "sub-0001_ses-01_task-pain_eeg.vhdr"
    eeg_path.write_text("", encoding="utf-8")
    channels_path = tmp_path / "sub-0001_ses-01_task-pain_channels.tsv"
    channels_path.write_text(
        "name\ttype\tstatus\tdescription\nCz\tEEG\tgood\t\nPz\tEEG\tgood\t\n",
        encoding="utf-8",
    )
    channels_df = pd.DataFrame(
        {
            "name": ["Cz", "Pz"],
            "type": ["EEG", "EEG"],
            "status": ["good", "good"],
            "description": ["", ""],
        }
    )

    def _capture_channels_tsv(frame: pd.DataFrame, path: str, index: bool = False) -> None:
        _ = index
        written[str(path)] = frame.copy()

    with patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
        return_value={"subject": "0001", "session": "01", "task": "pain"},
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
        return_value=str(channels_path),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
        return_value=channels_df,
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.io.write_channels_tsv",
        side_effect=_capture_channels_tsv,
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
        return_value=FakeRaw(),
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.BaseRaw",
        FakeRaw,
    ), patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
        FakeNoisyChannels,
    ):
        result = run_bads_detection_single_file(
            str(eeg_path),
            bids_path=tmp_path,
            l_pass=None,
            repeats=3,
            random_state=42,
        )

    assert raw_bad_snapshots == [[], [], []]
    assert random_states == [42, 43, 44]
    assert result.loc[str(eeg_path), "n_bads"] == 0
    assert written[str(channels_path)].loc[0, "status"] == "good"


def test_bads_detection_raises_when_no_eeg_files_match(tmp_path: Path) -> None:
    with patch(
        "eeg_pipeline.preprocessing.pipeline.preprocess.utils.find_bids_files",
        return_value=[],
    ):
        with pytest.raises(ValueError, match="No EEG files found for bad-channel detection"):
            run_bads_detection(
                bids_path=tmp_path,
                pipeline_path=tmp_path / "derivatives",
                task="pain",
                session="01",
                subjects=["0001"],
            )


def test_preprocessing_stats_rejects_missing_bad_channel_provenance(tmp_path: Path) -> None:
    from eeg_pipeline.preprocessing.pipeline import stats

    epo_path = tmp_path / "sub-0001_ses-01_task-pain_proc-clean_epo.fif"

    class FakeBIDSPath:
        def __init__(self, **_kwargs) -> None:
            pass

        def match(self) -> list[Path]:
            return [epo_path]

    class FakeEpochs:
        event_id = {"pain": 1}
        drop_log: list[tuple[str, ...]] = []

        def __len__(self) -> int:
            return 0

        def __getitem__(self, _key: str) -> "FakeEpochs":
            return self

    with patch.object(stats, "BIDSPath", FakeBIDSPath), patch.object(
        stats,
        "get_entities_from_fname",
        return_value={"subject": "0001", "session": "01"},
    ), patch.object(
        stats.utils,
        "get_derived_path",
        return_value=str(tmp_path / "missing_bads.tsv"),
    ), patch(
        "glob.glob",
        return_value=[],
    ), patch.object(
        stats.io,
        "read_components_tsv",
        return_value=None,
    ), patch.object(
        stats.io,
        "load_epochs",
        return_value=FakeEpochs(),
    ):
        with pytest.raises(FileNotFoundError, match="Missing bad-channel provenance"):
            stats.collect_preprocessing_stats(
                bids_path=tmp_path,
                pipeline_path=tmp_path,
                task="pain",
            )
