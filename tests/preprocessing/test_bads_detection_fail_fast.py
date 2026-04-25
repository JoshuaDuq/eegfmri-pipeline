from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from eeg_pipeline.preprocessing.pipeline.preprocess import run_bads_detection_single_file


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
            _ = random_state
            raw_bad_snapshots.append(list(raw.info["bads"]))

        def find_bad_by_deviation(self) -> None:
            return None

        def find_bad_by_correlation(self) -> None:
            return None

        def get_bads(self) -> list[str]:
            return next(self.outputs)

    raw_bad_snapshots: list[list[str]] = []
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
        )

    assert raw_bad_snapshots == [[], [], []]
    assert result.loc[str(eeg_path), "n_bads"] == 0
    assert written[str(channels_path)].loc[0, "status"] == "good"
