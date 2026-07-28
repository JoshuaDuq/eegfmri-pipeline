from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.pipeline.preprocess import (
    _find_bad_channels_by_ransac,
    run_bads_detection,
    run_bads_detection_single_file,
)


class _FakeRansacDetector:
    def __init__(
        self,
        correlations: np.ndarray,
        warning_messages: tuple[str, ...] = (),
    ) -> None:
        self._correlations = correlations
        self._warning_messages = warning_messages
        self._extra_info: dict[str, object] = {}

    def find_bad_by_ransac(self) -> None:
        for message in self._warning_messages:
            warnings.warn_explicit(
                message,
                RuntimeWarning,
                filename="pyprep/ransac.py",
                lineno=398,
                module="pyprep.ransac",
            )
        self._extra_info["bad_by_ransac"] = {
            "ransac_correlations": self._correlations,
        }


def test_ransac_warning_boundary_suppresses_verified_matmul_warnings() -> None:
    detector = _FakeRansacDetector(
        correlations=np.ones((3, 2)),
        warning_messages=(
            "divide by zero encountered in matmul",
            "overflow encountered in matmul",
            "invalid value encountered in matmul",
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _find_bad_channels_by_ransac(detector)

    assert caught == []


def test_ransac_warning_boundary_preserves_unrelated_runtime_warnings() -> None:
    detector = _FakeRansacDetector(
        correlations=np.ones((3, 2)),
        warning_messages=("unexpected numerical condition",),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _find_bad_channels_by_ransac(detector)

    assert [str(item.message) for item in caught] == ["unexpected numerical condition"]


def test_ransac_warning_boundary_rejects_nonfinite_correlations() -> None:
    detector = _FakeRansacDetector(
        correlations=np.array([[1.0, np.nan]]),
    )

    with pytest.raises(FloatingPointError, match="non-finite RANSAC correlations"):
        _find_bad_channels_by_ransac(detector)


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

    with (
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
            return_value={"subject": "0001", "session": "01", "task": "pain"},
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
            return_value=str(channels_path),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
            return_value=channels_df,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
            side_effect=RuntimeError("BIDS metadata broken"),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.read_raw",
            side_effect=AssertionError("raw-reader fallback should not be used"),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
            Mock(),
        ),
    ):
        with pytest.raises(RuntimeError, match="BIDS metadata broken"):
            run_bads_detection_single_file(
                str(eeg_path),
                bids_path=tmp_path,
            )


def test_bads_detection_preserves_run_entity_when_reading_bids(tmp_path: Path) -> None:
    eeg_path = tmp_path / "sub-0001_ses-01_task-pain_run-1_eeg.vhdr"
    eeg_path.write_text("", encoding="utf-8")
    channels_path = tmp_path / "sub-0001_ses-01_task-pain_run-1_channels.tsv"
    channels_path.write_text("name\ttype\tstatus\tdescription\nCz\tEEG\tgood\t\n", encoding="utf-8")
    channels_df = pd.DataFrame(
        {
            "name": ["Cz"],
            "type": ["EEG"],
            "status": ["good"],
            "description": [""],
        }
    )
    captured_run = None

    def _capture_bids_path(bids_path, verbose=False):
        nonlocal captured_run
        _ = verbose
        captured_run = bids_path.run
        raise RuntimeError("captured BIDSPath")

    with (
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
            return_value=str(channels_path),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
            return_value=channels_df,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
            side_effect=_capture_bids_path,
        ),
    ):
        with pytest.raises(RuntimeError, match="captured BIDSPath"):
            run_bads_detection_single_file(
                str(eeg_path),
                bids_path=tmp_path,
            )

    assert str(captured_run) == "1"


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

    with (
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
            return_value={"subject": "0001", "session": "01", "task": "pain"},
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
            return_value=str(channels_path),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
            return_value=channels_df,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
            return_value=FakeRaw(),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.BaseRaw",
            FakeRaw,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
            side_effect=AssertionError("PyPREP should not be reached after montage failure"),
        ),
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

        def find_bad_by_nan_flat(self) -> None:
            return None

        def find_bad_by_deviation(self) -> None:
            return None

        def find_bad_by_hfnoise(self) -> None:
            return None

        def find_bad_by_correlation(self) -> None:
            return None

        def find_bad_by_ransac(self) -> None:
            self._extra_info = {"bad_by_ransac": {"ransac_correlations": [[1.0, 1.0], [1.0, 1.0]]}}

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

    with (
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
            return_value={"subject": "0001", "session": "01", "task": "pain"},
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
            return_value=str(channels_path),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
            return_value=channels_df,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.write_channels_tsv",
            side_effect=_capture_channels_tsv,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
            return_value=FakeRaw(),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.BaseRaw",
            FakeRaw,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
            FakeNoisyChannels,
        ),
    ):
        result = run_bads_detection_single_file(
            str(eeg_path),
            bids_path=tmp_path,
            l_pass=None,
            repeats=3,
            # Only RANSAC consumes the random state, so repeats are meaningful only here.
            ransac=True,
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

    pipeline_path = tmp_path / "preprocessed" / "eeg"
    epo_path = pipeline_path / "sub-0001" / "eeg" / "sub-0001_task-pain_proc-clean_epo.fif"
    epo_path.parent.mkdir(parents=True)
    epo_path.write_text("", encoding="utf-8")

    with (
        patch.object(
            stats.utils,
            "get_derived_path",
            return_value=str(tmp_path / "missing_bads.tsv"),
        ),
        patch(
            "glob.glob",
            return_value=[],
        ),
        patch.object(
            stats.io,
            "read_components_tsv",
            return_value=None,
        ),
    ):
        with pytest.raises(FileNotFoundError, match="Missing bad-channel provenance"):
            stats.collect_preprocessing_stats(
                bids_path=tmp_path,
                pipeline_path=pipeline_path,
                task="pain",
            )


def test_preprocessing_stats_ignores_archived_epoch_files(tmp_path: Path) -> None:
    from eeg_pipeline.preprocessing.pipeline import stats

    pipeline_path = tmp_path / "preprocessed" / "eeg"
    active_path = pipeline_path / "sub-0001" / "eeg" / "sub-0001_task-pain_proc-clean_epo.fif"
    archived_path = (
        pipeline_path
        / "sub-0001"
        / "eeg"
        / "stale_before_restart_trigger_fix_20260516"
        / active_path.name
    )
    bads_path = active_path.with_name("sub-0001_task-pain_bads.tsv")
    components_path = active_path.with_name("sub-0001_task-pain_proc-ica_components.tsv")

    active_path.parent.mkdir(parents=True)
    archived_path.parent.mkdir(parents=True)
    active_path.write_text("", encoding="utf-8")
    archived_path.write_text("", encoding="utf-8")
    bads_path.write_text("name\nCz\n", encoding="utf-8")
    components_path.write_text("status\nbad\ngood\n", encoding="utf-8")

    loaded_epoch_paths = []

    class FakeEpochs:
        event_id = {"pain": 1}
        drop_log = [(), ("BAD boundary",)]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _key: str) -> "FakeEpochs":
            return self

    def load_epochs(path: str) -> FakeEpochs:
        loaded_epoch_paths.append(Path(path))
        return FakeEpochs()

    with (
        patch.object(stats.io, "read_channels_tsv", return_value=pd.DataFrame({"name": ["Cz"]})),
        patch.object(
            stats.io,
            "read_components_tsv",
            return_value=pd.DataFrame({"status": ["bad", "good"]}),
        ),
        patch.object(stats.io, "load_epochs", side_effect=load_epochs),
    ):
        stats.collect_preprocessing_stats(
            bids_path=tmp_path,
            pipeline_path=pipeline_path,
            task="pain",
        )

    assert loaded_epoch_paths == [active_path]


def test_clean_recording_logs_no_bad_channels_rather_than_an_empty_name(
    tmp_path: Path,
) -> None:
    """A recording with nothing wrong must log an empty list, not a nameless channel."""

    class FakeRaw:
        ch_names = ["Cz", "Pz"]

        def __init__(self) -> None:
            self.info = {"dig": "present", "bads": []}

        def load_data(self) -> None:
            return None

        def get_montage(self) -> str:
            return "existing"

    class FakeNoisyChannels:
        def __init__(self, raw, random_state=None) -> None:
            return None

        def find_bad_by_nan_flat(self) -> None:
            return None

        def find_bad_by_deviation(self) -> None:
            return None

        def find_bad_by_hfnoise(self) -> None:
            return None

        def find_bad_by_correlation(self) -> None:
            return None

        def get_bads(self) -> list[str]:
            return []

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

    with (
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.get_entities_from_fname",
            return_value={"subject": "0001", "session": "01", "task": "pain"},
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.utils.get_channels_path_from_eeg_file",
            return_value=str(channels_path),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.read_channels_tsv",
            return_value=channels_df,
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.io.write_channels_tsv",
            side_effect=lambda frame, path, index=False: written.update({str(path): frame.copy()}),
        ),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.read_raw_bids",
            return_value=FakeRaw(),
        ),
        patch("eeg_pipeline.preprocessing.pipeline.preprocess.mne.io.BaseRaw", FakeRaw),
        patch(
            "eeg_pipeline.preprocessing.pipeline.preprocess.pyprep.NoisyChannels",
            FakeNoisyChannels,
        ),
    ):
        result = run_bads_detection_single_file(
            str(eeg_path),
            bids_path=tmp_path,
            l_pass=None,
            repeats=1,
        )

    assert result.loc[str(eeg_path), "n_bads"] == 0
    assert result.loc[str(eeg_path), "bad_channels"] == []
    assert written[str(channels_path)]["status"].tolist() == ["good", "good"]


def test_run_sync_keeps_non_eeg_channel_status(tmp_path: Path) -> None:
    """Synchronizing EEG bads must not clear a hand-marked bad ECG or EOG channel."""
    from eeg_pipeline.preprocessing.pipeline.preprocess import (
        synchronize_bad_channels_across_runs,
    )

    eeg_dir = tmp_path / "sub-0001" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    header = "name\ttype\tstatus\n"
    run_1 = eeg_dir / "sub-0001_ses-01_task-pain_run-1_channels.tsv"
    run_2 = eeg_dir / "sub-0001_ses-01_task-pain_run-2_channels.tsv"
    run_1.write_text(header + "C3\teeg\tbad\nC4\teeg\tgood\nECG\tecg\tbad\n", encoding="utf-8")
    run_2.write_text(header + "C3\teeg\tgood\nC4\teeg\tbad\nECG\tecg\tbad\n", encoding="utf-8")

    synchronize_bad_channels_across_runs(str(tmp_path), "pain", subjects=["0001"])

    for path in (run_1, run_2):
        frame = pd.read_csv(path, sep="\t")
        eeg_status = dict(zip(frame["name"], frame["status"]))
        # Union of the EEG bads across runs, and the ECG row left exactly as it was.
        assert eeg_status["C3"] == "bad"
        assert eeg_status["C4"] == "bad"
        assert eeg_status["ECG"] == "bad"
