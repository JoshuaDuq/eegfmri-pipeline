"""Re-running detection must be able to un-mark a channel, not only mark one.

``read_raw_bids`` seeds ``info['bads']`` from channels.tsv, and ``NoisyChannels`` drops
``info['bads']`` before it measures anything. Feeding a previous run's own output back in
therefore exempts those channels from detection permanently: the bad-channel set becomes a
function of how many times the step has been run rather than of the recording. These tests
pin the split between marks this pipeline wrote (re-measured) and marks it did not
(carried forward).
"""

from __future__ import annotations

import pandas as pd
import pytest

from eeg_pipeline.preprocessing.pipeline.preprocess import (
    CUSTOM_BAD_DESCRIPTION,
    PYPREP_BAD_DESCRIPTION,
    SYNCHRONIZED_BAD_DESCRIPTION,
    _split_previous_bads,
    synchronize_bad_channels_across_runs,
)


def _channels(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["name", "type", "status", "description"])


# --------------------------------------------------------------------------------------
# Splitting curated marks from this pipeline's own output
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [PYPREP_BAD_DESCRIPTION, CUSTOM_BAD_DESCRIPTION, SYNCHRONIZED_BAD_DESCRIPTION],
)
def test_marks_this_pipeline_wrote_are_re_derivable(description) -> None:
    frame = _channels([("Cz", "EEG", "bad", description)])

    curated, derived = _split_previous_bads(frame)

    assert curated == []
    assert derived == ["Cz"]


def test_a_hand_marked_channel_is_curated_and_carried_forward() -> None:
    frame = _channels([("Cz", "EEG", "bad", "Broken during setup, see lab notes")])

    curated, derived = _split_previous_bads(frame)

    assert curated == ["Cz"]
    assert derived == []


def test_a_bad_channel_with_no_description_is_treated_as_curated() -> None:
    """The conservative direction: an unattributable mark is kept rather than
    re-derived, so a hand-marked channel is never silently dropped."""
    frame = _channels([("Cz", "EEG", "bad", "")])

    curated, derived = _split_previous_bads(frame)

    assert curated == ["Cz"]
    assert derived == []


def test_non_eeg_and_good_rows_are_not_split_into_either_list() -> None:
    frame = _channels(
        [
            ("ECG", "ECG", "bad", "Detached lead"),
            ("Pz", "EEG", "good", ""),
            ("Fz", "EEG", "bad", PYPREP_BAD_DESCRIPTION),
        ]
    )

    curated, derived = _split_previous_bads(frame)

    assert curated == []
    assert derived == ["Fz"]


def test_a_frame_without_a_description_column_yields_no_derived_marks() -> None:
    frame = pd.DataFrame(
        [("Cz", "EEG", "bad"), ("Pz", "EEG", "good")],
        columns=["name", "type", "status"],
    )

    curated, derived = _split_previous_bads(frame)

    assert curated == ["Cz"]
    assert derived == []


# --------------------------------------------------------------------------------------
# Synchronization must attribute the marks it propagates
# --------------------------------------------------------------------------------------


def _write_run(root, subject: str, run: int, rows) -> None:
    directory = root / f"sub-{subject}" / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sub-{subject}_task-thermalactive_run-{run}_channels.tsv"
    _channels(rows).to_csv(path, sep="\t", index=False)


def _read_run(root, subject: str, run: int) -> pd.DataFrame:
    path = (
        root
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-thermalactive_run-{run}_channels.tsv"
    )
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def test_a_propagated_mark_is_attributed_so_the_next_pass_re_derives_it(tmp_path) -> None:
    """Without a description, run 2's propagated mark reads as curated on the next pass,
    is exempted from detection, and the union becomes permanent."""
    _write_run(tmp_path, "0001", 1, [("Cz", "EEG", "bad", PYPREP_BAD_DESCRIPTION)])
    _write_run(tmp_path, "0001", 2, [("Cz", "EEG", "good", "")])

    synchronize_bad_channels_across_runs(str(tmp_path), "thermalactive", subjects=["0001"])

    run2 = _read_run(tmp_path, "0001", 2)
    assert run2.loc[0, "status"] == "bad"
    assert run2.loc[0, "description"] == SYNCHRONIZED_BAD_DESCRIPTION

    curated, derived = _split_previous_bads(run2)
    assert curated == []
    assert derived == ["Cz"]


def test_synchronization_preserves_the_reason_a_run_marked_a_channel_itself(tmp_path) -> None:
    _write_run(tmp_path, "0001", 1, [("Cz", "EEG", "bad", "Broken during setup")])
    _write_run(tmp_path, "0001", 2, [("Cz", "EEG", "good", "")])

    synchronize_bad_channels_across_runs(str(tmp_path), "thermalactive", subjects=["0001"])

    assert _read_run(tmp_path, "0001", 1).loc[0, "description"] == "Broken during setup"


def test_synchronization_clears_the_description_of_a_channel_it_un_marks(tmp_path) -> None:
    """A stale reason on a now-good channel would be read back as a curated mark the
    moment anything re-marked it."""
    _write_run(tmp_path, "0001", 1, [("Cz", "EEG", "good", "Broken during setup")])
    _write_run(tmp_path, "0001", 2, [("Cz", "EEG", "good", "")])

    synchronize_bad_channels_across_runs(str(tmp_path), "thermalactive", subjects=["0001"])

    assert _read_run(tmp_path, "0001", 1).loc[0, "description"] == ""


def test_synchronization_leaves_non_eeg_rows_alone(tmp_path) -> None:
    _write_run(
        tmp_path,
        "0001",
        1,
        [("ECG", "ECG", "bad", "Detached lead"), ("Cz", "EEG", "bad", PYPREP_BAD_DESCRIPTION)],
    )
    _write_run(tmp_path, "0001", 2, [("ECG", "ECG", "good", ""), ("Cz", "EEG", "good", "")])

    synchronize_bad_channels_across_runs(str(tmp_path), "thermalactive", subjects=["0001"])

    run1 = _read_run(tmp_path, "0001", 1)
    assert run1.loc[0, "status"] == "bad"
    assert run1.loc[0, "description"] == "Detached lead"
