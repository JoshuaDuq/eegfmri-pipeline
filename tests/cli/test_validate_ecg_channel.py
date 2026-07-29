"""The declared ECG channel is checked against what the recording actually carries.

Issue #14 asks that BIDS channel typing confirm the declared ECG channel. Naming one is
how a study turns on the cardiac review and the ECG coupling metric, and the name is a
free-text string that has to match ``channels.tsv`` exactly. Until now a typo — or the
inherited name from another study's montage — surfaced only when cardiac QC opened the
recording and failed, which is after preprocessing has started.
"""

from __future__ import annotations

import pytest

from eeg_pipeline.cli.commands.validate_checks import _validate_ecg_channels


class _Config:
    """Minimal stand-in: these checks read two keys and nothing else."""

    def __init__(self, bids_root, ecg_channels):
        self.bids_root = str(bids_root)
        self._values = {"eeg.ecg_channels": list(ecg_channels)}

    def get(self, key, default=None):
        return self._values.get(key, default)


def _bids_tree(tmp_path, rows):
    channels = tmp_path / "sub-0001" / "eeg"
    channels.mkdir(parents=True)
    lines = ["name\ttype\tunits"]
    lines += [f"{name}\t{kind}\tuV" for name, kind in rows]
    (channels / "sub-0001_task-oddball_channels.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def collected():
    return {"issues": [], "warnings": [], "passed": []}


def _run(config, collected):
    _validate_ecg_channels(config, collected["issues"], collected["warnings"], collected["passed"])


def test_a_declared_ecg_channel_present_and_typed_ecg_is_confirmed(tmp_path, collected) -> None:
    root = _bids_tree(tmp_path, [("Cz", "EEG"), ("ECG", "ECG")])

    _run(_Config(root, ["ECG"]), collected)

    assert collected["issues"] == []
    assert collected["warnings"] == []
    assert any("ECG" in entry for entry in collected["passed"])


def test_a_declared_channel_the_recording_does_not_carry_is_an_issue(tmp_path, collected) -> None:
    """The inherited name from another study's montage is the likely case, so the report
    names what the recording does carry rather than only what is missing."""
    root = _bids_tree(tmp_path, [("Cz", "EEG"), ("EKG", "ECG")])

    _run(_Config(root, ["ECG"]), collected)

    assert len(collected["issues"]) == 1
    message = collected["issues"][0]["message"]
    assert "ECG" in message
    assert "EKG" in message


def test_a_channel_present_but_typed_something_else_is_a_warning(tmp_path, collected) -> None:
    """Present under the right name but typed EEG: it will be filtered as a scalp channel
    wherever type is what selects, so the mismatch is worth saying out loud."""
    root = _bids_tree(tmp_path, [("Cz", "EEG"), ("ECG", "EEG")])

    _run(_Config(root, ["ECG"]), collected)

    assert collected["issues"] == []
    assert len(collected["warnings"]) == 1
    assert "EEG" in collected["warnings"][0]["message"]


def test_declaring_no_ecg_channel_checks_nothing(tmp_path, collected) -> None:
    """Not having an ECG lead is an ordinary configuration, not something to report."""
    root = _bids_tree(tmp_path, [("Cz", "EEG")])

    _run(_Config(root, []), collected)

    assert collected == {"issues": [], "warnings": [], "passed": []}


def test_a_dataset_without_channel_tables_is_not_reported_here(tmp_path, collected) -> None:
    """Whether the dataset has its sidecars is a different check's question; this one has
    nothing to say when there is nothing to compare against."""
    (tmp_path / "sub-0001" / "eeg").mkdir(parents=True)

    _run(_Config(tmp_path, ["ECG"]), collected)

    assert collected["issues"] == []
    assert collected["passed"] == []
