from __future__ import annotations

import argparse

import pytest

from fmri_pipeline.cli.commands.subject_selection import (
    resolve_subjects,
    subjects_from_bids_root,
)


class _Config:
    def __init__(self, subjects: list[str] | None = None) -> None:
        self.subjects = subjects or []


def test_subjects_from_bids_root_returns_all_when_requested_none(tmp_path) -> None:
    (tmp_path / "sub-0001").mkdir()
    (tmp_path / "sub-0002").mkdir()
    assert subjects_from_bids_root(tmp_path, None) == ["0001", "0002"]


def test_subjects_from_bids_root_raises_for_missing_subjects(tmp_path) -> None:
    (tmp_path / "sub-0001").mkdir()
    with pytest.raises(ValueError, match="not found in BIDS fMRI root"):
        subjects_from_bids_root(tmp_path, ["0001", "0002"])


def test_resolve_subjects_uses_group_argument(tmp_path) -> None:
    (tmp_path / "sub-0001").mkdir()
    (tmp_path / "sub-0002").mkdir()
    args = argparse.Namespace(group="0002;0001", all_subjects=False, subject=None, subjects=None)
    resolved = resolve_subjects(args, tmp_path, _Config())
    assert resolved == ["0002", "0001"]


def test_resolve_subjects_falls_back_to_config_subjects(tmp_path) -> None:
    (tmp_path / "sub-0001").mkdir()
    args = argparse.Namespace(group=None, all_subjects=False, subject=None, subjects=None)
    resolved = resolve_subjects(args, tmp_path, _Config(subjects=["0001"]))
    assert resolved == ["0001"]

