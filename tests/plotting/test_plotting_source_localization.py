from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eeg_pipeline.plotting.features.source_localization import (
    _resolve_fs_subject_name,
    _resolve_volume_source_space,
)


def test_resolve_fs_subject_name_prefers_existing_sub_prefixed_dir(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "freesurfer"
    (subjects_dir / "sub-0000").mkdir(parents=True)

    assert _resolve_fs_subject_name("0000", str(subjects_dir)) == "sub-0000"


def test_resolve_fs_subject_name_keeps_unprefixed_when_present(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "freesurfer"
    (subjects_dir / "0000").mkdir(parents=True)

    assert _resolve_fs_subject_name("0000", str(subjects_dir)) == "0000"


def test_resolve_volume_source_space_uses_subject_from_stc_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimates_dir = tmp_path / "source_estimates"
    estimates_dir.mkdir(parents=True)
    stc_path = estimates_dir / "sub-0000_task-thermalactive_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_path.touch()

    src_path = estimates_dir / "sub-0000_task-thermalactive_lcmv-src.fif"
    src_path.touch()

    captured: dict[str, Any] = {}

    def fake_read_source_spaces(path: str) -> object:
        captured["path"] = path
        return {"src": "ok"}

    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.source_localization.mne.read_source_spaces",
        fake_read_source_spaces,
    )

    src = _resolve_volume_source_space([stc_path])
    assert src == {"src": "ok"}
    assert captured["path"] == str(src_path)


def test_resolve_volume_source_space_supports_segmented_stc_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimates_dir = tmp_path / "source_estimates"
    estimates_dir.mkdir(parents=True)
    stc_path = estimates_dir / "sub-0000_task-thermalactive_seg-ramp_down_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_path.touch()

    src_path = estimates_dir / "sub-0000_task-thermalactive_lcmv-src.fif"
    src_path.touch()

    captured: dict[str, Any] = {}

    def fake_read_source_spaces(path: str) -> object:
        captured["path"] = path
        return {"src": "ok"}

    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.source_localization.mne.read_source_spaces",
        fake_read_source_spaces,
    )

    src = _resolve_volume_source_space([stc_path])
    assert src == {"src": "ok"}
    assert captured["path"] == str(src_path)


def test_resolve_volume_source_space_raises_when_src_missing(tmp_path: Path) -> None:
    estimates_dir = tmp_path / "source_estimates"
    estimates_dir.mkdir(parents=True)
    stc_path = estimates_dir / "sub-0000_task-thermalactive_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_path.touch()

    with pytest.raises(FileNotFoundError, match="sub-0000_task-thermalactive_lcmv-src.fif"):
        _resolve_volume_source_space([stc_path])
