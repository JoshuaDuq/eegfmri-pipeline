from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eeg_pipeline.plotting.features.source_localization import (
    _filter_stc_files_by_condition,
    _load_segment_fmri_cluster_rows,
    _load_volume_source_space,
    _resolve_source_views,
    _resolve_source_cluster_tfr_baseline_window,
    _resolve_fs_subject_name,
)


def test_resolve_fs_subject_name_prefers_existing_sub_prefixed_dir(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "freesurfer"
    (subjects_dir / "sub-0000").mkdir(parents=True)

    assert _resolve_fs_subject_name("0000", str(subjects_dir)) == "sub-0000"


def test_resolve_fs_subject_name_keeps_unprefixed_when_present(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "freesurfer"
    (subjects_dir / "0000").mkdir(parents=True)

    assert _resolve_fs_subject_name("0000", str(subjects_dir)) == "0000"


def test_load_volume_source_space_uses_subject_from_stc_name(
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

    src = _load_volume_source_space([stc_path])
    assert src == {"src": "ok"}
    assert captured["path"] == str(src_path)


def test_load_volume_source_space_supports_segmented_stc_name(
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

    src = _load_volume_source_space([stc_path])
    assert src == {"src": "ok"}
    assert captured["path"] == str(src_path)


def test_load_volume_source_space_raises_when_src_missing(tmp_path: Path) -> None:
    estimates_dir = tmp_path / "source_estimates"
    estimates_dir.mkdir(parents=True)
    stc_path = estimates_dir / "sub-0000_task-thermalactive_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_path.touch()

    with pytest.raises(FileNotFoundError, match="sub-0000_task-thermalactive_lcmv-src.fif"):
        _load_volume_source_space([stc_path])


def test_resolve_source_views_maps_long_and_short_names() -> None:
    assert _resolve_source_views(["lateral", "med", "dorsal"]) == ["lat", "med", "dor"]


def test_resolve_source_views_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Invalid source view"):
        _resolve_source_views(["lateral", "invalid_view"])


def test_filter_stc_files_by_condition_returns_matching_files(tmp_path: Path) -> None:
    estimates_dir = tmp_path / "source_estimates"
    estimates_dir.mkdir(parents=True)
    stc_a = estimates_dir / "sub-0000_task-thermalactive_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_b = estimates_dir / "sub-0000_task-thermalactive_cond-2.0_band-alpha_lcmv-vl.stc"
    stc_a.touch()
    stc_b.touch()

    filtered = _filter_stc_files_by_condition([stc_a, stc_b], "2.0")
    assert filtered == [stc_b]


def test_filter_stc_files_by_condition_raises_with_detected_conditions(tmp_path: Path) -> None:
    estimates_dir = tmp_path / "source_estimates"
    estimates_dir.mkdir(parents=True)
    stc_a = estimates_dir / "sub-0000_task-thermalactive_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_b = estimates_dir / "sub-0000_task-thermalactive_cond-2.0_band-alpha_lcmv-vl.stc"
    stc_a.touch()
    stc_b.touch()

    with pytest.raises(
        ValueError,
        match="requested source condition '3.0'.*Available conditions: 1.0, 2.0",
    ):
        _filter_stc_files_by_condition([stc_a, stc_b], "3.0")


def test_load_segment_fmri_cluster_rows_reads_segment_specific_metadata(
    tmp_path: Path,
) -> None:
    stc_dir = tmp_path / "features" / "sourcelocalization" / "lcmv" / "source_estimates" / "fmri_informed"
    stc_dir.mkdir(parents=True)
    stc_path = stc_dir / "sub-0000_task-thermalactive_seg-active_cond-1.0_band-alpha_lcmv-vl.stc"
    stc_path.touch()

    metadata_dir = stc_dir.parent.parent / "metadata"
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "fmri_constraint_active.json"
    metadata_path.write_text(
        json.dumps(
            {
                "segment": "active",
                "roi_survival": {
                    "fmri_cluster": {
                        "surviving_rois": ["c01_peak5p00", "c02_peak4p00"],
                        "surviving_stc_rows_per_roi": {
                            "c01_peak5p00": [0, 1],
                            "c02_peak4p00": [2],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cluster_rows = _load_segment_fmri_cluster_rows(
        stc_path,
        logging.getLogger("plot-src-cluster-rows"),
    )
    assert cluster_rows == {"c01_peak5p00": [0, 1], "c02_peak4p00": [2]}


def test_resolve_source_cluster_tfr_baseline_window_requires_visible_prestimulus() -> None:
    config = {"time_frequency_analysis": {"baseline_window": [-0.5, -0.1]}}
    baseline_window = _resolve_source_cluster_tfr_baseline_window(
        config,
        np.linspace(0.0, 1.0, 21, dtype=float),
        logging.getLogger("plot-src-tfr-baseline"),
    )
    assert baseline_window is None


def test_resolve_source_cluster_tfr_baseline_window_uses_valid_configured_window() -> None:
    config = {"time_frequency_analysis": {"baseline_window": [-0.5, -0.1]}}
    baseline_window = _resolve_source_cluster_tfr_baseline_window(
        config,
        np.linspace(-0.75, 1.0, 71, dtype=float),
        logging.getLogger("plot-src-tfr-baseline-valid"),
    )
    assert baseline_window == (-0.5, -0.1)
