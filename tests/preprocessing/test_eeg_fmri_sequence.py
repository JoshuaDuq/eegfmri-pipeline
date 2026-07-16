from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.sequence import load_multiband_slice_schedule


def _write_bold_metadata(path: Path, slice_timing: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "RepetitionTime": 0.9,
                "SliceTiming": slice_timing,
                "MultibandAccelerationFactor": 3,
            }
        ),
        encoding="utf-8",
    )


def test_load_multiband_slice_schedule_collapses_simultaneous_slices(tmp_path: Path) -> None:
    metadata_path = tmp_path / "sub-0001_task-test_run-01_bold.json"
    group_times = [0.0, 0.05, 0.0975, 0.1475]
    _write_bold_metadata(metadata_path, group_times * 3)

    schedule = load_multiband_slice_schedule(metadata_path)

    assert schedule.repetition_time_seconds == 0.9
    assert schedule.multiband_factor == 3
    assert schedule.slice_count == 12
    np.testing.assert_allclose(schedule.group_times_seconds, group_times)
    np.testing.assert_array_equal(
        schedule.group_boundaries_samples(5_000.0),
        np.array([0, 250, 488, 738, 4_500]),
    )


def test_load_multiband_slice_schedule_rejects_inconsistent_group_multiplicity(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "sub-0001_task-test_run-01_bold.json"
    _write_bold_metadata(metadata_path, [0.0, 0.0, 0.0, 0.05, 0.05])

    with pytest.raises(ValueError, match="exactly 3 slices"):
        load_multiband_slice_schedule(metadata_path)


def test_load_multiband_slice_schedule_requires_first_group_at_volume_onset(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "sub-0001_task-test_run-01_bold.json"
    _write_bold_metadata(metadata_path, [0.01, 0.01, 0.01, 0.05, 0.05, 0.05])

    with pytest.raises(ValueError, match="volume onset"):
        load_multiband_slice_schedule(metadata_path)
