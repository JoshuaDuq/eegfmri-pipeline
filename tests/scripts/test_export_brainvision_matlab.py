from __future__ import annotations

from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.scripts.conversion import export_brainvision_matlab

MODULE_PATH = (
    Path(__file__).parents[2]
    / "studies"
    / "pain_study"
    / "scripts"
    / "conversion"
    / "export_brainvision_matlab.py"
)


def test_export_module_exposes_main() -> None:
    specification = util.spec_from_file_location("export_brainvision_matlab", MODULE_PATH)

    assert specification is not None
    assert specification.loader is not None
    module = util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert callable(getattr(module, "main", None))


def _write_processed_triplet(directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / f"{stem}.vhdr"
    header.write_text("Brain Vision Data Exchange Header File Version 2.0\n", encoding="utf-8")
    header.with_suffix(".vmrk").write_text("marker\n", encoding="utf-8")
    header.with_suffix(".eeg").write_bytes(b"\x00\x00")
    return header


def test_require_run_header_reads_the_gap_recovery_export_suffix(tmp_path: Path) -> None:
    """The current export ends at `scanner_artifact_step2`, not `scannerpulse_corrected`."""
    header = _write_processed_triplet(
        tmp_path,
        "ThermalPainEEGFMRI_run4_sub0015_2026-07-13_11h18.48.310_scanner_artifact_step2",
    )

    assert export_brainvision_matlab._require_run_header(tmp_path, 4) == header


def test_require_run_header_rejects_two_export_generations_of_one_run(tmp_path: Path) -> None:
    """Holding both generations is an ambiguity, not a preference order."""
    stem = "ThermalPainEEGFMRI_run4_sub0015_2026-07-13_11h18.48.310"
    _write_processed_triplet(tmp_path, f"{stem}_scannerpulse_corrected")
    _write_processed_triplet(tmp_path, f"{stem}_scanner_artifact_step2")

    with pytest.raises(FileNotFoundError, match="found 2"):
        export_brainvision_matlab._require_run_header(tmp_path, 4)


def test_select_trials_requires_eleven_ordered_thermal_trials() -> None:
    events = pd.DataFrame(
        {
            "onset": np.arange(11, dtype=float),
            "sample": np.arange(11, dtype=int) * 1_000,
            "trial_type": ["Trig_therm/T  1"] * 11,
            "run_id": [2] * 11,
            "trial_number": np.arange(1, 12),
            "stimulus_temp": np.linspace(44.3, 49.3, 11),
            "selected_surface": np.ones(11),
            "pain_binary_coded": np.ones(11),
            "vas_final_coded_rating": np.linspace(0, 100, 11),
        }
    )

    trials = export_brainvision_matlab.select_trials(events, run_id=2)

    assert trials["trial_number"].tolist() == list(range(1, 12))
    assert trials["run_id"].tolist() == [2] * 11


def test_select_trials_rejects_incomplete_run() -> None:
    events = pd.DataFrame(
        {
            "trial_type": ["Trig_therm/T  1"],
            "run_id": [1],
            "trial_number": [1],
            "stimulus_temp": [49.3],
            "selected_surface": [1],
            "pain_binary_coded": [1],
            "vas_final_coded_rating": [100],
        }
    )

    with pytest.raises(ValueError, match="11 thermal trials"):
        export_brainvision_matlab.select_trials(events, run_id=1)


def test_epoch_bounds_include_minus_seven_through_plus_fifteen_seconds() -> None:
    starts, stops = export_brainvision_matlab.epoch_bounds(
        np.array([10_000, 40_000]),
        sampling_frequency=1_000.0,
        recording_samples=60_000,
    )

    np.testing.assert_array_equal(starts, [3_000, 33_000])
    np.testing.assert_array_equal(stops, [25_001, 55_001])
    np.testing.assert_array_equal(stops - starts, [22_001, 22_001])


def test_align_trial_onsets_matches_source_recording_clock() -> None:
    aligned = export_brainvision_matlab.align_trial_onsets(
        bids_onsets=np.array([40.622, 85.356]),
        annotation_onsets=np.array([40.622, 85.356]),
    )

    np.testing.assert_allclose(aligned, [40.622, 85.356], atol=1e-12)


def test_relative_volume_times_start_at_zero() -> None:
    relative_times = export_brainvision_matlab.relative_volume_times(
        np.array([19.407, 20.307, 21.207, 22.107])
    )

    np.testing.assert_allclose(relative_times, [0.0, 0.9, 1.8, 2.7], atol=1e-12)


def test_numeric_trialinfo_contains_only_analysis_columns() -> None:
    metadata = pd.DataFrame(
        {
            "run_id": [1.0],
            "trial_number": [7.0],
            "stimulus_temp": [44.3],
            "selected_surface": [1.0],
            "pain_binary_coded": [0.0],
            "vas_final_coded_rating": [48.62],
            "stim_start_time": [325.1969],
            "trial_onset_relative_to_first_volume_s": [288.754],
            "trial_type": ["Trig_therm/T  1"],
            "source_event_file": ["events.tsv"],
        }
    )

    values, labels = export_brainvision_matlab.numeric_trialinfo(metadata)

    assert values.shape == (1, 7)
    assert labels == (
        "run_id",
        "trial_number",
        "stimulus_temp",
        "selected_surface",
        "pain_binary_coded",
        "vas_final_coded_rating",
        "trial_onset_relative_to_first_volume_s",
    )


def test_match_trial_samples_rejects_ambiguous_vas_marker() -> None:
    info = export_brainvision_matlab.mne.create_info(["Cz"], 1_000.0, ["eeg"])
    raw = export_brainvision_matlab.mne.io.RawArray(
        np.zeros((1, 100_000)),
        info,
        verbose=False,
    )
    raw.set_annotations(
        export_brainvision_matlab.mne.Annotations(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            ["Volume/V  1", "Vas_on/V  1", "Trig_therm/T  1"],
        )
    )
    trials = pd.DataFrame({"onset": [2.0]})

    with pytest.raises(ValueError, match="Vas_on/V  1"):
        export_brainvision_matlab._match_trial_samples(raw, trials)
