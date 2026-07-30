from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
import pybv
import pytest

from studies.pain_study.scripts.sanitize_brainvision_vas_markers import (
    _remove_appledouble_files,
    discover_cohort_recordings,
    run_sanitization,
    stage_recording,
)


def _write_raw_brainvision_recording(
    raw_dir: Path,
    basename: str,
    *,
    sampling_frequency: float = 5_000.0,
) -> Path:
    ch_names = [f"EEG{index:02d}" for index in range(1, 64)] + ["ECG"]
    data = np.arange(64 * 20, dtype=float).reshape(64, 20) * 1e-9
    pybv.write_brainvision(
        data=data,
        sfreq=sampling_frequency,
        ch_names=ch_names,
        fname_base=basename,
        folder_out=raw_dir,
        fmt="binary_float32",
    )
    marker_path = raw_dir / f"{basename}.vmrk"
    marker_text = marker_path.read_text(encoding="utf-8")
    marker_text += "Mk2=Volume,V  1,1,1,0\n" "Mk3=Vas_on,V  1,5,1,0\n" "Mk4=Volume,V  1,10,1,0\n"
    marker_path.write_text(marker_text, encoding="utf-8")
    return raw_dir / f"{basename}.vhdr"


def test_discover_cohort_recordings_uses_all_original_untrimmed_5khz_recordings(
    tmp_path: Path,
) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = kingston_root / "EEG_fMRI_data" / "source_data"
    raw_dir = source_data_root / "sub-0001" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    source_vhdr = _write_raw_brainvision_recording(raw_dir, basename)

    recordings = discover_cohort_recordings(source_data_root)

    assert len(recordings) == 1
    assert recordings[0].subject == "0001"
    assert recordings[0].run == 1
    assert recordings[0].source_vhdr == source_vhdr


def test_discover_cohort_recordings_includes_original_and_processed_layouts(
    tmp_path: Path,
) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    original_dir = tmp_path / "sub-0001" / "eeg" / "original_untrimmed_5khz"
    processed_dir = tmp_path / "sub-0001" / "eeg" / "brainvision_processed_1khz"
    original_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_raw_brainvision_recording(original_dir, basename)
    _write_raw_brainvision_recording(
        processed_dir,
        f"{basename}_scannerpulse_corrected",
        sampling_frequency=1_000.0,
    )

    recordings = discover_cohort_recordings(tmp_path)

    assert [(recording.source_layout, recording.run) for recording in recordings] == [
        ("brainvision_processed_1khz", 1),
        ("original_untrimmed_5khz", 1),
    ]


def test_discover_cohort_recordings_rejects_empty_inventory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No supported thermal EEG-fMRI recordings"):
        discover_cohort_recordings(tmp_path)


def test_discover_cohort_recordings_filters_explicit_subjects(tmp_path: Path) -> None:
    for subject in ("0001", "0016"):
        basename = f"ThermalPainEEGFMRI_run1_sub{subject}_2026-03-02_10h55.27.564"
        raw_dir = tmp_path / f"sub-{subject}" / "eeg" / "original_untrimmed_5khz"
        raw_dir.mkdir(parents=True)
        _write_raw_brainvision_recording(raw_dir, basename)

    recordings = discover_cohort_recordings(tmp_path, subjects=("0016",))

    assert [(recording.subject, recording.run) for recording in recordings] == [("0016", 1)]


def test_discover_cohort_recordings_rejects_subject_directory_mismatch(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0002_2026-03-02_10h55.27.564"
    raw_dir = tmp_path / "sub-0001" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    _write_raw_brainvision_recording(raw_dir, basename)

    with pytest.raises(ValueError, match="subject mismatch"):
        discover_cohort_recordings(tmp_path)


def test_discover_cohort_recordings_requires_selection_for_duplicate_run(tmp_path: Path) -> None:
    raw_dir = tmp_path / "sub-0003" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    for timestamp in ("11h11.33.962", "11h30.23.962"):
        _write_raw_brainvision_recording(
            raw_dir,
            f"ThermalPainEEGFMRI_run1_sub0003_2026-03-23_{timestamp}",
        )

    with pytest.raises(ValueError, match="Ambiguous original_untrimmed_5khz.*sub-0003 run-1"):
        discover_cohort_recordings(tmp_path)


def test_discover_cohort_recordings_applies_explicit_run_override(tmp_path: Path) -> None:
    raw_dir = tmp_path / "sub-0003" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    selected_name = "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h30.23.962.vhdr"
    for name in (
        "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h11.33.962.vhdr",
        selected_name,
    ):
        _write_raw_brainvision_recording(raw_dir, Path(name).stem)

    recordings = discover_cohort_recordings(
        tmp_path,
        recording_overrides={
            "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h11.33.962.vhdr": 1,
            selected_name: 3,
        },
    )

    assert [(recording.run, recording.source_vhdr.name) for recording in recordings] == [
        (
            1,
            "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h11.33.962.vhdr",
        ),
        (3, selected_name),
    ]


def test_discover_cohort_recordings_applies_explicit_exclusion(tmp_path: Path) -> None:
    raw_dir = tmp_path / "sub-0003" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    false_start = "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h10.39.899.vhdr"
    valid_run = "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h11.33.962.vhdr"
    for name in (false_start, valid_run):
        _write_raw_brainvision_recording(raw_dir, Path(name).stem)

    recordings = discover_cohort_recordings(
        tmp_path,
        recording_overrides={false_start: None},
    )

    assert [recording.source_vhdr.name for recording in recordings] == [valid_run]


def test_stage_recording_reuses_signal_and_changes_only_vas_annotation(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    source_data_root = tmp_path / "KINGSTON" / "EEG_fMRI_data" / "source_data"
    raw_dir = source_data_root / "sub-0001" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    source_vhdr = _write_raw_brainvision_recording(raw_dir, basename)
    recording = discover_cohort_recordings(source_data_root)[0]

    manifest_row = stage_recording(recording, source_data_root, tmp_path / "staged")

    staged_vhdr = Path(manifest_row["staged_vhdr"])
    staged_vmrk = staged_vhdr.with_suffix(".vmrk")
    assert not staged_vhdr.with_suffix(".eeg").exists()
    assert "Vas_on,VAS_ON,5,1,0" in staged_vmrk.read_text(encoding="utf-8")
    assert "Volume,V  1,1,1,0" in staged_vmrk.read_text(encoding="utf-8")

    source_raw = mne.io.read_raw_brainvision(source_vhdr, preload=False, verbose=False)
    staged_raw = mne.io.read_raw_brainvision(staged_vhdr, preload=False, verbose=False)
    assert staged_raw.n_times == source_raw.n_times
    assert staged_raw.ch_names == source_raw.ch_names
    assert list(source_raw.annotations.description) == [
        "Volume/V  1",
        "Vas_on/V  1",
        "Volume/V  1",
    ]
    assert list(staged_raw.annotations.description) == [
        "Volume/V  1",
        "Vas_on/VAS_ON",
        "Volume/V  1",
    ]
    np.testing.assert_array_equal(staged_raw[:, :][0], source_raw[:, :][0])
    assert manifest_row["volume_count"] == 2
    assert manifest_row["vas_count"] == 1


def test_stage_recording_preserves_processed_layout_and_accepts_1khz(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0015_2026-07-13_10h50.22.477" "_scannerpulse_corrected"
    source_data_root = tmp_path / "source_data"
    raw_dir = source_data_root / "sub-0015" / "eeg" / "brainvision_processed_1khz"
    raw_dir.mkdir(parents=True)
    source_vhdr = _write_raw_brainvision_recording(
        raw_dir,
        basename,
        sampling_frequency=1_000.0,
    )
    recording = discover_cohort_recordings(source_data_root)[0]

    manifest_row = stage_recording(recording, source_data_root, tmp_path / "staged")

    staged_vhdr = Path(manifest_row["staged_vhdr"])
    assert staged_vhdr.relative_to(tmp_path / "staged") == source_vhdr.relative_to(source_data_root)
    assert manifest_row["source_layout"] == "brainvision_processed_1khz"
    assert manifest_row["sampling_frequency_hz"] == 1_000.0


def test_run_sanitization_publishes_final_manifest_paths(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = kingston_root / "EEG_fMRI_data" / "source_data"
    raw_dir = source_data_root / "sub-0001" / "eeg" / "original_untrimmed_5khz"
    raw_dir.mkdir(parents=True)
    _write_raw_brainvision_recording(raw_dir, basename)
    output_root = tmp_path / "brainvision_marker_sanitized-v1"

    result = run_sanitization(source_data_root, output_root)

    with (result / "marker_sanitization_manifest.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert Path(row["staged_vhdr"]).is_file()
    assert str(Path(row["staged_vhdr"])).startswith(str(output_root))
    assert not (tmp_path / ".brainvision_marker_sanitized-v1.tmp").exists()


def test_remove_appledouble_files_keeps_scientific_outputs(tmp_path: Path) -> None:
    nested = tmp_path / "sub-0001" / "eeg"
    nested.mkdir(parents=True)
    scientific_output = nested / "recording.vmrk"
    resource_fork = nested / "._recording.vmrk"
    scientific_output.write_text("markers", encoding="utf-8")
    resource_fork.write_text("metadata", encoding="utf-8")

    _remove_appledouble_files(tmp_path)

    assert scientific_output.read_text(encoding="utf-8") == "markers"
    assert not resource_fork.exists()
