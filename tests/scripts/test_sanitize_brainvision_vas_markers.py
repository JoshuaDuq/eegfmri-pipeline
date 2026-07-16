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


def _write_raw_brainvision_recording(raw_dir: Path, basename: str) -> Path:
    ch_names = [f"EEG{index:02d}" for index in range(1, 64)] + ["ECG"]
    data = np.arange(64 * 20, dtype=float).reshape(64, 20) * 1e-9
    pybv.write_brainvision(
        data=data,
        sfreq=5_000,
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


def _write_corrected_reference(source_data_root: Path, subject: str, basename: str) -> None:
    reference_dir = source_data_root / f"sub-{subject}" / "eeg" / "brainvision_processed_1khz"
    reference_dir.mkdir(parents=True)
    reference_name = f"{basename}_scannerpulse_corrected.vhdr"
    (reference_dir / reference_name).write_text("reference only", encoding="utf-8")


def test_discover_cohort_recordings_maps_reference_to_one_original(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = kingston_root / "EEG_fMRI_data" / "source_data"
    raw_dir = source_data_root / "sub-0001" / "eeg" / "original_5khz"
    raw_dir.mkdir(parents=True)
    source_vhdr = _write_raw_brainvision_recording(raw_dir, basename)
    _write_corrected_reference(source_data_root, "0001", basename)

    recordings = discover_cohort_recordings(
        source_data_root,
        subjects=("0001",),
        expected_count=1,
    )

    assert len(recordings) == 1
    assert recordings[0].subject == "0001"
    assert recordings[0].run == 1
    assert recordings[0].source_vhdr == source_vhdr


def test_discover_cohort_recordings_requires_original_in_labeled_directory(
    tmp_path: Path,
) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = kingston_root / "EEG_fMRI_data" / "source_data"
    _write_corrected_reference(source_data_root, "0001", basename)

    with pytest.raises(ValueError, match="exactly one original.*found 0"):
        discover_cohort_recordings(
            source_data_root,
            subjects=("0001",),
            expected_count=1,
        )


def test_stage_recording_reuses_signal_and_changes_only_vas_annotation(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    source_data_root = tmp_path / "KINGSTON" / "EEG_fMRI_data" / "source_data"
    raw_dir = source_data_root / "sub-0001" / "eeg" / "original_5khz"
    raw_dir.mkdir(parents=True)
    source_vhdr = _write_raw_brainvision_recording(raw_dir, basename)
    _write_corrected_reference(source_data_root, "0001", basename)
    recording = discover_cohort_recordings(
        source_data_root,
        subjects=("0001",),
        expected_count=1,
    )[0]

    manifest_row = stage_recording(recording, tmp_path / "staged")

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


def test_run_sanitization_publishes_final_manifest_paths(tmp_path: Path) -> None:
    basename = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = kingston_root / "EEG_fMRI_data" / "source_data"
    raw_dir = source_data_root / "sub-0001" / "eeg" / "original_5khz"
    raw_dir.mkdir(parents=True)
    _write_raw_brainvision_recording(raw_dir, basename)
    _write_corrected_reference(source_data_root, "0001", basename)
    output_root = tmp_path / "brainvision_marker_sanitized-v1"

    result = run_sanitization(
        source_data_root,
        output_root,
        subjects=("0001",),
        expected_count=1,
    )

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
