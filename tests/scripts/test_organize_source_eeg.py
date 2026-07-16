from pathlib import Path

import pytest

from studies.pain_study.scripts.organize_source_eeg import organize_subject_eeg


def _write_triplet(directory: Path, stem: str, sampling_interval_us: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={stem}.eeg\n"
        f"MarkerFile={stem}.vmrk\n"
        f"SamplingInterval={sampling_interval_us}\n",
        encoding="utf-8",
    )
    (directory / f"{stem}.vmrk").write_text(
        "Brain Vision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={stem}.eeg\n",
        encoding="utf-8",
    )
    (directory / f"{stem}.eeg").write_bytes(b"signal")


def test_organize_subject_eeg_copies_5khz_and_moves_processed_1khz(
    tmp_path: Path,
) -> None:
    participant_raw = tmp_path / "sub_0001_2026_03_02" / "raw"
    source_subject = tmp_path / "source_data" / "sub-0001"
    eeg_directory = source_subject / "eeg"
    _write_triplet(participant_raw, "task", 200)
    _write_triplet(eeg_directory, "task_scannerpulse_corrected", 1000)
    _write_triplet(eeg_directory / "resting", "rest_scannerpulse_corrected", 1000)

    result = organize_subject_eeg("0001", participant_raw, source_subject)

    raw_destination = eeg_directory / "original_5khz"
    processed_destination = eeg_directory / "brainvision_processed_1khz"
    assert (raw_destination / "task.vhdr").is_file()
    assert (participant_raw / "task.vhdr").is_file()
    assert (processed_destination / "task_scannerpulse_corrected.vhdr").is_file()
    assert (processed_destination / "resting" / "rest_scannerpulse_corrected.vhdr").is_file()
    assert not (eeg_directory / "task_scannerpulse_corrected.vhdr").exists()
    assert not (eeg_directory / "resting").exists()
    assert result.original_triplets == 1
    assert result.processed_triplets == 2


def test_organize_subject_eeg_rejects_nested_original_triplets(tmp_path: Path) -> None:
    participant_raw = tmp_path / "sub_0001_2026_03_02" / "raw"
    source_subject = tmp_path / "source_data" / "sub-0001"
    _write_triplet(participant_raw / "nested", "task", 200)
    _write_triplet(
        source_subject / "eeg",
        "task_scannerpulse_corrected",
        1000,
    )

    with pytest.raises(FileNotFoundError, match="No BrainVision headers"):
        organize_subject_eeg("0001", participant_raw, source_subject)
