from pathlib import Path

import pytest

from studies.pain_study.scripts.conversion.organize_source_eeg import (
    discover_unorganized_subjects,
    organize_cohort,
    organize_subject_eeg,
)


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

    raw_destination = eeg_directory / "original_untrimmed_5khz"
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


def test_discover_unorganized_subjects_finds_new_subjects_only(tmp_path: Path) -> None:
    source_data_root = tmp_path / "source_data"
    (source_data_root / "sub-0001" / "eeg" / "original_untrimmed_5khz").mkdir(parents=True)
    (source_data_root / "sub-0001" / "eeg" / "brainvision_processed_1khz").mkdir()
    (source_data_root / "sub-0016" / "eeg").mkdir(parents=True)

    assert discover_unorganized_subjects(source_data_root) == ("0016",)


def test_organize_cohort_discovers_new_subject_without_fixed_list(tmp_path: Path) -> None:
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = tmp_path / "source_data"
    participant_raw = kingston_root / "sub_0016_2026_07_16" / "raw"
    source_eeg = source_data_root / "sub-0016" / "eeg"
    _write_triplet(participant_raw, "task", 200)
    _write_triplet(source_eeg, "task_scannerpulse_corrected", 1000)

    results = organize_cohort(kingston_root, source_data_root)

    assert [result.subject for result in results] == ["0016"]


def test_organize_cohort_accepts_new_subject_without_brainvision_processed_data(
    tmp_path: Path,
) -> None:
    kingston_root = tmp_path / "KINGSTON"
    source_data_root = tmp_path / "source_data"
    participant_raw = kingston_root / "sub_0016_2026_07_16" / "raw"
    source_eeg = source_data_root / "sub-0016" / "eeg"
    _write_triplet(participant_raw, "task", 200)
    source_eeg.mkdir(parents=True)

    results = organize_cohort(kingston_root, source_data_root)

    assert results[0].original_triplets == 1
    assert results[0].processed_triplets == 0
    assert (source_eeg / "original_untrimmed_5khz" / "task.vhdr").is_file()
    assert not (source_eeg / "brainvision_processed_1khz").exists()


def test_discover_unorganized_subjects_rejects_empty_inventory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No unorganized EEG subjects"):
        discover_unorganized_subjects(tmp_path)
