from pathlib import Path

from eeg_pipeline.utils.data.preprocessing import find_brainvision_vhdrs


def test_find_brainvision_vhdrs_selects_brainvision_processed_1khz(
    tmp_path: Path,
) -> None:
    eeg_directory = tmp_path / "sub-0001" / "eeg"
    original = eeg_directory / "original_5khz" / "run.vhdr"
    processed = eeg_directory / "brainvision_processed_1khz" / "run.vhdr"
    original.parent.mkdir(parents=True)
    processed.parent.mkdir(parents=True)
    original.touch()
    processed.touch()

    assert find_brainvision_vhdrs(tmp_path) == [processed]
