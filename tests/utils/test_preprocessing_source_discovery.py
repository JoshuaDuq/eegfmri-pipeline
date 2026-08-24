from pathlib import Path

import pytest

from eeg_pipeline.utils.data.preprocessing import find_brainvision_vhdrs


def _two_layouts(tmp_path: Path) -> tuple[Path, Path]:
    eeg_directory = tmp_path / "sub-0001" / "eeg"
    original = eeg_directory / "original_untrimmed_5khz" / "run.vhdr"
    processed = eeg_directory / "brainvision_processed_1khz" / "run.vhdr"
    original.parent.mkdir(parents=True)
    processed.parent.mkdir(parents=True)
    original.touch()
    processed.touch()
    return original, processed


def test_find_brainvision_vhdrs_reads_the_named_layout(tmp_path: Path) -> None:
    _, processed = _two_layouts(tmp_path)

    assert find_brainvision_vhdrs(tmp_path, "brainvision_processed_1khz") == [processed]


def test_find_brainvision_vhdrs_reads_the_original_5khz_layout(tmp_path: Path) -> None:
    original, _ = _two_layouts(tmp_path)

    assert find_brainvision_vhdrs(tmp_path, "original_untrimmed_5khz") == [original]


def test_find_brainvision_vhdrs_rejects_a_layout_that_is_not_a_directory_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="single directory name"):
        find_brainvision_vhdrs(tmp_path, "eeg/original_untrimmed_5khz")
