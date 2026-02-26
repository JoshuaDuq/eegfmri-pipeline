from __future__ import annotations

import pytest

from eeg_pipeline.utils.parsing import (
    parse_frequency_band_definitions,
    parse_group_arg,
    parse_roi_definitions,
)


def test_parse_group_arg_handles_all_aliases() -> None:
    assert parse_group_arg("all") is None
    assert parse_group_arg("*") is None
    assert parse_group_arg("@all") is None


def test_parse_group_arg_normalizes_separators() -> None:
    assert parse_group_arg("001,002;003 004") == ["001", "002", "003", "004"]


def test_parse_frequency_band_definitions_parses_valid_values() -> None:
    parsed = parse_frequency_band_definitions(["alpha:8:12.5", "beta:13:30"])
    assert parsed == {"alpha": [8.0, 12.5], "beta": [13.0, 30.0]}


def test_parse_frequency_band_definitions_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="low must be < high"):
        parse_frequency_band_definitions(["alpha:12:8"])


def test_parse_roi_definitions_parses_expected_format() -> None:
    parsed = parse_roi_definitions(["Frontal:Fp1,Fp2,F3,F4"])
    assert parsed == {"Frontal": [r"^(Fp1|Fp2|F3|F4)$"]}


def test_parse_roi_definitions_rejects_missing_channels() -> None:
    with pytest.raises(ValueError, match="no channels specified"):
        parse_roi_definitions(["Frontal:"])
