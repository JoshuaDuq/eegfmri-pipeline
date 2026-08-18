from __future__ import annotations

import mne
import numpy as np
import pytest

from studies.pain_study.scripts.conversion.brainvision_markers import (
    sanitize_vas_marker_text,
    validate_unambiguous_vas_markers,
)


def _raw_with_annotations(descriptions: list[str]) -> mne.io.RawArray:
    info = mne.create_info(["Cz", "ECG"], 1_000.0, ["eeg", "eeg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    onsets = 0.1 + np.arange(len(descriptions)) * 0.1
    raw.set_annotations(mne.Annotations(onsets, 0.0, descriptions))
    return raw


def test_validate_unambiguous_vas_markers_rejects_ambiguous_vas_description() -> None:
    raw = _raw_with_annotations(["Volume/V  1", "Vas_on/V  1"])

    with pytest.raises(ValueError, match="Vas_on/V  1"):
        validate_unambiguous_vas_markers(raw)


def test_validate_unambiguous_vas_markers_accepts_sanitized_markers() -> None:
    raw = _raw_with_annotations(["Volume/V  1", "VAS_ON"])

    validate_unambiguous_vas_markers(raw)


def test_sanitize_vas_marker_text_changes_only_vas_descriptions() -> None:
    marker_text = (
        "BrainVision Data Exchange Marker File Version 1.0\r\n"
        "\r\n"
        "[Marker Infos]\r\n"
        "Mk1=New Segment,,1,1,0,20260713105002609017\r\n"
        "Mk2=Volume,V  1,101,1,0\r\n"
        "Mk3=Vas_on,V  1,200,1,0\r\n"
        "Mk4=Stim_on,S  1,300,1,0\r\n"
        "Mk5=Volume,V  1,4601,1,0\r\n"
    )

    result = sanitize_vas_marker_text(marker_text, n_samples=5_000)

    assert result.volume_count == 2
    assert result.vas_count == 1
    assert result.text == marker_text.replace(
        "Mk3=Vas_on,V  1,200,1,0\r\n",
        "Mk3=Vas_on,VAS_ON,200,1,0\r\n",
    )
    assert result.text.count("\r\n") == marker_text.count("\r\n")


def test_sanitize_vas_marker_text_rejects_other_v1_marker_types() -> None:
    marker_text = (
        "[Marker Infos]\n"
        "Mk1=Volume,V  1,1,1,0\n"
        "Mk2=Vas_on,V  1,2,1,0\n"
        "Mk3=Stim_on,V  1,3,1,0\n"
    )

    with pytest.raises(ValueError, match="Stim_on.*V  1"):
        sanitize_vas_marker_text(marker_text, n_samples=10)


def test_sanitize_vas_marker_text_supports_eeg_only_recordings() -> None:
    marker_text = "[Marker Infos]\n" "Mk1=Stim_on,S  1,1,1,0\n" "Mk2=Vas_on,V  1,2,1,0\n"

    result = sanitize_vas_marker_text(marker_text, n_samples=10)

    assert result.volume_count == 0
    assert result.vas_count == 1
    assert "Mk2=Vas_on,VAS_ON,2,1,0" in result.text


def test_sanitize_vas_marker_text_rejects_already_sanitized_input() -> None:
    marker_text = "[Marker Infos]\n" "Mk1=Volume,V  1,1,1,0\n" "Mk2=Vas_on,VAS_ON,2,1,0\n"

    with pytest.raises(ValueError, match="Vas_on.*VAS_ON"):
        sanitize_vas_marker_text(marker_text, n_samples=10)


@pytest.mark.parametrize(
    ("marker_record", "message"),
    [
        ("Mk1=Volume,V  1,0,1,0", "position"),
        ("Mk1=Volume,V  1,11,1,0", "position"),
        ("Mk1=Volume,V  1,1,0,0", "size"),
        ("Mk1=Volume,V  1,1,1,-1", "channel"),
    ],
)
def test_sanitize_vas_marker_text_rejects_invalid_marker_coordinates(
    marker_record: str,
    message: str,
) -> None:
    marker_text = "[Marker Infos]\n" f"{marker_record}\n" "Mk2=Vas_on,V  1,2,1,0\n"

    with pytest.raises(ValueError, match=message):
        sanitize_vas_marker_text(marker_text, n_samples=10)
