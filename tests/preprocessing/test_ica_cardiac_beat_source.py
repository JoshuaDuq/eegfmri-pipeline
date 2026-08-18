# Beat times come from markers, the channel, or whichever is available.

from __future__ import annotations

import pytest

from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings


def test_beat_source_defaults_to_auto():
    assert CardiacReviewSettings().beat_source == "auto"


def test_marker_description_defaults_to_none():
    assert CardiacReviewSettings().marker_description is None


def test_beat_source_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="beat_source"):
        CardiacReviewSettings.from_mapping({"beat_source": "guess"})


def test_marker_description_survives_a_slash():
    settings = CardiacReviewSettings.from_mapping(
        {"marker_description": "Pulse Artifact/R"}
    )
    assert settings.marker_description == "Pulse Artifact/R"
