from __future__ import annotations

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.ica_ocular_report import (
    OcularReviewSettings,
    _absolute_scores,
    _ocular_review_guide_html,
    _resolve_eog_channels,
    _surrogate_channels,
)


def _raw(channel_names: list[str], channel_types: list[str]) -> mne.io.BaseRaw:
    info = mne.create_info(channel_names, 100.0, channel_types)
    return mne.io.RawArray(np.zeros((len(channel_names), 100)), info, verbose="ERROR")


def test_frontopolar_surrogates_are_accepted_when_no_eog_electrode_exists() -> None:
    raw = _raw(["Fp1", "Fp2", "Cz"], ["eeg", "eeg", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"])

    assert _resolve_eog_channels(raw, settings) == ["Fp1", "Fp2"]
    assert _surrogate_channels(raw, ["Fp1", "Fp2"]) == ("Fp1", "Fp2")


def test_dedicated_eog_channels_are_not_treated_as_surrogates() -> None:
    raw = _raw(["EOG1", "Cz"], ["eog", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels="eog")

    assert _resolve_eog_channels(raw, settings) == ["EOG1"]
    assert _surrogate_channels(raw, ["EOG1"]) == ()


def test_missing_configured_ocular_channel_fails_fast() -> None:
    raw = _raw(["Fp1", "Cz"], ["eeg", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"])

    with pytest.raises(ValueError, match=r"\['Fp2'\]"):
        _resolve_eog_channels(raw, settings)


def test_automatic_eog_selection_names_the_surrogate_remedy() -> None:
    raw = _raw(["Fp1", "Cz"], ["eeg", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels="eog")

    with pytest.raises(ValueError, match="Fp1"):
        _resolve_eog_channels(raw, settings)


def test_guide_declares_circularity_only_when_surrogates_are_used() -> None:
    settings = OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"])

    with_surrogates = _ocular_review_guide_html(settings, surrogates=("Fp1", "Fp2"))
    without_surrogates = _ocular_review_guide_html(settings, surrogates=())

    assert "partly circular" in with_surrogates
    assert "Fp1, Fp2" in with_surrogates
    assert "partly circular" not in without_surrogates


def test_scores_combine_eog_channels_by_largest_absolute_correlation() -> None:
    scores = np.array([[0.1, -0.9, 0.2], [-0.7, 0.3, 0.05]])

    combined = _absolute_scores(scores, component_count=3)

    np.testing.assert_allclose(combined, [0.7, 0.9, 0.2])


def test_scores_that_do_not_describe_every_component_fail_fast() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        _absolute_scores(np.array([0.1, 0.2, 0.3]), component_count=4)


def test_non_finite_scores_fail_fast() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _absolute_scores(np.array([0.1, np.nan]), component_count=2)
