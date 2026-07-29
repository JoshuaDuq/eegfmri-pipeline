"""Config contradictions are reported together, before any recording is opened.

Each of these was already detected somewhere — one at a time, from whichever stage
tripped over it, and in two cases only after ICA had been fitted. Setting up a
resting-state study meant fixing a key, waiting for ICA, fixing the next key it named,
and waiting again.
"""

from __future__ import annotations

import pytest

from eeg_pipeline.utils.config.coherence import check_config_coherence
from eeg_pipeline.utils.config.loader import ConfigDict, load_config


def _config(**overrides) -> ConfigDict:
    config = load_config()
    for key, value in overrides.items():
        config[key.replace("__", ".")] = value
    return config


def _keys(issues) -> set[str]:
    return {issue.key for issue in issues}


def test_the_shipped_config_is_coherent() -> None:
    report = check_config_coherence(load_config())

    assert report.errors == (), [str(i) for i in report.errors]


def test_every_rest_incompatibility_is_reported_in_one_pass() -> None:
    """The behaviour this whole module exists for: three fixes named at once rather than
    discovered across three runs."""
    config = _config(
        preprocessing__task_is_rest=True,
        feature_engineering__task_is_rest=True,
        ica__band_specific_report__enabled=True,
        ica__band_specific_report__tfr__enabled=True,
        preprocessing__rest_epochs_overlap=2.0,
    )
    config["ica.band_specific_report.comparisons"] = [{"name": "a"}]

    report = check_config_coherence(config)

    assert {
        "ica.band_specific_report.tfr.enabled",
        "ica.band_specific_report.comparisons",
        "preprocessing.rest_epochs_overlap",
    } <= _keys(report.errors)


def test_the_raised_error_lists_every_key_at_once() -> None:
    config = _config(
        preprocessing__task_is_rest=True,
        feature_engineering__task_is_rest=True,
        ica__band_specific_report__enabled=True,
        ica__band_specific_report__tfr__enabled=True,
        preprocessing__rest_epochs_overlap=2.0,
    )

    with pytest.raises(ValueError) as excinfo:
        check_config_coherence(config).raise_if_errors()

    message = str(excinfo.value)
    assert "ica.band_specific_report.tfr.enabled" in message
    assert "preprocessing.rest_epochs_overlap" in message


def test_a_rest_config_without_the_task_only_keys_is_clean() -> None:
    config = _config(
        preprocessing__task_is_rest=True,
        feature_engineering__task_is_rest=True,
        ica__band_specific_report__enabled=True,
        ica__band_specific_report__tfr__enabled=False,
        preprocessing__rest_epochs_duration=10.0,
        preprocessing__rest_epochs_overlap=0.0,
    )
    config["ica.band_specific_report.comparisons"] = []

    assert check_config_coherence(config).errors == ()


def test_rest_mode_does_not_demand_a_task_label() -> None:
    """A resting-state recording has no task to name, and requiring one is how a
    baseline-only study got told to invent a label for its single condition."""
    config = _config(
        preprocessing__task_is_rest=True,
        feature_engineering__task_is_rest=True,
        ica__band_specific_report__enabled=False,
    )
    config["project.task"] = None

    assert "project.task" not in _keys(check_config_coherence(config).errors)


def test_a_task_run_without_a_task_label_is_reported() -> None:
    config = _config()
    config["project.task"] = None

    assert "project.task" in _keys(check_config_coherence(config).errors)


def test_disagreeing_rest_flags_are_reported_not_raised_from_inside() -> None:
    config = _config(
        preprocessing__task_is_rest=True,
        feature_engineering__task_is_rest=False,
    )

    assert "project.paradigm" in _keys(check_config_coherence(config).errors)


def test_scanner_keys_left_on_for_an_eeg_only_dataset_are_listed_as_warnings() -> None:
    """None of these stops the run — each is gated at its own call site — but a config
    still asking for four things it will not get is one nobody has finished adapting."""
    config = _config(
        preprocessing__eeg_fmri=False,
        preprocessing__brainvision_analyzer__enabled=True,
        ica__cardiac_review__enabled=True,
        preprocessing__clean_events_qc__ecg_coupling__enabled=True,
        alignment__trim_to_volume_bounds=True,
    )

    report = check_config_coherence(config)

    assert report.errors == ()
    assert _keys(report.warnings) == {
        "preprocessing.brainvision_analyzer.enabled",
        "ica.cardiac_review.enabled",
        "preprocessing.clean_events_qc.ecg_coupling.enabled",
        "alignment.trim_to_volume_bounds",
    }


def test_an_eeg_fmri_dataset_gets_no_scanner_warnings() -> None:
    config = _config(preprocessing__eeg_fmri=True)

    assert check_config_coherence(config).warnings == ()
