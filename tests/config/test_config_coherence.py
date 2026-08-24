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


def test_decomb_manifest_with_downstream_notch_is_rejected() -> None:
    config = _config(
        paths__decomb_manifest="/data/line_notch_manifest.tsv",
        preprocessing__notch_freq=60,
    )

    report = check_config_coherence(config)

    assert "preprocessing.notch_freq" in _keys(report.errors)


def test_decomb_manifest_without_downstream_notch_is_coherent() -> None:
    config = _config(
        paths__decomb_manifest="/data/line_notch_manifest.tsv",
        preprocessing__notch_freq=None,
    )

    assert check_config_coherence(config).errors == ()


def test_null_decomb_manifest_preserves_notch_behavior() -> None:
    config = _config(
        paths__decomb_manifest=None,
        preprocessing__notch_freq=60,
    )

    assert check_config_coherence(config).errors == ()


def test_every_icalabel_prerequisite_is_reported_before_processing() -> None:
    config = _config(
        ica__use_icalabel=True,
        ica__algorithm="fastica",
        ica__l_freq=2.0,
        ica__h_freq=80.0,
    )
    config["eeg.reference"] = ["P9", "P10"]

    report = check_config_coherence(config)

    assert {
        "ica.algorithm",
        "ica.l_freq",
        "ica.h_freq",
        "eeg.reference",
    } <= _keys(report.errors)


def test_icalabel_accepts_picard_extended_infomax() -> None:
    config = _config(ica__algorithm="picard-extended_infomax")

    assert check_config_coherence(config).errors == ()


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


def test_rest_mode_asks_for_the_task_label_too() -> None:
    """This check previously exempted rest mode, on the reasoning that a resting-state
    recording has no task to name. But the label is not a name for a condition — BIDS
    puts a 'task-' entity on every EEG file, resting-state ones included, and it is how
    both the raw recording and the cleaned epochs are found. So a rest study is not being
    asked to invent anything; it is being asked which of its own files to read.

    Exempting it meant a rest study that left the key unset passed validation, ran
    preprocessing to completion, and then failed at feature extraction, where the epochs
    are located by globbing 'sub-<id>_task-<label>*_epo.fif' and 'None' matched nothing.
    """
    config = _config(
        preprocessing__task_is_rest=True,
        feature_engineering__task_is_rest=True,
        ica__band_specific_report__enabled=False,
    )
    config["project.task"] = None

    assert "project.task" in _keys(check_config_coherence(config).errors)


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


def test_an_eeg_fmri_dataset_gets_no_scanner_warnings() -> None:
    config = _config(preprocessing__eeg_fmri=True)

    assert check_config_coherence(config).warnings == ()


###################################################################
# Being outside a scanner and having an ECG lead are separate facts
###################################################################


def test_out_of_scanner_ecg_stages_are_not_warned_about_when_an_ecg_lead_exists() -> None:
    """Reported in issue #14. Both stages were listed as scanner-only, on the reasoning
    that out-of-scanner montages carry no ECG. They often do, and the cardiac review
    falls back to detecting R peaks from the channel when there are no Analyzer markers,
    so outside a scanner it reviews ordinary cardiac artifact. Warning here told a user
    with an ECG lead that a stage they had correctly enabled would be skipped."""
    config = _config(
        preprocessing__eeg_fmri=False,
        preprocessing__brainvision_analyzer__enabled=False,
        alignment__trim_to_volume_bounds=False,
        ica__cardiac_review__enabled=True,
        preprocessing__clean_events_qc__ecg_coupling__enabled=True,
    )
    config["eeg.ecg_channels"] = ["ECG"]

    report = check_config_coherence(config)

    assert report.warnings == (), [str(warning) for warning in report.warnings]


def test_ecg_stages_without_an_ecg_lead_are_warned_about_naming_the_lead() -> None:
    """The dependency these actually have. Cardiac QC raises at its call site when
    eeg.ecg_channels names nothing, so the config can say this in advance."""
    config = _config(
        preprocessing__eeg_fmri=False,
        preprocessing__brainvision_analyzer__enabled=False,
        alignment__trim_to_volume_bounds=False,
        ica__cardiac_review__enabled=True,
        preprocessing__clean_events_qc__ecg_coupling__enabled=True,
    )
    config["eeg.ecg_channels"] = []

    report = check_config_coherence(config)

    assert _keys(report.warnings) == {
        "ica.cardiac_review.enabled",
        "preprocessing.clean_events_qc.ecg_coupling.enabled",
    }
    assert all("eeg.ecg_channels" in str(warning) for warning in report.warnings)


def test_a_missing_ecg_lead_is_reported_inside_the_scanner_too() -> None:
    """The condition is the lead, not the room. An EEG-fMRI config that enables cardiac
    review without naming an ECG channel is in exactly the same position."""
    config = _config(preprocessing__eeg_fmri=True, ica__cardiac_review__enabled=True)
    config["eeg.ecg_channels"] = []

    assert "ica.cardiac_review.enabled" in _keys(check_config_coherence(config).warnings)
