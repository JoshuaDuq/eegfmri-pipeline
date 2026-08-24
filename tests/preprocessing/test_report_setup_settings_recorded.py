"""A configurable setup description has to be recorded, or the report cannot be read.

Two follow-throughs the new keys imply. A report records the setup it assumed, so a
reader can tell an assumption from a measurement. And a cohort refuses to pool
participants whose settings disagree about what was measured, which now includes the
sensors the posterior rhythm was measured over -- configurable since the panel was
written and never compared until the rest of the acquisition settings joined it.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from eeg_pipeline.preprocessing.report.cohort.homogeneity import COMPARED_SETTINGS
from eeg_pipeline.preprocessing.report.provenance import PROVENANCE_KEYS, provenance_html
from eeg_pipeline.preprocessing.report.settings import ReportSettings


class _Config:
    """The dotted-key lookup the provenance table reads a config through."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def get(self, key: str, default=None):
        return self._values.get(key, default)


NEW_KEYS = (
    "project.random_state",
    "preprocessing.line_freq",
    "paths.decomb_manifest",
    "pyprep.detection_low_pass",
    "pyprep.ransac",
    "pyprep.repeats",
    "pyprep.consider_previous_bads",
    "ica.reject",
    "ica.h_freq",
    "ica.use_ecg_detection",
    "ica.ecg_threshold",
    "ica.use_eog_detection",
    "ica.require_manual_review",
    "ica.manual_review_complete",
    "ica.cardiac_review.promote_exclusions",
    "ica.cardiac_review.promotion_minimum_run_fraction",
    "ica.ocular_review.eog_channels",
    "report.analysis.aperiodic_exclude_hz",
    "report.analysis.alpha_reference_band_hz",
    "report.acquisition.non_event_prefixes",
    "report.acquisition.component_label_patterns",
    "report.thresholds.notch_exclusion_half_width_hz",
    "report.thresholds.plausible_heart_rate_bpm",
    "report.thresholds.marker_agreement_tolerance_s",
    "report.thresholds.channel_position_tolerance_m",
    "report.thresholds.min_subjects_for_median",
    "report.thresholds.min_subjects_for_outer_band",
)


@pytest.mark.parametrize("key", NEW_KEYS)
def test_every_new_setting_is_recorded_in_the_provenance_table(key: str) -> None:
    assert key in {recorded for recorded, _ in PROVENANCE_KEYS}


@pytest.mark.parametrize("key", NEW_KEYS)
def test_a_configured_setting_reaches_the_rendered_provenance_table(key: str) -> None:
    html = provenance_html(_Config({key: "recorded-value"}))
    assert key in html
    assert "recorded-value" in html


def test_a_setting_the_dataset_never_configured_is_omitted_rather_than_listed() -> None:
    """An EEG-only dataset gains no rows about a scanner it was not in."""
    html = provenance_html(_Config({"report.thresholds.plausible_heart_rate_bpm": [30, 220]}))
    assert "bcg_residual_window_s" not in html


def test_manual_review_status_is_stated_as_a_configuration_claim() -> None:
    html = provenance_html(
        _Config(
            {
                "ica.require_manual_review": True,
                "ica.manual_review_complete": True,
            }
        )
    )

    assert "marked complete" in html
    assert "configuration assertion" in html


def test_pending_manual_review_is_named_as_blocking_epoch_creation() -> None:
    html = provenance_html(
        _Config(
            {
                "ica.require_manual_review": True,
                "ica.manual_review_complete": False,
            }
        )
    )

    assert "pending" in html
    assert "epoch creation is blocked" in html


COMPARED_NEW_KEYS = (
    "aperiodic_exclude_hz",
    "alpha_reference_band_hz",
    "posterior_channel_pattern",
    "plausible_heart_rate_bpm",
    "marker_agreement_tolerance_s",
    "notch_exclusion_half_width_hz",
)


@pytest.mark.parametrize("key", COMPARED_NEW_KEYS)
def test_settings_that_change_a_pooled_number_are_compared_across_the_cohort(
    key: str,
) -> None:
    assert key in COMPARED_SETTINGS


@pytest.mark.parametrize("key", COMPARED_SETTINGS)
def test_every_compared_setting_is_a_field_the_sidecar_actually_carries(key: str) -> None:
    """The sidecar stores ``asdict(settings)``, so a typo here compares nothing forever."""
    assert key in asdict(ReportSettings())


def test_a_disagreement_about_the_posterior_pattern_is_reported() -> None:
    """It selects the sensors alpha is measured over, so it changes the pooled number."""
    from eeg_pipeline.preprocessing.report.cohort.homogeneity import _disagreements

    found = _disagreements(
        {
            "0001": {"posterior_channel_pattern": "^(P|PO|O)"},
            "0002": {"posterior_channel_pattern": "^(E6[0-9])"},
        },
        COMPARED_SETTINGS,
    )
    assert "posterior_channel_pattern" in found


def test_a_nested_setting_compares_without_failing_on_an_unhashable_value() -> None:
    """JSON gives a list of pairs back as nested lists, which a shallow tuple() cannot hash."""
    from eeg_pipeline.preprocessing.report.cohort.homogeneity import _disagreements

    found = _disagreements(
        {
            "0001": {"component_label_patterns": [["eog", "eye"]]},
            "0002": {"component_label_patterns": [["lidschlag", "eye"]]},
        },
        ("component_label_patterns",),
    )
    assert "component_label_patterns" in found


def test_agreeing_participants_are_not_reported_as_a_disagreement() -> None:
    from eeg_pipeline.preprocessing.report.cohort.homogeneity import _disagreements

    shared = {"component_label_patterns": [["eog", "eye"]], "alpha_reference_band_hz": [3.0, 25.0]}
    found = _disagreements({"0001": dict(shared), "0002": dict(shared)}, COMPARED_SETTINGS)
    assert found == {}


def test_aperiodic_exclude_hz_compares_without_failing_on_an_unhashable_value() -> None:
    # Same shape as component_label_patterns: a list of pairs, JSON gives back as nested lists.
    from eeg_pipeline.preprocessing.report.cohort.homogeneity import _disagreements

    found = _disagreements(
        {
            "0001": {"aperiodic_exclude_hz": [[20.0, 25.0]]},
            "0002": {"aperiodic_exclude_hz": [[58.0, 62.0]]},
        },
        ("aperiodic_exclude_hz",),
    )
    assert "aperiodic_exclude_hz" in found


def test_aperiodic_exclude_hz_agreement_is_not_reported_as_a_disagreement() -> None:
    from eeg_pipeline.preprocessing.report.cohort.homogeneity import _disagreements

    shared = {"aperiodic_exclude_hz": [[20.0, 25.0], [58.0, 62.0]]}
    found = _disagreements({"0001": dict(shared), "0002": dict(shared)}, ("aperiodic_exclude_hz",))
    assert found == {}


def test_an_empty_sequence_setting_reads_as_none_not_a_blank_cell() -> None:
    # aperiodic_exclude_hz ships empty by default; joining zero items gave back "" before.
    from eeg_pipeline.preprocessing.report.provenance import _format_value

    assert _format_value(()) == "none"
    assert _format_value([]) == "none"


def test_a_configured_empty_exclusion_list_is_recorded_as_none_in_the_table() -> None:
    html = provenance_html(_Config({"report.analysis.aperiodic_exclude_hz": []}))

    assert "Aperiodic exclusion windows" in html
    assert "none" in html
