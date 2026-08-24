"""The document's phases, and the flat section order derived from them."""

from __future__ import annotations

import json

from eeg_pipeline.preprocessing.report.phases import (
    PHASES,
    SECTION_ORDER,
    phases_as_json,
    phases_present,
)

#: The complete reading order, stated independently so a new section cannot land in the
#: document merely because it was appended last during the build.
EXPECTED_SECTION_ORDER = (
    "At a glance",
    "Configuration",
    "Filter response",
    "Channel and region coverage",
    "Electrode bridging",
    "Data quality over time",
    "Muscle artifact screening",
    "Raw (original)",
    "Raw (filtered)",
    "ICA: epochs for fitting",
    "ICA decomposition quality",
    "ICA cardiac artifact review",
    "Cardiac rhythm",
    "ICA ocular artifact review",
    "ICA component review",
    "ICA: components",
    "ICA: removals",
    "Exploratory band-fitted ICAs",
    "Sensor spectra before and after ICA",
    "Events",
    "Epoch rejection",
    "Evoked responses",
    "Signal preservation",
    "Epochs (before cleaning)",
    "Epochs (clean)",
    "Raw (clean)",
)


def test_the_derived_order_is_the_declared_reading_order():
    assert SECTION_ORDER == EXPECTED_SECTION_ORDER


def test_every_section_belongs_to_exactly_one_phase():
    seen = [section for phase in PHASES for section in phase.sections]

    assert sorted(seen) == sorted(set(seen))


def test_a_phase_with_no_section_present_is_not_returned():
    present = phases_present(["At a glance", "Configuration"])

    assert [phase.title for phase in present] == ["What this report says"]


def test_phases_are_returned_in_document_order():
    present = phases_present(["Epoch rejection", "At a glance", "Data quality over time"])

    assert [phase.title for phase in present] == [
        "What this report says",
        "What came in",
        "What survived",
    ]


def test_an_unknown_section_puts_no_phase_in_the_result():
    assert phases_present(["Something a later stage added"]) == ()


def test_a_report_with_no_cardiac_review_still_groups_the_decomposition():
    present = phases_present(["ICA decomposition quality", "ICA: removals"])

    assert [phase.title for phase in present] == ["What the decomposition did"]


def test_the_serialised_table_carries_every_phase_in_order():
    payload = json.loads(phases_as_json())

    assert [entry["title"] for entry in payload] == [phase.title for phase in PHASES]


def test_the_serialised_table_carries_each_phases_sections():
    payload = json.loads(phases_as_json())

    assert payload[0]["sections"] == list(PHASES[0].sections)
