"""The document's phases, and the flat section order derived from them."""

from __future__ import annotations

from eeg_pipeline.preprocessing.report.phases import (
    PHASES,
    SECTION_ORDER,
    phases_present,
)

#: The order the document was built in before the phase table existed. Copied from the
#: literal that stood in organize.py, so the derivation is checked against the shipped
#: document rather than against itself.
ORDER_BEFORE_PHASES = (
    "At a glance",
    "Configuration",
    "Filter response",
    "Channel and region coverage",
    "Data quality over time",
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
    "Signal preservation",
    "Epochs (before cleaning)",
    "Epochs (clean)",
    "Raw (clean)",
)


def test_the_derived_order_is_the_order_the_document_already_had():
    assert SECTION_ORDER == ORDER_BEFORE_PHASES


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
