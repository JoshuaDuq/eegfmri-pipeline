"""The acquisition, population and vendor descriptions the report used to hardcode.

Each of these was a module constant that only a lab with this project's recording could
live with. The tests here fix two things per setting: that its default still reproduces
the constant it replaced, so an unmodified config produces an unmodified report, and that
a configured value actually reaches the measurement rather than being read and dropped.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.report.settings import ReportSettings


def _settings(**block) -> ReportSettings:
    return ReportSettings.from_mapping(block)


# --------------------------------------------------------------------------------------
# Heart rate: one statement, two documents
# --------------------------------------------------------------------------------------


def test_the_default_heart_rate_range_spans_an_adult_under_load() -> None:
    assert ReportSettings().plausible_heart_rate_bpm == (30.0, 220.0)


def test_the_interval_range_is_derived_from_the_rate_rather_than_stored() -> None:
    """The two constants this replaced disagreed at 200 against 220 bpm.

    Deriving one from the other is what makes that impossible to reintroduce: there is no
    second value to forget to update.
    """
    settings = _settings(thresholds={"plausible_heart_rate_bpm": [40.0, 200.0]})

    low_s, high_s = settings.plausible_rr_range_s
    assert low_s == pytest.approx(60.0 / 200.0)
    assert high_s == pytest.approx(60.0 / 40.0)


def test_one_configured_rate_moves_both_the_subject_and_the_cohort_reading() -> None:
    """A paediatric cohort raises the ceiling once and both documents follow."""
    from studies.pain_study.analysis.bcg.cohort import _participant_row

    settings = _settings(thresholds={"plausible_heart_rate_bpm": [30.0, 260.0]})
    runs = pd.DataFrame({"median_bpm": [240.0], "n_beats": [500.0], "beat_dropouts": [1.0]})

    # The cohort side: 240 bpm is inside the configured ceiling, so it is not flagged.
    row = _participant_row(
        "0001", runs, plausible_bpm=settings.plausible_heart_rate_bpm
    )
    assert row["implausible_rate"] is False

    # The subject side reads the same statement, expressed as intervals.
    assert settings.plausible_rr_range_s[0] == pytest.approx(60.0 / 260.0)

    # And under the default ceiling the same participant is flagged, so the setting is
    # doing the work rather than the row being unflaggable.
    default_row = _participant_row("0001", runs)
    assert default_row["implausible_rate"] is True


def test_a_reversed_heart_rate_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="plausible_heart_rate_bpm"):
        _settings(thresholds={"plausible_heart_rate_bpm": [220.0, 30.0]})


def test_every_fallback_default_derives_from_the_one_heart_rate_statement() -> None:
    """The module defaults are what a caller gets without threading settings through.

    Two of them written independently is how the original disagreement happened, so the
    seconds are derived from the bpm and the cohort module imports the same constant.
    Anything that reintroduces a second literal fails here.
    """
    from eeg_pipeline.preprocessing.report.rr_intervals import (
        DEFAULT_PLAUSIBLE_HEART_RATE_BPM,
        PLAUSIBLE_RR_RANGE_S,
    )
    from studies.pain_study.analysis.bcg.cohort import PLAUSIBLE_BPM

    settings = ReportSettings()
    assert tuple(PLAUSIBLE_BPM) == tuple(DEFAULT_PLAUSIBLE_HEART_RATE_BPM)
    assert tuple(settings.plausible_heart_rate_bpm) == tuple(DEFAULT_PLAUSIBLE_HEART_RATE_BPM)
    assert settings.plausible_rr_range_s == pytest.approx(PLAUSIBLE_RR_RANGE_S)


# --------------------------------------------------------------------------------------
# Beat-marker agreement tolerance
# --------------------------------------------------------------------------------------


def test_the_agreement_tolerance_defaults_to_the_constant_it_replaced() -> None:
    assert ReportSettings().marker_agreement_tolerance_s == 0.1


def test_a_tolerance_that_could_match_the_next_beat_is_rejected() -> None:
    """At 220 bpm the shortest interval is 273 ms, so 140 ms could reach a neighbour."""
    with pytest.raises(ValueError, match="marker_agreement_tolerance_s"):
        _settings(thresholds={"marker_agreement_tolerance_s": 0.14})


def test_the_same_tolerance_becomes_admissible_under_a_slower_ceiling() -> None:
    """The bound is relative to the configured rate, not a second fixed number."""
    settings = _settings(
        thresholds={
            "plausible_heart_rate_bpm": [30.0, 120.0],
            "marker_agreement_tolerance_s": 0.14,
        }
    )
    assert settings.marker_agreement_tolerance_s == 0.14


def test_the_residual_windows_default_to_the_constants_they_replaced() -> None:
    settings = ReportSettings()
    assert settings.bcg_residual_window_s == (-0.2, 0.6)
    assert settings.bcg_residual_baseline_s == (-0.2, -0.1)
    assert settings.bcg_residual_measurement_s == (0.0, 0.5)


@pytest.mark.parametrize(
    "key, span",
    [
        ("bcg_residual_baseline_s", [-0.5, -0.4]),
        ("bcg_residual_measurement_s", [0.0, 0.9]),
    ],
)
def test_a_residual_window_outside_the_epoch_is_rejected(key: str, span: list) -> None:
    """A baseline outside the epoch is silently no baseline, which prints as a number."""
    with pytest.raises(ValueError, match=key):
        _settings(analysis={key: span})


def test_the_residual_windows_can_move_together_for_a_different_field_strength() -> None:
    settings = _settings(
        analysis={
            "bcg_residual_window_s": [-0.3, 0.8],
            "bcg_residual_baseline_s": [-0.3, -0.15],
            "bcg_residual_measurement_s": [0.0, 0.7],
        }
    )
    assert settings.bcg_residual_window_s == (-0.3, 0.8)
    assert settings.bcg_residual_measurement_s == (0.0, 0.7)


# --------------------------------------------------------------------------------------
# Posterior rhythm reference band
# --------------------------------------------------------------------------------------


def test_the_reference_band_defaults_to_the_constant_it_replaced() -> None:
    assert ReportSettings().alpha_reference_band_hz == (3.0, 25.0)


def test_a_reference_band_that_does_not_surround_the_rhythm_is_rejected() -> None:
    """The prominence is an excess over a background fitted outside the band."""
    with pytest.raises(ValueError, match="alpha_reference_band_hz"):
        _settings(
            analysis={
                "alpha_band_hz": [7.0, 14.0],
                "alpha_reference_band_hz": [8.0, 25.0],
            }
        )


def test_a_developmental_band_can_take_its_reference_window_with_it() -> None:
    settings = _settings(
        analysis={"alpha_band_hz": [6.0, 9.0], "alpha_reference_band_hz": [2.0, 20.0]}
    )
    assert settings.alpha_band_hz == (6.0, 9.0)
    assert settings.alpha_reference_band_hz == (2.0, 20.0)


# --------------------------------------------------------------------------------------
# Non-event annotation prefixes
# --------------------------------------------------------------------------------------


def _annotated_raw(descriptions: list[str]) -> mne.io.BaseRaw:
    info = mne.create_info(["Cz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((1, 3000)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[float(index + 1) for index in range(len(descriptions))],
            duration=[0.0] * len(descriptions),
            description=descriptions,
        )
    )
    return raw


def test_the_shipped_default_names_only_what_the_pipeline_writes_for_itself() -> None:
    """A site's own marker spellings in the default silently suppress another lab's events."""
    assert ReportSettings().non_event_prefixes == ("BAD", "EDGE", "NEW SEGMENT")


def test_a_response_marker_is_an_event_under_the_shipped_default() -> None:
    """In a response-locked paradigm the button press is the event, not bookkeeping."""
    from eeg_pipeline.preprocessing.report.continuity import _event_onsets

    raw = _annotated_raw(["Response/R  1", "BAD_movement", "New Segment/"])

    onsets = _event_onsets(
        raw,
        marker_descriptions=("Volume/V  1", "Pulse Artifact/R"),
        non_event_prefixes=ReportSettings().non_event_prefixes,
    )
    assert len(onsets) == 1


def test_the_pain_study_override_still_suppresses_its_own_bookkeeping() -> None:
    """The site spellings moved to the study config; its reports must not change."""
    from eeg_pipeline.preprocessing.report.continuity import _event_onsets

    settings = _settings(
        acquisition={
            "non_event_prefixes": [
                "BAD",
                "EDGE",
                "NEW SEGMENT",
                "VOLUME/",
                "R  ",
                "R/",
                "RESPONSE/",
            ]
        }
    )
    raw = _annotated_raw(["Trig_therm_48", "Response/R  1", "R  1", "BAD_movement"])

    onsets = _event_onsets(
        raw,
        marker_descriptions=("Volume/V  1", "Pulse Artifact/R"),
        non_event_prefixes=settings.non_event_prefixes,
    )
    assert len(onsets) == 1


def test_the_configured_marker_descriptions_need_no_entry_of_their_own() -> None:
    """They are merged in at the call site, which is why the default can omit them."""
    from eeg_pipeline.preprocessing.report.continuity import _event_onsets

    raw = _annotated_raw(["Scanner/Volume", "Cardiac/R", "Stimulus/S  1"])

    onsets = _event_onsets(
        raw,
        marker_descriptions=("Scanner/Volume", "Cardiac/R"),
        non_event_prefixes=ReportSettings().non_event_prefixes,
    )
    assert len(onsets) == 1


def test_prefixes_are_matched_case_insensitively() -> None:
    from eeg_pipeline.preprocessing.report.continuity import _event_onsets

    raw = _annotated_raw(["heartbeat/tick", "Stimulus/S  1"])

    onsets = _event_onsets(
        raw,
        marker_descriptions=(),
        non_event_prefixes=("Heartbeat/",),
    )
    assert len(onsets) == 1


def test_an_empty_prefix_list_is_rejected() -> None:
    """Nothing excluded means BAD spans are counted as trials."""
    with pytest.raises(ValueError, match="non_event_prefixes"):
        _settings(acquisition={"non_event_prefixes": []})


def test_a_bare_string_is_rejected_rather_than_read_as_its_characters() -> None:
    with pytest.raises(TypeError, match="non_event_prefixes"):
        _settings(acquisition={"non_event_prefixes": "BAD"})


# --------------------------------------------------------------------------------------
# Component label patterns
# --------------------------------------------------------------------------------------


def test_configured_label_patterns_reach_the_count() -> None:
    from eeg_pipeline.preprocessing.report.cohort.record import component_label_counts

    components = pd.DataFrame(
        {
            "status": ["bad", "bad"],
            "status_description": ["Detected lidschlag artifact", "Detected herz artifact"],
        }
    )

    # The shipped English patterns recognise neither description.
    assert component_label_counts(components)["other"] == 2

    counts = component_label_counts(
        components, label_patterns=(("lidschlag", "eye"), ("herz", "heart"))
    )
    assert counts["eye"] == 1
    assert counts["heart"] == 1
    assert counts["other"] == 0


def test_pattern_order_survives_the_config() -> None:
    """"channel noise" and "line noise" both contain "noise"; the specific one must win."""
    settings = _settings(
        acquisition={
            "component_label_patterns": [
                ["channel noise", "channel"],
                ["noise", "line"],
            ]
        }
    )
    assert settings.component_label_patterns[0] == ("channel noise", "channel")

    from eeg_pipeline.preprocessing.report.cohort.record import component_label_counts

    components = pd.DataFrame(
        {"status": ["bad"], "status_description": ["Auto-detected channel noise"]}
    )
    counts = component_label_counts(
        components, label_patterns=settings.component_label_patterns
    )
    assert counts["channel"] == 1
    assert counts["line"] == 0


@pytest.mark.parametrize("label", ["other", "unrecorded"])
def test_an_outcome_class_cannot_be_named_by_a_pattern(label: str) -> None:
    """Both are fallbacks; naming one as a rule describes a default as a decision."""
    with pytest.raises(ValueError, match="component_label_patterns"):
        _settings(acquisition={"component_label_patterns": [["blink", label]]})


def test_a_malformed_pattern_entry_is_rejected() -> None:
    with pytest.raises(TypeError, match="component_label_patterns"):
        _settings(acquisition={"component_label_patterns": [["blink"]]})


# --------------------------------------------------------------------------------------
# Tolerances and cohort gates
# --------------------------------------------------------------------------------------


def test_the_tolerances_default_to_the_constants_they_replaced() -> None:
    settings = ReportSettings()
    assert settings.notch_exclusion_half_width_hz == 2.0
    assert settings.repetition_time_tolerance_s == 1e-3
    assert settings.channel_position_tolerance_m == 5e-3


@pytest.mark.parametrize(
    "key",
    [
        "notch_exclusion_half_width_hz",
        "repetition_time_tolerance_s",
        "channel_position_tolerance_m",
    ],
)
def test_a_non_positive_tolerance_is_rejected(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _settings(thresholds={key: 0.0})


def test_the_cohort_gates_default_to_the_band_gate_constants() -> None:
    settings = ReportSettings()
    assert settings.min_subjects_for_median == 5
    assert settings.min_subjects_for_outer_band == 10
    assert settings.band_gates().min_subjects_for_median == 5


def test_a_configured_gate_reaches_the_band_gates() -> None:
    settings = _settings(
        thresholds={"min_subjects_for_median": 6, "min_subjects_for_outer_band": 12}
    )
    gates = settings.band_gates()
    assert gates.min_subjects_for_median == 6
    assert gates.min_subjects_for_outer_band == 12


def test_a_gate_that_would_extrapolate_a_quantile_is_rejected_by_the_config() -> None:
    """The arithmetic lives in BandGates; the config must not be able to bypass it."""
    with pytest.raises(ValueError, match="min_subjects_for_median"):
        _settings(thresholds={"min_subjects_for_median": 2})


def test_the_cohort_gates_are_distinct_from_the_within_participant_gate() -> None:
    """min_runs_for_quantile_band gates a band over runs, not over participants."""
    settings = _settings(
        thresholds={"min_runs_for_quantile_band": 3, "min_subjects_for_median": 8}
    )
    assert settings.min_runs_for_quantile_band == 3
    assert settings.min_subjects_for_median == 8
