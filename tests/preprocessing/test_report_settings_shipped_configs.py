"""The report block of every shipped config must parse, and must mean what it says.

``ReportSettings`` rejects an unknown key, so a config that drifts from the code fails
loudly rather than silently ignoring a setting a lab thought it had set. These tests run
that check against the configs this repository actually ships, which is the case a user
meets first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eeg_pipeline.preprocessing.report.settings import ReportSettings

CONFIG_ROOT = Path(__file__).resolve().parents[2]

SHIPPED_CONFIGS = (
    "eeg_pipeline/utils/config/eeg_config.yaml",
    "eeg_pipeline/utils/config/presets/eeg_only.yaml",
    "eeg_pipeline/utils/config/presets/rest.yaml",
    "studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml",
)


def _report_block(relative: str) -> dict | None:
    loaded = yaml.safe_load((CONFIG_ROOT / relative).read_text(encoding="utf-8"))
    return (loaded or {}).get("report")


@pytest.mark.parametrize("relative", SHIPPED_CONFIGS)
def test_the_shipped_report_block_parses(relative: str) -> None:
    block = _report_block(relative)
    if block is None:
        pytest.skip(f"{relative} carries no report block")
    ReportSettings.from_mapping(block)


def test_the_packaged_defaults_match_the_dataclass_defaults() -> None:
    """A YAML value that drifts from its default is a silent change of behaviour.

    Every key the packaged config states is also a field default in the code. Stating
    them twice is deliberate -- the file is where a user reads what the pipeline does --
    so the two have to be checked against each other.
    """
    settings = ReportSettings.from_mapping(_report_block(SHIPPED_CONFIGS[0]))
    defaults = ReportSettings()

    for field in (
        "plausible_heart_rate_bpm",
        "marker_agreement_tolerance_s",
        "notch_exclusion_half_width_hz",
        "repetition_time_tolerance_s",
        "channel_position_tolerance_m",
        "min_subjects_for_median",
        "min_subjects_for_outer_band",
        "alpha_reference_band_hz",
        "bcg_residual_window_s",
        "bcg_residual_baseline_s",
        "bcg_residual_measurement_s",
        "non_event_prefixes",
        "component_label_patterns",
    ):
        assert getattr(settings, field) == getattr(defaults, field), field


def test_the_pain_study_override_restates_its_own_bookkeeping() -> None:
    """The site marker spellings left the shipped default and must land here instead.

    Without this the study's continuity figure would start counting its button presses
    and scanner markers as trials, which is a change to a delivered report rather than a
    change to what other labs inherit.
    """
    settings = ReportSettings.from_mapping(_report_block(SHIPPED_CONFIGS[3]))

    for prefix in ("VOLUME/", "R  ", "R/", "RESPONSE/"):
        assert prefix in settings.non_event_prefixes, prefix
    # And still everything the pipeline writes for itself.
    for prefix in ReportSettings().non_event_prefixes:
        assert prefix in settings.non_event_prefixes, prefix


def test_no_shipped_default_names_a_trial_prefix_of_this_study() -> None:
    """The one way non_event_prefixes fails silently is by suppressing real events."""
    for relative in SHIPPED_CONFIGS:
        block = _report_block(relative)
        if block is None:
            continue
        prefixes = ReportSettings.from_mapping(block).non_event_prefixes
        assert not any(prefix.upper().startswith("TRIG") for prefix in prefixes), relative
