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
    # The config the loader actually reads for this study. The override template that
    # used to sit beside it was deleted: nothing loaded it, and once this file existed it
    # was a test-covered, authoritative-looking duplicate of live settings.
    "studies/pain_study/config/pain_study.yaml",
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
        "channel_position_tolerance_m",
        "min_subjects_for_median",
        "min_subjects_for_outer_band",
        "alpha_reference_band_hz",
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


def test_the_out_of_scanner_preset_needs_no_file_only_this_study_has() -> None:
    """``eeg_only`` is the entry point for a user with no scanner and no line removal.

    ``paths.decomb_manifest`` is inherited from the packaged config, where it points at
    the line-notch manifest of the study this repository was developed on. Building the
    report settings then raised FileNotFoundError on a path the new user has never heard
    of, before a single section was rendered — for a dataset that had no gradient comb to
    remove in the first place.
    """
    preset = yaml.safe_load(
        (CONFIG_ROOT / "eeg_pipeline/utils/config/presets/eeg_only.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert preset["paths"]["decomb_manifest"] is None


def test_a_configured_manifest_that_is_absent_is_still_an_error() -> None:
    """Opting in and mistyping the path must not silently drop the stopband mask.

    Without the mask the comb section scores filtered bins as measured residual, which is
    the failure the mask exists to prevent — so a study that asks for a manifest and does
    not have one has to hear about it.
    """

    class _Config:
        def get(self, key, default=None):
            return {"paths.decomb_manifest": "/no/such/manifest.tsv"}.get(key, default)

    with pytest.raises(FileNotFoundError, match="Decomb manifest"):
        ReportSettings.from_config(_Config())


def test_core_config_names_no_scanner_key():
    from pathlib import Path

    text = Path("eeg_pipeline/utils/config/eeg_config.yaml").read_text(encoding="utf-8")
    # eeg_fmri, brainvision_analyzer, scanner_harmonic_qc and trim_to_volume_bounds are
    # NOT here: pipelines/preprocessing.py still runs the stages that read them, and
    # Task 13 is what deletes those stages. Task 13 extends this list.
    for key in (
        "comb_frequency_range_hz:",
        "comb_welch_seconds:",
        "repetition_time_tolerance_s:",
        "volume_marker_description:",
        "pulse_marker_description:",
        "min_r_markers_per_volume:",
        "bcg_residual_window_s:",
    ):
        assert key not in text, f"{key} is still in the core config"
