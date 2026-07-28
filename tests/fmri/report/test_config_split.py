from __future__ import annotations

import pytest

from fmri_pipeline.analysis.plotting_config import (
    FmriReportConfig,
    FmriStatsConfig,
    split_legacy_plotting_config,
)


def test_compute_triggering_fields_live_on_the_stats_config() -> None:
    stats = FmriStatsConfig(space="both", include_effect_size=True)
    assert stats.space == "both"
    assert stats.include_effect_size is True


def test_the_report_config_has_no_compute_triggering_fields() -> None:
    # A rendering setting must never be able to cause a GLM to be fit.
    names = set(FmriReportConfig.__dataclass_fields__)
    assert names.isdisjoint(
        {"space", "include_effect_size", "include_standard_error", "include_signatures"}
    )


def test_a_moved_key_fails_with_its_new_location_named() -> None:
    with pytest.raises(ValueError, match=r"fmri_stats\.space"):
        split_legacy_plotting_config({"space": "mni", "enabled": True})


def test_every_moved_key_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as excinfo:
        split_legacy_plotting_config(
            {"space": "mni", "include_effect_size": True, "enabled": True}
        )
    message = str(excinfo.value)
    assert "space" in message and "include_effect_size" in message


def test_a_clean_config_splits_without_complaint() -> None:
    report = split_legacy_plotting_config({"enabled": True, "z_threshold": 3.1})
    assert report.enabled is True
    assert report.z_threshold == 3.1


def test_the_report_config_still_validates_its_own_fields() -> None:
    with pytest.raises(ValueError, match="z-threshold"):
        FmriReportConfig(enabled=True, threshold_mode="z", z_threshold=-1).validate()


def test_radiological_is_a_rendering_setting() -> None:
    assert FmriReportConfig(enabled=True, radiological=True).radiological is True


def test_the_stats_config_validates_its_space() -> None:
    with pytest.raises(ValueError, match="fmri_stats.space"):
        FmriStatsConfig(space="nonsense").validate()


def test_a_disabled_report_config_skips_validation() -> None:
    # Nothing is rendered, so an unreachable setting is not an error.
    FmriReportConfig(enabled=False, threshold_mode="nonsense").validate()


def test_an_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported plot format"):
        FmriReportConfig(enabled=True, formats=("pdf",)).validate()
