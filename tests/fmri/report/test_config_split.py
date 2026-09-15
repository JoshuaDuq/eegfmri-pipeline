from __future__ import annotations

import pytest

from fmri_pipeline.analysis.plotting_config import (
    FmriReportConfig,
    FmriStatsConfig,
    report_config_from_mapping,
    stats_config_from_mapping,
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
        report_config_from_mapping({"space": "mni", "enabled": True})


def test_every_moved_key_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as excinfo:
        report_config_from_mapping({"space": "mni", "include_effect_size": True, "enabled": True})
    message = str(excinfo.value)
    assert "space" in message and "include_effect_size" in message


def test_a_clean_config_splits_without_complaint() -> None:
    report = report_config_from_mapping({"enabled": True, "include_unthresholded": False})
    assert report.enabled is True
    assert report.include_unthresholded is False


def test_the_stats_config_validates_its_space() -> None:
    with pytest.raises(ValueError, match="fmri_stats.space"):
        FmriStatsConfig(space="nonsense").validate()


def test_the_stats_mapping_rejects_unknown_compute_keys() -> None:
    with pytest.raises(ValueError, match="Unknown fmri_stats key"):
        stats_config_from_mapping({"space": "native", "typo": True})


def test_the_stats_mapping_validates_inference_thresholds() -> None:
    with pytest.raises(ValueError, match="z_threshold"):
        stats_config_from_mapping({"threshold_mode": "z", "z_threshold": 0})


def test_a_disabled_report_config_still_validates_reproducibility_settings() -> None:
    with pytest.raises(ValueError, match="Unsupported plot format"):
        FmriReportConfig(enabled=False, formats=("pdf",)).validate()


def test_an_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported plot format"):
        FmriReportConfig(enabled=True, formats=("pdf",)).validate()


def test_an_unknown_report_key_is_rejected_instead_of_ignored() -> None:
    with pytest.raises(ValueError, match="Unknown fmri_report key"):
        report_config_from_mapping({"enabled": True, "typo": True})


def test_new_figure_settings_reject_an_unknown_value() -> None:
    for key, bad in (
        ("raster_conditions", "sideways"),
        ("design_matrix_runs", "every-other"),
    ):
        config = FmriReportConfig(**{key: bad})
        with pytest.raises(ValueError, match=key):
            config.validate()


def test_new_figure_settings_keep_their_documented_defaults() -> None:
    config = FmriReportConfig()
    config.validate()
    assert config.raster_conditions == "task"
    assert config.design_matrix_runs == "first"
    assert config.include_glass_brain is True


def test_the_pruned_panels_default_off() -> None:
    config = FmriReportConfig()
    config.validate()
    # Both duplicate a panel that is still drawn: the residual SD map reproduces the
    # standard-error map (r = 0.944), and the regressor matrix's readable structure is
    # the confound expansion the VIF panel already summarises per regressor.
    assert config.include_residual_sd_map is False
    assert config.include_regressor_correlation is False


def test_the_pruned_panels_can_be_switched_back_on() -> None:
    config = FmriReportConfig(
        include_residual_sd_map=True, include_regressor_correlation=True
    )
    config.validate()
    assert config.include_residual_sd_map and config.include_regressor_correlation
