"""The report settings added for the gradient, spectra, and continuity panels."""

from __future__ import annotations

import pytest

from eeg_pipeline.preprocessing.report.settings import ReportSettings


def test_defaults_match_the_cohort_scanner_harmonic_band() -> None:
    """The per-subject and cohort views must describe the same frequencies."""
    assert ReportSettings().comb_frequency_range_hz == (15.0, 90.0)
    assert ReportSettings().comb_welch_seconds == 8.0


def test_the_spectra_ceiling_is_unset_so_it_can_inherit_the_low_pass() -> None:
    assert ReportSettings().spectra_fmax is None


def test_configured_values_are_read() -> None:
    settings = ReportSettings.from_mapping(
        {
            "thresholds": {
                "comb_frequency_range_hz": [10.0, 80.0],
                "comb_welch_seconds": 4.0,
            },
            "display": {"spectra_fmax": 90.0, "continuity_window_seconds": 2.0},
            "analysis": {
                "aperiodic_fit_range_hz": [3.0, 35.0],
                "response_window_s": [0.1, 0.8],
                "alpha_band_hz": [7.5, 12.5],
            },
            "acquisition": {
                "volume_marker_description": "Scanner/Volume",
                "pulse_marker_description": "Cardiac/R",
                "posterior_channel_pattern": "^(O|PO)",
            },
        }
    )

    assert settings.comb_frequency_range_hz == (10.0, 80.0)
    assert settings.comb_welch_seconds == 4.0
    assert settings.spectra_fmax == 90.0
    assert settings.continuity_window_seconds == 2.0
    assert settings.aperiodic_fit_range_hz == (3.0, 35.0)
    assert settings.response_window_s == (0.1, 0.8)
    assert settings.alpha_band_hz == (7.5, 12.5)
    assert settings.volume_marker_description == "Scanner/Volume"
    assert settings.pulse_marker_description == "Cardiac/R"
    assert settings.posterior_channel_pattern == "^(O|PO)"


def test_an_explicit_null_ceiling_stays_unset() -> None:
    settings = ReportSettings.from_mapping({"display": {"spectra_fmax": None}})

    assert settings.spectra_fmax is None


def test_a_reversed_comb_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="0 < low < high"):
        ReportSettings.from_mapping({"thresholds": {"comb_frequency_range_hz": [90.0, 15.0]}})


def test_a_malformed_comb_band_is_rejected() -> None:
    with pytest.raises(TypeError, match="exactly two values"):
        ReportSettings.from_mapping({"thresholds": {"comb_frequency_range_hz": [15.0]}})


@pytest.mark.parametrize(
    ("block", "key", "value", "match"),
    [
        ("thresholds", "comb_welch_seconds", 0.0, "comb_welch_seconds must be positive"),
        ("display", "spectra_fmax", -1.0, "spectra_fmax must be positive"),
        (
            "display",
            "continuity_window_seconds",
            0.0,
            "continuity_window_seconds must be positive",
        ),
        ("analysis", "response_window_s", [0.5, 0.5], "response_window_s"),
        ("analysis", "alpha_band_hz", [13.0, 8.0], "alpha_band_hz"),
        (
            "analysis",
            "aperiodic_fit_range_hz",
            [45.0, 2.0],
            "aperiodic_fit_range_hz",
        ),
    ],
)
def test_meaningless_values_are_rejected(block, key, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        ReportSettings.from_mapping({block: {key: value}})


@pytest.mark.parametrize(
    "key",
    ["volume_marker_description", "pulse_marker_description", "posterior_channel_pattern"],
)
def test_empty_acquisition_identifiers_are_rejected(key) -> None:
    with pytest.raises(ValueError, match=key):
        ReportSettings.from_mapping({"acquisition": {key: ""}})


@pytest.mark.parametrize(
    "values",
    [
        {"analaysis": {"alpha_band_hz": [8.0, 12.0]}},
        {"analysis": {"alpha_bnad_hz": [8.0, 12.0]}},
    ],
)
def test_unknown_report_keys_are_rejected(values) -> None:
    with pytest.raises(ValueError, match="unknown report"):
        ReportSettings.from_mapping(values)


@pytest.mark.parametrize(
    "key",
    ["volume_marker_description", "pulse_marker_description", "posterior_channel_pattern"],
)
def test_null_acquisition_identifiers_are_rejected(key) -> None:
    with pytest.raises(TypeError, match=key):
        ReportSettings.from_mapping({"acquisition": {key: None}})


def test_enabled_must_be_a_boolean_not_a_truthy_string() -> None:
    with pytest.raises(TypeError, match="report.enabled"):
        ReportSettings.from_mapping({"enabled": "false"})


def test_null_method_bands_are_rejected_instead_of_reset_to_defaults() -> None:
    with pytest.raises(TypeError, match="alpha_band_hz"):
        ReportSettings.from_mapping({"analysis": {"alpha_band_hz": None}})


def test_the_shipped_config_block_validates() -> None:
    """The defaults written into eeg_config.yaml must be loadable as written."""
    from pathlib import Path

    import yaml

    import eeg_pipeline.utils.config as config_package

    path = Path(config_package.__file__).parent / "eeg_config.yaml"
    if not path.is_file():
        pytest.skip("packaged config not present")
    values = yaml.safe_load(path.read_text())

    settings = ReportSettings.from_mapping(values["report"])

    assert settings.comb_frequency_range_hz == (15.0, 90.0)
    assert settings.continuity_window_seconds == 1.0


class _Config:
    """Minimal stand-in for the pipeline config object."""

    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_the_spectra_ceiling_is_inherited_from_the_low_pass() -> None:
    """Above the low-pass the filter sets the trace, so the axis must stop there."""
    settings = ReportSettings.from_config(_Config(**{"preprocessing.h_freq": 100}))

    assert settings.spectra_fmax == 100.0


def test_the_line_frequency_is_inherited_from_the_notch() -> None:
    settings = ReportSettings.from_config(_Config(**{"preprocessing.notch_freq": 50}))

    assert settings.spectra_line_frequency == 50.0


def test_an_explicit_report_setting_beats_the_inherited_one() -> None:
    """A site that deliberately widens the axis must not have it overwritten."""
    settings = ReportSettings.from_config(
        _Config(
            **{
                "preprocessing.h_freq": 100,
                "report": {"display": {"spectra_fmax": 40.0}},
            }
        )
    )

    assert settings.spectra_fmax == 40.0


def test_nothing_is_inherited_when_the_preprocessing_values_are_absent() -> None:
    settings = ReportSettings.from_config(_Config())

    assert settings.spectra_fmax is None
    assert settings.spectra_line_frequency is None
