"""Configurable settings for the subject HTML report.

Two kinds of value live here, and the distinction decides what belongs.

Thresholds and display choices decide what the report *flags* and how it draws, so a
site can tune the warnings to its montage and acquisition without touching a derivative.

Acquisition descriptions decide what the report can *find*. Marker labels, montage
naming, and the window an evoked response occupies are properties of a recording setup,
not of this pipeline, and a report that hardcoded them would silently omit a panel on any
dataset that spells them differently. Defaults match this project's own acquisition.

Values that define a *method* rather than a setup stay as module constants where they are
used — the harmonic peak and background fractions, the residual quantile the aperiodic
fit trims at, the minimum counts below which a measurement is not attempted. Exposing
those would invite tuning an estimator per subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from eeg_pipeline.preprocessing.report.aperiodic import DEFAULT_FIT_RANGE_HZ
from eeg_pipeline.preprocessing.report.preservation import (
    ALPHA_BAND_HZ,
    DEFAULT_RESPONSE_WINDOW_S,
    POSTERIOR_PATTERN,
)
from eeg_pipeline.preprocessing.report.scanner import VOLUME_MARKER_DESCRIPTION
from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION

_REPORT_KEYS = frozenset({"enabled", "thresholds", "display", "analysis", "acquisition"})
_THRESHOLD_KEYS = frozenset(
    {
        "min_r_markers_per_volume",
        "min_roi_channels",
        "min_samples_per_squared_component",
        "low_variance_exclusion_floor",
        "max_group_levels",
        "min_runs_for_quantile_band",
        "comb_frequency_range_hz",
        "comb_welch_seconds",
    }
)
_DISPLAY_KEYS = frozenset(
    {
        "spectra_line_frequency",
        "spectra_marked_frequencies",
        "color_limit_percentile",
        "component_overview_columns",
        "spectra_fmax",
        "continuity_window_seconds",
    }
)
_ANALYSIS_KEYS = frozenset(
    {"aperiodic_fit_range_hz", "response_window_s", "alpha_band_hz"}
)
_ACQUISITION_KEYS = frozenset(
    {
        "volume_marker_description",
        "pulse_marker_description",
        "posterior_channel_pattern",
    }
)


def _reject_unknown_keys(
    values: Mapping[str, Any],
    allowed: frozenset[str],
    setting: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown {setting} configuration key(s): {names}.")


def _mapping_block(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in values:
        return {}
    block = values[key]
    if not isinstance(block, Mapping):
        raise TypeError(f"report.{key} must be a mapping.")
    return block


def _string(block: Mapping[str, Any], key: str, default: str) -> str:
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, str):
        raise TypeError(f"report.acquisition.{key} must be a string.")
    return value


def _boolean(values: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in values:
        return default
    value = values[key]
    if not isinstance(value, bool):
        raise TypeError(f"report.{key} must be a boolean.")
    return value


def _pair(
    block: Mapping[str, Any],
    key: str,
    *,
    default: tuple[float, float],
    setting: str,
) -> tuple[float, float]:
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{setting} must contain exactly two values.")
    return (float(value[0]), float(value[1]))


def _optional_float(block: Mapping[str, Any], key: str) -> float | None:
    """Read a setting whose absence means "resolve it from the pipeline config".

    An explicit ``null`` and a missing key both give ``None``; :meth:`from_config` then
    fills the value in from the preprocessing settings that determine it.
    """
    value = block.get(key)
    return None if value is None else float(value)


def _inherited(explicit: float | None, preprocessing_value: Any) -> float | None:
    """Return the report's own setting when it has one, otherwise what it inherits."""
    if explicit is not None:
        return explicit
    return None if preprocessing_value is None else float(preprocessing_value)


@dataclass(frozen=True)
class ReportSettings:
    """Thresholds and display choices for the subject report."""

    enabled: bool = True
    min_r_markers_per_volume: float = 0.5
    min_roi_channels: int = 2
    min_samples_per_squared_component: float = 20.0
    low_variance_exclusion_floor: float = 0.01
    max_group_levels: int = 12
    min_runs_for_quantile_band: int = 5
    color_limit_percentile: float = 98.0
    component_overview_columns: int = 8
    #: Line-noise fundamental marked on the sensor spectra. ``None`` marks nothing;
    #: leave unset to inherit ``preprocessing.notch_freq``.
    spectra_line_frequency: float | None = None
    #: Extra frequencies to mark, for example residual scanner-harmonic centres.
    spectra_marked_frequencies: tuple[float, ...] = ()
    #: Upper edge of the sensor-spectra axis. Above the configured low-pass the filter,
    #: not the recording, sets the trace, so leave unset to inherit
    #: ``preprocessing.h_freq`` rather than plotting roll-off as if it were data.
    spectra_fmax: float | None = None
    #: Band searched for the gradient comb. Defaults match the cohort scanner-harmonic
    #: QC so the per-subject and cohort views describe the same frequencies.
    comb_frequency_range_hz: tuple[float, float] = (15.0, 90.0)
    #: Welch window for the comb. Longer than the sensor-spectra window because the comb
    #: must be resolved between its teeth, not merely detected.
    comb_welch_seconds: float = 8.0
    #: Window over which time-resolved amplitude is pooled.
    continuity_window_seconds: float = 1.0
    #: Band the aperiodic background is fitted over. Keep it below the line-noise
    #: fundamental so the notch and its skirts cannot tilt the slope.
    aperiodic_fit_range_hz: tuple[float, float] = DEFAULT_FIT_RANGE_HZ
    #: Annotation marking each scanner volume. A dataset that spells it differently, or
    #: has none, simply gets no gradient section.
    volume_marker_description: str = VOLUME_MARKER_DESCRIPTION
    #: Annotation marking each detected heartbeat.
    pulse_marker_description: str = PULSE_MARKER_DESCRIPTION
    #: Window the evoked split halves are correlated over, in seconds from onset. A
    #: paradigm whose response falls outside it reports a reliability near zero for a
    #: sound recording, so this must match the paradigm rather than the other way round.
    response_window_s: tuple[float, float] = DEFAULT_RESPONSE_WINDOW_S
    #: Band searched for the posterior rhythm used as preservation evidence.
    alpha_band_hz: tuple[float, float] = ALPHA_BAND_HZ
    #: Regular expression selecting the posterior sensors that rhythm is expected over.
    #: Montage-dependent: a non-10-20 naming scheme needs its own pattern, or the
    #: preservation panel finds no channels to measure.
    posterior_channel_pattern: str = POSTERIOR_PATTERN

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ReportSettings:
        """Build settings from the ``report`` config block, validating every value."""
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("report configuration must be a mapping.")
        _reject_unknown_keys(values, _REPORT_KEYS, "report")
        thresholds = _mapping_block(values, "thresholds")
        display = _mapping_block(values, "display")
        analysis = _mapping_block(values, "analysis")
        acquisition = _mapping_block(values, "acquisition")
        _reject_unknown_keys(thresholds, _THRESHOLD_KEYS, "report.thresholds")
        _reject_unknown_keys(display, _DISPLAY_KEYS, "report.display")
        _reject_unknown_keys(analysis, _ANALYSIS_KEYS, "report.analysis")
        _reject_unknown_keys(acquisition, _ACQUISITION_KEYS, "report.acquisition")

        settings = cls(
            enabled=_boolean(values, "enabled", cls.enabled),
            min_r_markers_per_volume=float(
                thresholds.get("min_r_markers_per_volume", cls.min_r_markers_per_volume)
            ),
            min_roi_channels=int(thresholds.get("min_roi_channels", cls.min_roi_channels)),
            min_samples_per_squared_component=float(
                thresholds.get(
                    "min_samples_per_squared_component",
                    cls.min_samples_per_squared_component,
                )
            ),
            low_variance_exclusion_floor=float(
                thresholds.get("low_variance_exclusion_floor", cls.low_variance_exclusion_floor)
            ),
            max_group_levels=int(thresholds.get("max_group_levels", cls.max_group_levels)),
            min_runs_for_quantile_band=int(
                thresholds.get("min_runs_for_quantile_band", cls.min_runs_for_quantile_band)
            ),
            spectra_line_frequency=_optional_float(display, "spectra_line_frequency"),
            spectra_marked_frequencies=tuple(
                float(value) for value in display.get("spectra_marked_frequencies", ())
            ),
            color_limit_percentile=float(
                display.get("color_limit_percentile", cls.color_limit_percentile)
            ),
            component_overview_columns=int(
                display.get("component_overview_columns", cls.component_overview_columns)
            ),
            spectra_fmax=_optional_float(display, "spectra_fmax"),
            comb_frequency_range_hz=_pair(
                thresholds,
                "comb_frequency_range_hz",
                default=cls.comb_frequency_range_hz,
                setting="report.thresholds.comb_frequency_range_hz",
            ),
            comb_welch_seconds=float(thresholds.get("comb_welch_seconds", cls.comb_welch_seconds)),
            continuity_window_seconds=float(
                display.get("continuity_window_seconds", cls.continuity_window_seconds)
            ),
            aperiodic_fit_range_hz=_pair(
                analysis,
                "aperiodic_fit_range_hz",
                default=cls.aperiodic_fit_range_hz,
                setting="report.analysis.aperiodic_fit_range_hz",
            ),
            response_window_s=_pair(
                analysis,
                "response_window_s",
                default=cls.response_window_s,
                setting="report.analysis.response_window_s",
            ),
            alpha_band_hz=_pair(
                analysis,
                "alpha_band_hz",
                default=cls.alpha_band_hz,
                setting="report.analysis.alpha_band_hz",
            ),
            volume_marker_description=_string(
                acquisition,
                "volume_marker_description",
                cls.volume_marker_description,
            ),
            pulse_marker_description=_string(
                acquisition,
                "pulse_marker_description",
                cls.pulse_marker_description,
            ),
            posterior_channel_pattern=_string(
                acquisition,
                "posterior_channel_pattern",
                cls.posterior_channel_pattern,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject values that would make a threshold meaningless rather than strict."""
        if not 0.0 < self.min_r_markers_per_volume <= 2.0:
            raise ValueError(
                "report.thresholds.min_r_markers_per_volume must lie in (0, 2]; a heart "
                "rate cannot exceed a few markers per scanner volume."
            )
        if self.min_roi_channels < 2:
            raise ValueError(
                "report.thresholds.min_roi_channels must be at least 2; a single channel "
                "is not a regional average."
            )
        if self.min_samples_per_squared_component <= 0:
            raise ValueError(
                "report.thresholds.min_samples_per_squared_component must be positive."
            )
        if not 0.0 <= self.low_variance_exclusion_floor < 1.0:
            raise ValueError("report.thresholds.low_variance_exclusion_floor must lie in [0, 1).")
        if self.max_group_levels < 2:
            raise ValueError("report.thresholds.max_group_levels must be at least 2.")
        if self.min_runs_for_quantile_band < 2:
            raise ValueError("report.thresholds.min_runs_for_quantile_band must be at least 2.")
        if not 0.0 < self.color_limit_percentile <= 100.0:
            raise ValueError("report.display.color_limit_percentile must lie in (0, 100].")
        if self.component_overview_columns < 1:
            raise ValueError("report.display.component_overview_columns must be at least 1.")
        if self.spectra_fmax is not None and self.spectra_fmax <= 0:
            raise ValueError("report.display.spectra_fmax must be positive when set.")
        low, high = self.comb_frequency_range_hz
        if not 0 < low < high:
            raise ValueError(
                "report.thresholds.comb_frequency_range_hz must satisfy 0 < low < high."
            )
        # The comb is measured by contrasting each harmonic with the spectrum between it
        # and its neighbours, so the window has to resolve that gap. Whether it does
        # depends on the volume rate, which is only known once a run is read, so
        # compute_comb_residual reports the section as absent rather than failing here.
        if self.comb_welch_seconds <= 0:
            raise ValueError("report.thresholds.comb_welch_seconds must be positive.")
        if self.continuity_window_seconds <= 0:
            raise ValueError("report.display.continuity_window_seconds must be positive.")
        aperiodic_low, aperiodic_high = self.aperiodic_fit_range_hz
        if not 0 < aperiodic_low < aperiodic_high:
            raise ValueError(
                "report.analysis.aperiodic_fit_range_hz must satisfy 0 < low < high."
            )
        response_start, response_stop = self.response_window_s
        if response_stop <= response_start:
            raise ValueError(
                "report.analysis.response_window_s must have a stop after its start."
            )
        alpha_low, alpha_high = self.alpha_band_hz
        if not 0 < alpha_low < alpha_high:
            raise ValueError("report.analysis.alpha_band_hz must satisfy 0 < low < high.")
        for key, value in (
            ("volume_marker_description", self.volume_marker_description),
            ("pulse_marker_description", self.pulse_marker_description),
            ("posterior_channel_pattern", self.posterior_channel_pattern),
        ):
            if not value.strip():
                raise ValueError(f"report.acquisition.{key} must not be empty.")
        try:
            re.compile(self.posterior_channel_pattern)
        except re.error as error:
            raise ValueError(
                "report.acquisition.posterior_channel_pattern is not a valid regular "
                f"expression: {error}"
            ) from error

    @classmethod
    def from_config(cls, config: Any | None) -> ReportSettings:
        """Build settings from a pipeline config, resolving what the report inherits.

        Two display settings are properties of the preprocessing rather than of the
        report: which frequency carries line noise, and where the low-pass cuts the
        spectrum off. Resolving them here keeps every config lookup in one place, so a
        caller cannot render a report against settings it partially resolved itself.
        """
        if config is None:
            return cls()
        settings = cls.from_mapping(config.get("report", {}))
        return replace(
            settings,
            spectra_line_frequency=_inherited(
                settings.spectra_line_frequency, config.get("preprocessing.notch_freq")
            ),
            spectra_fmax=_inherited(settings.spectra_fmax, config.get("preprocessing.h_freq")),
        )


__all__ = ["ReportSettings"]
