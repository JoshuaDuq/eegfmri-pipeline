"""Configurable thresholds for the subject HTML report.

Every value here decides what the report *flags*, not what the pipeline computes, so a
site can tune the warnings to its montage and acquisition without touching a derivative.
Defaults match the module-level constants the report used before these became
configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


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

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ReportSettings:
        """Build settings from the ``report`` config block, validating every value."""
        values = values or {}
        if not isinstance(values, Mapping):
            raise TypeError("report configuration must be a mapping.")
        thresholds = values.get("thresholds", {}) or {}
        display = values.get("display", {}) or {}
        for name, block in (("thresholds", thresholds), ("display", display)):
            if not isinstance(block, Mapping):
                raise TypeError(f"report.{name} must be a mapping.")

        settings = cls(
            enabled=bool(values.get("enabled", cls.enabled)),
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
            comb_frequency_range_hz=cls._frequency_range(
                thresholds.get("comb_frequency_range_hz"),
                default=cls.comb_frequency_range_hz,
            ),
            comb_welch_seconds=float(thresholds.get("comb_welch_seconds", cls.comb_welch_seconds)),
            continuity_window_seconds=float(
                display.get("continuity_window_seconds", cls.continuity_window_seconds)
            ),
        )
        settings.validate()
        return settings

    @staticmethod
    def _frequency_range(
        value: Any,
        *,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(
                "report.thresholds.comb_frequency_range_hz must contain exactly two values."
            )
        return (float(value[0]), float(value[1]))

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
