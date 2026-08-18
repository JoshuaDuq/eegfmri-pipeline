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
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from eeg_pipeline.preprocessing.report.analyzer_qc import (
    DEFAULT_PLAUSIBLE_HEART_RATE_BPM,
    MARKER_AGREEMENT_TOLERANCE_S,
    RESIDUAL_BASELINE_S,
    RESIDUAL_MEASUREMENT_S,
    RESIDUAL_WINDOW_S,
)
from eeg_pipeline.preprocessing.report.aperiodic import DEFAULT_FIT_RANGE_HZ
from eeg_pipeline.preprocessing.report.continuity import NON_EVENT_PREFIXES
from eeg_pipeline.preprocessing.report.filtering import NOTCH_EXCLUSION_HALF_WIDTH_HZ
from eeg_pipeline.preprocessing.report.preservation import (
    ALPHA_BAND_HZ,
    ALPHA_REFERENCE_BAND_HZ,
    DEFAULT_RESPONSE_WINDOW_S,
    POSTERIOR_PATTERN,
)
from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION

#: Participant counts at which each cohort band becomes drawable, and the montage and
#: sequence tolerances the cohort panels pool across. Stated here rather than imported
#: from the cohort tree, which would pull the whole aggregation machinery into every
#: module that reads a setting; the cohort modules import these back.
DEFAULT_MIN_SUBJECTS_FOR_MEDIAN = 5
DEFAULT_MIN_SUBJECTS_FOR_OUTER_BAND = 10
DEFAULT_REPETITION_TIME_TOLERANCE_S = 1e-3
DEFAULT_CHANNEL_POSITION_TOLERANCE_M = 5e-3


#: Detector prose mapped to the class a cohort counts an excluded component under.
#:
#: Ordered, and the order is load-bearing: "channel noise" and "line noise" both contain
#: "noise", so the more specific reading has to win. A description no pattern matches
#: still lands in ``other`` and is still counted.
DEFAULT_COMPONENT_LABEL_PATTERNS = (
    ("eye blink", "eye"),
    ("eog", "eye"),
    ("ocular", "eye"),
    ("heart beat", "heart"),
    ("ecg", "heart"),
    ("cardiac", "heart"),
    ("muscle", "muscle"),
    ("line noise", "line"),
    ("channel noise", "channel"),
)

#: Classes a pattern may name. ``other`` and ``unrecorded`` are outcomes rather than
#: patterns -- the first collects descriptions nothing matched, the second exclusions no
#: detector explained -- so a config that named either would be describing a fallback as
#: though it were a rule.
_ASSIGNABLE_LABEL_CLASSES = frozenset(
    {"eye", "heart", "muscle", "line", "channel"}
)

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
        "plausible_heart_rate_bpm",
        "marker_agreement_tolerance_s",
        "notch_exclusion_half_width_hz",
        "repetition_time_tolerance_s",
        "channel_position_tolerance_m",
        "min_subjects_for_median",
        "min_subjects_for_outer_band",
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
    {
        "aperiodic_fit_range_hz",
        "aperiodic_exclude_hz",
        "response_window_s",
        "alpha_band_hz",
        "alpha_reference_band_hz",
        "bcg_residual_window_s",
        "bcg_residual_baseline_s",
        "bcg_residual_measurement_s",
    }
)
_ACQUISITION_KEYS = frozenset(
    {
        "volume_marker_description",
        "pulse_marker_description",
        "posterior_channel_pattern",
        "non_event_prefixes",
        "component_label_patterns",
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


# _pair reads a single bounded numeric pair; this reads a list of them, e.g. the windows
# aperiodic_exclude_hz names. Absent elsewhere in this module, so it is built to match
# _pair's numeric conversion and _label_patterns' per-entry validation rather than either
# alone.
def _pair_sequence(
    block: Mapping[str, Any],
    key: str,
    *,
    default: tuple[tuple[float, float], ...],
    setting: str,
) -> tuple[tuple[float, float], ...]:
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{setting} must be a list of [low, high] pairs.")
    pairs = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError(f"{setting} entries must each contain exactly two values.")
        pairs.append((float(entry[0]), float(entry[1])))
    return tuple(pairs)


def _string_tuple(
    block: Mapping[str, Any],
    key: str,
    *,
    default: tuple[str, ...],
    setting: str,
) -> tuple[str, ...]:
    """Read a list of strings, rejecting a bare string silently read as its characters."""
    if key not in block:
        return default
    value = block[key]
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{setting} must be a list of strings.")
    entries = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(f"{setting} must contain only strings.")
        if not entry.strip():
            raise ValueError(f"{setting} must not contain an empty string.")
        entries.append(entry)
    return tuple(entries)


def _label_patterns(
    block: Mapping[str, Any],
    key: str,
    *,
    default: tuple[tuple[str, str], ...],
    setting: str,
) -> tuple[tuple[str, str], ...]:
    """Read the ordered detector-prose patterns, preserving the order YAML gave them.

    Order decides which of two overlapping patterns wins, so the sequence is carried
    through unsorted and a mapping is rejected: a YAML mapping would express the same
    pairs while leaving the tie-break to whatever order the loader happened to produce.
    """
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{setting} must be a list of [pattern, class] pairs.")
    patterns = []
    for entry in value:
        if isinstance(entry, str) or not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError(f"{setting} entries must each be a [pattern, class] pair.")
        pattern, label = entry
        if not isinstance(pattern, str) or not isinstance(label, str):
            raise TypeError(f"{setting} patterns and classes must both be strings.")
        if not pattern.strip():
            raise ValueError(f"{setting} must not contain an empty pattern.")
        if label not in _ASSIGNABLE_LABEL_CLASSES:
            allowed = ", ".join(sorted(_ASSIGNABLE_LABEL_CLASSES))
            raise ValueError(
                f"{setting} class {label!r} is not one a pattern may assign; use one of "
                f"{allowed}. 'other' collects descriptions no pattern matched and "
                "'unrecorded' counts exclusions no detector explained, so neither can be "
                "named by a rule."
            )
        patterns.append((pattern, label))
    return tuple(patterns)


def _optional_float(block: Mapping[str, Any], key: str) -> float | None:
    """Read a setting whose absence means "resolve it from the pipeline config".

    An explicit ``null`` and a missing key both give ``None``; :meth:`from_config` then
    fills the value in from the preprocessing settings that determine it.
    """
    value = block.get(key)
    return None if value is None else float(value)


def _unavailable_intervals(config: Any) -> Mapping[str, tuple[tuple[float, float], ...]]:
    """Bands a line-removal manifest reports as unavailable, keyed by recording id.

    Empty unless ``paths.decomb_manifest`` is configured, so a dataset that does not opt
    in never imports the adapter and keeps the fixed harmonic grid it always had.
    """
    manifest_path = config.get("paths.decomb_manifest", None)
    if manifest_path is None:
        return {}

    from eeg_pipeline.spectral_availability.decomb import load_decomb_manifest

    intervals: dict[str, tuple[tuple[float, float], ...]] = {}
    for exclusion in load_decomb_manifest(manifest_path).exclusions:
        key = exclusion.key
        entities = [f"sub-{key.subject}"]
        if key.session is not None:
            entities.append(f"ses-{key.session}")
        entities.extend([f"task-{key.task}", f"run-{key.run}"])
        intervals["_".join(entities)] = tuple(
            (interval.low_hz, interval.high_hz) for interval in exclusion.intervals
        )
    return intervals


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
    # Withheld from the aperiodic fit, on top of the notch stopbands and the
    # decomb-unavailable intervals.
    aperiodic_exclude_hz: tuple[tuple[float, float], ...] = ()
    #: Annotation marking each scanner volume. A dataset that spells it differently, or
    #: has none, simply gets no gradient section.
    volume_marker_description: str = "Volume/V  1"
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
    #: Neighbourhood the posterior rhythm is scored against. Moves with
    #: ``alpha_band_hz``: a band shifted for a developmental cohort needs a reference
    #: window that still surrounds it, or the peak is scored against the wrong background.
    alpha_reference_band_hz: tuple[float, float] = ALPHA_REFERENCE_BAND_HZ
    #: Physiologically possible heart rate, in bpm. Drives both the interval range the
    #: subject tachogram shades and the bound the cohort panel counts against.
    plausible_heart_rate_bpm: tuple[float, float] = DEFAULT_PLAUSIBLE_HEART_RATE_BPM
    #: Distance within which a beat marker and a detected R peak are the same beat. Two
    #: detectors disagree by tens of milliseconds on the same beat; this must stay well
    #: below the shortest interval ``plausible_heart_rate_bpm`` allows, or one beat can
    #: match its neighbour.
    marker_agreement_tolerance_s: float = MARKER_AGREEMENT_TOLERANCE_S
    #: Epoch the beat-locked residual is cut over, the baseline removed from it, and the
    #: window the residual is read in. All three are properties of the artifact's timing:
    #: the ballistocardiogram follows the R peak by roughly a fifth of a second, which is
    #: a statement about this population and this field strength.
    bcg_residual_window_s: tuple[float, float] = RESIDUAL_WINDOW_S
    bcg_residual_baseline_s: tuple[float, float] = RESIDUAL_BASELINE_S
    bcg_residual_measurement_s: tuple[float, float] = RESIDUAL_MEASUREMENT_S
    #: Half-width of the band a notch is treated as having removed. Sized for the notch
    #: the pipeline applies: a wider filter, or a different line-removal method, leaves a
    #: different span of bins that are the filter rather than the data.
    notch_exclusion_half_width_hz: float = NOTCH_EXCLUSION_HALF_WIDTH_HZ
    #: Bands a per-recording line-removal manifest reports as unavailable, keyed by
    #: recording id. Empty unless a Decomb manifest is configured, in which case these
    #: are the measured stopbands and replace the fixed harmonic grid, which describes
    #: neither where that filter cut nor how wide it was.
    unavailable_intervals_by_recording: Mapping[str, tuple[tuple[float, float], ...]] = field(
        default_factory=dict
    )
    #: Repetition times within this of each other are one sequence rather than two.
    repetition_time_tolerance_s: float = DEFAULT_REPETITION_TIME_TOLERANCE_S
    #: Electrode positions within this of each other are the same site on the head.
    channel_position_tolerance_m: float = DEFAULT_CHANNEL_POSITION_TOLERANCE_M
    #: Participants required before a cohort median, and before the outer band, is drawn.
    #: Distinct from ``min_runs_for_quantile_band``, which gates a within-participant band
    #: over runs. Neither may be set low enough to extrapolate a quantile.
    min_subjects_for_median: int = DEFAULT_MIN_SUBJECTS_FOR_MEDIAN
    min_subjects_for_outer_band: int = DEFAULT_MIN_SUBJECTS_FOR_OUTER_BAND
    #: Annotation prefixes that are not task events, so the continuity figure's event rug
    #: counts trials rather than the acquisition's own bookkeeping. The configured volume
    #: and pulse marker descriptions are excluded automatically and need no entry here;
    #: what belongs is whatever else a site writes that is not a trial -- a response
    #: marker in a paradigm where responses are not the event, a stimulus-computer
    #: heartbeat, a scanner trigger spelled its own way.
    non_event_prefixes: tuple[str, ...] = NON_EVENT_PREFIXES
    #: Ordered detector-prose patterns mapped to the class a cohort counts them under.
    component_label_patterns: tuple[tuple[str, str], ...] = DEFAULT_COMPONENT_LABEL_PATTERNS

    @property
    def plausible_rr_range_s(self) -> tuple[float, float]:
        """The interval range :attr:`plausible_heart_rate_bpm` implies, in seconds.

        Derived rather than stored, so the subject panel's shaded band and the cohort
        panel's bound cannot drift apart the way the two constants this replaced did.
        """
        low_bpm, high_bpm = self.plausible_heart_rate_bpm
        return (60.0 / high_bpm, 60.0 / low_bpm)

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
            aperiodic_exclude_hz=_pair_sequence(
                analysis,
                "aperiodic_exclude_hz",
                default=cls.aperiodic_exclude_hz,
                setting="report.analysis.aperiodic_exclude_hz",
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
            alpha_reference_band_hz=_pair(
                analysis,
                "alpha_reference_band_hz",
                default=cls.alpha_reference_band_hz,
                setting="report.analysis.alpha_reference_band_hz",
            ),
            plausible_heart_rate_bpm=_pair(
                thresholds,
                "plausible_heart_rate_bpm",
                default=cls.plausible_heart_rate_bpm,
                setting="report.thresholds.plausible_heart_rate_bpm",
            ),
            marker_agreement_tolerance_s=float(
                thresholds.get(
                    "marker_agreement_tolerance_s", cls.marker_agreement_tolerance_s
                )
            ),
            bcg_residual_window_s=_pair(
                analysis,
                "bcg_residual_window_s",
                default=cls.bcg_residual_window_s,
                setting="report.analysis.bcg_residual_window_s",
            ),
            bcg_residual_baseline_s=_pair(
                analysis,
                "bcg_residual_baseline_s",
                default=cls.bcg_residual_baseline_s,
                setting="report.analysis.bcg_residual_baseline_s",
            ),
            bcg_residual_measurement_s=_pair(
                analysis,
                "bcg_residual_measurement_s",
                default=cls.bcg_residual_measurement_s,
                setting="report.analysis.bcg_residual_measurement_s",
            ),
            notch_exclusion_half_width_hz=float(
                thresholds.get(
                    "notch_exclusion_half_width_hz", cls.notch_exclusion_half_width_hz
                )
            ),
            repetition_time_tolerance_s=float(
                thresholds.get("repetition_time_tolerance_s", cls.repetition_time_tolerance_s)
            ),
            channel_position_tolerance_m=float(
                thresholds.get("channel_position_tolerance_m", cls.channel_position_tolerance_m)
            ),
            min_subjects_for_median=int(
                thresholds.get("min_subjects_for_median", cls.min_subjects_for_median)
            ),
            min_subjects_for_outer_band=int(
                thresholds.get("min_subjects_for_outer_band", cls.min_subjects_for_outer_band)
            ),
            non_event_prefixes=_string_tuple(
                acquisition,
                "non_event_prefixes",
                default=cls.non_event_prefixes,
                setting="report.acquisition.non_event_prefixes",
            ),
            component_label_patterns=_label_patterns(
                acquisition,
                "component_label_patterns",
                default=cls.component_label_patterns,
                setting="report.acquisition.component_label_patterns",
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
        reference_low, reference_high = self.alpha_reference_band_hz
        if not 0 < reference_low < reference_high:
            raise ValueError(
                "report.analysis.alpha_reference_band_hz must satisfy 0 < low < high."
            )
        # The prominence is the peak's excess over a background fitted outside the band,
        # so a reference window that does not surround the band scores the peak against a
        # neighbourhood it does not have.
        if not (reference_low < alpha_low and alpha_high < reference_high):
            raise ValueError(
                "report.analysis.alpha_reference_band_hz must surround "
                f"report.analysis.alpha_band_hz; {self.alpha_reference_band_hz} does not "
                f"contain {self.alpha_band_hz}."
            )
        bpm_low, bpm_high = self.plausible_heart_rate_bpm
        if not 0 < bpm_low < bpm_high:
            raise ValueError(
                "report.thresholds.plausible_heart_rate_bpm must satisfy 0 < low < high."
            )
        if self.marker_agreement_tolerance_s <= 0:
            raise ValueError(
                "report.thresholds.marker_agreement_tolerance_s must be positive."
            )
        # Half the shortest plausible interval: a tolerance at or above it lets a marker
        # match the beat after the one it belongs to, which reports agreement that the two
        # detectors never had.
        shortest_interval_s = self.plausible_rr_range_s[0]
        if self.marker_agreement_tolerance_s >= shortest_interval_s / 2:
            raise ValueError(
                "report.thresholds.marker_agreement_tolerance_s must stay below half the "
                f"shortest plausible interval ({shortest_interval_s / 2:.3f} s at "
                f"{bpm_high:g} bpm), or one beat can match its neighbour."
            )
        window_start, window_stop = self.bcg_residual_window_s
        if window_stop <= window_start:
            raise ValueError(
                "report.analysis.bcg_residual_window_s must have a stop after its start."
            )
        for key, span in (
            ("bcg_residual_baseline_s", self.bcg_residual_baseline_s),
            ("bcg_residual_measurement_s", self.bcg_residual_measurement_s),
        ):
            start, stop = span
            if stop <= start:
                raise ValueError(
                    f"report.analysis.{key} must have a stop after its start."
                )
            # Silently no baseline, and silently no measurement, are the two ways this
            # panel can report a number that describes nothing.
            if start < window_start or stop > window_stop:
                raise ValueError(
                    f"report.analysis.{key} must lie inside "
                    f"report.analysis.bcg_residual_window_s {self.bcg_residual_window_s}; "
                    f"{span} does not."
                )
        if self.notch_exclusion_half_width_hz <= 0:
            raise ValueError(
                "report.thresholds.notch_exclusion_half_width_hz must be positive."
            )
        if self.repetition_time_tolerance_s <= 0:
            raise ValueError("report.thresholds.repetition_time_tolerance_s must be positive.")
        if self.channel_position_tolerance_m <= 0:
            raise ValueError("report.thresholds.channel_position_tolerance_m must be positive.")
        self._validate_subject_gates()
        if not self.non_event_prefixes:
            raise ValueError(
                "report.acquisition.non_event_prefixes must name at least the annotation "
                "prefixes this pipeline writes for itself; an empty list counts BAD spans "
                "as trials."
            )
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

    def _validate_subject_gates(self) -> None:
        """Reject cohort gates the sample could not support.

        Delegated to :class:`BandGates` rather than restated, so the arithmetic that
        decides whether a quantile is interior to the sample lives in one place and a
        config value cannot be accepted here and rejected there. Imported lazily: the
        cohort tree imports these settings back, and the aggregation machinery has no
        business loading whenever a subject report reads a threshold.
        """
        from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates

        try:
            BandGates(
                min_subjects_for_median=self.min_subjects_for_median,
                min_subjects_for_outer_band=self.min_subjects_for_outer_band,
            )
        except ValueError as error:
            raise ValueError(
                "report.thresholds.min_subjects_for_median / "
                f"min_subjects_for_outer_band: {error}"
            ) from error

    def band_gates(self) -> Any:
        """The cohort band gates these settings describe."""
        from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates

        return BandGates(
            min_subjects_for_median=self.min_subjects_for_median,
            min_subjects_for_outer_band=self.min_subjects_for_outer_band,
        )

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
            unavailable_intervals_by_recording=_unavailable_intervals(config),
        )


__all__ = ["ReportSettings"]
