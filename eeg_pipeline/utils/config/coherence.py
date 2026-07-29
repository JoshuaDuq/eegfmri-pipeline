"""Config-level contradictions, all reported at once and before any work starts.

Every check here answers a question that the config alone settles: nothing in this module
opens a recording. That is the point. The failures it covers were all *already* detected
somewhere — but one at a time, each from the stage that happened to trip over it, and in
two cases only after ICA fitting had run. Setting up a resting-state study meant fixing a
key, waiting, fixing the next key it named, and waiting again.

Two severities, and the difference is whether the user has to do anything:

``errors``
    The run cannot produce what was asked for. Raised before the first step.

``warnings``
    Something is switched on that this dataset gives the pipeline no way to compute, so
    it will be skipped. The run is fine; the config says more than it means. These are
    reported rather than raised because they are the expected state of a config derived
    from another study's — and because silently skipping a requested stage, with nothing
    said, is the failure mode that hid an absent ocular detection for months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from eeg_pipeline.utils.config.acquisition import is_eeg_fmri
from eeg_pipeline.utils.config.loader import get_config_value

_MISSING = object()


@dataclass(frozen=True)
class ConfigIssue:
    """One contradiction, named by the key that has to change to resolve it."""

    key: str
    message: str

    def __str__(self) -> str:
        return f"{self.key}: {self.message}"


@dataclass(frozen=True)
class CoherenceReport:
    errors: Tuple[ConfigIssue, ...]
    warnings: Tuple[ConfigIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        """Raise once, listing every error, rather than once per key discovered."""
        if not self.errors:
            return
        lines = "\n".join(f"  - {issue}" for issue in self.errors)
        raise ValueError(
            f"The configuration cannot produce the requested run:\n{lines}\n"
            "Fix these keys and rerun. Run 'eeg-pipeline validate --config-only' to "
            "recheck without processing anything."
        )

    def log_warnings(self, logger: Any) -> None:
        for issue in self.warnings:
            logger.info("Config: %s", issue)


def _resting_state_flags(config: Any) -> Tuple[Any, Any]:
    return (
        get_config_value(config, "preprocessing.task_is_rest", _MISSING),
        get_config_value(config, "feature_engineering.task_is_rest", _MISSING),
    )


def _check_paradigm_agreement(config: Any, errors: List[ConfigIssue]) -> bool:
    """Return the resolved resting-state mode, recording a mismatch rather than raising.

    ``project.paradigm`` sets both flags when present, so a mismatch here means they were
    set individually. Reported as an error like any other, so it appears alongside the
    rest of the report instead of aborting it.
    """
    preprocessing_raw, feature_raw = _resting_state_flags(config)

    if preprocessing_raw is _MISSING and feature_raw is _MISSING:
        return False
    if preprocessing_raw is _MISSING:
        return bool(feature_raw)
    if feature_raw is _MISSING:
        return bool(preprocessing_raw)

    if bool(preprocessing_raw) != bool(feature_raw):
        errors.append(
            ConfigIssue(
                "project.paradigm",
                "preprocessing.task_is_rest is "
                f"{bool(preprocessing_raw)} but feature_engineering.task_is_rest is "
                f"{bool(feature_raw)}; the two must agree. Set project.paradigm to "
                "'task' or 'rest' and both are derived from it.",
            )
        )
    return bool(preprocessing_raw)


def _check_rest_settings(config: Any, errors: List[ConfigIssue]) -> None:
    """Settings that a fixed-length resting-state run cannot satisfy.

    Each of these already raised from inside a stage. They are checked here so that a
    config needing three changes reports three, once, instead of stopping at the first
    and re-running ICA to find the second.
    """
    if bool(get_config_value(config, "ica.band_specific_report.enabled", False)):
        if bool(get_config_value(config, "ica.band_specific_report.tfr.enabled", True)):
            errors.append(
                ConfigIssue(
                    "ica.band_specific_report.tfr.enabled",
                    "resting-state epochs are fixed-length segments with no event and no "
                    "pre-stimulus interval, so the baseline-relative component TFR has no "
                    "baseline. Set it false to review components by topography and spectrum.",
                )
            )
        if get_config_value(config, "ica.band_specific_report.comparisons", None):
            errors.append(
                ConfigIssue(
                    "ica.band_specific_report.comparisons",
                    "band-specific condition comparisons need event-related task epochs. "
                    "Set it to an empty list for resting-state.",
                )
            )

    duration = get_config_value(config, "preprocessing.rest_epochs_duration", None)
    if duration is None:
        errors.append(
            ConfigIssue(
                "preprocessing.rest_epochs_duration",
                "resting-state preprocessing segments the recording into fixed-length "
                "epochs and needs their duration in seconds.",
            )
        )
    else:
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            duration_value = float("nan")
        if not duration_value > 0:
            errors.append(
                ConfigIssue(
                    "preprocessing.rest_epochs_duration",
                    f"must be greater than 0, got {duration!r}.",
                )
            )

    overlap = get_config_value(config, "preprocessing.rest_epochs_overlap", 0.0)
    try:
        overlap_value = float(overlap)
    except (TypeError, ValueError):
        overlap_value = float("nan")
    if not overlap_value == 0.0:
        errors.append(
            ConfigIssue(
                "preprocessing.rest_epochs_overlap",
                "overlapping segments would be carried downstream as independent trial "
                f"rows, so only 0 is supported; got {overlap!r}.",
            )
        )


def _check_task_settings(config: Any, errors: List[ConfigIssue]) -> None:
    task = get_config_value(config, "project.task", None)
    if task is None or not str(task).strip():
        errors.append(
            ConfigIssue(
                "project.task",
                "event-related preprocessing needs the BIDS task label to select "
                "recordings. Set it, or set project.paradigm to 'rest'.",
            )
        )


def _check_scanner_settings(config: Any, warnings: List[ConfigIssue]) -> None:
    """Scanner-only stages left switched on for a dataset recorded outside one.

    None of these stops the run — each is gated at its own call site — but a config that
    still asks for four things it will not get is a config nobody has finished adapting,
    and the listing is how the reader finds out which four.
    """
    scanner_only = (
        (
            "preprocessing.brainvision_analyzer.enabled",
            "no Analyzer correction precedes an out-of-scanner recording, so the pulse "
            "marker and cardiac attenuation QC will be skipped.",
        ),
        (
            "ica.cardiac_review.enabled",
            "there is no ballistocardiogram to measure outside a scanner, so the ICA "
            "cardiac review will be skipped.",
        ),
        (
            "preprocessing.clean_events_qc.ecg_coupling.enabled",
            "this correlates EEG against a recorded ECG lead, which out-of-scanner "
            "montages do not carry, so the metric will be skipped.",
        ),
        (
            "alignment.trim_to_volume_bounds",
            "there are no scanner volume markers to trim to, so no trimming will occur.",
        ),
    )
    for key, explanation in scanner_only:
        if bool(get_config_value(config, key, False)):
            warnings.append(
                ConfigIssue(
                    key,
                    f"is true, but preprocessing.eeg_fmri is false: {explanation} "
                    "Set it false to say so in the config.",
                )
            )


def check_config_coherence(config: Any) -> CoherenceReport:
    """Check every config-only contradiction and report them together."""
    errors: List[ConfigIssue] = []
    warnings: List[ConfigIssue] = []

    task_is_rest = _check_paradigm_agreement(config, errors)

    if task_is_rest:
        _check_rest_settings(config, errors)
    else:
        _check_task_settings(config, errors)

    if not is_eeg_fmri(config):
        _check_scanner_settings(config, warnings)

    return CoherenceReport(errors=tuple(errors), warnings=tuple(warnings))


__all__ = [
    "CoherenceReport",
    "ConfigIssue",
    "check_config_coherence",
]
