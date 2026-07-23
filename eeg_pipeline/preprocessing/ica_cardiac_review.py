"""Direct ECG detection and run-resolved ICA cardiac evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mne
import numpy as np
import pandas as pd

PULSE_EVENT_ID = 999
ABRUPT_RR_CHANGE_FRACTION = 0.20


def _time_window(values: Any, *, path: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise TypeError(f"{path} must contain exactly two values.")
    window = (float(values[0]), float(values[1]))
    if window[0] >= window[1]:
        raise ValueError(f"{path} start must be earlier than its end.")
    return window


def _ctps_threshold(value: Any) -> str | float:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError("ica.cardiac_review.ctps_threshold must be 'auto' or numeric.")
        return value
    threshold = float(value)
    if not 0 < threshold <= 1:
        raise ValueError("ica.cardiac_review.ctps_threshold must be in (0, 1].")
    return threshold


def _string_tuple(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{path} must be a list of strings.")
    cleaned = tuple(item.strip() for item in value)
    if any(not item for item in cleaned) or len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{path} must contain unique, non-empty recording IDs.")
    return cleaned


@dataclass(frozen=True)
class CardiacReviewSettings:
    """Configuration for direct ECG detection and manual ICA review evidence."""

    enabled: bool = False
    ecg_channel: str = "ECG"
    epoch_window: tuple[float, float] = (-0.4, 0.6)
    baseline: tuple[float, float] = (-0.4, -0.1)
    measurement_window: tuple[float, float] = (0.0, 0.4)
    ctps_threshold: str | float = "auto"
    representative_window_seconds: float = 10.0
    plausible_heart_rate_bpm: tuple[float, float] = (40.0, 160.0)
    max_rr_outlier_fraction: float = 0.05
    min_template_correlation: float = 0.80
    accepted_questionable_runs: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CardiacReviewSettings:
        if not isinstance(values, Mapping):
            raise TypeError("ica.cardiac_review must be a mapping.")
        settings = cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            ecg_channel=str(values.get("ecg_channel", cls.ecg_channel)).strip(),
            epoch_window=_time_window(
                values.get("epoch_window", cls.epoch_window),
                path="ica.cardiac_review.epoch_window",
            ),
            baseline=_time_window(
                values.get("baseline", cls.baseline),
                path="ica.cardiac_review.baseline",
            ),
            measurement_window=_time_window(
                values.get("measurement_window", cls.measurement_window),
                path="ica.cardiac_review.measurement_window",
            ),
            ctps_threshold=_ctps_threshold(values.get("ctps_threshold", cls.ctps_threshold)),
            representative_window_seconds=float(
                values.get(
                    "representative_window_seconds",
                    cls.representative_window_seconds,
                )
            ),
            plausible_heart_rate_bpm=_time_window(
                values.get("plausible_heart_rate_bpm", cls.plausible_heart_rate_bpm),
                path="ica.cardiac_review.plausible_heart_rate_bpm",
            ),
            max_rr_outlier_fraction=float(
                values.get("max_rr_outlier_fraction", cls.max_rr_outlier_fraction)
            ),
            min_template_correlation=float(
                values.get("min_template_correlation", cls.min_template_correlation)
            ),
            accepted_questionable_runs=_string_tuple(
                values.get("accepted_questionable_runs", list(cls.accepted_questionable_runs)),
                path="ica.cardiac_review.accepted_questionable_runs",
            ),
        )
        if not settings.ecg_channel:
            raise ValueError("ica.cardiac_review.ecg_channel must not be empty.")
        epoch_start, epoch_end = settings.epoch_window
        for name, window in (
            ("baseline", settings.baseline),
            ("measurement_window", settings.measurement_window),
        ):
            if window[0] < epoch_start or window[1] > epoch_end:
                raise ValueError(f"ica.cardiac_review.{name} must lie inside epoch_window.")
        if settings.representative_window_seconds <= 0:
            raise ValueError("ica.cardiac_review.representative_window_seconds must be positive.")
        if not 0 <= settings.max_rr_outlier_fraction <= 1:
            raise ValueError("ica.cardiac_review.max_rr_outlier_fraction must be in [0, 1].")
        if not 0 <= settings.min_template_correlation <= 1:
            raise ValueError("ica.cardiac_review.min_template_correlation must be in [0, 1].")
        return settings


@dataclass(frozen=True)
class EcgDetection:
    """R peaks detected directly from an ECG channel."""

    events: np.ndarray
    average_pulse_bpm: float


@dataclass(frozen=True)
class EcgDetectionQuality:
    """Physiological and waveform-consistency checks for one run's R peaks."""

    median_heart_rate_bpm: float
    rr_outlier_fraction: float
    abrupt_rr_change_fraction: float
    median_template_correlation: float
    reliable: bool
    reason: str


@dataclass(frozen=True)
class RunCardiacReview:
    """Run-level ECG and cardiac-locked EEG evidence."""

    recording_id: str
    representative_times: np.ndarray
    representative_ecg_mv: np.ndarray
    representative_peak_times: np.ndarray
    rr_times: np.ndarray
    heart_rate_bpm: np.ndarray
    locked_times: np.ndarray
    locked_ecg_mv: np.ndarray
    before_gfp_uv: np.ndarray
    after_gfp_uv: np.ndarray
    attenuation_percent: float
    heartbeat_count: int
    average_pulse_bpm: float
    events: np.ndarray
    quality: EcgDetectionQuality
    included_in_component_review: bool
    before_topography_uv: np.ndarray
    after_topography_uv: np.ndarray
    topography_time: float


@dataclass(frozen=True)
class ComponentCardiacReview:
    """R-locked and score-based evidence for each ICA component."""

    run_ids: tuple[str, ...]
    times: np.ndarray
    run_mean_z: np.ndarray
    abs_correlations: np.ndarray
    ctps_scores: np.ndarray
    correlation_flags: np.ndarray
    ctps_flags: np.ndarray
    heartbeat_counts: np.ndarray


def _validate_ecg_channel(raw: mne.io.BaseRaw, channel: str) -> None:
    if channel not in raw.ch_names:
        raise ValueError(f"ECG review requires channel {channel!r}.")
    channel_type = raw.get_channel_types(picks=[channel])[0]
    if channel_type != "ecg":
        raise ValueError(
            f"ECG review channel {channel!r} has type {channel_type!r}, expected 'ecg'."
        )


def detect_ecg_events(
    raw: mne.io.BaseRaw,
    settings: CardiacReviewSettings,
) -> EcgDetection:
    """Detect R peaks from the ECG signal independently of event annotations."""
    _validate_ecg_channel(raw, settings.ecg_channel)
    events, _, average_pulse_bpm, _ = mne.preprocessing.find_ecg_events(
        raw,
        ch_name=settings.ecg_channel,
        event_id=PULSE_EVENT_ID,
        return_ecg=True,
        verbose="ERROR",
    )
    if len(events) < 3:
        raise ValueError(
            f"Direct ECG detection found only {len(events)} R peaks; at least 3 are required."
        )
    if not np.isfinite(average_pulse_bpm) or average_pulse_bpm <= 0:
        raise ValueError(f"Invalid average pulse estimate: {average_pulse_bpm!r}.")
    return EcgDetection(
        events=np.asarray(events, dtype=int),
        average_pulse_bpm=float(average_pulse_bpm),
    )


def standardize_source_epoch_runs(
    data_runs: list[np.ndarray],
    *,
    times: np.ndarray,
    baseline: tuple[float, float],
) -> list[np.ndarray]:
    """Baseline-center epochs and apply one robust component scale across runs."""
    if not data_runs:
        raise ValueError("At least one ICA source epoch run is required.")
    sample_times = np.asarray(times, dtype=float)
    baseline_mask = (sample_times >= baseline[0]) & (sample_times <= baseline[1])
    if baseline_mask.sum() < 2:
        raise ValueError("Cardiac review baseline must contain at least two samples.")

    centered_runs = []
    component_count = None
    for data in data_runs:
        values = np.asarray(data, dtype=float)
        if values.ndim != 3 or values.shape[-1] != len(sample_times):
            raise ValueError(
                "ICA source epochs must have shape (epochs, components, matching times)."
            )
        if component_count is None:
            component_count = values.shape[1]
        elif values.shape[1] != component_count:
            raise ValueError("ICA source runs have inconsistent component counts.")
        baseline_mean = values[..., baseline_mask].mean(axis=-1, keepdims=True)
        centered_runs.append(values - baseline_mean)

    pooled_baseline = np.concatenate(
        [values[..., baseline_mask] for values in centered_runs],
        axis=0,
    )
    component_baseline = np.moveaxis(pooled_baseline, 1, 0).reshape(component_count, -1)
    component_median = np.median(component_baseline, axis=1, keepdims=True)
    robust_scale = 1.4826 * np.median(
        np.abs(component_baseline - component_median),
        axis=1,
    )
    if np.any(~np.isfinite(robust_scale)) or np.any(robust_scale <= 0):
        raise ValueError("ICA source data contain a zero or invalid robust baseline scale.")
    return [values / robust_scale[None, :, None] for values in centered_runs]


def assess_ecg_detection_quality(
    peak_times: np.ndarray,
    *,
    template_correlations: np.ndarray,
    settings: CardiacReviewSettings,
) -> EcgDetectionQuality:
    """Grade R-peak timing and ECG-template consistency for one run."""
    peaks = np.asarray(peak_times, dtype=float)
    correlations = np.asarray(template_correlations, dtype=float)
    if peaks.ndim != 1 or len(peaks) < 3 or np.any(np.diff(peaks) <= 0):
        raise ValueError("ECG peak times must be a strictly increasing one-dimensional array.")
    if correlations.ndim != 1 or len(correlations) != len(peaks):
        raise ValueError("ECG template correlations must match the detected peak count.")
    if np.any(~np.isfinite(correlations)):
        raise ValueError("ECG template correlations contain non-finite values.")

    rr_intervals = np.diff(peaks)
    heart_rate = 60.0 / rr_intervals
    minimum_bpm, maximum_bpm = settings.plausible_heart_rate_bpm
    rr_outliers = (heart_rate < minimum_bpm) | (heart_rate > maximum_bpm)
    relative_rr_change = np.abs(np.diff(rr_intervals)) / rr_intervals[:-1]
    rr_outlier_fraction = float(rr_outliers.mean())
    median_template_correlation = float(np.median(correlations))
    issues = []
    if rr_outlier_fraction > settings.max_rr_outlier_fraction:
        issues.append("implausible RR intervals")
    if median_template_correlation < settings.min_template_correlation:
        issues.append("inconsistent ECG templates")
    return EcgDetectionQuality(
        median_heart_rate_bpm=float(np.median(heart_rate)),
        rr_outlier_fraction=rr_outlier_fraction,
        abrupt_rr_change_fraction=float(np.mean(relative_rr_change > ABRUPT_RR_CHANGE_FRACTION)),
        median_template_correlation=median_template_correlation,
        reliable=not issues,
        reason="; ".join(issues) if issues else "passed configured quality checks",
    )


def include_run_in_component_review(
    recording_id: str,
    quality: EcgDetectionQuality,
    settings: CardiacReviewSettings,
) -> bool:
    """Include reliable runs or exact recording IDs explicitly accepted by the user."""
    return quality.reliable or recording_id in settings.accepted_questionable_runs


def component_cardiac_evidence_table(
    review: ComponentCardiacReview,
    *,
    statuses: pd.DataFrame,
) -> pd.DataFrame:
    """Create review evidence without assigning or changing ICA exclusions."""
    run_count, component_count, _ = review.run_mean_z.shape
    expected_shape = (run_count, component_count)
    score_arrays = (
        review.abs_correlations,
        review.ctps_scores,
        review.correlation_flags,
        review.ctps_flags,
    )
    if any(array.shape != expected_shape for array in score_arrays):
        raise ValueError("Cardiac component evidence has inconsistent component dimensions.")
    expected_components = np.arange(component_count)
    required_columns = {"component", "status", "status_description"}
    if not required_columns.issubset(statuses.columns) or not np.array_equal(
        statuses["component"].to_numpy(), expected_components
    ):
        raise ValueError("ICA component statuses do not align with cardiac evidence.")
    correlation_count = review.correlation_flags.sum(axis=0)
    ctps_count = review.ctps_flags.sum(axis=0)
    evidence = pd.DataFrame(
        {
            "component": expected_components,
            "median_abs_ecg_correlation": np.median(review.abs_correlations, axis=0),
            "max_abs_ecg_correlation": np.max(review.abs_correlations, axis=0),
            "correlation_flagged_run_count": correlation_count,
            "median_ctps_score": np.median(review.ctps_scores, axis=0),
            "max_ctps_score": np.max(review.ctps_scores, axis=0),
            "ctps_flagged_run_count": ctps_count,
            "manual_review_recommended": (correlation_count + ctps_count) > 0,
            "included_run_count": run_count,
            "heartbeat_count": int(review.heartbeat_counts.sum()),
        }
    )
    current = statuses[["component", "status", "status_description"]].rename(
        columns={
            "status": "current_ica_status",
            "status_description": "current_status_description",
        }
    )
    return evidence.merge(current, on="component", validate="one_to_one")


def component_run_cardiac_evidence_table(
    review: ComponentCardiacReview,
) -> pd.DataFrame:
    """Create tidy run-resolved ECG evidence for every ICA component."""
    rows = []
    for run_index, recording_id in enumerate(review.run_ids):
        for component in range(review.run_mean_z.shape[1]):
            rows.append(
                {
                    "recording_id": recording_id,
                    "component": component,
                    "abs_ecg_correlation": review.abs_correlations[run_index, component],
                    "ctps_score": review.ctps_scores[run_index, component],
                    "mne_correlation_flag": review.correlation_flags[run_index, component],
                    "ctps_flag": review.ctps_flags[run_index, component],
                    "heartbeat_count": review.heartbeat_counts[run_index],
                }
            )
    return pd.DataFrame(rows)


def _heartbeat_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    ica: mne.preprocessing.ICA,
    settings: CardiacReviewSettings,
) -> mne.BaseEpochs:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=PULSE_EVENT_ID,
        tmin=settings.epoch_window[0],
        tmax=settings.epoch_window[1],
        baseline=None,
        picks=[*ica.ch_names, settings.ecg_channel],
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("No valid directly detected R-locked epochs remain.")
    return epochs


def _cardiac_locked_gfp(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    settings: CardiacReviewSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    referenced = raw.copy().set_eeg_reference("average", projection=False, verbose=False)
    epochs = mne.Epochs(
        referenced,
        events,
        event_id=PULSE_EVENT_ID,
        tmin=settings.epoch_window[0],
        tmax=settings.epoch_window[1],
        baseline=settings.baseline,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("No valid directly detected R-locked EEG epochs remain.")
    evoked_uv = epochs.average().data * 1e6
    return (
        epochs.times.copy(),
        np.sqrt(np.mean(evoked_uv**2, axis=0)),
        evoked_uv,
    )


def _ecg_template_correlations(
    epochs: mne.BaseEpochs,
    *,
    channel: str,
) -> np.ndarray:
    template_window = (epochs.times >= -0.1) & (epochs.times <= 0.2)
    if template_window.sum() < 3:
        raise ValueError("ECG template window contains fewer than three samples.")
    waveforms = epochs.get_data(picks=[channel])[:, 0, template_window]
    waveform_std = waveforms.std(axis=1, keepdims=True)
    if np.any(~np.isfinite(waveform_std)) or np.any(waveform_std <= 0):
        raise ValueError("ECG epochs contain a zero or invalid template standard deviation.")
    standardized = (waveforms - waveforms.mean(axis=1, keepdims=True)) / waveform_std
    template = np.median(standardized, axis=0)
    template_std = template.std()
    if not np.isfinite(template_std) or template_std <= 0:
        raise ValueError("Median ECG template has a zero or invalid standard deviation.")
    standardized_template = (template - template.mean()) / template_std
    return np.mean(standardized * standardized_template[None, :], axis=1)


def _representative_ecg(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    settings: CardiacReviewSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    peak_times = (events[:, 0] - raw.first_samp) / sfreq
    duration = raw.times[-1]
    half_window = settings.representative_window_seconds / 2.0
    center = float(np.median(peak_times))
    start = min(max(0.0, center - half_window), max(0.0, duration - 2.0 * half_window))
    stop = min(duration, start + settings.representative_window_seconds)
    sample_mask = (raw.times >= start) & (raw.times <= stop)
    peak_mask = (peak_times >= start) & (peak_times <= stop)
    return (
        raw.times[sample_mask],
        raw.get_data(picks=[settings.ecg_channel])[0, sample_mask] * 1e3,
        peak_times[peak_mask],
    )


def _build_run_cardiac_review(
    raw: mne.io.BaseRaw,
    *,
    ica: mne.preprocessing.ICA,
    recording_id: str,
    settings: CardiacReviewSettings,
) -> RunCardiacReview:
    detection = detect_ecg_events(raw, settings)
    before_times, before_gfp, before_evoked = _cardiac_locked_gfp(
        raw,
        detection.events,
        settings=settings,
    )
    after = raw.copy()
    ica.apply(after, verbose="ERROR")
    after_eeg = after.get_data(picks="eeg")
    if np.any(~np.isfinite(after_eeg)):
        raise ValueError(f"{recording_id}: ICA application produced non-finite EEG values.")
    after_times, after_gfp, after_evoked = _cardiac_locked_gfp(
        after,
        detection.events,
        settings=settings,
    )
    if not np.array_equal(before_times, after_times):
        raise ValueError(f"{recording_id}: before/after cardiac epoch times do not align.")
    measurement_mask = (before_times >= settings.measurement_window[0]) & (
        before_times <= settings.measurement_window[1]
    )
    if not measurement_mask.any():
        raise ValueError("Cardiac review measurement window contains no samples.")
    before_rms = float(np.sqrt(np.mean(before_gfp[measurement_mask] ** 2)))
    after_rms = float(np.sqrt(np.mean(after_gfp[measurement_mask] ** 2)))
    if before_rms == 0:
        raise ValueError(f"{recording_id}: pre-ICA cardiac-locked EEG RMS is zero.")

    representative_times, representative_ecg, representative_peaks = _representative_ecg(
        raw,
        detection.events,
        settings=settings,
    )
    sfreq = float(raw.info["sfreq"])
    peak_times = (detection.events[:, 0] - raw.first_samp) / sfreq
    rr_intervals = np.diff(peak_times)
    if np.any(rr_intervals <= 0):
        raise ValueError(f"{recording_id}: detected R peaks are not strictly increasing.")
    ecg_epochs = _heartbeat_epochs(
        raw,
        detection.events,
        ica=ica,
        settings=settings,
    )
    locked_ecg = ecg_epochs.get_data(picks=[settings.ecg_channel]).mean(axis=0)[0] * 1e3
    retained_peak_times = (ecg_epochs.events[:, 0] - raw.first_samp) / float(raw.info["sfreq"])
    quality = assess_ecg_detection_quality(
        retained_peak_times,
        template_correlations=_ecg_template_correlations(
            ecg_epochs,
            channel=settings.ecg_channel,
        ),
        settings=settings,
    )
    measurement_indices = np.flatnonzero(measurement_mask)
    peak_index = measurement_indices[np.argmax(before_gfp[measurement_mask])]
    return RunCardiacReview(
        recording_id=recording_id,
        representative_times=representative_times,
        representative_ecg_mv=representative_ecg,
        representative_peak_times=representative_peaks,
        rr_times=peak_times[1:],
        heart_rate_bpm=60.0 / rr_intervals,
        locked_times=ecg_epochs.times.copy(),
        locked_ecg_mv=locked_ecg,
        before_gfp_uv=before_gfp,
        after_gfp_uv=after_gfp,
        attenuation_percent=100.0 * (1.0 - after_rms / before_rms),
        heartbeat_count=len(ecg_epochs),
        average_pulse_bpm=detection.average_pulse_bpm,
        events=ecg_epochs.events.copy(),
        quality=quality,
        included_in_component_review=include_run_in_component_review(
            recording_id,
            quality,
            settings,
        ),
        before_topography_uv=before_evoked[:, peak_index],
        after_topography_uv=after_evoked[:, peak_index],
        topography_time=float(before_times[peak_index]),
    )


def _build_component_cardiac_review(
    raws: list[mne.io.BaseRaw],
    run_reviews: list[RunCardiacReview],
    *,
    ica: mne.preprocessing.ICA,
    settings: CardiacReviewSettings,
) -> ComponentCardiacReview:
    included = [
        (raw, review)
        for raw, review in zip(raws, run_reviews, strict=True)
        if review.included_in_component_review
    ]
    if not included:
        raise ValueError(
            "No ECG runs passed quality checks or were listed in accepted_questionable_runs."
        )
    source_data_runs = []
    run_ids = []
    heartbeat_counts = []
    correlation_scores = []
    correlation_flags = []
    ctps_scores = []
    ctps_flags = []
    source_times = None
    for raw, review in included:
        epochs = _heartbeat_epochs(raw, review.events, ica=ica, settings=settings)
        correlation_components, run_correlation_scores = ica.find_bads_ecg(
            raw,
            ch_name=settings.ecg_channel,
            method="correlation",
            threshold="auto",
            verbose="ERROR",
        )
        run_ctps_components, run_ctps_scores = ica.find_bads_ecg(
            epochs,
            ch_name=settings.ecg_channel,
            method="ctps",
            threshold=settings.ctps_threshold,
            verbose="ERROR",
        )
        run_correlation_scores = np.asarray(run_correlation_scores, dtype=float)
        run_ctps_scores = np.asarray(run_ctps_scores, dtype=float)
        if np.any(~np.isfinite(run_correlation_scores)) or np.any(~np.isfinite(run_ctps_scores)):
            raise ValueError(
                f"{review.recording_id}: ICA cardiac scores contain non-finite values."
            )
        sources = ica.get_sources(epochs)
        if source_times is None:
            source_times = sources.times.copy()
        elif not np.array_equal(source_times, sources.times):
            raise ValueError("R-locked ICA source times differ between runs.")
        run_correlation_flags = np.zeros(int(ica.n_components_), dtype=bool)
        run_correlation_flags[np.asarray(correlation_components, dtype=int)] = True
        run_ctps_flags = np.zeros(int(ica.n_components_), dtype=bool)
        run_ctps_flags[np.asarray(run_ctps_components, dtype=int)] = True
        run_ids.append(review.recording_id)
        heartbeat_counts.append(len(epochs))
        source_data_runs.append(sources.get_data(copy=False))
        correlation_scores.append(np.abs(run_correlation_scores))
        correlation_flags.append(run_correlation_flags)
        ctps_scores.append(run_ctps_scores)
        ctps_flags.append(run_ctps_flags)

    standardized_runs = standardize_source_epoch_runs(
        source_data_runs,
        times=source_times,
        baseline=settings.baseline,
    )
    return ComponentCardiacReview(
        run_ids=tuple(run_ids),
        times=source_times,
        run_mean_z=np.stack([values.mean(axis=0) for values in standardized_runs]),
        abs_correlations=np.stack(correlation_scores),
        ctps_scores=np.stack(ctps_scores),
        correlation_flags=np.stack(correlation_flags),
        ctps_flags=np.stack(ctps_flags),
        heartbeat_counts=np.asarray(heartbeat_counts, dtype=int),
    )


__all__ = [
    "CardiacReviewSettings",
    "ComponentCardiacReview",
    "EcgDetection",
    "EcgDetectionQuality",
    "RunCardiacReview",
    "assess_ecg_detection_quality",
    "component_cardiac_evidence_table",
    "component_run_cardiac_evidence_table",
    "detect_ecg_events",
    "include_run_in_component_review",
    "standardize_source_epoch_runs",
]
