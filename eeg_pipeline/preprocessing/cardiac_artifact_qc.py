"""Cardiac artifact detection and ICA review quality control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION

PULSE_EVENT_ID = 999


def _time_window(values: Any, *, path: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise TypeError(f"{path} must contain exactly two values.")
    window = (float(values[0]), float(values[1]))
    if window[0] >= window[1]:
        raise ValueError(f"{path} start must be earlier than its end.")
    return window


@dataclass(frozen=True)
class CardiacReviewSettings:
    """Configuration for direct ECG detection and manual ICA review evidence."""

    enabled: bool = False
    ecg_channel: str = "ECG"
    epoch_window: tuple[float, float] = (-0.4, 0.6)
    baseline: tuple[float, float] = (-0.4, -0.1)
    measurement_window: tuple[float, float] = (0.0, 0.4)
    ctps_threshold: float = 0.25
    representative_window_seconds: float = 10.0

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
            ctps_threshold=float(values.get("ctps_threshold", cls.ctps_threshold)),
            representative_window_seconds=float(
                values.get(
                    "representative_window_seconds",
                    cls.representative_window_seconds,
                )
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
        if not 0 < settings.ctps_threshold <= 1:
            raise ValueError("ica.cardiac_review.ctps_threshold must be in (0, 1].")
        if settings.representative_window_seconds <= 0:
            raise ValueError("ica.cardiac_review.representative_window_seconds must be positive.")
        return settings


@dataclass(frozen=True)
class EcgDetection:
    """R peaks detected directly from an ECG channel."""

    events: np.ndarray
    average_pulse_bpm: float


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


@dataclass(frozen=True)
class ComponentCardiacReview:
    """R-locked and score-based evidence for each ICA component."""

    times: np.ndarray
    mean_z: np.ndarray
    ci95_z: np.ndarray
    median_abs_correlation: np.ndarray
    ctps_scores: np.ndarray
    correlation_flags: np.ndarray
    ctps_flags: np.ndarray
    heartbeat_count: int


@dataclass(frozen=True)
class CardiacAttenuationMetrics:
    """Run-level marker-locked attenuation after ICA application."""

    recording_id: str
    marker_count: int
    before_rms_uv: float
    after_rms_uv: float
    attenuation_percent: float


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


def standardize_source_epochs(
    data: np.ndarray,
    *,
    times: np.ndarray,
    baseline: tuple[float, float],
) -> np.ndarray:
    """Standardize each component epoch using its own pre-R baseline."""
    values = np.asarray(data, dtype=float)
    sample_times = np.asarray(times, dtype=float)
    if values.ndim != 3:
        raise ValueError("ICA source epochs must have shape (epochs, components, times).")
    if values.shape[-1] != len(sample_times):
        raise ValueError("ICA source data and time vector lengths differ.")
    baseline_mask = (sample_times >= baseline[0]) & (sample_times <= baseline[1])
    if baseline_mask.sum() < 2:
        raise ValueError("Cardiac review baseline must contain at least two samples.")
    baseline_values = values[..., baseline_mask]
    baseline_mean = baseline_values.mean(axis=-1, keepdims=True)
    baseline_std = baseline_values.std(axis=-1, keepdims=True)
    if np.any(~np.isfinite(baseline_std)) or np.any(baseline_std <= 0):
        raise ValueError("ICA source epochs contain a zero or invalid baseline standard deviation.")
    return (values - baseline_mean) / baseline_std


def component_cardiac_evidence_table(
    review: ComponentCardiacReview,
) -> pd.DataFrame:
    """Create review evidence without assigning or changing ICA exclusions."""
    component_count = review.mean_z.shape[0]
    arrays = (
        review.ci95_z,
        review.median_abs_correlation,
        review.ctps_scores,
        review.correlation_flags,
        review.ctps_flags,
    )
    expected_lengths = (component_count, component_count, component_count, component_count)
    actual_lengths = tuple(len(array) for array in arrays[1:])
    if review.ci95_z.shape != review.mean_z.shape or actual_lengths != expected_lengths:
        raise ValueError("Cardiac component evidence has inconsistent component dimensions.")
    recommended = np.asarray(review.correlation_flags) | np.asarray(review.ctps_flags)
    return pd.DataFrame(
        {
            "component": np.arange(component_count),
            "median_abs_ecg_correlation": review.median_abs_correlation,
            "ctps_score": review.ctps_scores,
            "mne_correlation_flag": review.correlation_flags,
            "ctps_flag": review.ctps_flags,
            "manual_review_recommended": recommended,
            "heartbeat_count": review.heartbeat_count,
        }
    )


def pulse_marker_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """Create MNE events from preserved BrainVision Analyzer R annotations."""
    events, _ = mne.events_from_annotations(
        raw,
        event_id={PULSE_MARKER_DESCRIPTION: PULSE_EVENT_ID},
        use_rounding=True,
        verbose="ERROR",
    )
    if len(events) == 0:
        raise ValueError(f"Raw recording contains no {PULSE_MARKER_DESCRIPTION!r} markers.")
    return events


def _marker_locked_rms(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> float:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=PULSE_EVENT_ID,
        tmin=baseline[0],
        tmax=measurement_window[1],
        baseline=baseline,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("No valid marker-locked EEG epochs remain for cardiac QC.")
    evoked = epochs.average()
    window = (evoked.times >= measurement_window[0]) & (evoked.times <= measurement_window[1])
    if not np.any(window):
        raise ValueError("Cardiac QC measurement window contains no samples.")
    return float(np.sqrt(np.mean(evoked.data[:, window] ** 2)))


def compute_cardiac_attenuation(
    before: mne.io.BaseRaw,
    after: mne.io.BaseRaw,
    *,
    recording_id: str,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> CardiacAttenuationMetrics:
    """Measure cardiac-locked EEG RMS before and after ICA."""
    recordings_align = (
        before.ch_names == after.ch_names
        and before.n_times == after.n_times
        and before.first_samp == after.first_samp
        and before.info["sfreq"] == after.info["sfreq"]
    )
    if not recordings_align:
        raise ValueError(f"{recording_id}: before/after raw recordings do not align.")

    before_referenced = before.copy().set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )
    after_referenced = after.copy().set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )
    events = pulse_marker_events(before_referenced)
    before_rms = _marker_locked_rms(
        before_referenced,
        events,
        baseline=baseline,
        measurement_window=measurement_window,
    )
    after_rms = _marker_locked_rms(
        after_referenced,
        events,
        baseline=baseline,
        measurement_window=measurement_window,
    )
    if before_rms == 0:
        raise ValueError(f"{recording_id}: pre-ICA marker-locked EEG RMS is zero.")
    attenuation_percent = 100.0 * (1.0 - after_rms / before_rms)
    return CardiacAttenuationMetrics(
        recording_id=recording_id,
        marker_count=len(events),
        before_rms_uv=before_rms * 1e6,
        after_rms_uv=after_rms * 1e6,
        attenuation_percent=attenuation_percent,
    )


def add_marker_ctps_columns(
    components: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Add marker-based CTPS evidence without changing exclusion statuses."""
    expected_components = np.arange(len(scores))
    if "component" not in components or not np.array_equal(
        components["component"].to_numpy(), expected_components
    ):
        raise ValueError("ICA component table does not match the marker CTPS scores.")
    if not 0 < threshold <= 1:
        raise ValueError(f"CTPS threshold must be in (0, 1], got {threshold}.")

    result = components.copy()
    result["analyzer_marker_ctps_score"] = np.asarray(scores, dtype=float)
    result["analyzer_marker_ctps_flag"] = np.asarray(scores) >= threshold
    return result


def compute_marker_ctps_scores(
    raws: Iterable[mne.io.BaseRaw],
    ica: mne.preprocessing.ICA,
    *,
    threshold: float,
    epoch_window: tuple[float, float],
) -> np.ndarray:
    """Score ICA components using CTPS epochs anchored to Analyzer markers."""
    marker_epochs = []
    for raw in raws:
        marker_epochs.append(
            mne.Epochs(
                raw,
                pulse_marker_events(raw),
                event_id=PULSE_EVENT_ID,
                tmin=epoch_window[0],
                tmax=epoch_window[1],
                baseline=None,
                picks="eeg",
                preload=True,
                reject_by_annotation=True,
                verbose="ERROR",
            )
        )
    if not marker_epochs:
        raise ValueError("No filtered raw recordings were provided for marker CTPS QC.")
    epochs = mne.concatenate_epochs(marker_epochs, verbose="ERROR")
    if len(epochs) == 0:
        raise ValueError("No valid marker-locked epochs remain for CTPS QC.")
    _, scores = ica.find_bads_ecg(
        epochs,
        method="ctps",
        threshold=threshold,
        verbose="ERROR",
    )
    return np.asarray(scores, dtype=float)


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
) -> tuple[np.ndarray, np.ndarray]:
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
    evoked = epochs.average().data
    return epochs.times.copy(), np.sqrt(np.mean(evoked**2, axis=0)) * 1e6


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
    before_times, before_gfp = _cardiac_locked_gfp(
        raw,
        detection.events,
        settings=settings,
    )
    after = raw.copy()
    ica.apply(after, verbose="ERROR")
    after_times, after_gfp = _cardiac_locked_gfp(
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
    )


def _build_component_cardiac_review(
    raws: list[mne.io.BaseRaw],
    *,
    ica: mne.preprocessing.ICA,
    settings: CardiacReviewSettings,
) -> ComponentCardiacReview:
    heartbeat_epochs = []
    correlation_scores = []
    correlation_flags = np.zeros(int(ica.n_components_), dtype=bool)
    for raw in raws:
        detection = detect_ecg_events(raw, settings)
        heartbeat_epochs.append(
            _heartbeat_epochs(raw, detection.events, ica=ica, settings=settings)
        )
        flagged, scores = ica.find_bads_ecg(
            raw,
            ch_name=settings.ecg_channel,
            method="correlation",
            threshold="auto",
            verbose="ERROR",
        )
        correlation_scores.append(np.abs(np.asarray(scores, dtype=float)))
        correlation_flags[np.asarray(flagged, dtype=int)] = True

    epochs = mne.concatenate_epochs(heartbeat_epochs, verbose="ERROR")
    ctps_components, ctps_scores = ica.find_bads_ecg(
        epochs,
        ch_name=settings.ecg_channel,
        method="ctps",
        threshold=settings.ctps_threshold,
        verbose="ERROR",
    )
    sources = ica.get_sources(epochs)
    standardized = standardize_source_epochs(
        sources.get_data(copy=False),
        times=sources.times,
        baseline=settings.baseline,
    )
    ctps_flags = np.zeros(int(ica.n_components_), dtype=bool)
    ctps_flags[np.asarray(ctps_components, dtype=int)] = True
    return ComponentCardiacReview(
        times=sources.times.copy(),
        mean_z=standardized.mean(axis=0),
        ci95_z=1.96 * standardized.std(axis=0, ddof=1) / np.sqrt(len(standardized)),
        median_abs_correlation=np.median(np.stack(correlation_scores), axis=0),
        ctps_scores=np.asarray(ctps_scores, dtype=float),
        correlation_flags=correlation_flags,
        ctps_flags=ctps_flags,
        heartbeat_count=len(epochs),
    )


def _plot_run_cardiac_review(review: RunCardiacReview):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), layout="constrained")
    axes[0].plot(review.representative_times, review.representative_ecg_mv, color="#343A40")
    for peak_time in review.representative_peak_times:
        axes[0].axvline(peak_time, color="#C23B22", alpha=0.65, linewidth=1.0)
    axes[0].set(title="Representative ECG with signal-detected R peaks", ylabel="ECG (mV)")

    axes[1].plot(review.rr_times, review.heart_rate_bpm, color="#6B4C9A", marker=".")
    axes[1].axhline(review.average_pulse_bpm, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].set(title="Beat-to-beat heart rate", ylabel="Heart rate (bpm)")

    axes[2].plot(review.locked_times, review.before_gfp_uv, color="#B24C3B", label="Before ICA")
    axes[2].plot(review.locked_times, review.after_gfp_uv, color="#276B8A", label="After ICA")
    axes[2].axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[2].set(
        title=(
            "R-locked EEG global field power · "
            f"{review.attenuation_percent:+.1f}% attenuation in configured window"
        ),
        xlabel="Time from R peak (s)",
        ylabel="GFP (µV)",
    )
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{review.recording_id} · {review.heartbeat_count} retained beats · "
        f"mean {review.average_pulse_bpm:.1f} bpm"
    )
    plt.close(figure)
    return figure


def _plot_component_cardiac_review(
    review: ComponentCardiacReview,
    *,
    ica: mne.preprocessing.ICA,
    component: int,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    mne.viz.plot_topomap(
        ica.get_components()[:, component],
        ica.info,
        axes=axes[0],
        show=False,
        contours=6,
    )
    axes[0].set_title("Scalp topography")

    mean = review.mean_z[component]
    ci95 = review.ci95_z[component]
    axes[1].plot(review.times, mean, color="#276B8A")
    axes[1].fill_between(review.times, mean - ci95, mean + ci95, color="#276B8A", alpha=0.2)
    axes[1].axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].set(
        title=f"R-locked source average (n={review.heartbeat_count})",
        xlabel="Time from R peak (s)",
        ylabel="Baseline-standardized amplitude (z)",
    )

    values = [review.median_abs_correlation[component], review.ctps_scores[component]]
    flagged = [review.correlation_flags[component], review.ctps_flags[component]]
    colors = ["#C23B22" if value else "#7A8793" for value in flagged]
    axes[2].bar(["|ECG correlation|", "CTPS"], values, color=colors, width=0.62)
    axes[2].set(title="Independent cardiac evidence", ylabel="Score", ylim=(0.0, 1.0))
    for position, value in enumerate(values):
        axes[2].text(position, value + 0.025, f"{value:.3f}", ha="center")
    for axis in axes[1:]:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    recommendation = "REVIEW" if any(flagged) else "not algorithmically flagged"
    figure.suptitle(f"ICA{component:03d} · ECG evidence · {recommendation}")
    plt.close(figure)
    return figure


def _organize_cardiac_review(report: mne.Report) -> None:
    content = report._content
    cardiac_indices = [
        index for index, element in enumerate(content) if "ica-cardiac-review" in element.tags
    ]
    if not cardiac_indices:
        raise ValueError("The report has no ICA cardiac-review content.")
    remaining = [index for index in range(len(content)) if index not in cardiac_indices]
    targets = [
        index
        for index in remaining
        if "ica-component-review" in content[index].tags
        or content[index].section == "ICA: components"
    ]
    if not targets:
        raise ValueError("The report has no ICA component-review insertion point.")
    insertion_index = remaining.index(targets[0])
    report.reorder(remaining[:insertion_index] + cardiac_indices + remaining[insertion_index:])


def _cardiac_review_guide_html(settings: CardiacReviewSettings) -> str:
    return (
        "<p><strong>Manual ECG review; no components are excluded here.</strong> "
        "R peaks are detected directly from the configured ECG signal, so this review does "
        "not depend on BrainVision Analyzer R markers.</p>"
        "<p>Inspect each run for plausible peak placement and heart-rate continuity. Then "
        "review components with converging scalp, R-locked, ECG-correlation, and CTPS "
        "evidence. Red score bars indicate an MNE algorithmic flag and are evidence for "
        "manual judgment, not an automatic rejection.</p>"
        f"<p>R-locked epoch: {settings.epoch_window[0]:g} to "
        f"{settings.epoch_window[1]:g} s; baseline: {settings.baseline[0]:g} to "
        f"{settings.baseline[1]:g} s; CTPS threshold: {settings.ctps_threshold:g}.</p>"
    )


def generate_ica_cardiac_review(
    *,
    filtered_raw_paths: list[Path],
    ica_path: Path,
    report_path: Path,
    output_path: Path,
    settings: CardiacReviewSettings,
) -> Path:
    """Append direct ECG diagnostics and review-only ICA evidence to an MNE report."""
    if not settings.enabled:
        raise ValueError("generate_ica_cardiac_review requires cardiac_review.enabled=true.")
    if not filtered_raw_paths:
        raise ValueError("No filtered raw recordings were provided for ECG review.")
    ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
    raws = [mne.io.read_raw_fif(path, preload=True, verbose="ERROR") for path in filtered_raw_paths]
    run_reviews = [
        _build_run_cardiac_review(
            raw,
            ica=ica,
            recording_id=path.name.removesuffix("_proc-filt_raw.fif"),
            settings=settings,
        )
        for path, raw in zip(filtered_raw_paths, raws, strict=True)
    ]
    component_review = _build_component_cardiac_review(raws, ica=ica, settings=settings)
    table = component_cardiac_evidence_table(component_review)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)

    report = mne.open_report(report_path)
    section = "ICA cardiac artifact review"
    report.add_html(
        html=_cardiac_review_guide_html(settings),
        title="How to review ECG artifacts",
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review"),
        replace=True,
    )
    report.add_figure(
        fig=[_plot_run_cardiac_review(review) for review in run_reviews],
        title="ECG detection and provisional correction by run",
        caption=[review.recording_id for review in run_reviews],
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-run-review"),
        replace=True,
    )
    component_figures = [
        _plot_component_cardiac_review(component_review, ica=ica, component=component)
        for component in range(int(ica.n_components_))
    ]
    report.add_figure(
        fig=component_figures,
        title="ICA components: R-locked cardiac evidence",
        caption=[f"ICA{component:03d}" for component in range(int(ica.n_components_))],
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-component-review"),
        replace=True,
    )
    _organize_cardiac_review(report)
    report.save(report_path, overwrite=True, open_browser=False)
    report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    return output_path


def write_cardiac_attenuation_qc(
    recordings: Iterable[tuple[str, mne.io.BaseRaw, mne.io.BaseRaw]],
    *,
    output_path: Path,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> Path:
    """Write run-level marker-locked EEG attenuation before versus after ICA."""
    rows = []
    for recording_id, before, after in recordings:
        metrics = compute_cardiac_attenuation(
            before,
            after,
            recording_id=recording_id,
            baseline=baseline,
            measurement_window=measurement_window,
        )
        rows.append(
            {
                "recording_id": metrics.recording_id,
                "marker_count": metrics.marker_count,
                "before_rms_uv": metrics.before_rms_uv,
                "after_rms_uv": metrics.after_rms_uv,
                "attenuation_percent": metrics.attenuation_percent,
            }
        )
    if not rows:
        raise ValueError("No before/after recordings were provided for cardiac QC.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_path, sep="\t", index=False)
    _write_cardiac_attenuation_figure(table, output_path.with_suffix(".png"))
    return output_path


def _write_cardiac_attenuation_figure(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    positions = np.arange(len(table))
    figure_height = max(3.0, 0.45 * len(table) + 1.5)
    figure, axis = plt.subplots(figsize=(8.0, figure_height), constrained_layout=True)
    for position, before, after in zip(
        positions,
        table["before_rms_uv"],
        table["after_rms_uv"],
        strict=True,
    ):
        axis.plot([before, after], [position, position], color="0.65", linewidth=1.5)
    axis.scatter(table["before_rms_uv"], positions, label="Before ICA", color="#B24C3B")
    axis.scatter(table["after_rms_uv"], positions, label="After ICA", color="#276B8A")
    axis.set_yticks(positions, table["recording_id"])
    axis.set_xlabel("R-marker-locked EEG RMS (µV)")
    axis.set_title("Cardiac artifact attenuation after ICA")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _resolve_subjects(pipeline_root: Path, subjects: list[str]) -> list[str]:
    if subjects == ["all"]:
        resolved = sorted(
            path.name.removeprefix("sub-") for path in pipeline_root.glob("sub-*") if path.is_dir()
        )
    else:
        resolved = [subject.removeprefix("sub-") for subject in subjects]
    if not resolved:
        raise FileNotFoundError(f"No subject derivatives found under {pipeline_root}.")
    return resolved


def _visible_matches(directory: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and not path.name.startswith("._")
    )


def _require_single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {description}, found {len(paths)}: {paths}")
    return paths[0]


def run_marker_ctps_qc(
    *,
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
    threshold: float,
    epoch_window: tuple[float, float],
) -> Path:
    """Add Analyzer-marker CTPS flags to native MNE-BIDS component tables."""
    task_selector = f"_task-{task}_" if task is not None else "_task-"
    summary_frames = []
    for subject in _resolve_subjects(pipeline_root, subjects):
        eeg_directory = pipeline_root / f"sub-{subject}" / "eeg"
        ica_path = _require_single_path(
            _visible_matches(eeg_directory, f"sub-{subject}_proc-icafit_ica.fif"),
            f"sub-{subject} ICA fit",
        )
        components_path = _require_single_path(
            _visible_matches(eeg_directory, f"sub-{subject}_proc-ica_components.tsv"),
            f"sub-{subject} ICA component table",
        )
        filtered_paths = [
            path
            for path in _visible_matches(
                eeg_directory,
                f"sub-{subject}_task-*_run-*_proc-filt_raw.fif",
            )
            if task_selector in path.name
        ]
        if not filtered_paths:
            raise FileNotFoundError(f"No filtered raw runs found for sub-{subject}, task={task!r}.")

        ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
        raws = [mne.io.read_raw_fif(path, preload=True, verbose="ERROR") for path in filtered_paths]
        scores = compute_marker_ctps_scores(
            raws,
            ica,
            threshold=threshold,
            epoch_window=epoch_window,
        )
        components = pd.read_csv(components_path, sep="\t")
        updated = add_marker_ctps_columns(
            components,
            scores,
            threshold=threshold,
        )
        updated.to_csv(components_path, sep="\t", index=False)

        summary = updated[
            [
                "component",
                "status",
                "status_description",
                "analyzer_marker_ctps_score",
                "analyzer_marker_ctps_flag",
            ]
        ].copy()
        summary.insert(0, "participant_id", f"sub-{subject}")
        summary_frames.append(summary)

    output_path = (
        pipeline_root
        / "qc"
        / f"{'task-' + task + '_' if task is not None else ''}desc-markerctps_components.tsv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(summary_frames, ignore_index=True).to_csv(
        output_path,
        sep="\t",
        index=False,
    )
    return output_path


def run_cardiac_attenuation_qc(
    *,
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> Path:
    """Pair filtered and clean runs and write marker-locked attenuation QC."""
    task_selector = f"_task-{task}_" if task is not None else "_task-"
    recordings = []
    for subject in _resolve_subjects(pipeline_root, subjects):
        eeg_directory = pipeline_root / f"sub-{subject}" / "eeg"
        filtered_paths = [
            path
            for path in _visible_matches(
                eeg_directory,
                f"sub-{subject}_task-*_run-*_proc-filt_raw.fif",
            )
            if task_selector in path.name
        ]
        if not filtered_paths:
            raise FileNotFoundError(f"No filtered raw runs found for sub-{subject}, task={task!r}.")
        for filtered_path in filtered_paths:
            clean_path = filtered_path.with_name(
                filtered_path.name.replace("_proc-filt_raw.fif", "_proc-clean_raw.fif")
            )
            if not clean_path.is_file():
                raise FileNotFoundError(f"Missing ICA-cleaned raw file: {clean_path}")
            recording_id = filtered_path.name.removesuffix("_proc-filt_raw.fif")
            recordings.append(
                (
                    recording_id,
                    mne.io.read_raw_fif(filtered_path, preload=True, verbose="ERROR"),
                    mne.io.read_raw_fif(clean_path, preload=True, verbose="ERROR"),
                )
            )

    output_path = (
        pipeline_root
        / "qc"
        / f"{'task-' + task + '_' if task is not None else ''}desc-cardiacattenuation_qc.tsv"
    )
    return write_cardiac_attenuation_qc(
        recordings,
        output_path=output_path,
        baseline=baseline,
        measurement_window=measurement_window,
    )


__all__ = [
    "CardiacAttenuationMetrics",
    "CardiacReviewSettings",
    "ComponentCardiacReview",
    "EcgDetection",
    "RunCardiacReview",
    "add_marker_ctps_columns",
    "component_cardiac_evidence_table",
    "compute_cardiac_attenuation",
    "compute_marker_ctps_scores",
    "detect_ecg_events",
    "generate_ica_cardiac_review",
    "pulse_marker_events",
    "run_cardiac_attenuation_qc",
    "run_marker_ctps_qc",
    "standardize_source_epochs",
    "write_cardiac_attenuation_qc",
]
