"""Direct ECG detection and run-resolved ICA cardiac evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


def _ctps_threshold(value: Any) -> str | float:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError("ica.cardiac_review.ctps_threshold must be 'auto' or numeric.")
        return value
    threshold = float(value)
    if not 0 < threshold <= 1:
        raise ValueError("ica.cardiac_review.ctps_threshold must be in (0, 1].")
    return threshold


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
    #: Write this review's CTPS detections into the component table as exclusions.
    #:
    #: Off by default because it changes what the cleaned data contain, not just what the
    #: report says. On, it replaces the fifty seconds of one run that MNE-BIDS-Pipeline's
    #: own ECG step samples with the full beat train of every run. The manual review still
    #: runs afterwards and can undo any of it.
    promote_exclusions: bool = False
    #: Fraction of usable runs that must flag a component before it is promoted.
    promotion_minimum_run_fraction: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CardiacReviewSettings:
        if not isinstance(values, Mapping):
            raise TypeError("ica.cardiac_review must be a mapping.")
        supported = {
            "enabled",
            "ecg_channel",
            "epoch_window",
            "baseline",
            "measurement_window",
            "ctps_threshold",
            "representative_window_seconds",
            "promote_exclusions",
            "promotion_minimum_run_fraction",
        }
        unsupported = sorted(set(values) - supported)
        if unsupported:
            raise ValueError("Unsupported ica.cardiac_review settings: " + ", ".join(unsupported))
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
            promote_exclusions=bool(values.get("promote_exclusions", cls.promote_exclusions)),
            promotion_minimum_run_fraction=float(
                values.get(
                    "promotion_minimum_run_fraction",
                    cls.promotion_minimum_run_fraction,
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
        if settings.representative_window_seconds <= 0:
            raise ValueError("ica.cardiac_review.representative_window_seconds must be positive.")
        if not 0.0 < settings.promotion_minimum_run_fraction <= 1.0:
            raise ValueError(
                "ica.cardiac_review.promotion_minimum_run_fraction must lie in (0, 1]."
            )
        return settings


#: Beat train taken from the BrainVision Analyzer markers preserved in the recording.
ANALYZER_MARKER_SOURCE = "analyzer-markers"
#: Beat train detected from the ECG channel, independently of any annotation.
ECG_CHANNEL_SOURCE = "ecg-channel"

#: Beats below which a marker train is not a train, and the channel is tried instead.
#:
#: The same floor the channel detector is held to. Two markers give one interval, which is
#: not a rate, and a run whose export carries a handful of stray markers is better served by
#: the channel than by them.
MINIMUM_BEATS = 3


@dataclass(frozen=True)
class EcgDetection:
    """The run's beat train, and which of the two sources it came from."""

    events: np.ndarray
    average_pulse_bpm: float
    #: :data:`ANALYZER_MARKER_SOURCE` or :data:`ECG_CHANNEL_SOURCE`. Carried so every panel
    #: can say where its beats came from: the two sources fail on different runs, so a rate
    #: is not interpretable without knowing which one produced it.
    source: str = ECG_CHANNEL_SOURCE


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
    before_gfp_uv: np.ndarray
    after_gfp_uv: np.ndarray
    r_locked_epoch_count: int
    average_pulse_bpm: float
    events: np.ndarray
    before_topography_uv: np.ndarray
    after_topography_uv: np.ndarray
    topography_time: float
    #: Which detector produced the beats every panel here is locked to.
    #:
    #: :data:`ANALYZER_MARKER_SOURCE` or :data:`ECG_CHANNEL_SOURCE`, carried through from
    #: :class:`EcgDetection`. The panels are read to judge detection quality, so the one
    #: thing they cannot leave the reader to assume is which detection they are showing --
    #: and :func:`detect_ecg_events` prefers the marker train, so the answer is usually not
    #: the ECG channel.
    beat_source: str = ECG_CHANNEL_SOURCE


@dataclass(frozen=True)
class ComponentCardiacReview:
    """R-locked and score-based evidence for each ICA component."""

    run_ids: tuple[str, ...]
    times: np.ndarray
    run_mean_z: np.ndarray
    correlation_scores: np.ndarray
    ctps_scores: np.ndarray
    correlation_flags: np.ndarray
    ctps_flags: np.ndarray
    r_locked_epoch_counts: np.ndarray
    run_ecg_z: np.ndarray


class UnusableEcg(ValueError):
    """One run's ECG yielded no usable beat train.

    A distinct type because the caller's response is different in kind. A missing channel
    or a mistyped setting is a fault to fix; an ECG the detector cannot resolve is a
    property of the recording, and the reviewer needs it recorded and reported rather than
    thrown. Catching plain :class:`ValueError` around the detector to keep a study running
    would swallow every genuine bug inside it as well, which is why this is narrower.

    Kept a :class:`ValueError` subclass so existing callers that catch that still do.
    """


def _validate_ecg_channel(raw: mne.io.BaseRaw, channel: str) -> None:
    if channel not in raw.ch_names:
        raise ValueError(f"ECG review requires channel {channel!r}.")
    channel_type = raw.get_channel_types(picks=[channel])[0]
    if channel_type != "ecg":
        raise ValueError(
            f"ECG review channel {channel!r} has type {channel_type!r}, expected 'ecg'."
        )


def _marker_beats(raw: mne.io.BaseRaw) -> np.ndarray:
    """The Analyzer R-marker train preserved in the recording, if it carries one.

    Returns an empty array where the export has no markers, which on this dataset is a third
    of runs: Analyzer's R detection failed, so it could not compute the R-to-artifact delay,
    accepted its 0.21 s default and marked nothing. Absence here is an ordinary outcome.
    """
    try:
        events, _ = mne.events_from_annotations(
            raw,
            event_id={PULSE_MARKER_DESCRIPTION: PULSE_EVENT_ID},
            use_rounding=True,
            verbose="ERROR",
        )
    except ValueError as exc:
        # MNE's way of saying the description is absent. Any other ValueError is a fault.
        if "Could not find any of the events" in str(exc):
            return np.empty((0, 3), dtype=int)
        raise
    return np.asarray(events, dtype=int)


def _rate_from_beats(events: np.ndarray, sfreq: float) -> float:
    """Beats per minute from the median interval of a beat train.

    A median rather than the count over the recording length, because a train with gaps --
    which is what a partially failed detection produces -- reports a rate far below the
    heart's when divided by the whole duration. The median interval describes the beats that
    were found rather than the ones that were missed.
    """
    if events.shape[0] < 2:
        return float("nan")
    intervals = np.diff(np.sort(events[:, 0].astype(float))) / float(sfreq)
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        return float("nan")
    return float(60.0 / np.median(intervals))


def detect_ecg_events(
    raw: mne.io.BaseRaw,
    settings: CardiacReviewSettings,
) -> EcgDetection:
    """The run's beat train, preferring Analyzer's markers over channel detection.

    Analyzer's marker train is preferred where the export carries one, because it is the
    detection that actually drove the upstream pulse-artifact correction and it was
    validated against the recording. ``find_ecg_events`` on the ECG channel is used only
    where no marker train survives.

    The preference is not circular. This review compares the EEG either side of *MNE's* ICA
    exclusions; Analyzer's correction is already baked into ``raw`` and is not what is being
    judged, so taking the beat reference from Analyzer's markers does not let the correction
    grade itself.

    The order matters on real data. On this dataset the channel detector disagrees sharply
    with the marker train on the same runs -- reporting 8 bpm and 2 bpm where the markers
    report 61 and 60 -- and the two sources fail on *different* runs, so neither alone
    characterises a subject. :func:`pulse_marker_events` makes the same choice for the
    cardiac attenuation QC.
    """
    _validate_ecg_channel(raw, settings.ecg_channel)
    sfreq = float(raw.info["sfreq"])

    markers = _marker_beats(raw)
    if markers.shape[0] >= MINIMUM_BEATS:
        rate = _rate_from_beats(markers, sfreq)
        if np.isfinite(rate) and rate > 0:
            return EcgDetection(
                events=markers,
                average_pulse_bpm=rate,
                source=ANALYZER_MARKER_SOURCE,
            )

    events, _, average_pulse_bpm, _ = mne.preprocessing.find_ecg_events(
        raw,
        ch_name=settings.ecg_channel,
        event_id=PULSE_EVENT_ID,
        return_ecg=True,
        verbose="ERROR",
    )
    if len(events) < MINIMUM_BEATS:
        raise UnusableEcg(
            f"No Analyzer R markers, and direct ECG detection found only {len(events)} "
            f"R peaks; at least {MINIMUM_BEATS} are required."
        )
    if not np.isfinite(average_pulse_bpm) or average_pulse_bpm <= 0:
        raise UnusableEcg(f"Invalid average pulse estimate: {average_pulse_bpm!r}.")
    return EcgDetection(
        events=np.asarray(events, dtype=int),
        average_pulse_bpm=float(average_pulse_bpm),
        source=ECG_CHANNEL_SOURCE,
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


def component_cardiac_evidence_table(
    review: ComponentCardiacReview,
    *,
    statuses: pd.DataFrame,
) -> pd.DataFrame:
    """Create review evidence without assigning or changing ICA exclusions."""
    run_count, component_count, _ = review.run_mean_z.shape
    expected_shape = (run_count, component_count)
    score_arrays = (
        review.correlation_scores,
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
    return statuses[["component", "status", "status_description"]].rename(
        columns={
            "status": "current_ica_status",
            "status_description": "current_status_description",
        }
    )


def ctps_promotions(
    review: ComponentCardiacReview,
    *,
    minimum_run_fraction: float,
) -> dict[int, str]:
    """Return the components CTPS flags in enough runs, with the evidence for each.

    MNE-BIDS-Pipeline's own ECG detection builds its heartbeat epochs from the first run
    only, cropped to ``5 minutes / n_runs`` around that run's midpoint — roughly fifty
    seconds of one run scoring a decomposition fitted across all of them. This review
    already runs the same CTPS test per run over every beat in every recording, so these
    detections are what that step was meant to produce.

    A component has to clear ``minimum_run_fraction`` of the runs that yielded a usable
    beat train. A single-run flag is not enough on its own: BCG topography moves with head
    position, so one run disagreeing with five is as likely to be a threshold crossing as
    a cardiac component. The description names the runs so the reviewer can check the call
    rather than take it.
    """
    if not 0.0 < minimum_run_fraction <= 1.0:
        raise ValueError("minimum_run_fraction must lie in (0, 1].")
    run_count, component_count = review.ctps_flags.shape
    if run_count == 0:
        return {}

    required_runs = int(np.ceil(minimum_run_fraction * run_count))
    promotions: dict[int, str] = {}
    for component in range(component_count):
        flagged = np.flatnonzero(review.ctps_flags[:, component])
        if len(flagged) < required_runs:
            continue
        flagged_ids = ", ".join(review.run_ids[index] for index in flagged)
        promotions[int(component)] = (
            f"Auto-detected ECG artifact (full-recording CTPS in {len(flagged)}/{run_count} "
            f"runs, threshold {required_runs}: {flagged_ids})"
        )
    return promotions


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
                    "mne_ecg_correlation_score": review.correlation_scores[run_index, component],
                    "mne_ctps_score": review.ctps_scores[run_index, component],
                    "mne_correlation_flag": review.correlation_flags[run_index, component],
                    "mne_ctps_flag": review.ctps_flags[run_index, component],
                    "r_locked_epoch_count": review.r_locked_epoch_counts[run_index],
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
    ica: mne.preprocessing.ICA,
    settings: CardiacReviewSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R-locked evoked and GFP over the channels the decomposition actually spans.

    ``picks="eeg"`` keeps bad channels, which ICA was not fitted on and ``ICA.apply``
    restores untouched. Measuring the before/after comparison over them does two wrong
    things at once: the evoked array no longer matches ``ica.info``, which is a hard error
    the moment a topography is drawn from it, and every unmodifiable channel folded into
    the GFP shrinks the apparent difference between before and after. Both are avoided by
    measuring exactly the channel set ICA operated on.

    The average reference is still computed first, on the full montage, because that is
    the reference the decomposition was fitted under. MNE leaves bad channels out of the
    average itself, so picking afterwards changes what is measured, not what it is
    measured against.
    """
    referenced = raw.copy().set_eeg_reference("average", projection=False, verbose=False)
    epochs = mne.Epochs(
        referenced,
        events,
        event_id=PULSE_EVENT_ID,
        tmin=settings.epoch_window[0],
        tmax=settings.epoch_window[1],
        baseline=settings.baseline,
        picks=list(ica.ch_names),
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("No valid directly detected R-locked EEG epochs remain.")
    if list(epochs.ch_names) != list(ica.ch_names):
        # Epochs orders picks by the info, not by the list given, so an ICA whose channel
        # order differs from the recording's would silently mis-map every topography.
        raise ValueError(
            "R-locked epoch channels do not match the ICA channel order: "
            f"{epochs.ch_names} vs {list(ica.ch_names)}."
        )
    evoked_uv = epochs.average().data * 1e6
    return (
        epochs.times.copy(),
        np.sqrt(np.mean(evoked_uv**2, axis=0)),
        evoked_uv,
    )


def _standardized_average_ecg(epochs: mne.BaseEpochs, *, channel: str) -> np.ndarray:
    average = epochs.get_data(picks=[channel]).mean(axis=0)[0]
    scale = float(average.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Average R-locked ECG has a zero or invalid standard deviation.")
    return (average - average.mean()) / scale


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
        ica=ica,
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
        ica=ica,
        settings=settings,
    )
    if not np.array_equal(before_times, after_times):
        raise ValueError(f"{recording_id}: before/after cardiac epoch times do not align.")
    measurement_mask = (before_times >= settings.measurement_window[0]) & (
        before_times <= settings.measurement_window[1]
    )
    if not measurement_mask.any():
        raise ValueError("Cardiac review measurement window contains no samples.")
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
        before_gfp_uv=before_gfp,
        after_gfp_uv=after_gfp,
        r_locked_epoch_count=len(ecg_epochs),
        average_pulse_bpm=detection.average_pulse_bpm,
        events=ecg_epochs.events.copy(),
        before_topography_uv=before_evoked[:, peak_index],
        after_topography_uv=after_evoked[:, peak_index],
        topography_time=float(before_times[peak_index]),
        beat_source=detection.source,
    )


def _build_component_cardiac_review(
    raws: list[mne.io.BaseRaw],
    run_reviews: list[RunCardiacReview],
    *,
    ica: mne.preprocessing.ICA,
    settings: CardiacReviewSettings,
) -> ComponentCardiacReview:
    if len(raws) != len(run_reviews) or not raws:
        raise ValueError("Raw runs and ECG run reviews must be non-empty and aligned.")
    source_data_runs = []
    run_ids = []
    r_locked_epoch_counts = []
    correlation_scores = []
    correlation_flags = []
    ctps_scores = []
    ctps_flags = []
    ecg_runs = []
    source_times = None
    for raw, review in zip(raws, run_reviews, strict=True):
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
        r_locked_epoch_counts.append(len(epochs))
        source_data_runs.append(sources.get_data(copy=False))
        correlation_scores.append(run_correlation_scores)
        correlation_flags.append(run_correlation_flags)
        ctps_scores.append(run_ctps_scores)
        ctps_flags.append(run_ctps_flags)
        ecg_runs.append(_standardized_average_ecg(epochs, channel=settings.ecg_channel))

    standardized_runs = standardize_source_epoch_runs(
        source_data_runs,
        times=source_times,
        baseline=settings.baseline,
    )
    return ComponentCardiacReview(
        run_ids=tuple(run_ids),
        times=source_times,
        run_mean_z=np.stack([values.mean(axis=0) for values in standardized_runs]),
        correlation_scores=np.stack(correlation_scores),
        ctps_scores=np.stack(ctps_scores),
        correlation_flags=np.stack(correlation_flags),
        ctps_flags=np.stack(ctps_flags),
        r_locked_epoch_counts=np.asarray(r_locked_epoch_counts, dtype=int),
        run_ecg_z=np.stack(ecg_runs),
    )


__all__ = [
    "CardiacReviewSettings",
    "ComponentCardiacReview",
    "EcgDetection",
    "RunCardiacReview",
    "UnusableEcg",
    "component_cardiac_evidence_table",
    "component_run_cardiac_evidence_table",
    "ctps_promotions",
    "detect_ecg_events",
    "standardize_source_epoch_runs",
]
