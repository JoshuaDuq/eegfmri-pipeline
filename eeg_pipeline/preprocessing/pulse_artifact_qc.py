"""Description of BrainVision Analyzer pulse-artifact markers."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np

logger = logging.getLogger(__name__)

from eeg_pipeline.preprocessing.report.rr_intervals import (
    DEFAULT_BEAT_MARKER_DESCRIPTION as PULSE_MARKER_DESCRIPTION,
)

#: Marks a stretch the pulse correction never ran on. ``BAD_`` so that MNE excludes it
#: from epoching by default.
UNCORRECTED_PULSE_DESCRIPTION = "BAD_bcg_uncorrected"

#: An inter-marker interval longer than this multiple of the run's own median is counted
#: as a gap. Relative to the run rather than absolute, so it does not mistake a slow heart
#: for a dropout. The value matches the pulse-correction recovery investigation, so the
#: coverage reported here and the coverage quoted there are the same quantity.
GAP_INTERVAL_MULTIPLE = 1.75


def _number(value: float) -> str:
    """Format a measurement, leaving an undefined one visibly empty rather than absent."""
    return "" if not np.isfinite(value) else f"{value:.6f}"


@dataclass(frozen=True)
class PulseMarkerCriteria:
    """Bounds the pulse-marker measurements are described against.

    These are reference values for the reader, not acceptance criteria: nothing is
    excluded from the cohort by falling outside them. They are written into the QC table
    beside the measurements so the comparison can be re-derived at a different bound.
    """

    minimum_bpm: float
    maximum_bpm: float
    minimum_marker_fraction: float
    minimum_recording_coverage: float

    def __post_init__(self) -> None:
        if not 0 < self.minimum_bpm < self.maximum_bpm:
            raise ValueError("Pulse-marker BPM bounds must satisfy 0 < minimum < maximum.")
        for name in ("minimum_marker_fraction", "minimum_recording_coverage"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")


@dataclass(frozen=True)
class PulseMarkerMetrics:
    """Run-level pulse-marker measurements, taken without reference to any bound.

    ``duration_seconds`` and ``expected_marker_count`` are carried so that every derived
    quantity in the row can be recomputed by a reader who defines it differently.
    Interval-derived fields are ``nan`` for a run with fewer than three markers.
    """

    recording_id: str
    marker_count: int
    duration_seconds: float
    median_bpm: float
    marker_fraction: float
    #: Span of the marker train over the recording duration: ``(last - first) / duration``.
    #: Says where the train starts and ends, not whether it is continuous in between.
    recording_coverage: float
    #: Share of the recording inside the marker train and outside a gap, where a gap is an
    #: inter-marker interval longer than :data:`GAP_INTERVAL_MULTIPLE` times the run's own
    #: median. This is the quantity the pulse-correction recovery investigation reports,
    #: and it is the one that separates a continuous train from a sparse one — a run can
    #: span 99% of its recording while marking only a third of the beats in that span.
    gap_free_coverage: float
    expected_marker_count: int
    #: Number of inter-marker intervals counted as gaps.
    gap_count: int


def _pulse_onsets(raw: mne.io.BaseRaw) -> np.ndarray:
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    return np.asarray(
        raw.annotations.onset[descriptions == PULSE_MARKER_DESCRIPTION],
        dtype=float,
    )


def measure_pulse_markers(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
) -> PulseMarkerMetrics:
    """Measure a run's Analyzer R-marker train. Makes no judgement about it.

    Measurement is separated from evaluation because the previous combined version stopped
    at the first bound a run fell outside and recorded nothing else for it. On this cohort
    that blanked every measurement for 24 of 90 runs — the runs whose coverage most needed
    describing were the ones the table described least, with the numbers surviving only
    inside an error string.

    Everything a derived quantity is computed from is returned alongside it
    (``duration_seconds``, ``expected_marker_count``), so a reader who disagrees with how
    ``marker_fraction`` is defined can recompute their own from the same row.

    Runs too sparse to yield an interval return ``nan`` for the interval-derived fields
    rather than raising: "fewer than three markers" is itself the finding, and a run that
    cannot be characterised still belongs in the table.
    """
    if not recording_id.strip():
        raise ValueError("recording_id must not be empty.")

    onsets = _pulse_onsets(raw)
    duration_seconds = raw.n_times / float(raw.info["sfreq"])

    if len(onsets) < 3:
        return PulseMarkerMetrics(
            recording_id=recording_id,
            marker_count=len(onsets),
            duration_seconds=duration_seconds,
            median_bpm=float("nan"),
            marker_fraction=float("nan"),
            recording_coverage=float("nan"),
            gap_free_coverage=float("nan"),
            expected_marker_count=0,
            gap_count=0,
        )

    intervals = np.diff(onsets)
    if np.any(intervals <= 0):
        # Not a quality question. Non-increasing onsets mean the marker train is not a
        # time series, so no measurement below is defined on it.
        raise ValueError(f"{recording_id}: pulse marker onsets must be strictly increasing.")

    # The 20th-percentile interval, not the median: it estimates the beat period from the
    # run's faster beats, so a complete marker set scores near 1.0 while dropouts pull the
    # fraction down. It also means a complete set scores slightly below 1.0 at high
    # heart-rate variability, which is why the value is reported rather than thresholded
    # here.
    representative_interval = float(np.quantile(intervals, 0.2, method="lower"))
    marker_span = float(onsets[-1] - onsets[0])
    expected_marker_count = int(np.floor(marker_span / representative_interval)) + 1

    median_interval = float(np.median(intervals))
    gaps = intervals > GAP_INTERVAL_MULTIPLE * median_interval
    gap_seconds = float(intervals[gaps].sum())

    return PulseMarkerMetrics(
        recording_id=recording_id,
        marker_count=len(onsets),
        duration_seconds=duration_seconds,
        median_bpm=60.0 / median_interval,
        marker_fraction=min(1.0, len(onsets) / expected_marker_count),
        recording_coverage=float(marker_span / duration_seconds),
        gap_free_coverage=float((marker_span - gap_seconds) / duration_seconds),
        expected_marker_count=expected_marker_count,
        gap_count=int(gaps.sum()),
    )


def uncorrected_pulse_intervals(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    maximum_plausible_interval: float | None = None,
) -> mne.Annotations:
    """Mark the stretches of a run where no pulse template was subtracted.

    Analyzer's correction runs at the beats it marked. Where the marker train has a gap,
    or has not started or has ended, the ballistocardiogram is still in the EEG.

    This records what the delivered data has uncorrected; it does not claim the beats are
    unrecoverable. Some are: sub-0008 run-4 carries 110 markers where an independent
    detector finds 508 at 61 bpm, reproducing all 110 of Analyzer's own at 9.5 ms, and the
    merged train is regular (IQR 0.09 s). Recovering them is `eeg-pipeline cardiac-gaps`,
    which corrects only the gap stretches and keeps Analyzer's correction elsewhere.
    Whether a given gap is recoverable has to be established per run against that run's own
    markers, because a generic QRS detector is not a trustworthy arbiter on this in-scanner
    ECG — it reproduces Analyzer's markers on 17 of 90 runs and over-detects the T wave on
    the rest.

    A gap contributes the region closer to its missing beats than to the marked beats
    either side — half a beat period inside each flanking marker. The gap rule is the same
    :data:`GAP_INTERVAL_MULTIPLE` that produces ``gap_count``, so the annotations and the QC
    table cannot disagree about what a gap is.

    The beat period is the median interval, which survives corruption in both directions:
    scattered dropouts do not move it, and neither do the doubled markers some runs carry
    (sub-0005 marks an extra beat mid-cycle often enough that its 20th-percentile interval
    is half its true period, which would flag most of a well-marked run).

    The median has one blind spot, and it is the one that matters most: a run missing most
    of its beats has *every* interval inflated, so judged against itself it looks evenly
    covered at an impossible rate. sub-0008 run-4 carries 110 markers over 497 s — a median
    of 4.07 s against the same subject's 1.0 s elsewhere — and scores as nearly fully
    covered while being the worst uncorrected run in the cohort. ``maximum_plausible_
    interval`` — the caller's configured minimum heart rate as an interval — caps the
    estimate so such a run is measured against a rate a heart could actually have. Without
    it the function stays purely relative to the run.

    A run with fewer than three markers has no interval to measure against and is marked
    end to end: that it cannot be characterised is not a reason to call it corrected.
    """
    if not recording_id.strip():
        raise ValueError("recording_id must not be empty.")
    if maximum_plausible_interval is not None and maximum_plausible_interval <= 0:
        raise ValueError("maximum_plausible_interval must be positive.")

    onsets = _pulse_onsets(raw)
    duration = raw.n_times / float(raw.info["sfreq"])

    if len(onsets) < 3:
        return mne.Annotations(
            onset=[0.0], duration=[duration], description=[UNCORRECTED_PULSE_DESCRIPTION]
        )

    intervals = np.diff(onsets)
    if np.any(intervals <= 0):
        raise ValueError(f"{recording_id}: pulse marker onsets must be strictly increasing.")

    beat_period = float(np.median(intervals))
    if maximum_plausible_interval is not None:
        beat_period = min(beat_period, float(maximum_plausible_interval))
    half = beat_period / 2.0
    threshold = GAP_INTERVAL_MULTIPLE * beat_period

    spans: list[tuple[float, float]] = []
    for index in np.where(intervals > threshold)[0]:
        spans.append((float(onsets[index]) + half, float(onsets[index + 1]) - half))
    # The same rule applies to the ends: the correction cannot have run before the first
    # marked beat or after the last, and those stretches are gaps against the run's own
    # rate exactly as an interior one is.
    if onsets[0] - 0.0 > threshold:
        spans.append((0.0, float(onsets[0]) - half))
    if duration - onsets[-1] > threshold:
        spans.append((float(onsets[-1]) + half, duration))

    spans = [(max(0.0, lo), min(duration, hi)) for lo, hi in sorted(spans) if hi > lo]
    return mne.Annotations(
        onset=[lo for lo, _ in spans],
        duration=[hi - lo for lo, hi in spans],
        description=[UNCORRECTED_PULSE_DESCRIPTION] * len(spans),
    )


def _write_uncorrected_intervals(
    annotations: mne.Annotations,
    *,
    recording_id: str,
    directory: Path,
) -> Path:
    """Write one run's intervals, including when there are none.

    An empty file distinguishes "this run was measured and had nothing to mark" from
    "this run was never measured"; a missing file would conflate them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{recording_id}_desc-bcguncorrected_annotations.tsv"
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["onset", "duration", "description"], delimiter="\t"
        )
        writer.writeheader()
        for annotation in annotations:
            writer.writerow(
                {
                    "onset": f"{annotation['onset']:.3f}",
                    "duration": f"{annotation['duration']:.3f}",
                    "description": annotation["description"],
                }
            )
    return path


def pulse_marker_bound_notes(
    metrics: PulseMarkerMetrics,
    criteria: PulseMarkerCriteria,
) -> tuple[str, ...]:
    """Name every configured bound this run's measurements fall outside.

    Notes, not verdicts. Each states the measured value and the bound it is compared
    against, so the comparison can be re-derived at a different bound without re-running
    anything, and so a reader who thinks the bound is wrong still has the measurement.

    Every bound is checked; the run is not abandoned at the first one. A run can be sparse
    *and* poorly covered, and both facts are worth having.
    """
    notes: list[str] = []
    if metrics.marker_count < 3:
        notes.append(f"marker count {metrics.marker_count} is too few to derive an interval")
        return tuple(notes)

    if not criteria.minimum_bpm <= metrics.median_bpm <= criteria.maximum_bpm:
        notes.append(
            f"median rate {metrics.median_bpm:.1f} bpm is outside the configured "
            f"{criteria.minimum_bpm:.0f}-{criteria.maximum_bpm:.0f} bpm"
        )
    if metrics.marker_fraction < criteria.minimum_marker_fraction:
        notes.append(
            f"marker fraction {metrics.marker_fraction:.3f} is below the configured "
            f"{criteria.minimum_marker_fraction:.3f}"
        )
    if metrics.recording_coverage < criteria.minimum_recording_coverage:
        notes.append(
            f"marker span covers {metrics.recording_coverage:.3f} of the recording, below "
            f"the configured {criteria.minimum_recording_coverage:.3f}"
        )
    return tuple(notes)


def summarize_pulse_marker_recordings(
    recordings: Iterable[tuple[str, mne.io.BaseRaw]],
    criteria: PulseMarkerCriteria,
    *,
    output_path: Path,
    strict: bool = False,
    annotations_dir: Path | None = None,
) -> Path:
    """Write one row of pulse-marker measurements per recording, and the bounds they meet.

    Every run gets every measurement, whether or not it falls inside the configured
    bounds. ``outside_configured_bounds`` and ``notes`` describe the relation between the
    measurements and the bounds that were configured; they are not a quality grade, and
    the thresholds are written into the table so the comparison stays re-derivable.

    With ``annotations_dir``, each run's uncorrected-pulse intervals are written beside the
    table by :func:`uncorrected_pulse_intervals`. They come from the same measurement pass
    as ``gap_count``, so the stage that reports how much of a run the correction covered
    also emits the intervals it did not cover.

    ``strict`` remains available for a caller that wants a hard gate, and is off by
    default: on this dataset incomplete within-run coverage is a documented, open property
    of the Analyzer export rather than a reason to stop.
    """
    rows = []
    outside = []
    for recording_id, raw in recordings:
        metrics = measure_pulse_markers(raw, recording_id=recording_id)
        notes = pulse_marker_bound_notes(metrics, criteria)
        if notes:
            outside.append(f"{recording_id}: " + "; ".join(notes))
        if annotations_dir is not None:
            _write_uncorrected_intervals(
                uncorrected_pulse_intervals(
                    raw,
                    recording_id=recording_id,
                    maximum_plausible_interval=60.0 / criteria.minimum_bpm,
                ),
                recording_id=recording_id,
                directory=annotations_dir,
            )
        rows.append(
            {
                "recording_id": metrics.recording_id,
                "marker_count": metrics.marker_count,
                "duration_seconds": f"{metrics.duration_seconds:.3f}",
                "median_bpm": _number(metrics.median_bpm),
                "marker_fraction": _number(metrics.marker_fraction),
                "recording_coverage": _number(metrics.recording_coverage),
                "gap_free_coverage": _number(metrics.gap_free_coverage),
                "gap_count": metrics.gap_count,
                "expected_marker_count": metrics.expected_marker_count,
                "configured_bpm_range": f"{criteria.minimum_bpm:.0f}-{criteria.maximum_bpm:.0f}",
                "configured_minimum_marker_fraction": f"{criteria.minimum_marker_fraction:.3f}",
                "configured_minimum_coverage": (f"{criteria.minimum_recording_coverage:.3f}"),
                "outside_configured_bounds": "yes" if notes else "no",
                "notes": "; ".join(notes),
            }
        )

    if not rows:
        raise ValueError("No EEG recordings were provided for pulse-marker QC.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    if outside:
        summary = (
            f"{len(outside)} of {len(rows)} recordings have pulse-marker measurements "
            f"outside the configured bounds (see {output_path}): " + " | ".join(outside)
        )
        if strict:
            raise ValueError(summary)
        logger.info(summary)

    return output_path


__all__ = [
    "GAP_INTERVAL_MULTIPLE",
    "PULSE_MARKER_DESCRIPTION",
    "UNCORRECTED_PULSE_DESCRIPTION",
    "PulseMarkerCriteria",
    "PulseMarkerMetrics",
    "measure_pulse_markers",
    "pulse_marker_bound_notes",
    "summarize_pulse_marker_recordings",
    "uncorrected_pulse_intervals",
]
