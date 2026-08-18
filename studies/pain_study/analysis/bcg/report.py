"""Report evidence for the BrainVision Analyzer scanner-artifact correction.

Gradient and pulse-artifact correction happen in Analyzer, before anything in this
pipeline runs. Its quality is therefore an input, not something the pipeline controls,
and it is the one thing a reviewer cannot infer from the MNE stages: an uncorrected
pulse artifact simply looks like unusually strong cardiac structure later on.

The measurements already exist as cohort QC tables. This module puts the rows for one
subject in front of the person reviewing that subject, because a table nobody opens is
not quality control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.cardiac_artifact_qc import PULSE_EVENT_ID
PULSE_MARKER_DESCRIPTION = "Pulse Artifact/R"
from eeg_pipeline.preprocessing.report.annotations import annotation_onsets
from eeg_pipeline.preprocessing.report.rr_intervals import (
    MISSED_BEAT_FACTOR,
    PLAUSIBLE_RR_RANGE_S,
    RrIntervals,
    add_rr_interval_section,
    compute_rr_intervals,
)
from studies.pain_study.analysis.noise_floor import measure_locked_average
from eeg_pipeline.preprocessing.report.style import (
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
)

PULSE_MARKER_QC_SUFFIX = "desc-pulsemarkers_qc.tsv"
CARDIAC_ATTENUATION_QC_SUFFIX = "desc-cardiacattenuation_qc.tsv"







@dataclass(frozen=True)
class AnalyzerCorrectionQc:
    """Per-run Analyzer correction evidence for one subject."""

    subject: str
    runs: pd.DataFrame

    @property
    def runs_outside_bounds(self) -> tuple[str, ...]:
        """Runs whose measurements fall outside the bounds the QC was configured with.

        A pointer for the reader, not a grade. The bounds are reference values recorded
        beside the measurements; a run named here is one to look at, not one to drop.
        ``status``/``fail`` is read as well so that QC tables written before the column
        was renamed still resolve.
        """
        for column, marker in (("outside_configured_bounds", "yes"), ("status", "fail")):
            if column in self.runs.columns:
                selected = self.runs.loc[self.runs[column].astype(str) == marker, "run"]
                return tuple(str(value) for value in selected)
        return ()

    @property
    def fallback_runs(self) -> tuple[str, ...]:
        if "is_fallback" not in self.runs.columns:
            return ()
        fallback = self.runs.loc[self.runs["is_fallback"].fillna(False).astype(bool), "run"]
        return tuple(str(value) for value in fallback)


def _run_label(recording_id: object) -> str:
    """Bare run number for the "run" column of the per-run QC table."""
    return run_label(recording_id, bare=True)


def load_analyzer_qc(
    *,
    qc_dir: Path,
    task: str,
    subject: str,
) -> AnalyzerCorrectionQc | None:
    """Join the pulse-marker and cardiac-attenuation QC rows for one subject.

    Returns ``None`` when neither table exists, so a dataset corrected outside Analyzer
    simply has no such section rather than an empty one.
    """
    frames = {}
    for name, suffix in (
        ("markers", PULSE_MARKER_QC_SUFFIX),
        ("attenuation", CARDIAC_ATTENUATION_QC_SUFFIX),
    ):
        path = qc_dir / f"task-{task}_{suffix}"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, sep="\t")
        if "recording_id" not in frame.columns:
            raise ValueError(f"{path} is missing the recording_id column.")
        frame = frame[frame["recording_id"].astype(str).str.startswith(f"sub-{subject}_")]
        if frame.empty:
            continue
        frame = frame.assign(run=frame["recording_id"].map(_run_label)).drop(
            columns=["recording_id"]
        )
        frames[name] = frame
    if not frames:
        return None

    if len(frames) == 2:
        runs = frames["markers"].merge(
            frames["attenuation"],
            on="run",
            how="outer",
            suffixes=("_markers", "_attenuation"),
        )
    else:
        runs = next(iter(frames.values()))
    runs = runs.sort_values("run").reset_index(drop=True)
    if {"before_rms_uv", "after_rms_uv"}.issubset(runs.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            runs["attenuation_db"] = 20.0 * np.log10(runs["before_rms_uv"] / runs["after_rms_uv"])
    return AnalyzerCorrectionQc(subject=subject, runs=runs)


@dataclass(frozen=True)
class _QcColumn:
    """One column of the Analyzer QC table, and how its frame value is rendered.

    The frame this table is built from is a working structure whose column names are
    the pipeline's own (``marker_fraction``, ``before_rms_uv``). Rendering it with
    ``DataFrame.to_html`` published those names and a single ``float_format`` into a
    document whose every other table carries prose headers and per-quantity precision,
    which is how a beat count came to be printed as ``520.00``.
    """

    source: str
    column: Column
    format: Callable[[Any], str]

    def render(self, value: Any) -> str | None:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None
        try:
            return self.format(value)
        except (TypeError, ValueError):
            return str(value)


def _text(value: Any) -> str:
    return str(value)


#: Columns of the Analyzer QC table, in reading order. A column absent from the frame
#: is skipped, because which measurements exist depends on which QC stages ran.
_QC_COLUMNS: tuple[_QcColumn, ...] = (
    _QcColumn("run", Column("Run", align=Align.TEXT), lambda value: f"run-{value}"),
    _QcColumn(
        "outside_configured_bounds",
        Column("Outside bounds", align=Align.TEXT),
        _text,
    ),
    _QcColumn("marker_count", Column("R markers"), lambda value: f"{int(value):,}"),
    _QcColumn("median_bpm", Column("Median rate (bpm)"), lambda value: f"{float(value):.1f}"),
    _QcColumn(
        "marker_fraction",
        Column("Markers vs expected"),
        lambda value: f"{float(value):.2f}",
    ),
    _QcColumn(
        "recording_coverage",
        Column("Marker span"),
        lambda value: f"{float(value):.0%}",
    ),
    # Beside the span, not instead of it: the two answer different questions and the
    # difference between them is the finding. A run can span 99% of its recording while
    # marking beats in only a fifth of that span, and the span alone reads as complete.
    _QcColumn(
        "gap_free_coverage",
        Column("Gap-free covered"),
        lambda value: f"{float(value):.0%}",
    ),
    _QcColumn("gap_count", Column("Gaps"), lambda value: f"{int(value):,}"),
    _QcColumn(
        "before_rms_uv",
        Column("R-locked amplitude before ICA (µV)"),
        lambda value: f"{float(value):.2f}",
    ),
    _QcColumn(
        "after_rms_uv",
        Column("R-locked amplitude after ICA (µV)"),
        lambda value: f"{float(value):.2f}",
    ),
    _QcColumn(
        "attenuation_db",
        Column("Attenuation (dB)"),
        lambda value: f"{float(value):+.1f}",
    ),
    _QcColumn(
        "is_fallback",
        Column("Markers from fallback", align=Align.TEXT),
        lambda value: "yes" if value else "no",
    ),
)






@dataclass(frozen=True)
class CardiacResidual:
    """How much beat-locked artifact one run still carries, and what it was measured on.

    Measured before the ICA exclusions, because the question this answers is what the
    *upstream* pulse-artifact correction left behind. Measured after them it would credit
    Analyzer for whatever MNE's decomposition cleaned up, and a run that was never
    corrected at all would read as corrected.
    """

    recording_id: str
    #: Analyzer ``Pulse Artifact/R`` markers preserved in the recording. Zero means the
    #: upstream correction had no beat train to key on, so no subtraction was possible.
    marker_count: int
    #: Which source the beats came from, or ``None`` when neither resolved one.
    beat_source: str | None
    #: RMS of the beat-locked average over the measurement window, in microvolts.
    #: ``None`` where no beat train resolved -- an artifact that could not be measured is
    #: not an artifact that is absent, and the two must not print the same.
    #:
    #: Not evidence on its own. Averaging N beats suppresses everything not locked to them
    #: by sqrt(N), so this carries a floor that grows as the beat count falls; read it with
    #: :attr:`noise_floor_uv` and :attr:`excess_power_uv2` beside it.
    residual_uv: float | None
    #: Share of the recording the beat train spans, ``None`` where no train resolved.
    #: ``residual_uv`` is an average over the beats it was given and describes only the
    #: part of the run they cover, so the two belong together: a small residual over a
    #: quarter of a run is not a corrected run.
    beat_train_coverage: float | None = None
    #: The averaging floor at this beat count, in microvolts, from the odd-even split.
    noise_floor_uv: float | None = None
    #: Locked power after the floor is subtracted, in microvolts squared. Negative means
    #: the beat-locked signal was unresolved at this averaging floor, which is a finding
    #: rather than a failure and must not be clipped to zero.
    excess_power_uv2: float | None = None
    #: Beats the average was taken over. The floor is set by this, so a residual cannot be
    #: compared across runs without it.
    n_beats: int | None = None

    @property
    def is_resolved(self) -> bool | None:
        """Whether any beat-locked signal cleared the averaging floor."""
        if self.excess_power_uv2 is None:
            return None
        return self.excess_power_uv2 > 0.0


#: Window the beat-locked average is cut over, and the baseline removed from it.
RESIDUAL_WINDOW_S = (-0.2, 0.6)
RESIDUAL_BASELINE_S = (-0.2, -0.1)
#: Where the residual is read. The ballistocardiogram follows the R peak by roughly a
#: fifth of a second, so the window opens at the peak and closes after the deflection.
RESIDUAL_MEASUREMENT_S = (0.0, 0.5)
#: Beats below which an average is not an average.
MINIMUM_RESIDUAL_BEATS = 30


def _beat_train_coverage(beat_times_s: np.ndarray, duration_s: float) -> float | None:
    """Share of the recording the beat train spans, gaps excluded.

    Gaps are counted with the same :data:`MISSED_BEAT_FACTOR` the tachogram's dropout count
    uses, so the figure and this measurement cannot disagree about which intervals were
    missed. Time before the first beat and after the last counts as uncovered too: a train
    that starts four minutes in describes nothing about the four minutes before it.
    """
    if beat_times_s.size < 2 or duration_s <= 0:
        return None
    intervals = np.diff(beat_times_s)
    median = float(np.median(intervals))
    if not np.isfinite(median) or median <= 0:
        return None
    gaps = intervals[intervals > MISSED_BEAT_FACTOR * median]
    uncovered = (
        float(beat_times_s[0]) + float(duration_s - beat_times_s[-1]) + float(gaps.sum())
    )
    return float(np.clip((duration_s - uncovered) / duration_s, 0.0, 1.0))


def compute_cardiac_residual(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    ecg_channel: str = "ECG",
    marker_description: str = PULSE_MARKER_DESCRIPTION,
    window_s: tuple[float, float] = RESIDUAL_WINDOW_S,
    baseline_s: tuple[float, float] = RESIDUAL_BASELINE_S,
    measurement_s: tuple[float, float] = RESIDUAL_MEASUREMENT_S,
) -> CardiacResidual:
    """Measure the beat-locked EEG deflection one run still carries.

    Never raises for want of a measurement. A run with no resolvable beat train reports
    ``residual_uv=None`` and keeps its marker count, because a run that cannot be measured
    still has to appear in the re-export worklist -- it is precisely the run where both
    beat sources failed.
    """
    from eeg_pipeline.preprocessing.ica_cardiac_review import (
        CardiacReviewSettings,
        UnusableEcg,
        detect_ecg_events,
    )

    marker_count = int(np.asarray(annotation_onsets(raw, marker_description)).size)
    if ecg_channel not in raw.ch_names and marker_count == 0:
        return CardiacResidual(recording_id, marker_count, None, None)

    try:
        detection = detect_ecg_events(
            # Same marker this function was told to count. The detector used to hardcode
            # the description; now that it is configurable, the caller's name is the one
            # that keeps this measurement reading the train it reports coverage for.
            raw,
            CardiacReviewSettings(
                enabled=True,
                ecg_channel=ecg_channel,
                marker_description=marker_description,
            ),
        )
    except (UnusableEcg, ValueError):
        # ValueError as well: a recording with no ECG channel fails validation inside the
        # detector, which for this measurement is the same ordinary outcome.
        return CardiacResidual(recording_id, marker_count, None, None)

    sfreq = float(raw.info["sfreq"])
    coverage = _beat_train_coverage(
        (detection.events[:, 0] - raw.first_samp) / sfreq, raw.n_times / sfreq
    )

    epochs = mne.Epochs(
        raw,
        detection.events,
        event_id=int(detection.events[0, 2]),
        tmin=window_s[0],
        tmax=window_s[1],
        baseline=tuple(baseline_s),
        picks="eeg",
        preload=True,
        reject=None,
        verbose="ERROR",
    )
    if len(epochs) < MINIMUM_RESIDUAL_BEATS:
        return CardiacResidual(recording_id, marker_count, detection.source, None, coverage)

    inside = (epochs.times >= measurement_s[0]) & (epochs.times <= measurement_s[1])
    if not inside.any():
        return CardiacResidual(recording_id, marker_count, detection.source, None, coverage)

    # The same estimator the volume-locked gradient measurement uses, for the same reason:
    # averaging N beats suppresses everything not locked to them by sqrt(N), so the raw
    # amplitude carries a floor that grows as the beat count falls. Measured here on
    # sub-0008 run-1, the previous peak-to-peak read 0.14 uV over 493 beats and 2.78 uV
    # over 59 of the same beats -- a twentyfold swing on one run's data, and at the full
    # count it sat *below* its own circular-shift null. A cohort ordering runs by that
    # number was ordering them by how well their beats were detected.
    #
    # Across-channel rather than a single channel: the ballistocardiogram is focal and
    # which sensor carries it depends on head position, so a fixed channel would measure
    # where the artifact happened to land rather than how large it was.
    measured = measure_locked_average(epochs.get_data(copy=False)[:, :, inside])
    return CardiacResidual(
        recording_id,
        marker_count,
        detection.source,
        measured.locked_rms_uv,
        coverage,
        noise_floor_uv=measured.noise_floor_uv,
        excess_power_uv2=measured.excess_power_uv2,
        n_beats=measured.n_epochs,
    )




#: Distance within which a marker and a detected peak are taken to be the same beat.
#:
#: Analyzer marks the R peak its own correction modelled, and MNE picks the sample its
#: detector settles on; the two routinely differ by tens of milliseconds on the same beat.
#: A tolerance well below the shortest plausible interval (0.3 s at 200 bpm) therefore
#: absorbs that disagreement without ever letting one beat match its neighbour.
MARKER_AGREEMENT_TOLERANCE_S = 0.1


@dataclass(frozen=True)
class MarkerAgreement:
    """How far two independent beat detectors agree about one run.

    Analyzer's pulse correction is driven by its R markers, while this pipeline's cardiac
    review detects R peaks from the ECG signal. Both are reported, in sections far enough
    apart that a disagreement between them is easy to miss — and a disagreement is not a
    detail: it says the correction whose output everything downstream inherits was driven
    by a marker train the ECG does not support.
    """

    recording_id: str
    #: Analyzer R-marker onsets, in seconds from the run start.
    marker_onsets_s: np.ndarray
    #: Signal-detected R-peak onsets, on the same timeline.
    detected_onsets_s: np.ndarray
    tolerance_s: float
    #: Detected peaks that have a marker within the tolerance, matched one-to-one.
    n_matched: int
    #: Signed offset from each detected beat to its nearest marker, in seconds.
    #:
    #: Positive means the marker came first. Unbounded by :attr:`tolerance_s` on purpose:
    #: this is the measurement that says what the matched fraction means, so restricting
    #: it to the pairs that already matched would answer only for the runs that were never
    #: in question.
    lags_s: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)

    @property
    def median_lag_s(self) -> float | None:
        """Typical offset between the two trains, or ``None`` with nothing to compare.

        The pair of numbers that separates the two readings of a low matched fraction. On
        sub-0012 run-5 the markers described the heartbeat exactly and sat 303 ms ahead of
        it, which the share alone reported as complete disagreement.
        """
        if self.lags_s.size == 0:
            return None
        return float(np.median(self.lags_s))

    @property
    def lag_iqr_s(self) -> float | None:
        """Spread of that offset: tight means a delay, broad means real disagreement."""
        if self.lags_s.size == 0:
            return None
        return float(np.percentile(self.lags_s, 75) - np.percentile(self.lags_s, 25))

    @property
    def n_markers(self) -> int:
        return int(self.marker_onsets_s.size)

    @property
    def n_detected(self) -> int:
        return int(self.detected_onsets_s.size)

    @property
    def matched_fraction(self) -> float | None:
        """Share of detected beats Analyzer also marked.

        ``None`` rather than zero when nothing was detected: the fraction is taken over
        the detected beats, so with none there is no denominator. Zero would read as
        "the detectors disagree completely", which is a different and much stronger
        claim than "there was nothing to compare".
        """
        if self.n_detected == 0:
            return None
        return self.n_matched / self.n_detected

    @property
    def marker_precision(self) -> float | None:
        """Share of Analyzer markers supported by a detected ECG beat."""
        if self.n_markers == 0:
            return None
        return self.n_matched / self.n_markers


def compute_marker_agreement(
    *,
    recording_id: str,
    marker_onsets_s: np.ndarray,
    detected_onsets_s: np.ndarray,
    tolerance_s: float = MARKER_AGREEMENT_TOLERANCE_S,
) -> MarkerAgreement:
    """Match two beat trains against each other, one beat to one beat.

    Takes onsets rather than a raw on purpose. Detection is the caller's business — the
    per-run evidence pass already runs both detectors — and a function that re-ran them
    would be measuring its own detector rather than the one whose output is in the report.

    An empty train on either side is a result, not an error. The runs this panel exists to
    surface are exactly the ones where Analyzer wrote almost no markers, so raising there
    would suppress the evidence at the only moment it matters.
    """
    markers = np.sort(np.asarray(marker_onsets_s, dtype=float))
    detected = np.sort(np.asarray(detected_onsets_s, dtype=float))
    if tolerance_s <= 0:
        raise ValueError(f"Marker agreement needs a positive tolerance, got {tolerance_s!r}.")

    # Greedy nearest-neighbour matching with each marker consumed at most once. Counting
    # "detected peaks with any marker nearby" instead would let a single marker vouch for
    # a whole burst of beats, which reports full agreement for a train that has none.
    consumed = np.zeros(markers.size, dtype=bool)
    matched = 0
    for onset in detected:
        if not markers.size:
            break
        candidates = np.flatnonzero(
            (np.abs(markers - onset) <= tolerance_s) & ~consumed,
        )
        if candidates.size == 0:
            continue
        nearest = candidates[np.argmin(np.abs(markers[candidates] - onset))]
        consumed[nearest] = True
        matched += 1

    return MarkerAgreement(
        recording_id=recording_id,
        marker_onsets_s=markers,
        detected_onsets_s=detected,
        tolerance_s=float(tolerance_s),
        n_matched=matched,
        lags_s=_nearest_marker_lags(markers=markers, detected=detected),
    )


def _nearest_marker_lags(*, markers: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Signed offset from each detected beat to the nearest marker, positive if earlier.

    Nearest rather than matched: the runs worth measuring are the ones where nothing
    matched, so a lag taken over the matched pairs would be defined only where it was
    never needed.

    Both trains are sorted by the caller, so the nearest marker is one of the two
    straddling each beat.
    """
    if markers.size == 0 or detected.size == 0:
        return np.empty(0)
    index = np.clip(np.searchsorted(markers, detected), 1, markers.size - 1)
    before = markers[index - 1]
    after = markers[index]
    nearer = np.where(np.abs(detected - before) <= np.abs(detected - after), before, after)
    return detected - nearer












def compute_run_marker_agreement(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    description: str | None = None,
    tolerance_s: float = MARKER_AGREEMENT_TOLERANCE_S,
) -> MarkerAgreement | None:
    """Reconcile one run's Analyzer markers against R peaks detected from its ECG.

    ``None`` when the run carries no channel typed ``ecg``. MNE will happily synthesize a
    surrogate ECG out of the EEG channels when asked to detect without one, and that
    surrogate is a different detector from the one whose beats the cardiac review reports.
    Comparing Analyzer's markers against it would put a number in the table that no other
    panel in the report corroborates, so the reconciliation is skipped instead.
    """
    if "ecg" not in raw.get_channel_types():
        return None

    markers = annotation_onsets(raw, description or PULSE_MARKER_DESCRIPTION)
    events, _, _ = mne.preprocessing.find_ecg_events(
        raw,
        event_id=PULSE_EVENT_ID,
        verbose="ERROR",
    )
    events = np.asarray(events, dtype=float)
    detected = (
        (events[:, 0] - raw.first_samp) / float(raw.info["sfreq"])
        if events.size
        else np.array([], dtype=float)
    )
    return compute_marker_agreement(
        recording_id=recording_id,
        marker_onsets_s=markers,
        detected_onsets_s=detected,
        tolerance_s=tolerance_s,
    )
















__all__ = [
    "AnalyzerCorrectionQc",
    "CARDIAC_ATTENUATION_QC_SUFFIX",
    "MARKER_AGREEMENT_TOLERANCE_S",
    "MarkerAgreement",
    "PULSE_MARKER_QC_SUFFIX",
    "CardiacResidual",
    "RrIntervals",
    "compute_cardiac_residual",
    "add_rr_interval_section",
    "compute_marker_agreement",
    "compute_rr_intervals",
    "load_analyzer_qc",
    "PLAUSIBLE_RR_RANGE_S",
]
