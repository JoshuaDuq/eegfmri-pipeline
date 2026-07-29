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

import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, ScalarFormatter
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.cardiac_artifact_qc import PULSE_EVENT_ID
from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION
from eeg_pipeline.preprocessing.report.annotations import annotation_onsets
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    MARK_COLOR,
    PRIMARY_COLOR,
    RUN_COLORS,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    SpanningRow,
    grid_table,
)

PULSE_MARKER_QC_SUFFIX = "desc-pulsemarkers_qc.tsv"
CARDIAC_ATTENUATION_QC_SUFFIX = "desc-cardiacattenuation_qc.tsv"

#: Beats needed before an interval series describes a rhythm rather than a few markers.
MINIMUM_BEATS = 3

#: Multiple of the run's median interval at which a single missed beat lands.
MISSED_BEAT_FACTOR = 1.5

#: Interval range a working detector on a resting or task recording stays inside.
#:
#: 0.3–2.0 s spans 200 down to 30 bpm, which covers every rate such a recording plausibly
#: contains including the extremes. This is the physiological statement; it is not the
#: axis. See :data:`DRAWN_RR_RANGE_S`.
PLAUSIBLE_RR_RANGE_S = (0.3, 2.0)

#: Interval window the tachogram panels are drawn over, in seconds, on a log axis.
#:
#: Fixed rather than taken from the data, so a run with a failed detector cannot rescale
#: the panels beside it and the same interval occupies the same height in every report.
#:
#: Wider than :data:`PLAUSIBLE_RR_RANGE_S`, and logarithmic, because clipping to the
#: plausible range censored exactly the runs the panel exists to expose. On sub-0012
#: run-1, 82 of 84 long intervals fell outside a linear 0.3–2 s window and were drawn
#: stacked on the boundary: the count reached the title, but a 2.1 s gap and a 60 s one
#: became the same mark, and the magnitude is the measurement. Logarithmic keeps the
#: ordinary rhythm legible while placing a lapse where it actually falls — the plausible
#: band still owns more than half the panel height, which is what the fixed window was
#: protecting. Samples outside even this window are drawn on the boundary and counted.
DRAWN_RR_RANGE_S = (0.3, 10.0)


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


def analyzer_qc_html(qc: AnalyzerCorrectionQc) -> str:
    """Render the per-run Analyzer correction table and any failures."""
    present = [spec for spec in _QC_COLUMNS if spec.source in qc.runs.columns]
    rows = [
        [spec.render(record.get(spec.source)) for spec in present]
        for record in qc.runs.to_dict("records")
    ]
    table = grid_table([spec.column for spec in present], rows)
    document = (
        "<p>Gradient and pulse-artifact correction were performed in BrainVision "
        "Analyzer before this pipeline ran, so their quality is an input rather than "
        "something the MNE stages can fix. The R-locked EEG amplitude either side of "
        "the pipeline's own ICA is given as before and after: a large <em>before</em> "
        "value means residual pulse artifact reached this pipeline.</p>"
        f"{table}"
    )
    if qc.runs_outside_bounds or qc.fallback_runs:
        notes = []
        if qc.runs_outside_bounds:
            notes.append(
                f"Run(s) {', '.join(qc.runs_outside_bounds)} have pulse-marker "
                "measurements outside the bounds this QC was configured with. The "
                "measured values are in the table above and in the QC sidecar; the "
                "bounds are reference values recorded beside them, not a threshold "
                "any run was excluded by."
            )
        if qc.fallback_runs:
            notes.append(
                f"Run(s) {', '.join(qc.fallback_runs)} carried no Analyzer R markers, so "
                "the R peaks used for this measurement came from automated detection on "
                "the ECG channel. That substitution affects the measurement only, not the "
                "correction Analyzer applied."
            )
        document += "<p>" + " ".join(notes) + "</p>"
    return document


def plot_analyzer_qc(qc: AnalyzerCorrectionQc) -> plt.Figure:
    """Plot residual R-locked amplitude per run, and the attenuation achieved."""
    runs = qc.runs
    if not {"before_rms_uv", "after_rms_uv"}.issubset(runs.columns):
        raise ValueError("Analyzer QC figure requires before and after R-locked amplitudes.")
    labels = [f"run-{value}" for value in runs["run"]]
    positions = np.arange(len(runs))
    fallback = (
        runs["is_fallback"].fillna(False).astype(bool).to_numpy()
        if "is_fallback" in runs.columns
        else np.zeros(len(runs), dtype=bool)
    )

    figure, (amplitude_axis, attenuation_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        layout="constrained",
    )
    # Residual amplitude spans more than an order of magnitude between runs, so the axis
    # is logarithmic and the two states are drawn as paired markers rather than bars.
    for position, before, after in zip(
        positions,
        runs["before_rms_uv"],
        runs["after_rms_uv"],
        strict=True,
    ):
        amplitude_axis.plot([position, position], [before, after], color="0.75", linewidth=1.2)
    amplitude_axis.scatter(
        positions,
        runs["before_rms_uv"],
        color=BEFORE_COLOR,
        s=34,
        label="Reaching this pipeline",
        zorder=3,
    )
    amplitude_axis.scatter(
        positions,
        runs["after_rms_uv"],
        color=AFTER_COLOR,
        s=34,
        label="After pipeline ICA",
        zorder=3,
    )
    amplitude_axis.set(
        title="R-locked EEG amplitude per run",
        ylabel="Median RMS (µV)",
        yscale="log",
        xticks=positions,
        xticklabels=labels,
    )
    amplitude_axis.grid(axis="y", alpha=0.2)
    amplitude_axis.spines[["top", "right"]].set_visible(False)
    amplitude_axis.legend(frameon=False, fontsize=8, loc="best")

    if "attenuation_db" in runs.columns:
        # The highlight is vermillion and the pre-ICA state above is orange. They were
        # the same colour until the shared palette split them: one figure using one ink
        # for "before correction" in its left panel and "used fallback detection" in its
        # right panel is a reading trap, not a convention.
        colors = [FLAG_COLOR if flag else "0.80" for flag in fallback]
        attenuation_axis.bar(positions, runs["attenuation_db"], color=colors)
        median_attenuation = float(np.nanmedian(runs["attenuation_db"]))
        attenuation_axis.axhline(
            median_attenuation,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"median {median_attenuation:.1f} dB",
        )
        attenuation_axis.set(
            title="Cardiac attenuation achieved by pipeline ICA",
            ylabel="Attenuation (dB)",
            xticks=positions,
            xticklabels=labels,
        )
        handles, _ = attenuation_axis.get_legend_handles_labels()
        if fallback.any():
            # Explain the highlight in the legend, beside the bars it marks, rather than
            # in a second title line the eye has already left by the time it reaches them.
            handles.append(Patch(facecolor=FLAG_COLOR, label="fallback R-peak detection"))
        attenuation_axis.legend(handles=handles, frameon=False, fontsize=8)
        attenuation_axis.grid(axis="y", alpha=0.2)
        attenuation_axis.spines[["top", "right"]].set_visible(False)
    for axis in (amplitude_axis, attenuation_axis):
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.suptitle(
        f"sub-{html.unescape(qc.subject)} · BrainVision Analyzer correction quality",
        fontsize=10,
    )
    plt.close(figure)
    return figure


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
    #: Peak-to-peak of the across-channel RMS of the beat-locked average, in microvolts.
    #: ``None`` where no beat train resolved -- an artifact that could not be measured is
    #: not an artifact that is absent, and the two must not print the same.
    residual_uv: float | None
    #: Share of the recording the beat train spans, ``None`` where no train resolved.
    #: ``residual_uv`` is an average over the beats it was given and describes only the
    #: part of the run they cover, so the two belong together: a small residual over a
    #: quarter of a run is not a corrected run.
    beat_train_coverage: float | None = None


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

    marker_count = int(np.asarray(annotation_onsets(raw, PULSE_MARKER_DESCRIPTION)).size)
    if ecg_channel not in raw.ch_names and marker_count == 0:
        return CardiacResidual(recording_id, marker_count, None, None)

    try:
        detection = detect_ecg_events(
            raw, CardiacReviewSettings(enabled=True, ecg_channel=ecg_channel)
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
        tmin=RESIDUAL_WINDOW_S[0],
        tmax=RESIDUAL_WINDOW_S[1],
        baseline=RESIDUAL_BASELINE_S,
        picks="eeg",
        preload=True,
        reject=None,
        verbose="ERROR",
    )
    if len(epochs) < MINIMUM_RESIDUAL_BEATS:
        return CardiacResidual(recording_id, marker_count, detection.source, None, coverage)

    evoked = epochs.average()
    # Across-channel RMS rather than a single channel: the ballistocardiogram is focal and
    # which sensor carries it depends on head position, so a fixed channel would measure
    # where the artifact happened to land rather than how large it was.
    rms = np.sqrt((evoked.get_data() ** 2).mean(axis=0)) * 1e6
    inside = (evoked.times >= RESIDUAL_MEASUREMENT_S[0]) & (
        evoked.times <= RESIDUAL_MEASUREMENT_S[1]
    )
    if not inside.any():
        return CardiacResidual(recording_id, marker_count, detection.source, None, coverage)
    return CardiacResidual(
        recording_id, marker_count, detection.source, float(np.ptp(rms[inside])), coverage
    )


@dataclass(frozen=True)
class RrIntervals:
    """Beat-to-beat intervals derived from one run's R markers."""

    recording_id: str
    #: Onset of each beat, in seconds from the run start.
    beat_times_s: np.ndarray
    #: Interval preceding each beat after the first, in seconds.
    intervals_s: np.ndarray

    @property
    def median_interval_s(self) -> float:
        return float(np.median(self.intervals_s))

    @property
    def median_bpm(self) -> float:
        return 60.0 / self.median_interval_s

    @property
    def dropout_threshold_s(self) -> float:
        """Interval above which a beat was more likely missed than merely slow."""
        return MISSED_BEAT_FACTOR * self.median_interval_s

    @property
    def dropout_count(self) -> int:
        """Intervals long enough to be a missed beat rather than a slow one.

        A detector that misses one beat produces an interval near twice the median, so a
        threshold below that catches it while leaving ordinary variability alone. This
        separates "this participant's heart rate varied" from "the detector lost beats",
        which the median rate alone cannot distinguish and which decides whether the
        pulse correction had a complete marker set to work from.
        """
        return int(np.sum(self.intervals_s > self.dropout_threshold_s))


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


def marker_agreement_html(agreements: Sequence[MarkerAgreement]) -> str:
    """Render the two beat counts per run, side by side."""
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Analyzer markers"),
        Column("Beats detected from ECG"),
        Column("Matched"),
        Column("Share of detected beats marked"),
        Column("Nearest marker (ms)"),
        Column("Lag IQR (ms)"),
    )
    rows = [
        [
            run_label(agreement.recording_id),
            agreement.n_markers,
            agreement.n_detected,
            agreement.n_matched,
            None if agreement.matched_fraction is None else f"{agreement.matched_fraction:.1%}",
            None if agreement.median_lag_s is None else f"{agreement.median_lag_s * 1000:+.0f}",
            None if agreement.lag_iqr_s is None else f"{agreement.lag_iqr_s * 1000:.0f}",
        ]
        for agreement in agreements
    ]
    return (
        "<p>Two detectors, one heartbeat. Analyzer's R markers drove the pulse-artifact "
        "correction that ran before this pipeline; the R peaks in the third column were "
        "detected here from the ECG signal itself. The correction can only be as good as "
        "the marker train it was given, so the comparison says whether that train "
        "described the heartbeat the ECG recorded.</p>"
        + grid_table(columns, rows)
        + "<p>A beat counts as matched when a marker falls within "
        f"{MARKER_AGREEMENT_TOLERANCE_S * 1000:.0f} ms of it, and each marker is spent on "
        "at most one beat. The last column is a share of the beats the ECG shows, so it "
        "falls when Analyzer marked fewer beats than occurred; it is left blank when no "
        "beats were detected, because then there is nothing to take a share of.</p>"
        "<p>The last two columns say what a low share is made of. They give the signed "
        "distance from each detected beat to the nearest marker &mdash; positive when the "
        "marker came first &mdash; as a median and an interquartile range, over every "
        "beat rather than only the matched ones. A median well outside the tolerance with "
        "a narrow range means the two detectors found the same heartbeat and disagree "
        "about where in the beat to put it, which a share taken at a fixed tolerance "
        "reports as total disagreement. A wide range means they genuinely disagree.</p>"
        "<p>What a low share means is not decided here. Analyzer marking few beats while "
        "the ECG shows many says the correction ran on an incomplete train. The reverse "
        "&mdash; markers without detected beats &mdash; more often points at the ECG "
        "trace itself, which the beat-detection panels below show directly.</p>"
    )


#: Width of the window each detector's beat count is expressed over, in seconds.
#:
#: Wide enough that a healthy train gives a steady rate rather than a count that swings
#: between eight and nine beats, and narrow enough to place the moment a train stops to
#: within a few seconds of it.
MARKER_RATE_BIN_S = 10.0


def _binned_rate(onsets_s: np.ndarray, edges_s: np.ndarray) -> np.ndarray:
    """Beats per minute in each window, as a rate rather than a count."""
    counts, _ = np.histogram(onsets_s, bins=edges_s)
    return counts * (60.0 / MARKER_RATE_BIN_S)


def plot_marker_agreement(agreements: Sequence[MarkerAgreement]) -> plt.Figure:
    """Draw both detectors' beat rate against time, one panel per run.

    A total says the marker train was short; only the time axis says *when* it stopped, and
    a train that never started and one that lost the trace partway are different faults
    with different implications for the correction either side of that moment.

    The rate is binned rather than drawn one tick per beat. A run holds several hundred
    beats over a few hundred seconds, which is more events than the axis has pixels: drawn
    individually they alias, and the interference banding reads as structure in the marker
    train that is not in the data. Binning also puts both detectors in the same unit, so
    the panel answers "were beats being marked at this moment, at the rate the ECG shows"
    rather than leaving two tick rows to be compared by eye.
    """
    if not agreements:
        raise ValueError("The marker agreement figure needs at least one run.")

    figure, axes = plt.subplots(
        len(agreements),
        1,
        figsize=(11.0, 1.9 * len(agreements) + 0.8),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    for axis, agreement in zip(axes[:, 0], agreements, strict=True):
        onsets = np.concatenate([agreement.marker_onsets_s, agreement.detected_onsets_s])
        stop = float(onsets.max()) if onsets.size else MARKER_RATE_BIN_S
        # Whole windows only. A run rarely ends on a boundary, and counting the short
        # remainder at the full window's rate drove the trace to the floor at the right
        # edge of every panel — a collapse manufactured by where the recording stopped,
        # in the one figure whose subject is when a train really did stop.
        n_windows = max(int(stop // MARKER_RATE_BIN_S), 1)
        edges = np.arange(n_windows + 1) * MARKER_RATE_BIN_S
        centres = edges[:-1] + MARKER_RATE_BIN_S / 2.0
        rates = []
        for onsets_s, color, label in (
            (agreement.detected_onsets_s, PRIMARY_COLOR, "Detected from ECG"),
            (agreement.marker_onsets_s, BEFORE_COLOR, "Analyzer markers"),
        ):
            # Steps, not a smooth line: the rate is constant within each window by
            # construction, and interpolating between centres would draw a beat rate at
            # moments where none was measured.
            rate = _binned_rate(onsets_s, edges)
            rates.append(rate)
            axis.step(
                centres,
                rate,
                where="mid",
                color=color,
                linewidth=1.1,
                label=label,
            )
        # A run where Analyzer marked nothing draws one lone trace, and a reader who does
        # not check the legend reads a single-detector panel as agreement. Saying it on
        # the panel also states what the lone trace is evidence *of*: the ECG carried a
        # heartbeat that was there to be marked, so the gap is in the marker train and
        # not in the physiology.
        if agreement.n_markers == 0 and agreement.n_detected > 0:
            axis.annotate(
                "Analyzer marked no beats in this run — the ECG trace is what it missed",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=7.5,
                color=FLAG_COLOR,
            )
        fraction = agreement.matched_fraction
        share = "no beats detected" if fraction is None else f"{fraction:.1%} of beats marked"
        title = (
            f"{run_label(agreement.recording_id)} · "
            f"{agreement.n_markers} markers · {agreement.n_detected} detected · {share}"
        )
        # What a low share means, stated beside it. A tight lag says the two trains
        # describe the same heartbeat at an offset; a broad one says they disagree. On
        # sub-0012 run-5 the share was 0.0% and the lag was 303 ms with a 19 ms spread,
        # and the panel gave a reader no way to tell those apart.
        median_lag = agreement.median_lag_s
        if median_lag is not None and agreement.n_matched < agreement.n_detected:
            title += (
                f"\nnearest marker {median_lag * 1000:+.0f} ms away "
                f"(IQR {agreement.lag_iqr_s * 1000:.0f} ms)"
            )
        # Zero is on the axis only when a detector reached it. Anchoring there always
        # spent half the panel on rates neither trace visits, while the comparison the
        # panel exists for is between two traces a few beats per minute apart.
        observed = np.concatenate(rates) if rates else np.zeros(1)
        floor = 0.0 if observed.min() <= 0.0 else max(float(observed.min()) - 10.0, 0.0)
        axis.set(
            title=title,
            ylabel="Beats per min",
            ylim=(floor, None),
        )
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    # Below the grid rather than inside the first panel. Placed in-axes it sat on top of
    # the marker trace of whichever run came first, which on sub-0012 was the run with the
    # sparsest markers — the one the panel most needed to show.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
    )
    axes[-1, 0].set_xlabel(f"Time in run (s) · beats counted in {MARKER_RATE_BIN_S:.0f} s windows")
    plt.close(figure)
    return figure


def compute_run_marker_agreement(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    description: str | None = None,
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
    )


def add_marker_agreement_section(
    *,
    report: mne.Report,
    agreements: Sequence[MarkerAgreement],
    section: str = "Scanner artifact correction (Analyzer)",
) -> None:
    """Append the two-detector reconciliation, when there is a second detector to compare.

    A dataset recorded outside a scanner has no Analyzer stage and therefore no R markers,
    so there is one detector and nothing to reconcile; the section is omitted rather than
    rendered as a table of zeros, which would read as total disagreement. The gate is any
    marker anywhere in the session, not markers in every run: a train that collapsed in
    half the runs is precisely what this panel is for.
    """
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not any(agreement.n_markers for agreement in agreements):
        return

    remove_tagged_content(report, tag="marker-agreement")
    report.add_html(
        html=marker_agreement_html(agreements),
        title="Analyzer markers against the recorded ECG",
        section=section,
        tags=("raw", "marker-agreement"),
        replace=True,
    )
    report.add_figure(
        fig=plot_marker_agreement(agreements),
        title="Both beat trains over time",
        section=section,
        tags=("raw", "marker-agreement"),
        image_format=report_image_format(),
        replace=True,
    )
    move_tagged_content_before(report, tag="marker-agreement", anchor=before_raw_sections)


def compute_rr_intervals(
    raw: mne.io.BaseRaw,
    *,
    recording_id: str,
    description: str | None = None,
) -> RrIntervals | None:
    """Extract beat-to-beat intervals from the R markers in one run.

    Bad pulse correction traces back to bad R detection, and the QC table records only a
    marker count and a median rate. Neither shows a detector that worked for four
    minutes and then lost the trace, which is what the interval series makes visible.
    """
    onsets = annotation_onsets(raw, description or PULSE_MARKER_DESCRIPTION)
    if onsets.size < MINIMUM_BEATS:
        return None
    intervals = np.diff(onsets)
    if not np.all(intervals > 0):
        raise ValueError(f"{recording_id}: R markers are not strictly increasing.")
    return RrIntervals(
        recording_id=recording_id,
        beat_times_s=onsets[1:],
        intervals_s=intervals,
    )


def plot_rr_intervals(
    series: Sequence[RrIntervals],
    *,
    missing: Sequence[str] = (),
) -> plt.Figure:
    """Plot the beat-to-beat interval series for every run.

    ``missing`` names runs whose marker train was too short to describe a rhythm. They are
    listed rather than dropped: a reader who sees runs 1, 3, 5 and 6 cannot tell whether
    runs 2 and 4 were not acquired, were not measured, or failed, and the answer decides
    whether the pulse correction had markers to work from at all.

    Every panel shares one fixed logarithmic window, :data:`DRAWN_RR_RANGE_S`, so a run
    can be read against its neighbours.

    The window is a constant rather than the range of the data. Letting the data set it
    meant one run whose detector had collapsed — seven markers across eight minutes,
    intervals of two minutes — stretched the shared axis, and the ordinary beat-to-beat
    variation of every working run was flattened into a band a few pixels tall. What the
    figure resolves does not depend on which runs happen to share it.

    It is logarithmic, and wider than :data:`PLAUSIBLE_RR_RANGE_S`, because a linear
    window clipped to the plausible range censored the runs the panel exists to expose:
    on sub-0012 run-1, 82 of 84 long intervals landed on the boundary, which reported how
    many there were and not how long any of them was. The plausible band still occupies
    more than half the height, so the rhythm stays readable.

    Intervals outside even this window are drawn as markers on the boundary they exceeded
    and counted in the panel title, so a detector that collapsed entirely still reads as
    collapsed, and no threshold decides which runs "look like a rhythm".
    """
    if not series:
        raise ValueError("The tachogram requires at least one run with R markers.")
    low, high = DRAWN_RR_RANGE_S
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(10.0, 1.9 * len(series) + 1.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, run in zip(axes[:, 0], series, strict=True):
        minutes = run.beat_times_s / 60.0
        intervals = run.intervals_s
        inside = (intervals >= low) & (intervals <= high)
        axis.plot(
            minutes,
            np.where(inside, intervals, np.nan),
            color=PRIMARY_COLOR,
            linewidth=0.7,
            marker=".",
            markersize=1.6,
        )
        # Drawn on the boundary rather than dropped, and in the annotation colour rather
        # than the series colour, so a clipped sample cannot be mistaken for a measured
        # one sitting at the edge of the range.
        for outside, edge, marker in (
            (intervals > high, high, "^"),
            (intervals < low, low, "v"),
        ):
            if outside.any():
                axis.plot(
                    minutes[outside],
                    np.full(int(outside.sum()), edge),
                    linestyle="none",
                    marker=marker,
                    markersize=4.0,
                    color=MARK_COLOR,
                    clip_on=False,
                )
        axis.axhline(run.median_interval_s, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
        # The same threshold the dropout count uses, so the figure and the table cannot
        # disagree about which intervals were counted.
        axis.axhline(run.dropout_threshold_s, color=FLAG_COLOR, linestyle=":", linewidth=1.0)
        clipped = int((~inside).sum())
        title = (
            f"{run_label(run.recording_id)} · {run.intervals_s.size + 1} beats · "
            f"median {run.median_bpm:.0f} bpm · {run.dropout_count} interval(s) "
            f"above {MISSED_BEAT_FACTOR:g}× the median"
        )
        if clipped:
            title += f" · {clipped} outside the drawn range"
        # The band a working detector stays inside, drawn so the widened axis still says
        # where "plausible" ends without clipping anything to it.
        axis.axhspan(
            *PLAUSIBLE_RR_RANGE_S,
            color=GUIDE_COLOR,
            alpha=0.07,
            linewidth=0,
            zorder=0,
        )
        axis.set(title=title, ylabel="RR (s)", ylim=(low, high), yscale="log")
        # Plain seconds rather than the powers of ten a log axis labels by default: the
        # reader is comparing intervals against a heart rate, and "10^0" is not a number
        # anyone converts to bpm in their head.
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_yticks([0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time in run (min)")
    plausible_low, plausible_high = PLAUSIBLE_RR_RANGE_S
    caption = (
        "Beat-to-beat intervals from the R markers · dashed line is the run median\n"
        f"dotted line is {MISSED_BEAT_FACTOR:g}× the median, above which an interval is "
        "counted as a missed beat\n"
        f"log axis fixed to {low:g}–{high:g} s; shading marks the {plausible_low:g}–"
        f"{plausible_high:g} s a working detector stays inside\n"
        "triangles mark intervals outside the axis, drawn on the boundary they exceeded"
    )
    if missing:
        caption += (
            "\nNo interval series for "
            + ", ".join(run_label(recording_id) for recording_id in missing)
            + f": fewer than {MINIMUM_BEATS} R markers were found"
        )
    figure.suptitle(caption, fontsize=9)
    plt.close(figure)
    return figure


def plot_rr_poincare(series: Sequence[RrIntervals]) -> plt.Figure:
    """Plot each interval against the one after it, for every run.

    The time series answers "when did detection go wrong". This answers "what went
    wrong", which it cannot: a missed beat and a spuriously doubled one both appear there
    as a single point away from the median, and they call for opposite responses.

    Here they separate. Detection is self-correcting in a specific way — a missed beat
    merges two intervals into one near twice the median and the next interval is normal,
    so the pair lands on the horizontal 2× reference; a double detection splits one
    interval into two halves, landing the pair on the 0.5× reference. Genuine
    variability, which changes both intervals of a pair together, stays on the identity
    line. Reference lines are drawn for all three, and interpreting the scatter against
    them is left to the reviewer: an arrhythmia can put points off the identity line too.

    Intervals are pooled per run and coloured by run, so a detector that failed in one
    run only is visible as a cloud of one colour away from the diagonal.
    """
    if not series:
        raise ValueError("The Poincaré plot requires at least one run with R markers.")
    low, high = PLAUSIBLE_RR_RANGE_S
    figure, axis = plt.subplots(figsize=(5.6, 5.4), layout="constrained")

    for index, run in enumerate(series):
        intervals = run.intervals_s
        if intervals.size < 2:
            continue
        axis.scatter(
            intervals[:-1],
            intervals[1:],
            s=6,
            alpha=0.55,
            linewidths=0.0,
            color=RUN_COLORS[index % len(RUN_COLORS)],
            label=run_label(run.recording_id),
        )

    reference = np.array([low, high])
    # The detection-failure guides carry MARK_COLOR rather than FLAG_COLOR: the scatter
    # already spends the hues on runs, and RUN_COLORS hands vermillion to the second run,
    # so a vermillion reference line and run-2's cloud were the same ink. They also take
    # different dash patterns, because two guides that differ only in slope are told
    # apart by their legend entries otherwise, and the legend is not beside the line.
    for factor, style, color, label in (
        (1.0, "--", GUIDE_COLOR, "RRₙ₊₁ = RRₙ (no change)"),
        (2.0, ":", MARK_COLOR, "2× — one beat missed"),
        (0.5, "-.", MARK_COLOR, "0.5× — one beat counted twice"),
    ):
        axis.plot(
            reference,
            np.clip(reference * factor, low, high),
            color=color,
            linestyle=style,
            linewidth=1.0,
            label=label,
        )
    axis.set(
        title="Each interval against the next",
        xlabel="RRₙ (s)",
        ylabel="RRₙ₊₁ (s)",
        xlim=(low, high),
        ylim=(low, high),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    plt.close(figure)
    return figure


def rr_intervals_html(series: Sequence[RrIntervals], *, missing: Sequence[str] = ()) -> str:
    """Render the per-run beat detection record.

    Runs named in ``missing`` get a row stating that no series could be built, so the
    table lists every run that was measured rather than only those that succeeded.
    """
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Beats"),
        Column("Median (bpm)"),
        Column("RR 5th–95th percentile (s)"),
        Column(f"Intervals above {MISSED_BEAT_FACTOR:g}× median"),
    )
    rows: list[Sequence[object] | SpanningRow] = [
        [
            run_label(run.recording_id),
            run.intervals_s.size + 1,
            f"{run.median_bpm:.0f}",
            f"{float(np.percentile(run.intervals_s, 5)):.2f}–"
            f"{float(np.percentile(run.intervals_s, 95)):.2f}",
            run.dropout_count,
        ]
        for run in series
    ]
    rows += [
        SpanningRow(
            lead=[run_label(recording_id)],
            note=f"No interval series: fewer than {MINIMUM_BEATS} R markers",
        )
        for recording_id in missing
    ]
    return (
        "<p>Pulse-artifact correction can only be as good as the R markers it was "
        "driven by, and a marker count with a median rate cannot show a detector that "
        "worked for part of a run and then lost the trace. These are the intervals "
        "themselves.</p>"
        + grid_table(columns, rows)
        + "<p>A missed beat produces an interval near twice the median, so the last "
        "column separates heart-rate variability from detection dropout. Interpreting "
        "the count is left to the reviewer: a run with genuine arrhythmia and a run "
        "with a failing detector both raise it, and only the ECG trace distinguishes "
        "them.</p>"
    )


def add_rr_interval_section(
    *,
    report: mne.Report,
    series: Sequence[RrIntervals],
    missing: Sequence[str] = (),
    section: str = "Scanner artifact correction (Analyzer)",
) -> None:
    """Append the beat-detection record to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not series:
        raise ValueError("The tachogram requires at least one run with R markers.")
    remove_tagged_content(report, tag="rr-intervals")
    report.add_html(
        html=rr_intervals_html(series, missing=missing),
        title="Beat detection by run",
        section=section,
        tags=("raw", "rr-intervals"),
        replace=True,
    )
    report.add_figure(
        fig=plot_rr_intervals(series, missing=missing),
        title="Beat-to-beat intervals",
        section=section,
        tags=("raw", "rr-intervals"),
        image_format=report_image_format(),
        replace=True,
    )
    report.add_figure(
        fig=plot_rr_poincare(series),
        title="Each interval against the next",
        section=section,
        tags=("raw", "rr-intervals"),
        image_format=report_image_format(),
        replace=True,
    )
    move_tagged_content_before(report, tag="rr-intervals", anchor=before_raw_sections)


def add_analyzer_correction_review(
    *,
    report: mne.Report,
    qc_dir: Path,
    task: str,
    subject: str,
    section: str = "Scanner artifact correction (Analyzer)",
) -> AnalyzerCorrectionQc | None:
    """Append Analyzer correction evidence to a subject report, if any exists."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    qc = load_analyzer_qc(qc_dir=qc_dir, task=task, subject=subject)
    if qc is None:
        return None

    remove_tagged_content(report, tag="analyzer-correction")
    report.add_html(
        html=analyzer_qc_html(qc),
        title="Analyzer correction quality by run",
        section=section,
        tags=("raw", "analyzer-correction"),
        replace=True,
    )
    if {"before_rms_uv", "after_rms_uv"}.issubset(qc.runs.columns):
        report.add_figure(
            fig=plot_analyzer_qc(qc),
            title="Residual pulse artifact and attenuation by run",
            section=section,
            tags=("raw", "analyzer-correction"),
            image_format=report_image_format(),
            replace=True,
        )
    # This describes the data entering the pipeline, so it belongs ahead of the raw
    # sections rather than after everything the pipeline then did to it.
    move_tagged_content_before(
        report,
        tag="analyzer-correction",
        anchor=before_raw_sections,
    )
    return qc


__all__ = [
    "AnalyzerCorrectionQc",
    "CARDIAC_ATTENUATION_QC_SUFFIX",
    "MARKER_AGREEMENT_TOLERANCE_S",
    "MarkerAgreement",
    "PULSE_MARKER_QC_SUFFIX",
    "CardiacResidual",
    "RrIntervals",
    "compute_cardiac_residual",
    "add_analyzer_correction_review",
    "add_marker_agreement_section",
    "add_rr_interval_section",
    "analyzer_qc_html",
    "compute_marker_agreement",
    "compute_rr_intervals",
    "load_analyzer_qc",
    "marker_agreement_html",
    "plot_analyzer_qc",
    "plot_marker_agreement",
    "plot_rr_intervals",
    "plot_rr_poincare",
    "DRAWN_RR_RANGE_S",
    "PLAUSIBLE_RR_RANGE_S",
    "rr_intervals_html",
]
