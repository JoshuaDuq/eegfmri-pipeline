"""Find the stretches Analyzer left unmarked, and recover the beats inside them.

Analyzer detects conservatively: what it marks sits on the QRS complex more reliably than
any general-purpose detector measured on this cohort, but it marks too little. Across the
104 exports, 85 recordings contain at least one RR interval above 2 s, totalling 6,954 s
with a largest single gap of 53.9 s. An 11 s interval is not a heartbeat rate, so the
gaps are provable from Analyzer's own markers with no detector involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from studies.pain_study.analysis.bcg.metrics import epoch_stack, lock_ratio

MINIMUM_SEED_BEATS = 8


@dataclass(frozen=True)
class Gap:
    start_s: float
    end_s: float
    preceding_beat_s: float
    following_beat_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def read_analyzer_beats(vhdr_path: Path | str) -> np.ndarray:
    """Onsets in seconds of Analyzer's R markers."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    onsets = [
        onset
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
        if description.split("/")[-1].strip() == "R"
    ]
    return np.asarray(sorted(onsets), dtype=float)


#: The longest interval a beating heart is taken to produce, 40 bpm -- the same rate the
#: cardiac-gaps workflow already treats as the floor of plausibility. Used to cap estimates
#: of the beat period that are read off a marker train, since a train missing most of its
#: beats reports a period several times too long.
MAXIMUM_BEAT_PERIOD_S = 60.0 / 40.0


def find_gaps(
    beat_seconds: np.ndarray,
    *,
    minimum_seconds: float = 1.2,
    factor: float = 1.5,
    baseline_percentile: float = 25.0,
    maximum_baseline_s: float = MAXIMUM_BEAT_PERIOD_S,
) -> list[Gap]:
    """Intervals that are both absolutely long and long for this run.

    Both tests are required. The absolute floor alone flags ordinary bradycardia; the
    relative test alone flags a run whose beat interval cannot be estimated at all.

    **The relative test reads a low percentile of the intervals, not the median, and the
    multiple is below two.** Both were wrong for the commonest failure. One missed beat
    produces an interval of about twice the beat-to-beat interval, so a threshold at twice
    the rate is placed exactly where it cannot see them; on sub-0012 that hid 87% of the
    gaps and 85% of the missing time, and in five of its six runs it attempted nothing at
    all. The median compounds it, because a run missing many beats has an inflated one --
    sub-0012 run 1 reads 0.998 s against a true 0.85 s, which lifts the threshold past its
    own gaps. That is the very condition the relative test exists to survive, so the
    baseline is taken low in the distribution where consecutive marked beats still sit.
    ``physiological_floor`` already caps against an inflated median for the same reason.

    Being permissive here is cheap: this only proposes where to look, and a proposal with
    no beat in it yields nothing, because ``recover_beats`` still requires a QRS template
    correlation and a physiological floor before any beat is accepted.
    """
    beats = np.sort(np.asarray(beat_seconds, dtype=float))
    if beats.size < 3:
        return []
    intervals = np.diff(beats)
    # Capped as well as taken low in the distribution. The percentile survives a train
    # missing *some* of its beats; it cannot survive one missing most of them, where every
    # interval is inflated and no percentile of them is a beat period. sub-0008 run 4 marks
    # one beat in four, so its 25th percentile reads 2.9 s against a true 1.0 s and the
    # threshold lands above the 4 s intervals the missing beats are hiding in -- 27 gaps
    # found where the run is 92% gap. A heart does not beat slower than the workflow's own
    # floor of plausibility, so neither does the baseline.
    baseline = min(float(np.percentile(intervals, baseline_percentile)), maximum_baseline_s)
    threshold = max(minimum_seconds, factor * baseline)
    return [
        Gap(
            start_s=float(beats[index]),
            end_s=float(beats[index + 1]),
            preceding_beat_s=float(beats[index]),
            following_beat_s=float(beats[index + 1]),
        )
        for index in np.flatnonzero(intervals > threshold)
    ]


def gap_summary(
    beat_seconds: np.ndarray,
    duration_s: float,
    *,
    minimum_seconds: float = 1.2,
    factor: float = 1.5,
) -> dict[str, float]:
    """Per-run gap totals, including how many beats the gaps imply are missing."""
    beats = np.sort(np.asarray(beat_seconds, dtype=float))
    gaps = find_gaps(beats, minimum_seconds=minimum_seconds, factor=factor)
    intervals = np.diff(beats) if beats.size > 1 else np.array([np.nan])
    median_rr = float(np.median(intervals)) if beats.size > 1 else float("nan")
    total = float(sum(gap.duration_s for gap in gaps))
    return {
        "n_beats": float(beats.size),
        "n_gaps": float(len(gaps)),
        "gap_seconds": total,
        "gap_fraction": total / duration_s if duration_s else float("nan"),
        "median_rr_s": median_rr,
        "max_rr_s": float(intervals.max()) if intervals.size else float("nan"),
        "implied_missing_beats": total / median_rr if median_rr else float("nan"),
    }


@dataclass(frozen=True)
class RecoverySettings:
    template_window: tuple[float, float] = (-0.2, 0.4)
    correlation_threshold: float = 0.5
    refractory_fraction: float = 0.5
    iterations: int = 2
    #: Gap-detection thresholds, matching ``find_gaps``. Lowered from 2.0/2.0 once the
    #: commonest gap turned out to be a single missed beat, which is about twice the beat
    #: interval and so invisible to a threshold set at twice the rate.
    minimum_seconds: float = 1.2
    factor: float = 1.5
    refractory_percentile: float = 1.0
    refractory_cap_fraction: float = 0.75
    #: Clear Analyzer's own double marks before searching for gaps. On by default because
    #: leaving them in place silently loosens ``physiological_floor`` on the runs whose
    #: marks are least trustworthy, and because a mark where no beat is makes the
    #: correction subtract a pulse template against nothing.
    clear_double_marks: bool = True


#: Fraction of the modal interval below which an interval is not a heartbeat but a marker
#: landing twice inside one cycle. Chosen well below the shortest interval ordinary
#: heart-rate variability produces, so only a genuine second mark trips it.
DOUBLE_MARK_FRACTION = 0.65

#: Mass at twice the modal interval, relative to the mode's own, above which the mode is
#: read as a split cycle rather than the beat period. Set below 0.5 because a run that
#: doubles every cycle puts exactly half as many intervals at the period as at the split,
#: and that case must still resolve to the period.
SPLIT_CYCLE_RATIO = 0.45


def modal_interval(
    beat_seconds: np.ndarray,
    *,
    window_s: float = 0.06,
    minimum_s: float = 0.30,
    maximum_s: float = MAXIMUM_BEAT_PERIOD_S,
) -> float:
    """The period this heart actually beats at, as the densest interval in the train.

    Neither the median nor a low percentile survives the two failures that matter here.
    A train missing most of its beats has an inflated median -- sub-0008 run 4 reads
    4.07 s against a true 1.0 s -- while a train that marks some cycles twice has a
    deflated low percentile. The mode is set by whichever spacing occurs most often, which
    under both failures is still the beat-to-beat interval, because most cycles are marked
    once and only once.

    Returns nan when there are too few intervals to have a mode.
    """
    intervals = np.diff(np.sort(np.asarray(beat_seconds, dtype=float)))
    if intervals.size < 5:
        return float("nan")
    grid = np.arange(minimum_s, maximum_s, 0.005)
    counts = np.array([np.sum((intervals >= edge) & (intervals < edge + window_s)) for edge in grid])
    edge = float(grid[int(np.argmax(counts))])
    # The window only locates the cluster; the estimate is the median of what it caught.
    # Taking the window's own midpoint instead biases the result by up to half its width,
    # because `argmax` settles on the first of the several windows that tie when the
    # intervals are tightly grouped.
    caught = intervals[(intervals >= edge) & (intervals < edge + window_s)]
    estimate = float(np.median(caught)) if caught.size else float(edge + window_s / 2.0)

    # A run that marks most of its cycles twice puts more intervals at half the beat period
    # than at the period itself, and the plain mode then reports the split -- on sub-0005
    # run 4 it reads 0.53 s against a 1.05 s beat, after which no interval looks short and
    # the double marks become invisible to every test downstream. The asymmetry that makes
    # this safe to correct: marking a cycle twice creates a cluster at a *sub*-multiple of
    # the period, while nothing creates one at a multiple -- a missed beat spreads intervals
    # across 2P, 3P, ... rather than piling them at 2P. So comparable mass at twice the mode
    # means the mode is the split, and never the reverse.
    # The neighbourhood is searched rather than the doubling point itself, because a split
    # is not symmetric: Analyzer's two marks sit 0.53 s and 0.50 s apart on sub-0005, so the
    # period is a little under twice the denser half and a window centred on exactly twice
    # it clips the cluster. Run 5 missed by one interval that way.
    if 1.7 * estimate <= maximum_s:
        lower, upper = 1.7 * estimate, 2.3 * estimate
        span = grid[(grid >= lower - window_s) & (grid <= upper)]
        if span.size:
            span_counts = np.array(
                [np.sum((intervals >= edge) & (intervals < edge + window_s)) for edge in span]
            )
            best = float(span[int(np.argmax(span_counts))])
            if span_counts.max() >= SPLIT_CYCLE_RATIO * caught.size:
                near = intervals[(intervals >= best) & (intervals < best + window_s)]
                if near.size:
                    estimate = float(np.median(near))
    return estimate


@dataclass(frozen=True)
class DoubleMarkResult:
    kept: np.ndarray
    dropped: np.ndarray
    modal_interval_s: float


def _beat_scores(
    ecg_uv: np.ndarray,
    beats: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
    template: np.ndarray,
) -> np.ndarray:
    """Correlation of each beat's own ECG window against the template, index-aligned.

    Written out rather than routed through ``epoch_stack`` because that drops epochs
    running off either end, which would silently shift every score onto the wrong beat.
    """
    centred = template - template.mean()
    norm = np.linalg.norm(centred)
    starts = np.round(np.asarray(beats) * sfreq).astype(int) + int(round(window[0] * sfreq))
    length = centred.size
    scores = np.full(beats.size, -np.inf)
    if norm == 0:
        return scores
    for position, start in enumerate(starts):
        if start < 0 or start + length > ecg_uv.size:
            continue
        segment = ecg_uv[start : start + length]
        segment = segment - segment.mean()
        magnitude = np.linalg.norm(segment)
        scores[position] = float(segment @ centred / (magnitude * norm)) if magnitude else 0.0
    return scores


def drop_double_marks(
    ecg_uv: np.ndarray,
    beat_seconds: np.ndarray,
    sfreq: float,
    *,
    window: tuple[float, float] = (-0.2, 0.4),
    fraction: float = DOUBLE_MARK_FRACTION,
) -> DoubleMarkResult:
    """Remove marks that land twice inside one cardiac cycle, keeping the better-locked one.

    Analyzer marks the QRS more reliably than any general-purpose detector measured on this
    cohort, but on some runs it also marks the magnetohydrodynamic deflection riding the
    T-wave, giving one cycle two marks. Cohort-wide there are 180 such cycles, all six
    sub-0005 runs and sub-0007 runs 3 and 5, and the gap recovery leaves every one of them
    in place because it only ever adds beats.

    They are worth removing for two reasons. A mark where no beat is makes the correction
    subtract a pulse template against nothing, injecting artifact rather than removing it.
    And they corrupt ``physiological_floor``: their short intervals *are* the low percentile
    it reads, so the filter meant to reject beats that are too close together is disarmed by
    exactly the runs that need it.

    Which of the two marks to drop is decided by the ECG, not by position: the template is
    built from the cycles that are marked once, and whichever of the pair correlates less
    with it is the one that goes. Marks are removed one at a time, shortest interval first,
    so a cycle carrying more than two survives the process with one mark left.
    """
    beats = np.sort(np.asarray(beat_seconds, dtype=float))
    modal = modal_interval(beats)
    empty = np.empty(0)
    if beats.size < 5 or not np.isfinite(modal):
        return DoubleMarkResult(kept=beats, dropped=empty, modal_interval_s=modal)

    threshold = fraction * modal
    intervals = np.diff(beats)
    if not np.any(intervals < threshold):
        return DoubleMarkResult(kept=beats, dropped=empty, modal_interval_s=modal)

    # Seed the template only from cycles marked once, so the shape being matched against is
    # not itself an average of QRS complexes and T-waves.
    crowded = np.zeros(beats.size, dtype=bool)
    short = intervals < threshold
    crowded[:-1] |= short
    crowded[1:] |= short
    seed = beats[~crowded]
    template = qrs_template(ecg_uv, seed if seed.size >= MINIMUM_SEED_BEATS else beats, sfreq, window)
    scores = _beat_scores(ecg_uv, beats, sfreq, window, template)

    alive = np.ones(beats.size, dtype=bool)
    while True:
        live = np.flatnonzero(alive)
        if live.size < 2:
            break
        spacing = np.diff(beats[live])
        offending = np.flatnonzero(spacing < threshold)
        if offending.size == 0:
            break
        pair = offending[int(np.argmin(spacing[offending]))]
        left, right = live[pair], live[pair + 1]
        alive[left if scores[left] < scores[right] else right] = False

    return DoubleMarkResult(
        kept=beats[alive], dropped=beats[~alive], modal_interval_s=modal
    )


def physiological_floor(
    analyzer_beats: np.ndarray,
    combined_beats: np.ndarray,
    *,
    percentile: float = 1.0,
    cap_fraction: float = 0.75,
) -> float:
    """Shortest RR interval this run may plausibly contain, in seconds.

    Taken from Analyzer's own intervals, which are the trustworthy set, at a low percentile
    rather than the raw minimum so one bad Analyzer interval cannot set the floor. Measured
    across 45 cohort runs, that percentile sits at a median of 0.82 of the beat-to-beat
    interval.

    **Intervals short enough to be a cycle marked twice are excluded before the percentile
    is read, and both the percentile and the cap are measured against the modal interval
    rather than the median.** A low percentile of Analyzer's raw intervals is set by
    Analyzer's own double marks wherever it makes them: on sub-0005 it returns 0.52 s
    against a 1.10 s beat period -- 0.48 of it, where a clean subject reads 0.75 -- so the
    filter is loosened by precisely the runs whose marks are least trustworthy. The median
    fails the mirror-image case, a sparsely marked run whose intervals are all inflated.
    The mode survives both.

    Returns 0.0 when there is too little evidence to estimate, which disables filtering
    rather than guessing.
    """
    analyzer_rr = np.diff(np.sort(np.asarray(analyzer_beats, dtype=float)))
    combined_rr = np.diff(np.sort(np.asarray(combined_beats, dtype=float)))
    if analyzer_rr.size < 10 or combined_rr.size < 10:
        return 0.0
    modal = modal_interval(analyzer_beats)
    trusted = analyzer_rr
    if np.isfinite(modal):
        candidate = analyzer_rr[analyzer_rr >= DOUBLE_MARK_FRACTION * modal]
        if candidate.size >= 10:
            trusted = candidate
    subject = float(np.percentile(trusted, percentile))
    combined_modal = modal_interval(combined_beats)
    reference = combined_modal if np.isfinite(combined_modal) else float(np.median(combined_rr))
    return min(subject, cap_fraction * reference)


def _enforce_floor(
    analyzer_beats: np.ndarray, recovered_beats: np.ndarray, floor_s: float
) -> np.ndarray:
    """Recovered beats at least `floor_s` from every accepted beat, earliest first.

    Analyzer's marks are accepted unconditionally and never dropped; only our own are
    filtered, so the trusted set is never degraded by this step.
    """
    if floor_s <= 0.0 or recovered_beats.size == 0:
        return recovered_beats

    accepted = np.sort(np.asarray(analyzer_beats, dtype=float))
    kept: list[float] = []
    for beat in np.sort(np.asarray(recovered_beats, dtype=float)):
        pool = np.concatenate([accepted, np.asarray(kept)]) if kept else accepted
        if pool.size == 0 or float(np.abs(pool - beat).min()) >= floor_s:
            kept.append(float(beat))
    return np.asarray(kept, dtype=float)


@dataclass(frozen=True)
class BeatQuality:
    analyzer_lock_ratio: float
    recovered_lock_ratio: float
    combined_lock_ratio: float
    physiological_floor_s: float
    rr_median_s: float
    rr_min_s: float
    rr_max_s: float
    implied_bpm: float
    refractory_violations: int
    refractory_rejected: int
    #: Analyzer marks removed as a second mark inside one cardiac cycle, before any gap
    #: was searched. Zero on a run whose marks are one per beat, which is most of them.
    double_marks_dropped: int
    recovered_beats: int
    gap_seconds_before: float
    gap_seconds_after: float
    status: str


@dataclass(frozen=True)
class BeatRecovery:
    analyzer_beats: np.ndarray
    recovered_beats: np.ndarray
    combined_beats: np.ndarray
    quality: BeatQuality


def qrs_template(
    ecg_uv: np.ndarray, beat_seconds: np.ndarray, sfreq: float, window: tuple[float, float]
) -> np.ndarray:
    """Average ECG waveform around a set of beats, mean-removed."""
    samples = np.round(np.asarray(beat_seconds) * sfreq).astype(int)
    stack = epoch_stack(ecg_uv[None, :], samples, sfreq, window)
    template = stack.mean(axis=1)[0]
    return template - template.mean()


def _normalised_correlation(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sliding Pearson correlation of template against signal, aligned to window starts."""
    length = template.size
    centred = template - template.mean()
    norm = np.linalg.norm(centred)
    if norm == 0:
        return np.zeros(max(signal.size - length + 1, 0))
    windows = np.lib.stride_tricks.sliding_window_view(signal, length)
    windows = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(windows, axis=1)
    # `over` is suppressed alongside the division flags because Accelerate's BLAS reports
    # divide/overflow/invalid on this matmul for ordinary float64 input on Apple silicon.
    # test_sliding_correlation_matches_an_explicit_pearson_reference holds it to the
    # explicit per-window computation, so a real numerical fault would still surface.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(norms > 0, (windows @ centred) / (norms * norm), 0.0)


def _pick_peaks(scores: np.ndarray, threshold: float, refractory_samples: int) -> np.ndarray:
    """Greedy highest-first selection honouring a refractory period."""
    candidates = np.flatnonzero(scores >= threshold)
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    order = candidates[np.argsort(scores[candidates])[::-1]]
    chosen: list[int] = []
    for index in order:
        if all(abs(index - taken) >= refractory_samples for taken in chosen):
            chosen.append(int(index))
    return np.array(sorted(chosen), dtype=int)


def recover_beats(
    ecg_uv: np.ndarray,
    analyzer_beats: np.ndarray,
    sfreq: float,
    *,
    settings: RecoverySettings = RecoverySettings(),
) -> BeatRecovery:
    """Recover beats inside Analyzer's gaps by matching its own QRS shape.

    Analyzer's marks are never revisited: the search runs only inside the gaps, and the
    template is re-estimated once from the combined set so a run with few seed beats is
    not permanently limited by a thin initial template.
    """
    analyzer = np.sort(np.asarray(analyzer_beats, dtype=float))
    duration = ecg_uv.size / sfreq
    window = settings.template_window
    before = gap_summary(
        analyzer, duration, minimum_seconds=settings.minimum_seconds, factor=settings.factor
    )

    # Cleared first, so the gap search and the floor both read a marker train that carries
    # one mark per cycle. Done afterwards it would be too late: the floor is computed from
    # these intervals, and the gaps are measured against their median.
    double_marks_dropped = 0
    if settings.clear_double_marks and analyzer.size >= MINIMUM_SEED_BEATS:
        cleared = drop_double_marks(ecg_uv, analyzer, sfreq, window=window)
        double_marks_dropped = int(cleared.dropped.size)
        analyzer = cleared.kept

    if analyzer.size < MINIMUM_SEED_BEATS:
        return BeatRecovery(
            analyzer_beats=analyzer,
            recovered_beats=np.empty(0),
            combined_beats=analyzer,
            quality=_quality(
                ecg_uv,
                analyzer,
                np.empty(0),
                analyzer,
                sfreq,
                window,
                duration,
                before,
                before,
                "insufficient_seed_beats",
            ),
        )

    # Capped for the same reason the gap baseline is: on an under-marked run the median
    # interval is not a beat period. Uncapped, sub-0008 run 4 sets a 2.0 s refractory --
    # so even inside a gap the matcher cannot accept beats 1 s apart -- and trims 2.0 s
    # off each end of every gap it searches. Capping it recovers 407 beats there against
    # 66, at a QRS lock ratio of 4.33 where Analyzer's own 44 beats score 3.96.
    median_rr = min(float(np.median(np.diff(analyzer))), MAXIMUM_BEAT_PERIOD_S)
    refractory = max(int(round(settings.refractory_fraction * median_rr * sfreq)), 1)
    offset = int(round(window[0] * sfreq))

    recovered = np.empty(0)
    seed = analyzer
    for _ in range(max(settings.iterations, 1)):
        template = qrs_template(ecg_uv, seed, sfreq, window)
        found: list[float] = []
        for gap in find_gaps(
            analyzer, minimum_seconds=settings.minimum_seconds, factor=settings.factor
        ):
            lo = int(round((gap.start_s + median_rr * 0.5) * sfreq))
            hi = int(round((gap.end_s - median_rr * 0.5) * sfreq))
            lo, hi = max(lo, 0), min(hi, ecg_uv.size)
            if hi - lo <= template.size:
                continue
            scores = _normalised_correlation(ecg_uv[lo:hi], template)
            picks = _pick_peaks(scores, settings.correlation_threshold, refractory)
            found.extend(((lo + picks - offset) / sfreq).tolist())
        recovered = np.sort(np.asarray(found, dtype=float))
        seed = np.sort(np.concatenate([analyzer, recovered])) if recovered.size else analyzer

    # The matcher's refractory period is half the median RR, which measured across the
    # cohort admits 9.6% false positives -- beats forming intervals shorter than the run's
    # own heart ever produced. Filter against that physiology before anything downstream
    # sees them: a false R marker makes Analyzer subtract a pulse template where no beat is.
    provisional = np.sort(np.concatenate([analyzer, recovered])) if recovered.size else analyzer
    floor = physiological_floor(
        analyzer,
        provisional,
        percentile=settings.refractory_percentile,
        cap_fraction=settings.refractory_cap_fraction,
    )
    proposed = int(recovered.size)
    recovered = _enforce_floor(analyzer, recovered, floor)
    rejected = proposed - int(recovered.size)

    combined = np.sort(np.concatenate([analyzer, recovered])) if recovered.size else analyzer
    after = gap_summary(
        combined, duration, minimum_seconds=settings.minimum_seconds, factor=settings.factor
    )
    return BeatRecovery(
        analyzer_beats=analyzer,
        recovered_beats=recovered,
        combined_beats=combined,
        quality=_quality(
            ecg_uv,
            analyzer,
            recovered,
            combined,
            sfreq,
            window,
            duration,
            before,
            after,
            "ok",
            rejected=rejected,
            floor_s=floor,
            double_marks_dropped=double_marks_dropped,
        ),
    )


def _quality(
    ecg_uv,
    analyzer,
    recovered,
    combined,
    sfreq,
    window,
    duration,
    before,
    after,
    status,
    *,
    rejected: int = 0,
    floor_s: float = 0.0,
    double_marks_dropped: int = 0,
) -> BeatQuality:
    intervals = np.diff(combined) if combined.size > 1 else np.array([np.nan])
    median_rr = float(np.median(intervals)) if combined.size > 1 else float("nan")
    refractory_floor = 0.5 * median_rr if combined.size > 1 else np.inf
    return BeatQuality(
        analyzer_lock_ratio=(
            lock_ratio(ecg_uv, analyzer, sfreq, window) if analyzer.size else float("nan")
        ),
        recovered_lock_ratio=(
            lock_ratio(ecg_uv, recovered, sfreq, window) if recovered.size else float("nan")
        ),
        combined_lock_ratio=(
            lock_ratio(ecg_uv, combined, sfreq, window) if combined.size else float("nan")
        ),
        physiological_floor_s=float(floor_s),
        rr_median_s=median_rr,
        rr_min_s=float(np.min(intervals)) if intervals.size else float("nan"),
        rr_max_s=float(np.max(intervals)) if intervals.size else float("nan"),
        implied_bpm=60.0 * combined.size / duration if duration else float("nan"),
        refractory_violations=(
            int(np.sum(intervals < refractory_floor)) if combined.size > 1 else 0
        ),
        refractory_rejected=int(rejected),
        double_marks_dropped=int(double_marks_dropped),
        recovered_beats=int(recovered.size),
        gap_seconds_before=float(before["gap_seconds"]),
        gap_seconds_after=float(after["gap_seconds"]),
        status=status,
    )


def crosscheck_agreement(
    recovered_beats: np.ndarray,
    ecg_uv: np.ndarray,
    sfreq: float,
    *,
    tolerance: float = 0.05,
) -> dict[str, float]:
    """Compare our beat set against NeuroKit2, as a measurement only.

    Never used to accept or reject a beat. NeuroKit2 reads the same magnetohydrodynamically
    distorted ECG and inflates counts on the affected subjects, so its disagreement is
    evidence about the run, not about our beats.
    """
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(ecg_uv, sampling_rate=int(sfreq), correct_artifacts=True)
        other = np.asarray(info["ECG_R_Peaks"], dtype=float) / sfreq
    except Exception as error:
        return {
            "status": f"unavailable: {type(error).__name__}",
            "agreement_fraction": float("nan"),
            "crosscheck_beats": 0.0,
            "crosscheck_lock_ratio": float("nan"),
        }

    beats = np.asarray(recovered_beats, dtype=float)
    if beats.size == 0 or other.size == 0:
        return {
            "status": "no_beats",
            "agreement_fraction": float("nan"),
            "crosscheck_beats": float(other.size),
            "crosscheck_lock_ratio": float("nan"),
        }

    nearest = np.abs(beats[:, None] - other[None, :]).min(axis=1)
    return {
        "status": "ok",
        "agreement_fraction": float(np.mean(nearest <= tolerance)),
        "crosscheck_beats": float(other.size),
        "crosscheck_lock_ratio": lock_ratio(ecg_uv, other, sfreq, (-0.2, 0.4)),
    }
