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

from eeg_pipeline.preprocessing.bcg.metrics import epoch_stack, lock_ratio

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


def find_gaps(
    beat_seconds: np.ndarray,
    *,
    minimum_seconds: float = 2.0,
    factor: float = 2.0,
) -> list[Gap]:
    """Intervals that are both absolutely long and long for this run.

    Both tests are required. The absolute floor alone flags ordinary bradycardia; the
    relative test alone flags a run whose median RR is already inflated because detection
    failed nearly everywhere.
    """
    beats = np.sort(np.asarray(beat_seconds, dtype=float))
    if beats.size < 3:
        return []
    intervals = np.diff(beats)
    threshold = max(minimum_seconds, factor * float(np.median(intervals)))
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
    minimum_seconds: float = 2.0,
    factor: float = 2.0,
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
    minimum_seconds: float = 2.0
    factor: float = 2.0


@dataclass(frozen=True)
class BeatQuality:
    analyzer_lock_ratio: float
    recovered_lock_ratio: float
    combined_lock_ratio: float
    rr_median_s: float
    rr_min_s: float
    rr_max_s: float
    implied_bpm: float
    refractory_violations: int
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

    median_rr = float(np.median(np.diff(analyzer)))
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

    combined = np.sort(np.concatenate([analyzer, recovered])) if recovered.size else analyzer
    after = gap_summary(
        combined, duration, minimum_seconds=settings.minimum_seconds, factor=settings.factor
    )
    return BeatRecovery(
        analyzer_beats=analyzer,
        recovered_beats=recovered,
        combined_beats=combined,
        quality=_quality(
            ecg_uv, analyzer, recovered, combined, sfreq, window, duration, before, after, "ok"
        ),
    )


def _quality(
    ecg_uv, analyzer, recovered, combined, sfreq, window, duration, before, after, status
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
        rr_median_s=median_rr,
        rr_min_s=float(np.min(intervals)) if intervals.size else float("nan"),
        rr_max_s=float(np.max(intervals)) if intervals.size else float("nan"),
        implied_bpm=60.0 * combined.size / duration if duration else float("nan"),
        refractory_violations=(
            int(np.sum(intervals < refractory_floor)) if combined.size > 1 else 0
        ),
        recovered_beats=int(recovered.size),
        gap_seconds_before=float(before["gap_seconds"]),
        gap_seconds_after=float(after["gap_seconds"]),
        status=status,
    )
