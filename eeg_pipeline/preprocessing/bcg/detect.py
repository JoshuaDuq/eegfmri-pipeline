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
