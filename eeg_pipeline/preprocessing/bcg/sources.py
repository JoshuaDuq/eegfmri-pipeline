"""Pair the two Analyzer exports and prove they describe the same samples.

The pulse-markers-only export carries the uncorrected ballistocardiogram and Analyzer's
R marks; the corrected export is what currently feeds BIDS. Substituting gap stretches
from one into the other is only valid while they stay sample-aligned, so that identity is
measured per run rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUN_PATTERN = re.compile(r"_run(?P<run>\d+)_(?P<subject>sub\d+)_")
BASELINE_PATTERN = re.compile(r"^BaselineEEG_(?P<subject>sub\d+)_")

ECG_IDENTITY_TOLERANCE_UV = 1e-3


@dataclass(frozen=True)
class RunPair:
    subject: str
    run: str
    uncorrected_vhdr: Path
    corrected_vhdr: Path


@dataclass(frozen=True)
class PairValidation:
    subject: str
    run: str
    n_times_uncorrected: int
    n_times_corrected: int
    aligned: bool
    ecg_max_abs_diff_uv: float
    status: str


def _key(path: Path) -> tuple[str, str] | None:
    name = path.name
    match = RUN_PATTERN.search(name)
    if match:
        return match.group("subject"), match.group("run")
    baseline = BASELINE_PATTERN.match(name)
    if baseline:
        return baseline.group("subject"), "baseline"
    return None


def _index(root: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(root.glob("*.vhdr")):
        if path.name.startswith("._"):
            continue
        key = _key(path)
        if key is not None:
            found.setdefault(key, path)
    return found


def discover_run_pairs(uncorrected_root: Path, corrected_root: Path) -> list[RunPair]:
    """Every recording present in both exports, keyed by subject and run."""
    uncorrected = _index(Path(uncorrected_root))
    corrected = _index(Path(corrected_root))
    return [
        RunPair(subject, run, uncorrected[(subject, run)], corrected[(subject, run)])
        for subject, run in sorted(uncorrected.keys() & corrected.keys())
    ]


def validate_pair(pair: RunPair, ecg_channel: str = "ECG") -> PairValidation:
    """Measure sample alignment and ECG identity for one paired recording.

    Analyzer's pulse correction modifies EEG only, so a non-zero ECG difference means the
    two files are not the same recording and the pair must not be used.
    """
    import mne

    mne.set_log_level("ERROR")
    left = mne.io.read_raw_brainvision(pair.uncorrected_vhdr, preload=True, verbose="ERROR")
    right = mne.io.read_raw_brainvision(pair.corrected_vhdr, preload=True, verbose="ERROR")

    aligned = bool(left.n_times == right.n_times)
    difference = float("nan")
    status = "ok"
    if not aligned:
        status = "length_mismatch"
    elif ecg_channel not in left.ch_names or ecg_channel not in right.ch_names:
        status = "missing_ecg"
    else:
        a = left.copy().pick([ecg_channel]).get_data()[0] * 1e6
        b = right.copy().pick([ecg_channel]).get_data()[0] * 1e6
        difference = float(np.abs(a - b).max())
        if difference > ECG_IDENTITY_TOLERANCE_UV:
            status = "ecg_mismatch"

    return PairValidation(
        subject=pair.subject,
        run=pair.run,
        n_times_uncorrected=int(left.n_times),
        n_times_corrected=int(right.n_times),
        aligned=aligned,
        ecg_max_abs_diff_uv=difference,
        status=status,
    )
