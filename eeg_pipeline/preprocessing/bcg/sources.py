"""Pair the two Analyzer exports and prove they describe the same samples.

The pulse-markers-only export carries the uncorrected ballistocardiogram and Analyzer's
R marks; the corrected export is what currently feeds BIDS. Substituting gap stretches
from one into the other is only valid while they stay sample-aligned, so that identity is
measured per run rather than assumed.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHANNEL_PATTERN = re.compile(r"^Ch(\d+)=([^,]*),([^,]*),([^,]*)(?:,(.*))?$", re.MULTILINE)
SIDECAR_SUFFIXES = (".vhdr", ".vmrk")

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


def channel_scaling(vhdr_path: Path | str) -> tuple[list[str], np.ndarray]:
    """Channel names and their binary resolution, in the header's own unit.

    Analyzer writes these exports with an empty resolution field, which BrainVision reads
    as 1.0 -- the samples are already microvolts. Parsing it rather than assuming keeps the
    writer correct if a future export carries an explicit scale.
    """
    path = Path(vhdr_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    binary_format = re.search(r"BinaryFormat=(\S+)", text)
    orientation = re.search(r"DataOrientation=(\S+)", text)
    if binary_format is None or binary_format.group(1) != "IEEE_FLOAT_32":
        raise ValueError(f"{path.name}: expected IEEE_FLOAT_32 binary data.")
    if orientation is None or orientation.group(1) != "VECTORIZED":
        raise ValueError(
            f"{path.name}: expected VECTORIZED data orientation, "
            f"got {orientation.group(1) if orientation else 'none'}."
        )

    names, resolutions = [], []
    for match in CHANNEL_PATTERN.finditer(text):
        names.append(match.group(2))
        scale = match.group(4).strip()
        resolutions.append(float(scale) if scale else 1.0)
    if not names:
        raise ValueError(f"{path.name}: no channel definitions found.")
    return names, np.asarray(resolutions, dtype=float)


def write_corrected_recording(
    source_vhdr: Path | str, destination_dir: Path, data_uv: np.ndarray
) -> Path:
    """Write `data_uv` as a new recording, reusing the source header and marker file.

    Only the ``.eeg`` binary is rewritten; ``.vhdr`` and ``.vmrk`` are copied byte for
    byte, so channel metadata and every marker survive unchanged. Rebuilding the file
    through `mne.export.export_raw` instead would drop the stimulus markers the study
    depends on, and would rewrite the header in a different layout.
    """
    source = Path(source_vhdr)
    names, resolutions = channel_scaling(source)

    array = np.asarray(data_uv, dtype=float)
    if array.shape[0] != len(names):
        raise ValueError(
            f"{source.name}: header describes {len(names)} channels, got {array.shape[0]}."
        )

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for suffix in SIDECAR_SUFFIXES:
        companion = source.with_suffix(suffix)
        if companion.exists():
            shutil.copy2(companion, destination_dir / companion.name)

    # VECTORIZED is channel-major, so the array is written without transposing.
    scaled = array / resolutions[:, None]
    scaled.astype("<f4").tofile(destination_dir / source.with_suffix(".eeg").name)
    return destination_dir / source.name
