#!/usr/bin/env python3
"""One-off repair for runs where the scanner stopped and restarted mid-recording.

Use this when a single BIDS EEG run holds more than one scanner acquisition. The
EEG keeps recording across the break, so ``trim_to_volume_bounds`` -- which crops
between the first and last volume marker -- leaves the scanner-off gap and the
second acquisition inside the run. Nothing downstream can tell the two apart:
both blocks spell their volumes ``Volume/V  1``.

The repair keeps the *first* contiguous volume block and drops everything after
it. That is the right block when the fMRI run corresponds to the first
acquisition, which is the only case this script is written for -- check the BOLD
volume count against the block before running it.

The ``.eeg`` binary is truncated byte-wise rather than re-encoded, so the samples
that survive are bit-identical to the ones that went in.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

#: Bytes per sample for the BrainVision binary formats this project writes.
_FORMAT_WIDTHS = {"IEEE_FLOAT_32": 4, "IEEE_FLOAT_64": 8, "INT_16": 2}

#: Suffix for this repair's backups. Deliberately not a plain ``.bak``: the files it
#: touches may already carry one from a different repair.
BACKUP_SUFFIX = ".precrop.bak"

#: An interval this many times the median volume interval starts a new block. The
#: scanner-off breaks this script exists for are tens of seconds against a ~1 s
#: repetition time, so the threshold never has to be delicate.
DEFAULT_GAP_FACTOR = 1.5


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def read_header(vhdr_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in vhdr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(";") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields.setdefault(key.strip(), value.strip())
    return fields


def volume_blocks(
    onsets: list[float],
    *,
    gap_factor: float = DEFAULT_GAP_FACTOR,
) -> list[list[float]]:
    """Split volume marker onsets into contiguous acquisition blocks."""
    if not onsets:
        return []
    if len(onsets) == 1:
        return [list(onsets)]

    intervals = pd.Series(onsets).diff().dropna().to_numpy()
    median_interval = float(pd.Series(intervals).median())
    blocks: list[list[float]] = [[onsets[0]]]
    for previous, current in zip(onsets, onsets[1:]):
        if current - previous > gap_factor * median_interval:
            blocks.append([current])
        else:
            blocks[-1].append(current)
    return blocks


def median_interval(onsets: list[float]) -> float:
    return float(pd.Series(onsets).diff().dropna().median())


def _truncate_binary(eeg_path: Path, *, n_samples: int, n_channels: int, width: int) -> None:
    keep_bytes = n_samples * n_channels * width
    with eeg_path.open("r+b") as handle:
        handle.truncate(keep_bytes)


def _truncate_markers(vmrk_path: Path, *, n_samples: int) -> int:
    """Drop markers past the crop and renumber the survivors. Returns rows kept."""
    lines = vmrk_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    kept = 0
    for line in lines:
        match = re.match(r"^Mk(\d+)=(.*)$", line)
        if match is None:
            out.append(line)
            continue
        fields = match.group(2).split(",")
        if len(fields) < 3:
            out.append(line)
            continue
        try:
            position = int(fields[2])
        except ValueError:
            out.append(line)
            continue
        # BrainVision marker positions are 1-based sample indices.
        if position > n_samples:
            continue
        kept += 1
        out.append(f"Mk{kept}=" + ",".join(fields))
    vmrk_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return kept


def crop_run(
    eeg_dir: Path,
    subject: str,
    task: str,
    run: int,
    *,
    gap_factor: float,
    write_backup: bool,
) -> str:
    stem = f"{subject}_task-{task}_run-{run}"
    vhdr_path = eeg_dir / f"{stem}_eeg.vhdr"
    vmrk_path = eeg_dir / f"{stem}_eeg.vmrk"
    eeg_path = eeg_dir / f"{stem}_eeg.eeg"
    events_path = eeg_dir / f"{stem}_events.tsv"
    json_path = eeg_dir / f"{stem}_eeg.json"

    for required in (vhdr_path, vmrk_path, eeg_path, events_path):
        if not required.exists():
            return f"skip {stem}: missing {required.name}"

    header = read_header(vhdr_path)
    if header.get("DataFormat") != "BINARY" or header.get("DataOrientation") != "MULTIPLEXED":
        return f"skip {stem}: only multiplexed binary BrainVision data can be truncated"
    binary_format = header.get("BinaryFormat", "")
    if binary_format not in _FORMAT_WIDTHS:
        return f"skip {stem}: unsupported BinaryFormat {binary_format!r}"
    if "DataPoints" in header:
        return f"skip {stem}: header declares DataPoints, which truncation would invalidate"

    width = _FORMAT_WIDTHS[binary_format]
    n_channels = int(header["NumberOfChannels"])
    sfreq = 1e6 / float(header["SamplingInterval"])
    n_samples_before = eeg_path.stat().st_size // (n_channels * width)

    events = pd.read_csv(events_path, sep="\t")
    if "trial_type" not in events.columns or "onset" not in events.columns:
        return f"skip {stem}: events file needs onset and trial_type"

    is_volume = events["trial_type"].map(_normalize).str.startswith("Volume")
    volume_onsets = pd.to_numeric(events.loc[is_volume, "onset"], errors="coerce").dropna()
    if volume_onsets.empty:
        return f"skip {stem}: no volume markers"

    blocks = volume_blocks(sorted(float(v) for v in volume_onsets), gap_factor=gap_factor)
    if len(blocks) == 1:
        return f"ok   {stem}: one acquisition block ({len(blocks[0])} volumes), nothing to crop"

    first_block = blocks[0]
    repetition_time = median_interval(first_block)
    crop_end_s = first_block[-1] + repetition_time
    n_keep = min(int(round(crop_end_s * sfreq)), n_samples_before)
    if n_keep >= n_samples_before:
        return f"skip {stem}: first block already reaches the end of the recording"

    rewritten = [path for path in (eeg_path, vmrk_path, events_path, json_path) if path.exists()]
    if write_backup:
        # A distinct suffix, because these files already carry a plain ``.bak`` from
        # the restart-trigger repair and that record has to survive this one.
        backups = [path.with_suffix(path.suffix + BACKUP_SUFFIX) for path in rewritten]
        existing = [backup.name for backup in backups if backup.exists()]
        if existing:
            return (
                f"skip {stem}: backup already exists, refusing to overwrite: {', '.join(existing)}"
            )
        for path, backup in zip(rewritten, backups):
            shutil.copy2(path, backup)

    _truncate_binary(eeg_path, n_samples=n_keep, n_channels=n_channels, width=width)
    markers_kept = _truncate_markers(vmrk_path, n_samples=n_keep)

    sample_index = (pd.to_numeric(events["onset"], errors="coerce") * sfreq).round()
    events_kept = events.loc[sample_index < n_keep].copy()
    events_kept.to_csv(events_path, sep="\t", index=False)

    if json_path.exists():
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        sidecar["RecordingDuration"] = round((n_keep - 1) / sfreq, 6)
        json_path.write_text(json.dumps(sidecar, indent=4) + "\n", encoding="utf-8")

    dropped_blocks = len(blocks) - 1
    return (
        f"crop {stem}: kept block 1 of {len(blocks)} "
        f"({len(first_block)} volumes, {n_keep} of {n_samples_before} samples); "
        f"dropped {dropped_blocks} later block(s), "
        f"{len(events) - len(events_kept)} event row(s), "
        f"{markers_kept} marker(s) retained"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a BIDS EEG run to its first contiguous scanner volume block"
    )
    parser.add_argument("--bids-root", required=True, help="Path to the BIDS EEG root")
    parser.add_argument("--subject", required=True, help="Subject with or without 'sub-' prefix")
    parser.add_argument("--task", required=True, help="Task label")
    parser.add_argument("--run", required=True, type=int, help="Run number")
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=DEFAULT_GAP_FACTOR,
        help="Multiple of the median volume interval that starts a new block",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak copies")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    subject = args.subject if str(args.subject).startswith("sub-") else f"sub-{args.subject}"
    eeg_dir = Path(args.bids_root) / subject / "eeg"
    if not eeg_dir.exists():
        print(f"error: missing EEG directory: {eeg_dir}")
        return 2

    message = crop_run(
        eeg_dir,
        subject,
        args.task,
        int(args.run),
        gap_factor=float(args.gap_factor),
        write_backup=not bool(args.no_backup),
    )
    print(message)
    return 0 if not message.startswith("skip") else 1


if __name__ == "__main__":
    sys.exit(main())
