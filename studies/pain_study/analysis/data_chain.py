"""Say which generation of the source chain became the delivered data.

The chain is documented in ``source_data/EEG_SOURCE_LAYOUT.md``, but a document cannot
notice when the data stops matching it. On 2026-08-02 three ``step2_*`` directories existed
where the layout named one, and the one it named was not the one that fed step 3 — it agreed
with the delivered runs on 15 of 90 recordings against the real one's 89. Nothing detected
that, because nothing compared them.

The R-marker count per recording is the fingerprint that separates otherwise identical 13 GB
directories: every stage of the chain changes it, and it can be read from the ``.vmrk`` text
without opening a binary.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: ``ThermalPainEEGFMRI_run4_sub0008_<timestamp>_…`` and ``BaselineEEG_sub0008_…``.
RECORDING_NAME = re.compile(
    r"(?:ThermalPainEEGFMRI_run(?P<run>\d+)|BaselineEEG)_sub(?P<subject>\d{4})"
)

#: BrainVision writes ``Mk<n>=Pulse Artifact,R,<sample>,…``; Analyzer's own exports and our
#: rewritten marker files both use it, which is what makes the count comparable across stages.
PULSE_MARKER = re.compile(r"^Mk\d+=Pulse Artifact,R,", re.MULTILINE)

Key = tuple[str, str]


def recording_key(filename: str) -> Key | None:
    """``(subject, run)`` for a flat Analyzer batch filename, or None if it is not one."""
    match = RECORDING_NAME.search(filename)
    if match is None:
        return None
    return f"sub{match.group('subject')}", match.group("run") or "baseline"


def count_pulse_markers(vmrk_path: Path) -> int:
    """R markers in one ``.vmrk``, read as text so a 180 MB binary is never touched."""
    return len(PULSE_MARKER.findall(vmrk_path.read_text(encoding="latin-1", errors="replace")))


def stage_marker_counts(stage_root: Path | str) -> dict[Key, int]:
    """Fingerprint one stage: R markers per recording.

    macOS writes an AppleDouble ``._name`` beside every file on this drive. They parse as
    recordings and carry no markers, so an unfiltered scan reports every recording twice,
    once with a count of zero.
    """
    counts: dict[Key, int] = {}
    for entry in sorted(Path(stage_root).glob("*.vmrk")):
        if entry.name.startswith("._"):
            continue
        key = recording_key(entry.name)
        if key is not None:
            counts[key] = count_pulse_markers(entry)
    return counts


@dataclass(frozen=True)
class Agreement:
    """How far one stage's fingerprint matches another's, over what they share."""

    matched: int
    shared: int

    @property
    def fraction(self) -> float:
        return self.matched / self.shared if self.shared else 0.0


def agreement(counts: dict[Key, int], reference: dict[Key, int]) -> Agreement:
    shared = set(counts) & set(reference)
    return Agreement(
        matched=sum(1 for key in shared if counts[key] == reference[key]),
        shared=len(shared),
    )


def best_match(candidates: dict[str, dict[Key, int]], reference: dict[Key, int]) -> str | None:
    """Which candidate stage the reference was most likely built from."""
    scored = [(name, agreement(counts, reference)) for name, counts in candidates.items()]
    scored = [(name, score) for name, score in scored if score.shared]
    if not scored:
        return None
    return max(scored, key=lambda item: (item[1].fraction, item[1].shared))[0]


def verify_chain(
    stages: dict[str, dict[Key, int]],
    delivered: dict[Key, int],
    *,
    expected_source: str,
) -> list[str]:
    """Problems with the stage the delivered data is supposed to have come from.

    Reported rather than raised, so a caller sees every disagreement at once instead of the
    first. An empty list means the delivered data's marker counts are reproduced exactly by
    ``expected_source``, which is the only claim this makes — it does not check sample data.
    """
    problems: list[str] = []
    if expected_source not in stages:
        return [f"{expected_source}: not found on disk"]

    counts = stages[expected_source]
    missing = sorted(set(delivered) - set(counts))
    if missing:
        shown = ", ".join(f"{s} run-{r}" for s, r in missing[:5])
        problems.append(
            f"{expected_source}: missing {len(missing)} recording(s) the delivered data has"
            f" ({shown}{', …' if len(missing) > 5 else ''})"
        )

    differing = sorted(key for key in set(delivered) & set(counts) if counts[key] != delivered[key])
    if differing:
        shown = ", ".join(
            f"{s} run-{r}: {counts[(s, r)]} vs {delivered[(s, r)]} delivered"
            for s, r in differing[:5]
        )
        problems.append(
            f"{expected_source}: {len(differing)} of {len(set(delivered) & set(counts))}"
            f" recording(s) disagree with the delivered data ({shown}"
            f"{', …' if len(differing) > 5 else ''})"
        )
    return problems


def delivered_marker_counts(
    bids_root: Path | str, *, description: str = "Stimulus/S  1"
) -> dict[Key, int]:
    """Fingerprint the delivered BIDS runs the same way, from their own annotations.

    BIDS conversion renames the markers, so the pulse marker arrives as a raw stimulus code
    here rather than under the name Analyzer gave it.
    """
    import mne

    counts: dict[Key, int] = {}
    pattern = os.path.join(str(bids_root), "sub-*", "eeg", "sub-*_task-*_run-*_eeg.vhdr")
    import glob

    import numpy as np

    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if name.startswith("._"):
            continue
        match = re.search(r"sub-(\d{4})_task-[^_]+_run-(\d+)", name)
        if match is None:
            continue
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
        descriptions = np.asarray(raw.annotations.description, dtype=str)
        counts[(f"sub{match.group(1)}", match.group(2))] = int((descriptions == description).sum())
    return counts


__all__ = [
    "Agreement",
    "Key",
    "agreement",
    "best_match",
    "count_pulse_markers",
    "delivered_marker_counts",
    "recording_key",
    "stage_marker_counts",
    "verify_chain",
]
