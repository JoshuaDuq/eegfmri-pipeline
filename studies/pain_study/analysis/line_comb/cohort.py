"""Assemble cohort spectra for the scanner-harmonic diagnosis.

Two sources feed the diagnosis and they answer different questions.

*Final cleaned epochs* are what the analyses actually see, so they define which lines
matter. *Gradient-free head segments* are the stretch of every recording between the
amplifier starting and the scanner's first gradient pulse; a line present there cannot
have come from the imaging gradients. The scanner plays roughly ten dummy volumes before
it emits the first ``Volume`` marker, so the marker is useless for finding that stretch
and :func:`detect_quiet_interval` locates it from the signal instead.

Segment lengths are always whole multiples of the TR, which puts every volume-comb line
on a bin centre. The matched grid is a sub-grid of the high-resolution one, so the two
can be compared bin for bin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from studies.pain_study.analysis.line_comb.diagnosis import TR_SECONDS, hann_periodogram

HIGH_RESOLUTION_TR_COUNT = 24
"""Segments of 24 TR (21.6 s) resolve 0.0463 Hz, enough to separate the observed combs."""

MATCHED_TR_COUNT = 4
"""Segments of 4 TR (3.6 s) fit inside most gradient-free head windows.

The head windows run from 2.5 to 45 s with a median near 7 s, so the block length trades
frequency resolution against how many recordings contribute at all. At 3.6 s the grid
still separates the observed 2.4 Hz comb by nine bins, and it is a sub-grid of the
21.6 s catalogue grid, so the two line up bin for bin."""

QUIET_BLOCK_SECONDS = 0.25
QUIET_ONSET_FACTOR = 10.0
QUIET_LEAD_IN_SECONDS = 0.5
QUIET_GUARD_SECONDS = 0.25

_VOLUME_MARKER = re.compile(r"Mk\d+=Volume,V\s+\d+,(\d+),")


@dataclass(frozen=True)
class SubjectSpectra:
    """Per-run spectra for one participant's final cleaned epochs."""

    subject: str
    freqs_high: np.ndarray
    freqs_matched: np.ndarray
    psd_high: np.ndarray  # (n_runs, n_channels, n_freqs_high)
    psd_matched: np.ndarray  # (n_runs, n_channels, n_freqs_matched)
    channel_names: tuple[str, ...]
    bads: tuple[str, ...]
    run_labels: tuple[int, ...]
    epochs_per_run: tuple[int, ...]
    coefficients: np.ndarray  # (n_epochs, n_channels, n_freqs_high) complex
    volume_offsets_s: np.ndarray  # (n_epochs,)
    epoch_runs: np.ndarray  # (n_epochs,)


@dataclass(frozen=True)
class QuietSegment:
    """One gradient-free stretch located at the head of a recording."""

    subject: str
    recording: str
    kind: str
    start_sample: int
    stop_sample: int
    sfreq: float

    @property
    def duration_s(self) -> float:
        return (self.stop_sample - self.start_sample) / self.sfreq


def assign_runs(event_samples: Sequence[int], run_lengths: Sequence[int]) -> np.ndarray:
    """Label each epoch with the run it came from.

    The pipeline concatenates the runs before epoching, so an epoch's event sample is an
    offset into that concatenation and the cumulative run lengths partition it. The
    partition is verified rather than assumed: every run must contribute at least one
    epoch and the labels must not go backwards.
    """
    samples = np.asarray(event_samples, dtype=np.int64)
    lengths = np.asarray(run_lengths, dtype=np.int64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("event_samples must be a non-empty 1D array.")
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("run_lengths must be a non-empty 1D array.")
    if np.any(lengths <= 0):
        raise ValueError("run_lengths must all be positive.")
    if np.any(np.diff(samples) < 0):
        raise ValueError("event_samples must be non-decreasing.")

    boundaries = np.cumsum(lengths)
    if samples[-1] >= boundaries[-1]:
        raise ValueError(
            f"Last epoch at sample {samples[-1]} falls beyond the concatenated "
            f"length {boundaries[-1]}; run lengths do not describe these epochs."
        )
    labels = np.searchsorted(boundaries, samples, side="right") + 1
    missing = sorted(set(range(1, lengths.size + 1)) - set(labels.tolist()))
    if missing:
        raise ValueError(f"Runs {missing} contributed no epochs; refusing to guess.")
    return labels.astype(int)


def segment_periodograms(
    data: np.ndarray,
    sfreq: float,
    *,
    tr_count: int,
    tr: float = TR_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each trailing-axis segment into TR-commensurate blocks and average their PSD.

    ``data`` has shape ``(n_segments, n_channels, n_times)``. Blocks are contiguous and
    non-overlapping, taken from the start of the segment; a trailing remainder shorter
    than one block is dropped rather than zero-padded, because padding would smear the
    narrow lines this analysis is built to measure.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 3:
        raise ValueError("data must have shape (n_segments, n_channels, n_times).")
    block = int(round(tr * sfreq)) * int(tr_count)
    if block <= 0:
        raise ValueError("tr_count must be positive.")
    n_blocks = array.shape[-1] // block
    if n_blocks < 1:
        raise ValueError(
            f"Segments of {array.shape[-1]} samples are shorter than one "
            f"{tr_count}-TR block ({block} samples)."
        )
    trimmed = array[..., : n_blocks * block]
    blocks = trimmed.reshape(array.shape[0], array.shape[1], n_blocks, block)
    freqs, psd = hann_periodogram(blocks, sfreq)
    return freqs, psd.mean(axis=2)


def tile_intervals(
    start: int,
    stop: int,
    *,
    block: int,
    overlap: float = 0.5,
) -> list[tuple[int, int]]:
    """Cover an interval with fixed-length blocks, overlapping to use short windows well."""
    if block <= 0:
        raise ValueError("block must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be inside [0, 1).")
    step = max(int(round(block * (1.0 - overlap))), 1)
    tiles = []
    position = int(start)
    while position + block <= int(stop):
        tiles.append((position, position + block))
        position += step
    return tiles


def block_standard_deviation(
    data: np.ndarray,
    sfreq: float,
    *,
    block_seconds: float = QUIET_BLOCK_SECONDS,
) -> np.ndarray:
    """Median across channels of the per-block standard deviation.

    Each block is mean-removed first. These recordings are DC-coupled and carry channel
    offsets of hundreds of microvolts, so a raw root-mean-square would measure the offset
    and miss the gradient entirely.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_times).")
    block = int(round(block_seconds * sfreq))
    if block <= 0:
        raise ValueError("block_seconds must span at least one sample.")
    n_blocks = array.shape[1] // block
    if n_blocks < 1:
        raise ValueError("data is shorter than one block.")
    blocks = array[:, : n_blocks * block].reshape(array.shape[0], n_blocks, block)
    centred = blocks - blocks.mean(axis=2, keepdims=True)
    return np.median(centred.std(axis=2), axis=0)


def detect_quiet_interval(
    block_sd: Sequence[float],
    *,
    sfreq: float,
    block_seconds: float = QUIET_BLOCK_SECONDS,
    onset_factor: float = QUIET_ONSET_FACTOR,
    lead_in_seconds: float = QUIET_LEAD_IN_SECONDS,
    guard_seconds: float = QUIET_GUARD_SECONDS,
) -> tuple[int, int]:
    """Return the gradient-free sample range at the head of a recording.

    Gradient onset raises the per-block standard deviation by more than an order of
    magnitude, from tens of microvolts to hundreds. The threshold is placed at the
    geometric mean of the profile's smallest and largest block, which sits in the empty
    middle of that gap. Taking the levels from the extremes rather than from the opening
    blocks keeps the estimate honest when the quiet head is only a second or two long and
    would otherwise be outranked by the gradient it precedes. One anomalously quiet block
    moves a geometric threshold only by its square root, which stays inside a gap this
    wide; and it would shorten the reported window rather than admit gradient into it.

    A lead-in is dropped at the start for amplifier settling and a guard is left before
    onset. When the profile shows no gradient at all, the whole probe is returned.
    """
    values = np.asarray(block_sd, dtype=float)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("block_sd must be a 1D array with at least eight blocks.")
    if onset_factor <= 1.0:
        raise ValueError("onset_factor must exceed 1.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("block_sd must be finite and positive.")

    low = float(np.min(values))
    high = float(np.max(values))
    if high < onset_factor * low:
        onset_block = values.size
    else:
        threshold = float(np.sqrt(low * high))
        loud = np.flatnonzero(values > threshold)
        onset_block = int(loud[0]) if loud.size else values.size

    samples_per_block = int(round(block_seconds * sfreq))
    start = int(round(lead_in_seconds * sfreq))
    stop = onset_block * samples_per_block - int(round(guard_seconds * sfreq))
    return start, max(stop, start)


def read_volume_markers(vmrk_path: str | Path) -> list[int]:
    """Sample positions of the ``Volume`` markers in a BrainVision marker file."""
    path = Path(vmrk_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return [int(match.group(1)) for match in _VOLUME_MARKER.finditer(text)]


def align_channels(
    names: Sequence[str],
    reference: Sequence[str],
) -> np.ndarray:
    """Index `names` so it follows `reference`. Raises when the reference is not covered.

    Matching ignores case. BrainVision headers and the BIDS conversion disagree on the
    capitalisation of a few electrodes -- ``FPz`` against ``Fpz`` -- and treating those as
    different channels would drop a good electrode from every comparison.
    """
    lookup: dict[str, int] = {}
    for index, name in enumerate(names):
        lookup.setdefault(name.lower(), index)
    missing = [name for name in reference if name.lower() not in lookup]
    if missing:
        raise ValueError(f"Channels absent from the recording: {', '.join(missing)}")
    return np.array([lookup[name.lower()] for name in reference], dtype=int)


def good_channel_mask(
    channel_names: Sequence[str],
    bads: Sequence[str],
) -> np.ndarray:
    """Boolean mask over channels excluding the named bad ones."""
    bad_set = set(bads)
    mask = np.array([name not in bad_set for name in channel_names], dtype=bool)
    if not mask.any():
        raise ValueError("Every channel is marked bad.")
    return mask
