"""The response shape at a cluster peak, averaged over the events that drove it.

A cluster table says a voxel reached z = 6.3. It does not say whether the signal there
rose and fell like a haemodynamic response or whether a handful of frames moved
together and the model had no way to tell the difference. Those two produce the same
z, the same cluster, and the same peak coordinate -- and a single-subject map at an
uncorrected height contains both.

Averaged over events rather than drawn as a timecourse. The raw trace was tried first
and failed on its own terms: 3,413 frames across six runs render as a solid band of
ink in which no correspondence with the model is visible, so the panel conveyed less
than the correlation printed beside it. Epoched on each condition's own onsets, the
same data becomes the shape a reader is actually looking for, with the two conditions
of the contrast side by side.

Nothing is fitted. The design matrix is read from the file the analysis run wrote, its
weighted columns supply the onsets, and its nuisance columns are projected out of the
observed trace so the curve describes the variance the contrast was estimated from --
a linear transform of data already on disk, using weights already recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.figures.design import ONSET_FRACTION, onset_rows
from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)

logger = logging.getLogger(__name__)

#: Column of a written design matrix holding each row's acquisition time, in seconds.
#:
#: This is what aligns a design row to a BOLD frame. The two counts differ -- runs here
#: carry 570 frames against 568 or 569 design rows -- because the model drops
#: non-steady-state volumes and censored frames, and which ones it dropped is not
#: recoverable from the counts alone. Taking the last N frames, or assuming the
#: difference sits at the start, silently shifts every sample by a frame or two.
FRAME_TIME_COLUMN = "frame"

#: Epoch window around each onset, in seconds.
#:
#: Wide enough to carry the pre-onset baseline, the peak, and the undershoot, which is
#: the whole shape a reader checks a response against.
EPOCH_WINDOW_S: Tuple[float, float] = (-4.0, 18.0)

#: Peaks drawn. Past a few the rows crowd, and the peaks beyond the third are rarely
#: what a result rests on.
DEFAULT_MAX_PEAKS = 3


@dataclass(frozen=True)
class ConditionResponse:
    """One condition's mean response at one peak, with its spread across events."""

    name: str
    weight: float
    times: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    n_events: int


@dataclass(frozen=True)
class PeakResponse:
    """Every weighted condition's response at one cluster peak."""

    label: str
    coordinate: Tuple[float, float, float]
    conditions: Tuple[ConditionResponse, ...]
    n_runs: int


def _zscore(values: np.ndarray) -> np.ndarray:
    """Centre and scale, leaving a constant trace at zero rather than at infinity."""
    values = np.asarray(values, dtype=float)
    spread = float(np.std(values))
    if not np.isfinite(spread) or spread == 0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / spread


def _project_out(signal: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Remove the nuisance subspace from ``signal`` by least squares.

    The observed trace otherwise carries the drift, motion, and physiological variance
    the GLM removed before it estimated anything -- and on a 512-second run the cosine
    drift alone dominates, so the epoch average would describe the high-pass filter.
    """
    if nuisance.size == 0 or nuisance.shape[0] != signal.shape[0]:
        return signal
    # errstate: numpy on Accelerate BLAS raises spurious invalid/overflow flags from
    # matmul even for well-conditioned finite operands, as documented in volumes.py.
    # The finiteness check below is the real guard.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        try:
            coefficients, *_ = np.linalg.lstsq(nuisance, signal, rcond=None)
        except np.linalg.LinAlgError as exc:
            logger.info("Could not project out the nuisance columns (%s)", exc)
            return signal
        residual = signal - nuisance @ coefficients
    return residual if np.all(np.isfinite(residual)) else signal


def _frame_indices(design: Any, t_r: float) -> Optional[np.ndarray]:
    """Map each design row onto the BOLD frame it was built from."""
    if FRAME_TIME_COLUMN not in getattr(design, "columns", ()):
        return None
    if not t_r or t_r <= 0:
        return None
    times = design[FRAME_TIME_COLUMN].to_numpy(dtype=float)
    indices = np.rint(times / float(t_r)).astype(int)
    if np.any(indices < 0) or not np.all(np.diff(indices) > 0):
        return None
    return indices


def _epochs(
    signal: np.ndarray, onsets: np.ndarray, *, lo: int, hi: int
) -> List[np.ndarray]:
    """Cut ``signal`` around each onset, dropping epochs that run off either end."""
    out: List[np.ndarray] = []
    for onset in onsets:
        start, stop = int(onset) + lo, int(onset) + hi
        if start < 0 or stop > signal.size:
            continue
        epoch = signal[start:stop]
        # Baseline-corrected on its own pre-onset frames, so epochs that begin at
        # different points of the residual drift start from a common zero and the
        # spread across events describes the response rather than where it started.
        baseline = epoch[: max(-lo, 1)]
        out.append(epoch - float(np.mean(baseline)))
    return out


def collect_peak_responses(
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
    *,
    bold_paths: Sequence[Any],
    design_paths: Sequence[Any],
    contrast_columns: Sequence[str],
    contrast_vector: Sequence[float],
    t_r: Optional[float],
    max_peaks: int = DEFAULT_MAX_PEAKS,
    window_s: Tuple[float, float] = EPOCH_WINDOW_S,
) -> List[PeakResponse]:
    """Average each peak's signal around the onsets of each weighted condition.

    Returns an empty list rather than raising whenever the pieces do not line up --
    no design matrices recorded, no TR, a design without its frame-time column, a peak
    outside the field of view. Each is a property of the derivatives, not a fault.
    """
    import nibabel as nib
    import pandas as pd

    wanted = list(peaks)[: max(int(max_peaks), 1)]
    if not wanted or not bold_paths or not design_paths or not t_r:
        return []

    weights = {
        str(name): float(weight)
        for name, weight in zip(contrast_columns, contrast_vector)
        if float(weight) != 0.0
    }
    if not weights:
        return []

    lo = int(round(window_s[0] / float(t_r)))
    hi = int(round(window_s[1] / float(t_r)))
    if hi - lo < 3:
        return []

    # peak index -> condition -> list of epochs
    gathered: List[Dict[str, List[np.ndarray]]] = [
        {name: [] for name in weights} for _ in wanted
    ]
    n_runs = 0

    for index, (bold_path, design_path) in enumerate(zip(bold_paths, design_paths)):
        try:
            design = pd.read_csv(str(design_path), sep="\t")
            image = nib.load(str(bold_path))
        except (OSError, ValueError) as exc:
            logger.info("Could not read run %d for the peak response (%s)", index, exc)
            continue

        frames = _frame_indices(design, float(t_r))
        if frames is None or image.ndim != 4 or int(frames.max()) >= image.shape[3]:
            logger.info(
                "Run %d's design cannot be aligned to its BOLD; skipping it.", index
            )
            continue

        present = [name for name in weights if name in design.columns]
        if not present:
            continue

        nuisance_columns = [
            column
            for column in design.columns
            if column != FRAME_TIME_COLUMN and column not in weights
        ]
        nuisance = (
            design[nuisance_columns].to_numpy(dtype=float)
            if nuisance_columns
            else np.empty((len(design), 0))
        )
        onsets = {
            name: onset_rows(design[name].to_numpy(dtype=float)) for name in present
        }

        inverse = np.linalg.inv(np.asarray(image.affine))
        shape = np.asarray(image.shape[:3])
        volume = np.asanyarray(image.dataobj)

        for peak_index, (_label, coordinate) in enumerate(wanted):
            voxel = np.rint(
                (np.append(np.asarray(coordinate, dtype=float), 1.0) @ inverse.T)[:3]
            ).astype(int)
            if np.any(voxel < 0) or np.any(voxel >= shape):
                continue
            series = _zscore(
                _project_out(
                    np.asarray(volume[tuple(voxel)], dtype=float)[frames], nuisance
                )
            )
            for name in present:
                gathered[peak_index][name].extend(
                    _epochs(series, onsets[name], lo=lo, hi=hi)
                )
        n_runs += 1

    if not n_runs:
        return []

    times = (np.arange(lo, hi) * float(t_r)).astype(float)
    collected: List[PeakResponse] = []
    for peak_index, (label, coordinate) in enumerate(wanted):
        conditions: List[ConditionResponse] = []
        for name, epochs in gathered[peak_index].items():
            if not epochs:
                continue
            stacked = np.vstack(epochs)
            conditions.append(
                ConditionResponse(
                    name=name,
                    weight=weights[name],
                    times=times,
                    mean=stacked.mean(axis=0),
                    # Standard error across events, which is the spread a reader
                    # judges the mean shape against. Not a confidence interval on
                    # the effect: the events are not independent of one another.
                    sem=stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                    if stacked.shape[0] > 1
                    else np.zeros(stacked.shape[1]),
                    n_events=int(stacked.shape[0]),
                )
            )
        if conditions:
            collected.append(
                PeakResponse(
                    label=str(label),
                    coordinate=tuple(float(c) for c in coordinate),
                    conditions=tuple(
                        sorted(conditions, key=lambda c: -c.weight)
                    ),
                    n_runs=n_runs,
                )
            )
    return collected


def peak_response_figure(
    responses: Sequence[PeakResponse],
    *,
    title: str = "",
) -> plt.Figure:
    """Draw each peak's mean response to each condition the contrast weights.

    One column per peak, both conditions overlaid. Read for shape: a peak driven by
    the task carries a rise, a plateau, and an undershoot, and the two conditions
    separate in the direction the contrast weights them. A peak driven by a few
    coincident frames carries neither.
    """
    rows = [item for item in responses if item.conditions]
    if not rows:
        raise ValueError("The peak response panel requires at least one peak.")

    palette = (OKABE_ITO["vermillion"], OKABE_ITO["blue"], OKABE_ITO["bluish_green"])

    with plot_context():
        figure, axes = plt.subplots(
            1,
            len(rows),
            figsize=(max(3.3 * len(rows), 4.6), 3.1),
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        total_events = 0
        for axis, item in zip(axes, rows):
            for position, condition in enumerate(item.conditions):
                colour = palette[position % len(palette)]
                axis.fill_between(
                    condition.times,
                    condition.mean - condition.sem,
                    condition.mean + condition.sem,
                    color=colour,
                    alpha=0.18,
                    linewidth=0,
                )
                axis.plot(
                    condition.times,
                    condition.mean,
                    color=colour,
                    linewidth=1.4,
                    label=f"{condition.name} ({condition.weight:+g}, "
                    f"{condition.n_events} events)",
                )
                total_events += condition.n_events
            axis.axvline(0.0, color=GUIDE_COLOR, linewidth=0.8, linestyle=":")
            axis.axhline(0.0, color=GUIDE_COLOR, linewidth=0.8)
            x, y, z = item.coordinate
            axis.set_title(
                f"peak {item.label}  ({x:+.0f}, {y:+.0f}, {z:+.0f})", fontsize=8.5
            )
            axis.set_xlabel("Time from onset (s)")

        axes[0].set_ylabel("BOLD (z, baseline-corrected)")
        axes[0].legend(loc="upper left", fontsize=6.5, frameon=False)
        if title:
            figure.suptitle(title, fontsize=10)

        annotate_provenance(
            figure,
            [
                f"{len(rows)} peak(s) · {rows[0].n_runs} run(s) · "
                f"{total_events:,} epochs",
                "mean across events, shaded band is the standard error across them",
                "onsets from the recorded design's own weighted columns; nuisance "
                "columns projected out; nothing is refitted",
                "shape is descriptive: the map's z is the test, not this panel",
            ],
        )
        return figure


__all__ = [
    "DEFAULT_MAX_PEAKS",
    "EPOCH_WINDOW_S",
    "FRAME_TIME_COLUMN",
    "ConditionResponse",
    "PeakResponse",
    "collect_peak_responses",
    "peak_response_figure",
]
