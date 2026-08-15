"""Remove the room's line comb from the BIDS EEG runs.

    eeg-pipeline line-comb benchmark
    eeg-pipeline line-comb apply

``benchmark`` injects known signals into every run, removes the lines, and reports
what survived against the criteria in :class:`PreservationGate`. ``apply`` writes a cleaned
copy of the BIDS dataset. Run the benchmark first; the criteria are stated before the
measurement, and a failure means the settings are wrong, not that the criteria should move.

The cleaned dataset keeps every sidecar byte-identical and rewrites only the ``.eeg``
binaries. Sampling rate, channel set, length and annotations are untouched, so the BIDS
contract downstream tooling relies on -- including the ``Volume`` and ``R`` marker names
the pipeline reads from ``events.tsv`` -- cannot drift.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.analysis.line_comb import removal as lr

WORKFLOW = "line_comb"

TASK = "thermalactive"
ESTIMATION_TR_COUNT = 60  # 54 s segments -> 18.5 mHz bins
BACKGROUND_HALF_WIDTH_HZ = 100.0 / 21.6
FILTER_LENGTH = "27s"
ROUNDTRIP_RELATIVE_TOLERANCE = 1e-6
"""Largest round-trip error accepted when reading a written binary back.

The binaries are float32, whose 24-bit mantissa gives a relative precision near 6e-8 of
full scale. A decade of headroom above that distinguishes quantisation from corruption.
"""
NOTCH_WIDTH_RATIO = 450.0
NOTCH_WIDTH_MIN_HZ = 0.05
"""Width around each target within which bins are subtracted: ``freq / ratio``.

This, not ``mt_bandwidth``, decides how much spectrum the removal takes, because
spectrum_fit subtracts a sinusoid at every bin inside it. The width scales with frequency
because the uncertainty does: the comb is mains-locked, so harmonic *k* inherits *k*
times the fundamental's wander.

    A sweep on sub-0009 run-1 fixed the width ratio. Reading the narrowest setting that
    still pushes every line below its local background:

    f/200 (MNE default)  25.3% of band   worst line -6.6 dB
    f/300                17.1%           worst line -6.1 dB
    f/450                12.1%           worst line -4.3 dB   <- chosen
    f/600                 8.4%           worst line +4.0 dB   (a line survives)
    fixed 0.10 Hz         8.8%           worst line +4.1 dB   (a line survives)

Below f/450 the high harmonics escape the window their own wander needs. The minimum of
0.05 Hz exceeds one bin at the 27 s working resolution, which the lowest harmonics need.
"""
MT_BANDWIDTH = 0.6
"""Multitaper bandwidth for the sinusoid estimate, in Hz.

At 0.6 Hz the estimation band reaches +/-0.3 Hz, half the distance to the neighbouring
comb line, so no line's amplitude is estimated from a band containing another. A sweep
over 4/10/20 s windows and 0.3/0.6/1.0 Hz bandwidths put every 10 s and 20 s setting
inside the original gate; 4 s failed because it cannot resolve a 1.2 Hz spacing. A later
blind-control audit found that 20 s still left a bin-quantized shoulder of an automatically
validated 54.024 Hz harmonic in sub-0011 run 1. A 27 s fit lowered aggregate residual
excess from 2.51 dB to 0.58 dB without widening the removal bands. It is also exactly half
the 54 s adaptive estimator window, avoiding the irregular tail window that longer choices
create inside MNE. Thus 27 s is the shortest block-aligned fit that passed the real gate.
"""


def adaptive_window_bounds(
    *,
    n_times: int,
    window_samples: int,
    hop_samples: int,
) -> tuple[tuple[int, int], ...]:
    """Overlapping fixed-length windows that cover a run exactly once reconstructed."""
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")
    if not 0 < hop_samples < window_samples:
        raise ValueError("hop_samples must lie between zero and window_samples.")
    if n_times < window_samples:
        raise ValueError(
            f"The recording holds {n_times} samples, fewer than the "
            f"{window_samples} of one adaptive estimation window. Lower "
            "`line_comb_removal.estimation_window_s`, at the cost of coarser frequency "
            "resolution, or use a longer recording."
        )

    tail_start = n_times - window_samples
    starts = list(range(0, tail_start + 1, hop_samples))
    if starts[-1] != tail_start:
        if tail_start - starts[-1] < hop_samples / 2.0:
            starts[-1] = tail_start
        else:
            starts.append(tail_start)
    bounds = tuple((start, start + window_samples) for start in starts)
    if any(
        right_start >= left_stop for (_, left_stop), (right_start, _) in zip(bounds, bounds[1:])
    ):
        raise ValueError("Adaptive estimation windows must overlap.")
    return bounds


def squared_sine_weights(
    bounds: tuple[tuple[int, int], ...],
    *,
    n_times: int,
) -> tuple[np.ndarray, ...]:
    """Positive squared-sine synthesis weights normalized to a partition of unity."""
    if not bounds:
        raise ValueError("At least one adaptive window is required.")
    raw_weights = []
    total = np.zeros(n_times, dtype=float)
    for start, stop in bounds:
        if not 0 <= start < stop <= n_times:
            raise ValueError("Adaptive window bounds lie outside the recording.")
        length = stop - start
        phase = np.pi * (np.arange(length, dtype=float) + 0.5) / length
        weight = np.sin(phase) ** 2
        raw_weights.append(weight)
        total[start:stop] += weight
    if np.any(total <= 0.0):
        raise ValueError("Adaptive windows do not cover every sample.")
    return tuple(weight / total[start:stop] for weight, (start, stop) in zip(raw_weights, bounds))


def overlap_add_segments(
    segments: tuple[np.ndarray, ...],
    bounds: tuple[tuple[int, int], ...],
    n_times: int,
) -> np.ndarray:
    """Reconstruct channel-by-time data with exact normalized overlap-add."""
    if len(segments) != len(bounds) or not segments:
        raise ValueError("segments and bounds must have the same non-zero length.")
    arrays = tuple(np.asarray(segment, dtype=float) for segment in segments)
    n_channels = arrays[0].shape[0]
    for segment, (start, stop) in zip(arrays, bounds):
        if segment.shape != (n_channels, stop - start):
            raise ValueError("Each segment must match its adaptive window bounds.")
        if not np.all(np.isfinite(segment)):
            raise ValueError("Adaptive filtered segments must be finite.")

    reconstructed = np.zeros((n_channels, n_times), dtype=float)
    for segment, weight, (start, stop) in zip(
        arrays,
        squared_sine_weights(bounds, n_times=n_times),
        bounds,
    ):
        reconstructed[:, start:stop] += segment * weight
    return reconstructed


@dataclass(frozen=True)
class RemovalSettings:
    """Everything the removal needs, resolved from configuration."""

    task: str = TASK
    """BIDS task label to process. Only this study's own default is study-specific."""
    estimation_window_s: float = float(hd.TR_SECONDS) * ESTIMATION_TR_COUNT
    """Length of each adaptive estimation window, in seconds.

    54 s here, which at this site is 60 TRs of 0.9 s -- chosen so the window is
    commensurate with the volume repetition and its 18.5 mHz bins land cleanly relative to
    the scanner's own periodicity. That reasoning is specific to a simultaneous EEG-fMRI
    acquisition; anywhere else it is simply the window length, and what matters is that it
    resolves the lines being removed and that the recording holds at least one of them.
    """
    max_band_cost: float | None = None
    """Optional ceiling on the share of 28-95 Hz a broadband signal may lose.

    ``None`` by default, and deliberately so: the cost is already determined by the
    evidence -- the notch width times the number of targets, with the width ratio fixed by
    a documented sweep and each target admitted by the replication rules -- so a shipped
    number could only be one chosen after seeing the answer. The retired default was 0.18,
    set because the cohort had reached 0.170.

    A study that wants a stated budget declares it here. That declaration is its own
    scientific decision, is recorded in the derivative's provenance, and ``apply`` refuses
    against it. What the cost actually was is measured and reported either way.
    """
    mains_notch_hz: tuple[float, float] = lr.MAINS_NOTCH_HZ
    """Band left to a downstream wide notch rather than taken here.

    59.5-60.5 Hz for a 60 Hz region. Set 49.5-50.5 where mains is 50 Hz. Only consulted
    when ``exclude_mains`` is true.
    """
    nominal_fundamental_hz: float = lr.NOMINAL_FUNDAMENTAL_HZ
    harmonic_range: tuple[int, int] = lr.COMB_HARMONIC_RANGE
    removal_harmonic_range: tuple[int, int] = lr.REMOVAL_HARMONIC_RANGE
    search_hz: float = 0.25
    min_prominence_db: float = 1.0
    filter_length: str = FILTER_LENGTH
    filter_jobs: int = 4
    mt_bandwidth: float = MT_BANDWIDTH
    notch_width_ratio: float = NOTCH_WIDTH_RATIO
    notch_width_min_hz: float = NOTCH_WIDTH_MIN_HZ
    uncertainty_confidence_z: float = 2.0
    low_hz: float = 3.0
    high_hz: float = 95.0
    detection_min_prominence_db: float = lr.LINE_PROMINENCE_FLOOR_DB
    detection_candidate_prominence_db: float = 6.0
    """Per-run floor for candidates that still require independent session replication."""
    detection_block_min_prominence_db: float = 15.0
    """Run-balanced strength required after scanning many short block spectra."""
    detection_low_hz: float = 20.0
    detection_high_hz: float = 100.0
    detection_search_hz: float = 0.05
    """Refinement window for a nominal that came from detection.

    Detected nominals sit on the summit already, so they need only enough room to refine
    sub-bin -- and the window has to stay narrow, because ``estimate_comb`` refuses a
    nominal within ``isolated_search_hz`` of a comb position on the grounds that a search
    that wide would refine onto the harmonic instead. It is right to: at 0.15 Hz it would.
    sub-0001's 93.759 Hz line sits 0.137 Hz from harmonic 78, so the detector offered it
    and the estimator raised, stopping the benchmark.

    Kept below the detector's own floor of one line width, so a line the detector admits
    can never be one the estimator refuses.
    """
    min_runs_per_line: int = 3
    """Runs a line must appear in before it becomes a session-wide target.

    The runs are replication already in hand -- one machine, minutes apart -- and they
    separate a persistent line from a one-run fluctuation. Strong lines confined to one
    recording take the independent temporal-replication route instead.
    """
    min_runs_per_block_line: int = 2
    min_independent_windows_per_line: int = 3
    """Non-overlapping windows needed to support a recording-specific line."""
    exclude_mains: bool = True
    """Leave 59.5-60.5 Hz to the pipeline's own notch.

    False moves mains into this pass, which is the point of doing so: the pipeline's FIR
    notch measured 0.97 Hz wide on the delivered epochs (59.537-60.463 Hz) against
    0.133 Hz for spectrum_fit at freq/450. Exactly one of the two may remove mains --
    ``preprocessing.notch_freq`` has to be null when this is False, and
    tests/scripts/line_comb/test_config_pairing.py fails if the two ever disagree.
    """
    excluded_bands_hz: tuple[tuple[float, float], ...] = ()
    """Bands ``line-comb notch`` takes wholesale, read from the config's ``notch_bands``.

    Same division of labour as ``exclude_mains``, for the same reason and against a
    different stage. A band is declared there precisely when the contamination is a
    *cluster* -- 75-118 distinct non-stationary peaks in the 0.9 Hz of 56.8-57.7 Hz here --
    and subtracting sinusoids from a cluster cannot clear it: the summit aimed at goes and
    its neighbour becomes the new summit.

    Leaving those bands targeted deadlocked the workflow rather than merely wasting effort.
    The surviving peak fails the residual criterion, which refuses ``apply``; the notch
    stage that removes the band outright reads what ``apply`` wrote, so it could never run.
    Measured on sub-0008, every recording failed at p=0.0244 -- the floor with 40 controls
    -- each with its worst residual in 57.15-57.35 Hz, while sub-0000 at a tenth of the
    57 Hz burden passed at p=1.0.
    """

    @property
    def protected_bands_hz(self) -> tuple[tuple[float, float], ...]:
        """Every band some other stage owns, so this pass must leave all of it alone."""
        bands = list(self.excluded_bands_hz)
        if self.exclude_mains:
            bands.append(tuple(self.mains_notch_hz))
        return tuple(sorted(bands))

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must name a BIDS task label.")
        if not np.isfinite(self.estimation_window_s) or self.estimation_window_s <= 0.0:
            raise ValueError("estimation_window_s must be finite and positive.")
        if self.max_band_cost is not None and not 0.0 < self.max_band_cost <= 1.0:
            raise ValueError("max_band_cost must be a share of the band, or null for none.")
        low_hz, high_hz = self.mains_notch_hz
        if not np.all(np.isfinite((low_hz, high_hz))) or not 0.0 < low_hz < high_hz:
            raise ValueError("mains_notch_hz must be an increasing positive band.")
        for band in self.excluded_bands_hz:
            low, high = band
            if not np.all(np.isfinite((low, high))) or not 0.0 < low < high:
                raise ValueError(
                    f"excluded_bands_hz must hold increasing positive bands; got {band}."
                )
        if not np.isfinite(self.uncertainty_confidence_z) or self.uncertainty_confidence_z <= 0:
            raise ValueError("uncertainty_confidence_z must be finite and positive.")
        if self.detection_candidate_prominence_db > self.detection_min_prominence_db:
            raise ValueError("The candidate prominence floor cannot exceed the confirmation floor.")
        if self.detection_block_min_prominence_db < self.detection_min_prominence_db:
            raise ValueError("The block confirmation floor cannot be below the whole-run floor.")
        if not 0.0 < self.detection_search_hz < lr._LINE_CLAIM_HZ:
            raise ValueError(
                f"detection_search_hz must lie between zero and {lr._LINE_CLAIM_HZ} Hz."
            )
        if self.min_runs_per_line < 2:
            raise ValueError("min_runs_per_line must require at least two independent runs.")
        if not 2 <= self.min_runs_per_block_line <= self.min_runs_per_line:
            raise ValueError(
                "min_runs_per_block_line must be at least two and no larger than "
                "min_runs_per_line."
            )
        if self.min_independent_windows_per_line < self.min_runs_per_block_line:
            raise ValueError(
                "min_independent_windows_per_line cannot be smaller than its run requirement."
            )
        if self.filter_jobs < 1:
            raise ValueError("filter_jobs must be positive.")

    @classmethod
    def from_config(cls, config) -> "RemovalSettings":
        """Read ``line_comb_removal`` from the workflow configuration."""
        defaults = cls()
        block = config.get("line_comb_removal") or {}
        retired = {
            "detect_isolated",
            "isolated_hz",
            "isolated_search_hz",
            "max_isolated_lines",
        } & set(block)
        if retired:
            raise ValueError(
                "Static isolated-line targeting is not supported; remove retired setting(s): "
                f"{sorted(retired)}."
            )
        # A config naming these describes a transform and a set of criteria that no longer
        # exist. Silently ignoring them would leave the author believing the epochs were
        # still being scored.
        epoch_settings = {
            "study_event_name",
            "study_epoch_s",
            "expected_study_events_per_run",
        } & set(block)
        if epoch_settings:
            raise ValueError(
                "Exact-epoch scoping was removed: the transform is continuous everywhere, "
                "and a 20 s epoch spectrum resolves 50 mHz against the 54 s window's "
                "18.5 mHz, so it authorised no source the windows did not already carry. "
                f"Remove setting(s): {sorted(epoch_settings)}."
            )
        harmonic_range = block.get("harmonic_range", list(defaults.harmonic_range))
        removal_range = block.get("removal_harmonic_range", list(defaults.removal_harmonic_range))
        dataset = config.get("dataset") or {}
        mains = block.get("mains_notch_hz", defaults.mains_notch_hz)
        # Top-level, not inside `line_comb_removal`: the bands belong to the notch stage,
        # and this pass reads them only to stay out of them.
        notch_bands = config.get("notch_bands") or ()
        for band in notch_bands:
            if not isinstance(band, Sequence) or isinstance(band, str) or len(band) != 2:
                raise ValueError(
                    f"notch_bands must hold [low, high] edge pairs; got {band!r}. The "
                    "removal reads them to stay out of the bands `line-comb notch` takes."
                )
        return cls(
            task=str(dataset.get("task", defaults.task)),
            estimation_window_s=float(
                block.get("estimation_window_s", defaults.estimation_window_s)
            ),
            mains_notch_hz=(float(mains[0]), float(mains[1])),
            max_band_cost=(
                None if block.get("max_band_cost") is None else float(block["max_band_cost"])
            ),
            nominal_fundamental_hz=float(
                block.get("nominal_fundamental_hz", defaults.nominal_fundamental_hz)
            ),
            harmonic_range=(int(harmonic_range[0]), int(harmonic_range[1])),
            removal_harmonic_range=(int(removal_range[0]), int(removal_range[1])),
            search_hz=float(block.get("search_hz", defaults.search_hz)),
            min_prominence_db=float(block.get("min_prominence_db", defaults.min_prominence_db)),
            filter_length=str(block.get("filter_length", defaults.filter_length)),
            filter_jobs=int(block.get("filter_jobs", defaults.filter_jobs)),
            mt_bandwidth=float(block.get("mt_bandwidth", defaults.mt_bandwidth)),
            notch_width_ratio=float(block.get("notch_width_ratio", defaults.notch_width_ratio)),
            notch_width_min_hz=float(block.get("notch_width_min_hz", defaults.notch_width_min_hz)),
            uncertainty_confidence_z=float(
                block.get("uncertainty_confidence_z", defaults.uncertainty_confidence_z)
            ),
            low_hz=float(block.get("low_hz", defaults.low_hz)),
            high_hz=float(block.get("high_hz", defaults.high_hz)),
            detection_min_prominence_db=float(
                block.get("detection_min_prominence_db", defaults.detection_min_prominence_db)
            ),
            detection_candidate_prominence_db=float(
                block.get(
                    "detection_candidate_prominence_db",
                    defaults.detection_candidate_prominence_db,
                )
            ),
            detection_block_min_prominence_db=float(
                block.get(
                    "detection_block_min_prominence_db",
                    defaults.detection_block_min_prominence_db,
                )
            ),
            detection_low_hz=float(block.get("detection_low_hz", defaults.detection_low_hz)),
            detection_high_hz=float(block.get("detection_high_hz", defaults.detection_high_hz)),
            min_runs_per_line=int(block.get("min_runs_per_line", defaults.min_runs_per_line)),
            min_runs_per_block_line=int(
                block.get("min_runs_per_block_line", defaults.min_runs_per_block_line)
            ),
            min_independent_windows_per_line=int(
                block.get(
                    "min_independent_windows_per_line",
                    defaults.min_independent_windows_per_line,
                )
            ),
            detection_search_hz=float(
                block.get("detection_search_hz", defaults.detection_search_hz)
            ),
            exclude_mains=bool(block.get("exclude_mains", defaults.exclude_mains)),
            excluded_bands_hz=tuple((float(low), float(high)) for low, high in notch_bands),
        )


@dataclass(frozen=True)
class AdaptiveWindowRemovalPlan:
    """One window's independently estimated transformation."""

    bounds: tuple[int, int]
    estimate: lr.CombEstimate
    targets_hz: tuple[float, ...]
    notch_widths_hz: tuple[float, ...]
    narrow_targets_hz: tuple[float, ...]
    aggregate_residual_targets_hz: tuple[float, ...] = ()
    aggregate_residual_widths_hz: tuple[float, ...] = ()
    channel_residual_targets_hz: tuple[tuple[float, ...], ...] = ()
    channel_residual_widths_hz: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        start, stop = self.bounds
        if not 0 <= start < stop:
            raise ValueError("Adaptive-window bounds must be positive and stop-exclusive.")
        if not self.targets_hz or len(self.targets_hz) != len(self.notch_widths_hz):
            raise ValueError("An adaptive window requires matching non-empty targets and widths.")
        if not all(np.isfinite(value) for value in (*self.targets_hz, *self.notch_widths_hz)):
            raise ValueError("Adaptive-window targets and widths must be finite.")
        if any(width <= 0.0 for width in self.notch_widths_hz):
            raise ValueError("Adaptive-window notch widths must be positive.")
        _validate_residual_targets(self, "Adaptive-window")


def _validate_residual_targets(window, label: str) -> None:
    """Validate aggregate and channel-local residual transforms."""
    if len(window.aggregate_residual_targets_hz) != len(window.aggregate_residual_widths_hz):
        raise ValueError("Aggregate residual targets and widths must match.")
    if len(window.channel_residual_targets_hz) != len(window.channel_residual_widths_hz):
        raise ValueError("Every channel residual-target list requires matching widths.")
    if any(
        len(targets) != len(widths)
        for targets, widths in zip(
            window.channel_residual_targets_hz,
            window.channel_residual_widths_hz,
        )
    ):
        raise ValueError("Channel residual targets and widths must match.")
    residual_targets = (
        *window.aggregate_residual_targets_hz,
        *(target for channel in window.channel_residual_targets_hz for target in channel),
    )
    residual_widths = (
        *window.aggregate_residual_widths_hz,
        *(width for channel in window.channel_residual_widths_hz for width in channel),
    )
    if not all(np.isfinite(value) for value in (*residual_targets, *residual_widths)):
        raise ValueError(f"{label} residual targets and widths must be finite.")
    if any(width <= 0.0 for width in residual_widths):
        raise ValueError(f"{label} residual targets and widths must be positive.")


@dataclass(frozen=True)
class RunRemovalPlan:
    """The immutable adaptive transformation benchmarked and applied to one run."""

    model: lr.AdaptiveCombModel
    windows: tuple[AdaptiveWindowRemovalPlan, ...]

    @property
    def all_targets_hz(self) -> tuple[float, ...]:
        """Every distinct frequency used by any adaptive window."""
        return tuple(
            sorted(
                {
                    *(target for window in self.windows for target in window.targets_hz),
                    *(
                        target
                        for window in self.windows
                        for target in window.aggregate_residual_targets_hz
                    ),
                    *(
                        target
                        for window in self.windows
                        for channel in window.channel_residual_targets_hz
                        for target in channel
                    ),
                }
            )
        )

    @property
    def all_narrow_targets_hz(self) -> tuple[float, ...]:
        """Every distinct comb-adjacent source authorised by raw-data evidence."""
        return tuple(
            sorted({target for window in self.windows for target in window.narrow_targets_hz})
        )



@dataclass(frozen=True)
class SessionRunSpectra:
    """Whole-run and block spectra supplying independent isolated-line evidence."""

    whole: tuple[np.ndarray, np.ndarray, np.ndarray]
    windows: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.windows or len(self.windows) != len(self.bounds):
            raise ValueError("SessionRunSpectra requires one bound per non-empty window list.")


@dataclass(frozen=True)
class RunIsolatedLinePlan:
    """Automatically supported isolated-line targets for one recording."""

    whole_hz: tuple[float, ...]
    window_hz: tuple[tuple[float, ...], ...]
    narrow_window_hz: tuple[tuple[float, ...], ...]
    source_count: int

    def __post_init__(self) -> None:
        if not self.window_hz:
            raise ValueError("An isolated-line plan requires at least one adaptive window.")
        if len(self.window_hz) != len(self.narrow_window_hz):
            raise ValueError("Every adaptive window requires one narrow-target list.")
        values = (
            *self.whole_hz,
            *(value for window in self.window_hz for value in window),
            *(value for window in self.narrow_window_hz for value in window),
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Isolated-line plan frequencies must be finite.")
        if self.source_count < 0:
            raise ValueError("Isolated-line source_count must not be negative.")

    @property
    def all_hz(self) -> tuple[float, ...]:
        """Every nominal used by any spectrum in the recording."""
        return tuple(
            sorted(
                {
                    *self.whole_hz,
                    *(value for row in self.window_hz for value in row),
                    *(value for row in self.narrow_window_hz for value in row),
                }
            )
        )


def read_bids_raw(vhdr: Path):
    """Read one BIDS recording with strict sidecar-derived metadata."""
    from mne_bids import get_bids_path_from_fname, read_raw_bids

    bids_path = get_bids_path_from_fname(vhdr)
    return read_raw_bids(
        bids_path,
        extra_params={"preload": True},
        on_ch_mismatch="raise",
        verbose="ERROR",
    )


def _window_removal_plan(
    bounds: tuple[int, int],
    estimate: lr.CombEstimate,
    narrow_targets_hz: Sequence[float],
    settings: RemovalSettings,
    *,
    spectrum_resolution_hz: float,
) -> AdaptiveWindowRemovalPlan:
    """Resolve one independently supported target set and its physical widths."""
    model_targets = lr.removal_frequencies(
        estimate,
        harmonic_range=settings.removal_harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
        excluded_hz=settings.protected_bands_hz,
    )
    model_widths = lr.uncertainty_aware_notch_widths(
        estimate,
        model_targets,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
        confidence_z=settings.uncertainty_confidence_z,
        isolated_minimum_hz=spectrum_resolution_hz,
    )
    narrow_array = np.asarray(narrow_targets_hz, dtype=float)
    if narrow_array.ndim != 1 or not np.all(np.isfinite(narrow_array)):
        raise ValueError("Narrow targets must be finite one-dimensional sequences.")
    narrow_widths = lr.notch_widths_for(
        narrow_array,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
    )
    target_widths = {
        float(target): float(width) for target, width in zip(model_targets, model_widths)
    }
    retained_narrow_targets = []
    for target, width in zip(narrow_array, narrow_widths):
        # Narrow targets arrive beside the comb model rather than through it, so
        # `removal_frequencies` never saw them and the band exclusion has to be repeated.
        if any(low <= float(target) <= high for low, high in settings.protected_bands_hz):
            continue
        covered_by_model = any(
            abs(float(target) - float(model_target)) + float(width) / 2.0
            <= float(model_width) / 2.0
            for model_target, model_width in zip(model_targets, model_widths)
        )
        if covered_by_model:
            continue
        target_widths[float(target)] = float(width)
        retained_narrow_targets.append(float(target))
    targets = tuple(sorted(target_widths))
    return AdaptiveWindowRemovalPlan(
        bounds=bounds,
        estimate=estimate,
        targets_hz=targets,
        notch_widths_hz=tuple(target_widths[target] for target in targets),
        narrow_targets_hz=tuple(retained_narrow_targets),
    )


def build_removal_plan(
    model: lr.AdaptiveCombModel,
    *,
    bounds: tuple[tuple[int, int], ...],
    narrow_targets_hz: tuple[tuple[float, ...], ...],
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Resolve model-supported targets and widths once for benchmark and apply."""
    if not len(bounds) == len(model.window_estimates) == len(narrow_targets_hz):
        raise ValueError("Window bounds, estimates and narrow targets must have the same length.")
    resolution_hz = spectrum_fit_nominal_resolution_hz(settings.filter_length)
    windows = tuple(
        _window_removal_plan(
            window_bounds,
            estimate,
            narrow_targets,
            settings,
            spectrum_resolution_hz=resolution_hz,
        )
        for window_bounds, estimate, narrow_targets in zip(
            bounds,
            model.window_estimates,
            narrow_targets_hz,
        )
    )
    return RunRemovalPlan(model=model, windows=windows)


def parse_channel_scaling(vhdr_path: Path) -> tuple[list[str], np.ndarray]:
    """Channel names and their binary resolution, in the file's own unit."""
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    binary_format = re.search(r"BinaryFormat=(\S+)", text)
    orientation = re.search(r"DataOrientation=(\S+)", text)
    if binary_format is None or binary_format.group(1) != "IEEE_FLOAT_32":
        raise ValueError(f"{vhdr_path.name}: expected IEEE_FLOAT_32 binary data.")
    if orientation is None or orientation.group(1) != "MULTIPLEXED":
        raise ValueError(f"{vhdr_path.name}: expected MULTIPLEXED data orientation.")

    # Channel definitions carry four comma-separated fields: name, reference, resolution,
    # unit. The classes exclude newlines so a `[Coordinates]` line, which holds only three
    # numbers, cannot be run into the one below it and parsed as `"-72\nCh2=1"`.
    names, resolutions = [], []
    for match in re.finditer(r"^Ch(\d+)=([^,\n]*),([^,\n]*),([^,\n]*),", text, flags=re.MULTILINE):
        names.append(match.group(2))
        resolutions.append(float(match.group(4)))
    if not names:
        raise ValueError(f"{vhdr_path.name}: no channel definitions found.")
    return names, np.asarray(resolutions, dtype=float)


def write_eeg_binary(vhdr_path: Path, destination: Path, data_volts: np.ndarray) -> None:
    """Write one ``.eeg`` binary in the layout its existing header already describes."""

    array = np.asarray(data_volts, dtype=float)
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"Refusing to write {destination}: {bad} non-finite sample(s). The round-trip "
            "check cannot catch this -- a NaN makes the deviation NaN, and NaN > tolerance "
            "is False -- so it is caught here instead."
        )

    names, resolutions = parse_channel_scaling(vhdr_path)
    if array.shape[0] != len(names):
        raise ValueError(
            f"{vhdr_path.name}: header describes {len(names)} channels, got {array.shape[0]}."
        )
    scaled = (array * 1e6) / resolutions[:, None]
    scaled.T.astype("<f4").tofile(destination)


def write_derivative_description(
    output_root: Path,
    source_root: Path,
    settings: RemovalSettings,
    source_version: str,
    band_cost: dict[str, float] | None = None,
) -> Path:
    """Declare the cleaned root a derivative and record what produced it.

    ``mirror_sidecars`` copies every sidecar byte-for-byte, so without this the cleaned
    dataset carried the raw one's description: DatasetType "raw", credit to MNE-BIDS alone,
    and nothing tying it to the removal, its settings or the code revision. BIDS asks
    derivatives to carry ``GeneratedBy`` for that reason -- otherwise the delivered data
    cannot be traced to the transformation that made it, which is the whole question an
    audit asks first.
    """
    import json
    from dataclasses import asdict

    path = Path(output_root) / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset description was not mirrored to {path}.")
    described = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(described, dict):
        raise ValueError("BIDS dataset_description.json must contain a JSON object.")

    described["DatasetType"] = "derivative"
    described.setdefault("Name", "line-comb cleaned EEG")
    described.setdefault("BIDSVersion", "1.8.0")
    existing_generated = described.get("GeneratedBy", [])
    if not isinstance(existing_generated, list) or not all(
        isinstance(entry, dict) for entry in existing_generated
    ):
        raise ValueError("BIDS GeneratedBy must be a list of objects.")
    generated = [
        entry for entry in existing_generated if "line-comb" not in str(entry.get("Name", ""))
    ]
    generated.append(
        {
            "Name": "line-comb",
            "Version": _code_revision(),
            "Description": (
                "Projection onto sinusoids at the measured comb and isolated-line "
                "frequencies, estimated in overlapping windows and reconstructed by "
                "normalized squared-sine overlap-add. A Thomson-F detector authorizes "
                "sliding sub-bin sinusoid regression only inside established artifact "
                "regions, per channel. Sidecars are byte-identical to the source; only "
                "the .eeg binaries differ."
            ),
            "Parameters": {
                "settings_fingerprint": settings_fingerprint(settings),
                # What the removal actually cost, so the delivered data carries it rather
                # than the reader having to find the benchmark that produced it.
                **({"band_cost": band_cost} if band_cost else {}),
                **{
                    k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(settings).items()
                },
            },
        }
    )
    described["GeneratedBy"] = generated
    described["SourceDatasets"] = [
        {"URL": f"../{Path(source_root).name}", "Version": source_version}
    ]

    path.write_text(json.dumps(described, indent=2) + "\n", encoding="utf-8")
    return path


def mirror_sidecars(source_root: Path, output_root: Path) -> int:
    """Copy every BIDS file except the binaries, which get rewritten."""
    copied = 0
    for path in sorted(source_root.rglob("*")):
        if path.is_dir() or path.suffix in {".eeg", ".lock"}:
            continue
        target = output_root / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def estimation_window_samples(sampling_frequency_hz: float, settings: RemovalSettings) -> int:
    """Samples in one adaptive estimation window, rounded to a whole sample."""
    samples = int(round(settings.estimation_window_s * float(sampling_frequency_hz)))
    if samples < 2:
        raise ValueError(
            f"estimation_window_s={settings.estimation_window_s:g} s is under two samples "
            f"at {sampling_frequency_hz:g} Hz."
        )
    return samples


def _block_psd(raw, settings: RemovalSettings):
    """EEG channel-by-window spectra on the adaptive estimator's grid."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    sfreq = float(raw.info["sfreq"])
    window_samples = estimation_window_samples(sfreq, settings)
    hop_samples = window_samples // 2
    data = raw.get_data(picks=picks)
    bounds = adaptive_window_bounds(
        n_times=data.shape[-1],
        window_samples=window_samples,
        hop_samples=hop_samples,
    )
    windows = np.stack([data[:, start:stop] for start, stop in bounds], axis=1)
    freqs, psd = hd.hann_periodogram(windows, sfreq)
    return freqs, psd, bounds


def run_spectra(raw, settings: RemovalSettings):
    """Whole-run and equal-duration block spectra on the same frequency grid."""
    freqs, psd, bounds = _block_psd(raw, settings)
    half_width = int(round(BACKGROUND_HALF_WIDTH_HZ / float(freqs[1])))
    whole_db = hd.to_db(np.median(psd.mean(axis=1), axis=0))
    whole = (freqs, whole_db, hd.prominence_db(whole_db, half_width_bins=half_width))
    per_block = []
    for block_psd in np.moveaxis(psd, 1, 0):
        block_db = hd.to_db(np.median(block_psd, axis=0))
        per_block.append((freqs, block_db, hd.prominence_db(block_db, half_width_bins=half_width)))
    return whole, tuple(per_block), bounds


def session_run_spectra(raw, settings: RemovalSettings) -> SessionRunSpectra:
    """All raw-data evidence scopes used to plan one continuous recording."""
    whole, windows, bounds = run_spectra(raw, settings)
    return SessionRunSpectra(whole=whole, windows=windows, bounds=bounds)


def run_spectrum(raw, settings: RemovalSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-median whole-run spectrum on a TR-commensurate grid."""
    whole, _, _ = run_spectra(raw, settings)
    return whole


def spatiotemporal_line_metrics(
    raw_before,
    raw_after,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> dict:
    """Focal residual excess relative to the unchanged pre-clean background."""
    import mne

    freqs, before_psd, before_bounds = _block_psd(raw_before, settings)
    after_freqs, after_psd, after_bounds = _block_psd(raw_after, settings)
    plan_bounds = tuple(window.bounds for window in plan.windows)
    if before_bounds != plan_bounds or after_bounds != plan_bounds:
        raise ValueError("The adaptive plan window geometry does not match the recording.")
    if not np.array_equal(freqs, after_freqs):
        raise ValueError("Before and after adaptive spectra use different frequency grids.")
    metrics = lr.adaptive_spatiotemporal_suppression(
        freqs,
        hd.to_db(before_psd),
        hd.to_db(after_psd),
        tuple(window.targets_hz for window in plan.windows),
        tuple(window.notch_widths_hz for window in plan.windows),
        background_half_width_hz=BACKGROUND_HALF_WIDTH_HZ,
        search_hz=lr.RESIDUAL_SEARCH_HZ,
    )
    picks = mne.pick_types(raw_before.info, eeg=True, exclude=())
    channel_index = int(metrics["worst_focal_channel_index"])
    metrics["worst_focal_channel_name"] = raw_before.ch_names[int(picks[channel_index])]
    return metrics


def adaptive_spectrum_db(
    raw, settings: RemovalSettings
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Channel-median spectra for every adaptive estimation window."""
    freqs, psd, bounds = _block_psd(raw, settings)
    values = []
    for window_psd in np.moveaxis(psd, 1, 0):
        values.append(hd.to_db(np.median(window_psd, axis=0)))
    return freqs, np.stack(values), bounds


def _reference_prominence(
    background_spectrum_db: np.ndarray,
    peak_spectrum_db: np.ndarray,
    *,
    half_width_bins: int,
) -> np.ndarray:
    """Prominence whose local floor cannot be changed by the cleaner."""
    background = np.asarray(background_spectrum_db, dtype=float)
    peaks = np.asarray(peak_spectrum_db, dtype=float)
    if background.shape != peaks.shape or background.ndim != 2:
        raise ValueError("Background and peak spectra must be matching two-dimensional arrays.")
    floors = np.stack(
        [hd.local_background_db(row, half_width_bins=half_width_bins) for row in background]
    )
    return peaks - floors


def adaptive_suppression_metrics(
    raw_before, raw_after, plan: RunRemovalPlan, settings: RemovalSettings
) -> dict[str, float]:
    """Aggregate residual evidence over the model-supported target positions."""
    freqs, before_db, before_bounds = adaptive_spectrum_db(raw_before, settings)
    after_freqs, after_db, after_bounds = adaptive_spectrum_db(raw_after, settings)
    plan_bounds = tuple(window.bounds for window in plan.windows)
    if before_bounds != plan_bounds or after_bounds != plan_bounds:
        raise ValueError("Adaptive spectra and fitted plan use different window geometry.")
    if not np.array_equal(freqs, after_freqs):
        raise ValueError("Before and after adaptive spectra use different frequency grids.")
    half_width = int(round(BACKGROUND_HALF_WIDTH_HZ / float(freqs[1])))
    before = _reference_prominence(before_db, before_db, half_width_bins=half_width)
    after = _reference_prominence(before_db, after_db, half_width_bins=half_width)
    widths = tuple(window.notch_widths_hz for window in plan.windows)
    metrics = lr.adaptive_line_suppression(
        freqs,
        before,
        after,
        tuple(window.targets_hz for window in plan.windows),
        widths,
        search_hz=lr.RESIDUAL_SEARCH_HZ,
    )
    return metrics


def continuous_refinement_metrics(
    plan: RunRemovalPlan,
    eeg_names: Sequence[str],
) -> dict[str, int | str]:
    """Summarise adaptive residual refinements with window/channel provenance."""
    channel_names = tuple(str(name) for name in eeg_names)
    aggregate_details = []
    focal_details = []
    focal_channel_windows = 0
    for window_index, window in enumerate(plan.windows):
        if window.channel_residual_targets_hz and len(window.channel_residual_targets_hz) != len(
            channel_names
        ):
            raise ValueError("The adaptive residual plan does not match the EEG channel names.")
        aggregate_details.extend(
            f"{window_index}:{frequency_hz:.6f}"
            for frequency_hz in window.aggregate_residual_targets_hz
        )
        for channel_index, targets in enumerate(window.channel_residual_targets_hz):
            focal_channel_windows += bool(targets)
            focal_details.extend(
                f"{window_index}:{channel_names[channel_index]}:{frequency_hz:.6f}"
                for frequency_hz in targets
            )
    return {
        "n_continuous_common_targets": sum(len(window.targets_hz) for window in plan.windows),
        "n_continuous_aggregate_refinement_targets": len(aggregate_details),
        "n_continuous_focal_refinement_targets": len(focal_details),
        "n_continuous_focal_refinement_channel_windows": focal_channel_windows,
        "continuous_aggregate_refinement_hz": ";".join(aggregate_details),
        "continuous_focal_refinement_hz": ";".join(focal_details),
    }


def _epoch_mean_psd(raw, bounds: Sequence[tuple[int, int]], picks):
    data = raw.get_data(picks=picks)
    epochs = np.stack([data[:, start:stop] for start, stop in bounds], axis=1)
    freqs, psd = hd.hann_periodogram(epochs, float(raw.info["sfreq"]))
    return freqs, psd.mean(axis=1)


def plan_target_spans(
    windows: Sequence[AdaptiveWindowRemovalPlan],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Each distinct target and the widest span any window removed around it.

    ``RunRemovalPlan.all_targets_hz`` deduplicates across windows while the widths are
    per-window, so the two cannot be zipped. Taking the maximum width per frequency gives
    the span the transform could have reached at that frequency anywhere in the run, which
    is what a measurement excluding the removals has to exclude.
    """
    spans: dict[float, float] = {}
    for window in windows:
        groups = (
            (window.targets_hz, window.notch_widths_hz),
            (window.aggregate_residual_targets_hz, window.aggregate_residual_widths_hz),
            *zip(window.channel_residual_targets_hz, window.channel_residual_widths_hz),
        )
        for targets, widths in groups:
            for target, width in zip(targets, widths):
                key = float(target)
                spans[key] = max(spans.get(key, 0.0), float(width))
    ordered = tuple(sorted(spans))
    return ordered, tuple(spans[target] for target in ordered)


def spectrum_fit_nominal_resolution_hz(filter_length: str) -> float:
    """Nominal FFT-bin resolution implied by MNE's spectrum-fit duration."""
    duration = "10s" if filter_length.lower() == "auto" else filter_length.lower()
    match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s)", duration)
    if match is None:
        raise ValueError("filter_length must be a positive duration such as '20s'.")
    seconds = float(match.group("value"))
    if match.group("unit") == "ms":
        seconds /= 1_000.0
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("filter_length must be positive.")
    return 1.0 / seconds


def spectrum_fit_frequency_grids(
    *,
    sampling_frequency_hz: float,
    filter_length: str,
    window_samples: int,
) -> tuple[np.ndarray, ...]:
    """Frequency grids used by MNE's inner spectrum-fit overlap-add windows."""
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    if window_samples < 1:
        raise ValueError("window_samples must be positive.")
    seconds = 1.0 / spectrum_fit_nominal_resolution_hz(filter_length)
    filter_samples = min(
        max(int(np.ceil(seconds * sampling_frequency_hz)), 1),
        window_samples,
    )
    if filter_samples < 2:
        raise ValueError("filter_length must contain at least two samples.")
    overlap_samples = (filter_samples + 1) // 2
    hop_samples = filter_samples - overlap_samples
    starts = np.arange(0, window_samples - filter_samples + 1, hop_samples)
    stops = starts + filter_samples
    stops[-1] = window_samples
    sample_counts = tuple(dict.fromkeys((stops - starts).tolist()))
    return tuple(
        np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency_hz)
        for sample_count in sample_counts
    )


def adaptive_band_metrics(
    *,
    sampling_frequency_hz: float,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> dict[str, float]:
    """Worst channel-level spectral cost of the transform that is actually applied.

    Only the continuous overlap-add is measured, because only it is applied. The exact
    study-window transform this used to include as well has been removed from both
    ``benchmark_run`` and ``apply_run``; its targets survive on the plan as the scopes the
    residual *measurements* are scored in, and charging the opportunity-cost ceiling for
    spectrum no filter ever touches made the gated number describe something other than
    the delivered data.
    """
    window_samples = {stop - start for start, stop in (window.bounds for window in plan.windows)}
    if len(window_samples) != 1:
        raise ValueError("Adaptive removal windows must have one fixed sample length.")
    grids = spectrum_fit_frequency_grids(
        sampling_frequency_hz=sampling_frequency_hz,
        filter_length=settings.filter_length,
        window_samples=window_samples.pop(),
    )

    def maximum_fraction(target_width_plans_for):
        measurements = []
        for freqs in grids:
            band_bin_count = int(np.count_nonzero((freqs >= 28.0) & (freqs <= 95.0)))
            for window in plan.windows:
                for targets, widths in target_width_plans_for(window):
                    fraction = lr.removed_band_fraction(freqs, targets, widths)
                    measurements.append((fraction, 1.0 / band_bin_count))
        return max(measurements, key=lambda item: (item[0], item[1]))

    def continuous_plans(window):
        common_targets = (*window.targets_hz, *window.aggregate_residual_targets_hz)
        common_widths = (*window.notch_widths_hz, *window.aggregate_residual_widths_hz)
        focal_targets = window.channel_residual_targets_hz or ((),)
        focal_widths = window.channel_residual_widths_hz or ((),)
        return tuple(
            ((*common_targets, *targets), (*common_widths, *widths))
            for targets, widths in zip(focal_targets, focal_widths)
        )

    expanded_fraction, expanded_bin_size = maximum_fraction(continuous_plans)
    base_fraction, base_bin_size = maximum_fraction(
        lambda window: (
            (
                window.targets_hz,
                lr.notch_widths_for(
                    window.targets_hz,
                    ratio=settings.notch_width_ratio,
                    minimum_hz=settings.notch_width_min_hz,
                ),
            ),
        )
    )
    return {
        "base_removed_band_fraction": base_fraction,
        "base_band_fraction_bin_size": base_bin_size,
        "width_expansion_band_fraction": expanded_fraction - base_fraction,
        "continuous_removed_band_fraction": expanded_fraction,
        "removed_band_fraction": expanded_fraction,
        "band_fraction_bin_size": expanded_bin_size,
    }


def clean_raw(
    raw,
    targets,
    *,
    filter_length: str,
    filter_jobs: int,
    mt_bandwidth: float,
    notch_widths,
):
    """Project the listed frequencies out of the EEG channels.

    ``notch_widths`` is always passed explicitly. Left to its default it becomes
    ``freq / 200``, which turns a line removal into a band removal.
    """
    import warnings

    from joblib import parallel_backend

    with warnings.catch_warnings(), parallel_backend("threading", n_jobs=filter_jobs):
        # scipy's DPSS eigenvalue side-computation overflows on long windows. The tapers
        # themselves are finite and orthonormal to 5e-4, and spectrum_fit uses only the
        # tapers, never the eigenvalues.
        warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)
        return raw.notch_filter(
            freqs=list(targets),
            picks="eeg",
            method="spectrum_fit",
            filter_length=filter_length,
            mt_bandwidth=mt_bandwidth,
            notch_widths=notch_widths,
            n_jobs=filter_jobs,
            verbose="ERROR",
        )


def _refine_regression_frequencies(
    data: np.ndarray,
    times: np.ndarray,
    targets_hz: Sequence[float],
    widths_hz: Sequence[float],
) -> tuple[float, ...]:
    """Locate each authorised sinusoid within one regression window."""
    centered = np.asarray(data, dtype=float) - float(np.mean(data))
    duration_s = times.size * float(times[1] - times[0])
    resolution_hz = 1.0 / duration_s
    refined = []
    for target_hz, width_hz in zip(targets_hz, widths_hz):
        search_half_width_hz = max(float(width_hz) / 2.0, resolution_hz)
        search_step_hz = resolution_hz / 10.0
        offsets = np.arange(
            -search_half_width_hz,
            search_half_width_hz + search_step_hz / 2.0,
            search_step_hz,
        )
        candidates_hz = float(target_hz) + offsets
        phases = np.exp(-2j * np.pi * times[:, np.newaxis] * candidates_hz)
        coefficients = np.einsum("i,ij->j", centered, phases, optimize=True)
        refined.append(float(candidates_hz[int(np.argmax(np.abs(coefficients)))]))
    return tuple(refined)


def _clean_channel_residuals(
    data: np.ndarray,
    picked_info,
    targets_hz: Sequence[Sequence[float]],
    widths_hz: Sequence[Sequence[float]],
    settings: RemovalSettings,
) -> np.ndarray:
    """Regress detected sinusoids jointly in overlapping spectrum-fit-length windows."""
    values = np.asarray(data, dtype=float)
    target_plans = tuple(tuple(float(value) for value in plan) for plan in targets_hz)
    width_plans = tuple(tuple(float(value) for value in plan) for plan in widths_hz)
    if values.ndim != 2 or values.shape[0] != len(picked_info["ch_names"]):
        raise ValueError("Channel-local cleaning requires channel-by-time data matching info.")
    if len(target_plans) != values.shape[0] or len(width_plans) != values.shape[0]:
        raise ValueError("Every channel requires one residual target and width plan.")
    if any(len(targets) != len(widths) for targets, widths in zip(target_plans, width_plans)):
        raise ValueError("Every channel's residual targets and widths must match.")
    planned_values = (
        *(target for plan in target_plans for target in plan),
        *(width for plan in width_plans for width in plan),
    )
    if not all(np.isfinite(value) for value in planned_values):
        raise ValueError("Channel-local residual targets and widths must be finite.")
    if any(width <= 0.0 for plan in width_plans for width in plan):
        raise ValueError("Channel-local residual widths must be positive.")

    sampling_frequency_hz = float(picked_info["sfreq"])
    if any(
        not 0.0 < target < sampling_frequency_hz / 2.0 for plan in target_plans for target in plan
    ):
        raise ValueError("Channel-local residual targets must lie between DC and Nyquist.")
    filter_seconds = 0.5 / spectrum_fit_nominal_resolution_hz(settings.filter_length)
    filter_samples = min(
        max(int(np.ceil(filter_seconds * sampling_frequency_hz)), 1),
        values.shape[1],
    )
    if filter_samples < 2:
        raise ValueError("Channel-local regression requires at least two samples.")
    bounds = (
        ((0, values.shape[1]),)
        if filter_samples == values.shape[1]
        else adaptive_window_bounds(
            n_times=values.shape[1],
            window_samples=filter_samples,
            hop_samples=filter_samples // 2,
        )
    )
    channel_groups: dict[
        tuple[tuple[float, ...], tuple[float, ...]],
        list[int],
    ] = {}
    for channel_index, targets in enumerate(target_plans):
        if targets:
            channel_groups.setdefault(
                (targets, width_plans[channel_index]),
                [],
            ).append(channel_index)

    # Locate each sinusoid once, over the whole window that evidenced it, and hold that
    # frequency fixed while amplitude and phase are re-fitted in every sub-window.
    #
    # Refining inside each sub-window instead made the subtraction unaccountable. The
    # search spans +/-width/2, up to about 125 mHz, while a 13.5 s sub-window resolves
    # 74 mHz -- so the search covers barely more than one independent frequency and its
    # argmax is a maximum over noise. Where the line was present that changed little;
    # where it was absent in that sub-window it selected the largest local fluctuation and
    # subtracted it, which is the mechanism _residual_line_candidates documents and
    # rejects for the across-channel route, applied across time instead. Taking the
    # maximum before subtracting also removed more than the two degrees of freedom the
    # regression is charged for.
    #
    # Nothing is lost, because the search could not have been tracking drift: the measured
    # f0 wander is 934 uHz across a whole run, so within one 54 s window a harmonic moves
    # about 2.6 mHz at k=48 and 4.6 mHz at k=83 -- one to two orders below the sub-window
    # resolution. Amplitude modulation, which is real and does vary sub-window to
    # sub-window, is still followed by the per-sub-window regression.
    whole_window_times = np.arange(values.shape[1], dtype=float) / sampling_frequency_hz
    refined_by_channel = {
        channel_index: _refine_regression_frequencies(
            values[channel_index],
            whole_window_times,
            targets,
            widths,
        )
        for (targets, widths), channel_indices in channel_groups.items()
        for channel_index in channel_indices
    }

    segments = []
    for start, stop in bounds:
        segment = values[:, start:stop].copy()
        times = np.arange(start, stop, dtype=float) / sampling_frequency_hz
        for (targets, widths), channel_indices in channel_groups.items():
            for channel_index in channel_indices:
                refined_targets = refined_by_channel[channel_index]
                angular_phase = 2.0 * np.pi * times[:, np.newaxis] * np.asarray(refined_targets)
                sinusoid_basis = np.column_stack((np.sin(angular_phase), np.cos(angular_phase)))
                design = np.column_stack((np.ones(times.size), sinusoid_basis))
                coefficients, _, rank, _ = np.linalg.lstsq(
                    design,
                    segment[channel_index],
                    rcond=None,
                )
                if rank != design.shape[1]:
                    raise ValueError("Channel-local sinusoid regression is rank deficient.")
                fitted_lines = np.einsum(
                    "ij,j->i",
                    sinusoid_basis,
                    coefficients[1:],
                    optimize=True,
                )
                if not np.all(np.isfinite(fitted_lines)):
                    raise ValueError(
                        "Channel-local sinusoid regression produced non-finite values."
                    )
                segment[channel_index] -= fitted_lines
        segments.append(segment)
    return overlap_add_segments(tuple(segments), bounds, values.shape[1])


def clean_continuous_raw(
    raw,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
    *,
    eeg_plan_indices: Sequence[int] | None = None,
):
    """Apply and overlap-add only the independently fitted continuous windows."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Adaptive line-comb removal requires at least one EEG channel.")
    planned_channel_counts = {
        len(window.channel_residual_targets_hz)
        for window in plan.windows
        if window.channel_residual_targets_hz
    }
    if len(planned_channel_counts) > 1:
        raise ValueError("Residual plans disagree about the EEG channel count.")
    planned_channel_count = next(iter(planned_channel_counts), 0)
    if eeg_plan_indices is None:
        if planned_channel_count not in (0, len(picks)):
            raise ValueError("The residual plan does not match the EEG channel count.")
        channel_plan_indices = tuple(range(len(picks)))
    else:
        channel_plan_indices = tuple(int(index) for index in eeg_plan_indices)
        if len(channel_plan_indices) != len(picks):
            raise ValueError("eeg_plan_indices must map every filtered EEG channel.")
        if planned_channel_count and any(
            not 0 <= index < planned_channel_count for index in channel_plan_indices
        ):
            raise ValueError("eeg_plan_indices contains an out-of-range plan channel.")
    bounds = tuple(window.bounds for window in plan.windows)
    squared_sine_weights(bounds, n_times=raw.n_times)
    picked_info = mne.pick_info(raw.info, picks, copy=True)
    segments = []
    for window in plan.windows:
        start, stop = window.bounds
        cleaned = _clean_planned_segment(
            raw.get_data(picks=picks, start=start, stop=stop),
            picked_info,
            window,
            settings,
            channel_plan_indices=channel_plan_indices,
        )
        segments.append(cleaned)

    output = raw.copy()
    output._data[picks] = overlap_add_segments(
        tuple(segments),
        bounds,
        raw.n_times,
    )
    return output


def _clean_planned_segment(
    data: np.ndarray,
    picked_info,
    window: AdaptiveWindowRemovalPlan,
    settings: RemovalSettings,
    *,
    channel_plan_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply common, aggregate-residual, then channel-local line transforms.

    The input is copied because ``RawArray`` may wrap the caller's buffer and MNE filters
    it in place. Without the copy the caller's data is cleaned as a side effect, which is
    silent and wrong: residual detection holds the pre-clean segment to compare against,
    and would compare the cleaned data with itself.
    """
    import mne

    segment = mne.io.RawArray(
        np.array(data, dtype=float, copy=True),
        picked_info.copy(),
        verbose="ERROR",
    )
    cleaned = clean_raw(
        segment,
        window.targets_hz,
        filter_length=settings.filter_length,
        filter_jobs=settings.filter_jobs,
        mt_bandwidth=settings.mt_bandwidth,
        notch_widths=np.asarray(window.notch_widths_hz),
    ).get_data()
    if not window.aggregate_residual_targets_hz and not window.channel_residual_targets_hz:
        return cleaned
    plan_indices = (
        tuple(range(cleaned.shape[0]))
        if channel_plan_indices is None
        else tuple(channel_plan_indices)
    )
    if len(plan_indices) != cleaned.shape[0]:
        raise ValueError("The residual plan does not map every supplied EEG channel.")
    channel_plans = []
    for index in plan_indices:
        target_width_pairs = list(
            zip(
                window.aggregate_residual_targets_hz,
                window.aggregate_residual_widths_hz,
            )
        )
        if window.channel_residual_targets_hz:
            target_width_pairs.extend(
                zip(
                    window.channel_residual_targets_hz[index],
                    window.channel_residual_widths_hz[index],
                )
            )
        channel_plans.append(_merge_residual_support(target_width_pairs))
    channel_plans = tuple(channel_plans)
    targets = tuple(plan[0] for plan in channel_plans)
    widths = tuple(plan[1] for plan in channel_plans)
    return _clean_channel_residuals(cleaned, picked_info, targets, widths, settings)


def _source_digest() -> str:
    """Content hash of the modules that decide what the removal does.

    A commit id is not enough: on a dirty tree it reads the same for every uncommitted
    state, so two different implementations would share a fingerprint, which is the one
    thing the fingerprint exists to prevent. Untracked files are invisible to it as well.
    Hashing the sources themselves has neither problem.
    """
    import hashlib

    here = Path(__file__).resolve().parent
    pain_study = here.parent.parent
    analysis_package = pain_study / "analysis" / "line_comb"
    sources = [*here.glob("*.py"), *analysis_package.glob("*.py")]
    if not sources:
        return "unknown"
    digest = hashlib.sha256()
    for source in sorted(sources):
        digest.update(str(source.relative_to(pain_study)).encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def _code_revision() -> str:
    """Short git revision of the tree, so a benchmark cannot outlive the code it measured."""
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def settings_fingerprint(settings: RemovalSettings) -> str:
    """A short stable hash of every setting that changes what the removal does.

    Binds a benchmark to the configuration it measured, so a stale or mismatched
    benchmark.tsv cannot stand in for one describing the settings about to be applied.
    """
    import hashlib
    import importlib.metadata
    import platform
    from dataclasses import asdict

    digest = _source_digest()
    if digest == "unknown":
        raise RuntimeError(
            "Cannot identify the removal source, so a benchmark cannot be bound to it. "
            "Refusing to fingerprint rather than certify data against unknown code."
        )
    runtime = {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in ("mne", "numpy", "scipy")},
    }
    payload = repr((sorted(asdict(settings).items()), digest, sorted(runtime.items()))).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def recording_digest(vhdr: Path) -> str:
    """Hash the complete BrainVision recording consumed by the transformation."""
    import hashlib

    vhdr = Path(vhdr)
    references = {}
    for line in vhdr.read_text(encoding="utf-8", errors="strict").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in {"DataFile", "MarkerFile"}:
            references[name] = value.strip()
    missing = {"DataFile", "MarkerFile"} - references.keys()
    if missing:
        raise ValueError(f"{vhdr}: missing BrainVision reference(s): {sorted(missing)}")

    paths = (vhdr, *(vhdr.parent / references[name] for name in ("DataFile", "MarkerFile")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"BrainVision component does not exist: {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(recordings: dict[str, str], source_root: Path) -> str:
    """Content identity for the recording bytes and every mirrored source sidecar."""
    import hashlib

    digest = hashlib.sha256(repr(sorted(recordings.items())).encode("utf-8"))
    for path in sorted(Path(source_root).rglob("*")):
        if not path.is_file() or path.suffix in {".eeg", ".lock"}:
            continue
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def removal_plan_digest(plan: RunRemovalPlan) -> str:
    """Stable identity for the exact fitted transformation of one recording."""
    import hashlib
    from dataclasses import asdict

    return hashlib.sha256(repr(asdict(plan)).encode("utf-8")).hexdigest()


def partial_benchmark_path(report_dir) -> Path:
    """Where an in-progress benchmark journals the recordings it has already measured.

    Deliberately not ``benchmark.tsv``. ``require_passing_benchmark`` exists partly because
    a benchmark that died on its second recording once left the previous run's file in place
    and the gates appeared to pass; writing incomplete work to the authoritative name would
    recreate exactly that. Nothing reads this file except a resuming benchmark.
    """
    return Path(report_dir) / "benchmark_partial.tsv"


def resumable_benchmark_rows(
    path,
    fingerprint: str,
    recordings: dict[str, str],
    plans: dict[str, str],
) -> dict[str, dict]:
    """Rows from an interrupted benchmark that still describe exactly this work.

    A row survives only if it was produced under these settings, from this recording's
    content, and from this run's fitted plan. Anything else is measurement of something
    other than what is about to be applied, and is thrown away rather than trusted.

    Failure to read the journal is never an error: recomputing costs time, and time is the
    thing this function exists to save, not the thing it exists to protect.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, sep="\t")
    except Exception as error:  # a damaged journal is worth redoing, never worth trusting
        print(f"  ignoring unreadable partial benchmark {path.name}: {error}")
        return {}
    required = {"recording", "settings_fingerprint", "input_digest", "plan_digest"}
    if not required.issubset(frame.columns) or frame["recording"].duplicated().any():
        print(f"  ignoring partial benchmark {path.name}: not a usable journal")
        return {}

    reusable = {}
    for row in frame.to_dict("records"):
        recording = row["recording"]
        if row.get("settings_fingerprint") != fingerprint:
            continue
        if recordings.get(recording) != row.get("input_digest"):
            continue
        if plans.get(recording) != row.get("plan_digest"):
            continue
        reusable[recording] = row
    return reusable


def require_passing_benchmark(
    path,
    settings: RemovalSettings,
    *,
    recordings: dict[str, str] | None = None,
    plans: dict[str, str] | None = None,
) -> None:
    """Refuse to write derived data without a passing benchmark of these settings.

    Each clause here is a way this went wrong in practice rather than a hypothetical. A
    benchmark.tsv from an earlier configuration was read as though it described the current
    one. A benchmark that raised on its second recording left the previous run's file in
    place, so the gates appeared to pass. And an apply was started under gates that were
    later found unable to fail at all.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Refusing to apply: no benchmark at {path}. Run `line-comb benchmark` first; "
            "the criteria are stated before the measurement for a reason."
        )
    frame = pd.read_csv(path, sep="	")
    expected = settings_fingerprint(settings)
    recorded = set(frame.get("settings_fingerprint", pd.Series(dtype=str)).dropna().unique())
    if recorded != {expected}:
        raise RuntimeError(
            f"Refusing to apply: {path} was produced under different settings "
            f"({recorded or 'none recorded'} against {expected}). Re-run the benchmark."
        )
    if not bool(frame["gate_passed"].all()):
        failed = int((~frame["gate_passed"].astype(bool)).sum())
        raise RuntimeError(
            f"Refusing to apply: {failed} of {len(frame)} benchmarked runs did not pass. "
            "A failure means the settings are wrong, not that the criteria should move."
        )
    if recordings is not None:
        if frame["recording"].duplicated().any():
            raise RuntimeError("Refusing to apply: benchmark contains duplicate recordings.")
        covered = set(frame["recording"])
        required = set(recordings)
        if covered != required:
            missing = sorted(required - covered)
            unexpected = sorted(covered - required)
            raise RuntimeError(
                "Refusing to apply: benchmark recordings differ from the apply set; "
                f"missing={missing}, unexpected={unexpected}."
            )
        benchmark_digests = frame.set_index("recording")["input_digest"].to_dict()
        changed = sorted(
            recording
            for recording, digest in recordings.items()
            if benchmark_digests.get(recording) != digest
        )
        if changed:
            raise RuntimeError(
                "Refusing to apply: input digest changed since benchmarking for " f"{changed}."
            )
    if plans is not None:
        benchmark_plans = frame.set_index("recording")["plan_digest"].to_dict()
        changed = sorted(
            recording
            for recording, digest in plans.items()
            if benchmark_plans.get(recording) != digest
        )
        if changed:
            raise RuntimeError(
                "Refusing to apply: fitted removal plan changed since benchmarking for "
                f"{changed}."
            )
    # Last, because a cohort statistic means nothing until the cohort is known to be the
    # right one. The seam criterion is absent from gate_passed -- a 2/41 per-run test
    # cannot be required of 90 runs at once -- so every row passing says nothing about it
    # and apply has to ask the cohort question itself.
    required_seam_columns = {
        "boundary_discontinuity_max_v",
        "boundary_control_maxima_v",
    }
    if not required_seam_columns.issubset(frame.columns):
        raise RuntimeError(
            f"Refusing to apply: {path} carries no seam measurements, so the cohort seam "
            "criterion cannot be evaluated. Re-run the benchmark."
        )
    seam = lr.seam_randomization_verdict(_seam_evidence_from_frame(frame))
    if not seam["passed"]:
        raise RuntimeError(
            "Refusing to apply: the cohort seam criterion failed -- "
            f"{int(seam['n_exceeding'])} of {int(seam['n_runs'])} runs exceeded their control "
            f"scale (count p={seam['count_p_value']:.4f}, maximum p="
            f"{seam['max_p_value']:.4f}), worst ratio {seam['max_ratio']:.2f}."
        )
    if settings.max_band_cost is not None:
        if "measured_band_attenuated_1db" not in frame.columns:
            raise RuntimeError(
                f"Refusing to apply: {path} carries no measured band cost, so the declared "
                f"budget of {settings.max_band_cost:.3f} cannot be checked. Re-run the benchmark."
            )
        worst = float(frame["measured_band_attenuated_1db"].max())
        if worst > settings.max_band_cost:
            raise RuntimeError(
                f"Refusing to apply: a broadband signal loses {worst:.3f} of 28-95 Hz on the "
                f"worst recording, above the {settings.max_band_cost:.3f} this study declared "
                "in `line_comb_removal.max_band_cost`."
            )
    for scope, column in (
        ("whole-run residual", "residual_null_p"),
        ("focal residual", "focal_residual_null_p"),
    ):
        if column not in frame.columns:
            raise RuntimeError(
                f"Refusing to apply: {path} carries no {column}, so the {scope} "
                "criterion cannot be evaluated. Re-run the benchmark."
            )
        verdict = lr.residual_randomization_verdict(frame[column].to_numpy())
        if not verdict["passed"]:
            raise RuntimeError(
                f"Refusing to apply: the cohort {scope} criterion failed -- "
                f"{int(verdict['n_discoveries'])} of {int(verdict['n_runs'])} recordings "
                f"exceed what their own matched controls reach "
                f"(smallest p={verdict['min_run_p_value']:.3g})."
            )
    # The residual-sinusoid probabilities are deliberately NOT consulted here. Residual
    # targets are selected by Thomson's F test on each exact epoch, and this criterion
    # repeats that same test on the same epochs after removing exactly what it found. The
    # multiplicity arithmetic is sound, but the inference is post-selection: it measures
    # whether the selected sinusoids were subtracted, not whether the detector missed one,
    # so it cannot certify detection completeness. Measured symptom -- p=0.70 on sub-0009
    # runs 1 and 4 while 8-11 dB residuals stood at 57.2 Hz.
    #
    # The independent acceptance test is the PSD matched-control gate, which scores a
    # different statistic that the detector does not optimise. The probabilities stay in
    # the benchmark and the cohort verdict is still printed, as provenance.


def _boundary_metrics(
    original: np.ndarray,
    cleaned: np.ndarray,
    boundaries: Sequence[int],
) -> dict[str, float | str]:
    evidence = lr.boundary_discontinuity_evidence(original, cleaned, boundaries)
    return {
        "max_boundary_discontinuity_ratio": evidence.ratio,
        "boundary_discontinuity_max_v": evidence.observed_max,
        "boundary_control_maxima_v": ";".join(f"{value:.17g}" for value in evidence.control_maxima),
    }


def _plan_transition_boundaries(plan: RunRemovalPlan, n_times: int) -> tuple[int, ...]:
    """Every interior sample where the set of contributing windows changes.

    A seam can only occur where one estimate starts or stops contributing, so the
    boundaries are the adaptive windows' own starts and stops. Both are included: with a
    half-window hop most stops coincide with a later start, but the tail window is placed
    to end exactly at ``n_times`` and so contributes a stop that no start repeats.

    ``plan.study_windows`` are deliberately *not* included. They were boundaries while an
    exact per-epoch transform was spliced into the continuous reconstruction; that
    transform no longer exists, and the correction is now a single overlap-add that passes
    through an epoch edge without changing scope. Feeding those ~22 points per run back in
    would bias this test toward passing, because ``boundary_discontinuity_evidence``
    matches its controls to the *count* of boundaries: the observed maximum would come
    from the real seams alone while each control maximum was taken over twice as many
    shifted indices.
    """
    boundaries = {
        boundary for window in plan.windows for boundary in window.bounds if 0 < boundary < n_times
    }
    if not boundaries:
        raise ValueError("The plan has no interior adaptive boundary to test for seams.")
    return tuple(sorted(boundaries))


def _seam_evidence_from_frame(frame: pd.DataFrame) -> tuple[lr.BoundaryDiscontinuityEvidence, ...]:
    evidence = []
    for row in frame.itertuples(index=False):
        controls = tuple(float(value) for value in row.boundary_control_maxima_v.split(";"))
        evidence.append(
            lr.BoundaryDiscontinuityEvidence(
                float(row.boundary_discontinuity_max_v),
                controls,
            )
        )
    return tuple(evidence)


def _detection_scaffold(freqs, spectrum_db, prominence, settings: RemovalSettings):
    """Fit the comb-only model that defines isolated-line clearance."""
    return lr.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=(),
        search_hz=settings.search_hz,
        isolated_search_hz=settings.detection_search_hz,
        min_prominence_db=settings.min_prominence_db,
    )


def detect_comb_adjacent_lines(
    freqs,
    spectrum_db,
    prominence,
    *,
    estimate: lr.CombEstimate,
    settings: RemovalSettings,
) -> tuple[float, ...]:
    """Detect distinct narrow sources beside an already supported comb target."""
    targets, widths = _comb_detection_support(estimate, settings)
    return _detect_lines_adjacent_to_targets(
        freqs,
        spectrum_db,
        prominence,
        targets=targets,
        widths=widths,
        settings=settings,
    )


def _comb_detection_support(
    estimate: lr.CombEstimate,
    settings: RemovalSettings,
) -> tuple[tuple[float, ...], np.ndarray]:
    """Comb targets and physical support used by adjacent-line detection."""
    targets = lr.removal_frequencies(
        estimate,
        harmonic_range=settings.removal_harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
        excluded_hz=settings.protected_bands_hz,
    )
    widths = lr.uncertainty_aware_notch_widths(
        estimate,
        targets,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
        confidence_z=settings.uncertainty_confidence_z,
        isolated_minimum_hz=spectrum_fit_nominal_resolution_hz(settings.filter_length),
    )
    return targets, widths


def _detect_lines_adjacent_to_targets(
    freqs,
    spectrum_db,
    prominence,
    *,
    targets: Sequence[float],
    widths: Sequence[float],
    settings: RemovalSettings,
) -> tuple[float, ...]:
    """Narrow summits connected to, but not contained by, supported targets."""
    frequency_array = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    prominence_array = np.asarray(prominence, dtype=float)
    if not frequency_array.shape == spectrum.shape == prominence_array.shape:
        raise ValueError("freqs, spectrum_db and prominence must have the same shape.")
    if frequency_array.ndim != 1 or frequency_array.size < 3:
        raise ValueError("Comb-adjacent detection requires a one-dimensional spectrum.")
    frequency_steps = np.diff(frequency_array)
    if np.any(frequency_steps <= 0.0):
        raise ValueError("Comb-adjacent detection requires increasing frequencies.")
    frequency_resolution_hz = float(np.median(frequency_steps))

    target_array = np.asarray(targets, dtype=float)
    width_array = np.asarray(widths, dtype=float)
    if target_array.ndim != 1 or target_array.size == 0 or width_array.shape != target_array.shape:
        raise ValueError("Adjacent-line targets and widths must be matching non-empty vectors.")
    if not np.all(np.isfinite(target_array)) or np.any(width_array <= 0.0):
        raise ValueError("Adjacent-line targets and widths must be finite and positive.")
    covered_reaches = np.maximum(width_array / 2.0, frequency_resolution_hz)

    summits = np.zeros(prominence_array.shape, dtype=bool)
    summits[1:-1] = (
        np.isfinite(prominence_array[1:-1])
        & (prominence_array[1:-1] > prominence_array[:-2])
        & (prominence_array[1:-1] >= prominence_array[2:])
    )
    inside = (frequency_array >= settings.detection_low_hz) & (frequency_array <= settings.high_hz)
    candidate_indices = np.flatnonzero(
        summits & inside & (prominence_array >= settings.detection_min_prominence_db)
    )

    candidates = []
    for index in candidate_indices:
        position_hz = float(frequency_array[index])
        distances = np.abs(target_array - position_hz)
        if float(np.min(distances)) > lr.RESIDUAL_SEARCH_HZ:
            continue
        peak_width_hz = lr._peak_width_hz(frequency_array, prominence_array, int(index))
        if peak_width_hz > lr.LINE_WIDTH_CEILING_HZ:
            continue
        if np.any(distances <= covered_reaches):
            continue
        candidates.append((position_hz, float(prominence_array[index])))

    minimum_separation_hz = spectrum_fit_nominal_resolution_hz(settings.filter_length)
    accepted = []
    for position_hz, strength_db in sorted(
        candidates,
        key=lambda item: (-item[1], item[0]),
    ):
        if any(abs(position_hz - taken) <= minimum_separation_hz for taken in accepted):
            continue
        accepted.append(position_hz)
    return tuple(sorted(accepted))


def _estimate_spectrum(spectrum, settings: RemovalSettings, isolated_hz) -> lr.CombEstimate:
    freqs, spectrum_db, prominence = spectrum
    return lr.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=isolated_hz,
        search_hz=settings.search_hz,
        isolated_search_hz=settings.detection_search_hz,
        min_prominence_db=settings.min_prominence_db,
    )


def build_run_plan_from_spectra(
    spectra: SessionRunSpectra,
    settings: RemovalSettings,
    isolated_lines: RunIsolatedLinePlan,
) -> RunRemovalPlan:
    """Fit independently supported targets for every overlapping run window."""
    if len(spectra.windows) != len(isolated_lines.window_hz):
        raise ValueError("Each adaptive spectrum requires its own isolated-line target list.")
    whole_estimate = _estimate_spectrum(spectra.whole, settings, isolated_lines.whole_hz)
    window_estimates = tuple(
        _estimate_spectrum(window, settings, nominals)
        for window, nominals in zip(spectra.windows, isolated_lines.window_hz)
    )
    model = lr.build_adaptive_comb_model(whole_estimate, window_estimates)
    plan = build_removal_plan(
        model,
        bounds=spectra.bounds,
        narrow_targets_hz=isolated_lines.narrow_window_hz,
        settings=settings,
    )
    plan = _ensure_routed_isolated_targets(
        plan,
        isolated_lines.window_hz,
        settings,
    )
    return _expand_widths_to_observed_line_support(plan, spectra, settings)


def _ensure_routed_isolated_targets(
    plan: RunRemovalPlan,
    routed_targets_hz: tuple[tuple[float, ...], ...],
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Keep exact-window evidence even when a 54-second fit dilutes its source."""
    if len(plan.windows) != len(routed_targets_hz):
        raise ValueError("Every removal window requires one routed isolated-target list.")
    minimum_width_hz = spectrum_fit_nominal_resolution_hz(settings.filter_length)
    windows = []
    for window, routed_targets in zip(plan.windows, routed_targets_hz):
        target_widths = dict(zip(window.targets_hz, window.notch_widths_hz))
        for target in routed_targets:
            width = max(
                target / settings.notch_width_ratio,
                settings.notch_width_min_hz,
                minimum_width_hz,
            )
            covered = any(
                abs(target - existing_target) + width / 2.0 <= existing_width / 2.0
                for existing_target, existing_width in target_widths.items()
            )
            if not covered:
                target_widths[target] = width
        targets = tuple(sorted(target_widths))
        widths = tuple(target_widths[target] for target in targets)
        windows.append(
            replace(
                window,
                targets_hz=targets,
                notch_widths_hz=widths,
            )
        )
    return replace(plan, windows=tuple(windows))


def _expand_widths_to_observed_line_support(
    plan: RunRemovalPlan,
    spectra: SessionRunSpectra,
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Cover narrow line support observed around each independently valid target."""
    expanded_windows = []
    for window_index, window in enumerate(plan.windows):
        evidence = (spectra.windows[window_index],)
        expanded_windows.append(_expand_window_to_observed_support(window, evidence, settings))
    return replace(plan, windows=tuple(expanded_windows))


def _expand_window_to_observed_support(
    window: AdaptiveWindowRemovalPlan,
    evidence: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    settings: RemovalSettings,
    *,
    localization_margin_hz: float = 0.0,
) -> AdaptiveWindowRemovalPlan:
    """Cover strong narrow support beside a validated target, where it was observed.

    A notch is symmetric about its target, so stretching the target's own width out to an
    asymmetric peak also empties the mirror image of that peak, on the side where nothing
    was ever seen. Support running from ``target + 0.025`` to ``target + 0.275`` used to
    set a width of 0.55 Hz -- about half the comb spacing -- to cover 0.25 Hz of evidence,
    and half of what it removed was chosen by arithmetic rather than observed.

    The support gets its own notch instead, centred on the interval that was measured, so
    the cost is the width of the evidence. Where the support does sit symmetrically the two
    formulations agree, and where the existing notch already covers it nothing is added.
    """
    if not np.isfinite(localization_margin_hz) or localization_margin_hz < 0.0:
        raise ValueError("localization_margin_hz must be finite and non-negative.")
    targets = np.asarray(window.targets_hz, dtype=float)
    widths = np.asarray(window.notch_widths_hz, dtype=float).copy()
    covers: list[tuple[float, float]] = []
    for freqs, _, prominence in evidence:
        frequency_array = np.asarray(freqs, dtype=float)
        prominence_array = np.asarray(prominence, dtype=float)
        if frequency_array.size < 2:
            raise ValueError("Support evidence needs a frequency grid of at least two bins.")
        # Support is read off a discrete grid, so a peak confined to one bin still occupies
        # that bin's full span. Adding one resolution step states exactly the width of the
        # bins observed, and keeps a single-bin support from asking for a zero-width notch.
        resolution_hz = float(frequency_array[1] - frequency_array[0])
        summits = np.zeros(prominence_array.shape, dtype=bool)
        summits[1:-1] = (
            np.isfinite(prominence_array[1:-1])
            & (prominence_array[1:-1] > prominence_array[:-2])
            & (prominence_array[1:-1] >= prominence_array[2:])
        )
        candidate_indices = np.flatnonzero(
            summits & (prominence_array >= settings.detection_min_prominence_db)
        )
        for index in candidate_indices:
            if (
                lr._peak_width_hz(frequency_array, prominence_array, int(index))
                > lr.LINE_WIDTH_CEILING_HZ
            ):
                continue
            position_hz = float(frequency_array[index])
            target_index = int(np.argmin(np.abs(targets - position_hz)))
            target_hz = float(targets[target_index])
            if abs(target_hz - position_hz) > lr.RESIDUAL_SEARCH_HZ:
                continue
            left_hz, right_hz = lr._peak_support_bounds_hz(
                frequency_array,
                prominence_array,
                int(index),
            )
            reach_hz = float(widths[target_index]) / 2.0
            if abs(left_hz - target_hz) <= reach_hz and abs(right_hz - target_hz) <= reach_hz:
                continue
            covers.append(
                (
                    (left_hz + right_hz) / 2.0,
                    (right_hz - left_hz) + resolution_hz + 2.0 * localization_margin_hz,
                )
            )

    if not covers:
        return window
    cover_targets, cover_widths = _merge_residual_support(tuple(covers))
    return replace(
        window,
        targets_hz=(*window.targets_hz, *cover_targets),
        notch_widths_hz=(*tuple(widths), *cover_widths),
    )


def _refine_window_residual_plan(
    original: np.ndarray,
    picked_info,
    window: AdaptiveWindowRemovalPlan,
    settings: RemovalSettings,
    clean_segment,
    *,
    cleaned_data: np.ndarray | None = None,
) -> AdaptiveWindowRemovalPlan:
    """Encode the sinusoids that survived one window's first pass.

    What licenses a subtraction here is ``lr.ResidualDetection`` -- Thomson's F test on
    the first pass's own output -- and never the tolerance the benchmark will judge the
    result by. See that class for why the two have to stay apart.
    """
    sampling_frequency_hz = float(picked_info["sfreq"])
    residual_width_hz = 2.0 * max(
        sampling_frequency_hz / original.shape[1],
        2.0 * spectrum_fit_nominal_resolution_hz(settings.filter_length),
    )
    aggregate_targets = list(window.aggregate_residual_targets_hz)
    focal_targets = (
        [list(values) for values in window.channel_residual_targets_hz]
        if window.channel_residual_targets_hz
        else [[] for _ in range(original.shape[0])]
    )
    if len(focal_targets) != original.shape[0]:
        raise ValueError("The residual plan does not match the supplied EEG channels.")

    cleaned = (
        clean_segment(
            original,
            picked_info,
            _encode_residual_targets(window, aggregate_targets, focal_targets, residual_width_hz),
            settings,
        )
        if cleaned_data is None
        else np.asarray(cleaned_data, dtype=float)
    )
    if cleaned.shape != original.shape or not np.all(np.isfinite(cleaned)):
        raise ValueError("Residual refinement requires finite cleaned data matching the input.")

    shared_candidates, focal_candidates = _residual_line_candidates(
        (cleaned,),
        sampling_frequency_hz=sampling_frequency_hz,
        window=window,
        settings=settings,
    )

    for frequency_hz in shared_candidates:  # always empty; the shared route is gone
        if not _already_searched(frequency_hz, aggregate_targets, residual_width_hz):
            aggregate_targets.append(frequency_hz)
    for channel_index, channel_candidates in enumerate(focal_candidates):
        for frequency_hz in channel_candidates:
            if _already_searched(frequency_hz, aggregate_targets, residual_width_hz):
                continue
            if not _already_searched(frequency_hz, focal_targets[channel_index], residual_width_hz):
                focal_targets[channel_index].append(frequency_hz)
    return _encode_residual_targets(window, aggregate_targets, focal_targets, residual_width_hz)


def _residual_line_candidates(
    states: Sequence[np.ndarray],
    *,
    sampling_frequency_hz: float,
    window: AdaptiveWindowRemovalPlan,
    settings: RemovalSettings,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Sinusoids evidenced in one window's data, per channel.

    Every candidate is channel-local, and nothing is subtracted from a channel that did
    not evidence it. An earlier version routed a frequency carried by half the array into
    every channel's plan; ``_clean_channel_residuals`` then searched each channel
    separately and subtracted whichever fluctuation was largest inside the width, so in a
    channel without the artifact it removed whatever was there, possibly signal. It was
    never the joint array estimate its docstring claimed. It produced zero shared targets
    on both sub-0009 runs measured, so the route is gone rather than repaired.

    The state passed is the first pass's own output, which is the raw data with the
    already-modelled component accounted for: that is how a line hidden under a stronger
    neighbour's skirt becomes visible, and it is the case this second pass exists for.

    Also testing the raw segment was measured and reverted, on the continuous evidence
    alone: it added about eleven thousand focal targets per recording, raising the removed
    band fraction from 0.149 to 0.169 and lowering in-band probe survival from 0.204 to
    0.153, to move one gate. Every one of those targets is a hole in the delivered
    spectrum, so the trade was refused.

    What that experiment could NOT show is whether the raw segment helps inside the exact
    windows. There the caller passes no precomputed output, so the segment cleaner ran on
    the same array that was to serve as the raw state -- and cleaned it in place, which is
    why the exact-window result came back bit-identical. The aliasing is fixed in
    ``_clean_planned_segment``; the exact-window question is simply unmeasured.

    Whatever states are supplied, none of them is the tolerance the benchmark judges the
    result by. That is what keeps the acceptance gates able to fail.
    """
    if not states:
        raise ValueError("At least one state of the segment is required.")
    detection = lr.ResidualDetection()
    neighbourhood = {
        "targets_hz": window.targets_hz,
        "widths_hz": window.notch_widths_hz,
        "responsibility_hz": lr.RESIDUAL_SEARCH_HZ,
    }
    focal: list[list[float]] = [[] for _ in range(states[0].shape[0])]
    for state in states:
        frequencies, statistic, threshold, _ = lr.thomson_f_statistics(
            state,
            sampling_frequency_hz=sampling_frequency_hz,
            bandwidth_hz=settings.mt_bandwidth,
            family_alpha=detection.family_alpha,
        )
        channel_candidates = lr.focal_residual_line_candidates(
            frequencies,
            statistic,
            threshold=threshold,
            **neighbourhood,
        )
        for channel_index, values in enumerate(channel_candidates):
            focal[channel_index].extend(values)
    return (), tuple(tuple(sorted(values)) for values in focal)


def _already_searched(
    frequency_hz: float,
    encoded_hz: Sequence[float],
    width_hz: float,
) -> bool:
    """Whether a residual search already encoded covers this frequency."""
    return any(abs(frequency_hz - existing_hz) <= width_hz / 2.0 for existing_hz in encoded_hz)


def _encode_residual_targets(
    window: AdaptiveWindowRemovalPlan,
    aggregate_targets: Sequence[float],
    focal_targets: Sequence[Sequence[float]],
    width_hz: float,
) -> AdaptiveWindowRemovalPlan:
    """Attach one window's residual searches, each the same measured width."""
    aggregate = tuple(sorted(aggregate_targets))
    focal = tuple(tuple(sorted(values)) for values in focal_targets)
    return replace(
        window,
        aggregate_residual_targets_hz=aggregate,
        aggregate_residual_widths_hz=(width_hz,) * len(aggregate),
        channel_residual_targets_hz=focal,
        channel_residual_widths_hz=tuple((width_hz,) * len(values) for values in focal),
    )


def _merge_residual_support(
    target_width_pairs: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Merge overlapping frequency searches without duplicating a fitted source."""
    ordered = sorted((float(target), float(width)) for target, width in target_width_pairs)
    if not ordered:
        return (), ()
    clusters: list[list[tuple[float, float]]] = []
    cluster_right = float("-inf")
    for target, width in ordered:
        left = target - width / 2.0
        right = target + width / 2.0
        if clusters and left <= cluster_right:
            clusters[-1].append((target, width))
            cluster_right = max(cluster_right, right)
        else:
            clusters.append([(target, width)])
            cluster_right = right
    targets = []
    widths = []
    for cluster in clusters:
        if len(cluster) == 1:
            target, width = cluster[0]
        else:
            left = min(target - width / 2.0 for target, width in cluster)
            right = max(target + width / 2.0 for target, width in cluster)
            target = (left + right) / 2.0
            width = right - left
        targets.append(target)
        widths.append(width)
    return tuple(targets), tuple(widths)


def _route_continuous_residual_support(plan: RunRemovalPlan) -> RunRemovalPlan:
    """Give every overlapping synthesis window the residuals evidenced in its support."""
    channel_counts = {
        len(window.channel_residual_targets_hz)
        for window in plan.windows
        if window.channel_residual_targets_hz
    }
    if len(channel_counts) > 1:
        raise ValueError("Continuous residual plans disagree about the EEG channel count.")
    channel_count = next(iter(channel_counts), 0)
    routed = []
    for destination in plan.windows:
        sources = tuple(
            source
            for source in plan.windows
            if _intervals_overlap(destination.bounds, source.bounds)
        )
        aggregate_targets, aggregate_widths = _merge_residual_support(
            tuple(
                (target, width)
                for source in sources
                for target, width in zip(
                    source.aggregate_residual_targets_hz,
                    source.aggregate_residual_widths_hz,
                )
            )
        )
        focal_targets = []
        focal_widths = []
        for channel_index in range(channel_count):
            targets, widths = _merge_residual_support(
                tuple(
                    (target, width)
                    for source in sources
                    for target, width in zip(
                        source.channel_residual_targets_hz[channel_index],
                        source.channel_residual_widths_hz[channel_index],
                    )
                )
            )
            focal_targets.append(targets)
            focal_widths.append(widths)
        routed.append(
            replace(
                destination,
                aggregate_residual_targets_hz=aggregate_targets,
                aggregate_residual_widths_hz=aggregate_widths,
                channel_residual_targets_hz=tuple(focal_targets),
                channel_residual_widths_hz=tuple(focal_widths),
            )
        )
    return replace(plan, windows=tuple(routed))


def _refine_continuous_residual_plans(
    raw,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Resolve residuals independently inside every continuous estimation window."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Continuous residual refinement requires at least one EEG channel.")
    picked_info = mne.pick_info(raw.info, picks, copy=True)
    base_cleaned = clean_continuous_raw(
        raw.copy(),
        plan,
        settings,
    )
    refined = []
    for window in plan.windows:
        start, stop = window.bounds
        refined.append(
            _refine_window_residual_plan(
                raw.get_data(picks=picks, start=start, stop=stop),
                picked_info,
                window,
                settings,
                _clean_planned_segment,
                cleaned_data=base_cleaned.get_data(
                    picks=picks,
                    start=start,
                    stop=stop,
                ),
            )
        )
    return _route_continuous_residual_support(replace(plan, windows=tuple(refined)))


def isolated_line_summary(recording: str, plan: RunIsolatedLinePlan) -> str:
    """One recording's isolated-line evidence, with both of its counts named.

    A source is a cluster of nominals inside the spectral resolution, so the two numbers
    legitimately differ -- 27.9815 and 28.0000 are one source observed twice. Printing the
    source count beside the nominal list without saying which was which read as an
    arithmetic error in output that gets quoted in a methods section.
    """
    nominals = plan.all_hz
    listed = ", ".join(f"{frequency:.4f}" for frequency in nominals) or "none"
    return (
        f"  {recording}: {plan.source_count} artifact source(s) as "
        f"{len(nominals)} nominal(s) at {listed}"
    )


def build_run_plans(runs: list[Path], settings: RemovalSettings) -> dict[str, RunRemovalPlan]:
    """Fit every run independently, sharing only replicated isolated-line nominals."""
    import mne

    mne.set_log_level("ERROR")
    by_subject: dict[str, list[Path]] = {}
    for vhdr in runs:
        by_subject.setdefault(vhdr.parent.parent.name, []).append(vhdr)

    plans = {}
    for subject, subject_runs in by_subject.items():
        spectra = {}
        for vhdr in subject_runs:
            raw = read_bids_raw(vhdr)
            spectra[vhdr] = session_run_spectra(raw, settings)
        isolated_line_plans = automatic_line_plans(
            list(spectra.values()),
            settings,
        )
        for vhdr, isolated_lines in zip(subject_runs, isolated_line_plans):
            run_evidence = spectra[vhdr]
            print(isolated_line_summary(vhdr.stem, isolated_lines), flush=True)
            try:
                plan = build_run_plan_from_spectra(run_evidence, settings, isolated_lines)
                raw = read_bids_raw(vhdr)
                plan = _refine_continuous_residual_plans(raw, plan, settings)
            except ValueError as error:
                raise ValueError(f"{vhdr.stem}: {error}") from error
            plans[vhdr.stem] = plan
            print(
                f"  {vhdr.stem}: {len(plan.windows)} adaptive windows, "
                f"f0 range={plan.model.fundamental_range_hz * 1e6:.0f} uHz, "
                f"max step={plan.model.max_adjacent_shift_hz * 1e6:.0f} uHz",
                flush=True,
            )
    return plans


def _preservation_against_control(
    freqs,
    *,
    probe,
    probe_before,
    probe_after,
    probe_control,
    data_before,
    data_after,
    data_control,
    targets,
    widths,
    control_targets,
    control_widths,
) -> dict[str, float]:
    """Compare what the transform left alone with what a displaced one leaves alone.

    Both quantities should be zero and are not quite: the removal is local, but not
    perfectly, so a little of it reaches frequencies it never targeted. How little is
    "little enough" was a constant -- 0.5 dB on the probes and 0.2 dB across the band --
    with no derivation behind either, and margins of 3452x and 8x that made the first a
    formality and the second nearly one.

    Neither is replaced by a criterion, because neither admits a valid null. The control
    here is the whole transform displaced by a quarter of the comb spacing, and it is
    matched in size, width and window geometry -- but not in what it removes. Its targets
    land between harmonics where there is no line, so it subtracts almost nothing and
    therefore leaks almost nothing, while leakage from the real transform scales with the
    line power it took out. Counted against it the real transform "fails" at p=2e-16 on
    every recording, which says only that it removed something.

    No offset repairs that: a control that removes comparable power away from the lines
    cannot exist, because the power is only at the lines. So the control is reported beside
    the observation and nothing is decided from either. What the pair does show is the
    scale of the leakage -- both sit near 0.01 dB, four orders below the 0.2 dB that used
    to be the criterion.
    """
    observed_probe = np.abs(lr.probe_deviations_db(freqs, probe_before, probe_after, probe))
    control_probe = np.abs(lr.probe_deviations_db(freqs, probe_before, probe_control, probe))
    observed_nonline = np.abs(
        lr.nonline_change_db(freqs, data_before, data_after, targets, widths)
    )
    control_nonline = np.abs(
        lr.nonline_change_db(freqs, data_before, data_control, control_targets, control_widths)
    )
    return {
        # Four tones on one channel is four observations, and no test on four values can
        # reach 0.05 -- the best a sign test could return is 2^-4 = 0.0625. Reporting the
        # observation beside its control says what there is to say; inventing a criterion
        # that cannot fire is the defect this work removed, not a fix for it.
        "max_probe_deviation_db": float(np.max(observed_probe)),
        "control_probe_deviation_db": float(np.max(control_probe)),
        "max_nonline_change_db": float(np.max(observed_nonline)),
        "control_nonline_change_db": float(np.max(control_nonline)),
        "nonline_change_null_p": lr.paired_excess_p_value(observed_nonline, control_nonline),
    }


def matched_control_plan(plan: RunRemovalPlan, settings: RemovalSettings) -> RunRemovalPlan:
    """The same transform displaced to where no line is, as a null for the preservation checks.

    Same windows, same number of targets, same widths -- only the positions move, by a
    quarter of the comb spacing. That offset is not a free choice: harmonics sit at
    ``k * f0`` and the probe tones at ``(k + 0.5) * f0``, so the quarter point is the unique
    displacement equidistant from both, and a control target therefore lands where neither
    an artifact nor an injected signal is.

    What this buys is a threshold-free preservation check. A transform that leaves the
    probes and the untouched spectrum alone should do so no more and no less than an
    identical transform aimed somewhere harmless; asking whether it did needs a control of
    the same size, not a number in decibels chosen by hand.
    """
    offset_hz = settings.nominal_fundamental_hz / 4.0
    displaced = tuple(
        replace(
            window,
            targets_hz=tuple(target + offset_hz for target in window.targets_hz),
            aggregate_residual_targets_hz=tuple(
                target + offset_hz for target in window.aggregate_residual_targets_hz
            ),
            channel_residual_targets_hz=tuple(
                tuple(target + offset_hz for target in channel)
                for channel in window.channel_residual_targets_hz
            ),
        )
        for window in plan.windows
    )
    return replace(plan, windows=displaced)


def benchmark_run(
    vhdr: Path,
    settings: RemovalSettings,
    plan: RunRemovalPlan,
) -> dict:
    """Inject probes, remove the lines, and measure what came back."""
    import mne

    mne.set_log_level("ERROR")
    raw = read_bids_raw(vhdr)
    estimate = plan.model.whole_estimate
    isolated = sorted(
        {
            float(frequency)
            for window in plan.windows
            for frequency in window.estimate.isolated_hz
            if np.isfinite(frequency)
        }
    )
    targets = plan.all_targets_hz
    span_targets, span_widths = plan_target_spans(plan.windows)
    probe = lr.Probe()
    lr.check_probe_clearance(
        probe,
        targets,
    )

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    eeg_names = tuple(raw.ch_names[int(pick)] for pick in picks)
    times = raw.times
    waveform = probe.waveform(times)

    probe_only = mne.io.RawArray(
        waveform[None, :],
        mne.create_info(
            ["benchmark_probe"],
            sfreq=float(raw.info["sfreq"]),
            ch_types=["eeg"],
        ),
        verbose="ERROR",
    )
    benchmark_pick = int(picks[0])
    background_probe = mne.io.RawArray(
        raw.get_data(picks=[benchmark_pick]) + waveform[None, :],
        mne.pick_info(raw.info, [benchmark_pick], copy=True),
        verbose="ERROR",
    )

    # The one probe placed where the removal does act. Reported, never gated: signal at an
    # artifact frequency is not separable from the artifact, so a loss here is the method's
    # cost rather than a defect. Positions are read off this recording's own plan.
    in_band_hz = lr.in_band_probe_frequencies(targets)
    in_band_probe = mne.io.RawArray(
        lr.sinusoid_waveform(times, in_band_hz, probe.sinusoid_amplitude_v)[None, :],
        mne.create_info(
            ["in_band_probe"],
            sfreq=float(raw.info["sfreq"]),
            ch_types=["eeg"],
        ),
        verbose="ERROR",
    )

    # Broadband probe: what the transform costs a signal occupying the whole band, as
    # opposed to removed_band_fraction, which counts the widths the plan asked for. Four
    # channels, all carrying the first EEG channel's plan, so the figure covers that
    # channel's common and channel-local targets; focal targets on other channels are
    # counted separately and are NOT visible here.
    #
    # Seeded from a CRC of the recording name, not ``hash``: PYTHONHASHSEED salts str
    # hashing per process, so ``hash`` gave a different probe on every invocation and
    # measured_band_attenuation moved between benchmarks of identical data under identical
    # settings -- a difference neither settings_fingerprint nor _source_digest can see.
    broadband_channels = 4
    broadband_probe = mne.io.RawArray(
        np.random.default_rng(zlib.crc32(vhdr.stem.encode("utf-8"))).normal(
            scale=probe.sinusoid_amplitude_v,
            size=(broadband_channels, times.size),
        ),
        mne.create_info(
            [f"broadband_probe_{index}" for index in range(broadband_channels)],
            sfreq=float(raw.info["sfreq"]),
            ch_types=["eeg"] * broadband_channels,
        ),
        verbose="ERROR",
    )

    cleaned_continuous = clean_continuous_raw(raw.copy(), plan, settings)
    cleaned_bare = cleaned_continuous
    # The same transform aimed where no line is. Everything the preservation checks compare
    # against comes from here rather than from a decibel constant.
    control_plan = matched_control_plan(plan, settings)
    cleaned_control = clean_continuous_raw(raw.copy(), control_plan, settings)
    cleaned_probe_control = clean_continuous_raw(
        probe_only.copy(),
        control_plan,
        settings,
        eeg_plan_indices=(0,),
    )
    cleaned_background_probe = clean_continuous_raw(
        background_probe,
        plan,
        settings,
        eeg_plan_indices=(0,),
    )
    cleaned_probe = clean_continuous_raw(
        probe_only,
        plan,
        settings,
        eeg_plan_indices=(0,),
    )
    cleaned_in_band_probe = clean_continuous_raw(
        in_band_probe,
        plan,
        settings,
        eeg_plan_indices=(0,),
    )
    cleaned_broadband_probe = clean_continuous_raw(
        broadband_probe,
        plan,
        settings,
        eeg_plan_indices=(0,) * broadband_channels,
    )

    freqs, _, _ = run_spectrum(cleaned_bare, settings)
    _, probe_psd_before = _psd(probe_only, [0], settings)
    _, probe_psd_after = _psd(cleaned_probe, [0], settings)
    in_band_freqs, in_band_psd_before = _psd(in_band_probe, [0], settings)
    _, in_band_psd_after = _psd(cleaned_in_band_probe, [0], settings)
    broadband_picks = list(range(broadband_channels))
    broadband_freqs, broadband_psd_before = _psd(broadband_probe, broadband_picks, settings)
    _, broadband_psd_after = _psd(cleaned_broadband_probe, broadband_picks, settings)
    _, data_psd_before = _psd(raw, picks, settings)
    _, data_psd_after = _psd(cleaned_bare, picks, settings)
    _, data_psd_control = _psd(cleaned_control, picks, settings)
    _, probe_psd_control = _psd(cleaned_probe_control, [0], settings)
    control_targets, control_widths = plan_target_spans(control_plan.windows)

    recovered = lr.recover_probe(
        cleaned_background_probe.get_data(),
        cleaned_bare.get_data(picks=[benchmark_pick]),
    )
    boundaries = _plan_transition_boundaries(plan, raw.n_times)
    metrics = {
        **adaptive_suppression_metrics(raw, cleaned_continuous, plan, settings),
        **spatiotemporal_line_metrics(raw, cleaned_continuous, plan, settings),
        **continuous_refinement_metrics(plan, eeg_names),
        **_boundary_metrics(
            raw.get_data(picks=picks),
            cleaned_bare.get_data(picks=picks),
            boundaries,
        ),
        "min_probe_ratio": lr.probe_preservation(
            freqs, probe_psd_before, probe_psd_after, probe
        )["min_probe_ratio"],
        **_preservation_against_control(
            freqs,
            probe=probe,
            probe_before=probe_psd_before,
            probe_after=probe_psd_after,
            probe_control=probe_psd_control,
            data_before=data_psd_before,
            data_after=data_psd_after,
            data_control=data_psd_control,
            targets=span_targets,
            widths=span_widths,
            control_targets=control_targets,
            control_widths=control_widths,
        ),
        **adaptive_band_metrics(
            sampling_frequency_hz=float(raw.info["sfreq"]),
            plan=plan,
            settings=settings,
        ),
        **lr.probe_recovery(recovered, cleaned_probe.get_data()[0], times, probe),
        **lr.in_band_probe_survival(
            in_band_freqs,
            in_band_psd_before,
            in_band_psd_after,
            in_band_hz,
        ),
        "in_band_probe_hz": ";".join(f"{frequency:.4f}" for frequency in in_band_hz),
        **lr.measured_band_attenuation(
            broadband_freqs,
            hd.to_db(broadband_psd_before),
            hd.to_db(broadband_psd_after),
        ),
    }
    verdict = lr.PreservationGate().evaluate(metrics)
    return {
        "recording": vhdr.stem,
        "fundamental_hz": estimate.fundamental_hz,
        "n_adaptive_windows": len(plan.windows),
        "fundamental_range_hz": plan.model.fundamental_range_hz,
        "max_adjacent_shift_hz": plan.model.max_adjacent_shift_hz,
        "max_window_standard_error_hz": max(
            window.estimate.fundamental_jackknife_se_hz for window in plan.windows
        ),
        "n_harmonics": estimate.n_harmonics,
        "median_targets_per_window": float(
            np.median([len(window.targets_hz) for window in plan.windows])
        ),
        "n_adjacent_sources": len(plan.all_narrow_targets_hz),
        "n_isolated_sources": len(isolated),
        "isolated_hz": ";".join(f"{frequency:.4f}" for frequency in isolated),
        "adjacent_hz": ";".join(f"{frequency:.4f}" for frequency in plan.all_narrow_targets_hz),
        **metrics,
        **{f"gate_{name}": value for name, value in verdict.items()},
        "gate_passed": all(verdict.values()),
    }


def _psd(raw, picks, settings: RemovalSettings):
    sfreq = float(raw.info["sfreq"])
    block = estimation_window_samples(sfreq, settings)
    data = raw.get_data(picks=picks)
    n_blocks = data.shape[-1] // block
    blocks = data[..., : n_blocks * block].reshape(data.shape[0], n_blocks, block)
    freqs, psd = hd.hann_periodogram(blocks, sfreq)
    return freqs, psd.mean(axis=1)


@dataclass(frozen=True)
class _LineObservation:
    run_index: int
    position_hz: float
    prominence_db: float
    bounds: tuple[int, int] | None


@dataclass
class _LineCluster:
    observations: list[_LineObservation] = field(default_factory=list)

    @property
    def centre_hz(self) -> float:
        per_run = []
        for run_index in sorted({item.run_index for item in self.observations}):
            positions = [
                item.position_hz for item in self.observations if item.run_index == run_index
            ]
            per_run.append(float(np.median(positions)))
        return float(np.median(per_run))


def _line_observations(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[tuple[_LineObservation, ...], tuple[float, ...]]:
    observations = []
    fundamentals = []
    for run_index, run in enumerate(spectra):
        whole_scaffold = _detection_scaffold(*run.whole, settings)
        window_scaffolds = tuple(
            _detection_scaffold(*spectrum, settings) for spectrum in run.windows
        )
        sources = (
            (run.whole, None, whole_scaffold),
            *(
                (spectrum, bounds, scaffold)
                for spectrum, bounds, scaffold in zip(
                    run.windows,
                    run.bounds,
                    window_scaffolds,
                )
            ),
        )
        for (freqs, spectrum_db, prominence), bounds, scaffold in sources:
            frequency_array = np.asarray(freqs, dtype=float)
            prominence_array = np.asarray(prominence, dtype=float)
            fundamentals.append(scaffold.fundamental_hz)
            positions = lr.detect_isolated_lines(
                freqs,
                spectrum_db,
                prominence,
                fundamental_hz=scaffold.fundamental_hz,
                harmonic_range=settings.removal_harmonic_range,
                min_prominence_db=settings.detection_candidate_prominence_db,
                low_hz=settings.detection_low_hz,
                high_hz=settings.detection_high_hz,
                comb_clearance_hz=lr.RESIDUAL_SEARCH_HZ,
                excluded_bands_hz=settings.protected_bands_hz,
            )
            for position in positions:
                index = int(np.argmin(np.abs(frequency_array - position)))
                observations.append(
                    _LineObservation(
                        run_index=run_index,
                        position_hz=float(position),
                        prominence_db=float(prominence_array[index]),
                        bounds=bounds,
                    )
                )
    return tuple(observations), tuple(fundamentals)


def _cluster_line_observations(
    observations: Sequence[_LineObservation],
) -> tuple[_LineCluster, ...]:
    clusters: list[_LineCluster] = []
    for observation in sorted(observations, key=lambda item: item.position_hz):
        eligible = [
            cluster
            for cluster in clusters
            if abs(observation.position_hz - cluster.centre_hz) <= lr._LINE_CLAIM_HZ
            and all(
                (item.run_index, item.bounds) != (observation.run_index, observation.bounds)
                for item in cluster.observations
            )
        ]
        if eligible:
            nearest = min(
                eligible,
                key=lambda cluster: abs(observation.position_hz - cluster.centre_hz),
            )
            nearest.observations.append(observation)
        else:
            clusters.append(_LineCluster([observation]))
    return tuple(clusters)


def _distinct_source_count(positions_hz: Sequence[float], resolution_hz: float) -> int:
    """Count only sources separable by the sinusoid-fit frequency grid."""
    distinct = []
    for position_hz in sorted(positions_hz):
        if distinct and position_hz - distinct[-1] <= resolution_hz:
            continue
        distinct.append(position_hz)
    return len(distinct)


def _intervals_overlap(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _best_overlapping_comb_estimate(
    evidence_bounds: tuple[int, int],
    window_bounds: tuple[tuple[int, int], ...],
    window_estimates: tuple[lr.CombEstimate, ...],
) -> lr.CombEstimate:
    """Most strongly supported continuous comb model covering an evidence interval."""
    if len(window_bounds) != len(window_estimates):
        raise ValueError("Every continuous window requires one validated comb estimate.")
    candidates = []
    for index, (bounds, estimate) in enumerate(zip(window_bounds, window_estimates)):
        overlap_samples = min(evidence_bounds[1], bounds[1]) - max(evidence_bounds[0], bounds[0])
        if overlap_samples > 0:
            candidates.append((overlap_samples, estimate, index))
    if not candidates:
        raise ValueError("Study evidence overlaps no validated continuous comb estimate.")
    _, estimate, _ = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1].n_harmonics,
            -item[1].residual_rms_hz,
            -item[1].fundamental_jackknife_se_hz,
            -item[2],
        ),
    )
    return estimate


def _comb_adjacent_observations(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    """Comb-adjacent summits from every block and epoch spectrum, tagged by recording.

    The mirror of :func:`_line_observations`, which collects the summits that clear the
    comb by ``RESIDUAL_SEARCH_HZ``. These are the ones that do not: close enough to a
    validated harmonic to be part of the same source, far enough out that the harmonic's
    own notch never reaches them. Between the two, every narrow summit in the band is
    observed exactly once.
    """
    observations = []
    for run_index, run in enumerate(spectra):
        window_estimates = tuple(
            _detection_scaffold(*spectrum, settings) for spectrum in run.windows
        )
        sources = (
            *(
                (spectrum, bounds, estimate)
                for spectrum, bounds, estimate in zip(
                    run.windows,
                    run.bounds,
                    window_estimates,
                )
            ),
        )
        for (freqs, spectrum_db, prominence), bounds, estimate in sources:
            frequency_array = np.asarray(freqs, dtype=float)
            prominence_array = np.asarray(prominence, dtype=float)
            positions = detect_comb_adjacent_lines(
                freqs,
                spectrum_db,
                prominence,
                estimate=estimate,
                settings=settings,
            )
            for position_hz in positions:
                index = int(np.argmin(np.abs(frequency_array - position_hz)))
                observations.append(
                    _LineObservation(
                        run_index=run_index,
                        position_hz=position_hz,
                        prominence_db=float(prominence_array[index]),
                        bounds=bounds,
                    )
                )
    return tuple(observations)


def _comb_adjacent_support(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[tuple[tuple[float, tuple[tuple[int, int], ...]], ...], ...]:
    """Comb-adjacent positions that clear the replication rules, with each run's support.

    Adjacency to a validated harmonic narrows *where* a false positive can land; it does
    not supply replication, and these summits are read off the 54 s block and 20 s epoch
    spectra that config.yaml singles out as making "many more comparisons than a whole-run
    scan", where "replication alone is insufficient" and ordinary blocks threw recurrent
    10-13 dB maxima around 39-45 Hz. This path used to apply neither rule and was also
    fitted one recording at a time, so a single 10 dB block summit became a target in every
    overlapping window while a line replicated across three recordings earned no credit for
    it. Both follow from the same omission: it never asked the session.

    So the clusters are formed over the whole session and admitted by the routes an
    isolated line already has to pass -- cross-run block replication first, then the
    single-recording route. No floor here that is not already configured for those.
    """
    clusters = _cluster_line_observations(_comb_adjacent_observations(spectra, settings))
    per_run: list[list[tuple[float, tuple[tuple[int, int], ...]]]] = [[] for _ in spectra]
    for cluster in clusters:
        # No _clears_every_comb_grid test: sitting beside the grid is what defines this
        # set, and detect_comb_adjacent_lines has already refused anything a harmonic's
        # own notch covers.
        supported = _block_line_support(cluster, settings)
        if not supported:
            supported = tuple(
                observation
                for run_index in range(len(spectra))
                for observation in _single_run_block_support(cluster, run_index, settings)
            )
        if not supported:
            continue
        position_hz = _run_balanced_position(supported)
        for run_index in range(len(spectra)):
            support_bounds = tuple(
                observation.bounds
                for observation in supported
                if observation.run_index == run_index and observation.bounds is not None
            )
            if support_bounds:
                per_run[run_index].append((position_hz, support_bounds))
    return tuple(tuple(item) for item in per_run)


def _comb_adjacent_window_targets(
    run: SessionRunSpectra,
    support: Sequence[tuple[float, tuple[tuple[int, int], ...]]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Route each supported comb-adjacent position to the windows that evidenced it."""
    window_targets: list[list[float]] = [[] for _ in run.windows]
    positions = []
    for position_hz, support_bounds in support:
        positions.append(position_hz)
        for index, filter_bounds in enumerate(run.bounds):
            if any(
                _intervals_overlap(filter_bounds, source_bounds) for source_bounds in support_bounds
            ):
                window_targets[index].append(position_hz)
    return (
        tuple(tuple(sorted(set(targets))) for targets in window_targets),
        tuple(sorted(positions)),
    )


def _independent_window_count(observations: Sequence[_LineObservation]) -> int:
    count = 0
    run_indices = sorted({item.run_index for item in observations})
    for run_index in run_indices:
        bounds = sorted(
            (
                item.bounds
                for item in observations
                if item.run_index == run_index and item.bounds is not None
            ),
            key=lambda interval: interval[1],
        )
        previous_stop = -1
        for start, stop in bounds:
            if start >= previous_stop:
                count += 1
                previous_stop = stop
    return count


def _run_balanced_position(observations: Sequence[_LineObservation]) -> float:
    per_run = []
    for run_index in sorted({item.run_index for item in observations}):
        positions = [item.position_hz for item in observations if item.run_index == run_index]
        per_run.append(float(np.median(positions)))
    return float(np.median(per_run))


def _whole_line_support(
    cluster: _LineCluster,
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    whole_observations = tuple(item for item in cluster.observations if item.bounds is None)
    whole_runs = {item.run_index for item in whole_observations}
    if (
        len(whole_runs) >= settings.min_runs_per_line
        and max(item.prominence_db for item in whole_observations)
        >= settings.detection_min_prominence_db
    ):
        return whole_observations
    return ()


def _block_line_support(
    cluster: _LineCluster,
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    block_observations = tuple(item for item in cluster.observations if item.bounds is not None)
    run_indices = sorted({item.run_index for item in block_observations})
    strong_runs = {
        run_index
        for run_index in run_indices
        if max(item.prominence_db for item in block_observations if item.run_index == run_index)
        >= settings.detection_block_min_prominence_db
    }
    supported = tuple(item for item in block_observations if item.run_index in strong_runs)
    if (
        len(strong_runs) >= settings.min_runs_per_block_line
        and _independent_window_count(supported) >= settings.min_independent_windows_per_line
    ):
        return supported
    return ()


def _single_run_block_support(
    cluster: _LineCluster,
    run_index: int,
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    """Evidence for a line independently repeated within one recording."""
    observations = tuple(
        item
        for item in cluster.observations
        if item.run_index == run_index and item.bounds is not None
    )
    if not observations:
        return ()
    if (
        max(item.prominence_db for item in observations)
        < settings.detection_block_min_prominence_db
    ):
        return ()
    if _independent_window_count(observations) < settings.min_independent_windows_per_line:
        return ()
    return observations


def _supported_line_position(
    cluster: _LineCluster,
    settings: RemovalSettings,
) -> float | None:
    observations = _whole_line_support(cluster, settings) or _block_line_support(cluster, settings)
    if not observations:
        return None
    return _run_balanced_position(observations)


def _clears_every_comb_grid(
    nominal_hz: float,
    fundamentals_hz: Sequence[float],
    settings: RemovalSettings,
) -> bool:
    clearance = settings.detection_search_hz
    return all(
        abs(nominal_hz - round(nominal_hz / fundamental) * fundamental) > clearance
        for fundamental in fundamentals_hz
    )


def _session_supported_positions(
    clusters: Sequence[_LineCluster],
    fundamentals_hz: Sequence[float],
    settings: RemovalSettings,
) -> dict[int, float]:
    """Cluster positions supported across recordings and clear of every comb grid."""
    return {
        cluster_index: position
        for cluster_index, cluster in enumerate(clusters)
        if (position := _supported_line_position(cluster, settings)) is not None
        and _clears_every_comb_grid(position, fundamentals_hz, settings)
    }


def session_nominals(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[float, ...]:
    """Session lines supported by run replication or independent block replication."""
    if len(spectra) < settings.min_runs_per_line:
        raise ValueError(
            "Session isolated-line detection requires at least "
            f"{settings.min_runs_per_line} runs, got {len(spectra)}."
        )

    observations, fundamentals = _line_observations(spectra, settings)
    clusters = _cluster_line_observations(observations)
    supported_positions = tuple(
        _session_supported_positions(clusters, fundamentals, settings).values()
    )
    return tuple(sorted(supported_positions))


def automatic_line_plans(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[RunIsolatedLinePlan, ...]:
    """Resolve session-replicated and recording-specific artifact lines automatically.

    Session evidence supplies one stable nominal to every run. A line confined to one run
    is accepted only after it appears in at least three non-overlapping 54-second windows
    and one occurrence reaches 15 dB prominence. Those targets stay confined to the exact
    windows that detected them; absence in another run can therefore never authorize a
    notch there.

    Fewer recordings than ``min_runs_per_line`` is not an error. Cross-recording
    replication is simply unavailable, so only the single-recording route can fire -- and
    that route is the stricter of the two, wanting 15 dB and three non-overlapping windows
    where the cross-recording route wants 10 dB in three recordings. A session of one
    continuous acquisition, which is the usual shape of resting or baseline data, is
    therefore planned under a higher bar rather than refused.
    """
    observations, fundamentals = _line_observations(spectra, settings)
    clusters = _cluster_line_observations(observations)
    session_positions = _session_supported_positions(clusters, fundamentals, settings)
    comb_adjacent_support = _comb_adjacent_support(spectra, settings)

    plans = []
    for run_index, run_spectra in enumerate(spectra):
        narrow_window_hz, narrow_positions = _comb_adjacent_window_targets(
            run_spectra,
            comb_adjacent_support[run_index],
        )
        whole_hz = []
        window_hz = [[] for _ in run_spectra.windows]
        routed_session_positions = []
        for cluster_index, position_hz in session_positions.items():
            cluster = clusters[cluster_index]
            if any(
                observation.run_index == run_index and observation.bounds is None
                for observation in cluster.observations
            ):
                whole_hz.append(position_hz)
            support_bounds = tuple(
                observation.bounds
                for observation in cluster.observations
                if observation.run_index == run_index and observation.bounds is not None
            )
            if not support_bounds:
                continue
            routed_session_positions.append(position_hz)
            for index, filter_bounds in enumerate(run_spectra.bounds):
                if any(
                    _intervals_overlap(filter_bounds, source_bounds)
                    for source_bounds in support_bounds
                ):
                    window_hz[index].append(position_hz)
        local_support = {
            cluster_index: support
            for cluster_index, cluster in enumerate(clusters)
            if cluster_index not in session_positions
            and (support := _single_run_block_support(cluster, run_index, settings))
        }
        source_positions = (
            *routed_session_positions,
            *(_run_balanced_position(support) for support in local_support.values()),
            *narrow_positions,
        )
        source_count = _distinct_source_count(
            source_positions,
            spectrum_fit_nominal_resolution_hz(settings.filter_length),
        )
        for cluster_index in local_support:
            cluster = clusters[cluster_index]
            supported = local_support[cluster_index]
            position_hz = _run_balanced_position(supported)
            whole_hz.extend(
                item.position_hz
                for item in cluster.observations
                if item.run_index == run_index and item.bounds is None
            )
            support_bounds = tuple(
                observation.bounds for observation in supported if observation.bounds is not None
            )
            for index, filter_bounds in enumerate(run_spectra.bounds):
                if any(
                    _intervals_overlap(filter_bounds, source_bounds)
                    for source_bounds in support_bounds
                ):
                    window_hz[index].append(position_hz)

        plans.append(
            RunIsolatedLinePlan(
                whole_hz=tuple(sorted(set(whole_hz))),
                window_hz=tuple(tuple(sorted(set(values))) for values in window_hz),
                narrow_window_hz=narrow_window_hz,
                source_count=source_count,
            )
        )
    return tuple(plans)


def apply_run(
    vhdr: Path,
    output_root: Path,
    bids_root: Path,
    settings: RemovalSettings,
    plan: RunRemovalPlan,
):
    """Apply the exact per-run plan that passed the benchmark."""
    import mne

    mne.set_log_level("ERROR")
    raw = read_bids_raw(vhdr)
    estimate = plan.model.whole_estimate
    cleaned = clean_continuous_raw(raw.copy(), plan, settings)

    destination = output_root / vhdr.relative_to(bids_root).with_suffix(".eeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_eeg_binary(vhdr, destination, cleaned.get_data())

    verify = read_bids_raw(output_root / vhdr.relative_to(bids_root))
    expected = cleaned.get_data()
    deviation = float(np.max(np.abs(verify.get_data() - expected)))
    scale = float(np.max(np.abs(expected)))
    # The binary is float32, so a round trip loses about 2^-24 of full scale. Anything
    # beyond a decade above that is corruption, not quantisation.
    tolerance = ROUNDTRIP_RELATIVE_TOLERANCE * scale
    if deviation > tolerance:
        raise RuntimeError(
            f"{vhdr.name}: written data differs by {deviation:.3e} V, "
            f"above the {tolerance:.3e} V float32 round-trip tolerance."
        )

    suppression = adaptive_suppression_metrics(raw, cleaned, plan, settings)
    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    eeg_names = tuple(raw.ch_names[int(pick)] for pick in picks)
    boundaries = _plan_transition_boundaries(plan, raw.n_times)
    isolated = sorted(
        {
            float(frequency)
            for window in plan.windows
            for frequency in window.estimate.isolated_hz
            if np.isfinite(frequency)
        }
    )
    return {
        **adaptive_band_metrics(
            sampling_frequency_hz=float(raw.info["sfreq"]),
            plan=plan,
            settings=settings,
        ),
        "recording": vhdr.stem,
        "fundamental_hz": estimate.fundamental_hz,
        "n_adaptive_windows": len(plan.windows),
        "fundamental_range_hz": plan.model.fundamental_range_hz,
        "max_adjacent_shift_hz": plan.model.max_adjacent_shift_hz,
        "max_window_standard_error_hz": max(
            window.estimate.fundamental_jackknife_se_hz for window in plan.windows
        ),
        "residual_rms_hz": estimate.residual_rms_hz,
        "n_harmonics": estimate.n_harmonics,
        "median_targets_per_window": float(
            np.median([len(window.targets_hz) for window in plan.windows])
        ),
        "isolated_hz": ";".join(f"{frequency:.4f}" for frequency in isolated),
        "adjacent_hz": ";".join(f"{frequency:.4f}" for frequency in plan.all_narrow_targets_hz),
        "n_adjacent_sources": len(plan.all_narrow_targets_hz),
        **suppression,
        **spatiotemporal_line_metrics(raw, cleaned, plan, settings),
        **continuous_refinement_metrics(plan, eeg_names),
        **_boundary_metrics(
            raw.get_data(picks=picks),
            cleaned.get_data(picks=picks),
            boundaries,
        ),
        "roundtrip_max_deviation_v": deviation,
        "roundtrip_relative": deviation / scale if scale else 0.0,
    }


def record_manifest_provenance(
    metrics: dict,
    *,
    input_digest: str,
    plan_digest: str,
    fingerprint: str,
) -> dict:
    """Attach the identities needed to trace one applied transform."""
    return {
        **metrics,
        "input_digest": input_digest,
        "plan_digest": plan_digest,
        "settings_fingerprint": fingerprint,
    }


def verify_cohort(bids_root: Path, cleaned_root: Path, settings: RemovalSettings, runs):
    """Run the diagnosis's own line detector over cleaned and original data alike.

    The manifest reports each run against its own targets. This asks the question the
    diagnosis asked: sweeping the whole band with FDR control and no knowledge of where
    the lines were, what is still detectable?
    """
    import mne

    mne.set_log_level("ERROR")
    from studies.pain_study.scripts.line_comb import diagnose as ds

    plans = build_run_plans(list(runs), settings)
    spectra = {"original": {}, "cleaned": {}}
    targeted = {"original": [], "cleaned": []}
    for vhdr in runs:
        subject = vhdr.parent.parent.name
        original = read_bids_raw(vhdr)
        cleaned = read_bids_raw(cleaned_root / vhdr.relative_to(bids_root))
        for label, raw in (("original", original), ("cleaned", cleaned)):
            freqs, spectrum_db, _ = run_spectrum(raw, settings)
            spectra[label].setdefault(subject, []).append(10 ** (spectrum_db / 10.0))
        targeted["original"].append(
            spatiotemporal_line_metrics(original, original, plans[vhdr.stem], settings)
        )
        targeted["cleaned"].append(
            spatiotemporal_line_metrics(original, cleaned, plans[vhdr.stem], settings)
        )

    grids = {
        label: ds.build_grid(
            freqs,
            np.stack([np.median(by_run[subject], axis=0) for subject in sorted(by_run)]),
        )
        for label, by_run in spectra.items()
    }
    report = []
    for label, grid in grids.items():
        try:
            lines = ds.detect_cohort_lines(grid)
        except ds.NoLinesDetected:
            # A clean stage. Anything else -- no usable window, no usable background --
            # is the analysis failing and must not be written here as zero lines.
            summary = {"n_lines": 0, "n_comb_lines": 0, "max_prominence_db": float("nan")}
            report.append({"stage": label, **summary})
            continue
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        report.append(
            {
                "stage": label,
                "n_lines": int(len(classified)),
                "n_comb_lines": int(classified.kind.isin(("comb", "comb_wide")).sum()),
                "n_isolated": int((classified.kind == "isolated").sum()),
                "max_prominence_db": float(classified.cohort_median_prominence_db.max()),
                "median_prominence_db": float(classified.cohort_median_prominence_db.median()),
            }
        )
    frame = pd.DataFrame(report)
    for label, rows in targeted.items():
        maximum = max(row["max_channel_block_residual_prominence_db"] for row in rows)
        maximum_excess = max(row["focal_residual_excess_db"] for row in rows)
        count = focal_residual_discoveries(rows)
        selected = frame.stage == label
        frame.loc[selected, "max_channel_block_target_db"] = maximum
        frame.loc[selected, "max_focal_residual_excess_db"] = maximum_excess
        frame.loc[selected, "n_runs_with_focal_residual"] = count
    cleaned = frame.stage == "cleaned"
    frame.loc[cleaned, "verification_passed"] = (frame.loc[cleaned, "n_comb_lines"] == 0) & (
        frame.loc[cleaned, "n_runs_with_focal_residual"] == 0
    )
    return frame, grids


def focal_residual_discoveries(rows: Sequence[dict]) -> int:
    """Recordings whose focal residual exceeds what their own matched controls reach.

    The same calibrated verdict ``benchmark`` prints and ``apply`` refuses on, rather than
    a second rule of verification's own. This used to read a 1.0 dB cushion on
    ``PreservationGate``; the rewrite that replaced the decibel thresholds with matched
    controls deleted the attribute and left the call, so ``verify`` raised on every
    dataset -- unnoticed, because it only ever runs after an ``apply`` that was refusing.

    A missing ``focal_residual_null_p`` raises rather than counting as clean: an absent
    measurement reported as a pass is the one direction a verification must not fail in.
    """
    return int(
        lr.residual_randomization_verdict([row["focal_residual_null_p"] for row in rows])[
            "n_discoveries"
        ]
    )


def discover_runs(
    bids_root: Path,
    subjects: list[str] | None,
    task: str = TASK,
) -> list[Path]:
    """Every recording of ``task`` under a BIDS root, with or without run and session.

    The ``run-`` entity is optional because BIDS omits it when a task was acquired once,
    which is the normal shape of a resting or baseline acquisition. Sessions are searched
    too, so ``sub-*/ses-*/eeg/`` datasets are found without a second call.
    """
    # ``_*eeg.vhdr`` rather than ``_*_eeg.vhdr``: with no run entity the name ends
    # ``_task-<task>_eeg.vhdr``, with nothing at all between the task and the suffix.
    patterns = (
        f"sub-*/eeg/sub-*_task-{task}_*eeg.vhdr",
        f"sub-*/ses-*/eeg/sub-*_task-{task}_*eeg.vhdr",
    )
    paths = sorted({path for pattern in patterns for path in bids_root.glob(pattern)})
    if subjects:
        wanted = set(subjects)
        paths = [path for path in paths if _subject_of(path) in wanted]
    if not paths:
        raise FileNotFoundError(
            f"No recordings of task {task!r} found under {bids_root}. Set "
            "`dataset.task` in the workflow config to the BIDS task label to process."
        )
    return paths


def _subject_of(path: Path) -> str:
    """The ``sub-*`` directory owning a recording, whether or not a session sits between."""
    for parent in path.parents:
        if parent.name.startswith("sub-"):
            return parent.name
    raise ValueError(f"{path} does not lie under a BIDS subject directory.")


def _write_tsv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete table or leave the previous table untouched."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, sep="\t", index=False, float_format="%.6g")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("benchmark", "apply", "verify"), default="benchmark")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--filter-length", default=None)
    parser.add_argument("--mt-bandwidth", type=float, default=None)
    args = parser.parse_args(argv)

    run(args)


def run(args: argparse.Namespace) -> None:
    """Execute one stage. Split from ``main`` so the CLI command can call it with its own args."""
    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    args.bids_root = config.path("bids_root", override=args.bids_root)
    args.output_root = config.path("output_root", override=args.output_root)
    args.report_dir = config.path("removal_dir", override=args.report_dir)

    settings = RemovalSettings.from_config(config)
    overrides = {}
    if args.filter_length is not None:
        overrides["filter_length"] = args.filter_length
    if args.mt_bandwidth is not None:
        overrides["mt_bandwidth"] = args.mt_bandwidth
    if overrides:
        settings = replace(settings, **overrides)
    print(f"settings: {settings}")

    runs = discover_runs(args.bids_root, subjects=None, task=settings.task)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "verify":
        print(f"Verifying all {len(runs)} runs")
        report, grids = verify_cohort(args.bids_root, args.output_root, settings, runs)
        _write_tsv_atomic(report, args.report_dir / "verification.tsv")
        print(report.to_string(index=False))
        np.savez_compressed(
            args.report_dir / "verification_spectra.npz",
            freqs=grids["original"].freqs,
            original=grids["original"].subject_psd,
            cleaned=grids["cleaned"].subject_psd,
            subjects=np.array(sorted({p.parent.parent.name for p in runs})),
        )
        print(f"  wrote {args.report_dir/'verification.tsv'}")
        return

    if args.stage == "benchmark":
        print(f"Fitting immutable per-run plans for all {len(runs)} recordings")
        plans = build_run_plans(runs, settings)
        input_digests = {vhdr.stem: recording_digest(vhdr) for vhdr in runs}
        plan_digests = {recording: removal_plan_digest(plan) for recording, plan in plans.items()}
        fingerprint = settings_fingerprint(settings)

        # Journal each recording as it completes. A raise anywhere in this loop used to cost
        # every recording already measured -- 23 of them to a probe collision, and very nearly
        # all 90 to an unconstructible null. Resuming is only safe because a journalled row is
        # reused solely when the settings, the recording's content and its fitted plan are all
        # unchanged; anything else describes different work and is measured again.
        partial_path = partial_benchmark_path(args.report_dir)
        completed = resumable_benchmark_rows(partial_path, fingerprint, input_digests, plan_digests)
        if completed:
            print(f"Resuming: {len(completed)} of {len(runs)} recordings already measured")

        rows = []
        for index, vhdr in enumerate(runs, start=1):
            started = time.time()
            if vhdr.stem in completed:
                rows.append(completed[vhdr.stem])
                print(f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} reused from partial benchmark")
                continue
            plan = plans[vhdr.stem]
            row = benchmark_run(vhdr, settings, plan)
            row["input_digest"] = input_digests[vhdr.stem]
            row["plan_digest"] = plan_digests[vhdr.stem]
            row["settings_fingerprint"] = fingerprint
            rows.append(row)
            _write_tsv_atomic(pd.DataFrame(rows), partial_path)
            print(
                f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
                f"f0={row['fundamental_hz']:.6f} suppress={row['median_suppression_db']:5.1f} dB "
                f"probe={row['max_probe_deviation_db']:.3f} dB burst={row['burst_energy_ratio']:.3f} "
                f"{'PASS' if row['gate_passed'] else 'FAIL'} ({time.time()-started:.0f}s)"
            )
        frame = pd.DataFrame(rows)
        frame["settings_fingerprint"] = fingerprint
        if set(frame.recording) != set(plans):
            raise RuntimeError("The benchmark does not contain exactly one result per plan.")
        _write_tsv_atomic(frame, args.report_dir / "benchmark.tsv")
        partial_path.unlink(missing_ok=True)
        gate_columns = [c for c in frame.columns if c.startswith("gate_") and c != "gate_passed"]
        print(f"\npassed {int(frame.gate_passed.sum())}/{len(frame)} runs")
        for column in gate_columns:
            print(f"  {column:32s} {int(frame[column].sum())}/{len(frame)}")
        seam = lr.seam_randomization_verdict(_seam_evidence_from_frame(frame))
        print(
            f"  {'seam (cohort criterion)':32s} "
            f"{'PASS' if seam['passed'] else 'FAIL'}: {int(seam['n_exceeding'])} exceeded "
            f"(count p={seam['count_p_value']:.4f}, maximum p="
            f"{seam['max_p_value']:.4f}), worst ratio {seam['max_ratio']:.2f}"
        )
        for label, column in (
            ("residual (cohort criterion)", "residual_null_p"),
            ("focal residual (cohort)", "focal_residual_null_p"),
        ):
            verdict = lr.residual_randomization_verdict(frame[column].to_numpy())
            print(
                f"  {label:32s} "
                f"{'PASS' if verdict['passed'] else 'FAIL'}: "
                f"{int(verdict['n_discoveries'])} of {int(verdict['n_runs'])} recordings "
                f"(smallest p={verdict['min_run_p_value']:.3g})"
            )
        print(
            f"  {'preservation (measurement)':32s} "
            f"probes {frame['max_probe_deviation_db'].max():.2g} dB against a control's "
            f"{frame['control_probe_deviation_db'].max():.2g}; "
            f"off-target band {frame['max_nonline_change_db'].max():.3f} dB against "
            f"{frame['control_nonline_change_db'].max():.3f}"
        )
        print(
            f"  {'band cost (measurement)':32s} "
            f"median {frame['measured_band_attenuated_1db'].median():.3f}, "
            f"worst {frame['measured_band_attenuated_1db'].max():.3f} of 28-95 Hz lost by a "
            f"broadband probe"
        )
        print(
            f"  {'in-band probe survival':32s} "
            f"median {frame['median_in_band_probe_survival'].median():.3f}, "
            f"worst {frame['min_in_band_probe_survival'].min():.3f} "
            f"(measurement, not a criterion)"
        )
        print(f"  wrote {args.report_dir/'benchmark.tsv'}")
        return

    print(f"Re-fitting all {len(runs)} plans before authorising apply")
    plans = build_run_plans(runs, settings)
    input_digests = {vhdr.stem: recording_digest(vhdr) for vhdr in runs}
    plan_digests = {recording: removal_plan_digest(plan) for recording, plan in plans.items()}
    require_passing_benchmark(
        args.report_dir / "benchmark.tsv",
        settings,
        recordings=input_digests,
        plans=plan_digests,
    )
    fingerprint = settings_fingerprint(settings)
    print(f"Benchmark {fingerprint} passed on all {len(runs)} recordings; applying.")

    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to mix a new derivative with existing output: {args.output_root}"
        )
    staging = args.output_root.with_name(f".{args.output_root.name}.staging-{fingerprint}")
    if staging.exists():
        raise FileExistsError(
            f"Incomplete staging output exists at {staging}; inspect it before retrying."
        )
    staging.mkdir(parents=True)
    print(f"Staging a complete derivative in {staging}")
    print(f"  copied {mirror_sidecars(args.bids_root, staging)} sidecars")

    rows = []
    for index, vhdr in enumerate(runs, start=1):
        started = time.time()
        metrics = apply_run(vhdr, staging, args.bids_root, settings, plans[vhdr.stem])
        rows.append(
            record_manifest_provenance(
                metrics,
                input_digest=input_digests[vhdr.stem],
                plan_digest=plan_digests[vhdr.stem],
                fingerprint=fingerprint,
            )
        )
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"suppress={rows[-1]['median_suppression_db']:5.1f} dB "
            f"max_resid={rows[-1]['max_residual_prominence_db']:6.2f} dB "
            f"({time.time()-started:.0f}s)"
        )
    frame = pd.DataFrame(rows)
    if set(frame.recording) != set(plans):
        raise RuntimeError("The staged derivative does not contain exactly one result per plan.")
    _write_tsv_atomic(frame, staging / "line_comb_removal_manifest.tsv")
    described = write_derivative_description(
        staging,
        args.bids_root,
        settings,
        dataset_digest(input_digests, args.bids_root),
        band_cost={
            "measured_band_attenuated_1db_median": float(
                frame["measured_band_attenuated_1db"].median()
            ),
            "measured_band_attenuated_1db_worst": float(
                frame["measured_band_attenuated_1db"].max()
            ),
        }
        if "measured_band_attenuated_1db" in frame
        else None,
    )
    import os

    os.replace(staging, args.output_root)
    _write_tsv_atomic(frame, args.report_dir / "removal_manifest.tsv")
    print(
        f"\nmedian suppression {frame.median_suppression_db.median():.1f} dB; "
        f"worst residual line {frame.max_residual_prominence_db.max():.2f} dB"
    )
    print(f"  declared {(args.output_root / described.name)} a derivative of {args.bids_root}")
    print(f"  wrote {args.report_dir/'removal_manifest.tsv'}")


if __name__ == "__main__":
    main()
