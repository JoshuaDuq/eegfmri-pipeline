"""Acquired-lag autocorrelation of persisted fitted-model residuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

Quartiles = Tuple[float, float, float]
DEFAULT_MAX_LAG_FRAMES = 20
_VOXEL_CHUNK_SIZE = 2048


@dataclass(frozen=True)
class RunResidualAutocorrelation:
    """Voxelwise residual autocorrelation summarized for one fitted run."""

    label: str
    lags_frames: Tuple[int, ...]
    valid_pairs: Tuple[int, ...]
    quartiles: Tuple[Quartiles, ...]
    retained_frames: int
    acquired_frames: int
    voxel_count: int


def collect_residual_autocorrelation(
    *,
    run_labels: Sequence[str],
    residual_paths: Sequence[Path],
    retained_frame_indices: Sequence[Sequence[int]],
    acquired_frame_counts: Sequence[int],
    mask_path: Path,
    max_lag_frames: int = DEFAULT_MAX_LAG_FRAMES,
) -> Tuple[RunResidualAutocorrelation, ...]:
    """Measure residual ACF at exact acquired-frame lags for every run."""
    import nibabel as nib

    labels = tuple(str(label) for label in run_labels)
    paths = tuple(Path(path) for path in residual_paths)
    retained_by_run = tuple(tuple(indices) for indices in retained_frame_indices)
    acquired_by_run = tuple(int(count) for count in acquired_frame_counts)
    run_count = len(labels)
    if run_count == 0:
        raise ValueError("Residual autocorrelation requires at least one run.")
    if not (
        len(paths) == run_count
        and len(retained_by_run) == run_count
        and len(acquired_by_run) == run_count
    ):
        raise ValueError("Residual-autocorrelation run inputs must align.")
    if max_lag_frames < 1:
        raise ValueError(f"max_lag_frames must be positive, got {max_lag_frames}.")

    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError(f"The fitted analysis mask must be 3D, got {mask_image.shape}.")
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        raise ValueError("The fitted analysis mask contains no voxels.")

    runs = []
    for run_index, (label, path, retained, acquired_frames) in enumerate(
        zip(labels, paths, retained_by_run, acquired_by_run),
        start=1,
    ):
        image = nib.load(str(path))
        if len(image.shape) != 4 or image.shape[:3] != mask_image.shape:
            raise ValueError(f"{label} residual series do not match the fitted analysis mask.")
        if not np.allclose(image.affine, mask_image.affine):
            raise ValueError(f"{label} residual series do not share the fitted mask affine.")
        if max_lag_frames >= acquired_frames:
            raise ValueError(
                f"{label} has {acquired_frames} acquired frames, fewer than the requested "
                f"{max_lag_frames + 1}."
            )

        indices = np.asarray(retained, dtype=int)
        _validate_retained_indices(
            indices,
            residual_frames=int(image.shape[3]),
            acquired_frames=acquired_frames,
            label=label,
        )
        residual = np.asarray(image.dataobj, dtype=np.float32)[mask]
        if not np.isfinite(residual).all():
            raise ValueError(f"{label} residual series contain non-finite values.")

        lags = tuple(range(1, max_lag_frames + 1))
        pair_counts = _valid_pair_counts(indices, acquired_frames, lags)
        autocorrelation = _autocorrelation_by_voxel(
            residual,
            retained_indices=indices,
            acquired_frames=acquired_frames,
            max_lag_frames=max_lag_frames,
            label=label,
        )
        quartiles = np.percentile(
            autocorrelation,
            [25.0, 50.0, 75.0],
            axis=0,
        )
        if not np.isfinite(quartiles).all():
            raise ValueError(f"{label} residual-autocorrelation quartiles must be finite.")
        runs.append(
            RunResidualAutocorrelation(
                label=label,
                lags_frames=lags,
                valid_pairs=pair_counts,
                quartiles=tuple(
                    tuple(float(value) for value in quartiles[:, lag_index])
                    for lag_index in range(len(lags))
                ),
                retained_frames=int(indices.size),
                acquired_frames=acquired_frames,
                voxel_count=voxel_count,
            )
        )
    return tuple(runs)


def _validate_retained_indices(
    indices: np.ndarray,
    *,
    residual_frames: int,
    acquired_frames: int,
    label: str,
) -> None:
    """Require exact, increasing acquired-frame indices for one run."""
    if indices.ndim != 1 or indices.size != residual_frames:
        raise ValueError(f"{label} retained indices do not match residual timepoints.")
    if indices.size < 2:
        raise ValueError(f"{label} residual autocorrelation requires two retained frames.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError(f"{label} retained frame indices must increase strictly.")
    if indices[0] < 0 or indices[-1] >= acquired_frames:
        raise ValueError(f"{label} retained frame indices exceed acquired frames.")


def _valid_pair_counts(
    retained_indices: np.ndarray,
    acquired_frames: int,
    lags: Sequence[int],
) -> Tuple[int, ...]:
    """Count retained sample pairs separated by each acquired-frame lag."""
    retained = np.zeros(acquired_frames, dtype=bool)
    retained[retained_indices] = True
    counts = tuple(int(np.count_nonzero(retained[:-lag] & retained[lag:])) for lag in lags)
    if any(count == 0 for count in counts):
        raise ValueError("A requested residual-autocorrelation lag has no retained pairs.")
    return counts


def _autocorrelation_by_voxel(
    residual: np.ndarray,
    *,
    retained_indices: np.ndarray,
    acquired_frames: int,
    max_lag_frames: int,
    label: str,
) -> np.ndarray:
    """Compute acquired-lag autocorrelation in bounded-memory voxel chunks."""
    from scipy.fft import irfft, next_fast_len, rfft

    fft_length = next_fast_len(2 * acquired_frames - 1)
    autocorrelation = np.empty(
        (residual.shape[0], max_lag_frames),
        dtype=np.float64,
    )
    for start in range(0, residual.shape[0], _VOXEL_CHUNK_SIZE):
        stop = min(start + _VOXEL_CHUNK_SIZE, residual.shape[0])
        chunk = np.asarray(residual[start:stop], dtype=np.float64)
        centered = chunk - chunk.mean(axis=1, keepdims=True)
        denominator = np.einsum("ij,ij->i", centered, centered)
        if np.any(denominator <= 0):
            raise ValueError(f"{label} contains undefined residual autocorrelation.")

        acquired_axis = np.zeros((centered.shape[0], acquired_frames), dtype=np.float64)
        acquired_axis[:, retained_indices] = centered
        spectrum = rfft(acquired_axis, n=fft_length, axis=1)
        autocovariance = irfft(
            spectrum * np.conjugate(spectrum),
            n=fft_length,
            axis=1,
        )[:, 1 : max_lag_frames + 1]
        autocorrelation[start:stop] = autocovariance / denominator[:, np.newaxis]
    return autocorrelation


def residual_autocorrelation_figure(
    runs: Sequence[RunResidualAutocorrelation],
    *,
    tr: float,
    title: str = "",
):
    """Draw one median-and-IQR acquired-lag ACF panel per run."""
    import matplotlib.pyplot as plt

    from fmri_pipeline.analysis.report.style import (
        GUIDE_COLOR,
        OKABE_ITO,
        annotate_provenance,
        plot_context,
    )

    measured = _validated_runs(runs, tr=tr)
    column_count = min(3, len(measured))
    row_count = int(np.ceil(len(measured) / column_count))
    limits = np.asarray(
        [value for run in measured for quartile in run.quartiles for value in quartile],
        dtype=float,
    )
    lower = min(0.0, float(limits.min()))
    upper = max(0.0, float(limits.max()))
    padding = max(0.05, 0.08 * (upper - lower))

    with plot_context():
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(3.55 * column_count, 2.45 * row_count + 0.45),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for panel_index, (axis, run) in enumerate(zip(flat_axes, measured)):
            lag_seconds = np.asarray(run.lags_frames, dtype=float) * float(tr)
            quartiles = np.asarray(run.quartiles, dtype=float)
            axis.fill_between(
                lag_seconds,
                quartiles[:, 0],
                quartiles[:, 2],
                color=OKABE_ITO["sky_blue"],
                alpha=0.28,
                linewidth=0,
                label="Voxel IQR",
            )
            axis.plot(
                lag_seconds,
                quartiles[:, 1],
                color=OKABE_ITO["blue"],
                linewidth=1.4,
                label="Median",
            )
            axis.axhline(
                0.0,
                color=GUIDE_COLOR,
                linewidth=0.8,
                linestyle=":",
                label="_nolegend_",
            )
            axis.set_title(run.label, fontsize=9)
            axis.set_ylim(lower - padding, upper + padding)
            axis.set_xlabel("Lag (s)")
            if panel_index % column_count == 0:
                axis.set_ylabel("Residual autocorrelation")
            minimum_pairs = min(run.valid_pairs)
            maximum_pairs = max(run.valid_pairs)
            pair_range = (
                f"{minimum_pairs:,}"
                if minimum_pairs == maximum_pairs
                else f"{minimum_pairs:,}–{maximum_pairs:,}"
            )
            axis.text(
                0.02,
                0.04,
                f"{pair_range} retained pairs/voxel",
                transform=axis.transAxes,
                fontsize=6.5,
                color=GUIDE_COLOR,
                va="bottom",
            )

        for axis in flat_axes[len(measured) :]:
            figure.delaxes(axis)
        flat_axes[0].legend(loc="best", frameon=False, fontsize=7)
        if title:
            figure.suptitle(title)
        maximum_lag = max(run.lags_frames[-1] for run in measured)
        annotate_provenance(
            figure,
            [
                f"{measured[0].voxel_count:,} fitted-mask voxels per run",
                f"lags 1–{maximum_lag} acquired frames "
                f"({float(tr):.3g}–{maximum_lag * float(tr):.3g} s)",
                "line: voxel median · band: voxel IQR",
                "unwhitened model-response residuals",
                "zero line is a reference; no criterion is applied",
            ],
        )
        return figure


def write_residual_autocorrelation_tsv(
    runs: Sequence[RunResidualAutocorrelation],
    *,
    tr: float,
    path: Path,
) -> Path:
    """Write every plotted run-by-lag residual-ACF quantile."""
    measured = _validated_runs(runs, tr=tr)
    rows = [
        "\t".join(
            (
                "Run",
                "Lag (frames)",
                "Lag (s)",
                "Valid retained pairs per voxel",
                "ACF Q1",
                "ACF median",
                "ACF Q3",
            )
        )
    ]
    for run in measured:
        for lag, pair_count, quartiles in zip(
            run.lags_frames,
            run.valid_pairs,
            run.quartiles,
        ):
            q1, median, q3 = quartiles
            rows.append(
                f"{run.label}\t{lag}\t{lag * float(tr):.6f}\t{pair_count}\t"
                f"{q1:.6f}\t{median:.6f}\t{q3:.6f}"
            )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def _validated_runs(
    runs: Sequence[RunResidualAutocorrelation],
    *,
    tr: float,
) -> Tuple[RunResidualAutocorrelation, ...]:
    """Validate measurements shared by the figure and exact-data writer."""
    measured = tuple(runs)
    if not measured:
        raise ValueError("Residual-autocorrelation output requires at least one run.")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError(f"TR must be positive and finite, got {tr!r}.")
    expected_lags = measured[0].lags_frames
    expected_voxels = measured[0].voxel_count
    for run in measured:
        if not (len(run.lags_frames) == len(run.valid_pairs) == len(run.quartiles)):
            raise ValueError(f"{run.label} residual-autocorrelation values do not align.")
        if run.lags_frames != expected_lags:
            raise ValueError("Residual-autocorrelation runs must share displayed lags.")
        if run.voxel_count != expected_voxels:
            raise ValueError("Residual-autocorrelation runs must share a voxel count.")
    return measured


__all__ = [
    "DEFAULT_MAX_LAG_FRAMES",
    "RunResidualAutocorrelation",
    "collect_residual_autocorrelation",
    "residual_autocorrelation_figure",
    "write_residual_autocorrelation_tsv",
]
