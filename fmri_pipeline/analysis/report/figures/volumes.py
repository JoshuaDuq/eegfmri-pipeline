"""Volume renderings of unsigned magnitude maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.report.figures._display import figure_of, label_colorbar
from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    MAGNITUDE_CMAP,
    OKABE_ITO,
    RADIOLOGICAL,
    annotate_provenance,
    clipped_fraction,
    orientation_label,
    plot_context,
    robust_upper_limit,
)


#: Smallest residual standard deviation, as a fraction of a voxel's own mean, that
#: still counts as a measurement rather than floating-point noise.
#:
#: Real BOLD tSNR tops out in the low hundreds, so a genuine voxel sits many orders
#: of magnitude above this. Nothing measurable is excluded by it.
_RESIDUAL_FLOOR_RATIO = 1e-9


@dataclass(frozen=True)
class TsnrResult:
    """Mean tSNR map plus the per-run detail an average would hide."""

    mean_img: nib.Nifti1Image
    per_run_median: Tuple[float, ...]
    frames_used: Tuple[int, ...]
    frames_dropped: Tuple[int, ...]
    #: ``(q1, q3)`` of the in-mask tSNR of each run. A median alone cannot separate a
    #: run that lost signal everywhere from one that lost it in a region: both move
    #: the centre, only the second widens the spread.
    per_run_iqr: Tuple[Tuple[float, float], ...] = ()


def detrended_temporal_sd(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Temporal standard deviation after removing low-order drift.

    Scanner drift is not thermal or physiological noise, and the GLM's cosine
    high-pass removes it before any statistic is computed. Leaving it in the temporal
    standard deviation therefore reports a tSNR lower than the one the model actually
    works with -- a quantitative error in a reported metric, not a display choice.

    A cubic polynomial basis captures the drift the high-pass removes without needing
    the run's exact cutoff frequency.
    """
    n_frames = data.shape[3]
    if n_frames < 4:
        # Fewer frames than basis functions: no drift estimate is possible, and
        # fitting one would consume the signal rather than the trend.
        return np.std(data, axis=3)

    time = np.linspace(-1.0, 1.0, n_frames, dtype=np.float64)
    basis = np.vstack([np.ones_like(time), time, time**2, time**3]).T

    series = data[mask].astype(np.float64).T  # (frames, voxels)
    if series.size == 0:
        return np.std(data, axis=3)

    # errstate: numpy on Accelerate BLAS raises spurious invalid/overflow flags from
    # matmul even for well-conditioned finite operands, so the flags carry no
    # information here. The finiteness check below is the real guard.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        beta, *_ = np.linalg.lstsq(basis, series, rcond=None)
        residual = series - basis @ beta

    if not np.all(np.isfinite(residual)):
        # A degenerate solve must not silently produce NaN standard deviations,
        # which would render as holes in the tSNR map indistinguishable from
        # genuinely unmeasurable voxels.
        return np.std(data, axis=3)

    out = np.zeros(data.shape[:3], dtype=np.float64)
    out[mask] = residual.std(axis=0)
    return out


def compute_tsnr(
    bold_imgs: Sequence[Any],
    *,
    mask_img: Any = None,
    sample_masks: Optional[Sequence[np.ndarray]] = None,
) -> TsnrResult:
    """Return mean temporal SNR across runs, and each run's median separately.

    ``sample_masks`` is one boolean array per run marking the frames to keep. Pass
    the same censoring the GLM used. Non-steady-state frames in particular must be
    excluded: they sit at much higher intensity before longitudinal magnetisation
    saturates, and leaving them in inflates the temporal standard deviation, biasing
    every tSNR value low. fMRIPrep flags them as ``non_steady_state_outlier_XX``.

    Per-run medians are returned alongside the mean map because averaging maps
    across runs makes a single bad run disappear, which is the one thing this
    measurement exists to catch.
    """
    if not bold_imgs:
        raise ValueError("compute_tsnr requires at least one BOLD image.")
    if sample_masks is not None and len(sample_masks) != len(bold_imgs):
        raise ValueError(
            f"Got {len(sample_masks)} sample masks for {len(bold_imgs)} runs."
        )

    mask = None
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)

    total: Optional[np.ndarray] = None
    affine = None
    medians: List[float] = []
    quartiles: List[Tuple[float, float]] = []
    used: List[int] = []
    dropped: List[int] = []

    for index, img in enumerate(bold_imgs):
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4:
            raise ValueError(f"compute_tsnr requires 4D images, got shape {data.shape}.")
        if affine is None:
            affine = img.affine

        n_frames = data.shape[3]
        if sample_masks is not None:
            keep = np.asarray(sample_masks[index], dtype=bool)
            if keep.size != n_frames:
                raise ValueError(
                    f"Run {index} has {n_frames} frames but its sample mask "
                    f"has {keep.size}."
                )
            data = data[..., keep]
        used.append(int(data.shape[3]))
        dropped.append(int(n_frames - data.shape[3]))
        if data.shape[3] < 2:
            raise ValueError(f"Run {index} has fewer than two frames after censoring.")

        mean = np.mean(data, axis=3)
        # Drift is removed before the standard deviation is taken; see
        # detrended_temporal_sd. Without a mask every voxel is detrended, which is
        # more work than necessary but never wrong.
        sd_mask = (
            mask
            if mask is not None and mask.shape == data.shape[:3]
            else np.ones(data.shape[:3], dtype=bool)
        )
        std = detrended_temporal_sd(data, sd_mask)
        # A voxel with no residual variance has undefined tSNR, not infinite tSNR.
        #
        # The comparison is relative, not `std > 0`. Detrending leaves floating-point
        # dust rather than exact zero where a voxel is fully explained by the drift
        # basis, and dividing by dust reports a tSNR of order 1e15 -- a number that
        # would dominate the colour limit and flatten the real map to nothing.
        floor = np.abs(mean) * _RESIDUAL_FLOOR_RATIO
        usable = std > floor
        tsnr = np.divide(mean, std, out=np.zeros_like(mean, dtype=float), where=usable)
        if mask is not None and mask.shape == tsnr.shape:
            tsnr = np.where(mask, tsnr, 0.0)

        inside = tsnr[tsnr > 0]
        medians.append(float(np.median(inside)) if inside.size else 0.0)
        quartiles.append(
            (float(np.percentile(inside, 25)), float(np.percentile(inside, 75)))
            if inside.size
            else (0.0, 0.0)
        )

        if total is None:
            total = tsnr.astype(float)
        elif total.shape == tsnr.shape:
            total += tsnr
        else:
            raise ValueError(f"Runs disagree on shape: {total.shape} vs {tsnr.shape}.")

    assert total is not None  # guarded by the empty check above
    return TsnrResult(
        mean_img=nib.Nifti1Image(
            (total / float(len(bold_imgs))).astype("float32"), affine
        ),
        per_run_median=tuple(medians),
        frames_used=tuple(used),
        frames_dropped=tuple(dropped),
        per_run_iqr=tuple(quartiles),
    )


def dropout_note(result: TsnrResult, *, mask_img: Any = None) -> str:
    """State how much of the mask sits in the low tail of its own tSNR.

    A tSNR map is read for where the measurement fails, and "fails" only means
    anything relative to the rest of this acquisition: absolute tSNR depends on field
    strength, voxel size, coil, and echo time, so a fixed cutoff would be a verdict
    invented here rather than a measurement. The quartile is a property of this map.

    Reported as a count and a share, with no judgement attached to either.
    """
    data = np.asarray(result.mean_img.get_fdata())
    inside = np.isfinite(data) & (data > 0)
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)
        if mask.shape == data.shape:
            inside &= mask
    values = data[inside]
    if values.size == 0:
        return "no voxels with measurable tSNR"
    quartile = float(np.percentile(values, 25))
    below = int(np.count_nonzero(values < quartile))
    return (
        f"lowest quartile of tSNR is below {quartile:.0f} "
        f"({below:,} of {values.size:,} voxels)"
    )


def tsnr_volume(
    result: TsnrResult,
    *,
    bg_img: Any = None,
    mask_img: Any = None,
    title: str = "",
    vmax: Optional[float] = None,
    radiological: bool = RADIOLOGICAL,
) -> Any:
    """Draw a tSNR map in anatomical orientation.

    Rendered through nilearn so the affine determines what "sagittal" means. Slicing
    the voxel array directly and labelling the panels by anatomy is correct only for
    RAS-canonical data and silently mislabels -- including left/right -- otherwise.

    The orientation convention is passed explicitly and stated on the figure, like
    every other volume panel: a left/right error leaves no trace in the image.

    Drawn through the shared magnitude path, which leaves non-positive voxels
    transparent. Rendered without that, the zeros outside the analysis mask took the
    low end of the ramp and painted a solid block across the field-of-view box --
    hiding the anatomy the background exists for, and reading as a tSNR of nearly
    zero where in fact nothing was measured.

    ``mask_img`` is the analysis mask, and it is what the colour limit is taken
    inside. Without it the panel fell back on excluding exact zeros and said
    ``no mask supplied`` in its own provenance line, while the median beside it had
    been computed from the mask all along -- two numbers on one figure describing
    different populations of voxels.
    """
    from fmri_pipeline.analysis.report.figures.stat_maps import magnitude_mosaic

    data = np.asarray(result.mean_img.get_fdata())
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        raise ValueError("A tSNR panel requires at least one positive voxel.")

    return magnitude_mosaic(
        result.mean_img,
        bg_img=bg_img,
        mask_img=mask_img,
        vmax=vmax,
        radiological=radiological,
        title=title,
        cbar_label="tSNR",
        extra_provenance=[
            f"median tSNR {float(np.median(positive)):.1f}",
            f"mean of {len(result.per_run_median)} run(s); "
            f"{sum(result.frames_dropped):,} frames censored",
            dropout_note(result, mask_img=mask_img),
            # Named so this tSNR can be compared against one computed elsewhere. Two
            # pipelines differing only in drift handling report visibly different
            # numbers for identical data.
            "cubic drift removed before the temporal SD",
        ],
    )


__all__ = [
    "TsnrResult",
    "compute_tsnr",
    "detrended_temporal_sd",
    "dropout_note",
    "tsnr_volume",
]
