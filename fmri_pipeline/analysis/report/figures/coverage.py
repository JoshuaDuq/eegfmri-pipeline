"""Analysis-mask coverage and spatial smoothness.

Coverage answers a question a stat map cannot: a voxel outside the analysis mask
was never tested, so its absence of effect is not evidence of absence. Signal
dropout in orbitofrontal and inferior temporal cortex looks exactly like a true
null on a thresholded map.

Smoothness makes a cluster-extent threshold interpretable. ``cluster_min_voxels``
is configured as a bare count, and the same count is a strong constraint on
unsmoothed data and almost none at 8 mm.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.figures._display import figure_of
from fmri_pipeline.analysis.report.style import (
    MAGNITUDE_CMAP,
    annotate_provenance,
    plot_context,
)

logger = logging.getLogger(__name__)

#: Converts a Gaussian standard deviation to its full width at half maximum.
_FWHM_PER_SIGMA = float(np.sqrt(8.0 * np.log(2.0)))


def estimate_fwhm(
    img: Any, *, mask: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Estimate spatial smoothness per axis, in millimetres.

    Uses the Kiebel/Forman estimator: for a Gaussian field, the variance of the
    finite difference along an axis relates to the field's smoothness, giving
    ``sigma = sqrt(-1 / (4 * ln(1 - Var(dX) / (2 * Var(X)))))`` in voxels, which the
    affine converts to millimetres.

    Estimated from whatever map is supplied. Given a statistic map rather than
    model residuals this overestimates smoothness wherever real signal is present,
    because signal is spatially structured; callers state which input was used.

    Validated against Gaussian-filtered noise on a 40^3 grid: estimates land within
    4% of the true FWHM at 2.35, 4.71, and 7.06 mm, and scale exactly with voxel
    size. Below about one voxel of smoothness the finite difference saturates and
    the estimate floors at the voxel dimension -- at that point the honest statement
    is "at or below one voxel", which is what the floor returns.
    """
    data = np.asarray(img.get_fdata(), dtype=float)
    if min(data.shape[:3]) < 4:
        raise ValueError(f"Volume is too small to estimate smoothness: {data.shape}.")

    if mask is None:
        mask = np.isfinite(data) & (data != 0)
    values = np.where(mask, data, np.nan)
    total_variance = np.nanvar(values)
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("Cannot estimate smoothness from a constant map.")

    voxel_sizes = np.sqrt((np.asarray(img.affine)[:3, :3] ** 2).sum(axis=0))
    fwhm = []
    for axis in range(3):
        difference = np.diff(values, axis=axis)
        diff_variance = np.nanvar(difference)
        ratio = diff_variance / (2.0 * total_variance)
        if not np.isfinite(ratio) or ratio <= 0 or ratio >= 1:
            # A ratio at or past 1 means neighbouring voxels are uncorrelated:
            # smoothness is at or below one voxel.
            fwhm.append(float(voxel_sizes[axis]))
            continue
        sigma_voxels = np.sqrt(-1.0 / (4.0 * np.log(1.0 - ratio)))
        fwhm.append(float(sigma_voxels * _FWHM_PER_SIGMA * voxel_sizes[axis]))
    return (fwhm[0], fwhm[1], fwhm[2])


def mask_volume_mm3(mask_img: Any) -> Tuple[int, float]:
    """Return the voxel count of a mask and the volume it occupies, in mm^3.

    The count alone is not comparable to anything: it depends on the resampling grid,
    so the same brain at 2 mm and at 3 mm differs threefold. Volume is the quantity a
    reader can carry to another subject, another study, or a published figure.
    """
    data = np.asarray(mask_img.get_fdata())
    voxels = int(np.count_nonzero(data > 0))
    voxel_sizes = np.sqrt((np.asarray(mask_img.affine)[:3, :3] ** 2).sum(axis=0))
    return voxels, float(voxels * float(np.prod(voxel_sizes)))


def coverage_figure(
    mask_img: Any,
    *,
    bg_img: Any = None,
    extent_note: str = "",
    smoothness_note: str = "",
    title: str = "",
) -> plt.Figure:
    """Draw the analysis mask over the background, with its extent stated.

    Rendered as an ROI overlay rather than a bare binary volume so a reader can see
    which anatomy fell outside the model, which is the only reading that matters --
    and that reading requires ``bg_img``. Over an empty background the panel shows a
    blob whose missing regions cannot be named.

    ``extent_note`` describes how the mask was derived, and is supplied by the caller
    rather than assumed here. This panel previously asserted "intersection across N
    runs" for whatever mask it was handed, which was false whenever the mask recorded
    was a single run's.
    """
    from nilearn import plotting

    voxels, volume = mask_volume_mm3(mask_img)
    if voxels == 0:
        raise ValueError(
            "Coverage figure requires a mask with at least one voxel; got no voxels."
        )

    with plot_context():
        display = plotting.plot_roi(
            mask_img,
            bg_img=bg_img,
            title=title or None,
            display_mode="ortho",
            cmap=MAGNITUDE_CMAP,
            alpha=0.55,
            black_bg=False,
            annotate=True,
        )
        figure = figure_of(display)
        # The old first line was "N voxels modelled of M in the field of view". M
        # counts air, so the ratio measured how much empty space the acquisition box
        # contained and told a reader nothing about coverage.
        lines = [f"{voxels:,} voxels modelled ({volume / 1000.0:,.1f} cm³)"]
        if extent_note:
            lines.append(extent_note)
        if smoothness_note:
            lines.append(smoothness_note)
        lines.append("voxels outside this mask were not tested")
        annotate_provenance(figure, lines)
        return figure


def smoothness_note(fwhm: Tuple[float, float, float], *, source: str) -> str:
    """One line describing an estimated smoothness and where it came from.

    The source matters enough to be mandatory. Estimated from a statistic map this
    overestimates smoothness wherever real signal sits, because signal is spatially
    structured; the same number from model residuals would not. A reader comparing
    this against a published FWHM needs to know which they are looking at.
    """
    x, y, z = fwhm
    return f"smoothness {x:.1f} × {y:.1f} × {z:.1f} mm FWHM (estimated from the {source})"


def extent_in_resels(
    voxels: int, *, mask_img: Any, fwhm: Tuple[float, float, float]
) -> float:
    """Convert a cluster-extent threshold in voxels to resolution elements.

    A bare voxel count is not interpretable across datasets: the same 20 voxels is a
    strong constraint on unsmoothed 3 mm data and almost none at 8 mm FWHM. One resel
    is the volume of a single smoothing kernel, so an extent in resels says how many
    independent bumps of noise a surviving cluster has to span.
    """
    voxel_sizes = np.sqrt((np.asarray(mask_img.affine)[:3, :3] ** 2).sum(axis=0))
    resel_volume = float(np.prod([max(f, 1e-9) for f in fwhm]))
    return float(voxels) * float(np.prod(voxel_sizes)) / resel_volume


__all__ = [
    "coverage_figure",
    "estimate_fwhm",
    "extent_in_resels",
    "mask_volume_mm3",
    "smoothness_note",
]
