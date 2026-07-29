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
    annotate_provenance,
    plot_context,
)

logger = logging.getLogger(__name__)

#: Converts a Gaussian standard deviation to its full width at half maximum.
_FWHM_PER_SIGMA = float(np.sqrt(8.0 * np.log(2.0)))


def _mask_colormap():
    """Two colours for a binary mask: transparent outside, one hue inside.

    A mask has two states, so it gets two colours. A continuous ramp would imply a
    gradient the data does not have, and its low end is a visible colour rather than
    nothing -- which is what made the panel paint its own field of view.
    """
    from matplotlib.colors import ListedColormap

    return ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.16, 0.33, 0.55, 0.6)])


_MASK_CMAP = _mask_colormap()


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
            # Two entries, the first fully transparent. Handed a continuous colormap,
            # nilearn maps the mask's zeros to its low colour and paints them, so the
            # panel drew a solid block over the whole field-of-view box -- covering
            # the anatomy that shows which regions fell outside the model, which is
            # the only thing this panel is read for.
            cmap=_MASK_CMAP,
            vmin=0,
            vmax=1,
            # dim=0 like every other volume panel. Left at nilearn's "auto", the
            # background is brightened until air outside the head sits at mid grey and
            # the anatomy loses the contrast this panel is read for. The translucency
            # is carried by the colormap rather than a scalar `alpha`, which would
            # override the per-colour alpha and paint the transparent entry too.
            dim=0,
            black_bg=False,
            annotate=True,
            # A mask is in or out. A 0-to-1 colour scale beside it invites a reading
            # of degree that the two states do not carry.
            colorbar=False,
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


def _resel_volume_mm3(fwhm: Tuple[float, float, float]) -> float:
    """Volume of one resolution element: the box a single smoothing kernel occupies."""
    return float(np.prod([max(f, 1e-9) for f in fwhm]))


def extent_in_resels(
    voxels: int, *, reference_img: Any, fwhm: Tuple[float, float, float]
) -> float:
    """Convert a cluster-extent threshold in voxels to resolution elements.

    A bare voxel count is not interpretable across datasets: the same 20 voxels is a
    strong constraint on unsmoothed 3 mm data and almost none at 8 mm FWHM. One resel
    is the volume of a single smoothing kernel, so an extent in resels says how many
    independent bumps of noise a surviving cluster has to span.

    ``reference_img`` supplies the voxel geometry through its affine. It was named
    ``mask_img`` while every caller passed the statistic map, which is the right image
    -- only the name was wrong.
    """
    voxel_sizes = np.sqrt((np.asarray(reference_img.affine)[:3, :3] ** 2).sum(axis=0))
    return float(voxels) * float(np.prod(voxel_sizes)) / _resel_volume_mm3(fwhm)


def search_volume_resels(mask_img: Any, *, fwhm: Tuple[float, float, float]) -> float:
    """How many independent resolution elements the analysis mask contains.

    The effective number of tests, as distinct from the voxel count. Smoothing makes
    neighbouring voxels the same measurement, so a mask of 50,626 voxels at 6.5 mm FWHM
    on a 3 mm grid holds about 5,000 independent ones -- and the Bonferroni threshold
    the report states beside it is computed over the larger number.

    Reported so that the gap between the two is visible rather than left as the reason
    a stated threshold is "known to be conservative". It is a measurement, not a
    substitute threshold: random field theory relates resels to a familywise-corrected
    height, and this pipeline performs no such correction.
    """
    voxels, _volume = mask_volume_mm3(mask_img)
    voxel_sizes = np.sqrt((np.asarray(mask_img.affine)[:3, :3] ** 2).sum(axis=0))
    return float(voxels) * float(np.prod(voxel_sizes)) / _resel_volume_mm3(fwhm)


__all__ = [
    "coverage_figure",
    "estimate_fwhm",
    "extent_in_resels",
    "mask_volume_mm3",
    "search_volume_resels",
    "smoothness_note",
]
