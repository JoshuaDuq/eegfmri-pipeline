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
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.report.figures._mosaic import (
    DEFAULT_CUTS_PER_ROW,
    mosaic_figure,
)
from fmri_pipeline.analysis.report.style import plot_context

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


#: Residual volumes drawn when estimating smoothness from a 4D series.
#:
#: The estimate is a median over per-volume estimates and settles within a few percent
#: well before this many frames, while the series itself runs to several hundred
#: frames per run across six runs. Reading all of them would cost minutes per subject
#: to move an answer that has already converged.
_RESIDUAL_VOLUMES_SAMPLED = 24


def estimate_fwhm_from_residuals(
    residual_img: Any, *, mask: Any = None
) -> Tuple[float, float, float]:
    """Estimate spatial smoothness from a 4D residual series, in millimetres.

    The right input for the estimator, and the one this pipeline could not offer until
    it began writing residuals. Smoothness is a property of the noise field; taken from
    a statistic map instead, real activation inflates it, because signal is spatially
    structured and the estimator cannot tell that structure from smoothing. Excluding
    suprathreshold voxels recovers most of the difference and is what the map-based
    path does -- but under ``threshold_mode: none`` there is no height to exclude by,
    and that estimate can only be reported as an upper bound.

    A residual field carries no activation to exclude, so no heuristic is needed and
    the answer is an estimate rather than a bound.

    Each sampled volume is estimated separately and the per-axis median is returned.
    The median rather than the mean because a single volume caught mid-motion is
    spatially structured in a way that inflates its own estimate and nothing else's.
    """
    import nibabel as nib

    # Sliced from ``dataobj`` a volume at a time rather than materialised. A run's
    # residual series is several hundred frames over the whole mask, and six of them
    # per subject; reading all of it to estimate from two dozen volumes would cost
    # more memory than the rest of the report together.
    data = residual_img.dataobj
    shape = tuple(data.shape)
    if len(shape) != 4:
        raise ValueError(f"Residual smoothness needs a 4D series, got shape {shape}.")
    if shape[3] == 0:
        raise ValueError("Residual smoothness needs at least one volume.")

    mask_array: Optional[np.ndarray] = None
    if mask is not None:
        mask_array = np.asarray(
            mask if isinstance(mask, np.ndarray) else mask.dataobj
        ).astype(bool)

    # Evenly spaced across the series rather than the first N, so a run whose start is
    # atypical -- settling gradients, an early motion spike -- does not decide the
    # estimate on its own.
    count = min(_RESIDUAL_VOLUMES_SAMPLED, int(shape[3]))
    indices = np.unique(np.linspace(0, shape[3] - 1, count).astype(int))

    estimates = []
    for index in indices:
        volume = nib.Nifti1Image(
            np.asarray(data[..., int(index)], dtype=float), residual_img.affine
        )
        try:
            estimates.append(estimate_fwhm(volume, mask=mask_array))
        except ValueError:
            # A constant volume carries no smoothness. One such frame is not a reason
            # to lose the estimate the rest of the series supports.
            continue
    if not estimates:
        raise ValueError("No residual volume carried enough spread to estimate from.")

    median = np.median(np.asarray(estimates, dtype=float), axis=0)
    return (float(median[0]), float(median[1]), float(median[2]))


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


def run_contribution_map(bold_imgs: Sequence[Any]) -> Any:
    """How many runs reach each voxel, as an integer volume.

    The analysis mask is an intersection, so the voxels some runs hold and others do
    not are discarded before the coverage panel is drawn -- and those are precisely
    the voxels the panel is consulted for. A binary mask over anatomy can only say
    that the mask covers the brain; this says where coverage was lost and to how many
    runs.

    Each run's own extent comes from :func:`nilearn.masking.compute_epi_mask`, applied
    to the run's first volume rather than the whole series: coverage is a property of
    the field of view and of dropout, not of any one timepoint, and reading one volume
    per run keeps the panel cheap on a study with hundreds of frames.

    Runs are resampled onto the first run's grid when they do not already share it, so
    a study whose runs differ in resolution still yields a countable map.
    """
    from nilearn.image import index_img, resample_to_img
    from nilearn.masking import compute_epi_mask

    runs = list(bold_imgs)
    if len(runs) < 2:
        raise ValueError(
            "A run-contribution map needs more than one run; with one run every "
            "modelled voxel is reached by every run and the map is uniform."
        )

    reference = None
    counts = None
    for img in runs:
        volume = index_img(img, 0) if len(getattr(img, "shape", ())) == 4 else img
        mask = compute_epi_mask(volume)
        if reference is None:
            reference = mask
        elif not np.allclose(mask.affine, reference.affine) or mask.shape != reference.shape:
            # Nearest-neighbour: a mask resampled with interpolation is no longer a
            # mask, and a half-covered voxel counted as half a run is not a reading
            # this panel offers.
            mask = resample_to_img(
                mask, reference, interpolation="nearest", force_resample=True, copy_header=True
            )
        data = np.asanyarray(mask.dataobj).astype(bool)
        counts = data.astype(np.int16) if counts is None else counts + data
    return nib.Nifti1Image(counts, reference.affine)


def _contribution_colormap(n_runs: int):
    """One step per run count, so a voxel's count is readable off the key.

    A continuous ramp would invite reading a count of four as "somewhat covered". The
    counts are integers and the colours are too, so the map has exactly ``n_runs + 1``
    entries and is driven by ``vmin``/``vmax`` rather than by a ``BoundaryNorm``:
    Nilearn passes ``vmin``/``vmax`` down to Matplotlib itself, and supplying a norm as
    well raises "Passing a Normalize instance simultaneously with vmin/vmax is not
    supported" -- which ``mosaic_figure`` catches per row, so the panel rendered as
    three empty bands rather than failing outright.
    """
    from matplotlib.colors import ListedColormap

    ramp = plt.get_cmap("YlGnBu")
    # Zero is transparent -- outside every run is outside the picture, not a value.
    colours = [(0.0, 0.0, 0.0, 0.0)]
    colours += [ramp(0.25 + 0.72 * (index / max(n_runs - 1, 1))) for index in range(n_runs)]
    return ListedColormap(colours)


def _draw_count_legend(figure: plt.Figure, *, n_runs: int, cmap) -> None:
    """A discrete key for the run counts, in the band the colourbar would occupy.

    A continuous colourbar cannot label an integer count without inviting the reading
    that a voxel is fractionally covered, so the key is one swatch per count.
    """
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=cmap(index + 1), edgecolor="none", label=f"{index + 1}")
        for index in range(n_runs)
    ]
    figure.legend(
        handles=handles,
        title="runs",
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        handlelength=1.1,
        handleheight=1.1,
        labelspacing=0.32,
    )


def coverage_figure(
    mask_img: Any,
    *,
    bg_img: Any = None,
    extent_note: str = "",
    smoothness_note: str = "",
    title: str = "",
    radiological: bool = False,
    n_cuts: int = DEFAULT_CUTS_PER_ROW,
    contribution_img: Any = None,
    n_runs: int = 0,
) -> plt.Figure:
    """Draw the analysis mask over the background, with its extent stated.

    Rendered as an ROI overlay rather than a bare binary volume so a reader can see
    which anatomy fell outside the model, which is the only reading that matters --
    and that reading requires ``bg_img``. Over an empty background the panel shows a
    blob whose missing regions cannot be named.

    Drawn as a full mosaic rather than the three-slice ``ortho`` it used to use. This
    panel exists to show *which* anatomy fell outside the model, and orbitofrontal
    and inferior temporal dropout -- the regions it is most often read for -- can miss
    all three ortho slices entirely.

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

    contributions = (
        np.asanyarray(contribution_img.dataobj) if contribution_img is not None else None
    )
    show_counts = contributions is not None and n_runs > 1
    count_cmap = _contribution_colormap(int(n_runs)) if show_counts else None

    def draw_counts(figure, rect, direction, cuts):
        # ``plot_roi`` rather than ``plot_stat_map``: this is a label image, its values
        # are integers, and plot_roi is the plotter Nilearn documents for one. It also
        # keeps zero on the colormap's transparent entry instead of thresholding it.
        plotting.plot_roi(
            contribution_img,
            bg_img=bg_img,
            display_mode=direction,
            cut_coords=list(cuts),
            cmap=count_cmap,
            vmin=0,
            vmax=int(n_runs),
            dim=0,
            black_bg=False,
            annotate=False,
            radiological=radiological,
            colorbar=False,
            figure=figure,
            axes=rect,
        )

    def draw(figure, rect, direction, cuts):
        plotting.plot_roi(
            mask_img,
            bg_img=bg_img,
            display_mode=direction,
            cut_coords=list(cuts),
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
            annotate=False,
            radiological=radiological,
            # A mask is in or out. A 0-to-1 colour scale beside it invites a reading
            # of degree that the two states do not carry.
            colorbar=False,
            figure=figure,
            axes=rect,
        )

    # The old first line was "N voxels modelled of M in the field of view". M counts
    # air, so the ratio measured how much empty space the acquisition box contained
    # and told a reader nothing about coverage.
    lines = [f"{voxels:,} voxels modelled ({volume / 1000.0:,.1f} cm³)"]
    if extent_note:
        lines.append(extent_note)
    if smoothness_note:
        lines.append(smoothness_note)
    if show_counts:
        reached = contributions[contributions > 0]
        partial = int(np.count_nonzero((contributions > 0) & (contributions < n_runs)))
        lines.append(
            f"colour: how many of the {int(n_runs)} runs reach each voxel; "
            f"{partial:,} voxel(s) reached by some runs and not others"
        )
        lines.append(
            f"{int(np.count_nonzero(reached == n_runs)):,} voxel(s) reached by every run"
        )
    lines.append("voxels outside this mask were not tested")

    with plot_context():
        figure = mosaic_figure(
            draw_counts if show_counts else draw,
            reference_img=mask_img,
            mask_img=mask_img,
            n_cuts=n_cuts,
            title=title,
            radiological=radiological,
            colorbar=None,
            provenance=lines,
        )
        if show_counts:
            _draw_count_legend(figure, n_runs=int(n_runs), cmap=count_cmap)
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
    "estimate_fwhm_from_residuals",
    "extent_in_resels",
    "mask_volume_mm3",
    "search_volume_resels",
    "smoothness_note",
]
