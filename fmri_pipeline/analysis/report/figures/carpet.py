"""Voxel carpet with motion traces on a shared time axis.

The carpet and the motion traces are one figure because they are only diagnostic
together: a band in the carpet means nothing until you can see whether a motion
spike sits above it. They were previously two figures on independent axes.

Voxels are grouped by tissue class. Ordering by raw mask index -- which is what a
mask's own iteration order gives -- scatters grey matter, white matter, and CSF
through the image and destroys the banding that makes a carpet readable at all.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.assets import PlotAssets
from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)

logger = logging.getLogger(__name__)

#: Row-block order. Codes are the 1-based index into this tuple; 0 means unassigned.
TISSUE_ORDER: Tuple[str, ...] = ("GM", "WM", "CSF")

#: FreeSurfer aseg label ranges collapsed onto TISSUE_ORDER, used by the dseg path.
_ASEG_TO_CLASS = {
    **{
        label: 1
        for label in (3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)
    },
    **{label: 2 for label in (2, 41, 7, 46, 16, 28, 60, 77, 251, 252, 253, 254, 255)},
    **{label: 3 for label in (4, 43, 5, 44, 14, 15, 24, 31, 63)},
}

_CARPET_CLIP = 2.5

#: Framewise-displacement values worth comparing against, with their source.
#:
#: References, not verdicts. The line names a published convention so a reader can
#: locate the trace against it; the figure draws no conclusion, and no threshold here
#: was invented by this pipeline.
FD_REFERENCE_LINES: Tuple[Tuple[float, str], ...] = (
    (0.2, "0.2 mm (Power et al. 2014)"),
    (0.5, "0.5 mm (Power et al. 2012)"),
)


def standardise_carpet(
    voxel_timeseries: np.ndarray,
    *,
    sample_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Z-score each voxel, taking the scale from retained frames only.

    ``sample_mask`` marks the frames the GLM kept. Non-steady-state volumes sit far
    above the steady-state signal, so including them in the mean and standard
    deviation inflates every voxel's scale and compresses the rest of the carpet
    toward neutral -- flattening the very structure the panel exists to show.

    The excluded frames are still *drawn*. They are outliers, and a carpet that
    silently dropped them would hide the fact that the run began unsteady.
    """
    series = np.asarray(voxel_timeseries, dtype=float)
    scale_source = series
    if sample_mask is not None:
        keep = np.asarray(sample_mask, dtype=bool)
        if keep.size != series.shape[1]:
            raise ValueError(
                f"Sample mask has {keep.size} entries for {series.shape[1]} frames."
            )
        if keep.sum() < 2:
            raise ValueError("Standardisation needs at least two retained frames.")
        scale_source = series[:, keep]

    mean = np.mean(scale_source, axis=1, keepdims=True)
    std = np.std(scale_source, axis=1, keepdims=True)
    # A constant voxel has no meaningful z score; leave it at zero rather than
    # dividing by zero and painting it as an extreme value.
    std = np.where(std > 0, std, 1.0)
    return (series - mean) / std


def _resample_to(img: Any, reference_img: Any, *, order: int) -> Optional[np.ndarray]:
    from nilearn.image import resample_to_img

    interpolation = "nearest" if order == 0 else "continuous"
    resampled = resample_to_img(
        img,
        reference_img,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )
    return np.asarray(resampled.get_fdata())


def resolve_tissue_codes(
    shape: Tuple[int, int, int],
    *,
    assets: PlotAssets,
    reference_img: Any,
) -> Tuple[Optional[np.ndarray], str]:
    """Return per-voxel tissue class codes and the source they came from.

    Probability maps win over the discrete segmentation: assigning each voxel to its
    highest-probability class is closer to what the carpet wants than a label map
    built for a different purpose. Returns ``(None, "none")`` when neither is
    available, which the figure then declares rather than hiding.
    """
    import nibabel as nib

    if assets.probseg:
        stack: List[np.ndarray] = []
        classes: List[int] = []
        for index, tissue in enumerate(TISSUE_ORDER, start=1):
            path = assets.probseg.get(tissue)
            if path is None:
                continue
            resampled = _resample_to(nib.load(str(path)), reference_img, order=1)
            if resampled is not None and resampled.shape == shape:
                stack.append(resampled)
                classes.append(index)
        if stack:
            probabilities = np.stack(stack, axis=0)
            winner = np.argmax(probabilities, axis=0)
            codes = np.take(np.asarray(classes), winner)
            # A voxel with no probability anywhere belongs to no class.
            codes = np.where(probabilities.max(axis=0) > 0, codes, 0)
            return codes.astype(np.int8), "probseg"

    if assets.dseg is not None:
        labels = _resample_to(nib.load(str(assets.dseg)), reference_img, order=0)
        if labels is not None and labels.shape == shape:
            codes = np.zeros(shape, dtype=np.int8)
            integer_labels = np.asarray(labels).astype(int)
            for label, klass in _ASEG_TO_CLASS.items():
                codes[integer_labels == label] = klass
            if np.any(codes > 0):
                return codes, "dseg"

    logger.info("No tissue segmentation resolved; carpet will be unordered.")
    return None, "none"


def subsample_rows(
    carpet: np.ndarray,
    tissue_codes_flat: Optional[np.ndarray],
    *,
    max_rows: int = 6000,
    min_rows_per_class: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reduce the carpet to a drawable number of rows without losing a tissue class.

    Strictly proportional sampling is the obvious choice and the wrong one: CSF is a
    few percent of the voxels, so a proportional draw can leave it a handful of rows
    that disappear at figure resolution -- and CSF is where a global artefact shows
    up first. Each class present therefore keeps at least ``min_rows_per_class``
    rows, and the rest of the budget is shared in proportion. The resulting row
    counts are stated on the figure, because the block heights no longer represent
    tissue volume once a floor has been applied.

    Sampling is a deterministic stride rather than a random draw, so a report
    regenerated from the same derivatives is byte-identical.
    """
    n_rows = carpet.shape[0]
    if n_rows <= max_rows:
        return carpet, tissue_codes_flat

    if tissue_codes_flat is None:
        index = np.linspace(0, n_rows - 1, max_rows).astype(int)
        return carpet[index], None

    selected: List[np.ndarray] = []
    for code in np.unique(tissue_codes_flat):
        positions = np.flatnonzero(tissue_codes_flat == code)
        share = int(round(max_rows * positions.size / n_rows))
        take = min(positions.size, max(share, min_rows_per_class))
        stride = np.linspace(0, positions.size - 1, take).astype(int)
        selected.append(positions[stride])

    index = np.sort(np.concatenate(selected))
    return carpet[index], tissue_codes_flat[index]


def order_by_tissue(
    carpet: np.ndarray,
    tissue_codes_flat: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """Sort carpet rows into tissue blocks and return the block extents."""
    if tissue_codes_flat is None:
        return carpet, []
    order = np.argsort(tissue_codes_flat, kind="stable")
    ordered = carpet[order]
    blocks: List[Tuple[str, int, int]] = []
    sorted_codes = tissue_codes_flat[order]
    for index, tissue in enumerate(TISSUE_ORDER, start=1):
        positions = np.flatnonzero(sorted_codes == index)
        if positions.size:
            blocks.append((tissue, int(positions[0]), int(positions[-1]) + 1))
    return ordered, blocks


def _check_length(name: str, values: Optional[np.ndarray], n_frames: int) -> None:
    if values is not None and len(values) != n_frames:
        raise ValueError(
            f"{name} has {len(values)} samples but the carpet has {n_frames} frames."
        )


def carpet_figure(
    carpet: np.ndarray,
    *,
    tissue_codes: Optional[np.ndarray],
    tissue_source: str,
    tr: float,
    run_boundaries: Sequence[int],
    run_labels: Sequence[str],
    fd: Optional[np.ndarray] = None,
    dvars: Optional[np.ndarray] = None,
    dvars_label: str = "DVARS",
    title: str = "",
) -> plt.Figure:
    """Draw a carpet with FD and DVARS above it on a shared time axis.

    ``fd`` and ``dvars`` keep their ``NaN`` values. The first frame of a run has no
    defined framewise displacement; substituting zero draws a dip to "no motion" at
    every run boundary, which is a fabricated measurement. Matplotlib gaps a NaN.
    """
    carpet = np.asarray(carpet, dtype=float)
    n_frames = carpet.shape[1]
    _check_length("fd", fd, n_frames)
    _check_length("dvars", dvars, n_frames)

    drawn, drawn_codes = subsample_rows(carpet, tissue_codes)
    ordered, blocks = order_by_tissue(drawn, drawn_codes)
    times = np.arange(n_frames) * float(tr)
    boundary_times = [float(b) * float(tr) for b in run_boundaries]

    trace_count = sum(x is not None for x in (fd, dvars))
    heights = [0.6] * trace_count + [3.0]
    with plot_context():
        figure, axes = plt.subplots(
            len(heights),
            1,
            figsize=(11, 2.0 + 1.2 * trace_count),
            sharex=True,
            gridspec_kw={"height_ratios": heights},
        )
        axes = np.atleast_1d(axes)

        index = 0
        if fd is not None:
            fd_values = np.asarray(fd, dtype=float)
            axes[index].plot(
                times, fd_values, color=OKABE_ITO["vermillion"], linewidth=0.8
            )
            axes[index].set_ylabel("FD (mm)")
            # Published reference values, drawn only where they fall inside the
            # data's own range -- a line far above every sample adds no comparison
            # and costs the trace its vertical resolution.
            ceiling = (
                float(np.nanmax(fd_values)) if np.isfinite(fd_values).any() else 0.0
            )
            for level, label in FD_REFERENCE_LINES:
                if level <= ceiling * 1.5:
                    axes[index].axhline(
                        level, color=GUIDE_COLOR, linestyle=":", linewidth=0.8
                    )
                    axes[index].annotate(
                        label,
                        xy=(1.0, level),
                        xycoords=("axes fraction", "data"),
                        xytext=(-2, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        fontsize=6,
                        color=GUIDE_COLOR,
                    )
            index += 1
        if dvars is not None:
            axes[index].plot(
                times, np.asarray(dvars, dtype=float), color=OKABE_ITO["blue"],
                linewidth=0.8,
            )
            axes[index].set_ylabel(dvars_label)
            index += 1

        carpet_axis = axes[-1]
        # Grayscale, not a diverging map: the tissue block labels and the motion
        # traces above already carry this figure's colour, and a second colour
        # scale competing with them makes neither readable.
        image = carpet_axis.imshow(
            np.clip(ordered, -_CARPET_CLIP, _CARPET_CLIP),
            aspect="auto",
            cmap="gray",
            vmin=-_CARPET_CLIP,
            vmax=_CARPET_CLIP,
            extent=(0.0, float(times[-1] if n_frames else 0.0), ordered.shape[0], 0),
            rasterized=True,
        )
        carpet_axis.set_xlabel("Time (seconds, concatenated runs)")
        bar = figure.colorbar(image, ax=carpet_axis, fraction=0.015, pad=0.01)
        bar.set_label(f"z (per voxel, clipped at ±{_CARPET_CLIP})")

        if blocks:
            carpet_axis.set_yticks([0.5 * (start + stop) for _, start, stop in blocks])
            carpet_axis.set_yticklabels([name for name, _, _ in blocks])
            for _, _, stop in blocks[:-1]:
                carpet_axis.axhline(stop, color="white", linewidth=1.0)
            carpet_axis.set_ylabel("Voxels by tissue")
        else:
            carpet_axis.set_yticks([])
            carpet_axis.set_ylabel(f"Voxels (unordered: {tissue_source})")

        for axis in axes:
            for boundary in boundary_times:
                axis.axvline(boundary, color=GUIDE_COLOR, linewidth=0.7, alpha=0.6)

        starts = [0.0, *boundary_times]
        for label, start in zip(run_labels, starts):
            axes[0].annotate(
                label,
                xy=(start, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=7,
                color=GUIDE_COLOR,
            )

        if title:
            figure.suptitle(title)
        figure.tight_layout()

        provenance = [
            f"{ordered.shape[0]:,} of {carpet.shape[0]:,} voxels drawn",
            f"{n_frames:,} frames · TR {float(tr):.3g} s",
            f"voxel order: {tissue_source}",
        ]
        if blocks:
            # Row counts are stated because the per-class floor in subsample_rows
            # means block heights no longer represent tissue volume.
            provenance.append(
                " ".join(f"{name} {stop - start:,}" for name, start, stop in blocks)
            )
        annotate_provenance(figure, provenance)
        return figure


__all__ = [
    "FD_REFERENCE_LINES",
    "TISSUE_ORDER",
    "carpet_figure",
    "order_by_tissue",
    "resolve_tissue_codes",
    "standardise_carpet",
    "subsample_rows",
]
