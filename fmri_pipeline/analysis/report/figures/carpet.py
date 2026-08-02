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
_UNCLASSIFIED_TISSUE = "Unclassified"

#: FreeSurfer aseg label ranges collapsed onto TISSUE_ORDER, used by the dseg path.
_ASEG_TO_CLASS = {
    **{
        label: 1
        for label in (3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)
    },
    **{label: 2 for label in (2, 41, 7, 46, 16, 28, 60, 77, 251, 252, 253, 254, 255)},
    **{label: 3 for label in (4, 43, 5, 44, 14, 15, 24, 31, 63)},
}

#: Display limit for a carpet already in per-voxel standard deviations.
RESIDUAL_Z_CLIP = 2.5

#: Framewise-displacement values worth comparing against, with their source.
#:
#: References, not verdicts. The line names a published convention so a reader can
#: locate the trace against it; the figure draws no conclusion, and no threshold here
#: was invented by this pipeline.
FD_REFERENCE_LINES: Tuple[Tuple[float, str], ...] = (
    (0.2, "0.2 mm (Power et al. 2014)"),
    (0.5, "0.5 mm (Power et al. 2012)"),
)


def _reference_frames(
    series: np.ndarray, sample_mask: Optional[np.ndarray]
) -> np.ndarray:
    """The frames a voxel's own centre and scale are taken from.

    ``sample_mask`` marks the frames the GLM kept. Non-steady-state volumes sit far
    above the steady-state signal, so including them inflates every voxel's reference
    and compresses the rest of the carpet toward neutral.
    """
    if sample_mask is None:
        return series
    keep = np.asarray(sample_mask, dtype=bool)
    if keep.size != series.shape[1]:
        raise ValueError(
            f"Sample mask has {keep.size} entries for {series.shape[1]} frames."
        )
    if keep.sum() < 2:
        raise ValueError("Standardisation needs at least two retained frames.")
    return series[:, keep]


def scale_carpet(
    voxel_timeseries: np.ndarray,
    *,
    sample_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Express each voxel as percent deviation from its own temporal mean.

    Percent rather than z, because a carpet exists to show that some voxels move more
    than others. Dividing each voxel by its own standard deviation gives every row
    unit variance *by construction*: a quiet voxel and one corrupted by a spike are
    then drawn identically, the panel becomes uniform texture, and the one comparison
    it is read for is the one it has removed. See
    ``test_z_scoring_is_what_destroys_that_contrast``.

    Dividing by the voxel's own mean rather than by a global one keeps the units
    comparable across tissue: grey matter, white matter and CSF differ severalfold in
    absolute intensity, so a common denominator would make the brightest tissue look
    the most active whatever it did.

    The cost is that one genuinely high-variance region can set the display range for
    everything, which is why the colour limit is a robust percentile rather than the
    maximum -- see :func:`carpet_colour_limit`.

    Frames outside ``sample_mask`` are still drawn. They are outliers, and a carpet
    that silently dropped them would hide that the run began unsteady.
    """
    series = np.asarray(voxel_timeseries, dtype=float)
    reference = _reference_frames(series, sample_mask)

    mean = np.mean(reference, axis=1, keepdims=True)
    # A voxel whose mean is zero has no percentage to be expressed in. Left to divide
    # it would paint background as the most extreme signal in the run.
    safe = np.where(np.abs(mean) > 0, mean, 1.0)
    return 100.0 * (series - mean) / np.abs(safe)


def carpet_colour_limit(
    scaled: np.ndarray,
    *,
    sample_mask: Optional[np.ndarray] = None,
    percentile: float = 99.0,
) -> float:
    """A symmetric display limit neither a bad frame nor a bad voxel can dominate.

    Each voxel's own ``percentile``-th absolute deviation, then the median across
    voxels. Two levels because there are two ways for a carpet to be dominated, and a
    single pooled percentile only survives one of them: a spike inflates its voxel's
    mean, which pushes *every* frame of that row to a large percentage, so one bad
    voxel contributes an entire row of extreme values. Pooled, that survives any
    percentile once the voxel count is small enough -- and the count is a display
    choice, not a property of the data.

    Taking the median across voxels also means the limit describes a typical voxel
    rather than the loudest one, which is what makes the ordinary tissue structure
    visible instead of compressing it toward mid-grey.

    ``sample_mask`` excludes the same frames :func:`scale_carpet` excludes from its
    reference. Without it a censored frame sets the limit it was meant to fall
    outside: three bad frames in two hundred is 1.5% of a row, so the 99th percentile
    lands on one of them and the scale stretches to cover exactly the values the panel
    intends to draw as out-of-range.

    Values beyond the limit are drawn in their own colours rather than saturated, so
    nothing is hidden by the choice.
    """
    values = np.abs(np.atleast_2d(np.asarray(scaled, dtype=float)))
    if values.size == 0:
        return 1.0
    if sample_mask is not None:
        keep = np.asarray(sample_mask, dtype=bool)
        if keep.size == values.shape[1] and keep.any():
            values = values[:, keep]
    with np.errstate(invalid="ignore"):
        per_voxel = np.nanpercentile(
            np.where(np.isfinite(values), values, np.nan), percentile, axis=1
        )
    usable = per_voxel[np.isfinite(per_voxel)]
    if usable.size == 0:
        return 1.0
    limit = float(np.median(usable))
    return limit if limit > 0 else 1.0


def standardise_carpet(
    voxel_timeseries: np.ndarray,
    *,
    sample_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Z-score each voxel, taking the scale from retained frames only.

    Retained for callers that want a unit-variance carpet. :func:`scale_carpet` is
    what the report draws; this flattens the between-voxel amplitude differences that
    panel is read for.

    ``sample_mask`` marks the frames the GLM kept. Non-steady-state volumes sit far
    above the steady-state signal, so including them in the mean and standard
    deviation inflates every voxel's scale and compresses the rest of the carpet
    toward neutral -- flattening the very structure the panel exists to show.

    The excluded frames are still *drawn*. They are outliers, and a carpet that
    silently dropped them would hide the fact that the run began unsteady.
    """
    series = np.asarray(voxel_timeseries, dtype=float)
    scale_source = _reference_frames(series, sample_mask)

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

    codes = np.asarray(tissue_codes_flat)
    if codes.ndim != 1 or codes.size != carpet.shape[0]:
        raise ValueError("Tissue codes must contain one value per carpet row.")
    allowed_codes = np.arange(len(TISSUE_ORDER) + 1)
    invalid_codes = np.setdiff1d(np.unique(codes), allowed_codes)
    if invalid_codes.size:
        raise ValueError(f"Unknown tissue code(s): {invalid_codes.tolist()}.")

    block_specs = [
        *[(tissue, code) for code, tissue in enumerate(TISSUE_ORDER, start=1)],
        (_UNCLASSIFIED_TISSUE, 0),
    ]
    block_indices = [np.flatnonzero(codes == code) for _, code in block_specs]
    order = np.concatenate([indices for indices in block_indices if indices.size])
    ordered = carpet[order]
    blocks: List[Tuple[str, int, int]] = []
    start = 0
    for (tissue, _), indices in zip(block_specs, block_indices):
        if indices.size:
            stop = start + indices.size
            blocks.append((tissue, start, stop))
            start = stop
    return ordered, blocks


def _check_length(name: str, values: Optional[np.ndarray], n_frames: int) -> None:
    if values is not None and len(values) != n_frames:
        raise ValueError(
            f"{name} has {len(values)} samples but the carpet has {n_frames} frames."
        )


def _spans(flags: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs of True in a boolean array, as ``(start, stop)`` index pairs.

    Shading each censored frame individually draws one patch per frame, which on a
    3,420-frame concatenation is thousands of artists for a picture that a few dozen
    would produce identically.
    """
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    return spans


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
    censored: Optional[np.ndarray] = None,
    not_retained: Optional[np.ndarray] = None,
    voxel_source: str = "",
    voxel_count_total: Optional[int] = None,
    colour_limit: Optional[float] = None,
    value_label: str = "% signal change",
    title: str = "",
) -> plt.Figure:
    """Draw a carpet with FD and DVARS above it on a shared time axis.

    ``fd`` and ``dvars`` keep their ``NaN`` values. The first frame of a run has no
    defined framewise displacement; substituting zero draws a dip to "no motion" at
    every run boundary, which is a fabricated measurement. Matplotlib gaps a NaN.

    ``censored`` is a boolean per frame marking the volumes the GLM dropped. It is
    drawn on the motion trace rather than as a panel of its own, because the question
    it answers is whether the frames that were removed are the ones that moved -- and
    that is a comparison between two things, not a fact about either. The count is in
    the motion table; what the carpet adds is *which* frames.
    """
    carpet = np.asarray(carpet, dtype=float)
    n_frames = carpet.shape[1]
    _check_length("fd", fd, n_frames)
    _check_length("dvars", dvars, n_frames)
    _check_length("censored", censored, n_frames)
    _check_length("not_retained", not_retained, n_frames)

    total_voxels = carpet.shape[0] if voxel_count_total is None else int(voxel_count_total)
    if total_voxels < carpet.shape[0]:
        raise ValueError(
            f"voxel_count_total is {total_voxels}, below the {carpet.shape[0]} carpet rows."
        )

    drawn, drawn_codes = subsample_rows(carpet, tissue_codes)
    ordered, blocks = order_by_tissue(drawn, drawn_codes)
    not_retained_mask = np.asarray(not_retained, dtype=bool) if not_retained is not None else None
    if not_retained_mask is not None and not_retained_mask.any():
        ordered = ordered.copy()
        ordered[:, not_retained_mask] = np.nan
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

        censored_mask = (
            np.asarray(censored, dtype=bool) if censored is not None else None
        )

        index = 0
        if fd is not None:
            fd_values = np.asarray(fd, dtype=float)
            if censored_mask is not None and censored_mask.any():
                # Behind the trace, so the shading says which frames left the model
                # without obscuring the motion that is the reason they did.
                for start, stop in _spans(censored_mask):
                    axes[index].axvspan(
                        times[start],
                        times[min(stop, n_frames - 1)],
                        color=GUIDE_COLOR,
                        alpha=0.25,
                        linewidth=0,
                        zorder=0,
                    )
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
        # Derived here when the caller did not supply one, so a figure drawn directly
        # is scaled the same way the report scales it.
        limit = (
            float(colour_limit)
            if colour_limit is not None and float(colour_limit) > 0
            else carpet_colour_limit(carpet)
        )
        # Grayscale, not a diverging map: the tissue block labels and the motion
        # traces above already carry this figure's colour, and a second colour
        # scale competing with them makes neither readable.
        #
        # Out-of-range values are coloured rather than left to saturate. Clipped to
        # the top of a grey ramp they render white on a white page and read as
        # missing data -- which is the opposite of the truth, since those are the
        # most extreme voxels in the run. Non-steady-state volumes land here by
        # design, so this is the common case, not an edge case.
        colormap = plt.get_cmap("gray").with_extremes(
            bad="#d9d9d9",
            over=OKABE_ITO["vermillion"],
            under=OKABE_ITO["blue"],
        )
        image = carpet_axis.imshow(
            ordered,
            aspect="auto",
            cmap=colormap,
            vmin=-limit,
            vmax=limit,
            extent=(0.0, float(times[-1] if n_frames else 0.0), ordered.shape[0], 0),
            rasterized=True,
            interpolation="nearest",
        )
        carpet_axis.set_xlabel("Time (seconds, concatenated runs)")
        bar = figure.colorbar(
            image, ax=carpet_axis, fraction=0.015, pad=0.01, extend="both"
        )
        bar.set_label(f"{value_label} (beyond ±{limit:.2g} coloured)")

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

        # The denominator names where it came from. Counted against the field of view
        # it reads as a far smaller sampling fraction than the panel actually applied,
        # and says nothing about whether the voxels drawn are brain.
        drawn_of = f"{ordered.shape[0]:,} of {total_voxels:,} voxels drawn"
        if voxel_source:
            drawn_of += f" ({voxel_source})"
        provenance = [
            drawn_of,
            f"{n_frames:,} frames · TR {float(tr):.3g} s",
            f"voxel order: {tissue_source}",
        ]
        if censored_mask is not None:
            # Shaded on the motion trace, so the count needs a key: an unexplained
            # grey band is a mark a reader cannot read.
            provenance.append(
                f"shaded on the motion trace: {int(censored_mask.sum()):,} censored "
                f"frame(s), which the model did not use"
            )
        if not_retained_mask is not None:
            provenance.append(
                f"grey carpet columns: {int(not_retained_mask.sum()):,} acquired "
                "frame(s) not retained; no fitted series value exists"
            )
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
