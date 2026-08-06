"""Slice mosaic and glass-brain panels for a statistical map."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.figures._mosaic import (
    DEFAULT_CUTS_PER_ROW,
    ColorbarSpec,
    draw_colorbar,
    mosaic_figure,
    suppressed_band,
)
from fmri_pipeline.analysis.report.style import (
    MAGNITUDE_CMAP,
    OKABE_ITO,
    SIGNED_CMAP,
    annotate_provenance,
    clipped_fraction,
    orientation_label,
    plot_context,
    robust_symmetric_limit,
    robust_upper_limit,
    suprathreshold_limit,
)

logger = logging.getLogger(__name__)


def _masked_values(stat_img: Any, mask_img: Any = None) -> Tuple[np.ndarray, str]:
    """Return the values a colour limit should be computed from, and their source.

    A statistical map is mostly background, and the background is exactly zero. A
    percentile taken over the whole volume is therefore a percentile of a
    distribution dominated by zeros: it lands well below the real one, and every
    voxel above it saturates to a single colour. Measured on a map with a 21% brain
    fraction, the whole-volume limit was 1.44x too low; the error grows as the brain
    occupies less of the field of view.

    An explicit analysis mask is preferred. Failing that, exact zeros are excluded --
    an in-brain voxel is essentially never exactly zero, so this recovers the brain
    on any map that has a background at all, and changes nothing on a map that does
    not.
    """
    data = np.asarray(stat_img.get_fdata())
    finite = np.isfinite(data)

    # Why no mask was used, when none was. A rejected mask and an absent one are
    # different faults -- one is looked for in the geometry, the other in the manifest
    # -- and reporting them with the same words sends the reader to the wrong place.
    reason = "no mask supplied"
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)
        if mask.shape == data.shape:
            return data[finite & mask], "analysis mask"
        logger.warning(
            "Analysis mask shape %s does not match the map's %s; falling back to "
            "nonzero voxels for the colour limit.",
            mask.shape,
            data.shape,
        )
        reason = "supplied mask did not fit the map"

    nonzero = finite & (data != 0)
    if nonzero.any() and int(nonzero.sum()) < int(finite.sum()):
        return data[nonzero], f"nonzero voxels ({reason})"
    return data[finite], f"all voxels ({reason})"


def _resolve_vmax(
    stat_img: Any,
    *,
    threshold: Optional[float],
    vmax: Optional[float],
    mask_img: Any = None,
) -> float:
    """Choose a colour limit appropriate to whether the panel is thresholded.

    A thresholded panel takes its limit from the surviving voxels only. Reusing the
    whole-map limit is what makes these panels saturate: for z ~ N(0,1) the robust
    limit is about 2.58 against a typical 2.3 threshold, leaving no usable range.
    """
    if vmax is not None:
        return float(vmax)
    values, _source = _masked_values(stat_img, mask_img)
    if threshold is not None and threshold > 0:
        return suprathreshold_limit(values, threshold=float(threshold))
    return robust_symmetric_limit(values)


def apply_sidedness(stat_img: Any, *, two_sided: bool) -> Any:
    """Zero the negative half of ``stat_img`` when inference was one-sided.

    Nilearn's map plotters threshold on the absolute value, so they always render
    two-sided. A panel drawn from a one-sided test therefore shows negative clusters
    that the test never examined, and disagrees with the cluster table beside it.
    """
    if two_sided:
        return stat_img

    import nibabel as nib

    data = np.asarray(stat_img.get_fdata())
    return nib.Nifti1Image(
        np.clip(data, 0.0, None).astype(data.dtype), stat_img.affine, stat_img.header
    )


def _provenance(
    values: np.ndarray,
    *,
    threshold: Optional[float],
    limit: float,
    two_sided: bool,
    radiological: bool = False,
    limit_source: str = "",
) -> List[str]:
    """Build the self-description line for a map panel."""
    lines = [f"n = {values.size:,} voxels"]
    # The clipped fraction has to describe the voxels the panel draws. Taken over the
    # whole mask on a thresholded panel it is a property of a population the reader
    # cannot see, and it reads far lower than the saturation actually shown: the
    # sub-threshold voxels that dominate the count are nowhere near the colour limit.
    shown = values
    if threshold:
        comparison = "|z|" if two_sided else "z"
        lines.append(f"{comparison} > {float(threshold):.2f}")
        compared = np.abs(values) if two_sided else values
        shown = values[compared > float(threshold)]
    else:
        lines.append("unthresholded")
    fraction = clipped_fraction(shown, limit=limit) if shown.size else 0.0
    scope = "of drawn voxels" if threshold else ""
    limit_line = (
        f"colour limit ±{limit:.2f} ({fraction:.1%} clipped{f' {scope}' if scope else ''})"
    )
    if limit_source:
        # Which voxels the limit came from changes it substantially, so a reader
        # comparing two panels needs to know.
        limit_line += f", from {limit_source}"
    lines.append(limit_line)
    # A brain figure that does not state its convention cannot be checked, and a
    # left/right error is not visible in the image.
    lines.append(orientation_label(radiological))
    return lines


def stat_map_mosaic(
    stat_img: Any,
    *,
    bg_img: Any = None,
    mask_img: Any = None,
    threshold: Optional[float] = None,
    vmax: Optional[float] = None,
    two_sided: bool = True,
    radiological: bool = False,
    title: str = "",
    cbar_label: str = "z",
    cmap: str = SIGNED_CMAP,
    n_cuts: int = DEFAULT_CUTS_PER_ROW,
) -> Any:
    """Draw a slice mosaic of ``stat_img``.

    Laid out by :func:`~fmri_pipeline.analysis.report.figures._mosaic.mosaic_figure`
    rather than by nilearn's ``display_mode="mosaic"``, which drew the title on top of
    the first tile and spread its cuts across the underlay rather than the mask. Slice
    coordinates and a left/right key are still drawn -- a mosaic without them tells a
    reader neither where a cluster is nor which hemisphere it is in -- but by the
    layout, where they cannot collide with a neighbouring tile.
    """
    from nilearn import plotting

    values, limit_source = _masked_values(stat_img, mask_img)
    resolved_vmax = _resolve_vmax(
        stat_img, threshold=threshold, vmax=vmax, mask_img=mask_img
    )
    plotted = apply_sidedness(stat_img, two_sided=two_sided)

    def draw(figure, rect, direction, cuts):
        plotting.plot_stat_map(
            plotted,
            bg_img=bg_img,
            display_mode=direction,
            cut_coords=list(cuts),
            threshold=float(threshold) if threshold else None,
            colorbar=False,
            vmax=resolved_vmax,
            cmap=cmap,
            dim=0,
            black_bg=False,
            symmetric_cbar=True,
            annotate=False,
            radiological=radiological,
            figure=figure,
            axes=rect,
        )

    with plot_context():
        return mosaic_figure(
            draw,
            reference_img=stat_img,
            mask_img=mask_img,
            n_cuts=n_cuts,
            title=title,
            radiological=radiological,
            colorbar=ColorbarSpec(
                cmap=cmap,
                vmin=-resolved_vmax,
                vmax=resolved_vmax,
                label=cbar_label,
                suppressed=suppressed_band(threshold, two_sided=two_sided),
            ),
            provenance=_provenance(
                values,
                threshold=threshold,
                limit=resolved_vmax,
                two_sided=two_sided,
                radiological=radiological,
                limit_source=limit_source,
            ),
        )


def magnitude_mosaic(
    img: Any,
    *,
    bg_img: Any = None,
    mask_img: Any = None,
    vmax: Optional[float] = None,
    radiological: bool = False,
    title: str = "",
    cbar_label: str = "",
    cmap: str = MAGNITUDE_CMAP,
    n_cuts: int = DEFAULT_CUTS_PER_ROW,
    extra_provenance: Sequence[str] = (),
) -> Any:
    """Draw a slice mosaic of an unsigned magnitude -- a standard error, a tSNR.

    Separate from :func:`stat_map_mosaic` because a symmetric scale is wrong for a
    quantity with no negative half. Drawn through the signed path, a standard error
    got limits of ±1.09 for data spanning 0 to 1.09: half the ramp went to values
    that cannot occur, every voxel landed in the top quarter of the colours, and the
    panel rendered as a flat wash -- showing none of the spatial structure it exists
    to show. Exact zeros outside the brain landed on the ramp's midpoint and drew a
    solid block over the anatomy.

    The limit therefore comes from :func:`robust_upper_limit` over the positive
    values, the scale runs from zero, and non-positive voxels are left transparent so
    the background shows through.

    Drawn as a full mosaic rather than the three-slice ``ortho`` this used to use. A
    tSNR or standard-error map is read for *where* the measurement falls off, and
    three slices cannot show a dropout that misses all three.
    """
    from nilearn import plotting

    values, limit_source = _masked_values(img, mask_img)
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("A magnitude panel requires at least one positive voxel.")
    resolved_vmax = float(vmax) if vmax is not None else robust_upper_limit(positive)

    def draw(figure, rect, direction, cuts):
        plotting.plot_stat_map(
            img,
            bg_img=bg_img,
            display_mode=direction,
            cut_coords=list(cuts),
            # Just above zero: hides the background without hiding any measurement,
            # since a magnitude of exactly zero is an absent voxel rather than a small
            # one. Left unthresholded, those voxels take the ramp's low colour and
            # paint a solid block over the anatomy that reads as a measured value.
            threshold=float(np.finfo(np.float32).tiny),
            colorbar=False,
            vmin=0.0,
            vmax=resolved_vmax,
            cmap=cmap,
            dim=0,
            black_bg=False,
            symmetric_cbar=False,
            annotate=False,
            radiological=radiological,
            figure=figure,
            axes=rect,
        )

    with plot_context():
        return mosaic_figure(
            draw,
            reference_img=img,
            mask_img=mask_img,
            n_cuts=n_cuts,
            title=title,
            radiological=radiological,
            colorbar=ColorbarSpec(
                cmap=cmap, vmin=0.0, vmax=resolved_vmax, label=cbar_label
            ),
            provenance=[
                f"n = {positive.size:,} voxels",
                *extra_provenance,
                f"scale 0–{resolved_vmax:.3g} "
                f"({float(np.mean(positive > resolved_vmax)):.1%} clipped)"
                + (f", from {limit_source}" if limit_source else ""),
                "unsigned magnitude: the scale starts at zero and is not symmetric",
                orientation_label(radiological),
            ],
        )


def dual_coded_mosaic(
    effect_img: Any,
    *,
    stat_img: Any,
    bg_img: Any = None,
    mask_img: Any = None,
    threshold: float,
    vmax: Optional[float] = None,
    two_sided: bool = True,
    radiological: bool = False,
    title: str = "",
    cbar_label: str = "effect",
    n_cuts: int = DEFAULT_CUTS_PER_ROW,
) -> Any:
    """Draw effect magnitude in hue and statistical evidence in opacity.

    A binary threshold discards everything beneath it and shows the survivors as
    though the cut-off were a fact about the brain rather than a choice. Dual coding
    (Allen, Erhardt & Calhoun 2012) keeps the whole map: colour carries the effect,
    opacity carries the evidence, and a region just under the threshold fades rather
    than vanishing. A reader can then distinguish "nothing there" from "nearly
    there", which a thresholded panel makes impossible.

    Deliberately passes no ``threshold`` to nilearn: thresholding here would undo
    the point. The opacity ramp runs from half the threshold (fully transparent) to
    the threshold itself (fully opaque), so the boundary the cluster table uses is
    still legible as the point where the map becomes solid.
    """
    from nilearn import plotting

    effect_data = np.asarray(effect_img.get_fdata())
    stat_shape = np.asarray(stat_img.get_fdata()).shape
    if stat_shape != effect_data.shape:
        raise ValueError(
            "Dual coding requires the effect and statistic maps to share a shape; got "
            f"{effect_data.shape} and {stat_shape}."
        )

    # Inside the analysis mask, for the same reason stat_map_mosaic is: a percentile
    # over the whole volume is a percentile of a distribution dominated by background
    # zeros. Measured on this study's own effect map, the whole-volume limit was 1.53x
    # too low, so 5.4% of in-brain voxels saturated while the figure reported 2.0%
    # clipped -- a number computed over voxels the panel does not draw.
    values, limit_source = _masked_values(effect_img, mask_img)
    resolved_vmax = float(vmax) if vmax is not None else robust_symmetric_limit(values)
    plotted = apply_sidedness(effect_img, two_sided=two_sided)

    def draw(figure, rect, direction, cuts):
        plotting.plot_stat_map(
            plotted,
            bg_img=bg_img,
            display_mode=direction,
            cut_coords=list(cuts),
            threshold=None,
            transparency=stat_img,
            transparency_range=[0.5 * float(threshold), float(threshold)],
            colorbar=False,
            vmax=resolved_vmax,
            cmap=SIGNED_CMAP,
            dim=0,
            black_bg=False,
            symmetric_cbar=True,
            annotate=False,
            radiological=radiological,
            figure=figure,
            axes=rect,
        )

    with plot_context():
        return mosaic_figure(
            draw,
            reference_img=effect_img,
            mask_img=mask_img,
            n_cuts=n_cuts,
            title=title,
            radiological=radiological,
            # No suppressed band: dual coding hides nothing, which is its whole
            # point. Hatching one here would claim the panel drops what it fades.
            colorbar=ColorbarSpec(
                cmap=SIGNED_CMAP,
                vmin=-resolved_vmax,
                vmax=resolved_vmax,
                label=cbar_label,
            ),
            provenance=[
                f"n = {values.size:,} voxels",
                f"hue: effect · opacity: |z| ramped "
                f"{0.5 * float(threshold):.2f}–{float(threshold):.2f}",
                f"colour limit ±{resolved_vmax:.3g} "
                f"({clipped_fraction(values, limit=resolved_vmax):.1%} clipped)"
                + (f", from {limit_source}" if limit_source else ""),
                orientation_label(radiological),
            ],
        )


#: Which two world axes each glass-brain projection preserves.
_PROJECTION_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1), "l": (1, 2), "r": (1, 2)}


#: Colour for peak markers on a glass brain.
#:
#: Deliberately outside the diverging ramp the projection is drawn in. Marked in the
#: neutral guide grey, the markers were invisible against a dense single-subject
#: projection -- which is most of them, since a glass brain projects the maximum along
#: each axis and an uncorrected height fills it. Green is in neither half of RdBu_r,
#: so it cannot be mistaken for a value.
PEAK_MARKER_COLOR = OKABE_ITO["bluish_green"]


def _annotate_peaks(
    display: Any,
    peak_coords: Sequence[Tuple[float, float, float]],
    peak_labels: Optional[Sequence[str]] = None,
) -> None:
    """Mark each peak and write its label beside it, on every projection.

    The label is the point. A dot alone tells a reader that something is there,
    which the map already showed; the number is what lets a cluster in the table be
    located in the projection. Previously the index reached only a debug log, so the
    caption promised a key the figure did not carry.

    Both the marker and its label carry a white halo. A glass brain fills with colour
    wherever anything survives the threshold, and an unhaloed grey marker on top of it
    is not findable -- which costs the caption's promise a second time.
    """
    from matplotlib import patheffects

    labels = (
        [str(label) for label in peak_labels]
        if peak_labels is not None
        else [str(i) for i in range(1, len(peak_coords) + 1)]
    )
    if len(labels) != len(peak_coords):
        raise ValueError(
            f"Got {len(labels)} peak labels for {len(peak_coords)} coordinates."
        )

    display.add_markers(
        [tuple(c) for c in peak_coords],
        marker_color=PEAK_MARKER_COLOR,
        marker_size=26,
        marker="o",
        edgecolors="white",
        linewidths=0.8,
    )

    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    for direction, projection in display.axes.items():
        pair = _PROJECTION_AXES.get(str(direction))
        if pair is None:
            logger.debug("No projection mapping for direction %r", direction)
            continue
        first, second = pair
        for label, coord in zip(labels, peak_coords):
            projection.ax.annotate(
                label,
                xy=(float(coord[first]), float(coord[second])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                color="#111111",
                annotation_clip=False,
                path_effects=halo,
            )


def glass_brain(
    stat_img: Any,
    *,
    mask_img: Any = None,
    peak_labels: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
    vmax: Optional[float] = None,
    two_sided: bool = True,
    radiological: bool = False,
    title: str = "",
    cbar_label: str = "z",
    peak_coords: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Any:
    """Draw a glass-brain projection of ``stat_img``.

    ``plot_abs=False`` is not optional. Nilearn defaults it to ``True``, which
    projects the absolute value: activation and deactivation then render
    identically, and the panel contradicts the signed mosaic beside it.

    ``peak_coords`` annotates each peak with its 1-based index so the projection can
    be read against the cluster table.
    """
    from nilearn import plotting

    values, limit_source = _masked_values(stat_img, mask_img)
    resolved_vmax = _resolve_vmax(
        stat_img, threshold=threshold, vmax=vmax, mask_img=mask_img
    )
    with plot_context():
        # Laid out here for the same reasons as the mosaics: nilearn draws its title
        # inside the axes, where on a glass brain it lands on the sagittal projection,
        # and its colourbar sits at the figure edge with the ticks clipped.
        figure = plt.figure(figsize=(9.2, 3.5))
        figure.patch.set_facecolor("white")
        display = plotting.plot_glass_brain(
            apply_sidedness(stat_img, two_sided=two_sided),
            threshold=float(threshold) if threshold else None,
            colorbar=False,
            vmax=resolved_vmax,
            cmap=SIGNED_CMAP,
            plot_abs=False,
            symmetric_cbar=True,
            black_bg=False,
            radiological=radiological,
            figure=figure,
            axes=(0.02, 0.06, 0.84, 0.84),
        )
        if peak_coords:
            _annotate_peaks(display, peak_coords, peak_labels)

        draw_colorbar(
            figure,
            ColorbarSpec(
                cmap=SIGNED_CMAP,
                vmin=-resolved_vmax,
                vmax=resolved_vmax,
                label=cbar_label,
                suppressed=suppressed_band(threshold, two_sided=two_sided),
            ),
            rect=(0.905, 0.16, 0.016, 0.66),
        )
        if title:
            figure.text(
                0.02, 0.965, title, ha="left", va="center", fontsize=11,
                fontweight="bold",
            )
        annotate_provenance(
            figure,
            _provenance(
                values,
                threshold=threshold,
                limit=resolved_vmax,
                two_sided=two_sided,
                radiological=radiological,
                limit_source=limit_source,
            )
            + [
                # A glass brain projects the maximum along each axis, so it fills
                # wherever anything survives -- and on an uncorrected single-subject
                # height that is most of the brain. Said outright, because a saturated
                # projection otherwise reads as a very strong result.
                "maximum-intensity projection: a voxel anywhere along a ray fills it"
            ],
        )
        return figure


__all__ = [
    "apply_sidedness",
    "dual_coded_mosaic",
    "glass_brain",
    "magnitude_mosaic",
    "stat_map_mosaic",
]
