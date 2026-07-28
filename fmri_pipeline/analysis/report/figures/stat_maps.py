"""Slice mosaic and glass-brain panels for a statistical map."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    SIGNED_CMAP,
    annotate_provenance,
    clipped_fraction,
    plot_context,
    robust_symmetric_limit,
    suprathreshold_limit,
)

logger = logging.getLogger(__name__)


def _masked_values(stat_img: Any) -> np.ndarray:
    data = np.asarray(stat_img.get_fdata())
    return data[np.isfinite(data)]


def _resolve_vmax(
    stat_img: Any, *, threshold: Optional[float], vmax: Optional[float]
) -> float:
    """Choose a colour limit appropriate to whether the panel is thresholded.

    A thresholded panel takes its limit from the surviving voxels only. Reusing the
    whole-map limit is what makes these panels saturate: for z ~ N(0,1) the robust
    limit is about 2.58 against a typical 2.3 threshold, leaving no usable range.
    """
    if vmax is not None:
        return float(vmax)
    values = _masked_values(stat_img)
    if threshold is not None and threshold > 0:
        return suprathreshold_limit(values, threshold=float(threshold))
    return robust_symmetric_limit(values)


def _figure_of(display: Any) -> Any:
    figure = getattr(display, "figure", None) or getattr(display, "_fig", None)
    if figure is None and hasattr(display, "frame_axes"):
        figure = getattr(display.frame_axes, "figure", None)
    if figure is None:
        raise RuntimeError(
            "Could not resolve a Matplotlib figure from the nilearn display."
        )
    return figure


def _label_colorbar(display: Any, label: str) -> None:
    """Name the units on the colorbar. Nilearn exposes no parameter for this."""
    colorbar = getattr(display, "_cbar", None)
    if colorbar is None:
        logger.debug("Nilearn display exposed no colorbar to label.")
        return
    colorbar.set_label(label, rotation=90, labelpad=6)


def _orientation_label(radiological: bool) -> str:
    return "radiological (R on left)" if radiological else "neurological (L on left)"


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
) -> List[str]:
    """Build the self-description line for a map panel."""
    lines = [f"n = {values.size:,} voxels"]
    if threshold:
        comparison = "|z|" if two_sided else "z"
        lines.append(f"{comparison} > {float(threshold):.2f}")
    else:
        lines.append("unthresholded")
    fraction = clipped_fraction(values, limit=limit)
    lines.append(f"colour limit ±{limit:.2f} ({fraction:.1%} clipped)")
    # A brain figure that does not state its convention cannot be checked, and a
    # left/right error is not visible in the image.
    lines.append(_orientation_label(radiological))
    return lines


def stat_map_mosaic(
    stat_img: Any,
    *,
    bg_img: Any = None,
    threshold: Optional[float] = None,
    vmax: Optional[float] = None,
    two_sided: bool = True,
    radiological: bool = False,
    title: str = "",
    cbar_label: str = "z",
    cmap: str = SIGNED_CMAP,
) -> Any:
    """Draw a slice mosaic of ``stat_img``.

    ``annotate`` stays on: a mosaic without slice coordinates and left/right markers
    tells a reader neither where a cluster is nor which hemisphere it is in, which
    is most of what the panel exists to say.
    """
    from nilearn import plotting

    values = _masked_values(stat_img)
    resolved_vmax = _resolve_vmax(stat_img, threshold=threshold, vmax=vmax)
    plotted = apply_sidedness(stat_img, two_sided=two_sided)
    with plot_context():
        display = plotting.plot_stat_map(
            plotted,
            bg_img=bg_img,
            title=title or None,
            display_mode="mosaic",
            threshold=float(threshold) if threshold else None,
            colorbar=True,
            vmax=resolved_vmax,
            cmap=cmap,
            dim=0,
            black_bg=False,
            symmetric_cbar=True,
            annotate=True,
            radiological=radiological,
        )
        _label_colorbar(display, cbar_label)
        figure = _figure_of(display)
        annotate_provenance(
            figure,
            _provenance(
                values,
                threshold=threshold,
                limit=resolved_vmax,
                two_sided=two_sided,
                radiological=radiological,
            ),
        )
        return figure


def dual_coded_mosaic(
    effect_img: Any,
    *,
    stat_img: Any,
    bg_img: Any = None,
    threshold: float,
    vmax: Optional[float] = None,
    two_sided: bool = True,
    radiological: bool = False,
    title: str = "",
    cbar_label: str = "effect",
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

    values = effect_data[np.isfinite(effect_data)]
    resolved_vmax = float(vmax) if vmax is not None else robust_symmetric_limit(values)
    with plot_context():
        display = plotting.plot_stat_map(
            apply_sidedness(effect_img, two_sided=two_sided),
            bg_img=bg_img,
            title=title or None,
            display_mode="mosaic",
            threshold=None,
            transparency=stat_img,
            transparency_range=[0.5 * float(threshold), float(threshold)],
            colorbar=True,
            vmax=resolved_vmax,
            cmap=SIGNED_CMAP,
            dim=0,
            black_bg=False,
            symmetric_cbar=True,
            annotate=True,
            radiological=radiological,
        )
        _label_colorbar(display, cbar_label)
        figure = _figure_of(display)
        annotate_provenance(
            figure,
            [
                f"n = {values.size:,} voxels",
                f"hue: effect · opacity: |z| ramped "
                f"{0.5 * float(threshold):.2f}–{float(threshold):.2f}",
                f"colour limit ±{resolved_vmax:.3g} "
                f"({clipped_fraction(values, limit=resolved_vmax):.1%} clipped)",
                _orientation_label(radiological),
            ],
        )
        return figure


def glass_brain(
    stat_img: Any,
    *,
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

    values = _masked_values(stat_img)
    resolved_vmax = _resolve_vmax(stat_img, threshold=threshold, vmax=vmax)
    with plot_context():
        display = plotting.plot_glass_brain(
            apply_sidedness(stat_img, two_sided=two_sided),
            title=title or None,
            threshold=float(threshold) if threshold else None,
            colorbar=True,
            vmax=resolved_vmax,
            cmap=SIGNED_CMAP,
            plot_abs=False,
            symmetric_cbar=True,
            black_bg=False,
            radiological=radiological,
        )
        _label_colorbar(display, cbar_label)
        if peak_coords:
            for index, coord in enumerate(peak_coords, start=1):
                display.add_markers(
                    [tuple(coord)], marker_color=GUIDE_COLOR, marker_size=18, marker="o"
                )
                logger.debug("Annotated peak %d at %s", index, coord)
        figure = _figure_of(display)
        annotate_provenance(
            figure,
            _provenance(
                values,
                threshold=threshold,
                limit=resolved_vmax,
                two_sided=two_sided,
                radiological=radiological,
            ),
        )
        return figure


__all__ = ["apply_sidedness", "dual_coded_mosaic", "glass_brain", "stat_map_mosaic"]
