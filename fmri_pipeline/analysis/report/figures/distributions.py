"""Distribution panels: what a height threshold on this map is actually worth.

The panel here answers a question a thresholded brain picture cannot. A z map is
nominally N(0, 1) under the null, and the threshold applied to it is chosen against
that assumption -- but a single-subject GLM with unmodelled autocorrelation and
physiological noise is routinely over-dispersed, and nothing in a thresholded mosaic
reveals it. Drawing the map's own fitted null beside the theoretical one puts the
discrepancy on the same axis as the threshold, where it can be read directly.

The corrected thresholds share the axis for the same reason. Uncorrected, FDR and
Bonferroni are three points on one scale; separating them into a table makes the
reader do the comparison by arithmetic.

Every threshold is stated twice: what it is worth under N(0, 1), and what it is worth
under the null this map actually has. Drawing the fitted null and leaving the survivor
counts theoretical asks the reader to integrate a normal tail by eye off a log axis,
and the two answers differ by nearly an order of magnitude on real data.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.inference import ThresholdContext
from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)

_BINS = 160

#: Colour and dash for each threshold line. Distinct on both counts, so the panel
#: survives greyscale printing and the colour-blind readership the palette is chosen
#: for.
_THRESHOLD_STYLE = {
    "applied": (OKABE_ITO["vermillion"], (0, (4, 2))),
    "fdr": (OKABE_ITO["bluish_green"], (0, (5, 1, 1, 1))),
    "empirical_fdr": (OKABE_ITO["reddish_purple"], (0, (6, 1, 2, 1, 2, 1))),
    "bonferroni": (OKABE_ITO["blue"], (0, (1, 2))),
}


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("A distribution panel requires at least one finite value.")
    return finite


def _normal_counts(
    centres: np.ndarray, *, n: int, bin_width: float, centre: float, scale: float
) -> np.ndarray:
    """Expected per-bin counts if every one of ``n`` voxels were drawn from the null."""
    z = (centres - centre) / scale
    density = np.exp(-0.5 * z**2) / (scale * np.sqrt(2.0 * np.pi))
    return float(n) * float(bin_width) * density


def _symmetric(threshold: Optional[float], *, two_sided: bool) -> Tuple[float, ...]:
    """Where a single height falls on the axis, given the sidedness of the test."""
    if threshold is None:
        return ()
    return (threshold, -threshold) if two_sided else (threshold,)


def _threshold_entries(
    context: ThresholdContext,
) -> List[Tuple[str, Tuple[float, ...], str]]:
    """Name each threshold, where it falls on the z axis, and what survives it.

    Positions are a tuple rather than one height because a rejection region need not be
    symmetric: the empirical-null FDR bounds sit at different distances from zero
    whenever the fitted null is shifted, and collapsing them to a single ``|z| >``
    figure would reintroduce exactly the error that correction removes.

    An entry whose region is empty keeps its label and draws no line. Dropping it would
    make "no voxel survives correction" -- which is a finding -- indistinguishable from
    a panel that failed to draw it.
    """
    comparison = "|z|" if context.two_sided else "z"
    calibration = context.calibration
    entries: List[Tuple[str, Tuple[float, ...], str]] = []

    if context.applied is None:
        # threshold_mode: none. The corrected heights below still belong on the axis:
        # an unthresholded map is the one whose reader most needs to know where a
        # threshold would have fallen.
        entries.append(("applied", (), "no height threshold applied"))
    else:
        # Both expectations, on one line, because the comparison between them is the
        # reading. The theoretical count alone let a survivor count that its own map's
        # noise fully explains read as eightfold enrichment.
        expected = f"{context.expected_null_survivors:,.0f} expected under N(0, 1)"
        if calibration is not None and calibration.expected_survivors is not None:
            expected += f"; {calibration.expected_survivors:,.0f} under the fitted null"
        entries.append(
            (
                "applied",
                _symmetric(context.applied, two_sided=context.two_sided),
                f"applied {comparison} > {context.applied:.2f}: "
                f"{context.applied_survivors:,} voxels ({expected})",
            )
        )

    if context.fdr is None:
        entries.append(
            (
                "fdr",
                (),
                f"FDR q = {context.fdr_q:g} vs N(0, 1): no voxel survives correction",
            )
        )
    else:
        entries.append(
            (
                "fdr",
                _symmetric(context.fdr, two_sided=context.two_sided),
                f"FDR q = {context.fdr_q:g} vs N(0, 1) at {comparison} > "
                f"{context.fdr:.2f}: {context.fdr_survivors:,} voxels",
            )
        )

    if calibration is not None:
        bounds = tuple(
            bound
            for bound in (calibration.fdr_upper, calibration.fdr_lower)
            if bound is not None
        )
        if bounds:
            region = " or ".join(
                part
                for part in (
                    None if calibration.fdr_upper is None else f"z > {calibration.fdr_upper:.2f}",
                    None if calibration.fdr_lower is None else f"z < {calibration.fdr_lower:.2f}",
                )
                if part
            )
            label = (
                f"FDR q = {context.fdr_q:g} vs the fitted null at {region}: "
                f"{calibration.fdr_survivors:,} voxels"
            )
        else:
            label = (
                f"FDR q = {context.fdr_q:g} vs the fitted null: no voxel survives "
                "correction"
            )
        entries.append(("empirical_fdr", bounds, label))

    entries.append(
        (
            "bonferroni",
            _symmetric(context.bonferroni, two_sided=context.two_sided),
            f"Bonferroni {context.alpha:g} at {comparison} > {context.bonferroni:.2f}: "
            f"{context.bonferroni_survivors:,} voxels",
        )
    )
    return entries


def _tail_provenance(context: ThresholdContext) -> Sequence[str]:
    """State what the applied height is worth in each tail of the fitted null.

    The nominal p value of a symmetric ``|z| > c`` cut is one number, and it is the
    right number only when the fitted null is centred on zero. Shifted, the same cut
    buys different evidence in each direction -- measured here, p = 0.027 upward and
    p = 0.13 downward against a nominal 0.021 -- and a reader given one figure has no
    way to see that the negative clusters on the map above are the weaker half.
    """
    calibration = context.calibration
    if calibration is None or calibration.upper_tail_p is None:
        return ()
    nominal = (
        f"nominal p = "
        f"{(2.0 if context.two_sided else 1.0) * float(_normal_sf(context.applied)):.3g}"
        if context.applied is not None
        else ""
    )
    if calibration.lower_tail_p is None:
        line = f"against the fitted null: p = {calibration.upper_tail_p:.3g}"
    else:
        line = (
            f"against the fitted null: p = {calibration.upper_tail_p:.3g} upward, "
            f"{calibration.lower_tail_p:.3g} downward"
        )
    return (f"{line} ({nominal})",) if nominal else (line,)


def _normal_sf(threshold: float) -> float:
    from scipy import stats

    return float(stats.norm.sf(float(threshold)))


def null_calibration_figure(
    values: np.ndarray,
    *,
    context: ThresholdContext,
    mask_source: str = "",
    title: str = "",
) -> plt.Figure:
    """Draw the in-mask z distribution against both nulls and all three thresholds.

    ``values`` must already be restricted to the analysis mask. A whole volume is
    mostly background zeros -- over 60% of a typical map -- which would put a spike at
    the origin, inflate the test count behind every corrected threshold, and drag the
    fitted null toward zero.

    Both null curves are scaled to the full voxel count, which is exact only if no
    voxel is active. The figure says so. The diagnostic this panel exists for is the
    *width* of the fitted null against the theoretical one, and that comparison does
    not depend on the scaling.

    The y axis is logarithmic because the question lives in the tails; on a linear axis
    the null peak is the only visible feature.
    """
    finite = _finite(values)
    entries = _threshold_entries(context)

    with plot_context():
        figure, axis = plt.subplots(figsize=(7.6, 4.0))
        counts, edges, _ = axis.hist(
            finite, bins=_BINS, color="0.78", edgecolor="none", label="observed"
        )
        bin_width = float(edges[1] - edges[0])

        # Curves span the axis rather than only the data, so a null stays legible where
        # it predicts counts the map does not contain -- which is exactly the region
        # the corrected thresholds sit in.
        drawn = [abs(position) for _key, positions, _label in entries for position in positions]
        reach = max([float(np.max(np.abs(finite)))] + drawn) * 1.08
        span = np.linspace(-reach, reach, 512)

        axis.plot(
            span,
            _normal_counts(span, n=finite.size, bin_width=bin_width, centre=0.0, scale=1.0),
            color=GUIDE_COLOR,
            linewidth=1.6,
            label="theoretical N(0, 1)",
        )
        if context.null is not None:
            axis.plot(
                span,
                _normal_counts(
                    span,
                    n=finite.size,
                    bin_width=bin_width,
                    centre=context.null.centre,
                    scale=context.null.scale,
                ),
                color=OKABE_ITO["orange"],
                linewidth=1.6,
                label=(
                    f"empirical null N({context.null.centre:+.2f}, "
                    f"{context.null.scale:.2f}²)"
                ),
            )

        for key, positions, label in entries:
            colour, dashes = _THRESHOLD_STYLE[key]
            if not positions:
                # Carries the finding into the legend without drawing a line at a
                # height nothing reached.
                axis.plot([], [], color=colour, linestyle=dashes, linewidth=1.3, label=label)
                continue
            for index, position in enumerate(positions):
                # One legend entry describes the whole region; the remaining bounds are
                # drawn unlabelled.
                axis.axvline(
                    position,
                    color=colour,
                    linestyle=dashes,
                    linewidth=1.3,
                    label=label if index == 0 else None,
                )

        axis.set_xlim(-reach, reach)
        axis.set_yscale("log")
        positive = counts[counts > 0]
        if positive.size:
            axis.set_ylim(bottom=max(0.5, float(np.min(positive)) * 0.5))
        axis.set_xlabel("z")
        axis.set_ylabel("voxels")
        if title:
            axis.set_title(title)
        # Below the axes rather than inside it. The threshold lines are vertical and
        # span the full height, so any in-axes legend is crossed by the very lines it
        # describes -- and the entries carry the survivor counts, which is most of what
        # this panel says.
        axis.legend(
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            framealpha=0.0,
        )

        provenance = [
            f"n = {finite.size:,} voxels" + (f" ({mask_source})" if mask_source else ""),
            "both null curves assume every voxel is null",
            *_tail_provenance(context),
            "null fitted by median and MAD (robust to a signal tail)",
        ]
        annotate_provenance(figure, provenance)
        figure.tight_layout()
        return figure


__all__ = ["null_calibration_figure"]
