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

from html import escape
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


def _escape(value: object) -> str:
    """Escape a cell before it is interpolated into markup.

    Cells carry values from the data: a rejection region reads "z < -6.57",
    and a regressor name is whatever the design called it. Interpolated raw,
    the first of those opened a tag and swallowed the cell that contained it.
    """
    return escape("" if value is None else str(value))

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


#: Short name per threshold, for the axis. The full statement is the table's job.
_THRESHOLD_SHORT = {
    "applied": "applied",
    "fdr": "FDR vs N(0,1)",
    "empirical_fdr": "FDR vs fitted",
    "bonferroni": "Bonferroni",
}


def _region(positions: Sequence[float], *, comparison: str) -> str:
    """Describe a rejection region in one cell.

    A region need not be symmetric: the empirical-null bounds sit at different
    distances from zero whenever the fitted null is shifted, and collapsing them to a
    single ``|z| >`` figure reintroduces exactly the error the correction removes.
    """
    if not positions:
        return "none survives"
    values = sorted(float(p) for p in positions)
    if len(values) == 2 and abs(values[0] + values[1]) < 1e-9:
        return f"{comparison} > {values[1]:.2f}"
    return " or ".join(
        f"z {'<' if value < 0 else '>'} {value:.2f}" for value in values
    )


def threshold_table(context: ThresholdContext) -> Tuple[str, List[str]]:
    """Every threshold's height and what survives it, as a table.

    These numbers used to live in the figure's legend, as four sentences occupying a
    third of the canvas -- a results table set in 7-point type, wrapped in a legend
    box, beside the lines it described. Counts belong in a table; the figure keeps the
    distribution and the positions, which are the things only a picture shows.

    Both expectations are columns because the comparison between them is the reading:
    a survivor count that its own map's noise fully explains reads as eightfold
    enrichment against N(0, 1) alone.
    """
    comparison = "|z|" if context.two_sided else "z"
    calibration = context.calibration

    headers = [
        "Threshold",
        "Rejection region",
        "Voxels surviving",
        "Expected under N(0,1)",
        "Expected under fitted null",
    ]
    rows: List[List[str]] = []

    def _count(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{float(value):,.0f}"

    if context.applied is None:
        rows.append(["Applied", "no height applied", "n/a", "n/a", "n/a"])
    else:
        rows.append(
            [
                "Applied",
                _region(
                    _symmetric(context.applied, two_sided=context.two_sided),
                    comparison=comparison,
                ),
                f"{context.applied_survivors:,}",
                _count(context.expected_null_survivors),
                _count(
                    calibration.expected_survivors if calibration is not None else None
                ),
            ]
        )

    rows.append(
        [
            f"FDR q = {context.fdr_q:g} vs N(0,1)",
            _region(
                _symmetric(context.fdr, two_sided=context.two_sided)
                if context.fdr is not None
                else (),
                comparison=comparison,
            ),
            f"{context.fdr_survivors:,}" if context.fdr is not None else "0",
            "n/a",
            "n/a",
        ]
    )

    if calibration is not None:
        bounds = tuple(
            bound
            for bound in (calibration.fdr_upper, calibration.fdr_lower)
            if bound is not None
        )
        rows.append(
            [
                f"FDR q = {context.fdr_q:g} vs fitted null",
                _region(bounds, comparison=comparison),
                f"{calibration.fdr_survivors:,}" if bounds else "0",
                "n/a",
                "n/a",
            ]
        )

    rows.append(
        [
            f"Bonferroni {context.alpha:g}",
            _region(
                _symmetric(context.bonferroni, two_sided=context.two_sided),
                comparison=comparison,
            ),
            f"{context.bonferroni_survivors:,}",
            "n/a",
            "n/a",
        ]
    )

    # Last, because it is the only row whose null is the data's own. Both expectation
    # columns are "n/a" for it by construction: the height comes from a permutation
    # distribution rather than from a fitted or theoretical one, so there is no
    # closed-form count to expect under either.
    sign_flip = context.sign_flip
    if sign_flip is not None:
        rows.append(
            [
                f"Run sign-flip, FWE {context.alpha:g}",
                _region(
                    _symmetric(sign_flip.height, two_sided=context.two_sided),
                    comparison=comparison,
                ),
                f"{sign_flip.survivors:,}",
                "n/a",
                "n/a",
            ]
        )

    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    tsv = ["\t".join(headers)]
    tsv.extend("\t".join(row) for row in rows)
    return table_html, tsv


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

        # Each threshold is named on the axis at its own line, upward. The survivor
        # counts that used to ride in the legend are in the table beside this figure:
        # four sentences of 7-point type in a legend box is a results table drawn in
        # the wrong medium, and it took a third of the canvas.
        for key, positions, _label in entries:
            colour, dashes = _THRESHOLD_STYLE[key]
            if not positions:
                continue
            for index, position in enumerate(positions):
                axis.axvline(
                    position, color=colour, linestyle=dashes, linewidth=1.2, zorder=3
                )
                if index == 0:
                    axis.annotate(
                        _THRESHOLD_SHORT.get(key, key),
                        xy=(position, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(-3, -3),
                        textcoords="offset points",
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=6.5,
                        color=colour,
                    )

        # Trimmed to the data plus the outermost threshold. Spanning the curves' full
        # reach left a third of the axis empty on both sides, which on a log count axis
        # is a third of the panel showing nothing.
        edge = max([float(np.max(np.abs(finite)))] + drawn) * 1.04
        axis.set_xlim(-edge, edge)
        axis.set_yscale("log")
        positive = counts[counts > 0]
        if positive.size:
            axis.set_ylim(bottom=max(0.5, float(np.min(positive)) * 0.5))
        axis.set_xlabel("z")
        axis.set_ylabel("voxels")
        if title:
            axis.set_title(title)
        # Three entries: the observed distribution and the two nulls it is read
        # against. Placed inside, upper left, where the thresholds do not cross it.
        axis.legend(fontsize=7, loc="upper left", framealpha=0.0)

        provenance = [
            f"n = {finite.size:,} voxels" + (f" ({mask_source})" if mask_source else ""),
            "both null curves assume every voxel is null",
            *_tail_provenance(context),
            "null fitted by median and MAD (robust to a signal tail)",
        ]
        annotate_provenance(figure, provenance)
        figure.tight_layout()
        return figure


#: Voxels drawn on the effect-versus-evidence panel.
#:
#: A 50,000-voxel mask plotted point by point is a solid block of ink that hides its
#: own density, and the file it produces dominates the report's size. A random sample
#: of this many, drawn from a fixed seed so the panel is reproducible, shows the same
#: shape.
_SCATTER_SAMPLE = 12_000


def effect_versus_evidence_figure(
    effect: np.ndarray,
    stat: np.ndarray,
    *,
    standard_error: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    two_sided: bool = True,
    effect_units: str = "effect",
    title: str = "",
    seed: int = 0,
) -> plt.Figure:
    """Effect magnitude against statistical evidence, voxel by voxel.

    A thresholded map answers "where is the evidence" and says nothing about how large
    the effects are, which is the quantity a result is actually reported in. The two
    come apart in both directions: a small effect measured precisely clears any height
    threshold, and a large effect measured in a dropout region does not. Neither case
    is visible in a z map, and neither is visible in an effect map either -- only in
    the two together.

    The panel is therefore read for its *shape*. A cloud whose suprathreshold voxels
    sit at small effects says the surviving voxels survived on precision; a cloud whose
    large effects fall below threshold says the model is underpowered where the effect
    is biggest. Colour carries the standard error, which is what separates the two.

    No verdict is drawn. The threshold is marked because it is what the maps beside
    this panel were drawn at, not because a voxel's position relative to it is a
    finding.
    """
    effect = np.asarray(effect, dtype=float).ravel()
    stat_values = np.asarray(stat, dtype=float).ravel()
    if effect.size != stat_values.size:
        raise ValueError(
            f"Effect and statistic must describe the same voxels; got {effect.size} "
            f"and {stat_values.size}."
        )

    error = None
    if standard_error is not None:
        candidate = np.asarray(standard_error, dtype=float).ravel()
        if candidate.size == effect.size:
            error = candidate

    # Filtered jointly. Dropping each array's own non-finite values independently
    # would leave the three of different lengths and pair every voxel after the first
    # gap with a different voxel's statistic -- a scatter that looks entirely normal
    # and is scrambled.
    usable = np.isfinite(effect) & np.isfinite(stat_values)
    if error is not None:
        usable &= np.isfinite(error)
    effect, stat_values = effect[usable], stat_values[usable]
    if error is not None:
        error = error[usable]

    if effect.size == 0:
        raise ValueError("The effect-versus-evidence panel requires at least one voxel.")

    rng = np.random.default_rng(seed)
    if effect.size > _SCATTER_SAMPLE:
        picked = rng.choice(effect.size, size=_SCATTER_SAMPLE, replace=False)
    else:
        picked = np.arange(effect.size)

    # Signed, not folded. ``z = effect / SE`` with SE > 0, so a voxel's statistic
    # always carries its effect's sign: plotting |z| against a signed effect encodes
    # that sign twice and mirrors the negative branch onto the positive axis. The
    # asymmetry between the tails then has to be judged by comparing two lobes across
    # the axis -- and on a map whose fitted null is off-centre, that asymmetry is the
    # first thing worth seeing.
    x = stat_values[picked]
    y = effect[picked]

    with plot_context():
        figure, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)

        colour_note = ""
        if error is not None:
            colours = error[picked]
            # A robust upper limit, as everywhere else in these figures. Scaled to the
            # maximum, a handful of dropout voxels with very large standard errors
            # take the top of the ramp and every remaining point lands in its bottom
            # tenth -- the colour then separates nothing, which is the one thing this
            # channel is here to do.
            limit = float(np.percentile(colours, 98.0))
            if not np.isfinite(limit) or limit <= 0:
                limit = float(np.max(colours)) or 1.0
            points = ax.scatter(
                x,
                y,
                c=colours,
                s=2.0,
                alpha=0.45,
                cmap="cividis",
                linewidths=0,
                vmin=float(np.min(colours)),
                vmax=limit,
            )
            bar = figure.colorbar(points, ax=ax, fraction=0.04, pad=0.02, extend="max")
            bar.set_label(f"standard error ({effect_units})")
            clipped = float(np.mean(colours > limit))
            colour_note = f"colour limit {limit:.3g} ({clipped:.1%} clipped)"
        else:
            ax.scatter(
                x, y, s=2.0, alpha=0.35, color=OKABE_ITO["blue"], linewidths=0
            )

        # Zero effect, so the vertical spread reads against the value that means
        # "no difference" rather than against the middle of the cloud.
        ax.axhline(0.0, color=GUIDE_COLOR, linewidth=0.8, zorder=1)
        if threshold and threshold > 0:
            # Both tails when the test is two-sided, because both were cut. One line
            # over a signed axis would show half the rejection region and imply the
            # other half was never tested.
            heights = (
                (-float(threshold), float(threshold))
                if two_sided
                else (float(threshold),)
            )
            for height in heights:
                ax.axvline(
                    height,
                    color=OKABE_ITO["vermillion"],
                    linestyle=(0, (4, 2)),
                    linewidth=1.1,
                    zorder=2,
                )
            label = (
                f"drawn at |z| > {float(threshold):.2f}"
                if two_sided
                else f"drawn at z > {float(threshold):.2f}"
            )
            ax.annotate(
                label,
                xy=(max(heights), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=7,
                color=OKABE_ITO["vermillion"],
            )

        ax.set_xlabel("z (evidence)")
        ax.set_ylabel(f"effect ({effect_units})")
        if title:
            ax.set_title(title)

        lines = [
            f"{effect.size:,} voxels in the mask",
            f"{picked.size:,} drawn"
            + (" (random sample)" if picked.size < effect.size else ""),
        ]
        if colour_note:
            lines.append(colour_note)
        if threshold and threshold > 0:
            surviving = int(np.count_nonzero(np.abs(stat_values) > float(threshold)))
            if surviving:
                median_effect = float(
                    np.median(np.abs(effect[np.abs(stat_values) > float(threshold)]))
                )
                lines.append(
                    f"median |effect| above the threshold: {median_effect:.3g}"
                )
            lines.append(f"{surviving:,} voxels above the threshold")
        lines.append("no voxel is scored against a criterion here")
        annotate_provenance(figure, lines)
        return figure


__all__ = [
    "effect_versus_evidence_figure",
    "null_calibration_figure",
    "threshold_table",
]
