"""Where in the brain a result actually lives.

A BOLD effect is a grey-matter phenomenon. A map whose suprathreshold voxels sit
disproportionately in white matter, in the ventricles, or around the brain edge is
showing something other than neural activity -- residual motion, a coregistration
shift, a pulsatility artefact -- and it reaches the cluster table looking exactly like
a result. Nothing else in this report distinguishes them: the mosaics show anatomy
under the blobs but leave the reader to judge the overlap by eye across twenty-one
tiles, and the cluster table names coordinates, not tissue.

The comparison is deliberately relative. Grey matter should carry more of the effect
than white matter, and by how much depends on the contrast, the smoothing, and the
segmentation's own accuracy at this resolution -- so the panel reports the enrichment
it measured and attaches no criterion to it. What a reader takes from it is the
*ordering* and its size, which is a comparison no single number supports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)

logger = logging.getLogger(__name__)

#: Tissue classes, in the order :func:`carpet.resolve_tissue_codes` codes them.
TISSUE_NAMES: Tuple[str, ...] = ("GM", "WM", "CSF")

#: One colour per class, distinct in the Okabe-Ito palette and reused from the
#: carpet's own ordering so the two panels name the same thing the same way.
TISSUE_COLORS = (
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
)


@dataclass(frozen=True)
class TissueSlice:
    """One tissue class's statistic values inside the analysis mask."""

    name: str
    values: np.ndarray

    @property
    def n_voxels(self) -> int:
        return int(self.values.size)

    def surviving(self, threshold: float, *, two_sided: bool) -> int:
        """Voxels of this class above the display threshold."""
        compared = np.abs(self.values) if two_sided else self.values
        return int(np.count_nonzero(compared > float(threshold)))


def split_by_tissue(
    values_img: Any,
    *,
    tissue_codes: np.ndarray,
    mask_img: Any = None,
) -> List[TissueSlice]:
    """Split a statistic map's in-mask voxels by tissue class.

    ``tissue_codes`` is the volume-shaped code array
    :func:`~fmri_pipeline.analysis.report.figures.carpet.resolve_tissue_codes`
    returns: 1, 2, 3 for grey, white, and CSF, and 0 for unclassified.
    """
    data = np.asarray(values_img.get_fdata())
    if tissue_codes.shape != data.shape:
        raise ValueError(
            f"Tissue codes of shape {tissue_codes.shape} do not fit a map of "
            f"shape {data.shape}."
        )

    inside = np.isfinite(data)
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)
        if mask.shape == data.shape:
            inside &= mask
        else:
            logger.warning(
                "Analysis mask of shape %s does not fit the map's %s; splitting over "
                "every finite voxel instead.",
                mask.shape,
                data.shape,
            )

    slices: List[TissueSlice] = []
    for index, name in enumerate(TISSUE_NAMES, start=1):
        selected = inside & (tissue_codes == index)
        if not selected.any():
            continue
        slices.append(TissueSlice(name=name, values=data[selected]))
    return slices


def enrichment(
    slices: Sequence[TissueSlice], *, threshold: float, two_sided: bool
) -> List[Tuple[str, float, int, int]]:
    """Per class: name, share surviving, survivors, and the class's voxel count.

    The share rather than the count. Grey matter is the largest class in any brain
    mask, so a raw count puts it first whatever the map does, and the number a reader
    needs is the rate within each class.
    """
    out: List[Tuple[str, float, int, int]] = []
    for item in slices:
        survivors = item.surviving(threshold, two_sided=two_sided)
        share = survivors / item.n_voxels if item.n_voxels else 0.0
        out.append((item.name, float(share), survivors, item.n_voxels))
    return out


def _mark_offscale_threshold(axis: plt.Axes, position: float, stat_label: str) -> None:
    """Name a threshold that falls outside the drawn range, at the edge it left by.

    Clipping the axis to the data must not silently drop the threshold: a reader has
    to be able to tell that the rejection region lies beyond the view rather than that
    no threshold was applied.
    """
    at_right = position > 0
    axis.annotate(
        f"|{stat_label}| > {abs(position):.2f} \u2192" if at_right else f"\u2190 {abs(position):.2f}",
        xy=(1.0 if at_right else 0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-4 if at_right else 4, -4),
        textcoords="offset points",
        ha="right" if at_right else "left",
        va="top",
        fontsize=6.5,
        color=GUIDE_COLOR,
    )


def tissue_distribution_figure(
    slices: Sequence[TissueSlice],
    *,
    threshold: Optional[float],
    two_sided: bool = True,
    tissue_source: str = "",
    stat_label: str = "z",
    title: str = "",
) -> plt.Figure:
    """Draw the statistic's distribution per tissue class, and what survives in each.

    Two panels sharing the classes. The left is the whole distribution, which says
    whether the map is shifted or merely heavier-tailed in one class; the right is the
    share of each class above the threshold, which is what the cluster table was built
    from. A result can look ordinary in the first and be entirely white matter in the
    second, so neither panel replaces the other.
    """
    usable = [item for item in slices if item.n_voxels > 1]
    if not usable:
        raise ValueError("The tissue panel requires at least one populated class.")

    with plot_context():
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(9.0, 3.3), width_ratios=[2.0, 1.0], constrained_layout=True
        )

        limit = float(np.percentile(np.abs(np.concatenate([i.values for i in usable])), 99.5))
        bins = np.linspace(-limit, limit, 90)
        for position, item in enumerate(usable):
            colour = TISSUE_COLORS[position % len(TISSUE_COLORS)]
            # Density, not counts: the classes differ several-fold in size, and on a
            # count axis grey matter's curve buries the other two whatever the map does.
            left.hist(
                item.values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.3,
                color=colour,
                label=f"{item.name} ({item.n_voxels:,} voxels)",
            )
        # The bins already span the data's robust range; the axis is pinned to them so
        # that a threshold far outside it cannot stretch the view. Drawn with axvline
        # the marker autoscaled the axis instead: measured on this study, bins over
        # +-0.25 against an axis reaching +-2.4, so all three densities were one
        # vertical stroke occupying 8% of the panel.
        left.set_xlim(-limit, limit)
        if threshold:
            for sign in ((-1.0, 1.0) if two_sided else (1.0,)):
                position = sign * float(threshold)
                if abs(position) <= limit:
                    left.axvline(
                        position,
                        color=GUIDE_COLOR,
                        linestyle=(0, (4, 2)),
                        linewidth=1.0,
                    )
                else:
                    _mark_offscale_threshold(left, position, stat_label)
        left.axvline(0.0, color=GUIDE_COLOR, linewidth=0.8)
        left.set_xlabel(stat_label)
        left.set_ylabel("density")
        left.legend(fontsize=7, frameon=False)

        if threshold:
            rates = enrichment(usable, threshold=float(threshold), two_sided=two_sided)
            positions = np.arange(len(rates))
            right.bar(
                positions,
                [100.0 * share for _name, share, _n, _total in rates],
                color=[TISSUE_COLORS[i % len(TISSUE_COLORS)] for i in positions],
                width=0.65,
            )
            for position, (_name, share, survivors, total) in zip(positions, rates):
                right.annotate(
                    f"{survivors:,}",
                    xy=(position, 100.0 * share),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=GUIDE_COLOR,
                )
            right.set_xticks(positions)
            right.set_xticklabels([name for name, *_rest in rates], fontsize=8)
            right.set_ylabel("% of class above threshold")
            # From zero, always. With nothing above the threshold every bar is zero
            # and the autoscaler produced a +-0.04 axis around three of them, which
            # reads as measured precision. Three flat bars against a real axis is the
            # honest picture of a contrast where nothing survived.
            tallest = max((100.0 * share for _n, share, *_r in rates), default=0.0)
            right.set_ylim(0.0, max(tallest * 1.25, 1.0))
        else:
            right.set_axis_off()
            right.text(
                0.5,
                0.5,
                "no threshold applied,\nso nothing survives to compare",
                ha="center",
                va="center",
                fontsize=8,
                color=GUIDE_COLOR,
            )

        if title:
            figure.suptitle(title, fontsize=10)

        provenance = [
            f"{sum(i.n_voxels for i in usable):,} classified voxels in the mask",
        ]
        if tissue_source:
            provenance.append(f"tissue from {tissue_source}")
        if threshold:
            # Every class, ordered by rate. Quoting a single grey-to-white ratio hides
            # the reading that matters most: on this study's own contrast CSF survives
            # at a higher rate than grey matter, which a GM/WM ratio of 1.77 reports as
            # healthy enrichment. The ordering is the finding.
            rates = sorted(
                enrichment(usable, threshold=float(threshold), two_sided=two_sided),
                key=lambda entry: -entry[1],
            )
            provenance.append(
                "survival rate by class: "
                + ", ".join(f"{name} {100.0 * share:.1f}%" for name, share, *_ in rates)
            )
        provenance.append(
            "a BOLD effect is a grey-matter phenomenon; how much enrichment to expect "
            "depends on the contrast and the segmentation, so no criterion is applied"
        )
        annotate_provenance(figure, provenance)
        return figure


__all__ = [
    "TISSUE_COLORS",
    "TISSUE_NAMES",
    "TissueSlice",
    "enrichment",
    "split_by_tissue",
    "tissue_distribution_figure",
]
