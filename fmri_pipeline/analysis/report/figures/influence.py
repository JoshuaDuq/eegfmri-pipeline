"""How much of a cohort result rests on any one participant.

The companion to the input-map concordance matrix, and it answers a question that
matrix cannot. Concordance is a Pearson correlation, which is scale-invariant: a
participant whose map has the cohort's shape at several times its amplitude correlates
with everyone and is ranked unremarkable, while either carrying the group mean or -- by
inflating the between-subject variance -- suppressing it entirely.

Refitting the model without each participant and rethresholding under the same rule is
what measures that. The panel reports the surviving voxel count per omission against the
full-cohort count; it draws no conclusion about whether the spread is acceptable, because
what counts as acceptable depends on the design and the question, not on the picture.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)


def leave_one_out_figure(
    frame: pd.DataFrame,
    *,
    full_survivors: int,
    threshold_label: str,
    extra_provenance: Sequence[str] = (),
    title: str = "",
) -> Any:
    """Draw the surviving voxel count with each participant left out.

    Sorted by count rather than by participant, because the question is which omissions
    move the result, not who they belong to; the labels carry the identity either way.
    """
    estimable = frame.loc[frame["estimable"].astype(bool)].copy()
    if estimable.empty:
        raise ValueError("Leave-one-out influence requires at least one estimable refit.")
    estimable["surviving voxels"] = estimable["surviving voxels"].astype(int)
    ordered = estimable.sort_values("surviving voxels", ascending=True)
    counts = ordered["surviving voxels"].to_numpy(dtype=float)
    labels = [str(value) for value in ordered["subject"]]

    with plot_context():
        figure, axis = plt.subplots(
            figsize=(max(7.0, 0.34 * len(labels) + 4.0), 0.32 * len(labels) + 2.2),
            constrained_layout=True,
        )
        positions = np.arange(len(labels))
        # One hue for omissions that shrink the map and one for those that grow it. The
        # direction is the point: a participant whose removal enlarges the result was
        # working against it, which reads as the opposite of an outlier to be trimmed.
        colors = [
            OKABE_ITO["orange"] if count < full_survivors else OKABE_ITO["sky_blue"]
            for count in counts
        ]
        axis.barh(positions, counts, color=colors, height=0.72)
        axis.axvline(
            full_survivors,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=f"full cohort ({full_survivors:,})",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8.5)
        axis.set_xlabel(f"surviving voxels at {threshold_label}")
        axis.set_ylabel("cohort refitted without")
        axis.set_title(title or "Leave-one-participant-out influence")
        axis.legend(loc="lower right", frameon=False, fontsize=8.5)

        if full_survivors > 0:
            for position, count in zip(positions, counts):
                change = 100.0 * (count - full_survivors) / full_survivors
                axis.text(
                    count,
                    position,
                    f"  {change:+.0f}%",
                    va="center",
                    ha="left",
                    fontsize=7.6,
                    color="#333333",
                )
            axis.set_xlim(0, max(counts.max(), full_survivors) * 1.16)

        dropped = int((~frame["estimable"].astype(bool)).sum())
        lines = [
            f"{len(labels)} refits",
            f"full cohort {full_survivors:,} voxels",
            threshold_label,
            *extra_provenance,
        ]
        if dropped:
            lines.append(f"{dropped} omission(s) left the design rank-deficient, not refitted")
        annotate_provenance(figure, lines)
        return figure


__all__ = ["leave_one_out_figure"]
