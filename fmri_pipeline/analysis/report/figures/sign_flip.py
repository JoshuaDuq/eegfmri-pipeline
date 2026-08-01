"""Where the observed maximum falls among the maxima run relabelling can produce.

The only panel this measurement warrants. Its height, its survivor count and its p are
numbers, and they are stated in the threshold table; what a table cannot show is
whether the observed maximum stands apart from the null or sits inside it. That is a
position in a distribution, and position is what a picture is for.

Drawn as a strip of the actual maxima rather than a histogram. With ``2**(n-1)``
patterns there are 32 values for six runs and 8 for four, and a binned density over
that many points invents a shape the data does not have.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fmri_pipeline.analysis.report import style
from fmri_pipeline.analysis.report.inference import SignFlipSummary


def sign_flip_figure(
    null_max: Sequence[float], *, summary: SignFlipSummary, title: str = ""
):
    """One axis: every sign pattern's maximum |z|, the FWE height, the observed value."""
    import matplotlib.pyplot as plt

    values = np.asarray(list(null_max), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The sign-flip panel requires at least one enumerated maximum.")

    with style.plot_context():
        figure, axis = plt.subplots(figsize=(7.0, 2.1), constrained_layout=True)

        axis.plot(
            values,
            np.zeros_like(values),
            marker="|",
            linestyle="none",
            markersize=18,
            markeredgewidth=1.1,
            color="0.45",
            label=f"{values.size} sign patterns",
        )
        axis.axvline(
            summary.height,
            color=style.OKABE_ITO["blue"],
            linewidth=1.4,
            label=f"FWE 5%: |z| > {summary.height:.2f}",
        )
        axis.plot(
            [summary.observed_max],
            [0.0],
            marker="v",
            markersize=9,
            color=style.OKABE_ITO["vermillion"],
            linestyle="none",
            label=f"observed: {summary.observed_max:.2f}",
            zorder=4,
        )

        axis.set_yticks([])
        axis.set_ylim(-0.5, 0.5)
        for side in ("left", "top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlabel("max |z| over the analysis mask")
        axis.set_title(title or "Run sign-flip null", pad=26)
        # Anchored by its lower edge, so the legend clears the axes entirely. Anchored
        # by its upper edge it sits inside them, and the familywise line runs through
        # its own label.
        axis.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7
        )

        notes = [
            f"{summary.n_patterns} exact sign patterns over {summary.n_runs} runs, "
            "exchangeable by run",
            f"{summary.survivors:,} voxel(s) at the familywise height",
        ]
        if summary.floor_limited:
            notes.append(
                f"global p = {summary.global_p:.3f}, the smallest this test can "
                f"return: the unflipped pattern is always in the null and always ties "
                f"the observed maximum, so {summary.n_runs} runs cannot reach below "
                f"{summary.p_floor:.3f}"
            )
            notes.append("the height is unaffected by that floor")
        else:
            notes.append(
                f"global p = {summary.global_p:.3f} against a floor of "
                f"{summary.p_floor:.3f}"
            )
        notes.append("no threshold here is scored against a criterion")
        style.annotate_provenance(figure, notes)
        return figure


__all__ = ["sign_flip_figure"]
