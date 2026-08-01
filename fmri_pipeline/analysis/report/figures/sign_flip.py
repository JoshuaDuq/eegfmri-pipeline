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

    # The empirical CDF of the permutation distribution. Both quantities a reader
    # needs are then read off an axis rather than inferred: the familywise height is
    # where the curve crosses 1 - alpha, and the exceedance probability of the
    # observed value is one minus its height on the curve.
    #
    # An earlier version put rank on the vertical axis. Rank is an index, not a
    # measurement -- nothing follows from a pattern being 17th rather than 18th -- so
    # that panel spent its whole vertical dimension on a non-quantity, and was this
    # same curve unnormalised and mislabelled.
    ordered = np.sort(values)
    cumulative = np.arange(1, ordered.size + 1) / ordered.size
    #: The level the familywise height is the quantile of. Fixed at 0.95 because
    #: ``SignFlipSummary.height`` is defined as the 95th percentile of this null.
    alpha = 0.95

    with style.plot_context():
        figure, axis = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)

        axis.step(
            np.concatenate([[ordered[0]], ordered]),
            np.concatenate([[0.0], cumulative]),
            where="post",
            color="0.25",
            linewidth=1.3,
            zorder=3,
        )
        axis.plot(
            ordered,
            cumulative,
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor="0.25",
            markeredgewidth=0.8,
            zorder=4,
        )

        axis.axhline(alpha, color=style.OKABE_ITO["blue"], linewidth=1.0, linestyle=":")
        axis.axvline(summary.height, color=style.OKABE_ITO["blue"], linewidth=1.3)
        axis.annotate(
            f"familywise 5%: |z| > {summary.height:.2f}",
            xy=(summary.height, alpha),
            xytext=(-11, -10),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color=style.OKABE_ITO["blue"],
        )

        axis.axvline(
            summary.observed_max,
            color=style.OKABE_ITO["vermillion"],
            linewidth=1.3,
            zorder=5,
        )
        axis.annotate(
            f"observed {summary.observed_max:.2f}\np = {summary.global_p:.3f}",
            xy=(summary.observed_max, 0.06),
            xytext=(-7, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=style.OKABE_ITO["vermillion"],
        )

        axis.set_xlabel("max |z| over the analysis mask")
        axis.set_ylabel("proportion of sign patterns ≤ x")
        axis.set_ylim(0.0, 1.04)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, alpha, 1.0])
        axis.set_yticklabels(["0", "0.25", "0.50", "0.75", "0.95", "1"])
        axis.set_title(title or "Run sign-flip null", pad=10)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

        # Short lines only. The helper joins them into a single 6.5pt strip, so a
        # sentence here becomes an unreadable ribbon; the reasoning belongs in the
        # caption, and only what must travel with the figure belongs on it.
        notes = [
            f"{summary.n_patterns} sign patterns over {summary.n_runs} runs",
            f"{summary.survivors:,} voxel(s) at the familywise height",
            f"p = {summary.global_p:.3f}"
            + (
                f" (its floor for {summary.n_runs} runs)"
                if summary.floor_limited
                else f" (floor {summary.p_floor:.3f})"
            ),
            "no criterion applied",
        ]
        style.annotate_provenance(figure, notes)
        return figure


__all__ = ["sign_flip_figure"]
