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

    # Ranked rather than piled on one row. Every pattern gets its own line, so no
    # marker hides another and no jitter is invented to separate them; the shape of
    # the climb is the null's distribution, and the gap at the top is the observed
    # value's separation from it. Both readings are lost in a rug and faked in a
    # 32-point histogram.
    order = np.argsort(values)
    ranked = values[order]
    ranks = np.arange(1, ranked.size + 1)
    observed_rank = int(np.argmin(np.abs(ranked - summary.observed_max))) + 1

    with style.plot_context():
        height_in = max(2.0, 0.085 * ranked.size + 1.0)
        figure, axis = plt.subplots(figsize=(6.0, height_in), constrained_layout=True)

        axis.axvline(
            summary.height,
            color=style.OKABE_ITO["blue"],
            linewidth=1.3,
            zorder=2,
        )
        # Mid-height, not at the top: the ranked points crowd the top of the line,
        # which is exactly where a threshold label wants to sit.
        axis.annotate(
            f"familywise 5%\n|z| > {summary.height:.2f}",
            xy=(summary.height, ranked.size * 0.45),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
            color=style.OKABE_ITO["blue"],
        )

        is_observed = ranks == observed_rank
        axis.scatter(
            ranked[~is_observed],
            ranks[~is_observed],
            s=16,
            facecolor="white",
            edgecolor="0.4",
            linewidth=0.8,
            zorder=3,
        )
        axis.scatter(
            ranked[is_observed],
            ranks[is_observed],
            s=34,
            color=style.OKABE_ITO["vermillion"],
            zorder=4,
        )
        axis.annotate(
            f"observed  {summary.observed_max:.2f}",
            xy=(summary.observed_max, observed_rank),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=style.OKABE_ITO["vermillion"],
        )

        axis.set_xlabel("max |z| over the analysis mask")
        axis.set_ylabel("sign patterns, ranked")
        axis.set_ylim(0.3, ranked.size + 1.8)
        axis.set_yticks([1, ranked.size])
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
