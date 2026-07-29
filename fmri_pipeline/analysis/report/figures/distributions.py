"""Histogram panels for statistic and magnitude distributions."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import GUIDE_COLOR, OKABE_ITO, plot_context

_BINS = 120


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("A histogram requires at least one finite value.")
    return finite


def z_histogram(
    values: np.ndarray,
    *,
    threshold: Optional[float] = None,
    title: str = "",
) -> plt.Figure:
    """Draw the distribution of z statistics against the standard normal null.

    The y-axis is logarithmic and the null is drawn on top, because the question
    this panel answers is how far the tails depart from N(0, 1). On a linear axis
    the null peak is the only visible feature and the tails -- the signal -- are
    flat against the axis.

    The null curve is scaled to the total voxel count, which assumes most voxels
    are null. That is the conventional display and close to true for a typical
    contrast, but it understates the null for a map where a large fraction of the
    brain is genuinely active.
    """
    finite = _finite(values)
    with plot_context():
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        counts, edges, _ = axis.hist(
            finite, bins=_BINS, color=OKABE_ITO["blue"], edgecolor="none"
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        bin_width = float(edges[1] - edges[0])
        null_density = (
            finite.size * bin_width * np.exp(-0.5 * centres**2) / np.sqrt(2.0 * np.pi)
        )
        axis.plot(
            centres,
            null_density,
            color=GUIDE_COLOR,
            linewidth=1.5,
            label="N(0, 1) null",
        )
        if threshold is not None and threshold > 0:
            for sign in (1.0, -1.0):
                axis.axvline(
                    sign * float(threshold),
                    color=OKABE_ITO["vermillion"],
                    linestyle="--",
                    linewidth=1.2,
                )
        axis.set_yscale("log")
        positive = counts[counts > 0]
        if positive.size:
            axis.set_ylim(bottom=max(0.5, float(np.min(positive)) * 0.5))
        axis.set_xlabel("z")
        axis.set_ylabel("voxels")
        if title:
            axis.set_title(title)
        axis.legend(fontsize=8)
        figure.tight_layout()
        return figure


def magnitude_histogram(
    values: np.ndarray,
    *,
    xlabel: str,
    title: str = "",
) -> plt.Figure:
    """Draw the distribution of an unsigned magnitude with its median marked.

    No alpha on the bars: a translucent fill over a solid histogram produces visible
    seams at every bar boundary that read as structure in the data.
    """
    finite = _finite(values)
    with plot_context():
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        axis.hist(finite, bins=_BINS, color=OKABE_ITO["bluish_green"], edgecolor="none")
        median = float(np.median(finite))
        axis.axvline(median, color=GUIDE_COLOR, linestyle="--", linewidth=1.2)
        axis.annotate(
            f"median {median:.3g}",
            xy=(median, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
            color=GUIDE_COLOR,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("voxels")
        if title:
            axis.set_title(title)
        figure.tight_layout()
        return figure


__all__ = ["magnitude_histogram", "z_histogram"]
