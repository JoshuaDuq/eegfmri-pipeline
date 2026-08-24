"""What the second-level model leaves over, against what it assumes it would.

The parametric arm of a cohort report rests on the errors being Gaussian, and the report
says so in words. At the sizes second-level models actually run at, that assumption does
real work: the t distribution the p-values come from is exact only if it holds.

Two views, because they fail differently. Pooling the standardised residuals over every
voxel and participant shows the shape of the departure against the standard normal it is
assumed to be. The per-participant spread shows whether one contributor produces it --
a distributional plot alone cannot say who, and a per-participant plot alone cannot say
what. Neither carries a verdict: whether a departure matters depends on the design and
the question, not on the picture.
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


def residual_figure(
    normalized: np.ndarray,
    per_subject: pd.DataFrame,
    *,
    r_square: float,
    rank: int = 1,
    extra_provenance: Sequence[str] = (),
    title: str = "",
) -> Any:
    """Draw the pooled standardised residuals and the per-participant spread.

    The ceiling is stated because at these cohort sizes it binds. All of a voxel's
    residual can land on one participant at most, giving ``|e| = sqrt(SSE)`` against a
    denominator of ``sqrt(SSE / (n - rank))``: the standardised residual cannot exceed
    ``sqrt(n - rank)``. With thirteen participants and one column that is 3.46, so the
    tails are cut off by arithmetic rather than by the data, and a reader comparing them
    with the drawn normal would otherwise read the truncation as a finding.
    """
    values = np.asarray(normalized, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Residual diagnostics require at least one finite residual.")
    labels = [str(value) for value in per_subject["subject"]]
    rms = per_subject["residual RMS"].to_numpy(dtype=float)

    with plot_context():
        figure, (left, right) = plt.subplots(
            1,
            2,
            figsize=(11.0, max(3.6, 0.26 * len(labels) + 3.0)),
            constrained_layout=True,
        )

        # Density rather than counts, so the standard normal can be drawn on the same
        # axis without a second scale to reconcile.
        left.hist(finite, bins=80, density=True, color=OKABE_ITO["sky_blue"], alpha=0.85)
        grid = np.linspace(finite.min(), finite.max(), 400)
        left.plot(
            grid,
            np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi),
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.3,
            label="standard normal",
        )
        left.set_xlabel("standardised residual  (Nilearn e / sqrt(MSE))")
        left.set_ylabel("density")
        left.set_title("Pooled over voxels and participants")
        left.legend(loc="upper right", frameon=False, fontsize=8.5)

        order = np.argsort(rms)
        positions = np.arange(len(labels))
        right.barh(positions, rms[order], color=OKABE_ITO["orange"], height=0.72)
        right.axvline(
            float(np.median(rms)),
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=f"median {np.median(rms):.3g}",
        )
        right.set_yticks(positions)
        right.set_yticklabels([labels[i] for i in order], fontsize=8.5)
        right.set_xlabel("residual RMS")
        right.set_title("Per participant")
        right.legend(loc="lower right", frameon=False, fontsize=8.5)

        figure.suptitle(title or "Second-level residuals")
        annotate_provenance(
            figure,
            [
                f"{values.shape[0]} participants x {values.shape[1]:,} voxels",
                f"mean R² = {r_square:.4g}",
                f"residual skew {float(_moment(finite, 3)):+.3f}",
                f"excess kurtosis {float(_moment(finite, 4) - 3.0):+.3f}",
                f"bounded by sqrt(n - rank) = {np.sqrt(max(values.shape[0] - rank, 1)):.2f}",
                *extra_provenance,
            ],
        )
        return figure


def _moment(values: np.ndarray, order: int) -> float:
    centred = values - values.mean()
    scale = centred.std()
    if scale == 0:
        return 0.0
    return float((centred**order).mean() / scale**order)


__all__ = ["residual_figure"]
