"""Whether a first-level effect is carried by every run or by one of them.

A first-level contrast over several runs is a fixed-effects combination, weighted
equally per run: a noisy run contributes its effect at full strength while inflating
the variance. An effect resting entirely on run 4 and an effect present in all six
produce the same map, the same z, and the same cluster table -- and nothing else in
this report tells them apart.

The panel is a forest plot per cluster peak: each run's own estimate with its
confidence interval, against the combined estimate the result section reports. Read for
whether the run estimates straddle the combined one or scatter around it.

Nothing is scored. Runs disagreeing is not by itself a fault -- a task with a learning
or habituation effect should show exactly that -- so the panel measures the spread and
leaves the interpretation to the reader.
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

#: Multiplier turning a standard error into a 95% interval under a normal reference.
#:
#: Normal rather than a per-run t quantile: the residual degrees of freedom differ by
#: run, and a panel whose intervals were each drawn against a different distribution
#: would not be comparable across the rows it exists to compare. At the residual
#: degrees of freedom a run of this length carries -- 500-odd frames against 47
#: regressors -- the t and normal quantiles agree to within 1%.
_CI95 = 1.959964


@dataclass(frozen=True)
class PeakRunEstimates:
    """One cluster peak's estimate in each run, and the combined estimate."""

    label: str
    coordinate: Tuple[float, float, float]
    effects: Tuple[float, ...]
    errors: Tuple[float, ...]
    combined_effect: Optional[float] = None
    combined_error: Optional[float] = None

    @property
    def sign_agreement(self) -> float:
        """Share of runs whose effect has the same sign as the combined estimate.

        Reported rather than tested. With six runs the count is too small for a
        proportion to carry a useful confidence interval, so it is a description of
        what was observed.
        """
        finite = [e for e in self.effects if np.isfinite(e)]
        if not finite:
            return float("nan")
        reference = (
            self.combined_effect
            if self.combined_effect is not None and np.isfinite(self.combined_effect)
            else float(np.mean(finite))
        )
        if reference == 0:
            return float("nan")
        return float(np.mean([np.sign(e) == np.sign(reference) for e in finite]))


def _sample(volume: Any, coordinate: Sequence[float]) -> np.ndarray:
    """Read every frame of a 4D volume at one world coordinate.

    Nearest voxel, not interpolation: a peak is a voxel, and interpolating between it
    and its neighbours reports a number no run actually estimated.
    """
    data = np.asarray(volume.get_fdata())
    inverse = np.linalg.inv(np.asarray(volume.affine))
    voxel = np.rint(
        (np.append(np.asarray(coordinate, dtype=float), 1.0) @ inverse.T)[:3]
    ).astype(int)
    shape = np.asarray(data.shape[:3])
    if np.any(voxel < 0) or np.any(voxel >= shape):
        return np.full(data.shape[3] if data.ndim == 4 else 1, np.nan)
    if data.ndim == 3:
        return np.asarray([data[tuple(voxel)]], dtype=float)
    return np.asarray(data[tuple(voxel)], dtype=float)


def collect_peak_estimates(
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
    *,
    run_effect_img: Any,
    run_variance_img: Any,
    combined_effect_img: Any = None,
    combined_variance_img: Any = None,
) -> List[PeakRunEstimates]:
    """Read each peak's per-run estimate out of the 4D run-level maps."""
    collected: List[PeakRunEstimates] = []
    for label, coordinate in peaks:
        effects = _sample(run_effect_img, coordinate)
        variances = _sample(run_variance_img, coordinate)
        if effects.size != variances.size:
            continue
        errors = np.sqrt(np.clip(variances, 0.0, None))

        combined_effect = None
        combined_error = None
        if combined_effect_img is not None:
            values = _sample(combined_effect_img, coordinate)
            combined_effect = float(values[0]) if values.size else None
        if combined_variance_img is not None:
            values = _sample(combined_variance_img, coordinate)
            combined_error = (
                float(np.sqrt(max(values[0], 0.0))) if values.size else None
            )

        collected.append(
            PeakRunEstimates(
                label=str(label),
                coordinate=tuple(float(c) for c in coordinate),
                effects=tuple(float(e) for e in effects),
                errors=tuple(float(e) for e in errors),
                combined_effect=combined_effect,
                combined_error=combined_error,
            )
        )
    return collected


def peak_forest_figure(
    estimates: Sequence[PeakRunEstimates],
    *,
    run_labels: Sequence[str],
    effect_units: str = "effect",
    max_peaks: int = 6,
    title: str = "",
) -> plt.Figure:
    """Draw each peak's per-run estimate against the combined one.

    One panel per peak, capped at ``max_peaks``: past that the rows are shorter than
    their own labels, and the peaks beyond the sixth are rarely what a result rests on.
    The cap is stated on the figure rather than applied silently.
    """
    shown = list(estimates)[: max(int(max_peaks), 1)]
    if not shown:
        raise ValueError("The run-consistency panel requires at least one peak.")

    n_runs = max(len(estimate.effects) for estimate in shown)
    labels = [
        str(run_labels[i]) if i < len(run_labels) else f"run-{i + 1:02d}"
        for i in range(n_runs)
    ]

    with plot_context():
        figure, axes = plt.subplots(
            1,
            len(shown),
            figsize=(max(3.0 * len(shown), 4.2), 0.34 * n_runs + 2.0),
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for axis, estimate in zip(axes, shown):
            positions = np.arange(len(estimate.effects))
            effects = np.asarray(estimate.effects, dtype=float)
            errors = np.asarray(estimate.errors, dtype=float)

            # The combined estimate as a band, so every run is read against it rather
            # than against zero alone. Zero still gets its own line: it is what "no
            # difference" means, and the band is not centred on it.
            if estimate.combined_effect is not None and np.isfinite(
                estimate.combined_effect
            ):
                axis.axvline(
                    estimate.combined_effect,
                    color=OKABE_ITO["orange"],
                    linewidth=1.2,
                    zorder=1,
                )
                if estimate.combined_error is not None and np.isfinite(
                    estimate.combined_error
                ):
                    axis.axvspan(
                        estimate.combined_effect - _CI95 * estimate.combined_error,
                        estimate.combined_effect + _CI95 * estimate.combined_error,
                        color=OKABE_ITO["orange"],
                        alpha=0.16,
                        linewidth=0,
                        zorder=0,
                    )
            axis.axvline(0.0, color=GUIDE_COLOR, linewidth=0.8, linestyle=":", zorder=1)

            axis.errorbar(
                effects,
                positions,
                xerr=_CI95 * errors,
                fmt="o",
                markersize=4.5,
                color=OKABE_ITO["blue"],
                ecolor=OKABE_ITO["sky_blue"],
                elinewidth=2.0,
                capsize=0,
                zorder=3,
            )

            x, y, z = estimate.coordinate
            axis.set_title(
                f"peak {estimate.label}\n({x:+.0f}, {y:+.0f}, {z:+.0f})", fontsize=8.5
            )
            axis.set_xlabel(effect_units)

        axes[0].set_yticks(np.arange(n_runs))
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_ylim(n_runs - 0.5, -0.5)

        agreements = [
            estimate.sign_agreement
            for estimate in shown
            if np.isfinite(estimate.sign_agreement)
        ]
        # Kept short. annotate_provenance puts every entry on one line, and a strip
        # wider than the axes forces the tight bounding box to grow -- which padded
        # this figure to twice the width of the panels it describes.
        provenance = [
            f"{n_runs} run(s) · {len(shown)} of {len(estimates)} peak(s) shown"
            if len(shown) < len(estimates)
            else f"{n_runs} run(s) · {len(shown)} peak(s)",
            "dot: that run's estimate, bar: 95% interval",
            "orange: the combined estimate and its interval",
        ]
        if agreements:
            provenance.append(
                "sign agreement: " + ", ".join(f"{value:.0%}" for value in agreements)
            )
        provenance.append("runs differing is a measurement, not a fault")
        annotate_provenance(figure, provenance)
        return figure


__all__ = [
    "PeakRunEstimates",
    "collect_peak_estimates",
    "peak_forest_figure",
]
