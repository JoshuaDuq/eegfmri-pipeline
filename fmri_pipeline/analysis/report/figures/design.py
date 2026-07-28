"""Design matrix, contrast, and collinearity panels.

Collinearity is the figure that says whether a contrast is estimable at all. The
variance inflation factors were already computed in the reporting path, but their
only destination was a single cell of a summary table.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    SIGNED_CMAP,
    plot_context,
)

#: VIF plotted in place of infinity, so a perfectly collinear regressor stays on the axis.
_VIF_CEILING = 1e3


def vif_from_design(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor per column of ``X`` (n_samples, n_features).

    ``VIF_j = 1 / (1 - R^2_j)`` where ``R^2_j`` regresses column j on the others.
    Returns ``inf`` where ``R^2 >= 1`` (perfect collinearity) or the fit fails.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if p < 2:
        return np.array([], dtype=np.float64)

    # Constant columns are dropped from every predictor set. An intercept is added
    # explicitly below, so leaving the design's own constant in makes the system
    # singular: lstsq returns an enormous beta and the residual sum overflows.
    is_constant = np.std(X, axis=0) <= 0

    vif = np.full(p, np.nan, dtype=np.float64)
    for j in range(p):
        y = X[:, j]
        # A constant regressor has zero variance, so the proportion of it explained
        # by the others is undefined rather than large. Reported as infinite, which
        # is how the figure already renders "cannot be separated".
        if is_constant[j]:
            vif[j] = np.inf
            continue

        others = [k for k in range(p) if k != j and not is_constant[k]]
        Z = np.column_stack([np.ones(n), X[:, others]]) if others else np.ones((n, 1))
        try:
            beta, residuals, _rank, _ = np.linalg.lstsq(Z, y, rcond=None)
            ss_res = (
                float(residuals.flat[0])
                if residuals.size
                else float(np.sum((y - Z @ beta) ** 2))
            )
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0 or not np.isfinite(ss_res):
                vif[j] = np.inf
                continue
            r_sq = 1.0 - (ss_res / ss_tot)
            vif[j] = np.inf if (r_sq >= 1.0 or np.isnan(r_sq)) else 1.0 / (1.0 - r_sq)
        except np.linalg.LinAlgError:
            vif[j] = np.inf
    return vif


def _regressor_correlation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the correlation matrix and a mask of constant regressors.

    ``np.corrcoef`` divides by the standard deviation, so a constant column -- and
    every design has an intercept -- yields NaN across a whole row and column. Left
    alone that renders as blank cells indistinguishable from data that failed to
    load. Constant columns are therefore identified, excluded from the computation,
    and reported so the figure can mark them as undefined by construction.
    """
    constant = np.std(X, axis=0) <= 0
    correlation = np.full((X.shape[1], X.shape[1]), np.nan, dtype=float)
    varying = np.flatnonzero(~constant)
    if varying.size >= 2:
        block = np.corrcoef(X[:, varying], rowvar=False)
        correlation[np.ix_(varying, varying)] = block
    elif varying.size == 1:
        correlation[varying[0], varying[0]] = 1.0
    return correlation, constant


def _column_limit(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.percentile(np.abs(finite), 98.0)) if finite.size else 1.0


def design_matrix_figure(
    design_matrix: pd.DataFrame,
    *,
    contrast: Optional[np.ndarray] = None,
    contrast_name: str = "",
    title: str = "",
) -> plt.Figure:
    """Draw the design matrix, optionally with its contrast as a strip beneath.

    Drawn here rather than through ``nilearn.plotting.plot_design_matrix`` so the
    contrast strip shares the column axis: a contrast named in prose is not legible,
    and one drawn on its own axis does not line up with the regressors it weights.
    """
    columns = list(design_matrix.columns)
    if contrast is not None:
        contrast = np.asarray(contrast, dtype=float).ravel()
        if contrast.size != len(columns):
            raise ValueError(
                f"Contrast has {contrast.size} weights but the design has "
                f"{len(columns)} columns."
            )

    limit = _column_limit(design_matrix)
    heights = [6.0, 0.5] if contrast is not None else [6.0]
    with plot_context():
        figure, axes = plt.subplots(
            len(heights),
            1,
            figsize=(max(6.0, 0.32 * len(columns)), 5.5),
            gridspec_kw={"height_ratios": heights},
        )
        axes = np.atleast_1d(axes)

        axes[0].imshow(
            design_matrix.to_numpy(dtype=float),
            aspect="auto",
            cmap=SIGNED_CMAP,
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        axes[0].set_xticks(range(len(columns)))
        axes[0].set_xticklabels(columns, rotation=90, fontsize=7)
        axes[0].set_ylabel("Frame")
        if contrast is None:
            axes[0].set_xlabel("Regressor")
        if title:
            axes[0].set_title(title)

        if contrast is not None:
            strip_limit = float(np.max(np.abs(contrast))) or 1.0
            axes[1].imshow(
                contrast.reshape(1, -1),
                aspect="auto",
                cmap=SIGNED_CMAP,
                vmin=-strip_limit,
                vmax=strip_limit,
            )
            axes[1].set_yticks([0])
            axes[1].set_yticklabels([contrast_name or "contrast"], fontsize=8)
            axes[1].set_xticks(range(len(columns)))
            axes[1].set_xticklabels(columns, rotation=90, fontsize=7)
            axes[1].set_xlabel("Regressor")
            # The strip carries the names for both panels: the axes share a column
            # order, so printing the list twice costs a band of vertical space and
            # pushes the strip away from the matrix it annotates.
            axes[0].set_xticklabels([])

        figure.tight_layout()
        return figure


def collinearity_figure(design_matrix: pd.DataFrame, *, title: str = "") -> plt.Figure:
    """Draw per-regressor VIF beside the regressor correlation matrix.

    The VIF axis is logarithmic because the interesting range spans one to infinity;
    on a linear axis every well-conditioned regressor collapses against the origin.
    Infinite VIF is drawn at a ceiling and labelled, so a perfectly collinear
    regressor stays visible instead of leaving the axis.
    """
    columns = list(design_matrix.columns)
    X = design_matrix.to_numpy(dtype=float)
    vif = vif_from_design(X)
    plotted = np.where(np.isfinite(vif), vif, _VIF_CEILING) if vif.size else np.array([])

    with plot_context():
        # Height tracks the regressor count because the correlation panel is square:
        # a fixed height leaves it floating in an over-wide box with the VIF bars
        # stranded on the far side. Width is what is left over after the labels.
        side = max(3.4, 0.34 * len(columns))
        figure, (vif_axis, corr_axis) = plt.subplots(
            1,
            2,
            figsize=(2.0 * side + 2.4, side),
            gridspec_kw={"width_ratios": [1.0, 1.0]},
        )

        positions = np.arange(len(columns))
        vif_axis.barh(
            positions,
            plotted if plotted.size else np.ones(len(columns)),
            color=OKABE_ITO["sky_blue"],
        )
        vif_axis.set_yticks(positions)
        vif_axis.set_yticklabels(columns, fontsize=7)
        vif_axis.invert_yaxis()
        vif_axis.set_xscale("log")
        vif_axis.set_xlabel("Variance inflation factor")
        vif_axis.axvline(1.0, color=GUIDE_COLOR, linewidth=0.8)
        for index, value in enumerate(vif):
            if not np.isfinite(value):
                vif_axis.annotate(
                    "inf",
                    xy=(_VIF_CEILING, index),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=OKABE_ITO["vermillion"],
                )

        correlation, constant = _regressor_correlation(X)
        image = corr_axis.imshow(correlation, cmap=SIGNED_CMAP, vmin=-1.0, vmax=1.0)
        corr_axis.set_xticks(positions)
        corr_axis.set_xticklabels(columns, rotation=90, fontsize=7)
        corr_axis.set_yticks(positions)
        corr_axis.set_yticklabels(columns, fontsize=7)
        # A constant regressor -- every design has an intercept -- has undefined
        # correlation. Hatched and named, so it reads as "undefined by construction"
        # rather than as data that failed to load.
        for index in np.flatnonzero(constant):
            corr_axis.add_patch(
                plt.Rectangle(
                    (-0.5, index - 0.5), len(columns), 1.0,
                    facecolor="none", edgecolor=GUIDE_COLOR,
                    hatch="///", linewidth=0.0, alpha=0.45,
                )
            )
            corr_axis.add_patch(
                plt.Rectangle(
                    (index - 0.5, -0.5), 1.0, len(columns),
                    facecolor="none", edgecolor=GUIDE_COLOR,
                    hatch="///", linewidth=0.0, alpha=0.45,
                )
            )
        if constant.any():
            names = ", ".join(np.asarray(columns)[constant])
            corr_axis.set_xlabel(f"hatched: constant, correlation undefined ({names})",
                                 fontsize=7)
        bar = figure.colorbar(image, ax=corr_axis, fraction=0.04, pad=0.03)
        bar.set_label("Pearson r")

        if title:
            figure.suptitle(title)
        figure.tight_layout()
        return figure


__all__ = ["collinearity_figure", "design_matrix_figure", "vif_from_design"]
