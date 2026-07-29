"""Design-matrix diagnostics: is this contrast actually estimable?

A design matrix picture on its own shows that regressors exist. What determines whether
the contrast beside it means anything is how much *independent* variance carries it, and
that is invisible in the matrix. These figures show the matrix with the contrast that is
actually tested, how far the regressors duplicate one another, and the variance inflation
that results — so a near-singular design cannot produce a confident-looking map without
the reader seeing why.

Each figure stands alone and is laid out by the report, not by a subplot grid: cramming
them into one canvas forced tick labels of one panel through the title of the next.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fmri_pipeline.analysis.report import style

__all__ = [
    "DesignSummary",
    "RegressorGroup",
    "classify_regressors",
    "contrast_efficiency",
    "design_matrix_figure",
    "regressor_correlation_figure",
    "summarize_design",
    "variance_inflation_factors",
    "variance_inflation_figure",
]


#: Most non-zero contrast weights that can be written into the strip legibly. Past
#: this the cells are narrower than the digits, and an unreadable number on the wrong
#: cell is worse than none: the colour still carries the sign.
_MAX_ANNOTATED_WEIGHTS = 14

_DRIFT_RE = re.compile(r"^(drift[_\-]?\d+|cosine\d*|poly\d*)$", re.IGNORECASE)
_CONSTANT_NAMES = {"constant", "intercept"}
_CONFOUND_PREFIXES = (
    "trans_",
    "rot_",
    "framewise_displacement",
    "std_dvars",
    "dvars",
    "white_matter",
    "csf",
    "global_signal",
    "a_comp_cor",
    "t_comp_cor",
    "c_comp_cor",
    "w_comp_cor",
    "motion_outlier",
    "non_steady_state",
    "outlier",
)


@dataclass(frozen=True)
class RegressorGroup:
    """A contiguous block of design columns sharing a role."""

    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def _role(column: str) -> str:
    lowered = str(column).strip().lower()
    if lowered in _CONSTANT_NAMES:
        return "Constant"
    if _DRIFT_RE.match(lowered):
        return "Drift"
    if lowered.startswith(_CONFOUND_PREFIXES):
        return "Confound"
    return "Task"


def classify_regressors(columns: Sequence[str]) -> Tuple[List[str], List[RegressorGroup]]:
    """Order design columns by role and report each role's column span.

    Task, confound, drift and constant regressors are answering different questions, and
    interleaving them makes the matrix unreadable. The returned order groups them; the
    spans let the figure separate and label the blocks.
    """
    order_of_roles = ["Task", "Confound", "Drift", "Constant"]
    by_role: Dict[str, List[str]] = {role: [] for role in order_of_roles}
    for column in columns:
        by_role[_role(column)].append(str(column))

    ordered: List[str] = []
    groups: List[RegressorGroup] = []
    for role in order_of_roles:
        members = by_role[role]
        if not members:
            continue
        groups.append(RegressorGroup(role, len(ordered), len(ordered) + len(members)))
        ordered.extend(members)
    return ordered, groups


def variance_inflation_factors(design: "np.ndarray") -> "np.ndarray":
    """VIF per column: 1 / (1 - R²_j) from regressing each column on the others.

    Returns ``inf`` where a column is perfectly explained by the rest, which is the
    signature of an exactly singular design.
    """
    design = np.asarray(design, dtype=float)
    n_samples, n_features = design.shape
    if n_features < 2:
        return np.empty(0, dtype=float)

    out = np.full(n_features, np.nan, dtype=float)
    for index in range(n_features):
        target = design[:, index]
        total = float(np.sum((target - target.mean()) ** 2))
        if total <= 0:
            # A constant column explains no variance of its own; VIF is undefined and
            # reporting it as infinite would be indistinguishable from true collinearity.
            out[index] = np.inf
            continue

        others = np.column_stack([np.ones(n_samples), np.delete(design, index, axis=1)])
        # rcond=None lets numpy pick a cutoff from machine precision, which on a rank
        # deficient block produces inf/nan coefficients. An explicit cutoff truncates
        # the degenerate directions instead.
        #
        # errstate: numpy 1.26 on Accelerate BLAS raises spurious "divide by zero" /
        # "overflow" flags from matmul even for well-conditioned finite operands, so the
        # flags carry no information here. The finiteness checks below are the real
        # guard against a genuinely degenerate solve.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            beta, *_ = np.linalg.lstsq(others, target, rcond=1e-12)
            if not np.all(np.isfinite(beta)):
                out[index] = np.inf
                continue
            residual = target - others @ beta

        if not np.all(np.isfinite(residual)):
            out[index] = np.inf
            continue

        r_squared = 1.0 - float(np.sum(residual**2)) / total
        out[index] = (
            np.inf
            if r_squared >= 1.0 or not np.isfinite(r_squared)
            else 1.0 / (1.0 - r_squared)
        )
    return out


def contrast_efficiency(design: "np.ndarray", contrast: "np.ndarray") -> Optional[float]:
    """Efficiency of a contrast: ``1 / (cᵀ (XᵀX)⁻¹ c)``.

    Proportional to the precision the design affords this particular comparison. It is a
    relative quantity — comparable between designs for the same contrast, meaningless as
    an absolute number — so the figure reports it without a threshold attached.
    """
    design = np.asarray(design, dtype=float)
    contrast = np.asarray(contrast, dtype=float).ravel()
    if contrast.size != design.shape[1]:
        return None
    try:
        # errstate for the same reason as variance_inflation_factors: numpy on
        # Accelerate BLAS raises spurious invalid/overflow flags from matmul even for
        # well-conditioned finite operands. The finiteness check below is the guard.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            covariance = np.linalg.pinv(design.T @ design)
            denominator = float(contrast @ covariance @ contrast)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    return 1.0 / denominator


def _display_scaled(matrix: "np.ndarray") -> "np.ndarray":
    """Scale each column to [-1, 1] by its own peak magnitude.

    Motion regressors live in millimetres and task regressors in HRF units, so a shared
    colour scale renders the whole matrix as one flat block. Per-column scaling shows each
    regressor's shape, which is what the panel is read for; it is stated on the figure.
    """
    peak = np.max(np.abs(matrix), axis=0, keepdims=True)
    peak[peak == 0] = 1.0
    return matrix / peak


@dataclass(frozen=True)
class DesignSummary:
    """Scalars describing how well this design supports its contrast."""

    n_scans: int
    n_regressors: int
    condition_number: float
    max_vif: Optional[float]
    max_vif_regressor: str
    efficiency: Optional[float]
    #: Numerical rank of the design. Equal to ``n_regressors`` for a design of full
    #: rank, and smaller for one that is not -- which is the difference between a model
    #: with the parameters it appears to have and one carrying columns that add no
    #: information. Nothing else in the report distinguishes the two.
    rank: int = 0
    #: Scans minus rank: the degrees of freedom left to estimate the residual variance,
    #: which is the denominator of every t statistic this design produces. Reported
    #: because a confident-looking map from a design with little residual freedom is
    #: exactly the case a reader has no other way to detect.
    residual_dof: int = 0


def _prepare(
    design_matrix: "pd.DataFrame",
    contrast: Optional[Dict[str, float]],
) -> Tuple["pd.DataFrame", List[str], List[RegressorGroup], List[str], Optional["np.ndarray"]]:
    frame = design_matrix.drop(columns=[c for c in ("frame",) if c in design_matrix.columns])
    ordered_columns, groups = classify_regressors(list(frame.columns))
    frame = frame[ordered_columns]
    # Collinearity is a property of the modelled variance, so the constant is excluded:
    # every regressor correlates with a column of ones and it would swamp the panel.
    modelled = [c for c in ordered_columns if _role(c) != "Constant"]

    vector = None
    if contrast:
        # A contrast is matched to columns by name, which cannot go wrong positionally
        # but can go wrong silently: a misspelt key weights nothing and draws a strip
        # indistinguishable from a legitimately empty contrast. Refuse instead.
        unknown = sorted(set(contrast) - set(ordered_columns))
        if unknown:
            raise ValueError(
                f"Contrast names regressors absent from the design: {unknown}. "
                f"Design columns are: {ordered_columns}."
            )
        vector = np.array([float(contrast.get(c, 0.0)) for c in ordered_columns])

    return frame, ordered_columns, groups, modelled, vector


def summarize_design(
    design_matrix: "pd.DataFrame",
    *,
    contrast: Optional[Dict[str, float]] = None,
) -> DesignSummary:
    """Conditioning and efficiency scalars for one run's design."""
    frame, _ordered, _groups, modelled, vector = _prepare(design_matrix, contrast)
    matrix = frame.to_numpy(dtype=float)
    vifs = variance_inflation_factors(frame[modelled].to_numpy(dtype=float))

    max_vif: Optional[float] = None
    max_vif_regressor = ""
    if vifs.size:
        finite = np.isfinite(vifs)
        if not finite.all():
            # An infinite VIF is the most severe case there is: the regressor is exactly
            # reproducible from the others. Ranking by the finite values only would
            # report the mildest inflation in a design that is outright singular.
            index = int(np.argmax(~finite))
            max_vif = float("inf")
            max_vif_regressor = modelled[index] if index < len(modelled) else ""
        else:
            index = int(np.argmax(vifs))
            max_vif = float(vifs[index])
            max_vif_regressor = modelled[index] if index < len(modelled) else ""

    # Rank rather than column count. A design whose columns are linearly dependent has
    # fewer parameters than it has columns, and the residual degrees of freedom follow
    # the rank -- so subtracting the column count would understate them.
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0

    return DesignSummary(
        n_scans=int(matrix.shape[0]),
        n_regressors=int(matrix.shape[1]),
        condition_number=float(np.linalg.cond(matrix)) if matrix.size else float("nan"),
        max_vif=max_vif,
        max_vif_regressor=max_vif_regressor,
        efficiency=contrast_efficiency(matrix, vector) if vector is not None else None,
        rank=rank,
        residual_dof=int(matrix.shape[0]) - rank,
    )


def design_matrix_figure(
    design_matrix: "pd.DataFrame",
    *,
    contrast: Optional[Dict[str, float]] = None,
    tr_seconds: Optional[float] = None,
    run_label: str = "",
    max_labelled_columns: int = 28,
) -> Figure:
    """The design matrix, grouped by regressor role, with the contrast strip beneath it.

    The contrast shares this figure rather than standing alone because it is read
    *against* the columns: a weight only means something once you can see which regressor
    it lands on, and that requires the two to share an x axis.
    """
    import matplotlib.pyplot as plt

    frame, ordered_columns, groups, _modelled, vector = _prepare(design_matrix, contrast)
    matrix = frame.to_numpy(dtype=float)
    n_scans, n_regressors = matrix.shape
    task_group = next((g for g in groups if g.name == "Task"), None)

    with style.plot_context():
        figure, (ax, ax_contrast) = plt.subplots(
            2,
            1,
            figsize=(7.4, 6.4),
            height_ratios=[14, 1],
            sharex=True,
            constrained_layout=True,
        )
        y_extent = n_scans * tr_seconds if tr_seconds else n_scans
        image = ax.imshow(
            _display_scaled(matrix),
            aspect="auto",
            cmap=style.SIGNED_CMAP,
            vmin=-1,
            vmax=1,
            extent=(-0.5, n_regressors - 0.5, y_extent, 0),
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_ylabel("Time (s)" if tr_seconds else "Scan")
        ax.set_title("Design matrix", pad=20)
        ax.tick_params(axis="x", length=0)

        # Labelling every column is unreadable past a couple of dozen regressors, and the
        # nuisance block is not what a reader inspects by name. Task regressors keep
        # their labels; the rest are identified by their group band.
        #
        # Set on ax_contrast, the lower of the two shared axes. Under `sharex` the upper
        # axes' tick labels are hidden and the lower axes renders its own from the shared
        # locator -- without the rotation, because rotation belongs to the Text objects
        # that were created on the axes we set it on. Setting these on `ax` therefore
        # produced horizontal labels that overlapped into an unreadable smear, on a panel
        # whose whole purpose is to say which weight lands on which regressor.
        if n_regressors <= max_labelled_columns:
            ticks = list(range(n_regressors))
            fontsize = 6.4
        elif task_group is not None:
            ticks = list(range(task_group.start, task_group.stop))
            fontsize = 6.8
        else:
            ticks = []
            fontsize = 6.8
        ax_contrast.set_xticks(ticks)
        ax_contrast.set_xticklabels(
            [ordered_columns[i] for i in ticks], rotation=90, fontsize=fontsize
        )

        for group in groups[:-1]:
            for target in (ax, ax_contrast):
                target.axvline(group.stop - 0.5, color="#000000", linewidth=1.0, alpha=0.8)
        for group in groups:
            ax.annotate(
                f"{group.name} ({group.size})",
                xy=((group.start + group.stop - 1) / 2.0, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="bold",
            )
        bar = figure.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
        bar.set_label("column scaled to its own peak")

        ax_contrast.set_yticks([])
        ax_contrast.set_ylabel("Contrast", rotation=0, ha="right", va="center", fontsize=8.5)
        if vector is None:
            ax_contrast.text(
                0.5, 0.5, "no contrast supplied", ha="center", va="center",
                fontsize=8, color="#666666", transform=ax_contrast.transAxes,
            )
        else:
            limit = float(np.max(np.abs(vector))) or 1.0
            ax_contrast.imshow(
                vector[np.newaxis, :],
                aspect="auto",
                cmap=style.SIGNED_CMAP,
                vmin=-limit,
                vmax=limit,
                extent=(-0.5, n_regressors - 0.5, 0, 1),
                interpolation="nearest",
            )
            # The weight goes in its own cell, not in a caption below the strip. The
            # column is already named by the tick label underneath, so repeating it
            # here duplicated the name and collided with it; and a weight written on
            # the cell it belongs to needs no matching up at all.
            weighted = np.flatnonzero(vector)
            if 0 < weighted.size <= _MAX_ANNOTATED_WEIGHTS:
                for index in weighted:
                    ax_contrast.text(
                        index,
                        0.5,
                        f"{vector[index]:+g}",
                        ha="center",
                        va="center",
                        fontsize=6.4,
                        fontweight="bold",
                        # White on the saturated ends of the diverging map, dark in
                        # the pale middle, so the number stays legible at any weight.
                        color="white" if abs(vector[index]) > 0.55 * limit else "#222222",
                    )

        style.annotate_provenance(
            figure,
            [
                run_label,
                f"{n_scans} scans × {n_regressors} regressors",
                "columns scaled individually; nuisance blocks labelled by group",
            ],
        )
    return figure


def regressor_correlation_figure(
    design_matrix: "pd.DataFrame",
    *,
    run_label: str = "",
    max_labelled: int = 30,
) -> Figure:
    """How far the regressors duplicate one another.

    Degrades to a statement rather than a picture when there is nothing to correlate.
    A one-sample second-level design is intercept-only, so excluding the constant
    leaves no columns at all -- and that is the commonest group analysis there is.
    """
    import matplotlib.pyplot as plt

    frame, _ordered, _groups, modelled, _vector = _prepare(design_matrix, None)

    if len(modelled) < 2:
        # A one-sample second-level design is intercept-only, so nothing is left once
        # the constant is excluded, and `np.corrcoef` of a single column returns a
        # 0-d array that `imshow` rejects outright. Saying there is nothing to
        # correlate is the answer; crashing on the commonest group design is not.
        figure, ax = plt.subplots(figsize=(6.0, 2.0), constrained_layout=True)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"{len(modelled)} modelled regressor(s): nothing to correlate",
            ha="center",
            va="center",
            fontsize=9,
        )
        style.annotate_provenance(figure, [run_label, "constant term excluded"])
        return figure

    values = frame[modelled].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        correlation = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)

    # The worst pair, named. "Largest |r| off the diagonal: 0.98" tells a reader that
    # two columns duplicate each other but not which two, and on a design too wide to
    # label there is nowhere else to find out.
    worst, worst_pair = 0.0, ""
    if len(modelled) > 1:
        magnitude = np.abs(correlation).copy()
        np.fill_diagonal(magnitude, 0.0)
        flat = int(np.argmax(magnitude))
        row, column = divmod(flat, magnitude.shape[1])
        worst = float(magnitude[row, column])
        worst_pair = f"{modelled[row]} / {modelled[column]}"

    with style.plot_context():
        figure, ax = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
        image = ax.imshow(
            correlation, cmap=style.SIGNED_CMAP, vmin=-1, vmax=1, interpolation="nearest"
        )
        # Padded when the role bands are drawn: they sit at the top of the axes.
        ax.set_title(
            "Regressor correlation", pad=16 if len(modelled) > max_labelled else None
        )
        if len(modelled) <= max_labelled:
            ax.set_xticks(range(len(modelled)))
            ax.set_xticklabels(modelled, rotation=90, fontsize=6.0)
            ax.set_yticks(range(len(modelled)))
            ax.set_yticklabels(modelled, fontsize=6.0)
        else:
            # Role bands rather than names, as on the variance-inflation panel. A hot
            # off-diagonal block among the confounds is ordinary -- a motion parameter
            # and its own square are correlated by construction -- while the same
            # block reaching the task regressors is what costs the contrast its
            # variance. Unbanded, the two are indistinguishable.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{len(modelled)} modelled regressors, grouped by role")
            for span in _role_spans(modelled):
                if span.stop < len(modelled):
                    for line in (ax.axvline, ax.axhline):
                        line(span.stop - 0.5, color="0.35", linewidth=0.8)
                ax.annotate(
                    span.name,
                    xy=((span.start + span.stop - 1) / 2.0, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
        bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label("Pearson r")
        style.annotate_provenance(
            figure,
            [
                run_label,
                f"largest |r| off the diagonal: {worst:.2f}"
                + (f" ({worst_pair})" if worst_pair else ""),
                "constant term excluded",
            ],
        )
    return figure


def _role_spans(names: Sequence[str]) -> List[RegressorGroup]:
    """Contiguous role blocks over an already role-ordered list of columns."""
    spans: List[RegressorGroup] = []
    for index, name in enumerate(names):
        role = _role(name)
        if spans and spans[-1].name == role:
            spans[-1] = RegressorGroup(role, spans[-1].start, index + 1)
        else:
            spans.append(RegressorGroup(role, index, index + 1))
    return spans


def variance_inflation_figure(
    design_matrix: "pd.DataFrame",
    *,
    contrast: Optional[Dict[str, float]] = None,
    run_label: str = "",
    max_labelled: int = 30,
) -> Figure:
    """Variance inflation per regressor: how much collinearity costs each estimate.

    A wide design cannot label every bar, and unlabelled bars reduce the panel to
    "some regressor is inflated" -- which is not actionable, because the answer
    depends entirely on *which*. A variance inflation of 130 on a motion derivative's
    square is ordinary; the same number on a regressor the contrast weights means the
    comparison in the section above rests on almost no independent variance.

    So the panel names the roles, names the worst regressor outright, and marks the
    columns this contrast actually weights.
    """
    import matplotlib.pyplot as plt

    frame, _ordered, _groups, modelled, _vector = _prepare(design_matrix, contrast)
    vifs = variance_inflation_factors(frame[modelled].to_numpy(dtype=float))

    with style.plot_context():
        figure, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        if not vifs.size:
            ax.set_axis_off()
            return figure

        finite = np.isfinite(vifs)
        ceiling = float(np.nanmax(vifs[finite])) if finite.any() else 1.0
        plotted = np.where(finite, vifs, ceiling)
        positions = np.arange(vifs.size)
        weighted = {
            name for name, weight in (contrast or {}).items() if float(weight) != 0.0
        }

        colours = []
        for name, ok in zip(modelled, finite):
            if not ok:
                colours.append(style.OKABE_ITO["vermillion"])
            elif name in weighted:
                colours.append(style.OKABE_ITO["orange"])
            else:
                colours.append(style.OKABE_ITO["blue"])
        ax.bar(positions, plotted, color=colours, width=0.85)

        ax.set_yscale("log")
        ax.set_ylabel("VIF (log)")
        # Padded when the role bands are drawn: they sit at the top of the axes and
        # an unpadded title lands on them.
        ax.set_title(
            "Variance inflation per regressor",
            pad=20 if len(modelled) > max_labelled else None,
        )

        if len(modelled) <= max_labelled:
            ax.set_xticks(positions)
            ax.set_xticklabels(modelled, rotation=90, fontsize=6.0)
        else:
            # Role bands instead of names. Which role carries the inflation is the
            # distinction that decides whether it matters.
            ax.set_xticks([])
            ax.set_xlabel(f"{len(modelled)} modelled regressors, grouped by role")
            spans = _role_spans(modelled)
            for span in spans[:-1]:
                ax.axvline(span.stop - 0.5, color="0.6", linewidth=0.8)
            for span in spans:
                ax.annotate(
                    f"{span.name} ({span.size})",
                    xy=((span.start + span.stop - 1) / 2.0, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        # The worst regressor by name, always -- it is the single fact a reader takes
        # away, and on a wide design no tick label carries it.
        worst_index = int(np.argmax(~finite)) if not finite.all() else int(np.argmax(vifs))
        worst_name = modelled[worst_index]
        worst_value = "∞" if not finite[worst_index] else f"{vifs[worst_index]:.3g}"
        notes = [f"largest VIF: {worst_name} ({worst_value})"]
        if weighted:
            in_contrast = [i for i, name in enumerate(modelled) if name in weighted]
            if in_contrast:
                top = max(in_contrast, key=lambda i: (not finite[i], vifs[i]))
                value = "∞" if not finite[top] else f"{vifs[top]:.3g}"
                notes.append(f"largest among weighted: {modelled[top]} ({value})")

        legend = []
        if weighted:
            legend.append(("orange", "weighted by this contrast"))
        if not finite.all():
            legend.append(("vermillion", "perfectly collinear (VIF ∞)"))
        for offset, (colour, text) in enumerate(legend):
            ax.text(
                0.99,
                0.96 - 0.06 * offset,
                text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=style.OKABE_ITO[colour],
            )

        style.annotate_provenance(
            figure,
            [
                run_label,
                *notes,
                "VIF = 1/(1 - R²) of each regressor on the others",
                "constant term excluded",
            ],
        )
    return figure
