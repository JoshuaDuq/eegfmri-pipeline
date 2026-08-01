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
from html import escape
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fmri_pipeline.analysis.report import style


def _escape(value: object) -> str:
    """Escape a cell before it is interpolated into markup.

    Cells carry values from the data: a rejection region reads "z < -6.57",
    and a regressor name is whatever the design called it. Interpolated raw,
    the first of those opened a tag and swallowed the cell that contained it.
    """
    return escape("" if value is None else str(value))


__all__ = [
    "DesignSummary",
    "RegressorGroup",
    "classify_regressors",
    "contrast_efficiency",
    "count_events",
    "design_matrix_figure",
    "design_summary_table",
    "event_raster_figure",
    "onset_rows",
    "regressor_correlation_across_runs_figure",
    "regressor_correlation_figure",
    "summarize_design",
    "variance_inflation_factors",
    "variance_inflation_across_runs_figure",
    "variance_inflation_figure",
]


#: Most non-zero contrast weights that can be written into the strip legibly. Past
#: this the cells are narrower than the digits, and an unreadable number on the wrong
#: cell is worse than none: the colour still carries the sign.
_MAX_ANNOTATED_WEIGHTS = 14

#: Horizontal space one rotated regressor name needs to stay legible, in inches.
#:
#: These panels used to drop every name past a couple of dozen columns and fall back
#: on role bands. This study's designs carry 47 regressors, so in practice the VIF
#: panel drew 46 unlabelled bars -- a reader could see that something was inflated by
#: a factor of 130 and had no way to find out what. The panel widens with the design
#: instead: a name is what makes the measurement actionable, and width is cheap in a
#: report that scrolls.
_INCHES_PER_LABEL = 0.135

#: Past this many columns even a widened panel is wider than any screen, and role
#: bands become the honest summary. No design in this pipeline approaches it.
MAX_LABELLED_REGRESSORS = 90


def _labelled_width(n_labels: int, *, base: float, margin: float = 2.2) -> float:
    """Figure width that leaves every one of ``n_labels`` columns room for its name."""
    return float(max(base, _INCHES_PER_LABEL * int(n_labels) + margin))


def _label_fontsize(n_labels: int) -> float:
    """Font size for rotated regressor names, shrinking as the design grows."""
    if n_labels <= 28:
        return 6.4
    if n_labels <= 50:
        return 5.6
    return 4.8

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


#: Fraction of a regressor's own maximum that counts as an event starting.
#:
#: Taken relative to the column rather than as an absolute value: a convolved regressor
#: is scaled by the event duration, so a fixed cut would find every onset in a block
#: design and none in an event-related one.
ONSET_FRACTION = 0.35


def onset_rows(column: "np.ndarray", *, fraction: float = ONSET_FRACTION) -> "np.ndarray":
    """Rows where a convolved regressor rises through ``fraction`` of its maximum.

    Rising crossings only. A regressor crosses its own level twice per event, and
    counting both would double every count and, where this drives epoching, average
    each event's undershoot on top of its peak.

    A column already above its level at the first row counts as an onset there. Such
    an event began at or before the first modelled frame -- the design drops
    non-steady-state volumes, so a run whose first trial starts immediately loses its
    rising edge with them -- and reporting no event at all would say the run contained
    none.
    """
    values = np.asarray(column, dtype=float)
    if values.size < 1:
        return np.empty(0, dtype=int)
    peak = float(np.nanmax(values))
    if not np.isfinite(peak) or peak <= 0:
        return np.empty(0, dtype=int)
    above = values >= fraction * peak
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above[0]:
        rising = np.concatenate([[0], rising])
    return rising


def count_events(
    design_matrix: "pd.DataFrame", columns: Sequence[str]
) -> Dict[str, int]:
    """How many events of each named condition this run's design carries.

    The report described the model in every other respect -- its regressors, its
    conditioning, its confounds -- and never said how much data the contrast rested
    on. A contrast estimated from five trials and one estimated from fifty produce
    the same design matrix picture and the same column names.

    Counted from the convolved regressor rather than from an events file, because the
    events file is not in the manifest and the regressor is what the model actually
    used: a trial dropped by the model's own scoping never appears here, which is the
    number a reader wants.
    """
    counts: Dict[str, int] = {}
    for name in columns:
        if name in getattr(design_matrix, "columns", ()):
            counts[str(name)] = int(
                onset_rows(design_matrix[name].to_numpy(dtype=float)).size
            )
    return counts


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


def event_raster_figure(
    onsets_per_run: Sequence[Dict[str, "np.ndarray"]],
    *,
    run_labels: Sequence[str],
    condition_names: Sequence[str],
    tr_seconds: Optional[float] = None,
    title: str = "",
) -> Figure:
    """When each condition's events occurred, run by run.

    The summary table counts the events; it cannot show where they fell. Timing is
    what decides whether two conditions are separable at all: conditions that
    alternate are estimable, conditions that block against one another share their
    variance with drift, and a run whose conditions are ordered rather than
    interleaved confounds the contrast with time-on-task. None of that is visible in
    a count, in a design matrix thumbnail, or in a condition number.

    It also shows imbalance in place. Measured on this study, one run runs 8 events to
    3 and the next runs 3 to 8 -- a fact the report carried nowhere until the counts
    were tabulated, and one whose *shape* only a raster gives.
    """
    import matplotlib.pyplot as plt

    conditions = list(condition_names)
    runs = list(run_labels)
    if not conditions or not onsets_per_run:
        raise ValueError("An event raster needs at least one condition and one run.")

    scale = float(tr_seconds) if tr_seconds else 1.0
    unit = "Time (s)" if tr_seconds else "Design row"

    with style.plot_context():
        figure, ax = plt.subplots(
            figsize=(9.0, 0.42 * len(onsets_per_run) + 1.6), constrained_layout=True
        )

        # One lane per run, conditions offset within it, so a reader sees each run as
        # a line of trials rather than hunting across a grid.
        offsets = np.linspace(-0.24, 0.24, len(conditions)) if len(conditions) > 1 else [0.0]
        for run_index, per_condition in enumerate(onsets_per_run):
            for condition_index, name in enumerate(conditions):
                rows = per_condition.get(name)
                if rows is None or len(rows) == 0:
                    continue
                ax.eventplot(
                    np.asarray(rows, dtype=float) * scale,
                    lineoffsets=run_index + offsets[condition_index],
                    linelengths=0.36 / max(len(conditions), 1) * 1.7,
                    linewidths=1.6,
                    colors=style.OKABE_ITO[
                        ("vermillion", "blue", "bluish_green", "orange")[
                            condition_index % 4
                        ]
                    ],
                )

        labels = [
            str(runs[index]) if index < len(runs) else f"run-{index + 1:02d}"
            for index in range(len(onsets_per_run))
        ]
        ax.set_yticks(np.arange(len(onsets_per_run)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(len(onsets_per_run) - 0.5, -0.5)
        ax.set_xlabel(unit)
        if title:
            # Padded clear of the legend, which sits above the axes: placed inside, it
            # landed on the first run's own events.
            ax.set_title(title, pad=26)

        handles = [
            plt.Line2D(
                [],
                [],
                color=style.OKABE_ITO[
                    ("vermillion", "blue", "bluish_green", "orange")[index % 4]
                ],
                linewidth=2.0,
            )
            for index in range(len(conditions))
        ]
        ax.legend(
            handles,
            conditions,
            fontsize=7,
            frameon=False,
            ncol=len(conditions),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
        )

        totals = {
            name: sum(len(run.get(name, ())) for run in onsets_per_run)
            for name in conditions
        }
        style.annotate_provenance(
            figure,
            [
                f"{len(onsets_per_run)} run(s)",
                " · ".join(f"{name}: {count}" for name, count in totals.items()),
                "onsets are rising edges of each condition's own convolved regressor, "
                "so a trial the model's scoping dropped is already absent",
            ],
        )
        return figure


def _summary_number(value: Optional[float]) -> str:
    if value is None:
        return "not estimable"
    if not np.isfinite(value):
        return "∞"
    return f"{value:.3g}"


def contrast_confounding(
    frame: "pd.DataFrame", contrast: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """How far this run's contrast regressor is confounded with time and with drift.

    ``X @ c`` is the single time series the comparison actually tests. Correlating it
    with elapsed time measures time-on-task confounding: conditions that block against
    one another rather than interleaving make the contrast partly a measure of when in
    the run the scan happened. Correlating it with each drift column measures how much
    of the comparison the high-pass basis can absorb, and the strongest such
    correlation is what costs it -- an average over the basis dilutes exactly the
    column that does the damage.

    The event raster shows both and quantifies neither, which is why these are numbers
    rather than another picture.

    Both are measurements. A blocked design is *supposed* to correlate with time, so
    no cutoff is applied to either.
    """
    columns = [name for name in (contrast or {}) if float((contrast or {})[name]) != 0.0]
    empty = {"r_with_time": float("nan"), "r_with_drift_max": float("nan")}
    if not columns:
        return empty

    present = [name for name in columns if name in frame.columns]
    if not present:
        return empty

    weights = np.array([float(contrast[name]) for name in present], dtype=float)
    values = frame.loc[:, present].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return empty
    # An elementwise weighted sum rather than a matmul. The result is identical for a
    # handful of columns, and `@` dispatches to BLAS, which on this platform raises
    # spurious divide-by-zero and overflow flags on input that is entirely finite.
    regressor = (values * weights).sum(axis=1)

    def _r(other: np.ndarray) -> float:
        if regressor.size < 2 or np.std(regressor) == 0 or np.std(other) == 0:
            return float("nan")
        return float(np.corrcoef(regressor, other)[0, 1])

    drift_columns = [name for name in frame.columns if str(name).startswith("drift")]
    drift_correlations = [
        abs(_r(frame[name].to_numpy(dtype=float))) for name in drift_columns
    ]
    finite = [value for value in drift_correlations if np.isfinite(value)]

    return {
        "r_with_time": _r(np.linspace(-1.0, 1.0, regressor.size)),
        "r_with_drift_max": max(finite) if finite else float("nan"),
    }


def design_summary_table(
    summaries: Sequence[DesignSummary],
    *,
    run_labels: Sequence[str],
    event_counts: Sequence[Dict[str, int]] = (),
    condition_names: Sequence[str] = (),
    confounding: Sequence[Dict[str, float]] = (),
) -> Tuple[str, List[str]]:
    """Every run's design summary as one table, one row per run.

    Previously one key-value block per run: six runs of a six-run study produced six
    stacked blocks of the same seven labels, and comparing a condition number across
    runs meant scrolling between them. The comparison across runs is the entire reason
    these numbers are reported per run, and a column is what supports it.

    ``event_counts`` adds one column per condition the contrast weights, which is the
    figure the report never carried: how many events the contrast actually rests on.
    """
    conditions = list(condition_names)
    headers = [
        "Run",
        "Scans",
        "Regressors",
        "Rank",
        "Residual dof",
        "Condition number",
        "Largest VIF",
        "Most inflated",
        "Efficiency",
    ]
    headers.extend(f"Events: {name}" for name in conditions)
    if confounding:
        headers.extend(["r with elapsed time", "r with drift (max)"])

    rows: List[List[str]] = []
    for index, summary in enumerate(summaries):
        label = (
            str(run_labels[index])
            if index < len(run_labels)
            else f"run-{index + 1:02d}"
        )
        row = [
            label,
            f"{summary.n_scans:,}",
            str(summary.n_regressors),
            # Rank beside the count. A design whose columns are linearly dependent
            # carries fewer parameters than it appears to, and no other line says so.
            str(summary.rank)
            if summary.rank == summary.n_regressors
            else f"{summary.rank} (deficient)",
            f"{summary.residual_dof:,}",
            _summary_number(summary.condition_number),
            _summary_number(summary.max_vif),
            summary.max_vif_regressor or "n/a",
            _summary_number(summary.efficiency),
        ]
        counts = event_counts[index] if index < len(event_counts) else {}
        row.extend(
            str(counts[name]) if name in counts else "n/a" for name in conditions
        )
        if confounding:
            measured = confounding[index] if index < len(confounding) else {}
            row.extend(
                _summary_number(measured.get(key))
                for key in ("r_with_time", "r_with_drift_max")
            )
        rows.append(row)

    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    tsv = ["\t".join(headers)]
    tsv.extend("\t".join(row) for row in rows)
    return table_html, tsv


def design_matrix_figure(
    design_matrix: "pd.DataFrame",
    *,
    contrast: Optional[Dict[str, float]] = None,
    tr_seconds: Optional[float] = None,
    run_label: str = "",
    max_labelled_columns: int = MAX_LABELLED_REGRESSORS,
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
            # Widened to fit the column names rather than dropping them; see
            # _INCHES_PER_LABEL.
            figsize=(_labelled_width(n_regressors, base=7.4), 6.4),
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

        # Every column keeps its name. This used to label only the task block past 28
        # regressors, on the reasoning that the nuisance block is not inspected by
        # name -- but the panel exists to say which weight lands on which regressor,
        # and a confound the reader cannot name is one they cannot check the model
        # for. The figure widens instead; only a design past
        # MAX_LABELLED_REGRESSORS falls back on role bands.
        #
        # Set on ax_contrast, the lower of the two shared axes. Under `sharex` the upper
        # axes' tick labels are hidden and the lower axes renders its own from the shared
        # locator -- without the rotation, because rotation belongs to the Text objects
        # that were created on the axes we set it on. Setting these on `ax` therefore
        # produced horizontal labels that overlapped into an unreadable smear, on a panel
        # whose whole purpose is to say which weight lands on which regressor.
        if n_regressors <= max_labelled_columns:
            ticks = list(range(n_regressors))
            fontsize = _label_fontsize(n_regressors)
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
    max_labelled: int = MAX_LABELLED_REGRESSORS,
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
        # Square, and wide enough for the names on both axes.
        side = _labelled_width(len(modelled), base=6.0, margin=2.6)
        figure, ax = plt.subplots(figsize=(side, side * 0.92), constrained_layout=True)
        image = ax.imshow(
            correlation, cmap=style.SIGNED_CMAP, vmin=-1, vmax=1, interpolation="nearest"
        )
        # Padded when the role bands are drawn: they sit at the top of the axes.
        ax.set_title(
            "Regressor correlation", pad=16 if len(modelled) > max_labelled else None
        )
        if len(modelled) <= max_labelled:
            fontsize = _label_fontsize(len(modelled))
            ax.set_xticks(range(len(modelled)))
            ax.set_xticklabels(modelled, rotation=90, fontsize=fontsize)
            ax.set_yticks(range(len(modelled)))
            ax.set_yticklabels(modelled, fontsize=fontsize)
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


def _shared_regressors(frames: Sequence["pd.DataFrame"]) -> List[str]:
    """Modelled columns every run has, in role order.

    The intersection, not the union. A regressor one run lacks has no value to compare
    across runs, and padding it with a placeholder would put a gap in a panel whose
    entire reading is how a quantity varies between runs.
    """
    if not frames:
        return []
    shared: Optional[set] = None
    for frame in frames:
        columns = set(_prepare(frame, None)[3])
        shared = columns if shared is None else (shared & columns)
    ordered, _groups = classify_regressors(sorted(shared or set()))
    return [name for name in ordered if _role(name) != "Constant"]


def variance_inflation_across_runs_figure(
    frames: Sequence["pd.DataFrame"],
    *,
    contrast: Optional[Dict[str, float]] = None,
    run_labels: Sequence[str] = (),
    max_labelled: int = MAX_LABELLED_REGRESSORS,
    title: str = "",
) -> Figure:
    """Variance inflation per regressor, every run on one panel.

    Six runs of a six-run study drew six near-identical bar charts, one per run, and a
    reader comparing a regressor's inflation between them had to hold six pictures in
    mind. The between-run comparison is the reading, so it belongs on one axis.

    This adds information rather than only saving space: the spread across runs was
    never shown. A regressor inflated in every run is a property of the design; one
    inflated in a single run is a property of that run -- a lost condition, a censored
    block -- and the two call for different responses.
    """
    import matplotlib.pyplot as plt

    modelled = _shared_regressors(frames)
    if not modelled or not frames:
        raise ValueError("No regressor is present in every run, so none can be compared.")

    stacked = np.vstack(
        [
            variance_inflation_factors(
                _prepare(frame, None)[0][modelled].to_numpy(dtype=float)
            )
            for frame in frames
        ]
    )
    finite = np.isfinite(stacked)
    weighted = {
        name for name, weight in (contrast or {}).items() if float(weight) != 0.0
    }
    positions = np.arange(len(modelled))

    # The regressors the contrast weights are drawn apart from the rest. Inflation on
    # those is what costs the comparison its precision; inflation on a nuisance
    # regressor costs it nothing. Marked in colour they were still two bars among
    # forty-seven, at a height four dozen other bars also reach.
    #
    # Only when both groups are non-empty: with no contrast there is nothing to
    # separate, and with every regressor weighted the remainder axis would be blank
    # under a role band.
    weighted_at = [index for index, name in enumerate(modelled) if name in weighted]
    rest_at = [index for index in positions if modelled[index] not in weighted]
    split = bool(weighted_at) and bool(rest_at)

    with style.plot_context():
        width = _labelled_width(len(modelled), base=7.0)
        if split:
            figure, (ax_contrast, ax) = plt.subplots(
                2,
                1,
                figsize=(width, 6.0),
                sharey=True,
                gridspec_kw={"height_ratios": (1.0, 3.0)},
                constrained_layout=True,
            )
            ax_contrast.set_label("vif-contrast")
            ax.set_label("vif-rest")
        else:
            figure, ax = plt.subplots(
                figsize=(width, 4.6), constrained_layout=True
            )
            ax_contrast = None
            ax.set_label("vif-all")

        # An infinite VIF is exact collinearity, which has no place on a log axis and
        # is not a large number -- it is a different fact. Drawn at the top of the
        # axis in the marker reserved for it, and named in the provenance.
        ceiling = float(np.nanmax(stacked[finite])) if finite.any() else 1.0

        def _draw(axis, source_indices) -> None:
            for slot, index in enumerate(source_indices):
                column = stacked[:, index]
                usable = column[np.isfinite(column)]
                colour = (
                    style.OKABE_ITO["orange"]
                    if modelled[index] in weighted
                    else style.OKABE_ITO["blue"]
                )
                if usable.size:
                    axis.plot(
                        [slot, slot],
                        [float(usable.min()), float(usable.max())],
                        color=colour,
                        linewidth=1.4,
                        alpha=0.45,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    axis.scatter(
                        [slot], [float(np.median(usable))], s=16, color=colour, zorder=3
                    )
                if usable.size < column.size:
                    axis.scatter(
                        [slot],
                        [ceiling],
                        s=26,
                        marker="^",
                        color=style.OKABE_ITO["vermillion"],
                        zorder=4,
                    )

        def _label(axis, source_indices, *, roles: bool, rotate: bool = True) -> None:
            names = [modelled[index] for index in source_indices]
            slots = np.arange(len(names))
            if roles:
                spans = _role_spans(names)
                for span in spans[:-1]:
                    axis.axvline(span.stop - 0.5, color="0.6", linewidth=0.8)
                for span in spans:
                    axis.annotate(
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
            if len(names) <= max_labelled:
                axis.set_xticks(slots)
                # A handful of names fit upright. Rotated, the contrast panel's own
                # labels are taller than the panel and run into the axis beneath it.
                axis.set_xticklabels(
                    names,
                    rotation=90 if rotate else 0,
                    fontsize=_label_fontsize(len(modelled)),
                )
                for label, name in zip(axis.get_xticklabels(), names):
                    if name in weighted:
                        label.set_color(style.OKABE_ITO["orange"])
                        label.set_fontweight("bold")
            else:
                axis.set_xticks([])
                axis.set_xlabel(f"{len(names)} modelled regressors, grouped by role")
            axis.set_xlim(-0.8, len(names) - 0.2)

        ax.set_yscale("log")
        ax.set_ylabel("VIF (log)")

        if split:
            _draw(ax_contrast, weighted_at)
            _label(
                ax_contrast, weighted_at, roles=False, rotate=len(weighted_at) > 6
            )
            ax_contrast.set_yscale("log")
            ax_contrast.set_ylabel("VIF (log)")
            ax_contrast.set_title(
                title or "Variance inflation per regressor", pad=20
            )
            ax_contrast.annotate(
                "Regressors the contrast weights",
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(0, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=style.OKABE_ITO["orange"],
            )
            _draw(ax, rest_at)
            _label(ax, rest_at, roles=True)
        else:
            ax.set_title(title or "Variance inflation per regressor", pad=20)
            _draw(ax, list(positions))
            _label(ax, list(positions), roles=True)

        medians = np.array(
            [
                float(np.median(stacked[:, i][np.isfinite(stacked[:, i])]))
                if np.isfinite(stacked[:, i]).any()
                else np.nan
                for i in positions
            ]
        )
        worst = int(np.nanargmax(medians)) if np.isfinite(medians).any() else 0
        notes = [
            f"{stacked.shape[0]} run(s) · dot: median across runs, bar: range",
            f"largest median VIF: {modelled[worst]} ({medians[worst]:.3g})",
        ]
        if not finite.all():
            notes.append(
                f"{int((~finite).sum())} run-regressor pair(s) exactly collinear, "
                "marked ▲ at the top of the axis"
            )
        notes.extend(
            [
                "VIF = 1/(1 - R²) of each regressor on the others",
                "constant term excluded; reported as a measurement, with no cutoff",
            ]
        )
        style.annotate_provenance(figure, notes)
        return figure


def regressor_correlation_across_runs_figure(
    frames: Sequence["pd.DataFrame"],
    *,
    run_labels: Sequence[str] = (),
    max_labelled: int = MAX_LABELLED_REGRESSORS,
    title: str = "",
) -> Figure:
    """The strongest correlation each pair of regressors reaches in any run.

    One matrix rather than one per run. The question a correlation panel answers is
    whether any two columns duplicate one another, which is a question about magnitude
    and about the worst case: a pair that is collinear in a single run costs the
    contrast its precision in that run, and a per-run average would dilute exactly that
    away.

    The magnitude, so sign is deliberately absent. Sign is a property of a particular
    pair in a particular run and does not survive a maximum; showing a signed value
    that came from whichever run happened to be most extreme would name a number no
    single run holds.
    """
    import matplotlib.pyplot as plt

    modelled = _shared_regressors(frames)
    if len(modelled) < 2:
        figure, ax = plt.subplots(figsize=(6.0, 2.0), constrained_layout=True)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"{len(modelled)} shared modelled regressor(s): nothing to correlate",
            ha="center",
            va="center",
            fontsize=9,
        )
        style.annotate_provenance(figure, ["constant term excluded"])
        return figure

    worst = np.zeros((len(modelled), len(modelled)), dtype=float)
    for frame in frames:
        values = _prepare(frame, None)[0][modelled].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            correlation = np.abs(
                np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
            )
        worst = np.maximum(worst, correlation)

    off_diagonal = worst.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    flat = int(np.argmax(off_diagonal))
    row, column = divmod(flat, off_diagonal.shape[1])

    with style.plot_context():
        side = _labelled_width(len(modelled), base=6.0, margin=2.6)
        figure, ax = plt.subplots(figsize=(side, side * 0.92), constrained_layout=True)
        # Sequential, not diverging: this is a magnitude with a floor at zero, and a
        # diverging map would reserve half its range for values that cannot occur.
        image = ax.imshow(
            worst, cmap=style.MAGNITUDE_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        ax.set_title(
            title or "Regressor correlation (strongest across runs)",
            pad=16 if len(modelled) > max_labelled else None,
        )
        if len(modelled) <= max_labelled:
            fontsize = _label_fontsize(len(modelled))
            ax.set_xticks(range(len(modelled)))
            ax.set_xticklabels(modelled, rotation=90, fontsize=fontsize)
            ax.set_yticks(range(len(modelled)))
            ax.set_yticklabels(modelled, fontsize=fontsize)
        else:
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
        bar.set_label("|r|, strongest across runs")
        style.annotate_provenance(
            figure,
            [
                f"{len(frames)} run(s)",
                f"largest |r| off the diagonal: {off_diagonal[row, column]:.2f} "
                f"({modelled[row]} / {modelled[column]})",
                "magnitude only: a sign taken from whichever run was most extreme "
                "would name a number no single run holds",
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
    max_labelled: int = MAX_LABELLED_REGRESSORS,
) -> Figure:
    """Variance inflation per regressor: how much collinearity costs each estimate.

    Unlabelled bars reduce the panel to "some regressor is inflated" -- which is not
    actionable, because the answer depends entirely on *which*. A variance inflation
    of 130 on a motion derivative's square is ordinary; the same number on a regressor
    the contrast weights means the comparison in the section above rests on almost no
    independent variance.

    So every bar is named, the panel widening with the design to make room; the roles
    are banded; the worst regressor is named outright; and the columns this contrast
    weights are marked. Only a design past ``max_labelled`` falls back on role bands
    alone.
    """
    import matplotlib.pyplot as plt

    frame, _ordered, _groups, modelled, _vector = _prepare(design_matrix, contrast)
    vifs = variance_inflation_factors(frame[modelled].to_numpy(dtype=float))

    with style.plot_context():
        figure, ax = plt.subplots(
            figsize=(_labelled_width(len(modelled), base=7.0), 4.6),
            constrained_layout=True,
        )
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
        # The role bands sit at the top of the axes, so the title is always padded
        # clear of them.
        ax.set_title("Variance inflation per regressor", pad=20)

        # Roles band the axis whether or not the bars are named. Which role carries
        # the inflation is the distinction that decides whether it matters, and it
        # stays legible at a glance in a way forty-six names do not.
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

        if len(modelled) <= max_labelled:
            ax.set_xticks(positions)
            ax.set_xticklabels(
                modelled, rotation=90, fontsize=_label_fontsize(len(modelled))
            )
            # The weighted regressors again, in the tick labels. Colour already marks
            # them, but a reader scanning names for the contrast's own columns should
            # not have to match a bar to a swatch.
            for label, name in zip(ax.get_xticklabels(), modelled):
                if name in weighted:
                    label.set_color(style.OKABE_ITO["orange"])
                    label.set_fontweight("bold")
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"{len(modelled)} modelled regressors, grouped by role")

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
