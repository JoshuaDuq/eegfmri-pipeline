"""Head motion and censoring, as numbers, per run.

The carpet draws framewise displacement as a trace, which answers "was there a spike,
and did the carpet band with it". It cannot answer the questions a method section has
to: how much motion was there, how much of it survived into the model, and is one run
unlike the others. Those are the most widely reported quality figures in the field --
Power et al. (2012, 2014), and every exclusion criterion built on them -- and the
report stated none of them.

Nothing here is a verdict. The published reference levels are drawn and named so that
a run can be located against a convention a reader already knows; which runs that
makes acceptable is a study's decision, not this module's.
"""

from __future__ import annotations

import logging
from html import escape
from dataclasses import dataclass
from pathlib import Path
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


def _escape(value: object) -> str:
    """Escape a cell before it is interpolated into markup.

    Cells carry values from the data: a rejection region reads "z < -6.57",
    and a regressor name is whatever the design called it. Interpolated raw,
    the first of those opened a tag and swallowed the cell that contained it.
    """
    return escape("" if value is None else str(value))

#: Framewise-displacement levels worth locating a run against, with their source.
#:
#: References, not thresholds. Each names a published convention; none was invented
#: here, and no run is scored against them.
FD_REFERENCE_LEVELS: Tuple[Tuple[float, str], ...] = (
    (0.2, "0.2 mm (Power et al. 2014)"),
    (0.5, "0.5 mm (Power et al. 2012)"),
)


@dataclass(frozen=True)
class RunMotion:
    """One run's head motion and what censoring removed from it."""

    label: str
    n_frames: int
    n_censored: int
    #: Quartiles and extremes of framewise displacement, in millimetres. ``None``
    #: throughout when the run's confounds carry no framewise displacement column,
    #: which is a fact about the derivatives rather than a run without motion.
    median_fd: Optional[float]
    q1_fd: Optional[float]
    q3_fd: Optional[float]
    max_fd: Optional[float]
    mean_fd: Optional[float]
    #: Fraction of frames at or above each level in :data:`FD_REFERENCE_LEVELS`, in
    #: the same order.
    fraction_above: Tuple[float, ...] = ()

    @property
    def n_retained(self) -> int:
        return self.n_frames - self.n_censored


def _fd_column(frame: Any) -> Optional[np.ndarray]:
    if "framewise_displacement" not in frame.columns:
        return None
    values = frame["framewise_displacement"].to_numpy(dtype=float)
    # The first frame of a run has no defined displacement and fMRIPrep writes it as
    # n/a. Treating that as a zero would pull every summary statistic down by one
    # fabricated sample.
    return values[np.isfinite(values)]


def summarise_run_motion(
    confounds_paths: Sequence[Any],
    *,
    run_labels: Sequence[str],
    sample_masks: Optional[Sequence[np.ndarray]] = None,
) -> List[RunMotion]:
    """Read framewise displacement and censoring for each run.

    ``sample_masks`` is the same per-run keep-mask the GLM used, so the censored count
    reported here is the count the model actually applied rather than one recomputed
    under a different rule. A run whose confounds cannot be read is skipped with a
    warning: one unreadable file should cost its own row, not the panel.
    """
    import pandas as pd

    summaries: List[RunMotion] = []
    for index, path in enumerate(confounds_paths):
        label = (
            str(run_labels[index])
            if index < len(run_labels)
            else f"run-{index + 1:02d}"
        )
        try:
            frame = pd.read_csv(str(path), sep="\t")
        except (OSError, ValueError) as exc:
            logger.warning("Could not read confounds %s (%s)", path, exc)
            continue

        n_frames = int(len(frame))
        n_censored = 0
        if sample_masks is not None and index < len(sample_masks):
            keep = np.asarray(sample_masks[index], dtype=bool)
            if keep.size == n_frames:
                n_censored = int(n_frames - int(keep.sum()))

        fd = _fd_column(frame)
        if fd is None or fd.size == 0:
            summaries.append(
                RunMotion(
                    label=label,
                    n_frames=n_frames,
                    n_censored=n_censored,
                    median_fd=None,
                    q1_fd=None,
                    q3_fd=None,
                    max_fd=None,
                    mean_fd=None,
                )
            )
            continue

        summaries.append(
            RunMotion(
                label=label,
                n_frames=n_frames,
                n_censored=n_censored,
                median_fd=float(np.median(fd)),
                q1_fd=float(np.percentile(fd, 25)),
                q3_fd=float(np.percentile(fd, 75)),
                max_fd=float(np.max(fd)),
                mean_fd=float(np.mean(fd)),
                fraction_above=tuple(
                    float(np.mean(fd >= level)) for level, _label in FD_REFERENCE_LEVELS
                ),
            )
        )
    return summaries


def _measured(summaries: Sequence[RunMotion]) -> List[RunMotion]:
    return [run for run in summaries if run.median_fd is not None]


def run_motion_figure(
    summaries: Sequence[RunMotion], *, title: str = ""
) -> plt.Figure:
    """Draw each run's framewise displacement against the published reference levels.

    Dots with an interquartile bar and a separate marker for the maximum, matching the
    per-run tSNR panel: the question is whether any run is unlike the others, which is
    a relative comparison, and a bar chart from zero would spend the whole axis on the
    distance from an origin no reader is asking about.

    The maximum is drawn apart from the quartiles rather than as a whisker. Motion is
    dominated by isolated spikes -- measured here, runs whose interquartile range spans
    0.05 mm reach 0.27 mm -- so the single worst frame and the typical frame are two
    different measurements, and a reader deciding whether censoring was needed wants
    both.

    The x axis starts at zero, unlike the tSNR panel's, because displacement is a
    magnitude with a real origin: no motion is a value this quantity can take, and the
    distance from it is the reading.
    """
    measured = _measured(summaries)
    if not measured:
        raise ValueError(
            "A motion panel requires at least one run with framewise displacement."
        )

    positions = np.arange(len(measured))
    medians = np.array([run.median_fd for run in measured], dtype=float)
    maxima = np.array([run.max_fd for run in measured], dtype=float)

    with plot_context():
        figure, axis = plt.subplots(figsize=(7.0, 0.42 * len(measured) + 2.1))

        for index, run in enumerate(measured):
            axis.plot(
                [run.q1_fd, run.q3_fd],
                [index, index],
                color=OKABE_ITO["sky_blue"],
                linewidth=3.0,
                solid_capstyle="butt",
                alpha=0.55,
                zorder=2,
            )
        axis.scatter(
            medians, positions, s=42, color=OKABE_ITO["blue"], zorder=4, label="median"
        )
        axis.scatter(
            maxima,
            positions,
            s=34,
            facecolors="none",
            edgecolors=OKABE_ITO["vermillion"],
            linewidths=1.2,
            zorder=3,
            label="worst frame",
        )

        # Reference levels are drawn only where they fall within reach of the data. A
        # line far beyond every sample adds no comparison and costs the panel the
        # resolution that separates the runs from one another.
        ceiling = float(np.max(maxima))
        for level, label in FD_REFERENCE_LEVELS:
            if level > ceiling * 1.6:
                continue
            axis.axvline(level, color=GUIDE_COLOR, linestyle=":", linewidth=0.9, zorder=1)
            axis.annotate(
                label,
                xy=(level, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                color=GUIDE_COLOR,
                rotation=90,
                ha="left",
                va="bottom",
            )

        labels = [
            f"{run.label}\n({run.n_censored} censored)" if run.n_censored else run.label
            for run in measured
        ]
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8)
        axis.set_ylim(len(measured) - 0.5, -0.5)
        axis.set_xlim(left=0.0)
        axis.set_xlabel("Framewise displacement (mm)")
        # Above the axes, not inside it. Every horizontal position in this panel can
        # hold a marker -- a run's worst frame lands wherever it lands -- so an in-axes
        # legend has no corner it is guaranteed not to cover, and it covered the last
        # run's on real data.
        axis.legend(
            fontsize=7,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=2,
            framealpha=0.0,
        )
        if title:
            axis.set_title(title, loc="left")

        unmeasured = len(summaries) - len(measured)
        provenance = [
            f"{len(measured)} run(s)",
            f"{sum(r.n_frames for r in summaries):,} frames, "
            f"{sum(r.n_censored for r in summaries):,} censored",
            f"worst frame across runs: {ceiling:.2f} mm",
            "reference levels are published conventions, not criteria applied here",
        ]
        if unmeasured:
            # A run without the column is not a run without motion, and the panel must
            # not let the two look the same by quietly showing one row fewer.
            provenance.append(
                f"{unmeasured} run(s) carry no framewise_displacement column"
            )
        annotate_provenance(figure, provenance)
        figure.tight_layout()
        return figure


def _number(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def motion_table(
    summaries: Sequence[RunMotion],
    *,
    tsnr_median: Optional[Sequence[float]] = None,
    tsnr_iqr: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[str, List[str]]:
    """Return the per-run QC numbers as an HTML table and its TSV rows.

    A table rather than a figure because these are the values a methods section
    quotes verbatim, and a number read off a dot plot is not quotable.

    tSNR joins the motion columns rather than getting a panel of its own. It used to
    have one, and the panel's own docstring recorded why it did not work: six runs
    between 58.6 and 60.5 drew six visually indistinguishable marks. tSNR carries no
    published reference level to compare against -- unlike framewise displacement,
    which keeps its figure for exactly that reason -- so "is one run unlike the
    others" is a comparison between six numbers, and six numbers are a column.
    """
    with_tsnr = tsnr_median is not None and len(tsnr_median) == len(summaries)
    headers = [
        "Run",
        "Frames",
        "Censored",
        "Retained",
        "Median FD (mm)",
        "Mean FD (mm)",
        "Max FD (mm)",
        *[f"% ≥ {level:g} mm" for level, _label in FD_REFERENCE_LEVELS],
    ]
    if with_tsnr:
        headers.extend(["Median tSNR", "tSNR IQR"])

    rows: List[List[str]] = []
    for index, run in enumerate(summaries):
        fractions = run.fraction_above or tuple(
            None for _ in FD_REFERENCE_LEVELS
        )
        row = [
            run.label,
            f"{run.n_frames:,}",
            f"{run.n_censored:,}",
            f"{run.n_retained:,}",
            _number(run.median_fd),
            _number(run.mean_fd),
            _number(run.max_fd),
            *[
                "n/a" if fraction is None else f"{100.0 * fraction:.1f}"
                for fraction in fractions
            ],
        ]
        if with_tsnr:
            row.append(f"{float(tsnr_median[index]):.1f}")
            quartiles = (
                tsnr_iqr[index]
                if tsnr_iqr is not None and index < len(tsnr_iqr)
                else None
            )
            row.append(
                "n/a"
                if quartiles is None
                else f"{float(quartiles[0]):.1f}–{float(quartiles[1]):.1f}"
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


def write_motion_tsv(
    summaries: Sequence[RunMotion],
    *,
    path: Path,
    tsnr_median: Optional[Sequence[float]] = None,
    tsnr_iqr: Optional[Sequence[Tuple[float, float]]] = None,
) -> Path:
    """Write the run-level QC table beside the report, for a cohort script to read."""
    _html, rows = motion_table(
        summaries, tsnr_median=tsnr_median, tsnr_iqr=tsnr_iqr
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _dvars_column(frame: Any) -> Tuple[Optional[np.ndarray], str]:
    """Return a run's DVARS trace and the column it came from."""
    for column, label in (("std_dvars", "std DVARS"), ("dvars", "DVARS")):
        if column in getattr(frame, "columns", ()):
            return frame[column].to_numpy(dtype=float), label
    return None, "DVARS"


def _raw_fd_column(frame: Any) -> Optional[np.ndarray]:
    """A run's framewise displacement with its frames intact, gaps included.

    Distinct from :func:`_fd_column`, which drops the undefined first frame because
    its callers want summary statistics over real measurements. Anything pairing
    framewise displacement against another per-frame trace needs the frames to still
    line up, and dropping one array's gaps but not the other's silently pairs every
    subsequent frame with its neighbour's value.
    """
    if "framewise_displacement" not in getattr(frame, "columns", ()):
        return None
    return frame["framewise_displacement"].to_numpy(dtype=float)


def motion_coupling_figure(
    confounds_paths: Sequence[Any],
    *,
    run_labels: Sequence[str] = (),
    title: str = "",
) -> plt.Figure:
    """Framewise displacement against DVARS, frame by frame.

    Both traces already share the carpet's time axis, where they answer "was there a
    spike". They do not answer the question that decides whether motion contaminated
    the result: how tightly the two move *together*. DVARS that tracks framewise
    displacement is signal change driven by head motion; DVARS that moves independently
    of it is something else -- a physiological or hardware source that motion
    regressors will not remove and censoring on framewise displacement will not catch.

    Nothing here is scored. The correlation is reported as a measurement, and the
    reference levels are the same published conventions the per-run panel draws.
    """
    import pandas as pd

    fd_parts: List[np.ndarray] = []
    dvars_parts: List[np.ndarray] = []
    dvars_label = "DVARS"

    for path in confounds_paths:
        try:
            frame = pd.read_csv(str(path), sep="\t")
        except (OSError, ValueError) as exc:
            logger.warning("Could not read confounds %s (%s)", path, exc)
            continue
        fd = _raw_fd_column(frame)
        dvars, dvars_label = _dvars_column(frame)
        if fd is None or dvars is None or fd.size != dvars.size:
            continue
        fd_parts.append(fd)
        dvars_parts.append(dvars)

    if not fd_parts:
        raise ValueError(
            "No run supplied both framewise displacement and DVARS; there is nothing "
            "to relate."
        )

    # The first frame of a run has no defined framewise displacement, and fMRIPrep
    # writes it as NaN. Pairing has to drop those frames from both traces at once.
    paired: List[Tuple[np.ndarray, np.ndarray]] = []
    for fd, dvars in zip(fd_parts, dvars_parts):
        usable = np.isfinite(fd) & np.isfinite(dvars)
        if int(usable.sum()) >= 3:
            paired.append((fd[usable], dvars[usable]))
    if not paired:
        raise ValueError("Too few paired frames to relate motion to signal change.")

    fd_all = np.concatenate([fd for fd, _dvars in paired])
    dvars_all = np.concatenate([dvars for _fd, dvars in paired])
    if fd_all.size < 3:
        raise ValueError("Too few paired frames to relate motion to signal change.")

    # Measured within runs, not across them. Concatenating first and correlating once
    # mixes the coupling this panel is about with any difference in baseline between
    # runs: two runs each with no internal coupling, offset from one another, produce
    # a strong pooled correlation that describes only the offset. Centring each run
    # removes the between-run component and leaves the within-run relationship.
    centred_fd = np.concatenate([fd - fd.mean() for fd, _dvars in paired])
    centred_dvars = np.concatenate([dvars - dvars.mean() for _fd, dvars in paired])
    correlation = float(np.corrcoef(centred_fd, centred_dvars)[0, 1])

    per_run = [
        float(np.corrcoef(fd, dvars)[0, 1])
        for fd, dvars in paired
        if np.std(fd) > 0 and np.std(dvars) > 0
    ]

    with plot_context():
        figure, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
        ax.scatter(
            fd_all,
            dvars_all,
            s=4.0,
            alpha=0.35,
            color=OKABE_ITO["blue"],
            linewidths=0,
        )
        for level, label in FD_REFERENCE_LEVELS:
            if level <= float(np.nanmax(fd_all)):
                ax.axvline(
                    level, color=GUIDE_COLOR, linestyle=":", linewidth=0.9, zorder=1
                )
                ax.annotate(
                    label,
                    xy=(level, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -4),
                    textcoords="offset points",
                    rotation=90,
                    va="top",
                    fontsize=6.5,
                    color=GUIDE_COLOR,
                )
        # The relationship this panel exists to show, drawn. Reported only in the
        # provenance strip, the correlation was a number beside a cloud in which the
        # reader could not see it -- and how tightly the two move together is the
        # entire reading.
        #
        # Binned medians rather than a least-squares line: framewise displacement is
        # spike-dominated and strongly right-skewed, so a line is levered by the few
        # extreme frames it is least able to describe. The medians also show curvature,
        # which a straight fit reports as a weaker linear relationship instead.
        order = np.argsort(fd_all)
        n_bins = int(np.clip(fd_all.size // 250, 4, 12))
        bins = np.array_split(order, n_bins)
        centres = np.array([float(np.median(fd_all[chunk])) for chunk in bins])
        medians = np.array([float(np.median(dvars_all[chunk])) for chunk in bins])
        ax.plot(
            centres,
            medians,
            color=OKABE_ITO["vermillion"],
            linewidth=1.6,
            marker="o",
            markersize=4.0,
            zorder=3,
            label=f"median {dvars_label} per displacement bin",
        )
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.annotate(
            f"within-run r = {correlation:+.2f}",
            xy=(0.985, 0.03),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color=GUIDE_COLOR,
        )

        ax.set_xlabel("Framewise displacement (mm)")
        ax.set_ylabel(dvars_label)
        if title:
            ax.set_title(title)

        notes = [
            f"{fd_all.size:,} paired frames across {len(paired)} run(s)",
            f"within-run r = {correlation:+.2f} (each run centred first)",
        ]
        if len(per_run) > 1:
            labels = [
                str(run_labels[index]) if index < len(run_labels) else f"run-{index + 1:02d}"
                for index in range(len(per_run))
            ]
            weakest = labels[int(np.argmin(per_run))]
            strongest = labels[int(np.argmax(per_run))]
            notes.append(
                f"per run {min(per_run):+.2f} ({weakest}) to {max(per_run):+.2f} "
                f"({strongest})"
            )
        notes.extend(
            [
                "frames without a defined framewise displacement are excluded from "
                "both axes",
                "reference levels are published conventions, not criteria applied here",
            ]
        )
        annotate_provenance(figure, notes)
        return figure


__all__ = [
    "motion_coupling_figure",
    "FD_REFERENCE_LEVELS",
    "RunMotion",
    "motion_table",
    "run_motion_figure",
    "summarise_run_motion",
    "write_motion_tsv",
]
