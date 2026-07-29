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


def motion_table(summaries: Sequence[RunMotion]) -> Tuple[str, List[str]]:
    """Return the per-run motion numbers as an HTML table and its TSV rows.

    A table rather than a figure because these are the values a methods section
    quotes verbatim, and a number read off a dot plot is not quotable.
    """
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

    rows: List[List[str]] = []
    for run in summaries:
        fractions = run.fraction_above or tuple(
            None for _ in FD_REFERENCE_LEVELS
        )
        rows.append(
            [
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
        )

    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    tsv = ["\t".join(headers)]
    tsv.extend("\t".join(row) for row in rows)
    return table_html, tsv


def write_motion_tsv(summaries: Sequence[RunMotion], *, path: Path) -> Path:
    """Write the motion table beside the report, for a cohort script to read."""
    _html, rows = motion_table(summaries)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


__all__ = [
    "FD_REFERENCE_LEVELS",
    "RunMotion",
    "motion_table",
    "run_motion_figure",
    "summarise_run_motion",
    "write_motion_tsv",
]
