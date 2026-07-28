"""Cohort roll-up of the BrainVision Analyzer correction quality.

Per-run QC answers "is this run usable". It cannot answer "how much of the study is
affected", and that is the question that decides whether a group analysis is viable.
Analyzer reports some of its own failures interactively, but not all of them: a run can
end up with no usable R markers without any message being raised, so counting the
markers that actually reached the files is the only reliable measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    apply_report_style,
)
from eeg_pipeline.preprocessing.report.tables import Metric, metric_table

#: Default reference ratio of R markers to scanner volumes, overridable through
#: ``report.thresholds.min_r_markers_per_volume``. Analyzer writes one R marker per
#: heartbeat and one marker per volume, so at a short repetition time a complete marker
#: set sits near one per volume. The value is a reporting reference, not a verdict.
MINIMUM_R_MARKERS_PER_VOLUME = 0.5


@dataclass(frozen=True)
class CohortMarkerQc:
    """Per-run R-marker counts across the cohort."""

    runs: pd.DataFrame

    @property
    def usable(self) -> pd.DataFrame:
        return self.runs[~self.runs["is_broken"]]

    @property
    def broken(self) -> pd.DataFrame:
        return self.runs[self.runs["is_broken"]]

    @property
    def broken_fraction(self) -> float:
        return float(self.runs["is_broken"].mean()) if len(self.runs) else 0.0

    def subjects_fully_usable(self) -> list[str]:
        by_subject = self.runs.groupby("subject")["is_broken"].any()
        return sorted(by_subject.index[~by_subject].astype(str))


def summarize_cohort_markers(
    runs: pd.DataFrame,
    *,
    subjects: Sequence[str] | None = None,
    minimum_markers_per_volume: float = MINIMUM_R_MARKERS_PER_VOLUME,
) -> CohortMarkerQc:
    """Flag runs whose R-marker count is too low to have driven a pulse correction.

    ``runs`` needs ``subject``, ``run``, ``n_r_markers`` and ``n_volumes`` columns.

    ``subjects`` restricts the roll-up. Correction is usually applied in batches, and a
    cohort figure that mixes processed subjects with ones still awaiting correction
    describes neither. Pass the batch to get a number that means something.
    """
    required = {"subject", "run", "n_r_markers", "n_volumes"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Cohort marker QC requires columns {missing}.")
    frame = runs.copy()
    if subjects is not None:
        wanted = {str(subject) for subject in subjects}
        frame = frame[frame["subject"].astype(str).isin(wanted)]
        unknown = sorted(wanted - set(frame["subject"].astype(str)))
        if unknown:
            raise ValueError(f"No runs found for requested subjects: {unknown}")
    if frame.empty:
        raise ValueError("Cohort marker QC requires at least one run.")
    if (frame["n_volumes"] <= 0).any():
        raise ValueError("Cohort marker QC requires a positive volume count per run.")
    frame["markers_per_volume"] = frame["n_r_markers"] / frame["n_volumes"]
    frame["is_broken"] = frame["markers_per_volume"] < minimum_markers_per_volume
    frame = frame.sort_values(["subject", "run"]).reset_index(drop=True)
    return CohortMarkerQc(runs=frame)


def cohort_marker_html(qc: CohortMarkerQc) -> str:
    """Render the measured R-marker availability across the cohort."""
    total = len(qc.runs)
    broken = len(qc.broken)
    subjects = qc.runs["subject"].nunique()
    clean_subjects = qc.subjects_fully_usable()
    per_subject = (
        qc.runs.groupby("subject")["is_broken"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "runs_without_markers", "count": "runs"})
    )
    affected = [
        (f"sub-{subject}", f"{int(values['runs_without_markers'])} of {int(values['runs'])}")
        for subject, values in per_subject.iterrows()
        if values["runs_without_markers"] > 0
    ]
    document = (
        "<p>Analyzer builds its pulse-artifact template from the R markers it writes. "
        "This counts the markers present in the exported files, which is independent of "
        "whether Analyzer raised a message during processing. Runs are grouped against "
        "the configured reference ratio "
        "(<code>report.thresholds.min_r_markers_per_volume</code>); what a shortfall "
        "means for a given run is left to the reviewer.</p>"
        + metric_table(
            [
                ("Runs analysed", total),
                ("Subjects", subjects),
                Metric(
                    "Runs below the reference ratio",
                    f"{broken} ({qc.broken_fraction:.0%})",
                    emphasis=True,
                ),
                ("Subjects with every run above it", f"{len(clean_subjects)} of {subjects}"),
            ]
        )
    )
    if affected:
        document += (
            "<p><strong>Runs below the reference ratio, per subject.</strong></p>"
            f"{metric_table(affected)}"
        )
    return document


def plot_cohort_markers(qc: CohortMarkerQc) -> plt.Figure:
    """Plot R-marker availability for every run, subject by subject."""
    runs = qc.runs
    subjects = sorted(runs["subject"].astype(str).unique())
    run_labels = sorted(runs["run"].unique())
    grid = np.full((len(subjects), len(run_labels)), np.nan)
    for _, row in runs.iterrows():
        grid[subjects.index(str(row["subject"])), run_labels.index(row["run"])] = row[
            "markers_per_volume"
        ]

    figure, (matrix_axis, hist_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, max(3.6, 0.34 * len(subjects))),
        width_ratios=(2, 1),
        layout="constrained",
    )
    # A run either has a marker set or it does not, so the map is binary rather than a
    # continuous scale that would invite reading precision into the ratio.
    binary = np.where(np.isnan(grid), np.nan, grid >= MINIMUM_R_MARKERS_PER_VOLUME)
    matrix_axis.imshow(
        binary,
        cmap=plt.matplotlib.colors.ListedColormap([FLAG_COLOR, AFTER_COLOR]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    matrix_axis.set(
        xticks=range(len(run_labels)),
        xticklabels=[f"run-{value}" for value in run_labels],
        yticks=range(len(subjects)),
        yticklabels=[f"sub-{value}" for value in subjects],
        title=(
            f"R markers reaching the pipeline · {len(qc.broken)} of {len(runs)} runs "
            f"({qc.broken_fraction:.0%}) below the reference ratio"
        ),
    )
    matrix_axis.tick_params(labelsize=7)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)

    hist_axis.hist(
        runs["markers_per_volume"],
        bins=24,
        color="0.75",
    )
    hist_axis.axvline(
        MINIMUM_R_MARKERS_PER_VOLUME,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"reference ratio ({MINIMUM_R_MARKERS_PER_VOLUME:g})",
    )
    hist_axis.set(
        title="R markers per scanner volume",
        xlabel="markers / volume",
        ylabel="runs",
    )
    hist_axis.legend(frameon=False, fontsize=8)
    hist_axis.grid(axis="y", alpha=0.2)
    hist_axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


def write_cohort_marker_report(
    qc: CohortMarkerQc,
    *,
    output_path: Path,
    title: str = "Analyzer R-marker availability",
) -> Path:
    """Write a standalone cohort QC report and its table beside it."""
    import mne

    apply_report_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc.runs.to_csv(output_path.with_suffix(".tsv"), sep="\t", index=False)

    report = mne.Report(title=title, verbose="ERROR")
    report.add_html(
        html=cohort_marker_html(qc),
        title="R-marker counts",
        section=title,
        tags=("cohort", "analyzer-correction"),
        replace=True,
    )
    report.add_figure(
        fig=plot_cohort_markers(qc),
        title="R-marker availability by subject and run",
        section=title,
        tags=("cohort", "analyzer-correction"),
        image_format="svg",
        replace=True,
    )
    report.save(output_path, overwrite=True, open_browser=False)
    return output_path


__all__ = [
    "CohortMarkerQc",
    "MINIMUM_R_MARKERS_PER_VOLUME",
    "cohort_marker_html",
    "plot_cohort_markers",
    "summarize_cohort_markers",
    "write_cohort_marker_report",
]
