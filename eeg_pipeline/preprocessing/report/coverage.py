"""Channel and ROI coverage evidence for the subject report.

Bad-channel counts alone do not say whether an analysis is still possible. What matters
is where the losses fall: a montage can lose several channels and leave every region of
interest usable, or lose two from a three-channel midline region and quietly make every
ROI contrast that uses it meaningless.

The pipeline already computes this when it harmonizes bad channels across runs. This
module puts it in front of the reviewer instead of leaving it in a cohort table.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    RETAINED_COLOR,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    Metric,
    grid_table,
    metric_table,
)

BAD_CHANNEL_UNION_QC_STEM = "bad_channel_union_qc"

#: Channels an ROI needs to support an averaged measure at all. One channel is a single
#: sensor masquerading as a region.
MINIMUM_ROI_CHANNELS = 2


@dataclass(frozen=True)
class ChannelCoverage:
    """Bad-channel load and its consequences for each region of interest."""

    subject: str
    n_channels: int
    bad_channels: tuple[str, ...]
    sync_policy: str
    n_runs: int
    #: One row per ROI with channel counts before and after bad-channel removal.
    rois: pd.DataFrame

    @property
    def bad_fraction(self) -> float:
        return len(self.bad_channels) / self.n_channels if self.n_channels else 0.0

    @property
    def failed_rois(self) -> tuple[str, ...]:
        """Regions failing the default minimum recorded in the QC table."""
        return self.failed_rois_below(MINIMUM_ROI_CHANNELS)

    def failed_rois_below(self, minimum_channels: int) -> tuple[str, ...]:
        """Regions left with fewer than ``minimum_channels`` usable sensors.

        The QC table stores a pass flag against its own fixed minimum, so the count is
        re-tested here to honour a site-configured threshold.
        """
        if self.rois.empty:
            return ()
        failed = self.rois.loc[self.rois["n_remaining"] < minimum_channels, "roi"]
        return tuple(str(value) for value in failed)


def load_channel_coverage(
    *,
    deriv_eeg_root: Path,
    task: str,
    subject: str,
) -> ChannelCoverage | None:
    """Read the bad-channel union QC row for one subject."""
    path = deriv_eeg_root / f"{BAD_CHANNEL_UNION_QC_STEM}_task-{task}.tsv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, sep="\t", dtype={"subject": str})
    frame = frame[frame["subject"].astype(str).str.lstrip("0") == subject.lstrip("0")]
    if frame.empty:
        return None
    row = frame.iloc[0]

    raw_rois = row.get("roi_coverage")
    rois = pd.DataFrame(json.loads(raw_rois)) if isinstance(raw_rois, str) else pd.DataFrame()
    if not rois.empty:
        rois = rois.rename(
            columns={
                "n_channels": "n_total",
                "n_remaining_channels": "n_remaining",
                "passes_min_two_channels": "passes_minimum",
            }
        ).sort_values("roi")
    bad = row.get("union_bad_channels")
    channels = (
        tuple(name.strip() for name in str(bad).split(",") if name.strip())
        if isinstance(bad, str) and bad.strip()
        else ()
    )
    return ChannelCoverage(
        subject=subject,
        n_channels=int(row["n_channels"]),
        bad_channels=channels,
        sync_policy=str(row.get("bad_channel_sync_policy", "")),
        n_runs=int(row.get("n_runs", 0)),
        rois=rois.reset_index(drop=True),
    )


@dataclass(frozen=True)
class RunBadChannels:
    """The channels one run lost, and why the detector said so."""

    run_label: str
    bad_channels: tuple[str, ...]
    #: Detector reason per channel, keyed by channel name. Empty where none was recorded.
    reasons: dict[str, str]


def load_run_bad_channels(
    *,
    deriv_eeg_root: Path,
    task: str,
    subject: str,
) -> list[RunBadChannels]:
    """Read the per-run bad-channel record written beside each processed run.

    The union table says which channels the subject lost; it cannot say whether a channel
    failed once or throughout, and those call for different responses. The per-run
    ``_bads.tsv`` files carry that, so they are read here rather than inferred.

    An absent record yields an empty list rather than an error: a dataset processed by a
    configuration that writes no per-run record simply has no such table to show.
    """
    root = deriv_eeg_root / f"sub-{subject}"
    if not root.is_dir():
        return []
    paths = sorted(
        path
        for path in root.rglob(f"sub-{subject}_task-{task}*_bads.tsv")
        if path.is_file() and not path.name.startswith("._")
    )
    runs: list[RunBadChannels] = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        names = (
            sorted(str(name) for name in frame["name"].dropna()) if "name" in frame.columns else []
        )
        reasons = (
            {str(row["name"]): str(row.get("reason", "")) for _, row in frame.iterrows()}
            if "name" in frame.columns
            else {}
        )
        runs.append(
            RunBadChannels(
                run_label=_run_label_from_bads_path(path, subject=subject, task=task),
                bad_channels=tuple(names),
                reasons=reasons,
            )
        )
    return runs


def _run_label_from_bads_path(path: Path, *, subject: str, task: str) -> str:
    """Return the run-identifying label for one ``_bads.tsv`` file.

    A single-run dataset has no ``run-`` entity at all, so the label falls back to the
    session or to "run" rather than leaving the row unlabelled.
    """
    stem = path.name.removeprefix(f"sub-{subject}_").removesuffix("_bads.tsv")
    for part in stem.split("_"):
        if part.startswith("run-"):
            return part
    for part in stem.split("_"):
        if part.startswith("ses-"):
            return part
    return "run"


def run_matrix_is_informative(runs: Sequence[RunBadChannels]) -> bool:
    """Whether the runs disagree about which channels are bad.

    When every run names the same channels, the matrix is a block of identical columns
    and the table above it already states the block's contents. The matrix earns its
    space only when it can separate "bad throughout" from "bad during one run".
    """
    if len(runs) < 2:
        return False
    return len({run.bad_channels for run in runs}) > 1


def run_bad_channel_html(runs: Sequence[RunBadChannels]) -> str:
    """Render one row per run, clean runs included."""
    if not runs:
        return ""
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Bad channels"),
        Column("Names", align=Align.TEXT),
    )
    rows = [
        [run.run_label, len(run.bad_channels), ", ".join(run.bad_channels) or None]
        for run in runs
    ]
    return (
        "<p>Bad channels recorded for each run. A channel bad in every run is a property "
        "of the montage or the preparation; one bad in a single run is something that "
        "happened during the session, and only the second is worth reviewing the run "
        "for.</p>"
        f"{grid_table(columns, rows)}"
    )


def plot_run_bad_channel_matrix(runs: Sequence[RunBadChannels]) -> plt.Figure:
    """Plot which channels were bad in which run.

    Channels on the vertical axis and runs on the horizontal, so a channel bad throughout
    reads as a full row and a run that went bad reads as a full column. Only channels
    that failed somewhere are drawn: a montage of 63 rows, 61 of them empty, hides the
    two that carry the information.
    """
    if not runs:
        raise ValueError("The bad-channel matrix requires at least one run.")
    channels = sorted({name for run in runs for name in run.bad_channels})
    if not channels:
        raise ValueError("The bad-channel matrix requires at least one bad channel.")

    grid = np.array(
        [[name in run.bad_channels for run in runs] for name in channels],
        dtype=float,
    )
    figure, axis = plt.subplots(
        figsize=(max(4.0, 0.7 * len(runs) + 2.0), max(2.4, 0.32 * len(channels) + 1.2)),
        layout="constrained",
    )
    # Two states, so a two-colour listed map rather than a continuous ramp: an intensity
    # scale would invite reading a severity into a boolean.
    axis.imshow(
        grid,
        aspect="auto",
        cmap=ListedColormap([RETAINED_COLOR, FLAG_COLOR]),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set(
        xticks=np.arange(len(runs)),
        xticklabels=[run.run_label for run in runs],
        yticks=np.arange(len(channels)),
        yticklabels=channels,
        title="Bad channels by run",
    )
    axis.set_xticks(np.arange(len(runs) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(channels) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    axis.tick_params(axis="y", labelsize=8)
    handles = [
        Patch(facecolor=FLAG_COLOR, label="Bad"),
        Patch(facecolor=RETAINED_COLOR, label="Good"),
    ]
    axis.legend(
        handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    plt.close(figure)
    return figure


def _unassigned_rows(coverage: ChannelCoverage) -> list[tuple[str, object]]:
    """Account for channels that belong to no region of interest.

    The regions are analysis groupings, not a partition of the montage, so they routinely
    sum to fewer channels than the recording has. Left unstated that reads as the table
    contradicting itself; stated, it is just the montage.
    """
    if coverage.rois.empty:
        return []
    assigned = int(coverage.rois["n_total"].sum())
    unassigned = int(coverage.n_channels) - assigned
    if unassigned <= 0:
        return []
    return [("In no region of interest", unassigned)]


def coverage_figure_is_informative(rois: pd.DataFrame) -> bool:
    """Whether the per-region figure shows anything the summary table does not.

    The figure draws the region total behind the surviving count. When nothing was
    excluded the two layers are identical, so the front bars hide the back ones exactly
    and the legend names a series that is not visible anywhere on the axes. There is no
    loss to plot, and the sentence in the summary says so more directly.
    """
    if rois.empty:
        return False
    return bool((rois["n_remaining"].to_numpy() < rois["n_total"].to_numpy()).any())


def coverage_html(
    coverage: ChannelCoverage,
    *,
    minimum_roi_channels: int = MINIMUM_ROI_CHANNELS,
) -> str:
    """Render the bad-channel load and per-ROI consequences."""
    listed = ", ".join(coverage.bad_channels) if coverage.bad_channels else "none"
    document = (
        "<p>Bad channels are excluded rather than interpolated, so what matters for "
        "analysis is not how many were lost but whether each region of interest still "
        "has enough sensors to average over.</p>"
        + metric_table(
            [
                ("EEG channels", coverage.n_channels),
                ("Runs harmonized", f"{coverage.n_runs} ({coverage.sync_policy})"),
                Metric(
                    "Bad channels",
                    f"{len(coverage.bad_channels)} ({coverage.bad_fraction:.1%})",
                    emphasis=True,
                ),
                ("Excluded", listed),
                *_unassigned_rows(coverage),
            ]
        )
    )
    if not coverage_figure_is_informative(coverage.rois):
        document += (
            "<p>No channel was excluded, so every region keeps its full complement and "
            "there is no per-region loss to plot.</p>"
        )
    below = coverage.failed_rois_below(minimum_roi_channels)
    if below:
        document += (
            f"<p>Region(s) left with fewer than {minimum_roi_channels} channels: "
            f"{html.escape(', '.join(below))}. The per-region counts are plotted below "
            "against the configured reference "
            "(<code>report.thresholds.min_roi_channels</code>).</p>"
        )
    return document


def plot_coverage(
    coverage: ChannelCoverage,
    *,
    minimum_roi_channels: int = MINIMUM_ROI_CHANNELS,
) -> plt.Figure:
    """Plot remaining channels per region against the minimum needed."""
    rois = coverage.rois
    if rois.empty:
        raise ValueError("ROI coverage figure requires per-region channel counts.")
    positions = np.arange(len(rois))
    remaining = rois["n_remaining"].to_numpy()
    total = rois["n_total"].to_numpy()
    passes = (rois["n_remaining"] >= minimum_roi_channels).to_numpy(dtype=bool)

    figure, axis = plt.subplots(
        figsize=(max(7.0, 0.85 * len(rois)), 4.0),
        layout="constrained",
    )
    axis.bar(positions, total, color="0.90", label="Channels in region")
    axis.bar(
        positions,
        remaining,
        color=[AFTER_COLOR if flag else FLAG_COLOR for flag in passes],
        label="Remaining after bad-channel removal",
    )
    axis.axhline(
        minimum_roi_channels,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"minimum for a regional average ({minimum_roi_channels})",
    )
    axis.set(
        title=(f"Region coverage after excluding {len(coverage.bad_channels)} bad channel(s)"),
        ylabel="EEG channels",
        xticks=positions,
        xticklabels=[str(name) for name in rois["roi"]],
    )
    axis.tick_params(axis="x", labelrotation=35, labelsize=8)
    for label, flag in zip(axis.get_xticklabels(), passes, strict=True):
        if not flag:
            label.set_color(FLAG_COLOR)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


def add_coverage_review(
    *,
    report: mne.Report,
    deriv_eeg_root: Path,
    task: str,
    subject: str,
    settings: "ReportSettings | None" = None,
    section: str = "Channel and region coverage",
) -> ChannelCoverage | None:
    """Append channel and ROI coverage evidence to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        drop_replaced_per_run_bad_channels,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    coverage = load_channel_coverage(
        deriv_eeg_root=deriv_eeg_root,
        task=task,
        subject=subject,
    )
    if coverage is None:
        return None
    minimum = (settings or ReportSettings()).min_roi_channels

    remove_tagged_content(report, tag="channel-coverage")
    report.add_html(
        html=coverage_html(coverage, minimum_roi_channels=minimum),
        title="Bad channels and region coverage",
        section=section,
        tags=("data-quality", "channel-coverage"),
        replace=True,
    )
    runs = load_run_bad_channels(deriv_eeg_root=deriv_eeg_root, task=task, subject=subject)
    if runs:
        # Only once the replacement exists, so that a report built without this section
        # keeps MNE-BIDS-Pipeline's per-run items rather than losing both.
        drop_replaced_per_run_bad_channels(report)
        report.add_html(
            html=run_bad_channel_html(runs),
            title="Bad channels by run",
            section=section,
            tags=("data-quality", "channel-coverage"),
            replace=True,
        )
        if run_matrix_is_informative(runs):
            report.add_figure(
                fig=plot_run_bad_channel_matrix(runs),
                title="Which channels failed in which run",
                section=section,
                tags=("data-quality", "channel-coverage"),
                image_format=report_image_format(),
                replace=True,
            )
    if coverage_figure_is_informative(coverage.rois):
        report.add_figure(
            fig=plot_coverage(coverage, minimum_roi_channels=minimum),
            title="Channels remaining per region",
            section=section,
            tags=("data-quality", "channel-coverage"),
            image_format=report_image_format(),
            replace=True,
        )
    move_tagged_content_before(report, tag="channel-coverage", anchor=before_raw_sections)
    return coverage


__all__ = [
    "BAD_CHANNEL_UNION_QC_STEM",
    "MINIMUM_ROI_CHANNELS",
    "ChannelCoverage",
    "RunBadChannels",
    "add_coverage_review",
    "coverage_figure_is_informative",
    "coverage_html",
    "load_channel_coverage",
    "load_run_bad_channels",
    "plot_coverage",
    "plot_run_bad_channel_matrix",
    "run_bad_channel_html",
    "run_matrix_is_informative",
]
