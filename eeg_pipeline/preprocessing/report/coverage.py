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
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.preprocessing.report.style import AFTER_COLOR, FLAG_COLOR, GUIDE_COLOR

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
        "<table><tbody>"
        f"<tr><td>EEG channels</td><td>{coverage.n_channels}</td></tr>"
        f"<tr><td>Runs harmonized</td><td>{coverage.n_runs} "
        f"({html.escape(coverage.sync_policy)})</td></tr>"
        f"<tr><td><strong>Bad channels</strong></td>"
        f"<td><strong>{len(coverage.bad_channels)} "
        f"({coverage.bad_fraction:.1%})</strong></td></tr>"
        f"<tr><td>Excluded</td><td>{html.escape(listed)}</td></tr>"
        "</tbody></table>"
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
    if not coverage.rois.empty:
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
    "add_coverage_review",
    "coverage_html",
    "load_channel_coverage",
    "plot_coverage",
]
