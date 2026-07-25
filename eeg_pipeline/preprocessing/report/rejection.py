"""Trial-rejection evidence for the subject preprocessing report.

Epoch rejection is applied automatically and then never shown. That hides two things a
reviewer needs: whether enough trials survived to analyse the subject at all, and
whether the loss is spread evenly. Rejection concentrated in one run points at a
recording problem; rejection concentrated in one condition biases every contrast that
uses it, because the surviving trials of that condition are no longer a random sample.
"""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.style import AFTER_COLOR, FLAG_COLOR, GUIDE_COLOR

#: Grouping columns to report retention over, when the events table provides them.
#:
#: Run identity always matters, because rejection concentrated in one run points at a
#: recording problem. The experimental factors are resolved from the configured event
#: column candidates, since trial loss that falls unevenly across conditions biases
#: every contrast built from them.
DEFAULT_GROUPING_COLUMNS = ("run_id", "trial_type")

#: Maximum distinct values before a column is treated as continuous rather than a factor.
MAXIMUM_GROUP_LEVELS = 12


@dataclass(frozen=True)
class RejectionSummary:
    """How many epochs survived cleaning, and why the rest did not."""

    total: int
    kept: int
    #: Count of dropped epochs per reason string recorded by MNE or autoreject.
    reasons: Mapping[str, int]
    #: Position of every dropped epoch within the pre-cleaning set.
    dropped_positions: tuple[int, ...]

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    @property
    def dropped_fraction(self) -> float:
        return self.dropped / self.total if self.total else 0.0


def summarize_rejection(drop_log: Sequence[Sequence[str]]) -> RejectionSummary:
    """Summarize an MNE drop log into counts, reasons, and dropped positions.

    MNE records one entry per pre-cleaning epoch: empty when the epoch was kept, and
    otherwise the reasons it was dropped.
    """
    if drop_log is None:
        raise ValueError("A drop log is required to summarize epoch rejection.")
    entries = [tuple(entry) for entry in drop_log]
    dropped = [index for index, entry in enumerate(entries) if entry]
    reasons = Counter(reason for entry in entries if entry for reason in entry)
    return RejectionSummary(
        total=len(entries),
        kept=len(entries) - len(dropped),
        reasons=dict(reasons),
        dropped_positions=tuple(dropped),
    )


def resolve_grouping_columns(
    events: pd.DataFrame,
    *,
    config: object | None = None,
    max_levels: int | None = None,
) -> tuple[str, ...]:
    """Return the grouping columns worth reporting retention over.

    Continuous columns such as a rating are excluded: one bar per distinct value carries
    no information about whether loss is uneven across the design.
    """
    from eeg_pipeline.preprocessing.report.settings import ReportSettings

    if max_levels is None:
        max_levels = ReportSettings.from_config(config).max_group_levels
    candidates = list(DEFAULT_GROUPING_COLUMNS)
    if config is not None:
        for key in ("event_columns.condition", "event_columns.predictor"):
            configured = config.get(key) or []
            candidates.extend(str(name) for name in configured)
    resolved = []
    for name in candidates:
        if name in events.columns and name not in resolved:
            levels = events[name].nunique(dropna=True)
            if 2 <= levels <= max_levels or name in DEFAULT_GROUPING_COLUMNS:
                resolved.append(name)
    return tuple(resolved)


def retention_by_group(
    events: pd.DataFrame,
    *,
    total: int,
    columns: Sequence[str] = DEFAULT_GROUPING_COLUMNS,
) -> dict[str, pd.Series]:
    """Count retained trials per group for each grouping column that exists.

    Only the retained trials appear in the clean events table, so this reports the
    surviving distribution rather than a per-group rejection rate. An uneven surviving
    distribution is the signal worth acting on.
    """
    available = [column for column in columns if column in events.columns]
    counts = {column: events[column].value_counts().sort_index() for column in available}
    if total < len(events):
        raise ValueError("Clean events cannot contain more trials than the pre-cleaning set.")
    return counts


def run_of_position(
    summary: RejectionSummary,
    events: pd.DataFrame,
    *,
    column: str = "run_id",
) -> np.ndarray | None:
    """Map each pre-cleaning epoch position to its run, using the retained trials.

    Only retained trials appear in the clean events table, but they appear in the same
    order as the surviving epochs, so the run label can be carried back onto the
    pre-cleaning positions. Dropped positions are left unlabelled. Without this the
    reviewer cannot see that a block of dropped epochs all belonged to one run.
    """
    if column not in events.columns:
        return None
    retained = [
        index for index in range(summary.total) if index not in set(summary.dropped_positions)
    ]
    if len(retained) != len(events):
        raise ValueError(
            f"{len(events)} clean events cannot be aligned to {len(retained)} retained epochs."
        )
    assignment = np.full(summary.total, np.nan)
    assignment[retained] = pd.to_numeric(events[column], errors="coerce").to_numpy()
    return assignment


def rejection_summary_html(
    summary: RejectionSummary,
    *,
    group_counts: Mapping[str, pd.Series] | None = None,
) -> str:
    """Render trial retention and its reasons."""
    rows = [
        ("Epochs before cleaning", f"{summary.total}"),
        ("Epochs retained", f"{summary.kept}"),
        (
            "<strong>Epochs dropped</strong>",
            f"<strong>{summary.dropped} ({summary.dropped_fraction:.1%})</strong>",
        ),
    ]
    for reason, count in sorted(summary.reasons.items(), key=lambda item: -item[1]):
        rows.append((f"&nbsp;&nbsp;dropped by {html.escape(reason)}", f"{count}"))
    body = "".join(f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in rows)
    document = (
        "<p>Trial counts entering and leaving epoch cleaning. Rejection is applied "
        "automatically; this is the record of what it removed.</p>"
        f"<table><tbody>{body}</tbody></table>"
    )
    for column, counts in (group_counts or {}).items():
        if counts.size < 2:
            continue
        cells = "".join(
            f"<tr><td>{html.escape(str(name))}</td><td>{int(value)}</td></tr>"
            for name, value in counts.items()
        )
        document += (
            f"<p><strong>Retained trials by {html.escape(column)}.</strong> An uneven "
            "distribution means the surviving trials are no longer a random sample, "
            "which biases any contrast computed across these groups.</p>"
            f"<table><tbody>{cells}</tbody></table>"
        )
    return document


def plot_rejection(
    summary: RejectionSummary,
    *,
    group_counts: Mapping[str, pd.Series] | None = None,
    run_assignment: np.ndarray | None = None,
) -> plt.Figure:
    """Plot where the dropped epochs sit, and the surviving group distribution."""
    group_counts = {
        column: counts for column, counts in (group_counts or {}).items() if counts.size >= 2
    }
    panels = 1 + len(group_counts)
    figure, axes = plt.subplots(
        1,
        panels,
        figsize=(5.4 * panels, 3.4),
        squeeze=False,
        layout="constrained",
    )
    position_axis = axes[0][0]
    kept = np.ones(summary.total, dtype=bool)
    kept[list(summary.dropped_positions)] = False
    position_axis.bar(
        np.arange(summary.total),
        1.0,
        width=1.0,
        color=[AFTER_COLOR if flag else FLAG_COLOR for flag in kept],
    )
    if run_assignment is not None:
        # Run boundaries turn "a block of epochs was dropped" into "run N lost them".
        boundaries = np.flatnonzero(np.diff(pd.Series(run_assignment).ffill().bfill().to_numpy()))
        for boundary in boundaries:
            position_axis.axvline(boundary + 0.5, color="black", linewidth=0.8)
        labelled = pd.Series(run_assignment).ffill().bfill().to_numpy()
        for value in np.unique(labelled):
            positions = np.flatnonzero(labelled == value)
            position_axis.annotate(
                f"run-{int(value)}",
                xy=(positions.mean(), 0.5),
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                bbox={
                    "boxstyle": "square,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                },
            )
    position_axis.set(
        title=f"{summary.dropped} of {summary.total} epochs dropped "
        f"({summary.dropped_fraction:.1%})",
        xlabel="Epoch position before cleaning",
        yticks=[],
        ylim=(0, 1),
    )
    position_axis.spines[["top", "right", "left"]].set_visible(False)

    for axis, (column, counts) in zip(axes[0][1:], group_counts.items(), strict=True):
        positions = np.arange(counts.size)
        axis.bar(positions, counts.to_numpy(), color="0.80")
        axis.axhline(
            float(counts.mean()),
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="mean",
        )
        axis.set(
            title=f"Retained trials by {column}",
            xlabel=column,
            ylabel="Trials retained",
            xticks=positions,
            xticklabels=[str(name) for name in counts.index],
        )
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


def add_rejection_review(
    *,
    report: mne.Report,
    clean_epochs: mne.BaseEpochs,
    clean_events: pd.DataFrame | None = None,
    config: object | None = None,
    section: str = "Epoch rejection",
) -> RejectionSummary:
    """Append trial-retention evidence to the report and return the summary."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_epoch_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    summary = summarize_rejection(clean_epochs.drop_log)
    group_counts = (
        retention_by_group(
            clean_events,
            total=summary.total,
            columns=resolve_grouping_columns(clean_events, config=config),
        )
        if clean_events is not None
        else None
    )
    run_assignment = run_of_position(summary, clean_events) if clean_events is not None else None
    remove_tagged_content(report, tag="epoch-rejection")
    report.add_html(
        html=rejection_summary_html(summary, group_counts=group_counts),
        title="Trial retention",
        section=section,
        tags=("epochs", "epoch-rejection"),
        replace=True,
    )
    report.add_figure(
        fig=plot_rejection(
            summary,
            group_counts=group_counts,
            run_assignment=run_assignment,
        ),
        title="Dropped epochs and retained trial distribution",
        section=section,
        tags=("epochs", "epoch-rejection"),
        image_format=report_image_format(),
        replace=True,
    )
    move_tagged_content_before(report, tag="epoch-rejection", anchor=before_epoch_sections)
    return summary


__all__ = [
    "DEFAULT_GROUPING_COLUMNS",
    "RejectionSummary",
    "add_rejection_review",
    "plot_rejection",
    "rejection_summary_html",
    "resolve_grouping_columns",
    "retention_by_group",
    "run_of_position",
    "summarize_rejection",
]
