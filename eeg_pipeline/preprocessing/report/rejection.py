"""Trial-rejection evidence for the subject preprocessing report.

Epoch rejection is applied automatically and then never shown. That hides two things a
reviewer needs: whether enough trials survived to analyse the subject at all, and
whether the loss is spread evenly. Rejection concentrated in one run points at a
recording problem; rejection concentrated in one condition biases every contrast that
uses it, because the surviving trials of that condition are no longer a random sample.
"""

from __future__ import annotations

import html
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.style import (
    EXCLUDED_COLOR,
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

#: Grouping columns to report retention over, when the events table provides them.
#:
#: Run identity always matters, because rejection concentrated in one run points at a
#: recording problem. The experimental factors are resolved from the configured event
#: column candidates, since trial loss that falls unevenly across conditions biases
#: every contrast built from them.
DEFAULT_GROUPING_COLUMNS = ("run_id", "trial_type")

_LOGGER = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class GroupRetention:
    """Retained counts and, when observed, their presented denominators."""

    retained: pd.Series
    presented: pd.Series | None = None

    @property
    def n_groups(self) -> int:
        reference = self.presented if self.presented is not None else self.retained
        return int(reference.size)

    @property
    def rates(self) -> pd.Series | None:
        if self.presented is None:
            return None
        return self.retained / self.presented


def autoreject_log_html(log) -> str:
    """Summarize channel-level interpolation and unresolved bad verdicts."""
    labels = np.asarray(log.labels, dtype=int)
    bad_epochs = np.asarray(log.bad_epochs, dtype=bool)
    if labels.shape != (bad_epochs.size, len(log.ch_names)):
        raise ValueError(
            "AutoReject labels must be epochs × channels and match bad_epochs/ch_names."
        )
    interpolated = (labels == 2).sum(axis=0)
    unresolved = (labels == 1).sum(axis=0)
    rows = [
        [channel, int(interpolated[index]), int(unresolved[index])]
        for index, channel in sorted(
            enumerate(log.ch_names),
            key=lambda item: (-interpolated[item[0]], -unresolved[item[0]], item[1]),
        )
        if interpolated[index] or unresolved[index]
    ]
    details = (
        grid_table(
            (
                Column("Channel", align=Align.TEXT),
                Column("Interpolated epochs"),
                Column("Bad, not interpolated"),
            ),
            rows,
        )
        if rows
        else "<p>No channel-level repairs or unresolved bad channel-epochs were recorded.</p>"
    )
    return (
        "<p>AutoReject's channel-by-epoch verdict distinguishes recorded samples from "
        "whole-epoch spline estimates. Recurrent interpolation can alter spatial and "
        "connectivity measures even when the epoch itself was retained. Codes in the "
        "matrix are: good, bad but not interpolated, and bad then interpolated.</p>"
        f"<p>Selected <code>n_interpolate={int(log.n_interpolate)}</code>; "
        f"<code>consensus={float(log.consensus):.2f}</code>; "
        f"{int(bad_epochs.sum())} of {bad_epochs.size} epochs dropped.</p>" + details
    )


def plot_autoreject_log(log) -> plt.Figure:
    """Plot AutoReject's complete channel × pre-rejection epoch verdict matrix."""
    labels = np.asarray(log.labels, dtype=int)
    bad_epochs = np.asarray(log.bad_epochs, dtype=bool)
    if labels.shape != (bad_epochs.size, len(log.ch_names)):
        raise ValueError(
            "AutoReject labels must be epochs × channels and match bad_epochs/ch_names."
        )
    figure, axis = plt.subplots(figsize=(10.0, 5.5), layout="constrained")
    colors = ListedColormap([RETAINED_COLOR, EXCLUDED_COLOR, "#D99B2B"])
    axis.imshow(
        labels.T,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=colors,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], colors.N),
    )
    dropped = np.flatnonzero(bad_epochs)
    if dropped.size:
        axis.scatter(
            dropped,
            np.full(dropped.size, -0.65),
            marker="v",
            color=EXCLUDED_COLOR,
            s=18,
            clip_on=False,
            label="Dropped epoch",
        )
    tick_step = max(1, int(np.ceil(len(log.ch_names) / 24)))
    ticks = np.arange(0, len(log.ch_names), tick_step)
    axis.set(
        title="AutoReject repair matrix · all pre-rejection epochs",
        xlabel="Epoch position before cleaning",
        ylabel="EEG channel",
        yticks=ticks,
        yticklabels=[log.ch_names[index] for index in ticks],
    )
    axis.legend(
        handles=[
            Patch(facecolor=RETAINED_COLOR, label="Good"),
            Patch(facecolor=EXCLUDED_COLOR, label="Bad, not interpolated"),
            Patch(facecolor="#D99B2B", label="Bad, interpolated"),
            Line2D(
                [],
                [],
                marker="v",
                linestyle="none",
                color=EXCLUDED_COLOR,
                label="Dropped epoch",
            ),
        ],
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper right",
    )
    plt.close(figure)
    return figure


def summarize_rejection(drop_log: Sequence[Sequence[str]]) -> RejectionSummary:
    """Summarize an MNE drop log into counts, reasons, and dropped positions.

    MNE records one entry per input event, including ``IGNORED`` entries for event types
    that were never selected into the epoch set. Those are not rejected trials and are
    removed before the denominator and positions are calculated. Among considered events,
    an empty entry means the epoch was kept and any reason means it was dropped.
    """
    if drop_log is None:
        raise ValueError("A drop log is required to summarize epoch rejection.")
    entries = [tuple(entry) for entry in drop_log if "IGNORED" not in entry]
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
    presented_events: pd.DataFrame | None = None,
    columns: Sequence[str] = DEFAULT_GROUPING_COLUMNS,
) -> dict[str, GroupRetention]:
    """Measure retained trials and per-group rates where denominators are available.

    Without ``presented_events``, only the surviving composition is defined. With it,
    each retained count is paired with the number originally presented, so differential
    rejection can be reported as a rate rather than inferred from unequal counts.
    """
    available = [column for column in columns if column in events.columns]
    if total < len(events):
        raise ValueError("Clean events cannot contain more trials than the pre-cleaning set.")
    if presented_events is not None and len(presented_events) != total:
        raise ValueError(
            f"Presented events contain {len(presented_events)} rows but rejection considered "
            f"{total} epochs."
        )

    retention: dict[str, GroupRetention] = {}
    for column in available:
        retained = events[column].value_counts().sort_index()
        if presented_events is None or column not in presented_events.columns:
            retention[column] = GroupRetention(retained=retained)
            continue
        presented = presented_events[column].value_counts().sort_index()
        unexpected = retained.index.difference(presented.index)
        if not unexpected.empty:
            raise ValueError(
                f"Retained {column} levels are absent from the presented events: "
                f"{unexpected.tolist()}."
            )
        retained = retained.reindex(presented.index, fill_value=0).astype(int)
        if (retained > presented).any():
            raise ValueError(f"Retained {column} counts exceed the presented counts.")
        retention[column] = GroupRetention(retained=retained, presented=presented)
    return retention


def run_of_position(
    summary: RejectionSummary,
    events: pd.DataFrame,
    *,
    column: str = "run_id",
) -> np.ndarray | None:
    """Map each pre-cleaning epoch position to its run from presented events.

    The full event table is required. Reconstructing run labels from retained epochs
    leaves drops unlabelled, so filling those gaps can move a run boundary when the last
    epoch of one run or the first epoch of the next was rejected.
    """
    if column not in events.columns:
        return None
    if len(events) != summary.total:
        raise ValueError(
            f"Presented events contain {len(events)} rows but rejection considered "
            f"{summary.total} epochs."
        )
    assignment = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(assignment).all():
        raise ValueError(f"Presented-event column {column!r} must contain finite run labels.")
    return assignment


def rejection_summary_html(
    summary: RejectionSummary,
    *,
    group_retention: Mapping[str, GroupRetention] | None = None,
) -> str:
    """Render trial retention and its reasons."""
    rows: list[Metric | tuple[str, object]] = [
        ("Epochs before cleaning", f"{summary.total}"),
        ("Epochs retained", f"{summary.kept}"),
        Metric(
            "Epochs dropped",
            f"{summary.dropped} ({summary.dropped_fraction:.1%})",
            emphasis=True,
        ),
    ]
    for reason, count in sorted(summary.reasons.items(), key=lambda item: -item[1]):
        rows.append(Metric(f"dropped by {reason}", f"{count}", indent=True))
    document = (
        "<p>Trial counts entering and leaving epoch cleaning. Rejection is applied "
        "automatically; this is the record of what it removed. Drop reasons may overlap "
        "because MNE can attach more than one reason to the same epoch, so their counts "
        "must not be summed as a second dropped total.</p>"
        f"{metric_table(rows)}"
    )
    for column, retention in (group_retention or {}).items():
        if retention.n_groups < 2:
            continue
        if retention.presented is None:
            document += (
                f"<p><strong>Retained trials by {html.escape(column)}.</strong> An uneven "
                "retained distribution cannot distinguish the original design from "
                "differential rejection without the corresponding presented count for "
                "each group. It is shown as composition evidence, not as a rejection "
                "rate.</p>"
                f"{metric_table((str(name), int(value)) for name, value in retention.retained.items())}"
            )
            continue
        rates = retention.rates
        document += (
            f"<p><strong>Retention rate by {html.escape(column)}.</strong> Each denominator "
            "is the number of trials in that group before epoch rejection, so differences "
            "between rows measure differential loss rather than the original design.</p>"
            + metric_table(
                (
                    str(name),
                    f"{int(retention.retained[name])} / {int(presented)} "
                    f"({float(rates[name]):.1%})",
                )
                for name, presented in retention.presented.items()
            )
        )
    return document


def plot_rejection(
    summary: RejectionSummary,
    *,
    group_retention: Mapping[str, GroupRetention] | None = None,
    run_assignment: np.ndarray | None = None,
) -> plt.Figure:
    """Plot where the dropped epochs sit, and the surviving group distribution."""
    group_retention = {
        column: retention
        for column, retention in (group_retention or {}).items()
        if retention.n_groups >= 2
    }
    # A retention breakdown of a session that dropped nothing is a row of bars at 1.0 in
    # every panel. It answers the question these panels exist for -- did rejection fall
    # unevenly across runs or conditions -- with the same "no" the header already gave,
    # and spends two panels of the figure doing it. The position strip stays, because
    # "which epochs went" is still worth showing as an empty strip: it is the difference
    # between nothing dropped and nothing measured.
    if not summary.dropped_positions:
        group_retention = {}
    panels = 1 + len(group_retention)
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
    # Each epoch carries one bit — kept or dropped — so it gets a strip rather than a
    # full-height bar. Drawn floor to ceiling in a saturated colour, sixty-odd epochs
    # became a wall of ink that dominated the figure and made the handful of dropped
    # ones no easier to find than a thin band does.
    strip_bottom, strip_height = 0.44, 0.12
    position_axis.bar(
        np.arange(summary.total),
        strip_height,
        bottom=strip_bottom,
        width=1.0,
        color=[RETAINED_COLOR if flag else EXCLUDED_COLOR for flag in kept],
    )
    if run_assignment is not None:
        run_assignment = np.asarray(run_assignment, dtype=float)
        if run_assignment.shape != (summary.total,):
            raise ValueError("Run assignment must contain one label per pre-cleaning epoch.")
        if not np.isfinite(run_assignment).all():
            raise ValueError("Run assignment must contain only finite labels.")
        # Run boundaries turn "a block of epochs was dropped" into "run N lost them".
        boundaries = np.flatnonzero(np.diff(run_assignment))
        for boundary in boundaries:
            position_axis.axvline(
                boundary + 0.5,
                ymin=strip_bottom,
                ymax=strip_bottom + strip_height,
                color="black",
                linewidth=0.8,
            )
        for value in np.unique(run_assignment):
            positions = np.flatnonzero(run_assignment == value)
            position_axis.annotate(
                f"run-{int(value)}",
                xy=(positions.mean(), strip_bottom + strip_height + 0.03),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="black",
            )
    position_axis.legend(
        handles=[
            Patch(facecolor=RETAINED_COLOR, label="Kept"),
            Patch(facecolor=EXCLUDED_COLOR, label="Dropped"),
        ],
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
    )
    position_axis.set(
        title=f"{summary.dropped} of {summary.total} epochs dropped "
        f"({summary.dropped_fraction:.1%})",
        xlabel="Epoch position before cleaning",
        yticks=[],
        ylim=(0, 1),
    )
    position_axis.spines[["top", "right", "left"]].set_visible(False)

    for axis, (column, retention) in zip(axes[0][1:], group_retention.items(), strict=True):
        rates = retention.rates
        values = retention.retained if rates is None else rates
        positions = np.arange(values.size)
        axis.bar(positions, values.to_numpy(), color="0.80")
        baseline = float(values.mean()) if rates is None else summary.kept / summary.total
        axis.axhline(
            baseline,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=(f"mean {baseline:.1f}" if rates is None else f"overall {baseline:.1%}"),
        )
        axis.set(
            title=(
                f"Retained trials by {column}" if rates is None else f"Retention rate by {column}"
            ),
            xlabel=column,
            ylabel="Trials retained" if rates is None else "Retained / presented",
            xticks=positions,
            xticklabels=[str(name) for name in values.index],
            **({"ylim": (0.0, 1.0)} if rates is not None else {}),
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
    presented_events: pd.DataFrame | None = None,
    autoreject_log=None,
    config: object | None = None,
    section: str = "Epoch rejection",
) -> RejectionSummary:
    """Append trial-retention evidence to the report and return the summary."""
    from eeg_pipeline.preprocessing.report.organize import (
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    summary = summarize_rejection(clean_epochs.drop_log)
    retained_events = clean_events
    if presented_events is not None:
        if len(presented_events) != summary.total:
            raise ValueError(
                f"Presented events contain {len(presented_events)} rows but rejection "
                f"considered {summary.total} epochs."
            )
        kept = np.ones(summary.total, dtype=bool)
        kept[list(summary.dropped_positions)] = False
        retained_events = presented_events.loc[kept].reset_index(drop=True)
        if len(retained_events) != summary.kept:
            raise ValueError("Presented-event alignment does not reproduce the retained count.")

    grouping_events = presented_events if presented_events is not None else retained_events
    group_retention = (
        retention_by_group(
            retained_events,
            total=summary.total,
            presented_events=presented_events,
            columns=resolve_grouping_columns(grouping_events, config=config),
        )
        if retained_events is not None
        else None
    )
    run_assignment = (
        run_of_position(summary, presented_events) if presented_events is not None else None
    )
    remove_tagged_content(report, tag="epoch-rejection")
    report.add_html(
        html=rejection_summary_html(summary, group_retention=group_retention),
        title="Trial retention",
        section=section,
        tags=("epochs", "epoch-rejection"),
        replace=True,
    )
    report.add_figure(
        fig=plot_rejection(
            summary,
            group_retention=group_retention,
            run_assignment=run_assignment,
        ),
        title="Dropped epochs and retained trial distribution",
        section=section,
        tags=("epochs", "epoch-rejection"),
        image_format=report_image_format(),
        replace=True,
    )
    if autoreject_log is not None:
        if len(autoreject_log.bad_epochs) != summary.total:
            raise ValueError("AutoReject log epoch count does not match the rejection denominator.")
        logged_drops = tuple(np.flatnonzero(autoreject_log.bad_epochs).tolist())
        if logged_drops != summary.dropped_positions:
            raise ValueError(
                "AutoReject log dropped epochs do not match the clean epochs drop log."
            )
        report.add_html(
            html=autoreject_log_html(autoreject_log),
            title="AutoReject interpolation burden",
            section=section,
            tags=("epochs", "epoch-rejection"),
            replace=True,
        )
        report.add_figure(
            fig=plot_autoreject_log(autoreject_log),
            title="AutoReject channel-by-epoch repair matrix",
            section=section,
            tags=("epochs", "epoch-rejection"),
            image_format=report_image_format(has_dense_image=True),
            replace=True,
        )
    # The counts above say how many trials survived; this says what they are like. The
    # report measures time-resolved quality on the raw runs and, without this, nothing at
    # all on the epochs it delivers -- so a trial set could pass every count in this
    # section while carrying a channel that is noisy in a third of its trials.
    from eeg_pipeline.preprocessing.report.continuity import plot_epoch_channel_amplitude

    try:
        amplitude_figure = plot_epoch_channel_amplitude(clean_epochs)
    except ValueError as exc:
        # No EEG channels, or none with usable positions. The section keeps its counts
        # rather than losing them to a panel that could not be drawn.
        amplitude_figure = None
        _LOGGER.info("No epoch amplitude panel: %s", exc)
    if amplitude_figure is not None:
        report.add_figure(
            fig=amplitude_figure,
            title="Amplitude by epoch and channel (retained epochs)",
            section=section,
            tags=("epochs", "epoch-rejection"),
            image_format=report_image_format(has_dense_image=True),
            replace=True,
        )
    return summary


__all__ = [
    "DEFAULT_GROUPING_COLUMNS",
    "GroupRetention",
    "RejectionSummary",
    "add_rejection_review",
    "autoreject_log_html",
    "plot_autoreject_log",
    "plot_rejection",
    "rejection_summary_html",
    "resolve_grouping_columns",
    "retention_by_group",
    "run_of_position",
    "summarize_rejection",
]
