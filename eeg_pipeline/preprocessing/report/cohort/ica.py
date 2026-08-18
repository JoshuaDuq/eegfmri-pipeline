"""ICA decomposition quality across the cohort.

Three things a subject report cannot say, because each is a statement about a group.

*Was the decomposition well-posed for everybody.* A decomposition estimates n² mixing
parameters from the samples it was given, and the ratio decides whether the components
mean anything. So does the rank: an ICA fitted with more components than the data's rank
supports is not merely optimistic, it is solving for directions the data does not contain.
That last one is an *algebraic* fact rather than an empirical threshold, so it is the one
thing in this report that is flagged as a violation rather than reported as a number.

*How much was removed, in context.* Variance removed is the headline number of the whole
pipeline and it is meaningless pooled across acquisitions: 86% is unremarkable for a
recording of one kind and alarming for another, so a single median over
both describes neither. It is reported per context, never pooled across them.

*What was removed.* Twelve components taken out for eye movement and twelve taken out for
muscle are different recordings with the same headline. The label composition is what
separates them, and it is the panel that catches a participant whose muscle detector ran
away with the decomposition.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    BandRegime,
    Contribution,
    pool_participants_scalar,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.record import COMPONENT_LABEL_CLASSES
from eeg_pipeline.preprocessing.report.style import (
    OKABE_ITO,
    apply_report_style,
    report_image_format,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table, metric_table

ICA_SECTION = "ICA decomposition quality"
ICA_TITLE = "Decomposition across the cohort"
ICA_TAG = "cohort-ica"

#: Colour per detector class in the composition bars.
#:
#: A qualitative, colour-blind-safe assignment: these are categories, not an ordered scale,
#: so nothing here should read as "more" or "less" of anything.
_LABEL_COLOURS = {
    "eye": OKABE_ITO["sky_blue"],
    "heart": OKABE_ITO["vermillion"],
    "muscle": OKABE_ITO["bluish_green"],
    "line": OKABE_ITO["orange"],
    "channel": OKABE_ITO["reddish_purple"],
    "other": "0.55",
    "unrecorded": "0.8",
}


def decomposition_frame(cohort: Cohort) -> pd.DataFrame:
    """One row per participant, holding what the decomposition recorded about itself."""
    rows: list[dict[str, object]] = []
    for participant in cohort.participants:
        measurements = participant.measurements
        if "n_components" not in measurements:
            continue
        row: dict[str, object] = {
            "subject": participant.subject,
            "n_channels": measurements.get("n_channels"),
            "n_components": measurements.get("n_components"),
            "n_excluded": measurements.get("n_excluded"),
            "retained_dimensions": measurements.get("retained_dimensions"),
            "data_rank": measurements.get("data_rank"),
            "condition_number": measurements.get("condition_number"),
            "variance_removed": measurements.get("variance_removed"),
            "samples_per_squared_component": measurements.get("samples_per_squared_component"),
        }
        for label in COMPONENT_LABEL_CLASSES:
            row[f"n_excluded_{label}"] = measurements.get(f"n_excluded_{label}")
        rows.append(row)
    return pd.DataFrame(rows)


def rank_violations(frame: pd.DataFrame) -> list[str]:
    """Participants whose decomposition asked for more components than the rank supports.

    The one thing this report states as a violation rather than as a number. It is
    algebraic: a rank-r dataset spans r directions, and fitting more than r components
    solves for directions the data does not contain. Unlike every threshold this pipeline
    declines to invent, that is true independently of the recording, the montage and the
    reviewer.
    """
    if frame.empty or "data_rank" not in frame.columns:
        return []
    offenders: list[str] = []
    for _, participant in frame.iterrows():
        rank = participant.get("data_rank")
        components = participant.get("n_components")
        if rank is None or components is None:
            continue
        if float(components) > float(rank):
            offenders.append(str(participant["subject"]))
    return offenders


def variance_pooled(
    frame: pd.DataFrame, *, gates: BandGates = DEFAULT_GATES
) -> object | None:
    # One population now. This used to refuse to pool across the acquisition-context
    # axis, which left the sidecar in schema version 4.
    selected = frame[frame["variance_removed"].notna()]
    if selected.empty:
        return None
    return pool_participants_scalar(
        [
            Contribution(
                subject=str(row["subject"]),
                value=np.asarray([float(row["variance_removed"])]),
                n_runs=1,
            )
            for _, row in selected.iterrows()
        ],
        gates=gates,
    )


def plot_label_composition(frame: pd.DataFrame) -> plt.Figure:
    """Excluded components per participant, broken down by what marked them.

    A stacked bar rather than a grouped one: the total is the number a reader already knows
    from the headline, and the question this panel answers is what it was made of.
    """
    apply_report_style()
    present = [
        label
        for label in COMPONENT_LABEL_CLASSES
        if frame.get(f"n_excluded_{label}") is not None
        and pd.to_numeric(frame[f"n_excluded_{label}"], errors="coerce").fillna(0).sum() > 0
    ]
    height = max(2.4, 0.32 * len(frame) + 1.4)
    figure, axis = plt.subplots(figsize=(7.4, height), layout="constrained")

    subjects = [str(value) for value in frame["subject"]]
    positions = np.arange(len(subjects))
    left = np.zeros(len(subjects))
    for label in present:
        counts = pd.to_numeric(frame[f"n_excluded_{label}"], errors="coerce").fillna(0).to_numpy()
        axis.barh(
            positions,
            counts,
            left=left,
            height=0.68,
            color=_LABEL_COLOURS.get(label, "0.5"),
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
        left += counts

    axis.set_yticks(positions, subjects, fontsize="small")
    axis.invert_yaxis()
    axis.set_xlabel("Components excluded")
    axis.set_title(f"What was removed · {len(subjects)} participant(s)")
    # Outside the axes, because the bars have no reserved corner: their lengths are the
    # measurement, so whichever corner a legend took would sit on top of whichever
    # participant had the most removed -- which is the row a reader most wants to see.
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=min(len(present), 4),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16 - 0.9 / max(len(subjects), 1)),
    )
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def decomposition_table(frame: pd.DataFrame) -> str:
    """Per-participant decomposition numbers, sorted by how much was removed."""
    if frame.empty:
        return ""
    ordered = frame.sort_values("variance_removed", ascending=False, na_position="last")
    rows = []
    for _, participant in ordered.iterrows():
        rows.append(
            [
                str(participant["subject"]),
                _count(participant["n_channels"]),
                _count(participant["n_components"]),
                _count(participant["data_rank"]),
                _count(participant["n_excluded"]),
                _count(participant["retained_dimensions"]),
                _percentage(participant["variance_removed"]),
                _decimal(participant["samples_per_squared_component"], places=0),
                _decimal(participant["condition_number"], places=0),
            ]
        )
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Channels"),
        Column("Fitted", group="Components"),
        Column("Rank", group="Components"),
        Column("Excluded", group="Components"),
        Column("Left", group="Components"),
        Column("Variance removed"),
        Column("Samples / n²"),
        Column("Condition"),
    )
    return (
        grid_table(columns, rows)
        + "<p>Sorted by variance removed, so the participants at the ends of that "
        "distribution are the first rows a reader meets. Sorting is the only emphasis "
        "here: no row is marked, because where a value stops being ordinary depends on the "
        "acquisition and is the reader's call.</p>"
    )


def _count(value: object) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return f"{int(value):,}"


def _percentage(value: object) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return f"{float(value):.1%}"


def _decimal(value: object, *, places: int = 2) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return f"{float(value):.{places}f}"


def variance_summary_html(pooled: object | None) -> str:
    """Variance removed across the cohort."""
    if pooled is None:
        return ""
    denominator = pooled.denominator
    if pooled.regime is BandRegime.INDIVIDUALS:
        values = ", ".join(
            f"{subject} {value:.1%}" for subject, value in pooled.per_subject.items()
        )
        rows = [(f"Variance removed (n={denominator.n_subjects})", values)]
    else:
        quartiles = pooled.quartiles
        spread = "" if quartiles is None else f" [{quartiles[0]:.1%}\u2013{quartiles[1]:.1%}]"
        rows = [
            (f"Variance removed (n={denominator.n_subjects})", f"{pooled.median:.1%}{spread}")
        ]
    return "<h4>Variance removed</h4>" + metric_table(rows)


def add_ica_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> pd.DataFrame:
    """Add the cohort decomposition section, or nothing when no participant recorded one."""
    frame = decomposition_frame(cohort)
    if frame.empty:
        return frame

    parts = [
        "<p>A decomposition estimates one mixing parameter per pair of channels, so how "
        "many samples went into it and how many independent directions the data actually "
        "held decide whether its components mean anything. Both travel with each "
        "participant below.</p>"
    ]

    offenders = rank_violations(frame)
    if offenders:
        parts.append(
            "<p><strong>More components than rank: "
            f"{', '.join(offenders)}.</strong> A rank-r dataset spans r directions, so a "
            "decomposition fitted with more than that is solving for directions the data "
            "does not contain. This is stated as a violation rather than as a number "
            "because it is algebraic &mdash; unlike every other value in this report, it "
            "does not depend on the recording, the montage or the reviewer.</p>"
        )

    parts.append(
        variance_summary_html(variance_pooled(frame, gates=gates))
    )
    parts.append(decomposition_table(frame))

    label_columns = [f"n_excluded_{label}" for label in COMPONENT_LABEL_CLASSES]
    if any(column in frame.columns and frame[column].notna().any() for column in label_columns):
        report.add_figure(
            fig=plot_label_composition(frame),
            title="What was removed, by detector",
            section=ICA_SECTION,
            tags=(ICA_TAG,),
            image_format=report_image_format(),
            replace=True,
        )
        plt.close("all")
        parts.append(
            "<h4>What was removed</h4>"
            "<p>Twelve components taken out for eye movement and twelve taken out for "
            "muscle are different recordings with the same headline number. The classes "
            "come from the detector that marked each component, read from the component "
            "table the exclusions themselves are applied from, so this cannot drift from "
            "the derivative. A description no detector class matched is counted as "
            "<em>other</em> rather than dropped.</p>"
        )

    report.add_html(
        html="".join(part for part in parts if part),
        title=ICA_TITLE,
        section=ICA_SECTION,
        tags=(ICA_TAG,),
        replace=True,
    )
    return frame


__all__: Sequence[str] = [
    "ICA_SECTION",
    "ICA_TAG",
    "ICA_TITLE",
    "add_ica_section",
    "decomposition_frame",
    "decomposition_table",
    "plot_label_composition",
    "rank_violations",
    "variance_pooled",
    "variance_summary_html",
]
