"""Which electrodes failed, and whether the same ones failed for everybody.

A subject report can say that four channels were bad. Only a cohort can say that three of
those four were bad for everybody, which is a fact about the cap rather than about the
participant, and the two call for entirely different responses: one is a hardware problem
to fix before the next session, the other is a recording to weigh.

That distinction is spatial, which is why this section carries a figure at all. A table
sorted by failure rate names the same electrodes; what it cannot show is that they are
adjacent -- that a whole posterior region went rather than four scattered contacts. The
topography is drawn from the sensor positions each recording carries, never from a montage
name, because a guessed montage places electrodes plausibly and, when the guess is wrong,
silently wrongly.

Where positions are unavailable or disagree between participants, the figure is dropped
and the ranked table stands alone. A cohort that cannot be drawn correctly is not drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.style import (
    apply_report_style,
    report_image_format,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

COVERAGE_SECTION = "Channel and region coverage"
COVERAGE_TITLE = "Which electrodes failed, and for how many"
COVERAGE_TAG = "cohort-coverage"

#: Distance below which two recordings are taken to have placed a sensor in the same spot,
#: in metres. Sensor positions come from digitisation or a montage lookup, so nominally
#: identical electrodes differ by rounding rather than by millimetres.
POSITION_TOLERANCE_M = 5e-3

#: Electrodes listed in the ranked table. The rest are named in the audit table rather
#: than filling a page with channels no participant had trouble with.
MAX_RANKED_CHANNELS = 15


@dataclass(frozen=True)
class CohortCoverage:
    """Per-electrode failure across the cohort, with each electrode's own denominator."""

    #: ``channel, n_participants, n_bad_participants, bad_fraction, x, y, z``.
    channels: pd.DataFrame
    #: Participants whose sensor positions could not be reconciled with the rest.
    disagreeing_subjects: tuple[str, ...]

    @property
    def has_positions(self) -> bool:
        return not self.disagreeing_subjects and self.channels[["x", "y", "z"]].notna().all().all()


def cohort_coverage(cohort: Cohort) -> CohortCoverage | None:
    """Count, per electrode, how many participants had it bad and how many had it at all.

    The denominator is per electrode, not per cohort. A montage that gained two channels
    partway through a study has electrodes only some participants ever recorded, and
    scoring those against the full cohort would understate their failure rate by exactly
    the number of participants who never had them.
    """
    frames = [
        (participant.subject, participant.channels)
        for participant in cohort.participants
        if not participant.channels.empty
    ]
    if not frames:
        return None

    counts: dict[str, dict[str, float]] = {}
    positions: dict[str, list[np.ndarray]] = {}
    disagreeing: set[str] = set()
    reference: dict[str, np.ndarray] = {}

    for subject, table in frames:
        for _, row in table.iterrows():
            name = str(row["channel"])
            entry = counts.setdefault(name, {"n_participants": 0.0, "n_bad": 0.0})
            entry["n_participants"] += 1.0
            if float(pd.to_numeric(row["n_runs_bad"], errors="coerce") or 0.0) > 0.0:
                entry["n_bad"] += 1.0
            location = pd.to_numeric(row[["x", "y", "z"]], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(location)):
                continue
            positions.setdefault(name, []).append(location)
            if name not in reference:
                reference[name] = location
            elif float(np.linalg.norm(location - reference[name])) > POSITION_TOLERANCE_M:
                disagreeing.add(subject)

    rows = []
    for name, entry in counts.items():
        placed = positions.get(name)
        median = np.median(np.vstack(placed), axis=0) if placed else np.full(3, np.nan)
        rows.append(
            {
                "channel": name,
                "n_participants": int(entry["n_participants"]),
                "n_bad_participants": int(entry["n_bad"]),
                "bad_fraction": entry["n_bad"] / entry["n_participants"],
                "x": float(median[0]),
                "y": float(median[1]),
                "z": float(median[2]),
            }
        )
    channels = pd.DataFrame(rows).sort_values(
        ["bad_fraction", "channel"], ascending=[False, True]
    )
    return CohortCoverage(
        channels=channels.reset_index(drop=True),
        disagreeing_subjects=tuple(sorted(disagreeing)),
    )


def _info_from(channels: pd.DataFrame) -> mne.Info:
    """Build the minimal Info a topography needs, from the recorded positions."""
    names = [str(name) for name in channels["channel"]]
    info = mne.create_info(names, sfreq=1.0, ch_types="eeg", verbose="ERROR")
    for index, (_, row) in enumerate(channels.iterrows()):
        info["chs"][index]["loc"][:3] = [float(row["x"]), float(row["y"]), float(row["z"])]
    return info


def plot_failure_topography(coverage: CohortCoverage) -> plt.Figure:
    """The share of participants for whom each electrode was bad, drawn on the head.

    The only thing in this section a table cannot do: show that the electrodes which
    failed are neighbours. Four scattered contacts and one dead region produce the same
    ranked list and mean entirely different things.
    """
    apply_report_style()
    figure, axis = plt.subplots(figsize=(4.6, 4.4), layout="constrained")
    fractions = coverage.channels["bad_fraction"].to_numpy(dtype=float)
    image, _ = mne.viz.plot_topomap(
        fractions,
        _info_from(coverage.channels),
        axes=axis,
        show=False,
        cmap="Reds",
        vlim=(0.0, max(float(fractions.max()), 1e-6)),
        contours=0,
        sensors="k.",
        # Interpolate only where there are sensors to interpolate between. The default
        # paints out to the head outline, which invents a failure rate for scalp the
        # montage never covered -- and does it most vividly next to the electrodes that
        # failed, which is exactly where a reader is looking.
        extrapolate="local",
    )
    bar = figure.colorbar(image, ax=axis, shrink=0.75, format=PercentFormatter(xmax=1.0))
    bar.set_label("Share of participants for whom it was bad")
    total = int(coverage.channels["n_participants"].max())
    axis.set_title(f"Where the cap failed · up to {total} participant(s) per electrode")
    return figure


def ranked_table(coverage: CohortCoverage) -> str:
    """Electrodes ordered by how often they failed, with their own denominators."""
    affected = coverage.channels[coverage.channels["n_bad_participants"] > 0]
    if affected.empty:
        return (
            "<p>No electrode was marked bad for any participant, so there is nothing to "
            "rank. The per-electrode denominators are still recorded in the audit "
            "table.</p>"
        )
    shown = affected.head(MAX_RANKED_CHANNELS)
    rows = [
        [
            str(row["channel"]),
            int(row["n_bad_participants"]),
            int(row["n_participants"]),
            f"{float(row['bad_fraction']):.0%}",
        ]
        for _, row in shown.iterrows()
    ]
    columns = (
        Column("Electrode", align=Align.TEXT, code=True),
        Column("Bad for"),
        Column("Recorded by"),
        Column("Share"),
    )
    more = len(affected) - len(shown)
    note = (
        f"<p>{more} further electrode(s) were bad for at least one participant; the audit "
        f"table carries them all.</p>"
        if more > 0
        else ""
    )
    return (
        grid_table(columns, rows)
        + "<p>Each electrode is scored against the participants that actually recorded it, "
        "not against the cohort. A montage that gained channels partway through a study "
        "has electrodes only some participants ever had, and scoring those against "
        "everybody would understate their failure rate by exactly the number of "
        "participants who never carried them.</p>" + note
    )


def participant_table(cohort: Cohort) -> str:
    """Bad-channel counts per participant, sorted by how many they lost."""
    rows = []
    for participant in cohort.participants:
        table = participant.channels
        if table.empty:
            continue
        bad = pd.to_numeric(table["n_runs_bad"], errors="coerce").fillna(0)
        rows.append(
            [
                participant.subject,
                int((bad > 0).sum()),
                int(len(table)),
                f"{float((bad > 0).mean()):.0%}",
            ]
        )
    if not rows:
        return ""
    rows.sort(key=lambda row: row[1], reverse=True)
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Bad channels"),
        Column("Channels recorded"),
        Column("Share"),
    )
    return grid_table(columns, rows)


def add_coverage_section(*, report: mne.Report, cohort: Cohort) -> CohortCoverage | None:
    """Add the cohort coverage section, or nothing when no participant recorded channels."""
    coverage = cohort_coverage(cohort)
    if coverage is None:
        return None

    parts = [
        "<p>A subject report can say four channels were bad. Only a cohort can say that "
        "three of them were bad for everybody, which is a fact about the cap rather than "
        "about any participant &mdash; a hardware problem to fix before the next session, "
        "not a recording to weigh.</p>"
    ]

    # Drawn only when some electrode actually failed. With none, every sensor is the same
    # colour and the scale beside it stretches to a fifth decimal place of nothing, which
    # reads as a measurement too small to see rather than as no failures at all. The
    # ranked table below says it in a sentence instead.
    any_failed = bool((coverage.channels["n_bad_participants"] > 0).any())
    if coverage.has_positions and any_failed:
        report.add_figure(
            fig=plot_failure_topography(coverage),
            title="Where the cap failed",
            section=COVERAGE_SECTION,
            tags=(COVERAGE_TAG,),
            image_format=report_image_format(has_dense_image=True),
            replace=True,
        )
        plt.close("all")
    elif coverage.has_positions:
        parts.append(
            "<p>No topography is drawn because no electrode was bad for any participant. "
            "A head with every sensor at zero and a colour scale spanning nothing would "
            "read as a failure too small to see rather than as no failure at all.</p>"
        )
    elif coverage.disagreeing_subjects:
        parts.append(
            "<p>No topography is drawn: "
            f"{', '.join(coverage.disagreeing_subjects)} placed at least one sensor more "
            "than a few millimetres from where the other participants placed it, so there "
            "is no single head the cohort can be drawn on. The ranked table below names "
            "the same electrodes without asserting where they sat.</p>"
        )
    else:
        parts.append(
            "<p>No topography is drawn, because the recordings carry no sensor positions. "
            "Placing the electrodes from a montage name would draw them plausibly and, "
            "where the name did not match the cap, silently wrongly.</p>"
        )

    parts.append(ranked_table(coverage))
    participants = participant_table(cohort)
    if participants:
        parts.append("<h4>Per participant</h4>" + participants)

    report.add_html(
        html="".join(parts),
        title=COVERAGE_TITLE,
        section=COVERAGE_SECTION,
        tags=(COVERAGE_TAG,),
        replace=True,
    )
    return coverage


__all__: Sequence[str] = [
    "COVERAGE_SECTION",
    "COVERAGE_TAG",
    "COVERAGE_TITLE",
    "MAX_RANKED_CHANNELS",
    "POSITION_TOLERANCE_M",
    "CohortCoverage",
    "add_coverage_section",
    "cohort_coverage",
    "participant_table",
    "plot_failure_topography",
    "ranked_table",
]
