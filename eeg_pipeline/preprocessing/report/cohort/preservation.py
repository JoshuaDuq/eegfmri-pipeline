"""Whether the cohort kept any brain signal through cleaning.

Every other section in this report measures what cleaning *removed*. This is the one that
asks what it left, and without it a cohort in which ICA ate the neural signal looks
excellent on every other page: the comb is gone, the spectra are quiet, the variance
removed is high, and there is nothing left to analyse.

Three lines of evidence, chosen because each is scale-free and therefore comparable
between participants.

*Posterior alpha prominence.* The height of the rhythm over the aperiodic background
directly beneath it, not its power. Absolute alpha power varies by an order of magnitude
between people and with electrode impedance, and it is mechanically coupled to variance
removed -- take out more variance and there is less power, whatever the rhythm did. A
prominence is a contrast and is immune to both.

*Split-half reliability*, for a paradigm that has trials. Projected to a common trial
count, because reliability grows with test length and a participant measured on forty
retained trials would otherwise score below one measured on a hundred and twenty for
reasons that have nothing to do with the recording.

*The two together.* A distribution of variance removed cannot separate "removed a lot and
kept the signal" from "removed a lot and destroyed it". Plotting cleaning against what
survived it can, one point per participant, and it is the only figure here that answers
the question the section is named for.

Nothing is fitted through that scatter and no coefficient is reported. A correlation over a
few dozen participants is not a finding, and a line drawn through a QC screening plot
invites one to be read out of it.
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
    paired_differences,
    project_reliability,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.record import AFTER, BEFORE
from eeg_pipeline.preprocessing.report.cohort.sidecar import Paradigm
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    MARK_COLOR,
    apply_report_style,
    report_image_format,
    separated_labels,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

PRESERVATION_SECTION = "Signal preservation"
PRESERVATION_TITLE = "What survived cleaning"
PRESERVATION_TAG = "cohort-preservation"

#: Participants above which the paired panel stops labelling its endpoints.
#:
#: Higher than the spaghetti panels allow, because the geometry is different: endpoints sit
#: in one column and are nudged apart when they crowd, so twenty fit in a panel where
#: twenty overlapping spectra would not.
MAX_LABELLED_PARTICIPANTS = 20

#: Participants above which the scatter stops labelling its points.
#:
#: Higher still, and deliberately so. This figure exists to identify which participant to
#: look at; one that shows two recordings in the bottom-right corner without naming them
#: has failed at the only thing it is for. Points are separated in two dimensions, so they
#: crowd far later than endpoints in a column. Above this, the audit table carries the
#: identities.
MAX_LABELLED_SCATTER = 30

#: Movement in a peak frequency worth remarking on, in hertz.
#:
#: The same separation two maxima need before they count as different peaks: below it the
#: frequency has stayed on one bump and moved within it, which is resolution rather than
#: change. Not a threshold on data quality -- nothing is graded by it -- only the point at
#: which a shift is large enough to be worth pairing with the contest that explains it.
CONTESTED_SHIFT_HZ = 1.0

#: Prominence at which a rhythm is exactly level with the background beneath it.
#:
#: Definitional rather than a threshold: it is where "no measurable peak" sits. Drawn as a
#: guide so a reader can see which participants are near it, never as a pass mark.
NO_PROMINENCE_DB = 0.0


def alpha_frame(cohort: Cohort) -> pd.DataFrame:
    """Prominence and peak frequency per participant, per stage.

    A peak frequency is present only where the participant's prominence cleared its own
    spectrum's roughness. Where it did not, the column is blank rather than carrying the
    argmax of noise, and the resolvable flag records that as a measurement.
    """
    rows: list[dict[str, object]] = []
    for participant in cohort.participants:
        measurements = participant.measurements
        row: dict[str, object] = {"subject": participant.subject}
        found = False
        for stage in (BEFORE, AFTER):
            prominence = measurements.get(f"alpha_prominence_db_{stage}")
            row[f"prominence_{stage}"] = (
                np.nan if prominence is None else float(prominence)
            )
            row[f"resolvable_{stage}"] = bool(
                measurements.get(f"alpha_peak_resolvable_{stage}", False)
            )
            peak = measurements.get(f"alpha_peak_frequency_hz_{stage}")
            row[f"peak_hz_{stage}"] = np.nan if peak is None else float(peak)
            row[f"contested_{stage}"] = bool(
                measurements.get(f"alpha_peak_contested_{stage}", False)
            )
            rival = measurements.get(f"alpha_runner_up_hz_{stage}")
            row[f"runner_up_hz_{stage}"] = np.nan if rival is None else float(rival)
            gap = measurements.get(f"alpha_runner_up_gap_db_{stage}")
            row[f"runner_up_gap_db_{stage}"] = np.nan if gap is None else float(gap)
            found = found or prominence is not None
        if found:
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_frame(cohort: Cohort) -> pd.DataFrame:
    """Split-half reliability per participant, observed and projected to a common length.

    The reference is the smallest retained trial count in the cohort. Projecting *down* to
    it rather than up to some nominal figure means no participant's value is extrapolated
    beyond the evidence it actually has.
    """
    rows: list[dict[str, object]] = []
    for participant in cohort.participants:
        if participant.paradigm is not Paradigm.TASK:
            continue
        reliability = participant.measurements.get("split_half_r")
        trials = participant.measurements.get("split_half_n_trials")
        if reliability is None or trials is None:
            continue
        rows.append(
            {
                "subject": participant.subject,
                "reliability": float(reliability),
                "n_trials": int(trials),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    reference = int(frame["n_trials"].min())
    frame["reference_trials"] = reference
    frame["projected"] = [
        project_reliability(
            float(row["reliability"]),
            observed_trials=int(row["n_trials"]),
            reference_trials=reference,
        )
        for _, row in frame.iterrows()
    ]
    return frame


def cleaning_versus_signal(cohort: Cohort) -> pd.DataFrame:
    """Variance removed against the rhythm that survived it, one row per participant."""
    rows: list[dict[str, object]] = []
    for participant in cohort.participants:
        measurements = participant.measurements
        removed = measurements.get("variance_removed")
        prominence = measurements.get(f"alpha_prominence_db_{AFTER}")
        if removed is None or prominence is None:
            continue
        rows.append(
            {
                "subject": participant.subject,
                "variance_removed": float(removed),
                "prominence_db": float(prominence),
                "resolvable": bool(measurements.get(f"alpha_peak_resolvable_{AFTER}", False)),
            }
        )
    return pd.DataFrame(rows)


def _label_points(axis: plt.Axes, xs, ys, labels) -> None:
    for x, y, label in zip(xs, ys, labels):
        axis.annotate(
            str(label),
            xy=(x, y),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize="x-small",
            color=GUIDE_COLOR,
            va="center",
            annotation_clip=False,
        )


def plot_alpha_prominence(frame: pd.DataFrame) -> plt.Figure:
    """Prominence before against after, paired within participant.

    Paired because before and after are two measurements of one recording. The guide at
    zero is where a rhythm sits exactly level with the background beneath it, which is
    definitional rather than a threshold anyone chose.
    """
    apply_report_style()
    figure, axis = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
    label_points = len(frame) <= MAX_LABELLED_PARTICIPANTS

    endpoints: list[tuple[float, str]] = []
    for _, participant in frame.iterrows():
        before = participant[f"prominence_{BEFORE}"]
        after = participant[f"prominence_{AFTER}"]
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        axis.plot([0, 1], [before, after], color=GUIDE_COLOR, linewidth=1.0, zorder=1)
        axis.plot([0], [before], marker="o", markersize=4.0, color=BEFORE_COLOR, zorder=2)
        axis.plot([1], [after], marker="o", markersize=4.0, color=AFTER_COLOR, zorder=2)
        endpoints.append((float(after), str(participant["subject"])))

    axis.axhline(NO_PROMINENCE_DB, color=MARK_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
    axis.set_xticks([0, 1], ["Before ICA", "After ICA"])
    axis.set_xlim(-0.25, 1.4)
    axis.set_ylabel("Posterior alpha prominence (dB)")
    axis.set_title(f"Alpha over its own background · {len(endpoints)} participant(s)")
    if label_points and endpoints:
        low, high = axis.get_ylim()
        for value, label in separated_labels(endpoints, minimum_gap=(high - low) * 0.05):
            axis.annotate(
                label,
                xy=(1.06, value),
                fontsize="x-small",
                color=GUIDE_COLOR,
                va="center",
                annotation_clip=False,
            )
    return figure


def plot_cleaning_versus_signal(frame: pd.DataFrame) -> plt.Figure:
    """Cleaning against what survived it, one point per participant.

    No line is fitted and no coefficient is reported. A correlation over a few dozen
    participants is not a finding, and a trend line through a screening plot invites one
    to be read out of it. The figure is here so that a participant with a great deal
    removed and nothing left is visible as itself, which no distribution of either axis
    alone can show.
    """
    apply_report_style()
    figure, axis = plt.subplots(figsize=(6.0, 4.4), layout="constrained")
    resolvable = frame[frame["resolvable"]]
    unresolved = frame[~frame["resolvable"]]

    axis.scatter(
        resolvable["variance_removed"] * 100.0,
        resolvable["prominence_db"],
        s=42,
        color=AFTER_COLOR,
        edgecolor="none",
        label="Resolvable alpha peak",
        zorder=3,
    )
    if not unresolved.empty:
        axis.scatter(
            unresolved["variance_removed"] * 100.0,
            unresolved["prominence_db"],
            s=42,
            facecolor="none",
            edgecolor=MARK_COLOR,
            linewidth=1.0,
            label="No resolvable peak",
            zorder=3,
        )
    axis.axhline(NO_PROMINENCE_DB, color=MARK_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)

    if len(frame) <= MAX_LABELLED_SCATTER:
        _label_points(
            axis,
            frame["variance_removed"] * 100.0,
            frame["prominence_db"],
            frame["subject"],
        )
    axis.set_xlabel("Sensor variance removed by ICA (%)")
    axis.set_ylabel("Posterior alpha prominence after ICA (dB)")
    axis.set_title(f"Cleaning against what survived it · {len(frame)} participant(s)")
    axis.legend(frameon=False, fontsize=8, loc="best")
    return figure


def reliability_table(frame: pd.DataFrame) -> str:
    """Reliability per participant, observed and projected to a common trial count."""
    if frame.empty:
        return ""
    reference = int(frame["reference_trials"].iloc[0])
    rows = [
        [
            str(row["subject"]),
            int(row["n_trials"]),
            f"{float(row['reliability']):.2f}",
            f"{float(row['projected']):.2f}",
        ]
        for _, row in frame.iterrows()
    ]
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Retained trials"),
        Column("Observed"),
        Column(f"At {reference} trials"),
    )
    return (
        grid_table(columns, rows)
        + f"<p>Reliability grows with the number of trials it is measured on, so the "
        f"observed column compares recordings of different lengths and the projected one "
        f"does not. Every participant is stepped by Spearman-Brown to {reference} trials, "
        f"the smallest retained count in this cohort, so that no value is extrapolated "
        f"beyond the evidence behind it.</p>"
    )


def alpha_table(frame: pd.DataFrame) -> str:
    """Prominence and peak frequency per participant, with the shift through cleaning."""
    if frame.empty:
        return ""
    paired = {
        str(row["subject"]): float(row[f"prominence_{BEFORE}"])
        for _, row in frame.iterrows()
        if np.isfinite(row[f"prominence_{BEFORE}"]) and np.isfinite(row[f"prominence_{AFTER}"])
    }
    after = {
        str(row["subject"]): float(row[f"prominence_{AFTER}"])
        for _, row in frame.iterrows()
        if str(row["subject"]) in paired
    }
    shifts = paired_differences(paired, after) if paired else {}

    rows = []
    contested: list[str] = []
    for _, participant in frame.iterrows():
        subject = str(participant["subject"])
        peak = participant[f"peak_hz_{AFTER}"]
        rival = participant.get(f"runner_up_hz_{AFTER}", np.nan)
        gap = participant.get(f"runner_up_gap_db_{AFTER}", np.nan)
        # Either stage. A contest before cleaning is what makes an apparent shift through
        # cleaning meaningless, and that is the case the paired panel above puts in front
        # of the reader -- so checking only the final stage would flag the one participant
        # whose frequency is settled and stay quiet about the one whose is not.
        if bool(participant.get(f"contested_{BEFORE}", False)) or bool(
            participant.get(f"contested_{AFTER}", False)
        ):
            contested.append(subject)
        rows.append(
            [
                subject,
                _decimal(participant[f"prominence_{BEFORE}"], places=1),
                _decimal(participant[f"prominence_{AFTER}"], places=1),
                _decimal(shifts.get(subject), places=1),
                "—" if not np.isfinite(peak) else f"{float(peak):.1f}",
                "—" if not np.isfinite(rival) else f"{float(rival):.1f}",
                "—" if not np.isfinite(gap) else f"{float(gap):.1f}",
            ]
        )
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Before", group="Alpha prominence (dB)"),
        Column("After", group="Alpha prominence (dB)"),
        Column("Shift", group="Alpha prominence (dB)"),
        Column("Peak (Hz)"),
        Column("Next peak (Hz)", group="Second candidate"),
        Column("Below by (dB)", group="Second candidate"),
    )
    unresolved = int((~frame[f"resolvable_{AFTER}"]).sum())
    note = ""
    if unresolved:
        note = (
            f"<p>{unresolved} participant(s) carry no peak frequency: their prominence did "
            f"not clear their own spectrum's roughness, or their largest bin sat on the "
            f"edge of the band rather than inside it, where the spectrum may simply go on "
            f"rising. Recording a frequency for them would put a plausible-looking number "
            f"into the cohort that no rhythm produced.</p>"
        )
    if contested:
        note += (
            "<p><strong>The peak frequency is contested for "
            + ", ".join(contested)
            + ".</strong> A second bump in the band sits closer to the winner than the "
            "spectrum's own roughness, so which one the maximum lands on is not settled by "
            "the data: a change too small to matter anywhere else moves the reported "
            "frequency by several hertz. Their entries in the distribution above should be "
            "read as either candidate rather than as a located rhythm.</p>"
        )
    moved = _moved_while_contested(frame)
    if moved:
        note += (
            "<p>"
            + ", ".join(f"{subject} ({shift})" for subject, shift in moved)
            + " appear to have changed peak frequency through cleaning, but were contested "
            "at one end of that comparison. The maximum crossed from one bump to the other "
            "rather than a rhythm moving, so the apparent shift is a property of taking an "
            "argmax over a band with two candidates in it.</p>"
        )
    return grid_table(columns, rows) + note


def _moved_while_contested(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Participants whose peak frequency moved between stages while a contest was open.

    The two facts are only alarming together. A frequency that moved with one clear peak at
    each end is a real change worth reading; a frequency that moved while two bumps were
    within the noise of each other is the argmax changing its mind.
    """
    moved: list[tuple[str, str]] = []
    for _, participant in frame.iterrows():
        before = participant.get(f"peak_hz_{BEFORE}", np.nan)
        after = participant.get(f"peak_hz_{AFTER}", np.nan)
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        contested = bool(participant.get(f"contested_{BEFORE}", False)) or bool(
            participant.get(f"contested_{AFTER}", False)
        )
        if contested and abs(float(after) - float(before)) >= CONTESTED_SHIFT_HZ:
            moved.append(
                (str(participant["subject"]), f"{float(before):.1f} to {float(after):.1f} Hz")
            )
    return moved


def _decimal(value: object, *, places: int = 2) -> str | None:
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else f"{number:.{places}f}"


def add_preservation_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> None:
    """Add the cohort preservation section, or nothing when no evidence supports one."""
    prominence = alpha_frame(cohort)
    reliability = reliability_frame(cohort)
    scatter = cleaning_versus_signal(cohort)
    if prominence.empty and reliability.empty and scatter.empty:
        return

    parts = [
        "<p>Every other section measures what cleaning removed. This one asks what it "
        "left. A cohort in which the decomposition took the neural signal with the "
        "artifact looks excellent everywhere else &mdash; quiet spectra, no comb, a high "
        "variance removed &mdash; and has nothing left to analyse.</p>"
    ]

    if not prominence.empty:
        report.add_figure(
            fig=plot_alpha_prominence(prominence),
            title="Alpha prominence either side of the exclusions",
            section=PRESERVATION_SECTION,
            tags=(PRESERVATION_TAG,),
            image_format=report_image_format(),
            replace=True,
        )
        plt.close("all")
        parts.append(
            "<h4>Posterior alpha</h4>"
            "<p>Prominence is the height of the rhythm over the aperiodic background "
            "directly beneath it, not its power. Absolute alpha power varies by an order "
            "of magnitude between people and with electrode impedance, and it falls "
            "mechanically as variance is removed; a contrast against the local background "
            "is immune to both.</p>" + alpha_table(prominence)
        )
    if not reliability.empty:
        parts.append("<h4>Evoked reliability</h4>" + reliability_table(reliability))
    if not scatter.empty:
        report.add_figure(
            fig=plot_cleaning_versus_signal(scatter),
            title="Cleaning against what survived it",
            section=PRESERVATION_SECTION,
            tags=(PRESERVATION_TAG,),
            image_format=report_image_format(),
            replace=True,
        )
        plt.close("all")
        parts.append(
            "<h4>Cleaning against signal</h4>"
            "<p>A distribution of variance removed cannot separate a participant that had "
            "a great deal taken out and kept its rhythm from one that had a great deal "
            "taken out and lost it. Both axes together can, and that is the only reason "
            "this figure exists. No line is fitted through it and no coefficient is "
            "reported: a correlation over this many participants would not be a finding, "
            "and a trend line invites one to be read out of a screening plot.</p>"
        )

    report.add_html(
        html="".join(parts),
        title=PRESERVATION_TITLE,
        section=PRESERVATION_SECTION,
        tags=(PRESERVATION_TAG,),
        replace=True,
    )


__all__: Sequence[str] = [
    "MAX_LABELLED_PARTICIPANTS",
    "MAX_LABELLED_SCATTER",
    "NO_PROMINENCE_DB",
    "PRESERVATION_SECTION",
    "PRESERVATION_TAG",
    "PRESERVATION_TITLE",
    "add_preservation_section",
    "alpha_frame",
    "alpha_table",
    "cleaning_versus_signal",
    "plot_alpha_prominence",
    "plot_cleaning_versus_signal",
    "reliability_frame",
    "reliability_table",
]
