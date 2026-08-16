"""MNE report rendering for ocular artifact review (EOG blinks and saccades)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.ica_exclusions import (
    components_path_for_ica,
    read_component_statuses,
    read_ica_with_reviewed_exclusions,
)
from eeg_pipeline.preprocessing.report.build_record import save_subject_report
from eeg_pipeline.preprocessing.report.organize import (
    before_ica_component_review,
    drop_replaced_ica_eog_panels,
    move_tagged_content_before,
    open_subject_report,
)
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    draw_component_status_strip,
    report_image_format,
    apply_report_style,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Panel stating how many blinks each run's correlations were measured from.
OCULAR_DETECTION_TITLE = "Blink detection by run"

OCULAR_REPORT_TITLES = (
    "How to review EOG artifacts",
    OCULAR_DETECTION_TITLE,
    "Blink-locked EEG before and after provisional ICA",
    "ICA components: EOG correlation by run",
)


@dataclass(frozen=True)
class OcularReviewSettings:
    """Configuration for EOG detection and manual ICA review evidence."""

    enabled: bool = False
    eog_channels: str | list[str] = "eog"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OcularReviewSettings:
        return cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            eog_channels=values.get("eog_channels", cls.eog_channels),
        )


@dataclass(frozen=True)
class RunOcularReview:
    """MNE EOG detection evidence for one recording run."""

    recording_id: str
    blink_epoch_count: int
    #: Length of the run the blinks were counted over, which the rate is taken against.
    duration_s: float
    #: Absolute EOG correlation per component, maximized over the EOG channels.
    absolute_scores: np.ndarray
    #: Components flagged by MNE ``find_bads_eog`` for this run.
    flagged_components: tuple[int, ...]
    overlay_figure: plt.Figure


def _resolve_eog_channels(raw: mne.io.BaseRaw, settings: OcularReviewSettings) -> list[str]:
    """Return the configured ocular channels, failing fast on any missing name.

    Montages without dedicated EOG electrodes commonly use frontopolar EEG channels
    (Fp1, Fp2) as blink surrogates. Those names are accepted here and passed to MNE
    unchanged; the circularity that surrogates introduce is declared in the report
    guide rather than corrected for silently.
    """
    configured = settings.eog_channels
    channels = [configured] if isinstance(configured, str) else list(configured)
    if channels == ["eog"]:
        resolved = [
            name
            for name, channel_type in zip(raw.ch_names, raw.get_channel_types(), strict=True)
            if channel_type == "eog"
        ]
        if not resolved:
            raise ValueError(
                f"{raw.filenames[0]} contains no EOG-typed channel. Set "
                "ica.ocular_review.eog_channels to the surrogate channels used by this "
                'montage, for example ["Fp1", "Fp2"].'
            )
        return resolved

    missing = [name for name in channels if name not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Configured ocular channels {missing} do not exist in {raw.filenames[0]}."
        )
    return channels


def _surrogate_channels(raw: mne.io.BaseRaw, channels: list[str]) -> tuple[str, ...]:
    """Return the configured ocular channels that are EEG rather than EOG."""
    channel_types = dict(zip(raw.ch_names, raw.get_channel_types(), strict=True))
    return tuple(name for name in channels if channel_types[name] != "eog")


def _absolute_scores(scores: Any, *, component_count: int) -> np.ndarray:
    """Reduce MNE EOG scores to one absolute correlation per component.

    ``find_bads_eog`` returns one score vector per EOG channel. A component that
    correlates strongly with any single EOG channel is ocular evidence, so the
    channels are combined by taking the largest absolute correlation.
    """
    stacked = np.atleast_2d(np.asarray(scores, dtype=float))
    if stacked.shape[-1] != component_count:
        raise ValueError(
            f"EOG scores describe {stacked.shape[-1]} components, expected {component_count}."
        )
    if not np.isfinite(stacked).all():
        raise ValueError("MNE find_bads_eog returned non-finite correlations.")
    return np.max(np.abs(stacked), axis=0)


class UnusableEog(ValueError):
    """One run yielded no blink epoch to build ocular evidence from.

    The counterpart of :class:`~eeg_pipeline.preprocessing.ica_cardiac_review.UnusableEcg`,
    and a distinct type for the same reason: a run the blink detector cannot resolve is a
    property of the recording, to be recorded and reported, while a missing channel or a
    mistyped setting is a fault to fix. Catching plain ``RuntimeError`` around the overlay
    to keep a study running would swallow every genuine bug inside it as well.
    """


def _plot_run_overlay(
    raw: mne.io.BaseRaw,
    *,
    ica: mne.preprocessing.ICA,
    eog_channels: list[str],
    surrogates: tuple[str, ...],
    recording_id: str,
) -> tuple[plt.Figure, int]:
    """Plot blink-locked EEG global field power before and after provisional ICA.

    When frontopolar EEG channels stand in for EOG, they are excluded from the global
    field power. Blinks were detected on those channels, so the attenuation they show
    is guaranteed by construction; the remaining scalp is the honest evidence that
    ocular artifact was removed from the data that will be analyzed.
    """
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw,
        ch_name=eog_channels,
        verbose="ERROR",
    )
    # Checked before averaging rather than after. MNE raises a bare RuntimeError from
    # ``average()`` on an empty epoch set, which the caller cannot tell apart from a real
    # fault -- so it propagated out of the pipeline and ended a fifteen-subject run.
    if len(eog_epochs) == 0:
        raise UnusableEog(
            f"{recording_id}: the blink detector resolved no epoch, so there is no "
            f"blink-locked average to overlay."
        )
    # The ICA's own channels, not every EEG channel. A bad channel is not in the
    # decomposition and ``ICA.apply`` hands it back unchanged, so counting it in the
    # global field power below adds the same value to "before" and "after" and shrinks
    # the difference the panel exists to show.
    evoked = eog_epochs.average(picks=list(ica.ch_names))
    corrected = ica.apply(evoked.copy(), exclude=ica.exclude, verbose="ERROR")

    # Only the surrogates that are in the decomposition can be dropped from it, and only
    # those needed dropping. A surrogate PyPREP marked bad — Fp1 on sub-0012 here — was
    # never fitted, so it carries none of the circularity this exclusion exists to remove,
    # and asking to drop it raises "Channel(s) Fp1 not found, nothing dropped".
    fitted_surrogates = [name for name in surrogates if name in evoked.ch_names]
    excluded_surrogates = [name for name in surrogates if name not in evoked.ch_names]
    if fitted_surrogates:
        evoked = evoked.copy().drop_channels(fitted_surrogates)
        corrected = corrected.copy().drop_channels(fitted_surrogates)
        scope = f"excluding surrogates {', '.join(fitted_surrogates)}"
        if excluded_surrogates:
            # Named, because a reader comparing subjects would otherwise see the scope
            # change between them with nothing to explain it.
            scope += (
                f" ({', '.join(excluded_surrogates)} bad, so outside the decomposition)"
            )
    elif surrogates:
        scope = (
            f"all decomposed channels ({', '.join(surrogates)} bad, "
            "so outside the decomposition)"
        )
    else:
        scope = "all EEG channels"

    figure, axis = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    # Seconds, matching the R-locked panels of the cardiac review. The two sections answer
    # the same question about two artifacts over windows of the same length, so a reader
    # moving between them should not have to rescale by a thousand.
    times_s = evoked.times
    axis.plot(
        times_s,
        evoked.data.std(axis=0) * 1e6,
        color=BEFORE_COLOR,
        label="Before ICA",
    )
    axis.plot(
        times_s,
        corrected.data.std(axis=0) * 1e6,
        color=AFTER_COLOR,
        label="After ICA",
    )
    axis.axvline(0.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
    # A before/after pair invites eyeballing. Stating the attenuation makes the panel
    # decidable: 20*log10 of the peak amplitude ratio, the same convention the cardiac
    # QC uses for the equivalent quantity.
    before_peak = float(np.max(evoked.data.std(axis=0)))
    after_peak = float(np.max(corrected.data.std(axis=0)))
    attenuation_db = 20.0 * np.log10(before_peak / after_peak) if after_peak > 0 else float("inf")
    axis.annotate(
        f"peak GFP attenuation {attenuation_db:.1f} dB",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
    )
    axis.set(
        title=f"{recording_id} · {len(eog_epochs)} blink-locked epochs · {scope}",
        xlabel="Time from blink peak (s)",
        ylabel="Global field power (µV)",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure, len(eog_epochs)


def _threshold_band(run_reviews: Sequence[RunOcularReview]) -> tuple[float, float] | None:
    """Bracket the correlation at which ``find_bads_eog`` separated flagged from kept.

    MNE thresholds an adaptive z-score of the scores, so the cutoff is a property of each
    run's own distribution and no single correlation describes it. The decisions bracket
    it exactly, though: within one run the cutoff lies above every component left
    unflagged and at or below the lowest one flagged.

    Runs disagree about where that falls, so the returned band spans every run's bracket
    and is a statement about the session rather than about one run.

    ``None`` when no run flagged anything: the cutoff is then above every score the run
    produced and is unbounded above, and drawing a band there would put a threshold on the
    figure that no decision supports.
    """
    lows: list[float] = []
    highs: list[float] = []
    for review in run_reviews:
        flagged = set(review.flagged_components)
        if not flagged:
            continue
        scores = np.asarray(review.absolute_scores, dtype=float)
        kept = [score for index, score in enumerate(scores) if index not in flagged]
        highs.append(float(min(scores[index] for index in flagged)))
        # A run that flagged every component leaves no unflagged score to bound from
        # below; the bracket then starts at the lowest flagged score itself.
        lows.append(float(max(kept)) if kept else highs[-1])
    if not highs:
        return None
    return min(lows), max(highs)


def _plot_component_scores(
    run_reviews: list[RunOcularReview],
    *,
    excluded: Sequence[int],
) -> plt.Figure:
    """Plot every component's EOG correlation across runs on one axis.

    One figure carries the whole decomposition so that components can be compared
    against each other, which a per-component figure cannot show.
    """
    component_count = run_reviews[0].absolute_scores.size
    components = np.arange(component_count)
    # Width is capped: a decomposition with many components would otherwise produce a
    # figure so wide that the browser scales it down until nothing is readable.
    width = float(np.clip(0.22 * component_count, 8.0, 14.0))
    figure, (axis, status_axis) = plt.subplots(
        2,
        1,
        figsize=(width, 4.6),
        height_ratios=(12, 1),
        sharex=True,
        layout="constrained",
    )


    scores = np.stack([review.absolute_scores for review in run_reviews])
    medians = np.median(scores, axis=0)
    # A blink component correlates at ~0.9 while the rest of the decomposition sits
    # below 0.15. On a linear axis that one component sets the scale and flattens every
    # other one — including the ones the detector flagged — onto the floor, so the axis
    # is logarithmic and the comparison the panel exists for stays legible.
    #
    # Logarithmic also rules out bars, for the reason given in
    # ``report.summary.plot_variance_overview``: a bar reads its quantity as a length
    # from zero, and zero is at negative infinity here, so the length would be set by
    # the axis limit rather than by the data. Markers encode position only.
    positive = scores[scores > 0.0]
    floor = float(positive.min()) / 2.0 if positive.size else 1e-3
    axis.vlines(components, floor, medians, color="0.90", linewidth=0.7, zorder=1)
    axis.scatter(
        components,
        medians,
        s=26,
        color="0.45",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label="Median across runs",
    )
    for review in run_reviews:
        axis.scatter(
            components,
            review.absolute_scores,
            s=14,
            facecolor="none",
            edgecolor=AFTER_COLOR,
            linewidth=0.8,
            zorder=2,
            label="Individual runs" if review is run_reviews[0] else None,
        )

    flagged = sorted(
        {component for review in run_reviews for component in review.flagged_components}
    )
    if flagged:
        axis.scatter(
            flagged,
            np.max(scores, axis=0)[flagged],
            marker="x",
            color=FLAG_COLOR,
            s=48,
            zorder=4,
            label="Flagged by MNE find_bads_eog",
        )

    # Where the detector drew its line, measured from the decisions it made.
    #
    # ``find_bads_eog`` thresholds an adaptive z-score, so there is no fixed correlation
    # to draw and reimplementing the rule here would let the line drift away from the
    # crosses beside it. The decisions bracket it instead: within a run the cutoff sits
    # above every component left unflagged and no higher than the lowest one flagged.
    # Runs disagree about where that is, so the band spans every run's bracket.
    band = _threshold_band(run_reviews)
    if band is not None:
        low, high = band
        # ``fill_between`` rather than ``axhspan``: this axis draws no patches on purpose,
        # so that "there are no bars here" stays a checkable property of it. A shaded
        # region is a collection and leaves that intact.
        axis.fill_between(
            [-0.7, component_count - 0.3],
            low,
            high,
            color=FLAG_COLOR,
            alpha=0.10,
            linewidth=0,
            zorder=0,
            label="where find_bads_eog drew its line",
        )
        # Kept for the test that pins the bracket to the decisions it came from; the
        # drawn span alone cannot say which scores defined it.
        axis._eog_threshold_band = (low, high)
    axis.set(
        title=f"Absolute EOG correlation per component ({len(run_reviews)} runs)",
        ylabel="Absolute correlation",
        yscale="log",
        ylim=(floor, 1.2),
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

    draw_component_status_strip(
        status_axis,
        excluded=excluded,
        component_count=component_count,
    )
    plt.close(figure)
    return figure


def _ocular_review_guide_html(
    settings: OcularReviewSettings,
    *,
    surrogates: tuple[str, ...],
) -> str:
    guide = (
        "<p><strong>Manual ocular review; no components are excluded here.</strong> Blinks are "
        "detected from the configured ocular channels with MNE "
        "<code>create_eog_epochs</code>.</p>"
        "<p>Correlations and flags come from MNE <code>find_bads_eog</code> and are displayed "
        "without additional pipeline classification. Every run contributes its own score; a "
        "component that is ocular in one run only deserves scrutiny before exclusion.</p>"
        f"<p>Configured ocular channels: <code>{settings.eog_channels}</code>.</p>"
    )
    if not surrogates:
        return guide
    return guide + (
        "<p><strong>EEG surrogates in use, not dedicated EOG.</strong> Blink detection "
        f"uses <code>{', '.join(surrogates)}</code>, which are EEG channels entering the "
        "ICA decomposition itself. Component-to-surrogate correlation is therefore "
        "partly circular: a frontal brain component correlates with these channels by "
        "construction, so the correlation alone does not separate ocular from frontal "
        "sources. The scalp topography and the blink-locked time course are shown "
        "alongside it for that reason, and the global field power panels exclude the "
        "surrogate channels.</p>"
    )


def _clear_ocular_review(report: mne.Report) -> None:
    titles = {element.name for element in report._content if "ica-ocular-review" in element.tags}
    for title in titles:
        report.remove(title=title, tags=("ica-ocular-review",), remove_all=True)


def ocular_detection_html(run_reviews: Sequence[RunOcularReview]) -> str:
    """Render what the blink detector found in each run, before any correlation is read.

    Every ocular number downstream is a correlation against blink-locked activity, and a
    correlation is only as meaningful as the number of blinks behind it. That number was
    already measured — it titles each slide of the overlay carousel — but a carousel shows
    one run at a time, so comparing runs meant stepping through the slider and holding
    counts in memory. A run with four detected blinks produces a correlation that looks
    exactly like a run with two hundred, and nothing on the page distinguished them.

    The rate is given beside the count because runs differ in length, and a count alone
    confounds how often the participant blinked with how long the run was. What counts as
    too few blinks is left to the reviewer: it depends on the montage, the surrogate
    channels, and the task, none of which this table can see.
    """
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Duration (min)"),
        Column("Blink epochs"),
        Column("Blinks per minute"),
        Column("Components flagged"),
    )
    rows = [
        [
            run_label(review.recording_id),
            f"{review.duration_s / 60.0:.1f}",
            review.blink_epoch_count,
            f"{review.blink_epoch_count / (review.duration_s / 60.0):.1f}",
            len(review.flagged_components),
        ]
        for review in run_reviews
    ]
    return (
        "<p>What the blink detector found in each run. Every correlation in this section "
        "is measured against these blinks, so a run with few of them carries a "
        "correlation that means correspondingly less &mdash; while looking no different "
        "from any other run's.</p>"
        + grid_table(columns, rows)
        + "<p>Blink rate varies with the participant, the task, and whether the ocular "
        "channels are dedicated electrodes or frontopolar surrogates, so no count is "
        "read as adequate or inadequate here. The comparison that is available on this "
        "table is between runs of one session, where the montage and the participant are "
        "held fixed and a run standing apart from its neighbours is a fact about that "
        "run.</p>"
    )


def ocular_evidence_table(run_reviews: list[RunOcularReview]) -> pd.DataFrame:
    """Create a per-component summary of MNE EOG detection outputs."""
    scores = np.stack([review.absolute_scores for review in run_reviews])
    flag_counts = np.zeros(scores.shape[1], dtype=int)
    for review in run_reviews:
        flag_counts[list(review.flagged_components)] += 1
    return pd.DataFrame(
        {
            "component": np.arange(scores.shape[1]),
            "median_absolute_eog_correlation": np.median(scores, axis=0),
            "max_absolute_eog_correlation": np.max(scores, axis=0),
            "runs_flagged_by_find_bads_eog": flag_counts,
            "run_count": len(run_reviews),
        }
    )


def _unusable_ocular_html(unusable: Sequence[tuple[str, str]], *, n_total: int) -> str:
    """Name the runs excluded from the ocular evidence, and why.

    The overlays and correlations below are measured from the blinks that were resolved, so
    a reader comparing them against the session has to know which runs contributed none.
    """
    if not unusable:
        return ""
    rows = [[run_label(recording_id), reason] for recording_id, reason in unusable]
    columns = (
        Column("Run", align=Align.TEXT, code=True),
        Column("Why it is not in the evidence below", align=Align.TEXT),
    )
    return (
        "<h4>Runs excluded from the ocular evidence</h4>"
        f"<p><strong>{len(unusable)} of {n_total} run(s) yielded no blink epoch.</strong> "
        "Every panel in this section is built from the remaining runs and its counts describe "
        "those alone. The runs are named rather than dropped: a blink correction that cannot "
        "be verified against a blink is not the same as one that was verified and passed.</p>"
        + grid_table(columns, rows)
        + "<p>A run reaching this table has not necessarily got clean frontopolar data. The "
        "detector's amplitude threshold is relative to the recording, so a session whose "
        "frontopolar channels carry less variance than usual can resolve very few blinks "
        "while the participant blinked normally. The blink counts above are what to read this "
        "against.</p>"
    )


def _write_unusable_ocular_review(
    *,
    report_path: Path,
    output_path: Path,
    unusable: Sequence[tuple[str, str]],
) -> Path:
    """Record that no run yielded a blink epoch, in the report and in the sidecar.

    The empty table is written on purpose: a downstream reader distinguishes "reviewed and
    found nothing to show" from "never reviewed" by the file existing, and only the first is
    true here.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"recording_id": recording_id, "unusable_reason": reason}
            for recording_id, reason in unusable
        ]
    ).to_csv(output_path, sep="\t", index=False)

    report = open_subject_report(report_path)
    _clear_ocular_review(report)
    report.add_html(
        html=(
            "<p><strong>No run of this subject yielded a blink epoch, so there is no ocular "
            "review to draw.</strong> The frontopolar channels were read and the blink "
            "detector ran; it resolved no blink in any run.</p>"
            "<p>Reported rather than omitted, because the absence is the measurement: the "
            "ocular correction applied to this subject cannot be verified against a blink "
            "here, which is a different statement from its having been checked and found "
            "adequate.</p>"
            + _unusable_ocular_html(unusable, n_total=len(unusable))
        ),
        title=OCULAR_REPORT_TITLES[1],
        section="ICA ocular artifact review",
        tags=("ica", "eog", "ica-ocular-review", "eog-detection-summary"),
        replace=True,
    )
    save_subject_report(report, report_path, stage="ica-ocular-review")
    return output_path


def generate_ica_ocular_review(
    *,
    filtered_raw_paths: list[Path],
    ica_path: Path,
    report_path: Path,
    output_path: Path,
    settings: OcularReviewSettings,
) -> Path:
    """Append EOG diagnostics and review-only ICA evidence to an MNE report."""
    if not settings.enabled:
        raise ValueError("generate_ica_ocular_review requires ocular_review.enabled=true.")
    if not filtered_raw_paths:
        raise ValueError("No filtered raw recordings were provided for EOG review.")
    apply_report_style()

    # The blink overlays and the excluded-component marks must show the exclusions that
    # build the cleaned data, which live in the component table rather than the ICA file.
    ica = read_ica_with_reviewed_exclusions(ica_path)
    component_count = int(ica.n_components_)

    run_reviews = []
    unusable: list[tuple[str, str]] = []
    surrogates: tuple[str, ...] = ()
    for path in filtered_raw_paths:
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        eog_channels = _resolve_eog_channels(raw, settings)
        surrogates = _surrogate_channels(raw, eog_channels)
        recording_id = path.name.removesuffix("_proc-filt_raw.fif")
        flagged, scores = ica.find_bads_eog(raw, ch_name=eog_channels, verbose="ERROR")
        try:
            overlay_figure, blink_count = _plot_run_overlay(
                raw,
                ica=ica,
                eog_channels=eog_channels,
                surrogates=surrogates,
                recording_id=recording_id,
            )
        except UnusableEog as exc:
            # Only this signal: a broad except here would hide real faults in the overlay.
            unusable.append((recording_id, str(exc)))
            continue
        run_reviews.append(
            RunOcularReview(
                recording_id=recording_id,
                blink_epoch_count=blink_count,
                duration_s=float(raw.n_times) / float(raw.info["sfreq"]),
                absolute_scores=_absolute_scores(scores, component_count=component_count),
                flagged_components=tuple(int(index) for index in flagged),
                overlay_figure=overlay_figure,
            )
        )

    if not run_reviews:
        return _write_unusable_ocular_review(
            report_path=report_path,
            output_path=output_path,
            unusable=unusable,
        )

    read_component_statuses(
        components_path_for_ica(ica_path),
        component_count=component_count,
    )

    table = ocular_evidence_table(run_reviews)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)

    report = open_subject_report(report_path)
    _clear_ocular_review(report)
    # MNE's own EOG panels measure the same two quantities this section is about to render
    # per run, for one concatenated recording and with no indication of how many blinks
    # were behind them. Dropped here so a report built without the ocular review keeps
    # them, exactly as the cardiac review does with the ECG pair.
    drop_replaced_ica_eog_panels(report)
    section = "ICA ocular artifact review"
    report.add_html(
        html=_ocular_review_guide_html(settings, surrogates=surrogates),
        title=OCULAR_REPORT_TITLES[0],
        section=section,
        tags=("ica", "eog", "ica-ocular-review"),
        replace=True,
    )
    # Ahead of the overlays and the correlations, both of which are measured from these
    # blinks and neither of which states how many there were.
    report.add_html(
        html=ocular_detection_html(run_reviews)
        + _unusable_ocular_html(unusable, n_total=len(filtered_raw_paths)),
        title=OCULAR_REPORT_TITLES[1],
        section=section,
        tags=("ica", "eog", "ica-ocular-review", "eog-detection-summary"),
        replace=True,
    )
    report.add_figure(
        fig=[review.overlay_figure for review in run_reviews],
        title=OCULAR_REPORT_TITLES[2],
        caption=[review.recording_id for review in run_reviews],
        section=section,
        tags=("ica", "eog", "ica-ocular-review", "eog-run-review"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    report.add_figure(
        fig=_plot_component_scores(run_reviews, excluded=ica.exclude),
        title=OCULAR_REPORT_TITLES[3],
        section=section,
        tags=("ica", "eog", "ica-ocular-review", "eog-component-review"),
        image_format=report_image_format(),
        replace=True,
    )
    move_tagged_content_before(
        report,
        tag="ica-ocular-review",
        anchor=before_ica_component_review,
    )
    save_subject_report(report, report_path, stage="ica-ocular-review")
    return output_path


__all__ = [
    "OCULAR_DETECTION_TITLE",
    "OcularReviewSettings",
    "generate_ica_ocular_review",
    "ocular_detection_html",
    "ocular_evidence_table",
]
