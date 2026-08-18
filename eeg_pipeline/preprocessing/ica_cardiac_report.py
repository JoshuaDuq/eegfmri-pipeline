"""MNE report rendering for direct ECG and ICA cardiac review."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.build_record import save_subject_report
from eeg_pipeline.preprocessing.report.organize import (
    drop_replaced_ica_ecg_panels,
    open_subject_report,
)
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    MARK_COLOR,
    REFERENCE_COLOR,
    RUN_COLORS,
    draw_component_status_strip,
    report_image_format,
    apply_report_style,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table
from eeg_pipeline.preprocessing.ica_exclusions import (
    components_path_for_ica,
    promote_exclusions,
    read_component_statuses,
    read_ica_with_reviewed_exclusions,
)
from eeg_pipeline.preprocessing.ica_cardiac_review import (
    MARKER_TRAIN_SOURCE,
    CardiacReviewSettings,
    ComponentCardiacReview,
    RunCardiacReview,
    UnusableEcg,
    _build_component_cardiac_review,
    _build_run_cardiac_review,
    component_cardiac_evidence_table,
    component_run_cardiac_evidence_table,
    ctps_promotions,
)

#: Runs below this count give a quantile band no more meaning than a min-max envelope.
MINIMUM_RUNS_FOR_QUANTILE_BAND = 5


def beat_source_phrase(source: str) -> str:
    """Name the detector a panel's beats came from, for a title or a caption.

    Written out rather than printed as the internal constant, because the distinction the
    reader needs is not which code path ran but whose detection they are looking at: the
    marker train drove the upstream pulse correction, and the intervals it produced are
    drawn in the cardiac rhythm section, which is where a poor one shows up.
    """
    if source == MARKER_TRAIN_SOURCE:
        return "beat markers"
    return "R peaks detected from the ECG signal"

CARDIAC_REPORT_TITLES = (
    "How to review ECG artifacts",
    "ECG detection summary",
    "ECG detection and provisional correction by run",
    "ICA components: cardiac scores across the decomposition",
    "ICA components: R-locked cardiac evidence",
)


def _cardiac_threshold_band(review: ComponentCardiacReview) -> tuple[float, float] | None:
    """Bracket the score at which the detectors separated flagged from kept.

    The counterpart of the ocular panel's bracket, and derived the same way and for the
    same reason: both detectors threshold a statistic of each run's own distribution, so
    no single score describes the cutoff, but the decisions bound it exactly -- above every
    component a run left unflagged, and at or below the lowest one it flagged.

    Spans both detectors and every run, so it is a statement about the session. ``None``
    when nothing was flagged anywhere: the cutoff is then above every score observed, and
    a band drawn there would put a threshold on the figure that no decision supports.
    """
    lows: list[float] = []
    highs: list[float] = []
    for scores, flags in (
        (review.ctps_scores, review.ctps_flags),
        (review.correlation_scores, review.correlation_flags),
    ):
        magnitudes = np.abs(np.asarray(scores, dtype=float))
        marked = np.asarray(flags, dtype=bool)
        for run_index in range(magnitudes.shape[0]):
            flagged = np.flatnonzero(marked[run_index])
            if not flagged.size:
                continue
            kept = magnitudes[run_index][~marked[run_index]]
            highs.append(float(magnitudes[run_index][flagged].min()))
            lows.append(float(kept.max()) if kept.size else highs[-1])
    if not highs:
        return None
    return min(lows), max(highs)


def _plot_component_cardiac_scores(
    review: ComponentCardiacReview,
    *,
    excluded: Sequence[int],
):
    """Plot every component's cardiac scores on one axis, both detectors together.

    The per-component slides that follow show one component at a time, which cannot
    answer the question screening has to answer first: which components stand out
    *against the rest of this decomposition*. That matters more here than anywhere else
    in the report, because a cardiac component the classifier labelled brain and kept is
    the failure the per-component slides cannot lead you to — you have to already suspect
    a component to go and look at it.

    Both detectors are drawn because they disagree, and the disagreement is information.
    CTPS responds to phase locking with the beat and correlation to waveform similarity,
    so a component high on one and low on the other is a different review decision from
    one high on both.

    This is added beside the per-component slides, never instead of them: a suspicion
    raised here has to be answerable, and the answer is the R-locked evidence.
    """
    import matplotlib.pyplot as plt

    component_count = int(review.ctps_scores.shape[1])
    components = np.arange(component_count)
    # Width is capped for the reason the ocular panel caps it: a wide decomposition
    # otherwise renders a figure the browser scales down until nothing is readable.
    width = float(np.clip(0.22 * component_count, 8.0, 14.0))
    figure, (axis, status_axis) = plt.subplots(
        2,
        1,
        figsize=(width, 4.8),
        height_ratios=(12, 1),
        sharex=True,
        layout="constrained",
    )

    # Individual runs behind their median, and a logarithmic axis, for the reasons the
    # ocular panel states and this one first ignored: one cardiac component scores an
    # order of magnitude above the rest, and on a linear axis it sets the scale and
    # presses every other component -- including the ones the detector flagged -- onto the
    # floor. Drawn first, on the earlier version, most of a fifty-eight component
    # decomposition sat inside the bottom tenth of the panel.
    #
    # Runs are shown because consistency is the finding: a component cardiac in one run of
    # six is a different review decision from one cardiac throughout, and a median alone
    # cannot separate them.
    positive: list[float] = []
    for scores, flags, color, name in (
        (review.ctps_scores, review.ctps_flags, AFTER_COLOR, "CTPS"),
        (review.correlation_scores, review.correlation_flags, MARK_COLOR, "ECG correlation"),
    ):
        magnitudes = np.abs(np.asarray(scores, dtype=float))
        positive.extend(magnitudes[magnitudes > 0.0].ravel().tolist())
        for run_index in range(magnitudes.shape[0]):
            axis.scatter(
                components,
                magnitudes[run_index],
                s=12,
                facecolor="none",
                edgecolor=color,
                linewidth=0.7,
                zorder=2,
                label=f"{name}, individual runs" if run_index == 0 else None,
            )
        axis.scatter(
            components,
            np.median(magnitudes, axis=0),
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=f"{name}, median across runs",
        )
        flagged = np.flatnonzero(np.asarray(flags).any(axis=0))
        if flagged.size:
            axis.scatter(
                flagged,
                magnitudes.max(axis=0)[flagged],
                marker="x",
                color=FLAG_COLOR,
                s=48,
                zorder=4,
                label=f"Flagged by {name}",
            )

    # A low percentile rather than the minimum. A cardiac score is a correlation or a
    # kappa and is free to land arbitrarily close to zero, and one component that happens
    # to do so drags a logarithmic floor down several decades and flattens the panel it
    # was meant to open up. The ocular panel bounds on its minimum safely because an
    # absolute EOG correlation does not approach zero the same way.
    floor = float(np.percentile(positive, 1)) / 2.0 if positive else 1e-3
    axis.set(
        title=(
            f"Cardiac scores per component ({len(review.run_ids)} runs). "
            "Screening view: the slides below carry the evidence for any one component."
        ),
        ylabel="Score (absolute)",
        yscale="log",
        ylim=(floor, None),
    )
    band = _cardiac_threshold_band(review)
    if band is not None:
        low, high = band
        # ``fill_between`` rather than ``axhspan``, as in the ocular panel: this axis draws
        # no patches on purpose, so "there are no bars here" stays a checkable property.
        axis.fill_between(
            [-0.7, component_count - 0.3],
            low,
            high,
            color=FLAG_COLOR,
            alpha=0.10,
            linewidth=0,
            zorder=0,
            label="where the detectors drew their line",
        )
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

    draw_component_status_strip(
        status_axis,
        excluded=excluded,
        component_count=component_count,
    )
    plt.close(figure)
    return figure


def _plot_run_cardiac_review(
    review: RunCardiacReview,
    *,
    ica: mne.preprocessing.ICA,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplot_mosaic(
        [["ecg", "ecg", "heart_rate", "heart_rate"], ["gfp", "gfp", "before", "after"]],
        figsize=(16.0, 7.2),
        layout="constrained",
    )
    axes["ecg"].plot(
        review.representative_times,
        review.representative_ecg_mv,
        color="#000000",
        linewidth=0.9,
        zorder=2,
    )
    # Marking each detection at the amplitude the ECG actually had there, rather than
    # with a full-height rule, is what makes a misplaced detection visible: a marker
    # floating off the R peak is obvious, whereas a vertical line beside one is not.
    peak_amplitudes = np.interp(
        review.representative_peak_times,
        review.representative_times,
        review.representative_ecg_mv,
    )
    axes["ecg"].scatter(
        review.representative_peak_times,
        peak_amplitudes,
        marker="v",
        s=42,
        color=FLAG_COLOR,
        zorder=3,
        label="Detected R peak",
    )
    axes["ecg"].set(
        title=f"Representative ECG with {beat_source_phrase(review.beat_source)}",
        xlabel="Recording time (s)",
        ylabel="ECG (mV)",
    )
    axes["ecg"].legend(frameon=False, fontsize=7, loc="upper right")

    heart_rate_axis = axes["heart_rate"]
    # No connecting line. Successive beats are not a continuous signal, and joining a
    # few hundred of them turns an alternating detection pattern — the signature of a
    # doubled or missed R peak — into a solid block of ink that hides it.
    heart_rate_axis.scatter(
        review.rr_times,
        review.heart_rate_bpm,
        color=REFERENCE_COLOR,
        s=7,
        alpha=0.8,
        linewidth=0,
    )
    median_bpm = float(np.median(review.heart_rate_bpm))
    heart_rate_axis.axhline(
        median_bpm,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"median {median_bpm:.0f} bpm",
    )
    # Half and double the median, where the two detector failures land. A missed beat
    # spans two intervals and halves the instantaneous rate; a T wave counted as an R
    # peak doubles it. Both produce a second cloud parallel to the median rather than
    # scattered noise, and sub-0015's run-1 carries a clear one at half rate that the
    # panel drew without naming. The lines are guides, not thresholds: which of genuine
    # bradycardia, a pause, and a dropout produced a point on them is not decided here.
    for factor, style in ((0.5, (0, (4, 2))), (2.0, (0, (4, 2)))):
        heart_rate_axis.axhline(
            median_bpm * factor,
            color=GUIDE_COLOR,
            linestyle=style,
            linewidth=0.8,
            alpha=0.6,
        )
    heart_rate_axis.annotate(
        "½ × median: one beat spanned",
        xy=(0.0, median_bpm * 0.5),
        xycoords=("axes fraction", "data"),
        xytext=(3, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=6,
        color=GUIDE_COLOR,
    )
    # A handful of implausible intervals would otherwise set the axis and compress every
    # real beat into a band a few pixels tall, so the range is taken from a percentile
    # and the beats left outside it are counted rather than silently dropped.
    low, high = np.percentile(review.heart_rate_bpm, [1.0, 99.0])
    margin = max(5.0, 0.1 * (high - low))
    lower_limit, upper_limit = low - margin, high + margin
    outside = int(
        np.count_nonzero(
            (review.heart_rate_bpm < lower_limit) | (review.heart_rate_bpm > upper_limit)
        )
    )
    heart_rate_axis.set_ylim(lower_limit, upper_limit)
    notes = []
    if outside:
        notes.append(f"{outside} of {review.heart_rate_bpm.size} beats outside this range")
    # The suptitle reports beats per recording minute; this median is the typical
    # instantaneous rate. The two measure different things and agree only when every beat
    # was detected, so their ratio is stated here rather than left as an apparent
    # contradiction between the panel and the title. No threshold decides when to show it:
    # the ratio is a measurement, and what it implies is the reviewer's call.
    notes.append(
        f"detected {review.average_pulse_bpm:.0f}/recording min "
        f"= {review.average_pulse_bpm / median_bpm:.0%} of the median rate"
    )
    if notes:
        heart_rate_axis.annotate(
            "\n".join(notes),
            xy=(1.0, 1.0),
            xycoords="axes fraction",
            xytext=(-4, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.5,
            color=GUIDE_COLOR,
        )
    heart_rate_axis.set(
        title="Beat-to-beat heart rate",
        xlabel="Recording time (s)",
        ylabel="Heart rate (bpm)",
    )
    heart_rate_axis.legend(frameon=False, fontsize=7, loc="lower right")

    axes["gfp"].plot(
        review.locked_times,
        review.before_gfp_uv,
        color=BEFORE_COLOR,
        label="Before ICA",
    )
    axes["gfp"].plot(
        review.locked_times,
        review.after_gfp_uv,
        color=AFTER_COLOR,
        label="After ICA",
    )
    axes["gfp"].axvline(0.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
    axes["gfp"].set(
        title="R-locked EEG global field power",
        xlabel="Time from R peak (s)",
        ylabel="GFP (µV)",
    )
    axes["gfp"].legend(frameon=False)

    color_limit = float(np.max(np.abs([review.before_topography_uv, review.after_topography_uv])))
    topomap_image = None
    for name, values, title in (
        ("before", review.before_topography_uv, "Before provisional ICA"),
        ("after", review.after_topography_uv, "After provisional ICA"),
    ):
        topomap_image, _ = mne.viz.plot_topomap(
            values,
            ica.info,
            axes=axes[name],
            show=False,
            vlim=(-color_limit, color_limit),
        )
        axes[name].set_title(f"{title}\n{review.topography_time:+.3f} s")
    figure.colorbar(
        topomap_image,
        ax=[axes["before"], axes["after"]],
        label="Amplitude (µV)",
        shrink=0.72,
    )
    for name in ("ecg", "heart_rate", "gfp"):
        axis = axes[name]
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{review.recording_id} · {review.r_locked_epoch_count} R-locked epochs · "
        f"{review.average_pulse_bpm:.1f} detected beats per recording minute"
    )
    plt.close(figure)
    return figure


def _plot_component_cardiac_review(
    review: ComponentCardiacReview,
    *,
    ica: mne.preprocessing.ICA,
    component: int,
    status: str,
    status_description: str,
    minimum_runs_for_band: int = MINIMUM_RUNS_FOR_QUANTILE_BAND,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Four panels, not three. The two score panels are narrow because each carries one
    # column of points, while the waveform panel needs the width to resolve a QRS-width
    # deflection. Laid out as one grid rather than by splitting a panel afterwards:
    # removing an axes that a subgridspec was taken from collapses constrained layout.
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(14.6, 4.2),
        width_ratios=(1.0, 2.0, 0.62, 0.62),
        layout="constrained",
    )
    mne.viz.plot_topomap(
        ica.get_components()[:, component],
        ica.info,
        axes=axes[0],
        show=False,
        contours=6,
    )
    axes[0].set_title("Scalp topography")

    run_means = review.run_mean_z[:, component]
    median = np.median(run_means, axis=0)
    run_count = len(review.run_ids)
    run_colors = [RUN_COLORS[index % len(RUN_COLORS)] for index in range(run_count)]

    # Runs are drawn in their own colours so this panel can be read together with the
    # scores panel: a cardiac deflection present in one run only is a different finding
    # from one present in all of them, and a single shared colour hides which is which.
    for run_index, run_mean in enumerate(run_means):
        axes[1].plot(
            review.times,
            run_mean,
            color=run_colors[run_index],
            alpha=0.85,
            linewidth=0.9,
        )
    axes[1].plot(review.times, median, color="black", linewidth=2.0)
    # A quantile band drawn from a handful of runs is just the min-max envelope wearing
    # the clothes of a distribution, so it is only shown once there are enough runs.
    if run_count >= minimum_runs_for_band:
        lower, upper = np.quantile(run_means, [0.16, 0.84], axis=0)
        axes[1].fill_between(review.times, lower, upper, color="0.6", alpha=0.20)
    axes[1].axvline(0.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
    axes[1].set(
        title=f"R-locked ICA waveform and ECG timing ({run_count} runs)",
        xlabel="Time from R peak (s)",
        ylabel="Baseline-standardized amplitude (z)",
    )
    ecg_axis = axes[1].twinx()
    ecg_axis.plot(
        review.times,
        np.median(review.run_ecg_z, axis=0),
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="ECG median",
    )
    # The ECG trace is shown for timing only and its normalized amplitude carries no
    # interpretable scale, so it gets no numeric ticks to compete with the z axis.
    ecg_axis.set_ylabel("Normalized ECG (timing only)", color=GUIDE_COLOR, fontsize=8)
    ecg_axis.set_yticks([])

    # Correlation and CTPS are not commensurate: find_bads_ecg returns a signed Pearson
    # correlation against the ECG channel and a CTPS kappa that cannot be negative.
    # Sharing one axis put a correlation of -0.13 opposite a kappa of 0.15 and made them
    # read as one effect reflected about zero, so each gets its own axis and its own
    # scale.
    correlation_axis, ctps_axis = axes[2], axes[3]

    correlation = review.correlation_scores[:, component]
    ctps = review.ctps_scores[:, component]
    run_positions = np.linspace(-0.16, 0.16, run_count)
    for axis, scores, flags, title, label, floor in (
        (
            correlation_axis,
            correlation,
            review.correlation_flags[:, component],
            "ECG correlation (r)",
            "Correlation score (r)",
            None,
        ),
        (
            ctps_axis,
            ctps,
            review.ctps_flags[:, component],
            "CTPS (kappa)",
            "CTPS score (κ)",
            0.0,
        ),
    ):
        for run_index, offset in enumerate(run_positions):
            axis.scatter(
                offset,
                scores[run_index],
                color=run_colors[run_index],
                edgecolor="white",
                linewidth=0.8,
                s=32,
            )
            # Ring a flagged score rather than stamping over it: an opaque marker would
            # hide the run colour, which is what connects this panel to the waveform
            # panel. The ring is black because every hue is already spoken for by a run.
            if flags[run_index]:
                axis.scatter(
                    offset,
                    scores[run_index],
                    s=150,
                    facecolor="none",
                    edgecolor=MARK_COLOR,
                    linewidth=1.5,
                    zorder=1,
                )
        # The limits follow the scores. A fixed ±0.3 floor was padding every panel out to
        # a range most components never reach, which left their scores in a flat line
        # near zero and hid the differences between runs that this panel exists to show.
        span = float(scores.max() - scores.min())
        margin = max(0.02, 0.15 * span)
        lower = float(scores.min()) - margin
        axis.axhline(0.0, color=GUIDE_COLOR, linewidth=0.8)
        axis.set(
            title=title,
            ylabel=label,
            xticks=[],
            xlim=(-0.35, 0.35),
            # CTPS is bounded below at zero, so an axis that opens negative space invites
            # reading a sign into a quantity that has none.
            ylim=(lower if floor is None else max(floor, lower), float(scores.max()) + margin),
        )
    for axis in (axes[1], correlation_axis, ctps_axis):
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    # One run legend for the whole figure. Both data panels are keyed by the same run
    # colours, so repeating the key inside each of them cost two blocks of 7 pt text
    # sitting on top of the traces they were meant to explain.
    run_handles = [
        Line2D([], [], color=color, linewidth=1.6, label=run_id.rsplit("_", maxsplit=1)[-1])
        for color, run_id in zip(run_colors, review.run_ids, strict=True)
    ]
    run_handles.append(Line2D([], [], color="black", linewidth=2.0, label="Median"))
    run_handles.append(
        Line2D([], [], color=GUIDE_COLOR, linestyle="--", linewidth=1.2, label="ECG median")
    )
    # The flag key joins the run key rather than sitting inside a score panel, where it
    # covered the lowest points on an axis scaled to those very points. Naming the
    # detector here also keeps the scores' provenance on the figure when a slide is
    # exported on its own, away from the section prose that otherwise carries it.
    run_handles.append(
        Line2D(
            [],
            [],
            color=MARK_COLOR,
            marker="o",
            markerfacecolor="none",
            markersize=9,
            linestyle="none",
            label="flagged by MNE find_bads_ecg",
        )
    )
    figure.legend(
        handles=run_handles,
        loc="outside lower center",
        ncol=min(len(run_handles), 8),
        frameon=False,
        fontsize=7,
    )

    description = status_description or "No exclusion reason recorded"
    figure.suptitle(f"ICA{component:03d} · Current ICA status: {status} — {description}")
    plt.close(figure)
    return figure


def _ordered_cardiac_indices(content) -> list[int]:
    cardiac_indices = {
        element.name: index
        for index, element in enumerate(content)
        if "ica-cardiac-review" in element.tags
    }
    # A subset rather than the whole set. A subject whose ECG resolved in no run contributes
    # the detection summary and no figures, which is a shorter review rather than a broken
    # one. An entry that is *not* on the list is still a fault: that is a misnamed or stray
    # panel, which is what this guard was written to catch.
    unexpected = sorted(set(cardiac_indices) - set(CARDIAC_REPORT_TITLES))
    if unexpected:
        raise ValueError(
            "ICA cardiac-review content carries unrecognized report entries: "
            + ", ".join(unexpected)
        )
    return [
        cardiac_indices[title] for title in CARDIAC_REPORT_TITLES if title in cardiac_indices
    ]


def _organize_cardiac_review(report: mne.Report) -> None:
    content = report._content
    cardiac_indices = _ordered_cardiac_indices(content)
    remaining = [index for index in range(len(content)) if index not in cardiac_indices]
    targets = [
        index
        for index in remaining
        if "ica-component-review" in content[index].tags
        or content[index].section == "ICA: components"
    ]
    if not targets:
        raise ValueError("The report has no ICA component-review insertion point.")
    insertion_index = remaining.index(targets[0])
    report.reorder(remaining[:insertion_index] + cardiac_indices + remaining[insertion_index:])


def _clear_cardiac_review(report: mne.Report) -> None:
    titles = {element.name for element in report._content if "ica-cardiac-review" in element.tags}
    for title in titles:
        report.remove(title=title, tags=("ica-cardiac-review",), remove_all=True)


def _beat_source_sentence(beat_sources: Sequence[str]) -> str:
    """State where this section's beats came from, reading the runs rather than asserting.

    ``detect_ecg_events`` prefers an annotated beat train wherever a run carries one, and
    the two sources fail on different runs, so a section can hold both. The sentence has
    to be built from what the runs actually used: the panels are read to judge detection
    quality, and a claim about which detection they show is the one thing that must not be
    guessed. Where the markers were used, the cardiac rhythm section draws the intervals
    that same train produced, and is where a poor one becomes visible.
    """
    if not beat_sources:
        return ""
    total = len(beat_sources)
    from_markers = sum(1 for source in beat_sources if source == MARKER_TRAIN_SOURCE)
    if from_markers == 0:
        return (
            "<p>R peaks are detected directly from the configured ECG signal, so this "
            "review does not depend on an annotated beat train.</p>"
        )
    if from_markers == total:
        return (
            "<p>The beats every panel here is locked to are <strong>the recording's "
            "markers</strong>, not peaks detected from the ECG signal: the marker train "
            "is preferred wherever a run carries one, because it is the detection that "
            "drove the upstream pulse-artifact correction. The beat-detection panel in the "
            "cardiac rhythm section draws the intervals that train produced, and a run "
            "where the detector lost the trace carries every panel below on it.</p>"
        )
    return (
        f"<p>The beats these panels are locked to come from <strong>the recording's markers "
        f"on {from_markers} of {total} run(s)</strong> and from R peaks detected in the "
        "ECG signal on the rest: the marker train is preferred wherever a run carries "
        "one, because it is the detection that drove the upstream pulse-artifact "
        "correction. Each run panel names its own source in the ECG panel title, and the "
        "cardiac rhythm section draws the intervals each train produced.</p>"
    )


def _cardiac_review_guide_html(
    settings: CardiacReviewSettings,
    beat_sources: Sequence[str] = (),
) -> str:
    ctps_threshold = str(settings.ctps_threshold)
    return (
        "<p><strong>Manual ECG review; no components are excluded here.</strong></p>"
        + _beat_source_sentence(beat_sources)
        + "<p>Inspect the detected peaks, beat-to-beat heart rate, R-locked EEG, component "
        "topography, and R-locked component waveform directly. Correlation and CTPS scores "
        "and red × markers come from MNE <code>find_bads_ecg</code>. They are displayed "
        "without additional pipeline classification or recommendation.</p>"
        "<p>Each run reports two rates, which measure different things. The heading gives "
        "detected beats per minute <em>of recording</em>, so a detector that misses beats "
        "lowers it; the dashed line on the heart-rate panel is the median instantaneous "
        "<code>60/RR</code>, which a missed beat barely moves. The panel states the first "
        "as a percentage of the second. They agree at 100% only when every beat was found, "
        "so a lower figure means intervals were spanned rather than detected — by dropout, "
        "by a stretch of unusable ECG, or by a genuine pause. Which of those it was is "
        "visible in the interval series, not in the percentage.</p>"
        f"<p>R-locked epoch: {settings.epoch_window[0]:g} to "
        f"{settings.epoch_window[1]:g} s; baseline: {settings.baseline[0]:g} to "
        f"{settings.baseline[1]:g} s; MNE CTPS threshold: {ctps_threshold}. "
        "All runs contribute to the component displays.</p>"
    )


def _component_statuses(path: Path, *, component_count: int) -> pd.DataFrame:
    return read_component_statuses(path, component_count=component_count)


def run_cardiac_review_table(run_reviews: list[RunCardiacReview]) -> pd.DataFrame:
    """Create a neutral summary of MNE ECG detection outputs by run."""
    return pd.DataFrame(
        [
            {
                "recording_id": review.recording_id,
                "r_locked_epoch_count": review.r_locked_epoch_count,
                "mne_average_pulse_bpm": review.average_pulse_bpm,
            }
            for review in run_reviews
        ]
    )


def run_cardiac_review_html(run_reviews: list[RunCardiacReview]) -> str:
    """Render the per-run detection summary for the report.

    Rendered from the reviews rather than from :func:`run_cardiac_review_table`, whose
    column names belong to the TSV sidecar it is written as. Publishing that frame with
    ``DataFrame.to_html`` put ``r_locked_epoch_count`` and full BIDS recording ids into
    a document where every other table says "Run" and "run-1".
    """
    columns = (
        Column("Run", align=Align.TEXT),
        Column("R-locked epochs"),
        Column("Average rate (bpm)"),
    )
    rows = [
        [
            run_label(review.recording_id),
            f"{int(review.r_locked_epoch_count):,}",
            f"{float(review.average_pulse_bpm):.1f}",
        ]
        for review in run_reviews
    ]
    return grid_table(columns, rows)


def _cardiac_sidecar(output_path: Path, suffix: str) -> Path:
    expected_suffix = "_components.tsv"
    if not output_path.name.endswith(expected_suffix):
        raise ValueError(f"ECG component output must end with {expected_suffix!r}.")
    return output_path.with_name(output_path.name.removesuffix(expected_suffix) + f"_{suffix}.tsv")


def _unusable_runs_html(unusable: Sequence[tuple[str, str]], *, n_total: int) -> str:
    """Name the runs excluded from the cardiac evidence, and why.

    A denominator that quietly shrinks is the failure this exists to prevent: the panels
    below are built from the runs whose ECG resolved, and a reader comparing them against
    the session has to know which runs are not in them. The reason is carried verbatim from
    the detector rather than summarised, because "no R peaks" and "an implausible rate" are
    different recordings.
    """
    if not unusable:
        return ""
    rows = [[run_label(recording_id), reason] for recording_id, reason in unusable]
    columns = (
        Column("Run", align=Align.TEXT, code=True),
        Column("Why it is not in the evidence below", align=Align.TEXT),
    )
    return (
        f"<h4>Runs excluded from the cardiac evidence</h4>"
        f"<p><strong>{len(unusable)} of {n_total} run(s) carry no usable beat train.</strong> "
        "Their ECG did not yield a detectable R-peak series, so every panel in this section "
        "is built from the remaining runs and its counts describe those alone. The runs are "
        "named rather than dropped: a cardiac correction that cannot be verified against a "
        "beat train is not the same as one that was verified and passed.</p>"
        + grid_table(columns, rows)
        + "<p>No rate is reported for them. Forcing the detector's threshold low enough to "
        "return peaks on a weak ECG produces a beat count, not a heart rate &mdash; it "
        "double-counts T waves, and a fabricated rate in this table would be indistinguishable "
        "from a measured one.</p>"
    )


def _write_unusable_cardiac_review(
    *,
    report_path: Path,
    output_path: Path,
    unusable: Sequence[tuple[str, str]],
    settings: CardiacReviewSettings,
) -> Path:
    """Record that no run's ECG resolved, in the report and in the sidecars.

    The empty tables are written on purpose. A downstream reader distinguishes "this
    subject's cardiac review found nothing to report" from "this subject was never
    reviewed" by the presence of the file, and only the first of those is true here.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["component", "status", "status_description"]).to_csv(
        output_path, sep="\t", index=False
    )
    pd.DataFrame(
        [
            {"recording_id": recording_id, "unusable_reason": reason}
            for recording_id, reason in unusable
        ]
    ).to_csv(_cardiac_sidecar(output_path, "runs"), sep="\t", index=False)

    report = open_subject_report(report_path)
    _clear_cardiac_review(report)
    drop_replaced_ica_ecg_panels(report)
    report.add_html(
        html=(
            "<p><strong>No run of this subject carries a usable beat train, so there is no "
            "cardiac review to draw.</strong> The ECG channel exists and was read; the "
            "detector could not resolve an R-peak series from any run of it.</p>"
            "<p>This is reported rather than omitted because the absence is the measurement. "
            "It means the cardiac correction applied to this subject cannot be verified "
            "against its own heartbeat here, which is a different statement from the "
            "correction having been checked and found adequate.</p>"
            + _unusable_runs_html(unusable, n_total=len(unusable))
        ),
        title="ECG detection summary",
        section="ICA cardiac artifact review",
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-detection-summary"),
        replace=True,
    )
    _organize_cardiac_review(report)
    save_subject_report(report, report_path, stage="ica-cardiac-review")
    return output_path


def generate_ica_cardiac_review(
    *,
    filtered_raw_paths: list[Path],
    ica_path: Path,
    report_path: Path,
    output_path: Path,
    settings: CardiacReviewSettings,
    _rebuilding: bool = False,
) -> Path:
    """Append direct ECG diagnostics and ICA cardiac evidence to an MNE report.

    With ``settings.promote_exclusions`` the CTPS detections this review computes are also
    written into the component table as exclusions; otherwise nothing here changes what the
    cleaned data contain. ``_rebuilding`` is internal: a promotion invalidates the
    before/after panels drawn from the old exclusion set, so the function re-enters itself
    once to redraw them, and the flag stops it recursing again.
    """
    if not settings.enabled:
        raise ValueError("generate_ica_cardiac_review requires cardiac_review.enabled=true.")
    if not filtered_raw_paths:
        raise ValueError("No filtered raw recordings were provided for ECG review.")
    apply_report_style()
    # The "after ICA" traces below must show the exclusions that build the cleaned data,
    # which live in the component table rather than in the ICA file.
    ica = read_ica_with_reviewed_exclusions(ica_path)

    # A run whose ECG the detector cannot resolve is excluded from the evidence and named,
    # not thrown. Raising here ended the whole pipeline at the review stage: one detached
    # ECG lead in one run of one participant left fifteen participants with no report,
    # including the fourteen whose recordings were fine. The reviewer needs the runs that
    # did resolve, and needs to be told which ones did not.
    raws: list[mne.io.BaseRaw] = []
    run_reviews: list[RunCardiacReview] = []
    unusable: list[tuple[str, str]] = []
    for path in filtered_raw_paths:
        recording_id = path.name.removesuffix("_proc-filt_raw.fif")
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        try:
            review = _build_run_cardiac_review(
                raw,
                ica=ica,
                recording_id=recording_id,
                settings=settings,
            )
        except UnusableEcg as exc:
            # Only this signal. A broad except here would hide real faults in the review.
            unusable.append((recording_id, str(exc)))
            continue
        raws.append(raw)
        run_reviews.append(review)

    if not run_reviews:
        # Nothing to draw, and that is the finding rather than a failure to report one.
        return _write_unusable_cardiac_review(
            report_path=report_path,
            output_path=output_path,
            unusable=unusable,
            settings=settings,
        )

    component_review = _build_component_cardiac_review(
        raws,
        run_reviews,
        ica=ica,
        settings=settings,
    )

    if settings.promote_exclusions and not _rebuilding:
        newly_promoted = promote_exclusions(
            components_path_for_ica(ica_path),
            components=ctps_promotions(
                component_review,
                minimum_run_fraction=settings.promotion_minimum_run_fraction,
            ),
            component_count=int(ica.n_components_),
        )
        if newly_promoted:
            # Everything drawn below applies ``ica.exclude``, which was read before those
            # components were added. Redraw once against the exclusion set that will
            # actually build the cleaned data, so the report and the derivative agree.
            return generate_ica_cardiac_review(
                filtered_raw_paths=filtered_raw_paths,
                ica_path=ica_path,
                report_path=report_path,
                output_path=output_path,
                settings=settings,
                _rebuilding=True,
            )

    statuses = _component_statuses(
        components_path_for_ica(ica_path),
        component_count=int(ica.n_components_),
    )
    table = component_cardiac_evidence_table(component_review, statuses=statuses)
    run_table = run_cardiac_review_table(run_reviews)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)
    run_table.to_csv(
        _cardiac_sidecar(output_path, "runs"),
        sep="\t",
        index=False,
    )
    component_run_cardiac_evidence_table(component_review).to_csv(
        _cardiac_sidecar(output_path, "componentruns"),
        sep="\t",
        index=False,
    )

    report = open_subject_report(report_path)
    _clear_cardiac_review(report)
    # MNE's own ECG panels measure the same thing this section is about to render per
    # run, without saying whether the beats behind them were detected well. Dropped here
    # so that a report built without the cardiac review keeps them.
    drop_replaced_ica_ecg_panels(report)
    section = "ICA cardiac artifact review"
    report.add_html(
        html=_cardiac_review_guide_html(
            settings,
            beat_sources=tuple(review.beat_source for review in run_reviews),
        ),
        title="How to review ECG artifacts",
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review"),
        replace=True,
    )
    report.add_html(
        html=run_cardiac_review_html(run_reviews)
        + _unusable_runs_html(unusable, n_total=len(filtered_raw_paths)),
        title="ECG detection summary",
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-detection-summary"),
        replace=True,
    )
    report.add_figure(
        fig=[_plot_run_cardiac_review(review, ica=ica) for review in run_reviews],
        title="ECG detection and provisional correction by run",
        caption=[review.recording_id for review in run_reviews],
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-run-review"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    # Ahead of the per-component slides: screening comes before drilling down, and this
    # is the only panel that can show a cardiac component the classifier kept.
    report.add_figure(
        fig=_plot_component_cardiac_scores(component_review, excluded=ica.exclude),
        title="ICA components: cardiac scores across the decomposition",
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-component-review"),
        image_format=report_image_format(),
        replace=True,
    )
    component_figures = [
        _plot_component_cardiac_review(
            component_review,
            ica=ica,
            component=component,
            status=str(statuses.iloc[component]["status"]),
            status_description=str(statuses.iloc[component]["status_description"]),
        )
        for component in range(int(ica.n_components_))
    ]
    report.add_figure(
        fig=component_figures,
        title="ICA components: R-locked cardiac evidence",
        caption=[f"ICA{component:03d}" for component in range(int(ica.n_components_))],
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review", "ecg-component-review"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    _organize_cardiac_review(report)
    save_subject_report(report, report_path, stage="ica-cardiac-review")
    return output_path


__all__ = ["generate_ica_cardiac_review"]
