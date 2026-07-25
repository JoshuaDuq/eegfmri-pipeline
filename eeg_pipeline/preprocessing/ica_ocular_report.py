"""MNE report rendering for ocular artifact review (EOG blinks and saccades)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.organize import (
    before_ica_component_review,
    move_tagged_content_before,
)
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    report_image_format,
    apply_report_style,
)

OCULAR_REPORT_TITLES = (
    "How to review EOG artifacts",
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
    evoked = eog_epochs.average(picks="eeg")
    corrected = ica.apply(evoked.copy(), exclude=ica.exclude, verbose="ERROR")
    if surrogates:
        evoked = evoked.copy().drop_channels(list(surrogates))
        corrected = corrected.copy().drop_channels(list(surrogates))
        scope = f"excluding surrogates {', '.join(surrogates)}"
    else:
        scope = "all EEG channels"

    figure, axis = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    times_ms = evoked.times * 1e3
    axis.plot(
        times_ms,
        evoked.data.std(axis=0) * 1e6,
        color=BEFORE_COLOR,
        label="Before ICA",
    )
    axis.plot(
        times_ms,
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
        xlabel="Time from blink peak (ms)",
        ylabel="Global field power (µV)",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure, len(eog_epochs)


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
    axis.bar(
        components,
        np.median(scores, axis=0),
        color="0.80",
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
            label="Flagged by MNE find_bads_eog",
        )
    axis.set(
        title=f"Absolute EOG correlation per component ({len(run_reviews)} runs)",
        ylabel="Absolute correlation",
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

    # Exclusion status lives in its own strip. Drawn as full-height shading it would
    # cover a third of the axis and outweigh the correlations the panel is about.
    excluded_set = set(int(component) for component in excluded)
    status_axis.bar(
        components,
        1.0,
        width=1.0,
        color=[FLAG_COLOR if c in excluded_set else "0.88" for c in components],
    )
    status_axis.set(
        xlim=(-0.7, component_count - 0.3),
        ylim=(0, 1),
        yticks=[],
        xlabel="ICA component",
    )
    status_axis.set_ylabel(
        f"excluded\n({len(excluded_set)}/{component_count})",
        fontsize=6,
        rotation=0,
        ha="right",
        va="center",
    )
    status_axis.spines[["top", "right", "left"]].set_visible(False)
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

    ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
    component_count = int(ica.n_components_)

    run_reviews = []
    surrogates: tuple[str, ...] = ()
    for path in filtered_raw_paths:
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        eog_channels = _resolve_eog_channels(raw, settings)
        surrogates = _surrogate_channels(raw, eog_channels)
        recording_id = path.name.removesuffix("_proc-filt_raw.fif")
        flagged, scores = ica.find_bads_eog(raw, ch_name=eog_channels, verbose="ERROR")
        overlay_figure, blink_count = _plot_run_overlay(
            raw,
            ica=ica,
            eog_channels=eog_channels,
            surrogates=surrogates,
            recording_id=recording_id,
        )
        run_reviews.append(
            RunOcularReview(
                recording_id=recording_id,
                blink_epoch_count=blink_count,
                absolute_scores=_absolute_scores(scores, component_count=component_count),
                flagged_components=tuple(int(index) for index in flagged),
                overlay_figure=overlay_figure,
            )
        )

    component_status_path = ica_path.with_name(
        ica_path.name.replace("_proc-ica_ica.fif", "_proc-ica_components.tsv")
    )
    statuses = pd.read_csv(component_status_path, sep="\t")
    if not np.array_equal(statuses["component"].to_numpy(), np.arange(component_count)):
        raise ValueError(f"ICA component status table is invalid: {component_status_path}")

    table = ocular_evidence_table(run_reviews)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)

    report = mne.open_report(report_path)
    _clear_ocular_review(report)
    section = "ICA ocular artifact review"
    report.add_html(
        html=_ocular_review_guide_html(settings, surrogates=surrogates),
        title=OCULAR_REPORT_TITLES[0],
        section=section,
        tags=("ica", "eog", "ica-ocular-review"),
        replace=True,
    )
    report.add_figure(
        fig=[review.overlay_figure for review in run_reviews],
        title=OCULAR_REPORT_TITLES[1],
        caption=[review.recording_id for review in run_reviews],
        section=section,
        tags=("ica", "eog", "ica-ocular-review", "eog-run-review"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    report.add_figure(
        fig=_plot_component_scores(run_reviews, excluded=ica.exclude),
        title=OCULAR_REPORT_TITLES[2],
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
    report.save(report_path, overwrite=True, open_browser=False)
    report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    return output_path


__all__ = ["OcularReviewSettings", "generate_ica_ocular_review", "ocular_evidence_table"]
