"""MNE report rendering for direct ECG and ICA cardiac review."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.ica_cardiac_review import (
    CardiacReviewSettings,
    ComponentCardiacReview,
    RunCardiacReview,
    _build_component_cardiac_review,
    _build_run_cardiac_review,
    component_cardiac_evidence_table,
    component_run_cardiac_evidence_table,
)

CARDIAC_REPORT_TITLES = (
    "How to review ECG artifacts",
    "ECG detection summary",
    "ECG detection and provisional correction by run",
    "ICA components: R-locked cardiac evidence",
)


def _plot_run_cardiac_review(
    review: RunCardiacReview,
    *,
    ica: mne.preprocessing.ICA,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplot_mosaic(
        [["ecg", "ecg", "heart_rate"], ["gfp", "before", "after"]],
        figsize=(15.0, 7.2),
        layout="constrained",
    )
    axes["ecg"].plot(
        review.representative_times,
        review.representative_ecg_mv,
        color="#343A40",
    )
    for peak_time in review.representative_peak_times:
        axes["ecg"].axvline(peak_time, color="#C23B22", alpha=0.65, linewidth=1.0)
    axes["ecg"].set(
        title="Representative ECG with signal-detected R peaks",
        xlabel="Recording time (s)",
        ylabel="ECG (mV)",
    )

    heart_rate_axis = axes["heart_rate"]
    heart_rate_axis.plot(
        review.rr_times,
        review.heart_rate_bpm,
        color="#6B4C9A",
        linewidth=0.8,
        alpha=0.75,
    )
    heart_rate_axis.scatter(review.rr_times, review.heart_rate_bpm, color="#6B4C9A", s=9)
    heart_rate_axis.axhline(
        np.median(review.heart_rate_bpm),
        color="0.35",
        linestyle="--",
        linewidth=1.0,
    )
    heart_rate_axis.set(
        title="Beat-to-beat heart rate",
        xlabel="Recording time (s)",
        ylabel="Heart rate (bpm)",
    )

    axes["gfp"].plot(
        review.locked_times,
        review.before_gfp_uv,
        color="#B24C3B",
        label="Before ICA",
    )
    axes["gfp"].plot(
        review.locked_times,
        review.after_gfp_uv,
        color="#276B8A",
        label="After ICA",
    )
    axes["gfp"].axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
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
        f"MNE average pulse {review.average_pulse_bpm:.1f} bpm"
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
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
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
    lower, upper = np.quantile(run_means, [0.16, 0.84], axis=0)
    for run_mean in run_means:
        axes[1].plot(review.times, run_mean, color="#7FA8B8", alpha=0.35, linewidth=0.8)
    axes[1].plot(review.times, median, color="#276B8A", linewidth=1.8, label="ICA median")
    axes[1].fill_between(review.times, lower, upper, color="#276B8A", alpha=0.18)
    axes[1].axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].set(
        title=f"R-locked ICA waveform and ECG timing ({len(review.run_ids)} runs)",
        xlabel="Time from R peak (s)",
        ylabel="Baseline-standardized amplitude (z)",
    )
    ecg_axis = axes[1].twinx()
    ecg_axis.plot(
        review.times,
        np.median(review.run_ecg_z, axis=0),
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label="ECG median",
    )
    ecg_axis.set_ylabel("Normalized ECG", color="0.35")
    source_handles, source_labels = axes[1].get_legend_handles_labels()
    ecg_handles, ecg_labels = ecg_axis.get_legend_handles_labels()
    axes[1].legend(
        source_handles + ecg_handles,
        source_labels + ecg_labels,
        frameon=False,
        fontsize=8,
    )

    correlation = review.correlation_scores[:, component]
    ctps = review.ctps_scores[:, component]
    run_positions = np.linspace(-0.16, 0.16, len(review.run_ids))
    run_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.7, len(review.run_ids)))
    for run_index, (recording_id, offset) in enumerate(
        zip(review.run_ids, run_positions, strict=True)
    ):
        short_id = recording_id.rsplit("_", maxsplit=1)[-1]
        axes[2].scatter(
            offset,
            correlation[run_index],
            color=run_colors[run_index],
            edgecolor="white",
            linewidth=0.8,
            s=32,
            label=short_id,
        )
        axes[2].scatter(
            1.0 + offset,
            ctps[run_index],
            color=run_colors[run_index],
            edgecolor="white",
            linewidth=0.8,
            s=32,
        )
        if review.correlation_flags[run_index, component]:
            axes[2].scatter(offset, correlation[run_index], color="#C23B22", marker="x", s=52)
        if review.ctps_flags[run_index, component]:
            axes[2].scatter(
                1.0 + offset,
                ctps[run_index],
                color="#C23B22",
                marker="x",
                s=52,
            )
    lower_limit = min(-0.3, 1.15 * float(correlation.min()))
    upper_limit = max(0.3, 1.15 * float(max(correlation.max(), ctps.max())))
    axes[2].set(
        title="MNE find_bads_ecg scores by run",
        ylabel="Score",
        xticks=[0, 1],
        xticklabels=["ECG correlation", "CTPS"],
        xlim=(-0.35, 1.35),
        ylim=(lower_limit, upper_limit),
    )
    score_handles, score_labels = axes[2].get_legend_handles_labels()
    score_handles.append(
        Line2D([], [], color="#C23B22", marker="x", linestyle="none", label="MNE flag")
    )
    score_labels.append("MNE flag")
    axes[2].legend(score_handles, score_labels, frameon=False, fontsize=7, ncol=2)
    for axis in axes[1:]:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
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
    if set(cardiac_indices) != set(CARDIAC_REPORT_TITLES):
        raise ValueError("ICA cardiac-review content does not match the required report entries.")
    return [cardiac_indices[title] for title in CARDIAC_REPORT_TITLES]


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


def _cardiac_review_guide_html(settings: CardiacReviewSettings) -> str:
    ctps_threshold = str(settings.ctps_threshold)
    return (
        "<p><strong>Manual ECG review; no components are excluded here.</strong> "
        "R peaks are detected directly from the configured ECG signal, so this review does "
        "not depend on BrainVision Analyzer R markers.</p>"
        "<p>Inspect the detected peaks, beat-to-beat heart rate, R-locked EEG, component "
        "topography, and R-locked component waveform directly. Correlation and CTPS scores "
        "and red × markers come from MNE <code>find_bads_ecg</code>. They are displayed "
        "without additional pipeline classification or recommendation.</p>"
        f"<p>R-locked epoch: {settings.epoch_window[0]:g} to "
        f"{settings.epoch_window[1]:g} s; baseline: {settings.baseline[0]:g} to "
        f"{settings.baseline[1]:g} s; MNE CTPS threshold: {ctps_threshold}. "
        "All runs contribute to the component displays.</p>"
    )


def _component_statuses(path: Path, *, component_count: int) -> pd.DataFrame:
    statuses = pd.read_csv(path, sep="\t")
    required = {"component", "status", "status_description"}
    if not required.issubset(statuses.columns) or not np.array_equal(
        statuses["component"].to_numpy(), np.arange(component_count)
    ):
        raise ValueError(f"ICA component status table is invalid: {path}")
    statuses = statuses.copy()
    statuses["status_description"] = statuses["status_description"].fillna("")
    return statuses


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


def _cardiac_sidecar(output_path: Path, suffix: str) -> Path:
    expected_suffix = "_components.tsv"
    if not output_path.name.endswith(expected_suffix):
        raise ValueError(f"ECG component output must end with {expected_suffix!r}.")
    return output_path.with_name(output_path.name.removesuffix(expected_suffix) + f"_{suffix}.tsv")


def generate_ica_cardiac_review(
    *,
    filtered_raw_paths: list[Path],
    ica_path: Path,
    report_path: Path,
    output_path: Path,
    settings: CardiacReviewSettings,
) -> Path:
    """Append direct ECG diagnostics and review-only ICA evidence to an MNE report."""
    if not settings.enabled:
        raise ValueError("generate_ica_cardiac_review requires cardiac_review.enabled=true.")
    if not filtered_raw_paths:
        raise ValueError("No filtered raw recordings were provided for ECG review.")
    ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
    raws = [mne.io.read_raw_fif(path, preload=True, verbose="ERROR") for path in filtered_raw_paths]
    run_reviews = [
        _build_run_cardiac_review(
            raw,
            ica=ica,
            recording_id=path.name.removesuffix("_proc-filt_raw.fif"),
            settings=settings,
        )
        for path, raw in zip(filtered_raw_paths, raws, strict=True)
    ]
    component_review = _build_component_cardiac_review(
        raws,
        run_reviews,
        ica=ica,
        settings=settings,
    )
    component_status_path = ica_path.with_name(
        ica_path.name.replace("_proc-ica_ica.fif", "_proc-ica_components.tsv")
    )
    statuses = _component_statuses(
        component_status_path,
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

    report = mne.open_report(report_path)
    _clear_cardiac_review(report)
    section = "ICA cardiac artifact review"
    report.add_html(
        html=_cardiac_review_guide_html(settings),
        title="How to review ECG artifacts",
        section=section,
        tags=("ica", "ecg", "ica-cardiac-review"),
        replace=True,
    )
    report.add_html(
        html=run_table.to_html(
            index=False,
            float_format=lambda value: f"{value:.3f}",
            border=0,
            classes="table table-striped table-sm",
        ),
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
        replace=True,
    )
    _organize_cardiac_review(report)
    report.save(report_path, overwrite=True, open_browser=False)
    report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    return output_path


__all__ = ["generate_ica_cardiac_review"]
