"""Publication-oriented quality-control figures for native EEG-fMRI correction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np

from eeg_pipeline.preprocessing.eeg_fmri.pipeline import NativeCorrectionResult

FIGURE_DPI = 300
HARMONIC_PREFIXES = (
    "harmonic_18_23",
    "harmonic_38_43",
    "harmonic_56_67",
    "harmonic_77_85",
)
HARMONIC_LABELS = ("~20 Hz", "~41 Hz", "~61 Hz", "~82 Hz")
STAGE_COLORS = {
    "Raw": "#6B6B6B",
    "Gradient-corrected": "#D55E00",
    "Final": "#0072B2",
}
BEFORE_COLOR = "#D55E00"
AFTER_COLOR = "#0072B2"


def _style_axis(axis, panel_label: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.text(
        -0.12,
        1.08,
        panel_label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def _representative_samples(n_samples: int, sampling_frequency_hz: float) -> slice:
    window_samples = min(n_samples, int(round(20.0 * sampling_frequency_hz)))
    start = (n_samples - window_samples) // 2
    return slice(start, start + window_samples)


def _plot_qrs_window(axis, result: NativeCorrectionResult) -> None:
    diagnostics = result.qrs.diagnostics
    if diagnostics.filtered_ecg.shape != diagnostics.probabilities.shape:
        raise ValueError("ECG and NeuXus probability timelines must have equal length")
    samples = _representative_samples(
        diagnostics.filtered_ecg.size,
        diagnostics.sampling_frequency_hz,
    )
    sample_indices = np.arange(diagnostics.filtered_ecg.size)[samples]
    times = sample_indices / diagnostics.sampling_frequency_hz
    peak_mask = (diagnostics.peak_samples >= samples.start) & (
        diagnostics.peak_samples < samples.stop
    )
    peak_samples = diagnostics.peak_samples[peak_mask]

    ecg_line = axis.plot(
        times,
        1_000.0 * diagnostics.filtered_ecg[samples],
        color="#222222",
        linewidth=0.8,
        label="Filtered ECG",
    )[0]
    peak_points = axis.scatter(
        peak_samples / diagnostics.sampling_frequency_hz,
        1_000.0 * diagnostics.filtered_ecg[peak_samples],
        color=BEFORE_COLOR,
        edgecolor="white",
        linewidth=0.4,
        s=22,
        zorder=3,
        label="Accepted R peak",
    )
    probability_axis = axis.twinx()
    probability_line = probability_axis.plot(
        times,
        diagnostics.probabilities[samples],
        color=AFTER_COLOR,
        linewidth=0.7,
        alpha=0.7,
        label="NeuXus probability",
    )[0]
    probability_axis.set(ylabel="R-peak probability", ylim=(-0.02, 1.02))
    probability_axis.spines["top"].set_visible(False)
    axis.set(
        xlabel="Time (s)",
        ylabel="Filtered ECG (mV)",
        title="Representative 20-second QRS detection window",
    )
    axis.legend(
        [ecg_line, peak_points, probability_line],
        ["Filtered ECG", "Accepted R peak", "NeuXus probability"],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )


def _plot_rr_timeline(axis, result: NativeCorrectionResult) -> None:
    rr_intervals = np.diff(result.qrs.times)
    axis.scatter(
        result.qrs.times[1:] / 60.0,
        rr_intervals,
        color="#222222",
        s=9,
        alpha=0.75,
        linewidth=0,
    )
    axis.axhline(0.375, color=BEFORE_COLOR, linestyle="--", linewidth=1.0)
    axis.axhline(1.5, color=BEFORE_COLOR, linestyle="--", linewidth=1.0)
    axis.axhline(
        result.qrs.quality.median_rr_seconds,
        color=AFTER_COLOR,
        linewidth=1.2,
        label=f"Median {result.qrs.quality.median_rr_seconds:.3f} s",
    )
    axis.set_yscale("log")
    upper_limit = max(2.0, 1.15 * float(np.max(rr_intervals)))
    ticks = [tick for tick in (0.375, 0.5, 1.0, 1.5, 3.0, 6.0, 10.0) if tick < upper_limit]
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.set(
        xlabel="Recording time (min)",
        ylabel="RR interval (s, log scale)",
        ylim=(0.3, upper_limit),
        title=(
            f"RR quality: {result.qrs.quality.qrs_count} peaks, "
            f"{result.qrs.quality.abnormal_rr_count} warning intervals"
        ),
    )
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def _plot_cardiac_locked(axis, result: NativeCorrectionResult) -> None:
    before = result.cardiac_qc.before
    after = result.cardiac_qc.after
    before_rms = 1e6 * np.sqrt(np.mean(before.median_evoked**2, axis=0))
    after_rms = 1e6 * np.sqrt(np.mean(after.median_evoked**2, axis=0))
    axis.plot(before.times, before_rms, label="Before OBS", color=BEFORE_COLOR, linewidth=1.8)
    axis.plot(after.times, after_rms, label="After OBS", color=AFTER_COLOR, linewidth=1.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle=":")
    axis.set(
        xlabel="Time from R peak (s)",
        ylabel="Across-channel RMS (µV)",
        title=(
            f"Cardiac-locked EEG: {result.cardiac_qc.rms_attenuation_db:.1f} dB " "RMS attenuation"
        ),
    )
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def _plot_scanner_lines(axis, result: NativeCorrectionResult) -> None:
    stages = result.harmonic_stages
    raw = stages["raw"]
    stage_values = {
        "Raw": [float(raw[f"{prefix}_reference_power_db"]) for prefix in HARMONIC_PREFIXES],
        "Gradient-corrected": [
            float(stages["gradient_corrected"][f"{prefix}_reference_power_db"])
            for prefix in HARMONIC_PREFIXES
        ],
        "Final": [
            float(stages["final"][f"{prefix}_reference_power_db"]) for prefix in HARMONIC_PREFIXES
        ],
    }
    frequencies = [float(raw[f"{prefix}_reference_hz"]) for prefix in HARMONIC_PREFIXES]
    labels = [f"{frequency:.1f} Hz" for frequency in frequencies]
    locations = np.arange(len(HARMONIC_PREFIXES), dtype=float)
    width = 0.24
    for stage_index, (stage, values) in enumerate(stage_values.items()):
        axis.bar(
            locations + (stage_index - 1) * width,
            values,
            width=width,
            color=STAGE_COLORS[stage],
            label=stage,
        )
    axis.set(
        xticks=locations,
        xticklabels=labels,
        ylabel="Reference-line PSD (dB V²/Hz)",
        title="Scanner-harmonic power at matched frequencies",
    )
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def save_run_qc_figure(
    result: NativeCorrectionResult,
    path: str | Path,
    *,
    recording_label: str,
) -> None:
    """Save one deterministic four-panel run-level QC figure at 300 dpi."""
    figure = Figure(figsize=(13, 9), layout="constrained", facecolor="white")
    axes = figure.subplots(2, 2)
    _plot_qrs_window(axes[0, 0], result)
    _plot_rr_timeline(axes[0, 1], result)
    _plot_cardiac_locked(axes[1, 0], result)
    _plot_scanner_lines(axes[1, 1], result)
    for axis, label in zip(axes.flat, ("A", "B", "C", "D"), strict=True):
        _style_axis(axis, label)
    figure.suptitle(
        f"{recording_label} | Native EEG–fMRI MRI-artifact QC",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    figure.clear()


def _row_values(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Cohort QC field {key!r} contains non-finite values")
    return values


def _boxplot(
    axis, values: Sequence[np.ndarray], labels: Sequence[str], colors: Sequence[str]
) -> None:
    artists = axis.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for box, color in zip(artists["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    for index, samples in enumerate(values, start=1):
        offsets = np.linspace(-0.12, 0.12, samples.size) if samples.size > 1 else np.zeros(1)
        axis.scatter(index + offsets, samples, color="#222222", s=8, alpha=0.5, linewidth=0)


def save_cohort_qc_figure(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
) -> None:
    """Save the cohort correction summary used to qualify publication."""
    if not rows:
        raise ValueError("Cohort QC requires at least one completed run")
    harmonic_values = [
        _row_values(rows, f"{prefix}_raw_to_final_db") for prefix in HARMONIC_PREFIXES
    ]
    cardiac_values = [
        _row_values(rows, "cardiac_rms_attenuation_db"),
        _row_values(rows, "cardiac_peak_to_peak_attenuation_db"),
    ]
    heart_rates = _row_values(rows, "median_heart_rate_bpm")
    abnormal_fractions = 100.0 * _row_values(rows, "abnormal_rr_fraction")
    run_indices = np.arange(1, len(rows) + 1)

    figure = Figure(figsize=(13, 9), layout="constrained", facecolor="white")
    axes = figure.subplots(2, 2)
    _boxplot(
        axes[0, 0],
        harmonic_values,
        HARMONIC_LABELS,
        ("#56B4E9", "#0072B2", "#009E73", "#CC79A7"),
    )
    axes[0, 0].axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0, 0].set(
        ylabel="Raw-to-final attenuation (dB)",
        title="Scanner-harmonic attenuation",
    )

    _boxplot(
        axes[0, 1],
        cardiac_values,
        ("RMS", "Peak-to-peak"),
        (BEFORE_COLOR, AFTER_COLOR),
    )
    axes[0, 1].axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0, 1].set(ylabel="Attenuation (dB)", title="Cardiac-locked EEG attenuation")

    axes[1, 0].scatter(run_indices, heart_rates, color=AFTER_COLOR, s=16, linewidth=0)
    axes[1, 0].axhspan(40.0, 160.0, color="#009E73", alpha=0.08)
    axes[1, 0].set(
        xlabel="Cohort run index",
        ylabel="Median heart rate (bpm)",
        ylim=(35.0, 165.0),
        title="Automatic QRS physiological plausibility",
    )

    axes[1, 1].scatter(
        run_indices,
        abnormal_fractions,
        color=BEFORE_COLOR,
        s=16,
        linewidth=0,
    )
    axes[1, 1].set(
        xlabel="Cohort run index",
        ylabel="RR intervals outside limits (%)",
        title="Warning-level RR interval burden",
    )
    for axis, label in zip(axes.flat, ("A", "B", "C", "D"), strict=True):
        _style_axis(axis, label)
    figure.suptitle(
        f"Native EEG–fMRI correction cohort QC | {len(rows)} runs",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    figure.clear()
