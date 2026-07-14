"""Publication-oriented quality-control figures for native EEG-fMRI correction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np

from eeg_pipeline.analysis.qc.scanner_harmonics import DEFAULT_HARMONIC_WINDOWS
from eeg_pipeline.preprocessing.eeg_fmri.cohort_spectrum import (
    CohortScannerSpectra,
    CohortStageSpectrum,
)
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


def build_physiology_qc_figure(
    result: NativeCorrectionResult,
    *,
    recording_label: str,
) -> Figure:
    """Build the run-level QRS and cardiac-artifact QC figure."""
    figure = Figure(figsize=(13, 9), layout="constrained", facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    )
    _plot_qrs_window(axes[0], result)
    _plot_rr_timeline(axes[1], result)
    _plot_cardiac_locked(axes[2], result)
    for axis, label in zip(axes, ("A", "B", "C"), strict=True):
        _style_axis(axis, label)
    figure.suptitle(
        f"{recording_label} | Physiological EEG–fMRI artifact QC",
        fontsize=15,
        fontweight="bold",
    )
    return figure


def _stage_display() -> tuple[tuple[str, str], ...]:
    return (
        ("raw", "Raw"),
        ("gradient_corrected", "Gradient-corrected"),
        ("final", "Final"),
    )


def _frequency_mask(frequencies: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Scanner spectrum has insufficient bins in {low_hz:g}–{high_hz:g} Hz")
    return mask


def _shared_local_power_limits(result: NativeCorrectionResult) -> tuple[float, float]:
    selected_power = []
    for window in DEFAULT_HARMONIC_WINDOWS:
        for stage_key, _ in _stage_display():
            spectrum = result.harmonic_stages[stage_key].spectrum
            mask = _frequency_mask(
                spectrum.frequencies_hz,
                window.low_hz,
                window.high_hz,
            )
            selected_power.append(spectrum.median_power_db[mask])
    values = np.concatenate(selected_power)
    if not np.all(np.isfinite(values)):
        raise ValueError("Scanner spectra contain non-finite PSD values")
    padding = max(1.0, 0.05 * float(np.ptp(values)))
    return float(np.min(values) - padding), float(np.max(values) + padding)


def _plot_stage_spectra(
    axis,
    result: NativeCorrectionResult,
    *,
    low_hz: float,
    high_hz: float,
) -> None:
    for stage_key, stage_label in _stage_display():
        spectrum = result.harmonic_stages[stage_key].spectrum
        mask = _frequency_mask(spectrum.frequencies_hz, low_hz, high_hz)
        axis.plot(
            spectrum.frequencies_hz[mask],
            spectrum.median_power_db[mask],
            color=STAGE_COLORS[stage_label],
            linewidth=1.2 if stage_key != "final" else 1.6,
            label=stage_label,
        )


def _plot_full_scanner_spectrum(axis, result: NativeCorrectionResult) -> None:
    _plot_stage_spectra(axis, result, low_hz=15.0, high_hz=90.0)
    raw_summary = result.harmonic_stages["raw"].summary
    for window, prefix in zip(DEFAULT_HARMONIC_WINDOWS, HARMONIC_PREFIXES, strict=True):
        axis.axvspan(window.low_hz, window.high_hz, color="#BDBDBD", alpha=0.12, linewidth=0)
        axis.axvline(
            float(raw_summary[f"{prefix}_reference_hz"]),
            color="#777777",
            linestyle=":",
            linewidth=0.7,
        )
    axis.set(
        xlim=(15.0, 90.0),
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB V²/Hz)",
        title="Full scanner-harmonic comb",
    )
    axis.legend(loc="upper right", frameon=False, ncols=3, fontsize=8)


def _plot_local_scanner_spectrum(
    axis,
    result: NativeCorrectionResult,
    *,
    index: int,
    y_limits: tuple[float, float],
) -> None:
    window = DEFAULT_HARMONIC_WINDOWS[index]
    prefix = HARMONIC_PREFIXES[index]
    stages = result.harmonic_stages
    raw_summary = stages["raw"].summary
    final_summary = stages["final"].summary
    reference_frequency = float(raw_summary[f"{prefix}_reference_hz"])
    gradient_attenuation = float(raw_summary[f"{prefix}_reference_power_db"]) - float(
        stages["gradient_corrected"].summary[f"{prefix}_reference_power_db"]
    )
    final_prominence = float(final_summary[f"{prefix}_reference_local_prominence_db"])

    _plot_stage_spectra(
        axis,
        result,
        low_hz=window.low_hz,
        high_hz=window.high_hz,
    )
    axis.axvline(
        reference_frequency,
        color="#777777",
        linestyle=":",
        linewidth=0.8,
    )
    axis.text(
        0.02,
        0.96,
        f"AAS attenuation {gradient_attenuation:.1f} dB\n"
        f"Final prominence {final_prominence:.1f} dB",
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
    )
    axis.set(
        xlim=(window.low_hz, window.high_hz),
        ylim=y_limits,
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB V²/Hz)",
        title=f"{reference_frequency:.1f} Hz raw reference",
    )


def build_scanner_spectrum_qc_figure(
    result: NativeCorrectionResult,
    *,
    recording_label: str,
) -> Figure:
    """Build the run-level full and local scanner-harmonic PSD figure."""
    figure = Figure(figsize=(13, 11), layout="constrained", facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.15, 1.0, 1.0))
    axes = (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    )
    _plot_full_scanner_spectrum(axes[0], result)
    local_limits = _shared_local_power_limits(result)
    for index, axis in enumerate(axes[1:]):
        _plot_local_scanner_spectrum(
            axis,
            result,
            index=index,
            y_limits=local_limits,
        )
    for axis, label in zip(axes, ("A", "B", "C", "D", "E"), strict=True):
        _style_axis(axis, label)
    channel_count = int(result.harmonic_stages["raw"].summary["n_channels"])
    figure.suptitle(
        f"{recording_label} | Scanner-gradient spectral QC\n"
        f"Median across {channel_count} prespecified EEG channels",
        fontsize=15,
        fontweight="bold",
    )
    return figure


def _save_figure(figure: Figure, path: str | Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    figure.clear()


def save_physiology_qc_figure(
    result: NativeCorrectionResult,
    path: str | Path,
    *,
    recording_label: str,
) -> None:
    """Save the run-level physiological artifact QC figure at 300 dpi."""
    _save_figure(
        build_physiology_qc_figure(result, recording_label=recording_label),
        path,
    )


def save_scanner_spectrum_qc_figure(
    result: NativeCorrectionResult,
    path: str | Path,
    *,
    recording_label: str,
) -> None:
    """Save the run-level scanner-harmonic spectral QC figure at 300 dpi."""
    _save_figure(
        build_scanner_spectrum_qc_figure(result, recording_label=recording_label),
        path,
    )


def _cohort_stage_display(
    cohort: CohortScannerSpectra,
) -> tuple[tuple[str, str, CohortStageSpectrum], ...]:
    return (
        ("raw", "Raw", cohort.raw),
        ("gradient_corrected", "Gradient-corrected", cohort.gradient_corrected),
        ("final", "Final", cohort.final),
    )


def _plot_cohort_stage_spectra(
    axis,
    cohort: CohortScannerSpectra,
    *,
    low_hz: float,
    high_hz: float,
) -> None:
    for stage_key, stage_label, spectrum in _cohort_stage_display(cohort):
        mask = _frequency_mask(spectrum.frequencies_hz, low_hz, high_hz)
        color = STAGE_COLORS[stage_label]
        axis.fill_between(
            spectrum.frequencies_hz[mask],
            spectrum.confidence_low_power_db[mask],
            spectrum.confidence_high_power_db[mask],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            spectrum.frequencies_hz[mask],
            spectrum.median_power_db[mask],
            color=color,
            linewidth=1.2 if stage_key != "final" else 1.6,
            label=stage_label,
        )


def _cohort_reference_index(
    cohort: CohortScannerSpectra,
    index: int,
) -> int:
    window = DEFAULT_HARMONIC_WINDOWS[index]
    frequencies = cohort.raw.frequencies_hz
    mask = _frequency_mask(frequencies, window.low_hz, window.high_hz)
    selected_indices = np.flatnonzero(mask)
    return int(selected_indices[np.argmax(cohort.raw.median_power_db[mask])])


def _cohort_local_metrics(
    cohort: CohortScannerSpectra,
    index: int,
) -> tuple[float, float, float]:
    reference_index = _cohort_reference_index(cohort, index)
    reference_frequency = float(cohort.raw.frequencies_hz[reference_index])
    gradient_attenuation = float(
        cohort.raw.median_power_db[reference_index]
        - cohort.gradient_corrected.median_power_db[reference_index]
    )
    distance = np.abs(cohort.final.frequencies_hz - reference_frequency)
    background = (distance >= 0.35) & (distance <= 2.0)
    if np.count_nonzero(background) < 2:
        raise ValueError(f"Insufficient cohort PSD bins around {reference_frequency:g} Hz")
    final_prominence = float(
        cohort.final.median_power_db[reference_index]
        - np.median(cohort.final.median_power_db[background])
    )
    return reference_frequency, gradient_attenuation, final_prominence


def _cohort_local_power_limits(cohort: CohortScannerSpectra) -> tuple[float, float]:
    selected_power = []
    for window in DEFAULT_HARMONIC_WINDOWS:
        for _, _, spectrum in _cohort_stage_display(cohort):
            mask = _frequency_mask(
                spectrum.frequencies_hz,
                window.low_hz,
                window.high_hz,
            )
            selected_power.extend(
                (
                    spectrum.confidence_low_power_db[mask],
                    spectrum.confidence_high_power_db[mask],
                )
            )
    values = np.concatenate(selected_power)
    if not np.all(np.isfinite(values)):
        raise ValueError("Cohort scanner spectra contain non-finite PSD values")
    padding = max(1.0, 0.05 * float(np.ptp(values)))
    return float(np.min(values) - padding), float(np.max(values) + padding)


def _plot_cohort_full_scanner_spectrum(axis, cohort: CohortScannerSpectra) -> None:
    _plot_cohort_stage_spectra(axis, cohort, low_hz=15.0, high_hz=90.0)
    for index, window in enumerate(DEFAULT_HARMONIC_WINDOWS):
        reference_frequency, _, _ = _cohort_local_metrics(cohort, index)
        axis.axvspan(window.low_hz, window.high_hz, color="#BDBDBD", alpha=0.12, linewidth=0)
        axis.axvline(
            reference_frequency,
            color="#777777",
            linestyle=":",
            linewidth=0.7,
        )
    axis.set(
        xlim=(15.0, 90.0),
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB V²/Hz)",
        title="Full scanner-harmonic comb",
    )
    axis.legend(loc="upper right", frameon=False, ncols=3, fontsize=8)


def _plot_cohort_local_scanner_spectrum(
    axis,
    cohort: CohortScannerSpectra,
    *,
    index: int,
    y_limits: tuple[float, float],
) -> None:
    window = DEFAULT_HARMONIC_WINDOWS[index]
    reference_frequency, gradient_attenuation, final_prominence = _cohort_local_metrics(
        cohort,
        index,
    )
    _plot_cohort_stage_spectra(
        axis,
        cohort,
        low_hz=window.low_hz,
        high_hz=window.high_hz,
    )
    axis.axvline(reference_frequency, color="#777777", linestyle=":", linewidth=0.8)
    axis.text(
        0.02,
        0.96,
        f"AAS attenuation {gradient_attenuation:.1f} dB\n"
        f"Final prominence {final_prominence:.1f} dB",
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
    )
    axis.set(
        xlim=(window.low_hz, window.high_hz),
        ylim=y_limits,
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB V²/Hz)",
        title=f"{reference_frequency:.1f} Hz raw reference",
    )


def build_cohort_scanner_spectrum_qc_figure(cohort: CohortScannerSpectra) -> Figure:
    """Build the participant-first cohort analogue of the run spectral QC figure."""
    figure = Figure(figsize=(13, 11), layout="constrained", facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.15, 1.0, 1.0))
    axes = (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    )
    _plot_cohort_full_scanner_spectrum(axes[0], cohort)
    local_limits = _cohort_local_power_limits(cohort)
    for index, axis in enumerate(axes[1:]):
        _plot_cohort_local_scanner_spectrum(
            axis,
            cohort,
            index=index,
            y_limits=local_limits,
        )
    for axis, label in zip(axes, ("A", "B", "C", "D", "E"), strict=True):
        _style_axis(axis, label)
    figure.suptitle(
        "Native EEG–fMRI correction | Cohort scanner-gradient spectral QC\n"
        f"Participant-first median across {cohort.participant_count} participants | "
        f"{cohort.run_count} runs | {cohort.channel_count} prespecified EEG channels",
        fontsize=15,
        fontweight="bold",
    )
    return figure


def save_cohort_scanner_spectrum_qc_figure(
    cohort: CohortScannerSpectra,
    path: str | Path,
) -> None:
    """Save participant-first cohort scanner spectra at 300 dpi."""
    _save_figure(build_cohort_scanner_spectrum_qc_figure(cohort), path)


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


def build_cohort_qc_figure(rows: Sequence[Mapping[str, object]]) -> Figure:
    """Build the cohort correction summary used to qualify publication."""
    if not rows:
        raise ValueError("Cohort QC requires at least one completed run")
    harmonic_values = [
        _row_values(rows, f"{prefix}_raw_to_final_db") for prefix in HARMONIC_PREFIXES
    ]
    prominence_values = [
        _row_values(rows, f"{prefix}_final_prominence_db") for prefix in HARMONIC_PREFIXES
    ]
    cardiac_values = [
        _row_values(rows, "cardiac_rms_attenuation_db"),
        _row_values(rows, "cardiac_peak_to_peak_attenuation_db"),
    ]
    heart_rates = _row_values(rows, "median_heart_rate_bpm")
    abnormal_fractions = 100.0 * _row_values(rows, "abnormal_rr_fraction")

    figure = Figure(figsize=(13, 9), layout="constrained", facecolor="white")
    axes = figure.subplots(2, 2)
    harmonic_colors = ("#56B4E9", "#0072B2", "#009E73", "#CC79A7")
    _boxplot(
        axes[0, 0],
        harmonic_values,
        HARMONIC_LABELS,
        harmonic_colors,
    )
    axes[0, 0].axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0, 0].set(
        ylabel="Raw-to-final attenuation (dB)",
        title="Scanner-harmonic attenuation",
    )

    _boxplot(
        axes[0, 1],
        prominence_values,
        HARMONIC_LABELS,
        harmonic_colors,
    )
    axes[0, 1].axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0, 1].set(
        ylabel="Final local prominence (dB)",
        title="Residual scanner-line prominence",
    )

    _boxplot(
        axes[1, 0],
        cardiac_values,
        ("RMS", "Peak-to-peak"),
        (BEFORE_COLOR, AFTER_COLOR),
    )
    axes[1, 0].axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[1, 0].set(
        ylabel="Attenuation (dB)",
        title="Cardiac-locked EEG attenuation",
    )

    axes[1, 1].scatter(
        heart_rates,
        abnormal_fractions,
        color="#222222",
        edgecolor="white",
        linewidth=0.3,
        s=24,
        alpha=0.75,
    )
    axes[1, 1].axvspan(40.0, 160.0, color="#009E73", alpha=0.08, linewidth=0)
    axes[1, 1].set(
        xlabel="Median heart rate (bpm)",
        ylabel="RR intervals outside limits (%)",
        xlim=(35.0, 165.0),
        title="Automatic QRS quality",
    )
    for axis, label in zip(axes.flat, ("A", "B", "C", "D"), strict=True):
        _style_axis(axis, label)
    figure.suptitle(
        f"Native EEG–fMRI correction cohort QC | {len(rows)} runs",
        fontsize=15,
        fontweight="bold",
    )
    return figure


def save_cohort_qc_figure(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
) -> None:
    """Save the cohort correction summary at 300 dpi."""
    _save_figure(
        build_cohort_qc_figure(rows),
        path,
    )
