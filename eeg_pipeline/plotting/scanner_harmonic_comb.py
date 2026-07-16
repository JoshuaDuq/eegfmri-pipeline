"""Publication-quality cohort scanner-harmonic comb output."""

from __future__ import annotations

import re
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from eeg_pipeline.analysis.qc.scanner_harmonic_comb import ScannerCombSummary
from eeg_pipeline.analysis.qc.scanner_harmonics import DEFAULT_HARMONIC_WINDOWS

FIGURE_DPI = 300
INPUT_COLOR = "#6B6B6B"
FINAL_COLOR = "#0072B2"
WINDOW_COLOR = "#BDBDBD"
REFERENCE_COLOR = "#777777"
SCANNER_REFERENCE_FREQUENCIES_HZ = (
    20.01953125,
    41.1376953125,
    61.09619140625,
    82.21435546875,
)
SCANNER_COMB_COLUMNS = (
    "frequency_hz",
    "input_median_db",
    "input_ci_low_db",
    "input_ci_high_db",
    "final_median_db",
    "final_ci_low_db",
    "final_ci_high_db",
    "n_participants",
)


def build_scanner_harmonic_comb_figure(
    summary: ScannerCombSummary,
    *,
    task: str,
) -> Figure:
    """Build full and harmonic-local MNE scanner-comb cohort panels."""
    task_label = _validate_task(task)
    _summary_frame(summary)

    figure = Figure(figsize=(13.0, 11.0), layout="constrained", facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.15, 1.0, 1.0))
    axes = (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    )

    _plot_full_comb(axes[0], summary)
    local_limits = _shared_local_power_limits(summary)
    for index, axis in enumerate(axes[1:]):
        _plot_local_comb(
            axis,
            summary,
            index=index,
            y_limits=local_limits,
        )
    for axis, label in zip(axes, ("A", "B", "C", "D", "E"), strict=True):
        _style_axis(axis, label)

    figure.suptitle(
        f"Task-{task_label} MNE preprocessing | Cohort scanner-gradient spectral QC\n"
        f"Participant-first median across {summary.participant_count} participants | "
        "MRI-corrected input vs final cleaned epochs",
        fontsize=15,
        fontweight="bold",
    )
    return figure


def _plot_full_comb(axis, summary: ScannerCombSummary) -> None:
    _plot_stage_spectra(axis, summary, low_hz=15.0, high_hz=90.0)

    for window, frequency in zip(
        DEFAULT_HARMONIC_WINDOWS,
        SCANNER_REFERENCE_FREQUENCIES_HZ,
        strict=True,
    ):
        axis.axvspan(
            window.low_hz,
            window.high_hz,
            color=WINDOW_COLOR,
            alpha=0.12,
            linewidth=0,
        )
        axis.axvline(
            frequency,
            color=REFERENCE_COLOR,
            linestyle=":",
            linewidth=0.8,
        )

    axis.set(
        xlim=(15.0, 90.0),
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB V²/Hz)",
        title="Full scanner-harmonic comb",
    )
    axis.legend(loc="upper right", frameon=False, ncols=2, fontsize=8)


def _plot_local_comb(
    axis,
    summary: ScannerCombSummary,
    *,
    index: int,
    y_limits: tuple[float, float],
) -> None:
    window = DEFAULT_HARMONIC_WINDOWS[index]
    reference_frequency = SCANNER_REFERENCE_FREQUENCIES_HZ[index]
    reference_index = int(
        np.argmin(np.abs(summary.frequencies_hz - reference_frequency))
    )
    attenuation = float(
        summary.input_median_db[reference_index]
        - summary.final_median_db[reference_index]
    )
    distance = np.abs(summary.frequencies_hz - reference_frequency)
    background = (distance >= 0.35) & (distance <= 2.0)
    final_prominence = float(
        summary.final_median_db[reference_index]
        - np.median(summary.final_median_db[background])
    )

    _plot_stage_spectra(
        axis,
        summary,
        low_hz=window.low_hz,
        high_hz=window.high_hz,
    )
    axis.axvline(
        reference_frequency,
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=0.8,
    )
    axis.text(
        0.02,
        0.96,
        f"MNE attenuation {attenuation:.1f} dB\n"
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
        title=f"{reference_frequency:.1f} Hz input reference",
    )


def _plot_stage_spectra(
    axis,
    summary: ScannerCombSummary,
    *,
    low_hz: float,
    high_hz: float,
) -> None:
    mask = (summary.frequencies_hz >= low_hz) & (
        summary.frequencies_hz <= high_hz
    )
    _plot_stage(
        axis,
        summary.frequencies_hz[mask],
        summary.input_median_db[mask],
        summary.input_ci_low_db[mask],
        summary.input_ci_high_db[mask],
        color=INPUT_COLOR,
        label="MRI-corrected BIDS input",
    )
    _plot_stage(
        axis,
        summary.frequencies_hz[mask],
        summary.final_median_db[mask],
        summary.final_ci_low_db[mask],
        summary.final_ci_high_db[mask],
        color=FINAL_COLOR,
        label="Final MNE-cleaned epochs",
    )


def _shared_local_power_limits(summary: ScannerCombSummary) -> tuple[float, float]:
    selected_power = []
    for window in DEFAULT_HARMONIC_WINDOWS:
        mask = (summary.frequencies_hz >= window.low_hz) & (
            summary.frequencies_hz <= window.high_hz
        )
        selected_power.extend(
            (
                summary.input_ci_low_db[mask],
                summary.input_ci_high_db[mask],
                summary.final_ci_low_db[mask],
                summary.final_ci_high_db[mask],
            )
        )
    values = np.concatenate(selected_power)
    padding = max(1.0, 0.05 * float(np.ptp(values)))
    return float(np.min(values) - padding), float(np.max(values) + padding)


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


def _plot_stage(
    axis,
    frequencies_hz: np.ndarray,
    median_db: np.ndarray,
    ci_low_db: np.ndarray,
    ci_high_db: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    axis.fill_between(
        frequencies_hz,
        ci_low_db,
        ci_high_db,
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(
        frequencies_hz,
        median_db,
        color=color,
        linewidth=1.6,
        label=label,
    )


def write_scanner_harmonic_comb(
    summary: ScannerCombSummary,
    *,
    output_dir: Path,
    task: str,
) -> tuple[Path, Path]:
    """Write the cohort comb PNG and its matching numerical TSV."""
    task_label = _validate_task(task)
    frame = _summary_frame(summary)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"task-{task_label}_desc-scannerharmoniccomb_qc"
    png_path = output_path / f"{stem}.png"
    tsv_path = output_path / f"{stem}.tsv"

    frame.to_csv(tsv_path, sep="\t", index=False)
    figure = build_scanner_harmonic_comb_figure(summary, task=task_label)
    figure.savefig(png_path, dpi=FIGURE_DPI, facecolor="white")
    figure.clear()
    return png_path, tsv_path


def _summary_frame(summary: ScannerCombSummary) -> pd.DataFrame:
    frequencies = np.asarray(summary.frequencies_hz, dtype=float)
    arrays = {
        "input_median_db": summary.input_median_db,
        "input_ci_low_db": summary.input_ci_low_db,
        "input_ci_high_db": summary.input_ci_high_db,
        "final_median_db": summary.final_median_db,
        "final_ci_low_db": summary.final_ci_low_db,
        "final_ci_high_db": summary.final_ci_high_db,
    }
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("Scanner-comb summary requires a one-dimensional frequency grid.")
    columns: dict[str, np.ndarray] = {"frequency_hz": frequencies}
    for name, values in arrays.items():
        array = np.asarray(values, dtype=float)
        if array.shape != frequencies.shape:
            raise ValueError(f"{name} must match the scanner-comb frequency grid.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        columns[name] = array
    if summary.participant_count < 1:
        raise ValueError("Scanner-comb summary requires at least one participant.")
    if len(summary.harmonic_frequencies_hz) != len(DEFAULT_HARMONIC_WINDOWS):
        raise ValueError("Scanner-comb summary must contain four harmonic references.")
    columns["n_participants"] = np.full(frequencies.size, summary.participant_count)
    return pd.DataFrame(columns, columns=SCANNER_COMB_COLUMNS)


def _validate_task(task: str) -> str:
    task_label = str(task).strip()
    if not task_label:
        raise ValueError("task must be non-empty.")
    if re.fullmatch(r"[A-Za-z0-9]+", task_label) is None:
        raise ValueError("task must be a valid alphanumeric BIDS label.")
    return task_label


__all__ = [
    "SCANNER_COMB_COLUMNS",
    "build_scanner_harmonic_comb_figure",
    "write_scanner_harmonic_comb",
]
