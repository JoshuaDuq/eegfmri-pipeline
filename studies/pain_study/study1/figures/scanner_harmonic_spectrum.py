"""Outcome-blind scanner-harmonic spectral QC for Study 1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import find_peaks

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    DEFAULT_HARMONIC_WINDOWS,
    FrequencyWindow,
)
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.scanner_contamination import SCANNER_CLEAN_GAMMA_RANGES_HZ
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

NUMBERED_SUBJECT_PATTERN = re.compile(r"^sub-\d+$")
FINAL_CLEAN_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_task-[^_]+_run-(?P<run>[^_]+)_proc-clean_raw\.fif$"
)
BOOTSTRAP_BATCH_SIZE = 256


@dataclass(frozen=True)
class ScannerHarmonicSpecification:
    """Fixed acquisition and peak-detection settings for the QC figure."""

    frequency_range_hz: tuple[float, float]
    n_fft: int
    n_overlap: int
    sampling_frequency_hz: float
    peak_prominence_db: float
    peak_distance_bins: int
    volume_repetition_time_s: float
    harmonic_orders: tuple[int, ...]
    excluded_subjects: tuple[str, ...]
    harmonic_windows: tuple[FrequencyWindow, ...] = DEFAULT_HARMONIC_WINDOWS


@dataclass(frozen=True)
class HarmonicPeak:
    """Strongest qualifying peak in one prespecified frequency window."""

    window_name: str
    peak_frequency_hz: float
    prominence_db: float


@dataclass(frozen=True)
class RunSpectrum:
    """One final-clean run reduced to a robust channel-median spectrum."""

    subject_id: str
    run_id: str
    source_file: Path
    frequencies_hz: np.ndarray
    median_psd_db: np.ndarray
    n_channels: int
    sampling_frequency_hz: float
    n_samples: int
    peaks: tuple[HarmonicPeak, ...]


@dataclass(frozen=True)
class ParticipantBootstrapSpecification:
    """Participant-level percentile-bootstrap settings."""

    iterations: int
    confidence_level: float
    seed: int


@dataclass(frozen=True)
class ScannerHarmonicSummary:
    """Participant-first spectra, peak offsets, and reproducibility audits."""

    participant_spectra: pd.DataFrame
    cohort_spectrum: pd.DataFrame
    participant_offsets: pd.DataFrame
    cohort_offsets: pd.DataFrame
    run_audit: pd.DataFrame
    participant_audit: pd.DataFrame

    @property
    def n_subjects(self) -> int:
        return int(self.participant_audit["subject_id"].nunique())

    @property
    def n_runs(self) -> int:
        return int(len(self.run_audit))


def scanner_harmonic_specification(config: Any) -> ScannerHarmonicSpecification:
    """Load the fixed scanner-harmonic analysis settings."""
    scanner = require_config_value(config, "study1.figures.scanner_harmonics")
    frequency_range = tuple(float(value) for value in scanner["frequency_range_hz"])
    harmonic_orders = tuple(int(value) for value in scanner["harmonic_orders"])
    if len(frequency_range) != 2:
        raise ValueError("Scanner-harmonic frequency_range_hz must contain two values.")
    if len(harmonic_orders) != len(DEFAULT_HARMONIC_WINDOWS):
        raise ValueError(
            "Scanner-harmonic order count must match the fixed harmonic-window count."
        )
    return ScannerHarmonicSpecification(
        frequency_range_hz=(frequency_range[0], frequency_range[1]),
        n_fft=int(scanner["n_fft"]),
        n_overlap=int(scanner["n_overlap"]),
        sampling_frequency_hz=float(scanner["sampling_frequency_hz"]),
        peak_prominence_db=float(scanner["peak_prominence_db"]),
        peak_distance_bins=int(scanner["peak_distance_bins"]),
        volume_repetition_time_s=float(scanner["volume_repetition_time_s"]),
        harmonic_orders=harmonic_orders,
        excluded_subjects=tuple(str(value) for value in scanner["excluded_subjects"]),
    )


def validity_bootstrap_specification(config: Any) -> ParticipantBootstrapSpecification:
    """Load participant-bootstrap settings shared by Study 1 validity figures."""
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    return ParticipantBootstrapSpecification(
        iterations=int(bootstrap["iterations"]),
        confidence_level=float(bootstrap["confidence_level"]),
        seed=int(bootstrap["seed"]),
    )


def discover_final_clean_runs(
    derivative_root: Path,
    *,
    task: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Discover final-clean FIF runs for numbered, eligible participants."""
    excluded = set(excluded_subjects)
    requested = set(requested_subjects)
    candidates = Path(derivative_root).glob(
        f"sub-*/eeg/sub-*_task-{task}_run-*_proc-clean_raw.fif"
    )
    selected: list[Path] = []
    for path in candidates:
        subject = path.parts[-3]
        if NUMBERED_SUBJECT_PATTERN.fullmatch(subject) is None or subject in excluded:
            continue
        if requested and subject not in requested:
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            "No numbered-participant final-clean FIF files found for task "
            f"{task!r} in {derivative_root}."
        )
    return tuple(sorted(selected))


def select_scanner_harmonic_peaks(
    frequencies: np.ndarray,
    spectrum_db: np.ndarray,
    specification: ScannerHarmonicSpecification,
) -> tuple[HarmonicPeak, ...]:
    """Select the greatest-prominence eligible peak from each fixed window."""
    frequency_values = np.asarray(frequencies, dtype=float)
    spectrum_values = np.asarray(spectrum_db, dtype=float)
    if frequency_values.ndim != 1 or spectrum_values.shape != frequency_values.shape:
        raise ValueError("Scanner-harmonic frequencies and spectrum must be aligned vectors.")
    peak_indices, properties = find_peaks(
        spectrum_values,
        prominence=specification.peak_prominence_db,
        distance=specification.peak_distance_bins,
    )
    peak_frequencies = frequency_values[peak_indices]
    prominences = properties["prominences"]

    selected: list[HarmonicPeak] = []
    for window in specification.harmonic_windows:
        eligible = np.flatnonzero(
            (peak_frequencies >= window.low_hz)
            & (peak_frequencies <= window.high_hz)
        )
        if eligible.size == 0:
            raise ValueError(
                "No qualifying spectral peak in scanner-harmonic window "
                f"{window.label} Hz."
            )
        strongest = int(eligible[np.argmax(prominences[eligible])])
        selected.append(
            HarmonicPeak(
                window_name=window.name,
                peak_frequency_hz=float(peak_frequencies[strongest]),
                prominence_db=float(prominences[strongest]),
            )
        )
    return tuple(selected)


def estimate_run_spectrum(
    path: Path,
    specification: ScannerHarmonicSpecification,
) -> RunSpectrum:
    """Estimate the robust Welch spectrum and scanner peaks for one run."""
    import mne

    source_path = Path(path)
    entities = FINAL_CLEAN_FILENAME_PATTERN.fullmatch(source_path.name)
    if entities is None:
        raise ValueError(f"Invalid final-clean EEG filename: {source_path.name}")

    raw = mne.io.read_raw_fif(source_path, preload=False, verbose="ERROR")
    sampling_frequency = float(raw.info["sfreq"])
    if sampling_frequency != specification.sampling_frequency_hz:
        raise ValueError(
            f"Unexpected sampling frequency in {source_path}: {sampling_frequency} Hz."
        )
    lower_frequency, upper_frequency = specification.frequency_range_hz
    spectrum = raw.compute_psd(
        method="welch",
        fmin=lower_frequency,
        fmax=upper_frequency,
        n_fft=specification.n_fft,
        n_per_seg=specification.n_fft,
        n_overlap=specification.n_overlap,
        picks="eeg",
        verbose=False,
    )
    frequencies = np.asarray(spectrum.freqs, dtype=float)
    channel_psd = np.asarray(spectrum.get_data(), dtype=float)
    if channel_psd.ndim != 2 or channel_psd.shape[1] != frequencies.size:
        raise ValueError(f"Unexpected PSD shape for {source_path}: {channel_psd.shape}.")
    median_psd = np.median(channel_psd, axis=0)
    if not np.isfinite(median_psd).all() or np.any(median_psd <= 0.0):
        raise ValueError(f"PSD contains nonpositive or non-finite values: {source_path}")
    median_psd_db = 10.0 * np.log10(median_psd)
    return RunSpectrum(
        subject_id=entities.group("subject"),
        run_id=entities.group("run"),
        source_file=source_path,
        frequencies_hz=frequencies,
        median_psd_db=median_psd_db,
        n_channels=int(channel_psd.shape[0]),
        sampling_frequency_hz=sampling_frequency,
        n_samples=int(raw.n_times),
        peaks=select_scanner_harmonic_peaks(
            frequencies,
            median_psd_db,
            specification,
        ),
    )


def paired_participant_bootstrap(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cohort medians and paired participant-bootstrap intervals."""
    value_matrix = np.asarray(values, dtype=float)
    if value_matrix.ndim != 2 or value_matrix.shape[0] < 1:
        raise ValueError("Participant bootstrap requires a non-empty two-dimensional matrix.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        value_matrix.shape[0],
        size=(iterations, value_matrix.shape[0]),
    )
    estimates = np.empty((iterations, value_matrix.shape[1]), dtype=float)
    for start in range(0, iterations, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, iterations)
        estimates[start:stop] = np.median(value_matrix[indices[start:stop]], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        np.median(value_matrix, axis=0),
        np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )


def build_scanner_harmonic_summary(
    run_spectra: Sequence[RunSpectrum],
    specification: ScannerHarmonicSpecification,
    *,
    bootstrap: ParticipantBootstrapSpecification,
) -> ScannerHarmonicSummary:
    """Aggregate runs within participants before estimating cohort quantities."""
    runs = tuple(run_spectra)
    if not runs:
        raise ValueError("Scanner-harmonic summary requires at least one run spectrum.")
    frequencies = runs[0].frequencies_hz
    if any(not np.array_equal(run.frequencies_hz, frequencies) for run in runs[1:]):
        raise ValueError("Scanner-harmonic run spectra use inconsistent frequency bins.")

    run_audit = _run_audit_table(runs)
    subjects = sorted({run.subject_id for run in runs})
    participant_spectrum_rows: list[dict[str, float | str]] = []
    participant_audit_rows: list[dict[str, float | int | str]] = []
    participant_offset_rows: list[dict[str, float | int | str]] = []
    participant_spectrum_matrix = np.empty((len(subjects), frequencies.size), dtype=float)
    participant_offset_matrix = np.empty(
        (len(subjects), len(specification.harmonic_windows)),
        dtype=float,
    )

    predicted_frequencies = (
        np.asarray(specification.harmonic_orders, dtype=float)
        / specification.volume_repetition_time_s
    )
    for subject_index, subject_id in enumerate(subjects):
        subject_runs = tuple(run for run in runs if run.subject_id == subject_id)
        participant_spectrum = np.median(
            np.stack([run.median_psd_db for run in subject_runs]),
            axis=0,
        )
        relative_spectrum = participant_spectrum - np.median(participant_spectrum)
        participant_spectrum_matrix[subject_index] = relative_spectrum
        participant_spectrum_rows.extend(
            {
                "subject_id": subject_id,
                "frequency_hz": float(frequency),
                "relative_psd_db": float(value),
            }
            for frequency, value in zip(frequencies, relative_spectrum, strict=True)
        )

        audit_row: dict[str, float | int | str] = {
            "subject_id": subject_id,
            "n_runs": len(subject_runs),
        }
        for window_index, (window, harmonic_order, predicted_frequency) in enumerate(
            zip(
                specification.harmonic_windows,
                specification.harmonic_orders,
                predicted_frequencies,
                strict=True,
            )
        ):
            peaks = [_peak_for_window(run, window.name) for run in subject_runs]
            peak_frequency = float(np.median([peak.peak_frequency_hz for peak in peaks]))
            prominence = float(np.median([peak.prominence_db for peak in peaks]))
            offset = peak_frequency - float(predicted_frequency)
            participant_offset_matrix[subject_index, window_index] = offset
            prefix = window.name
            audit_row[f"{prefix}_median_peak_frequency_hz"] = peak_frequency
            audit_row[f"{prefix}_median_prominence_db"] = prominence
            audit_row[f"{prefix}_predicted_frequency_hz"] = float(predicted_frequency)
            audit_row[f"{prefix}_peak_offset_hz"] = offset
            participant_offset_rows.append(
                {
                    "subject_id": subject_id,
                    "window_name": window.name,
                    "harmonic_order": harmonic_order,
                    "predicted_frequency_hz": float(predicted_frequency),
                    "peak_frequency_hz": peak_frequency,
                    "offset_hz": offset,
                }
            )
        participant_audit_rows.append(audit_row)

    cohort_median, spectrum_ci_low, spectrum_ci_high = paired_participant_bootstrap(
        participant_spectrum_matrix,
        iterations=bootstrap.iterations,
        confidence_level=bootstrap.confidence_level,
        seed=bootstrap.seed,
    )
    offset_median, offset_ci_low, offset_ci_high = paired_participant_bootstrap(
        participant_offset_matrix,
        iterations=bootstrap.iterations,
        confidence_level=bootstrap.confidence_level,
        seed=bootstrap.seed,
    )
    cohort_spectrum = pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "median_relative_psd_db": cohort_median,
            "ci_low_relative_psd_db": spectrum_ci_low,
            "ci_high_relative_psd_db": spectrum_ci_high,
            "n_subjects": len(subjects),
        }
    )
    cohort_offsets = pd.DataFrame(
        {
            "window_name": [window.name for window in specification.harmonic_windows],
            "harmonic_order": specification.harmonic_orders,
            "predicted_frequency_hz": predicted_frequencies,
            "median_offset_hz": offset_median,
            "ci_low_offset_hz": offset_ci_low,
            "ci_high_offset_hz": offset_ci_high,
            "n_subjects": len(subjects),
        }
    )
    return ScannerHarmonicSummary(
        participant_spectra=pd.DataFrame(participant_spectrum_rows),
        cohort_spectrum=cohort_spectrum,
        participant_offsets=pd.DataFrame(participant_offset_rows),
        cohort_offsets=cohort_offsets,
        run_audit=run_audit,
        participant_audit=pd.DataFrame(participant_audit_rows),
    )


def _run_audit_table(runs: Sequence[RunSpectrum]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for run in runs:
        row: dict[str, float | int | str] = {
            "subject_id": run.subject_id,
            "run": run.run_id,
            "source_file": str(run.source_file),
            "n_channels": run.n_channels,
            "sampling_frequency_hz": run.sampling_frequency_hz,
            "n_samples": run.n_samples,
            "frequency_resolution_hz": float(np.median(np.diff(run.frequencies_hz))),
        }
        for peak in run.peaks:
            row[f"{peak.window_name}_peak_frequency_hz"] = peak.peak_frequency_hz
            row[f"{peak.window_name}_prominence_db"] = peak.prominence_db
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["subject_id", "run"], kind="stable").reset_index(
        drop=True
    )


def _peak_for_window(run: RunSpectrum, window_name: str) -> HarmonicPeak:
    matching = tuple(peak for peak in run.peaks if peak.window_name == window_name)
    if len(matching) != 1:
        raise ValueError(
            f"Run spectrum requires exactly one {window_name!r} peak: {run.source_file}"
        )
    return matching[0]


def build_scanner_harmonic_figure(
    summary: ScannerHarmonicSummary,
    config: Any,
) -> Figure:
    """Render participant spectra and their MRI-timing agreement."""
    dimensions = require_config_value(
        config,
        "study1.figures.scanner_harmonics.dimensions_mm",
    )
    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=(2.25, 1.0),
            left=0.075,
            right=0.985,
            bottom=0.22,
            top=0.91,
            wspace=0.34,
        )
        spectrum_axis = figure.add_subplot(grid[0, 0])
        offset_axis = figure.add_subplot(grid[0, 1])
        _draw_spectrum_panel(spectrum_axis, summary, config)
        _draw_offset_panel(offset_axis, summary, config)
        _add_panel_labels(spectrum_axis, offset_axis)
    return figure


def _draw_spectrum_panel(axis, summary: ScannerHarmonicSummary, config: Any) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    scanner = require_config_value(config, "study1.figures.scanner_harmonics")
    excluded_color = str(scanner["colors"]["excluded"])
    retained_color = str(scanner["colors"]["retained"])
    specification = scanner_harmonic_specification(config)

    for window in specification.harmonic_windows:
        axis.axvspan(
            window.low_hz,
            window.high_hz,
            color=excluded_color,
            alpha=0.12,
            linewidth=0.0,
            zorder=0,
        )

    cohort = summary.cohort_spectrum
    frequencies = cohort["frequency_hz"].to_numpy(dtype=float)
    axis.fill_between(
        frequencies,
        cohort["ci_low_relative_psd_db"].to_numpy(dtype=float),
        cohort["ci_high_relative_psd_db"].to_numpy(dtype=float),
        color="#202020",
        alpha=0.12,
        linewidth=0.0,
        zorder=1,
    )
    participant_matrix = summary.participant_spectra.pivot(
        index="subject_id",
        columns="frequency_hz",
        values="relative_psd_db",
    ).sort_index()
    for participant_spectrum in participant_matrix.to_numpy(dtype=float):
        axis.plot(
            frequencies,
            participant_spectrum,
            color=str(style["participant_color"]),
            alpha=max(float(style["participant_alpha"]), 0.28),
            linewidth=float(style["participant_line_width_pt"]),
            zorder=2,
        )
    axis.plot(
        frequencies,
        cohort["median_relative_psd_db"].to_numpy(dtype=float),
        color="#111111",
        linewidth=float(style["cohort_line_width_pt"]),
        zorder=3,
    )
    for lower_frequency, upper_frequency in SCANNER_CLEAN_GAMMA_RANGES_HZ.values():
        axis.plot(
            (lower_frequency, upper_frequency),
            (0.025, 0.025),
            color=retained_color,
            linewidth=2.4,
            solid_capstyle="butt",
            transform=axis.get_xaxis_transform(),
            clip_on=False,
            zorder=4,
        )

    lower_frequency, upper_frequency = specification.frequency_range_hz
    axis.set_xlim(lower_frequency, upper_frequency)
    axis.set_xticks((15, 20, 30, 40, 50, 60, 70, 80, 90))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD relative to participant median (dB)")
    axis.margins(y=0.08)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.01,
        0.98,
        f"n = {summary.n_subjects}; {summary.n_runs} runs",
        ha="left",
        va="top",
        transform=axis.transAxes,
        fontsize=float(
            require_config_value(config, "study1.figures.validity.font.annotation_pt")
        ),
    )
    axis.legend(
        handles=(
            Line2D(
                [],
                [],
                color=str(style["participant_color"]),
                linewidth=float(style["participant_line_width_pt"]),
                label="Participant median",
            ),
            Line2D(
                [],
                [],
                color="#111111",
                linewidth=float(style["cohort_line_width_pt"]),
                label="Cohort median",
            ),
            Patch(facecolor="#202020", alpha=0.12, label="95% bootstrap CI"),
            Patch(facecolor=excluded_color, alpha=0.12, label="Scanner-harmonic window"),
            Line2D([], [], color=retained_color, linewidth=2.4, label="Retained gamma"),
        ),
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),
        handlelength=2.0,
    )


def _draw_offset_panel(axis, summary: ScannerHarmonicSummary, config: Any) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    scanner = require_config_value(config, "study1.figures.scanner_harmonics")
    retained_color = str(scanner["colors"]["retained"])
    specification = scanner_harmonic_specification(config)
    bin_width = specification.sampling_frequency_hz / specification.n_fft

    axis.axhspan(-bin_width, bin_width, color=retained_color, alpha=0.10, linewidth=0.0)
    axis.axhline(0.0, color="#333333", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    participants = summary.participant_offsets
    cohort = summary.cohort_offsets.set_index("window_name")
    x_positions = np.arange(len(specification.harmonic_windows), dtype=float)
    for window_index, (x_position, window) in enumerate(
        zip(x_positions, specification.harmonic_windows, strict=True)
    ):
        values = participants.loc[
            participants["window_name"] == window.name,
            ["subject_id", "offset_hz"],
        ].sort_values("subject_id", kind="stable")
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            x_position + jitter,
            values["offset_hz"].to_numpy(dtype=float),
            s=float(style["participant_marker_size_pt"]) ** 2,
            color=str(style["participant_color"]),
            alpha=max(float(style["participant_alpha"]), 0.55),
            edgecolors="white",
            linewidths=0.25,
            zorder=2,
        )
        estimate = cohort.loc[window.name]
        median = float(estimate["median_offset_hz"])
        axis.errorbar(
            x_position,
            median,
            yerr=np.asarray(
                [
                    [median - float(estimate["ci_low_offset_hz"])],
                    [float(estimate["ci_high_offset_hz"]) - median],
                ]
            ),
            color="#111111",
            linestyle="none",
            elinewidth=float(style["confidence_line_width_pt"]),
            marker="D",
            markersize=float(style["cohort_marker_size_pt"]),
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=float(style["confidence_line_width_pt"]),
            capsize=0.0,
            zorder=3,
        )

    axis.set_xticks(
        x_positions,
        labels=[
            f"{order} × fTR\n{predicted:.3f} Hz"
            for order, predicted in zip(
                specification.harmonic_orders,
                cohort["predicted_frequency_hz"].to_numpy(dtype=float),
                strict=True,
            )
        ],
    )
    axis.set_ylabel("Peak offset from predicted\nTR harmonic (Hz)")
    y_values = np.concatenate(
        (
            participants["offset_hz"].to_numpy(dtype=float),
            cohort[["ci_low_offset_hz", "ci_high_offset_hz"]].to_numpy(dtype=float).ravel(),
            np.asarray((-bin_width, bin_width)),
        )
    )
    half_range = max(1.15 * float(np.max(np.abs(y_values))), 0.08)
    axis.set_ylim(-half_range, half_range)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.02,
        0.98,
        f"blue band: ±1 Welch bin ({bin_width:.3f} Hz)",
        ha="left",
        va="top",
        color=retained_color,
        transform=axis.transAxes,
        fontsize=float(
            require_config_value(config, "study1.figures.validity.font.annotation_pt")
        ),
    )


def _add_panel_labels(spectrum_axis, offset_axis) -> None:
    spectrum_axis.text(
        -0.13,
        1.04,
        "a",
        transform=spectrum_axis.transAxes,
        fontweight="bold",
        fontsize=8.0,
    )
    offset_axis.text(
        -0.23,
        1.04,
        "b",
        transform=offset_axis.transAxes,
        fontweight="bold",
        fontsize=8.0,
    )


__all__ = [
    "HarmonicPeak",
    "ParticipantBootstrapSpecification",
    "RunSpectrum",
    "ScannerHarmonicSpecification",
    "ScannerHarmonicSummary",
    "build_scanner_harmonic_summary",
    "build_scanner_harmonic_figure",
    "discover_final_clean_runs",
    "estimate_run_spectrum",
    "paired_participant_bootstrap",
    "scanner_harmonic_specification",
    "select_scanner_harmonic_peaks",
    "validity_bootstrap_specification",
]
