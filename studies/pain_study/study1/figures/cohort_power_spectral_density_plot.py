"""Publication rendering for the Study 1 cohort power spectrum."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from studies.pain_study.analysis.gradient.scanner_harmonics import DEFAULT_HARMONIC_WINDOWS
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.scanner_contamination import SCANNER_CLEAN_GAMMA_RANGES_HZ
from studies.pain_study.study1.figures.cohort_power_spectral_density import (
    CohortPsdSummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

NEURAL_BANDS = (
    ("δ", 1.0, 3.9),
    ("θ", 4.0, 7.9),
    ("α", 8.0, 12.9),
    ("β", 13.0, 30.0),
    ("low γ", *SCANNER_CLEAN_GAMMA_RANGES_HZ["gamma_low_clean"]),
    ("mid γ", *SCANNER_CLEAN_GAMMA_RANGES_HZ["gamma_mid_clean"]),
    ("high γ", *SCANNER_CLEAN_GAMMA_RANGES_HZ["gamma_high_clean"]),
)


def build_cohort_psd_figure(summary: CohortPsdSummary, config: Any) -> Figure:
    """Render participant spectra, cohort uncertainty, and frequency annotations."""
    _validate_summary(summary)
    figure_config = require_config_value(
        config,
        "study1.figures.cohort_power_spectral_density",
    )
    colors = figure_config["colors"]
    if not isinstance(colors, Mapping):
        raise ValueError("Cohort PSD colors must be a mapping.")
    dimensions = figure_config["dimensions_mm"]
    frequency_range = tuple(float(value) for value in figure_config["frequency_range_hz"])

    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=(1.0, 0.075),
            left=0.085,
            right=0.985,
            bottom=0.18,
            top=0.72,
            hspace=0.10,
        )
        spectrum_axis = figure.add_subplot(grid[0, 0], label="spectrum")
        band_axis = figure.add_subplot(
            grid[1, 0],
            sharex=spectrum_axis,
            label="frequency-bands",
        )
        _draw_spectrum(
            spectrum_axis,
            summary,
            scanner_color=str(colors["scanner_window"]),
            config=config,
        )
        _draw_band_strip(
            band_axis,
            neural_color=str(colors["neural_band"]),
        )
        spectrum_axis.set_xlim(frequency_range)
        figure.legend(
            handles=_legend_handles(
                scanner_color=str(colors["scanner_window"]),
                config=config,
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.835),
            ncol=2,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.5,
            columnspacing=1.6,
        )
        title = figure.text(
            0.085,
            0.965,
            "Continuous EEG power spectrum after preprocessing",
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
        title.set_gid("cohort-psd-title")
        method = figure.text(
            0.085,
            0.925,
            "Participant spectra: median across runs in linear power, then dB · "
            "cohort: median across participants",
            ha="left",
            va="top",
            fontsize=6.0,
        )
        method.set_gid("cohort-psd-method")
        sample = figure.text(
            0.085,
            0.89,
            _sample_metadata(summary),
            ha="left",
            va="top",
            fontsize=6.0,
        )
        sample.set_gid("cohort-psd-sample")
    return figure


def _draw_spectrum(
    axis,
    summary: CohortPsdSummary,
    *,
    scanner_color: str,
    config: Any,
) -> None:
    style = require_config_value(config, "study1.figures.validity.style")
    for window in DEFAULT_HARMONIC_WINDOWS:
        axis.axvspan(
            window.low_hz,
            window.high_hz,
            color=scanner_color,
            alpha=0.10,
            linewidth=0.0,
            zorder=0,
        )

    cohort = summary.cohort_spectrum
    frequencies = cohort["frequency_hz"].to_numpy(dtype=float)
    axis.fill_between(
        frequencies,
        cohort["ci_low_psd_db_uv2_hz"].to_numpy(dtype=float),
        cohort["ci_high_psd_db_uv2_hz"].to_numpy(dtype=float),
        color="#202020",
        alpha=0.12,
        linewidth=0.0,
        zorder=1,
    )
    participant_matrix = summary.participant_spectra.pivot(
        index="subject_id",
        columns="frequency_hz",
        values="psd_db_uv2_hz",
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
        cohort["median_psd_db_uv2_hz"].to_numpy(dtype=float),
        color="#111111",
        linewidth=float(style["cohort_line_width_pt"]),
        zorder=3,
    )
    axis.set_ylabel("PSD (dB µV²/Hz)")
    axis.tick_params(axis="x", labelbottom=False)
    axis.margins(y=0.08)


def _draw_band_strip(axis, *, neural_color: str) -> None:
    for label, lower_frequency, upper_frequency in NEURAL_BANDS:
        axis.axvspan(
            lower_frequency,
            upper_frequency,
            facecolor=neural_color,
            alpha=0.18,
            linewidth=0.35,
            edgecolor="white",
        )
        axis.text(
            (lower_frequency + upper_frequency) / 2.0,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=5.5,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    axis.set_xticks((1, 10, 20, 30, 40, 50, 60, 70, 80, 90))
    axis.set_xlabel("Frequency (Hz)")
    axis.tick_params(axis="x", pad=2.0)
    axis.spines[["left", "right", "top"]].set_visible(False)


def _legend_handles(*, scanner_color: str, config: Any) -> tuple:
    style = require_config_value(config, "study1.figures.validity.style")
    return (
        Line2D(
            [],
            [],
            color=str(style["participant_color"]),
            linewidth=float(style["participant_line_width_pt"]),
            label="Participant spectrum",
        ),
        Line2D(
            [],
            [],
            color="#111111",
            linewidth=float(style["cohort_line_width_pt"]),
            label="Cohort median",
        ),
        Patch(
            facecolor="#202020",
            alpha=0.12,
            label="Pointwise 95% participant-bootstrap CI",
        ),
        Patch(
            facecolor=scanner_color,
            alpha=0.10,
            label="Scanner-harmonic exclusion window",
        ),
    )


def _sample_metadata(summary: CohortPsdSummary) -> str:
    metadata = summary.participant_spectra.loc[:, ["subject_id", "n_runs"]].drop_duplicates()
    if metadata["subject_id"].duplicated().any():
        raise ValueError("Cohort PSD run counts must be constant within participants.")
    run_counts = pd.to_numeric(metadata["n_runs"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(run_counts).all() or np.any(run_counts < 1.0):
        raise ValueError("Cohort PSD participant run counts must be positive and finite.")
    if not np.equal(run_counts, np.floor(run_counts)).all():
        raise ValueError("Cohort PSD participant run counts must be integers.")
    run_counts = run_counts.astype(int)
    if int(run_counts.sum()) != summary.n_runs:
        raise ValueError("Cohort PSD participant run counts do not match the run audit.")
    median_runs = float(np.median(run_counts))
    return (
        f"n={summary.n_subjects} participants · {summary.n_runs} runs · "
        f"runs/participant: median {median_runs:g}, range {run_counts.min()}–{run_counts.max()}"
    )


def _validate_summary(summary: CohortPsdSummary) -> None:
    cohort_frequencies = summary.cohort_spectrum["frequency_hz"].to_numpy(dtype=float)
    if (
        cohort_frequencies.ndim != 1
        or cohort_frequencies.size < 2
        or not np.isfinite(cohort_frequencies).all()
        or np.any(np.diff(cohort_frequencies) <= 0.0)
    ):
        raise ValueError("Cohort PSD frequency axis must be finite and strictly increasing.")
    participant_frequencies = np.sort(
        summary.participant_spectra["frequency_hz"].unique().astype(float)
    )
    if not np.array_equal(cohort_frequencies, participant_frequencies):
        raise ValueError("Cohort and participant PSD frequency axes must match.")
    if summary.n_subjects < 1 or summary.n_runs < 1:
        raise ValueError("Cohort PSD figure requires participants and runs.")
    if summary.participant_spectra.duplicated(["subject_id", "frequency_hz"]).any():
        raise ValueError("Cohort PSD requires a complete participant-frequency grid.")
    participant_matrix = summary.participant_spectra.pivot(
        index="subject_id",
        columns="frequency_hz",
        values="psd_db_uv2_hz",
    ).reindex(columns=cohort_frequencies)
    if participant_matrix.isna().any().any():
        raise ValueError("Cohort PSD requires a complete participant-frequency grid.")
    participant_values = participant_matrix.to_numpy(dtype=float)
    cohort_values = summary.cohort_spectrum[
        ["median_psd_db_uv2_hz", "ci_low_psd_db_uv2_hz", "ci_high_psd_db_uv2_hz"]
    ].to_numpy(dtype=float)
    if not np.isfinite(participant_values).all() or not np.isfinite(cohort_values).all():
        raise ValueError("Cohort PSD spectra and intervals must be finite.")
    median = cohort_values[:, 0]
    ci_low = cohort_values[:, 1]
    ci_high = cohort_values[:, 2]
    if np.any(ci_low > median) or np.any(median > ci_high):
        raise ValueError("Cohort PSD intervals must be ordered around the cohort median.")
    cohort_counts = pd.to_numeric(
        summary.cohort_spectrum["n_subjects"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.all(cohort_counts == summary.n_subjects):
        raise ValueError("Cohort PSD cohort participant count is inconsistent.")


__all__ = ["build_cohort_psd_figure"]
