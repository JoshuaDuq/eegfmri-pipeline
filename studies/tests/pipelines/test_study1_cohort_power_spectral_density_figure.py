from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config


def test_build_cohort_psd_figure_has_publication_structure() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density_plot import (
        build_cohort_psd_figure,
    )

    summary = _summary()
    figure = build_cohort_psd_figure(summary, load_study1_config())

    assert len(figure.axes) == 2
    spectrum_axis, band_axis = figure.axes
    assert spectrum_axis.get_ylabel() == "PSD (dB µV²/Hz)"
    assert band_axis.get_xlabel() == "Frequency (Hz)"
    assert spectrum_axis.get_xlim() == pytest.approx((1.0, 90.0))
    assert len(spectrum_axis.lines) == summary.n_subjects + 1
    assert len(spectrum_axis.collections) == 1
    assert not spectrum_axis.spines["top"].get_visible()
    assert not spectrum_axis.spines["right"].get_visible()
    plt.close(figure)


def test_build_cohort_psd_figure_draws_exact_frequency_annotations() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density_plot import (
        build_cohort_psd_figure,
    )

    figure = build_cohort_psd_figure(_summary(), load_study1_config())
    spectrum_axis, band_axis = figure.axes

    harmonic_bounds = sorted(
        (patch.get_x(), patch.get_x() + patch.get_width())
        for patch in spectrum_axis.patches
    )
    assert harmonic_bounds == pytest.approx(
        [(18.0, 23.0), (38.0, 43.0), (56.0, 67.0), (77.0, 85.0)]
    )
    band_bounds = [
        (patch.get_x(), patch.get_x() + patch.get_width())
        for patch in band_axis.patches
    ]
    assert band_bounds == pytest.approx(
        [
            (1.0, 3.9),
            (4.0, 7.9),
            (8.0, 12.9),
            (13.0, 30.0),
            (30.1, 38.0),
            (43.0, 56.0),
            (67.0, 77.0),
        ]
    )
    assert [text.get_text() for text in band_axis.texts] == [
        "δ",
        "θ",
        "α",
        "β",
        "low γ",
        "mid γ",
        "high γ",
    ]
    plt.close(figure)


def test_build_cohort_psd_figure_keeps_legend_outside_data_axes() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density_plot import (
        build_cohort_psd_figure,
    )

    figure = build_cohort_psd_figure(_summary(), load_study1_config())

    assert len(figure.legends) == 1
    assert figure.legends[0]._ncols == 4
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert max(axis.get_position().y1 for axis in figure.axes) < 0.86
    plt.close(figure)


def _summary():
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        CohortPsdSummary,
    )

    frequencies = np.linspace(1.0, 90.0, 180)
    subjects = ("sub-01", "sub-02", "sub-03")
    participant_rows = []
    participant_values = []
    for subject_index, subject_id in enumerate(subjects):
        values = 20.0 - 10.0 * np.log10(frequencies) + subject_index
        participant_values.append(values)
        participant_rows.extend(
            {
                "subject_id": subject_id,
                "frequency_hz": frequency,
                "psd_db_uv2_hz": value,
                "n_runs": 1,
            }
            for frequency, value in zip(frequencies, values, strict=True)
        )
    participant_matrix = np.vstack(participant_values)
    cohort_median = np.median(participant_matrix, axis=0)
    run_audit = pd.DataFrame(
        {
            "subject_id": subjects,
            "run": ["1"] * len(subjects),
            "source_file": [f"{subject}_run-1.fif" for subject in subjects],
        }
    )
    return CohortPsdSummary(
        participant_spectra=pd.DataFrame(participant_rows),
        cohort_spectrum=pd.DataFrame(
            {
                "frequency_hz": frequencies,
                "median_psd_db_uv2_hz": cohort_median,
                "ci_low_psd_db_uv2_hz": cohort_median - 1.0,
                "ci_high_psd_db_uv2_hz": cohort_median + 1.0,
                "n_subjects": len(subjects),
            }
        ),
        run_audit=run_audit,
    )
