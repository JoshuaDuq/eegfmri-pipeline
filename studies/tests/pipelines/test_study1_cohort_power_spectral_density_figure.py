from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

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


def test_cohort_psd_writer_creates_exact_svg_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_cohort_power_spectral_density as module

    summary = _summary()
    monkeypatch.setattr(module, "_build_summary", lambda **kwargs: summary)
    outputs = []
    for directory_name in ("first", "second"):
        output = tmp_path / directory_name / "cohort_power_spectral_density.svg"
        outputs.append(
            module.write_cohort_power_spectral_density(
                derivative_root=tmp_path / "derivatives",
                task="thermalactive",
                config=load_study1_config(),
                output_path=output,
            )
        )

    first, second = outputs
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(183.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(92.0, abs=0.01)
    assert first.svg.read_text(encoding="utf-8").count("<text") > 0
    assert pd.read_csv(first.run_tsv, sep="\t").columns.tolist() == (
        summary.run_audit.columns.tolist()
    )
    assert pd.read_parquet(first.run_parquet).columns.tolist() == (
        summary.run_audit.columns.tolist()
    )
    assert pd.read_csv(first.participant_tsv, sep="\t").columns.tolist() == (
        summary.participant_spectra.columns.tolist()
    )
    assert pd.read_parquet(first.participant_parquet).columns.tolist() == (
        summary.participant_spectra.columns.tolist()
    )
    assert pd.read_csv(first.summary_tsv, sep="\t").columns.tolist() == (
        summary.cohort_spectrum.columns.tolist()
    )
    assert pd.read_parquet(first.summary_parquet).columns.tolist() == (
        summary.cohort_spectrum.columns.tolist()
    )
    assert sorted(path.name for path in first.svg.parent.iterdir()) == [
        "cohort_power_spectral_density.svg",
        "cohort_power_spectral_density_by_run.parquet",
        "cohort_power_spectral_density_by_run.tsv",
        "cohort_power_spectral_density_by_subject.parquet",
        "cohort_power_spectral_density_by_subject.tsv",
        "cohort_power_spectral_density_summary.parquet",
        "cohort_power_spectral_density_summary.tsv",
    ]


def test_cohort_psd_plot_main_writes_only_its_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_cohort_power_spectral_density as module

    output = tmp_path / "cohort.svg"
    expected = module.CohortPsdFigurePaths(
        svg=output,
        run_tsv=tmp_path / "run.tsv",
        run_parquet=tmp_path / "run.parquet",
        participant_tsv=tmp_path / "participant.tsv",
        participant_parquet=tmp_path / "participant.parquet",
        summary_tsv=tmp_path / "summary.tsv",
        summary_parquet=tmp_path / "summary.parquet",
    )
    config = load_study1_config()
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "write_cohort_power_spectral_density", lambda **kwargs: expected)

    result = module.main(
        [
            "--config",
            "pipeline.yaml",
            "--task",
            "thermalactive",
            "--derivative-root",
            str(tmp_path / "derivatives"),
            "--output",
            str(output),
        ]
    )

    assert result == expected
    assert capsys.readouterr().out.strip() == str(output)


def test_cohort_psd_cli_module_help_has_no_runtime_warning(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study1.figures.plot_cohort_power_spectral_density",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


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
