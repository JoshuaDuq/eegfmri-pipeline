from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.config.loader import load_study1_config

WINDOW_NAMES = (
    "scanner_18_23",
    "scanner_38_43",
    "scanner_56_67",
    "scanner_77_85",
)
HARMONIC_ORDERS = (18, 37, 55, 74)
PREDICTED_FREQUENCIES = np.asarray(HARMONIC_ORDERS, dtype=float) / 0.9


def test_build_scanner_harmonic_figure_draws_scientific_layers(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        build_scanner_harmonic_figure,
    )

    summary = _summary()
    figure = build_scanner_harmonic_figure(summary, _config(tmp_path))

    assert len(figure.axes) == 2
    spectrum_axis, offset_axis = figure.axes
    assert spectrum_axis.get_xlabel() == "Frequency (Hz)"
    assert spectrum_axis.get_ylabel() == "PSD relative to participant median (dB)"
    assert offset_axis.get_ylabel() == "Peak offset from predicted\nTR harmonic (Hz)"
    assert spectrum_axis.get_title() == ""
    assert offset_axis.get_title() == ""
    assert len(spectrum_axis.lines) >= summary.n_subjects + 4
    assert len(spectrum_axis.collections) >= 1
    assert len(offset_axis.collections) >= 5
    assert not spectrum_axis.spines["top"].get_visible()
    assert not offset_axis.spines["right"].get_visible()
    plt.close(figure)


def test_scanner_harmonic_writer_creates_exact_svg_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_scanner_harmonic_spectrum as module

    summary = _summary()
    monkeypatch.setattr(module, "_build_summary", lambda **kwargs: summary)
    output_paths = []
    for directory_name in ("first", "second"):
        output = tmp_path / directory_name / "scanner_harmonic_spectrum.svg"
        output_paths.append(
            module.write_scanner_harmonic_spectrum(
                derivative_root=tmp_path / "derivatives",
                task="thermalactive",
                config=_config(tmp_path),
                output_path=output,
            )
        )

    first, second = output_paths
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(183.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(82.0, abs=0.01)
    assert first.svg.read_text(encoding="utf-8").count("<text") > 0
    assert pd.read_csv(first.run_tsv, sep="\t").columns.tolist() == _run_audit_columns()
    assert pd.read_parquet(first.run_parquet).columns.tolist() == _run_audit_columns()
    assert pd.read_csv(first.participant_tsv, sep="\t").columns.tolist() == (
        _participant_audit_columns()
    )
    assert pd.read_parquet(first.participant_parquet).columns.tolist() == (
        _participant_audit_columns()
    )
    assert sorted(path.name for path in first.svg.parent.iterdir()) == [
        "scanner_harmonic_spectrum.svg",
        "scanner_harmonic_spectrum_by_run.parquet",
        "scanner_harmonic_spectrum_by_run.tsv",
        "scanner_harmonic_spectrum_by_subject.parquet",
        "scanner_harmonic_spectrum_by_subject.tsv",
    ]


def test_scanner_harmonic_plot_main_writes_only_its_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_scanner_harmonic_spectrum as module

    output = tmp_path / "scanner.svg"
    expected = module.ScannerHarmonicFigurePaths(
        svg=output,
        run_tsv=tmp_path / "run.tsv",
        run_parquet=tmp_path / "run.parquet",
        participant_tsv=tmp_path / "participant.tsv",
        participant_parquet=tmp_path / "participant.parquet",
    )
    config = _config(tmp_path)
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "write_scanner_harmonic_spectrum", lambda **kwargs: expected)

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


def _config(tmp_path: Path) -> ConfigDict:
    config = load_study1_config()
    derivative_root = str(tmp_path / "derivatives")
    config["paths"] = {"deriv_root": derivative_root}
    config["deriv_root"] = derivative_root
    config["study1"]["figures"]["validity"]["bootstrap"].update(
        iterations=200,
        confidence_level=0.95,
        seed=42,
    )
    return ConfigDict(config)


def _summary():
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        ScannerHarmonicSummary,
    )

    subjects = ("sub-01", "sub-02", "sub-03")
    frequencies = np.linspace(15.0, 90.0, 61)
    participant_rows = []
    participant_values = []
    for subject_index, subject_id in enumerate(subjects):
        values = (
            5.0 * np.sin(frequencies / 9.0)
            + subject_index
            + sum(
                (7.0 + peak_index)
                * np.exp(-0.5 * ((frequencies - peak) / 0.45) ** 2)
                for peak_index, peak in enumerate((20.0, 41.0, 61.0, 82.0))
            )
        )
        participant_values.append(values)
        participant_rows.extend(
            {
                "subject_id": subject_id,
                "frequency_hz": frequency,
                "relative_psd_db": value,
            }
            for frequency, value in zip(frequencies, values, strict=True)
        )
    matrix = np.vstack(participant_values)
    cohort_median = np.median(matrix, axis=0)

    offset_rows = []
    for subject_index, subject_id in enumerate(subjects):
        for window_name, order, predicted in zip(
            WINDOW_NAMES,
            HARMONIC_ORDERS,
            PREDICTED_FREQUENCIES,
            strict=True,
        ):
            offset = (-0.03, 0.01, 0.05)[subject_index]
            offset_rows.append(
                {
                    "subject_id": subject_id,
                    "window_name": window_name,
                    "harmonic_order": order,
                    "predicted_frequency_hz": predicted,
                    "peak_frequency_hz": predicted + offset,
                    "offset_hz": offset,
                }
            )
    cohort_offsets = pd.DataFrame(
        {
            "window_name": WINDOW_NAMES,
            "harmonic_order": HARMONIC_ORDERS,
            "predicted_frequency_hz": PREDICTED_FREQUENCIES,
            "median_offset_hz": [0.01] * 4,
            "ci_low_offset_hz": [-0.03] * 4,
            "ci_high_offset_hz": [0.05] * 4,
            "n_subjects": [3] * 4,
        }
    )
    run_audit = pd.DataFrame(
        [
            {
                **dict.fromkeys(_run_audit_columns(), 1.0),
                "subject_id": subject_id,
                "run": "1",
                "source_file": f"{subject_id}_run-1.fif",
            }
            for subject_id in subjects
        ],
        columns=_run_audit_columns(),
    )
    participant_audit = pd.DataFrame(
        [
            {
                **dict.fromkeys(_participant_audit_columns(), 1.0),
                "subject_id": subject_id,
                "n_runs": 1,
            }
            for subject_id in subjects
        ],
        columns=_participant_audit_columns(),
    )
    return ScannerHarmonicSummary(
        participant_spectra=pd.DataFrame(participant_rows),
        cohort_spectrum=pd.DataFrame(
            {
                "frequency_hz": frequencies,
                "median_relative_psd_db": cohort_median,
                "ci_low_relative_psd_db": cohort_median - 1.0,
                "ci_high_relative_psd_db": cohort_median + 1.0,
                "n_subjects": 3,
            }
        ),
        participant_offsets=pd.DataFrame(offset_rows),
        cohort_offsets=cohort_offsets,
        run_audit=run_audit,
        participant_audit=participant_audit,
    )


def _run_audit_columns() -> list[str]:
    columns = [
        "subject_id",
        "run",
        "source_file",
        "n_channels",
        "sampling_frequency_hz",
        "n_samples",
        "frequency_resolution_hz",
    ]
    for window_name in WINDOW_NAMES:
        columns.extend(
            [
                f"{window_name}_peak_frequency_hz",
                f"{window_name}_prominence_db",
            ]
        )
    return columns


def _participant_audit_columns() -> list[str]:
    columns = ["subject_id", "n_runs"]
    for window_name in WINDOW_NAMES:
        columns.extend(
            [
                f"{window_name}_median_peak_frequency_hz",
                f"{window_name}_median_prominence_db",
                f"{window_name}_predicted_frequency_hz",
                f"{window_name}_peak_offset_hz",
            ]
        )
    return columns
