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
from studies.pain_study.study1.figures.primary_prediction import (
    PrimaryPredictionSummary,
)


def test_build_primary_prediction_figure_has_estimation_structure() -> None:
    from studies.pain_study.study1.figures.primary_prediction_plot import (
        build_primary_prediction_figure,
    )

    figure = build_primary_prediction_figure(primary_summary(), load_study1_config())

    try:
        assert len(figure.axes) == 4
        assert len(figure.legends) == 1
        assert figure.legends[0]._ncols == 3
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert {text.get_text() for text in figure.texts}.issuperset({"a", "b", "NPS", "SIIPS1"})
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 86.0 / 25.4))
        assert max(axis.get_position().y1 for axis in figure.axes) < 0.82
    finally:
        plt.close(figure)


def test_build_primary_prediction_figure_uses_shared_scientific_limits() -> None:
    from studies.pain_study.study1.figures.primary_prediction_plot import (
        build_primary_prediction_figure,
    )

    figure = build_primary_prediction_figure(primary_summary(), load_study1_config())

    try:
        nps_absolute, nps_delta, siips_absolute, siips_delta = figure.axes
        assert np.allclose(nps_absolute.get_ylim(), siips_absolute.get_ylim())
        assert np.allclose(nps_delta.get_ylim(), siips_delta.get_ylim())
        delta_limits = nps_delta.get_ylim()
        assert np.isclose(delta_limits[0], -delta_limits[1])
        for axis in figure.axes:
            assert any(
                np.allclose(line.get_ydata(), (0.0, 0.0))
                for line in axis.lines
                if len(line.get_ydata()) == 2
            )
    finally:
        plt.close(figure)


def test_build_primary_prediction_figure_draws_paired_participants_and_target_colors() -> None:
    from matplotlib.colors import to_hex

    from studies.pain_study.study1.figures.primary_prediction_plot import (
        build_primary_prediction_figure,
    )

    summary = primary_summary()
    figure = build_primary_prediction_figure(summary, load_study1_config())

    try:
        expected_colors = ("#0072b2", "#d55e00")
        for absolute_axis, delta_axis, target, expected_color in zip(
            figure.axes[::2],
            figure.axes[1::2],
            summary.targets,
            expected_colors,
            strict=True,
        ):
            n_subjects = int(summary.participant_performance["target"].eq(target).sum())
            paired_lines = [
                line
                for line in absolute_axis.lines
                if line.get_linestyle() == "-" and len(line.get_xdata()) == 2
            ]
            assert len(paired_lines) == n_subjects
            full_colors = {to_hex(color) for color in absolute_axis.collections[1].get_facecolors()}
            assert full_colors == {expected_color}
            cohort_markers = [line for line in delta_axis.lines if line.get_marker() == "D"]
            assert len(cohort_markers) == 1
            assert to_hex(cohort_markers[0].get_markeredgecolor()) == expected_color
    finally:
        plt.close(figure)


def test_build_primary_prediction_figure_jitter_is_deterministic() -> None:
    from studies.pain_study.study1.figures.primary_prediction_plot import (
        build_primary_prediction_figure,
    )

    config = load_study1_config()
    first = build_primary_prediction_figure(primary_summary(), config)
    second = build_primary_prediction_figure(primary_summary(), config)

    try:
        first_offsets = first.axes[1].collections[0].get_offsets()
        second_offsets = second.axes[1].collections[0].get_offsets()
        assert np.allclose(first_offsets, second_offsets)
    finally:
        plt.close(first)
        plt.close(second)


def test_primary_prediction_writer_creates_exact_svg_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_primary_prediction as module

    summary = primary_summary()
    monkeypatch.setattr(
        module,
        "load_primary_prediction_summary",
        lambda *args, **kwargs: summary,
    )
    outputs = [
        module.write_primary_prediction(
            report_path=tmp_path / "report.tsv",
            config=load_study1_config(),
            output_path=tmp_path / directory / "primary_prediction_estimation.svg",
        )
        for directory in ("first", "second")
    ]

    first, second = outputs
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(183.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(86.0, abs=0.01)
    assert first.svg.read_text(encoding="utf-8").count("<text") > 0
    assert pd.read_csv(first.subject_tsv, sep="\t").columns.tolist() == (
        summary.participant_performance.columns.tolist()
    )
    assert pd.read_parquet(first.subject_parquet).columns.tolist() == (
        summary.participant_performance.columns.tolist()
    )
    assert pd.read_csv(first.summary_tsv, sep="\t").columns.tolist() == (
        summary.cohort_performance.columns.tolist()
    )
    assert pd.read_parquet(first.summary_parquet).columns.tolist() == (
        summary.cohort_performance.columns.tolist()
    )
    assert sorted(path.name for path in first.svg.parent.iterdir()) == [
        "primary_prediction_by_subject.parquet",
        "primary_prediction_by_subject.tsv",
        "primary_prediction_estimation.svg",
        "primary_prediction_summary.parquet",
        "primary_prediction_summary.tsv",
    ]


def test_primary_prediction_plot_main_writes_only_its_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_primary_prediction as module

    output = tmp_path / "primary.svg"
    expected = module.PrimaryPredictionFigurePaths(
        svg=output,
        subject_tsv=tmp_path / "subject.tsv",
        subject_parquet=tmp_path / "subject.parquet",
        summary_tsv=tmp_path / "summary.tsv",
        summary_parquet=tmp_path / "summary.parquet",
    )
    monkeypatch.setattr(module, "load_config", lambda path: load_study1_config())
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "write_primary_prediction", lambda **kwargs: expected)

    result = module.main(
        [
            "--config",
            "pipeline.yaml",
            "--report",
            str(tmp_path / "study1_report.tsv"),
            "--output",
            str(output),
        ]
    )

    assert result == expected
    assert capsys.readouterr().out.strip() == str(output)


def test_primary_prediction_cli_module_help_has_no_runtime_warning(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study1.figures.plot_primary_prediction",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def primary_summary() -> PrimaryPredictionSummary:
    participants: list[dict[str, object]] = []
    cohorts: list[dict[str, object]] = []
    for target_index, target in enumerate(("NPS", "SIIPS1")):
        nuisance = np.asarray((-0.45, -0.18, 0.02, 0.08, 0.14), dtype=float)
        nuisance -= target_index * 0.10
        delta = np.asarray((-0.12, -0.02, 0.03, 0.08, 0.16), dtype=float)
        delta -= target_index * 0.01
        full = nuisance + delta
        for fold, (nuisance_r2, full_r2, delta_r2) in enumerate(
            zip(nuisance, full, delta, strict=True)
        ):
            participants.append(
                {
                    "target": target,
                    "model": "elasticnet",
                    "feature_spec": "alpha_beta_gamma",
                    "fold": fold,
                    "subject_id": f"sub-{fold + 1:02d}",
                    "r2_nuisance": nuisance_r2,
                    "r2": full_r2,
                    "delta_r2": delta_r2,
                    "fold_table_path": "/tmp/model_comparison.tsv",
                }
            )
        mean_full = float(full.mean())
        mean_delta = float(delta.mean())
        cohorts.append(
            {
                "target": target,
                "model": "elasticnet",
                "feature_spec": "alpha_beta_gamma",
                "mean_nuisance_r2": float(nuisance.mean()),
                "mean_r2": mean_full,
                "ci_low_r2": mean_full - 0.25,
                "ci_high_r2": mean_full + 0.25,
                "mean_delta_r2": mean_delta,
                "ci_low_delta_r2": mean_delta - 0.08,
                "ci_high_delta_r2": mean_delta + 0.08,
                "p_value_delta_r2": 0.2,
                "p_value_delta_r2_holm": 0.4,
                "n_folds": 5,
                "n_subjects": 5,
                "summary_path": "/tmp/summary.json",
                "fold_table_path": "/tmp/model_comparison.tsv",
            }
        )
    return PrimaryPredictionSummary(
        targets=("NPS", "SIIPS1"),
        participant_performance=pd.DataFrame(participants),
        cohort_performance=pd.DataFrame(cohorts),
    )
