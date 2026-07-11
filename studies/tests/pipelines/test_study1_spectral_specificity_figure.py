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
from studies.pain_study.study1.figures.spectral_specificity import (
    SpectralSpecificitySummary,
)

FEATURE_SPECS = (
    "alpha",
    "beta",
    "gamma",
    "alpha_beta",
    "alpha_beta_gamma",
)
FEATURE_LABELS = tuple(f"Feature {index + 1}" for index in range(5))


def test_build_spectral_specificity_figure_has_publication_structure() -> None:
    from studies.pain_study.study1.figures.spectral_specificity_plot import (
        build_spectral_specificity_figure,
    )

    figure = build_spectral_specificity_figure(spectral_summary(), load_study1_config())

    try:
        assert len(figure.axes) == 2
        assert [axis.get_title() for axis in figure.axes] == ["NPS", "SIIPS1"]
        assert len(figure.legends) == 1
        assert figure.legends[0]._ncols == 2
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == list(
            FEATURE_LABELS
        )
        assert not figure.axes[1].get_yticklabels()
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 84.0 / 25.4))
        assert max(axis.get_position().y1 for axis in figure.axes) < 0.84
    finally:
        plt.close(figure)


def test_build_spectral_specificity_figure_uses_shared_symmetric_scale() -> None:
    from studies.pain_study.study1.figures.spectral_specificity_plot import (
        build_spectral_specificity_figure,
    )

    figure = build_spectral_specificity_figure(spectral_summary(), load_study1_config())

    try:
        assert np.allclose(figure.axes[0].get_xlim(), figure.axes[1].get_xlim())
        limits = figure.axes[0].get_xlim()
        assert np.isclose(limits[0], -limits[1])
        for axis in figure.axes:
            assert any(
                np.allclose(line.get_xdata(), (0.0, 0.0))
                for line in axis.lines
                if len(line.get_xdata()) == 2
            )
    finally:
        plt.close(figure)


def test_build_spectral_specificity_figure_uses_target_colors_without_traces() -> None:
    from matplotlib.colors import to_hex

    from studies.pain_study.study1.figures.spectral_specificity_plot import (
        build_spectral_specificity_figure,
    )

    figure = build_spectral_specificity_figure(spectral_summary(), load_study1_config())

    try:
        for axis, expected_color in zip(
            figure.axes,
            ("#0072b2", "#d55e00"),
            strict=True,
        ):
            cohort_markers = [line for line in axis.lines if line.get_marker() == "D"]
            assert len(cohort_markers) == 5
            assert {to_hex(line.get_markeredgecolor()) for line in cohort_markers} == {
                expected_color
            }
            assert all(len(line.get_xdata()) == 1 for line in cohort_markers)
            assert not any(
                len(np.unique(line.get_xdata())) > 2
                for line in axis.lines
                if line.get_linestyle() not in ("None", "none", "")
            )
    finally:
        plt.close(figure)


def test_build_spectral_specificity_figure_jitter_is_deterministic() -> None:
    from studies.pain_study.study1.figures.spectral_specificity_plot import (
        build_spectral_specificity_figure,
    )

    config = load_study1_config()
    first = build_spectral_specificity_figure(spectral_summary(), config)
    second = build_spectral_specificity_figure(spectral_summary(), config)

    try:
        first_offsets = [collection.get_offsets() for collection in first.axes[0].collections[:5]]
        second_offsets = [collection.get_offsets() for collection in second.axes[0].collections[:5]]
        assert all(
            np.allclose(first_offset, second_offset)
            for first_offset, second_offset in zip(
                first_offsets,
                second_offsets,
                strict=True,
            )
        )
    finally:
        plt.close(first)
        plt.close(second)


def test_spectral_specificity_writer_creates_exact_svg_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_spectral_specificity as module

    summary = spectral_summary()
    monkeypatch.setattr(
        module,
        "load_spectral_specificity_summary",
        lambda *args, **kwargs: summary,
    )
    outputs = [
        module.write_spectral_specificity(
            report_path=tmp_path / "report.tsv",
            config=load_study1_config(),
            output_path=tmp_path / directory / "spectral_specificity.svg",
        )
        for directory in ("first", "second")
    ]

    first, second = outputs
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(183.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(84.0, abs=0.01)
    assert first.svg.read_text(encoding="utf-8").count("<text") > 0
    assert pd.read_csv(first.subject_tsv, sep="\t").columns.tolist() == (
        summary.participant_effects.columns.tolist()
    )
    assert pd.read_parquet(first.subject_parquet).columns.tolist() == (
        summary.participant_effects.columns.tolist()
    )
    assert pd.read_csv(first.summary_tsv, sep="\t").columns.tolist() == (
        summary.cohort_effects.columns.tolist()
    )
    assert pd.read_parquet(first.summary_parquet).columns.tolist() == (
        summary.cohort_effects.columns.tolist()
    )
    assert sorted(path.name for path in first.svg.parent.iterdir()) == [
        "spectral_specificity.svg",
        "spectral_specificity_by_subject.parquet",
        "spectral_specificity_by_subject.tsv",
        "spectral_specificity_summary.parquet",
        "spectral_specificity_summary.tsv",
    ]


def test_spectral_specificity_plot_main_writes_only_its_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_spectral_specificity as module

    output = tmp_path / "spectral.svg"
    expected = module.SpectralSpecificityFigurePaths(
        svg=output,
        subject_tsv=tmp_path / "subject.tsv",
        subject_parquet=tmp_path / "subject.parquet",
        summary_tsv=tmp_path / "summary.tsv",
        summary_parquet=tmp_path / "summary.parquet",
    )
    monkeypatch.setattr(module, "load_config", lambda path: load_study1_config())
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "write_spectral_specificity", lambda **kwargs: expected)

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


def test_spectral_specificity_cli_module_help_has_no_runtime_warning(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study1.figures.plot_spectral_specificity",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def spectral_summary() -> SpectralSpecificitySummary:
    participants: list[dict[str, object]] = []
    cohorts: list[dict[str, object]] = []
    for target_index, target in enumerate(("NPS", "SIIPS1")):
        for feature_order, (feature_spec, feature_label) in enumerate(
            zip(FEATURE_SPECS, FEATURE_LABELS, strict=True)
        ):
            delta = np.asarray((-0.12, -0.02, 0.05, 0.14), dtype=float)
            delta += feature_order * 0.015 - target_index * 0.01
            nuisance = np.asarray((-0.30, -0.10, 0.02, 0.08), dtype=float)
            full = nuisance + delta
            common = {
                "target": target,
                "model": "elasticnet",
                "feature_spec": feature_spec,
                "feature_order": feature_order,
                "feature_label": feature_label,
            }
            participants.extend(
                {
                    **common,
                    "fold": fold,
                    "subject_id": f"sub-{fold + 1:02d}",
                    "r2_nuisance": nuisance_r2,
                    "r2": full_r2,
                    "delta_r2": delta_r2,
                    "fold_table_path": "/tmp/model_comparison.tsv",
                }
                for fold, (nuisance_r2, full_r2, delta_r2) in enumerate(
                    zip(nuisance, full, delta, strict=True)
                )
            )
            mean = float(delta.mean())
            cohorts.append(
                {
                    **common,
                    "mean_delta_r2": mean,
                    "ci_low_delta_r2": mean - 0.08,
                    "ci_high_delta_r2": mean + 0.08,
                    "p_value_delta_r2": 0.2,
                    "p_value_delta_r2_holm": 0.5,
                    "n_folds": 4,
                    "n_subjects": 4,
                    "summary_path": "/tmp/summary.json",
                    "fold_table_path": "/tmp/model_comparison.tsv",
                }
            )
    return SpectralSpecificitySummary(
        targets=("NPS", "SIIPS1"),
        feature_specs=FEATURE_SPECS,
        participant_effects=pd.DataFrame(participants),
        cohort_effects=pd.DataFrame(cohorts),
    )
