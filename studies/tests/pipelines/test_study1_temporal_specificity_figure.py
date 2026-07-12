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
from studies.pain_study.study1.figures.temporal_specificity import (
    TemporalSpecificitySummary,
)

WINDOWS = (
    "prestimulus_wide",
    "immediate_prestimulus",
    "ramp_up",
    "early_plateau",
    "mid_plateau",
    "late_plateau",
)
LABELS = tuple(f"Window {index + 1}" for index in range(6))


def test_build_temporal_specificity_figure_has_publication_structure() -> None:
    from studies.pain_study.study1.figures.temporal_specificity_plot import (
        build_temporal_specificity_figure,
    )

    figure = build_temporal_specificity_figure(_summary(), load_study1_config())

    try:
        assert [axis.get_title() for axis in figure.axes] == ["NPS", "SIIPS1"]
        assert len(figure.axes) == 2
        assert len(figure.legends) == 1
        assert figure.legends[0]._ncols == 2
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == list(LABELS)
        assert not figure.axes[1].get_yticklabels()
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 88.0 / 25.4))
    finally:
        plt.close(figure)


def test_build_temporal_specificity_figure_uses_shared_symmetric_effect_scale() -> None:
    from studies.pain_study.study1.figures.temporal_specificity_plot import (
        build_temporal_specificity_figure,
    )

    figure = build_temporal_specificity_figure(_summary(), load_study1_config())

    try:
        nps_limits = figure.axes[0].get_xlim()
        siips_limits = figure.axes[1].get_xlim()
        assert np.allclose(nps_limits, siips_limits)
        assert np.isclose(nps_limits[0], -nps_limits[1])
        assert nps_limits[1] > 0.17
        for axis in figure.axes:
            assert any(
                np.allclose(line.get_xdata(), (0.0, 0.0))
                for line in axis.lines
                if len(line.get_xdata()) == 2
            )
    finally:
        plt.close(figure)


def test_build_temporal_specificity_figure_uses_target_colors_and_no_mean_trace() -> None:
    from matplotlib.colors import to_hex

    from studies.pain_study.study1.figures.temporal_specificity_plot import (
        build_temporal_specificity_figure,
    )

    figure = build_temporal_specificity_figure(_summary(), load_study1_config())

    try:
        expected_colors = ("#0072b2", "#d55e00")
        for axis, expected_color in zip(figure.axes, expected_colors, strict=True):
            cohort_markers = [line for line in axis.lines if line.get_marker() == "D"]
            assert len(cohort_markers) == 6
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


def test_build_temporal_specificity_figure_jitter_is_deterministic() -> None:
    from studies.pain_study.study1.figures.temporal_specificity_plot import (
        build_temporal_specificity_figure,
    )

    config = load_study1_config()
    first = build_temporal_specificity_figure(_summary(), config)
    second = build_temporal_specificity_figure(_summary(), config)

    try:
        first_offsets = [collection.get_offsets() for collection in first.axes[0].collections[:6]]
        second_offsets = [collection.get_offsets() for collection in second.axes[0].collections[:6]]
        assert all(
            np.allclose(first_offset, second_offset)
            for first_offset, second_offset in zip(first_offsets, second_offsets, strict=True)
        )
    finally:
        plt.close(first)
        plt.close(second)


def test_temporal_specificity_writer_creates_exact_svg_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_temporal_specificity as module

    summary = _summary()
    monkeypatch.setattr(
        module,
        "load_temporal_specificity_summary",
        lambda *args, **kwargs: summary,
    )
    outputs = [
        module.write_temporal_specificity(
            report_path=tmp_path / "report.tsv",
            config=load_study1_config(),
            output_path=tmp_path / directory / "temporal_specificity.svg",
        )
        for directory in ("first", "second")
    ]

    first, second = outputs
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(183.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(88.0, abs=0.01)
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
        "temporal_specificity.svg",
            "temporal_specificity_by_subject.parquet",
            "temporal_specificity_by_subject.tsv",
            "temporal_specificity_primary_minus_control_by_subject.parquet",
            "temporal_specificity_primary_minus_control_by_subject.tsv",
            "temporal_specificity_primary_minus_control_summary.parquet",
            "temporal_specificity_primary_minus_control_summary.tsv",
            "temporal_specificity_summary.parquet",
        "temporal_specificity_summary.tsv",
    ]


def test_temporal_specificity_plot_main_writes_only_its_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_temporal_specificity as module

    output = tmp_path / "temporal.svg"
    expected = module.TemporalSpecificityFigurePaths(
        svg=output,
        subject_tsv=tmp_path / "subject.tsv",
        subject_parquet=tmp_path / "subject.parquet",
        summary_tsv=tmp_path / "summary.tsv",
        summary_parquet=tmp_path / "summary.parquet",
        matched_subject_tsv=tmp_path / "matched_subject.tsv",
        matched_subject_parquet=tmp_path / "matched_subject.parquet",
        matched_summary_tsv=tmp_path / "matched_summary.tsv",
        matched_summary_parquet=tmp_path / "matched_summary.parquet",
    )
    monkeypatch.setattr(module, "load_config", lambda path: load_study1_config())
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "write_temporal_specificity", lambda **kwargs: expected)

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


def test_temporal_specificity_cli_module_help_has_no_runtime_warning(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study1.figures.plot_temporal_specificity",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def _summary() -> TemporalSpecificitySummary:
    participant_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(("NPS", "SIIPS1")):
        for window_order, (window, label) in enumerate(zip(WINDOWS, LABELS, strict=True)):
            effects = np.asarray((-0.08, -0.01, 0.07, 0.11), dtype=float)
            effects += target_index * 0.02 + window_order * 0.008
            common = {
                "target": target,
                "model": "elasticnet",
                "window_name": window,
                "window_kind": "test",
                "window_start_s": float(window_order),
                "window_end_s": float(window_order + 1),
                "window_order": window_order,
                "window_label": label,
            }
            participant_rows.extend(
                {
                    **common,
                    "fold": subject_index,
                    "subject_id": f"sub-{subject_index + 1:02d}",
                    "delta_r2": effect,
                }
                for subject_index, effect in enumerate(effects)
            )
            mean = float(effects.mean())
            cohort_rows.append(
                {
                    **common,
                    "feature_spec": f"temporal_{window}",
                    "mean_delta_r2": mean,
                    "ci_low_delta_r2": mean - 0.07,
                    "ci_high_delta_r2": mean + 0.07,
                    "p_value_delta_r2_holm": 0.5,
                    "n_folds": 4,
                    "n_subjects": 4,
                    "summary_path": "/tmp/summary.json",
                    "fold_table_path": "/tmp/model_comparison.tsv",
                }
            )
    return TemporalSpecificitySummary(
        targets=("NPS", "SIIPS1"),
        windows=WINDOWS,
        participant_effects=pd.DataFrame(participant_rows),
        cohort_effects=pd.DataFrame(cohort_rows),
    )
