"""Tests for studies.pain_study.study1.reporting aggregation and multiplicity correction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


def _config(root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(root / "derivatives")},
            "study1": {
                "outputs": {"root_name": "study1"},
                "targets": {"names": ["NPS", "SIIPS1"]},
                "deep_regression": {
                    "presets": {
                        "alpha": ["alpha"],
                        "beta": ["beta"],
                        "alpha_beta": ["alpha", "beta"],
                    }
                },
            },
        }
    )


def _study1_root(config: DotConfig) -> Path:
    return Path(config.get("paths.deriv_root")) / "group" / "multimodal" / "study1"


def _write_feature_summary(
    root: Path,
    target: str,
    feature_spec: str,
    *,
    partition: str = "primary",
    p_value_r2: float | None = 0.05,
    p_value_delta_r2: float | None = 0.01,
) -> None:
    feature_summary = (
        root / "feature_benchmark" / partition / target / feature_spec
        / "model_comparison" / "metrics"
    )
    feature_summary.mkdir(parents=True, exist_ok=True)
    payload = {
        "elasticnet": {
            "mean_r2": 0.4,
            "mean_delta_r2": 0.12,
            "p_value_r2": p_value_r2,
            "p_value_delta_r2": p_value_delta_r2,
            "mean_mae": 0.2,
            "n_folds": 4,
        },
        "ridge": {
            "mean_r2": 0.3,
            "mean_delta_r2": 0.08,
            "p_value_r2": p_value_r2,
            "p_value_delta_r2": p_value_delta_r2,
            "mean_mae": 0.3,
            "n_folds": 4,
        },
    }
    with open(feature_summary / "model_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _write_deep_summary(root: Path, target: str, preset: str) -> None:
    deep_summary = root / "deep_regression" / target / preset
    deep_summary.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "band_temporal_regressor",
        "mean_r2": 0.5,
        "mean_mae": 0.1,
        "n_folds": 4,
    }
    with open(deep_summary / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _write_complete_outputs(root: Path, config: DotConfig) -> None:
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec)
            _write_deep_summary(root, target, feature_spec)


###################################################################
# write_study1_report
###################################################################


def test_report_writes_three_output_formats(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)

    assert report_path.name == "study1_report.tsv"
    report_dir = report_path.parent
    assert (report_dir / "study1_report.parquet").exists()
    assert (report_dir / "study1_report.json").exists()


def test_report_json_contains_summary_metadata(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    json_path = report_path.parent / "study1_report.json"
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert summary["task"] == "pain"
    assert summary["n_records"] > 0
    assert "feature_benchmark" in summary["lanes"]
    assert "deep_regression" in summary["lanes"]
    assert set(summary["targets"]) == {"NPS", "SIIPS1"}


def test_report_rejects_empty_results(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="No Study 1"):
        write_study1_report(task="pain", config=cfg)


###################################################################
# Holm multiplicity correction
###################################################################


def test_report_holm_corrected_p_values_are_present(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert "p_value_r2_holm" in report.columns
    assert "p_value_delta_r2_holm" in report.columns

    primary_feature = report.loc[
        (report["lane"] == "feature_benchmark")
        & (report["analysis_partition"] == "primary")
    ]
    assert primary_feature["p_value_r2_holm"].notna().any()
    assert primary_feature["p_value_delta_r2_holm"].notna().any()


def test_report_holm_correction_inflates_p_values(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    primary = report.loc[
        (report["lane"] == "feature_benchmark")
        & (report["analysis_partition"] == "primary")
    ]
    raw = pd.to_numeric(primary["p_value_r2"], errors="coerce").dropna()
    adjusted = pd.to_numeric(primary["p_value_r2_holm"], errors="coerce").dropna()

    assert len(adjusted) == len(raw)
    assert (adjusted >= raw.values[: len(adjusted)]).all()


###################################################################
# _validate_complete_primary_outputs
###################################################################


def test_validate_detects_missing_deep_regression_preset(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)

    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec)
        _write_deep_summary(root, target, "alpha")

    with pytest.raises(FileNotFoundError, match="missing prespecified"):
        write_study1_report(task="pain", config=cfg)


###################################################################
# Exploratory records included alongside primary
###################################################################


def test_report_includes_exploratory_records(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    _write_feature_summary(
        root, "NPS", "spectral", partition="exploratory"
    )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    exploratory = report.loc[report["analysis_partition"] == "exploratory"]
    assert len(exploratory) >= 2
    assert set(exploratory["feature_spec"]) == {"spectral"}
