from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def _write_feature_summary(root: Path, target: str, feature_spec: str) -> None:
    feature_summary = (
        root
        / "feature_benchmark"
        / "primary"
        / target
        / feature_spec
        / "model_comparison"
        / "metrics"
    )
    feature_summary.mkdir(parents=True, exist_ok=True)
    payload = {
        "elasticnet": {
            "mean_r2": 0.4,
            "mean_delta_r2": 0.12,
            "p_value_delta_r2": 0.01,
            "mean_mae": 0.2,
            "n_folds": 4,
        },
        "ridge": {
            "mean_r2": 0.3,
            "mean_delta_r2": 0.08,
            "p_value_delta_r2": 0.03,
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


def test_write_study1_report_aggregates_feature_and_deep_summaries(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1"

    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec)
            _write_deep_summary(root, target, feature_spec)

    exploratory_summary = (
        root
        / "feature_benchmark"
        / "exploratory"
        / "NPS"
        / "spectral"
        / "model_comparison"
        / "metrics"
    )
    exploratory_summary.mkdir(parents=True, exist_ok=True)
    with open(
        exploratory_summary / "model_comparison_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump({"rf": {"mean_r2": 0.2, "mean_mae": 0.3, "n_folds": 2}}, handle)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert len(report) == 19
    assert set(report["lane"]) == {"feature_benchmark", "deep_regression"}
    assert set(report["analysis_partition"]) == {"primary", "exploratory"}
    assert set(report["feature_spec"]) == {"alpha", "beta", "alpha_beta", "spectral"}
    assert "mean_delta_r2" in report.columns
    assert "p_value_delta_r2" in report.columns
    primary_feature_models = set(
        report.loc[
            (report["lane"] == "feature_benchmark")
            & (report["analysis_partition"] == "primary"),
            "model",
        ]
    )
    assert primary_feature_models == {"elasticnet", "ridge"}


def test_write_study1_report_rejects_incomplete_primary_outputs(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1"
    _write_feature_summary(root, "NPS", "alpha")

    try:
        write_study1_report(task="pain", config=cfg)
    except FileNotFoundError as exc:
        assert "missing prespecified Study 1 outputs" in str(exc)
    else:
        raise AssertionError("Expected incomplete Study 1 outputs to be rejected.")
