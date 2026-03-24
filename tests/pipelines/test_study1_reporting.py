from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.pipelines_test_utils import DotConfig


def _config(root: Path) -> DotConfig:
    return DotConfig({"paths": {"deriv_root": str(root / "derivatives")}, "study1": {"outputs": {"root_name": "study1"}}})


def test_write_study1_report_aggregates_feature_and_deep_summaries(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1"

    feature_summary = root / "feature_benchmark" / "primary" / "NPS" / "alpha" / "model_comparison" / "metrics"
    feature_summary.mkdir(parents=True, exist_ok=True)
    with open(feature_summary / "model_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"ridge": {"mean_r2": 0.4, "mean_mae": 0.2, "n_folds": 2}}, handle)

    exploratory_summary = root / "feature_benchmark" / "exploratory" / "NPS" / "spectral" / "model_comparison" / "metrics"
    exploratory_summary.mkdir(parents=True, exist_ok=True)
    with open(exploratory_summary / "model_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"rf": {"mean_r2": 0.2, "mean_mae": 0.3, "n_folds": 2}}, handle)

    deep_summary = root / "deep_regression" / "SIIPS1" / "alpha_beta_gamma"
    deep_summary.mkdir(parents=True, exist_ok=True)
    with open(deep_summary / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"model_name": "band_temporal_regressor", "mean_r2": 0.5, "mean_mae": 0.1, "n_folds": 2}, handle)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert len(report) == 3
    assert set(report["lane"]) == {"feature_benchmark", "deep_regression"}
    assert set(report["analysis_partition"]) == {"primary", "exploratory"}
    assert set(report["feature_spec"]) == {"alpha", "spectral", "alpha_beta_gamma"}
