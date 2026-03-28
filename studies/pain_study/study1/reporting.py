"""Study 1 report aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from studies.pain_study.study1.cohort import study1_output_root


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _feature_records(config: Any) -> list[dict[str, Any]]:
    root = study1_output_root(config) / "feature_benchmark"
    if not root.exists():
        return []

    records: list[dict[str, Any]] = []
    pattern = "*/" + "*/" + "*/model_comparison/metrics/model_comparison_summary.json"
    for summary_path in sorted(root.glob(pattern)):
        partition = summary_path.parents[4].name
        target_name = summary_path.parents[3].name
        feature_spec = summary_path.parents[2].name
        payload = _read_json(summary_path)
        for model_name, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            if "mean_r2" not in metrics and "mean_mae" not in metrics:
                continue
            records.append(
                {
                    "lane": "feature_benchmark",
                    "analysis_partition": partition,
                    "target": target_name,
                    "feature_spec": feature_spec,
                    "model": model_name,
                    "mean_r2": metrics.get("mean_r2"),
                    "mean_mae": metrics.get("mean_mae"),
                    "n_folds": metrics.get("n_folds"),
                    "summary_path": str(summary_path),
                }
            )
    return records


def _deep_records(config: Any) -> list[dict[str, Any]]:
    root = study1_output_root(config) / "deep_regression"
    if not root.exists():
        return []

    records: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        target_name = summary_path.parents[1].name
        feature_spec = summary_path.parent.name
        payload = _read_json(summary_path)
        records.append(
            {
                "lane": "deep_regression",
                "analysis_partition": "primary",
                "target": target_name,
                "feature_spec": feature_spec,
                "model": str(payload.get("model_name", "band_temporal_regressor")),
                "mean_r2": payload.get("mean_r2"),
                "mean_mae": payload.get("mean_mae"),
                "n_folds": payload.get("n_folds"),
                "summary_path": str(summary_path),
            }
        )
    return records


def write_study1_report(
    *,
    task: str,
    config: Any,
    logger: logging.Logger | None = None,
) -> Path:
    if logger is None:
        logger = logging.getLogger(__name__)

    report_root = study1_output_root(config) / "reports"
    feature_records = _feature_records(config)
    deep_records = _deep_records(config)
    all_records = feature_records + deep_records
    if not all_records:
        raise FileNotFoundError(
            "No Study 1 feature-benchmark or deep-regression summaries were found. "
            "Run 'feature-benchmark' and/or 'deep-regression' first."
        )

    frame = pd.DataFrame(all_records).sort_values(
        by=["lane", "analysis_partition", "target", "feature_spec", "model"],
        kind="stable",
    )
    summary_payload = {
        "task": task,
        "n_records": int(len(frame)),
        "lanes": sorted(set(frame["lane"].astype(str).tolist())),
        "targets": sorted(set(frame["target"].astype(str).tolist())),
    }

    tsv_path = report_root / "study1_report.tsv"
    parquet_path = report_root / "study1_report.parquet"
    json_path = report_root / "study1_report.json"
    write_tsv(frame, tsv_path)
    write_parquet(frame, parquet_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    logger.info("Wrote Study 1 report to %s", tsv_path)
    return tsv_path


__all__ = ["write_study1_report"]
