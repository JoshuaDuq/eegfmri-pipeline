"""Study 1 deep-regression output writing and stage orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from studies.pain_study.study1.cohort import resolve_primary_subjects, study1_output_root
from studies.pain_study.study1.deep_regression.dataset import load_band_tensor_matrix
from studies.pain_study.study1.deep_regression.training import run_loso_deep_regression
from studies.pain_study.study1.targets import PRIMARY_SIGNATURES


def _deep_results_root(
    config: Any,
    *,
    target_name: str,
    preset_name: str,
) -> Path:
    return study1_output_root(config) / "deep_regression" / target_name / preset_name


def _preset_mapping(config: Any) -> dict[str, list[str]]:
    raw = config.get("study1.deep_regression.presets", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError("study1.deep_regression.presets must be a non-empty mapping.")

    presets: dict[str, list[str]] = {}
    for preset_name, bands in raw.items():
        if not isinstance(bands, list) or not bands:
            raise ValueError(
                f"study1.deep_regression.presets.{preset_name} must be a non-empty list of bands."
            )
        presets[str(preset_name).strip()] = [str(band).strip() for band in bands if str(band).strip()]
    return presets


def _write_deep_outputs(
    *,
    result_dir: Path,
    result: Any,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_tsv = result_dir / "predictions.tsv"
    predictions_parquet = result_dir / "predictions.parquet"
    fold_metrics_tsv = result_dir / "fold_metrics.tsv"
    summary_json = result_dir / "summary.json"

    write_tsv(result.predictions, predictions_tsv)
    write_parquet(result.predictions, predictions_parquet)
    write_tsv(result.fold_metrics, fold_metrics_tsv)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2)
    return summary_json


def run_deep_regression(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    logger: logging.Logger | None = None,
) -> list[Path]:
    if logger is None:
        logger = logging.getLogger(__name__)

    resolved_subjects = resolve_primary_subjects(
        requested_subjects=subjects,
        task=task,
        config=config,
    )
    outputs: list[Path] = []
    for target_name in PRIMARY_SIGNATURES:
        for preset_name, bands in _preset_mapping(config).items():
            X, y, groups, _channels, meta = load_band_tensor_matrix(
                subjects=resolved_subjects,
                task=task,
                config=config,
                target_name=target_name,
                bands=bands,
                logger=logger,
            )
            result = run_loso_deep_regression(
                X=X,
                y=y,
                groups=groups,
                meta=meta,
                target_name=target_name,
                preset_name=preset_name,
                bands=bands,
                config=config,
                logger=logger,
            )
            outputs.append(
                _write_deep_outputs(
                    result_dir=_deep_results_root(
                        config,
                        target_name=target_name,
                        preset_name=preset_name,
                    ),
                    result=result,
                )
            )
    return outputs


__all__ = ["run_deep_regression"]
