"""Study 1 report aggregation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import get_config_value
from studies.pain_study.study1.cohort import study1_output_root
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.targets import PRIMARY_SIGNATURES


FEATURE_MODELS = ("elasticnet", "ridge")
PRIMARY_GATE_TARGET = "NPS"
PRIMARY_GATE_FEATURE_SPEC = "alpha_beta"
PRIMARY_GATE_MODEL = "elasticnet"
PRIMARY_REQUIRED_NUMERIC_FIELDS = (
    "mean_delta_r2",
    "ci_low_delta_r2",
    "ci_high_delta_r2",
    "overall_r2",
    "p_value_delta_r2",
    "n_perm",
    "n_perm_completed",
    "n_perm_attempted",
    "n_invalid_permutations",
    "n_folds",
    "n_subjects_requested",
    "n_subjects_included",
    "n_subjects_excluded",
    "subject_excluded_fraction",
)
INTERPRETATION_DIAGNOSTIC_FIELDS = (
    "target_split_half_reliability",
    "target_reliability_n_trials",
    "precision_flag_passed",
    "level2_mean_delta_r2",
    "within_subject_centered_delta_r2",
    "temporal_negative_controls_passed",
    "artifact_censoring_robustness_passed",
    "hrf_timing_robustness_passed",
    "first_exposure_robustness_passed",
    "baseline_robustness_passed",
    "smoothing_robustness_passed",
)
SOURCE_ENTRY_DIAGNOSTIC_FIELDS = (
    "target_split_half_reliability",
    "target_reliability_n_trials",
    "level2_mean_delta_r2",
    "within_subject_centered_delta_r2",
    "temporal_negative_controls_passed",
    "artifact_censoring_robustness_passed",
)
PRIMARY_P_VALUE_ALPHA = 0.05
SOURCE_ENTRY_MIN_DELTA_R2 = 0.02
SOURCE_ENTRY_MIN_DELTA_R2_LOWER_CI = 0.005
SOURCE_ENTRY_MIN_LEVEL2_DELTA_R2 = 0.005
SOURCE_ENTRY_MIN_TARGET_RELIABILITY = 0.4
MIN_TARGET_RELIABILITY_TRIALS = 30


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _best_params_by_fold(summary_path: Path, *, model_name: str) -> tuple[str, int]:
    fold_path = summary_path.parent / "model_comparison.tsv"
    if not fold_path.exists():
        raise FileNotFoundError(f"Study 1 report requires fold-level model table: {fold_path}")

    frame = pd.read_csv(fold_path, sep="\t")
    required_columns = {"model", "fold", "test_subject", "best_params"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Study 1 fold-level model table is missing columns: {missing}")

    rows = frame.loc[frame["model"].astype(str) == model_name].copy()
    if rows.empty:
        raise ValueError(f"Study 1 fold-level model table has no rows for model {model_name!r}.")

    rows = rows.sort_values(["fold", "test_subject"], kind="stable")
    payload = [
        {
            "fold": int(row["fold"]),
            "test_subject": str(row["test_subject"]),
            "best_params": str(row["best_params"]),
        }
        for _, row in rows.iterrows()
    ]
    unique_params = sorted(set(rows["best_params"].astype(str).tolist()))
    return json.dumps(payload, sort_keys=True), len(unique_params)


def _claim_tier(
    *,
    lane: str,
    partition: str,
    target_name: str,
    feature_spec: str,
    model_name: str,
) -> str:
    if lane == "feature_benchmark" and partition == "primary":
        is_primary_gate = (
            target_name == PRIMARY_GATE_TARGET
            and feature_spec == PRIMARY_GATE_FEATURE_SPEC
            and model_name == PRIMARY_GATE_MODEL
        )
        if is_primary_gate:
            return "primary_gate"
        return "secondary_confirmatory"
    return "exploratory"


def _join_labels(labels: list[str]) -> str:
    return ";".join(labels) if labels else "none"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_float(record: pd.Series, field: str) -> float | None:
    value = record.get(field)
    if _is_missing(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Study 1 report diagnostic field '{field}' must be numeric.")
    return float(numeric)


def _optional_bool(record: pd.Series, field: str) -> bool | None:
    value = record.get(field)
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"Study 1 report diagnostic field '{field}' must be boolean.")


def _primary_prediction_status(record: pd.Series) -> str:
    if str(record.get("claim_tier", "")) != "primary_gate":
        return "not_primary_gate"

    delta_r2 = _optional_float(record, "mean_delta_r2")
    p_value = _optional_float(record, "p_value_delta_r2_holm")
    if delta_r2 is None or p_value is None:
        return "primary_prediction_not_evaluated"
    if delta_r2 > 0.0 and p_value <= PRIMARY_P_VALUE_ALPHA:
        return "primary_prediction_positive"
    return "primary_prediction_not_supported"


def _missing_interpretation_diagnostics(record: pd.Series) -> str:
    if str(record.get("lane", "")) != "feature_benchmark":
        return "not_applicable"
    missing = [
        field
        for field in INTERPRETATION_DIAGNOSTIC_FIELDS
        if field not in record.index or _is_missing(record.get(field))
    ]
    return _join_labels(missing)


def _interpretation_flags(record: pd.Series) -> str:
    if str(record.get("lane", "")) != "feature_benchmark":
        return "not_applicable"

    flags: list[str] = []
    target_reliability = _optional_float(record, "target_split_half_reliability")
    reliability_trials = _optional_float(record, "target_reliability_n_trials")
    if (
        target_reliability is not None
        and target_reliability < SOURCE_ENTRY_MIN_TARGET_RELIABILITY
    ) or (
        reliability_trials is not None
        and reliability_trials < MIN_TARGET_RELIABILITY_TRIALS
    ):
        flags.append("target_reliability_limited")

    precision_passed = _optional_bool(record, "precision_flag_passed")
    if precision_passed is False:
        flags.append("precision_limited")

    level2_delta_r2 = _optional_float(record, "level2_mean_delta_r2")
    if (
        level2_delta_r2 is not None
        and level2_delta_r2 < SOURCE_ENTRY_MIN_LEVEL2_DELTA_R2
    ):
        flags.append("level2_convergence_limited")

    within_subject_delta_r2 = _optional_float(record, "within_subject_centered_delta_r2")
    if within_subject_delta_r2 is not None and within_subject_delta_r2 <= 0.0:
        flags.append("within_subject_tracking_limited")

    temporal_passed = _optional_bool(record, "temporal_negative_controls_passed")
    if temporal_passed is False:
        flags.append("temporal_specificity_limited")

    artifact_passed = _optional_bool(record, "artifact_censoring_robustness_passed")
    if artifact_passed is False:
        flags.append("artifact_robustness_limited")

    robustness_fields = (
        ("hrf_timing_robustness_passed", "hrf_timing_robustness_limited"),
        ("first_exposure_robustness_passed", "first_exposure_robustness_limited"),
        ("baseline_robustness_passed", "baseline_robustness_limited"),
        ("smoothing_robustness_passed", "smoothing_robustness_limited"),
    )
    for field, flag in robustness_fields:
        passed = _optional_bool(record, field)
        if passed is False:
            flags.append(flag)

    return _join_labels(flags)


def _study2_source_entry_status(record: pd.Series) -> str:
    if str(record.get("claim_tier", "")) != "primary_gate":
        return "not_primary_gate"
    if _primary_prediction_status(record) != "primary_prediction_positive":
        return "source_interpretation_exploratory"

    required_fields = (
        "mean_delta_r2",
        "ci_low_delta_r2",
        *SOURCE_ENTRY_DIAGNOSTIC_FIELDS,
    )
    missing = [
        field
        for field in required_fields
        if field not in record.index or _is_missing(record.get(field))
    ]
    if missing:
        return "source_entry_not_evaluated"

    failed = (
        _optional_float(record, "mean_delta_r2") < SOURCE_ENTRY_MIN_DELTA_R2
        or _optional_float(record, "ci_low_delta_r2") <= SOURCE_ENTRY_MIN_DELTA_R2_LOWER_CI
        or _optional_float(record, "level2_mean_delta_r2")
        < SOURCE_ENTRY_MIN_LEVEL2_DELTA_R2
        or _optional_float(record, "target_split_half_reliability")
        < SOURCE_ENTRY_MIN_TARGET_RELIABILITY
        or _optional_float(record, "target_reliability_n_trials")
        < MIN_TARGET_RELIABILITY_TRIALS
        or _optional_float(record, "within_subject_centered_delta_r2") <= 0.0
        or _optional_bool(record, "temporal_negative_controls_passed") is False
        or _optional_bool(record, "artifact_censoring_robustness_passed") is False
    )
    if failed:
        return "source_interpretation_exploratory"
    return "source_interpretation_confirmatory"


def _append_interpretation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for field in INTERPRETATION_DIAGNOSTIC_FIELDS:
        if field not in out.columns:
            out[field] = pd.NA

    feature_primary = (
        (out["lane"].astype(str) == "feature_benchmark")
        & (out["analysis_partition"].astype(str) == "primary")
    )
    out["analysis_validity_status"] = "not_primary_analysis"
    out.loc[feature_primary, "analysis_validity_status"] = "analysis_valid"
    out["primary_prediction_status"] = out.apply(_primary_prediction_status, axis=1)
    out["missing_interpretation_diagnostics"] = out.apply(
        _missing_interpretation_diagnostics,
        axis=1,
    )
    out["interpretation_flags"] = out.apply(_interpretation_flags, axis=1)
    out["study2_source_entry_status"] = out.apply(_study2_source_entry_status, axis=1)
    return out


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
        subject_selection = payload.get("subject_selection", {})
        if subject_selection is None:
            subject_selection = {}
        if not isinstance(subject_selection, dict):
            raise ValueError(f"Expected subject_selection object in {summary_path}.")
        for model_name, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            if "mean_r2" not in metrics and "mean_mae" not in metrics:
                continue
            best_params, n_unique_best_params = _best_params_by_fold(
                summary_path,
                model_name=str(model_name),
            )
            records.append(
                {
                    "lane": "feature_benchmark",
                    "analysis_partition": partition,
                    "claim_tier": _claim_tier(
                        lane="feature_benchmark",
                        partition=partition,
                        target_name=target_name,
                        feature_spec=feature_spec,
                        model_name=str(model_name),
                    ),
                    "target": target_name,
                    "feature_spec": feature_spec,
                    "model": model_name,
                    "mean_r2": metrics.get("mean_r2"),
                    "std_r2": metrics.get("std_r2"),
                    "ci_low_r2": metrics.get("ci_low_r2"),
                    "ci_high_r2": metrics.get("ci_high_r2"),
                    "mean_nuisance_r2": metrics.get("mean_nuisance_r2"),
                    "std_nuisance_r2": metrics.get("std_nuisance_r2"),
                    "mean_delta_r2": metrics.get("mean_delta_r2"),
                    "std_delta_r2": metrics.get("std_delta_r2"),
                    "ci_low_delta_r2": metrics.get("ci_low_delta_r2"),
                    "ci_high_delta_r2": metrics.get("ci_high_delta_r2"),
                    "overall_r2": metrics.get("overall_r2"),
                    "mean_mae": metrics.get("mean_mae"),
                    "std_mae": metrics.get("std_mae"),
                    "ci_low_mae": metrics.get("ci_low_mae"),
                    "ci_high_mae": metrics.get("ci_high_mae"),
                    "mean_nuisance_mae": metrics.get("mean_nuisance_mae"),
                    "p_value_r2": metrics.get("p_value_r2"),
                    "p_value_delta_r2": metrics.get("p_value_delta_r2"),
                    "n_perm": metrics.get("n_perm"),
                    "n_perm_requested": metrics.get("n_perm_requested"),
                    "n_perm_completed": metrics.get("n_perm_completed"),
                    "n_perm_attempted": metrics.get("n_perm_attempted"),
                    "n_invalid_permutations": metrics.get("n_invalid_permutations"),
                    "n_folds": metrics.get("n_folds"),
                    "n_subjects_requested": subject_selection.get("n_requested"),
                    "n_subjects_included": subject_selection.get("n_included"),
                    "n_subjects_excluded": subject_selection.get("n_excluded"),
                    "subject_excluded_fraction": subject_selection.get("excluded_fraction"),
                    "best_params_by_fold": best_params,
                    "n_unique_best_params": n_unique_best_params,
                    "summary_path": str(summary_path),
                    **{
                        field: metrics.get(field)
                        for field in INTERPRETATION_DIAGNOSTIC_FIELDS
                    },
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
                "claim_tier": "exploratory",
                "target": target_name,
                "feature_spec": feature_spec,
                "model": str(payload.get("model_name", "band_temporal_regressor")),
                "mean_r2": payload.get("mean_r2"),
                "mean_mae": payload.get("mean_mae"),
                "p_value_r2": payload.get("p_value_r2"),
                "n_folds": payload.get("n_folds"),
                "summary_path": str(summary_path),
            }
        )
    return records


def _validate_complete_primary_outputs(
    *,
    records: list[dict[str, Any]],
    config: Any,
) -> None:
    present = {
        (
            str(record.get("lane", "")),
            str(record.get("analysis_partition", "")),
            str(record.get("target", "")),
            str(record.get("feature_spec", "")),
            str(record.get("model", "")),
        )
        for record in records
    }

    expected: set[tuple[str, str, str, str, str]] = set()
    for target_name in PRIMARY_SIGNATURES:
        for feature_spec in PRIMARY_BAND_PRESETS:
            for model_name in FEATURE_MODELS:
                expected.add(
                    (
                        "feature_benchmark",
                        "primary",
                        target_name,
                        feature_spec,
                        model_name,
                    )
                )
    missing = sorted(expected - present)
    if missing:
        examples = "; ".join("/".join(item) for item in missing[:8])
        raise FileNotFoundError(
            "Study 1 report is missing prespecified Study 1 outputs. "
            f"Missing {len(missing)} required record(s): {examples}"
        )
    expected_n_perm = get_config_value(config, "study1.feature_benchmark.n_perm", None)
    for record in records:
        key = (
            str(record.get("lane", "")),
            str(record.get("analysis_partition", "")),
            str(record.get("target", "")),
            str(record.get("feature_spec", "")),
            str(record.get("model", "")),
        )
        if key not in expected:
            continue
        for field in PRIMARY_REQUIRED_NUMERIC_FIELDS:
            value = pd.to_numeric(pd.Series([record.get(field)]), errors="coerce").iloc[0]
            if pd.isna(value):
                raise ValueError(
                    "Study 1 primary feature report is missing required numeric field "
                    f"{field!r} for {'/'.join(key)}."
                )
        if expected_n_perm is not None:
            completed = int(float(record["n_perm_completed"]))
            if completed != int(expected_n_perm):
                raise ValueError(
                    "Study 1 primary feature report has the wrong valid permutation count for "
                    f"{'/'.join(key)}: completed={completed}, expected={int(expected_n_perm)}."
                )
            _validate_permutation_budget(
                record=record,
                config=config,
                key=key,
                expected_n_perm=int(expected_n_perm),
            )


def _validate_permutation_budget(
    *,
    record: dict[str, Any],
    config: Any,
    key: tuple[str, str, str, str, str],
    expected_n_perm: int,
) -> None:
    completed = int(float(record["n_perm_completed"]))
    attempted = int(float(record["n_perm_attempted"]))
    invalid = int(float(record["n_invalid_permutations"]))
    label = "/".join(key)
    if attempted < completed:
        raise ValueError(
            "Study 1 primary feature report has an invalid permutation accounting "
            f"record for {label}: attempted={attempted}, completed={completed}."
        )
    if invalid != attempted - completed:
        raise ValueError(
            "Study 1 primary feature report has an invalid permutation accounting "
            f"record for {label}: invalid={invalid}, attempted-completed={attempted - completed}."
        )

    raw_fraction = get_config_value(
        config,
        "study1.feature_benchmark.max_invalid_permutation_fraction",
        None,
    )
    if raw_fraction is None:
        return
    max_invalid_fraction = float(raw_fraction)
    if not 0.0 <= max_invalid_fraction < 1.0:
        raise ValueError(
            "study1.feature_benchmark.max_invalid_permutation_fraction must be in [0, 1)."
        )

    max_attempts = int(math.ceil(expected_n_perm / (1.0 - max_invalid_fraction)))
    if attempted > max_attempts:
        raise ValueError(
            "Study 1 primary feature report exceeds the invalid permutation budget for "
            f"{label}: attempted={attempted}, maximum={max_attempts}, "
            f"valid={completed}, invalid={invalid}, "
            f"max_invalid_fraction={max_invalid_fraction}."
        )


def _append_primary_feature_multiplicity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_value_r2_holm"] = pd.NA
    out["p_value_delta_r2_holm"] = pd.NA
    primary_mask = (
        (out["lane"].astype(str) == "feature_benchmark")
        & (out["analysis_partition"].astype(str) == "primary")
    )
    try:
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:
        raise RuntimeError("Study 1 report multiplicity correction requires statsmodels.") from exc

    for raw_column, adjusted_column in (
        ("p_value_r2", "p_value_r2_holm"),
        ("p_value_delta_r2", "p_value_delta_r2_holm"),
    ):
        if raw_column not in out.columns:
            continue
        for claim_tier in ("primary_gate", "secondary_confirmatory"):
            mask = primary_mask & (out["claim_tier"].astype(str) == claim_tier)
            p_values = pd.to_numeric(out.loc[mask, raw_column], errors="coerce")
            valid = p_values.notna()
            if not valid.any():
                continue
            values = p_values.loc[valid].to_numpy(dtype=float)
            adjusted = multipletests(values, method="holm")[1]
            out.loc[p_values.loc[valid].index, adjusted_column] = adjusted
    return out


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
    _validate_complete_primary_outputs(records=all_records, config=config)

    frame = pd.DataFrame(all_records).sort_values(
        by=["lane", "analysis_partition", "target", "feature_spec", "model"],
        kind="stable",
    )
    frame = _append_primary_feature_multiplicity(frame)
    frame = _append_interpretation_columns(frame)
    summary_payload = {
        "task": task,
        "n_records": int(len(frame)),
        "lanes": sorted(set(frame["lane"].astype(str).tolist())),
        "claim_tiers": sorted(set(frame["claim_tier"].astype(str).tolist())),
        "primary_prediction_statuses": sorted(
            set(frame["primary_prediction_status"].astype(str).tolist())
        ),
        "study2_source_entry_statuses": sorted(
            set(frame["study2_source_entry_status"].astype(str).tolist())
        ),
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
