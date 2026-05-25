"""Study 1 report aggregation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import find_clean_events_path
from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import get_config_value
from studies.pain_study.study1.cohort import (
    load_primary_target_table,
    primary_targets_parquet_path,
    study1_output_root,
)
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.targets import (
    PRIMARY_SIGNATURES,
    residualization_columns_for_target_table,
)

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
ARTICLE_MODEL_COLUMNS = (
    "target",
    "claim_tier",
    "feature_spec",
    "model",
    "mean_r2",
    "mean_nuisance_r2",
    "mean_delta_r2",
    "ci_low_delta_r2",
    "ci_high_delta_r2",
    "p_value_delta_r2",
    "p_value_delta_r2_holm",
    "n_perm_completed",
    "n_folds",
    "n_subjects_included",
    "primary_prediction_status",
    "interpretation_flags",
    "study2_source_entry_status",
)
ARTICLE_REQUIRED_TARGET_COLUMNS = (
    "subject_id",
    "block",
    "within_block_trial",
    "onset",
    "NPS",
    "SIIPS1",
    "hrf_weighted_framewise_displacement",
    "hrf_weighted_std_dvars",
    "hrf_weighted_fp1_fp2_high_frequency_power",
    "residual_ecg_coupling",
    "stimulus_temp",
    "selected_surface",
)
ARTICLE_REQUIRED_EVENT_COLUMNS = (
    "run_id",
    "trial_number",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
    "residual_ecg_coupling",
    "peripheral_low_gamma_power",
)
FULL_PICTURE_MODEL_COLUMNS = (
    "lane",
    "analysis_partition",
    "target",
    "claim_tier",
    "feature_spec",
    "model",
    "mean_r2",
    "mean_nuisance_r2",
    "mean_delta_r2",
    "ci_low_delta_r2",
    "ci_high_delta_r2",
    "p_value_r2",
    "p_value_r2_holm",
    "p_value_delta_r2",
    "p_value_delta_r2_holm",
    "n_perm_completed",
    "n_folds",
    "n_subjects_included",
    "primary_prediction_status",
    "interpretation_flags",
    "study2_source_entry_status",
    "summary_path",
)
SENSITIVITY_MODEL_COLUMNS = (
    "analysis_label",
    "analysis_root",
    "analysis_partition",
    "target",
    "feature_spec",
    "model",
    "mean_r2",
    "overall_r2",
    "mean_nuisance_r2",
    "mean_delta_r2",
    "p_value_r2",
    "p_value_delta_r2",
    "n_perm_completed",
    "n_folds",
    "summary_path",
)
SENSITIVITY_PIVOT_VALUE_COLUMNS = (
    "mean_r2",
    "mean_nuisance_r2",
    "mean_delta_r2",
    "p_value_r2",
    "p_value_delta_r2",
)


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
        target_reliability is not None and target_reliability < SOURCE_ENTRY_MIN_TARGET_RELIABILITY
    ) or (reliability_trials is not None and reliability_trials < MIN_TARGET_RELIABILITY_TRIALS):
        flags.append("target_reliability_limited")

    precision_passed = _optional_bool(record, "precision_flag_passed")
    if precision_passed is False:
        flags.append("precision_limited")

    level2_delta_r2 = _optional_float(record, "level2_mean_delta_r2")
    if level2_delta_r2 is not None and level2_delta_r2 < SOURCE_ENTRY_MIN_LEVEL2_DELTA_R2:
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
        or _optional_float(record, "level2_mean_delta_r2") < SOURCE_ENTRY_MIN_LEVEL2_DELTA_R2
        or _optional_float(record, "target_split_half_reliability")
        < SOURCE_ENTRY_MIN_TARGET_RELIABILITY
        or _optional_float(record, "target_reliability_n_trials") < MIN_TARGET_RELIABILITY_TRIALS
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

    feature_primary = (out["lane"].astype(str) == "feature_benchmark") & (
        out["analysis_partition"].astype(str) == "primary"
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
                    **{field: metrics.get(field) for field in INTERPRETATION_DIAGNOSTIC_FIELDS},
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
    primary_mask = (out["lane"].astype(str) == "feature_benchmark") & (
        out["analysis_partition"].astype(str) == "primary"
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


def _write_article_tables(
    *,
    frame: pd.DataFrame,
    task: str,
    config: Any,
    report_root: Path,
    report_path: Path,
) -> None:
    target_table = load_primary_target_table(config)
    _require_columns(
        target_table,
        ARTICLE_REQUIRED_TARGET_COLUMNS,
        table_name="Study 1 primary target table",
    )
    included_subjects = sorted(target_table["subject_id"].astype(str).unique().tolist())
    if not included_subjects:
        raise ValueError("Study 1 article tables require at least one included subject.")

    events = _load_article_clean_events(
        subjects=included_subjects,
        task=task,
        config=config,
    )
    enriched_targets = _merge_targets_with_clean_events(target_table, events)

    article_root = report_root / "article_tables"
    model_table = _article_model_results(frame)
    cohort_table = _article_cohort_summary(target_table, events, enriched_targets)
    diagnostics_table = _article_target_diagnostics(
        target_table=target_table,
        enriched_targets=enriched_targets,
        config=config,
    )

    table_paths = {
        "model_results": _write_article_table(
            model_table,
            article_root / "article_model_results",
        ),
        "cohort_summary": _write_article_table(
            cohort_table,
            article_root / "article_cohort_summary",
        ),
        "target_diagnostics": _write_article_table(
            diagnostics_table,
            article_root / "article_target_diagnostics",
        ),
    }
    manifest = {
        "task": task,
        "source_report": str(report_path),
        "target_table": str(primary_targets_parquet_path(config)),
        "included_subjects": included_subjects,
        "n_subjects": int(len(included_subjects)),
        "n_trials": int(len(target_table)),
        "tables": {
            name: {"tsv": str(paths["tsv"]), "parquet": str(paths["parquet"])}
            for name, paths in table_paths.items()
        },
    }
    manifest_path = article_root / "article_table_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _write_full_picture_tables(
    *,
    frame: pd.DataFrame,
    config: Any,
    report_root: Path,
    report_path: Path,
) -> None:
    target_table = load_primary_target_table(config)
    full_picture_root = report_root / "full_picture"
    table_paths = {
        "primary_feature_model_summary": _write_article_table(
            _primary_feature_model_summary(frame),
            full_picture_root / "primary_feature_model_summary",
        ),
        "model_leaderboard_by_mean_r2": _write_article_table(
            _model_leaderboard(frame, score_column="mean_r2"),
            full_picture_root / "model_leaderboard_by_mean_r2",
        ),
        "model_leaderboard_by_delta_r2": _write_article_table(
            _model_leaderboard(frame, score_column="mean_delta_r2"),
            full_picture_root / "model_leaderboard_by_delta_r2",
        ),
        "target_by_stimulus_temp": _write_article_table(
            _target_by_stimulus_temp(target_table),
            full_picture_root / "target_by_stimulus_temp",
        ),
        "target_by_subject_and_stimulus_temp": _write_article_table(
            _target_by_subject_and_stimulus_temp(target_table),
            full_picture_root / "target_by_subject_and_stimulus_temp",
        ),
    }

    sensitivity_roots = _configured_sensitivity_output_roots(config)
    if sensitivity_roots:
        sensitivity_summary = _configured_sensitivity_model_summary(sensitivity_roots)
        table_paths["configured_sensitivity_model_summary"] = _write_article_table(
            sensitivity_summary,
            full_picture_root / "configured_sensitivity_model_summary",
        )
        table_paths["primary_sensitivity_comparison"] = _write_article_table(
            _primary_sensitivity_comparison(sensitivity_summary),
            full_picture_root / "primary_sensitivity_comparison",
        )

    manifest = {
        "source_report": str(report_path),
        "target_table": str(primary_targets_parquet_path(config)),
        "tables": {
            name: {"tsv": str(paths["tsv"]), "parquet": str(paths["parquet"])}
            for name, paths in table_paths.items()
        },
    }
    manifest_path = full_picture_root / "full_picture_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _write_article_table(frame: pd.DataFrame, stem: Path) -> dict[str, Path]:
    tsv_path = stem.with_suffix(".tsv")
    parquet_path = stem.with_suffix(".parquet")
    write_tsv(frame, tsv_path)
    write_parquet(frame, parquet_path)
    return {"tsv": tsv_path, "parquet": parquet_path}


def _primary_feature_model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, FULL_PICTURE_MODEL_COLUMNS, table_name="Study 1 report")
    rows = frame.loc[
        (frame["lane"].astype(str) == "feature_benchmark")
        & (frame["analysis_partition"].astype(str) == "primary"),
        list(FULL_PICTURE_MODEL_COLUMNS),
    ].copy()
    if rows.empty:
        raise ValueError("Study 1 full-picture model table requires primary feature rows.")
    return rows.sort_values(["target", "feature_spec", "model"], kind="stable").reset_index(
        drop=True
    )


def _model_leaderboard(frame: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    _require_columns(frame, FULL_PICTURE_MODEL_COLUMNS, table_name="Study 1 report")
    if score_column not in frame.columns:
        raise ValueError(f"Study 1 report is missing leaderboard score column: {score_column}")

    rows = frame.loc[:, list(FULL_PICTURE_MODEL_COLUMNS)].copy()
    scores = pd.to_numeric(rows[score_column], errors="coerce")
    rows = rows.loc[scores.notna()].copy()
    if rows.empty:
        raise ValueError(f"Study 1 full-picture leaderboard has no finite {score_column} values.")

    rows["_score"] = pd.to_numeric(rows[score_column], errors="raise")
    rows = rows.sort_values(
        ["lane", "analysis_partition", "target", "_score"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    rows["rank_within_target"] = (
        rows.groupby(["lane", "analysis_partition", "target"], sort=False)["_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return rows.drop(columns=["_score"]).reset_index(drop=True)


def _target_by_stimulus_temp(target_table: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        target_table,
        ("subject_id", "stimulus_temp", *PRIMARY_SIGNATURES),
        table_name="Study 1 primary target table",
    )
    rows = _target_table_with_numeric_signatures(target_table)
    grouped = rows.groupby("stimulus_temp", dropna=False, sort=True)
    summary = grouped.agg(
        n_trials=("subject_id", "size"),
        n_subjects=("subject_id", "nunique"),
    )
    for target_name in PRIMARY_SIGNATURES:
        summary[f"mean_{target_name}"] = grouped[target_name].mean()
        summary[f"sd_{target_name}"] = grouped[target_name].std(ddof=1)
    return summary.reset_index()


def _target_by_subject_and_stimulus_temp(target_table: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        target_table,
        ("subject_id", "stimulus_temp", *PRIMARY_SIGNATURES),
        table_name="Study 1 primary target table",
    )
    rows = _target_table_with_numeric_signatures(target_table)
    grouped = rows.groupby(["subject_id", "stimulus_temp"], dropna=False, sort=True)
    summary = grouped.agg(n_trials=("subject_id", "size"))
    for target_name in PRIMARY_SIGNATURES:
        summary[f"mean_{target_name}"] = grouped[target_name].mean()
    return summary.reset_index()


def _target_table_with_numeric_signatures(target_table: pd.DataFrame) -> pd.DataFrame:
    rows = target_table[["subject_id", "stimulus_temp", *PRIMARY_SIGNATURES]].copy()
    rows["subject_id"] = rows["subject_id"].astype(str)
    rows["stimulus_temp"] = pd.to_numeric(rows["stimulus_temp"], errors="raise")
    for target_name in PRIMARY_SIGNATURES:
        rows[target_name] = pd.to_numeric(rows[target_name], errors="raise")
    return rows


def _configured_sensitivity_output_roots(config: Any) -> list[tuple[str, Path]]:
    raw = get_config_value(config, "study1.reporting.sensitivity_output_roots", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("study1.reporting.sensitivity_output_roots must be a list.")
    if not raw:
        return []

    current_root = study1_output_root(config)
    roots: list[tuple[str, Path]] = [(current_root.name, current_root)]
    labels = {current_root.name}
    root_names = {current_root.name}
    multimodal_root = current_root.parent

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("study1.reporting.sensitivity_output_roots entries must be mappings.")
        label = str(item.get("label", "")).strip()
        root_name = str(item.get("root_name", "")).strip()
        if not label:
            raise ValueError(
                "study1.reporting.sensitivity_output_roots entries require a non-empty label."
            )
        if not root_name:
            raise ValueError(
                "study1.reporting.sensitivity_output_roots entries require a non-empty root_name."
            )
        root_path = Path(root_name)
        if root_path.is_absolute() or root_path.name != root_name:
            raise ValueError(
                "study1.reporting.sensitivity_output_roots.root_name must be a directory name, "
                f"not a path: {root_name!r}."
            )
        if label in labels:
            raise ValueError(f"Duplicate Study 1 sensitivity output label configured: {label!r}.")
        if root_name in root_names:
            raise ValueError(
                f"Duplicate Study 1 sensitivity output root_name configured: {root_name!r}."
            )

        resolved = multimodal_root / root_name
        if not resolved.exists():
            raise FileNotFoundError(
                "Configured Study 1 sensitivity output root not found: "
                f"{resolved} (label={label!r})."
            )
        labels.add(label)
        root_names.add(root_name)
        roots.append((label, resolved))
    return roots


def _configured_sensitivity_model_summary(
    roots: list[tuple[str, Path]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, root in roots:
        rows.extend(_sensitivity_model_rows_for_root(label=label, root=root))
    if not rows:
        raise FileNotFoundError(
            "Configured Study 1 sensitivity roots contained no model summaries."
        )
    frame = pd.DataFrame(rows)
    _require_columns(frame, SENSITIVITY_MODEL_COLUMNS, table_name="Study 1 sensitivity summary")
    return (
        frame.loc[:, list(SENSITIVITY_MODEL_COLUMNS)]
        .sort_values(
            ["analysis_label", "analysis_partition", "target", "feature_spec", "model"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _sensitivity_model_rows_for_root(*, label: str, root: Path) -> list[dict[str, Any]]:
    feature_root = root / "feature_benchmark"
    if not feature_root.exists():
        raise FileNotFoundError(
            f"Study 1 sensitivity root has no feature_benchmark directory: {feature_root}"
        )

    rows: list[dict[str, Any]] = []
    pattern = "*/" + "*/" + "*/model_comparison/metrics/model_comparison_summary.json"
    for summary_path in sorted(feature_root.glob(pattern)):
        partition, target_name, feature_spec = summary_path.relative_to(feature_root).parts[:3]
        payload = _read_json(summary_path)
        for model_name, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            if "mean_r2" not in metrics and "mean_mae" not in metrics:
                continue
            rows.append(
                {
                    "analysis_label": label,
                    "analysis_root": str(root),
                    "analysis_partition": partition,
                    "target": target_name,
                    "feature_spec": feature_spec,
                    "model": str(model_name),
                    "mean_r2": metrics.get("mean_r2"),
                    "overall_r2": metrics.get("overall_r2"),
                    "mean_nuisance_r2": metrics.get("mean_nuisance_r2"),
                    "mean_delta_r2": metrics.get("mean_delta_r2"),
                    "p_value_r2": metrics.get("p_value_r2"),
                    "p_value_delta_r2": metrics.get("p_value_delta_r2"),
                    "n_perm_completed": metrics.get("n_perm_completed"),
                    "n_folds": metrics.get("n_folds"),
                    "summary_path": str(summary_path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"Study 1 sensitivity root contained no model summaries: {root}")
    return rows


def _primary_sensitivity_comparison(sensitivity_summary: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        sensitivity_summary,
        SENSITIVITY_MODEL_COLUMNS,
        table_name="Study 1 sensitivity summary",
    )
    primary = sensitivity_summary.loc[
        sensitivity_summary["analysis_partition"].astype(str) == "primary"
    ].copy()
    if primary.empty:
        raise ValueError("Study 1 sensitivity comparison requires primary model rows.")

    pivot = primary.pivot_table(
        index=["target", "feature_spec", "model"],
        columns="analysis_label",
        values=list(SENSITIVITY_PIVOT_VALUE_COLUMNS),
        aggfunc="first",
    )
    pivot.columns = [
        f"{metric}_{_sensitivity_column_label(label)}" for metric, label in pivot.columns
    ]
    return pivot.reset_index()


def _sensitivity_column_label(label: Any) -> str:
    clean = "".join(
        character if character.isalnum() else "_" for character in str(label).strip()
    ).strip("_")
    if not clean:
        raise ValueError("Study 1 sensitivity labels must contain letters or numbers.")
    return clean


def _article_model_results(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ARTICLE_MODEL_COLUMNS, table_name="Study 1 report")
    rows = frame.loc[
        (frame["lane"].astype(str) == "feature_benchmark")
        & (frame["analysis_partition"].astype(str) == "primary"),
        list(ARTICLE_MODEL_COLUMNS),
    ].copy()
    if rows.empty:
        raise ValueError("Study 1 article model table requires primary feature-benchmark rows.")
    return rows.sort_values(["target", "feature_spec", "model"], kind="stable").reset_index(
        drop=True
    )


def _load_article_clean_events(
    *,
    subjects: list[str],
    task: str,
    config: Any,
) -> pd.DataFrame:
    event_frames: list[pd.DataFrame] = []
    for subject in subjects:
        event_path = find_clean_events_path(subject, task, config=config)
        if event_path is None or not event_path.exists():
            raise FileNotFoundError(
                "Study 1 article tables require clean EEG events for every included "
                f"subject. Missing: {subject}, task-{task}."
            )
        events = pd.read_csv(event_path, sep="\t")
        _require_columns(
            events,
            ARTICLE_REQUIRED_EVENT_COLUMNS,
            table_name=f"clean events for {subject}",
        )
        events = events.copy()
        events["subject_id"] = subject
        events["_block_key"] = _required_integer_series(events, "run_id")
        events["_within_block_trial_key"] = _required_integer_series(events, "trial_number")
        event_frames.append(events)

    if not event_frames:
        raise ValueError("Study 1 article tables require at least one clean events table.")
    return pd.concat(event_frames, axis=0, ignore_index=True)


def _merge_targets_with_clean_events(
    target_table: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    targets = target_table.copy()
    targets["_block_key"] = _required_integer_series(targets, "block")
    targets["_within_block_trial_key"] = _required_integer_series(targets, "within_block_trial")
    event_columns = [
        "subject_id",
        "_block_key",
        "_within_block_trial_key",
        "pain_binary_coded",
        "vas_final_coded_rating",
        "peripheral_low_gamma_power",
        "stimulus_temp",
        "selected_surface",
        "residual_ecg_coupling",
    ]
    merged = targets.merge(
        events[event_columns].rename(
            columns={
                "stimulus_temp": "event_stimulus_temp",
                "selected_surface": "event_selected_surface",
                "residual_ecg_coupling": "event_residual_ecg_coupling",
            }
        ),
        how="left",
        on=["subject_id", "_block_key", "_within_block_trial_key"],
        validate="one_to_one",
    )
    if merged["vas_final_coded_rating"].isna().any():
        missing = merged.loc[
            merged["vas_final_coded_rating"].isna(),
            ["subject_id", "block", "within_block_trial"],
        ]
        raise ValueError(
            "Study 1 article tables found target rows without matching clean events:\n"
            f"{missing.to_string(index=False)}"
        )
    return merged.drop(columns=["_block_key", "_within_block_trial_key"])


def _article_cohort_summary(
    target_table: pd.DataFrame,
    events: pd.DataFrame,
    enriched_targets: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted(target_table["subject_id"].astype(str).unique().tolist())
    for subject in subjects:
        target_rows = target_table.loc[target_table["subject_id"].astype(str) == subject].copy()
        event_rows = events.loc[events["subject_id"].astype(str) == subject].copy()
        enriched_rows = enriched_targets.loc[
            enriched_targets["subject_id"].astype(str) == subject
        ].copy()
        target_keys = set(
            zip(
                _required_integer_series(target_rows, "block"),
                _required_integer_series(target_rows, "within_block_trial"),
            )
        )
        event_keys = set(
            zip(
                event_rows["_block_key"].astype(int),
                event_rows["_within_block_trial_key"].astype(int),
            )
        )

        row = {
            "subject_id": subject,
            "n_target_trials": int(len(target_rows)),
            "n_clean_event_trials": int(len(event_rows)),
            "n_target_event_matches": int(len(target_keys & event_keys)),
            "n_event_without_target": int(len(event_keys - target_keys)),
            "n_blocks": int(_numeric_series(target_rows, "block").nunique()),
            "n_stimulus_temperatures": int(_numeric_series(target_rows, "stimulus_temp").nunique()),
            "min_stimulus_temp": float(_numeric_series(target_rows, "stimulus_temp").min()),
            "max_stimulus_temp": float(_numeric_series(target_rows, "stimulus_temp").max()),
            "mean_framewise_displacement": _mean(
                target_rows, "hrf_weighted_framewise_displacement"
            ),
            "mean_std_dvars": _mean(target_rows, "hrf_weighted_std_dvars"),
            "mean_residual_ecg_coupling": _mean(target_rows, "residual_ecg_coupling"),
            "mean_peripheral_low_gamma_power": _mean(
                enriched_rows,
                "peripheral_low_gamma_power",
            ),
            "mean_vas_rating": _mean(enriched_rows, "vas_final_coded_rating"),
            "vas_temp_r": _correlation(
                enriched_rows,
                "stimulus_temp",
                "vas_final_coded_rating",
            ),
            "vas_high_minus_low_temp": _high_minus_low(
                enriched_rows,
                value_column="vas_final_coded_rating",
            ),
        }
        for target_name in PRIMARY_SIGNATURES:
            row[f"{target_name}_mean"] = _mean(target_rows, target_name)
            row[f"{target_name}_sd"] = _std(target_rows, target_name)
            row[f"{target_name}_temp_r"] = _correlation(
                target_rows,
                "stimulus_temp",
                target_name,
            )
            row[f"{target_name}_high_minus_low_temp"] = _high_minus_low(
                target_rows,
                value_column=target_name,
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject_id", kind="stable").reset_index(drop=True)


def _article_target_diagnostics(
    *,
    target_table: pd.DataFrame,
    enriched_targets: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    nuisance_columns = residualization_columns_for_target_table(
        config,
        primary_targets_parquet_path(config),
    )
    stimulus_surface_columns = _stimulus_surface_design_columns(target_table)

    rows = []
    for target_name in PRIMARY_SIGNATURES:
        rows.append(
            {
                "target": target_name,
                "n_trials": int(len(target_table)),
                "n_subjects": int(target_table["subject_id"].astype(str).nunique()),
                "mean": _mean(target_table, target_name),
                "sd": _std(target_table, target_name),
                "stimulus_temp_r": _correlation(target_table, "stimulus_temp", target_name),
                "vas_rating_r": _correlation(
                    enriched_targets,
                    "vas_final_coded_rating",
                    target_name,
                ),
                "pain_binary_r": _correlation(
                    enriched_targets,
                    "pain_binary_coded",
                    target_name,
                ),
                "framewise_displacement_r": _correlation(
                    target_table,
                    "hrf_weighted_framewise_displacement",
                    target_name,
                ),
                "std_dvars_r": _correlation(
                    target_table,
                    "hrf_weighted_std_dvars",
                    target_name,
                ),
                "residual_ecg_coupling_r": _correlation(
                    target_table,
                    "residual_ecg_coupling",
                    target_name,
                ),
                "peripheral_low_gamma_r": _correlation(
                    enriched_targets,
                    "peripheral_low_gamma_power",
                    target_name,
                ),
                "stimulus_surface_in_sample_r2": _in_sample_design_r2(
                    target_table,
                    target_name,
                    stimulus_surface_columns,
                ),
                "official_nuisance_in_sample_r2": _in_sample_r2(
                    target_table,
                    target_name,
                    nuisance_columns,
                ),
                **_split_half_temperature_reliability(target_table, target_name),
            }
        )
    return pd.DataFrame(rows)


def _stimulus_surface_design_columns(target_table: pd.DataFrame) -> pd.DataFrame:
    _require_columns(target_table, ("stimulus_temp", "selected_surface"), table_name="target table")
    design = pd.get_dummies(
        target_table[["stimulus_temp", "selected_surface"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    if design.empty:
        raise ValueError("Study 1 article diagnostics require stimulus/surface variation.")
    return design


def _split_half_temperature_reliability(
    target_table: pd.DataFrame,
    target_name: str,
) -> dict[str, Any]:
    _require_columns(
        target_table,
        ("subject_id", "block", "stimulus_temp", target_name),
        table_name="target table",
    )
    frame = target_table[["subject_id", "block", "stimulus_temp", target_name]].copy()
    frame["block_parity"] = np.where(
        _required_integer_series(frame, "block") % 2 == 0, "even", "odd"
    )
    pivot = frame.pivot_table(
        index=["subject_id", "stimulus_temp"],
        columns="block_parity",
        values=target_name,
        aggfunc="mean",
    )
    if "odd" not in pivot.columns or "even" not in pivot.columns:
        return {
            "split_half_subject_temperature_r": float("nan"),
            "split_half_subject_temperature_n_cells": 0,
        }
    cells = pivot.dropna(subset=["odd", "even"])
    return {
        "split_half_subject_temperature_r": _series_correlation(cells["odd"], cells["even"]),
        "split_half_subject_temperature_n_cells": int(len(cells)),
    }


def _high_minus_low(frame: pd.DataFrame, *, value_column: str) -> float:
    _require_columns(frame, ("stimulus_temp", value_column), table_name="article input")
    values = frame[["stimulus_temp", value_column]].copy()
    values["stimulus_temp"] = _numeric_series(values, "stimulus_temp")
    values[value_column] = _numeric_series(values, value_column)
    by_temp = values.groupby("stimulus_temp", sort=True)[value_column].mean()
    if len(by_temp) < 2:
        return float("nan")
    return float(by_temp.iloc[-1] - by_temp.iloc[0])


def _in_sample_r2(frame: pd.DataFrame, target_column: str, predictors: tuple[str, ...]) -> float:
    _require_columns(frame, (target_column, *predictors), table_name="article diagnostics")
    y = _numeric_series(frame, target_column).to_numpy(dtype=float)
    if not predictors:
        return 0.0
    x_columns = [_numeric_series(frame, column).to_numpy(dtype=float) for column in predictors]
    design = np.column_stack([np.ones(len(y), dtype=float), *x_columns])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _in_sample_design_r2(
    frame: pd.DataFrame,
    target_column: str,
    design_frame: pd.DataFrame,
) -> float:
    _require_columns(frame, (target_column,), table_name="article diagnostics")
    if len(frame) != len(design_frame):
        raise ValueError(
            "Article diagnostic design matrix row count does not match target table: "
            f"design={len(design_frame)}, target={len(frame)}."
        )
    y = _numeric_series(frame, target_column).to_numpy(dtype=float)
    design_values = design_frame.apply(pd.to_numeric, errors="coerce")
    if design_values.isna().any().any():
        raise ValueError("Article diagnostic design matrix contains non-numeric values.")
    design = np.column_stack(
        [
            np.ones(len(y), dtype=float),
            design_values.to_numpy(dtype=float),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _correlation(frame: pd.DataFrame, x_column: str, y_column: str) -> float:
    _require_columns(frame, (x_column, y_column), table_name="article input")
    return _series_correlation(_numeric_series(frame, x_column), _numeric_series(frame, y_column))


def _series_correlation(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].corr(pair["y"]))


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(_numeric_series(frame, column).mean())


def _std(frame: pd.DataFrame, column: str) -> float:
    return float(_numeric_series(frame, column).std(ddof=1))


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    _require_columns(frame, (column,), table_name="article input")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Article table column '{column}' contains non-numeric values.")
    return values.astype(float)


def _required_integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric_series(frame, column)
    if not np.allclose(values, np.round(values)):
        raise ValueError(f"Article table column '{column}' must contain integer-valued labels.")
    return values.round().astype(int)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, table_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required article column(s): {missing}.")


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
    _write_article_tables(
        frame=frame,
        task=task,
        config=config,
        report_root=report_root,
        report_path=tsv_path,
    )
    _write_full_picture_tables(
        frame=frame,
        config=config,
        report_root=report_root,
        report_path=tsv_path,
    )
    logger.info("Wrote Study 1 report to %s", tsv_path)
    return tsv_path


__all__ = ["write_study1_report"]
