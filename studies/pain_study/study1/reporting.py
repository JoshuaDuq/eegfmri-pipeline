"""Study 1 report aggregation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import get_config_value
from studies.pain_study.study1.cohort import (
    primary_targets_parquet_path,
    study1_output_root,
)
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.figures import (
    write_behavioral_dose_response,
    write_nps_behavioral_validity,
    write_nps_dose_response,
    write_siips1_behavioral_validity,
    write_siips1_dose_response,
)
from studies.pain_study.study1.figures.behavioral_validity import (
    NPS_SPECIFICATION,
    SIIPS1_SPECIFICATION,
    BehavioralValiditySummary,
    build_behavioral_validity_summary,
)
from studies.pain_study.study1.figures.validity_data import (
    WITHIN_SCALE_INTENSITY_COLUMN,
    ValidityTrialData,
    load_validity_trial_data,
)
from studies.pain_study.study1.targets import (
    PRIMARY_SIGNATURES,
    resolve_target_residualization_columns,
)
from studies.pain_study.study1.temporal_controls import (
    TEMPORAL_CONTROL_PARTITION,
    temporal_control_window_for_feature_spec,
)

FEATURE_MODELS = ("elasticnet", "ridge")
PRIMARY_GATE_TARGET = "NPS"
PRIMARY_GATE_FEATURE_SPEC = "alpha_beta_gamma"
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
    "within_condition_centered_delta_r2",
    "within_condition_centered_n_subjects",
    "within_condition_centered_n_trials",
    "temporal_negative_controls_passed",
    "artifact_censoring_robustness_passed",
    "hrf_timing_robustness_passed",
    "first_exposure_robustness_passed",
    "baseline_robustness_passed",
    "smoothing_robustness_passed",
)
SPLIT_HALF_RELIABILITY_N_SPLITS = 1000
SPLIT_HALF_RELIABILITY_SEED = 42
PRIMARY_P_VALUE_ALPHA = 0.05
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
)
ARTICLE_REQUIRED_TARGET_COLUMNS = (
    "subject_id",
    "run",
    "within_run_trial",
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


def _append_derived_qc_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for field in INTERPRETATION_DIAGNOSTIC_FIELDS:
        if field not in out.columns:
            out[field] = pd.NA

    # Window-wise significance cannot establish equivalence or temporal superiority.
    # Stored booleans carry no evidence of either test and must not propagate as a pass.
    out["temporal_negative_controls_passed"] = pd.NA
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
        temporal_window = (
            temporal_control_window_for_feature_spec(config, feature_spec)
            if partition == TEMPORAL_CONTROL_PARTITION
            else None
        )
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
                    "temporal_control_window": (
                        temporal_window.name if temporal_window is not None else pd.NA
                    ),
                    "temporal_control_kind": (
                        temporal_window.kind if temporal_window is not None else pd.NA
                    ),
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


def _append_feature_multiplicity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_value_r2_holm"] = pd.NA
    out["p_value_delta_r2_holm"] = pd.NA
    feature_mask = out["lane"].astype(str) == "feature_benchmark"
    primary_mask = feature_mask & (out["analysis_partition"].astype(str) == "primary")
    temporal_control_mask = feature_mask & (
        out["analysis_partition"].astype(str) == TEMPORAL_CONTROL_PARTITION
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
            _apply_holm_to_mask(
                out,
                mask=mask,
                raw_column=raw_column,
                adjusted_column=adjusted_column,
                multipletests_fn=multipletests,
            )
        _apply_holm_to_mask(
            out,
            mask=temporal_control_mask,
            raw_column=raw_column,
            adjusted_column=adjusted_column,
            multipletests_fn=multipletests,
        )
    return out


def _apply_holm_to_mask(
    frame: pd.DataFrame,
    *,
    mask: pd.Series,
    raw_column: str,
    adjusted_column: str,
    multipletests_fn: Any,
) -> None:
    p_values = pd.to_numeric(frame.loc[mask, raw_column], errors="coerce")
    valid = p_values.notna()
    if not valid.any():
        return
    values = p_values.loc[valid].to_numpy(dtype=float)
    adjusted = multipletests_fn(values, method="holm")[1]
    frame.loc[p_values.loc[valid].index, adjusted_column] = adjusted


def _write_article_tables(
    *,
    frame: pd.DataFrame,
    task: str,
    trial_data: ValidityTrialData,
    config: Any,
    report_root: Path,
    report_path: Path,
) -> None:
    target_table = trial_data.targets
    _require_columns(
        target_table,
        ARTICLE_REQUIRED_TARGET_COLUMNS,
        table_name="Study 1 primary target table",
    )
    included_subjects = sorted(target_table["subject_id"].astype(str).unique().tolist())
    if not included_subjects:
        raise ValueError("Study 1 article tables require at least one included subject.")

    events = trial_data.clean_events
    enriched_targets = trial_data.enriched_targets

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
    trial_data: ValidityTrialData,
    behavioral_validity_summaries: dict[str, BehavioralValiditySummary],
    supplementary_figures: dict[str, Path],
    config: Any,
    report_root: Path,
    report_path: Path,
) -> None:
    target_table = trial_data.targets
    enriched_targets = trial_data.enriched_targets
    target_diagnostics = _article_target_diagnostics(
        target_table=target_table,
        enriched_targets=enriched_targets,
        config=config,
    )
    full_picture_root = report_root / "full_picture"
    participant_validity, cohort_validity = _behavioral_validity_tables(
        behavioral_validity_summaries
    )
    table_paths = {
        "behavior_signature_validity_by_subject": _write_article_table(
            participant_validity,
            full_picture_root / "behavior_signature_validity_by_subject",
        ),
        "behavior_signature_validity_summary": _write_article_table(
            cohort_validity,
            full_picture_root / "behavior_signature_validity_summary",
        ),
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
        "target_qc_metrics": _write_article_table(
            _target_qc_metrics(
                diagnostics=target_diagnostics,
                behavioral_cohort_estimates=cohort_validity,
            ),
            full_picture_root / "target_qc_metrics",
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
        "supplementary_figures": {name: str(path) for name, path in supplementary_figures.items()},
        "tables": {
            name: {"tsv": str(paths["tsv"]), "parquet": str(paths["parquet"])}
            for name, paths in table_paths.items()
        },
    }
    manifest_path = full_picture_root / "full_picture_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _behavioral_validity_summaries(
    trial_data: ValidityTrialData,
    config: Any,
) -> dict[str, BehavioralValiditySummary]:
    return {
        specification.target: build_behavioral_validity_summary(
            trial_data.enriched_targets,
            specification=specification,
            config=config,
        )
        for specification in (NPS_SPECIFICATION, SIIPS1_SPECIFICATION)
    }


def _behavioral_validity_tables(
    summaries: dict[str, BehavioralValiditySummary],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = [summaries[target] for target in PRIMARY_SIGNATURES]
    participants = pd.concat(
        [summary.participant_models for summary in ordered],
        ignore_index=True,
    )
    cohort = pd.concat(
        [summary.cohort_estimates for summary in ordered],
        ignore_index=True,
    )
    return participants, cohort


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
                _required_integer_series(target_rows, "run"),
                _required_integer_series(target_rows, "within_run_trial"),
            )
        )
        event_keys = set(
            zip(
                event_rows["_run_key"].astype(int),
                event_rows["_within_run_trial_key"].astype(int),
            )
        )

        row = {
            "subject_id": subject,
            "n_target_trials": int(len(target_rows)),
            "n_clean_event_trials": int(len(event_rows)),
            "n_target_event_matches": int(len(target_keys & event_keys)),
            "n_event_without_target": int(len(event_keys - target_keys)),
            "n_runs": int(_numeric_series(target_rows, "run").nunique()),
            "n_stimulus_temperatures": int(_numeric_series(target_rows, "stimulus_temp").nunique()),
            "min_stimulus_temp": float(_numeric_series(target_rows, "stimulus_temp").min()),
            "max_stimulus_temp": float(_numeric_series(target_rows, "stimulus_temp").max()),
            "mean_framewise_displacement": _mean(
                target_rows, "hrf_weighted_framewise_displacement"
            ),
            "mean_std_dvars": _mean(target_rows, "hrf_weighted_std_dvars"),
            "mean_residual_ecg_coupling": _mean(target_rows, "residual_ecg_coupling"),
            "mean_fp1_fp2_high_frequency_power": _mean(
                enriched_rows,
                "fp1_fp2_high_frequency_power",
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
    stimulus_surface_columns = _stimulus_surface_design_columns(target_table)

    rows = []
    for target_name in PRIMARY_SIGNATURES:
        nuisance_columns = resolve_target_residualization_columns(
            frame=target_table, config=config, target_name=target_name
        )
        nuisance_in_sample_r2 = _in_sample_r2(target_table, target_name, nuisance_columns)
        rows.append(
            {
                "target": target_name,
                "n_trials": int(len(target_table)),
                "n_subjects": int(target_table["subject_id"].astype(str).nunique()),
                "mean": _mean(target_table, target_name),
                "sd": _std(target_table, target_name),
                "stimulus_temp_r": _correlation(target_table, "stimulus_temp", target_name),
                "within_scale_intensity_r": _correlation(
                    enriched_targets,
                    "within_scale_intensity",
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
                "fp1_fp2_high_frequency_r": _correlation(
                    enriched_targets,
                    "fp1_fp2_high_frequency_power",
                    target_name,
                ),
                "stimulus_surface_in_sample_r2": _in_sample_design_r2(
                    target_table,
                    target_name,
                    stimulus_surface_columns,
                ),
                "official_nuisance_in_sample_r2": nuisance_in_sample_r2,
                "residual_target_variance_fraction": _residual_variance_fraction(
                    nuisance_in_sample_r2
                ),
                **_split_half_temperature_reliability(target_table, target_name),
            }
        )
    return pd.DataFrame(rows)


def _target_qc_metrics(
    *,
    diagnostics: pd.DataFrame,
    behavioral_cohort_estimates: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = (
        "target",
        "n_trials",
        "n_subjects",
        "stimulus_temp_r",
        "within_scale_intensity_r",
        "pain_binary_r",
        "split_half_subject_temperature_r",
        "split_half_subject_temperature_n_cells",
    )
    _require_columns(
        diagnostics,
        required_columns,
        table_name="Study 1 target diagnostics",
    )
    intensity_betas = _within_participant_intensity_betas(behavioral_cohort_estimates)
    rows: list[dict[str, Any]] = []
    for _, diagnostic in diagnostics.sort_values("target", kind="stable").iterrows():
        target_name = str(diagnostic["target"])
        rows.append(
            {
                "target": target_name,
                "n_trials": int(diagnostic["n_trials"]),
                "n_subjects": int(diagnostic["n_subjects"]),
                "stimulus_temp_r": diagnostic["stimulus_temp_r"],
                "within_scale_intensity_r": diagnostic["within_scale_intensity_r"],
                "pain_binary_r": diagnostic["pain_binary_r"],
                "within_participant_intensity_standardized_beta": intensity_betas[target_name],
                "split_half_subject_temperature_r": diagnostic["split_half_subject_temperature_r"],
                "split_half_subject_temperature_n_cells": int(
                    diagnostic["split_half_subject_temperature_n_cells"]
                ),
                "residual_target_variance_fraction": _optional_float(
                    diagnostic,
                    "residual_target_variance_fraction",
                ),
            }
        )
    return pd.DataFrame(rows)


def _within_participant_intensity_betas(cohort_estimates: pd.DataFrame) -> dict[str, float]:
    """Equally weighted participant slope for intensity, per target.

    Taken from the behavioral-validity models rather than recomputed here: those adjust
    temperature categorically within each participant, and SIIPS1 additionally for NPS.
    A pooled linear temperature adjustment over all trials answers a different question.
    """
    _require_columns(
        cohort_estimates,
        ("target", "term", "mean"),
        table_name="Study 1 behavioral validity cohort estimates",
    )
    intensity = cohort_estimates.loc[
        cohort_estimates["term"].astype(str) == WITHIN_SCALE_INTENSITY_COLUMN
    ]
    betas: dict[str, float] = {}
    for target_name in PRIMARY_SIGNATURES:
        matching = intensity.loc[intensity["target"].astype(str) == target_name, "mean"]
        if len(matching) != 1:
            raise ValueError(
                f"Study 1 behavioral validity estimates must define one "
                f"{WITHIN_SCALE_INTENSITY_COLUMN} row for {target_name}; got {len(matching)}."
            )
        betas[target_name] = float(matching.iloc[0])
    return betas


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


def _residual_variance_fraction(in_sample_r2: float) -> float:
    """Fraction of target variance remaining after Level-2 nuisance residualization.

    This descriptive, in-sample fraction is neither residual-trial reliability nor
    an upper bound on out-of-sample incremental prediction.
    """
    if not np.isfinite(in_sample_r2):
        return float("nan")
    return float(np.clip(1.0 - in_sample_r2, 0.0, 1.0))


def _spearman_brown_corrected(r_value: float) -> float:
    if not np.isfinite(r_value) or r_value <= -1.0:
        return float("nan")
    return (2.0 * r_value) / (1.0 + r_value)


def _split_half_temperature_reliability(
    target_table: pd.DataFrame,
    target_name: str,
    *,
    n_splits: int = SPLIT_HALF_RELIABILITY_N_SPLITS,
    seed: int = SPLIT_HALF_RELIABILITY_SEED,
) -> dict[str, Any]:
    """Split-half reliability stratified within subject and stimulus temperature.

    Implements README section 6. For each of ``n_splits`` random partitions, every
    subject-by-temperature cell with at least two trials contributes the mean of half A
    and half B; the per-split statistic is the Spearman-Brown-corrected Pearson
    correlation across cells, and the reported value is the median across valid splits.
    """
    _require_columns(
        target_table,
        ("subject_id", "stimulus_temp", target_name),
        table_name="target table",
    )
    frame = target_table[["subject_id", "stimulus_temp", target_name]].copy()
    frame[target_name] = _numeric_series(frame, target_name)

    splittable = [
        cell[target_name].to_numpy(dtype=float)
        for _, cell in frame.groupby(["subject_id", "stimulus_temp"], sort=True)
        if len(cell) >= 2
    ]
    n_cells = len(splittable)
    if n_cells < 2:
        return {
            "split_half_subject_temperature_r": float("nan"),
            "split_half_subject_temperature_n_cells": n_cells,
        }

    rng = np.random.default_rng(seed)
    corrected: list[float] = []
    for _ in range(n_splits):
        half_a = np.empty(n_cells, dtype=float)
        half_b = np.empty(n_cells, dtype=float)
        for cell_idx, values in enumerate(splittable):
            order = rng.permutation(values.size)
            half = values.size // 2
            half_a[cell_idx] = values[order[:half]].mean()
            half_b[cell_idx] = values[order[half : 2 * half]].mean()
        sb = _spearman_brown_corrected(_series_correlation(pd.Series(half_a), pd.Series(half_b)))
        if np.isfinite(sb):
            corrected.append(sb)

    reliability = float(np.median(corrected)) if corrected else float("nan")
    return {
        "split_half_subject_temperature_r": reliability,
        "split_half_subject_temperature_n_cells": n_cells,
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
    frame = _append_feature_multiplicity(frame)
    frame = _append_derived_qc_metrics(frame)
    trial_data = load_validity_trial_data(task=task, config=config)
    behavioral_validity_summaries = _behavioral_validity_summaries(trial_data, config)
    summary_payload = {
        "task": task,
        "n_records": int(len(frame)),
        "lanes": sorted(set(frame["lane"].astype(str).tolist())),
        "claim_tiers": sorted(set(frame["claim_tier"].astype(str).tolist())),
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
        trial_data=trial_data,
        config=config,
        report_root=report_root,
        report_path=tsv_path,
    )
    supplementary_figures = {
        "behavioral_dose_response": write_behavioral_dose_response(
            trial_data=trial_data,
            config=config,
        ),
        "nps_behavioral_validity": write_nps_behavioral_validity(
            summary=behavioral_validity_summaries["NPS"],
            config=config,
        ),
        "nps_dose_response": write_nps_dose_response(
            trial_data=trial_data,
            config=config,
        ),
        "siips1_behavioral_validity": write_siips1_behavioral_validity(
            summary=behavioral_validity_summaries["SIIPS1"],
            config=config,
        ),
        "siips1_dose_response": write_siips1_dose_response(
            trial_data=trial_data,
            config=config,
        ),
    }
    _write_full_picture_tables(
        frame=frame,
        trial_data=trial_data,
        behavioral_validity_summaries=behavioral_validity_summaries,
        supplementary_figures=supplementary_figures,
        config=config,
        report_root=report_root,
        report_path=tsv_path,
    )
    logger.info("Wrote Study 1 report to %s", tsv_path)
    return tsv_path


__all__ = ["write_study1_report"]
