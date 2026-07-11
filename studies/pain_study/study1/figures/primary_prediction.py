"""Validated data assembly for the Study 1 primary-prediction figure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value

REPORT_COLUMNS = (
    "lane",
    "analysis_partition",
    "target",
    "feature_spec",
    "model",
    "mean_nuisance_r2",
    "mean_r2",
    "ci_low_r2",
    "ci_high_r2",
    "mean_delta_r2",
    "ci_low_delta_r2",
    "ci_high_delta_r2",
    "p_value_delta_r2",
    "p_value_delta_r2_holm",
    "n_folds",
    "n_subjects_included",
    "summary_path",
)
FOLD_COLUMNS = (
    "model",
    "fold",
    "test_subject",
    "r2_nuisance",
    "r2",
    "delta_r2",
)


@dataclass(frozen=True)
class PrimaryPredictionSummary:
    """Plot-ready participant and cohort primary-prediction estimates."""

    targets: tuple[str, ...]
    participant_performance: pd.DataFrame
    cohort_performance: pd.DataFrame


@dataclass(frozen=True)
class PrimaryPredictionSelection:
    """Fixed report-cell identifiers configured for the figure."""

    lane: str
    analysis_partition: str
    model: str
    feature_spec: str
    targets: tuple[str, ...]


def load_primary_prediction_summary(
    report_path: str | Path,
    config: Any,
) -> PrimaryPredictionSummary:
    """Load and cross-check the configured primary model and LOSO folds."""

    resolved_report_path = Path(report_path).expanduser()
    if not resolved_report_path.is_file():
        raise FileNotFoundError(f"Study 1 report does not exist: {resolved_report_path}")

    report = pd.read_csv(resolved_report_path, sep="\t")
    _require_columns(report, REPORT_COLUMNS, source="Study 1 report")
    selection = _figure_selection(config)
    selected = report.loc[
        report["lane"].eq(selection.lane)
        & report["analysis_partition"].eq(selection.analysis_partition)
        & report["model"].eq(selection.model)
        & report["feature_spec"].eq(selection.feature_spec)
        & report["target"].isin(selection.targets)
    ].copy()
    _validate_selected_rows(selected, selection)

    participant_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for target in selection.targets:
        report_row = selected.loc[selected["target"].eq(target)].iloc[0]
        participants, cohort = _load_target(
            report_row,
            target=target,
            selection=selection,
        )
        participant_rows.extend(participants)
        cohort_rows.append(cohort)

    return PrimaryPredictionSummary(
        targets=selection.targets,
        participant_performance=pd.DataFrame(participant_rows),
        cohort_performance=pd.DataFrame(cohort_rows),
    )


def _figure_selection(config: Any) -> PrimaryPredictionSelection:
    raw = require_config_value(config, "study1.figures.primary_prediction")
    if not isinstance(raw, Mapping):
        raise ValueError("study1.figures.primary_prediction must be a mapping.")

    values = {
        key: str(raw.get(key, "")).strip()
        for key in ("lane", "analysis_partition", "model", "feature_spec")
    }
    empty = sorted(key for key, value in values.items() if not value)
    if empty:
        raise ValueError(f"Primary-prediction figure fields must be non-empty: {empty}.")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("study1.figures.primary_prediction.targets must be a non-empty list.")
    targets = tuple(str(target).strip() for target in raw_targets)
    if any(not target for target in targets) or len(set(targets)) != len(targets):
        raise ValueError("Primary-prediction targets must be unique and non-empty.")
    return PrimaryPredictionSelection(targets=targets, **values)


def _validate_selected_rows(
    selected: pd.DataFrame,
    selection: PrimaryPredictionSelection,
) -> None:
    observed_targets = set(selected["target"].astype(str))
    expected_targets = set(selection.targets)
    if observed_targets != expected_targets:
        raise ValueError(
            "Configured primary target set does not match figure configuration: "
            f"expected {sorted(expected_targets)}, observed {sorted(observed_targets)}."
        )
    duplicates = selected["target"].duplicated(keep=False)
    if duplicates.any():
        targets = sorted(selected.loc[duplicates, "target"].astype(str).unique())
        raise ValueError(f"Duplicate primary-prediction report rows for targets: {targets}.")


def _load_target(
    report_row: pd.Series,
    *,
    target: str,
    selection: PrimaryPredictionSelection,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    report_metrics = {
        field: _finite_float(report_row[field], field=field)
        for field in (
            "mean_nuisance_r2",
            "mean_r2",
            "ci_low_r2",
            "ci_high_r2",
            "mean_delta_r2",
            "ci_low_delta_r2",
            "ci_high_delta_r2",
            "p_value_delta_r2",
            "p_value_delta_r2_holm",
        )
    }
    _validate_interval(
        report_metrics["ci_low_r2"],
        report_metrics["mean_r2"],
        report_metrics["ci_high_r2"],
        target=target,
        estimand="R²",
    )
    _validate_interval(
        report_metrics["ci_low_delta_r2"],
        report_metrics["mean_delta_r2"],
        report_metrics["ci_high_delta_r2"],
        target=target,
        estimand="ΔR²",
    )
    for field in ("p_value_delta_r2", "p_value_delta_r2_holm"):
        if not 0.0 <= report_metrics[field] <= 1.0:
            raise ValueError(f"{field} for {target} must lie in [0, 1].")

    n_folds = _positive_integer(report_row["n_folds"], field="n_folds")
    n_subjects = _positive_integer(
        report_row["n_subjects_included"],
        field="n_subjects_included",
    )
    summary_path = Path(str(report_row["summary_path"])).expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Model summary does not exist: {summary_path}")
    fold_path = summary_path.parent / "model_comparison.tsv"
    if not fold_path.is_file():
        raise FileNotFoundError(f"Primary-prediction fold table does not exist: {fold_path}")

    folds = pd.read_csv(fold_path, sep="\t")
    _require_columns(folds, FOLD_COLUMNS, source=str(fold_path))
    model_folds = folds.loc[folds["model"].eq(selection.model)].copy()
    if model_folds.empty:
        raise ValueError(f"Fold table contains no {selection.model!r} rows: {fold_path}")
    subjects = model_folds["test_subject"]
    if subjects.isna().any() or subjects.astype(str).str.strip().eq("").any():
        raise ValueError(f"Fold table contains empty held-out subject identifiers: {fold_path}")
    if subjects.duplicated().any():
        raise ValueError(f"Fold table contains duplicate held-out subjects: {fold_path}")

    numeric = model_folds.loc[:, ["r2_nuisance", "r2", "delta_r2"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Fold table contains non-finite prediction metrics: {fold_path}")
    arithmetic_delta = numeric["r2"] - numeric["r2_nuisance"]
    if not np.allclose(
        numeric["delta_r2"],
        arithmetic_delta,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(f"Fold delta_r2 does not equal r2 - r2_nuisance for {target}: {fold_path}")
    if len(model_folds) != n_folds:
        raise ValueError(f"Fold count for {target} is {len(model_folds)}, expected {n_folds}.")
    if subjects.nunique() != n_subjects:
        raise ValueError(
            f"Subject count for {target} is {subjects.nunique()}, expected {n_subjects}."
        )

    mean_pairs = (
        ("r2_nuisance", "mean_nuisance_r2"),
        ("r2", "mean_r2"),
        ("delta_r2", "mean_delta_r2"),
    )
    for fold_field, report_field in mean_pairs:
        fold_mean = float(numeric[fold_field].mean())
        report_mean = report_metrics[report_field]
        if not np.isclose(fold_mean, report_mean, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"Primary-prediction fold mean does not match report mean for "
                f"{target}/{fold_field}: {fold_mean:g} versus {report_mean:g}."
            )

    participant_rows = [
        {
            "target": target,
            "model": selection.model,
            "feature_spec": selection.feature_spec,
            "fold": fold,
            "subject_id": subject,
            "r2_nuisance": nuisance,
            "r2": full,
            "delta_r2": delta,
            "fold_table_path": str(fold_path),
        }
        for fold, subject, nuisance, full, delta in zip(
            model_folds["fold"],
            subjects.astype(str),
            numeric["r2_nuisance"],
            numeric["r2"],
            numeric["delta_r2"],
            strict=True,
        )
    ]
    cohort_row = {
        "target": target,
        "model": selection.model,
        "feature_spec": selection.feature_spec,
        **report_metrics,
        "n_folds": n_folds,
        "n_subjects": n_subjects,
        "summary_path": str(summary_path),
        "fold_table_path": str(fold_path),
    }
    return participant_rows, cohort_row


def _validate_interval(
    low: float,
    mean: float,
    high: float,
    *,
    target: str,
    estimand: str,
) -> None:
    if low > mean or mean > high:
        raise ValueError(f"Invalid {estimand} confidence interval for {target}.")


def _require_columns(
    table: pd.DataFrame,
    required: tuple[str, ...],
    *,
    source: str,
) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}.")


def _finite_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric; observed {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite; observed {value!r}.")
    return number


def _positive_integer(value: object, *, field: str) -> int:
    number = _finite_float(value, field=field)
    if number <= 0 or not number.is_integer():
        raise ValueError(f"{field} must be a positive integer; observed {value!r}.")
    return int(number)


__all__ = [
    "PrimaryPredictionSummary",
    "load_primary_prediction_summary",
]
