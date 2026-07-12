"""Validated data assembly for the Study 1 temporal-specificity figure."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.temporal_controls import (
    TemporalControlWindow,
    resolve_temporal_control_windows,
)

REPORT_COLUMNS = (
    "lane",
    "analysis_partition",
    "target",
    "feature_spec",
    "temporal_control_window",
    "temporal_control_kind",
    "model",
    "mean_delta_r2",
    "ci_low_delta_r2",
    "ci_high_delta_r2",
    "p_value_delta_r2_holm",
    "n_folds",
    "n_subjects_included",
    "summary_path",
)
FOLD_COLUMNS = ("model", "fold", "test_subject", "delta_r2")
WINDOW_ROLE_LABELS = {
    "prestimulus_wide": "Pre-stimulus control",
    "immediate_prestimulus": "Immediate pre-stimulus",
    "ramp_up": "Ramp-up wrong-lag",
    "early_plateau": "Early plateau sensitivity",
    "mid_plateau": "Mid-plateau sensitivity",
    "late_plateau": "Late plateau sensitivity",
}


@dataclass(frozen=True)
class TemporalSpecificitySummary:
    """Plot-ready held-out effects and their cohort summaries."""

    targets: tuple[str, ...]
    windows: tuple[str, ...]
    participant_effects: pd.DataFrame
    cohort_effects: pd.DataFrame
    matched_participant_contrasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    matched_cohort_contrasts: pd.DataFrame = field(default_factory=pd.DataFrame)


def temporal_window_label(window: TemporalControlWindow) -> str:
    """Return a concise scientific label with the exact analysis interval."""

    try:
        role = WINDOW_ROLE_LABELS[window.name]
    except KeyError as exc:
        raise ValueError(
            f"No display label is defined for temporal window {window.name!r}."
        ) from exc
    start = f"{window.start:g}".replace("-", "−")
    end = f"{window.end:g}".replace("-", "−")
    return f"{role} ({start}–{end} s)"


def load_temporal_specificity_summary(
    report_path: str | Path,
    config: Any,
) -> TemporalSpecificitySummary:
    """Load and cross-check the current temporal-control LOSO results."""

    resolved_report_path = Path(report_path).expanduser()
    if not resolved_report_path.is_file():
        raise FileNotFoundError(f"Study 1 report does not exist: {resolved_report_path}")

    report = pd.read_csv(resolved_report_path, sep="\t")
    _require_columns(report, REPORT_COLUMNS, source="Study 1 report")
    model, targets = _figure_selection(config)
    windows = resolve_temporal_control_windows(config)
    if not windows:
        raise ValueError("The current Study 1 configuration defines no temporal-control windows.")

    selected = report.loc[
        report["lane"].eq("feature_benchmark")
        & report["analysis_partition"].eq("temporal_control")
        & report["model"].eq(model)
        & report["target"].isin(targets)
    ].copy()
    _validate_protocol_rows(selected, targets=targets, windows=windows)

    participant_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for target in targets:
        for window_order, window in enumerate(windows):
            report_row = selected.loc[
                selected["target"].eq(target) & selected["temporal_control_window"].eq(window.name)
            ].iloc[0]
            participant, cohort = _load_result_cell(
                report_row,
                target=target,
                model=model,
                window=window,
                window_order=window_order,
            )
            participant_rows.extend(participant)
            cohort_rows.append(cohort)

    participant_effects = pd.DataFrame(participant_rows)
    matched_participants, matched_cohort = _matched_primary_control_contrasts(
        report,
        participant_effects=participant_effects,
        targets=targets,
        model=model,
    )
    return TemporalSpecificitySummary(
        targets=targets,
        windows=tuple(window.name for window in windows),
        participant_effects=participant_effects,
        cohort_effects=pd.DataFrame(cohort_rows),
        matched_participant_contrasts=matched_participants,
        matched_cohort_contrasts=matched_cohort,
    )


def _matched_primary_control_contrasts(
    report: pd.DataFrame,
    *,
    participant_effects: pd.DataFrame,
    targets: tuple[str, ...],
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for target in targets:
        primary_rows = report.loc[
            report["lane"].eq("feature_benchmark")
            & report["analysis_partition"].eq("primary")
            & report["target"].eq(target)
            & report["feature_spec"].eq("alpha_beta_gamma")
            & report["model"].eq(model)
        ]
        if len(primary_rows) != 1:
            raise ValueError(
                f"Temporal specificity requires one primary result for {target}/{model}."
            )
        primary_summary = Path(str(primary_rows.iloc[0]["summary_path"])).expanduser()
        primary_folds = pd.read_csv(primary_summary.parent / "model_comparison.tsv", sep="\t")
        _require_columns(primary_folds, FOLD_COLUMNS, source=str(primary_summary.parent))
        primary_folds = primary_folds.loc[primary_folds["model"].eq(model)].copy()
        primary_folds["primary_delta_r2"] = pd.to_numeric(
            primary_folds["delta_r2"], errors="coerce"
        )
        if primary_folds["test_subject"].duplicated().any():
            raise ValueError(f"Primary fold table contains duplicate subjects for {target}.")

        controls = participant_effects.loc[participant_effects["target"].eq(target)].copy()
        matched = controls.merge(
            primary_folds[["test_subject", "primary_delta_r2"]],
            left_on="subject_id",
            right_on="test_subject",
            how="inner",
            validate="many_to_one",
        )
        if len(matched) != len(controls):
            raise ValueError(f"Primary and temporal-control subjects do not match for {target}.")
        matched["control_delta_r2"] = matched["delta_r2"]
        matched["primary_minus_control_delta_r2"] = (
            matched["primary_delta_r2"] - matched["control_delta_r2"]
        )
        records.append(matched.drop(columns=["test_subject", "delta_r2"]))

    participant = pd.concat(records, ignore_index=True)
    cohort = (
        participant.groupby(
            ["target", "model", "window_name", "window_kind", "window_order"],
            sort=False,
        )["primary_minus_control_delta_r2"]
        .agg([("n_subjects", "size"), ("mean_primary_minus_control_delta_r2", "mean")])
        .reset_index()
    )
    return participant, cohort


def _figure_selection(config: Any) -> tuple[str, tuple[str, ...]]:
    raw = require_config_value(config, "study1.figures.temporal_specificity")
    if not isinstance(raw, Mapping):
        raise ValueError("study1.figures.temporal_specificity must be a mapping.")

    model = str(raw.get("model", "")).strip()
    if not model:
        raise ValueError("study1.figures.temporal_specificity.model must be non-empty.")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("study1.figures.temporal_specificity.targets must be a non-empty list.")
    targets = tuple(str(target).strip() for target in raw_targets)
    if any(not target for target in targets) or len(set(targets)) != len(targets):
        raise ValueError(
            "study1.figures.temporal_specificity.targets must contain unique, non-empty names."
        )
    return model, targets


def _validate_protocol_rows(
    selected: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    windows: tuple[TemporalControlWindow, ...],
) -> None:
    observed_targets = set(selected["target"].astype(str))
    expected_targets = set(targets)
    if observed_targets != expected_targets:
        raise ValueError(
            "Temporal-specificity target set does not match figure configuration: "
            f"expected {sorted(expected_targets)}, observed {sorted(observed_targets)}."
        )

    expected_windows = {window.name for window in windows}
    for target in targets:
        target_rows = selected.loc[selected["target"].eq(target)]
        observed_windows = set(target_rows["temporal_control_window"].astype(str))
        if observed_windows != expected_windows:
            raise ValueError(
                "Temporal-specificity window set does not match current configuration for "
                f"{target}: expected {sorted(expected_windows)}, "
                f"observed {sorted(observed_windows)}."
            )
        duplicate_windows = target_rows["temporal_control_window"].duplicated(keep=False)
        if duplicate_windows.any():
            names = sorted(target_rows.loc[duplicate_windows, "temporal_control_window"].unique())
            raise ValueError(f"Duplicate temporal-specificity report rows for {target}: {names}.")

    window_by_name = {window.name: window for window in windows}
    for row in selected.itertuples(index=False):
        window = window_by_name[str(row.temporal_control_window)]
        if str(row.feature_spec) != window.feature_spec:
            raise ValueError(
                f"Feature specification for {row.target}/{window.name} must be "
                f"{window.feature_spec!r}; observed {row.feature_spec!r}."
            )
        if str(row.temporal_control_kind) != window.kind:
            raise ValueError(
                f"Temporal-control kind for {row.target}/{window.name} must be "
                f"{window.kind!r}; observed {row.temporal_control_kind!r}."
            )


def _load_result_cell(
    report_row: pd.Series,
    *,
    target: str,
    model: str,
    window: TemporalControlWindow,
    window_order: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mean = _finite_float(report_row["mean_delta_r2"], field="mean_delta_r2")
    ci_low = _finite_float(report_row["ci_low_delta_r2"], field="ci_low_delta_r2")
    ci_high = _finite_float(report_row["ci_high_delta_r2"], field="ci_high_delta_r2")
    p_holm = _finite_float(
        report_row["p_value_delta_r2_holm"],
        field="p_value_delta_r2_holm",
    )
    n_folds = _positive_integer(report_row["n_folds"], field="n_folds")
    n_subjects = _positive_integer(
        report_row["n_subjects_included"],
        field="n_subjects_included",
    )
    if ci_low > mean or mean > ci_high:
        raise ValueError(f"Invalid confidence interval for {target}/{window.name}.")
    if not 0.0 <= p_holm <= 1.0:
        raise ValueError(f"Adjusted p-value for {target}/{window.name} must lie in [0, 1].")

    summary_path = Path(str(report_row["summary_path"])).expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Model summary does not exist: {summary_path}")
    fold_path = summary_path.parent / "model_comparison.tsv"
    if not fold_path.is_file():
        raise FileNotFoundError(f"Temporal-specificity fold table does not exist: {fold_path}")

    folds = pd.read_csv(fold_path, sep="\t")
    _require_columns(folds, FOLD_COLUMNS, source=str(fold_path))
    model_folds = folds.loc[folds["model"].eq(model)].copy()
    if model_folds.empty:
        raise ValueError(f"Fold table contains no {model!r} rows: {fold_path}")
    if (
        model_folds["test_subject"].isna().any()
        or model_folds["test_subject"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"Fold table contains empty held-out subject identifiers: {fold_path}")
    if model_folds["test_subject"].duplicated().any():
        raise ValueError(f"Fold table contains duplicate held-out subjects: {fold_path}")

    delta_r2 = pd.to_numeric(model_folds["delta_r2"], errors="coerce").to_numpy(float)
    if not np.isfinite(delta_r2).all():
        raise ValueError(f"Fold table contains non-finite delta_r2 values: {fold_path}")
    if len(model_folds) != n_folds:
        raise ValueError(
            f"Fold count for {target}/{window.name} is {len(model_folds)}, expected {n_folds}."
        )
    if model_folds["test_subject"].nunique() != n_subjects:
        raise ValueError(
            f"Subject count for {target}/{window.name} is "
            f"{model_folds['test_subject'].nunique()}, expected {n_subjects}."
        )
    fold_mean = float(delta_r2.mean())
    if not np.isclose(fold_mean, mean, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"Temporal-specificity fold mean does not match report mean for "
            f"{target}/{window.name}: {fold_mean:g} versus {mean:g}."
        )

    common = {
        "target": target,
        "model": model,
        "window_name": window.name,
        "window_kind": window.kind,
        "window_start_s": window.start,
        "window_end_s": window.end,
        "window_order": window_order,
        "window_label": temporal_window_label(window),
    }
    participant_rows = [
        {
            **common,
            "fold": fold,
            "subject_id": subject,
            "delta_r2": effect,
        }
        for fold, subject, effect in zip(
            model_folds["fold"],
            model_folds["test_subject"].astype(str),
            delta_r2,
            strict=True,
        )
    ]
    cohort_row = {
        **common,
        "feature_spec": window.feature_spec,
        "mean_delta_r2": mean,
        "ci_low_delta_r2": ci_low,
        "ci_high_delta_r2": ci_high,
        "p_value_delta_r2_holm": p_holm,
        "n_folds": n_folds,
        "n_subjects": n_subjects,
        "summary_path": str(summary_path),
        "fold_table_path": str(fold_path),
    }
    return participant_rows, cohort_row


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
    "TemporalSpecificitySummary",
    "load_temporal_specificity_summary",
    "temporal_window_label",
]
