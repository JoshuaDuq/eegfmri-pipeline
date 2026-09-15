"""Validated data assembly for the Study 1 spectral-specificity figure."""

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
FEATURE_LABELS = {
    "alpha": "Alpha (8–12.9 Hz)",
    "beta": "Beta (13–30 Hz)",
    "gamma": "Gamma (30.1–77 Hz)",
    "alpha_beta": "Alpha + beta",
    "alpha_beta_gamma": "Alpha + beta + gamma",
}


@dataclass(frozen=True)
class SpectralSpecificitySummary:
    """Plot-ready participant and cohort spectral estimates."""

    targets: tuple[str, ...]
    feature_specs: tuple[str, ...]
    participant_effects: pd.DataFrame
    cohort_effects: pd.DataFrame


@dataclass(frozen=True)
class SpectralSpecificitySelection:
    model: str
    targets: tuple[str, ...]
    feature_specs: tuple[str, ...]


def load_spectral_specificity_summary(
    report_path: str | Path,
    config: Any,
) -> SpectralSpecificitySummary:
    """Load and cross-check confirmatory spectral LOSO estimates."""

    resolved_report_path = Path(report_path).expanduser()
    if not resolved_report_path.is_file():
        raise FileNotFoundError(f"Study 1 report does not exist: {resolved_report_path}")

    report = pd.read_csv(resolved_report_path, sep="\t")
    _require_columns(report, REPORT_COLUMNS, source="Study 1 report")
    selection = _figure_selection(config)
    selected = report.loc[
        report["lane"].eq("feature_benchmark")
        & report["analysis_partition"].eq("primary")
        & report["model"].eq(selection.model)
        & report["target"].isin(selection.targets)
        & report["feature_spec"].isin(selection.feature_specs)
    ].copy()
    _validate_selected_rows(selected, selection)

    participant_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for target in selection.targets:
        for feature_order, feature_spec in enumerate(selection.feature_specs):
            report_row = selected.loc[
                selected["target"].eq(target) & selected["feature_spec"].eq(feature_spec)
            ].iloc[0]
            participants, cohort = _load_cell(
                report_row,
                target=target,
                feature_spec=feature_spec,
                feature_order=feature_order,
                model=selection.model,
            )
            participant_rows.extend(participants)
            cohort_rows.append(cohort)

    return SpectralSpecificitySummary(
        targets=selection.targets,
        feature_specs=selection.feature_specs,
        participant_effects=pd.DataFrame(participant_rows),
        cohort_effects=pd.DataFrame(cohort_rows),
    )


def _figure_selection(config: Any) -> SpectralSpecificitySelection:
    raw = require_config_value(config, "study1.figures.spectral_specificity")
    if not isinstance(raw, Mapping):
        raise ValueError("study1.figures.spectral_specificity must be a mapping.")
    model = str(raw.get("model", "")).strip()
    targets = _unique_strings(raw.get("targets"), field="targets")
    feature_specs = _unique_strings(raw.get("feature_specs"), field="feature_specs")
    if not model:
        raise ValueError("study1.figures.spectral_specificity.model must be non-empty.")
    missing_labels = sorted(set(feature_specs).difference(FEATURE_LABELS))
    if missing_labels:
        raise ValueError(f"Spectral feature labels are undefined: {missing_labels}.")
    return SpectralSpecificitySelection(
        model=model,
        targets=targets,
        feature_specs=feature_specs,
    )


def _unique_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"study1.figures.spectral_specificity.{field} must be a non-empty list.")
    strings = tuple(str(item).strip() for item in value)
    if any(not item for item in strings) or len(set(strings)) != len(strings):
        raise ValueError(f"Spectral-specificity {field} must be unique and non-empty.")
    return strings


def _validate_selected_rows(
    selected: pd.DataFrame,
    selection: SpectralSpecificitySelection,
) -> None:
    observed_targets = set(selected["target"].astype(str))
    expected_targets = set(selection.targets)
    if observed_targets != expected_targets:
        raise ValueError(
            "Spectral-specificity target set does not match figure configuration: "
            f"expected {sorted(expected_targets)}, observed {sorted(observed_targets)}."
        )
    expected_features = set(selection.feature_specs)
    for target in selection.targets:
        target_rows = selected.loc[selected["target"].eq(target)]
        observed_features = set(target_rows["feature_spec"].astype(str))
        if observed_features != expected_features:
            raise ValueError(
                "Spectral-specificity feature set does not match figure configuration for "
                f"{target}: expected {sorted(expected_features)}, "
                f"observed {sorted(observed_features)}."
            )
        duplicates = target_rows["feature_spec"].duplicated(keep=False)
        if duplicates.any():
            features = sorted(target_rows.loc[duplicates, "feature_spec"].unique())
            raise ValueError(
                f"Duplicate spectral-specificity report rows for {target}: {features}."
            )


def _load_cell(
    report_row: pd.Series,
    *,
    target: str,
    feature_spec: str,
    feature_order: int,
    model: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mean = _finite_float(report_row["mean_delta_r2"], field="mean_delta_r2")
    ci_low = _finite_float(report_row["ci_low_delta_r2"], field="ci_low_delta_r2")
    ci_high = _finite_float(report_row["ci_high_delta_r2"], field="ci_high_delta_r2")
    p_value = _probability(report_row["p_value_delta_r2"], field="p_value_delta_r2")
    p_holm = _probability(
        report_row["p_value_delta_r2_holm"],
        field="p_value_delta_r2_holm",
    )
    n_folds = _positive_integer(report_row["n_folds"], field="n_folds")
    n_subjects = _positive_integer(
        report_row["n_subjects_included"],
        field="n_subjects_included",
    )
    if ci_low > mean or mean > ci_high:
        raise ValueError(f"Invalid ΔR² confidence interval for {target}/{feature_spec}.")

    summary_path = Path(str(report_row["summary_path"])).expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Model summary does not exist: {summary_path}")
    fold_path = summary_path.parent / "model_comparison.tsv"
    if not fold_path.is_file():
        raise FileNotFoundError(f"Spectral-specificity fold table does not exist: {fold_path}")

    folds = pd.read_csv(fold_path, sep="\t")
    _require_columns(folds, FOLD_COLUMNS, source=str(fold_path))
    model_folds = folds.loc[folds["model"].eq(model)].copy()
    if model_folds.empty:
        raise ValueError(f"Fold table contains no {model!r} rows: {fold_path}")
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
        raise ValueError(
            f"Fold delta_r2 does not equal r2 - r2_nuisance for "
            f"{target}/{feature_spec}: {fold_path}"
        )
    if len(model_folds) != n_folds:
        raise ValueError(
            f"Fold count for {target}/{feature_spec} is {len(model_folds)}, expected {n_folds}."
        )
    if subjects.nunique() != n_subjects:
        raise ValueError(
            f"Subject count for {target}/{feature_spec} is "
            f"{subjects.nunique()}, expected {n_subjects}."
        )
    fold_mean = float(numeric["delta_r2"].mean())
    if not np.isclose(fold_mean, mean, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"Spectral-specificity fold mean does not match report mean for "
            f"{target}/{feature_spec}: {fold_mean:g} versus {mean:g}."
        )

    common = {
        "target": target,
        "model": model,
        "feature_spec": feature_spec,
        "feature_order": feature_order,
        "feature_label": FEATURE_LABELS[feature_spec],
    }
    participant_rows = [
        {
            **common,
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
        **common,
        "mean_delta_r2": mean,
        "ci_low_delta_r2": ci_low,
        "ci_high_delta_r2": ci_high,
        "p_value_delta_r2": p_value,
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


def _probability(value: object, *, field: str) -> float:
    probability = _finite_float(value, field=field)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]; observed {value!r}.")
    return probability


def _positive_integer(value: object, *, field: str) -> int:
    number = _finite_float(value, field=field)
    if number <= 0 or not number.is_integer():
        raise ValueError(f"{field} must be a positive integer; observed {value!r}.")
    return int(number)


__all__ = [
    "SpectralSpecificitySummary",
    "load_spectral_specificity_summary",
]
