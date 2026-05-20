"""Study 1 cohort resolution from the prepared primary target table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root

REQUIRED_PRIMARY_COLUMNS = (
    "subject_id",
    "task",
    "block",
    "trial_index",
    "within_block_trial",
    "onset",
    "duration",
    "NPS",
    "SIIPS1",
)


def study1_output_root(config: Any) -> Path:
    root_name = str(get_config_value(config, "study1.outputs.root_name", "study1")).strip()
    if not root_name:
        raise ValueError("study1.outputs.root_name must be a non-empty string.")
    return resolve_eeg_deriv_root(config) / "group" / "multimodal" / root_name


def primary_targets_dir(config: Any) -> Path:
    return study1_output_root(config) / "targets"


def study1_feature_root(config: Any) -> Path:
    return study1_output_root(config) / "features_trial_ml_safe"


def study1_feature_subject_root(config: Any, subject_id: str) -> Path:
    return study1_feature_root(config) / _normalize_subject(subject_id)


def study1_feature_family_dir(
    config: Any,
    subject_id: str,
    family: str,
) -> Path:
    family_name = str(family).strip()
    if not family_name:
        raise ValueError("Feature family names must be non-empty.")
    return study1_feature_subject_root(config, subject_id) / "eeg" / "features" / family_name


def study1_feature_table_path(
    config: Any,
    subject_id: str,
    family: str,
) -> Path:
    family_name = str(family).strip()
    if not family_name:
        raise ValueError("Feature family names must be non-empty.")
    return (
        study1_feature_family_dir(config, subject_id, family_name)
        / f"features_{family_name}.parquet"
    )


def study1_feature_metadata_path(
    config: Any,
    subject_id: str,
    family: str,
) -> Path:
    return (
        study1_feature_family_dir(config, subject_id, family)
        / "metadata"
        / "extraction_config.json"
    )


def primary_targets_parquet_path(config: Any) -> Path:
    return primary_targets_dir(config) / "primary_targets.parquet"


def primary_targets_tsv_path(config: Any) -> Path:
    return primary_targets_dir(config) / "primary_targets.tsv"


def load_primary_target_table(config: Any) -> pd.DataFrame:
    target_path = primary_targets_parquet_path(config)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Study 1 primary target table not found: {target_path}. "
            "Run 'signature-prediction prepare-targets' first."
        )

    frame = pd.read_parquet(target_path)
    missing = [column for column in REQUIRED_PRIMARY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Study 1 primary target table is missing required columns: "
            f"{missing}. Path: {target_path}"
        )
    if frame.empty:
        raise ValueError(f"Study 1 primary target table is empty: {target_path}")
    return frame


def _normalize_subject(subject: str) -> str:
    subject_label = str(subject).strip()
    if not subject_label:
        raise ValueError("Subject identifiers must be non-empty.")
    if subject_label.startswith("sub-"):
        return subject_label
    return f"sub-{subject_label}"


def resolve_primary_subjects(
    *,
    requested_subjects: list[str],
    task: str,
    config: Any,
) -> list[str]:
    min_subjects = int(get_config_value(config, "study1.cohort.min_subjects", 2))
    table = load_primary_target_table(config)
    task_rows = table.loc[table["task"].astype(str) == str(task)].copy()
    if task_rows.empty:
        raise ValueError(f"Study 1 primary target table contains no rows for task '{task}'.")

    available_subjects = sorted(
        {_normalize_subject(subject) for subject in task_rows["subject_id"].astype(str).tolist()}
    )
    requested = sorted({_normalize_subject(subject) for subject in requested_subjects})
    if requested:
        missing = [subject for subject in requested if subject not in available_subjects]
        if missing:
            raise ValueError(
                "Requested subjects are missing from the Study 1 primary target table: "
                f"{missing}"
            )
        resolved = requested
    else:
        resolved = available_subjects

    if len(resolved) < min_subjects:
        raise ValueError(
            f"Study 1 requires at least {min_subjects} subjects for LOSO analyses, got {len(resolved)}."
        )
    return resolved


def subject_target_rows(
    *,
    subject_id: str,
    task: str,
    config: Any,
) -> pd.DataFrame:
    subject_label = _normalize_subject(subject_id)
    table = load_primary_target_table(config)
    subject_rows = table.loc[
        (table["subject_id"].astype(str) == subject_label)
        & (table["task"].astype(str) == str(task))
    ].copy()
    if subject_rows.empty:
        raise ValueError(
            f"Study 1 primary target table contains no rows for {subject_label}, task '{task}'."
        )
    return subject_rows.reset_index(drop=True)


def resolve_primary_target_name(config: Any, target_name: str) -> str:
    names = require_config_value(config, "study1.targets.names")
    if not isinstance(names, (list, tuple)):
        raise ValueError("study1.targets.names must be a list containing 'NPS' and 'SIIPS1'.")

    allowed = {str(name).strip() for name in names if str(name).strip()}
    resolved = str(target_name).strip()
    if resolved not in allowed:
        raise ValueError(
            f"Invalid Study 1 target '{target_name}'. Expected one of {sorted(allowed)}."
        )
    return resolved


__all__ = [
    "load_primary_target_table",
    "primary_targets_dir",
    "study1_feature_family_dir",
    "study1_feature_metadata_path",
    "primary_targets_parquet_path",
    "primary_targets_tsv_path",
    "resolve_primary_subjects",
    "resolve_primary_target_name",
    "study1_feature_root",
    "study1_feature_subject_root",
    "study1_feature_table_path",
    "study1_output_root",
    "subject_target_rows",
]
