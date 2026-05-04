"""Study 1 primary target preparation."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import load_events_df
from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root, resolve_fmri_bids_root
from eeg_pipeline.utils.data.fmri_signature_targets import (
    find_block_column,
    load_fmri_signature_target_for_subject,
)
from fmri_pipeline.analysis.trial_signatures import (
    TrialSignatureExtractionConfig,
    run_trial_signature_extraction_for_subject,
)
from fmri_pipeline.utils.signature_paths import discover_signature_root_and_specs

PRIMARY_SIGNATURES = ("NPS", "SIIPS1")


def _study1_output_root(config: Any) -> Path:
    root_name = str(get_config_value(config, "study1.outputs.root_name", "study1")).strip()
    if not root_name:
        raise ValueError("study1.outputs.root_name must be a non-empty string.")
    return resolve_eeg_deriv_root(config) / "group" / "multimodal" / root_name


def _study1_target_config(config: Any) -> dict[str, Any]:
    raw = config.get("study1.targets")
    if not isinstance(raw, dict):
        raise ValueError("study1.targets must be a mapping.")
    return raw


def _primary_signatures(config: Any) -> tuple[str, str]:
    raw_names = require_config_value(config, "study1.targets.names")
    if not isinstance(raw_names, (list, tuple)):
        raise ValueError("study1.targets.names must be a list containing NPS and SIIPS1.")
    names = tuple(str(name).strip() for name in raw_names if str(name).strip())
    if names != PRIMARY_SIGNATURES:
        raise ValueError("study1.targets.names must be exactly ['NPS', 'SIIPS1'] in that order.")
    return names


def _validate_signature_space(config: Any, signature_specs: list[dict[str, str]]) -> None:
    if not signature_specs:
        raise ValueError("paths.signature_maps must include NPS and SIIPS1.")

    names = {spec["name"] for spec in signature_specs}
    missing = [name for name in PRIMARY_SIGNATURES if name not in names]
    if missing:
        raise ValueError(f"Missing required primary signature maps: {missing}")

    space = str(require_config_value(config, "study1.targets.fmriprep_space")).strip().lower()
    if "mni" not in space:
        raise ValueError(
            "Study 1 target preparation requires MNI-space fMRI inputs for signature extraction."
        )
    _validate_signature_provenance(config, signature_specs)


def _validate_signature_provenance(config: Any, signature_specs: list[dict[str, str]]) -> None:
    expected = get_config_value(config, "study1.targets.signature_provenance", None)
    if not isinstance(expected, dict):
        raise ValueError(
            "study1.targets.signature_provenance must define expected NPS/SIIPS1 maps."
        )

    signature_root = Path(str(require_config_value(config, "paths.signature_dir"))).expanduser()
    specs_by_name = {str(spec["name"]).strip(): spec for spec in signature_specs}
    for name in PRIMARY_SIGNATURES:
        expected_spec = expected.get(name)
        if not isinstance(expected_spec, dict):
            raise ValueError(f"Missing Study 1 signature provenance entry for {name}.")
        expected_path = str(expected_spec.get("path", "")).strip()
        if not expected_path:
            raise ValueError(f"Study 1 signature provenance for {name} must define a path.")
        actual_path = str(specs_by_name[name].get("path", "")).strip()
        if actual_path != expected_path:
            raise ValueError(
                "Study 1 signature provenance mismatch for "
                f"{name}: expected {expected_path!r}, got {actual_path!r}."
            )

        expected_space = str(expected_spec.get("space", "")).strip().lower()
        configured_space = (
            str(require_config_value(config, "study1.targets.fmriprep_space")).strip().lower()
        )
        if expected_space != configured_space:
            raise ValueError(
                "Study 1 signature provenance space mismatch for "
                f"{name}: expected {expected_space!r}, configured {configured_space!r}."
            )
        _validate_signature_image(signature_root / actual_path, name=name)


def _validate_signature_image(path: Path, *, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Study 1 signature map for {name} does not exist: {path}")
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature provenance validation requires nibabel.") from exc

    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Study 1 signature map for {name} must be 3D, got shape {image.shape}.")
    data = np.asanyarray(image.dataobj, dtype=float)
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError(f"Study 1 signature map for {name} contains no finite voxels: {path}")
    if not np.any(np.abs(data[finite]) > 0):
        raise ValueError(f"Study 1 signature map for {name} contains only zero weights: {path}")


def _build_trial_signature_config(config: Any, *, task: str) -> TrialSignatureExtractionConfig:
    target_cfg = _study1_target_config(config)
    return TrialSignatureExtractionConfig(
        input_source=str(require_config_value(config, "study1.targets.input_source")).strip(),
        fmriprep_space=str(require_config_value(config, "study1.targets.fmriprep_space")).strip(),
        require_fmriprep=bool(require_config_value(config, "study1.targets.require_fmriprep")),
        runs=None,
        task=task,
        name=str(require_config_value(config, "study1.targets.contrast_name")).strip(),
        condition_a_column=str(
            require_config_value(config, "study1.targets.condition_a_column")
        ).strip(),
        condition_a_value=str(
            require_config_value(config, "study1.targets.condition_a_value")
        ).strip(),
        condition_b_column=str(
            require_config_value(config, "study1.targets.condition_b_column")
        ).strip(),
        condition_b_value=str(
            require_config_value(config, "study1.targets.condition_b_value")
        ).strip(),
        hrf_model=str(require_config_value(config, "study1.targets.hrf_model")).strip(),
        drift_model=str(require_config_value(config, "study1.targets.drift_model")).strip(),
        high_pass_hz=float(require_config_value(config, "study1.targets.high_pass_hz")),
        low_pass_hz=target_cfg.get("low_pass_hz"),
        smoothing_fwhm=target_cfg.get("smoothing_fwhm"),
        confounds_strategy=str(
            require_config_value(config, "study1.targets.confounds_strategy")
        ).strip(),
        method=str(require_config_value(config, "study1.targets.method")).strip(),
        lss_other_regressors=str(
            require_config_value(config, "study1.targets.lss_other_regressors")
        ).strip(),
        condition_scope_trial_type_column=str(
            target_cfg.get("condition_scope_trial_type_column", "") or ""
        ).strip(),
        condition_scope_phase_column=str(
            target_cfg.get("condition_scope_phase_column", "") or ""
        ).strip(),
        condition_scope_trial_types=_optional_string_tuple(
            target_cfg.get("condition_scope_trial_types"),
            field_name="study1.targets.condition_scope_trial_types",
        ),
        condition_scope_stim_phases=_optional_string_tuple(
            target_cfg.get("condition_scope_stim_phases"),
            field_name="study1.targets.condition_scope_stim_phases",
        ),
        signatures=PRIMARY_SIGNATURES,
    )


def _optional_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings when provided.")
    normalized = tuple(str(item).strip() for item in value if str(item).strip())
    return normalized or None


def _config_for_signature(config: Any, signature_name: str) -> Any:
    config_copy = copy.deepcopy(config)
    study1_cfg = config_copy.setdefault("study1", {})
    targets_cfg = study1_cfg.setdefault("targets", {})
    targets_cfg["signature_name"] = signature_name
    return config_copy


def nuisance_regression_enabled(config: Any) -> bool:
    return bool(get_config_value(config, "study1.targets.nuisance_regression.enabled", False))


def nuisance_continuous_columns(config: Any) -> tuple[str, ...]:
    raw_columns = get_config_value(
        config,
        "study1.targets.nuisance_regression.continuous_columns",
        [],
    )
    return _required_string_tuple(
        raw_columns,
        field_name="study1.targets.nuisance_regression.continuous_columns",
    )


def nuisance_categorical_columns(config: Any) -> tuple[str, ...]:
    raw_columns = get_config_value(
        config,
        "study1.targets.nuisance_regression.categorical_columns",
        [],
    )
    return _required_string_tuple(
        raw_columns,
        field_name="study1.targets.nuisance_regression.categorical_columns",
    )


def nuisance_source_columns(config: Any) -> tuple[str, ...]:
    continuous = nuisance_continuous_columns(config)
    categorical = nuisance_categorical_columns(config)
    columns = continuous + categorical
    if nuisance_regression_enabled(config) and not columns:
        raise ValueError(
            "Study 1 nuisance regression requires at least one continuous or categorical column."
        )
    if len(columns) != len(set(columns)):
        raise ValueError("Study 1 nuisance regression columns must be unique.")
    return columns


def residualization_columns_for_target_table(config: Any, table_path: Path) -> tuple[str, ...]:
    target_table = pd.read_parquet(table_path)
    return resolve_residualization_columns(frame=target_table, config=config)


def resolve_residualization_columns(*, frame: pd.DataFrame, config: Any) -> tuple[str, ...]:
    if not nuisance_regression_enabled(config):
        return tuple()

    continuous = nuisance_continuous_columns(config)
    missing = [column for column in continuous if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 1 target table is missing continuous nuisance columns: {missing}.")

    dummy_columns: list[str] = []
    for column in nuisance_categorical_columns(config):
        dummy_columns.extend(_categorical_dummy_columns(frame=frame, column=column))

    columns = continuous + tuple(dummy_columns)
    if not columns:
        raise ValueError("Study 1 nuisance regression resolved to no target-table columns.")
    return columns


def _required_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of column names.")
    return tuple(str(column).strip() for column in value if str(column).strip())


def _append_categorical_nuisance_columns(
    *,
    frame: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    out = frame.copy()
    for column in nuisance_categorical_columns(config):
        _validate_categorical_source(out, column=column)
        levels = _ordered_categorical_levels(out[column])
        if len(levels) < 2:
            raise ValueError(
                f"Study 1 categorical nuisance column '{column}' must contain at least two levels."
            )
        for level in levels[1:]:
            dummy_column = _categorical_level_column(column, level)
            if dummy_column in out.columns:
                raise ValueError(
                    f"Study 1 categorical nuisance dummy column already exists: {dummy_column}."
                )
            out[dummy_column] = (out[column] == level).astype(float)
    return out


def _categorical_dummy_columns(*, frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    _validate_categorical_source(frame, column=column)
    levels = _ordered_categorical_levels(frame[column])
    if len(levels) < 2:
        raise ValueError(
            f"Study 1 categorical nuisance column '{column}' must contain at least two levels."
        )

    dummy_columns = tuple(_categorical_level_column(column, level) for level in levels[1:])
    missing = [dummy_column for dummy_column in dummy_columns if dummy_column not in frame.columns]
    if missing:
        raise ValueError(
            f"Study 1 target table is missing categorical nuisance dummies: {missing}."
        )
    return dummy_columns


def _validate_categorical_source(frame: pd.DataFrame, *, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"Study 1 categorical nuisance column '{column}' is missing.")
    if frame[column].isna().any():
        raise ValueError(f"Study 1 categorical nuisance column '{column}' contains missing values.")


def _ordered_categorical_levels(values: pd.Series) -> tuple[Any, ...]:
    unique = list(pd.unique(values))
    numeric = pd.to_numeric(pd.Series(unique), errors="coerce")
    if numeric.notna().all():
        ordered = sorted(unique, key=lambda value: float(value))
    else:
        ordered = sorted(unique, key=lambda value: str(value))
    return tuple(ordered)


def _categorical_level_column(column: str, level: Any) -> str:
    return f"{column}_level_{_level_token(level)}"


def _level_token(level: Any) -> str:
    numeric = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        text = str(float(numeric))
    else:
        text = str(level).strip()
    token = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    if not token:
        raise ValueError(f"Cannot create a categorical nuisance dummy name for level {level!r}.")
    if token[0].isdigit():
        return token
    return token.lower()


def _append_required_event_columns(
    *,
    frame: pd.DataFrame,
    events_df: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    columns = nuisance_source_columns(config) if nuisance_regression_enabled(config) else tuple()
    for column in columns:
        if column in frame.columns:
            continue
        if column not in events_df.columns:
            raise ValueError(
                f"Study 1 nuisance regression column '{column}' is missing from clean EEG events."
            )
        frame[column] = events_df[column].reset_index(drop=True)
    return frame


def _subject_target_rows(
    *,
    subject: str,
    task: str,
    config: Any,
    deriv_root: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    events_df = load_events_df(subject, task, config=config, prefer_clean=True)
    if events_df is None or events_df.empty:
        raise FileNotFoundError(
            f"Clean events.tsv not found (or empty) for sub-{subject}, task-{task}."
        )
    events_df = events_df.reset_index(drop=True)

    nps, _nps_label, nps_extra = load_fmri_signature_target_for_subject(
        subject_raw=subject,
        task=task,
        deriv_root=deriv_root,
        config=_config_for_signature(config, "NPS"),
        events_df=events_df,
        logger=logger,
        config_path="study1.targets",
    )
    siips1, _siips1_label, siips1_extra = load_fmri_signature_target_for_subject(
        subject_raw=subject,
        task=task,
        deriv_root=deriv_root,
        config=_config_for_signature(config, "SIIPS1"),
        events_df=events_df,
        logger=logger,
        config_path="study1.targets",
    )

    block = find_block_column(events_df)
    if block is None:
        block = pd.Series(pd.NA, index=events_df.index, dtype="float64")

    if "trial_number" in events_df.columns:
        trial_index = pd.to_numeric(events_df["trial_number"], errors="coerce")
    elif "trial_index" in events_df.columns:
        trial_index = pd.to_numeric(events_df["trial_index"], errors="coerce")
    else:
        trial_index = pd.Series(range(1, len(events_df) + 1), dtype="int64")

    frame = pd.DataFrame(
        {
            "subject_id": f"sub-{subject}" if not str(subject).startswith("sub-") else str(subject),
            "task": task,
            "block": block,
            "trial_index": trial_index,
            "onset": pd.to_numeric(events_df["onset"], errors="coerce"),
            "duration": pd.to_numeric(events_df["duration"], errors="coerce"),
            "NPS": pd.to_numeric(nps, errors="coerce"),
            "SIIPS1": pd.to_numeric(siips1, errors="coerce"),
        }
    )
    frame = _append_signature_metadata(frame=frame, prefix="NPS", extra=nps_extra)
    frame = _append_signature_metadata(frame=frame, prefix="SIIPS1", extra=siips1_extra)
    frame = _append_required_event_columns(frame=frame, events_df=events_df, config=config)
    if not frame["NPS"].notna().all() or not frame["SIIPS1"].notna().all():
        raise ValueError(
            f"Study 1 primary target table requires finite values for both NPS and SIIPS1 "
            f"for every retained trial in sub-{subject}."
        )
    return frame


def _append_signature_metadata(
    *,
    frame: pd.DataFrame,
    prefix: str,
    extra: pd.DataFrame,
) -> pd.DataFrame:
    if extra is None or extra.empty:
        return frame
    if len(extra) != len(frame):
        raise ValueError(
            f"Study 1 signature metadata length mismatch for {prefix}: "
            f"metadata={len(extra)}, target_rows={len(frame)}."
        )
    for column in extra.columns:
        values = pd.to_numeric(extra[column], errors="coerce")
        if not values.notna().any():
            continue
        frame[f"{prefix}_{column}"] = values.reset_index(drop=True)
    return frame


def _validate_signature_voxel_counts(frame: pd.DataFrame) -> None:
    for prefix in PRIMARY_SIGNATURES:
        column = f"{prefix}_fmri_n_voxels"
        if column not in frame.columns:
            raise ValueError(
                f"Study 1 target table is missing required voxel-count column: {column}."
            )
        counts = pd.to_numeric(frame[column], errors="coerce")
        if not counts.notna().all():
            raise ValueError(f"Study 1 voxel-count column '{column}' contains non-finite values.")
        unique_counts = sorted({int(value) for value in counts.to_numpy(dtype=float)})
        if len(unique_counts) != 1:
            raise ValueError(
                f"Study 1 {prefix} scoring requires an identical voxel count across all "
                f"retained observations, got {unique_counts}."
            )


def _target_output_paths(config: Any) -> tuple[Path, Path]:
    base = _study1_output_root(config) / "targets"
    return base / "primary_targets.parquet", base / "primary_targets.tsv"


def prepare_primary_targets(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    logger: logging.Logger | None = None,
) -> Path:
    if logger is None:
        logger = logging.getLogger(__name__)
    if not subjects:
        raise ValueError("No subjects specified for Study 1 target preparation.")

    _primary_signatures(config)
    deriv_root = resolve_eeg_deriv_root(config)
    bids_fmri_root = resolve_fmri_bids_root(config, task_is_rest=False)
    signature_root, signature_specs = discover_signature_root_and_specs(config, deriv_root)
    _validate_signature_space(config, signature_specs)
    trial_cfg = _build_trial_signature_config(config, task=task)

    subject_frames: list[pd.DataFrame] = []
    for subject in subjects:
        result = run_trial_signature_extraction_for_subject(
            bids_fmri_root=bids_fmri_root,
            bids_derivatives=deriv_root,
            deriv_root=deriv_root,
            subject=subject,
            cfg=trial_cfg,
            signature_root=signature_root,
            signature_specs=signature_specs,
        )
        logger.info(
            "Prepared trial signatures for sub-%s at %s",
            subject,
            result.get("output_dir", "unknown"),
        )
        subject_frames.append(
            _subject_target_rows(
                subject=subject,
                task=task,
                config=config,
                deriv_root=deriv_root,
                logger=logger,
            )
        )

    primary_table = pd.concat(subject_frames, axis=0, ignore_index=True)
    _validate_signature_voxel_counts(primary_table)
    if nuisance_regression_enabled(config):
        primary_table = _append_categorical_nuisance_columns(frame=primary_table, config=config)
    parquet_path, tsv_path = _target_output_paths(config)
    write_parquet(primary_table, parquet_path)
    write_tsv(primary_table, tsv_path)
    return parquet_path


def iter_primary_subjects(primary_table: pd.DataFrame) -> Iterable[str]:
    return sorted({str(subject).strip() for subject in primary_table["subject_id"].tolist()})


__all__ = [
    "PRIMARY_SIGNATURES",
    "iter_primary_subjects",
    "nuisance_categorical_columns",
    "nuisance_continuous_columns",
    "nuisance_regression_enabled",
    "nuisance_source_columns",
    "prepare_primary_targets",
    "residualization_columns_for_target_table",
    "resolve_residualization_columns",
]
