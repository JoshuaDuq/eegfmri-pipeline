"""Study 1 primary target preparation."""

from __future__ import annotations

import copy
import hashlib
import json
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
    load_fmri_signature_target_for_subject,
    parse_run_label_to_int,
)
from fmri_pipeline.analysis.trial_signatures import (
    TrialSignatureExtractionConfig,
    run_trial_signature_extraction_for_subject,
)
from fmri_pipeline.utils.signature_paths import discover_signature_root_and_specs

PRIMARY_SIGNATURES = ("NPS", "SIIPS1")
RAW_LEVEL2_ARTIFACT_COLUMNS = {
    "framewise_displacement": "hrf_weighted_framewise_displacement",
    "std_dvars": "hrf_weighted_std_dvars",
    "fp1_fp2_high_frequency_power": "hrf_weighted_fp1_fp2_high_frequency_power",
}
ACQUISITION_RUN_COLUMNS = ("run_id", "run", "session")


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
    _validate_signature_manifest(config, signature_specs)


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


def _signature_manifest_path(config: Any) -> Path:
    raw_value = get_config_value(config, "study1.targets.signature_manifest_path", None)
    if raw_value is None:
        raise ValueError("study1.targets.signature_manifest_path must be configured.")
    raw_path = str(raw_value).strip()
    if not raw_path:
        raise ValueError("study1.targets.signature_manifest_path must be non-empty.")
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    signature_root = Path(str(require_config_value(config, "paths.signature_dir"))).expanduser()
    return signature_root / path


def _load_signature_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Study 1 frozen signature manifest does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature manifest validation requires PyYAML.") from exc

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Study 1 signature manifest must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature_support_summary(path: Path) -> dict[str, float | int]:
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature manifest validation requires nibabel.") from exc

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj, dtype=float)
    finite = np.isfinite(data)
    nonzero = finite & (np.abs(data) > 0.0)
    positive = finite & (data > 0.0)
    negative = finite & (data < 0.0)
    return {
        "nonzero_voxels": int(np.count_nonzero(nonzero)),
        "positive_voxels": int(np.count_nonzero(positive)),
        "negative_voxels": int(np.count_nonzero(negative)),
        "positive_abs_weight_mass": float(np.sum(np.abs(data[positive]))),
        "negative_abs_weight_mass": float(np.sum(np.abs(data[negative]))),
    }


def _validate_manifest_support(
    *,
    name: str,
    entry: dict[str, Any],
    actual: dict[str, float | int],
) -> None:
    support = entry.get("support")
    if not isinstance(support, dict):
        raise ValueError(f"Study 1 signature manifest entry for {name} must define support.")
    required = (
        "nonzero_voxels",
        "positive_voxels",
        "negative_voxels",
        "positive_abs_weight_mass",
        "negative_abs_weight_mass",
    )
    missing = [field for field in required if field not in support]
    if missing:
        raise ValueError(
            f"Study 1 signature manifest support for {name} is missing fields: {missing}."
        )
    count_fields = ("nonzero_voxels", "positive_voxels", "negative_voxels")
    for field in count_fields:
        if int(support[field]) != int(actual[field]):
            raise ValueError(
                f"Study 1 signature manifest support mismatch for {name}.{field}: "
                f"expected {support[field]!r}, actual {actual[field]!r}."
            )
    mass_fields = ("positive_abs_weight_mass", "negative_abs_weight_mass")
    for field in mass_fields:
        if not np.isclose(float(support[field]), float(actual[field]), rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"Study 1 signature manifest support mismatch for {name}.{field}: "
                f"expected {support[field]!r}, actual {actual[field]!r}."
            )


def _validate_manifest_entry(
    *,
    name: str,
    entry: dict[str, Any],
    configured_path: str,
    configured_space: str,
    image_path: Path,
) -> None:
    required_text_fields = ("source_publication", "source_repository_or_access_record")
    missing_text = [
        field for field in required_text_fields if not str(entry.get(field, "")).strip()
    ]
    if missing_text:
        raise ValueError(
            f"Study 1 signature manifest provenance for {name} is missing fields: {missing_text}."
        )
    if str(entry.get("path", "")).strip() != configured_path:
        raise ValueError(
            f"Study 1 signature manifest path mismatch for {name}: "
            f"expected {configured_path!r}, got {entry.get('path')!r}."
        )
    if str(entry.get("space", "")).strip().lower() != configured_space:
        raise ValueError(
            f"Study 1 signature manifest space mismatch for {name}: "
            f"expected {configured_space!r}, got {entry.get('space')!r}."
        )
    checksum = str(entry.get("sha256", "")).strip().lower()
    if not checksum:
        raise ValueError(f"Study 1 signature manifest entry for {name} must define sha256.")
    actual_checksum = _sha256(image_path)
    if checksum != actual_checksum:
        raise ValueError(
            f"Study 1 signature manifest checksum mismatch for {name}: "
            f"expected {checksum}, actual {actual_checksum}."
        )

    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature manifest validation requires nibabel.") from exc

    image = nib.load(str(image_path))
    expected_shape = tuple(int(value) for value in entry.get("shape", ()))
    if expected_shape != tuple(image.shape):
        raise ValueError(
            f"Study 1 signature manifest shape mismatch for {name}: "
            f"expected {expected_shape!r}, actual {tuple(image.shape)!r}."
        )
    expected_affine = np.asarray(entry.get("affine"), dtype=float)
    if expected_affine.shape != (4, 4) or not np.allclose(expected_affine, image.affine):
        raise ValueError(f"Study 1 signature manifest affine mismatch for {name}.")
    _validate_manifest_support(
        name=name,
        entry=entry,
        actual=_signature_support_summary(image_path),
    )


def _validate_signature_manifest(config: Any, signature_specs: list[dict[str, str]]) -> None:
    manifest_path = _signature_manifest_path(config)
    manifest = _load_signature_manifest(manifest_path)
    entries = manifest.get("signatures")
    if not isinstance(entries, dict):
        raise ValueError("Study 1 signature manifest must define a 'signatures' mapping.")

    signature_root = Path(str(require_config_value(config, "paths.signature_dir"))).expanduser()
    configured_space = (
        str(require_config_value(config, "study1.targets.fmriprep_space")).strip().lower()
    )
    specs_by_name = {str(spec["name"]).strip(): spec for spec in signature_specs}
    for name in PRIMARY_SIGNATURES:
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Study 1 signature manifest is missing entry for {name}.")
        configured_path = str(specs_by_name[name].get("path", "")).strip()
        _validate_manifest_entry(
            name=name,
            entry=entry,
            configured_path=configured_path,
            configured_space=configured_space,
            image_path=signature_root / configured_path,
        )


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
        min_signature_support_fraction=target_cfg.get("min_signature_support_fraction"),
        max_signature_weight_mass_change_fraction=target_cfg.get(
            "max_signature_weight_mass_change_fraction"
        ),
        max_design_condition_number=target_cfg.get("max_design_condition_number"),
        min_target_design_efficiency=target_cfg.get("min_target_design_efficiency"),
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
    if nuisance_regression_enabled(config):
        raw_columns = [column for column in columns if column in RAW_LEVEL2_ARTIFACT_COLUMNS]
        if raw_columns:
            replacements = {column: RAW_LEVEL2_ARTIFACT_COLUMNS[column] for column in raw_columns}
            raise ValueError(
                "Study 1 Level 2 nuisance regression requires HRF-weighted artifact "
                f"covariates, got raw columns: {json.dumps(replacements, sort_keys=True)}."
            )
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


def _required_trial_index(events_df: pd.DataFrame) -> pd.Series:
    if "trial_number" in events_df.columns:
        source = events_df["trial_number"]
    elif "trial_index" in events_df.columns:
        source = events_df["trial_index"]
    else:
        raise ValueError(
            "Study 1 target preparation requires clean EEG events to contain "
            "'trial_number' or 'trial_index' with original trial-order labels."
        )

    trial_index = pd.to_numeric(source, errors="coerce")
    if not trial_index.notna().all():
        raise ValueError(
            "Study 1 target preparation requires finite values in 'trial_number' "
            "or 'trial_index' for every clean EEG event row."
        )
    return trial_index


def _required_task_block(events_df: pd.DataFrame) -> pd.Series:
    if "block" not in events_df.columns:
        raise ValueError(
            "Study 1 target preparation requires an explicit task block column named 'block'. "
            "BIDS run/session labels are acquisition metadata and cannot substitute for task blocks."
        )
    block = pd.to_numeric(events_df["block"], errors="coerce")
    if not block.notna().all():
        raise ValueError("Study 1 task block column 'block' must contain finite values.")
    return block


def _parse_acquisition_run(series: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric

    parsed = pd.Series(
        [parse_run_label_to_int(value) for value in series],
        index=series.index,
        dtype="float64",
    )
    if parsed.notna().all():
        return parsed

    raise ValueError(
        f"Study 1 acquisition run column '{column}' must contain finite run labels "
        "for every clean EEG event row."
    )


def _required_acquisition_run(events_df: pd.DataFrame, task_block: pd.Series) -> pd.Series:
    for column in ACQUISITION_RUN_COLUMNS:
        if column in events_df.columns:
            return _parse_acquisition_run(events_df[column], column=column)
    return task_block


def _events_for_signature_alignment(
    *,
    events_df: pd.DataFrame,
    acquisition_run: pd.Series,
) -> pd.DataFrame:
    aligned_events = events_df.copy()
    aligned_events["block"] = acquisition_run.reset_index(drop=True)
    return aligned_events


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
    trial_index = _required_trial_index(events_df)
    block = _required_task_block(events_df)
    acquisition_run = _required_acquisition_run(events_df, task_block=block)
    signature_events = _events_for_signature_alignment(
        events_df=events_df,
        acquisition_run=acquisition_run,
    )

    nps, _nps_label, nps_extra = load_fmri_signature_target_for_subject(
        subject_raw=subject,
        task=task,
        deriv_root=deriv_root,
        config=_config_for_signature(config, "NPS"),
        events_df=signature_events,
        logger=logger,
        config_path="study1.targets",
    )
    siips1, _siips1_label, siips1_extra = load_fmri_signature_target_for_subject(
        subject_raw=subject,
        task=task,
        deriv_root=deriv_root,
        config=_config_for_signature(config, "SIIPS1"),
        events_df=signature_events,
        logger=logger,
        config_path="study1.targets",
    )

    frame = pd.DataFrame(
        {
            "subject_id": f"sub-{subject}" if not str(subject).startswith("sub-") else str(subject),
            "task": task,
            "block": block,
            "acquisition_run": acquisition_run,
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
        series = extra[column].reset_index(drop=True)
        if column.endswith("scoring_mask_sha256"):
            if not series.notna().any():
                continue
            frame[f"{prefix}_{column}"] = series.astype("string")
            continue
        numeric_values = pd.to_numeric(series, errors="coerce")
        if numeric_values.notna().any():
            frame[f"{prefix}_{column}"] = numeric_values
            continue
        if not series.notna().any():
            continue
        frame[f"{prefix}_{column}"] = series
    return frame


def _validate_signature_scoring_masks(frame: pd.DataFrame) -> None:
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
        hash_column = f"{prefix}_fmri_scoring_mask_sha256"
        if hash_column not in frame.columns:
            raise ValueError(
                f"Study 1 target table is missing required scoring-mask column: {hash_column}."
            )
        hashes = frame[hash_column].astype("string").str.strip()
        if hashes.isna().any() or (hashes == "").any():
            raise ValueError(
                f"Study 1 scoring-mask column '{hash_column}' contains missing values."
            )
        unique_hashes = sorted(set(hashes.astype(str).tolist()))
        if len(unique_hashes) != 1:
            raise ValueError(
                f"Study 1 {prefix} scoring requires an identical scoring-mask extent across "
                f"all retained observations, got {len(unique_hashes)} extents."
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
    _validate_signature_scoring_masks(primary_table)
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
