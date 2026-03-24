"""Study 1 primary target preparation."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterable

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
        raise ValueError(
            "study1.targets.names must be exactly ['NPS', 'SIIPS1'] in that order."
        )
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


def _build_trial_signature_config(config: Any, *, task: str) -> TrialSignatureExtractionConfig:
    target_cfg = _study1_target_config(config)
    return TrialSignatureExtractionConfig(
        input_source=str(require_config_value(config, "study1.targets.input_source")).strip(),
        fmriprep_space=str(require_config_value(config, "study1.targets.fmriprep_space")).strip(),
        require_fmriprep=bool(require_config_value(config, "study1.targets.require_fmriprep")),
        runs=None,
        task=task,
        name=str(require_config_value(config, "study1.targets.contrast_name")).strip(),
        condition_a_column=str(require_config_value(config, "study1.targets.condition_a_column")).strip(),
        condition_a_value=str(require_config_value(config, "study1.targets.condition_a_value")).strip(),
        condition_b_column=str(require_config_value(config, "study1.targets.condition_b_column")).strip(),
        condition_b_value=str(require_config_value(config, "study1.targets.condition_b_value")).strip(),
        hrf_model=str(require_config_value(config, "study1.targets.hrf_model")).strip(),
        drift_model=str(require_config_value(config, "study1.targets.drift_model")).strip(),
        high_pass_hz=float(require_config_value(config, "study1.targets.high_pass_hz")),
        low_pass_hz=target_cfg.get("low_pass_hz"),
        smoothing_fwhm=target_cfg.get("smoothing_fwhm"),
        confounds_strategy=str(require_config_value(config, "study1.targets.confounds_strategy")).strip(),
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
        raise FileNotFoundError(f"Clean events.tsv not found (or empty) for sub-{subject}, task-{task}.")
    events_df = events_df.reset_index(drop=True)

    nps, _nps_label, _nps_extra = load_fmri_signature_target_for_subject(
        subject_raw=subject,
        task=task,
        deriv_root=deriv_root,
        config=_config_for_signature(config, "NPS"),
        events_df=events_df,
        logger=logger,
        config_path="study1.targets",
    )
    siips1, _siips1_label, _siips1_extra = load_fmri_signature_target_for_subject(
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
    if not frame["NPS"].notna().all() or not frame["SIIPS1"].notna().all():
        raise ValueError(
            f"Study 1 primary target table requires finite values for both NPS and SIIPS1 "
            f"for every retained trial in sub-{subject}."
        )
    return frame


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
    parquet_path, tsv_path = _target_output_paths(config)
    write_parquet(primary_table, parquet_path)
    write_tsv(primary_table, tsv_path)
    return parquet_path


def iter_primary_subjects(primary_table: pd.DataFrame) -> Iterable[str]:
    return sorted({str(subject).strip() for subject in primary_table["subject_id"].tolist()})


__all__ = ["PRIMARY_SIGNATURES", "iter_primary_subjects", "prepare_primary_targets"]
