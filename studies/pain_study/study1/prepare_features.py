"""Study 1-owned EEG feature preparation."""

from __future__ import annotations

import json
import logging
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from eeg_pipeline.pipelines.features import FeaturePipeline
from eeg_pipeline.utils.config.loader import ConfigDict, get_config_value
from studies.pain_study.study1.cohort import (
    resolve_primary_subjects,
    study1_feature_metadata_path,
    study1_feature_root,
    study1_feature_subject_root,
    study1_feature_table_path,
)
from studies.pain_study.study1.feature_spec import resolve_study1_feature_families
from studies.pain_study.study1.output_cleanup import (
    prune_windowed_feature_artifacts,
    remove_appledouble_sidecars,
)


WINDOWED_FEATURE_FAMILIES = {"erds", "bursts"}


def _clear_subject_feature_outputs(
    *,
    subjects: list[str],
    config: Any,
) -> None:
    for subject_id in subjects:
        subject_root = study1_feature_subject_root(config, subject_id)
        if subject_root.exists():
            try:
                shutil.rmtree(subject_root)
            except FileNotFoundError:
                continue


def _load_feature_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Study 1 feature metadata JSON: {metadata_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Study 1 feature metadata must be a JSON object: {metadata_path}"
        )
    return payload


def _require_boolean_metadata(
    metadata: dict[str, Any],
    *,
    metadata_path: Path,
    key: str,
    expected: bool,
    setting_name: str,
) -> None:
    if key not in metadata:
        raise ValueError(
            f"Study 1 feature metadata must record {setting_name}: {metadata_path}"
        )
    actual = bool(metadata[key])
    if actual is not expected:
        raise ValueError(
            f"Study 1 prepared features require {setting_name}={expected!s}. "
            f"Metadata {metadata_path} recorded {actual!s}."
        )


def _validate_prepared_feature_metadata(
    *,
    family: str,
    metadata_path: Path,
) -> None:
    metadata = _load_feature_metadata(metadata_path)
    analysis_mode = str(metadata.get("analysis_mode", "")).strip().lower()
    if analysis_mode != "trial_ml_safe":
        raise ValueError(
            "Study 1 prepared feature metadata must declare analysis_mode='trial_ml_safe'. "
            f"Invalid metadata file: {metadata_path}"
        )

    _require_boolean_metadata(
        metadata,
        metadata_path=metadata_path,
        key="bands_use_iaf",
        expected=False,
        setting_name="feature_engineering.bands.use_iaf",
    )

    if family == "power":
        _require_boolean_metadata(
            metadata,
            metadata_path=metadata_path,
            key="power_subtract_evoked",
            expected=False,
            setting_name="feature_engineering.power.subtract_evoked",
        )
        return

    _require_boolean_metadata(
        metadata,
        metadata_path=metadata_path,
        key="precomputed_subtract_evoked",
        expected=False,
        setting_name="feature_engineering.precomputed.subtract_evoked",
    )

    if family == "aperiodic":
        _require_boolean_metadata(
            metadata,
            metadata_path=metadata_path,
            key="aperiodic_subtract_evoked",
            expected=False,
            setting_name="feature_engineering.aperiodic.subtract_evoked",
        )

    if family == "bursts":
        if "bursts_threshold_reference" not in metadata:
            raise ValueError(
                "Study 1 feature metadata must record feature_engineering.bursts.threshold_reference: "
                f"{metadata_path}"
            )
        threshold_reference = str(metadata["bursts_threshold_reference"]).strip().lower()
        if threshold_reference != "trial":
            raise ValueError(
                "Study 1 prepared burst features require "
                "feature_engineering.bursts.threshold_reference='trial'. "
                f"Metadata {metadata_path} recorded '{threshold_reference}'."
            )


def require_prepared_study1_features(
    *,
    subjects: list[str],
    config: Any,
    feature_families: list[str] | None = None,
) -> None:
    families = feature_families or resolve_study1_feature_families(config)
    missing_paths: list[Path] = []
    for subject_id in subjects:
        for family in families:
            feature_path = study1_feature_table_path(config, subject_id, family)
            metadata_path = study1_feature_metadata_path(config, subject_id, family)
            if not feature_path.exists():
                missing_paths.append(feature_path)
                continue
            if not metadata_path.exists():
                missing_paths.append(metadata_path)
                continue
            _validate_prepared_feature_metadata(
                family=family,
                metadata_path=metadata_path,
            )

    if missing_paths:
        raise FileNotFoundError(
            "Study 1 prepared features are missing or incomplete. "
            "Run 'signature-prediction prepare-features' first. "
            "Missing paths: "
            f"{missing_paths}"
        )


def _study1_feature_config(config: Any) -> Any:
    feature_config = ConfigDict(deepcopy(dict(config)))
    feature_config["feature_engineering.analysis_mode"] = "trial_ml_safe"
    feature_config["feature_engineering.power.subtract_evoked"] = False
    feature_config["feature_engineering.precomputed.subtract_evoked"] = False
    feature_config["feature_engineering.aperiodic.subtract_evoked"] = False
    feature_config["feature_engineering.bands.use_iaf"] = False
    feature_config["feature_engineering.bursts.threshold_reference"] = "trial"
    return feature_config


def _require_time_window_pair(
    *,
    config: Any,
    config_path: str,
) -> tuple[float, float]:
    raw_value = get_config_value(config, config_path, None)
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 2:
        raise ValueError(f"{config_path} must be a list or tuple of length 2.")

    try:
        start = float(raw_value[0])
        end = float(raw_value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{config_path} must contain finite numeric bounds. Got {raw_value!r}."
        ) from exc

    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"{config_path} must contain finite numeric bounds. Got {raw_value!r}.")
    if start >= end:
        raise ValueError(
            f"{config_path} must satisfy start < end. Got [{start}, {end}]."
        )
    return start, end


def _study1_windowed_time_ranges(config: Any) -> list[dict[str, float | str]]:
    baseline_start, baseline_end = _require_time_window_pair(
        config=config,
        config_path="time_frequency_analysis.baseline_window",
    )
    active_start, active_end = _require_time_window_pair(
        config=config,
        config_path="time_frequency_analysis.active_window",
    )
    return [
        {"name": "baseline", "tmin": baseline_start, "tmax": baseline_end},
        {"name": "active", "tmin": active_start, "tmax": active_end},
    ]


def _standard_feature_families(feature_families: list[str]) -> list[str]:
    return [family for family in feature_families if family not in WINDOWED_FEATURE_FAMILIES]


def _windowed_feature_families(feature_families: list[str]) -> list[str]:
    return [family for family in feature_families if family in WINDOWED_FEATURE_FAMILIES]


def _run_feature_batch(
    *,
    pipeline: FeaturePipeline,
    subjects: list[str],
    task: str,
    feature_root: Path,
    feature_families: list[str],
    time_ranges: list[dict[str, float | str]] | None = None,
) -> None:
    if not feature_families:
        return

    run_kwargs: dict[str, Any] = {
        "subjects": subjects,
        "task": task,
        "fail_fast": True,
        "analysis_mode": "trial_ml_safe",
        "feature_categories": list(feature_families),
        "feature_output_root": feature_root,
        "save_canonical_trial_table": False,
    }
    if time_ranges is not None:
        run_kwargs["time_ranges"] = list(time_ranges)
    pipeline.run_batch(**run_kwargs)


def prepare_study1_features(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    logger: logging.Logger | None = None,
) -> Path:
    if logger is None:
        logger = logging.getLogger(__name__)

    resolved_subjects = resolve_primary_subjects(
        requested_subjects=subjects,
        task=task,
        config=config,
    )
    feature_root = study1_feature_root(config)
    feature_root.mkdir(parents=True, exist_ok=True)

    _clear_subject_feature_outputs(subjects=resolved_subjects, config=config)

    feature_families = resolve_study1_feature_families(config)
    standard_families = _standard_feature_families(feature_families)
    windowed_families = _windowed_feature_families(feature_families)
    windowed_time_ranges = (
        _study1_windowed_time_ranges(config) if windowed_families else None
    )
    pipeline = FeaturePipeline(config=_study1_feature_config(config))

    _run_feature_batch(
        pipeline=pipeline,
        subjects=resolved_subjects,
        task=task,
        feature_root=feature_root,
        feature_families=standard_families,
    )
    _run_feature_batch(
        pipeline=pipeline,
        subjects=resolved_subjects,
        task=task,
        feature_root=feature_root,
        feature_families=windowed_families,
        time_ranges=windowed_time_ranges,
    )
    removed_windowed = prune_windowed_feature_artifacts(
        feature_root=feature_root,
        subjects=resolved_subjects,
        feature_families=windowed_families,
    )
    removed_sidecars = remove_appledouble_sidecars(feature_root)

    require_prepared_study1_features(
        subjects=resolved_subjects,
        config=config,
        feature_families=feature_families,
    )
    if removed_windowed:
        logger.info("Pruned %d redundant Study 1 windowed feature artifacts", removed_windowed)
    if removed_sidecars:
        logger.info("Removed %d AppleDouble sidecars from Study 1 feature outputs", removed_sidecars)
    logger.info(
        "Prepared Study 1 trial_ml_safe features for %d subjects at %s",
        len(resolved_subjects),
        feature_root,
    )
    return feature_root


__all__ = ["prepare_study1_features", "require_prepared_study1_features"]
