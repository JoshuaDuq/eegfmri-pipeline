"""Study 1 feature-based ML benchmark."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

from eeg_pipeline.analysis.machine_learning.orchestration import run_model_comparison_ml
from eeg_pipeline.utils.config.loader import ConfigDict, get_config_value, require_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from studies.pain_study.study1.cohort import (
    primary_targets_parquet_path,
    resolve_primary_target_name,
    resolve_primary_subjects,
    study1_feature_root,
    study1_output_root,
)
from studies.pain_study.study1.feature_spec import (
    PRIMARY_FEATURE_FAMILY,
    resolve_exploratory_feature_families,
)
from studies.pain_study.study1.output_cleanup import remove_appledouble_sidecars
from studies.pain_study.study1.prepare_features import require_prepared_study1_features
from studies.pain_study.study1.targets import (
    PRIMARY_SIGNATURES,
    nuisance_regression_enabled,
    residualization_columns_for_target_table,
)

PRIMARY_BAND_PRESETS: dict[str, list[str]] = {
    "alpha": ["alpha"],
    "beta": ["beta"],
    "alpha_beta": ["alpha", "beta"],
}
PRIMARY_FEATURE_SEGMENTS = ("active",)
PRIMARY_FEATURE_SCOPES = ("ch",)
PRIMARY_FEATURE_STATS = ("logratio",)


def _rng_seed(config: Any) -> int:
    return int(get_config_value(config, "project.random_state", 42))


def _feature_results_root(
    config: Any,
    *,
    partition: str,
    target_name: str,
    feature_spec: str,
) -> Path:
    return study1_output_root(config) / "feature_benchmark" / partition / target_name / feature_spec


def _feature_benchmark_config(
    config: Any,
    *,
    target_name: str,
) -> Any:
    feature_config = ConfigDict(copy.deepcopy(dict(config)))
    feature_config["machine_learning.targets.regression"] = "fmri_signature"
    feature_config["machine_learning.data.require_trial_ml_safe"] = True
    feature_config["feature_engineering.analysis_mode"] = "trial_ml_safe"
    feature_config["machine_learning.fmri_signature.method"] = get_config_value(
        config,
        "study1.targets.method",
        None,
    )
    feature_config["machine_learning.fmri_signature.contrast_name"] = get_config_value(
        config,
        "study1.targets.contrast_name",
        None,
    )
    feature_config["machine_learning.fmri_signature.signature_name"] = target_name
    feature_config["machine_learning.fmri_signature.target_column"] = resolve_primary_target_name(
        config,
        target_name,
    )
    feature_config["machine_learning.fmri_signature.target_table_path"] = str(
        primary_targets_parquet_path(config)
    )
    feature_config["machine_learning.fmri_signature.metric"] = get_config_value(
        config,
        "study1.targets.metric",
        None,
    )
    feature_config["machine_learning.fmri_signature.normalization"] = get_config_value(
        config,
        "study1.targets.normalization",
        "none",
    )
    feature_config["machine_learning.fmri_signature.round_decimals"] = int(
        get_config_value(config, "study1.targets.round_decimals", 3)
    )
    feature_config["machine_learning.preprocessing.subject_standardize_features"] = False
    feature_config["machine_learning.preprocessing.variance_threshold_grid"] = [0.0]
    excluded_channels = get_config_value(
        config,
        "study1.feature_benchmark.excluded_channels",
        ["Fp1", "Fp2"],
    )
    if not isinstance(excluded_channels, (list, tuple)):
        raise ValueError("study1.feature_benchmark.excluded_channels must be a list.")
    feature_config["machine_learning.data.excluded_channels"] = [
        str(channel).strip()
        for channel in excluded_channels
        if str(channel).strip()
    ]
    permutation_scheme = str(
        require_config_value(config, "study1.feature_benchmark.permutation_scheme")
    ).strip()
    feature_config["machine_learning.cv.permutation_scheme"] = permutation_scheme
    if permutation_scheme == "circular_shift_within_run":
        for key in ("min_valid_blocks_per_subject", "min_retained_trials_per_subject"):
            value = require_config_value(
                config,
                f"study1.feature_benchmark.circular_shift.{key}",
            )
            feature_config[f"machine_learning.cv.circular_shift.{key}"] = int(value)
    feature_config["machine_learning.cv.max_invalid_permutation_fraction"] = float(
        require_config_value(
            config,
            "study1.feature_benchmark.max_invalid_permutation_fraction",
        )
    )
    columns = (
        list(
            residualization_columns_for_target_table(
                config,
                primary_targets_parquet_path(config),
            )
        )
        if nuisance_regression_enabled(config)
        else []
    )
    feature_config["machine_learning.target_residualization.enabled"] = bool(columns)
    feature_config["machine_learning.target_residualization.columns"] = columns
    feature_config["machine_learning.target_residualization.strategy"] = (
        "staged_residual_learning"
    )
    return feature_config


def _require_permutation_inference(config: Any) -> int:
    n_perm = int(get_config_value(config, "study1.feature_benchmark.n_perm", 0))
    if n_perm <= 0:
        raise ValueError("Study 1 feature benchmark requires study1.feature_benchmark.n_perm > 0.")
    return n_perm


def _run_primary_presets(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    logger: logging.Logger,
) -> list[Path]:
    deriv_root = resolve_eeg_deriv_root(config)
    feature_root = study1_feature_root(config)
    n_perm = _require_permutation_inference(config)
    inner_splits = int(get_config_value(config, "study1.feature_benchmark.inner_splits", 5))
    outer_jobs = int(get_config_value(config, "study1.feature_benchmark.outer_jobs", 1))
    harmonization = str(
        get_config_value(config, "study1.feature_benchmark.feature_harmonization", "intersection")
    ).strip()

    outputs: list[Path] = []
    for target_name in PRIMARY_SIGNATURES:
        target_config = _feature_benchmark_config(config, target_name=target_name)
        for preset_name, preset_bands in PRIMARY_BAND_PRESETS.items():
            results_root = _feature_results_root(
                config,
                partition="primary",
                target_name=target_name,
                feature_spec=preset_name,
            )
            outputs.append(
                run_model_comparison_ml(
                    subjects=subjects,
                    task=task,
                    deriv_root=deriv_root,
                    config=target_config,
                    n_perm=n_perm,
                    inner_splits=inner_splits,
                    outer_jobs=outer_jobs,
                    rng_seed=_rng_seed(config),
                    results_root=results_root,
                    logger=logger,
                    target="fmri_signature",
                    feature_families=[PRIMARY_FEATURE_FAMILY],
                    feature_input_root=feature_root,
                    feature_bands=list(preset_bands),
                    feature_segments=list(PRIMARY_FEATURE_SEGMENTS),
                    feature_scopes=list(PRIMARY_FEATURE_SCOPES),
                    feature_stats=list(PRIMARY_FEATURE_STATS),
                    feature_harmonization=harmonization,
                    model_names=["elasticnet", "ridge"],
                )
            )
    return outputs


def _run_exploratory_benchmark(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    logger: logging.Logger,
) -> list[Path]:
    deriv_root = resolve_eeg_deriv_root(config)
    feature_root = study1_feature_root(config)
    exploratory_families = resolve_exploratory_feature_families(config)
    if not exploratory_families:
        return []

    n_perm = _require_permutation_inference(config)
    inner_splits = int(get_config_value(config, "study1.feature_benchmark.inner_splits", 5))
    outer_jobs = int(get_config_value(config, "study1.feature_benchmark.outer_jobs", 1))
    harmonization = str(
        get_config_value(config, "study1.feature_benchmark.feature_harmonization", "intersection")
    ).strip()

    outputs: list[Path] = []
    for target_name in PRIMARY_SIGNATURES:
        target_config = _feature_benchmark_config(config, target_name=target_name)
        for family in exploratory_families:
            results_root = _feature_results_root(
                config,
                partition="exploratory",
                target_name=target_name,
                feature_spec=family,
            )
            outputs.append(
                run_model_comparison_ml(
                    subjects=subjects,
                    task=task,
                    deriv_root=deriv_root,
                    config=target_config,
                    n_perm=n_perm,
                    inner_splits=inner_splits,
                    outer_jobs=outer_jobs,
                    rng_seed=_rng_seed(config),
                    results_root=results_root,
                    logger=logger,
                    target="fmri_signature",
                    feature_families=[family],
                    feature_input_root=feature_root,
                    feature_bands=None,
                    feature_harmonization=harmonization,
                )
            )
    return outputs


def run_feature_benchmark(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    logger: logging.Logger | None = None,
) -> list[Path]:
    if logger is None:
        logger = logging.getLogger(__name__)

    target_table_path = primary_targets_parquet_path(config)
    if not target_table_path.exists():
        raise FileNotFoundError(
            f"Study 1 primary target table not found: {target_table_path}. "
            "Run 'signature-prediction prepare-targets' first."
        )
    _require_permutation_inference(config)

    resolved_subjects = resolve_primary_subjects(
        requested_subjects=subjects,
        task=task,
        config=config,
    )
    require_prepared_study1_features(
        subjects=resolved_subjects,
        config=config,
    )

    benchmark_root = study1_output_root(config) / "feature_benchmark"
    try:
        outputs = _run_primary_presets(
            subjects=resolved_subjects,
            task=task,
            config=config,
            logger=logger,
        )
        outputs.extend(
            _run_exploratory_benchmark(
                subjects=resolved_subjects,
                task=task,
                config=config,
                logger=logger,
            )
        )
    finally:
        removed_sidecars = remove_appledouble_sidecars(benchmark_root)
        if removed_sidecars:
            logger.info(
                "Removed %d AppleDouble sidecars from Study 1 feature benchmark outputs",
                removed_sidecars,
            )
    return outputs


__all__ = ["PRIMARY_BAND_PRESETS", "run_feature_benchmark"]
