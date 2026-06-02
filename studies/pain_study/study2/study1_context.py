"""Load the frozen Study 1 model context that the Study 2 null refits.

Study 2's target-retrained null (README Section 6) reuses the exact Study 1
primary-confirmatory model: the same feature matrix, leave-one-subject-out
folds, ElasticNet pipeline, and per-fold frozen hyperparameters selected by
Study 1's nested cross-validation. This module rebuilds that context from the
Study 1 feature pipeline and the persisted ``model_comparison.tsv`` so the null
never reselects hyperparameters, matching the observed analysis fold-for-fold.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from eeg_pipeline.analysis.machine_learning.orchestration import (
    filter_circular_shift_permutation_rows,
)
from eeg_pipeline.analysis.machine_learning.pipelines import (
    build_elasticnet_param_grid,
    create_elasticnet_pipeline,
)
from eeg_pipeline.analysis.machine_learning.target_residualization import (
    configured_target_residualization_columns,
)
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from eeg_pipeline.utils.data.machine_learning import load_active_matrix
from studies.pain_study.study1.cohort import study1_feature_root
from studies.pain_study.study1.feature_benchmark import (
    PRIMARY_BAND_PRESETS,
    PRIMARY_FEATURE_SCOPES,
    PRIMARY_FEATURE_SEGMENTS,
    PRIMARY_FEATURE_STATS,
    feature_benchmark_config,
    feature_results_root,
)
from studies.pain_study.study1.feature_spec import PRIMARY_FEATURE_FAMILY
from studies.pain_study.study1.reporting import (
    PRIMARY_GATE_FEATURE_SPEC,
    PRIMARY_GATE_TARGET,
)
from studies.pain_study.study2.target_retrained_null import OuterFold, Study1ModelContext

ELASTICNET_MODEL_NAME = "elasticnet"


def load_study1_model_context(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    target_name: str = PRIMARY_GATE_TARGET,
    feature_spec: str = PRIMARY_GATE_FEATURE_SPEC,
    logger: logging.Logger | None = None,
) -> Study1ModelContext:
    """Rebuild Study 1's primary-confirmatory model context for the Study 2 null.

    ``subjects``/``task`` select the Study 1 cohort; ``target_name``/``feature_spec``
    select the confirmatory cell (default NPS, alpha+beta+gamma). The returned
    context carries the same trial matrix, folds, pipeline, and per-fold frozen
    hyperparameters that produced ``model_comparison.tsv`` for that cell.
    """
    logger = logger or logging.getLogger(__name__)
    feature_bands = _resolve_feature_bands(feature_spec)
    feature_config = feature_benchmark_config(config, target_name=target_name)

    deriv_root = resolve_eeg_deriv_root(config)
    feature_root = study1_feature_root(config)
    harmonization = str(
        get_config_value(
            config, "study1.feature_benchmark.feature_harmonization", "intersection"
        )
    ).strip()
    inner_splits = int(get_config_value(config, "study1.feature_benchmark.inner_splits", 5))
    rng_seed = int(get_config_value(config, "project.random_state", 42))

    X, y, groups, _feature_names, meta = load_active_matrix(
        subjects,
        task,
        deriv_root,
        feature_config,
        logger,
        feature_families=[PRIMARY_FEATURE_FAMILY],
        feature_input_root=feature_root,
        feature_harmonization=harmonization,
        target="fmri_signature",
        target_kind="continuous",
        feature_bands=feature_bands,
        feature_segments=list(PRIMARY_FEATURE_SEGMENTS),
        feature_scopes=list(PRIMARY_FEATURE_SCOPES),
        feature_stats=list(PRIMARY_FEATURE_STATS),
    )
    target_residualization_columns = configured_target_residualization_columns(feature_config)
    X, y, groups, meta = filter_circular_shift_permutation_rows(
        X=X,
        y=y,
        groups=groups,
        meta=meta,
        config=feature_config,
        logger=logger,
    )

    outer_folds = tuple(LeaveOneGroupOut().split(X, y, groups))
    fixed_params_by_fold = _frozen_params_by_fold(
        config=config,
        target_name=target_name,
        feature_spec=feature_spec,
        outer_folds=outer_folds,
        groups=groups,
    )
    blocks, trial_indices = _block_and_trial_indices(meta)
    scheme = str(
        get_config_value(feature_config, "machine_learning.cv.permutation_scheme", "within_subject")
    ).strip().lower()

    return Study1ModelContext(
        X=X,
        y=y,
        groups=groups,
        meta=meta,
        outer_folds=outer_folds,
        fixed_params_by_fold=fixed_params_by_fold,
        pipe=create_elasticnet_pipeline(seed=rng_seed, config=feature_config),
        param_grid=build_elasticnet_param_grid(feature_config),
        config=feature_config,
        target_residualization_columns=target_residualization_columns,
        blocks=blocks,
        trial_indices=trial_indices,
        scheme=scheme,
        inner_splits=inner_splits,
        harmonization_mode=harmonization,
        covariates=None,
    )


def _resolve_feature_bands(feature_spec: str) -> list[str]:
    if feature_spec not in PRIMARY_BAND_PRESETS:
        raise ValueError(
            f"Unknown Study 1 feature spec '{feature_spec}'. "
            f"Expected one of: {sorted(PRIMARY_BAND_PRESETS)}."
        )
    return list(PRIMARY_BAND_PRESETS[feature_spec])


def study1_model_comparison_path(
    config: Any,
    *,
    target_name: str = PRIMARY_GATE_TARGET,
    feature_spec: str = PRIMARY_GATE_FEATURE_SPEC,
) -> Path:
    """Locate the persisted Study 1 model-comparison metrics for a confirmatory cell."""
    return (
        feature_results_root(
            config,
            partition="primary",
            target_name=target_name,
            feature_spec=feature_spec,
        )
        / "model_comparison"
        / "model_comparison.tsv"
    )


def _frozen_params_by_fold(
    *,
    config: Any,
    target_name: str,
    feature_spec: str,
    outer_folds: tuple[OuterFold, ...],
    groups: np.ndarray,
) -> tuple[Mapping[str, Any], ...]:
    """Map each outer fold to the ElasticNet hyperparameters Study 1 selected."""
    comparison_path = study1_model_comparison_path(
        config, target_name=target_name, feature_spec=feature_spec
    )
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Study 2 null requires Study 1 model comparison metrics at {comparison_path}."
        )

    comparison = pd.read_csv(comparison_path, sep="\t")
    elasticnet_rows = comparison.loc[comparison["model"] == ELASTICNET_MODEL_NAME]
    if elasticnet_rows.empty:
        raise ValueError(
            f"Study 1 model comparison at {comparison_path} has no '{ELASTICNET_MODEL_NAME}' folds."
        )

    params_by_subject = {
        str(row["test_subject"]): _parse_best_params(row["best_params"])
        for _index, row in elasticnet_rows.iterrows()
    }
    fold_params: list[Mapping[str, Any]] = []
    for _train_idx, test_idx in outer_folds:
        test_subject = str(groups[test_idx[0]])
        if test_subject not in params_by_subject:
            raise ValueError(
                f"Study 1 model comparison lacks frozen hyperparameters for held-out "
                f"subject '{test_subject}'."
            )
        fold_params.append(params_by_subject[test_subject])
    return tuple(fold_params)


def _parse_best_params(raw: Any) -> Mapping[str, Any]:
    params = ast.literal_eval(str(raw))
    if not isinstance(params, dict):
        raise ValueError(f"Study 1 best_params is not a mapping: {raw!r}.")
    return params


def _block_and_trial_indices(meta: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    blocks = (
        pd.to_numeric(meta["block"], errors="coerce").to_numpy(dtype=float)
        if "block" in meta.columns
        else None
    )
    trial_indices = None
    for trial_column in ("trial_index", "trial_number"):
        if trial_column in meta.columns:
            trial_indices = pd.to_numeric(meta[trial_column], errors="coerce").to_numpy(dtype=float)
            break
    return blocks, trial_indices


__all__ = ["load_study1_model_context", "study1_model_comparison_path"]
