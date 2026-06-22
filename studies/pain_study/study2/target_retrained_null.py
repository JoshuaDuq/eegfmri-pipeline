"""Drive the Study 2 target-retrained null with the frozen Study 1 model.

The null estimates the source-power association expected when the EEG-to-fMRI
relationship is broken (README Section 6). For each draw and outer fold, the
Level 2 nuisance residual is circularly shifted within block, the permuted
target is rebuilt, and the frozen ElasticNet is refit (reusing Study 1's frozen
hyperparameters) to regenerate the held-out prediction-derived score. The score
then flows through the same source-stage association as the observed maps.

This module composes the public Study 1 functions
(:func:`reconstruct_staged_permutation_target_for_fold`,
:func:`model_comparison_cv_predictions`) into the three callables consumed by
:func:`run_target_retrained_source_permutations`. Each outer fold is reconstructed
and predicted independently, matching Study 1's staged-residual permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.machine_learning.orchestration import (
    model_comparison_cv_predictions,
    reconstruct_staged_permutation_target_for_fold,
)
from studies.pain_study.study2.source_maps import compute_cohort_source_association_maps
from studies.pain_study.study2.target_permutations import (
    InvalidPermutationDraw,
    TargetRetrainedSourcePermutationResult,
    run_target_retrained_source_permutations,
)
from studies.pain_study.study2.validation import require_config_string

OuterFold = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Study1ModelContext:
    """Frozen Study 1 model state needed to refit on permuted targets.

    Trials in ``X``/``y``/``meta``/``groups`` share one canonical order, and
    ``fixed_params_by_fold`` holds the frozen hyperparameters for each outer fold.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    outer_folds: tuple[OuterFold, ...]
    fixed_params_by_fold: tuple[Mapping[str, Any], ...]
    pipe: Any
    param_grid: Mapping[str, Any]
    config: Any
    target_residualization_columns: tuple[str, ...]
    blocks: np.ndarray | None
    trial_indices: np.ndarray | None
    scheme: str
    inner_splits: int
    harmonization_mode: str
    covariates: list[str] | None


def build_target_retrained_null_maps(
    *,
    context: Study1ModelContext,
    source_power_by_band: Mapping[str, Mapping[str, np.ndarray]],
    score_frame_template: pd.DataFrame,
    bands: tuple[str, ...],
    expected_subject_ids: tuple[str, ...],
    n_valid_draws: int,
    max_invalid_fraction: float,
    random_state: int,
) -> tuple[dict[str, np.ndarray], TargetRetrainedSourcePermutationResult]:
    """Produce per-band null association maps for the source-family cluster test.

    ``expected_subject_ids`` is the observed source-stage subject set. A draw in which
    that set changes because the permuted score violates source-stage criteria is
    rejected and resampled, matching README Section 6. Returns one
    ``(n_draws, n_subjects, n_vertices)`` array per band plus the accounting result.
    """
    _validate_bands(bands, source_power_by_band)
    if not expected_subject_ids:
        raise ValueError("Study 2 null requires a non-empty observed source-stage cohort.")
    permute_target, fit_predict_score, compute_source_maps = _null_callables(
        context=context,
        source_power_by_band=source_power_by_band,
        score_frame_template=score_frame_template,
        bands=bands,
        expected_subject_ids=expected_subject_ids,
    )
    result = run_target_retrained_source_permutations(
        n_valid_draws=n_valid_draws,
        max_invalid_fraction=max_invalid_fraction,
        random_state=random_state,
        permute_target=permute_target,
        fit_predict_score=fit_predict_score,
        compute_source_maps=compute_source_maps,
    )
    null_by_band = {
        band: result.null_source_maps[:, band_index]
        for band_index, band in enumerate(bands)
    }
    return null_by_band, result


def _null_callables(
    *,
    context: Study1ModelContext,
    source_power_by_band: Mapping[str, Mapping[str, np.ndarray]],
    score_frame_template: pd.DataFrame,
    bands: tuple[str, ...],
    expected_subject_ids: tuple[str, ...],
) -> tuple[
    Callable[[np.random.Generator, int], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    score_column = _combined_score_column(context.config)
    if len(score_frame_template) != len(context.y):
        raise ValueError(
            "Study 2 null score frame and Study 1 target must share trial counts: "
            f"{len(score_frame_template)} vs {len(context.y)}."
        )

    def permute_target(rng: np.random.Generator, _attempt_index: int) -> np.ndarray:
        return stacked_permuted_targets(context, rng)

    def fit_predict_score(stacked_targets: np.ndarray) -> np.ndarray:
        scores = assemble_permuted_scores(context, stacked_targets)
        return standardize_scores_within_subject(scores, context.groups)

    def compute_source_maps(score: np.ndarray) -> np.ndarray:
        frame = score_frame_template.copy()
        frame[score_column] = np.asarray(score, dtype=float)
        band_maps = []
        for band in bands:
            result = compute_cohort_source_association_maps(
                frame,
                source_power_by_band[band],
                band=band,
                config=context.config,
            )
            if result.subject_ids != expected_subject_ids:
                raise InvalidPermutationDraw(
                    "Study 2 null draw changed the observed source-stage cohort for band "
                    f"'{band}': expected {expected_subject_ids}, got {result.subject_ids}."
                )
            band_maps.append(result.fisher_z_maps)
        return np.stack(band_maps, axis=0)

    return permute_target, fit_predict_score, compute_source_maps


def stacked_permuted_targets(
    context: Study1ModelContext,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reconstruct one permuted target per outer fold, stacked row-wise."""
    fold_targets = [
        reconstruct_staged_permutation_target_for_fold(
            y=context.y,
            groups=context.groups,
            meta=context.meta,
            train_idx=train_idx,
            test_idx=test_idx,
            columns=context.target_residualization_columns,
            blocks=context.blocks,
            trial_indices=context.trial_indices,
            rng=rng,
            scheme=context.scheme,
        )
        for train_idx, test_idx in context.outer_folds
    ]
    return np.vstack(fold_targets)


def assemble_permuted_scores(
    context: Study1ModelContext,
    stacked_targets: np.ndarray,
) -> np.ndarray:
    """Refit per fold with frozen hyperparameters and assemble held-out scores."""
    if stacked_targets.shape != (len(context.outer_folds), len(context.y)):
        raise ValueError(
            "Study 2 stacked permuted targets must be (n_folds, n_trials): "
            f"got {stacked_targets.shape}."
        )

    scores = np.full(len(context.y), np.nan, dtype=float)
    for fold_index, (train_idx, test_idx) in enumerate(context.outer_folds):
        _y_true, y_pred, _records = model_comparison_cv_predictions(
            model_name="study2_target_retrained_null",
            pipe=context.pipe,
            param_grid=dict(context.param_grid),
            X=context.X,
            y=stacked_targets[fold_index],
            groups=context.groups,
            meta=context.meta,
            outer_folds=[(train_idx, test_idx)],
            inner_splits=context.inner_splits,
            outer_jobs=1,
            config=context.config,
            harmonization_mode=context.harmonization_mode,
            covariates=context.covariates,
            target_residualization_columns=context.target_residualization_columns,
            collect_records=False,
            fixed_params=dict(context.fixed_params_by_fold[fold_index]),
        )
        scores[np.asarray(test_idx, dtype=int)] = np.asarray(y_pred, dtype=float)[
            np.asarray(test_idx, dtype=int)
        ]

    if not np.all(np.isfinite(scores)):
        raise InvalidPermutationDraw(
            "Study 2 permuted held-out score has unassigned or non-finite trials."
        )
    return scores


def standardize_scores_within_subject(
    scores: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Center and scale the score to unit variance within each held-out subject."""
    score_values = np.asarray(scores, dtype=float)
    group_values = np.asarray(groups, dtype=object)
    standardized = np.empty_like(score_values)
    for subject in np.unique(group_values):
        subject_mask = group_values == subject
        subject_scores = score_values[subject_mask]
        standard_deviation = float(np.std(subject_scores, ddof=0))
        if standard_deviation <= 1.0e-12:
            raise InvalidPermutationDraw(
                f"Study 2 permuted held-out score has zero variance for {subject}."
            )
        standardized[subject_mask] = (
            subject_scores - float(np.mean(subject_scores))
        ) / standard_deviation
    return standardized


def _validate_bands(
    bands: tuple[str, ...],
    source_power_by_band: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    if not bands:
        raise ValueError("Study 2 target-retrained null requires at least one band.")
    missing = [band for band in bands if band not in source_power_by_band]
    if missing:
        raise ValueError(f"Study 2 null is missing source power for bands: {missing}.")


def _combined_score_column(config: Any) -> str:
    return require_config_string(config, "study2.contributions.combined_standardized_column")


__all__ = [
    "Study1ModelContext",
    "assemble_permuted_scores",
    "build_target_retrained_null_maps",
    "stacked_permuted_targets",
    "standardize_scores_within_subject",
]
