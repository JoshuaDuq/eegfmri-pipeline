"""Foldwise sensor-pattern artifacts for the Study 2 NPS model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from eeg_pipeline.analysis.machine_learning.config import get_ml_config
from eeg_pipeline.analysis.machine_learning.orchestration import (
    _apply_fold_feature_harmonization_foldwise,
)
from eeg_pipeline.analysis.machine_learning.preprocessing import (
    transform_feature_names_through_steps,
)
from eeg_pipeline.analysis.machine_learning.target_residualization import (
    _design_matrix,
    fit_nuisance_model_for_fold,
)
from studies.pain_study.study2.haufe import compute_haufe_pattern
from studies.pain_study.study2.target_retrained_null import Study1ModelContext

REQUIRED_METRIC_COLUMNS = {
    "model",
    "fold",
    "test_subject",
    "r2",
    "r2_nuisance",
    "delta_r2",
}


@dataclass(frozen=True)
class SensorPatternSummary:
    """Fold-level patterns, median sensor maps, and descriptive stability."""

    fold_patterns: pd.DataFrame
    aggregate_patterns: pd.DataFrame
    stability: pd.DataFrame
    target: str
    bands: tuple[str, ...]
    channels: tuple[str, ...]
    n_subjects: int
    article_ready: bool


@dataclass(frozen=True)
class FrozenStudy1FoldFit:
    transformed_training: np.ndarray
    transformed_test: np.ndarray
    transformed_feature_names: tuple[str, ...]
    coefficients: np.ndarray
    nuisance_prediction: np.ndarray
    residual_prediction: np.ndarray
    full_prediction: np.ndarray
    evaluation_target: np.ndarray
    train_target_mean: float
    parameters: dict[str, object]


def compute_sensor_pattern_summary(
    *,
    context: Study1ModelContext,
    fold_metrics: pd.DataFrame,
    target: str,
    bands: tuple[str, ...],
    refit_tolerance: float,
    minimum_article_subjects: int,
) -> SensorPatternSummary:
    """Refit frozen LOSO models and compute training-only Haufe patterns."""

    if target != "NPS":
        raise ValueError("Study 2 sensor patterns require the predesignated NPS target.")
    _require_columns(fold_metrics, REQUIRED_METRIC_COLUMNS, "Study 1 fold metrics")
    metrics = fold_metrics.loc[fold_metrics["model"].eq("elasticnet")].copy()
    if len(metrics) != len(context.outer_folds):
        raise ValueError("Study 1 metrics and reconstructed LOSO folds have different counts.")
    if metrics.duplicated("fold").any() or metrics.duplicated("test_subject").any():
        raise ValueError("Study 1 metrics must contain one row per fold and held-out subject.")
    if len(context.feature_names) != context.X.shape[1]:
        raise ValueError("Study 1 context feature names do not match the feature matrix.")

    fold_frames = []
    for fold, (train_indices, test_indices) in enumerate(context.outer_folds):
        metric = metrics.loc[metrics["fold"].eq(fold)]
        if len(metric) != 1:
            raise ValueError(f"Study 1 metrics do not uniquely identify fold {fold}.")
        test_subject = str(context.groups[test_indices[0]])
        if str(metric.iloc[0]["test_subject"]) != test_subject:
            raise ValueError(f"Held-out subject disagreement in fold {fold}.")
        fold_frames.append(
            _compute_fold_pattern(
                context=context,
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
                saved_metric=metric.iloc[0],
                target=target,
                bands=bands,
                tolerance=refit_tolerance,
            )
        )

    fold_patterns = pd.concat(fold_frames, ignore_index=True)
    aggregate, stability = aggregate_fold_patterns(fold_patterns, bands)
    channels = tuple(sorted(aggregate["channel"].astype(str).unique()))
    n_subjects = int(len(np.unique(context.groups)))
    return SensorPatternSummary(
        fold_patterns=fold_patterns,
        aggregate_patterns=aggregate,
        stability=stability,
        target=target,
        bands=bands,
        channels=channels,
        n_subjects=n_subjects,
        article_ready=n_subjects >= minimum_article_subjects,
    )


def parse_channel_band(feature_name: str, bands: Sequence[str]) -> tuple[str, str]:
    """Parse an exact Study 1 active channel-power feature."""

    matches: list[tuple[str, str]] = []
    for band in bands:
        prefix = f"power_active_{band}_ch_"
        suffix = "_logratio"
        if feature_name.startswith(prefix) and feature_name.endswith(suffix):
            channel = feature_name[len(prefix) : -len(suffix)]
            if channel and "_" not in channel:
                matches.append((channel, str(band)))
    if len(matches) != 1:
        raise ValueError(
            "Feature is not an exact Study 1 primary channel-power feature: " f"{feature_name!r}."
        )
    return matches[0]


def normalize_band_patterns(
    patterns: pd.DataFrame,
    bands: Sequence[str],
) -> pd.DataFrame:
    """Add unit-norm values independently within each spectral band."""

    _require_columns(patterns, {"band", "channel", "haufe_pattern"}, "Fold pattern")
    if set(patterns["band"].astype(str)) != set(bands):
        raise ValueError("Fold pattern bands differ from the configured bands.")
    output = patterns.copy()
    normalized = np.full(len(output), np.nan, dtype=float)
    for band in bands:
        positions = np.flatnonzero(output["band"].astype(str).eq(str(band)).to_numpy())
        values = output.iloc[positions]["haufe_pattern"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Haufe pattern for band {band!r} contains non-finite values.")
        norm = float(np.linalg.norm(values))
        if norm <= np.finfo(float).eps:
            raise ValueError(f"Haufe pattern for band {band!r} has zero norm.")
        normalized[positions] = values / norm
    output["normalized_pattern"] = normalized
    return output


def aggregate_fold_patterns(
    fold_patterns: pd.DataFrame,
    bands: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return channel medians and all pairwise fold-map correlations."""

    required = {
        "target",
        "fold",
        "test_subject",
        "band",
        "channel",
        "normalized_pattern",
    }
    _require_columns(fold_patterns, required, "Fold pattern")
    if set(fold_patterns["target"].astype(str)) != {"NPS"}:
        raise ValueError("Study 2 sensor artifacts must contain NPS only.")
    if fold_patterns.duplicated(["fold", "band", "channel"]).any():
        raise ValueError("Fold patterns contain duplicate fold/band/channel rows.")

    aggregate_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for band, cell in fold_patterns.groupby("band", sort=False):
        if str(band) not in bands:
            raise ValueError(f"Unexpected sensor-pattern band {band!r}.")
        channel_sets = [
            tuple(sorted(group["channel"].astype(str)))
            for _, group in cell.groupby("fold", sort=True)
        ]
        if len(channel_sets) < 2 or len(set(channel_sets)) != 1:
            raise ValueError("All folds must have identical channel sets for each band.")
        channels = channel_sets[0]
        matrix = (
            cell.pivot(index="fold", columns="channel", values="normalized_pattern")
            .reindex(columns=channels)
            .sort_index()
        )
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Fold patterns must be finite before aggregation.")
        for channel, median in zip(channels, np.median(values, axis=0), strict=True):
            aggregate_rows.append(
                {
                    "target": "NPS",
                    "band": str(band),
                    "channel": channel,
                    "median_normalized_pattern": float(median),
                    "n_folds": int(len(matrix)),
                }
            )
        fold_ids = matrix.index.to_numpy(dtype=int)
        comparison_index = 0
        for left in range(len(matrix) - 1):
            for right in range(left + 1, len(matrix)):
                correlation = float(pearsonr(values[left], values[right]).statistic)
                if not np.isfinite(correlation):
                    raise ValueError("Fold-map spatial correlation is undefined.")
                stability_rows.append(
                    {
                        "target": "NPS",
                        "band": str(band),
                        "comparison_index": comparison_index,
                        "fold_a": int(fold_ids[left]),
                        "fold_b": int(fold_ids[right]),
                        "spatial_correlation": correlation,
                    }
                )
                comparison_index += 1
    aggregate = pd.DataFrame(aggregate_rows)
    stability = pd.DataFrame(stability_rows)
    if set(aggregate["band"].astype(str)) != set(bands):
        raise ValueError("Fold patterns do not contain every configured band.")
    return aggregate, stability


def _compute_fold_pattern(
    *,
    context: Study1ModelContext,
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    saved_metric: pd.Series,
    target: str,
    bands: tuple[str, ...],
    tolerance: float,
) -> pd.DataFrame:
    fit = fit_frozen_study1_fold(context, fold)
    prediction = fit.full_prediction
    evaluation_target = fit.evaluation_target
    total_sum = float(np.sum((evaluation_target - fit.train_target_mean) ** 2))
    if total_sum <= 1.0e-12:
        raise ValueError(f"Fold {fold} has undefined held-out R².")
    model_r2 = 1.0 - float(np.sum((evaluation_target - prediction) ** 2)) / total_sum
    nuisance_r2 = (
        1.0
        - float(np.sum((evaluation_target - fit.nuisance_prediction) ** 2)) / total_sum
    )
    _require_metric_agreement(saved_metric, model_r2, nuisance_r2, tolerance)

    pattern = compute_haufe_pattern(
        fit.transformed_training,
        fit.coefficients,
    ).pattern
    parsed = [parse_channel_band(name, bands) for name in fit.transformed_feature_names]
    frame = pd.DataFrame(
        {
            "target": target,
            "fold": fold,
            "test_subject": str(context.groups[test_indices[0]]),
            "band": [band for _, band in parsed],
            "channel": [channel for channel, _ in parsed],
            "haufe_pattern": pattern,
            "n_train_trials": len(train_indices),
            "n_test_trials": len(test_indices),
            "refit_r2": model_r2,
            "saved_r2": float(saved_metric["r2"]),
            "refit_delta_r2": model_r2 - nuisance_r2,
            "saved_delta_r2": float(saved_metric["delta_r2"]),
            "best_params": str(fit.parameters),
        }
    )
    return normalize_band_patterns(frame, bands)


def fit_frozen_study1_fold(
    context: Study1ModelContext,
    fold: int,
) -> FrozenStudy1FoldFit:
    """Refit one frozen Study 1 outer fold and expose its transformed feature space."""
    train_indices, test_indices = context.outer_folds[fold]
    nuisance = fit_nuisance_model_for_fold(
        y=context.y,
        meta=context.meta,
        train_idx=train_indices,
        test_idx=test_indices,
        columns=context.target_residualization_columns,
    )
    ml_config = get_ml_config(context.config)
    target_transformer = PowerTransformer(
        method=ml_config.get("power_transformer_method", "yeo-johnson"),
        standardize=ml_config.get("power_transformer_standardize", True),
    )
    transformed_target = target_transformer.fit_transform(
        nuisance.train_residual.reshape(-1, 1)
    ).ravel()
    design_train = _design_matrix(
        context.meta.iloc[train_indices],
        context.target_residualization_columns,
        check_rank=True,
    )
    design_test = _design_matrix(
        context.meta.iloc[test_indices],
        context.target_residualization_columns,
        check_rank=False,
    )
    feature_coefficients, *_ = np.linalg.lstsq(
        design_train,
        context.X[train_indices],
        rcond=None,
    )
    training_matrix = context.X[train_indices] - design_train @ feature_coefficients
    test_matrix = context.X[test_indices] - design_test @ feature_coefficients
    training_matrix, test_matrix, keep = _apply_fold_feature_harmonization_foldwise(
        training_matrix,
        test_matrix,
        context.groups[train_indices],
        context.harmonization_mode,
    )
    retained_names = [
        name for name, retained in zip(context.feature_names, keep, strict=True) if retained
    ]

    if not isinstance(context.pipe, TransformedTargetRegressor) or not isinstance(
        context.pipe.regressor, Pipeline
    ):
        raise TypeError("Frozen Study 1 ElasticNet is not the expected wrapped pipeline.")
    estimator = clone(context.pipe.regressor)
    parameters = dict(context.fixed_params_by_fold[fold])
    estimator.set_params(**parameters)
    estimator.fit(training_matrix, transformed_target)

    transformed_prediction = estimator.predict(test_matrix)
    residual_prediction = target_transformer.inverse_transform(
        transformed_prediction.reshape(-1, 1)
    ).ravel()
    transformed_training = estimator[:-1].transform(training_matrix)
    transformed_test = estimator[:-1].transform(test_matrix)
    transformed_names = transform_feature_names_through_steps(estimator.steps[:-1], retained_names)
    if (
        transformed_training.shape[1] != len(transformed_names)
        or transformed_test.shape[1] != len(transformed_names)
    ):
        raise ValueError("Fitted preprocessing cannot be mapped one-to-one to EEG features.")
    weights = np.asarray(estimator.named_steps["regressor"].coef_, dtype=float)
    if weights.shape != (len(transformed_names),):
        raise ValueError("Frozen Study 1 coefficients do not match transformed features.")
    return FrozenStudy1FoldFit(
        transformed_training=np.asarray(transformed_training, dtype=float),
        transformed_test=np.asarray(transformed_test, dtype=float),
        transformed_feature_names=tuple(transformed_names),
        coefficients=weights,
        nuisance_prediction=np.asarray(nuisance.test_prediction, dtype=float),
        residual_prediction=np.asarray(residual_prediction, dtype=float),
        full_prediction=np.asarray(nuisance.test_prediction + residual_prediction, dtype=float),
        evaluation_target=np.asarray(context.y[test_indices], dtype=float),
        train_target_mean=float(np.mean(context.y[train_indices])),
        parameters=parameters,
    )


def _require_metric_agreement(
    saved: pd.Series,
    model_r2: float,
    nuisance_r2: float,
    tolerance: float,
) -> None:
    differences = {
        "r2": abs(model_r2 - float(saved["r2"])),
        "r2_nuisance": abs(nuisance_r2 - float(saved["r2_nuisance"])),
        "delta_r2": abs((model_r2 - nuisance_r2) - float(saved["delta_r2"])),
    }
    if max(differences.values()) > tolerance:
        raise ValueError(
            "Refitted fold metrics disagree with the primary result: "
            f"differences={differences}, tolerance={tolerance}."
        )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


__all__ = [
    "SensorPatternSummary",
    "aggregate_fold_patterns",
    "compute_sensor_pattern_summary",
    "normalize_band_patterns",
    "parse_channel_band",
]
