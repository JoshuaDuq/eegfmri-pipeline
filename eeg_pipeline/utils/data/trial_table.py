"""
Subject-Level Trial Table (Canonical)
====================================

Builds a single per-trial table that merges:
- aligned events metadata
- targets (rating)
- temperature and pain condition labels
- optional covariates and derived columns
- optional feature columns (prefixed by feature type)

This table is intended to be the single source of truth for subject-level
behavior/statistics and plotting, to avoid silent misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eeg_pipeline.context.behavior import BehaviorContext


@dataclass
class TrialTableBuildResult:
    df: pd.DataFrame
    metadata: Dict[str, Any]


def _safe_numeric(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _get_pain_column_from_events(config: Any, events: pd.DataFrame) -> Optional[str]:
    """Extract pain column name from config and events, returning None on failure."""
    try:
        from eeg_pipeline.utils.data.columns import get_pain_column_from_config

        return get_pain_column_from_config(config, events)
    except (AttributeError, KeyError, ValueError):
        return None


def _pick_event_columns(events: pd.DataFrame, candidates: List[str]) -> Dict[str, pd.Series]:
    selected_columns: Dict[str, pd.Series] = {}
    for candidate in candidates:
        if candidate in events.columns and candidate not in selected_columns:
            selected_columns[candidate] = events[candidate]
    return selected_columns


def build_subject_trial_table(
    ctx: Any,  # BehaviorContext
    *,
    include_features: bool = True,
    include_covariates: bool = True,
    include_events: bool = True,
    extra_event_columns: Optional[List[str]] = None,
) -> TrialTableBuildResult:
    """
    Build the canonical subject-level trial table.

    Notes
    -----
    - Assumes ctx.load_data() already ran and alignment is validated.
    - Uses ctx.epochs.selection as the original event index when available.
    """
    if ctx.aligned_events is None:
        raise ValueError("ctx.aligned_events is required")
    if ctx.targets is None:
        raise ValueError("ctx.targets is required")

    n_trials = int(len(ctx.targets))
    events = ctx.aligned_events.reset_index(drop=True)
    if len(events) != n_trials:
        raise ValueError(
            f"Trial table requires aligned events length {len(events)} == targets length {n_trials}"
        )

    subject = getattr(ctx, "subject", None)
    task = getattr(ctx, "task", None)
    meta: Dict[str, Any] = {
        "subject": subject,
        "task": task,
        "n_trials": n_trials,
        "include_features": bool(include_features),
        "include_covariates": bool(include_covariates),
        "include_events": bool(include_events),
    }

    df = pd.DataFrame(index=np.arange(n_trials))
    df["subject"] = str(subject or "")
    df["task"] = str(task or "")
    df["epoch"] = np.arange(n_trials, dtype=int)

    epochs = getattr(ctx, "epochs", None)
    selection = getattr(epochs, "selection", None) if epochs is not None else None
    if selection is not None:
        try:
            selection_arr = np.asarray(selection, dtype=int)
            if selection_arr.shape[0] == n_trials:
                df["original_event_index"] = selection_arr
                meta["has_original_event_index"] = True
            else:
                meta["has_original_event_index"] = False
        except (ValueError, TypeError) as exc:
            meta["has_original_event_index"] = False
            meta["selection_error"] = str(exc)

    df["rating"] = _safe_numeric(ctx.targets).reset_index(drop=True)
    temperature = getattr(ctx, "temperature", None)
    if temperature is not None:
        df["temperature"] = _safe_numeric(temperature).reset_index(drop=True)

    pain_col = _get_pain_column_from_events(ctx.config, events)
    if pain_col is not None and pain_col in events.columns:
        df["pain_binary"] = pd.to_numeric(events[pain_col], errors="coerce")
        meta["pain_column"] = str(pain_col)

    if include_events:
        standard_event_columns = [
            "trial",
            "trial_type",
            "condition",
            "run",
            "block",
            "response_time",
        ]
        columns_to_keep = standard_event_columns.copy()
        if extra_event_columns:
            columns_to_keep.extend([str(c) for c in extra_event_columns])
        event_columns = _pick_event_columns(events, columns_to_keep)
        for column_name, column_data in event_columns.items():
            if column_name in df.columns:
                continue
            df[column_name] = column_data.reset_index(drop=True)

    if include_covariates:
        covariates_df = getattr(ctx, "covariates_df", None)
        has_covariates = covariates_df is not None and not covariates_df.empty
        if has_covariates:
            covariates = covariates_df.copy().reset_index(drop=True)
            for cov_column in covariates.columns:
                column_name = str(cov_column)
                if column_name in df.columns:
                    column_name = f"cov_{column_name}"
                df[column_name] = pd.to_numeric(covariates[cov_column], errors="coerce")

    if include_features:
        from eeg_pipeline.analysis.behavior.orchestration import combine_features

        features = combine_features(ctx)
        if features is not None and not features.empty:
            features = features.reset_index(drop=True)
            existing_columns = set(df.columns)
            feature_column_names = {str(c) for c in features.columns}
            column_collisions = existing_columns.intersection(feature_column_names)
            if column_collisions:
                rename_map = {
                    c: f"feat_{c}" for c in features.columns if str(c) in column_collisions
                }
                features = features.rename(columns=rename_map)
                meta["feature_column_collisions"] = sorted(column_collisions)
            df = pd.concat([df, features], axis=1)

    meta["n_columns"] = int(df.shape[1])
    return TrialTableBuildResult(df=df, metadata=meta)


def _compute_lag_and_delta_for_group(
    group: pd.DataFrame,
    temperature_col: str,
    rating_col: str,
    has_temperature: bool,
    has_rating: bool,
) -> pd.DataFrame:
    """Compute lagged and delta features for a single group."""
    result = group.copy()
    if has_temperature:
        temperature = pd.to_numeric(result[temperature_col], errors="coerce")
        result["prev_temperature"] = temperature.shift(1)
        result["delta_temperature"] = temperature - temperature.shift(1)
    if has_rating:
        rating = pd.to_numeric(result[rating_col], errors="coerce")
        result["prev_rating"] = rating.shift(1)
        result["delta_rating"] = rating - rating.shift(1)
    result["trial_index_within_group"] = np.arange(len(result), dtype=int)
    return result


def add_lag_and_delta_features(
    df: pd.DataFrame,
    *,
    temperature_col: str = "temperature",
    rating_col: str = "rating",
    group_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add lagged and delta variables (prev_*, delta_*) within runs/blocks if available."""
    result_df = df.copy()

    default_group_columns = ["run_id", "run", "block"]
    candidate_columns = group_columns or default_group_columns
    available_group_columns = [
        col for col in candidate_columns if col in result_df.columns
    ]
    meta: Dict[str, Any] = {"group_columns": available_group_columns}

    if "epoch" in result_df.columns:
        result_df = result_df.sort_values("epoch", kind="stable").reset_index(drop=True)

    has_temperature = temperature_col in result_df.columns
    has_rating = rating_col in result_df.columns
    if not has_temperature and not has_rating:
        meta["status"] = "skipped_no_columns"
        return result_df, meta

    def apply_to_group(group: pd.DataFrame) -> pd.DataFrame:
        return _compute_lag_and_delta_for_group(
            group, temperature_col, rating_col, has_temperature, has_rating
        )

    if available_group_columns:
        result_df = result_df.groupby(
            available_group_columns, dropna=False, sort=False, group_keys=False
        ).apply(apply_to_group)
    else:
        result_df = apply_to_group(result_df)

    meta["status"] = "ok"
    return result_df.reset_index(drop=True), meta


def _get_config_value_safe(config: Any, key: str, default: Any = None) -> Any:
    """Safely get config value, returning default if config lacks get method."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def add_pain_residual(
    df: pd.DataFrame,
    config: Any,
    *,
    temperature_col: str = "temperature",
    rating_col: str = "rating",
    out_pred_col: str = "rating_hat_from_temp",
    out_resid_col: str = "pain_residual",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add a flexible temperature→rating fit and define pain_residual = rating - f(temp)."""
    enabled = bool(
        _get_config_value_safe(config, "behavior_analysis.pain_residual.enabled", True)
    )
    meta: Dict[str, Any] = {"enabled": enabled}
    if not enabled:
        return df, meta

    has_required_columns = (
        temperature_col in df.columns and rating_col in df.columns
    )
    if not has_required_columns:
        meta["status"] = "skipped_missing_columns"
        return df, meta

    from eeg_pipeline.utils.analysis.stats.pain_residual import fit_temperature_rating_curve

    temperature = pd.to_numeric(df[temperature_col], errors="coerce")
    rating = pd.to_numeric(df[rating_col], errors="coerce")
    prediction, residual, model_meta = fit_temperature_rating_curve(
        temperature, rating, config=config
    )
    meta.update(model_meta)

    result = df.copy()
    result[out_pred_col] = prediction
    result[out_resid_col] = residual
    crossfit_enabled = bool(
        _get_config_value_safe(
            config, "behavior_analysis.pain_residual.crossfit.enabled", False
        )
    )
    if crossfit_enabled:
        from eeg_pipeline.utils.config.loader import get_config_value

        default_group_col = get_config_value(
            config, "behavior_analysis.run_adjustment.column", "run_id"
        )
        group_col_raw = get_config_value(
            config,
            "behavior_analysis.pain_residual.crossfit.group_column",
            default_group_col,
        )
        group_col = str(group_col_raw or "").strip()
        n_splits_required = int(
            get_config_value(
                config, "behavior_analysis.pain_residual.crossfit.n_splits", 5
            )
        )
        method_raw = get_config_value(
            config, "behavior_analysis.pain_residual.crossfit.method", "spline"
        )
        method = str(method_raw).strip().lower()

        meta["crossfit"] = {
            "enabled": True,
            "status": "init",
            "group_column": group_col or None,
            "method": method,
        }

        groups = (
            result[group_col] if group_col and group_col in result.columns else None
        )
        valid_mask = temperature.notna() & rating.notna()
        if groups is not None:
            groups_series = pd.Series(groups, index=result.index)
            valid_mask = valid_mask & groups_series.notna()

        min_samples_required = int(
            get_config_value(config, "behavior_analysis.pain_residual.min_samples", 10)
        )
        n_valid_samples = int(valid_mask.sum())
        if n_valid_samples < min_samples_required:
            meta["crossfit"]["status"] = "skipped_insufficient_samples"
        else:
            try:
                from sklearn.model_selection import GroupKFold
                from sklearn.pipeline import Pipeline
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
            except ImportError as exc:
                meta["crossfit"]["status"] = "skipped_missing_sklearn"
                meta["crossfit"]["error"] = str(exc)
            else:
                valid_indices = result.index[valid_mask]
                X = temperature.loc[valid_indices].to_numpy(dtype=float)[:, None]
                y = rating.loc[valid_indices].to_numpy(dtype=float)

                if groups is None:
                    meta["crossfit"]["status"] = "skipped_missing_groups"
                else:
                    group_values = (
                        pd.Series(groups, index=result.index)
                        .loc[valid_indices]
                        .to_numpy()
                    )
                    unique_groups = pd.unique(group_values)
                    unique_groups = unique_groups[~pd.isna(unique_groups)]
                    n_groups = int(len(unique_groups))
                    if n_groups < 2:
                        meta["crossfit"]["status"] = "skipped_insufficient_groups"
                    else:
                        n_splits = max(2, min(n_splits_required, n_groups))
                        model = _build_crossfit_model(config, method)
                        _update_crossfit_meta_for_method(meta, config, method)

                        cv_predictions = np.full(len(valid_indices), np.nan, dtype=float)
                        splitter = GroupKFold(n_splits=n_splits)
                        for train_idx, test_idx in splitter.split(X, y, groups=group_values):
                            model.fit(X[train_idx], y[train_idx])
                            cv_predictions[test_idx] = model.predict(X[test_idx])

                        prediction_full = pd.Series(np.nan, index=result.index, dtype=float)
                        prediction_full.loc[valid_indices] = cv_predictions
                        residual_full = pd.Series(np.nan, index=result.index, dtype=float)
                        residual_full.loc[valid_indices] = y - cv_predictions

                        result[f"{out_pred_col}_cv"] = prediction_full
                        result[f"{out_resid_col}_cv"] = residual_full
                        meta["crossfit"]["status"] = "ok"
                        meta["crossfit"]["n_valid"] = int(n_valid_samples)
                        meta["crossfit"]["n_groups"] = int(n_groups)
                        meta["crossfit"]["n_splits"] = int(n_splits)
    return result, meta


def _build_crossfit_model(config: Any, method: str) -> Any:
    """Build sklearn Pipeline model for crossfit based on method."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
    from eeg_pipeline.utils.config.loader import get_config_value

    if method == "poly":
        degree = int(
            get_config_value(config, "behavior_analysis.pain_residual.poly_degree", 2)
        )
        degree = max(1, min(degree, 5))
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
    else:
        n_knots = int(
            get_config_value(
                config, "behavior_analysis.pain_residual.crossfit.spline_n_knots", 5
            )
        )
        n_knots = max(3, min(n_knots, 12))
        return Pipeline(
            [
                ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )


def _update_crossfit_meta_for_method(meta: Dict[str, Any], config: Any, method: str) -> None:
    """Update crossfit metadata with method-specific parameters."""
    from eeg_pipeline.utils.config.loader import get_config_value

    if method == "poly":
        degree = int(
            get_config_value(config, "behavior_analysis.pain_residual.poly_degree", 2)
        )
        degree = max(1, min(degree, 5))
        meta["crossfit"]["poly_degree"] = int(degree)
    else:
        n_knots = int(
            get_config_value(
                config, "behavior_analysis.pain_residual.crossfit.spline_n_knots", 5
            )
        )
        n_knots = max(3, min(n_knots, 12))
        meta["crossfit"]["spline_n_knots"] = int(n_knots)


def save_trial_table(
    result: TrialTableBuildResult,
    out_path: Path,
    *,
    format: str = "tsv",  # noqa: A002
) -> Path:
    """Save trial table to file in specified format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    format_normalized = str(format).strip().lower()
    if format_normalized in {"tsv", "txt"}:
        from eeg_pipeline.infra.tsv import write_tsv

        write_tsv(result.df, out_path, index=False)
    elif format_normalized == "parquet":
        result.df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported trial table format: {format}")
    return out_path


__all__ = [
    "TrialTableBuildResult",
    "build_subject_trial_table",
    "add_lag_and_delta_features",
    "add_pain_residual",
    "save_trial_table",
]
