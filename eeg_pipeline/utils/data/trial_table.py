"""
Subject-Level Trial Table
=========================

Builds a per-trial table: clean events.tsv columns + feature columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrialTableBuildResult:
    df: pd.DataFrame
    metadata: Dict[str, Any]


TRIAL_TABLE_CONTRACT_VERSION = "1.0"


def _schema_entries(df: pd.DataFrame) -> List[Dict[str, str]]:
    return [
        {"name": str(col), "dtype": str(df[col].dtype)}
        for col in df.columns
    ]


def compute_trial_table_schema_hash(df: pd.DataFrame) -> str:
    payload = json.dumps(_schema_entries(df), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_trial_table_contract(
    df: pd.DataFrame,
    *,
    n_events_rows: int,
    n_feature_rows: int,
    feature_signature: Optional[str],
) -> Dict[str, Any]:
    return {
        "version": TRIAL_TABLE_CONTRACT_VERSION,
        "schema_hash": compute_trial_table_schema_hash(df),
        "n_events_rows": int(n_events_rows),
        "n_feature_rows": int(n_feature_rows),
        "feature_signature": str(feature_signature) if feature_signature else None,
    }


def validate_trial_table_contract(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    n_trials_meta = metadata.get("n_trials")
    n_columns_meta = metadata.get("n_columns")
    if n_trials_meta is not None and int(n_trials_meta) != int(len(df)):
        errors.append(f"n_trials metadata mismatch: expected {n_trials_meta}, got {len(df)}")
    if n_columns_meta is not None and int(n_columns_meta) != int(df.shape[1]):
        errors.append(f"n_columns metadata mismatch: expected {n_columns_meta}, got {df.shape[1]}")

    contract = metadata.get("contract", {}) or {}
    expected_hash = contract.get("schema_hash")
    if expected_hash:
        actual_hash = compute_trial_table_schema_hash(df)
        if str(expected_hash) != actual_hash:
            errors.append(
                "contract schema_hash mismatch: expected "
                f"{expected_hash}, got {actual_hash}"
            )

    return errors


def build_subject_trial_table(ctx: Any) -> TrialTableBuildResult:
    """Build trial table: clean events columns + feature columns."""
    if ctx.aligned_events is None:
        raise ValueError("ctx.aligned_events is required")

    from eeg_pipeline.analysis.behavior.orchestration import combine_features

    events = ctx.aligned_events.reset_index(drop=True)
    features = combine_features(ctx)

    if features is not None and not features.empty:
        features = features.reset_index(drop=True)
        df = pd.concat([events, features], axis=1)
    else:
        df = events.copy()
    n_feature_rows = int(len(features)) if features is not None else 0

    meta: Dict[str, Any] = {
        "subject": getattr(ctx, "subject", None),
        "task": getattr(ctx, "task", None),
        "n_trials": len(df),
        "n_columns": df.shape[1],
        "contract": build_trial_table_contract(
            df,
            n_events_rows=len(events),
            n_feature_rows=n_feature_rows,
            feature_signature=getattr(ctx, "_combined_features_signature", None),
        ),
    }
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
    from eeg_pipeline.utils.config.loader import get_config_value

    enabled = bool(
        get_config_value(config, "behavior_analysis.pain_residual.enabled", True)
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
        get_config_value(
            config, "behavior_analysis.pain_residual.crossfit.enabled", False
        )
    )
    if crossfit_enabled:
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
                        model, model_params = _build_crossfit_model(config, method)
                        meta["crossfit"].update(model_params)

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


def _build_crossfit_model(config: Any, method: str) -> Tuple[Any, Dict[str, Any]]:
    """Build sklearn Pipeline model for crossfit based on method.
    
    Returns
    -------
    model : sklearn Pipeline
        The configured model.
    params : dict
        Method-specific parameters for metadata.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
    from eeg_pipeline.utils.config.loader import get_config_value

    if method == "poly":
        degree = int(
            get_config_value(config, "behavior_analysis.pain_residual.poly_degree", 2)
        )
        degree = max(1, min(degree, 5))
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        return model, {"poly_degree": int(degree)}
    else:
        n_knots = int(
            get_config_value(
                config, "behavior_analysis.pain_residual.crossfit.spline_n_knots", 5
            )
        )
        n_knots = max(3, min(n_knots, 12))
        model = Pipeline(
            [
                ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        return model, {"spline_n_knots": int(n_knots)}


def save_trial_table(
    result: TrialTableBuildResult,
    out_path: Path,
    *,
    format: str = "parquet",  # noqa: A002
) -> Path:
    """Save trial table to file in specified format (parquet preferred)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    format_normalized = str(format).strip().lower()
    if format_normalized == "parquet":
        from eeg_pipeline.infra.tsv import write_parquet
        write_parquet(result.df, out_path)
    elif format_normalized in {"tsv", "txt"}:
        from eeg_pipeline.infra.tsv import write_tsv
        write_tsv(result.df, out_path, index=False)
    else:
        raise ValueError(f"Unsupported trial table format: {format}")
    return out_path


__all__ = [
    "TRIAL_TABLE_CONTRACT_VERSION",
    "TrialTableBuildResult",
    "build_subject_trial_table",
    "build_trial_table_contract",
    "compute_trial_table_schema_hash",
    "validate_trial_table_contract",
    "add_lag_and_delta_features",
    "add_pain_residual",
    "save_trial_table",
]
