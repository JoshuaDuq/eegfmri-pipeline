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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrialTableBuildResult:
    df: pd.DataFrame
    metadata: Dict[str, Any]


TRIAL_TABLE_CONTRACT_VERSION = "1.0"


def _is_dataframe_valid(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _sanitize_path_component(value: str) -> str:
    value = str(value).strip()
    if not value:
        return "all"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    out = []
    for ch in value:
        out.append(ch if ch in allowed else "_")
    cleaned = "".join(out).strip("._-")
    return cleaned if cleaned else "all"


def normalize_trial_table_feature_selection(feature_files: Optional[List[str]]) -> List[str]:
    """Normalize feature selection list for trial-table naming and lookup."""
    if not feature_files:
        return []
    normalized = {
        str(item).strip()
        for item in feature_files
        if str(item).strip() and str(item).strip().lower() != "all"
    }
    return sorted(normalized)


def trial_table_suffix_from_features(feature_files: Optional[List[str]]) -> str:
    """Return trial-table filename suffix for a feature selection."""
    selected = normalize_trial_table_feature_selection(feature_files)
    if len(selected) > 1:
        return "_all"
    if selected:
        return "_" + "_".join(selected)
    return "_all"


def trial_table_feature_folder_from_features(feature_files: Optional[List[str]]) -> str:
    """Return trial-table folder name for a feature selection."""
    selected = normalize_trial_table_feature_selection(feature_files)
    if len(selected) > 1:
        return "all"
    if not selected:
        return "all"
    return _sanitize_path_component("_".join(selected))


def _select_preferred_trial_table_candidate(
    existing: List[Path],
    *,
    stats_dir: Path,
    error_context: str,
) -> Path:
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for path in existing:
        key = (str(path.parent), path.stem)
        grouped.setdefault(key, []).append(path)
    if len(grouped) > 1:
        raise ValueError(
            f"Multiple trial table files found in {stats_dir}{error_context}: {existing}. "
            "Specify feature files to disambiguate."
        )
    options = next(iter(grouped.values()))
    parquet = [p for p in options if p.suffix == ".parquet"]
    if parquet:
        return parquet[0]
    return options[0]


def discover_trial_table_candidates(stats_dir: Path) -> List[Path]:
    """Discover all trial-table candidates under trial_table* roots."""
    candidates: List[Path] = []
    for root in sorted(p for p in stats_dir.glob("trial_table*") if p.is_dir()):
        candidates.extend(sorted(root.glob("*/trials_*.parquet")))
        candidates.extend(sorted(root.glob("*/trials_*.tsv")))
    return candidates


def select_preferred_trial_tables(candidates: List[Path]) -> List[Path]:
    """Prefer parquet over TSV for each (parent, stem) candidate key."""
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for path in candidates:
        key = (str(path.parent), path.stem)
        grouped.setdefault(key, []).append(path)
    selected: List[Path] = []
    for key in sorted(grouped.keys()):
        options = grouped[key]
        parquet = [p for p in options if p.suffix == ".parquet"]
        selected.append(parquet[0] if parquet else sorted(options)[0])
    return selected


def find_trial_table_path(
    stats_dir: Path,
    feature_files: Optional[List[str]] = None,
) -> Optional[Path]:
    """Find the preferred trial-table path under a stats directory."""
    trial_table_roots = sorted(p for p in stats_dir.glob("trial_table*") if p.is_dir())
    if not trial_table_roots:
        return None

    if feature_files:
        suffix = trial_table_suffix_from_features(feature_files)
        fname = f"trials{suffix}"
        feature_dir = trial_table_feature_folder_from_features(feature_files)
        candidate_keys: List[Tuple[str, str]] = [(feature_dir, fname)]

        candidates: List[Path] = []
        for root in trial_table_roots:
            for folder, stem in candidate_keys:
                candidates.append(root / folder / f"{stem}.parquet")
                candidates.append(root / folder / f"{stem}.tsv")
        existing = [p for p in candidates if p.exists()]
        if len(existing) == 0:
            return None
        return _select_preferred_trial_table_candidate(
            existing,
            stats_dir=stats_dir,
            error_context=f" for {feature_dir}",
        )

    # Canonical precedence for unfiltered discovery:
    # 1) all/trials_all.{parquet,tsv}
    canonical_candidates: List[Tuple[int, int, float, Path]] = []
    stem_priority = {"trials_all": 0}
    ext_priority = {".parquet": 0, ".tsv": 1}
    for root in trial_table_roots:
        for stem, stem_rank in stem_priority.items():
            for ext, ext_rank in ext_priority.items():
                candidate = root / "all" / f"{stem}{ext}"
                if candidate.exists():
                    try:
                        mtime = float(candidate.stat().st_mtime)
                    except OSError:
                        mtime = 0.0
                    canonical_candidates.append((stem_rank, ext_rank, -mtime, candidate))
    if canonical_candidates:
        canonical_candidates.sort()
        return canonical_candidates[0][3]

    pattern_paths = discover_trial_table_candidates(stats_dir)
    if len(pattern_paths) == 0:
        return None
    return _select_preferred_trial_table_candidate(
        pattern_paths,
        stats_dir=stats_dir,
        error_context="",
    )


def merge_trial_tables(trial_paths: List[Path]) -> pd.DataFrame:
    """Merge multiple trial tables into a single DataFrame for discovery/plotting."""
    if not trial_paths:
        raise FileNotFoundError("No trial table files available for merge.")

    from eeg_pipeline.infra.tsv import read_table

    merged: Optional[pd.DataFrame] = None
    key_sets: List[List[str]] = [
        ["subject", "task", "run_id", "trial"],
        ["subject", "run_id", "trial"],
        ["run_id", "trial"],
        ["trial"],
        ["epoch"],
        ["onset", "duration"],
    ]

    for path in trial_paths:
        df = read_table(path)
        if df.empty:
            continue

        if merged is None:
            merged = df.copy()
            continue

        merge_keys: Optional[List[str]] = None
        for keys in key_sets:
            if not all(k in merged.columns and k in df.columns for k in keys):
                continue
            if merged.duplicated(subset=keys).any() or df.duplicated(subset=keys).any():
                continue
            merge_keys = keys
            break

        if merge_keys is not None:
            merged = merged.merge(df, on=merge_keys, how="outer", suffixes=("", "__dup"))
            dup_cols = [c for c in merged.columns if c.endswith("__dup")]
            if dup_cols:
                merged = merged.drop(columns=dup_cols)
            continue

        if len(merged) == len(df):
            extra = df.drop(columns=[c for c in df.columns if c in merged.columns], errors="ignore")
            merged = pd.concat([merged.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
            continue

        raise ValueError(
            "Unable to merge multiple trial tables because no unique join keys were found "
            f"and row counts differ ({len(merged)} vs {len(df)})."
        )

    if merged is None or merged.empty:
        raise ValueError("Merged trial table is empty.")
    return merged


def _rename_feature_columns_with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {
        col: col if str(col).startswith(prefix) else f"{prefix}{col}"
        for col in df.columns
    }
    if all(k == v for k, v in rename_map.items()):
        return df
    return df.rename(columns=rename_map)


def combine_feature_tables(
    feature_tables: Iterable[Tuple[str, Optional[pd.DataFrame]]],
) -> pd.DataFrame:
    """Combine feature tables into one DataFrame with stable prefixed column names."""
    feature_dataframes: List[pd.DataFrame] = []
    base_index: Optional[pd.Index] = None
    existing_columns = pd.Index([])

    for name, df in feature_tables:
        if not _is_dataframe_valid(df):
            continue

        if base_index is None:
            base_index = df.index
        elif not df.index.equals(base_index):
            raise ValueError(
                f"Feature index mismatch for {name}: expected alignment of {len(base_index)} rows."
            )

        prefix = f"{name}_"
        df_renamed = _rename_feature_columns_with_prefix(df, prefix)
        if df_renamed.columns.duplicated().any():
            dup_names = [str(c) for c in df_renamed.columns[df_renamed.columns.duplicated()].unique()]
            raise ValueError(f"Duplicate feature columns within {name}: {dup_names}")

        overlap = existing_columns.intersection(df_renamed.columns)
        if not overlap.empty:
            overlap_list = [str(c) for c in overlap.tolist()]
            raise ValueError(f"Duplicate feature columns across tables: {overlap_list}")

        feature_dataframes.append(df_renamed)
        existing_columns = existing_columns.append(df_renamed.columns)

    combined = pd.concat(feature_dataframes, axis=1) if feature_dataframes else pd.DataFrame()
    if not combined.empty and combined.columns.duplicated().any():
        dup_names = [str(c) for c in combined.columns[combined.columns.duplicated()].unique()]
        raise ValueError(f"Duplicate feature columns after combining: {dup_names}")
    return combined


def compute_feature_tables_signature(
    feature_tables: Iterable[Tuple[str, Optional[pd.DataFrame]]],
) -> str:
    """Compute a deterministic signature of feature tables for metadata contracts."""
    parts: List[str] = []
    for name, df in feature_tables:
        if not _is_dataframe_valid(df):
            continue
        column_names = ",".join(str(c) for c in df.columns)
        parts.append(f"{name}:{df.shape[0]}:{df.shape[1]}:{column_names}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    events = ctx.aligned_events.reset_index(drop=True)
    feature_tables = list(ctx.iter_feature_tables()) if hasattr(ctx, "iter_feature_tables") else []
    features = combine_feature_tables(feature_tables)

    if features is not None and not features.empty:
        features = features.reset_index(drop=True)
        df = pd.concat([events, features], axis=1)
    else:
        df = events.copy()
    n_feature_rows = int(len(features)) if features is not None else 0
    feature_signature = getattr(ctx, "_combined_features_signature", None)
    if not feature_signature:
        feature_signature = compute_feature_tables_signature(feature_tables)

    meta: Dict[str, Any] = {
        "subject": getattr(ctx, "subject", None),
        "task": getattr(ctx, "task", None),
        "n_trials": len(df),
        "n_columns": df.shape[1],
        "contract": build_trial_table_contract(
            df,
            n_events_rows=len(events),
            n_feature_rows=n_feature_rows,
            feature_signature=feature_signature,
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


def add_predictor_residual(
    df: pd.DataFrame,
    config: Any,
    *,
    temperature_col: str = "temperature",
    rating_col: str = "rating",
    out_pred_col: str = "rating_hat_from_temp",
    out_resid_col: str = "predictor_residual",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add a flexible temperature→rating fit and define predictor_residual = rating - f(temp)."""
    from eeg_pipeline.utils.config.loader import get_config_value

    enabled = bool(
        get_config_value(config, "behavior_analysis.predictor_residual.enabled", True)
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

    from eeg_pipeline.utils.analysis.stats.predictor_residual import fit_temperature_rating_curve

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
            config, "behavior_analysis.predictor_residual.crossfit.enabled", False
        )
    )
    if crossfit_enabled:
        default_group_col = get_config_value(
            config, "behavior_analysis.run_adjustment.column", "run_id"
        )
        group_col_raw = get_config_value(
            config,
            "behavior_analysis.predictor_residual.crossfit.group_column",
            default_group_col,
        )
        group_col = str(group_col_raw or "").strip()
        n_splits_required = int(
            get_config_value(
                config, "behavior_analysis.predictor_residual.crossfit.n_splits", 5
            )
        )
        method_raw = get_config_value(
            config, "behavior_analysis.predictor_residual.crossfit.method", "spline"
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
            get_config_value(config, "behavior_analysis.predictor_residual.min_samples", 10)
        )
        n_valid_samples = int(valid_mask.sum())
        if n_valid_samples < min_samples_required:
            meta["crossfit"]["status"] = "skipped_insufficient_samples"
        else:
            try:
                from sklearn.model_selection import GroupKFold
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
            get_config_value(config, "behavior_analysis.predictor_residual.poly_degree", 2)
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
                config, "behavior_analysis.predictor_residual.crossfit.spline_n_knots", 5
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
    "combine_feature_tables",
    "compute_feature_tables_signature",
    "build_subject_trial_table",
    "build_trial_table_contract",
    "compute_trial_table_schema_hash",
    "validate_trial_table_contract",
    "normalize_trial_table_feature_selection",
    "trial_table_suffix_from_features",
    "trial_table_feature_folder_from_features",
    "discover_trial_table_candidates",
    "select_preferred_trial_tables",
    "find_trial_table_path",
    "merge_trial_tables",
    "add_lag_and_delta_features",
    "add_predictor_residual",
    "save_trial_table",
]
