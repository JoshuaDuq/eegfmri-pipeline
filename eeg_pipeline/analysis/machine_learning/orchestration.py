"""
Machine Learning Orchestration (Canonical)
=============================================

Single source of truth for ML *compute* stages.

Design
------
- ML uses per-trial feature tables saved by the feature pipeline (derivatives/*/eeg/features/*).
- Targets/covariates come from clean events.tsv.
- Outer CV is group-aware (typically LOSO).

This module intentionally avoids maintaining multiple legacy CV implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from eeg_pipeline.analysis.machine_learning.cv import (
    create_within_subject_folds,
    create_block_aware_inner_cv,
    create_inner_cv,
    apply_fold_feature_harmonization,
    create_scoring_dict,
    determine_inner_n_jobs,
    grid_search_with_warning_logging,
    _fit_default_pipeline,
    nested_loso_predictions_matrix,
    compute_subject_level_r,
    compute_subject_level_errors,
    is_effective_permutation,
    safe_pearsonr,
)
from eeg_pipeline.analysis.machine_learning.pipelines import (
    create_elasticnet_pipeline,
    create_ridge_pipeline,
    create_rf_pipeline,
    build_elasticnet_param_grid,
    build_ridge_param_grid,
    build_rf_param_grid,
)
from eeg_pipeline.utils.data.machine_learning import load_active_matrix, load_epoch_tensor_matrix
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.machine_learning import (
    export_predictions,
    export_indices,
    prepare_best_params_path,
)
from eeg_pipeline.infra.tsv import write_tsv, write_parquet
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.analysis.machine_learning.time_generalization import time_generalization_regression

logger = get_logger(__name__)


###################################################################
# Within-Subject Helper
###################################################################

def _warn_or_raise_if_binary_like_regression_target(
    y: np.ndarray,
    target: Optional[str],
    logger: logging.Logger,
    config: Any,
    *,
    context: str,
) -> Dict[str, Any]:
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"binary_like": False, "unique_values": []}

    unique = np.unique(finite)
    # Strict "binary-like" detection: exactly 2 unique values and subset of {0,1}.
    binary_like = unique.size == 2 and set(unique.tolist()).issubset({0.0, 1.0})
    uniques_list = unique.tolist()
    if len(uniques_list) > 10:
        uniques_list = uniques_list[:10]

    if binary_like:
        strict = bool(get_config_value(config, "machine_learning.targets.strict_regression_target_continuous", False))
        msg = (
            f"{context}: regression target appears binary-like (unique={unique.tolist()}). "
            f"target={target!r}. Prefer ML mode 'classify' for binary outcomes."
        )
        if strict:
            raise ValueError(msg + " (Blocked by machine_learning.targets.strict_regression_target_continuous=true)")
        logger.warning(msg)

    return {"binary_like": bool(binary_like), "unique_values": uniques_list}


def _subject_mean_metric(
    per_subject_metrics: Optional[Dict[str, Dict[str, Any]]],
    key: str,
) -> float:
    """Mean of a per-subject metric, treating subject as the inferential unit."""
    if not per_subject_metrics:
        return np.nan

    vals: List[float] = []
    for rec in per_subject_metrics.values():
        if not isinstance(rec, dict):
            continue
        value = rec.get(key)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _count_finite_subject_metric(
    per_subject_metrics: Optional[Dict[str, Dict[str, Any]]],
    key: str,
) -> int:
    if not per_subject_metrics:
        return 0
    count = 0
    for rec in per_subject_metrics.values():
        if not isinstance(rec, dict):
            continue
        value = rec.get(key)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            count += 1
    return int(count)


def _subject_metric_values(
    per_subject_metrics: Optional[Dict[str, Dict[str, Any]]],
    key: str,
) -> np.ndarray:
    """Collect finite per-subject metric values as float array."""
    if not per_subject_metrics:
        return np.asarray([], dtype=float)
    vals: List[float] = []
    for rec in per_subject_metrics.values():
        if not isinstance(rec, dict):
            continue
        value = rec.get(key)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        v = float(vals[0])
        return v, v

    boot = np.empty(int(max(10, n_boot)), dtype=float)
    n = int(vals.size)
    for i in range(len(boot)):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = float(np.mean(vals[idx]))
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _paired_signflip_p_value(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    vals = np.asarray(deltas, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or int(n_perm) <= 0:
        return np.nan

    observed = float(abs(np.mean(vals)))
    exceed = 0
    n = int(vals.size)
    for _ in range(int(n_perm)):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=n, replace=True)
        stat = float(abs(np.mean(vals * signs)))
        if stat >= observed:
            exceed += 1
    return float((exceed + 1) / (int(n_perm) + 1))


def _resolve_permutation_scheme(config: Any) -> str:
    scheme = str(
        get_config_value(config, "machine_learning.cv.permutation_scheme", "within_subject_within_block")
    ).strip().lower()
    if scheme not in {"within_subject", "within_subject_within_block"}:
        scheme = "within_subject_within_block"
    return scheme


def _permute_labels_by_scheme(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    blocks: Optional[np.ndarray],
    rng: np.random.Generator,
    scheme: str,
) -> np.ndarray:
    """Permute labels within-subject or within-subject×block."""
    y_perm = np.asarray(y, dtype=float).copy()
    groups_arr = np.asarray(groups, dtype=object)
    blocks_arr = np.asarray(blocks) if blocks is not None else None
    mode = str(scheme).strip().lower()
    if mode not in {"within_subject", "within_subject_within_block"}:
        mode = "within_subject_within_block"
    if mode == "within_subject_within_block" and blocks_arr is None:
        mode = "within_subject"

    for subj in np.unique(groups_arr):
        subj_mask = groups_arr == subj
        if np.sum(subj_mask) < 2:
            continue
        if mode == "within_subject_within_block" and blocks_arr is not None:
            subj_blocks = blocks_arr[subj_mask]
            subj_y = y_perm[subj_mask]
            for block_id in np.unique(subj_blocks):
                if pd.isna(block_id):
                    block_mask = pd.isna(subj_blocks)
                else:
                    block_mask = subj_blocks == block_id
                block_indices = np.where(block_mask)[0]
                if block_indices.size >= 2:
                    subj_y[block_indices] = rng.permutation(subj_y[block_indices])
            y_perm[subj_mask] = subj_y
        else:
            y_perm[subj_mask] = rng.permutation(y_perm[subj_mask])
    return y_perm


def _generate_effective_permutation(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    blocks: Optional[np.ndarray],
    rng: np.random.Generator,
    requested_scheme: str,
    min_changed_fraction: float,
) -> Tuple[np.ndarray, bool, float, str]:
    """
    Generate one permutation, falling back from subject×block to within-subject
    when the requested shuffle is ineffective.
    """
    y_perm = _permute_labels_by_scheme(
        y,
        groups,
        blocks=blocks,
        rng=rng,
        scheme=requested_scheme,
    )
    effective, changed_fraction = is_effective_permutation(
        y,
        y_perm,
        min_changed_fraction=min_changed_fraction,
    )
    if effective:
        return y_perm, True, float(changed_fraction), requested_scheme

    if requested_scheme == "within_subject_within_block":
        y_perm_fallback = _permute_labels_by_scheme(
            y,
            groups,
            blocks=blocks,
            rng=rng,
            scheme="within_subject",
        )
        effective_fallback, changed_fraction_fallback = is_effective_permutation(
            y,
            y_perm_fallback,
            min_changed_fraction=min_changed_fraction,
        )
        if effective_fallback:
            return y_perm_fallback, True, float(changed_fraction_fallback), "within_subject"

    return y_perm, False, float(changed_fraction), requested_scheme


def _normalize_subject_ids(subjects: List[str]) -> List[str]:
    out: List[str] = []
    for s in subjects:
        s_str = str(s).strip()
        out.append(s_str if s_str.startswith("sub-") else f"sub-{s_str}")
    return out


def _target_covariate_aliases(target: Optional[str], config: Optional[Any] = None) -> set[str]:
    """Return standardized target aliases to guard against predictor leakage."""
    target_raw = str(target or "").strip()
    target_key = target_raw.lower()
    aliases: set[str] = set()

    if target_key in {"", "rating", "pain_rating", "vas"}:
        aliases.add("rating")
    elif target_key in {"temperature", "temp"}:
        aliases.add("temperature")
    elif target_key in {"pain", "pain_binary", "binary"}:
        aliases.add("pain_binary")
    elif target_key in {"fmri_signature", "fmri-signature"}:
        aliases.add("fmri_signature")

    # If target is an explicit events.tsv column name, map back to canonical
    # aliases so baseline-predictor leakage guards remain effective.
    if config is not None:
        event_alias_map = {
            "rating": get_config_value(config, "event_columns.rating", []),
            "temperature": get_config_value(config, "event_columns.temperature", []),
            "pain_binary": get_config_value(config, "event_columns.pain_binary", []),
        }
        for canonical, aliases_raw in event_alias_map.items():
            if isinstance(aliases_raw, str):
                alias_list = [aliases_raw]
            elif isinstance(aliases_raw, (list, tuple)):
                alias_list = [str(v) for v in aliases_raw]
            else:
                alias_list = []
            normalized = {str(v).strip().lower() for v in alias_list if str(v).strip()}
            if target_key and target_key in normalized:
                aliases.add(canonical)

    if target_raw:
        aliases.add(target_raw)
    return aliases


def _build_regression_model_spec(
    model_name: Optional[str],
    *,
    seed: int,
    config: Any,
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    name = str(model_name or "elasticnet").strip().lower()
    if name == "ridge":
        return "ridge", create_ridge_pipeline(seed=seed, config=config), build_ridge_param_grid(config)
    if name == "rf":
        return "rf", create_rf_pipeline(seed=seed, config=config), build_rf_param_grid(config)
    return "elasticnet", create_elasticnet_pipeline(seed=seed, config=config), build_elasticnet_param_grid(config)


def _fit_tuned_regression_estimator(
    *,
    base_pipe: Pipeline,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    inner_splits: int,
    seed: int,
    logger: logging.Logger,
    fold_info: str,
) -> Pipeline:
    n_unique = len(np.unique(groups_train))
    if n_unique < 2:
        est = clone(base_pipe)
        est.fit(X_train, y_train)
        return est

    inner_cv = create_inner_cv(groups_train, inner_splits)
    scoring = create_scoring_dict()
    gs = GridSearchCV(
        estimator=clone(base_pipe),
        param_grid=param_grid,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=1,
        refit="r",
        error_score="raise",
    )
    gs = grid_search_with_warning_logging(gs, X_train, y_train, fold_info=fold_info, log=logger, groups=groups_train)
    return gs.best_estimator_


def export_subject_selection_report(
    results_dir: Path,
    subjects_requested: List[str],
    groups_used: np.ndarray,
    meta: Optional[pd.DataFrame],
    config: Any,
) -> Dict[str, Any]:
    """Write included/excluded subject reports for ML run transparency."""
    requested_ids = _normalize_subject_ids(subjects_requested)
    included_ids = sorted({str(g) for g in np.asarray(groups_used, dtype=object).tolist()})

    excluded_records: List[Dict[str, str]] = []
    if meta is not None and hasattr(meta, "attrs"):
        raw = meta.attrs.get("excluded_subjects", [])
        if isinstance(raw, list):
            for rec in raw:
                if not isinstance(rec, dict):
                    continue
                subject_id = str(rec.get("subject_id", "")).strip()
                reason = str(rec.get("reason", "")).strip()
                if subject_id:
                    excluded_records.append(
                        {
                            "subject_id": subject_id,
                            "reason": reason or "Excluded during matrix assembly.",
                        }
                    )

    reason_map = {r["subject_id"]: r["reason"] for r in excluded_records}
    excluded_ids = sorted(set(requested_ids) - set(included_ids))
    excluded_rows = [
        {"subject_id": sid, "reason": reason_map.get(sid, "Excluded during matrix assembly.")}
        for sid in excluded_ids
    ]

    included_df = pd.DataFrame({"subject_id": included_ids})
    excluded_df = pd.DataFrame(excluded_rows, columns=["subject_id", "reason"])
    included_df.to_csv(results_dir / "included_subjects.tsv", sep="\t", index=False)
    excluded_df.to_csv(results_dir / "excluded_subjects.tsv", sep="\t", index=False)

    n_requested = len(requested_ids)
    n_excluded = len(excluded_ids)
    exclusion_fraction = float(n_excluded / n_requested) if n_requested > 0 else 0.0
    threshold = float(get_config_value(config, "machine_learning.data.max_excluded_subject_fraction", 1.0))
    return {
        "n_requested": n_requested,
        "n_included": int(len(included_ids)),
        "n_excluded": n_excluded,
        "excluded_fraction": exclusion_fraction,
        "max_excluded_subject_fraction": threshold,
    }


def _fit_within_subject_fold(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    blocks_train: Optional[np.ndarray],
    fold: int,
    subject_id: str,
    random_state: int,
    n_jobs: int,
    logger: logging.Logger,
    inner_splits: int,
    param_grid: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    if blocks_train is not None:
        n_unique_blocks = len(np.unique(blocks_train))
        n_splits_inner = max(2, min(n_unique_blocks, int(inner_splits)))
        
        if n_splits_inner < 2:
            return _fit_default_pipeline(pipe, X_train, y_train, fold)
        
        inner_cv_splits = create_block_aware_inner_cv(blocks_train, n_splits_inner, random_state, fold, subject_id)
        
        if inner_cv_splits is not None and len(inner_cv_splits) >= 2:
            scoring = create_scoring_dict()
            refit_metric = 'r'
            
            effective_param_grid = dict(param_grid or {})
            pipe_seeded = clone(pipe)
            regressor_step = pipe_seeded.named_steps.get("regressor")
            if regressor_step is not None and hasattr(regressor_step, "regressor") and hasattr(regressor_step.regressor, "random_state"):
                regressor_step.regressor.random_state = random_state
            gs = GridSearchCV(
                estimator=pipe_seeded,
                param_grid=effective_param_grid,
                scoring=scoring,
                cv=inner_cv_splits,
                n_jobs=n_jobs,
                refit=refit_metric,
                error_score="raise",
            )
            try:
                gs = grid_search_with_warning_logging(
                    gs, X_train, y_train,
                    fold_info=f"within-subject fold {fold} (subject {subject_id})",
                    log=logger,
                    groups=blocks_train
                )
                return gs.best_estimator_
            except Exception as exc:
                logger.warning(
                    "Within-subject fold %s (%s): inner CV failed (%s); fitting default pipeline.",
                    int(fold),
                    str(subject_id),
                    exc,
                )
                return _fit_default_pipeline(pipe, X_train, y_train, fold, random_state)
    
    return _fit_default_pipeline(pipe, X_train, y_train, fold, random_state)


def _maybe_generate_mode_plots(
    *,
    mode: str,
    results_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Best-effort plotting hook; never raises into compute stages."""
    try:
        from eeg_pipeline.analysis.machine_learning.plotting import generate_ml_mode_plots

        enabled = bool(get_config_value(config, "machine_learning.plotting.enabled", True))
        if not enabled:
            return
        formats_raw = get_config_value(config, "machine_learning.plotting.formats", ["png"])
        if isinstance(formats_raw, (list, tuple)):
            formats = [str(v).strip() for v in formats_raw if str(v).strip()]
        elif isinstance(formats_raw, str):
            formats = [p for p in formats_raw.replace(",", " ").split() if p]
        else:
            formats = ["png"]
        dpi_val = int(get_config_value(config, "machine_learning.plotting.dpi", 300))
        top_n = int(get_config_value(config, "machine_learning.plotting.top_n_features", 20))
        include_diagnostics = bool(
            get_config_value(config, "machine_learning.plotting.include_diagnostics", True)
        )
        outputs = generate_ml_mode_plots(
            mode=mode,
            results_dir=results_dir,
            logger=logger,
            formats=formats,
            dpi=dpi_val,
            top_n_features=top_n,
            include_diagnostics=include_diagnostics,
        )
        if outputs:
            logger.info("Generated %d ML plot(s) for mode=%s", len(outputs), mode)
    except Exception as exc:
        logger.warning("Failed to generate ML plots for mode=%s (continuing): %s", mode, exc)


###################################################################
# Pipeline-Level Runners
###################################################################


def run_regression_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    model: str = "elasticnet",
    *,
    target: Optional[str] = None,
    feature_families: Optional[List[str]] = None,
    feature_harmonization: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Path:
    """Run LOSO regression on per-trial feature-table inputs."""
    if target is None:
        target = get_config_value(config, "machine_learning.targets.regression", None)

    X, y, groups, _feature_names, meta = load_active_matrix(
        subjects,
        task,
        deriv_root,
        config,
        logger,
        feature_families=feature_families,
        feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
        target=target,
        target_kind="continuous",
        covariates=covariates,
        feature_bands=feature_bands,
        feature_segments=feature_segments,
        feature_scopes=feature_scopes,
        feature_stats=feature_stats,
    )
    target_detection = _warn_or_raise_if_binary_like_regression_target(
        y,
        target,
        logger,
        config,
        context="LOSO regression",
    )
    blocks = None
    if meta is not None and hasattr(meta, "columns") and "block" in meta.columns:
        blocks = pd.to_numeric(meta["block"], errors="coerce").to_numpy()

    results_dir = results_root / "regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    subject_selection = export_subject_selection_report(
        results_dir,
        subjects,
        groups,
        meta,
        config,
    )

    if model == "ridge":
        pipe = create_ridge_pipeline(seed=rng_seed, config=config)
        param_grid = build_ridge_param_grid(config)
        model_name = "ridge"
    elif model == "rf":
        pipe = create_rf_pipeline(seed=rng_seed, config=config)
        param_grid = build_rf_param_grid(config)
        model_name = "rf"
    else:
        pipe = create_elasticnet_pipeline(seed=rng_seed, config=config)
        param_grid = build_elasticnet_param_grid(config)
        model_name = "elasticnet"
    
    best_params_path = prepare_best_params_path(results_dir / f"best_params_{model_name}.jsonl", mode="truncate")
    null_path = results_dir / f"loso_null_{model_name}.npz" if n_perm > 0 else None

    n_subjects = len(np.unique(groups))
    logger.info(
        "Regression: model=%s, %d features \u00d7 %d trials, %d subjects, target='%s'",
        model_name, X.shape[1], X.shape[0], n_subjects, target,
    )

    y_true, y_pred, groups_ordered, test_indices, fold_ids = nested_loso_predictions_matrix(
        X=X,
        y=y,
        groups=groups,
        blocks=blocks,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_splits,
        n_jobs=1,
        seed=rng_seed,
        best_params_log_path=best_params_path,
        model_name=model_name,
        outer_n_jobs=outer_jobs,
        null_n_perm=n_perm,
        null_output_path=null_path,
        config=config,
        harmonization_mode=feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")),
    )

    pred_path = results_dir / "loso_predictions.tsv"
    pred_df = export_predictions(
        y_true,
        y_pred,
        groups_ordered,
        test_indices,
        fold_ids,
        model_name,
        meta.reset_index(drop=True),
        pred_path,
    )
    export_indices(
        groups_ordered,
        test_indices,
        fold_ids,
        meta.reset_index(drop=True),
        results_dir / "loso_indices.tsv",
        add_heldout_subject_id=True,
    )

    ci_method = str(get_config_value(config, "machine_learning.evaluation.ci_method", "bootstrap"))
    r_subj, _per_subj_r, ci_low, ci_high = compute_subject_level_r(pred_df, config, ci_method=ci_method)
    if _per_subj_r:
        pd.DataFrame(_per_subj_r, columns=["subject_id", "pearson_r"]).to_csv(
            results_dir / "per_subject_correlations.tsv", sep="\t", index=False
        )
    p_val = np.nan

    if null_path and null_path.exists():
        data = np.load(null_path)
        null_rs = data.get("null_r")
        if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subj):
            finite = null_rs[np.isfinite(null_rs)]
            if finite.size > 0:
                p_val = float(((np.abs(finite) >= abs(r_subj)).sum() + 1) / (finite.size + 1))

    try:
        from sklearn.metrics import r2_score

        r2_val = float(r2_score(y_true, y_pred))
    except Exception:
        r2_val = np.nan

    # Compute and export baseline predictions for sanity check
    baseline_metrics = export_baseline_predictions(y_true, groups_ordered, results_dir, task="regression")

    # Compute pooled (trial-level) metrics for secondary reporting
    pooled_r, _ = safe_pearsonr(y_true, y_pred)
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        pooled_mae = float(mean_absolute_error(y_true, y_pred))
        pooled_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        pooled_mae = np.nan
        pooled_rmse = np.nan

    subj_errors = compute_subject_level_errors(pred_df, config, ci_method=ci_method)
    if subj_errors.get("per_subject"):
        pd.DataFrame(subj_errors["per_subject"]).to_csv(
            results_dir / "per_subject_errors.tsv", sep="\t", index=False
        )

    # Structure metrics with subject-level as PRIMARY (statistical unit for LOSO)
    metrics = {
        "model": model_name,
        "data": {
            "target": target,
            "target_kind": "continuous",
            "detected_target": target_detection,
            "feature_families": feature_families,
            "feature_bands": feature_bands,
            "feature_segments": feature_segments,
            "feature_scopes": feature_scopes,
            "feature_stats": feature_stats,
            "feature_harmonization": feature_harmonization,
            "covariates": covariates,
        },
        "n_subjects": len(np.unique(groups_ordered)),
        "n_trials": len(y_true),
        "n_features": int(X.shape[1]),
        "subject_selection": subject_selection,
        "subject_level": {
            "r": r_subj,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_val,
            "n_subjects": len(_per_subj_r),
        },
        "subject_level_errors": {
            "mean_mae": subj_errors.get("mean_mae"),
            "ci_low_mae": subj_errors.get("ci_low_mae"),
            "ci_high_mae": subj_errors.get("ci_high_mae"),
            "mean_rmse": subj_errors.get("mean_rmse"),
            "ci_low_rmse": subj_errors.get("ci_low_rmse"),
            "ci_high_rmse": subj_errors.get("ci_high_rmse"),
        },
        "pooled_trials": {
            "r": float(pooled_r) if np.isfinite(pooled_r) else None,
            "r2": r2_val,
            "mae": pooled_mae,
            "rmse": pooled_rmse,
        },
        **baseline_metrics,
    }

    # Write reproducibility info
    write_reproducibility_info(results_dir, subjects, config, rng_seed)

    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    p_str = f", p={p_val:.4f}" if np.isfinite(p_val) else ""
    logger.info(
        "Regression results: r=%.3f [%.3f, %.3f]%s, R\u00b2=%.3f, RMSE=%.3f",
        r_subj, ci_low, ci_high, p_str, r2_val, pooled_rmse,
    )
    _maybe_generate_mode_plots(mode="regression", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    return results_dir


def run_within_subject_regression_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    model: str = "elasticnet",
    *,
    target: Optional[str] = None,
    feature_families: Optional[List[str]] = None,
    feature_harmonization: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Path:
    """Run within-subject (block-aware) regression on per-trial feature-table inputs."""
    if target is None:
        target = get_config_value(config, "machine_learning.targets.regression", None)

    X, y, groups, _feature_names, meta = load_active_matrix(
        subjects,
        task,
        deriv_root,
        config,
        logger,
        feature_families=feature_families,
        feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
        target=target,
        target_kind="continuous",
        covariates=covariates,
        feature_bands=feature_bands,
        feature_segments=feature_segments,
        feature_scopes=feature_scopes,
        feature_stats=feature_stats,
    )
    target_detection = _warn_or_raise_if_binary_like_regression_target(
        y,
        target,
        logger,
        config,
        context="Within-subject regression",
    )

    if meta is None or not hasattr(meta, "columns"):
        raise ValueError("Within-subject machine learning requires trial metadata with block/run labels.")

    if "block" not in meta.columns:
        raise ValueError(
            "Within-subject machine learning requires block/run labels to prevent temporal data leakage. "
            "Ensure your events.tsv contains one of: block, run_id, run, session."
        )

    blocks_all = pd.to_numeric(meta["block"], errors="coerce").to_numpy()

    finite_mask = np.isfinite(y)
    if not np.all(finite_mask):
        X = X[finite_mask]
        y = y[finite_mask]
        groups = groups[finite_mask]
        meta = meta.loc[finite_mask].reset_index(drop=True)
        blocks_all = blocks_all[finite_mask]

    finite_blocks = np.isfinite(blocks_all)
    if not np.all(finite_blocks):
        dropped = int((~finite_blocks).sum())
        logger.warning(f"Dropping {dropped} trials with missing block labels for within-subject machine learning.")
        X = X[finite_blocks]
        y = y[finite_blocks]
        groups = groups[finite_blocks]
        meta = meta.loc[finite_blocks].reset_index(drop=True)
        blocks_all = blocks_all[finite_blocks]

    n_subjects = len(np.unique(groups))
    logger.info(
        "Within-subject regression: model=%s, %d features \u00d7 %d trials, %d subjects, target='%s'",
        model or "elasticnet", X.shape[1], X.shape[0], n_subjects, target,
    )

    results_dir = results_root / "within_subject_regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    
    model_name = model if model else "elasticnet"
    harmonization_mode = (
        feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection"))
    )

    outer_cv_splits = int(
        get_config_value(
            config,
            "machine_learning.cv.outer_splits",
            get_config_value(config, "machine_learning.cv.default_n_splits", 5),
        )
    )
    folds = create_within_subject_folds(
        groups=groups,
        blocks_all=blocks_all,
        inner_cv_splits=inner_splits,
        seed=rng_seed,
        outer_cv_splits=outer_cv_splits,
        config=config,
        epochs=None,
        apply_hygiene=False,
    )
    if not folds:
        raise ValueError(
            "No within-subject folds could be created. "
            "Ensure each selected subject has at least 2 unique block/run labels."
        )

    fold_records = []
    for fold_counter, train_idx, test_idx, subject_id, _fold_params in folds:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        blocks_train = blocks_all[train_idx] if blocks_all is not None else None
        X_train, X_test, _ = apply_fold_feature_harmonization(
            X_train,
            X_test,
            groups_train,
            harmonization_mode,
        )

        if model == "ridge":
            pipe = create_ridge_pipeline(seed=rng_seed + int(fold_counter), config=config)
            param_grid = build_ridge_param_grid(config)
        elif model == "rf":
            pipe = create_rf_pipeline(seed=rng_seed + int(fold_counter), config=config)
            param_grid = build_rf_param_grid(config)
        else:
            pipe = create_elasticnet_pipeline(seed=rng_seed + int(fold_counter), config=config)
            param_grid = build_elasticnet_param_grid(config)
        
        best_estimator = _fit_within_subject_fold(
            pipe=pipe,
            X_train=X_train,
            y_train=y_train,
            blocks_train=blocks_train,
            fold=int(fold_counter),
            subject_id=str(subject_id),
            random_state=rng_seed + int(fold_counter),
            n_jobs=1,
            logger=logger,
            inner_splits=inner_splits,
            param_grid=param_grid,
        )
        y_pred = best_estimator.predict(X_test)
        fold_mean_baseline = float(np.mean(y_train)) if len(y_train) > 0 else np.nan
        y_pred_baseline = np.full(len(y_test), fold_mean_baseline, dtype=float)

        fold_records.append(
            {
                "fold": int(fold_counter),
                "y_true": y_test,
                "y_pred": y_pred,
                "y_pred_baseline": y_pred_baseline,
                "subject_id": [str(subject_id)] * len(y_test),
                "test_idx": np.asarray(test_idx, dtype=int),
            }
        )

    fold_records = sorted(fold_records, key=lambda r: r["fold"])
    y_true_all = np.concatenate([np.asarray(r["y_true"]) for r in fold_records])
    y_pred_all = np.concatenate([np.asarray(r["y_pred"]) for r in fold_records])
    y_pred_baseline_all = np.concatenate(
        [np.asarray(r["y_pred_baseline"], dtype=float) for r in fold_records]
    )
    groups_ordered: List[str] = []
    test_indices: List[int] = []
    fold_ids: List[int] = []
    for rec in fold_records:
        n = len(rec["y_true"])
        groups_ordered.extend(rec["subject_id"])
        test_indices.extend(rec["test_idx"].tolist())
        fold_ids.extend([rec["fold"]] * n)
    groups_ordered_arr = np.asarray(groups_ordered, dtype=object)
    subject_selection = export_subject_selection_report(
        results_dir,
        subjects,
        groups_ordered_arr,
        meta,
        config,
    )

    pred_path = results_dir / "cv_predictions.tsv"
    pred_df = export_predictions(
        y_true_all,
        y_pred_all,
        groups_ordered,
        test_indices,
        fold_ids,
        model_name,
        meta.reset_index(drop=True),
        pred_path,
    )
    export_indices(
        groups_ordered,
        test_indices,
        fold_ids,
        meta.reset_index(drop=True),
        results_dir / "cv_indices.tsv",
    )

    ci_method = str(get_config_value(config, "machine_learning.evaluation.ci_method", "bootstrap"))
    r_subj, _per_subj_r, ci_low, ci_high = compute_subject_level_r(pred_df, config, ci_method=ci_method)
    if _per_subj_r:
        pd.DataFrame(_per_subj_r, columns=["subject_id", "pearson_r"]).to_csv(
            results_dir / "per_subject_correlations.tsv", sep="\t", index=False
        )
    try:
        from sklearn.metrics import r2_score

        r2_val = float(r2_score(y_true_all, y_pred_all))
    except Exception:
        r2_val = np.nan

    p_value = np.nan
    null_rs: List[float] = []
    n_perm_completed = 0
    
    if n_perm > 0:
        logger.info(f"Running {n_perm} block-aware permutations for within-subject inference...")
        rng = np.random.default_rng(rng_seed)
        n_effective = 0
        n_fallback_permutations = 0
        perm_scheme = _resolve_permutation_scheme(config)
        min_shuffle_fraction = float(
            get_config_value(config, "machine_learning.cv.min_label_shuffle_fraction", 0.01)
        )
        min_perm_fold_completion = float(
            get_config_value(config, "machine_learning.cv.min_valid_permutation_fold_fraction", 1.0)
        )
        
        for perm_idx in range(n_perm):
            y_perm, effective, _changed_fraction, used_scheme = _generate_effective_permutation(
                y,
                groups,
                blocks=blocks_all,
                rng=rng,
                requested_scheme=perm_scheme,
                min_changed_fraction=min_shuffle_fraction,
            )
            if not effective:
                continue
            n_effective += 1
            if used_scheme != perm_scheme:
                n_fallback_permutations += 1
            
            perm_fold_records: List[Dict[str, Any]] = []

            for fold_counter, train_idx, test_idx, subject_id, _ in folds:
                X_train_p = X[train_idx]
                X_test_p = X[test_idx]
                y_train_p = y_perm[train_idx]
                y_test_p = y_perm[test_idx]
                groups_train_p = groups[train_idx]

                blocks_train_p = blocks_all[train_idx] if blocks_all is not None else None
                X_train_p, X_test_p, _ = apply_fold_feature_harmonization(
                    X_train_p,
                    X_test_p,
                    groups_train_p,
                    harmonization_mode,
                )

                if model == "ridge":
                    pipe_p = create_ridge_pipeline(seed=rng_seed + perm_idx + fold_counter, config=config)
                    param_grid_p = build_ridge_param_grid(config)
                elif model == "rf":
                    pipe_p = create_rf_pipeline(seed=rng_seed + perm_idx + fold_counter, config=config)
                    param_grid_p = build_rf_param_grid(config)
                else:
                    pipe_p = create_elasticnet_pipeline(seed=rng_seed + perm_idx + fold_counter, config=config)
                    param_grid_p = build_elasticnet_param_grid(config)

                try:
                    best_estimator_p = _fit_within_subject_fold(
                        pipe=pipe_p,
                        X_train=X_train_p,
                        y_train=y_train_p,
                        blocks_train=blocks_train_p,
                        fold=int(fold_counter),
                        subject_id=str(subject_id),
                        random_state=rng_seed + perm_idx + int(fold_counter),
                        n_jobs=1,
                        logger=logger,
                        inner_splits=inner_splits,
                        param_grid=param_grid_p,
                    )
                    y_pred_p = best_estimator_p.predict(X_test_p)
                    perm_fold_records.append(
                        {
                            "y_true": np.asarray(y_test_p, dtype=float),
                            "y_pred": np.asarray(y_pred_p, dtype=float),
                            "subject_id": [str(subject_id)] * len(y_test_p),
                        }
                    )
                except Exception:
                    continue

            fold_completion = float(len(perm_fold_records) / max(len(folds), 1))
            if fold_completion < min_perm_fold_completion:
                continue

            if perm_fold_records:
                y_true_perm = np.concatenate([r["y_true"] for r in perm_fold_records])
                y_pred_perm = np.concatenate([r["y_pred"] for r in perm_fold_records])
                subject_perm = np.concatenate([np.asarray(r["subject_id"], dtype=object) for r in perm_fold_records])
                perm_df = pd.DataFrame(
                    {"y_true": y_true_perm, "y_pred": y_pred_perm, "subject_id": subject_perm}
                )
                r_perm, _, _, _ = compute_subject_level_r(perm_df, config, ci_method=ci_method)
                if np.isfinite(r_perm):
                    null_rs.append(float(r_perm))
            
            if (perm_idx + 1) % 10 == 0:
                logger.info(f"Permutation {perm_idx + 1}/{n_perm}")

        if n_fallback_permutations > 0:
            logger.info(
                "Within-subject regression permutations: %d/%d effective shuffles required fallback to within-subject scheme.",
                int(n_fallback_permutations),
                int(n_effective),
            )

        if n_effective == 0:
            raise RuntimeError(
                "No effective permutations could be generated for within-subject regression. "
                "Try machine_learning.cv.permutation_scheme='within_subject' and/or lower "
                "machine_learning.cv.min_label_shuffle_fraction."
            )
        
        n_perm_completed = int(len(null_rs))
        completion_rate = (n_perm_completed / int(n_perm)) if int(n_perm) > 0 else 0.0
        min_completion = float(
            get_config_value(config, "machine_learning.cv.min_valid_permutation_fraction", 0.5)
        )
        if int(n_perm) > 0 and completion_rate < min_completion:
            raise RuntimeError(
                f"Insufficient valid within-subject regression permutations ({n_perm_completed}/{int(n_perm)}, "
                f"rate={completion_rate:.3f} < required {min_completion:.3f})"
            )

        if n_perm_completed > 0 and np.isfinite(r_subj):
            null_rs_arr = np.asarray(null_rs, dtype=float)
            p_value = float(((np.abs(null_rs_arr) >= np.abs(r_subj)).sum() + 1) / (len(null_rs_arr) + 1))
            np.savez(results_dir / "within_subject_null.npz", null_r=null_rs_arr, empirical_r=r_subj)
            logger.info(f"Within-subject permutation p-value: {p_value:.4f} (n_perm={len(null_rs_arr)})")

    # Compute pooled r for secondary reporting
    pooled_r, _ = safe_pearsonr(y_true_all, y_pred_all)
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        pooled_mae = float(mean_absolute_error(y_true_all, y_pred_all))
        pooled_rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    except Exception:
        pooled_mae = np.nan
        pooled_rmse = np.nan

    subj_errors = compute_subject_level_errors(pred_df, config, ci_method=ci_method)
    if subj_errors.get("per_subject"):
        pd.DataFrame(subj_errors["per_subject"]).to_csv(
            results_dir / "per_subject_errors.tsv", sep="\t", index=False
        )
    
    # Structure metrics with subject-level as PRIMARY
    metrics = {
        "model": model_name,
        "cv_scope": "subject",
        "data": {
            "target": target,
            "target_kind": "continuous",
            "detected_target": target_detection,
            "feature_families": feature_families,
            "feature_bands": feature_bands,
            "feature_segments": feature_segments,
            "feature_scopes": feature_scopes,
            "feature_stats": feature_stats,
            "feature_harmonization": feature_harmonization,
            "covariates": covariates,
        },
        "n_subjects": len(np.unique(groups_ordered_arr)),
        "n_trials": len(y_true_all),
        "n_features": int(X.shape[1]),
        "subject_selection": subject_selection,
        "subject_level": {
            "r": r_subj,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
        },
        "subject_level_errors": {
            "mean_mae": subj_errors.get("mean_mae"),
            "ci_low_mae": subj_errors.get("ci_low_mae"),
            "ci_high_mae": subj_errors.get("ci_high_mae"),
            "mean_rmse": subj_errors.get("mean_rmse"),
            "ci_low_rmse": subj_errors.get("ci_low_rmse"),
            "ci_high_rmse": subj_errors.get("ci_high_rmse"),
        },
        "pooled_trials": {
            "r": float(pooled_r) if np.isfinite(pooled_r) else None,
            "r2": r2_val,
            "mae": pooled_mae,
            "rmse": pooled_rmse,
        },
        "n_perm": n_perm if n_perm > 0 else 0,
    }
    if int(n_perm) > 0:
        metrics["n_perm_requested"] = int(n_perm)
        metrics["n_perm_completed"] = int(n_perm_completed)
    baseline_df = pd.DataFrame(
        {
            "subject_id": groups_ordered_arr,
            "y_true": y_true_all,
            "y_pred_baseline": y_pred_baseline_all,
        }
    )
    baseline_df.to_csv(results_dir / "baseline_predictions.tsv", sep="\t", index=False)
    try:
        from sklearn.metrics import mean_absolute_error, r2_score

        baseline_metrics = {
            "baseline_r2": float(r2_score(y_true_all, y_pred_baseline_all)),
            "baseline_mae": float(mean_absolute_error(y_true_all, y_pred_baseline_all)),
            "baseline_method": "within_subject_fold_mean",
        }
    except Exception:
        baseline_metrics = {
            "baseline_r2": np.nan,
            "baseline_mae": np.nan,
            "baseline_method": "within_subject_fold_mean",
        }
    metrics.update(baseline_metrics)
    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    p_str = f", p={p_value:.4f}" if np.isfinite(p_value) else ""
    logger.info(
        "Within-subject regression results: r=%.3f [%.3f, %.3f]%s, R\u00b2=%.3f, RMSE=%.3f",
        r_subj, ci_low, ci_high, p_str, r2_val, pooled_rmse,
    )
    _maybe_generate_mode_plots(mode="regression", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    return results_dir


def run_time_generalization(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    *,
    target: Optional[str] = None,
) -> Path:
    """Run time-generalization machine learning analysis."""
    import time as _time

    logger.info(
        "Time generalization: %d subjects, %d permutations",
        len(subjects), n_perm,
    )

    results_dir = results_root / "time_generalization"
    ensure_dir(results_dir)

    t0 = _time.perf_counter()
    try:
        tg_r, tg_r2, window_centers = time_generalization_regression(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            results_dir=results_dir,
            config_dict=config,
            n_perm=n_perm,
            seed=rng_seed,
            target=target,
        )
    except Exception as exc:
        logger.warning("Time-generalization failed after %.1fs: %s", _time.perf_counter() - t0, exc)
        raise RuntimeError("Time-generalization stage failed.") from exc

    n_windows = len(window_centers) if window_centers is not None else 0
    n_tested_cells = int(np.sum(np.isfinite(tg_r))) if tg_r is not None and len(tg_r) > 0 else 0
    if n_windows == 0 or n_tested_cells == 0:
        raise RuntimeError(
            "Time-generalization produced no valid outputs "
            f"(n_windows={n_windows}, n_tested_cells={n_tested_cells})."
        )
    logger.info(
        "Time generalization complete: %d windows, peak r=%.3f (%.1fs)",
        n_windows,
        float(np.nanmax(tg_r)) if tg_r is not None and len(tg_r) > 0 else float("nan"),
        _time.perf_counter() - t0,
    )
    _maybe_generate_mode_plots(mode="timegen", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    return results_dir


###################################################################
# Classification Pipeline Runner
###################################################################


def run_classification_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    *,
    classification_model: Optional[str] = None,
    target: Optional[str] = None,
    binary_threshold: Optional[float] = None,
    feature_families: Optional[List[str]] = None,
    feature_harmonization: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Path:
    """Run LOSO classification machine learning for pain vs no-pain.
    
    Uses nested CV with hyperparameter tuning in inner loop.
    Reports AUC, balanced accuracy, and calibrated metrics.
    """
    from eeg_pipeline.analysis.machine_learning.classification import (
        nested_loso_classification,
    )
    from eeg_pipeline.analysis.machine_learning.cnn import nested_loso_cnn_classification
    
    if target is None:
        target = str(get_config_value(config, "machine_learning.targets.classification", "pain_binary"))
    if binary_threshold is None:
        binary_threshold = get_config_value(config, "machine_learning.targets.binary_threshold", None)

    if classification_model is not None and str(classification_model).strip():
        model_type = str(classification_model).strip().lower()
    else:
        model_type = str(get_config_value(config, "machine_learning.classification.model", "svm")).strip().lower()
    harmonization_mode = (
        feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection"))
    )

    if model_type == "cnn":
        X, y_binary, groups, feature_names, meta = load_epoch_tensor_matrix(
            subjects,
            task,
            deriv_root,
            config,
            logger,
            target=target,
            target_kind="binary",
            binary_threshold=binary_threshold,
        )
    else:
        X, y_binary, groups, feature_names, meta = load_active_matrix(
            subjects,
            task,
            deriv_root,
            config,
            logger,
            feature_families=feature_families,
            feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
            target=target,
            target_kind="binary",
            binary_threshold=binary_threshold,
            covariates=covariates,
            feature_bands=feature_bands,
            feature_segments=feature_segments,
            feature_scopes=feature_scopes,
            feature_stats=feature_stats,
        )
    blocks = None
    if meta is not None and hasattr(meta, "columns") and "block" in meta.columns:
        blocks = pd.to_numeric(meta["block"], errors="coerce").to_numpy()

    n_subjects = len(np.unique(groups))
    n_features_desc = (
        f"{X.shape[1]}ch \u00d7 {X.shape[2]}t" if X.ndim == 3
        else f"{X.shape[1]} features"
    )
    class_balance = float(np.mean(y_binary)) if len(y_binary) else float("nan")
    logger.info(
        "Classification: model=%s, %s, %d trials, %d subjects, target='%s', balance=%.2f",
        model_type, n_features_desc, len(y_binary), n_subjects, target, class_balance,
    )

    results_dir = results_root / "classification"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    subject_selection = export_subject_selection_report(results_dir, subjects, groups, meta, config)

    if model_type == "cnn":
        result, best_params_df = nested_loso_cnn_classification(
            X=X,
            y=y_binary,
            groups=groups,
            seed=rng_seed,
            config=config,
            logger=logger,
        )
    else:
        result, best_params_df = nested_loso_classification(
            X=X,
            y=y_binary,
            groups=groups,
            model=model_type,
            inner_splits=inner_splits,
            seed=rng_seed,
            config=config,
            logger=logger,
            harmonization_mode=feature_harmonization
            or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")),
        )
    failed_fold_count = int(getattr(result, "failed_fold_count", 0) or 0)
    n_folds_total = int(getattr(result, "n_folds_total", len(np.unique(groups))) or len(np.unique(groups)))
    failed_fold_fraction = float(failed_fold_count / max(n_folds_total, 1))
    max_failed_fold_fraction = float(
        get_config_value(config, "machine_learning.classification.max_failed_fold_fraction", 0.25)
    )
    if failed_fold_fraction > max_failed_fold_fraction:
        raise RuntimeError(
            "Classification fold failure fraction exceeded validity threshold: "
            f"failed={failed_fold_count}/{n_folds_total} ({failed_fold_fraction:.3f}) > "
            f"machine_learning.classification.max_failed_fold_fraction={max_failed_fold_fraction:.3f}."
        )

    # Export predictions/indices with trial-level provenance.
    groups_for_predictions = np.asarray(
        result.groups if result.groups is not None else groups,
        dtype=object,
    )
    if len(groups_for_predictions) != len(result.y_true):
        groups_for_predictions = np.asarray(groups, dtype=object)

    raw_test_indices = getattr(result, "test_indices", None)
    if raw_test_indices is None:
        test_indices_arr = np.arange(len(result.y_true), dtype=int)
    else:
        test_indices_arr = np.asarray(raw_test_indices, dtype=int)
        if (
            len(test_indices_arr) != len(result.y_true)
            or np.any(test_indices_arr < 0)
            or np.any(test_indices_arr >= len(meta))
        ):
            test_indices_arr = np.arange(len(result.y_true), dtype=int)

    raw_fold_ids = getattr(result, "fold_ids", None)
    if raw_fold_ids is None:
        fold_ids_arr = np.ones(len(result.y_true), dtype=int)
    else:
        fold_ids_arr = np.asarray(raw_fold_ids, dtype=int)
        if len(fold_ids_arr) != len(result.y_true):
            fold_ids_arr = np.ones(len(result.y_true), dtype=int)
    if np.any(fold_ids_arr <= 0):
        fold_ids_arr = fold_ids_arr.copy()
        fold_ids_arr[fold_ids_arr <= 0] = 1

    pred_path = results_dir / "loso_predictions.tsv"
    pred_df = export_predictions(
        np.asarray(result.y_true),
        np.asarray(result.y_pred),
        groups_for_predictions.tolist(),
        test_indices_arr.tolist(),
        fold_ids_arr.tolist(),
        model_type,
        meta.reset_index(drop=True),
        pred_path,
    )
    if result.y_prob is not None and len(result.y_prob) == len(pred_df):
        pred_df["y_prob"] = np.asarray(result.y_prob, dtype=float)
    else:
        pred_df["y_prob"] = np.full(len(pred_df), np.nan, dtype=float)
    write_tsv(pred_df, pred_path)
    write_parquet(pred_df, pred_path.with_suffix(".parquet"))
    export_indices(
        groups_for_predictions.tolist(),
        test_indices_arr.tolist(),
        fold_ids_arr.tolist(),
        meta.reset_index(drop=True),
        results_dir / "loso_indices.tsv",
        add_heldout_subject_id=True,
    )

    # Export best params
    if not best_params_df.empty:
        best_params_df.to_csv(results_dir / f"best_params_{model_type}.tsv", sep="\t", index=False)

    # Export per-subject metrics (LOSO)
    if getattr(result, "per_subject_metrics", None):
        rows = []
        for subj, rec in result.per_subject_metrics.items():
            row = {"subject_id": subj}
            row.update(rec)
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(results_dir / "per_subject_metrics.tsv", sep="\t", index=False)
    auc_subject_mean = _subject_mean_metric(result.per_subject_metrics, "auc")
    balanced_accuracy_subject_mean = _subject_mean_metric(result.per_subject_metrics, "balanced_accuracy")
    accuracy_subject_mean = _subject_mean_metric(result.per_subject_metrics, "accuracy")
    precision_subject_mean = _subject_mean_metric(result.per_subject_metrics, "precision")
    recall_subject_mean = _subject_mean_metric(result.per_subject_metrics, "recall")
    f1_subject_mean = _subject_mean_metric(result.per_subject_metrics, "f1")
    specificity_subject_mean = _subject_mean_metric(result.per_subject_metrics, "specificity")
    average_precision_subject_mean = _subject_mean_metric(result.per_subject_metrics, "average_precision")
    min_subjects_auc = int(
        get_config_value(config, "machine_learning.classification.min_subjects_with_auc_for_inference", 2)
    )
    n_subjects_with_auc = _count_finite_subject_metric(result.per_subject_metrics, "auc")
    n_subjects_total = int(len(np.unique(groups_for_predictions)))
    auc_inference_valid = n_subjects_with_auc >= min_subjects_auc
    if not auc_inference_valid:
        logger.warning(
            "Classification AUC inference disabled: only %d/%d subjects had evaluable AUC (< %d required).",
            int(n_subjects_with_auc),
            int(n_subjects_total),
            int(min_subjects_auc),
        )
    auc_for_inference = (
        auc_subject_mean if (np.isfinite(auc_subject_mean) and auc_inference_valid) else np.nan
    )
    n_boot = int(get_config_value(config, "machine_learning.evaluation.bootstrap_iterations", 1000))
    ci_rng = np.random.default_rng(int(rng_seed) + 301)
    auc_vals = _subject_metric_values(result.per_subject_metrics, "auc")
    bal_vals = _subject_metric_values(result.per_subject_metrics, "balanced_accuracy")
    acc_vals = _subject_metric_values(result.per_subject_metrics, "accuracy")
    precision_vals = _subject_metric_values(result.per_subject_metrics, "precision")
    recall_vals = _subject_metric_values(result.per_subject_metrics, "recall")
    f1_vals = _subject_metric_values(result.per_subject_metrics, "f1")
    specificity_vals = _subject_metric_values(result.per_subject_metrics, "specificity")
    average_precision_vals = _subject_metric_values(result.per_subject_metrics, "average_precision")
    auc_ci_low, auc_ci_high = (
        _bootstrap_mean_ci(auc_vals, rng=ci_rng, n_boot=n_boot)
        if auc_inference_valid
        else (np.nan, np.nan)
    )
    bal_ci_low, bal_ci_high = _bootstrap_mean_ci(bal_vals, rng=ci_rng, n_boot=n_boot)
    acc_ci_low, acc_ci_high = _bootstrap_mean_ci(acc_vals, rng=ci_rng, n_boot=n_boot)
    precision_ci_low, precision_ci_high = _bootstrap_mean_ci(precision_vals, rng=ci_rng, n_boot=n_boot)
    recall_ci_low, recall_ci_high = _bootstrap_mean_ci(recall_vals, rng=ci_rng, n_boot=n_boot)
    f1_ci_low, f1_ci_high = _bootstrap_mean_ci(f1_vals, rng=ci_rng, n_boot=n_boot)
    specificity_ci_low, specificity_ci_high = _bootstrap_mean_ci(
        specificity_vals, rng=ci_rng, n_boot=n_boot
    )
    average_precision_ci_low, average_precision_ci_high = _bootstrap_mean_ci(
        average_precision_vals, rng=ci_rng, n_boot=n_boot
    )

    # Compute calibration metrics (Brier score, reliability) when probabilities are available
    calibration_data = {}
    brier = np.nan
    ece = np.nan
    if result.y_prob is not None:
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import calibration_curve
        probs = np.asarray(result.y_prob, dtype=float)
        y_true = np.asarray(result.y_true, dtype=float)
        finite_prob_mask = np.isfinite(probs) & np.isfinite(y_true)
        probs_finite = probs[finite_prob_mask]
        y_true_finite = y_true[finite_prob_mask]

        if len(probs_finite) >= 2 and len(np.unique(y_true_finite)) == 2:
            try:
                brier = float(brier_score_loss(y_true_finite, probs_finite))
            except Exception:
                brier = np.nan

            try:
                prob_true, prob_pred = calibration_curve(
                    y_true_finite, probs_finite, n_bins=10, strategy="uniform"
                )
                calibration_data = {
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                    "n_bins": 10,
                }
                # Expected calibration error (ECE) on finite probabilities only.
                edges = np.linspace(0.0, 1.0, 11)
                ece_acc = 0.0
                n = len(probs_finite)
                for i in range(10):
                    lo, hi = edges[i], edges[i + 1]
                    if i < 9:
                        mask = (probs_finite >= lo) & (probs_finite < hi)
                    else:
                        mask = (probs_finite >= lo) & (probs_finite <= hi)
                    if not np.any(mask):
                        continue
                    w = float(np.sum(mask) / n)
                    frac_pos = float(np.mean(y_true_finite[mask]))
                    avg_p = float(np.mean(probs_finite[mask]))
                    ece_acc += w * abs(frac_pos - avg_p)
                ece = float(ece_acc) if n > 0 else np.nan
            except Exception:
                calibration_data = {}
                ece = np.nan
    
    # Compute and save metrics
    metrics = {
        "data": {
            "target": target,
            "target_kind": "binary",
            "binary_threshold": binary_threshold,
            "feature_families": feature_families,
            "feature_bands": feature_bands,
            "feature_segments": feature_segments,
            "feature_scopes": feature_scopes,
            "feature_stats": feature_stats,
            "feature_harmonization": feature_harmonization,
            "covariates": covariates,
        },
        "subject_selection": subject_selection,
        "auc": float(auc_for_inference) if np.isfinite(auc_for_inference) else np.nan,
        "balanced_accuracy": (
            float(balanced_accuracy_subject_mean)
            if np.isfinite(balanced_accuracy_subject_mean)
            else np.nan
        ),
        "accuracy": (
            float(accuracy_subject_mean)
            if np.isfinite(accuracy_subject_mean)
            else np.nan
        ),
        "average_precision": (
            float(average_precision_subject_mean)
            if np.isfinite(average_precision_subject_mean)
            else np.nan
        ),
        "precision": (
            float(precision_subject_mean)
            if np.isfinite(precision_subject_mean)
            else np.nan
        ),
        "recall": (
            float(recall_subject_mean)
            if np.isfinite(recall_subject_mean)
            else np.nan
        ),
        "f1": float(f1_subject_mean) if np.isfinite(f1_subject_mean) else np.nan,
        "specificity": (
            float(specificity_subject_mean)
            if np.isfinite(specificity_subject_mean)
            else np.nan
        ),
        "brier_score": brier,
        "expected_calibration_error": ece,
        "model": model_type,
        "n_subjects": int(len(np.unique(groups_for_predictions))),
        "n_samples": int(len(y_binary)),
        "n_features": int(X.shape[1] * X.shape[2]) if X.ndim == 3 else int(X.shape[1]),
        "n_channels": int(X.shape[1]) if X.ndim == 3 else None,
        "n_timepoints": int(X.shape[2]) if X.ndim == 3 else None,
        "class_balance": float(np.mean(y_binary)) if len(y_binary) else np.nan,
        "failed_fold_count": int(failed_fold_count),
        "n_folds_total": int(n_folds_total),
        "failed_fold_fraction": float(failed_fold_fraction),
        "max_failed_fold_fraction": float(max_failed_fold_fraction),
        "subject_level": {
            "auc_mean": float(auc_subject_mean) if np.isfinite(auc_subject_mean) else np.nan,
            "auc_ci_low": auc_ci_low,
            "auc_ci_high": auc_ci_high,
            "balanced_accuracy_mean": (
                float(balanced_accuracy_subject_mean)
                if np.isfinite(balanced_accuracy_subject_mean)
                else np.nan
            ),
            "balanced_accuracy_ci_low": bal_ci_low,
            "balanced_accuracy_ci_high": bal_ci_high,
            "accuracy_mean": float(accuracy_subject_mean) if np.isfinite(accuracy_subject_mean) else np.nan,
            "accuracy_ci_low": acc_ci_low,
            "accuracy_ci_high": acc_ci_high,
            "average_precision_mean": (
                float(average_precision_subject_mean)
                if np.isfinite(average_precision_subject_mean)
                else np.nan
            ),
            "average_precision_ci_low": average_precision_ci_low,
            "average_precision_ci_high": average_precision_ci_high,
            "precision_mean": float(precision_subject_mean) if np.isfinite(precision_subject_mean) else np.nan,
            "precision_ci_low": precision_ci_low,
            "precision_ci_high": precision_ci_high,
            "recall_mean": float(recall_subject_mean) if np.isfinite(recall_subject_mean) else np.nan,
            "recall_ci_low": recall_ci_low,
            "recall_ci_high": recall_ci_high,
            "f1_mean": float(f1_subject_mean) if np.isfinite(f1_subject_mean) else np.nan,
            "f1_ci_low": f1_ci_low,
            "f1_ci_high": f1_ci_high,
            "specificity_mean": (
                float(specificity_subject_mean)
                if np.isfinite(specificity_subject_mean)
                else np.nan
            ),
            "specificity_ci_low": specificity_ci_low,
            "specificity_ci_high": specificity_ci_high,
            "n_subjects_with_auc": int(n_subjects_with_auc),
            "n_subjects_without_auc": int(max(0, n_subjects_total - n_subjects_with_auc)),
            "min_subjects_with_auc_for_inference": int(min_subjects_auc),
            "auc_inference_valid": bool(auc_inference_valid),
        },
        "pooled_trials": {
            "auc": float(result.auc) if np.isfinite(result.auc) else np.nan,
            "balanced_accuracy": float(result.balanced_accuracy)
            if np.isfinite(result.balanced_accuracy)
            else np.nan,
            "accuracy": float(result.accuracy) if np.isfinite(result.accuracy) else np.nan,
            "average_precision": float(result.average_precision)
            if np.isfinite(result.average_precision)
            else np.nan,
            "precision": float(result.precision) if np.isfinite(result.precision) else np.nan,
            "recall": float(result.recall) if np.isfinite(result.recall) else np.nan,
            "f1": float(result.f1) if np.isfinite(result.f1) else np.nan,
            "specificity": float(result.specificity) if np.isfinite(result.specificity) else np.nan,
        },
    }
    if not auc_inference_valid:
        metrics["auc_reporting_note"] = (
            "Subject-level AUC inference is invalid for this run; pooled-trial AUC is reported for diagnostics only."
        )
    
    # Save calibration data separately
    if calibration_data:
        with open(results_dir / "calibration_data.json", "w") as f:
            json.dump(calibration_data, f, indent=2)

    # Add permutation p-value if available
    if n_perm > 0:
        logger.info(f"Running {n_perm} permutations for classification...")
        null_aucs = _run_classification_permutations(
            X,
            y_binary,
            groups,
            blocks,
            model_type,
            inner_splits,
            rng_seed,
            n_perm,
            config,
            logger,
            harmonization_mode=feature_harmonization
            or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")),
        )
        if null_aucs is not None and len(null_aucs) > 0 and np.isfinite(auc_for_inference):
            metrics["n_perm_requested"] = int(n_perm)
            metrics["n_perm_completed"] = int(len(null_aucs))
            p_val = float(((null_aucs >= auc_for_inference).sum() + 1) / (len(null_aucs) + 1))
            metrics["p_value"] = p_val
            np.savez(
                results_dir / f"loso_null_{model_type}.npz",
                null_auc_subject_mean=null_aucs,
                empirical_auc_subject_mean=auc_for_inference,
            )

    # Write reproducibility info
    write_reproducibility_info(results_dir, subjects, config, rng_seed)

    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    p_info = ""
    if "p_value" in metrics:
        p_info = f", p={metrics['p_value']:.4f}"
    if not auc_inference_valid:
        logger.warning(
            "Subject-level AUC inference is invalid; pooled-trial AUC is reported for diagnostics only."
        )
    logger.info(
        "Classification results: AUC=%.3f, balanced_acc=%.3f, F1=%.3f, Brier=%.3f%s",
        float(auc_for_inference) if np.isfinite(auc_for_inference) else result.auc,
        float(balanced_accuracy_subject_mean) if np.isfinite(balanced_accuracy_subject_mean) else result.balanced_accuracy,
        float(f1_subject_mean) if np.isfinite(f1_subject_mean) else result.f1, brier, p_info,
    )
    _maybe_generate_mode_plots(mode="classify", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    return results_dir


def run_within_subject_classification_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    *,
    classification_model: Optional[str] = None,
    target: Optional[str] = None,
    binary_threshold: Optional[float] = None,
    feature_families: Optional[List[str]] = None,
    feature_harmonization: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Path:
    """Run within-subject (block-aware) classification on per-trial feature-table inputs.

    Uses block/run labels to prevent temporal leakage. Statistical unit is subject.
    """
    from eeg_pipeline.analysis.machine_learning.classification import (
        ClassificationResult,
        build_logistic_param_grid,
        build_rf_classification_param_grid,
        build_svm_param_grid,
        create_logistic_pipeline,
        create_rf_classification_pipeline,
        create_svm_pipeline,
    )
    from eeg_pipeline.analysis.machine_learning.cnn import fit_predict_cnn_binary_classifier
    from sklearn.model_selection import StratifiedGroupKFold

    if target is None:
        target = str(get_config_value(config, "machine_learning.targets.classification", "pain_binary"))
    if binary_threshold is None:
        binary_threshold = get_config_value(config, "machine_learning.targets.binary_threshold", None)

    if classification_model is not None and str(classification_model).strip():
        model_type = str(classification_model).strip().lower()
    else:
        model_type = str(get_config_value(config, "machine_learning.classification.model", "svm")).strip().lower()
    harmonization_mode = (
        feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection"))
    )

    if model_type == "cnn":
        X, y, groups, _feature_names, meta = load_epoch_tensor_matrix(
            subjects,
            task,
            deriv_root,
            config,
            logger,
            target=target,
            target_kind="binary",
            binary_threshold=binary_threshold,
        )
    else:
        X, y, groups, _feature_names, meta = load_active_matrix(
            subjects,
            task,
            deriv_root,
            config,
            logger,
            feature_families=feature_families,
            feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
            target=target,
            target_kind="binary",
            binary_threshold=binary_threshold,
            covariates=covariates,
            feature_bands=feature_bands,
            feature_segments=feature_segments,
            feature_scopes=feature_scopes,
            feature_stats=feature_stats,
        )

    if meta is None or not hasattr(meta, "columns") or "block" not in meta.columns:
        raise ValueError(
            "Within-subject classification requires block/run labels to prevent temporal leakage. "
            "Ensure your events.tsv contains one of: block, run_id, run, session."
        )

    blocks_all = pd.to_numeric(meta["block"], errors="coerce").to_numpy()
    finite_blocks = np.isfinite(blocks_all)
    if not np.all(finite_blocks):
        dropped = int((~finite_blocks).sum())
        logger.warning(
            "Dropping %d trials with missing block labels for within-subject classification.",
            dropped,
        )
        X = X[finite_blocks]
        y = y[finite_blocks]
        groups = groups[finite_blocks]
        meta = meta.loc[finite_blocks].reset_index(drop=True)
        blocks_all = blocks_all[finite_blocks]

    n_subjects = len(np.unique(groups))
    n_features_desc = (
        f"{X.shape[1]}ch \u00d7 {X.shape[2]}t" if X.ndim == 3
        else f"{X.shape[1]} features"
    )
    class_balance = float(np.mean(y)) if len(y) else float("nan")
    logger.info(
        "Within-subject classification: model=%s, %s, %d trials, %d subjects, target='%s', balance=%.2f",
        model_type, n_features_desc, len(y), n_subjects, target, class_balance,
    )

    if model_type == "cnn":
        base_pipe = None
        param_grid = {}
    elif model_type == "lr":
        base_pipe = create_logistic_pipeline(seed=rng_seed, config=config)
        param_grid = build_logistic_param_grid(config)
    elif model_type == "rf":
        base_pipe = create_rf_classification_pipeline(seed=rng_seed, config=config)
        param_grid = build_rf_classification_param_grid(config)
    else:
        base_pipe = create_svm_pipeline(seed=rng_seed, config=config)
        param_grid = build_svm_param_grid(config)
        model_type = "svm"

    outer_cv_splits = int(
        get_config_value(
            config,
            "machine_learning.cv.outer_splits",
            get_config_value(config, "machine_learning.cv.default_n_splits", 5),
        )
    )
    folds = create_within_subject_folds(
        groups=groups,
        blocks_all=blocks_all,
        inner_cv_splits=inner_splits,
        seed=rng_seed,
        outer_cv_splits=outer_cv_splits,
        config=config,
        epochs=None,
        apply_hygiene=False,
    )
    if not folds:
        raise ValueError(
            "No within-subject folds could be created. "
            "Ensure each selected subject has at least 2 unique block/run labels."
        )

    def _run_with_labels(y_labels: np.ndarray) -> Tuple[List[Dict[str, Any]], int]:
        recs: List[Dict[str, Any]] = []
        failed_folds = 0
        for fold_counter, train_idx, test_idx, subject_id, _fold_params in folds:
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_labels[train_idx]
            y_test = y_labels[test_idx]
            blocks_train = blocks_all[train_idx]
            groups_train = groups[train_idx]

            if model_type != "cnn":
                X_train, X_test, _ = apply_fold_feature_harmonization(
                    X_train,
                    X_test,
                    groups_train,
                    harmonization_mode,
                )

            # If training fold has only one class, fall back to majority-class prediction.
            unique_train = np.unique(y_train)
            if len(unique_train) < 2:
                maj = int(unique_train[0]) if len(unique_train) == 1 else 0
                y_pred = np.full(len(y_test), maj, dtype=int)
                y_prob = np.full(len(y_test), np.nan, dtype=float)
                failed_folds += 1
                recs.append(
                    {
                        "fold": int(fold_counter),
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "y_prob": y_prob,
                        "subject_id": [str(subject_id)] * len(y_test),
                        "test_idx": np.asarray(test_idx, dtype=int),
                    }
                )
                continue

            # Inner CV: stratified and block-aware (within subject).
            n_unique_blocks = len(np.unique(blocks_train))
            effective_splits = min(max(2, int(inner_splits)), n_unique_blocks)
            if model_type == "cnn":
                try:
                    y_pred, y_prob = fit_predict_cnn_binary_classifier(
                        X_train=X_train,
                        y_train=y_train,
                        groups_train=blocks_train.astype(str),
                        X_test=X_test,
                        seed=rng_seed + int(fold_counter),
                        config=config,
                        logger=logger,
                    )
                except Exception:
                    maj = int(np.median(y_train)) if len(y_train) else 0
                    y_pred = np.full(len(y_test), maj, dtype=int)
                    y_prob = np.full(len(y_test), np.nan, dtype=float)
                    failed_folds += 1
            else:
                best_estimator = clone(base_pipe)

                if effective_splits >= 2:
                    inner_cv = StratifiedGroupKFold(
                        n_splits=effective_splits,
                        shuffle=True,
                        random_state=rng_seed + int(fold_counter),
                    )
                    try:
                        grid = GridSearchCV(
                            estimator=clone(base_pipe),
                            param_grid=param_grid,
                            scoring="roc_auc",
                            cv=inner_cv,
                            n_jobs=1,
                            refit=True,
                            error_score="raise",
                        )
                        grid.fit(X_train, y_train, groups=blocks_train)
                        best_estimator = grid.best_estimator_
                    except Exception as exc:
                        logger.warning(
                            "Within-subject fold %s (%s): inner CV failed (%s); fitting default pipeline.",
                            int(fold_counter),
                            str(subject_id),
                            exc,
                        )
                        best_estimator.fit(X_train, y_train)
                else:
                    best_estimator.fit(X_train, y_train)

                try:
                    y_pred = best_estimator.predict(X_test).astype(int)
                    y_prob = None
                    if hasattr(best_estimator, "predict_proba"):
                        try:
                            y_prob = best_estimator.predict_proba(X_test)[:, 1]
                        except Exception:
                            y_prob = None
                except Exception:
                    maj = int(np.median(y_train)) if len(y_train) else 0
                    y_pred = np.full(len(y_test), maj, dtype=int)
                    y_prob = np.full(len(y_test), np.nan, dtype=float)
                    failed_folds += 1

            recs.append(
                {
                    "fold": int(fold_counter),
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "subject_id": [str(subject_id)] * len(y_test),
                    "test_idx": np.asarray(test_idx, dtype=int),
                }
            )
        return recs, int(failed_folds)

    fold_records, failed_fold_count = _run_with_labels(y)

    fold_records = sorted(fold_records, key=lambda r: r["fold"])
    y_true_all = np.concatenate([np.asarray(r["y_true"]) for r in fold_records]).astype(int)
    y_pred_all = np.concatenate([np.asarray(r["y_pred"]) for r in fold_records]).astype(int)
    groups_ordered: List[str] = []
    test_indices: List[int] = []
    fold_ids: List[int] = []
    y_prob_all_list = []
    has_prob = True

    for rec in fold_records:
        n = len(rec["y_true"])
        groups_ordered.extend(rec["subject_id"])
        test_indices.extend(rec["test_idx"].tolist())
        fold_ids.extend([rec["fold"]] * n)
        if rec.get("y_prob") is None:
            has_prob = False
        y_prob_all_list.append(rec.get("y_prob"))

    y_prob_all = None
    if has_prob:
        try:
            y_prob_all = np.concatenate([np.asarray(v, dtype=float) for v in y_prob_all_list])  # type: ignore[arg-type]
        except Exception:
            y_prob_all = None

    results_dir = results_root / "within_subject_classification"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    subject_selection = export_subject_selection_report(
        results_dir,
        subjects,
        np.asarray(groups_ordered, dtype=object),
        meta,
        config,
    )

    pred_path = results_dir / "cv_predictions.tsv"
    pred_df = export_predictions(
        y_true_all,
        y_pred_all,
        groups_ordered,
        test_indices,
        fold_ids,
        model_type,
        meta.reset_index(drop=True),
        pred_path,
    )
    if y_prob_all is not None and len(y_prob_all) == len(pred_df):
        pred_df["y_prob"] = np.asarray(y_prob_all, dtype=float)
    else:
        pred_df["y_prob"] = np.full(len(pred_df), np.nan, dtype=float)
    write_tsv(pred_df, pred_path)
    write_parquet(pred_df, pred_path.with_suffix(".parquet"))
    export_indices(
        groups_ordered,
        test_indices,
        fold_ids,
        meta.reset_index(drop=True),
        results_dir / "cv_indices.tsv",
    )

    result = ClassificationResult(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all,
        groups=np.asarray(groups_ordered),
        failed_fold_count=int(failed_fold_count),
        n_folds_total=int(len(folds)),
    )
    n_folds_total = int(getattr(result, "n_folds_total", len(folds)) or len(folds))
    failed_fold_fraction = float(int(failed_fold_count) / max(n_folds_total, 1))
    max_failed_fold_fraction = float(
        get_config_value(config, "machine_learning.classification.max_failed_fold_fraction", 0.25)
    )
    if failed_fold_fraction > max_failed_fold_fraction:
        raise RuntimeError(
            "Within-subject classification fold failure fraction exceeded validity threshold: "
            f"failed={int(failed_fold_count)}/{n_folds_total} ({failed_fold_fraction:.3f}) > "
            f"machine_learning.classification.max_failed_fold_fraction={max_failed_fold_fraction:.3f}."
        )

    # Save per-subject metrics
    if getattr(result, "per_subject_metrics", None):
        rows = []
        for subj, rec in result.per_subject_metrics.items():
            row = {"subject_id": subj}
            row.update(rec)
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(results_dir / "per_subject_metrics.tsv", sep="\t", index=False)

    auc_subj_mean = _subject_mean_metric(result.per_subject_metrics, "auc")
    bal_acc_subj_mean = _subject_mean_metric(result.per_subject_metrics, "balanced_accuracy")
    acc_subj_mean = _subject_mean_metric(result.per_subject_metrics, "accuracy")
    precision_subj_mean = _subject_mean_metric(result.per_subject_metrics, "precision")
    recall_subj_mean = _subject_mean_metric(result.per_subject_metrics, "recall")
    f1_subj_mean = _subject_mean_metric(result.per_subject_metrics, "f1")
    specificity_subj_mean = _subject_mean_metric(result.per_subject_metrics, "specificity")
    avg_precision_subj_mean = _subject_mean_metric(result.per_subject_metrics, "average_precision")
    min_subjects_auc = int(
        get_config_value(config, "machine_learning.classification.min_subjects_with_auc_for_inference", 2)
    )
    n_subjects_total = int(len(np.unique(groups_ordered)))
    n_subjects_with_auc = _count_finite_subject_metric(result.per_subject_metrics, "auc")
    auc_inference_valid = n_subjects_with_auc >= min_subjects_auc
    if not auc_inference_valid:
        logger.warning(
            "Within-subject classification AUC inference disabled: only %d/%d subjects had evaluable AUC (< %d required).",
            int(n_subjects_with_auc),
            int(n_subjects_total),
            int(min_subjects_auc),
        )
    auc_for_inference = auc_subj_mean if (np.isfinite(auc_subj_mean) and auc_inference_valid) else np.nan
    n_boot = int(get_config_value(config, "machine_learning.evaluation.bootstrap_iterations", 1000))
    ci_rng = np.random.default_rng(int(rng_seed) + 401)
    auc_vals = _subject_metric_values(result.per_subject_metrics, "auc")
    bal_vals = _subject_metric_values(result.per_subject_metrics, "balanced_accuracy")
    acc_vals = _subject_metric_values(result.per_subject_metrics, "accuracy")
    precision_vals = _subject_metric_values(result.per_subject_metrics, "precision")
    recall_vals = _subject_metric_values(result.per_subject_metrics, "recall")
    f1_vals = _subject_metric_values(result.per_subject_metrics, "f1")
    specificity_vals = _subject_metric_values(result.per_subject_metrics, "specificity")
    avg_precision_vals = _subject_metric_values(result.per_subject_metrics, "average_precision")
    auc_ci_low, auc_ci_high = (
        _bootstrap_mean_ci(auc_vals, rng=ci_rng, n_boot=n_boot)
        if auc_inference_valid
        else (np.nan, np.nan)
    )
    bal_ci_low, bal_ci_high = _bootstrap_mean_ci(bal_vals, rng=ci_rng, n_boot=n_boot)
    acc_ci_low, acc_ci_high = _bootstrap_mean_ci(acc_vals, rng=ci_rng, n_boot=n_boot)
    precision_ci_low, precision_ci_high = _bootstrap_mean_ci(precision_vals, rng=ci_rng, n_boot=n_boot)
    recall_ci_low, recall_ci_high = _bootstrap_mean_ci(recall_vals, rng=ci_rng, n_boot=n_boot)
    f1_ci_low, f1_ci_high = _bootstrap_mean_ci(f1_vals, rng=ci_rng, n_boot=n_boot)
    specificity_ci_low, specificity_ci_high = _bootstrap_mean_ci(
        specificity_vals, rng=ci_rng, n_boot=n_boot
    )
    avg_precision_ci_low, avg_precision_ci_high = _bootstrap_mean_ci(
        avg_precision_vals, rng=ci_rng, n_boot=n_boot
    )

    metrics: Dict[str, Any] = {
        "cv_scope": "subject",
        "model": model_type,
        "subject_selection": subject_selection,
        "data": {
            "target": target,
            "target_kind": "binary",
            "binary_threshold": binary_threshold,
            "feature_families": feature_families,
            "feature_bands": feature_bands,
            "feature_segments": feature_segments,
            "feature_scopes": feature_scopes,
            "feature_stats": feature_stats,
            "feature_harmonization": feature_harmonization,
            "covariates": covariates,
        },
        "n_subjects": int(len(np.unique(groups_ordered))),
        "n_samples": int(len(y_true_all)),
        "n_features": int(X.shape[1] * X.shape[2]) if X.ndim == 3 else int(X.shape[1]),
        "n_channels": int(X.shape[1]) if X.ndim == 3 else None,
        "n_timepoints": int(X.shape[2]) if X.ndim == 3 else None,
        "failed_fold_count": int(failed_fold_count),
        "n_folds_total": int(n_folds_total),
        "failed_fold_fraction": float(failed_fold_fraction),
        "max_failed_fold_fraction": float(max_failed_fold_fraction),
        "balanced_accuracy": (
            float(bal_acc_subj_mean) if np.isfinite(bal_acc_subj_mean) else np.nan
        ),
        "accuracy": float(acc_subj_mean) if np.isfinite(acc_subj_mean) else np.nan,
        "auc": float(auc_for_inference) if np.isfinite(auc_for_inference) else np.nan,
        "average_precision": (
            float(avg_precision_subj_mean) if np.isfinite(avg_precision_subj_mean) else np.nan
        ),
        "f1": float(f1_subj_mean) if np.isfinite(f1_subj_mean) else np.nan,
        "precision": float(precision_subj_mean) if np.isfinite(precision_subj_mean) else np.nan,
        "recall": float(recall_subj_mean) if np.isfinite(recall_subj_mean) else np.nan,
        "specificity": (
            float(specificity_subj_mean) if np.isfinite(specificity_subj_mean) else np.nan
        ),
        "subject_level": {
            "auc_mean": float(auc_subj_mean) if np.isfinite(auc_subj_mean) else np.nan,
            "auc_ci_low": auc_ci_low,
            "auc_ci_high": auc_ci_high,
            "balanced_accuracy_mean": float(bal_acc_subj_mean)
            if np.isfinite(bal_acc_subj_mean)
            else np.nan,
            "balanced_accuracy_ci_low": bal_ci_low,
            "balanced_accuracy_ci_high": bal_ci_high,
            "accuracy_mean": float(acc_subj_mean) if np.isfinite(acc_subj_mean) else np.nan,
            "accuracy_ci_low": acc_ci_low,
            "accuracy_ci_high": acc_ci_high,
            "average_precision_mean": (
                float(avg_precision_subj_mean) if np.isfinite(avg_precision_subj_mean) else np.nan
            ),
            "average_precision_ci_low": avg_precision_ci_low,
            "average_precision_ci_high": avg_precision_ci_high,
            "precision_mean": float(precision_subj_mean) if np.isfinite(precision_subj_mean) else np.nan,
            "precision_ci_low": precision_ci_low,
            "precision_ci_high": precision_ci_high,
            "recall_mean": float(recall_subj_mean) if np.isfinite(recall_subj_mean) else np.nan,
            "recall_ci_low": recall_ci_low,
            "recall_ci_high": recall_ci_high,
            "f1_mean": float(f1_subj_mean) if np.isfinite(f1_subj_mean) else np.nan,
            "f1_ci_low": f1_ci_low,
            "f1_ci_high": f1_ci_high,
            "specificity_mean": (
                float(specificity_subj_mean) if np.isfinite(specificity_subj_mean) else np.nan
            ),
            "specificity_ci_low": specificity_ci_low,
            "specificity_ci_high": specificity_ci_high,
            "n_subjects_with_auc": int(n_subjects_with_auc),
            "n_subjects_without_auc": int(max(0, n_subjects_total - n_subjects_with_auc)),
            "min_subjects_with_auc_for_inference": int(min_subjects_auc),
            "auc_inference_valid": bool(auc_inference_valid),
        },
        "pooled_trials": {
            "auc": float(result.auc) if np.isfinite(result.auc) else np.nan,
            "balanced_accuracy": float(result.balanced_accuracy)
            if np.isfinite(result.balanced_accuracy)
            else np.nan,
            "accuracy": float(result.accuracy) if np.isfinite(result.accuracy) else np.nan,
            "average_precision": float(result.average_precision)
            if np.isfinite(result.average_precision)
            else np.nan,
            "precision": float(result.precision) if np.isfinite(result.precision) else np.nan,
            "recall": float(result.recall) if np.isfinite(result.recall) else np.nan,
            "f1": float(result.f1) if np.isfinite(result.f1) else np.nan,
            "specificity": float(result.specificity) if np.isfinite(result.specificity) else np.nan,
        },
        "class_balance": float(np.mean(y_true_all)) if len(y_true_all) else np.nan,
        "n_perm": int(n_perm) if n_perm else 0,
    }
    if not auc_inference_valid:
        metrics["auc_reporting_note"] = (
            "Subject-level AUC inference is invalid for this run; pooled-trial AUC is reported for diagnostics only."
        )

    # Optional permutation p-value (AUC) under label randomization (full refit per permutation).
    if n_perm and n_perm > 0:
        perm_scheme = _resolve_permutation_scheme(config)
        logger.info("Running %d permutations for within-subject classification...", int(n_perm))
        rng = np.random.default_rng(rng_seed)
        null_auc = []
        n_effective = 0
        n_fallback_permutations = 0
        min_shuffle_fraction = float(
            get_config_value(config, "machine_learning.cv.min_label_shuffle_fraction", 0.01)
        )
        max_failed_perm_fraction = float(
            get_config_value(config, "machine_learning.classification.max_failed_fold_fraction", 0.25)
        )
        for i in range(int(n_perm)):
            y_perm, effective, _changed_fraction, used_scheme = _generate_effective_permutation(
                y,
                groups,
                blocks=blocks_all,
                rng=rng,
                requested_scheme=perm_scheme,
                min_changed_fraction=min_shuffle_fraction,
            )
            if not effective:
                continue
            n_effective += 1
            if used_scheme != perm_scheme:
                n_fallback_permutations += 1

            perm_records, perm_failed_folds = _run_with_labels(y_perm)
            perm_failed_fraction = float(perm_failed_folds / max(len(folds), 1))
            if perm_failed_fraction > max_failed_perm_fraction:
                continue
            perm_records = sorted(perm_records, key=lambda r: r["fold"])
            y_true_p = np.concatenate([np.asarray(r["y_true"]) for r in perm_records]).astype(int)
            try:
                y_pred_p = np.concatenate([np.asarray(r["y_pred"]) for r in perm_records]).astype(int)
                groups_p: List[str] = []
                y_prob_parts = []
                has_prob = True
                for r in perm_records:
                    groups_p.extend([str(s) for s in r["subject_id"]])
                    if r.get("y_prob") is None:
                        has_prob = False
                    y_prob_parts.append(r.get("y_prob"))
                y_prob_p = None
                if has_prob:
                    y_prob_p = np.concatenate([np.asarray(v, dtype=float) for v in y_prob_parts])  # type: ignore[arg-type]
                perm_result = ClassificationResult(
                    y_true=y_true_p,
                    y_pred=y_pred_p,
                    y_prob=y_prob_p,
                    groups=np.asarray(groups_p),
                )
                perm_auc_subj_mean = _subject_mean_metric(perm_result.per_subject_metrics, "auc")
                perm_n_subjects_with_auc = _count_finite_subject_metric(
                    perm_result.per_subject_metrics,
                    "auc",
                )
                if (
                    perm_n_subjects_with_auc >= int(min_subjects_auc)
                    and np.isfinite(perm_auc_subj_mean)
                ):
                    null_auc.append(float(perm_auc_subj_mean))
            except Exception:
                continue

        if n_fallback_permutations > 0:
            logger.info(
                "Within-subject classification permutations: %d/%d effective shuffles required fallback to within-subject scheme.",
                int(n_fallback_permutations),
                int(n_effective),
            )

        if n_effective == 0:
            raise RuntimeError(
                "No effective permutations could be generated for within-subject classification. "
                "Try machine_learning.cv.permutation_scheme='within_subject' and/or lower "
                "machine_learning.cv.min_label_shuffle_fraction."
            )

        n_completed = len(null_auc)
        completion_rate = (n_completed / int(n_perm)) if int(n_perm) > 0 else 0.0
        min_completion = float(
            get_config_value(config, "machine_learning.cv.min_valid_permutation_fraction", 0.5)
        )
        if int(n_perm) > 0 and completion_rate < min_completion:
            raise RuntimeError(
                f"Insufficient valid within-subject classification permutations ({n_completed}/{int(n_perm)}, "
                f"rate={completion_rate:.3f} < required {min_completion:.3f})"
            )

        if null_auc and np.isfinite(auc_for_inference):
            null_auc_arr = np.asarray(null_auc, dtype=float)
            metrics["n_perm_requested"] = int(n_perm)
            metrics["n_perm_completed"] = int(len(null_auc_arr))
            p_val = float(((null_auc_arr >= float(auc_for_inference)).sum() + 1) / (len(null_auc_arr) + 1))
            metrics["p_value_auc"] = p_val
            np.savez(
                results_dir / f"cv_null_{model_type}.npz",
                null_auc_subject_mean=null_auc_arr,
                empirical_auc_subject_mean=auc_for_inference,
            )

    write_reproducibility_info(results_dir, subjects, config, rng_seed)
    with open(results_dir / "pooled_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    p_info = ""
    if "p_value_auc" in metrics:
        p_info = f", p={metrics['p_value_auc']:.4f}"
    if not auc_inference_valid:
        logger.warning(
            "Subject-level AUC inference is invalid; pooled-trial AUC is reported for diagnostics only."
        )
    logger.info(
        "Within-subject classification results: AUC=%.3f, balanced_acc=%.3f, F1=%.3f%s",
        float(auc_for_inference) if np.isfinite(auc_for_inference) else result.auc,
        float(bal_acc_subj_mean) if np.isfinite(bal_acc_subj_mean) else result.balanced_accuracy,
        float(f1_subj_mean) if np.isfinite(f1_subj_mean) else result.f1, p_info,
    )
    _maybe_generate_mode_plots(mode="classify", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    return results_dir


def _run_classification_permutations(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    blocks: Optional[np.ndarray],
    model: str,
    inner_splits: int,
    seed: int,
    n_perm: int,
    config: Any,
    logger: logging.Logger,
    harmonization_mode: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Run permutation test for classification."""
    from eeg_pipeline.analysis.machine_learning.classification import nested_loso_classification
    from eeg_pipeline.analysis.machine_learning.cnn import nested_loso_cnn_classification
    
    rng = np.random.default_rng(seed)
    null_aucs = []
    n_effective = 0
    n_fallback_permutations = 0
    min_shuffle_fraction = float(
        get_config_value(config, "machine_learning.cv.min_label_shuffle_fraction", 0.01)
    )

    perm_scheme = "within_subject_within_block"
    try:
        perm_scheme = str(get_config_value(config, "machine_learning.cv.permutation_scheme", perm_scheme)).strip().lower()
    except Exception:
        perm_scheme = "within_subject_within_block"
    if perm_scheme not in {"within_subject", "within_subject_within_block"}:
        perm_scheme = "within_subject_within_block"

    blocks_arr = None
    if perm_scheme == "within_subject_within_block":
        if blocks is None:
            logger.warning(
                "Permutation scheme 'within_subject_within_block' requested but blocks are missing; falling back to within_subject."
            )
            perm_scheme = "within_subject"
        else:
            blocks_arr = np.asarray(blocks)
            if len(blocks_arr) != len(y):
                logger.warning("Permutation blocks length mismatch; falling back to within_subject.")
                perm_scheme = "within_subject"
                blocks_arr = None
    
    max_failed_perm_fraction = float(
        get_config_value(config, "machine_learning.classification.max_failed_fold_fraction", 0.25)
    )
    min_subjects_auc = int(
        get_config_value(config, "machine_learning.classification.min_subjects_with_auc_for_inference", 2)
    )

    for i in range(n_perm):
        y_perm, effective, _changed_fraction, used_scheme = _generate_effective_permutation(
            y,
            groups,
            blocks=blocks_arr,
            rng=rng,
            requested_scheme=perm_scheme,
            min_changed_fraction=min_shuffle_fraction,
        )
        if not effective:
            continue
        n_effective += 1
        if used_scheme != perm_scheme:
            n_fallback_permutations += 1
        
        try:
            if str(model).strip().lower() == "cnn":
                result, _ = nested_loso_cnn_classification(
                    X=X,
                    y=y_perm,
                    groups=groups,
                    seed=seed + i + 1,
                    config=config,
                    logger=None,
                )
            else:
                result, _ = nested_loso_classification(
                    X=X,
                    y=y_perm,
                    groups=groups,
                    model=model,
                    inner_splits=inner_splits,
                    seed=seed + i + 1,
                    config=config,
                    logger=None,
                    harmonization_mode=harmonization_mode,
                )
            perm_failed_fold_count = int(getattr(result, "failed_fold_count", 0) or 0)
            perm_n_folds_total = int(getattr(result, "n_folds_total", len(np.unique(groups))) or len(np.unique(groups)))
            perm_failed_fraction = float(perm_failed_fold_count / max(perm_n_folds_total, 1))
            if perm_failed_fraction > max_failed_perm_fraction:
                continue
            auc_subj_mean = _subject_mean_metric(result.per_subject_metrics, "auc")
            n_subjects_with_auc = _count_finite_subject_metric(result.per_subject_metrics, "auc")
            if n_subjects_with_auc >= min_subjects_auc and np.isfinite(auc_subj_mean):
                null_aucs.append(float(auc_subj_mean))
        except Exception:
            continue
        
        if (i + 1) % 10 == 0:
            logger.info(f"Permutation {i + 1}/{n_perm}")

    if n_fallback_permutations > 0:
        logger.info(
            "Classification permutations: %d/%d effective shuffles required fallback to within-subject scheme.",
            int(n_fallback_permutations),
            int(n_effective),
        )

    if n_effective == 0:
        raise RuntimeError(
            "No effective permutations could be generated for classification. "
            "Try machine_learning.cv.permutation_scheme='within_subject' and/or lower "
            "machine_learning.cv.min_label_shuffle_fraction."
        )

    n_completed = len(null_aucs)
    completion_rate = (n_completed / n_perm) if n_perm > 0 else 0.0
    min_completion = float(get_config_value(config, "machine_learning.cv.min_valid_permutation_fraction", 0.5))
    if n_perm > 0 and completion_rate < min_completion:
        raise RuntimeError(
            f"Insufficient valid classification permutations ({n_completed}/{n_perm}, "
            f"rate={completion_rate:.3f} < required {min_completion:.3f})"
        )

    return np.array(null_aucs, dtype=float) if null_aucs else None


###################################################################
# Reproducibility & Baseline Outputs
###################################################################


def write_reproducibility_info(
    results_dir: Path,
    subjects: List[str],
    config: Any,
    rng_seed: int,
) -> Path:
    """Write reproducibility information for ML results.
    
    Includes: config snapshot, data signature, sklearn version, RNG seed.
    """
    import sklearn
    import hashlib
    
    # Data signature: subjects list + hash
    subjects_str = ",".join(sorted(subjects))
    data_hash = hashlib.sha256(subjects_str.encode()).hexdigest()[:16]
    
    repro_info = {
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rng_seed": rng_seed,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "data_signature": data_hash,
        "config_snapshot": {
            "machine_learning": config.get("machine_learning", {}) if hasattr(config, "get") else {},
        },
    }
    
    repro_path = results_dir / "reproducibility_info.json"
    with open(repro_path, "w") as f:
        json.dump(repro_info, f, indent=2, default=str)
    
    return repro_path


def compute_baseline_predictions(
    y: np.ndarray,
    groups: np.ndarray,
    task: str = "regression",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute baseline (null) model predictions for sanity checks.
    
    Regression: mean predictor (leave-one-subject-out mean)
    Classification: majority class predictor
    
    Returns:
        y_pred_baseline: Baseline predictions
        baseline_metrics: Baseline model metrics
    """
    from sklearn.model_selection import LeaveOneGroupOut
    
    y_pred = np.zeros_like(y, dtype=float)
    logo = LeaveOneGroupOut()
    
    for train_idx, test_idx in logo.split(y, y, groups):
        if task == "regression":
            y_pred[test_idx] = np.nanmean(y[train_idx])
        else:
            # Majority class
            classes, counts = np.unique(y[train_idx], return_counts=True)
            y_pred[test_idx] = classes[np.argmax(counts)]
    
    if task == "regression":
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        metrics = {"baseline_r2": r2, "baseline_mae": mae}
    else:
        from sklearn.metrics import balanced_accuracy_score, accuracy_score
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        metrics = {"baseline_accuracy": acc, "baseline_balanced_accuracy": bal_acc}
    
    return y_pred, metrics


def export_baseline_predictions(
    y_true: np.ndarray,
    groups: np.ndarray,
    results_dir: Path,
    task: str = "regression",
) -> Dict[str, float]:
    """Compute and export baseline model predictions."""
    y_pred_baseline, baseline_metrics = compute_baseline_predictions(y_true, groups, task)
    
    baseline_df = pd.DataFrame({
        "subject_id": groups,
        "y_true": y_true,
        "y_pred_baseline": y_pred_baseline,
    })
    baseline_df.to_csv(results_dir / "baseline_predictions.tsv", sep="\t", index=False)
    
    return baseline_metrics


###################################################################
# Model Comparison Compute Stage
###################################################################


def run_model_comparison_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    *,
    target: Optional[str] = None,
    feature_families: Optional[List[str]] = None,
    feature_harmonization: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Path:
    """Compare multiple model families with identical outer folds.
    
    Compares ElasticNet vs Random Forest vs Ridge/SVR using nested CV.
    All models share the same outer folds for valid comparison.
    
    Outputs:
        model_comparison.tsv: Per-fold metrics for each model
        model_comparison_summary.json: Aggregated comparison statistics
    """
    if int(n_perm) < 0:
        raise ValueError("n_perm must be >= 0 for model comparison.")

    if target is None:
        target = get_config_value(config, "machine_learning.targets.regression", None)

    X, y, groups, feature_names, meta = load_active_matrix(
        subjects,
        task,
        deriv_root,
        config,
        logger,
        feature_families=feature_families,
        feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
        target=target,
        target_kind="continuous",
        covariates=covariates,
        feature_bands=feature_bands,
        feature_segments=feature_segments,
        feature_scopes=feature_scopes,
        feature_stats=feature_stats,
    )
    target_detection = _warn_or_raise_if_binary_like_regression_target(
        y,
        target,
        logger,
        config,
        context="Model comparison",
    )
    
    n_subjects = len(np.unique(groups))
    logger.info(
        "Model comparison: %d features \u00d7 %d trials, %d subjects, target='%s'",
        X.shape[1], X.shape[0], n_subjects, target,
    )

    results_dir = results_root / "model_comparison"
    ensure_dir(results_dir)
    subject_selection = export_subject_selection_report(results_dir, subjects, groups, meta, config)
    
    # Define model pipelines (shared preprocessing + config)
    models = {
        "elasticnet": {
            "pipe": create_elasticnet_pipeline(seed=rng_seed, config=config),
            "param_grid": build_elasticnet_param_grid(config),
        },
        "ridge": {
            "pipe": create_ridge_pipeline(seed=rng_seed, config=config),
            "param_grid": build_ridge_param_grid(config),
        },
        "rf": {
            "pipe": create_rf_pipeline(seed=rng_seed, config=config),
            "param_grid": build_rf_param_grid(config),
        },
    }
    
    # Shared outer CV folds
    from sklearn.model_selection import LeaveOneGroupOut
    outer_cv = LeaveOneGroupOut()
    harmonization_mode = (
        feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection"))
    )
    outer_folds = list(outer_cv.split(X, y, groups))
    
    comparison_records = []
    
    import time as _time

    for model_name, model_spec in models.items():
        t_model = _time.perf_counter()
        pipe = model_spec["pipe"]
        param_grid = model_spec["param_grid"]
        
        y_pred = np.zeros(len(y))
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            X_train, X_test, _ = apply_fold_feature_harmonization(
                X_train,
                X_test,
                groups_train,
                harmonization_mode,
            )
            
            # Inner CV for hyperparameter tuning (group-aware)
            if len(np.unique(groups_train)) >= 2:
                inner_cv = create_inner_cv(groups_train, inner_splits)
                grid_n_jobs = determine_inner_n_jobs(outer_jobs, n_jobs=1)
                scoring = create_scoring_dict()
                grid = GridSearchCV(
                    clone(pipe),
                    param_grid,
                    cv=inner_cv,
                    scoring=scoring,
                    n_jobs=grid_n_jobs,
                    refit="r",
                    error_score="raise",
                )
                grid.fit(X_train, y_train, groups=groups_train)
                fold_pred = grid.predict(X_test)
                best_params_repr = str(grid.best_params_)
            else:
                est = clone(pipe)
                est.fit(X_train, y_train)
                fold_pred = est.predict(X_test)
                best_params_repr = "{}"

            y_pred[test_idx] = fold_pred
            
            # Record fold metrics
            from sklearn.metrics import r2_score, mean_absolute_error
            fold_r2 = r2_score(y_test, fold_pred)
            fold_mae = mean_absolute_error(y_test, fold_pred)
            
            comparison_records.append({
                "model": model_name,
                "fold": fold_idx,
                "test_subject": groups[test_idx[0]],
                "r2": fold_r2,
                "mae": fold_mae,
                "best_params": best_params_repr,
            })
        
        # Overall metrics
        from sklearn.metrics import r2_score
        overall_r2 = r2_score(y, y_pred)
        logger.info(
            "  \u2713 %s: R\u00b2=%.4f (%.1fs)",
            model_name, overall_r2, _time.perf_counter() - t_model,
        )
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(results_dir / "model_comparison.tsv", sep="\t", index=False)
    
    # Summary statistics (fold-level; outer unit = subject)
    summary: Dict[str, Any] = {
        "data": {
            "target": target,
            "target_kind": "continuous",
            "detected_target": target_detection,
            "feature_families": feature_families,
            "feature_bands": feature_bands,
            "feature_segments": feature_segments,
            "feature_scopes": feature_scopes,
            "feature_stats": feature_stats,
            "feature_harmonization": feature_harmonization,
            "covariates": covariates,
        },
        "subject_selection": subject_selection,
    }
    n_boot = int(get_config_value(config, "machine_learning.evaluation.bootstrap_iterations", 1000))
    boot_rng = np.random.default_rng(int(rng_seed) + 101)
    for model_name in models.keys():
        model_rows = comparison_df[comparison_df["model"] == model_name]
        r2_vals = pd.to_numeric(model_rows["r2"], errors="coerce").to_numpy(dtype=float)
        mae_vals = pd.to_numeric(model_rows["mae"], errors="coerce").to_numpy(dtype=float)
        r2_ci_low, r2_ci_high = _bootstrap_mean_ci(
            r2_vals,
            rng=boot_rng,
            n_boot=n_boot,
        )
        mae_ci_low, mae_ci_high = _bootstrap_mean_ci(
            mae_vals,
            rng=boot_rng,
            n_boot=n_boot,
        )
        summary[model_name] = {
            "mean_r2": float(model_rows["r2"].mean()),
            "std_r2": float(model_rows["r2"].std()),
            "ci_low_r2": r2_ci_low,
            "ci_high_r2": r2_ci_high,
            "mean_mae": float(model_rows["mae"].mean()),
            "std_mae": float(model_rows["mae"].std()),
            "ci_low_mae": mae_ci_low,
            "ci_high_mae": mae_ci_high,
            "n_folds": int(len(model_rows)),
        }

    # Pairwise model-difference inference (subject-paired by held-out fold).
    pairwise: Dict[str, Any] = {}
    pair_rng = np.random.default_rng(int(rng_seed) + 102)
    ordered_models = list(models.keys())
    for i in range(len(ordered_models)):
        for j in range(i + 1, len(ordered_models)):
            m_a = ordered_models[i]
            m_b = ordered_models[j]
            a_rows = comparison_df.loc[comparison_df["model"] == m_a, ["test_subject", "r2", "mae"]]
            b_rows = comparison_df.loc[comparison_df["model"] == m_b, ["test_subject", "r2", "mae"]]
            merged = a_rows.merge(
                b_rows,
                on="test_subject",
                how="inner",
                suffixes=("_a", "_b"),
            )
            if merged.empty:
                continue

            delta_r2 = (
                pd.to_numeric(merged["r2_a"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(merged["r2_b"], errors="coerce").to_numpy(dtype=float)
            )
            delta_mae = (
                pd.to_numeric(merged["mae_a"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(merged["mae_b"], errors="coerce").to_numpy(dtype=float)
            )
            r2_ci_low, r2_ci_high = _bootstrap_mean_ci(delta_r2, rng=pair_rng, n_boot=n_boot)
            mae_ci_low, mae_ci_high = _bootstrap_mean_ci(delta_mae, rng=pair_rng, n_boot=n_boot)
            pair_rec: Dict[str, Any] = {
                "n_subjects": int(np.sum(np.isfinite(delta_r2))),
                "mean_delta_r2": float(np.nanmean(delta_r2)) if np.any(np.isfinite(delta_r2)) else np.nan,
                "ci_low_delta_r2": r2_ci_low,
                "ci_high_delta_r2": r2_ci_high,
                "mean_delta_mae": float(np.nanmean(delta_mae)) if np.any(np.isfinite(delta_mae)) else np.nan,
                "ci_low_delta_mae": mae_ci_low,
                "ci_high_delta_mae": mae_ci_high,
            }
            if int(n_perm) > 0:
                pair_rec["n_perm"] = int(n_perm)
                pair_rec["p_value_delta_r2"] = _paired_signflip_p_value(
                    delta_r2,
                    rng=pair_rng,
                    n_perm=int(n_perm),
                )
                pair_rec["p_value_delta_mae"] = _paired_signflip_p_value(
                    delta_mae,
                    rng=pair_rng,
                    n_perm=int(n_perm),
                )
            pairwise[f"{m_a}_minus_{m_b}"] = pair_rec

    if int(n_perm) > 0 and pairwise:
        try:
            from statsmodels.stats.multitest import multipletests
        except Exception:
            multipletests = None

        for raw_key in ("p_value_delta_r2", "p_value_delta_mae"):
            pair_names: List[str] = []
            p_values: List[float] = []
            for pair_name, pair_rec in pairwise.items():
                value = pair_rec.get(raw_key)
                if value is None:
                    continue
                try:
                    p_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(p_value):
                    continue
                pair_names.append(pair_name)
                p_values.append(p_value)

            if not p_values:
                continue

            adj_key = f"{raw_key}_holm"
            if multipletests is None:
                for pair_name, p_value in zip(pair_names, p_values):
                    pairwise[pair_name][adj_key] = float(p_value)
            else:
                _reject, p_adjusted, _alphac_sidak, _alphac_bonf = multipletests(
                    p_values,
                    alpha=0.05,
                    method="holm",
                )
                for pair_name, p_adj in zip(pair_names, p_adjusted):
                    pairwise[pair_name][adj_key] = float(p_adj)
    summary["pairwise_inference"] = pairwise
    
    with open(results_dir / "model_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    write_reproducibility_info(results_dir, subjects, config, rng_seed)
    best_model = max(models.keys(), key=lambda m: summary[m]["mean_r2"])
    logger.info(
        "Model comparison complete: best=%s (mean R\u00b2=%.4f)",
        best_model, summary[best_model]["mean_r2"],
    )
    _maybe_generate_mode_plots(mode="model_comparison", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    
    return results_dir


###################################################################
# Incremental Validity Compute Stage
###################################################################


def run_incremental_validity_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
    *,
    target: Optional[str] = None,
    baseline_predictors: Optional[List[str]] = None,
    feature_families: Optional[List[str]] = None,
    feature_harmonization: Optional[str] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Path:
    """Quantify Δperformance when adding EEG features over baseline predictors.
    
    Compares:
    - Baseline: temperature-only predictor
    - Full: temperature + EEG features
    
    All evaluations strictly out-of-fold to avoid leakage.
    
    Outputs:
        incremental_validity.tsv: Per-fold performance for baseline vs full
        incremental_validity_summary.json: Aggregated Δ statistics
    """
    if int(n_perm) < 0:
        raise ValueError("n_perm must be >= 0 for incremental validity.")

    if target is None:
        target = get_config_value(config, "machine_learning.targets.regression", None)

    X, y, groups, feature_names, meta = load_active_matrix(
        subjects,
        task,
        deriv_root,
        config,
        logger,
        feature_families=feature_families,
        feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
        target=target,
        target_kind="continuous",
        feature_bands=feature_bands,
        feature_segments=feature_segments,
        feature_scopes=feature_scopes,
        feature_stats=feature_stats,
    )
    target_detection = _warn_or_raise_if_binary_like_regression_target(
        y,
        target,
        logger,
        config,
        context="Incremental validity",
    )
    
    n_subjects = len(np.unique(groups))
    logger.info(
        "Incremental validity: %d features \u00d7 %d trials, %d subjects, target='%s'",
        X.shape[1], X.shape[0], n_subjects, target,
    )

    results_dir = results_root / "incremental_validity"
    ensure_dir(results_dir)
    subject_selection = export_subject_selection_report(results_dir, subjects, groups, meta, config)
    
    if baseline_predictors is None:
        raw_preds = get_config_value(config, "machine_learning.incremental_validity.baseline_predictors", ["temperature"])
        if isinstance(raw_preds, (list, tuple)):
            baseline_predictors = [str(v) for v in raw_preds if str(v).strip() != ""]
        elif isinstance(raw_preds, str) and raw_preds.strip():
            baseline_predictors = [raw_preds.strip()]
        else:
            baseline_predictors = ["temperature"]

    # Guard against target leakage through baseline predictors.
    forbidden_predictors = {v.lower() for v in _target_covariate_aliases(target, config=config)}
    leaking = [p for p in baseline_predictors if str(p).strip().lower() in forbidden_predictors]
    if leaking:
        raise ValueError(
            "Baseline predictors include the selected target, which invalidates incremental-validity estimates: "
            f"{leaking}. Target={target!r}."
        )

    require_baseline_predictors = bool(
        get_config_value(config, "machine_learning.incremental_validity.require_baseline_predictors", True)
    )

    # Extract baseline predictors from meta (meta uses standardized names: temperature, trial_index, block, etc.)
    missing = [c for c in baseline_predictors if c not in meta.columns]
    if missing:
        msg = (
            "Missing baseline predictors in meta: %s. Available meta columns=%s."
            % (",".join(missing), ",".join(list(meta.columns)))
        )
        if require_baseline_predictors:
            raise ValueError(
                msg
                + " Baseline fallback is disabled by "
                "machine_learning.incremental_validity.require_baseline_predictors=true."
            )
        logger.warning("%s Falling back to intercept-only baseline.", msg)
        X_baseline = np.ones((len(y), 1))
        baseline_predictors = ["intercept_only"]
    else:
        X_baseline = meta[baseline_predictors].apply(pd.to_numeric, errors="coerce").to_numpy()
    
    # Full model includes baseline predictors + EEG features
    X_full = np.concatenate([X_baseline, X], axis=1)
    n_baseline_features = int(X_baseline.shape[1])
    
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import r2_score, mean_absolute_error
    
    outer_cv = LeaveOneGroupOut()
    harmonization_mode = (
        feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection"))
    )
    # Use the same model family/hyperparameter space for baseline and full models
    # so ΔR² isolates information gain from EEG predictors (not algorithm changes).
    shared_pipe = create_elasticnet_pipeline(seed=rng_seed, config=config)
    shared_param_grid = build_elasticnet_param_grid(config)
    
    records = []
    y_pred_baseline = np.zeros(len(y))
    y_pred_full = np.zeros(len(y))
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        test_subj = groups[test_idx[0]]
        groups_train = groups[train_idx]
        X_full_train = X_full[train_idx]
        X_full_test = X_full[test_idx]
        X_full_train, X_full_test, keep_mask = apply_fold_feature_harmonization(
            X_full_train,
            X_full_test,
            groups_train,
            harmonization_mode,
        )
        baseline_keep_mask = np.asarray(keep_mask, dtype=bool)[:n_baseline_features]
        if baseline_keep_mask.size == 0 or not np.any(baseline_keep_mask):
            raise ValueError(
                f"Fold {int(fold_idx)}: feature harmonization removed all baseline predictors. "
                "This invalidates incremental-validity comparison."
            )
        X_baseline_train = X_baseline[train_idx][:, baseline_keep_mask]
        X_baseline_test = X_baseline[test_idx][:, baseline_keep_mask]
        
        # Baseline model (out-of-fold; tuned under the same group-aware inner CV logic).
        base_pipe = clone(shared_pipe)
        n_train_subjects = len(np.unique(groups_train))
        if n_train_subjects >= 2:
            inner_cv_base = create_inner_cv(groups_train, inner_splits)
            grid_n_jobs_base = determine_inner_n_jobs(outer_n_jobs=1, n_jobs=1)
            scoring = create_scoring_dict()
            base_grid = GridSearchCV(
                estimator=base_pipe,
                param_grid=shared_param_grid,
                cv=inner_cv_base,
                scoring=scoring,
                n_jobs=grid_n_jobs_base,
                refit="r",
                error_score="raise",
            )
            base_grid.fit(X_baseline_train, y[train_idx], groups=groups_train)
            y_pred_baseline[test_idx] = base_grid.predict(X_baseline_test)
        else:
            base_pipe.fit(X_baseline_train, y[train_idx])
            y_pred_baseline[test_idx] = base_pipe.predict(X_baseline_test)
        r2_base = r2_score(y[test_idx], y_pred_baseline[test_idx])
        
        # Full model (same family as baseline; adds EEG predictors to baseline columns).
        pipe_full = clone(shared_pipe)
        if n_train_subjects >= 2:
            inner_cv = create_inner_cv(groups_train, inner_splits)
            grid_n_jobs = determine_inner_n_jobs(outer_n_jobs=1, n_jobs=1)
            scoring = create_scoring_dict()
            grid = GridSearchCV(
                pipe_full,
                shared_param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=grid_n_jobs,
                refit="r",
                error_score="raise",
            )
            grid.fit(X_full_train, y[train_idx], groups=groups_train)
            y_pred_full[test_idx] = grid.predict(X_full_test)
        else:
            pipe_full.fit(X_full_train, y[train_idx])
            y_pred_full[test_idx] = pipe_full.predict(X_full_test)
        r2_full = r2_score(y[test_idx], y_pred_full[test_idx])
        
        records.append({
            "fold": fold_idx,
            "test_subject": test_subj,
            "r2_baseline": r2_base,
            "r2_full": r2_full,
            "delta_r2": r2_full - r2_base,
            "mae_baseline": mean_absolute_error(y[test_idx], y_pred_baseline[test_idx]),
            "mae_full": mean_absolute_error(y[test_idx], y_pred_full[test_idx]),
        })
    
    records_df = pd.DataFrame(records)
    records_df.to_csv(results_dir / "incremental_validity.tsv", sep="\t", index=False)
    
    # Overall summary
    r2_baseline_overall = r2_score(y, y_pred_baseline)
    r2_full_overall = r2_score(y, y_pred_full)
    pooled_delta_r2 = float(r2_full_overall - r2_baseline_overall)
    mean_fold_delta_r2 = float(records_df["delta_r2"].mean())
    
    summary = {
        "data": {
            "target": target,
            "target_kind": "continuous",
            "detected_target": target_detection,
            "baseline_predictors": baseline_predictors,
            "feature_families": feature_families,
            "feature_bands": feature_bands,
            "feature_segments": feature_segments,
            "feature_scopes": feature_scopes,
            "feature_stats": feature_stats,
            "feature_harmonization": feature_harmonization,
        },
        "subject_selection": subject_selection,
        # Keep pooled trial-level metrics explicit and secondary.
        "pooled_trials": {
            "r2_baseline": float(r2_baseline_overall),
            "r2_full": float(r2_full_overall),
            "delta_r2": pooled_delta_r2,
        },
        "r2_baseline": float(r2_baseline_overall),
        "r2_full": float(r2_full_overall),
        "delta_r2_pooled_trials": pooled_delta_r2,
        # Primary incremental-validity estimate (subject-level; LOSO fold unit = subject).
        "delta_r2": mean_fold_delta_r2,
        "mean_fold_delta_r2": mean_fold_delta_r2,
        "std_fold_delta_r2": float(records_df["delta_r2"].std()),
        "n_folds_positive_delta": int((records_df["delta_r2"] > 0).sum()),
        "n_folds_total": len(records_df),
    }

    delta_vals = pd.to_numeric(records_df["delta_r2"], errors="coerce").to_numpy(dtype=float)
    n_boot = int(get_config_value(config, "machine_learning.evaluation.bootstrap_iterations", 1000))
    inf_rng = np.random.default_rng(int(rng_seed) + 201)
    ci_low, ci_high = _bootstrap_mean_ci(delta_vals, rng=inf_rng, n_boot=n_boot)
    delta_inference: Dict[str, Any] = {
        "mean_delta_r2": float(np.nanmean(delta_vals)) if np.any(np.isfinite(delta_vals)) else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_subjects": int(np.sum(np.isfinite(delta_vals))),
    }
    if int(n_perm) > 0:
        delta_inference["n_perm"] = int(n_perm)
        delta_inference["p_value"] = _paired_signflip_p_value(
            delta_vals,
            rng=inf_rng,
            n_perm=int(n_perm),
        )
    summary["delta_r2_inference"] = delta_inference
    
    with open(results_dir / "incremental_validity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    write_reproducibility_info(results_dir, subjects, config, rng_seed)
    logger.info(
        "Incremental validity: subject-level \u0394R\u00b2=%.4f (pooled-trials \u0394R\u00b2=%.4f; %d/%d folds positive)",
        summary["delta_r2"], summary["pooled_trials"]["delta_r2"],
        summary["n_folds_positive_delta"], summary["n_folds_total"],
    )
    _maybe_generate_mode_plots(mode="incremental_validity", results_dir=results_dir, logger=logger, config=config)
    logger.info("Saved results to %s", results_dir)
    
    return results_dir




def _run_permutation_importance_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    config: Any,
    seed: int,
    n_repeats: int,
    results_dir: Path,
    logger: logging.Logger,
    model_name: str = "elasticnet",
) -> Optional[Path]:
    """Run per-fold permutation importance and aggregate."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.inspection import permutation_importance
    from eeg_pipeline.analysis.machine_learning.feature_metadata import (
        aggregate_importance,
        build_feature_metadata,
    )
    
    resolved_model, base_pipe, param_grid = _build_regression_model_spec(
        model_name,
        seed=seed,
        config=config,
    )
    harmonization_mode = str(
        get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")
    ).strip().lower()
    inner_splits = int(get_config_value(config, "machine_learning.cv.inner_splits", 5))
    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(X, y, groups))
    
    all_importances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        X_train, X_test, keep_mask = apply_fold_feature_harmonization(
            X_train,
            X_test,
            groups_train,
            harmonization_mode,
        )

        pipe_fold = clone(base_pipe)
        try:
            pipe_fold = _fit_tuned_regression_estimator(
                base_pipe=pipe_fold,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                inner_splits=inner_splits,
                seed=seed + fold_idx,
                logger=logger,
                fold_info=f"perm-importance fold {fold_idx}",
            )
            result = permutation_importance(
                pipe_fold, X_test, y_test,
                n_repeats=n_repeats,
                random_state=seed + fold_idx,
                scoring="r2",
                n_jobs=1,
            )
            full_importance = np.full(len(feature_names), np.nan, dtype=float)
            full_importance[np.asarray(keep_mask, dtype=bool)] = result.importances_mean
            all_importances.append(full_importance)
        except Exception as e:
            logger.warning(f"Fold {fold_idx} permutation importance failed: {e}")
            continue
    
    if not all_importances:
        logger.warning("No permutation importance results")
        return None
    n_folds_requested = int(len(fold_splits))
    n_folds_completed = int(len(all_importances))
    completion_rate = float(n_folds_completed / max(n_folds_requested, 1))
    min_completion = float(
        get_config_value(
            config,
            "machine_learning.analysis.permutation_importance.min_valid_fold_fraction",
            0.8,
        )
    )
    if completion_rate < min_completion:
        raise RuntimeError(
            "Insufficient valid permutation-importance folds: "
            f"completed={n_folds_completed}/{n_folds_requested} "
            f"(rate={completion_rate:.3f} < required {min_completion:.3f})."
        )
    
    mean_importance = np.mean(np.stack(all_importances), axis=0)
    std_importance = np.std(np.stack(all_importances), axis=0)
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": mean_importance,
        "importance_std": std_importance,
        "n_folds": len(all_importances),
    })
    importance_df = importance_df.loc[np.isfinite(importance_df["importance_mean"].to_numpy(dtype=float))]
    importance_df = importance_df.sort_values("importance_mean", ascending=False)
    
    output_path = results_dir / "permutation_importance.tsv"
    importance_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved permutation importance to {output_path}")

    # Optional grouped summaries for interpretability.
    try:
        if bool(get_config_value(config, "machine_learning.interpretability.grouped_outputs", True)):
            meta_df = build_feature_metadata(feature_names, config=config)
            merged = meta_df.merge(importance_df, on="feature", how="right")

            by_group_band = aggregate_importance(
                merged, value_col="importance_mean", group_cols=["group", "band"]
            )
            if not by_group_band.empty:
                by_group_band.to_csv(results_dir / "permutation_importance_by_group_band.tsv", sep="\t", index=False)

            by_group_band_roi = aggregate_importance(
                merged, value_col="importance_mean", group_cols=["group", "band", "roi"]
            )
            if not by_group_band_roi.empty:
                by_group_band_roi.to_csv(
                    results_dir / "permutation_importance_by_group_band_roi.tsv", sep="\t", index=False
                )
    except Exception as exc:
        logger.debug("Grouped permutation-importance export failed: %s", exc)

    _maybe_generate_mode_plots(mode="permutation", results_dir=results_dir, logger=logger, config=config)
    
    return output_path


def _run_shap_importance_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    config: Any,
    seed: int,
    results_dir: Path,
    logger: logging.Logger,
    model_name: str = "elasticnet",
) -> Optional[Path]:
    """Run per-fold SHAP importance and aggregate."""
    try:
        from eeg_pipeline.analysis.machine_learning.shap_importance import compute_shap_for_cv_folds
    except ImportError:
        logger.warning("SHAP not available; skipping SHAP importance")
        return None
    from eeg_pipeline.analysis.machine_learning.feature_metadata import (
        aggregate_importance,
        build_feature_metadata,
    )
    
    from sklearn.model_selection import LeaveOneGroupOut
    
    logo = LeaveOneGroupOut()
    cv_splits = list(logo.split(X, y, groups))
    harmonization_mode = str(
        get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")
    ).strip().lower()
    inner_splits = int(get_config_value(config, "machine_learning.cv.inner_splits", 5))
    _resolved_model, base_pipe, param_grid = _build_regression_model_spec(
        model_name,
        seed=seed,
        config=config,
    )

    def model_factory():
        return clone(base_pipe)
    
    try:
        importance_df = compute_shap_for_cv_folds(
            model_factory,
            X,
            y,
            cv_splits,
            feature_names,
            seed,
            groups=groups,
            harmonization_mode=harmonization_mode,
            param_grid=param_grid,
            inner_cv_splits=inner_splits,
        )
        
        if importance_df.empty:
            logger.warning("SHAP importance computation returned empty results")
            return None
        n_folds_attempted = int(
            importance_df["n_folds_attempted"].max()
            if "n_folds_attempted" in importance_df.columns
            else len(cv_splits)
        )
        n_folds_used = int(
            importance_df["n_folds_used"].max()
            if "n_folds_used" in importance_df.columns
            else n_folds_attempted
        )
        completion_rate = float(n_folds_used / max(n_folds_attempted, 1))
        min_completion = float(
            get_config_value(config, "machine_learning.analysis.shap.min_valid_fold_fraction", 0.8)
        )
        if completion_rate < min_completion:
            raise RuntimeError(
                "Insufficient valid SHAP folds: "
                f"completed={n_folds_used}/{n_folds_attempted} "
                f"(rate={completion_rate:.3f} < required {min_completion:.3f})."
            )
        
        output_path = results_dir / "shap_importance.tsv"
        importance_df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved SHAP importance to {output_path}")

        # Optional grouped summaries for interpretability.
        try:
            if bool(get_config_value(config, "machine_learning.interpretability.grouped_outputs", True)):
                # SHAP importance may operate in transformed space (e.g., after feature selection/PCA).
                meta_df = build_feature_metadata(importance_df["feature"].astype(str).tolist(), config=config)
                merged = meta_df.merge(importance_df, on="feature", how="right")

                by_group_band = aggregate_importance(
                    merged, value_col="shap_importance", group_cols=["group", "band"]
                )
                if not by_group_band.empty:
                    by_group_band.to_csv(results_dir / "shap_importance_by_group_band.tsv", sep="\t", index=False)

                by_group_band_roi = aggregate_importance(
                    merged, value_col="shap_importance", group_cols=["group", "band", "roi"]
                )
                if not by_group_band_roi.empty:
                    by_group_band_roi.to_csv(results_dir / "shap_importance_by_group_band_roi.tsv", sep="\t", index=False)
        except Exception as exc:
            logger.debug("Grouped SHAP-importance export failed: %s", exc)

        _maybe_generate_mode_plots(mode="shap", results_dir=results_dir, logger=logger, config=config)
        
        return output_path
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def _run_uncertainty_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Any,
    seed: int,
    alpha: float,
    results_dir: Path,
    logger: logging.Logger,
    model_name: str = "elasticnet",
) -> Optional[Path]:
    """Run conformal prediction for uncertainty quantification."""
    from eeg_pipeline.analysis.machine_learning.uncertainty import (
        compute_prediction_intervals,
    )
    from sklearn.model_selection import LeaveOneGroupOut
    
    _resolved_model, base_pipe, param_grid = _build_regression_model_spec(
        model_name,
        seed=seed,
        config=config,
    )
    harmonization_mode = str(
        get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")
    ).strip().lower()
    inner_splits = int(get_config_value(config, "machine_learning.cv.inner_splits", 5))
    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(X, y, groups))
    
    all_intervals = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        X_train, X_test, _ = apply_fold_feature_harmonization(
            X_train,
            X_test,
            groups_train,
            harmonization_mode,
        )

        try:
            tuned_model = _fit_tuned_regression_estimator(
                base_pipe=clone(base_pipe),
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                inner_splits=inner_splits,
                seed=seed + fold_idx,
                logger=logger,
                fold_info=f"uncertainty fold {fold_idx}",
            )
            result = compute_prediction_intervals(
                model=clone(tuned_model),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                alpha=alpha,
                method="cv_plus",
                groups=groups_train,
                seed=seed + fold_idx,
            )
            result.compute_coverage(y_test)
            
            all_intervals.append({
                "fold": fold_idx,
                "y_test": y_test,
                "y_pred": result.y_pred,
                "lower": result.lower,
                "upper": result.upper,
                "coverage": result.coverage,
                "mean_width": result.mean_width,
                "subject_id": np.asarray(groups[test_idx], dtype=object),
                "test_idx": np.asarray(test_idx, dtype=int),
            })
        except Exception as e:
            logger.warning(f"Fold {fold_idx} uncertainty failed: {e}")
            continue
    
    if not all_intervals:
        logger.warning("No uncertainty results")
        return None
    n_folds_requested = int(len(fold_splits))
    n_folds_completed = int(len(all_intervals))
    completion_rate = float(n_folds_completed / max(n_folds_requested, 1))
    min_completion = float(
        get_config_value(config, "machine_learning.analysis.uncertainty.min_valid_fold_fraction", 0.8)
    )
    if completion_rate < min_completion:
        raise RuntimeError(
            "Insufficient valid uncertainty folds: "
            f"completed={n_folds_completed}/{n_folds_requested} "
            f"(rate={completion_rate:.3f} < required {min_completion:.3f})."
        )
    
    y_pred_all = np.concatenate([r["y_pred"] for r in all_intervals])
    lower_all = np.concatenate([r["lower"] for r in all_intervals])
    upper_all = np.concatenate([r["upper"] for r in all_intervals])
    y_test_all = np.concatenate([r["y_test"] for r in all_intervals])
    subject_all = np.concatenate([np.asarray(r["subject_id"], dtype=object) for r in all_intervals])
    test_idx_all = np.concatenate([np.asarray(r["test_idx"], dtype=int) for r in all_intervals])
    fold_all = np.concatenate(
        [
            np.full(len(np.asarray(r["y_pred"])), int(r["fold"]), dtype=int)
            for r in all_intervals
        ]
    )

    coverage = np.mean((y_test_all >= lower_all) & (y_test_all <= upper_all))
    mean_width = np.mean(upper_all - lower_all)

    intervals_df = pd.DataFrame({
        "fold": fold_all,
        "subject_id": subject_all,
        "test_index": test_idx_all,
        "y_pred": y_pred_all,
        "lower": lower_all,
        "upper": upper_all,
        "y_true": y_test_all,
        "in_interval": (y_test_all >= lower_all) & (y_test_all <= upper_all),
    })
    intervals_df["width"] = intervals_df["upper"] - intervals_df["lower"]
    per_subject_df = (
        intervals_df.groupby("subject_id", as_index=False)
        .agg(
            coverage=("in_interval", "mean"),
            mean_width=("width", "mean"),
            n_trials=("in_interval", "size"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    output_path = results_dir / "prediction_intervals.tsv"
    intervals_df.to_csv(output_path, sep="\t", index=False)
    per_subject_df.to_csv(results_dir / "per_subject_uncertainty.tsv", sep="\t", index=False)

    metrics = {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": float(coverage),
        "mean_width": float(mean_width),
        "n_folds": len(all_intervals),
        "n_folds_requested": n_folds_requested,
        "n_folds_completed": n_folds_completed,
        "valid_fold_fraction": completion_rate,
        "min_valid_fold_fraction": min_completion,
        "subject_level": {
            "mean_coverage": float(per_subject_df["coverage"].mean()) if len(per_subject_df) else np.nan,
            "std_coverage": float(per_subject_df["coverage"].std()) if len(per_subject_df) > 1 else np.nan,
            "mean_width": float(per_subject_df["mean_width"].mean()) if len(per_subject_df) else np.nan,
            "n_subjects": int(len(per_subject_df)),
        },
    }
    with open(results_dir / "uncertainty_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Uncertainty: coverage={coverage:.1%}, mean_width={mean_width:.3f}")
    _maybe_generate_mode_plots(mode="uncertainty", results_dir=results_dir, logger=logger, config=config)
    logger.info(f"Saved uncertainty to {output_path}")
    
    return output_path
