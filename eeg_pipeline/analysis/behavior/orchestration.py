"""Behavior pipeline orchestration.

This module contains the implementation of the behavior pipeline stages.
The pipeline layer (`eeg_pipeline.pipelines.behavior`) should remain a thin wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.config.loader import get_config_value


def combine_features(ctx: BehaviorContext) -> pd.DataFrame:
    def _signature() -> str:
        parts = []
        for name, df in ctx.iter_feature_tables():
            if df is None or df.empty:
                continue
            col_blob = ",".join(str(c) for c in df.columns)
            parts.append(f"{name}:{df.shape[0]}:{df.shape[1]}:{col_blob}")
        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    signature = _signature()
    if ctx._combined_features_df is not None and ctx._combined_features_signature == signature:
        return ctx._combined_features_df

    dfs = []
    base_index = None
    for name, df in ctx.iter_feature_tables():
        if df is not None and not df.empty:
            if base_index is None:
                base_index = df.index
            elif not df.index.equals(base_index):
                msg = (
                    f"Feature index mismatch for {name}: expected alignment of "
                    f"{len(base_index)} rows."
                )
                ctx.logger.error(msg)
                raise ValueError(msg)
            prefix = f"{name}_"
            df = df.rename(columns={c: c if str(c).startswith(prefix) else f"{prefix}{c}" for c in df.columns})

            if df.columns.duplicated().any():
                dupes = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
                msg = f"Duplicate feature columns within {name}: {dupes}"
                ctx.logger.error(msg)
                raise ValueError(msg)

            if dfs:
                existing_cols = pd.Index([])
                for prev in dfs:
                    existing_cols = existing_cols.append(prev.columns)
                overlap = existing_cols.intersection(df.columns)
                if not overlap.empty:
                    msg = f"Duplicate feature columns across tables: {[str(c) for c in overlap.tolist()]}"
                    ctx.logger.error(msg)
                    raise ValueError(msg)
            dfs.append(df)

    combined = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
    if not combined.empty and combined.columns.duplicated().any():
        dupes = [str(c) for c in combined.columns[combined.columns.duplicated()].unique()]
        msg = f"Duplicate feature columns after combining: {dupes}"
        ctx.logger.error(msg)
        raise ValueError(msg)
    ctx._combined_features_df = combined
    ctx._combined_features_signature = signature
    return combined


def run_combine_features_utility(
    subjects: List[str],
    categories: List[str],
    deriv_root: Path,
    config: Any,
    logger: logging.Logger,
    progress: Optional[Any] = None,
) -> int:
    """Utility to manually combine individual feature files for multiple subjects."""
    from eeg_pipeline.infra.paths import deriv_features_path
    from eeg_pipeline.infra.tsv import write_tsv, read_tsv, read_parquet
    from eeg_pipeline.utils.data.feature_io import find_connectivity_features_path
    
    mapping = {
        "power": "features_power.tsv",
        "connectivity": "features_connectivity.parquet",
        "aperiodic": "features_aperiodic.tsv",
        "itpc": "features_itpc.tsv",
        "pac": "features_pac.tsv",
        "complexity": "features_complexity.tsv",
        "erds": "features_erds.tsv",
        "spectral": "features_spectral.tsv",
        "ratios": "features_ratios.tsv",
        "asymmetry": "features_asymmetry.tsv",
        "quality": "features_quality.tsv",
        "temporal": "features_temporal.tsv",
    }
    
    count = 0
    if progress:
        progress.start("combine_features", subjects)
        
    for subj in subjects:
        if progress:
            progress.subject_start(f"sub-{subj}")
            
        features_dir = deriv_features_path(deriv_root, subj)
        if not features_dir.exists():
            logger.warning(f"Feature directory missing for sub-{subj}")
            if progress:
                progress.subject_done(f"sub-{subj}", success=False)
            continue
            
        dfs = []
        for cat in categories:
            fname = mapping.get(cat)
            if not fname:
                # Try default fallback
                fname = f"features_{cat}.tsv"
            
            fpath = features_dir / fname
            if cat == "connectivity":
                fpath = find_connectivity_features_path(deriv_root, subj)
                
            if not fpath.exists():
                logger.debug(f"Skipping {cat}: {fpath.name} not found")
                continue
            
            try:
                if fpath.suffix == ".parquet":
                    df = read_parquet(fpath)
                else:
                    df = read_tsv(fpath)
                
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {fpath.name}: {e}")

        if not dfs:
            logger.warning(f"No features found to combine for sub-{subj}")
            if progress:
                progress.subject_done(f"sub-{subj}", success=False)
            continue
            
        try:
            combined = pd.concat(dfs, axis=1)
            # Basic sanity check
            if combined.columns.duplicated().any():
                dupes = combined.columns[combined.columns.duplicated()].unique().tolist()
                logger.warning(f"Duplicate columns found while combining features for sub-{subj}: {dupes[:10]}...; keep first occurrences")
                combined = combined.loc[:, ~combined.columns.duplicated()]
            
            out_path = features_dir / "features_all.tsv"
            write_tsv(combined, out_path)
            logger.info(f"Successfully created {out_path.name} ({len(combined.columns)} columns)")
            count += 1
            if progress:
                progress.subject_done(f"sub-{subj}", success=True)
        except Exception as e:
            logger.error(f"Failed to combine features for sub-{subj}: {e}")
            if progress:
                progress.subject_done(f"sub-{subj}", success=False)

    if progress:
        progress.complete(success=(count > 0))
        
    return count


def add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (active-baseline) once per context."""
    if ctx._change_scores_added or not ctx.compute_change_scores:
        return

    from eeg_pipeline.utils.analysis.stats.correlation import compute_change_features

    def _augment(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        change_df = compute_change_features(df)
        if change_df is None or change_df.empty:
            return df
        new_cols = [c for c in change_df.columns if c not in df.columns]
        return pd.concat([df, change_df[new_cols]], axis=1) if new_cols else df

    ctx.power_df = _augment(ctx.power_df)
    ctx.connectivity_df = _augment(ctx.connectivity_df)
    ctx.aperiodic_df = _augment(ctx.aperiodic_df)
    ctx.itpc_df = _augment(ctx.itpc_df)
    ctx.pac_df = _augment(ctx.pac_df)
    ctx.complexity_df = _augment(ctx.complexity_df)
    ctx._change_scores_added = True


def stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False

    ctx.logger.info(f"Loaded {ctx.n_trials} trials")
    return True


def stage_correlate(ctx: BehaviorContext, config: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from eeg_pipeline.analysis.behavior.api import (
        run_pain_sensitivity_correlations,
        run_unified_feature_correlations,
    )

    add_change_scores(ctx)

    ctx.logger.info("Running unified feature correlator...")
    result = run_unified_feature_correlations(ctx)
    corr_df = result.to_dataframe()
    if corr_df is None:
        corr_df = pd.DataFrame()
    if not corr_df.empty:
        ctx.logger.info(f"Obtained {len(corr_df)} correlation results from unified correlator")

    psi_df = pd.DataFrame()
    if config.compute_pain_sensitivity and ctx.temperature is not None:
        features = combine_features(ctx)
        if not features.empty:
            robust_method = get_config_value(ctx.config, "behavior_analysis.robust_correlation", None)
            if robust_method is not None:
                robust_method = str(robust_method).strip().lower() or None
            psi_df = run_pain_sensitivity_correlations(
                features_df=features,
                ratings=ctx.targets,
                temperatures=ctx.temperature,
                method=config.method,
                robust_method=robust_method,
                min_samples=config.min_samples,
                logger=ctx.logger,
            )

    return corr_df, psi_df


def build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    qc: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "n_trials": ctx.n_trials,
        "has_temperature": ctx.has_temperature,
        "temperature_column": ctx.temperature_column,
        "group_column": ctx.group_column,
    }

    if ctx.data_qc:
        qc["data_qc"] = ctx.data_qc

    if ctx.targets is not None:
        s = pd.to_numeric(ctx.targets, errors="coerce")
        qc["rating"] = {
            "n_non_nan": int(s.notna().sum()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
        }

    if ctx.temperature is not None:
        t = pd.to_numeric(ctx.temperature, errors="coerce")
        qc["temperature"] = {
            "n_non_nan": int(t.notna().sum()),
            "min": float(t.min()) if t.notna().any() else np.nan,
            "max": float(t.max()) if t.notna().any() else np.nan,
            "mean": float(t.mean()) if t.notna().any() else np.nan,
            "std": float(t.std(ddof=1)) if t.notna().sum() > 1 else np.nan,
        }

    if ctx.targets is not None and ctx.temperature is not None:
        s = pd.to_numeric(ctx.targets, errors="coerce")
        t = pd.to_numeric(ctx.temperature, errors="coerce")
        valid = s.notna() & t.notna()
        if int(valid.sum()) >= 3:
            r, p = compute_correlation(s[valid].values, t[valid].values, method="spearman")
            qc["rating_temperature_sanity"] = {
                "method": "spearman",
                "n": int(valid.sum()),
                "r": float(r) if np.isfinite(r) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            }

    if ctx.aligned_events is not None and ctx.targets is not None:
        from eeg_pipeline.analysis.behavior.api import split_by_condition

        try:
            pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
                ctx.aligned_events, ctx.config, ctx.logger
            )
        except Exception as exc:
            qc["pain_vs_nonpain"] = {
                "status": "failed",
                "error": str(exc),
            }
        else:
            if int(n_pain) > 0 or int(n_nonpain) > 0:
                s = pd.to_numeric(ctx.targets, errors="coerce")
                pain_ratings = s[pain_mask] if len(pain_mask) == len(s) else pd.Series(dtype=float)
                nonpain_ratings = s[nonpain_mask] if len(nonpain_mask) == len(s) else pd.Series(dtype=float)
                qc["pain_vs_nonpain"] = {
                    "status": "ok",
                    "n_pain": int(n_pain),
                    "n_nonpain": int(n_nonpain),
                    "mean_rating_pain": float(pain_ratings.mean()) if pain_ratings.notna().any() else np.nan,
                    "mean_rating_nonpain": float(nonpain_ratings.mean()) if nonpain_ratings.notna().any() else np.nan,
                    "mean_rating_difference_pain_minus_nonpain": (
                        float(pain_ratings.mean() - nonpain_ratings.mean())
                        if pain_ratings.notna().any() and nonpain_ratings.notna().any()
                        else np.nan
                    ),
                }

    if ctx.covariates_df is not None and not ctx.covariates_df.empty:
        cov = ctx.covariates_df
        qc["covariates"] = {
            "n_covariates": int(cov.shape[1]),
            "columns": [str(c) for c in cov.columns],
            "missing_fraction_by_column": {
                str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
            },
        }

    feature_counts: Dict[str, int] = {}
    for name, df in ctx.iter_feature_tables():
        if df is None or df.empty:
            continue
        feature_counts[name] = int(df.shape[1])
    qc["feature_counts"] = feature_counts

    return qc


def _ensure_method_columns(
    df: pd.DataFrame,
    method: str,
    robust_method: Optional[str],
    method_label: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "method" not in df.columns:
        df["method"] = method
    if "robust_method" not in df.columns:
        df["robust_method"] = robust_method
    if "method_label" not in df.columns:
        df["method_label"] = method_label
    return df


_STANDARD_COLUMNS = (
    "analysis_type",
    "feature_id",
    "feature_type",
    "target",
    "method",
    "robust_method",
    "method_label",
    "n",
    "r",
    "p_raw",
    "p_primary",
    "p_fdr",
    "notes",
)


def _infer_feature_type(feature: str, config: Any) -> str:
    from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry
    try:
        registry = get_feature_registry(config)
        ftype, _, _ = classify_feature(feature, include_subtype=False, registry=registry)
        return ftype
    except Exception:
        return "unknown"


def _build_normalized_records(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    method = pipeline_config.method
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        for _, row in corr_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            target = row.get("target", "rating")
            p_raw = row.get("p_raw", row.get("p_value", row.get("p")))
            p_primary = row.get("p_primary", p_raw)
            p_fdr = row.get("p_fdr", row.get("q_value"))
            r_val = row.get("r_primary", row.get("r"))
            records.append({
                "analysis_type": "correlations",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": target,
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": row.get("n"),
                "r": r_val,
                "p_raw": p_raw,
                "p_primary": p_primary,
                "p_fdr": p_fdr,
                "notes": None,
            })

    psi_df = getattr(results, "pain_sensitivity", None)
    if psi_df is not None and not psi_df.empty:
        for _, row in psi_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            p_raw = row.get("p_psi", row.get("p_value"))
            r_val = row.get("r_psi", row.get("r"))
            records.append({
                "analysis_type": "pain_sensitivity",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": row.get("target", "pain_sensitivity"),
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": row.get("n"),
                "r": r_val,
                "p_raw": p_raw,
                "p_primary": p_raw,
                "p_fdr": row.get("p_fdr", row.get("q_value")),
                "notes": None,
            })

    cond_df = getattr(results, "condition_effects", None)
    if cond_df is not None and not cond_df.empty:
        for _, row in cond_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            n_pain = row.get("n_pain")
            n_nonpain = row.get("n_nonpain")
            n_total = None
            if pd.notna(n_pain) and pd.notna(n_nonpain):
                n_total = int(n_pain) + int(n_nonpain)
            records.append({
                "analysis_type": "condition_effects",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": "pain_vs_nonpain",
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": n_total,
                "r": row.get("hedges_g"),
                "p_raw": row.get("p_value"),
                "p_primary": row.get("p_value"),
                "p_fdr": row.get("q_value"),
                "notes": "hedges_g",
            })

    med_df = getattr(results, "mediation", None)
    if med_df is not None and not med_df.empty:
        for _, row in med_df.iterrows():
            feature = row.get("mediator")
            records.append({
                "analysis_type": "mediation",
                "feature_id": feature,
                "feature_type": "mediator",
                "target": "rating",
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": None,
                "r": row.get("indirect_effect"),
                "p_raw": row.get("sobel_p"),
                "p_primary": row.get("sobel_p"),
                "p_fdr": None,
                "notes": "indirect_effect",
            })

    mixed_df = getattr(results, "mixed_effects", None)
    if mixed_df is not None and not mixed_df.empty:
        for _, row in mixed_df.iterrows():
            feature = row.get("feature")
            behavior = row.get("behavior", "rating")
            records.append({
                "analysis_type": "mixed_effects",
                "feature_id": feature,
                "feature_type": row.get("feature_type", "mixed_effects"),
                "target": behavior,
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": row.get("n_observations"),
                "r": row.get("fixed_effect"),
                "p_raw": row.get("fixed_p"),
                "p_primary": row.get("fixed_p"),
                "p_fdr": row.get("fixed_p_fdr"),
                "notes": "fixed_effect",
            })

    return records


def _write_normalized_results(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
) -> Optional[Path]:
    records = _build_normalized_records(ctx, pipeline_config, results)
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.reindex(columns=_STANDARD_COLUMNS)
    
    # Build feature suffix from selected feature files or categories
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    feature_suffix = "_" + "_".join(sorted(feature_files)) if feature_files else ""
    
    path = ctx.stats_dir / f"normalized_results{feature_suffix}.tsv"
    from eeg_pipeline.infra.tsv import write_tsv
    write_tsv(df, path)
    return path


def _summarize_covariates_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    cov = ctx.covariates_df
    if cov is None or cov.empty:
        return {"status": "empty"}
    return {
        "status": "ok",
        "columns": [str(c) for c in cov.columns],
        "missing_fraction_by_column": {
            str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
        },
    }


def write_analysis_metadata(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
    outputs_manifest: Optional[Path] = None,
) -> None:
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": pipeline_config.method,
        "method_label": method_label,
        "robust_method": robust_method,
        "min_samples": pipeline_config.min_samples,
        "control_temperature": pipeline_config.control_temperature,
        "control_trial_order": pipeline_config.control_trial_order,
        "compute_change_scores": pipeline_config.compute_change_scores,
        "compute_pain_sensitivity": pipeline_config.compute_pain_sensitivity,
        "compute_reliability": pipeline_config.compute_reliability,
        "n_permutations": pipeline_config.n_permutations,
        "fdr_alpha": pipeline_config.fdr_alpha,
        "n_trials": ctx.n_trials,
        "statistics_config": {
            "method": pipeline_config.method,
            "robust_method": robust_method,
            "method_label": method_label,
            "min_samples": pipeline_config.min_samples,
            "bootstrap": pipeline_config.bootstrap,
            "n_permutations": pipeline_config.n_permutations,
            "fdr_alpha": pipeline_config.fdr_alpha,
            "control_temperature": pipeline_config.control_temperature,
            "control_trial_order": pipeline_config.control_trial_order,
            "compute_change_scores": pipeline_config.compute_change_scores,
            "compute_reliability": pipeline_config.compute_reliability,
            "compute_bayes_factors": getattr(pipeline_config, "compute_bayes_factors", False),
            "compute_loso_stability": getattr(pipeline_config, "compute_loso_stability", False),
        },
        "outputs": {
            "has_correlations": bool(getattr(results, "correlations", None) is not None and not results.correlations.empty),
            "has_pain_sensitivity": bool(getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty),
            "has_condition_effects": bool(getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty),
        },
        "qc": build_behavior_qc(ctx),
    }

    payload["temperature_status"] = {
        "available": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        "control_enabled": bool(ctx.control_temperature),
    }
    if not payload["temperature_status"]["available"]:
        payload["temperature_status"]["reason"] = "missing_temperature"

    if not pipeline_config.compute_pain_sensitivity:
        payload["pain_sensitivity_status"] = "disabled"
    elif payload["temperature_status"]["available"]:
        payload["pain_sensitivity_status"] = "computed" if payload["outputs"]["has_pain_sensitivity"] else "skipped"
    else:
        payload["pain_sensitivity_status"] = "skipped_no_temperature"

    payload["covariates_qc"] = _summarize_covariates_qc(ctx)

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    if outputs_manifest is not None:
        payload["outputs_manifest"] = str(outputs_manifest)

    if ctx.data_qc:
        payload["data_qc"] = ctx.data_qc

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        df = corr_df
        partial_cols = [
            ("p_partial_cov", "partial_cov"),
            ("p_partial_temp", "partial_temp"),
            ("p_partial_cov_temp", "partial_cov_temp"),
        ]
        partial_ok: Dict[str, Any] = {}
        for col, label in partial_cols:
            if col not in df.columns:
                continue
            pvals = pd.to_numeric(df[col], errors="coerce")
            partial_ok[label] = {
                "n_non_nan": int(pvals.notna().sum()),
                "fraction_non_nan": float(pvals.notna().mean()) if len(pvals) else np.nan,
            }
        if partial_ok:
            payload["partial_correlation_feasibility"] = partial_ok

        if "p_primary_source" in df.columns and df["p_primary_source"].notna().any():
            payload["primary_test_source_counts"] = df["p_primary_source"].fillna("unknown").value_counts().to_dict()

        if "within_family_p_kind" in df.columns and df["within_family_p_kind"].notna().any():
            payload["within_family_p_kind_counts"] = df["within_family_p_kind"].fillna("unknown").value_counts().to_dict()

    (ctx.stats_dir / "analysis_metadata.json").write_text(json.dumps(payload, indent=2, default=str))


def stage_condition(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    from eeg_pipeline.analysis.behavior.api import split_by_condition, compute_condition_effects

    if ctx.aligned_events is None:
        return pd.DataFrame()

    fail_fast = get_config_value(ctx.config, "behavior_analysis.condition.fail_fast", True)

    try:
        pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(ctx.aligned_events, ctx.config, ctx.logger)
    except Exception as exc:
        if fail_fast:
            raise
        ctx.logger.warning(f"Condition split failed; skipping condition effects: {exc}")
        return pd.DataFrame()

    if n_pain == 0 and n_nonpain == 0:
        msg = "Condition split produced zero trials; check pain coding and config event_columns.pain_binary"
        if fail_fast:
            raise ValueError(msg)
        ctx.logger.warning(msg)
        return pd.DataFrame()

    min_samples = max(config.min_samples, ctx.get_min_samples("default"))
    if n_pain < min_samples or n_nonpain < min_samples:
        ctx.logger.warning(
            f"Insufficient trials: {n_pain} pain, {n_nonpain} non-pain (need >= {min_samples})"
        )
        return pd.DataFrame()

    features = combine_features(ctx)
    if features.empty:
        return pd.DataFrame()

    return compute_condition_effects(
        features,
        pain_mask,
        nonpain_mask,
        min_samples=min_samples,
        fdr_alpha=config.fdr_alpha,
        logger=ctx.logger,
        n_jobs=config.n_jobs,
        config=ctx.config,
    )


def stage_temporal(ctx: BehaviorContext) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    from eeg_pipeline.analysis.behavior.api import (
        compute_time_frequency_from_context,
        compute_temporal_from_context,
        run_power_topomap_correlations,
    )

    status = {"time_frequency": "success", "temporal": "success", "topomap": "success"}
    tf_results = None
    temporal_results = None

    ctx.logger.info("Computing time-frequency correlations...")
    try:
        tf_results = compute_time_frequency_from_context(ctx)
    except Exception as exc:
        status["time_frequency"] = f"failed: {exc}"
        ctx.logger.error(f"Time-frequency correlations failed: {exc}")

    ctx.logger.info("Computing temporal correlations by condition...")
    try:
        temporal_results = compute_temporal_from_context(ctx)
    except Exception as exc:
        status["temporal"] = f"failed: {exc}"
        ctx.logger.error(f"Temporal correlations failed: {exc}")

    try:
        run_power_topomap_correlations(
            subject=ctx.subject,
            task=ctx.task,
            power_df=ctx.power_df,
            temperature=ctx.temperature,
            epochs_info=ctx.epochs_info,
            stats_dir=ctx.stats_dir,
            config=ctx.config,
            logger=ctx.logger,
            use_spearman=ctx.use_spearman,
            rng=ctx.rng,
            bootstrap=ctx.bootstrap,
            n_perm=ctx.n_perm,
        )
    except Exception as exc:
        status["topomap"] = f"failed: {exc}"
        ctx.logger.error(f"Topomap correlations failed: {exc}")

    failed = {k: v for k, v in status.items() if v != "success"}
    if failed:
        ctx.logger.info(f"Temporal stage status: {failed}")
    
    return tf_results, temporal_results


def stage_cluster(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context

    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations

    try:
        results = run_cluster_test_from_context(ctx)
        return results if results else {"status": "completed"}
    except Exception as exc:
        ctx.logger.warning(f"Cluster tests failed: {exc}")
        return {"status": "failed", "error": str(exc)}


_METADATA_COLUMNS = frozenset(["subject", "epoch", "trial", "condition", "temperature", "rating"])


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns excluding metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLUMNS]


def stage_advanced(ctx: BehaviorContext, config: Any, results: Any) -> None:
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis, run_multilevel_correlation_analysis

    features = combine_features(ctx).copy()
    if features.empty:
        return

    # Add targets and temperature to the dataframe for analysis
    if ctx.targets is not None:
        features["rating"] = ctx.targets.values
    if ctx.temperature is not None:
        features["temperature"] = ctx.temperature.values
    
    features["subject"] = ctx.subject

    feature_cols = _get_feature_columns(features)
    if not feature_cols:
        return

    if config.run_mediation:
        if "temperature" in features.columns and "rating" in features.columns:
            ctx.logger.info("Running mediation analysis...")
            n_bootstrap = get_config_value(ctx.config, "behavior_analysis.mediation.n_bootstrap", 1000)
            variances = features[feature_cols].var()
            mediators = variances.nlargest(20).index.tolist()
            results.mediation = run_mediation_analysis(
                features,
                "temperature",
                mediators,
                "rating",
                n_bootstrap=n_bootstrap,
            )
        else:
            ctx.logger.warning("Skipping mediation: 'temperature' or 'rating' missing.")

    if config.run_mixed_effects and "subject" in features.columns and "rating" in features.columns:
        ctx.logger.info("Running mixed-effects analysis...")
        results.mixed_effects = run_multilevel_correlation_analysis(
            features,
            feature_cols[:50],
            "rating",
            "subject",
            config=ctx.config,
        )


def stage_validate(ctx: BehaviorContext, config: Any, results: Optional[Any] = None) -> None:
    from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr
    from eeg_pipeline.utils.analysis.stats.reliability import compute_hierarchical_fdr_summary

    fdr_alpha = config.fdr_alpha

    # Determine which analysis files to include in FDR based on current computation flags
    patterns = []
    
    # 1. Condition effects
    if getattr(config, "run_condition_comparison", False):
        patterns.append("condition_effects*.tsv")
        
    # 2. Mediation and Mixed Effects
    if getattr(config, "run_mediation", False):
        patterns.append("mediation*.tsv")
    if getattr(config, "run_mixed_effects", False):
        patterns.append("mixed_effects*.tsv")

    # 3. Correlations and Pain Sensitivity
    # We include these if they were requested. 
    # Note: ctx.selected_feature_files tracks which feature categories were loaded for correlation-style analyses.
    include_cats = ctx.selected_feature_files or ctx.feature_categories
    
    # Check if correlations or pain sensitivity were requested (using heuristic or explicit flags)
    # The orchestration layer usually has these flags in config.
    run_corrs = getattr(config, "run_correlations", True) # Default to True if not explicit, but check config
    run_psi = getattr(config, "compute_pain_sensitivity", False)

    if include_cats:
        for cat in include_cats:
            if run_corrs:
                patterns.append(f"corr_stats_{cat}_vs_*.tsv")
            if run_psi:
                patterns.append(f"pain_sensitivity_{cat}*.tsv")
        
        # Always include 'all_features' and base 'pain_sensitivity' if they might have been written
        if run_corrs:
            patterns.append("corr_stats_all_features_vs_*.tsv")
        if run_psi:
            patterns.append("pain_sensitivity_*.tsv")
    elif run_corrs or run_psi:
        # Fallback if no specific categories selected
        if run_corrs:
            patterns.append("corr_stats_*.tsv")
        if run_psi:
            patterns.append("pain_sensitivity_*.tsv")

    # 4. Temporal and Time-Frequency
    if getattr(config, "run_temporal_correlations", False):
        patterns.extend([
            "corr_stats_tf_*.tsv",
            "corr_stats_temporal_*.tsv",
            "*_topomap_*_correlations_*.tsv"
        ])

    # 5. Cluster Tests
    if getattr(config, "run_cluster_tests", False):
        patterns.append("cluster_results_*.tsv")

    # Remove duplicates
    patterns = list(set(patterns))
    
    # If no specific patterns were identified but we're in validation, 
    # it means all specific modules were disabled. 
    # We only use the broad fallback if absolutely no patterns were added.
    if not patterns:
        patterns = ["corr_stats_*.tsv", "condition_effects.tsv", "pain_sensitivity_*.tsv"]

    ctx.logger.info("Applying global FDR correction...")
    summary = apply_global_fdr(
        ctx.stats_dir,
        alpha=fdr_alpha,
        logger=ctx.logger,
        include_glob=patterns,
    )

    # 4. If results object is provided, update its dataframes with global FDR info
    if results is not None:
        from eeg_pipeline.infra.tsv import read_tsv
        from eeg_pipeline.utils.analysis.stats.fdr import select_p_column_for_fdr
        
        # Mapping of result types to files
        mapping = {
            "correlations": "correlations*.tsv",
            "pain_sensitivity": "pain_sensitivity*.tsv",
            "condition_effects": "condition_effects*.tsv",
            "tf": "corr_stats_tf_*.tsv",
            "temporal": "corr_stats_temporal_*.tsv",
            "cluster": "cluster_results_*.tsv",
            "mediation": "mediation*.tsv",
            "mixed_effects": "mixed_effects*.tsv",
        }
        
        for attr, pattern in mapping.items():
            current_value = getattr(results, attr, None)
            if current_value is None:
                continue
                
            match_files = list(ctx.stats_dir.glob(pattern))
            if not match_files:
                continue
            
            for fpath in match_files:
                df_disk = read_tsv(fpath)
                if df_disk is None or "q_global" not in df_disk.columns:
                    continue

                if isinstance(current_value, pd.DataFrame):
                    if current_value.empty: continue
                    merge_cols = [c for c in ["feature", "feature_id", "target", "mediator"] if c in df_disk.columns]
                    if merge_cols:
                        if "q_global" in current_value.columns:
                            current_value = current_value.drop(columns=["q_global"])
                        current_value = pd.merge(current_value, df_disk[merge_cols + ["q_global"]], on=merge_cols, how="left")
                        setattr(results, attr, current_value)
                    elif len(current_value) == len(df_disk):
                        current_value["q_global"] = df_disk["q_global"].values
                        setattr(results, attr, current_value)
                elif isinstance(current_value, dict):
                    # For cluster, find the band from filename if possible
                    if attr == "cluster":
                        band_name = fpath.stem.replace("cluster_results_", "")
                        if band_name in current_value and "cluster_records" in current_value[band_name]:
                            recs = current_value[band_name]["cluster_records"]
                            if len(recs) == len(df_disk):
                                for r, q in zip(recs, df_disk["q_global"]):
                                    r["q_global"] = q
                    else:
                        # For temporal/tf summary dicts
                        current_value["n_sig_fdr"] = int((df_disk["q_global"] < fdr_alpha).sum())
                        setattr(results, attr, current_value)

    ctx.logger.info("Computing hierarchical FDR summary...")
    try:
        hier_summary = compute_hierarchical_fdr_summary(
            ctx.stats_dir,
            alpha=fdr_alpha,
            config=ctx.config,
            include_glob=patterns,
        )
        if not hier_summary.empty:
            hier_path = ctx.stats_dir / "hierarchical_fdr_summary.tsv"
            hier_summary.to_csv(hier_path, sep="\t", index=False)
            ctx.logger.info(f"Hierarchical FDR summary saved to {hier_path}")
            
            for _, row in hier_summary.iterrows():
                ctx.logger.info(
                    f"  {row['analysis_type']}: {row['n_reject_within']}/{row['n_tests']} "
                    f"({row['pct_reject_within']:.1f}%) reject within-family"
                )
    except Exception as exc:
        ctx.logger.warning(f"Hierarchical FDR failed: {exc}")


def stage_export(ctx: BehaviorContext, pipeline_config: Any, results: Any) -> List[Path]:
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.infra.tsv import write_tsv

    ensure_dir(ctx.stats_dir)
    saved: List[Path] = []
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    method_suffix = f"_{method_label}" if method_label else ""
    
    # Build feature suffix from selected feature files or categories
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    feature_suffix = "_" + "_".join(sorted(feature_files)) if feature_files else ""

    if getattr(results, "correlations", None) is not None:
        results.correlations = _ensure_method_columns(
            results.correlations, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "pain_sensitivity", None) is not None:
        results.pain_sensitivity = _ensure_method_columns(
            results.pain_sensitivity, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "condition_effects", None) is not None:
        results.condition_effects = _ensure_method_columns(
            results.condition_effects, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "mediation", None) is not None:
        results.mediation = _ensure_method_columns(
            results.mediation, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "mixed_effects", None) is not None:
        results.mixed_effects = _ensure_method_columns(
            results.mixed_effects, pipeline_config.method, robust_method, method_label
        )

    if getattr(results, "correlations", None) is not None and not results.correlations.empty:
        path = ctx.stats_dir / f"correlations{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.correlations, path)
        saved.append(path)

    if getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty:
        path = ctx.stats_dir / f"pain_sensitivity{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.pain_sensitivity, path)
        saved.append(path)

    if getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty:
        path = ctx.stats_dir / f"condition_effects{feature_suffix}.tsv"
        write_tsv(results.condition_effects, path)
        saved.append(path)

    if getattr(results, "mediation", None) is not None and not results.mediation.empty:
        path = ctx.stats_dir / f"mediation{feature_suffix}.tsv"
        write_tsv(results.mediation, path)
        saved.append(path)

    if getattr(results, "mixed_effects", None) is not None and not results.mixed_effects.empty:
        path = ctx.stats_dir / f"mixed_effects{feature_suffix}.tsv"
        write_tsv(results.mixed_effects, path)
        saved.append(path)

    normalized_path = _write_normalized_results(ctx, pipeline_config, results)
    if normalized_path is not None:
        saved.append(normalized_path)

    summary = results.to_summary()
    (ctx.stats_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    saved.append(ctx.stats_dir / "summary.json")

    ctx.logger.info(f"Saved {len(saved)} output files")
    return saved


def _infer_output_kind(name: str) -> str:
    if name.startswith("corr_stats_"):
        return "correlation_stats"
    if name.startswith("correlations"):
        return "correlations"
    if name.startswith("pain_sensitivity"):
        return "pain_sensitivity"
    if name.startswith("condition_effects"):
        return "condition_effects"
    if name.startswith("mediation"):
        return "mediation"
    if name.startswith("mixed_effects"):
        return "mixed_effects"
    if name.startswith("normalized_results"):
        return "normalized_results"
    if name.startswith("feature_screening"):
        return "feature_screening"
    if name.startswith("summary"):
        return "summary"
    if name.startswith("analysis_metadata"):
        return "analysis_metadata"
    if name.startswith("time_frequency_correlation_data"):
        return "time_frequency"
    if name.startswith("temporal_correlations"):
        return "temporal_correlations"
    if name.startswith("hierarchical_fdr_summary"):
        return "hierarchical_fdr_summary"
    return "unknown"


def _count_rows(path: Path) -> Optional[int]:
    if path.suffix not in {".tsv", ".csv"}:
        return None
    try:
        with path.open("r") as f:
            header = f.readline()
            if not header:
                return 0
            return sum(1 for _ in f)
    except Exception:
        return None


def write_outputs_manifest(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    from datetime import datetime

    outputs = []
    for path in sorted(p for p in ctx.stats_dir.iterdir() if p.is_file() and p.name != "outputs_manifest.json"):
        outputs.append({
            "name": path.name,
            "path": str(path),
            "kind": _infer_output_kind(path.name),
            "rows": _count_rows(path),
            "size_bytes": int(path.stat().st_size),
            "method_label": pipeline_config.method_label,
        })

    feature_types = [name for name, df in ctx.iter_feature_tables() if df is not None and not df.empty]

    payload = {
        "subject": ctx.subject,
        "task": ctx.task,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": pipeline_config.method,
        "robust_method": pipeline_config.robust_method,
        "method_label": pipeline_config.method_label,
        "n_trials": ctx.n_trials,
        "feature_types": feature_types,
        "feature_categories": ctx.feature_categories or [],
        "feature_files": ctx.selected_feature_files or [],
        "targets": {
            "rating": bool(ctx.targets is not None and ctx.targets.notna().any()) if ctx.targets is not None else False,
            "temperature": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        },
        "covariates_qc": ctx.data_qc.get("covariates_qc", {}),
        "outputs": outputs,
    }

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    path = ctx.stats_dir / "outputs_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
