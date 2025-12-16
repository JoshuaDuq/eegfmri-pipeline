"""Behavior pipeline orchestration.

This module contains the implementation of the behavior pipeline stages.
The pipeline layer (`eeg_pipeline.pipelines.behavior`) should remain a thin wrapper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation


def combine_features(ctx: BehaviorContext) -> pd.DataFrame:
    dfs = []
    base_index = None
    for name, df in [
        ("power", ctx.power_df),
        ("connectivity", ctx.connectivity_df),
        ("microstates", ctx.microstates_df),
        ("aperiodic", ctx.aperiodic_df),
        ("itpc", ctx.itpc_df),
        ("pac", ctx.pac_df),
        ("precomputed", ctx.precomputed_df),
    ]:
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
            if not any(str(c).startswith(name) for c in df.columns):
                df = df.add_prefix(f"{name}_")
            dfs.append(df)

    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()


def add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (plateau-baseline) once per context."""
    if getattr(ctx, "_change_scores_added", False) or not getattr(ctx, "compute_change_scores", False):
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
    ctx.microstates_df = _augment(ctx.microstates_df)
    ctx.aperiodic_df = _augment(ctx.aperiodic_df)
    ctx.itpc_df = _augment(ctx.itpc_df)
    ctx.pac_df = _augment(ctx.pac_df)
    ctx.precomputed_df = _augment(ctx.precomputed_df)
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
    corr_df = result.to_dataframe() if getattr(result, "dataframe", None) is not None else result.to_dataframe()
    if corr_df is None:
        corr_df = pd.DataFrame()
    if not corr_df.empty:
        ctx.logger.info(f"Obtained {len(corr_df)} correlation results from unified correlator")

    psi_df = pd.DataFrame()
    if getattr(config, "compute_pain_sensitivity", False) and ctx.temperature is not None:
        features = combine_features(ctx)
        if not features.empty:
            psi_df = run_pain_sensitivity_correlations(
                features_df=features,
                ratings=ctx.targets,
                temperatures=ctx.temperature,
                method=getattr(config, "method", "spearman"),
                min_samples=int(getattr(config, "min_samples", 10)),
                logger=ctx.logger,
            )

    return corr_df, psi_df


def build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    qc: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "n_trials": int(ctx.n_trials),
        "has_temperature": bool(ctx.has_temperature),
        "temperature_column": ctx.temperature_column,
        "group_column": getattr(ctx, "group_column", None),
    }

    if getattr(ctx, "data_qc", None):
        qc["data_qc"] = getattr(ctx, "data_qc")

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

    if getattr(ctx, "covariates_df", None) is not None and not ctx.covariates_df.empty:
        cov = ctx.covariates_df
        qc["covariates"] = {
            "n_covariates": int(cov.shape[1]),
            "columns": [str(c) for c in cov.columns],
            "missing_fraction_by_column": {
                str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
            },
        }

    feature_counts: Dict[str, int] = {}
    for name, df in [
        ("power", ctx.power_df),
        ("connectivity", ctx.connectivity_df),
        ("microstates", ctx.microstates_df),
        ("aperiodic", ctx.aperiodic_df),
        ("itpc", ctx.itpc_df),
        ("pac", ctx.pac_df),
        ("precomputed", ctx.precomputed_df),
    ]:
        if df is None or df.empty:
            continue
        feature_counts[name] = int(df.shape[1])
    qc["feature_counts"] = feature_counts

    return qc


def write_analysis_metadata(ctx: BehaviorContext, pipeline_config: Any, results: Any) -> None:
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": getattr(pipeline_config, "method", None),
        "min_samples": getattr(pipeline_config, "min_samples", None),
        "control_temperature": bool(getattr(pipeline_config, "control_temperature", True)),
        "control_trial_order": bool(getattr(pipeline_config, "control_trial_order", True)),
        "compute_change_scores": bool(getattr(pipeline_config, "compute_change_scores", True)),
        "compute_pain_sensitivity": bool(getattr(pipeline_config, "compute_pain_sensitivity", True)),
        "compute_reliability": bool(getattr(pipeline_config, "compute_reliability", True)),
        "n_permutations": int(getattr(pipeline_config, "n_permutations", 0)),
        "fdr_alpha": float(getattr(pipeline_config, "fdr_alpha", 0.05)),
        "n_trials": int(ctx.n_trials),
        "outputs": {
            "has_correlations": bool(getattr(results, "correlations", None) is not None and not results.correlations.empty),
            "has_pain_sensitivity": bool(getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty),
            "has_condition_effects": bool(getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty),
        },
        "qc": build_behavior_qc(ctx),
    }

    if getattr(ctx, "data_qc", None):
        payload["data_qc"] = getattr(ctx, "data_qc")

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

    try:
        pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(ctx.aligned_events, ctx.config, ctx.logger)
    except Exception as exc:
        fail_fast = bool(getattr(ctx.config, "get", lambda *_: True)("behavior_analysis.condition.fail_fast", True))
        if fail_fast:
            raise
        ctx.logger.warning(f"Condition split failed; skipping condition effects: {exc}")
        return pd.DataFrame()

    if n_pain == 0 and n_nonpain == 0:
        fail_fast = bool(getattr(ctx.config, "get", lambda *_: True)("behavior_analysis.condition.fail_fast", True))
        msg = "Condition split produced zero trials; check pain coding and config event_columns.pain_binary"
        if fail_fast:
            raise ValueError(msg)
        ctx.logger.warning(msg)
        return pd.DataFrame()

    min_samples = max(int(getattr(config, "min_samples", 10)), ctx.get_min_samples("default"))
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
        fdr_alpha=float(getattr(config, "fdr_alpha", 0.05)),
        logger=ctx.logger,
        n_jobs=int(getattr(config, "n_jobs", -1)),
        config=ctx.config,
    )


def stage_temporal(ctx: BehaviorContext) -> None:
    from eeg_pipeline.analysis.behavior.api import (
        compute_time_frequency_from_context,
        compute_temporal_from_context,
        run_power_topomap_correlations,
    )

    status = {"time_frequency": "success", "temporal": "success", "topomap": "success"}

    ctx.logger.info("Computing time-frequency correlations...")
    try:
        compute_time_frequency_from_context(ctx)
    except Exception as exc:
        status["time_frequency"] = f"failed: {exc}"
        ctx.logger.error(f"Time-frequency correlations failed: {exc}")

    ctx.logger.info("Computing temporal correlations by condition...")
    try:
        compute_temporal_from_context(ctx)
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


def stage_cluster(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context

    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = int(getattr(config, "n_permutations", getattr(config, "n_permutations", 0)))

    try:
        run_cluster_test_from_context(ctx)
        return {"status": "completed"}
    except Exception as exc:
        ctx.logger.warning(f"Cluster tests failed: {exc}")
        return {"status": "failed", "error": str(exc)}


def stage_advanced(ctx: BehaviorContext, config: Any, results: Any) -> None:
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis, run_multilevel_correlation_analysis

    if ctx.precomputed_df is None:
        return

    if bool(getattr(config, "run_mediation", False)):
        ctx.logger.info("Running mediation analysis...")
        feature_cols = [
            c
            for c in ctx.precomputed_df.columns
            if c
            not in [
                "subject",
                "epoch",
                "trial",
                "condition",
                "temperature",
                "rating",
            ]
        ]
        if feature_cols:
            variances = ctx.precomputed_df[feature_cols].var()
            mediators = variances.nlargest(20).index.tolist()
            results.mediation = run_mediation_analysis(
                ctx.precomputed_df,
                "temperature",
                mediators,
                "rating",
                n_bootstrap=1000,
            )

    if bool(getattr(config, "run_mixed_effects", False)) and "subject" in ctx.precomputed_df.columns:
        ctx.logger.info("Running mixed-effects analysis...")
        feature_cols = [
            c
            for c in ctx.precomputed_df.columns
            if c
            not in [
                "subject",
                "epoch",
                "trial",
                "condition",
                "temperature",
                "rating",
            ]
        ]
        if feature_cols:
            results.mixed_effects = run_multilevel_correlation_analysis(
                ctx.precomputed_df,
                feature_cols[:50],
                "rating",
                "subject",
                config=ctx.config,
            )


def stage_validate(ctx: BehaviorContext, config: Any) -> None:
    from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr
    from eeg_pipeline.utils.analysis.stats.reliability import compute_hierarchical_fdr_summary

    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))

    ctx.logger.info("Applying global FDR correction...")
    apply_global_fdr(
        ctx.stats_dir,
        alpha=fdr_alpha,
        logger=ctx.logger,
        include_glob="corr_stats_*.tsv",
    )

    ctx.logger.info("Computing hierarchical FDR summary...")
    try:
        hier_summary = compute_hierarchical_fdr_summary(
            ctx.stats_dir,
            alpha=fdr_alpha,
            config=ctx.config,
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


def stage_export(ctx: BehaviorContext, results: Any) -> List[Path]:
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.infra.tsv import write_tsv

    ensure_dir(ctx.stats_dir)
    saved: List[Path] = []

    if getattr(results, "correlations", None) is not None and not results.correlations.empty:
        path = ctx.stats_dir / "correlations.tsv"
        write_tsv(results.correlations, path)
        saved.append(path)

    if getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty:
        path = ctx.stats_dir / "pain_sensitivity.tsv"
        write_tsv(results.pain_sensitivity, path)
        saved.append(path)

    if getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty:
        path = ctx.stats_dir / "condition_effects.tsv"
        write_tsv(results.condition_effects, path)
        saved.append(path)

    if getattr(results, "mediation", None) is not None and not results.mediation.empty:
        path = ctx.stats_dir / "mediation.tsv"
        write_tsv(results.mediation, path)
        saved.append(path)

    summary = results.to_summary()
    (ctx.stats_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    saved.append(ctx.stats_dir / "summary.json")

    ctx.logger.info(f"Saved {len(saved)} output files")
    return saved
