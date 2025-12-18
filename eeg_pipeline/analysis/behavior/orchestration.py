"""Behavior pipeline orchestration.

This module contains the implementation of the behavior pipeline stages.
The pipeline layer (`eeg_pipeline.pipelines.behavior`) should remain a thin wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.config.loader import get_config_value, get_min_samples


def combine_features(ctx: BehaviorContext) -> pd.DataFrame:
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
    return combined


def add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (plateau-baseline) once per context."""
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
    corr_df = result.to_dataframe()
    if corr_df is None:
        corr_df = pd.DataFrame()
    if not corr_df.empty:
        ctx.logger.info(f"Obtained {len(corr_df)} correlation results from unified correlator")

    psi_df = pd.DataFrame()
    if config.compute_pain_sensitivity and ctx.temperature is not None:
        features = combine_features(ctx)
        if not features.empty:
            psi_df = run_pain_sensitivity_correlations(
                features_df=features,
                ratings=ctx.targets,
                temperatures=ctx.temperature,
                method=config.method,
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


def write_analysis_metadata(ctx: BehaviorContext, pipeline_config: Any, results: Any) -> None:
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": pipeline_config.method,
        "min_samples": pipeline_config.min_samples,
        "control_temperature": pipeline_config.control_temperature,
        "control_trial_order": pipeline_config.control_trial_order,
        "compute_change_scores": pipeline_config.compute_change_scores,
        "compute_pain_sensitivity": pipeline_config.compute_pain_sensitivity,
        "compute_reliability": pipeline_config.compute_reliability,
        "n_permutations": pipeline_config.n_permutations,
        "fdr_alpha": pipeline_config.fdr_alpha,
        "n_trials": ctx.n_trials,
        "outputs": {
            "has_correlations": bool(getattr(results, "correlations", None) is not None and not results.correlations.empty),
            "has_pain_sensitivity": bool(getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty),
            "has_condition_effects": bool(getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty),
        },
        "qc": build_behavior_qc(ctx),
    }

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
    ctx.n_perm = config.n_permutations

    try:
        run_cluster_test_from_context(ctx)
        return {"status": "completed"}
    except Exception as exc:
        ctx.logger.warning(f"Cluster tests failed: {exc}")
        return {"status": "failed", "error": str(exc)}


_METADATA_COLUMNS = frozenset(["subject", "epoch", "trial", "condition", "temperature", "rating"])


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns excluding metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLUMNS]


def stage_advanced(ctx: BehaviorContext, config: Any, results: Any) -> None:
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis, run_multilevel_correlation_analysis

    if ctx.precomputed_df is None:
        return

    feature_cols = _get_feature_columns(ctx.precomputed_df)
    if not feature_cols:
        return

    if config.run_mediation:
        ctx.logger.info("Running mediation analysis...")
        n_bootstrap = get_config_value(ctx.config, "behavior_analysis.mediation.n_bootstrap", 1000)
        variances = ctx.precomputed_df[feature_cols].var()
        mediators = variances.nlargest(20).index.tolist()
        results.mediation = run_mediation_analysis(
            ctx.precomputed_df,
            "temperature",
            mediators,
            "rating",
            n_bootstrap=n_bootstrap,
        )

    if config.run_mixed_effects and "subject" in ctx.precomputed_df.columns:
        ctx.logger.info("Running mixed-effects analysis...")
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

    fdr_alpha = config.fdr_alpha

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
