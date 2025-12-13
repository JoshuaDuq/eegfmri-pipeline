"""
Behavior Analysis Pipeline (Canonical)
======================================

Pipeline class for EEG-behavior correlation analysis.
This module provides the PipelineBase subclass for behavior analysis,
with statistical routines consolidated in eeg_pipeline.analysis.behavior.api.

Usage:
    pipeline = BehaviorPipeline(config=config)
    pipeline.process_subject("0001", "thermalactive")
    
    # Or batch processing
    pipeline.run_batch(["0001", "0002"])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.analysis.stats.reliability import get_subject_seed


###################################################################
# Configuration
###################################################################


BEHAVIOR_COMPUTATION_FLAGS = [
    "correlations",
    "pain_sensitivity",
    "condition",
    "temporal",
    "cluster",
    "mediation",
    "mixed_effects",
    "export",
]

# Legacy aliases supported for backward-compatible CLI choices
_BEHAVIOR_COMP_ALIASES = {
    "condition_correlations": "condition",
    "time_frequency": "temporal",
    "temporal_correlations": "temporal",
    "cluster_test": "cluster",
    "exports": "export",
    "precomputed_correlations": "correlations",
    "power_roi": "correlations",
    "connectivity_roi": "correlations",
    "connectivity_heatmaps": "correlations",
    "sliding_connectivity": "temporal",
}


def _resolve_behavior_computation_flags(
    requested: Optional[List[str]],
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, bool]]:
    """
    Normalize requested behavior computations into stage flags.
    
    If requested is None, returns None to indicate no override (use config).
    """
    if requested is None:
        return None
    
    flags = {k: False for k in BEHAVIOR_COMPUTATION_FLAGS}
    normalized: List[str] = []
    for item in requested:
        key = str(item).lower()
        key = _BEHAVIOR_COMP_ALIASES.get(key, key)
        normalized.append(key)
    
    unknown = [k for k in normalized if k not in BEHAVIOR_COMPUTATION_FLAGS]
    if unknown and logger:
        logger.warning("Ignoring unrecognized behavior computations: %s", ", ".join(sorted(set(unknown))))
    
    for key in normalized:
        if key in flags:
            flags[key] = True
    
    return flags


@dataclass
class BehaviorPipelineConfig:
    method: str = "spearman"
    min_samples: int = 10
    control_temperature: bool = True
    control_trial_order: bool = True
    compute_change_scores: bool = True
    compute_pain_sensitivity: bool = True
    compute_reliability: bool = True
    
    run_condition_comparison: bool = True
    run_temporal_correlations: bool = True
    run_cluster_tests: bool = False
    run_mediation: bool = False
    run_mixed_effects: bool = False
    
    fdr_alpha: float = 0.05
    n_permutations: int = 1000
    n_jobs: int = -1
    
    @classmethod
    def from_config(cls, config: Any) -> "BehaviorPipelineConfig":
        from eeg_pipeline.utils.config.loader import get_config_value
        return cls(
            method=get_config_value(config, "behavior_analysis.statistics.correlation_method", "spearman"),
            min_samples=int(get_config_value(config, "behavior_analysis.min_samples.default", 10)),
            fdr_alpha=float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)),
            n_permutations=int(get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000)),
            run_temporal_correlations=get_config_value(config, "behavior_analysis.time_frequency_heatmap.enabled", True),
            run_cluster_tests=get_config_value(config, "behavior_analysis.cluster_correction.enabled", False),
            run_mediation=get_config_value(config, "behavior_analysis.mediation.enabled", False),
            run_mixed_effects=get_config_value(config, "behavior_analysis.mixed_effects.enabled", False),
            n_jobs=int(get_config_value(config, "behavior_analysis.n_jobs", -1)),
        )


@dataclass
class BehaviorPipelineResults:
    subject: str
    correlations: Optional[pd.DataFrame] = None
    pain_sensitivity: Optional[pd.DataFrame] = None
    condition_effects: Optional[pd.DataFrame] = None
    mediation: Optional[pd.DataFrame] = None
    mixed_effects: Optional[pd.DataFrame] = None
    cluster: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self) -> Dict[str, Any]:
        s = {"subject": self.subject}
        if self.correlations is not None:
            df = self.correlations
            s["n_features"] = len(df)
            p_raw = (
                df["p_raw"]
                if "p_raw" in df.columns
                else df.get("p_value")
                if "p_value" in df.columns
                else df.get("p")
            )
            p_ctrl = (
                df["p_partial_temp"]
                if "p_partial_temp" in df.columns
                else df.get("p_partial_cov")
            )
            if "p_fdr" in df.columns:
                p_fdr = df["p_fdr"]
            elif "q_value" in df.columns:
                p_fdr = df["q_value"]
            else:
                p_fdr = None
            s["n_sig_raw"] = int((p_raw.fillna(1) < 0.05).sum()) if p_raw is not None else 0
            s["n_sig_controlled"] = int((p_ctrl.fillna(1) < 0.05).sum()) if p_ctrl is not None else 0
            if p_fdr is not None:
                s["n_sig_fdr"] = int((p_fdr.fillna(1) < 0.05).sum())
        if self.condition_effects is not None:
            s["n_condition_effects"] = len(self.condition_effects)
            s["n_large_effects"] = int((self.condition_effects["hedges_g"].abs() >= 0.8).sum())
        return s


###################################################################
# Helper Functions
###################################################################


def _combine_features(ctx: BehaviorContext) -> pd.DataFrame:
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


def _add_change_scores(ctx: BehaviorContext) -> None:
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
        # Avoid duplicate columns
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


###################################################################
# Pipeline Stages
###################################################################


def _stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False
    
    ctx.logger.info(f"Loaded {ctx.n_trials} trials")
    return True


def _stage_correlate(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
) -> tuple:
    from eeg_pipeline.analysis.behavior.api import (
        run_pain_sensitivity_correlations,
        run_unified_feature_correlations,
    )
    
    # Attach change scores before correlation if enabled
    _add_change_scores(ctx)

    ctx.logger.info("Running unified feature correlator...")
    result = run_unified_feature_correlations(ctx)
    corr_df = result.to_dataframe() if getattr(result, "dataframe", None) is not None else result.to_dataframe()
    if corr_df is None:
        corr_df = pd.DataFrame()
    if not corr_df.empty:
        ctx.logger.info(f"Obtained {len(corr_df)} correlation results from unified correlator")
    
    psi_df = pd.DataFrame()
    if config.compute_pain_sensitivity and ctx.temperature is not None:
        features = _combine_features(ctx)
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


def _build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    qc: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "n_trials": int(ctx.n_trials),
        "has_temperature": bool(ctx.has_temperature),
        "temperature_column": ctx.temperature_column,
        "group_column": getattr(ctx, "group_column", None),
    }
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
        try:
            from eeg_pipeline.analysis.behavior.api import split_by_condition

            pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
                ctx.aligned_events, ctx.config, ctx.logger
            )
            if int(n_pain) > 0 or int(n_nonpain) > 0:
                s = pd.to_numeric(ctx.targets, errors="coerce")
                pain_ratings = s[pain_mask] if len(pain_mask) == len(s) else pd.Series(dtype=float)
                nonpain_ratings = s[nonpain_mask] if len(nonpain_mask) == len(s) else pd.Series(dtype=float)
                qc["pain_vs_nonpain"] = {
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
        except Exception:
            pass

    if ctx.covariates_df is not None and not ctx.covariates_df.empty:
        cov = ctx.covariates_df
        qc["covariates"] = {
            "n_covariates": int(cov.shape[1]),
            "columns": [str(c) for c in cov.columns],
            "missing_fraction_by_column": {
                str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean())
                for c in cov.columns
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


def _write_analysis_metadata(
    ctx: BehaviorContext,
    pipeline_config: BehaviorPipelineConfig,
    results: BehaviorPipelineResults,
) -> None:
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": pipeline_config.method,
        "min_samples": pipeline_config.min_samples,
        "control_temperature": bool(pipeline_config.control_temperature),
        "control_trial_order": bool(pipeline_config.control_trial_order),
        "compute_change_scores": bool(pipeline_config.compute_change_scores),
        "compute_pain_sensitivity": bool(pipeline_config.compute_pain_sensitivity),
        "compute_reliability": bool(pipeline_config.compute_reliability),
        "n_permutations": int(pipeline_config.n_permutations),
        "fdr_alpha": float(pipeline_config.fdr_alpha),
        "n_trials": int(ctx.n_trials),
        "outputs": {
            "has_correlations": bool(results.correlations is not None and not results.correlations.empty),
            "has_pain_sensitivity": bool(results.pain_sensitivity is not None and not results.pain_sensitivity.empty),
            "has_condition_effects": bool(results.condition_effects is not None and not results.condition_effects.empty),
        },
        "qc": _build_behavior_qc(ctx),
    }

    if results.correlations is not None and not results.correlations.empty:
        df = results.correlations
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
            payload["primary_test_source_counts"] = (
                df["p_primary_source"].fillna("unknown").value_counts().to_dict()
            )

        if "within_family_p_kind" in df.columns and df["within_family_p_kind"].notna().any():
            payload["within_family_p_kind_counts"] = (
                df["within_family_p_kind"].fillna("unknown").value_counts().to_dict()
            )

    with open(ctx.stats_dir / "analysis_metadata.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _stage_condition(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
) -> pd.DataFrame:
    from eeg_pipeline.analysis.behavior.api import split_by_condition, compute_condition_effects
    
    if ctx.aligned_events is None:
        return pd.DataFrame()
    
    pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
        ctx.aligned_events, ctx.config, ctx.logger
    )
    
    min_samples = max(config.min_samples, ctx.get_min_samples("default"))
    if n_pain < min_samples or n_nonpain < min_samples:
        ctx.logger.warning(f"Insufficient trials: {n_pain} pain, {n_nonpain} non-pain (need >= {min_samples})")
        return pd.DataFrame()
    
    features = _combine_features(ctx)
    if features.empty:
        return pd.DataFrame()
    
    return compute_condition_effects(
        features, pain_mask, nonpain_mask,
        min_samples=min_samples, fdr_alpha=config.fdr_alpha, logger=ctx.logger,
        n_jobs=config.n_jobs, config=ctx.config
    )


def _stage_temporal(ctx: BehaviorContext) -> None:
    from eeg_pipeline.analysis.behavior.api import (
        compute_time_frequency_from_context,
        compute_temporal_from_context,
        run_power_topomap_correlations,
    )
    
    status = {"time_frequency": "success", "temporal": "success", "topomap": "success"}

    ctx.logger.info("Computing time-frequency correlations...")
    try:
        compute_time_frequency_from_context(ctx)
    except Exception as e:
        status["time_frequency"] = f"failed: {e}"
        ctx.logger.error(f"Time-frequency correlations failed: {e}")
    
    ctx.logger.info("Computing temporal correlations by condition...")
    try:
        compute_temporal_from_context(ctx)
    except Exception as e:
        status["temporal"] = f"failed: {e}"
        ctx.logger.error(f"Temporal correlations failed: {e}")

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
    except Exception as e:
        status["topomap"] = f"failed: {e}"
        ctx.logger.error(f"Topomap correlations failed: {e}")

    failed = {k: v for k, v in status.items() if v != "success"}
    if failed:
        ctx.logger.info(f"Temporal stage status: {failed}")


def _stage_cluster(ctx: BehaviorContext, config: BehaviorPipelineConfig) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context
    
    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations
    
    try:
        run_cluster_test_from_context(ctx)
        return {"status": "completed"}
    except Exception as e:
        ctx.logger.warning(f"Cluster tests failed: {e}")
        return {"status": "failed", "error": str(e)}


def _stage_advanced(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
    results: BehaviorPipelineResults,
) -> None:
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis, run_multilevel_correlation_analysis
    
    if ctx.precomputed_df is None:
        return
    
    if config.run_mediation:
        ctx.logger.info("Running mediation analysis...")
        feature_cols = [c for c in ctx.precomputed_df.columns 
                       if c not in ["subject", "epoch", "trial", "condition", "temperature", "rating"]]
        if feature_cols:
            variances = ctx.precomputed_df[feature_cols].var()
            mediators = variances.nlargest(20).index.tolist()
            results.mediation = run_mediation_analysis(
                ctx.precomputed_df, "temperature", mediators, "rating", n_bootstrap=1000
            )
    
    if config.run_mixed_effects and "subject" in ctx.precomputed_df.columns:
        ctx.logger.info("Running mixed-effects analysis...")
        feature_cols = [c for c in ctx.precomputed_df.columns 
                       if c not in ["subject", "epoch", "trial", "condition", "temperature", "rating"]]
        if feature_cols:
            results.mixed_effects = run_multilevel_correlation_analysis(
                ctx.precomputed_df, feature_cols[:50], "rating", "subject", config=ctx.config
            )


def _stage_validate(ctx: BehaviorContext, config: BehaviorPipelineConfig) -> None:
    from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr
    
    ctx.logger.info("Applying global FDR correction...")
    apply_global_fdr(ctx.stats_dir, alpha=config.fdr_alpha, logger=ctx.logger)


def _stage_export(
    ctx: BehaviorContext,
    results: BehaviorPipelineResults,
) -> List[Path]:
    from eeg_pipeline.utils.io.tsv import write_tsv
    from eeg_pipeline.utils.io.paths import ensure_dir
    
    ensure_dir(ctx.stats_dir)
    saved = []
    
    if results.correlations is not None and not results.correlations.empty:
        path = ctx.stats_dir / "correlations.tsv"
        write_tsv(results.correlations, path)
        saved.append(path)
    
    if results.pain_sensitivity is not None and not results.pain_sensitivity.empty:
        path = ctx.stats_dir / "pain_sensitivity.tsv"
        write_tsv(results.pain_sensitivity, path)
        saved.append(path)
    
    if results.condition_effects is not None and not results.condition_effects.empty:
        path = ctx.stats_dir / "condition_effects.tsv"
        write_tsv(results.condition_effects, path)
        saved.append(path)
    
    if results.mediation is not None and not results.mediation.empty:
        path = ctx.stats_dir / "mediation.tsv"
        write_tsv(results.mediation, path)
        saved.append(path)
    
    summary = results.to_summary()
    with open(ctx.stats_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    saved.append(ctx.stats_dir / "summary.json")
    
    ctx.logger.info(f"Saved {len(saved)} output files")
    return saved


###################################################################
# Pipeline Class
###################################################################


class BehaviorPipeline(PipelineBase):
    """Pipeline for EEG-behavior correlation analysis."""
    
    def __init__(
        self,
        config: Optional[Any] = None,
        pipeline_config: Optional[BehaviorPipelineConfig] = None,
        computations: Optional[List[str]] = None,
    ):
        super().__init__(name="behavior_analysis", config=config)
        self.pipeline_config = pipeline_config or BehaviorPipelineConfig.from_config(self.config)
        
        self._run_correlations = True
        self._run_export = True
        self._run_validation = True
        
        comp_flags = _resolve_behavior_computation_flags(computations, logger=self.logger)
        if comp_flags is not None:
            self._run_correlations = comp_flags["correlations"]
            self._run_export = comp_flags["export"]
            self.pipeline_config.run_condition_comparison = comp_flags["condition"]
            self.pipeline_config.run_temporal_correlations = comp_flags["temporal"]
            self.pipeline_config.run_cluster_tests = comp_flags["cluster"]
            self.pipeline_config.run_mediation = comp_flags["mediation"]
            self.pipeline_config.run_mixed_effects = comp_flags["mixed_effects"]
            self.pipeline_config.compute_pain_sensitivity = comp_flags["pain_sensitivity"]
            self._run_validation = any(
                comp_flags[key] for key in BEHAVIOR_COMPUTATION_FLAGS if key != "export"
            )
            
            selected = [k for k, v in comp_flags.items() if v]
            self.logger.info("Behavior computations (override): %s", ", ".join(selected) if selected else "none")
        else:
            self._run_validation = any(
                [
                    self._run_correlations,
                    self.pipeline_config.run_condition_comparison,
                    self.pipeline_config.run_temporal_correlations,
                    self.pipeline_config.run_cluster_tests,
                    self.pipeline_config.run_mediation,
                    self.pipeline_config.run_mixed_effects,
                    self.pipeline_config.compute_pain_sensitivity,
                ]
            )

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> BehaviorPipelineResults:
        from eeg_pipeline.utils.io.paths import deriv_stats_path, ensure_dir
        from eeg_pipeline.utils.io.logging import get_subject_logger
        
        task = task or self.config.get("project.task", "thermalactive")
        
        stats_dir = deriv_stats_path(self.deriv_root, subject)
        ensure_dir(stats_dir)
        
        logger = get_subject_logger(
            "behavior_analysis", subject,
            self.config.get("logging.log_file_name", "behavior_analysis.log"),
            config=self.config
        )
        
        logger.info(f"{'='*60}")
        logger.info(f"Behavior Pipeline: sub-{subject}")
        logger.info(f"{'='*60}")

        base_seed = 42
        try:
            base_seed = int(self.config.get("behavior_analysis.statistics.base_seed", base_seed))
        except Exception:
            base_seed = 42
        rng = np.random.default_rng(get_subject_seed(base_seed, subject))

        ctx = BehaviorContext(
            subject=subject,
            task=task,
            config=self.config,
            logger=logger,
            deriv_root=self.deriv_root,
            stats_dir=stats_dir,
            use_spearman=(self.pipeline_config.method == "spearman"),
            rng=rng,
            control_temperature=self.pipeline_config.control_temperature,
            control_trial_order=self.pipeline_config.control_trial_order,
            compute_change_scores=self.pipeline_config.compute_change_scores,
            compute_reliability=self.pipeline_config.compute_reliability,
        )
        
        results = BehaviorPipelineResults(subject=subject)
        
        logger.info("Stage 1: Loading data...")
        if not _stage_load(ctx):
            return results
        
        if self._run_correlations:
            logger.info("Stage 2: Running unified correlations...")
            results.correlations, results.pain_sensitivity = _stage_correlate(ctx, self.pipeline_config)
        else:
            logger.info("Stage 2: Skipped correlations (CLI selection)")
        
        if self.pipeline_config.run_condition_comparison:
            logger.info("Stage 3: Condition comparison...")
            results.condition_effects = _stage_condition(ctx, self.pipeline_config)
        else:
            logger.info("Stage 3: Skipped condition comparison (CLI selection)")
        
        if self.pipeline_config.run_temporal_correlations:
            logger.info("Stage 4: Temporal correlations...")
            _stage_temporal(ctx)
        else:
            logger.info("Stage 4: Skipped temporal correlations (CLI selection)")
        
        if self.pipeline_config.run_cluster_tests:
            logger.info("Stage 5: Cluster permutation tests...")
            results.cluster = _stage_cluster(ctx, self.pipeline_config)
        else:
            logger.info("Stage 5: Skipped cluster tests (CLI selection)")
        
        if self.pipeline_config.run_mediation or self.pipeline_config.run_mixed_effects:
            logger.info("Stage 6: Advanced analyses...")
            _stage_advanced(ctx, self.pipeline_config, results)
        else:
            logger.info("Stage 6: Skipped advanced analyses (CLI selection)")
        
        if self._run_validation:
            logger.info("Stage 7: Global FDR correction...")
            _stage_validate(ctx, self.pipeline_config)
        else:
            logger.info("Stage 7: Skipped global FDR (no computations selected)")
        
        if self._run_export:
            logger.info("Stage 8: Saving results...")
            _stage_export(ctx, results)
        else:
            logger.info("Stage 8: Skipped export (CLI selection)")

        try:
            _write_analysis_metadata(ctx, self.pipeline_config, results)
        except Exception as e:
            logger.warning(f"Failed to write analysis metadata: {e}")
        
        summary = results.to_summary()
        logger.info(f"{'='*60}")
        logger.info(f"Complete: {summary.get('n_features', 0)} features")
        logger.info(f"  Significant (raw): {summary.get('n_sig_raw', 0)}")
        logger.info(f"  Significant (controlled): {summary.get('n_sig_controlled', 0)}")
        logger.info(f"{'='*60}")
        
        return results


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
]
