"""
Unified Feature-Behavior Correlator
====================================

Single entry point for correlating ALL EEG features with behavioral measures.
Uses core.correlate_features_loop for all correlations - no duplicate logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import re

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext, ComputationResult, ComputationStatus
from eeg_pipeline.utils.analysis.stats.correlation import (
    CorrelationRecord,
    correlate_features_loop,
    save_correlation_results,
    align_groups_to_series as _align_groups_to_series,
    align_features_and_targets,
    build_temp_record_unified,
    compute_roi_correlation_stats,
    format_correlation_method_label,
    normalize_correlation_method,
    safe_correlation,
)
from eeg_pipeline.infra.paths import deriv_features_path
from eeg_pipeline.infra.tsv import read_tsv, write_tsv
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.context.behavior import AnalysisConfig
from eeg_pipeline.utils.config.loader import get_min_samples, get_config_value, load_config
from eeg_pipeline.utils.analysis.stats import (
    prepare_aligned_data,
    compute_correlation,
    compute_bootstrap_ci,
    compute_partial_correlations_with_cov_temp,
    compute_permutation_pvalues_with_cov_temp,
    compute_temp_permutation_pvalues,
    CorrelationStats,
    fdr_bh,
)
from eeg_pipeline.utils.parallel import parallel_feature_types, get_n_jobs
from joblib import Parallel, delayed, cpu_count
from eeg_pipeline.domain.features.registry import (
    FeatureRegistry,
    classify_feature,
    get_feature_registry,
)
from eeg_pipeline.utils.analysis.stats.reliability import compute_correlation_split_half_reliability


@dataclass
class CorrelationConfig:
    """Configuration for feature-behavior correlations."""

    method: str
    min_samples: int
    apply_fdr: bool = True
    fdr_alpha: Optional[float] = None
    n_bootstrap: int = 0
    n_permutations: int = 0
    rng: Optional[np.random.Generator] = None
    filter_threshold: float = 0.0
    compute_bayes_factor: bool = False
    robust_method: Optional[str] = None  # "percentage_bend", "winsorized", "shepherd"
    method_label: Optional[str] = None
    compute_loso_stability: bool = False
    compute_reliability: bool = False
    covariates_df: Optional[pd.DataFrame] = None
    covariates_without_temp_df: Optional[pd.DataFrame] = None
    temperature_series: Optional[pd.Series] = None
    groups: Optional[np.ndarray] = None
    control_temperature: bool = True
    control_trial_order: bool = True
    n_jobs: int = -1

    @classmethod
    def from_config(cls, config: Any, ctx: Optional[BehaviorContext] = None) -> "CorrelationConfig":
        """Build configuration from config dict with optional context overrides.
        
        Parameters
        ----------
        config : Any
            Configuration dictionary with behavior_analysis settings.
        ctx : BehaviorContext, optional
            If provided, runtime overrides from context take precedence.
        """
        raw_method = get_config_value(
            config, "behavior_analysis.statistics.correlation_method", None
        )
        if raw_method is None:
            raw_method = get_config_value(config, "behavior_analysis.correlation_method", "spearman")
        method = normalize_correlation_method(raw_method, default="spearman")
        min_samples = get_min_samples(config, "channel")
        fdr_alpha = float(get_config_value(
            config, "behavior_analysis.statistics.fdr_alpha",
            get_config_value(config, "statistics.fdr_alpha", 0.05)
        ))
        n_bootstrap = int(get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 1000))
        n_permutations = int(get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000))
        compute_bayes_factor = bool(get_config_value(config, "behavior_analysis.compute_bayes_factors", False))
        robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
        if robust_method is not None:
            robust_method = str(robust_method).strip().lower() or None
        compute_loso_stability = bool(get_config_value(config, "behavior_analysis.loso_stability", True))
        compute_reliability = bool(get_config_value(config, "behavior_analysis.statistics.compute_reliability", False))
        n_jobs = int(get_config_value(config, "behavior_analysis.n_jobs", -1))
        
        if ctx is not None:
            method = normalize_correlation_method(ctx.method or method, default=method)
            min_samples = ctx.min_samples_channel or min_samples
            n_bootstrap = ctx.bootstrap if ctx.bootstrap is not None else n_bootstrap
            n_permutations = ctx.n_perm if ctx.n_perm is not None else n_permutations
            compute_reliability = ctx.compute_reliability
        
        method_label = format_correlation_method_label(method, robust_method)

        return cls(
            method=method,
            min_samples=min_samples,
            fdr_alpha=fdr_alpha,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            rng=ctx.rng if ctx else None,
            compute_bayes_factor=compute_bayes_factor,
            robust_method=robust_method,
            method_label=method_label,
            compute_loso_stability=compute_loso_stability,
            compute_reliability=compute_reliability,
            covariates_df=ctx.covariates_df if ctx else None,
            covariates_without_temp_df=ctx.covariates_without_temp_df if ctx else None,
            temperature_series=ctx.temperature if ctx and ctx.control_temperature else None,
            groups=ctx.group_ids if ctx else None,
            control_temperature=ctx.control_temperature if ctx else True,
            control_trial_order=ctx.control_trial_order if ctx else True,
            n_jobs=n_jobs,
        )

    @classmethod
    def from_context(cls, ctx: BehaviorContext) -> "CorrelationConfig":
        """Context-aware configuration that honors runtime overrides."""
        if getattr(ctx, "stats_config", None) is not None:
            stats_cfg = ctx.stats_config
            robust_method = getattr(stats_cfg, "robust_method", None)
            method_label = format_correlation_method_label(stats_cfg.method, robust_method)
            return cls(
                method=stats_cfg.method,
                min_samples=stats_cfg.min_samples,
                fdr_alpha=stats_cfg.fdr_alpha,
                n_bootstrap=stats_cfg.bootstrap,
                n_permutations=stats_cfg.n_permutations,
                rng=ctx.rng,
                compute_bayes_factor=bool(getattr(stats_cfg, "compute_bayes_factors", False)),
                robust_method=robust_method,
                method_label=method_label,
                compute_loso_stability=bool(getattr(stats_cfg, "compute_loso_stability", True)),
                compute_reliability=ctx.compute_reliability,
                covariates_df=ctx.covariates_df,
                covariates_without_temp_df=ctx.covariates_without_temp_df,
                temperature_series=ctx.temperature if ctx.control_temperature else None,
                groups=ctx.group_ids,
                control_temperature=ctx.control_temperature,
                control_trial_order=ctx.control_trial_order,
                n_jobs=int(getattr(stats_cfg, "n_jobs", -1)),
            )

        return cls.from_config(ctx.config or {}, ctx=ctx)


@dataclass
class FeatureCorrelationResult:
    """Result for a single feature type's correlations."""

    feature_type: str
    n_features: int
    n_significant: int
    n_total: int = 0
    n_dropped: int = 0
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records) if self.records else pd.DataFrame()


# =============================================================================
# Parallel Column Processing Helpers
# =============================================================================


@dataclass
class ColumnProcessorParams:
    """Parameters for parallel column processing. Bundles config to reduce function args."""
    method: str
    min_samples: int
    n_permutations: int
    n_bootstrap: int
    control_temperature: bool
    control_trial_order: bool
    compute_bayes_factor: bool
    compute_loso_stability: bool
    compute_reliability: bool


def _infer_min_samples_for_column(
    col_name: str,
    feature_type: str,
    config: Any,
) -> int:
    parsed = NamingSchema.parse(str(col_name))
    if parsed.get("valid"):
        scope = str(parsed.get("scope") or "")
        if scope == "chpair":
            return get_min_samples(config, "edge")
        if scope == "roi":
            return get_min_samples(config, "roi")
        if scope == "ch":
            return get_min_samples(config, "channel")
        return get_min_samples(config, "default")

    name = str(col_name)
    if "_chpair_" in name:
        return get_min_samples(config, "edge")
    if "_roi_" in name:
        return get_min_samples(config, "roi")
    if "_ch_" in name:
        return get_min_samples(config, "channel")
    if feature_type == "connectivity":
        return get_min_samples(config, "edge")
    return get_min_samples(config, "default")


def _build_base_record(
    col_name: str,
    feature_type: str,
    r: float,
    p: float,
    n_valid: int,
    n_total: int,
    missing_fraction: float,
    variance: float,
    method: str,
    n_covariates: int,
    robust_method: Optional[str],
    method_label: str,
    target_name: str,
) -> Dict[str, Any]:
    """Build base correlation record with raw statistics."""
    return {
        "feature": col_name,
        "feature_type": feature_type,
        "r": float(r),
        "p": float(p),
        "n": n_valid,
        "n_total": int(n_total),
        "missing_fraction": float(missing_fraction),
        "variance": float(variance) if np.isfinite(variance) else np.nan,
        "method": method,
        "robust_method": robust_method,
        "method_label": method_label,
        "target": target_name,
        "r_raw": float(r),
        "p_raw": float(p),
        "p_value": float(p),
        "p_kind_primary": "p_raw",
        "p_primary": float(p),
        "r_primary": float(r),
        "p_primary_source": "raw",
        "p_primary_perm": np.nan,
        "p_primary_is_permutation": False,
        "n_covariates_used": n_covariates,
    }


def _add_bootstrap_ci(
    record: Dict[str, Any],
    col_values: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int,
    method: str,
    rng: np.random.Generator,
) -> None:
    """Add bootstrap confidence intervals to record."""
    from eeg_pipeline.utils.analysis.stats.eeg_stats import compute_bootstrap_ci
    
    valid_mask = np.isfinite(col_values) & np.isfinite(targets)
    ci_low, ci_high = compute_bootstrap_ci(
        col_values[valid_mask], targets[valid_mask],
        n_bootstrap=n_bootstrap, ci_level=0.95,
        method=method, rng=rng
    )
    record["ci_low"] = ci_low
    record["ci_high"] = ci_high


def _add_bayes_factor(
    record: Dict[str, Any],
    col_values: np.ndarray,
    targets: np.ndarray,
    method: str,
) -> None:
    """Add Bayes factor to record."""
    from eeg_pipeline.utils.analysis.stats.correlation import compute_bayes_factor_correlation
    
    bf10, bf_interp = compute_bayes_factor_correlation(col_values, targets, method=method)
    record["bf10"] = bf10
    record["bf_interpretation"] = bf_interp


def _add_partial_correlations(
    record: Dict[str, Any],
    feature_series: pd.Series,
    target_series: pd.Series,
    cov_aligned: Optional[pd.DataFrame],
    temp_aligned: Optional[pd.Series],
    method: str,
    feature_type: str,
    min_samples: int,
) -> None:
    """Add partial correlation results to record."""
    (
        r_pc, p_pc, n_pc,
        r_temp, p_temp, n_temp,
        r_cov_temp, p_cov_temp, n_cov_temp,
    ) = compute_partial_correlations_with_cov_temp(
        roi_values=feature_series,
        target_values=target_series,
        covariates_df=cov_aligned,
        temperature_series=temp_aligned,
        method=method,
        context=feature_type,
        logger=None,
        min_samples=min_samples,
        config=None,
    )
    record["r_partial_cov"] = r_pc
    record["p_partial_cov"] = p_pc
    record["n_partial_cov"] = n_pc
    record["r_partial_temp"] = r_temp
    record["p_partial_temp"] = p_temp
    record["n_partial_temp"] = n_temp
    record["r_partial_cov_temp"] = r_cov_temp
    record["p_partial_cov_temp"] = p_cov_temp
    record["n_partial_cov_temp"] = n_cov_temp


def _add_permutation_pvalues(
    record: Dict[str, Any],
    feature_series: pd.Series,
    target_series: pd.Series,
    cov_aligned: Optional[pd.DataFrame],
    temp_aligned: Optional[pd.Series],
    method: str,
    n_permutations: int,
    rng: np.random.Generator,
    perm_groups: Optional[np.ndarray],
) -> None:
    """Add permutation p-values to record."""
    n_eff = int(pd.concat([feature_series, target_series], axis=1).dropna().shape[0])
    (
        p_perm_raw,
        p_perm_partial_cov,
        p_perm_partial_temp,
        p_perm_partial_cov_temp,
    ) = compute_permutation_pvalues_with_cov_temp(
        x_aligned=feature_series,
        y_aligned=target_series,
        covariates_df=cov_aligned,
        temp_series=temp_aligned,
        method=method,
        n_perm=n_permutations,
        n_eff=n_eff,
        rng=rng,
        config=None,
        groups=perm_groups,
    )
    record["p_perm_raw"] = p_perm_raw
    record["p_perm_partial_cov"] = p_perm_partial_cov
    record["p_perm_partial_temp"] = p_perm_partial_temp
    record["p_perm_partial_cov_temp"] = p_perm_partial_cov_temp
    record["p_perm"] = p_perm_raw


def _select_primary_correlation(
    record: Dict[str, Any],
    control_temperature: bool,
    control_trial_order: bool,
) -> None:
    """Select primary correlation based on control settings."""
    if control_temperature and control_trial_order:
        if pd.notna(record.get("p_partial_cov_temp", np.nan)):
            record["p_kind_primary"] = "p_partial_cov_temp"
            record["p_primary"] = record.get("p_partial_cov_temp", np.nan)
            record["r_primary"] = record.get("r_partial_cov_temp", np.nan)
            record["p_primary_source"] = "partial_cov_temp"
    elif control_temperature:
        if pd.notna(record.get("p_partial_temp", np.nan)):
            record["p_kind_primary"] = "p_partial_temp"
            record["p_primary"] = record.get("p_partial_temp", np.nan)
            record["r_primary"] = record.get("r_partial_temp", np.nan)
            record["p_primary_source"] = "partial_temp"
    elif control_trial_order:
        if pd.notna(record.get("p_partial_cov", np.nan)):
            record["p_kind_primary"] = "p_partial_cov"
            record["p_primary"] = record.get("p_partial_cov", np.nan)
            record["r_primary"] = record.get("r_partial_cov", np.nan)
            record["p_primary_source"] = "partial_cov"


def _update_primary_perm_pvalue(record: Dict[str, Any]) -> None:
    """Update primary permutation p-value based on selected primary."""
    p_perm_col_by_primary = {
        "p_raw": "p_perm_raw",
        "p_partial_cov": "p_perm_partial_cov",
        "p_partial_temp": "p_perm_partial_temp",
        "p_partial_cov_temp": "p_perm_partial_cov_temp",
    }
    perm_col = p_perm_col_by_primary.get(record.get("p_kind_primary"))
    if perm_col is not None:
        record["p_primary_perm"] = record.get(perm_col, np.nan)
        record["p_primary_is_permutation"] = bool(pd.notna(record.get("p_primary_perm", np.nan)))


def _add_loso_stability(
    record: Dict[str, Any],
    col_values: np.ndarray,
    targets: np.ndarray,
    loso_groups: np.ndarray,
    method: str,
) -> None:
    """Add LOSO stability metrics to record."""
    from eeg_pipeline.utils.analysis.stats.correlation import compute_loso_correlation_stability
    
    r_mean, r_std, stability, _ = compute_loso_correlation_stability(
        col_values, targets, loso_groups, method
    )
    record["loso_r_mean"] = r_mean
    record["loso_r_std"] = r_std
    record["loso_stability"] = stability


def _add_reliability(
    record: Dict[str, Any],
    col_values: np.ndarray,
    targets: np.ndarray,
    method: str,
) -> None:
    """Add split-half reliability to record."""
    rel = compute_correlation_split_half_reliability(col_values, targets, method, n_splits=50)
    record["reliability_split_half"] = rel


def _process_single_column(
    col_name: str,
    col_values: np.ndarray,
    feature_type: str,
    targets_aligned: np.ndarray,
    column_min_samples: int,
    n_total: int,
    missing_fraction: float,
    variance: float,
    cov_aligned: Optional[pd.DataFrame],
    temp_aligned: Optional[pd.Series],
    loso_groups: Optional[np.ndarray],
    perm_groups: Optional[np.ndarray],
    config_method: str,
    config_robust_method: Optional[str],
    config_method_label: str,
    target_name: str,
    config_n_permutations: int,
    config_n_bootstrap: int,
    config_rng_seed: int,
    control_temperature: bool,
    control_trial_order: bool,
    compute_bayes_factor: bool,
    compute_loso_stability: bool,
    compute_reliability: bool,
) -> Optional[Dict[str, Any]]:
    """Process correlations for a single column. Designed for parallel execution."""
    from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
    
    rng = np.random.default_rng(config_rng_seed)
    
    r, p, n_valid = safe_correlation(
        col_values, targets_aligned, config_method, column_min_samples, robust_method=config_robust_method
    )
    if not np.isfinite(r):
        return None
    
    n_cov = int(cov_aligned.shape[1]) if cov_aligned is not None else 0
    record = _build_base_record(
        col_name,
        feature_type,
        r,
        p,
        n_valid,
        n_total,
        missing_fraction,
        variance,
        config_method,
        n_cov,
        config_robust_method, config_method_label, target_name
    )
    
    feature_series = pd.Series(col_values)
    target_series = pd.Series(targets_aligned)

    if config_n_bootstrap and config_n_bootstrap > 0:
        try:
            _add_bootstrap_ci(record, col_values, targets_aligned, config_n_bootstrap, config_method, rng)
        except Exception as exc:
            record["bootstrap_error"] = str(exc)

    if compute_bayes_factor:
        try:
            _add_bayes_factor(record, col_values, targets_aligned, config_method)
        except Exception as exc:
            record["bayes_factor_error"] = str(exc)

    if cov_aligned is not None or temp_aligned is not None:
        try:
            _add_partial_correlations(
                record, feature_series, target_series, cov_aligned, temp_aligned,
                config_method, feature_type, column_min_samples
            )
        except Exception as exc:
            record["partial_corr_error"] = str(exc)

    if config_n_permutations and config_n_permutations > 0:
        try:
            _add_permutation_pvalues(
                record, feature_series, target_series, cov_aligned, temp_aligned,
                config_method, config_n_permutations, rng, perm_groups
            )
        except Exception as exc:
            record["permutation_error"] = str(exc)

    _select_primary_correlation(record, control_temperature, control_trial_order)

    if config_n_permutations and config_n_permutations > 0:
        _update_primary_perm_pvalue(record)

    if compute_loso_stability and loso_groups is not None:
        try:
            _add_loso_stability(record, col_values, targets_aligned, loso_groups, config_method)
        except Exception as exc:
            record["loso_error"] = str(exc)

    if compute_reliability:
        try:
            _add_reliability(record, col_values, targets_aligned, config_method)
        except Exception as exc:
            record["reliability_error"] = str(exc)

    return record


# =============================================================================
# Core Correlator Class
# =============================================================================


class FeatureBehaviorCorrelator:
    """Unified correlator for all EEG features with behavioral measures."""

    def __init__(
        self,
        subject: str,
        deriv_root: Path,
        config: Any,
        logger: logging.Logger,
        stats_dir: Optional[Path] = None,
    ):
        self.subject = subject
        self.deriv_root = deriv_root
        self.config = config
        self.logger = logger
        self.stats_dir = stats_dir or deriv_root / f"sub-{subject}" / "eeg" / "stats"
        self.features_dir = deriv_features_path(deriv_root, subject)
        self._feature_dfs: Dict[str, pd.DataFrame] = {}
        self._loaded = False
        self.registry = get_feature_registry(config)
        self.default_corr_config = CorrelationConfig.from_config(config)

    def load_all_features(self) -> Dict[str, int]:
        """Load all available feature files. Returns feature counts."""
        if self._loaded:
            return {k: len(v.columns) for k, v in self._feature_dfs.items()}

        counts = {}
        for name, filename in self.registry.files.items():
            path = self.features_dir / filename
            if path.exists():
                df = read_tsv(path)
                if df is not None and not df.empty:
                    self._feature_dfs[name] = df
                    counts[name] = len(df.columns)

        self._loaded = True
        self.logger.info(f"Loaded {len(self._feature_dfs)} feature files, {sum(counts.values())} features")
        return counts

    def correlate_all(
        self,
        targets: pd.Series,
        target_name: str = "rating",
        corr_config: Optional[CorrelationConfig] = None,
        n_jobs: int = -1,
    ) -> Dict[str, FeatureCorrelationResult]:
        """Correlate all loaded features with behavioral target."""
        if not self._loaded:
            self.load_all_features()

        if corr_config is None:
            corr_config = self.default_corr_config

        n_jobs_actual = get_n_jobs(self.config, n_jobs)
        self.logger.info(f"Correlating features with {target_name}... (n_jobs={n_jobs_actual})")
        
        def _corr_with_groups(df, tvals, cfg, name):
            return self._correlate_df(
                df, tvals, cfg, name, target_name=target_name, subject_ids=cfg.groups
            )

        # Parallel correlation across feature types
        results = parallel_feature_types(
            feature_dfs=self._feature_dfs,
            targets=targets,
            correlate_func=_corr_with_groups,
            corr_config=corr_config,
            n_jobs=n_jobs_actual,
            logger=self.logger,
        )
        
        for name, result in results.items():
            if result.n_features > 0:
                dropped = result.n_dropped
                total = result.n_total or (result.n_features + dropped)
                self.logger.debug(
                    f"  {name}: {result.n_features}/{total} tested, {dropped} dropped, {result.n_significant} sig"
                )

        return results

    def _correlate_df(
        self,
        df: pd.DataFrame,
        targets: pd.Series,
        config: CorrelationConfig,
        feature_type: str,
        target_name: str = "rating",
        subject_ids: Optional[np.ndarray] = None,
    ) -> FeatureCorrelationResult:
        """Correlate a single feature dataframe using core function."""
        if df is None or df.empty or targets is None or len(targets) == 0:
            return FeatureCorrelationResult(feature_type, 0, 0)

        min_samples_align = get_min_samples(self.config, "default")
        df_aligned, targets_aligned = align_features_and_targets(
            df, targets, min_samples_align, self.logger
        )
        if df_aligned is None or targets_aligned is None:
            return FeatureCorrelationResult(feature_type, 0, 0)

        cov_aligned = None
        temp_aligned = None
        loso_groups = None
        perm_groups = None
        if config.control_temperature:
            if config.covariates_without_temp_df is not None:
                cov_aligned = config.covariates_without_temp_df.reindex(df_aligned.index)
            else:
                cov_aligned = None
        elif config.covariates_df is not None:
            cov_aligned = config.covariates_df.reindex(df_aligned.index)
        if config.control_temperature and config.temperature_series is not None:
            temp_aligned = config.temperature_series.reindex(df_aligned.index)
        if subject_ids is not None:
            try:
                loso_groups = _align_groups_to_series(targets_aligned, subject_ids)
                perm_groups = loso_groups
            except ValueError as exc:
                self.logger.debug(f"Group alignment failed: {exc}")
        if perm_groups is None and config.groups is not None:
            try:
                perm_groups = _align_groups_to_series(targets_aligned, config.groups)
                if loso_groups is None:
                    loso_groups = perm_groups
            except ValueError as exc:
                self.logger.debug(f"Permutation group alignment failed: {exc}")
        
        # Parallel column processing - all computation done in _process_single_column
        n_jobs_actual = config.n_jobs
        if n_jobs_actual == -1:
            n_jobs_actual = max(1, cpu_count() - 1)
        
        base_seed = int(get_config_value(self.config, "behavior_analysis.statistics.base_seed", 42))
        if config.rng is not None:
            base_seed = int(config.rng.integers(0, 2**31))
        
        targets_arr = targets_aligned.values if hasattr(targets_aligned, 'values') else np.asarray(targets_aligned)
        n_total = int(len(targets_arr))
        loso_groups_arr = loso_groups if loso_groups is not None else None
        perm_groups_arr = perm_groups if perm_groups is not None else None
        
        # Prepare arguments for each column (no pre-computed records needed)
        col_args = []
        screening_records = []
        for i, col_name in enumerate(df_aligned.columns):
            col_values = pd.to_numeric(df_aligned[col_name], errors="coerce").values
            min_samples = _infer_min_samples_for_column(col_name, feature_type, self.config)
            valid_mask = np.isfinite(col_values) & np.isfinite(targets_arr)
            n_valid = int(valid_mask.sum())
            missing_fraction = 1.0 - (float(n_valid) / n_total) if n_total else 1.0
            variance = float(np.nanvar(col_values[valid_mask])) if n_valid > 1 else np.nan
            status = "kept"
            reason = None
            if n_valid < min_samples:
                status = "dropped"
                reason = "insufficient_samples"
            elif not np.isfinite(variance) or np.isclose(variance, 0.0):
                status = "dropped"
                reason = "zero_variance"

            screening_records.append(
                {
                    "feature": str(col_name),
                    "feature_type": feature_type,
                    "min_samples": int(min_samples),
                    "n_valid": int(n_valid),
                    "n_total": int(n_total),
                    "missing_fraction": float(missing_fraction),
                    "variance": float(variance) if np.isfinite(variance) else np.nan,
                    "status": status,
                    "reason": reason,
                }
            )

            if status != "kept":
                continue

            col_args.append((
                col_name,
                col_values,
                feature_type,
                targets_arr,
                min_samples,
                n_total,
                missing_fraction,
                variance,
                cov_aligned,
                temp_aligned,
                loso_groups_arr,
                perm_groups_arr,
                config.method,
                config.robust_method,
                config.method_label or format_correlation_method_label(config.method, config.robust_method),
                target_name,
                config.n_permutations,
                config.n_bootstrap,
                base_seed + i,
                config.control_temperature,
                config.control_trial_order,
                config.compute_bayes_factor,
                config.compute_loso_stability,
                config.compute_reliability,
            ))

        if screening_records and self.stats_dir is not None:
            try:
                from eeg_pipeline.infra.paths import ensure_dir
                screen_dir = self.stats_dir / "feature_screening"
                ensure_dir(screen_dir)
                screen_df = pd.DataFrame(screening_records)
                screen_path = screen_dir / f"feature_screening_{feature_type}_vs_{target_name}.tsv"
                write_tsv(screen_df, screen_path)
            except Exception as exc:
                self.logger.debug(f"Failed to write screening report: {exc}")

        n_cols = len(col_args)
        
        # Use parallel processing when beneficial (>10 columns and n_jobs > 1)
        if n_jobs_actual > 1 and n_cols > 10:
            record_dicts = Parallel(n_jobs=n_jobs_actual, backend="loky")(
                delayed(_process_single_column)(*args) for args in col_args
            )
            record_dicts = [d for d in record_dicts if d is not None]
        else:
            record_dicts = []
            for args in col_args:
                d = _process_single_column(*args)
                if d is not None:
                    record_dicts.append(d)

        # Apply within-type FDR if requested
        if record_dicts and config.apply_fdr:
            use_perm_p = bool(config.n_permutations and config.n_permutations > 0)
            if use_perm_p:
                p_vals = [r.get("p_primary_perm", np.nan) for r in record_dicts]
                p_kind = "p_primary_perm"
            else:
                p_vals = [r.get("p_primary", np.nan) for r in record_dicts]
                p_kind = "p_primary"
            valid_idx = [i for i, pv in enumerate(p_vals) if pd.notna(pv)]
            if valid_idx:
                q_vals = fdr_bh(
                    np.array([p_vals[i] for i in valid_idx]),
                    alpha=config.fdr_alpha,
                    config=self.config,
                )
                for idx, q in zip(valid_idx, q_vals):
                    record_dicts[idx]["p_fdr"] = float(q)
                    record_dicts[idx]["q_within_family"] = float(q)
                    record_dicts[idx]["within_family_p_kind"] = p_kind

        alpha = float(get_config_value(self.config, "statistics.sig_alpha", 0.05))
        n_sig = sum(
            1
            for rec in record_dicts
            if pd.notna(rec.get("p_primary", np.nan)) and float(rec.get("p_primary")) < alpha
        )
        n_dropped = len(df_aligned.columns) - len(col_args)
        return FeatureCorrelationResult(
            feature_type,
            len(record_dicts),
            n_sig,
            n_total=len(df_aligned.columns),
            n_dropped=n_dropped,
            records=record_dicts,
        )
    
    def save_results(
        self,
        results: Dict[str, FeatureCorrelationResult],
        target_name: str = "rating",
        apply_fdr: bool = True,
        method_label: Optional[str] = None,
    ) -> List[Path]:
        """Save correlation results to TSV files."""
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        method_suffix = f"_{method_label}" if method_label else ""

        for name, result in results.items():
            if result.n_features == 0:
                continue
            df = result.to_dataframe()
            if df.empty:
                continue

            if "target" not in df.columns:
                df["target"] = target_name

            path = self.stats_dir / f"corr_stats_{name}_vs_{target_name}{method_suffix}.tsv"
            save_correlation_results(df, path)
            saved_files.append(path)

        return saved_files

    def compute_roi_correlations(
        self,
        power_df: pd.DataFrame,
        targets: pd.Series,
        target_name: str,
        corr_config: CorrelationConfig,
    ) -> Optional[pd.DataFrame]:
        """Compute ROI-level power correlations by averaging channels within ROIs.
        
        Handles column naming patterns:
        - power_{segment}_{band}_ch_{channel}_{stat} (e.g., power_active_delta_ch_Fp2_logratio)
        """
        from eeg_pipeline.utils.analysis.tfr import get_rois
        
        roi_defs = get_rois(self.config)
        if not roi_defs:
            self.logger.debug("No ROI definitions found in config")
            return None
        
        bands = self.config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        
        parsed_cols: List[Tuple[str, str, str, str]] = []
        col_to_channel: Dict[str, str] = {}
        for col in power_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not (parsed.get("valid") and parsed.get("group") == "power"):
                continue
            if parsed.get("scope") != "ch":
                continue
            band = parsed.get("band")
            segment = parsed.get("segment")
            identifier = parsed.get("identifier")
            if band and segment and identifier:
                parsed_cols.append((str(col), str(band), str(segment), str(identifier)))
                col_to_channel[str(col)] = str(identifier)
        
        method_label = format_correlation_method_label(corr_config.method, corr_config.robust_method)
        records = []
        for band in bands:
            band_l = str(band).lower()
            band_cols = [col for col, b, _seg, _ch in parsed_cols if b.lower() == band_l]
            
            # Prefer active columns if available
            active_cols = [
                col for col, b, seg, _ch in parsed_cols
                if b.lower() == band_l and seg.lower() == "active"
            ]
            if active_cols:
                band_cols = active_cols
            
            if not band_cols:
                continue
            
            for roi_name, patterns in roi_defs.items():
                roi_cols = []
                for col in band_cols:
                    ch_name = col_to_channel.get(col)
                    if ch_name is None:
                        continue
                    for pattern in patterns:
                        if re.match(pattern, ch_name, re.IGNORECASE):
                            roi_cols.append(col)
                            break
                
                if not roi_cols:
                    continue
                
                roi_vals = power_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                valid = roi_vals.notna() & targets.notna()
                
                if valid.sum() < corr_config.min_samples:
                    continue
                
                r, p, n = safe_correlation(
                    roi_vals[valid].values,
                    targets[valid].values,
                    corr_config.method,
                    corr_config.min_samples,
                    robust_method=corr_config.robust_method,
                )
                if not np.isfinite(r):
                    continue
                
                records.append({
                    "roi": roi_name,
                    "band": band,
                    "r": r,
                    "p": p,
                    "n": n,
                    "method": corr_config.method,
                    "robust_method": corr_config.robust_method,
                    "method_label": method_label,
                    "target": target_name,
                })
            
            overall_vals = power_df[band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            valid = overall_vals.notna() & targets.notna()
            if valid.sum() >= corr_config.min_samples:
                r, p, n = safe_correlation(
                    overall_vals[valid].values,
                    targets[valid].values,
                    corr_config.method,
                    corr_config.min_samples,
                    robust_method=corr_config.robust_method,
                )
                if not np.isfinite(r):
                    continue
                records.append({
                    "roi": "overall",
                    "band": band,
                    "r": r,
                    "p": p,
                    "n": n,
                    "method": corr_config.method,
                    "robust_method": corr_config.robust_method,
                    "method_label": method_label,
                    "target": target_name,
                })
        
        if not records:
            self.logger.warning("No ROI correlations computed - check power column naming")
            return None
        
        df = pd.DataFrame(records)
        # Add multiplicity control (per band) and raw aliases for consistency
        df["p_raw"] = df["p"]
        df["p_primary"] = df["p"]
        df["p_kind_primary"] = "p"
        df["p_primary_source"] = "raw"
        if corr_config.apply_fdr and "band" in df.columns:
            for band, mask in df.groupby("band").groups.items():
                band_idx = list(mask)
                p_vals = df.loc[band_idx, "p"].to_numpy()
                df.loc[band_idx, "p_fdr"] = fdr_bh(p_vals)
        elif corr_config.apply_fdr:
            df["p_fdr"] = fdr_bh(df["p"].to_numpy())
        suffix = "rating" if "rating" in target_name.lower() else "temp"
        method_suffix = f"_{method_label}" if method_label else ""
        path = self.stats_dir / f"corr_stats_pow_roi_vs_{suffix}{method_suffix}.tsv"
        write_tsv(df, path)
        self.logger.info(f"Saved ROI correlations: {path.name} ({len(df)} rows)")
        return df

    def run_complete_analysis(
        self,
        rating_series: pd.Series,
        temperature_series: Optional[pd.Series] = None,
        corr_config: Optional[CorrelationConfig] = None,
    ) -> ComputationResult:
        """Run complete feature-behavior correlation analysis."""
        self.load_all_features()

        if not self._feature_dfs:
            return ComputationResult(
                name="feature_correlator",
                status=ComputationStatus.SKIPPED,
                metadata={"reason": "No features loaded"},
            )

        if corr_config is None:
            corr_config = self.default_corr_config

        rating_records: List[Dict[str, Any]] = []
        temperature_records: List[Dict[str, Any]] = []
        metadata = {"n_feature_types": len(self._feature_dfs)}
        method_label = format_correlation_method_label(corr_config.method, corr_config.robust_method)

        if rating_series is not None and len(rating_series) > 0:
            rating_results = self.correlate_all(rating_series, "rating", corr_config)
            self.save_results(rating_results, "rating", method_label=method_label)

            for name, result in rating_results.items():
                metadata[f"{name}_n_features"] = result.n_features
                metadata[f"{name}_n_total"] = result.n_total or result.n_features
                metadata[f"{name}_n_dropped"] = result.n_dropped
                metadata[f"{name}_n_significant"] = result.n_significant
                rating_records.extend(result.records)
            
            if "power" in self._feature_dfs:
                self.compute_roi_correlations(
                    self._feature_dfs["power"], rating_series, "rating", corr_config
                )

        min_samples_default = get_min_samples(self.config, "default")
        if temperature_series is not None and len(temperature_series.dropna()) > min_samples_default:
            temp_results = self.correlate_all(temperature_series, "temperature", corr_config)
            self.save_results(temp_results, "temperature", method_label=method_label)

            for name, result in temp_results.items():
                metadata[f"{name}_n_features_temperature"] = result.n_features
                metadata[f"{name}_n_total_temperature"] = result.n_total or result.n_features
                metadata[f"{name}_n_dropped_temperature"] = result.n_dropped
                metadata[f"{name}_n_significant_temperature"] = result.n_significant
                temperature_records.extend(result.records)
            
            if "power" in self._feature_dfs:
                self.compute_roi_correlations(
                    self._feature_dfs["power"], temperature_series, "temp", corr_config
                )

        method_suffix = f"_{method_label}" if method_label else ""
        combined_rating_df = pd.DataFrame(rating_records) if rating_records else pd.DataFrame()
        if not combined_rating_df.empty:
            if corr_config.apply_fdr:
                if (
                    "p_primary_perm" in combined_rating_df.columns
                    and combined_rating_df["p_primary_perm"].notna().any()
                ):
                    p_for_fdr = pd.to_numeric(combined_rating_df["p_primary_perm"], errors="coerce").to_numpy()
                else:
                    p_for_fdr = pd.to_numeric(combined_rating_df["p_primary"], errors="coerce").to_numpy()
                combined_rating_df["p_fdr"] = fdr_bh(p_for_fdr, alpha=corr_config.fdr_alpha, config=self.config)
            combined_path = self.stats_dir / f"corr_stats_all_features_vs_rating{method_suffix}.tsv"
            save_correlation_results(combined_rating_df, combined_path)

        combined_temperature_df = pd.DataFrame(temperature_records) if temperature_records else pd.DataFrame()
        if not combined_temperature_df.empty:
            if corr_config.apply_fdr:
                if (
                    "p_primary_perm" in combined_temperature_df.columns
                    and combined_temperature_df["p_primary_perm"].notna().any()
                ):
                    p_for_fdr = pd.to_numeric(combined_temperature_df["p_primary_perm"], errors="coerce").to_numpy()
                else:
                    p_for_fdr = pd.to_numeric(combined_temperature_df["p_primary"], errors="coerce").to_numpy()
                combined_temperature_df["p_fdr"] = fdr_bh(p_for_fdr, alpha=corr_config.fdr_alpha, config=self.config)
            combined_path = self.stats_dir / f"corr_stats_all_features_vs_temperature{method_suffix}.tsv"
            save_correlation_results(combined_temperature_df, combined_path)

        alpha = float(get_config_value(self.config, "statistics.sig_alpha", 0.05))
        n_sig = sum(1 for r in rating_records if r.get("p", 1) < alpha)
        self.logger.info(
            f"Complete (rating): {len(rating_records)} correlations, {n_sig} significant "
            f"(alpha={alpha})"
        )

        if temperature_records:
            n_sig_temp = sum(1 for r in temperature_records if r.get("p", 1) < alpha)
            self.logger.info(
                f"Complete (temperature): {len(temperature_records)} correlations, {n_sig_temp} significant "
                f"(alpha={alpha})"
            )

        combined_df = pd.DataFrame([*rating_records, *temperature_records]) if (rating_records or temperature_records) else pd.DataFrame()

        return ComputationResult(
            name="feature_correlator",
            status=ComputationStatus.SUCCESS,
            metadata=metadata,
            dataframe=combined_df if not combined_df.empty else None,
        )


# =============================================================================
# Entry Points
# =============================================================================


def run_unified_feature_correlations(ctx: BehaviorContext) -> ComputationResult:
    """Run unified feature correlations from BehaviorContext."""
    correlator = FeatureBehaviorCorrelator(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        config=ctx.config,
        logger=ctx.logger,
        stats_dir=ctx.stats_dir,
    )
    
    # Inject loaded data from context so context is the source of truth for data loading.
    # Keys here should match registry feature file types where possible.
    for name, df in ctx.iter_feature_tables():
        if df is not None and not df.empty:
            correlator._feature_dfs[name] = df
    
    # Mark as loaded so it doesn't try to reload from registry files
    correlator._loaded = True
    
    return correlator.run_complete_analysis(
        rating_series=ctx.targets,
        temperature_series=ctx.temperature,
        corr_config=CorrelationConfig.from_context(ctx),
    )


def _load_pain_feature_patterns(config: Any) -> List[str]:
    """Fetch pain-relevant feature regex patterns from config."""
    patterns = get_config_value(config, "behavior_analysis.pain_relevant_patterns", None)
    if patterns is None:
        raise ValueError(
            "Define behavior_analysis.pain_relevant_patterns in eeg_config.yaml "
            "as a list of regex strings for pain-relevant features."
        )
    if not isinstance(patterns, (list, tuple)):
        raise ValueError("behavior_analysis.pain_relevant_patterns must be a list.")
    cleaned = [str(p).strip() for p in patterns if str(p).strip()]
    if not cleaned:
        raise ValueError("behavior_analysis.pain_relevant_patterns is empty.")
    return cleaned


def correlate_pain_relevant_features(ctx: BehaviorContext) -> ComputationResult:
    """Correlate features most relevant to pain processing."""
    try:
        pain_patterns = _load_pain_feature_patterns(ctx.config)
    except ValueError as err:
        return ComputationResult(
            name="pain_relevant",
            status=ComputationStatus.FAILED,
            error=str(err),
        )

    correlator = FeatureBehaviorCorrelator(
        subject=ctx.subject, deriv_root=ctx.deriv_root,
        config=ctx.config, logger=ctx.logger, stats_dir=ctx.stats_dir,
    )
    correlator.load_all_features()

    # Filter to pain-relevant columns
    pain_features = {}
    for name, df in correlator._feature_dfs.items():
        cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pain_patterns)]
        if cols:
            pain_features[name] = df[cols]

    if not pain_features:
        return ComputationResult(name="pain_relevant", status=ComputationStatus.SKIPPED,
                                metadata={"reason": "No pain-relevant features"})

    corr_config = CorrelationConfig.from_context(ctx)
    method_label = format_correlation_method_label(corr_config.method, corr_config.robust_method)
    all_records = []

    for name, df in pain_features.items():
        result = correlator._correlate_df(df, ctx.targets, corr_config, f"{name}_pain")
        all_records.extend(result.records)

    if all_records:
        df = pd.DataFrame(all_records)
        method_suffix = f"_{method_label}" if method_label else ""
        save_correlation_results(
            df,
            ctx.stats_dir / f"corr_stats_pain_relevant_vs_rating{method_suffix}.tsv",
        )

    alpha = float(get_config_value(ctx.config, "statistics.sig_alpha", 0.05))
    n_sig = sum(1 for r in all_records if r.get("p", 1) < alpha)
    return ComputationResult(name="pain_relevant", status=ComputationStatus.SUCCESS,
                            metadata={"n_features": len(all_records), "n_significant": n_sig})


def generate_feature_coverage_report(ctx: BehaviorContext) -> Dict[str, Any]:
    """Generate feature coverage report."""
    features_dir = deriv_features_path(ctx.deriv_root, ctx.subject)
    registry = get_feature_registry(ctx.config)
    
    report = {
        "subject": ctx.subject,
        "feature_files": {},
        "feature_types": {},
        "total_features": 0,
        "available_files": [],
        "missing_files": [],
    }

    for name, filename in registry.files.items():
        path = features_dir / filename
        if path.exists():
            df = read_tsv(path)
            if df is not None and not df.empty:
                report["available_files"].append(name)
                report["feature_files"][name] = {"n_features": len(df.columns), "n_epochs": len(df)}
                report["total_features"] += len(df.columns)
                for col in df.columns:
                    ftype, _, _ = classify_feature(col, include_subtype=True, registry=registry)
                    report["feature_types"][ftype] = report["feature_types"].get(ftype, 0) + 1
        else:
            report["missing_files"].append(name)

    report["coverage_pct"] = len(report["available_files"]) / len(registry.files) * 100
    ctx.logger.info(f"Coverage: {len(report['available_files'])}/{len(registry.files)} files")
    return report


def compute_feature_importance_summary(ctx: BehaviorContext) -> pd.DataFrame:
    """Compute feature importance summary."""
    from eeg_pipeline.utils.analysis.stats import fdr_bh

    correlator = FeatureBehaviorCorrelator(
        subject=ctx.subject, deriv_root=ctx.deriv_root,
        config=ctx.config, logger=ctx.logger, stats_dir=ctx.stats_dir,
    )
    correlator.load_all_features()
    corr_config = CorrelationConfig.from_context(ctx)

    rating_results = correlator.correlate_all(ctx.targets, "rating", corr_config)
    temp_results = None
    min_samples_default = get_min_samples(ctx.config, "default")
    if ctx.temperature is not None and len(ctx.temperature.dropna()) > min_samples_default:
        temp_results = correlator.correlate_all(ctx.temperature, "temperature", corr_config)

    summary = []
    for name, result in rating_results.items():
        for rec in result.records:
            feature = rec.get("feature") or rec.get("identifier", "unknown")
            r_temp, p_temp = np.nan, np.nan
            if temp_results and name in temp_results:
                for tr in temp_results[name].records:
                    if tr.get("feature") == feature or tr.get("identifier") == feature:
                        r_temp, p_temp = tr.get("r", np.nan), tr.get("p", np.nan)
                        break

            abs_r = abs(rec.get("r", 0))
            effect = "large" if abs_r >= 0.5 else "medium" if abs_r >= 0.3 else "small" if abs_r >= 0.1 else "negligible"

            summary.append({
                "feature": feature,
                "feature_type": rec.get("feature_type", name),
                "band": rec.get("band", "N/A"),
                "r_rating": rec.get("r"),
                "p_rating": rec.get("p"),
                "r_temperature": r_temp,
                "p_temperature": p_temp,
                "effect_size": effect,
                "n": rec.get("n"),
            })

    if not summary:
        return pd.DataFrame()

    df = pd.DataFrame(summary)
    valid_p = df["p_rating"].notna()
    if valid_p.any():
        df.loc[valid_p, "q_rating"] = fdr_bh(df.loc[valid_p, "p_rating"].values)

    df = df.assign(abs_r=df["r_rating"].abs()).sort_values("abs_r", ascending=False).drop(columns=["abs_r"])
    write_tsv(df, ctx.stats_dir / "feature_importance_summary.tsv")
    ctx.logger.info(f"Saved feature importance: {len(df)} features")
    return df
