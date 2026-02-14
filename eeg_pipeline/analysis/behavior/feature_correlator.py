"""
Unified Feature-Behavior Correlator
====================================

Single entry point for correlating ALL EEG features with behavioral measures.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from eeg_pipeline.context.behavior import BehaviorContext, ComputationResult, ComputationStatus
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.registry import get_feature_registry
from eeg_pipeline.infra.paths import deriv_features_path
from eeg_pipeline.infra.tsv import read_tsv, write_tsv
from eeg_pipeline.utils.analysis.stats import (
    compute_partial_correlations_with_cov_temp,
    compute_permutation_pvalues_with_cov_temp,
    fdr_bh,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    align_features_and_targets,
    align_groups_to_series as _align_groups_to_series,
    format_correlation_method_label,
    normalize_correlation_method,
    safe_correlation,
    save_correlation_results,
)
from eeg_pipeline.utils.analysis.stats.reliability import compute_correlation_split_half_reliability
from eeg_pipeline.utils.config.loader import get_config_value, get_min_samples
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_feature_types
from eeg_pipeline.analysis.behavior.config_resolver import resolve_correlation_method


def _build_stats_config_snapshot(config: Any) -> Dict[str, Any]:
    """Build a small, pickle-friendly config subset for parallel stats workers."""
    temperature_control = str(
        get_config_value(config, "behavior_analysis.statistics.temperature_control", "spline")
    ).strip().lower() or "spline"
    perm_scheme = str(
        get_config_value(config, "behavior_analysis.permutation.scheme", "shuffle")
    ).strip().lower() or "shuffle"
    spline_cfg = get_config_value(config, "behavior_analysis.regression.temperature_spline", {}) or {}
    spline_cfg = dict(spline_cfg) if isinstance(spline_cfg, dict) else {}

    return {
        "behavior_analysis": {
            "statistics": {"temperature_control": temperature_control},
            "permutation": {"scheme": perm_scheme},
            "regression": {"temperature_spline": spline_cfg},
        }
    }


def _is_temperature_target_name(target_name: str) -> bool:
    name = str(target_name or "").strip().lower()
    return name in {"temp", "temperature"} or "temperature" in name


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
        logger = getattr(ctx, "logger", None) if ctx is not None else None
        method = resolve_correlation_method(
            config,
            logger=logger,
            default="spearman",
        )
        min_samples = get_min_samples(config, "channel")
        fdr_alpha = float(get_config_value(
            config, "behavior_analysis.statistics.fdr_alpha",
            get_config_value(config, "statistics.fdr_alpha", 0.05)
        ))
        n_bootstrap = int(get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 1000))
        n_permutations = int(get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000))
        compute_bayes_factor = bool(
            get_config_value(config, "behavior_analysis.correlations.compute_bayes_factors", False)
        )
        robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
        if robust_method is not None:
            robust_method = str(robust_method).strip().lower() or None
        compute_loso_stability = bool(
            get_config_value(config, "behavior_analysis.correlations.loso_stability", True)
        )
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


@dataclass
class ColumnProcessingParams:
    """Parameters for processing a single feature column."""

    column_name: str
    column_values: np.ndarray
    feature_type: str
    targets_aligned: np.ndarray
    config: Any
    column_min_samples: int
    n_total: int
    missing_fraction: float
    variance: float
    covariates_aligned: Optional[pd.DataFrame]
    temperature_aligned: Optional[pd.Series]
    loso_groups: Optional[np.ndarray]
    permutation_groups: Optional[np.ndarray]
    correlation_method: str
    robust_method: Optional[str]
    method_label: str
    target_name: str
    n_permutations: int
    n_bootstrap: int
    random_seed: int
    control_temperature: bool
    control_trial_order: bool
    compute_bayes_factor: bool
    compute_loso_stability: bool
    compute_reliability: bool


# =============================================================================
# Parallel Column Processing Helpers
# =============================================================================


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
    from eeg_pipeline.utils.analysis.stats.bootstrap import compute_bootstrap_ci
    
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
    config: Optional[Any],
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
        config=config,
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
    config: Optional[Any],
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
        config=config,
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
    primary_to_permutation_column = {
        "p_raw": "p_perm_raw",
        "p_partial_cov": "p_perm_partial_cov",
        "p_partial_temp": "p_perm_partial_temp",
        "p_partial_cov_temp": "p_perm_partial_cov_temp",
    }
    primary_kind = record.get("p_kind_primary")
    permutation_column = primary_to_permutation_column.get(primary_kind)
    if permutation_column is not None:
        record["p_primary_perm"] = record.get(permutation_column, np.nan)
        record["p_primary_is_permutation"] = bool(pd.notna(record.get("p_primary_perm", np.nan)))


def _apply_fdr_correction(
    records: List[Dict[str, Any]],
    config: CorrelationConfig,
    use_permutation_pvalues: bool,
    analysis_config: Optional[Any] = None,
) -> None:
    """Apply FDR correction to correlation records."""
    if not records or not config.apply_fdr:
        return
    
    if use_permutation_pvalues:
        p_values = [record.get("p_primary_perm", np.nan) for record in records]
        p_kind = "p_primary_perm"
    else:
        p_values = [record.get("p_primary", np.nan) for record in records]
        p_kind = "p_primary"
    
    valid_indices = [i for i, p_val in enumerate(p_values) if pd.notna(p_val)]
    if not valid_indices:
        return
    
    valid_p_values = np.array([p_values[i] for i in valid_indices])
    q_values = fdr_bh(valid_p_values, alpha=config.fdr_alpha, config=analysis_config)
    
    for index, q_value in zip(valid_indices, q_values):
        records[index]["p_fdr"] = float(q_value)
        records[index]["q_within_family"] = float(q_value)
        records[index]["within_family_p_kind"] = p_kind


def _save_screening_records(
    screening_records: List[Dict[str, Any]],
    feature_type: str,
    target_name: str,
    stats_dir: Optional[Path],
    _logger: logging.Logger,
) -> None:
    """Save feature screening records to disk."""
    if not screening_records or stats_dir is None:
        return

    from eeg_pipeline.infra.paths import ensure_dir

    screen_dir = stats_dir / "feature_screening"
    ensure_dir(screen_dir)
    screen_df = pd.DataFrame(screening_records)
    screen_path = screen_dir / f"feature_screening_{feature_type}_vs_{target_name}.tsv"
    write_tsv(screen_df, screen_path)


def _apply_fdr_to_dataframe(
    dataframe: pd.DataFrame,
    config: CorrelationConfig,
    analysis_config: Any,
) -> None:
    """Apply FDR correction to a dataframe of correlation results."""
    if not config.apply_fdr or dataframe.empty:
        return
    
    has_permutation_pvalues = (
        "p_primary_perm" in dataframe.columns
        and dataframe["p_primary_perm"].notna().any()
    )
    
    if has_permutation_pvalues:
        p_values = pd.to_numeric(dataframe["p_primary_perm"], errors="coerce").to_numpy()
    else:
        p_values = pd.to_numeric(dataframe["p_primary"], errors="coerce").to_numpy()
    
    dataframe["p_fdr"] = fdr_bh(p_values, alpha=config.fdr_alpha, config=analysis_config)
    dataframe["q_within_family"] = dataframe["p_fdr"]
    dataframe["within_family_p_kind"] = "p_primary_perm" if has_permutation_pvalues else "p_primary"


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


def _process_single_column(params: ColumnProcessingParams) -> Optional[Dict[str, Any]]:
    """Process correlations for a single column. Designed for parallel execution."""
    from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
    
    random_generator = np.random.default_rng(params.random_seed)
    
    correlation_coefficient, p_value, n_valid = safe_correlation(
        params.column_values,
        params.targets_aligned,
        params.correlation_method,
        params.column_min_samples,
        robust_method=params.robust_method,
    )
    if not np.isfinite(correlation_coefficient):
        return None
    
    n_covariates = int(params.covariates_aligned.shape[1]) if params.covariates_aligned is not None else 0
    record = _build_base_record(
        params.column_name,
        params.feature_type,
        correlation_coefficient,
        p_value,
        n_valid,
        params.n_total,
        params.missing_fraction,
        params.variance,
        params.correlation_method,
        n_covariates,
        params.robust_method,
        params.method_label,
        params.target_name,
    )
    
    feature_series = pd.Series(params.column_values)
    target_series = pd.Series(params.targets_aligned)

    if params.n_bootstrap > 0:
        try:
            _add_bootstrap_ci(
                record,
                params.column_values,
                params.targets_aligned,
                params.n_bootstrap,
                params.correlation_method,
                random_generator,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Bootstrap CI failed for feature '{params.column_name}' "
                f"(type={params.feature_type}, target={params.target_name})"
            ) from exc

    if params.compute_bayes_factor:
        try:
            _add_bayes_factor(
                record,
                params.column_values,
                params.targets_aligned,
                params.correlation_method,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Bayes factor failed for feature '{params.column_name}' "
                f"(type={params.feature_type}, target={params.target_name})"
            ) from exc

    has_covariates = params.covariates_aligned is not None
    has_temperature = params.temperature_aligned is not None
    if has_covariates or has_temperature:
        try:
            _add_partial_correlations(
                record,
                feature_series,
                target_series,
                params.covariates_aligned,
                params.temperature_aligned,
                params.correlation_method,
                params.feature_type,
                params.column_min_samples,
                params.config,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Partial correlations failed for feature '{params.column_name}' "
                f"(type={params.feature_type}, target={params.target_name})"
            ) from exc

    if params.n_permutations > 0:
        try:
            _add_permutation_pvalues(
                record,
                feature_series,
                target_series,
                params.covariates_aligned,
                params.temperature_aligned,
                params.correlation_method,
                params.n_permutations,
                random_generator,
                params.permutation_groups,
                params.config,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Permutation p-values failed for feature '{params.column_name}' "
                f"(type={params.feature_type}, target={params.target_name})"
            ) from exc

    _select_primary_correlation(record, params.control_temperature, params.control_trial_order)

    if params.n_permutations > 0:
        _update_primary_perm_pvalue(record)

    has_loso_groups = params.loso_groups is not None
    if params.compute_loso_stability and has_loso_groups:
        try:
            _add_loso_stability(
                record,
                params.column_values,
                params.targets_aligned,
                params.loso_groups,
                params.correlation_method,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"LOSO stability failed for feature '{params.column_name}' "
                f"(type={params.feature_type}, target={params.target_name})"
            ) from exc

    if params.compute_reliability:
        try:
            _add_reliability(
                record,
                params.column_values,
                params.targets_aligned,
                params.correlation_method,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Split-half reliability failed for feature '{params.column_name}' "
                f"(type={params.feature_type}, target={params.target_name})"
            ) from exc

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
        if not subject or not isinstance(subject, str):
            raise ValueError("subject must be a non-empty string")
        if not isinstance(deriv_root, Path):
            raise TypeError("deriv_root must be a Path object")
        if config is None:
            raise ValueError("config cannot be None")
        if logger is None:
            raise ValueError("logger cannot be None")
        
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
            return {feature_type: len(dataframe.columns) for feature_type, dataframe in self._feature_dfs.items()}

        feature_counts = {}
        for feature_type, filename in self.registry.files.items():
            file_path = self.features_dir / filename
            if not file_path.exists():
                continue
            
            dataframe = read_tsv(file_path)
            if dataframe is not None and not dataframe.empty:
                self._feature_dfs[feature_type] = dataframe
                feature_counts[feature_type] = len(dataframe.columns)

        self._loaded = True
        total_features = sum(feature_counts.values())
        self.logger.info(f"Loaded {len(self._feature_dfs)} feature files, {total_features} features")
        return feature_counts

    def correlate_all(
        self,
        targets: pd.Series,
        target_name: str = "rating",
        corr_config: Optional[CorrelationConfig] = None,
        n_jobs: int = -1,
    ) -> Dict[str, FeatureCorrelationResult]:
        """Correlate all loaded features with behavioral target."""
        if targets is None or len(targets) == 0:
            raise ValueError("targets must be a non-empty Series")
        if not isinstance(target_name, str) or not target_name:
            raise ValueError("target_name must be a non-empty string")
        
        if not self._loaded:
            self.load_all_features()

        if corr_config is None:
            corr_config = self.default_corr_config

        n_jobs_actual = get_n_jobs(self.config, n_jobs)
        self.logger.info(f"Correlating features with {target_name}... (n_jobs={n_jobs_actual})")
        
        def correlate_with_groups(dataframe, target_values, correlation_config, feature_type_name):
            return self._correlate_df(
                dataframe,
                target_values,
                correlation_config,
                feature_type_name,
                target_name=target_name,
                subject_ids=correlation_config.groups,
            )

        results = parallel_feature_types(
            feature_dfs=self._feature_dfs,
            targets=targets,
            correlate_func=correlate_with_groups,
            corr_config=corr_config,
            n_jobs=n_jobs_actual,
            logger=self.logger,
        )
        
        for feature_type_name, result in results.items():
            if result.n_features > 0:
                n_dropped = result.n_dropped
                n_total = result.n_total or (result.n_features + n_dropped)
                self.logger.debug(
                    f"  {feature_type_name}: {result.n_features}/{n_total} tested, {n_dropped} dropped, {result.n_significant} sig"
                )

        return results

    def _align_data_for_correlation(
        self,
        df_aligned: pd.DataFrame,
        targets_aligned: pd.Series,
        config: CorrelationConfig,
        subject_ids: Optional[np.ndarray],
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[np.ndarray], Optional[np.ndarray]]:
        """Align covariates, temperature, and groups for correlation analysis."""
        covariates_aligned = None
        if config.control_temperature:
            if config.covariates_without_temp_df is not None:
                covariates_aligned = config.covariates_without_temp_df.reindex(df_aligned.index)
        elif config.covariates_df is not None:
            covariates_aligned = config.covariates_df.reindex(df_aligned.index)
        
        temperature_aligned = None
        if config.control_temperature and config.temperature_series is not None:
            temperature_aligned = config.temperature_series.reindex(df_aligned.index)
        
        loso_groups = None
        permutation_groups = None
        if subject_ids is not None:
            loso_groups = _align_groups_to_series(targets_aligned, subject_ids)
            permutation_groups = loso_groups
        
        if permutation_groups is None and config.groups is not None:
            permutation_groups = _align_groups_to_series(targets_aligned, config.groups)
            if loso_groups is None:
                loso_groups = permutation_groups
        
        return covariates_aligned, temperature_aligned, loso_groups, permutation_groups

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

        effective_config = config
        if config.control_temperature and _is_temperature_target_name(target_name):
            # Scientific validity: never "control for temperature" when temperature is the target.
            effective_config = replace(config, control_temperature=False, temperature_series=None)

        min_samples_align = get_min_samples(self.config, "default")
        df_aligned, targets_aligned = align_features_and_targets(
            df, targets, min_samples_align, self.logger
        )
        if df_aligned is None or targets_aligned is None:
            return FeatureCorrelationResult(feature_type, 0, 0)

        cov_aligned, temp_aligned, loso_groups, perm_groups = self._align_data_for_correlation(
            df_aligned, targets_aligned, effective_config, subject_ids
        )
        
        n_jobs_actual = effective_config.n_jobs if effective_config.n_jobs != -1 else max(1, cpu_count() - 1)
        
        base_seed = int(get_config_value(self.config, "behavior_analysis.statistics.base_seed", 42))
        if effective_config.rng is not None:
            base_seed = int(effective_config.rng.integers(0, 2**31))
        
        targets_arr = targets_aligned.values if hasattr(targets_aligned, 'values') else np.asarray(targets_aligned)
        n_total = int(len(targets_arr))
        loso_groups_arr = loso_groups if loso_groups is not None else None
        perm_groups_arr = perm_groups if perm_groups is not None else None
        
        column_params_list = []
        screening_records = []
        method_label = effective_config.method_label or format_correlation_method_label(
            effective_config.method, effective_config.robust_method
        )
        stats_config_snapshot = _build_stats_config_snapshot(self.config)
        
        for column_index, column_name in enumerate(df_aligned.columns):
            column_values = pd.to_numeric(df_aligned[column_name], errors="coerce").values
            min_samples = _infer_min_samples_for_column(column_name, feature_type, self.config)
            valid_mask = np.isfinite(column_values) & np.isfinite(targets_arr)
            n_valid = int(valid_mask.sum())
            missing_fraction = 1.0 - (float(n_valid) / n_total) if n_total > 0 else 1.0
            variance = float(np.nanvar(column_values[valid_mask])) if n_valid > 1 else np.nan
            
            has_zero_variance = not np.isfinite(variance) or np.isclose(variance, 0.0)
            status = "dropped" if has_zero_variance else "kept"
            reason = "zero_variance" if has_zero_variance else None

            screening_records.append(
                {
                    "feature": str(column_name),
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

            params = ColumnProcessingParams(
                column_name=column_name,
                column_values=column_values,
                feature_type=feature_type,
                targets_aligned=targets_arr,
                config=stats_config_snapshot,
                column_min_samples=min_samples,
                n_total=n_total,
                missing_fraction=missing_fraction,
                variance=variance,
                covariates_aligned=cov_aligned,
                temperature_aligned=temp_aligned,
                loso_groups=loso_groups_arr,
                permutation_groups=perm_groups_arr,
                correlation_method=effective_config.method,
                robust_method=effective_config.robust_method,
                method_label=method_label,
                target_name=target_name,
                n_permutations=effective_config.n_permutations,
                n_bootstrap=effective_config.n_bootstrap,
                random_seed=base_seed + column_index,
                control_temperature=effective_config.control_temperature,
                control_trial_order=effective_config.control_trial_order,
                compute_bayes_factor=effective_config.compute_bayes_factor,
                compute_loso_stability=effective_config.compute_loso_stability,
                compute_reliability=effective_config.compute_reliability,
            )
            column_params_list.append(params)

        _save_screening_records(screening_records, feature_type, target_name, self.stats_dir, self.logger)

        n_columns = len(column_params_list)
        should_use_parallel = n_jobs_actual > 1 and n_columns > 10
        
        if should_use_parallel:
            record_dicts = Parallel(n_jobs=n_jobs_actual, backend="loky")(
                delayed(_process_single_column)(params) for params in column_params_list
            )
            record_dicts = [record for record in record_dicts if record is not None]
        else:
            record_dicts = []
            for params in column_params_list:
                record = _process_single_column(params)
                if record is not None:
                    record_dicts.append(record)

        use_permutation_pvalues = config.n_permutations > 0
        _apply_fdr_correction(record_dicts, config, use_permutation_pvalues, self.config)

        significance_alpha = float(get_config_value(self.config, "statistics.sig_alpha", 0.05))
        n_significant = sum(
            1
            for record in record_dicts
            if pd.notna(record.get("p_primary", np.nan)) and float(record.get("p_primary")) < significance_alpha
        )
        n_dropped = len(df_aligned.columns) - len(column_params_list)
        return FeatureCorrelationResult(
            feature_type,
            len(record_dicts),
            n_significant,
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
        """Save correlation results to TSV files in correlations subfolder."""
        from eeg_pipeline.infra.paths import ensure_dir
        corr_dir = self.stats_dir / "correlations"
        ensure_dir(corr_dir)
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

            path = corr_dir / f"corr_stats_{name}_vs_{target_name}{method_suffix}.tsv"
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
        
        if power_df is None or power_df.empty or targets is None or len(targets) == 0:
            return None
        
        roi_definitions = get_rois(self.config)
        if not roi_definitions:
            self.logger.debug("No ROI definitions found in config")
            return None
        
        bands = self.config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        
        parsed_columns: List[Tuple[str, str, str, str]] = []
        column_to_channel: Dict[str, str] = {}
        for column in power_df.columns:
            parsed = NamingSchema.parse(str(column))
            is_valid_power_column = parsed.get("valid") and parsed.get("group") == "power"
            is_channel_scope = parsed.get("scope") == "ch"
            if not (is_valid_power_column and is_channel_scope):
                continue
            
            band = parsed.get("band")
            segment = parsed.get("segment")
            identifier = parsed.get("identifier")
            if band and segment and identifier:
                parsed_columns.append((str(column), str(band), str(segment), str(identifier)))
                column_to_channel[str(column)] = str(identifier)
        
        effective_config = corr_config
        if corr_config.control_temperature and _is_temperature_target_name(target_name):
            effective_config = replace(corr_config, control_temperature=False, temperature_series=None)

        method_label = format_correlation_method_label(effective_config.method, effective_config.robust_method)
        records: List[Dict[str, Any]] = []
        for band in bands:
            band_lower = str(band).lower()
            band_columns = [col for col, b, _seg, _ch in parsed_columns if b.lower() == band_lower]
            
            active_columns = [
                col for col, b, seg, _ch in parsed_columns
                if b.lower() == band_lower and seg.lower() == "active"
            ]
            if active_columns:
                band_columns = active_columns
            
            if not band_columns:
                continue

            band_matrix = power_df[band_columns].apply(pd.to_numeric, errors="coerce")

            for roi_name, patterns in roi_definitions.items():
                roi_columns = []
                for col in band_columns:
                    channel_name = column_to_channel.get(col)
                    if channel_name is None:
                        continue
                    for pattern in patterns:
                        if re.match(pattern, channel_name, re.IGNORECASE):
                            roi_columns.append(col)
                            break
                
                if not roi_columns:
                    continue
                
                roi_values = band_matrix[roi_columns].mean(axis=1)
                df_pair = pd.concat([roi_values.rename("x"), targets.rename("y")], axis=1).dropna()
                if df_pair.empty:
                    continue

                cov_aligned = None
                if effective_config.control_trial_order:
                    if effective_config.control_temperature:
                        if effective_config.covariates_without_temp_df is not None:
                            cov_aligned = effective_config.covariates_without_temp_df.reindex(df_pair.index)
                    elif effective_config.covariates_df is not None:
                        cov_aligned = effective_config.covariates_df.reindex(df_pair.index)

                temp_aligned = None
                if effective_config.control_temperature and effective_config.temperature_series is not None:
                    temp_aligned = effective_config.temperature_series.reindex(df_pair.index)
                
                correlation_coefficient, p_value, n_valid = safe_correlation(
                    df_pair["x"].values,
                    df_pair["y"].values,
                    effective_config.method,
                    effective_config.min_samples,
                    robust_method=effective_config.robust_method,
                )
                if not np.isfinite(correlation_coefficient):
                    continue

                record: Dict[str, Any] = {
                    "roi": roi_name,
                    "band": band,
                    "r": correlation_coefficient,
                    "p": p_value,
                    "n": n_valid,
                    "method": effective_config.method,
                    "robust_method": effective_config.robust_method,
                    "method_label": method_label,
                    "target": target_name,
                    "r_raw": correlation_coefficient,
                    "p_raw": p_value,
                    "p_primary": p_value,
                    "r_primary": correlation_coefficient,
                    "p_kind_primary": "p_raw",
                    "p_primary_source": "raw",
                    "p_primary_perm": np.nan,
                    "p_primary_is_permutation": False,
                }

                if cov_aligned is not None or temp_aligned is not None:
                    _add_partial_correlations(
                        record,
                        df_pair["x"],
                        df_pair["y"],
                        cov_aligned,
                        temp_aligned,
                        effective_config.method,
                        feature_type="power_roi",
                        min_samples=effective_config.min_samples,
                        config=self.config,
                    )
                    _select_primary_correlation(
                        record,
                        effective_config.control_temperature,
                        effective_config.control_trial_order,
                    )

                if effective_config.n_permutations and effective_config.n_permutations > 0:
                    rng = effective_config.rng
                    if rng is None:
                        base_seed = int(get_config_value(self.config, "behavior_analysis.statistics.base_seed", 42))
                        rng = np.random.default_rng(base_seed)
                    _add_permutation_pvalues(
                        record,
                        df_pair["x"],
                        df_pair["y"],
                        cov_aligned,
                        temp_aligned,
                        effective_config.method,
                        effective_config.n_permutations,
                        rng,
                        perm_groups=None,
                        config=self.config,
                    )
                    _update_primary_perm_pvalue(record)

                records.append(record)
            
            overall_values = band_matrix.mean(axis=1)
            df_pair = pd.concat([overall_values.rename("x"), targets.rename("y")], axis=1).dropna()
            if not df_pair.empty:
                cov_aligned = None
                if effective_config.control_trial_order:
                    if effective_config.control_temperature:
                        if effective_config.covariates_without_temp_df is not None:
                            cov_aligned = effective_config.covariates_without_temp_df.reindex(df_pair.index)
                    elif effective_config.covariates_df is not None:
                        cov_aligned = effective_config.covariates_df.reindex(df_pair.index)

                temp_aligned = None
                if effective_config.control_temperature and effective_config.temperature_series is not None:
                    temp_aligned = effective_config.temperature_series.reindex(df_pair.index)

                correlation_coefficient, p_value, n_valid = safe_correlation(
                    df_pair["x"].values,
                    df_pair["y"].values,
                    effective_config.method,
                    effective_config.min_samples,
                    robust_method=effective_config.robust_method,
                )
                if not np.isfinite(correlation_coefficient):
                    continue

                record: Dict[str, Any] = {
                    "roi": "overall",
                    "band": band,
                    "r": correlation_coefficient,
                    "p": p_value,
                    "n": n_valid,
                    "method": effective_config.method,
                    "robust_method": effective_config.robust_method,
                    "method_label": method_label,
                    "target": target_name,
                    "r_raw": correlation_coefficient,
                    "p_raw": p_value,
                    "p_primary": p_value,
                    "r_primary": correlation_coefficient,
                    "p_kind_primary": "p_raw",
                    "p_primary_source": "raw",
                    "p_primary_perm": np.nan,
                    "p_primary_is_permutation": False,
                }

                if cov_aligned is not None or temp_aligned is not None:
                    _add_partial_correlations(
                        record,
                        df_pair["x"],
                        df_pair["y"],
                        cov_aligned,
                        temp_aligned,
                        effective_config.method,
                        feature_type="power_roi",
                        min_samples=effective_config.min_samples,
                        config=self.config,
                    )
                    _select_primary_correlation(
                        record,
                        effective_config.control_temperature,
                        effective_config.control_trial_order,
                    )

                if effective_config.n_permutations and effective_config.n_permutations > 0:
                    rng = effective_config.rng
                    if rng is None:
                        base_seed = int(get_config_value(self.config, "behavior_analysis.statistics.base_seed", 42))
                        rng = np.random.default_rng(base_seed)
                    _add_permutation_pvalues(
                        record,
                        df_pair["x"],
                        df_pair["y"],
                        cov_aligned,
                        temp_aligned,
                        effective_config.method,
                        effective_config.n_permutations,
                        rng,
                        perm_groups=None,
                        config=self.config,
                    )
                    _update_primary_perm_pvalue(record)

                records.append(record)
        
        if not records:
            self.logger.warning("No ROI correlations computed - check power column naming")
            return None
        
        df = pd.DataFrame(records)

        if effective_config.apply_fdr:
            _apply_fdr_to_dataframe(df, effective_config, self.config)
        target_suffix = "rating" if "rating" in target_name.lower() else "temp"
        method_suffix = f"_{method_label}" if method_label else ""
        from eeg_pipeline.infra.paths import ensure_dir
        corr_dir = self.stats_dir / "correlations"
        ensure_dir(corr_dir)
        output_path = corr_dir / f"corr_stats_pow_roi_vs_{target_suffix}{method_suffix}.tsv"
        write_tsv(df, output_path)
        self.logger.info(f"Saved ROI correlations: {output_path.name} ({len(df)} rows)")
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
        from eeg_pipeline.analysis.behavior.orchestration import _get_stats_subfolder_with_overwrite
        from eeg_pipeline.utils.config.loader import get_config_bool
        
        overwrite = get_config_bool(self.config, "behavior_analysis.output.overwrite", True)
        corr_dir = _get_stats_subfolder_with_overwrite(self.stats_dir, "correlations", overwrite)
        combined_rating_df = pd.DataFrame(rating_records) if rating_records else pd.DataFrame()
        if not combined_rating_df.empty:
            _apply_fdr_to_dataframe(combined_rating_df, corr_config, self.config)
            combined_path = corr_dir / f"corr_stats_all_features_vs_rating{method_suffix}.tsv"
            save_correlation_results(combined_rating_df, combined_path)

        combined_temperature_df = pd.DataFrame(temperature_records) if temperature_records else pd.DataFrame()
        if not combined_temperature_df.empty:
            _apply_fdr_to_dataframe(combined_temperature_df, corr_config, self.config)
            combined_path = corr_dir / f"corr_stats_all_features_vs_temperature{method_suffix}.tsv"
            save_correlation_results(combined_temperature_df, combined_path)

        significance_alpha = float(get_config_value(self.config, "statistics.sig_alpha", 0.05))
        n_significant_rating = sum(
            1 for record in rating_records
            if record.get("p", 1.0) < significance_alpha
        )
        self.logger.info(
            f"Complete (rating): {len(rating_records)} correlations, {n_significant_rating} significant "
            f"(alpha={significance_alpha})"
        )

        if temperature_records:
            n_significant_temperature = sum(
                1 for record in temperature_records
                if record.get("p", 1.0) < significance_alpha
            )
            self.logger.info(
                f"Complete (temperature): {len(temperature_records)} correlations, {n_significant_temperature} significant "
                f"(alpha={significance_alpha})"
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
    
    # Get rating from aligned_events using event_columns.rating config
    rating_series = None
    rating_col = ctx._find_rating_column() if hasattr(ctx, "_find_rating_column") else None
    if rating_col is not None and ctx.aligned_events is not None:
        rating_series = pd.to_numeric(ctx.aligned_events[rating_col], errors="coerce")
    
    return correlator.run_complete_analysis(
        rating_series=rating_series,
        temperature_series=ctx.temperature,
        corr_config=CorrelationConfig.from_context(ctx),
    )
