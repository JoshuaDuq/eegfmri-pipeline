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
    _align_groups_to_series,
    _align_features_and_targets,
    _build_temp_record_unified,
    _compute_roi_correlation_stats,
)
from eeg_pipeline.utils.io.paths import deriv_features_path
from eeg_pipeline.utils.io.tsv import read_tsv, write_tsv
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.context.behavior import AnalysisConfig
from eeg_pipeline.utils.config.loader import get_min_samples, get_config_value, load_config
from eeg_pipeline.utils.analysis.stats import (
    prepare_aligned_data,
    compute_correlation,
    compute_bootstrap_ci,
    compute_partial_correlations,
    compute_permutation_pvalues,
    compute_temp_permutation_pvalues,
    CorrelationStats,
    fdr_bh,
)
from eeg_pipeline.utils.parallel import parallel_feature_types, get_n_jobs
from eeg_pipeline.analysis.features.registry import (
    FeatureRegistry,
    FeatureRule,
    classify_feature,
    get_feature_registry,
)
import eeg_pipeline.analysis.features.registry as feature_registry
from eeg_pipeline.utils.analysis.stats.reliability import compute_correlation_split_half_reliability

_CHANNEL_NAMES = feature_registry._CHANNEL_NAMES


@dataclass
class CorrelationConfig:
    """Configuration for feature-behavior correlations."""

    method: str
    min_samples: int
    apply_fdr: bool = True
    n_bootstrap: int = 0
    n_permutations: int = 0
    rng: Optional[np.random.Generator] = None
    filter_threshold: float = 0.0
    compute_bayes_factor: bool = False
    robust_method: Optional[str] = None  # "percentage_bend", "winsorized", "shepherd"
    compute_loso_stability: bool = False
    compute_reliability: bool = False
    covariates_df: Optional[pd.DataFrame] = None
    covariates_without_temp_df: Optional[pd.DataFrame] = None
    temperature_series: Optional[pd.Series] = None
    groups: Optional[np.ndarray] = None
    control_temperature: bool = True
    control_trial_order: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "CorrelationConfig":
        """Build configuration using behavior_analysis settings."""
        return cls(
            method=get_config_value(
                config, "behavior_analysis.statistics.correlation_method", "spearman"
            ),
            min_samples=get_min_samples(config, "channel"),
            n_bootstrap=int(get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 0)),
            n_permutations=int(get_config_value(config, "behavior_analysis.statistics.n_permutations", 0)),
            rng=None,
            compute_bayes_factor=bool(get_config_value(config, "behavior_analysis.compute_bayes_factors", False)),
            robust_method=get_config_value(config, "behavior_analysis.robust_correlation", None),
            compute_loso_stability=bool(get_config_value(config, "behavior_analysis.loso_stability", False)),
            compute_reliability=bool(get_config_value(config, "behavior_analysis.statistics.compute_reliability", False)),
        )

    @classmethod
    def from_context(cls, ctx: BehaviorContext) -> "CorrelationConfig":
        """Context-aware configuration that honors runtime overrides."""
        base = cls.from_config(ctx.config or {})
        return cls(
            method=ctx.method or base.method,
            min_samples=ctx.min_samples_channel or base.min_samples,
            apply_fdr=base.apply_fdr,
            n_bootstrap=ctx.bootstrap if ctx.bootstrap is not None else base.n_bootstrap,
            n_permutations=ctx.n_perm if ctx.n_perm is not None else base.n_permutations,
            rng=ctx.rng,
            filter_threshold=base.filter_threshold,
            compute_bayes_factor=base.compute_bayes_factor,
            robust_method=base.robust_method,
            compute_loso_stability=base.compute_loso_stability,
            compute_reliability=getattr(ctx, "compute_reliability", False),
            covariates_df=getattr(ctx, "covariates_df", None),
            covariates_without_temp_df=getattr(ctx, "covariates_without_temp_df", None),
            temperature_series=getattr(ctx, "temperature", None) if getattr(ctx, "control_temperature", True) else None,
            groups=getattr(ctx, "group_ids", None),
            control_temperature=getattr(ctx, "control_temperature", True),
            control_trial_order=getattr(ctx, "control_trial_order", True),
        )


@dataclass
class FeatureCorrelationResult:
    """Result for a single feature type's correlations."""

    feature_type: str
    n_features: int
    n_significant: int
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records) if self.records else pd.DataFrame()


# =============================================================================
# Feature Classification
# =============================================================================


def _rule_matches(column: str, rule: FeatureRule) -> bool:
    """Delegate to shared registry rule matcher."""
    return feature_registry._rule_matches(column, rule)


def _classify_subtype(
    column: str,
    feature_type: str,
    registry: FeatureRegistry,
    source_file: Optional[str] = None,
) -> str:
    """Classify feature subtype using registry patterns and known hierarchies."""
    col_lower = column.lower()
    parts = column.split("_")

    # Pattern-based subtype hints
    patterns = registry.patterns

    if feature_type == "power":
        if source_file and "plateau" in source_file:
            return "plateau"
        if patterns.get("powcorr") and patterns["powcorr"].match(column):
            return "correlation"
        if "plateau" in col_lower:
            return "plateau"
        if col_lower.startswith("baseline_") or col_lower.startswith("power_baseline_"):
            return "baseline"
        return "direct"

    if feature_type == "connectivity":
        if patterns.get("conn_graph") and patterns["conn_graph"].match(column):
            return "graph"
        if len(parts) > 0:
            measure = parts[0].lower()
            if measure in ["aec", "wpli", "plv", "pli", "imcoh", "coh", "icoh", "corr", "sync"]:
                return measure
            if measure.startswith("sw") and "corr" in measure:
                return "sliding_window"
        return "unknown"

    if feature_type == "microstate":
        if patterns.get("ms_transition") and patterns["ms_transition"].match(column):
            return "transition"
        if len(parts) >= 2 and parts[0].lower() == "ms":
            metric = parts[1].lower()
            if metric in registry.type_hierarchy.get("microstate", {}).get("subtypes", []):
                return metric
        if column.isdigit():
            return "state"
        return "unknown"

    if feature_type == "precomputed":
        if len(parts) > 0:
            prefix = parts[0].lower()
            if prefix == "gfp":
                return "gfp"
            if prefix == "roi":
                return "roi"
            if prefix in ["pow", "power", "logpow", "relpow"]:
                return "power"
            if prefix in [
                "iaf",
                "sef50",
                "sef75",
                "sef90",
                "sef95",
                "se",
                "spec",
                "spectral",
                "peakfreq",
                "peakpow",
                "peakprom",
                "bandwidth",
                "relative",
                "ratio",
                "slope",
                "edge",
            ]:
                return "spectral"
            if prefix in ["mean", "var", "std", "skew", "kurt", "median", "iqr", "rms", "p2p"]:
                return "temporal"
            if prefix in ["pe", "sampen", "hjorth", "lzc", "hurst", "dfa", "entropy", "complexity"]:
                return "complexity"
            if prefix in ["conn", "plv", "imcoh", "aec", "psi"]:
                return "connectivity"
            if prefix in ["pac"]:
                return "pac"
            if prefix in ["itpc"]:
                return "itpc"
            if prefix in ["dynamics"]:
                return "dynamics"
            if prefix in ["asym", "asymmetry"]:
                return "asymmetry"
        return "other"

    if feature_type == "itpc":
        # Check source file or pattern
        if "trial" in col_lower: return "single_trial"
        if "map" in col_lower: return "map"
        if "plateau" in col_lower: return "plateau"
        if "ramp" in col_lower: return "ramp"
        return "summary"

    if feature_type == "pac":
        if "trial" in col_lower: return "trial"
        if "amp" in col_lower: return "amplitude"
        if "phase" in col_lower: return "phase"
        return "comodulogram"

    hierarchy_subtypes = registry.type_hierarchy.get(feature_type, {}).get("subtypes", [])
    if hierarchy_subtypes:
        for subtype in hierarchy_subtypes:
            if subtype in col_lower:
                return subtype

    return "unknown"


def _match_feature_patterns(
    column: str, registry: FeatureRegistry
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Try regex-based pattern matching for detailed classification."""
    patterns = registry.patterns

    for ftype, pattern in patterns.items():
        match = pattern.match(column)
        if not match:
            continue
        groups = match.groups()
        meta = {"identifier": column, "band": "N/A", "source": "inferred", "subtype": "unknown"}

        if ftype == "erds":
            meta.update({"band": groups[0], "identifier": groups[1], "channel": groups[1]})
            return "power", "erds", meta
        if ftype == "erds_windowed":
            meta.update({"band": groups[0], "channel": groups[1], "window": groups[2], "identifier": f"{groups[1]}_{groups[2]}"})
            return "power", "erds_windowed", meta
        if ftype == "relative_power":
            meta.update({"band": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "power", "relative", meta
        if ftype == "band_ratio":
            meta.update({"band": f"{groups[0]}/{groups[1]}", "channel": groups[2], "identifier": groups[2]})
            return "power", "ratio", meta
        if ftype in ("temporal_stat", "amplitude"):
            meta.update({"stat": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "temporal", groups[0], meta
        if ftype == "hjorth":
            meta.update({"param": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "complexity", "hjorth", meta
        if ftype == "roi_power":
            meta.update({"band": groups[0], "roi": groups[1], "identifier": groups[1]})
            return "power", "roi", meta
        if ftype in ("roi_asymmetry", "roi_laterality"):
            meta.update({"band": groups[0], "pair": groups[1], "identifier": groups[1]})
            return "roi", "asymmetry", meta
        if ftype == "ms_transition":
            meta.update({"from_state": groups[0], "to_state": groups[1], "identifier": f"{groups[0]}->{groups[1]}"})
            return "microstate", "transition", meta
        if ftype.startswith("ms_"):
            meta.update({"state": groups[0], "identifier": groups[0]})
            return "microstate", ftype.replace("ms_", ""), meta
        if ftype == "itpc":
            meta.update({"band": groups[0], "channel": groups[1], "time_bin": groups[2], "identifier": groups[1]})
            return "itpc", "itpc", meta
        if ftype == "aperiodic":
            meta.update({"param": groups[0], "channel": groups[1], "identifier": groups[1], "band": "aperiodic"})
            return "aperiodic", groups[0], meta
        if ftype == "powcorr":
            meta.update({"band": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "power", "correlation", meta
        if ftype == "conn_graph":
            meta.update({"measure": groups[0], "band": groups[1], "metric": groups[2], "identifier": f"{groups[0]}_{groups[2]}"})
            return "connectivity", "graph", meta
        if ftype == "gfp":
            meta.update({"metric": groups[0], "identifier": groups[0], "band": "global"})
            return "gfp", groups[0], meta
        if ftype == "power_segmented":
            segment, band, ident = groups[0], groups[1], groups[2]
            meta.update({"band": band, "identifier": ident, "segment": segment})
            return "power", segment, meta
        if ftype == "connectivity_segmented":
            measure, segment, band, ident = groups[0], groups[1], groups[2], groups[3]
            meta.update({"band": band, "identifier": ident, "measure": measure, "segment": segment})
            return "connectivity", segment, meta
        if ftype == "itpc_segmented":
            band, ch, segment = groups[0], groups[1], groups[2]
            meta.update({"band": band, "channel": ch, "segment": segment, "identifier": ch})
            return "itpc", segment, meta
        if ftype == "dynamics_burst_segmented":
            band, segment, metric = groups[0], groups[1], groups[2]
            meta.update({"band": band, "segment": segment, "metric": metric, "identifier": f"{band}_{segment}_{metric}"})
            return "dynamics", segment, meta

        meta.update({"identifier": groups[0] if groups else column})
        return ftype, "unknown", meta

    return None


def classify_feature(
    column: str,
    source_file_type: Optional[str] = None,
    include_subtype: bool = True,
    registry: Optional[FeatureRegistry] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify feature and extract metadata using config-driven registry."""
    if not column or not isinstance(column, str):
        meta = {
            "identifier": str(column) if column else "unknown",
            "band": "N/A",
            "source": source_file_type or "unknown",
            "subtype": "unknown",
        }
        return ("unknown", "unknown", meta) if include_subtype else ("unknown", "", meta)

    registry = registry or get_feature_registry()

    # PRIMARY: source file type mapping
    feature_type = registry.source_to_type.get(source_file_type, source_file_type)
    meta: Dict[str, Any] = {"identifier": column, "band": "N/A", "source": source_file_type or "inferred", "subtype": "unknown"}

    # FIRST PRIORITY: NamingSchema parsing
    # This covers all new features (ITPC, PAC, modern Power) generically
    parsed = NamingSchema.parse(column)
    if parsed.get("valid"):
        # Map schema group to feature_type
        # Schema groups: power, itpc, pac, connectivity, microstate, etc.
        feature_type = parsed["group"]
        subtype = parsed.get("segment", "unknown")
        
        # Build meta from parsed
        meta.update({
            "identifier": parsed.get("identifier") or parsed.get("stat") or column,
            "band": parsed.get("band", "N/A"),
            "stat": parsed.get("stat"),
            "scope": parsed.get("scope"),
            "segment": parsed.get("segment"),
            "source": source_file_type or "inferred",
        })
        
        # Specific overrides/refinements
        if feature_type == "power":
            if subtype == "baseline":
                meta["subtype"] = "baseline"
            elif subtype == "plateau":
                meta["subtype"] = "plateau" # normalized
                
        meta["subtype"] = subtype
        return (feature_type, subtype, meta) if include_subtype else (feature_type, "", meta)

    # SECONDARY: regex-based patterns (most specific legacy)
    pattern_match = _match_feature_patterns(column, registry)
    if pattern_match:
        feature_type, subtype, meta = pattern_match
        meta["source"] = source_file_type or meta.get("source", "inferred")
        meta["subtype"] = subtype
        return (feature_type, subtype, meta) if include_subtype else (feature_type, "", meta)

    # TERTIARY: classifier rules
    for rule in registry.classifiers:
        if _rule_matches(column, rule):
            feature_type = rule.label
            break

    feature_type = registry.source_to_type.get(feature_type, feature_type) or "unknown"
    subtype = _classify_subtype(column, feature_type, registry, source_file_type)
    meta.update(_parse_feature_metadata(column, feature_type))
    meta["subtype"] = subtype

    return (feature_type, subtype, meta) if include_subtype else (feature_type, "", meta)


def _parse_feature_metadata(column: str, ftype: str) -> Dict[str, Any]:
    """Parse feature metadata from column name.
    
    Attempts to extract frequency band and identifier from various naming conventions.
    """
    parts = column.split("_")
    meta = {"identifier": column, "band": "N/A"}
    
    # Common frequency bands (case-insensitive)
    freq_bands = {"delta", "theta", "alpha", "beta", "gamma", "low_gamma", "high_gamma", 
                  "low_beta", "high_beta", "mu", "spindle"}
    
    # Try to extract band from parts
    band_candidates = [p.lower() for p in parts if p.lower() in freq_bands]
    if band_candidates:
        meta["band"] = band_candidates[0]  # Take first match
    
    # Parse based on feature type
    if ftype == "power":
        # Patterns: pow_alpha_C3, power_alpha_C3, pow_alpha_roi_frontal
        if len(parts) >= 2:
            # Check if second part is a band
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                # Band might be elsewhere, identifier is everything after pow/power
                meta["identifier"] = "_".join(parts[1:])
    
    elif ftype == "connectivity":
        # Patterns: wpli_alpha_C3-Fz, plv_theta_C3-C4, conn_alpha_all
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])
    
    elif ftype == "graph":
        # Patterns: graph_alpha_geff, graph_theta_clust
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])
    
    elif ftype == "aperiodic":
        meta["band"] = "aperiodic"
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column
    
    elif ftype in ("microstate", "itpc", "pac"):
        # These might have bands or not
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])
    
    elif ftype == "spectral":
        # Patterns: iaf_C3, relative_alpha_C3, spectral_entropy_C3
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])
    
    elif ftype == "temporal":
        # Patterns: mean_C3, var_C3, rms_C3
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column
    
    elif ftype == "complexity":
        # Patterns: pe_C3, sampen_C3, hjorth_mobility_C3
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column
    
    elif ftype == "roi":
        # Patterns: roi_frontal, asymmetry_alpha, laterality_beta
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])
    
    elif ftype == "precomputed":
        # Precomputed features: gfp_*, iaf_*, pow_*, spec_*, slope_*, sef*_*, etc.
        # Many have band info embedded: pow_alpha_C3, spec_alpha_C3, iaf_cog_C3
        if len(parts) >= 2:
            # Check all parts for band info
            for i, part in enumerate(parts):
                if part.lower() in freq_bands:
                    meta["band"] = part.lower()
                    # Identifier is everything except the band
                    remaining = parts[:i] + parts[i+1:]
                    meta["identifier"] = "_".join(remaining) if remaining else column
                    break
            else:
                # No band found - use full name as identifier
                meta["identifier"] = column
    
    elif ftype == "gfp":
        # GFP features: gfp_mean, gfp_max, gfp_baseline_change
        meta["band"] = "global"
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column
    
    return meta


###################################################################
# Shared classification bindings
###################################################################
# Ensure all classification helpers use the shared registry module.
_rule_matches = feature_registry._rule_matches
_classify_subtype = feature_registry._classify_subtype
_match_feature_patterns = feature_registry._match_feature_patterns
_parse_feature_metadata = feature_registry._parse_feature_metadata
classify_feature = feature_registry.classify_feature


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
            return self._correlate_df(df, tvals, cfg, name, subject_ids=cfg.groups)

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
                self.logger.debug(f"  {name}: {result.n_features} features, {result.n_significant} sig")

        return results

    def _correlate_df(
        self,
        df: pd.DataFrame,
        targets: pd.Series,
        config: CorrelationConfig,
        feature_type: str,
        subject_ids: Optional[np.ndarray] = None,
    ) -> FeatureCorrelationResult:
        """Correlate a single feature dataframe using core function."""
        if df is None or df.empty or targets is None or len(targets) == 0:
            return FeatureCorrelationResult(feature_type, 0, 0)

        df_aligned, targets_aligned = _align_features_and_targets(
            df, targets, config.min_samples, self.logger
        )
        if df_aligned is None or targets_aligned is None:
            return FeatureCorrelationResult(feature_type, 0, 0)

        # Use core correlation loop
        classifier = lambda col, source_file_type=None, include_subtype=True: classify_feature(
            col, source_file_type=source_file_type, include_subtype=include_subtype, registry=self.registry
        )
        cov_aligned = None
        temp_aligned = None
        loso_groups = None
        perm_groups = None
        if config.control_trial_order and config.covariates_without_temp_df is not None:
            cov_aligned = config.covariates_without_temp_df.reindex(df_aligned.index)
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
        records, _ = correlate_features_loop(
            feature_df=df_aligned,
            target_values=targets_aligned,
            method=config.method,
            min_samples=config.min_samples,
            logger=None,
            identifier_type="feature",
            analysis_type=feature_type,
            feature_classifier=classifier,
            robust_method=config.robust_method,
            config=self.config,
            n_bootstrap=config.n_bootstrap,
            n_permutations=config.n_permutations,
            rng=config.rng,
            groups=perm_groups,
        )

        # Convert CorrelationRecord to dict and add feature_type
        record_dicts = []
        for col_name, rec in zip(df_aligned.columns, records):
            d = rec.to_dict()
            d["feature_type"] = feature_type
            d["p"] = d.get("p_value", np.nan)
            d["p_raw"] = d.get("p_value", np.nan)
            
            # Compute Bayes Factor if requested
            if config.compute_bayes_factor:
                if col_name in df_aligned.columns:
                    bf10, bf_interp = self._compute_bayes_factor(df_aligned[col_name], targets_aligned, config.method)
                    d["bf10"] = bf10
                    d["bf_interpretation"] = bf_interp

            # Partial correlations with covariates / temperature
            if cov_aligned is not None or temp_aligned is not None:
                try:
                    r_pc, p_pc, n_pc, r_temp, p_temp, n_temp = compute_partial_correlations(
                        df_aligned[col_name],
                        targets_aligned,
                        cov_aligned,
                        temp_aligned,
                        config.method,
                        feature_type,
                        logger=self.logger,
                        min_samples=config.min_samples,
                        config=self.config,
                    )
                    d["r_partial_cov"] = r_pc
                    d["p_partial_cov"] = p_pc
                    d["n_partial_cov"] = n_pc
                    d["r_partial_temp"] = r_temp
                    d["p_partial_temp"] = p_temp
                    d["n_partial_temp"] = n_temp
                except Exception as exc:
                    self.logger.debug(f"Partial correlation failed for {col_name}: {exc}")
            
            # Compute LOSO stability if requested
            if config.compute_loso_stability and loso_groups is not None:
                if col_name in df_aligned.columns:
                    r_mean, r_std, stability, _ = self._compute_loso_stability(
                        df_aligned[col_name].values, targets_aligned.values, loso_groups, config.method
                    )
                    d["loso_r_mean"] = r_mean
                    d["loso_r_std"] = r_std
                    d["loso_stability"] = stability

            if config.compute_reliability:
                try:
                    rel = compute_correlation_split_half_reliability(
                        df_aligned[col_name].values, targets_aligned.values, config.method, n_splits=50
                    )
                    d["reliability_split_half"] = rel
                except Exception as exc:
                    self.logger.debug(f"Reliability failed for {col_name}: {exc}")
            
            record_dicts.append(d)

        # Apply within-type FDR if requested
        if record_dicts and config.apply_fdr:
            p_vals = [r.get("p_value", np.nan) for r in record_dicts]
            valid_idx = [i for i, pv in enumerate(p_vals) if pd.notna(pv)]
            if valid_idx:
                q_vals = fdr_bh(np.array([p_vals[i] for i in valid_idx]))
                for idx, q in zip(valid_idx, q_vals):
                    record_dicts[idx]["p_fdr"] = float(q)

        n_sig = sum(1 for r in records if r.is_significant)
        return FeatureCorrelationResult(feature_type, len(df.columns), n_sig, record_dicts)
    
    def _compute_bayes_factor(
        self,
        feature_series: pd.Series,
        target_series: pd.Series,
        method: str,
    ) -> Tuple[float, str]:
        """Compute Bayes Factor for correlation."""
        from eeg_pipeline.utils.analysis.stats.correlation import compute_bayes_factor_correlation
        return compute_bayes_factor_correlation(
            feature_series.values, target_series.values, method=method
        )
    
    def _compute_loso_stability(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        subject_ids: np.ndarray,
        method: str,
    ) -> Tuple[float, float, float, List[float]]:
        """Compute LOSO correlation stability."""
        from eeg_pipeline.utils.analysis.stats.correlation import compute_loso_correlation_stability
        return compute_loso_correlation_stability(
            feature_values, target_values, subject_ids, method
        )

    def save_results(
        self,
        results: Dict[str, FeatureCorrelationResult],
        target_name: str = "rating",
        apply_fdr: bool = True,
    ) -> List[Path]:
        """Save correlation results to TSV files."""
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for name, result in results.items():
            if result.n_features == 0:
                continue
            df = result.to_dataframe()
            if df.empty:
                continue

            path = self.stats_dir / f"corr_stats_{name}_vs_{target_name}.tsv"
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
        - power_{segment}_{band}_ch_{channel}_{stat} (e.g., power_plateau_delta_ch_Fp2_logratio)
        - pow_{band}_{channel} (legacy)
        """
        from eeg_pipeline.utils.analysis.tfr import get_rois
        
        roi_defs = get_rois(self.config)
        if not roi_defs:
            self.logger.debug("No ROI definitions found in config")
            return None
        
        bands = self.config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        
        def extract_channel_from_col(col: str, band: str) -> Optional[str]:
            """Extract channel name from power column."""
            # Pattern: power_{segment}_{band}_ch_{channel}_{stat}
            match = re.search(rf"_{band}_ch_([A-Za-z0-9]+)_", col, re.IGNORECASE)
            if match:
                return match.group(1)
            # Legacy: pow_{band}_{channel}
            match = re.search(rf"pow_{band}_([A-Za-z0-9]+)$", col, re.IGNORECASE)
            if match:
                return match.group(1)
            return None
        
        records = []
        for band in bands:
            # Find all columns for this band (plateau segment preferred for pain analysis)
            band_cols = [c for c in power_df.columns 
                        if f"_{band}_ch_" in c.lower() or f"pow_{band}_" in c.lower()]
            
            # Prefer plateau columns if available
            plateau_cols = [c for c in band_cols if "plateau" in c.lower()]
            if plateau_cols:
                band_cols = plateau_cols
            
            if not band_cols:
                continue
            
            for roi_name, patterns in roi_defs.items():
                roi_cols = []
                for col in band_cols:
                    ch_name = extract_channel_from_col(col, band)
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
                
                r, p = compute_correlation(
                    roi_vals[valid].values, targets[valid].values,
                    corr_config.method == "spearman"
                )
                
                records.append({
                    "roi": roi_name,
                    "band": band,
                    "r": r,
                    "p": p,
                    "n": int(valid.sum()),
                    "method": corr_config.method,
                })
            
            overall_vals = power_df[band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            valid = overall_vals.notna() & targets.notna()
            if valid.sum() >= corr_config.min_samples:
                r, p = compute_correlation(
                    overall_vals[valid].values, targets[valid].values,
                    corr_config.method == "spearman"
                )
                records.append({
                    "roi": "overall",
                    "band": band,
                    "r": r,
                    "p": p,
                    "n": int(valid.sum()),
                    "method": corr_config.method,
                })
        
        if not records:
            self.logger.warning("No ROI correlations computed - check power column naming")
            return None
        
        df = pd.DataFrame(records)
        # Add multiplicity control (per band) and raw aliases for consistency
        df["p_raw"] = df["p"]
        if corr_config.apply_fdr and "band" in df.columns:
            for band, mask in df.groupby("band").groups.items():
                band_idx = list(mask)
                p_vals = df.loc[band_idx, "p"].to_numpy()
                df.loc[band_idx, "p_fdr"] = fdr_bh(p_vals)
        elif corr_config.apply_fdr:
            df["p_fdr"] = fdr_bh(df["p"].to_numpy())
        suffix = "rating" if "rating" in target_name.lower() else "temp"
        path = self.stats_dir / f"corr_stats_pow_roi_vs_{suffix}.tsv"
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

        all_records = []
        metadata = {"n_feature_types": len(self._feature_dfs)}

        if rating_series is not None and len(rating_series) > 0:
            rating_results = self.correlate_all(rating_series, "rating", corr_config)
            self.save_results(rating_results, "rating")

            for name, result in rating_results.items():
                metadata[f"{name}_n_features"] = result.n_features
                metadata[f"{name}_n_significant"] = result.n_significant
                all_records.extend(result.records)
            
            if "power" in self._feature_dfs:
                self.compute_roi_correlations(
                    self._feature_dfs["power"], rating_series, "rating", corr_config
                )

        min_samples_default = get_min_samples(self.config, "default")
        if temperature_series is not None and len(temperature_series.dropna()) > min_samples_default:
            temp_results = self.correlate_all(temperature_series, "temperature", corr_config)
            self.save_results(temp_results, "temperature")
            
            if "power" in self._feature_dfs:
                self.compute_roi_correlations(
                    self._feature_dfs["power"], temperature_series, "temp", corr_config
                )

        if all_records:
            combined_df = pd.DataFrame(all_records)
            if "p_value" in combined_df.columns and "p" not in combined_df.columns:
                combined_df["p"] = combined_df["p_value"]
                combined_df["p_raw"] = combined_df["p_value"]
            if corr_config.apply_fdr and "p" in combined_df.columns:
                combined_df["p_fdr"] = fdr_bh(combined_df["p"].to_numpy())
            combined_path = self.stats_dir / "corr_stats_all_features_vs_rating.tsv"
            save_correlation_results(combined_df, combined_path)

        alpha = float(get_config_value(self.config, "statistics.sig_alpha", 0.05))
        n_sig = sum(1 for r in all_records if r.get("p", 1) < alpha)
        self.logger.info(
            f"Complete: {len(all_records)} correlations, {n_sig} significant "
            f"(alpha={alpha})"
        )

        return ComputationResult(
            name="feature_correlator",
            status=ComputationStatus.SUCCESS,
            metadata=metadata,
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
    
    # Inject loaded data from context to ensure new features are included
    # (BehaviorContext is the source of truth for data loading)
    if ctx.power_df is not None: correlator._feature_dfs["power"] = ctx.power_df
    if ctx.connectivity_df is not None: correlator._feature_dfs["connectivity"] = ctx.connectivity_df
    if ctx.microstates_df is not None: correlator._feature_dfs["microstate"] = ctx.microstates_df
    if ctx.aperiodic_df is not None: correlator._feature_dfs["aperiodic"] = ctx.aperiodic_df
    if ctx.itpc_df is not None: correlator._feature_dfs["itpc"] = ctx.itpc_df
    if ctx.pac_df is not None: correlator._feature_dfs["pac"] = ctx.pac_df
    if ctx.precomputed_df is not None: correlator._feature_dfs["feature"] = ctx.precomputed_df # legacy name support
    
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
    all_records = []

    for name, df in pain_features.items():
        result = correlator._correlate_df(df, ctx.targets, corr_config, f"{name}_pain")
        all_records.extend(result.records)

    if all_records:
        df = pd.DataFrame(all_records)
        save_correlation_results(df, ctx.stats_dir / "corr_stats_pain_relevant_vs_rating.tsv")

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
