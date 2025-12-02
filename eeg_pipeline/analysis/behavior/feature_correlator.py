"""
Unified Feature-Behavior Correlator
====================================

Single entry point for correlating ALL EEG features with behavioral measures.
Uses core.correlate_features_loop for all correlations - no duplicate logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.behavior.core import (
    BehaviorContext,
    ComputationResult,
    ComputationStatus,
    CorrelationRecord,
    correlate_features_loop,
    save_correlation_results,
    MIN_SAMPLES_DEFAULT,
)
from eeg_pipeline.utils.io.general import deriv_features_path, read_tsv, write_tsv


# =============================================================================
# Feature File Registry
# =============================================================================

FEATURE_FILES = {
    "power": "features_eeg_direct.tsv",
    "power_plateau": "features_eeg_plateau.tsv",
    "precomputed": "features_precomputed.tsv",
    "connectivity": "features_connectivity.tsv",
    "microstates": "features_microstates.tsv",
    "aperiodic": "features_aperiodic.tsv",
    "itpc": "features_itpc.tsv",
    "pac": "features_pac_trials.tsv",
    "source": "features_source.tsv",
    "dynamics": "features_dynamics.tsv",
}

# Map source file types to standardized feature type names for visualization
SOURCE_TO_FEATURE_TYPE = {
    "power": "power",
    "power_plateau": "power",  # Group plateau with power
    "precomputed": "precomputed",  # Keep separate as it may contain mixed types
    "connectivity": "connectivity",
    "microstates": "microstate",  # Standardize to singular
    "aperiodic": "aperiodic",
    "itpc": "itpc",
    "pac": "pac",
    "source": "source",
    "dynamics": "dynamics",
}

# Feature type hierarchy with subtypes
FEATURE_TYPE_HIERARCHY = {
    "power": {
        "subtypes": ["direct", "plateau"],
        "structure": "band × channel × time",
    },
    "connectivity": {
        "subtypes": ["aec", "wpli", "plv", "pli", "imcoh", "coh", "icoh"],
        "structure": "measure × band × channel_pair",
    },
    "microstate": {
        "subtypes": ["coverage", "duration", "occurrence", "transition", "gev", "entropy", "valid"],
        "structure": "metric × state",
    },
    "precomputed": {
        "subtypes": ["gfp", "roi", "temporal", "spectral", "complexity", "other"],
        "structure": "mixed",
    },
    "itpc": {
        "subtypes": ["itpc"],
        "structure": "band × channel × time_bin",
    },
    "pac": {
        "subtypes": ["pac"],
        "structure": "roi × phase_freq × amp_freq × time",
    },
}

# Feature classifiers - ordered by specificity (most specific first)
# This ensures that more specific patterns are matched before general ones
FEATURE_CLASSIFIERS = [
    # Most specific patterns first
    ("gfp", lambda c: c.startswith("gfp_") or "global_field_power" in c.lower()),
    ("erds", lambda c: c.startswith("erds_")),
    ("itpc", lambda c: c.startswith("itpc_") or ("itpc" in c.lower() and not "itpc_" in c)),
    ("pac", lambda c: c.startswith("pac_") or (("phase" in c.lower() and "amplitude" in c.lower()) and "pac_" not in c)),
    ("microstate", lambda c: c.startswith("ms_") or ("microstate" in c.lower() and not c.startswith("ms_"))),
    ("aperiodic", lambda c: c.startswith(("aper_", "powcorr_", "aperiodic_"))),
    ("source", lambda c: c.startswith("src_") or (c.startswith("source_") and not c.startswith("src_"))),
    
    # Connectivity patterns (before graph, as graph is more general)
    ("connectivity", lambda c: any(c.startswith(p) for p in ["wpli_", "plv_", "aec_", "imcoh_", "pli_", "conn_", "coh_", "icoh_", "corr_", "sync_", "coherence_", "connectivity_"])),
    
    # Graph metrics (check for graph-specific suffixes)
    ("graph", lambda c: any(m in c for m in ["_geff", "_clust", "_pc", "_smallworld", "_modularity", "_betweenness", "_pathlength", "_efficiency", "_charpath"])),
    
    # Power features (common, but check after more specific ones)
    ("power", lambda c: (c.startswith("pow_") or c.startswith("power_")) and not any(c.startswith(p) for p in ["powcorr_", "power_plateau_", "power_relative_"])),
    
    # Spectral features (check for spectral-specific prefixes)
    ("spectral", lambda c: any(c.startswith(p) for p in ["iaf_", "relative_", "ratio_", "spectral_", "peak_", "bandwidth_", "spectral_edge_", "spectral_entropy_", "freq_", "dominant_freq_"])),
    
    # Complexity features
    ("complexity", lambda c: any(c.startswith(p) for p in ["pe_", "sampen_", "hjorth_", "lzc_", "hurst_", "dfa_", "permutation_entropy_", "sample_entropy_", "approximate_entropy_", "shannon_", "renyi_"])),
    
    # Temporal features (more general, check after specific ones)
    ("temporal", lambda c: any(c.startswith(p) for p in ["mean_", "var_", "std_", "skew_", "kurt_", "rms_", "p2p_", "median_", "iqr_", "line_length_", "nle_", "zero_cross_", "slope_", "variance_", "stddev_"])),
    
    # ROI features (check last as it's very general)
    ("roi", lambda c: (c.startswith("roi_") or "_roi_" in c) and not any(c.startswith(p) for p in ["pow_", "power_", "conn_", "connectivity_"])),
    
    # Asymmetry/laterality (very general, check last)
    ("roi", lambda c: ("asymmetry_" in c or "laterality_" in c) and not any(c.startswith(p) for p in ["pow_", "power_", "conn_", "connectivity_"])),
]


@dataclass
class CorrelationConfig:
    """Configuration for feature-behavior correlations."""
    method: str = "spearman"
    min_samples: int = MIN_SAMPLES_DEFAULT
    apply_fdr: bool = True
    n_bootstrap: int = 0
    n_permutations: int = 0
    filter_threshold: float = 0.0
    # New options for robust statistics
    compute_bayes_factor: bool = False
    robust_method: Optional[str] = None  # "percentage_bend", "winsorized", "shepherd"
    compute_loso_stability: bool = False

    @classmethod
    def from_context(cls, ctx: BehaviorContext) -> "CorrelationConfig":
        cfg = ctx.config or {}
        behavior_cfg = cfg.get("behavior_analysis", {})
        return cls(
            method=ctx.method,
            min_samples=ctx.min_samples_channel,
            n_bootstrap=ctx.bootstrap,
            n_permutations=ctx.n_perm,
            compute_bayes_factor=behavior_cfg.get("compute_bayes_factors", False),
            robust_method=behavior_cfg.get("robust_correlation", None),
            compute_loso_stability=behavior_cfg.get("loso_stability", False),
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


def _classify_subtype(column: str, feature_type: str, source_file: Optional[str] = None) -> str:
    """Classify feature subtype within a feature type.
    
    Parameters
    ----------
    column : str
        Feature column name
    feature_type : str
        Primary feature type (e.g., "power", "connectivity")
    source_file : str, optional
        Source file type (e.g., "power_plateau", "connectivity")
    
    Returns
    -------
    str
        Subtype (e.g., "direct", "aec", "coverage")
    """
    col_lower = column.lower()
    parts = column.split("_")
    
    if feature_type == "power":
        # Check source file first
        if source_file and ("plateau" in source_file or "_plateau" in source_file):
            return "plateau"
        # Check column name
        if "plateau" in col_lower:
            return "plateau"
        return "direct"
    
    elif feature_type == "connectivity":
        # Extract measure from column name (first part)
        if len(parts) > 0:
            measure = parts[0].lower()
            if measure in ["aec", "wpli", "plv", "pli", "imcoh", "coh", "icoh", "corr", "sync"]:
                return measure
        return "unknown"
    
    elif feature_type == "microstate":
        # Extract metric from column name (second part after "ms")
        if len(parts) >= 2 and parts[0].lower() == "ms":
            metric = parts[1].lower()
            if metric in ["coverage", "duration", "occurrence", "transition", "gev", "entropy", "valid"]:
                return metric
        return "unknown"
    
    elif feature_type == "precomputed":
        # Classify based on prefix
        if len(parts) > 0:
            prefix = parts[0].lower()
            if prefix == "gfp":
                return "gfp"
            elif prefix == "roi":
                return "roi"
            elif prefix in ["mean", "var", "std", "skew", "kurt", "median", "iqr", "rms", "p2p"]:
                return "temporal"
            elif prefix in ["iaf", "peak", "bandwidth", "spectral", "relative", "ratio"]:
                return "spectral"
            elif prefix in ["pe", "sampen", "hjorth", "lzc", "hurst", "dfa", "entropy"]:
                return "complexity"
        return "other"
    
    elif feature_type == "itpc":
        return "itpc"
    
    elif feature_type == "pac":
        return "pac"
    
    return "unknown"


def classify_feature(column: str, source_file_type: Optional[str] = None, include_subtype: bool = True) -> Tuple[str, str, Dict[str, Any]]:
    """Classify feature and extract metadata with subtype information.
    
    Uses a two-tier approach:
    1. PRIMARY: If source_file_type is provided, use it (most reliable)
    2. SECONDARY: Fall back to column name pattern matching
    
    This ensures features are classified correctly based on their source file,
    which is more reliable than inferring from column names.
    
    Parameters
    ----------
    column : str
        Feature column name or identifier (e.g., "pow_alpha_C3", "C3_alpha_power", "wpli_alpha_C3-Fz")
    source_file_type : str, optional
        Source file type from FEATURE_FILES (e.g., "power", "connectivity", "microstates")
        If provided, this takes precedence over column name matching.
    include_subtype : bool
        If True, returns (type, subtype, metadata). If False, returns (type, "", metadata) for backward compatibility.
    
    Returns
    -------
    Tuple[str, str, Dict[str, Any]]
        (feature_type, subtype, metadata_dict) where metadata contains "identifier", "band", "source", and "subtype"
    """
    if not column or not isinstance(column, str):
        meta = {"identifier": str(column) if column else "unknown", "band": "N/A", "source": source_file_type or "unknown", "subtype": "unknown"}
        return ("unknown", "unknown", meta) if include_subtype else ("unknown", "", meta)
    
    # PRIMARY: Use source file type if available (most reliable)
    if source_file_type:
        standardized_type = SOURCE_TO_FEATURE_TYPE.get(source_file_type, source_file_type)
        subtype = _classify_subtype(column, standardized_type, source_file_type)
        meta = _parse_feature_metadata(column, standardized_type)
        meta["source"] = source_file_type
        meta["subtype"] = subtype
        return (standardized_type, subtype, meta) if include_subtype else (standardized_type, "", meta)
    
    # SECONDARY: Fall back to column name pattern matching
    col_lower = column.lower()
    col_original = column
    
    # Try each classifier in order (most specific first)
    for ftype, classifier in FEATURE_CLASSIFIERS:
        if classifier(col_lower):
            subtype = _classify_subtype(col_original, ftype)
            meta = _parse_feature_metadata(col_original, ftype)
            meta["source"] = "inferred"
            meta["subtype"] = subtype
            return (ftype, subtype, meta) if include_subtype else (ftype, "", meta)
    
    # If no match, check for common patterns that might indicate feature type
    # This helps catch edge cases
    if any(x in col_lower for x in ["_delta", "_theta", "_alpha", "_beta", "_gamma"]):
        # Likely a frequency band feature - try to infer type
        if "pow" in col_lower or "power" in col_lower:
            subtype = _classify_subtype(col_original, "power")
            meta = _parse_feature_metadata(col_original, "power")
            meta["source"] = "inferred"
            meta["subtype"] = subtype
            return ("power", subtype, meta) if include_subtype else ("power", "", meta)
        elif "conn" in col_lower or "connectivity" in col_lower:
            subtype = _classify_subtype(col_original, "connectivity")
            meta = _parse_feature_metadata(col_original, "connectivity")
            meta["source"] = "inferred"
            meta["subtype"] = subtype
            return ("connectivity", subtype, meta) if include_subtype else ("connectivity", "", meta)
    
    meta = {"identifier": col_original, "band": "N/A", "source": "unknown", "subtype": "unknown"}
    return ("unknown", "unknown", meta) if include_subtype else ("unknown", "", meta)


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
    
    return meta


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

    def load_all_features(self) -> Dict[str, int]:
        """Load all available feature files. Returns feature counts."""
        if self._loaded:
            return {k: len(v.columns) for k, v in self._feature_dfs.items()}

        counts = {}
        for name, filename in FEATURE_FILES.items():
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
    ) -> Dict[str, FeatureCorrelationResult]:
        """Correlate all loaded features with behavioral target."""
        if not self._loaded:
            self.load_all_features()

        if corr_config is None:
            corr_config = CorrelationConfig()

        self.logger.info(f"Correlating features with {target_name}...")
        results = {}

        for name, df in self._feature_dfs.items():
            result = self._correlate_df(df, targets, corr_config, name)
            results[name] = result
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

        # Use core correlation loop
        records, _ = correlate_features_loop(
            feature_df=df,
            target_values=targets,
            method=config.method,
            min_samples=config.min_samples,
            logger=None,  # Suppress per-column logging
            identifier_type="feature",
            analysis_type=feature_type,
            feature_classifier=classify_feature,
            robust_method=config.robust_method,
        )

        # Convert CorrelationRecord to dict and add feature_type
        record_dicts = []
        for rec in records:
            d = rec.to_dict()
            d["feature_type"] = feature_type
            
            # Compute Bayes Factor if requested
            if config.compute_bayes_factor:
                col = rec.identifier
                if col in df.columns:
                    bf10, bf_interp = self._compute_bayes_factor(df[col], targets, config.method)
                    d["bf10"] = bf10
                    d["bf_interpretation"] = bf_interp
            
            # Compute LOSO stability if requested
            if config.compute_loso_stability and subject_ids is not None:
                col = rec.identifier
                if col in df.columns:
                    r_mean, r_std, stability, _ = self._compute_loso_stability(
                        df[col].values, targets.values, subject_ids, config.method
                    )
                    d["loso_r_mean"] = r_mean
                    d["loso_r_std"] = r_std
                    d["loso_stability"] = stability
            
            record_dicts.append(d)

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
            save_correlation_results(df, path, apply_fdr=apply_fdr, config=self.config, logger=self.logger)
            saved_files.append(path)

        return saved_files

    def run_complete_analysis(
        self,
        rating_series: pd.Series,
        temperature_series: Optional[pd.Series] = None,
        corr_config: Optional[CorrelationConfig] = None,
    ) -> ComputationResult:
        """Run complete feature-behavior correlation analysis."""
        try:
            self.load_all_features()

            if not self._feature_dfs:
                return ComputationResult(
                    name="feature_correlator",
                    status=ComputationStatus.SKIPPED,
                    metadata={"reason": "No features loaded"},
                )

            all_records = []
            metadata = {"n_feature_types": len(self._feature_dfs)}

            # Rating correlations
            if rating_series is not None and len(rating_series) > 0:
                rating_results = self.correlate_all(rating_series, "rating", corr_config)
                self.save_results(rating_results, "rating")

                for name, result in rating_results.items():
                    metadata[f"{name}_n_features"] = result.n_features
                    metadata[f"{name}_n_significant"] = result.n_significant
                    all_records.extend(result.records)

            # Temperature correlations
            if temperature_series is not None and len(temperature_series.dropna()) > MIN_SAMPLES_DEFAULT:
                temp_results = self.correlate_all(temperature_series, "temperature", corr_config)
                self.save_results(temp_results, "temperature")

            # Combined output
            if all_records:
                combined_df = pd.DataFrame(all_records)
                combined_path = self.stats_dir / "corr_stats_all_features_vs_rating.tsv"
                save_correlation_results(combined_df, combined_path, apply_fdr=True,
                                        config=self.config, logger=self.logger)

            n_sig = sum(1 for r in all_records if r.get("p", 1) < 0.05)
            self.logger.info(f"Complete: {len(all_records)} correlations, {n_sig} significant")

            return ComputationResult(
                name="feature_correlator",
                status=ComputationStatus.SUCCESS,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Feature correlation failed: {e}")
            return ComputationResult(
                name="feature_correlator",
                status=ComputationStatus.FAILED,
                error=str(e),
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
    return correlator.run_complete_analysis(
        rating_series=ctx.targets,
        temperature_series=ctx.temperature,
        corr_config=CorrelationConfig.from_context(ctx),
    )


def correlate_pain_relevant_features(ctx: BehaviorContext) -> ComputationResult:
    """Correlate features most relevant to pain processing."""
    pain_patterns = [
        r"pow_alpha_(C3|C4|CP3|CP4|Cz)",
        r"pow_theta_(Fz|FCz|F3|F4)",
        r"pow_(loGamma|hiGamma)_(C3|C4|CP3|CP4)",
        r"aper_(slope|offset)_",
        r"wpli_(alpha|theta)_",
        r"ms_(coverage|duration)_",
    ]

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
        save_correlation_results(df, ctx.stats_dir / "corr_stats_pain_relevant_vs_rating.tsv",
                                apply_fdr=True, config=ctx.config, logger=ctx.logger)

    n_sig = sum(1 for r in all_records if r.get("p", 1) < 0.05)
    return ComputationResult(name="pain_relevant", status=ComputationStatus.SUCCESS,
                            metadata={"n_features": len(all_records), "n_significant": n_sig})


def generate_feature_coverage_report(ctx: BehaviorContext) -> Dict[str, Any]:
    """Generate feature coverage report."""
    features_dir = deriv_features_path(ctx.deriv_root, ctx.subject)
    
    report = {
        "subject": ctx.subject,
        "feature_files": {},
        "feature_types": {},
        "total_features": 0,
        "available_files": [],
        "missing_files": [],
    }

    for name, filename in FEATURE_FILES.items():
        path = features_dir / filename
        if path.exists():
            df = read_tsv(path)
            if df is not None and not df.empty:
                report["available_files"].append(name)
                report["feature_files"][name] = {"n_features": len(df.columns), "n_epochs": len(df)}
                report["total_features"] += len(df.columns)
                for col in df.columns:
                    ftype, _, _ = classify_feature(col, include_subtype=True)
                    report["feature_types"][ftype] = report["feature_types"].get(ftype, 0) + 1
        else:
            report["missing_files"].append(name)

    report["coverage_pct"] = len(report["available_files"]) / len(FEATURE_FILES) * 100
    ctx.logger.info(f"Coverage: {len(report['available_files'])}/{len(FEATURE_FILES)} files")
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
    if ctx.temperature is not None and len(ctx.temperature.dropna()) > MIN_SAMPLES_DEFAULT:
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
