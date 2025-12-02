"""Core structures and utilities for behavioral correlation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eeg_pipeline.utils.validation import validate_epochs

# Constants (defaults, overridden by config)
MIN_SAMPLES_CHANNEL = 10
MIN_SAMPLES_ROI = 20
MIN_SAMPLES_DEFAULT = 5
MIN_SAMPLES_EDGE = 30
MIN_SAMPLES_TEMPORAL = 15
MIN_TRIALS_PER_CONDITION = 15
MIN_EPOCHS_FOR_TFR = 20
DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_PERMUTATIONS = 100
EPSILON_CORR = 1e-10
MAX_CORRELATION = 0.9999


def get_min_samples(config: Any, sample_type: str = "default") -> int:
    """Get minimum samples threshold from config."""
    defaults = {"channel": MIN_SAMPLES_CHANNEL, "roi": MIN_SAMPLES_ROI,
                "default": MIN_SAMPLES_DEFAULT, "edge": MIN_SAMPLES_EDGE,
                "temporal": MIN_SAMPLES_TEMPORAL}
    if config is None:
        return defaults.get(sample_type, MIN_SAMPLES_DEFAULT)
    return int(config.get(f"behavior_analysis.min_samples.{sample_type}",
                         defaults.get(sample_type, MIN_SAMPLES_DEFAULT)))


def get_min_trials(config: Any, trial_type: str = "per_condition") -> int:
    """Get minimum trials threshold from config."""
    defaults = {"per_condition": MIN_TRIALS_PER_CONDITION, "for_tfr": MIN_EPOCHS_FOR_TFR}
    if config is None:
        return defaults.get(trial_type, MIN_TRIALS_PER_CONDITION)
    return int(config.get(f"behavior_analysis.min_trials.{trial_type}",
                         defaults.get(trial_type, MIN_TRIALS_PER_CONDITION)))


# =============================================================================
# Result Structures
# =============================================================================


class ComputationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class CorrelationRecord:
    """Standard record for a single correlation result."""
    identifier: str
    band: str
    correlation: float
    p_value: float
    n_valid: int
    method: str
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_perm: float = np.nan
    q_value: float = np.nan
    r_partial: float = np.nan
    p_partial: float = np.nan
    n_partial: int = 0
    p_partial_perm: float = np.nan
    r_partial_temp: float = np.nan
    p_partial_temp: float = np.nan
    n_partial_temp: int = 0
    p_partial_temp_perm: float = np.nan
    identifier_type: str = "channel"
    analysis_type: str = "power"
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stats(cls, identifier: str, band: str, stats: Any, n_valid: int,
                   method: str, identifier_type: str = "channel",
                   analysis_type: str = "power", **extra) -> "CorrelationRecord":
        """Create from CorrelationStats object."""
        def safe_float(v): return float(v) if np.isfinite(v) else np.nan
        return cls(
            identifier=identifier, band=band,
            correlation=safe_float(stats.correlation),
            p_value=safe_float(stats.p_value), n_valid=n_valid, method=method,
            ci_low=safe_float(stats.ci_low), ci_high=safe_float(stats.ci_high),
            p_perm=safe_float(stats.p_perm),
            r_partial=safe_float(stats.r_partial),
            p_partial=safe_float(stats.p_partial),
            n_partial=int(getattr(stats, 'n_partial', 0)),
            p_partial_perm=safe_float(stats.p_partial_perm),
            r_partial_temp=safe_float(stats.r_partial_temp),
            p_partial_temp=safe_float(stats.p_partial_temp),
            n_partial_temp=int(getattr(stats, 'n_partial_temp', 0)),
            p_partial_temp_perm=safe_float(stats.p_partial_temp_perm),
            identifier_type=identifier_type, analysis_type=analysis_type,
            extra_fields=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame."""
        d = {
            self.identifier_type: self.identifier, "band": self.band,
            "r": self.correlation, "p": self.p_value, "n": self.n_valid,
            "method": self.method, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_perm": self.p_perm, "q": self.q_value, "analysis": self.analysis_type,
        }
        if np.isfinite(self.r_partial): d["r_partial"] = self.r_partial
        if np.isfinite(self.p_partial): d["p_partial"] = self.p_partial
        if self.n_partial > 0: d["n_partial"] = self.n_partial
        if np.isfinite(self.r_partial_temp): d["r_partial_given_temp"] = self.r_partial_temp
        if np.isfinite(self.p_partial_temp): d["p_partial_given_temp"] = self.p_partial_temp
        d.update(self.extra_fields)
        return d

    @property
    def is_significant(self) -> bool:
        return self.p_value < DEFAULT_ALPHA


@dataclass
class ComputationResult:
    """Result of a computation."""
    name: str
    status: ComputationStatus
    records: List[CorrelationRecord] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def n_significant(self) -> int:
        return sum(1 for r in self.records if r.is_significant)

    def to_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe
        return pd.DataFrame([r.to_dict() for r in self.records]) if self.records else pd.DataFrame()


# =============================================================================
# Context
# =============================================================================


@dataclass
class BehaviorContext:
    """Shared context for behavior analysis."""
    subject: str
    task: str
    config: Any
    logger: Any
    deriv_root: Path
    stats_dir: Path
    use_spearman: bool = True
    bootstrap: int = 0
    n_perm: int = 100
    rng: Optional[np.random.Generator] = None
    partial_covars: Optional[List[str]] = None
    
    epochs: Any = None
    epochs_info: Any = None
    aligned_events: Optional[pd.DataFrame] = None
    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    microstates_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    precomputed_df: Optional[pd.DataFrame] = None
    targets: Optional[pd.Series] = None
    temperature: Optional[pd.Series] = None
    temperature_column: Optional[str] = None
    covariates_df: Optional[pd.DataFrame] = None
    covariates_without_temp_df: Optional[pd.DataFrame] = None
    results: Dict[str, ComputationResult] = field(default_factory=dict)
    _data_loaded: bool = False

    @property
    def method(self) -> str:
        return "spearman" if self.use_spearman else "pearson"

    @property
    def n_trials(self) -> int:
        if self.targets is not None: return len(self.targets)
        if self.power_df is not None: return len(self.power_df)
        return 0

    @property
    def power_bands(self) -> List[str]:
        from eeg_pipeline.utils.config.loader import get_frequency_band_names
        return get_frequency_band_names(self.config)

    @property
    def min_samples_channel(self) -> int:
        return get_min_samples(self.config, "channel")

    @property
    def min_samples_roi(self) -> int:
        return get_min_samples(self.config, "roi")

    @property
    def has_temperature(self) -> bool:
        return self.temperature is not None and len(self.temperature) > 0

    @property
    def has_covariates(self) -> bool:
        return self.covariates_df is not None and not self.covariates_df.empty

    def load_data(self) -> bool:
        """Load all features and targets. Returns True if successful."""
        if self._data_loaded:
            return self.targets is not None

        from eeg_pipeline.utils.data.loading import (
            _load_features_and_targets, load_epochs_for_analysis,
            extract_temperature_data, build_covariate_matrix, build_covariates_without_temp,
        )
        from eeg_pipeline.utils.io.general import deriv_features_path, read_tsv

        self.logger.info("Loading data...")
        try:
            self.epochs, self.aligned_events = load_epochs_for_analysis(
                self.subject, self.task, align="strict", preload=False,
                deriv_root=self.deriv_root, bids_root=self.config.bids_root, config=self.config,
            )
            if self.epochs is None:
                self._data_loaded = True
                return False

            validation = validate_epochs(self.epochs, self.config, logger=self.logger)
            if validation.critical:
                self.logger.error(f"Critical: {validation.critical}")
                self._data_loaded = True
                return False

            self.epochs_info = self.epochs.info
            _, self.power_df, self.connectivity_df, self.targets, _ = _load_features_and_targets(
                self.subject, self.task, self.deriv_root, self.config, epochs=self.epochs,
            )

            if self.aligned_events is not None:
                self.temperature, self.temperature_column = extract_temperature_data(
                    self.aligned_events, self.config
                )
                self.covariates_df = build_covariate_matrix(
                    self.aligned_events, self.partial_covars, self.config
                )
                self.covariates_without_temp_df = build_covariates_without_temp(
                    self.covariates_df, self.temperature_column
                )

            features_dir = deriv_features_path(self.deriv_root, self.subject)
            for fname, attr in [("features_microstates.tsv", "microstates_df"),
                               ("features_precomputed.tsv", "precomputed_df")]:
                path = features_dir / fname
                if path.exists():
                    setattr(self, attr, read_tsv(path))

            for fname in ["features_aperiodic.tsv", "features_eeg_direct.tsv"]:
                path = features_dir / fname
                if path.exists():
                    df = read_tsv(path)
                    aper_cols = [c for c in df.columns if str(c).startswith("aper_")]
                    if aper_cols:
                        self.aperiodic_df = df[aper_cols]
                        break

            self._data_loaded = True
            n_feat = sum(len(getattr(self, a).columns) if getattr(self, a) is not None else 0
                        for a in ["power_df", "connectivity_df", "microstates_df", "precomputed_df"])
            self.logger.info(f"Loaded: {self.n_trials} trials, {n_feat} features")
            return self.targets is not None

        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            self._data_loaded = True
            return False

    def add_result(self, name: str, result: ComputationResult) -> None:
        self.results[name] = result

    def get_combined_dataframe(self) -> pd.DataFrame:
        dfs = []
        for name, result in self.results.items():
            df = result.to_dataframe()
            if not df.empty:
                df["computation"] = name
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# =============================================================================
# Core Functions
# =============================================================================


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str = "spearman",
                     min_samples: int = MIN_SAMPLES_DEFAULT, 
                     robust_method: str = None) -> Tuple[float, float, int]:
    """Compute correlation with validation. Returns (r, p, n_valid).
    
    If robust_method is specified, uses robust correlation from utils.
    Options: "percentage_bend", "winsorized", "shepherd"
    """
    from scipy import stats
    x, y = np.asarray(x), np.asarray(y)
    if x.size != y.size:
        return np.nan, np.nan, 0

    mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(mask.sum())
    if n_valid < min_samples:
        return np.nan, np.nan, n_valid

    x_c, y_c = x[mask], y[mask]
    if np.std(x_c) == 0 or np.std(y_c) == 0:
        return np.nan, np.nan, n_valid

    try:
        if robust_method:
            from eeg_pipeline.utils.analysis.stats import compute_robust_correlation
            r, p = compute_robust_correlation(x_c, y_c, method=robust_method)
        elif method == "spearman":
            r, p = stats.spearmanr(x_c, y_c, nan_policy="omit")
        else:
            r, p = stats.pearsonr(x_c, y_c)
        return (float(r) if np.isfinite(r) else np.nan,
                float(p) if np.isfinite(p) else np.nan, n_valid)
    except Exception:
        return np.nan, np.nan, n_valid


def build_correlation_record(identifier: str, band: str, r: float, p: float, n: int,
                              method: str = "spearman", *, ci_low: float = np.nan,
                              ci_high: float = np.nan, p_perm: float = np.nan,
                              r_partial: float = np.nan, p_partial: float = np.nan,
                              n_partial: int = 0, p_partial_perm: float = np.nan,
                              r_partial_temp: float = np.nan, p_partial_temp: float = np.nan,
                              n_partial_temp: int = 0, p_partial_temp_perm: float = np.nan,
                              identifier_type: str = "channel", analysis_type: str = "power",
                              **extra) -> CorrelationRecord:
    """Build standardized correlation record."""
    def sf(v): return float(v) if np.isfinite(v) else np.nan
    return CorrelationRecord(
        identifier=identifier, band=band, correlation=sf(r), p_value=sf(p),
        n_valid=int(n), method=method, ci_low=sf(ci_low), ci_high=sf(ci_high),
        p_perm=sf(p_perm), r_partial=sf(r_partial), p_partial=sf(p_partial),
        n_partial=int(n_partial), p_partial_perm=sf(p_partial_perm),
        r_partial_temp=sf(r_partial_temp), p_partial_temp=sf(p_partial_temp),
        n_partial_temp=int(n_partial_temp), p_partial_temp_perm=sf(p_partial_temp_perm),
        identifier_type=identifier_type, analysis_type=analysis_type, extra_fields=extra,
    )


def correlate_features_loop(
    feature_df: pd.DataFrame,
    target_values: Union[pd.Series, np.ndarray],
    method: str = "spearman",
    min_samples: int = MIN_SAMPLES_DEFAULT,
    logger: Optional[Any] = None,
    condition_mask: Optional[np.ndarray] = None,
    identifier_type: str = "feature",
    analysis_type: str = "unknown",
    feature_classifier: Optional[Any] = None,
    robust_method: Optional[str] = None,
) -> Tuple[List[CorrelationRecord], pd.DataFrame]:
    """Correlate all features with target values.
    
    Parameters
    ----------
    robust_method : str, optional
        If specified, uses robust correlation: "percentage_bend", "winsorized", "shepherd"
    """
    if feature_df.empty:
        return [], pd.DataFrame()

    target_arr = target_values.values if isinstance(target_values, pd.Series) else np.asarray(target_values)
    if condition_mask is not None:
        # Convert boolean mask to integer indices for iloc
        if hasattr(condition_mask, 'dtype') and condition_mask.dtype == bool:
            idx = np.where(condition_mask)[0]
        else:
            idx = condition_mask
        feature_df = feature_df.iloc[idx]
        target_arr = target_arr[condition_mask]  # numpy handles boolean masks correctly

    n_f, n_t = len(feature_df), len(target_arr)
    if n_f != n_t:
        n_use = min(n_f, n_t)
        feature_df, target_arr = feature_df.iloc[:n_use], target_arr[:n_use]

    records = []
    for col in feature_df.columns:
        vals = pd.to_numeric(feature_df[col], errors="coerce").to_numpy()
        if feature_classifier:
            ft, meta = feature_classifier(col)
            ident = meta.get("identifier", col)
            band = meta.get("band", "N/A")
        else:
            ft, ident, band = analysis_type, col, "N/A"

        r, p, n = safe_correlation(vals, target_arr, method, min_samples, robust_method=robust_method)
        if np.isfinite(r):
            records.append(build_correlation_record(
                ident, band, r, p, n, method, identifier_type=identifier_type, analysis_type=ft
            ))

    if logger:
        logger.info(f"  {len(records)} features, {sum(1 for r in records if r.is_significant)} sig")
    return records, pd.DataFrame([r.to_dict() for r in records]) if records else pd.DataFrame()


def iterate_feature_columns(feature_df: pd.DataFrame,
                             col_prefix: Optional[str] = None) -> Tuple[List[str], pd.DataFrame]:
    """Get filtered feature columns."""
    if feature_df is None or feature_df.empty:
        return [], pd.DataFrame()
    if col_prefix:
        cols = [c for c in feature_df.columns if str(c).startswith(col_prefix)]
        return (cols, feature_df[cols]) if cols else ([], pd.DataFrame())
    return list(feature_df.columns), feature_df


# fisher_z_test moved to eeg_pipeline.utils.analysis.stats


def build_output_filename(analysis_type: str, level: str, target: str,
                          method: Optional[str] = None, condition: Optional[str] = None,
                          config: Optional[Any] = None) -> str:
    """Build standardized output filename."""
    cfg = config.get("behavior_analysis.output_files", {}) if config else {}
    prefix = cfg.get("prefix", "corr_stats")
    sep = cfg.get("separator", "_")
    ext = cfg.get("extension", ".tsv")
    parts = [prefix, analysis_type, level, "vs", target]
    if condition:
        parts.insert(-2, condition)
    if cfg.get("include_method_suffix", True) and method:
        parts.append(method)
    return sep.join(parts) + ext


def save_correlation_results(records: Union[List[CorrelationRecord], pd.DataFrame],
                             output_path: Path, apply_fdr: bool = True,
                             config: Optional[Any] = None, logger: Optional[Any] = None,
                             use_permutation_p: bool = True) -> None:
    """Save correlation results to TSV with optional FDR."""
    from eeg_pipeline.utils.io.general import write_tsv
    from eeg_pipeline.utils.analysis.stats import fdr_bh, get_fdr_alpha_from_config

    if isinstance(records, list):
        if not records:
            return
        df = pd.DataFrame([r.to_dict() for r in records])
    else:
        df = records.copy()

    if df.empty:
        return

    if apply_fdr and "p" in df.columns:
        p_col = "p_perm" if use_permutation_p and "p_perm" in df.columns else "p"
        p_vals = df[p_col].to_numpy(dtype=float)
        valid = np.isfinite(p_vals)
        if valid.any():
            alpha = get_fdr_alpha_from_config(config) if config else 0.05
            q = np.full(len(df), np.nan)
            q[valid] = fdr_bh(p_vals[valid], alpha=alpha, config=config)
            df["q"] = q

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(df, output_path)
    if logger:
        n_sig = int((df["q"] < 0.05).sum()) if "q" in df.columns else 0
        logger.info(f"Saved {len(df)} to {output_path.name} ({n_sig} sig)")


# =============================================================================
# Pipeline
# =============================================================================


@dataclass
class AnalysisPipelineConfig:
    """Pipeline configuration."""
    run_power_roi: bool = True
    run_connectivity: bool = True
    run_precomputed: bool = True
    run_condition_split: bool = True
    run_temporal: bool = True
    run_tf_correlations: bool = True
    run_cluster_tests: bool = True
    run_topomaps: bool = True
    use_spearman: bool = True
    apply_global_fdr: bool = True
    export_summary: bool = True
    export_top_n: int = 20

    @classmethod
    def from_config(cls, config: Any) -> "AnalysisPipelineConfig":
        cfg = config.get("behavior_analysis", {}) if config else {}
        stats = cfg.get("statistics", {})
        return cls(
            use_spearman=stats.get("correlation_method", "spearman") == "spearman",
            export_top_n=int(cfg.get("predictors", {}).get("top_n", 20)),
        )


def run_behavior_analysis_pipeline(
    ctx: BehaviorContext,
    cfg: Optional[AnalysisPipelineConfig] = None,
) -> Dict[str, ComputationResult]:
    """Run complete behavioral correlation analysis pipeline."""
    if cfg is None:
        cfg = AnalysisPipelineConfig.from_config(ctx.config)

    log = ctx.logger
    results: Dict[str, ComputationResult] = {}

    log.info(f"Starting behavioral analysis for sub-{ctx.subject}")

    if not ctx._data_loaded and not ctx.load_data():
        log.error("Failed to load data")
        return results

    # Run unified feature correlator first (comprehensive, efficient)
    try:
        from eeg_pipeline.analysis.behavior.feature_correlator import run_unified_feature_correlations
        unified_result = run_unified_feature_correlations(ctx)
        results["unified_features"] = unified_result
    except Exception as e:
        log.warning(f"Unified feature correlator skipped: {e}")
    
    # Analysis steps (specialized analyses with advanced statistics)
    steps = [
        ("power_roi", cfg.run_power_roi, "power_roi", "compute_power_roi_stats_from_context"),
        ("connectivity", cfg.run_connectivity, "connectivity", "correlate_connectivity_roi_from_context"),
        ("precomputed", cfg.run_precomputed, "precomputed_correlations", "compute_precomputed_correlations"),
        ("condition", cfg.run_condition_split, "condition_correlations", "compute_condition_correlations"),
        ("temporal", cfg.run_tf_correlations, "temporal", "compute_time_frequency_from_context"),
        ("cluster_tests", cfg.run_cluster_tests, "cluster_tests", "run_cluster_test_from_context"),
        ("topomaps", cfg.run_topomaps, "topomaps", "correlate_power_topomaps_from_context"),
    ]

    for name, enabled, module, func in steps:
        if not enabled:
            continue
        try:
            mod = __import__(f"eeg_pipeline.analysis.behavior.{module}", fromlist=[func])
            result = getattr(mod, func)(ctx)
            if result is None:
                result = ComputationResult(name=name, status=ComputationStatus.SUCCESS)
            results[name] = result
        except Exception as e:
            log.error(f"{name} failed: {e}")
            results[name] = ComputationResult(name=name, status=ComputationStatus.FAILED, error=str(e))

    # Global FDR
    if cfg.apply_global_fdr:
        try:
            from eeg_pipeline.analysis.behavior.fdr_correction import apply_global_fdr
            apply_global_fdr(ctx.subject)
        except Exception as e:
            log.error(f"Global FDR failed: {e}")

    # Exports
    if cfg.export_summary:
        try:
            from eeg_pipeline.analysis.behavior.exports import (
                export_all_significant_predictors, export_analysis_summary, export_top_predictors
            )
            export_all_significant_predictors(ctx.subject)
            export_analysis_summary(ctx.subject)
            export_top_predictors(ctx.subject, n_top=cfg.export_top_n)
        except Exception as e:
            log.warning(f"Export failed: {e}")

    n_ok = sum(1 for r in results.values() if r.status == ComputationStatus.SUCCESS)
    n_fail = sum(1 for r in results.values() if r.status == ComputationStatus.FAILED)
    log.info(f"Pipeline complete: {n_ok} ok, {n_fail} failed")
    return results
