"""
Unified Correlation Analysis
=============================

Single module for all EEG-behavior correlations with statistical controls.

Features:
- Temperature-controlled partial correlations (PRIMARY)
- Trial-order control for autocorrelation
- Baseline-corrected (change) features
- Pain sensitivity index
- Split-half reliability
- Effect size interpretation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats import (
    compute_correlation,
    partial_corr_xy_given_Z,
    fdr_bh,
)
from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
from eeg_pipeline.analysis.behavior.parallel import (
    parallel_correlate_features,
    get_n_jobs,
)


###################################################################
# Effect Size Benchmarks
###################################################################

EFFECT_SIZE_BENCHMARKS = {
    "negligible": (0.0, 0.2),
    "small": (0.2, 0.5),
    "medium": (0.5, 0.8),
    "large": (0.8, float("inf")),
}

CORRELATION_BENCHMARKS = {
    "negligible": (0.0, 0.1),
    "small": (0.1, 0.3),
    "medium": (0.3, 0.5),
    "large": (0.5, float("inf")),
}


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    for label, (lo, hi) in EFFECT_SIZE_BENCHMARKS.items():
        if lo <= d_abs < hi:
            return label
    return "unknown"


def interpret_correlation(r: float) -> str:
    """Interpret correlation magnitude."""
    r_abs = abs(r)
    for label, (lo, hi) in CORRELATION_BENCHMARKS.items():
        if lo <= r_abs < hi:
            return label
    return "unknown"


###################################################################
# Pain Sensitivity Index
###################################################################

def compute_pain_sensitivity_index(
    ratings: pd.Series,
    temperatures: pd.Series,
) -> pd.Series:
    """Compute pain sensitivity as residual from temperature-rating regression.
    
    This captures individual trial sensitivity independent of temperature.
    """
    valid = ratings.notna() & temperatures.notna()
    psi = pd.Series(np.nan, index=ratings.index)
    
    if valid.sum() < 5:
        return psi
    
    r_valid = ratings[valid].values
    t_valid = temperatures[valid].values
    
    X = np.column_stack([np.ones(len(t_valid)), t_valid])
    try:
        beta = np.linalg.lstsq(X, r_valid, rcond=None)[0]
        predicted = X @ beta
        psi.loc[valid] = r_valid - predicted
    except np.linalg.LinAlgError:
        psi.loc[valid] = r_valid
    
    return psi


###################################################################
# Baseline-Corrected Features
###################################################################

def compute_change_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute plateau - baseline change for matching feature pairs."""
    baseline_cols = [c for c in features_df.columns if "_baseline_" in c]
    
    # Build all change columns at once to avoid DataFrame fragmentation
    change_data = {}
    for bl_col in baseline_cols:
        pl_col = bl_col.replace("_baseline_", "_plateau_")
        if pl_col in features_df.columns:
            bl_vals = features_df[bl_col].values
            pl_vals = features_df[pl_col].values
            
            # Skip if either column has non-1D data
            if bl_vals.ndim != 1 or pl_vals.ndim != 1:
                continue
            
            change_col = bl_col.replace("_baseline_", "_change_")
            change_data[change_col] = pl_vals - bl_vals
    
    if not change_data:
        return pd.DataFrame(index=features_df.index)
    
    return pd.DataFrame(change_data, index=features_df.index)


###################################################################
# Split-Half Reliability
###################################################################

from eeg_pipeline.utils.analysis.stats.reliability import (
    compute_correlation_split_half_reliability as compute_split_half_reliability,
)


###################################################################
# Core Correlation Function
###################################################################

@dataclass
class CorrelationResult:
    """Single feature correlation result."""
    feature: str
    band: str
    r_raw: float
    p_raw: float
    n: int
    r_partial_temp: float
    p_partial_temp: float
    r_partial_order: float
    p_partial_order: float
    r_partial_full: float
    p_partial_full: float
    effect_interpretation: str
    reliability: float
    is_change_score: bool
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "band": self.band,
            "r_raw": self.r_raw,
            "p_raw": self.p_raw,
            "n": self.n,
            "r_partial_temp": self.r_partial_temp,
            "p_partial_temp": self.p_partial_temp,
            "r_partial_order": self.r_partial_order,
            "p_partial_order": self.p_partial_order,
            "r_partial_full": self.r_partial_full,
            "p_partial_full": self.p_partial_full,
            "effect_interpretation": self.effect_interpretation,
            "reliability": self.reliability,
            "is_change_score": self.is_change_score,
            "method": self.method,
            "r_primary": self.r_partial_temp if np.isfinite(self.r_partial_temp) else self.r_raw,
            "p_primary": self.p_partial_temp if np.isfinite(self.p_partial_temp) else self.p_raw,
        }


def _correlate_single_feature(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    temperature: Optional[np.ndarray],
    trial_order: Optional[np.ndarray],
    method: str,
    min_samples: int,
) -> Tuple[float, float, float, float, float, float, float, float, int]:
    """Compute all correlations for a single feature."""
    valid = np.isfinite(feature_values) & np.isfinite(target_values)
    if temperature is not None:
        valid &= np.isfinite(temperature)
    if trial_order is not None:
        valid &= np.isfinite(trial_order)
    
    n_valid = valid.sum()
    if n_valid < min_samples:
        return (np.nan,) * 8 + (0,)
    
    x, y = feature_values[valid], target_values[valid]
    
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return (np.nan,) * 8 + (n_valid,)
    
    # Raw correlation (use utils)
    r_raw, p_raw = compute_correlation(x, y, method)
    
    # Partial correlations
    r_pt, p_pt, r_po, p_po, r_pf, p_pf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    if temperature is not None:
        temp_df = pd.DataFrame({"temp": temperature[valid]})
        r_pt, p_pt, _ = partial_corr_xy_given_Z(pd.Series(x), pd.Series(y), temp_df, method)
    
    if trial_order is not None:
        order_df = pd.DataFrame({"order": trial_order[valid]})
        r_po, p_po, _ = partial_corr_xy_given_Z(pd.Series(x), pd.Series(y), order_df, method)
    
    if temperature is not None and trial_order is not None:
        full_df = pd.DataFrame({"temp": temperature[valid], "order": trial_order[valid]})
        r_pf, p_pf, _ = partial_corr_xy_given_Z(pd.Series(x), pd.Series(y), full_df, method)
    elif temperature is not None:
        r_pf, p_pf = r_pt, p_pt
    elif trial_order is not None:
        r_pf, p_pf = r_po, p_po
    
    return r_raw, p_raw, r_pt, p_pt, r_po, p_po, r_pf, p_pf, n_valid


def run_correlations(
    features_df: pd.DataFrame,
    targets: pd.Series,
    temperature: Optional[pd.Series] = None,
    trial_order: Optional[pd.Series] = None,
    method: str = "spearman",
    min_samples: int = 10,
    compute_reliability: bool = True,
    include_change_scores: bool = True,
    logger: Optional[logging.Logger] = None,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Run correlations with all statistical controls.
    
    This is the PRIMARY correlation function.
    
    Args:
        features_df: Feature DataFrame (trials x features)
        targets: Pain ratings
        temperature: Temperature per trial (for partial correlations)
        trial_order: Trial index (for autocorrelation control)
        method: "spearman" or "pearson"
        min_samples: Minimum valid samples
        compute_reliability: Whether to compute split-half reliability
        include_change_scores: Whether to add baseline-corrected features
        logger: Logger instance
        rng: Random number generator
    
    Returns:
        DataFrame with correlation results
    """
    if features_df is None or features_df.empty:
        return pd.DataFrame()
    
    n_jobs_actual = get_n_jobs(config, n_jobs)
    
    if logger:
        logger.info(f"Correlating {len(features_df.columns)} features with {method} (n_jobs={n_jobs_actual})")
    
    # Add change scores
    all_features = features_df.copy()
    if include_change_scores:
        change_df = compute_change_features(features_df)
        if not change_df.empty:
            all_features = pd.concat([all_features, change_df], axis=1)
            if logger:
                logger.info(f"Added {len(change_df.columns)} change features")
    
    # Prepare arrays
    target_arr = targets.values
    temp_arr = temperature.values if temperature is not None else None
    order_arr = trial_order.values if trial_order is not None else None
    
    if rng is None:
        rng = np.random.default_rng(42)
    rng_seed = int(rng.integers(0, 2**31))
    
    # Parallel correlation computation
    feature_columns = list(all_features.columns)
    result_dicts = parallel_correlate_features(
        feature_columns=feature_columns,
        feature_df=all_features,
        target_arr=target_arr,
        temp_arr=temp_arr,
        order_arr=order_arr,
        method=method,
        min_samples=min_samples,
        compute_reliability=compute_reliability,
        rng_seed=rng_seed,
        n_jobs=n_jobs_actual,
    )
    
    if logger:
        n_sig = sum(1 for r in result_dicts if r.get("p_raw", 1) < 0.05)
        n_sig_ctrl = sum(1 for r in result_dicts if np.isfinite(r.get("p_partial_temp", np.nan)) and r.get("p_partial_temp", 1) < 0.05)
        logger.info(f"  {len(result_dicts)} features: {n_sig} sig (raw), {n_sig_ctrl} sig (controlled)")
    
    return pd.DataFrame(result_dicts)


def run_pain_sensitivity_correlations(
    features_df: pd.DataFrame,
    ratings: pd.Series,
    temperatures: pd.Series,
    method: str = "spearman",
    min_samples: int = 10,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Correlate features with pain sensitivity index."""
    psi = compute_pain_sensitivity_index(ratings, temperatures)
    
    if psi.isna().all():
        if logger:
            logger.warning("Could not compute pain sensitivity index")
        return pd.DataFrame()
    
    records = []
    for col in features_df.columns:
        vals = pd.to_numeric(features_df[col], errors="coerce").values
        r, p, n = safe_correlation(vals, psi.values, method, min_samples)
        
        if np.isfinite(r):
            records.append({
                "feature": col,
                "r_psi": float(r),
                "p_psi": float(p),
                "n": n,
                "effect_interpretation": interpret_correlation(r),
            })
    
    if logger:
        n_sig = sum(1 for r in records if r["p_psi"] < 0.05)
        logger.info(f"Pain sensitivity: {len(records)} features, {n_sig} significant")
    
    return pd.DataFrame(records)
