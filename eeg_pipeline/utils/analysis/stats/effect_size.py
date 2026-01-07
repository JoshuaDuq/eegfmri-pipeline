"""
Effect Size Statistics
======================

Cohen's d, correlation difference effects, and Fisher z-tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.data.columns import get_pain_column_from_config
from eeg_pipeline.utils.analysis.stats.base import get_config_value
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.validation import validate_pain_binary_values
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_condition_effects


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
) -> float:
    """
    Compute Cohen's d effect size.
    
    Uses pooled SD by default; set pooled=False for Cohen's d_s.
    """
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()

    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    n1, n2 = len(g1), len(g2)

    if pooled:
        pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        sd = np.sqrt(pooled_var)
    else:
        sd = np.sqrt((s1**2 + s2**2) / 2)

    if sd < 1e-12:
        return np.nan

    return float((m1 - m2) / sd)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Hedges' g (bias-corrected Cohen's d)."""
    d = cohens_d(group1, group2, pooled=True)
    if not np.isfinite(d):
        return np.nan

    n1 = np.sum(np.isfinite(group1))
    n2 = np.sum(np.isfinite(group2))
    df = n1 + n2 - 2

    if df < 2:
        return d

    # Approximate correction factor
    correction = 1 - 3 / (4 * df - 1)
    return float(d * correction)


def glass_delta(group1: np.ndarray, group2: np.ndarray, control: int = 2) -> float:
    """Glass' delta using control group SD."""
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()

    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    m1, m2 = np.mean(g1), np.mean(g2)

    if control == 1:
        sd = np.std(g1, ddof=1)
    else:
        sd = np.std(g2, ddof=1)

    if sd < 1e-12:
        return np.nan

    return float((m1 - m2) / sd)


def fisher_z_test(
    r1: float,
    r2: float,
    n1: int,
    n2: int,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Fisher z-test for difference between two correlations.
    
    Returns (z_statistic, p_value).
    """
    from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values
    if n1 < 4 or n2 < 4:
        return np.nan, np.nan

    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan, np.nan

    clip_min, clip_max = get_fisher_z_clip_values(config)
    r1 = np.clip(r1, clip_min, clip_max)
    r2 = np.clip(r2, clip_min, clip_max)

    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se

    p = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return float(z_stat), float(p)


def cohens_q(r1: float, r2: float, config: Optional[Any] = None) -> float:
    """Cohen's q for difference between correlations."""
    from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan

    clip_min, clip_max = get_fisher_z_clip_values(config)
    r1 = np.clip(r1, clip_min, clip_max)
    r2 = np.clip(r2, clip_min, clip_max)

    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    return float(z1 - z2)


def correlation_difference_effect(
    r1: float,
    r2: float,
    n1: int,
    n2: int,
    config: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Comprehensive effect statistics for correlation difference.
    
    Returns dict with z_stat, p_value, cohens_q, r_diff.
    """
    z_stat, p_val = fisher_z_test(r1, r2, n1, n2, config)
    q = cohens_q(r1, r2, config)

    return {
        "r_diff": float(r1 - r2) if np.isfinite(r1) and np.isfinite(r2) else np.nan,
        "z_stat": z_stat,
        "p_value": p_val,
        "cohens_q": q,
    }


def r_to_d(r: float) -> float:
    """Convert correlation to Cohen's d approximation."""
    if not np.isfinite(r) or np.abs(r) >= 1:
        return np.nan
    return 2 * r / np.sqrt(1 - r**2)


def d_to_r(d: float) -> float:
    """Convert Cohen's d to correlation approximation."""
    if not np.isfinite(d):
        return np.nan
    return d / np.sqrt(d**2 + 4)


def compute_effect_sizes(
    r_val: float,
    p_val: float,
    n_samples: int,
    group1_data: Optional[np.ndarray] = None,
    group2_data: Optional[np.ndarray] = None,
    effect_size_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multiple effect size metrics.
    
    Supported metrics: r, r_squared, d_from_r, cohens_d, hedges_g.
    """
    if effect_size_metrics is None:
        effect_size_metrics = ["r", "r_squared", "d_from_r"]

    results = {}

    if "r" in effect_size_metrics:
        results["r"] = float(r_val) if np.isfinite(r_val) else np.nan

    if "r_squared" in effect_size_metrics:
        results["r_squared"] = float(r_val**2) if np.isfinite(r_val) else np.nan

    if "d_from_r" in effect_size_metrics:
        results["d_from_r"] = r_to_d(r_val)

    if group1_data is not None and group2_data is not None:
        if "cohens_d" in effect_size_metrics:
            results["cohens_d"] = cohens_d(group1_data, group2_data)
        if "hedges_g" in effect_size_metrics:
            results["hedges_g"] = hedges_g(group1_data, group2_data)

    return results


###################################################################
# Condition Effects (Pain vs Non-Pain)
###################################################################


def split_by_condition(
    events_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split trials into two conditions based on a column and values.
    
    Supports user-configurable condition column and values via:
    - config.event_columns.pain_binary: column name (or list of candidates)
    - config.behavior_analysis.condition.compare_values: values to compare [val1, val2]
    
    If compare_values is not specified, defaults to [1, 0] for backward compatibility.
    Returns (group1_mask, group2_mask, n_group1, n_group2).
    """
    pain_col = get_pain_column_from_config(config, events_df)

    if pain_col is None or pain_col not in events_df.columns:
        logger.error("Condition column not found in events")
        return np.array([]), np.array([]), 0, 0

    raw = events_df[pain_col]
    
    # Get user-configured compare values
    compare_values = get_config_value(config, "behavior_analysis.condition.compare_values", None)
    
    if compare_values and len(compare_values) >= 2:
        # User specified explicit values to compare
        val1, val2 = compare_values[0], compare_values[1]
        logger.info(f"Using user-specified condition values: {val1} vs {val2} (column: {pain_col})")
        
        # Convert to appropriate types for comparison
        condition_series = raw.copy()
        if condition_series.dtype == object:
            condition_series = condition_series.astype(str).str.strip().str.lower()
            val1_str = str(val1).strip().lower()
            val2_str = str(val2).strip().lower()
            
            group1_mask = condition_series == val1_str
            group2_mask = condition_series == val2_str
        else:
            # Try numeric comparison
            try:
                condition_series = pd.to_numeric(condition_series, errors="coerce")
                val1_num = float(val1) if str(val1).replace('.', '').replace('-', '').isdigit() else val1
                val2_num = float(val2) if str(val2).replace('.', '').replace('-', '').isdigit() else val2
                
                group1_mask = condition_series == val1_num
                group2_mask = condition_series == val2_num
            except (ValueError, TypeError):
                # Fallback to string comparison
                condition_series = raw.astype(str).str.strip()
                group1_mask = condition_series == str(val1).strip()
                group2_mask = condition_series == str(val2).strip()
        
        n_group1 = int(group1_mask.sum())
        n_group2 = int(group2_mask.sum())
        
        logger.info(f"Condition split: {n_group1} condition={val1}, {n_group2} condition={val2} trials")
        
        return group1_mask.to_numpy(), group2_mask.to_numpy(), n_group1, n_group2
    
    # Default behavior: use standard pain binary coding (1=pain, 0=nonpain)
    pain_series = raw.copy()
    if pain_series.dtype == object:
        mapped = (
            pain_series.astype(str)
            .str.strip()
            .str.lower()
        )
        pain_series = mapped

    pain_series = pd.to_numeric(pain_series, errors="coerce")
    try:
        pain_vals, _n_bad = validate_pain_binary_values(pain_series, column_name=str(pain_col), logger=logger)
    except Exception:
        return np.array([]), np.array([]), 0, 0

    pain_mask = pain_vals == 1
    nonpain_mask = pain_vals == 0

    n_pain = int(pain_mask.sum())
    n_nonpain = int(nonpain_mask.sum())

    logger.info(f"Condition split: {n_pain} pain, {n_nonpain} non-pain trials")

    return pain_mask, nonpain_mask, n_pain, n_nonpain


def compute_condition_effects(
    features_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int = 5,
    fdr_alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = -1,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute effect sizes for pain vs non-pain comparison."""
    n_jobs_actual = get_n_jobs(config, n_jobs)

    if logger:
        logger.debug(f"Computing condition effects for {len(features_df.columns)} features (n_jobs={n_jobs_actual})")

    perm_enabled = bool(get_config_value(config, "behavior_analysis.condition.permutation.enabled", False))
    n_perm = int(
        get_config_value(
            config,
            "behavior_analysis.condition.permutation.n_permutations",
            get_config_value(config, "behavior_analysis.statistics.n_permutations", 0),
        )
        or 0
    )
    p_primary_mode = str(get_config_value(config, "behavior_analysis.condition.p_primary_mode", "asymptotic")).strip().lower()
    base_seed = int(get_config_value(config, "behavior_analysis.statistics.base_seed", 42))

    feature_columns = list(features_df.columns)
    records = parallel_condition_effects(
        feature_columns=feature_columns,
        features_df=features_df,
        pain_mask=pain_mask,
        nonpain_mask=nonpain_mask,
        min_samples=min_samples,
        n_jobs=n_jobs_actual,
        groups=groups,
        n_perm=n_perm if perm_enabled else 0,
        base_seed=base_seed,
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "p_raw" not in df.columns:
        df["p_raw"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")
    if "p_primary" not in df.columns:
        use_perm = p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}
        if use_perm and "p_perm" in df.columns:
            pprim = pd.to_numeric(df["p_perm"], errors="coerce")
            fallback = pd.to_numeric(df["p_raw"], errors="coerce")
            df["p_primary"] = pprim.where(pprim.notna(), fallback)
        else:
            df["p_primary"] = pd.to_numeric(df["p_raw"], errors="coerce")

    df["q_value"] = fdr_bh(pd.to_numeric(df["p_primary"], errors="coerce").values, alpha=fdr_alpha, config=config)
    df["significant_fdr"] = df["q_value"] < fdr_alpha

    df = df.sort_values("hedges_g", key=abs, ascending=False)

    if logger:
        n_sig = df["significant_fdr"].sum()
        n_large = (df["hedges_g"].abs() >= 0.8).sum()
        logger.info(f"Condition effects: {n_sig}/{len(df)} FDR significant, {n_large} large effects")

    return df
