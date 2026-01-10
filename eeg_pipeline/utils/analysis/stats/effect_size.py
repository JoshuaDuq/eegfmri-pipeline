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
from eeg_pipeline.utils.analysis.stats.base import get_config_value, get_epsilon_std
from eeg_pipeline.utils.analysis.stats.correlation import fisher_z
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.validation import validate_pain_binary_values
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_condition_effects


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
    config: Optional[Any] = None,
) -> float:
    """
    Compute Cohen's d effect size.
    
    Uses pooled SD by default; set pooled=False for Cohen's d_s.
    """
    group1_clean = np.asarray(group1).ravel()
    group2_clean = np.asarray(group2).ravel()

    group1_clean = group1_clean[np.isfinite(group1_clean)]
    group2_clean = group2_clean[np.isfinite(group2_clean)]

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan

    mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
    std1, std2 = np.std(group1_clean, ddof=1), np.std(group2_clean, ddof=1)
    n1, n2 = len(group1_clean), len(group2_clean)

    if pooled:
        pooled_variance = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_variance)
    else:
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)

    epsilon = get_epsilon_std(config)
    if pooled_std < epsilon:
        return np.nan

    return float((mean1 - mean2) / pooled_std)


def hedges_g(
    group1: np.ndarray,
    group2: np.ndarray,
    config: Optional[Any] = None,
) -> float:
    """Hedges' g (bias-corrected Cohen's d)."""
    d = cohens_d(group1, group2, pooled=True, config=config)
    if not np.isfinite(d):
        return np.nan

    n1 = np.sum(np.isfinite(group1))
    n2 = np.sum(np.isfinite(group2))
    degrees_of_freedom = n1 + n2 - 2

    if degrees_of_freedom < 2:
        return d

    correction_factor = 1 - 3 / (4 * degrees_of_freedom - 1)
    return float(d * correction_factor)


def glass_delta_control_group1(
    group1: np.ndarray,
    group2: np.ndarray,
    config: Optional[Any] = None,
) -> float:
    """Glass' delta using group1 SD as control."""
    group1_clean = np.asarray(group1).ravel()
    group2_clean = np.asarray(group2).ravel()

    group1_clean = group1_clean[np.isfinite(group1_clean)]
    group2_clean = group2_clean[np.isfinite(group2_clean)]

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan

    mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
    control_std = np.std(group1_clean, ddof=1)

    epsilon = get_epsilon_std(config)
    if control_std < epsilon:
        return np.nan

    return float((mean1 - mean2) / control_std)


def glass_delta_control_group2(
    group1: np.ndarray,
    group2: np.ndarray,
    config: Optional[Any] = None,
) -> float:
    """Glass' delta using group2 SD as control."""
    group1_clean = np.asarray(group1).ravel()
    group2_clean = np.asarray(group2).ravel()

    group1_clean = group1_clean[np.isfinite(group1_clean)]
    group2_clean = group2_clean[np.isfinite(group2_clean)]

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan

    mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
    control_std = np.std(group2_clean, ddof=1)

    epsilon = get_epsilon_std(config)
    if control_std < epsilon:
        return np.nan

    return float((mean1 - mean2) / control_std)


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
    if n1 < 4 or n2 < 4:
        return np.nan, np.nan

    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan, np.nan

    z1 = fisher_z(r1, config=config)
    z2 = fisher_z(r2, config=config)

    standard_error = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_statistic = (z1 - z2) / standard_error

    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic)))

    return float(z_statistic), float(p_value)


def cohens_q(r1: float, r2: float, config: Optional[Any] = None) -> float:
    """Cohen's q for difference between correlations."""
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan

    z1 = fisher_z(r1, config=config)
    z2 = fisher_z(r2, config=config)

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
    config: Optional[Any] = None,
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
            results["cohens_d"] = cohens_d(group1_data, group2_data, config=config)
        if "hedges_g" in effect_size_metrics:
            results["hedges_g"] = hedges_g(group1_data, group2_data, config=config)

    return results


def compute_cohens_d_with_bootstrap_ci(
    group_a_data: np.ndarray,
    group_b_data: np.ndarray,
    random_seed: int,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """Compute Cohen's d with bootstrap confidence intervals.

    Args:
        group_a_data: Data for group A (1D array)
        group_b_data: Data for group B (1D array)
        random_seed: Random seed for reproducibility
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (cohens_d, ci_low, ci_high)
    """
    n_group_a = len(group_a_data)
    n_group_b = len(group_b_data)

    mean_diff = group_a_data.mean() - group_b_data.mean()
    pooled_std = np.sqrt(
        ((n_group_a - 1) * group_a_data.std() ** 2 + (n_group_b - 1) * group_b_data.std() ** 2)
        / (n_group_a + n_group_b - 2)
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    rng = np.random.default_rng(random_seed)
    boot_indices_a = rng.integers(0, n_group_a, size=(n_bootstrap, n_group_a))
    boot_indices_b = rng.integers(0, n_group_b, size=(n_bootstrap, n_group_b))

    boot_samples_a = group_a_data[boot_indices_a]
    boot_samples_b = group_b_data[boot_indices_b]

    boot_means_a = boot_samples_a.mean(axis=1)
    boot_means_b = boot_samples_b.mean(axis=1)
    boot_vars_a = boot_samples_a.var(axis=1, ddof=1)
    boot_vars_b = boot_samples_b.var(axis=1, ddof=1)

    boot_pooled_std = np.sqrt(
        ((n_group_a - 1) * boot_vars_a + (n_group_b - 1) * boot_vars_b)
        / (n_group_a + n_group_b - 2)
    )
    boot_ds = np.where(
        boot_pooled_std > 0,
        (boot_means_a - boot_means_b) / boot_pooled_std,
        np.nan,
    )
    boot_ds_valid = boot_ds[np.isfinite(boot_ds)]

    ci_low = (
        np.percentile(boot_ds_valid, 2.5) if len(boot_ds_valid) > 0 else np.nan
    )
    ci_high = (
        np.percentile(boot_ds_valid, 97.5) if len(boot_ds_valid) > 0 else np.nan
    )

    return float(cohens_d), float(ci_low), float(ci_high)


###################################################################
# Condition Effects (Pain vs Non-Pain)
###################################################################


def _get_condition_column(
    events_df: pd.DataFrame,
    config: Any,
) -> Optional[str]:
    """Get condition column name from config or default pain column."""
    compare_col = str(
        get_config_value(config, "behavior_analysis.condition.compare_column", "") or ""
    ).strip()
    
    if compare_col and compare_col in events_df.columns:
        return compare_col
    
    return get_pain_column_from_config(config, events_df)


def _create_masks_from_compare_values(
    condition_series: pd.Series,
    value1: Any,
    value2: Any,
    logger: logging.Logger,
) -> Tuple[pd.Series, pd.Series]:
    """Create boolean masks for two condition values."""
    if condition_series.dtype == object:
        condition_normalized = condition_series.astype(str).str.strip().str.lower()
        value1_normalized = str(value1).strip().lower()
        value2_normalized = str(value2).strip().lower()
        
        mask1 = condition_normalized == value1_normalized
        mask2 = condition_normalized == value2_normalized
        return mask1, mask2
    
    try:
        condition_numeric = pd.to_numeric(condition_series, errors="coerce")
        is_numeric_value1 = str(value1).replace('.', '').replace('-', '').isdigit()
        is_numeric_value2 = str(value2).replace('.', '').replace('-', '').isdigit()
        
        value1_numeric = float(value1) if is_numeric_value1 else value1
        value2_numeric = float(value2) if is_numeric_value2 else value2
        
        mask1 = condition_numeric == value1_numeric
        mask2 = condition_numeric == value2_numeric
        return mask1, mask2
    except (ValueError, TypeError):
        condition_string = condition_series.astype(str).str.strip()
        mask1 = condition_string == str(value1).strip()
        mask2 = condition_string == str(value2).strip()
        return mask1, mask2


def _split_by_pain_binary(
    condition_series: pd.Series,
    column_name: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split by standard pain binary coding (1=pain, 0=nonpain)."""
    if condition_series.dtype == object:
        condition_normalized = (
            condition_series.astype(str)
            .str.strip()
            .str.lower()
        )
        condition_series = condition_normalized

    condition_numeric = pd.to_numeric(condition_series, errors="coerce")
    try:
        pain_values, _n_bad = validate_pain_binary_values(
            condition_numeric, column_name=column_name, logger=logger
        )
    except Exception:
        return np.array([]), np.array([]), 0, 0

    pain_mask = pain_values == 1
    nonpain_mask = pain_values == 0

    n_pain = int(pain_mask.sum())
    n_nonpain = int(nonpain_mask.sum())

    logger.info(f"Condition split: {n_pain} pain, {n_nonpain} non-pain trials")

    return pain_mask.to_numpy(), nonpain_mask.to_numpy(), n_pain, n_nonpain


def split_by_condition(
    events_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split trials into two conditions based on a column and values.
    
    Supports user-configurable condition column and values via:
    - config.event_columns.pain_binary: column name (or list of candidates)
    - config.behavior_analysis.condition.compare_column: explicit events column override (optional)
    - config.behavior_analysis.condition.compare_values: values to compare [val1, val2]
    
    If compare_values is not specified, defaults to [1, 0] for backward compatibility.
    Returns (group1_mask, group2_mask, n_group1, n_group2).
    """
    condition_column = _get_condition_column(events_df, config)

    if condition_column is None or condition_column not in events_df.columns:
        logger.error("Condition column not found in events")
        return np.array([]), np.array([]), 0, 0

    condition_series = events_df[condition_column]
    compare_values = get_config_value(
        config, "behavior_analysis.condition.compare_values", None
    )
    
    if compare_values and len(compare_values) >= 2:
        value1, value2 = compare_values[0], compare_values[1]
        logger.info(
            f"Using user-specified condition values: {value1} vs {value2} "
            f"(column: {condition_column})"
        )
        
        mask1, mask2 = _create_masks_from_compare_values(
            condition_series, value1, value2, logger
        )
        
        n_group1 = int(mask1.sum())
        n_group2 = int(mask2.sum())
        
        logger.info(
            f"Condition split: {n_group1} condition={value1}, "
            f"{n_group2} condition={value2} trials"
        )
        
        return mask1.to_numpy(), mask2.to_numpy(), n_group1, n_group2
    
    return _split_by_pain_binary(condition_series, condition_column, logger)


def _compute_p_primary_column(
    df: pd.DataFrame,
    p_primary_mode: str,
) -> pd.Series:
    """Compute p_primary column based on mode and available p-values."""
    if "p_raw" not in df.columns:
        df["p_raw"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")
    
    use_permutation = p_primary_mode in {
        "perm",
        "permutation",
        "perm_if_available",
        "permutation_if_available",
    }
    
    if use_permutation and "p_perm" in df.columns:
        p_permutation = pd.to_numeric(df["p_perm"], errors="coerce")
        p_raw = pd.to_numeric(df["p_raw"], errors="coerce")
        return p_permutation.where(p_permutation.notna(), p_raw)
    
    return pd.to_numeric(df["p_raw"], errors="coerce")


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
    import warnings
    
    n_jobs_actual = get_n_jobs(config, n_jobs)

    if logger:
        logger.debug(
            f"Computing condition effects for {len(features_df.columns)} features "
            f"(n_jobs={n_jobs_actual})"
        )

    perm_enabled = bool(
        get_config_value(
            config, "behavior_analysis.condition.permutation.enabled", False
        )
    )
    n_perm = int(
        get_config_value(
            config,
            "behavior_analysis.condition.permutation.n_permutations",
            get_config_value(
                config, "behavior_analysis.statistics.n_permutations", 0
            ),
        )
        or 0
    )
    p_primary_mode = str(
        get_config_value(
            config, "behavior_analysis.condition.p_primary_mode", "asymptotic"
        )
    ).strip().lower()
    base_seed = int(
        get_config_value(config, "behavior_analysis.statistics.base_seed", 42)
    )

    feature_columns = list(features_df.columns)
    
    # Suppress numpy RuntimeWarnings (empty slices, low variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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

    if "p_primary" not in df.columns:
        df["p_primary"] = _compute_p_primary_column(df, p_primary_mode)

    p_primary_values = pd.to_numeric(df["p_primary"], errors="coerce").values
    df["q_value"] = fdr_bh(p_primary_values, alpha=fdr_alpha, config=config)
    df["significant_fdr"] = df["q_value"] < fdr_alpha

    df = df.sort_values("hedges_g", key=abs, ascending=False)

    if logger:
        n_significant = df["significant_fdr"].sum()
        n_large_effect = (df["hedges_g"].abs() >= 0.8).sum()
        logger.info(
            f"Condition effects: {n_significant}/{len(df)} FDR significant, "
            f"{n_large_effect} large effects"
        )

    return df
