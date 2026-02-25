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

from eeg_pipeline.utils.data.columns import get_binary_outcome_column_from_config
from eeg_pipeline.utils.analysis.stats.base import get_config_value, get_epsilon_std
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
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


def compute_cohens_d_with_bootstrap_ci(
    group_a_data: np.ndarray,
    group_b_data: np.ndarray,
    random_seed: int,
    n_bootstrap: int = 1000,
    config: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Compute Cohen's d with bootstrap confidence intervals.

    Args:
        group_a_data: Data for group A (1D array)
        group_b_data: Data for group B (1D array)
        random_seed: Random seed for reproducibility
        n_bootstrap: Number of bootstrap samples
        config: Optional configuration object

    Returns:
        Tuple of (cohens_d, ci_low, ci_high)
    """
    group_a_clean = np.asarray(group_a_data).ravel()
    group_b_clean = np.asarray(group_b_data).ravel()

    group_a_clean = group_a_clean[np.isfinite(group_a_clean)]
    group_b_clean = group_b_clean[np.isfinite(group_b_clean)]

    if len(group_a_clean) < 2 or len(group_b_clean) < 2:
        return np.nan, np.nan, np.nan

    n_group_a = len(group_a_clean)
    n_group_b = len(group_b_clean)

    mean_diff = np.mean(group_a_clean) - np.mean(group_b_clean)
    pooled_std = np.sqrt(
        ((n_group_a - 1) * np.std(group_a_clean, ddof=1) ** 2
         + (n_group_b - 1) * np.std(group_b_clean, ddof=1) ** 2)
        / (n_group_a + n_group_b - 2)
    )

    epsilon = get_epsilon_std(config)
    if pooled_std < epsilon:
        return np.nan, np.nan, np.nan

    cohens_d = mean_diff / pooled_std

    rng = np.random.default_rng(random_seed)
    boot_indices_a = rng.integers(0, n_group_a, size=(n_bootstrap, n_group_a))
    boot_indices_b = rng.integers(0, n_group_b, size=(n_bootstrap, n_group_b))

    boot_samples_a = group_a_clean[boot_indices_a]
    boot_samples_b = group_b_clean[boot_indices_b]

    boot_means_a = np.mean(boot_samples_a, axis=1)
    boot_means_b = np.mean(boot_samples_b, axis=1)
    boot_vars_a = np.var(boot_samples_a, axis=1, ddof=1)
    boot_vars_b = np.var(boot_samples_b, axis=1, ddof=1)

    boot_pooled_std = np.sqrt(
        ((n_group_a - 1) * boot_vars_a + (n_group_b - 1) * boot_vars_b)
        / (n_group_a + n_group_b - 2)
    )
    boot_ds = np.where(
        boot_pooled_std > epsilon,
        (boot_means_a - boot_means_b) / boot_pooled_std,
        np.nan,
    )
    boot_ds_valid = boot_ds[np.isfinite(boot_ds)]

    if len(boot_ds_valid) == 0:
        return float(cohens_d), np.nan, np.nan

    ci_low = float(np.percentile(boot_ds_valid, 2.5))
    ci_high = float(np.percentile(boot_ds_valid, 97.5))

    return float(cohens_d), ci_low, ci_high


_NUMERIC_TOLERANCE = 1e-12


def compute_batch_condition_effects(
    feature_columns: List[str],
    features_df: pd.DataFrame,
    cond_a_mask: np.ndarray,
    cond_b_mask: np.ndarray,
    min_samples: int = 5,
    n_perm: int = 0,
    base_seed: int = 42,
    groups: Optional[np.ndarray] = None,
    scheme: str = "shuffle",
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Vectorized batch computation of condition effects.

    Computes t-tests and effect sizes for all features using numpy broadcasting.
    Much faster than per-feature loops for large feature sets.
    """
    from eeg_pipeline.utils.analysis.stats.correlation import interpret_effect_size

    n_features = len(feature_columns)
    if n_features == 0:
        return []

    if logger:
        logger.info(f"Condition effects: testing {n_features} features (vectorized batch mode)")

    data_matrix = features_df[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)

    cond_a_data = data_matrix[cond_a_mask, :]
    cond_b_data = data_matrix[cond_b_mask, :]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mean_condition_a = np.nanmean(cond_a_data, axis=0)
        mean_condition_b = np.nanmean(cond_b_data, axis=0)

        std_condition_a = np.nanstd(cond_a_data, axis=0, ddof=1)
        std_condition_b = np.nanstd(cond_b_data, axis=0, ddof=1)

        n_cond_a_per_feature = np.sum(np.isfinite(cond_a_data), axis=0)
        n_cond_b_per_feature = np.sum(np.isfinite(cond_b_data), axis=0)

    hedges_g_values = _compute_batch_hedges_g(
        mean_condition_a, mean_condition_b, std_condition_a, std_condition_b,
        n_cond_a_per_feature, n_cond_b_per_feature,
    )

    t_stats, p_values = _compute_batch_welch_ttest(
        mean_condition_a, mean_condition_b, std_condition_a, std_condition_b,
        n_cond_a_per_feature, n_cond_b_per_feature,
    )

    p_perm_values = np.full(n_features, np.nan)
    if n_perm > 0:
        if logger:
            logger.info(f"Computing {n_perm} permutations for {n_features} features...")
        p_perm_values = _compute_batch_permutation_pvalues(
            data_matrix,
            cond_a_mask,
            cond_b_mask,
            n_perm,
            base_seed,
            groups,
            scheme=scheme,
            logger=logger,
        )
    
    results: List[Dict[str, Any]] = []
    min_required = max(int(min_samples), 2)

    for i, col in enumerate(feature_columns):
        if int(n_cond_a_per_feature[i]) < min_required or int(n_cond_b_per_feature[i]) < min_required:
            continue
        hg = hedges_g_values[i]
        effect_interp = interpret_effect_size(hg) if np.isfinite(hg) else "unknown"

        results.append({
            "feature": col,
            "mean_condition_a": float(mean_condition_a[i]),
            "mean_condition_b": float(mean_condition_b[i]),
            "std_condition_a": float(std_condition_a[i]),
            "std_condition_b": float(std_condition_b[i]),
            "hedges_g": float(hg),
            "effect_interpretation": effect_interp,
            "t_statistic": float(t_stats[i]),
            "p_value": float(p_values[i]),
            "p_raw": float(p_values[i]),
            "p_perm": float(p_perm_values[i]) if np.isfinite(p_perm_values[i]) else np.nan,
            "n_permutations": n_perm if n_perm > 0 else 0,
            "n_condition_a": int(n_cond_a_per_feature[i]),
            "n_condition_b": int(n_cond_b_per_feature[i]),
        })

    if logger:
        logger.info(f"Condition effects batch computation complete: {len(results)} features")

    return results


def _compute_batch_hedges_g(
    mean1: np.ndarray,
    mean2: np.ndarray,
    std1: np.ndarray,
    std2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
) -> np.ndarray:
    """Compute Hedges' g for all features vectorized."""
    import warnings
    
    # Pooled variance
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / np.maximum(n1 + n2 - 2, 1)
    pooled_std = np.sqrt(pooled_var)
    
    # Cohen's d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cohens_d = (mean1 - mean2) / np.where(pooled_std > _NUMERIC_TOLERANCE, pooled_std, np.nan)
    
    # Hedges' g correction factor
    df = n1 + n2 - 2
    correction = np.where(df >= 2, 1 - 3 / (4 * df - 1), 1.0)
    
    return cohens_d * correction


def _compute_batch_welch_ttest(
    mean1: np.ndarray,
    mean2: np.ndarray,
    std1: np.ndarray,
    std2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch's t-test for all features vectorized."""
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Variance of means
        var1 = std1**2 / np.maximum(n1, 1)
        var2 = std2**2 / np.maximum(n2, 1)
        
        # Standard error of difference
        se_diff = np.sqrt(var1 + var2)
        
        # t-statistic
        t_stats = np.where(
            se_diff > _NUMERIC_TOLERANCE,
            (mean1 - mean2) / se_diff,
            0.0
        )
        
        # Welch-Satterthwaite degrees of freedom
        numerator = (var1 + var2) ** 2
        denominator = var1**2 / np.maximum(n1 - 1, 1) + var2**2 / np.maximum(n2 - 1, 1)
        df = np.where(denominator > _NUMERIC_TOLERANCE, numerator / denominator, 1.0)
        df = np.maximum(df, 1.0)
        
        # Two-tailed p-values
        p_values = 2 * stats.t.sf(np.abs(t_stats), df)
    
    return t_stats, p_values


def _compute_batch_permutation_pvalues(
    data_matrix: np.ndarray,
    cond_a_mask: np.ndarray,
    cond_b_mask: np.ndarray,
    n_perm: int,
    base_seed: int,
    groups: Optional[np.ndarray],
    *,
    scheme: str,
    logger: Optional[logging.Logger],
) -> np.ndarray:
    """Compute permutation p-values for all features using vectorized operations.

    Uses matrix operations to compute all permutations for all features at once,
    which is much faster than per-feature permutation loops.
    """
    import warnings

    n_samples, n_features = data_matrix.shape
    combined_mask = cond_a_mask | cond_b_mask

    valid_indices = np.where(combined_mask)[0]
    n_valid = len(valid_indices)

    if n_valid < 4:
        return np.full(n_features, np.nan)

    valid_data = data_matrix[valid_indices, :]
    valid_labels = cond_a_mask[valid_indices]  # True = condition A

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cond_a_means_obs = np.nanmean(valid_data[valid_labels, :], axis=0)
        cond_b_means_obs = np.nanmean(valid_data[~valid_labels, :], axis=0)
        observed_diff = np.abs(cond_a_means_obs - cond_b_means_obs)

    n_exceeded = np.ones(n_features)  # Start at 1 for observed

    rng = np.random.default_rng(base_seed)
    log_interval = max(1, n_perm // 10)
    scheme = str(scheme or "shuffle").strip().lower()

    for perm_i in range(n_perm):
        if logger and perm_i > 0 and perm_i % log_interval == 0:
            logger.debug(f"  Permutation {perm_i}/{n_perm}")

        from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

        valid_groups = groups[valid_indices] if groups is not None else None
        try:
            perm_indices = permute_within_groups(
                n_valid,
                rng,
                valid_groups,
                scheme=scheme,
                strict=True,
            )
        except ValueError:
            if logger:
                logger.warning(
                    "Condition permutation: grouped permutation invalid after masking; returning NaN permutation p-values."
                )
            return np.full(n_features, np.nan)
        perm_labels = valid_labels[perm_indices]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            perm_cond_a_means = np.nanmean(valid_data[perm_labels, :], axis=0)
            perm_cond_b_means = np.nanmean(valid_data[~perm_labels, :], axis=0)
            perm_diff = np.abs(perm_cond_a_means - perm_cond_b_means)

        n_exceeded += (perm_diff >= observed_diff - _NUMERIC_TOLERANCE).astype(float)

    return n_exceeded / (n_perm + 1)


def _get_condition_column(
    events_df: pd.DataFrame,
    config: Any,
) -> Optional[str]:
    """Get condition column name from config or default binary-outcome column."""
    compare_col = str(
        get_config_value(config, "behavior_analysis.condition.compare_column", "") or ""
    ).strip()
    
    if compare_col and compare_col in events_df.columns:
        return compare_col
    
    return get_binary_outcome_column_from_config(config, events_df)


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


def _split_by_binary_outcome(
    condition_series: pd.Series,
    column_name: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split by binary coding (1=condition A, 0=condition B).

    Performs numeric coercion; non-{0,1} and NaN values are excluded from both masks.
    """
    if condition_series.dtype == object:
        condition_series = condition_series.astype(str).str.strip().str.lower()

    numeric_values = pd.to_numeric(condition_series, errors="coerce").to_numpy()
    cond_a_mask = numeric_values == 1
    cond_b_mask = numeric_values == 0

    n_condition_a = int(cond_a_mask.sum())
    n_condition_b = int(cond_b_mask.sum())

    logger.info(f"Condition split: {n_condition_a} condition A, {n_condition_b} condition B trials")

    return (
        np.asarray(cond_a_mask, dtype=bool),
        np.asarray(cond_b_mask, dtype=bool),
        n_condition_a,
        n_condition_b,
    )


def split_by_condition(
    events_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split trials into two conditions based on a column and values.
    
    Supports user-configurable condition column and values via:
    - config.event_columns.binary_outcome: column name (or list of candidates)
    - config.behavior_analysis.condition.compare_column: explicit events column override (optional)
    - config.behavior_analysis.condition.compare_values: values to compare [val1, val2]
    
    If compare_values is not specified:
    - Uses [1, 0] when a binary-coded condition column is detected.
    - Otherwise auto-selects the first two observed condition values.
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
    
    condition_numeric = pd.to_numeric(condition_series, errors="coerce")
    numeric_values = {
        float(v)
        for v in pd.Series(condition_numeric).dropna().unique().tolist()
    }
    if {0.0, 1.0}.issubset(numeric_values):
        return _split_by_binary_outcome(condition_series, condition_column, logger)

    unique_values = [v for v in pd.Series(condition_series).dropna().unique().tolist()]
    if len(unique_values) >= 2:
        value1, value2 = unique_values[0], unique_values[1]
        logger.info(
            "Auto-selected condition values (no compare_values set): %s vs %s (column: %s)",
            value1,
            value2,
            condition_column,
        )
        mask1, mask2 = _create_masks_from_compare_values(
            condition_series,
            value1,
            value2,
            logger,
        )
        n_group1 = int(mask1.sum())
        n_group2 = int(mask2.sum())
        return mask1.to_numpy(), mask2.to_numpy(), n_group1, n_group2

    return _split_by_binary_outcome(condition_series, condition_column, logger)


def _compute_p_primary_columns(
    df: pd.DataFrame,
    p_primary_mode: str,
) -> Tuple[pd.Series, pd.Series]:
    """Compute p_primary and p_primary_source columns."""
    if "p_raw" not in df.columns:
        df["p_raw"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")

    p_raw = pd.to_numeric(df["p_raw"], errors="coerce")
    mode = str(p_primary_mode or "").strip().lower()

    if mode in {"perm", "permutation"}:
        if "p_perm" in df.columns:
            p_permutation = pd.to_numeric(df["p_perm"], errors="coerce")
            return (
                p_permutation.where(p_permutation.notna(), np.nan),
                pd.Series(np.where(p_permutation.notna(), "perm", "perm_missing_required"), index=df.index),
            )
        return (
            pd.Series(np.nan, index=df.index, dtype=float),
            pd.Series("perm_missing_required", index=df.index),
        )

    if "p_perm" in df.columns and mode in {"perm_if_available", "permutation_if_available"}:
        p_permutation = pd.to_numeric(df["p_perm"], errors="coerce")
        return (
            p_permutation.where(p_permutation.notna(), p_raw),
            pd.Series(np.where(p_permutation.notna(), "perm", "asymptotic"), index=df.index),
        )

    return p_raw, pd.Series("asymptotic", index=df.index)


def compute_condition_effects(
    features_df: pd.DataFrame,
    cond_a_mask: np.ndarray,
    cond_b_mask: np.ndarray,
    min_samples: int = 5,
    fdr_alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = -1,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
    paired: bool = False,
    pair_ids: Optional[np.ndarray] = None,
    p_primary_mode: Optional[str] = None,
) -> pd.DataFrame:
    """Compute effect sizes for condition A vs condition B across all features.

    Uses vectorized batch computation for large feature sets, which is
    significantly faster than per-feature computation. Constant features
    (std ≤ 1e-12) are automatically filtered and assigned zero effect.
    """
    import warnings
    
    n_jobs_actual = get_n_jobs(config, n_jobs)
    n_features = len(features_df.columns)

    if logger:
        logger.info(
            f"Computing condition effects for {n_features} features"
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
    p_primary_mode_value = (
        p_primary_mode
        if p_primary_mode is not None
        else get_config_value(
            config, "behavior_analysis.condition.p_primary_mode", "asymptotic"
        )
    )
    p_primary_mode_resolved = str(p_primary_mode_value).strip().lower()
    scheme = str(
        get_config_value(config, "behavior_analysis.permutation.scheme", "shuffle")
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
            cond_a_mask=cond_a_mask,
            cond_b_mask=cond_b_mask,
            min_samples=min_samples,
            n_jobs=n_jobs_actual,
            paired=paired,
            pair_ids=pair_ids,
            groups=groups,
            n_perm=n_perm if perm_enabled else 0,
            base_seed=base_seed,
            scheme=scheme,
            logger=logger,
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "p_primary" not in df.columns:
        p_primary, p_source = _compute_p_primary_columns(df, p_primary_mode_resolved)
        df["p_primary"] = p_primary
        if "p_primary_source" not in df.columns:
            df["p_primary_source"] = p_source

    p_primary_values = pd.to_numeric(df["p_primary"], errors="coerce").values
    df["q_value"] = fdr_bh(p_primary_values, alpha=fdr_alpha, config=config)
    df["significant_fdr"] = df["q_value"] < fdr_alpha

    df = df.sort_values("hedges_g", key=abs, ascending=False)

    if logger:
        n_significant = df["significant_fdr"].sum()
        effect_threshold = float(
            get_config_value(
                config, "behavior_analysis.condition.effect_size_threshold", 0.8
            )
        )
        n_large_effect = (df["hedges_g"].abs() >= effect_threshold).sum()
        logger.info(
            f"Condition effects summary: {n_significant}/{len(df)} FDR significant, "
            f"{n_large_effect} large effects (|g|≥{effect_threshold:.3g})"
        )

    return df


def compute_multigroup_condition_effects(
    features_df: pd.DataFrame,
    group_masks: Dict[str, np.ndarray],
    group_labels: List[str],
    fdr_alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
    paired: bool = False,
    pair_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute pairwise effect sizes for multi-group comparison (3+ groups).
    
    Performs all pairwise tests between groups with FDR correction across all tests.
    Uses Mann-Whitney U for unpaired mode and Wilcoxon signed-rank for paired mode.
    
    Args:
        features_df: DataFrame with feature columns
        group_masks: Dict mapping group_label -> boolean mask array
        group_labels: Ordered list of group labels for consistent pair ordering
        fdr_alpha: FDR significance threshold
        logger: Optional logger
        config: Optional config object
        
    Returns:
        DataFrame with columns: feature, group1, group2, n1, n2, mean1, mean2,
        cohens_d, hedges_g, p_value, q_value, significant_fdr
    """
    from itertools import combinations
    from scipy.stats import mannwhitneyu
    from scipy.stats import wilcoxon

    if len(group_labels) < 2:
        if logger:
            logger.warning("Multi-group comparison requires at least 2 groups")
        return pd.DataFrame()

    pair_ids_array = None
    if paired:
        if pair_ids is None:
            raise ValueError("pair_ids are required when paired=True for multi-group comparisons.")
        pair_ids_array = np.asarray(pair_ids)
        if len(pair_ids_array) != len(features_df):
            raise ValueError("pair_ids length must match features_df rows for paired multi-group comparisons.")
    
    available_groups = [g for g in group_labels if g in group_masks and np.any(group_masks[g])]
    if len(available_groups) < 2:
        if logger:
            logger.warning(f"Only {len(available_groups)} groups have data; need at least 2")
        return pd.DataFrame()
    
    feature_columns = list(features_df.columns)
    n_features = len(feature_columns)
    n_pairs = len(list(combinations(available_groups, 2)))
    
    if logger:
        logger.info(
            f"Computing multi-group condition effects: {n_features} features × "
            f"{n_pairs} group pairs ({len(available_groups)} groups)"
        )
    
    records = []
    
    for feature in feature_columns:
        values = pd.to_numeric(features_df[feature], errors="coerce").values
        
        for g1, g2 in combinations(available_groups, 2):
            mask1, mask2 = group_masks[g1], group_masks[g2]
            v1 = values[mask1]
            v2 = values[mask2]
            
            v1_clean = v1[np.isfinite(v1)]
            v2_clean = v2[np.isfinite(v2)]
            
            n1, n2 = len(v1_clean), len(v2_clean)
            
            if n1 < 3 or n2 < 3:
                continue
            
            if paired and pair_ids_array is not None:
                ids1 = pair_ids_array[mask1]
                ids2 = pair_ids_array[mask2]

                s1 = pd.Series(v1, index=ids1).groupby(level=0).mean()
                s2 = pd.Series(v2, index=ids2).groupby(level=0).mean()
                paired_df = pd.concat([s1.rename("v1"), s2.rename("v2")], axis=1, join="inner").dropna()
                n_pairs = int(len(paired_df))
                if n_pairs < 3:
                    continue

                x1 = paired_df["v1"].to_numpy(dtype=float)
                x2 = paired_df["v2"].to_numpy(dtype=float)
                diffs = x1 - x2

                mean1, mean2 = float(np.mean(x1)), float(np.mean(x2))
                diff_std = float(np.std(diffs, ddof=1)) if n_pairs > 1 else np.nan
                d = float(np.mean(diffs) / diff_std) if np.isfinite(diff_std) and diff_std > 0 else np.nan
                g = float(d * (1 - (3 / (4 * n_pairs - 1)))) if np.isfinite(d) and n_pairs > 1 else np.nan

                try:
                    _, p_value = wilcoxon(x1, x2, alternative="two-sided")
                except (ValueError, RuntimeError):
                    p_value = np.nan

                n1 = n_pairs
                n2 = n_pairs
            else:
                n_pairs = np.nan
                mean1, mean2 = np.mean(v1_clean), np.mean(v2_clean)
                d = cohens_d(v1_clean, v2_clean, pooled=True, config=config)
                g = hedges_g(v1_clean, v2_clean, config=config)

                try:
                    _, p_value = mannwhitneyu(v1_clean, v2_clean, alternative="two-sided")
                except (ValueError, RuntimeError):
                    p_value = np.nan
            
            records.append({
                "feature": feature,
                "group1": g1,
                "group2": g2,
                "n1": n1,
                "n2": n2,
                "mean1": mean1,
                "mean2": mean2,
                "cohens_d": d,
                "hedges_g": g,
                "p_value": p_value,
                "paired_test": bool(paired),
                "n_pairs": n_pairs,
            })
    
    if not records:
        if logger:
            logger.warning("No valid comparisons computed for multi-group analysis")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    p_values = pd.to_numeric(df["p_value"], errors="coerce").values
    df["q_value"] = fdr_bh(p_values, alpha=fdr_alpha, config=config)
    df["significant_fdr"] = df["q_value"] < fdr_alpha
    
    df["comparison_type"] = "multigroup"
    df["analysis_kind"] = "condition_multigroup"
    
    df = df.sort_values("hedges_g", key=abs, ascending=False)
    
    if logger:
        n_significant = df["significant_fdr"].sum()
        n_tests = len(df)
        logger.info(
            f"Multi-group condition effects: {n_significant}/{n_tests} FDR significant"
        )
    
    return df
