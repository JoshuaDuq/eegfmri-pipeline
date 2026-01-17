"""
Permutation Testing
===================

Permutation tests for partial correlations and group comparisons.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_statistics_constants, get_n_permutations
from eeg_pipeline.utils.config.loader import get_config_value

# Constants for numerical stability
_DESIGN_MATRIX_CONDITION_TOLERANCE = 1e8
_DESIGN_MATRIX_RANK_TOLERANCE = 1e-10
_RESIDUAL_VARIANCE_TOLERANCE_FACTOR = 1e-12


def _get_permutation_scheme(config: Optional[Any]) -> str:
    """Extract permutation scheme from config."""
    if config is None:
        return "shuffle"
    scheme = str(get_config_value(config, "behavior_analysis.permutation.scheme", "shuffle")).strip().lower()
    return scheme if scheme in {"shuffle", "circular_shift"} else "shuffle"


def permute_within_groups(
    n: int,
    rng: np.random.Generator,
    groups: Optional[np.ndarray] = None,
    min_group_size: int = 2,
    *,
    scheme: str = "shuffle",
) -> np.ndarray:
    """Generate permutation indices, optionally within groups.
    
    Raises ValueError if any group has fewer than min_group_size samples.
    """
    scheme = str(scheme or "shuffle").strip().lower()
    if scheme not in {"shuffle", "circular_shift"}:
        scheme = "shuffle"

    if groups is None:
        idx = np.arange(n)
        if scheme == "circular_shift":
            if n <= 1:
                return idx
            shift = int(rng.integers(0, n))
            return np.roll(idx, shift)
        rng.shuffle(idx)
        return idx

    unique, counts = np.unique(groups, return_counts=True)
    small_groups = unique[counts < min_group_size]
    if len(small_groups) > 0:
        raise ValueError(
            f"Groups {small_groups.tolist()} have fewer than {min_group_size} samples. "
            f"Permutation within groups requires at least {min_group_size} per group."
        )

    idx = np.arange(n)
    for g in unique:
        mask = groups == g
        sub = idx[mask]
        if scheme == "circular_shift":
            if sub.size <= 1:
                continue
            shift = int(rng.integers(0, sub.size))
            sub = np.roll(sub, shift)
        else:
            rng.shuffle(sub)
        idx[mask] = sub
    return idx


def _align_groups_to_dataframe(
    groups: Optional[np.ndarray],
    x_index: pd.Index,
) -> Optional[pd.Series]:
    """Convert groups array to Series aligned with x's index."""
    if groups is None:
        return None
    
    if isinstance(groups, pd.Series):
        return groups
    
    groups_array = np.asarray(groups)
    if len(groups_array) != len(x_index):
        raise ValueError(
            f"groups length ({len(groups_array)}) must match x length "
            f"({len(x_index)}) before dropna subsetting"
        )
    
    return pd.Series(groups_array, index=x_index)


def _subset_groups_after_dropna(
    groups_series: Optional[pd.Series],
    df_index: pd.Index,
) -> Optional[np.ndarray]:
    """Subset groups to match non-NaN rows in dataframe."""
    if groups_series is None:
        return None
    
    groups_subset = groups_series.reindex(df_index)
    if groups_subset.isna().any():
        return None
    return groups_subset.to_numpy()


def _prepare_ranked_data(
    df: pd.DataFrame,
    z_columns: pd.Index,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare ranked or raw data for correlation."""
    if method == "spearman":
        x_values = stats.rankdata(df["x"].to_numpy())
        y_values = stats.rankdata(df["y"].to_numpy())
        z_values = np.column_stack([
            stats.rankdata(df[col].to_numpy())
            for col in z_columns
        ]) if len(z_columns) > 0 else np.empty((len(df), 0))
    else:
        x_values = df["x"].to_numpy()
        y_values = df["y"].to_numpy()
        z_values = (
            df[z_columns].to_numpy()
            if len(z_columns) > 0
            else np.empty((len(df), 0))
        )
    
    return x_values, y_values, z_values


def _validate_design_matrix(design: np.ndarray) -> bool:
    """Validate design matrix for numerical stability."""
    condition_number = np.linalg.cond(design)
    is_finite = np.isfinite(condition_number)
    is_well_conditioned = condition_number <= _DESIGN_MATRIX_CONDITION_TOLERANCE
    
    matrix_rank = np.linalg.matrix_rank(design, tol=_DESIGN_MATRIX_RANK_TOLERANCE)
    is_full_rank = matrix_rank >= design.shape[1]
    
    return is_finite and is_well_conditioned and is_full_rank


def _compute_residuals(
    design: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute residuals and fitted values for x and y."""
    x_coefficients = np.linalg.lstsq(design, x_values, rcond=None)[0]
    y_coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
    
    x_residuals = x_values - design @ x_coefficients
    y_residuals = y_values - design @ y_coefficients
    y_fitted = design @ y_coefficients
    
    return x_residuals, y_residuals, y_fitted


def _validate_residual_variance(
    x_residuals: np.ndarray,
    y_residuals: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> bool:
    """Validate that residuals have sufficient variance."""
    max_variance = max(np.var(x_values), np.var(y_values), 1.0)
    variance_tolerance = _RESIDUAL_VARIANCE_TOLERANCE_FACTOR * max_variance
    
    x_residual_variance = np.var(x_residuals, ddof=1)
    y_residual_variance = np.var(y_residuals, ddof=1)
    
    has_sufficient_variance = (
        x_residual_variance >= variance_tolerance
        and y_residual_variance >= variance_tolerance
    )
    
    return has_sufficient_variance


def perm_pval_simple(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_perm: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    *,
    scheme: str = "shuffle",
) -> float:
    """
    Simple permutation p-value for correlation.
    
    Returns two-sided p-value.
    """
    if n_perm is None:
        n_perm = get_n_permutations(config)
    if rng is None:
        rng = np.random.default_rng()

    # Local import to avoid circular dependency
    from .correlation import compute_correlation

    x_array = np.asarray(x).ravel()
    y_array = np.asarray(y).ravel()
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_valid = x_array[valid]
    y_valid = y_array[valid]
    groups_valid = groups[valid] if groups is not None else None
    
    n_valid = len(x_valid)
    MIN_SAMPLES_PERMUTATION = 3
    if n_valid < MIN_SAMPLES_PERMUTATION:
        return np.nan

    observed_correlation, _ = compute_correlation(x_valid, y_valid, method)
    if not np.isfinite(observed_correlation):
        return np.nan

    observed_abs = np.abs(observed_correlation)
    n_extreme = 0
    for _ in range(n_perm):
        perm_indices = permute_within_groups(
            n_valid,
            rng,
            groups_valid,
            scheme=scheme,
        )
        perm_correlation, _ = compute_correlation(
            x_valid[perm_indices], y_valid, method
        )
        if np.isfinite(perm_correlation):
            perm_abs = np.abs(perm_correlation)
            if perm_abs >= observed_abs:
                n_extreme += 1

    return (n_extreme + 1) / (n_perm + 1)


def perm_pval_partial_freedman_lane(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    *,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    scheme: str = "shuffle",
) -> float:
    """Freedman-Lane permutation test for partial correlation.
    
    Note: groups can be np.ndarray or pd.Series. If np.ndarray, it must be
    aligned to the original x/y indices. After dropna, groups will be subset
    to match the kept rows.
    """
    groups_series = _align_groups_to_dataframe(groups, x.index)
    
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    has_sufficient_samples = len(df) >= min_samples
    has_valid_permutations = n_perm is not None and n_perm > 0
    
    if not has_sufficient_samples or not has_valid_permutations:
        return np.nan
    
    groups_array = _subset_groups_after_dropna(groups_series, df.index)
    
    intercept = np.ones(len(df))
    x_values, y_values, z_values = _prepare_ranked_data(df, Z.columns, method)
    
    design = np.column_stack([intercept, z_values])
    if not _validate_design_matrix(design):
        return np.nan
    
    x_residuals, y_residuals, y_fitted = _compute_residuals(
        design, x_values, y_values
    )
    
    if not _validate_residual_variance(
        x_residuals, y_residuals, x_values, y_values
    ):
        return np.nan
    
    observed_correlation, _ = stats.pearsonr(x_residuals, y_residuals)
    exceed_count = 1
    
    max_variance = max(np.var(x_values), np.var(y_values), 1.0)
    variance_tolerance = _RESIDUAL_VARIANCE_TOLERANCE_FACTOR * max_variance
    
    for _ in range(n_perm):
        permuted_indices = permute_within_groups(
            len(y_residuals),
            rng,
            groups_array,
            scheme=scheme,
        )
        y_permuted = y_fitted + y_residuals[permuted_indices]
        
        try:
            y_permuted_coefficients = np.linalg.lstsq(
                design, y_permuted, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            continue
        
        y_permuted_residuals = y_permuted - design @ y_permuted_coefficients
        y_permuted_variance = np.var(y_permuted_residuals, ddof=1)
        
        if y_permuted_variance < variance_tolerance:
            continue
        
        permuted_correlation, _ = stats.pearsonr(
            x_residuals, y_permuted_residuals
        )
        
        if np.abs(permuted_correlation) >= np.abs(observed_correlation):
            exceed_count += 1
    
    return exceed_count / (n_perm + 1)


def compute_perm_and_partial_perm(
    x: pd.Series,
    y: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    *,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    scheme: str = "shuffle",
) -> Tuple[float, float]:
    """Compute permutation p-values for simple and partial correlation."""
    p_perm = p_partial_perm = np.nan
    if n_perm is None or n_perm <= 0:
        return p_perm, p_partial_perm
    
    p_perm = perm_pval_simple(
        x,
        y,
        method,
        n_perm,
        rng,
        groups=groups,
        config=config,
        scheme=scheme,
    )
    
    if covariates_df is not None and not covariates_df.empty:
        p_partial_perm = perm_pval_partial_freedman_lane(
            x,
            y,
            covariates_df,
            method,
            n_perm,
            rng,
            groups=groups,
            config=config,
            scheme=scheme,
        )
    
    return p_perm, p_partial_perm


def compute_permutation_pvalues(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    method: str,
    n_perm: Optional[int],
    n_eff: int,
    rng: np.random.Generator,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute all permutation p-values for ROI analysis."""
    
    p_perm = p_partial_perm = p_temp_perm = np.nan
    
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    if n_perm is None or n_perm <= 0 or n_eff < min_samples:
        return p_perm, p_partial_perm, p_temp_perm

    scheme = _get_permutation_scheme(config)
    p_perm = perm_pval_simple(
        x_aligned,
        y_aligned,
        method,
        n_perm,
        rng,
        groups=groups,
        config=config,
        scheme=scheme,
    )

    if covariates_df is not None and not covariates_df.empty:
        p_partial_perm = perm_pval_partial_freedman_lane(
            x_aligned,
            y_aligned,
            covariates_df,
            method,
            n_perm,
            rng,
            groups=groups,
            config=config,
            scheme=scheme,
        )

    if temp_series is not None and not temp_series.empty:
        temperature_covariates = pd.DataFrame({"temp": temp_series})
        p_temp_perm = perm_pval_partial_freedman_lane(
            x_aligned,
            y_aligned,
            temperature_covariates,
            method,
            n_perm,
            rng,
            groups=groups,
            config=config,
            scheme=scheme,
        )

    return p_perm, p_partial_perm, p_temp_perm


def _compute_combined_covariates_temp_pvalue(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    method: str,
    n_perm: Optional[int],
    n_eff: int,
    rng: np.random.Generator,
    min_samples: Optional[int],
    config: Optional[Any],
    groups: Optional[np.ndarray],
) -> float:
    """Compute permutation p-value for combined covariates and temperature."""
    has_valid_permutations = n_perm is not None and n_perm > 0
    has_covariates = covariates_df is not None and not covariates_df.empty
    has_temperature = temp_series is not None and not temp_series.empty
    
    if not has_valid_permutations or not has_covariates or not has_temperature:
        return np.nan
    
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    if n_eff < min_samples:
        return np.nan
    
    combined_covariates = covariates_df.copy()
    combined_covariates["temp"] = temp_series
    combined_covariates = combined_covariates.dropna()
    
    if combined_covariates.empty:
        return np.nan
    
    x_subset = x_aligned.reindex(combined_covariates.index)
    y_subset = y_aligned.reindex(combined_covariates.index)
    
    if x_subset.empty or y_subset.empty:
        return np.nan
    
    try:
        return perm_pval_partial_freedman_lane(
            x_subset,
            y_subset,
            combined_covariates,
            method,
            n_perm,
            rng,
            groups=groups,
            config=config,
            scheme=_get_permutation_scheme(config),
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def perm_pval_mean_difference(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
    groups: Optional[np.ndarray] = None,
    scheme: str = "shuffle",
    min_samples_per_condition: int = 2,
) -> float:
    """Permutation p-value for absolute difference in means between two boolean groups."""
    values = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if values.size != labels.size:
        raise ValueError("values and labels must have same length")

    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    groups_use = groups[finite] if groups is not None else None

    if values.size < 4:
        return np.nan
    if int(labels.sum()) < int(min_samples_per_condition) or int((~labels).sum()) < int(min_samples_per_condition):
        return np.nan

    observed = float(np.abs(np.nanmean(values[labels]) - np.nanmean(values[~labels])))
    if not np.isfinite(observed):
        return np.nan

    exceed = 1
    for _ in range(int(n_perm)):
        idx = permute_within_groups(values.size, rng, groups_use, scheme=scheme)
        perm_labels = labels[idx]
        perm_stat = float(np.abs(np.nanmean(values[perm_labels]) - np.nanmean(values[~perm_labels])))
        if np.isfinite(perm_stat) and perm_stat >= observed:
            exceed += 1
    return float(exceed / (int(n_perm) + 1))


def compute_permutation_pvalues_with_cov_temp(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    method: str,
    n_perm: Optional[int],
    n_eff: int,
    rng: np.random.Generator,
    *,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Compute permutation p-values including combined covariates+temp."""
    p_perm, p_partial_cov, p_partial_temp = compute_permutation_pvalues(
        x_aligned=x_aligned,
        y_aligned=y_aligned,
        covariates_df=covariates_df,
        temp_series=temp_series,
        method=method,
        n_perm=n_perm,
        n_eff=n_eff,
        rng=rng,
        min_samples=min_samples,
        config=config,
        groups=groups,
    )

    p_partial_cov_temp = _compute_combined_covariates_temp_pvalue(
        x_aligned,
        y_aligned,
        covariates_df,
        temp_series,
        method,
        n_perm,
        n_eff,
        rng,
        min_samples,
        config,
        groups,
    )

    return p_perm, p_partial_cov, p_partial_temp, p_partial_cov_temp


def compute_temp_permutation_pvalues(
    roi_values: pd.Series,
    temp_values: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    method: str,
    n_perm: Optional[int],
    rng: np.random.Generator,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute temperature permutation p-values."""
    
    p_temp_perm = p_partial_perm = np.nan
    
    if n_perm is None or n_perm <= 0:
        return p_temp_perm, p_partial_perm
    
    p_temp_perm = perm_pval_simple(roi_values, temp_values, method, n_perm, rng, groups=groups, config=config)
    
    if covariates_without_temp_df is not None and not covariates_without_temp_df.empty:
        p_partial_perm = perm_pval_partial_freedman_lane(roi_values, temp_values, covariates_without_temp_df, method, n_perm, rng, groups=groups, config=config)
    
    return p_temp_perm, p_partial_perm


def _extract_groups_from_covariates(
    covariates_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Extract group information from covariates if present."""
    group_column_names = ["run", "run_id", "block"]
    
    for column in covariates_df.columns:
        column_lower = str(column).lower()
        if column_lower in group_column_names:
            group_candidate = pd.to_numeric(
                covariates_df[column], errors="coerce"
            ).to_numpy()
            valid_groups = group_candidate[~np.isnan(group_candidate)]
            has_multiple_groups = len(np.unique(valid_groups)) < len(group_candidate)
            
            if has_multiple_groups:
                return group_candidate
    
    return None


def _subset_covariates_with_mask(
    covariates_df: pd.DataFrame,
    mask: pd.Series,
) -> Optional[pd.DataFrame]:
    """Subset covariates dataframe using mask."""
    if isinstance(mask, pd.Series):
        covariates_subset = covariates_df.iloc[mask]
    else:
        covariates_subset = covariates_df[mask]
    
    if covariates_subset.empty:
        return None
    
    return covariates_subset


def compute_permutation_pvalues_for_roi_pair(
    x_masked: pd.Series,
    y_masked: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    mask: pd.Series,
    method: str,
    n_perm: Optional[int],
    n_eff: int,
    rng: np.random.Generator,
    min_samples: int = 5,
) -> Tuple[float, float]:
    """Compute permutation p-values for ROI pair."""
    p_perm = p_partial_perm = np.nan
    
    has_valid_permutations = n_perm is not None and n_perm > 0
    has_sufficient_samples = n_eff >= min_samples
    
    if not has_valid_permutations or not has_sufficient_samples:
        return p_perm, p_partial_perm
    
    covariates_valid = None
    groups = None
    
    if covariates_df is not None and not covariates_df.empty:
        covariates_valid = _subset_covariates_with_mask(covariates_df, mask)
        
        if covariates_valid is not None:
            groups = _extract_groups_from_covariates(covariates_valid)
    
    p_perm, p_partial_perm = compute_perm_and_partial_perm(
        x_masked, y_masked, covariates_valid, method, n_perm, rng, groups=groups
    )
    
    return p_perm, p_partial_perm


def permutation_null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 1000,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate permutation null distribution for correlation.
    
    Returns
    -------
    null_rs : array
        Null distribution of correlations
    observed_r : float
        Observed correlation
    p_perm : float
        Permutation p-value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x_array = np.asarray(x).ravel()
    y_array = np.asarray(y).ravel()
    
    is_finite_x = np.isfinite(x_array)
    is_finite_y = np.isfinite(y_array)
    valid_mask = is_finite_x & is_finite_y
    n_valid = valid_mask.sum()
    
    # Minimum sample size for valid permutation test
    MIN_SAMPLE_SIZE = 4
    if n_valid < MIN_SAMPLE_SIZE:
        return np.array([]), np.nan, np.nan
    
    x_valid = x_array[valid_mask]
    y_valid = y_array[valid_mask]
    
    from .correlation import compute_correlation
    observed_r, _ = compute_correlation(x_valid, y_valid, method)
    observed_r = observed_r if np.isfinite(observed_r) else np.nan
    
    null_correlations = np.zeros(n_perm)
    for perm_idx in range(n_perm):
        y_permuted = rng.permutation(y_valid)
        r_perm, _ = compute_correlation(x_valid, y_permuted, method)
        null_correlations[perm_idx] = r_perm if np.isfinite(r_perm) else np.nan
    
    abs_observed = np.abs(observed_r)
    abs_null = np.abs(null_correlations)
    n_extreme = np.sum(abs_null >= abs_observed)
    p_perm = (n_extreme + 1) / (n_perm + 1)
    
    return null_correlations, float(observed_r), float(p_perm)


