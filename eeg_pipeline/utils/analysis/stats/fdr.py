"""
FDR Correction
==============

Benjamini-Hochberg false discovery rate correction functions and utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import get_config_value, get_fdr_alpha
from .correlation import compute_correlation

if TYPE_CHECKING:
    from pathlib import Path


def fdr_bh(
    pvals: Iterable[float],
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    
    Returns q-values (adjusted p-values).
    """
    if alpha is None:
        alpha = get_fdr_alpha(config)

    pvals_arr = np.asarray(list(pvals), dtype=float)
    qvals = np.full_like(pvals_arr, np.nan, dtype=float)

    valid_mask = np.isfinite(pvals_arr)
    if not np.any(valid_mask):
        return qvals

    pv = pvals_arr[valid_mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size

    denom = np.arange(1, n + 1, dtype=float)
    adjusted = ranked * n / denom
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted

    qvals[valid_mask] = restored
    return qvals


def fdr_bh_reject(
    pvals: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, float]:
    """
    BH-FDR rejection decision.
    
    Returns (reject_mask, critical_value).
    """
    if alpha is None:
        alpha = get_fdr_alpha(config)

    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.nan

    valid_mask = np.isfinite(p)
    if not np.any(valid_mask):
        return np.zeros_like(p, dtype=bool), np.nan

    p_valid = p[valid_mask]
    order = np.argsort(p_valid)
    ranked = np.arange(1, len(p_valid) + 1)
    thresh = (ranked / len(p_valid)) * alpha
    passed = p_valid[order] <= thresh

    if not np.any(passed):
        return np.zeros_like(p, dtype=bool), np.nan

    k_max = np.max(np.where(passed)[0])
    crit = float(p_valid[order][k_max])

    reject = np.zeros_like(p, dtype=bool)
    reject[valid_mask] = p_valid <= crit

    return reject, crit


def fdr_bh_mask(
    p_vals: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Return boolean mask of significant values after BH-FDR."""
    if p_vals is None or len(p_vals) == 0:
        return None
    reject, _ = fdr_bh_reject(p_vals, alpha, config)
    return reject


def fdr_bh_values(
    p_vals: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (q_values, reject_mask) after BH-FDR."""
    if p_vals is None or len(p_vals) == 0:
        return None, None
    qvals = fdr_bh(p_vals, alpha, config)
    reject, _ = fdr_bh_reject(p_vals, alpha, config)
    return qvals, reject


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Simple BH adjustment without config."""
    return fdr_bh(pvals, alpha=0.05)


def select_p_values_for_fdr(
    results_df: pd.DataFrame,
    use_permutation_p: bool,
) -> np.ndarray:
    """Select appropriate p-values from results DataFrame for FDR."""
    if use_permutation_p and "p_perm" in results_df.columns:
        return results_df["p_perm"].values
    if "p_value" in results_df.columns:
        return results_df["p_value"].values
    return results_df["p"].values if "p" in results_df.columns else np.array([])


def filter_significant_predictors(
    results_df: pd.DataFrame,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
    p_col: str = "p",
    q_col: str = "q_value",
) -> pd.DataFrame:
    """Filter results to significant predictors after FDR."""
    if alpha is None:
        alpha = get_fdr_alpha(config)

    if q_col not in results_df.columns:
        results_df = results_df.copy()
        results_df[q_col] = fdr_bh(results_df[p_col].values, alpha, config)

    return results_df[results_df[q_col] <= alpha].copy()


# ============================================================================
# FDR Utilities
# ============================================================================

def apply_fdr_correction_and_save(
    results_df: pd.DataFrame,
    output_path: "Path",
    config: Any,
    logger: logging.Logger,
    use_permutation_p: bool = True,
) -> None:
    """Apply FDR correction and save results."""
    if results_df.empty or "p" not in results_df.columns:
        return
    
    alpha = get_fdr_alpha(config)
    p_vec = select_p_values_for_fdr(results_df, use_permutation_p)
    
    rej, crit = fdr_bh_reject(p_vec, alpha=alpha)
    results_df["fdr_reject"] = rej
    results_df["fdr_crit_p"] = crit
    
    results_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(results_df)} results to {output_path}")


def get_pvalue_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Extract permutation and raw p-value series."""
    p_perm = pd.Series(index=df.index, dtype=float)
    p_raw = pd.Series(index=df.index, dtype=float)
    
    for col in ["p_partial_perm", "p_partial_temp_perm", "p_perm"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            mask = vals.notna()
            p_perm.loc[mask] = vals.loc[mask]
    
    for col in ["p", "p_value", "p_partial", "p_partial_temp"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            mask = vals.notna()
            p_raw.loc[mask] = vals.loc[mask]
    
    return p_perm, p_raw


def extract_pvalue_from_dataframe(df: pd.DataFrame, row_idx: int) -> Tuple[float, str]:
    """Extract p-value for a row with column name."""
    if row_idx < 0 or row_idx >= len(df):
        return np.nan, ""
    
    for col in ["p_partial_perm", "p_partial_temp_perm", "p_perm", "p", "p_value", "p_partial"]:
        if col in df.columns:
            p = pd.to_numeric(df.iloc[row_idx][col], errors="coerce")
            if pd.notna(p):
                return float(p), col
    
    return np.nan, ""


def should_apply_fisher_transform(prefix: str) -> bool:
    """Check if Fisher transform should be applied for measure."""
    measure = prefix.split("_", 1)[0].lower()
    return measure in ("aec", "aec_orth", "corr", "pearsonr")


def get_cluster_correction_config(
    heatmap_config: Dict[str, Any],
    config: Any,
    alpha: float,
    default_rng_seed: Optional[int] = None,
) -> Tuple[float, int, np.random.Generator, int]:
    """Get cluster correction configuration."""
    if default_rng_seed is None:
        default_rng_seed = int(get_config_value(config, "project.random_state", 42))
    
    cluster_cfg = config.get("behavior_analysis.cluster_correction", {}) if config else {}
    cluster_alpha = float(heatmap_config.get("cluster_alpha", cluster_cfg.get("alpha", alpha)))
    n_perm = int(heatmap_config.get("n_cluster_perm", cluster_cfg.get("n_permutations", 100)))
    seed = int(heatmap_config.get("cluster_rng_seed", cluster_cfg.get("rng_seed", default_rng_seed)))
    
    return cluster_alpha, n_perm, np.random.default_rng(seed), seed


def compute_fdr_rejections_for_heatmap(
    p_value_matrix: np.ndarray,
    n_nodes: int,
    config: Any,
) -> Tuple[Dict[Tuple[int, int], bool], float]:
    """Compute FDR rejections for heatmap."""
    upper_idx = np.triu_indices(n_nodes, k=1)
    p_upper = p_value_matrix[upper_idx]
    valid = np.isfinite(p_upper)
    p_valid = p_upper[valid]
    
    alpha = get_fdr_alpha(config)
    rej, crit = fdr_bh_reject(p_valid, alpha=alpha)
    
    pairs = [(upper_idx[0][k], upper_idx[1][k]) for k in np.where(valid)[0]]
    rej_map = {pair: bool(rej[k]) for k, pair in enumerate(pairs)}
    crit_val = float(np.max(p_valid[rej])) if np.any(rej) else np.nan
    
    return rej_map, crit_val


def build_correlation_matrices_for_prefix(
    prefix: str,
    prefix_columns: list,
    connectivity_df: pd.DataFrame,
    target_values: pd.Series,
    node_to_index: Dict[str, int],
    use_spearman: bool,
    min_samples: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build correlation matrices for connectivity prefix."""
    n = len(node_to_index)
    r_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)
    
    for col in prefix_columns:
        parts = col.split(prefix + "_", 1)
        if len(parts) < 2:
            continue
        pair = parts[-1]
        
        for sep in ["--", "-", "_"]:
            if sep in pair:
                nodes = pair.split(sep)
                if len(nodes) == 2:
                    a, b = nodes
                    if a in node_to_index and b in node_to_index:
                        i, j = node_to_index[a], node_to_index[b]
                        edge = pd.to_numeric(connectivity_df[col], errors="coerce")
                        valid = edge.notna() & target_values.notna()
                        if valid.sum() >= min_samples:
                            r, p = compute_correlation(edge[valid].values, target_values[valid].values, "spearman" if use_spearman else "pearson")
                            r_mat[i, j] = r_mat[j, i] = r
                            p_mat[i, j] = p_mat[j, i] = p
                    break
    
    return r_mat, p_mat
