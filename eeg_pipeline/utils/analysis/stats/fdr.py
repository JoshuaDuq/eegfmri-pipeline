"""
FDR Correction
==============

Benjamini-Hochberg false discovery rate correction functions and utilities.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .base import get_fdr_alpha


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

    p_values_array = np.asarray(list(pvals), dtype=float)
    q_values = np.full_like(p_values_array, np.nan, dtype=float)

    valid_mask = np.isfinite(p_values_array)
    if not np.any(valid_mask):
        return q_values

    valid_p_values = p_values_array[valid_mask]
    sort_order = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[sort_order]
    n_tests = sorted_p_values.size

    ranks = np.arange(1, n_tests + 1, dtype=float)
    adjusted_p_values = sorted_p_values * n_tests / ranks
    adjusted_p_values = np.minimum.accumulate(adjusted_p_values[::-1])[::-1]
    adjusted_p_values = np.clip(adjusted_p_values, 0.0, 1.0)

    restored_q_values = np.empty_like(adjusted_p_values)
    restored_q_values[sort_order] = adjusted_p_values

    q_values[valid_mask] = restored_q_values
    return q_values


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

    p_values = np.asarray(pvals, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=bool), np.nan

    valid_mask = np.isfinite(p_values)
    if not np.any(valid_mask):
        return np.zeros_like(p_values, dtype=bool), np.nan

    valid_p_values = p_values[valid_mask]
    sort_order = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[sort_order]
    n_tests = len(valid_p_values)
    
    ranks = np.arange(1, n_tests + 1)
    thresholds = (ranks / n_tests) * alpha
    passed_threshold = sorted_p_values <= thresholds

    if not np.any(passed_threshold):
        return np.zeros_like(p_values, dtype=bool), np.nan

    max_passed_index = np.max(np.where(passed_threshold)[0])
    critical_value = float(sorted_p_values[max_passed_index])

    reject_mask = np.zeros_like(p_values, dtype=bool)
    reject_mask[valid_mask] = valid_p_values <= critical_value

    return reject_mask, critical_value


def fdr_bh_values(
    p_values: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q_values, reject_mask) after BH-FDR."""
    q_values = fdr_bh(p_values, alpha, config)
    reject_mask, _ = fdr_bh_reject(p_values, alpha, config)
    return q_values, reject_mask


def _simes_family_pvalue(p_values: np.ndarray) -> float:
    """Compute Simes family-level p-value for a vector of p-values."""
    p_arr = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p_arr)
    if not np.any(valid):
        return np.nan

    sorted_p = np.sort(p_arr[valid])
    n = sorted_p.size
    ranks = np.arange(1, n + 1, dtype=float)
    simes = np.min(np.clip(sorted_p * n / ranks, 0.0, 1.0))
    return float(simes)


def hierarchical_fdr(
    df: pd.DataFrame,
    p_col: str = "p_value",
    family_col: str = "family_id",
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Apply hierarchical FDR correction with explicit family structure.
    
    Two-level procedure:
    1) Family-level gate: compute one p-value per family (Simes) and run BH-FDR
    2) Within-family BH-FDR, with final within-family rejection requiring a
       passed family-level gate.

    Also computes global BH q-values across all tests for reporting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with p-values and family assignments
    p_col : str
        Column containing p-values
    family_col : str
        Column containing family assignments
    alpha : float, optional
        FDR alpha level
    config : Any, optional
        Configuration object
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - family_id: Family identifier (copied if not present)
        - family_kind: Type of family grouping
        - family_p_gate: Family-level Simes p-value
        - family_q_gate: Family-level BH q-value
        - family_reject_gate: Family-level rejection decision
        - q_within_family: FDR q-values within family
        - q_global: Global FDR q-values
        - reject_within_family: Boolean rejection within family
        - reject_global: Boolean rejection globally
        - family_n_tests: Number of tests in family
        - family_n_reject: Number of rejections in family
    """
    if alpha is None:
        alpha = get_fdr_alpha(config)
    
    df = df.copy()
    
    if family_col not in df.columns:
        df["family_id"] = "default"
        family_col = "family_id"
    else:
        df["family_id"] = df[family_col]
    
    if "family_kind" not in df.columns:
        if "feature_type" in df.columns:
            df["family_kind"] = "feature_type"
        elif "analysis_type" in df.columns:
            df["family_kind"] = "analysis_type"
        else:
            df["family_kind"] = "inferred"
    
    df["q_within_family"] = np.nan
    df["reject_within_family"] = False
    df["family_n_tests"] = 0
    df["family_n_reject"] = 0
    df["family_p_gate"] = np.nan
    df["family_q_gate"] = np.nan
    df["family_reject_gate"] = False
    
    for family in df["family_id"].unique():
        mask = df["family_id"] == family
        family_df = df.loc[mask]
        
        if p_col not in family_df.columns:
            continue
        
        p_values = pd.to_numeric(family_df[p_col], errors="coerce").to_numpy()
        valid_mask = np.isfinite(p_values)
        
        if not np.any(valid_mask):
            continue
        
        q_values = fdr_bh(p_values, alpha=alpha, config=config)
        reject_mask = q_values < alpha
        
        df.loc[mask, "q_within_family"] = q_values
        df.loc[mask, "reject_within_family"] = reject_mask
        df.loc[mask, "family_n_tests"] = int(valid_mask.sum())
        df.loc[mask, "family_n_reject"] = int(reject_mask.sum())
        df.loc[mask, "family_p_gate"] = _simes_family_pvalue(p_values)

    family_gate = (
        df[["family_id", "family_p_gate"]]
        .drop_duplicates(subset=["family_id"])
        .reset_index(drop=True)
    )
    if not family_gate.empty:
        family_gate_q = fdr_bh(
            pd.to_numeric(family_gate["family_p_gate"], errors="coerce").to_numpy(),
            alpha=alpha,
            config=config,
        )
        family_gate["family_q_gate"] = family_gate_q
        family_gate["family_reject_gate"] = family_gate_q < alpha
        q_map = dict(zip(family_gate["family_id"], family_gate["family_q_gate"]))
        reject_map = dict(zip(family_gate["family_id"], family_gate["family_reject_gate"]))
        df["family_q_gate"] = df["family_id"].map(q_map)
        df["family_reject_gate"] = df["family_id"].map(reject_map).fillna(False).astype(bool)

    # Final within-family rejection requires both:
    # (i) within-family BH significance and (ii) family-level gate pass.
    df["reject_within_family"] = (
        df["reject_within_family"].astype(bool) & df["family_reject_gate"].astype(bool)
    )
    if not df.empty:
        family_reject_counts = (
            df.groupby("family_id")["reject_within_family"]
            .sum()
            .astype(int)
            .to_dict()
        )
        df["family_n_reject"] = df["family_id"].map(family_reject_counts).fillna(0).astype(int)
    
    if p_col in df.columns:
        all_p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
        df["q_global"] = fdr_bh(all_p, alpha=alpha, config=config)
        df["reject_global"] = df["q_global"] < alpha
    
    return df



