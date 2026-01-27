"""
FDR Correction
==============

Benjamini-Hochberg false discovery rate correction functions and utilities.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import get_config_value, get_fdr_alpha


def select_p_column_for_fdr(df: pd.DataFrame) -> Optional[str]:
    """Select appropriate p-value column for FDR correction.
    
    Priority order:
    1. p_primary_perm (if has valid values)
    2. p_primary (if has valid values)
    3. p_psi, sobel_p, fixed_p, p_raw, p, p_value (if exists)
    """
    priority_columns = [
        "p_primary_perm",
        "p_primary",
    ]
    
    for column in priority_columns:
        if column not in df.columns:
            continue
        p_values = pd.to_numeric(df[column], errors="coerce")
        if p_values.notna().any():
            return column
    
    fallback_columns = ["p_psi", "sobel_p", "fixed_p", "p_raw", "p", "p_value"]
    for column in fallback_columns:
        if column in df.columns:
            return column
    
    return None


def infer_fdr_family(file_path: Path, df: pd.DataFrame) -> str:
    """Infer FDR family identifier from file path and dataframe."""
    file_name = file_path.stem
    
    if "test_family" in df.columns and df["test_family"].notna().any():
        try:
            family_value = df["test_family"].dropna().iloc[0]
            return str(family_value)
        except (IndexError, KeyError):
            pass

    correlation_match = re.match(r"corr_stats_(.+)_vs_(.+)", file_name)
    if correlation_match:
        feature_part, target_part = correlation_match.groups()
        return f"target:{target_part}|features:{feature_part}"

    if "target" in df.columns and df["target"].notna().any():
        try:
            target_value = str(df["target"].dropna().iloc[0])
            feature_type = None
            if "feature_type" in df.columns and df["feature_type"].notna().any():
                feature_type = str(df["feature_type"].dropna().iloc[0])
            
            if feature_type:
                return f"target:{target_value}|features:{feature_type}"
            return f"target:{target_value}"
        except (IndexError, KeyError):
            pass

    return file_name


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


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply Benjamini-Hochberg FDR correction.
    
    Returns:
        q_values: Adjusted p-values
        reject_mask: Boolean mask of rejected hypotheses
        critical_p: Critical p-value threshold
    """
    q_values = fdr_bh(p_values, alpha=alpha)
    reject_mask, critical_p = fdr_bh_reject(p_values, alpha)
    return q_values, reject_mask, critical_p


def fdr_bh_values(
    p_values: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q_values, reject_mask) after BH-FDR."""
    q_values = fdr_bh(p_values, alpha, config)
    reject_mask, _ = fdr_bh_reject(p_values, alpha, config)
    return q_values, reject_mask


def bh_adjust(
    p_values: np.ndarray,
    config: Optional[Any] = None,
) -> np.ndarray:
    """Simple BH adjustment without explicit alpha parameter."""
    alpha = get_fdr_alpha(config)
    return fdr_bh(p_values, alpha=alpha, config=config)


def _collect_tsv_files(
    stats_dir: Path,
    include_glob: Union[str, Iterable[str]],
    exclude_globs: Optional[Iterable[str]],
) -> list[Path]:
    """Collect TSV files matching include/exclude patterns."""
    stats_dir = Path(stats_dir)
    
    if isinstance(include_glob, str):
        files = list(stats_dir.rglob(include_glob))
    else:
        files = []
        for pattern in include_glob:
            files.extend(list(stats_dir.rglob(pattern)))
        seen = set()
        files = [f for f in files if f not in seen and not seen.add(f)]
    
    if exclude_globs:
        excluded = set()
        for pattern in exclude_globs:
            excluded.update(stats_dir.glob(pattern))
        files = [f for f in files if f not in excluded]
    
    return files


def _extract_p_values_by_family(
    files: list[Path],
) -> Tuple[Dict[str, list[float]], Dict[str, list[Tuple[Path, int, str]]]]:
    """Extract p-values grouped by family from TSV files."""
    from eeg_pipeline.infra.tsv import read_tsv
    
    all_p_by_family: Dict[str, list[float]] = defaultdict(list)
    file_refs_by_family: Dict[str, list[Tuple[Path, int, str]]] = defaultdict(list)

    for file_path in files:
        df = read_tsv(file_path)
        if df is None or df.empty:
            continue

        p_column = select_p_column_for_fdr(df)
        if p_column is None:
            continue

        family = infer_fdr_family(file_path, df)
        
        p_series = pd.to_numeric(df[p_column], errors="coerce")
        valid_mask = p_series.notna()
        valid_indices = np.where(valid_mask)[0]
        
        for index in valid_indices:
            p_value = float(p_series.iloc[index])
            all_p_by_family[family].append(p_value)
            file_refs_by_family[family].append((file_path, index, p_column))
    
    return all_p_by_family, file_refs_by_family


def _apply_fdr_to_families(
    all_p_by_family: Dict[str, list[float]],
    file_refs_by_family: Dict[str, list[Tuple[Path, int, str]]],
    alpha: float,
    logger: Optional[logging.Logger],
) -> Tuple[Dict[str, Any], Dict[Path, list[Tuple[int, float, bool, str, str]]]]:
    """Apply FDR correction to each family and prepare file updates."""
    summary = {"n_tests": 0, "n_rejected": 0, "alpha": alpha, "families": {}}
    file_updates: Dict[Path, list[Tuple[int, float, bool, str, str]]] = defaultdict(list)

    for family, p_list in all_p_by_family.items():
        p_array = np.array(p_list)
        q_array, reject_mask, critical_p = fdr_correction(p_array, alpha)

        n_tests = len(p_array)
        n_rejected = int(reject_mask.sum())
        summary["n_tests"] += n_tests
        summary["n_rejected"] += n_rejected
        
        p_kind = None
        if file_refs_by_family[family]:
            p_kind = str(file_refs_by_family[family][0][2])
        
        summary["families"][family] = {
            "n_tests": n_tests,
            "n_rejected": n_rejected,
            "critical_p": float(critical_p) if np.isfinite(critical_p) else np.nan,
            "p_kind": p_kind,
        }

        if logger:
            logger.info(
                f"Global FDR [{family}]: {n_rejected}/{n_tests} rejected at alpha={alpha}"
            )

        for (file_path, index, p_column), q_value, rejected in zip(
            file_refs_by_family[family], q_array, reject_mask
        ):
            file_updates[file_path].append((index, q_value, rejected, family, p_column))
    
    return summary, file_updates


def _update_dataframes_with_fdr_results(
    file_updates: Dict[Path, list[Tuple[int, float, bool, str, str]]],
    logger: Optional[logging.Logger],
) -> None:
    """Update dataframes with FDR correction results."""
    from eeg_pipeline.infra.tsv import read_tsv, write_tsv
    
    for file_path, updates in file_updates.items():
        try:
            df = read_tsv(file_path)
            if df is None:
                continue
            
            _ensure_fdr_columns_exist(df)
            
            for index, q_value, rejected, family, p_column in updates:
                if index < len(df):
                    df.loc[index, "q_global"] = q_value
                    df.loc[index, "fdr_reject"] = bool(rejected)
                    df.loc[index, "fdr_family"] = family
                    df.loc[index, "fdr_p_kind"] = p_column
            
            write_tsv(df, file_path)
        except (IOError, ValueError, KeyError) as error:
            if logger:
                logger.warning(f"Failed to update {file_path.name}: {error}")


def _ensure_fdr_columns_exist(df: pd.DataFrame) -> None:
    """Ensure required FDR columns exist in dataframe."""
    if "q_global" not in df.columns:
        df["q_global"] = np.nan
    if "fdr_reject" not in df.columns:
        df["fdr_reject"] = False
    if "fdr_family" not in df.columns:
        df["fdr_family"] = pd.Series([None] * len(df), dtype="object")
    else:
        df["fdr_family"] = df["fdr_family"].astype("object")
    if "fdr_p_kind" not in df.columns:
        df["fdr_p_kind"] = pd.Series([None] * len(df), dtype="object")
    else:
        df["fdr_p_kind"] = df["fdr_p_kind"].astype("object")


def hierarchical_fdr(
    df: pd.DataFrame,
    p_col: str = "p_value",
    family_col: str = "family_id",
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Apply hierarchical FDR correction with explicit family structure.
    
    Corrects p-values within families first, then globally. Adds columns
    documenting the family structure for auditability.
    
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
    
    if p_col in df.columns:
        all_p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
        df["q_global"] = fdr_bh(all_p, alpha=alpha, config=config)
        df["reject_global"] = df["q_global"] < alpha
    
    return df




