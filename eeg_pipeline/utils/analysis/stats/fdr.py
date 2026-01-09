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
from typing import Any, Dict, Iterable, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import get_config_value, get_fdr_alpha

if TYPE_CHECKING:
    pass


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


def fdr_bhy(
    pvals: Iterable[float],
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> np.ndarray:
    """Benjamini-Hochberg-Yekutieli FDR correction for dependent tests.
    
    Applies a more conservative correction appropriate when tests
    have positive regression dependency (common in EEG/neuroimaging).
    
    Returns q-values (adjusted p-values).
    """
    if alpha is None:
        alpha = get_fdr_alpha(config)

    p_values_array = np.asarray(list(pvals), dtype=float)
    n_tests = len(p_values_array)
    
    if n_tests == 0:
        return np.array([])
    
    harmonic_sum = sum(1.0 / i for i in range(1, n_tests + 1))
    adjusted_alpha = alpha / harmonic_sum
    
    return fdr_bh(p_values_array, alpha=adjusted_alpha, config=config)


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
        files = [f for f in files if not (f in seen or seen.add(f))]
    
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


def apply_global_fdr(
    stats_dir: Path,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
    include_glob: Union[str, Iterable[str]] = "*.tsv",
    exclude_globs: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply global FDR correction across correlation files within families."""
    files = _collect_tsv_files(stats_dir, include_glob, exclude_globs)
    if not files:
        if logger:
            logger.warning(f"No TSV files found in {stats_dir}")
        return {"n_tests": 0, "n_rejected": 0}
    
    all_p_by_family, file_refs_by_family = _extract_p_values_by_family(files)
    if not all_p_by_family:
        if logger:
            logger.warning("No valid p-values found for FDR correction")
        return {"n_tests": 0, "n_rejected": 0}
    
    summary, file_updates = _apply_fdr_to_families(
        all_p_by_family, file_refs_by_family, alpha, logger
    )
    _update_dataframes_with_fdr_results(file_updates, logger)
    
    return summary


def fdr_bh_mask(
    p_values: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Return boolean mask of significant values after BH-FDR."""
    if p_values is None or len(p_values) == 0:
        return None
    reject_mask, _ = fdr_bh_reject(p_values, alpha, config)
    return reject_mask


def fdr_bh_values(
    p_values: np.ndarray,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (q_values, reject_mask) after BH-FDR."""
    if p_values is None or len(p_values) == 0:
        return None, None
    q_values = fdr_bh(p_values, alpha, config)
    reject_mask, _ = fdr_bh_reject(p_values, alpha, config)
    return q_values, reject_mask


def bh_adjust(
    p_values: np.ndarray,
    config: Optional[Any] = None,
) -> np.ndarray:
    """Simple BH adjustment without config."""
    from eeg_pipeline.utils.config.loader import get_config_value, load_config
    if config is None:
        config = load_config()
    alpha = float(get_config_value(config, "statistics.fdr_alpha", 0.05))
    return fdr_bh(p_values, alpha=alpha)


def select_permutation_p_values_for_fdr(
    results_df: pd.DataFrame,
) -> np.ndarray:
    """Select permutation p-values from results DataFrame for FDR."""
    if "p_perm" in results_df.columns:
        return results_df["p_perm"].values
    return select_raw_p_values_for_fdr(results_df)


def select_raw_p_values_for_fdr(
    results_df: pd.DataFrame,
) -> np.ndarray:
    """Select raw p-values from results DataFrame for FDR."""
    if "p_value" in results_df.columns:
        return results_df["p_value"].values
    if "p" in results_df.columns:
        return results_df["p"].values
    return np.array([])


def select_p_values_for_fdr(
    results_df: pd.DataFrame,
    use_permutation_p: bool = True,
) -> np.ndarray:
    """Select appropriate p-values from results DataFrame for FDR.
    
    Maintains backward compatibility with flag argument.
    Prefer using select_permutation_p_values_for_fdr or 
    select_raw_p_values_for_fdr directly.
    """
    if use_permutation_p:
        return select_permutation_p_values_for_fdr(results_df)
    return select_raw_p_values_for_fdr(results_df)


def filter_significant_predictors(
    results_df: pd.DataFrame,
    alpha: Optional[float] = None,
    config: Optional[Any] = None,
    p_column: str = "p",
    q_column: str = "q_value",
) -> pd.DataFrame:
    """Filter results to significant predictors after FDR."""
    if alpha is None:
        alpha = get_fdr_alpha(config)

    results_df = results_df.copy()
    if q_column not in results_df.columns:
        p_values = results_df[p_column].values
        results_df[q_column] = fdr_bh(p_values, alpha, config)

    significant_mask = results_df[q_column] <= alpha
    return results_df[significant_mask].copy()


def apply_fdr_correction_with_permutation_p_and_save(
    results_df: pd.DataFrame,
    output_path: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Apply FDR correction using permutation p-values and save results."""
    if results_df.empty or "p" not in results_df.columns:
        return
    
    alpha = get_fdr_alpha(config)
    p_values = select_permutation_p_values_for_fdr(results_df)
    
    reject_mask, critical_value = fdr_bh_reject(p_values, alpha=alpha)
    results_df["fdr_reject"] = reject_mask
    results_df["fdr_crit_p"] = critical_value
    
    results_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(results_df)} results to {output_path}")


def apply_fdr_correction_with_raw_p_and_save(
    results_df: pd.DataFrame,
    output_path: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Apply FDR correction using raw p-values and save results."""
    if results_df.empty or "p" not in results_df.columns:
        return
    
    alpha = get_fdr_alpha(config)
    p_values = select_raw_p_values_for_fdr(results_df)
    
    reject_mask, critical_value = fdr_bh_reject(p_values, alpha=alpha)
    results_df["fdr_reject"] = reject_mask
    results_df["fdr_crit_p"] = critical_value
    
    results_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(results_df)} results to {output_path}")


def apply_fdr_correction_and_save(
    results_df: pd.DataFrame,
    output_path: Path,
    config: Any,
    logger: logging.Logger,
    use_permutation_p: bool = True,
) -> None:
    """Apply FDR correction and save results.
    
    Maintains backward compatibility with flag argument.
    Prefer using apply_fdr_correction_with_permutation_p_and_save or
    apply_fdr_correction_with_raw_p_and_save directly.
    """
    if use_permutation_p:
        apply_fdr_correction_with_permutation_p_and_save(
            results_df, output_path, config, logger
        )
    else:
        apply_fdr_correction_with_raw_p_and_save(
            results_df, output_path, config, logger
        )


def get_pvalue_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Extract permutation and raw p-value series."""
    permutation_p_series = pd.Series(index=df.index, dtype=float)
    raw_p_series = pd.Series(index=df.index, dtype=float)
    
    permutation_columns = ["p_partial_perm", "p_partial_temp_perm", "p_perm"]
    for column in permutation_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        valid_mask = values.notna()
        permutation_p_series.loc[valid_mask] = values.loc[valid_mask]
    
    raw_columns = ["p", "p_value", "p_partial", "p_partial_temp"]
    for column in raw_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        valid_mask = values.notna()
        raw_p_series.loc[valid_mask] = values.loc[valid_mask]
    
    return permutation_p_series, raw_p_series


def extract_pvalue_from_dataframe(
    df: pd.DataFrame,
    row_index: int,
) -> Tuple[float, str]:
    """Extract p-value for a row with column name."""
    if row_index < 0 or row_index >= len(df):
        return np.nan, ""
    
    p_value_columns = [
        "p_partial_perm",
        "p_partial_temp_perm",
        "p_perm",
        "p",
        "p_value",
        "p_partial",
    ]
    
    for column in p_value_columns:
        if column not in df.columns:
            continue
        
        p_value = pd.to_numeric(df.iloc[row_index][column], errors="coerce")
        if pd.notna(p_value):
            return float(p_value), column
    
    return np.nan, ""


def should_apply_fisher_transform(prefix: str) -> bool:
    """Check if Fisher transform should be applied for measure."""
    measure_name = prefix.split("_", 1)[0].lower()
    measures_requiring_fisher = ("aec", "aec_orth", "corr", "pearsonr")
    return measure_name in measures_requiring_fisher


def get_cluster_correction_config(
    heatmap_config: Dict[str, Any],
    config: Any,
    alpha: float,
    default_rng_seed: Optional[int] = None,
) -> Tuple[float, int, np.random.Generator, int]:
    """Get cluster correction configuration."""
    if default_rng_seed is None:
        default_rng_seed = int(get_config_value(config, "project.random_state", 42))
    
    cluster_config = (
        config.get("behavior_analysis.cluster_correction", {}) if config else {}
    )
    
    cluster_alpha = float(
        heatmap_config.get("cluster_alpha", cluster_config.get("alpha", alpha))
    )
    
    from eeg_pipeline.utils.config.loader import get_config_value
    
    default_n_permutations = int(
        get_config_value(
            config,
            "behavior_analysis.cluster_correction.default_n_permutations",
            get_config_value(config, "statistics.cluster_n_perm", 100),
        )
    )
    n_permutations = int(
        heatmap_config.get(
            "n_cluster_perm",
            cluster_config.get("n_permutations", default_n_permutations),
        )
    )
    
    rng_seed = int(
        heatmap_config.get(
            "cluster_rng_seed",
            cluster_config.get("rng_seed", default_rng_seed),
        )
    )
    
    random_generator = np.random.default_rng(rng_seed)
    return cluster_alpha, n_permutations, random_generator, rng_seed


def compute_fdr_rejections_for_heatmap(
    p_value_matrix: np.ndarray,
    n_nodes: int,
    config: Any,
) -> Tuple[Dict[Tuple[int, int], bool], float]:
    """Compute FDR rejections for heatmap."""
    upper_triangle_indices = np.triu_indices(n_nodes, k=1)
    upper_p_values = p_value_matrix[upper_triangle_indices]
    valid_mask = np.isfinite(upper_p_values)
    valid_p_values = upper_p_values[valid_mask]
    
    alpha = get_fdr_alpha(config)
    reject_mask, critical_value = fdr_bh_reject(valid_p_values, alpha=alpha)
    
    valid_indices = np.where(valid_mask)[0]
    node_pairs = [
        (upper_triangle_indices[0][idx], upper_triangle_indices[1][idx])
        for idx in valid_indices
    ]
    rejection_map = {
        pair: bool(reject_mask[k]) for k, pair in enumerate(node_pairs)
    }
    
    if np.any(reject_mask):
        critical_p_value = float(np.max(valid_p_values[reject_mask]))
    else:
        critical_p_value = np.nan
    
    return rejection_map, critical_p_value


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
    
    # Ensure family_id column exists
    if family_col not in df.columns:
        df["family_id"] = "default"
        family_col = "family_id"
    else:
        df["family_id"] = df[family_col]
    
    # Infer family_kind if not present
    if "family_kind" not in df.columns:
        if "feature_type" in df.columns:
            df["family_kind"] = "feature_type"
        elif "analysis_type" in df.columns:
            df["family_kind"] = "analysis_type"
        else:
            df["family_kind"] = "inferred"
    
    # Initialize output columns
    df["q_within_family"] = np.nan
    df["reject_within_family"] = False
    df["family_n_tests"] = 0
    df["family_n_reject"] = 0
    
    # Apply FDR within each family
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
    
    # Apply global FDR
    if p_col in df.columns:
        all_p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
        df["q_global"] = fdr_bh(all_p, alpha=alpha, config=config)
        df["reject_global"] = df["q_global"] < alpha
    
    return df


def compute_effective_n(n_tests: int, correlation_matrix: Optional[np.ndarray] = None) -> int:
    """Compute effective number of independent tests.
    
    Uses eigenvalue decomposition of correlation matrix if provided,
    following the Li & Ji (2005) method.
    
    Parameters
    ----------
    n_tests : int
        Total number of tests
    correlation_matrix : np.ndarray, optional
        Correlation matrix between tests
        
    Returns
    -------
    int
        Effective number of independent tests
    """
    if correlation_matrix is None:
        return n_tests
    
    try:
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        positive_eigenvalues = eigenvalues[eigenvalues > 0]
        
        eigenvalue_variance = np.var(positive_eigenvalues)
        effective_n = 1 + (n_tests - 1) * (1 - eigenvalue_variance / n_tests)
        return max(1, int(np.ceil(effective_n)))
    except (np.linalg.LinAlgError, ValueError):
        return n_tests


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
    from .correlation import compute_correlation
    
    n_nodes = len(node_to_index)
    correlation_matrix = np.full((n_nodes, n_nodes), np.nan)
    p_value_matrix = np.full((n_nodes, n_nodes), np.nan)
    
    for column in prefix_columns:
        parts = column.split(prefix + "_", 1)
        if len(parts) < 2:
            continue
        pair_string = parts[-1]
        
        for separator in ["--", "-", "_"]:
            if separator in pair_string:
                node_names = pair_string.split(separator)
                if len(node_names) != 2:
                    break
                
                node_a, node_b = node_names
                if node_a not in node_to_index or node_b not in node_to_index:
                    break
                
                node_index_a = node_to_index[node_a]
                node_index_b = node_to_index[node_b]
                
                edge_values = pd.to_numeric(connectivity_df[column], errors="coerce")
                valid_mask = edge_values.notna() & target_values.notna()
                
                if valid_mask.sum() >= min_samples:
                    correlation_method = "spearman" if use_spearman else "pearson"
                    correlation_value, p_value = compute_correlation(
                        edge_values[valid_mask].values,
                        target_values[valid_mask].values,
                        correlation_method,
                    )
                    correlation_matrix[node_index_a, node_index_b] = correlation_value
                    correlation_matrix[node_index_b, node_index_a] = correlation_value
                    p_value_matrix[node_index_a, node_index_b] = p_value
                    p_value_matrix[node_index_b, node_index_a] = p_value
                break
    
    return correlation_matrix, p_value_matrix
