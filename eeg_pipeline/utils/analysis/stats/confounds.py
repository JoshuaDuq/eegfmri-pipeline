"""
Confound Auditing (Subject-Level)
=================================

Audits whether signal-quality (QC) metrics are associated with:
- rating
- temperature

If strong QC→target associations exist, downstream analyses can optionally
include selected QC metrics as covariates.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats.base import get_config_value, get_fdr_alpha
from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh


def _matches_any_pattern(name: str, patterns: List[str]) -> bool:
    """Check if name matches any of the provided regex patterns."""
    for pattern in patterns:
        try:
            if re.search(pattern, name):
                return True
        except re.error:
            if pattern in name:
                return True
    return False


def select_qc_columns(df: pd.DataFrame, config: Any) -> List[str]:
    """Select QC columns from dataframe based on config patterns."""
    default_patterns = [
        r"^quality_.*_global_",
        r"^quality_.*_ch_",
    ]
    patterns = get_config_value(
        config,
        "behavior_analysis.confounds.qc_column_patterns",
        default_patterns,
    )
    if isinstance(patterns, (list, tuple)):
        patterns = [str(p) for p in patterns]
    else:
        patterns = [str(patterns)]
    
    excluded_columns = {"rating", "temperature", "pain_residual"}
    candidates = []
    for col in df.columns:
        column_name = str(col)
        if column_name in excluded_columns:
            continue
        if _matches_any_pattern(column_name, patterns):
            candidates.append(column_name)
    return candidates


def _compute_correlation_record(
    qc_values: np.ndarray,
    target_values: np.ndarray,
    qc_name: str,
    target_name: str,
    method: str,
    robust_method: Optional[str],
    min_samples: int,
) -> Optional[Dict[str, Any]]:
    """Compute correlation between QC and target, return record if valid."""
    correlation, p_value, n_valid = safe_correlation(
        qc_values,
        target_values,
        method=method,
        min_samples=min_samples,
        robust_method=robust_method,
    )
    if not np.isfinite(correlation) or not np.isfinite(p_value):
        return None
    
    return {
        "qc_metric": qc_name,
        "target": target_name,
        "r": float(correlation),
        "p": float(p_value),
        "n": int(n_valid),
        "method": method,
        "robust_method": robust_method,
    }


def _apply_fdr_correction_by_target(audit_df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Apply FDR correction within each target group."""
    audit_df = audit_df.copy()
    audit_df["q"] = np.nan
    fdr_alpha = get_fdr_alpha(config)
    
    for target in audit_df["target"].unique():
        target_mask = audit_df["target"] == target
        p_values = pd.to_numeric(
            audit_df.loc[target_mask, "p"],
            errors="coerce"
        ).to_numpy()
        audit_df.loc[target_mask, "q"] = fdr_bh(
            p_values,
            alpha=fdr_alpha,
            config=config,
        )
    return audit_df


def _add_fdr_aliases(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Add convenience aliases for integration with global FDR tooling."""
    audit_df = audit_df.copy()
    audit_df["p_primary"] = audit_df["p"]
    audit_df["p_raw"] = audit_df["p"]
    audit_df["p_kind_primary"] = "p"
    audit_df["p_primary_source"] = "raw"
    return audit_df


def audit_qc_confounds(
    trial_df: pd.DataFrame,
    *,
    config: Any,
    targets: Optional[List[str]] = None,
    method: str = "spearman",
    robust_method: Optional[str] = None,
    min_samples: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audit QC metrics for associations with target variables.
    
    Returns
    -------
    audit_df : pd.DataFrame
        DataFrame with correlation results (r, p, q) for each QC-target pair
    metadata : Dict[str, Any]
        Metadata about the audit process
    """
    default_targets = ["rating", "temperature"]
    targets = targets or default_targets
    qc_columns = select_qc_columns(trial_df, config)

    metadata: Dict[str, Any] = {
        "n_qc_candidates": int(len(qc_columns)),
        "targets": list(targets),
    }
    
    if not qc_columns:
        return pd.DataFrame(), {**metadata, "status": "empty"}

    records: List[Dict[str, Any]] = []
    for target in targets:
        if target not in trial_df.columns:
            continue
        
        target_values = pd.to_numeric(
            trial_df[target],
            errors="coerce"
        ).to_numpy()
        
        for qc_column in qc_columns:
            qc_values = pd.to_numeric(
                trial_df[qc_column],
                errors="coerce"
            ).to_numpy()
            
            record = _compute_correlation_record(
                qc_values,
                target_values,
                qc_column,
                target,
                method,
                robust_method,
                min_samples,
            )
            if record is not None:
                records.append(record)

    if not records:
        return pd.DataFrame(), {**metadata, "status": "no_valid_tests"}

    audit_df = pd.DataFrame(records)
    audit_df = _apply_fdr_correction_by_target(audit_df, config)
    audit_df = _add_fdr_aliases(audit_df)

    metadata["status"] = "ok"
    metadata["n_tests"] = int(len(audit_df))
    return audit_df, metadata


def select_significant_qc_covariates(
    audit_df: pd.DataFrame,
    *,
    config: Any,
    alpha: float = 0.05,
    max_covariates: int = 3,
    prefer_target: str = "rating",
) -> List[str]:
    """Select QC metrics to use as covariates based on FDR q-values.
    
    Prioritizes metrics confounded with the preferred target, then by
    absolute correlation strength.
    """
    if audit_df is None or audit_df.empty:
        return []
    
    filtered_df = audit_df.copy()
    filtered_df["q"] = pd.to_numeric(
        filtered_df.get("q", np.nan),
        errors="coerce"
    )
    filtered_df = filtered_df[
        np.isfinite(filtered_df["q"]) & (filtered_df["q"] < float(alpha))
    ]
    
    if filtered_df.empty:
        return []
    
    filtered_df["abs_r"] = pd.to_numeric(
        filtered_df.get("r", np.nan),
        errors="coerce"
    ).abs()
    filtered_df["is_prefer"] = (
        filtered_df["target"].astype(str) == str(prefer_target)
    )
    filtered_df = filtered_df.sort_values(
        ["is_prefer", "abs_r"],
        ascending=[False, False],
    )
    
    selected_metrics = filtered_df["qc_metric"].astype(str).tolist()
    unique_metrics = list(dict.fromkeys(selected_metrics))
    return unique_metrics[:int(max_covariates)]


__all__ = [
    "audit_qc_confounds",
    "select_qc_columns",
    "select_significant_qc_covariates",
]

