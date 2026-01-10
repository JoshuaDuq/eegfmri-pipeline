"""
Trial Table Validation (Subject-Level)
=====================================

Validates the canonical per-trial analysis table (`trials*.parquet/tsv`) and
emits a lightweight "data contract" report. This is intentionally non-gating:
it reports issues and summaries but does not drop trials/features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value


@dataclass
class TrialTableValidationResult:
    summary_df: pd.DataFrame
    report: Dict[str, Any]


def _compute_numeric_statistics(series: pd.Series) -> Dict[str, Any]:
    """Compute basic statistics for a numeric series."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    has_valid_values = numeric_series.notna()
    
    if not has_valid_values.any():
        return {
            "n_non_nan": 0,
            "min": np.nan,
            "max": np.nan,
        }
    
    return {
        "n_non_nan": int(has_valid_values.sum()),
        "min": float(numeric_series.min()),
        "max": float(numeric_series.max()),
    }


def _summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and uniqueness for each column."""
    records = []
    n_rows = len(df)
    
    for col in df.columns:
        series = df[col]
        n_missing = series.isna().sum()
        missing_frac = float(n_missing / n_rows) if n_rows > 0 else np.nan
        n_unique = series.nunique(dropna=True)
        
        records.append(
            {
                "column": str(col),
                "dtype": str(series.dtype),
                "n_rows": n_rows,
                "n_missing": int(n_missing),
                "missing_frac": missing_frac,
                "n_unique_non_nan": int(n_unique),
            }
        )
    
    sort_by_missing_frac = False
    sort_by_uniqueness = True
    return pd.DataFrame(records).sort_values(
        ["missing_frac", "n_unique_non_nan"],
        ascending=[sort_by_missing_frac, sort_by_uniqueness]
    )


def validate_trial_table(
    df: pd.DataFrame,
    *,
    config: Optional[Any] = None,
) -> TrialTableValidationResult:
    """
    Validate a subject's canonical trial table.

    This function returns:
      - `summary_df`: per-column missingness/uniqueness summary
      - `report`: a structured JSON-serializable dict with warnings and key checks
    """
    df = df.copy()
    warnings: List[str] = []

    required_cols = ["subject", "task", "epoch", "rating"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        warnings.append(f"Missing required columns: {missing_required}")

    report: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "required_columns": required_cols,
        "missing_required_columns": missing_required,
        "warnings": warnings,
        "checks": {},
    }

    # Epoch monotonicity / duplicates
    if "epoch" in df.columns:
        epoch = pd.to_numeric(df["epoch"], errors="coerce")
        has_valid_epochs = epoch.notna()
        n_valid_epochs = int(has_valid_epochs.sum())
        n_duplicates = int(epoch.duplicated().sum())
        is_monotonic = (
            bool(epoch.is_monotonic_increasing)
            if has_valid_epochs.any()
            else False
        )
        
        report["checks"]["epoch"] = {
            "n_non_nan": n_valid_epochs,
            "n_duplicates": n_duplicates,
            "is_monotonic_increasing": is_monotonic,
        }
        
        if n_duplicates > 0:
            warnings.append("Epoch column contains duplicates")
        if has_valid_epochs.any() and not is_monotonic:
            warnings.append("Epoch column is not monotonic increasing (table may be unsorted)")

    # Rating statistics (no range validation)
    if "rating" in df.columns:
        report["checks"]["rating"] = _compute_numeric_statistics(df["rating"])

    # Temperature statistics (no range validation)
    if "temperature" in df.columns:
        report["checks"]["temperature"] = _compute_numeric_statistics(df["temperature"])

    # pain_binary validation
    if "pain_binary" in df.columns:
        pain_binary = pd.to_numeric(df["pain_binary"], errors="coerce")
        has_valid_values = pain_binary.notna()
        valid_values = pain_binary[has_valid_values]
        unique_values = sorted(valid_values.unique().tolist())
        
        is_binary = pain_binary.isin([0, 1])
        non_binary_mask = ~is_binary & has_valid_values
        n_non_binary = int(non_binary_mask.sum())
        
        report["checks"]["pain_binary"] = {
            "n_non_nan": int(has_valid_values.sum()),
            "unique_values": unique_values,
            "n_non_binary": n_non_binary,
        }
        if n_non_binary > 0:
            warnings.append(f"pain_binary contains {n_non_binary} non-(0/1) values")

    # Trial index within group should be non-negative and (usually) restart within run/block.
    if "trial_index_within_group" in df.columns:
        trial_index = pd.to_numeric(df["trial_index_within_group"], errors="coerce")
        has_valid_values = trial_index.notna()
        n_negative = int((trial_index < 0).sum())
        
        report["checks"]["trial_index_within_group"] = {
            "n_non_nan": int(has_valid_values.sum()),
            "min": float(trial_index.min()) if has_valid_values.any() else np.nan,
            "max": float(trial_index.max()) if has_valid_values.any() else np.nan,
            "n_negative": n_negative,
        }
        if n_negative > 0:
            warnings.append("trial_index_within_group has negative values")

        group_cols = [c for c in ["run", "block"] if c in df.columns]
        if group_cols and "epoch" in df.columns:
            n_groups_mismatch = 0
            for _group_key, group_df in df.groupby(group_cols, dropna=False, sort=False):
                trial_indices = pd.to_numeric(
                    group_df["trial_index_within_group"],
                    errors="coerce"
                )
                has_valid_indices = trial_indices.notna()
                
                if not has_valid_indices.any():
                    continue
                
                valid_indices = trial_indices[has_valid_indices].astype(int).values
                n_valid = len(valid_indices)
                expected_indices = np.arange(n_valid)
                is_sequential = np.array_equal(valid_indices, expected_indices)
                
                if not is_sequential:
                    n_groups_mismatch += 1
            
            report["checks"]["trial_index_within_group"]["n_groups_mismatch"] = n_groups_mismatch
            if n_groups_mismatch > 0:
                warnings.append(
                    f"trial_index_within_group does not look like 0..N-1 within {n_groups_mismatch} groups"
                )

    # Feature column inventory (non-gating)
    feature_prefixes = (
        "power_",
        "conn_",
        "aperiodic_",
        "erp_",
        "itpc_",
        "pac_",
        "comp_",
        "bursts_",
        "quality_",
        "erds_",
        "spectral_",
        "ratios_",
        "asymmetry_",
        "summary_",
    )
    feature_columns = [
        col for col in df.columns
        if str(col).startswith(feature_prefixes)
    ]
    n_feature_columns = len(feature_columns)
    
    report["checks"]["features"] = {
        "n_feature_columns": n_feature_columns,
        "n_total_columns": int(df.shape[1]),
        "has_any_features": n_feature_columns > 0,
    }
    if n_feature_columns == 0:
        warnings.append("No feature columns detected (expected prefixes not found)")

    # Constant columns (non-gating)
    constant_columns = []
    numeric_std_threshold = 1e-12
    min_values_for_std_check = 2
    
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            continue
        
        numeric_values = pd.to_numeric(series, errors="coerce")
        finite_values = numeric_values[np.isfinite(numeric_values)]
        
        if finite_values.size < min_values_for_std_check:
            continue
        
        std_value = np.nanstd(finite_values, ddof=1)
        is_effectively_constant = float(std_value) <= numeric_std_threshold
        
        if is_effectively_constant:
            constant_columns.append(str(col))
    
    if constant_columns:
        max_reported_constant_cols = 200
        report["checks"]["constant_columns"] = constant_columns[:max_reported_constant_cols]
        warnings.append(
            f"{len(constant_columns)} numeric columns look constant (std≤{numeric_std_threshold})"
        )

    summary_df = _summarize_missingness(df)
    default_high_missing_threshold = 0.5
    config_key = "behavior_analysis.trial_table.validate.high_missing_frac"
    high_missing_threshold = float(
        get_config_value(config, config_key, default_high_missing_threshold)
    )
    
    columns_above_threshold = summary_df["missing_frac"] > high_missing_threshold
    n_high_missing = int(columns_above_threshold.sum())
    
    report["checks"]["missingness"] = {
        "high_missing_frac_threshold": high_missing_threshold,
        "n_columns_high_missing": n_high_missing,
    }
    if n_high_missing > 0:
        warnings.append(
            f"{n_high_missing} columns have missing_frac > {high_missing_threshold}"
        )

    report["warnings"] = warnings
    has_missing_required = len(missing_required) > 0
    report["status"] = "missing_required_columns" if has_missing_required else "ok"
    return TrialTableValidationResult(summary_df=summary_df, report=report)


__all__ = ["TrialTableValidationResult", "validate_trial_table"]

