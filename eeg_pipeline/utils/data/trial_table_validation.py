"""
Trial Table Validation (Subject-Level)
=====================================

Validates the canonical per-trial analysis table (`trials*.parquet/tsv`) and
emits a lightweight "data contract" report. This is intentionally non-gating:
it reports issues and summaries but does not drop trials/features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        pass
    return default


@dataclass
class TrialTableValidationResult:
    summary_df: pd.DataFrame
    report: Dict[str, Any]


def _summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    n = int(len(df))
    for col in df.columns:
        s = df[col]
        frac = float(s.isna().mean()) if n else np.nan
        recs.append(
            {
                "column": str(col),
                "dtype": str(s.dtype),
                "n_rows": n,
                "n_missing": int(s.isna().sum()),
                "missing_frac": frac,
                "n_unique_non_nan": int(pd.Series(s).nunique(dropna=True)),
            }
        )
    return pd.DataFrame(recs).sort_values(["missing_frac", "n_unique_non_nan"], ascending=[False, True])


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
        report["checks"]["epoch"] = {
            "n_non_nan": int(epoch.notna().sum()),
            "n_duplicates": int(epoch.duplicated().sum()),
            "is_monotonic_increasing": bool(epoch.is_monotonic_increasing) if epoch.notna().any() else False,
        }
        if report["checks"]["epoch"]["n_duplicates"] > 0:
            warnings.append("Epoch column contains duplicates")
        if epoch.notna().any() and not report["checks"]["epoch"]["is_monotonic_increasing"]:
            warnings.append("Epoch column is not monotonic increasing (table may be unsorted)")

    # Rating statistics (no range validation)
    if "rating" in df.columns:
        r = pd.to_numeric(df["rating"], errors="coerce")
        ok = r.notna()
        report["checks"]["rating"] = {
            "n_non_nan": int(ok.sum()),
            "min": float(r.min()) if ok.any() else np.nan,
            "max": float(r.max()) if ok.any() else np.nan,
        }

    # Temperature statistics (no range validation)
    if "temperature" in df.columns:
        t = pd.to_numeric(df["temperature"], errors="coerce")
        ok = t.notna()
        report["checks"]["temperature"] = {
            "n_non_nan": int(ok.sum()),
            "min": float(t.min()) if ok.any() else np.nan,
            "max": float(t.max()) if ok.any() else np.nan,
        }

    # pain_binary sanity
    if "pain_binary" in df.columns:
        pb = pd.to_numeric(df["pain_binary"], errors="coerce")
        ok = pb.notna()
        unique = sorted(pd.Series(pb[ok]).unique().tolist())
        non_binary = int((~pb.isin([0, 1]) & ok).sum())
        report["checks"]["pain_binary"] = {
            "n_non_nan": int(ok.sum()),
            "unique_values": unique,
            "n_non_binary": non_binary,
        }
        if non_binary > 0:
            warnings.append(f"pain_binary contains {non_binary} non-(0/1) values")

    # Trial index within group should be non-negative and (usually) restart within run/block.
    if "trial_index_within_group" in df.columns:
        ti = pd.to_numeric(df["trial_index_within_group"], errors="coerce")
        ok = ti.notna()
        neg = int((ti < 0).sum())
        report["checks"]["trial_index_within_group"] = {
            "n_non_nan": int(ok.sum()),
            "min": float(ti.min()) if ok.any() else np.nan,
            "max": float(ti.max()) if ok.any() else np.nan,
            "n_negative": neg,
        }
        if neg > 0:
            warnings.append("trial_index_within_group has negative values")

        group_cols = [c for c in ["run", "block"] if c in df.columns]
        if group_cols and "epoch" in df.columns:
            bad_groups = 0
            for _g, gdf in df.groupby(group_cols, dropna=False, sort=False):
                x = pd.to_numeric(gdf["trial_index_within_group"], errors="coerce")
                if x.notna().any() and not (x.dropna().astype(int).values == np.arange(int(x.notna().sum()))).all():
                    bad_groups += 1
            report["checks"]["trial_index_within_group"]["n_groups_mismatch"] = int(bad_groups)
            if bad_groups > 0:
                warnings.append("trial_index_within_group does not look like 0..N-1 within some groups")

    # Feature column inventory (non-gating)
    prefixes = (
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
    feature_cols = [c for c in df.columns if str(c).startswith(prefixes)]
    report["checks"]["features"] = {
        "n_feature_columns": int(len(feature_cols)),
        "n_total_columns": int(df.shape[1]),
        "has_any_features": bool(len(feature_cols) > 0),
    }
    if len(feature_cols) == 0:
        warnings.append("No feature columns detected (expected prefixes not found)")

    # Constant columns (non-gating)
    constant_cols = []
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            continue
        x = pd.to_numeric(s, errors="coerce")
        x = x[np.isfinite(x)]
        if x.size >= 2 and float(np.nanstd(x, ddof=1)) <= 1e-12:
            constant_cols.append(str(c))
    if constant_cols:
        report["checks"]["constant_columns"] = constant_cols[:200]
        warnings.append(f"{len(constant_cols)} numeric columns look constant (std≈0)")

    summary_df = _summarize_missingness(df)
    high_missing = float(_get(config, "behavior_analysis.trial_table.validate.high_missing_frac", 0.5))
    n_high_missing = int((summary_df["missing_frac"] > high_missing).sum())
    report["checks"]["missingness"] = {
        "high_missing_frac_threshold": high_missing,
        "n_columns_high_missing": n_high_missing,
    }
    if n_high_missing > 0:
        warnings.append(f"{n_high_missing} columns have missing_frac > {high_missing}")

    report["warnings"] = warnings
    report["status"] = "ok" if not missing_required else "missing_required_columns"
    return TrialTableValidationResult(summary_df=summary_df, report=report)


__all__ = ["TrialTableValidationResult", "validate_trial_table"]

