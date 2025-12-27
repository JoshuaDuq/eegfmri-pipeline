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

from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh


def _get(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        pass
    return default


def _match_any(name: str, patterns: List[str]) -> bool:
    for pat in patterns:
        try:
            if re.search(pat, name):
                return True
        except re.error:
            if pat in name:
                return True
    return False


def select_qc_columns(df: pd.DataFrame, config: Any) -> List[str]:
    patterns = _get(
        config,
        "behavior_analysis.confounds.qc_column_patterns",
        [
            r"^quality_.*_global_",
            r"^quality_.*_ch_",
        ],
    )
    patterns = [str(p) for p in patterns] if isinstance(patterns, (list, tuple)) else [str(patterns)]
    candidates = []
    for col in df.columns:
        name = str(col)
        if name in {"rating", "temperature", "pain_residual"}:
            continue
        if _match_any(name, patterns):
            candidates.append(name)
    return candidates


def audit_qc_confounds(
    trial_df: pd.DataFrame,
    *,
    config: Any,
    targets: Optional[List[str]] = None,
    method: str = "spearman",
    robust_method: Optional[str] = None,
    min_samples: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (audit_df, metadata)."""
    targets = targets or ["rating", "temperature"]
    qc_cols = select_qc_columns(trial_df, config)

    meta: Dict[str, Any] = {
        "n_qc_candidates": int(len(qc_cols)),
        "targets": list(targets),
    }
    if not qc_cols:
        return pd.DataFrame(), {**meta, "status": "empty"}

    records: List[Dict[str, Any]] = []
    for target in targets:
        if target not in trial_df.columns:
            continue
        y = pd.to_numeric(trial_df[target], errors="coerce").to_numpy()
        for qc in qc_cols:
            x = pd.to_numeric(trial_df[qc], errors="coerce").to_numpy()
            r, p, n = safe_correlation(x, y, method=method, min_samples=min_samples, robust_method=robust_method)
            if not np.isfinite(r) or not np.isfinite(p):
                continue
            records.append(
                {
                    "qc_metric": qc,
                    "target": target,
                    "r": float(r),
                    "p": float(p),
                    "n": int(n),
                    "method": method,
                    "robust_method": robust_method,
                }
            )

    if not records:
        return pd.DataFrame(), {**meta, "status": "no_valid_tests"}

    out = pd.DataFrame(records)

    # Within-target FDR.
    out["q"] = np.nan
    for tgt in out["target"].unique():
        mask = out["target"] == tgt
        pvals = pd.to_numeric(out.loc[mask, "p"], errors="coerce").to_numpy()
        out.loc[mask, "q"] = fdr_bh(pvals, alpha=float(_get(config, "behavior_analysis.statistics.fdr_alpha", 0.05)), config=config)

    # Convenience aliases for integration with global FDR tooling (optional).
    out["p_primary"] = out["p"]
    out["p_raw"] = out["p"]
    out["p_kind_primary"] = "p"
    out["p_primary_source"] = "raw"

    meta["status"] = "ok"
    meta["n_tests"] = int(len(out))
    return out, meta


def select_significant_qc_covariates(
    audit_df: pd.DataFrame,
    *,
    config: Any,
    alpha: float = 0.05,
    max_covariates: int = 3,
    prefer_target: str = "rating",
) -> List[str]:
    """Pick QC metrics to add as covariates, based on FDR q-values."""
    if audit_df is None or audit_df.empty:
        return []
    df = audit_df.copy()
    df["q"] = pd.to_numeric(df.get("q", np.nan), errors="coerce")
    df = df[np.isfinite(df["q"]) & (df["q"] < float(alpha))]
    if df.empty:
        return []
    # Prefer metrics confounded with rating (or specified target), then strongest |r|.
    df["abs_r"] = pd.to_numeric(df.get("r", np.nan), errors="coerce").abs()
    df["is_prefer"] = (df["target"].astype(str) == str(prefer_target))
    df = df.sort_values(["is_prefer", "abs_r"], ascending=[False, False])
    picked = df["qc_metric"].astype(str).tolist()
    return list(dict.fromkeys(picked))[: int(max_covariates)]


__all__ = [
    "audit_qc_confounds",
    "select_qc_columns",
    "select_significant_qc_covariates",
]

