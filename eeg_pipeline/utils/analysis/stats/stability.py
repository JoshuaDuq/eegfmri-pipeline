"""
Block/Run Stability (Subject-Level)
===================================

Non-gating stability diagnostics for feature→outcome associations across repeated
blocks/runs within a subject (common in pain paradigms).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation


def _get(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        pass
    return default


def compute_groupwise_stability(
    trial_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    outcome: str,
    group_col: str,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute per-feature association stability across groups (e.g., runs/blocks).

    This is intentionally descriptive and non-gating: it reports variability and
    sign consistency but does not exclude features.
    """
    method = str(_get(config, "behavior_analysis.stability.method", "spearman")).strip().lower()
    min_group_trials = int(_get(config, "behavior_analysis.stability.min_group_trials", 8))
    max_features = int(_get(config, "behavior_analysis.stability.max_features", 50))
    alpha = float(_get(config, "behavior_analysis.stability.alpha", 0.05))
    use_partial_temp = bool(_get(config, "behavior_analysis.stability.partial_temperature", True))

    meta: Dict[str, Any] = {
        "method": method,
        "min_group_trials": min_group_trials,
        "max_features": max_features,
        "alpha": alpha,
        "partial_temperature": use_partial_temp,
        "outcome": outcome,
        "group_col": group_col,
    }

    if outcome not in trial_df.columns or group_col not in trial_df.columns:
        return pd.DataFrame(), {**meta, "status": "missing_columns"}

    y_all = pd.to_numeric(trial_df[outcome], errors="coerce")
    g_all = trial_df[group_col]

    candidates = []
    for col in feature_cols:
        if col not in trial_df.columns:
            continue
        x = pd.to_numeric(trial_df[col], errors="coerce")
        if int((x.notna() & y_all.notna()).sum()) < max(min_group_trials, 10):
            continue
        if float(np.nanstd(x.to_numpy(dtype=float), ddof=1)) <= 1e-12:
            continue
        r, p = compute_correlation(x.to_numpy(dtype=float), y_all.to_numpy(dtype=float), method=method)
        candidates.append((abs(float(r)) if np.isfinite(r) else 0.0, float(r) if np.isfinite(r) else np.nan, float(p) if np.isfinite(p) else np.nan, col))

    if not candidates:
        return pd.DataFrame(), {**meta, "status": "empty"}

    candidates.sort(reverse=True)
    selected = [c for *_rest, c in candidates[:max_features]]
    meta["n_features_considered"] = int(len(candidates))
    meta["n_features_selected"] = int(len(selected))

    # Optional partial correlations controlling temperature within each group.
    try:
        from eeg_pipeline.utils.analysis.stats import compute_partial_corr
        _has_partial = True
    except Exception:
        compute_partial_corr = None  # type: ignore[assignment]
        _has_partial = False
    meta["has_partial_corr"] = _has_partial

    has_temp = "temperature" in trial_df.columns

    records: List[Dict[str, Any]] = []
    groups = list(pd.Series(g_all).dropna().unique())
    meta["n_groups_total"] = int(len(groups))

    for feat in selected:
        x_all = pd.to_numeric(trial_df[feat], errors="coerce")
        valid_all = x_all.notna() & y_all.notna()
        r_overall, p_overall = compute_correlation(
            x_all[valid_all].to_numpy(dtype=float),
            y_all[valid_all].to_numpy(dtype=float),
            method=method,
        )

        rs = []
        ps = []
        ns = []
        rs_partial = []
        ps_partial = []

        for g in groups:
            mask = (g_all == g) & x_all.notna() & y_all.notna()
            n = int(mask.sum())
            if n < min_group_trials:
                continue
            r, p = compute_correlation(
                x_all[mask].to_numpy(dtype=float),
                y_all[mask].to_numpy(dtype=float),
                method=method,
            )
            rs.append(float(r) if np.isfinite(r) else np.nan)
            ps.append(float(p) if np.isfinite(p) else np.nan)
            ns.append(n)

            if use_partial_temp and _has_partial and has_temp:
                t = pd.to_numeric(trial_df.loc[mask, "temperature"], errors="coerce")
                ok = t.notna()
                if int(ok.sum()) >= max(min_group_trials, 10):
                    try:
                        r_p, p_p, _n_p = compute_partial_corr(
                            pd.Series(x_all[mask][ok].to_numpy(dtype=float)),
                            pd.Series(y_all[mask][ok].to_numpy(dtype=float)),
                            pd.DataFrame({"temperature": t[ok].to_numpy(dtype=float)}),
                            method=method,
                        )
                        rs_partial.append(float(r_p) if np.isfinite(r_p) else np.nan)
                        ps_partial.append(float(p_p) if np.isfinite(p_p) else np.nan)
                    except Exception:
                        rs_partial.append(np.nan)
                        ps_partial.append(np.nan)

        rs_arr = np.asarray(rs, dtype=float)
        ps_arr = np.asarray(ps, dtype=float)
        ok = np.isfinite(rs_arr)
        n_groups_valid = int(ok.sum())
        sign_consistency = float((np.sign(rs_arr[ok]) == np.sign(r_overall)).mean()) if ok.any() and np.isfinite(r_overall) else np.nan
        frac_sig = float((ps_arr[ok] < alpha).mean()) if ok.any() else np.nan

        rec: Dict[str, Any] = {
            "feature": feat,
            "target": outcome,
            "group_column": group_col,
            "method": method,
            "n_groups_total": int(len(groups)),
            "n_groups_valid": n_groups_valid,
            "n_trials_total": int(valid_all.sum()),
            "r_overall": float(r_overall) if np.isfinite(r_overall) else np.nan,
            "p_overall": float(p_overall) if np.isfinite(p_overall) else np.nan,
            "r_group_mean": float(np.nanmean(rs_arr)) if ok.any() else np.nan,
            "r_group_std": float(np.nanstd(rs_arr, ddof=1)) if ok.sum() > 1 else np.nan,
            "r_group_min": float(np.nanmin(rs_arr)) if ok.any() else np.nan,
            "r_group_max": float(np.nanmax(rs_arr)) if ok.any() else np.nan,
            "sign_consistency": sign_consistency,
            "frac_groups_p_lt_alpha": frac_sig,
            "mean_group_n": float(np.mean(ns)) if ns else np.nan,
        }

        if rs_partial:
            rp = np.asarray(rs_partial, dtype=float)
            okp = np.isfinite(rp)
            rec["r_partial_temp_group_mean"] = float(np.nanmean(rp)) if okp.any() else np.nan
            rec["r_partial_temp_group_std"] = float(np.nanstd(rp, ddof=1)) if okp.sum() > 1 else np.nan
            pp = np.asarray(ps_partial, dtype=float)
            rec["frac_groups_partial_p_lt_alpha"] = float((pp[okp] < alpha).mean()) if okp.any() else np.nan

        records.append(rec)

    out = pd.DataFrame(records)
    return out, {**meta, "status": "ok"}


__all__ = ["compute_groupwise_stability"]

