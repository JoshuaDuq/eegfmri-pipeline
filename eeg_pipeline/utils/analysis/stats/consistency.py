"""
Effect Direction Consistency (Subject-Level)
===========================================

Builds a per-feature summary that compares effect direction across:
- correlations (feature vs rating/temperature)
- trialwise regression (beta_feature)
- model families (beta_feature by family)

This is non-gating and intended to surface sign flips and contradictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _sign(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    if not np.isfinite(v) or v == 0:
        return np.nan if not np.isfinite(v) else 0.0
    return float(np.sign(v))


def _pick_corr_r(df: pd.DataFrame) -> pd.Series:
    for col in ["r_primary", "r", "rho", "correlation"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pick_corr_p(df: pd.DataFrame) -> pd.Series:
    for col in ["p_primary", "p_raw", "p_value", "p"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pick_beta(df: pd.DataFrame) -> pd.Series:
    for col in ["beta_feature", "beta", "coef", "coefficient"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_effect_direction_consistency_summary(
    *,
    corr_df: Optional[pd.DataFrame],
    regression_df: Optional[pd.DataFrame],
    models_df: Optional[pd.DataFrame],
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge per-feature evidence across analyses and flag sign flips.

    Output is one row per feature with columns like:
      corr_rating_r, reg_rating_beta, model_ols_rating_beta, ...
      flip_* booleans for contradictions.
    """
    meta: Dict[str, Any] = {"status": "init"}

    features: List[str] = []
    for df in [corr_df, regression_df, models_df]:
        if df is None or df.empty or "feature" not in df.columns:
            continue
        features.extend([str(x) for x in df["feature"].dropna().unique().tolist()])
    features = sorted(set(features))
    if not features:
        return pd.DataFrame(), {"status": "empty"}

    out = pd.DataFrame({"feature": features})

    # Correlations: rating and temperature (when present)
    if corr_df is not None and not corr_df.empty and "target" in corr_df.columns:
        cdf = corr_df.copy()
        cdf["target"] = cdf["target"].astype(str)
        cdf["r_val"] = _pick_corr_r(cdf)
        cdf["p_val"] = _pick_corr_p(cdf)

        def _agg(target: str, prefix: str) -> None:
            sub = cdf[cdf["target"] == target]
            if sub.empty:
                return
            # Choose row with min p (primary) per feature.
            sub = sub.sort_values(["feature", "p_val"], ascending=[True, True])
            best = sub.groupby("feature", sort=False).head(1).set_index("feature")
            out[f"{prefix}_r"] = out["feature"].map(best["r_val"]).astype(float)
            out[f"{prefix}_p"] = out["feature"].map(best["p_val"]).astype(float)
            out[f"{prefix}_sign"] = out[f"{prefix}_r"].map(_sign)

        _agg("rating", "corr_rating")
        _agg("temperature", "corr_temperature")

    # Regression: beta_feature for given outcome/target.
    if regression_df is not None and not regression_df.empty and "target" in regression_df.columns:
        rdf = regression_df.copy()
        rdf["target"] = rdf["target"].astype(str)
        rdf["beta"] = _pick_beta(rdf)
        for target, prefix in [("rating", "reg_rating"), ("pain_residual", "reg_pain_residual"), ("temperature", "reg_temperature")]:
            sub = rdf[rdf["target"] == target]
            if sub.empty:
                continue
            sub = sub.sort_values(["feature", "p_primary"], ascending=[True, True]) if "p_primary" in sub.columns else sub
            best = sub.groupby("feature", sort=False).head(1).set_index("feature")
            out[f"{prefix}_beta"] = out["feature"].map(best["beta"]).astype(float)
            out[f"{prefix}_p"] = out["feature"].map(pd.to_numeric(best.get("p_primary", best.get("p_feature", np.nan)), errors="coerce")).astype(float)
            out[f"{prefix}_sign"] = out[f"{prefix}_beta"].map(_sign)

    # Models: beta_feature by family (default use ols_hc3 for sign comparisons, but keep robust too if present)
    if models_df is not None and not models_df.empty:
        mdf = models_df.copy()
        for col in ["target", "model_family"]:
            if col in mdf.columns:
                mdf[col] = mdf[col].astype(str)
        mdf["beta"] = _pick_beta(mdf)
        p_col = "p_primary" if "p_primary" in mdf.columns else ("p_feature" if "p_feature" in mdf.columns else None)
        if p_col is not None:
            mdf["p_val"] = pd.to_numeric(mdf[p_col], errors="coerce")
        else:
            mdf["p_val"] = np.nan

        for fam in ["ols_hc3", "robust_rlm", "quantile_50", "logit"]:
            if "model_family" not in mdf.columns:
                continue
            sub_f = mdf[mdf["model_family"] == fam]
            if sub_f.empty:
                continue
            for target, prefix in [("rating", f"model_{fam}_rating"), ("pain_residual", f"model_{fam}_pain_residual")]:
                if "target" not in sub_f.columns:
                    continue
                sub = sub_f[sub_f["target"] == target]
                if sub.empty:
                    continue
                sub = sub.sort_values(["feature", "p_val"], ascending=[True, True])
                best = sub.groupby("feature", sort=False).head(1).set_index("feature")
                out[f"{prefix}_beta"] = out["feature"].map(best["beta"]).astype(float)
                out[f"{prefix}_p"] = out["feature"].map(best["p_val"]).astype(float)
                out[f"{prefix}_sign"] = out[f"{prefix}_beta"].map(_sign)

    # Flag sign flips: rating vs pain_residual, and across methods.
    def _flip(a: pd.Series, b: pd.Series) -> pd.Series:
        a_s = a.map(_sign)
        b_s = b.map(_sign)
        ok = np.isfinite(a_s) & np.isfinite(b_s) & (a_s != 0) & (b_s != 0)
        return (a_s * b_s < 0).where(ok, np.nan)

    if "reg_rating_beta" in out.columns and "reg_pain_residual_beta" in out.columns:
        out["flip_reg_rating_vs_pain_residual"] = _flip(out["reg_rating_beta"], out["reg_pain_residual_beta"])
    if "corr_rating_r" in out.columns and "reg_rating_beta" in out.columns:
        out["flip_corr_vs_reg_rating"] = _flip(out["corr_rating_r"], out["reg_rating_beta"])
    if "corr_rating_r" in out.columns and "model_ols_hc3_rating_beta" in out.columns:
        out["flip_corr_vs_model_ols_rating"] = _flip(out["corr_rating_r"], out["model_ols_hc3_rating_beta"])
    if "reg_rating_beta" in out.columns and "model_ols_hc3_rating_beta" in out.columns:
        out["flip_reg_vs_model_ols_rating"] = _flip(out["reg_rating_beta"], out["model_ols_hc3_rating_beta"])

    meta["status"] = "ok"
    meta["n_features"] = int(len(out))
    meta["has_correlations"] = bool(corr_df is not None and not corr_df.empty)
    meta["has_regression"] = bool(regression_df is not None and not regression_df.empty)
    meta["has_models"] = bool(models_df is not None and not models_df.empty)
    return out, meta


__all__ = ["build_effect_direction_consistency_summary"]

